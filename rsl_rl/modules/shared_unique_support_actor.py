# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-auditable selected-only support actor for the H1 mechanism.

The adapter scores all valid deployment-observable tokens, selects a shared
unique budget, gathers only those selected raw values, and only then creates
terrain embeddings.  Consequently no full-token terrain embedding exists
behind the bottleneck.  A selected token is charged once but may be consumed
by every calibrated support role for which it is eligible.

This is deliberately a component contract, not a registered simulator task.
The caller remains responsible for constructing role masks from calibrated
deployment geometry, proprioception, and gait phase.  A mandatory provenance
receipt rejects declarations involving simulator contact or terrain truth.
The causal interventions below are component-level probes only: they do not
claim matched training fairness and do not claim that real geometry is wired.
"""

from __future__ import annotations

import hashlib
import math
import torch
import torch.nn as nn
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Final

from .sparse_support_evidence_bottleneck import NUM_QUERIES, QUERY_NAMES
from .support_selection_ablation import FixedBudgetSupportSelector

DEPLOYMENT_GEOMETRY_SOURCES: Final[tuple[str, ...]] = (
    "calibrated_lidar_ray_geometry",
    "calibrated_depth_ray_geometry",
    "calibrated_multisensor_ray_geometry",
)
SUPPORT_CAUSAL_INTERVENTION_MODES: Final[tuple[str, ...]] = (
    "native",
    "zero_selected",
    "matched_substitution",
    "role_shuffle",
    "cell_shuffle",
)
MATCHED_SUBSTITUTION_SCHEMA: Final[str] = (
    "support_matched_substitution_strata_v1"
)


def _tensor_sha256(schema: str, *tensors: torch.Tensor) -> str:
    digest = hashlib.sha256(schema.encode("utf-8"))
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MatchedSubstitutionMetadata:
    """Pre-registered exact matching strata for one causal evaluation batch.

    Candidate priority is an explicit deterministic ordering, not an implicit
    global random draw.  Exact matching additionally requires the complete
    four-role eligibility signature and range/angle/age stratum IDs.
    """

    candidate_mask: torch.Tensor
    role_eligibility: torch.Tensor
    range_stratum: torch.Tensor
    angle_stratum: torch.Tensor
    age_stratum: torch.Tensor
    candidate_priority: torch.Tensor
    registration_sha256: str
    schema: str = MATCHED_SUBSTITUTION_SCHEMA

    def __post_init__(self) -> None:
        """Reject an altered matching schema or malformed registration hash."""
        if self.schema != MATCHED_SUBSTITUTION_SCHEMA:
            raise ValueError(
                f"Unsupported matched-substitution schema {self.schema!r}."
            )
        if (
            not isinstance(self.registration_sha256, str)
            or len(self.registration_sha256) != 64
        ):
            raise ValueError(
                "registration_sha256 must be a 64-character SHA-256 hex string."
            )

    @classmethod
    def register(
        cls,
        *,
        candidate_mask: torch.Tensor,
        role_eligibility: torch.Tensor,
        range_stratum: torch.Tensor,
        angle_stratum: torch.Tensor,
        age_stratum: torch.Tensor,
        candidate_priority: torch.Tensor,
    ) -> MatchedSubstitutionMetadata:
        """Clone tensors and freeze their exact matching contract by SHA-256."""
        values = tuple(
            tensor.detach().clone()
            for tensor in (
                candidate_mask,
                role_eligibility,
                range_stratum,
                angle_stratum,
                age_stratum,
                candidate_priority,
            )
        )
        digest = _tensor_sha256(MATCHED_SUBSTITUTION_SCHEMA, *values)
        return cls(*values, registration_sha256=digest)

    def computed_sha256(self) -> str:
        """Recompute the receipt hash so post-registration mutation fails."""
        return _tensor_sha256(
            self.schema,
            self.candidate_mask,
            self.role_eligibility,
            self.range_stratum,
            self.angle_stratum,
            self.age_stratum,
            self.candidate_priority,
        )

    def receipt(self) -> dict[str, object]:
        """Return claims that are valid before simulator/task integration."""
        return {
            "schema": self.schema,
            "registration_sha256": self.registration_sha256,
            "exact_match_fields": (
                "role_eligibility",
                "range_stratum",
                "angle_stratum",
                "age_stratum",
            ),
            "candidate_order": "explicit_candidate_priority",
            "global_random_fallback": False,
            "fair_performance_claim": False,
            "real_geometry_connected": False,
        }


class MatchedSubstitutionShortfallError(RuntimeError):
    """Fail-closed error carrying frozen-selection shortfall diagnostics."""

    def __init__(self, audit: dict[str, object]) -> None:
        """Preserve frozen-selection evidence on an exact-match shortfall."""
        self.audit = audit
        shortfall = audit["matched_substitution_shortfall"]
        super().__init__(
            "Exact role/range/angle/age matched substitution has candidate "
            f"shortfall: {shortfall}."
        )


@dataclass(frozen=True)
class SupportMaskProvenance:
    """Fail-closed caller declaration for support-role mask construction."""

    geometry_source: str
    uses_proprioception: bool
    uses_gait_phase: bool
    uses_simulator_contact_truth: bool = False
    uses_simulator_terrain_truth: bool = False
    schema: str = "deployment_observable_support_mask_v1"

    def __post_init__(self) -> None:
        """Reject privileged or unsupported mask provenance declarations."""
        if self.schema != "deployment_observable_support_mask_v1":
            raise ValueError(
                "Unsupported support-mask provenance schema "
                f"{self.schema!r}."
            )
        if self.geometry_source not in DEPLOYMENT_GEOMETRY_SOURCES:
            raise ValueError(
                "geometry_source must be calibrated deployment-observable "
                f"geometry, got {self.geometry_source!r}."
            )
        if not isinstance(self.uses_proprioception, bool) or not isinstance(
            self.uses_gait_phase, bool
        ):
            raise ValueError(
                "uses_proprioception and uses_gait_phase must be boolean."
            )
        if not isinstance(self.uses_simulator_contact_truth, bool) or not isinstance(
            self.uses_simulator_terrain_truth, bool
        ):
            raise ValueError("Simulator-truth declarations must be boolean.")
        if self.uses_simulator_contact_truth:
            raise ValueError(
                "Support masks must not use simulator contact truth."
            )
        if self.uses_simulator_terrain_truth:
            raise ValueError(
                "Support masks must not use simulator terrain truth."
            )

    def receipt(self) -> dict[str, object]:
        """Return a serialization-friendly provenance audit receipt."""
        receipt = asdict(self)
        receipt["deployment_observable_only"] = True
        optional_inputs = []
        if self.uses_proprioception:
            optional_inputs.append("proprioception")
        if self.uses_gait_phase:
            optional_inputs.append("gait_phase")
        receipt["allowed_role_inputs"] = (
            self.geometry_source,
            *optional_inputs,
        )
        receipt["caller_attested"] = True
        return receipt


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {normalized}.")
    return normalized


class SharedUniqueSupportActorAdapter(nn.Module):
    """Map selected-only support evidence to deterministic action and value.

    The selector owns one unique token per budget slot.  Ownership is bounded
    by ``M / 4`` for every role and therefore cannot cross-role backfill.
    Consumption is separate: once selected, an overlapping token is visible to
    all of its eligible roles without spending another unique slot.
    """

    query_names: Final[tuple[str, ...]] = QUERY_NAMES

    def __init__(
        self,
        score_feature_dim: int,
        terrain_value_dim: int,
        proprio_dim: int,
        action_dim: int,
        *,
        total_budget: int = 16,
        score_dim: int = 32,
        value_embedding_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        """Construct one selected-only actor/value adapter."""
        super().__init__()
        self.score_feature_dim = _positive_integer(
            "score_feature_dim", score_feature_dim
        )
        self.terrain_value_dim = _positive_integer(
            "terrain_value_dim", terrain_value_dim
        )
        self.proprio_dim = _positive_integer("proprio_dim", proprio_dim)
        self.action_dim = _positive_integer("action_dim", action_dim)
        self.score_dim = _positive_integer("score_dim", score_dim)
        self.value_embedding_dim = _positive_integer(
            "value_embedding_dim", value_embedding_dim
        )
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)

        self.selector = FixedBudgetSupportSelector(
            strategy="role_quota_shared_unique_m",
            total_budget=total_budget,
        )
        self.total_budget = int(self.selector.total_budget)
        self.score_key_projection = nn.Linear(
            self.score_feature_dim,
            self.score_dim,
            bias=False,
        )
        self.score_query_projection = nn.Linear(
            self.proprio_dim,
            NUM_QUERIES * self.score_dim,
        )
        self.query_embedding = nn.Parameter(
            torch.zeros(1, NUM_QUERIES, self.score_dim)
        )
        self.score_scale = 1.0 / math.sqrt(self.score_dim)
        self.selected_value_projection = nn.Linear(
            self.terrain_value_dim,
            self.value_embedding_dim,
            bias=False,
        )

        fused_dim = self.proprio_dim + NUM_QUERIES * self.value_embedding_dim
        self.actor_backbone = nn.Sequential(
            nn.Linear(fused_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.critic_backbone = nn.Sequential(
            nn.Linear(fused_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
        *,
        mask_provenance: SupportMaskProvenance,
        intervention_mode: str = "native",
        matched_substitution: MatchedSubstitutionMetadata | None = None,
        role_permutation: torch.Tensor | None = None,
        cell_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic action and value from selected evidence."""
        action, value, _ = self.forward_with_diagnostics(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            role_eligibility,
            mask_provenance=mask_provenance,
            intervention_mode=intervention_mode,
            matched_substitution=matched_substitution,
            role_permutation=role_permutation,
            cell_permutation=cell_permutation,
        )
        return action, value

    def forward_with_diagnostics(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
        *,
        mask_provenance: SupportMaskProvenance,
        intervention_mode: str = "native",
        matched_substitution: MatchedSubstitutionMetadata | None = None,
        role_permutation: torch.Tensor | None = None,
        cell_permutation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Return action/value plus masks and shortfall audit tensors."""
        self._validate_provenance(mask_provenance)
        self._validate_inputs(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            role_eligibility,
        )
        compute_dtype = self.query_embedding.dtype
        score_features = score_features.to(dtype=compute_dtype)
        terrain_values = terrain_values.to(dtype=compute_dtype)
        proprio = proprio.to(dtype=compute_dtype)
        for name, tensor in (
            ("score_features", score_features),
            ("terrain_values", terrain_values),
            ("proprio", proprio),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"{name} is not representable as finite {compute_dtype}."
                )

        keys = self.score_key_projection(score_features)
        queries = self.score_query_projection(proprio).reshape(
            proprio.shape[0], NUM_QUERIES, self.score_dim
        )
        scores = torch.einsum(
            "bqd,bnd->bqn",
            queries + self.query_embedding,
            keys,
        ) * self.score_scale
        _, selector_diagnostics = self.selector(
            scores,
            token_valid,
            role_eligibility,
        )

        indices = selector_diagnostics["selection_indices"]
        slot_valid = selector_diagnostics["selection_slot_valid"]
        clean_unique_mask = selector_diagnostics["selection_unique_mask"]
        clean_membership_sha256 = tuple(
            _tensor_sha256(
                "shared_unique_clean_membership_v1",
                indices[batch_index],
                slot_valid[batch_index],
                clean_unique_mask[batch_index],
            )
            for batch_index in range(score_features.shape[0])
        )
        safe_indices = torch.where(
            slot_valid,
            indices,
            torch.zeros_like(indices),
        )
        selected_values = torch.gather(
            terrain_values,
            1,
            safe_indices[:, :, None].expand(
                -1, -1, self.terrain_value_dim
            ),
        )
        selected_values = torch.where(
            slot_valid[:, :, None],
            selected_values,
            torch.zeros_like(selected_values),
        )
        selected_role_eligibility = torch.gather(
            role_eligibility,
            2,
            safe_indices[:, None, :].expand(-1, NUM_QUERIES, -1),
        )
        clean_consumption_mask = (
            selected_role_eligibility & slot_valid[:, None, :]
        )
        (
            effective_selected_values,
            consumption_mask,
            intervention_diagnostics,
        ) = self._apply_intervention(
            intervention_mode=intervention_mode,
            selected_values=selected_values,
            clean_consumption_mask=clean_consumption_mask,
            clean_indices=indices,
            clean_slot_valid=slot_valid,
            clean_unique_mask=clean_unique_mask,
            clean_membership_sha256=clean_membership_sha256,
            terrain_values=terrain_values,
            token_valid=token_valid,
            role_eligibility=role_eligibility,
            matched_substitution=matched_substitution,
            role_permutation=role_permutation,
            cell_permutation=cell_permutation,
        )
        # The value projection sees [B,M,V], never [B,N,V].
        selected_embeddings = self.selected_value_projection(
            effective_selected_values
        )

        selected_scores = torch.gather(
            scores,
            2,
            safe_indices[:, None, :].expand(-1, NUM_QUERIES, -1),
        )
        consumption_count = consumption_mask.sum(dim=-1, keepdim=True)
        weights = (
            consumption_mask.to(scores.dtype) * torch.sigmoid(selected_scores)
        ) / consumption_count.clamp_min(1).to(scores.dtype)
        role_values = torch.einsum(
            "bqm,bmd->bqd", weights, selected_embeddings
        )
        fused = torch.cat((proprio, role_values.flatten(start_dim=1)), dim=-1)
        action = self.action_head(self.actor_backbone(fused))
        value = self.value_head(self.critic_backbone(fused))

        batch_size = score_features.shape[0]
        diagnostics = dict(selector_diagnostics)
        diagnostics.update(
            {
                "clean_selection_indices": indices.clone(),
                "clean_selection_slot_valid": slot_valid.clone(),
                "clean_selection_unique_mask": clean_unique_mask.clone(),
                "effective_selection_indices": indices.clone(),
                "effective_selection_unique_mask": clean_unique_mask.clone(),
                "clean_membership_sha256": clean_membership_sha256,
                "clean_selection_frozen": torch.ones(
                    batch_size,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
                "no_reselection": torch.ones(
                    batch_size,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
                "selection_recomputed_count": torch.zeros(
                    batch_size,
                    device=score_features.device,
                    dtype=torch.long,
                ),
                "clean_role_consumption_mask": clean_consumption_mask,
                "role_consumption_mask": consumption_mask,
                "role_consumption_count": consumption_count.squeeze(-1),
                "selected_aggregation_weights": weights,
                "projected_selected_token_count": slot_valid.sum(dim=-1),
                "post_bottleneck_token_width": torch.full(
                    (batch_size,),
                    self.total_budget,
                    device=score_features.device,
                    dtype=torch.long,
                ),
                "mask_provenance_verified": torch.ones(
                    batch_size,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
                "mask_geometry_source_code": torch.full(
                    (batch_size,),
                    DEPLOYMENT_GEOMETRY_SOURCES.index(
                        mask_provenance.geometry_source
                    ),
                    device=score_features.device,
                    dtype=torch.long,
                ),
                "mask_uses_proprioception": torch.full(
                    (batch_size,),
                    mask_provenance.uses_proprioception,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
                "mask_uses_gait_phase": torch.full(
                    (batch_size,),
                    mask_provenance.uses_gait_phase,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
                "mask_uses_simulator_contact_truth": torch.zeros(
                    batch_size,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
                "mask_uses_simulator_terrain_truth": torch.zeros(
                    batch_size,
                    device=score_features.device,
                    dtype=torch.bool,
                ),
            }
        )
        diagnostics.update(intervention_diagnostics)
        return action, value, diagnostics

    def forward_native_training(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
        *,
        mask_provenance: SupportMaskProvenance,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run native selected-only training without tensor-to-host audit work.

        The returned boolean ``finite_gate`` has shape ``[B]``.  A production
        runner must reject or quarantine every row for which it is false.  The
        method deliberately omits membership SHA, causal interventions, and
        Python exceptions based on tensor values.  GPU latency remains
        unmeasured and no performance claim follows from this API.
        """
        self._validate_provenance(mask_provenance)
        self._validate_input_structure(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            role_eligibility,
        )
        compute_dtype = self.query_embedding.dtype
        score_features = score_features.to(dtype=compute_dtype)
        terrain_values = terrain_values.to(dtype=compute_dtype)
        proprio = proprio.to(dtype=compute_dtype)
        input_finite = (
            torch.isfinite(score_features).all(dim=(-1, -2))
            & torch.isfinite(terrain_values).all(dim=(-1, -2))
            & torch.isfinite(proprio).all(dim=-1)
        )
        role_contract_valid = ~(
            role_eligibility & ~token_valid[:, None, :]
        ).any(dim=(-1, -2))

        keys = self.score_key_projection(score_features)
        queries = self.score_query_projection(proprio).reshape(
            proprio.shape[0], NUM_QUERIES, self.score_dim
        )
        scores = torch.einsum(
            "bqd,bnd->bqn",
            queries + self.query_embedding,
            keys,
        ) * self.score_scale
        _, selector_diagnostics = self.selector(
            scores,
            token_valid,
            role_eligibility,
        )
        indices = selector_diagnostics["selection_indices"]
        slot_valid = selector_diagnostics["selection_slot_valid"]
        safe_indices = torch.where(
            slot_valid,
            indices,
            torch.zeros_like(indices),
        )
        selected_values = torch.gather(
            terrain_values,
            1,
            safe_indices[:, :, None].expand(
                -1, -1, self.terrain_value_dim
            ),
        )
        selected_values = torch.where(
            slot_valid[:, :, None],
            selected_values,
            torch.zeros_like(selected_values),
        )
        selected_role_eligibility = torch.gather(
            role_eligibility,
            2,
            safe_indices[:, None, :].expand(-1, NUM_QUERIES, -1),
        )
        consumption_mask = (
            selected_role_eligibility & slot_valid[:, None, :]
        )
        selected_embeddings = self.selected_value_projection(selected_values)
        selected_scores = torch.gather(
            scores,
            2,
            safe_indices[:, None, :].expand(-1, NUM_QUERIES, -1),
        )
        consumption_count = consumption_mask.sum(dim=-1, keepdim=True)
        weights = (
            consumption_mask.to(scores.dtype) * torch.sigmoid(selected_scores)
        ) / consumption_count.clamp_min(1).to(scores.dtype)
        role_values = torch.einsum(
            "bqm,bmd->bqd", weights, selected_embeddings
        )
        fused = torch.cat((proprio, role_values.flatten(start_dim=1)), dim=-1)
        action = self.action_head(self.actor_backbone(fused))
        value = self.value_head(self.critic_backbone(fused))
        finite_gate = (
            input_finite
            & role_contract_valid
            & selector_diagnostics["valid_scores_finite"]
            & torch.isfinite(action).all(dim=-1)
            & torch.isfinite(value).all(dim=-1)
        )
        return action, value, finite_gate

    def _apply_intervention(
        self,
        *,
        intervention_mode: str,
        selected_values: torch.Tensor,
        clean_consumption_mask: torch.Tensor,
        clean_indices: torch.Tensor,
        clean_slot_valid: torch.Tensor,
        clean_unique_mask: torch.Tensor,
        clean_membership_sha256: tuple[str, ...],
        terrain_values: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
        matched_substitution: MatchedSubstitutionMetadata | None,
        role_permutation: torch.Tensor | None,
        cell_permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        mode = self._normalize_intervention_mode(intervention_mode)
        self._validate_intervention_arguments(
            mode,
            matched_substitution,
            role_permutation,
            cell_permutation,
        )
        batch_size, budget, _ = selected_values.shape
        device = selected_values.device
        identity_roles = torch.arange(
            NUM_QUERIES, device=device, dtype=torch.long
        )[None, :].expand(batch_size, -1)
        identity_cells = torch.arange(
            budget, device=device, dtype=torch.long
        )[None, :].expand(batch_size, -1)
        target_count = clean_slot_valid.sum(dim=-1)
        matched_indices = torch.full(
            (batch_size, budget), -1, device=device, dtype=torch.long
        )
        matched_slot_valid = torch.zeros_like(clean_slot_valid)
        matched_shortfall = torch.zeros(
            batch_size, device=device, dtype=torch.long
        )
        effective_values = selected_values
        effective_consumption = clean_consumption_mask
        resolved_role_permutation = identity_roles
        resolved_cell_permutation = identity_cells
        matching_sha256: str | None = None

        if mode == "zero_selected":
            effective_values = torch.zeros_like(selected_values)
        elif mode == "matched_substitution":
            (
                effective_values,
                matched_indices,
                matched_slot_valid,
                matched_shortfall,
            ) = self._exact_matched_substitution(
                metadata=matched_substitution,
                terrain_values=terrain_values,
                token_valid=token_valid,
                role_eligibility=role_eligibility,
                clean_indices=clean_indices,
                clean_slot_valid=clean_slot_valid,
                clean_unique_mask=clean_unique_mask,
                clean_membership_sha256=clean_membership_sha256,
            )
            matching_sha256 = matched_substitution.registration_sha256
        elif mode == "role_shuffle":
            resolved_role_permutation = self._validate_role_permutation(
                role_permutation,
                batch_size=batch_size,
                device=device,
            )
            effective_consumption = torch.gather(
                clean_consumption_mask,
                1,
                resolved_role_permutation[:, :, None].expand(
                    -1, -1, budget
                ),
            )
        elif mode == "cell_shuffle":
            resolved_cell_permutation = self._validate_cell_permutation(
                cell_permutation,
                clean_slot_valid=clean_slot_valid,
                device=device,
            )
            effective_values = torch.gather(
                selected_values,
                1,
                resolved_cell_permutation[:, :, None].expand(
                    -1, -1, self.terrain_value_dim
                ),
            )

        if mode == "native":
            applicable = torch.ones(
                batch_size, device=device, dtype=torch.bool
            )
        elif mode == "matched_substitution":
            applicable = (target_count > 0) & (matched_shortfall == 0)
        else:
            applicable = target_count > 0
        diagnostics: dict[str, object] = {
            "intervention_mode": mode,
            "intervention_mode_code": torch.full(
                (batch_size,),
                SUPPORT_CAUSAL_INTERVENTION_MODES.index(mode),
                device=device,
                dtype=torch.long,
            ),
            "intervention_applicable": applicable,
            "matched_substitution_indices": matched_indices,
            "matched_substitution_slot_valid": matched_slot_valid,
            "matched_substitution_target_count": (
                target_count
                if mode == "matched_substitution"
                else torch.zeros_like(target_count)
            ),
            "matched_substitution_realized_count": matched_slot_valid.sum(
                dim=-1
            ),
            "matched_substitution_shortfall": matched_shortfall,
            "matched_substitution_registration_sha256": matching_sha256,
            "matched_substitution_global_random_fallback": torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
            "role_shuffle_permutation": resolved_role_permutation,
            "role_shuffle_permutation_sha256": (
                _tensor_sha256(
                    "support_role_shuffle_permutation_v1",
                    resolved_role_permutation,
                )
                if mode == "role_shuffle"
                else None
            ),
            "cell_shuffle_permutation": resolved_cell_permutation,
            "cell_shuffle_permutation_sha256": (
                _tensor_sha256(
                    "support_cell_shuffle_permutation_v1",
                    resolved_cell_permutation,
                )
                if mode == "cell_shuffle"
                else None
            ),
            "fair_performance_claim": torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
            "real_geometry_connected": torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
        }
        return effective_values, effective_consumption, diagnostics

    def _exact_matched_substitution(
        self,
        *,
        metadata: MatchedSubstitutionMetadata,
        terrain_values: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
        clean_indices: torch.Tensor,
        clean_slot_valid: torch.Tensor,
        clean_unique_mask: torch.Tensor,
        clean_membership_sha256: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_matched_metadata(
            metadata,
            token_valid=token_valid,
            role_eligibility=role_eligibility,
        )
        batch_size, budget = clean_indices.shape
        substitution_indices = torch.full_like(clean_indices, -1)
        substitution_slot_valid = torch.zeros_like(clean_slot_valid)

        for batch_index in range(batch_size):
            unused_candidates = (
                metadata.candidate_mask[batch_index]
                & token_valid[batch_index]
                & ~clean_unique_mask[batch_index]
            )
            candidate_indices = torch.where(unused_candidates)[0]
            ordered_candidates = sorted(
                (int(index) for index in candidate_indices),
                key=lambda index: (
                    int(metadata.candidate_priority[batch_index, index]),
                    index,
                ),
            )
            used: set[int] = set()
            for slot in range(budget):
                if not bool(clean_slot_valid[batch_index, slot]):
                    continue
                selected = int(clean_indices[batch_index, slot])
                for candidate in ordered_candidates:
                    if candidate in used:
                        continue
                    if not torch.equal(
                        metadata.role_eligibility[batch_index, :, candidate],
                        role_eligibility[batch_index, :, selected],
                    ):
                        continue
                    if any(
                        int(stratum[batch_index, candidate])
                        != int(stratum[batch_index, selected])
                        for stratum in (
                            metadata.range_stratum,
                            metadata.angle_stratum,
                            metadata.age_stratum,
                        )
                    ):
                        continue
                    substitution_indices[batch_index, slot] = candidate
                    substitution_slot_valid[batch_index, slot] = True
                    used.add(candidate)
                    break

        target_count = clean_slot_valid.sum(dim=-1)
        realized_count = substitution_slot_valid.sum(dim=-1)
        shortfall = target_count - realized_count
        if bool((shortfall > 0).any()):
            raise MatchedSubstitutionShortfallError(
                {
                    "clean_selection_indices": clean_indices.detach().clone(),
                    "clean_selection_slot_valid": (
                        clean_slot_valid.detach().clone()
                    ),
                    "clean_selection_unique_mask": (
                        clean_unique_mask.detach().clone()
                    ),
                    "clean_membership_sha256": clean_membership_sha256,
                    "clean_selection_frozen": torch.ones_like(
                        target_count, dtype=torch.bool
                    ),
                    "no_reselection": torch.ones_like(
                        target_count, dtype=torch.bool
                    ),
                    "matched_substitution_indices": substitution_indices,
                    "matched_substitution_slot_valid": (
                        substitution_slot_valid
                    ),
                    "matched_substitution_target_count": target_count,
                    "matched_substitution_realized_count": realized_count,
                    "matched_substitution_shortfall": shortfall,
                    "matched_substitution_registration_sha256": (
                        metadata.registration_sha256
                    ),
                    "matched_substitution_global_random_fallback": False,
                }
            )

        safe_indices = torch.where(
            substitution_slot_valid,
            substitution_indices,
            torch.zeros_like(substitution_indices),
        )
        substituted_values = torch.gather(
            terrain_values,
            1,
            safe_indices[:, :, None].expand(
                -1, -1, self.terrain_value_dim
            ),
        )
        substituted_values = torch.where(
            substitution_slot_valid[:, :, None],
            substituted_values,
            torch.zeros_like(substituted_values),
        )
        return (
            substituted_values,
            substitution_indices,
            substitution_slot_valid,
            shortfall,
        )

    @staticmethod
    def _normalize_intervention_mode(mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError("intervention_mode must be a string.")
        normalized = mode.lower().replace("-", "_")
        if normalized not in SUPPORT_CAUSAL_INTERVENTION_MODES:
            raise ValueError(
                "intervention_mode must be one of "
                f"{SUPPORT_CAUSAL_INTERVENTION_MODES}, got {mode!r}."
            )
        return normalized

    @staticmethod
    def _validate_intervention_arguments(
        mode: str,
        matched_substitution: MatchedSubstitutionMetadata | None,
        role_permutation: torch.Tensor | None,
        cell_permutation: torch.Tensor | None,
    ) -> None:
        required = {
            "matched_substitution": matched_substitution,
            "role_shuffle": role_permutation,
            "cell_shuffle": cell_permutation,
        }
        for argument_mode, value in required.items():
            if mode == argument_mode and value is None:
                raise ValueError(
                    f"{mode} requires explicit pre-registered metadata."
                )
            if mode != argument_mode and value is not None:
                raise ValueError(
                    f"{argument_mode} metadata is only valid in "
                    f"intervention_mode={argument_mode!r}."
                )

    @staticmethod
    def _validate_role_permutation(
        permutation: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(permutation, torch.Tensor):
            raise ValueError("role_permutation must be a tensor.")
        if permutation.device != device:
            raise ValueError("role_permutation device must match actor inputs.")
        if permutation.dtype != torch.long:
            raise ValueError("role_permutation must use torch.long.")
        if tuple(permutation.shape) != (batch_size, NUM_QUERIES):
            raise ValueError("role_permutation must have shape [B,4].")
        expected = torch.arange(
            NUM_QUERIES, device=device, dtype=torch.long
        )[None, :].expand(batch_size, -1)
        if not torch.equal(torch.sort(permutation, dim=-1).values, expected):
            raise ValueError("Every role_permutation row must permute 0..3.")
        return permutation

    @staticmethod
    def _validate_cell_permutation(
        permutation: torch.Tensor,
        *,
        clean_slot_valid: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(permutation, torch.Tensor):
            raise ValueError("cell_permutation must be a tensor.")
        if permutation.device != device:
            raise ValueError("cell_permutation device must match actor inputs.")
        if permutation.dtype != torch.long:
            raise ValueError("cell_permutation must use torch.long.")
        if tuple(permutation.shape) != tuple(clean_slot_valid.shape):
            raise ValueError("cell_permutation must have shape [B,M].")
        batch_size, budget = clean_slot_valid.shape
        expected = torch.arange(
            budget, device=device, dtype=torch.long
        )[None, :].expand(batch_size, -1)
        if not torch.equal(torch.sort(permutation, dim=-1).values, expected):
            raise ValueError("Every cell_permutation row must permute 0..M-1.")
        if not torch.equal(
            torch.gather(clean_slot_valid, 1, permutation),
            clean_slot_valid,
        ):
            raise ValueError(
                "cell_permutation may only permute selected valid slots."
            )
        invalid = ~clean_slot_valid
        if not torch.equal(permutation[invalid], expected[invalid]):
            raise ValueError("Padded cell slots must remain fixed.")
        return permutation

    @staticmethod
    def _validate_matched_metadata(
        metadata: MatchedSubstitutionMetadata,
        *,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
    ) -> None:
        if not isinstance(metadata, MatchedSubstitutionMetadata):
            raise ValueError(
                "matched_substitution must be registered metadata."
            )
        if metadata.computed_sha256() != metadata.registration_sha256:
            raise ValueError(
                "Matched-substitution metadata changed after registration."
            )
        batch_size, num_tokens = token_valid.shape
        token_shape = (batch_size, num_tokens)
        if tuple(metadata.candidate_mask.shape) != token_shape:
            raise ValueError("candidate_mask must have shape [B,N].")
        if tuple(metadata.role_eligibility.shape) != tuple(role_eligibility.shape):
            raise ValueError(
                "Registered role_eligibility must have shape [B,4,N]."
            )
        for name, tensor in (
            ("range_stratum", metadata.range_stratum),
            ("angle_stratum", metadata.angle_stratum),
            ("age_stratum", metadata.age_stratum),
            ("candidate_priority", metadata.candidate_priority),
        ):
            if tuple(tensor.shape) != token_shape:
                raise ValueError(f"{name} must have shape [B,N].")
            if tensor.dtype != torch.long:
                raise ValueError(f"{name} must use torch.long.")
            if bool((tensor < 0).any()):
                raise ValueError(f"{name} must be non-negative.")
        if metadata.candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean.")
        if metadata.role_eligibility.dtype != torch.bool:
            raise ValueError("Registered role_eligibility must be boolean.")
        tensors = (
            metadata.candidate_mask,
            metadata.role_eligibility,
            metadata.range_stratum,
            metadata.angle_stratum,
            metadata.age_stratum,
            metadata.candidate_priority,
        )
        if any(tensor.device != token_valid.device for tensor in tensors):
            raise ValueError(
                "Matched-substitution metadata device must match actor inputs."
            )
        if not torch.equal(metadata.role_eligibility, role_eligibility):
            raise ValueError(
                "Registered role eligibility does not match clean selection input."
            )
        if bool((metadata.candidate_mask & ~token_valid).any()):
            raise ValueError("candidate_mask cannot include invalid tokens.")
        for batch_index in range(batch_size):
            priorities = metadata.candidate_priority[batch_index][
                metadata.candidate_mask[batch_index]
            ]
            if priorities.unique().numel() != priorities.numel():
                raise ValueError(
                    "candidate_priority must be unique within each candidate pool."
                )

    @staticmethod
    def _validate_provenance(
        provenance: SupportMaskProvenance,
    ) -> None:
        if not isinstance(provenance, SupportMaskProvenance):
            raise ValueError(
                "mask_provenance must be a validated SupportMaskProvenance."
            )
        # Reconstructing forces validation even for objects restored through
        # unusual serialization paths that bypassed dataclass construction.
        SupportMaskProvenance(**asdict(provenance))

    def _validate_input_structure(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
    ) -> None:
        if score_features.ndim != 3 or score_features.shape[-1] != self.score_feature_dim:
            raise ValueError(
                "score_features must have shape [B,N,S] with "
                f"S={self.score_feature_dim}."
            )
        batch_size, num_tokens, _ = score_features.shape
        if tuple(terrain_values.shape) != (
            batch_size,
            num_tokens,
            self.terrain_value_dim,
        ):
            raise ValueError(
                "terrain_values must have shape [B,N,V] with "
                f"V={self.terrain_value_dim}."
            )
        if tuple(proprio.shape) != (batch_size, self.proprio_dim):
            raise ValueError(
                f"proprio must have shape [B,{self.proprio_dim}]."
            )
        if tuple(token_valid.shape) != (batch_size, num_tokens):
            raise ValueError("token_valid must have shape [B,N].")
        if tuple(role_eligibility.shape) != (
            batch_size,
            NUM_QUERIES,
            num_tokens,
        ):
            raise ValueError("role_eligibility must have shape [B,4,N].")
        if token_valid.dtype != torch.bool or role_eligibility.dtype != torch.bool:
            raise ValueError("token_valid and role_eligibility must be boolean.")
        tensors = (
            score_features,
            terrain_values,
            proprio,
            token_valid,
            role_eligibility,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("All actor adapter inputs must share one device.")
        if score_features.device != self.query_embedding.device:
            raise ValueError("Inputs and actor adapter parameters must share one device.")
        for name, tensor in (
            ("score_features", score_features),
            ("terrain_values", terrain_values),
            ("proprio", proprio),
        ):
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating point.")

    def _validate_inputs(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
    ) -> None:
        self._validate_input_structure(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            role_eligibility,
        )
        for name, tensor in (
            ("score_features", score_features),
            ("terrain_values", terrain_values),
            ("proprio", proprio),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite floating point.")
        if bool((role_eligibility & ~token_valid[:, None, :]).any()):
            raise ValueError("role_eligibility cannot contain invalid tokens.")


__all__ = [
    "DEPLOYMENT_GEOMETRY_SOURCES",
    "MATCHED_SUBSTITUTION_SCHEMA",
    "SUPPORT_CAUSAL_INTERVENTION_MODES",
    "MatchedSubstitutionMetadata",
    "MatchedSubstitutionShortfallError",
    "SharedUniqueSupportActorAdapter",
    "SupportMaskProvenance",
]
