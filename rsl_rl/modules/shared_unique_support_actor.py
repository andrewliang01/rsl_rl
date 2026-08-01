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
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Final

import torch
import torch.nn as nn

from .sparse_support_evidence_bottleneck import NUM_QUERIES, QUERY_NAMES
from .support_selection_ablation import FixedBudgetSupportSelector


DEPLOYMENT_GEOMETRY_SOURCES: Final[tuple[str, ...]] = (
    "calibrated_lidar_ray_geometry",
    "calibrated_depth_ray_geometry",
    "calibrated_multisensor_ray_geometry",
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, value, _ = self.forward_with_diagnostics(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            role_eligibility,
            mask_provenance=mask_provenance,
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        # The value projection sees [B,M,V], never [B,N,V].
        selected_embeddings = self.selected_value_projection(selected_values)

        selected_role_eligibility = torch.gather(
            role_eligibility,
            2,
            safe_indices[:, None, :].expand(-1, NUM_QUERIES, -1),
        )
        consumption_mask = (
            selected_role_eligibility & slot_valid[:, None, :]
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
        return action, value, diagnostics

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

    def _validate_inputs(
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
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite floating point.")
        if bool((role_eligibility & ~token_valid[:, None, :]).any()):
            raise ValueError("role_eligibility cannot contain invalid tokens.")


__all__ = [
    "DEPLOYMENT_GEOMETRY_SOURCES",
    "SharedUniqueSupportActorAdapter",
    "SupportMaskProvenance",
]
