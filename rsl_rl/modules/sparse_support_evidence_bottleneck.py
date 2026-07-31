# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auditable fixed-budget sparse support-evidence bottleneck.

This module is an isolated CPU-testable scaffold.  It is deliberately not
wired into an actor, simulator, or training configuration.

The caller supplies four semantic eligibility masks in this fixed order:

``left_near, left_far, right_near, right_far``.

Keeping the geometry mask outside this module avoids inventing camera or LiDAR
calibration, near/far thresholds, or body-frame conventions.  A real
integration must construct those masks from calibrated ray/token geometry.

For integer total budgets ``M_total in {4, 8, 16, 32, 64}``, every query receives
``M_total / 4`` slots.  The selected union may contain fewer than ``M_total``
unique tokens when eligibility overlaps or a query has insufficient evidence.
The special budget ``"all"`` selects every valid token and is the non-sparse
upper-bound contract.

Hard top-k has a deliberately honest gradient limitation: score-gated selected
tokens can receive gradients, but the discrete membership boundary and
unselected scores do not.  ``selector_gradient="straight_through"`` adds a
soft-membership surrogate while preserving the exact hard sparse forward value
path.  In both cases, unselected terrain values have exact zero forward weight.

Intervention modes are mechanism hooks, not automatically fair trained
baselines:

* ``full`` uses every clean valid value, whether or not it belongs to a foot
  query's sparse-selection eligibility region.
* ``selected_only`` uses only the clean hard top-k selection.
* ``delete_selected`` removes the frozen clean selected union, without
  reselection.
* ``delete_random`` attempts to remove the same number of unique valid tokens
  using an explicit seed or generator, without reselection.  It samples
  globally without replacement only from ``V minus S``.  If that pool cannot
  supply ``|S|`` tokens, it deletes the available candidates, reports the
  shortfall, and marks the trial unmatched rather than silently touching ``S``.
  A trial with ``|S| = 0`` is reported separately as not applicable.
* ``zero_terrain`` removes every terrain value.

Every mode computes the same clean scores and frozen clean selection first.
Only the subsequent value mask changes.  The output contains no pooled/global
terrain bypass: it is formed solely from the effective per-query values plus a
proprioception-only stability contribution.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Final

import torch
import torch.nn as nn


QUERY_NAMES: Final[tuple[str, ...]] = (
    "left_near",
    "left_far",
    "right_near",
    "right_far",
)
NUM_QUERIES: Final[int] = len(QUERY_NAMES)
SUPPORTED_TOTAL_BUDGETS: Final[tuple[int, ...]] = (4, 8, 16, 32, 64)
SUPPORTED_MODES: Final[tuple[str, ...]] = (
    "full",
    "selected_only",
    "delete_selected",
    "delete_random",
    "zero_terrain",
)
SUPPORTED_SELECTOR_GRADIENTS: Final[tuple[str, ...]] = (
    "hard",
    "straight_through",
)


def _masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Return finite probabilities and exact zeros for fully masked rows."""
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean, got {mask.dtype}.")
    try:
        expanded_mask = torch.broadcast_to(mask, logits.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to logits "
            f"shape {tuple(logits.shape)}."
        ) from error

    masked_logits = torch.where(
        expanded_mask,
        logits,
        torch.full_like(logits, -torch.inf),
    )
    maximum = masked_logits.amax(dim=dim, keepdim=True)
    maximum = torch.where(
        torch.isfinite(maximum),
        maximum,
        torch.zeros_like(maximum),
    )
    exponentials = (
        torch.exp(
            torch.where(
                expanded_mask,
                logits - maximum,
                torch.zeros_like(logits),
            )
        )
        * expanded_mask.to(logits.dtype)
    )
    denominator = exponentials.sum(dim=dim, keepdim=True)
    return exponentials / denominator.clamp_min(
        torch.finfo(logits.dtype).tiny
    )


def _strict_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}, got {normalized}."
        )
    return normalized


class SparseSupportEvidenceBottleneck(nn.Module):
    """Select and aggregate a fixed total budget of support evidence.

    Args:
        score_feature_dim: Width of per-token features used only for scoring.
        terrain_value_dim: Width of the terrain values behind the bottleneck.
        proprio_dim: Width of the proprioceptive state.
        output_dim: Width of the returned policy embedding.
        total_budget: One of ``4, 8, 16, 32, 64`` or ``"all"``.
        score_dim: Width of the score key/query space.
        value_embedding_dim: Width of each query's aggregated terrain value.
        mode: Default intervention mode.
        selector_gradient: ``hard`` or optional ``straight_through``.
    """

    query_names: Final[tuple[str, ...]] = QUERY_NAMES

    def __init__(
        self,
        score_feature_dim: int,
        terrain_value_dim: int,
        proprio_dim: int,
        output_dim: int,
        *,
        total_budget: int | str = 16,
        score_dim: int = 32,
        value_embedding_dim: int = 32,
        mode: str = "selected_only",
        selector_gradient: str = "hard",
    ) -> None:
        super().__init__()
        self.score_feature_dim = _strict_integer(
            "score_feature_dim",
            score_feature_dim,
        )
        self.terrain_value_dim = _strict_integer(
            "terrain_value_dim",
            terrain_value_dim,
        )
        self.proprio_dim = _strict_integer("proprio_dim", proprio_dim)
        self.output_dim = _strict_integer(
            "output_dim",
            output_dim,
            minimum=2,
        )
        self.score_dim = _strict_integer("score_dim", score_dim)
        self.value_embedding_dim = _strict_integer(
            "value_embedding_dim",
            value_embedding_dim,
        )
        self.total_budget = self._normalize_total_budget(total_budget)
        self.mode = self._normalize_mode(mode)
        self.selector_gradient = self._normalize_selector_gradient(
            selector_gradient
        )

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

        self.value_projection = nn.Linear(
            self.terrain_value_dim,
            self.value_embedding_dim,
            bias=False,
        )
        self.terrain_output_projection = nn.Linear(
            NUM_QUERIES * self.value_embedding_dim,
            self.output_dim,
            bias=False,
        )
        self.stability_projection = nn.Sequential(
            nn.Linear(self.proprio_dim, self.output_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    @property
    def per_query_quota(self) -> int | None:
        """Return the fixed quota, or ``None`` for the dynamic ``all`` case."""
        if self.total_budget == "all":
            return None
        return int(self.total_budget) // NUM_QUERIES

    def forward(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        query_eligibility: torch.Tensor,
        *,
        mode: str | None = None,
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return only the bottleneck output embedding."""
        output, _ = self.forward_with_diagnostics(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            query_eligibility,
            mode=mode,
            random_seed=random_seed,
            generator=generator,
        )
        return output

    def forward_with_diagnostics(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        query_eligibility: torch.Tensor,
        *,
        mode: str | None = None,
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the embedding and complete sparse-selection audit tensors."""
        resolved_mode = self.mode if mode is None else self._normalize_mode(mode)
        self._validate_inputs(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            query_eligibility,
        )
        self._validate_random_contract(
            resolved_mode,
            random_seed,
            generator,
            device=terrain_values.device,
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

        batch_size, num_tokens, _ = score_features.shape
        eligible = query_eligibility & token_valid[:, None, :]
        full_valid = token_valid[:, None, :].expand(
            -1,
            NUM_QUERIES,
            -1,
        )
        if self.total_budget == "all":
            selection_candidates = full_valid
        else:
            selection_candidates = eligible
        # Sparse/zero modes only need scores on the clean selection domain.
        # The full and deletion interventions aggregate all remaining valid
        # tokens, including valid wall/ceiling evidence outside every foot
        # query's eligibility region, so their confidence gates must see the
        # real score features for all of V.
        score_input_mask = selection_candidates.any(dim=1)
        if resolved_mode in ("full", "delete_selected", "delete_random"):
            score_input_mask = token_valid
        safe_score_features = torch.where(
            score_input_mask[:, :, None],
            score_features,
            torch.zeros_like(score_features),
        )
        scores = self._clean_scores(safe_score_features, proprio)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(
                "Candidate score projection produced non-finite values."
            )
        masked_scores = torch.where(
            selection_candidates,
            scores,
            torch.full_like(scores, -torch.inf),
        )
        score_probabilities = _masked_softmax(
            scores,
            selection_candidates,
            dim=-1,
        )
        score_entropy = -(
            score_probabilities
            * torch.where(
                score_probabilities > 0.0,
                torch.log(score_probabilities),
                torch.zeros_like(score_probabilities),
            )
        ).sum(dim=-1)

        (
            clean_indices,
            clean_slot_valid,
            clean_query_mask,
            quota,
        ) = self._hard_selection(masked_scores, selection_candidates)
        clean_unique_mask = clean_query_mask.any(dim=1)
        realized_per_query = clean_query_mask.sum(dim=-1)
        realized_slots = realized_per_query.sum(dim=-1)
        unique_selected = clean_unique_mask.sum(dim=-1)
        overlap_count = realized_slots - unique_selected
        pairwise_query_overlap_count = (
            clean_query_mask[:, :, None, :]
            & clean_query_mask[:, None, :, :]
        ).sum(dim=-1)
        if self.total_budget == "all":
            nominal_total_slots = quota.sum(dim=-1)
        else:
            nominal_total_slots = torch.full(
                (batch_size,),
                int(self.total_budget),
                dtype=torch.long,
                device=terrain_values.device,
            )
        nominal_slot_shortfall = nominal_total_slots - realized_slots

        random_delete_mask = torch.zeros_like(token_valid)
        random_delete_indices = torch.full(
            (batch_size, num_tokens),
            -1,
            dtype=torch.long,
            device=terrain_values.device,
        )
        random_delete_slot_valid = torch.zeros(
            batch_size,
            num_tokens,
            dtype=torch.bool,
            device=terrain_values.device,
        )
        if resolved_mode == "delete_random":
            random_delete_mask, random_delete_indices, random_delete_slot_valid = (
                self._random_deletion(
                    token_valid,
                    clean_unique_mask,
                    random_seed=random_seed,
                    generator=generator,
                )
            )

        if resolved_mode == "full":
            effective_query_mask = full_valid
            deleted_mask = torch.zeros_like(token_valid)
        elif resolved_mode == "selected_only":
            effective_query_mask = clean_query_mask
            deleted_mask = torch.zeros_like(token_valid)
        elif resolved_mode == "delete_selected":
            deleted_mask = clean_unique_mask
            effective_query_mask = full_valid & ~deleted_mask[:, None, :]
        elif resolved_mode == "delete_random":
            deleted_mask = random_delete_mask
            effective_query_mask = full_valid & ~deleted_mask[:, None, :]
        else:
            deleted_mask = token_valid
            effective_query_mask = torch.zeros_like(eligible)
        effective_unique_mask = effective_query_mask.any(dim=1)

        if (
            resolved_mode == "selected_only"
            and self.selector_gradient == "straight_through"
        ):
            soft_membership = (
                score_probabilities
                * realized_per_query[:, :, None].to(scores.dtype)
            )
            membership = (
                clean_query_mask.to(scores.dtype)
                + (soft_membership - soft_membership.detach())
            )
        else:
            membership = effective_query_mask.to(scores.dtype)
        effective_count = effective_query_mask.sum(
            dim=-1,
            keepdim=True,
        ).to(scores.dtype)
        aggregation_weights = (
            membership * torch.sigmoid(scores)
        ) / effective_count.clamp_min(1.0)

        safe_terrain_values = torch.where(
            effective_unique_mask[:, :, None],
            terrain_values,
            torch.zeros_like(terrain_values),
        )
        projected_values = self.value_projection(safe_terrain_values)
        if not bool(torch.isfinite(projected_values).all()):
            raise ValueError(
                "Effective terrain value projection produced non-finite "
                "values."
            )
        if not bool(torch.isfinite(aggregation_weights).all()):
            raise ValueError(
                "Sparse aggregation weights produced non-finite values."
            )
        query_values = torch.einsum(
            "bqn,bnd->bqd",
            aggregation_weights,
            projected_values,
        )
        if not bool(torch.isfinite(query_values).all()):
            raise ValueError(
                "Aggregated terrain evidence produced non-finite values."
            )
        terrain_contribution = self.terrain_output_projection(
            query_values.flatten(start_dim=1)
        )
        if not bool(torch.isfinite(terrain_contribution).all()):
            raise ValueError(
                "Terrain output projection produced non-finite values."
            )
        stability_contribution = self.stability_projection(proprio)
        if not bool(torch.isfinite(stability_contribution).all()):
            raise ValueError(
                "Stability projection produced non-finite values."
            )
        fused_contribution = stability_contribution + terrain_contribution
        if not bool(torch.isfinite(fused_contribution).all()):
            raise ValueError(
                "Fused terrain and stability contribution produced "
                "non-finite values."
            )
        output = self.output_norm(fused_contribution)
        if not bool(torch.isfinite(output).all()):
            raise ValueError(
                "Output normalization produced non-finite values."
            )

        if resolved_mode in ("delete_selected", "delete_random"):
            deletion_unique_budget = unique_selected
        elif resolved_mode == "zero_terrain":
            deletion_unique_budget = token_valid.sum(dim=-1)
        else:
            deletion_unique_budget = torch.zeros_like(unique_selected)
        deletion_unique_realized = deleted_mask.sum(dim=-1)
        random_selected_intersection = (
            random_delete_mask & clean_unique_mask
        ).sum(dim=-1)
        random_preferred_candidate_count = (
            token_valid & ~clean_unique_mask
        ).sum(dim=-1)
        random_candidate_shortfall = (
            unique_selected - random_preferred_candidate_count
        ).clamp_min(0)
        if resolved_mode == "delete_random":
            random_not_applicable = deletion_unique_budget == 0
            random_primary_matched = (
                ~random_not_applicable
                & (deletion_unique_realized == deletion_unique_budget)
                & (random_selected_intersection == 0)
            )
            random_unmatched = ~random_not_applicable & ~random_primary_matched
            random_exact_match = (
                ~random_not_applicable
                & (random_delete_mask == clean_unique_mask).all(dim=-1)
            )
        else:
            random_not_applicable = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=terrain_values.device,
            )
            random_primary_matched = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=terrain_values.device,
            )
            random_unmatched = torch.zeros_like(random_primary_matched)
            random_exact_match = torch.zeros_like(random_primary_matched)
        clean_selection_frozen = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=terrain_values.device,
        )
        no_reselection = torch.ones_like(clean_selection_frozen)
        random_seed_diagnostic = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=terrain_values.device,
        )
        if random_seed is not None:
            random_seed_diagnostic.fill_(int(random_seed))

        diagnostics = {
            "selection_indices": clean_indices,
            "selection_slot_valid": clean_slot_valid,
            "selection_query_mask": clean_query_mask,
            "selection_unique_mask": clean_unique_mask,
            "selection_candidate_mask": selection_candidates,
            "selection_scores": masked_scores,
            "score_probabilities": score_probabilities,
            "score_entropy": score_entropy,
            "quota_per_query": quota,
            "nominal_total_slot_budget": nominal_total_slots,
            "realized_per_query": realized_per_query,
            "realized_slot_count": realized_slots,
            "nominal_slot_shortfall": nominal_slot_shortfall,
            "unique_selected_count": unique_selected,
            "overlap_count": overlap_count,
            "pairwise_query_overlap_count": pairwise_query_overlap_count,
            "eligible_count_per_query": eligible.sum(dim=-1),
            "valid_token_count": token_valid.sum(dim=-1),
            "effective_query_mask": effective_query_mask,
            "effective_unique_mask": effective_unique_mask,
            "aggregation_weights": aggregation_weights,
            "deleted_mask": deleted_mask,
            "deletion_unique_budget": deletion_unique_budget,
            "deletion_unique_realized_count": deletion_unique_realized,
            "deletion_unique_budget_match": (
                deletion_unique_realized == deletion_unique_budget
            ),
            "random_delete_mask": random_delete_mask,
            "random_delete_indices": random_delete_indices,
            "random_delete_slot_valid": random_delete_slot_valid,
            "random_selected_intersection_count": random_selected_intersection,
            "random_preferred_candidate_count": (
                random_preferred_candidate_count
            ),
            "random_candidate_shortfall": random_candidate_shortfall,
            "random_not_applicable": random_not_applicable,
            "random_primary_matched": random_primary_matched,
            "random_unmatched": random_unmatched,
            "random_delete_exactly_matches_selected": random_exact_match,
            "random_seed": random_seed_diagnostic,
            "clean_selection_frozen": clean_selection_frozen,
            "no_reselection": no_reselection,
            "query_values": query_values,
            "terrain_contribution": terrain_contribution,
            "stability_contribution": stability_contribution,
        }
        return output, diagnostics

    def _clean_scores(
        self,
        score_features: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = score_features.shape[0]
        keys = self.score_key_projection(score_features)
        queries = self.score_query_projection(proprio).reshape(
            batch_size,
            NUM_QUERIES,
            self.score_dim,
        )
        queries = queries + self.query_embedding
        return (
            torch.einsum("bqd,bnd->bqn", queries, keys) * self.score_scale
        )

    def _hard_selection(
        self,
        masked_scores: torch.Tensor,
        eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, num_tokens = masked_scores.shape
        sorted_indices = torch.argsort(
            masked_scores,
            dim=-1,
            descending=True,
            stable=True,
        )
        if self.total_budget == "all":
            maximum_slots = num_tokens
            quota = eligible.sum(dim=-1)
        else:
            maximum_slots = int(self.per_query_quota)
            quota = torch.full(
                (batch_size, NUM_QUERIES),
                maximum_slots,
                dtype=torch.long,
                device=masked_scores.device,
            )

        available_width = min(maximum_slots, num_tokens)
        candidate_indices = sorted_indices[..., :available_width]
        candidate_valid = torch.gather(
            eligible,
            dim=-1,
            index=candidate_indices,
        )
        indices = torch.full(
            (batch_size, NUM_QUERIES, maximum_slots),
            -1,
            dtype=torch.long,
            device=masked_scores.device,
        )
        slot_valid = torch.zeros(
            batch_size,
            NUM_QUERIES,
            maximum_slots,
            dtype=torch.bool,
            device=masked_scores.device,
        )
        indices[..., :available_width] = torch.where(
            candidate_valid,
            candidate_indices,
            torch.full_like(candidate_indices, -1),
        )
        slot_valid[..., :available_width] = candidate_valid

        # The extra sentinel column prevents padded -1 slots from overwriting a
        # real selection at token index zero during scatter.
        selection_with_sentinel = torch.zeros(
            batch_size,
            NUM_QUERIES,
            num_tokens + 1,
            dtype=torch.bool,
            device=masked_scores.device,
        )
        scatter_indices = torch.where(
            slot_valid,
            indices,
            torch.full_like(indices, num_tokens),
        )
        selection_with_sentinel.scatter_(
            dim=-1,
            index=scatter_indices,
            src=torch.ones_like(slot_valid),
        )
        selection_mask = selection_with_sentinel[..., :num_tokens]
        return indices, slot_valid, selection_mask, quota

    def _random_deletion(
        self,
        token_valid: torch.Tensor,
        clean_selected: torch.Tensor,
        *,
        random_seed: int | None,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = token_valid.shape
        target_count = clean_selected.sum(dim=-1)
        if generator is None:
            generator = torch.Generator(device=token_valid.device)
            generator.manual_seed(int(random_seed))
        priorities = torch.rand(
            batch_size,
            num_tokens,
            dtype=torch.float32,
            device=token_valid.device,
            generator=generator,
        )
        preferred = token_valid & ~clean_selected
        priorities = torch.where(
            preferred,
            priorities,
            torch.full_like(priorities, torch.inf),
        )
        sorted_indices = torch.argsort(
            priorities,
            dim=-1,
            descending=False,
            stable=True,
        )
        ranks = torch.arange(
            num_tokens,
            device=token_valid.device,
        )[None, :]
        slot_valid = ranks < target_count[:, None]
        sorted_preferred = torch.gather(
            preferred,
            dim=-1,
            index=sorted_indices,
        )
        slot_valid = slot_valid & sorted_preferred
        indices = torch.where(
            slot_valid,
            sorted_indices,
            torch.full_like(sorted_indices, -1),
        )

        deletion_with_sentinel = torch.zeros(
            batch_size,
            num_tokens + 1,
            dtype=torch.bool,
            device=token_valid.device,
        )
        scatter_indices = torch.where(
            slot_valid,
            indices,
            torch.full_like(indices, num_tokens),
        )
        deletion_with_sentinel.scatter_(
            dim=-1,
            index=scatter_indices,
            src=torch.ones_like(slot_valid),
        )
        return (
            deletion_with_sentinel[:, :num_tokens],
            indices,
            slot_valid,
        )

    @staticmethod
    def _normalize_total_budget(total_budget: int | str) -> int | str:
        if isinstance(total_budget, str):
            normalized = total_budget.lower()
            if normalized != "all":
                raise ValueError(
                    "total_budget must be one of "
                    f"{SUPPORTED_TOTAL_BUDGETS} or 'all', got "
                    f"{total_budget!r}."
                )
            return normalized
        if isinstance(total_budget, bool) or not isinstance(
            total_budget,
            Integral,
        ):
            raise ValueError(
                "total_budget must be an integer or 'all', got "
                f"{total_budget!r}."
            )
        normalized = int(total_budget)
        if normalized not in SUPPORTED_TOTAL_BUDGETS:
            raise ValueError(
                "total_budget must be one of "
                f"{SUPPORTED_TOTAL_BUDGETS} or 'all', got {normalized}."
            )
        return normalized

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError(
                f"mode must be a string, got {type(mode).__name__}."
            )
        normalized = mode.lower().replace("-", "_")
        if normalized not in SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_MODES}, got {mode!r}."
            )
        return normalized

    @staticmethod
    def _normalize_selector_gradient(selector_gradient: str) -> str:
        if not isinstance(selector_gradient, str):
            raise ValueError(
                "selector_gradient must be a string, got "
                f"{type(selector_gradient).__name__}."
            )
        normalized = selector_gradient.lower().replace("-", "_")
        if normalized not in SUPPORTED_SELECTOR_GRADIENTS:
            raise ValueError(
                "selector_gradient must be one of "
                f"{SUPPORTED_SELECTOR_GRADIENTS}, got "
                f"{selector_gradient!r}."
            )
        return normalized

    @staticmethod
    def _validate_random_contract(
        mode: str,
        random_seed: int | None,
        generator: torch.Generator | None,
        *,
        device: torch.device,
    ) -> None:
        if mode != "delete_random":
            if random_seed is not None or generator is not None:
                raise ValueError(
                    "random_seed/generator are only valid in "
                    "delete_random mode."
                )
            return
        if (random_seed is None) == (generator is None):
            raise ValueError(
                "delete_random mode requires exactly one explicit "
                "random_seed or generator."
            )
        if random_seed is not None:
            if isinstance(random_seed, bool) or not isinstance(
                random_seed,
                Integral,
            ):
                raise ValueError(
                    f"random_seed must be an integer, got {random_seed!r}."
                )
            if int(random_seed) < 0 or int(random_seed) >= 2**63:
                raise ValueError(
                    "random_seed must satisfy 0 <= seed < 2**63, got "
                    f"{random_seed}."
                )
        if generator is not None and generator.device != device:
            raise ValueError(
                "generator and inputs must use the same device, got "
                f"generator={generator.device}, inputs={device}."
            )

    def _validate_inputs(
        self,
        score_features: torch.Tensor,
        terrain_values: torch.Tensor,
        proprio: torch.Tensor,
        token_valid: torch.Tensor,
        query_eligibility: torch.Tensor,
    ) -> None:
        if score_features.ndim != 3 or score_features.shape[-1] != (
            self.score_feature_dim
        ):
            raise ValueError(
                "score_features must have shape [B, N, S] with "
                f"S={self.score_feature_dim}, got "
                f"{tuple(score_features.shape)}."
            )
        batch_size, num_tokens, _ = score_features.shape
        if batch_size <= 0 or num_tokens <= 0:
            raise ValueError(
                "score_features batch and token dimensions must be positive."
            )
        if tuple(terrain_values.shape) != (
            batch_size,
            num_tokens,
            self.terrain_value_dim,
        ):
            raise ValueError(
                "terrain_values must have shape [B, N, V] with "
                f"V={self.terrain_value_dim}, got "
                f"{tuple(terrain_values.shape)}."
            )
        if tuple(proprio.shape) != (batch_size, self.proprio_dim):
            raise ValueError(
                "proprio must have shape [B, P] with "
                f"P={self.proprio_dim}, got {tuple(proprio.shape)}."
            )
        if tuple(token_valid.shape) != (batch_size, num_tokens):
            raise ValueError(
                "token_valid must have shape [B, N], got "
                f"{tuple(token_valid.shape)}."
            )
        if tuple(query_eligibility.shape) != (
            batch_size,
            NUM_QUERIES,
            num_tokens,
        ):
            raise ValueError(
                "query_eligibility must have shape [B, 4, N] in order "
                f"{QUERY_NAMES}, got {tuple(query_eligibility.shape)}."
            )
        if token_valid.dtype != torch.bool:
            raise ValueError(
                f"token_valid must be boolean, got {token_valid.dtype}."
            )
        if query_eligibility.dtype != torch.bool:
            raise ValueError(
                "query_eligibility must be boolean, got "
                f"{query_eligibility.dtype}."
            )

        tensors = {
            "score_features": score_features,
            "terrain_values": terrain_values,
            "proprio": proprio,
            "token_valid": token_valid,
            "query_eligibility": query_eligibility,
        }
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
            detail = ", ".join(
                f"{name}={tensor.device}" for name, tensor in tensors.items()
            )
            raise ValueError(
                f"All inputs must use the same device, got {detail}."
            )
        parameter_device = self.query_embedding.device
        if score_features.device != parameter_device:
            raise ValueError(
                "Inputs and module parameters must use the same device, got "
                f"inputs={score_features.device}, "
                f"parameters={parameter_device}."
            )
        for name, tensor in (
            ("score_features", score_features),
            ("terrain_values", terrain_values),
            ("proprio", proprio),
        ):
            if not tensor.is_floating_point():
                raise ValueError(
                    f"{name} must be floating-point, got {tensor.dtype}."
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"{name} must contain only finite values."
                )

        if bool((query_eligibility & ~token_valid[:, None, :]).any()):
            raise ValueError(
                "query_eligibility cannot include invalid tokens."
            )


__all__ = [
    "NUM_QUERIES",
    "QUERY_NAMES",
    "SUPPORTED_MODES",
    "SUPPORTED_SELECTOR_GRADIENTS",
    "SUPPORTED_TOTAL_BUDGETS",
    "SparseSupportEvidenceBottleneck",
]
