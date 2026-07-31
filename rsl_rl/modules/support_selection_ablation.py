# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auditable selectors for the fixed-support H1 ablation family.

This module only chooses evidence tokens.  It deliberately does not own ray
geometry, role calibration, value aggregation, or a policy head.  Keeping
selection separate makes the comparison usable with one common downstream
backbone and prevents a hidden full-token bypass.

The sparse strategies all spend one *shared total* budget ``M`` on unique
tokens:

``glad_top_m``
    State-conditioned global top-M control.  Role eligibility is ignored for
    membership, and the best query score only assigns a selected token to one
    common aggregation slot.  This is an overlap baseline, not a novelty
    claim about GLAD.
``role_shared_total_m``
    Top-M over the union of caller-supplied calibrated support roles.  A token
    is assigned to its highest-scoring eligible role and can spend only one
    slot even when role masks overlap.
``matched_random_m``
    Random-M without replacement from the exact same eligible union as the
    role-constrained selector.  It is matched in candidate opportunity and
    realized unique cardinality.  It is not silently described as
    range/angle/age-stratified matching; those controls require explicit
    stratum metadata in a later integration.
``full``
    Every valid token is retained and replicated to all four role aggregation
    slots.  This is the non-bottleneck upper-bound contract.

All top-M operations use stable sorting, so equal scores resolve toward the
lowest token index.  The random control requires an explicit seed or generator
and exports enough tensors to audit exact membership.
"""

from __future__ import annotations

from numbers import Integral
from typing import Final

import torch
import torch.nn as nn

from .sparse_support_evidence_bottleneck import NUM_QUERIES, QUERY_NAMES


SUPPORTED_SHARED_TOTAL_BUDGETS: Final[tuple[int, ...]] = (8, 16, 32, 64)
SUPPORTED_SUPPORT_SELECTION_STRATEGIES: Final[tuple[str, ...]] = (
    "full",
    "glad_top_m",
    "role_shared_total_m",
    "matched_random_m",
)


class FixedBudgetSupportSelector(nn.Module):
    """Choose a full or shared-total-M support-token set.

    Inputs use ``Q=4`` in the fixed order
    ``left_current_support, left_landing_support, right_current_support,
    right_landing_support``.  ``scores`` must already
    be conditioned on the policy state; the selector adds no trainable
    parameters and therefore keeps the scorer/backbone common across arms.
    """

    query_names: Final[tuple[str, ...]] = QUERY_NAMES

    def __init__(
        self,
        *,
        strategy: str = "role_shared_total_m",
        total_budget: int | str = 16,
    ) -> None:
        super().__init__()
        self.strategy = self._normalize_strategy(strategy)
        self.total_budget = self._normalize_budget(total_budget)
        if self.strategy == "full" and self.total_budget != "all":
            raise ValueError("strategy='full' requires total_budget='all'.")
        if self.strategy != "full" and self.total_budget == "all":
            raise ValueError(
                "Sparse selection strategies require an integer total_budget."
            )

    def forward(
        self,
        scores: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
        *,
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``[B,Q,N]`` selected membership and audit diagnostics."""
        self._validate_inputs(scores, token_valid, role_eligibility)
        self._validate_randomness(random_seed, generator, scores.device)

        batch_size, _, num_tokens = scores.shape
        role_candidates = role_eligibility & token_valid[:, None, :]
        role_union = role_candidates.any(dim=1)
        full_query_mask = token_valid[:, None, :].expand(
            -1, NUM_QUERIES, -1
        )

        if self.strategy == "full":
            query_mask = full_query_mask
            unique_mask = token_valid
            indices = torch.arange(
                num_tokens, device=scores.device, dtype=torch.long
            )[None, :].expand(batch_size, -1)
            slot_valid = token_valid
            candidate_mask = token_valid
            assignment = torch.full(
                (batch_size, num_tokens),
                -1,
                device=scores.device,
                dtype=torch.long,
            )
            requested = token_valid.sum(dim=-1)
            selector_entropy = torch.zeros(
                batch_size, device=scores.device, dtype=scores.dtype
            )
            selector_entropy_applicable = torch.zeros(
                batch_size, device=scores.device, dtype=torch.bool
            )
        else:
            budget = int(self.total_budget)
            if self.strategy == "glad_top_m":
                candidate_mask = token_valid
                token_scores, assignment = scores.max(dim=1)
                selector_entropy_applicable = torch.ones(
                    batch_size, device=scores.device, dtype=torch.bool
                )
            elif self.strategy == "role_shared_total_m":
                candidate_mask = role_union
                token_scores, assignment = self._eligible_token_scores(
                    scores, role_candidates
                )
                selector_entropy_applicable = torch.ones(
                    batch_size, device=scores.device, dtype=torch.bool
                )
            else:
                candidate_mask = role_union
                # Role assignment is deterministic and independent of learned
                # score magnitude for the random-selection control.
                assignment = role_candidates.to(torch.int64).argmax(dim=1)
                priorities = torch.rand(
                    batch_size,
                    num_tokens,
                    dtype=torch.float32,
                    device=scores.device,
                    generator=self._resolved_generator(
                        random_seed, generator, scores.device
                    ),
                )
                token_scores = -priorities.to(dtype=scores.dtype)
                selector_entropy_applicable = torch.zeros(
                    batch_size, device=scores.device, dtype=torch.bool
                )

            selector_entropy = self._masked_entropy(
                token_scores, candidate_mask
            )
            selector_entropy = torch.where(
                selector_entropy_applicable,
                selector_entropy,
                torch.zeros_like(selector_entropy),
            )

            indices, slot_valid, unique_mask = self._top_m_unique(
                token_scores,
                candidate_mask,
                budget,
            )
            query_mask = torch.zeros_like(role_candidates)
            safe_indices = torch.where(
                slot_valid,
                indices,
                torch.zeros_like(indices),
            )
            selected_roles = torch.gather(assignment, 1, safe_indices)
            batch_indices = torch.arange(
                batch_size, device=scores.device
            )[:, None].expand_as(indices)
            query_mask[
                batch_indices[slot_valid],
                selected_roles[slot_valid],
                indices[slot_valid],
            ] = True
            requested = torch.full(
                (batch_size,),
                budget,
                device=scores.device,
                dtype=torch.long,
            )

        realized = unique_mask.sum(dim=-1)
        per_role = query_mask.sum(dim=-1)
        pairwise_overlap = (
            query_mask[:, :, None, :] & query_mask[:, None, :, :]
        ).sum(dim=-1)
        selected_score = torch.where(
            query_mask,
            scores,
            torch.zeros_like(scores),
        )
        seed_value = -1 if random_seed is None else int(random_seed)
        diagnostics = {
            "selection_query_mask": query_mask,
            "selection_unique_mask": unique_mask,
            "selection_indices": indices,
            "selection_slot_valid": slot_valid,
            "selection_candidate_mask": candidate_mask,
            "role_candidate_mask": role_candidates,
            "role_union_candidate_mask": role_union,
            "requested_unique_budget": requested,
            "realized_unique_count": realized,
            "unique_budget_shortfall": requested - realized,
            "realized_per_role": per_role,
            "pairwise_role_overlap_count": pairwise_overlap,
            "selected_score_contribution": selected_score,
            "valid_token_count": token_valid.sum(dim=-1),
            "candidate_count": candidate_mask.sum(dim=-1),
            "selector_score_entropy": selector_entropy,
            "selector_score_entropy_applicable": selector_entropy_applicable,
            "candidate_opportunity_matched_to_role_selector": torch.full(
                (batch_size,),
                self.strategy in (
                    "role_shared_total_m",
                    "matched_random_m",
                ),
                device=scores.device,
                dtype=torch.bool,
            ),
            "random_seed": torch.full(
                (batch_size,),
                seed_value,
                device=scores.device,
                dtype=torch.long,
            ),
        }
        return query_mask, diagnostics

    @staticmethod
    def _masked_entropy(
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = torch.where(
            mask, logits, torch.full_like(logits, -torch.inf)
        )
        maximum = masked.amax(dim=-1, keepdim=True)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        exponentials = torch.where(
            mask,
            torch.exp(logits - maximum),
            torch.zeros_like(logits),
        )
        probabilities = exponentials / exponentials.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(logits.dtype).tiny)
        return -(
            probabilities
            * torch.where(
                probabilities > 0,
                torch.log(probabilities),
                torch.zeros_like(probabilities),
            )
        ).sum(dim=-1)

    @staticmethod
    def _eligible_token_scores(
        scores: torch.Tensor,
        role_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = torch.where(
            role_candidates,
            scores,
            torch.full_like(scores, -torch.inf),
        )
        return masked.max(dim=1)

    @staticmethod
    def _top_m_unique(
        token_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = token_scores.shape
        masked = torch.where(
            candidate_mask,
            token_scores,
            torch.full_like(token_scores, -torch.inf),
        )
        order = torch.argsort(masked, dim=-1, descending=True, stable=True)
        width = min(budget, num_tokens)
        selected = order[:, :width]
        selected_valid = torch.gather(candidate_mask, 1, selected)
        indices = torch.full(
            (batch_size, budget),
            -1,
            device=token_scores.device,
            dtype=torch.long,
        )
        slot_valid = torch.zeros(
            batch_size,
            budget,
            device=token_scores.device,
            dtype=torch.bool,
        )
        indices[:, :width] = torch.where(
            selected_valid,
            selected,
            torch.full_like(selected, -1),
        )
        slot_valid[:, :width] = selected_valid
        mask_with_sentinel = torch.zeros(
            batch_size,
            num_tokens + 1,
            device=token_scores.device,
            dtype=torch.bool,
        )
        scatter_indices = torch.where(
            slot_valid,
            indices,
            torch.full_like(indices, num_tokens),
        )
        mask_with_sentinel.scatter_(
            1,
            scatter_indices,
            torch.ones_like(slot_valid),
        )
        return indices, slot_valid, mask_with_sentinel[:, :num_tokens]

    @staticmethod
    def _resolved_generator(
        random_seed: int | None,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Generator:
        if generator is not None:
            return generator
        resolved = torch.Generator(device=device)
        resolved.manual_seed(int(random_seed))
        return resolved

    def _validate_randomness(
        self,
        random_seed: int | None,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> None:
        if random_seed is not None and generator is not None:
            raise ValueError("Provide random_seed or generator, not both.")
        if self.strategy == "matched_random_m":
            if random_seed is None and generator is None:
                raise ValueError(
                    "matched_random_m requires an explicit random_seed or generator."
                )
            if random_seed is not None and (
                isinstance(random_seed, bool)
                or not isinstance(random_seed, Integral)
                or int(random_seed) < 0
            ):
                raise ValueError("random_seed must be a non-negative integer.")
            if generator is not None and generator.device != device:
                raise ValueError(
                    "generator device must match scores device: "
                    f"{generator.device} != {device}."
                )
        elif random_seed is not None or generator is not None:
            raise ValueError(
                "Randomness is accepted only by strategy='matched_random_m'."
            )

    @staticmethod
    def _validate_inputs(
        scores: torch.Tensor,
        token_valid: torch.Tensor,
        role_eligibility: torch.Tensor,
    ) -> None:
        if scores.ndim != 3 or scores.shape[1] != NUM_QUERIES:
            raise ValueError(
                f"scores must have shape [B,{NUM_QUERIES},N], got "
                f"{tuple(scores.shape)}."
            )
        if not scores.is_floating_point():
            raise ValueError("scores must be floating point.")
        expected_token = (scores.shape[0], scores.shape[2])
        if tuple(token_valid.shape) != expected_token:
            raise ValueError(
                f"token_valid must have shape {expected_token}, got "
                f"{tuple(token_valid.shape)}."
            )
        if token_valid.dtype != torch.bool:
            raise ValueError("token_valid must be boolean.")
        if tuple(role_eligibility.shape) != tuple(scores.shape):
            raise ValueError(
                "role_eligibility must match scores shape, got "
                f"{tuple(role_eligibility.shape)} and {tuple(scores.shape)}."
            )
        if role_eligibility.dtype != torch.bool:
            raise ValueError("role_eligibility must be boolean.")
        if token_valid.device != scores.device or role_eligibility.device != scores.device:
            raise ValueError("scores and masks must be on the same device.")
        relevant = token_valid[:, None, :].expand_as(scores)
        if not bool(torch.isfinite(scores[relevant]).all()):
            raise ValueError("Scores on valid tokens must be finite.")

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        if not isinstance(strategy, str):
            raise ValueError("strategy must be a string.")
        resolved = strategy.lower().replace("-", "_")
        if resolved not in SUPPORTED_SUPPORT_SELECTION_STRATEGIES:
            raise ValueError(
                "strategy must be one of "
                f"{SUPPORTED_SUPPORT_SELECTION_STRATEGIES}, got {strategy!r}."
            )
        return resolved

    @staticmethod
    def _normalize_budget(total_budget: int | str) -> int | str:
        if isinstance(total_budget, str):
            if total_budget.lower() != "all":
                raise ValueError("String total_budget must be 'all'.")
            return "all"
        if isinstance(total_budget, bool) or not isinstance(total_budget, Integral):
            raise ValueError("total_budget must be an integer or 'all'.")
        resolved = int(total_budget)
        if resolved not in SUPPORTED_SHARED_TOTAL_BUDGETS:
            raise ValueError(
                "total_budget must be one of "
                f"{SUPPORTED_SHARED_TOTAL_BUDGETS}, got {resolved}."
            )
        return resolved


__all__ = [
    "FixedBudgetSupportSelector",
    "SUPPORTED_SHARED_TOTAL_BUDGETS",
    "SUPPORTED_SUPPORT_SELECTION_STRATEGIES",
]
