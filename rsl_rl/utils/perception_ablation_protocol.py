# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical names and validation for the H1/H2 experiment matrix.

The registry intentionally distinguishes a CPU-ready component protocol from
a simulator-ready training task.  Today the actor observation contains only
range and hit mask, while H1 additionally needs calibrated role eligibility
and H2 needs per-return acquisition time (plus an externally produced
rerender).  Consequently these specs export *reserved override keys* and
``training_ready=False`` rather than registering deceptive Gym task names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


H1_BUDGETS: Final[tuple[int, ...]] = (8, 16, 32, 64)
H1_ROLE_SEMANTICS: Final[str] = "cteq_current_landing_v2"
H1_SELECTORS: Final[tuple[str, ...]] = (
    "full",
    "glad_top_m",
    "role_shared_total_m",
    "matched_random_m",
)
H2_GEOMETRIES: Final[tuple[str, ...]] = ("native", "rerender")
H2_ASSOCIATIONS: Final[tuple[str, ...]] = ("correct", "shuffled")
H2_TEMPORAL_BASELINES: Final[tuple[str, ...]] = (
    "per_return_age",
    "packet_age",
    "age_zero",
)
H2_HISTORY_REDUCTIONS: Final[tuple[str, ...]] = (
    "history",
    "exact_union_k1",
    "raster_latest_event_prototype",
)
H1_METRIC_KEYS: Final[tuple[str, ...]] = (
    "h1/requested_unique_budget",
    "h1/realized_unique_count",
    "h1/unique_budget_shortfall",
    "h1/realized_per_role",
    "h1/pairwise_role_overlap_count",
    "h1/candidate_count",
    "h1/valid_token_count",
    "h1/selector_score_entropy",
)
H2_METRIC_KEYS: Final[tuple[str, ...]] = (
    "h2/valid_return_count_per_frame",
    "h2/return_age_mean_s",
    "h2/return_age_min_s",
    "h2/return_age_max_s",
    "h2/return_age_span_s",
    "h2/shuffled_multiset_conserved",
    "h2/changed_age_association_count",
    "h2/exact_union_collision_cell_count",
)
PAIRED_CAUSAL_METRIC_KEYS: Final[tuple[str, ...]] = (
    "causal/action_mean_l2_clean_vs_intervention",
    "causal/action_kl_clean_vs_intervention",
    "causal/value_delta_clean_vs_intervention",
    "episode/terrain_success_rate",
    "episode/fall_rate",
    "episode/edge_clearance_min_m",
    "episode/unsafe_step_count",
)


@dataclass(frozen=True)
class H1AblationSpec:
    """One fixed-support evidence training arm."""

    name: str
    selector: str
    total_budget: int | str
    value_intervention: str
    random_seed: int | None
    training_ready: bool = False

    def __post_init__(self) -> None:
        if self.selector not in H1_SELECTORS:
            raise ValueError(f"Unsupported H1 selector {self.selector!r}.")
        if self.value_intervention not in ("full", "selected_only"):
            raise ValueError("H1 value intervention must be full or selected_only.")
        if self.selector == "full":
            if self.total_budget != "all" or self.value_intervention != "full":
                raise ValueError("The H1 full arm requires budget=all and full values.")
        else:
            if self.total_budget not in H1_BUDGETS:
                raise ValueError(f"Sparse H1 budget must be one of {H1_BUDGETS}.")
            if self.value_intervention != "selected_only":
                raise ValueError("Sparse H1 arms must use selected_only values.")
        if self.selector == "matched_random_m":
            if self.random_seed is None or self.random_seed < 0:
                raise ValueError("matched_random_m requires a non-negative seed.")
        elif self.random_seed is not None:
            raise ValueError("Only matched_random_m accepts random_seed.")
        if self.training_ready:
            raise ValueError(
                "H1 cannot be marked training-ready until calibrated role masks "
                "and the no-bypass actor adapter are connected."
            )

    @property
    def reserved_overrides(self) -> tuple[str, ...]:
        """Return stable future Hydra keys; these are not active today."""
        values = (
            "actor.support_bottleneck_enabled=true",
            f"actor.support_selector={self.selector}",
            f"actor.support_total_budget={self.total_budget}",
            f"actor.support_value_intervention={self.value_intervention}",
        )
        if self.random_seed is not None:
            values += (f"actor.support_random_seed={self.random_seed}",)
        return values

    def receipt(self) -> dict[str, object]:
        result = asdict(self)
        result["reserved_overrides"] = self.reserved_overrides
        result["blocking_inputs"] = (
            "calibrated left/right current-support/landing-support role masks",
            "actor token scorer/value adapter with no full-token bypass",
        )
        return result


@dataclass(frozen=True)
class H2AblationSpec:
    """One ray-event geometry/time causal arm."""

    name: str
    geometry: str
    time_association: str
    temporal_baseline: str = "per_return_age"
    history_reduction: str = "history"
    shuffle_seed: int | None = None
    training_ready: bool = False

    def __post_init__(self) -> None:
        if self.geometry not in H2_GEOMETRIES:
            raise ValueError(f"Unsupported H2 geometry {self.geometry!r}.")
        if self.time_association not in H2_ASSOCIATIONS:
            raise ValueError(
                f"Unsupported H2 association {self.time_association!r}."
            )
        if self.temporal_baseline not in H2_TEMPORAL_BASELINES:
            raise ValueError(
                f"Unsupported H2 baseline {self.temporal_baseline!r}."
            )
        if self.history_reduction not in H2_HISTORY_REDUCTIONS:
            raise ValueError(
                f"Unsupported H2 history reduction {self.history_reduction!r}."
            )
        if self.time_association == "shuffled":
            if self.shuffle_seed is None or self.shuffle_seed < 0:
                raise ValueError("Shuffled time requires a non-negative seed.")
        elif self.shuffle_seed is not None:
            raise ValueError("Correct time association rejects shuffle_seed.")
        if self.training_ready:
            raise ValueError(
                "H2 cannot be marked training-ready until event-time tensors "
                "are part of the actor observation receipt."
            )

    @property
    def reserved_overrides(self) -> tuple[str, ...]:
        """Return stable future Hydra keys; these are not active today."""
        values = (
            "actor.elevation_encoder_type=ray_event_time",
            "actor.ray_event_training_ready=false",
            "actor.ray_event_time_source=livox_per_return",
            f"actor.ray_event_time_mode={self.temporal_baseline}",
            f"observation.ray_event_geometry={self.geometry}",
            f"observation.ray_event_time_association={self.time_association}",
            f"observation.ray_event_history_reduction={self.history_reduction}",
        )
        if self.shuffle_seed is not None:
            values += (f"actor.ray_event_shuffle_seed={self.shuffle_seed}",)
        return values

    def receipt(self) -> dict[str, object]:
        result = asdict(self)
        result["reserved_overrides"] = self.reserved_overrides
        blockers = [
            "formal 64-environment smoke receipt for this exact contract",
            "training-ready promotion bound to the smoke SHA-256",
        ]
        if self.geometry == "rerender":
            blockers.append("externally rerendered range/valid/age tensor receipt")
        result["blocking_inputs"] = tuple(blockers)
        return result


def build_h1_protocol(*, random_seed: int = 42) -> dict[str, H1AblationSpec]:
    """Build full plus M={8,16,32,64} sparse controls."""
    specs: dict[str, H1AblationSpec] = {}
    full = H1AblationSpec(
        name="h1_full",
        selector="full",
        total_budget="all",
        value_intervention="full",
        random_seed=None,
    )
    specs[full.name] = full
    for budget in H1_BUDGETS:
        for label, selector in (
            ("glad", "glad_top_m"),
            ("role", "role_shared_total_m"),
            ("random", "matched_random_m"),
        ):
            name = f"h1_{label}_m{budget:02d}_selected_only"
            spec = H1AblationSpec(
                name=name,
                selector=selector,
                total_budget=budget,
                value_intervention="selected_only",
                random_seed=random_seed if selector == "matched_random_m" else None,
            )
            specs[name] = spec
    return specs


def build_h2_protocol(*, shuffle_seed: int = 42) -> dict[str, H2AblationSpec]:
    """Build the causal 2x2 and the exact-union/time-baseline controls."""
    specs: dict[str, H2AblationSpec] = {}
    for geometry in H2_GEOMETRIES:
        for association in H2_ASSOCIATIONS:
            name = f"h2_{geometry}_{association}_history_per_return_age"
            spec = H2AblationSpec(
                name=name,
                geometry=geometry,
                time_association=association,
                shuffle_seed=shuffle_seed if association == "shuffled" else None,
            )
            specs[name] = spec

    for history_reduction, baseline in (
        ("exact_union_k1", "per_return_age"),
        ("raster_latest_event_prototype", "per_return_age"),
        ("history", "packet_age"),
        ("history", "age_zero"),
    ):
        name = f"h2_native_correct_{history_reduction}_{baseline}"
        spec = H2AblationSpec(
            name=name,
            geometry="native",
            time_association="correct",
            temporal_baseline=baseline,
            history_reduction=history_reduction,
        )
        specs[name] = spec
    return specs


def perception_ablation_receipt() -> dict[str, object]:
    """Return a serialization-friendly complete experiment receipt."""
    h1 = build_h1_protocol()
    h2 = build_h2_protocol()
    return {
        "schema": "g1_perception_ablation_protocol_v2",
        "h1_role_semantics": H1_ROLE_SEMANTICS,
        "incompatible_role_migration": {
            "rejected_legacy_order": (
                "left_near",
                "left_far",
                "right_near",
                "right_far",
            ),
            "required_order": (
                "left_current_support",
                "left_landing_support",
                "right_current_support",
                "right_landing_support",
            ),
            "automatic_aliasing": False,
        },
        "training_task_registration": "intentionally_deferred",
        "metric_keys": {
            "h1": H1_METRIC_KEYS,
            "h2": H2_METRIC_KEYS,
            "paired_causal": PAIRED_CAUSAL_METRIC_KEYS,
        },
        "h1": {name: spec.receipt() for name, spec in h1.items()},
        "h2": {name: spec.receipt() for name, spec in h2.items()},
    }


__all__ = [
    "H1AblationSpec",
    "H1_BUDGETS",
    "H1_SELECTORS",
    "H1_METRIC_KEYS",
    "H1_ROLE_SEMANTICS",
    "H2AblationSpec",
    "H2_ASSOCIATIONS",
    "H2_GEOMETRIES",
    "H2_HISTORY_REDUCTIONS",
    "H2_TEMPORAL_BASELINES",
    "H2_METRIC_KEYS",
    "PAIRED_CAUSAL_METRIC_KEYS",
    "build_h1_protocol",
    "build_h2_protocol",
    "perception_ablation_receipt",
]
