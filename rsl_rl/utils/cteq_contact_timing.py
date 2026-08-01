# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-first CTEQ PR-0/PR-1 contact-timing primitives.

This module deliberately has no actor, Gym, Isaac Sim, or CUDA integration.
Future contact truth is represented by :class:`IndependentEventLabels` and can
only be released to a loss or evaluator consumer.  Runtime role weights are
computed exclusively from causal hazard predictions and a separately supplied
current contact state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


CTEQ_BIN_WIDTH_S = 0.020
CTEQ_NUM_BINS = 25
CTEQ_FORECAST_HORIZON_S = CTEQ_BIN_WIDTH_S * CTEQ_NUM_BINS
CTEQ_LABEL_SCHEMA = "cteq-independent-td-lo-label-v1"
CTEQ_HAZARD_SCHEMA = "cteq-dual-event-hazard-v1"
CTEQ_DIAGNOSTIC_SCHEMA = "cteq-pr01-diagnostics-v1"
CTEQ_ALLOWED_TRUTH_CONSUMERS = ("loss", "evaluator")


class CteqContractError(ValueError):
    """Raised when CTEQ data would violate the frozen causal contract."""


class PrivilegedLabelLeakageError(CteqContractError):
    """Raised when future-contact truth is requested by an observation path."""


class FootIndex(IntEnum):
    LEFT = 0
    RIGHT = 1


class EventIndex(IntEnum):
    TOUCHDOWN = 0
    LIFTOFF = 1


@dataclass(frozen=True)
class ContactEvent:
    foot_index: int
    event_index: int
    sample_index: int
    time_s: float

    @property
    def foot_name(self) -> str:
        return ("left", "right")[self.foot_index]

    @property
    def event_name(self) -> str:
        return ("touchdown", "liftoff")[self.event_index]


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _contact_trace(value: Any, name: str) -> np.ndarray:
    trace = np.asarray(value)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise CteqContractError(f"{name} must have shape [T, 2 feet].")
    if trace.shape[0] < 2:
        raise CteqContractError(f"{name} must contain at least two samples.")
    if trace.dtype != np.bool_:
        raise CteqContractError(f"{name} must use boolean dtype.")
    return np.ascontiguousarray(trace)


@dataclass(frozen=True)
class StableContactTrace:
    raw_contact: np.ndarray
    stable_contact: np.ndarray
    events: Tuple[ContactEvent, ...]
    sample_period_s: float
    min_stable_steps: int
    truth_only: bool = True
    training_ready: bool = False


def debounce_contact_trace(
    raw_contact: np.ndarray,
    *,
    sample_period_s: float,
    min_stable_steps: int,
) -> StableContactTrace:
    """Remove short contact glitches and extract stable TD/LO transitions.

    A new state is accepted only after ``min_stable_steps`` consecutive raw
    samples.  Because this is an offline truth builder, the accepted transition
    is backdated to the first sample of that sustained run rather than shifted
    to the confirmation sample.
    """

    raw = _contact_trace(raw_contact, "raw_contact")
    if not math.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise CteqContractError("sample_period_s must be positive and finite.")
    if type(min_stable_steps) is not int or min_stable_steps < 1:
        raise CteqContractError("min_stable_steps must be a positive integer.")
    if raw.shape[0] < min_stable_steps:
        raise CteqContractError(
            "Contact trace must include at least min_stable_steps of initial pre-roll."
        )
    initial_window = raw[:min_stable_steps]
    if np.any(initial_window != initial_window[:1]):
        raise CteqContractError(
            "Initial contact state requires min_stable_steps of stable pre-roll; "
            "do not turn an episode-boundary glitch into TD/LO truth."
        )

    stable = np.empty_like(raw)
    for foot in range(2):
        state = bool(raw[0, foot])
        stable[:, foot] = state
        candidate_state: Optional[bool] = None
        candidate_start = -1
        candidate_count = 0
        for sample in range(1, raw.shape[0]):
            observed = bool(raw[sample, foot])
            stable[sample, foot] = state
            if observed == state:
                candidate_state = None
                candidate_start = -1
                candidate_count = 0
                continue
            if candidate_state is None or observed != candidate_state:
                candidate_state = observed
                candidate_start = sample
                candidate_count = 1
            else:
                candidate_count += 1
            if candidate_count >= min_stable_steps:
                state = bool(candidate_state)
                stable[candidate_start : sample + 1, foot] = state
                candidate_state = None
                candidate_start = -1
                candidate_count = 0

    events = []
    for sample in range(1, stable.shape[0]):
        for foot in range(2):
            previous = bool(stable[sample - 1, foot])
            current = bool(stable[sample, foot])
            if current == previous:
                continue
            event_index = (
                EventIndex.TOUCHDOWN if current else EventIndex.LIFTOFF
            )
            events.append(
                ContactEvent(
                    foot_index=foot,
                    event_index=int(event_index),
                    sample_index=sample,
                    time_s=sample * sample_period_s,
                )
            )
    events.sort(key=lambda event: (event.sample_index, event.foot_index))
    return StableContactTrace(
        raw_contact=_readonly_copy(raw, dtype=np.bool_),
        stable_contact=_readonly_copy(stable, dtype=np.bool_),
        events=tuple(events),
        sample_period_s=float(sample_period_s),
        min_stable_steps=min_stable_steps,
    )


@dataclass(frozen=True)
class IndependentEventLabels:
    anchor_indices: np.ndarray
    contact_state_truth: np.ndarray
    event_bin: np.ndarray
    event_observed: np.ndarray
    right_censored: np.ndarray
    time_to_event_s: np.ndarray
    bin_width_s: float = CTEQ_BIN_WIDTH_S
    num_bins: int = CTEQ_NUM_BINS
    schema: str = CTEQ_LABEL_SCHEMA
    truth_only: bool = True
    allowed_consumers: Tuple[str, ...] = CTEQ_ALLOWED_TRUTH_CONSUMERS
    training_ready: bool = False

    def for_consumer(self, consumer: str) -> Mapping[str, np.ndarray]:
        """Release copies only to the auxiliary loss or offline evaluator."""

        if consumer not in self.allowed_consumers:
            raise PrivilegedLabelLeakageError(
                "Future TD/LO labels are privileged truth and may only enter "
                "loss/evaluator paths, never actor observations."
            )
        return {
            "anchor_indices": self.anchor_indices.copy(),
            "contact_state_truth": self.contact_state_truth.copy(),
            "event_bin": self.event_bin.copy(),
            "event_observed": self.event_observed.copy(),
            "right_censored": self.right_censored.copy(),
            "time_to_event_s": self.time_to_event_s.copy(),
        }

    def observation_payload(self) -> Mapping[str, np.ndarray]:
        raise PrivilegedLabelLeakageError(
            "Future contact labels cannot be converted to an observation."
        )


def _event_bin(delay_s: float, *, bin_width_s: float, num_bins: int) -> int:
    ratio = delay_s / bin_width_s
    index = int(math.ceil(ratio - 1.0e-12)) - 1
    return min(num_bins - 1, max(0, index))


def build_independent_event_labels(
    trace: StableContactTrace,
    *,
    anchor_indices: Optional[Sequence[int]] = None,
    bin_width_s: float = CTEQ_BIN_WIDTH_S,
    num_bins: int = CTEQ_NUM_BINS,
) -> IndependentEventLabels:
    """Build independent next-TD and next-LO labels for both feet.

    Events are strictly future relative to the anchor.  The horizon is right
    closed: delays in ``(0, H * Delta]`` are observed, while absence of that
    event type in the complete horizon is a right-censored label.  Trace-end
    truncation is rejected instead of being mislabeled as horizon censoring.
    """

    if not isinstance(trace, StableContactTrace):
        raise TypeError("trace must be StableContactTrace.")
    if not math.isfinite(bin_width_s) or bin_width_s <= 0.0:
        raise CteqContractError("bin_width_s must be positive and finite.")
    if type(num_bins) is not int or num_bins < 1:
        raise CteqContractError("num_bins must be a positive integer.")
    horizon_s = bin_width_s * num_bins
    final_time_s = (trace.stable_contact.shape[0] - 1) * trace.sample_period_s
    if anchor_indices is None:
        anchors = np.asarray(
            [
                index
                for index in range(trace.stable_contact.shape[0])
                if index * trace.sample_period_s + horizon_s
                <= final_time_s + 1.0e-12
            ],
            dtype=np.int64,
        )
    else:
        anchors = np.asarray(anchor_indices)
        if anchors.ndim != 1 or anchors.dtype.kind not in "iu":
            raise CteqContractError("anchor_indices must be a 1-D integer sequence.")
        anchors = np.ascontiguousarray(anchors, dtype=np.int64)
    if anchors.size == 0:
        raise CteqContractError("No anchor has a complete forecast horizon.")
    if len(set(int(value) for value in anchors)) != anchors.size:
        raise CteqContractError("anchor_indices must be unique.")
    if np.any(anchors < 0) or np.any(anchors >= trace.stable_contact.shape[0]):
        raise CteqContractError("anchor_indices contain an out-of-range sample.")
    anchor_times = anchors.astype(np.float64) * trace.sample_period_s
    if np.any(anchor_times + horizon_s > final_time_s + 1.0e-12):
        raise CteqContractError(
            "Every anchor must have a complete horizon; early episode "
            "termination needs a separate administrative-censor contract."
        )

    shape = (anchors.size, 2, 2)
    event_bin = np.full(shape, -1, dtype=np.int64)
    observed = np.zeros(shape, dtype=np.bool_)
    time_to_event = np.full(shape, np.nan, dtype=np.float64)
    events_by_type = {
        (foot, event): [] for foot in range(2) for event in range(2)
    }
    for event in trace.events:
        events_by_type[(event.foot_index, event.event_index)].append(event)

    for anchor_position, (anchor, anchor_time) in enumerate(
        zip(anchors, anchor_times)
    ):
        for foot in range(2):
            for event_index in range(2):
                next_event = next(
                    (
                        event
                        for event in events_by_type[(foot, event_index)]
                        if event.sample_index > int(anchor)
                    ),
                    None,
                )
                if next_event is None:
                    continue
                delay = next_event.time_s - float(anchor_time)
                if delay <= 0.0:
                    raise CteqContractError("Future event delay must be positive.")
                if delay <= horizon_s + 1.0e-12:
                    observed[anchor_position, foot, event_index] = True
                    event_bin[anchor_position, foot, event_index] = _event_bin(
                        delay,
                        bin_width_s=bin_width_s,
                        num_bins=num_bins,
                    )
                    time_to_event[anchor_position, foot, event_index] = delay
    censored = ~observed
    if np.any(observed & censored):
        raise AssertionError("Observed and censored labels must be disjoint.")
    return IndependentEventLabels(
        anchor_indices=_readonly_copy(anchors, dtype=np.int64),
        contact_state_truth=_readonly_copy(
            trace.stable_contact[anchors], dtype=np.bool_
        ),
        event_bin=_readonly_copy(event_bin, dtype=np.int64),
        event_observed=_readonly_copy(observed, dtype=np.bool_),
        right_censored=_readonly_copy(censored, dtype=np.bool_),
        time_to_event_s=_readonly_copy(time_to_event, dtype=np.float64),
        bin_width_s=float(bin_width_s),
        num_bins=num_bins,
    )


def _hazard_shape(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 3 or array.shape[-3:] != (2, 2, CTEQ_NUM_BINS):
        raise CteqContractError(
            f"{name} must end with [2 feet, 2 independent events, 25 bins]."
        )
    if array.dtype.kind not in "fc":
        raise CteqContractError(f"{name} must use a floating dtype.")
    if not np.isfinite(array).all():
        raise CteqContractError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass(frozen=True)
class DualEventHazardDistribution:
    hazard: np.ndarray
    survival_before: np.ndarray
    event_mass: np.ndarray
    censor_mass: np.ndarray
    log_event_mass: np.ndarray
    log_censor_mass: np.ndarray
    schema: str = CTEQ_HAZARD_SCHEMA
    independent_td_lo: bool = True
    training_ready: bool = False


def _distribution_from_log_hazards(
    hazard: np.ndarray,
    log_hazard: np.ndarray,
    log_one_minus_hazard: np.ndarray,
) -> DualEventHazardDistribution:
    log_survival_before = np.concatenate(
        (
            np.zeros_like(log_one_minus_hazard[..., :1]),
            np.cumsum(log_one_minus_hazard[..., :-1], axis=-1),
        ),
        axis=-1,
    )
    log_event_mass = log_survival_before + log_hazard
    log_censor_mass = np.sum(log_one_minus_hazard, axis=-1)
    survival = np.exp(log_survival_before)
    event_mass = np.exp(log_event_mass)
    censor_mass = np.exp(log_censor_mass)
    total = np.sum(event_mass, axis=-1) + censor_mass
    if not np.allclose(total, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise CteqContractError("Each TD/LO survival distribution must sum to one.")
    return DualEventHazardDistribution(
        hazard=_readonly_copy(hazard, dtype=np.float64),
        survival_before=_readonly_copy(survival, dtype=np.float64),
        event_mass=_readonly_copy(event_mass, dtype=np.float64),
        censor_mass=_readonly_copy(censor_mass, dtype=np.float64),
        log_event_mass=_readonly_copy(log_event_mass, dtype=np.float64),
        log_censor_mass=_readonly_copy(log_censor_mass, dtype=np.float64),
    )


def dual_event_hazard_from_logits(logits: np.ndarray) -> DualEventHazardDistribution:
    """Convert ``[..., left/right, TD/LO, 25]`` logits stably on CPU."""

    values = _hazard_shape(logits, "logits")
    hazard = np.empty_like(values)
    positive = values >= 0.0
    hazard[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    hazard[~positive] = exp_values / (1.0 + exp_values)
    log_hazard = -np.logaddexp(0.0, -values)
    log_one_minus_hazard = -np.logaddexp(0.0, values)
    return _distribution_from_log_hazards(
        hazard,
        log_hazard,
        log_one_minus_hazard,
    )


def dual_event_hazard_from_probabilities(
    hazards: np.ndarray,
) -> DualEventHazardDistribution:
    values = _hazard_shape(hazards, "hazards")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise CteqContractError("hazards must lie in [0, 1].")
    tiny = np.finfo(np.float64).tiny
    log_hazard = np.log(np.clip(values, tiny, 1.0))
    log_one_minus = np.log(np.clip(1.0 - values, tiny, 1.0))
    return _distribution_from_log_hazards(
        values,
        log_hazard,
        log_one_minus,
    )


def _label_shape(
    distribution: DualEventHazardDistribution,
    labels: IndependentEventLabels,
) -> Tuple[int, ...]:
    prefix = distribution.event_mass.shape[:-1]
    if prefix != labels.event_bin.shape:
        raise CteqContractError(
            "Hazard leading dimensions must equal label [anchors, feet, events]."
        )
    if labels.num_bins != CTEQ_NUM_BINS or labels.bin_width_s != CTEQ_BIN_WIDTH_S:
        raise CteqContractError("Labels do not use the frozen 20 ms x 25 contract.")
    if not np.array_equal(labels.right_censored, ~labels.event_observed):
        raise CteqContractError("TD/LO censor flags must be independent complements.")
    return prefix


@dataclass(frozen=True)
class HazardLoss:
    mean_nll: float
    mean_brier: float
    per_target_nll: np.ndarray
    per_target_brier: np.ndarray
    td_event_nll_sum: float
    lo_event_nll_sum: float
    td_censored_nll_sum: float
    lo_censored_nll_sum: float
    td_event_count: int
    lo_event_count: int
    td_censored_count: int
    lo_censored_count: int
    training_ready: bool = False


def dual_event_survival_loss(
    distribution: DualEventHazardDistribution,
    labels: IndependentEventLabels,
) -> HazardLoss:
    """Return independent TD/LO survival NLL and multiclass Brier score."""

    _label_shape(distribution, labels)
    observed = labels.event_observed
    censored = labels.right_censored
    indices = np.clip(labels.event_bin, 0, CTEQ_NUM_BINS - 1)[..., None]
    selected_log_mass = np.take_along_axis(
        distribution.log_event_mass,
        indices,
        axis=-1,
    )[..., 0]
    nll = np.where(observed, -selected_log_mass, -distribution.log_censor_mass)
    if not np.isfinite(nll).all():
        raise CteqContractError("Survival NLL must remain finite.")

    probabilities = np.concatenate(
        (distribution.event_mass, distribution.censor_mass[..., None]),
        axis=-1,
    )
    targets = np.zeros_like(probabilities)
    np.put_along_axis(targets, indices, observed[..., None], axis=-1)
    targets[..., -1] = censored
    brier = np.sum(np.square(probabilities - targets), axis=-1)

    td_observed = observed[..., EventIndex.TOUCHDOWN]
    lo_observed = observed[..., EventIndex.LIFTOFF]
    td_censored = censored[..., EventIndex.TOUCHDOWN]
    lo_censored = censored[..., EventIndex.LIFTOFF]
    return HazardLoss(
        mean_nll=float(np.mean(nll)),
        mean_brier=float(np.mean(brier)),
        per_target_nll=_readonly_copy(nll, dtype=np.float64),
        per_target_brier=_readonly_copy(brier, dtype=np.float64),
        td_event_nll_sum=float(np.sum(nll[..., EventIndex.TOUCHDOWN][td_observed])),
        lo_event_nll_sum=float(np.sum(nll[..., EventIndex.LIFTOFF][lo_observed])),
        td_censored_nll_sum=float(np.sum(nll[..., EventIndex.TOUCHDOWN][td_censored])),
        lo_censored_nll_sum=float(np.sum(nll[..., EventIndex.LIFTOFF][lo_censored])),
        td_event_count=int(np.count_nonzero(td_observed)),
        lo_event_count=int(np.count_nonzero(lo_observed)),
        td_censored_count=int(np.count_nonzero(td_censored)),
        lo_censored_count=int(np.count_nonzero(lo_censored)),
    )


@dataclass(frozen=True)
class RoleTimeWeights:
    current: np.ndarray
    landing: np.ndarray
    training_ready: bool = False


def cteq_role_time_weights(
    distribution: DualEventHazardDistribution,
    contact_state_now: np.ndarray,
) -> RoleTimeWeights:
    """Compute ``current=c_now*S_LO`` and ``landing=pi_TD`` weights."""

    contact = np.asarray(contact_state_now)
    expected_shape = distribution.event_mass.shape[:-3] + (2,)
    if contact.shape != expected_shape or contact.dtype != np.bool_:
        raise CteqContractError(
            "contact_state_now must be causal bool data with shape [..., 2 feet]."
        )
    landing = distribution.event_mass[..., EventIndex.TOUCHDOWN, :]
    liftoff_survival = distribution.survival_before[..., EventIndex.LIFTOFF, :]
    current = contact[..., :, None].astype(np.float64) * liftoff_survival
    return RoleTimeWeights(
        current=_readonly_copy(current, dtype=np.float64),
        landing=_readonly_copy(landing, dtype=np.float64),
    )


def _order_violation_probability(
    distribution: DualEventHazardDistribution,
    contact_state: np.ndarray,
) -> np.ndarray:
    masses = np.concatenate(
        (distribution.event_mass, distribution.censor_mass[..., None]),
        axis=-1,
    )
    td = masses[..., EventIndex.TOUCHDOWN, :]
    lo = masses[..., EventIndex.LIFTOFF, :]
    indices = np.arange(CTEQ_NUM_BINS + 1)
    lo_later_than_td = indices[:, None] > indices[None, :]
    td_later_than_lo = indices[:, None] > indices[None, :]
    # First matrix axes are LO,TD; second are TD,LO.
    contact_violation = np.einsum(
        "...i,...j,ij->...",
        lo,
        td,
        lo_later_than_td,
    )
    flight_violation = np.einsum(
        "...i,...j,ij->...",
        td,
        lo,
        td_later_than_lo,
    )
    return np.where(contact_state, contact_violation, flight_violation)


def hazard_calibration_diagnostics(
    distribution: DualEventHazardDistribution,
    labels: IndependentEventLabels,
) -> Mapping[str, Any]:
    """Emit order, calibration, censor, and TTC diagnostics for evaluators."""

    _label_shape(distribution, labels)
    loss = dual_event_survival_loss(distribution, labels)
    within_horizon_probability = 1.0 - distribution.censor_mass
    conditional_mass = np.sum(distribution.event_mass, axis=-1)
    bin_times = (np.arange(CTEQ_NUM_BINS, dtype=np.float64) + 1.0) * CTEQ_BIN_WIDTH_S
    expected_ttc = np.divide(
        np.sum(distribution.event_mass * bin_times, axis=-1),
        conditional_mass,
        out=np.full_like(conditional_mass, np.nan),
        where=conditional_mass > 0.0,
    )
    observed_errors = np.abs(expected_ttc - labels.time_to_event_s)
    current_contact = labels.contact_state_truth
    order_probability = _order_violation_probability(
        distribution,
        current_contact,
    )
    both_observed = np.all(labels.event_observed, axis=-1)
    td_bin = labels.event_bin[..., EventIndex.TOUCHDOWN]
    lo_bin = labels.event_bin[..., EventIndex.LIFTOFF]
    actual_violation = np.where(
        current_contact,
        lo_bin > td_bin,
        td_bin > lo_bin,
    ) & both_observed
    observed_error_values = observed_errors[labels.event_observed]
    return {
        "schema": CTEQ_DIAGNOSTIC_SCHEMA,
        "training_ready": False,
        "independent_td_lo": True,
        "mean_nll": loss.mean_nll,
        "mean_brier": loss.mean_brier,
        "td_event_nll_sum": loss.td_event_nll_sum,
        "lo_event_nll_sum": loss.lo_event_nll_sum,
        "td_censored_nll_sum": loss.td_censored_nll_sum,
        "lo_censored_nll_sum": loss.lo_censored_nll_sum,
        "td_event_count": loss.td_event_count,
        "lo_event_count": loss.lo_event_count,
        "td_censored_count": loss.td_censored_count,
        "lo_censored_count": loss.lo_censored_count,
        "within_horizon_probability": within_horizon_probability.copy(),
        "censor_probability": distribution.censor_mass.copy(),
        "event_observed": labels.event_observed.copy(),
        "right_censored": labels.right_censored.copy(),
        "expected_ttc_conditional_s": expected_ttc,
        "observed_ttc_mae_s": (
            float(np.mean(observed_error_values))
            if observed_error_values.size
            else None
        ),
        "order_violation_probability": order_probability,
        "mean_order_violation_probability": float(np.mean(order_probability)),
        "actual_order_comparable_count": int(np.count_nonzero(both_observed)),
        "actual_order_violation_count": int(np.count_nonzero(actual_violation)),
        "reliability_input": {
            "td_predicted_within_horizon": within_horizon_probability[
                ..., EventIndex.TOUCHDOWN
            ].copy(),
            "td_observed_within_horizon": labels.event_observed[
                ..., EventIndex.TOUCHDOWN
            ].copy(),
            "lo_predicted_within_horizon": within_horizon_probability[
                ..., EventIndex.LIFTOFF
            ].copy(),
            "lo_observed_within_horizon": labels.event_observed[
                ..., EventIndex.LIFTOFF
            ].copy(),
        },
    }


_FORBIDDEN_OBSERVATION_KEY_PARTS = (
    "administrative_censor",
    "censor_after_bin",
    "censor_reason",
    "future_contact",
    "future_event",
    "event_bin_target",
    "right_censored",
    "time_to_event_truth",
    "touchdown_label",
    "liftoff_label",
)


def validate_causal_observation(observation: Any, *, path: str = "observation") -> None:
    """Reject future-label objects or reserved truth keys recursively."""

    if isinstance(observation, (IndependentEventLabels, StableContactTrace, ContactEvent)) or (
        getattr(observation, "_cteq_privileged_truth_marker", False) is True
    ):
        raise PrivilegedLabelLeakageError(
            f"{path} contains simulation future-contact truth."
        )
    if isinstance(observation, Mapping):
        for key, value in observation.items():
            if not isinstance(key, str):
                raise CteqContractError(f"{path} keys must be strings.")
            normalized = key.lower()
            if any(part in normalized for part in _FORBIDDEN_OBSERVATION_KEY_PARTS):
                raise PrivilegedLabelLeakageError(
                    f"{path}.{key} is reserved for loss/evaluator truth."
                )
            validate_causal_observation(value, path=f"{path}.{key}")
    elif isinstance(observation, (list, tuple)):
        for index, value in enumerate(observation):
            validate_causal_observation(value, path=f"{path}[{index}]")


def cteq_pr01_status() -> Mapping[str, Any]:
    """Return the explicit implementation boundary for receipts and callers."""

    return {
        "schema": "cteq-pr01-status-v1",
        "implemented": (
            "stable_contact_trace",
            "independent_td_lo_labels",
            "dual_event_survival_distribution",
            "survival_nll",
            "multiclass_brier",
            "role_time_weights",
            "order_and_calibration_diagnostics",
            "torch_dual_event_hazard_head",
            "torch_survival_loss",
            "administrative_censor_label_adapter",
            "termination_reason_audit_receipt",
        ),
        "training_ready": False,
        "actor_integrated": False,
        "gym_task_registered": False,
        "gpu_required": False,
        "future_truth_allowed_consumers": CTEQ_ALLOWED_TRUTH_CONSUMERS,
        "blocked_next_steps": (
            "real_runner_termination_and_terminal_sample_provenance",
            "administrative_censor_aware_survival_loss",
            "actor_query_integration",
            "ppo_training",
            "export_resume_validation",
        ),
    }
