# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fail-closed administrative-censor labels for offline CTEQ truth.

This module does not inspect actor outputs or environment observations.  It
only adapts already-audited, debounced foot-contact event bins and an explicit
episode-boundary taxonomy into loss/evaluator-only labels.  In particular, an
episode termination is never converted into touchdown or liftoff.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
from typing import Any, ClassVar, Final, Mapping, Tuple

import numpy as np

from .cteq_contact_timing import (
    CTEQ_ALLOWED_TRUTH_CONSUMERS,
    CTEQ_BIN_WIDTH_S,
    CTEQ_NUM_BINS,
    CteqContractError,
    DualEventHazardDistribution,
    EventIndex,
    PrivilegedLabelLeakageError,
)
from .cteq_runner_termination_provenance import (
    CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA,
    CteqOnPolicyTerminationProvenance,
    validate_cteq_runner_termination_provenance_receipt,
)


CTEQ_ADMINISTRATIVE_CENSOR_SCHEMA: Final[str] = (
    "cteq-administrative-censor-label-v1"
)
CTEQ_ADMINISTRATIVE_CENSOR_RECEIPT_SCHEMA: Final[str] = (
    "cteq-administrative-censor-receipt-v1"
)
CTEQ_FOOT_EVENT_SOURCE_CONTRACT: Final[str] = (
    "debounced_foot_contact_transition_only_v1"
)
CTEQ_OBSERVED_BIN_CONTRACT: Final[str] = (
    "complete_post_anchor_bins_before_or_including_valid_terminal_sample_v1"
)
CTEQ_ADMINISTRATIVE_LOSS_SCHEMA: Final[str] = (
    "cteq-administrative-censor-survival-loss-v1"
)


class CteqCensorReason(IntEnum):
    """Stable per-target event/censor reason codes."""

    OBSERVED_TOUCHDOWN = 0
    OBSERVED_LIFTOFF = 1
    NATURAL_HORIZON_RIGHT_CENSOR = 2
    TIME_LIMIT_ADMINISTRATIVE_CENSOR = 3
    BASE_CONTACT_ADMINISTRATIVE_CENSOR = 4
    OTHER_EARLY_TERMINATION_ADMINISTRATIVE_CENSOR = 5


_REASON_NAMES: Final[dict[int, str]] = {
    int(reason): reason.name.lower() for reason in CteqCensorReason
}


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _bool_batch(value: Any, name: str, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (batch_size,) or array.dtype != np.bool_:
        raise CteqContractError(f"{name} must be bool with shape [B].")
    return np.ascontiguousarray(array)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _label_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _receipt_sha256(receipt: Mapping[str, Any]) -> str:
    payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CteqAdministrativeCensorBatch:
    """Batched offline labels with explicit exposure and reason provenance."""

    event_bin: np.ndarray
    event_observed: np.ndarray
    right_censored: np.ndarray
    censor_after_bin: np.ndarray
    loss_eligible: np.ndarray
    reason_code: np.ndarray
    fully_observed_bins: np.ndarray
    episode_done: np.ndarray
    time_limit: np.ndarray
    base_contact_termination: np.ndarray
    other_early_termination: np.ndarray
    reason_counts: Mapping[str, int]
    audit_receipt: Mapping[str, Any]
    schema: str = CTEQ_ADMINISTRATIVE_CENSOR_SCHEMA
    truth_only: bool = True
    allowed_consumers: Tuple[str, ...] = CTEQ_ALLOWED_TRUTH_CONSUMERS
    training_ready: bool = False
    _cteq_privileged_truth_marker: ClassVar[bool] = True

    def for_consumer(self, consumer: str) -> Mapping[str, np.ndarray]:
        """Release per-sample future truth only to loss/evaluator code."""
        if consumer not in self.allowed_consumers:
            raise PrivilegedLabelLeakageError(
                "Administrative-censor truth may only enter loss/evaluator "
                "paths, never actor forward or observations."
            )
        return {
            "event_bin": self.event_bin.copy(),
            "event_observed": self.event_observed.copy(),
            "right_censored": self.right_censored.copy(),
            "censor_after_bin": self.censor_after_bin.copy(),
            "loss_eligible": self.loss_eligible.copy(),
            "reason_code": self.reason_code.copy(),
        }

    def observation_payload(self) -> Mapping[str, np.ndarray]:
        raise PrivilegedLabelLeakageError(
            "Administrative-censor labels cannot become actor observations."
        )

    def receipt(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.audit_receipt))


@dataclass(frozen=True)
class CteqAdministrativeHazardLoss:
    """NumPy loss and denominator audit for administrative censoring."""

    mean_nll: float
    mean_brier: float
    per_target_nll: np.ndarray
    per_target_brier: np.ndarray
    eligible_mask: np.ndarray
    eligible_target_count: int
    excluded_target_count: int
    per_role_eligible_count: np.ndarray
    per_role_nll_sum: np.ndarray
    per_role_brier_sum: np.ndarray
    td_event_nll_sum: float
    lo_event_nll_sum: float
    td_censored_nll_sum: float
    lo_censored_nll_sum: float
    td_event_count: int
    lo_event_count: int
    td_censored_count: int
    lo_censored_count: int
    schema: str = CTEQ_ADMINISTRATIVE_LOSS_SCHEMA
    target_weighting: str = "equal_per_loss_eligible_foot_event_target"
    training_ready: bool = False


def build_cteq_administrative_censor_batch(
    true_event_bin: np.ndarray,
    *,
    fully_observed_bins: np.ndarray,
    episode_done: np.ndarray,
    time_limit: np.ndarray,
    base_contact_termination: np.ndarray,
    other_early_termination: np.ndarray,
    foot_event_source_contract: str = CTEQ_FOOT_EVENT_SOURCE_CONTRACT,
    observed_bin_contract: str = CTEQ_OBSERVED_BIN_CONTRACT,
    runner_termination_provenance_sha256: str | None = None,
    runner_termination_provenance: (
        CteqOnPolicyTerminationProvenance | None
    ) = None,
) -> CteqAdministrativeCensorBatch:
    """Adapt audited event bins and explicit boundaries without guessing.

    ``true_event_bin`` is ``[B,2 feet,2 events]`` and contains only debounced
    foot-contact transitions actually observed before the boundary.  Values
    are ``-1`` when no such event was observed.  ``fully_observed_bins`` is the
    number of complete post-anchor intervals with usable contact truth.  A
    nonterminal row must have the full 25-bin horizon; partial nonterminal
    rollouts are rejected so a rollout cut cannot silently become censoring.
    """
    event_bin = np.asarray(true_event_bin)
    if event_bin.ndim != 3 or event_bin.shape[1:] != (2, 2):
        raise CteqContractError("true_event_bin must have shape [B,2,2].")
    if event_bin.dtype.kind not in "iu":
        raise CteqContractError("true_event_bin must use an integer dtype.")
    event_bin = np.ascontiguousarray(event_bin, dtype=np.int64)
    batch_size = event_bin.shape[0]
    if batch_size < 1:
        raise CteqContractError("Administrative-censor batch cannot be empty.")
    if np.any(event_bin < -1) or np.any(event_bin >= CTEQ_NUM_BINS):
        raise CteqContractError("Event bins must be -1 or lie in [0,24].")
    if foot_event_source_contract != CTEQ_FOOT_EVENT_SOURCE_CONTRACT:
        raise CteqContractError(
            "Foot events require the debounced contact-transition-only source; "
            "termination flags are not contact events."
        )
    if observed_bin_contract != CTEQ_OBSERVED_BIN_CONTRACT:
        raise CteqContractError(
            "The terminal-sample/exposure convention is missing or changed."
        )
    if (
        runner_termination_provenance is not None
        and runner_termination_provenance_sha256 is not None
    ):
        raise CteqContractError(
            "Supply either a validated runner provenance batch or a legacy "
            "unverified hash, not both."
        )

    exposure = np.asarray(fully_observed_bins)
    if exposure.shape != (batch_size,) or exposure.dtype.kind not in "iu":
        raise CteqContractError(
            "fully_observed_bins must be integer with shape [B]."
        )
    exposure = np.ascontiguousarray(exposure, dtype=np.int64)
    if np.any(exposure < 0) or np.any(exposure > CTEQ_NUM_BINS):
        raise CteqContractError("fully_observed_bins must lie in [0,25].")

    done = _bool_batch(episode_done, "episode_done", batch_size)
    timeout = _bool_batch(time_limit, "time_limit", batch_size)
    base_contact = _bool_batch(
        base_contact_termination, "base_contact_termination", batch_size
    )
    other_early = _bool_batch(
        other_early_termination, "other_early_termination", batch_size
    )
    provenance_receipt: dict[str, Any] | None = None
    provenance_authenticated = False
    if runner_termination_provenance is not None:
        if not isinstance(
            runner_termination_provenance,
            CteqOnPolicyTerminationProvenance,
        ):
            raise CteqContractError(
                "runner_termination_provenance has the wrong contract type."
            )
        provenance = runner_termination_provenance
        if not (
            np.array_equal(done, provenance.episode_done)
            and np.array_equal(timeout, provenance.time_limit)
            and np.array_equal(
                base_contact, provenance.base_contact_termination
            )
            and np.array_equal(
                other_early, provenance.other_early_termination
            )
        ):
            raise CteqContractError(
                "Administrative labels and runner termination provenance differ."
            )
        provenance_receipt = provenance.receipt()
        validate_cteq_runner_termination_provenance_receipt(
            provenance_receipt
        )
        runner_termination_provenance_sha256 = provenance_receipt[
            "receipt_sha256"
        ]
        provenance_authenticated = True
    termination_sum = (
        timeout.astype(np.int8)
        + base_contact.astype(np.int8)
        + other_early.astype(np.int8)
    )
    if np.any(termination_sum > 1):
        raise CteqContractError(
            "Time-limit, base-contact, and other termination reasons are "
            "mutually exclusive."
        )
    if not np.array_equal(done, termination_sum == 1):
        raise CteqContractError(
            "episode_done must equal the exhaustive union of explicit "
            "termination reasons."
        )
    if np.any(~done & (exposure != CTEQ_NUM_BINS)):
        raise CteqContractError(
            "Partial nonterminal horizons are not censor labels; wait for the "
            "remaining future truth or supply an explicit episode boundary."
        )
    observed = event_bin >= 0
    exposure_map = exposure[:, None, None]
    if np.any(observed & (event_bin >= exposure_map)):
        raise CteqContractError(
            "A true event bin must be strictly before the audited observation "
            "boundary; future/reset-side events are forbidden."
        )

    censored = ~observed
    censor_after_bin = np.where(censored, exposure_map, -1).astype(np.int64)
    loss_eligible = observed | (censored & (exposure_map > 0))
    reasons = np.empty(event_bin.shape, dtype=np.int8)
    reasons[..., 0] = int(CteqCensorReason.OBSERVED_TOUCHDOWN)
    reasons[..., 1] = int(CteqCensorReason.OBSERVED_LIFTOFF)
    complete_horizon = exposure == CTEQ_NUM_BINS
    for batch_index in range(batch_size):
        if complete_horizon[batch_index]:
            censor_reason = CteqCensorReason.NATURAL_HORIZON_RIGHT_CENSOR
        elif timeout[batch_index]:
            censor_reason = CteqCensorReason.TIME_LIMIT_ADMINISTRATIVE_CENSOR
        elif base_contact[batch_index]:
            censor_reason = CteqCensorReason.BASE_CONTACT_ADMINISTRATIVE_CENSOR
        elif other_early[batch_index]:
            censor_reason = (
                CteqCensorReason.OTHER_EARLY_TERMINATION_ADMINISTRATIVE_CENSOR
            )
        else:  # guarded by the exhaustive boundary checks above
            raise AssertionError("Incomplete horizon has no termination reason.")
        reasons[batch_index][censored[batch_index]] = int(censor_reason)

    reason_counts = {
        _REASON_NAMES[code]: int(np.count_nonzero(reasons == code))
        for code in sorted(_REASON_NAMES)
    }
    reason_counts["zero_exposure_ineligible_target"] = int(
        np.count_nonzero(~loss_eligible)
    )
    reason_counts["episode_time_limit"] = int(np.count_nonzero(timeout))
    reason_counts["episode_base_contact"] = int(np.count_nonzero(base_contact))
    reason_counts["episode_other_early_termination"] = int(
        np.count_nonzero(other_early)
    )

    label_sha = _label_sha256(
        event_bin,
        observed,
        censored,
        censor_after_bin,
        loss_eligible,
        reasons,
        exposure,
        done,
        timeout,
        base_contact,
        other_early,
    )
    provenance_receipted = _is_sha256(
        runner_termination_provenance_sha256
    )
    if (
        runner_termination_provenance_sha256 is not None
        and not provenance_receipted
    ):
        raise CteqContractError(
            "Runner termination provenance must be lowercase SHA-256 when supplied."
        )
    receipt: dict[str, Any] = {
        "schema": CTEQ_ADMINISTRATIVE_CENSOR_RECEIPT_SCHEMA,
        "label_schema": CTEQ_ADMINISTRATIVE_CENSOR_SCHEMA,
        "batch_size": batch_size,
        "target_shape": [batch_size, 2, 2],
        "bin_width_s": CTEQ_BIN_WIDTH_S,
        "num_bins": CTEQ_NUM_BINS,
        "forecast_horizon_s": CTEQ_BIN_WIDTH_S * CTEQ_NUM_BINS,
        "foot_event_source_contract": foot_event_source_contract,
        "observed_bin_contract": observed_bin_contract,
        "reason_codes": {
            str(code): _REASON_NAMES[code] for code in sorted(_REASON_NAMES)
        },
        "reason_counts": dict(reason_counts),
        "label_tensor_sha256": label_sha,
        "runner_termination_provenance_sha256": (
            runner_termination_provenance_sha256
        ),
        "runner_termination_provenance_receipted": provenance_receipted,
        "runner_termination_provenance_contract": (
            CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA
            if provenance_authenticated
            else None
        ),
        "runner_termination_provenance_receipt": provenance_receipt,
        "runner_termination_provenance_authenticated": (
            provenance_authenticated
        ),
        "termination_is_event": False,
        "future_truth_consumers": list(CTEQ_ALLOWED_TRUTH_CONSUMERS),
        "actor_future_truth_access": False,
        "administrative_censor_likelihood": (
            "survival_through_bins_[0,censor_after_bin)"
        ),
        "existing_full_horizon_loss_supports_early_censor": False,
        "administrative_censor_loss_contract": CTEQ_ADMINISTRATIVE_LOSS_SCHEMA,
        "administrative_censor_loss_interface_closed": True,
        "label_adapter_ready": True,
        "loss_interface_closed": True,
        "actor_integrated": False,
        "gym_task_registered": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = _receipt_sha256(receipt)
    return CteqAdministrativeCensorBatch(
        event_bin=_readonly_copy(event_bin, dtype=np.int64),
        event_observed=_readonly_copy(observed, dtype=np.bool_),
        right_censored=_readonly_copy(censored, dtype=np.bool_),
        censor_after_bin=_readonly_copy(censor_after_bin, dtype=np.int64),
        loss_eligible=_readonly_copy(loss_eligible, dtype=np.bool_),
        reason_code=_readonly_copy(reasons, dtype=np.int8),
        fully_observed_bins=_readonly_copy(exposure, dtype=np.int64),
        episode_done=_readonly_copy(done, dtype=np.bool_),
        time_limit=_readonly_copy(timeout, dtype=np.bool_),
        base_contact_termination=_readonly_copy(base_contact, dtype=np.bool_),
        other_early_termination=_readonly_copy(other_early, dtype=np.bool_),
        reason_counts=dict(reason_counts),
        audit_receipt=receipt,
    )


def administrative_censor_survival_loss(
    distribution: DualEventHazardDistribution,
    labels: CteqAdministrativeCensorBatch,
) -> CteqAdministrativeHazardLoss:
    """Compute exact-event or boundary-survival loss over eligible targets.

    For an observed event in bin ``j``, NLL is ``-log(S(j) * h_j)``.  For a
    censored target with ``censor_after_bin=m``, NLL is ``-log(S(m))`` where
    ``S(m)=prod_{k<m}(1-h_k)``.  Censored Brier uses the collapsed categorical
    distribution ``[p_0,...,p_{m-1},S(m)]``.  At ``m=25`` this is identical to
    the legacy full-horizon 26-category Brier score.
    """
    if not isinstance(distribution, DualEventHazardDistribution):
        raise TypeError("distribution must be DualEventHazardDistribution.")
    if not isinstance(labels, CteqAdministrativeCensorBatch):
        raise TypeError("labels must be CteqAdministrativeCensorBatch.")
    expected_shape = labels.event_bin.shape
    if distribution.event_mass.shape != (*expected_shape, CTEQ_NUM_BINS):
        raise CteqContractError(
            "Hazard leading dimensions must equal administrative labels [B,2,2]."
        )
    observed = labels.event_observed
    censored = labels.right_censored
    eligible = labels.loss_eligible
    event_bin = labels.event_bin
    censor_after = labels.censor_after_bin
    if not np.array_equal(censored, ~observed):
        raise CteqContractError("Administrative censor flags must complement events.")
    if np.any(observed & ((event_bin < 0) | (event_bin >= CTEQ_NUM_BINS))):
        raise CteqContractError("Observed events require a bin in [0,24].")
    if np.any(observed & (censor_after != -1)):
        raise CteqContractError("Observed events require censor_after_bin=-1.")
    if np.any(censored & (event_bin != -1)):
        raise CteqContractError("Censored targets require event_bin=-1.")
    if np.any(
        censored
        & ((censor_after < 0) | (censor_after > CTEQ_NUM_BINS))
    ):
        raise CteqContractError("Censored targets require boundary in [0,25].")
    expected_eligible = observed | (censored & (censor_after > 0))
    if not np.array_equal(eligible, expected_eligible):
        raise CteqContractError(
            "loss_eligible must include events and positive-exposure censors only."
        )
    denominator = int(np.count_nonzero(eligible))
    if denominator == 0:
        raise CteqContractError(
            "Administrative-censor loss has zero eligible targets."
        )

    event_index = np.clip(event_bin, 0, CTEQ_NUM_BINS - 1)[..., None]
    selected_event_log_mass = np.take_along_axis(
        distribution.log_event_mass,
        event_index,
        axis=-1,
    )[..., 0]
    boundary_index = np.clip(censor_after, 0, CTEQ_NUM_BINS)[..., None]
    selected_log_survival = np.take_along_axis(
        distribution.log_survival_at_boundary,
        boundary_index,
        axis=-1,
    )[..., 0]
    raw_nll = -np.where(observed, selected_event_log_mass, selected_log_survival)

    full_probability = np.concatenate(
        (distribution.event_mass, distribution.censor_mass[..., None]),
        axis=-1,
    )
    event_target = np.zeros_like(distribution.event_mass)
    np.put_along_axis(event_target, event_index, observed[..., None], axis=-1)
    full_target = np.concatenate(
        (event_target, np.zeros_like(observed[..., None], dtype=np.float64)),
        axis=-1,
    )
    event_brier = np.sum(np.square(full_probability - full_target), axis=-1)

    prefix_mask = (
        np.arange(CTEQ_NUM_BINS, dtype=np.int64)
        < censor_after[..., None]
    )
    prefix_event_mass = distribution.event_mass * prefix_mask
    selected_survival = np.take_along_axis(
        distribution.survival_at_boundary,
        boundary_index,
        axis=-1,
    )[..., 0]
    censor_brier = np.sum(np.square(prefix_event_mass), axis=-1) + np.square(
        selected_survival - 1.0
    )
    raw_brier = np.where(observed, event_brier, censor_brier)
    per_target_nll = np.where(eligible, raw_nll, 0.0)
    per_target_brier = np.where(eligible, raw_brier, 0.0)
    if not np.isfinite(per_target_nll).all() or not np.isfinite(
        per_target_brier
    ).all():
        raise CteqContractError("Administrative CTEQ loss must remain finite.")

    td_event = eligible[..., EventIndex.TOUCHDOWN] & observed[..., EventIndex.TOUCHDOWN]
    lo_event = eligible[..., EventIndex.LIFTOFF] & observed[..., EventIndex.LIFTOFF]
    td_censored = eligible[..., EventIndex.TOUCHDOWN] & censored[..., EventIndex.TOUCHDOWN]
    lo_censored = eligible[..., EventIndex.LIFTOFF] & censored[..., EventIndex.LIFTOFF]
    return CteqAdministrativeHazardLoss(
        mean_nll=float(np.sum(per_target_nll) / denominator),
        mean_brier=float(np.sum(per_target_brier) / denominator),
        per_target_nll=_readonly_copy(per_target_nll, dtype=np.float64),
        per_target_brier=_readonly_copy(per_target_brier, dtype=np.float64),
        eligible_mask=_readonly_copy(eligible, dtype=np.bool_),
        eligible_target_count=denominator,
        excluded_target_count=int(eligible.size - denominator),
        per_role_eligible_count=_readonly_copy(
            np.count_nonzero(eligible, axis=0), dtype=np.int64
        ),
        per_role_nll_sum=_readonly_copy(
            np.sum(per_target_nll, axis=0), dtype=np.float64
        ),
        per_role_brier_sum=_readonly_copy(
            np.sum(per_target_brier, axis=0), dtype=np.float64
        ),
        td_event_nll_sum=float(
            np.sum(per_target_nll[..., EventIndex.TOUCHDOWN][td_event])
        ),
        lo_event_nll_sum=float(
            np.sum(per_target_nll[..., EventIndex.LIFTOFF][lo_event])
        ),
        td_censored_nll_sum=float(
            np.sum(per_target_nll[..., EventIndex.TOUCHDOWN][td_censored])
        ),
        lo_censored_nll_sum=float(
            np.sum(per_target_nll[..., EventIndex.LIFTOFF][lo_censored])
        ),
        td_event_count=int(np.count_nonzero(td_event)),
        lo_event_count=int(np.count_nonzero(lo_event)),
        td_censored_count=int(np.count_nonzero(td_censored)),
        lo_censored_count=int(np.count_nonzero(lo_censored)),
    )


def cteq_administrative_loss_status() -> Mapping[str, Any]:
    """Return the math/interface receipt while real runner truth stays unbound."""
    return {
        "schema": "cteq-administrative-loss-status-v1",
        "loss_schema": CTEQ_ADMINISTRATIVE_LOSS_SCHEMA,
        "event_nll": "-log_hazard[j]-sum_log_one_minus_hazard[k<j]",
        "censor_nll": "-sum_log_one_minus_hazard[k<censor_after_bin]",
        "censor_brier": "collapsed_prefix_events_plus_survival_at_boundary",
        "target_weighting": "equal_per_loss_eligible_foot_event_target",
        "denominator": "count_nonzero(loss_eligible)",
        "zero_exposure_excluded": True,
        "numpy_torch_interfaces_closed": True,
        "future_truth_consumers": list(CTEQ_ALLOWED_TRUTH_CONSUMERS),
        "actor_integrated": False,
        "gym_task_registered": False,
        "runner_termination_provenance_authenticated": False,
        "training_ready": False,
    }


def validate_cteq_administrative_censor_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Validate the immutable contract fields and receipt digest."""
    if not isinstance(receipt, Mapping):
        raise CteqContractError("CTEQ censor receipt must be a mapping.")
    if receipt.get("schema") != CTEQ_ADMINISTRATIVE_CENSOR_RECEIPT_SCHEMA:
        raise CteqContractError("CTEQ censor receipt schema changed.")
    if receipt.get("label_schema") != CTEQ_ADMINISTRATIVE_CENSOR_SCHEMA:
        raise CteqContractError("CTEQ administrative label schema changed.")
    if receipt.get("reason_codes") != {
        str(code): _REASON_NAMES[code] for code in sorted(_REASON_NAMES)
    }:
        raise CteqContractError("CTEQ censor reason-code table changed.")
    batch_size = receipt.get("batch_size")
    if type(batch_size) is not int or batch_size < 1:
        raise CteqContractError("CTEQ censor receipt batch_size is invalid.")
    if receipt.get("target_shape") != [batch_size, 2, 2]:
        raise CteqContractError("CTEQ censor receipt target shape changed.")
    if receipt.get("bin_width_s") != CTEQ_BIN_WIDTH_S:
        raise CteqContractError("CTEQ censor bin width changed.")
    if receipt.get("num_bins") != CTEQ_NUM_BINS:
        raise CteqContractError("CTEQ censor bin count changed.")
    if receipt.get("foot_event_source_contract") != (
        CTEQ_FOOT_EVENT_SOURCE_CONTRACT
    ):
        raise CteqContractError("CTEQ foot-event source contract changed.")
    if receipt.get("observed_bin_contract") != CTEQ_OBSERVED_BIN_CONTRACT:
        raise CteqContractError("CTEQ observed-bin contract changed.")
    if receipt.get("future_truth_consumers") != list(
        CTEQ_ALLOWED_TRUTH_CONSUMERS
    ):
        raise CteqContractError("CTEQ future-truth consumers changed.")
    if receipt.get("administrative_censor_likelihood") != (
        "survival_through_bins_[0,censor_after_bin)"
    ):
        raise CteqContractError("CTEQ censor likelihood semantics changed.")
    for field, expected in (
        ("termination_is_event", False),
        ("actor_future_truth_access", False),
        ("existing_full_horizon_loss_supports_early_censor", False),
        ("administrative_censor_loss_interface_closed", True),
        ("label_adapter_ready", True),
        ("loss_interface_closed", True),
        ("actor_integrated", False),
        ("gym_task_registered", False),
        ("training_ready", False),
    ):
        if receipt.get(field) is not expected:
            raise CteqContractError(f"CTEQ censor receipt field {field} changed.")
    if receipt.get("administrative_censor_loss_contract") != (
        CTEQ_ADMINISTRATIVE_LOSS_SCHEMA
    ):
        raise CteqContractError("CTEQ administrative loss contract changed.")
    receipt_sha = receipt.get("receipt_sha256")
    if not _is_sha256(receipt_sha):
        raise CteqContractError("CTEQ censor receipt requires a SHA-256 digest.")
    unhashed = dict(receipt)
    del unhashed["receipt_sha256"]
    if receipt_sha != _receipt_sha256(unhashed):
        raise CteqContractError("CTEQ censor receipt digest mismatch.")
    provenance = receipt.get("runner_termination_provenance_sha256")
    if provenance is not None and not _is_sha256(provenance):
        raise CteqContractError(
            "Runner termination provenance must be lowercase SHA-256."
        )
    if receipt.get("runner_termination_provenance_receipted") is not (
        _is_sha256(provenance)
    ):
        raise CteqContractError("Runner termination receipt hash is inconsistent.")
    nested_provenance = receipt.get("runner_termination_provenance_receipt")
    provenance_contract = receipt.get("runner_termination_provenance_contract")
    provenance_authenticated = receipt.get(
        "runner_termination_provenance_authenticated"
    )
    if nested_provenance is None:
        if provenance_contract is not None or provenance_authenticated is not False:
            raise CteqContractError(
                "Unbound runner provenance cannot be authenticated."
            )
    else:
        validate_cteq_runner_termination_provenance_receipt(nested_provenance)
        if provenance_contract != CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA:
            raise CteqContractError("Runner provenance contract changed.")
        if provenance_authenticated is not True:
            raise CteqContractError("Validated runner provenance lost authentication.")
        if provenance != nested_provenance.get("receipt_sha256"):
            raise CteqContractError("Nested runner provenance digest differs.")
    counts = receipt.get("reason_counts")
    expected_count_keys = {
        *_REASON_NAMES.values(),
        "zero_exposure_ineligible_target",
        "episode_time_limit",
        "episode_base_contact",
        "episode_other_early_termination",
    }
    if (
        not isinstance(counts, Mapping)
        or set(counts) != expected_count_keys
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise CteqContractError("CTEQ censor reason counts are invalid.")
    if sum(counts[name] for name in _REASON_NAMES.values()) != batch_size * 4:
        raise CteqContractError("CTEQ per-target reason counts are incomplete.")
    if nested_provenance is not None:
        nested_counts = nested_provenance["counts"]
        if nested_provenance["batch_size"] != batch_size:
            raise CteqContractError("Nested runner provenance batch size differs.")
        if (
            nested_counts["time_limit"] != counts["episode_time_limit"]
            or nested_counts["base_contact"] != counts["episode_base_contact"]
            or nested_counts["other_termination"]
            != counts["episode_other_early_termination"]
        ):
            raise CteqContractError(
                "Nested runner termination counts differ from label reasons."
            )
    if not _is_sha256(receipt.get("label_tensor_sha256")):
        raise CteqContractError("CTEQ label tensor digest is invalid.")
    horizon = receipt.get("forecast_horizon_s")
    if not isinstance(horizon, (int, float)) or not math.isclose(
        float(horizon), CTEQ_BIN_WIDTH_S * CTEQ_NUM_BINS
    ):
        raise CteqContractError("CTEQ censor forecast horizon changed.")


__all__ = [
    "CTEQ_ADMINISTRATIVE_LOSS_SCHEMA",
    "CTEQ_ADMINISTRATIVE_CENSOR_RECEIPT_SCHEMA",
    "CTEQ_ADMINISTRATIVE_CENSOR_SCHEMA",
    "CTEQ_FOOT_EVENT_SOURCE_CONTRACT",
    "CTEQ_OBSERVED_BIN_CONTRACT",
    "CteqAdministrativeCensorBatch",
    "CteqAdministrativeHazardLoss",
    "CteqCensorReason",
    "administrative_censor_survival_loss",
    "build_cteq_administrative_censor_batch",
    "cteq_administrative_loss_status",
    "validate_cteq_administrative_censor_receipt",
]
