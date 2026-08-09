# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fail-closed bridge from opt-in IsaacLab extras to CTEQ label truth.

This module is not called by the on-policy runner.  A label collector must opt
in explicitly, pass the returned ``dones``/``extras``, and (for CUDA tensors)
authorize the visible device-to-CPU copy.  The resulting object rejects actor,
critic, and reward access and remains ``training_ready=False``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, ClassVar, Final, Mapping, Tuple

import numpy as np
import torch

from .cteq_contact_timing import CteqContractError, PrivilegedLabelLeakageError
from .cteq_runner_termination_provenance import (
    CTEQ_BASE_CONTACT_KEY,
    CTEQ_EPISODE_ID_BEFORE_KEY,
    CTEQ_OTHER_TERMINATION_KEY,
    CTEQ_POST_STEP_EPISODE_ID_KEY,
    CTEQ_TERMINAL_CONTACT_KEY,
    CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT,
    CTEQ_TERMINAL_CONTACT_VALID_KEY,
    CTEQ_TERMINAL_EPISODE_ID_KEY,
    CTEQ_TIME_LIMIT_KEY,
    CteqOnPolicyTerminationProvenance,
    build_cteq_on_policy_termination_provenance,
)


CTEQ_ISAACLAB_EXTRAS_BRIDGE_SCHEMA: Final[str] = (
    "cteq-isaaclab-extras-bridge-v1"
)
CTEQ_ISAACLAB_EXTRAS_RECEIPT_SCHEMA: Final[str] = (
    "cteq-isaaclab-extras-provenance-receipt-v1"
)
CTEQ_TERMINAL_CONTACT_RECEIPT_SCHEMA: Final[str] = (
    "cteq-terminal-contact-source-receipt-v1"
)
CTEQ_FULLY_OBSERVED_BINS_SOURCE_CONTRACT: Final[str] = (
    "cteq_anchor_to_boundary_fully_observed_bins_v1"
)
CTEQ_FULLY_OBSERVED_BINS_RECEIPT_SCHEMA: Final[str] = (
    "cteq-fully-observed-bins-source-receipt-v1"
)
CTEQ_CAPTURE_HOOK: Final[str] = (
    "isaaclab_record_post_step_before_record_pre_reset_v1"
)
CTEQ_ISAACLAB_ALLOWED_CONSUMERS: Final[tuple[str, ...]] = (
    "label_builder",
    "loss",
    "evaluator",
)

CTEQ_RAW_TIME_LIMIT_KEY: Final[str] = "cteq_raw_time_limit_terminations"
CTEQ_ISAACLAB_TIME_LIMIT_KEY: Final[str] = "cteq_time_limit_terminations"
CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY: Final[str] = (
    "cteq_terminal_contact_source_receipt"
)
CTEQ_FULLY_OBSERVED_BINS_KEY: Final[str] = "cteq_fully_observed_bins"
CTEQ_FULLY_OBSERVED_BINS_VALID_KEY: Final[str] = (
    "cteq_fully_observed_bins_valid"
)
CTEQ_FULLY_OBSERVED_BINS_SOURCE_RECEIPT_KEY: Final[str] = (
    "cteq_fully_observed_bins_source_receipt"
)
CTEQ_BRIDGE_METADATA_KEY: Final[str] = "cteq_extras_bridge_metadata"

CTEQ_ISAACLAB_REQUIRED_EXTRAS_KEYS: Final[tuple[str, ...]] = (
    "time_outs",
    CTEQ_RAW_TIME_LIMIT_KEY,
    CTEQ_ISAACLAB_TIME_LIMIT_KEY,
    CTEQ_BASE_CONTACT_KEY,
    CTEQ_OTHER_TERMINATION_KEY,
    CTEQ_TERMINAL_CONTACT_KEY,
    CTEQ_TERMINAL_CONTACT_VALID_KEY,
    CTEQ_EPISODE_ID_BEFORE_KEY,
    CTEQ_TERMINAL_EPISODE_ID_KEY,
    CTEQ_POST_STEP_EPISODE_ID_KEY,
    CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY,
    CTEQ_FULLY_OBSERVED_BINS_KEY,
    CTEQ_FULLY_OBSERVED_BINS_VALID_KEY,
    CTEQ_FULLY_OBSERVED_BINS_SOURCE_RECEIPT_KEY,
    CTEQ_BRIDGE_METADATA_KEY,
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _cpu_array(
    value: Any,
    name: str,
    *,
    allow_device_transfer: bool,
    transfer_audit: list[str],
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            if not allow_device_transfer:
                raise CteqContractError(
                    f"{name} is on {value.device}; pass allow_device_transfer=True "
                    "to the explicit privileged CPU bridge."
                )
            transfer_audit.append(name)
            value = value.detach().to(device="cpu")
        else:
            value = value.detach()
        value = value.numpy()
    return np.asarray(value)


def _array(
    value: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype | None,
    integer: bool = False,
    allow_device_transfer: bool,
    transfer_audit: list[str],
) -> np.ndarray:
    result = _cpu_array(
        value,
        name,
        allow_device_transfer=allow_device_transfer,
        transfer_audit=transfer_audit,
    )
    if result.shape != shape:
        raise CteqContractError(f"{name} must have shape {list(shape)}.")
    if integer:
        if result.dtype.kind not in "iu":
            raise CteqContractError(f"{name} must use integer dtype.")
        result = result.astype(np.int64, copy=False)
    elif result.dtype != dtype:
        raise CteqContractError(f"{name} has an invalid dtype.")
    return np.ascontiguousarray(result)


def _validate_hashed_receipt(
    receipt: Any, *, schema: str, contract: str
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise CteqContractError(f"{schema} is missing.")
    result = copy.deepcopy(dict(receipt))
    if result.get("schema") != schema or result.get("contract") != contract:
        raise CteqContractError(f"{schema} contract changed.")
    digest = result.get("receipt_sha256")
    if not _is_sha256(digest):
        raise CteqContractError(f"{schema} digest is invalid.")
    unhashed = dict(result)
    del unhashed["receipt_sha256"]
    if digest != _canonical_sha256(unhashed):
        raise CteqContractError(f"{schema} digest mismatch.")
    return result


def _validate_terminal_contact_receipt(receipt: Any) -> dict[str, Any]:
    result = _validate_hashed_receipt(
        receipt,
        schema=CTEQ_TERMINAL_CONTACT_RECEIPT_SCHEMA,
        contract=CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT,
    )
    exact = {
        "capture_hook": CTEQ_CAPTURE_HOOK,
        "foot_order": ["left", "right"],
        "source_field": "net_forces_w",
        "force_reduction": "vector_l2_norm",
        "contact_rule": "force_norm_strictly_greater_than_threshold",
        "terminal_rows_only": True,
        "truth_only": True,
        "allowed_consumers": list(CTEQ_ISAACLAB_ALLOWED_CONSUMERS),
        "actor_observation_access": False,
        "critic_hidden_access": False,
        "reward_access": False,
        "training_ready": False,
    }
    for key, value in exact.items():
        if result.get(key) != value:
            raise CteqContractError(f"Terminal-contact receipt field {key} changed.")
    body_names = result.get("foot_body_names")
    threshold = result.get("contact_force_threshold_n")
    if (
        not isinstance(result.get("sensor_name"), str)
        or not result["sensor_name"]
        or not isinstance(body_names, list)
        or len(body_names) != 2
        or any(not isinstance(name, str) or not name for name in body_names)
        or type(threshold) not in {int, float}
        or isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise CteqContractError("Terminal-contact sensor semantics are invalid.")
    return result


def _validate_fully_observed_bins_receipt(receipt: Any) -> dict[str, Any]:
    result = _validate_hashed_receipt(
        receipt,
        schema=CTEQ_FULLY_OBSERVED_BINS_RECEIPT_SCHEMA,
        contract=CTEQ_FULLY_OBSERVED_BINS_SOURCE_CONTRACT,
    )
    exact = {
        "capture_hook": CTEQ_CAPTURE_HOOK,
        "sample_period_s": 0.02,
        "max_bins": 25,
        "count_semantics": (
            "complete_post_anchor_contact_samples_before_boundary"
        ),
        "truth_only": True,
        "allowed_consumers": list(CTEQ_ISAACLAB_ALLOWED_CONSUMERS),
        "training_ready": False,
    }
    for key, value in exact.items():
        if result.get(key) != value:
            raise CteqContractError(f"fully_observed_bins receipt field {key} changed.")
    if not isinstance(result.get("provider_id"), str) or not result["provider_id"]:
        raise CteqContractError("fully_observed_bins provider id is missing.")
    if not _is_sha256(result.get("provider_source_sha256")):
        raise CteqContractError("fully_observed_bins provider source SHA is invalid.")
    return result


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CteqIsaacLabTerminationBatch:
    """Privileged, CPU-owned boundary truth for label construction only."""

    termination: CteqOnPolicyTerminationProvenance
    fully_observed_bins: np.ndarray
    fully_observed_bins_valid: np.ndarray
    audit_receipt: Mapping[str, Any]
    truth_only: bool = True
    allowed_consumers: Tuple[str, ...] = CTEQ_ISAACLAB_ALLOWED_CONSUMERS
    training_ready: bool = False
    _cteq_privileged_truth_marker: ClassVar[bool] = True

    def for_consumer(self, consumer: str) -> Mapping[str, Any]:
        if consumer not in self.allowed_consumers:
            raise PrivilegedLabelLeakageError(
                "IsaacLab CTEQ extras are label_builder/loss/evaluator truth only."
            )
        return {
            "termination": self.termination.for_consumer(consumer),
            "fully_observed_bins": self.fully_observed_bins.copy(),
            "fully_observed_bins_valid": self.fully_observed_bins_valid.copy(),
        }

    def observation_payload(self) -> Mapping[str, Any]:
        raise PrivilegedLabelLeakageError(
            "IsaacLab CTEQ extras cannot enter actor observations."
        )

    def critic_hidden_payload(self) -> Mapping[str, Any]:
        raise PrivilegedLabelLeakageError(
            "IsaacLab CTEQ extras cannot enter critic hidden state."
        )

    def reward_payload(self) -> Mapping[str, Any]:
        raise PrivilegedLabelLeakageError(
            "IsaacLab CTEQ extras cannot enter reward computation."
        )

    def receipt(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.audit_receipt))


def build_cteq_isaaclab_termination_batch(
    dones: Any,
    extras: Mapping[str, Any],
    *,
    allow_device_transfer: bool = False,
) -> CteqIsaacLabTerminationBatch:
    """Validate one post-autoreset return using only receipted pre-reset truth."""
    if not isinstance(extras, Mapping):
        raise CteqContractError("IsaacLab extras must be a mapping.")
    missing = [key for key in CTEQ_ISAACLAB_REQUIRED_EXTRAS_KEYS if key not in extras]
    if missing:
        raise CteqContractError(
            "IsaacLab CTEQ extras are incomplete; missing " + ", ".join(missing)
        )
    transfer_audit: list[str] = []
    done_raw = _cpu_array(
        dones,
        "dones",
        allow_device_transfer=allow_device_transfer,
        transfer_audit=transfer_audit,
    )
    if done_raw.ndim != 1 or done_raw.shape[0] < 1:
        raise CteqContractError("dones must have shape [B].")
    if done_raw.dtype == np.bool_:
        done = np.ascontiguousarray(done_raw)
    elif done_raw.dtype.kind in "iu" and np.all((done_raw == 0) | (done_raw == 1)):
        done = np.ascontiguousarray(done_raw, dtype=np.bool_)
    else:
        raise CteqContractError("dones must be boolean or binary integer.")
    batch_size = done.shape[0]

    arrays: dict[str, np.ndarray] = {}
    for key in (
        "time_outs",
        CTEQ_RAW_TIME_LIMIT_KEY,
        CTEQ_ISAACLAB_TIME_LIMIT_KEY,
        CTEQ_BASE_CONTACT_KEY,
        CTEQ_OTHER_TERMINATION_KEY,
        CTEQ_TERMINAL_CONTACT_VALID_KEY,
        CTEQ_FULLY_OBSERVED_BINS_VALID_KEY,
    ):
        arrays[key] = _array(
            extras[key],
            key,
            shape=(batch_size,),
            dtype=np.dtype(np.bool_),
            allow_device_transfer=allow_device_transfer,
            transfer_audit=transfer_audit,
        )
    if not np.array_equal(arrays["time_outs"], arrays[CTEQ_RAW_TIME_LIMIT_KEY]):
        raise CteqContractError(
            "RSL-RL time_outs differs from the recorder's raw pre-reset timeout."
        )
    raw_terminated = arrays[CTEQ_BASE_CONTACT_KEY] | arrays[CTEQ_OTHER_TERMINATION_KEY]
    if np.any(arrays[CTEQ_BASE_CONTACT_KEY] & arrays[CTEQ_OTHER_TERMINATION_KEY]):
        raise CteqContractError("Base-contact and other termination overlap.")
    expected_time = arrays[CTEQ_RAW_TIME_LIMIT_KEY] & ~raw_terminated
    if not np.array_equal(arrays[CTEQ_ISAACLAB_TIME_LIMIT_KEY], expected_time):
        raise CteqContractError("Exclusive time-limit classification is inconsistent.")
    if not np.array_equal(done, arrays[CTEQ_RAW_TIME_LIMIT_KEY] | raw_terminated):
        raise CteqContractError("dones does not equal the raw IsaacLab termination union.")

    contact = _array(
        extras[CTEQ_TERMINAL_CONTACT_KEY],
        CTEQ_TERMINAL_CONTACT_KEY,
        shape=(batch_size, 2),
        dtype=np.dtype(np.bool_),
        allow_device_transfer=allow_device_transfer,
        transfer_audit=transfer_audit,
    )
    episodes: dict[str, np.ndarray] = {}
    for key in (
        CTEQ_EPISODE_ID_BEFORE_KEY,
        CTEQ_TERMINAL_EPISODE_ID_KEY,
        CTEQ_POST_STEP_EPISODE_ID_KEY,
    ):
        episodes[key] = _array(
            extras[key],
            key,
            shape=(batch_size,),
            dtype=None,
            integer=True,
            allow_device_transfer=allow_device_transfer,
            transfer_audit=transfer_audit,
        )
    bins = _array(
        extras[CTEQ_FULLY_OBSERVED_BINS_KEY],
        CTEQ_FULLY_OBSERVED_BINS_KEY,
        shape=(batch_size,),
        dtype=None,
        integer=True,
        allow_device_transfer=allow_device_transfer,
        transfer_audit=transfer_audit,
    )
    bins_valid = arrays[CTEQ_FULLY_OBSERVED_BINS_VALID_KEY]
    if not np.array_equal(bins_valid, done):
        raise CteqContractError("fully_observed_bins must be valid exactly on done rows.")
    if np.any((bins < 0) | (bins > 25)) or np.any(~bins_valid & (bins != 0)):
        raise CteqContractError("fully_observed_bins values are invalid or contain latent truth.")

    metadata = extras[CTEQ_BRIDGE_METADATA_KEY]
    expected_metadata = {
        "schema": CTEQ_ISAACLAB_EXTRAS_BRIDGE_SCHEMA,
        "capture_hook": CTEQ_CAPTURE_HOOK,
        "truth_only": True,
        "allowed_consumers": list(CTEQ_ISAACLAB_ALLOWED_CONSUMERS),
        "actor_observation_access": False,
        "critic_hidden_access": False,
        "reward_access": False,
        "training_ready": False,
    }
    if metadata != expected_metadata:
        raise CteqContractError("IsaacLab CTEQ bridge metadata changed.")
    terminal_receipt = _validate_terminal_contact_receipt(
        extras[CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY]
    )
    bins_receipt = _validate_fully_observed_bins_receipt(
        extras[CTEQ_FULLY_OBSERVED_BINS_SOURCE_RECEIPT_KEY]
    )

    canonical_extras = {
        CTEQ_TIME_LIMIT_KEY: arrays[CTEQ_ISAACLAB_TIME_LIMIT_KEY],
        CTEQ_BASE_CONTACT_KEY: arrays[CTEQ_BASE_CONTACT_KEY],
        CTEQ_OTHER_TERMINATION_KEY: arrays[CTEQ_OTHER_TERMINATION_KEY],
        CTEQ_TERMINAL_CONTACT_KEY: contact,
        CTEQ_TERMINAL_CONTACT_VALID_KEY: arrays[CTEQ_TERMINAL_CONTACT_VALID_KEY],
        CTEQ_EPISODE_ID_BEFORE_KEY: episodes[CTEQ_EPISODE_ID_BEFORE_KEY],
        CTEQ_TERMINAL_EPISODE_ID_KEY: episodes[CTEQ_TERMINAL_EPISODE_ID_KEY],
        CTEQ_POST_STEP_EPISODE_ID_KEY: episodes[CTEQ_POST_STEP_EPISODE_ID_KEY],
    }
    termination = build_cteq_on_policy_termination_provenance(
        done,
        canonical_extras,
        terminal_contact_source_sha256=terminal_receipt["receipt_sha256"],
    )
    termination_receipt = termination.receipt()
    tensor_sha = _tensor_sha256(done, bins, bins_valid)
    receipt: dict[str, Any] = {
        "schema": CTEQ_ISAACLAB_EXTRAS_RECEIPT_SCHEMA,
        "batch_size": batch_size,
        "required_extras_keys": list(CTEQ_ISAACLAB_REQUIRED_EXTRAS_KEYS),
        "capture_hook": CTEQ_CAPTURE_HOOK,
        "autoreset_boundary_authenticated": True,
        "raw_timeout_cross_checked": True,
        "simultaneous_timeout_mdp_priority": "mdp_termination",
        "fully_observed_bins_authenticated": True,
        "terminal_contact_source_receipt_sha256": terminal_receipt[
            "receipt_sha256"
        ],
        "fully_observed_bins_source_receipt_sha256": bins_receipt[
            "receipt_sha256"
        ],
        "runner_termination_receipt_sha256": termination_receipt[
            "receipt_sha256"
        ],
        "device_transfer_authorized": allow_device_transfer,
        "device_transfer_fields": sorted(set(transfer_audit)),
        "truth_only": True,
        "allowed_consumers": list(CTEQ_ISAACLAB_ALLOWED_CONSUMERS),
        "actor_observation_access": False,
        "critic_hidden_access": False,
        "reward_access": False,
        "tensor_sha256": tensor_sha,
        "gym_task_registered": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return CteqIsaacLabTerminationBatch(
        termination=termination,
        fully_observed_bins=_readonly(bins, np.int64),
        fully_observed_bins_valid=_readonly(bins_valid, np.bool_),
        audit_receipt=receipt,
    )


def validate_cteq_isaaclab_extras_receipt(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping):
        raise CteqContractError("IsaacLab extras receipt must be a mapping.")
    if receipt.get("schema") != CTEQ_ISAACLAB_EXTRAS_RECEIPT_SCHEMA:
        raise CteqContractError("IsaacLab extras receipt schema changed.")
    if receipt.get("required_extras_keys") != list(CTEQ_ISAACLAB_REQUIRED_EXTRAS_KEYS):
        raise CteqContractError("IsaacLab extras receipt interface changed.")
    exact = {
        "capture_hook": CTEQ_CAPTURE_HOOK,
        "autoreset_boundary_authenticated": True,
        "raw_timeout_cross_checked": True,
        "simultaneous_timeout_mdp_priority": "mdp_termination",
        "fully_observed_bins_authenticated": True,
        "truth_only": True,
        "allowed_consumers": list(CTEQ_ISAACLAB_ALLOWED_CONSUMERS),
        "actor_observation_access": False,
        "critic_hidden_access": False,
        "reward_access": False,
        "gym_task_registered": False,
        "training_ready": False,
    }
    for key, value in exact.items():
        if receipt.get(key) != value:
            raise CteqContractError(f"IsaacLab extras receipt field {key} changed.")
    if type(receipt.get("device_transfer_authorized")) is not bool:
        raise CteqContractError("IsaacLab device-transfer authorization is invalid.")
    fields = receipt.get("device_transfer_fields")
    if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
        raise CteqContractError("IsaacLab device-transfer audit is invalid.")
    if not receipt["device_transfer_authorized"] and fields:
        raise CteqContractError("Device transfer occurred without authorization.")
    if type(receipt.get("batch_size")) is not int or receipt["batch_size"] < 1:
        raise CteqContractError("IsaacLab extras batch size is invalid.")
    for key in (
        "terminal_contact_source_receipt_sha256",
        "fully_observed_bins_source_receipt_sha256",
        "runner_termination_receipt_sha256",
        "tensor_sha256",
        "receipt_sha256",
    ):
        if not _is_sha256(receipt.get(key)):
            raise CteqContractError(f"IsaacLab extras receipt field {key} is invalid.")
    unhashed = dict(receipt)
    digest = unhashed.pop("receipt_sha256")
    if digest != _canonical_sha256(unhashed):
        raise CteqContractError("IsaacLab extras receipt digest mismatch.")


def cteq_isaaclab_extras_bridge_status() -> Mapping[str, Any]:
    return {
        "schema": "cteq-isaaclab-extras-bridge-status-v1",
        "implemented": True,
        "opt_in_only": True,
        "stock_runner_modified": False,
        "registered_task_modified": False,
        "authoritative_hook_required": CTEQ_CAPTURE_HOOK,
        "explicit_gpu_to_cpu_transfer_available": True,
        "explicit_gpu_to_cpu_transfer_default": False,
        "fully_observed_bins_provider_required": True,
        "fully_observed_bins_provider_implemented": False,
        "actor_integrated": False,
        "critic_hidden_integrated": False,
        "reward_integrated": False,
        "gym_task_registered": False,
        "training_ready": False,
    }


__all__ = [
    "CTEQ_BRIDGE_METADATA_KEY",
    "CTEQ_FULLY_OBSERVED_BINS_KEY",
    "CTEQ_FULLY_OBSERVED_BINS_SOURCE_RECEIPT_KEY",
    "CTEQ_FULLY_OBSERVED_BINS_VALID_KEY",
    "CTEQ_ISAACLAB_EXTRAS_BRIDGE_SCHEMA",
    "CTEQ_ISAACLAB_REQUIRED_EXTRAS_KEYS",
    "CTEQ_ISAACLAB_TIME_LIMIT_KEY",
    "CTEQ_RAW_TIME_LIMIT_KEY",
    "CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY",
    "CteqIsaacLabTerminationBatch",
    "build_cteq_isaaclab_termination_batch",
    "cteq_isaaclab_extras_bridge_status",
    "validate_cteq_isaaclab_extras_receipt",
]
