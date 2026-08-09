# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-first provenance contract for CTEQ on-policy episode boundaries.

The stock runner exposes ``dones`` and optionally ``extras['time_outs']`` but
cannot identify base-contact/other termination or prove that a contact sample
was captured before auto-reset.  This adapter therefore requires explicit
environment-owned extras and fails closed when any field is missing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, ClassVar, Final, Mapping, Tuple

import numpy as np
import torch

from .cteq_contact_timing import (
    CTEQ_ALLOWED_TRUTH_CONSUMERS,
    CteqContractError,
    PrivilegedLabelLeakageError,
)


CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA: Final[str] = (
    "cteq-on-policy-termination-provenance-v1"
)
CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT: Final[str] = (
    "environment_pre_reset_terminal_foot_contact_v1"
)
CTEQ_RUNNER_PROVENANCE_ALLOWED_CONSUMERS: Final[tuple[str, ...]] = (
    "label_builder",
    *CTEQ_ALLOWED_TRUTH_CONSUMERS,
)
CTEQ_TIME_LIMIT_KEY: Final[str] = "time_outs"
CTEQ_BASE_CONTACT_KEY: Final[str] = "cteq_base_contact_terminations"
CTEQ_OTHER_TERMINATION_KEY: Final[str] = "cteq_other_terminations"
CTEQ_TERMINAL_CONTACT_KEY: Final[str] = "cteq_terminal_foot_contact"
CTEQ_TERMINAL_CONTACT_VALID_KEY: Final[str] = "cteq_terminal_contact_valid"
CTEQ_EPISODE_ID_BEFORE_KEY: Final[str] = "cteq_episode_id_before_step"
CTEQ_TERMINAL_EPISODE_ID_KEY: Final[str] = "cteq_terminal_contact_episode_id"
CTEQ_POST_STEP_EPISODE_ID_KEY: Final[str] = "cteq_post_step_episode_id"
CTEQ_REQUIRED_EXTRAS_KEYS: Final[tuple[str, ...]] = (
    CTEQ_TIME_LIMIT_KEY,
    CTEQ_BASE_CONTACT_KEY,
    CTEQ_OTHER_TERMINATION_KEY,
    CTEQ_TERMINAL_CONTACT_KEY,
    CTEQ_TERMINAL_CONTACT_VALID_KEY,
    CTEQ_EPISODE_ID_BEFORE_KEY,
    CTEQ_TERMINAL_EPISODE_ID_KEY,
    CTEQ_POST_STEP_EPISODE_ID_KEY,
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _cpu_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise CteqContractError(
                f"{name} must be transferred by an explicit future adapter; "
                "this provenance contract is CPU-only."
            )
        value = value.detach().numpy()
    return np.asarray(value)


def _bool_vector(value: Any, name: str, batch_size: int) -> np.ndarray:
    array = _cpu_array(value, name)
    if array.shape != (batch_size,) or array.dtype != np.bool_:
        raise CteqContractError(f"{name} must be bool with shape [B].")
    return np.ascontiguousarray(array)


def _episode_vector(value: Any, name: str, batch_size: int) -> np.ndarray:
    array = _cpu_array(value, name)
    if array.shape != (batch_size,) or array.dtype.kind not in "iu":
        raise CteqContractError(f"{name} must be integer with shape [B].")
    result = np.ascontiguousarray(array, dtype=np.int64)
    if np.any(result < -1):
        raise CteqContractError(f"{name} contains an invalid episode id.")
    return result


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CteqOnPolicyTerminationProvenance:
    """Authenticated pre-reset boundary tensors for the offline label builder."""

    episode_done: np.ndarray
    time_limit: np.ndarray
    base_contact_termination: np.ndarray
    other_early_termination: np.ndarray
    terminal_foot_contact: np.ndarray
    terminal_contact_valid: np.ndarray
    episode_id_before_step: np.ndarray
    terminal_contact_episode_id: np.ndarray
    post_step_episode_id: np.ndarray
    audit_receipt: Mapping[str, Any]
    truth_only: bool = True
    allowed_consumers: Tuple[str, ...] = CTEQ_RUNNER_PROVENANCE_ALLOWED_CONSUMERS
    training_ready: bool = False
    _cteq_privileged_truth_marker: ClassVar[bool] = True

    def for_consumer(self, consumer: str) -> Mapping[str, np.ndarray]:
        if consumer not in self.allowed_consumers:
            raise PrivilegedLabelLeakageError(
                "Runner termination/contact provenance is privileged truth for "
                "label_builder/loss/evaluator only."
            )
        return {
            "episode_done": self.episode_done.copy(),
            "time_limit": self.time_limit.copy(),
            "base_contact_termination": self.base_contact_termination.copy(),
            "other_early_termination": self.other_early_termination.copy(),
            "terminal_foot_contact": self.terminal_foot_contact.copy(),
            "terminal_contact_valid": self.terminal_contact_valid.copy(),
            "episode_id_before_step": self.episode_id_before_step.copy(),
            "terminal_contact_episode_id": (
                self.terminal_contact_episode_id.copy()
            ),
            "post_step_episode_id": self.post_step_episode_id.copy(),
        }

    def observation_payload(self) -> Mapping[str, np.ndarray]:
        raise PrivilegedLabelLeakageError(
            "Termination provenance cannot enter actor observations."
        )

    def critic_hidden_payload(self) -> Mapping[str, np.ndarray]:
        raise PrivilegedLabelLeakageError(
            "Termination provenance cannot enter critic hidden state."
        )

    def receipt(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.audit_receipt))


def build_cteq_on_policy_termination_provenance(
    dones: Any,
    extras: Mapping[str, Any],
    *,
    terminal_contact_source_contract: str = (
        CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT
    ),
    terminal_contact_source_sha256: str | None,
) -> CteqOnPolicyTerminationProvenance:
    """Validate a canonical on-policy step without reading post-reset truth."""
    done_array = _cpu_array(dones, "dones")
    if done_array.ndim != 1 or done_array.dtype != np.bool_:
        raise CteqContractError("dones must be bool with shape [B].")
    batch_size = done_array.shape[0]
    if batch_size < 1:
        raise CteqContractError("Termination provenance batch cannot be empty.")
    if not isinstance(extras, Mapping):
        raise CteqContractError("On-policy extras must be a mapping.")
    missing = [key for key in CTEQ_REQUIRED_EXTRAS_KEYS if key not in extras]
    if missing:
        raise CteqContractError(
            "Current runner termination provenance is incomplete; missing "
            + ", ".join(missing)
        )
    if terminal_contact_source_contract != CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT:
        raise CteqContractError("Terminal contact source contract changed.")
    if not _is_sha256(terminal_contact_source_sha256):
        raise CteqContractError(
            "Terminal contact source requires a lowercase SHA-256 receipt."
        )

    done = np.ascontiguousarray(done_array)
    timeout = _bool_vector(extras[CTEQ_TIME_LIMIT_KEY], CTEQ_TIME_LIMIT_KEY, batch_size)
    base_contact = _bool_vector(
        extras[CTEQ_BASE_CONTACT_KEY], CTEQ_BASE_CONTACT_KEY, batch_size
    )
    other = _bool_vector(
        extras[CTEQ_OTHER_TERMINATION_KEY], CTEQ_OTHER_TERMINATION_KEY, batch_size
    )
    reason_sum = (
        timeout.astype(np.int8)
        + base_contact.astype(np.int8)
        + other.astype(np.int8)
    )
    if np.any(reason_sum > 1):
        raise CteqContractError("Termination reasons must be mutually exclusive.")
    if not np.array_equal(done, reason_sum == 1):
        raise CteqContractError(
            "dones must equal the exhaustive union of time-limit, base-contact, "
            "and other termination."
        )

    contact = _cpu_array(extras[CTEQ_TERMINAL_CONTACT_KEY], CTEQ_TERMINAL_CONTACT_KEY)
    if contact.shape != (batch_size, 2) or contact.dtype != np.bool_:
        raise CteqContractError(
            "cteq_terminal_foot_contact must be bool with shape [B,2]."
        )
    contact = np.ascontiguousarray(contact)
    contact_valid = _bool_vector(
        extras[CTEQ_TERMINAL_CONTACT_VALID_KEY],
        CTEQ_TERMINAL_CONTACT_VALID_KEY,
        batch_size,
    )
    if not np.array_equal(contact_valid, done):
        raise CteqContractError(
            "Every done row requires one authoritative pre-reset terminal "
            "contact sample; non-done rows must mark it invalid."
        )
    if np.any(~contact_valid[:, None] & contact):
        raise CteqContractError(
            "Invalid terminal-contact rows must be exactly false, not latent truth."
        )

    before_id = _episode_vector(
        extras[CTEQ_EPISODE_ID_BEFORE_KEY],
        CTEQ_EPISODE_ID_BEFORE_KEY,
        batch_size,
    )
    terminal_id = _episode_vector(
        extras[CTEQ_TERMINAL_EPISODE_ID_KEY],
        CTEQ_TERMINAL_EPISODE_ID_KEY,
        batch_size,
    )
    post_id = _episode_vector(
        extras[CTEQ_POST_STEP_EPISODE_ID_KEY],
        CTEQ_POST_STEP_EPISODE_ID_KEY,
        batch_size,
    )
    if np.any(before_id < 0) or np.any(post_id < 0):
        raise CteqContractError("Before/post episode ids must be non-negative.")
    if np.any(done & (terminal_id != before_id)):
        raise CteqContractError(
            "Terminal contact must belong to the pre-reset episode."
        )
    if np.any(~done & (terminal_id != -1)):
        raise CteqContractError(
            "Non-done rows require terminal_contact_episode_id=-1."
        )
    if np.any(done & (post_id <= before_id)):
        raise CteqContractError(
            "Done rows require a strictly newer post-reset episode id."
        )
    if np.any(~done & (post_id != before_id)):
        raise CteqContractError(
            "Non-done rows cannot cross an episode/reset boundary."
        )

    counts = {
        "done": int(np.count_nonzero(done)),
        "time_limit": int(np.count_nonzero(timeout)),
        "base_contact": int(np.count_nonzero(base_contact)),
        "other_termination": int(np.count_nonzero(other)),
        "terminal_contact_valid": int(np.count_nonzero(contact_valid)),
        "reset_boundary": int(np.count_nonzero(post_id != before_id)),
    }
    tensor_sha = _tensor_sha256(
        done,
        timeout,
        base_contact,
        other,
        contact,
        contact_valid,
        before_id,
        terminal_id,
        post_id,
    )
    receipt: dict[str, Any] = {
        "schema": CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA,
        "batch_size": batch_size,
        "required_extras_keys": list(CTEQ_REQUIRED_EXTRAS_KEYS),
        "terminal_contact_source_contract": terminal_contact_source_contract,
        "terminal_contact_source_sha256": terminal_contact_source_sha256,
        "terminal_contact_stage": "pre_reset_terminal_transition",
        "episode_boundary_contract": (
            "terminal_id_equals_before_and_post_id_strictly_newer_on_done"
        ),
        "termination_reasons_exhaustive": True,
        "termination_reasons_mutually_exclusive": True,
        "terminal_contact_authenticated": True,
        "episode_reset_boundary_authenticated": True,
        "provenance_authenticated": True,
        "termination_is_foot_event": False,
        "actor_observation_access": False,
        "critic_hidden_access": False,
        "allowed_consumers": list(CTEQ_RUNNER_PROVENANCE_ALLOWED_CONSUMERS),
        "counts": counts,
        "tensor_sha256": tensor_sha,
        "gpu_required": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return CteqOnPolicyTerminationProvenance(
        episode_done=_readonly(done, np.bool_),
        time_limit=_readonly(timeout, np.bool_),
        base_contact_termination=_readonly(base_contact, np.bool_),
        other_early_termination=_readonly(other, np.bool_),
        terminal_foot_contact=_readonly(contact, np.bool_),
        terminal_contact_valid=_readonly(contact_valid, np.bool_),
        episode_id_before_step=_readonly(before_id, np.int64),
        terminal_contact_episode_id=_readonly(terminal_id, np.int64),
        post_step_episode_id=_readonly(post_id, np.int64),
        audit_receipt=receipt,
    )


def validate_cteq_runner_termination_provenance_receipt(
    receipt: Mapping[str, Any],
) -> None:
    if not isinstance(receipt, Mapping):
        raise CteqContractError("Runner termination receipt must be a mapping.")
    if receipt.get("schema") != CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA:
        raise CteqContractError("Runner termination receipt schema changed.")
    if receipt.get("required_extras_keys") != list(CTEQ_REQUIRED_EXTRAS_KEYS):
        raise CteqContractError("Runner termination extras interface changed.")
    if receipt.get("terminal_contact_source_contract") != (
        CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT
    ):
        raise CteqContractError("Terminal contact source contract changed.")
    if not _is_sha256(receipt.get("terminal_contact_source_sha256")):
        raise CteqContractError("Terminal contact source receipt is invalid.")
    if receipt.get("terminal_contact_stage") != (
        "pre_reset_terminal_transition"
    ):
        raise CteqContractError("Terminal contact stage changed.")
    if receipt.get("episode_boundary_contract") != (
        "terminal_id_equals_before_and_post_id_strictly_newer_on_done"
    ):
        raise CteqContractError("Episode reset boundary contract changed.")
    for field, expected in (
        ("termination_reasons_exhaustive", True),
        ("termination_reasons_mutually_exclusive", True),
        ("terminal_contact_authenticated", True),
        ("episode_reset_boundary_authenticated", True),
        ("provenance_authenticated", True),
        ("termination_is_foot_event", False),
        ("actor_observation_access", False),
        ("critic_hidden_access", False),
        ("gpu_required", False),
        ("training_ready", False),
    ):
        if receipt.get(field) is not expected:
            raise CteqContractError(f"Runner termination receipt field {field} changed.")
    if receipt.get("allowed_consumers") != list(
        CTEQ_RUNNER_PROVENANCE_ALLOWED_CONSUMERS
    ):
        raise CteqContractError("Runner provenance consumers changed.")
    batch_size = receipt.get("batch_size")
    counts = receipt.get("counts")
    if type(batch_size) is not int or batch_size < 1 or not isinstance(counts, Mapping):
        raise CteqContractError("Runner receipt batch/count metadata is invalid.")
    expected_count_keys = {
        "done",
        "time_limit",
        "base_contact",
        "other_termination",
        "terminal_contact_valid",
        "reset_boundary",
    }
    if set(counts) != expected_count_keys or any(
        type(value) is not int or value < 0 or value > batch_size
        for value in counts.values()
    ):
        raise CteqContractError("Runner termination counts are invalid.")
    if counts["done"] != (
        counts["time_limit"]
        + counts["base_contact"]
        + counts["other_termination"]
    ):
        raise CteqContractError("Runner termination counts are not exhaustive.")
    if counts["terminal_contact_valid"] != counts["done"]:
        raise CteqContractError("Terminal contact count differs from done count.")
    if counts["reset_boundary"] != counts["done"]:
        raise CteqContractError("Episode reset count differs from done count.")
    if not _is_sha256(receipt.get("tensor_sha256")):
        raise CteqContractError("Runner provenance tensor digest is invalid.")
    receipt_sha = receipt.get("receipt_sha256")
    if not _is_sha256(receipt_sha):
        raise CteqContractError("Runner provenance receipt digest is invalid.")
    unhashed = dict(receipt)
    del unhashed["receipt_sha256"]
    if receipt_sha != _canonical_sha256(unhashed):
        raise CteqContractError("Runner provenance receipt digest mismatch.")


def cteq_current_on_policy_runner_provenance_status() -> Mapping[str, Any]:
    """Audit the stock runner boundary without claiming unavailable extras."""
    return {
        "schema": "cteq-current-runner-provenance-status-v1",
        "stock_runner_visible_fields": ["dones", "extras.time_outs_optional"],
        "required_extras_keys": list(CTEQ_REQUIRED_EXTRAS_KEYS),
        "missing_from_stock_runner_contract": [
            key for key in CTEQ_REQUIRED_EXTRAS_KEYS if key != CTEQ_TIME_LIMIT_KEY
        ],
        "opt_in_isaaclab_extras_bridge_available": True,
        "opt_in_bridge_module": "rsl_rl.utils.cteq_isaaclab_extras_bridge",
        "stock_runner_modified": False,
        "remaining_interfaces": [
            "explicit_environment_opt_in_to_pre_reset_recorder",
            "real_anchor_to_terminal_fully_observed_bins_provider",
        ],
        "terminal_contact_authoritative": False,
        "episode_reset_boundary_authenticated": False,
        "provenance_authenticated": False,
        "actor_integrated": False,
        "critic_hidden_integrated": False,
        "gpu_required": False,
        "training_ready": False,
    }


__all__ = [
    "CTEQ_BASE_CONTACT_KEY",
    "CTEQ_EPISODE_ID_BEFORE_KEY",
    "CTEQ_OTHER_TERMINATION_KEY",
    "CTEQ_POST_STEP_EPISODE_ID_KEY",
    "CTEQ_REQUIRED_EXTRAS_KEYS",
    "CTEQ_RUNNER_TERMINATION_PROVENANCE_SCHEMA",
    "CTEQ_TERMINAL_CONTACT_KEY",
    "CTEQ_TERMINAL_CONTACT_SOURCE_CONTRACT",
    "CTEQ_TERMINAL_CONTACT_VALID_KEY",
    "CTEQ_TERMINAL_EPISODE_ID_KEY",
    "CTEQ_TIME_LIMIT_KEY",
    "CteqOnPolicyTerminationProvenance",
    "build_cteq_on_policy_termination_provenance",
    "cteq_current_on_policy_runner_provenance_status",
    "validate_cteq_runner_termination_provenance_receipt",
]
