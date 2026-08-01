# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One fail-closed software path from Livox ``CustomMsg`` to actor tensors.

This module closes interface wiring only.  It does not authenticate that the
message came from a physical MID-360, verify the external clock evidence, or
promote the raw-event PIES contract.  In particular, the current production
tensor builder resolves a packet-cell collision by nearest range first; PIES
uses latest event first.  The receipt keeps those claims separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .mid360_clock_alignment import (
    Mid360ClockAlignment,
    map_livox_packet_to_action_clock,
)
from .mid360_ray_time_builder import (
    Mid360PacketStats,
    Mid360RayTimeBuilderError,
    Mid360RayTimeTensorBuilder,
)
from .mid360_ros_adapter import livox_custom_msg_to_sensor_clock_packet
from .ray_event_observation import aligned_history_to_ray_event_observation

MID360_RUNTIME_INTEGRATION_SCHEMA: Final[str] = (
    "mid360_custom_msg_action_tensor_software_v1"
)
MID360_PACKET_CELL_WINNER: Final[str] = (
    "nearest_range_then_earliest_timestamp_on_exact_range_tie"
)
_SOFTWARE_GATES: Final[dict[str, bool]] = {
    "custom_msg_count_frame_numeric_contract_checked": True,
    "sensor_to_action_clock_mapping_checked": True,
    "transport_latency_recorded": True,
    "range_valid_cross_view_identity_checked": True,
    "external_clock_evidence_verified_by_this_module": False,
    "physical_sensor_recording_authenticated": False,
    "raw_event_stable_ids_present": False,
    "raw_event_pies_reducer_connected": False,
    "pies_same_winner_control_ready": False,
    "training_ready": False,
    "g1_closed_loop_verified": False,
}
_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "claim_scope",
        "window_index",
        "action_clock_domain",
        "builder_manifest_payload_sha256",
        "clock_alignment_receipt_sha256",
        "clock_uncertainty_s",
        "transport_latency_s",
        "ray_history_shape",
        "ray_history_dtype",
        "ray_history_sha256",
        "ray_event_history_shape",
        "ray_event_history_dtype",
        "ray_event_history_sha256",
        "packet_cell_winner",
        "software_gates",
    }
)


@dataclass(frozen=True)
class Mid360SoftwareRuntimeStep:
    """Two actor views and their deliberately narrow software receipt."""

    ray_history: np.ndarray
    ray_event_history: np.ndarray
    packet_stats: Mid360PacketStats
    receipt: dict[str, Any]
    receipt_payload_sha256: str


def ingest_livox_custom_msg_runtime_step(
    message: Any,
    *,
    expected_frame_id: str,
    window_index: int,
    capture_end_sensor_s: float,
    received_action_time_s: float,
    action_time_s: float,
    alignment: Mid360ClockAlignment,
    builder: Mid360RayTimeTensorBuilder,
) -> Mid360SoftwareRuntimeStep:
    """Adapt, map, ingest, and pack one native Livox message.

    ``action_time_s`` is the later policy evaluation time in the alignment's
    action clock.  The caller must supply the externally calibrated mapping;
    receive time is never used to estimate it.
    """
    if not isinstance(alignment, Mid360ClockAlignment):
        raise TypeError("alignment must be Mid360ClockAlignment.")
    if not isinstance(builder, Mid360RayTimeTensorBuilder):
        raise TypeError("builder must be Mid360RayTimeTensorBuilder.")
    if builder.monotonic_clock_domain != alignment.action_clock_domain:
        raise Mid360RayTimeBuilderError(
            "Builder action clock differs from the supplied alignment."
        )
    now = _finite_float(action_time_s, "action_time_s")
    received = _finite_float(
        received_action_time_s,
        "received_action_time_s",
    )
    if now < received:
        raise Mid360RayTimeBuilderError(
            "action_time_s cannot precede received_action_time_s."
        )

    sensor_packet = livox_custom_msg_to_sensor_clock_packet(
        message,
        expected_frame_id=expected_frame_id,
        window_index=window_index,
        capture_end_sensor_s=capture_end_sensor_s,
        sensor_clock_domain=alignment.sensor_clock_domain,
    )
    mapped = map_livox_packet_to_action_clock(
        sensor_packet,
        alignment,
        received_action_time_s=received,
    )
    stats = builder.ingest_point_packet(mapped.packet)
    aligned = builder.aligned_event_time_history(
        now_s=now,
        monotonic_clock_domain=alignment.action_clock_domain,
    )
    ray_history = builder.policy_tensor(now_s=now)
    ray_event_history = aligned_history_to_ray_event_observation(aligned)[None]
    _validate_cross_view_identity(ray_history, ray_event_history)

    receipt = {
        "schema": MID360_RUNTIME_INTEGRATION_SCHEMA,
        "claim_scope": "software_interface_closure_only",
        "window_index": int(stats.window_index),
        "action_clock_domain": alignment.action_clock_domain,
        "builder_manifest_payload_sha256": stats.manifest_payload_sha256,
        "clock_alignment_receipt_sha256": (
            mapped.clock_alignment_receipt_sha256
        ),
        "clock_uncertainty_s": float(mapped.clock_uncertainty_s),
        "transport_latency_s": stats.transport_latency_s,
        "ray_history_shape": list(ray_history.shape),
        "ray_history_dtype": str(ray_history.dtype),
        "ray_history_sha256": _array_sha256(ray_history),
        "ray_event_history_shape": list(ray_event_history.shape),
        "ray_event_history_dtype": str(ray_event_history.dtype),
        "ray_event_history_sha256": _array_sha256(ray_event_history),
        "packet_cell_winner": MID360_PACKET_CELL_WINNER,
        "software_gates": dict(_SOFTWARE_GATES),
    }
    receipt_hash = _canonical_sha256(receipt)
    result = Mid360SoftwareRuntimeStep(
        ray_history=np.array(ray_history, copy=True),
        ray_event_history=np.array(ray_event_history, copy=True),
        packet_stats=stats,
        receipt=receipt,
        receipt_payload_sha256=receipt_hash,
    )
    validate_mid360_software_runtime_step(result)
    return result


def validate_mid360_software_runtime_step(
    step: Mid360SoftwareRuntimeStep,
) -> None:
    """Rebuild all software bindings and reject receipt or tensor mutation."""
    if not isinstance(step, Mid360SoftwareRuntimeStep):
        raise TypeError("step must be Mid360SoftwareRuntimeStep.")
    if not isinstance(step.receipt, Mapping):
        raise Mid360RayTimeBuilderError("Runtime receipt must be a mapping.")
    receipt = dict(step.receipt)
    if set(receipt) != _RECEIPT_KEYS:
        raise Mid360RayTimeBuilderError("Runtime receipt keys differ.")
    if step.receipt_payload_sha256 != _canonical_sha256(receipt):
        raise Mid360RayTimeBuilderError("Runtime receipt payload SHA-256 differs.")
    if receipt.get("schema") != MID360_RUNTIME_INTEGRATION_SCHEMA:
        raise Mid360RayTimeBuilderError("Runtime receipt schema differs.")
    if receipt.get("claim_scope") != "software_interface_closure_only":
        raise Mid360RayTimeBuilderError("Runtime receipt claim scope differs.")
    if receipt.get("packet_cell_winner") != MID360_PACKET_CELL_WINNER:
        raise Mid360RayTimeBuilderError("Runtime packet-cell winner differs.")
    if receipt.get("software_gates") != _SOFTWARE_GATES:
        raise Mid360RayTimeBuilderError("Runtime software gates differ.")

    for name, value in (
        ("ray_history", step.ray_history),
        ("ray_event_history", step.ray_event_history),
    ):
        if not isinstance(value, np.ndarray):
            raise Mid360RayTimeBuilderError(f"{name} must be a NumPy array.")
        if receipt.get(f"{name}_shape") != list(value.shape):
            raise Mid360RayTimeBuilderError(f"{name} shape receipt differs.")
        if receipt.get(f"{name}_dtype") != str(value.dtype):
            raise Mid360RayTimeBuilderError(f"{name} dtype receipt differs.")
        if receipt.get(f"{name}_sha256") != _array_sha256(value):
            raise Mid360RayTimeBuilderError(f"{name} SHA-256 differs.")
    _validate_cross_view_identity(step.ray_history, step.ray_event_history)

    stats = step.packet_stats
    if not isinstance(stats, Mid360PacketStats):
        raise Mid360RayTimeBuilderError(
            "packet_stats must be Mid360PacketStats."
        )
    expected_stats = {
        "window_index": stats.window_index,
        "action_clock_domain": stats.monotonic_clock_domain,
        "builder_manifest_payload_sha256": stats.manifest_payload_sha256,
        "transport_latency_s": stats.transport_latency_s,
    }
    for name, expected in expected_stats.items():
        if receipt.get(name) != expected:
            raise Mid360RayTimeBuilderError(
                f"Runtime receipt {name} differs from packet stats."
            )
    for name in (
        "builder_manifest_payload_sha256",
        "clock_alignment_receipt_sha256",
    ):
        _lower_sha256(receipt.get(name), name)
    uncertainty = _finite_float(
        receipt.get("clock_uncertainty_s"),
        "clock_uncertainty_s",
    )
    if uncertainty < 0.0 or uncertainty > 0.005:
        raise Mid360RayTimeBuilderError(
            "clock_uncertainty_s must stay inside the 5 ms software gate."
        )


def _validate_cross_view_identity(
    ray_history: np.ndarray,
    ray_event_history: np.ndarray,
) -> None:
    if ray_history.ndim != 5 or ray_history.shape[2] != 2:
        raise Mid360RayTimeBuilderError(
            "ray_history must have shape [1,K,2,H,W]."
        )
    if (
        ray_history.shape[0] != 1
        or tuple(ray_history.shape[3:]) != (16, 96)
        or ray_history.dtype != np.float16
    ):
        raise Mid360RayTimeBuilderError(
            "ray_history must be float16 [1,K,2,16,96]."
        )
    expected = (
        ray_history.shape[0],
        ray_history.shape[1],
        5,
        ray_history.shape[3],
        ray_history.shape[4],
    )
    if ray_event_history.shape != expected:
        raise Mid360RayTimeBuilderError(
            f"ray_event_history must have shape {expected}."
        )
    if ray_event_history.dtype != np.float32:
        raise Mid360RayTimeBuilderError(
            "ray_event_history must use float32."
        )
    range_equal = np.array_equal(
        ray_event_history[:, :, 0],
        ray_history[:, :, 0].astype(np.float32),
    )
    valid_equal = np.array_equal(
        ray_event_history[:, :, 1],
        ray_history[:, :, 1].astype(np.float32),
    )
    if not range_equal or not valid_equal:
        raise Mid360RayTimeBuilderError(
            "Range/event actor views do not share range and validity."
        )


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Mid360RayTimeBuilderError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise Mid360RayTimeBuilderError(f"{name} must be finite.")
    return result


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _lower_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Mid360RayTimeBuilderError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


__all__ = [
    "MID360_PACKET_CELL_WINNER",
    "MID360_RUNTIME_INTEGRATION_SCHEMA",
    "Mid360SoftwareRuntimeStep",
    "ingest_livox_custom_msg_runtime_step",
    "validate_mid360_software_runtime_step",
]
