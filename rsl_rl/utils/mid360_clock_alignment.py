# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed MID-360 sensor-clock to policy action-clock mapping.

The module applies an externally calibrated affine mapping; it does not infer
clock alignment from ROS receive time.  Receive time contains transport and
queue latency and therefore cannot authenticate acquisition time.  Real H2
evidence must bind the returned payload hash to an independent hardware clock
calibration receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Final

from .mid360_ray_time_builder import (
    MID360_CAPTURE_END_ACQUISITION_WINDOW,
    MID360_TIMESTAMP_LIVOX_CUSTOM_MSG,
    MID360_TIMESTAMP_LIVOX_CUSTOM_MSG_ACTION_CLOCK,
    Mid360PointPacket,
    Mid360RayTimeBuilderError,
)

MID360_CLOCK_ALIGNMENT_SCHEMA: Final[str] = "mid360_action_clock_alignment_v1"
MID360_CLOCK_MIN_SAMPLES: Final[int] = 32
MID360_CLOCK_MIN_CALIBRATION_SPAN_S: Final[float] = 30.0
MID360_CLOCK_MAX_DRIFT_PPM: Final[float] = 100.0
MID360_CLOCK_MAX_P99_RESIDUAL_S: Final[float] = 0.002
MID360_CLOCK_MAX_RESIDUAL_S: Final[float] = 0.005
_SHA256_HEX_LENGTH = 64


class Mid360ClockAlignmentError(Mid360RayTimeBuilderError):
    """Raised when clock evidence or a mapped packet violates the contract."""


@dataclass(frozen=True)
class Mid360ClockAlignment:
    """Externally measured affine mapping from sensor seconds to action seconds.

    ``action_time = scale * sensor_time + offset_s``.  The accepted packet
    interval must lie inside the measured calibration interval; this module
    deliberately forbids unbounded extrapolation.
    """

    sensor_clock_domain: str
    action_clock_domain: str
    sensor_serial: str
    host_boot_id: str
    calibration_method: str
    calibration_evidence_sha256: str
    scale: float
    offset_s: float
    calibrated_sensor_start_s: float
    calibrated_sensor_end_s: float
    sample_count: int
    residual_p99_s: float
    residual_max_s: float
    uncertainty_s: float

    def __post_init__(self) -> None:
        """Validate identity, calibration coverage, drift and residual gates."""
        for name in (
            "sensor_clock_domain",
            "action_clock_domain",
            "sensor_serial",
            "host_boot_id",
            "calibration_method",
        ):
            _nonempty_text(getattr(self, name), name)
        if self.sensor_clock_domain == self.action_clock_domain:
            raise Mid360ClockAlignmentError(
                "Sensor and action clocks already share an identity; an affine cross-clock mapping would be ambiguous."
            )
        _sha256(self.calibration_evidence_sha256, "calibration_evidence_sha256")
        scale = _finite(self.scale, "scale")
        offset = _finite(self.offset_s, "offset_s")
        del offset
        if scale <= 0.0:
            raise Mid360ClockAlignmentError("scale must be positive.")
        drift_ppm = abs(scale - 1.0) * 1.0e6
        if drift_ppm > MID360_CLOCK_MAX_DRIFT_PPM:
            raise Mid360ClockAlignmentError(
                f"Clock drift {drift_ppm:.6f} ppm exceeds {MID360_CLOCK_MAX_DRIFT_PPM:.6f} ppm."
            )
        start = _finite(
            self.calibrated_sensor_start_s,
            "calibrated_sensor_start_s",
        )
        end = _finite(
            self.calibrated_sensor_end_s,
            "calibrated_sensor_end_s",
        )
        if end <= start:
            raise Mid360ClockAlignmentError("calibrated_sensor_end_s must follow calibrated_sensor_start_s.")
        if end - start < MID360_CLOCK_MIN_CALIBRATION_SPAN_S:
            raise Mid360ClockAlignmentError(
                f"Clock calibration span is shorter than {MID360_CLOCK_MIN_CALIBRATION_SPAN_S:.1f} s."
            )
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count < MID360_CLOCK_MIN_SAMPLES
        ):
            raise Mid360ClockAlignmentError(f"sample_count must be at least {MID360_CLOCK_MIN_SAMPLES}.")
        p99 = _nonnegative(self.residual_p99_s, "residual_p99_s")
        maximum = _nonnegative(self.residual_max_s, "residual_max_s")
        uncertainty = _nonnegative(self.uncertainty_s, "uncertainty_s")
        if p99 > maximum:
            raise Mid360ClockAlignmentError("residual_p99_s cannot exceed residual_max_s.")
        if p99 > MID360_CLOCK_MAX_P99_RESIDUAL_S:
            raise Mid360ClockAlignmentError("Clock residual P99 exceeds the 2 ms deployment gate.")
        if maximum > MID360_CLOCK_MAX_RESIDUAL_S:
            raise Mid360ClockAlignmentError("Clock maximum residual exceeds the 5 ms deployment gate.")
        if uncertainty < maximum or uncertainty > MID360_CLOCK_MAX_RESIDUAL_S:
            raise Mid360ClockAlignmentError("uncertainty_s must cover residual_max_s without exceeding 5 ms.")

    @property
    def drift_ppm(self) -> float:
        """Return absolute affine scale deviation in parts per million."""
        return abs(float(self.scale) - 1.0) * 1.0e6

    def map_sensor_times(self, sensor_times_s: np.ndarray) -> np.ndarray:
        """Map ordered float64 sensor times inside the calibrated interval."""
        if not isinstance(sensor_times_s, np.ndarray):
            raise TypeError("sensor_times_s must be a NumPy array.")
        if sensor_times_s.dtype != np.float64 or sensor_times_s.ndim != 1:
            raise Mid360ClockAlignmentError("sensor_times_s must be a one-dimensional float64 array.")
        if not np.isfinite(sensor_times_s).all():
            raise Mid360ClockAlignmentError("sensor_times_s must be finite.")
        if np.any(np.diff(sensor_times_s) < 0.0):
            raise Mid360ClockAlignmentError("sensor_times_s must be nondecreasing.")
        if sensor_times_s.size and (
            sensor_times_s[0] < self.calibrated_sensor_start_s or sensor_times_s[-1] > self.calibrated_sensor_end_s
        ):
            raise Mid360ClockAlignmentError(
                "Sensor time lies outside the calibrated interval; unbounded extrapolation is forbidden."
            )
        mapped = sensor_times_s * float(self.scale) + float(self.offset_s)
        if not np.isfinite(mapped).all() or np.any(np.diff(mapped) < 0.0):
            raise Mid360ClockAlignmentError("Affine mapping produced non-finite or reordered action times.")
        return np.asarray(mapped, dtype=np.float64)

    def receipt_payload(self) -> dict[str, Any]:
        """Return canonical semantics without claiming evidence authentication."""
        return {
            "schema": MID360_CLOCK_ALIGNMENT_SCHEMA,
            "mapping": {
                "sensor_clock_domain": self.sensor_clock_domain,
                "action_clock_domain": self.action_clock_domain,
                "formula": "action_time_s=scale*sensor_time_s+offset_s",
                "scale": float(self.scale),
                "offset_s": float(self.offset_s),
                "drift_ppm": self.drift_ppm,
                "calibrated_sensor_start_s": float(self.calibrated_sensor_start_s),
                "calibrated_sensor_end_s": float(self.calibrated_sensor_end_s),
            },
            "identity": {
                "sensor_serial": self.sensor_serial,
                "host_boot_id": self.host_boot_id,
            },
            "calibration": {
                "method": self.calibration_method,
                "sample_count": self.sample_count,
                "residual_p99_s": float(self.residual_p99_s),
                "residual_max_s": float(self.residual_max_s),
                "uncertainty_s": float(self.uncertainty_s),
                "external_evidence_sha256": self.calibration_evidence_sha256,
                "external_evidence_verified_by_this_module": False,
            },
            "software_gates": {
                "min_samples": MID360_CLOCK_MIN_SAMPLES,
                "min_calibration_span_s": MID360_CLOCK_MIN_CALIBRATION_SPAN_S,
                "max_abs_drift_ppm": MID360_CLOCK_MAX_DRIFT_PPM,
                "max_residual_p99_s": MID360_CLOCK_MAX_P99_RESIDUAL_S,
                "max_residual_s": MID360_CLOCK_MAX_RESIDUAL_S,
                "packet_interval_must_be_inside_calibration": True,
                "receive_time_used_for_calibration": False,
            },
            "claim_scope": "software_mapping_contract_only",
        }

    def receipt_payload_sha256(self) -> str:
        """Hash the exact canonical clock-mapping payload."""
        payload = json.dumps(
            self.receipt_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ActionClockMappedPacket:
    """Mapped packet plus the clock receipt hash that must follow it."""

    packet: Mid360PointPacket
    clock_alignment_receipt_sha256: str
    clock_uncertainty_s: float


def map_livox_packet_to_action_clock(
    packet: Mid360PointPacket,
    alignment: Mid360ClockAlignment,
    *,
    received_action_time_s: float,
) -> ActionClockMappedPacket:
    """Map one raw CustomMsg-derived packet into the policy action clock.

    The source packet must not contain a receive timestamp because that value
    is normally measured in the action clock, not the device clock.  The
    caller supplies it after mapping so transport latency is never used as a
    clock calibration surrogate.
    """
    if not isinstance(packet, Mid360PointPacket):
        raise TypeError("packet must be Mid360PointPacket.")
    if not isinstance(alignment, Mid360ClockAlignment):
        raise TypeError("alignment must be Mid360ClockAlignment.")
    if packet.timestamp_semantics != MID360_TIMESTAMP_LIVOX_CUSTOM_MSG:
        raise Mid360ClockAlignmentError("Only raw Livox CustomMsg return timestamps may enter this mapper.")
    if packet.monotonic_clock_domain != alignment.sensor_clock_domain:
        raise Mid360ClockAlignmentError("Packet sensor clock domain does not match the clock alignment.")
    if packet.capture_end_semantics != MID360_CAPTURE_END_ACQUISITION_WINDOW:
        raise Mid360ClockAlignmentError("Clock mapping requires an explicit acquisition-window end.")
    if packet.received_time_s is not None:
        raise Mid360ClockAlignmentError("Source packet received_time_s must be None before cross-clock mapping.")
    if packet.point_timestamps_s is None:
        raise Mid360ClockAlignmentError("Clock mapping requires one raw sensor timestamp per return.")
    boundaries = np.asarray(
        (packet.capture_start_s, packet.capture_end_s),
        dtype=np.float64,
    )
    mapped_boundaries = alignment.map_sensor_times(boundaries)
    mapped_points = alignment.map_sensor_times(np.asarray(packet.point_timestamps_s, dtype=np.float64))
    received = _finite(received_action_time_s, "received_action_time_s")
    if received < mapped_boundaries[1]:
        raise Mid360ClockAlignmentError("received_action_time_s cannot precede the mapped capture end.")
    mapped_packet = Mid360PointPacket(
        xyz_m=np.array(packet.xyz_m, copy=True),
        coordinate_frame=packet.coordinate_frame,
        timestamp_semantics=MID360_TIMESTAMP_LIVOX_CUSTOM_MSG_ACTION_CLOCK,
        window_index=packet.window_index,
        capture_start_s=float(mapped_boundaries[0]),
        capture_end_s=float(mapped_boundaries[1]),
        received_time_s=received,
        point_timestamps_s=mapped_points,
        emitted_sensor_mask=(
            None if packet.emitted_sensor_mask is None else np.array(packet.emitted_sensor_mask, copy=True)
        ),
        monotonic_clock_domain=alignment.action_clock_domain,
        capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
    )
    return ActionClockMappedPacket(
        packet=mapped_packet,
        clock_alignment_receipt_sha256=alignment.receipt_payload_sha256(),
        clock_uncertainty_s=float(alignment.uncertainty_s),
    )


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Mid360ClockAlignmentError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise Mid360ClockAlignmentError(f"{name} must be finite.")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise Mid360ClockAlignmentError(f"{name} must be non-negative.")
    return result


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Mid360ClockAlignmentError(f"{name} must be a non-empty string without surrounding whitespace.")
    if not value.isprintable():
        raise Mid360ClockAlignmentError(f"{name} must be printable.")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Mid360ClockAlignmentError(f"{name} must be a lowercase SHA-256 hex digest.")
    return value


__all__ = [
    "MID360_CLOCK_ALIGNMENT_SCHEMA",
    "MID360_CLOCK_MAX_DRIFT_PPM",
    "MID360_CLOCK_MAX_P99_RESIDUAL_S",
    "MID360_CLOCK_MAX_RESIDUAL_S",
    "MID360_CLOCK_MIN_CALIBRATION_SPAN_S",
    "MID360_CLOCK_MIN_SAMPLES",
    "ActionClockMappedPacket",
    "Mid360ClockAlignment",
    "Mid360ClockAlignmentError",
    "map_livox_packet_to_action_clock",
]
