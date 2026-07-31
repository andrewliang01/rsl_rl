# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-side MID-360 point-return to ray-time tensor conversion.

This module is deliberately independent of Isaac Sim and ROS.  A driver
adapter must first normalize its native message into :class:`Mid360PointPacket`.
That boundary is explicit because Livox driver message fields, timestamp
domains, point units, and coordinate-frame conventions differ between
deployments and cannot be inferred safely here.

Ray-time schemas v1/v2 preserve only successful returns.  A zero hit mask means
``unknown``: it intentionally does not distinguish an unscanned direction, a
no-return, a filtered point, or a lost packet.  In particular, the synthetic
20-percent phase mask used during simulation is never applied to real points.

The legacy ``[range, hit]`` interface remains available.  H2 event-time users
must opt into an explicit monotonic-clock domain and consume
:class:`Mid360AlignedRayTimeHistory`.  That stricter interface keeps each
range, validity bit, and acquisition timestamp attached to one deterministic
winning point return through spherical binning, width flip/roll, and history
shifts.  It never invents timestamps for no-return or capture-window-only data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .ray_time_deployment_manifest import (
    RayTimeManifestError,
    read_ray_time_deployment_manifest,
    validate_ray_time_deployment_manifest,
)


MID360_NORMALIZED_SENSOR_FRAME = (
    "mid360_physical_sensor_frame_x_forward_y_left_z_up_metres"
)
"""The sole coordinate convention accepted by the ray-time point builder."""

MID360_TIMESTAMP_LIVOX_CUSTOM_MSG = (
    "livox_custom_msg_timebase_ns_plus_point_offset_time_ns"
)
MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS = (
    "adapter_declared_absolute_point_times_in_capture_clock"
)
MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY = (
    "adapter_declared_capture_window_without_verified_point_times"
)
MID360_CAPTURE_END_ACQUISITION_WINDOW = (
    "adapter_declared_policy_acquisition_window_end"
)
MID360_CAPTURE_END_LATEST_RETURN_LEGACY = (
    "legacy_helper_inferred_latest_successful_return_time"
)
_MID360_TIMESTAMP_SEMANTICS = (
    MID360_TIMESTAMP_LIVOX_CUSTOM_MSG,
    MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
    MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
)


class Mid360RayTimeBuilderError(ValueError):
    """Raised when real sensor data cannot satisfy the deployment contract."""


class StaleMid360PacketError(Mid360RayTimeBuilderError):
    """Raised instead of silently reusing an expired or future packet."""


@dataclass(frozen=True)
class Mid360PointPacket:
    """One completed policy-rate acquisition window of successful returns.

    ``xyz_m`` must already be expressed from the optical origin in
    :data:`MID360_NORMALIZED_SENSOR_FRAME`.  A driver-specific adapter is
    responsible for unit conversion, field extraction, de-skewing decisions,
    and any verified transform into this frame.

    ``window_index`` is a deployment-owned, monotonically increasing policy
    packet index, not an assumed Livox sequence field.  It lets the builder
    represent lost acquisition windows exactly without guessing from timestamp
    jitter.  Strict event-time mode additionally requires
    ``monotonic_clock_domain`` to identify the shared capture/action clock and
    ``capture_end_semantics`` to declare the policy acquisition-window end.
    The domain covers capture bounds, point timestamps, and received time; it
    must also identify the clock used for the later policy action time.
    """

    xyz_m: np.ndarray
    coordinate_frame: str
    timestamp_semantics: str
    window_index: int
    capture_start_s: float
    capture_end_s: float
    received_time_s: float | None = None
    point_timestamps_s: np.ndarray | None = None
    emitted_sensor_mask: np.ndarray | None = None
    monotonic_clock_domain: str | None = None
    capture_end_semantics: str | None = None


@runtime_checkable
class Mid360PacketAdapter(Protocol):
    """Explicit boundary from an arbitrary driver message to normalized data."""

    def to_mid360_point_packet(self, raw_packet: Any) -> Mid360PointPacket:
        """Normalize one driver-specific packet window without hidden defaults."""


@dataclass(frozen=True)
class Mid360PacketStats:
    """Auditable tensorization and packet-timing diagnostics."""

    window_index: int
    capture_start_s: float
    capture_end_s: float
    received_time_s: float | None
    transport_latency_s: float | None
    input_return_points: int
    accepted_return_points: int
    below_min_range_points: int
    outside_vertical_fov_points: int
    collided_return_points: int
    policy_clipped_points: int
    observed_hit_bins: int
    observed_hit_fraction: float
    emitted_coverage_known: bool
    emitted_bin_fraction: float | None
    implicit_missing_packets_inserted: int
    capture_interval_error_s: float | None
    explicit_drop: bool
    manifest_payload_sha256: str
    max_packet_age_s: float
    packet_cadence_tolerance_s: float
    timestamp_tolerance_s: float
    timed_return_bins: int = 0
    event_time_frame_valid: bool = False
    monotonic_clock_domain: str | None = None
    capture_end_semantics: str | None = None


@dataclass(frozen=True)
class Mid360AlignedRayTimeHistory:
    """One unbatched, oldest-to-newest range/event-time history snapshot.

    Shapes are ``ray_history=[K,2,H,W]``, ``return_valid/return_age_s=
    [K,H,W]``, and ``packet_age_s/frame_valid=[K]``.  ``return_valid`` is
    bitwise identical to the range-history hit channel.  Invalid returns have
    exactly zero age.  Empty, dropped, and never-filled frames have
    ``frame_valid=False`` and exactly zero packet age.

    The snapshot can only be produced by a builder configured with an explicit
    ``monotonic_clock_domain``.  The caller must present the same domain for
    ``now_s`` so clock identity is checked rather than inferred from floats.
    """

    ray_history: np.ndarray
    return_valid: np.ndarray
    return_age_s: np.ndarray
    packet_age_s: np.ndarray
    frame_valid: np.ndarray
    window_indices: np.ndarray
    capture_end_times_s: np.ndarray
    monotonic_clock_domain: str


def point_packet_from_livox_custom_msg_arrays(
    *,
    xyz_m: np.ndarray,
    timebase_ns: int,
    offset_time_ns: np.ndarray,
    coordinate_frame: str,
    window_index: int,
    capture_end_s: float | None = None,
    received_time_s: float | None = None,
    emitted_sensor_mask: np.ndarray | None = None,
    monotonic_clock_domain: str | None = None,
) -> Mid360PointPacket:
    """Normalize verified ``CustomMsg timebase + offset_time`` arrays.

    The official ``livox_ros_driver2`` timing relationship is represented
    explicitly: ``timebase`` is the message time base and every
    ``offset_time`` is added to it.  This helper deliberately accepts arrays
    instead of importing ROS message classes or guessing fields from a
    ``PointCloud2`` layout.

    A caller may append the result directly only when that CustomMsg is the
    deployment's complete policy-rate acquisition window.  If several native
    messages form one 0.1-second window, the driver adapter must aggregate them
    first and return one :class:`Mid360PointPacket`.  ``capture_end_s`` must be
    supplied for strict H2 use; omitting it preserves the legacy latest-return
    inference, which the strict builder rejects because it changes when late
    returns are absent.
    """
    points = _floating_matrix(xyz_m, "xyz_m")
    if not np.isfinite(points).all():
        raise Mid360RayTimeBuilderError(
            "xyz_m must contain only finite values."
        )
    base_ns = _nonnegative_int(timebase_ns, "timebase_ns")
    if not isinstance(offset_time_ns, np.ndarray):
        raise TypeError("offset_time_ns must be a NumPy array.")
    if offset_time_ns.shape != (points.shape[0],):
        raise Mid360RayTimeBuilderError(
            "offset_time_ns must have one value per point, got "
            f"{offset_time_ns.shape} for {points.shape[0]} points."
        )
    if offset_time_ns.dtype.kind not in ("i", "u"):
        raise TypeError("offset_time_ns must use an integer dtype.")
    if offset_time_ns.dtype.kind == "i" and np.any(offset_time_ns < 0):
        raise Mid360RayTimeBuilderError(
            "offset_time_ns must be non-negative."
        )

    base_s = base_ns * 1.0e-9
    point_times = base_s + offset_time_ns.astype(np.float64) * 1.0e-9
    latest_return_s = (
        base_s if point_times.size == 0 else float(point_times.max())
    )
    if capture_end_s is None:
        resolved_capture_end_s = latest_return_s
        capture_end_semantics = MID360_CAPTURE_END_LATEST_RETURN_LEGACY
    else:
        resolved_capture_end_s = _finite_float(capture_end_s, "capture_end_s")
        if resolved_capture_end_s < latest_return_s:
            raise Mid360RayTimeBuilderError(
                "capture_end_s cannot precede the latest point timestamp."
            )
        capture_end_semantics = MID360_CAPTURE_END_ACQUISITION_WINDOW
    return Mid360PointPacket(
        xyz_m=points,
        coordinate_frame=coordinate_frame,
        timestamp_semantics=MID360_TIMESTAMP_LIVOX_CUSTOM_MSG,
        window_index=_nonnegative_int(window_index, "window_index"),
        capture_start_s=base_s,
        capture_end_s=resolved_capture_end_s,
        received_time_s=received_time_s,
        point_timestamps_s=point_times,
        emitted_sensor_mask=emitted_sensor_mask,
        monotonic_clock_domain=monotonic_clock_domain,
        capture_end_semantics=capture_end_semantics,
    )


class Mid360RayTimeTensorBuilder:
    """Build a manifest-bound ``[K, 2, H, W]`` real MID-360 history.

    The output is always finite ``float16`` with channels
    ``[range_m, hit_mask]`` and oldest-to-newest history ordering.  Use
    :meth:`policy_tensor` for the batched ``[1, K, 2, H, W]`` policy input.
    Freshness is measured from the declared acquisition-window end, not from
    the timestamp of the latest successful return. Every packet statistic
    records the runtime safety tolerances alongside the manifest hash.  Passing
    ``monotonic_clock_domain`` enables the stricter aligned event-time contract;
    it is intentionally opt-in so existing two-channel deployments are not
    silently reinterpreted.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        *,
        max_packet_age_s: float,
        timestamp_tolerance_s: float = 1.0e-6,
        packet_cadence_tolerance_s: float = 0.02,
        monotonic_clock_domain: str | None = None,
    ) -> None:
        if isinstance(manifest, (str, os.PathLike)):
            parsed = read_ray_time_deployment_manifest(
                Path(manifest),
                require_export_artifact=False,
            )
        elif isinstance(manifest, Mapping):
            validate_ray_time_deployment_manifest(
                manifest,
                require_export_artifact=False,
            )
            parsed = manifest
        else:
            raise TypeError(
                "manifest must be a validated mapping or a JSON sidecar path."
            )

        self.max_packet_age_s = _positive_finite_float(
            max_packet_age_s,
            "max_packet_age_s",
        )
        self.timestamp_tolerance_s = _nonnegative_finite_float(
            timestamp_tolerance_s,
            "timestamp_tolerance_s",
        )
        self.packet_cadence_tolerance_s = _nonnegative_finite_float(
            packet_cadence_tolerance_s,
            "packet_cadence_tolerance_s",
        )
        self._monotonic_clock_domain = _optional_clock_domain(
            monotonic_clock_domain,
            "monotonic_clock_domain",
        )

        contract = parsed["contract"]
        ray = contract["ray_history"]
        tensorization = ray["tensorization"]
        history_length, channels, image_height, image_width = ray["shape"]
        if channels != 2:
            raise RayTimeManifestError(
                "Ray-time real builder requires exactly two ray channels."
            )
        if ray["dtype"] != "float16":
            raise RayTimeManifestError(
                "Ray-time real builder requires float16 ray history."
            )
        if tensorization["real_packet_mask"] != (
            "use actual sensor-observed rays; never apply the simulation "
            "synthetic phase mask a second time"
        ):
            raise RayTimeManifestError(
                "Manifest does not prohibit a second synthetic real-packet mask."
            )

        width_reorder = tensorization["upside_down_width_reorder"]
        if width_reorder["operation_order"] != [
            "flip_width",
            "circular_roll_width",
        ]:
            raise RayTimeManifestError(
                "Unsupported upside-down width reordering contract."
            )

        self.history_length = int(history_length)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.min_range_m = float(ray["valid_range_m"]["min"])
        self.max_range_m = float(ray["valid_range_m"]["max"])
        self.packet_interval_s = float(
            ray["packet_interval_control_steps"] * ray["control_period_s"]
        )
        if (
            self.monotonic_clock_domain is not None
            and self.timestamp_tolerance_s >= self.packet_interval_s
        ):
            raise Mid360RayTimeBuilderError(
                "Strict event-time timestamp_tolerance_s must be smaller than "
                f"the manifest packet interval {self.packet_interval_s:.6f}s."
            )
        if self.packet_cadence_tolerance_s >= self.packet_interval_s:
            raise Mid360RayTimeBuilderError(
                "packet_cadence_tolerance_s must be smaller than the "
                f"manifest packet interval {self.packet_interval_s:.6f}s."
            )
        self.policy_row_first_deg = float(
            ray["row_elevation_degrees"]["first"]
        )
        self.policy_row_last_deg = float(
            ray["row_elevation_degrees"]["last"]
        )
        pattern_fov = tensorization[
            "pattern_sensor_vertical_fov_degrees"
        ]
        self.sensor_row_first_deg = float(pattern_fov[0])
        self.sensor_row_last_deg = float(pattern_fov[1])
        column = ray["column_azimuth_degrees"]
        self.sensor_column_first_deg = float(column["first"])
        self.sensor_column_step_deg = float(column["step"])
        self.width_roll_bins = int(width_reorder["circular_roll_bins"])
        self.manifest_payload_sha256 = str(
            parsed["integrity"]["payload_sha256"]
        )

        self._history = np.zeros(
            (
                self.history_length,
                2,
                self.image_height,
                self.image_width,
            ),
            dtype=np.float16,
        )
        self._window_indices = np.full(
            self.history_length,
            -1,
            dtype=np.int64,
        )
        self._capture_end_times_s = np.full(
            self.history_length,
            np.nan,
            dtype=np.float64,
        )
        self._return_timestamps_s = np.zeros(
            (
                self.history_length,
                self.image_height,
                self.image_width,
            ),
            dtype=np.float64,
        )
        self._return_timestamp_valid = np.zeros(
            (
                self.history_length,
                self.image_height,
                self.image_width,
            ),
            dtype=np.bool_,
        )
        self._frame_valid = np.zeros(
            self.history_length,
            dtype=np.bool_,
        )
        self._last_window_index: int | None = None
        self._last_window_capture_end_s: float | None = None
        self._last_acquisition_capture_end_s: float | None = None
        self._last_stats: Mid360PacketStats | None = None
        self._implicit_missing_packets_total = 0
        self._explicit_dropped_packets_total = 0

    @property
    def last_stats(self) -> Mid360PacketStats | None:
        """Return diagnostics for the last ingested or explicitly lost packet."""
        return self._last_stats

    @property
    def monotonic_clock_domain(self) -> str | None:
        """Return the immutable clock identity configured for strict H2 use."""
        return self._monotonic_clock_domain

    @property
    def implicit_missing_packets_total(self) -> int:
        return self._implicit_missing_packets_total

    @property
    def explicit_dropped_packets_total(self) -> int:
        return self._explicit_dropped_packets_total

    @property
    def window_indices(self) -> np.ndarray:
        """Return a copy of history slot indices; ``-1`` means never filled."""
        return self._window_indices.copy()

    @property
    def capture_end_times_s(self) -> np.ndarray:
        """Return capture times; implicit/unfilled unknown slots contain NaN."""
        return self._capture_end_times_s.copy()

    def ingest_driver_packet(
        self,
        raw_packet: Any,
        adapter: Mid360PacketAdapter,
    ) -> Mid360PacketStats:
        """Normalize a native packet through ``adapter`` and ingest it."""
        if not isinstance(adapter, Mid360PacketAdapter):
            raise TypeError(
                "adapter must implement to_mid360_point_packet(raw_packet)."
            )
        packet = adapter.to_mid360_point_packet(raw_packet)
        if not isinstance(packet, Mid360PointPacket):
            raise TypeError(
                "adapter.to_mid360_point_packet must return Mid360PointPacket."
            )
        return self.ingest_point_packet(packet)

    def ingest_point_packet(
        self,
        packet: Mid360PointPacket,
    ) -> Mid360PacketStats:
        """Bin successful point returns, append the packet, and report coverage."""
        validated = self._validate_point_packet(packet)
        (
            xyz,
            point_timestamps_s,
            emitted_sensor_mask,
            received_time_s,
            transport_latency_s,
        ) = validated

        ranges = np.linalg.norm(xyz.astype(np.float64, copy=False), axis=1)
        if not np.isfinite(ranges).all():
            raise Mid360RayTimeBuilderError(
                "packet.xyz_m produces a non-finite range."
            )
        above_min = ranges >= self.min_range_m
        safe_ranges = np.where(above_min, ranges, 1.0)
        elevation_deg = np.rad2deg(
            np.arcsin(
                np.clip(
                    xyz[:, 2].astype(np.float64, copy=False) / safe_ranges,
                    -1.0,
                    1.0,
                )
            )
        )
        in_vertical_fov = (
            elevation_deg
            >= min(self.sensor_row_first_deg, self.sensor_row_last_deg)
            - 1.0e-9
        ) & (
            elevation_deg
            <= max(self.sensor_row_first_deg, self.sensor_row_last_deg)
            + 1.0e-9
        )
        accepted = above_min & in_vertical_fov

        sensor_ranges = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.float64,
        )
        sensor_hits = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.bool_,
        )
        sensor_return_timestamps_s = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.float64,
        )
        sensor_return_timestamp_valid = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.bool_,
        )
        accepted_count = int(np.count_nonzero(accepted))
        clipped_count = 0
        if accepted_count:
            accepted_xyz = xyz[accepted].astype(np.float64, copy=False)
            accepted_ranges = ranges[accepted]
            accepted_elevation = elevation_deg[accepted]
            azimuth_deg = np.rad2deg(
                np.arctan2(accepted_xyz[:, 1], accepted_xyz[:, 0])
            )

            sensor_rows = self._sensor_rows_from_elevation(
                accepted_elevation
            )
            sensor_columns = self._sensor_columns_from_azimuth(azimuth_deg)
            flat_ids = sensor_rows * self.image_width + sensor_columns
            if point_timestamps_s is None:
                accepted_timestamps_s = np.zeros(
                    accepted_count,
                    dtype=np.float64,
                )
            else:
                accepted_timestamps_s = point_timestamps_s[accepted]

            # Deterministic winner contract: nearest physical range wins;
            # exact range ties use the earliest acquisition timestamp.  Any
            # remaining ties have identical exported range/time values, so
            # input point ordering cannot alter the aligned tensor.
            order = np.lexsort(
                (accepted_timestamps_s, accepted_ranges, flat_ids)
            )
            ordered_flat_ids = flat_ids[order]
            first_for_bin = np.empty(order.shape, dtype=np.bool_)
            first_for_bin[0] = True
            first_for_bin[1:] = (
                ordered_flat_ids[1:] != ordered_flat_ids[:-1]
            )
            winners = order[first_for_bin]
            winning_flat_ids = flat_ids[winners]
            sensor_hits.reshape(-1)[winning_flat_ids] = True
            sensor_ranges.reshape(-1)[winning_flat_ids] = accepted_ranges[
                winners
            ]
            if point_timestamps_s is not None:
                sensor_return_timestamp_valid.reshape(-1)[
                    winning_flat_ids
                ] = True
                sensor_return_timestamps_s.reshape(-1)[
                    winning_flat_ids
                ] = accepted_timestamps_s[winners]
            clipped_count = int(
                np.count_nonzero(accepted_ranges > self.max_range_m)
            )

        if emitted_sensor_mask is not None:
            if np.any(sensor_hits & ~emitted_sensor_mask):
                raise Mid360RayTimeBuilderError(
                    "emitted_sensor_mask must cover every successful return bin."
                )
            emitted_fraction: float | None = float(
                emitted_sensor_mask.mean(dtype=np.float64)
            )
        else:
            emitted_fraction = None

        policy_packet = self.sensor_grid_to_policy_packet(
            sensor_ranges,
            sensor_hits,
        )
        (
            policy_return_timestamps_s,
            policy_return_timestamp_valid,
        ) = self._sensor_timing_to_policy(
            sensor_return_timestamps_s,
            sensor_return_timestamp_valid,
        )
        hit_bins = int(np.count_nonzero(policy_packet[1]))
        timed_bins = int(np.count_nonzero(policy_return_timestamp_valid))
        event_time_frame_valid = timed_bins > 0
        stats = Mid360PacketStats(
            window_index=packet.window_index,
            capture_start_s=float(packet.capture_start_s),
            capture_end_s=float(packet.capture_end_s),
            received_time_s=received_time_s,
            transport_latency_s=transport_latency_s,
            input_return_points=int(xyz.shape[0]),
            accepted_return_points=accepted_count,
            below_min_range_points=int(np.count_nonzero(~above_min)),
            outside_vertical_fov_points=int(
                np.count_nonzero(above_min & ~in_vertical_fov)
            ),
            collided_return_points=accepted_count - hit_bins,
            policy_clipped_points=clipped_count,
            observed_hit_bins=hit_bins,
            observed_hit_fraction=float(
                hit_bins / (self.image_height * self.image_width)
            ),
            emitted_coverage_known=emitted_sensor_mask is not None,
            emitted_bin_fraction=emitted_fraction,
            implicit_missing_packets_inserted=0,
            capture_interval_error_s=None,
            explicit_drop=False,
            manifest_payload_sha256=self.manifest_payload_sha256,
            timed_return_bins=timed_bins,
            event_time_frame_valid=event_time_frame_valid,
            monotonic_clock_domain=packet.monotonic_clock_domain,
            capture_end_semantics=packet.capture_end_semantics,
            **self._safety_stats_fields(),
        )
        return self._append_packet(
            policy_packet,
            stats,
            policy_return_timestamps_s=policy_return_timestamps_s,
            policy_return_timestamp_valid=policy_return_timestamp_valid,
            frame_valid=event_time_frame_valid,
        )

    def ingest_sensor_grid(
        self,
        sensor_ranges_m: np.ndarray,
        sensor_hit_mask: np.ndarray,
        *,
        window_index: int,
        capture_start_s: float,
        capture_end_s: float,
        received_time_s: float | None = None,
        emitted_sensor_mask: np.ndarray | None = None,
        sensor_return_timestamps_s: np.ndarray | None = None,
        monotonic_clock_domain: str | None = None,
    ) -> Mid360PacketStats:
        """Append a pre-binned physical-sensor grid.

        Rows must follow the manifest's physical pattern order ``+52 -> -7``
        degrees and columns must be ascending sensor-frame azimuth on
        ``[-180, 180)``.  This method exists for verified upstream binners and
        simulation golden tests; it does not infer a native driver layout.
        In strict event-time mode, every ``True`` hit requires an aligned
        ``sensor_return_timestamps_s`` entry, while every non-hit timestamp
        must be exactly zero.
        """
        ranges = _floating_array(
            sensor_ranges_m,
            (self.image_height, self.image_width),
            "sensor_ranges_m",
        )
        if not np.isfinite(ranges).all():
            raise Mid360RayTimeBuilderError(
                "sensor_ranges_m must contain only finite values."
            )
        hits = _bool_array(
            sensor_hit_mask,
            (self.image_height, self.image_width),
            "sensor_hit_mask",
        )
        emitted = self._validate_emitted_mask(emitted_sensor_mask)
        valid_hits = hits & (ranges >= self.min_range_m)
        if emitted is not None and np.any(valid_hits & ~emitted):
            raise Mid360RayTimeBuilderError(
                "emitted_sensor_mask must cover every successful return bin."
            )

        window_index = _nonnegative_int(window_index, "window_index")
        capture_start_s = _finite_float(capture_start_s, "capture_start_s")
        capture_end_s = _finite_float(capture_end_s, "capture_end_s")
        if capture_end_s < capture_start_s:
            raise Mid360RayTimeBuilderError(
                "capture_end_s must be at or after capture_start_s."
            )
        self._validate_capture_duration(capture_start_s, capture_end_s)
        received, transport_latency = self._validate_received_time(
            capture_end_s,
            received_time_s,
        )
        clock_domain = self._validate_clock_domain(
            monotonic_clock_domain,
            context="ingest_sensor_grid",
        )
        if sensor_return_timestamps_s is None:
            if self.monotonic_clock_domain is not None and np.any(hits):
                raise Mid360RayTimeBuilderError(
                    "Strict event-time mode requires one timestamp for every "
                    "successful sensor-grid return."
                )
            sensor_timestamps = np.zeros_like(ranges, dtype=np.float64)
            sensor_timestamp_valid = np.zeros_like(hits)
        else:
            sensor_timestamps = self._validate_sensor_return_timestamps(
                sensor_return_timestamps_s,
                hits,
                capture_start_s,
                capture_end_s,
            )
            sensor_timestamp_valid = valid_hits.copy()
            sensor_timestamps = np.where(
                sensor_timestamp_valid,
                sensor_timestamps,
                0.0,
            )

        policy_packet = self.sensor_grid_to_policy_packet(ranges, hits)
        (
            policy_return_timestamps_s,
            policy_return_timestamp_valid,
        ) = self._sensor_timing_to_policy(
            sensor_timestamps,
            sensor_timestamp_valid,
        )
        hit_bins = int(np.count_nonzero(policy_packet[1]))
        timed_bins = int(np.count_nonzero(policy_return_timestamp_valid))
        event_time_frame_valid = timed_bins > 0
        stats = Mid360PacketStats(
            window_index=window_index,
            capture_start_s=capture_start_s,
            capture_end_s=capture_end_s,
            received_time_s=received,
            transport_latency_s=transport_latency,
            input_return_points=int(np.count_nonzero(hits)),
            accepted_return_points=int(np.count_nonzero(valid_hits)),
            below_min_range_points=int(
                np.count_nonzero(hits & ~valid_hits)
            ),
            outside_vertical_fov_points=0,
            collided_return_points=0,
            policy_clipped_points=int(
                np.count_nonzero(valid_hits & (ranges > self.max_range_m))
            ),
            observed_hit_bins=hit_bins,
            observed_hit_fraction=float(
                hit_bins / (self.image_height * self.image_width)
            ),
            emitted_coverage_known=emitted is not None,
            emitted_bin_fraction=(
                None
                if emitted is None
                else float(emitted.mean(dtype=np.float64))
            ),
            implicit_missing_packets_inserted=0,
            capture_interval_error_s=None,
            explicit_drop=False,
            manifest_payload_sha256=self.manifest_payload_sha256,
            timed_return_bins=timed_bins,
            event_time_frame_valid=event_time_frame_valid,
            monotonic_clock_domain=clock_domain,
            capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
            **self._safety_stats_fields(),
        )
        return self._append_packet(
            policy_packet,
            stats,
            policy_return_timestamps_s=policy_return_timestamps_s,
            policy_return_timestamp_valid=policy_return_timestamp_valid,
            frame_valid=event_time_frame_valid,
        )

    def record_dropped_packet(
        self,
        *,
        window_index: int,
        capture_start_s: float,
        capture_end_s: float,
        received_time_s: float | None = None,
        monotonic_clock_domain: str | None = None,
    ) -> Mid360PacketStats:
        """Append a known lost window as all-unknown rather than repeating data."""
        window_index = _nonnegative_int(window_index, "window_index")
        capture_start_s = _finite_float(capture_start_s, "capture_start_s")
        capture_end_s = _finite_float(capture_end_s, "capture_end_s")
        if capture_end_s < capture_start_s:
            raise Mid360RayTimeBuilderError(
                "capture_end_s must be at or after capture_start_s."
            )
        self._validate_capture_duration(capture_start_s, capture_end_s)
        received, transport_latency = self._validate_received_time(
            capture_end_s,
            received_time_s,
        )
        clock_domain = self._validate_clock_domain(
            monotonic_clock_domain,
            context="record_dropped_packet",
        )
        stats = Mid360PacketStats(
            window_index=window_index,
            capture_start_s=capture_start_s,
            capture_end_s=capture_end_s,
            received_time_s=received,
            transport_latency_s=transport_latency,
            input_return_points=0,
            accepted_return_points=0,
            below_min_range_points=0,
            outside_vertical_fov_points=0,
            collided_return_points=0,
            policy_clipped_points=0,
            observed_hit_bins=0,
            observed_hit_fraction=0.0,
            emitted_coverage_known=False,
            emitted_bin_fraction=None,
            implicit_missing_packets_inserted=0,
            capture_interval_error_s=None,
            explicit_drop=True,
            manifest_payload_sha256=self.manifest_payload_sha256,
            monotonic_clock_domain=clock_domain,
            capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
            **self._safety_stats_fields(),
        )
        result = self._append_packet(np.zeros_like(self._history[0]), stats)
        self._explicit_dropped_packets_total += 1
        return result

    def sensor_grid_to_policy_packet(
        self,
        sensor_ranges_m: np.ndarray,
        sensor_hit_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply range encoding and the manifest's width flip-then-roll."""
        ranges = _floating_array(
            sensor_ranges_m,
            (self.image_height, self.image_width),
            "sensor_ranges_m",
        )
        if not np.isfinite(ranges).all():
            raise Mid360RayTimeBuilderError(
                "sensor_ranges_m must contain only finite values."
            )
        requested_hits = _bool_array(
            sensor_hit_mask,
            (self.image_height, self.image_width),
            "sensor_hit_mask",
        )
        valid_hits = requested_hits & (ranges >= self.min_range_m)
        encoded_ranges = np.where(
            valid_hits,
            np.clip(ranges, self.min_range_m, self.max_range_m),
            0.0,
        )
        encoded_hits = valid_hits.astype(np.float64)

        encoded_ranges = self._sensor_grid_to_policy_order(encoded_ranges)
        encoded_hits = self._sensor_grid_to_policy_order(encoded_hits)
        packet = np.stack((encoded_ranges, encoded_hits), axis=0).astype(
            np.float16,
            copy=False,
        )
        self._validate_policy_packet(packet)
        return packet

    def history_tensor(self, *, now_s: float) -> np.ndarray:
        """Return an unbatched manifest-shape copy after a freshness check."""
        self._validate_freshness(now_s)
        self._validate_history()
        return self._history.copy()

    def policy_tensor(self, *, now_s: float) -> np.ndarray:
        """Return a batched ``[1, K, 2, H, W]`` copy for inference."""
        return self.history_tensor(now_s=now_s)[None, ...]

    def aligned_event_time_history(
        self,
        *,
        now_s: float,
        monotonic_clock_domain: str,
    ) -> Mid360AlignedRayTimeHistory:
        """Return the strict H2 range/time snapshot without a batch axis.

        This is deliberately separate from :meth:`history_tensor` so legacy
        two-channel deployments remain byte-for-byte compatible.  Enabling it
        requires ``monotonic_clock_domain`` at builder construction, matching
        domain declarations at ingestion, and the same declaration here for
        action time.  Successful returns without verified per-return times are
        rejected during ingestion in that mode.
        """
        if self.monotonic_clock_domain is None:
            raise Mid360RayTimeBuilderError(
                "aligned_event_time_history requires a builder configured "
                "with an explicit monotonic_clock_domain."
            )
        self._validate_clock_domain(
            monotonic_clock_domain,
            context="aligned_event_time_history",
        )
        now = self._validate_freshness(now_s)
        self._validate_history()

        frame_valid = self._frame_valid.copy()
        return_valid = self._return_timestamp_valid.copy()
        packet_age_s = np.zeros(self.history_length, dtype=np.float32)
        return_age_s = np.zeros(
            (
                self.history_length,
                self.image_height,
                self.image_width,
            ),
            dtype=np.float32,
        )
        if np.any(frame_valid):
            capture_end = self._capture_end_times_s[frame_valid]
            if not np.isfinite(capture_end).all():
                raise Mid360RayTimeBuilderError(
                    "Valid event-time frames require finite capture_end_s."
                )
            packet_ages = now - capture_end
            if np.any(packet_ages < 0.0) or not np.isfinite(packet_ages).all():
                raise Mid360RayTimeBuilderError(
                    "Computed packet ages must be finite and non-negative."
                )
            packet_age_s[frame_valid] = packet_ages.astype(np.float32)

        if np.any(return_valid):
            return_ages = now - self._return_timestamps_s[return_valid]
            if np.any(return_ages < 0.0) or not np.isfinite(return_ages).all():
                raise Mid360RayTimeBuilderError(
                    "Computed return ages must be finite and non-negative."
                )
            return_age_s[return_valid] = return_ages.astype(np.float32)

        return Mid360AlignedRayTimeHistory(
            ray_history=self._history.copy(),
            return_valid=return_valid,
            return_age_s=return_age_s,
            packet_age_s=packet_age_s,
            frame_valid=frame_valid,
            window_indices=self._window_indices.copy(),
            capture_end_times_s=self._capture_end_times_s.copy(),
            monotonic_clock_domain=self.monotonic_clock_domain,
        )

    def reset(self) -> None:
        """Clear all history, timing, and loss counters."""
        self._history.fill(0)
        self._window_indices.fill(-1)
        self._capture_end_times_s.fill(np.nan)
        self._return_timestamps_s.fill(0)
        self._return_timestamp_valid.fill(False)
        self._frame_valid.fill(False)
        self._last_window_index = None
        self._last_window_capture_end_s = None
        self._last_acquisition_capture_end_s = None
        self._last_stats = None
        self._implicit_missing_packets_total = 0
        self._explicit_dropped_packets_total = 0

    def _validate_point_packet(
        self,
        packet: Mid360PointPacket,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        float | None,
        float | None,
    ]:
        if not isinstance(packet, Mid360PointPacket):
            raise TypeError("packet must be Mid360PointPacket.")
        if packet.coordinate_frame != MID360_NORMALIZED_SENSOR_FRAME:
            raise Mid360RayTimeBuilderError(
                "Driver adapter must normalize points to "
                f"{MID360_NORMALIZED_SENSOR_FRAME!r}; got "
                f"{packet.coordinate_frame!r}."
            )
        if packet.timestamp_semantics not in _MID360_TIMESTAMP_SEMANTICS:
            raise Mid360RayTimeBuilderError(
                "packet.timestamp_semantics must explicitly identify one of "
                f"{_MID360_TIMESTAMP_SEMANTICS}; got "
                f"{packet.timestamp_semantics!r}."
            )
        _nonnegative_int(packet.window_index, "packet.window_index")
        start = _finite_float(
            packet.capture_start_s,
            "packet.capture_start_s",
        )
        end = _finite_float(packet.capture_end_s, "packet.capture_end_s")
        if end < start:
            raise Mid360RayTimeBuilderError(
                "packet.capture_end_s must be at or after capture_start_s."
            )
        self._validate_capture_duration(start, end)

        xyz = _floating_matrix(packet.xyz_m, "packet.xyz_m")
        if not np.isfinite(xyz).all():
            raise Mid360RayTimeBuilderError(
                "packet.xyz_m must contain only finite values."
            )
        point_times: np.ndarray | None = None
        if packet.point_timestamps_s is not None:
            point_times = _floating_array(
                packet.point_timestamps_s,
                (xyz.shape[0],),
                "packet.point_timestamps_s",
            )
            if not np.isfinite(point_times).all():
                raise Mid360RayTimeBuilderError(
                    "packet.point_timestamps_s must contain only finite values."
                )
            if np.any(point_times < start - self.timestamp_tolerance_s) or np.any(
                point_times > end + self.timestamp_tolerance_s
            ):
                raise Mid360RayTimeBuilderError(
                    "Every point timestamp must lie inside its declared "
                    "capture window."
                )
            # The tolerance absorbs only timestamp quantization/jitter at the
            # declared acquisition boundary.  Clipping here guarantees that a
            # later action-time subtraction cannot produce a negative age or a
            # return younger than its packet.
            point_times = np.clip(point_times, start, end)
        if packet.timestamp_semantics in (
            MID360_TIMESTAMP_LIVOX_CUSTOM_MSG,
            MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
        ) and packet.point_timestamps_s is None:
            raise Mid360RayTimeBuilderError(
                "The declared timestamp semantics require one verified "
                "timestamp per point."
            )
        if (
            packet.timestamp_semantics
            == MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY
            and packet.point_timestamps_s is not None
        ):
            raise Mid360RayTimeBuilderError(
                "Capture-window-only timestamp semantics cannot carry "
                "unverified per-point timestamps."
            )
        clock_domain = self._validate_clock_domain(
            packet.monotonic_clock_domain,
            context="ingest_point_packet",
        )
        if packet.capture_end_semantics not in (
            None,
            MID360_CAPTURE_END_ACQUISITION_WINDOW,
            MID360_CAPTURE_END_LATEST_RETURN_LEGACY,
        ):
            raise Mid360RayTimeBuilderError(
                "packet.capture_end_semantics is not recognized."
            )
        if (
            self.monotonic_clock_domain is not None
            and packet.capture_end_semantics
            != MID360_CAPTURE_END_ACQUISITION_WINDOW
        ):
            raise Mid360RayTimeBuilderError(
                "Strict event-time mode requires capture_end_s to be the "
                "explicit policy acquisition-window boundary, not the latest "
                "successful return."
            )
        if (
            self.monotonic_clock_domain is not None
            and xyz.shape[0] > 0
            and point_times is None
        ):
            raise Mid360RayTimeBuilderError(
                "Strict event-time mode requires one verified timestamp for "
                "every successful point return."
            )

        emitted = self._validate_emitted_mask(packet.emitted_sensor_mask)
        received, latency = self._validate_received_time(
            end,
            packet.received_time_s,
        )
        # Exact equality is intentional: this field records the checked
        # identity, not merely a descriptive packet label.
        if clock_domain != packet.monotonic_clock_domain:
            raise Mid360RayTimeBuilderError(
                "Normalized monotonic clock identity changed unexpectedly."
            )
        return xyz, point_times, emitted, received, latency

    def _validate_emitted_mask(
        self,
        emitted_sensor_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if emitted_sensor_mask is None:
            return None
        return _bool_array(
            emitted_sensor_mask,
            (self.image_height, self.image_width),
            "emitted_sensor_mask",
        )

    def _validate_received_time(
        self,
        capture_end_s: float,
        received_time_s: float | None,
    ) -> tuple[float | None, float | None]:
        if received_time_s is None:
            return None, None
        received = _finite_float(received_time_s, "received_time_s")
        latency = received - capture_end_s
        if latency < -self.timestamp_tolerance_s:
            raise Mid360RayTimeBuilderError(
                "received_time_s precedes capture_end_s; timestamps must share "
                "one monotonic clock domain."
            )
        return received, max(0.0, latency)

    def _validate_clock_domain(
        self,
        value: str | None,
        *,
        context: str,
    ) -> str | None:
        clock_domain = _optional_clock_domain(
            value,
            f"{context}.monotonic_clock_domain",
        )
        if self.monotonic_clock_domain is None:
            return clock_domain
        if clock_domain != self.monotonic_clock_domain:
            raise Mid360RayTimeBuilderError(
                f"{context} must use monotonic clock domain "
                f"{self.monotonic_clock_domain!r}; got {clock_domain!r}."
            )
        return clock_domain

    def _validate_sensor_return_timestamps(
        self,
        value: np.ndarray,
        requested_hits: np.ndarray,
        capture_start_s: float,
        capture_end_s: float,
    ) -> np.ndarray:
        timestamps = _floating_array(
            value,
            (self.image_height, self.image_width),
            "sensor_return_timestamps_s",
        )
        if not np.isfinite(timestamps).all():
            raise Mid360RayTimeBuilderError(
                "sensor_return_timestamps_s must contain only finite values."
            )
        if np.any(timestamps[~requested_hits] != 0.0):
            raise Mid360RayTimeBuilderError(
                "Bins without a successful return must have exactly zero "
                "sensor_return_timestamps_s; no-return is unknown, not "
                "observed free space."
            )
        hit_timestamps = timestamps[requested_hits]
        if np.any(
            hit_timestamps < capture_start_s - self.timestamp_tolerance_s
        ) or np.any(
            hit_timestamps > capture_end_s + self.timestamp_tolerance_s
        ):
            raise Mid360RayTimeBuilderError(
                "Every sensor-grid return timestamp must lie inside its "
                "declared capture window."
            )
        normalized = np.zeros_like(timestamps, dtype=np.float64)
        normalized[requested_hits] = np.clip(
            hit_timestamps,
            capture_start_s,
            capture_end_s,
        )
        return normalized

    def _sensor_grid_to_policy_order(self, value: np.ndarray) -> np.ndarray:
        """Apply the one manifest-bound flip/roll to any aligned sensor grid."""
        return np.roll(
            np.flip(value, axis=-1),
            shift=self.width_roll_bins,
            axis=-1,
        )

    def _sensor_timing_to_policy(
        self,
        sensor_return_timestamps_s: np.ndarray,
        sensor_return_timestamp_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        timestamps = _floating_array(
            sensor_return_timestamps_s,
            (self.image_height, self.image_width),
            "sensor_return_timestamps_s",
        )
        valid = _bool_array(
            sensor_return_timestamp_valid,
            (self.image_height, self.image_width),
            "sensor_return_timestamp_valid",
        )
        if not np.isfinite(timestamps).all():
            raise Mid360RayTimeBuilderError(
                "sensor return timestamps must contain only finite values."
            )
        if np.any(timestamps[~valid] != 0.0):
            raise Mid360RayTimeBuilderError(
                "Invalid sensor return timestamps must be exactly zero."
            )
        policy_timestamps = self._sensor_grid_to_policy_order(timestamps)
        policy_valid = self._sensor_grid_to_policy_order(valid)
        return (
            policy_timestamps.astype(np.float64, copy=False),
            policy_valid.astype(np.bool_, copy=False),
        )

    def _sensor_rows_from_elevation(
        self,
        elevation_deg: np.ndarray,
    ) -> np.ndarray:
        if self.image_height == 1:
            return np.zeros(elevation_deg.shape, dtype=np.int64)
        step = (
            self.sensor_row_first_deg - self.sensor_row_last_deg
        ) / (self.image_height - 1)
        row = np.floor(
            (self.sensor_row_first_deg - elevation_deg) / step + 0.5
        ).astype(np.int64)
        return np.clip(row, 0, self.image_height - 1)

    def _sensor_columns_from_azimuth(
        self,
        azimuth_deg: np.ndarray,
    ) -> np.ndarray:
        wrapped = np.mod(
            azimuth_deg - self.sensor_column_first_deg,
            360.0,
        )
        column = np.floor(
            wrapped / self.sensor_column_step_deg + 0.5
        ).astype(np.int64)
        return np.mod(column, self.image_width)

    def _append_packet(
        self,
        policy_packet: np.ndarray,
        stats: Mid360PacketStats,
        *,
        policy_return_timestamps_s: np.ndarray | None = None,
        policy_return_timestamp_valid: np.ndarray | None = None,
        frame_valid: bool = False,
    ) -> Mid360PacketStats:
        self._validate_policy_packet(policy_packet)
        if policy_return_timestamps_s is None:
            policy_return_timestamps_s = np.zeros(
                (self.image_height, self.image_width),
                dtype=np.float64,
            )
        if policy_return_timestamp_valid is None:
            policy_return_timestamp_valid = np.zeros(
                (self.image_height, self.image_width),
                dtype=np.bool_,
            )
        self._validate_timing_packet(
            policy_packet,
            policy_return_timestamps_s,
            policy_return_timestamp_valid,
            frame_valid,
        )
        timed_bins = int(np.count_nonzero(policy_return_timestamp_valid))
        if stats.timed_return_bins != timed_bins:
            raise Mid360RayTimeBuilderError(
                "Packet statistics do not match aligned timed-return bins."
            )
        if bool(stats.event_time_frame_valid) != bool(frame_valid):
            raise Mid360RayTimeBuilderError(
                "Packet statistics do not match event-time frame validity."
            )
        index = stats.window_index
        capture_end_s = stats.capture_end_s
        interval_error_s: float | None = None
        missing = 0
        if self._last_window_index is not None:
            if index <= self._last_window_index:
                raise Mid360RayTimeBuilderError(
                    "window_index must be strictly increasing; got "
                    f"{index} after {self._last_window_index}."
                )
            if (
                self._last_window_capture_end_s is None
                or capture_end_s
                <= self._last_window_capture_end_s + self.timestamp_tolerance_s
            ):
                raise Mid360RayTimeBuilderError(
                    "capture_end_s must increase strictly with window_index."
                )
            index_delta = index - self._last_window_index
            interval_error_s = (
                capture_end_s
                - self._last_window_capture_end_s
                - index_delta * self.packet_interval_s
            )
            if (
                abs(interval_error_s)
                > self.packet_cadence_tolerance_s
                + self.timestamp_tolerance_s
            ):
                raise Mid360RayTimeBuilderError(
                    "Packet capture cadence violates the manifest 0.1-second "
                    "grid: "
                    f"window_index delta={index_delta}, "
                    f"interval_error={interval_error_s:.6f}s, allowed="
                    f"{self.packet_cadence_tolerance_s:.6f}s."
                )
            missing = index_delta - 1
            self._insert_implicit_missing(index, missing)

        self._shift_append(
            policy_packet,
            index,
            capture_end_s,
            policy_return_timestamps_s,
            policy_return_timestamp_valid,
            frame_valid,
        )
        self._last_window_index = index
        self._last_window_capture_end_s = capture_end_s
        if not stats.explicit_drop:
            self._last_acquisition_capture_end_s = capture_end_s
        self._implicit_missing_packets_total += missing
        final_stats = replace(
            stats,
            implicit_missing_packets_inserted=missing,
            capture_interval_error_s=interval_error_s,
        )
        self._last_stats = final_stats
        self._validate_history()
        return final_stats

    def _insert_implicit_missing(self, current_index: int, count: int) -> None:
        if count <= 0:
            return
        unknown = np.zeros_like(self._history[0])
        unknown_timestamps = np.zeros_like(self._return_timestamps_s[0])
        unknown_timestamp_valid = np.zeros_like(
            self._return_timestamp_valid[0]
        )
        # Only the K-1 missing windows immediately preceding the new packet can
        # survive after that packet is appended.  Avoid work proportional to an
        # arbitrarily large driver outage.
        retained = min(count, self.history_length - 1)
        if count >= self.history_length:
            self._history.fill(0)
            self._window_indices.fill(-1)
            self._capture_end_times_s.fill(np.nan)
            self._return_timestamps_s.fill(0)
            self._return_timestamp_valid.fill(False)
            self._frame_valid.fill(False)
        first = current_index - retained
        for missing_index in range(first, current_index):
            self._shift_append(
                unknown,
                missing_index,
                np.nan,
                unknown_timestamps,
                unknown_timestamp_valid,
                False,
            )

    def _shift_append(
        self,
        packet: np.ndarray,
        window_index: int,
        capture_end_s: float,
        return_timestamps_s: np.ndarray,
        return_timestamp_valid: np.ndarray,
        frame_valid: bool,
    ) -> None:
        if self.history_length > 1:
            self._history[:-1] = self._history[1:]
            self._window_indices[:-1] = self._window_indices[1:]
            self._capture_end_times_s[:-1] = self._capture_end_times_s[1:]
            self._return_timestamps_s[:-1] = self._return_timestamps_s[1:]
            self._return_timestamp_valid[:-1] = (
                self._return_timestamp_valid[1:]
            )
            self._frame_valid[:-1] = self._frame_valid[1:]
        self._history[-1] = packet
        self._window_indices[-1] = window_index
        self._capture_end_times_s[-1] = capture_end_s
        self._return_timestamps_s[-1] = return_timestamps_s
        self._return_timestamp_valid[-1] = return_timestamp_valid
        self._frame_valid[-1] = frame_valid

    def _validate_freshness(self, now_s: float) -> float:
        now = _finite_float(now_s, "now_s")
        if self._last_acquisition_capture_end_s is None:
            raise StaleMid360PacketError(
                "No acquired MID-360 packet has been ingested since reset; "
                "explicitly dropped windows do not refresh sensor freshness."
            )
        age = now - self._last_acquisition_capture_end_s
        if age < -self.timestamp_tolerance_s:
            raise StaleMid360PacketError(
                "The newest MID-360 capture is in the future relative to now_s; "
                "timestamps must share one monotonic clock domain."
            )
        if age > self.max_packet_age_s + self.timestamp_tolerance_s:
            raise StaleMid360PacketError(
                f"Newest MID-360 packet is stale by {age:.6f}s; maximum "
                f"allowed age is {self.max_packet_age_s:.6f}s."
            )
        # Values inside tolerance describe the same acquisition boundary.  A
        # single normalization point prevents negative packet/return ages.
        return max(now, self._last_acquisition_capture_end_s)

    def _validate_capture_duration(
        self,
        capture_start_s: float,
        capture_end_s: float,
    ) -> None:
        duration = capture_end_s - capture_start_s
        maximum = self.packet_interval_s + self.packet_cadence_tolerance_s
        if duration > maximum + self.timestamp_tolerance_s:
            raise Mid360RayTimeBuilderError(
                "Capture window exceeds the manifest packet interval: "
                f"duration={duration:.6f}s, maximum={maximum:.6f}s."
            )

    def _safety_stats_fields(self) -> dict[str, float]:
        return {
            "max_packet_age_s": self.max_packet_age_s,
            "packet_cadence_tolerance_s": self.packet_cadence_tolerance_s,
            "timestamp_tolerance_s": self.timestamp_tolerance_s,
        }

    def _validate_policy_packet(self, packet: np.ndarray) -> None:
        expected = (2, self.image_height, self.image_width)
        if not isinstance(packet, np.ndarray) or packet.shape != expected:
            raise Mid360RayTimeBuilderError(
                f"policy packet must have shape {expected}."
            )
        if packet.dtype != np.float16:
            raise Mid360RayTimeBuilderError(
                "policy packet must have dtype float16."
            )
        if not np.isfinite(packet).all():
            raise Mid360RayTimeBuilderError(
                "policy packet must contain only finite values."
            )
        ranges = packet[0]
        hits = packet[1]
        if not np.all((hits == 0) | (hits == 1)):
            raise Mid360RayTimeBuilderError(
                "policy hit_mask must be exactly binary."
            )
        hit_bool = hits.astype(np.bool_)
        if not np.all(ranges[~hit_bool] == 0):
            raise Mid360RayTimeBuilderError(
                "Unknown policy rays must have exactly zero range."
            )
        if np.any(ranges[hit_bool] < self.min_range_m) or np.any(
            ranges[hit_bool] > self.max_range_m
        ):
            raise Mid360RayTimeBuilderError(
                "Observed policy ranges violate the manifest valid range."
            )

    def _validate_timing_packet(
        self,
        policy_packet: np.ndarray,
        return_timestamps_s: np.ndarray,
        return_timestamp_valid: np.ndarray,
        frame_valid: bool,
    ) -> None:
        timestamps = _floating_array(
            return_timestamps_s,
            (self.image_height, self.image_width),
            "policy_return_timestamps_s",
        )
        valid = _bool_array(
            return_timestamp_valid,
            (self.image_height, self.image_width),
            "policy_return_timestamp_valid",
        )
        if not isinstance(frame_valid, (bool, np.bool_)):
            raise TypeError("frame_valid must be boolean.")
        if not np.isfinite(timestamps).all():
            raise Mid360RayTimeBuilderError(
                "Policy return timestamps must contain only finite values."
            )
        if np.any(timestamps[~valid] != 0.0):
            raise Mid360RayTimeBuilderError(
                "Invalid policy return timestamps must be exactly zero."
            )
        hits = policy_packet[1].astype(np.bool_)
        if np.any(valid & ~hits):
            raise Mid360RayTimeBuilderError(
                "A timed return must be the winning observed range in its bin."
            )
        if self.monotonic_clock_domain is not None and not np.array_equal(
            valid,
            hits,
        ):
            raise Mid360RayTimeBuilderError(
                "Strict event-time mode requires range and timestamp validity "
                "to be bitwise identical."
            )
        if bool(frame_valid) != bool(np.any(valid)):
            raise Mid360RayTimeBuilderError(
                "frame_valid must be true exactly when at least one timed "
                "winning return exists."
            )

    def _validate_history(self) -> None:
        expected = (
            self.history_length,
            2,
            self.image_height,
            self.image_width,
        )
        if self._history.shape != expected or self._history.dtype != np.float16:
            raise Mid360RayTimeBuilderError(
                "Internal ray history no longer matches the manifest."
            )
        if not np.isfinite(self._history).all():
            raise Mid360RayTimeBuilderError(
                "Internal ray history contains a non-finite value."
            )
        hits = self._history[:, 1]
        if not np.all((hits == 0) | (hits == 1)):
            raise Mid360RayTimeBuilderError(
                "Internal ray history hit masks are not binary."
            )
        if not np.all(self._history[:, 0][hits == 0] == 0):
            raise Mid360RayTimeBuilderError(
                "Internal unknown rays do not have exactly zero range."
            )
        timing_shape = (
            self.history_length,
            self.image_height,
            self.image_width,
        )
        if (
            self._return_timestamps_s.shape != timing_shape
            or self._return_timestamps_s.dtype != np.float64
            or self._return_timestamp_valid.shape != timing_shape
            or self._return_timestamp_valid.dtype != np.bool_
            or self._frame_valid.shape != (self.history_length,)
            or self._frame_valid.dtype != np.bool_
        ):
            raise Mid360RayTimeBuilderError(
                "Internal event-time history has an invalid shape or dtype."
            )
        if not np.isfinite(self._return_timestamps_s).all():
            raise Mid360RayTimeBuilderError(
                "Internal event-time history contains a non-finite timestamp."
            )
        if np.any(
            self._return_timestamps_s[~self._return_timestamp_valid] != 0.0
        ):
            raise Mid360RayTimeBuilderError(
                "Internal invalid return timestamps are not exactly zero."
            )
        hit_bool = hits.astype(np.bool_)
        if np.any(self._return_timestamp_valid & ~hit_bool):
            raise Mid360RayTimeBuilderError(
                "Internal timed-return validity exceeds observed range support."
            )
        if self.monotonic_clock_domain is not None and not np.array_equal(
            self._return_timestamp_valid,
            hit_bool,
        ):
            raise Mid360RayTimeBuilderError(
                "Strict event-time history lost range/timestamp alignment."
            )
        expected_frame_valid = np.any(
            self._return_timestamp_valid,
            axis=(1, 2),
        )
        if not np.array_equal(self._frame_valid, expected_frame_valid):
            raise Mid360RayTimeBuilderError(
                "Internal frame validity disagrees with timed return support."
            )


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(result):
        raise Mid360RayTimeBuilderError(f"{name} must be finite.")
    return result


def _positive_finite_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise Mid360RayTimeBuilderError(f"{name} must be positive.")
    return result


def _nonnegative_finite_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0:
        raise Mid360RayTimeBuilderError(f"{name} must be non-negative.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(
        value,
        (bool, np.bool_),
    ):
        raise TypeError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise Mid360RayTimeBuilderError(f"{name} must be non-negative.")
    return result


def _optional_clock_domain(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if not value or value != value.strip() or not value.isprintable():
        raise Mid360RayTimeBuilderError(
            f"{name} must be a non-empty printable identifier without "
            "leading or trailing whitespace."
        )
    return value


def _floating_matrix(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise Mid360RayTimeBuilderError(
            f"{name} must have shape [N, 3], got {value.shape}."
        )
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating-point dtype.")
    return value


def _floating_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if value.shape != shape:
        raise Mid360RayTimeBuilderError(
            f"{name} must have shape {shape}, got {value.shape}."
        )
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating-point dtype.")
    return value


def _bool_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if value.shape != shape:
        raise Mid360RayTimeBuilderError(
            f"{name} must have shape {shape}, got {value.shape}."
        )
    if value.dtype != np.bool_:
        raise TypeError(f"{name} must use dtype bool.")
    return value
