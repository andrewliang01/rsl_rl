# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed ROS message adapters for the real MID-360 tensor builder.

The module deliberately imports neither ROS 1 nor ROS 2.  It accepts the
public attribute layout of ``livox_ros_driver2/CustomMsg`` and
``sensor_msgs/PointCloud2`` so the numerical extraction can be unit-tested on
machines without ROS.  A deployment callback remains responsible for assigning
one monotonically increasing policy-window index and for presenting capture
times in the explicitly named action-clock domain.

Only two PointCloud2 timing modes are admitted:

* no per-point field, producing a range-only capture-window packet; or
* the Livox ``PointXYZRTLT`` ``float64 timestamp`` field, interpreted as
  absolute seconds in the caller-declared clock domain.

No header time, field unit, coordinate transform, or missing return is guessed.
"""

from __future__ import annotations

import math
import numpy as np
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .mid360_ray_time_builder import (
    MID360_CAPTURE_END_ACQUISITION_WINDOW,
    MID360_NORMALIZED_SENSOR_FRAME,
    MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
    MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
    Mid360PointPacket,
    Mid360RayTimeBuilderError,
    point_packet_from_livox_custom_msg_arrays,
)

POINT_FIELD_INT8 = 1
POINT_FIELD_UINT8 = 2
POINT_FIELD_INT16 = 3
POINT_FIELD_UINT16 = 4
POINT_FIELD_INT32 = 5
POINT_FIELD_UINT32 = 6
POINT_FIELD_FLOAT32 = 7
POINT_FIELD_FLOAT64 = 8

LIVOX_POINTCLOUD2_TIMESTAMP_FIELD = "timestamp"

_UINT32_MAX = int(np.iinfo(np.uint32).max)
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_POINT_FIELD_FORMATS = {
    POINT_FIELD_INT8: ("b", 1),
    POINT_FIELD_UINT8: ("B", 1),
    POINT_FIELD_INT16: ("h", 2),
    POINT_FIELD_UINT16: ("H", 2),
    POINT_FIELD_INT32: ("i", 4),
    POINT_FIELD_UINT32: ("I", 4),
    POINT_FIELD_FLOAT32: ("f", 4),
    POINT_FIELD_FLOAT64: ("d", 8),
}


class Mid360RosAdapterError(Mid360RayTimeBuilderError):
    """Raised when a ROS message cannot satisfy the frozen sensor contract."""


@dataclass(frozen=True)
class PointCloud2Extraction:
    """Byte-accounted PointCloud2 fields before packet construction."""

    xyz_m: np.ndarray
    point_timestamps_s: np.ndarray | None
    point_count: int
    data_bytes: int
    row_padding_bytes: int
    frame_id: str
    is_bigendian: bool


@dataclass(frozen=True)
class LivoxCustomMsgExtraction:
    """All successful CustomMsg returns in their native sensor clock."""

    xyz_m: np.ndarray
    timebase_ns: int
    offset_time_ns: np.ndarray
    point_count: int
    frame_id: str


def extract_livox_custom_msg(
    message: Any,
    *,
    expected_frame_id: str,
) -> LivoxCustomMsgExtraction:
    """Extract every CustomMsg return without assigning an action clock."""
    frame_id = _require_frame_id(message, expected_frame_id)
    timebase_ns = _bounded_int(
        _attribute(message, "timebase"),
        "CustomMsg.timebase",
        maximum=_UINT64_MAX,
    )
    declared_count = _bounded_int(
        _attribute(message, "point_num"),
        "CustomMsg.point_num",
        maximum=_UINT32_MAX,
    )
    raw_points = _attribute(message, "points")
    if isinstance(raw_points, (str, bytes, bytearray, memoryview)):
        raise Mid360RosAdapterError("CustomMsg.points must be a point sequence.")
    try:
        points = list(raw_points)
    except TypeError as error:
        raise Mid360RosAdapterError("CustomMsg.points must be an iterable point sequence.") from error
    if len(points) != declared_count:
        raise Mid360RosAdapterError(
            "CustomMsg.point_num does not equal the unsliced points length: "
            f"declared={declared_count}, actual={len(points)}."
        )

    xyz_m = np.empty((declared_count, 3), dtype=np.float32)
    offset_time_ns = np.empty((declared_count,), dtype=np.uint32)
    for index, point in enumerate(points):
        xyz_m[index] = (
            _finite_number(_attribute(point, "x"), f"points[{index}].x"),
            _finite_number(_attribute(point, "y"), f"points[{index}].y"),
            _finite_number(_attribute(point, "z"), f"points[{index}].z"),
        )
        offset_time_ns[index] = _bounded_int(
            _attribute(point, "offset_time"),
            f"points[{index}].offset_time",
            maximum=_UINT32_MAX,
        )
    maximum_offset_ns = 0 if offset_time_ns.size == 0 else int(offset_time_ns.max())
    if timebase_ns > _UINT64_MAX - maximum_offset_ns:
        raise Mid360RosAdapterError("CustomMsg.timebase + offset_time overflows uint64.")
    return LivoxCustomMsgExtraction(
        xyz_m=xyz_m,
        timebase_ns=timebase_ns,
        offset_time_ns=offset_time_ns,
        point_count=declared_count,
        frame_id=frame_id,
    )


def livox_custom_msg_to_mid360_packet(
    message: Any,
    *,
    expected_frame_id: str,
    window_index: int,
    capture_end_s: float,
    received_time_s: float,
    monotonic_clock_domain: str,
) -> Mid360PointPacket:
    """Convert one native Livox ``CustomMsg`` without dropping any point.

    ``capture_end_s`` and ``received_time_s`` must share the same numeric clock
    domain as ``timebase + offset_time``.  This is an explicit deployment
    assertion; the adapter never substitutes ``header.stamp``.
    """
    end = _finite_number(capture_end_s, "capture_end_s")
    received = _finite_number(received_time_s, "received_time_s")
    if received < end:
        raise Mid360RosAdapterError("received_time_s cannot precede capture_end_s.")
    extracted = extract_livox_custom_msg(
        message,
        expected_frame_id=expected_frame_id,
    )

    return point_packet_from_livox_custom_msg_arrays(
        xyz_m=extracted.xyz_m,
        timebase_ns=extracted.timebase_ns,
        offset_time_ns=extracted.offset_time_ns,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        window_index=_bounded_int(
            window_index,
            "window_index",
            maximum=_INT64_MAX,
        ),
        capture_end_s=end,
        received_time_s=received,
        monotonic_clock_domain=_nonempty_string(
            monotonic_clock_domain,
            "monotonic_clock_domain",
        ),
    )


def livox_custom_msg_to_sensor_clock_packet(
    message: Any,
    *,
    expected_frame_id: str,
    window_index: int,
    capture_end_sensor_s: float,
    sensor_clock_domain: str,
) -> Mid360PointPacket:
    """Build a pre-alignment CustomMsg packet with no mixed receive time.

    This is the only CustomMsg adapter intended for a later cross-clock
    mapping.  ``received_time_s`` remains ``None`` until the caller applies a
    validated action-clock alignment and supplies the callback receive time in
    that action clock.
    """
    extracted = extract_livox_custom_msg(
        message,
        expected_frame_id=expected_frame_id,
    )
    return point_packet_from_livox_custom_msg_arrays(
        xyz_m=extracted.xyz_m,
        timebase_ns=extracted.timebase_ns,
        offset_time_ns=extracted.offset_time_ns,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        window_index=_bounded_int(
            window_index,
            "window_index",
            maximum=_INT64_MAX,
        ),
        capture_end_s=_finite_number(
            capture_end_sensor_s,
            "capture_end_sensor_s",
        ),
        received_time_s=None,
        monotonic_clock_domain=_nonempty_string(
            sensor_clock_domain,
            "sensor_clock_domain",
        ),
    )


def extract_livox_pointcloud2(
    message: Any,
    *,
    expected_frame_id: str,
    use_livox_point_timestamps: bool,
    reject_constant_point_timestamps: bool = True,
) -> PointCloud2Extraction:
    """Extract x/y/z and optional PointXYZRTLT timestamps from PointCloud2.

    The extractor handles organized clouds, row padding and both byte orders.
    It requires the official Livox field types: x/y/z are scalar float32 and
    ``timestamp`` is scalar float64.  Every point record is retained; a cloud
    containing NaN/Inf is rejected instead of being silently filtered.
    """
    use_point_times = _strict_bool(
        use_livox_point_timestamps,
        "use_livox_point_timestamps",
    )
    reject_constant_times = _strict_bool(
        reject_constant_point_timestamps,
        "reject_constant_point_timestamps",
    )
    frame_id = _require_frame_id(message, expected_frame_id)
    height = _positive_int(_attribute(message, "height"), "PointCloud2.height")
    width = _positive_int(_attribute(message, "width"), "PointCloud2.width")
    point_step = _positive_int(_attribute(message, "point_step"), "PointCloud2.point_step")
    row_step = _positive_int(_attribute(message, "row_step"), "PointCloud2.row_step")
    minimum_row_step = width * point_step
    if row_step < minimum_row_step:
        raise Mid360RosAdapterError("PointCloud2.row_step is smaller than width * point_step.")
    is_bigendian = _strict_bool(_attribute(message, "is_bigendian"), "PointCloud2.is_bigendian")
    _strict_bool(_attribute(message, "is_dense"), "PointCloud2.is_dense")
    data = _byte_view(_attribute(message, "data"), "PointCloud2.data")
    required_bytes = height * row_step
    if len(data) != required_bytes:
        raise Mid360RosAdapterError(
            f"PointCloud2.data byte length must equal height * row_step: expected={required_bytes}, actual={len(data)}."
        )

    fields = _field_table(_attribute(message, "fields"), point_step=point_step)
    required_names = {"x", "y", "z"}
    if use_point_times:
        required_names.add(LIVOX_POINTCLOUD2_TIMESTAMP_FIELD)
    missing = sorted(required_names - set(fields))
    if missing:
        raise Mid360RosAdapterError(f"PointCloud2 is missing required fields: {missing}.")
    for name in ("x", "y", "z"):
        _require_field_layout(fields[name], name=name, datatype=POINT_FIELD_FLOAT32)
    if use_point_times:
        _require_field_layout(
            fields[LIVOX_POINTCLOUD2_TIMESTAMP_FIELD],
            name=LIVOX_POINTCLOUD2_TIMESTAMP_FIELD,
            datatype=POINT_FIELD_FLOAT64,
        )

    point_count = height * width
    xyz_m = np.empty((point_count, 3), dtype=np.float32)
    point_timestamps_s = np.empty((point_count,), dtype=np.float64) if use_point_times else None
    endian = ">" if is_bigendian else "<"
    xyz_unpackers = tuple(struct.Struct(endian + "f") for _ in range(3))
    timestamp_unpacker = struct.Struct(endian + "d")
    cursor = 0
    for row in range(height):
        row_base = row * row_step
        for column in range(width):
            point_base = row_base + column * point_step
            for axis, name in enumerate(("x", "y", "z")):
                xyz_m[cursor, axis] = xyz_unpackers[axis].unpack_from(
                    data,
                    point_base + fields[name][0],
                )[0]
            if point_timestamps_s is not None:
                point_timestamps_s[cursor] = timestamp_unpacker.unpack_from(
                    data,
                    point_base + fields[LIVOX_POINTCLOUD2_TIMESTAMP_FIELD][0],
                )[0]
            cursor += 1

    if not np.isfinite(xyz_m).all():
        raise Mid360RosAdapterError("PointCloud2 x/y/z contains NaN or Inf; silent point filtering is forbidden.")
    if point_timestamps_s is not None:
        if not np.isfinite(point_timestamps_s).all():
            raise Mid360RosAdapterError("PointCloud2 timestamp contains NaN or Inf.")
        if np.any(np.diff(point_timestamps_s) < 0.0):
            raise Mid360RosAdapterError("PointCloud2 timestamps must be nondecreasing in message order.")
        if reject_constant_times and point_count > 1 and float(np.ptp(point_timestamps_s)) == 0.0:
            raise Mid360RosAdapterError(
                "PointCloud2 per-point timestamps are constant; this cannot prove "
                "Livox acquisition timing. Use CustomMsg or explicitly disable "
                "per-point timing for range-only deployment."
            )

    return PointCloud2Extraction(
        xyz_m=xyz_m,
        point_timestamps_s=point_timestamps_s,
        point_count=point_count,
        data_bytes=len(data),
        row_padding_bytes=height * (row_step - minimum_row_step),
        frame_id=frame_id,
        is_bigendian=is_bigendian,
    )


def livox_pointcloud2_to_mid360_packet(
    message: Any,
    *,
    expected_frame_id: str,
    window_index: int,
    capture_start_s: float,
    capture_end_s: float,
    received_time_s: float,
    monotonic_clock_domain: str | None,
    use_livox_point_timestamps: bool,
    reject_constant_point_timestamps: bool = True,
) -> Mid360PointPacket:
    """Convert official Livox PointXYZRTLT or range-only PointCloud2.

    ``use_livox_point_timestamps=True`` is valid only for the official scalar
    ``float64 timestamp`` field in absolute seconds.  ``False`` intentionally
    discards no field: it declares that per-return time is unavailable and
    produces a capture-window-only packet, which strict H2 builders reject.
    """
    use_point_times = _strict_bool(
        use_livox_point_timestamps,
        "use_livox_point_timestamps",
    )
    reject_constant_times = _strict_bool(
        reject_constant_point_timestamps,
        "reject_constant_point_timestamps",
    )
    start = _finite_number(capture_start_s, "capture_start_s")
    end = _finite_number(capture_end_s, "capture_end_s")
    received = _finite_number(received_time_s, "received_time_s")
    if end < start:
        raise Mid360RosAdapterError("capture_end_s cannot precede capture_start_s.")
    if received < end:
        raise Mid360RosAdapterError("received_time_s cannot precede capture_end_s.")
    clock = (
        _nonempty_string(monotonic_clock_domain, "monotonic_clock_domain")
        if monotonic_clock_domain is not None
        else None
    )
    if use_point_times and clock is None:
        raise Mid360RosAdapterError("Livox PointXYZRTLT timestamps require an explicit monotonic_clock_domain.")

    extracted = extract_livox_pointcloud2(
        message,
        expected_frame_id=expected_frame_id,
        use_livox_point_timestamps=use_point_times,
        reject_constant_point_timestamps=reject_constant_times,
    )
    timestamps = extracted.point_timestamps_s
    if timestamps is not None and timestamps.size:
        tolerance = 1.0e-6
        if timestamps[0] < start - tolerance or timestamps[-1] > end + tolerance:
            raise Mid360RosAdapterError("PointCloud2 point timestamps fall outside the declared capture window.")
    return Mid360PointPacket(
        xyz_m=extracted.xyz_m,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        timestamp_semantics=(
            MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS if timestamps is not None else MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY
        ),
        window_index=_bounded_int(window_index, "window_index", maximum=_INT64_MAX),
        capture_start_s=start,
        capture_end_s=end,
        received_time_s=received,
        point_timestamps_s=timestamps,
        monotonic_clock_domain=clock,
        capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
    )


def _field_table(raw_fields: Any, *, point_step: int) -> dict[str, tuple[int, int, int]]:
    if isinstance(raw_fields, (str, bytes, bytearray, memoryview)):
        raise Mid360RosAdapterError("PointCloud2.fields must be a field sequence.")
    try:
        values: Iterable[Any] = list(raw_fields)
    except TypeError as error:
        raise Mid360RosAdapterError("PointCloud2.fields must be an iterable field sequence.") from error
    table: dict[str, tuple[int, int, int]] = {}
    occupied: set[int] = set()
    for index, field in enumerate(values):
        name = _nonempty_string(_attribute(field, "name"), f"fields[{index}].name")
        if name in table:
            raise Mid360RosAdapterError(f"PointCloud2 field is duplicated: {name}.")
        offset = _bounded_int(
            _attribute(field, "offset"),
            f"fields[{index}].offset",
            maximum=point_step - 1,
        )
        datatype = _bounded_int(
            _attribute(field, "datatype"),
            f"fields[{index}].datatype",
            maximum=POINT_FIELD_FLOAT64,
        )
        if datatype not in _POINT_FIELD_FORMATS:
            raise Mid360RosAdapterError(f"PointCloud2 field {name} uses unsupported datatype {datatype}.")
        count = _positive_int(_attribute(field, "count"), f"fields[{index}].count")
        size = _POINT_FIELD_FORMATS[datatype][1] * count
        if offset + size > point_step:
            raise Mid360RosAdapterError(f"PointCloud2 field {name} exceeds point_step.")
        field_bytes = set(range(offset, offset + size))
        if occupied & field_bytes:
            raise Mid360RosAdapterError(f"PointCloud2 field {name} overlaps another declared field.")
        occupied.update(field_bytes)
        table[name] = (offset, datatype, count)
    return table


def _require_field_layout(record: tuple[int, int, int], *, name: str, datatype: int) -> None:
    if record[1] != datatype or record[2] != 1:
        raise Mid360RosAdapterError(f"PointCloud2 field {name} must be scalar datatype {datatype}.")


def _require_frame_id(message: Any, expected_frame_id: str) -> str:
    expected = _nonempty_string(expected_frame_id, "expected_frame_id")
    header = _attribute(message, "header")
    actual = _nonempty_string(_attribute(header, "frame_id"), "header.frame_id")
    if actual != expected:
        raise Mid360RosAdapterError(f"ROS frame_id differs from calibrated sensor frame: {actual!r} != {expected!r}.")
    return actual


def _attribute(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise Mid360RosAdapterError(f"ROS message is missing attribute {name}.") from error


def _byte_view(value: Any, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise Mid360RosAdapterError(f"{name} must expose a contiguous byte buffer.") from error
    if not view.c_contiguous or view.itemsize != 1:
        raise Mid360RosAdapterError(f"{name} must be a C-contiguous byte buffer.")
    return view.cast("B")


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise Mid360RosAdapterError(f"{name} must be bool.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _bounded_int(value, name, maximum=_UINT64_MAX)
    if result == 0:
        raise Mid360RosAdapterError(f"{name} must be positive.")
    return result


def _bounded_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise Mid360RosAdapterError(f"{name} must be an integer.")
    result = int(value)
    if result < 0 or result > maximum:
        raise Mid360RosAdapterError(f"{name} is outside [0, {maximum}].")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise Mid360RosAdapterError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise Mid360RosAdapterError(f"{name} must be finite.")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or not value.isprintable():
        raise Mid360RosAdapterError(f"{name} must be a non-empty printable stripped string.")
    return value


__all__ = [
    "LIVOX_POINTCLOUD2_TIMESTAMP_FIELD",
    "LivoxCustomMsgExtraction",
    "Mid360RosAdapterError",
    "PointCloud2Extraction",
    "extract_livox_custom_msg",
    "extract_livox_pointcloud2",
    "livox_custom_msg_to_mid360_packet",
    "livox_custom_msg_to_sensor_clock_packet",
    "livox_pointcloud2_to_mid360_packet",
]
