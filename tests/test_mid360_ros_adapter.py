from __future__ import annotations

import numpy as np
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from rsl_rl.utils.mid360_ray_time_builder import (
    MID360_NORMALIZED_SENSOR_FRAME,
    MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
    MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
    Mid360RayTimeTensorBuilder,
)
from rsl_rl.utils.mid360_ros_adapter import (
    POINT_FIELD_FLOAT32,
    POINT_FIELD_FLOAT64,
    POINT_FIELD_UINT8,
    Mid360RosAdapterError,
    extract_livox_pointcloud2,
    livox_custom_msg_to_mid360_packet,
    livox_pointcloud2_to_mid360_packet,
)
from tests.test_ray_time_deployment_manifest import _manifest


@dataclass
class Header:
    """Minimal ROS header fixture."""

    frame_id: str


@dataclass
class CustomPoint:
    """Minimal Livox CustomPoint fixture."""

    offset_time: int
    x: float
    y: float
    z: float


@dataclass
class CustomMsg:
    """Minimal Livox CustomMsg fixture."""

    header: Header
    timebase: int
    point_num: int
    points: list[CustomPoint]


@dataclass
class PointField:
    """Minimal sensor_msgs PointField fixture."""

    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass
class PointCloud2:
    """Minimal sensor_msgs PointCloud2 fixture."""

    header: Header
    height: int
    width: int
    fields: list[PointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool = True


def _custom_msg() -> CustomMsg:
    return CustomMsg(
        header=Header("livox_frame"),
        timebase=10_000_000_000,
        point_num=2,
        points=[
            CustomPoint(0, 1.0, 0.0, 0.0),
            CustomPoint(25_000_000, 2.0, 0.0, 0.0),
        ],
    )


def _pointcloud2(
    *,
    bigendian: bool = False,
    timestamps: tuple[float, ...] = (10.01, 10.02, 10.03, 10.04),
    xyz: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    ),
) -> PointCloud2:
    fields = [
        PointField("x", 0, POINT_FIELD_FLOAT32),
        PointField("y", 4, POINT_FIELD_FLOAT32),
        PointField("z", 8, POINT_FIELD_FLOAT32),
        PointField("intensity", 12, POINT_FIELD_FLOAT32),
        PointField("tag", 16, POINT_FIELD_UINT8),
        PointField("line", 17, POINT_FIELD_UINT8),
        PointField("timestamp", 18, POINT_FIELD_FLOAT64),
    ]
    point_step = 26
    width = 2
    height = 2
    row_step = width * point_step + 4
    payload = bytearray(height * row_step)
    endian = ">" if bigendian else "<"
    for index, (point, timestamp) in enumerate(zip(xyz, timestamps, strict=True)):
        row, column = divmod(index, width)
        base = row * row_step + column * point_step
        struct.pack_into(endian + "ffffBBd", payload, base, *point, 42.0, 3, 1, timestamp)
    return PointCloud2(
        header=Header("livox_frame"),
        height=height,
        width=width,
        fields=fields,
        is_bigendian=bigendian,
        point_step=point_step,
        row_step=row_step,
        data=bytes(payload),
    )


def _builder(tmp_path: Path, history_length: int = 1) -> Mid360RayTimeTensorBuilder:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return Mid360RayTimeTensorBuilder(
        _manifest(checkpoint, history_length=history_length, variant="Global"),
        max_packet_age_s=0.5,
        monotonic_clock_domain="PTP_TAI:mid360-g1",
    )


def test_custom_msg_preserves_all_points_offsets_and_explicit_clock() -> None:
    """Preserve every return and its Livox acquisition offset."""
    packet = livox_custom_msg_to_mid360_packet(
        _custom_msg(),
        expected_frame_id="livox_frame",
        window_index=7,
        capture_end_s=10.1,
        received_time_s=10.105,
        monotonic_clock_domain="PTP_TAI:mid360-g1",
    )

    assert packet.coordinate_frame == MID360_NORMALIZED_SENSOR_FRAME
    assert packet.timestamp_semantics != MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY
    assert packet.window_index == 7
    assert packet.capture_start_s == 10.0
    assert packet.capture_end_s == 10.1
    assert packet.received_time_s == 10.105
    assert packet.monotonic_clock_domain == "PTP_TAI:mid360-g1"
    np.testing.assert_array_equal(packet.xyz_m[:, 0], np.asarray((1.0, 2.0)))
    np.testing.assert_allclose(packet.point_timestamps_s, (10.0, 10.025))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: setattr(value, "point_num", 1), "point_num"),
        (lambda value: setattr(value.header, "frame_id", "base"), "frame_id"),
        (lambda value: setattr(value.points[0], "offset_time", -1), "outside"),
        (lambda value: setattr(value.points[0], "x", float("nan")), "finite"),
        (lambda value: setattr(value, "timebase", True), "integer"),
    ],
)
def test_custom_msg_rejects_count_frame_numeric_and_type_drift(
    mutation: Callable[[CustomMsg], None], message: str
) -> None:
    """Reject count, frame, numeric and scalar-type drift."""
    raw = _custom_msg()
    mutation(raw)
    with pytest.raises(Mid360RosAdapterError, match=message):
        livox_custom_msg_to_mid360_packet(
            raw,
            expected_frame_id="livox_frame",
            window_index=0,
            capture_end_s=10.1,
            received_time_s=10.11,
            monotonic_clock_domain="PTP_TAI:mid360-g1",
        )


def test_custom_msg_rejects_receive_before_capture_end() -> None:
    """Reject an impossible negative transport latency."""
    with pytest.raises(Mid360RosAdapterError, match="received_time_s"):
        livox_custom_msg_to_mid360_packet(
            _custom_msg(),
            expected_frame_id="livox_frame",
            window_index=0,
            capture_end_s=10.1,
            received_time_s=10.09,
            monotonic_clock_domain="PTP_TAI:mid360-g1",
        )


def test_custom_msg_rejects_timebase_offset_uint64_overflow() -> None:
    """Reject a wrapped CustomMsg per-return acquisition timestamp."""
    raw = _custom_msg()
    raw.timebase = int(np.iinfo(np.uint64).max)
    raw.points[1].offset_time = 1
    with pytest.raises(Mid360RosAdapterError, match="overflows uint64"):
        livox_custom_msg_to_mid360_packet(
            raw,
            expected_frame_id="livox_frame",
            window_index=0,
            capture_end_s=float(raw.timebase) * 1.0e-9,
            received_time_s=float(raw.timebase) * 1.0e-9,
            monotonic_clock_domain="PTP_TAI:mid360-g1",
        )


@pytest.mark.parametrize("bigendian", (False, True))
def test_pointcloud2_extracts_official_fields_row_padding_and_byte_order(
    bigendian: bool,
) -> None:
    """Handle official fields, organized padding and both byte orders."""
    extracted = extract_livox_pointcloud2(
        _pointcloud2(bigendian=bigendian),
        expected_frame_id="livox_frame",
        use_livox_point_timestamps=True,
    )

    assert extracted.point_count == 4
    assert extracted.data_bytes == 112
    assert extracted.row_padding_bytes == 8
    assert extracted.is_bigendian is bigendian
    np.testing.assert_allclose(extracted.xyz_m[:, 0], (1.0, 2.0, 3.0, 4.0))
    np.testing.assert_allclose(extracted.point_timestamps_s, (10.01, 10.02, 10.03, 10.04))


def test_pointcloud2_timed_and_range_only_packets_have_distinct_semantics() -> None:
    """Never promote a range-only cloud into per-return timing evidence."""
    timed = livox_pointcloud2_to_mid360_packet(
        _pointcloud2(),
        expected_frame_id="livox_frame",
        window_index=3,
        capture_start_s=10.0,
        capture_end_s=10.1,
        received_time_s=10.11,
        monotonic_clock_domain="PTP_TAI:mid360-g1",
        use_livox_point_timestamps=True,
    )
    range_only = livox_pointcloud2_to_mid360_packet(
        _pointcloud2(),
        expected_frame_id="livox_frame",
        window_index=3,
        capture_start_s=10.0,
        capture_end_s=10.1,
        received_time_s=10.11,
        monotonic_clock_domain=None,
        use_livox_point_timestamps=False,
    )

    assert timed.timestamp_semantics == MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS
    assert timed.point_timestamps_s is not None
    assert range_only.timestamp_semantics == MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY
    assert range_only.point_timestamps_s is None


def test_pointcloud2_adapter_integrates_with_manifest_bound_tensor_builder(tmp_path: Path) -> None:
    """Feed an adapted packet through the frozen production builder."""
    packet = livox_pointcloud2_to_mid360_packet(
        _pointcloud2(),
        expected_frame_id="livox_frame",
        window_index=0,
        capture_start_s=10.0,
        capture_end_s=10.1,
        received_time_s=10.11,
        monotonic_clock_domain="PTP_TAI:mid360-g1",
        use_livox_point_timestamps=True,
    )
    builder = _builder(tmp_path)
    stats = builder.ingest_point_packet(packet)
    policy = builder.policy_tensor(now_s=10.12)
    aligned = builder.aligned_event_time_history(
        now_s=10.12,
        monotonic_clock_domain="PTP_TAI:mid360-g1",
    )

    assert policy.shape == (1, 1, 2, 16, 96)
    assert policy.dtype == np.float16
    assert stats.input_return_points == 4
    assert stats.accepted_return_points == 4
    assert stats.timed_return_bins > 0
    assert np.array_equal(aligned.return_valid, policy[0, :, 1].astype(np.bool_))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: setattr(value.header, "frame_id", "wrong"), "frame_id"),
        (lambda value: setattr(value, "row_step", 1), "row_step"),
        (lambda value: setattr(value, "data", value.data[:-1]), "byte length"),
        (lambda value: setattr(value.fields[0], "datatype", POINT_FIELD_FLOAT64), "overlaps"),
        (lambda value: setattr(value.fields[-1], "count", 2), "exceeds point_step"),
    ],
)
def test_pointcloud2_rejects_layout_and_frame_drift(mutation: Callable[[PointCloud2], None], message: str) -> None:
    """Reject malformed layouts and uncalibrated frame identities."""
    raw = _pointcloud2()
    mutation(raw)
    with pytest.raises(Mid360RosAdapterError, match=message):
        extract_livox_pointcloud2(
            raw,
            expected_frame_id="livox_frame",
            use_livox_point_timestamps=True,
        )


def test_pointcloud2_rejects_nonfinite_reordered_or_constant_timestamps() -> None:
    """Reject nonfinite geometry and untrustworthy acquisition times."""
    bad_xyz = _pointcloud2(xyz=((float("nan"), 0.0, 0.0),) * 4)
    with pytest.raises(Mid360RosAdapterError, match="NaN or Inf"):
        extract_livox_pointcloud2(
            bad_xyz,
            expected_frame_id="livox_frame",
            use_livox_point_timestamps=True,
        )

    reordered = _pointcloud2(timestamps=(10.01, 10.03, 10.02, 10.04))
    with pytest.raises(Mid360RosAdapterError, match="nondecreasing"):
        extract_livox_pointcloud2(
            reordered,
            expected_frame_id="livox_frame",
            use_livox_point_timestamps=True,
        )

    constant = _pointcloud2(timestamps=(10.01,) * 4)
    with pytest.raises(Mid360RosAdapterError, match="constant"):
        extract_livox_pointcloud2(
            constant,
            expected_frame_id="livox_frame",
            use_livox_point_timestamps=True,
        )

    extraction = extract_livox_pointcloud2(
        constant,
        expected_frame_id="livox_frame",
        use_livox_point_timestamps=False,
    )
    assert extraction.point_timestamps_s is None


def test_pointcloud2_rejects_ambiguous_clock_or_capture_window() -> None:
    """Require explicit clock identity and in-window point times."""
    with pytest.raises(Mid360RosAdapterError, match="monotonic_clock_domain"):
        livox_pointcloud2_to_mid360_packet(
            _pointcloud2(),
            expected_frame_id="livox_frame",
            window_index=0,
            capture_start_s=10.0,
            capture_end_s=10.1,
            received_time_s=10.11,
            monotonic_clock_domain=None,
            use_livox_point_timestamps=True,
        )
    with pytest.raises(Mid360RosAdapterError, match="outside the declared"):
        livox_pointcloud2_to_mid360_packet(
            _pointcloud2(timestamps=(9.9, 10.02, 10.03, 10.04)),
            expected_frame_id="livox_frame",
            window_index=0,
            capture_start_s=10.0,
            capture_end_s=10.1,
            received_time_s=10.11,
            monotonic_clock_domain="PTP_TAI:mid360-g1",
            use_livox_point_timestamps=True,
        )


def test_pointcloud2_rejects_truthy_integer_mode_flags() -> None:
    """Do not reinterpret integer flags as explicit timing declarations."""
    with pytest.raises(Mid360RosAdapterError, match="must be bool"):
        extract_livox_pointcloud2(
            _pointcloud2(),
            expected_frame_id="livox_frame",
            use_livox_point_timestamps=1,  # type: ignore[arg-type]
        )
    with pytest.raises(Mid360RosAdapterError, match="must be bool"):
        extract_livox_pointcloud2(
            _pointcloud2(),
            expected_frame_id="livox_frame",
            use_livox_point_timestamps=True,
            reject_constant_point_timestamps=1,  # type: ignore[arg-type]
        )
