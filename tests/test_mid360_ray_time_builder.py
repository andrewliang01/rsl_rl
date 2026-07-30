from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from rsl_rl.utils.mid360_ray_time_builder import (
    MID360_NORMALIZED_SENSOR_FRAME,
    MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
    Mid360PointPacket,
    Mid360RayTimeBuilderError,
    Mid360RayTimeTensorBuilder,
    StaleMid360PacketError,
    point_packet_from_livox_custom_msg_arrays,
)
from tests.test_ray_time_deployment_manifest import _manifest


def _builder(
    tmp_path: Path,
    *,
    history_length: int = 5,
    max_packet_age_s: float = 0.15,
) -> Mid360RayTimeTensorBuilder:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / f"model_k{history_length}.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = _manifest(
        checkpoint,
        history_length=history_length,
        variant="Global",
    )
    return Mid360RayTimeTensorBuilder(
        manifest,
        max_packet_age_s=max_packet_age_s,
    )


def _empty_packet(
    *,
    window_index: int,
    capture_start_s: float,
    capture_end_s: float,
) -> Mid360PointPacket:
    return Mid360PointPacket(
        xyz_m=np.empty((0, 3), dtype=np.float64),
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        timestamp_semantics=MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
        window_index=window_index,
        capture_start_s=capture_start_s,
        capture_end_s=capture_end_s,
    )


def _xyz_from_spherical(
    range_m: float,
    elevation_deg: float,
    azimuth_deg: float,
) -> np.ndarray:
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    return np.asarray(
        [
            range_m * np.cos(elevation) * np.cos(azimuth),
            range_m * np.cos(elevation) * np.sin(azimuth),
            range_m * np.sin(elevation),
        ],
        dtype=np.float64,
    )


def test_sensor_grid_matches_policy_flip_roll_and_never_reapplies_sparse_mask(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path)
    ranges = np.full((16, 96), 2.0, dtype=np.float32)
    hits = np.ones((16, 96), dtype=np.bool_)

    stats = builder.ingest_sensor_grid(
        ranges,
        hits,
        window_index=0,
        capture_start_s=1.0,
        capture_end_s=1.1,
    )
    policy = builder.policy_tensor(now_s=1.1)

    assert policy.shape == (1, 5, 2, 16, 96)
    assert policy.dtype == np.float16
    assert np.count_nonzero(policy[0, :-1]) == 0
    assert np.all(policy[0, -1, 0] == np.float16(2.0))
    assert np.all(policy[0, -1, 1] == np.float16(1.0))
    assert stats.observed_hit_fraction == 1.0
    assert stats.emitted_coverage_known is False
    assert stats.max_packet_age_s == 0.15
    assert stats.packet_cadence_tolerance_s == 0.02
    assert stats.timestamp_tolerance_s == 1.0e-6

    ranges = np.zeros((16, 96), dtype=np.float32)
    hits = np.zeros((16, 96), dtype=np.bool_)
    ranges[0, 1] = 3.0
    hits[0, 1] = True
    packet = builder.sensor_grid_to_policy_packet(ranges, hits)
    # flip_width then roll(+1): physical-sensor column 1 maps to policy 95.
    assert packet[0, 0, 95] == np.float16(3.0)
    assert packet[1, 0, 95] == np.float16(1.0)
    assert np.count_nonzero(packet) == 2


def test_point_binning_uses_nearest_return_and_expected_spherical_pixel(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path)
    # Sensor row zero is +52 degrees. Sensor column one is -176.25 degrees,
    # which the upside-down flip/roll maps to policy row zero, column 95.
    near = _xyz_from_spherical(2.5, 52.0, -176.25)
    far_same_bin = _xyz_from_spherical(4.0, 52.0, -176.25)
    outside_fov = _xyz_from_spherical(2.0, 60.0, 0.0)
    below_min = _xyz_from_spherical(0.05, 0.0, 0.0)
    clipped = _xyz_from_spherical(8.0, -7.0, -180.0)
    points = np.stack(
        (near, far_same_bin, outside_fov, below_min, clipped),
        axis=0,
    )
    packet = Mid360PointPacket(
        xyz_m=points,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        timestamp_semantics=MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
        window_index=3,
        capture_start_s=10.0,
        capture_end_s=10.1,
    )

    stats = builder.ingest_point_packet(packet)
    policy = builder.history_tensor(now_s=10.1)[-1]

    assert policy[0, 0, 95] == np.float16(2.5)
    assert policy[1, 0, 95] == np.float16(1.0)
    assert policy[0, 15, 0] == np.float16(6.0)
    assert policy[1, 15, 0] == np.float16(1.0)
    assert stats.input_return_points == 5
    assert stats.accepted_return_points == 3
    assert stats.below_min_range_points == 1
    assert stats.outside_vertical_fov_points == 1
    assert stats.collided_return_points == 1
    assert stats.policy_clipped_points == 1
    assert stats.observed_hit_bins == 2


def test_history_inserts_unknown_for_implicit_and_explicit_packet_loss(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path)
    ranges = np.zeros((16, 96), dtype=np.float32)
    hits = np.zeros((16, 96), dtype=np.bool_)
    ranges[4, 7] = 1.5
    hits[4, 7] = True

    builder.ingest_sensor_grid(
        ranges,
        hits,
        window_index=10,
        capture_start_s=1.0,
        capture_end_s=1.1,
    )
    stats = builder.ingest_sensor_grid(
        ranges,
        hits,
        window_index=12,
        capture_start_s=1.2,
        capture_end_s=1.3,
    )
    dropped = builder.record_dropped_packet(
        window_index=13,
        capture_start_s=1.3,
        capture_end_s=1.4,
    )

    assert stats.implicit_missing_packets_inserted == 1
    assert stats.capture_interval_error_s == pytest.approx(0.0)
    assert dropped.explicit_drop is True
    assert builder.window_indices.tolist() == [-1, 10, 11, 12, 13]
    history = builder.history_tensor(now_s=1.4)
    assert np.count_nonzero(history[2]) == 0
    assert np.count_nonzero(history[4]) == 0
    assert np.count_nonzero(history[1]) == 2
    assert np.count_nonzero(history[3]) == 2
    assert builder.implicit_missing_packets_total == 1
    assert builder.explicit_dropped_packets_total == 1


def test_dropped_windows_do_not_refresh_last_real_acquisition(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path, max_packet_age_s=0.15)
    builder.ingest_point_packet(
        _empty_packet(
            window_index=0,
            capture_start_s=0.0,
            capture_end_s=0.1,
        )
    )
    builder.record_dropped_packet(
        window_index=1,
        capture_start_s=0.1,
        capture_end_s=0.2,
    )
    assert builder.policy_tensor(now_s=0.24).shape == (1, 5, 2, 16, 96)

    builder.record_dropped_packet(
        window_index=2,
        capture_start_s=0.2,
        capture_end_s=0.3,
    )
    with pytest.raises(StaleMid360PacketError, match="stale"):
        builder.policy_tensor(now_s=0.3)

    only_drops = _builder(tmp_path / "only_drops")
    only_drops.record_dropped_packet(
        window_index=0,
        capture_start_s=0.0,
        capture_end_s=0.1,
    )
    with pytest.raises(
        StaleMid360PacketError,
        match="No acquired MID-360 packet",
    ):
        only_drops.policy_tensor(now_s=0.1)


def test_capture_duration_and_cadence_are_manifest_bound(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path)
    builder.ingest_point_packet(
        _empty_packet(
            window_index=0,
            capture_start_s=0.0,
            capture_end_s=0.1,
        )
    )
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="capture cadence",
    ):
        builder.ingest_point_packet(
            _empty_packet(
                window_index=1,
                capture_start_s=9.9,
                capture_end_s=10.0,
            )
        )
    assert builder.window_indices.tolist() == [-1, -1, -1, -1, 0]

    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="Capture window exceeds",
    ):
        builder.ingest_sensor_grid(
            np.zeros((16, 96), dtype=np.float32),
            np.zeros((16, 96), dtype=np.bool_),
            window_index=1,
            capture_start_s=0.1,
            capture_end_s=0.3,
        )
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="Capture window exceeds",
    ):
        builder.record_dropped_packet(
            window_index=1,
            capture_start_s=0.1,
            capture_end_s=0.3,
        )

    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="packet_cadence_tolerance_s must be smaller",
    ):
        _builder_with_cadence_tolerance(tmp_path / "bad_tolerance", 0.1)


def _builder_with_cadence_tolerance(
    tmp_path: Path,
    tolerance_s: float,
) -> Mid360RayTimeTensorBuilder:
    checkpoint = tmp_path / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    manifest = _manifest(
        checkpoint,
        history_length=5,
        variant="Global",
    )
    return Mid360RayTimeTensorBuilder(
        manifest,
        max_packet_age_s=0.15,
        packet_cadence_tolerance_s=tolerance_s,
    )


def test_out_of_order_and_stale_data_fail_closed(tmp_path: Path) -> None:
    builder = _builder(tmp_path, history_length=1)
    builder.ingest_point_packet(
        _empty_packet(
            window_index=0,
            capture_start_s=0.9,
            capture_end_s=1.0,
        )
    )
    assert builder.policy_tensor(now_s=1.15).shape == (1, 1, 2, 16, 96)

    with pytest.raises(StaleMid360PacketError, match="stale"):
        builder.policy_tensor(now_s=1.16)
    with pytest.raises(StaleMid360PacketError, match="future"):
        builder.policy_tensor(now_s=0.9)
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="strictly increasing",
    ):
        builder.ingest_point_packet(
            _empty_packet(
                window_index=0,
                capture_start_s=1.0,
                capture_end_s=1.1,
            )
        )
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="precedes capture_end_s",
    ):
        builder.ingest_sensor_grid(
            np.zeros((16, 96), dtype=np.float32),
            np.zeros((16, 96), dtype=np.bool_),
            window_index=1,
            capture_start_s=1.0,
            capture_end_s=1.1,
            received_time_s=1.05,
        )


def test_livox_custom_msg_timebase_and_offsets_are_explicit() -> None:
    xyz = np.asarray(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=np.float32)
    packet = point_packet_from_livox_custom_msg_arrays(
        xyz_m=xyz,
        timebase_ns=2_000_000_000,
        offset_time_ns=np.asarray((0, 25_000_000), dtype=np.uint32),
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        window_index=7,
        received_time_s=2.03,
    )

    assert packet.capture_start_s == pytest.approx(2.0)
    assert packet.capture_end_s == pytest.approx(2.025)
    np.testing.assert_allclose(packet.point_timestamps_s, (2.0, 2.025))

    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="one value per point",
    ):
        point_packet_from_livox_custom_msg_arrays(
            xyz_m=xyz,
            timebase_ns=0,
            offset_time_ns=np.asarray((0,), dtype=np.uint32),
            coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
            window_index=0,
        )


@dataclass
class _Adapter:
    packet: Mid360PointPacket

    def to_mid360_point_packet(self, raw_packet: object) -> Mid360PointPacket:
        assert raw_packet == "native"
        return self.packet


def test_driver_adapter_boundary_is_explicit(tmp_path: Path) -> None:
    builder = _builder(tmp_path, history_length=1)
    stats = builder.ingest_driver_packet(
        "native",
        _Adapter(
            _empty_packet(
                window_index=4,
                capture_start_s=5.0,
                capture_end_s=5.1,
            )
        ),
    )
    assert stats.window_index == 4

    wrong_frame = _empty_packet(
        window_index=5,
        capture_start_s=5.1,
        capture_end_s=5.2,
    )
    wrong_frame = Mid360PointPacket(
        **{**wrong_frame.__dict__, "coordinate_frame": "guessed_frame"}
    )
    with pytest.raises(Mid360RayTimeBuilderError, match="normalize points"):
        builder.ingest_point_packet(wrong_frame)
