from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from rsl_rl.utils.mid360_ray_time_builder import (
    MID360_CAPTURE_END_ACQUISITION_WINDOW,
    MID360_CAPTURE_END_LATEST_RETURN_LEGACY,
    MID360_NORMALIZED_SENSOR_FRAME,
    MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
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


def _event_time_builder(
    tmp_path: Path,
    *,
    history_length: int = 5,
    max_packet_age_s: float = 0.5,
    timestamp_tolerance_s: float = 1.0e-6,
    clock_domain: str = "CLOCK_MONOTONIC_RAW:boot-7",
) -> Mid360RayTimeTensorBuilder:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / f"event_model_k{history_length}.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = _manifest(
        checkpoint,
        history_length=history_length,
        variant="Global",
    )
    return Mid360RayTimeTensorBuilder(
        manifest,
        max_packet_age_s=max_packet_age_s,
        timestamp_tolerance_s=timestamp_tolerance_s,
        monotonic_clock_domain=clock_domain,
    )


def _timed_packet(
    *,
    points: np.ndarray,
    point_timestamps_s: np.ndarray,
    window_index: int,
    capture_start_s: float,
    capture_end_s: float,
    clock_domain: str = "CLOCK_MONOTONIC_RAW:boot-7",
) -> Mid360PointPacket:
    return Mid360PointPacket(
        xyz_m=points,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        timestamp_semantics=MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
        window_index=window_index,
        capture_start_s=capture_start_s,
        capture_end_s=capture_end_s,
        point_timestamps_s=point_timestamps_s,
        monotonic_clock_domain=clock_domain,
        capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
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
    xyz = np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float32)
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
    assert (
        packet.capture_end_semantics
        == MID360_CAPTURE_END_LATEST_RETURN_LEGACY
    )
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


def test_strict_event_time_requires_declared_capture_window_end(
    tmp_path: Path,
) -> None:
    xyz = np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float32)
    offsets = np.asarray((0, 25_000_000), dtype=np.uint32)
    builder = _event_time_builder(tmp_path / "legacy_end", history_length=1)
    inferred_end = point_packet_from_livox_custom_msg_arrays(
        xyz_m=xyz,
        timebase_ns=2_000_000_000,
        offset_time_ns=offsets,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        window_index=0,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="acquisition-window boundary",
    ):
        builder.ingest_point_packet(inferred_end)

    declared_builder = _event_time_builder(
        tmp_path / "declared_end",
        history_length=1,
    )
    declared_end = point_packet_from_livox_custom_msg_arrays(
        xyz_m=xyz,
        timebase_ns=2_000_000_000,
        offset_time_ns=offsets,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        window_index=0,
        capture_end_s=2.1,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )
    stats = declared_builder.ingest_point_packet(declared_end)
    aligned = declared_builder.aligned_event_time_history(
        now_s=2.1,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )
    assert (
        stats.capture_end_semantics
        == MID360_CAPTURE_END_ACQUISITION_WINDOW
    )
    assert aligned.packet_age_s.tolist() == pytest.approx([0.0])
    assert sorted(aligned.return_age_s[aligned.return_valid].tolist()) == (
        pytest.approx([0.075, 0.1])
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


def test_collision_range_and_age_come_from_same_nearest_return(
    tmp_path: Path,
) -> None:
    builder = _event_time_builder(tmp_path, history_length=1)
    near = _xyz_from_spherical(2.0, 52.0, -176.25)
    far = _xyz_from_spherical(4.0, 52.0, -176.25)
    stats = builder.ingest_point_packet(
        _timed_packet(
            points=np.stack((far, near)),
            point_timestamps_s=np.asarray((10.09, 10.02)),
            window_index=0,
            capture_start_s=10.0,
            capture_end_s=10.1,
        )
    )

    aligned = builder.aligned_event_time_history(
        now_s=10.1,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )

    assert aligned.ray_history.shape == (1, 2, 16, 96)
    assert aligned.ray_history.dtype == np.float16
    assert aligned.return_valid.dtype == np.bool_
    assert aligned.return_age_s.dtype == np.float32
    assert aligned.packet_age_s.dtype == np.float32
    assert aligned.frame_valid.dtype == np.bool_
    assert aligned.ray_history[0, 0, 0, 95] == np.float16(2.0)
    assert aligned.return_valid[0, 0, 95]
    # The far return is newer (age 0.01), so independently reducing age would
    # expose the bug.  The winning near return must retain its own age 0.08.
    assert aligned.return_age_s[0, 0, 95] == pytest.approx(0.08)
    assert aligned.packet_age_s.tolist() == pytest.approx([0.0])
    assert aligned.frame_valid.tolist() == [True]
    assert stats.collided_return_points == 1
    assert stats.timed_return_bins == 1
    assert stats.event_time_frame_valid is True


def test_exact_range_tie_uses_earliest_timestamp_independent_of_input_order(
    tmp_path: Path,
) -> None:
    point = _xyz_from_spherical(3.0, 52.0, -176.25)
    outputs = []
    for suffix, timestamps in (
        ("forward", (20.08, 20.02)),
        ("reverse", (20.02, 20.08)),
    ):
        builder = _event_time_builder(tmp_path / suffix, history_length=1)
        builder.ingest_point_packet(
            _timed_packet(
                points=np.stack((point, point)),
                point_timestamps_s=np.asarray(timestamps),
                window_index=0,
                capture_start_s=20.0,
                capture_end_s=20.1,
            )
        )
        outputs.append(
            builder.aligned_event_time_history(
                now_s=20.1,
                monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
            )
        )

    np.testing.assert_array_equal(outputs[0].ray_history, outputs[1].ray_history)
    np.testing.assert_array_equal(outputs[0].return_valid, outputs[1].return_valid)
    np.testing.assert_allclose(outputs[0].return_age_s, outputs[1].return_age_s)
    assert outputs[0].return_age_s[0, 0, 95] == pytest.approx(0.08)


def test_randomized_collision_oracle_is_order_invariant(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260731)
    points: list[np.ndarray] = []
    timestamps: list[float] = []
    oracle: dict[tuple[int, int], tuple[float, float]] = {}
    sensor_bins = ((0, 0), (3, 1), (8, 20), (15, 95))
    elevation_step = 59.0 / 15.0
    for row, column in sensor_bins:
        elevation_deg = 52.0 - row * elevation_step
        azimuth_deg = -180.0 + column * 3.75
        candidate_ranges = rng.uniform(1.5, 5.5, size=6)
        candidate_ranges[:2] = 1.25  # Exact nearest-range tie.
        candidate_times = rng.uniform(30.0, 30.1, size=6)
        for range_m, timestamp_s in zip(
            candidate_ranges,
            candidate_times,
            strict=True,
        ):
            points.append(
                _xyz_from_spherical(range_m, elevation_deg, azimuth_deg)
            )
            timestamps.append(float(timestamp_s))
        winner = min(
            zip(candidate_ranges, candidate_times, strict=True),
            key=lambda item: (item[0], item[1]),
        )
        policy_column = (-column) % 96
        oracle[(row, policy_column)] = (float(winner[0]), float(winner[1]))

    point_array = np.stack(points)
    timestamp_array = np.asarray(timestamps)
    outputs = []
    for suffix, permutation in (
        ("native", np.arange(point_array.shape[0])),
        ("permuted", rng.permutation(point_array.shape[0])),
    ):
        builder = _event_time_builder(tmp_path / suffix, history_length=1)
        builder.ingest_point_packet(
            _timed_packet(
                points=point_array[permutation],
                point_timestamps_s=timestamp_array[permutation],
                window_index=0,
                capture_start_s=30.0,
                capture_end_s=30.1,
            )
        )
        outputs.append(
            builder.aligned_event_time_history(
                now_s=30.1,
                monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
            )
        )

    np.testing.assert_array_equal(outputs[0].ray_history, outputs[1].ray_history)
    np.testing.assert_array_equal(outputs[0].return_valid, outputs[1].return_valid)
    np.testing.assert_allclose(outputs[0].return_age_s, outputs[1].return_age_s)
    for (row, column), (range_m, timestamp_s) in oracle.items():
        assert outputs[0].ray_history[0, 0, row, column] == np.float16(range_m)
        assert outputs[0].return_valid[0, row, column]
        assert outputs[0].return_age_s[0, row, column] == pytest.approx(
            30.1 - timestamp_s,
        )


def test_range_time_flip_roll_and_history_shifts_are_identical(
    tmp_path: Path,
) -> None:
    builder = _event_time_builder(tmp_path, history_length=3)
    first = _xyz_from_spherical(2.0, 52.0, -176.25)
    latest = _xyz_from_spherical(3.0, 52.0, -172.5)
    builder.ingest_point_packet(
        _timed_packet(
            points=first[None, :],
            point_timestamps_s=np.asarray((1.05,)),
            window_index=0,
            capture_start_s=1.0,
            capture_end_s=1.1,
        )
    )
    builder.ingest_point_packet(
        _timed_packet(
            points=latest[None, :],
            point_timestamps_s=np.asarray((1.28,)),
            window_index=2,
            capture_start_s=1.2,
            capture_end_s=1.3,
        )
    )

    aligned = builder.aligned_event_time_history(
        now_s=1.3,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )

    assert aligned.window_indices.tolist() == [0, 1, 2]
    assert aligned.frame_valid.tolist() == [True, False, True]
    np.testing.assert_array_equal(
        aligned.return_valid,
        aligned.ray_history[:, 1].astype(np.bool_),
    )
    assert aligned.packet_age_s.tolist() == pytest.approx([0.2, 0.0, 0.0])
    # Sensor columns 1 and 2 become policy columns 95 and 94 under the same
    # flip(+roll) used by range, valid, and return age.
    assert aligned.ray_history[0, 0, 0, 95] == np.float16(2.0)
    assert aligned.return_valid[0, 0, 95]
    assert aligned.return_age_s[0, 0, 95] == pytest.approx(0.25)
    assert aligned.ray_history[2, 0, 0, 94] == np.float16(3.0)
    assert aligned.return_valid[2, 0, 94]
    assert aligned.return_age_s[2, 0, 94] == pytest.approx(0.02)
    assert not aligned.return_valid[1].any()
    assert np.count_nonzero(aligned.return_age_s[1]) == 0
    for frame_index in (0, 2):
        assert np.all(
            aligned.return_age_s[frame_index][
                aligned.return_valid[frame_index]
            ]
            >= aligned.packet_age_s[frame_index]
        )


def test_empty_unfilled_and_dropped_frames_have_zero_age_and_false_validity(
    tmp_path: Path,
) -> None:
    builder = _event_time_builder(tmp_path, history_length=3)
    empty = Mid360PointPacket(
        xyz_m=np.empty((0, 3), dtype=np.float64),
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        timestamp_semantics=MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
        window_index=0,
        capture_start_s=0.0,
        capture_end_s=0.1,
        point_timestamps_s=np.empty((0,), dtype=np.float64),
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
        capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
    )
    builder.ingest_point_packet(empty)
    builder.record_dropped_packet(
        window_index=1,
        capture_start_s=0.1,
        capture_end_s=0.2,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )

    aligned = builder.aligned_event_time_history(
        now_s=0.2,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )

    assert aligned.window_indices.tolist() == [-1, 0, 1]
    assert aligned.frame_valid.tolist() == [False, False, False]
    assert np.count_nonzero(aligned.packet_age_s) == 0
    assert not aligned.return_valid.any()
    assert np.count_nonzero(aligned.return_age_s) == 0
    assert np.count_nonzero(aligned.ray_history) == 0


def test_emitted_no_return_is_not_converted_to_observed_free_space(
    tmp_path: Path,
) -> None:
    builder = _event_time_builder(tmp_path, history_length=1)
    ranges = np.zeros((16, 96), dtype=np.float32)
    hits = np.zeros((16, 96), dtype=np.bool_)
    emitted = np.ones((16, 96), dtype=np.bool_)
    stats = builder.ingest_sensor_grid(
        ranges,
        hits,
        window_index=0,
        capture_start_s=3.0,
        capture_end_s=3.1,
        emitted_sensor_mask=emitted,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )
    aligned = builder.aligned_event_time_history(
        now_s=3.1,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )

    assert stats.emitted_bin_fraction == 1.0
    assert stats.observed_hit_bins == 0
    assert not aligned.frame_valid[0]
    assert np.count_nonzero(aligned.ray_history) == 0
    assert not aligned.return_valid.any()

    bad_timestamps = np.zeros((16, 96), dtype=np.float64)
    bad_timestamps[0, 0] = 3.05
    bad_builder = _event_time_builder(tmp_path / "bad", history_length=1)
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="Bins without a successful return",
    ):
        bad_builder.ingest_sensor_grid(
            ranges,
            hits,
            window_index=0,
            capture_start_s=3.0,
            capture_end_s=3.1,
            sensor_return_timestamps_s=bad_timestamps,
            monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
        )


def test_strict_event_time_requires_verified_returns_and_common_clock(
    tmp_path: Path,
) -> None:
    point = _xyz_from_spherical(2.0, 0.0, 0.0)[None, :]
    builder = _event_time_builder(tmp_path, history_length=1)
    untimed = Mid360PointPacket(
        xyz_m=point,
        coordinate_frame=MID360_NORMALIZED_SENSOR_FRAME,
        timestamp_semantics=MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
        window_index=0,
        capture_start_s=4.0,
        capture_end_s=4.1,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
        capture_end_semantics=MID360_CAPTURE_END_ACQUISITION_WINDOW,
    )
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="requires one verified timestamp",
    ):
        builder.ingest_point_packet(untimed)

    wrong_clock = _timed_packet(
        points=point,
        point_timestamps_s=np.asarray((4.05,)),
        window_index=0,
        capture_start_s=4.0,
        capture_end_s=4.1,
        clock_domain="CLOCK_MONOTONIC_RAW:other-boot",
    )
    with pytest.raises(Mid360RayTimeBuilderError, match="clock domain"):
        builder.ingest_point_packet(wrong_clock)

    builder.ingest_point_packet(
        _timed_packet(
            points=point,
            point_timestamps_s=np.asarray((4.05,)),
            window_index=0,
            capture_start_s=4.0,
            capture_end_s=4.1,
        )
    )
    with pytest.raises(Mid360RayTimeBuilderError, match="clock domain"):
        builder.aligned_event_time_history(
            now_s=4.1,
            monotonic_clock_domain="CLOCK_MONOTONIC_RAW:other-boot",
        )

    legacy = _builder(tmp_path / "legacy", history_length=1)
    legacy.ingest_point_packet(untimed)
    assert legacy.history_tensor(now_s=4.1).shape == (1, 2, 16, 96)
    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="explicit monotonic_clock_domain",
    ):
        legacy.aligned_event_time_history(
            now_s=4.1,
            monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
        )


def test_timestamp_tolerance_clamps_boundary_but_rejects_negative_age(
    tmp_path: Path,
) -> None:
    point = _xyz_from_spherical(2.0, 0.0, 0.0)[None, :]
    builder = _event_time_builder(
        tmp_path / "within",
        history_length=1,
        timestamp_tolerance_s=1.0e-3,
    )
    builder.ingest_point_packet(
        _timed_packet(
            points=point,
            point_timestamps_s=np.asarray((5.1005,)),
            window_index=0,
            capture_start_s=5.0,
            capture_end_s=5.1,
        )
    )
    aligned = builder.aligned_event_time_history(
        now_s=5.0995,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )
    assert aligned.packet_age_s.tolist() == pytest.approx([0.0])
    assert aligned.return_age_s[aligned.return_valid].tolist() == pytest.approx(
        [0.0]
    )

    with pytest.raises(StaleMid360PacketError, match="future"):
        builder.aligned_event_time_history(
            now_s=5.0989,
            monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
        )

    for suffix, bad_timestamp in (
        ("outside", 5.1011),
        ("nan", float("nan")),
    ):
        bad_builder = _event_time_builder(
            tmp_path / suffix,
            history_length=1,
            timestamp_tolerance_s=1.0e-3,
        )
        with pytest.raises(
            Mid360RayTimeBuilderError,
            match="timestamp|finite",
        ):
            bad_builder.ingest_point_packet(
                _timed_packet(
                    points=point,
                    point_timestamps_s=np.asarray((bad_timestamp,)),
                    window_index=0,
                    capture_start_s=5.0,
                    capture_end_s=5.1,
                )
            )

    with pytest.raises(
        Mid360RayTimeBuilderError,
        match="timestamp_tolerance_s must be smaller",
    ):
        _event_time_builder(
            tmp_path / "oversized_tolerance",
            history_length=1,
            timestamp_tolerance_s=0.1,
        )
