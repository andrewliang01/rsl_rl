from __future__ import annotations

import numpy as np
from dataclasses import replace
from pathlib import Path

import pytest

from rsl_rl.utils.mid360_clock_alignment import (
    MID360_CLOCK_ALIGNMENT_SCHEMA,
    Mid360ClockAlignment,
    Mid360ClockAlignmentError,
    map_livox_packet_to_action_clock,
)
from rsl_rl.utils.mid360_ray_time_builder import (
    MID360_TIMESTAMP_LIVOX_CUSTOM_MSG_ACTION_CLOCK,
    Mid360PointPacket,
    Mid360RayTimeTensorBuilder,
)
from rsl_rl.utils.mid360_ros_adapter import (
    livox_custom_msg_to_sensor_clock_packet,
)
from tests.test_mid360_ros_adapter import _custom_msg
from tests.test_ray_time_deployment_manifest import _manifest

SENSOR_CLOCK = "livox_device_time:mid360-sn-001"
ACTION_CLOCK = "CLOCK_MONOTONIC_RAW:boot-test-001"


def _alignment(**overrides: object) -> Mid360ClockAlignment:
    values = {
        "sensor_clock_domain": SENSOR_CLOCK,
        "action_clock_domain": ACTION_CLOCK,
        "sensor_serial": "mid360-sn-001",
        "host_boot_id": "boot-test-001",
        "calibration_method": "external_hardware_cross_timestamp",
        "calibration_evidence_sha256": "a" * 64,
        "scale": 1.00001,
        "offset_s": 1_000.0,
        "calibrated_sensor_start_s": 100.0,
        "calibrated_sensor_end_s": 140.0,
        "sample_count": 64,
        "residual_p99_s": 0.0008,
        "residual_max_s": 0.0015,
        "uncertainty_s": 0.002,
    }
    values.update(overrides)
    return Mid360ClockAlignment(**values)


def _sensor_packet() -> Mid360PointPacket:
    message = _custom_msg()
    message.timebase = 100_000_000_000
    return livox_custom_msg_to_sensor_clock_packet(
        message,
        expected_frame_id="livox_frame",
        window_index=0,
        capture_end_sensor_s=100.1,
        sensor_clock_domain=SENSOR_CLOCK,
    )


def _builder(tmp_path: Path) -> Mid360RayTimeTensorBuilder:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"clock-contract-checkpoint")
    manifest = _manifest(checkpoint, history_length=1, variant="Global")
    return Mid360RayTimeTensorBuilder(
        manifest,
        max_packet_age_s=0.5,
        timestamp_tolerance_s=1.0e-6,
        monotonic_clock_domain=ACTION_CLOCK,
    )


def test_mapping_is_deterministic_and_builder_consumes_action_clock_packet(
    tmp_path: Path,
) -> None:
    """Mapped returns keep float64 timing and pass the strict builder."""
    alignment = _alignment()
    source = _sensor_packet()
    mapped_end = 100.1 * alignment.scale + alignment.offset_s
    result = map_livox_packet_to_action_clock(
        source,
        alignment,
        received_action_time_s=mapped_end + 0.003,
    )
    packet = result.packet

    assert packet.timestamp_semantics == (MID360_TIMESTAMP_LIVOX_CUSTOM_MSG_ACTION_CLOCK)
    assert packet.monotonic_clock_domain == ACTION_CLOCK
    assert packet.point_timestamps_s is not None
    expected = source.point_timestamps_s * alignment.scale + alignment.offset_s
    np.testing.assert_array_equal(packet.point_timestamps_s, expected)
    assert packet.xyz_m is not source.xyz_m
    assert result.clock_alignment_receipt_sha256 == (alignment.receipt_payload_sha256())
    assert alignment.receipt_payload()["schema"] == MID360_CLOCK_ALIGNMENT_SCHEMA
    assert alignment.receipt_payload()["calibration"]["external_evidence_verified_by_this_module"] is False

    builder = _builder(tmp_path)
    stats = builder.ingest_point_packet(packet)
    assert stats.monotonic_clock_domain == ACTION_CLOCK
    assert stats.transport_latency_s == pytest.approx(0.003)
    aligned = builder.aligned_event_time_history(
        now_s=mapped_end + 0.004,
        monotonic_clock_domain=ACTION_CLOCK,
    )
    assert aligned.frame_valid.tolist() == [True]
    assert aligned.return_valid.any()
    assert aligned.return_age_s[aligned.return_valid].min() > 0.0


def test_mapping_rejects_extrapolation_cross_domain_and_receive_time_mix() -> None:
    """No packet may silently extrapolate or mix sensor/action receive time."""
    alignment = _alignment()
    packet = _sensor_packet()
    too_late = replace(
        packet,
        capture_start_s=139.95,
        capture_end_s=140.05,
        point_timestamps_s=np.asarray((139.96, 140.0), dtype=np.float64),
    )
    with pytest.raises(Mid360ClockAlignmentError, match="extrapolation"):
        map_livox_packet_to_action_clock(
            too_late,
            alignment,
            received_action_time_s=2_000.0,
        )

    wrong_domain = replace(packet, monotonic_clock_domain="another-clock")
    with pytest.raises(Mid360ClockAlignmentError, match="clock domain"):
        map_livox_packet_to_action_clock(
            wrong_domain,
            alignment,
            received_action_time_s=2_000.0,
        )

    mixed_receive = replace(packet, received_time_s=100.11)
    with pytest.raises(Mid360ClockAlignmentError, match="must be None"):
        map_livox_packet_to_action_clock(
            mixed_receive,
            alignment,
            received_action_time_s=2_000.0,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action_clock_domain": SENSOR_CLOCK}, "share an identity"),
        ({"scale": 1.000101}, "drift"),
        ({"sample_count": 31}, "sample_count"),
        ({"calibrated_sensor_end_s": 129.9}, "span"),
        (
            {
                "residual_p99_s": 0.0021,
                "residual_max_s": 0.003,
                "uncertainty_s": 0.003,
            },
            "P99",
        ),
        ({"residual_max_s": 0.0051, "uncertainty_s": 0.0051}, "maximum"),
        ({"uncertainty_s": 0.001}, "uncertainty"),
        ({"calibration_evidence_sha256": "A" * 64}, "SHA-256"),
    ],
)
def test_alignment_quality_gates_fail_closed(
    overrides: dict[str, object],
    match: str,
) -> None:
    """Weak or ambiguous calibration receipts cannot construct a mapper."""
    with pytest.raises(Mid360ClockAlignmentError, match=match):
        _alignment(**overrides)


def test_clock_payload_hash_changes_with_any_mapping_semantic() -> None:
    """The deployment binding hash covers affine and external evidence data."""
    baseline = _alignment()
    assert baseline.receipt_payload_sha256() == baseline.receipt_payload_sha256()
    assert baseline.receipt_payload_sha256() != _alignment(offset_s=1_000.0001).receipt_payload_sha256()
    assert (
        baseline.receipt_payload_sha256() != _alignment(calibration_evidence_sha256="b" * 64).receipt_payload_sha256()
    )
