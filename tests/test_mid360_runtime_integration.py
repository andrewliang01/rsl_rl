from __future__ import annotations

import copy
import numpy as np
from pathlib import Path

import pytest

from rsl_rl.utils.mid360_clock_alignment import Mid360ClockAlignment
from rsl_rl.utils.mid360_ray_time_builder import (
    Mid360RayTimeBuilderError,
    Mid360RayTimeTensorBuilder,
)
from rsl_rl.utils.mid360_runtime_integration import (
    MID360_PACKET_CELL_WINNER,
    ingest_livox_custom_msg_runtime_step,
    validate_mid360_software_runtime_step,
)
from tests.test_mid360_clock_alignment import ACTION_CLOCK, _alignment
from tests.test_mid360_ros_adapter import _custom_msg
from tests.test_ray_time_deployment_manifest import _manifest


def _builder(
    tmp_path: Path,
    *,
    clock_domain: str = ACTION_CLOCK,
) -> Mid360RayTimeTensorBuilder:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"runtime-integration-checkpoint")
    return Mid360RayTimeTensorBuilder(
        _manifest(checkpoint, history_length=1, variant="Global"),
        max_packet_age_s=0.5,
        monotonic_clock_domain=clock_domain,
    )


def test_custom_msg_clock_builder_actor_views_close_one_software_path(
    tmp_path: Path,
) -> None:
    """Close the numeric adapter, clock, builder, and two actor views."""
    message = _custom_msg()
    message.timebase = 100_000_000_000
    alignment = _alignment()
    mapped_end = 100.1 * alignment.scale + alignment.offset_s
    result = ingest_livox_custom_msg_runtime_step(
        message,
        expected_frame_id="livox_frame",
        window_index=0,
        capture_end_sensor_s=100.1,
        received_action_time_s=mapped_end + 0.003,
        action_time_s=mapped_end + 0.004,
        alignment=alignment,
        builder=_builder(tmp_path),
    )

    assert result.ray_history.shape == (1, 1, 2, 16, 96)
    assert result.ray_history.dtype == np.float16
    assert result.ray_event_history.shape == (1, 1, 5, 16, 96)
    assert result.ray_event_history.dtype == np.float32
    np.testing.assert_array_equal(
        result.ray_event_history[:, :, 0],
        result.ray_history[:, :, 0].astype(np.float32),
    )
    np.testing.assert_array_equal(
        result.ray_event_history[:, :, 1],
        result.ray_history[:, :, 1].astype(np.float32),
    )
    assert result.packet_stats.transport_latency_s == pytest.approx(0.003)
    assert result.receipt["packet_cell_winner"] == MID360_PACKET_CELL_WINNER
    gates = result.receipt["software_gates"]
    assert gates["range_valid_cross_view_identity_checked"] is True
    assert gates["external_clock_evidence_verified_by_this_module"] is False
    assert gates["physical_sensor_recording_authenticated"] is False
    assert gates["raw_event_pies_reducer_connected"] is False
    assert gates["training_ready"] is False
    assert len(result.receipt_payload_sha256) == 64
    validate_mid360_software_runtime_step(result)


def test_runtime_receipt_does_not_promote_nearest_surface_to_pies(
    tmp_path: Path,
) -> None:
    """Keep the production nearest-surface rule distinct from PIES."""
    message = _custom_msg()
    message.timebase = 100_000_000_000
    # Both returns occupy one angular cell.  The later return is at 2 m, but
    # the production range builder deliberately keeps the nearer 1 m return.
    alignment = _alignment()
    mapped_end = 100.1 * alignment.scale + alignment.offset_s
    result = ingest_livox_custom_msg_runtime_step(
        message,
        expected_frame_id="livox_frame",
        window_index=0,
        capture_end_sensor_s=100.1,
        received_action_time_s=mapped_end + 0.003,
        action_time_s=mapped_end + 0.004,
        alignment=alignment,
        builder=_builder(tmp_path),
    )
    valid = result.ray_history[0, 0, 1] > 0.5
    assert valid.sum() == 1
    assert result.ray_history[0, 0, 0][valid].item() == 1.0
    assert result.receipt["software_gates"]["raw_event_stable_ids_present"] is False
    assert result.receipt["software_gates"]["pies_same_winner_control_ready"] is False


def test_runtime_rejects_clock_domain_and_action_before_receive(
    tmp_path: Path,
) -> None:
    """Reject a mixed clock or an action timestamp before callback receive."""
    alignment = _alignment()
    message = _custom_msg()
    message.timebase = 100_000_000_000
    mapped_end = 100.1 * alignment.scale + alignment.offset_s
    with pytest.raises(Mid360RayTimeBuilderError, match="Builder action clock"):
        ingest_livox_custom_msg_runtime_step(
            message,
            expected_frame_id="livox_frame",
            window_index=0,
            capture_end_sensor_s=100.1,
            received_action_time_s=mapped_end + 0.003,
            action_time_s=mapped_end + 0.004,
            alignment=alignment,
            builder=_builder(tmp_path, clock_domain="wrong-action-clock"),
        )
    with pytest.raises(Mid360RayTimeBuilderError, match="cannot precede"):
        ingest_livox_custom_msg_runtime_step(
            message,
            expected_frame_id="livox_frame",
            window_index=0,
            capture_end_sensor_s=100.1,
            received_action_time_s=mapped_end + 0.004,
            action_time_s=mapped_end + 0.003,
            alignment=alignment,
            builder=_builder(tmp_path / "second"),
        )


def test_runtime_requires_exact_alignment_type(tmp_path: Path) -> None:
    """Require a validated alignment object, not a duck-typed mapping."""
    assert isinstance(_alignment(), Mid360ClockAlignment)
    with pytest.raises(TypeError, match="alignment"):
        ingest_livox_custom_msg_runtime_step(
            _custom_msg(),
            expected_frame_id="livox_frame",
            window_index=0,
            capture_end_sensor_s=100.1,
            received_action_time_s=1.0,
            action_time_s=1.0,
            alignment=None,
            builder=_builder(tmp_path),
        )


def test_runtime_validator_rejects_tensor_and_gate_mutation(tmp_path: Path) -> None:
    """Bind actor tensor bytes and non-promotion gates to one receipt."""
    message = _custom_msg()
    message.timebase = 100_000_000_000
    alignment = _alignment()
    mapped_end = 100.1 * alignment.scale + alignment.offset_s
    result = ingest_livox_custom_msg_runtime_step(
        message,
        expected_frame_id="livox_frame",
        window_index=0,
        capture_end_sensor_s=100.1,
        received_action_time_s=mapped_end + 0.003,
        action_time_s=mapped_end + 0.004,
        alignment=alignment,
        builder=_builder(tmp_path),
    )

    changed_tensor = copy.deepcopy(result)
    changed_tensor.ray_history[0, 0, 0, 0, 0] += np.float16(0.5)
    with pytest.raises(Mid360RayTimeBuilderError, match="SHA-256"):
        validate_mid360_software_runtime_step(changed_tensor)

    promoted = copy.deepcopy(result)
    promoted.receipt["software_gates"]["training_ready"] = True
    promoted = type(promoted)(
        ray_history=promoted.ray_history,
        ray_event_history=promoted.ray_event_history,
        packet_stats=promoted.packet_stats,
        receipt=promoted.receipt,
        receipt_payload_sha256=result.receipt_payload_sha256,
    )
    with pytest.raises(Mid360RayTimeBuilderError, match="payload SHA-256"):
        validate_mid360_software_runtime_step(promoted)

    extra_key = copy.deepcopy(result)
    extra_key.receipt["unregistered_claim"] = False
    extra_key = type(extra_key)(
        ray_history=extra_key.ray_history,
        ray_event_history=extra_key.ray_event_history,
        packet_stats=extra_key.packet_stats,
        receipt=extra_key.receipt,
        receipt_payload_sha256=result.receipt_payload_sha256,
    )
    with pytest.raises(Mid360RayTimeBuilderError, match="keys differ"):
        validate_mid360_software_runtime_step(extra_key)
