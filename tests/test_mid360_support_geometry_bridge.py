from __future__ import annotations

import torch
from pathlib import Path

import pytest

from rsl_rl.modules import (
    CalibratedSphericalSupportRoleGeometry,
    SharedUniqueSupportActorAdapter,
)
from rsl_rl.utils.mid360_runtime_integration import (
    Mid360SoftwareRuntimeStep,
    ingest_livox_custom_msg_runtime_step,
)
from rsl_rl.utils.mid360_support_geometry_bridge import (
    MID360_SUPPORT_GEOMETRY_BRIDGE_SCHEMA,
    mid360_runtime_step_to_support_geometry,
    ray_event_history_to_support_geometry,
)
from tests.test_mid360_clock_alignment import _alignment
from tests.test_mid360_ros_adapter import _custom_msg
from tests.test_mid360_runtime_integration import _builder


def _geometry() -> CalibratedSphericalSupportRoleGeometry:
    rays = torch.zeros(16, 96, 3)
    rays[..., 0] = 1.0
    return CalibratedSphericalSupportRoleGeometry(
        rays,
        torch.eye(3),
        torch.zeros(3),
        external_calibration_sha256="c" * 64,
        current_radius=0.1,
        landing_radius=0.1,
        vertical_half_extent=0.1,
        min_range=0.1,
        max_range=6.0,
        range_strata_edges=(0.5, 1.5, 3.0),
        age_strata_edges=(0.01, 0.05, 0.2),
    )


def _body_inputs(
    history_length: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(
        1, history_length, -1, -1
    ).clone()
    translation = torch.zeros(1, history_length, 3)
    current = torch.tensor([[[1.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]])
    landing = torch.tensor([[[10.0, 0.0, 0.0], [-10.0, 1.0, 0.0]]])
    return rotation, translation, current, landing


def _runtime_step(tmp_path: Path) -> Mid360SoftwareRuntimeStep:
    message = _custom_msg()
    message.timebase = 100_000_000_000
    alignment = _alignment()
    mapped_end = 100.1 * alignment.scale + alignment.offset_s
    return ingest_livox_custom_msg_runtime_step(
        message,
        expected_frame_id="livox_frame",
        window_index=0,
        capture_end_sensor_s=100.1,
        received_action_time_s=mapped_end + 0.003,
        action_time_s=mapped_end + 0.004,
        alignment=alignment,
        builder=_builder(tmp_path),
    )


def test_real_message_software_path_reaches_selected_only_actor(
    tmp_path: Path,
) -> None:
    """Connect CustomMsg, clock, builder, geometry, selector, and actor mean."""
    geometry = _geometry()
    rotation, translation, current, landing = _body_inputs()
    output = mid360_runtime_step_to_support_geometry(
        _runtime_step(tmp_path),
        geometry=geometry,
        history_body_to_current_rotation=rotation,
        history_body_to_current_translation=translation,
        current_foot_centres_body=current,
        landing_foot_centres_body=landing,
    )
    batch = output.geometry_batch

    assert batch.token_valid.sum().item() == 1
    assert batch.candidate_mask.sum().item() == 1
    assert batch.role_eligibility[:, 0].sum().item() == 1
    assert not batch.role_eligibility[:, 1:].any()
    actor = SharedUniqueSupportActorAdapter(
        geometry.score_feature_dim,
        geometry.terrain_value_dim,
        proprio_dim=6,
        action_dim=3,
        total_budget=8,
    ).eval()
    with torch.inference_mode():
        action, value = actor(
            batch.score_features,
            batch.terrain_values,
            torch.zeros(1, 6),
            batch.token_valid,
            batch.role_eligibility,
            mask_provenance=geometry.provenance(),
        )
    assert action.shape == (1, 3)
    assert value.shape == (1, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(value).all()
    assert output.receipt["schema"] == MID360_SUPPORT_GEOMETRY_BRIDGE_SCHEMA
    assert output.receipt["software_geometry_path_connected"] is True
    assert output.receipt["dynamic_body_input_bytes_bound"] is False
    assert output.receipt["training_ready"] is False


def test_tensor_bridge_rejects_nonbinary_and_nonbroadcast_channels() -> None:
    """Reject event channels that no longer satisfy the strict actor contract."""
    geometry = _geometry()
    body_inputs = _body_inputs()
    event = torch.zeros(1, 1, 5, 16, 96)
    event[:, :, 4] = 1.0

    nonbinary = event.clone()
    nonbinary[:, :, 1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="exactly binary"):
        ray_event_history_to_support_geometry(
            nonbinary,
            geometry=geometry,
            history_body_to_current_rotation=body_inputs[0],
            history_body_to_current_translation=body_inputs[1],
            current_foot_centres_body=body_inputs[2],
            landing_foot_centres_body=body_inputs[3],
        )

    nonbroadcast = event.clone()
    nonbroadcast[:, :, 3, 0, 1] = 0.1
    with pytest.raises(ValueError, match="spatially constant"):
        ray_event_history_to_support_geometry(
            nonbroadcast,
            geometry=geometry,
            history_body_to_current_rotation=body_inputs[0],
            history_body_to_current_translation=body_inputs[1],
            current_foot_centres_body=body_inputs[2],
            landing_foot_centres_body=body_inputs[3],
        )


def test_tensor_bridge_rejects_double_age_and_invalid_return_semantics() -> None:
    """Keep return age absolute and invalid cells exactly zero-valued."""
    geometry = _geometry()
    body_inputs = _body_inputs()
    event = torch.zeros(1, 1, 5, 16, 96)
    event[:, :, 3] = 0.1
    event[:, :, 4] = 1.0
    event[:, :, 0, 0, 0] = 1.0
    event[:, :, 1, 0, 0] = 1.0
    event[:, :, 2, 0, 0] = 0.05
    with pytest.raises(ValueError, match="cannot precede"):
        ray_event_history_to_support_geometry(
            event,
            geometry=geometry,
            history_body_to_current_rotation=body_inputs[0],
            history_body_to_current_translation=body_inputs[1],
            current_foot_centres_body=body_inputs[2],
            landing_foot_centres_body=body_inputs[3],
        )

    invalid_range = torch.zeros(1, 1, 5, 16, 96)
    invalid_range[:, :, 4] = 1.0
    invalid_range[:, :, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="Invalid return ranges"):
        ray_event_history_to_support_geometry(
            invalid_range,
            geometry=geometry,
            history_body_to_current_rotation=body_inputs[0],
            history_body_to_current_translation=body_inputs[1],
            current_foot_centres_body=body_inputs[2],
            landing_foot_centres_body=body_inputs[3],
        )
