# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict bridge from the five-channel MID-360 tensor to H1 geometry."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any, Final

from rsl_rl.modules.support_role_geometry import (
    CalibratedSphericalSupportRoleGeometry,
    SupportRoleGeometryBatch,
)

from .mid360_runtime_integration import (
    Mid360SoftwareRuntimeStep,
    validate_mid360_software_runtime_step,
)

MID360_SUPPORT_GEOMETRY_BRIDGE_SCHEMA: Final[str] = (
    "mid360_support_geometry_software_bridge_v1"
)


@dataclass(frozen=True)
class Mid360SupportGeometryBridgeOutput:
    """H1 tensors plus deliberately narrow software-closure evidence."""

    geometry_batch: SupportRoleGeometryBatch
    receipt: dict[str, Any]


def ray_event_history_to_support_geometry(
    ray_event_history: torch.Tensor,
    *,
    geometry: CalibratedSphericalSupportRoleGeometry,
    history_body_to_current_rotation: torch.Tensor,
    history_body_to_current_translation: torch.Tensor,
    current_foot_centres_body: torch.Tensor,
    landing_foot_centres_body: torch.Tensor,
) -> SupportRoleGeometryBatch:
    """Validate and convert strict ``[B,K,5,H,W]`` event observations."""
    if not isinstance(geometry, CalibratedSphericalSupportRoleGeometry):
        raise TypeError("geometry must be calibrated support-role geometry.")
    if not isinstance(ray_event_history, torch.Tensor):
        raise TypeError("ray_event_history must be a tensor.")
    expected_tail = (5, geometry.height, geometry.width)
    if ray_event_history.ndim != 5 or tuple(ray_event_history.shape[-3:]) != expected_tail:
        raise ValueError(
            "ray_event_history must have shape [B,K,5,H,W] matching geometry."
        )
    if not ray_event_history.dtype.is_floating_point:
        raise TypeError("ray_event_history must use floating-point channels.")
    if ray_event_history.device != geometry.body_ray_directions.device:
        raise ValueError("ray_event_history and geometry must share one device.")
    companion_tensors = (
        history_body_to_current_rotation,
        history_body_to_current_translation,
        current_foot_centres_body,
        landing_foot_centres_body,
    )
    if any(value.device != ray_event_history.device for value in companion_tensors):
        raise ValueError("Geometry, event history, and body inputs must share one device.")
    if not bool(torch.isfinite(ray_event_history).all()):
        raise ValueError("ray_event_history must be finite.")

    range_m = ray_event_history[:, :, 0]
    valid_channel = ray_event_history[:, :, 1]
    return_age_s = ray_event_history[:, :, 2]
    packet_age_map = ray_event_history[:, :, 3]
    frame_valid_map = ray_event_history[:, :, 4]
    if not bool(((valid_channel == 0.0) | (valid_channel == 1.0)).all()):
        raise ValueError("return_valid channel must be exactly binary.")
    if not bool(((frame_valid_map == 0.0) | (frame_valid_map == 1.0)).all()):
        raise ValueError("frame_valid channel must be exactly binary.")
    packet_age_s = packet_age_map[:, :, 0, 0]
    frame_valid = frame_valid_map[:, :, 0, 0] == 1.0
    if not torch.equal(
        packet_age_map,
        packet_age_s[:, :, None, None].expand_as(packet_age_map),
    ):
        raise ValueError("packet_age channel must be spatially constant per frame.")
    if not torch.equal(
        frame_valid_map,
        frame_valid_map[:, :, :1, :1].expand_as(frame_valid_map),
    ):
        raise ValueError("frame_valid channel must be spatially constant per frame.")

    return_valid = valid_channel == 1.0
    if bool((return_valid & ~frame_valid[:, :, None, None]).any()):
        raise ValueError("Invalid history frames cannot contain returns.")
    if bool((return_valid & (range_m <= 0.0)).any()):
        raise ValueError("Valid returns require positive range.")
    if bool((~return_valid & (range_m != 0.0)).any()):
        raise ValueError("Invalid return ranges must be exactly zero.")
    if bool((~return_valid & (return_age_s != 0.0)).any()):
        raise ValueError("Invalid return ages must be exactly zero.")
    if bool((return_age_s < 0.0).any()) or bool((packet_age_s < 0.0).any()):
        raise ValueError("Acquisition ages must be non-negative.")
    if bool(
        (
            return_valid
            & (return_age_s + 1.0e-7 < packet_age_s[:, :, None, None])
        ).any()
    ):
        raise ValueError("Return age cannot precede its packet age.")

    return geometry(
        range_m,
        return_valid,
        return_age_s,
        packet_age_s,
        history_body_to_current_rotation,
        history_body_to_current_translation,
        current_foot_centres_body,
        landing_foot_centres_body,
    )


def mid360_runtime_step_to_support_geometry(
    step: Mid360SoftwareRuntimeStep,
    *,
    geometry: CalibratedSphericalSupportRoleGeometry,
    history_body_to_current_rotation: torch.Tensor,
    history_body_to_current_translation: torch.Tensor,
    current_foot_centres_body: torch.Tensor,
    landing_foot_centres_body: torch.Tensor,
) -> Mid360SupportGeometryBridgeOutput:
    """Validate a real-message software step and construct H1 actor inputs."""
    validate_mid360_software_runtime_step(step)
    device = geometry.body_ray_directions.device
    event_history = torch.from_numpy(step.ray_event_history).to(
        device=device,
        dtype=geometry.body_ray_directions.dtype,
    )
    geometry_batch = ray_event_history_to_support_geometry(
        event_history,
        geometry=geometry,
        history_body_to_current_rotation=history_body_to_current_rotation,
        history_body_to_current_translation=history_body_to_current_translation,
        current_foot_centres_body=current_foot_centres_body,
        landing_foot_centres_body=landing_foot_centres_body,
    )
    geometry_receipt = geometry.receipt()
    receipt = {
        "schema": MID360_SUPPORT_GEOMETRY_BRIDGE_SCHEMA,
        "claim_scope": "software_mid360_to_support_geometry_only",
        "runtime_receipt_payload_sha256": step.receipt_payload_sha256,
        "geometry_configuration_sha256": geometry_receipt[
            "configuration_sha256"
        ],
        "ray_event_history_shape": list(step.ray_event_history.shape),
        "support_token_shape": list(geometry_batch.token_valid.shape),
        "support_role_shape": list(geometry_batch.role_eligibility.shape),
        "software_event_semantics_checked": True,
        "software_geometry_path_connected": True,
        "dynamic_body_input_bytes_bound": False,
        "external_clock_evidence_verified": False,
        "physical_sensor_recording_authenticated": False,
        "external_calibration_evidence_verified": False,
        "body_transform_evidence_verified": False,
        "foot_fk_evidence_verified": False,
        "landing_generator_evidence_verified": False,
        "actor_checkpoint_bound": False,
        "registered_lab_task": False,
        "training_ready": False,
        "g1_closed_loop_verified": False,
    }
    return Mid360SupportGeometryBridgeOutput(
        geometry_batch=geometry_batch,
        receipt=receipt,
    )


__all__ = [
    "MID360_SUPPORT_GEOMETRY_BRIDGE_SCHEMA",
    "Mid360SupportGeometryBridgeOutput",
    "mid360_runtime_step_to_support_geometry",
    "ray_event_history_to_support_geometry",
]
