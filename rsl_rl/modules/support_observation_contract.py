# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor-only observation unpacking for the H1 support evidence actor."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class RayEventSupportObservation:
    """Strict five-channel return-event fields and a per-row health gate."""

    range_m: torch.Tensor
    return_valid: torch.Tensor
    return_age_s: torch.Tensor
    packet_age_s: torch.Tensor
    frame_valid: torch.Tensor
    finite_gate: torch.Tensor


@dataclass(frozen=True)
class SupportMotionObservation:
    """Causal body registration, current FK feet, and gait clock phase."""

    history_body_to_current_rotation: torch.Tensor
    history_body_to_current_translation: torch.Tensor
    current_foot_centres_body: torch.Tensor
    gait_phase_sin_cos: torch.Tensor
    finite_gate: torch.Tensor


def unpack_ray_event_support_observation(
    observation: torch.Tensor,
) -> RayEventSupportObservation:
    """Unpack ``[B,K,5,H,W]`` without tensor-dependent Python branching."""
    if observation.ndim != 5 or observation.shape[2] != 5:
        raise ValueError("Ray-event observation must have shape [B,K,5,H,W].")
    if not observation.dtype.is_floating_point:
        raise TypeError("Ray-event observation must be floating point.")
    range_m = observation[:, :, 0]
    valid_channel = observation[:, :, 1]
    return_age_s = observation[:, :, 2]
    packet_age_map = observation[:, :, 3]
    frame_valid_map = observation[:, :, 4]
    return_valid = valid_channel == 1.0
    frame_valid = frame_valid_map[:, :, 0, 0] == 1.0
    packet_age_s = packet_age_map[:, :, 0, 0]
    finite = torch.isfinite(observation).all(dim=(-1, -2, -3, -4))
    valid_binary = (
        (valid_channel == 0.0) | (valid_channel == 1.0)
    ).all(dim=(-1, -2, -3))
    frame_binary = (
        (frame_valid_map == 0.0) | (frame_valid_map == 1.0)
    ).all(dim=(-1, -2, -3))
    packet_broadcast = (
        packet_age_map
        == packet_age_s[:, :, None, None].expand_as(packet_age_map)
    ).all(dim=(-1, -2, -3))
    frame_broadcast = (
        frame_valid_map
        == frame_valid_map[:, :, :1, :1].expand_as(frame_valid_map)
    ).all(dim=(-1, -2, -3))
    valid_semantics = (
        ~return_valid
        | (
            (range_m > 0.0)
            & (return_age_s >= 0.0)
            & frame_valid[:, :, None, None]
        )
    ).all(dim=(-1, -2, -3))
    invalid_semantics = (
        return_valid | ((range_m == 0.0) & (return_age_s == 0.0))
    ).all(dim=(-1, -2, -3))
    packet_semantics = (
        torch.isfinite(packet_age_s) & (packet_age_s >= 0.0)
    ).all(dim=-1)
    finite_gate = (
        finite
        & valid_binary
        & frame_binary
        & packet_broadcast
        & frame_broadcast
        & valid_semantics
        & invalid_semantics
        & packet_semantics
    )
    return RayEventSupportObservation(
        range_m=range_m,
        return_valid=return_valid,
        return_age_s=return_age_s,
        packet_age_s=packet_age_s,
        frame_valid=frame_valid,
        finite_gate=finite_gate,
    )


def unpack_support_motion_observation(
    observation: torch.Tensor,
    *,
    history_length: int,
) -> SupportMotionObservation:
    """Unpack ``[B,12K+8]`` body transforms, FK feet, and phase clock."""
    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, Integral)
        or int(history_length) <= 0
    ):
        raise ValueError("history_length must be a positive integer.")
    resolved_history = int(history_length)
    expected_width = 12 * resolved_history + 8
    if observation.ndim != 2 or observation.shape[1] != expected_width:
        raise ValueError(
            f"Support motion observation must have shape [B,{expected_width}]."
        )
    if not observation.dtype.is_floating_point:
        raise TypeError("Support motion observation must be floating point.")
    batch_size = observation.shape[0]
    rotation_end = 9 * resolved_history
    translation_end = rotation_end + 3 * resolved_history
    foot_end = translation_end + 6
    history_rotation = observation[:, :rotation_end].reshape(
        batch_size, resolved_history, 3, 3
    )
    history_translation = observation[:, rotation_end:translation_end].reshape(
        batch_size, resolved_history, 3
    )
    current_feet = observation[:, translation_end:foot_end].reshape(
        batch_size, 2, 3
    )
    gait_phase = observation[:, foot_end:]
    finite_gate = torch.isfinite(observation).all(dim=-1)
    return SupportMotionObservation(
        history_body_to_current_rotation=history_rotation,
        history_body_to_current_translation=history_translation,
        current_foot_centres_body=current_feet,
        gait_phase_sin_cos=gait_phase,
        finite_gate=finite_gate,
    )


__all__ = [
    "RayEventSupportObservation",
    "SupportMotionObservation",
    "unpack_ray_event_support_observation",
    "unpack_support_motion_observation",
]
