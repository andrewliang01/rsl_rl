# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Causal command/phase projection of future foot support-query centres."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from numbers import Real
from typing import Final


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


@dataclass(frozen=True)
class CausalFootSupportProjection:
    """Projected centres and causal time-to-query horizons."""

    future_centres_body: torch.Tensor
    support_horizon_s: torch.Tensor
    finite_gate: torch.Tensor


class CausalCommandFootSupportProjector(nn.Module):
    """Project nominal foot support centres under a commanded planar twist.

    This is a deployable query generator, not a terrain-aware footstep planner.
    It rotates each current FK foot offset and translates it by the exact
    constant-twist SE(2) displacement up to that foot's next phase anchor.
    """

    role_order: Final[tuple[str, ...]] = ("left_future", "right_future")

    def __init__(
        self,
        *,
        gait_cycle_s: float,
        min_horizon_s: float,
        max_horizon_s: float,
        left_phase_anchor_rad: float = 0.0,
        right_phase_anchor_rad: float = math.pi,
    ) -> None:
        """Freeze the gait clock and bounded command-projection horizons."""
        super().__init__()
        self.gait_cycle_s = _positive_real("gait_cycle_s", gait_cycle_s)
        self.min_horizon_s = _positive_real("min_horizon_s", min_horizon_s)
        self.max_horizon_s = _positive_real("max_horizon_s", max_horizon_s)
        if self.max_horizon_s < self.min_horizon_s:
            raise ValueError("max_horizon_s must be at least min_horizon_s.")
        anchors = torch.tensor(
            (left_phase_anchor_rad, right_phase_anchor_rad),
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(anchors).all()):
            raise ValueError("Phase anchors must be finite.")
        self.register_buffer("phase_anchors_rad", anchors)

    def receipt(self) -> dict[str, object]:
        """Return the narrow evidence scope of this deterministic projector."""
        return {
            "schema": "causal_command_foot_support_projector_v1",
            "role_order": self.role_order,
            "inputs": (
                "joint_encoder_forward_kinematics_foot_centres",
                "commanded_body_planar_twist",
                "monotonic_clock_gait_phase",
            ),
            "projection": "constant_body_twist_se2_until_next_phase_anchor",
            "terrain_aware_footstep_planner": False,
            "simulator_contact_truth": False,
            "simulator_terrain_truth": False,
            "predicted_touchdown_accuracy_validated": False,
            "training_ready": False,
            "g1_closed_loop_validated": False,
            "gait_cycle_s": self.gait_cycle_s,
            "min_horizon_s": self.min_horizon_s,
            "max_horizon_s": self.max_horizon_s,
            "phase_anchors_rad": tuple(
                float(value) for value in self.phase_anchors_rad.cpu()
            ),
        }

    def forward(
        self,
        current_foot_centres_body: torch.Tensor,
        commanded_planar_twist_body: torch.Tensor,
        gait_phase_sin_cos: torch.Tensor,
    ) -> CausalFootSupportProjection:
        """Return future support-query centres using only causal inputs."""
        self._validate_inputs(
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
            validate_values=True,
        )
        return self._forward_impl(
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
        )

    def forward_native_training(
        self,
        current_foot_centres_body: torch.Tensor,
        commanded_planar_twist_body: torch.Tensor,
        gait_phase_sin_cos: torch.Tensor,
    ) -> CausalFootSupportProjection:
        """Project without host-value checks and return a per-row tensor gate."""
        self._validate_inputs(
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
            validate_values=False,
        )
        return self._forward_impl(
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
        )

    def _forward_impl(
        self,
        current_foot_centres_body: torch.Tensor,
        commanded_planar_twist_body: torch.Tensor,
        gait_phase_sin_cos: torch.Tensor,
    ) -> CausalFootSupportProjection:
        """Run the common tensor-only command projection."""
        input_finite = (
            torch.isfinite(current_foot_centres_body).all(dim=(-1, -2))
            & torch.isfinite(commanded_planar_twist_body).all(dim=-1)
            & torch.isfinite(gait_phase_sin_cos).all(dim=-1)
        )
        phase_norm = torch.linalg.vector_norm(gait_phase_sin_cos, dim=-1)
        phase_gate = phase_norm >= 1.0e-6
        safe_phase_norm = phase_norm.clamp_min(1.0e-6)
        phase = torch.atan2(
            gait_phase_sin_cos[:, 0] / safe_phase_norm,
            gait_phase_sin_cos[:, 1] / safe_phase_norm,
        )
        phase_delta = torch.remainder(
            self.phase_anchors_rad[None, :] - phase[:, None],
            2.0 * math.pi,
        )
        phase_delta = torch.where(
            phase_delta < 1.0e-6,
            torch.full_like(phase_delta, 2.0 * math.pi),
            phase_delta,
        )
        horizon = (
            phase_delta * (self.gait_cycle_s / (2.0 * math.pi))
        ).clamp(self.min_horizon_s, self.max_horizon_s)

        theta = commanded_planar_twist_body[:, 2, None] * horizon
        coefficient_a = horizon * torch.sinc(theta / math.pi)
        coefficient_b = (
            horizon
            * (theta / 2.0)
            * torch.sinc(theta / (2.0 * math.pi)).square()
        )
        velocity_x = commanded_planar_twist_body[:, 0, None]
        velocity_y = commanded_planar_twist_body[:, 1, None]
        translation_x = coefficient_a * velocity_x - coefficient_b * velocity_y
        translation_y = coefficient_b * velocity_x + coefficient_a * velocity_y

        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        foot_x = current_foot_centres_body[..., 0]
        foot_y = current_foot_centres_body[..., 1]
        projected_x = cosine * foot_x - sine * foot_y + translation_x
        projected_y = sine * foot_x + cosine * foot_y + translation_y
        future_centres = torch.stack(
            (projected_x, projected_y, current_foot_centres_body[..., 2]),
            dim=-1,
        )
        finite_gate = (
            input_finite
            & phase_gate
            & torch.isfinite(future_centres).all(dim=(-1, -2))
            & torch.isfinite(horizon).all(dim=-1)
        )
        return CausalFootSupportProjection(
            future_centres_body=future_centres,
            support_horizon_s=horizon,
            finite_gate=finite_gate,
        )

    def _validate_inputs(
        self,
        current_foot_centres_body: torch.Tensor,
        commanded_planar_twist_body: torch.Tensor,
        gait_phase_sin_cos: torch.Tensor,
        *,
        validate_values: bool,
    ) -> None:
        """Validate static structure and optionally synchronize value checks."""
        if current_foot_centres_body.ndim != 3 or tuple(
            current_foot_centres_body.shape[1:]
        ) != (2, 3):
            raise ValueError("current_foot_centres_body must have shape [B,2,3].")
        batch_size = current_foot_centres_body.shape[0]
        if commanded_planar_twist_body.shape != (batch_size, 3):
            raise ValueError("commanded_planar_twist_body must have shape [B,3].")
        if gait_phase_sin_cos.shape != (batch_size, 2):
            raise ValueError("gait_phase_sin_cos must have shape [B,2].")
        values = (
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
        )
        if len({value.device for value in values}) != 1:
            raise ValueError("All support-projector inputs must share one device.")
        if current_foot_centres_body.device != self.phase_anchors_rad.device:
            raise ValueError("Projector inputs and parameters must share one device.")
        if any(not value.dtype.is_floating_point for value in values):
            raise TypeError("All support-projector inputs must be floating point.")
        if not validate_values:
            return
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("All support-projector inputs must be finite.")
        phase_norm = torch.linalg.vector_norm(gait_phase_sin_cos, dim=-1)
        if bool((phase_norm < 1.0e-6).any()):
            raise ValueError("gait_phase_sin_cos must encode a nonzero phase vector.")


__all__ = [
    "CausalCommandFootSupportProjector",
    "CausalFootSupportProjection",
]
