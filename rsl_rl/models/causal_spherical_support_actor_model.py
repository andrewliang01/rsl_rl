# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL actor model for the causal spherical support evidence bottleneck."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from tensordict import TensorDict

from rsl_rl.modules import (
    CalibratedSphericalSupportRoleGeometry,
    CausalCommandFootSupportProjector,
    CausalSphericalSupportEvidencePipeline,
    EmpiricalNormalization,
    HiddenState,
    SharedUniqueSupportActorAdapter,
    unpack_ray_event_support_observation,
    unpack_support_motion_observation,
)
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def uniform_spherical_ray_directions(
    height: int,
    width: int,
    *,
    vertical_fov_degrees: tuple[float, float],
    azimuth_offset_degrees: float = 0.0,
) -> torch.Tensor:
    """Return row-major unit rays on ``[-pi,pi)`` spherical angle bins."""
    if height <= 0 or width <= 0:
        raise ValueError("Spherical ray image dimensions must be positive.")
    lower, upper = (float(value) for value in vertical_fov_degrees)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("vertical_fov_degrees must be finite and increasing.")
    if not math.isfinite(float(azimuth_offset_degrees)):
        raise ValueError("azimuth_offset_degrees must be finite.")
    elevation = torch.linspace(
        math.radians(lower),
        math.radians(upper),
        steps=height,
        dtype=torch.float32,
    )
    azimuth = (
        torch.arange(width, dtype=torch.float32) * (2.0 * math.pi / width)
        - math.pi
        + math.radians(float(azimuth_offset_degrees))
    )
    elevation_grid, azimuth_grid = torch.meshgrid(
        elevation,
        azimuth,
        indexing="ij",
    )
    cosine = torch.cos(elevation_grid)
    return torch.stack(
        (
            cosine * torch.cos(azimuth_grid),
            cosine * torch.sin(azimuth_grid),
            torch.sin(elevation_grid),
        ),
        dim=-1,
    )


class CausalSphericalSupportActorModel(nn.Module):
    """Map explicit proprio/event/motion groups to selected-only actions."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        ray_event_set: str,
        support_motion_set: str,
        command_indices: Sequence[int],
        external_calibration_sha256: str,
        history_length: int = 5,
        ray_spatial_size: tuple[int, int] = (16, 96),
        vertical_fov_degrees: tuple[float, float] = (-52.0, 7.0),
        azimuth_offset_degrees: float = 0.0,
        sensor_to_body_rotation: Sequence[float] = (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        sensor_origin_body: Sequence[float] = (0.0, 0.0, 0.0),
        min_range: float = 0.1,
        max_range: float = 6.0,
        range_strata_edges: Sequence[float] = (0.5, 1.5, 3.0, 5.0),
        age_strata_edges: Sequence[float] = (0.05, 0.15, 0.3, 0.5),
        current_support_radius: float = 0.18,
        future_support_radius: float = 0.28,
        support_vertical_half_extent: float = 0.20,
        gait_cycle_s: float = 0.8,
        min_support_horizon_s: float = 0.05,
        max_support_horizon_s: float = 0.8,
        total_budget: int = 16,
        score_dim: int = 32,
        value_embedding_dim: int = 32,
        hidden_dim: int = 128,
        obs_normalization: bool = False,
        distribution_cfg: Mapping[str, object] | None = None,
    ) -> None:
        """Build the actor while freezing every observation and geometry contract."""
        super().__init__()
        if obs_set not in obs_groups:
            raise ValueError(f"Unknown observation set {obs_set!r}.")
        self.obs_set = obs_set
        self.ray_event_set = ray_event_set
        self.support_motion_set = support_motion_set
        self.history_length = int(history_length)
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        if ray_event_set == support_motion_set:
            raise ValueError("Ray-event and support-motion groups must differ.")
        active_groups = list(obs_groups[obs_set])
        required = {ray_event_set, support_motion_set}
        if not required.issubset(active_groups):
            raise ValueError(
                "Actor observation groups must include ray-event and support-motion sets."
            )
        self.proprio_groups = [
            group for group in active_groups if group not in required
        ]
        if not self.proprio_groups:
            raise ValueError("At least one proprioception group is required.")
        self.proprio_dim = 0
        for group in self.proprio_groups:
            if obs[group].ndim != 2:
                raise ValueError(f"Proprioception group {group!r} must be [B,D].")
            self.proprio_dim += int(obs[group].shape[-1])

        height, width = (int(value) for value in ray_spatial_size)
        expected_ray_tail = (self.history_length, 5, height, width)
        if tuple(obs[ray_event_set].shape[1:]) != expected_ray_tail:
            raise ValueError(
                f"Ray-event group must have tail {expected_ray_tail}."
            )
        expected_motion_width = 12 * self.history_length + 8
        if tuple(obs[support_motion_set].shape[1:]) != (expected_motion_width,):
            raise ValueError(
                f"Support-motion group must have width {expected_motion_width}."
            )
        resolved_command_indices = tuple(int(value) for value in command_indices)
        if len(resolved_command_indices) != 3 or len(set(resolved_command_indices)) != 3:
            raise ValueError("command_indices must contain three unique indices.")
        if min(resolved_command_indices) < 0 or max(resolved_command_indices) >= self.proprio_dim:
            raise ValueError("command_indices are outside concatenated proprioception.")
        self.command_indices = resolved_command_indices
        self.register_buffer(
            "_command_index",
            torch.tensor(resolved_command_indices, dtype=torch.long),
            persistent=True,
        )

        rays = uniform_spherical_ray_directions(
            height,
            width,
            vertical_fov_degrees=vertical_fov_degrees,
            azimuth_offset_degrees=azimuth_offset_degrees,
        )
        rotation = torch.as_tensor(sensor_to_body_rotation, dtype=torch.float32)
        if rotation.numel() != 9:
            raise ValueError("sensor_to_body_rotation must contain nine values.")
        origin = torch.as_tensor(sensor_origin_body, dtype=torch.float32)
        if origin.numel() != 3:
            raise ValueError("sensor_origin_body must contain three values.")
        geometry = CalibratedSphericalSupportRoleGeometry(
            rays,
            rotation.reshape(3, 3),
            origin.reshape(3),
            external_calibration_sha256=external_calibration_sha256,
            current_radius=current_support_radius,
            landing_radius=future_support_radius,
            vertical_half_extent=support_vertical_half_extent,
            min_range=min_range,
            max_range=max_range,
            range_strata_edges=tuple(float(value) for value in range_strata_edges),
            age_strata_edges=tuple(float(value) for value in age_strata_edges),
        )
        projector = CausalCommandFootSupportProjector(
            gait_cycle_s=gait_cycle_s,
            min_horizon_s=min_support_horizon_s,
            max_horizon_s=max_support_horizon_s,
        )
        self.support_pipeline = CausalSphericalSupportEvidencePipeline(
            projector,
            geometry,
        )
        self.obs_normalization = bool(obs_normalization)
        self.obs_normalizer: nn.Module
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.proprio_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            resolved_distribution = copy.deepcopy(dict(distribution_cfg))
            class_name = resolved_distribution.pop("class_name")
            if not isinstance(class_name, str):
                raise TypeError("distribution_cfg.class_name must be a string.")
            distribution_class: type[Distribution] = resolve_callable(class_name)
            self.distribution: Distribution | None = distribution_class(
                output_dim,
                **resolved_distribution,
            )
            distribution_input_dim = self.distribution.input_dim
            if not isinstance(distribution_input_dim, int):
                raise TypeError(
                    "Causal support actor currently requires a distribution with "
                    "one integer input dimension."
                )
            actor_output_dim = distribution_input_dim
        else:
            self.distribution = None
            actor_output_dim = int(output_dim)
        self.support_actor = SharedUniqueSupportActorAdapter(
            score_feature_dim=geometry.score_feature_dim,
            terrain_value_dim=geometry.terrain_value_dim,
            proprio_dim=self.proprio_dim,
            action_dim=actor_output_dim,
            total_budget=total_budget,
            score_dim=score_dim,
            value_embedding_dim=value_embedding_dim,
            hidden_dim=hidden_dim,
        )
        # This model is actor-only.  Keep the adapter's legacy value modules out
        # of optimization and, crucially, out of the native forward path.
        self.support_actor.critic_backbone.requires_grad_(False)
        self.support_actor.value_head.requires_grad_(False)
        self.register_buffer(
            "_last_finite_gate",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run selected-only native training and poison invalid rows with NaN."""
        del hidden_state
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        raw_proprio = torch.cat(
            [obs[group] for group in self.proprio_groups],
            dim=-1,
        )
        proprio = self.obs_normalizer(raw_proprio)
        event = unpack_ray_event_support_observation(obs[self.ray_event_set])
        motion = unpack_support_motion_observation(
            obs[self.support_motion_set],
            history_length=self.history_length,
        )
        command = torch.index_select(raw_proprio, 1, self._command_index)
        evidence = self.support_pipeline.forward_native_training(
            event.range_m,
            event.return_valid,
            event.return_age_s,
            event.packet_age_s,
            motion.history_body_to_current_rotation,
            motion.history_body_to_current_translation,
            motion.current_foot_centres_body,
            command,
            motion.gait_phase_sin_cos,
        )
        action_parameters, actor_gate = (
            self.support_actor.forward_native_actor_training(
                evidence.geometry.score_features,
                evidence.geometry.terrain_values,
                proprio,
                evidence.geometry.token_valid,
                evidence.geometry.role_eligibility,
                mask_provenance=self.support_pipeline.geometry.provenance(),
            )
        )
        finite_gate = (
            event.finite_gate
            & motion.finite_gate
            & evidence.finite_gate
            & actor_gate
        )
        self._last_finite_gate = finite_gate
        action_parameters = torch.where(
            finite_gate[:, None],
            action_parameters,
            torch.full_like(action_parameters, torch.nan),
        )
        if self.distribution is None:
            return action_parameters
        if stochastic_output:
            self.distribution.update(action_parameters)
            return self.distribution.sample()
        return self.distribution.deterministic_output(action_parameters)

    @property
    def last_finite_gate(self) -> torch.Tensor:
        """Return the most recent device-side fail gate."""
        return self._last_finite_gate

    def receipt(self) -> dict[str, object]:
        """Return model architecture facts without claiming task validation."""
        return {
            "schema": "causal_spherical_support_actor_model_v1",
            "ray_event_set": self.ray_event_set,
            "support_motion_set": self.support_motion_set,
            "proprio_groups": tuple(self.proprio_groups),
            "command_indices": self.command_indices,
            "history_length": self.history_length,
            "invalid_row_action": "nan_fail_closed",
            "embedded_legacy_value_modules_frozen": True,
            "external_critic_configuration_unchanged": False,
            "support_pipeline": self.support_pipeline.receipt(),
            "registered_lab_task": False,
            "gpu_latency_measured": False,
            "training_ready": False,
        }

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        """Reset recurrent state (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return no recurrent hidden state."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent state (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the current distribution mean."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the current distribution standard deviation."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return summed action entropy."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return distribution parameters for PPO storage."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return action log probability."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return KL(old || new)."""
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update only concatenated proprioception normalization."""
        if self.obs_normalization:
            proprio = torch.cat(
                [obs[group] for group in self.proprio_groups],
                dim=-1,
            )
            self.obs_normalizer.update(proprio)

    def as_jit(self) -> nn.Module:
        """Fail closed until the dynamic geometry export is receipt-bound."""
        raise RuntimeError("Causal support actor JIT export is not yet authorized.")

    def as_onnx(self, verbose: bool, input_mode: str = "split") -> nn.Module:
        """Fail closed until the dynamic geometry export is receipt-bound."""
        del verbose, input_mode
        raise RuntimeError("Causal support actor ONNX export is not yet authorized.")


__all__ = [
    "CausalSphericalSupportActorModel",
    "uniform_spherical_ray_directions",
]
