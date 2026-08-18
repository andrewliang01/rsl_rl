# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map-free motion-conditioned deformable memory for strict MID-360 packets."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class _CircularConvBlock(nn.Module):
    """Convolve with circular azimuth and ordinary elevation padding."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (1, 1, 0, 0), mode="circular")
        value = F.pad(value, (0, 0, 1, 1))
        return F.silu(self.norm(self.conv(value)))


class _StrictPacketSpatialEncoder(nn.Module):
    """Encode range/valid/age while keeping the spatial raster topology."""

    frame_shape = (1, 4, 16, 96)
    frame_channels = (
        "range_m",
        "valid",
        "winning_subframe_age_20ms",
        "new_frame",
    )

    def __init__(
        self,
        *,
        near_range_m: float,
        far_range_m: float,
        max_age_s: float,
        hidden_channels: Sequence[int],
    ) -> None:
        super().__init__()
        if not 0.0 <= near_range_m < far_range_m:
            raise ValueError("range bounds must satisfy 0 <= near < far")
        if not math.isfinite(max_age_s) or max_age_s <= 0.0:
            raise ValueError("max_age_s must be finite and positive")
        widths = tuple(hidden_channels)
        if len(widths) != 2 or any(type(width) is not int or width <= 0 for width in widths):
            raise ValueError("spatial_hidden_channels must contain exactly two positive integers")
        self.near_range_m = float(near_range_m)
        self.far_range_m = float(far_range_m)
        self.max_age_s = float(max_age_s)
        self.output_channels = widths[-1]
        self.output_height = 4
        self.output_width = 24
        self.encoder = nn.Sequential(
            _CircularConvBlock(3, widths[0], stride=2),
            _CircularConvBlock(widths[0], widths[1], stride=2),
        )

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frame.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("strict packet frame must use fp16, bf16, or fp32")
        if frame.ndim != 5 or tuple(frame.shape[-4:]) != self.frame_shape:
            raise ValueError("strict packet frame must have shape [B,1,4,16,96]")
        channels = frame[:, 0].float()
        range_m = torch.nan_to_num(
            channels[:, 0:1],
            nan=self.far_range_m,
            posinf=self.far_range_m,
            neginf=self.near_range_m,
        ).clamp(self.near_range_m, self.far_range_m)
        valid = torch.nan_to_num(channels[:, 1:2], nan=0.0).clamp(0.0, 1.0)
        age = torch.nan_to_num(
            channels[:, 2:3],
            nan=self.max_age_s,
            posinf=self.max_age_s,
            neginf=0.0,
        ).clamp(0.0, self.max_age_s)
        new_frame = torch.nan_to_num(channels[:, 3, 0, 0], nan=0.0).clamp(0.0, 1.0)
        range_unit = (range_m - self.near_range_m) / (self.far_range_m - self.near_range_m)
        normalized_range = torch.where(
            valid > 0.5,
            2.0 * range_unit - 1.0,
            torch.zeros_like(range_unit),
        )
        inputs = torch.cat((normalized_range, valid, age / self.max_age_s), dim=1)
        spatial = self.encoder(inputs)
        expected = (frame.shape[0], self.output_channels, self.output_height, self.output_width)
        if tuple(spatial.shape) != expected:
            raise RuntimeError(f"spatial encoder produced {tuple(spatial.shape)}, expected {expected}")
        return spatial, new_frame > 0.5


class _MotionConditionedDeformableFusion(nn.Module):
    """Sample a persistent spatial memory with learned motion-conditioned offsets."""

    def __init__(
        self,
        *,
        channels: int,
        height: int,
        width: int,
        prop_dim: int,
        gru_dim: int,
        motion_dim: int,
        samples: int,
        max_elevation_offset_cells: float,
        max_azimuth_offset_cells: float,
    ) -> None:
        super().__init__()
        if type(samples) is not int or samples <= 0:
            raise ValueError("deformable_samples must be a positive integer")
        if max_elevation_offset_cells <= 0.0 or max_azimuth_offset_cells <= 0.0:
            raise ValueError("deformable offset bounds must be positive")
        self.channels = channels
        self.height = height
        self.width = width
        self.samples = samples
        self.max_elevation_offset_cells = float(max_elevation_offset_cells)
        self.max_azimuth_offset_cells = float(max_azimuth_offset_cells)
        self.motion_encoder = nn.Sequential(
            nn.Linear(prop_dim * 2 + gru_dim, motion_dim),
            nn.LayerNorm(motion_dim),
            nn.SiLU(),
        )
        self.global_offset = nn.Linear(motion_dim, 2)
        self.local_offset = nn.Conv2d(channels + motion_dim, samples * 2, kernel_size=1)
        self.attention_logits = nn.Conv2d(channels + motion_dim, samples + 1, kernel_size=1)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )

    def _sample(self, memory: torch.Tensor, offsets_cells: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = memory.shape
        dtype = memory.dtype
        device = memory.device
        y = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
        x = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
        offset_y = offsets_cells[:, :, 0]
        offset_x = offsets_cells[:, :, 1]
        sample_y = (y + offset_y).clamp(0.0, float(height - 1))
        sample_x = torch.remainder(x + offset_x, float(width))

        # Add one circular column on both sides.  Wrapped coordinates are then
        # represented without a discontinuity at +/-pi for bilinear sampling.
        extended = torch.cat((memory[..., -1:], memory, memory[..., :1]), dim=-1)
        grid_x = 2.0 * (sample_x + 1.0) / float(width + 1) - 1.0
        grid_y = 2.0 * sample_y / float(max(height - 1, 1)) - 1.0
        grids = torch.stack((grid_x, grid_y), dim=-1)
        repeated = extended[:, None].expand(-1, self.samples, -1, -1, -1)
        sampled = F.grid_sample(
            repeated.reshape(batch * self.samples, self.channels, height, width + 2),
            grids.reshape(batch * self.samples, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(batch, self.samples, self.channels, height, width)

    def forward(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
        current_prop: torch.Tensor,
        previous_prop: torch.Tensor,
        gru_hidden: torch.Tensor,
        initialized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion = self.motion_encoder(torch.cat((current_prop, previous_prop, gru_hidden), dim=-1))
        motion_map = motion[:, :, None, None].expand(-1, -1, self.height, self.width)
        conditioned = torch.cat((current, motion_map), dim=1)
        local = self.local_offset(conditioned).reshape(
            current.shape[0], self.samples, 2, self.height, self.width
        )
        global_offset = self.global_offset(motion)[:, None, :, None, None]
        offset_scale = local.new_tensor(
            [self.max_elevation_offset_cells, self.max_azimuth_offset_cells]
        ).view(1, 1, 2, 1, 1)
        offsets = torch.tanh(local + global_offset) * offset_scale
        sampled = self._sample(memory, offsets)
        initialized_5d = initialized.view(current.shape[0], 1, 1, 1, 1)
        sampled = sampled * initialized_5d

        logits = self.attention_logits(conditioned)
        initialized_4d = initialized.view(current.shape[0], 1, 1, 1)
        history_logits = logits[:, 1:] + (initialized_4d - 1.0) * 1.0e4
        weights = torch.softmax(torch.cat((logits[:, :1], history_logits), dim=1), dim=1)
        fused = weights[:, :1] * current
        fused = fused + torch.sum(weights[:, 1:, None] * sampled, dim=1)
        return self.output(fused), offsets


class M2MLivoxDeformableMemoryActor(nn.Module):
    """Persistent 10 Hz spatial memory with map-free deformable alignment.

    A strict complete packet is spatially encoded only as robot-frame evidence.
    At each ``new_frame`` pulse, proprioception-conditioned offsets sample the
    previous latent raster, local attention fuses it with the current raster,
    and a persistent GRU produces ``A64``.  Between packet deliveries the
    complete hidden state and ``A64`` are held, while the 50 Hz proprioception
    branch and control head remain live.
    """

    is_recurrent: bool = True
    proprio_dim: int = 96
    action_dim: int = 29
    latent_dim: int = 64

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        strict_frame_set: str,
        proprio_sets: Sequence[str],
        frame_near_range_m: float,
        frame_far_range_m: float,
        frame_max_age_s: float,
        frame_age_semantics: str,
        spatial_hidden_channels: Sequence[int] = (16, 16),
        deformable_samples: int = 4,
        motion_dim: int = 64,
        max_elevation_offset_cells: float = 2.0,
        max_azimuth_offset_cells: float = 8.0,
        gru_hidden_dim: int = 128,
        latent_hidden_dim: int = 128,
        prop_feature_dim: int = 64,
        prop_hidden_dims: Sequence[int] = (128,),
        fusion_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs, TensorDict) or len(obs.batch_size) != 1:
            raise TypeError("construction obs must be a one-batch TensorDict")
        if output_dim != self.action_dim:
            raise ValueError(f"Unitree-G1 actor requires output_dim={self.action_dim}")
        if frame_age_semantics != "winning_subframe_age_20ms":
            raise ValueError("deformable Livox memory requires winning_subframe_age_20ms")
        proprio_groups = tuple(proprio_sets)
        expected_groups = (*proprio_groups, strict_frame_set)
        if tuple(obs_groups.get(obs_set, ())) != expected_groups:
            raise ValueError(
                "actor groups must be exactly proprio followed by strict frame: "
                f"expected={expected_groups}, actual={tuple(obs_groups.get(obs_set, ()))}"
            )
        if strict_frame_set not in obs:
            raise KeyError(f"construction obs is missing {strict_frame_set!r}")
        if tuple(obs[strict_frame_set].shape[1:]) != _StrictPacketSpatialEncoder.frame_shape:
            raise ValueError("strict frame construction shape must be [B,1,4,16,96]")

        group_dims: dict[str, int] = {}
        proprio_dim = 0
        for group in proprio_groups:
            value = obs[group]
            if value.ndim != 2 or value.dtype != torch.float32:
                raise ValueError(f"proprio group {group!r} must be float32 [B,D]")
            group_dims[group] = value.shape[-1]
            proprio_dim += value.shape[-1]
        if proprio_dim != self.proprio_dim:
            raise ValueError(f"actor requires {self.proprio_dim} proprio values")
        for name, value in (
            ("motion_dim", motion_dim),
            ("gru_hidden_dim", gru_hidden_dim),
            ("latent_hidden_dim", latent_hidden_dim),
            ("prop_feature_dim", prop_feature_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        distribution_values = copy.deepcopy(
            dict(
                distribution_cfg
                or {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
            )
        )
        class_name = distribution_values.pop("class_name", None)
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("distribution_cfg.class_name must be a non-empty string")
        distribution_class: type[Distribution] = resolve_callable(class_name)

        self.spatial_encoder = _StrictPacketSpatialEncoder(
            near_range_m=frame_near_range_m,
            far_range_m=frame_far_range_m,
            max_age_s=frame_max_age_s,
            hidden_channels=spatial_hidden_channels,
        )
        channels = self.spatial_encoder.output_channels
        height = self.spatial_encoder.output_height
        width = self.spatial_encoder.output_width
        self.obs_normalizer: nn.Module = (
            EmpiricalNormalization(proprio_dim) if obs_normalization else nn.Identity()
        )
        self.prop_mlp = MLP(proprio_dim, prop_feature_dim, prop_hidden_dims, activation)
        self.deformable_fusion = _MotionConditionedDeformableFusion(
            channels=channels,
            height=height,
            width=width,
            prop_dim=prop_feature_dim,
            gru_dim=gru_hidden_dim,
            motion_dim=motion_dim,
            samples=deformable_samples,
            max_elevation_offset_cells=max_elevation_offset_cells,
            max_azimuth_offset_cells=max_azimuth_offset_cells,
        )
        self.recurrent_cell = nn.GRUCell(channels + prop_feature_dim, gru_hidden_dim)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, latent_hidden_dim),
            nn.SiLU(),
            nn.Linear(latent_hidden_dim, self.latent_dim),
        )
        self.distribution = distribution_class(output_dim, **distribution_values)
        self.mlp = MLP(
            prop_feature_dim + self.latent_dim,
            self.distribution.input_dim,
            fusion_hidden_dims,
            activation,
        )
        self.distribution.init_mlp_weights(self.mlp)

        self.strict_frame_set = strict_frame_set
        self.proprio_sets = proprio_groups
        self.proprio_group_dims = group_dims
        self.obs_groups = list(expected_groups)
        self.obs_set = obs_set
        self.frame_age_semantics = frame_age_semantics
        self.obs_normalization = obs_normalization
        self.spatial_hidden_channels = tuple(spatial_hidden_channels)
        self.deformable_samples = deformable_samples
        self.motion_dim = motion_dim
        self.max_elevation_offset_cells = float(max_elevation_offset_cells)
        self.max_azimuth_offset_cells = float(max_azimuth_offset_cells)
        self.gru_hidden_dim = gru_hidden_dim
        self.latent_hidden_dim = latent_hidden_dim
        self.prop_feature_dim = prop_feature_dim
        self.prop_hidden_dims = tuple(prop_hidden_dims)
        self.fusion_hidden_dims = tuple(fusion_hidden_dims)
        self.activation = activation
        self.distribution_config = {"class_name": class_name, **distribution_values}
        self.memory_shape = (channels, height, width)
        self.memory_dim = channels * height * width
        self.hidden_state_dim = (
            gru_hidden_dim + self.memory_dim + prop_feature_dim + self.latent_dim + 1
        )
        self._hidden_state: torch.Tensor | None = None
        self._last_offsets: torch.Tensor | None = None

    def _proprio(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([obs[group] for group in self.proprio_sets], dim=-1)
        return self.prop_mlp(self.obs_normalizer(proprio))

    def _zero_state(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(1, batch, self.hidden_state_dim)

    def _validate_state(
        self, state: torch.Tensor | None, batch: int, reference: torch.Tensor
    ) -> torch.Tensor:
        if state is None:
            return self._zero_state(batch, reference)
        expected = (1, batch, self.hidden_state_dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"hidden_state must be {expected}, got {tuple(state.shape)}")
        if state.device != reference.device or state.dtype != reference.dtype:
            raise ValueError("hidden_state must match observation dtype and device")
        return state

    def _unpack(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = state.squeeze(0)
        cursor = 0
        gru = flat[:, cursor : cursor + self.gru_hidden_dim]
        cursor += self.gru_hidden_dim
        memory = flat[:, cursor : cursor + self.memory_dim].reshape(-1, *self.memory_shape)
        cursor += self.memory_dim
        previous_prop = flat[:, cursor : cursor + self.prop_feature_dim]
        cursor += self.prop_feature_dim
        latent = flat[:, cursor : cursor + self.latent_dim]
        cursor += self.latent_dim
        initialized = flat[:, cursor : cursor + 1]
        return gru, memory, previous_prop, latent, initialized

    def _pack(
        self,
        gru: torch.Tensor,
        memory: torch.Tensor,
        previous_prop: torch.Tensor,
        latent: torch.Tensor,
        initialized: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (gru, memory.flatten(1), previous_prop, latent, initialized), dim=-1
        ).unsqueeze(0)

    def _transition(
        self,
        frame: torch.Tensor,
        prop: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current, new_frame = self.spatial_encoder(frame)
        gru, memory, previous_prop, held_latent, initialized = self._unpack(state)
        fused, offsets = self.deformable_fusion(
            current,
            memory,
            prop,
            previous_prop,
            gru,
            initialized,
        )
        pooled = fused.mean(dim=(-2, -1))
        candidate_gru = self.recurrent_cell(torch.cat((pooled, prop), dim=-1), gru)
        candidate_latent = self.latent_head(candidate_gru)
        update = new_frame[:, None]
        next_gru = torch.where(update, candidate_gru, gru)
        next_memory = torch.where(update[:, :, None, None], fused, memory)
        next_prop = torch.where(update, prop, previous_prop)
        next_latent = torch.where(update, candidate_latent, held_latent)
        next_initialized = torch.where(update, torch.ones_like(initialized), initialized)
        self._last_offsets = offsets
        return next_latent, self._pack(
            next_gru, next_memory, next_prop, next_latent, next_initialized
        )

    def _forward_latent_and_prop(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame = obs[self.strict_frame_set]
        if masks is None:
            if len(obs.batch_size) != 1:
                raise ValueError("multi-step recurrent observations require trajectory masks")
            prop = self._proprio(obs)
            state = self._validate_state(
                hidden_state if hidden_state is not None else self._hidden_state,
                frame.shape[0],
                prop,
            )
            latent, next_state = self._transition(frame, prop, state)
            self._hidden_state = next_state
            return latent, prop

        if len(obs.batch_size) != 2 or masks.dtype != torch.bool:
            raise ValueError("padded recurrent observations require bool masks [T,N]")
        if tuple(masks.shape) != tuple(obs.batch_size):
            raise ValueError("trajectory masks must match TensorDict [T,N] batch size")
        time, batch = obs.batch_size
        state = self._validate_state(hidden_state, batch, frame)
        padded_latents: list[torch.Tensor] = []
        padded_props: list[torch.Tensor] = []
        for step in range(time):
            step_obs = obs[step]
            prop = self._proprio(step_obs)
            latent, candidate_state = self._transition(step_obs[self.strict_frame_set], prop, state)
            active = masks[step].view(1, batch, 1)
            state = torch.where(active, candidate_state, state)
            padded_latents.append(latent)
            padded_props.append(prop)
        return (
            unpad_trajectories(torch.stack(padded_latents), masks),
            unpad_trajectories(torch.stack(padded_props), masks),
        )

    def predict_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_latent_and_prop(obs, masks, hidden_state)[0]

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent, prop = self._forward_latent_and_prop(obs, masks, hidden_state)
        raw_action = self.mlp(torch.cat((prop, latent), dim=-1))
        if stochastic_output:
            self.distribution.update(raw_action)
            return self.distribution.sample()
        return self.distribution.deterministic_output(raw_action)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> None:
        if dones is None:
            self._hidden_state = hidden_state
            return
        if hidden_state is not None:
            raise NotImplementedError("per-environment reset with custom hidden_state is unsupported")
        if self._hidden_state is None:
            return
        done = dones.reshape(-1).to(device=self._hidden_state.device, dtype=torch.bool)
        if done.numel() != self._hidden_state.shape[1]:
            raise ValueError("dones must contain one value per recurrent environment")
        self._hidden_state = self._hidden_state.masked_fill(done.view(1, -1, 1), 0.0)

    def get_hidden_state(self) -> torch.Tensor | None:
        return self._hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._hidden_state is None:
            return
        self._hidden_state = self._hidden_state.detach()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            proprio = torch.cat([obs[group] for group in self.proprio_sets], dim=-1)
            self.obs_normalizer.update(proprio)  # type: ignore[attr-defined]

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def architecture_receipt(self) -> dict[str, Any]:
        return {
            "training_initialization": "random_no_pretrained_policy",
            "temporal_model": "motion_conditioned_deformable_spatial_memory_plus_persistent_gru",
            "actor_inputs": {
                "ordered_groups": list(self.obs_groups),
                "strict_frame_set": self.strict_frame_set,
                "updates_only_on_new_10hz_packet": True,
                "uses_map": False,
                "uses_ground_truth_pose": False,
                "uses_future_frames": False,
            },
            "strict_frame_channels": list(_StrictPacketSpatialEncoder.frame_channels),
            "frame_age_semantics": self.frame_age_semantics,
            "spatial_hidden_channels": list(self.spatial_hidden_channels),
            "spatial_memory_shape": list(self.memory_shape),
            "deformable_samples": self.deformable_samples,
            "motion_conditioning": "current_B64_previous_packet_B64_persistent_GRU",
            "max_elevation_offset_cells": self.max_elevation_offset_cells,
            "max_azimuth_offset_cells": self.max_azimuth_offset_cells,
            "gru_hidden_dim": self.gru_hidden_dim,
            "packed_hidden_state_dim": self.hidden_state_dim,
            "latent_a_dim": self.latent_dim,
            "latent_b_dim": self.prop_feature_dim,
            "fusion_hidden_dims": list(self.fusion_hidden_dims),
            "distribution": copy.deepcopy(self.distribution_config),
        }
