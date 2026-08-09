# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map-free current-frame/recurrent MID-360 student for frozen ECMM control.

The deployment boundary is intentionally narrow.  At every policy step the
student receives only the original 96-D ECMM proprioception, one strict
MID-360 frame with channels ``range/valid/message_age/new_frame``, and, in GRU
mode, its hidden state.  The frame token is concatenated with the frozen
proprioception feature B64; a current-frame encoder or GRU then predicts the
64-D perception latent ``A_hat`` consumed by the same frozen fusion head C and
action distribution as the integrity-bound M90 ECMM actor.

Teacher maps, ground-truth poses, terrain metadata, and future frames are not
constructor inputs and are never indexed during ``forward``.  They may coexist
in a training TensorDict for a critic or an auxiliary loss, but cannot enter
the deployable action path.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.m2m_frozen_ecmm import M2MFrozenECMMCore
from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import unpad_trajectories


class _SphericalConvBlock(nn.Module):
    """2-D convolution with a circular azimuth seam and zero elevation padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        if any(size <= 0 or size % 2 == 0 for size in kernel_size):
            raise ValueError(f"kernel_size must contain positive odd values, got {kernel_size}.")
        self.vertical_padding = kernel_size[0] // 2
        self.horizontal_padding = kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.activation = nn.ELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Azimuth is periodic, while the top and bottom elevation rows are not.
        value = F.pad(
            value,
            (self.horizontal_padding, self.horizontal_padding, 0, 0),
            mode="circular",
        )
        value = F.pad(value, (0, 0, self.vertical_padding, self.vertical_padding))
        return self.activation(self.conv(value))


class M2MStrictFrameTokenizer(nn.Module):
    """Normalize and tokenize one strict ``[1,4,16,96]`` MID-360 frame."""

    frame_shape: tuple[int, int, int, int] = (1, 4, 16, 96)
    channels: tuple[str, str, str, str] = (
        "range_m",
        "valid",
        "message_age_s",
        "new_frame",
    )

    def __init__(
        self,
        *,
        near_range_m: float,
        far_range_m: float,
        message_period_s: float,
        max_age_s: float,
        hidden_channels: Sequence[int] = (16, 32),
        token_dim: int = 128,
        pooled_spatial_size: tuple[int, int] = (2, 6),
    ) -> None:
        super().__init__()
        for name, value in (
            ("near_range_m", near_range_m),
            ("far_range_m", far_range_m),
            ("message_period_s", message_period_s),
            ("max_age_s", max_age_s),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite scalar, got {value!r}.")
        if near_range_m < 0.0 or far_range_m <= near_range_m:
            raise ValueError(
                "Range bounds must satisfy 0 <= near_range_m < far_range_m, got "
                f"{near_range_m} and {far_range_m}."
            )
        if message_period_s <= 0.0:
            raise ValueError(f"message_period_s must be positive, got {message_period_s}.")
        if max_age_s < message_period_s:
            raise ValueError(
                "max_age_s must be greater than or equal to message_period_s, got "
                f"{max_age_s} < {message_period_s}."
            )
        if type(token_dim) is not int or token_dim <= 0:
            raise ValueError(f"token_dim must be a positive integer, got {token_dim!r}.")
        if (
            len(pooled_spatial_size) != 2
            or any(type(size) is not int or size <= 0 for size in pooled_spatial_size)
        ):
            raise ValueError(
                "pooled_spatial_size must contain two positive integers, got "
                f"{pooled_spatial_size}."
            )
        channels_tuple = tuple(hidden_channels)
        if not channels_tuple or any(type(width) is not int or width <= 0 for width in channels_tuple):
            raise ValueError(f"hidden_channels must contain positive integers, got {channels_tuple}.")

        self.near_range_m = float(near_range_m)
        self.far_range_m = float(far_range_m)
        self.message_period_s = float(message_period_s)
        self.max_age_s = float(max_age_s)
        self.token_dim = token_dim
        self.pooled_spatial_size = pooled_spatial_size

        blocks: list[nn.Module] = []
        in_channels = len(self.channels)
        for index, out_channels in enumerate(channels_tuple):
            blocks.append(
                _SphericalConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 5) if index == 0 else (3, 3),
                    stride=(2, 2),
                )
            )
            in_channels = out_channels
        self.spatial_encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pooled_spatial_size)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pooled_spatial_size[0] * pooled_spatial_size[1], token_dim),
            nn.LayerNorm(token_dim),
            nn.ELU(),
        )

    def _normalize(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Strict MID-360 frame must use float16, bfloat16, or float32 metric channels, "
                f"got {frame.dtype}."
            )
        if frame.ndim not in (5, 6) or tuple(frame.shape[-4:]) != self.frame_shape:
            raise ValueError(
                "Strict MID-360 frame must have layout [B,1,4,16,96] or "
                f"[T,B,1,4,16,96], got {tuple(frame.shape)}."
            )

        # Rollout storage may retain strict frames in fp16/bf16.  The
        # tokenizer always performs normalization and convolution in float32,
        # matching the student weights and avoiding low-precision range math.
        channels = frame.squeeze(-4).float()
        range_m = channels[..., 0:1, :, :]
        valid = torch.nan_to_num(channels[..., 1:2, :, :], nan=0.0).clamp(0.0, 1.0)
        message_age_s = channels[..., 2:3, :, :]
        new_frame = torch.nan_to_num(channels[..., 3:4, :, :], nan=0.0).clamp(0.0, 1.0)

        # Invalid ranges cannot affect the token through arbitrary stored
        # values.  The observation producer owns exact binary/finiteness
        # checks; this path remains free of per-step GPU-to-host syncs.
        range_m = torch.nan_to_num(
            range_m,
            nan=self.far_range_m,
            posinf=self.far_range_m,
            neginf=self.near_range_m,
        ).clamp(self.near_range_m, self.far_range_m)
        range_unit = (range_m - self.near_range_m) / (self.far_range_m - self.near_range_m)
        range_normalized = torch.where(valid > 0.5, 2.0 * range_unit - 1.0, torch.zeros_like(range_unit))
        age_normalized = torch.nan_to_num(
            message_age_s,
            nan=self.max_age_s,
            posinf=self.max_age_s,
            neginf=0.0,
        ).clamp(0.0, self.max_age_s) / self.max_age_s
        return torch.cat((range_normalized, valid, age_normalized, new_frame), dim=-3)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize(frame)
        leading_shape = normalized.shape[:-3]
        flattened = normalized.reshape(-1, len(self.channels), *self.frame_shape[-2:])
        feature = self.projection(self.pool(self.spatial_encoder(flattened)))
        return feature.reshape(*leading_shape, self.token_dim)


class M2MMapFreeRecurrentStudent(nn.Module):
    """Current-frame or GRU student for causal strict MID-360 observations."""

    is_recurrent: bool = True
    latent_dim: int = 64
    proprio_dim: int = 96
    action_dim: int = 29

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        strict_frame_set: str,
        proprio_sets: Sequence[str],
        frozen_ecmm_checkpoint_path: str,
        frozen_ecmm_expected_sha256: str,
        frozen_ecmm_actor_cfg: Mapping[str, Any],
        frozen_ecmm_actor_state_dict_key: str = "actor_state_dict",
        frame_near_range_m: float,
        frame_far_range_m: float,
        frame_message_period_s: float,
        frame_max_age_s: float,
        tokenizer_hidden_channels: Sequence[int] = (16, 32),
        tokenizer_dim: int = 128,
        tokenizer_pooled_spatial_size: tuple[int, int] = (2, 6),
        temporal_mode: str = "gru",
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        latent_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(obs, TensorDict):
            raise TypeError(f"obs must be a TensorDict, got {type(obs).__name__}.")
        if len(obs.batch_size) != 1 or obs.batch_size[0] <= 0:
            raise ValueError(f"Construction obs must have one positive batch dimension, got {obs.batch_size}.")
        if type(output_dim) is not int or output_dim != self.action_dim:
            raise ValueError(f"M2M Unitree-G1 student requires output_dim={self.action_dim}, got {output_dim!r}.")
        if not isinstance(strict_frame_set, str) or not strict_frame_set:
            raise ValueError("strict_frame_set must be a non-empty observation-group name.")
        if strict_frame_set not in obs:
            raise KeyError(f"Construction obs is missing strict frame group {strict_frame_set!r}.")
        strict_sample = obs[strict_frame_set]
        if (
            strict_sample.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tuple(strict_sample.shape[1:]) != M2MStrictFrameTokenizer.frame_shape
        ):
            raise ValueError(
                f"{strict_frame_set!r} must have float16/bfloat16/float32 shape [B,1,4,16,96], got "
                f"dtype={strict_sample.dtype}, shape={tuple(strict_sample.shape)}."
            )

        if not isinstance(obs_set, str) or obs_set not in obs_groups:
            raise KeyError(f"obs_groups is missing observation set {obs_set!r}.")
        proprio_groups = tuple(proprio_sets)
        if not proprio_groups or any(not isinstance(group, str) or not group for group in proprio_groups):
            raise ValueError("proprio_sets must contain non-empty observation-group names.")
        if len(set(proprio_groups)) != len(proprio_groups):
            raise ValueError(f"proprio_sets contains duplicate groups: {proprio_groups}.")
        if strict_frame_set in proprio_groups:
            raise ValueError("strict_frame_set cannot also be a proprioception group.")
        active_groups = tuple(obs_groups[obs_set])
        expected_active = (*proprio_groups, strict_frame_set)
        if active_groups != expected_active:
            raise ValueError(
                "Deployable actor observation groups must contain exactly the explicit original proprio groups "
                f"plus the strict frame: expected={expected_active}, actual={active_groups}."
            )
        proprio_dim = 0
        proprio_group_dims: dict[str, int] = {}
        for group in proprio_groups:
            if group not in obs:
                raise KeyError(f"Construction obs is missing proprioception group {group!r}.")
            value = obs[group]
            if value.ndim != 2:
                raise ValueError(f"Proprioception group {group!r} must be [B,D], got {tuple(value.shape)}.")
            if value.dtype != torch.float32:
                raise ValueError(f"Proprioception group {group!r} must use float32, got {value.dtype}.")
            proprio_group_dims[group] = value.shape[-1]
            proprio_dim += value.shape[-1]
        if proprio_dim != self.proprio_dim:
            raise ValueError(
                f"Frozen ECMM B requires exactly {self.proprio_dim} proprioception values, got {proprio_dim}."
            )

        if not isinstance(frozen_ecmm_actor_cfg, Mapping):
            raise TypeError("frozen_ecmm_actor_cfg must be a mapping.")
        actor_cfg = copy.deepcopy(dict(frozen_ecmm_actor_cfg))
        reserved = {"obs", "obs_groups", "obs_set", "output_dim"}
        conflicts = sorted(reserved.intersection(actor_cfg))
        if conflicts:
            raise ValueError(
                "frozen_ecmm_actor_cfg cannot override constructor-owned fields: "
                f"{conflicts}."
            )
        elevation_set = actor_cfg.get("elevation_set", "height_scan_actor")
        if not isinstance(elevation_set, str) or not elevation_set:
            raise ValueError("Frozen actor elevation_set must be a non-empty string.")
        if elevation_set == strict_frame_set or elevation_set in proprio_groups:
            raise ValueError("Frozen actor elevation_set must be separate from every deployment input group.")
        spatial_size = tuple(actor_cfg.get("vision_spatial_size", (25, 17)))
        if spatial_size != (16, 96):
            raise ValueError(f"Frozen M90 ECMM actor must use vision_spatial_size=(16,96), got {spatial_size}.")

        # Construct the frozen actor with a local shape-only placeholder.  No
        # M90 observation is required at deployment and no external teacher
        # tensor is consulted even if one exists in ``obs``.
        reference = obs[proprio_groups[0]]
        frozen_obs = TensorDict(
            {
                **{group: obs[group] for group in proprio_groups},
                elevation_set: torch.full(
                    (obs.batch_size[0], 1, *spatial_size),
                    float(actor_cfg.get("depth_camera_far", 6.0)),
                    dtype=reference.dtype,
                    device=reference.device,
                ),
            },
            batch_size=obs.batch_size,
            device=obs.device,
        )
        frozen_obs_groups = {obs_set: [*proprio_groups, elevation_set]}
        frozen_actor = PropMLPElevationFusionModel(
            obs=frozen_obs,
            obs_groups=frozen_obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            **actor_cfg,
        )
        self.ecmm_core = M2MFrozenECMMCore(
            frozen_actor,
            checkpoint_path=frozen_ecmm_checkpoint_path,
            expected_sha256=frozen_ecmm_expected_sha256,
            actor_state_dict_key=frozen_ecmm_actor_state_dict_key,
        )

        for name, value in (
            ("gru_hidden_dim", gru_hidden_dim),
            ("gru_num_layers", gru_num_layers),
            ("latent_hidden_dim", latent_hidden_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if temporal_mode not in ("current", "gru"):
            raise ValueError(f"temporal_mode must be 'current' or 'gru', got {temporal_mode!r}.")

        self.strict_frame_set = strict_frame_set
        self.proprio_sets = proprio_groups
        self.proprio_group_dims = proprio_group_dims
        self.obs_groups = list(expected_active)
        self.obs_set = obs_set
        self.temporal_mode = temporal_mode
        self.is_recurrent = temporal_mode == "gru"
        self.frame_tokenizer = M2MStrictFrameTokenizer(
            near_range_m=frame_near_range_m,
            far_range_m=frame_far_range_m,
            message_period_s=frame_message_period_s,
            max_age_s=frame_max_age_s,
            hidden_channels=tokenizer_hidden_channels,
            token_dim=tokenizer_dim,
            pooled_spatial_size=tokenizer_pooled_spatial_size,
        )
        temporal_input_dim = tokenizer_dim + self.ecmm_core.proprio_feature_dim
        if temporal_mode == "gru":
            self.gru: nn.GRU | None = nn.GRU(
                input_size=temporal_input_dim,
                hidden_size=gru_hidden_dim,
                num_layers=gru_num_layers,
            )
            self.current_encoder: nn.Module | None = None
        else:
            self.gru = None
            self.current_encoder = nn.Sequential(
                nn.Linear(temporal_input_dim, gru_hidden_dim),
                nn.LayerNorm(gru_hidden_dim),
                nn.ELU(),
            )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, latent_hidden_dim),
            nn.ELU(),
            nn.Linear(latent_hidden_dim, self.latent_dim),
        )
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.temporal_input_dim = temporal_input_dim
        self._hidden_state: torch.Tensor | None = None

    @property
    def distribution(self) -> Distribution:
        """Expose the exact frozen ECMM distribution to PPO."""
        distribution = self.ecmm_core.actor.distribution
        if distribution is None:  # Guarded by M2MFrozenECMMCore's contract.
            raise RuntimeError("Frozen ECMM action distribution is unavailable.")
        return distribution

    def train(self, mode: bool = True) -> M2MMapFreeRecurrentStudent:
        """Train only the tokenizer/GRU/latent head and keep ECMM frozen."""
        super().train(mode)
        self.ecmm_core.eval()
        return self

    def _validate_hidden_state(
        self,
        hidden_state: torch.Tensor | None,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if hidden_state is None:
            return None
        if not isinstance(hidden_state, torch.Tensor):
            raise TypeError("M2M GRU hidden_state must be a tensor or None.")
        expected_shape = (self.gru_num_layers, batch_size, self.gru_hidden_dim)
        if tuple(hidden_state.shape) != expected_shape:
            raise ValueError(
                f"GRU hidden_state must have shape {expected_shape}, got {tuple(hidden_state.shape)}."
            )
        if hidden_state.device != reference.device or hidden_state.dtype != reference.dtype:
            raise ValueError(
                "GRU hidden_state must match token dtype/device, got "
                f"{hidden_state.dtype}/{hidden_state.device} and {reference.dtype}/{reference.device}."
            )
        return hidden_state

    def _validate_deployment_observations(self, obs: TensorDict) -> None:
        if not isinstance(obs, TensorDict):
            raise TypeError(f"obs must be a TensorDict, got {type(obs).__name__}.")
        if len(obs.batch_size) not in (1, 2) or any(size <= 0 for size in obs.batch_size):
            raise ValueError(
                "Student observations must have batch layout [B] or [T,B], got "
                f"{obs.batch_size}."
            )
        if self.strict_frame_set not in obs:
            raise KeyError(f"Observation is missing strict frame group {self.strict_frame_set!r}.")
        frame = obs[self.strict_frame_set]
        expected_frame_shape = (*obs.batch_size, *M2MStrictFrameTokenizer.frame_shape)
        if (
            frame.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tuple(frame.shape) != expected_frame_shape
        ):
            raise ValueError(
                "Strict MID-360 frame has an invalid runtime contract: expected "
                f"shape={expected_frame_shape} and fp16/bf16/fp32, got "
                f"shape={tuple(frame.shape)}, dtype={frame.dtype}."
            )
        for group, dimension in self.proprio_group_dims.items():
            if group not in obs:
                raise KeyError(f"Observation is missing proprioception group {group!r}.")
            value = obs[group]
            expected_shape = (*obs.batch_size, dimension)
            if value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Proprioception group {group!r} must have float32 shape {expected_shape}, "
                    f"got dtype={value.dtype}, shape={tuple(value.shape)}."
                )

    @staticmethod
    def _validate_masks(masks: torch.Tensor, leading_shape: torch.Size) -> None:
        if masks.dtype != torch.bool or tuple(masks.shape) != tuple(leading_shape):
            raise ValueError(
                "Trajectory masks must be bool [T,N] matching padded inputs, got "
                f"dtype={masks.dtype}, shape={tuple(masks.shape)}, inputs={tuple(leading_shape)}."
            )

    def _predict_latent_and_proprio(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned ``(A_hat, B64)`` without encoding proprioception twice."""
        temporal_input, proprio_features = self._encode_temporal_input(obs)

        if self.temporal_mode == "current":
            if hidden_state is not None:
                raise ValueError("temporal_mode='current' does not accept a recurrent hidden_state.")
            if self.current_encoder is None:
                raise RuntimeError("Current-frame encoder was not constructed.")
            current_feature = self.current_encoder(temporal_input)
            latent_a = self.latent_head(current_feature)
            if masks is None:
                return latent_a, proprio_features
            if temporal_input.ndim != 3:
                raise ValueError("Masked current-frame mode requires padded [T,N,...] observations.")
            self._validate_masks(masks, temporal_input.shape[:2])
            return (
                unpad_trajectories(latent_a, masks),  # type: ignore[return-value]
                unpad_trajectories(proprio_features, masks),  # type: ignore[return-value]
            )

        if self.gru is None:
            raise RuntimeError("GRU temporal encoder was not constructed.")
        if masks is None:
            if temporal_input.ndim != 2:
                raise ValueError(
                    "A multi-step GRU observation requires trajectory masks; "
                    f"got temporal input shape {tuple(temporal_input.shape)} with masks=None."
                )
            recurrent_state = hidden_state if hidden_state is not None else self._hidden_state
            recurrent_state = self._validate_hidden_state(
                recurrent_state,
                batch_size=temporal_input.shape[0],
                reference=temporal_input,
            )
            output, next_hidden = self.gru(temporal_input.unsqueeze(0), recurrent_state)
            self._hidden_state = next_hidden
            return self.latent_head(output.squeeze(0)), proprio_features

        if temporal_input.ndim != 3:
            raise ValueError(
                "Padded GRU mode requires [T,N,1,4,16,96] strict frames, "
                f"got temporal input shape {tuple(temporal_input.shape)}."
            )
        self._validate_masks(masks, temporal_input.shape[:2])
        initial_hidden = self._validate_hidden_state(
            hidden_state,
            batch_size=temporal_input.shape[1],
            reference=temporal_input,
        )
        if initial_hidden is None:
            initial_hidden = temporal_input.new_zeros(
                self.gru_num_layers,
                temporal_input.shape[1],
                self.gru_hidden_dim,
            )
        padded_output, _ = self.gru(temporal_input, initial_hidden)
        padded_latent = self.latent_head(padded_output)
        return (
            unpad_trajectories(padded_latent, masks),  # type: ignore[return-value]
            unpad_trajectories(proprio_features, masks),  # type: ignore[return-value]
        )

    def _encode_temporal_input(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode deployable observations into ``(frame_token+B64, B64)``."""
        self._validate_deployment_observations(obs)
        token = self.frame_tokenizer(obs[self.strict_frame_set])
        proprio_features = self.ecmm_core.encode_proprio(obs)
        if token.shape[:-1] != proprio_features.shape[:-1]:
            raise RuntimeError(
                "Strict-frame token and frozen B leading dimensions disagree: "
                f"token={tuple(token.shape)}, B={tuple(proprio_features.shape)}."
            )
        temporal_input = torch.cat((token, proprio_features), dim=-1)
        if temporal_input.shape[-1] != self.temporal_input_dim:
            raise RuntimeError(f"Temporal input has invalid shape {tuple(temporal_input.shape)}.")
        return temporal_input, proprio_features

    def predict_latent_A(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict A_hat64 in rollout or padded-sequence/BPTT mode."""
        return self._predict_latent_and_proprio(obs, masks, hidden_state)[0]

    def predict_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility alias used by M2M latent/action supervision code."""
        return self.predict_latent_A(obs, masks=masks, hidden_state=hidden_state)

    def _action_from_latent(
        self,
        proprio_features: torch.Tensor,
        latent_a: torch.Tensor,
        *,
        stochastic_output: bool,
    ) -> torch.Tensor:
        if proprio_features.shape[:-1] != latent_a.shape[:-1]:
            raise RuntimeError(
                "Proprioception and student latent layouts disagree: "
                f"B={tuple(proprio_features.shape)}, A={tuple(latent_a.shape)}."
            )
        raw_action = self.ecmm_core.actor.mlp(torch.cat((proprio_features, latent_a), dim=-1))
        if stochastic_output:
            self.distribution.update(raw_action)
            return self.distribution.sample()
        return self.distribution.deterministic_output(raw_action)

    def forward_with_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action29, A_hat64)`` for PPO or auxiliary M2M losses."""
        latent_a, proprio_features = self._predict_latent_and_proprio(obs, masks, hidden_state)
        action = self._action_from_latent(
            proprio_features,
            latent_a,
            stochastic_output=stochastic_output,
        )
        return action, latent_a

    def predict_latent_and_action_mean(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(A_hat64, action_mean29)`` from one recurrent transition.

        Auxiliary M2M training should prefer this method over separately
        calling ``predict_latent`` and ``forward``: two separate calls would
        advance rollout-mode GRU state twice.
        """
        action_mean, latent_a = self.forward_with_latent(
            obs,
            masks=masks,
            hidden_state=hidden_state,
            stochastic_output=False,
        )
        return latent_a, action_mean

    def predict_padded_latent_and_action_mean(
        self,
        obs: TensorDict,
        masks: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded ``(A_hat, action_mean)`` for arbitrary TBPTT chunks.

        Unlike the standard PPO ``forward(..., masks=...)`` path, this method
        deliberately does **not** call :func:`unpad_trajectories`.  C10 batches
        are independent time-major chunks ``[L,B]`` whose valid lengths need
        not satisfy the reshape invariant of RSL-RL's split-trajectory PPO
        layout.  Outputs remain ``[L,B,64]`` and ``[L,B,29]``; C11 must apply
        ``masks`` to every padded loss term.

        The GRU is evaluated exactly once and the model's rollout hidden state
        is not mutated.
        """
        if self.temporal_mode != "gru" or self.gru is None:
            raise ValueError("Padded TBPTT inference requires temporal_mode='gru'.")
        temporal_input, proprio_features = self._encode_temporal_input(obs)
        if temporal_input.ndim != 3:
            raise ValueError(
                "Padded TBPTT observations must have layout [L,B,...], got "
                f"temporal input shape {tuple(temporal_input.shape)}."
            )
        if masks.dtype != torch.bool:
            raise ValueError(f"Padded TBPTT masks must use bool, got {masks.dtype}.")
        if tuple(masks.shape) == (*temporal_input.shape[:2], 1):
            masks_2d = masks.squeeze(-1)
        elif tuple(masks.shape) == tuple(temporal_input.shape[:2]):
            masks_2d = masks
        else:
            raise ValueError(
                "Padded TBPTT masks must have shape [L,B] or [L,B,1], got "
                f"{tuple(masks.shape)} for input {tuple(temporal_input.shape[:2])}."
            )
        if masks_2d.device != temporal_input.device:
            raise ValueError(
                "Padded TBPTT masks and observations must share a device, got "
                f"{masks_2d.device} and {temporal_input.device}."
            )
        initial_hidden = self._validate_hidden_state(
            hidden_state,
            batch_size=temporal_input.shape[1],
            reference=temporal_input,
        )
        if initial_hidden is None:  # The public signature requires a tensor.
            raise ValueError("Padded TBPTT hidden_state cannot be None.")

        # C10 zero-fills padding already; applying the mask again prevents a
        # biased frozen B from turning padded zero proprioception into a
        # non-zero temporal input.  Valid-prefix outputs are unchanged.
        masked_input = temporal_input * masks_2d.unsqueeze(-1).to(temporal_input.dtype)
        padded_output, _ = self.gru(masked_input, initial_hidden)
        padded_latent = self.latent_head(padded_output)
        padded_action_mean = self._action_from_latent(
            proprio_features,
            padded_latent,
            stochastic_output=False,
        )
        return padded_latent, padded_action_mean

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return student actions with the standard RSL-RL model signature."""
        return self.forward_with_latent(
            obs,
            masks=masks,
            hidden_state=hidden_state,
            stochastic_output=stochastic_output,
        )[0]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> None:
        """Reset all hidden state or zero only environments marked done."""
        if self.temporal_mode == "current":
            if hidden_state is not None:
                raise ValueError("temporal_mode='current' has no hidden state to reset.")
            self._hidden_state = None
            return
        if dones is None:
            if hidden_state is not None:
                if hidden_state.ndim != 3 or hidden_state.shape[0] != self.gru_num_layers:
                    raise ValueError(
                        "Custom GRU hidden state must be [num_layers,B,hidden_dim], got "
                        f"{tuple(hidden_state.shape)}."
                    )
                if hidden_state.shape[-1] != self.gru_hidden_dim:
                    raise ValueError(
                        f"Custom GRU hidden dimension must be {self.gru_hidden_dim}, "
                        f"got {hidden_state.shape[-1]}."
                    )
            self._hidden_state = hidden_state
            return
        if hidden_state is not None:
            raise NotImplementedError("Per-environment reset with a custom hidden state is unsupported.")
        if self._hidden_state is None:
            return
        done_mask = dones.reshape(-1).to(device=self._hidden_state.device, dtype=torch.bool)
        if done_mask.numel() != self._hidden_state.shape[1]:
            raise ValueError(
                "dones must contain one value per recurrent environment, got "
                f"{done_mask.numel()} for batch {self._hidden_state.shape[1]}."
            )
        self._hidden_state = self._hidden_state.masked_fill(done_mask.view(1, -1, 1), 0.0)

    def get_hidden_state(self) -> torch.Tensor | None:
        """Return the rollout GRU hidden state for recurrent storage."""
        return self._hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach all hidden state, or only selected environments, from autograd."""
        if self.temporal_mode == "current":
            return
        if self._hidden_state is None:
            return
        if dones is None:
            self._hidden_state = self._hidden_state.detach()
            return
        done_mask = dones.reshape(-1).to(device=self._hidden_state.device, dtype=torch.bool)
        if done_mask.numel() != self._hidden_state.shape[1]:
            raise ValueError(
                "dones must contain one value per recurrent environment, got "
                f"{done_mask.numel()} for batch {self._hidden_state.shape[1]}."
            )
        self._hidden_state = torch.where(
            done_mask.view(1, -1, 1),
            self._hidden_state.detach(),
            self._hidden_state,
        )

    def update_normalization(self, obs: TensorDict) -> None:
        """Keep checkpoint-bound ECMM normalization frozen (intentional no-op)."""
        del obs

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

    def parameter_audit(self) -> dict[str, Any]:
        """Return a machine-readable deployment/freeze receipt."""

        def counts(module: nn.Module | None) -> dict[str, int]:
            if module is None:
                return {"total": 0, "trainable": 0}
            parameters = list(module.parameters())
            return {
                "total": sum(parameter.numel() for parameter in parameters),
                "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            }

        return {
            "model": "m2m_map_free_student",
            "temporal_mode": self.temporal_mode,
            "is_recurrent": self.is_recurrent,
            "deployment_inputs": {
                "ordered_groups": self.obs_groups,
                "proprio_sets": list(self.proprio_sets),
                "proprio_group_dims": dict(self.proprio_group_dims),
                "proprio_dim": self.proprio_dim,
                "proprio_feature_to_temporal_dim": self.ecmm_core.proprio_feature_dim,
                "strict_frame_set": self.strict_frame_set,
                "strict_frame_shape": list(M2MStrictFrameTokenizer.frame_shape),
                "strict_frame_channels": list(M2MStrictFrameTokenizer.channels),
                "strict_frame_storage_dtypes": ["float16", "bfloat16", "float32"],
                "strict_frame_compute_dtype": "float32",
                "strict_frame_near_range_m": self.frame_tokenizer.near_range_m,
                "strict_frame_far_range_m": self.frame_tokenizer.far_range_m,
                "strict_frame_message_period_s": self.frame_tokenizer.message_period_s,
                "strict_frame_max_age_s": self.frame_tokenizer.max_age_s,
                "recurrent_state": (
                    [self.gru_num_layers, "batch", self.gru_hidden_dim]
                    if self.is_recurrent
                    else False
                ),
                "teacher_map": False,
                "ground_truth_pose": False,
                "terrain_metadata": False,
                "future_frames": False,
            },
            "temporal_input": {
                "sources": ["strict_frame_token", "frozen_proprio_B64"],
                "dimension": self.temporal_input_dim,
                "uses_frozen_teacher_encoder_A": False,
            },
            "frozen_control_path": [
                "obs_normalizer",
                "proprio_encoder_B",
                "fusion_action_head_C",
                "action_distribution",
            ],
            "sequence_contracts": {
                "ppo_masks": "split_trajectory_then_unpad",
                "m2m_tbptt": "retain_padded_L_B_and_mask_losses",
            },
            "latent_a_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "components": {
                "frame_tokenizer": counts(self.frame_tokenizer),
                "gru": counts(self.gru),
                "current_encoder": counts(self.current_encoder),
                "latent_head": counts(self.latent_head),
                "frozen_ecmm": counts(self.ecmm_core),
            },
            "frozen_ecmm": self.ecmm_core.parameter_audit(),
        }


__all__ = ["M2MMapFreeRecurrentStudent", "M2MStrictFrameTokenizer"]
