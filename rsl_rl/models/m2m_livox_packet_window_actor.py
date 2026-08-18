# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map-free fixed-packet-window actor for strict 10 Hz MID-360 input."""

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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        self.vertical_padding = kernel_size[0] // 2
        self.horizontal_padding = kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.activation = nn.ELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(
            value,
            (self.horizontal_padding, self.horizontal_padding, 0, 0),
            mode="circular",
        )
        value = F.pad(value, (0, 0, self.vertical_padding, self.vertical_padding))
        return self.activation(self.conv(value))


class _PacketTokenizer(nn.Module):
    """Normalize and tokenize one strict ``[1,4,16,96]`` packet raster."""

    frame_shape = (1, 4, 16, 96)
    channels = ("range_m", "valid", "winning_subframe_age_20ms", "packet_present")

    def __init__(
        self,
        *,
        near_range_m: float,
        far_range_m: float,
        message_period_s: float,
        max_age_s: float,
        frame_age_semantics: str,
        hidden_channels: Sequence[int],
        token_dim: int,
        pooled_spatial_size: tuple[int, int],
    ) -> None:
        super().__init__()
        for name, value in (
            ("near_range_m", near_range_m),
            ("far_range_m", far_range_m),
            ("message_period_s", message_period_s),
            ("max_age_s", max_age_s),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= near_range_m < far_range_m:
            raise ValueError("range bounds must satisfy 0 <= near < far")
        if message_period_s <= 0.0 or max_age_s < message_period_s:
            raise ValueError("age horizon must cover at least one message period")
        if frame_age_semantics != "winning_subframe_age_20ms":
            raise ValueError("F17 requires winning_subframe_age_20ms")
        widths = tuple(hidden_channels)
        if not widths or any(type(width) is not int or width <= 0 for width in widths):
            raise ValueError("hidden_channels must contain positive integers")
        if type(token_dim) is not int or token_dim <= 0:
            raise ValueError("token_dim must be a positive integer")
        if len(pooled_spatial_size) != 2 or any(
            type(size) is not int or size <= 0 for size in pooled_spatial_size
        ):
            raise ValueError("pooled_spatial_size must contain two positive integers")

        self.near_range_m = float(near_range_m)
        self.far_range_m = float(far_range_m)
        self.message_period_s = float(message_period_s)
        self.max_age_s = float(max_age_s)
        self.frame_age_semantics = frame_age_semantics
        self.token_dim = token_dim
        blocks: list[nn.Module] = []
        in_channels = len(self.channels)
        for index, out_channels in enumerate(widths):
            blocks.append(
                _CircularConvBlock(
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
            nn.Linear(
                in_channels * pooled_spatial_size[0] * pooled_spatial_size[1],
                token_dim,
            ),
            nn.LayerNorm(token_dim),
            nn.ELU(),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("packet history must use fp16, bf16, or fp32")
        if frame.ndim < 5 or tuple(frame.shape[-4:]) != self.frame_shape:
            raise ValueError("packet raster must end with [1,4,16,96]")
        channels = frame.squeeze(-4).float()
        range_m = channels[..., 0:1, :, :]
        valid = torch.nan_to_num(channels[..., 1:2, :, :], nan=0.0).clamp(0.0, 1.0)
        age = channels[..., 2:3, :, :]
        present = torch.nan_to_num(channels[..., 3:4, :, :], nan=0.0).clamp(0.0, 1.0)
        range_m = torch.nan_to_num(
            range_m,
            nan=self.far_range_m,
            posinf=self.far_range_m,
            neginf=self.near_range_m,
        ).clamp(self.near_range_m, self.far_range_m)
        range_unit = (range_m - self.near_range_m) / (
            self.far_range_m - self.near_range_m
        )
        normalized_range = torch.where(
            valid > 0.5,
            2.0 * range_unit - 1.0,
            torch.zeros_like(range_unit),
        )
        normalized_age = torch.nan_to_num(
            age,
            nan=self.max_age_s,
            posinf=self.max_age_s,
            neginf=0.0,
        ).clamp(0.0, self.max_age_s) / self.max_age_s
        normalized = torch.cat(
            (normalized_range, valid, normalized_age, present), dim=-3
        )
        leading = normalized.shape[:-3]
        flattened = normalized.reshape(-1, len(self.channels), 16, 96)
        token = self.projection(self.pool(self.spatial_encoder(flattened)))
        return token.reshape(*leading, self.token_dim)


class M2MLivoxPacketWindowActor(nn.Module):
    """Encode exactly ``K`` delivered packets without a map or persistent state.

    The observation producer owns the rolling packet window.  Each packet is
    encoded by one shared circular CNN and the ordered tokens are reduced by a
    shared GRU initialized from zero on every policy call.  Consequently H1,
    H3, H5, and H10 have identical trainable parameter counts; only the causal
    evidence window differs.
    """

    is_recurrent: bool = False
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
        packet_window_set: str,
        proprio_sets: Sequence[str],
        history_packets: int,
        frame_near_range_m: float,
        frame_far_range_m: float,
        frame_message_period_s: float,
        frame_max_age_s: float,
        frame_age_semantics: str,
        tokenizer_hidden_channels: Sequence[int] = (16, 32),
        tokenizer_dim: int = 128,
        tokenizer_pooled_spatial_size: tuple[int, int] = (2, 6),
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
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
        if type(output_dim) is not int or output_dim != self.action_dim:
            raise ValueError(f"Unitree-G1 packet-window actor requires output_dim={self.action_dim}")
        if type(history_packets) is not int or history_packets not in (1, 3, 5, 10):
            raise ValueError("history_packets must be exactly one of 1, 3, 5, or 10")
        if not isinstance(obs_set, str) or obs_set not in obs_groups:
            raise KeyError(f"obs_groups is missing {obs_set!r}")
        if not isinstance(packet_window_set, str) or packet_window_set not in obs:
            raise KeyError(f"construction obs is missing packet window {packet_window_set!r}")

        proprio_groups = tuple(proprio_sets)
        expected_groups = (*proprio_groups, packet_window_set)
        if tuple(obs_groups[obs_set]) != expected_groups:
            raise ValueError(
                "actor groups must be exactly proprio followed by the packet window: "
                f"expected={expected_groups}, actual={tuple(obs_groups[obs_set])}"
            )
        window = obs[packet_window_set]
        expected_window = (history_packets, *_PacketTokenizer.frame_shape)
        if window.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("packet window must use fp16, bf16, or fp32")
        if tuple(window.shape[1:]) != expected_window:
            raise ValueError(
                f"packet window must be [B,{history_packets},1,4,16,96], got {tuple(window.shape)}"
            )

        group_dims: dict[str, int] = {}
        proprio_dim = 0
        for group in proprio_groups:
            if group not in obs:
                raise KeyError(f"construction obs is missing proprio group {group!r}")
            value = obs[group]
            if value.ndim != 2 or value.dtype != torch.float32:
                raise ValueError(f"proprio group {group!r} must be float32 [B,D]")
            group_dims[group] = value.shape[-1]
            proprio_dim += value.shape[-1]
        if proprio_dim != self.proprio_dim:
            raise ValueError(f"packet-window actor requires {self.proprio_dim} proprio values")

        for name, value in (
            ("tokenizer_dim", tokenizer_dim),
            ("gru_hidden_dim", gru_hidden_dim),
            ("gru_num_layers", gru_num_layers),
            ("latent_hidden_dim", latent_hidden_dim),
            ("prop_feature_dim", prop_feature_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        distribution_values = copy.deepcopy(
            dict(
                distribution_cfg
                or {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                }
            )
        )
        class_name = distribution_values.pop("class_name", None)
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("distribution_cfg.class_name must be a non-empty string")
        distribution_class: type[Distribution] = resolve_callable(class_name)

        self.frame_tokenizer = _PacketTokenizer(
            near_range_m=frame_near_range_m,
            far_range_m=frame_far_range_m,
            message_period_s=frame_message_period_s,
            max_age_s=frame_max_age_s,
            frame_age_semantics=frame_age_semantics,
            hidden_channels=tokenizer_hidden_channels,
            token_dim=tokenizer_dim,
            pooled_spatial_size=tokenizer_pooled_spatial_size,
        )
        self.packet_gru = nn.GRU(
            input_size=tokenizer_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, latent_hidden_dim),
            nn.ELU(),
            nn.Linear(latent_hidden_dim, self.latent_dim),
        )
        self.obs_normalizer: nn.Module
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(proprio_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.prop_mlp = MLP(proprio_dim, prop_feature_dim, prop_hidden_dims, activation)
        self.distribution = distribution_class(output_dim, **distribution_values)
        self.mlp = MLP(
            prop_feature_dim + self.latent_dim,
            self.distribution.input_dim,
            fusion_hidden_dims,
            activation,
        )
        self.distribution.init_mlp_weights(self.mlp)

        self.packet_window_set = packet_window_set
        self.proprio_sets = proprio_groups
        self.proprio_group_dims = group_dims
        self.obs_groups = list(expected_groups)
        self.obs_set = obs_set
        self.history_packets = history_packets
        self.obs_normalization = obs_normalization
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.latent_hidden_dim = latent_hidden_dim
        self.prop_feature_dim = prop_feature_dim
        self.prop_hidden_dims = tuple(prop_hidden_dims)
        self.fusion_hidden_dims = tuple(fusion_hidden_dims)
        self.activation = activation
        self.distribution_config = {"class_name": class_name, **distribution_values}

    def _validate_runtime_obs(self, obs: TensorDict) -> None:
        if self.packet_window_set not in obs:
            raise KeyError(f"observation is missing {self.packet_window_set!r}")
        window = obs[self.packet_window_set]
        expected = (*obs.batch_size, self.history_packets, *_PacketTokenizer.frame_shape)
        if tuple(window.shape) != expected:
            raise ValueError(f"packet history shape must be {expected}, got {tuple(window.shape)}")

    def _encode_packet_history(self, obs: TensorDict) -> torch.Tensor:
        self._validate_runtime_obs(obs)
        tokens = self.frame_tokenizer(obs[self.packet_window_set])
        leading = tokens.shape[:-2]
        packets = tokens.shape[-2]
        flattened = tokens.reshape(-1, packets, tokens.shape[-1])
        output, _ = self.packet_gru(flattened)
        latent = self.latent_head(output[:, -1])
        return latent.reshape(*leading, self.latent_dim)

    def _encode_proprio(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([obs[group] for group in self.proprio_sets], dim=-1)
        return self.prop_mlp(self.obs_normalizer(proprio))

    def predict_latent(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_packet_history(obs)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if hidden_state is not None:
            raise ValueError("fixed packet-window actor has no persistent hidden state")
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent_a = self._encode_packet_history(obs)
        raw_action = self.mlp(torch.cat((self._encode_proprio(obs), latent_a), dim=-1))
        if stochastic_output:
            self.distribution.update(raw_action)
            return self.distribution.sample()
        return self.distribution.deterministic_output(raw_action)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: None = None) -> None:
        del dones
        if hidden_state is not None:
            raise ValueError("fixed packet-window actor has no persistent hidden state")

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

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
            "actor_inputs": {
                "ordered_groups": list(self.obs_groups),
                "packet_window_set": self.packet_window_set,
                "history_packets": self.history_packets,
                "history_span_s": (self.history_packets - 1)
                * self.frame_tokenizer.message_period_s,
                "uses_map": False,
                "uses_ground_truth_pose": False,
                "uses_future_frames": False,
            },
            "strict_frame_channels": list(_PacketTokenizer.channels),
            "frame_age_semantics": self.frame_tokenizer.frame_age_semantics,
            "tokenizer_dim": self.frame_tokenizer.token_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "latent_a_dim": self.latent_dim,
            "latent_b_dim": self.prop_feature_dim,
            "fusion_hidden_dims": list(self.fusion_hidden_dims),
            "distribution": copy.deepcopy(self.distribution_config),
        }
