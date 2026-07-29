# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(num_channels: int, maximum: int = 8) -> int:
    """Return the largest small GroupNorm group count that divides the channels."""
    for num_groups in range(min(maximum, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class CircularAzimuthConv2d(nn.Module):
    """2-D convolution with circular padding only along the azimuth (width) axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if any(size <= 0 or size % 2 == 0 for size in kernel_size):
            raise ValueError(f"kernel_size must contain positive odd values, got {kernel_size}.")
        if any(value <= 0 for value in stride):
            raise ValueError(f"stride must contain positive values, got {stride}.")

        self.vertical_padding = kernel_size[0] // 2
        self.azimuth_padding = kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.pad(
            inputs,
            (self.azimuth_padding, self.azimuth_padding, 0, 0),
            mode="circular",
        )
        inputs = F.pad(
            inputs,
            (0, 0, self.vertical_padding, self.vertical_padding),
            mode="constant",
            value=0.0,
        )
        return self.conv(inputs)


class _SphericalFrameBlock(nn.Module):
    """Circular spatial convolution followed by deterministic normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = CircularAzimuthConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class RayTimeAttentionEncoder(nn.Module):
    """Encode a masked spherical range-image history into a terrain feature.

    Input channel 0 contains metric range in metres and uses zero for unknown
    cells. Input channel 1 is a binary hit mask. The encoder sanitizes those
    channels, applies logarithmic range normalization, preserves all temporal
    and spatial tokens, and uses proprioception-conditioned queries to select
    action-relevant ray-time features.
    """

    def __init__(
        self,
        history_length: int = 5,
        vision_spatial_size: tuple[int, int] = (16, 96),
        proprio_feature_dim: int = 64,
        output_dim: int = 64,
        spatial_channels: tuple[int, ...] | list[int] = (24, 32, 64),
        token_dim: int = 64,
        num_heads: int = 4,
        num_queries: int = 4,
        min_range: float = 0.1,
        max_range: float = 6.0,
        vertical_fov_degrees: tuple[float, float] = (-52.0, 7.0),
        use_query_attention: bool = True,
    ) -> None:
        super().__init__()

        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}.")
        if len(vision_spatial_size) != 2 or min(vision_spatial_size) <= 0:
            raise ValueError(
                f"vision_spatial_size must contain two positive values, got {vision_spatial_size}."
            )
        if len(spatial_channels) != 3 or min(spatial_channels) <= 0:
            raise ValueError(
                "spatial_channels must contain exactly three positive channel sizes, "
                f"got {spatial_channels}."
            )
        if min(proprio_feature_dim, output_dim, token_dim, num_heads, num_queries) <= 0:
            raise ValueError(
                "proprio_feature_dim, output_dim, token_dim, num_heads, and num_queries "
                "must all be positive."
            )
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim must be divisible by num_heads, got {token_dim} and {num_heads}."
            )
        if not 0.0 < min_range < max_range:
            raise ValueError(
                f"Expected 0 < min_range < max_range, got {min_range} and {max_range}."
            )
        if (
            len(vertical_fov_degrees) != 2
            or vertical_fov_degrees[1] <= vertical_fov_degrees[0]
        ):
            raise ValueError(
                "vertical_fov_degrees must contain increasing lower/upper angles, "
                f"got {vertical_fov_degrees}."
            )

        self.history_length = int(history_length)
        self.vision_spatial_size = tuple(int(value) for value in vision_spatial_size)
        self.proprio_feature_dim = int(proprio_feature_dim)
        self.output_dim = int(output_dim)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.num_queries = int(num_queries)
        self.head_dim = self.token_dim // self.num_heads
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.use_query_attention = bool(use_query_attention)
        self.attention_scale = 1.0 / math.sqrt(self.head_dim)

        channel_1, channel_2, channel_3 = (int(value) for value in spatial_channels)
        self.frame_encoder = nn.Sequential(
            _SphericalFrameBlock(2, channel_1, (3, 5), (1, 2)),
            _SphericalFrameBlock(channel_1, channel_2, (3, 3), (2, 2)),
            _SphericalFrameBlock(channel_2, channel_3, (3, 3), (2, 2)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, *self.vision_spatial_size)
            encoded_dummy = self.frame_encoder(dummy)
        self.token_spatial_size = tuple(int(value) for value in encoded_dummy.shape[-2:])
        self.num_spatial_tokens = self.token_spatial_size[0] * self.token_spatial_size[1]
        self.num_tokens = self.history_length * self.num_spatial_tokens

        self.spatial_projection = nn.Conv2d(channel_3, self.token_dim, kernel_size=1, bias=False)
        self.spatial_norm = nn.GroupNorm(_group_count(self.token_dim), self.token_dim)

        self.temporal_depthwise = nn.Conv1d(
            self.token_dim,
            self.token_dim,
            kernel_size=3,
            padding=1,
            groups=self.token_dim,
            bias=False,
        )
        self.temporal_pointwise = nn.Conv1d(
            self.token_dim,
            self.token_dim,
            kernel_size=1,
            bias=False,
        )
        self.temporal_norm = nn.LayerNorm(self.token_dim)

        self.query_projection = nn.Linear(
            self.proprio_feature_dim,
            self.num_queries * self.token_dim,
        )
        self.query_bias = nn.Parameter(torch.zeros(1, self.num_queries, self.token_dim))
        self.key_projection = nn.Linear(self.token_dim, self.token_dim)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim)
        self.attended_projection = nn.Linear(
            self.num_queries * self.token_dim,
            self.output_dim,
        )
        # The global-only ablation keeps the same module graph and replaces the
        # active query/QKV path (41,664 parameters at the default 64-D setup)
        # with a closely matched 64 -> 320 -> 64 adapter (41,344 parameters).
        # Both variants therefore have identical total parameter counts while
        # their active trainable capacities differ by less than one percent.
        self.global_ablation_adapter = nn.Sequential(
            nn.Linear(self.token_dim, 320),
            nn.SiLU(inplace=True),
            nn.Linear(320, self.output_dim),
        )
        self.global_projection = nn.Linear(self.token_dim, self.output_dim)
        self.output_projection = nn.Linear(2 * self.output_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

        spherical_encoding = self._build_spherical_encoding(
            self.token_spatial_size,
            self.token_dim,
            vertical_fov_degrees,
        )
        time_encoding = self._build_time_encoding(self.history_length, self.token_dim)
        self.register_buffer(
            "spherical_position_encoding",
            spherical_encoding,
            persistent=False,
        )
        self.register_buffer("time_encoding", time_encoding, persistent=False)

    def forward(
        self,
        ray_history: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return a fixed-size terrain embedding."""
        terrain_embedding, _, _ = self.forward_with_attention(
            ray_history,
            proprio_features,
        )
        return terrain_embedding

    def forward_with_attention(
        self,
        ray_history: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embedding, cross-attention weights, and valid-token mask."""
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._validate_inputs(ray_history, proprio_features)

        # Rollout storage may keep the observation in fp16. Run the compact
        # policy encoder in fp32 to match the remaining actor modules.
        ray_history = ray_history.float()
        proprio_features = proprio_features.float()
        frame_inputs, hit_mask = self._prepare_inputs(ray_history)

        batch_size = frame_inputs.shape[0]
        frame_features = frame_inputs.flatten(0, 1)
        frame_features = self.frame_encoder(frame_features)
        frame_features = F.silu(self.spatial_norm(self.spatial_projection(frame_features)))
        frame_features = frame_features.flatten(start_dim=2).transpose(1, 2)
        tokens = frame_features.reshape(
            batch_size,
            self.history_length,
            self.num_spatial_tokens,
            self.token_dim,
        )
        tokens = tokens + self.spherical_position_encoding.view(
            1, 1, self.num_spatial_tokens, self.token_dim
        )
        tokens = tokens + self.time_encoding.view(
            1, self.history_length, 1, self.token_dim
        )

        temporal_tokens = tokens.permute(0, 2, 3, 1).reshape(
            batch_size * self.num_spatial_tokens,
            self.token_dim,
            self.history_length,
        )
        temporal_update = self.temporal_pointwise(
            self.temporal_depthwise(temporal_tokens)
        )
        temporal_update = F.silu(temporal_update)
        temporal_update = temporal_update.reshape(
            batch_size,
            self.num_spatial_tokens,
            self.token_dim,
            self.history_length,
        ).permute(0, 3, 1, 2)
        tokens = self.temporal_norm(tokens + temporal_update)
        tokens = tokens.flatten(start_dim=1, end_dim=2)

        token_valid = F.adaptive_max_pool2d(
            hit_mask.flatten(0, 1),
            self.token_spatial_size,
        )
        token_valid = token_valid.reshape(
            batch_size,
            self.history_length,
            self.num_spatial_tokens,
        ).flatten(start_dim=1)
        token_valid = token_valid > 0.5

        valid_float = token_valid.to(tokens.dtype).unsqueeze(-1)
        valid_count = valid_float.sum(dim=1).clamp_min(1.0)
        global_token = (tokens * valid_float).sum(dim=1) / valid_count
        global_feature = self.global_projection(global_token)

        if self.use_query_attention:
            attended_feature, attention_weights = self._query_attention(
                tokens,
                token_valid,
                proprio_features,
            )
        else:
            attended_feature = self.global_ablation_adapter(global_token)
            attention_weights = torch.zeros(
                batch_size,
                self.num_heads,
                self.num_queries,
                self.num_tokens,
                device=tokens.device,
                dtype=tokens.dtype,
            )

        terrain_embedding = self.output_norm(
            self.output_projection(torch.cat((attended_feature, global_feature), dim=-1))
        )
        return terrain_embedding, attention_weights, token_valid

    def _prepare_inputs(
        self,
        ray_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric_range = ray_history[:, :, 0]
        requested_hit = ray_history[:, :, 1] > 0.5
        valid_hit = (
            requested_hit
            & torch.isfinite(metric_range)
            & (metric_range > 0.0)
        )

        safe_range = torch.where(
            valid_hit,
            torch.clamp(metric_range, self.min_range, self.max_range),
            torch.full_like(metric_range, self.min_range),
        )
        log_denominator = math.log(self.max_range / self.min_range)
        normalized_range = torch.log(safe_range / self.min_range) / log_denominator
        normalized_range = normalized_range * valid_hit.to(normalized_range.dtype)
        hit_float = valid_hit.to(normalized_range.dtype)
        return torch.stack((normalized_range, hit_float), dim=2), hit_float

    def _query_attention(
        self,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = tokens.shape[0]
        query = self.query_projection(proprio_features).reshape(
            batch_size,
            self.num_queries,
            self.token_dim,
        )
        query = query + self.query_bias
        query = query.reshape(
            batch_size,
            self.num_queries,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        keys = self.key_projection(tokens).reshape(
            batch_size,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        values = self.value_projection(tokens).reshape(
            batch_size,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        logits = torch.matmul(query, keys.transpose(-2, -1)) * self.attention_scale
        # A fully unknown scan must remain finite. In that case the fixed
        # position/time tokens provide a deterministic, non-geometric fallback.
        safe_valid = torch.where(
            token_valid.any(dim=1, keepdim=True),
            token_valid,
            torch.ones_like(token_valid),
        )
        logits = logits.masked_fill(
            ~safe_valid[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        attention_weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention_weights, values)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_queries * self.token_dim,
        )
        return self.attended_projection(attended), attention_weights

    @staticmethod
    def _build_spherical_encoding(
        spatial_size: tuple[int, int],
        feature_dim: int,
        vertical_fov_degrees: tuple[float, float],
    ) -> torch.Tensor:
        if feature_dim % 4 != 0:
            raise ValueError(
                "token_dim must be divisible by four for fixed spherical encoding, "
                f"got {feature_dim}."
            )
        height, width = spatial_size
        elevation = torch.linspace(
            math.radians(vertical_fov_degrees[0]),
            math.radians(vertical_fov_degrees[1]),
            height,
        )
        azimuth = -math.pi + (torch.arange(width, dtype=torch.float32) + 0.5) * (
            2.0 * math.pi / width
        )
        elevation_grid, azimuth_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
        frequencies = torch.arange(
            1,
            feature_dim // 4 + 1,
            dtype=torch.float32,
        )
        encoding = torch.cat(
            (
                torch.sin(azimuth_grid[..., None] * frequencies),
                torch.cos(azimuth_grid[..., None] * frequencies),
                torch.sin(elevation_grid[..., None] * frequencies),
                torch.cos(elevation_grid[..., None] * frequencies),
            ),
            dim=-1,
        )
        return encoding.flatten(start_dim=0, end_dim=1)

    @staticmethod
    def _build_time_encoding(history_length: int, feature_dim: int) -> torch.Tensor:
        if feature_dim % 2 != 0:
            raise ValueError(
                f"token_dim must be even for fixed time encoding, got {feature_dim}."
            )
        history_position = (
            torch.zeros(1)
            if history_length == 1
            else torch.linspace(-1.0, 0.0, history_length)
        )
        frequencies = torch.arange(1, feature_dim // 2 + 1, dtype=torch.float32)
        phase = history_position[:, None] * math.pi * frequencies[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    @torch.jit.unused
    def _validate_inputs(
        self,
        ray_history: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> None:
        expected_shape = (
            self.history_length,
            2,
            self.vision_spatial_size[0],
            self.vision_spatial_size[1],
        )
        if ray_history.ndim != 5 or tuple(ray_history.shape[1:]) != expected_shape:
            raise ValueError(
                "ray_history must have shape [B, T, 2, H, W] with "
                f"[T, 2, H, W]={expected_shape}, got {tuple(ray_history.shape)}."
            )
        if (
            proprio_features.ndim != 2
            or proprio_features.shape[1] != self.proprio_feature_dim
        ):
            raise ValueError(
                "proprio_features must have shape [B, D] with "
                f"D={self.proprio_feature_dim}, got {tuple(proprio_features.shape)}."
            )
        if ray_history.shape[0] != proprio_features.shape[0]:
            raise ValueError(
                "ray_history and proprio_features must have the same batch size, got "
                f"{ray_history.shape[0]} and {proprio_features.shape[0]}."
            )
        if ray_history.device != proprio_features.device:
            raise ValueError(
                "ray_history and proprio_features must be on the same device, got "
                f"{ray_history.device} and {proprio_features.device}."
            )
        if not ray_history.is_floating_point() or not proprio_features.is_floating_point():
            raise ValueError("ray_history and proprio_features must be floating-point tensors.")
