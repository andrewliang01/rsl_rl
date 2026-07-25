# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AME2Encoder(nn.Module):
    """Terrain encoder with metric xyz positional features.

    The public interface consumes the raw elevation history so the physical
    coordinate adapter is shared by eager and exported policies.  Spatial axes
    are interpreted as ``[x, y]`` (not image-style ``[y, x]``), and the input
    downward clearance is converted to upward metric height before encoding.

    1. select one frame from the raw elevation history;
    2. construct the metric ``(x, y, z)`` elevation representation;
    3. extract aligned local CNN and positional features;
    4. aggregate global terrain context with max pooling;
    5. attend to fused point features with a proprioceptive/global query.

    The global and attended features are concatenated into the final map
    embedding while the public input/output contract remains unchanged.
    """

    def __init__(
        self,
        history_length: int = 5,
        history_index: int = -1,
        vision_spatial_size: tuple[int, int] = (28, 20),
        token_spatial_size: tuple[int, int] = (7, 5),
        local_channels: tuple[int, ...] | list[int] = (8, 16),
        kernel_sizes: tuple[int, ...] | list[int] = (3, 3),
        strides: tuple[int, ...] | list[int] = (2, 2),
        proprio_feature_dim: int = 64,
        map_extent: tuple[float, float] = (1.35, 0.95),
        position_feature_dim: int = 16,
        point_feature_dim: int = 64,
        global_feature_dim: int = 32,
        attention_dim: int = 32,
        output_dim: int = 64,
        height_scale: float = 0.6,
        height_clip: float = 3.0,
        num_heads: int = 4,
    ) -> None:
        """Initialize the AME-2 encoder.

        Args:
            history_length: Number of frames in ``height_history``.
            history_index: Frame selected from the elevation history. ``-1`` is
                the latest frame because the observation history is time ordered.
            vision_spatial_size: Input height-map size ``(H, W)``.
            token_spatial_size: Expected local-token grid size.
            local_channels: Output channels of the local CNN layers.
            kernel_sizes: Kernel size for each local CNN layer.
            strides: Stride for each local CNN layer.
            proprio_feature_dim: Dimension of the proprioception embedding.
            map_extent: Metric map extent ``(x_extent, y_extent)``.
            position_feature_dim: Dimension of every pooled xyz feature.
            point_feature_dim: Dimension of every local terrain token.
            global_feature_dim: Dimension of the pooled terrain context.
            attention_dim: Dimension of the query and keys/values.
            output_dim: Dimension of the returned terrain embedding.
            height_scale: Scale applied after spatial mean centering.
            height_clip: Symmetric clipping limit after scaling.
            num_heads: Number of attention heads.
        """
        super().__init__()

        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}.")
        if len(vision_spatial_size) != 2 or min(vision_spatial_size) <= 0:
            raise ValueError(f"vision_spatial_size must contain two positive values, got {vision_spatial_size}.")
        if len(token_spatial_size) != 2 or min(token_spatial_size) <= 0:
            raise ValueError(f"token_spatial_size must contain two positive values, got {token_spatial_size}.")
        if any(
            input_size % token_size != 0
            for input_size, token_size in zip(vision_spatial_size, token_spatial_size)
        ):
            raise ValueError(
                "vision_spatial_size must be evenly divisible by token_spatial_size for xyz pooling, got "
                f"{vision_spatial_size} and {token_spatial_size}."
            )
        if not local_channels:
            raise ValueError("local_channels must contain at least one channel size.")
        if not (len(local_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                "local_channels, kernel_sizes, and strides must have equal lengths, got "
                f"{len(local_channels)}, {len(kernel_sizes)}, and {len(strides)}."
            )
        if any(channel <= 0 for channel in local_channels):
            raise ValueError(f"All local_channels must be positive, got {local_channels}.")
        if any(kernel <= 0 for kernel in kernel_sizes):
            raise ValueError(f"All kernel_sizes must be positive, got {kernel_sizes}.")
        if any(stride <= 0 for stride in strides):
            raise ValueError(f"All strides must be positive, got {strides}.")
        if len(map_extent) != 2 or min(map_extent) <= 0.0:
            raise ValueError(f"map_extent must contain two positive values, got {map_extent}.")
        if min(
            proprio_feature_dim,
            position_feature_dim,
            point_feature_dim,
            global_feature_dim,
            attention_dim,
            output_dim,
        ) <= 0:
            raise ValueError(
                "proprio_feature_dim, position_feature_dim, point_feature_dim, "
                "global_feature_dim, attention_dim, and output_dim must all be positive."
            )
        if height_scale <= 0.0:
            raise ValueError(f"height_scale must be positive, got {height_scale}.")
        if height_clip <= 0.0:
            raise ValueError(f"height_clip must be positive, got {height_clip}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim must be divisible by num_heads, got {attention_dim} and {num_heads}."
            )
        if global_feature_dim + attention_dim != output_dim:
            raise ValueError(
                "global_feature_dim + attention_dim must equal output_dim, got "
                f"{global_feature_dim} + {attention_dim} != {output_dim}."
            )

        resolved_history_index = history_index
        if resolved_history_index < 0:
            resolved_history_index += history_length
        if not 0 <= resolved_history_index < history_length:
            raise ValueError(
                f"history_index {history_index} is invalid for history length {history_length}."
            )

        self.history_length = int(history_length)
        self.vision_spatial_size = tuple(int(value) for value in vision_spatial_size)
        self.token_spatial_size = tuple(int(value) for value in token_spatial_size)
        self.proprio_feature_dim = int(proprio_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.output_dim = int(output_dim)
        self.history_index = int(resolved_history_index)
        self.height_scale = float(height_scale)
        self.height_clip = float(height_clip)
        self.map_extent = (float(map_extent[0]), float(map_extent[1]))

        x_coordinates = torch.linspace(
            -0.5 * self.map_extent[0],
            0.5 * self.map_extent[0],
            self.vision_spatial_size[0],
        )
        y_coordinates = torch.linspace(
            -0.5 * self.map_extent[1],
            0.5 * self.map_extent[1],
            self.vision_spatial_size[1],
        )
        x_grid, y_grid = torch.meshgrid(x_coordinates, y_coordinates, indexing="ij")
        self.register_buffer("xy_grid", torch.stack((x_grid, y_grid), dim=-1), persistent=False)

        local_layers: list[nn.Module] = []
        in_channels = 1
        spatial_size = self.vision_spatial_size
        for out_channels, kernel_size, stride in zip(local_channels, kernel_sizes, strides):
            padding = kernel_size // 2
            local_layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            spatial_size = self._conv_output_spatial_size(spatial_size, kernel_size, stride, padding)
            in_channels = out_channels

        if spatial_size != self.token_spatial_size:
            raise ValueError(
                "Local CNN output size does not match token_spatial_size: "
                f"computed {spatial_size}, configured {self.token_spatial_size}."
            )

        self.local_encoder = nn.Sequential(*local_layers)
        xyz_pool_size = (
            self.vision_spatial_size[0] // self.token_spatial_size[0],
            self.vision_spatial_size[1] // self.token_spatial_size[1],
        )
        self.xyz_pool = nn.AvgPool2d(kernel_size=xyz_pool_size, stride=xyz_pool_size)
        self.positional_encoder = nn.Sequential(
            nn.Linear(3, position_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.point_fusion = nn.Sequential(
            nn.Linear(local_channels[-1] + position_feature_dim, point_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, global_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.query_projection = nn.Linear(proprio_feature_dim + global_feature_dim, attention_dim)
        self.key_projection = nn.Linear(point_feature_dim, attention_dim)
        self.value_projection = nn.Linear(point_feature_dim, attention_dim)
        self.map_projection = nn.Identity()

        self.attention_scale = 1.0 / math.sqrt(self.head_dim)
        self.num_tokens = self.token_spatial_size[0] * self.token_spatial_size[1]

    def forward(self, height_history: torch.Tensor, proprio_features: torch.Tensor) -> torch.Tensor:
        """Encode a raw elevation history into one terrain embedding."""
        terrain_embedding, _ = self.forward_with_attention(height_history, proprio_features)
        return terrain_embedding

    def forward_with_attention(
        self, height_history: torch.Tensor, proprio_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the terrain embedding and multi-head attention weights.

        The attention weights have shape ``[B, heads, 1, N]`` and are exposed only for
        tests and diagnostics. They are not retained as module state.
        """
        terrain_embedding, _, _, _, _, _, _, attention_weights = self.forward_with_intermediates(
            height_history, proprio_features
        )
        return terrain_embedding, attention_weights

    def forward_with_intermediates(
        self, height_history: torch.Tensor, proprio_features: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the map embedding and the main Phase-2 intermediate tensors."""
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._validate_inputs(height_history, proprio_features)

        elevation_xyz = self.build_elevation_xyz(height_history)
        local_features, position_features = self.extract_local_position_features(elevation_xyz)
        point_local_features = self.point_fusion(
            torch.cat((local_features, position_features), dim=-1)
        )
        global_point_features = self.global_encoder(point_local_features)
        global_feature = torch.max(global_point_features, dim=1).values

        query_input = torch.cat((proprio_features, global_feature), dim=-1)
        query = self.query_projection(query_input).reshape(-1, self.num_heads, 1, self.head_dim)
        keys = self.key_projection(point_local_features)
        keys = keys.reshape(-1, self.num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.value_projection(point_local_features)
        values = values.reshape(-1, self.num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_logits = torch.matmul(query, keys.transpose(-2, -1)) * self.attention_scale
        attention_weights = torch.softmax(attention_logits, dim=-1)
        weighted_local_feature = torch.matmul(attention_weights, values)
        weighted_local_feature = weighted_local_feature.squeeze(2).reshape(-1, self.attention_dim)
        terrain_embedding = self.map_projection(
            torch.cat((global_feature, weighted_local_feature), dim=-1)
        )
        return (
            terrain_embedding,
            point_local_features,
            global_feature,
            query,
            keys,
            values,
            weighted_local_feature,
            attention_weights,
        )

    def build_elevation_xyz(self, height_history: torch.Tensor) -> torch.Tensor:
        """Convert raw downward clearance into an upward-z metric xyz map."""
        latest_height = height_history[:, self.history_index]
        z_metric = -latest_height
        xy_grid = self.xy_grid.unsqueeze(0).expand(height_history.size(0), -1, -1, -1)
        return torch.cat((xy_grid, z_metric.unsqueeze(-1)), dim=-1)

    def extract_local_position_features(
        self, elevation_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned CNN-local and pooled xyz positional features."""
        z_metric = elevation_xyz[..., 2].unsqueeze(1)
        spatial_mean = z_metric.mean(dim=(-2, -1), keepdim=True)
        normalized_height = torch.clamp(
            (z_metric - spatial_mean) / self.height_scale,
            min=-self.height_clip,
            max=self.height_clip,
        )

        local_features = self.local_encoder(normalized_height)
        local_features = local_features.flatten(start_dim=2).transpose(1, 2)

        pooled_xyz = self.pool_elevation_xyz(elevation_xyz)
        position_features = self.positional_encoder(pooled_xyz)
        return local_features, position_features

    def pool_elevation_xyz(self, elevation_xyz: torch.Tensor) -> torch.Tensor:
        """Pool metric xyz to the local-token grid and flatten in x-major order."""
        pooled_xyz = self.xyz_pool(elevation_xyz.permute(0, 3, 1, 2))
        return pooled_xyz.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

    @torch.jit.unused
    def _validate_inputs(self, height_history: torch.Tensor, proprio_features: torch.Tensor) -> None:
        expected_height_shape = (
            self.history_length,
            self.vision_spatial_size[0],
            self.vision_spatial_size[1],
        )
        if height_history.ndim != 4 or tuple(height_history.shape[1:]) != expected_height_shape:
            raise ValueError(
                "height_history must have shape [B, T, H, W] with "
                f"[T, H, W]={expected_height_shape}, got {tuple(height_history.shape)}."
            )
        if proprio_features.ndim != 2 or proprio_features.shape[1] != self.proprio_feature_dim:
            raise ValueError(
                "proprio_features must have shape [B, D] with "
                f"D={self.proprio_feature_dim}, got {tuple(proprio_features.shape)}."
            )
        if height_history.shape[0] != proprio_features.shape[0]:
            raise ValueError(
                "height_history and proprio_features must have the same batch size, got "
                f"{height_history.shape[0]} and {proprio_features.shape[0]}."
            )
        if height_history.device != proprio_features.device:
            raise ValueError(
                "height_history and proprio_features must be on the same device, got "
                f"{height_history.device} and {proprio_features.device}."
            )
        if height_history.dtype != proprio_features.dtype:
            raise ValueError(
                "height_history and proprio_features must have the same dtype, got "
                f"{height_history.dtype} and {proprio_features.dtype}."
            )
        if not height_history.is_floating_point() or not proprio_features.is_floating_point():
            raise ValueError("height_history and proprio_features must be floating-point tensors.")

    @staticmethod
    def _conv_output_spatial_size(
        spatial_size: tuple[int, int], kernel_size: int, stride: int, padding: int
    ) -> tuple[int, int]:
        height = (spatial_size[0] + 2 * padding - kernel_size) // stride + 1
        width = (spatial_size[1] + 2 * padding - kernel_size) // stride + 1
        return height, width
