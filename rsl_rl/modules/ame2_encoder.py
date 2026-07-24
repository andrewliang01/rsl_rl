# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AME2Encoder(nn.Module):
    """Minimal Phase-1 terrain encoder with proprioception-conditioned attention.

    The encoder intentionally contains only the Phase-1 data path:

    1. select one frame from the raw elevation history;
    2. spatially mean-center, scale, and clip the selected height map;
    3. extract a small grid of local CNN tokens;
    4. attend to those tokens with a single query derived from proprioception.

    Positional features, global terrain context, and multi-head attention are
    deliberately deferred to Phase 2. The public input/output contract is kept
    stable so those additions can remain internal to this module.
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
        point_feature_dim: int = 64,
        attention_dim: int = 64,
        output_dim: int = 64,
        height_scale: float = 0.6,
        height_clip: float = 3.0,
        num_heads: int = 1,
    ) -> None:
        """Initialize the minimal AME-2 encoder.

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
            point_feature_dim: Dimension of every local terrain token.
            attention_dim: Dimension of the single-head query and keys/values.
            output_dim: Dimension of the returned terrain embedding.
            height_scale: Scale applied after spatial mean centering.
            height_clip: Symmetric clipping limit after scaling.
            num_heads: Phase 1 supports exactly one attention head.
        """
        super().__init__()

        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got {history_length}.")
        if len(vision_spatial_size) != 2 or min(vision_spatial_size) <= 0:
            raise ValueError(f"vision_spatial_size must contain two positive values, got {vision_spatial_size}.")
        if len(token_spatial_size) != 2 or min(token_spatial_size) <= 0:
            raise ValueError(f"token_spatial_size must contain two positive values, got {token_spatial_size}.")
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
        if min(proprio_feature_dim, point_feature_dim, attention_dim, output_dim) <= 0:
            raise ValueError(
                "proprio_feature_dim, point_feature_dim, attention_dim, and output_dim must all be positive."
            )
        if height_scale <= 0.0:
            raise ValueError(f"height_scale must be positive, got {height_scale}.")
        if height_clip <= 0.0:
            raise ValueError(f"height_clip must be positive, got {height_clip}.")
        if num_heads != 1:
            raise ValueError(f"Phase-1 AME2Encoder supports exactly one attention head, got {num_heads}.")

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
        self.output_dim = int(output_dim)
        self.history_index = int(resolved_history_index)
        self.height_scale = float(height_scale)
        self.height_clip = float(height_clip)

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
        self.point_projection = nn.Sequential(
            nn.Linear(local_channels[-1], point_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.query_projection = nn.Linear(proprio_feature_dim, attention_dim)
        self.key_projection = nn.Linear(point_feature_dim, attention_dim)
        self.value_projection = nn.Linear(point_feature_dim, attention_dim)
        self.output_projection: nn.Module
        if attention_dim == output_dim:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(attention_dim, output_dim)

        self.attention_scale = 1.0 / math.sqrt(attention_dim)
        self.num_tokens = self.token_spatial_size[0] * self.token_spatial_size[1]

    def forward(self, height_history: torch.Tensor, proprio_features: torch.Tensor) -> torch.Tensor:
        """Encode a raw elevation history into one terrain embedding."""
        terrain_embedding, _ = self.forward_with_attention(height_history, proprio_features)
        return terrain_embedding

    def forward_with_attention(
        self, height_history: torch.Tensor, proprio_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the terrain embedding and single-head attention weights.

        The attention weights have shape ``[B, 1, N]`` and are exposed only for
        tests and diagnostics. They are not retained as module state.
        """
        self._validate_inputs(height_history, proprio_features)

        latest_height = height_history[:, self.history_index : self.history_index + 1]
        spatial_mean = latest_height.mean(dim=(-2, -1), keepdim=True)
        normalized_height = torch.clamp(
            (latest_height - spatial_mean) / self.height_scale,
            min=-self.height_clip,
            max=self.height_clip,
        )

        local_features = self.local_encoder(normalized_height)
        point_tokens = local_features.flatten(start_dim=2).transpose(1, 2)
        point_tokens = self.point_projection(point_tokens)

        query = self.query_projection(proprio_features).unsqueeze(1)
        keys = self.key_projection(point_tokens)
        values = self.value_projection(point_tokens)

        attention_logits = torch.bmm(query, keys.transpose(1, 2)) * self.attention_scale
        attention_weights = torch.softmax(attention_logits, dim=-1)
        weighted_local_feature = torch.bmm(attention_weights, values).squeeze(1)
        terrain_embedding = self.output_projection(weighted_local_feature)
        return terrain_embedding, attention_weights

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
