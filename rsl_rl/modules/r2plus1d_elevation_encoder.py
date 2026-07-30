# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factorized spatiotemporal encoder for elevation-map history."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def _positive_int_tuple(
    values: Sequence[int],
    *,
    name: str,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    result = tuple(values)
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, got {len(result)}."
        )
    if not result or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in result
    ):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def _factorized_intermediate_channels(
    in_channels: int,
    out_channels: int,
    *,
    temporal_kernel_size: int,
    spatial_kernel_size: int,
) -> int:
    """Match the parameter count of one dense 3-D convolution as closely as possible."""
    numerator = (
        temporal_kernel_size
        * spatial_kernel_size
        * spatial_kernel_size
        * in_channels
        * out_channels
    )
    denominator = (
        spatial_kernel_size * spatial_kernel_size * in_channels
        + temporal_kernel_size * out_channels
    )
    return max(1, int(math.floor(numerator / denominator)))


class R2Plus1DBlock(nn.Module):
    """One spatial-then-temporal factorization of a 3-D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temporal_kernel_size: int = 3,
        spatial_kernel_size: int = 3,
        spatial_stride: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (in_channels, "in_channels"),
            (out_channels, "out_channels"),
            (temporal_kernel_size, "temporal_kernel_size"),
            (spatial_kernel_size, "spatial_kernel_size"),
            (spatial_stride, "spatial_stride"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if temporal_kernel_size % 2 == 0 or spatial_kernel_size % 2 == 0:
            raise ValueError(
                "R(2+1)D kernel sizes must be odd so time and space remain aligned."
            )

        intermediate_channels = _factorized_intermediate_channels(
            in_channels,
            out_channels,
            temporal_kernel_size=temporal_kernel_size,
            spatial_kernel_size=spatial_kernel_size,
        )
        self.spatial = nn.Conv3d(
            in_channels,
            intermediate_channels,
            kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, spatial_kernel_size // 2, spatial_kernel_size // 2),
            bias=False,
        )
        self.spatial_norm = nn.BatchNorm3d(intermediate_channels)
        self.spatial_activation = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(
            intermediate_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            stride=1,
            padding=(temporal_kernel_size // 2, 0, 0),
            bias=False,
        )
        self.temporal_norm = nn.BatchNorm3d(out_channels)
        self.temporal_activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.spatial_activation(self.spatial_norm(self.spatial(value)))
        return self.temporal_activation(
            self.temporal_norm(self.temporal(value))
        )


class R2Plus1DElevationEncoder(nn.Module):
    """Encode ``[B,T,H,W]`` elevation history with factorized 3-D convolutions.

    Time is kept explicit through every block. The final temporal mean is
    applied only after the temporal convolutions have mixed ordered frames.
    Spatial down-sampling and the final linear projection intentionally mirror
    the existing 2-D CNN baseline interface.
    """

    def __init__(
        self,
        *,
        history_length: int,
        hidden_dims: Sequence[int] = (16, 24, 44),
        spatial_kernel_sizes: Sequence[int] = (3, 3, 3),
        temporal_kernel_sizes: Sequence[int] = (3, 3, 3),
        spatial_strides: Sequence[int] = (2, 2, 2),
        out_dim: int = 64,
        vision_spatial_size: tuple[int, int] = (25, 17),
    ) -> None:
        super().__init__()
        if (
            not isinstance(history_length, int)
            or isinstance(history_length, bool)
            or history_length <= 0
        ):
            raise ValueError("history_length must be a positive integer.")
        if (
            not isinstance(out_dim, int)
            or isinstance(out_dim, bool)
            or out_dim <= 0
        ):
            raise ValueError("out_dim must be a positive integer.")
        spatial_size = _positive_int_tuple(
            vision_spatial_size,
            name="vision_spatial_size",
            expected_length=2,
        )
        channels = _positive_int_tuple(hidden_dims, name="hidden_dims")
        spatial_kernels = _positive_int_tuple(
            spatial_kernel_sizes,
            name="spatial_kernel_sizes",
            expected_length=len(channels),
        )
        temporal_kernels = _positive_int_tuple(
            temporal_kernel_sizes,
            name="temporal_kernel_sizes",
            expected_length=len(channels),
        )
        strides = _positive_int_tuple(
            spatial_strides,
            name="spatial_strides",
            expected_length=len(channels),
        )
        if any(value % 2 == 0 for value in (*spatial_kernels, *temporal_kernels)):
            raise ValueError("R(2+1)D kernel sizes must all be odd.")

        self.history_length = history_length
        self.vision_spatial_size = spatial_size
        self.image_height = spatial_size[0]
        self.image_width = spatial_size[1]
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels, spatial_kernel, temporal_kernel, stride in zip(
            channels,
            spatial_kernels,
            temporal_kernels,
            strides,
        ):
            blocks.append(
                R2Plus1DBlock(
                    in_channels,
                    out_channels,
                    temporal_kernel_size=temporal_kernel,
                    spatial_kernel_size=spatial_kernel,
                    spatial_stride=stride,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        encoded_height, encoded_width = spatial_size
        for stride in strides:
            encoded_height = math.ceil(encoded_height / stride)
            encoded_width = math.ceil(encoded_width / stride)
        flattened_dim = channels[-1] * encoded_height * encoded_width
        self.projection = nn.Linear(flattened_dim, out_dim)

    def forward(self, elevation_history: torch.Tensor) -> torch.Tensor:
        if (
            elevation_history.dim() != 4
            or elevation_history.size(1) != self.history_length
            or elevation_history.size(2) != self.image_height
            or elevation_history.size(3) != self.image_width
        ):
            raise ValueError(
                "R(2+1)D elevation history must have shape [B,T,H,W] "
                "matching the configured history and spatial dimensions."
            )
        if not elevation_history.is_floating_point():
            raise TypeError("R(2+1)D elevation history must be floating point.")
        features = self.blocks(elevation_history.unsqueeze(1))
        features = features.mean(dim=2)
        return self.projection(features.flatten(start_dim=1))
