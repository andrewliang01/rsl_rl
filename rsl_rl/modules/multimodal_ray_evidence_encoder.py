# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-testable multimodal ray-evidence encoder.

This module is intentionally independent from the existing Ray-Time encoder and
is not wired into any training configuration.  It defines an explicit deployable
input contract for LiDAR and perspective depth histories:

``[range_m, return_valid, ray_observed]``.

An observed ray without a return is valid evidence represented by
``[0, 0, 1]``.  An unobserved ray is ``[0, 0, 0]``.  Metric range is non-zero
only for valid returns.  LiDAR uses circular azimuth padding while perspective
depth uses ordinary zero padding.

The raster-coordinate encodings in this scaffold deliberately avoid assumed
sensor fields of view.  LiDAR tokens receive periodic azimuth ``sin/cos`` plus
a normalized vertical-row coordinate; perspective-depth tokens receive
normalized image ``x/y`` coordinates.  A deployable sensor integration should
replace or augment these raster coordinates with calibrated per-ray directions.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ray_time_attention_encoder import CircularAzimuthConv2d


_SUPPORTED_MODES = (
    "reliability",
    "concat",
    "lidar_only",
    "depth_only",
    "no_reliability",
)


def build_lidar_raster_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return auditable LiDAR raster coordinates ``[H, W, 3]``.

    The channels are ``[sin(azimuth), cos(azimuth), normalized_vertical]``.
    Azimuth samples are periodic token-bin centers.  The vertical coordinate is
    a normalized raster index, not a claim about the sensor's physical FOV.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"LiDAR raster dimensions must be positive, got {(height, width)}."
        )
    azimuth = (
        2.0
        * math.pi
        * (torch.arange(width, device=device, dtype=dtype) + 0.5)
        / float(width)
    )
    if height == 1:
        vertical = torch.zeros(1, device=device, dtype=dtype)
    else:
        vertical = (
            2.0
            * (torch.arange(height, device=device, dtype=dtype) + 0.5)
            / float(height)
            - 1.0
        )
    sin_azimuth = torch.sin(azimuth)[None, :].expand(height, -1)
    cos_azimuth = torch.cos(azimuth)[None, :].expand(height, -1)
    vertical = vertical[:, None].expand(-1, width)
    return torch.stack((sin_azimuth, cos_azimuth, vertical), dim=-1)


def build_depth_raster_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized perspective-image token centers ``[H, W, 2]``."""
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Depth raster dimensions must be positive, got {(height, width)}."
        )
    x = (
        2.0
        * (torch.arange(width, device=device, dtype=dtype) + 0.5)
        / float(width)
        - 1.0
    )
    y = (
        2.0
        * (torch.arange(height, device=device, dtype=dtype) + 0.5)
        / float(height)
        - 1.0
    )
    return torch.stack(
        (
            x[None, :].expand(height, -1),
            y[:, None].expand(-1, width),
        ),
        dim=-1,
    )


def _pool_stem_receptive_fields(
    inputs: torch.Tensor,
    *,
    sensor: str,
) -> torch.Tensor:
    """Pool with the two stems' exact kernel, stride, and boundary topology.

    This fixed average pooling is not intended to reproduce learned convolution
    weights.  It aligns evidence fractions and raster geometry with every ray
    that can enter the corresponding convolution token's local receptive field.
    """
    pooled = inputs
    for stride in ((1, 2), (2, 2)):
        if sensor == "lidar":
            pooled = F.pad(pooled, (1, 1, 0, 0), mode="circular")
            pooled = F.pad(pooled, (0, 0, 1, 1), mode="constant", value=0.0)
        elif sensor == "depth":
            pooled = F.pad(pooled, (1, 1, 1, 1), mode="constant", value=0.0)
        else:
            raise ValueError(f"Unsupported sensor kind '{sensor}'.")
        pooled = F.avg_pool2d(
            pooled,
            kernel_size=3,
            stride=stride,
            padding=0,
            count_include_pad=True,
        )
    return pooled


def _group_count(num_channels: int, maximum: int = 8) -> int:
    """Return a GroupNorm count with at least two channels in every group."""
    if num_channels < 2:
        raise ValueError(
            "Group-normalized ray stems require at least two channels, got "
            f"{num_channels}."
        )
    maximum_groups = min(maximum, num_channels // 2)
    for num_groups in range(maximum_groups, 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    raise AssertionError("At least one GroupNorm group must divide num_channels.")


def _masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Return a finite softmax with exact zeros outside ``mask``.

    Unlike ``masked_fill(-inf)`` followed by ``softmax``, this function also
    supports rows for which every entry is masked.  Such rows return all zeros.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean, got {mask.dtype}.")
    try:
        expanded_mask = torch.broadcast_to(mask, logits.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to logits "
            f"shape {tuple(logits.shape)}."
        ) from error

    masked_logits = torch.where(
        expanded_mask,
        logits,
        torch.full_like(logits, -torch.inf),
    )
    maximum = masked_logits.amax(dim=dim, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    safe_delta = torch.where(
        expanded_mask,
        logits - maximum,
        torch.zeros_like(logits),
    )
    exponentials = torch.exp(safe_delta) * expanded_mask.to(logits.dtype)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    return exponentials / denominator.clamp_min(torch.finfo(logits.dtype).tiny)


class LidarRayEvidenceStem(nn.Module):
    """LiDAR frame stem with circular padding only along azimuth."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        if out_channels < 2:
            raise ValueError(
                f"out_channels must be at least two, got {out_channels}."
            )
        self.first_conv = CircularAzimuthConv2d(
            3,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 2),
            bias=False,
        )
        self.first_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.second_conv = CircularAzimuthConv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            bias=False,
        )
        self.second_norm = nn.GroupNorm(_group_count(out_channels), out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.first_norm(self.first_conv(inputs)))
        return F.silu(self.second_norm(self.second_conv(features)))


class DepthRayEvidenceStem(nn.Module):
    """Perspective-depth frame stem with ordinary non-periodic boundaries."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        if out_channels < 2:
            raise ValueError(
                f"out_channels must be at least two, got {out_channels}."
            )
        self.first_conv = nn.Conv2d(
            3,
            out_channels,
            kernel_size=3,
            stride=(1, 2),
            padding=1,
            padding_mode="zeros",
            bias=False,
        )
        self.first_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.second_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=(2, 2),
            padding=1,
            padding_mode="zeros",
            bias=False,
        )
        self.second_norm = nn.GroupNorm(_group_count(out_channels), out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.first_norm(self.first_conv(inputs)))
        return F.silu(self.second_norm(self.second_conv(features)))


class _QueryEvidenceAttention(nn.Module):
    """Multi-head query attention that safely accepts an absent modality."""

    def __init__(self, token_dim: int, num_heads: int) -> None:
        super().__init__()
        if token_dim <= 0 or num_heads <= 0:
            raise ValueError("token_dim and num_heads must be positive.")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim must be divisible by num_heads, got {token_dim} "
                f"and {num_heads}."
            )
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_projection = nn.Linear(self.token_dim, self.token_dim)
        self.key_projection = nn.Linear(self.token_dim, self.token_dim)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim)
        self.output_projection = nn.Linear(self.token_dim, self.token_dim)

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        token_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_queries, _ = queries.shape
        num_tokens = tokens.shape[1]

        query = self.query_projection(queries).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        keys = self.key_projection(tokens).reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        values = self.value_projection(tokens).reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        logits = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        attention = _masked_softmax(
            logits,
            token_available[:, None, None, :],
            dim=-1,
        )
        attended = torch.matmul(attention, values)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch_size,
            num_queries,
            self.token_dim,
        )
        # Fully absent modalities must remain exactly zero even if the output
        # projection gains a bias in a future revision.
        available = token_available.any(dim=1).to(attended.dtype).view(
            batch_size,
            1,
            1,
        )
        return self.output_projection(attended) * available, attention


class MultimodalRayEvidenceEncoder(nn.Module):
    """Fuse timestamped LiDAR and depth ray evidence into a fixed embedding.

    The two sensor stems are independent.  Each stem keeps a private projection
    and also passes through a shared projection before a sensor-specific token
    projection.  This creates a common token dimension without an alignment
    loss or an implicit requirement that the two modalities encode identical
    information.

    ``query_quality`` is query-local reliability metadata: attention-weighted
    observed fraction, return fraction, freshness, and bounded age.  It is not
    a claim that learned token content is spatially independent, because the
    convolutional stems currently use frame-level GroupNorm.

    A missing modality contributes exactly zero evidence and zero gate mass.
    If every modality admitted by the selected mode is missing, the encoder
    returns an exact zero terrain embedding plus ``terrain_available=False`` in
    diagnostics.  The enclosing policy is expected to retain a separate
    proprioceptive stability path.

    Args:
        proprio_dim: Width of the deployable proprioceptive input.
        lidar_max_range: Finite positive LiDAR range represented by normalized
            value one.  Values above it are clamped.
        depth_max_range: Finite positive depth-camera range represented by
            normalized value one.  Values above it are clamped.
        lidar_min_range: Finite non-negative lower LiDAR clamp.
        depth_min_range: Finite non-negative lower depth-camera clamp.
        output_dim: Width of the returned embedding.
        stem_channels: Width of both sensor-specific convolutional stems.
        private_dim: Width reserved for sensor-specific token features.
        shared_dim: Width produced by the shared projection.
        token_dim: Common query/key/value token width.
        num_heads: Number of attention heads per modality.
        num_queries: Number of generic proprioception-conditioned queries.
        age_time_scale: Positive time scale used to normalize frame ages.
        mode: One of ``reliability``, ``concat``, ``lidar_only``,
            ``depth_only``, or ``no_reliability``.
    """

    def __init__(
        self,
        proprio_dim: int,
        *,
        lidar_max_range: float,
        depth_max_range: float,
        lidar_min_range: float = 0.0,
        depth_min_range: float = 0.0,
        output_dim: int = 64,
        stem_channels: int = 32,
        private_dim: int = 24,
        shared_dim: int = 24,
        token_dim: int = 64,
        num_heads: int = 4,
        num_queries: int = 4,
        age_time_scale: float = 0.5,
        mode: str = "reliability",
    ) -> None:
        super().__init__()
        integer_values = {
            "proprio_dim": proprio_dim,
            "output_dim": output_dim,
            "stem_channels": stem_channels,
            "private_dim": private_dim,
            "shared_dim": shared_dim,
            "token_dim": token_dim,
            "num_heads": num_heads,
            "num_queries": num_queries,
        }
        non_integer = {
            name: value
            for name, value in integer_values.items()
            if isinstance(value, bool) or not isinstance(value, Integral)
        }
        if non_integer:
            raise ValueError(
                f"All dimensions must be integers, got {non_integer}."
            )
        invalid = {name: value for name, value in integer_values.items() if value <= 0}
        if invalid:
            raise ValueError(f"All dimensions must be positive, got {invalid}.")
        if stem_channels < 2:
            raise ValueError(
                f"stem_channels must be at least two, got {stem_channels}."
            )
        if token_dim < 2:
            raise ValueError(f"token_dim must be at least two, got {token_dim}.")
        if output_dim < 2:
            raise ValueError(f"output_dim must be at least two, got {output_dim}.")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim must be divisible by num_heads, got {token_dim} "
                f"and {num_heads}."
            )
        if not math.isfinite(age_time_scale) or age_time_scale <= 0.0:
            raise ValueError(
                f"age_time_scale must be finite and positive, got {age_time_scale}."
            )
        self._validate_range_contract(
            "lidar",
            lidar_min_range,
            lidar_max_range,
        )
        self._validate_range_contract(
            "depth",
            depth_min_range,
            depth_max_range,
        )

        self.proprio_dim = int(proprio_dim)
        self.output_dim = int(output_dim)
        self.stem_channels = int(stem_channels)
        self.private_dim = int(private_dim)
        self.shared_dim = int(shared_dim)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.num_queries = int(num_queries)
        self.age_time_scale = float(age_time_scale)
        self.lidar_min_range = float(lidar_min_range)
        self.lidar_max_range = float(lidar_max_range)
        self.depth_min_range = float(depth_min_range)
        self.depth_max_range = float(depth_max_range)
        self.mode = self._normalize_mode(mode)

        self.lidar_stem = LidarRayEvidenceStem(self.stem_channels)
        self.depth_stem = DepthRayEvidenceStem(self.stem_channels)
        self.lidar_private_projection = nn.Conv2d(
            self.stem_channels,
            self.private_dim,
            kernel_size=1,
            bias=False,
        )
        self.depth_private_projection = nn.Conv2d(
            self.stem_channels,
            self.private_dim,
            kernel_size=1,
            bias=False,
        )
        self.shared_projection = nn.Conv2d(
            self.stem_channels,
            self.shared_dim,
            kernel_size=1,
            bias=False,
        )
        combined_dim = self.private_dim + self.shared_dim
        self.lidar_token_projection = nn.Conv2d(
            combined_dim,
            self.token_dim,
            kernel_size=1,
            bias=False,
        )
        self.depth_token_projection = nn.Conv2d(
            combined_dim,
            self.token_dim,
            kernel_size=1,
            bias=False,
        )
        self.lidar_token_norm = nn.LayerNorm(self.token_dim)
        self.depth_token_norm = nn.LayerNorm(self.token_dim)
        self.lidar_position_projection = nn.Linear(3, self.token_dim, bias=False)
        self.depth_position_projection = nn.Linear(2, self.token_dim, bias=False)
        self.lidar_state_projection = nn.Linear(2, self.token_dim, bias=False)
        self.depth_state_projection = nn.Linear(2, self.token_dim, bias=False)

        self.lidar_age_projection = nn.Sequential(
            nn.Linear(2, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.depth_age_projection = nn.Sequential(
            nn.Linear(2, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )

        self.state_query_projection = nn.Linear(
            self.proprio_dim,
            self.num_queries * self.token_dim,
        )
        self.query_bias = nn.Parameter(torch.zeros(1, self.num_queries, self.token_dim))
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.lidar_attention = _QueryEvidenceAttention(self.token_dim, self.num_heads)
        self.depth_attention = _QueryEvidenceAttention(self.token_dim, self.num_heads)

        self.gate_query_projection = nn.Linear(self.token_dim, self.token_dim)
        self.lidar_gate_evidence_projection = nn.Linear(self.token_dim, self.token_dim)
        self.depth_gate_evidence_projection = nn.Linear(self.token_dim, self.token_dim)
        self.lidar_quality_projection = nn.Linear(4, self.token_dim)
        self.depth_quality_projection = nn.Linear(4, self.token_dim)
        self.lidar_gate_score = nn.Linear(self.token_dim, 1)
        self.depth_gate_score = nn.Linear(self.token_dim, 1)

        self.concat_projection = nn.Sequential(
            nn.Linear(2 * self.token_dim + 2, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.output_projection = nn.Linear(
            self.num_queries * self.token_dim,
            self.output_dim,
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        lidar_frame_ages: torch.Tensor,
        depth_frame_ages: torch.Tensor,
        proprio: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Return only the fixed-width policy embedding."""
        embedding, _ = self.forward_with_diagnostics(
            lidar,
            depth,
            lidar_frame_ages,
            depth_frame_ages,
            proprio,
            mode=mode,
        )
        return embedding

    def forward_with_diagnostics(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        lidar_frame_ages: torch.Tensor,
        depth_frame_ages: torch.Tensor,
        proprio: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the embedding and evidence-selection diagnostics."""
        resolved_mode = self.mode if mode is None else self._normalize_mode(mode)
        self._validate_inputs(
            lidar,
            depth,
            lidar_frame_ages,
            depth_frame_ages,
            proprio,
        )

        compute_dtype = self.query_bias.dtype
        compute_age_scale = self.query_bias.new_tensor(self.age_time_scale)
        if not bool(torch.isfinite(compute_age_scale)) or not bool(
            compute_age_scale > 0.0
        ):
            raise ValueError(
                "age_time_scale is not representable as a finite positive "
                f"value in compute dtype {compute_dtype}."
            )
        lidar = lidar.to(dtype=compute_dtype)
        depth = depth.to(dtype=compute_dtype)
        lidar_frame_ages = lidar_frame_ages.to(dtype=compute_dtype)
        depth_frame_ages = depth_frame_ages.to(dtype=compute_dtype)
        proprio = proprio.to(dtype=compute_dtype)
        converted_inputs = {
            "lidar": lidar,
            "depth": depth,
            "lidar_frame_ages": lidar_frame_ages,
            "depth_frame_ages": depth_frame_ages,
            "proprio": proprio,
        }
        for name, tensor in converted_inputs.items():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"{name} is not representable as finite {compute_dtype} "
                    "encoder input."
                )

        lidar_encoded = self._encode_modality(
            lidar,
            lidar_frame_ages,
            sensor="lidar",
            stem=self.lidar_stem,
            private_projection=self.lidar_private_projection,
            token_projection=self.lidar_token_projection,
            token_norm=self.lidar_token_norm,
            age_projection=self.lidar_age_projection,
            position_projection=self.lidar_position_projection,
            state_projection=self.lidar_state_projection,
            minimum_range=self.lidar_min_range,
            maximum_range=self.lidar_max_range,
        )
        depth_encoded = self._encode_modality(
            depth,
            depth_frame_ages,
            sensor="depth",
            stem=self.depth_stem,
            private_projection=self.depth_private_projection,
            token_projection=self.depth_token_projection,
            token_norm=self.depth_token_norm,
            age_projection=self.depth_age_projection,
            position_projection=self.depth_position_projection,
            state_projection=self.depth_state_projection,
            minimum_range=self.depth_min_range,
            maximum_range=self.depth_max_range,
        )

        lidar_available = lidar_encoded["token_observed"].any(dim=1)
        depth_available = depth_encoded["token_observed"].any(dim=1)
        both_missing = ~lidar_available & ~depth_available

        queries = self.state_query_projection(proprio).reshape(
            proprio.shape[0],
            self.num_queries,
            self.token_dim,
        )
        queries = self.query_norm(queries + self.query_bias)
        lidar_token_available = lidar_available[:, None].expand_as(
            lidar_encoded["token_observed"]
        )
        depth_token_available = depth_available[:, None].expand_as(
            depth_encoded["token_observed"]
        )
        lidar_evidence, lidar_attention = self.lidar_attention(
            queries,
            lidar_encoded["tokens"],
            lidar_token_available,
        )
        depth_evidence, depth_attention = self.depth_attention(
            queries,
            depth_encoded["tokens"],
            depth_token_available,
        )
        lidar_query_quality = self._attention_weighted_query_quality(
            lidar_attention,
            lidar_encoded["token_quality"],
        )
        depth_query_quality = self._attention_weighted_query_quality(
            depth_attention,
            depth_encoded["token_quality"],
        )

        availability = torch.stack((lidar_available, depth_available), dim=-1)
        if resolved_mode == "reliability":
            gates = self._reliability_gates(
                queries,
                lidar_evidence,
                depth_evidence,
                lidar_query_quality,
                depth_query_quality,
                availability,
            )
            fused_queries = (
                gates[..., 0:1] * lidar_evidence
                + gates[..., 1:2] * depth_evidence
            )
        elif resolved_mode == "no_reliability":
            # Fixed availability-normalized averaging is the explicit
            # no-reliability ablation.  It cannot silently learn a content gate.
            gates = self._fixed_availability_gates(availability)
            fused_queries = (
                gates[..., 0:1] * lidar_evidence
                + gates[..., 1:2] * depth_evidence
            )
        elif resolved_mode == "concat":
            gates = self._fixed_availability_gates(availability)
            availability_features = availability.to(queries.dtype)
            availability_features = availability_features[:, None, :].expand(
                -1,
                self.num_queries,
                -1,
            )
            fused_queries = self.concat_projection(
                torch.cat(
                    (
                        lidar_evidence,
                        depth_evidence,
                        availability_features,
                    ),
                    dim=-1,
                )
            )
        elif resolved_mode == "lidar_only":
            gates = torch.zeros(
                proprio.shape[0],
                self.num_queries,
                2,
                dtype=queries.dtype,
                device=queries.device,
            )
            gates[..., 0] = lidar_available.to(queries.dtype).unsqueeze(-1)
            fused_queries = lidar_evidence
        else:
            gates = torch.zeros(
                proprio.shape[0],
                self.num_queries,
                2,
                dtype=queries.dtype,
                device=queries.device,
            )
            gates[..., 1] = depth_available.to(queries.dtype).unsqueeze(-1)
            fused_queries = depth_evidence

        if resolved_mode == "lidar_only":
            terrain_available = lidar_available
        elif resolved_mode == "depth_only":
            terrain_available = depth_available
        else:
            terrain_available = lidar_available | depth_available
        embedding = self.output_norm(
            self.output_projection(fused_queries.flatten(start_dim=1))
        )
        # Sensor blackout is an expected deployment state, not an exception.
        # Return an exact zero terrain feature so the enclosing actor can fall
        # back to its separate proprioceptive stability branch without
        # hallucinating evidence from projection biases.
        embedding = embedding * terrain_available.to(embedding.dtype).unsqueeze(-1)
        diagnostics = {
            "query_gates": gates,
            "lidar_attention": lidar_attention,
            "depth_attention": depth_attention,
            "lidar_token_observed": lidar_encoded["token_observed"],
            "depth_token_observed": depth_encoded["token_observed"],
            "lidar_token_return_valid": lidar_encoded["token_return_valid"],
            "depth_token_return_valid": depth_encoded["token_return_valid"],
            "lidar_token_observed_fraction": lidar_encoded[
                "token_observed_fraction"
            ],
            "depth_token_observed_fraction": depth_encoded[
                "token_observed_fraction"
            ],
            "lidar_token_return_fraction": lidar_encoded["token_return_fraction"],
            "depth_token_return_fraction": depth_encoded["token_return_fraction"],
            "lidar_token_quality": lidar_encoded["token_quality"],
            "depth_token_quality": depth_encoded["token_quality"],
            "lidar_frame_coverage": lidar_encoded["frame_coverage"],
            "depth_frame_coverage": depth_encoded["frame_coverage"],
            "lidar_frame_validity": lidar_encoded["frame_validity"],
            "depth_frame_validity": depth_encoded["frame_validity"],
            "lidar_frame_ages": lidar_frame_ages,
            "depth_frame_ages": depth_frame_ages,
            "lidar_global_quality": lidar_encoded["global_quality"],
            "depth_global_quality": depth_encoded["global_quality"],
            "lidar_query_quality": lidar_query_quality,
            "depth_query_quality": depth_query_quality,
            "lidar_available": lidar_available,
            "depth_available": depth_available,
            "both_modalities_missing": both_missing,
            "terrain_available": terrain_available,
        }
        return embedding, diagnostics

    def _encode_modality(
        self,
        history: torch.Tensor,
        frame_ages: torch.Tensor,
        *,
        sensor: str,
        stem: nn.Module,
        private_projection: nn.Module,
        token_projection: nn.Module,
        token_norm: nn.Module,
        age_projection: nn.Module,
        position_projection: nn.Module,
        state_projection: nn.Module,
        minimum_range: float,
        maximum_range: float,
    ) -> dict[str, torch.Tensor]:
        batch_size, history_length, _, height, width = history.shape
        metric_range = history[:, :, 0]
        return_valid = history[:, :, 1]
        ray_observed = history[:, :, 2]
        frame_coverage = ray_observed.mean(dim=(-2, -1))
        frame_validity = return_valid.mean(dim=(-2, -1))

        normalized_range = self._normalize_metric_range(
            metric_range,
            return_valid,
            minimum_range=minimum_range,
            maximum_range=maximum_range,
        )
        prepared = torch.stack(
            (
                normalized_range,
                return_valid,
                ray_observed,
            ),
            dim=2,
        )
        frame_features = stem(prepared.flatten(0, 1))
        private_features = private_projection(frame_features)
        shared_features = self.shared_projection(frame_features)
        token_features = token_projection(
            torch.cat((private_features, shared_features), dim=1)
        )
        token_height, token_width = token_features.shape[-2:]
        tokens_per_frame = token_height * token_width
        tokens = token_features.flatten(start_dim=2).transpose(1, 2).reshape(
            batch_size,
            history_length,
            tokens_per_frame,
            self.token_dim,
        )

        raw_state = torch.stack((ray_observed, return_valid), dim=2).flatten(0, 1)
        pooled_state = _pool_stem_receptive_fields(raw_state, sensor=sensor)
        pooled_support = _pool_stem_receptive_fields(
            torch.ones(
                1,
                1,
                height,
                width,
                device=history.device,
                dtype=history.dtype,
            ),
            sensor=sensor,
        )
        if tuple(pooled_state.shape[-2:]) != (token_height, token_width):
            raise RuntimeError(
                f"{sensor} quality pooling produced spatial shape "
                f"{tuple(pooled_state.shape[-2:])}, but its stem produced "
                f"{(token_height, token_width)}."
            )
        pooled_state = pooled_state / pooled_support.clamp_min(
            torch.finfo(pooled_state.dtype).tiny
        )
        token_observed_fraction = pooled_state[:, 0].reshape(
            batch_size,
            history_length,
            tokens_per_frame,
        )
        token_return_fraction = pooled_state[:, 1].reshape(
            batch_size,
            history_length,
            tokens_per_frame,
        )
        token_state = torch.stack(
            (token_observed_fraction, token_return_fraction),
            dim=-1,
        )
        state_encoding = state_projection(token_state)

        bounded_age = frame_ages / (frame_ages + self.age_time_scale)
        freshness = torch.exp(-frame_ages / self.age_time_scale)
        age_features = torch.stack(
            (
                bounded_age,
                freshness,
            ),
            dim=-1,
        )
        age_encoding = age_projection(age_features)
        if sensor == "lidar":
            raw_coordinates = build_lidar_raster_coordinates(
                height,
                width,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        elif sensor == "depth":
            raw_coordinates = build_depth_raster_coordinates(
                height,
                width,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            raise ValueError(f"Unsupported sensor kind '{sensor}'.")
        pooled_coordinates = _pool_stem_receptive_fields(
            raw_coordinates.permute(2, 0, 1).unsqueeze(0),
            sensor=sensor,
        )
        raster_coordinates = (
            pooled_coordinates / pooled_support.clamp_min(
                torch.finfo(pooled_coordinates.dtype).tiny
            )
        ).squeeze(0).permute(1, 2, 0)
        if sensor == "lidar":
            azimuth_norm = torch.linalg.vector_norm(
                raster_coordinates[..., :2],
                dim=-1,
                keepdim=True,
            )
            raster_coordinates = torch.cat(
                (
                    raster_coordinates[..., :2]
                    / azimuth_norm.clamp_min(
                        torch.finfo(raster_coordinates.dtype).eps
                    ),
                    raster_coordinates[..., 2:3],
                ),
                dim=-1,
            )
        position_encoding = position_projection(
            raster_coordinates.reshape(tokens_per_frame, -1)
        )
        tokens = token_norm(
            tokens
            + age_encoding[:, :, None, :]
            + position_encoding[None, None, :, :]
            + state_encoding
        )
        tokens = tokens.flatten(start_dim=1, end_dim=2)

        token_observed = token_observed_fraction.flatten(start_dim=1) > 0.0
        token_return_valid = token_return_fraction.flatten(start_dim=1) > 0.0
        token_observed_fraction = token_observed_fraction.flatten(start_dim=1)
        token_return_fraction = token_return_fraction.flatten(start_dim=1)

        token_freshness = freshness[:, :, None].expand(
            -1,
            -1,
            tokens_per_frame,
        )
        token_normalized_age = bounded_age[:, :, None].expand(
            -1,
            -1,
            tokens_per_frame,
        )
        token_quality = torch.stack(
            (
                token_observed_fraction,
                token_return_fraction,
                token_freshness.flatten(start_dim=1),
                token_normalized_age.flatten(start_dim=1),
            ),
            dim=-1,
        )
        global_quality = self._quality_summary(
            frame_coverage,
            frame_validity,
            frame_ages,
        )
        return {
            "tokens": tokens,
            "token_observed": token_observed,
            "token_return_valid": token_return_valid,
            "token_observed_fraction": token_observed_fraction,
            "token_return_fraction": token_return_fraction,
            "token_quality": token_quality,
            "frame_coverage": frame_coverage,
            "frame_validity": frame_validity,
            "global_quality": global_quality,
        }

    @staticmethod
    def _normalize_metric_range(
        metric_range: torch.Tensor,
        return_valid: torch.Tensor,
        *,
        minimum_range: float,
        maximum_range: float,
    ) -> torch.Tensor:
        """Linearly normalize a metric range using an explicit sensor contract."""
        limits = metric_range.new_tensor((minimum_range, maximum_range))
        if not bool(torch.isfinite(limits).all()) or not bool(
            limits[1] > limits[0]
        ):
            raise ValueError(
                "Range limits are not representable with positive span in "
                f"compute dtype {metric_range.dtype}."
            )
        clamped = metric_range.clamp(min=limits[0], max=limits[1])
        normalized = (clamped - limits[0]) / (limits[1] - limits[0])
        return normalized * return_valid

    @staticmethod
    def _attention_weighted_query_quality(
        attention: torch.Tensor,
        token_quality: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate local token quality separately for every policy query."""
        mean_attention = attention.mean(dim=1)
        return torch.einsum("bqn,bnf->bqf", mean_attention, token_quality)

    def _quality_summary(
        self,
        frame_coverage: torch.Tensor,
        frame_validity: torch.Tensor,
        frame_ages: torch.Tensor,
    ) -> torch.Tensor:
        coverage = frame_coverage.mean(dim=1)
        observed_mass = frame_coverage.sum(dim=1)
        validity_given_observed = (
            frame_validity.sum(dim=1) / observed_mass.clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        bounded_frame_age = frame_ages / (
            frame_ages + self.age_time_scale
        )
        freshness = (
            torch.exp(-frame_ages / self.age_time_scale) * frame_coverage
        ).sum(dim=1) / observed_mass.clamp_min(1.0e-6)
        normalized_age = (
            bounded_frame_age * frame_coverage
        ).sum(dim=1) / observed_mass.clamp_min(1.0e-6)
        available = observed_mass > 0.0
        return torch.stack(
            (
                coverage,
                torch.where(
                    available,
                    validity_given_observed,
                    torch.zeros_like(validity_given_observed),
                ),
                torch.where(
                    available,
                    freshness,
                    torch.zeros_like(freshness),
                ),
                torch.where(
                    available,
                    normalized_age,
                    torch.zeros_like(normalized_age),
                ),
            ),
            dim=-1,
        )

    def _reliability_gates(
        self,
        queries: torch.Tensor,
        lidar_evidence: torch.Tensor,
        depth_evidence: torch.Tensor,
        lidar_quality: torch.Tensor,
        depth_quality: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        query_features = self.gate_query_projection(queries)
        lidar_hidden = query_features + self.lidar_gate_evidence_projection(
            lidar_evidence
        )
        depth_hidden = query_features + self.depth_gate_evidence_projection(
            depth_evidence
        )
        lidar_hidden = lidar_hidden + self.lidar_quality_projection(
            lidar_quality
        )
        depth_hidden = depth_hidden + self.depth_quality_projection(
            depth_quality
        )
        lidar_score = self.lidar_gate_score(torch.tanh(lidar_hidden))
        depth_score = self.depth_gate_score(torch.tanh(depth_hidden))
        logits = torch.cat((lidar_score, depth_score), dim=-1)
        return _masked_softmax(
            logits,
            availability[:, None, :],
            dim=-1,
        )

    def _fixed_availability_gates(self, availability: torch.Tensor) -> torch.Tensor:
        weights = availability.to(self.query_bias.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return weights[:, None, :].expand(-1, self.num_queries, -1)

    @staticmethod
    def _validate_range_contract(
        sensor: str,
        minimum_range: float,
        maximum_range: float,
    ) -> None:
        try:
            minimum = float(minimum_range)
            maximum = float(maximum_range)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{sensor} range limits must be real numbers, got "
                f"minimum={minimum_range!r}, maximum={maximum_range!r}."
            ) from error
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < 0.0
            or maximum <= minimum
        ):
            raise ValueError(
                f"{sensor} range limits must be finite with "
                f"0 <= minimum < maximum, got {minimum} and {maximum}."
            )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError(f"mode must be a string, got {type(mode).__name__}.")
        normalized = mode.lower().replace("-", "_")
        if normalized not in _SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of {_SUPPORTED_MODES}, got '{mode}'."
            )
        return normalized

    def _validate_inputs(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        lidar_frame_ages: torch.Tensor,
        depth_frame_ages: torch.Tensor,
        proprio: torch.Tensor,
    ) -> None:
        self._validate_history_shape("lidar", lidar)
        self._validate_history_shape("depth", depth)
        batch_size = lidar.shape[0]
        if depth.shape[0] != batch_size:
            raise ValueError(
                "lidar and depth must have the same batch size, got "
                f"{batch_size} and {depth.shape[0]}."
            )
        expected_lidar_age_shape = (batch_size, lidar.shape[1])
        expected_depth_age_shape = (batch_size, depth.shape[1])
        if tuple(lidar_frame_ages.shape) != expected_lidar_age_shape:
            raise ValueError(
                "lidar_frame_ages must have shape [B, Tl] with "
                f"{expected_lidar_age_shape}, got {tuple(lidar_frame_ages.shape)}."
            )
        if tuple(depth_frame_ages.shape) != expected_depth_age_shape:
            raise ValueError(
                "depth_frame_ages must have shape [B, Td] with "
                f"{expected_depth_age_shape}, got {tuple(depth_frame_ages.shape)}."
            )
        if proprio.ndim != 2 or tuple(proprio.shape) != (
            batch_size,
            self.proprio_dim,
        ):
            raise ValueError(
                "proprio must have shape [B, P] with "
                f"P={self.proprio_dim}, got {tuple(proprio.shape)}."
            )

        tensors = {
            "lidar": lidar,
            "depth": depth,
            "lidar_frame_ages": lidar_frame_ages,
            "depth_frame_ages": depth_frame_ages,
            "proprio": proprio,
        }
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
            details = ", ".join(
                f"{name}={tensor.device}" for name, tensor in tensors.items()
            )
            raise ValueError(f"All inputs must use the same device, got {details}.")
        parameter_device = self.query_bias.device
        input_device = lidar.device
        if input_device != parameter_device:
            raise ValueError(
                "Inputs and encoder parameters must use the same device, got "
                f"inputs={input_device} and parameters={parameter_device}."
            )
        for name, tensor in tensors.items():
            if not tensor.is_floating_point():
                raise ValueError(
                    f"{name} must be a real floating-point tensor, got {tensor.dtype}."
                )
        for name, tensor in (
            ("lidar_frame_ages", lidar_frame_ages),
            ("depth_frame_ages", depth_frame_ages),
            ("proprio", proprio),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values.")
        if bool((lidar_frame_ages < 0.0).any()):
            raise ValueError("lidar_frame_ages must be non-negative.")
        if bool((depth_frame_ages < 0.0).any()):
            raise ValueError("depth_frame_ages must be non-negative.")

        self._validate_evidence_semantics("lidar", lidar)
        self._validate_evidence_semantics("depth", depth)

    @staticmethod
    def _validate_history_shape(name: str, history: torch.Tensor) -> None:
        if history.ndim != 5 or history.shape[2] != 3:
            raise ValueError(
                f"{name} must have shape [B, T, 3, H, W], got "
                f"{tuple(history.shape)}."
            )
        if history.shape[0] <= 0 or history.shape[1] <= 0:
            raise ValueError(f"{name} batch and history dimensions must be positive.")
        if history.shape[3] < 2 or history.shape[4] < 2:
            raise ValueError(
                f"{name} spatial dimensions must both be at least 2, got "
                f"{tuple(history.shape[-2:])}."
            )

    @staticmethod
    def _validate_evidence_semantics(name: str, history: torch.Tensor) -> None:
        if not bool(torch.isfinite(history).all()):
            raise ValueError(f"{name} must contain only finite values.")
        metric_range = history[:, :, 0]
        return_valid = history[:, :, 1]
        ray_observed = history[:, :, 2]
        for channel_name, mask in (
            ("return_valid", return_valid),
            ("ray_observed", ray_observed),
        ):
            binary = (mask == 0.0) | (mask == 1.0)
            if not bool(binary.all()):
                raise ValueError(
                    f"{name} {channel_name} must contain only exact 0/1 values."
                )
        if bool((return_valid > ray_observed).any()):
            raise ValueError(
                f"{name} requires return_valid <= ray_observed for every ray."
            )
        if bool((metric_range < 0.0).any()):
            raise ValueError(f"{name} metric range must be non-negative.")
        invalid_return = return_valid == 0.0
        if bool((metric_range[invalid_return] != 0.0).any()):
            raise ValueError(
                f"{name} metric range must be exactly zero without a valid return."
            )
        if bool((metric_range[return_valid == 1.0] <= 0.0).any()):
            raise ValueError(
                f"{name} valid returns must have strictly positive metric range."
            )


__all__ = [
    "DepthRayEvidenceStem",
    "LidarRayEvidenceStem",
    "MultimodalRayEvidenceEncoder",
    "build_depth_raster_coordinates",
    "build_lidar_raster_coordinates",
]
