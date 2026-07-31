# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ray_return_event_time import RayReturnEventTimeEncoder


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
        fusion_mode: str | None = None,
        event_time_mode: str | None = None,
        event_time_source: str = "none",
        event_time_scale_s: float = 0.5,
        acquisition_delta_proprio_dim: int = 0,
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
        self.event_time_mode = (
            None
            if event_time_mode is None
            else event_time_mode.lower().replace("-", "_")
        )
        self.event_time_source = event_time_source.lower().replace("-", "_")
        if (
            isinstance(acquisition_delta_proprio_dim, bool)
            or not isinstance(acquisition_delta_proprio_dim, int)
            or acquisition_delta_proprio_dim < 0
        ):
            raise ValueError(
                "acquisition_delta_proprio_dim must be a non-negative integer."
            )
        self.acquisition_delta_proprio_dim = int(
            acquisition_delta_proprio_dim
        )
        if self.event_time_mode not in (
            None,
            "packet_age",
            "per_return_age",
            "quantized_event_age",
            "age_zero",
        ):
            raise ValueError(
                "event_time_mode must be None, packet_age, per_return_age, "
                "quantized_event_age, or age_zero, got "
                f"{event_time_mode!r}."
            )
        if self.event_time_source not in (
            "none",
            "raycaster_packet",
            "raycaster_quantized_event",
            "livox_per_return",
        ):
            raise ValueError(
                "event_time_source must be none, raycaster_packet, "
                "raycaster_quantized_event, or livox_per_return, got "
                f"{event_time_source!r}."
            )
        if self.event_time_mode is None:
            if self.event_time_source != "none":
                raise ValueError(
                    "event_time_source must be 'none' when event time is disabled."
                )
            self.input_channels = 2
        else:
            if self.event_time_source == "none":
                raise ValueError(
                    "Enabled event time requires an explicit authenticated source."
                )
            if (
                self.event_time_mode == "per_return_age"
                and self.event_time_source != "livox_per_return"
            ):
                raise ValueError(
                    "per_return_age requires source='livox_per_return'; "
                    "RayCaster exposes packet-level acquisition only."
                )
            if (
                self.event_time_mode == "quantized_event_age"
                and self.event_time_source != "raycaster_quantized_event"
            ):
                raise ValueError(
                    "quantized_event_age requires source="
                    "'raycaster_quantized_event'."
                )
            if (
                self.event_time_source == "raycaster_quantized_event"
                and self.event_time_mode != "quantized_event_age"
            ):
                raise ValueError(
                    "raycaster_quantized_event is valid only for the "
                    "explicit quantized_event_age mode."
                )
            self.input_channels = 5
        if self.acquisition_delta_proprio_dim > 0 and (
            self.event_time_mode != "per_return_age"
            or self.event_time_source != "livox_per_return"
        ):
            raise ValueError(
                "Acquisition delta-proprio requires authenticated Livox "
                "per-return timing."
            )
        resolved_fusion_mode = (
            "attention" if use_query_attention else "global"
        ) if fusion_mode is None else fusion_mode.lower().replace("-", "_")
        if resolved_fusion_mode not in ("attention", "global", "query_global"):
            raise ValueError(
                "fusion_mode must be one of ('attention', 'global', 'query_global'), "
                f"got '{fusion_mode}'."
            )
        self.fusion_mode = resolved_fusion_mode
        # Preserve this public compatibility flag for existing callers. The
        # QueryGlobal control deliberately does not perform spatial attention.
        self.use_query_attention = self.fusion_mode == "attention"
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
        if self.event_time_mode is None:
            self.event_time_encoder: RayReturnEventTimeEncoder | None = None
        else:
            encoder_mode = (
                "per_return_age"
                if self.event_time_mode in (
                    "per_return_age",
                    "quantized_event_age",
                )
                else "packet_age"
            )
            self.event_time_encoder = RayReturnEventTimeEncoder(
                history_length=self.history_length,
                input_spatial_size=self.vision_spatial_size,
                token_spatial_size=self.token_spatial_size,
                token_dim=self.token_dim,
                mode=encoder_mode,
                age_time_scale_s=event_time_scale_s,
            )
        if (
            self.vision_spatial_size[0] % self.token_spatial_size[0] == 0
            and self.vision_spatial_size[1] % self.token_spatial_size[1] == 0
        ):
            self.hit_pool_kernel = (
                self.vision_spatial_size[0] // self.token_spatial_size[0],
                self.vision_spatial_size[1] // self.token_spatial_size[1],
            )
        else:
            self.hit_pool_kernel = (0, 0)

        if self.acquisition_delta_proprio_dim == 0:
            self.delta_proprio_projection: nn.Module = nn.Identity()
        else:
            # Bias is deliberately disabled: a token with no valid return must
            # contribute exactly zero acquisition-state information.
            self.delta_proprio_projection = nn.Linear(
                self.acquisition_delta_proprio_dim,
                self.token_dim,
                bias=False,
            )

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
        acquisition_delta_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a fixed-size terrain embedding."""
        terrain_embedding, _, _ = self.forward_with_attention(
            ray_history,
            proprio_features,
            acquisition_delta_proprio,
        )
        return terrain_embedding

    def forward_with_attention(
        self,
        ray_history: torch.Tensor,
        proprio_features: torch.Tensor,
        acquisition_delta_proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embedding, cross-attention weights, and valid-token mask."""
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            self._validate_inputs(
                ray_history,
                proprio_features,
                acquisition_delta_proprio,
            )

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
        if self.event_time_encoder is not None:
            return_valid = hit_mask > 0.5
            packet_age_s = ray_history[:, :, 3, 0, 0]
            frame_valid = ray_history[:, :, 4, 0, 0] > 0.5
            if self.event_time_mode == "age_zero":
                packet_age_s = torch.zeros_like(packet_age_s)
                return_age_s: torch.Tensor | None = None
            elif self.event_time_mode in (
                "per_return_age",
                "quantized_event_age",
            ):
                return_age_s = ray_history[:, :, 2]
            else:
                return_age_s = None
            event_time_encoding = self.event_time_encoder(
                return_valid,
                packet_age_s,
                frame_valid,
                return_age_s,
            )
            tokens = tokens + event_time_encoding
        if self.acquisition_delta_proprio_dim > 0:
            if acquisition_delta_proprio is None:
                raise ValueError(
                    "Enabled acquisition delta-proprio input is required."
                )
            delta_encoding = self._encode_acquisition_delta_proprio(
                acquisition_delta_proprio.float(),
                hit_mask,
            )
            tokens = tokens + delta_encoding

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

        flattened_hit_mask = hit_mask.flatten(0, 1)
        if self.hit_pool_kernel[0] > 0:
            # This is exactly equivalent to adaptive max-pooling when the
            # fixed input raster is divisible by the token raster (including
            # MID-360's 16x96 -> 4x12 path), and has complete ONNX support.
            token_valid = F.max_pool2d(
                flattened_hit_mask,
                kernel_size=self.hit_pool_kernel,
                stride=self.hit_pool_kernel,
            )
        else:
            token_valid = F.adaptive_max_pool2d(
                flattened_hit_mask,
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

        if self.fusion_mode == "attention":
            attended_feature, attention_weights = self._query_attention(
                tokens,
                token_valid,
                proprio_features,
            )
        elif self.fusion_mode == "global":
            attended_feature = self.global_ablation_adapter(global_token)
            attention_weights = torch.zeros(
                batch_size,
                self.num_heads,
                self.num_queries,
                self.num_tokens,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            # Match Attention's fully-unknown deterministic fallback without
            # exposing the spatial sequence: collapse the same fixed
            # position/time tokens to one mean token.
            query_global_token = torch.where(
                token_valid.any(dim=1, keepdim=True),
                global_token,
                tokens.mean(dim=1),
            )
            attended_feature, query_gates = self._query_global(
                query_global_token,
                proprio_features,
            )
            # Keep the diagnostic tensor contract shared by all modes. Every
            # valid spatial/time entry receives the same mass, making it
            # explicit that QueryGlobal cannot select an individual token.
            # Fully unknown input uses a uniform deterministic fallback.
            safe_valid = torch.where(
                token_valid.any(dim=1, keepdim=True),
                token_valid,
                torch.ones_like(token_valid),
            )
            uniform_weights = safe_valid.to(tokens.dtype)
            uniform_weights = uniform_weights / uniform_weights.sum(
                dim=1,
                keepdim=True,
            )
            attention_weights = (
                query_gates.unsqueeze(-1)
                * uniform_weights[:, None, None, :]
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

    def _encode_acquisition_delta_proprio(
        self,
        acquisition_delta_proprio: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked acquisition-state tokens with shape ``[B,K,N,C]``."""
        batch_size = acquisition_delta_proprio.shape[0]
        valid = hit_mask[:, :, None] > 0.5
        # torch.where, rather than multiplication, prevents invalid NaN/Inf
        # leakage in exported graphs. Eager validation still rejects it.
        masked_delta = torch.where(
            valid,
            acquisition_delta_proprio,
            torch.zeros_like(acquisition_delta_proprio),
        )
        flat_delta = masked_delta.flatten(0, 1)
        flat_valid = hit_mask.flatten(0, 1).unsqueeze(1)
        if self.hit_pool_kernel[0] > 0:
            pooled_delta = F.avg_pool2d(
                flat_delta,
                kernel_size=self.hit_pool_kernel,
                stride=self.hit_pool_kernel,
            )
            pooled_valid = F.avg_pool2d(
                flat_valid,
                kernel_size=self.hit_pool_kernel,
                stride=self.hit_pool_kernel,
            )
        else:
            pooled_delta = F.adaptive_avg_pool2d(
                flat_delta,
                self.token_spatial_size,
            )
            pooled_valid = F.adaptive_avg_pool2d(
                flat_valid,
                self.token_spatial_size,
            )
        pooled_delta = pooled_delta / pooled_valid.clamp_min(1.0e-6)
        pooled_delta = torch.where(
            pooled_valid > 0.0,
            pooled_delta,
            torch.zeros_like(pooled_delta),
        )
        pooled_delta = pooled_delta.flatten(start_dim=2).transpose(1, 2)
        pooled_delta = pooled_delta.reshape(
            batch_size,
            self.history_length,
            self.num_spatial_tokens,
            self.acquisition_delta_proprio_dim,
        )
        return self.delta_proprio_projection(pooled_delta)

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
            -3.4028234663852886e38,
        )
        attention_weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention_weights, values)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_queries * self.token_dim,
        )
        return self.attended_projection(attended), attention_weights

    def _query_global(
        self,
        global_token: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate one masked global terrain token with proprioceptive queries.

        Unlike spatial cross-attention, this causal control never receives the
        token sequence or its coordinates. Keys and values can therefore only
        encode the masked global average computed by the caller.
        """
        batch_size = global_token.shape[0]
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

        key = self.key_projection(global_token).reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
        )
        value = self.value_projection(global_token).reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
        )
        # Attention's softmax produces a unit-sum value mixture for every
        # query/head.  Center this non-spatial control at the same unit gain so
        # its branch is not initialized at roughly half the Attention scale.
        # These gates are gains in ``(0, 2)``, not probabilities.
        query_gates = 2.0 * torch.sigmoid(
            (query * key.unsqueeze(2)).sum(dim=-1) * self.attention_scale
        )
        attended = query_gates.unsqueeze(-1) * value.unsqueeze(2)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_queries * self.token_dim,
        )
        return self.attended_projection(attended), query_gates

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
        acquisition_delta_proprio: torch.Tensor | None,
    ) -> None:
        expected_shape = (
            self.history_length,
            self.input_channels,
            self.vision_spatial_size[0],
            self.vision_spatial_size[1],
        )
        if ray_history.ndim != 5 or tuple(ray_history.shape[1:]) != expected_shape:
            raise ValueError(
                "ray_history has the wrong event-time observation shape: "
                f"expected [B,T,C,H,W] with [T,C,H,W]={expected_shape}, "
                f"got {tuple(ray_history.shape)}."
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
        if self.acquisition_delta_proprio_dim == 0:
            if acquisition_delta_proprio is not None:
                raise ValueError(
                    "Disabled acquisition delta-proprio rejects an unused tensor."
                )
        else:
            if acquisition_delta_proprio is None:
                raise ValueError(
                    "Enabled acquisition delta-proprio input is required."
                )
            expected_delta_shape = (
                ray_history.shape[0],
                self.history_length,
                self.acquisition_delta_proprio_dim,
                self.vision_spatial_size[0],
                self.vision_spatial_size[1],
            )
            if tuple(acquisition_delta_proprio.shape) != expected_delta_shape:
                raise ValueError(
                    "acquisition_delta_proprio must have shape [B,K,D,H,W] "
                    f"equal to {expected_delta_shape}, got "
                    f"{tuple(acquisition_delta_proprio.shape)}."
                )
            if acquisition_delta_proprio.device != ray_history.device:
                raise ValueError(
                    "acquisition_delta_proprio and ray_history must share a device."
                )
            if not acquisition_delta_proprio.is_floating_point():
                raise ValueError("acquisition_delta_proprio must be floating point.")
            if not bool(torch.isfinite(acquisition_delta_proprio).all()):
                raise ValueError("acquisition_delta_proprio must be finite.")
            requested_hit = ray_history[:, :, 1] > 0.5
            if bool(
                (
                    ~requested_hit[:, :, None]
                    & (acquisition_delta_proprio != 0.0)
                ).any()
            ):
                raise ValueError(
                    "Invalid returns must carry exactly zero acquisition delta-proprio."
                )
        if self.input_channels == 5:
            return_age = ray_history[:, :, 2]
            packet_age_map = ray_history[:, :, 3]
            frame_valid_map = ray_history[:, :, 4]
            packet_age = packet_age_map[:, :, :1, :1]
            frame_valid = frame_valid_map[:, :, :1, :1]
            if not bool((packet_age_map == packet_age).all()):
                raise ValueError("Packet-age channel must be spatially constant per frame.")
            if not bool((frame_valid_map == frame_valid).all()):
                raise ValueError("Frame-valid channel must be spatially constant per frame.")
            if not bool(((frame_valid == 0.0) | (frame_valid == 1.0)).all()):
                raise ValueError("Frame-valid channel must be exactly binary.")
            if not bool(torch.isfinite(packet_age).all()) or bool((packet_age < 0).any()):
                raise ValueError("Packet age must be finite and non-negative.")
            requested_hit = ray_history[:, :, 1] > 0.5
            if bool((~requested_hit & (return_age != 0.0)).any()):
                raise ValueError("Invalid returns must carry exactly zero return age.")
            if self.event_time_mode in (
                "per_return_age",
                "quantized_event_age",
            ):
                if not bool(torch.isfinite(return_age).all()):
                    raise ValueError("Per-return age must be finite.")
                if bool(
                    (
                        requested_hit
                        & (return_age + 1.0e-7 < packet_age)
                    ).any()
                ):
                    raise ValueError("Per-return age must be >= packet age.")
            elif bool((return_age != 0.0).any()):
                raise ValueError(
                    "Packet-age and age-zero observations must keep the "
                    "per-return-age channel exactly zero."
                )
