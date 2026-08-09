# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous event-time conditioning for sparse ray-return tokens.

The MID-360 ``CustomMsg`` interface can provide one acquisition timestamp for
every *successful return*.  A policy-rate spherical raster normally discards
that information and treats every return in a packet as simultaneous.  This
module preserves the timing distinction without constructing or registering a
terrain map.

The module is deliberately a CPU-testable integration scaffold.  It does not
bin points, infer missing rays, or modify an actor.  Callers must provide:

* ``return_valid``: successful-return support only;
* ``packet_age_s``: action-time minus packet capture-end time;
* optional ``return_age_s``: action-time minus each successful return time;
* ``frame_valid``: whether a history slot contains a real acquisition window.

Unknown cells never receive timing features.  In particular, a missing return
is not interpreted as an emitted no-return ray.  Invalid return-age cells must
be exactly zero, preventing hidden information from leaking through padding.

Three modes form a controlled causal family with the same output shape:

``index_only``
    Uses only the fixed oldest-to-newest history-slot position.
``packet_age``
    Uses the measured age of each packet, shared by its successful returns.
``per_return_age``
    Uses measured per-return acquisition ages, including within-packet spread.

The output is an additive ``[B, T, Nt, D]`` token encoding.  A token with no
successful return is bitwise zero even when projection layers have biases.
Optional packet-time motion deltas can be added through a separate, explicitly
switchable branch so event-time and ego-motion effects remain independently
ablatable.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_EVENT_TIME_MODES: Final[tuple[str, ...]] = (
    "index_only",
    "packet_age",
    "per_return_age",
)


def _strict_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {resolved}.")
    return resolved


def _strict_spatial_size(
    name: str,
    value: tuple[int, int] | list[int],
) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    return (
        _strict_positive_integer(f"{name}[0]", value[0]),
        _strict_positive_integer(f"{name}[1]", value[1]),
    )


class RayReturnEventTimeEncoder(nn.Module):
    """Encode packet/return acquisition time at the ray-token resolution.

    Args:
        history_length: Number of oldest-to-newest packet slots.
        input_spatial_size: Input spherical raster size ``(H, W)``.
        token_spatial_size: Downsampled token raster size ``(Ht, Wt)``.
            Both input dimensions must be divisible by their token dimensions
            so every token has a fixed, auditable ray footprint.
        token_dim: Width of the returned additive token encoding.
        mode: One of :data:`SUPPORTED_EVENT_TIME_MODES`.
        age_time_scale_s: Positive scale used by ``age/(age+scale)`` and the
            Fourier features.  It is not a freshness cutoff.
        num_fourier_frequencies: Number of deterministic age frequencies.
        hidden_dim: Width of the learned time projection.
        motion_delta_dim: Width of optional packet-to-action motion deltas.
            Zero removes the motion branch entirely.
        motion_hidden_dim: Hidden width of the optional motion projection.
    """

    def __init__(
        self,
        *,
        history_length: int,
        input_spatial_size: tuple[int, int] = (16, 96),
        token_spatial_size: tuple[int, int] = (4, 12),
        token_dim: int = 64,
        mode: str = "per_return_age",
        age_time_scale_s: float = 0.5,
        num_fourier_frequencies: int = 4,
        hidden_dim: int = 64,
        motion_delta_dim: int = 0,
        motion_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.history_length = _strict_positive_integer(
            "history_length",
            history_length,
        )
        self.input_spatial_size = _strict_spatial_size(
            "input_spatial_size",
            input_spatial_size,
        )
        self.token_spatial_size = _strict_spatial_size(
            "token_spatial_size",
            token_spatial_size,
        )
        for input_size, token_size, axis in zip(
            self.input_spatial_size,
            self.token_spatial_size,
            ("height", "width"),
            strict=True,
        ):
            if input_size % token_size != 0:
                raise ValueError(
                    f"input {axis} {input_size} must be divisible by token "
                    f"{axis} {token_size}."
                )
        self.pool_kernel = (
            self.input_spatial_size[0] // self.token_spatial_size[0],
            self.input_spatial_size[1] // self.token_spatial_size[1],
        )
        self.rays_per_token = self.pool_kernel[0] * self.pool_kernel[1]
        self.num_tokens_per_frame = (
            self.token_spatial_size[0] * self.token_spatial_size[1]
        )
        self.token_dim = _strict_positive_integer("token_dim", token_dim)
        self.mode = self._normalize_mode(mode)
        if (
            isinstance(age_time_scale_s, bool)
            or not math.isfinite(float(age_time_scale_s))
            or float(age_time_scale_s) <= 0.0
        ):
            raise ValueError(
                "age_time_scale_s must be finite and positive, got "
                f"{age_time_scale_s!r}."
            )
        self.age_time_scale_s = float(age_time_scale_s)
        self.num_fourier_frequencies = _strict_positive_integer(
            "num_fourier_frequencies",
            num_fourier_frequencies,
        )
        hidden_dim = _strict_positive_integer("hidden_dim", hidden_dim)
        if (
            isinstance(motion_delta_dim, bool)
            or not isinstance(motion_delta_dim, Integral)
            or int(motion_delta_dim) < 0
        ):
            raise ValueError(
                "motion_delta_dim must be a non-negative integer, got "
                f"{motion_delta_dim!r}."
            )
        self.motion_delta_dim = int(motion_delta_dim)
        motion_hidden_dim = _strict_positive_integer(
            "motion_hidden_dim",
            motion_hidden_dim,
        )

        # Five explicit summary channels plus sin/cos features for mean, min,
        # and max acquisition age.
        time_feature_dim = 5 + 6 * self.num_fourier_frequencies
        self.time_projection = nn.Sequential(
            nn.Linear(time_feature_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, self.token_dim),
        )
        # Preserve the association between acquisition age and the ray's
        # within-token angular location.  Pooling only mean/min/max/span would
        # be invariant to swapping two return ages inside one token footprint,
        # which is precisely the motion-distortion signal H2 needs to test.
        # Keep the dense raster branch to two channels.  Fourier expansion is
        # deliberately deferred to the pooled summaries above; materializing
        # every frequency at every ray would add several gigabytes of PPO
        # activation memory at the formal 6,144-sample minibatch size.
        ray_time_feature_dim = 2
        self.ray_time_projection = nn.Conv2d(
            ray_time_feature_dim,
            self.token_dim,
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
            bias=False,
        )
        if self.motion_delta_dim > 0:
            self.motion_projection: nn.Module | None = nn.Sequential(
                nn.Linear(self.motion_delta_dim, motion_hidden_dim),
                nn.SiLU(inplace=False),
                nn.Linear(motion_hidden_dim, self.token_dim),
            )
        else:
            self.motion_projection = None
        frequencies = torch.arange(
            1,
            self.num_fourier_frequencies + 1,
            dtype=torch.float32,
        )
        self.register_buffer(
            "age_frequencies",
            frequencies,
            persistent=False,
        )

    def forward(
        self,
        return_valid: torch.Tensor,
        packet_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
        return_age_s: torch.Tensor | None = None,
        frame_motion_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the additive ``[B,T,Nt,D]`` event-time encoding."""
        output, _ = self.forward_with_diagnostics(
            return_valid,
            packet_age_s,
            frame_valid,
            return_age_s,
            frame_motion_delta,
        )
        return output

    def forward_with_diagnostics(
        self,
        return_valid: torch.Tensor,
        packet_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
        return_age_s: torch.Tensor | None = None,
        frame_motion_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return event-time encoding and auditable token summaries."""
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            self._validate_inputs(
                return_valid,
                packet_age_s,
                frame_valid,
                return_age_s,
                frame_motion_delta,
            )

        compute_dtype = self.age_frequencies.dtype
        packet_age_raw = packet_age_s.to(dtype=compute_dtype)
        packet_age_valid = (
            torch.isfinite(packet_age_raw)
            & (packet_age_raw >= 0.0)
        )
        packet_age = torch.where(
            packet_age_valid,
            packet_age_raw,
            torch.zeros_like(packet_age_raw),
        )
        batch_size = return_valid.shape[0]
        history_length = return_valid.shape[1]
        frame_contract_valid = frame_valid & packet_age_valid
        safe_motion: torch.Tensor | None = None
        if self.motion_projection is not None:
            if frame_motion_delta is None:
                raise ValueError(
                    "motion_delta_dim > 0 requires frame_motion_delta."
                )
            motion_raw = frame_motion_delta.to(dtype=compute_dtype)
            motion_finite = torch.isfinite(motion_raw).all(dim=-1)
            frame_contract_valid = (
                frame_contract_valid & motion_finite
            )
            safe_motion = torch.where(
                frame_contract_valid.unsqueeze(-1),
                motion_raw,
                torch.zeros_like(motion_raw),
            )

        if self.mode == "index_only":
            if self.history_length == 1:
                slot_value = torch.zeros(
                    1,
                    device=return_valid.device,
                    dtype=compute_dtype,
                )
            else:
                slot_value = torch.linspace(
                    1.0,
                    0.0,
                    self.history_length,
                    device=return_valid.device,
                    dtype=compute_dtype,
                )
            base_age = slot_value.view(1, history_length, 1, 1).expand(
                batch_size,
                -1,
                *self.input_spatial_size,
            )
            age_contract_valid = torch.ones_like(
                return_valid,
                dtype=torch.bool,
            )
            already_bounded = True
        elif self.mode == "packet_age":
            base_age = packet_age[:, :, None, None].expand(
                -1,
                -1,
                *self.input_spatial_size,
            )
            age_contract_valid = packet_age_valid[
                :, :, None, None
            ].expand_as(return_valid)
            already_bounded = False
        else:
            # Eager validation makes this non-optional in per-return mode.  The
            # explicit branch also keeps TorchScript's Optional narrowing
            # local and auditable.
            if return_age_s is None:
                raise ValueError(
                    "per_return_age mode requires return_age_s."
                )
            return_age_raw = return_age_s.to(dtype=compute_dtype)
            packet_age_per_ray = packet_age[
                :, :, None, None
            ].expand_as(return_age_raw)
            age_contract_valid = (
                torch.isfinite(return_age_raw)
                & (return_age_raw >= 0.0)
                & (
                    return_age_raw + 1.0e-7
                    >= packet_age_per_ray
                )
                & packet_age_valid[:, :, None, None]
            )
            base_age = torch.where(
                age_contract_valid,
                return_age_raw,
                torch.zeros_like(return_age_raw),
            )
            already_bounded = False

        effective_return_valid = (
            return_valid
            & frame_contract_valid[:, :, None, None]
            & age_contract_valid
        )
        valid_float = effective_return_valid.to(dtype=compute_dtype)
        flattened_valid = valid_float.flatten(0, 1).unsqueeze(1)
        return_count = (
            F.avg_pool2d(
                flattened_valid,
                kernel_size=self.pool_kernel,
                stride=self.pool_kernel,
            )
            * float(self.rays_per_token)
        )
        valid_fraction = return_count / float(self.rays_per_token)
        token_valid = return_count > 0.0
        safe_age = torch.where(
            effective_return_valid,
            base_age,
            torch.zeros_like(base_age),
        )
        flattened_age = safe_age.flatten(0, 1).unsqueeze(1)
        age_sum = (
            F.avg_pool2d(
                flattened_age,
                kernel_size=self.pool_kernel,
                stride=self.pool_kernel,
            )
            * float(self.rays_per_token)
        )
        mean_age = age_sum / return_count.clamp_min(1.0)
        max_age = F.max_pool2d(
            flattened_age,
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
        )
        # The detached global maximum is greater than or equal to every valid
        # age and remains representable in the input dtype.  It therefore acts
        # as a minimum-pooling sentinel without TorchScript-unsupported
        # ``torch.finfo`` or non-finite values.
        minimum_sentinel = safe_age.detach().amax(
            dim=(0, 1, 2, 3),
        )
        min_input = torch.where(
            flattened_valid > 0.5,
            flattened_age,
            minimum_sentinel,
        )
        min_age = -F.max_pool2d(
            -min_input,
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
        )
        mean_age = torch.where(token_valid, mean_age, torch.zeros_like(mean_age))
        min_age = torch.where(token_valid, min_age, torch.zeros_like(min_age))
        max_age = torch.where(token_valid, max_age, torch.zeros_like(max_age))
        age_span = max_age - min_age

        if already_bounded:
            bounded_mean = mean_age
            bounded_min = min_age
            bounded_max = max_age
            bounded_span = age_span
            bounded_ray_age = safe_age
        else:
            bounded_mean = mean_age / (
                mean_age + self.age_time_scale_s
            )
            bounded_min = min_age / (
                min_age + self.age_time_scale_s
            )
            bounded_max = max_age / (
                max_age + self.age_time_scale_s
            )
            # The bounded span is defined by bounded extrema instead of
            # separately saturating a duration, keeping it in [0,1].
            bounded_span = bounded_max - bounded_min
            bounded_ray_age = safe_age / (
                safe_age + self.age_time_scale_s
            )

        # [BT, 1, Ht, Wt] -> [B, T, Nt, 1].
        token_shape = (
            batch_size,
            history_length,
            self.num_tokens_per_frame,
            1,
        )
        mean_token = bounded_mean.reshape(token_shape)
        min_token = bounded_min.reshape(token_shape)
        max_token = bounded_max.reshape(token_shape)
        span_token = bounded_span.reshape(token_shape)
        fraction_token = valid_fraction.reshape(token_shape)

        frequency = self.age_frequencies.to(
            device=return_valid.device,
            dtype=compute_dtype,
        ).view(1, 1, 1, -1)
        mean_phase = mean_token * math.pi * frequency
        min_phase = min_token * math.pi * frequency
        max_phase = max_token * math.pi * frequency
        mean_sin, mean_cos = torch.sin(mean_phase), torch.cos(mean_phase)
        min_sin, min_cos = torch.sin(min_phase), torch.cos(min_phase)
        max_sin, max_cos = torch.sin(max_phase), torch.cos(max_phase)
        time_features = torch.cat(
            (
                mean_token,
                min_token,
                max_token,
                span_token,
                fraction_token,
                mean_sin,
                mean_cos,
                min_sin,
                min_cos,
                max_sin,
                max_cos,
            ),
            dim=-1,
        )
        summary_encoding = self.time_projection(time_features)

        ray_valid = valid_float.unsqueeze(-1)
        ray_time_features = torch.cat(
            (
                bounded_ray_age.unsqueeze(-1),
                ray_valid,
            ),
            dim=-1,
        ) * ray_valid
        ray_time_features = (
            ray_time_features.flatten(0, 1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        spatial_encoding = self.ray_time_projection(
            ray_time_features
        )
        spatial_encoding = (
            spatial_encoding.permute(0, 2, 3, 1)
            .reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
                self.token_dim,
            )
        )
        event_encoding = summary_encoding + spatial_encoding

        if self.motion_projection is not None:
            if safe_motion is None:
                raise ValueError(
                    "motion_delta_dim > 0 requires frame_motion_delta."
                )
            motion_encoding = self.motion_projection(
                safe_motion
            )
            event_encoding = (
                event_encoding
                + motion_encoding[:, :, None, :]
            )
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            if not bool(torch.isfinite(event_encoding).all()):
                raise ValueError(
                    "Event-time projection produced non-finite values."
                )

        projection_valid = torch.isfinite(event_encoding).all(dim=-1)
        event_encoding = torch.where(
            torch.isfinite(event_encoding),
            event_encoding,
            torch.zeros_like(event_encoding),
        )
        final_token_valid = token_valid.reshape(
            batch_size,
            history_length,
            self.num_tokens_per_frame,
        ) & projection_valid
        event_encoding = torch.where(
            final_token_valid.unsqueeze(-1),
            event_encoding,
            torch.zeros_like(event_encoding),
        )
        if self.mode == "index_only":
            diagnostic_mean_age = torch.zeros_like(mean_age)
            diagnostic_min_age = torch.zeros_like(min_age)
            diagnostic_max_age = torch.zeros_like(max_age)
            diagnostic_age_span = torch.zeros_like(age_span)
            slot_coordinate = mean_age
            time_value_is_seconds = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=return_valid.device,
            )
            event_time_mode_id = 0
        else:
            diagnostic_mean_age = mean_age
            diagnostic_min_age = min_age
            diagnostic_max_age = max_age
            diagnostic_age_span = age_span
            slot_coordinate = torch.zeros_like(mean_age)
            time_value_is_seconds = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=return_valid.device,
            )
            event_time_mode_id = 1 if self.mode == "packet_age" else 2

        diagnostics = {
            "token_valid": final_token_valid,
            "return_count": return_count.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "valid_fraction": valid_fraction.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "mean_age_s": diagnostic_mean_age.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "min_age_s": diagnostic_min_age.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "max_age_s": diagnostic_max_age.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "age_span_s": diagnostic_age_span.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "slot_coordinate": slot_coordinate.reshape(
                batch_size,
                history_length,
                self.num_tokens_per_frame,
            ),
            "packet_age_s": packet_age,
            "frame_valid": frame_valid,
            "effective_frame_valid": frame_contract_valid,
            "time_value_is_seconds": time_value_is_seconds,
            "event_time_mode_id": torch.full(
                (batch_size,),
                event_time_mode_id,
                dtype=torch.long,
                device=return_valid.device,
            ),
        }
        return event_encoding, diagnostics

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError(
                f"mode must be a string, got {type(mode).__name__}."
            )
        resolved = mode.lower().replace("-", "_")
        if resolved not in SUPPORTED_EVENT_TIME_MODES:
            raise ValueError(
                "mode must be one of "
                f"{SUPPORTED_EVENT_TIME_MODES}, got {mode!r}."
            )
        return resolved

    @torch.jit.unused
    def _validate_inputs(
        self,
        return_valid: torch.Tensor,
        packet_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
        return_age_s: torch.Tensor | None,
        frame_motion_delta: torch.Tensor | None,
    ) -> None:
        expected_return_shape = (
            return_valid.shape[0] if return_valid.ndim > 0 else -1,
            self.history_length,
            *self.input_spatial_size,
        )
        if return_valid.ndim != 4 or tuple(return_valid.shape) != (
            expected_return_shape
        ):
            raise ValueError(
                "return_valid must have shape [B,T,H,W] with "
                f"[T,H,W]={(self.history_length, *self.input_spatial_size)}, "
                f"got {tuple(return_valid.shape)}."
            )
        if return_valid.shape[0] <= 0:
            raise ValueError("return_valid batch dimension must be positive.")
        if return_valid.dtype != torch.bool:
            raise ValueError(
                f"return_valid must be boolean, got {return_valid.dtype}."
            )
        batch_size = return_valid.shape[0]
        expected_frame_shape = (batch_size, self.history_length)
        if tuple(packet_age_s.shape) != expected_frame_shape:
            raise ValueError(
                "packet_age_s must have shape [B,T] with "
                f"{expected_frame_shape}, got {tuple(packet_age_s.shape)}."
            )
        if tuple(frame_valid.shape) != expected_frame_shape:
            raise ValueError(
                "frame_valid must have shape [B,T] with "
                f"{expected_frame_shape}, got {tuple(frame_valid.shape)}."
            )
        if frame_valid.dtype != torch.bool:
            raise ValueError(
                f"frame_valid must be boolean, got {frame_valid.dtype}."
            )
        if not packet_age_s.is_floating_point():
            raise ValueError("packet_age_s must be floating-point.")
        if not bool(torch.isfinite(packet_age_s).all()):
            raise ValueError("packet_age_s must contain only finite values.")
        if bool((packet_age_s < 0.0).any()):
            raise ValueError("packet_age_s must be non-negative.")
        if bool((packet_age_s[~frame_valid] != 0.0).any()):
            raise ValueError(
                "Invalid history frames must have exactly zero packet age."
            )
        if bool((return_valid & ~frame_valid[:, :, None, None]).any()):
            raise ValueError(
                "Successful returns cannot belong to an invalid frame."
            )

        tensors: list[tuple[str, torch.Tensor]] = [
            ("return_valid", return_valid),
            ("packet_age_s", packet_age_s),
            ("frame_valid", frame_valid),
        ]
        if return_age_s is not None:
            if tuple(return_age_s.shape) != tuple(return_valid.shape):
                raise ValueError(
                    "return_age_s must match return_valid shape, got "
                    f"{tuple(return_age_s.shape)} and "
                    f"{tuple(return_valid.shape)}."
                )
            if not return_age_s.is_floating_point():
                raise ValueError("return_age_s must be floating-point.")
            if not bool(torch.isfinite(return_age_s).all()):
                raise ValueError(
                    "return_age_s must contain only finite values."
                )
            if bool((return_age_s < 0.0).any()):
                raise ValueError("return_age_s must be non-negative.")
            if bool((return_age_s[~return_valid] != 0.0).any()):
                raise ValueError(
                    "Cells without a successful return must have exactly "
                    "zero return age."
                )
            packet_age_per_cell = packet_age_s[:, :, None, None]
            if bool(
                (
                    return_age_s[return_valid]
                    + 1.0e-7
                    < packet_age_per_cell.expand_as(return_age_s)[return_valid]
                ).any()
            ):
                raise ValueError(
                    "A return age cannot be younger than its packet "
                    "capture-end age."
                )
            tensors.append(("return_age_s", return_age_s))
        elif self.mode == "per_return_age":
            raise ValueError("per_return_age mode requires return_age_s.")

        if self.motion_delta_dim > 0:
            expected_motion_shape = (
                batch_size,
                self.history_length,
                self.motion_delta_dim,
            )
            if frame_motion_delta is None:
                raise ValueError(
                    "motion_delta_dim > 0 requires frame_motion_delta."
                )
            if tuple(frame_motion_delta.shape) != expected_motion_shape:
                raise ValueError(
                    "frame_motion_delta must have shape [B,T,M] with "
                    f"{expected_motion_shape}, got "
                    f"{tuple(frame_motion_delta.shape)}."
                )
            if not frame_motion_delta.is_floating_point():
                raise ValueError(
                    "frame_motion_delta must be floating-point."
                )
            if not bool(torch.isfinite(frame_motion_delta).all()):
                raise ValueError(
                    "frame_motion_delta must contain only finite values."
                )
            if bool(
                (
                    frame_motion_delta[~frame_valid]
                    != 0.0
                ).any()
            ):
                raise ValueError(
                    "Invalid history frames must have exactly zero motion "
                    "delta."
                )
            tensors.append(("frame_motion_delta", frame_motion_delta))
        elif frame_motion_delta is not None:
            raise ValueError(
                "frame_motion_delta is only valid when motion_delta_dim > 0."
            )

        devices = {tensor.device for _, tensor in tensors}
        if len(devices) != 1:
            detail = ", ".join(
                f"{name}={tensor.device}" for name, tensor in tensors
            )
            raise ValueError(f"All inputs must share one device, got {detail}.")
        parameter_device = next(self.parameters()).device
        if return_valid.device != parameter_device:
            raise ValueError(
                "Inputs and module parameters must share one device, got "
                f"inputs={return_valid.device}, parameters={parameter_device}."
            )


__all__ = [
    "SUPPORTED_EVENT_TIME_MODES",
    "RayReturnEventTimeEncoder",
]
