# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack authenticated range/event-time tensors into the actor contract.

Channel order is fixed as ``range, return_valid, return_age, packet_age,
frame_valid``. Packet-age and frame-valid are spatially broadcast so the
manager observation remains one dense ``[K,C,H,W]`` tensor. The duplicated
values are checked by the actor and deployment receipt.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch

from .mid360_ray_time_builder import Mid360AlignedRayTimeHistory


RAY_EVENT_CHANNELS: Final[tuple[str, ...]] = (
    "range_m",
    "return_valid",
    "return_age_s",
    "packet_age_s",
    "frame_valid",
)
RAY_EVENT_SOURCES: Final[tuple[str, ...]] = (
    "raycaster_packet",
    "raycaster_quantized_event",
    "livox_per_return",
)
RAY_EVENT_TEMPORAL_BASELINES: Final[tuple[str, ...]] = (
    "per_return_age",
    "quantized_event_age",
    "packet_age",
    "age_zero",
)


def pack_ray_event_observation(
    range_m: torch.Tensor,
    return_valid: torch.Tensor,
    return_age_s: torch.Tensor,
    packet_age_s: torch.Tensor,
    frame_valid: torch.Tensor,
    *,
    source: str,
    temporal_baseline: str,
) -> torch.Tensor:
    """Return a strict float32 ``[B,K,5,H,W]`` observation tensor."""
    source = _choice("source", source, RAY_EVENT_SOURCES)
    temporal_baseline = _choice(
        "temporal_baseline",
        temporal_baseline,
        RAY_EVENT_TEMPORAL_BASELINES,
    )
    if temporal_baseline == "per_return_age" and source != "livox_per_return":
        raise ValueError(
            "per_return_age requires authenticated Livox per-return timestamps; "
            "RayCaster only provides packet-level capture time."
        )
    if (
        temporal_baseline == "quantized_event_age"
        and source != "raycaster_quantized_event"
    ):
        raise ValueError(
            "quantized_event_age requires timestamped RayCaster packet events."
        )
    if (
        source == "raycaster_quantized_event"
        and temporal_baseline != "quantized_event_age"
    ):
        raise ValueError(
            "raycaster_quantized_event is reserved for quantized_event_age."
        )
    if range_m.ndim != 4:
        raise ValueError("range_m must have shape [B,K,H,W].")
    if tuple(return_valid.shape) != tuple(range_m.shape):
        raise ValueError("return_valid must match range_m shape.")
    if tuple(return_age_s.shape) != tuple(range_m.shape):
        raise ValueError("return_age_s must match range_m shape.")
    if return_valid.dtype != torch.bool:
        raise ValueError("return_valid must be boolean.")
    expected_frame = tuple(range_m.shape[:2])
    if tuple(packet_age_s.shape) != expected_frame:
        raise ValueError(f"packet_age_s must have shape {expected_frame}.")
    if tuple(frame_valid.shape) != expected_frame or frame_valid.dtype != torch.bool:
        raise ValueError(f"frame_valid must be boolean with shape {expected_frame}.")
    tensors = (range_m, return_valid, return_age_s, packet_age_s, frame_valid)
    if any(tensor.device != range_m.device for tensor in tensors):
        raise ValueError("All ray-event tensors must share one device.")
    if not range_m.is_floating_point() or not return_age_s.is_floating_point():
        raise ValueError("Range and return age must be floating point.")
    if not packet_age_s.is_floating_point():
        raise ValueError("Packet age must be floating point.")
    if not bool(torch.isfinite(range_m).all()):
        raise ValueError("Range must be finite.")
    if not bool(torch.isfinite(return_age_s).all()):
        raise ValueError("Return age must be finite.")
    if not bool(torch.isfinite(packet_age_s).all()):
        raise ValueError("Packet age must be finite.")
    if bool((return_valid & (range_m <= 0)).any()):
        raise ValueError("Valid returns require positive range.")
    if bool((~return_valid & (range_m != 0)).any()):
        raise ValueError("Invalid ranges must be exactly zero.")
    if bool((~return_valid & (return_age_s != 0)).any()):
        raise ValueError("Invalid return ages must be exactly zero.")
    if bool((return_age_s < 0).any()) or bool((packet_age_s < 0).any()):
        raise ValueError("Acquisition ages must be non-negative.")
    if bool((return_valid & ~frame_valid[:, :, None, None]).any()):
        raise ValueError("Invalid history frames cannot contain returns.")
    if bool(
        (
            return_valid
            & (return_age_s + 1.0e-7 < packet_age_s[:, :, None, None])
        ).any()
    ):
        raise ValueError("Return age must be >= packet age.")

    range_channel = range_m.to(torch.float32)
    valid_channel = return_valid.to(torch.float32)
    if temporal_baseline in ("per_return_age", "quantized_event_age"):
        return_age_channel = return_age_s.to(torch.float32)
        packet_age_channel = packet_age_s.to(torch.float32)
    elif temporal_baseline == "packet_age":
        return_age_channel = torch.zeros_like(range_channel)
        packet_age_channel = packet_age_s.to(torch.float32)
    else:
        return_age_channel = torch.zeros_like(range_channel)
        packet_age_channel = torch.zeros_like(packet_age_s, dtype=torch.float32)
    packet_age_map = packet_age_channel[:, :, None, None].expand_as(range_channel)
    frame_valid_map = frame_valid[:, :, None, None].expand_as(return_valid)
    return torch.stack(
        (
            range_channel,
            valid_channel,
            return_age_channel,
            packet_age_map,
            frame_valid_map.to(torch.float32),
        ),
        dim=2,
    )


def aligned_history_to_ray_event_observation(
    history: Mid360AlignedRayTimeHistory,
    *,
    temporal_baseline: str = "per_return_age",
) -> np.ndarray:
    """Pack a strict real MID-360 builder snapshot as ``[K,5,H,W]``."""
    ray_history = torch.from_numpy(history.ray_history.astype(np.float32, copy=False))
    packed = pack_ray_event_observation(
        ray_history[:, 0].unsqueeze(0),
        torch.from_numpy(history.return_valid).unsqueeze(0),
        torch.from_numpy(history.return_age_s.astype(np.float32, copy=False)).unsqueeze(0),
        torch.from_numpy(history.packet_age_s.astype(np.float32, copy=False)).unsqueeze(0),
        torch.from_numpy(history.frame_valid).unsqueeze(0),
        source="livox_per_return",
        temporal_baseline=temporal_baseline,
    )
    return packed.squeeze(0).numpy()


def pack_acquisition_delta_proprio_observation(
    acquisition_delta_proprio: torch.Tensor,
    return_valid: torch.Tensor,
    range_age_winner_id: torch.Tensor,
    delta_winner_id: torch.Tensor,
) -> torch.Tensor:
    """Pack a strict same-winner ``[B,K,D,H,W]`` acquisition-state tensor.

    Stable winner ids are proof inputs and are not exposed to the actor. Their
    equality closes the builder boundary that the actor cannot infer from
    range/age/delta values alone.
    """
    if acquisition_delta_proprio.ndim != 5:
        raise ValueError(
            "acquisition_delta_proprio must have shape [B,K,D,H,W]."
        )
    if acquisition_delta_proprio.shape[2] <= 0:
        raise ValueError("Acquisition delta-proprio dimension must be positive.")
    expected_cell_shape = (
        acquisition_delta_proprio.shape[0],
        acquisition_delta_proprio.shape[1],
        acquisition_delta_proprio.shape[3],
        acquisition_delta_proprio.shape[4],
    )
    if tuple(return_valid.shape) != expected_cell_shape or return_valid.dtype != torch.bool:
        raise ValueError("return_valid must be boolean with shape [B,K,H,W].")
    for name, winner_id in (
        ("range_age_winner_id", range_age_winner_id),
        ("delta_winner_id", delta_winner_id),
    ):
        if tuple(winner_id.shape) != expected_cell_shape or winner_id.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long with shape [B,K,H,W].")
    tensors = (
        return_valid,
        range_age_winner_id,
        delta_winner_id,
    )
    if any(tensor.device != acquisition_delta_proprio.device for tensor in tensors):
        raise ValueError("Delta-proprio and winner tensors must share one device.")
    if not acquisition_delta_proprio.is_floating_point():
        raise ValueError("acquisition_delta_proprio must be floating point.")
    if not bool(torch.isfinite(acquisition_delta_proprio).all()):
        raise ValueError("acquisition_delta_proprio must be finite.")
    if not bool((range_age_winner_id == delta_winner_id).all()):
        raise ValueError(
            "Range/age and acquisition delta-proprio must use the same winner id."
        )
    if bool((return_valid & (range_age_winner_id < 0)).any()):
        raise ValueError("Valid returns require a non-negative stable winner id.")
    if bool((~return_valid & (range_age_winner_id != -1)).any()):
        raise ValueError("Invalid returns require winner id -1.")
    if bool(
        (
            ~return_valid[:, :, None]
            & (acquisition_delta_proprio != 0.0)
        ).any()
    ):
        raise ValueError(
            "Invalid returns must carry exactly zero acquisition delta-proprio."
        )
    return acquisition_delta_proprio.to(torch.float32)


def _choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    resolved = value.lower().replace("-", "_")
    if resolved not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")
    return resolved


__all__ = [
    "RAY_EVENT_CHANNELS",
    "RAY_EVENT_SOURCES",
    "RAY_EVENT_TEMPORAL_BASELINES",
    "aligned_history_to_ray_event_observation",
    "pack_acquisition_delta_proprio_observation",
    "pack_ray_event_observation",
]
