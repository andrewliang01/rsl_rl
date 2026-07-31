# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Packetization-invariant latest-event reduction for raw LiDAR returns.

This module deliberately operates *before* packet rasterization.  Every raw
return carries a stable ``event_id``, action-time age, and acquisition-time
delta-proprio. For each angular cell, PIES keeps the newest valid event in a
fixed causal window. Exact-time
ties use nearest range and then the stable event id, so changing packet
boundaries or packet order cannot change the raster.

The current IsaacLab RayCaster path cannot provide these raw events.  This
utility is therefore a conformance primitive and proof fixture, not permission
to relabel packet timestamps as per-return timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Final, Sequence

import torch


PIES_EVENT_WINDOW_S: Final[float] = 0.5


@dataclass(frozen=True)
class RawRayEventPacket:
    """One partition; scalar fields are ``[B,N]``, delta-proprio is ``[B,N,D]``."""

    cell_index: torch.Tensor
    range_m: torch.Tensor
    return_age_s: torch.Tensor
    event_id: torch.Tensor
    return_valid: torch.Tensor
    acquisition_delta_proprio: torch.Tensor


@dataclass(frozen=True)
class LatestEventRaster:
    """Same-winner output; delta-proprio uses ``[B,D,H,W]``."""

    range_m: torch.Tensor
    return_valid: torch.Tensor
    return_age_s: torch.Tensor
    event_id: torch.Tensor
    acquisition_delta_proprio: torch.Tensor


def reduce_raw_event_packets_to_latest_raster(
    packets: Sequence[RawRayEventPacket],
    *,
    spatial_size: tuple[int, int],
    event_window_s: float = PIES_EVENT_WINDOW_S,
) -> LatestEventRaster:
    """Reduce an event set independently of its packet partition.

    Selection is lexicographic per angular cell: minimum age (newest), minimum
    range for exact-age ties, then minimum globally stable event id.  Invalid
    events and events outside ``[0,event_window_s]`` are removed observations;
    they are not emitted as no-return evidence.
    """
    if not packets:
        raise ValueError("PIES requires at least one raw-event packet.")
    height, width = spatial_size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("spatial_size must contain two positive integers.")
    if (
        isinstance(event_window_s, bool)
        or not isinstance(event_window_s, (int, float))
        or not math.isfinite(float(event_window_s))
        or float(event_window_s) <= 0.0
    ):
        raise ValueError("event_window_s must be finite and positive.")

    reference = packets[0].cell_index
    if reference.ndim != 2:
        raise ValueError("Raw event tensors must have shape [B,N].")
    batch_size = reference.shape[0]
    device = reference.device
    delta_dim = packets[0].acquisition_delta_proprio.shape[-1]
    if delta_dim <= 0:
        raise ValueError("acquisition_delta_proprio must have positive width.")
    fields: dict[str, list[torch.Tensor]] = {
        "cell_index": [],
        "range_m": [],
        "return_age_s": [],
        "event_id": [],
        "return_valid": [],
    }
    for packet in packets:
        packet_fields = {
            "cell_index": packet.cell_index,
            "range_m": packet.range_m,
            "return_age_s": packet.return_age_s,
            "event_id": packet.event_id,
            "return_valid": packet.return_valid,
        }
        shape = packet.cell_index.shape
        if packet.cell_index.ndim != 2 or shape[0] != batch_size:
            raise ValueError("All raw event packets must share [B,N] batch shape.")
        for name, tensor in packet_fields.items():
            if tensor.shape != shape or tensor.device != device:
                raise ValueError(
                    f"{name} must match its packet cell_index shape/device."
                )
            fields[name].append(tensor)
        if (
            packet.acquisition_delta_proprio.shape
            != (*shape, delta_dim)
            or packet.acquisition_delta_proprio.device != device
            or not packet.acquisition_delta_proprio.is_floating_point()
        ):
            raise ValueError(
                "acquisition_delta_proprio must share [B,N,D], device, and D."
            )

    cell_index = torch.cat(fields["cell_index"], dim=1)
    range_m = torch.cat(fields["range_m"], dim=1).to(torch.float32)
    return_age_s = torch.cat(fields["return_age_s"], dim=1).to(torch.float32)
    event_id = torch.cat(fields["event_id"], dim=1)
    return_valid = torch.cat(fields["return_valid"], dim=1)
    acquisition_delta_proprio = torch.cat(
        [packet.acquisition_delta_proprio for packet in packets], dim=1
    ).to(torch.float32)
    if cell_index.dtype != torch.long or event_id.dtype != torch.long:
        raise ValueError("cell_index and event_id must be torch.long.")
    if return_valid.dtype != torch.bool:
        raise ValueError("return_valid must be torch.bool.")
    num_cells = height * width
    contract_valid = return_valid
    if bool(
        (
            contract_valid
            & (
                (cell_index < 0)
                | (cell_index >= num_cells)
                | ~torch.isfinite(range_m)
                | (range_m <= 0.0)
                | ~torch.isfinite(return_age_s)
                | (return_age_s < 0.0)
                | (event_id < 0)
            )
        ).any()
    ):
        raise ValueError("Valid raw events violate cell/range/age/id contract.")
    if bool(
        (
            contract_valid[:, :, None]
            & ~torch.isfinite(acquisition_delta_proprio)
        ).any()
    ):
        raise ValueError("Valid acquisition delta-proprio must be finite.")
    selected = contract_valid & (return_age_s <= float(event_window_s))
    for batch_index in range(batch_size):
        valid_ids = event_id[batch_index, contract_valid[batch_index]]
        if torch.unique(valid_ids).numel() != valid_ids.numel():
            raise ValueError("Valid raw events require globally stable unique ids.")

    output_range = torch.zeros(
        batch_size, num_cells, device=device, dtype=torch.float32
    )
    output_age = torch.zeros_like(output_range)
    output_valid = torch.zeros(
        batch_size, num_cells, device=device, dtype=torch.bool
    )
    output_id = torch.full(
        (batch_size, num_cells), -1, device=device, dtype=torch.long
    )
    output_delta = torch.zeros(
        batch_size,
        delta_dim,
        num_cells,
        device=device,
        dtype=torch.float32,
    )
    # The cell loop is intentional: it gives the proof fixture an explicit,
    # auditable lexicographic reduction independent of concatenation order.
    for cell in range(num_cells):
        candidates = selected & (cell_index == cell)
        newest_age = torch.where(
            candidates, return_age_s, torch.full_like(return_age_s, torch.inf)
        ).amin(dim=1)
        age_ties = candidates & (return_age_s == newest_age[:, None])
        nearest_range = torch.where(
            age_ties, range_m, torch.full_like(range_m, torch.inf)
        ).amin(dim=1)
        range_ties = age_ties & (range_m == nearest_range[:, None])
        winner_id = torch.where(
            range_ties,
            event_id,
            torch.full_like(event_id, torch.iinfo(torch.long).max),
        ).amin(dim=1)
        winner = range_ties & (event_id == winner_id[:, None])
        cell_valid = winner.any(dim=1)
        winner_range = torch.where(winner, range_m, torch.zeros_like(range_m)).amax(
            dim=1
        )
        output_valid[:, cell] = cell_valid
        output_range[:, cell] = torch.where(
            cell_valid, winner_range, torch.zeros_like(winner_range)
        )
        output_age[:, cell] = torch.where(
            cell_valid, newest_age, torch.zeros_like(newest_age)
        )
        output_id[:, cell] = torch.where(
            cell_valid, winner_id, torch.full_like(winner_id, -1)
        )
        event_index = torch.arange(
            event_id.shape[1], device=device, dtype=torch.long
        )[None]
        winner_index = torch.where(
            winner,
            event_index,
            torch.full_like(event_index, event_id.shape[1]),
        ).amin(dim=1)
        safe_winner_index = winner_index.clamp(0, event_id.shape[1] - 1)
        winner_delta = torch.gather(
            acquisition_delta_proprio,
            1,
            safe_winner_index[:, None, None].expand(-1, 1, delta_dim),
        ).squeeze(1)
        output_delta[:, :, cell] = torch.where(
            cell_valid[:, None], winner_delta, torch.zeros_like(winner_delta)
        )

    raster_shape = (batch_size, 1, height, width)
    return LatestEventRaster(
        range_m=output_range.reshape(raster_shape),
        return_valid=output_valid.reshape(raster_shape),
        return_age_s=output_age.reshape(raster_shape),
        event_id=output_id.reshape(raster_shape),
        acquisition_delta_proprio=output_delta.reshape(
            batch_size, delta_dim, height, width
        ),
    )


def latest_event_raster_sha256(raster: LatestEventRaster) -> str:
    """Hash shape, dtype, and bitwise CPU bytes of a PIES raster."""
    digest = hashlib.sha256()
    for name in (
        "range_m",
        "return_valid",
        "return_age_s",
        "event_id",
        "acquisition_delta_proprio",
    ):
        tensor = getattr(raster, name).detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "LatestEventRaster",
    "PIES_EVENT_WINDOW_S",
    "RawRayEventPacket",
    "latest_event_raster_sha256",
    "reduce_raw_event_packets_to_latest_raster",
]
