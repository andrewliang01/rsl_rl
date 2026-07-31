# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Causal H2 geometry/time interventions with aligned range-age winners.

The router never fabricates a rerendered observation.  ``geometry='rerender'``
requires the caller to provide a second, independently receipted range, valid,
and per-return-age raster.  This matters because moving a return to a new
spherical cell without moving its timestamp would invalidate the H2 test.

The exact-union K1 control chooses one return per angular cell over all history
frames.  Nearest range wins; an equal-range tie chooses the earliest acquired
return (largest action-time age).  Range and age are gathered from the same
winner.  No-return cells stay zero and invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import torch
import torch.nn as nn


SUPPORTED_GEOMETRY_VARIANTS: Final[tuple[str, ...]] = ("native", "rerender")
SUPPORTED_TIME_ASSOCIATIONS: Final[tuple[str, ...]] = ("correct", "shuffled")
SUPPORTED_TEMPORAL_BASELINES: Final[tuple[str, ...]] = (
    "per_return_age",
    "packet_age",
    "age_zero",
)
SUPPORTED_HISTORY_REDUCTIONS: Final[tuple[str, ...]] = (
    "history",
    "exact_union_k1",
    "raster_latest_event_prototype",
)
LATEST_EVENT_WINDOW_S: Final[float] = 0.5


@dataclass(frozen=True)
class RayEventAblationOutput:
    """Aligned H2 tensors after one declared intervention."""

    range_m: torch.Tensor
    return_valid: torch.Tensor
    return_age_s: torch.Tensor
    packet_age_s: torch.Tensor
    frame_valid: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class RayEventAblationRouter(nn.Module):
    """Apply native/rerender, correct/shuffled, and time-baseline controls."""

    def __init__(
        self,
        *,
        geometry: str = "native",
        time_association: str = "correct",
        temporal_baseline: str = "per_return_age",
        history_reduction: str = "history",
        shuffle_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.geometry = self._choice(
            "geometry", geometry, SUPPORTED_GEOMETRY_VARIANTS
        )
        self.time_association = self._choice(
            "time_association",
            time_association,
            SUPPORTED_TIME_ASSOCIATIONS,
        )
        self.temporal_baseline = self._choice(
            "temporal_baseline",
            temporal_baseline,
            SUPPORTED_TEMPORAL_BASELINES,
        )
        self.history_reduction = self._choice(
            "history_reduction",
            history_reduction,
            SUPPORTED_HISTORY_REDUCTIONS,
        )
        if self.time_association == "shuffled":
            if (
                isinstance(shuffle_seed, bool)
                or not isinstance(shuffle_seed, Integral)
                or int(shuffle_seed) < 0
            ):
                raise ValueError(
                    "time_association='shuffled' requires a non-negative "
                    "integer shuffle_seed."
                )
            self.shuffle_seed = int(shuffle_seed)
        else:
            if shuffle_seed is not None:
                raise ValueError(
                    "shuffle_seed is accepted only for shuffled timing."
                )
            self.shuffle_seed = None

    def forward(
        self,
        native_range_m: torch.Tensor,
        native_return_valid: torch.Tensor,
        native_return_age_s: torch.Tensor,
        packet_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
        *,
        rerender_range_m: torch.Tensor | None = None,
        rerender_return_valid: torch.Tensor | None = None,
        rerender_return_age_s: torch.Tensor | None = None,
    ) -> RayEventAblationOutput:
        """Return intervention tensors; every raster is ``[B,T,H,W]``."""
        self._validate_primary(
            native_range_m,
            native_return_valid,
            native_return_age_s,
            packet_age_s,
            frame_valid,
        )
        if self.geometry == "native":
            if any(
                value is not None
                for value in (
                    rerender_range_m,
                    rerender_return_valid,
                    rerender_return_age_s,
                )
            ):
                raise ValueError(
                    "Native geometry rejects unused rerender tensors so an "
                    "experiment cannot silently carry an unaudited second path."
                )
            range_m = native_range_m
            return_valid = native_return_valid
            return_age_s = native_return_age_s
        else:
            if any(
                value is None
                for value in (
                    rerender_range_m,
                    rerender_return_valid,
                    rerender_return_age_s,
                )
            ):
                raise ValueError(
                    "Rerender geometry requires aligned range, valid, and "
                    "per-return-age rasters."
                )
            self._validate_geometry_triplet(
                rerender_range_m,  # type: ignore[arg-type]
                rerender_return_valid,  # type: ignore[arg-type]
                rerender_return_age_s,  # type: ignore[arg-type]
                native_range_m.shape,
                native_range_m.device,
            )
            range_m = rerender_range_m  # type: ignore[assignment]
            return_valid = rerender_return_valid  # type: ignore[assignment]
            return_age_s = rerender_return_age_s  # type: ignore[assignment]

        if bool((return_valid & ~frame_valid[:, :, None, None]).any()):
            raise ValueError(
                "The selected geometry contains a return in a frame_valid=False slot."
            )
        if bool(
            (
                return_valid
                & (return_age_s < packet_age_s[:, :, None, None])
            ).any()
        ):
            raise ValueError(
                "Every selected valid return age must be >= its packet age."
            )

        range_m = range_m.to(dtype=torch.float32)
        return_age_s = return_age_s.to(dtype=torch.float32)
        packet_age_s = packet_age_s.to(dtype=torch.float32)
        range_m = torch.where(return_valid, range_m, torch.zeros_like(range_m))
        return_age_s = torch.where(
            return_valid, return_age_s, torch.zeros_like(return_age_s)
        )

        original_age = return_age_s
        if self.time_association == "shuffled":
            return_age_s = self._shuffle_valid_ages(
                return_age_s,
                return_valid,
                self.shuffle_seed,
            )
        shuffled_multiset_conserved = self._age_multiset_equal(
            original_age, return_age_s, return_valid
        )
        changed_age_association_count = (
            return_valid & (original_age != return_age_s)
        ).sum(dim=(-2, -1))

        original_history_length = range_m.shape[1]
        collision_valid = return_valid
        if self.history_reduction == "raster_latest_event_prototype":
            collision_valid = collision_valid & (
                return_age_s <= LATEST_EVENT_WINDOW_S
            )
        collision_count = (collision_valid.sum(dim=1) > 1).sum(
            dim=(-2, -1)
        )
        winner_frame_index = torch.full(
            (range_m.shape[0], *range_m.shape[-2:]),
            -1,
            device=range_m.device,
            dtype=torch.long,
        )
        if self.history_reduction == "exact_union_k1":
            (
                range_m,
                return_valid,
                return_age_s,
                packet_age_s,
                frame_valid,
                winner_frame_index,
            ) = self._exact_union_k1(
                range_m,
                return_valid,
                return_age_s,
                frame_valid,
            )
        elif self.history_reduction == "raster_latest_event_prototype":
            (
                range_m,
                return_valid,
                return_age_s,
                packet_age_s,
                frame_valid,
                winner_frame_index,
            ) = self._raster_latest_event_prototype(
                range_m,
                return_valid,
                return_age_s,
                frame_valid,
            )

        if self.temporal_baseline == "packet_age":
            return_age_s = packet_age_s[:, :, None, None].expand_as(
                return_age_s
            )
            return_age_s = torch.where(
                return_valid, return_age_s, torch.zeros_like(return_age_s)
            )
        elif self.temporal_baseline == "age_zero":
            return_age_s = torch.zeros_like(return_age_s)
            packet_age_s = torch.zeros_like(packet_age_s)

        valid_count = return_valid.sum(dim=(-2, -1))
        age_sum = return_age_s.sum(dim=(-2, -1))
        age_mean = age_sum / valid_count.clamp_min(1).to(return_age_s.dtype)
        age_min = torch.where(
            return_valid,
            return_age_s,
            torch.full_like(return_age_s, torch.inf),
        ).amin(dim=(-2, -1))
        age_max = torch.where(
            return_valid,
            return_age_s,
            torch.full_like(return_age_s, -torch.inf),
        ).amax(dim=(-2, -1))
        age_min = torch.where(valid_count > 0, age_min, torch.zeros_like(age_min))
        age_max = torch.where(valid_count > 0, age_max, torch.zeros_like(age_max))
        diagnostics = {
            "input_history_length": torch.full(
                (range_m.shape[0],),
                original_history_length,
                device=range_m.device,
                dtype=torch.long,
            ),
            "output_history_length": torch.full(
                (range_m.shape[0],),
                range_m.shape[1],
                device=range_m.device,
                dtype=torch.long,
            ),
            "valid_return_count_per_frame": valid_count,
            "return_age_mean_s": age_mean,
            "return_age_min_s": age_min,
            "return_age_max_s": age_max,
            "return_age_span_s": age_max - age_min,
            "shuffled_multiset_conserved": shuffled_multiset_conserved,
            "changed_age_association_count": changed_age_association_count,
            "exact_union_collision_cell_count": collision_count,
            "exact_union_winner_frame_index": winner_frame_index,
            "history_reduction_winner_frame_index": winner_frame_index,
            "shuffle_seed": torch.full(
                (range_m.shape[0],),
                -1 if self.shuffle_seed is None else self.shuffle_seed,
                device=range_m.device,
                dtype=torch.long,
            ),
        }
        return RayEventAblationOutput(
            range_m=range_m,
            return_valid=return_valid,
            return_age_s=return_age_s,
            packet_age_s=packet_age_s,
            frame_valid=frame_valid,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _raster_latest_event_prototype(
        range_m: torch.Tensor,
        return_valid: torch.Tensor,
        return_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build a raster latest-event prototype from the newest cell event.

        Inputs are already packet-rasterized, so this operation is invariant
        to empty-frame padding but not yet proven invariant to arbitrary raw
        event repartitioning when multiple returns collide in one packet cell.
        It must not be named PIES or promoted to training.
        """
        return_valid = return_valid & (return_age_s <= LATEST_EVENT_WINDOW_S)
        masked_age = torch.where(
            return_valid,
            return_age_s,
            torch.full_like(return_age_s, torch.inf),
        )
        newest_age = masked_age.amin(dim=1)
        union_valid = return_valid.any(dim=1)
        age_ties = return_valid & (masked_age == newest_age[:, None])
        tie_range = torch.where(
            age_ties,
            range_m,
            torch.full_like(range_m, torch.inf),
        )
        nearest_tied_range = tie_range.amin(dim=1)
        winner_candidates = age_ties & (
            range_m == nearest_tied_range[:, None]
        )
        history_index = torch.arange(
            range_m.shape[1], device=range_m.device
        )[None, :, None, None]
        winner_index = torch.where(
            winner_candidates,
            history_index,
            torch.full_like(history_index, range_m.shape[1]),
        ).amin(dim=1)
        safe_index = winner_index.clamp(min=0, max=range_m.shape[1] - 1)
        union_range = torch.gather(
            range_m,
            1,
            safe_index[:, None],
        ).squeeze(1)
        union_range = torch.where(
            union_valid, union_range, torch.zeros_like(union_range)
        )
        union_age = torch.where(
            union_valid, newest_age, torch.zeros_like(newest_age)
        )
        winner_index = torch.where(
            union_valid,
            winner_index,
            torch.full_like(winner_index, -1),
        )
        union_valid = union_valid[:, None]
        union_age = union_age[:, None]
        union_range = union_range[:, None]
        flat_valid = union_valid.flatten(start_dim=2)
        flat_age = union_age.flatten(start_dim=2)
        packet_age_floor = torch.where(
            flat_valid,
            flat_age,
            torch.full_like(flat_age, torch.inf),
        ).amin(dim=-1)
        packet_age_floor = torch.where(
            flat_valid.any(dim=-1),
            packet_age_floor,
            torch.zeros_like(packet_age_floor),
        )
        return (
            union_range,
            union_valid,
            union_age,
            packet_age_floor,
            frame_valid.any(dim=1, keepdim=True),
            winner_index,
        )

    @staticmethod
    def _exact_union_k1(
        range_m: torch.Tensor,
        return_valid: torch.Tensor,
        return_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        masked_range = torch.where(
            return_valid,
            range_m,
            torch.full_like(range_m, torch.inf),
        )
        nearest = masked_range.amin(dim=1)
        union_valid = return_valid.any(dim=1)
        nearest_ties = return_valid & (masked_range == nearest[:, None])
        tie_age = torch.where(
            nearest_ties,
            return_age_s,
            torch.full_like(return_age_s, -torch.inf),
        )
        # Equal range uses the earliest acquisition: greatest action-time age.
        winning_age = tie_age.amax(dim=1)
        age_ties = nearest_ties & (return_age_s == winning_age[:, None])
        history_index = torch.arange(
            range_m.shape[1], device=range_m.device
        )[None, :, None, None]
        # If timestamp also ties, the oldest (lowest) declared history index wins.
        winner_index = torch.where(
            age_ties,
            history_index,
            torch.full_like(history_index, range_m.shape[1]),
        ).amin(dim=1)
        winner_index = torch.where(
            union_valid,
            winner_index,
            torch.full_like(winner_index, -1),
        )
        union_range = torch.where(
            union_valid, nearest, torch.zeros_like(nearest)
        )[:, None]
        union_age = torch.where(
            union_valid, winning_age, torch.zeros_like(winning_age)
        )[:, None]
        union_valid = union_valid[:, None]
        union_frame_valid = frame_valid.any(dim=1, keepdim=True)
        # A synthetic exact union has no single physical packet.  Export the
        # freshest winning-return age as an explicit lower age bound, which is
        # compatible with the event-time encoder's return_age >= packet_age
        # invariant and avoids pretending all cells came from one packet.
        flat_valid = union_valid.flatten(start_dim=2)
        flat_age = union_age.flatten(start_dim=2)
        union_packet_age = torch.where(
            flat_valid,
            flat_age,
            torch.full_like(flat_age, torch.inf),
        ).amin(dim=-1)
        union_packet_age = torch.where(
            flat_valid.any(dim=-1),
            union_packet_age,
            torch.zeros_like(union_packet_age),
        )
        return (
            union_range,
            union_valid,
            union_age,
            union_packet_age,
            union_frame_valid,
            winner_index,
        )

    @staticmethod
    def _shuffle_valid_ages(
        age: torch.Tensor,
        valid: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device=age.device)
        generator.manual_seed(seed)
        flat_age = age.flatten(start_dim=2)
        flat_valid = valid.flatten(start_dim=2)
        priorities = torch.rand(
            flat_age.shape,
            device=age.device,
            dtype=torch.float32,
            generator=generator,
        )
        priorities = torch.where(
            flat_valid,
            priorities,
            torch.full_like(priorities, torch.inf),
        )
        permutation = torch.argsort(
            priorities, dim=-1, descending=False, stable=True
        )
        ordered_age = torch.gather(flat_age, -1, permutation)
        valid_count = flat_valid.sum(dim=-1)
        rank = torch.arange(flat_age.shape[-1], device=age.device)
        ordered_valid = rank[None, None, :] < valid_count[:, :, None]

        # Scatter the permuted valid-age prefix back into the original valid
        # cell locations in ascending cell-index order.
        destination_order = torch.argsort(
            (~flat_valid).to(torch.int64),
            dim=-1,
            descending=False,
            stable=True,
        )
        destination = torch.where(
            ordered_valid,
            destination_order,
            torch.full_like(destination_order, flat_age.shape[-1]),
        )
        shuffled_with_sentinel = torch.zeros(
            *flat_age.shape[:-1],
            flat_age.shape[-1] + 1,
            device=age.device,
            dtype=age.dtype,
        )
        shuffled_with_sentinel.scatter_(
            -1,
            destination,
            torch.where(ordered_valid, ordered_age, torch.zeros_like(ordered_age)),
        )
        return shuffled_with_sentinel[..., :-1].reshape_as(age)

    @staticmethod
    def _age_multiset_equal(
        before: torch.Tensor,
        after: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        before_masked = torch.where(
            valid, before, torch.full_like(before, torch.inf)
        ).flatten(start_dim=2)
        after_masked = torch.where(
            valid, after, torch.full_like(after, torch.inf)
        ).flatten(start_dim=2)
        return (
            torch.sort(before_masked, dim=-1, stable=True).values
            == torch.sort(after_masked, dim=-1, stable=True).values
        ).all(dim=-1)

    @classmethod
    def _validate_primary(
        cls,
        range_m: torch.Tensor,
        valid: torch.Tensor,
        return_age_s: torch.Tensor,
        packet_age_s: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> None:
        cls._validate_geometry_triplet(
            range_m, valid, return_age_s, range_m.shape, range_m.device
        )
        if range_m.ndim != 4:
            raise ValueError(
                f"Ray rasters must have shape [B,T,H,W], got {tuple(range_m.shape)}."
            )
        expected_frame = tuple(range_m.shape[:2])
        if tuple(packet_age_s.shape) != expected_frame:
            raise ValueError(
                f"packet_age_s must have shape {expected_frame}, got "
                f"{tuple(packet_age_s.shape)}."
            )
        if tuple(frame_valid.shape) != expected_frame or frame_valid.dtype != torch.bool:
            raise ValueError(
                f"frame_valid must be boolean with shape {expected_frame}."
            )
        if packet_age_s.device != range_m.device or frame_valid.device != range_m.device:
            raise ValueError("All H2 tensors must be on the same device.")
        if not packet_age_s.is_floating_point() or not bool(
            torch.isfinite(packet_age_s).all()
        ):
            raise ValueError("packet_age_s must be finite floating point.")
        if bool((packet_age_s < 0).any()):
            raise ValueError("packet_age_s must be non-negative.")
        if bool((valid & ~frame_valid[:, :, None, None]).any()):
            raise ValueError("A frame_valid=False slot cannot contain a return.")
        if bool((valid & (return_age_s < packet_age_s[:, :, None, None])).any()):
            raise ValueError("Every valid return age must be >= its packet age.")

    @staticmethod
    def _validate_geometry_triplet(
        range_m: torch.Tensor,
        valid: torch.Tensor,
        return_age_s: torch.Tensor,
        expected_shape: torch.Size,
        expected_device: torch.device,
    ) -> None:
        if tuple(range_m.shape) != tuple(expected_shape):
            raise ValueError("Geometry range shape does not match native shape.")
        if tuple(valid.shape) != tuple(expected_shape) or valid.dtype != torch.bool:
            raise ValueError("Return-valid raster must be boolean and shape-aligned.")
        if tuple(return_age_s.shape) != tuple(expected_shape):
            raise ValueError("Return-age raster must be shape-aligned.")
        if (
            range_m.device != expected_device
            or valid.device != expected_device
            or return_age_s.device != expected_device
        ):
            raise ValueError("Geometry triplet must share the native device.")
        if not range_m.is_floating_point() or not return_age_s.is_floating_point():
            raise ValueError("Range and return age must be floating point.")
        if not bool(torch.isfinite(range_m).all()) or not bool(
            torch.isfinite(return_age_s).all()
        ):
            raise ValueError("Range and return age must be finite.")
        if bool((valid & (range_m <= 0)).any()):
            raise ValueError("Valid returns require positive range.")
        if bool((~valid & (range_m != 0)).any()):
            raise ValueError("Invalid ranges must be exactly zero.")
        if bool((return_age_s < 0).any()):
            raise ValueError("Return age must be non-negative.")
        if bool((~valid & (return_age_s != 0)).any()):
            raise ValueError("Invalid return ages must be exactly zero.")

    @staticmethod
    def _choice(name: str, value: str, choices: tuple[str, ...]) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string.")
        resolved = value.lower().replace("-", "_")
        if resolved not in choices:
            raise ValueError(
                f"{name} must be one of {choices}, got {value!r}."
            )
        return resolved


__all__ = [
    "RayEventAblationOutput",
    "RayEventAblationRouter",
    "SUPPORTED_GEOMETRY_VARIANTS",
    "SUPPORTED_HISTORY_REDUCTIONS",
    "SUPPORTED_TEMPORAL_BASELINES",
    "SUPPORTED_TIME_ASSOCIATIONS",
]
