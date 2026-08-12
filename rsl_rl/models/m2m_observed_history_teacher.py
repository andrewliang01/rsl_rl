# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen Phase-0 teacher for causally observed M52 history.

This is intentionally a proxy teacher, not a completion network.  A training-
only mapper may align past, physically observed M52 returns with ground-truth
poses and rasterize them on the M90 grid.  The proxy passes only valid measured
ranges to the frozen M90 ECMM encoder.  Unknown cells are converted to the
M90 policy's far/no-return value; their stored range and age can never reach the
encoder.

No trainable adapter is present here.  A learned map encoder belongs to the
separate Phase-1 teacher implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.m2m_frozen_ecmm import M2MFrozenECMMCore


@dataclass(frozen=True)
class ObservedHistoryMapContract:
    """Semantic trust contract for the privileged Phase-0 map observation."""

    source: str
    alignment: str
    target_grid: str
    uses_future_frames: bool
    uses_privileged_terrain_mesh: bool
    uses_synthetic_fill: bool
    near_range_m: float
    far_range_m: float
    max_age_s: float
    storage_backend: str = "frame_ring"
    voxel_size_m: float | None = None
    hash_capacity: int | None = None
    hash_max_probes: int | None = None

    SOURCE: ClassVar[str] = "observed_m52_history"
    ALIGNMENT: ClassVar[str] = "gt_pose_training_only"
    TARGET_GRID: ClassVar[str] = "m90_spherical_16x96"
    HISTORY_LAYOUT: ClassVar[str] = "current_and_past_only"
    TENSOR_LAYOUT: ClassVar[str] = "B_K_C_H_W"
    SHAPE: ClassVar[tuple[int, int, int, int]] = (1, 3, 16, 96)
    CHANNELS: ClassVar[tuple[str, str, str]] = ("range_m", "valid", "age_s")

    def __post_init__(self) -> None:
        for name in ("uses_future_frames", "uses_privileged_terrain_mesh", "uses_synthetic_fill"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit bool.")
        if self.source != self.SOURCE:
            raise ValueError(
                "Phase-0 source must be physically observed M52 history; "
                f"got source={self.source!r}."
            )
        if self.alignment != self.ALIGNMENT:
            raise ValueError(
                "Phase-0 alignment must use GT pose during training only; "
                f"got alignment={self.alignment!r}."
            )
        if self.target_grid != self.TARGET_GRID:
            raise ValueError(
                "Phase-0 target must be the M90 16x96 spherical grid; "
                f"got target_grid={self.target_grid!r}."
            )
        if self.uses_future_frames:
            raise ValueError("Future-frame leakage is forbidden by the observed-history contract.")
        if self.uses_privileged_terrain_mesh:
            raise ValueError("Terrain-mesh leakage is forbidden by the observed-history contract.")
        if self.uses_synthetic_fill:
            raise ValueError("Synthetic or oracle filling is forbidden by the observed-history contract.")
        for name in ("near_range_m", "far_range_m", "max_age_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite scalar, got {value!r}.")
        if self.near_range_m < 0.0:
            raise ValueError(f"near_range_m must be non-negative, got {self.near_range_m}.")
        if self.far_range_m <= self.near_range_m:
            raise ValueError(
                "far_range_m must be greater than near_range_m, got "
                f"{self.far_range_m} <= {self.near_range_m}."
            )
        if self.max_age_s <= 0.0:
            raise ValueError(f"max_age_s must be positive, got {self.max_age_s}.")
        if self.storage_backend not in ("frame_ring", "voxel_hash_2p5d"):
            raise ValueError(
                "storage_backend must be 'frame_ring' or 'voxel_hash_2p5d', got "
                f"{self.storage_backend!r}."
            )
        spatial_values = (self.voxel_size_m, self.hash_capacity, self.hash_max_probes)
        if self.storage_backend == "frame_ring":
            if any(value is not None for value in spatial_values):
                raise ValueError("frame_ring contract cannot declare voxel-hash parameters.")
        else:
            if (
                not isinstance(self.voxel_size_m, (int, float))
                or not math.isfinite(float(self.voxel_size_m))
                or self.voxel_size_m <= 0.0
            ):
                raise ValueError("voxel_size_m must be finite and positive for voxel_hash_2p5d.")
            if type(self.hash_capacity) is not int or self.hash_capacity < 2:
                raise ValueError("hash_capacity must be an integer >= 2 for voxel_hash_2p5d.")
            if self.hash_capacity & (self.hash_capacity - 1):
                raise ValueError("hash_capacity must be a power of two.")
            if (
                type(self.hash_max_probes) is not int
                or self.hash_max_probes < 1
                or self.hash_max_probes > self.hash_capacity
            ):
                raise ValueError("hash_max_probes must be in [1, hash_capacity].")

    def audit(self) -> dict[str, Any]:
        """Return a serializable description of the causal map contract."""
        fields = asdict(self)
        fields.update(
            {
                "history_layout": self.HISTORY_LAYOUT,
                "tensor_layout": self.TENSOR_LAYOUT,
                "shape_without_batch": list(self.SHAPE),
                "channels": list(self.CHANNELS),
                "unknown_encoder_value": "far_range_m",
                "valid_semantics": "finite_exact_binary_0_or_1",
                "valid_range_semantics": "finite_and_within_near_far",
                "age_semantics": "seconds_in_closed_interval_0_to_max_age",
                "unknown_age_semantics": "exactly_max_age_s",
            }
        )
        return fields


class M2MObservedHistoryProxyTeacher(nn.Module):
    """Frozen M90 proxy operating only on aligned, observed M52 history."""

    map_shape: tuple[int, int, int, int] = ObservedHistoryMapContract.SHAPE
    latent_dim: int = 64
    action_dim: int = 29

    def __init__(
        self,
        ecmm_core: M2MFrozenECMMCore,
        *,
        map_set: str,
        contract: ObservedHistoryMapContract,
    ) -> None:
        super().__init__()
        if not isinstance(ecmm_core, M2MFrozenECMMCore):
            raise TypeError(f"ecmm_core must be M2MFrozenECMMCore, got {type(ecmm_core).__name__}.")
        if not ecmm_core.teacher_loaded:
            raise ValueError("The frozen M90 ECMM artifact must be loaded before constructing the proxy teacher.")
        if not isinstance(map_set, str) or not map_set:
            raise ValueError("map_set must be a non-empty observation-group name.")
        if not isinstance(contract, ObservedHistoryMapContract):
            raise TypeError(f"contract must be ObservedHistoryMapContract, got {type(contract).__name__}.")

        actor = ecmm_core.actor
        if actor.vision_spatial_size != self.map_shape[-2:]:
            raise ValueError(
                "Frozen M90 encoder spatial contract must be 16x96, got "
                f"{actor.vision_spatial_size}."
            )
        if actor.cnn_observation_type != "depthcamera":
            raise ValueError(
                "Phase-0 range proxy requires the frozen M90 depthcamera normalization, "
                f"got {actor.cnn_observation_type!r}."
            )
        if not math.isclose(actor.depth_camera_near, contract.near_range_m, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "Map/frozen-encoder near range mismatch: "
                f"contract={contract.near_range_m}, actor={actor.depth_camera_near}."
            )
        if not math.isclose(actor.depth_camera_far, contract.far_range_m, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "Map/frozen-encoder far range mismatch: "
                f"contract={contract.far_range_m}, actor={actor.depth_camera_far}."
            )

        self.ecmm_core = ecmm_core
        self.map_set = map_set
        self.contract = contract
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> M2MObservedHistoryProxyTeacher:
        """Keep the proxy, frozen core, and every frozen BatchNorm in eval mode."""
        super().train(mode)
        self.ecmm_core.eval()
        return self

    def _validate_map(self, teacher_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if teacher_map.dtype != torch.float32:
            raise ValueError(f"Teacher map must use float32 metric channels, got {teacher_map.dtype}.")
        if teacher_map.ndim != 5 or tuple(teacher_map.shape[1:]) != self.map_shape:
            raise ValueError(
                "Teacher map must have shape [B,1,3,16,96], got "
                f"{tuple(teacher_map.shape)}."
            )
        if teacher_map.shape[0] <= 0:
            raise ValueError("Teacher map batch dimension must be positive.")

        range_m = teacher_map[:, 0, 0:1]
        valid = teacher_map[:, 0, 1:2]
        age_s = teacher_map[:, 0, 2:3]

        valid_is_finite = torch.isfinite(valid)
        valid_is_binary = torch.logical_or(valid == 0.0, valid == 1.0)
        valid_mask = valid == 1.0
        invalid_mask = ~valid_mask
        range_is_finite = torch.isfinite(range_m)
        age_is_finite = torch.isfinite(age_s)
        unknown_age_is_max = torch.isclose(
            age_s,
            torch.full_like(age_s, self.contract.max_age_s),
            rtol=0.0,
            atol=1e-6,
        )
        # Consolidating these device predicates into one host transfer avoids
        # one GPU synchronization per invariant at every policy step.
        checks = torch.stack(
            (
                valid_is_finite.all(),
                valid_is_binary.all(),
                torch.logical_or(~valid_mask, range_is_finite).all(),
                torch.logical_or(~valid_mask, range_m >= self.contract.near_range_m).all(),
                torch.logical_or(~valid_mask, range_m <= self.contract.far_range_m).all(),
                age_is_finite.all(),
                (age_s >= 0.0).all(),
                (age_s <= self.contract.max_age_s).all(),
                torch.logical_or(~invalid_mask, unknown_age_is_max).all(),
            )
        ).detach().cpu().tolist()
        error_messages = (
            "Teacher valid channel must be finite and exactly binary.",
            "Teacher valid channel must contain only exact {0,1} values.",
            "Every valid teacher range must be finite.",
            "A valid teacher range is below near_range_m.",
            "A valid teacher range exceeds far_range_m.",
            "Teacher age channel must be finite.",
            "Teacher age must lie in [0,max_age_s].",
            "Teacher age must lie in [0,max_age_s].",
            "Unknown cells must encode age exactly as max_age_s.",
        )
        for passed, message in zip(checks, error_messages):
            if not passed:
                raise ValueError(message)
        return range_m, valid, age_s

    def prepare_m90_range(
        self, teacher_map: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Discard unknown values and return range-only M90 input plus diagnostics."""
        range_m, valid, age_s = self._validate_map(teacher_map)
        valid_mask = valid == 1.0
        far = torch.full_like(range_m, self.contract.far_range_m)
        # This is the M90 sensor's no-return representation, not a geometric
        # completion.  Neither unknown range values nor unknown ages survive.
        observed_only_range = torch.where(valid_mask, range_m, far)

        valid_float = valid_mask.to(dtype=range_m.dtype)
        reduce_dims = (-3, -2, -1)
        observed_count = valid_float.sum(dim=reduce_dims)
        total_count = float(range_m.shape[-2] * range_m.shape[-1])
        coverage = observed_count / total_count
        observed_age_sum = (age_s * valid_float).sum(dim=reduce_dims)
        mean_age = observed_age_sum / observed_count.clamp_min(1.0)
        max_age = torch.where(valid_mask, age_s, torch.zeros_like(age_s)).amax(dim=reduce_dims)
        diagnostics = {
            "observed_coverage": coverage,
            "unknown_fraction": 1.0 - coverage,
            "observed_count": observed_count,
            "observed_mean_age_s": mean_age,
            "observed_max_age_s": max_age,
        }
        return observed_only_range, diagnostics

    @torch.no_grad()
    def teacher_labels(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return detached A64/action29 labels and observed-history diagnostics."""
        if self.map_set not in obs:
            raise KeyError(
                f"Observation is missing teacher map group {self.map_set!r}; "
                f"available groups are {list(obs.keys())}."
            )
        m90_range, diagnostics = self.prepare_m90_range(obs[self.map_set])
        proprio_features = self.ecmm_core.encode_proprio(obs)
        latent_a = self.ecmm_core.encode_teacher_A(m90_range)
        action_mean = self.ecmm_core.action_mean_from_A(proprio_features, latent_a)
        return latent_a, action_mean, diagnostics

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Return the deterministic Phase-0 teacher action."""
        return self.teacher_labels(obs)[1]

    def parameter_audit(self) -> dict[str, Any]:
        """Return freeze, causal-source, input, and frozen-artifact evidence."""
        parameters = list(self.parameters())
        batch_norms = [
            module
            for module in self.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        return {
            "phase": "phase0_observed_history_proxy",
            "trainable_adapter_present": False,
            "map_set": self.map_set,
            "map_contract": self.contract.audit(),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
            "batch_norm": {
                "count": len(batch_norms),
                "training_count": sum(int(module.training) for module in batch_norms),
            },
            "frozen_ecmm": self.ecmm_core.parameter_audit(),
        }


__all__ = ["M2MObservedHistoryProxyTeacher", "ObservedHistoryMapContract"]
