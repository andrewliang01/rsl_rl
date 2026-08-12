# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""From-scratch privileged-map teacher for M2M-360.

Unlike :mod:`m2m_observed_history_formal_teacher`, this actor does not load a
previous locomotion policy.  The observed-history map encoder ``A``,
proprioception encoder ``B``, fusion/action head ``C``, action distribution,
and proprioception normalizer are initialized by the current training process
and optimized together by ordinary PPO.  A trained teacher checkpoint can
later become the frozen B/C dependency for a map-free student.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any, ClassVar

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.m2m_observed_history_formal_teacher import M2MObservedHistoryMapEncoder
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


_SUPPORTED_MAP_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FORBIDDEN_INPUT_TOKENS = (
    "future",
    "m90",
    "mesh",
    "oracle",
    "terrain_id",
    "height_scan",
)


@dataclass(frozen=True)
class M2MScratchTeacherMapContract:
    """Two-channel, episode-retained map contract used only by scratch F07."""

    source: str
    alignment: str
    target_grid: str
    uses_future_frames: bool
    uses_privileged_terrain_mesh: bool
    uses_synthetic_fill: bool
    near_range_m: float
    far_range_m: float
    storage_backend: str
    retention_mode: str
    voxel_size_m: float
    hash_capacity: int
    hash_max_probes: int

    SHAPE: ClassVar[tuple[int, int, int, int]] = (1, 2, 16, 96)
    CHANNELS: ClassVar[tuple[str, str]] = ("range_m", "valid")

    def __post_init__(self) -> None:
        if self.source != "observed_m52_history":
            raise ValueError("Scratch teacher source must be observed M52 history.")
        if self.alignment != "gt_pose_training_only":
            raise ValueError("Scratch teacher alignment must be training-only GT pose.")
        if self.target_grid != "m90_spherical_16x96":
            raise ValueError("Scratch teacher target grid must be m90_spherical_16x96.")
        for name in ("uses_future_frames", "uses_privileged_terrain_mesh", "uses_synthetic_fill"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an explicit bool.")
            if getattr(self, name):
                if name == "uses_future_frames":
                    raise ValueError("Future-frame leakage is forbidden for the scratch teacher.")
                raise ValueError(f"{name} must be false for the causal scratch teacher.")
        if not math.isfinite(self.near_range_m) or self.near_range_m < 0.0:
            raise ValueError("near_range_m must be finite and non-negative.")
        if not math.isfinite(self.far_range_m) or self.far_range_m <= self.near_range_m:
            raise ValueError("far_range_m must be finite and greater than near_range_m.")
        if self.storage_backend != "voxel_hash_2p5d":
            raise ValueError("Scratch F07 requires voxel_hash_2p5d storage.")
        if self.retention_mode != "episode":
            raise ValueError("Scratch F07 points must persist until episode reset.")
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive.")
        if self.hash_capacity < 2 or self.hash_capacity & (self.hash_capacity - 1):
            raise ValueError("hash_capacity must be a power of two >= 2.")
        if not 1 <= self.hash_max_probes <= self.hash_capacity:
            raise ValueError("hash_max_probes must be within hash capacity.")

    def audit(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "history_layout": "current_and_past_until_episode_reset",
                "tensor_layout": "B_K_C_H_W",
                "shape_without_batch": list(self.SHAPE),
                "channels": list(self.CHANNELS),
                "timestamp_visibility": "mapper_internal_only",
            }
        )
        return result


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class M2MObservedHistoryScratchTeacher(nn.Module):
    """Jointly train A/B/C from causal, training-only observed history."""

    is_recurrent: bool = False
    latent_dim: int = 64
    proprio_dim: int = 96
    action_dim: int = 29
    map_shape: tuple[int, int, int, int] = M2MScratchTeacherMapContract.SHAPE

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        teacher_map_set: str,
        proprio_sets: Sequence[str],
        map_contract: M2MScratchTeacherMapContract | Mapping[str, Any],
        encoder_hidden_channels: Sequence[int] = (16, 32, 64),
        encoder_pooled_spatial_size: tuple[int, int] = (2, 6),
        encoder_mlp_hidden_dim: int = 128,
        prop_feature_dim: int = 64,
        prop_hidden_dims: Sequence[int] = (128,),
        fusion_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: Mapping[str, Any] | None = None,
        strict_runtime_value_checks: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(obs, TensorDict):
            raise TypeError(f"obs must be a TensorDict, got {type(obs).__name__}.")
        if len(obs.batch_size) != 1 or obs.batch_size[0] <= 0:
            raise ValueError(f"Construction obs requires one positive batch dimension, got {obs.batch_size}.")
        if type(output_dim) is not int or output_dim != self.action_dim:
            raise ValueError(f"Unitree-G1 scratch teacher requires output_dim={self.action_dim}.")
        if type(strict_runtime_value_checks) is not bool:
            raise TypeError("strict_runtime_value_checks must be an explicit bool.")
        if type(obs_normalization) is not bool:
            raise TypeError("obs_normalization must be an explicit bool.")
        if not isinstance(obs_set, str) or obs_set not in obs_groups:
            raise KeyError(f"obs_groups is missing observation set {obs_set!r}.")
        if not isinstance(teacher_map_set, str) or not teacher_map_set:
            raise ValueError("teacher_map_set must be a non-empty string.")
        forbidden_map_tokens = [
            token for token in _FORBIDDEN_INPUT_TOKENS if token in teacher_map_set.lower()
        ]
        if forbidden_map_tokens:
            raise ValueError(f"teacher_map_set contains forbidden tokens {forbidden_map_tokens}.")
        if teacher_map_set not in obs:
            raise KeyError(f"Construction obs is missing teacher map {teacher_map_set!r}.")
        map_sample = obs[teacher_map_set]
        if map_sample.dtype not in _SUPPORTED_MAP_DTYPES or tuple(map_sample.shape[1:]) != self.map_shape:
            raise ValueError(
                f"{teacher_map_set!r} must be float [B,1,2,16,96], got "
                f"dtype={map_sample.dtype}, shape={tuple(map_sample.shape)}."
            )

        if isinstance(map_contract, Mapping):
            contract = M2MScratchTeacherMapContract(**copy.deepcopy(dict(map_contract)))
        elif isinstance(map_contract, M2MScratchTeacherMapContract):
            contract = map_contract
        else:
            raise TypeError("map_contract must be M2MScratchTeacherMapContract or a mapping.")

        proprio_groups = tuple(proprio_sets)
        if not proprio_groups or any(not isinstance(group, str) or not group for group in proprio_groups):
            raise ValueError("proprio_sets must contain non-empty group names.")
        if len(set(proprio_groups)) != len(proprio_groups):
            raise ValueError(f"proprio_sets contains duplicates: {proprio_groups}.")
        expected_groups = (*proprio_groups, teacher_map_set)
        if tuple(obs_groups[obs_set]) != expected_groups:
            raise ValueError(
                "Scratch teacher actor groups must be exactly proprio followed by observed history: "
                f"expected={expected_groups}, actual={tuple(obs_groups[obs_set])}."
            )

        group_dims: dict[str, int] = {}
        proprio_dim = 0
        for group in proprio_groups:
            forbidden = [token for token in _FORBIDDEN_INPUT_TOKENS if token in group.lower()]
            if forbidden:
                raise ValueError(f"Proprio group {group!r} contains forbidden tokens {forbidden}.")
            if group not in obs:
                raise KeyError(f"Construction obs is missing proprio group {group!r}.")
            value = obs[group]
            if value.ndim != 2 or value.dtype != torch.float32:
                raise ValueError(
                    f"Proprio group {group!r} must be float32 [B,D], got "
                    f"dtype={value.dtype}, shape={tuple(value.shape)}."
                )
            group_dims[group] = value.shape[-1]
            proprio_dim += value.shape[-1]
        if proprio_dim != self.proprio_dim:
            raise ValueError(f"Scratch teacher requires {self.proprio_dim} proprio values, got {proprio_dim}.")

        prop_feature_dim = _positive_int("prop_feature_dim", prop_feature_dim)
        prop_hidden_dims = tuple(_positive_int("prop hidden dimension", value) for value in prop_hidden_dims)
        fusion_hidden_dims = tuple(
            _positive_int("fusion hidden dimension", value) for value in fusion_hidden_dims
        )
        if not prop_hidden_dims or not fusion_hidden_dims:
            raise ValueError("prop_hidden_dims and fusion_hidden_dims cannot be empty.")

        distribution_values = copy.deepcopy(
            dict(
                distribution_cfg
                or {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                }
            )
        )
        class_name = distribution_values.pop("class_name", None)
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("distribution_cfg.class_name must be a non-empty string.")
        distribution_class: type[Distribution] = resolve_callable(class_name)

        self.map_encoder = M2MObservedHistoryMapEncoder(
            hidden_channels=encoder_hidden_channels,
            pooled_spatial_size=encoder_pooled_spatial_size,
            mlp_hidden_dim=encoder_mlp_hidden_dim,
            input_channels=M2MScratchTeacherMapContract.CHANNELS,
        )
        self.obs_normalization = obs_normalization
        self.obs_normalizer: nn.Module
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(proprio_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.prop_mlp = MLP(proprio_dim, prop_feature_dim, prop_hidden_dims, activation)
        self.distribution = distribution_class(output_dim, **distribution_values)
        self.mlp = MLP(
            prop_feature_dim + self.latent_dim,
            self.distribution.input_dim,
            fusion_hidden_dims,
            activation,
        )
        self.distribution.init_mlp_weights(self.mlp)

        self.teacher_map_set = teacher_map_set
        self.proprio_sets = proprio_groups
        self.proprio_group_dims = group_dims
        self.obs_groups = list(expected_groups)
        self.obs_set = obs_set
        self.contract = contract
        self.strict_runtime_value_checks = strict_runtime_value_checks
        self.prop_feature_dim = prop_feature_dim
        self.prop_hidden_dims = prop_hidden_dims
        self.fusion_hidden_dims = fusion_hidden_dims
        self.activation = activation
        self.distribution_config = {"class_name": class_name, **distribution_values}

    def _strict_validate_map_values(
        self,
        range_m: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        valid_mask = valid == 1.0
        invalid_mask = valid == 0.0
        tolerance = float(torch.finfo(valid.dtype).eps) * max(
            1.0,
            abs(self.contract.far_range_m),
        )
        checks = torch.stack(
            (
                torch.isfinite(valid).all(),
                torch.logical_or(valid_mask, invalid_mask).all(),
                torch.logical_or(~valid_mask, torch.isfinite(range_m)).all(),
                torch.logical_or(~valid_mask, range_m >= self.contract.near_range_m - tolerance).all(),
                torch.logical_or(~valid_mask, range_m <= self.contract.far_range_m + tolerance).all(),
                torch.logical_or(~invalid_mask, torch.isfinite(range_m)).all(),
            )
        ).detach().cpu().tolist()
        messages = (
            "valid must be finite",
            "valid must contain exact binary values",
            "valid range must be finite",
            "valid range is below near_range_m",
            "valid range exceeds far_range_m",
            "unknown range must be finite",
        )
        for passed, message in zip(checks, messages):
            if not passed:
                raise ValueError(message)

    def _normalize_map(self, teacher_map: torch.Tensor) -> torch.Tensor:
        if teacher_map.dtype not in _SUPPORTED_MAP_DTYPES:
            raise ValueError(f"Teacher map dtype is unsupported: {teacher_map.dtype}.")
        if teacher_map.ndim != 5 or tuple(teacher_map.shape[1:]) != self.map_shape:
            raise ValueError(f"Teacher map must be [B,1,2,16,96], got {tuple(teacher_map.shape)}.")
        range_m = teacher_map[:, 0, 0:1]
        valid = teacher_map[:, 0, 1:2]
        if self.strict_runtime_value_checks:
            self._strict_validate_map_values(range_m, valid)

        compute_dtype = next(self.map_encoder.parameters()).dtype
        range_m = range_m.to(dtype=compute_dtype)
        valid = valid.to(dtype=compute_dtype)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = valid > 0.5
        valid = valid_mask.to(dtype=compute_dtype)
        range_m = torch.nan_to_num(
            range_m,
            nan=self.contract.far_range_m,
            posinf=self.contract.far_range_m,
            neginf=self.contract.near_range_m,
        ).clamp(self.contract.near_range_m, self.contract.far_range_m)
        range_m = torch.where(valid_mask, range_m, torch.full_like(range_m, self.contract.far_range_m))
        range_unit = (range_m - self.contract.near_range_m) / (
            self.contract.far_range_m - self.contract.near_range_m
        )
        return torch.cat((2.0 * range_unit - 1.0, 2.0 * valid - 1.0), dim=1)

    def _proprio(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([obs[group] for group in self.proprio_sets], dim=-1)
        return self.prop_mlp(self.obs_normalizer(proprio))

    def predict_latent(self, obs: TensorDict) -> torch.Tensor:
        if self.teacher_map_set not in obs:
            raise KeyError(f"Observation is missing {self.teacher_map_set!r}.")
        return self.map_encoder(self._normalize_map(obs[self.teacher_map_set]))

    def predict_latent_and_action_mean(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        latent_a = self.predict_latent(obs)
        raw_action = self.mlp(torch.cat((self._proprio(obs), latent_a), dim=-1))
        return latent_a, self.distribution.deterministic_output(raw_action)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if hidden_state is not None:
            raise ValueError("Scratch teacher is non-recurrent and rejects hidden_state.")
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent_a = self.predict_latent(obs)
        raw_action = self.mlp(torch.cat((self._proprio(obs), latent_a), dim=-1))
        if stochastic_output:
            self.distribution.update(raw_action)
            return self.distribution.sample()
        return self.distribution.deterministic_output(raw_action)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: None = None) -> None:
        del dones
        if hidden_state is not None:
            raise ValueError("Scratch teacher has no hidden state.")

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            proprio = torch.cat([obs[group] for group in self.proprio_sets], dim=-1)
            self.obs_normalizer.update(proprio)  # type: ignore[attr-defined]

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def architecture_receipt(self) -> dict[str, Any]:
        return {
            "training_initialization": "random_no_pretrained_policy",
            "teacher_map_set": self.teacher_map_set,
            "proprio_sets": list(self.proprio_sets),
            "proprio_group_dims": dict(self.proprio_group_dims),
            "map_contract": self.contract.audit(),
            "map_encoder": self.map_encoder.architecture_receipt(),
            "prop_feature_dim": self.prop_feature_dim,
            "prop_hidden_dims": list(self.prop_hidden_dims),
            "fusion_hidden_dims": list(self.fusion_hidden_dims),
            "activation": self.activation,
            "obs_normalization": self.obs_normalization,
            "distribution": copy.deepcopy(self.distribution_config),
            "latent_a_dim": self.latent_dim,
            "latent_b_dim": self.prop_feature_dim,
            "action_dim": self.action_dim,
        }

    def parameter_audit(self) -> dict[str, Any]:
        trainable_names = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        frozen_names = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
        required_prefixes = ("map_encoder.", "prop_mlp.", "mlp.", "distribution.")
        represented = {
            prefix: any(name.startswith(prefix) for name in trainable_names)
            for prefix in required_prefixes
        }
        return {
            "phase": "F07_from_scratch_privileged_map_teacher",
            "pretrained_policy_loaded": False,
            "checkpoint_path_fields_accepted": False,
            "all_actor_parameters_trainable": bool(trainable_names) and not frozen_names,
            "trainable_parameter_names": trainable_names,
            "frozen_parameter_names": frozen_names,
            "required_trainable_components_present": represented,
            "parameter_counts": {
                "map_encoder_A": _parameter_count(self.map_encoder),
                "proprio_encoder_B": _parameter_count(self.prop_mlp),
                "control_head_C": _parameter_count(self.mlp),
                "action_distribution": _parameter_count(self.distribution),
                "total_actor": _parameter_count(self),
            },
            "actor_inputs": {
                "ordered_groups": list(self.obs_groups),
                "uses_future_frames": False,
                "uses_m90_observation": False,
                "uses_terrain_mesh": False,
                "uses_synthetic_fill": False,
            },
            "checkpoint_contract": {
                "owner": "ordinary_PPO_full_actor_state_dict",
                "saved_components": ["map_encoder_A", "proprio_encoder_B", "control_head_C", "distribution"],
            },
            "architecture": self.architecture_receipt(),
        }


__all__ = ["M2MObservedHistoryScratchTeacher", "M2MScratchTeacherMapContract"]
