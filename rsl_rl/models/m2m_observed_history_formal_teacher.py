# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trainable causal observed-history teacher with a frozen ECMM control head.

The teacher consumes only a training-time spherical raster assembled from
physically observed M52 returns.  Its three channels are metric range, an
explicit valid bit, and observation age.  A small circular-azimuth CNN maps
those channels to the 64-D perception latent ``A`` used by the frozen M90 ECMM
controller.  The checkpoint-bound proprioception encoder ``B``, fusion/action
head ``C``, and action distribution never become trainable.

The class intentionally owns a narrow map-only checkpoint API.  Reconstructing
a teacher still requires the frozen ECMM path and expected SHA-256 from
external configuration; frozen control weights are never copied into a C06
teacher checkpoint.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.m2m_frozen_ecmm import M2MFrozenECMMCore
from rsl_rl.models.m2m_observed_history_teacher import ObservedHistoryMapContract
from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import unpad_trajectories


_MAP_CHECKPOINT_SCHEMA = "m2m_observed_history_formal_teacher_map_encoder_v1"
_SUPPORTED_MAP_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FORBIDDEN_INPUT_TOKENS = (
    "future",
    "m90",
    "mesh",
    "oracle",
    "terrain_id",
    "height_scan",
)


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _parameter_counts(module: nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }


class _CircularAzimuthBlock(nn.Module):
    """Convolution with circular azimuth and ordinary elevation padding."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=(1, 0),
        )
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (1, 1, 0, 0), mode="circular")
        return self.activation(self.norm(self.conv(value)))


class M2MObservedHistoryMapEncoder(nn.Module):
    """Small configurable-channel circular CNN producing latent ``A64``."""

    input_channels: tuple[str, ...] = ("range_m", "valid", "age_s")
    spatial_size: tuple[int, int] = (16, 96)
    output_dim: int = 64

    def __init__(
        self,
        *,
        hidden_channels: Sequence[int] = (16, 32, 64),
        pooled_spatial_size: tuple[int, int] = (2, 6),
        mlp_hidden_dim: int = 128,
        input_channels: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        channels_semantics = tuple(self.input_channels if input_channels is None else input_channels)
        if not channels_semantics or any(not isinstance(value, str) or not value for value in channels_semantics):
            raise ValueError("input_channels must contain non-empty semantic names.")
        if len(set(channels_semantics)) != len(channels_semantics):
            raise ValueError("input_channels cannot contain duplicate semantics.")
        self.input_channels = channels_semantics
        channels = tuple(_positive_int("hidden channel", value) for value in hidden_channels)
        if not channels:
            raise ValueError("hidden_channels must contain at least one channel width.")
        if len(pooled_spatial_size) != 2:
            raise ValueError("pooled_spatial_size must contain exactly (height,width).")
        pooled = tuple(_positive_int("pooled spatial size", value) for value in pooled_spatial_size)
        mlp_hidden_dim = _positive_int("mlp_hidden_dim", mlp_hidden_dim)

        blocks: list[nn.Module] = []
        in_channels = len(self.input_channels)
        for out_channels in channels:
            blocks.append(_CircularAzimuthBlock(in_channels, out_channels, stride=2))
            in_channels = out_channels
        self.spatial_encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pooled)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pooled[0] * pooled[1], mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, self.output_dim),
        )
        self.hidden_channels = channels
        self.pooled_spatial_size = pooled
        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, normalized_map: torch.Tensor) -> torch.Tensor:
        if normalized_map.ndim != 4 or tuple(normalized_map.shape[1:]) != (
            len(self.input_channels),
            *self.spatial_size,
        ):
            raise ValueError(
                f"Normalized teacher map must be [B,{len(self.input_channels)},16,96], got "
                f"{tuple(normalized_map.shape)}."
            )
        latent_a = self.projection(self.pool(self.spatial_encoder(normalized_map)))
        if latent_a.shape[-1] != self.output_dim:
            raise RuntimeError(f"Observed-history encoder produced invalid shape {tuple(latent_a.shape)}.")
        return latent_a

    def architecture_receipt(self) -> dict[str, Any]:
        return {
            "type": "small_circular_azimuth_cnn_mlp",
            "input_channels": list(self.input_channels),
            "spatial_size": list(self.spatial_size),
            "hidden_channels": list(self.hidden_channels),
            "pooled_spatial_size": list(self.pooled_spatial_size),
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "output_dim": self.output_dim,
        }


class M2MObservedHistoryFormalTeacher(nn.Module):
    """Phase-1 observed-history actor; only its map encoder is trainable."""

    is_recurrent: bool = False
    latent_dim: int = 64
    proprio_dim: int = 96
    action_dim: int = 29
    map_shape: tuple[int, int, int, int] = ObservedHistoryMapContract.SHAPE

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        teacher_map_set: str,
        proprio_sets: Sequence[str],
        map_contract: ObservedHistoryMapContract | Mapping[str, Any],
        frozen_ecmm_checkpoint_path: str | None = None,
        frozen_ecmm_expected_sha256: str | None = None,
        frozen_ecmm_actor_cfg: Mapping[str, Any] | None = None,
        frozen_ecmm_actor_state_dict_key: str = "actor_state_dict",
        shared_ecmm_core: M2MFrozenECMMCore | None = None,
        encoder_hidden_channels: Sequence[int] = (16, 32, 64),
        encoder_pooled_spatial_size: tuple[int, int] = (2, 6),
        encoder_mlp_hidden_dim: int = 128,
        strict_runtime_value_checks: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(obs, TensorDict):
            raise TypeError(f"obs must be a TensorDict, got {type(obs).__name__}.")
        if len(obs.batch_size) != 1 or obs.batch_size[0] <= 0:
            raise ValueError(f"Construction obs must have one positive batch dimension, got {obs.batch_size}.")
        if type(output_dim) is not int or output_dim != self.action_dim:
            raise ValueError(f"M2M Unitree-G1 teacher requires output_dim={self.action_dim}, got {output_dim!r}.")
        if type(strict_runtime_value_checks) is not bool:
            raise ValueError(
                "strict_runtime_value_checks must be an explicit bool, got "
                f"{strict_runtime_value_checks!r}."
            )
        if not isinstance(teacher_map_set, str) or not teacher_map_set:
            raise ValueError("teacher_map_set must be a non-empty observation-group name.")
        lowered_map_set = teacher_map_set.lower()
        forbidden_tokens = [token for token in _FORBIDDEN_INPUT_TOKENS if token in lowered_map_set]
        if forbidden_tokens:
            raise ValueError(
                "teacher_map_set must describe causal observed M52 history, not privileged inputs; "
                f"forbidden tokens={forbidden_tokens}."
            )
        if teacher_map_set not in obs:
            raise KeyError(f"Construction obs is missing teacher map group {teacher_map_set!r}.")
        map_sample = obs[teacher_map_set]
        if map_sample.dtype not in _SUPPORTED_MAP_DTYPES or tuple(map_sample.shape[1:]) != self.map_shape:
            raise ValueError(
                f"{teacher_map_set!r} must have float16/bfloat16/float32 shape [B,1,3,16,96], got "
                f"dtype={map_sample.dtype}, shape={tuple(map_sample.shape)}."
            )

        if isinstance(map_contract, Mapping):
            contract = ObservedHistoryMapContract(**copy.deepcopy(dict(map_contract)))
        elif isinstance(map_contract, ObservedHistoryMapContract):
            contract = map_contract
        else:
            raise TypeError("map_contract must be ObservedHistoryMapContract or a mapping.")

        if not isinstance(obs_set, str) or obs_set not in obs_groups:
            raise KeyError(f"obs_groups is missing observation set {obs_set!r}.")
        proprio_groups = tuple(proprio_sets)
        if not proprio_groups or any(not isinstance(group, str) or not group for group in proprio_groups):
            raise ValueError("proprio_sets must contain non-empty observation-group names.")
        if len(set(proprio_groups)) != len(proprio_groups):
            raise ValueError(f"proprio_sets contains duplicate groups: {proprio_groups}.")
        if teacher_map_set in proprio_groups:
            raise ValueError("teacher_map_set cannot also be a proprioception group.")
        expected_active = (*proprio_groups, teacher_map_set)
        active_groups = tuple(obs_groups[obs_set])
        if active_groups != expected_active:
            raise ValueError(
                "Formal teacher actor groups must be exactly the explicit ECMM proprio groups plus the causal map: "
                f"expected={expected_active}, actual={active_groups}."
            )

        proprio_dim = 0
        proprio_group_dims: dict[str, int] = {}
        for group in proprio_groups:
            lowered_group = group.lower()
            forbidden = [token for token in _FORBIDDEN_INPUT_TOKENS if token in lowered_group]
            if forbidden:
                raise ValueError(f"Proprioception group {group!r} contains forbidden tokens {forbidden}.")
            if group not in obs:
                raise KeyError(f"Construction obs is missing proprioception group {group!r}.")
            value = obs[group]
            if value.ndim != 2 or value.dtype != torch.float32:
                raise ValueError(
                    f"Proprioception group {group!r} must be float32 [B,D], got "
                    f"dtype={value.dtype}, shape={tuple(value.shape)}."
                )
            proprio_group_dims[group] = value.shape[-1]
            proprio_dim += value.shape[-1]
        if proprio_dim != self.proprio_dim:
            raise ValueError(f"Frozen ECMM B requires exactly {self.proprio_dim} values, got {proprio_dim}.")

        external_fields = (
            frozen_ecmm_checkpoint_path,
            frozen_ecmm_expected_sha256,
            frozen_ecmm_actor_cfg,
        )
        if shared_ecmm_core is not None:
            if not isinstance(shared_ecmm_core, M2MFrozenECMMCore):
                raise TypeError("shared_ecmm_core must be M2MFrozenECMMCore or None.")
            if any(value is not None for value in external_fields):
                raise ValueError(
                    "Provide either shared_ecmm_core or the complete external frozen-ECMM configuration, not both."
                )
            ecmm_core = shared_ecmm_core
        else:
            if any(value is None for value in external_fields):
                raise ValueError(
                    "Without shared_ecmm_core, frozen checkpoint path, expected SHA-256, and actor config are required."
                )
            assert frozen_ecmm_actor_cfg is not None
            actor_cfg = copy.deepcopy(dict(frozen_ecmm_actor_cfg))
            reserved = {"obs", "obs_groups", "obs_set", "output_dim"}
            conflicts = sorted(reserved.intersection(actor_cfg))
            if conflicts:
                raise ValueError(f"frozen_ecmm_actor_cfg cannot override constructor fields: {conflicts}.")
            elevation_set = actor_cfg.get("elevation_set", "height_scan_actor")
            if not isinstance(elevation_set, str) or not elevation_set:
                raise ValueError("Frozen actor elevation_set must be a non-empty string.")
            if elevation_set in expected_active:
                raise ValueError("Frozen M90 elevation placeholder must not be a formal-teacher actor input.")
            spatial_size = tuple(actor_cfg.get("vision_spatial_size", (25, 17)))
            if spatial_size != self.map_shape[-2:]:
                raise ValueError(f"Frozen M90 ECMM must use vision_spatial_size=(16,96), got {spatial_size}.")
            reference = obs[proprio_groups[0]]
            frozen_obs = TensorDict(
                {
                    **{group: obs[group] for group in proprio_groups},
                    elevation_set: torch.full(
                        (obs.batch_size[0], 1, *spatial_size),
                        float(actor_cfg.get("depth_camera_far", 6.0)),
                        dtype=reference.dtype,
                        device=reference.device,
                    ),
                },
                batch_size=obs.batch_size,
                device=obs.device,
            )
            frozen_actor = PropMLPElevationFusionModel(
                obs=frozen_obs,
                obs_groups={obs_set: [*proprio_groups, elevation_set]},
                obs_set=obs_set,
                output_dim=output_dim,
                **actor_cfg,
            )
            ecmm_core = M2MFrozenECMMCore(
                frozen_actor,
                checkpoint_path=frozen_ecmm_checkpoint_path,  # type: ignore[arg-type]
                expected_sha256=frozen_ecmm_expected_sha256,  # type: ignore[arg-type]
                actor_state_dict_key=frozen_ecmm_actor_state_dict_key,
            )

        if tuple(ecmm_core.actor.obs_groups) != proprio_groups:
            raise ValueError(
                "Frozen ECMM B observation order differs from formal-teacher proprio_sets: "
                f"core={tuple(ecmm_core.actor.obs_groups)}, teacher={proprio_groups}."
            )
        if ecmm_core.actor.vision_spatial_size != self.map_shape[-2:]:
            raise ValueError("Frozen ECMM teacher artifact must use the 16x96 M90 grid.")
        if ecmm_core.actor.cnn_observation_type != "depthcamera":
            raise ValueError("Frozen ECMM teacher artifact must use depthcamera range normalization.")
        if ecmm_core.actor.depth_camera_near != contract.near_range_m:
            raise ValueError("Map contract and frozen ECMM near ranges differ.")
        if ecmm_core.actor.depth_camera_far != contract.far_range_m:
            raise ValueError("Map contract and frozen ECMM far ranges differ.")

        self.ecmm_core = ecmm_core
        self.ecmm_core.requires_grad_(False)
        self.ecmm_core.eval()
        self.map_encoder = M2MObservedHistoryMapEncoder(
            hidden_channels=encoder_hidden_channels,
            pooled_spatial_size=encoder_pooled_spatial_size,
            mlp_hidden_dim=encoder_mlp_hidden_dim,
        )
        self.teacher_map_set = teacher_map_set
        self.proprio_sets = proprio_groups
        self.proprio_group_dims = proprio_group_dims
        self.obs_groups = list(expected_active)
        self.obs_set = obs_set
        self.contract = contract
        self.strict_runtime_value_checks = strict_runtime_value_checks

    @property
    def distribution(self) -> Distribution:
        distribution = self.ecmm_core.actor.distribution
        if distribution is None:  # Guarded by M2MFrozenECMMCore.
            raise RuntimeError("Frozen ECMM action distribution is unavailable.")
        return distribution

    def train(self, mode: bool = True) -> M2MObservedHistoryFormalTeacher:
        super().train(mode)
        self.ecmm_core.eval()
        return self

    def _strict_validate_map_values(
        self,
        range_m: torch.Tensor,
        valid: torch.Tensor,
        age_s: torch.Tensor,
    ) -> None:
        """Run synchronized fail-closed checks for tests/preflight, never fast rollout."""
        valid_mask = valid == 1.0
        invalid_mask = valid == 0.0
        quantization_tolerance = float(torch.finfo(valid.dtype).eps) * max(
            1.0,
            abs(self.contract.far_range_m),
            abs(self.contract.max_age_s),
        )
        unknown_age = torch.full_like(age_s, self.contract.max_age_s)
        checks = torch.stack(
            (
                torch.isfinite(valid).all(),
                torch.logical_or(valid_mask, invalid_mask).all(),
                torch.logical_or(~valid_mask, torch.isfinite(range_m)).all(),
                torch.logical_or(
                    ~valid_mask,
                    range_m >= self.contract.near_range_m - quantization_tolerance,
                ).all(),
                torch.logical_or(
                    ~valid_mask,
                    range_m <= self.contract.far_range_m + quantization_tolerance,
                ).all(),
                torch.isfinite(age_s).all(),
                (age_s >= -quantization_tolerance).all(),
                (age_s <= self.contract.max_age_s + quantization_tolerance).all(),
                torch.logical_or(
                    ~invalid_mask,
                    torch.isclose(age_s, unknown_age, rtol=0.0, atol=quantization_tolerance),
                ).all(),
            )
        ).detach().cpu().tolist()
        messages = (
            "Teacher valid channel must be finite.",
            "Teacher valid channel must contain exact binary {0,1} values.",
            "Every valid teacher range must be finite.",
            "A valid teacher range is below near_range_m.",
            "A valid teacher range exceeds far_range_m.",
            "Teacher age channel must be finite.",
            "Teacher age is below zero.",
            "Teacher age exceeds max_age_s.",
            "Unknown cells must encode age as max_age_s.",
        )
        for passed, message in zip(checks, messages):
            if not passed:
                raise ValueError(message)

    def _validate_and_normalize_map(self, teacher_map: torch.Tensor) -> torch.Tensor:
        if not isinstance(teacher_map, torch.Tensor):
            raise TypeError("Teacher map must be a tensor.")
        if teacher_map.dtype not in _SUPPORTED_MAP_DTYPES:
            raise ValueError(
                "Teacher map storage dtype must be float16, bfloat16, or float32; "
                f"got {teacher_map.dtype}."
            )
        if teacher_map.ndim != 5 or tuple(teacher_map.shape[1:]) != self.map_shape:
            raise ValueError(f"Teacher map must have shape [B,1,3,16,96], got {tuple(teacher_map.shape)}.")
        if teacher_map.shape[0] <= 0:
            raise ValueError("Teacher map batch dimension must be positive.")

        range_m = teacher_map[:, 0, 0:1]
        valid = teacher_map[:, 0, 1:2]
        age_s = teacher_map[:, 0, 2:3]
        if self.strict_runtime_value_checks:
            self._strict_validate_map_values(range_m, valid, age_s)

        compute_dtype = next(self.map_encoder.parameters()).dtype
        range_m = range_m.to(dtype=compute_dtype)
        valid = valid.to(dtype=compute_dtype)
        age_s = age_s.to(dtype=compute_dtype)
        # The rollout fast path deliberately contains no Tensor->host
        # conversion.  C02 owns production of exact values and C16 preflights
        # that contract; this local sanitization prevents a corrupted value
        # from propagating NaN/Inf through the policy while preserving 50 Hz
        # asynchronous CUDA execution.
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = valid > 0.5
        valid = valid_mask.to(dtype=compute_dtype)
        range_m = torch.nan_to_num(
            range_m,
            nan=self.contract.far_range_m,
            posinf=self.contract.far_range_m,
            neginf=self.contract.near_range_m,
        ).clamp(self.contract.near_range_m, self.contract.far_range_m)
        age_s = torch.nan_to_num(
            age_s,
            nan=self.contract.max_age_s,
            posinf=self.contract.max_age_s,
            neginf=0.0,
        ).clamp(0.0, self.contract.max_age_s)
        sanitized_range = torch.where(
            valid_mask,
            range_m,
            torch.full_like(range_m, self.contract.far_range_m),
        )
        sanitized_age = torch.where(
            valid_mask,
            age_s,
            torch.full_like(age_s, self.contract.max_age_s),
        )
        range_unit = (sanitized_range - self.contract.near_range_m) / (
            self.contract.far_range_m - self.contract.near_range_m
        )
        age_unit = sanitized_age / self.contract.max_age_s
        return torch.cat((2.0 * range_unit - 1.0, 2.0 * valid - 1.0, 2.0 * age_unit - 1.0), dim=1)

    def predict_latent(self, obs: TensorDict) -> torch.Tensor:
        if self.teacher_map_set not in obs:
            raise KeyError(
                f"Observation is missing causal teacher map {self.teacher_map_set!r}; "
                f"available groups={list(obs.keys())}."
            )
        normalized_map = self._validate_and_normalize_map(obs[self.teacher_map_set])
        latent_a = self.map_encoder(normalized_map)
        if latent_a.shape[-1] != self.latent_dim:
            raise RuntimeError(f"Formal teacher produced invalid A shape {tuple(latent_a.shape)}.")
        return latent_a

    def predict_latent_and_action_mean(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        latent_a = self.predict_latent(obs)
        proprio_features = self.ecmm_core.encode_proprio(obs)
        return latent_a, self.ecmm_core.action_mean_from_A(proprio_features, latent_a)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if hidden_state is not None:
            raise ValueError("The formal observed-history teacher is non-recurrent and rejects hidden_state.")
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent_a = self.predict_latent(obs)
        proprio_features = self.ecmm_core.encode_proprio(obs)
        raw_action = self.ecmm_core.actor.mlp(torch.cat((proprio_features, latent_a), dim=-1))
        if stochastic_output:
            self.distribution.update(raw_action)
            return self.distribution.sample()
        return self.distribution.deterministic_output(raw_action)

    def get_actions(self, obs: TensorDict) -> torch.Tensor:
        """Compatibility alias returning stochastic policy actions."""
        return self(obs, stochastic_output=True)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        """Compatibility alias returning the deterministic action mean."""
        return self(obs, stochastic_output=False)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: None = None) -> None:
        del dones
        if hidden_state is not None:
            raise ValueError("The formal observed-history teacher has no hidden state.")

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    @property
    def mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_mean(self) -> torch.Tensor:
        return self.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.entropy

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

    def _config_receipt(self) -> dict[str, Any]:
        return {
            "teacher_map_set": self.teacher_map_set,
            "proprio_sets": list(self.proprio_sets),
            "proprio_group_dims": dict(self.proprio_group_dims),
            "map_contract": self.contract.audit(),
            "map_encoder": self.map_encoder.architecture_receipt(),
            "strict_runtime_value_checks": self.strict_runtime_value_checks,
            "latent_a_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "frozen_ecmm": {
                "checkpoint_sha256": self.ecmm_core.checkpoint_sha256,
                "actor_state_dict_key": self.ecmm_core.actor_state_dict_key,
                "checkpoint_path_source": "external_constructor_configuration",
                "weights_in_teacher_checkpoint": False,
            },
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the complete map-only C06 checkpoint payload."""
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.map_encoder.state_dict().items()
        }
        return {
            "schema": _MAP_CHECKPOINT_SCHEMA,
            "config_receipt": copy.deepcopy(self._config_receipt()),
            "map_encoder_state_dict": state,
        }

    def load_checkpoint_state(self, checkpoint: Mapping[str, Any]) -> None:
        """Load a map-only checkpoint after strict architecture/artifact checks."""
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Formal-teacher checkpoint must be a mapping.")
        expected_keys = {"schema", "config_receipt", "map_encoder_state_dict"}
        if set(checkpoint) != expected_keys:
            raise ValueError(
                "Formal-teacher checkpoint keys differ: "
                f"expected={sorted(expected_keys)}, actual={sorted(str(key) for key in checkpoint)}."
            )
        if checkpoint["schema"] != _MAP_CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported formal-teacher checkpoint schema {checkpoint['schema']!r}.")
        if checkpoint["config_receipt"] != self._config_receipt():
            raise ValueError("Formal-teacher checkpoint config/frozen-artifact receipt differs.")
        saved_state = checkpoint["map_encoder_state_dict"]
        if not isinstance(saved_state, Mapping):
            raise ValueError("map_encoder_state_dict must be a mapping.")
        expected_state = self.map_encoder.state_dict()
        if set(saved_state) != set(expected_state):
            raise ValueError("Formal-teacher map encoder state keys differ.")
        for key, expected_value in expected_state.items():
            saved_value = saved_state[key]
            if not isinstance(saved_value, torch.Tensor):
                raise ValueError(f"Map encoder state {key!r} must be a tensor.")
            if saved_value.shape != expected_value.shape or saved_value.dtype != expected_value.dtype:
                raise ValueError(
                    f"Map encoder state {key!r} shape/dtype differs: "
                    f"checkpoint={tuple(saved_value.shape)}/{saved_value.dtype}, "
                    f"model={tuple(expected_value.shape)}/{expected_value.dtype}."
                )
        self.map_encoder.load_state_dict(saved_state, strict=True)

    def parameter_audit(self) -> dict[str, Any]:
        trainable_names = [name for name, value in self.named_parameters() if value.requires_grad]
        unexpected = [name for name in trainable_names if not name.startswith("map_encoder.")]
        batch_norms = [
            module
            for module in self.ecmm_core.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        return {
            "phase": "phase1_formal_observed_history_teacher",
            "actor_inputs": {
                "ordered_groups": list(self.obs_groups),
                "proprio_sets": list(self.proprio_sets),
                "teacher_map_set": self.teacher_map_set,
                "teacher_map_shape": list(self.map_shape),
                "teacher_map_channels": ["range_m", "valid", "age_s"],
                "uses_future_frames": False,
                "uses_m90_observation": False,
                "uses_terrain_mesh": False,
                "uses_synthetic_fill": False,
                "strict_runtime_value_checks": self.strict_runtime_value_checks,
                "fast_path_value_contract_owner": "C02_producer_and_C16_preflight",
                "fast_path_nonfinite_policy": "gpu_side_sanitize_without_host_sync",
            },
            "components": {
                "map_encoder": _parameter_counts(self.map_encoder),
                "frozen_ecmm": _parameter_counts(self.ecmm_core),
            },
            "trainable_parameter_names": trainable_names,
            "unexpected_trainable_parameter_names": unexpected,
            "only_map_encoder_trainable": bool(trainable_names) and not unexpected,
            "frozen_ecmm_batch_norm": {
                "count": len(batch_norms),
                "training_count": sum(int(module.training) for module in batch_norms),
            },
            "map_contract": self.contract.audit(),
            "map_encoder_architecture": self.map_encoder.architecture_receipt(),
            "checkpoint_contract": {
                "schema": _MAP_CHECKPOINT_SCHEMA,
                "saved_components": ["map_encoder", "config_receipt"],
                "frozen_ecmm_weights_saved": False,
                "frozen_ecmm_path_and_hash_source": "external_constructor_configuration",
            },
            "frozen_ecmm": self.ecmm_core.parameter_audit(),
        }


__all__ = ["M2MObservedHistoryFormalTeacher", "M2MObservedHistoryMapEncoder"]
