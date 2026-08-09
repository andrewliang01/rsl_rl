# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone deployment policy for the map-to-memory MID-360 student.

This module contains no teacher, rolling mapper, terrain interface, or M90
perception encoder.  The policy owns only the current strict MID-360 frame
tokenizer, optional GRU, predicted-A head, frozen proprioception encoder B,
and frozen action head/distribution C required at deployment.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections.abc import Mapping
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.m2m_recurrent_student import M2MStrictFrameTokenizer
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable

_NETWORK_SCHEMA = "m2m_student_only_network_v1"
_FORBIDDEN_INPUT_TOKENS = (
    "teacher",
    "map",
    "pose",
    "terrain",
    "oracle",
    "m90",
    "mesh",
    "future",
    "height_scan",
)


def _exact_positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}.")
    return value


def _exact_finite_float(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}.")
    resolved = float(value)
    if not torch.isfinite(torch.tensor(resolved)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {resolved}.")
    return resolved


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    missing = expected.difference(value)
    unexpected = set(value).difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )


def normalize_m2m_student_network_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a standalone, JSON-compatible network config."""
    if not isinstance(config, Mapping):
        raise TypeError("M2M student-only network config must be a mapping.")
    expected = {
        "schema",
        "obs_set",
        "output_dim",
        "strict_frame_set",
        "proprio_sets",
        "proprio_group_dims",
        "frame_near_range_m",
        "frame_far_range_m",
        "frame_message_period_s",
        "frame_max_age_s",
        "tokenizer_hidden_channels",
        "tokenizer_dim",
        "tokenizer_pooled_spatial_size",
        "temporal_mode",
        "gru_hidden_dim",
        "gru_num_layers",
        "latent_hidden_dim",
        "control",
    }
    _exact_keys(config, expected, label="M2M student-only network config")
    if config["schema"] != _NETWORK_SCHEMA:
        raise ValueError(f"Unsupported M2M student-only network schema {config['schema']!r}.")

    obs_set = config["obs_set"]
    strict_frame_set = config["strict_frame_set"]
    if not isinstance(obs_set, str) or not obs_set:
        raise ValueError("obs_set must be a non-empty string.")
    if not isinstance(strict_frame_set, str) or not strict_frame_set:
        raise ValueError("strict_frame_set must be a non-empty string.")
    proprio_sets_raw = config["proprio_sets"]
    if not isinstance(proprio_sets_raw, (list, tuple)) or not proprio_sets_raw:
        raise ValueError("proprio_sets must be a non-empty list.")
    proprio_sets = [str(group) for group in proprio_sets_raw]
    if any(not group for group in proprio_sets) or len(set(proprio_sets)) != len(proprio_sets):
        raise ValueError("proprio_sets must contain unique non-empty strings.")
    if strict_frame_set in proprio_sets:
        raise ValueError("strict_frame_set cannot also be a proprioception group.")
    for group in [*proprio_sets, strict_frame_set]:
        forbidden = [token for token in _FORBIDDEN_INPUT_TOKENS if token in group.lower()]
        if forbidden:
            raise ValueError(f"Deployment input group {group!r} contains forbidden tokens {forbidden}.")

    dims_raw = config["proprio_group_dims"]
    if not isinstance(dims_raw, Mapping) or set(dims_raw) != set(proprio_sets):
        raise ValueError("proprio_group_dims must match proprio_sets exactly.")
    proprio_group_dims = {
        group: _exact_positive_int(f"proprio_group_dims[{group!r}]", dims_raw[group])
        for group in proprio_sets
    }
    if sum(proprio_group_dims.values()) != 96:
        raise ValueError("M2M frozen B requires exactly 96 ordered proprioception values.")
    output_dim = _exact_positive_int("output_dim", config["output_dim"])
    if output_dim != 29:
        raise ValueError(f"Unitree-G1 student-only action dimension must be 29, got {output_dim}.")

    near = _exact_finite_float("frame_near_range_m", config["frame_near_range_m"], minimum=0.0)
    far = _exact_finite_float("frame_far_range_m", config["frame_far_range_m"])
    period = _exact_finite_float("frame_message_period_s", config["frame_message_period_s"])
    max_age = _exact_finite_float("frame_max_age_s", config["frame_max_age_s"])
    if far <= near or period <= 0.0 or max_age < period:
        raise ValueError("Frame ranges/period/max-age violate the strict MID-360 contract.")

    tokenizer_channels_raw = config["tokenizer_hidden_channels"]
    pooled_raw = config["tokenizer_pooled_spatial_size"]
    if not isinstance(tokenizer_channels_raw, (list, tuple)) or not tokenizer_channels_raw:
        raise ValueError("tokenizer_hidden_channels must be a non-empty list.")
    if not isinstance(pooled_raw, (list, tuple)) or len(pooled_raw) != 2:
        raise ValueError("tokenizer_pooled_spatial_size must contain exactly two integers.")
    tokenizer_channels = [
        _exact_positive_int("tokenizer hidden channel", value) for value in tokenizer_channels_raw
    ]
    pooled = [_exact_positive_int("tokenizer pooled size", value) for value in pooled_raw]
    tokenizer_dim = _exact_positive_int("tokenizer_dim", config["tokenizer_dim"])
    gru_hidden_dim = _exact_positive_int("gru_hidden_dim", config["gru_hidden_dim"])
    gru_num_layers = _exact_positive_int("gru_num_layers", config["gru_num_layers"])
    latent_hidden_dim = _exact_positive_int("latent_hidden_dim", config["latent_hidden_dim"])
    temporal_mode = config["temporal_mode"]
    if temporal_mode not in ("current", "gru"):
        raise ValueError("temporal_mode must be exactly 'current' or 'gru'.")

    control_raw = config["control"]
    if not isinstance(control_raw, Mapping):
        raise ValueError("control must be a mapping.")
    control_expected = {
        "obs_normalization",
        "activation",
        "prop_feature_dim",
        "prop_hidden_dims",
        "use_prop_encoder",
        "vision_feature_dim",
        "fusion_hidden_dims",
        "distribution_cfg",
    }
    _exact_keys(control_raw, control_expected, label="M2M frozen B/C config")
    if type(control_raw["obs_normalization"]) is not bool:
        raise ValueError("control.obs_normalization must be an explicit bool.")
    if type(control_raw["use_prop_encoder"]) is not bool or control_raw["use_prop_encoder"] is not True:
        raise ValueError("M2M standalone policy requires frozen proprio encoder B.")
    activation = control_raw["activation"]
    if not isinstance(activation, str) or not activation:
        raise ValueError("control.activation must be a non-empty string.")
    prop_feature_dim = _exact_positive_int("control.prop_feature_dim", control_raw["prop_feature_dim"])
    vision_feature_dim = _exact_positive_int("control.vision_feature_dim", control_raw["vision_feature_dim"])
    if prop_feature_dim != 64 or vision_feature_dim != 64:
        raise ValueError("Frozen ECMM standalone contract requires B64 and A64.")
    prop_hidden_raw = control_raw["prop_hidden_dims"]
    fusion_hidden_raw = control_raw["fusion_hidden_dims"]
    if not isinstance(prop_hidden_raw, (list, tuple)) or not prop_hidden_raw:
        raise ValueError("control.prop_hidden_dims must be non-empty.")
    if not isinstance(fusion_hidden_raw, (list, tuple)) or not fusion_hidden_raw:
        raise ValueError("control.fusion_hidden_dims must be non-empty.")
    prop_hidden = [_exact_positive_int("prop hidden dim", value) for value in prop_hidden_raw]
    fusion_hidden = [_exact_positive_int("fusion hidden dim", value) for value in fusion_hidden_raw]
    distribution_raw = control_raw["distribution_cfg"]
    if not isinstance(distribution_raw, Mapping) or "class_name" not in distribution_raw:
        raise ValueError("control.distribution_cfg requires class_name.")
    distribution_cfg = copy.deepcopy(dict(distribution_raw))
    class_name = distribution_cfg["class_name"]
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("control.distribution_cfg.class_name must be a non-empty string.")

    return {
        "schema": _NETWORK_SCHEMA,
        "obs_set": obs_set,
        "output_dim": output_dim,
        "strict_frame_set": strict_frame_set,
        "proprio_sets": proprio_sets,
        "proprio_group_dims": proprio_group_dims,
        "frame_near_range_m": near,
        "frame_far_range_m": far,
        "frame_message_period_s": period,
        "frame_max_age_s": max_age,
        "tokenizer_hidden_channels": tokenizer_channels,
        "tokenizer_dim": tokenizer_dim,
        "tokenizer_pooled_spatial_size": pooled,
        "temporal_mode": temporal_mode,
        "gru_hidden_dim": gru_hidden_dim,
        "gru_num_layers": gru_num_layers,
        "latent_hidden_dim": latent_hidden_dim,
        "control": {
            "obs_normalization": control_raw["obs_normalization"],
            "activation": activation,
            "prop_feature_dim": prop_feature_dim,
            "prop_hidden_dims": prop_hidden,
            "use_prop_encoder": True,
            "vision_feature_dim": vision_feature_dim,
            "fusion_hidden_dims": fusion_hidden,
            "distribution_cfg": distribution_cfg,
        },
    }


class M2MStudentOnlyPolicy(nn.Module):
    """Independent recurrent/current policy with no training-only dependency."""

    latent_dim: int = 64
    proprio_dim: int = 96
    action_dim: int = 29

    def __init__(self, network_config: Mapping[str, Any]) -> None:
        """Construct the immutable deployment network from strict artifact metadata."""
        super().__init__()
        config = normalize_m2m_student_network_config(network_config)
        control = config["control"]
        self.network_config = copy.deepcopy(config)
        self.obs_set = config["obs_set"]
        self.strict_frame_set = config["strict_frame_set"]
        self.proprio_sets = tuple(config["proprio_sets"])
        self.proprio_group_dims = dict(config["proprio_group_dims"])
        self.obs_groups = [*self.proprio_sets, self.strict_frame_set]
        self.temporal_mode = config["temporal_mode"]
        self.is_recurrent = self.temporal_mode == "gru"
        self.gru_hidden_dim = config["gru_hidden_dim"]
        self.gru_num_layers = config["gru_num_layers"]

        self.frame_tokenizer = M2MStrictFrameTokenizer(
            near_range_m=config["frame_near_range_m"],
            far_range_m=config["frame_far_range_m"],
            message_period_s=config["frame_message_period_s"],
            max_age_s=config["frame_max_age_s"],
            hidden_channels=config["tokenizer_hidden_channels"],
            token_dim=config["tokenizer_dim"],
            pooled_spatial_size=tuple(config["tokenizer_pooled_spatial_size"]),
        )
        if control["obs_normalization"]:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(self.proprio_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.prop_mlp = MLP(
            self.proprio_dim,
            control["prop_feature_dim"],
            control["prop_hidden_dims"],
            control["activation"],
        )

        distribution_cfg = copy.deepcopy(control["distribution_cfg"])
        distribution_class = resolve_callable(distribution_cfg.pop("class_name"))
        if not isinstance(distribution_class, type) or not issubclass(distribution_class, Distribution):
            raise ValueError("Student-only distribution class must derive from rsl_rl Distribution.")
        self.distribution = distribution_class(self.action_dim, **distribution_cfg)
        self.action_head = MLP(
            control["prop_feature_dim"] + control["vision_feature_dim"],
            self.distribution.input_dim,
            control["fusion_hidden_dims"],
            control["activation"],
        )

        temporal_input_dim = config["tokenizer_dim"] + control["prop_feature_dim"]
        self.temporal_input_dim = temporal_input_dim
        if self.is_recurrent:
            self.gru: nn.GRU | None = nn.GRU(
                temporal_input_dim,
                self.gru_hidden_dim,
                num_layers=self.gru_num_layers,
            )
            self.current_encoder: nn.Module | None = None
        else:
            self.gru = None
            self.current_encoder = nn.Sequential(
                nn.Linear(temporal_input_dim, self.gru_hidden_dim),
                nn.LayerNorm(self.gru_hidden_dim),
                nn.ELU(),
            )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(self.gru_hidden_dim),
            nn.Linear(self.gru_hidden_dim, config["latent_hidden_dim"]),
            nn.ELU(),
            nn.Linear(config["latent_hidden_dim"], self.latent_dim),
        )
        self._hidden_state: torch.Tensor | None = None
        self.artifact_receipt: dict[str, Any] = {}
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> M2MStudentOnlyPolicy:
        """Deployment artifacts are immutable and remain in evaluation mode."""
        del mode
        super().train(False)
        return self

    def _validate_observations(self, obs: TensorDict) -> None:
        if not isinstance(obs, TensorDict) or len(obs.batch_size) != 1 or obs.batch_size[0] <= 0:
            raise ValueError("Student-only observations must be a TensorDict with batch layout [B].")
        # TensorDict iteration yields batch slices rather than mapping keys.
        actual_keys = {str(key) for key in obs.keys()}  # noqa: SIM118
        expected_keys = set(self.obs_groups)
        if actual_keys != expected_keys:
            raise ValueError(
                "Student-only observation keys differ from the deployment allowlist: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}."
            )
        frame = obs[self.strict_frame_set]
        expected_frame = (obs.batch_size[0], *M2MStrictFrameTokenizer.frame_shape)
        if frame.dtype not in (torch.float16, torch.bfloat16, torch.float32) or tuple(frame.shape) != expected_frame:
            raise ValueError(
                f"Strict MID-360 frame must be fp16/bf16/fp32 {expected_frame}, got "
                f"{frame.dtype}/{tuple(frame.shape)}."
            )
        for group, dimension in self.proprio_group_dims.items():
            value = obs[group]
            if value.dtype != torch.float32 or tuple(value.shape) != (obs.batch_size[0], dimension):
                raise ValueError(f"Deployment proprio group {group!r} violates its float32 [B,D] contract.")

    def _tensor_features(self, proprio: torch.Tensor, strict_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.frame_tokenizer(strict_frame)
        proprio_features = self.prop_mlp(self.obs_normalizer(proprio))
        return torch.cat((token, proprio_features), dim=-1), proprio_features

    def step_tensors(
        self,
        proprio: torch.Tensor,
        strict_frame: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run one deterministic 50 Hz step using the tensor deployment API."""
        if proprio.dtype != torch.float32 or proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("Concatenated deployment proprioception must be float32 [B,96].")
        if tuple(strict_frame.shape) != (proprio.shape[0], *M2MStrictFrameTokenizer.frame_shape):
            raise ValueError("Strict frame and proprioception batch/shape contracts differ.")
        temporal_input, proprio_features = self._tensor_features(proprio, strict_frame)
        if self.is_recurrent:
            if self.gru is None:
                raise RuntimeError("GRU deployment policy was not constructed.")
            if hidden_state is None:
                hidden_state = temporal_input.new_zeros(
                    self.gru_num_layers,
                    proprio.shape[0],
                    self.gru_hidden_dim,
                )
            expected_hidden = (self.gru_num_layers, proprio.shape[0], self.gru_hidden_dim)
            if tuple(hidden_state.shape) != expected_hidden:
                raise ValueError(f"GRU hidden state must have shape {expected_hidden}.")
            recurrent_output, next_hidden = self.gru(temporal_input.unsqueeze(0), hidden_state)
            temporal_feature = recurrent_output.squeeze(0)
        else:
            if hidden_state is not None:
                raise ValueError("Current-frame deployment policy rejects hidden_state.")
            if self.current_encoder is None:
                raise RuntimeError("Current-frame deployment encoder was not constructed.")
            temporal_feature = self.current_encoder(temporal_input)
            next_hidden = None
        latent_a = self.latent_head(temporal_feature)
        action = self.distribution.deterministic_output(
            self.action_head(torch.cat((proprio_features, latent_a), dim=-1))
        )
        return action, latent_a, next_hidden

    def forward(self, obs: TensorDict, stochastic_output: bool = False) -> torch.Tensor:
        """Advance one deployment step through the standard actor interface."""
        self._validate_observations(obs)
        proprio = torch.cat([obs[group] for group in self.proprio_sets], dim=-1)
        action, latent_a, next_hidden = self.step_tensors(
            proprio,
            obs[self.strict_frame_set],
            self._hidden_state,
        )
        self._hidden_state = next_hidden
        if stochastic_output:
            # Recompute only the final frozen head to initialize the exact
            # frozen distribution.  Deployment normally uses deterministic C.
            temporal_input, proprio_features = self._tensor_features(proprio, obs[self.strict_frame_set])
            del temporal_input
            raw_action = self.action_head(torch.cat((proprio_features, latent_a), dim=-1))
            self.distribution.update(raw_action)
            return self.distribution.sample()
        return action

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset all recurrent state or only environments marked done."""
        if not self.is_recurrent or self._hidden_state is None:
            self._hidden_state = None
            return
        if dones is None:
            self._hidden_state = None
            return
        done_mask = dones.reshape(-1).to(device=self._hidden_state.device, dtype=torch.bool)
        if done_mask.numel() != self._hidden_state.shape[1]:
            raise ValueError("dones must contain one value per recurrent environment.")
        self._hidden_state = self._hidden_state.masked_fill(done_mask.view(1, -1, 1), 0.0)

    def get_hidden_state(self) -> torch.Tensor | None:
        """Return the current deployment GRU state."""
        return self._hidden_state

    def as_jit(self) -> _M2MStudentOnlyGRUExport | _M2MStudentOnlyCurrentExport:
        """Return a tensor-only wrapper suitable for TorchScript."""
        return _M2MStudentOnlyGRUExport(self) if self.is_recurrent else _M2MStudentOnlyCurrentExport(self)

    def as_onnx(self, verbose: bool = False) -> _M2MStudentOnlyGRUExport | _M2MStudentOnlyCurrentExport:
        """Return a tensor-only wrapper suitable for ONNX export."""
        del verbose
        return self.as_jit()

    def dependency_audit(self) -> dict[str, Any]:
        """Report that the deployment graph contains no training-only components."""
        state_keys = list(self.state_dict())
        module_names = [name for name, _module in self.named_modules()]
        forbidden_tokens = ("teacher", "mapper", "rolling", "elevation_encoder", "height_scan", "m90")
        forbidden_state = [key for key in state_keys if any(token in key.lower() for token in forbidden_tokens)]
        forbidden_modules = [name for name in module_names if any(token in name.lower() for token in forbidden_tokens)]
        return {
            "model": "m2m_student_only_policy",
            "deployment_inputs": list(self.obs_groups),
            "state_keys": state_keys,
            "forbidden_state_keys": forbidden_state,
            "forbidden_module_names": forbidden_modules,
            "constructs_teacher": False,
            "constructs_mapper": False,
            "constructs_teacher_observation": False,
            "contains_m90_perception_encoder": False,
            "external_checkpoint_required_at_runtime": False,
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in self.parameters()),
        }


class _M2MStudentOnlyTensorExportBase(nn.Module):
    """Tensor-only copy of the student deployment path for JIT/ONNX."""

    def __init__(self, policy: M2MStudentOnlyPolicy) -> None:
        super().__init__()
        self.tokenizer_spatial_encoder = copy.deepcopy(policy.frame_tokenizer.spatial_encoder)
        post_height, post_width = M2MStrictFrameTokenizer.frame_shape[-2:]
        for _block in policy.frame_tokenizer.spatial_encoder:
            post_height = (post_height + 1) // 2
            post_width = (post_width + 1) // 2
        pooled_height, pooled_width = policy.frame_tokenizer.pooled_spatial_size
        if post_height % pooled_height != 0 or post_width % pooled_width != 0:
            raise ValueError(
                "Student-only JIT/ONNX export requires the fixed tokenizer feature map to divide "
                "its pooled spatial size exactly."
            )
        # AdaptiveAvgPool2d is exactly equivalent to this fixed pool when the
        # dimensions divide, while the fixed operator preserves ONNX dynamic
        # batch export (spatial dimensions are contractually fixed 16x96).
        self.tokenizer_pool = nn.AvgPool2d(
            kernel_size=(post_height // pooled_height, post_width // pooled_width),
            stride=(post_height // pooled_height, post_width // pooled_width),
        )
        self.tokenizer_projection = copy.deepcopy(policy.frame_tokenizer.projection)
        self.frame_near_range_m = policy.frame_tokenizer.near_range_m
        self.frame_far_range_m = policy.frame_tokenizer.far_range_m
        self.frame_max_age_s = policy.frame_tokenizer.max_age_s
        self.obs_normalizer = copy.deepcopy(policy.obs_normalizer)
        self.prop_mlp = copy.deepcopy(policy.prop_mlp)
        self.latent_head = copy.deepcopy(policy.latent_head)
        self.action_head = copy.deepcopy(policy.action_head)
        self.deterministic_output = copy.deepcopy(policy.distribution.as_deterministic_output_module())
        self.proprio_dim = policy.proprio_dim

    def _features(self, proprio: torch.Tensor, strict_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio_features = self.prop_mlp(self.obs_normalizer(proprio))
        channels = strict_frame.squeeze(1).float()
        range_m = channels[:, 0:1]
        valid = torch.nan_to_num(channels[:, 1:2], nan=0.0).clamp(0.0, 1.0)
        age_s = channels[:, 2:3]
        new_frame = torch.nan_to_num(channels[:, 3:4], nan=0.0).clamp(0.0, 1.0)
        range_m = torch.nan_to_num(
            range_m,
            nan=self.frame_far_range_m,
            posinf=self.frame_far_range_m,
            neginf=self.frame_near_range_m,
        ).clamp(self.frame_near_range_m, self.frame_far_range_m)
        range_unit = (range_m - self.frame_near_range_m) / (
            self.frame_far_range_m - self.frame_near_range_m
        )
        normalized_range = torch.where(
            valid > 0.5,
            2.0 * range_unit - 1.0,
            torch.zeros_like(range_unit),
        )
        normalized_age = torch.nan_to_num(
            age_s,
            nan=self.frame_max_age_s,
            posinf=self.frame_max_age_s,
            neginf=0.0,
        ).clamp(0.0, self.frame_max_age_s) / self.frame_max_age_s
        normalized = torch.cat((normalized_range, valid, normalized_age, new_frame), dim=1)
        token = self.tokenizer_projection(
            self.tokenizer_pool(self.tokenizer_spatial_encoder(normalized))
        )
        return torch.cat((token, proprio_features), dim=-1), proprio_features

    def _action(self, proprio_features: torch.Tensor, latent_a: torch.Tensor) -> torch.Tensor:
        return self.deterministic_output(self.action_head(torch.cat((proprio_features, latent_a), dim=-1)))


class _M2MStudentOnlyGRUExport(_M2MStudentOnlyTensorExportBase):
    def __init__(self, policy: M2MStudentOnlyPolicy) -> None:
        super().__init__(policy)
        if policy.gru is None:
            raise ValueError("GRU export requires a recurrent student-only policy.")
        self.gru = copy.deepcopy(policy.gru)
        self.gru_num_layers = policy.gru_num_layers
        self.gru_hidden_dim = policy.gru_hidden_dim
        self.input_names = ["proprio", "strict_frame", "hidden_state"]
        self.output_names = ["action", "latent_A", "next_hidden_state"]
        self.dynamic_axes = {
            "proprio": {0: "batch"},
            "strict_frame": {0: "batch"},
            "hidden_state": {1: "batch"},
            "action": {0: "batch"},
            "latent_A": {0: "batch"},
            "next_hidden_state": {1: "batch"},
        }

    def forward(
        self,
        proprio: torch.Tensor,
        strict_frame: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_input, proprio_features = self._features(proprio, strict_frame)
        recurrent_output, next_hidden = self.gru(temporal_input.unsqueeze(0), hidden_state)
        latent_a = self.latent_head(recurrent_output.squeeze(0))
        return self._action(proprio_features, latent_a), latent_a, next_hidden

    @torch.jit.unused
    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.proprio_dim),
            torch.zeros(1, *M2MStrictFrameTokenizer.frame_shape),
            torch.zeros(self.gru_num_layers, 1, self.gru_hidden_dim),
        )


class _M2MStudentOnlyCurrentExport(_M2MStudentOnlyTensorExportBase):
    def __init__(self, policy: M2MStudentOnlyPolicy) -> None:
        super().__init__(policy)
        if policy.current_encoder is None:
            raise ValueError("Current export requires a current-frame student-only policy.")
        self.current_encoder = copy.deepcopy(policy.current_encoder)
        self.input_names = ["proprio", "strict_frame"]
        self.output_names = ["action", "latent_A"]
        self.dynamic_axes = {
            "proprio": {0: "batch"},
            "strict_frame": {0: "batch"},
            "action": {0: "batch"},
            "latent_A": {0: "batch"},
        }

    def forward(self, proprio: torch.Tensor, strict_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal_input, proprio_features = self._features(proprio, strict_frame)
        latent_a = self.latent_head(self.current_encoder(temporal_input))
        return self._action(proprio_features, latent_a), latent_a

    @torch.jit.unused
    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.proprio_dim),
            torch.zeros(1, *M2MStrictFrameTokenizer.frame_shape),
        )


__all__ = ["M2MStudentOnlyPolicy", "normalize_m2m_student_network_config"]
