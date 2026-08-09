# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed loading of the frozen M90 ECMM control interface.

The M2M teacher and student both target the perception latent ``A`` consumed by
an already trained ECMM policy.  This module deliberately does not construct a
policy from checkpoint metadata: the caller must construct the exact
``PropMLPElevationFusionModel`` architecture from its experiment config and
must provide both the checkpoint path and its trusted SHA-256 digest.

The resulting core owns a completely frozen actor and exposes only the three
operations needed by M2M:

``encode_proprio`` (B), ``encode_teacher_A`` (A), and
``action_mean_from_A`` (C).  Although C's parameters are frozen, C is not run
under ``no_grad`` so a student latent retains a gradient path through it.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise ValueError(f"Expected at least one Linear layer in {type(module).__name__}.")


def _last_linear(module: nn.Module) -> nn.Linear:
    linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linear_layers:
        raise ValueError(f"Expected at least one Linear layer in {type(module).__name__}.")
    return linear_layers[-1]


def _validate_ecmm_actor_contract(actor: PropMLPElevationFusionModel) -> None:
    """Reject an architecture that cannot be the frozen M90-H1 ECMM actor."""
    if not isinstance(actor, PropMLPElevationFusionModel):
        raise TypeError(
            "M2M frozen control requires PropMLPElevationFusionModel, "
            f"got {type(actor).__name__}."
        )
    if actor.elevation_encoder_type != "cnn":
        raise ValueError("M90 ECMM artifact must use the baseline CNN elevation encoder.")
    if actor.elevation_history_length != 1:
        raise ValueError(
            "M90 ECMM artifact must be the H1 actor; "
            f"got elevation_history_length={actor.elevation_history_length}."
        )
    if not actor.use_prop_encoder:
        raise ValueError("M90 ECMM artifact must contain the proprioception encoder B.")
    if actor.obs_dim != 96 or actor.prop_feature_dim != 64:
        raise ValueError(
            "M90 ECMM proprioception contract is 96 -> 64, got "
            f"{actor.obs_dim} -> {actor.prop_feature_dim}."
        )
    if actor.vision_feature_dim != 64:
        raise ValueError(f"M2M requires A=64, got A={actor.vision_feature_dim}.")
    if actor.distribution is None:
        raise ValueError("M90 ECMM actor must include its action distribution.")
    if actor.distribution.output_dim != 29:
        raise ValueError(
            "M90 Unitree-G1 ECMM action contract requires 29 actions, got "
            f"{actor.distribution.output_dim}."
        )
    fusion_input = _first_linear(actor.mlp).in_features
    if fusion_input != actor.prop_feature_dim + actor.vision_feature_dim:
        raise ValueError(
            "ECMM fusion head input must be concat(B64, A64); "
            f"got {fusion_input} features."
        )
    fusion_output = _last_linear(actor.mlp).out_features
    if fusion_output != actor.distribution.input_dim:
        raise ValueError(
            "ECMM fusion head/distribution shape mismatch: "
            f"head={fusion_output}, distribution input={actor.distribution.input_dim}."
        )


def _extract_strict_actor_state(
    checkpoint: object,
    *,
    actor_state_dict_key: str,
    expected_state: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")
    if actor_state_dict_key not in checkpoint:
        raise KeyError(
            f"Checkpoint is missing required key '{actor_state_dict_key}'. "
            f"Available keys: {sorted(str(key) for key in checkpoint.keys())}."
        )
    state = checkpoint[actor_state_dict_key]
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint '{actor_state_dict_key}' must be a state-dict mapping.")
    if any(not isinstance(key, str) for key in state):
        raise ValueError("Actor state dict keys must all be strings.")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Actor state dict values must all be tensors.")

    expected_keys = set(expected_state)
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Actor state-dict key mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )

    shape_errors = []
    dtype_errors = []
    for key, expected_value in expected_state.items():
        actual_value = state[key]
        if tuple(actual_value.shape) != tuple(expected_value.shape):
            shape_errors.append(
                f"{key}: checkpoint={tuple(actual_value.shape)}, model={tuple(expected_value.shape)}"
            )
        if actual_value.dtype != expected_value.dtype:
            dtype_errors.append(f"{key}: checkpoint={actual_value.dtype}, model={expected_value.dtype}")
    if shape_errors:
        raise ValueError("Actor state-dict shape mismatch: " + "; ".join(shape_errors))
    if dtype_errors:
        raise ValueError("Actor state-dict dtype mismatch: " + "; ".join(dtype_errors))
    return state  # type: ignore[return-value]


class M2MFrozenECMMCore(nn.Module):
    """Frozen, integrity-bound ``B + A -> C`` interface of an M90 ECMM actor."""

    latent_dim: int = 64
    proprio_dim: int = 96
    proprio_feature_dim: int = 64
    action_dim: int = 29

    def __init__(
        self,
        actor: PropMLPElevationFusionModel,
        *,
        checkpoint_path: str | Path,
        expected_sha256: str,
        actor_state_dict_key: str = "actor_state_dict",
    ) -> None:
        super().__init__()
        _validate_ecmm_actor_contract(actor)

        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"M90 ECMM checkpoint is not a file: {path}")
        if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters.")
        expected_digest = expected_sha256.lower()
        actual_digest = _file_sha256(path)
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise ValueError(
                "M90 ECMM checkpoint SHA-256 mismatch: "
                f"expected={expected_digest}, actual={actual_digest}."
            )
        if not isinstance(actor_state_dict_key, str) or not actor_state_dict_key:
            raise ValueError("actor_state_dict_key must be a non-empty string.")

        # Hash validation is intentionally completed before deserialization.  A
        # weights-only load then rejects arbitrary pickle reducers.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = _extract_strict_actor_state(
            checkpoint,
            actor_state_dict_key=actor_state_dict_key,
            expected_state=actor.state_dict(),
        )
        actor.load_state_dict(state, strict=True)

        self.actor = actor
        self.checkpoint_path = str(path.resolve())
        self.checkpoint_sha256 = actual_digest
        self.actor_state_dict_key = actor_state_dict_key
        self.teacher_loaded = True
        self._freeze_actor()

    def _freeze_actor(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.eval()

    def train(self, mode: bool = True) -> M2MFrozenECMMCore:
        """Keep every frozen component, including BatchNorm, in evaluation mode."""
        super().train(mode)
        self.actor.eval()
        return self

    def encode_proprio(self, obs: TensorDict) -> torch.Tensor:
        """Encode the deployable 96-D observation groups into frozen B64."""
        proprio = torch.cat([obs[group] for group in self.actor.obs_groups], dim=-1)
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Proprioception must have final dimension {self.proprio_dim}, got {tuple(proprio.shape)}."
            )
        features = self.actor.prop_mlp(self.actor.obs_normalizer(proprio))
        if features.shape[-1] != self.proprio_feature_dim:
            raise RuntimeError(f"Frozen B produced invalid shape {tuple(features.shape)}.")
        return features

    def encode_teacher_A(self, teacher_obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Encode the M90-H1 observation into the frozen global latent A64."""
        elevation = (
            teacher_obs[self.actor.elevation_set]
            if isinstance(teacher_obs, TensorDict)
            else teacher_obs
        )
        if elevation.ndim != 4:
            raise ValueError(
                "M90 teacher input must have layout [B,1,H,W], "
                f"got {tuple(elevation.shape)}."
            )
        if elevation.shape[1] != self.actor.elevation_history_length:
            raise ValueError(
                "M90 teacher history-channel mismatch: "
                f"expected {self.actor.elevation_history_length}, got {elevation.shape[1]}."
            )
        if tuple(elevation.shape[-2:]) != self.actor.vision_spatial_size:
            raise ValueError(
                "M90 teacher spatial-shape mismatch: "
                f"expected {self.actor.vision_spatial_size}, got {tuple(elevation.shape[-2:])}."
            )
        normalized = self.actor._normalize_cnn_observation(elevation)
        latent_a = self.actor.elevation_encoder(normalized)
        if latent_a.shape[-1] != self.latent_dim:
            raise RuntimeError(f"Frozen M90 encoder produced invalid A shape {tuple(latent_a.shape)}.")
        return latent_a

    def action_mean_from_A(
        self,
        proprio_features: torch.Tensor,
        latent_a: torch.Tensor,
    ) -> torch.Tensor:
        """Apply frozen C while preserving gradients with respect to ``latent_a``."""
        if proprio_features.shape[:-1] != latent_a.shape[:-1]:
            raise ValueError(
                "B and A leading dimensions must match, got "
                f"{tuple(proprio_features.shape)} and {tuple(latent_a.shape)}."
            )
        if proprio_features.shape[-1] != self.proprio_feature_dim:
            raise ValueError(f"B must be 64-D, got {tuple(proprio_features.shape)}.")
        if latent_a.shape[-1] != self.latent_dim:
            raise ValueError(f"A must be 64-D, got {tuple(latent_a.shape)}.")
        raw_action = self.actor.mlp(torch.cat((proprio_features, latent_a), dim=-1))
        action_mean = self.actor.distribution.deterministic_output(raw_action)  # type: ignore[union-attr]
        if action_mean.shape[-1] != self.action_dim:
            raise RuntimeError(f"Frozen C produced invalid action shape {tuple(action_mean.shape)}.")
        return action_mean

    @torch.no_grad()
    def teacher_labels(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached ``(A64, mean_action29)`` labels for online collection."""
        proprio_features = self.encode_proprio(obs)
        latent_a = self.encode_teacher_A(obs)
        action_mean = self.action_mean_from_A(proprio_features, latent_a)
        return latent_a, action_mean

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Return the deterministic frozen teacher action for a complete M90 observation."""
        return self.teacher_labels(obs)[1]

    def parameter_audit(self) -> dict[str, Any]:
        """Return a machine-readable freeze and architecture receipt."""

        def parameter_counts(module: nn.Module) -> dict[str, int]:
            parameters = list(module.parameters())
            return {
                "total": sum(parameter.numel() for parameter in parameters),
                "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            }

        batch_norms = [
            module
            for module in self.actor.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        return {
            "teacher_loaded": self.teacher_loaded,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "actor_state_dict_key": self.actor_state_dict_key,
            "contract": {
                "proprio_dim": self.proprio_dim,
                "proprio_feature_dim": self.proprio_feature_dim,
                "latent_a_dim": self.latent_dim,
                "action_dim": self.action_dim,
                "history_length": self.actor.elevation_history_length,
                "spatial_size": list(self.actor.vision_spatial_size),
                "observation_type": self.actor.cnn_observation_type,
                "depth_camera_near": self.actor.depth_camera_near,
                "depth_camera_far": self.actor.depth_camera_far,
            },
            "components": {
                "obs_normalizer": parameter_counts(self.actor.obs_normalizer),
                "proprio_encoder_B": parameter_counts(self.actor.prop_mlp),
                "teacher_encoder_A": parameter_counts(self.actor.elevation_encoder),
                "fusion_action_head_C": parameter_counts(self.actor.mlp),
                "distribution": parameter_counts(self.actor.distribution),  # type: ignore[arg-type]
                "actor": parameter_counts(self.actor),
            },
            "batch_norm": {
                "count": len(batch_norms),
                "training_count": sum(int(module.training) for module in batch_norms),
            },
        }


def load_frozen_m90_ecmm_core(
    actor: PropMLPElevationFusionModel,
    *,
    checkpoint_path: str | Path,
    expected_sha256: str,
    actor_state_dict_key: str = "actor_state_dict",
) -> M2MFrozenECMMCore:
    """Load and integrity-bind an explicitly constructed M90 ECMM actor."""
    return M2MFrozenECMMCore(
        actor,
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
        actor_state_dict_key=actor_state_dict_key,
    )


__all__ = ["M2MFrozenECMMCore", "load_frozen_m90_ecmm_core"]
