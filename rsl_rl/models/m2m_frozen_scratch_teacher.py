# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integrity-bound frozen control interface from a scratch M2M teacher.

The artifact is an ordinary PPO checkpoint produced by F07/F19-style
``M2MObservedHistoryScratchTeacher`` actors.  The whole actor is loaded and
frozen so online labels use the exact privileged A/B/C policy.  A map-free
student shares the same frozen normalizer, proprio encoder B, fusion head C,
and action distribution; only its replacement A encoder is trainable.
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

from rsl_rl.models.m2m_observed_history_scratch_teacher import (
    M2MObservedHistoryScratchTeacher,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_actor_state(
    checkpoint: object,
    *,
    state_key: str,
    expected: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Scratch-teacher checkpoint root must be a mapping.")
    if state_key not in checkpoint:
        raise KeyError(f"Scratch-teacher checkpoint is missing {state_key!r}.")
    state = checkpoint[state_key]
    if not isinstance(state, Mapping):
        raise TypeError(f"Scratch-teacher checkpoint {state_key!r} must be a mapping.")
    if any(not isinstance(key, str) for key in state):
        raise TypeError("Scratch-teacher actor state keys must be strings.")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("Scratch-teacher actor state values must be tensors.")

    expected_keys = set(expected)
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Scratch-teacher actor state key mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    shape_errors: list[str] = []
    dtype_errors: list[str] = []
    for name, expected_value in expected.items():
        actual_value = state[name]
        if actual_value.shape != expected_value.shape:
            shape_errors.append(
                f"{name}: checkpoint={tuple(actual_value.shape)}, model={tuple(expected_value.shape)}"
            )
        if actual_value.dtype != expected_value.dtype:
            dtype_errors.append(
                f"{name}: checkpoint={actual_value.dtype}, model={expected_value.dtype}"
            )
    if shape_errors:
        raise ValueError("Scratch-teacher actor state shape mismatch: " + "; ".join(shape_errors))
    if dtype_errors:
        raise ValueError("Scratch-teacher actor state dtype mismatch: " + "; ".join(dtype_errors))
    return state  # type: ignore[return-value]


class M2MFrozenScratchTeacherCore(nn.Module):
    """Frozen full teacher plus the shared B/C student interface."""

    latent_dim: int = 64
    proprio_dim: int = 96
    proprio_feature_dim: int = 64
    action_dim: int = 29

    def __init__(
        self,
        actor: M2MObservedHistoryScratchTeacher,
        *,
        checkpoint_path: str | Path,
        expected_sha256: str,
        actor_state_dict_key: str = "actor_state_dict",
    ) -> None:
        super().__init__()
        if not isinstance(actor, M2MObservedHistoryScratchTeacher):
            raise TypeError(
                "Scratch-teacher core requires M2MObservedHistoryScratchTeacher, "
                f"got {type(actor).__name__}."
            )
        if actor.proprio_dim != self.proprio_dim or actor.prop_feature_dim != self.proprio_feature_dim:
            raise ValueError("Scratch teacher must implement proprio B: 96 -> 64.")
        if actor.latent_dim != self.latent_dim or actor.action_dim != self.action_dim:
            raise ValueError("Scratch teacher must implement A64 and 29 Unitree-G1 actions.")
        if actor.distribution.output_dim != self.action_dim:
            raise ValueError("Scratch-teacher distribution must output 29 actions.")

        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Scratch-teacher checkpoint is not a file: {path}")
        if not isinstance(expected_sha256, str) or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256.")
        actual_sha256 = _file_sha256(path)
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise ValueError(
                "Scratch-teacher checkpoint SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}."
            )
        if not isinstance(actor_state_dict_key, str) or not actor_state_dict_key:
            raise ValueError("actor_state_dict_key must be a non-empty string.")

        # Integrity validation deliberately precedes weights-only deserialization.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = _strict_actor_state(
            checkpoint,
            state_key=actor_state_dict_key,
            expected=actor.state_dict(),
        )
        actor.load_state_dict(state, strict=True)

        self.actor = actor
        self.checkpoint_path = str(path.resolve())
        self.checkpoint_sha256 = actual_sha256
        self.actor_state_dict_key = actor_state_dict_key
        self.teacher_loaded = True
        self._freeze_actor()

    def _freeze_actor(self) -> None:
        self.actor.requires_grad_(False)
        self.actor.eval()

    def train(self, mode: bool = True) -> "M2MFrozenScratchTeacherCore":
        super().train(mode)
        self.actor.eval()
        return self

    def encode_proprio(self, obs: TensorDict) -> torch.Tensor:
        features = self.actor.encode_proprio(obs)
        if features.shape[-1] != self.proprio_feature_dim:
            raise RuntimeError(f"Frozen scratch B produced invalid shape {tuple(features.shape)}.")
        return features

    def encode_teacher_A(self, obs: TensorDict) -> torch.Tensor:  # noqa: N802
        latent = self.actor.predict_latent(obs)
        if latent.shape[-1] != self.latent_dim:
            raise RuntimeError(f"Frozen scratch A produced invalid shape {tuple(latent.shape)}.")
        return latent

    def action_mean_from_A(
        self,
        proprio_features: torch.Tensor,
        latent_a: torch.Tensor,
    ) -> torch.Tensor:
        raw_action = self.actor.mlp(torch.cat((proprio_features, latent_a), dim=-1))
        return self.actor.distribution.deterministic_output(raw_action)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(obs, stochastic_output=False)

    def parameter_audit(self) -> dict[str, Any]:
        architecture = self.actor.architecture_receipt()
        return {
            "phase": "scratch_teacher_full_actor_frozen_for_m2m_student",
            "artifact_kind": "ordinary_ppo_full_actor_state_dict",
            "checkpoint_sha256": self.checkpoint_sha256,
            "actor_state_dict_key": self.actor_state_dict_key,
            "contract": {
                "teacher_class": f"{type(self.actor).__module__}.{type(self.actor).__qualname__}",
                "proprio_dim": self.proprio_dim,
                "proprio_feature_dim": self.proprio_feature_dim,
                "latent_a_dim": self.latent_dim,
                "action_dim": self.action_dim,
                "map_contract": architecture["map_contract"],
                "control_architecture": {
                    "prop_feature_dim": architecture["prop_feature_dim"],
                    "prop_hidden_dims": architecture["prop_hidden_dims"],
                    "fusion_hidden_dims": architecture["fusion_hidden_dims"],
                    "activation": architecture["activation"],
                    "obs_normalization": architecture["obs_normalization"],
                    "distribution": architecture["distribution"],
                },
            },
            "all_parameters_frozen": not any(
                parameter.requires_grad for parameter in self.actor.parameters()
            ),
            "actor_eval": not self.actor.training,
            "checkpoint_path_embedded_in_training_checkpoint": False,
        }


__all__ = ["M2MFrozenScratchTeacherCore"]
