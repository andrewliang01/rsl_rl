# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online map-to-memory latent/action distillation core.

This module intentionally implements the algorithm core, not the runner-side
configuration factory.  During rollout the privileged teacher map is consumed
only transiently under ``no_grad``.  Persistent C10 storage receives a
deployable student observation subset plus detached ``A64`` and action-mean
labels; it has no field in which a teacher map could be retained.
"""

from __future__ import annotations

import copy
import json
import math
import re
import torch
import torch.nn as nn
from collections.abc import Mapping
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms.m2m_distillation_loss import (
    M2MDistillationLossConfig,
    M2MMaskedLatentActionLoss,
)
from rsl_rl.storage import M2MSequenceBatch, M2MSequenceRolloutStorage, M2MSequenceTransition
from rsl_rl.utils import resolve_optimizer

_CHECKPOINT_SCHEMA = "m2m_latent_action_distillation_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TRAINABLE_PREFIXES = (
    "frame_tokenizer.",
    "gru.",
    "current_encoder.",
    "latent_head.",
)


@dataclass
class _PendingM2MTransition:
    student_observations: TensorDict
    teacher_latent_A: torch.Tensor  # noqa: N815 - paper/contract symbol is A
    teacher_action_mean: torch.Tensor
    student_hidden_state: torch.Tensor


def _exact_positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}.")
    return value


def _exact_positive_float_or_none(name: str, value: object) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be None or an exact finite float > 0, got {value!r}.")
    return value


class M2MLatentActionDistillation:
    """C11 sequence-distillation algorithm core for a C07 map-free student.

    A later runner factory must construct the student, frozen teacher and C10
    storage from experiment configuration.  Keeping construction out of this
    class prevents a generic runner from silently falling back to legacy
    ``RolloutStorage``, which would persist complete observations including a
    privileged teacher map.
    """

    teacher_loaded: bool = True
    rnd: None = None
    intrinsic_rewards: None = None

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        storage: M2MSequenceRolloutStorage,
        *,
        loss_config: M2MDistillationLossConfig,
        frozen_artifact_receipt: Mapping[str, Any] | None = None,
        learning_rate: float = 1.0e-3,
        optimizer: str = "adam",
        num_learning_epochs: int = 1,
        num_mini_batches: int = 4,
        sequence_length: int = 8,
        max_grad_norm: float | None = 1.0,
        rollout_action_source: str = "student_sample",
        strict_teacher_label_checks: bool = False,
        shuffle_sequences: bool = True,
        sequence_seed: int | None = None,
        device: str | torch.device = "cpu",
        rnd_cfg: None = None,
        symmetry_cfg: None = None,
        multi_gpu_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the strict sequence-distillation training core."""
        if not isinstance(student, nn.Module) or not isinstance(teacher, nn.Module):
            raise TypeError("student and teacher must be torch modules.")
        if not isinstance(storage, M2MSequenceRolloutStorage):
            raise TypeError("storage must be M2MSequenceRolloutStorage; legacy RolloutStorage is forbidden.")
        if not isinstance(loss_config, M2MDistillationLossConfig):
            raise TypeError("loss_config must be M2MDistillationLossConfig.")
        if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be an exact finite float > 0, got {learning_rate!r}.")
        if rollout_action_source not in {"student_sample", "student_mean", "teacher_mean"}:
            raise ValueError(
                "rollout_action_source must be 'student_sample', 'student_mean', or 'teacher_mean', got "
                f"{rollout_action_source!r}."
            )
        if type(strict_teacher_label_checks) is not bool:
            raise ValueError("strict_teacher_label_checks must be an explicit bool.")
        if type(shuffle_sequences) is not bool:
            raise ValueError("shuffle_sequences must be an explicit bool.")
        if sequence_seed is not None and type(sequence_seed) is not int:
            raise ValueError("sequence_seed must be an exact integer or None.")
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise ValueError("C11 is incompatible with RND and symmetry extensions.")
        if multi_gpu_cfg is not None:
            raise NotImplementedError("C11 distributed gradient synchronization is not implemented or claimed.")

        self.device = torch.device(device)
        # Runtime interface checks below are stricter than nn.Module's static
        # surface; keep dynamic runner-compatible actor methods type-visible.
        self.student: Any = student.to(self.device)
        self.teacher: Any = teacher.to(self.device)
        self.storage = storage
        if self.storage.device != self.device:
            raise ValueError(
                f"C11 storage and algorithm devices differ: {self.storage.device} and {self.device}."
            )

        self.loss_config = loss_config
        self.loss_fn = M2MMaskedLatentActionLoss(loss_config)
        self.learning_rate = learning_rate
        self.num_learning_epochs = _exact_positive_int("num_learning_epochs", num_learning_epochs)
        self.num_mini_batches = _exact_positive_int("num_mini_batches", num_mini_batches)
        self.sequence_length = _exact_positive_int("sequence_length", sequence_length)
        self.max_grad_norm = _exact_positive_float_or_none("max_grad_norm", max_grad_norm)
        self.rollout_action_source = rollout_action_source
        self.strict_teacher_label_checks = strict_teacher_label_checks
        self.shuffle_sequences = shuffle_sequences
        self.sequence_seed = sequence_seed
        self.optimizer_name = optimizer

        self._validate_student_interface()
        self.student_architecture_receipt = self._resolve_student_architecture_receipt()
        # A C06 formal teacher becomes immutable for C11.  A shared ECMM core
        # is already frozen; freezing the whole teacher also closes the map
        # encoder gradient path during student distillation.
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self._trainable_parameters = self._validate_gradient_boundary()
        optimizer_factory: Any = resolve_optimizer(optimizer)
        self.optimizer = optimizer_factory(self._trainable_parameters, lr=learning_rate)
        self.frozen_artifact_receipt = self._resolve_frozen_artifact_receipt(frozen_artifact_receipt)

        self._student_key_by_path = {
            self._key_path(key): key
            for key in self.storage.student_observations.keys(include_nested=True, leaves_only=True)
        }
        if set(self._student_key_by_path) != set(self.storage.allowed_student_keys):
            raise RuntimeError("C10 storage key receipt is internally inconsistent.")
        self._pending: _PendingM2MTransition | None = None
        self.num_updates = 0
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _key_path(key: str | tuple[str, ...]) -> str:
        return ".".join(key) if isinstance(key, tuple) else key

    def _validate_student_interface(self) -> None:
        required = (
            "forward_with_latent",
            "predict_padded_latent_and_action_mean",
            "predict_latent_and_action_mean",
            "get_hidden_state",
            "reset",
            "detach_hidden_state",
            "architecture_receipt",
        )
        missing = [name for name in required if not callable(getattr(self.student, name, None))]
        if missing:
            raise TypeError(f"C11 student is missing required C07 interfaces: {missing}.")
        if not hasattr(self.student, "ecmm_core"):
            raise TypeError("C11 student must expose its frozen ecmm_core.")
        if len(self.storage.hidden_state_shape) != 2:
            raise ValueError(
                "C11 currently requires GRU-compatible C10 hidden_state_shape=(num_layers,hidden_size)."
            )

    def _resolve_student_architecture_receipt(self) -> dict[str, Any]:
        receipt = self.student.architecture_receipt()
        if type(receipt) is not dict:
            raise TypeError("C07 architecture_receipt() must return an exact dictionary.")
        try:
            json.dumps(receipt, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("C07 architecture receipt must be finite JSON-compatible metadata.") from error
        return copy.deepcopy(receipt)

    def _validate_gradient_boundary(self) -> tuple[nn.Parameter, ...]:
        trainable_names = [name for name, value in self.student.named_parameters() if value.requires_grad]
        unexpected = [name for name in trainable_names if not name.startswith(_TRAINABLE_PREFIXES)]
        if unexpected:
            raise ValueError(
                "C11 rejects trainable parameters outside tokenizer/temporal/A-head: "
                f"{unexpected}."
            )
        if not trainable_names:
            raise ValueError("C11 student has no trainable tokenizer/temporal/A-head parameters.")
        frozen_core_trainable = [
            name for name, value in self.student.ecmm_core.named_parameters() if value.requires_grad
        ]
        if frozen_core_trainable:
            raise ValueError(f"Frozen ECMM core unexpectedly has trainable parameters: {frozen_core_trainable}.")
        teacher_trainable = [name for name, value in self.teacher.named_parameters() if value.requires_grad]
        if teacher_trainable:
            raise RuntimeError(f"Teacher freeze failed for parameters: {teacher_trainable}.")
        parameters = tuple(value for value in self.student.parameters() if value.requires_grad)
        if len({id(value) for value in parameters}) != len(parameters):
            raise ValueError("C11 trainable student parameter list contains shared duplicates.")
        return parameters

    def _resolve_frozen_artifact_receipt(
        self,
        supplied: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        core = self.student.ecmm_core
        if supplied is None:
            digest = getattr(core, "checkpoint_sha256", None)
            state_key = getattr(core, "actor_state_dict_key", None)
            contract = (
                core.parameter_audit().get("contract")
                if callable(getattr(core, "parameter_audit", None))
                else None
            )
            receipt: dict[str, Any] = {
                "checkpoint_sha256": digest,
                "actor_state_dict_key": state_key,
                "contract": contract,
                "checkpoint_path_source": "external_constructor_configuration",
                "checkpoint_bytes_saved": False,
            }
        else:
            if not isinstance(supplied, Mapping):
                raise TypeError("frozen_artifact_receipt must be a mapping or None.")
            receipt = copy.deepcopy(dict(supplied))
        digest = receipt.get("checkpoint_sha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("Frozen artifact receipt requires a lowercase 64-character checkpoint_sha256.")
        state_key = receipt.get("actor_state_dict_key")
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("Frozen artifact receipt requires a non-empty actor_state_dict_key.")
        if "checkpoint_bytes_saved" not in receipt or receipt["checkpoint_bytes_saved"] is not False:
            raise ValueError("Frozen artifact receipt must state checkpoint_bytes_saved=False.")
        try:
            json.dumps(receipt, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Frozen artifact receipt must be finite JSON-compatible metadata.") from error
        return receipt

    def _student_observation_subset(self, obs: TensorDict) -> TensorDict:
        if not isinstance(obs, TensorDict) or tuple(obs.batch_size) != (self.storage.num_envs,):
            raise ValueError(
                "Rollout observations must be a TensorDict with batch size "
                f"[{self.storage.num_envs}], got {getattr(obs, 'batch_size', None)}."
            )
        selected = TensorDict({}, batch_size=obs.batch_size, device=obs.device)
        for path in self.storage.allowed_student_keys:
            key = self._student_key_by_path[path]
            if key not in obs:
                raise KeyError(f"Rollout observation is missing deployable student key {path!r}.")
            # Detach immediately and retain only the exact deployment allowlist;
            # the privileged map is never placed in the pending transition.
            selected[key] = obs[key].detach()
        return selected

    def _hidden_before_rollout(self, reference: torch.Tensor) -> torch.Tensor:
        hidden = self.student.get_hidden_state()
        expected = (self.storage.num_envs, *self.storage.hidden_state_shape)
        if hidden is None:
            return torch.zeros(expected, device=self.device, dtype=self.storage.hidden_state_dtype)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("C07 get_hidden_state() must return a tensor or None.")
        expected_layer_first = (
            self.storage.hidden_state_shape[0],
            self.storage.num_envs,
            self.storage.hidden_state_shape[1],
        )
        if tuple(hidden.shape) != expected_layer_first:
            raise ValueError(
                f"C07 rollout hidden must be layer-first {expected_layer_first}, got {tuple(hidden.shape)}."
            )
        if hidden.device != reference.device:
            raise ValueError("C07 rollout hidden and observations must share a device.")
        return hidden.detach().permute(1, 0, 2).to(dtype=self.storage.hidden_state_dtype).contiguous()

    def _teacher_labels(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            predictor = getattr(self.teacher, "predict_latent_and_action_mean", None)
            if callable(predictor):
                result = predictor(obs)
            else:
                labeler = getattr(self.teacher, "teacher_labels", None)
                if not callable(labeler):
                    raise TypeError(
                        "C11 teacher must expose predict_latent_and_action_mean(obs) or teacher_labels(obs)."
                    )
                result = labeler(obs)
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise ValueError("Teacher label interface must return (A64, action29[, diagnostics]).")
        latent_a, action_mean = result[:2]
        expected_latent = (self.storage.num_envs, self.storage.teacher_latent_dim)
        expected_action = (self.storage.num_envs, self.storage.action_dim)
        if not isinstance(latent_a, torch.Tensor) or tuple(latent_a.shape) != expected_latent:
            raise ValueError(f"Teacher A label must have shape {expected_latent}.")
        if not isinstance(action_mean, torch.Tensor) or tuple(action_mean.shape) != expected_action:
            raise ValueError(f"Teacher action-mean label must have shape {expected_action}.")
        latent_a = latent_a.detach().to(device=self.device, dtype=self.storage.target_dtype)
        action_mean = action_mean.detach().to(device=self.device, dtype=self.storage.target_dtype)
        if self.strict_teacher_label_checks:
            # This opt-in diagnostic intentionally synchronizes device state
            # to fail closed.  It is unsuitable for the default high-rate
            # rollout path and is recorded in the checkpoint receipt.
            labels_finite = torch.stack(
                (torch.isfinite(latent_a).all(), torch.isfinite(action_mean).all())
            )
            if not bool(labels_finite.all().item()):
                raise FloatingPointError("Teacher produced non-finite C11 labels.")
        else:
            # C11 owns the online label collection boundary.  Keep its default
            # hot path entirely on device and deterministically neutralize a
            # corrupt teacher output without Tensor.item/cpu/tolist/bool.
            latent_a = torch.nan_to_num(latent_a, nan=0.0, posinf=0.0, neginf=0.0)
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return latent_a, action_mean

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Advance the student once and stage one map-free C10 transition."""
        if self._pending is not None:
            raise RuntimeError("act() called twice without process_env_step(); pending transition would be lost.")
        student_observations = self._student_observation_subset(obs)
        reference = next(iter(student_observations.values()))
        if not isinstance(reference, torch.Tensor):
            raise TypeError("C11 deployment observations must have tensor leaves.")
        hidden_before = self._hidden_before_rollout(reference)
        teacher_latent, teacher_action_mean = self._teacher_labels(obs)

        stochastic = self.rollout_action_source == "student_sample"
        student_action, student_latent = self.student.forward_with_latent(
            student_observations,
            stochastic_output=stochastic,
        )
        if tuple(student_action.shape) != (self.storage.num_envs, self.storage.action_dim):
            raise ValueError(f"Student rollout action has invalid shape {tuple(student_action.shape)}.")
        if tuple(student_latent.shape) != (self.storage.num_envs, self.storage.teacher_latent_dim):
            raise ValueError(f"Student rollout latent has invalid shape {tuple(student_latent.shape)}.")

        self._pending = _PendingM2MTransition(
            student_observations=student_observations,
            teacher_latent_A=teacher_latent,
            teacher_action_mean=teacher_action_mean,
            student_hidden_state=hidden_before,
        )
        if self.rollout_action_source == "teacher_mean":
            return teacher_action_mean
        return student_action.detach()

    @torch.no_grad()
    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping[str, Any],
    ) -> None:
        """Commit staged labels and reset only terminated student states."""
        del obs, rewards, extras
        if self._pending is None:
            raise RuntimeError("process_env_step() called without a preceding act().")
        pending = self._pending
        self.storage.add_transition(
            M2MSequenceTransition(
                student_observations=pending.student_observations,
                teacher_latent_A=pending.teacher_latent_A,
                teacher_action_mean=pending.teacher_action_mean,
                dones=dones,
                student_hidden_state=pending.student_hidden_state,
            )
        )
        self._pending = None
        self.student.reset(dones)
        self.student.detach_hidden_state()
        teacher_reset = getattr(self.teacher, "reset", None)
        if callable(teacher_reset):
            teacher_reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op: C11 has supervised sequence targets, not return targets."""
        del obs

    def _predict_batch(self, batch: M2MSequenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        masks = batch.masks
        if bool(getattr(self.student, "is_recurrent", True)):
            return self.student.predict_padded_latent_and_action_mean(
                batch.student_observations,
                masks,
                batch.gru_initial_hidden_state(),
            )
        # Current-frame C07 mode can process the padded tensor directly.  It
        # must not call RSL-RL's trajectory-unpad helper because C10 chunks may
        # have arbitrary valid-prefix lengths.
        return self.student.predict_latent_and_action_mean(batch.student_observations)

    def _assert_gradient_whitelist(self) -> None:
        unexpected = [
            name
            for name, value in self.student.named_parameters()
            if value.grad is not None and not name.startswith(_TRAINABLE_PREFIXES)
        ]
        teacher_gradients = [name for name, value in self.teacher.named_parameters() if value.grad is not None]
        if unexpected or teacher_gradients:
            raise RuntimeError(
                "C11 gradient boundary violation: "
                f"student={unexpected}, teacher={teacher_gradients}."
            )

    def update(self) -> dict[str, float]:
        """Optimize C07 from C10 padded chunks and clear the consumed rollout."""
        if self._pending is not None:
            raise RuntimeError("Cannot update while an act() transition is still pending.")
        self.train_mode()
        weighted_metrics = {
            "loss": 0.0,
            "latent_smooth_l1": 0.0,
            "latent_cosine": 0.0,
            "action_mean_mse": 0.0,
        }
        valid_total = 0.0
        grad_norm_total = 0.0
        batch_count = 0
        seed = None if self.sequence_seed is None else self.sequence_seed + self.num_updates
        generator = self.storage.sequence_mini_batch_generator(
            num_mini_batches=self.num_mini_batches,
            sequence_length=self.sequence_length,
            num_epochs=self.num_learning_epochs,
            shuffle=self.shuffle_sequences,
            seed=seed,
            restore_observation_dtypes=False,
        )
        for batch in generator:
            student_latent, student_action_mean = self._predict_batch(batch)
            total, components = self.loss_fn(
                student_latent,
                batch.teacher_latent_A,
                student_action_mean,
                batch.teacher_action_mean,
                batch.masks,
            )
            if not torch.isfinite(total):
                raise FloatingPointError("C11 distillation loss is not finite.")
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self._assert_gradient_whitelist()
            gradients = [value.grad for value in self._trainable_parameters if value.grad is not None]
            if not gradients or any(not torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError("C11 trainable gradients are missing or non-finite.")
            if self.max_grad_norm is None:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(value.detach()) for value in gradients])
                )
            else:
                grad_norm = nn.utils.clip_grad_norm_(self._trainable_parameters, self.max_grad_norm)
            self.optimizer.step()

            valid_count = float(components["valid_steps"].detach().item())
            valid_total += valid_count
            weighted_metrics["loss"] += float(total.detach().item()) * valid_count
            for name in ("latent_smooth_l1", "latent_cosine", "action_mean_mse"):
                weighted_metrics[name] += float(components[name].detach().item()) * valid_count
            grad_norm_total += float(grad_norm.detach().item())
            batch_count += 1

        if batch_count == 0 or valid_total <= 0.0:
            raise RuntimeError("C11 storage yielded no valid optimization batches.")
        self.storage.clear()
        self.num_updates += 1
        metrics = {name: value / valid_total for name, value in weighted_metrics.items()}
        metrics.update(
            {
                "valid_steps": valid_total,
                "gradient_norm": grad_norm_total / batch_count,
                "optimizer_steps": float(batch_count),
                "algorithm_updates": float(self.num_updates),
            }
        )
        self.last_metrics = metrics
        return metrics

    def train_mode(self) -> None:
        """Enable student training while keeping the teacher frozen in eval mode."""
        self.student.train()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Put both student and teacher into evaluation mode."""
        self.student.eval()
        self.teacher.eval()

    def get_policy(self) -> nn.Module:
        """Return the deployable student policy used by the runner."""
        return self.student

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return the exact C11 gradient whitelist in module traversal order."""
        return tuple(name for name, value in self.student.named_parameters() if value.requires_grad)

    def _configuration_receipt(self) -> dict[str, Any]:
        return {
            "algorithm": type(self).__name__,
            "loss": self.loss_config.receipt(),
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer_name,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "sequence_length": self.sequence_length,
            "max_grad_norm": self.max_grad_norm,
            "rollout_action_source": self.rollout_action_source,
            "teacher_label_integrity": {
                "owner": "C11_online_collection_boundary",
                "strict_device_sync_checks": self.strict_teacher_label_checks,
                "default_hot_path_host_sync": False,
                "nonfinite_sanitization": "torch.nan_to_num(nan=0,posinf=0,neginf=0)",
                "strict_mode": "fail_closed_before_sanitization",
            },
            "shuffle_sequences": self.shuffle_sequences,
            "sequence_seed": self.sequence_seed,
            "student": {
                "class": f"{type(self.student).__module__}.{type(self.student).__qualname__}",
                "is_recurrent": bool(getattr(self.student, "is_recurrent", False)),
                "temporal_mode": getattr(self.student, "temporal_mode", None),
                "trainable_parameter_names": list(self.trainable_parameter_names),
                "allowed_observation_keys": list(self.storage.allowed_student_keys),
                "hidden_state_shape": list(self.storage.hidden_state_shape),
                "architecture_receipt": copy.deepcopy(self.student_architecture_receipt),
            },
            "storage": {
                "class": type(self.storage).__name__,
                "target_dtype": str(self.storage.target_dtype),
                "hidden_state_dtype": str(self.storage.hidden_state_dtype),
                "observation_storage": self.storage.observation_storage_audit(),
                "teacher_map_field": False,
                "next_observation_field": False,
            },
        }

    def save(self) -> dict[str, Any]:
        """Serialize trainable student state only, never teacher/map weights."""
        trainable_state = {
            name: value.detach().cpu().clone()
            for name, value in self.student.named_parameters()
            if value.requires_grad
        }
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "config_receipt": copy.deepcopy(self._configuration_receipt()),
            "student_trainable_state_dict": trainable_state,
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "algorithm_iteration": self.num_updates,
            "frozen_artifact_receipt": copy.deepcopy(self.frozen_artifact_receipt),
        }

    def load(self, loaded_dict: dict[str, Any], load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore an exact C11 payload after config/artifact/state checks."""
        if strict is not True:
            raise ValueError("C11 checkpoint loading is always strict.")
        if load_cfg not in (None, {"student": True, "optimizer": True, "iteration": True}):
            raise ValueError("C11 rejects partial load_cfg; student, optimizer and iteration are atomic.")
        if not isinstance(loaded_dict, dict):
            raise TypeError("C11 checkpoint root must be an exact dict.")
        required = {
            "schema",
            "config_receipt",
            "student_trainable_state_dict",
            "optimizer_state_dict",
            "algorithm_iteration",
            "frozen_artifact_receipt",
        }
        allowed_extras = {"iter", "infos", "training_receipt"}
        missing = required.difference(loaded_dict)
        unexpected = set(loaded_dict).difference(required | allowed_extras)
        if missing or unexpected:
            raise ValueError(
                f"C11 checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        if loaded_dict["schema"] != _CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported C11 checkpoint schema {loaded_dict['schema']!r}.")
        if loaded_dict["config_receipt"] != self._configuration_receipt():
            raise ValueError("C11 checkpoint configuration receipt differs from the live algorithm.")
        if loaded_dict["frozen_artifact_receipt"] != self.frozen_artifact_receipt:
            raise ValueError("C11 checkpoint frozen-artifact receipt differs from the live algorithm.")
        iteration = loaded_dict["algorithm_iteration"]
        if type(iteration) is not int or iteration < 0:
            raise ValueError("C11 algorithm_iteration must be an exact non-negative integer.")

        state = loaded_dict["student_trainable_state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("student_trainable_state_dict must be a mapping.")
        live = {name: value for name, value in self.student.named_parameters() if value.requires_grad}
        if set(state) != set(live):
            raise ValueError("C11 student trainable-state keys differ from the live gradient whitelist.")
        for name, parameter in live.items():
            saved = state[name]
            if not isinstance(saved, torch.Tensor):
                raise ValueError(f"Saved trainable parameter {name!r} must be a tensor.")
            if saved.shape != parameter.shape or saved.dtype != parameter.dtype:
                raise ValueError(
                    f"Saved trainable parameter {name!r} shape/dtype differs from the live model."
                )
        with torch.no_grad():
            for name, parameter in live.items():
                parameter.copy_(state[name].to(device=parameter.device))
        self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.num_updates = iteration
        self.teacher.eval()
        self._validate_gradient_boundary()
        return True

    def audit(self) -> dict[str, Any]:
        """Return the C11 gradient, storage and artifact contract."""
        return {
            "phase": "C11_m2m_latent_action_distillation_core",
            "runner_integration_owner": "M2MDistillationRunner",
            "algorithm_core_runner_agnostic": True,
            "configuration": self._configuration_receipt(),
            "frozen_artifact": copy.deepcopy(self.frozen_artifact_receipt),
            "gradient_boundary": {
                "allowed_prefixes": list(_TRAINABLE_PREFIXES),
                "trainable_parameter_names": list(self.trainable_parameter_names),
                "frozen_ecmm_trainable_count": sum(
                    value.numel() for value in self.student.ecmm_core.parameters() if value.requires_grad
                ),
                "teacher_trainable_count": sum(
                    value.numel() for value in self.teacher.parameters() if value.requires_grad
                ),
            },
            "teacher_label_integrity": {
                "owner": "C11_online_collection_boundary",
                "strict_device_sync_checks": self.strict_teacher_label_checks,
                "default_hot_path_host_sync": False,
                "default_nonfinite_replacement": 0.0,
                "strict_mode_fail_closed": True,
            },
            "rollout_storage": {
                "allowed_student_keys": list(self.storage.allowed_student_keys),
                "stores_teacher_latent_A64": True,
                "stores_teacher_action_mean29": True,
                "stores_teacher_map": False,
                "stores_next_observation": False,
            },
            "checkpoint": {
                "schema": _CHECKPOINT_SCHEMA,
                "student_trainable_state_only": True,
                "optimizer_and_iteration": True,
                "teacher_state_saved": False,
                "teacher_map_saved": False,
                "frozen_checkpoint_bytes_saved": False,
            },
            "last_metrics": dict(self.last_metrics),
        }


__all__ = ["M2MLatentActionDistillation"]
