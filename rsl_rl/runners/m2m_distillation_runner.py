# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed runner construction for map-to-memory distillation.

The legacy :class:`DistillationRunner` builds generic student/teacher models
and stores complete observations in ``RolloutStorage``.  That is intentionally
not reused for M2M: a privileged teacher map must never become a persistent
rollout field.  This runner constructs the C07 student, frozen C06 teacher,
C10 sequence storage and C11 algorithm as one checked unit.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from pathlib import Path
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms import M2MDistillationLossConfig, M2MLatentActionDistillation
from rsl_rl.env import VecEnv
from rsl_rl.models import M2MFrozenScratchTeacherCore
from rsl_rl.storage import M2MSequenceRolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger

from .distillation_runner import DistillationRunner

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_ARTIFACT_PATH_KEYS = {
    "checkpoint_path",
    "expected_sha256",
    "resume_path",
    "load_checkpoint",
}
_CONSTRUCTOR_OWNED_KEYS = {"obs", "obs_groups", "output_dim", "shared_ecmm_core"}
_FROZEN_ECMM_INJECTION_KEYS = {
    "frozen_ecmm_checkpoint_path",
    "frozen_ecmm_expected_sha256",
    "frozen_ecmm_actor_state_dict_key",
}


def _mapping_copy(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{field} keys must be non-empty strings.")
    return copy.deepcopy(dict(value))


def _exact_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be an exact positive integer, got {value!r}.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_config(
    value: object,
    *,
    field: str,
    allow_actor_state_key: bool,
    allow_checkpoint_state_key: bool = False,
) -> tuple[Path, str, str | None, str | None]:
    config = _mapping_copy(value, field=field)
    required = {"checkpoint_path", "expected_sha256"}
    optional: set[str] = set()
    if allow_actor_state_key:
        optional.add("actor_state_dict_key")
    if allow_checkpoint_state_key:
        optional.add("checkpoint_state_key")
    missing = required.difference(config)
    unexpected = set(config).difference(required | optional)
    if missing or unexpected:
        raise ValueError(
            f"{field} key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}; "
            f"allowed={sorted(required | optional)}."
        )
    path_value = config["checkpoint_path"]
    if not isinstance(path_value, (str, os.PathLike)):
        raise TypeError(f"{field}.checkpoint_path must be path-like.")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{field} checkpoint is not a file: {path}")
    expected = config["expected_sha256"]
    if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"{field}.expected_sha256 must be a lowercase 64-character SHA-256.")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch: expected={expected}, actual={actual}.")
    state_key = config.get("actor_state_dict_key", "actor_state_dict")
    if allow_actor_state_key and (not isinstance(state_key, str) or not state_key):
        raise ValueError(f"{field}.actor_state_dict_key must be a non-empty string.")
    checkpoint_state_key = config.get("checkpoint_state_key")
    if checkpoint_state_key is not None and (not isinstance(checkpoint_state_key, str) or not checkpoint_state_key):
        raise ValueError(f"{field}.checkpoint_state_key must be a non-empty string or omitted.")
    return (
        path,
        expected,
        state_key if allow_actor_state_key else None,
        checkpoint_state_key if allow_checkpoint_state_key else None,
    )


def _dtype(value: object, *, field: str) -> torch.dtype:
    if not isinstance(value, str) or value not in _DTYPES:
        raise ValueError(f"{field} must be one of {sorted(_DTYPES)}, got {value!r}.")
    return _DTYPES[value]


def _model_class(config: dict[str, Any], *, field: str) -> type[nn.Module]:
    class_name = config.pop("class_name", None)
    if not isinstance(class_name, str) or not class_name:
        raise ValueError(f"{field}.class_name must be an explicit non-empty string.")
    resolved = resolve_callable(class_name)
    if not isinstance(resolved, type) or not issubclass(resolved, nn.Module):
        raise TypeError(f"{field}.class_name must resolve to a torch module class.")
    return resolved


def _artifact_receipt(
    *,
    student: nn.Module,
    frozen_sha256: str,
    actor_state_dict_key: str,
    teacher_sha256: str,
    teacher_schema: str,
    teacher_checkpoint_state_key: str | None,
    teacher_container_schema: str | None,
) -> dict[str, Any]:
    core = getattr(student, "ecmm_core", None)
    if not isinstance(core, nn.Module):
        raise TypeError("M2M student must expose its frozen ecmm_core module.")
    actual_digest = getattr(core, "checkpoint_sha256", None)
    actual_state_key = getattr(core, "actor_state_dict_key", None)
    if actual_digest != frozen_sha256 or actual_state_key != actor_state_dict_key:
        raise ValueError("Student frozen ECMM receipt differs from frozen_ecmm_artifact configuration.")
    audit = getattr(core, "parameter_audit", None)
    audit_result = audit() if callable(audit) else None
    if audit_result is not None and not isinstance(audit_result, Mapping):
        raise TypeError("Student frozen ECMM parameter_audit() must return a mapping.")
    contract = audit_result.get("contract") if audit_result is not None else None
    return {
        "checkpoint_sha256": frozen_sha256,
        "actor_state_dict_key": actor_state_dict_key,
        "contract": contract,
        "checkpoint_path_source": "external_runner_configuration",
        "checkpoint_bytes_saved": False,
        "teacher_artifact": {
            "checkpoint_sha256": teacher_sha256,
            "schema": teacher_schema,
            "checkpoint_state_key": teacher_checkpoint_state_key,
            "container_schema": teacher_container_schema,
            "checkpoint_path_source": "external_runner_configuration",
            "checkpoint_bytes_saved": False,
        },
    }


class M2MDistillationRunner(DistillationRunner):
    """Runner that makes the C06/C07/C10/C11 boundary non-optional."""

    alg: M2MLatentActionDistillation

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct all M2M components from explicit artifact-backed configuration."""
        if not isinstance(train_cfg, dict):
            raise TypeError("M2M train_cfg must be an exact dictionary.")
        # C11 deliberately rejects distributed gradients.  Fail before the
        # inherited helper initializes a torch.distributed process group.
        if int(os.getenv("WORLD_SIZE", "1")) > 1:
            raise NotImplementedError("M2M distillation does not yet support multi-GPU synchronization.")

        self.env = env
        self.cfg = train_cfg
        self.device = device
        self._configure_multi_gpu()
        obs = self.env.get_observations()
        if not isinstance(obs, TensorDict):
            raise TypeError("M2M environment observations must be a TensorDict.")
        self.alg = self._construct_m2m_algorithm(obs)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0

        # Keep the base runner's save/load state initialized.  Formal training
        # remains explicitly unsupported below.
        self._formal_training_io = None
        self._formal_launch_receipt = None
        self._formal_launch_receipt_bytes = None
        self._formal_schedule = None
        self._formal_parent_checkpoint = None
        self._formal_last_local_embedded_receipt = None
        self._formal_updates_completed = 0
        self._formal_resume_loaded = False

    def _construct_scratch_teacher_models(
        self,
        obs: TensorDict,
    ) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
        """Build one full frozen teacher artifact and its shared map-free student."""
        if self.cfg.get("frozen_ecmm_artifact") is not None or self.cfg.get("teacher_artifact") is not None:
            raise ValueError(
                "scratch_teacher_artifact is mutually exclusive with legacy frozen_ecmm_artifact/teacher_artifact."
            )
        path, digest, actor_state_key, _ = _validate_artifact_config(
            self.cfg.get("scratch_teacher_artifact"),
            field="scratch_teacher_artifact",
            allow_actor_state_key=True,
        )
        assert actor_state_key is not None

        teacher_cfg = _mapping_copy(self.cfg.get("teacher"), field="teacher")
        teacher_owned = (
            _ARTIFACT_PATH_KEYS
            | _CONSTRUCTOR_OWNED_KEYS
            | _FROZEN_ECMM_INJECTION_KEYS
            | {"frozen_ecmm_actor_cfg"}
        ).intersection(teacher_cfg)
        if teacher_owned:
            raise ValueError(
                "Scratch teacher constructor/artifact fields are runner-owned: "
                f"{sorted(teacher_owned)}."
            )
        teacher_class = _model_class(teacher_cfg, field="teacher")
        teacher_obs_set = teacher_cfg.pop("obs_set", None)
        if teacher_obs_set != "teacher":
            raise ValueError("teacher.obs_set must be exactly 'teacher'.")
        teacher = teacher_class(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            obs_set=teacher_obs_set,
            output_dim=self.env.num_actions,
            **teacher_cfg,
        )
        core = M2MFrozenScratchTeacherCore(
            teacher,
            checkpoint_path=path,
            expected_sha256=digest,
            actor_state_dict_key=actor_state_key,
        )

        student_cfg = _mapping_copy(self.cfg.get("student"), field="student")
        student_owned = (
            _ARTIFACT_PATH_KEYS | _CONSTRUCTOR_OWNED_KEYS | _FROZEN_ECMM_INJECTION_KEYS
        ).intersection(student_cfg)
        if student_owned:
            raise ValueError(
                "Student constructor/artifact fields are runner-owned; configure scratch_teacher_artifact: "
                f"{sorted(student_owned)}."
            )
        # A scratch core supplies its own architecture; admitting the legacy
        # actor mapping here would silently mix two control policies.
        if "frozen_ecmm_actor_cfg" in student_cfg:
            raise ValueError("Scratch-teacher student config must not contain frozen_ecmm_actor_cfg.")
        student_class = _model_class(student_cfg, field="student")
        student_obs_set = student_cfg.pop("obs_set", None)
        if student_obs_set != "student":
            raise ValueError("student.obs_set must be exactly 'student'.")
        student = student_class(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            obs_set=student_obs_set,
            output_dim=self.env.num_actions,
            shared_ecmm_core=core,
            **student_cfg,
        )
        if getattr(student, "ecmm_core", None) is not core:
            raise RuntimeError("Scratch-teacher student did not preserve the shared frozen core identity.")

        audit = core.parameter_audit()
        receipt = {
            "checkpoint_sha256": digest,
            "actor_state_dict_key": actor_state_key,
            "contract": audit["contract"],
            "artifact_kind": "scratch_teacher_ordinary_ppo_full_actor",
            "checkpoint_path_source": "external_runner_configuration",
            "checkpoint_bytes_saved": False,
            "teacher_artifact": {
                "checkpoint_sha256": digest,
                "schema": "ordinary_ppo_full_actor_state_dict",
                "checkpoint_state_key": actor_state_key,
                "container_schema": None,
                "checkpoint_path_source": "same_scratch_teacher_artifact",
                "checkpoint_bytes_saved": False,
            },
        }
        self._artifact_mode = "scratch_teacher_full_actor"
        self._frozen_ecmm_artifact_path = path.resolve()
        self._teacher_artifact_path = path.resolve()
        self._frozen_ecmm_sha256 = digest
        self._teacher_artifact_sha256 = digest
        self._teacher_checkpoint_state_key = actor_state_key
        return student, core.actor, receipt

    def _construct_legacy_models(
        self,
        obs: TensorDict,
    ) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
        """Preserve the original frozen-M90 plus C06 map-only path."""
        frozen_path, frozen_sha256, actor_state_key, _ = _validate_artifact_config(
            self.cfg.get("frozen_ecmm_artifact"),
            field="frozen_ecmm_artifact",
            allow_actor_state_key=True,
        )
        assert actor_state_key is not None
        teacher_path, teacher_sha256, _, teacher_checkpoint_state_key = _validate_artifact_config(
            self.cfg.get("teacher_artifact"),
            field="teacher_artifact",
            allow_actor_state_key=False,
            allow_checkpoint_state_key=True,
        )
        self._artifact_mode = "legacy_m90_plus_c06"
        self._frozen_ecmm_artifact_path = frozen_path.resolve()
        self._teacher_artifact_path = teacher_path.resolve()
        self._frozen_ecmm_sha256 = frozen_sha256
        self._teacher_artifact_sha256 = teacher_sha256
        self._teacher_checkpoint_state_key = teacher_checkpoint_state_key

        student_cfg = _mapping_copy(self.cfg.get("student"), field="student")
        student_owned = (
            _ARTIFACT_PATH_KEYS | _CONSTRUCTOR_OWNED_KEYS | _FROZEN_ECMM_INJECTION_KEYS
        ).intersection(student_cfg)
        if student_owned:
            raise ValueError(
                "student constructor/artifact fields are runner-owned; configure frozen_ecmm_artifact instead: "
                f"{sorted(student_owned)}."
            )
        student_class = _model_class(student_cfg, field="student")
        student_obs_set = student_cfg.pop("obs_set", None)
        if student_obs_set != "student":
            raise ValueError("student.obs_set must be exactly 'student'.")
        student = student_class(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            obs_set=student_obs_set,
            output_dim=self.env.num_actions,
            frozen_ecmm_checkpoint_path=str(frozen_path),
            frozen_ecmm_expected_sha256=frozen_sha256,
            frozen_ecmm_actor_state_dict_key=actor_state_key,
            **student_cfg,
        )

        teacher_cfg = _mapping_copy(self.cfg.get("teacher"), field="teacher")
        teacher_owned = (
            _ARTIFACT_PATH_KEYS
            | _CONSTRUCTOR_OWNED_KEYS
            | _FROZEN_ECMM_INJECTION_KEYS
            | {"frozen_ecmm_actor_cfg"}
        ).intersection(teacher_cfg)
        if teacher_owned:
            raise ValueError(
                "teacher constructor/checkpoint fields are runner-owned; configure teacher_artifact instead: "
                f"{sorted(teacher_owned)}."
            )
        teacher_class = _model_class(teacher_cfg, field="teacher")
        teacher_obs_set = teacher_cfg.pop("obs_set", None)
        if teacher_obs_set != "teacher":
            raise ValueError("teacher.obs_set must be exactly 'teacher'.")
        teacher = teacher_class(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            obs_set=teacher_obs_set,
            output_dim=self.env.num_actions,
            shared_ecmm_core=student.ecmm_core,
            **teacher_cfg,
        )
        load_teacher = getattr(teacher, "load_checkpoint_state", None)
        if not callable(load_teacher):
            raise TypeError("M2M teacher must expose load_checkpoint_state(checkpoint).")
        teacher_container = torch.load(teacher_path, map_location="cpu", weights_only=True)
        if not isinstance(teacher_container, Mapping):
            raise TypeError("teacher_artifact checkpoint root must be a mapping.")
        teacher_container_schema: str | None = None
        if teacher_checkpoint_state_key is None:
            teacher_checkpoint = teacher_container
        else:
            container_schema = teacher_container.get("schema")
            if not isinstance(container_schema, str) or not container_schema:
                raise ValueError("Nested teacher artifact container must contain a non-empty schema string.")
            teacher_container_schema = container_schema
            if teacher_checkpoint_state_key not in teacher_container:
                raise KeyError(
                    "teacher_artifact checkpoint is missing configured checkpoint_state_key "
                    f"{teacher_checkpoint_state_key!r}."
                )
            teacher_checkpoint = teacher_container[teacher_checkpoint_state_key]
            if not isinstance(teacher_checkpoint, Mapping):
                raise TypeError("Configured teacher checkpoint_state_key must select a mapping.")
        teacher_schema = teacher_checkpoint.get("schema")
        if not isinstance(teacher_schema, str) or not teacher_schema:
            raise ValueError("teacher_artifact must contain a non-empty schema string.")
        load_teacher(teacher_checkpoint)
        receipt = _artifact_receipt(
            student=student,
            frozen_sha256=frozen_sha256,
            actor_state_dict_key=actor_state_key,
            teacher_sha256=teacher_sha256,
            teacher_schema=teacher_schema,
            teacher_checkpoint_state_key=teacher_checkpoint_state_key,
            teacher_container_schema=teacher_container_schema,
        )
        return student, teacher, receipt

    def _construct_m2m_algorithm(self, obs: TensorDict) -> M2MLatentActionDistillation:
        if type(self.env.num_envs) is not int or self.env.num_envs <= 0:
            raise ValueError("M2M env.num_envs must be an exact positive integer.")
        if type(self.env.num_actions) is not int or self.env.num_actions != 29:
            raise ValueError(f"M2M Unitree-G1 requires env.num_actions=29, got {self.env.num_actions!r}.")
        rollout_length = _exact_positive_int(self.cfg.get("num_steps_per_env"), field="num_steps_per_env")

        obs_groups_value = self.cfg.get("obs_groups")
        obs_groups = _mapping_copy(obs_groups_value, field="obs_groups")
        # Unlike generic RSL-RL fallback behavior, both scientific input sets
        # are mandatory and explicit.
        if "student" not in obs_groups or "teacher" not in obs_groups:
            raise ValueError("M2M obs_groups must explicitly define both 'student' and 'teacher'.")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, obs_groups, ["student", "teacher"])

        if self.cfg.get("scratch_teacher_artifact") is not None:
            student, teacher, receipt = self._construct_scratch_teacher_models(obs)
        else:
            student, teacher, receipt = self._construct_legacy_models(obs)

        storage_cfg = _mapping_copy(self.cfg.get("storage"), field="storage")
        allowed_storage_keys = {
            "allowed_student_keys",
            "hidden_state_shape",
            "target_dtype",
            "hidden_state_dtype",
            "student_obs_storage_dtypes",
        }
        unexpected_storage = set(storage_cfg).difference(allowed_storage_keys)
        if unexpected_storage:
            raise ValueError(f"Unsupported M2M storage keys: {sorted(unexpected_storage)}.")
        allowed_keys_value = storage_cfg.pop("allowed_student_keys", None)
        if not isinstance(allowed_keys_value, Sequence) or isinstance(allowed_keys_value, (str, bytes)):
            raise TypeError("storage.allowed_student_keys must be an explicit sequence of group names.")
        allowed_keys = tuple(allowed_keys_value)
        if any(not isinstance(key, str) or not key for key in allowed_keys):
            raise ValueError("storage.allowed_student_keys must contain non-empty strings.")
        expected_student_groups = tuple(self.cfg["obs_groups"]["student"])
        if allowed_keys != expected_student_groups:
            raise ValueError(
                "storage.allowed_student_keys must exactly preserve student obs-group order: "
                f"expected={expected_student_groups}, actual={allowed_keys}."
            )
        student_obs = TensorDict({}, batch_size=obs.batch_size, device=obs.device)
        for key in allowed_keys:
            student_obs[key] = obs[key]

        recurrent = bool(getattr(student, "is_recurrent", False))
        if recurrent:
            derived_hidden_shape = (
                _exact_positive_int(getattr(student, "gru_num_layers", None), field="student.gru_num_layers"),
                _exact_positive_int(getattr(student, "gru_hidden_dim", None), field="student.gru_hidden_dim"),
            )
        else:
            # C10 keeps one uniform env-first hidden tensor contract.  Current-
            # frame students carry a one-element zero placeholder only.
            derived_hidden_shape = (1, 1)
        configured_hidden = storage_cfg.pop("hidden_state_shape", None)
        if configured_hidden is not None and tuple(configured_hidden) != derived_hidden_shape:
            raise ValueError(
                "storage.hidden_state_shape differs from the constructed student: "
                f"expected={derived_hidden_shape}, actual={configured_hidden}."
            )

        storage_dtypes_value = storage_cfg.pop("student_obs_storage_dtypes", {})
        storage_dtypes_cfg = _mapping_copy(storage_dtypes_value, field="storage.student_obs_storage_dtypes")
        storage_dtypes: dict[str | tuple[str, ...], torch.dtype] = {
            key: _dtype(value, field=f"storage.student_obs_storage_dtypes.{key}")
            for key, value in storage_dtypes_cfg.items()
        }
        target_dtype = _dtype(storage_cfg.pop("target_dtype", "float32"), field="storage.target_dtype")
        hidden_dtype = _dtype(storage_cfg.pop("hidden_state_dtype", "float32"), field="storage.hidden_state_dtype")
        storage = M2MSequenceRolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=rollout_length,
            student_obs=student_obs,
            allowed_student_keys=allowed_keys,
            hidden_state_shape=derived_hidden_shape,
            device=self.device,
            target_dtype=target_dtype,
            hidden_state_dtype=hidden_dtype,
            student_obs_storage_dtypes=storage_dtypes,
        )

        algorithm_cfg = _mapping_copy(self.cfg.get("algorithm"), field="algorithm")
        algorithm_owned = {"device", "multi_gpu_cfg", "frozen_artifact_receipt"}.intersection(algorithm_cfg)
        if algorithm_owned:
            raise ValueError(f"M2M algorithm fields are runner-owned: {sorted(algorithm_owned)}.")
        algorithm_class_name = algorithm_cfg.pop("class_name", None)
        if not isinstance(algorithm_class_name, str) or not algorithm_class_name:
            raise ValueError("algorithm.class_name must be explicit.")
        algorithm_class = resolve_callable(algorithm_class_name)
        if not isinstance(algorithm_class, type) or not issubclass(algorithm_class, M2MLatentActionDistillation):
            raise TypeError("algorithm.class_name must resolve to M2MLatentActionDistillation.")
        for extension in ("rnd_cfg", "symmetry_cfg", "dwaq_cfg", "amp_cfg"):
            if algorithm_cfg.get(extension) is not None:
                raise ValueError(f"M2M distillation requires algorithm.{extension}=None.")
            algorithm_cfg.pop(extension, None)
        # The inherited collection/logger loop indexes this key directly.
        self.cfg["algorithm"]["rnd_cfg"] = None
        self.cfg["algorithm"]["symmetry_cfg"] = None

        loss_value = algorithm_cfg.pop("loss_config", None)
        if not isinstance(loss_value, Mapping):
            raise TypeError("algorithm.loss_config must be an explicit mapping.")
        loss_config = M2MDistillationLossConfig(**copy.deepcopy(dict(loss_value)))
        algorithm = algorithm_class(
            student,
            teacher,
            storage,
            loss_config=loss_config,
            frozen_artifact_receipt=receipt,
            device=self.device,
            rnd_cfg=None,
            symmetry_cfg=None,
            multi_gpu_cfg=self.cfg["multi_gpu"],
            **algorithm_cfg,
        )
        if not isinstance(algorithm.storage, M2MSequenceRolloutStorage):
            raise RuntimeError("M2M runner constructed a non-C10 rollout storage.")
        algorithm.teacher_loaded = True
        return algorithm

    def configure_formal_training(self, context: dict[str, Any]) -> None:
        """Reject the PPO-only formal checkpoint protocol."""
        del context
        raise NotImplementedError("Formal PPO checkpoint semantics do not support M2M distillation yet.")

    def _assert_checkpoint_boundary(self) -> None:
        if getattr(self.alg, "_pending", None) is not None or self.alg.storage.step != 0:
            raise RuntimeError("M2M checkpoints may only be saved/loaded at a cleared rollout boundary.")

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save a student-training checkpoint at a completed rollout boundary."""
        self._assert_checkpoint_boundary()
        completed_updates = self.alg.num_updates
        if type(completed_updates) is not int or completed_updates <= 0:
            raise RuntimeError("M2M resume checkpoints require at least one completed optimizer update.")
        checkpoint_iteration = completed_updates - 1
        if self.current_learning_iteration not in {checkpoint_iteration, completed_updates}:
            raise RuntimeError(
                "M2M runner progress differs from algorithm completed updates: "
                f"runner={self.current_learning_iteration}, completed={completed_updates}."
            )
        saved_dict = self.alg.save()
        if saved_dict.get("algorithm_iteration") != completed_updates:
            raise RuntimeError("M2M algorithm save receipt changed its completed-update count.")
        saved_dict["iter"] = checkpoint_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, checkpoint_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Restore an exact M2M training checkpoint and resume at the next update."""
        self._assert_checkpoint_boundary()
        candidate = Path(path).expanduser().resolve()
        if candidate in {self._teacher_artifact_path, self._frozen_ecmm_artifact_path}:
            raise ValueError(
                "M2M runner.load expects a student-training resume checkpoint, not a teacher/frozen artifact."
            )
        if map_location is not None and torch.device(map_location) != torch.device(self.device):
            raise ValueError("M2M resume map_location must be None or exactly the configured runner device.")
        location = self.device if map_location is None else map_location
        loaded_dict = torch.load(candidate, weights_only=True, map_location=location)
        if not isinstance(loaded_dict, dict):
            raise TypeError("M2M resume checkpoint root must be an exact dictionary.")
        checkpoint_iteration = loaded_dict.get("iter")
        completed_updates = loaded_dict.get("algorithm_iteration")
        if (
            type(checkpoint_iteration) is not int
            or checkpoint_iteration < 0
            or type(completed_updates) is not int
            or completed_updates <= 0
            or checkpoint_iteration + 1 != completed_updates
        ):
            raise ValueError("M2M resume progress must satisfy iter + 1 == algorithm_iteration > 0.")
        infos = loaded_dict.get("infos")
        if infos is not None and not isinstance(infos, dict):
            raise TypeError("M2M checkpoint infos must be a dictionary or None.")
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration is not True or self.alg.num_updates != completed_updates:
            raise RuntimeError("M2M algorithm refused the exact completed-update resume state.")
        # OnPolicyRunner's loop interprets this field as the next range start.
        # Checkpoint ``iter`` remains the zero-based last completed update.
        self.current_learning_iteration = completed_updates
        return {} if infos is None else infos

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Require the C13 student-only packager instead of generic export."""
        del path, filename
        raise NotImplementedError("Use the C13 student-only artifact packager; generic runner JIT export is forbidden.")

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
        input_mode: str = "split",
    ) -> None:
        """Require the C13 student-only packager instead of generic export."""
        del path, filename, verbose, input_mode
        raise NotImplementedError(
            "Use the C13 student-only artifact packager; generic runner ONNX export is forbidden."
        )

    def audit(self) -> dict[str, Any]:
        """Return the runner, artifact, storage, resume and export boundaries."""
        return {
            "phase": "C12_m2m_runner_factory",
            "runner_factory_integrated": True,
            "teacher_loaded": self.alg.teacher_loaded,
            "artifacts": {
                "mode": self._artifact_mode,
                "frozen_ecmm_sha256": self._frozen_ecmm_sha256,
                "teacher_sha256": self._teacher_artifact_sha256,
                "teacher_checkpoint_state_key": self._teacher_checkpoint_state_key,
                "paths_in_checkpoint": False,
                "resume_path_is_teacher_path": False,
            },
            "storage": {
                "class": type(self.alg.storage).__name__,
                "allowed_student_keys": list(self.alg.storage.allowed_student_keys),
                "legacy_rollout_storage": False,
                "stores_teacher_map": False,
                "stores_next_observation": False,
            },
            "checkpoint": {
                "student_trainable_state_only": True,
                "teacher_weights_saved": False,
                "resume_supported": True,
                "rollout_boundary_required": True,
                "progress_contract": "iter + 1 == algorithm_iteration == completed_updates",
            },
            "extensions": {"rnd_cfg": None, "symmetry_cfg": None, "multi_gpu": False},
            "export": {
                "generic_runner_export_enabled": False,
                "owner": "C13_student_only_artifact_export",
            },
        }


__all__ = ["M2MDistillationRunner"]
