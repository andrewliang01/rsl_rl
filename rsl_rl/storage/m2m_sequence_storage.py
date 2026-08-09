# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Memory-bounded sequence storage for M2M latent distillation.

The M2M student is deployable without the privileged teacher map.  This
storage therefore accepts only an explicit student observation ``TensorDict``
and detached teacher labels (A64 and action mean 29).  It deliberately has no
teacher-observation or next-observation buffer.

Hidden states use an environment-first contract: ``[N, *hidden_state_shape]``.
For a GRU, ``hidden_state_shape`` is normally ``(num_layers, hidden_size)``.
The sequence generator returns only the hidden state at the beginning of each
truncated-BPTT chunk.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from tensordict import TensorDict


@dataclass
class M2MSequenceTransition:
    """One vectorized M2M distillation transition.

    ``student_hidden_state`` is the state immediately before consuming
    ``student_observations``.  ``episode_starts`` is optional because a done at
    the preceding storage step already creates the same boundary.
    """

    student_observations: TensorDict
    teacher_latent_A: torch.Tensor  # noqa: N815 - paper/contract symbol is A
    teacher_action_mean: torch.Tensor
    dones: torch.Tensor
    student_hidden_state: torch.Tensor
    episode_starts: torch.Tensor | None = None


@dataclass(frozen=True)
class M2MSequenceBatch:
    """Padded, time-major truncated-BPTT batch.

    All time-dependent tensors use layout ``[L, B, ...]``.  ``masks`` marks
    real transitions; padded positions are zero.  The initial hidden state is
    environment-first ``[B, *hidden_state_shape]``.
    """

    student_observations: TensorDict
    teacher_latent_A: torch.Tensor  # noqa: N815 - paper/contract symbol is A
    teacher_action_mean: torch.Tensor
    dones: torch.Tensor
    episode_starts: torch.Tensor
    masks: torch.Tensor
    initial_student_hidden: torch.Tensor
    env_ids: torch.Tensor
    start_steps: torch.Tensor
    sequence_lengths: torch.Tensor

    def gru_initial_hidden_state(self) -> torch.Tensor:
        """Return ``[num_layers, B, hidden_size]`` for a standard GRU.

        This convenience method is valid when the configured per-environment
        hidden shape is ``(num_layers, hidden_size)``.
        """
        if self.initial_student_hidden.ndim != 3:
            raise ValueError("GRU conversion requires initial_hidden_state with layout [B,num_layers,hidden_size].")
        return self.initial_student_hidden.permute(1, 0, 2).contiguous()


@dataclass(frozen=True)
class M2MStorageMemoryEstimate:
    """Persistent rollout-buffer memory estimate."""

    num_envs: int
    rollout_length: int
    field_bytes: Mapping[str, int]
    field_dtypes: Mapping[str, str]

    @property
    def total_bytes(self) -> int:
        """Total estimated persistent bytes."""
        return sum(self.field_bytes.values())

    @property
    def total_mib(self) -> float:
        """Total estimated persistent mebibytes."""
        return self.total_bytes / (1024.0**2)

    @property
    def total_gib(self) -> float:
        """Total estimated persistent gibibytes."""
        return self.total_bytes / (1024.0**3)

    def audit(self) -> dict[str, object]:
        """Return a serialization-friendly capacity report."""
        return {
            "num_envs": self.num_envs,
            "rollout_length": self.rollout_length,
            "field_bytes": dict(self.field_bytes),
            "field_dtypes": dict(self.field_dtypes),
            "total_bytes": self.total_bytes,
            "total_mib": self.total_mib,
            "total_gib": self.total_gib,
            "includes_teacher_map": False,
            "includes_next_observation": False,
        }


@dataclass(frozen=True)
class M2MStorageMemoryComparison:
    """Like-for-like persistent-byte comparison with legacy distillation storage."""

    m2m: M2MStorageMemoryEstimate
    legacy_field_bytes: Mapping[str, int]
    assumptions: tuple[str, ...]

    @property
    def legacy_total_bytes(self) -> int:
        """Total analytically allocated by legacy distillation storage."""
        return sum(self.legacy_field_bytes.values())

    @property
    def difference_bytes(self) -> int:
        """Legacy minus M2M bytes; negative means M2M is larger."""
        return self.legacy_total_bytes - self.m2m.total_bytes

    def audit(self) -> dict[str, object]:
        """Return both totals and every accounting assumption."""
        return {
            "m2m_sequence_storage": self.m2m.audit(),
            "legacy_distillation_rollout": {
                "field_bytes": dict(self.legacy_field_bytes),
                "total_bytes": self.legacy_total_bytes,
                "total_mib": self.legacy_total_bytes / (1024.0**2),
                "total_gib": self.legacy_total_bytes / (1024.0**3),
            },
            "legacy_minus_m2m_bytes": self.difference_bytes,
            "assumptions": list(self.assumptions),
        }


class M2MSequenceRolloutStorage:
    """Specialized rollout storage for map-free recurrent M2M students."""

    teacher_latent_dim = 64
    action_dim = 29
    _forbidden_student_key_fragments = (
        "teacher",
        "privileged",
        "ground_truth",
        "terrain_map",
        "oracle",
    )

    def __init__(
        self,
        *,
        num_envs: int,
        num_transitions_per_env: int,
        student_obs: TensorDict,
        allowed_student_keys: Iterable[str | tuple[str, ...]],
        hidden_state_shape: tuple[int, ...] | list[int],
        device: str | torch.device = "cpu",
        target_dtype: torch.dtype = torch.float32,
        hidden_state_dtype: torch.dtype = torch.float32,
        student_obs_storage_dtypes: Mapping[str | tuple[str, ...], torch.dtype] | None = None,
    ) -> None:
        """Allocate only fields required by M2M sequence distillation.

        Args:
            num_envs: Number of vectorized environments.
            num_transitions_per_env: Rollout horizon.
            student_obs: Example *deployable* observation ``TensorDict`` with
                batch size ``[num_envs]``.
            allowed_student_keys: Explicit, frozen allowlist.  Its normalized
                key set must exactly match ``student_obs`` and every transition.
            hidden_state_shape: Per-environment hidden shape.  For a GRU this
                is ``(num_layers, hidden_size)``.
            device: Storage device.
            target_dtype: Dtype for A64 and action-mean labels.
            hidden_state_dtype: Dtype for saved recurrent states.
            student_obs_storage_dtypes: Optional per-leaf floating-point
                compression.  Unspecified leaves preserve their source dtype.
        """
        self.num_envs = self._positive_int("num_envs", num_envs)
        self.num_transitions_per_env = self._positive_int("num_transitions_per_env", num_transitions_per_env)
        self.device = torch.device(device)
        self.hidden_state_shape = self._validate_hidden_shape(hidden_state_shape)
        self.target_dtype = self._validate_float_dtype("target_dtype", target_dtype)
        self.hidden_state_dtype = self._validate_float_dtype("hidden_state_dtype", hidden_state_dtype)
        self._allowed_student_keys = self._validate_allowed_student_keys(allowed_student_keys)
        self._validate_student_obs(
            student_obs,
            expected_num_envs=self.num_envs,
            allowed_student_keys=self.allowed_student_keys,
        )
        self._student_obs_keys = tuple(student_obs.keys(include_nested=True, leaves_only=True))
        self._source_obs_dtypes = {self._key_path(key): student_obs[key].dtype for key in self._student_obs_keys}
        self._storage_obs_dtypes = self._resolve_storage_dtypes(
            student_obs,
            allowed_student_keys=self.allowed_student_keys,
            requested=student_obs_storage_dtypes,
        )

        self.student_observations = TensorDict(
            {},
            batch_size=[self.num_transitions_per_env, self.num_envs],
            device=self.device,
        )
        for key in self._student_obs_keys:
            value = student_obs[key]
            path = self._key_path(key)
            self.student_observations[key] = torch.zeros(
                self.num_transitions_per_env,
                *value.shape,
                dtype=self._storage_obs_dtypes[path],
                device=self.device,
            )

        shape_prefix = (self.num_transitions_per_env, self.num_envs)
        self.teacher_latent_A = torch.zeros(
            *shape_prefix,
            self.teacher_latent_dim,
            dtype=self.target_dtype,
            device=self.device,
        )
        self.teacher_action_mean = torch.zeros(
            *shape_prefix,
            self.action_dim,
            dtype=self.target_dtype,
            device=self.device,
        )
        self.dones = torch.zeros(*shape_prefix, 1, dtype=torch.bool, device=self.device)
        self.episode_starts = torch.zeros(*shape_prefix, 1, dtype=torch.bool, device=self.device)
        self.student_hidden_states = torch.zeros(
            *shape_prefix,
            *self.hidden_state_shape,
            dtype=self.hidden_state_dtype,
            device=self.device,
        )

        # This vector carries asynchronous done/reset boundaries into the next
        # transition, including across ``clear()`` at a rollout boundary.
        self._next_episode_starts = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.step = 0

    @property
    def allowed_student_keys(self) -> tuple[str, ...]:
        """Immutable normalized deployable-observation allowlist."""
        return self._allowed_student_keys

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        return value

    @staticmethod
    def _validate_hidden_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        if not isinstance(shape, (tuple, list)) or not shape:
            raise ValueError("hidden_state_shape must be a non-empty tuple/list.")
        normalized = tuple(shape)
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in normalized):
            raise ValueError(f"Every hidden_state_shape dimension must be a positive integer, got {shape!r}.")
        return normalized

    @staticmethod
    def _validate_float_dtype(name: str, dtype: torch.dtype) -> torch.dtype:
        if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError(f"{name} must be a floating-point torch dtype, got {dtype!r}.")
        return dtype

    @staticmethod
    def _key_path(key: str | tuple[str, ...]) -> str:
        if isinstance(key, str):
            path = key
        elif isinstance(key, tuple) and key and all(isinstance(component, str) and component for component in key):
            path = ".".join(key)
        else:
            raise TypeError(f"Observation keys must be non-empty strings or tuples of non-empty strings, got {key!r}.")
        if not path:
            raise ValueError("Observation key paths cannot be empty.")
        return path

    @classmethod
    def _validate_allowed_student_keys(cls, keys: Iterable[str | tuple[str, ...]]) -> tuple[str, ...]:
        if isinstance(keys, (str, bytes)):
            raise TypeError("allowed_student_keys must be an iterable of key paths, not a string.")
        try:
            normalized = tuple(cls._key_path(key) for key in keys)
        except TypeError as error:
            raise TypeError("allowed_student_keys must be an iterable of key paths.") from error
        if not normalized:
            raise ValueError("allowed_student_keys must not be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"allowed_student_keys contains duplicates: {normalized}.")
        return normalized

    @classmethod
    def _validate_student_obs(
        cls,
        student_obs: TensorDict,
        *,
        expected_num_envs: int | None,
        allowed_student_keys: tuple[str, ...],
    ) -> None:
        if not isinstance(student_obs, TensorDict):
            raise TypeError(f"student_obs must be a TensorDict, got {type(student_obs).__name__}.")
        if len(student_obs.batch_size) != 1:
            raise ValueError(
                "student_obs must have exactly one vector-environment batch dimension, "
                f"got batch_size={tuple(student_obs.batch_size)}."
            )
        if expected_num_envs is not None and student_obs.batch_size[0] != expected_num_envs:
            raise ValueError(
                f"student_obs batch size must be [{expected_num_envs}], got {tuple(student_obs.batch_size)}."
            )
        keys = tuple(student_obs.keys(include_nested=True, leaves_only=True))
        if not keys:
            raise ValueError("student_obs must contain at least one tensor leaf.")
        actual_paths = tuple(cls._key_path(key) for key in keys)
        if set(actual_paths) != set(allowed_student_keys):
            raise ValueError(
                "student_obs key set must exactly match allowed_student_keys: "
                f"allowed={allowed_student_keys}, actual={actual_paths}."
            )
        for key in keys:
            value = student_obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"student_obs[{key!r}] must be a tensor.")
            path = cls._key_path(key)
            normalized_path = path.lower()
            if any(fragment in normalized_path for fragment in cls._forbidden_student_key_fragments):
                raise ValueError(f"Observation key {path!r} is privileged/non-deployable and cannot enter M2M storage.")

    @classmethod
    def _resolve_storage_dtypes(
        cls,
        student_obs: TensorDict,
        *,
        allowed_student_keys: tuple[str, ...],
        requested: Mapping[str | tuple[str, ...], torch.dtype] | None,
    ) -> dict[str, torch.dtype]:
        normalized_requested: dict[str, torch.dtype] = {}
        if requested is not None:
            if not isinstance(requested, Mapping):
                raise TypeError("student_obs_storage_dtypes must be a mapping or None.")
            for raw_key, dtype in requested.items():
                path = cls._key_path(raw_key)
                if path in normalized_requested:
                    raise ValueError(f"Duplicate storage dtype override for {path!r}.")
                normalized_requested[path] = dtype
        unknown = set(normalized_requested).difference(allowed_student_keys)
        if unknown:
            raise ValueError(
                f"student_obs_storage_dtypes contains keys outside allowed_student_keys: {sorted(unknown)}."
            )

        source_by_path = {
            cls._key_path(key): student_obs[key].dtype
            for key in student_obs.keys(include_nested=True, leaves_only=True)
        }
        resolved: dict[str, torch.dtype] = {}
        for path in allowed_student_keys:
            source_dtype = source_by_path[path]
            storage_dtype = normalized_requested.get(path, source_dtype)
            if not isinstance(storage_dtype, torch.dtype):
                raise TypeError(f"Storage dtype for {path!r} must be torch.dtype, got {storage_dtype!r}.")
            if storage_dtype != source_dtype:
                source_is_float = torch.empty((), dtype=source_dtype).is_floating_point()
                storage_is_float = torch.empty((), dtype=storage_dtype).is_floating_point()
                if not source_is_float or not storage_is_float:
                    raise ValueError(
                        f"Only floating observation leaves may change storage dtype; "
                        f"{path!r} requested {source_dtype} -> {storage_dtype}."
                    )
                if cls._dtype_bytes(storage_dtype) > cls._dtype_bytes(source_dtype):
                    raise ValueError(
                        f"Storage dtype override for {path!r} is not compression: {source_dtype} -> {storage_dtype}."
                    )
            resolved[path] = storage_dtype
        return resolved

    @staticmethod
    def _dtype_bytes(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    @classmethod
    def estimate_memory(
        cls,
        *,
        num_envs: int,
        num_transitions_per_env: int,
        student_obs: TensorDict,
        allowed_student_keys: Iterable[str | tuple[str, ...]],
        hidden_state_shape: tuple[int, ...] | list[int],
        target_dtype: torch.dtype = torch.float32,
        hidden_state_dtype: torch.dtype = torch.float32,
        student_obs_storage_dtypes: Mapping[str | tuple[str, ...], torch.dtype] | None = None,
    ) -> M2MStorageMemoryEstimate:
        """Estimate persistent storage without allocating the requested capacity.

        ``student_obs`` may use any positive example batch size; only each
        leaf's per-environment trailing shape and dtype contribute.  This makes
        a 4096-environment estimate possible from a one-environment sample.
        """
        num_envs = cls._positive_int("num_envs", num_envs)
        rollout_length = cls._positive_int("num_transitions_per_env", num_transitions_per_env)
        allowed_keys = cls._validate_allowed_student_keys(allowed_student_keys)
        cls._validate_student_obs(
            student_obs,
            expected_num_envs=None,
            allowed_student_keys=allowed_keys,
        )
        storage_dtypes = cls._resolve_storage_dtypes(
            student_obs,
            allowed_student_keys=allowed_keys,
            requested=student_obs_storage_dtypes,
        )
        hidden_shape = cls._validate_hidden_shape(hidden_state_shape)
        target_dtype = cls._validate_float_dtype("target_dtype", target_dtype)
        hidden_state_dtype = cls._validate_float_dtype("hidden_state_dtype", hidden_state_dtype)

        time_env = rollout_length * num_envs
        field_bytes: dict[str, int] = {}
        field_dtypes: dict[str, str] = {}
        for key in student_obs.keys(include_nested=True, leaves_only=True):
            value = student_obs[key]
            trailing_numel = math.prod(value.shape[len(student_obs.batch_size) :])
            path = cls._key_path(key)
            field_name = f"student_observations.{path}"
            storage_dtype = storage_dtypes[path]
            field_bytes[field_name] = time_env * trailing_numel * cls._dtype_bytes(storage_dtype)
            field_dtypes[field_name] = f"source={value.dtype},storage={storage_dtype}"
        field_bytes["teacher_latents_A64"] = time_env * cls.teacher_latent_dim * cls._dtype_bytes(target_dtype)
        field_dtypes["teacher_latents_A64"] = str(target_dtype)
        field_bytes["teacher_action_means_29"] = time_env * cls.action_dim * cls._dtype_bytes(target_dtype)
        field_dtypes["teacher_action_means_29"] = str(target_dtype)
        field_bytes["dones"] = time_env * cls._dtype_bytes(torch.bool)
        field_dtypes["dones"] = str(torch.bool)
        field_bytes["episode_starts"] = time_env * cls._dtype_bytes(torch.bool)
        field_dtypes["episode_starts"] = str(torch.bool)
        field_bytes["student_hidden_states"] = time_env * math.prod(hidden_shape) * cls._dtype_bytes(hidden_state_dtype)
        field_dtypes["student_hidden_states"] = str(hidden_state_dtype)
        field_bytes["pending_episode_starts"] = num_envs * cls._dtype_bytes(torch.bool)
        field_dtypes["pending_episode_starts"] = str(torch.bool)
        return M2MStorageMemoryEstimate(
            num_envs=num_envs,
            rollout_length=rollout_length,
            field_bytes=field_bytes,
            field_dtypes=field_dtypes,
        )

    @classmethod
    def compare_memory_with_legacy_distillation(
        cls,
        *,
        num_envs: int,
        num_transitions_per_env: int,
        student_obs: TensorDict,
        allowed_student_keys: Iterable[str | tuple[str, ...]],
        legacy_obs: TensorDict,
        hidden_state_shape: tuple[int, ...] | list[int],
        target_dtype: torch.dtype = torch.float32,
        hidden_state_dtype: torch.dtype = torch.float32,
        student_obs_storage_dtypes: Mapping[str | tuple[str, ...], torch.dtype] | None = None,
    ) -> M2MStorageMemoryComparison:
        """Compare exact persistent tensor payloads under explicit obs specs.

        ``legacy_obs`` must contain exactly the groups that would be passed to
        the old ``RolloutStorage`` (including a teacher map if that runner
        passes one).  Keeping it explicit avoids claiming map-memory savings
        for a legacy configuration that never stored a map.
        """
        m2m = cls.estimate_memory(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            student_obs=student_obs,
            allowed_student_keys=allowed_student_keys,
            hidden_state_shape=hidden_state_shape,
            target_dtype=target_dtype,
            hidden_state_dtype=hidden_state_dtype,
            student_obs_storage_dtypes=student_obs_storage_dtypes,
        )
        num_envs = cls._positive_int("num_envs", num_envs)
        rollout_length = cls._positive_int("num_transitions_per_env", num_transitions_per_env)
        if not isinstance(legacy_obs, TensorDict):
            raise TypeError(f"legacy_obs must be a TensorDict, got {type(legacy_obs).__name__}.")
        if len(legacy_obs.batch_size) != 1 or not legacy_obs.keys(include_nested=True, leaves_only=True):
            raise ValueError("legacy_obs must contain tensor leaves and have one example batch dimension.")

        time_env = rollout_length * num_envs
        legacy_bytes: dict[str, int] = {}
        for key in legacy_obs.keys(include_nested=True, leaves_only=True):
            value = legacy_obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"legacy_obs[{key!r}] must be a tensor.")
            trailing_numel = math.prod(value.shape[len(legacy_obs.batch_size) :])
            path = ".".join(key) if isinstance(key, tuple) else str(key)
            leaf_bytes = time_env * trailing_numel * value.element_size()
            legacy_bytes[f"observations.{path}"] = leaf_bytes
            legacy_bytes[f"next_observations.{path}"] = leaf_bytes
        float32_bytes = cls._dtype_bytes(torch.float32)
        legacy_bytes["rewards"] = time_env * float32_bytes
        legacy_bytes["actions_29"] = time_env * cls.action_dim * float32_bytes
        legacy_bytes["dones"] = time_env * cls._dtype_bytes(torch.uint8)
        legacy_bytes["privileged_actions_29"] = time_env * cls.action_dim * float32_bytes
        return M2MStorageMemoryComparison(
            m2m=m2m,
            legacy_field_bytes=legacy_bytes,
            assumptions=(
                "legacy_obs is caller-supplied and is not assumed to contain a teacher map",
                "legacy mode is RolloutStorage(training_type='distillation', num_critics=1)",
                "legacy lazy recurrent hidden buffers are zero because current "
                "distillation transitions do not save them",
                "both totals exclude TensorDict/Python allocator metadata and ephemeral minibatch tensors",
            ),
        )

    def memory_estimate(self) -> M2MStorageMemoryEstimate:
        """Report bytes of the currently allocated persistent tensors."""
        field_bytes: dict[str, int] = {}
        field_dtypes: dict[str, str] = {}
        for key in self._student_obs_keys:
            value = self.student_observations[key]
            path = self._key_path(key)
            field_name = f"student_observations.{path}"
            field_bytes[field_name] = value.numel() * value.element_size()
            field_dtypes[field_name] = f"source={self._source_obs_dtypes[path]},storage={value.dtype}"
        fixed_fields = {
            "teacher_latents_A64": self.teacher_latent_A,
            "teacher_action_means_29": self.teacher_action_mean,
            "dones": self.dones,
            "episode_starts": self.episode_starts,
            "student_hidden_states": self.student_hidden_states,
            "pending_episode_starts": self._next_episode_starts,
        }
        field_bytes.update({name: value.numel() * value.element_size() for name, value in fixed_fields.items()})
        field_dtypes.update({name: str(value.dtype) for name, value in fixed_fields.items()})
        return M2MStorageMemoryEstimate(
            num_envs=self.num_envs,
            rollout_length=self.num_transitions_per_env,
            field_bytes=field_bytes,
            field_dtypes=field_dtypes,
        )

    def observation_storage_audit(self) -> dict[str, dict[str, object]]:
        """Describe every source/storage dtype decision by frozen allowed key."""
        return {
            path: {
                "source_dtype": str(self._source_obs_dtypes[path]),
                "storage_dtype": str(self._storage_obs_dtypes[path]),
                "compressed": self._storage_obs_dtypes[path] != self._source_obs_dtypes[path],
            }
            for path in self.allowed_student_keys
        }

    def _validate_transition_obs(self, observations: TensorDict) -> None:
        self._validate_student_obs(
            observations,
            expected_num_envs=self.num_envs,
            allowed_student_keys=self.allowed_student_keys,
        )
        keys = tuple(observations.keys(include_nested=True, leaves_only=True))
        if set(keys) != set(self._student_obs_keys):
            raise ValueError(
                "Transition student observation keys do not match storage keys: "
                f"expected={self._student_obs_keys}, got={keys}."
            )
        for key in self._student_obs_keys:
            expected = self.student_observations[key].shape[1:]
            value = observations[key]
            if value.shape != expected:
                raise ValueError(
                    f"student_observations[{key!r}] shape must be {tuple(expected)}, got {tuple(value.shape)}."
                )
            path = self._key_path(key)
            if value.dtype != self._source_obs_dtypes[path]:
                raise ValueError(
                    f"student_observations[{key!r}] dtype must be {self._source_obs_dtypes[path]}, got {value.dtype}."
                )

    def _validate_label(self, name: str, value: torch.Tensor, feature_dim: int) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value).__name__}.")
        expected_shape = (self.num_envs, feature_dim)
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(value.shape)}.")
        if value.dtype != self.target_dtype:
            raise ValueError(f"{name} must use {self.target_dtype}, got {value.dtype}.")

    def _as_bool_column(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value).__name__}.")
        if value.shape not in ((self.num_envs,), (self.num_envs, 1)):
            raise ValueError(
                f"{name} must have shape [{self.num_envs}] or [{self.num_envs},1], got {tuple(value.shape)}."
            )
        return value.detach().to(device=self.device, dtype=torch.bool).view(self.num_envs, 1)

    def add_transition(self, transition: M2MSequenceTransition) -> None:
        """Append one vectorized transition."""
        if not isinstance(transition, M2MSequenceTransition):
            raise TypeError(f"transition must be M2MSequenceTransition, got {type(transition).__name__}.")
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("M2M rollout buffer overflow; call clear() before adding transitions.")
        self._validate_transition_obs(transition.student_observations)
        self._validate_label("teacher_latent_A", transition.teacher_latent_A, self.teacher_latent_dim)
        self._validate_label("teacher_action_mean", transition.teacher_action_mean, self.action_dim)
        expected_hidden_shape = (self.num_envs, *self.hidden_state_shape)
        if not isinstance(transition.student_hidden_state, torch.Tensor):
            raise TypeError(
                f"student_hidden_state must be a tensor, got {type(transition.student_hidden_state).__name__}."
            )
        if transition.student_hidden_state.shape != expected_hidden_shape:
            raise ValueError(
                "student_hidden_state must have environment-first shape "
                f"{expected_hidden_shape}, got {tuple(transition.student_hidden_state.shape)}."
            )
        if transition.student_hidden_state.dtype != self.hidden_state_dtype:
            raise ValueError(
                f"student_hidden_state must use {self.hidden_state_dtype}, got {transition.student_hidden_state.dtype}."
            )

        dones = self._as_bool_column("dones", transition.dones)
        current_episode_starts = self._next_episode_starts.clone()
        if transition.episode_starts is not None:
            current_episode_starts.logical_or_(self._as_bool_column("episode_starts", transition.episode_starts))

        with torch.no_grad():
            for key in self._student_obs_keys:
                self.student_observations[key][self.step].copy_(transition.student_observations[key].detach())
            self.teacher_latent_A[self.step].copy_(transition.teacher_latent_A.detach())
            self.teacher_action_mean[self.step].copy_(transition.teacher_action_mean.detach())
            self.dones[self.step].copy_(dones)
            self.episode_starts[self.step].copy_(current_episode_starts)
            self.student_hidden_states[self.step].copy_(transition.student_hidden_state.detach())
            # Never permit hidden-state leakage into an explicitly reset env,
            # even if the caller supplied a stale state for only that env.
            reset_mask = current_episode_starts
            for _ in self.hidden_state_shape[1:]:
                reset_mask = reset_mask.unsqueeze(-1)
            self.student_hidden_states[self.step].masked_fill_(reset_mask, 0)
            self._next_episode_starts.copy_(dones)
        self.step += 1

    def reset_envs(self, env_ids: torch.Tensor | Iterable[int]) -> None:
        """Mark an asynchronous subset as reset before its next transition."""
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.dtype == torch.bool:
            if ids.shape != (self.num_envs,):
                raise ValueError(f"Boolean env_ids mask must have shape [{self.num_envs}], got {tuple(ids.shape)}.")
            ids = ids.nonzero(as_tuple=False).squeeze(-1)
        else:
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if ids.dtype not in integer_dtypes:
                raise TypeError(f"env_ids must contain integers, got dtype={ids.dtype}.")
            ids = ids.to(dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError(f"env_ids must lie in [0,{self.num_envs - 1}].")
        self._next_episode_starts[ids] = True

    def clear(self, *, reset_episode_state: bool = False) -> None:
        """Reset the write cursor while preserving cross-rollout done state.

        Set ``reset_episode_state=True`` only when all environment/recurrent
        state is also being discarded (for example, a completely new run).
        """
        self.step = 0
        if reset_episode_state:
            self._next_episode_starts.zero_()

    def _sequence_descriptors(self, sequence_length: int) -> list[tuple[int, int, int]]:
        """Return ``(env_id, start, end)`` chunks without device sync per step."""
        if self.step == 0:
            raise RuntimeError("Cannot generate batches from an empty M2M rollout.")
        starts = self.episode_starts[: self.step, :, 0]
        if self.step > 1:
            starts = starts.clone()
            starts[1:].logical_or_(self.dones[: self.step - 1, :, 0])
        boundary_cpu = starts.detach().cpu()

        descriptors: list[tuple[int, int, int]] = []
        for env_id in range(self.num_envs):
            boundaries = boundary_cpu[:, env_id].nonzero(as_tuple=False).flatten().tolist()
            segment_starts = [0]
            segment_starts.extend(boundary for boundary in boundaries if boundary > 0)
            # OR-combination above can only produce unique timestep indices,
            # but ``dict.fromkeys`` also documents the invariant defensively.
            segment_starts = list(dict.fromkeys(segment_starts))
            segment_ends = [*segment_starts[1:], self.step]
            for segment_start, segment_end in zip(segment_starts, segment_ends):
                descriptors.extend(
                    (
                        env_id,
                        chunk_start,
                        min(chunk_start + sequence_length, segment_end),
                    )
                    for chunk_start in range(segment_start, segment_end, sequence_length)
                )
        return descriptors

    def _make_sequence_batch(
        self,
        descriptors: list[tuple[int, int, int]],
        *,
        sequence_length: int,
        restore_observation_dtypes: bool,
    ) -> M2MSequenceBatch:
        batch_size = len(descriptors)
        env_ids = torch.tensor([item[0] for item in descriptors], dtype=torch.long, device=self.device)
        start_steps = torch.tensor([item[1] for item in descriptors], dtype=torch.long, device=self.device)
        end_steps = torch.tensor([item[2] for item in descriptors], dtype=torch.long, device=self.device)
        sequence_lengths = end_steps - start_steps
        offsets = torch.arange(sequence_length, device=self.device).view(-1, 1)
        time_indices = start_steps.view(1, -1) + offsets
        masks_2d = time_indices < end_steps.view(1, -1)
        safe_time_indices = time_indices.clamp(max=self.step - 1)
        env_index = env_ids.view(1, -1).expand(sequence_length, -1)

        observations = TensorDict({}, batch_size=[sequence_length, batch_size], device=self.device)
        for key in self._student_obs_keys:
            selected = self.student_observations[key][safe_time_indices, env_index].clone()
            padding_mask = ~masks_2d
            for _ in range(selected.ndim - 2):
                padding_mask = padding_mask.unsqueeze(-1)
            selected.masked_fill_(padding_mask, 0)
            if restore_observation_dtypes:
                selected = selected.to(self._source_obs_dtypes[self._key_path(key)])
            observations[key] = selected

        def select_and_pad(source: torch.Tensor) -> torch.Tensor:
            selected = source[safe_time_indices, env_index].clone()
            padding_mask = ~masks_2d
            for _ in range(selected.ndim - 2):
                padding_mask = padding_mask.unsqueeze(-1)
            selected.masked_fill_(padding_mask, 0)
            return selected

        return M2MSequenceBatch(
            student_observations=observations,
            teacher_latent_A=select_and_pad(self.teacher_latent_A),
            teacher_action_mean=select_and_pad(self.teacher_action_mean),
            dones=select_and_pad(self.dones),
            episode_starts=select_and_pad(self.episode_starts),
            masks=masks_2d.unsqueeze(-1),
            initial_student_hidden=self.student_hidden_states[start_steps, env_ids].clone(),
            env_ids=env_ids,
            start_steps=start_steps,
            sequence_lengths=sequence_lengths,
        )

    def sequence_mini_batch_generator(
        self,
        *,
        num_mini_batches: int,
        sequence_length: int,
        num_epochs: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
        restore_observation_dtypes: bool = False,
    ) -> Generator[M2MSequenceBatch, None, None]:
        """Yield padded minibatches that never cross done/reset boundaries.

        Compressed leaves remain in their storage dtype by default so the
        student can cast at its encoder boundary.  Set
        ``restore_observation_dtypes=True`` for an explicit batch-time cast to
        the source observation dtype.
        """
        num_mini_batches = self._positive_int("num_mini_batches", num_mini_batches)
        sequence_length = self._positive_int("sequence_length", sequence_length)
        num_epochs = self._positive_int("num_epochs", num_epochs)
        if type(shuffle) is not bool:
            raise ValueError(f"shuffle must be an explicit bool, got {shuffle!r}.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"seed must be an integer or None, got {seed!r}.")
        if type(restore_observation_dtypes) is not bool:
            raise ValueError(
                f"restore_observation_dtypes must be an explicit bool, got {restore_observation_dtypes!r}."
            )

        descriptors = self._sequence_descriptors(sequence_length)
        if num_mini_batches > len(descriptors):
            raise ValueError(f"num_mini_batches={num_mini_batches} exceeds the {len(descriptors)} available sequences.")
        rng = None
        if seed is not None:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed)

        for _ in range(num_epochs):
            if shuffle:
                order = torch.randperm(len(descriptors), generator=rng).tolist()
                epoch_descriptors = [descriptors[index] for index in order]
            else:
                epoch_descriptors = descriptors
            base_size, remainder = divmod(len(epoch_descriptors), num_mini_batches)
            cursor = 0
            for batch_index in range(num_mini_batches):
                current_size = base_size + int(batch_index < remainder)
                batch_descriptors = epoch_descriptors[cursor : cursor + current_size]
                cursor += current_size
                yield self._make_sequence_batch(
                    batch_descriptors,
                    sequence_length=sequence_length,
                    restore_observation_dtypes=restore_observation_dtypes,
                )


__all__ = [
    "M2MSequenceBatch",
    "M2MSequenceRolloutStorage",
    "M2MSequenceTransition",
    "M2MStorageMemoryComparison",
    "M2MStorageMemoryEstimate",
]
