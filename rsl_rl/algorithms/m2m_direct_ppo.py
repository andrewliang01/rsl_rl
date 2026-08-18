# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-PPO baseline for the map-free C07 recurrent actor.

The algorithm deliberately keeps ordinary PPO rollout, GAE, clipped losses,
recurrent mini-batches, and critic updates.  Its only special responsibilities
are enforcing the F14 frozen-control boundary, excluding frozen M90 parameters
from the optimizer, and saving a resumable checkpoint without copying the
externally identified M90 artifact bytes.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_builders import construct_single_critic_algorithm
from rsl_rl.models import M2MMapFreeRecurrentStudent, M2MSequenceCompatibleCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


_CHECKPOINT_SCHEMA = "m2m_direct_ppo_v2"
_RUNNER_PROGRESS_SCHEMA = "m2m_direct_ppo_runner_progress_v1"
_TRAINABLE_ACTOR_PREFIXES = (
    "frame_tokenizer.",
    "gru.",
    "latent_head.",
)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


class M2MDirectPPO(PPO):
    """Standard PPO with an exact C07-GRU trainable-parameter boundary."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        storage: RolloutStorage,
        **kwargs: Any,
    ) -> None:
        for extension in ("rnd_cfg", "symmetry_cfg", "dwaq_cfg"):
            if kwargs.get(extension) is not None:
                raise ValueError(f"M2M direct PPO requires {extension}=None.")
        optimizer_name = kwargs.get("optimizer", "adam")
        if not isinstance(optimizer_name, str) or not optimizer_name:
            raise ValueError("M2M direct PPO optimizer must be an explicit non-empty string.")
        initial_learning_rate = kwargs.get("learning_rate", 0.001)
        if (
            type(initial_learning_rate) is not float
            or not math.isfinite(initial_learning_rate)
            or initial_learning_rate <= 0.0
        ):
            raise ValueError("M2M direct PPO learning_rate must be an exact positive float.")

        super().__init__(actor, critic, storage, **kwargs)
        self.optimizer_name = optimizer_name
        self.initial_learning_rate = initial_learning_rate
        self._validate_actor_contract()
        if not isinstance(self.critic, M2MSequenceCompatibleCritic):
            raise TypeError("M2M direct PPO requires the C09 M2MSequenceCompatibleCritic.")

        actor_trainable = tuple(
            parameter for parameter in self.actor.parameters() if parameter.requires_grad
        )
        critic_trainable = tuple(
            parameter for parameter in self.critic.parameters() if parameter.requires_grad
        )
        if not actor_trainable:
            raise ValueError("M2M direct PPO actor has no trainable student parameters.")
        if not critic_trainable:
            raise ValueError("M2M direct PPO critic has no trainable parameters.")
        trainable = (*actor_trainable, *critic_trainable)
        if len({id(parameter) for parameter in trainable}) != len(trainable):
            raise ValueError("M2M direct PPO actor/critic trainable parameters overlap.")

        # Generic PPO includes requires_grad=False parameters in its optimizer.
        # Replace it without changing any PPO loss or update semantics.
        self.optimizer = resolve_optimizer(optimizer_name)(trainable, lr=self.learning_rate)
        self._optimizer_parameter_ids = frozenset(id(parameter) for parameter in trainable)
        self.completed_updates = 0
        self._assert_optimizer_and_gradient_boundary()

    def _validate_actor_contract(self) -> None:
        if not isinstance(self.actor, M2MMapFreeRecurrentStudent):
            raise TypeError("M2M direct PPO requires the C07 M2MMapFreeRecurrentStudent.")
        for method in ("architecture_receipt", "parameter_audit"):
            if not callable(getattr(self.actor, method, None)):
                raise TypeError(f"M2M direct PPO actor must expose {method}().")
        if getattr(self.actor, "temporal_mode", None) != "gru" or getattr(
            self.actor, "is_recurrent", None
        ) is not True:
            raise ValueError("M2M direct PPO requires the C07 temporal_mode='gru' actor.")

        trainable_names = [
            name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
        ]
        unexpected = [
            name
            for name in trainable_names
            if not name.startswith(_TRAINABLE_ACTOR_PREFIXES)
        ]
        required_components = {
            prefix.rstrip(".")
            for prefix in _TRAINABLE_ACTOR_PREFIXES
            if any(name.startswith(prefix) for name in trainable_names)
        }
        if unexpected or required_components != {"frame_tokenizer", "gru", "latent_head"}:
            raise ValueError(
                "M2M direct PPO permits only tokenizer/GRU/A-head actor gradients; "
                f"trainable={trainable_names}, unexpected={unexpected}."
            )

        audit = self.actor.parameter_audit()
        if not isinstance(audit, Mapping):
            raise TypeError("M2M direct PPO actor parameter_audit() must return a mapping.")
        components = audit.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("M2M direct PPO actor audit lacks component counts.")
        frozen = components.get("frozen_ecmm")
        if not isinstance(frozen, Mapping) or frozen.get("trainable") != 0:
            raise ValueError("M2M direct PPO actor audit did not confirm a frozen ECMM core.")
        if audit.get("temporal_mode") != "gru" or audit.get("is_recurrent") is not True:
            raise ValueError("M2M direct PPO actor audit differs from the GRU contract.")

        core = getattr(self.actor, "ecmm_core", None)
        if not isinstance(core, nn.Module):
            raise TypeError("M2M direct PPO actor must expose its frozen ecmm_core.")
        if any(parameter.requires_grad for parameter in core.parameters()):
            raise ValueError("M2M direct PPO frozen ECMM core contains trainable parameters.")

    def _assert_optimizer_and_gradient_boundary(self) -> None:
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != self._optimizer_parameter_ids:
            raise RuntimeError("M2M direct PPO optimizer membership changed.")
        frozen = [
            (name, parameter)
            for name, parameter in self.actor.named_parameters()
            if not parameter.requires_grad
        ]
        frozen_in_optimizer = [name for name, value in frozen if id(value) in optimizer_ids]
        frozen_with_grad = [name for name, value in frozen if value.grad is not None]
        if frozen_in_optimizer or frozen_with_grad:
            raise RuntimeError(
                "M2M direct PPO frozen-control boundary violation: "
                f"in_optimizer={frozen_in_optimizer}, with_grad={frozen_with_grad}."
            )

    def update(self) -> dict[str, float]:
        """Run ordinary PPO and audit the frozen boundary after the update."""
        self._assert_optimizer_and_gradient_boundary()
        metrics = super().update()
        self._assert_optimizer_and_gradient_boundary()
        self.completed_updates += 1
        return metrics

    def _actor_architecture_receipt(self) -> dict[str, Any]:
        # C07's receipt is deliberately retained byte-for-byte relative to
        # F09.  Frozen artifact provenance is owned separately by
        # _frozen_artifact_receipt(), not rewritten into architecture.
        return copy.deepcopy(self.actor.architecture_receipt())

    def _frozen_artifact_receipt(self) -> dict[str, Any]:
        core = self.actor.ecmm_core
        digest = getattr(core, "checkpoint_sha256", None)
        state_key = getattr(core, "actor_state_dict_key", None)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("M2M direct PPO frozen ECMM core lacks a SHA-256 receipt.")
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("M2M direct PPO frozen ECMM core lacks an actor state key receipt.")
        return {
            "checkpoint_sha256": digest,
            "actor_state_dict_key": state_key,
            "checkpoint_path_source": "external_actor_configuration",
            "checkpoint_path_saved": False,
            "checkpoint_bytes_saved": False,
        }

    def _critic_signature(self) -> dict[str, Any]:
        return {
            "class": f"{type(self.critic).__module__}.{type(self.critic).__qualname__}",
            "state": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in self.critic.state_dict().items()
            },
        }

    @property
    def actor_trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
        )

    def _configuration_receipt(self) -> dict[str, Any]:
        return {
            "algorithm": type(self).__name__,
            "optimizer": self.optimizer_name,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "clip_param": self.clip_param,
            "gamma": self.gamma,
            "lam": self.lam,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "initial_learning_rate": self.initial_learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "use_clipped_value_loss": self.use_clipped_value_loss,
            "desired_kl": self.desired_kl,
            "schedule": self.schedule,
            "normalize_advantage_per_mini_batch": self.normalize_advantage_per_mini_batch,
            "actor_architecture": self._actor_architecture_receipt(),
            "actor_trainable_parameter_names": list(self.actor_trainable_parameter_names),
            "critic": self._critic_signature(),
            "extensions": {"rnd": None, "symmetry": None, "dwaq": None, "amp": None},
        }

    def save(self) -> dict[str, Any]:
        """Save only trainable student PPO state plus the frozen SHA receipt."""
        if type(self.completed_updates) is not int or self.completed_updates < 1:
            raise ValueError("M2M direct PPO save requires at least one completed update.")
        if type(self.storage.step) is not int or self.storage.step != 0:
            raise ValueError("M2M direct PPO save requires a cleared rollout boundary.")
        actor_trainable_state = {
            name: value.detach().cpu().clone()
            for name, value in self.actor.named_parameters()
            if value.requires_grad
        }
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "config_receipt": copy.deepcopy(self._configuration_receipt()),
            "frozen_artifact_receipt": self._frozen_artifact_receipt(),
            "algorithm_updates_completed": self.completed_updates,
            "actor_trainable_state_dict": actor_trainable_state,
            "critic_state_dict": _cpu_state_dict(self.critic),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
        }

    @staticmethod
    def _validate_parameter_state(
        state: object,
        live: Mapping[str, nn.Parameter],
    ) -> None:
        if not isinstance(state, Mapping) or set(state) != set(live):
            raise ValueError("M2M direct PPO actor trainable state keys differ.")
        for name, parameter in live.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"M2M direct PPO actor state {name!r} must be a tensor.")
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(
                    f"M2M direct PPO actor state {name!r} shape/dtype differs."
                )

    def _prevalidate_optimizer_state(self, state: dict[str, Any]) -> None:
        """Prove an optimizer payload can load and step without touching live state."""
        try:
            if len(self.optimizer.param_groups) != 1:
                raise ValueError("live optimizer must contain exactly one parameter group")
            live_parameters = list(self.optimizer.param_groups[0]["params"])
            candidate_parameters = [
                nn.Parameter(torch.zeros_like(parameter), requires_grad=True)
                for parameter in live_parameters
            ]
            candidate = resolve_optimizer(self.optimizer_name)(
                candidate_parameters,
                lr=self.initial_learning_rate,
            )
            candidate.load_state_dict(copy.deepcopy(state))
            restored_rates = {group["lr"] for group in candidate.param_groups}
            restored_rate = next(iter(restored_rates)) if len(restored_rates) == 1 else None
            if (
                type(restored_rate) is not float
                or not math.isfinite(restored_rate)
                or restored_rate <= 0.0
            ):
                raise ValueError("optimizer learning rate must be one positive finite float")
            candidate.zero_grad(set_to_none=True)
            for parameter in candidate_parameters:
                parameter.grad = torch.zeros_like(parameter)
            candidate.step()
        except Exception as error:
            raise ValueError("M2M direct PPO optimizer state is not loadable/step-safe.") from error

    def load(
        self,
        loaded_dict: dict[str, Any],
        load_cfg: dict | None,
        strict: bool,
    ) -> bool:
        """Strictly restore trainable actor, critic and optimizer state."""
        if strict is not True:
            raise ValueError("M2M direct PPO checkpoint loading is always strict.")
        full_load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": True,
            "iteration": True,
        }
        if load_cfg not in (None, full_load_cfg):
            raise ValueError("M2M direct PPO rejects partial checkpoint loading.")
        if not isinstance(loaded_dict, dict):
            raise TypeError("M2M direct PPO checkpoint root must be an exact dictionary.")
        required = {
            "schema",
            "config_receipt",
            "frozen_artifact_receipt",
            "algorithm_updates_completed",
            "actor_trainable_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
        }
        allowed_extras = {
            "iter",
            "infos",
            "training_receipt",
            "m2m_direct_ppo_progress",
        }
        missing = required.difference(loaded_dict)
        unexpected = set(loaded_dict).difference(required | allowed_extras)
        if missing or unexpected:
            raise ValueError(
                "M2M direct PPO checkpoint key mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        if loaded_dict["schema"] != _CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported M2M direct PPO schema {loaded_dict['schema']!r}.")
        if loaded_dict["config_receipt"] != self._configuration_receipt():
            raise ValueError("M2M direct PPO configuration receipt differs.")
        if loaded_dict["frozen_artifact_receipt"] != self._frozen_artifact_receipt():
            raise ValueError("M2M direct PPO frozen artifact receipt differs.")
        completed_updates = loaded_dict["algorithm_updates_completed"]
        if type(completed_updates) is not int or completed_updates < 1:
            raise ValueError(
                "M2M direct PPO algorithm_updates_completed must be a positive integer."
            )
        progress = loaded_dict.get("m2m_direct_ppo_progress")
        if progress is not None:
            expected_progress_keys = {
                "schema",
                "iter",
                "updates_completed",
                "resume_starts_at_update",
            }
            if type(progress) is not dict or set(progress) != expected_progress_keys:
                raise ValueError("M2M direct PPO runner progress receipt keys differ.")
            if progress["schema"] != _RUNNER_PROGRESS_SCHEMA:
                raise ValueError("M2M direct PPO runner progress schema differs.")
            iteration = loaded_dict.get("iter")
            if (
                type(iteration) is not int
                or iteration < 0
                or type(progress["iter"]) is not int
                or type(progress["updates_completed"]) is not int
                or type(progress["resume_starts_at_update"]) is not int
                or progress["iter"] != iteration
                or completed_updates != iteration + 1
                or progress["updates_completed"] != completed_updates
                or progress["resume_starts_at_update"] != completed_updates
            ):
                raise ValueError("M2M direct PPO runner progress receipt is inconsistent.")

        live_actor = {
            name: parameter
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad
        }
        actor_state = loaded_dict["actor_trainable_state_dict"]
        self._validate_parameter_state(actor_state, live_actor)
        critic_state = loaded_dict["critic_state_dict"]
        if not isinstance(critic_state, Mapping):
            raise ValueError("M2M direct PPO critic_state_dict must be a mapping.")
        live_critic = self.critic.state_dict()
        if set(critic_state) != set(live_critic):
            raise ValueError("M2M direct PPO critic state keys differ.")
        for name, expected in live_critic.items():
            value = critic_state[name]
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"M2M direct PPO critic state {name!r} shape/dtype differs.")
        optimizer_state = loaded_dict["optimizer_state_dict"]
        if not isinstance(optimizer_state, dict):
            raise ValueError("M2M direct PPO optimizer_state_dict must be an exact dictionary.")
        self._prevalidate_optimizer_state(optimizer_state)

        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in live_actor.items()
        }
        critic_before = {
            name: value.detach().clone()
            for name, value in live_critic.items()
        }
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        learning_rate_before = self.learning_rate
        try:
            with torch.no_grad():
                for name, parameter in live_actor.items():
                    parameter.copy_(actor_state[name].to(device=parameter.device))
            self.critic.load_state_dict(critic_state, strict=True)
            self.optimizer.load_state_dict(optimizer_state)
        except Exception:
            with torch.no_grad():
                for name, parameter in live_actor.items():
                    parameter.copy_(actor_before[name])
            self.critic.load_state_dict(critic_before, strict=True)
            self.optimizer.load_state_dict(optimizer_before)
            self.learning_rate = learning_rate_before
            raise
        restored_rates = {group["lr"] for group in self.optimizer.param_groups}
        if len(restored_rates) != 1:
            raise ValueError("M2M direct PPO optimizer learning rates disagree.")
        restored_rate = next(iter(restored_rates))
        if type(restored_rate) is not float or not math.isfinite(restored_rate) or restored_rate <= 0.0:
            raise ValueError("M2M direct PPO restored learning rate must be positive.")
        self.learning_rate = restored_rate
        self._validate_actor_contract()
        self._assert_optimizer_and_gradient_boundary()
        self.completed_updates = completed_updates
        return True

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> M2MDirectPPO:
        """Use the standard one-critic PPO builder with the F14 boundary."""
        algorithm_cfg = cfg.get("algorithm")
        if not isinstance(algorithm_cfg, Mapping):
            raise TypeError("M2M direct PPO requires an algorithm configuration mapping.")
        # The generic builder resolves extensions before invoking the
        # algorithm constructor and removes amp_cfg entirely.  Reject them at
        # this outer boundary so no auxiliary module or observation set can be
        # constructed for F14.
        for extension in ("rnd_cfg", "symmetry_cfg", "dwaq_cfg", "amp_cfg"):
            if algorithm_cfg.get(extension) is not None:
                raise ValueError(f"M2M direct PPO requires {extension}=None.")
        if algorithm_cfg.get("num_critics", 1) != 1:
            raise ValueError("M2M direct PPO requires exactly one C09 critic.")
        for flag in ("shared_critic", "share_cnn_encoders"):
            if algorithm_cfg.get(flag, False) is not False:
                raise ValueError(f"M2M direct PPO requires {flag}=False.")
        return construct_single_critic_algorithm(
            M2MDirectPPO,
            obs,
            env,
            cfg,
            device,
            actor_constructor_obs_set="student",
        )

    def audit(self) -> dict[str, Any]:
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        frozen_ids = {
            id(parameter)
            for parameter in self.actor.parameters()
            if not parameter.requires_grad
        }
        return {
            "phase": "F14_same_architecture_direct_ppo",
            "standard_ppo_objective": True,
            "actor_trainable_parameter_names": list(self.actor_trainable_parameter_names),
            "optimizer_matches_trainable_boundary": optimizer_ids == self._optimizer_parameter_ids,
            "frozen_actor_parameters_in_optimizer": len(optimizer_ids & frozen_ids),
            "frozen_actor_parameters_with_grad": sum(
                int(parameter.grad is not None)
                for parameter in self.actor.parameters()
                if not parameter.requires_grad
            ),
            "checkpoint": {
                "schema": _CHECKPOINT_SCHEMA,
                "trainable_actor_only": True,
                "critic_and_optimizer_saved": True,
                "frozen_ecmm_weights_saved": False,
                "frozen_artifact_path_saved": False,
            },
            "completed_updates": self.completed_updates,
        }


__all__ = ["M2MDirectPPO"]
