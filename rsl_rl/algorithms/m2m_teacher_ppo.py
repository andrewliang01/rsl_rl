# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO variant for the observed-history M2M teacher.

The formal teacher reuses a frozen M90 ECMM core and learns only its causal
``range+valid+age -> A64`` map encoder.  A generic PPO checkpoint serializes
the whole actor, which would duplicate the frozen artifact and blur the
teacher's information boundary.  This variant keeps PPO rollout/update
semantics unchanged while making save/load and optimizer membership explicit.
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
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


_CHECKPOINT_SCHEMA = "m2m_observed_history_teacher_ppo_v1"
_TRAINABLE_ACTOR_PREFIX = "map_encoder."


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


class M2MObservedHistoryTeacherPPO(PPO):
    """Train C06 without saving or optimizing the frozen ECMM weights."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        storage: RolloutStorage,
        **kwargs: Any,
    ) -> None:
        for extension in ("rnd_cfg", "symmetry_cfg", "dwaq_cfg"):
            if kwargs.get(extension) is not None:
                raise ValueError(f"M2M teacher PPO requires {extension}=None.")
        optimizer_name = kwargs.get("optimizer", "adam")
        if not isinstance(optimizer_name, str) or not optimizer_name:
            raise ValueError("M2M teacher PPO optimizer must be an explicit non-empty string.")
        initial_learning_rate = kwargs.get("learning_rate", 0.001)
        if (
            type(initial_learning_rate) is not float
            or not math.isfinite(initial_learning_rate)
            or initial_learning_rate <= 0.0
        ):
            raise ValueError("M2M teacher PPO learning_rate must be an exact positive float.")

        super().__init__(actor, critic, storage, **kwargs)
        self.optimizer_name = optimizer_name
        self.initial_learning_rate = initial_learning_rate
        self._validate_actor_contract()

        actor_trainable = tuple(
            parameter for parameter in self.actor.parameters() if parameter.requires_grad
        )
        critic_trainable = tuple(
            parameter for parameter in self.critic.parameters() if parameter.requires_grad
        )
        if not actor_trainable:
            raise ValueError("M2M teacher has no trainable map-encoder parameters.")
        if not critic_trainable:
            raise ValueError("M2M teacher PPO critic has no trainable parameters.")
        trainable = (*actor_trainable, *critic_trainable)
        if len({id(parameter) for parameter in trainable}) != len(trainable):
            raise ValueError("M2M teacher actor/critic trainable parameters unexpectedly overlap.")

        # Replace PPO's generic optimizer, whose param groups also contain the
        # frozen ECMM parameters.  No update semantics change: only parameters
        # that could ever receive gradients remain in the optimizer.
        self.optimizer = resolve_optimizer(optimizer_name)(trainable, lr=self.learning_rate)
        self._optimizer_parameter_ids = frozenset(id(parameter) for parameter in trainable)

    def _validate_actor_contract(self) -> None:
        for method in ("checkpoint_state", "load_checkpoint_state", "parameter_audit"):
            if not callable(getattr(self.actor, method, None)):
                raise TypeError(f"M2M teacher actor must expose {method}().")
        trainable_names = [
            name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
        ]
        unexpected = [
            name for name in trainable_names if not name.startswith(_TRAINABLE_ACTOR_PREFIX)
        ]
        if not trainable_names or unexpected:
            raise ValueError(
                "M2M teacher PPO permits only map_encoder actor gradients; "
                f"trainable={trainable_names}, unexpected={unexpected}."
            )
        audit = self.actor.parameter_audit()
        if not isinstance(audit, Mapping) or audit.get("only_map_encoder_trainable") is not True:
            raise ValueError("M2M teacher actor parameter audit did not confirm the freeze boundary.")

    def _critic_signature(self) -> dict[str, Any]:
        return {
            "class": f"{type(self.critic).__module__}.{type(self.critic).__qualname__}",
            "state": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in self.critic.state_dict().items()
            },
        }

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
            "critic": self._critic_signature(),
            "actor_gradient_prefix": _TRAINABLE_ACTOR_PREFIX,
            "frozen_ecmm_weights_saved": False,
        }

    def teacher_artifact(self) -> dict[str, Any]:
        """Return the exact C06 map-only artifact for student distillation."""
        artifact = self.actor.checkpoint_state()
        if not isinstance(artifact, dict):
            raise TypeError("M2M teacher checkpoint_state() must return an exact dictionary.")
        return copy.deepcopy(artifact)

    def save(self) -> dict[str, Any]:
        """Serialize resumable PPO state without frozen actor weights."""
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "config_receipt": copy.deepcopy(self._configuration_receipt()),
            "teacher_artifact": self.teacher_artifact(),
            "critic_state_dict": _cpu_state_dict(self.critic),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
        }

    def load(
        self,
        loaded_dict: dict[str, Any],
        load_cfg: dict | None,
        strict: bool,
    ) -> bool:
        """Strictly restore map encoder, critic and optimizer as one unit."""
        if strict is not True:
            raise ValueError("M2M teacher PPO checkpoint loading is always strict.")
        full_load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": True,
            "iteration": True,
        }
        if load_cfg not in (None, full_load_cfg):
            raise ValueError("M2M teacher PPO rejects partial checkpoint loading.")
        if not isinstance(loaded_dict, dict):
            raise TypeError("M2M teacher PPO checkpoint root must be an exact dictionary.")
        required = {
            "schema",
            "config_receipt",
            "teacher_artifact",
            "critic_state_dict",
            "optimizer_state_dict",
        }
        allowed_extras = {"iter", "infos", "training_receipt"}
        missing = required.difference(loaded_dict)
        unexpected = set(loaded_dict).difference(required | allowed_extras)
        if missing or unexpected:
            raise ValueError(
                "M2M teacher PPO checkpoint key mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        if loaded_dict["schema"] != _CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported M2M teacher PPO schema {loaded_dict['schema']!r}.")
        if loaded_dict["config_receipt"] != self._configuration_receipt():
            raise ValueError("M2M teacher PPO configuration receipt differs from the live algorithm.")

        self.actor.load_checkpoint_state(loaded_dict["teacher_artifact"])
        critic_state = loaded_dict["critic_state_dict"]
        if not isinstance(critic_state, Mapping):
            raise ValueError("critic_state_dict must be a mapping.")
        self.critic.load_state_dict(critic_state, strict=True)
        optimizer_state = loaded_dict["optimizer_state_dict"]
        if not isinstance(optimizer_state, dict):
            raise ValueError("optimizer_state_dict must be an exact dictionary.")
        self.optimizer.load_state_dict(optimizer_state)
        restored_rates = {group["lr"] for group in self.optimizer.param_groups}
        if len(restored_rates) != 1:
            raise ValueError("M2M teacher PPO optimizer param groups disagree on restored learning rate.")
        restored_rate = next(iter(restored_rates))
        if type(restored_rate) is not float or not math.isfinite(restored_rate) or restored_rate <= 0.0:
            raise ValueError("M2M teacher PPO restored learning rate must be an exact positive float.")
        self.learning_rate = restored_rate
        self._validate_actor_contract()
        self.actor.train()
        return True

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> M2MObservedHistoryTeacherPPO:
        """Use the standard single-critic PPO builder with this exact class."""
        return construct_single_critic_algorithm(
            M2MObservedHistoryTeacherPPO,
            obs,
            env,
            cfg,
            device,
        )

    def audit(self) -> dict[str, Any]:
        actor_trainable = [
            name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
        ]
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
            "phase": "F07_observed_history_teacher_ppo",
            "configuration": self._configuration_receipt(),
            "actor_trainable_parameter_names": actor_trainable,
            "optimizer_matches_trainable_boundary": optimizer_ids == self._optimizer_parameter_ids,
            "frozen_actor_parameters_in_optimizer": len(optimizer_ids & frozen_ids),
            "checkpoint": {
                "schema": _CHECKPOINT_SCHEMA,
                "teacher_artifact_field": "teacher_artifact",
                "map_encoder_saved": True,
                "critic_and_optimizer_saved_for_resume": True,
                "frozen_ecmm_weights_saved": False,
            },
        }


__all__ = ["M2MObservedHistoryTeacherPPO"]
