# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with UniFP encoder-decoder adaptation supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_builders import construct_single_critic_algorithm
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


_ADAPTATION_PART_NAMES = (
    "adaptation_base_velocity",
    "adaptation_gripper_pos",
    "adaptation_force_ee",
    "adaptation_force_base",
)
_ADAPTATION_PART_NAMES_ORN = (
    "adaptation_base_velocity",
    "adaptation_gripper_pos",
    "adaptation_gripper_orn",
)


def _adaptation_part_names(num_chunks: int) -> tuple[str, ...]:
    """Return legacy manipulation loss names for backward compatibility."""
    if num_chunks == 3:
        return _ADAPTATION_PART_NAMES_ORN
    return _ADAPTATION_PART_NAMES[:num_chunks]


class UniFPAdaptationMixin:
    """Reusable UniFP auxiliary-update lifecycle for PPO-family algorithms."""

    def _init_adaptation(
        self,
        freeze_adaptation_after_iter: int | None = 4000,
        adaptation_learning_rate: float = 5.0e-6,
        obs_pred_group: str = "obs_pred",
        adaptation_weights: tuple[float, ...] = (0.6, 0.8, 1.0, 1.0),
        adaptation_part_names: tuple[str, ...] | None = None,
        adaptation_part_dims: tuple[int, ...] | None = None,
        adaptation_loss_types: tuple[str, ...] | None = None,
        reconstruction_obs_group: str | None = None,
        reconstruction_weight: float = 0.0,
    ) -> None:
        self.freeze_adaptation_after_iter = freeze_adaptation_after_iter
        self.obs_pred_group = obs_pred_group
        self.adaptation_weights = torch.tensor(adaptation_weights, device=self.device)
        self._legacy_chunk_weighting = adaptation_part_dims is None
        self.adaptation_part_dims = (
            (3,) * len(adaptation_weights) if adaptation_part_dims is None else tuple(adaptation_part_dims)
        )
        if len(self.adaptation_part_dims) != len(adaptation_weights):
            raise ValueError(
                "len(adaptation_part_dims) must match len(adaptation_weights), got "
                f"{len(self.adaptation_part_dims)} and {len(adaptation_weights)}."
            )
        if any(dim <= 0 for dim in self.adaptation_part_dims):
            raise ValueError(f"adaptation_part_dims must be positive, got {self.adaptation_part_dims}.")
        self.adaptation_loss_types = (
            ("mse",) * len(adaptation_weights)
            if adaptation_loss_types is None
            else tuple(adaptation_loss_types)
        )
        if len(self.adaptation_loss_types) != len(adaptation_weights):
            raise ValueError(
                "len(adaptation_loss_types) must match len(adaptation_weights), got "
                f"{len(self.adaptation_loss_types)} and {len(adaptation_weights)}."
            )
        supported_losses = {"mse", "bce_logits"}
        unsupported_losses = set(self.adaptation_loss_types) - supported_losses
        if unsupported_losses:
            raise ValueError(
                f"Unsupported adaptation loss types {sorted(unsupported_losses)}; expected {sorted(supported_losses)}."
            )
        if adaptation_part_names is None:
            self.adaptation_part_names = _adaptation_part_names(len(adaptation_weights))
        else:
            if len(adaptation_part_names) != len(adaptation_weights):
                raise ValueError(
                    "len(adaptation_part_names) must match len(adaptation_weights), got "
                    f"{len(adaptation_part_names)} and {len(adaptation_weights)}."
                )
            self.adaptation_part_names = tuple(
                name if name.startswith("adaptation_") else f"adaptation_{name}"
                for name in adaptation_part_names
            )
        self.reconstruction_obs_group = reconstruction_obs_group
        self.reconstruction_weight = float(reconstruction_weight)
        if self.reconstruction_weight < 0.0:
            raise ValueError("reconstruction_weight must be non-negative.")
        has_reconstruction_decoder = getattr(self.actor, "reconstruction_decoder", None) is not None
        if (self.reconstruction_obs_group is None) != (not has_reconstruction_decoder):
            raise ValueError(
                "reconstruction_obs_group and the actor reconstruction decoder must be configured together."
            )
        self.adaptation_frozen = False
        self._update_count = 0
        if hasattr(self.actor, "adaptation_modules"):
            self._adaptation_modules = tuple(self.actor.adaptation_modules())
        else:
            self._adaptation_modules = (self.actor.encoder, self.actor.decoder)
        self._adaptation_params = [
            parameter
            for module in self._adaptation_modules
            for parameter in module.parameters()
        ]
        self.adaptation_optimizer = torch.optim.Adam(self._adaptation_params, lr=adaptation_learning_rate)

    def update(self) -> dict[str, float]:
        self._update_count += 1
        if (
            self.freeze_adaptation_after_iter is not None
            and not self.adaptation_frozen
            and self._update_count >= self.freeze_adaptation_after_iter
        ):
            self._freeze_adaptation()

        logs = self._empty_adaptation_logs()
        original_generator = self.storage.mini_batch_generator
        ppo_consumed = False

        def wrapped_generator(num_mini_batches, num_learning_epochs=8):
            nonlocal ppo_consumed
            generator = original_generator(num_mini_batches, num_learning_epochs)
            run_adapt = not ppo_consumed
            train_adapt = run_adapt and not self.adaptation_frozen
            ppo_consumed = True
            prev = None
            for batch in generator:
                if run_adapt and prev is not None:
                    self._adaptation_step(prev, logs, train=train_adapt)
                prev = batch
                yield batch
            if run_adapt and prev is not None:
                self._adaptation_step(prev, logs, train=train_adapt)

        self.storage.mini_batch_generator = wrapped_generator
        try:
            loss_dict = super().update()
        finally:
            self.storage.mini_batch_generator = original_generator

        count = logs.pop("_count")
        if count > 0:
            for key, value in list(logs.items()):
                if key == "adaptation_frozen":
                    continue
                logs[key] = value / count
        self._average_adaptation_logs(logs)
        loss_dict.update(logs)
        return loss_dict

    def _freeze_adaptation(self) -> None:
        # PPO synchronizes its own gradients, but the adaptation optimizer is
        # independent.  Broadcast once more at the optional freeze boundary so
        # every rank freezes the exact same encoder/decoder parameters.
        self._broadcast_adaptation_parameters()
        for module in self._get_adaptation_modules():
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        self.adaptation_frozen = True
        print(
            f"[{type(self).__name__}] Frozen encoder+decoder at iter {self._update_count}; "
            "actor body and critic stay trainable."
        )

    def train_mode(self) -> None:
        """Keep optionally frozen adaptation modules in evaluation mode."""
        super().train_mode()
        if self.adaptation_frozen:
            for module in self._get_adaptation_modules():
                module.eval()

    def _get_adaptation_modules(self) -> tuple[torch.nn.Module, ...]:
        """Return all supervised modules, including legacy actors used by old checkpoints/tests."""
        if hasattr(self, "_adaptation_modules"):
            return self._adaptation_modules
        return (self.actor.encoder, self.actor.decoder)

    def _empty_adaptation_logs(self) -> dict[str, float]:
        logs: dict[str, float] = {
            "adaptation": 0.0,
            "adaptation_frozen": float(self.adaptation_frozen),
            "_count": 0.0,
        }
        for name, loss_type in zip(
            self.adaptation_part_names,
            self.adaptation_loss_types,
            strict=True,
        ):
            logs[name] = 0.0
            metric_prefix = "adaptation_accuracy_" if loss_type == "bce_logits" else "adaptation_rmse_"
            logs[name.replace("adaptation_", metric_prefix, 1)] = 0.0
        if self.reconstruction_obs_group is not None:
            logs["adaptation_next_state_reconstruction"] = 0.0
            logs["adaptation_rmse_next_state_reconstruction"] = 0.0
        return logs

    def _adaptation_step(self, batch, logs: dict[str, float], *, train: bool) -> None:
        if self.obs_pred_group not in batch.observations.keys():
            raise KeyError(
                f"Adaptation target group '{self.obs_pred_group}' is missing from rollout observations."
            )
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            pred = self.actor.predict_obs_pred(batch.observations)
            target = batch.observations[self.obs_pred_group]
            if pred.shape[-1] != target.shape[-1]:
                raise ValueError(
                    f"Adaptation pred dim {pred.shape[-1]} does not match target dim {target.shape[-1]}."
                )
            expected = sum(self.adaptation_part_dims)
            if pred.shape[-1] != expected:
                raise ValueError(
                    f"Adaptation pred dim {pred.shape[-1]} does not match "
                    f"sum(adaptation_part_dims) = {expected}."
                )
            loss = pred.new_zeros(())
            parts = []
            metrics = []
            offset = 0
            for dim, loss_type, weight in zip(
                self.adaptation_part_dims,
                self.adaptation_loss_types,
                self.adaptation_weights.tolist(),
                strict=True,
            ):
                sl = slice(offset, offset + dim)
                pred_part = pred[:, sl]
                target_part = target[:, sl]
                if loss_type == "mse":
                    raw_part = F.mse_loss(pred_part, target_part)
                    metric = torch.sqrt(raw_part).item()
                    part = (
                        F.mse_loss(pred_part * weight, target_part * weight)
                        if self._legacy_chunk_weighting
                        else raw_part * weight
                    )
                else:
                    raw_part = F.binary_cross_entropy_with_logits(pred_part, target_part)
                    part = raw_part * weight
                    metric = ((pred_part >= 0.0) == (target_part >= 0.5)).float().mean().item()
                parts.append(part.item())
                metrics.append(metric)
                loss = loss + part
                offset += dim

            reconstruction_part = None
            reconstruction_rmse = None
            if self.reconstruction_obs_group is not None:
                if batch.next_observations is None:
                    raise ValueError("UniFP next-state reconstruction requires batch.next_observations.")
                if self.reconstruction_obs_group not in batch.next_observations.keys():
                    raise KeyError(
                        f"Reconstruction target group '{self.reconstruction_obs_group}' is missing from "
                        "next rollout observations."
                    )
                reconstruction_pred = self.actor.predict_reconstruction(batch.observations)
                reconstruction_target = batch.next_observations[self.reconstruction_obs_group]
                if reconstruction_pred.shape != reconstruction_target.shape:
                    raise ValueError(
                        f"Reconstruction pred shape {tuple(reconstruction_pred.shape)} does not match "
                        f"next target shape {tuple(reconstruction_target.shape)}."
                    )
                valid = torch.ones(
                    reconstruction_pred.shape[0],
                    dtype=torch.bool,
                    device=reconstruction_pred.device,
                )
                if batch.dones is not None:
                    valid &= ~batch.dones.reshape(batch.dones.shape[0], -1).any(dim=-1).bool()
                if torch.any(valid):
                    raw_reconstruction = F.mse_loss(
                        reconstruction_pred[valid],
                        reconstruction_target[valid],
                    )
                else:
                    raw_reconstruction = reconstruction_pred.sum() * 0.0
                reconstruction_part = raw_reconstruction * self.reconstruction_weight
                reconstruction_rmse = torch.sqrt(raw_reconstruction).item()
                loss = loss + reconstruction_part
            if train:
                self.adaptation_optimizer.zero_grad()
                loss.backward()
                self._reduce_adaptation_gradients()
                self.adaptation_optimizer.step()
        logs["adaptation"] += loss.item()
        for name, loss_type, value, metric in zip(
            self.adaptation_part_names,
            self.adaptation_loss_types,
            parts,
            metrics,
            strict=True,
        ):
            logs[name] += value
            metric_prefix = "adaptation_accuracy_" if loss_type == "bce_logits" else "adaptation_rmse_"
            logs[name.replace("adaptation_", metric_prefix, 1)] += metric
        if reconstruction_part is not None and reconstruction_rmse is not None:
            logs["adaptation_next_state_reconstruction"] += reconstruction_part.item()
            logs["adaptation_rmse_next_state_reconstruction"] += reconstruction_rmse
        logs["_count"] += 1.0

    def _reduce_adaptation_gradients(self) -> None:
        """Average UniFP encoder/decoder gradients across distributed ranks."""
        if self.is_multi_gpu:
            self._reduce_gradients(self._adaptation_params)

    def _average_adaptation_logs(self, logs: dict[str, float]) -> None:
        """Report globally averaged UniFP losses instead of rank-zero-only values."""
        if not self.is_multi_gpu:
            return
        metric_names = [name for name in logs if name != "adaptation_frozen"]
        metric_values = torch.tensor([logs[name] for name in metric_names], device=self.device)
        torch.distributed.all_reduce(metric_values, op=torch.distributed.ReduceOp.SUM)
        metric_values /= self.gpu_world_size
        for name, value in zip(metric_names, metric_values.tolist(), strict=True):
            logs[name] = value

    def _broadcast_adaptation_parameters(self) -> None:
        """Make all ranks use rank zero's UniFP parameters before freezing."""
        if not self.is_multi_gpu:
            return
        for module in self._get_adaptation_modules():
            for param in module.parameters():
                torch.distributed.broadcast(param.data, src=0)
            for buffer in module.buffers():
                torch.distributed.broadcast(buffer.data, src=0)

    def save(self) -> dict:
        """Save PPO and UniFP adaptation optimizer/lifecycle state."""
        saved_dict = super().save()
        saved_dict["adaptation_optimizer_state_dict"] = self.adaptation_optimizer.state_dict()
        saved_dict["adaptation_update_count"] = self._update_count
        saved_dict["adaptation_frozen"] = self.adaptation_frozen
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore PPO and UniFP adaptation optimizer/lifecycle state."""
        load_all = load_cfg is None
        load_optimizer = load_all or bool(load_cfg and load_cfg.get("optimizer"))
        load_iteration = load_all or bool(load_cfg and load_cfg.get("iteration"))
        loaded_iteration = super().load(loaded_dict, load_cfg, strict)

        if load_optimizer and "adaptation_optimizer_state_dict" in loaded_dict:
            self.adaptation_optimizer.load_state_dict(loaded_dict["adaptation_optimizer_state_dict"])

        if load_iteration:
            self._update_count = int(loaded_dict.get("adaptation_update_count", loaded_dict.get("iter", 0)))
            restored_frozen = bool(
                loaded_dict.get(
                    "adaptation_frozen",
                    self.freeze_adaptation_after_iter is not None
                    and self._update_count >= self.freeze_adaptation_after_iter,
                )
            )
            if restored_frozen:
                self._freeze_adaptation()
            else:
                for module in self._get_adaptation_modules():
                    for param in module.parameters():
                        param.requires_grad = True
                    module.train()
                self.adaptation_frozen = False

        return loaded_iteration


class UniFPAdaptationPPO(UniFPAdaptationMixin, PPO):
    """PPO with a weighted UniFP auxiliary update after every minibatch."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        freeze_adaptation_after_iter: int | None = 4000,
        adaptation_learning_rate: float = 5.0e-6,
        obs_pred_group: str = "obs_pred",
        adaptation_weights: tuple[float, ...] = (0.6, 0.8, 1.0, 1.0),
        adaptation_part_names: tuple[str, ...] | None = None,
        adaptation_part_dims: tuple[int, ...] | None = None,
        adaptation_loss_types: tuple[str, ...] | None = None,
        reconstruction_obs_group: str | None = None,
        reconstruction_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self._init_adaptation(
            freeze_adaptation_after_iter=freeze_adaptation_after_iter,
            adaptation_learning_rate=adaptation_learning_rate,
            obs_pred_group=obs_pred_group,
            adaptation_weights=adaptation_weights,
            adaptation_part_names=adaptation_part_names,
            adaptation_part_dims=adaptation_part_dims,
            adaptation_loss_types=adaptation_loss_types,
            reconstruction_obs_group=reconstruction_obs_group,
            reconstruction_weight=reconstruction_weight,
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "UniFPAdaptationPPO":
        return construct_single_critic_algorithm(UniFPAdaptationPPO, obs, env, cfg, device)
