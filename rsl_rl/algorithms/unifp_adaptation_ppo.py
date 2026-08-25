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


class UniFPAdaptationPPO(PPO):
    """Run a weighted UniFP reconstruction update after every PPO minibatch."""

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
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.freeze_adaptation_after_iter = freeze_adaptation_after_iter
        self.obs_pred_group = obs_pred_group
        self.adaptation_weights = torch.tensor(adaptation_weights, device=self.device)
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
        self.adaptation_frozen = False
        self._update_count = 0
        adaptation_params = list(self.actor.encoder.parameters()) + list(self.actor.decoder.parameters())
        self.adaptation_optimizer = torch.optim.Adam(adaptation_params, lr=adaptation_learning_rate)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> UniFPAdaptationPPO:
        return construct_single_critic_algorithm(UniFPAdaptationPPO, obs, env, cfg, device)

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
        loss_dict.update(logs)
        return loss_dict

    def _freeze_adaptation(self) -> None:
        for module in (self.actor.encoder, self.actor.decoder):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        self.adaptation_frozen = True
        print(
            f"[UniFPAdaptationPPO] Frozen encoder+decoder at iter {self._update_count}; "
            "actor body and critic stay trainable."
        )

    def _empty_adaptation_logs(self) -> dict[str, float]:
        logs: dict[str, float] = {
            "adaptation": 0.0,
            "adaptation_frozen": float(self.adaptation_frozen),
            "_count": 0.0,
        }
        for name in self.adaptation_part_names:
            logs[name] = 0.0
            logs[name.replace("adaptation_", "adaptation_rmse_", 1)] = 0.0
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
            expected = 3 * int(self.adaptation_weights.numel())
            if pred.shape[-1] != expected:
                raise ValueError(
                    f"Adaptation pred dim {pred.shape[-1]} does not match "
                    f"3 * len(adaptation_weights) = {expected}."
                )
            loss = pred.new_zeros(())
            parts = []
            rmses = []
            for i, weight in enumerate(self.adaptation_weights.tolist()):
                sl = slice(i * 3, (i + 1) * 3)
                part = F.mse_loss(pred[:, sl] * weight, target[:, sl] * weight)
                parts.append(part.item())
                rmses.append(torch.sqrt(F.mse_loss(pred[:, sl], target[:, sl])).item())
                loss = loss + part
            if train:
                self.adaptation_optimizer.zero_grad()
                loss.backward()
                self.adaptation_optimizer.step()
        logs["adaptation"] += loss.item()
        for name, value, rmse in zip(self.adaptation_part_names, parts, rmses, strict=True):
            logs[name] += value
            logs[name.replace("adaptation_", "adaptation_rmse_", 1)] += rmse
        logs["_count"] += 1.0
