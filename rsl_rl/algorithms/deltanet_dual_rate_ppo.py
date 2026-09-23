# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Short on-policy PPO updates with independent long-window DeltaNet supervision."""

from __future__ import annotations

import math
import torch
from collections.abc import Generator
from itertools import chain
from tensordict import TensorDict
from torch import nn

from rsl_rl.algorithms.deltanet_ppo import DeltaNetPPO
from rsl_rl.models.mid360_deltanet import MID360DeltaNetActor, SequenceElevationHistoryCritic
from rsl_rl.storage.deltanet_estimator_storage import DeltaNetEstimatorStorage, DeltaNetPolicyStorage
from rsl_rl.storage.rollout_storage import RolloutStorage


class DeltaNetDualRatePPO(DeltaNetPPO):
    """Freeze rollout estimates for PPO; train the estimator only from labels.

    An estimator update is delayed to the next complete PPO boundary. Surplus
    samples are retained, so 24/320 gives update intervals of 336/312/312 steps.
    """

    cached_input_key = "_deltanet_policy_input"

    def __init__(
        self,
        actor: MID360DeltaNetActor,
        critic: SequenceElevationHistoryCritic,
        storage: DeltaNetPolicyStorage,
        *,
        estimator_window_steps: int = 320,
        estimator_num_learning_epochs: int = 3,
        estimator_num_mini_batches: int = 4,
        estimator_learning_rate: float = 1.0e-3,
        **kwargs: object,
    ) -> None:
        """Create disjoint optimizers and an independent supervised sequence buffer."""
        super().__init__(actor, critic, storage, **kwargs)
        for name, value in (
            ("estimator_window_steps", estimator_window_steps),
            ("estimator_num_learning_epochs", estimator_num_learning_epochs),
            ("estimator_num_mini_batches", estimator_num_mini_batches),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(estimator_learning_rate) or estimator_learning_rate <= 0:
            raise ValueError("estimator_learning_rate must be finite and positive.")
        if storage.num_envs % estimator_num_mini_batches:
            raise ValueError("Estimator mini-batches must evenly divide per-rank environments.")
        if storage.num_transitions_per_env > self.tbptt_steps:
            raise ValueError("PPO rollout must not exceed the estimator gradient window.")
        self.estimator_num_learning_epochs = estimator_num_learning_epochs
        self.estimator_num_mini_batches = estimator_num_mini_batches
        self.estimator_updates = 0
        self.ppo_updates = 0
        self.dual_rate_config = {
            "ppo_steps": storage.num_transitions_per_env,
            "estimator_window_steps": estimator_window_steps,
            "estimator_num_learning_epochs": estimator_num_learning_epochs,
            "estimator_num_mini_batches": estimator_num_mini_batches,
            "estimator_learning_rate": estimator_learning_rate,
        }
        # Estimator parameters never participate in the policy optimizer.
        self.policy_parameters = [p for name, p in actor.named_parameters() if not name.startswith("estimator.")]
        self.optimizer = type(self.optimizer)(
            chain(self.policy_parameters, critic.parameters()),
            **self.optimizer.defaults,
        )
        self.estimator_optimizer = torch.optim.Adam(actor.estimator.parameters(), lr=estimator_learning_rate)
        template = self._estimator_inputs(storage.source_template)
        state = actor.estimator.initial_state(storage.num_envs, template["proprio"])
        self.estimator_storage = DeltaNetEstimatorStorage(
            template,
            state,
            estimator_window_steps,
            self.tbptt_steps,
            storage.num_transitions_per_env,
        )
        del storage.source_template

    @staticmethod
    def _validate_storage(storage: DeltaNetPolicyStorage, critic: SequenceElevationHistoryCritic) -> None:
        if not isinstance(storage, DeltaNetPolicyStorage) or critic.is_recurrent:
            raise ValueError("Dual-rate DeltaNet requires policy storage and a feed-forward critic.")

    @classmethod
    def _make_storage(
        cls,
        training_type: str,
        num_envs: int,
        steps: int,
        obs: TensorDict,
        actions_shape: list[int],
        device: str,
        *,
        tbptt_steps: int,
        actor: MID360DeltaNetActor,
        critic: SequenceElevationHistoryCritic,
    ) -> DeltaNetPolicyStorage:
        inputs = obs.select(*critic.obs_groups, critic.elevation_set).clone()
        inputs[cls.cached_input_key] = torch.zeros(num_envs, actor._get_latent_dim(), device=device)
        storage = DeltaNetPolicyStorage(
            training_type,
            num_envs,
            steps,
            inputs,
            actions_shape,
            device,
            store_next_observations=False,
        )
        storage.tbptt_steps = tbptt_steps
        storage.source_template = obs
        return storage

    def _estimator_inputs(self, obs: TensorDict) -> TensorDict:
        target_map = obs[self.map_target_set]
        if self.map_target_history_index is not None:
            target_map = target_map[:, self.map_target_history_index]
        return TensorDict(
            {
                "proprio": self.actor.obs_normalizer(obs[self.actor.proprio_set]).detach(),
                "panorama": obs[self.actor.panorama_set].detach(),
                "map": target_map.flatten(1).detach(),
                "velocity": obs[self.velocity_target_set].detach(),
            },
            batch_size=obs.batch_size,
            device=obs.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample once and cache the behavior policy inputs for flat PPO replay."""
        actions = super().act(obs)
        inputs = self._estimator_inputs(obs)
        self.estimator_storage.record_inputs(inputs, self.transition.hidden_states[0])
        policy_obs = obs.select(*self.critic.obs_groups, self.critic.elevation_set)
        policy_obs[self.cached_input_key] = torch.cat(
            (inputs["proprio"], self.actor.last_map, self.actor.last_velocity),
            dim=-1,
        ).detach()
        self.transition.observations = policy_obs
        self.transition.hidden_states = (None, None)
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Record termination flags in both buffers and reset online memory."""
        self.estimator_storage.finish_step(dones)
        super().process_env_step(obs, rewards, dones, extras)

    def _batches(self) -> Generator[RolloutStorage.Batch, None, None]:
        return self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    def _forward_actor(self, batch: RolloutStorage.Batch) -> None:
        # Replay precisely the normalized inputs used by the behavior policy.
        latent = batch.observations[self.cached_input_key]
        self.actor.distribution.update(self.actor.mlp(latent))

    def _auxiliary_losses(self, batch: RolloutStorage.Batch) -> tuple[torch.Tensor, torch.Tensor]:
        zero = batch.values.new_zeros(())
        return zero, zero

    def update(self) -> dict[str, float]:
        """Update PPO, then update the estimator only when its buffer is ready."""
        started = self._profile_mark()
        metrics = super().update()
        for key in ("map_mse", "velocity_mse", "map_rmse_m", "velocity_rmse_mps"):
            metrics.pop(key)
        if self.profile_learning:
            metrics["profile_ppo_s"] = self._profile_mark() - started
        self.ppo_updates += 1
        ready = self.estimator_storage.step >= self.estimator_storage.window
        if ready:
            metrics.update(self._update_estimator())
        if self.profile_learning and self.is_multi_gpu:
            names = [key for key in metrics if key.startswith("profile_")]
            timings = torch.tensor([metrics[key] for key in names], device=self.device)
            torch.distributed.all_reduce(timings, op=torch.distributed.ReduceOp.MAX)
            metrics.update(zip(names, timings.tolist()))
        metrics.update({
            "estimator_updated": float(ready),
            "estimator_updates": self.estimator_updates,
            "estimator_buffer_steps": self.estimator_storage.step,
        })
        return metrics

    def _update_estimator(self) -> dict[str, float]:
        mark = self._profile_mark()
        sums = torch.zeros(4, device=self.device)
        updates = 0
        for inputs, resets, state in self.estimator_storage.batches(
            self.estimator_num_mini_batches,
            self.estimator_num_learning_epochs,
        ):
            height, velocity, _ = self.actor.estimator(
                inputs["proprio"],
                inputs["panorama"],
                state,
                resets,
            )
            target = inputs["map"]
            finite = torch.isfinite(target)
            error = height - torch.where(finite, target, 0.0)
            map_sum = torch.where(finite, error.square(), 0.0).sum()
            map_count = finite.sum().clamp_min(1)
            velocity_sum = (velocity - inputs["velocity"]).square().sum()
            velocity_count = velocity.numel()
            loss = self.map_loss_coef * map_sum / map_count
            loss = loss + self.velocity_loss_coef * velocity_sum / velocity_count
            self.estimator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_gradients(self.actor.estimator.parameters())
            nn.utils.clip_grad_norm_(self.actor.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()
            sums += torch.stack((map_sum.detach(), map_count, velocity_sum.detach(), sums.new_tensor(velocity_count)))
            updates += 1
        self.estimator_optimizer.zero_grad(set_to_none=True)
        self.estimator_updates += 1
        if self.is_multi_gpu:
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        metrics = {}
        if self.profile_learning:
            metrics["profile_estimator_train_s"] = self._profile_mark() - mark
        mark = self._profile_mark()
        self._refresh_state_and_consume()
        if self.profile_learning:
            metrics["profile_estimator_refresh_s"] = self._profile_mark() - mark
        # Transfer metrics after refresh so pending replay kernels are accounted
        # for in learning time even when phase profiling is disabled.
        map_mse, velocity_mse = (sums[0] / sums[1]).item(), (sums[2] / sums[3]).item()
        metrics.update({
            "map_mse": map_mse,
            "velocity_mse": velocity_mse,
            "map_rmse_m": math.sqrt(map_mse),
            "velocity_rmse_mps": math.sqrt(velocity_mse),
            "estimator_optimizer_steps": updates,
        })
        return metrics

    @torch.no_grad()
    def _refresh_state_and_consume(self) -> None:
        storage = self.estimator_storage
        start = storage.window - storage.tbptt_steps
        tail_initial = self.actor.estimator.initial_state(storage.data.batch_size[1], storage.data["proprio"])
        live_state = torch.empty_like(tail_initial)
        # Bound peak refresh memory by the same environment batch as training.
        for ids in torch.arange(storage.data.batch_size[1], device=self.device).chunk(self.estimator_num_mini_batches):
            state = storage.states[start // storage.tbptt_steps, :, ids]
            for first, stop in ((start, storage.window), (storage.window, storage.step)):
                if first == stop:
                    continue
                inputs = storage.data[first:stop, ids]
                _, _, state = self.actor.estimator(
                    inputs["proprio"],
                    inputs["panorama"],
                    state,
                    storage.reset_before(first, stop, ids),
                )
                state = state * (~storage.dones[stop - 1, ids])[None, :, None]
                if stop == storage.window:
                    tail_initial[:, ids] = state
            live_state[:, ids] = state
        # The starting snapshot remains detached, as with conventional TBPTT;
        # replay with new weights reduces state staleness without clearing memory.
        self.actor.reset(hidden_state=live_state)
        storage.consume(tail_initial)

    def save(self) -> dict:
        """Save both optimizers and the dual-rate configuration for strict resume."""
        checkpoint = super().save()
        checkpoint["deltanet_dual_rate_config"] = self.dual_rate_config
        checkpoint["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        checkpoint["estimator_updates"] = self.estimator_updates
        checkpoint["ppo_updates"] = self.ppo_updates
        return checkpoint

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore training state while discarding samples from restarted episodes."""
        if strict and loaded_dict.get("deltanet_dual_rate_config") != self.dual_rate_config:
            raise ValueError("Checkpoint dual-rate schedule differs; joint PPO checkpoints require a new run.")
        restore_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        if restore_optimizer and "estimator_optimizer_state_dict" not in loaded_dict:
            raise ValueError("Dual-rate resume requires the estimator optimizer state.")
        resumed = super().load(loaded_dict, load_cfg, strict)
        if restore_optimizer:
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        if resumed:
            self.estimator_updates = loaded_dict.get("estimator_updates", 0)
            self.ppo_updates = loaded_dict.get("ppo_updates", 0)
        # Simulation episodes are restarted on load; do not reuse their old buffer/state.
        self.estimator_storage.step = 0
        return resumed


__all__ = ["DeltaNetDualRatePPO"]
