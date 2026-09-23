# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with concurrent DeltaNet map and velocity supervision."""

from __future__ import annotations

import math

import torch
from torch import nn

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import DeltaNetRolloutStorage
from rsl_rl.utils import resolve_callable, unpad_trajectories


class DeltaNetPPO(PPO):
    """Jointly optimize locomotion, local-map recovery and base velocity."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        map_loss_coef: float = 1.0,
        velocity_loss_coef: float = 1.0,
        map_target_set: str = "height_scan_critic",
        map_target_history_index: int | None = -1,
        velocity_target_set: str = "deltanet_velocity_target",
        **kwargs,
    ) -> None:
        for extension in ("rnd_cfg", "symmetry_cfg", "dwaq_cfg"):
            if kwargs.get(extension) is not None:
                raise ValueError(f"DeltaNetPPO does not support {extension}.")
        for name, coefficient in (
            ("map_loss_coef", map_loss_coef),
            ("velocity_loss_coef", velocity_loss_coef),
        ):
            if not math.isfinite(coefficient) or coefficient <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(storage, DeltaNetRolloutStorage) or critic.is_recurrent:
            raise ValueError("DeltaNetPPO requires compact-state storage and a feed-forward critic.")
        super().__init__(actor, critic, storage, **kwargs)
        self.map_loss_coef = map_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.map_target_set = map_target_set
        self.map_target_history_index = map_target_history_index
        self.velocity_target_set = velocity_target_set

    def act(self, obs):
        if self.actor.get_hidden_state() is None:
            proprio = obs[self.actor.proprio_set]
            state = self.actor.estimator.initial_state(obs.batch_size[0], proprio)
            self.actor.reset(hidden_state=state)
        return super().act(obs)

    def save(self):
        checkpoint = super().save()
        checkpoint["deltanet_actor_config"] = self.actor.model_config
        checkpoint["deltanet_estimator_config"] = {
            "map_loss_coef": self.map_loss_coef,
            "velocity_loss_coef": self.velocity_loss_coef,
            "map_target_set": self.map_target_set,
            "map_target_history_index": self.map_target_history_index,
            "velocity_target_set": self.velocity_target_set,
        }
        return checkpoint

    def load(self, loaded_dict, load_cfg=None, strict=True):
        expected_estimator = {
            "map_loss_coef": self.map_loss_coef,
            "velocity_loss_coef": self.velocity_loss_coef,
            "map_target_set": self.map_target_set,
            "map_target_history_index": self.map_target_history_index,
            "velocity_target_set": self.velocity_target_set,
        }
        if strict and loaded_dict.get("deltanet_actor_config") != self.actor.model_config:
            raise ValueError("Checkpoint DeltaNet architecture differs from the task configuration.")
        if strict and loaded_dict.get("deltanet_estimator_config") != expected_estimator:
            raise ValueError("Checkpoint estimator supervision differs from the task configuration.")
        resumed = super().load(loaded_dict, load_cfg, strict)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.actor.reset()
        return resumed

    def update(self):
        metrics = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "map_mse": 0.0,
            "velocity_mse": 0.0,
        }
        updates = 0
        generator = self.storage.recurrent_mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for batch in generator:
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            log_prob = self.actor.get_output_log_prob(batch.actions)
            entropy = self.actor.output_entropy.mean()
            values = self.critic(batch.observations, masks=batch.masks)
            with torch.no_grad():
                kl_mean = self.actor.get_kl_divergence(
                    batch.old_distribution_params,
                    self.actor.output_distribution_params,
                ).mean()
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size
                if self.desired_kl is not None and self.schedule == "adaptive":
                    if self.gpu_global_rank == 0:
                        if kl_mean > 2.0 * self.desired_kl:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            advantage = batch.advantages.squeeze(-1)
            if self.normalize_advantage_per_mini_batch:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1.0e-8)
            ratio = (log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
            surrogate = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
            ).mean()
            value_loss = (values - batch.returns).square()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.maximum(value_loss, (clipped - batch.returns).square())
            value_loss = value_loss.mean()

            target_map = unpad_trajectories(
                batch.observations[self.map_target_set],
                batch.masks,
            )
            if self.map_target_history_index is not None:
                target_map = target_map[..., self.map_target_history_index, :, :]
            target_map = target_map.flatten(2).detach()
            target_velocity = unpad_trajectories(
                batch.observations[self.velocity_target_set],
                batch.masks,
            ).detach()
            finite = torch.isfinite(target_map)
            safe_target = torch.where(finite, target_map, 0.0)
            map_error = self.actor.last_map - safe_target
            map_loss = torch.where(finite, map_error.square(), 0.0).sum()
            map_loss = map_loss / finite.sum().clamp_min(1)
            velocity_loss = (self.actor.last_velocity - target_velocity).square().mean()

            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            loss = loss + self.map_loss_coef * map_loss + self.velocity_loss_coef * velocity_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            for name, value in (
                ("value", value_loss),
                ("surrogate", surrogate),
                ("entropy", entropy),
                ("kl", kl_mean),
                ("map_mse", map_loss),
                ("velocity_mse", velocity_loss),
            ):
                metrics[name] += value.item()
            updates += 1

        self.storage.clear()
        self.actor.detach_hidden_state()
        metrics = {name: value / updates for name, value in metrics.items()}
        if self.is_multi_gpu:
            metric_names = tuple(metrics)
            metric_values = torch.tensor(
                [metrics[name] for name in metric_names],
                device=self.device,
            )
            torch.distributed.all_reduce(metric_values, op=torch.distributed.ReduceOp.SUM)
            metric_values /= self.gpu_world_size
            metrics = dict(zip(metric_names, metric_values.tolist(), strict=True))
        metrics["map_rmse_m"] = math.sqrt(metrics["map_mse"])
        metrics["velocity_rmse_mps"] = math.sqrt(metrics["velocity_mse"])
        return metrics

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        actor_cfg, critic_cfg, algorithm_cfg = (
            cfg[name].copy() for name in ("actor", "critic", "algorithm")
        )
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        critic_class = resolve_callable(critic_cfg.pop("class_name"))
        algorithm_cfg.pop("class_name")
        for flag in ("shared_critic", "share_cnn_encoders", "amp_cfg"):
            if algorithm_cfg.pop(flag, None):
                raise ValueError(f"DeltaNetPPO does not support {flag}.")
        if algorithm_cfg.pop("num_critics", 1) != 1:
            raise ValueError("DeltaNetPPO requires exactly one critic.")
        algorithm_cfg.pop("reward_group_names", None)
        algorithm_cfg.pop("reward_group_weights", None)

        map_target_set = algorithm_cfg.get("map_target_set", "height_scan_critic")
        velocity_target_set = algorithm_cfg.get("velocity_target_set", "deltanet_velocity_target")
        required_sets = {
            actor_cfg.get("proprio_set", "policy"),
            actor_cfg.get("panorama_set", "height_scan_policy"),
            critic_cfg.get("proprio_set", "critic"),
            critic_cfg.get("elevation_set", "height_scan_critic"),
            map_target_set,
            velocity_target_set,
        }
        missing = sorted(required_sets.difference(obs.keys()))
        if missing:
            raise ValueError(f"DeltaNetPPO training observations are missing: {missing}.")
        map_shape = tuple(actor_cfg.get("map_shape", (28, 20)))
        if tuple(obs[map_target_set].shape[-2:]) != map_shape:
            raise ValueError(
                f"Map target must end in {map_shape}, got {tuple(obs[map_target_set].shape)}."
            )
        if tuple(obs[velocity_target_set].shape[1:]) != (3,):
            raise ValueError("Velocity supervision must have shape [B,3].")

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        storage = DeltaNetRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )
        return cls(
            actor,
            critic,
            storage,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **algorithm_cfg,
        )


__all__ = ["DeltaNetPPO"]
