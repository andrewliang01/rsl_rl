# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-Critic PPO Algorithm for rsl_rl 5.0.

This algorithm extends standard PPO to support multiple critics, where each critic
corresponds to a different reward group (e.g., task, regu, style, target).

The implementation is backward compatible - it can work with:
1. Single critic (num_critics=1): Falls back to standard PPO behavior
2. Multiple critics (num_critics>1): Uses multi-critic aggregation

Key features:
- Each critic has its own value network
- Each critic computes its own advantage
- Advantages are weighted and aggregated for policy updates
- Value losses are computed for all critics jointly
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config


class MultiPPO(PPO):
    """Multi-Critic Proximal Policy Optimization algorithm.

    Supports both single-critic (backward compatible) and multi-critic training.

    For multi-critic training:
    - Creates K independent critic networks
    - Each critic outputs a value for its reward group
    - Advantages are computed separately for each critic
    - Advantages are weighted and aggregated for policy updates
    - Value losses are computed for all critics jointly

    Example configuration:
        .. code-block:: python

            algorithm = {
                "class_name": "MultiPPO",
                "num_critics": 4,  # Number of reward groups
                "reward_group_weights": [2.5, 0.1, 1.0, 1.0],
                # ... other PPO params
            }
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel | nn.ModuleList,
        storage: RolloutStorage,
        num_critics: int = 1,
        reward_group_names: Optional[List[str]] = None,
        reward_group_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """Initialize Multi-Critic PPO.

        Args:
            actor: The actor (policy) network.
            critic: Single critic (MLPModel) or list of critics (ModuleList) for multi-critic.
            storage: Rollout storage for transitions.
            num_critics: Number of critics (reward groups). 1 for standard PPO.
            reward_group_names: Names of reward groups in reward tensor order.
                Used for validation and logging only.
            reward_group_weights: Weights for aggregating advantages from each critic.
                Length must match num_critics. If None, uses equal weights.
            **kwargs: Additional arguments passed to PPO base class.
        """
        # Store multi-critic parameters
        self.num_critics = num_critics

        if reward_group_names is None:
            reward_group_names = [f"critic_{i}" for i in range(num_critics)]
        if len(reward_group_names) != num_critics:
            raise ValueError(
                f"Length of reward_group_names ({len(reward_group_names)}) "
                f"must match num_critics ({num_critics})"
            )
        if len(set(reward_group_names)) != len(reward_group_names):
            raise ValueError(f"reward_group_names must be unique, got {reward_group_names}")
        self.reward_group_names = list(reward_group_names)

        # Initialize reward group weights
        if reward_group_weights is None:
            reward_group_weights = [1.0] * num_critics
        if len(reward_group_weights) != num_critics:
            raise ValueError(
                f"Length of reward_group_weights ({len(reward_group_weights)}) "
                f"must match num_critics ({num_critics})"
            )
        self.reward_group_weights = torch.tensor(
            reward_group_weights, dtype=torch.float32
        )

        # Handle critic(s)
        self.is_multi_critic = num_critics > 1

        if self.is_multi_critic:
            # Multi-critic mode: critic should be a ModuleList
            if not isinstance(critic, nn.ModuleList):
                raise ValueError(
                    f"For multi-critic mode, critic must be nn.ModuleList, "
                    f"got {type(critic)}"
                )
            if len(critic) != num_critics:
                raise ValueError(
                    f"Number of critics ({len(critic)}) must match "
                    f"num_critics ({num_critics})"
                )
            self.critics = critic
            # For compatibility with base PPO, set self.critic to the first critic
            # (used when calling parent methods)
            self._first_critic = critic[0]
        else:
            # Single critic mode: use standard PPO behavior
            self.critics = nn.ModuleList([critic])
            self._first_critic = critic

        # Initialize base PPO with first critic
        # This ensures backward compatibility
        super().__init__(actor, critic, storage, **kwargs)

        # Move reward weights to device
        self.reward_group_weights = self.reward_group_weights.to(self.device)

        # Re-create optimizer to include all critics (in multi-critic mode)
        if self.is_multi_critic:
            self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        """Create optimizer that includes actor and all critics."""
        params = list(self.actor.parameters())
        for critic in self.critics:
            params.extend(list(critic.parameters()))
        return resolve_optimizer(self._optimizer_name)(params, lr=self.learning_rate)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data.

        In multi-critic mode, computes values from all critics.
        """
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self._first_critic.get_hidden_state())

        # Compute the actions
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)

        # Compute values from all critics
        if self.is_multi_critic:
            values = []
            for critic in self.critics:
                value = critic(obs).detach()
                values.append(value)
            # Concatenate to [num_envs, num_critics]
            self.transition.values = torch.cat(values, dim=-1)
        else:
            # Single critic mode
            self.transition.values = self._first_critic(obs).detach()

        # Record observations
        self.transition.observations = obs

        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers.

        In multi-critic mode, expects rewards of shape [num_envs, num_critics].
        Single critic mode accepts rewards of shape [num_envs] or [num_envs, 1].
        """
        # Update the normalizers
        self.actor.update_normalization(obs)
        self._first_critic.update_normalization(obs)
        for critic in self.critics:
            critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Handle reward shape
        if self.is_multi_critic:
            # Multi-critic mode: rewards should be [num_envs, num_critics]
            if rewards.dim() == 1:
                # If 1D, expand to [num_envs, num_critics]
                rewards = rewards.unsqueeze(-1).expand(-1, self.num_critics)
            elif rewards.shape[-1] != self.num_critics:
                raise ValueError(
                    f"Expected rewards shape [..., {self.num_critics}], "
                    f"got {rewards.shape}"
                )
        else:
            # Single critic mode: standard shape [num_envs] or [num_envs, 1]
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(-1)

        # Record the rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards (only to first reward dimension for multi-critic)
            if self.is_multi_critic:
                self.transition.rewards[:, 0] += self.intrinsic_rewards.squeeze(-1)
            else:
                self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            timeouts = extras["time_outs"].unsqueeze(-1).to(self.device)
            if self.is_multi_critic:
                # Each critic gets its own bootstrap value
                self.transition.rewards += self.gamma * self.transition.values * timeouts
            else:
                self.transition.rewards += self.gamma * torch.squeeze(
                    self.transition.values * timeouts, 1
                )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        for critic in self.critics:
            critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions.

        For multi-critic: computes separate advantages for each critic,
        then aggregates them weighted by reward_group_weights.
        """
        st = self.storage

        # Compute value for the last step
        if self.is_multi_critic:
            last_values = []
            for critic in self.critics:
                last_values.append(critic(obs).detach())
            last_values = torch.cat(last_values, dim=-1)  # [num_envs, num_critics]
        else:
            last_values = self._first_critic(obs).detach()

        if self.is_multi_critic:
            # Multi-critic GAE computation
            advantage = torch.zeros(self.num_envs, self.num_critics, device=self.device)

            for step in reversed(range(st.num_transitions_per_env)):
                if step == st.num_transitions_per_env - 1:
                    next_values = last_values
                    next_is_not_terminal = 1.0 - st.dones[step].float().unsqueeze(-1)
                else:
                    next_values = st.values[step + 1]
                    next_is_not_terminal = 1.0 - st.dones[step].float().unsqueeze(-1)

                # TD error for each critic: r_t + gamma * V(s_{t+1}) - V(s_t)
                delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]

                # GAE for each critic
                advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage

                # Return for each critic
                st.returns[step] = advantage + st.values[step]

            # Compute and normalize advantages per critic
            st.advantages = st.returns - st.values

            # Normalize each critic's advantage separately
            for i in range(self.num_critics):
                mean = st.advantages[:, :, i].mean()
                std = st.advantages[:, :, i].std()
                st.advantages[:, :, i] = (st.advantages[:, :, i] - mean) / (std + 1e-8)

            # Store weighted advantages for policy update
            # [T, N, K] * [K] -> [T, N, K] -> sum -> [T, N]
            st.weighted_advantages = torch.sum(
                st.advantages * self.reward_group_weights.view(1, 1, -1), dim=-1
            )

            # Normalize weighted advantages
            if not self.normalize_advantage_per_mini_batch:
                st.weighted_advantages = (
                    st.weighted_advantages - st.weighted_advantages.mean()
                ) / (st.weighted_advantages.std() + 1e-8)
        else:
            # Single critic: use parent class logic
            super().compute_returns(obs)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses.

        For multi-critic: uses weighted advantages for policy loss,
        computes value loss for all critics jointly.
        """
        if not self.is_multi_critic:
            # Single critic: use parent class update
            return super().update()

        # Multi-critic update
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        # Track value loss per critic for logging
        per_critic_value_losses = [0.0] * self.num_critics

        # Get mini batch generator
        if self.actor.is_recurrent or self._first_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # Use weighted advantages for multi-critic
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs, use_weighted_advantages=True
            )

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # For multi-critic with use_weighted_advantages=True,
            # batch.advantages is already the weighted advantages [batch]
            batch_weighted_advantages = batch.advantages

            # Normalize advantages per mini-batch if requested
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch_weighted_advantages = (
                        batch_weighted_advantages - batch_weighted_advantages.mean()
                    ) / (batch_weighted_advantages.std() + 1e-8)

            # Symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch_weighted_advantages = batch_weighted_advantages.repeat(num_aug)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Actor forward
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Critic forward (all critics)
            values_list = []
            for critic in self.critics:
                values = critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                values_list.append(values)
            values = torch.cat(values_list, dim=-1)  # [batch, num_critics]

            # KL divergence and learning rate adaptation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss (using weighted advantages)
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -batch_weighted_advantages * ratio
            surrogate_clipped = -batch_weighted_advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss (all critics jointly)
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # Track per-critic value losses
            with torch.no_grad():
                for i in range(self.num_critics):
                    critic_loss = (values[:, i] - batch.returns[:, i]).pow(2).mean()
                    per_critic_value_losses[i] += critic_loss.item()

            # Total loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )

                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            for critic in self.critics:
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        # Clear storage
        self.storage.clear()

        # Build loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

        # Add per-critic value losses
        for i, name in enumerate(self.reward_group_names):
            loss_dict[f"value_{name}"] = per_critic_value_losses[i] / num_updates

        if self.rnd:
            mean_rnd_loss /= num_updates
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            mean_symmetry_loss /= num_updates
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def _get_batch_weighted_advantages(self, batch) -> torch.Tensor:
        """Get weighted advantages for a batch.

        This is a helper to extract the correct advantages from the storage.
        """
        # The batch already has advantages, but we need to use our weighted ones
        # For simplicity, we recompute from the batch indices
        # In practice, you might want to modify the storage to include weighted advantages
        return self._weighted_advantages.flatten()[:batch.advantages.shape[0]]

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.actor

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.is_multi_critic:
            # Save all critics
            for i, critic in enumerate(self.critics):
                saved_dict[f"critic_{i}_state_dict"] = critic.state_dict()
        else:
            saved_dict["critic_state_dict"] = self._first_critic.state_dict()

        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()

        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)

        if load_cfg.get("critic"):
            if self.is_multi_critic:
                # Load all critics
                for i in range(self.num_critics):
                    key = f"critic_{i}_state_dict"
                    if key in loaded_dict:
                        self.critics[i].load_state_dict(loaded_dict[key], strict=strict)
            else:
                self._first_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)

        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        return load_cfg.get("iteration", False)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str):
        """Construct the Multi-Critic PPO algorithm.

        This factory method creates the appropriate algorithm based on configuration:
        - If num_critics=1 or not specified: creates single critic (backward compatible)
        - If num_critics>1: creates multiple critics
        """
        # Extract multi-critic parameters
        alg_cfg = cfg.get("algorithm", {}).copy()
        alg_cfg.pop("class_name", None)
        share_cnn_encoders = alg_cfg.pop("share_cnn_encoders", False)
        num_critics = alg_cfg.pop("num_critics", 1)
        reward_group_names = alg_cfg.pop("reward_group_names", None)
        reward_group_weights = alg_cfg.pop("reward_group_weights", None)

        # Resolve class callables
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups"), default_sets)

        # Resolve RND and symmetry configs
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the actor
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")

        # Initialize critic(s)
        if num_critics > 1:
            # Multi-critic mode
            critics = nn.ModuleList()
            for i in range(num_critics):
                critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
                critics.append(critic)
            print(f"Created {num_critics} critics for multi-critic training")
            critic = critics
        else:
            # Single critic mode
            if share_cnn_encoders:
                cfg["critic"]["cnns"] = actor.cnns
            critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
            print(f"Critic Model: {critic}")

        # Initialize storage with num_critics
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device, num_critics=num_critics
        )

        # Initialize algorithm
        alg = MultiPPO(
            actor,
            critic,
            storage,
            num_critics=num_critics,
            reward_group_names=reward_group_names,
            reward_group_weights=reward_group_weights,
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )

        return alg
