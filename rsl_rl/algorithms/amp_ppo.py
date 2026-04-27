# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with Adversarial Motion Prior (AMP) support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import AMPDiscriminator, MLPModel
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import AMPLoader, resolve_callable, resolve_obs_groups


class AMPPPO(PPO):
    """PPO algorithm with Adversarial Motion Prior (AMP).

    Extends standard PPO with a discriminator that provides style rewards
    based on expert motion data.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # AMP specific parameters
        amp_cfg: dict | None = None,
    ) -> None:
        """Initialize AMPPPO with PPO and AMP components."""
        # Initialize base PPO
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # AMP components
        self.amp_cfg = amp_cfg
        if amp_cfg is not None:
            self._init_amp(amp_cfg)
        else:
            self.amp_discriminator = None
            self.amp_expert_data = None
            self.amp_storage = None
            self.amp_optimizer = None

        # Reward tracking for logging
        self.task_rewards = None
        self.style_rewards = None
        self.final_rewards = None

    def _init_amp(self, amp_cfg: dict) -> None:
        """Initialize AMP components."""
        # Initialize expert data loader
        self.amp_expert_data = AMPLoader(
            device=self.device,
            time_between_frames=amp_cfg.get("time_between_frames", 0.02),
            motion_files=amp_cfg["motion_files"],
            preload_transitions=amp_cfg.get("preload_transitions", True),
            num_preload_transitions=amp_cfg.get("num_preload_transitions", 100000),
        )

        # Initialize discriminator
        amp_obs_dim = self.amp_expert_data.amp_obs_dim
        self.amp_discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim * 2,  # state + next_state
            hidden_dims=amp_cfg.get("discriminator_hidden_dims", [256, 128]),
            activation=amp_cfg.get("discriminator_activation", "elu"),
            amp_reward_coef=amp_cfg.get("reward_coef", 1.0),
            normalize=amp_cfg.get("discriminator_normalize", False),
            task_reward_lerp=amp_cfg.get("task_reward_lerp", 0.0),
        ).to(self.device)

        # Initialize policy data storage for discriminator
        self.amp_storage = ReplayBuffer(
            obs_dim=amp_obs_dim,
            buffer_size=amp_cfg.get("amp_buffer_size", 100000),
            device=self.device,
        )

        # Initialize discriminator optimizer
        self.amp_optimizer = optim.Adam(
            self.amp_discriminator.parameters(),
            lr=amp_cfg.get("discriminator_learning_rate", 1e-4),
        )

        # Discriminator update settings
        self.disc_update_decimation = amp_cfg.get("discriminator_update_decimation", 1)
        self.disc_update_counter = 0

        print(f"[AMPPPO] AMP initialized with obs_dim={amp_obs_dim}")
        print(f"[AMPPPO] Discriminator: {self.amp_discriminator}")
        print(f"[AMPPPO] Expert motions: {self.amp_expert_data.num_motions}")

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Process environment step with AMP reward computation."""
        # Store original task reward before modification
        self.task_rewards = rewards.clone()

        # Compute AMP rewards if enabled
        if self.amp_discriminator is not None:
            # Get AMP observations from extras or obs
            if "amp_obs" in extras:
                amp_obs = extras["amp_obs"]
                amp_next_obs = extras["amp_next_obs"]
            elif "amp" in obs:
                amp_obs = obs["amp"]
                amp_next_obs = extras.get("amp_next_obs", amp_obs)  # Fallback
            else:
                # AMP not available in this step
                amp_obs = None
                amp_next_obs = None

            if amp_obs is not None and amp_next_obs is not None:
                # Compute AMP rewards
                self.final_rewards, self.style_rewards, _, _, _ = self.amp_discriminator.predict_amp_reward(
                    amp_obs, amp_next_obs, self.task_rewards
                )

                # Store policy transitions for discriminator training
                if dones.any():
                    not_done_mask = ~dones
                    if not_done_mask.any():
                        self.amp_storage.add_transitions(amp_obs[not_done_mask], amp_next_obs[not_done_mask])
                else:
                    self.amp_storage.add_transitions(amp_obs, amp_next_obs)

                # Use AMP-combined reward
                rewards = self.final_rewards
        else:
            self.style_rewards = None
            self.final_rewards = None

        # Call parent process_env_step with potentially modified rewards
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Run optimization with AMP discriminator updates."""
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        disc_actual_updates = 0

        # Get mini batch generators
        if self.actor.is_recurrent or self.critic.is_recurrent:
            ppo_generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            ppo_generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # Create AMP generators if enabled
        if self.amp_discriminator is not None:
            total_batches = self.num_learning_epochs * self.num_mini_batches
            batch_size = self.storage.num_transitions_per_env * self.storage.num_envs // self.num_mini_batches

            amp_policy_generator = self.amp_storage.mini_batch_generator(total_batches, batch_size)
            amp_expert_generator = self.amp_expert_data.feed_forward_generator(total_batches, batch_size)

            combined_generator = zip(ppo_generator, amp_policy_generator, amp_expert_generator)
        else:
            combined_generator = ((batch, None, None) for batch in ppo_generator)

        # Training loop
        for ppo_batch, amp_policy_data, amp_expert_data in combined_generator:
            # Standard PPO update (forward pass)
            self._update_ppo_batch(ppo_batch)

            # AMP discriminator update
            if self.amp_discriminator is not None and amp_policy_data is not None and amp_expert_data is not None:
                if self.disc_update_counter % self.disc_update_decimation == 0:
                    amp_loss, grad_pen_loss, policy_pred, expert_pred = self._update_discriminator(
                        amp_policy_data, amp_expert_data
                    )
                    mean_amp_loss += amp_loss
                    mean_grad_pen_loss += grad_pen_loss
                    mean_policy_pred += policy_pred
                    mean_expert_pred += expert_pred
                    disc_actual_updates += 1

                self.disc_update_counter += 1

        # Compute average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = self._compute_loss_dict(num_updates)

        # Add AMP losses if applicable
        if self.amp_discriminator is not None and disc_actual_updates > 0:
            loss_dict["amp"] = mean_amp_loss / disc_actual_updates
            loss_dict["amp_grad_pen"] = mean_grad_pen_loss / disc_actual_updates
            loss_dict["amp_policy_pred"] = mean_policy_pred / disc_actual_updates
            loss_dict["amp_expert_pred"] = mean_expert_pred / disc_actual_updates

        return loss_dict

    def _update_ppo_batch(self, batch) -> None:
        """Update PPO with a single batch."""
        # This is a placeholder - actual implementation would be in the main loop
        pass

    def _update_discriminator(
        self, policy_data: tuple[torch.Tensor, torch.Tensor], expert_data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[float, float, float, float]:
        """Update AMP discriminator.

        Args:
            policy_data: Tuple of (state, next_state) from policy rollouts.
            expert_data: Tuple of (state, next_state) from expert data.

        Returns:
            Tuple of losses (amp_loss, grad_pen_loss, policy_pred, expert_pred).
        """
        policy_state, policy_next_state = policy_data
        expert_state, expert_next_state = expert_data

        # Update normalizer if enabled
        if self.amp_discriminator.normalize:
            self.amp_discriminator.update_normalization(policy_state)
            self.amp_discriminator.update_normalization(expert_state)
            with torch.no_grad():
                policy_state = self.amp_discriminator.obs_normalizer(policy_state)
                policy_next_state = self.amp_discriminator.obs_normalizer(policy_next_state)
                expert_state = self.amp_discriminator.obs_normalizer(expert_state)
                expert_next_state = self.amp_discriminator.obs_normalizer(expert_next_state)

        # Compute discriminator outputs
        policy_input = torch.cat([policy_state, policy_next_state], dim=-1)
        expert_input = torch.cat([expert_state, expert_next_state], dim=-1)

        policy_d = self.amp_discriminator(policy_input)
        expert_d = self.amp_discriminator(expert_input)

        # Compute losses (policy should output -1, expert should output +1)
        expert_loss = nn.MSELoss()(expert_d, torch.ones_like(expert_d))
        policy_loss = nn.MSELoss()(policy_d, -torch.ones_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)

        # Gradient penalty on expert data
        grad_pen_loss = self.amp_discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10.0)

        # Total discriminator loss
        discriminator_loss = amp_loss + grad_pen_loss

        # Update discriminator
        self.amp_optimizer.zero_grad()
        discriminator_loss.backward()
        nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.max_grad_norm)
        self.amp_optimizer.step()

        return (
            amp_loss.item(),
            grad_pen_loss.item(),
            policy_d.mean().item(),
            expert_d.mean().item(),
        )

    def _compute_loss_dict(self, num_updates: int) -> dict[str, float]:
        """Compute loss dictionary from PPO updates."""
        # This would be populated during the training loop
        # For now return empty dict, actual values computed in main loop
        return {}

    def train_mode(self) -> None:
        """Set train mode for all models including discriminator."""
        super().train_mode()
        if self.amp_discriminator is not None:
            self.amp_discriminator.train()

    def eval_mode(self) -> None:
        """Set eval mode for all models including discriminator."""
        super().eval_mode()
        if self.amp_discriminator is not None:
            self.amp_discriminator.eval()

    def save(self) -> dict:
        """Save all models including AMP discriminator."""
        saved_dict = super().save()
        if self.amp_discriminator is not None:
            saved_dict["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
            saved_dict["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load all models including AMP discriminator."""
        result = super().load(loaded_dict, load_cfg, strict)
        if self.amp_discriminator is not None and "amp_discriminator_state_dict" in loaded_dict:
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Construct the AMPPPO algorithm."""
        # Resolve classes
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        if "amp_cfg" in cfg["algorithm"] and cfg["algorithm"]["amp_cfg"] is not None:
            default_sets.append("amp")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Initialize actor and critic
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

        # Initialize storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize algorithm
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        return alg
