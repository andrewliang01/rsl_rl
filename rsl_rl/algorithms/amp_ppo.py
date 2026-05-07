from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import AMPDiscriminator, MLPModel
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import AMPLoader, resolve_callable, resolve_obs_groups
from .multi_ppo import MultiPPO
from .ppo_factory import construct_ppo_algorithm


class AMPPPO(MultiPPO):
    """PPO with AMP support that is compatible with the local rsl_rl 5.x APIs."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        amp_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor=actor, critic=critic, storage=storage, **kwargs)

        self.amp_cfg = amp_cfg
        self.amp_discriminator: AMPDiscriminator | None = None
        self.amp_expert_data: AMPLoader | None = None
        self.amp_storage: ReplayBuffer | None = None
        self.amp_optimizer: optim.Optimizer | None = None
        self.disc_update_decimation = 1
        self.disc_update_counter = 0

        self.task_rewards: torch.Tensor | None = None
        self.style_rewards: torch.Tensor | None = None
        self.final_rewards: torch.Tensor | None = None
        self.amp_task_reward_lerp = amp_cfg.get("task_reward_lerp", 0.0) if amp_cfg is not None else 0.0
        self._amp_style_rewards_rollout: torch.Tensor | None = None
        if self.is_multi_critic:
            self._amp_style_rewards_rollout = torch.zeros(
                self.storage.num_transitions_per_env,
                self.storage.num_envs,
                device=self.device,
            )

        if amp_cfg is not None:
            self._init_amp(amp_cfg)

    def _init_amp(self, amp_cfg: dict) -> None:
        motion_files = amp_cfg.get("motion_files", [])
        self.amp_expert_data = AMPLoader(
            device=self.device,
            time_between_frames=amp_cfg.get("time_between_frames", 0.02),
            motion_files=motion_files,
            preload_transitions=amp_cfg.get("preload_transitions", True),
            num_preload_transitions=amp_cfg.get("num_preload_transitions", 100000),
        )

        amp_obs_dim = self.amp_expert_data.amp_obs_dim
        self.amp_discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim * 2,
            hidden_dims=amp_cfg.get("discriminator_hidden_dims", [256, 128]),
            activation=amp_cfg.get("discriminator_activation", "elu"),
            amp_reward_coef=amp_cfg.get("reward_coef", 1.0),
            normalize=amp_cfg.get("discriminator_normalize", False),
            task_reward_lerp=amp_cfg.get("task_reward_lerp", 0.0),
        ).to(self.device)

        self.amp_storage = ReplayBuffer(
            obs_dim=amp_obs_dim,
            buffer_size=amp_cfg.get("amp_buffer_size", 100000),
            device=self.device,
        )
        self.amp_optimizer = optim.Adam(
            self.amp_discriminator.parameters(),
            lr=amp_cfg.get("discriminator_learning_rate", 1e-4),
        )
        self.disc_update_decimation = amp_cfg.get("discriminator_update_decimation", 1)

    def _extract_amp_task_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.dim() == 1:
            return rewards
        if rewards.shape[-1] == 1:
            return rewards.squeeze(-1)
        return rewards.sum(dim=-1)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.task_rewards = self._extract_amp_task_rewards(rewards).clone()
        self.final_rewards = self.task_rewards.clone()
        self.style_rewards = torch.zeros_like(self.task_rewards)

        if self.amp_discriminator is not None and self.amp_storage is not None:
            current_amp_obs = None
            next_amp_obs = None
            has_terminal_safe_next_obs = "amp_next_obs" in extras

            if self.transition.observations is not None and "amp" in self.transition.observations.keys():
                current_amp_obs = self.transition.observations["amp"]
            if has_terminal_safe_next_obs:
                next_amp_obs = extras["amp_next_obs"]
            elif "amp" in obs.keys():
                next_amp_obs = obs["amp"]

            if current_amp_obs is not None and next_amp_obs is not None:
                done_mask = dones.to(dtype=torch.bool).view(-1)
                reward_mask = torch.ones_like(done_mask)
                # If the environment only exposes post-reset observations, terminal transitions would use
                # reset states as AMP next observations. Fall back to pure task reward on those transitions.
                if not has_terminal_safe_next_obs:
                    reward_mask = ~done_mask

                if reward_mask.any():
                    valid_final_rewards, valid_style_rewards, _, _, _ = self.amp_discriminator.predict_amp_reward(
                        current_amp_obs[reward_mask],
                        next_amp_obs[reward_mask],
                        self.task_rewards[reward_mask],
                    )
                    self.final_rewards[reward_mask] = valid_final_rewards
                    self.style_rewards[reward_mask] = valid_style_rewards

                valid_mask = ~done_mask
                if valid_mask.any():
                    self.amp_storage.insert(current_amp_obs[valid_mask], next_amp_obs[valid_mask])
                elif has_terminal_safe_next_obs:
                    # Keep behavior explicit: AMP replay buffer never stores reset transitions.
                    pass
                else:
                    # No valid AMP transitions this step.
                    pass

        if self.is_multi_critic and self._amp_style_rewards_rollout is not None and self.style_rewards is not None:
            self._amp_style_rewards_rollout[self.storage.step].copy_(self.style_rewards)
        elif self.final_rewards is not None:
            rewards = self.final_rewards

        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        super().compute_returns(obs)

        if not self.is_multi_critic or self._amp_style_rewards_rollout is None:
            return

        st = self.storage
        weighted_env_advantages = st.weighted_advantages.clone()
        style_advantages = torch.zeros(
            st.num_transitions_per_env,
            st.num_envs,
            device=self.device,
        )
        style_advantage = torch.zeros(st.num_envs, device=self.device)
        for step in reversed(range(st.num_transitions_per_env)):
            next_is_not_terminal = 1.0 - st.dones[step].float().squeeze(-1)
            style_advantage = (
                self._amp_style_rewards_rollout[step]
                + next_is_not_terminal * self.gamma * self.lam * style_advantage
            )
            style_advantages[step] = style_advantage

        if not self.normalize_advantage_per_mini_batch:
            style_advantages = (style_advantages - style_advantages.mean()) / (style_advantages.std() + 1e-8)

        st.weighted_advantages = (
            self.amp_task_reward_lerp * weighted_env_advantages
            + (1.0 - self.amp_task_reward_lerp) * style_advantages
        )
        if not self.normalize_advantage_per_mini_batch:
            st.weighted_advantages = (
                st.weighted_advantages - st.weighted_advantages.mean()
            ) / (st.weighted_advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = torch.zeros((), device=self.device)
        mean_surrogate_loss = torch.zeros((), device=self.device)
        mean_entropy = torch.zeros((), device=self.device)
        mean_rnd_loss = torch.zeros((), device=self.device) if self.rnd else None
        mean_symmetry_loss = torch.zeros((), device=self.device) if self.symmetry else None
        mean_amp_loss = torch.zeros((), device=self.device)
        mean_grad_pen_loss = torch.zeros((), device=self.device)
        mean_policy_pred = torch.zeros((), device=self.device)
        mean_expert_pred = torch.zeros((), device=self.device)
        disc_actual_updates = 0
        per_critic_value_losses = (
            [torch.zeros((), device=self.device) for _ in range(self.num_critics)] if self.is_multi_critic else None
        )

        if self.actor.is_recurrent or self._first_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.is_multi_critic:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs, use_weighted_advantages=True
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        if (
            self.amp_discriminator is not None
            and self.amp_storage is not None
            and self.amp_expert_data is not None
            and self.amp_storage.num_samples > 0
        ):
            total_batches = self.num_learning_epochs * self.num_mini_batches
            batch_size = self.storage.num_transitions_per_env * self.storage.num_envs // self.num_mini_batches
            amp_policy_generator = self.amp_storage.feed_forward_generator(total_batches, batch_size)
            amp_expert_generator = self.amp_expert_data.feed_forward_generator(total_batches, batch_size)
            combined_generator = zip(generator, amp_policy_generator, amp_expert_generator)
        else:
            combined_generator = ((batch, None, None) for batch in generator)

        for batch, amp_policy_data, amp_expert_data in combined_generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.is_multi_critic:
                batch_weighted_advantages = batch.advantages
                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        batch_weighted_advantages = (
                            batch_weighted_advantages - batch_weighted_advantages.mean()
                        ) / (batch_weighted_advantages.std() + 1e-8)
            else:
                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

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
                batch.returns = batch.returns.repeat(num_aug, 1)
                if self.is_multi_critic:
                    batch_weighted_advantages = batch_weighted_advantages.repeat(num_aug)
                else:
                    batch.advantages = batch.advantages.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.is_multi_critic:
                if self.shared_critic:
                    values = self._first_critic(
                        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
                    )
                else:
                    values_list = []
                    for critic in self.critics:
                        critic_values = critic(
                            batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
                        )
                        values_list.append(critic_values)
                    values = torch.cat(values_list, dim=-1)
            else:
                values = self._first_critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
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

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            if self.is_multi_critic:
                surrogate = -batch_weighted_advantages * ratio
                surrogate_clipped = -batch_weighted_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
            else:
                surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
                surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            if self.is_multi_critic and per_critic_value_losses is not None:
                with torch.no_grad():
                    for i in range(self.num_critics):
                        critic_loss = (values[:, i] - batch.returns[:, i]).pow(2).mean()
                        per_critic_value_losses[i] += critic_loss.detach()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

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

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad(set_to_none=True)
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            for critic in self.critics:
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if amp_policy_data is not None and amp_expert_data is not None:
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

            mean_value_loss += value_loss.detach()
            mean_surrogate_loss += surrogate_loss.detach()
            mean_entropy += entropy.detach().mean()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.detach()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss.item(),
            "surrogate": mean_surrogate_loss.item(),
            "entropy": mean_entropy.item(),
        }
        if self.is_multi_critic and per_critic_value_losses is not None:
            for i, name in enumerate(self.reward_group_names):
                loss_dict[f"value_{name}"] = (per_critic_value_losses[i] / num_updates).item()
        if self.rnd and mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
            loss_dict["rnd"] = mean_rnd_loss.item()
        if self.symmetry and mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
            loss_dict["symmetry"] = mean_symmetry_loss.item()
        if disc_actual_updates > 0:
            loss_dict["amp"] = (mean_amp_loss / disc_actual_updates).item()
            loss_dict["amp_grad_pen"] = (mean_grad_pen_loss / disc_actual_updates).item()
            loss_dict["amp_policy_pred"] = (mean_policy_pred / disc_actual_updates).item()
            loss_dict["amp_expert_pred"] = (mean_expert_pred / disc_actual_updates).item()

        return loss_dict

    def _update_discriminator(
        self, policy_data: tuple[torch.Tensor, torch.Tensor], expert_data: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.amp_discriminator is None or self.amp_optimizer is None:
            raise RuntimeError("AMP discriminator is not initialized.")

        policy_state, policy_next_state = policy_data
        expert_state, expert_next_state = expert_data

        if self.amp_discriminator.normalize:
            self.amp_discriminator.update_normalization(policy_state)
            self.amp_discriminator.update_normalization(expert_state)
            with torch.no_grad():
                policy_state = self.amp_discriminator.obs_normalizer(policy_state)
                policy_next_state = self.amp_discriminator.obs_normalizer(policy_next_state)
                expert_state = self.amp_discriminator.obs_normalizer(expert_state)
                expert_next_state = self.amp_discriminator.obs_normalizer(expert_next_state)

        policy_input = torch.cat([policy_state, policy_next_state], dim=-1)
        expert_input = torch.cat([expert_state, expert_next_state], dim=-1).requires_grad_(True)

        policy_d = self.amp_discriminator(policy_input)
        expert_d = self.amp_discriminator(expert_input)

        expert_loss = (expert_d - 1.0).pow(2).mean()
        policy_loss = (policy_d + 1.0).pow(2).mean()
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen_loss = self.amp_discriminator.compute_grad_pen_from_disc(expert_input, expert_d, lambda_=10.0)
        total_loss = amp_loss + grad_pen_loss

        self.amp_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.max_grad_norm)
        self.amp_optimizer.step()

        return (
            amp_loss.detach(),
            grad_pen_loss.detach(),
            policy_d.detach().mean(),
            expert_d.detach().mean(),
        )

    def train_mode(self) -> None:
        super().train_mode()
        if self.amp_discriminator is not None:
            self.amp_discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.amp_discriminator is not None:
            self.amp_discriminator.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        if self.amp_discriminator is not None and self.amp_optimizer is not None:
            saved_dict["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
            saved_dict["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if self.amp_discriminator is not None and "amp_discriminator_state_dict" in loaded_dict:
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
        if self.amp_optimizer is not None and "amp_optimizer_state_dict" in loaded_dict:
            self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMPPPO":
        return construct_ppo_algorithm(obs, env, cfg, device, variant="auto")
