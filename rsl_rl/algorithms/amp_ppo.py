# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import string_to_callable


class AMPPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        discriminator,
        # --------- AMP components START ---------
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
        amp_loss_scale=1.0,
        amp_grad_pen_scale=10.0,
        amp_discriminator_learning_rate=None,  # 判别器独立学习率
        amp_discriminator_update_freq=1,  # 判别器更新频率
        # --------- AMP components END---------
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # --------- AMP 判别器 ---------
        self.amploss_coef = amp_loss_scale  #1.0
        self.amp_grad_pen_scale = amp_grad_pen_scale  #10.0
        self.min_std = min_std
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # 判别器更新频率控制
        self.amp_discriminator_update_freq = amp_discriminator_update_freq
        self.amp_update_counter = 0  # 用于跟踪更新次数

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        # --------- 分离优化器：策略使用独立的优化器 ---------
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # --------- 判别器使用独立的优化器和学习率 ---------
        # 如果没有指定判别器学习率，则使用策略学习率
        discriminator_lr = amp_discriminator_learning_rate if amp_discriminator_learning_rate is not None else learning_rate
        discriminator_params = [
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2},
        ]
        self.discriminator_optimizer = optim.Adam(discriminator_params, lr=discriminator_lr)
        self.discriminator_learning_rate = discriminator_lr

        # 打印学习率信息
        print(f"[AMP PPO] Policy learning rate: {learning_rate}")
        print(f"[AMP PPO] Discriminator learning rate: {discriminator_lr}")
        print(f"[AMP PPO] Discriminator update frequency: 1/{amp_discriminator_update_freq}")

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    # --------- 多加入了amp的观测值amp_obs ---------
    def act(self, obs, critic_obs, amp_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        # --------- amp ---------
        self.amp_transition.observations = amp_obs
        # --------- amp ---------
        return self.transition.actions

    # --------- 多加入了amp的观测值 ---------
    def process_env_step(self, rewards, dones, infos, amp_obs):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # --------- amp ---------
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # --------- amp ---------

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # --------- amp ---------
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        # --------- amp ---------

        # 从环境采样的数据 + AMP 策略样本 + AMP 专家样本 同时取出一批
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # --------- 策略损失（PPO）：不包含判别器损失 ---------
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    ppo_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)
            else:
                rnd_loss = None

            # ========== 第一步：更新策略网络（Actor-Critic） ==========
            self.optimizer.zero_grad()
            ppo_loss.backward()

            # -- For RND
            if self.rnd and rnd_loss is not None:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for policy
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # ========== 第二步：独立更新判别器（根据更新频率） ==========
            # 判别器更新频率控制：只在特定的迭代中更新判别器
            self.amp_update_counter += 1
            should_update_discriminator = (self.amp_update_counter % self.amp_discriminator_update_freq == 0)

            if should_update_discriminator:
                # Discriminator loss
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert

                # ===== 🔍 诊断：打印数据顺序验证信息 =====
                if not hasattr(self, '_amp_order_diagnosed'):
                    self._amp_order_diagnosed = True
                    print("\n" + "="*80)
                    print("🔍 AMP 数据顺序诊断 (仅打印一次)")
                    print("="*80)

                    # 打印原始数据（归一化前）
                    print("\n【归一化前的原始数据】")
                    print(f"policy_state shape: {policy_state.shape}")
                    print(f"expert_state shape: {expert_state.shape}")

                    # 数据结构说明
                    print(f"\n数据结构: [0:29]关节位置 + [29:58]关节速度 + [58:70]末端位置")
                    print(f"末端位置: [58:61]末端1 + [61:64]末端2 + [64:67]末端3 + [67:70]末端4")
                    print(f"预期顺序: 左手 → 右手 → 左脚 → 右脚")

                    # 打印第一个样本的末端位置
                    print(f"\n【Policy 末端位置 (第1个样本)】")
                    print(f"  [58:61] 末端1: [{policy_state[0,58]:.4f}, {policy_state[0,59]:.4f}, {policy_state[0,60]:.4f}]")
                    print(f"  [61:64] 末端2: [{policy_state[0,61]:.4f}, {policy_state[0,62]:.4f}, {policy_state[0,63]:.4f}]")
                    print(f"  [64:67] 末端3: [{policy_state[0,64]:.4f}, {policy_state[0,65]:.4f}, {policy_state[0,66]:.4f}]")
                    print(f"  [67:70] 末端4: [{policy_state[0,67]:.4f}, {policy_state[0,68]:.4f}, {policy_state[0,69]:.4f}]")

                    print(f"\n【Expert 末端位置 (第1个样本)】")
                    print(f"  [58:61] 末端1: [{expert_state[0,58]:.4f}, {expert_state[0,59]:.4f}, {expert_state[0,60]:.4f}]")
                    print(f"  [61:64] 末端2: [{expert_state[0,61]:.4f}, {expert_state[0,62]:.4f}, {expert_state[0,63]:.4f}]")
                    print(f"  [64:67] 末端3: [{expert_state[0,64]:.4f}, {expert_state[0,65]:.4f}, {expert_state[0,66]:.4f}]")
                    print(f"  [67:70] 末端4: [{expert_state[0,67]:.4f}, {expert_state[0,68]:.4f}, {expert_state[0,69]:.4f}]")

                    # ===== 新增：统计分析 =====
                    print(f"\n【统计分析 - 检查所有样本的符号分布】")

                    # 统计 Policy 的末端Y坐标符号
                    p_y1_signs = (policy_state[:, 59] > 0).float()
                    p_y2_signs = (policy_state[:, 62] < 0).float()
                    p_y3_signs = (policy_state[:, 65] > 0).float()
                    p_y4_signs = (policy_state[:, 68] < 0).float()

                    # 统计 Expert 的末端Y坐标符号
                    e_y1_signs = (expert_state[:, 59] > 0).float()
                    e_y2_signs = (expert_state[:, 62] < 0).float()
                    e_y3_signs = (expert_state[:, 65] > 0).float()
                    e_y4_signs = (expert_state[:, 68] < 0).float()

                    total_samples = policy_state.shape[0]

                    print(f"总样本数: {total_samples}")
                    print(f"\nPolicy 符号正确率 (期望: 左正右负):")
                    print(f"  末端1 (左手) Y>0: {p_y1_signs.mean()*100:.1f}%  ✅" if p_y1_signs.mean() > 0.8 else f"  末端1 (左手) Y>0: {p_y1_signs.mean()*100:.1f}%  ⚠️")
                    print(f"  末端2 (右手) Y<0: {p_y2_signs.mean()*100:.1f}%  ✅" if p_y2_signs.mean() > 0.8 else f"  末端2 (右手) Y<0: {p_y2_signs.mean()*100:.1f}%  ⚠️")
                    print(f"  末端3 (左脚) Y>0: {p_y3_signs.mean()*100:.1f}%  ✅" if p_y3_signs.mean() > 0.8 else f"  末端3 (左脚) Y>0: {p_y3_signs.mean()*100:.1f}%  ⚠️")
                    print(f"  末端4 (右脚) Y<0: {p_y4_signs.mean()*100:.1f}%  ✅" if p_y4_signs.mean() > 0.8 else f"  末端4 (右脚) Y<0: {p_y4_signs.mean()*100:.1f}%  ⚠️")

                    print(f"\nExpert 符号正确率 (期望: 左正右负):")
                    print(f"  末端1 (左手) Y>0: {e_y1_signs.mean()*100:.1f}%  ✅" if e_y1_signs.mean() > 0.8 else f"  末端1 (左手) Y>0: {e_y1_signs.mean()*100:.1f}%  ⚠️")
                    print(f"  末端2 (右手) Y<0: {e_y2_signs.mean()*100:.1f}%  ✅" if e_y2_signs.mean() > 0.8 else f"  末端2 (右手) Y<0: {e_y2_signs.mean()*100:.1f}%  ⚠️")
                    print(f"  末端3 (左脚) Y>0: {e_y3_signs.mean()*100:.1f}%  ✅" if e_y3_signs.mean() > 0.8 else f"  末端3 (左脚) Y>0: {e_y3_signs.mean()*100:.1f}%  ⚠️")
                    print(f"  末端4 (右脚) Y<0: {e_y4_signs.mean()*100:.1f}%  ✅" if e_y4_signs.mean() > 0.8 else f"  末端4 (右脚) Y<0: {e_y4_signs.mean()*100:.1f}%  ⚠️")

                    # 判断是否是数据顺序问题
                    print(f"\n【诊断结论】")
                    if e_y1_signs.mean() > 0.9 and e_y2_signs.mean() > 0.9 and e_y3_signs.mean() > 0.9 and e_y4_signs.mean() > 0.9:
                        print(f"✅ Expert 数据顺序正确 (所有末端符号正确率 >90%)")
                    else:
                        print(f"❌ Expert 数据顺序可能有问题！")

                    if p_y4_signs.mean() < 0.5:
                        print(f"⚠️  Policy 右脚符号正确率仅 {p_y4_signs.mean()*100:.1f}%")
                        print(f"   这是训练初期的正常现象，策略还在学习中")
                        print(f"   判别器仍然能学习，因为其他特征是正确的")
                    elif p_y4_signs.mean() < 0.8:
                        print(f"⚠️  Policy 右脚符号正确率 {p_y4_signs.mean()*100:.1f}% (略低)")
                        print(f"   建议增加训练时间，让策略继续学习")
                    else:
                        print(f"✅ Policy 数据质量良好 (所有末端符号正确率 >80%)")

                    # ===== 统计分析结束 =====

                    # Y坐标符号检查（右手坐标系：左正右负）
                    print(f"\n【第1个样本的Y坐标符号检查】")
                    print(f"Policy: 末端1_Y={policy_state[0,59]:+.4f}, 末端2_Y={policy_state[0,62]:+.4f}, "
                          f"末端3_Y={policy_state[0,65]:+.4f}, 末端4_Y={policy_state[0,68]:+.4f}")
                    print(f"Expert: 末端1_Y={expert_state[0,59]:+.4f}, 末端2_Y={expert_state[0,62]:+.4f}, "
                          f"末端3_Y={expert_state[0,65]:+.4f}, 末端4_Y={expert_state[0,68]:+.4f}")

                    # 自动判断（保留原有逻辑，但简化输出）
                    p_y = [policy_state[0,59].item(), policy_state[0,62].item(),
                           policy_state[0,65].item(), policy_state[0,68].item()]
                    e_y = [expert_state[0,59].item(), expert_state[0,62].item(),
                           expert_state[0,65].item(), expert_state[0,68].item()]

                    print(f"\n【第1个样本的自动判断】")
                    if abs(p_y[0]) > 0.05 and abs(p_y[1]) > 0.05:
                        if p_y[0] > 0 and p_y[1] < 0:
                            print(f"✅ Policy 末端1/2: 左正 右负 → 左手-右手")
                        elif p_y[0] < 0 and p_y[1] > 0:
                            print(f"❌ Policy 末端1/2: 左负 右正 → 数据顺序反了!")

                    if abs(e_y[0]) > 0.05 and abs(e_y[1]) > 0.05:
                        if e_y[0] > 0 and e_y[1] < 0:
                            print(f"✅ Expert 末端1/2: 左正 右负 → 左手-右手")
                        elif e_y[0] < 0 and e_y[1] > 0:
                            print(f"❌ Expert 末端1/2: 左负 右正 → 数据顺序反了!")

                    if abs(e_y[2]) > 0.05 and abs(e_y[3]) > 0.05:
                        if e_y[2] > 0 and e_y[3] < 0:
                            print(f"✅ Expert 末端3/4: 左正 右负 → 左脚-右脚")
                        elif e_y[2] < 0 and e_y[3] > 0:
                            print(f"❌ Expert 末端3/4: 左负 右正 → 数据顺序反了!")

                    print("="*80 + "\n")
                # ===== 诊断代码结束 =====

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

                # ===== 🔍 诊断：打印归一化后的数据 =====
                if hasattr(self, '_amp_order_diagnosed') and not hasattr(self, '_amp_normalized_diagnosed'):
                    self._amp_normalized_diagnosed = True
                    if self.amp_normalizer is not None:
                        print("\n【归一化后的数据 (第1个样本末端位置)】")
                        print(f"Policy: 末端1_Y={policy_state[0,59]:+.4f}, 末端2_Y={policy_state[0,62]:+.4f}, "
                              f"末端3_Y={policy_state[0,65]:+.4f}, 末端4_Y={policy_state[0,68]:+.4f}")
                        print(f"Expert: 末端1_Y={expert_state[0,59]:+.4f}, 末端2_Y={expert_state[0,62]:+.4f}, "
                              f"末端3_Y={expert_state[0,65]:+.4f}, 末端4_Y={expert_state[0,68]:+.4f}")
                        print("注意: 归一化后数值范围会改变，但相对关系应保持\n")

                        # ===== 新增：检查归一化器的统计量 =====
                        print("="*80)
                        print("⚠️  归一化器诊断")
                        print("="*80)

                        # 获取归一化器的统计量
                        if hasattr(self.amp_normalizer, 'mean') and hasattr(self.amp_normalizer, 'std'):
                            import numpy as np
                            mean = self.amp_normalizer.mean
                            std = self.amp_normalizer.std

                            print(f"\n【归一化器统计量 (末端位置部分)】")
                            print(f"末端位置维度: [58:70] (12维)")
                            print(f"\nMean (均值):")
                            for i in range(58, 70):
                                ee_idx = (i - 58) // 3
                                coord = ['X', 'Y', 'Z'][(i - 58) % 3]
                                print(f"  [{i}] 末端{ee_idx+1}_{coord}: mean={mean[i]:+.4f}, std={std[i]:.4f}")

                            # 检查末端Y坐标的均值
                            ee_y_means = [mean[59], mean[62], mean[65], mean[68]]
                            ee_y_stds = [std[59], std[62], std[65], std[68]]

                            print(f"\n【末端Y坐标的统计量】")
                            print(f"末端1_Y: mean={ee_y_means[0]:+.6f}, std={ee_y_stds[0]:.6f}")
                            print(f"末端2_Y: mean={ee_y_means[1]:+.6f}, std={ee_y_stds[1]:.6f}")
                            print(f"末端3_Y: mean={ee_y_means[2]:+.6f}, std={ee_y_stds[2]:.6f}")
                            print(f"末端4_Y: mean={ee_y_means[3]:+.6f}, std={ee_y_stds[3]:.6f}")

                            # 自动诊断
                            print(f"\n【自动诊断】")

                            # 检查1: 左右脚的均值是否对称
                            if abs(ee_y_means[0] + ee_y_means[1]) < 0.01:
                                print(f"✅ 左右手Y均值接近对称: {ee_y_means[0]:+.4f} vs {ee_y_means[1]:+.4f}")
                            else:
                                print(f"⚠️  左右手Y均值不对称: {ee_y_means[0]:+.4f} vs {ee_y_means[1]:+.4f}")

                            if abs(ee_y_means[2] + ee_y_means[3]) < 0.01:
                                print(f"✅ 左右脚Y均值接近对称: {ee_y_means[2]:+.4f} vs {ee_y_means[3]:+.4f}")
                            else:
                                print(f"⚠️  左右脚Y均值不对称: {ee_y_means[2]:+.4f} vs {ee_y_means[3]:+.4f}")

                            # 检查2: 均值是否接近0
                            if all(abs(m) < 0.05 for m in ee_y_means):
                                print(f"✅ 所有末端Y均值接近0，归一化器统计量正常")
                            else:
                                print(f"⚠️  某些末端Y均值偏离0，这可能导致符号翻转问题")
                                print(f"   建议: 使用更多样化的数据初始化归一化器")

                            # 检查3: 标准差是否合理
                            if all(0.05 < s < 0.5 for s in ee_y_stds):
                                print(f"✅ 所有末端Y标准差在合理范围内 (0.05-0.5)")
                            else:
                                print(f"⚠️  某些末端Y标准差异常:")
                                for i, s in enumerate(ee_y_stds):
                                    if s < 0.05:
                                        print(f"   末端{i+1}_Y std={s:.4f} 过小，数据可能缺乏多样性")
                                    elif s > 0.5:
                                        print(f"   末端{i+1}_Y std={s:.4f} 过大，数据可能有异常值")

                            print(f"\n【问题根源分析】")
                            print(f"从诊断结果看，Policy 末端4 的原始Y坐标为 +0.0253")
                            print(f"这说明在训练初期，某些环境的机器人姿态异常（右脚偏向左侧）")
                            print(f"这是正常现象，因为策略还在学习中")
                            print(f"\n但更严重的是，归一化后 Expert 的符号发生了变化！")
                            print(f"这说明归一化器的统计量可能是用 Policy 的混乱数据初始化的")

                            print(f"\n【解决方案】")
                            print(f"方案1 (推荐): 用专家数据初始化归一化器")
                            print(f"   - 在训练开始前，用专家数据计算 mean 和 std")
                            print(f"   - 这样可以确保归一化器基于正确的分布")
                            print(f"\n方案2: 禁用归一化")
                            print(f"   - 将 amp_normalizer 设置为 None")
                            print(f"   - 简单但可能影响判别器性能")
                            print(f"\n方案3: 使用 running mean/std")
                            print(f"   - 让归一化器在训练过程中动态更新统计量")
                            print(f"   - 但初期可能不稳定")

                        print("="*80 + "\n")
                # ===== 诊断代码结束 =====

                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=self.amp_grad_pen_scale)

                # 判别器的总损失
                discriminator_loss = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

                # 更新判别器
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.discriminator_optimizer.step()


                # Store the losses
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
            else:
                # 如果不更新判别器，仍然需要计算判别器输出用于记录（可选）
                with torch.no_grad():
                    policy_state, policy_next_state = sample_amp_policy
                    expert_state, expert_next_state = sample_amp_expert
                    if self.amp_normalizer is not None:
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

                    policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                    expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                    expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                    policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                    amp_loss = 0.5 * (expert_loss + policy_loss)
                    grad_pen_loss = torch.tensor(0.0, device=self.device)  # 不计算梯度惩罚

                    # Store the losses for logging
                    mean_amp_loss += amp_loss.item()
                    mean_grad_pen_loss += grad_pen_loss.item()
                    mean_policy_pred += policy_d.mean().item()
                    mean_expert_pred += expert_d.mean().item()

            # Store other losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
