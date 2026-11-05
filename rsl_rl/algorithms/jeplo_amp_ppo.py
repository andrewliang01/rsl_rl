"""Pinned JEPLO CTS + PE-JEPA behind the existing RSL-RL 5 runner contract.

No RangeJEPA reconstruction/mapping loss or modified perception trunk is used.
StairMaster AMP rewards reuse the local AMPPPO loader/discriminator/update code.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import time

import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.models.jeplo import (
    JEPLOEstimator, JEPLOInferencePolicy, body_dimensions, teacher_layout,
)
from rsl_rl.third_party.jeplo.actor_critic_teacher_cts import ActorCriticTeacherCTS
from rsl_rl.third_party.jeplo.rollout_storage_jepa_ss import RolloutStorageJepaSS
from rsl_rl.storage.jeplo_batches import StreamingCTSStorage
from rsl_rl.third_party.jeplo.teacher_cts import TeacherCTS


def synchronize_auxiliary_gradients(parameters):
    """All ranks reduce the same shapes, even for all-done prediction windows."""
    parameters = list(parameters)
    used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.long,
                        device=parameters[0].device)
    torch.distributed.all_reduce(used)
    gradients = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                           for p in parameters])
    torch.distributed.all_reduce(gradients)
    gradients /= torch.distributed.get_world_size()
    used_flags = used.cpu().tolist()
    offset = 0
    for index, p in enumerate(parameters):
        size = p.numel()
        if used_flags[index]:
            p.grad = gradients[offset:offset+size].view_as(p).clone()
        else:
            p.grad = None
        offset += size


class StairMasterAMP:
    """AMP-only adapter; calls the same implementations as existing AMPPPO."""

    _init_amp = AMPPPO._init_amp
    _sync_discriminator_learning_rate = AMPPPO._sync_discriminator_learning_rate
    _update_discriminator = AMPPPO._update_discriminator

    def __init__(self, cfg, device, cts):
        self.device, self.cts = device, cts
        self._gmq_compat = False
        self.amp_cfg = deepcopy(cfg)
        self.amp_discriminator = self.amp_optimizer = None
        self.amp_expert_data = self.amp_storage = None
        self.discriminator_follow_policy_lr = False
        self.disc_update_counter = 0
        self.max_grad_norm = cts.max_grad_norm
        self.task_rewards = self.style_rewards = self.final_rewards = None
        if cfg is not None:
            if cfg.get("loader_type") == "chitu_max_gmq":
                raise ValueError("JEPLO-G1 uses the StairMaster AMP loader, not GMQ compatibility mode.")
            self._init_amp(cfg)
        else:
            self.min_normalized_std = None

    @property
    def learning_rate(self):
        return self.cts.learning_rate

    def _resolve_min_normalized_std(self, value):
        floor = torch.as_tensor(value, device=self.device, dtype=torch.float32).flatten()
        if floor.numel() not in (1, self.cts.policy.num_actions) or bool((floor < 0).any()):
            raise ValueError("Invalid StairMaster action-standard-deviation floor.")
        return floor.expand(self.cts.policy.num_actions).clone()

    @torch.no_grad()
    def clamp_policy_std(self):
        if self.min_normalized_std is None:
            return
        if hasattr(self.cts.policy, "std"):
            self.cts.policy.std.clamp_(min=self.min_normalized_std)
        else:
            self.cts.policy.log_std.clamp_(min=self.min_normalized_std.clamp_min(1.e-6).log())

    @torch.no_grad()
    def mix(self, previous, obs, rewards, dones, extras):
        task = rewards.reshape(rewards.shape[0], -1).sum(-1)
        self.task_rewards = task.clone()
        self.final_rewards = task.clone()
        self.style_rewards = torch.zeros_like(task)
        if self.amp_discriminator is None:
            return self.final_rewards
        if previous is None or "amp" not in obs:
            raise ValueError("StairMaster AMP observations are required by JEPLO-G1.")
        terminal_safe = "amp_next_obs" in extras
        next_obs = extras["amp_next_obs"] if terminal_safe else obs["amp"]
        done = dones.reshape(-1).bool()
        reward_rows = torch.ones_like(done) if terminal_safe else ~done
        if reward_rows.any():
            final, style, *_ = self.amp_discriminator.predict_amp_reward(
                previous[reward_rows], next_obs[reward_rows], task[reward_rows])
            self.final_rewards[reward_rows] = final
            self.style_rewards[reward_rows] = style
        # Exactly like the G1 AMPPPO branch: never store reset transitions.
        if (~done).any():
            self.amp_storage.insert(previous[~done], next_obs[~done])
        return self.final_rewards

    def update(self, epochs, mini_batches, transitions):
        if self.amp_storage is None or self.amp_storage.num_samples == 0:
            return {}
        count = epochs * mini_batches
        size = transitions // mini_batches
        policy = self.amp_storage.feed_forward_generator(count, size)
        expert = self.amp_expert_data.feed_forward_generator(count, size)
        sums = torch.zeros(4, device=self.device)
        updates = 0
        for _ in range(count):
            if self.disc_update_counter % self.disc_update_decimation == 0:
                values = self._update_discriminator(next(policy), next(expert))
                sums += torch.stack(values)
                updates += 1
            self.disc_update_counter += 1
        out = dict(zip(("amp", "amp_grad_pen", "amp_policy_pred", "amp_expert_pred"),
                       (sums / max(updates, 1)).tolist()))
        out["amp_updates"] = updates
        out["amp_learning_rate"] = self.amp_optimizer.param_groups[0]["lr"]
        return out


class JEPLOAMPPPO:
    """A full JEPLO adapted baseline, constructed by the normal OnPolicyRunner."""

    checkpoint_schema = "jeplo-g1-adapted-v1"

    def __init__(self, obs, env, cfg, device):
        self.device = device
        self.cfg = deepcopy(cfg)
        alg_cfg = deepcopy(cfg["algorithm"])
        self.jeplo_cfg = options = alg_cfg.pop("jeplo_cfg")
        self.num_envs, self.num_actions = env.num_envs, env.num_actions
        self.num_steps = cfg["num_steps_per_env"]
        self.next_iteration = 0
        self.rnd = None
        self.is_multi_critic = False
        self.intrinsic_rewards = None
        if alg_cfg.get("num_critics", 1) != 1:
            raise ValueError("The reference JEPLO CTS baseline has one privileged critic.")
        for key in ("rnd_cfg", "symmetry_cfg"):
            if alg_cfg.get(key):
                raise ValueError(f"JEPLO reference does not implement {key}.")
        multi_gpu = cfg.get("multi_gpu")
        self.is_multi_gpu = multi_gpu is not None
        self.gpu_world_size = 1 if multi_gpu is None else multi_gpu["world_size"]
        layout = teacher_layout(body_dimensions(self.num_actions),
                                obs["height_scan_critic"].flatten(1).shape[1],
                                options.get("proprio_history_frames", 10))
        self.network = ActorCriticTeacherCTS(
            num_actions=self.num_actions, env_cfg=layout,
            student_proprio_dim=layout.obs_proprio_dim, jepa_feature_dim=512,
            actor_hidden_dims=cfg["actor"]["hidden_dims"],
            critic_hidden_dims=cfg["critic"]["hidden_dims"],
            init_noise_std=options.get("init_noise_std", 1.0),
        ).to(device)
        self.storage = StreamingCTSStorage(
            num_envs=self.num_envs, teacher_num_envs=self.num_envs,
            num_transitions_per_env=self.num_steps, obs_shape=[layout.observation_space],
            high_feature_shape=[512], actions_shape=[self.num_actions], device=device,
        )
        cts_fields = set(inspect.signature(TeacherCTS).parameters)
        cts_cfg = {key: value for key, value in alg_cfg.items() if key in cts_fields}
        cts_cfg.update(student_env_ratio=options.get("student_env_ratio", .25),
                       warmup_iters=tuple(options.get("warmup_iters", (5000, 10000))),
                       student_encoder_learning_rate=options.get("student_encoder_learning_rate", 2.e-4),
                       student_encoder_loss_coef=options.get("student_encoder_loss_coef", 1.0))
        self.cts = TeacherCTS(self.network, self.storage, self.num_envs,
                              device=device, multi_gpu_cfg=multi_gpu, **cts_cfg)
        self.optimizer = self.cts.optimizer
        self.estimator_stride = options.get("estimator_update_interval", 5)
        if self.estimator_stride < 1:
            raise ValueError("JEPLO estimator interval must be positive.")
        self.estimator = JEPLOEstimator(
            image_height=options.get("image_height", 40), image_width=options.get("image_width", 96),
            temporal_steps=layout.num_obs_hist, num_one_step_obs=layout.obs_proprio_dim,
            prop_hist_dim=layout.obs_proprio_hist_dim, num_depth_channels=2,
            action_hist_dim=self.num_actions * self.estimator_stride,
        ).to(device)
        if self.is_multi_gpu:
            self.estimator.gradient_sync = synchronize_auxiliary_gradients
        self.policy = JEPLOInferencePolicy(self.network, self.estimator, layout, self.num_actions).to(device)
        expected = (self.num_envs, 2, *self.estimator.raw_depth_shape)
        if tuple(obs["jeplo_depth"].shape) != expected:
            raise ValueError(f"JEPLO expected two depth frames {expected}, got {obs['jeplo_depth'].shape}.")
        sensor = getattr(getattr(env, "unwrapped", env), "jeplo_sensor_cfg", None)
        if sensor is not None:
            step_dt = getattr(getattr(env, "unwrapped", env), "step_dt")
            if abs(sensor.output_period_s - self.estimator_stride * step_dt) > 1.e-6:
                raise ValueError("JEPLO sensor output and estimator update clocks must agree.")
        self.estimator_storage = RolloutStorageJepaSS(
            num_envs=self.num_envs,
            num_transitions_per_env=self.num_steps // self.estimator_stride + 1,
            prop_hist_shape=(layout.obs_proprio_hist_dim,), depth_stack_shape=self.estimator.get_depth_shape(),
            action_hist_shape=(self.num_actions * self.estimator_stride,), gru_hidden_dim=512, device=device,
        )
        self.estimator_transition = RolloutStorageJepaSS.Transition()
        self.pending = False
        self.action_history = torch.zeros(self.num_envs, self.estimator_stride, self.num_actions, device=device)
        self.interval_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.previous_amp = self.last_actions = None
        self._last_estimator_control = -1
        self.amp = StairMasterAMP(alg_cfg.get("amp_cfg"), device, self.cts)
        self.amp.clamp_policy_std()
        self.profile_timing = options.get("profile_timing", False)
        print(f"[JEPLO] pinned=460e227, raw_depth={expected[1:]}, no validity channel; "
              f"JEPA epochs={options.get('jepa_epochs', 5)}, sampled envs/epoch<=256; "
              f"network params={sum(p.numel() for p in self.network.parameters())}, "
              f"estimator params={sum(p.numel() for p in self.estimator.parameters())}")

    @property
    def learning_rate(self):
        return self.cts.learning_rate

    @property
    def amp_discriminator(self):
        return self.amp.amp_discriminator

    @property
    def task_rewards(self):
        return self.amp.task_rewards

    @property
    def final_rewards(self):
        return self.amp.final_rewards

    @property
    def style_rewards(self):
        return self.amp.style_rewards

    @torch.inference_mode(False)
    @torch.no_grad()
    def _finish_estimator_transition(self):
        if self.pending:
            self.estimator_transition.action_hist = self.action_history.flatten(1)
            self.estimator_transition.dones = self.interval_dones
            self.estimator_storage.add_transitions(self.estimator_transition)
            self.estimator_transition.clear()
            self.pending = False
        self.action_history.zero_()
        self.interval_dones.zero_()

    @torch.inference_mode(False)
    @torch.no_grad()
    def act(self, obs):
        if self.storage.step == 0:
            self.cts.set_iteration(self.next_iteration)
        state = self.policy.state
        state.observe(obs)
        control = int(obs["jeplo_control_id"][0, 0])
        if control % self.estimator_stride == 0 and control != self._last_estimator_control:
            self._finish_estimator_transition()
            self.estimator_transition.proprio_hist = state.body.flatten(1).clone()
            self.estimator_transition.depth_stack = obs["jeplo_depth"].clone()
            self.estimator_transition.hidden_states = state.hidden.clone()
            state.advance(obs, self.estimator)
            self.pending = True
            self._last_estimator_control = control
        packed = self.policy.pack_training_observations(obs)
        actions = self.cts.act(packed, state.features)
        self.last_actions = actions.clone()
        self.previous_amp = obs["amp"].clone() if "amp" in obs else None
        return actions

    @torch.inference_mode(False)
    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras):
        if self.last_actions is None:
            raise RuntimeError("JEPLO process_env_step requires a preceding act().")
        mixed = self.amp.mix(self.previous_amp, obs, rewards, dones, extras)
        done = dones.reshape(-1).bool()
        self.action_history.copy_(torch.cat((self.action_history[:, 1:], self.last_actions[:, None]), dim=1))
        self.interval_dones.logical_or_(done)
        self.action_history[done] = 0
        self.policy.state.observe(obs)
        self.cts.process_env_step(mixed, dones, extras)
        self.last_actions = None

    @torch.inference_mode(False)
    @torch.no_grad()
    def compute_returns(self, obs):
        self._finish_estimator_transition()
        self.policy.state.observe(obs)
        self.cts.compute_returns(self.policy.pack_training_observations(obs), self.policy.state.features)

    def _clock(self):
        if self.profile_timing and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def update(self):
        start = self._clock()
        out = self.cts.update()
        cts_end = self._clock()
        self.amp.clamp_policy_std()
        metrics = {}
        updates = 0
        if self.estimator_storage.step > 1:
            for batch in self.estimator_storage.mini_batch_generator(
                    self.jeplo_cfg.get("jepa_mini_batches", 2), self.jeplo_cfg.get("jepa_epochs", 5)):
                losses = self.estimator.update(batch)
                for key, value in losses.items():
                    metrics[key] = metrics.get(key, 0.0) + value
                updates += 1
        out.update({f"jepa_{key}": value / max(updates, 1) for key, value in metrics.items()})
        out["jepa_updates"] = updates
        out["jepa_sequence_steps"] = self.estimator_storage.step
        self.estimator_storage.clear()
        jepa_end = self._clock()
        out.update(self.amp.update(self.jeplo_cfg.get("amp_update_epochs", self.cts.num_learning_epochs),
                                  self.jeplo_cfg.get("amp_update_mini_batches", self.cts.num_mini_batches),
                                  self.num_envs * self.num_steps))
        amp_end = self._clock()
        if self.profile_timing:
            out.update(time_cts=cts_end-start, time_jepa=jepa_end-cts_end, time_amp=amp_end-jepa_end)
            if str(self.device).startswith("cuda"):
                out.update(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 1024**3,
                           cuda_reserved_gib=torch.cuda.memory_reserved(self.device) / 1024**3)
        self.next_iteration += 1
        return out

    def get_policy(self):
        return self.policy

    def train_mode(self):
        self.policy.train()
        if self.amp_discriminator is not None:
            self.amp_discriminator.train()

    def eval_mode(self):
        self.policy.eval()
        if self.amp_discriminator is not None:
            self.amp_discriminator.eval()

    def broadcast_parameters(self):
        if not self.is_multi_gpu:
            return
        self.cts.broadcast_parameters()
        for tensor in self.estimator.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)
        if self.amp_discriminator is not None:
            for tensor in self.amp_discriminator.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def save(self):
        result = {
            "jeplo_schema": self.checkpoint_schema,
            "jeplo_configuration": deepcopy(self.jeplo_cfg),
            "model_state_dict": self.network.state_dict(),
            "estimator_state_dict": self.estimator.state_dict(),
            "optimizer_state_dict": self.cts.optimizer.state_dict(),
            "student_encoder_optimizer_state_dict": self.cts.optimizer_student_encoder.state_dict(),
            "estimator_optimizer_state_dict": self.estimator.optimizer.state_dict(),
            "jeplo_next_iteration": self.next_iteration,
            "jeplo_learning_rate": self.learning_rate,
            "amp_discriminator_update_counter": self.amp.disc_update_counter,
        }
        if self.amp_discriminator is not None:
            result["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
            result["amp_optimizer_state_dict"] = self.amp.amp_optimizer.state_dict()
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("jeplo_schema") != self.checkpoint_schema:
            raise ValueError("Expected a JEPLO-G1-adapted checkpoint; Go2/RangeJEPA checkpoints are different.")
        self.network.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
        self.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=strict)
        restore_optimizers = load_cfg is None or load_cfg.get("optimizer", True)
        if restore_optimizers:
            self.cts.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.cts.optimizer_student_encoder.load_state_dict(loaded_dict["student_encoder_optimizer_state_dict"])
            self.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        self.cts.learning_rate = float(loaded_dict["jeplo_learning_rate"])
        self.next_iteration = int(loaded_dict["jeplo_next_iteration"])
        self.amp.disc_update_counter = int(loaded_dict.get("amp_discriminator_update_counter", 0))
        if self.amp_discriminator is not None:
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            if restore_optimizers:
                self.amp.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
        self.policy.reset()
        # The common runner's legacy label is the just-finished iteration.
        # Resume this opt-in algorithm at its saved NEXT iteration, like JEPLO.
        loaded_dict["iter"] = self.next_iteration
        self.cts.set_iteration(self.next_iteration)
        return True

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        return JEPLOAMPPPO(obs, env, cfg, device)
