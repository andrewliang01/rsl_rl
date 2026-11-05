"""G1/image adapters around the pinned JEPLO reference networks.

The image channels are two depth frames, never depth plus validity. Unknown
pixels have already been filled with normalized maximum range by the sensor.
Only the public policy group is read by the deployable state/encoder path.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from rsl_rl.third_party.jeplo.actor_critic_teacher_cts import ActorCriticTeacherCTS
from rsl_rl.third_party.jeplo.jepa_estimator_sensor_ss import JepaEstimatorSensorSS


class JEPLOEstimator(JepaEstimatorSensorSS):
    """Keep the original ViT/GRU/predictors; configure the angular image shape."""

    def __init__(self, image_height=40, image_width=96, **kwargs):
        if image_height < 1 or image_width < 1:
            raise ValueError("JEPLO image dimensions must be positive.")
        self.raw_depth_shape = (image_height, image_width)
        self.depth_shape = ((32, 64) if self.raw_depth_shape == (25, 60) else
                            (math.ceil(image_height / 4) * 4, math.ceil(image_width / 8) * 8))
        super().__init__(**kwargs)


def body_dimensions(num_actions: int) -> int:
    return 9 + 3 * num_actions


def scale_policy_body(policy: torch.Tensor, num_actions: int) -> torch.Tensor:
    """G1 term order -> original JEPLO body order and fixed physical scales.

    G1: angular, gravity, joint position, joint velocity, command, last action.
    JEPLO: angular*.25, gravity, command*(2,2,.25), position, velocity*.05,
    last action*.1. This never reads base linear velocity or simulator pose.
    """
    if policy.shape[-1] != body_dimensions(num_actions):
        raise ValueError("Expected G1 policy terms: angular/gravity/joints/command/action.")
    j = num_actions
    command_start = 6 + 2 * j
    command = policy[..., command_start:command_start + 3]
    command = command * command.new_tensor((2.0, 2.0, .25))
    return torch.cat((policy[..., :3] * .25, policy[..., 3:6], command,
                      policy[..., 6:6+j], policy[..., 6+j:6+2*j] * .05,
                      policy[..., command_start+3:] * .1), dim=-1)


def teacher_layout(policy_dim: int, height_dim: int, num_obs_hist=10):
    """Expose the same four observation sections expected by the original CTS."""
    hist_end = policy_dim * num_obs_hist
    height_end = hist_end + height_dim
    priv_end = height_end + 3  # existing privileged critic base-linear velocity
    return SimpleNamespace(
        num_obs_hist=num_obs_hist, obs_proprio_dim=policy_dim,
        obs_proprio_hist_dim=hist_end, obs_height_scan_dim=height_dim,
        obs_priv_dim=3, obs_critic_proprio_dim=policy_dim,
        obs_proprio_hist_range=(0, hist_end), obs_curr_proprio_range=(0, policy_dim),
        obs_height_scan_range=(hist_end, height_end), obs_priv_range=(height_end, priv_end),
        obs_critic_proprio_range=(priv_end, priv_end + policy_dim),
        observation_space=priv_end + policy_dim,
    )


class JEPLOState:
    """Bounded body history/GRU state; observation reads are idempotent.

    The control/capture/episode IDs are bookkeeping only and never reach a
    learnable layer. They prevent duplicate updates and cross-episode history.
    """

    def __init__(self, num_actions, history_frames=10):
        self.num_actions = num_actions
        self.history_frames = history_frames
        self.clear()

    def clear(self):
        self.body = self.hidden = self.features = None
        self.control = self.capture = self.episode = None

    def _initialize(self, obs):
        p = obs["policy"]
        n, dim = p.shape
        self.body = p.new_zeros(n, self.history_frames, dim)
        self.hidden = p.new_zeros(n, 512)
        self.features = p.new_zeros(n, 512)
        self.control = torch.full((n,), -1, dtype=torch.long, device=p.device)
        self.capture = self.control.clone()
        self.episode = obs["jeplo_episode_id"].reshape(-1).clone()

    @torch.inference_mode(False)
    @torch.no_grad()
    def observe(self, obs):
        if self.body is None:
            self._initialize(obs)
        if self.body.shape[0] != obs["policy"].shape[0]:
            raise ValueError("JEPLO batch changed; reset the policy state first.")
        episode = obs["jeplo_episode_id"].reshape(-1)
        control = obs["jeplo_control_id"].reshape(-1)
        reset = episode != self.episode
        fresh = (control != self.control) | reset
        self.body.masked_fill_(reset[:, None, None], 0)
        self.hidden.masked_fill_(reset[:, None], 0)
        self.features.masked_fill_(reset[:, None], 0)
        self.capture.masked_fill_(reset, -1)
        current = scale_policy_body(obs["policy"], self.num_actions)
        # The upstream Go2 environment puts the newest body observation first.
        shifted = torch.cat((current[:, None], self.body[:, :-1]), dim=1)
        self.body.copy_(torch.where(fresh[:, None, None], shifted, self.body))
        self.control.copy_(control)
        self.episode.copy_(episode)
        return self.body.flatten(1)

    @torch.inference_mode(False)
    @torch.no_grad()
    def advance(self, obs, estimator):
        capture = obs["jeplo_capture_id"].reshape(-1)
        fresh = (capture >= 0) & (capture != self.capture)
        ids = fresh.nonzero(as_tuple=True)[0]
        if ids.numel():
            features, hidden = estimator(self.body[ids].flatten(1), obs["jeplo_depth"][ids], self.hidden[ids])
            self.features[ids] = features
            self.hidden[ids] = hidden
            self.capture[ids] = capture[ids]
        return self.features

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self, dones=None):
        if dones is None:
            self.clear()
        elif self.body is not None:
            mask = dones.reshape(-1).bool()
            self.body[mask] = 0
            self.hidden[mask] = 0
            self.features[mask] = 0
            self.control[mask] = -1
            self.capture[mask] = -1


class JEPLOInferencePolicy(nn.Module):
    """Always deploy the original JEPLO student, irrespective of rollout roles."""

    is_recurrent = True

    def __init__(self, network: ActorCriticTeacherCTS, estimator, layout, num_actions):
        super().__init__()
        self.network = network
        self.estimator = estimator
        self.layout = layout
        self.num_actions = num_actions
        self.state = JEPLOState(num_actions, layout.num_obs_hist)

    @property
    def output_std(self):
        return self.network.std if hasattr(self.network, "std") else self.network.log_std.exp()

    def pack_training_observations(self, obs):
        """Only the training path packs height/linear velocity for teacher/critic."""
        j = self.num_actions
        critic = obs["critic"]
        if critic.shape[-1] != body_dimensions(j) + 3:
            raise ValueError("Expected the existing G1 privileged critic terms including base linear velocity.")
        lin_start = 6 + 2 * j
        clean_policy = torch.cat((critic[:, :lin_start], critic[:, lin_start+3:]), dim=-1)
        clean_body = scale_policy_body(clean_policy, j)
        return torch.cat((self.state.body.flatten(1), obs["height_scan_critic"].flatten(1),
                          critic[:, lin_start:lin_start+3], clean_body), dim=-1)

    @torch.inference_mode(False)
    @torch.no_grad()
    def forward(self, obs, *args, **kwargs):
        history = self.state.observe(obs)
        features = self.state.advance(obs, self.estimator)
        # Zeros occupy training-only sections. No privileged group is even read.
        packed = history.new_zeros(history.shape[0], self.layout.observation_space)
        packed[:, :history.shape[1]] = history
        return self.network.act_inference_student(packed, features)

    def reset_memory(self, dones=None):
        self.state.reset(dones)

    def reset(self, dones=None):
        self.reset_memory(dones)

    def as_jit(self):
        raise NotImplementedError("JEPLO needs explicit body-history/depth/GRU state I/O; play with --skip_export.")

    def as_onnx(self, **kwargs):
        raise NotImplementedError("Export JEPLO estimator and student with explicit recurrent I/O; use --skip_export for play.")
