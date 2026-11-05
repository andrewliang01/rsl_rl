"""Task gradients through the existing one-pass range auxiliary replay.

Ordinary AMP-PPO continues to use detached rollout features. Only a fresh
capture's first control action contributes a task loss in auxiliary replay.
This adds no deployable parameters, extra image pass, or policy teacher.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy
import math

import torch

from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO
from rsl_rl.models.range_jepa import RangeJEPA, RangeJEPAActor
from rsl_rl.modules.distribution import GaussianDistribution


@dataclass(frozen=True)
class TaskRolloutSnapshot:
    """Owned scalar rollout tensors; no raw images or cached visual features."""
    proprioception: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor


def validate_task_bridge_config(config):
    config = {} if config is None else deepcopy(config)
    if not isinstance(config, dict) or set(config) - {"version", "coefficient"}:
        raise ValueError("task_bridge_cfg accepts only version and coefficient.")
    if type(config.get("version", 1)) is not int or config.get("version", 1) != 1:
        raise ValueError("Unsupported task bridge version.")
    value = config.get("coefficient", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Task bridge coefficient must be finite and nonnegative.")
    return {"version": 1, "coefficient": float(value)}


@contextmanager
def frozen_actor(actor):
    """Freeze the same PPO actor, preserving its parameters, grads and modes.

    We do not clear PPO's existing gradients or mutate the cached distribution.
    A fixed actor still differentiates its output with respect to visual input.
    """
    parameters = [(p, p.requires_grad) for p in actor.parameters()]
    modes = [(module, module.training) for module in actor.modules()]
    try:
        actor.eval()
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


class RangeTaskBridgeAMPPPO(RangeJEPAAMPPPO):
    """Opt-in task-aligned A0 auxiliary learning, leaving AMP-PPO intact."""

    def __init__(self, actor, critic, storage, task_bridge_cfg=None, **kwargs):
        self._requested_task_bridge_cfg = None if task_bridge_cfg is None else validate_task_bridge_config(task_bridge_cfg)
        self.task_bridge_cfg = validate_task_bridge_config(task_bridge_cfg)
        self._task_rollout = None
        super().__init__(actor, critic, storage, **kwargs)
        self._validate_task_bridge_model()

    @property
    def task_bridge_coefficient(self):
        return self.task_bridge_cfg["coefficient"]

    @property
    def perception_extra_metric_names(self):
        return ("task",) if self.task_bridge_coefficient else ()

    def _validate_task_bridge_model(self):
        if not self.task_bridge_coefficient:
            return
        if type(self.perception) is not RangeJEPA or type(self.actor) is not RangeJEPAActor:
            raise ValueError("Task bridge v1 requires the original A0 RangeJEPA and RangeJEPAActor.")
        if type(self.actor.distribution) is not GaussianDistribution:
            raise ValueError("Task bridge v1 requires state-independent Gaussian actions.")
        if self.normalize_advantage_per_mini_batch:
            raise ValueError("Task bridge v1 requires rollout-normalized advantages.")
        if getattr(self, "symmetry", None) or getattr(self, "dwaq", None):
            raise ValueError("Task bridge v1 does not support symmetry or DWAQ augmentation.")
        if self.perception_epochs != 1:
            raise ValueError("Task bridge v1 requires exactly one auxiliary pass per rollout.")

    def _record_perception_capture(self, *args):
        if not self.task_bridge_coefficient:
            return super()._record_perception_capture(*args)
        return self.perception_storage.add_indexed(*args, control_step=self.storage.step)

    @torch.inference_mode(False)
    @torch.no_grad()
    def _snapshot_task_rollout(self):
        st = self.storage
        if st.step != st.num_transitions_per_env:
            raise ValueError("Task bridge requires a complete PPO rollout before update.")
        # Concatenation owns its result. The other scalar fields require clones
        # because AMP-PPO clears/reuses its storage before auxiliary learning.
        prop = torch.cat([st.observations[key] for key in self.actor.obs_groups], dim=-1).detach()
        advantage = st.weighted_advantages if self.is_multi_critic else st.advantages.squeeze(-1)
        if advantage.shape != st.actions.shape[:2]:
            raise ValueError("Task bridge requires one policy advantage per control step and environment.")
        for capture in self.perception_storage._batches:
            steps = capture.get("control_steps")
            if steps is None or steps.dtype != torch.long or steps.shape != capture["env_ids"].shape:
                raise ValueError("Fresh captures lack paired PPO control step indices.")
            if torch.any((steps < 0) | (steps >= st.step)):
                raise ValueError("Fresh-capture PPO step is outside the completed rollout.")
        return TaskRolloutSnapshot(prop, st.actions.detach().clone(),
            st.actions_log_prob.detach().squeeze(-1).clone(), advantage.detach().clone())

    def update(self):
        if not self.task_bridge_coefficient:
            return super().update()
        # This must happen BEFORE parent.update() clears PPO's rollout cursor.
        self._task_rollout = self._snapshot_task_rollout()
        try:
            return super().update()
        finally:
            self._task_rollout = None

    def _perception_auxiliary_kwargs(self, batch):
        if not self.task_bridge_coefficient:
            return {}
        if self._task_rollout is None or batch.control_steps is None:
            raise RuntimeError("Task bridge labels must be snapshotted before AMP-PPO update.")
        snapshot = self._task_rollout
        steps, envs = batch.control_steps, batch.env_ids
        prop = snapshot.proprioception[steps, envs]
        actions = snapshot.actions[steps, envs]
        old_log_prob = snapshot.old_log_prob[steps, envs]
        advantages = snapshot.advantages[steps, envs]

        def feature_loss(features, t):
            mean = self.actor.task_bridge_mean(prop[t], features)
            distribution = self.actor.distribution
            # GaussianDistribution.update() clamps scalar std in-place and
            # overwrites its cache. Use a pure expression on frozen parameters.
            std = (distribution.std_param if distribution.std_type == "scalar"
                   else distribution.log_std_param.exp()).clamp(*distribution.std_range)
            log_prob = torch.distributions.Normal(mean, std).log_prob(actions[t]).sum(-1)
            ratio = (log_prob - old_log_prob[t]).exp()
            surrogate = -advantages[t] * ratio
            clipped = -advantages[t] * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            return torch.maximum(surrogate, clipped)

        return {"feature_loss_fn": feature_loss, "feature_loss_weight": self.task_bridge_coefficient}

    @torch.inference_mode(False)
    @torch.enable_grad()
    def _update_perception(self):
        if not self.task_bridge_coefficient:
            return super()._update_perception()
        with frozen_actor(self.actor):
            metrics = super()._update_perception()
        self.last_perception_receipt["task_bridge"] = {
            **self.task_bridge_cfg,
            "sampling": "first_control_action_of_each_consumed_fresh_capture",
            "actor": "same_post_PPO_actor_frozen_during_auxiliary_pass",
            "scalar_labels_owned_before_PPO_clear": True,
            "extra_image_forwards": 0,
            "ppo_uses_cached_features_only": True,
            "new_deployment_parameters": 0,
        }
        return metrics

    def save(self):
        return {**super().save(), "task_bridge_cfg": deepcopy(self.task_bridge_cfg)}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        saved = validate_task_bridge_config(loaded_dict.get("task_bridge_cfg"))
        chosen = saved if self._requested_task_bridge_cfg is None else self._requested_task_bridge_cfg
        previous = self.task_bridge_cfg
        self.task_bridge_cfg = deepcopy(chosen)
        try:
            self._validate_task_bridge_model()
            result = super().load(loaded_dict, load_cfg, strict)
        except Exception:
            self.task_bridge_cfg = previous
            raise
        self.last_perception_receipt["task_bridge_cfg"] = deepcopy(self.task_bridge_cfg)
        return result
