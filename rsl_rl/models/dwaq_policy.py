"""Inference and export for a feedforward DWAQ actor with frame-major history."""
import copy
import torch
from torch import nn


class DWAQInferencePolicy(nn.Module):
    def __init__(self, actor, estimator):
        super().__init__()
        if estimator.frame_term_dims is not None or estimator.use_adaboot:
            raise ValueError('DWAQ inference supports frame-major history and no privileged bootstrapping')
        self.actor = actor
        self.estimator = estimator
        self.group = estimator.code_append_obs_group

    def forward(self, observations):
        data = observations.clone()
        data[self.group] = self.estimator.get_actor_observation(observations, deterministic=True)
        return self.actor(data)

    @property
    def output_std(self):
        """Expose the actor's current action standard deviation to the runner."""
        return self.actor.output_std

    def as_jit(self):
        return DWAQExport(self.actor, self.estimator)

    def as_onnx(self, verbose=False, input_mode='single'):
        if input_mode not in ('single', 'split'):
            raise ValueError('Unsupported DWAQ export input mode')
        return DWAQExport(self.actor, self.estimator)


class DWAQExport(nn.Module):
    def __init__(self, actor, estimator):
        super().__init__()
        self.normalizer = copy.deepcopy(estimator.state_normalizer)
        self.backbone = copy.deepcopy(estimator.encoder_backbone)
        self.vel = copy.deepcopy(estimator.encoder_vel_mean)
        self.latent = copy.deepcopy(estimator.encoder_latent_mean)
        self.actor = actor.as_jit()
        self.input_size = estimator.num_history_len * estimator.obs_one_frame_len
        self.frame_size = estimator.obs_one_frame_len

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer(history)
        features = self.backbone(normalized)
        current = normalized[..., -self.frame_size:]
        return self.actor(torch.cat((current, self.vel(features), self.latent(features)), dim=-1))

    @torch.jit.export
    def reset(self) -> None:
        pass

    @torch.jit.ignore
    def get_dummy_inputs(self):
        return (torch.zeros(1, self.input_size),)

    @property
    @torch.jit.ignore
    def input_names(self):
        return ['obs_history']

    @property
    @torch.jit.ignore
    def output_names(self):
        return ['actions']
