"""CAMP uses the existing AMP/PPO and fresh-capture supervised update loops."""

import hashlib
from pathlib import Path
import torch
from rsl_rl.modules import EmpiricalNormalization

from .range_evidence_amp_ppo import RangeEvidenceAMPPPO


class CampAMPPPO(RangeEvidenceAMPPPO):
    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        from .range_jepa_amp_ppo import _model_configuration
        model = _model_configuration(cfg['algorithm'].get('perception_cfg'))
        environment = getattr(env, 'unwrapped', env)
        if model['bev_grid_size'] != environment.cfg.camp_grid_size or model['bev_extent_m'] != environment.cfg.camp_extent_m:
            raise ValueError('CAMP terrain supervision and neural grid metadata differ.')
        return super().construct_algorithm(obs, env, cfg, device)

    def __init__(self, *args, perception_bootstrap=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Rank-local observations differ, but deployment and checkpoints must
        # use one shared set of Actor/Critic/AMP running statistics.
        modules = [self.actor, *self.critics]
        if self.amp_discriminator is not None:
            modules.append(self.amp_discriminator)
        for module in modules:
            for normalizer in module.modules():
                if isinstance(normalizer, EmpiricalNormalization):
                    normalizer.synchronize_distributed = self.is_multi_gpu
        self.bootstrap_receipt = None
        self._camp_auxiliary_source = None
        if perception_bootstrap:
            self.bootstrap_recovery(perception_bootstrap)

    @torch.no_grad()
    def _sync_critic_batchnorm_buffers(self):
        if self.is_multi_gpu:
            # Keep the original per-minibatch BatchNorm training behavior.
            # Publish rank 0's running statistics at each update boundary so
            # checkpoints/evaluation describe one shared Critic, like DDP's
            # buffer broadcast. These buffers do not affect train-mode output.
            for module in self.critics.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    for buffer in module.buffers(recurse=False):
                        torch.distributed.broadcast(buffer, src=0)

    def update(self):
        result = super().update()
        self._sync_critic_batchnorm_buffers()
        return result

    @torch.no_grad()
    def bootstrap_recovery(self, path):
        path = Path(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        contract = saved.get('perception_model_cfg', {})
        if contract.get('variant') != 'H' or contract.get('auxiliary_objective') != 'direct_geometry':
            raise ValueError('CAMP bootstrap must be an explicit HDirect recovery checkpoint, not a resume.')
        if contract.get('image_encoding', 'depth_valid') != 'depth_valid':
            raise ValueError('Bootstrap image encoding must match binary validity.')
        source = saved['perception_state_dict']
        current = self.perception.state_dict()
        parameter_names = set(dict(self.perception.named_parameters()))
        copied = [key for key in parameter_names if key in source and source[key].shape == current[key].shape]
        required = [key for key in parameter_names if key.startswith(('encoder.', 'gru.'))]
        if not required or not set(required) <= set(copied):
            raise ValueError('Bootstrap does not match the observed CNN/GRU architecture.')
        for key in copied:
            current[key].copy_(source[key])
        body_source = saved.get('actor_state_dict', {})
        own = self.actor.state_dict()
        body_copied = []
        for key, value in body_source.items():
            if key.startswith(('prop_mlp.', 'obs_normalizer.')) and key in own and own[key].shape == value.shape:
                own[key].copy_(value)
                body_copied.append(key)
        self.bootstrap_receipt = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            recovery_parameters=copied, body_parameters=body_copied, input_fov=contract.get('vertical_fov'),
            output_fov=self.perception_model_cfg['vertical_fov'],
            mode='parameter transfer; new native sensor frame, geometric buffers rebuilt; no policy/optimizer resume')
        print('CAMP_BOOTSTRAP', self.bootstrap_receipt, flush=True)

    def act(self, obs):
        if 'camp_terrain_targets' not in obs:
            raise KeyError('CAMP supervised training requires terrain targets outside the Actor input.')
        self._camp_auxiliary_source = obs['camp_terrain_targets']
        try:
            return super().act(obs)
        finally:
            self._camp_auxiliary_source = None

    def _record_perception_capture(self, *args):
        saved = super()._record_perception_capture(*args)
        if saved is not None:
            saved['terrain_targets'] = self._camp_auxiliary_source.detach().index_select(0, args[4])
        return saved

    def _perception_auxiliary_kwargs(self, batch):
        return dict(terrain_targets=batch.terrain_targets)

    def save(self):
        return {**super().save(), 'camp_bootstrap_receipt': self.bootstrap_receipt}
