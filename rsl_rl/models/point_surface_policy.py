"""Native XYZ -> fixed accurate surface recovery -> trainable joint policy.

The first point baseline freezes the small offline recovery model. PPO updates
the spatial control encoder and joint policy. It neither re-trains recovery
nor routes privileged observations into its deployment computation.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from .point_surface import PointSurfaceProjector


def compact_points(obs,features):
    allowed=('policy','critic','height_scan_critic','amp')
    return TensorDict({**{k:obs[k] for k in allowed if k in obs},'range_features':features.detach()},
                      batch_size=obs.batch_size,device=features.device)


class PointSurfaceActor(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,hidden_dims=(512,256,128),activation='elu',
                 obs_normalization=True,distribution_cfg=None,grid_size=32,encoder_amp=True,perception_enabled=True):
        groups=copy.deepcopy(obs_groups)
        groups[obs_set]=[k for k in groups[obs_set] if k!='range_features']
        if groups[obs_set]!=['policy'] or obs['policy'].shape[-1]!=96:
            raise ValueError('Point Actor only accepts the original 96-value policy proprioception.')
        if obs['range_features'].shape[-1]!=3*grid_size**2:
            raise ValueError('Point Actor surface cache must contain H, measured validity and confidence.')
        self.grid_size,self.encoder_amp=grid_size,encoder_amp
        if type(perception_enabled) is not bool:
            raise TypeError('perception_enabled must be a boolean experimental constant.')
        self.perception_enabled=perception_enabled
        super().__init__(obs,groups,obs_set,output_dim,hidden_dims,activation,obs_normalization,copy.deepcopy(distribution_cfg))
        self.prop_mlp=MLP(self.obs_dim,64,(128,),activation)
        self.surface_encoder=nn.Sequential(nn.Conv2d(3,12,3,stride=2,padding=1),nn.ELU(),
            nn.Conv2d(12,24,3,stride=2,padding=1),nn.ELU(),nn.Conv2d(24,24,3,stride=2,padding=1),nn.ELU(),
            nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(384,64),nn.ELU())

    def _get_latent_dim(self):return 128

    def get_latent(self,obs,masks=None,hidden_state=None):
        prop=self.prop_mlp(self.obs_normalizer(obs['policy']))
        surface=obs['range_features'].detach().reshape(-1,3,self.grid_size,self.grid_size)
        if not self.perception_enabled:
            # Same model and compute graph; no point-derived value or mask can
            # reach the trained blind controller, including NaN/validity cues.
            surface=torch.zeros_like(surface)
        with torch.autocast(device_type=surface.device.type,dtype=torch.bfloat16,
                            enabled=surface.is_cuda and self.encoder_amp):
            visual=self.surface_encoder(surface)
        return torch.cat((prop,visual.float()),-1)

    def as_jit(self):raise NotImplementedError('Use raw-point policy with --skip_export until stateful export is validated.')
    def as_onnx(self,**kwargs):return self.as_jit()


class PointSurfacePolicy(nn.Module):
    def __init__(self,actor,perception,*,grid_size=32,extent_m=2.4,projection_chunk_size=128,projection='max'):
        super().__init__()
        self.actor,self.perception=actor,perception
        self.projector=PointSurfaceProjector(grid_size,extent_m,projection=projection)
        self.grid_size,self.projection_chunk_size=grid_size,projection_chunk_size
        for name in ('features','last_capture','episode'):
            self.register_buffer(name,None,persistent=False)

    @property
    def output_std(self):return self.actor.output_std
    @property
    def is_recurrent(self):return False

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self,dones=None):
        if dones is None:
            self.features=self.last_capture=self.episode=None
        elif self.features is not None:
            ids=dones.bool().reshape(-1)
            self.features[ids]=0;self.last_capture[ids]=-1
        self.actor.reset(dones)

    clear_memory=reset

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self,obs):
        points=obs['range_obs']
        if points.ndim!=3 or points.shape[-1]!=3:
            raise ValueError('Point policy requires native [B,R,3] XYZ only.')
        b=len(points);capture=obs['range_capture_id'].reshape(-1);episode=obs['range_episode_id'].reshape(-1)
        if capture.dtype!=torch.long or episode.dtype!=torch.long:
            raise ValueError('Capture identities must be int64, not extra point attributes.')
        if self.features is None:
            self.features=points.new_zeros(b,3*self.grid_size**2)
            self.last_capture=torch.full_like(capture,-1);self.episode=episode.clone()
        if len(self.features)!=b:raise ValueError('Batch changed; reset the point policy.')
        changed=episode!=self.episode
        self.features[changed]=0;self.last_capture[changed]=-1;self.episode.copy_(episode)
        fresh=((capture>=0)&(capture!=self.last_capture)).nonzero().flatten()
        for ids in fresh.split(self.projection_chunk_size):
            if not ids.numel():
                continue
            measured=self.projector(points.index_select(0,ids),obs['policy'].index_select(0,ids))
            # Partial resets make the tail batch vary almost every control.
            # Searching cuDNN plans for each new row count dominates rollout
            # time; only fixed PPO batches may use the surrounding setting.
            with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=False,
                    deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32):
                output=self.perception(measured)
            confidence=torch.exp(-output['log_error_scale'].exp()/.05)
            self.features[ids]=torch.cat((output['height'],output['observed'],confidence),1).flatten(1)
        self.last_capture[fresh]=capture[fresh]
        return compact_points(obs,self.features)

    @torch.no_grad()
    def forward(self,obs,**kwargs):return self.actor(self.prepare(obs),**kwargs)
