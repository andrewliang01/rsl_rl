"""Native-point memory policy with cached frozen query-height recovery."""
import torch
from torch import nn
from rsl_rl.models.point_surface_policy import compact_points
from .inputs import PointMemoryInputBuilder


class PointMemoryPolicy(nn.Module):
    def __init__(self,actor,perception,kinematic_contract,*,grid_size=32,extent_m=2.4,
                 max_age_s=2.4,dt=.02,recovery_chunk_size=128):
        super().__init__()
        self.actor,self.perception=actor,perception
        self.size=int(grid_size);self.chunk=int(recovery_chunk_size)
        self.builder=PointMemoryInputBuilder(kinematic_contract,grid_size=grid_size,extent_m=extent_m,
            max_age_s=max_age_s,dt=dt)
        self.history_override=None
        for name in ('features','last_evidence','current_raw_surface'):
            self.register_buffer(name,None,persistent=False)

    @property
    def output_std(self):return self.actor.output_std
    @property
    def is_recurrent(self):return False

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self,dones=None):
        self.builder.reset(dones)
        if dones is None:
            self.features=self.last_evidence=self.current_raw_surface=None
        elif self.features is not None:
            ids=dones.reshape(-1).bool()
            self.features[ids]=0;self.last_evidence[ids]=0;self.current_raw_surface[ids]=0
        self.actor.reset(dones)

    clear_memory=reset

    def _recover(self,evidence):
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=False,
                deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32):
            prediction=self.perception(evidence)
        confidence=torch.exp(-prediction['log_error_scale'].exp()/.05)
        return torch.cat((prediction['height'],prediction['observed'],confidence),1).flatten(1)

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self,obs):
        points=obs['range_obs'];proprio=obs['policy'];b=len(points)
        result=self.builder.step(points,proprio,obs['range_capture_id'],obs['range_episode_id'])
        if self.features is None:
            self.features=points.new_zeros(b,3*self.size**2)
            self.last_evidence=points.new_zeros(b,11,self.size,self.size)
            self.current_raw_surface=points.new_zeros(b,2,self.size,self.size)
        if b!=len(self.features):raise ValueError('Batch changed; reset memory policy')
        ids=result['ids']
        if ids.numel():
            evidence=result['inputs']
            if self.history_override is False:
                evidence=evidence.clone();evidence[:,5:]=0
            self.last_evidence.index_copy_(0,ids,evidence)
            self.current_raw_surface.index_copy_(0,ids,evidence[:,:2])
            for i,chunk in enumerate(ids.split(self.chunk)):
                x=evidence[i*self.chunk:i*self.chunk+len(chunk)]
                self.features[chunk]=self._recover(x)
        return compact_points(obs,self.features)

    @torch.inference_mode(False)
    @torch.no_grad()
    def without_history_features(self):
        """Same-state, same-current-evidence intervention for diagnostics only."""
        if self.last_evidence is None:raise RuntimeError('Prepare a physical scan before intervention')
        outputs=[]
        for x in self.last_evidence.split(self.chunk):
            x=x.clone();x[:,5:]=0
            outputs.append(self._recover(x))
        return torch.cat(outputs,0)

    @torch.no_grad()
    def forward(self,obs,**kwargs):return self.actor(self.prepare(obs),**kwargs)
