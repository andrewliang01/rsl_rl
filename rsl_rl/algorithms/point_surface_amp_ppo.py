"""Original AMP/PPO updates on cached point-derived surface geometry."""
from pathlib import Path
import torch
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.models.point_surface import DilatedSurfaceCompleter
from rsl_rl.models.point_surface_policy import PointSurfacePolicy,compact_points
from .amp_ppo import AMPPPO
from .ppo_builders import construct_multi_critic_algorithm


class PointSurfaceAMPPPO(AMPPPO):
    def __init__(self,*args,point_surface_checkpoint=None,point_grid_size=32,point_extent_m=2.4,
                 point_projection_chunk_size=128,**kwargs):
        super().__init__(*args,**kwargs)
        self.point_contract=dict(architecture='dilated',grid_size=point_grid_size,extent_m=point_extent_m,
            perception_inputs='native XYZ plus existing gravity/waist joints for projection',
            perception_gradient='frozen offline recovery; PPO trains spatial control encoder and Actor')
        if not self.actor.perception_enabled:
            self.point_contract['perception_enabled']=False
        self.perception=DilatedSurfaceCompleter().to(self.device)
        self.perception_ready=point_surface_checkpoint is not None
        if point_surface_checkpoint:
            saved=torch.load(Path(point_surface_checkpoint),map_location=self.device,weights_only=False)
            if saved['architecture']!='dilated':raise ValueError('Point recovery architecture mismatch.')
            projection=saved.get('projection','max')
            if projection!='max':
                self.point_contract['projection']=projection
            self.perception.load_state_dict(saved['state_dict'])
        self.perception.requires_grad_(False).eval()
        options=dict(grid_size=point_grid_size,extent_m=point_extent_m,projection_chunk_size=point_projection_chunk_size,
                     projection=self.point_contract.get('projection','max'))
        self._live_policy=PointSurfacePolicy(self.actor,self.perception,**options).to(self.device)
        self._inference_policy=PointSurfacePolicy(self.actor,self.perception,**options).to(self.device)
        for module in [self.actor,*self.critics,self.amp_discriminator]:
            if module is not None:
                for child in module.modules():
                    if isinstance(child,EmpiricalNormalization):child.synchronize_distributed=self.is_multi_gpu

    @classmethod
    def construct_algorithm(cls,obs,env,cfg,device):
        unwrapped=getattr(env,'unwrapped',env)
        if unwrapped.cfg.camp_perception_backend!='point_surface':
            raise ValueError('Point agent requires the point_surface backend in the original G1CampEnvCfg.')
        size=cfg['algorithm'].get('point_grid_size',32)
        if size!=unwrapped.cfg.camp_grid_size:raise ValueError('Point grid metadata differs from environment.')
        compact=compact_points(obs,torch.zeros(env.num_envs,3*size**2,device=device))
        return construct_multi_critic_algorithm(cls,compact,env,cfg,device,include_amp_obs=True)

    def act(self,obs):
        if not self.perception_ready:
            raise RuntimeError('Load a validated point_surface_checkpoint before training frozen recovery.')
        return super().act(self._live_policy.prepare(obs))

    def process_env_step(self,obs,rewards,dones,extras):
        super().process_env_step(compact_points(obs,self._live_policy.features.clone()),rewards,dones,extras)
        self._live_policy.reset(dones)

    def compute_returns(self,obs):return super().compute_returns(compact_points(obs,self._live_policy.features.clone()))

    def get_policy(self):return self._inference_policy

    def train_mode(self):
        super().train_mode();self.perception.eval()

    def eval_mode(self):
        super().eval_mode();self.perception.eval()

    @torch.no_grad()
    def broadcast_parameters(self):
        super().broadcast_parameters()
        for value in self.perception.state_dict().values():torch.distributed.broadcast(value,src=0)

    def update(self):
        result=super().update()
        if self.is_multi_gpu:
            for child in self.critics.modules():
                if isinstance(child,torch.nn.modules.batchnorm._BatchNorm):
                    for value in child.buffers(recurse=False):torch.distributed.broadcast(value,src=0)
        return result

    def save(self):
        return {**super().save(),'point_surface_state_dict':self.perception.state_dict(),
                'point_surface_contract':self.point_contract}

    def load(self,loaded_dict,load_cfg=None,strict=True):
        if loaded_dict.get('point_surface_contract')!=self.point_contract:
            raise ValueError('Checkpoint is not a matching point-surface policy; no implicit old-policy resume.')
        result=super().load(loaded_dict,load_cfg,strict)
        self.perception.load_state_dict(loaded_dict['point_surface_state_dict'],strict=True)
        self.perception_ready=True
        self._live_policy.reset();self._inference_policy.reset()
        return result
