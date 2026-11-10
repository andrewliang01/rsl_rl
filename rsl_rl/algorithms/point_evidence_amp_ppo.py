"""Original CAMP AMP/PPO with optional direct point-evidence control input."""
import torch
from rsl_rl.models.point_evidence_control import (
    PointEvidencePolicy, compact_evidence, control_contract,
)
from .point_memory_amp_ppo import PointMemoryAMPPPO
from .amp_ppo import AMPPPO
from .ppo_builders import construct_multi_critic_algorithm


class PointEvidenceAMPPPO(PointMemoryAMPPPO):
    def __init__(self,*args,point_memory_checkpoint=None,point_policy_bootstrap=None,**kwargs):
        if point_policy_bootstrap:
            raise ValueError('Full control checkpoints require the original explicit resume entrypoint')
        self._recovery_only_initialization = point_memory_checkpoint is not None
        self._recovery_only_source = point_memory_checkpoint
        if self._recovery_only_initialization:
            from .point_recovery_init import validate_recovery_only
            validate_recovery_only(point_memory_checkpoint)
        super().__init__(*args,point_memory_checkpoint=point_memory_checkpoint,point_policy_bootstrap=None,**kwargs)
        self._recovery_only_initialization = False

    @classmethod
    def construct_algorithm(cls,obs,env,cfg,device):
        unwrapped=getattr(env,'unwrapped',env)
        size=cfg['algorithm'].get('point_grid_size',32)
        if unwrapped.cfg.camp_perception_backend!='point_surface' or size!=unwrapped.cfg.camp_grid_size:
            raise ValueError('Direct evidence requires the original CAMP point backend and matching grid')
        compact=compact_evidence(obs,torch.zeros(env.num_envs,3*size**2,device=device),
            torch.zeros(env.num_envs,13,size,size,device=device,dtype=torch.float16))
        return construct_multi_critic_algorithm(cls,compact,env,cfg,device,include_amp_obs=True)

    def _install_recovery(self,contract,state):
        before = None
        if self._recovery_only_initialization:
            from .point_recovery_init import snapshot_core
            before = snapshot_core(self)
        super()._install_recovery(contract,state)
        options=dict(grid_size=self._memory_size,extent_m=self._memory_extent,
            max_age_s=contract['memory_cfg']['max_age_s'],dt=contract['memory_cfg']['dt'])
        self._live_policy=PointEvidencePolicy(self.actor,self.perception,contract['kinematic_contract'],**options).to(self.device)
        self._inference_policy=PointEvidencePolicy(self.actor,self.perception,contract['kinematic_contract'],**options).to(self.device)
        if before is not None:
            from .point_recovery_init import verify_recovery_only_initialization
            self.memory_bootstrap_receipt = verify_recovery_only_initialization(self, before, self._recovery_only_source)

    def process_env_step(self,obs,rewards,dones,extras):
        compact=compact_evidence(obs,self._live_policy.features.clone(),self._live_policy.point_evidence.clone())
        AMPPPO.process_env_step(self,compact,rewards,dones,extras)
        self._live_policy.reset(dones)

    def compute_returns(self,obs):
        compact=compact_evidence(obs,self._live_policy.features.clone(),self._live_policy.point_evidence.clone())
        return AMPPPO.compute_returns(self,compact)

    def load(self,loaded_dict,load_cfg=None,strict=True):
        expected=control_contract(self.actor.evidence_mode)
        if loaded_dict.get('point_control_contract')!=expected:
            raise ValueError('Checkpoint and direct-evidence Actor control mode differ')
        return super().load(loaded_dict,load_cfg,strict)

    def save(self):
        return {**super().save(),'point_control_contract':control_contract(self.actor.evidence_mode)}
