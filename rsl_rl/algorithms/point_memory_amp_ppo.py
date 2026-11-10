"""Original CAMP AMP/PPO with deployment-only point memory, frozen recovery."""
from pathlib import Path
import copy
import torch
from rsl_rl.models.point_memory.recovery import EvidenceSurfaceCompleter
from rsl_rl.models.point_memory.policy import PointMemoryPolicy
from .amp_ppo import AMPPPO
from .point_surface_amp_ppo import PointSurfaceAMPPPO


class PointMemoryAMPPPO(PointSurfaceAMPPPO):
    def __init__(self,*args,point_memory_checkpoint=None,point_policy_bootstrap=None,**kwargs):
        # The frozen recovery is intentionally different from the legacy2channel
        # model; Actor/critics/AMP and cached PPO feature shape stay unchanged.
        kwargs.pop('point_surface_checkpoint',None)
        super().__init__(*args,point_surface_checkpoint=None,**kwargs)
        self._memory_size=self.point_contract['grid_size'];self._memory_extent=self.point_contract['extent_m']
        self.perception=EvidenceSurfaceCompleter().to(self.device).eval().requires_grad_(False)
        self.perception_ready=False;self._live_policy=None;self._inference_policy=None
        self.memory_bootstrap_receipt=None
        if point_memory_checkpoint:
            saved=torch.load(Path(point_memory_checkpoint),map_location=self.device,weights_only=False)
            if saved.get('architecture')=='evidence':
                contract=dict(architecture='evidence',grid_size=self._memory_size,extent_m=self._memory_extent,
                    projection='nearest',model_cfg=copy.deepcopy(saved['model_cfg']),
                    kinematic_contract=copy.deepcopy(saved['kinematic_contract']),
                    memory_cfg=dict(max_age_s=saved['model_cfg']['max_age_s'],dt=.02),
                    perception_inputs='Only existing nativeXYZ and96proprio/commands/staticcalibration; no truepose/contact/terrain.',
                    perception_gradient='Frozen offline query-height recovery and geometric memory; PPO updates spatial encoder/Actor.',
                    height_semantics=('Estimated query-center height including occupied cells.' if saved['model_cfg'].get('correct_current',True) else 'Measured cells retain actual point-anchor height; missing cells are estimated query heights. An off-center measured anchor is not exact query-center truth.'))
                if not self.actor.perception_enabled:contract['perception_enabled']=False
                self._install_recovery(contract,saved['state_dict'])
            elif saved.get('point_surface_contract',{}).get('architecture')=='evidence':
                self._install_recovery(saved['point_surface_contract'],saved['point_surface_state_dict'])
            else:raise ValueError('Expected an explicit evidence recovery or memory-policy checkpoint')
        if point_policy_bootstrap:
            if not self.perception_ready:raise ValueError('Validate frozen evidence recovery before control bootstrap')
            source=torch.load(Path(point_policy_bootstrap),map_location=self.device,weights_only=False)
            contract=source.get('point_surface_contract',{})
            if contract.get('architecture')!='dilated' or contract.get('grid_size')!=self._memory_size or contract.get('extent_m')!=self._memory_extent:
                raise ValueError('Bootstrap requires the documented matching legacy point Actor')
            # Explicit warmstart of the SAME Actor/critic/optimizer/AMP, without
            # copying the old2channel recovery into the new11channel recovery.
            AMPPPO.load(self,source,None,True)
            self.memory_bootstrap_receipt=dict(source=str(Path(point_policy_bootstrap).resolve()),
                source_iteration=source.get('iter'),actor_critic_optimizer_amp_loaded=True,
                old_recovery_loaded=False,scope='Explicit warmstart, not fromscratch or exact old observation continuation')

    def _install_recovery(self,contract,state):
        contract=copy.deepcopy(contract)
        if contract.get('architecture')!='evidence' or contract.get('grid_size')!=self._memory_size or contract.get('extent_m')!=self._memory_extent:
            raise ValueError('Memory checkpoint and original CAMP grid metadata differ')
        model_cfg=contract['model_cfg'];static=contract['kinematic_contract']
        if not static or len(static['joint_names'])!=29:raise ValueError('Missing static G1 calibration, never infer it from liveposes')
        self.actor.perception_enabled=contract.get('perception_enabled',True)
        self.perception=EvidenceSurfaceCompleter(**model_cfg).to(self.device)
        self.perception.load_state_dict(state,strict=True);self.perception.eval().requires_grad_(False)
        self.point_contract=contract
        options=dict(grid_size=self._memory_size,extent_m=self._memory_extent,
            max_age_s=contract['memory_cfg']['max_age_s'],dt=contract['memory_cfg']['dt'])
        self._live_policy=PointMemoryPolicy(self.actor,self.perception,static,**options).to(self.device)
        self._inference_policy=PointMemoryPolicy(self.actor,self.perception,static,**options).to(self.device)
        self.perception_ready=True

    def get_policy(self):
        if not self.perception_ready:raise RuntimeError('Load a validated point-memory checkpoint before inference')
        return self._inference_policy

    def load(self,loaded_dict,load_cfg=None,strict=True):
        contract=loaded_dict.get('point_surface_contract',{})
        self._install_recovery(contract,loaded_dict['point_surface_state_dict'])
        result=super().load(loaded_dict,load_cfg,strict)
        self.memory_bootstrap_receipt=loaded_dict.get('point_memory_bootstrap_receipt')
        return result

    def save(self):
        if not self.perception_ready:raise RuntimeError('Cannot save uninitialized memory recovery')
        return {**super().save(),'point_memory_bootstrap_receipt':self.memory_bootstrap_receipt}
