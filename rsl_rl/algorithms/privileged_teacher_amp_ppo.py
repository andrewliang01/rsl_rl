"""Training-only geometric teacher inside originalCAMP, never a point policy."""
from .amp_ppo import AMPPPO
from .ppo_builders import construct_multi_critic_algorithm


class CampPrivilegedTeacherAMPPPO(AMPPPO):
    @classmethod
    def construct_algorithm(cls,obs,env,cfg,device):
        raw=getattr(env,'unwrapped',env)
        if not raw.cfg.camp_training_supervision or raw.cfg.camp_perception_backend!='point_surface':
            raise ValueError('Privileged teacher requires explicit originalCAMP supervision role.')
        if cfg['obs_groups']['actor']!=['policy','height_scan_policy'] or 'range_obs' in obs:
            raise ValueError('Teacher may not mix its GTheight branch with native-point actor observations.')
        algorithm=construct_multi_critic_algorithm(cls,obs,env,cfg,device,include_amp_obs=True)
        algorithm.teacher_contract=dict(role='PRIVILEGED_TRAINING_SUPERVISION_ONLY',
            inputs='96proprio and5x28x20referenceheight; NOT deployablepointpolicy',
            purpose='Refine gait/control for later point-only student supervision, not lidar-onlyperformance')
        return algorithm

    def save(self):
        return {**super().save(),'privileged_teacher_contract':self.teacher_contract}
