"""Optional direct measured-point evidence for the existing compact CAMP Actor.

All 13 extra channels are derived from native XYZ and the original96 proprio.
The final two are purely leg/gyro body XY velocity; registration never updates
that estimate. Frozen recovery is preserved; PPO trains the control encoder.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

from .point_surface_policy import PointSurfaceActor, compact_points
from .point_memory.policy import PointMemoryPolicy

MODES = ('none', 'evidence', 'evidence_motion', 'motion')


def control_contract(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown point control mode: {mode}')
    return dict(version=1, architecture='direct_evidence', evidence_mode=mode,
                extra_channels=13, normalization='centered_anchor_v1',
                motion='Existing pure leg/gyro body XY estimate, no LiDAR correction or GT velocity.',
                gradient='PPO updates control encoder/Actor; recovery and geometric memory remain frozen.')


def normalize_evidence(evidence, max_age_s):
    """Normalize only available values; never use a label or future observation."""
    cur_v=(evidence[:,1:2]>.5).to(evidence.dtype)
    hist_v=(evidence[:,6:7]>.5).to(evidence.dtype)
    count=cur_v.sum((2,3),keepdim=True)
    hist_count=hist_v.sum((2,3),keepdim=True)
    cur_mean=(evidence[:,:1]*cur_v).sum((2,3),keepdim=True)/count.clamp_min(1)
    hist_mean=(evidence[:,5:6]*hist_v).sum((2,3),keepdim=True)/hist_count.clamp_min(1)
    mean=torch.where(count>0,cur_mean,torch.where(hist_count>0,hist_mean,torch.full_like(cur_mean,-1.2)))
    return torch.cat(((evidence[:,:1]-mean)*cur_v,cur_v,evidence[:,2:4]*cur_v,
        evidence[:,4:5]/.30*cur_v,(evidence[:,5:6]-mean)*hist_v,hist_v,
        evidence[:,7:9]*hist_v,evidence[:,9:10]/max_age_s*hist_v,
        evidence[:,10:11]/.05*hist_v),1)


def compact_evidence(obs, features, evidence):
    result=compact_points(obs,features)
    result['point_evidence']=evidence.detach().flatten(1)
    return result


def change_evidence_cache(compact,actor,mode):
    """Intervene on every point-derived path, retaining own leg XY velocity."""
    result=compact.clone(False)
    hvc=compact['range_features'].reshape(-1,3,actor.grid_size,actor.grid_size)
    extra=compact['point_evidence'].reshape(-1,13,actor.grid_size,actor.grid_size)
    raw,motion=extra[:,:11],extra[:,11:]
    if mode=='zero_cache':
        hvc=torch.zeros_like(hvc);raw=torch.zeros_like(raw)
    elif mode in ('shuffle_cache','same_terrain_shuffle'):
        if mode=='shuffle_cache':
            hvc=hvc.roll(len(hvc)//3,0);raw=raw.roll(len(raw)//3,0)
        else:
            hvc=hvc.reshape(3,-1,*hvc.shape[1:]).roll(1,1).reshape_as(hvc)
            raw=raw.reshape(3,-1,*raw.shape[1:]).roll(1,1).reshape_as(raw)
    elif mode=='flat_height':
        hvc=hvc.clone();hvc[:,0]=hvc[:,0].mean((-2,-1),keepdim=True)
        raw=raw.clone();raw[:,[0,4,5]]=0
    elif mode=='flip_height':
        # Mirror all geometric evidence coherently in forward x. Point
        # offsets change sign along that axis; own robot velocity does not.
        hvc=hvc.flip(-1);raw=raw.flip(-1).clone();raw[:,[2,7]]*=-1
    elif mode not in ('normal','no_history','empty_scan','front_occluded','rotate_scan'):
        raise ValueError(mode)
    result['range_features']=hvc.flatten(1)
    result['point_evidence']=torch.cat((raw,motion),1).flatten(1)
    return result


class PointEvidenceActor(PointSurfaceActor):
    def __init__(self,*args,evidence_mode='evidence_motion',**kwargs):
        control_contract(evidence_mode)
        super().__init__(*args,**kwargs)
        self.evidence_mode=evidence_mode
        old=self.surface_encoder[0]
        expanded=nn.Conv2d(16,12,3,stride=2,padding=1)
        with torch.no_grad():
            expanded.weight.zero_();expanded.weight[:,:3].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.surface_encoder[0]=expanded

    def get_latent(self,obs,masks=None,hidden_state=None):
        prop=self.prop_mlp(self.obs_normalizer(obs['policy']))
        hvc=obs['range_features'].detach().reshape(-1,3,self.grid_size,self.grid_size)
        extra=obs['point_evidence'].detach().reshape(-1,13,self.grid_size,self.grid_size).float()
        raw,motion=extra[:,:11],extra[:,11:]
        if self.evidence_mode=='none':
            raw=torch.zeros_like(raw);motion=torch.zeros_like(motion)
        elif self.evidence_mode=='evidence':
            motion=torch.zeros_like(motion)
        elif self.evidence_mode=='motion':
            raw=torch.zeros_like(raw)
        if not self.perception_enabled:
            # A trained blind arm keeps the same available proprioceptive
            # motion estimate, while removing EVERY point-derived channel.
            hvc=torch.zeros_like(hvc);raw=torch.zeros_like(raw)
        surface=torch.cat((hvc,raw,motion),1)
        with torch.autocast(device_type=surface.device.type,dtype=torch.bfloat16,
                            enabled=surface.is_cuda and self.encoder_amp):
            visual=self.surface_encoder(surface)
        return torch.cat((prop,visual.float()),-1)


class PointEvidencePolicy(PointMemoryPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for name in ('point_evidence','extra_capture','extra_episode'):
            self.register_buffer(name,None,persistent=False)

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self,dones=None):
        super().reset(dones)
        if dones is None:
            self.point_evidence=self.extra_capture=self.extra_episode=None
        elif self.point_evidence is not None:
            ids=dones.reshape(-1).bool()
            self.point_evidence[ids]=0;self.extra_capture[ids]=-1

    clear_memory=reset

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self,obs):
        capture=obs['range_capture_id'].reshape(-1)
        episode=obs['range_episode_id'].reshape(-1)
        if self.extra_episode is not None:
            self.reset(episode!=self.extra_episode)
        super().prepare(obs)
        if self.point_evidence is None:
            self.point_evidence=torch.zeros(len(capture),13,self.size,self.size,
                                           device=capture.device,dtype=torch.float16)
            self.extra_capture=torch.full_like(capture,-1)
            self.extra_episode=episode.clone()
        ids=((capture>=0)&(capture!=self.extra_capture)).nonzero().flatten()
        if ids.numel():
            normalized=normalize_evidence(self.last_evidence.index_select(0,ids),self.builder.max_age)
            self.point_evidence[ids,:11]=normalized.to(torch.float16)
        # This value comes from the existing motion prior BEFORE point
        # registration. It is not a ground-truth velocity or a new sensor.
        xy=self.builder.motion.velocity[:,:2]
        self.point_evidence[:,11:]=xy[:,:,None,None].to(torch.float16)
        self.extra_capture.copy_(capture);self.extra_episode.copy_(episode)
        return compact_evidence(obs,self.features,self.point_evidence)

    @torch.inference_mode(False)
    @torch.no_grad()
    def without_history_observations(self,compact):
        evidence=self.last_evidence.clone();evidence[:,5:]=0
        extra=self.point_evidence.clone()
        extra[:,:11]=normalize_evidence(evidence,self.builder.max_age).to(extra.dtype)
        return compact_evidence(compact,self.without_history_features(),extra)


def expand_legacy_checkpoint(saved,mode,actor):
    """Explicit warm start; preserve old weights and Adam moments exactly.

    The actual original optimizer builds actor.parameters() first, followed by
    critics. Locate the first conv by that parameter order, not a hardcoded ID.
    All original state and counters are retained; only new input weights and
    their matching moments start at zero.
    """
    if saved.get('point_control_contract') is not None:
        raise ValueError('Expected the original three-channel memory Actor')
    if saved['point_surface_contract'].get('architecture')!='evidence':
        raise ValueError('Expected the validated point-memory checkpoint')
    result=copy.deepcopy(saved)
    key='surface_encoder.0.weight'
    old=saved['actor_state_dict'][key]
    if tuple(old.shape)!=(12,3,3,3):raise ValueError('Unexpected legacy input convolution')
    weight=old.new_zeros(12,16,3,3);weight[:,:3]=old
    result['actor_state_dict'][key]=weight
    names=[name for name,_ in actor.named_parameters()]
    position=names.index(key)
    groups=result['optimizer_state_dict']['param_groups']
    if len(groups)!=1:raise ValueError('Unrecognized optimizer parameter groups')
    state_id=groups[0]['params'][position]
    moments=result['optimizer_state_dict']['state'][state_id]
    for name in ('exp_avg','exp_avg_sq'):
        value=moments[name]
        if value.shape!=old.shape:raise ValueError('Optimizer parameter order does not match original Actor')
        padded=value.new_zeros(weight.shape);padded[:,:3]=value;moments[name]=padded
    result['point_control_contract']=control_contract(mode)
    result['point_control_bootstrap_receipt']=dict(source_iteration=saved.get('iter'),
        actor_input_expanded='3->16',old_channels_and_optimizer_moments_preserved=True,
        new_channels_and_moments_zero=True,scope='Explicit architecture warm start, not from scratch.')
    return result
