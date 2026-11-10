"""NativeXYZ + existing96proprio -> current and transported-point evidence.

Reset/capture IDs are internal bookkeeping, not extra LiDAR attributes.
The builder never accepts simulation labels or a world-pose estimate.
"""
import torch
from torch import nn
from .kinematics import KinematicMotionPrior
from .state import RollingPointMemory
from .registration import register
from rsl_rl.models.point_evidence import PointEvidenceReducer


class PointMemoryInputBuilder(nn.Module):
    def __init__(self,contract,*,grid_size=32,extent_m=2.4,max_age_s=2.4,dt=.02):
        super().__init__()
        self.size=int(grid_size);self.extent=float(extent_m);self.max_age=float(max_age_s);self.dt=float(dt)
        self.motion=KinematicMotionPrior(contract,dt)
        self.memory=RollingPointMemory(grid_size,extent_m,max_age_s)
        self.reduce=PointEvidenceReducer(grid_size,extent_m)
        self.tick=0
        for name in ('previous_native','previous_level','last_position','last_rotation','last_capture',
                     'last_tick','episode','has_capture','uncertain'):
            self.register_buffer(name,None,persistent=False)

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self,dones=None):
        self.motion.reset(dones);self.memory.reset(dones)
        if dones is None:
            for name in ('previous_native','previous_level','last_position','last_rotation','last_capture',
                         'last_tick','episode','has_capture','uncertain'):
                setattr(self,name,None)
            self.tick=0
        elif self.last_capture is not None:
            ids=dones.reshape(-1).bool()
            self.last_capture[ids]=-1;self.has_capture[ids]=False;self.uncertain[ids]=False

    @torch.inference_mode(False)
    @torch.no_grad()
    def step(self,native,proprio,capture,episode):
        if native.ndim!=3 or native.shape[-1]!=3 or len(native)!=len(proprio):
            raise ValueError('Expected native[B,R,3] and existingproprio[B,96]')
        b=len(native);capture=capture.reshape(-1);episode=episode.reshape(-1)
        if capture.dtype!=torch.long or episode.dtype!=torch.long:
            raise TypeError('Capture/reset counters must be int64')
        if self.previous_native is None:
            self.previous_native=torch.zeros_like(native)
            eye=torch.eye(3,device=native.device,dtype=native.dtype)[None].expand(b,-1,-1)
            self.previous_level=eye.clone();self.last_rotation=eye.clone()
            self.last_position=native.new_zeros(b,3)
            self.last_capture=torch.full_like(capture,-1);self.last_tick=torch.zeros_like(capture)
            self.episode=episode.clone();self.has_capture=torch.zeros(b,device=native.device,dtype=torch.bool)
            self.uncertain=torch.zeros_like(self.has_capture)
            self.memory.initialize(b,native)
        if self.previous_native.shape!=native.shape:raise ValueError('Batch/raylayout changed; reset builder')
        changed=episode!=self.episode
        self.reset(changed);self.episode.copy_(episode)
        motion=self.motion.step(proprio)
        self.uncertain|=motion['support_uncertain']
        ids=((capture>=0)&(capture!=self.last_capture)).nonzero().flatten()
        if not ids.numel():
            self.tick+=1
            return dict(ids=ids,inputs=None)
        points=native.index_select(0,ids);prop=proprio.index_select(0,ids)
        reduced=self.reduce(points,prop)
        current=reduced['anchors'][:,2].flatten(1,2)
        current_valid=reduced['observed'].flatten(1)
        level=motion['level'].index_select(0,ids)
        new_r=motion['initial_from_level'].index_select(0,ids)
        new_p=motion['sensor_position'].index_select(0,ids)
        old_r=self.last_rotation.index_select(0,ids)
        prior_r=old_r.transpose(1,2)@new_r
        prior_t=(old_r.transpose(1,2)@(new_p-self.last_position.index_select(0,ids))[...,None])[...,0]
        first=~self.has_capture.index_select(0,ids)
        identity=torch.eye(3,device=native.device,dtype=native.dtype)[None]
        prior_r=torch.where(first[:,None,None],identity,prior_r)
        prior_t=torch.where(first[:,None],0.,prior_t)
        rotation,translation,info=register(self.previous_native.index_select(0,ids),points,
            self.previous_level.index_select(0,ids),level,prior_r,prior_t)
        usable=(info['inliers']>=12)&(info['observable_dimensions']>=3)&~first
        residual=info['residual_mae_m'].clamp(0.,.12)
        sigma=(residual.square()+.002**2+torch.where(usable,0.,.02**2)
               +self.uncertain.index_select(0,ids).to(residual.dtype)*.005**2).sqrt()
        elapsed=(self.tick-self.last_tick.index_select(0,ids)).to(native.dtype)*self.dt
        history=self.memory.update(current,current_valid,rotation,translation,elapsed,sigma,indices=ids)
        centers=self.memory.centers[None]
        cell=self.extent/self.size
        cur_offset=(current[...,:2]-centers)/cell
        cur_span=(reduced['height_max']-reduced['height_min']).flatten(1)
        old=history['anchors'];old_valid=history['valid']
        hist_offset=(old[...,:2]-centers)/cell
        values=(current[...,2],current_valid.to(native.dtype),
            cur_offset[...,0],cur_offset[...,1],cur_span,
            old[...,2],old_valid.to(native.dtype),hist_offset[...,0],hist_offset[...,1],
            history['age'],history['error_hint'])
        inputs=torch.stack(values,1).reshape(-1,11,self.size,self.size)
        # Missing values carry explicit masks and no fabricated point offset.
        cur_mask=current_valid.reshape(-1,1,self.size,self.size)
        hist_mask=old_valid.reshape(-1,1,self.size,self.size)
        inputs[:,:5]*=cur_mask;inputs[:,5:]*=hist_mask
        self.previous_native.index_copy_(0,ids,points);self.previous_level.index_copy_(0,ids,level)
        self.last_rotation.index_copy_(0,ids,new_r);self.last_position.index_copy_(0,ids,new_p)
        self.last_capture[ids]=capture[ids];self.last_tick[ids]=self.tick
        self.has_capture[ids]=True;self.uncertain[ids]=False
        self.tick+=1
        return dict(ids=ids,inputs=inputs,current_anchors=current,history_anchors=old,
                    registration=info,relative_rotation=rotation,relative_translation=translation,
                    history_age=history['age'],history_error_hint=history['error_hint'])
