"""Existing96-proprio -> relative sensor motion; static calibration only.

No simulator imports, contact sensor, world pose, terrain or odometry input.
The support heuristic is approximate; real point registration corrects it.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def axis_rotation(angle,axis):
    c,s=angle.cos(),angle.sin()
    z=torch.zeros_like(c);o=torch.ones_like(c)
    if axis==0:values=(o,z,z,z,c,-s,z,s,c)
    elif axis==1:values=(c,z,s,z,o,z,-s,z,c)
    elif axis==2:values=(c,-s,z,s,c,z,z,z,o)
    else:raise ValueError(axis)
    return torch.stack(values,-1).reshape(-1,3,3)


def quaternion_rotation(q):
    x,y,z,w=q
    return torch.tensor(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
        (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
        (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))),dtype=torch.float32)


def rotvec_matrix(vector):
    angle=vector.norm(dim=-1)
    x,y,z=vector.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    eye=torch.eye(3,device=vector.device,dtype=vector.dtype)[None]
    return eye+torch.sinc(angle/torch.pi)[:,None,None]*skew+ \
        .5*torch.sinc(angle/(2*torch.pi)).square()[:,None,None]*(skew@skew)


class G1KinematicChain(nn.Module):
    def __init__(self,contract):
        super().__init__()
        names=contract['joint_names'];indices={name:i for i,name in enumerate(names)}
        body_ids={contract['root']:0};spec=[];parent_r=[];child_r=[];parent_p=[];child_p=[]
        for joint in contract['joints']:
            if joint['parent'] not in body_ids:raise ValueError('Joint graph must be topologically sorted')
            body_ids[joint['child']]=len(body_ids)
            spec.append((body_ids[joint['parent']],'XYZ'.index(joint['axis']),indices[joint['name']]))
            parent_r.append(quaternion_rotation(joint['parent_quat']))
            child_r.append(quaternion_rotation(joint['child_quat']).T)
            parent_p.append(joint['parent_pos']);child_p.append(joint['child_pos'])
        self.spec=spec
        self.left=body_ids['left_ankle_roll_link'];self.right=body_ids['right_ankle_roll_link']
        self.torso=body_ids['torso_link']
        self.register_buffer('parent_r',torch.stack(parent_r));self.register_buffer('child_r',torch.stack(child_r))
        self.register_buffer('parent_p',torch.tensor(parent_p));self.register_buffer('child_p',torch.tensor(child_p))
        self.register_buffer('nominal',torch.tensor(contract['default_positions']))
        self.register_buffer('sensor_offset',torch.tensor(contract['sensor_in_torso_position']))
        self.register_buffer('sensor_mount',torch.tensor(contract['sensor_in_torso_rotation']))

    def forward(self,proprio):
        q=proprio[:,6:35]+self.nominal
        batch=q.shape[0]
        rotations=[torch.eye(3,device=q.device,dtype=q.dtype)[None].expand(batch,-1,-1)]
        positions=[q.new_zeros(batch,3)]
        for i,(parent,axis,joint) in enumerate(self.spec):
            old_r=rotations[parent]
            new_r=(old_r@self.parent_r[i])@axis_rotation(q[:,joint],axis)@self.child_r[i]
            new_p=positions[parent]+(old_r@self.parent_p[i,:,None])[...,0] \
                -(new_r@self.child_p[i,:,None])[...,0]
            rotations.append(new_r);positions.append(new_p)
        foot_r=torch.stack((rotations[self.left],rotations[self.right]),1)
        foot_p=torch.stack((positions[self.left],positions[self.right]),1)
        sensor_r=rotations[self.torso]@self.sensor_mount
        sensor_p=positions[self.torso]+(rotations[self.torso]@self.sensor_offset[:,None])[...,0]
        return foot_r,foot_p,sensor_r,sensor_p


class KinematicMotionPrior(nn.Module):
    def __init__(self,contract,dt=.02):
        super().__init__()
        self.dt=float(dt)
        if self.dt<=0:raise ValueError('Positive control period required')
        self.chain=G1KinematicChain(contract)
        self.register_buffer('capsule_endpoints',torch.tensor(contract['foot_capsules']['endpoints']))
        self.register_buffer('capsule_radii',torch.tensor(contract['foot_capsules']['radii']))
        for name in ('ready','previous_foot_r','previous_foot_p','previous_gyro','velocity','rotation','position'):
            self.register_buffer(name,None,persistent=False)

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self,dones=None):
        if dones is None:
            for name in ('ready','previous_foot_r','previous_foot_p','previous_gyro','velocity','rotation','position'):
                setattr(self,name,None)
        elif self.ready is not None:
            self.ready[dones.reshape(-1).bool()]=False

    @torch.inference_mode(False)
    @torch.no_grad()
    def step(self,proprio):
        if proprio.ndim!=2 or proprio.shape[1]!=96:raise ValueError('Expected existing[B,96]proprioception')
        foot_r,foot_p,sensor_r,sensor_p=self.chain(proprio)
        b=len(proprio)
        if self.ready is None:
            self.ready=torch.zeros(b,dtype=torch.bool,device=proprio.device)
            self.previous_foot_r=foot_r.clone();self.previous_foot_p=foot_p.clone()
            self.previous_gyro=proprio[:,:3].clone()
            self.velocity=proprio.new_zeros(b,3);self.rotation=torch.eye(3,device=proprio.device)[None].expand(b,-1,-1).clone()
            self.position=proprio.new_zeros(b,3)
        if b!=len(self.ready):raise ValueError('Batch changed; reset motion prior first')
        up_body=F.normalize(-proprio[:,3:6],dim=-1)
        up_foot=(foot_r.transpose(-1,-2)@up_body[:,None,:,None])[...,0]
        candidate=self.capsule_endpoints[None]-self.capsule_radii[None,:,:,None,None]*up_foot[:,:,None,None,:]
        heights=(candidate*up_foot[:,:,None,None,:]).sum(-1)
        bottom=heights<=heights.amin((-1,-2),keepdim=True)+.001
        material=(candidate*bottom[...,None]).sum((2,3))/bottom.sum((2,3))[...,None].clamp_min(1)
        gyro=.5*(self.previous_gyro+proprio[:,:3])
        change=rotvec_matrix(gyro*self.dt)
        prev=(self.previous_foot_r@material[...,None])[...,0]+self.previous_foot_p
        cur=(foot_r@material[...,None])[...,0]+foot_p
        delta=prev-(change[:,None]@cur[...,None])[...,0]
        predicted=self.velocity*self.dt
        distance=(delta-predicted[:,None]).norm(dim=-1)/self.dt
        winner=distance.argmin(-1)
        body=delta.gather(1,winner[:,None,None].expand(-1,1,3))[:,0]
        both=(delta[:,0]-delta[:,1]).norm(dim=-1)/self.dt<.20
        body=torch.where(both[:,None],delta.mean(1),body)
        uncertain=distance.amin(-1)>1.
        body=torch.where(uncertain[:,None],predicted,body)
        next_position=self.position+(self.rotation@body[...,None])[...,0]
        next_rotation=self.rotation@change
        next_velocity=(change.transpose(1,2)@(.8*self.velocity+.2*body/self.dt)[...,None])[...,0]
        initial_velocity=proprio[:,64:67].clone();initial_velocity[:,2]=0.
        self.position=torch.where(self.ready[:,None],next_position,torch.zeros_like(next_position))
        eye=torch.eye(3,device=proprio.device,dtype=proprio.dtype)[None]
        self.rotation=torch.where(self.ready[:,None,None],next_rotation,eye)
        self.velocity=torch.where(self.ready[:,None],next_velocity,initial_velocity)
        self.previous_foot_r.copy_(foot_r);self.previous_foot_p.copy_(foot_p)
        self.previous_gyro.copy_(proprio[:,:3]);self.ready.fill_(True)
        position=self.position+(self.rotation@sensor_p[...,None])[...,0]
        # Horizontal projection of sensor+X defines the level X axis.
        up_sensor=F.normalize((sensor_r.transpose(1,2)@up_body[...,None])[...,0],dim=-1)
        forward=torch.zeros_like(up_sensor);forward[:,0]=1.
        horizontal=forward-up_sensor*up_sensor[:,:1]
        alternate=torch.zeros_like(up_sensor);alternate[:,1]=1.
        alternate=alternate-up_sensor*up_sensor[:,1:2]
        forward=F.normalize(torch.where(horizontal.square().sum(-1,keepdim=True)>1e-8,horizontal,alternate),dim=-1)
        level=torch.stack((forward,torch.linalg.cross(up_sensor,forward,dim=-1),up_sensor),1)
        initial_from_level=self.rotation@sensor_r@level.transpose(1,2)
        return dict(sensor_position=position,initial_from_level=initial_from_level,level=level,
                    support_uncertain=uncertain)
