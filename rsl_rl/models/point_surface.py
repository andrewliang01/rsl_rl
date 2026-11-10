"""Deployment-only point geometry and a small surface-completion baseline.

Inputs are native sensor XYZ returns and the existing 96-value proprioception.
No world pose, map, terrain labels or simulator imports belong in this module.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


G1_SENSOR_IN_TORSO = (
    (0.998693622, 0., -0.0510984318),
    (0., -1., 0.),
    (-0.0510984318, 0., -0.998693622),
)


def axis_rotation(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Batched right-handed rotation; angles in radians."""
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device).expand(len(angle), -1, -1)
    unit = eye[:, axis]
    zero = torch.zeros_like(angle)
    x, y, z = unit.unbind(-1)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    return eye + angle.sin()[:, None, None]*skew + (1-angle.cos())[:, None, None]*(skew@skew)


def sensor_to_level(proprio: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
    """Level about gravity, keeping the horizontal projection of sensor +X.

    Layout: gyro3, projected_gravity3, relative_joint_pos29, joint_vel29,
    command3, previous_action29. Waist joints 12:15 have zero nominal angles.
    Fixed mounting orientation is from the user's current capsule-foot USD.
    """
    if proprio.ndim != 2 or proprio.shape[1] != 96:
        raise ValueError('Expected the existing [B,96] G1 proprioception layout.')
    waist = proprio[:, 18:21]
    body_sensor = (axis_rotation(waist[:, 0], 2) @ axis_rotation(waist[:, 1], 0)
                   @ axis_rotation(waist[:, 2], 1) @ calibration)
    up_body = F.normalize(-proprio[:, 3:6], dim=-1)
    up = F.normalize((body_sensor.transpose(1, 2) @ up_body[..., None])[..., 0], dim=-1)
    forward = torch.zeros_like(up)
    forward[:, 0] = 1.
    horizontal = forward-up*up[:, :1]
    alternate = torch.zeros_like(up)
    alternate[:, 1] = 1.
    alternate = alternate-up*up[:, 1:2]
    horizontal = torch.where(horizontal.square().sum(-1, keepdim=True)>1e-8, horizontal, alternate)
    forward = F.normalize(horizontal, dim=-1)
    left = torch.linalg.cross(up, forward, dim=-1)
    return torch.stack((forward, left, up), 1)


class PointSurfaceProjector(nn.Module):
    """Metric measured height and validity, with missing cells left unknown."""

    def __init__(self, grid_size=32, extent_m=2.4, calibration=G1_SENSOR_IN_TORSO, *, projection='max'):
        super().__init__()
        if type(grid_size) is not int or grid_size < 2 or not math.isfinite(extent_m) or extent_m<=0:
            raise ValueError('Positive extent and grid size >=2 required.')
        self.grid_size, self.extent_m = grid_size, float(extent_m)
        if projection not in ('max', 'nearest'):
            raise ValueError('Point projection must be max or nearest.')
        self.projection = projection
        matrix = torch.tensor(calibration, dtype=torch.float32)
        if matrix.shape!=(3,3) or not torch.allclose(matrix.T@matrix,torch.eye(3),atol=1e-5):
            raise ValueError('Calibration must be a 3x3 rotation.')
        self.register_buffer('calibration',matrix)

    def level(self, native_points, proprio):
        if native_points.ndim!=3 or native_points.shape[2]!=3 or len(native_points)!=len(proprio):
            raise ValueError('Native points must be [B,R,3] aligned with proprioception.')
        rotation = sensor_to_level(proprio, self.calibration.to(proprio))
        return native_points @ rotation.transpose(1,2)

    def forward(self, native_points, proprio):
        points = self.level(native_points,proprio)
        valid = (torch.isfinite(points).all(-1) & (native_points.square().sum(-1)>.05**2)
                 & (points[:,:,:2]>=-self.extent_m/2).all(-1)
                 & (points[:,:,:2]<self.extent_m/2).all(-1)
                 & (points[:,:,2]>=-3.) & (points[:,:,2]<=.5))
        safe = torch.where(valid[...,None],points,torch.zeros_like(points))
        cell = ((safe[:,:,:2]/self.extent_m+.5)*self.grid_size).floor().long().clamp(0,self.grid_size-1)
        index = cell[:,:,1]*self.grid_size+cell[:,:,0]
        if self.projection == 'nearest':
            rays = points.shape[1]
            if rays == 0:
                return points.new_zeros(len(points),2,self.grid_size,self.grid_size)
            centers = (cell.to(points.dtype)+.5)*self.extent_m/self.grid_size-self.extent_m/2
            distance = (safe[:,:,:2]-centers).square().sum(-1)
            closest = points.new_full((len(points),self.grid_size**2),torch.inf)
            closest.scatter_reduce_(1,index,torch.where(valid,distance,torch.inf),reduce='amin')
            candidate = valid & (distance==closest.gather(1,index))
            ray_index = torch.arange(rays,device=points.device)[None].expand(len(points),-1)
            selected = torch.full_like(closest,rays,dtype=torch.long)
            selected.scatter_reduce_(1,index,torch.where(candidate,ray_index,rays),reduce='amin')
            visible = selected<rays
            height = points[:,:,2].gather(1,selected.clamp_max(rays-1))
            shape = (-1,self.grid_size,self.grid_size)
            return torch.stack((torch.where(visible,height,0.).reshape(shape),
                                visible.to(height.dtype).reshape(shape)),1)
        height = points.new_full((len(points),self.grid_size**2),-torch.inf)
        height.scatter_reduce_(1,index,torch.where(valid,points[:,:,2],-torch.inf),reduce='amax')
        height = height.reshape(-1,self.grid_size,self.grid_size)
        visible = torch.isfinite(height)
        return torch.stack((torch.where(visible,height,0.),visible.to(height.dtype)),1)


class SurfaceCompleter(nn.Module):
    """Small baseline: retain measurements exactly and predict missing heights.

    The mean measured height supplies only a local coordinate origin. It is
    recomputed from input points and is never the ground-truth mean plane.
    Log error scale is learned separately; it is not a safety probability.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2,channels,3,padding=1),nn.SiLU(),
            nn.Conv2d(channels,2*channels,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(2*channels,2*channels,3,stride=2,padding=1),nn.SiLU())
        self.decoder = nn.Sequential(nn.Conv2d(2*channels,channels,3,padding=1),nn.SiLU(),
            nn.Conv2d(channels,2,3,padding=1))
        with torch.no_grad():
            self.decoder[-1].bias[1]=-3.

    def forward(self,surface):
        height,valid = surface[:,:1],surface[:,1:2]
        valid = (valid>.5).to(height.dtype)
        count = valid.sum((2,3),keepdim=True)
        mean = (height*valid).sum((2,3),keepdim=True)/count.clamp_min(1)
        mean = torch.where(count>0,mean,torch.full_like(mean,-1.2))
        encoded = self.encoder(torch.cat(((height-mean)*valid,valid),1))
        decoded = self.decoder(F.interpolate(encoded,size=height.shape[-2:],mode='bilinear',align_corners=False))
        completed = mean+decoded[:,:1]
        return dict(height=torch.where(valid.bool(),height,completed),
                    inferred_height=completed,log_error_scale=decoded[:,1:2].clamp(-6.,0.),observed=valid)


class DilatedSurfaceCompleter(SurfaceCompleter):
    """Full-resolution control: same task, no angular attention or downsampling."""

    def __init__(self,channels=12):
        nn.Module.__init__(self)
        layers=[]
        incoming=2
        for dilation in (1,2,4,8):
            layers.extend((nn.Conv2d(incoming,channels,3,padding=dilation,dilation=dilation),nn.SiLU()))
            incoming=channels
        self.encoder=nn.Sequential(*layers)
        self.decoder=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,2,1))
        with torch.no_grad():
            self.decoder[-1].bias[1]=-3.
