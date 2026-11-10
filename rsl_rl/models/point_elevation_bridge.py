"""Native-point reconstruction to a previously learned geometric controller.

This module has no simulator imports or privileged observations. The only
forward inputs are the existing96proprio and a point-reconstructed32x32height.
No world pose/odometry is used; the scan is held for its original0.1s period.
"""
from pathlib import Path
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict
from .point_surface import G1_SENSOR_IN_TORSO,sensor_to_level
from .prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel


class PointElevationBridge(nn.Module):
    def __init__(self,*,extent_m=2.4,sensor_position=(.0002835,.00003,.428434),
                 reference_position=(.0002835,.00003,.41618),
                 reference_quaternion=(0.,.020069072,0.,.9997986)):
        super().__init__();self.extent_m=extent_m
        self.register_buffer('calibration',torch.tensor(G1_SENSOR_IN_TORSO,dtype=torch.float32))
        delta=torch.tensor(reference_position,dtype=torch.float32)-torch.tensor(sensor_position,dtype=torch.float32)
        self.register_buffer('offset_delta',delta)
        x,y,z,w=reference_quaternion;norm=x*x+y*y+z*z+w*w;v=2/norm
        rot=((1-v*(y*y+z*z),v*(x*y-z*w),v*(x*z+y*w)),
             (v*(x*y+z*w),1-v*(x*x+z*z),v*(y*z-x*w)),
             (v*(x*z-y*w),v*(y*z+x*w),1-v*(x*x+y*y)))
        self.register_buffer('reference_rotation',torch.tensor(rot,dtype=torch.float32))
        # Same yx ordering as the original28x20 reference ray pattern.
        gx,gy=torch.meshgrid(torch.linspace(-.675,.675,28),torch.linspace(-.475,.475,20),indexing='ij')
        self.register_buffer('query_x',gx);self.register_buffer('query_y',gy)

    def queries(self,proprio):
        rotation=sensor_to_level(proprio,self.calibration.to(proprio))
        torso_in_level=rotation@self.calibration.to(proprio).T
        reference_in_level=torso_in_level@self.reference_rotation.to(proprio)
        forward=F.normalize(reference_in_level[:,:2,0],dim=-1)
        left=torch.stack((-forward[:,1],forward[:,0]),-1)
        origin=(torso_in_level@self.offset_delta.to(proprio)[None,:,None])[...,0]
        xy=(origin[:,None,None,:2]+self.query_x[None,:,:,None]*forward[:,None,None,:]
             +self.query_y[None,:,:,None]*left[:,None,None,:])
        return xy,origin[:,2]

    def forward(self,proprio,height):
        if height.ndim!=4 or tuple(height.shape[1:])!=(1,32,32):
            raise ValueError('Point bridge requires reconstructed[B,1,32,32]height.')
        xy,origin_z=self.queries(proprio)
        sampled=F.grid_sample(height,xy*(2/self.extent_m),mode='bilinear',padding_mode='border',align_corners=False)
        # Original reference uses positive sensorZ-groundZ, opposite point H.
        return origin_z[:,None,None,None]-sampled


class ReconstructedElevationController(nn.Module):
    """Frozen controller probe: five channels repeat the latest point map.

    Repetition is explicit, not a claim of historical measurements. PPO use
    would require replaying the exact cached inputs; this class is a probe.
    """
    def __init__(self,checkpoint,contract,device='cpu'):
        super().__init__()
        cfg=copy.deepcopy(contract['actor_cfg']);cfg.pop('class_name')
        if cfg['elevation_history_length']!=5 or list(cfg['vision_spatial_size'])!=[28,20]:
            raise ValueError('Unsupported geometric controller contract.')
        self.bridge=PointElevationBridge().to(device)
        example=TensorDict({'policy':torch.zeros(1,96,device=device),
            'height_scan_policy':torch.zeros(1,5,28,20,device=device)},batch_size=[1])
        self.actor=PropMLPElevationFusionModel(example,{'actor':['policy','height_scan_policy']},'actor',29,**cfg).to(device)
        state=torch.load(Path(checkpoint),map_location=device,weights_only=False)
        self.actor.load_state_dict(state['actor_state_dict'],strict=True)
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self,proprio,height):
        depth=self.bridge(proprio,height).expand(-1,5,-1,-1)
        inputs=TensorDict({'policy':proprio,'height_scan_policy':depth},batch_size=[len(proprio)])
        return self.actor(inputs)
