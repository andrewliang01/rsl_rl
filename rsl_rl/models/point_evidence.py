"""Point-coordinate evidence retained within each terrain raster cell.

No simulator state, labels or extra LiDAR attributes enter this reduction.
Max/min/nearest anchors are actual measured points, not cell-center positions.
"""
import torch
from torch import nn
from .point_surface import PointSurfaceProjector


class PointEvidenceReducer(nn.Module):
    def __init__(self,grid_size=32,extent_m=2.4):
        super().__init__()
        self.projector=PointSurfaceProjector(grid_size,extent_m)
        self.size,self.extent=grid_size,float(extent_m)

    @torch.no_grad()
    def forward(self,native,proprio):
        p=self.projector.level(native,proprio)
        valid=(torch.isfinite(p).all(-1)&(native.square().sum(-1)>.05**2)
               &(p[...,:2]>=-self.extent/2).all(-1)&(p[...,:2]<self.extent/2).all(-1)
               &(p[...,2]>=-3.)&(p[...,2]<=.5))
        safe=torch.where(valid[...,None],p,0.)
        cells=((safe[...,:2]/self.extent+.5)*self.size).floor().long().clamp(0,self.size-1)
        index=cells[...,1]*self.size+cells[...,0]
        batch,rays=valid.shape
        maximum=p.new_full((batch,self.size**2),-torch.inf)
        minimum=p.new_full((batch,self.size**2),torch.inf)
        maximum.scatter_reduce_(1,index,torch.where(valid,p[...,2],-torch.inf),reduce='amax')
        minimum.scatter_reduce_(1,index,torch.where(valid,p[...,2],torch.inf),reduce='amin')
        count=torch.zeros(batch,self.size**2,device=p.device,dtype=torch.int32)
        count.scatter_add_(1,index,valid.to(torch.int32))
        observed=count>0
        center=(cells.to(p.dtype)+.5)*self.extent/self.size-self.extent/2
        squared=(safe[...,:2]-center).square().sum(-1)
        ray_index=torch.arange(rays,device=p.device)[None].expand(batch,-1)
        anchors=[]
        for candidate in [valid&(p[...,2]==maximum.gather(1,index)),
                          valid&(p[...,2]==minimum.gather(1,index)),valid]:
            closest=p.new_full((batch,self.size**2),torch.inf)
            closest.scatter_reduce_(1,index,torch.where(candidate,squared,torch.inf),reduce='amin')
            chosen=candidate&(squared==closest.gather(1,index))
            selected=torch.full((batch,self.size**2),rays,device=p.device,dtype=torch.long)
            selected.scatter_reduce_(1,index,torch.where(chosen,ray_index,rays),reduce='amin')
            points=p.gather(1,selected.clamp_max(rays-1)[...,None].expand(-1,-1,3))
            anchors.append(torch.where(observed[...,None],points,0.).reshape(batch,self.size,self.size,3))
        anchors=torch.stack(anchors,1)  # max, min, nearest to cell center
        maximum=torch.where(observed,maximum,0.).reshape(batch,self.size,self.size)
        minimum=torch.where(observed,minimum,0.).reshape_as(maximum)
        return dict(anchors=anchors,height_max=maximum,height_min=minimum,
                    count=count.reshape_as(maximum),observed=observed.reshape_as(maximum))
