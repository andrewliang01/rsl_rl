"""A bounded rolling map of actual past returns in the current level frame."""
import torch
from torch import nn


class RollingPointMemory(nn.Module):
    def __init__(self,grid_size=32,extent_m=2.4,max_age_s=2.4):
        super().__init__()
        self.size=int(grid_size);self.extent=float(extent_m);self.max_age=float(max_age_s)
        if self.size<2 or self.extent<=0 or self.max_age<=0:raise ValueError('Invalid memory geometry')
        axis=(torch.arange(self.size)+.5)*self.extent/self.size-self.extent/2
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('centers',torch.stack((xx,yy),-1).reshape(-1,2))
        for name in ('anchors','valid','age','variance'):
            self.register_buffer(name,None,persistent=False)

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self,dones=None):
        if dones is None:
            self.anchors=self.valid=self.age=self.variance=None
        elif self.valid is not None:
            ids=dones.reshape(-1).bool()
            self.valid[ids]=False;self.age[ids]=0;self.variance[ids]=0

    @torch.inference_mode(False)
    @torch.no_grad()
    def initialize(self,batch,like):
        k=self.size**2
        if self.anchors is not None:
            if len(self.anchors)!=batch:raise ValueError('Batch changed; reset memory first')
            return
        self.anchors=like.new_zeros(batch,k,3)
        self.valid=torch.zeros(batch,k,dtype=torch.bool,device=like.device)
        self.age=like.new_zeros(batch,k)
        self.variance=like.new_zeros(batch,k)

    @torch.inference_mode(False)
    @torch.no_grad()
    def update(self,current_anchors,current_valid,rotation,translation,elapsed_s,transport_sigma,indices=None):
        """rotation,translation map new level coordinates into old coordinates.

        All points remain actual measured anchors, transformed by estimated
        motion. Nothing is replaced by a cell-center pseudo measurement.
        Returned history is separate from current measurements, including when
        the two cover the same cell. New current returns overwrite stored past.
        """
        b=len(current_anchors);k=self.size**2
        current_anchors=current_anchors.reshape(b,k,3)
        current_valid=current_valid.reshape(b,k).bool()
        if self.anchors is None:
            if indices is not None:raise ValueError('Initialize the full environment batch before subset updates')
            self.initialize(b,current_anchors)
        if indices is None:
            if b!=len(self.anchors):raise ValueError('Batch changed; reset memory first')
            old_anchors,old_valid,old_age,old_variance=self.anchors,self.valid,self.age,self.variance
        else:
            old_anchors=self.anchors.index_select(0,indices)
            old_valid=self.valid.index_select(0,indices)
            old_age=self.age.index_select(0,indices)
            old_variance=self.variance.index_select(0,indices)
        moved=(old_anchors-translation[:,None])@rotation
        age=old_age+elapsed_s.reshape(b,1)
        variance=old_variance+transport_sigma.reshape(b,1).square()
        eligible=old_valid&(age<=self.max_age)&torch.isfinite(moved).all(-1)
        eligible&=(moved[...,:2]>=-self.extent/2).all(-1)&(moved[...,:2]<self.extent/2).all(-1)
        eligible&=(moved[...,2]>=-3.)&(moved[...,2]<=.5)
        safe=torch.where(eligible[...,None],moved,0.)
        cells=((safe[...,:2]/self.extent+.5)*self.size).floor().long().clamp(0,self.size-1)
        index=cells[...,1]*self.size+cells[...,0]
        centers=self.centers[index]
        distance=(safe[...,:2]-centers).square().sum(-1)
        best=current_anchors.new_full((b,k),torch.inf)
        best.scatter_reduce_(1,index,torch.where(eligible,distance,torch.inf),reduce='amin')
        candidate=eligible&(distance==best.gather(1,index))
        # Equal-position returns use the younger sample, without any truth test.
        youngest=current_anchors.new_full((b,k),torch.inf)
        youngest.scatter_reduce_(1,index,torch.where(candidate,age,torch.inf),reduce='amin')
        candidate&=age==youngest.gather(1,index)
        order=torch.arange(k,device=current_anchors.device)[None].expand(b,-1)
        selected=torch.full((b,k),k,device=current_anchors.device,dtype=torch.long)
        selected.scatter_reduce_(1,index,torch.where(candidate,order,k),reduce='amin')
        valid=selected<k;selected=selected.clamp_max(k-1)
        anchors=moved.gather(1,selected[...,None].expand(-1,-1,3))
        anchors=torch.where(valid[...,None],anchors,0.)
        history_age=torch.where(valid,age.gather(1,selected),0.)
        history_variance=torch.where(valid,variance.gather(1,selected),0.)
        history=dict(anchors=anchors,valid=valid,age=history_age,error_hint=history_variance.sqrt())
        new_anchors=torch.where(current_valid[...,None],current_anchors,anchors)
        new_valid=current_valid|valid
        new_age=torch.where(current_valid,0.,history_age)
        new_variance=torch.where(current_valid,0.,history_variance)
        if indices is None:
            self.anchors,self.valid,self.age,self.variance=new_anchors,new_valid,new_age,new_variance
        else:
            self.anchors.index_copy_(0,indices,new_anchors);self.valid.index_copy_(0,indices,new_valid)
            self.age.index_copy_(0,indices,new_age);self.variance.index_copy_(0,indices,new_variance)
        return history
