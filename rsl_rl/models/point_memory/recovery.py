"""Small query-height recovery with distinct current and transported evidence.

All inputs derive from XYZ, existingproprio/clock and staticcalibration.
Missing-cell predictions target grid query centers. With correct_current=True,
occupied-cell anchors can also be corrected; with False their raw heights are
preserved. An off-center measured anchor is not exact query-center truth.
"""
import torch
from torch import nn


class EvidenceSurfaceCompleter(nn.Module):
    def __init__(self,channels=12,max_age_s=2.4,history_enabled=True,correct_current=True):
        super().__init__()
        self.history_enabled=bool(history_enabled);self.correct_current=bool(correct_current)
        self.max_age=float(max_age_s)
        layers=[];incoming=11
        for dilation in (1,2,4,8):
            layers.extend((nn.Conv2d(incoming,channels,3,padding=dilation,dilation=dilation),nn.SiLU()))
            incoming=channels
        self.encoder=nn.Sequential(*layers)
        self.decoder=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,4,1))
        with torch.no_grad():
            self.decoder[-1].bias[1]=-3.
            self.decoder[-1].weight[3].zero_();self.decoder[-1].bias[3]=0.

    def forward(self,evidence):
        if evidence.ndim!=4 or evidence.shape[1]!=11:raise ValueError('Expected[B,11,H,W]point evidence')
        cur_h=evidence[:,:1];cur_v=(evidence[:,1:2]>.5).to(evidence.dtype)
        hist_h=evidence[:,5:6];hist_v=(evidence[:,6:7]>.5).to(evidence.dtype)
        if not self.history_enabled:
            # The trained current-only control cannot see any historical value,
            # mask, age, transport error or point-offset cue.
            hist_v=torch.zeros_like(hist_v);hist_h=torch.zeros_like(hist_h)
        count=cur_v.sum((2,3),keepdim=True)
        current_mean=(cur_h*cur_v).sum((2,3),keepdim=True)/count.clamp_min(1)
        hist_count=hist_v.sum((2,3),keepdim=True)
        hist_mean=(hist_h*hist_v).sum((2,3),keepdim=True)/hist_count.clamp_min(1)
        fallback=torch.full_like(current_mean,-1.2)
        mean=torch.where(count>0,current_mean,torch.where(hist_count>0,hist_mean,fallback))
        current=torch.cat(((cur_h-mean)*cur_v,cur_v,evidence[:,2:5]*cur_v),1)
        historical=torch.cat(((hist_h-mean)*hist_v,hist_v,evidence[:,7:9]*hist_v,
            (evidence[:,9:10]/self.max_age)*hist_v,(evidence[:,10:11]/.05)*hist_v),1)
        if not self.history_enabled:historical=torch.zeros_like(historical)
        decoded=self.decoder(self.encoder(torch.cat((current,historical),1)))
        inferred=mean+decoded[:,:1]
        gate=decoded[:,2:3].sigmoid()*hist_v
        missing=gate*hist_h+(1-gate)*inferred
        correction=.30*decoded[:,3:4].tanh() if self.correct_current else torch.zeros_like(cur_h)
        height=torch.where(cur_v.bool(),cur_h+correction,missing)
        return dict(height=height,inferred_height=inferred,log_error_scale=decoded[:,1:2].clamp(-6.,0.),
            observed=cur_v,history_observed=hist_v,history_gate=gate,current_correction=correction)
