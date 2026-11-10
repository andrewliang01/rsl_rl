"""Bounded spherical-image point registration prototype; actual XYZ only.

No simulator imports or pose labels. Image cells retain actual returns.
This is a quality/speed experiment, not an installed online memory module.
"""
import math
import torch
from torch.nn import functional as F

ROWS, COLS = 64, 256
LOW, HIGH = math.radians(-7), math.radians(52)


def angular_cells(points):
    azimuth = torch.atan2(points[..., 1], points[..., 0])
    elevation = torch.atan2(points[..., 2], points[..., :2].norm(dim=-1))
    col = ((azimuth + math.pi) * (COLS / (2*math.pi))).floor().long() % COLS
    row = ((elevation-LOW) * (ROWS/(HIGH-LOW))).floor().long()
    valid = torch.isfinite(points).all(-1) & (row >= 0) & (row < ROWS)
    return row.clamp(0, ROWS-1), col, valid


@torch.no_grad()
def plane_image(native, level):
    b, rays, _ = native.shape
    row, col, valid = angular_cells(native)
    distance = native.square().sum(-1)
    valid &= (distance > .05**2) & (distance < 8.**2)
    cell = row*COLS+col
    best = native.new_full((b, ROWS*COLS), torch.inf)
    best.scatter_reduce_(1, cell, torch.where(valid, distance, torch.inf), reduce='amin')
    is_best = valid & (distance == best.gather(1, cell))
    indices = torch.arange(rays, device=native.device)[None].expand(b, -1)
    selected = torch.full((b, ROWS*COLS), rays, dtype=torch.long, device=native.device)
    selected.scatter_reduce_(1, cell, torch.where(is_best, indices, rays), reduce='amin')
    observed = selected < rays
    if not rays:
        points = native.new_zeros(b, ROWS, COLS, 3)
    else:
        actual = native.gather(1, selected.clamp_max(rays-1)[..., None].expand(-1, -1, 3))
        actual = torch.where(observed[..., None], actual, 0.)
        points = (actual @ level.transpose(1, 2)).reshape(b, ROWS, COLS, 3)
    observed = observed.reshape(b, ROWS, COLS)
    left, right = points.roll(1, 2), points.roll(-1, 2)
    above, below = points.roll(1, 1), points.roll(-1, 1)
    cross = torch.linalg.cross(right-left, below-above, dim=-1)
    normal = F.normalize(cross, dim=-1)
    valid_plane = observed & observed.roll(1, 2) & observed.roll(-1, 2)
    valid_plane &= observed.roll(1, 1) & observed.roll(-1, 1)
    valid_plane[:, 0] = False
    valid_plane[:, -1] = False
    valid_plane &= cross.norm(dim=-1) > 1e-6
    # Reject a finite-difference normal that straddles different surfaces.
    planar_error = torch.stack([((p-points)*normal).sum(-1).abs()
        for p in (left, right, above, below)], -1).amax(-1)
    valid_plane &= planar_error < .025
    return points.reshape(b, -1, 3), normal.reshape(b, -1, 3), valid_plane.flatten(1)


def rotvec_matrix(vector):
    theta = vector.norm(dim=-1)
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)[None]
    return eye + torch.sinc(theta/torch.pi)[:,None,None]*skew + \
        .5*torch.sinc(theta/(2*torch.pi)).square()[:,None,None]*(skew@skew)


@torch.no_grad()
def register(previous_native, current_native, previous_level, current_level,
             prior_rotation, prior_translation, *, iterations=4, source_limit=512):
    b, rays, _ = current_native.shape
    points, normals, planes = plane_image(previous_native, previous_level)
    if rays:
        indices = torch.linspace(0, rays-1, min(rays,source_limit), device=current_native.device).long()
        native = current_native[:, indices]
    else:
        native = current_native
    source_valid = torch.isfinite(native).all(-1) & (native.norm(dim=-1)>.05) & (native.norm(dim=-1)<8.)
    source = torch.where(source_valid[...,None],native,0.) @ current_level.transpose(1,2)
    rotation, translation = prior_rotation.clone(), prior_translation.clone()
    batch = torch.arange(b, device=source.device)[:,None,None]
    dr, dc = torch.meshgrid(torch.arange(-1,2,device=source.device),
                           torch.arange(-1,2,device=source.device),indexing='ij')
    dr, dc = dr.flatten(), dc.flatten()
    info = {}
    for _ in range(iterations):
        moved = source @ rotation.transpose(1,2)+translation[:,None]
        old_sensor = moved @ previous_level
        row, col, inside = angular_cells(old_sensor)
        rr, cc = row[...,None]+dr, (col[...,None]+dc)%COLS
        candidates = rr.clamp(0,ROWS-1)*COLS+cc
        target = points[batch,candidates]
        normal = normals[batch,candidates]
        eligible = planes[batch,candidates] & (rr>=0) & (rr<ROWS)
        distance = (target-moved[:,:,None]).square().sum(-1)
        closest = torch.where(eligible,distance,torch.inf).argmin(-1)
        pick = closest[...,None,None].expand(-1,-1,1,3)
        target = target.gather(2,pick)[:,:,0]
        normal = normal.gather(2,pick)[:,:,0]
        nearest = torch.where(eligible,distance,torch.inf).gather(-1,closest[...,None])[...,0]
        residual = (normal*(moved-target)).sum(-1)
        good = source_valid & inside & (nearest<.25**2) & (residual.abs()<.12)
        weight = good.to(source.dtype)*(.03/residual.abs().clamp_min(1e-8)).clamp_max(1.)
        jacobian = torch.cat((normal,torch.linalg.cross(moved,normal,dim=-1)),-1)
        total = weight.sum(-1).clamp_min(1.)
        hessian = (jacobian.transpose(1,2)*weight[:,None])@jacobian/total[:,None,None]
        gradient = (jacobian*weight[...,None]*residual[...,None]).sum(1)/total[:,None]
        symmetric = (hessian.double()+hessian.transpose(1,2).double())*.5
        values, vectors = torch.linalg.eigh(symmetric)
        observed = values > torch.maximum(torch.full_like(values,1e-4), values[:,-1:]*1e-4)
        usable = good.sum(-1)>=12
        inverse = torch.where(observed & usable[:,None],values.clamp_min(1e-12).reciprocal(),0.)
        correction = -(vectors@(inverse*(vectors.transpose(1,2)@gradient.double()[...,None])[...,0])[...,None])[...,0]
        correction = correction.to(source.dtype)
        correction[:,:3] *= (.10/correction[:,:3].norm(dim=-1).clamp_min(1e-12)).clamp_max(1.)[:,None]
        correction[:,3:] *= (.05/correction[:,3:].norm(dim=-1).clamp_min(1e-12)).clamp_max(1.)[:,None]
        change = rotvec_matrix(correction[:,3:])
        rotation = change@rotation
        translation = (change@translation[...,None])[...,0]+correction[:,:3]
        info = dict(inliers=good.sum(-1),planes=planes.sum(-1),
            residual_mae_m=(residual.abs()*good).sum(-1)/good.sum(-1).clamp_min(1),
            observable_dimensions=(observed & usable[:,None]).sum(-1),information_eigenvalues=values)
    return rotation, translation, info
