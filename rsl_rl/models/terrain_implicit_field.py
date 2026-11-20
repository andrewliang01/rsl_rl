"""G1 雷达条件下在连续坐标查询占据概率的网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TerrainImplicitField(nn.Module):
    """按扫描编码点集，用三个正交特征面保留局部三维信息。"""

    plane_axes = ((0, 1), (0, 2), (1, 2))  # xy、xz、yz，坐标依次对应列、行。

    def __init__(self, bounds_m, history_frames: int, plane_cells: int = 64, channels: int = 24):
        '''
        构造逐点编码器、时序特征面与连续占据头。
        bounds_m 给出三轴裁剪盒的下界和上界（m），其余参数为固定扫描数、每轴格数和通道数。
        '''
        super().__init__()
        bounds = torch.as_tensor(bounds_m, dtype=torch.float32)
        if bounds.shape != (2, 3) or not bounds.isfinite().all() or not (bounds[1] > bounds[0]).all():
            raise ValueError("bounds_m 必须提供有限且严格递增的三轴上下界")
        if history_frames < 1 or plane_cells < 2 or channels < 1:
            raise ValueError("请提供正扫描数、正通道数和至少两个特征格")
        self.register_buffer("lower_m", bounds[0].clone())
        self.register_buffer("span_m", bounds[1] - bounds[0])
        self.history_frames = history_frames
        self.plane_cells = plane_cells
        self.point_encoder = nn.Sequential(nn.Linear(4, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.plane_encoder = nn.Sequential(
            nn.Conv2d(history_frames * (channels + 1), channels, 1),  # 固定从旧到新融合扫描。
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),  # 为运动造成的跨格偏移保留邻域。
        )
        self.occupancy_head = nn.Sequential(
            nn.Linear(channels * len(self.plane_axes) + 3, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, 2),  # 占据与 H 证据覆盖分别监督。
        )

    def encode(self, points_m: torch.Tensor, valid: torch.Tensor, age_s: torch.Tensor) -> torch.Tensor:
        '''
        把 B×T×P×3 的回波及 B×T×P 的有效掩码、点龄期（秒）编码成 B×3×C×M×M。
        扫描按旧到新排列，保留采集时坐标，只允许固定标定，缺帧用无效掩码填充。
        '''
        if points_m.ndim != 4 or points_m.shape[1] != self.history_frames or points_m.shape[-1] != 3:
            raise ValueError("points_m 必须是 B×T×P×3 且 T 等于配置窗长")
        if valid.shape != points_m.shape[:-1] or age_s.shape != valid.shape or valid.dtype != torch.bool:
            raise ValueError("掩码需为布尔值，且与点龄期同为 B×T×P")
        batch, scans, rays, _ = points_m.shape
        cells = self.plane_cells
        xyz = (points_m - self.lower_m) * (2.0 / self.span_m) - 1.0  # 保留各帧原坐标。
        valid = valid & xyz.isfinite().all(-1) & (xyz.abs() <= 1.0).all(-1) & age_s.isfinite() & (age_s >= 0)
        xyz = torch.where(valid[..., None], xyz, 0.0)
        age = torch.where(valid, age_s, 0.0)  # 遮去填充值，避免无效射线传播 NaN。
        features = self.point_encoder(torch.cat((xyz, age[..., None]), dim=-1))
        features = torch.where(valid[..., None], features, 0.0).flatten(0, 1).transpose(1, 2)
        channels = features.shape[1]
        indices = (((xyz + 1.0) * (cells / 2.0)).long().clamp(0, cells - 1)).flatten(0, 1)
        ones = valid.reshape(batch * scans, 1, rays).to(features.dtype)
        planes = []
        for col, row in self.plane_axes:
            index = (indices[..., col] + cells * indices[..., row]).unsqueeze(1)
            count = features.new_zeros(batch * scans, 1, cells**2).scatter_add_(2, index, ones)
            sums = features.new_zeros(batch * scans, channels, cells**2)
            sums.scatter_add_(2, index.expand_as(features), features)
            plane = torch.cat((sums / count.clamp_min(1), (count > 0).to(features.dtype)), dim=1)
            planes.append(plane.reshape(batch, scans * (channels + 1), cells, cells))
        encoded = self.plane_encoder(torch.stack(planes, dim=1).flatten(0, 1))
        return encoded.reshape(batch, len(self.plane_axes), channels, cells, cells)

    def query_logits(self, planes: torch.Tensor, points_m: torch.Tensor) -> torch.Tensor:
        '''
        在 B×Q×3 的当前查询坐标（m）双线性读取三个特征面。
        输出 B×Q×2，依次为占据和 H 证据覆盖 logit。
        各坐标应在配置盒内。planes 可复用，所有查询共享同一个占据头。
        '''
        if points_m.ndim != 3 or points_m.shape[0] != planes.shape[0] or points_m.shape[-1] != 3:
            raise ValueError("查询坐标必须是与特征面同批的 B×Q×3")
        xyz = ((points_m - self.lower_m) * (2.0 / self.span_m) - 1.0).to(planes.dtype)
        features = [xyz]
        for index, axes in enumerate(self.plane_axes):
            grid = xyz[..., axes].unsqueeze(2)  # align_corners=False 与编码格心对齐。
            sampled = F.grid_sample(planes[:, index], grid, padding_mode="border", align_corners=False)
            features.append(sampled.squeeze(-1).transpose(1, 2))
        return self.occupancy_head(torch.cat(features, dim=-1))

    def query(self, planes: torch.Tensor, points_m: torch.Tensor) -> torch.Tensor:
        '''
        从已编码扫描窗预测 B×Q 的占据 logits，供几何读出或实际查询。
        覆盖估计通过 query_logits 的第二个输出读取。
        '''
        return self.query_logits(planes, points_m)[..., 0]

    def forward(self, points_m: torch.Tensor, valid: torch.Tensor, age_s: torch.Tensor, queries_m: torch.Tensor):
        '''
        从一个因果扫描窗拟合指定三维点的占据，输出 B×Q 的 logits。
        反复查询同一窗时分别调用 encode 和 query，省去重复编码。
        '''
        return self.query(self.encode(points_m, valid, age_s), queries_m)
