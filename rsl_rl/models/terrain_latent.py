# SPDX-License-Identifier: BSD-3-Clause

"""G1 单雷达短窗编码器与足端几何读出头。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TerrainLatentEncoder(nn.Module):
    """逐帧提取原始点特征，按观测顺序融合成一个地形向量。"""

    def __init__(self, latent_dim: int):
        '''
        构建共享逐点编码、时序融合和输出瓶颈。
        参数 latent_dim 是地形向量维度，初始化无返回值。
        '''
        super().__init__()
        if not 1 <= latent_dim <= 256:
            raise ValueError("latent_dim must be between 1 and 256")
        self.point_encoder = nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 64), nn.SiLU())
        self.temporal = nn.GRU(130, 256, batch_first=True)  # 最大池化、均值、扫描龄期和命中标记
        self.bottleneck = nn.Linear(256, latent_dim)

    def forward(
        self, points: torch.Tensor, hit: torch.Tensor, scan_age_s: torch.Tensor
    ) -> torch.Tensor:
        '''
        将按时间从旧到新排列的雷达短窗编码为 [B, D]。
        points 为 [B, T, P, 4]，各帧保留本帧雷达系 XYZ（m）和点相对帧末的时间（s）。
        hit 为 [B, T, P] 命中掩码，scan_age_s 为 [B, T] 帧末相对查询时刻的龄期（s）。
        '''
        xyz_time = points.masked_fill(~hit[..., None], 0.0)  # 避免无效回波将 NaN 传入编码
        scaled = torch.cat((xyz_time[..., :3], 10.0 * xyz_time[..., 3:]), dim=-1)  # 点时间以 0.1 s 为单位
        local = self.point_encoder(scaled)
        any_hit = hit.any(dim=-1, keepdim=True)
        strongest = local.masked_fill(~hit[..., None], -torch.inf).amax(dim=-2)
        strongest = strongest.masked_fill(~any_hit, 0.0)
        mean = local.masked_fill(~hit[..., None], 0.0).sum(dim=-2)
        mean = mean / hit.sum(dim=-1, keepdim=True).clamp_min(1)
        frames = torch.cat((strongest, mean, scan_age_s[..., None], any_hit.to(local.dtype)), dim=-1)
        temporal, _ = self.temporal(frames)  # 每个短窗从零状态开始，防止跨轨迹串帧
        return self.bottleneck(temporal[:, -1])


class TerrainGeometryProbe(nn.Module):
    """相同规格的读出头用于隐向量和显式高程图。"""

    def __init__(self, grid_shape: tuple[int, int], hidden_dim: int = 128):
        '''
        建立固定宽度的 MLP，输出每格高度和有支撑的 logit。
        grid_shape 是上游评分网格形状，hidden_dim 是各对照共用的隐藏层宽度。
        '''
        super().__init__()
        self.grid_shape = grid_shape
        self.input_dim = 2 * grid_shape[0] * grid_shape[1]  # 显式图含高度和有效掩码两个通道
        self.decoder = nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, self.input_dim))

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        '''
        对 [B, D] 表示补零以共用读出结构。
        返回 [B, 2, H, W]，通道依次为支撑高度（m）和支撑 logit。
        '''
        if representation.ndim != 2 or not 1 <= representation.shape[-1] <= self.input_dim:
            raise ValueError("Expected a [batch, feature] representation that fits the probe")
        padded = F.pad(representation, (0, self.input_dim - representation.shape[-1]))
        return self.decoder(padded).reshape(-1, 2, *self.grid_shape)


def explicit_map_features(height: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    '''
    将同窗雷达配准地图变成固定读出头的输入。
    输入为 [B, H, W] 米制高程和有效掩码，返回平铺的 [B, 2*H*W] 两通道表示。
    '''
    return torch.stack((height.masked_fill(~valid, 0.0), valid.to(height.dtype)), dim=1).flatten(1)
