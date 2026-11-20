# SPDX-License-Identifier: BSD-3-Clause

'''从逐帧雷达点恢复足端支撑高度，可同时输出分层占据。'''

from __future__ import annotations

from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F


class PillarHeightMap(nn.Module):
    '''用同一俯视骨干比较单高度、格内高度统计和可训练点柱。'''

    def __init__(
        self,
        *,
        scans: int,
        shape_xy: tuple[int, int],
        lower_xy_m: tuple[float, float],
        cell_m: float = 0.05,
        representation: str = "pillars",
        layers: int = 0,
        width: int = 32,
    ):
        '''
        设置时间窗、俯视网格和头部，网格下界指首格的边界。
        输入扫描数、XY 格数、米制下界与柱宽；layers 为竖直占据通道数。
        初始化按同一卷积结构解码三个输入对照，layers=0 只输出支撑几何。
        '''
        super().__init__()
        if representation not in {"single", "statistics", "pillars"}:
            raise ValueError("representation 须为 single、statistics 或 pillars")
        if len(shape_xy) != 2 or min(shape_xy) < 2 or scans < 1 or layers < 0 or width < 1:
            raise ValueError("网格尺寸至少为 2，扫描数和网络宽度为正，分层数非负")
        if len(lower_xy_m) != 2 or not all(map(isfinite, lower_xy_m)) or not (isfinite(cell_m) and cell_m > 0):
            raise ValueError("网格下界须为有限 XY 坐标，柱宽须为有限正数")
        self.scans = scans
        self.shape_xy = tuple(shape_xy)
        self.cell_m = cell_m
        self.representation = representation
        self.layers = layers
        self.register_buffer("lower_xy_m", torch.tensor(lower_xy_m, dtype=torch.float32))
        self.register_buffer("shape_xy_tensor", torch.tensor(shape_xy, dtype=torch.long), persistent=False)

        # 所有组共用格心最近高度、平均回波时间和命中掩码。
        channels = 3
        if representation == "statistics":
            channels += 5  # 最低、均值、最高、标准差和对数命中数。
        if representation == "pillars":
            # XYZ、相对时间、柱内质心偏移和水平柱心偏移。
            self.point_encoder = nn.Sequential(nn.Linear(9, width), nn.ReLU())
            channels += width + 1  # 池化特征与对数命中数。

        self.stem = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.SiLU())
        self.coarse = nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())
        self.fusion = nn.Sequential(nn.Conv2d(scans * width, width, 1), nn.SiLU())
        self.decode = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), nn.SiLU())
        self.support_head = nn.Conv2d(width, 3, 1)
        self.layer_head = nn.Conv2d(width, layers, 1) if layers else None

    def rasterize(self, xyz: torch.Tensor, age_s: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        '''
        逐帧将雷达 XYZ 分到水平格，保留高度和各点的采集时间。
        输入 [B,T,R,3] 米、[B,T,R] 距查询时刻秒数和有效掩码。
        输出 [B,T,C,Nx,Ny]，Nx、Ny 对应 X、Y，空柱全零。
        '''
        if xyz.ndim != 4 or xyz.shape[-1] != 3 or xyz.shape[1] != self.scans or not xyz.is_floating_point():
            raise ValueError("xyz 须为 [B,T,R,3]，T 与固定时窗一致")
        if age_s.shape != xyz.shape[:-1] or valid.shape != age_s.shape or valid.dtype != torch.bool:
            raise ValueError("采集时间与布尔掩码须为匹配的 [B,T,R]")
        batch, scans = xyz.shape[:2]
        nx, ny = self.shape_xy
        cells = batch * scans * nx * ny
        xyz = xyz.flatten(0, 1)
        age_s = age_s.flatten(0, 1).to(dtype=xyz.dtype)
        valid = valid.flatten(0, 1)
        # 未来回波与填充点都不进桶，历史帧保持各自采集坐标。
        scaled_xy = (xyz[..., :2] - self.lower_xy_m) / self.cell_m
        valid = valid & xyz.isfinite().all(-1) & age_s.isfinite() & (age_s >= 0)
        valid = valid & (scaled_xy >= 0).all(-1) & (scaled_xy < self.shape_xy_tensor).all(-1)
        # 先裁掉格外回波，避免为一轮扫描的数万无效点分配网络特征。
        scan, ray = valid.nonzero(as_tuple=True)
        points = xyz[scan, ray]
        age = age_s[scan, ray]
        grid_xy = scaled_xy[scan, ray]
        cell_xy = grid_xy.floor().long()
        index = (scan * nx + cell_xy[:, 0]) * ny + cell_xy[:, 1]
        z = points[:, 2]
        zeros = xyz.new_zeros(cells)
        count = zeros.scatter_add(0, index, torch.ones_like(z))
        observed = count > 0
        mean_age = zeros.scatter_add(0, index, age) / count.clamp_min(1)

        # 单高度取最接近格心的实际回波。距离相同时先取较新回波，再取较低点。
        center_offset = grid_xy - cell_xy - 0.5  # 单位为柱宽。
        distance = center_offset.square().sum(-1)
        closest = zeros.new_full(zeros.shape, torch.inf).scatter_reduce_(0, index, distance, "amin")
        nearest = distance == closest[index]
        nearest_age = zeros.new_full(zeros.shape, torch.inf).scatter_reduce_(
            0, index, age.masked_fill(~nearest, torch.inf), "amin"
        )
        nearest = nearest & (age == nearest_age[index])
        height = zeros.new_full(zeros.shape, torch.inf).scatter_reduce_(0, index, z.masked_fill(~nearest, torch.inf), "amin")
        height = torch.where(observed, height, 0.0)
        features = [height[..., None], mean_age[..., None], observed[..., None].to(xyz.dtype)]

        if self.representation == "statistics":
            z_min = zeros.new_full(zeros.shape, torch.inf).scatter_reduce_(0, index, z, "amin")
            z_max = zeros.new_full(zeros.shape, -torch.inf).scatter_reduce_(0, index, z, "amax")
            mean_z = zeros.scatter_add(0, index, z) / count.clamp_min(1)
            residual = z - mean_z[index]
            std_z = (zeros.scatter_add(0, index, residual.square()) / count.clamp_min(1)).sqrt()
            stats = torch.stack((z_min, mean_z, z_max, std_z, count.log1p()), dim=-1)
            features.append(torch.where(observed[..., None], stats, 0.0))
        if self.representation == "pillars":
            sums = xyz.new_zeros(cells, 3).index_add_(0, index, points)
            mean_xyz = sums / count.clamp_min(1)[..., None]
            residual = points - mean_xyz[index]
            point_features = torch.cat((points, age[:, None], residual, center_offset), dim=-1)
            encoded = self.point_encoder(point_features)
            # 池化仅在同一帧、同一水平柱内，保留逐帧顺序交给融合层。
            expanded_index = index[..., None].expand_as(encoded)
            pooled = encoded.new_zeros(cells, encoded.shape[-1])
            pooled.scatter_reduce_(0, expanded_index, encoded, "amax")
            features.extend((pooled, count.log1p()[..., None]))

        raster = torch.cat(features, dim=-1)
        return raster.reshape(batch, scans, nx, ny, -1).permute(0, 1, 4, 2, 3)

    def forward(self, xyz: torch.Tensor, age_s: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        '''
        编码因果回波窗口，融合帧特征后读出当前格的预测。
        输入与 rasterize 相同，时间轴按旧到新排列。
        输出米制高度 [B,Nx,Ny]、支撑和覆盖 logits，以及可选 [B,L,Nx,Ny] 占据 logits。
        '''
        raster = self.rasterize(xyz, age_s, valid)
        local = self.stem(raster.flatten(0, 1))
        coarse = self.coarse(local)
        fused = self.fusion(coarse.reshape(xyz.shape[0], -1, *coarse.shape[-2:]))
        fused = F.interpolate(fused, size=self.shape_xy, mode="bilinear", align_corners=False)
        hidden = self.decode(torch.cat((fused, local[self.scans - 1 :: self.scans]), dim=1))
        height_m, support, coverage = self.support_head(hidden).unbind(dim=1)
        outputs = {"height_m": height_m, "support_logits": support, "coverage_logits": coverage}
        if self.layer_head is not None:
            outputs["occupancy_logits"] = self.layer_head(hidden)
        return outputs
