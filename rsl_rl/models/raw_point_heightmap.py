# SPDX-License-Identifier: BSD-3-Clause

"""G1 原始雷达点集的足端高度查询模型。"""

import math
from typing import NamedTuple, Optional

import torch
from torch import nn


class RawPointHeightmapOutput(NamedTuple):
    height_m: torch.Tensor
    observed_logits: torch.Tensor


class RawPointHeightmap(nn.Module):
    def __init__(
        self,
        width: int = 48,  # 邻域编码与融合头的隐层宽度。
        neighbors: int = 16,  # 每个格心每扫保留的三维近邻上限。
        local_radius_m: float = 0.15,  # 三维邻域半径，单位为米。
        sensor_pitch_deg: float = 2.3,  # 固定安装俯仰角，绕 +y 轴。
    ):
        '''
        构造共享点编码器、跨扫加权及高度与覆盖读出头。
        输入为网络规格及雷达安装俯仰角，输出为可调用模型。
        '''
        super().__init__()
        if width < 1 or neighbors < 1 or not math.isfinite(local_radius_m) or local_radius_m <= 0:
            raise ValueError("width, neighbors and local_radius_m must be positive")
        if not math.isfinite(sensor_pitch_deg):
            raise ValueError("sensor_pitch_deg must be finite")
        self.neighbors = neighbors
        self.local_radius_m = local_radius_m
        angle = math.radians(sensor_pitch_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
        self.register_buffer("lidar_to_nominal", rotation)

        # StairMaster 的 yx 顺序为 x 外层、y 内层，查询原点与雷达安装位置一致。
        x = torch.linspace(-0.675, 0.675, 28)
        y = torch.linspace(-0.475, 0.475, 20)
        self.register_buffer("query_xy_m", torch.cartesian_prod(x, y))

        self.local_encoder = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.frame_score = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, 1))
        self.query_head = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(
        self,
        points_xyz: torch.Tensor,
        point_valid: torch.Tensor,
        age_s: torch.Tensor,
        query_xy_m: Optional[torch.Tensor] = None,
    ) -> RawPointHeightmapOutput:
        '''
        逐扫在局部三维邻域编码点特征，再根据格心和时间跨扫融合。
        输入为同设备的雷达系 [B,T,N,3]、布尔掩码 [B,T,N]、观测年龄 [B,T] 或
        [B,T,N]（查询时间减采集时间，秒），以及可选的固定航向系格心 [Q,2]。
        每扫保留采集时的独立坐标，调用方按回合截窗并用掩码标记填充。
        输出为 [B,Q] 的相对高度（参考高减地表高，按标签零点）及可靠覆盖 logits。
        '''
        if points_xyz.ndim != 4 or points_xyz.size(-1) != 3 or not points_xyz.is_floating_point():
            raise ValueError("points_xyz must be floating point [B,T,N,3]")
        batch_size, frames, point_count = points_xyz.shape[:3]
        if min(batch_size, frames, point_count) < 1:
            raise ValueError("B, T and N must be positive. Use point_valid for empty scans.")
        if point_valid.shape != points_xyz.shape[:-1] or point_valid.dtype != torch.bool:
            raise ValueError("point_valid must be boolean [B,T,N]")
        if age_s.shape == points_xyz.shape[:2]:
            age_s = age_s.unsqueeze(-1).expand_as(point_valid)
        if age_s.shape != point_valid.shape:
            raise ValueError("age_s must be [B,T] or [B,T,N]")
        if query_xy_m is None:
            query_xy_m = self.query_xy_m
        if query_xy_m.ndim != 2 or query_xy_m.size(-1) != 2 or query_xy_m.size(0) < 1:
            raise ValueError("query_xy_m must be [Q,2] with Q > 0")

        # 缺测、非有限位置及查询以后的回波均不参与邻域或融合。
        valid = point_valid & torch.isfinite(points_xyz).all(dim=-1) & torch.isfinite(age_s) & (age_s >= 0)
        points = torch.where(valid.unsqueeze(-1), points_xyz, 0.0).to(dtype=self.lidar_to_nominal.dtype)
        points = (points @ self.lidar_to_nominal.T).flatten(0, 1)
        age = torch.where(valid, age_s, 0.0).flatten(0, 1).to(points)
        valid = valid.flatten(0, 1)
        query = query_xy_m.to(points).expand(points.size(0), -1, -1)

        # 先按水平距离找格心的回波锚点，再围绕锚点作三维搜索。
        center_index = (
            torch.cdist(query.float(), points[..., :2].float())
            .masked_fill_(~valid[:, None], torch.inf)
            .argmin(dim=-1)
        )
        scan_index = torch.arange(points.size(0), device=points.device)[:, None, None]
        centers = points[scan_index.squeeze(-1), center_index]
        distance, index = (
            torch.cdist(centers.float(), points.float())
            .masked_fill_(~valid[:, None], torch.inf)
            .topk(min(self.neighbors, point_count), dim=-1, largest=False)
        )
        local_valid = distance <= self.local_radius_m
        local_points = points[scan_index, index]
        local_input = torch.cat(
            (
                local_points - centers.unsqueeze(-2),
                local_points[..., :2] - query.unsqueeze(-2),
                local_points[..., 2:3],
                age[scan_index, index].unsqueeze(-1),
            ),
            dim=-1,
        )
        features = self.local_encoder(local_input)
        pooled = features.masked_fill(~local_valid.unsqueeze(-1), torch.finfo(features.dtype).min).amax(dim=-2)
        has_neighbor = local_valid.any(dim=-1)
        pooled = torch.where(has_neighbor.unsqueeze(-1), pooled, 0.0)

        # 全空格的帧权重均为零，读出头仍返回密集预测。
        frame_logits = self.frame_score(torch.cat((pooled, query), dim=-1)).squeeze(-1)
        frame_logits = frame_logits.masked_fill(~has_neighbor, torch.finfo(frame_logits.dtype).min)
        frame_logits = frame_logits.reshape(batch_size, frames, -1)
        frame_weights = frame_logits.softmax(dim=1) * has_neighbor.reshape_as(frame_logits)
        fused = (frame_weights.unsqueeze(-1) * pooled.unflatten(0, (batch_size, frames))).sum(dim=1)
        output = self.query_head(torch.cat((fused, query_xy_m.to(points).expand(batch_size, -1, -1)), dim=-1))
        return RawPointHeightmapOutput(output[..., 0], output[..., 1])
