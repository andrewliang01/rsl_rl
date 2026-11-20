"""阶段 2.8 从多帧三态射线体素恢复当前帧的三维占据。"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def encode_voxel_history(evidence: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
    '''
    将三态体素和采集龄期编码为卷积输入。
    输入按 B,K,Z,Y,X 排列，龄期为 B,K 或与体素同形的归一化时间。
    各帧保留自身雷达坐标，输出 B,3K,Z,Y,X，通道为命中、自由、龄期。
    '''
    if evidence.ndim != 5 or age.shape not in (evidence.shape[:2], evidence.shape):
        raise ValueError("evidence or acquisition age shape does not match B,K,Z,Y,X")
    if age.ndim == 2:
        age = age[:, :, None, None, None]
    hit = evidence == 1
    free = evidence == -1
    # 无可靠射线的格子保留两个零掩码和零龄期。
    channels = torch.stack((hit.float(), free.float(), torch.where(hit | free, age.float(), 0.0)), dim=2)
    return channels.flatten(1, 2)


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    '''
    构造一层局部三维编码。
    输入通道数及步长，输出带归一化、可用于小批训练的卷积块。
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


class VoxelOccupancyNet(nn.Module):
    """两级三维编码和跳接解码，输出占据与教师覆盖的 logits。"""

    def __init__(self, history: int = 5, width: int = 12):
        '''
        根据因果窗帧数和基础通道数初始化网络。
        三维下采样扩大感受野，原分辨率跳接保留细沟与薄障碍。
        '''
        super().__init__()
        if history < 1 or width < 4 or width % 4:
            raise ValueError("history must be positive and width must be a positive multiple of four")
        self.history = history
        self.full = _conv_block(3 * history, width)
        self.half_scale = _conv_block(width, width * 2, stride=2)
        self.quarter_scale = _conv_block(width * 2, width * 4, stride=2)
        self.context = _conv_block(width * 4, width * 4)
        self.decode_half = _conv_block(width * 6, width * 2)
        self.decode_full = _conv_block(width * 3, width)
        self.output = nn.Conv3d(width, 2, kernel_size=1)

    def forward(self, evidence: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        '''
        将 B,K,Z,Y,X 的三态证据和 B,K（或逐体素）龄期送入网络。
        显式按跳接大小上采样，输出对应当前格子的 B,2,Z,Y,X logits。
        '''
        if evidence.ndim != 5 or evidence.shape[1] != self.history:
            raise ValueError(f"expected B,{self.history},Z,Y,X evidence")
        full = self.full(encode_voxel_history(evidence, age))
        half = self.half_scale(full)
        coarse = self.context(self.quarter_scale(half))
        up = F.interpolate(coarse, size=half.shape[-3:], mode="trilinear", align_corners=False)
        decoded = self.decode_half(torch.cat((half, up), dim=1))
        up = F.interpolate(decoded, size=full.shape[-3:], mode="trilinear", align_corners=False)
        return self.output(self.decode_full(torch.cat((full, up), dim=1)))


def _balanced_bce(logits: torch.Tensor, occupied: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    '''
    在有效体素中分别求占据与自由的平均二元交叉熵。
    输入概率 logits、布尔占据标签与掩码，输出标量，单类批次按该类归一化。
    '''
    loss = F.binary_cross_entropy_with_logits(logits.float(), occupied.float(), reduction="none")
    positive = valid & occupied
    negative = valid & ~occupied
    n_positive = positive.sum()
    n_negative = negative.sum()
    n_classes = (n_positive > 0).float() + (n_negative > 0).float()
    loss_positive = (loss * positive).sum() / n_positive.clamp_min(1)
    loss_negative = (loss * negative).sum() / n_negative.clamp_min(1)
    return (loss_positive + loss_negative) / n_classes.clamp_min(1)


def voxel_occupancy_loss(
    logits: torch.Tensor,
    history_target: torch.Tensor,
    volume_valid: torch.Tensor,
    simulation_target: Optional[torch.Tensor] = None,
    coverage_weight: float = 0.2,
    completion_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    '''
    计算观测监督损失，补全组额外监督 H 未知且有仿真真值的体素。
    输入 B,2,Z,Y,X 的预测与 B,Z,Y,X 的三态 H、有效掩码和可选布尔 S。
    返回标量总损失和分项，供两组用相同训练和诊断入口。
    '''
    occupancy_logits, coverage_logits = logits.unbind(dim=1)
    known = (history_target != 0) & volume_valid
    observed = _balanced_bce(occupancy_logits, history_target == 1, known)
    coverage = _balanced_bce(coverage_logits, known, volume_valid)
    parts = {"observed": observed, "coverage": coverage}
    total = observed + coverage_weight * coverage
    if simulation_target is not None:
        unknown = (history_target == 0) & volume_valid
        completion = _balanced_bce(occupancy_logits, simulation_target, unknown)
        parts["completion"] = completion
        total = total + completion_weight * completion
    return total, parts
