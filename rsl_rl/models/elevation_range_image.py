# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

'''球面距离图与短时雷达记忆的足端高程重建。'''

import torch
import torch.nn.functional as F
from torch import nn


class _AzimuthBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        '''
        初始化逐帧卷积块，按俯仰和方位角各下采样两倍。
        输入为通道数，输出为带组归一化和 SiLU 的卷积模块。
        '''
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=(1, 0), bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        '''
        对扫描图卷积，方位角首尾相接，俯仰角由卷积层补零。
        输入为 [N,C,H,W] 的球面图，输出为下采样特征。
        '''
        return super().forward(F.pad(image, (1, 1, 0, 0), mode="circular"))


class SphericalRangeHeightmap(nn.Module):
    '''从各帧各自的雷达坐标重建当前足端高程和可靠观测覆盖。'''

    def __init__(self):
        '''
        初始化共享逐帧编码器、因果 GRU 和足端网格解码器。
        无构造参数，单帧、五帧和两种分辨率共用同样的参数结构。
        '''
        super().__init__()
        self.encoder = nn.Sequential(
            _AzimuthBlock(2, 16),
            _AzimuthBlock(16, 24),
            _AzimuthBlock(24, 32),
            nn.AdaptiveAvgPool2d((2, 12)),  # 固定特征尺寸，细分箱沿用相同 GRU 参数量。
            nn.Flatten(),
        )
        hidden_size = 128  # GRU 隐状态维数，所有试次固定。
        self.memory = nn.GRU(32 * 2 * 12 + 1, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, 32 * 7 * 5)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(24, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 2, 3, padding=1),  # 分别输出米制高程和覆盖 logit。
        )

    def forward(
        self, range_image: torch.Tensor, frame_age_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        按从旧到新的顺序编码窗口，使用最后一次 GRU 更新解码。
        输入 [B,K,2,H,W] 存归一化距离与有效位，帧龄为 [B,K] 秒。
        返回两个 [B,28,20] 张量，分别为米制高程和覆盖 logit。
        '''
        if range_image.ndim != 5 or range_image.shape[1] not in (1, 5):
            raise ValueError("range_image 必须包含 1 或 5 帧")
        if range_image.shape[2:] not in ((2, 16, 96), (2, 32, 192)):
            raise ValueError("每帧需要距离和有效位两个通道，以及 16×96 或 32×192 角格")
        if frame_age_s.shape != range_image.shape[:2]:
            raise ValueError("frame_age_s 必须为 [B,K]")

        valid = range_image[:, :, 1] == 1
        distance = torch.where(valid, range_image[:, :, 0], 0.0)  # 空格不携带距离。
        frames = torch.stack((distance, valid.to(distance.dtype)), dim=2)
        features = self.encoder(frames.flatten(0, 1)).reshape(*frames.shape[:2], -1)
        age = frame_age_s.to(features.dtype) / 0.4  # 以五帧 10 Hz 的首末间隔归一化。
        sequence = torch.cat((features, age.unsqueeze(-1)), dim=-1)
        _, hidden = self.memory(sequence)  # 每次查询独立从零初始化，不跨窗保留状态。
        maps = self.decoder(self.projection(hidden[-1]).reshape(-1, 32, 7, 5))
        return maps[:, 0], maps[:, 1]
