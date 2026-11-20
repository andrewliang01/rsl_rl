"""面向足端高程栅格的稀疏测量补洞。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SparseBEVCompleter(nn.Module):
    """用二维 U-Net 补空格，按当前帧掩码写回可靠实测值。"""

    def __init__(self, max_frames: int = 5, channels: int = 16):
        '''
        配置单帧和短历史共用的双尺度 U-Net。
        max_frames 为输入帧数上限，channels 为首层宽度。
        '''
        super().__init__()
        if max_frames < 1 or channels < 1:
            raise ValueError("max_frames and channels must be positive")
        self.max_frames = max_frames
        self.encoder_full = nn.Sequential(
            nn.Conv2d(3 * max_frames, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
        )
        self.encoder_half = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1), nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * channels, 4 * channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(4 * channels, 4 * channels, 3, padding=1), nn.SiLU(),
        )
        self.decode_half = nn.Sequential(
            nn.Conv2d(6 * channels, 2 * channels, 3, padding=1), nn.SiLU(),
        )
        self.decode_full = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(
        self, height: torch.Tensor, observed: torch.Tensor, age_s: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        '''
        输入 height 和布尔 observed 为 [B,T,H,W]，从最早帧到当前帧。
        各帧栅格按其采集坐标生成，轴序、高度单位（m）和零点遵循共用考卷。
        多帧传 age_s[B,T]，表示距当前的秒数，当前帧为 0。
        返回 [B,H,W] 的稠密 height 和当前有效实测掩码 observed。
        '''
        if height.ndim != 4 or observed.shape != height.shape or observed.dtype != torch.bool:
            raise ValueError("height and boolean observed must have matching [B,T,H,W] shapes")
        frame_count = height.shape[1]
        if not 1 <= frame_count <= self.max_frames:
            raise ValueError("height frame count is outside [1, max_frames]")
        if age_s is None:
            if frame_count != 1:
                raise ValueError("age_s is required for historical frames")
            age_s = height.new_zeros(height.shape[:2])
        if age_s.shape != height.shape[:2]:
            raise ValueError("age_s must have shape [B,T]")

        valid = observed & torch.isfinite(height)
        measured = torch.where(valid, height, 0.0)
        current_valid = valid[:, -1:]
        current = measured[:, -1:]
        # 高度原点只取当帧实测均值，空帧使用坐标零点。
        count = current_valid.sum(dim=(-2, -1), keepdim=True)
        origin = current.sum(dim=(-2, -1), keepdim=True) / count.clamp_min(1)
        relative = torch.where(valid, measured - origin, 0.0)
        age = torch.where(valid, age_s.to(height)[:, :, None, None], 0.0)
        features = torch.stack((relative, valid.to(height.dtype), age), dim=2).flatten(1, 2)
        # 前补空帧，使单帧与历史对照保持相同网络容量和当前帧通道。
        features = F.pad(features, (0, 0, 0, 0, 3 * (self.max_frames - frame_count), 0))

        full = self.encoder_full(features)
        half = self.encoder_half(full)
        deep = self.bottleneck(half)
        deep = F.interpolate(deep, size=half.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_half(torch.cat((deep, half), dim=1))
        decoded = F.interpolate(decoded, size=full.shape[-2:], mode="bilinear", align_corners=False)
        inferred = origin + self.decode_full(torch.cat((decoded, full), dim=1))
        return {
            "height": torch.where(current_valid, current, inferred)[:, 0],
            "observed": current_valid[:, 0],
        }
