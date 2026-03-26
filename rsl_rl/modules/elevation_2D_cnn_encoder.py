from __future__ import annotations

import torch
import torch.nn as nn


class Elevation2DCNNEncoder(nn.Module):
    """2DCNN编码器，用于处理多帧高程图历史（作为通道）"""
    
    def __init__(self, 
                 in_channels=1,  # 输入通道数，通常对应历史高程图帧数，例如 5
                 hidden_dims=[16, 32, 64],  # 每层卷积的输出通道数
                 kernel_sizes=[3, 3, 3],  # 每层卷积核大小
                 strides=[2, 2, 2],  # 每层卷积步长
                 out_dim=64,  # 最终输出特征向量维度
                 vision_spatial_size=(25, 17)):  # 输入高程图的空间尺寸 (H, W)
        super().__init__()
        
        # 构建2DCNN卷积层
        layers = []
        now_channels = in_channels
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            layers.append(nn.Conv2d(
                now_channels, 
                hidden_dim, 
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2
            ))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            now_channels = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        
        # 计算经过卷积后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.numel()
        
        self.fc = nn.Linear(conv_output_size, out_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W] 多帧高程图历史（已归一化），T为历史帧数
        Returns:
            features: [B, out_dim] 特征向量
        """
        # 不需要添加通道维度，输入已经是 [B, T, H, W] 格式
        # 直接通过2DCNN，CNN会处理T个通道
        
        x = self.conv(x)
        
        # 展平并通过全连接层
        x = x.flatten(1)
        x = self.fc(x)
        
        return xfrom __future__ import annotations

import torch
import torch.nn as nn


class Elevation2DCNNEncoder(nn.Module):
    """2DCNN编码器，用于处理多帧高程图历史（作为通道）"""
    
    def __init__(self, 
                 in_channels=1,  # 输入通道数，通常对应历史高程图帧数，例如 5
                 hidden_dims=[16, 32, 64],  # 每层卷积的输出通道数
                 kernel_sizes=[3, 3, 3],  # 每层卷积核大小
                 strides=[2, 2, 2],  # 每层卷积步长
                 out_dim=64,  # 最终输出特征向量维度
                 vision_spatial_size=(25, 17)):  # 输入高程图的空间尺寸 (H, W)
        super().__init__()
        
        # 构建2DCNN卷积层
        layers = []
        now_channels = in_channels
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            layers.append(nn.Conv2d(
                now_channels, 
                hidden_dim, 
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2
            ))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            now_channels = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        
        # 计算经过卷积后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.numel()
        
        self.fc = nn.Linear(conv_output_size, out_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W] 多帧高程图历史（已归一化），T为历史帧数
        Returns:
            features: [B, out_dim] 特征向量
        """
        # 不需要添加通道维度，输入已经是 [B, T, H, W] 格式
        # 直接通过2DCNN，CNN会处理T个通道
        
        x = self.conv(x)
        
        # 展平并通过全连接层
        x = x.flatten(1)
        x = self.fc(x)
        
        return x