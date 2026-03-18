# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ActorCriticAweNet: 基于跨模态注意力机制的足式机器人强化学习网络

核心创新：通过Cross-Attention让网络学会"带着目的去观察"
- 结合机器人自身的运动状态和全局地形大环境
- 主动聚焦（Attend）高程图中最关键的局部踏步点

网络结构:
    模块一：本体感知编码 (Proprioception Encoding)
        - 本体观测历史 -> MLP Encoder -> Proprioception Embedding
        - 输出兵分两路：一路直通动作解码器，一路参与生成Attention Query
    
    模块二：地形地图特征提取 (Map Feature Extraction)
        - 高程图历史帧 -> CNN -> Pointwise Local Features [B, Seq_Len, Feature_Dim]
        - Pointwise Local Features -> MLP -> MaxPool -> Global Features [B, Feature_Dim]
    
    模块三：核心注意力编码器 (Attention-Based Map Encoder)
        - Query: Proprioception Embedding + Global Features -> MLP -> Q [B, 1, Embed_Dim]
        - Key/Value: Pointwise Local Features [B, Seq_Len, Embed_Dim]
        - Multi-Head Attention -> Map Embedding [B, Embed_Dim]
    
    模块四：动作解码器 (Action Decoder)
        - Proprioception Embedding + Map Embedding -> MLP -> Actions
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
import copy
import os


class TerrainEncoder(nn.Module):
    """地形编码器：从高程图提取局部特征和全局特征
    
    输入: 高程图历史 [B, T, H, W]
    输出: 
        - pointwise_features: [B, Seq_Len, Feature_Dim] 局部特征序列
        - global_features: [B, Feature_Dim] 全局特征
    """
    
    def __init__(
        self,
        in_channels: int = 5,  # 历史帧数
        cnn_hidden_dims: list[int] = [16, 32, 64],
        cnn_kernel_sizes: list[int] = [3, 3, 3],
        cnn_strides: list[int] = [2, 2, 2],
        feature_dim: int = 64,
        vision_spatial_size: tuple[int, int] = (25, 17),
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # CNN骨干网络提取空间特征
        layers = []
        now_channels = in_channels
        for i, (hidden_dim, kernel_size, stride) in enumerate(zip(cnn_hidden_dims, cnn_kernel_sizes, cnn_strides)):
            layers.append(nn.Conv2d(
                now_channels,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            ))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            now_channels = hidden_dim
        
        self.conv = nn.Sequential(*layers)
        
        # 计算经过卷积后的特征图尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, vision_spatial_size[0], vision_spatial_size[1])
            dummy_output = self.conv(dummy_input)
            self.conv_output_shape = dummy_output.shape[1:]  # [C, H, W]
            self.seq_len = dummy_output.shape[2] * dummy_output.shape[3]  # H * W
        
        # 将CNN输出投影到特征维度
        self.conv_flat_dim = self.conv_output_shape[0] * self.conv_output_shape[1] * self.conv_output_shape[2]
        self.pointwise_proj = nn.Linear(self.conv_flat_dim // self.seq_len, feature_dim)
        
        # 全局特征提取：MLP + MaxPool
        self.global_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        print(f"TerrainEncoder: conv_output_shape={self.conv_output_shape}, seq_len={self.seq_len}, feature_dim={feature_dim}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, H, W] 多帧高程图历史
        
        Returns:
            pointwise_features: [B, Seq_Len, Feature_Dim] 局部特征序列
            global_features: [B, Feature_Dim] 全局特征
        """
        B = x.shape[0]
        
        # CNN提取空间特征
        x = self.conv(x)  # [B, C, H, W]
        
        # 展平为序列格式: [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).permute(0, 2, 1)  # [B, Seq_Len, C]
        
        # 投影到特征维度
        pointwise_features = self.pointwise_proj(x)  # [B, Seq_Len, Feature_Dim]
        
        # 提取全局特征
        global_features = self.global_mlp(pointwise_features)  # [B, Seq_Len, Feature_Dim]
        global_features = global_features.max(dim=1)[0]  # [B, Feature_Dim]
        
        return pointwise_features, global_features


class AttentionMapEncoder(nn.Module):
    """注意力地图编码器：通过Cross-Attention提取关键地形特征
    
    Query = MLP(Proprioception_Embedding + Global_Features)
    Key, Value = Pointwise_Local_Features
    Output = MultiHeadAttention(Q, K, V) -> Map_Embedding
    """
    
    def __init__(
        self,
        proprio_dim: int,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Query生成器：融合本体特征和全局特征
        self.query_mlp = nn.Sequential(
            nn.Linear(proprio_dim + feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer Norm
        self.norm = nn.LayerNorm(feature_dim)
        
        print(f"AttentionMapEncoder: proprio_dim={proprio_dim}, feature_dim={feature_dim}, num_heads={num_heads}")

    def forward(
        self,
        proprio_embedding: torch.Tensor,
        pointwise_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            proprio_embedding: [B, Proprio_Dim] 本体感知嵌入
            pointwise_features: [B, Seq_Len, Feature_Dim] 局部特征序列 (作为K, V)
            global_features: [B, Feature_Dim] 全局特征
        
        Returns:
            map_embedding: [B, Feature_Dim] 加权地图嵌入
        """
        B = proprio_embedding.shape[0]
        
        # 生成Query: 融合本体特征和全局特征
        q_input = torch.cat([proprio_embedding, global_features], dim=-1)  # [B, Proprio_Dim + Feature_Dim]
        query = self.query_mlp(q_input)  # [B, Feature_Dim]
        query = query.unsqueeze(1)  # [B, 1, Feature_Dim]
        
        # Key和Value直接使用局部特征
        key = pointwise_features  # [B, Seq_Len, Feature_Dim]
        value = pointwise_features  # [B, Seq_Len, Feature_Dim]
        
        # 多头注意力计算
        attn_output, attn_weights = self.attention(query, key, value)  # [B, 1, Feature_Dim]
        
        # 去掉序列维度
        map_embedding = attn_output.squeeze(1)  # [B, Feature_Dim]
        
        # Layer Norm
        map_embedding = self.norm(map_embedding)
        
        return map_embedding


class ActorCriticAweNet(nn.Module):
    """基于跨模态注意力机制的Actor-Critic网络
    
    核心特点：
    1. 本体感知编码器提取机器人状态特征
    2. 地形编码器从高程图提取局部和全局特征
    3. 注意力机制让网络主动聚焦关键地形区域
    4. 动作解码器融合所有信息输出动作
    """
    
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        env_cfg=None,
        alg_cfg: dict | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # 高程图配置
        vision_spatial_size: tuple[int, int] = (28, 20),
        elevation_history_length: int = 5,
        # 特征维度配置
        proprio_embed_dim: int = 64,
        feature_dim: int = 64,
        # 地形编码器配置
        terrain_cnn_hidden_dims: list[int] = [16, 32, 64],
        terrain_cnn_kernel_sizes: list[int] = [3, 3, 3],
        terrain_cnn_strides: list[int] = [2, 2, 2],
        # 注意力配置
        attention_num_heads: int = 4,
        attention_dropout: float = 0.1,
        # 本体编码器配置
        proprio_encoder_hidden_dims: list[int] = [128, 128],
        # Critic地形编码器配置（默认使用Actor的配置）
        critic_terrain_cnn_hidden_dims: list[int] = None,
        critic_terrain_cnn_kernel_sizes: list[int] = None,
        critic_terrain_cnn_strides: list[int] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticAweNet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        # 传递回Env的额外信息
        self.extra_info = dict()
        
        # 配置保存
        self.vision_spatial_size = vision_spatial_size
        self.elevation_history_length = elevation_history_length
        self.proprio_embed_dim = proprio_embed_dim
        self.feature_dim = feature_dim
        
        # Critic地形编码器配置
        if critic_terrain_cnn_hidden_dims is None:
            critic_terrain_cnn_hidden_dims = terrain_cnn_hidden_dims
        if critic_terrain_cnn_kernel_sizes is None:
            critic_terrain_cnn_kernel_sizes = terrain_cnn_kernel_sizes
        if critic_terrain_cnn_strides is None:
            critic_terrain_cnn_strides = terrain_cnn_strides
        
        # ============ 解析观测维度 ============
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if obs_group == "height_scan_policy":
                continue  # 跳过高程图，单独处理
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if obs_group == "height_scan_critic":
                continue  # 跳过高程图，单独处理
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        
        print(f"[AweNet] num_actor_obs (proprio): {num_actor_obs}")
        print(f"[AweNet] num_critic_obs (proprio privileged): {num_critic_obs}")
        
        self.state_dependent_std = state_dependent_std

        # ============ 模块一：本体感知编码器 (Actor) ============
        self.proprio_encoder_actor = MLP(
            num_actor_obs,
            proprio_embed_dim,
            proprio_encoder_hidden_dims,
            activation,
        )
        print(f"[AweNet] Actor Proprio Encoder: {num_actor_obs} -> {proprio_embed_dim}")
        
        # Actor观测归一化
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # ============ 模块二：地形编码器 (Actor) ============
        self.terrain_encoder_actor = TerrainEncoder(
            in_channels=elevation_history_length,
            cnn_hidden_dims=terrain_cnn_hidden_dims,
            cnn_kernel_sizes=terrain_cnn_kernel_sizes,
            cnn_strides=terrain_cnn_strides,
            feature_dim=feature_dim,
            vision_spatial_size=vision_spatial_size,
        )
        
        # ============ 模块三：注意力地图编码器 (Actor) ============
        self.attention_encoder_actor = AttentionMapEncoder(
            proprio_dim=proprio_embed_dim,
            feature_dim=feature_dim,
            num_heads=attention_num_heads,
            dropout=attention_dropout,
        )
        
        # ============ 模块四：动作解码器 (Actor) ============
        actor_input_dim = proprio_embed_dim + feature_dim
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"[AweNet] Actor Decoder: {actor_input_dim} (proprio_embed: {proprio_embed_dim} + map_embed: {feature_dim}) -> {num_actions}")
        print(f"[AweNet] Actor MLP: {self.actor}")

        # ============ Critic网络 ============
        # Critic复用Actor的前半部分架构
        # 1. 本体特权观测编码器
        self.proprio_encoder_critic = MLP(
            num_critic_obs,
            proprio_embed_dim,
            proprio_encoder_hidden_dims,
            activation,
        )
        print(f"[AweNet] Critic Proprio Encoder: {num_critic_obs} -> {proprio_embed_dim}")
        
        # 2. 地形编码器（共享结构，独立参数）
        self.terrain_encoder_critic = TerrainEncoder(
            in_channels=elevation_history_length,
            cnn_hidden_dims=critic_terrain_cnn_hidden_dims,
            cnn_kernel_sizes=critic_terrain_cnn_kernel_sizes,
            cnn_strides=critic_terrain_cnn_strides,
            feature_dim=feature_dim,
            vision_spatial_size=vision_spatial_size,
        )
        
        # 3. 注意力地图编码器（共享结构，独立参数）
        self.attention_encoder_critic = AttentionMapEncoder(
            proprio_dim=proprio_embed_dim,
            feature_dim=feature_dim,
            num_heads=attention_num_heads,
            dropout=attention_dropout,
        )
        
        # 4. Critic价值解码器
        critic_input_dim = proprio_embed_dim + feature_dim
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        print(f"[AweNet] Critic Value Decoder: {critic_input_dim} (proprio_embed: {proprio_embed_dim} + map_embed: {feature_dim}) -> 1")
        print(f"[AweNet] Critic MLP: {self.critic}")
        
        # Critic观测归一化
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # ============ 动作噪声配置 ============
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            proprio_obs = obs["policy"]
            self.actor_obs_normalizer.update(proprio_obs)
        if self.critic_obs_normalization:
            proprio_obs = obs["critic"]
            self.critic_obs_normalizer.update(proprio_obs)

    def _update_distribution(self, fused_features: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(fused_features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        else:
            mean = self.actor(fused_features)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def _normalize_elevation_map(self, height_map: torch.Tensor) -> torch.Tensor:
        """归一化多帧高程图历史
        
        Args:
            height_map: [B, T, H, W] 高程图历史（相对于机器人的高度）
        
        Returns:
            归一化后的高程图
        """
        # 不减均值，保留相对高度信息（楼梯台阶高度等）
        # 直接用固定scale归一化，让±0.3m映射到±1.0
        height_map = torch.clip(height_map / 0.3, -3.0, 3.0)
        return height_map

    def _check_height_map_shape(self, height_map: torch.Tensor, obs_key: str) -> None:
        if height_map.dim() != 4:
            raise ValueError(
                f"[AweNet] {obs_key} shape invalid: expected [B, T, H, W], got {tuple(height_map.shape)}"
            )
        if height_map.shape[1] != self.elevation_history_length:
            raise ValueError(
                f"[AweNet] {obs_key} history length mismatch: expected {self.elevation_history_length}, got {height_map.shape[1]}; full shape={tuple(height_map.shape)}"
            )
        if tuple(height_map.shape[-2:]) != self.vision_spatial_size:
            raise ValueError(
                f"[AweNet] {obs_key} spatial size mismatch: expected {self.vision_spatial_size}, got {tuple(height_map.shape[-2:])}"
            )

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """训练时的动作采样
        
        Actor Pipeline:
        1. 本体观测 -> MLP Encoder -> Proprioception Embedding
        2. 高程图 -> Terrain Encoder -> Pointwise Features + Global Features
        3. Cross-Attention -> Map Embedding
        4. [Proprio Embedding + Map Embedding] -> Action Decoder -> Actions
        """
        # 提取观测
        height_map = obs["height_scan_policy"]  # [B, T, H, W]
        proprio_obs = obs["policy"]  # [B, proprio_dim]
        self._check_height_map_shape(height_map, "height_scan_policy")
        
        # 归一化
        height_map = self._normalize_elevation_map(height_map)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 模块一：本体感知编码
        proprio_embedding = self.proprio_encoder_actor(proprio_obs)  # [B, proprio_embed_dim]
        
        # 模块二：地形特征提取
        pointwise_features, global_features = self.terrain_encoder_actor(height_map)
        # pointwise_features: [B, Seq_Len, feature_dim]
        # global_features: [B, feature_dim]
        
        # 模块三：注意力地图编码
        map_embedding = self.attention_encoder_actor(
            proprio_embedding, pointwise_features, global_features
        )  # [B, feature_dim]
        
        # 模块四：动作解码
        fused_features = torch.cat([proprio_embedding, map_embedding], dim=-1)
        self._update_distribution(fused_features)
        
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """推理时的确定性动作"""
        # 提取观测
        height_map = obs["height_scan_policy"]  # [B, T, H, W]
        proprio_obs = obs["policy"]  # [B, proprio_dim]
        self._check_height_map_shape(height_map, "height_scan_policy")
        
        # 归一化
        height_map = self._normalize_elevation_map(height_map)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        
        # 模块一：本体感知编码
        proprio_embedding = self.proprio_encoder_actor(proprio_obs)  # [B, proprio_embed_dim]
        
        # 模块二：地形特征提取
        pointwise_features, global_features = self.terrain_encoder_actor(height_map)
        
        # 模块三：注意力地图编码
        map_embedding = self.attention_encoder_actor(
            proprio_embedding, pointwise_features, global_features
        )  # [B, feature_dim]
        
        # 模块四：动作解码
        fused_features = torch.cat([proprio_embedding, map_embedding], dim=-1)
        
        if self.state_dependent_std:
            return self.actor(fused_features)[..., 0, :], self.extra_info
        else:
            return self.actor(fused_features), self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """评估状态价值
        
        Critic Pipeline (复用Actor架构):
        1. 本体特权观测 -> MLP Encoder -> Proprioception Embedding
        2. 高程图 -> Terrain Encoder -> Pointwise Features + Global Features
        3. Cross-Attention -> Map Embedding
        4. [Proprio Embedding + Map Embedding] -> Value Decoder -> Value
        """
        # 提取观测
        height_map = obs["height_scan_critic"]  # [B, T, H, W]
        proprio_obs = obs["critic"]  # [B, critic_dim]
        self._check_height_map_shape(height_map, "height_scan_critic")
        
        # 归一化
        height_map = self._normalize_elevation_map(height_map)
        proprio_obs = self.critic_obs_normalizer(proprio_obs)
        
        # 模块一：本体特权观测编码
        proprio_embedding = self.proprio_encoder_critic(proprio_obs)  # [B, proprio_embed_dim]
        
        # 模块二：地形特征提取
        pointwise_features, global_features = self.terrain_encoder_critic(height_map)
        # pointwise_features: [B, Seq_Len, feature_dim]
        # global_features: [B, feature_dim]
        
        # 模块三：注意力地图编码
        map_embedding = self.attention_encoder_critic(
            proprio_embedding, pointwise_features, global_features
        )  # [B, feature_dim]
        
        # 模块四：价值解码
        fused_features = torch.cat([proprio_embedding, map_embedding], dim=-1)
        
        # Critic输出
        value = self.critic(fused_features)
        
        return value

    def get_predictions(
        self, obs: TensorDict, actions: torch.Tensor, **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """获取预测值，用于AMP等算法"""
        raise NotImplementedError("AweNet does not support get_predictions")

    def get_value(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """获取状态价值"""
        return self.evaluate(obs, **kwargs)

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """创建优化器
        
        Args:
            learning_rate: 学习率
            
        Returns:
            优化器字典
        """
        import torch.optim as optim
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return {"optimizer": optimizer}

    def export_to_onnx(self, path: str, filename: str = "AweNet_policy.onnx", normalizer: torch.nn.Module | None = None, verbose: bool = False) -> None:
        """将AweNet策略导出为ONNX格式
        
        Args:
            path: 保存目录的路径
            filename: 导出的ONNX文件名，默认为"AweNet_policy.onnx"
            normalizer: 归一化模块，如果为None则使用Identity
            verbose: 是否打印模型摘要，默认为False
        """
        import copy
        import os
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        # 创建AweNet专用的导出器
        exporter = _AweNetOnnxPolicyExporter(self, normalizer, verbose)
        exporter.export(path, filename)


class _AweNetOnnxPolicyExporter(torch.nn.Module):
    """AweNet策略的ONNX导出器
    
    AweNet的推理流程:
    1. 从合并输入中分离：本体观测 + 高程图历史
    2. 本体编码：proprio_obs -> proprio_encoder -> proprio_embedding
    3. 地形编码：height_map -> terrain_encoder -> pointwise_features + global_features
    4. 注意力编码：proprio_embedding + features -> attention_encoder -> map_embedding
    5. Actor推理：[proprio_embedding + map_embedding] -> actor -> actions
    """

    def __init__(self, policy: ActorCriticAweNet, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # 复制策略所需的模块
        # 本体编码器
        if hasattr(policy, "proprio_encoder_actor"):
            self.proprio_encoder_actor = copy.deepcopy(policy.proprio_encoder_actor)
        
        # 地形编码器
        if hasattr(policy, "terrain_encoder_actor"):
            self.terrain_encoder_actor = copy.deepcopy(policy.terrain_encoder_actor)
        
        # 注意力编码器
        if hasattr(policy, "attention_encoder_actor"):
            self.attention_encoder_actor = copy.deepcopy(policy.attention_encoder_actor)
        
        # Actor网络
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        
        # 保存维度信息
        self.vision_spatial_size = policy.vision_spatial_size
        self.elevation_history_length = policy.elevation_history_length
        self.state_dependent_std = policy.state_dependent_std
        
        # 计算本体观测维度（需要从policy中获取）
        # 由于在__init__中没有存储，我们需要从encoder推断
        self.proprio_dim = policy.proprio_encoder_actor[0].in_features
        
        # 计算高程图维度
        height, width = self.vision_spatial_size
        self.elevation_dim = self.elevation_history_length * height * width
        
        # 复制归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs_input):
        """前向传播（单输入版本）
        
        Args:
            obs_input: 合并的观测数据，形状为 [batch_size, total_obs_dim]
                       组成：[本体观测 | 高程图]
                       - 本体观测：proprio_dim 维
                       - 高程图：elevation_dim 维
        
        Returns:
            actions_mean: 动作均值，形状为 [batch_size, num_actions]
        """
        batch_size = obs_input.shape[0]
        
        # 切片分离各部分数据
        offset = 0
        # 1. 本体观测
        proprio_obs = obs_input[:, offset:offset + self.proprio_dim]
        offset += self.proprio_dim
        
        # 2. 高程图数据
        elevation_data_flat = obs_input[:, offset:]
        
        # 将高程图数据reshape为 [B, T, H, W]
        height, width = self.vision_spatial_size
        elevation_data = elevation_data_flat.reshape(
            batch_size, self.elevation_history_length, height, width
        )
        
        # 应用归一化器到本体观测
        proprio_obs_normalized = self.normalizer(proprio_obs)
        
        # 归一化高程图
        elevation_data = torch.clip(elevation_data / 0.3, -3.0, 3.0)
        
        # 模块一：本体感知编码
        proprio_embedding = self.proprio_encoder_actor(proprio_obs_normalized)  # [B, proprio_embed_dim]
        
        # 模块二：地形特征提取
        pointwise_features, global_features = self.terrain_encoder_actor(elevation_data)
        # pointwise_features: [B, Seq_Len, feature_dim]
        # global_features: [B, feature_dim]
        
        # 模块三：注意力地图编码
        map_embedding = self.attention_encoder_actor(
            proprio_embedding, pointwise_features, global_features
        )  # [B, feature_dim]
        
        # 模块四：动作解码
        fused_features = torch.cat([proprio_embedding, map_embedding], dim=-1)
        
        # 输出动作（推理时只使用均值）
        if self.state_dependent_std:
            actions_mean = self.actor(fused_features)[..., 0, :]
        else:
            actions_mean = self.actor(fused_features)
        
        return actions_mean

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        
        # 计算总输入维度
        total_obs_dim = self.proprio_dim + self.elevation_dim
        
        # 创建单个合并的输入示例
        obs_input = torch.zeros(1, total_obs_dim)
        
        print(f"\n{'='*80}")
        print(f"ONNX导出配置 (AweNet - 单输入模式):")
        print(f"{'='*80}")
        print(f"  本体观测维度:     {self.proprio_dim}")
        print(f"  高程图维度:       {self.elevation_dim} ({self.elevation_history_length}×{self.vision_spatial_size[0]}×{self.vision_spatial_size[1]})")
        print(f"  总输入维度:       {total_obs_dim}")
        print(f"  ")
        print(f"  输入切片方式:")
        print(f"    [:, 0:{self.proprio_dim}] = 本体观测")
        print(f"    [:, {self.proprio_dim}:] = 高程图")
        print(f"  ")
        print(f"  处理流程:")
        print(f"    1. 本体观测归一化")
        print(f"    2. 本体感知编码 -> proprio_embedding")
        print(f"    3. 地形特征提取 (CNN) -> pointwise_features + global_features")
        print(f"    4. 注意力地图编码 -> map_embedding")
        print(f"    5. Actor: [proprio_embedding + map_embedding] -> actions")
        print(f"{'='*80}\n")
        
        torch.onnx.export(
            self,
            obs_input,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
