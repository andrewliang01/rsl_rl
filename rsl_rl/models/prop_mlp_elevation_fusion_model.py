# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.ame2_encoder import AME2Encoder
from rsl_rl.modules.bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    load_frozen_reconstructor_checkpoint,
    normalize_heightmap_target_contract,
    preflight_validate_lidar_history,
    reconstructor_checkpoint_schema,
)
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder
from rsl_rl.modules.r2plus1d_elevation_encoder import R2Plus1DElevationEncoder
from rsl_rl.modules.ray_time_attention_encoder import RayTimeAttentionEncoder
from rsl_rl.utils import resolve_callable, unpad_trajectories


class PropMLPElevationFusionModel(nn.Module):
    """Proprioception-MLP + elevation-encoder fusion model.

    This model follows the new ``rsl_rl.models`` style while reproducing the old actor-critic pattern of:

    - proprioception observation groups -> MLP encoder
    - elevation-map observation group -> 2D CNN encoder
    - fused proprio and elevation features -> MLP head -> outputs

    The same class can be instantiated for both actor and critic by passing different ``obs_set`` and
    ``output_dim`` values.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    _CNN_OBSERVATION_MODES = {
        "elevationmap": 0,
        "depthcamera": 1,
        "inverse_depth": 2,
    }
    _ELEVATION_ENCODER_TYPES = {
        "cnn",
        "ame2",
        "r2plus1d",
        "ray_time",
        "ray_event_time",
        "ray_event_time_delta",
        "bank_lidar_heightmap",
    }

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        elevation_set: str = "height_scan_actor",
        cnn_observation_type: str = "elevationmap",
        depth_camera_near: float = 0.05,
        depth_camera_far: float = 6.0,
        vision_range_scale: float = 1.0,
        vision_range_max: float = 10.0,
        vision_spatial_size: tuple[int, int] = (25, 17),
        vision_feature_dim: int = 64,
        elevation_history_length: int = 5,
        cnn_hidden_dims: tuple[int, ...] | list[int] = (16, 32, 64),
        cnn_kernel_sizes: tuple[int, ...] | list[int] = (3, 3, 3),
        cnn_strides: tuple[int, ...] | list[int] = (2, 2, 2),
        cnn_history_index: int | None = None,
        prop_feature_dim: int = 64,
        prop_hidden_dims: tuple[int, ...] | list[int] = (128,),
        use_prop_encoder: bool = True,
        elevation_encoder_type: str = "cnn",
        ame2_map_extent: tuple[float, float] = (1.35, 0.95),
        ame2_history_index: int = -1,
        ame2_token_spatial_size: tuple[int, int] = (7, 5),
        ame2_local_channels: tuple[int, ...] | list[int] = (8, 16),
        ame2_position_feature_dim: int = 16,
        ame2_point_feature_dim: int = 64,
        ame2_global_feature_dim: int = 32,
        ame2_attention_dim: int = 32,
        ame2_num_heads: int = 4,
        ame2_height_scale: float = 0.6,
        ray_time_set: str | None = None,
        ray_time_history_length: int | None = None,
        ray_time_spatial_size: tuple[int, int] | None = None,
        ray_time_spatial_channels: tuple[int, ...] | list[int] = (24, 32, 64),
        ray_time_token_dim: int = 64,
        ray_time_num_heads: int = 4,
        ray_time_num_queries: int = 4,
        ray_time_min_range: float = 0.1,
        ray_time_max_range: float = 6.0,
        ray_time_vertical_fov_degrees: tuple[float, float] = (-52.0, 7.0),
        ray_time_use_query_attention: bool = True,
        ray_time_fusion_mode: str | None = None,
        ray_event_time_mode: str | None = None,
        ray_event_time_source: str = "none",
        ray_event_time_scale_s: float = 0.5,
        ray_event_delta_proprio_set: str | None = None,
        ray_event_delta_proprio_dim: int = 0,
        ray_event_training_ready: bool = False,
        r2plus1d_hidden_dims: tuple[int, ...] | list[int] = (16, 24, 44),
        r2plus1d_spatial_kernel_sizes: tuple[int, ...] | list[int] = (3, 3, 3),
        r2plus1d_temporal_kernel_sizes: tuple[int, ...] | list[int] = (3, 3, 3),
        r2plus1d_spatial_strides: tuple[int, ...] | list[int] = (2, 2, 2),
        bank_heightmap_target_contract: dict | None = None,
        bank_downstream_heightmap_contract: dict | None = None,
        bank_reconstructor_checkpoint: Mapping | None = None,
    ) -> None:
        """Initialize the proprio-elevation fusion model.

        Args:
            obs: Observation dictionary.
            obs_groups: Mapping from observation set names (e.g. ``actor`` / ``critic``) to observation-group lists.
            obs_set: Observation set used by this model instance.
            output_dim: Output dimension of the final head.
            hidden_dims: Hidden dimensions of the fusion MLP head.
            activation: Activation function used by the MLP modules.
            obs_normalization: Whether to normalize proprio observations before the proprio MLP.
            distribution_cfg: Optional output distribution configuration.
            elevation_set: CNN observation-group name used by this model instance.
            cnn_observation_type: Normalization path for the CNN observation. Supported values are
                ``"elevationmap"``, ``"depthcamera"``, and ``"inverse_depth"``.
            depth_camera_near: Near depth used by the depth-camera normalization branch.
            depth_camera_far: Far depth used by the depth-camera normalization branch.
            vision_range_scale: Distance scale used by legacy inverse-depth normalization.
            vision_range_max: Maximum distance used by legacy inverse-depth normalization.
            vision_spatial_size: Spatial size ``(H, W)`` of the elevation map.
            vision_feature_dim: Output dimension of the elevation encoder.
            elevation_history_length: Number of elevation-history frames, used as CNN input channels.
            cnn_hidden_dims: Hidden channels of the elevation CNN.
            cnn_kernel_sizes: Kernel sizes of the elevation CNN.
            cnn_strides: Strides of the elevation CNN.
            cnn_history_index: Optional history frame used by the CNN. ``None`` preserves
                the baseline multi-frame input; an integer selects exactly one frame.
            prop_feature_dim: Output feature dimension of the proprio MLP encoder.
            prop_hidden_dims: Hidden dimensions of the proprio MLP encoder.
            use_prop_encoder: Whether to pass proprioception through the encoder before fusion.
            elevation_encoder_type: Elevation encoder implementation. ``"cnn"`` preserves the baseline.
            ame2_map_extent: Metric ``(x, y)`` extent of the AME2 elevation map.
            ame2_history_index: History frame selected by the AME2 encoder.
            ame2_token_spatial_size: Spatial size of the AME2 local-token grid.
            ame2_local_channels: Local CNN channels used by the AME2 encoder.
            ame2_position_feature_dim: Dimension of pooled xyz positional features.
            ame2_point_feature_dim: Point-wise local feature dimension.
            ame2_global_feature_dim: Dimension of the pooled global map feature.
            ame2_attention_dim: Query/key/value dimension for multi-head attention.
            ame2_num_heads: Number of AME2 attention heads.
            ame2_height_scale: Scale applied after spatial mean centering the selected height map.
            ray_time_set: Optional observation-group alias used by the ray-time config.
            ray_time_history_length: Optional ray-time history-length alias.
            ray_time_spatial_size: Optional ray-time ``(H, W)`` alias.
            ray_time_spatial_channels: Per-frame circular CNN channels.
            ray_time_token_dim: Ray-time token dimension.
            ray_time_num_heads: Number of cross-attention heads.
            ray_time_num_queries: Number of proprioception-conditioned queries.
            ray_time_min_range: Minimum metric LiDAR range.
            ray_time_max_range: Maximum metric LiDAR range.
            ray_time_vertical_fov_degrees: Lower/upper elevation angles for fixed position encoding.
            ray_time_use_query_attention: Enable query attention; false is the global-only ablation.
            ray_time_fusion_mode: Optional explicit fusion mode. ``None`` preserves
                ``ray_time_use_query_attention``; supported values are ``"attention"``,
                ``"global"``, and the non-spatial ``"query_global"`` causal control.
            ray_event_delta_proprio_set: Separate ``[B,K,D,H,W]`` acquisition
                delta-proprio observation group for ``ray_event_time_delta``.
            ray_event_delta_proprio_dim: Configured semantic dimension ``D``.
            r2plus1d_hidden_dims: Output channels of the factorized R(2+1)D
                blocks. The default is parameter-matched to the default
                five-frame 2-D CNN elevation encoder.
            r2plus1d_spatial_kernel_sizes: Spatial kernel sizes of the
                factorized R(2+1)D blocks.
            r2plus1d_temporal_kernel_sizes: Temporal kernel sizes of the
                factorized R(2+1)D blocks.
            r2plus1d_spatial_strides: Spatial strides of the factorized
                R(2+1)D blocks; temporal stride remains one.
            bank_heightmap_target_contract: Semantic metadata bound to the
                Bank reconstructor output. Required only by the explicit Bank
                branch and never inferred from tensor shape.
            bank_downstream_heightmap_contract: Independently supplied input
                contract for the downstream height encoder. It must exactly
                match the reconstructed target contract.
            bank_reconstructor_checkpoint: Optional strict frozen Bank
                checkpoint. Absence creates a trainable reconstructor; presence
                must bind the same history length and semantic target contract.
        """
        super().__init__()

        self.obs_set = obs_set
        self.elevation_encoder_type = elevation_encoder_type.lower()
        if self.elevation_encoder_type not in self._ELEVATION_ENCODER_TYPES:
            raise ValueError(
                f"Unsupported elevation_encoder_type '{elevation_encoder_type}'. "
                f"Expected one of {tuple(sorted(self._ELEVATION_ENCODER_TYPES))}."
            )
        ray_encoder_types = {
            "ray_time",
            "ray_event_time",
            "ray_event_time_delta",
            "bank_lidar_heightmap",
        }
        ray_event_encoder_types = {
            "ray_event_time",
            "ray_event_time_delta",
        }
        if self.elevation_encoder_type in ray_encoder_types:
            # These aliases let the dedicated lab-side config use modality
            # names without changing the legacy fusion model/checkpoint API.
            if ray_time_set is not None:
                elevation_set = ray_time_set
            if ray_time_history_length is not None:
                elevation_history_length = int(ray_time_history_length)
            if ray_time_spatial_size is not None:
                vision_spatial_size = ray_time_spatial_size
        self.elevation_set = elevation_set
        self.cnn_observation_type = cnn_observation_type.lower()
        if self.cnn_observation_type not in self._CNN_OBSERVATION_MODES:
            raise ValueError(
                f"Unsupported cnn_observation_type '{cnn_observation_type}'. "
                f"Expected one of {tuple(self._CNN_OBSERVATION_MODES.keys())}."
            )
        if depth_camera_far <= depth_camera_near:
            raise ValueError(
                f"depth_camera_far must be greater than depth_camera_near, got {depth_camera_far} <= {depth_camera_near}."
            )
        self._cnn_observation_mode = self._CNN_OBSERVATION_MODES[self.cnn_observation_type]
        self.depth_camera_near = float(depth_camera_near)
        self.depth_camera_far = float(depth_camera_far)
        self.vision_range_scale = float(vision_range_scale)
        self.vision_range_max = float(vision_range_max)
        if self.vision_range_scale <= 0.0:
            raise ValueError(f"vision_range_scale must be positive, got {self.vision_range_scale}.")
        if self.vision_range_max <= 0.0:
            raise ValueError(f"vision_range_max must be positive, got {self.vision_range_max}.")

        self.vision_feature_dim = vision_feature_dim
        self.vision_spatial_size = tuple(vision_spatial_size)
        self.prop_feature_dim = prop_feature_dim
        self.elevation_history_length = elevation_history_length
        self.use_prop_encoder = use_prop_encoder
        self.ray_input_channels = (
            5 if self.elevation_encoder_type in ray_event_encoder_types else 2
        )
        self.ray_event_delta_proprio_set = ray_event_delta_proprio_set
        self.ray_event_delta_proprio_dim = int(ray_event_delta_proprio_dim)
        self.ray_event_training_ready = bool(ray_event_training_ready)
        if (
            self.elevation_encoder_type not in ray_event_encoder_types
            and (
                ray_event_time_mode is not None
                or ray_event_time_source != "none"
                or ray_event_delta_proprio_set is not None
                or ray_event_delta_proprio_dim != 0
                or ray_event_training_ready
            )
        ):
            raise ValueError(
                "Non-event encoders reject ray-event fields."
            )
        if self.elevation_encoder_type in ray_event_encoder_types:
            if ray_event_time_mode is None:
                raise ValueError("Ray-event encoders require ray_event_time_mode.")
            if self.ray_event_training_ready:
                raise ValueError(
                    "ray_event_training_ready remains false until the formal "
                    "64-environment smoke receipt is validated."
                )
        if self.elevation_encoder_type == "ray_event_time_delta":
            if (
                not isinstance(ray_event_delta_proprio_set, str)
                or not ray_event_delta_proprio_set
            ):
                raise ValueError(
                    "ray_event_time_delta requires a delta-proprio observation set."
                )
            if (
                isinstance(ray_event_delta_proprio_dim, bool)
                or not isinstance(ray_event_delta_proprio_dim, int)
                or ray_event_delta_proprio_dim <= 0
            ):
                raise ValueError(
                    "ray_event_time_delta requires a positive integer delta dimension."
                )
            if (
                ray_event_time_mode != "per_return_age"
                or ray_event_time_source != "livox_per_return"
            ):
                raise ValueError(
                    "ray_event_time_delta requires Livox per-return timing."
                )
        elif (
            ray_event_delta_proprio_set is not None
            or ray_event_delta_proprio_dim != 0
        ):
            raise ValueError(
                "Only ray_event_time_delta accepts acquisition delta-proprio."
            )
        self.bank_heightmap_contract: dict | None = None
        self.bank_ray_input_contract: dict | None = None
        self.bank_heightmap_spatial_size: tuple[int, int] | None = None
        self.bank_reconstructor_loaded_frozen = False
        if self.elevation_encoder_type == "bank_lidar_heightmap":
            if (
                bank_heightmap_target_contract is None
                or bank_downstream_heightmap_contract is None
            ):
                raise ValueError(
                    "bank_lidar_heightmap fails closed without both reconstructed "
                    "target and downstream heightmap contracts."
                )
            reconstructed_contract = normalize_heightmap_target_contract(
                bank_heightmap_target_contract
            )
            downstream_contract = normalize_heightmap_target_contract(
                bank_downstream_heightmap_contract
            )
            if reconstructed_contract != downstream_contract:
                raise ValueError(
                    "Bank reconstructed and downstream heightmap contracts differ."
                )
            if self.cnn_observation_type != "elevationmap":
                raise ValueError(
                    "bank_lidar_heightmap only supports elevationmap normalization."
                )
            self.bank_heightmap_contract = reconstructed_contract
            self.bank_heightmap_spatial_size = (28, 20)
            self.bank_ray_input_contract = {
                "layout": "B_K_C_H_W",
                "flatten_order": "C_contiguous_row_major_K_C_H_W",
                "channels": ["range_m", "valid"],
                "range_unit": "metre",
                "valid_semantics": "finite exact binary {0,1}",
                "history_order": "oldest_to_newest",
                "spatial_shape": [16, 96],
            }
        elif (
            bank_heightmap_target_contract is not None
            or bank_downstream_heightmap_contract is not None
            or bank_reconstructor_checkpoint is not None
        ):
            raise ValueError(
                "Bank heightmap fields are only accepted by bank_lidar_heightmap."
            )
        self._cnn_use_single_frame = cnn_history_index is not None
        self._cnn_history_index = 0
        if self._cnn_use_single_frame:
            if self.elevation_encoder_type != "cnn":
                raise ValueError("cnn_history_index is only supported by the CNN elevation encoder.")
            resolved_history_index = int(cnn_history_index)  # type: ignore[arg-type]
            if resolved_history_index < 0:
                resolved_history_index += elevation_history_length
            if resolved_history_index < 0 or resolved_history_index >= elevation_history_length:
                raise ValueError(
                    "cnn_history_index is out of range for elevation history: "
                    f"got {cnn_history_index}, history length {elevation_history_length}."
                )
            elevation_shape = obs[elevation_set].shape
            if elevation_shape[1] != elevation_history_length:
                raise ValueError(
                    "CNN elevation history mismatch: "
                    f"observation has {elevation_shape[1]} frames, config expects {elevation_history_length}."
                )
            self._cnn_history_index = resolved_history_index

        # Resolve proprio observation groups and dimension.
        self.obs_groups, self.obs_dim = self._get_prop_obs_dim(
            obs,
            obs_groups,
            obs_set,
            self.elevation_set,
            expected_perception_ndim=(
                5 if self.elevation_encoder_type in ray_encoder_types else 4
            ),
            additional_perception_sets=(
                (ray_event_delta_proprio_set,)
                if self.elevation_encoder_type == "ray_event_time_delta"
                else ()
            ),
        )

        # Observation normalization only applies to proprio inputs.
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # Distribution.
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            fusion_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            fusion_output_dim = output_dim

        # Proprio encoder.
        if use_prop_encoder:
            self.prop_mlp = MLP(self.obs_dim, prop_feature_dim, prop_hidden_dims, activation)
            fusion_prop_dim = prop_feature_dim
        else:
            self.prop_mlp = nn.Identity()
            fusion_prop_dim = self.obs_dim

        # Elevation encoder. Keep the default CNN construction byte-for-byte compatible with existing checkpoints.
        if self.elevation_encoder_type == "cnn":
            self.elevation_encoder = Elevation2DCNNEncoder(
                in_channels=1 if self._cnn_use_single_frame else elevation_history_length,
                hidden_dims=list(cnn_hidden_dims),
                kernel_sizes=list(cnn_kernel_sizes),
                strides=list(cnn_strides),
                out_dim=vision_feature_dim,
                vision_spatial_size=vision_spatial_size,
            )
        elif self.elevation_encoder_type == "r2plus1d":
            if self.cnn_observation_type != "elevationmap":
                raise ValueError(
                    "The R(2+1)D encoder only supports "
                    "cnn_observation_type='elevationmap', "
                    f"got '{cnn_observation_type}'."
                )
            elevation_shape = obs[elevation_set].shape
            expected_elevation_shape = (
                elevation_history_length,
                *self.vision_spatial_size,
            )
            if tuple(elevation_shape[1:]) != expected_elevation_shape:
                raise ValueError(
                    "R(2+1)D elevation observation shape mismatch: expected "
                    f"[B,T,H,W] with [T,H,W]={expected_elevation_shape}, "
                    f"got {tuple(elevation_shape)}."
                )
            self.elevation_encoder = R2Plus1DElevationEncoder(
                history_length=elevation_history_length,
                hidden_dims=r2plus1d_hidden_dims,
                spatial_kernel_sizes=r2plus1d_spatial_kernel_sizes,
                temporal_kernel_sizes=r2plus1d_temporal_kernel_sizes,
                spatial_strides=r2plus1d_spatial_strides,
                out_dim=vision_feature_dim,
                vision_spatial_size=vision_spatial_size,
            )
        elif self.elevation_encoder_type == "ame2":
            if self.cnn_observation_type != "elevationmap":
                raise ValueError(
                    "The AME2 encoder only supports cnn_observation_type='elevationmap', "
                    f"got '{cnn_observation_type}'."
                )
            elevation_shape = obs[elevation_set].shape
            if elevation_shape[1] != elevation_history_length:
                raise ValueError(
                    "AME2 elevation history mismatch: "
                    f"observation has {elevation_shape[1]} frames, config expects {elevation_history_length}."
                )
            if tuple(elevation_shape[-2:]) != self.vision_spatial_size:
                raise ValueError(
                    "AME2 elevation spatial shape mismatch: "
                    f"observation has {tuple(elevation_shape[-2:])}, config expects {self.vision_spatial_size}."
                )
            self.elevation_encoder = AME2Encoder(
                history_length=elevation_history_length,
                history_index=ame2_history_index,
                vision_spatial_size=vision_spatial_size,
                token_spatial_size=ame2_token_spatial_size,
                proprio_feature_dim=fusion_prop_dim,
                map_extent=ame2_map_extent,
                local_channels=ame2_local_channels,
                position_feature_dim=ame2_position_feature_dim,
                point_feature_dim=ame2_point_feature_dim,
                global_feature_dim=ame2_global_feature_dim,
                attention_dim=ame2_attention_dim,
                output_dim=vision_feature_dim,
                height_scale=ame2_height_scale,
                num_heads=ame2_num_heads,
            )
        elif self.elevation_encoder_type == "bank_lidar_heightmap":
            ray_shape = obs[elevation_set].shape
            expected_ray_shape = (
                elevation_history_length,
                2,
                *self.vision_spatial_size,
            )
            if tuple(ray_shape[1:]) != expected_ray_shape:
                raise ValueError(
                    "Bank LiDAR observation shape mismatch: expected "
                    f"[B,K,2,H,W] with [K,2,H,W]={expected_ray_shape}, "
                    f"got {tuple(ray_shape)}."
                )
            if self.vision_spatial_size != (16, 96):
                raise ValueError(
                    "bank_lidar_heightmap input spatial size must be (16,96)."
                )
            if bank_reconstructor_checkpoint is None:
                self.heightmap_reconstructor = BankLidarHeightmapReconstructor(
                    history_length=elevation_history_length,
                    target_contract=self.bank_heightmap_contract,
                )
            else:
                self.heightmap_reconstructor = (
                    load_frozen_reconstructor_checkpoint(
                        bank_reconstructor_checkpoint
                    )
                )
                if (
                    self.heightmap_reconstructor.history_length
                    != elevation_history_length
                    or self.heightmap_reconstructor.target_contract
                    != self.bank_heightmap_contract
                ):
                    raise ValueError(
                        "Frozen Bank reconstructor history/target contract mismatch."
                    )
                self.bank_reconstructor_loaded_frozen = True
            self.elevation_encoder = Elevation2DCNNEncoder(
                in_channels=1,
                hidden_dims=list(cnn_hidden_dims),
                kernel_sizes=list(cnn_kernel_sizes),
                strides=list(cnn_strides),
                out_dim=vision_feature_dim,
                vision_spatial_size=self.bank_heightmap_spatial_size,
            )
        else:
            ray_shape = obs[elevation_set].shape
            expected_ray_shape = (
                elevation_history_length,
                self.ray_input_channels,
                *self.vision_spatial_size,
            )
            if tuple(ray_shape[1:]) != expected_ray_shape:
                raise ValueError(
                    "Ray-time observation shape mismatch: expected [B,T,C,H,W] "
                    f"with [T,C,H,W]={expected_ray_shape}, got {tuple(ray_shape)}."
                )
            if self.elevation_encoder_type == "ray_event_time_delta":
                delta_shape = obs[ray_event_delta_proprio_set].shape
                expected_delta_shape = (
                    elevation_history_length,
                    ray_event_delta_proprio_dim,
                    *self.vision_spatial_size,
                )
                if tuple(delta_shape[1:]) != expected_delta_shape:
                    raise ValueError(
                        "Acquisition delta-proprio observation shape mismatch: "
                        f"expected [B,K,D,H,W] with [K,D,H,W]="
                        f"{expected_delta_shape}, got {tuple(delta_shape)}."
                    )
            self.elevation_encoder = RayTimeAttentionEncoder(
                history_length=elevation_history_length,
                vision_spatial_size=self.vision_spatial_size,
                proprio_feature_dim=fusion_prop_dim,
                output_dim=vision_feature_dim,
                spatial_channels=ray_time_spatial_channels,
                token_dim=ray_time_token_dim,
                num_heads=ray_time_num_heads,
                num_queries=ray_time_num_queries,
                min_range=ray_time_min_range,
                max_range=ray_time_max_range,
                vertical_fov_degrees=ray_time_vertical_fov_degrees,
                use_query_attention=ray_time_use_query_attention,
                fusion_mode=ray_time_fusion_mode,
                event_time_mode=(
                    ray_event_time_mode
                    if self.elevation_encoder_type in ray_event_encoder_types
                    else None
                ),
                event_time_source=(
                    ray_event_time_source
                    if self.elevation_encoder_type in ray_event_encoder_types
                    else "none"
                ),
                event_time_scale_s=ray_event_time_scale_s,
                acquisition_delta_proprio_dim=(
                    ray_event_delta_proprio_dim
                    if self.elevation_encoder_type == "ray_event_time_delta"
                    else 0
                ),
            )

        # Fusion head.
        self.mlp = MLP(fusion_prop_dim + vision_feature_dim, fusion_output_dim, hidden_dims, activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the fusion model."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build fused latent features from proprio and elevation observations."""
        proprio_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)

        elevation_obs = obs[self.elevation_set]
        if self.elevation_encoder_type in {"cnn", "r2plus1d"}:
            if self._cnn_use_single_frame:
                elevation_obs = elevation_obs[:, self._cnn_history_index : self._cnn_history_index + 1]
            elevation_obs = self._normalize_cnn_observation(elevation_obs)
            elevation_features = self.elevation_encoder(elevation_obs)
        elif self.elevation_encoder_type == "bank_lidar_heightmap":
            reconstructed_height_m = self.heightmap_reconstructor(elevation_obs)
            normalized_height = self._normalize_cnn_observation(
                reconstructed_height_m
            )
            elevation_features = self.elevation_encoder(normalized_height)
        elif self.elevation_encoder_type == "ray_event_time_delta":
            elevation_features = self.elevation_encoder(
                elevation_obs,
                proprio_features,
                obs[self.ray_event_delta_proprio_set],
            )
        else:
            elevation_features = self.elevation_encoder(elevation_obs, proprio_features)

        return torch.cat((proprio_features, elevation_features), dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def train(self, mode: bool = True) -> PropMLPElevationFusionModel:
        """Preserve an explicitly loaded frozen reconstructor in eval mode."""
        super().train(mode)
        if self.bank_reconstructor_loaded_frozen:
            self.heightmap_reconstructor.eval()
        return self

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for non-recurrent models)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if self.elevation_encoder_type == "bank_lidar_heightmap":
            raise RuntimeError(
                "Bank heightmap export remains fail-closed until its external "
                "target contract is bound by a deployment manifest."
            )
        if self.elevation_encoder_type == "ray_event_time_delta":
            return _TorchRayEventDeltaPropMLPElevationFusionModel(self)
        if self.elevation_encoder_type in {"ray_time", "ray_event_time"}:
            return _TorchRayTimePropMLPElevationFusionModel(self)
        if self.elevation_encoder_type == "ame2":
            return _TorchAME2PropMLPElevationFusionModel(self)
        return _TorchPropMLPElevationFusionModel(self)

    def as_onnx(self, verbose: bool, input_mode: str = "split") -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        if self.elevation_encoder_type == "bank_lidar_heightmap":
            raise RuntimeError(
                "Bank heightmap export remains fail-closed until its external "
                "target contract is bound by a deployment manifest."
            )
        if self.elevation_encoder_type == "ray_event_time_delta":
            return _OnnxRayEventDeltaPropMLPElevationFusionModel(
                self,
                verbose,
                input_mode,
            )
        if self.elevation_encoder_type in {"ray_time", "ray_event_time"}:
            return _OnnxRayTimePropMLPElevationFusionModel(self, verbose, input_mode)
        if self.elevation_encoder_type == "ame2":
            return _OnnxAME2PropMLPElevationFusionModel(self, verbose, input_mode)
        return _OnnxPropMLPElevationFusionModel(self, verbose, input_mode)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update proprio observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            proprio_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(proprio_obs)  # type: ignore

    def preflight_bank_heightmap_observation(
        self,
        obs: TensorDict,
    ) -> dict:
        """Run explicit synchronized Bank source validation outside collection."""
        if self.elevation_encoder_type != "bank_lidar_heightmap":
            raise RuntimeError(
                "Bank observation preflight requires bank_lidar_heightmap."
            )
        audit = preflight_validate_lidar_history(
            obs[self.elevation_set],
            history_length=self.elevation_history_length,
        )
        return {
            **audit,
            "actor_observation_contract": dict(self.bank_ray_input_contract),
            "heightmap_contract": dict(self.bank_heightmap_contract),
        }

    def bank_heightmap_parameter_audit(self) -> dict:
        """Return an offline parameter/schema audit for the explicit H0b branch."""
        if self.elevation_encoder_type != "bank_lidar_heightmap":
            raise RuntimeError(
                "Bank parameter audit requires bank_lidar_heightmap."
            )

        def count(module: nn.Module, *, trainable_only: bool = False) -> int:
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if not trainable_only or parameter.requires_grad
            )

        return {
            "reconstructor_parameter_count": count(
                self.heightmap_reconstructor
            ),
            "reconstructor_trainable_parameter_count": count(
                self.heightmap_reconstructor,
                trainable_only=True,
            ),
            "reconstructor_training": self.heightmap_reconstructor.training,
            "reconstructor_loaded_frozen": (
                self.bank_reconstructor_loaded_frozen
            ),
            "downstream_elevation_encoder_parameter_count": count(
                self.elevation_encoder
            ),
            "total_model_parameter_count": count(self),
            "total_model_trainable_parameter_count": count(
                self,
                trainable_only=True,
            ),
            "reconstructor_checkpoint_schema": (
                reconstructor_checkpoint_schema(
                    self.heightmap_reconstructor
                )
            ),
            "heightmap_contract": dict(self.bank_heightmap_contract),
        }

    def _get_prop_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        elevation_set: str,
        expected_perception_ndim: int = 4,
        additional_perception_sets: tuple[str, ...] = (),
    ) -> tuple[list[str], int]:
        """Select proprio observation groups and compute their flattened dimension."""
        active_obs_groups = []
        obs_dim = 0
        for obs_group in obs_groups[obs_set]:
            if (
                obs_group == elevation_set
                or obs_group in additional_perception_sets
            ):
                continue
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The proprio branch only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
            active_obs_groups.append(obs_group)
            obs_dim += obs[obs_group].shape[-1]

        if len(obs[elevation_set].shape) != expected_perception_ndim:
            expected_layout = (
                "[B, T, C, H, W]"
                if expected_perception_ndim == 5
                else "[B, T, H, W]"
            )
            raise ValueError(
                f"The perception branch expects a {expected_perception_ndim}D tensor "
                f"{expected_layout}, got shape {obs[elevation_set].shape} "
                f"for '{elevation_set}'."
            )
        return active_obs_groups, obs_dim

    def _normalize_cnn_observation(self, cnn_observation: torch.Tensor) -> torch.Tensor:
        """Normalize the CNN observation according to its modality."""
        if self._cnn_observation_mode == 0:
            elevation_mean = cnn_observation.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((cnn_observation - elevation_mean) / 0.6, -3.0, 3.0)

        if self._cnn_observation_mode == 2:
            distance_map = torch.clamp(cnn_observation, min=0.0, max=self.vision_range_max)
            return 2.0 / (1.0 + distance_map / self.vision_range_scale) - 1.0

        invalid_mask = (~torch.isfinite(cnn_observation)) | (cnn_observation <= 0.0)
        cnn_observation = torch.where(
            invalid_mask,
            torch.full_like(cnn_observation, self.depth_camera_far),
            cnn_observation,
        )
        cnn_observation = torch.clamp(cnn_observation, self.depth_camera_near, self.depth_camera_far)
        cnn_observation = (cnn_observation - self.depth_camera_near) / (
            self.depth_camera_far - self.depth_camera_near
        )
        return cnn_observation * 2.0 - 1.0


class _TorchPropMLPElevationFusionModel(nn.Module):
    """Exportable fusion model for Torch JIT."""

    def __init__(self, model: PropMLPElevationFusionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self._cnn_observation_mode = model._cnn_observation_mode
        self.depth_camera_near = model.depth_camera_near
        self.depth_camera_far = model.depth_camera_far
        self.vision_range_scale = model.vision_range_scale
        self.vision_range_max = model.vision_range_max
        self.use_single_frame = model._cnn_use_single_frame
        self.history_index = model._cnn_history_index
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, proprio_obs: torch.Tensor, elevation_obs: torch.Tensor) -> torch.Tensor:
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        if self.use_single_frame:
            elevation_obs = elevation_obs[:, self.history_index : self.history_index + 1]
        elevation_obs = self._normalize_cnn_observation(elevation_obs)
        elevation_features = self.elevation_encoder(elevation_obs)
        fused_features = torch.cat((proprio_features, elevation_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def _normalize_cnn_observation(self, cnn_observation: torch.Tensor) -> torch.Tensor:
        if self._cnn_observation_mode == 0:
            elevation_mean = cnn_observation.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((cnn_observation - elevation_mean) / 0.6, -3.0, 3.0)

        if self._cnn_observation_mode == 2:
            distance_map = torch.clamp(cnn_observation, min=0.0, max=self.vision_range_max)
            return 2.0 / (1.0 + distance_map / self.vision_range_scale) - 1.0

        invalid_mask = (~torch.isfinite(cnn_observation)) | (cnn_observation <= 0.0)
        cnn_observation = torch.where(
            invalid_mask,
            torch.full_like(cnn_observation, self.depth_camera_far),
            cnn_observation,
        )
        cnn_observation = torch.clamp(cnn_observation, self.depth_camera_near, self.depth_camera_far)
        cnn_observation = (cnn_observation - self.depth_camera_near) / (
            self.depth_camera_far - self.depth_camera_near
        )
        return cnn_observation * 2.0 - 1.0

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchAME2PropMLPElevationFusionModel(nn.Module):
    """Exportable AME2 actor for Torch JIT.

    Unlike the CNN wrapper, raw elevation history is passed directly to the
    AME2 encoder together with the encoded proprioception features.
    """

    def __init__(self, model: PropMLPElevationFusionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, proprio_obs: torch.Tensor, elevation_obs: torch.Tensor) -> torch.Tensor:
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        elevation_features = self.elevation_encoder(elevation_obs, proprio_features)
        fused_features = torch.cat((proprio_features, elevation_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchRayTimePropMLPElevationFusionModel(nn.Module):
    """Exportable Ray-Time actor for Torch JIT.

    The deployment interface accepts proprioception as ``float32`` and the
    Ray-Time history as either ``float16`` (matching rollout storage) or
    ``float32``. The encoder promotes the history to ``float32`` internally.
    """

    def __init__(self, model: PropMLPElevationFusionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, proprio_obs: torch.Tensor, ray_history: torch.Tensor) -> torch.Tensor:
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        ray_features = self.elevation_encoder(ray_history, proprio_features)
        fused_features = torch.cat((proprio_features, ray_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchRayEventDeltaPropMLPElevationFusionModel(nn.Module):
    """TorchScript actor with a separate acquisition delta-proprio tensor."""

    def __init__(self, model: PropMLPElevationFusionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self,
        proprio_obs: torch.Tensor,
        ray_history: torch.Tensor,
        acquisition_delta_proprio: torch.Tensor,
    ) -> torch.Tensor:
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        ray_features = self.elevation_encoder(
            ray_history,
            proprio_features,
            acquisition_delta_proprio,
        )
        fused_features = torch.cat((proprio_features, ray_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxPropMLPElevationFusionModel(nn.Module):
    """Exportable fusion model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: PropMLPElevationFusionModel, verbose: bool, input_mode: str = "split") -> None:
        super().__init__()
        if input_mode not in ("split", "single"):
            raise ValueError(f"Unsupported ONNX input mode: {input_mode}")
        self.verbose = verbose
        self.input_mode = input_mode
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self._cnn_observation_mode = model._cnn_observation_mode
        self.depth_camera_near = model.depth_camera_near
        self.depth_camera_far = model.depth_camera_far
        self.vision_range_scale = model.vision_range_scale
        self.vision_range_max = model.vision_range_max
        self.proprio_input_size = model.obs_dim
        self.elevation_history_length = model.elevation_history_length
        self.vision_spatial_size = model.vision_spatial_size
        self.use_single_frame = model._cnn_use_single_frame
        self.history_index = model._cnn_history_index

    def forward(self, proprio_obs: torch.Tensor, elevation_obs: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_mode == "single":
            obs = proprio_obs
            elevation_size = self.elevation_history_length * self.vision_spatial_size[0] * self.vision_spatial_size[1]
            proprio_obs = obs[:, :self.proprio_input_size]
            elevation_obs = obs[:, self.proprio_input_size:self.proprio_input_size + elevation_size]
            if torch.compiler.is_compiling():
                # The Dynamo ONNX exporter preserves the symbolic batch through unflatten.
                elevation_obs = elevation_obs.unflatten(
                    1,
                    (
                        self.elevation_history_length,
                        self.vision_spatial_size[0],
                        self.vision_spatial_size[1],
                    ),
                )
            else:
                # Eager and legacy ONNX paths must infer, rather than capture, batch size.
                elevation_obs = elevation_obs.reshape(
                    -1,
                    self.elevation_history_length,
                    self.vision_spatial_size[0],
                    self.vision_spatial_size[1],
                )
        elif elevation_obs is None:
            raise ValueError("elevation_obs is required when ONNX input_mode='split'")

        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        if self.use_single_frame:
            elevation_obs = elevation_obs[:, self.history_index : self.history_index + 1]
        elevation_obs = self._normalize_cnn_observation(elevation_obs)
        elevation_features = self.elevation_encoder(elevation_obs)
        fused_features = torch.cat((proprio_features, elevation_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def _normalize_cnn_observation(self, cnn_observation: torch.Tensor) -> torch.Tensor:
        if self._cnn_observation_mode == 0:
            elevation_mean = cnn_observation.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((cnn_observation - elevation_mean) / 0.6, -3.0, 3.0)

        if self._cnn_observation_mode == 2:
            distance_map = torch.clamp(cnn_observation, min=0.0, max=self.vision_range_max)
            return 2.0 / (1.0 + distance_map / self.vision_range_scale) - 1.0

        invalid_mask = (~torch.isfinite(cnn_observation)) | (cnn_observation <= 0.0)
        cnn_observation = torch.where(
            invalid_mask,
            torch.full_like(cnn_observation, self.depth_camera_far),
            cnn_observation,
        )
        cnn_observation = torch.clamp(cnn_observation, self.depth_camera_near, self.depth_camera_far)
        cnn_observation = (cnn_observation - self.depth_camera_near) / (
            self.depth_camera_far - self.depth_camera_near
        )
        return cnn_observation * 2.0 - 1.0

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_mode == "single":
            elevation_size = self.elevation_history_length * self.vision_spatial_size[0] * self.vision_spatial_size[1]
            return (torch.zeros(1, self.proprio_input_size + elevation_size),)
        return (
            torch.zeros(1, self.proprio_input_size),
            torch.zeros(1, self.elevation_history_length, *self.vision_spatial_size),
        )

    @property
    def input_names(self) -> list[str]:
        if self.input_mode == "single":
            return ["obs"]
        return ["proprio_obs", "elevation_obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        if self.input_mode == "single":
            return {"obs": {0: "batch_size"}, "actions": {0: "batch_size"}}
        return {
            "proprio_obs": {0: "batch_size"},
            "elevation_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }


class _OnnxAME2PropMLPElevationFusionModel(nn.Module):
    """Exportable AME2 actor for split-input and flat single-input ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: PropMLPElevationFusionModel, verbose: bool, input_mode: str = "split") -> None:
        super().__init__()
        if input_mode not in ("split", "single"):
            raise ValueError(f"Unsupported ONNX input mode: {input_mode}")
        self.verbose = verbose
        self.input_mode = input_mode
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.proprio_input_size = model.obs_dim
        self.elevation_history_length = model.elevation_history_length
        self.vision_spatial_size = model.vision_spatial_size

    def forward(self, proprio_obs: torch.Tensor, elevation_obs: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_mode == "single":
            obs = proprio_obs
            elevation_size = self.elevation_history_length * self.vision_spatial_size[0] * self.vision_spatial_size[1]
            proprio_obs = obs[:, :self.proprio_input_size]
            elevation_obs = obs[:, self.proprio_input_size:self.proprio_input_size + elevation_size]
            if torch.compiler.is_compiling():
                # The Dynamo ONNX exporter preserves the symbolic batch through unflatten.
                elevation_obs = elevation_obs.unflatten(
                    1,
                    (
                        self.elevation_history_length,
                        self.vision_spatial_size[0],
                        self.vision_spatial_size[1],
                    ),
                )
            else:
                # Eager and legacy ONNX paths must infer, rather than capture, batch size.
                elevation_obs = elevation_obs.reshape(
                    -1,
                    self.elevation_history_length,
                    self.vision_spatial_size[0],
                    self.vision_spatial_size[1],
                )
        elif elevation_obs is None:
            raise ValueError("elevation_obs is required when ONNX input_mode='split'")

        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        elevation_features = self.elevation_encoder(elevation_obs, proprio_features)
        fused_features = torch.cat((proprio_features, elevation_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.input_mode == "single":
            elevation_size = self.elevation_history_length * self.vision_spatial_size[0] * self.vision_spatial_size[1]
            return (torch.zeros(1, self.proprio_input_size + elevation_size),)
        return (
            torch.zeros(1, self.proprio_input_size),
            torch.zeros(1, self.elevation_history_length, *self.vision_spatial_size),
        )

    @property
    def input_names(self) -> list[str]:
        if self.input_mode == "single":
            return ["obs"]
        return ["proprio_obs", "elevation_obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        if self.input_mode == "single":
            return {"obs": {0: "batch_size"}, "actions": {0: "batch_size"}}
        return {
            "proprio_obs": {0: "batch_size"},
            "elevation_obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }


class _OnnxRayTimePropMLPElevationFusionModel(nn.Module):
    """Exportable Ray-Time actor for split-input and flat single-input ONNX.

    ONNX inputs are ``float32``. In ``single`` mode the exact feature layout is
    ``[proprio, ray_history.flatten()]``, where ``ray_history`` has logical
    shape ``[K, C, H, W]`` in row-major order. Legacy Ray-Time uses ``C=2``;
    the explicit Ray-Event contract uses ``C=5``.
    """

    is_recurrent: bool = False

    def __init__(self, model: PropMLPElevationFusionModel, verbose: bool, input_mode: str = "split") -> None:
        super().__init__()
        if input_mode not in ("split", "single"):
            raise ValueError(f"Unsupported ONNX input mode: {input_mode}")
        self.verbose = verbose
        self.input_mode = input_mode
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.proprio_input_size = model.obs_dim
        self.ray_history_length = model.elevation_history_length
        self.ray_input_channels = model.ray_input_channels
        self.vision_spatial_size = model.vision_spatial_size
        self.single_input_size = (
            self.proprio_input_size
            + self.ray_history_length
            * self.ray_input_channels
            * self.vision_spatial_size[0]
            * self.vision_spatial_size[1]
        )

    def forward(self, proprio_obs: torch.Tensor, ray_history: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_mode == "single":
            obs = proprio_obs
            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and (obs.ndim != 2 or obs.shape[1] != self.single_input_size)
            ):
                raise ValueError(
                    "Single-input Ray-Time observations must have shape "
                    f"[B, {self.single_input_size}] with layout "
                    "[proprio, flatten(K, C, H, W)]."
                )
            ray_size = (
                self.ray_history_length
                * self.ray_input_channels
                * self.vision_spatial_size[0]
                * self.vision_spatial_size[1]
            )
            proprio_obs = obs[:, :self.proprio_input_size]
            ray_history = obs[:, self.proprio_input_size:self.proprio_input_size + ray_size]
            if torch.compiler.is_compiling():
                ray_history = ray_history.unflatten(
                    1,
                    (
                        self.ray_history_length,
                        self.ray_input_channels,
                        self.vision_spatial_size[0],
                        self.vision_spatial_size[1],
                    ),
                )
            else:
                ray_history = ray_history.reshape(
                    -1,
                    self.ray_history_length,
                    self.ray_input_channels,
                    self.vision_spatial_size[0],
                    self.vision_spatial_size[1],
                )
        elif ray_history is None:
            raise ValueError("ray_history is required when ONNX input_mode='split'")

        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        ray_features = self.elevation_encoder(ray_history, proprio_features)
        fused_features = torch.cat((proprio_features, ray_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.input_mode == "single":
            return (torch.zeros(1, self.single_input_size),)
        return (
            torch.zeros(1, self.proprio_input_size),
            torch.zeros(
                1,
                self.ray_history_length,
                self.ray_input_channels,
                *self.vision_spatial_size,
            ),
        )

    @property
    def input_names(self) -> list[str]:
        if self.input_mode == "single":
            return ["obs"]
        return ["proprio_obs", "ray_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        if self.input_mode == "single":
            return {"obs": {0: "batch_size"}, "actions": {0: "batch_size"}}
        return {
            "proprio_obs": {0: "batch_size"},
            "ray_history": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }


class _OnnxRayEventDeltaPropMLPElevationFusionModel(nn.Module):
    """ONNX actor with independent event and acquisition-state tensors."""

    is_recurrent: bool = False

    def __init__(
        self,
        model: PropMLPElevationFusionModel,
        verbose: bool,
        input_mode: str = "split",
    ) -> None:
        super().__init__()
        if input_mode not in ("split", "single"):
            raise ValueError(f"Unsupported ONNX input mode: {input_mode}")
        self.verbose = verbose
        self.input_mode = input_mode
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()
        self.proprio_input_size = model.obs_dim
        self.ray_history_length = model.elevation_history_length
        self.ray_input_channels = model.ray_input_channels
        self.delta_proprio_dim = model.ray_event_delta_proprio_dim
        self.vision_spatial_size = model.vision_spatial_size
        self.ray_size = (
            self.ray_history_length
            * self.ray_input_channels
            * self.vision_spatial_size[0]
            * self.vision_spatial_size[1]
        )
        self.delta_size = (
            self.ray_history_length
            * self.delta_proprio_dim
            * self.vision_spatial_size[0]
            * self.vision_spatial_size[1]
        )
        self.single_input_size = (
            self.proprio_input_size + self.ray_size + self.delta_size
        )

    def forward(
        self,
        proprio_obs: torch.Tensor,
        ray_history: torch.Tensor | None = None,
        acquisition_delta_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_mode == "single":
            obs = proprio_obs
            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and (obs.ndim != 2 or obs.shape[1] != self.single_input_size)
            ):
                raise ValueError(
                    "Single-input Ray-Event-Delta observations have the wrong shape."
                )
            proprio_obs = obs[:, : self.proprio_input_size]
            ray_start = self.proprio_input_size
            delta_start = ray_start + self.ray_size
            ray_history = obs[:, ray_start:delta_start]
            acquisition_delta_proprio = obs[
                :, delta_start : delta_start + self.delta_size
            ]
            ray_shape = (
                self.ray_history_length,
                self.ray_input_channels,
                self.vision_spatial_size[0],
                self.vision_spatial_size[1],
            )
            delta_shape = (
                self.ray_history_length,
                self.delta_proprio_dim,
                self.vision_spatial_size[0],
                self.vision_spatial_size[1],
            )
            if torch.compiler.is_compiling():
                ray_history = ray_history.unflatten(1, ray_shape)
                acquisition_delta_proprio = (
                    acquisition_delta_proprio.unflatten(1, delta_shape)
                )
            else:
                ray_history = ray_history.reshape(-1, *ray_shape)
                acquisition_delta_proprio = (
                    acquisition_delta_proprio.reshape(-1, *delta_shape)
                )
        elif ray_history is None or acquisition_delta_proprio is None:
            raise ValueError(
                "Split Ray-Event-Delta export requires ray history and "
                "acquisition delta-proprio."
            )

        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        ray_features = self.elevation_encoder(
            ray_history,
            proprio_features,
            acquisition_delta_proprio,
        )
        fused_features = torch.cat((proprio_features, ray_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.input_mode == "single":
            return (torch.zeros(1, self.single_input_size),)
        return (
            torch.zeros(1, self.proprio_input_size),
            torch.zeros(
                1,
                self.ray_history_length,
                self.ray_input_channels,
                *self.vision_spatial_size,
            ),
            torch.zeros(
                1,
                self.ray_history_length,
                self.delta_proprio_dim,
                *self.vision_spatial_size,
            ),
        )

    @property
    def input_names(self) -> list[str]:
        if self.input_mode == "single":
            return ["obs"]
        return [
            "proprio_obs",
            "ray_history",
            "acquisition_delta_proprio",
        ]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        if self.input_mode == "single":
            return {"obs": {0: "batch_size"}, "actions": {0: "batch_size"}}
        return {
            "proprio_obs": {0: "batch_size"},
            "ray_history": {0: "batch_size"},
            "acquisition_delta_proprio": {0: "batch_size"},
            "actions": {0: "batch_size"},
        }
