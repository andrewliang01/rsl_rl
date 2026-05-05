# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder
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
        vision_spatial_size: tuple[int, int] = (25, 17),
        vision_feature_dim: int = 64,
        elevation_history_length: int = 5,
        cnn_hidden_dims: tuple[int, ...] | list[int] = (16, 32, 64),
        cnn_kernel_sizes: tuple[int, ...] | list[int] = (3, 3, 3),
        cnn_strides: tuple[int, ...] | list[int] = (2, 2, 2),
        prop_feature_dim: int = 64,
        prop_hidden_dims: tuple[int, ...] | list[int] = (128,),
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
                ``"elevationmap"`` and ``"depthcamera"``.
            depth_camera_near: Near depth used by the depth-camera normalization branch.
            depth_camera_far: Far depth used by the depth-camera normalization branch.
            vision_spatial_size: Spatial size ``(H, W)`` of the elevation map.
            vision_feature_dim: Output dimension of the elevation encoder.
            elevation_history_length: Number of elevation-history frames, used as CNN input channels.
            cnn_hidden_dims: Hidden channels of the elevation CNN.
            cnn_kernel_sizes: Kernel sizes of the elevation CNN.
            cnn_strides: Strides of the elevation CNN.
            prop_feature_dim: Output feature dimension of the proprio MLP encoder.
            prop_hidden_dims: Hidden dimensions of the proprio MLP encoder.
        """
        super().__init__()

        self.obs_set = obs_set
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

        self.vision_feature_dim = vision_feature_dim
        self.vision_spatial_size = tuple(vision_spatial_size)
        self.prop_feature_dim = prop_feature_dim
        self.elevation_history_length = elevation_history_length

        # Resolve proprio observation groups and dimension.
        self.obs_groups, self.obs_dim = self._get_prop_obs_dim(obs, obs_groups, obs_set, self.elevation_set)

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
        self.prop_mlp = MLP(self.obs_dim, prop_feature_dim, prop_hidden_dims, activation)

        # Elevation encoder.
        self.elevation_encoder = Elevation2DCNNEncoder(
            in_channels=elevation_history_length,
            hidden_dims=list(cnn_hidden_dims),
            kernel_sizes=list(cnn_kernel_sizes),
            strides=list(cnn_strides),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size,
        )

        # Fusion head.
        self.mlp = MLP(prop_feature_dim + vision_feature_dim, fusion_output_dim, hidden_dims, activation)

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
        elevation_obs = self._normalize_cnn_observation(elevation_obs)
        elevation_features = self.elevation_encoder(elevation_obs)

        return torch.cat((proprio_features, elevation_features), dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

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
        return _TorchPropMLPElevationFusionModel(self)

    def as_onnx(self, verbose: bool, input_mode: str = "split") -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxPropMLPElevationFusionModel(self, verbose, input_mode)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update proprio observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            proprio_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(proprio_obs)  # type: ignore

    def _get_prop_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, elevation_set: str
    ) -> tuple[list[str], int]:
        """Select proprio observation groups and compute their flattened dimension."""
        active_obs_groups = []
        obs_dim = 0
        for obs_group in obs_groups[obs_set]:
            if obs_group == elevation_set:
                continue
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The proprio branch only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
            active_obs_groups.append(obs_group)
            obs_dim += obs[obs_group].shape[-1]

        if len(obs[elevation_set].shape) != 4:
            raise ValueError(
                f"The elevation branch expects a 4D tensor [B, T, H, W], got shape {obs[elevation_set].shape} "
                f"for '{elevation_set}'."
            )
        return active_obs_groups, obs_dim

    def _normalize_cnn_observation(self, cnn_observation: torch.Tensor) -> torch.Tensor:
        """Normalize the CNN observation according to its modality."""
        if self._cnn_observation_mode == 0:
            elevation_mean = cnn_observation.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((cnn_observation - elevation_mean) / 0.6, -3.0, 3.0)

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
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, proprio_obs: torch.Tensor, elevation_obs: torch.Tensor) -> torch.Tensor:
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        elevation_obs = self._normalize_cnn_observation(elevation_obs)
        elevation_features = self.elevation_encoder(elevation_obs)
        fused_features = torch.cat((proprio_features, elevation_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def _normalize_cnn_observation(self, cnn_observation: torch.Tensor) -> torch.Tensor:
        if self._cnn_observation_mode == 0:
            elevation_mean = cnn_observation.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((cnn_observation - elevation_mean) / 0.6, -3.0, 3.0)

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
        self.proprio_input_size = model.obs_dim
        self.elevation_history_length = model.elevation_history_length
        self.vision_spatial_size = model.vision_spatial_size

    def forward(self, proprio_obs: torch.Tensor, elevation_obs: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_mode == "single":
            obs = proprio_obs
            batch_size = obs.shape[0]
            elevation_size = self.elevation_history_length * self.vision_spatial_size[0] * self.vision_spatial_size[1]
            proprio_obs = obs[:, :self.proprio_input_size]
            elevation_obs = obs[:, self.proprio_input_size:self.proprio_input_size + elevation_size]
            elevation_obs = elevation_obs.reshape(
                batch_size, self.elevation_history_length, self.vision_spatial_size[0], self.vision_spatial_size[1]
            )
        elif elevation_obs is None:
            raise ValueError("elevation_obs is required when ONNX input_mode='split'")

        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)
        elevation_obs = self._normalize_cnn_observation(elevation_obs)
        elevation_features = self.elevation_encoder(elevation_obs)
        fused_features = torch.cat((proprio_features, elevation_features), dim=-1)
        out = self.mlp(fused_features)
        return self.deterministic_output(out)

    def _normalize_cnn_observation(self, cnn_observation: torch.Tensor) -> torch.Tensor:
        if self._cnn_observation_mode == 0:
            elevation_mean = cnn_observation.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((cnn_observation - elevation_mean) / 0.6, -3.0, 3.0)

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
