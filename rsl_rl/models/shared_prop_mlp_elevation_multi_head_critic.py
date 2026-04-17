from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder
from rsl_rl.utils import unpad_trajectories


class SharedPropMLPElevationMultiHeadCritic(nn.Module):
    """Shared proprio/elevation critic with multiple value heads.

    The expensive observation encoders are evaluated once, then separate value
    heads predict one value per reward group.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_heads: int = 4,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        elevation_set: str = "height_scan_critic",
        vision_spatial_size: tuple[int, int] = (28, 20),
        vision_feature_dim: int = 64,
        elevation_history_length: int = 1,
        cnn_hidden_dims: tuple[int, ...] | list[int] = (16, 32, 64),
        cnn_kernel_sizes: tuple[int, ...] | list[int] = (3, 3),
        cnn_strides: tuple[int, ...] | list[int] = (2, 2),
        prop_feature_dim: int = 64,
        prop_hidden_dims: tuple[int, ...] | list[int] = (128,),
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        del output_dim, distribution_cfg  # The output dimension is determined by num_heads.

        self.obs_set = obs_set
        self.elevation_set = elevation_set
        self.num_heads = num_heads
        self.vision_spatial_size = tuple(vision_spatial_size)
        self.elevation_history_length = elevation_history_length

        self.obs_groups, self.obs_dim = self._get_prop_obs_dim(obs, obs_groups, obs_set, elevation_set)
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.prop_mlp = MLP(self.obs_dim, prop_feature_dim, prop_hidden_dims, activation)
        self.elevation_encoder = Elevation2DCNNEncoder(
            in_channels=elevation_history_length,
            hidden_dims=list(cnn_hidden_dims),
            kernel_sizes=list(cnn_kernel_sizes),
            strides=list(cnn_strides),
            out_dim=vision_feature_dim,
            vision_spatial_size=vision_spatial_size,
        )
        self.shared_mlp = MLP(prop_feature_dim + vision_feature_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        self.value_heads = nn.ModuleList([nn.Linear(hidden_dims[-1], 1) for _ in range(num_heads)])

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs)
        return torch.cat([head(latent) for head in self.value_heads], dim=-1)

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        proprio_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        proprio_obs = self.obs_normalizer(proprio_obs)
        proprio_features = self.prop_mlp(proprio_obs)

        elevation_obs = self._normalize_elevation_map(obs[self.elevation_set])
        elevation_features = self.elevation_encoder(elevation_obs)

        return self.shared_mlp(torch.cat((proprio_features, elevation_features), dim=-1))

    def update_normalization(self, obs: TensorDict) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            proprio_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(proprio_obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def _get_prop_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, elevation_set: str
    ) -> tuple[list[str], int]:
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
                f"The elevation branch expects [B, T, H, W], got {obs[elevation_set].shape} for '{elevation_set}'."
            )
        return active_obs_groups, obs_dim

    def _normalize_elevation_map(self, elevation_map: torch.Tensor) -> torch.Tensor:
        elevation_mean = elevation_map.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((elevation_map - elevation_mean) / 0.6, -3.0, 3.0)
