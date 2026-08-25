# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ECMM elevation fusion actor with a UniFP base-twist adaptation encoder."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP

from .prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel


class PropMLPElevationUniFPFusionModel(PropMLPElevationFusionModel):
    """Fuse ECMM proprio/depth features with a UniFP 6D twist estimate.

    The ECMM branch encodes the current proprioceptive frame and elevation or
    depth history. A separate UniFP history encoder produces an adaptation
    latent, while its decoder estimates body-frame linear and angular velocity.
    Both the latent and explicit six-dimensional estimate are consumed by the
    locomotion head.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_set: str = "estimator_history",
        history_length: int = 32,
        estimator_latent_dim: int = 64,
        estimator_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        num_pred_obs: int = 6,
        history_normalization: bool = True,
        **kwargs,
    ) -> None:
        if history_set not in obs:
            raise KeyError(f"Missing UniFP history observation group '{history_set}'.")
        if len(obs[history_set].shape) != 2:
            raise ValueError(f"UniFP history must be [B,D], got {tuple(obs[history_set].shape)}.")
        if num_pred_obs != 6:
            raise ValueError(
                "The ECMM-UniFP decoder must predict exactly six values: "
                "base linear velocity xyz and base angular velocity xyz."
            )

        active_groups = list(obs_groups[obs_set])
        if history_set not in active_groups:
            raise ValueError(
                f"Actor observation set '{obs_set}' must include history group '{history_set}'."
            )
        base_obs_groups = copy.deepcopy(obs_groups)
        base_obs_groups[obs_set] = [name for name in active_groups if name != history_set]

        hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        prop_feature_dim = int(kwargs.get("prop_feature_dim", 64))
        vision_feature_dim = int(kwargs.get("vision_feature_dim", 64))
        use_prop_encoder = bool(kwargs.get("use_prop_encoder", True))

        super().__init__(
            obs=obs,
            obs_groups=base_obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            **kwargs,
        )

        self.history_set = history_set
        self.history_length = int(history_length)
        self.history_dim = int(obs[history_set].shape[-1])
        if self.history_dim % self.history_length != 0:
            raise ValueError(
                f"History dim {self.history_dim} is not divisible by history_length "
                f"{self.history_length}."
            )
        self.num_history_frame_obs = self.history_dim // self.history_length
        self.estimator_latent_dim = int(estimator_latent_dim)
        self.num_pred_obs = int(num_pred_obs)
        self.history_normalization = bool(history_normalization)
        self.history_normalizer: nn.Module
        if self.history_normalization:
            self.history_normalizer = EmpiricalNormalization(self.history_dim)
        else:
            self.history_normalizer = nn.Identity()

        # These names intentionally match UniFPAdaptationPPO's optimizer
        # contract, so no task-local algorithm fork is needed.
        self.encoder = MLP(
            self.history_dim,
            self.estimator_latent_dim,
            estimator_hidden_dims,
            activation,
        )
        self.decoder = MLP(
            self.estimator_latent_dim,
            self.num_pred_obs,
            decoder_hidden_dims,
            activation,
        )

        base_fusion_dim = (
            prop_feature_dim if use_prop_encoder else self.obs_dim
        ) + vision_feature_dim
        actor_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MLP(
            base_fusion_dim + self.estimator_latent_dim + self.num_pred_obs,
            actor_output_dim,
            hidden_dims,
            activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _history_latent_and_twist(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.history_normalizer(obs[self.history_set])
        latent = self.encoder(history)
        estimated_twist = self.decoder(latent)
        return latent, estimated_twist

    def predict_obs_pred(self, obs: TensorDict) -> torch.Tensor:
        """Return ``[v_x, v_y, v_z, w_x, w_y, w_z]`` for UniFP supervision."""
        _, estimated_twist = self._history_latent_and_twist(obs)
        return estimated_twist

    def get_estimated_base_twist(self, obs: TensorDict) -> torch.Tensor:
        """Expose the estimated 6D body twist for diagnostics."""
        return self.predict_obs_pred(obs)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        ecmm_features = super().get_latent(obs, masks, hidden_state)
        estimator_latent, estimated_twist = self._history_latent_and_twist(obs)
        return torch.cat((ecmm_features, estimator_latent, estimated_twist), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        super().update_normalization(obs)
        if self.history_normalization:
            self.history_normalizer.update(obs[self.history_set])  # type: ignore[attr-defined]

    def as_jit(self) -> nn.Module:
        return _TorchPropMLPElevationUniFPFusionModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        del verbose
        return _TorchPropMLPElevationUniFPFusionModel(self)


class _TorchPropMLPElevationUniFPFusionModel(nn.Module):
    """Deterministic deployment wrapper with three explicit sensor inputs."""

    is_recurrent: bool = False

    def __init__(self, model: PropMLPElevationUniFPFusionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.decoder = copy.deepcopy(model.decoder)
        self.elevation_encoder = copy.deepcopy(model.elevation_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self._cnn_observation_mode = model._cnn_observation_mode
        self.depth_camera_near = model.depth_camera_near
        self.depth_camera_far = model.depth_camera_far
        self.current_input_size = model.obs_dim
        self.history_input_size = model.history_dim
        self.elevation_history_length = model.elevation_history_length
        self.vision_spatial_size = model.vision_spatial_size
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self,
        current_proprio: torch.Tensor,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
    ) -> torch.Tensor:
        current_features = self.prop_mlp(self.obs_normalizer(current_proprio))
        normalized_history = self.history_normalizer(proprio_history)
        estimator_latent = self.encoder(normalized_history)
        estimated_twist = self.decoder(estimator_latent)
        depth_features = self.elevation_encoder(self._normalize_depth(depth_history))
        fused = torch.cat(
            (current_features, depth_features, estimator_latent, estimated_twist),
            dim=-1,
        )
        return self.deterministic_output(self.mlp(fused))

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        invalid = (~torch.isfinite(depth)) | (depth <= 0.0)
        depth = torch.where(invalid, torch.full_like(depth, self.depth_camera_far), depth)
        depth = torch.clamp(depth, self.depth_camera_near, self.depth_camera_far)
        depth = (depth - self.depth_camera_near) / (
            self.depth_camera_far - self.depth_camera_near
        )
        return depth * 2.0 - 1.0

    @torch.jit.export
    def reset(self) -> None:
        return None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.current_input_size),
            torch.zeros(1, self.history_input_size),
            torch.zeros(1, self.elevation_history_length, *self.vision_spatial_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["current_proprio", "proprio_history", "depth_history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
