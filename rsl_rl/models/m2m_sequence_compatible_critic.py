# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequence-safe non-recurrent critic for recurrent MID-360 PPO.

RSL-RL selects the recurrent rollout generator when *either* model is
recurrent.  Consequently, a recurrent actor is paired with padded
``[time, trajectory, ...]`` observations even when the critic itself is a
feed-forward CNN.  The legacy elevation critic correctly unpads the
trajectories, but then leaves the restored ``[time, env]`` batch dimensions in
front of the image.  ``Conv2d`` cannot consume that five-dimensional tensor.

This module deliberately adds no recurrent state.  It preserves the legacy
critic's modules and state-dict keys, and changes only the forward boundary:

1. unpad with the same utility used by RSL-RL's recurrent models;
2. flatten all TensorDict batch dimensions before the legacy CNN path; and
3. restore the value tensor to those batch dimensions.

The constructor is fail-closed around the formal G1 ECMM critic contract:
exactly one 99-D proprioception group and one single-frame 28x20 height scan.
Other observation groups may coexist in the environment TensorDict, but they
cannot silently enter the critic observation set.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories


class M2MSequenceCompatibleCritic(PropMLPElevationFusionModel):
    """Legacy-compatible elevation critic with safe sequence batch handling.

    The class remains non-recurrent.  ``hidden_state`` is accepted solely to
    match the PPO model call signature and must be ``None``.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        proprio_set: str = "critic",
        expected_proprio_dim: int = 99,
        elevation_set: str = "height_scan_critic",
        vision_spatial_size: tuple[int, int] | list[int] = (28, 20),
        elevation_history_length: int = 1,
        cnn_observation_type: str = "elevationmap",
        elevation_encoder_type: str = "cnn",
        distribution_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        self._validate_constructor_contract(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            proprio_set=proprio_set,
            expected_proprio_dim=expected_proprio_dim,
            elevation_set=elevation_set,
            vision_spatial_size=vision_spatial_size,
            elevation_history_length=elevation_history_length,
            cnn_observation_type=cnn_observation_type,
            elevation_encoder_type=elevation_encoder_type,
            distribution_cfg=distribution_cfg,
        )

        self.proprio_set = proprio_set
        self.expected_proprio_dim = expected_proprio_dim
        self.expected_elevation_shape = (
            elevation_history_length,
            *tuple(vision_spatial_size),
        )

        # The parent creates exactly the same modules and parameter names as
        # PropMLPElevationFusionModel configured for the same legacy critic.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            elevation_set=elevation_set,
            vision_spatial_size=vision_spatial_size,
            elevation_history_length=elevation_history_length,
            cnn_observation_type=cnn_observation_type,
            elevation_encoder_type=elevation_encoder_type,
            distribution_cfg=distribution_cfg,
            **kwargs,
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Evaluate step or sequence observations with PPO-compatible shapes.

        Step input ``[B, ...]`` returns ``[B, 1]``.  Sequence input
        ``[T, N, ...]`` returns ``[T, N, 1]``.  If ``masks`` is supplied, the
        input is interpreted as RSL-RL's padded ``[T, K, ...]`` layout and is
        restored to the rollout ``[T, N, ...]`` layout before CNN evaluation.
        """
        if hidden_state is not None:
            raise ValueError("M2MSequenceCompatibleCritic is non-recurrent and rejects hidden_state.")
        if stochastic_output:
            raise ValueError("A value critic has no stochastic output mode.")

        self._validate_forward_observations(obs)
        if masks is not None:
            self._validate_masks(obs, masks)
            obs = unpad_trajectories(obs, masks)

        if obs.batch_dims == 1:
            return super().forward(obs, masks=None, hidden_state=None, stochastic_output=False)
        if obs.batch_dims != 2:
            raise ValueError(
                "M2MSequenceCompatibleCritic expects TensorDict batch_size [B] or [T,N], "
                f"got {tuple(obs.batch_size)}."
            )

        sequence_batch_size = tuple(obs.batch_size)
        flat_obs = obs.flatten(0, 1)
        flat_values = super().forward(flat_obs, masks=None, hidden_state=None, stochastic_output=False)
        return flat_values.reshape(*sequence_batch_size, 1)

    def _validate_forward_observations(self, obs: TensorDict) -> None:
        if obs.batch_dims not in (1, 2):
            raise ValueError(
                "M2MSequenceCompatibleCritic expects TensorDict batch_size [B] or [T,N], "
                f"got {tuple(obs.batch_size)}."
            )
        self._validate_leaf(
            obs,
            key=self.proprio_set,
            trailing_shape=(self.expected_proprio_dim,),
        )
        self._validate_leaf(
            obs,
            key=self.elevation_set,
            trailing_shape=self.expected_elevation_shape,
        )

    @staticmethod
    def _validate_leaf(obs: TensorDict, *, key: str, trailing_shape: tuple[int, ...]) -> None:
        if key not in obs:
            raise KeyError(f"Required critic observation group '{key}' is missing.")
        value = obs[key]
        if not torch.is_floating_point(value):
            raise TypeError(f"Critic observation '{key}' must be floating point, got {value.dtype}.")
        expected_shape = (*tuple(obs.batch_size), *trailing_shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Critic observation '{key}' must have exact shape {expected_shape}, got {tuple(value.shape)}."
            )

    def _validate_masks(self, obs: TensorDict, masks: torch.Tensor) -> None:
        if obs.batch_dims != 2:
            raise ValueError("PPO trajectory masks require a [T,K] TensorDict batch.")
        if masks.dtype is not torch.bool:
            raise TypeError(f"PPO trajectory masks must be torch.bool, got {masks.dtype}.")
        if tuple(masks.shape) != tuple(obs.batch_size):
            raise ValueError(
                "PPO trajectory masks must exactly match the TensorDict batch size: "
                f"expected {tuple(obs.batch_size)}, got {tuple(masks.shape)}."
            )
        proprio_device = obs[self.proprio_set].device
        elevation_device = obs[self.elevation_set].device
        if elevation_device != proprio_device or masks.device != proprio_device:
            raise ValueError(
                "PPO trajectory masks and selected observations must share a device, "
                f"got masks={masks.device}, proprio={proprio_device}, elevation={elevation_device}."
            )
        if not torch.all(masks[0]):
            raise ValueError("Every padded PPO trajectory must contain a valid first step.")
        if torch.any((~masks[:-1]) & masks[1:]):
            raise ValueError("PPO trajectory masks must be right-padded (no false-to-true transition).")
        valid_steps = int(masks.sum().item())
        time_steps = obs.batch_size[0]
        if valid_steps % time_steps != 0:
            raise ValueError(
                "Valid padded trajectory steps must reconstruct an integral [T,N] rollout: "
                f"got {valid_steps} valid steps for T={time_steps}."
            )

    @staticmethod
    def _validate_constructor_contract(
        *,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        proprio_set: str,
        expected_proprio_dim: int,
        elevation_set: str,
        vision_spatial_size: tuple[int, int] | list[int],
        elevation_history_length: int,
        cnn_observation_type: str,
        elevation_encoder_type: str,
        distribution_cfg: dict | None,
    ) -> None:
        if output_dim != 1:
            raise ValueError(f"M2M critic output_dim must be exactly 1, got {output_dim}.")
        if distribution_cfg is not None:
            raise ValueError("M2M value critic rejects distribution_cfg.")
        if expected_proprio_dim != 99:
            raise ValueError(
                "The formal G1 ECMM critic proprio contract is exactly 99-D; "
                f"got expected_proprio_dim={expected_proprio_dim}."
            )
        if elevation_history_length != 1:
            raise ValueError(
                "The formal M2M critic requires one height-scan frame, "
                f"got {elevation_history_length}."
            )
        if tuple(vision_spatial_size) != (28, 20):
            raise ValueError(
                "The formal M2M critic height scan is exactly (28,20), "
                f"got {tuple(vision_spatial_size)}."
            )
        if cnn_observation_type.lower() != "elevationmap":
            raise ValueError("The formal M2M critic requires cnn_observation_type='elevationmap'.")
        if elevation_encoder_type.lower() != "cnn":
            raise ValueError("The formal M2M critic requires elevation_encoder_type='cnn'.")
        if obs.batch_dims != 1:
            raise ValueError(
                "M2M critic construction requires a step observation TensorDict with batch_size [B], "
                f"got {tuple(obs.batch_size)}."
            )
        if obs_set not in obs_groups:
            raise KeyError(f"Critic observation set '{obs_set}' is missing from obs_groups.")
        selected_groups = obs_groups[obs_set]
        expected_groups = [proprio_set, elevation_set]
        if selected_groups != expected_groups:
            raise ValueError(
                f"Critic obs_groups['{obs_set}'] must be exactly {expected_groups}, got {selected_groups}."
            )
        M2MSequenceCompatibleCritic._validate_leaf(
            obs,
            key=proprio_set,
            trailing_shape=(expected_proprio_dim,),
        )
        M2MSequenceCompatibleCritic._validate_leaf(
            obs,
            key=elevation_set,
            trailing_shape=(elevation_history_length, *tuple(vision_spatial_size)),
        )
