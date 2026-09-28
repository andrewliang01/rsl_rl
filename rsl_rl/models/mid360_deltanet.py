# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current-panorama MID360 DeltaNet actor and asymmetric elevation critic."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules import MLP, DeltaNetBlock, RMSNorm
from rsl_rl.utils import unpad_trajectories


class _CircularAzimuthConv2d(nn.Conv2d):
    """Wrap panorama width while retaining zero padding along elevation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = self.padding
        if pad_w:
            x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
        if pad_h:
            x = F.pad(x, (0, 0, pad_h, pad_h))
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class CurrentPanoramaEncoder(nn.Module):
    """Encode one sparse range panorama without explicit frame stacking."""

    def __init__(
        self,
        out_dim: int,
        spatial_size: tuple[int, int] = (16, 96),
        hidden_dims: tuple[int, ...] = (8, 16),
        kernel_sizes: tuple[int, ...] = (3, 3),
        strides: tuple[int, ...] = (2, 2),
        circular_azimuth: bool = True,
        near: float = 0.05,
        far: float = 1.857,
    ) -> None:
        super().__init__()
        if not (len(hidden_dims) == len(kernel_sizes) == len(strides) and hidden_dims):
            raise ValueError("Panorama CNN dimensions, kernels and strides must have equal non-zero length.")
        if far <= near:
            raise ValueError("Panorama far range must exceed near range.")
        self.near = float(near)
        self.far = float(far)
        self.circular_azimuth = bool(circular_azimuth)
        conv_type = _CircularAzimuthConv2d if self.circular_azimuth else nn.Conv2d
        layers: list[nn.Module] = []
        channels = 1
        for width, kernel, stride in zip(hidden_dims, kernel_sizes, strides):
            layers.extend(
                (
                    conv_type(
                        channels,
                        width,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                    ),
                    nn.SiLU(),
                )
            )
            channels = width
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            conv_size = self.conv(torch.zeros(1, 1, *spatial_size)).numel()
        self.projection = nn.Linear(conv_size, out_dim)

    def forward(self, panorama: torch.Tensor) -> torch.Tensor:
        if panorama.shape[-3] != 1:
            raise ValueError("DeltaNet actor accepts exactly one current panorama frame.")
        x = torch.nan_to_num(panorama, nan=self.far, posinf=self.far, neginf=self.near)
        x = x.clamp(self.near, self.far)
        x = 2.0 * (x - self.near) / (self.far - self.near) - 1.0
        return F.silu(self.projection(self.conv(x).flatten(1)))


class MID360DeltaEstimator(nn.Module):
    """Fuse current proprioception and panorama while retaining DeltaNet state."""

    def __init__(
        self,
        proprio_dim: int,
        map_size: int,
        dim: int = 256,
        num_delta_layers: int = 2,
        heads: int = 4,
        head_dim: int = 32,
        ffn_dim: int = 512,
        conv_size: int = 4,
        chunk_size: int = 32,
        panorama_shape: tuple[int, int] = (16, 96),
        cnn_hidden_dims: tuple[int, ...] = (8, 16),
        cnn_kernel_sizes: tuple[int, ...] = (3, 3),
        cnn_strides: tuple[int, ...] = (2, 2),
        cnn_circular_azimuth: bool = True,
        near: float = 0.05,
        far: float = 1.857,
    ) -> None:
        super().__init__()
        if num_delta_layers < 1:
            raise ValueError("num_delta_layers must be at least one.")
        if dim < 2 or proprio_dim < 1 or map_size < 1:
            raise ValueError("Estimator feature and output dimensions must be positive.")
        vision_dim = dim // 2
        self.num_layers = int(num_delta_layers)
        self.vision = CurrentPanoramaEncoder(
            vision_dim,
            panorama_shape,
            cnn_hidden_dims,
            cnn_kernel_sizes,
            cnn_strides,
            cnn_circular_azimuth,
            near,
            far,
        )
        self.proprio = MLP(proprio_dim, dim - vision_dim, (128,), "swish")
        self.blocks = nn.ModuleList(
            DeltaNetBlock(dim, heads, head_dim, ffn_dim, conv_size, chunk_size)
            for _ in range(self.num_layers)
        )
        self.state_size = self.blocks[0].state_size
        self.norm = RMSNorm(dim)
        self.map_head = MLP(dim, map_size, (256,), "swish")
        self.velocity_head = MLP(dim, 3, (128,), "swish")

    def initial_state(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros((self.num_layers, batch, self.state_size), dtype=torch.float32)

    def forward(
        self,
        proprio: torch.Tensor,
        panorama: torch.Tensor,
        state: torch.Tensor,
        reset_before: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length, batch = proprio.shape[:2]
        vision = self.vision(panorama.flatten(0, 1)).reshape(length, batch, -1)
        x = torch.cat((vision, self.proprio(proprio)), dim=-1)
        next_states = []
        for index, block in enumerate(self.blocks):
            x, next_state = block(x, state[index], reset_before)
            next_states.append(next_state)
        x = self.norm(x)
        return self.map_head(x), self.velocity_head(x), torch.stack(next_states)


class MID360DeltaNetActor(MLPModel):
    """Recurrent actor driven by current inputs and implicit DeltaNet memory."""

    is_recurrent = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        proprio_set: str = "policy",
        expected_proprio_dim: int = 96,
        expected_action_dim: int = 29,
        panorama_set: str = "height_scan_policy",
        map_shape: tuple[int, int] = (28, 20),
        panorama_shape: tuple[int, int] = (16, 96),
        dim: int = 256,
        num_delta_layers: int = 2,
        heads: int = 4,
        head_dim: int = 32,
        ffn_dim: int = 512,
        conv_size: int = 4,
        chunk_size: int = 32,
        cnn_hidden_dims: tuple[int, ...] = (8, 16),
        cnn_kernel_sizes: tuple[int, ...] = (3, 3),
        cnn_strides: tuple[int, ...] = (2, 2),
        cnn_circular_azimuth: bool = True,
        near: float = 0.05,
        far: float = 1.857,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        expected_groups = [proprio_set, panorama_set]
        if list(obs_groups[obs_set]) != expected_groups:
            raise ValueError(f"DeltaNet actor observations must be exactly {expected_groups}.")
        if tuple(obs[proprio_set].shape[1:]) != (expected_proprio_dim,):
            raise ValueError(
                "DeltaNet actor proprioception must have shape "
                f"[B,{expected_proprio_dim}], got {tuple(obs[proprio_set].shape)}."
            )
        if output_dim != expected_action_dim:
            raise ValueError(
                f"DeltaNet actor expects {expected_action_dim} actions, got {output_dim}."
            )
        if tuple(obs[panorama_set].shape[1:]) != (1, *panorama_shape):
            raise ValueError(
                "DeltaNet actor panorama must contain one current frame with shape "
                f"[B,1,{panorama_shape[0]},{panorama_shape[1]}]."
            )
        self.proprio_set = proprio_set
        self.proprio_dim = expected_proprio_dim
        self.panorama_set = panorama_set
        self.panorama_shape = tuple(panorama_shape)
        self.map_shape = tuple(map_shape)
        self.map_size = self.map_shape[0] * self.map_shape[1]
        distribution_contract = copy.deepcopy(
            distribution_cfg
            or {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
        )
        super().__init__(
            obs,
            {obs_set: [proprio_set]},
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            copy.deepcopy(distribution_contract),
        )
        self.estimator = MID360DeltaEstimator(
            proprio_dim=self.proprio_dim,
            map_size=self.map_size,
            dim=dim,
            num_delta_layers=num_delta_layers,
            heads=heads,
            head_dim=head_dim,
            ffn_dim=ffn_dim,
            conv_size=conv_size,
            chunk_size=chunk_size,
            panorama_shape=panorama_shape,
            cnn_hidden_dims=cnn_hidden_dims,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_circular_azimuth=cnn_circular_azimuth,
            near=near,
            far=far,
        )
        self.model_config = {
            "proprio_set": proprio_set,
            "expected_proprio_dim": expected_proprio_dim,
            "expected_action_dim": expected_action_dim,
            "panorama_set": panorama_set,
            "map_shape": list(map_shape),
            "panorama_shape": list(panorama_shape),
            "dim": dim,
            "num_delta_layers": num_delta_layers,
            "heads": heads,
            "head_dim": head_dim,
            "ffn_dim": ffn_dim,
            "conv_size": conv_size,
            "chunk_size": chunk_size,
            "cnn_hidden_dims": list(cnn_hidden_dims),
            "cnn_kernel_sizes": list(cnn_kernel_sizes),
            "cnn_strides": list(cnn_strides),
            "cnn_circular_azimuth": cnn_circular_azimuth,
            "near": near,
            "far": far,
            "hidden_dims": list(hidden_dims),
            "activation": activation,
            "obs_normalization": obs_normalization,
            "distribution_cfg": distribution_contract,
        }
        self._hidden_state: torch.Tensor | None = None
        self.last_map: torch.Tensor | None = None
        self.last_velocity: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        return self.proprio_dim + self.map_size + 3

    def get_latent(self, obs, masks=None, hidden_state=None):
        proprio = self.obs_normalizer(obs[self.proprio_set])
        panorama = obs[self.panorama_set]
        is_sequence = proprio.ndim == 3
        if not is_sequence:
            proprio = proprio.unsqueeze(0)
            panorama = panorama.unsqueeze(0)
        if hidden_state is None:
            if is_sequence:
                raise ValueError("Sequence replay requires an explicit DeltaNet initial state.")
            if self._hidden_state is None or self._hidden_state.shape[1] != proprio.shape[1]:
                self._hidden_state = self.estimator.initial_state(proprio.shape[1], proprio)
            hidden_state = self._hidden_state
        height, velocity, next_state = self.estimator(proprio, panorama, hidden_state)
        if is_sequence:
            if masks is None:
                raise ValueError("Padded DeltaNet replay requires trajectory masks.")
            height = unpad_trajectories(height, masks)
            velocity = unpad_trajectories(velocity, masks)
            proprio = unpad_trajectories(proprio, masks)
        else:
            self._hidden_state = next_state.detach()
            height = height.squeeze(0)
            velocity = velocity.squeeze(0)
            proprio = proprio.squeeze(0)
        self.last_map = height
        self.last_velocity = velocity
        return torch.cat((proprio, height, velocity), dim=-1)

    def get_hidden_state(self):
        return self._hidden_state

    def reset(self, dones=None, hidden_state=None):
        if hidden_state is not None:
            self._hidden_state = hidden_state.detach().clone()
        elif dones is None:
            self._hidden_state = None
        elif self._hidden_state is not None:
            keep = (~dones.bool().flatten()).to(self._hidden_state.dtype)[None, :, None]
            self._hidden_state = self._hidden_state * keep

    def detach_hidden_state(self, dones=None):
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    def as_jit(self) -> nn.Module:
        return MID360DeltaNetDeployment(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return MID360DeltaNetOnnxDeployment(self, verbose)


class MID360DeltaNetDeployment(nn.Module):
    """Deployment wrapper with explicit recurrent state input and output."""

    def __init__(self, actor: MID360DeltaNetActor) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.estimator = copy.deepcopy(actor.estimator)
        self.mlp = copy.deepcopy(actor.mlp)
        self.output = actor.distribution.as_deterministic_output_module()

    def forward(
        self,
        proprio: torch.Tensor,
        panorama: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio = self.obs_normalizer(proprio)
        height, velocity, next_state = self.estimator(
            proprio.unsqueeze(0),
            panorama.unsqueeze(0),
            state,
        )
        height = height.squeeze(0)
        velocity = velocity.squeeze(0)
        action = self.output(self.mlp(torch.cat((proprio, height, velocity), dim=-1)))
        return action, height, velocity, next_state


class MID360DeltaNetOnnxDeployment(nn.Module):
    """ONNX wrapper for explicit-state MID360 DeltaNet deployment."""

    is_recurrent = True

    def __init__(self, actor: MID360DeltaNetActor, verbose: bool = False) -> None:
        super().__init__()
        self.core = MID360DeltaNetDeployment(actor)
        self.verbose = verbose
        self.proprio_dim = actor.proprio_dim
        self.panorama_shape = actor.panorama_shape

    def forward(self, proprio, panorama, state):
        return self.core(proprio, panorama, state)

    def get_dummy_inputs(self):
        proprio = torch.zeros(1, self.proprio_dim)
        panorama = torch.zeros(1, 1, *self.panorama_shape)
        state = self.core.estimator.initial_state(1, proprio)
        return proprio, panorama, state

    @property
    def input_names(self):
        return ["proprio", "panorama", "state"]

    @property
    def output_names(self):
        return ["actions", "height", "velocity", "next_state"]

    @property
    def dynamic_axes(self):
        return {
            "proprio": {0: "batch"},
            "panorama": {0: "batch"},
            "state": {1: "batch"},
            "actions": {0: "batch"},
            "height": {0: "batch"},
            "velocity": {0: "batch"},
            "next_state": {1: "batch"},
        }


class SequenceElevationHistoryCritic(PropMLPElevationFusionModel):
    """Feed-forward MLP/CNN critic compatible with recurrent PPO replay."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        proprio_set: str = "critic",
        expected_proprio_dim: int = 495,
        elevation_set: str = "height_scan_critic",
        elevation_history_length: int = 5,
        vision_spatial_size: tuple[int, int] = (28, 20),
        distribution_cfg=None,
        **kwargs,
    ) -> None:
        if output_dim != 1 or distribution_cfg is not None:
            raise ValueError("History critic requires one deterministic value output.")
        expected_groups = [proprio_set, elevation_set]
        if list(obs_groups[obs_set]) != expected_groups:
            raise ValueError(f"History critic observations must be exactly {expected_groups}.")
        expected_prop_shape = (expected_proprio_dim,)
        expected_map_shape = (elevation_history_length, *vision_spatial_size)
        if tuple(obs[proprio_set].shape[1:]) != expected_prop_shape:
            raise ValueError(f"Critic proprio history must have shape [B,{expected_proprio_dim}].")
        if tuple(obs[elevation_set].shape[1:]) != expected_map_shape:
            raise ValueError(f"Critic elevation history must have shape [B,{expected_map_shape}].")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            elevation_set=elevation_set,
            elevation_history_length=elevation_history_length,
            vision_spatial_size=vision_spatial_size,
            distribution_cfg=None,
            **kwargs,
        )

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if hidden_state is not None or stochastic_output:
            raise ValueError("History value critic rejects recurrent state and stochastic output.")
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        if obs.batch_dims == 1:
            return super().forward(obs)
        if obs.batch_dims != 2:
            raise ValueError("History critic expects [B] or [T,N] TensorDict batches.")
        batch_size = tuple(obs.batch_size)
        values = super().forward(obs.flatten(0, 1))
        return values.reshape(*batch_size, 1)


__all__ = [
    "CurrentPanoramaEncoder",
    "MID360DeltaEstimator",
    "MID360DeltaNetActor",
    "MID360DeltaNetDeployment",
    "MID360DeltaNetOnnxDeployment",
    "SequenceElevationHistoryCritic",
]
