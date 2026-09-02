# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UniFP actor with one fixed history window encoded by a single GRU."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from .unifp_adaptation_actor import _PredictionTransform


class _GRUHistoryEncoder(nn.Module):
    """Encode a chronological fixed window with a GRU and projection head."""

    def __init__(
        self,
        frame_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        projection_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = MLP(
            hidden_dim,
            output_dim,
            projection_hidden_dims,
            activation,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(frames)
        return self.projection(hidden[-1])


class UniFPGRUHistoryActor(nn.Module):
    """Encode the complete policy window with one GRU.

    Isaac Lab concatenates history term by term.  This model reconstructs one
    chronological ``[batch, time, frame]`` tensor and passes all frames,
    including the newest one, through a single GRU.  The newest frame is also
    fed directly to the control MLP.  The GRU is evaluated on a fixed window
    for every policy call and does not retain state across calls or episodes.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_length: int = 32,
        history_term_dims: tuple[int, ...] | list[int] = (3, 3, 3, 12, 12, 12),
        latent_dim: int = 64,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        num_pred_obs: int = 22,
        use_prediction_in_actor: bool = True,
        prediction_part_dims: tuple[int, ...] | list[int] | None = None,
        prediction_part_activations: tuple[str, ...] | list[str] | None = None,
        num_reconstruction_obs: int = 30,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.history_length = int(history_length)
        self.history_term_dims = tuple(int(dim) for dim in history_term_dims)
        if self.history_length <= 0:
            raise ValueError(f"history_length must be positive, got {self.history_length}.")
        if any(dim <= 0 for dim in self.history_term_dims):
            raise ValueError(f"history_term_dims must be positive, got {self.history_term_dims}.")

        self.num_single_obs = sum(self.history_term_dims)
        expected_obs_dim = self.history_length * self.num_single_obs
        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                f"Stacked policy dim {self.obs_dim} does not match history_length "
                f"{self.history_length} * frame_dim {self.num_single_obs} = {expected_obs_dim}."
            )

        frame_major_indices: list[int] = []
        term_offsets: list[int] = []
        offset = 0
        for term_dim in self.history_term_dims:
            term_offsets.append(offset)
            offset += self.history_length * term_dim
        for time_index in range(self.history_length):
            for term_offset, term_dim in zip(term_offsets, self.history_term_dims, strict=True):
                start = term_offset + time_index * term_dim
                frame_major_indices.extend(range(start, start + term_dim))
        self.register_buffer(
            "frame_major_indices",
            torch.tensor(frame_major_indices, dtype=torch.long),
            persistent=False,
        )

        self.num_latent_dim = int(latent_dim)
        if self.num_latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.num_latent_dim}.")
        self.obs_normalization = bool(obs_normalization)
        self.frame_normalizer = (
            EmpiricalNormalization(self.num_single_obs)
            if self.obs_normalization
            else nn.Identity()
        )
        self.history_encoder = _GRUHistoryEncoder(
            self.num_single_obs,
            self.num_latent_dim,
            int(gru_hidden_dim),
            int(gru_num_layers),
            encoder_hidden_dims,
            activation,
        )

        self.num_pred_obs = int(num_pred_obs)
        self.num_reconstruction_obs = int(num_reconstruction_obs)
        if self.num_pred_obs <= 0 or self.num_reconstruction_obs < 0:
            raise ValueError("num_pred_obs must be positive and num_reconstruction_obs non-negative.")
        self.decoder = MLP(
            self.num_latent_dim,
            self.num_pred_obs,
            decoder_hidden_dims,
            activation,
        )
        self.prediction_transform = _PredictionTransform(
            self.num_pred_obs,
            prediction_part_dims,
            prediction_part_activations,
        )
        self.reconstruction_decoder: nn.Module | None
        if self.num_reconstruction_obs > 0:
            self.reconstruction_decoder = MLP(
                self.num_latent_dim,
                self.num_reconstruction_obs,
                decoder_hidden_dims,
                activation,
            )
        else:
            self.reconstruction_decoder = None
        self.use_prediction_in_actor = bool(use_prediction_in_actor)

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        actor_input_dim = self.num_single_obs + self.num_latent_dim
        if self.use_prediction_in_actor:
            actor_input_dim += self.num_pred_obs
        self.actor_body = MLP(actor_input_dim, actor_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_body)

    def _flatten_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def _term_major_to_frames(self, stacked: torch.Tensor) -> torch.Tensor:
        chronological = stacked.index_select(-1, self.frame_major_indices)
        return chronological.reshape(-1, self.history_length, self.num_single_obs)

    def _history_and_current(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.frame_normalizer(self._term_major_to_frames(self._flatten_obs(obs)))
        return history, history[:, -1]

    def _latent_and_raw_prediction(
        self,
        obs: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, current = self._history_and_current(obs)
        latent = self.history_encoder(history)
        return latent, self.decoder(latent), current

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        latent, raw_prediction, current = self._latent_and_raw_prediction(obs)
        parts = [current, latent]
        if self.use_prediction_in_actor:
            parts.append(self.prediction_transform(raw_prediction))
        return torch.cat(parts, dim=-1)

    def predict_obs_pred(self, obs: TensorDict) -> torch.Tensor:
        _, prediction, _ = self._latent_and_raw_prediction(obs)
        return prediction

    def predict_reconstruction(self, obs: TensorDict) -> torch.Tensor:
        if self.reconstruction_decoder is None:
            raise RuntimeError("This UniFP GRU-history actor has no reconstruction decoder.")
        history, _ = self._history_and_current(obs)
        return self.reconstruction_decoder(self.history_encoder(history))

    def adaptation_modules(self) -> tuple[nn.Module, ...]:
        modules = [self.history_encoder, self.decoder]
        if self.reconstruction_decoder is not None:
            modules.append(self.reconstruction_decoder)
        return tuple(modules)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        mlp_output = self.actor_body(self.get_latent(obs, masks, hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        return None

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        return None

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchUniFPGRUHistoryActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        del verbose
        return _TorchUniFPGRUHistoryActor(self)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            frames = self._term_major_to_frames(self._flatten_obs(obs))
            self.frame_normalizer.update(frames.reshape(-1, self.num_single_obs))

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    "UniFPGRUHistoryActor only supports 1D observations, got shape "
                    f"{obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchUniFPGRUHistoryActor(nn.Module):
    """JIT/ONNX export of the deterministic unified-history GRU policy."""

    is_recurrent: bool = False

    def __init__(self, model: UniFPGRUHistoryActor) -> None:
        super().__init__()
        self.register_buffer(
            "frame_major_indices",
            model.frame_major_indices.detach().clone(),
            persistent=False,
        )
        self.frame_normalizer = copy.deepcopy(model.frame_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.decoder = copy.deepcopy(model.decoder)
        self.prediction_transform = copy.deepcopy(model.prediction_transform)
        self.actor_body = copy.deepcopy(model.actor_body)
        self.use_prediction_in_actor = model.use_prediction_in_actor
        self.history_length = model.history_length
        self.num_single_obs = model.num_single_obs
        self.input_size = model.obs_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        history = obs.index_select(-1, self.frame_major_indices).reshape(
            -1,
            self.history_length,
            self.num_single_obs,
        )
        history = self.frame_normalizer(history)
        latent = self.history_encoder(history)
        parts = [history[:, -1], latent]
        if self.use_prediction_in_actor:
            parts.append(self.prediction_transform(self.decoder(latent)))
        output = self.actor_body(torch.cat(parts, dim=-1))
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        return None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
