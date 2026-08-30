# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UniFP history encoder-decoder actor."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class _PredictionTransform(nn.Module):
    """Convert raw auxiliary-head outputs into policy-facing physical values."""

    def __init__(
        self,
        output_dim: int,
        part_dims: tuple[int, ...] | list[int] | None,
        part_activations: tuple[str, ...] | list[str] | None,
    ) -> None:
        super().__init__()
        if part_dims is None:
            part_dims = (output_dim,)
        if part_activations is None:
            part_activations = ("identity",) * len(part_dims)
        if sum(part_dims) != output_dim:
            raise ValueError(
                f"Prediction part dims sum to {sum(part_dims)}, expected num_pred_obs={output_dim}."
            )
        if len(part_dims) != len(part_activations):
            raise ValueError("prediction_part_dims and prediction_part_activations must have equal lengths.")

        activation_codes: list[int] = []
        for dim, activation in zip(part_dims, part_activations, strict=True):
            if activation == "identity":
                code = 0
            elif activation == "sigmoid":
                code = 1
            elif activation == "tanh":
                code = 2
            else:
                raise ValueError(
                    "Unsupported prediction activation "
                    f"'{activation}'. Expected one of: identity, sigmoid, tanh."
                )
            activation_codes.extend([code] * dim)
        self.register_buffer(
            "activation_codes",
            torch.tensor(activation_codes, dtype=torch.long),
            persistent=False,
        )

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        codes = self.activation_codes
        output = torch.where(codes == 1, torch.sigmoid(prediction), prediction)
        return torch.where(codes == 2, torch.tanh(prediction), output)


class UniFPAdaptationActor(nn.Module):
    """Encode observation history, decode adaptation targets, and act from its latent."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_length: int = 32,
        num_pred_obs: int = 12,
        history_set: str | None = None,
        current_obs_set: str | None = None,
        latent_dim: int | None = None,
        use_prediction_in_actor: bool = False,
        prediction_part_dims: tuple[int, ...] | list[int] | None = None,
        prediction_part_activations: tuple[str, ...] | list[str] | None = None,
        num_reconstruction_obs: int = 0,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.history_set = history_set
        self.current_obs_set = current_obs_set
        self.separate_history_and_current = history_set is not None or current_obs_set is not None
        if self.separate_history_and_current:
            if history_set is None or current_obs_set is None:
                raise ValueError("history_set and current_obs_set must be configured together.")
            for group in (history_set, current_obs_set):
                if group not in self.obs_groups:
                    raise ValueError(
                        f"UniFP group '{group}' must be included in actor observation groups {self.obs_groups}."
                    )
                if len(obs[group].shape) != 2:
                    raise ValueError(f"UniFP group '{group}' must be 2D, got {tuple(obs[group].shape)}.")
            self.history_dim = int(obs[history_set].shape[-1])
            self.num_single_obs = int(obs[current_obs_set].shape[-1])
        else:
            self.history_dim = self.obs_dim
            self.num_single_obs = self.obs_dim // history_length

        if self.history_dim % history_length != 0:
            raise ValueError(
                f"History obs dim {self.history_dim} is not divisible by history_length {history_length}."
            )
        self.history_length = history_length
        self.num_history_frame_obs = self.history_dim // history_length
        self.num_latent_dim = history_length * 2 if latent_dim is None else int(latent_dim)
        if self.num_latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.num_latent_dim}.")
        self.num_pred_obs = num_pred_obs
        self.num_reconstruction_obs = int(num_reconstruction_obs)
        if self.num_reconstruction_obs < 0:
            raise ValueError("num_reconstruction_obs must be non-negative.")
        self.use_prediction_in_actor = bool(use_prediction_in_actor)
        self.obs_normalization = obs_normalization
        if self.separate_history_and_current:
            self.history_normalizer = (
                EmpiricalNormalization(self.history_dim) if obs_normalization else nn.Identity()
            )
            self.current_obs_normalizer = (
                EmpiricalNormalization(self.num_single_obs) if obs_normalization else nn.Identity()
            )
        else:
            # Preserve legacy state-dict keys for existing UniFP checkpoints.
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        self.encoder = MLP(self.history_dim, self.num_latent_dim, encoder_hidden_dims, activation)
        self.decoder = MLP(self.num_latent_dim, num_pred_obs, decoder_hidden_dims, activation)
        self.prediction_transform = _PredictionTransform(
            num_pred_obs,
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
        actor_input_dim = self.num_single_obs + self.num_latent_dim
        if self.use_prediction_in_actor:
            actor_input_dim += self.num_pred_obs
        self.actor_body = MLP(
            actor_input_dim, actor_output_dim, hidden_dims, activation
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor_body)

    def _flatten_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def _history_and_current(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.separate_history_and_current:
            history = self.history_normalizer(obs[self.history_set])
            current = self.current_obs_normalizer(obs[self.current_obs_set])
        else:
            history = self.obs_normalizer(self._flatten_obs(obs))
            current = history[..., -self.num_single_obs :]
        return history, current

    def _latent_and_raw_prediction(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, current = self._history_and_current(obs)
        latent = self.encoder(history)
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
            raise RuntimeError("This UniFP actor has no reconstruction decoder.")
        history, _ = self._history_and_current(obs)
        return self.reconstruction_decoder(self.encoder(history))

    def adaptation_modules(self) -> tuple[nn.Module, ...]:
        modules = [self.encoder, self.decoder]
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
        actor_in = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.actor_body(actor_in)
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
        return _TorchUniFPAdaptationActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        del verbose
        return _TorchUniFPAdaptationActor(self)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            if self.separate_history_and_current:
                self.history_normalizer.update(obs[self.history_set])
                self.current_obs_normalizer.update(obs[self.current_obs_set])
            else:
                self.obs_normalizer.update(self._flatten_obs(obs))

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
                    "UniFPAdaptationActor only supports 1D observations, got shape "
                    f"{obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchUniFPAdaptationActor(nn.Module):
    """JIT/ONNX export of the UniFP actor deterministic mean."""

    is_recurrent: bool = False

    def __init__(self, model: UniFPAdaptationActor) -> None:
        super().__init__()
        self.separate_history_and_current = model.separate_history_and_current
        if self.separate_history_and_current:
            self.history_normalizer = copy.deepcopy(model.history_normalizer)
            self.current_obs_normalizer = copy.deepcopy(model.current_obs_normalizer)
            self.obs_normalizer = nn.Identity()
        else:
            self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
            self.history_normalizer = nn.Identity()
            self.current_obs_normalizer = nn.Identity()
        self.encoder = copy.deepcopy(model.encoder)
        self.decoder = copy.deepcopy(model.decoder)
        self.prediction_transform = copy.deepcopy(model.prediction_transform)
        self.actor_body = copy.deepcopy(model.actor_body)
        self.use_prediction_in_actor = model.use_prediction_in_actor
        self.num_single_obs = model.num_single_obs
        self.input_size = model.obs_dim
        self.history_input_size = model.history_dim
        self.current_input_size = model.num_single_obs
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, history: torch.Tensor, current: torch.Tensor | None = None) -> torch.Tensor:
        if self.separate_history_and_current:
            if current is None:
                raise ValueError("The separated UniFP export requires both history and current observations.")
            history = self.history_normalizer(history)
            current = self.current_obs_normalizer(current)
        else:
            history = self.obs_normalizer(history)
            current = history[..., -self.num_single_obs :]
        latent = self.encoder(history)
        parts = [current, latent]
        if self.use_prediction_in_actor:
            parts.append(self.prediction_transform(self.decoder(latent)))
        out = self.actor_body(torch.cat(parts, dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        return None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.separate_history_and_current:
            return (
                torch.zeros(1, self.history_input_size),
                torch.zeros(1, self.current_input_size),
            )
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        if self.separate_history_and_current:
            return ["history", "current_obs"]
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
