# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""Small depth/mask JEPA and a compact-feature actor for the normal RSL runner.

Perception has explicit external recurrent state. The algorithm owns capture
scheduling, reset, rollout caching and the auxiliary optimizer. No pose, EMA,
teacher, map builder or simulator dependency is present in this module.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


class _PanoramaConv(nn.Module):
    """Wrap azimuth only; elevation does not wrap at its two boundaries."""

    def __init__(self, incoming: int, outgoing: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=0)
        self.norm = nn.GroupNorm(math.gcd(outgoing, 4), outgoing)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.pad(image, (1, 1, 0, 0), mode="circular")
        image = F.pad(image, (0, 0, 1, 1))
        return F.silu(self.norm(self.conv(image)))


class RangeJEPA(nn.Module):
    """Shared CNN -> GRU bank -> geometric readout -> completion attention.

    Images are ``[range_metres/max_range, valid_mask]`` by default. Rows run from the first to the
    second elevation in ``vertical_fov``; columns cover a full 360 degrees.
    ``features`` is the flattened ``[B, 8, 64]`` concatenation of sector-pooled
    recovery e and completion z. Only the actor's fusion consumes these features.
    """

    default_weights = dict(obs=1.0, codec=1.0, reg=0.01, mem=1.0, full=1.0, range=1.0)

    def __init__(
        self,
        image_height: int = 40,
        image_width: int = 96,
        channels: int = 32,
        hidden_size: int = 128,
        bank_size: int = 32,
        sectors: int = 8,
        vertical_fov: tuple[float, float] = (52.0, -90.0),
        max_range: float = 40.0,
        attention_heads: int = 4,
        input_normalized: bool = True,
        variance_floor: float = 0.5,
        covariance_weight: float = 0.01,
    ):
        super().__init__()
        if min(image_height, image_width, channels, hidden_size, bank_size, sectors) <= 0:
            raise ValueError("RangeJEPA dimensions must be positive")
        if channels % attention_heads or sectors != 8 or channels != 32:
            raise ValueError("The compact actor contract requires 32 channels and 8 sectors; heads must divide channels")
        if max_range <= 0 or not all(-90 <= v <= 90 for v in vertical_fov) or vertical_fov[0] == vertical_fov[1]:
            raise ValueError("Positive max_range and distinct elevation endpoints in [-90,90] are required")
        self.image_height, self.image_width = int(image_height), int(image_width)
        self.channels, self.hidden_size, self.bank_size, self.sectors = channels, hidden_size, bank_size, sectors
        self.max_range = float(max_range)
        self.input_normalized = bool(input_normalized)
        self.variance_floor, self.covariance_weight = float(variance_floor), float(covariance_weight)
        self.token_height = (image_height + 3) // 4
        self.token_width = (image_width + 3) // 4
        self.encoder = nn.Sequential(_PanoramaConv(2, 16), _PanoramaConv(16, channels))
        self.gru = nn.GRUCell(sectors * channels, hidden_size)
        self.memory_bank = nn.Linear(hidden_size, bank_size * channels)
        self.query = nn.Sequential(nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.readout = nn.MultiheadAttention(channels, attention_heads, batch_first=True, dropout=0.0)
        self.recovery_norm = nn.LayerNorm(channels)
        self.completion = nn.MultiheadAttention(channels, attention_heads, batch_first=True, dropout=0.0)
        self.completion_norm = nn.LayerNorm(channels)
        self.completion_mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels))
        self.decoder = nn.Linear(channels, 2)
        azimuth = (torch.arange(image_width) + 0.5) * (2 * math.pi / image_width) - math.pi
        elevation = (torch.arange(image_height) + 0.5) / image_height
        elevation = (vertical_fov[0] + elevation * (vertical_fov[1] - vertical_fov[0])) * math.pi / 180
        el, az = torch.meshgrid(elevation, azimuth, indexing="ij")
        rays = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), 0)[None]
        rays = F.adaptive_avg_pool2d(rays, (self.token_height, self.token_width)).flatten(2).transpose(1, 2)
        self.register_buffer("query_directions", F.normalize(rays, dim=-1))

    def initial_hidden(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(batch_size, self.hidden_size, device=device or parameter.device, dtype=dtype or parameter.dtype)

    def _clean(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (2, self.image_height, self.image_width):
            raise ValueError("Expected [B,2,image_height,image_width] depth/mask image")
        if not image.is_floating_point():
            raise ValueError("Depth/mask image must be floating point")
        depth, supplied = image[:, :1], image[:, 1:2]
        valid = (supplied > 0.5) & torch.isfinite(depth) & (depth > 0) & (depth <= (1.0 if self.input_normalized else self.max_range))
        depth = torch.where(valid, depth, torch.zeros_like(depth)) / (1.0 if self.input_normalized else self.max_range)
        return torch.cat((depth, valid.to(depth.dtype)), 1)

    def encode(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode any of obs/map/sim using the same trainable CNN."""
        clean = self._clean(image)
        spatial = self.encoder(clean)
        tokens = spatial.flatten(2).transpose(1, 2)
        coverage = F.adaptive_avg_pool2d(clean[:, 1:2], spatial.shape[-2:]).flatten(2).transpose(1, 2)
        return {"tokens": tokens, "coverage": coverage, "image": clean}

    def encode_target(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same encoder, not a copied or frozen teacher. Caller stops JEPA targets."""
        return self.encode(image)

    def _step_encoded(self, encoded: dict[str, torch.Tensor], hidden: torch.Tensor, reset=None) -> dict[str, torch.Tensor]:
        tokens, coverage = encoded["tokens"], encoded["coverage"]
        if hidden.shape != (tokens.shape[0], self.hidden_size):
            raise ValueError("Hidden state must have shape [B,hidden_size]")
        if reset is not None:
            if reset.numel() != tokens.shape[0]:
                raise ValueError("reset must contain one flag per environment")
            hidden = torch.where(reset.reshape(-1, 1).bool(), torch.zeros_like(hidden), hidden)
        # Keep absolute azimuth: a global average would erase where an old
        # return occurred, making geometric queries ambiguous under history.
        sector_weight = self.pool_sectors(coverage.expand(-1, -1, self.channels))
        sector_value = self.pool_sectors(tokens * coverage)
        pooled = (sector_value / sector_weight.clamp_min(1e-6)).flatten(1)
        proposal = self.gru(pooled, hidden)
        hidden = torch.where((coverage.sum(1) > 0), proposal, hidden)
        bank = self.memory_bank(hidden).reshape(-1, self.bank_size, self.channels)
        query = self.query(self.query_directions).expand(tokens.shape[0], -1, -1)
        recovered = self.readout(query, bank, bank, need_weights=False)[0]
        e = self.recovery_norm(tokens + recovered + query)
        # Bound attention cost by the fixed bank size rather than P x P.
        context = F.adaptive_avg_pool1d(e.transpose(1, 2), self.bank_size).transpose(1, 2)
        completed = self.completion(e + query, context, context, need_weights=False)[0]
        z = self.completion_norm(e + completed)
        z = z + self.completion_mlp(z)
        ep = self.pool_sectors(e)
        zp = self.pool_sectors(z)
        return {"e": e, "z": z, "hidden": hidden, "features": torch.cat((ep, zp), -1).flatten(1)}

    def step(self, image: torch.Tensor, hidden: torch.Tensor, reset=None) -> dict[str, torch.Tensor]:
        """One genuine new capture; held frames must be handled by the caller."""
        return self._step_encoded(self.encode(image), hidden, reset)

    def forward(self, image: torch.Tensor, hidden: torch.Tensor, reset=None) -> dict[str, torch.Tensor]:
        return self.step(image, hidden, reset)

    def pool_sectors(self, tokens: torch.Tensor) -> torch.Tensor:
        spatial = tokens.transpose(1, 2).reshape(-1, self.channels, self.token_height, self.token_width)
        return F.adaptive_avg_pool2d(spatial, (1, self.sectors)).squeeze(2).transpose(1, 2)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        prediction = self.decoder(tokens).transpose(1, 2).reshape(-1, 2, self.token_height, self.token_width)
        return F.interpolate(prediction, size=(self.image_height, self.image_width), mode="bilinear", align_corners=False)

    @staticmethod
    def _weighted(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, value.ndim))
        return (value * weight).sum(dims) / weight.sum(dims).clamp_min(1e-6)

    def _reconstruction(self, prediction, target, *, predict_mask=True):
        depth = self._weighted(F.smooth_l1_loss(prediction[:, :1], target[:, :1], reduction="none"), target[:, 1:2])
        if predict_mask:
            depth = depth + F.binary_cross_entropy_with_logits(prediction[:, 1:2], target[:, 1:2], reduction="none").mean((1, 2, 3))
        return depth

    def _regularizer(self, tokens, coverage):
        weight = coverage.squeeze(-1)
        count = weight.sum(1)
        mean = (tokens * coverage).sum(1) / count[:, None].clamp_min(1e-6)
        centered = tokens - mean[:, None]
        covariance = torch.einsum("bpc,bpd,bp->bcd", centered, centered, weight) / count[:, None, None].clamp_min(1e-6)
        variance = covariance.diagonal(dim1=-2, dim2=-1)
        variance_loss = F.relu(self.variance_floor - (variance + 1e-4).sqrt()).mean(-1)
        off_diagonal = covariance - torch.diag_embed(variance)
        covariance_loss = off_diagonal.square().sum((1, 2)) / self.channels
        return torch.where(count > 1, variance_loss + self.covariance_weight * covariance_loss, torch.zeros_like(count))

    def auxiliary_loss(self, observed, mapped, simulated, initial_hidden=None, reset=None, support=None, active=None, weights=None, profile=None):
        """Fresh shared-encoder sequence objective; gradients cross GRU timesteps.

        Input images: [T,B,2,H,W] (or one [B,2,H,W] frame). Reset [T,B]
        clears memory *before* that frame; active masks only padded sequence
        slots. Support [T,B,H,W] defaults to map validity and cannot add invalid
        map pixels. All losses use a per-anchor mean, then active-anchor mean.
        Learned targets are recomputed on every call, never stored across Adam.
        """
        if observed.ndim == 4:
            observed, mapped, simulated = observed[None], mapped[None], simulated[None]
        if observed.ndim != 5 or observed.shape != mapped.shape or observed.shape != simulated.shape:
            raise ValueError("obs/map/sim must have identical [T,B,2,H,W] shapes")
        T, B = observed.shape[:2]
        if profile is not None:
            # All three views are encoded, including padded sequence slots.
            # Shapes are Python integers: no active-mask CUDA reduction/sync.
            profile.count("encoded_image_rows", int(T) * int(B) * 3)
        if not T or not B:
            raise ValueError("Auxiliary sequence must contain at least one step and environment")
        def flags(value, default):
            if value is None:
                return torch.full((T, B), default, device=observed.device, dtype=torch.bool)
            if value.numel() != T * B:
                raise ValueError("Sequence reset/active flags must have shape [T,B]")
            return value.reshape(T, B).bool()
        reset, active = flags(reset, False), flags(active, True)
        if support is not None:
            if support.numel() != T * B * self.image_height * self.image_width:
                raise ValueError("support must have shape [T,B,H,W]")
            support = support.reshape(T, B, 1, self.image_height, self.image_width).bool()
        w = dict(self.default_weights)
        if weights is not None:
            if set(weights) - set(w) or any(not math.isfinite(float(v)) or v < 0 for v in weights.values()):
                raise ValueError("Unknown or invalid auxiliary loss weight")
            w.update(weights)
        hidden = self.initial_hidden(B, observed.device, observed.dtype) if initial_hidden is None else initial_hidden
        terms = {key: [] for key in self.default_weights}
        for t in range(T):
            # One shared encoder invocation per view; obs features are reused by prediction and codec.
            with profile.phase("shared_cnn", "forward") if profile is not None else nullcontext():
                encoded = self.encode(torch.cat((observed[t], mapped[t], simulated[t]), 0))
                tok = encoded["tokens"].split(B)
                cov = encoded["coverage"].split(B)
                clean = encoded["image"].split(B)
            with profile.phase("memory_readout_completion", "forward") if profile is not None else nullcontext():
                output = self._step_encoded({"tokens": tok[0], "coverage": cov[0]}, hidden, reset[t] & active[t])
                hidden = torch.where(active[t, :, None], output["hidden"], hidden)
            with profile.phase("decoder", "forward") if profile is not None else nullcontext():
                predictions = self.decode(torch.cat((output["e"], *tok, output["z"]), 0)).split(B)
            with profile.phase("loss_regularizer_mask", "forward") if profile is not None else nullcontext():
                terms["obs"].append(self._reconstruction(predictions[0], clean[0], predict_mask=False))
                terms["codec"].append(sum(self._reconstruction(predictions[1+i], clean[i]) for i in range(3)) / 3)
                terms["reg"].append(sum(self._regularizer(tok[i], cov[i]) for i in range(3)) / 3)
                recovery = cov[1]
                if support is not None:
                    valid_support = support[t] & (clean[1][:, 1:2] > 0.5)
                    recovery = F.adaptive_avg_pool2d(valid_support.to(observed.dtype), (self.token_height, self.token_width)).flatten(2).transpose(1, 2)
                terms["mem"].append(self._weighted((output["e"] - tok[1].detach()).square().mean(-1), recovery.squeeze(-1)))
                terms["full"].append(self._weighted((output["z"] - tok[2].detach()).square().mean(-1), cov[2].squeeze(-1)))
                terms["range"].append(self._reconstruction(predictions[4], clean[2]))
        with profile.phase("loss_regularizer_mask", "forward") if profile is not None else nullcontext():
            denominator = active.sum().clamp_min(1)
            results = {key: (torch.stack(values) * active).sum() / denominator for key, values in terms.items()}
            results["loss"] = sum(w[key] * results[key] for key in terms)
            results["hidden"] = hidden
            results["anchors"] = active.sum()
            results["anchor_count"] = active.sum()
        return results


class RangeJEPAActor(MLPModel):
    """Nonrecurrent PPO model on cached, detached perception sector features.

    The recurrent perception is algorithm-owned. Thus normal RSL feedforward
    storage and sampling remain unchanged and PPO never re-encodes raw images.
    Only proprioception enters the empirical normalizer.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256), activation="elu",
                 obs_normalization=False, distribution_cfg=None, range_features_set="range_features"):
        self.range_features_set = range_features_set
        groups = dict(obs_groups)
        groups[obs_set] = [key for key in obs_groups[obs_set] if key != range_features_set]
        if not groups[obs_set] or obs[range_features_set].shape[-1] != 512:
            raise ValueError("Actor needs proprioception groups and separate 512-dimensional range_features")
        super().__init__(obs, groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         copy.deepcopy(distribution_cfg))
        self.prop_mlp = MLP(self.obs_dim, 64, (128,), activation)
        self.fusion = nn.Linear(66, 32)
        self.range_adapter = nn.Sequential(nn.Linear(256, 64), nn.ELU())
        angles = (torch.arange(8) + 0.5) * (2 * math.pi / 8) - math.pi
        self.register_buffer("sector_directions", torch.stack((angles.cos(), angles.sin()), -1)[None])

    def _get_latent_dim(self):
        return 128

    def get_latent(self, obs, masks=None, hidden_state=None):
        prop = self.obs_normalizer(torch.cat([obs[key] for key in self.obs_groups], -1))
        features = obs[self.range_features_set]
        if features.shape[-1] != 512 or features.shape[:-1] != prop.shape[:-1]:
            raise ValueError("range_features must match proprioception batch and have 512 entries")
        sectors = features.detach().reshape(*features.shape[:-1], 8, 64)
        directions = self.sector_directions.expand(*sectors.shape[:-2], 8, 2)
        visual = F.silu(self.fusion(torch.cat((sectors, directions), -1))).flatten(-2)
        return torch.cat((self.prop_mlp(prop), self.range_adapter(visual)), -1)

    def update_normalization(self, obs):
        # Explicitly use only original proprioception, not the learned cache.
        return super().update_normalization(obs)

    def as_jit(self):
        return _RangeActorExport(self)

    def as_onnx(self, verbose=False):
        return _RangeActorExport(self, verbose)


class _RangeActorExport(nn.Module):
    """Policy-only deployment: caller runs perception step and manages its state.

    Explicit split inputs prevent silent export of an ordinary proprio-only MLP.
    """
    is_recurrent = False

    def __init__(self, model, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.fusion = copy.deepcopy(model.fusion)
        self.range_adapter = copy.deepcopy(model.range_adapter)
        self.mlp = copy.deepcopy(model.mlp)
        self.register_buffer("sector_directions", model.sector_directions.detach().clone())
        self.deterministic_output = (model.distribution.as_deterministic_output_module()
                                     if model.distribution is not None else nn.Identity())
        self.proprio_input_size = model.obs_dim

    def forward(self, proprio_obs: torch.Tensor, range_features: torch.Tensor) -> torch.Tensor:
        prop = self.prop_mlp(self.obs_normalizer(proprio_obs))
        sectors = range_features.detach().reshape(-1, 8, 64)
        directions = self.sector_directions.expand(sectors.shape[0], -1, -1)
        visual = F.silu(self.fusion(torch.cat((sectors, directions), -1))).flatten(1)
        latent = torch.cat((prop, self.range_adapter(visual)), -1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self):
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.proprio_input_size), torch.zeros(1, 512))

    @property
    def input_names(self):
        return ["proprio_obs", "range_features"]

    @property
    def output_names(self):
        return ["actions"]

    @property
    def dynamic_axes(self):
        return {"proprio_obs": {0: "batch_size"}, "range_features": {0: "batch_size"}, "actions": {0: "batch_size"}}
