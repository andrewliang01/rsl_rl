# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""Local sensor-axis surface grids for Range experiment E/F.

These are NOT gravity-aligned base elevations. X/Y/Z use the input ray axes,
origin is the sensor, no world pose or hidden IMU transform is used. Each cell
retains the highest valid ray endpoint inside the configured Z band. Empty
cells remain unknown; upper surfaces are not guaranteed traversable support.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


class RangeBEVProjector(nn.Module):
    """Vectorized endpoint rasterizer. Output [B,scale,(height,valid),Y,X].

    Heights are metres. Cell indices increase with positive X/Y. Angular pixel
    centres follow RangeJEPA's convention, including a periodic azimuth seam.
    Out-of-band/range predictions are rejected, never clipped onto a surface.
    """

    def __init__(
        self,
        image_height=40,
        image_width=96,
        vertical_fov=(52.0, -90.0),
        max_range=40.0,
        input_normalized=True,
        grid_size=32,
        extents=(1.6, 3.2),
        height_band=(-3.0, 0.5),
    ):
        super().__init__()
        if min(image_height, image_width, grid_size) <= 0 or max_range <= 0:
            raise ValueError("Positive image/grid dimensions and range required")
        if not extents or any(not math.isfinite(e) or e <= 0 for e in extents):
            raise ValueError("Positive finite grid extents required")
        if len(height_band) != 2 or not all(math.isfinite(v) for v in height_band) or height_band[0] >= height_band[1]:
            raise ValueError("Finite ascending height band required")
        if (
            len(vertical_fov) != 2
            or vertical_fov[0] == vertical_fov[1]
            or any(not -90 <= v <= 90 for v in vertical_fov)
        ):
            raise ValueError("Distinct elevation endpoints in [-90,90] required")
        self.image_height, self.image_width = image_height, image_width
        self.grid_size, self.scales = int(grid_size), len(extents)
        self.max_range, self.input_normalized = float(max_range), bool(input_normalized)
        self.height_band = tuple(float(v) for v in height_band)
        az = (torch.arange(image_width) + 0.5) * (2 * math.pi / image_width) - math.pi
        el = (
            (vertical_fov[0] + (torch.arange(image_height) + 0.5) / image_height * (vertical_fov[1] - vertical_fov[0]))
            * math.pi
            / 180
        )
        el, az = torch.meshgrid(el, az, indexing="ij")
        self.register_buffer(
            "rays", torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), -1).reshape(1, -1, 3)
        )
        self.register_buffer("extents", torch.tensor(extents).reshape(1, -1, 1, 1))

    def forward(self, image, mask_logits=False):
        if image.ndim != 4 or image.shape[1:] != (2, self.image_height, self.image_width):
            raise ValueError("Expected [B,2,H,W] range and validity image")
        # Float32 scatter avoids half precision height quantization and works
        # independently of auxiliary AMP. Projection is not a learned layer.
        depth = image[:, 0].float().flatten(1)
        if self.input_normalized:
            depth = depth * self.max_range
        raw_mask = image[:, 1].flatten(1)
        valid = (
            (raw_mask > (0.0 if mask_logits else 0.5)) & torch.isfinite(depth) & (depth > 0) & (depth <= self.max_range)
        )
        safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
        xyz = safe_depth[..., None] * self.rays
        z = xyz[..., 2]
        valid = valid & (z >= self.height_band[0]) & (z <= self.height_band[1])
        # Scale dimension is broadcast; a single GPU scatter handles all grids.
        xy = xyz[:, None, :, :2]
        inside = ((xy >= -self.extents / 2) & (xy < self.extents / 2)).all(-1)
        xy_index = torch.floor((xy / self.extents + 0.5) * self.grid_size).long()
        index = (xy_index[..., 1] * self.grid_size + xy_index[..., 0]).clamp(0, self.grid_size**2 - 1)
        keep = valid[:, None] & inside
        source = torch.where(keep, z[:, None].expand_as(keep), torch.full_like(keep, -torch.inf, dtype=depth.dtype))
        result = depth.new_full((image.shape[0], self.scales, self.grid_size**2), -torch.inf)
        result.scatter_reduce_(2, index, source, reduce="amax", include_self=True)
        mask = torch.isfinite(result)
        height = torch.where(mask, result, torch.zeros_like(result))
        return torch.stack((height, mask.to(height.dtype)), 2).reshape(
            image.shape[0], self.scales, 2, self.grid_size, self.grid_size
        )


class RangeBEVBackend(nn.Module):
    """F: XY queries recover map-supported surfaces then predict completion.

    Angular context comes from the observation/history network. Target BEV CNN
    is shared by obs/map/full surface images, distinct from the angular CNN.
    JEPA targets detach; codec and variance/covariance losses train target CNN.
    """

    def __init__(
        self,
        channels=32,
        attention_heads=4,
        image_height=40,
        image_width=96,
        vertical_fov=(52.0, -90.0),
        max_range=40.0,
        input_normalized=True,
        grid_size=32,
        extents=(1.6, 3.2),
        height_band=(-3.0, 0.5),
        token_grid=8,
        variance_floor=0.5,
        covariance_weight=0.01,
    ):
        super().__init__()
        if grid_size != token_grid * 4 or channels % attention_heads:
            raise ValueError("Grid must be four times token_grid; heads must divide channels")
        self.projector = RangeBEVProjector(
            image_height, image_width, vertical_fov, max_range, input_normalized, grid_size, extents, height_band
        )
        self.channels, self.token_grid, self.scales = channels, token_grid, len(extents)
        self.feature_size = self.scales * channels * token_grid**2
        self.variance_floor, self.covariance_weight = variance_floor, covariance_weight
        self.target_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, channels, 3, 2, 1),
            nn.GroupNorm(math.gcd(4, channels), channels),
            nn.SiLU(),
        )
        self.query = nn.Sequential(nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.recovery = nn.MultiheadAttention(channels, attention_heads, batch_first=True)
        self.recovery_norm = nn.LayerNorm(channels)
        self.completion = nn.MultiheadAttention(channels, attention_heads, batch_first=True)
        self.completion_norm = nn.LayerNorm(channels)
        self.completion_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 2, 3, padding=1),
        )
        # Physical XY and extent distinguish fine/coarse cells with same index.
        unit = (torch.arange(token_grid) + 0.5) / token_grid - 0.5
        y, x = torch.meshgrid(unit, unit, indexing="ij")
        pos = [
            torch.stack((x * extent, y * extent, torch.full_like(x, extent)), -1).reshape(-1, 3) for extent in extents
        ]
        self.register_buffer("query_positions", torch.cat(pos, 0)[None])

    def predict(self, angular_tokens):
        q = self.query(self.query_positions).expand(angular_tokens.shape[0], -1, -1)
        e = self.recovery_norm(q + self.recovery(q, angular_tokens, angular_tokens, need_weights=False)[0])
        z = self.completion_norm(e + self.completion(e + q, e, e, need_weights=False)[0])
        z = z + self.completion_mlp(z)
        spatial = z.reshape(z.shape[0], self.scales, self.token_grid, self.token_grid, self.channels).permute(
            0, 1, 4, 2, 3
        )
        return {"e": e, "z": z, "features": spatial.reshape(z.shape[0], -1)}

    def encode_target(self, image):
        surface = self.projector(image)
        batch = surface.shape[0]
        flat = surface.flatten(0, 1)
        encoded = self.target_encoder(flat)
        tokens = encoded.flatten(2).transpose(1, 2).reshape(batch, -1, self.channels)
        coverage = (
            F.adaptive_avg_pool2d(flat[:, 1:2], (self.token_grid, self.token_grid))
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch, -1, 1)
        )
        return {"tokens": tokens, "coverage": coverage, "image": surface}

    def decode(self, tokens):
        spatial = tokens.reshape(-1, self.token_grid, self.token_grid, self.channels).permute(0, 3, 1, 2)
        image = self.decoder(spatial)
        return image.reshape(tokens.shape[0], self.scales, 2, self.projector.grid_size, self.projector.grid_size)

    @staticmethod
    def _weighted(value, weight):
        dims = tuple(range(1, value.ndim))
        return (value * weight).sum(dims) / weight.sum(dims).clamp_min(1e-6)

    def _reconstruction(self, prediction, target, predict_mask=True):
        height, truth, valid = prediction[:, :, 0], target[:, :, 0], target[:, :, 1]
        loss = self._weighted(F.smooth_l1_loss(height, truth, reduction="none", beta=0.1), valid)
        # Penalize local height differences only where both target cells exist.
        for axis in (-1, -2):
            weight = valid.diff(dim=axis).abs() == 0
            valid_pair = valid.narrow(axis, 0, valid.shape[axis] - 1) * weight
            loss = loss + 0.1 * self._weighted(
                F.smooth_l1_loss(height.diff(dim=axis), truth.diff(dim=axis), reduction="none", beta=0.1), valid_pair
            )
        if predict_mask:
            loss = loss + F.binary_cross_entropy_with_logits(prediction[:, :, 1], valid, reduction="none").mean(
                (1, 2, 3)
            )
        return loss

    def _regularizer(self, tokens, coverage):
        count = coverage.sum(1)
        mean = (tokens * coverage).sum(1) / count.clamp_min(1e-6)
        centered = tokens - mean[:, None]
        cov = torch.einsum("bpc,bpd,bp->bcd", centered, centered, coverage.squeeze(-1)) / count[:, :, None].clamp_min(
            1e-6
        )
        variance = cov.diagonal(dim1=-2, dim2=-1)
        penalty = F.relu(self.variance_floor - (variance + 1e-4).sqrt()).mean(-1)
        penalty = (
            penalty + self.covariance_weight * (cov - torch.diag_embed(variance)).square().sum((1, 2)) / self.channels
        )
        return torch.where(count[:, 0] > 1, penalty, torch.zeros_like(penalty))

    def auxiliary_terms(self, output, observed, mapped, simulated):
        batch = observed.shape[0]
        targets = self.encode_target(torch.cat((observed, mapped, simulated), 0))
        tok = targets["tokens"].split(batch)
        cov = targets["coverage"].split(batch)
        image = targets["image"].split(batch)
        pred = self.decode(torch.cat((output["e"], *tok, output["z"]), 0)).split(batch)
        return {
            "obs": self._reconstruction(pred[0], image[0], False),
            "codec": sum(self._reconstruction(pred[i + 1], image[i]) for i in range(3)) / 3,
            "reg": sum(self._regularizer(tok[i], cov[i]) for i in range(3)) / 3,
            "mem": self._weighted((output["e"] - tok[1].detach()).square().mean(-1), cov[1].squeeze(-1)),
            "full": self._weighted((output["z"] - tok[2].detach()).square().mean(-1), cov[2].squeeze(-1)),
            "range": self._reconstruction(pred[4], image[2]),
            "surface_height_mse_m2": self._weighted((pred[4][:, :, 0] - image[2][:, :, 0]).square(), image[2][:, :, 1]),
        }


class RangeBEVActor(MLPModel):
    """PPO trains spatial CNN/fusion; cached perception always detaches."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(256, 256, 256),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        range_features_set="range_features",
        representation="height",
        scales=2,
        grid_size=32,
        latent_grid=8,
        channels=32,
    ):
        if representation not in ("height", "latent"):
            raise ValueError("representation must be height or latent")
        self.range_features_set = range_features_set
        self.scales, self.feature_channels, self.feature_grid = (
            scales,
            (2 if representation == "height" else channels),
            (grid_size if representation == "height" else latent_grid),
        )
        self.feature_size = self.scales * self.feature_channels * self.feature_grid**2
        groups = dict(obs_groups)
        groups[obs_set] = [k for k in groups[obs_set] if k != range_features_set]
        if not groups[obs_set] or obs[range_features_set].shape[-1] != self.feature_size:
            raise ValueError("Actor requires proprioception and matching BEV features")
        super().__init__(
            obs,
            groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            copy.deepcopy(distribution_cfg),
        )
        self.prop_mlp = MLP(self.obs_dim, 64, (128,), activation)
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(self.feature_channels, 16, 3, 2, 1),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.range_adapter = nn.Sequential(nn.Linear(scales * 32 * 4 * 4, 64), nn.ELU())

    def _get_latent_dim(self):
        return 128

    def get_latent(self, obs, masks=None, hidden_state=None):
        prop = self.obs_normalizer(torch.cat([obs[key] for key in self.obs_groups], -1))
        feat = obs[self.range_features_set].detach()
        if feat.shape[:-1] != prop.shape[:-1] or feat.shape[-1] != self.feature_size:
            raise ValueError("BEV feature dimensions mismatch")
        spatial = feat.reshape(-1, self.feature_channels, self.feature_grid, self.feature_grid)
        visual = self.bev_cnn(spatial).reshape(*feat.shape[:-1], -1)
        return torch.cat((self.prop_mlp(prop), self.range_adapter(visual)), -1)

    def as_jit(self):
        return _RangeBEVActorExport(self)

    def as_onnx(self, verbose=False):
        return _RangeBEVActorExport(self, verbose)


class _RangeBEVActorExport(nn.Module):
    is_recurrent = False

    def __init__(self, model, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.prop_mlp = copy.deepcopy(model.prop_mlp)
        self.bev_cnn = copy.deepcopy(model.bev_cnn)
        self.range_adapter = copy.deepcopy(model.range_adapter)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.proprio_input_size = model.obs_dim
        self.feature_size = model.feature_size
        self.feature_channels, self.feature_grid = model.feature_channels, model.feature_grid

    def forward(self, proprio_obs, range_features):
        prop = self.prop_mlp(self.obs_normalizer(proprio_obs))
        visual = self.bev_cnn(
            range_features.reshape(-1, self.feature_channels, self.feature_grid, self.feature_grid)
        ).reshape(range_features.shape[0], -1)
        return self.deterministic_output(self.mlp(torch.cat((prop, self.range_adapter(visual)), -1)))

    @torch.jit.export
    def reset(self):
        pass

    def get_dummy_inputs(self):
        return torch.zeros(1, self.proprio_input_size), torch.zeros(1, self.feature_size)

    @property
    def input_names(self):
        return ["proprio_obs", "range_features"]

    @property
    def output_names(self):
        return ["actions"]

    @property
    def dynamic_axes(self):
        return {"proprio_obs": {0: "batch_size"}, "range_features": {0: "batch_size"}, "actions": {0: "batch_size"}}
