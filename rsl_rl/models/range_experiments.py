# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""Controlled RangeJEPA experiments; the A0 implementation remains untouched.

Historical tokens are external state, not an inferred pose or a generated map.
The runner supplies *observed* frames only, clears the buffer on reset and freshly
re-encodes selected raw history for auxiliary updates. Historical encoding can be
truncated under no_grad; current encoding, predictor and decoder remain trainable.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.models.range_jepa import RangeJEPA, _PanoramaConv


class _FineDecoder(nn.Module):
    """Two learned spatial refinement stages, with azimuth-only circular padding."""

    def __init__(self, channels: int):
        super().__init__()
        self.refine1 = _PanoramaConv(channels, channels, stride=1)
        self.refine2 = _PanoramaConv(channels, 16, stride=1)
        self.head = nn.Conv2d(16, 2, 1)

    def forward(self, spatial, size):
        intermediate = ((size[0] + 1) // 2, (size[1] + 1) // 2)
        spatial = self.refine1(F.interpolate(spatial, size=intermediate, mode="bilinear", align_corners=False))
        spatial = self.refine2(F.interpolate(spatial, size=size, mode="bilinear", align_corners=False))
        return self.head(spatial)


class RangeExperiment(RangeJEPA):
    """A1: fine decoder; B: spatial window; C: multirate window; D: window+GRU.

    E shares C's temporal path and gives deterministic projected decoded surfaces
    to PPO. F predicts a local BEV latent, with independently encoded BEV targets.
    See ``range_bev`` for the explicitly sensor-aligned geometry convention.

    Context shapes are tokens [B,K,P,C], valid [B,K,P] (or frame-only [B,K]),
    age [B,K] in capture seconds. K contains *prior* frames; current is encoded
    once here. Invalid history is never an attention key. A reset ignores all
    supplied history for that row even if the caller forgot to clear it.
    """

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
        variant: str = "A1",
        history_seconds: float | None = None,
        history_frames: int | None = None,
        history_grid: tuple[int, int] = (4, 8),
        capture_dt: float = 0.1,
        huber_delta_m: float = 0.1,
        edge_weight: float = 0.2,
        near_weight: float = 1.0,
        near_range_m: float = 3.0,
        progressive_decoder: bool = True,
        geometry_loss: bool = True,
        bev_grid_size: int = 32,
        bev_extents: tuple[float, float] = (1.6, 3.2),
        bev_height_band: tuple[float, float] = (-3.0, 0.5),
    ):
        super().__init__(
            image_height,
            image_width,
            channels,
            hidden_size,
            bank_size,
            sectors,
            vertical_fov,
            max_range,
            attention_heads,
            input_normalized,
            variance_floor,
            covariance_weight,
        )
        self.variant = variant.upper()
        self.metric_names = ("surface_height_mse_m2",) if self.variant == "F" else ("depth_mse_m2", "near_depth_mse_m2")
        if self.variant not in {"A1", "B", "C", "D", "E", "F"}:
            raise ValueError("variant must be A1, B, C, D, E or F; use RangeJEPA unchanged for A0")
        if capture_dt <= 0 or huber_delta_m <= 0 or near_range_m <= 0 or edge_weight < 0 or near_weight < 0:
            raise ValueError("Positive time/Huber/range scales and nonnegative loss weights required")
        self.history_seconds = float(
            history_seconds if history_seconds is not None else (5.0 if self.variant in {"C", "E", "F"} else 2.0)
        )
        self.capture_dt = float(capture_dt)
        if self.history_seconds <= 0 or min(history_grid) <= 0:
            raise ValueError("History span and grid dimensions must be positive")
        self.history_grid = (self.token_height, self.token_width) if self.variant == "B" else tuple(history_grid)
        self.huber_delta_m, self.edge_weight = float(huber_delta_m), float(edge_weight)
        self.near_weight, self.near_range_m = float(near_weight), float(near_range_m)
        self.progressive_decoder, self.geometry_loss = bool(progressive_decoder), bool(geometry_loss)
        if self.progressive_decoder:
            self.decoder = _FineDecoder(channels)
        self.history_offsets = self._history_offsets(history_frames)
        self.history_frames = len(self.history_offsets) + 1
        self.feature_size = 512
        self.history_token_count = math.prod(self.history_grid)
        if self.variant != "A1":
            self.age_encoder = nn.Sequential(nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels))
            spatial_directions = self.query_directions.transpose(1, 2).reshape(
                1, 3, self.token_height, self.token_width
            )
            historical_directions = (
                F.adaptive_avg_pool2d(spatial_directions, self.history_grid).flatten(2).transpose(1, 2)
            )
            self.register_buffer("history_directions", F.normalize(historical_directions, dim=-1))
            if self.variant != "D":
                # Remove unused parameters rather than carrying a dormant GRU into DDP/Adam.
                del self.gru
                del self.memory_bank
        if self.variant in {"E", "F"}:
            from rsl_rl.models.range_bev import RangeBEVBackend, RangeBEVProjector

            self.feature_size = len(bev_extents) * 2 * bev_grid_size * bev_grid_size if self.variant == "E" else 4096
            self.bev_settings = dict(
                image_height=image_height,
                image_width=image_width,
                vertical_fov=vertical_fov,
                max_range=max_range,
                grid_size=bev_grid_size,
                extents=bev_extents,
                height_band=bev_height_band,
            )
            if self.variant == "E":
                self.bev_projector = RangeBEVProjector(**self.bev_settings, input_normalized=True)
            else:
                self.bev_backend = RangeBEVBackend(
                    **self.bev_settings,
                    input_normalized=input_normalized,
                    channels=channels,
                    attention_heads=attention_heads,
                    variance_floor=variance_floor,
                    covariance_weight=covariance_weight,
                )
                self.feature_size = self.bev_backend.feature_size
                # F has its own BEV completion and decoder. Do not keep unused angular heads.
                del self.decoder
                del self.completion
                del self.completion_norm
                del self.completion_mlp

    def _history_offsets(self, requested_frames):
        if self.variant == "A1":
            return ()
        maximum = max(1, round(self.history_seconds / self.capture_dt))
        if self.variant in {"B", "D"}:
            count = int(requested_frames) if requested_frames is not None else maximum
            if count < 2:
                raise ValueError("Explicit history requires at least two total frames")
            return tuple(range(1, count))
        dense = range(1, min(maximum + 1, max(1, round(1 / self.capture_dt))))
        sparse = (round(second / self.capture_dt) for second in range(1, math.floor(self.history_seconds) + 1))
        offsets = tuple(sorted(set(dense) | set(sparse)))
        if requested_frames is not None and int(requested_frames) != len(offsets) + 1:
            raise ValueError(
                "Multirate history_frames must match dense first second plus integer-second sparse samples"
            )
        return offsets

    def encode_history(self, image):
        return self.history_from_encoded(self.encode(image))

    def history_from_encoded(self, encoded):
        spatial = encoded["tokens"].transpose(1, 2).reshape(-1, self.channels, self.token_height, self.token_width)
        coverage = encoded["coverage"].transpose(1, 2).reshape(-1, 1, self.token_height, self.token_width)
        pooled_coverage = F.adaptive_avg_pool2d(coverage, self.history_grid)
        pooled = F.adaptive_avg_pool2d(spatial * coverage, self.history_grid) / pooled_coverage.clamp_min(1e-6)
        return {"tokens": pooled.flatten(2).transpose(1, 2), "valid": pooled_coverage.flatten(1) > 0}

    def _context(self, tokens, coverage, history_tokens, history_valid, history_age, reset):
        batch = tokens.shape[0]
        current_position = self.query(self.query_directions).expand(batch, -1, -1)
        keys, values, valid = [tokens + current_position], [tokens], [coverage.squeeze(-1) > 0]
        if history_tokens is not None:
            if (
                history_tokens.ndim != 4
                or history_tokens.shape[0] != batch
                or history_tokens.shape[2:] != (self.history_token_count, self.channels)
            ):
                raise ValueError("history_tokens must be [B,K,history_token_count,channels]")
            K, P = history_tokens.shape[1:3]
            history_tokens = history_tokens.to(device=tokens.device, dtype=tokens.dtype)
            if history_valid is None:
                raise ValueError("Historical tokens require an explicit validity mask")
            hv = history_valid.to(device=tokens.device).bool()
            if hv.shape == (batch, K):
                hv = hv[:, :, None].expand(-1, -1, P)
            if hv.shape != (batch, K, P):
                raise ValueError("history_valid must have shape [B,K] or [B,K,P]")
            if history_age is None or history_age.shape != (batch, K):
                raise ValueError("history_age must have shape [B,K] in capture seconds")
            ages = history_age.to(device=tokens.device, dtype=tokens.dtype)
            hv = hv & (torch.isfinite(ages) & (ages > 0))[..., None]
            ages = torch.where(hv.any(-1), ages, torch.zeros_like(ages))
            # Age is a position cue only: no decay, age confidence or old-frame weighting.
            age_features = torch.stack((ages / self.history_seconds, ages.sin(), ages.cos()), -1)
            age_position = self.age_encoder(age_features)[:, :, None, :]
            angular_position = self.query(self.history_directions)[:, None]
            if reset is not None:
                hv = hv & ~reset.reshape(batch, 1, 1).bool()
            # Invalid inputs are zeroed as well as masked: NaN padding cannot poison QK/V.
            safe = torch.where(hv[..., None], history_tokens, torch.zeros_like(history_tokens))
            keys.append((safe + angular_position + age_position).flatten(1, 2))
            values.append(safe.flatten(1, 2))
            valid.append(hv.flatten(1, 2))
        # Always-valid zero sentinel prevents softmax(all -inf) for an empty first scan.
        zero = tokens.new_zeros(batch, 1, self.channels)
        return (
            torch.cat((*keys, zero), 1),
            torch.cat((*values, zero), 1),
            torch.cat((*valid, torch.ones(batch, 1, device=tokens.device, dtype=torch.bool)), 1),
            current_position,
        )

    def _step_encoded(
        self,
        encoded,
        hidden,
        reset=None,
        history_tokens=None,
        history_valid=None,
        history_age=None,
        actor_features=True,
    ):
        if self.variant == "A1":
            return super()._step_encoded(encoded, hidden, reset)
        tokens, coverage = encoded["tokens"], encoded["coverage"]
        if hidden.shape != (tokens.shape[0], self.hidden_size):
            raise ValueError("Hidden state must have shape [B,hidden_size]")
        if reset is not None and reset.numel() != tokens.shape[0]:
            raise ValueError("reset must contain one flag per environment")
        keys, values, valid, query = self._context(tokens, coverage, history_tokens, history_valid, history_age, reset)
        if self.variant == "D":
            if reset is not None:
                hidden = torch.where(reset.reshape(-1, 1).bool(), torch.zeros_like(hidden), hidden)
            sector_weight = self.pool_sectors(coverage.expand(-1, -1, self.channels))
            pooled = (self.pool_sectors(tokens * coverage) / sector_weight.clamp_min(1e-6)).flatten(1)
            proposed = self.gru(pooled, hidden)
            hidden = torch.where(coverage.sum(1) > 0, proposed, hidden)
            bank = self.memory_bank(hidden).reshape(-1, self.bank_size, self.channels)
            keys, values = torch.cat((keys, bank), 1), torch.cat((values, bank), 1)
            valid = torch.cat((valid, torch.ones(bank.shape[:2], device=valid.device, dtype=torch.bool)), 1)
        else:
            hidden = torch.zeros_like(hidden)
        recovered = self.readout(tokens + query, keys, values, key_padding_mask=~valid, need_weights=False)[0]
        e = self.recovery_norm(tokens + query + recovered)
        if self.variant == "F":
            result = self.bev_backend.predict(e)
            result.update(hidden=hidden, angular_e=e)
            return result
        context = F.adaptive_avg_pool1d(e.transpose(1, 2), self.bank_size).transpose(1, 2)
        completed = self.completion(e + query, context, context, need_weights=False)[0]
        z = self.completion_norm(e + completed)
        z = z + self.completion_mlp(z)
        result = {
            "e": e,
            "z": z,
            "hidden": hidden,
            "features": torch.cat((self.pool_sectors(e), self.pool_sectors(z)), -1).flatten(1),
        }
        if self.variant == "E" and actor_features:
            result["features"] = self.bev_projector(self.decode(z), mask_logits=True).flatten(1)
        return result

    def step(self, image, hidden, reset=None, history_tokens=None, history_valid=None, history_age=None):
        encoded = self.encode(image)
        result = self._step_encoded(encoded, hidden, reset, history_tokens, history_valid, history_age)
        if self.variant != "A1":
            result["history_encoded"] = self.history_from_encoded(encoded)
        return result

    def forward(self, image, hidden, reset=None, history_tokens=None, history_valid=None, history_age=None):
        return self.step(image, hidden, reset, history_tokens, history_valid, history_age)

    def decode(self, tokens):
        if self.variant == "F":
            return self.bev_backend.decode(tokens)
        if not self.progressive_decoder:
            return super().decode(tokens)
        spatial = tokens.transpose(1, 2).reshape(-1, self.channels, self.token_height, self.token_width)
        return self.decoder(spatial, (self.image_height, self.image_width))

    def _reconstruction(self, prediction, target, *, predict_mask=True):
        if not self.geometry_loss:
            return super()._reconstruction(prediction, target, predict_mask=predict_mask)
        pred, truth, valid = prediction[:, :1] * self.max_range, target[:, :1] * self.max_range, target[:, 1:2]
        near = (truth < self.near_range_m).to(valid.dtype)
        weight = valid * (1.0 + self.near_weight * near)
        depth = self._weighted(F.huber_loss(pred, truth, reduction="none", delta=self.huber_delta_m), weight)
        # Only adjacent *valid* surfaces are supervised. Azimuth includes the seam;
        # vertical rows stop at the FOV edge instead of wrapping pole to sky.
        horizontal_valid = weight * valid.roll(1, -1)
        horizontal_error = F.huber_loss(
            pred - pred.roll(1, -1), truth - truth.roll(1, -1), reduction="none", delta=self.huber_delta_m
        )
        vertical_valid = weight[:, :, 1:] * valid[:, :, :-1]
        vertical_error = F.huber_loss(
            pred[:, :, 1:] - pred[:, :, :-1],
            truth[:, :, 1:] - truth[:, :, :-1],
            reduction="none",
            delta=self.huber_delta_m,
        )
        depth = (
            depth
            + self.edge_weight
            * (self._weighted(horizontal_error, horizontal_valid) + self._weighted(vertical_error, vertical_valid))
            / 2
        )
        if predict_mask:
            depth = depth + F.binary_cross_entropy_with_logits(prediction[:, 1:2], valid, reduction="none").mean(
                (1, 2, 3)
            )
        return depth

    def auxiliary_loss(
        self,
        observed,
        mapped,
        simulated,
        initial_hidden=None,
        reset=None,
        support=None,
        active=None,
        weights=None,
        profile=None,
        history_tokens=None,
        history_valid=None,
        history_age=None,
        history_images=None,
    ):
        """Same anchor objective as A0, plus optional prior observed context.

        Extra history has [T,B,K,...] dimensions (or [B,K,...] for a single image
        batch). history_images re-encodes with gradients. A runner may instead
        supply freshly re-encoded no_grad history_tokens for truncated history.
        It must not silently supply an old-weight training cache.
        """
        single = observed.ndim == 4
        if single:
            observed, mapped, simulated = observed[None], mapped[None], simulated[None]
            history_tokens = None if history_tokens is None else history_tokens[None]
            history_valid = None if history_valid is None else history_valid[None]
            history_age = None if history_age is None else history_age[None]
            history_images = None if history_images is None else history_images[None]
        if observed.ndim != 5 or observed.shape != mapped.shape or observed.shape != simulated.shape:
            raise ValueError("obs/map/sim must have identical [T,B,2,H,W] shapes")
        T, B = observed.shape[:2]
        if not T or not B:
            raise ValueError("Auxiliary sequence cannot be empty")

        def flags(value, default):
            if value is None:
                return torch.full((T, B), default, device=observed.device, dtype=torch.bool)
            if value.numel() != T * B:
                raise ValueError("Sequence reset/active flags must have shape [T,B]")
            return value.reshape(T, B).bool()

        reset, active = flags(reset, False), flags(active, True)
        if support is not None:
            support = support.reshape(T, B, 1, self.image_height, self.image_width).bool()
        if history_images is not None and history_tokens is not None:
            raise ValueError("Supply raw history or pre-encoded history, not both")
        w = dict(self.default_weights)
        if weights is not None:
            if set(weights) - set(w) or any(not math.isfinite(float(v)) or v < 0 for v in weights.values()):
                raise ValueError("Unknown or invalid auxiliary weight")
            w.update(weights)
        hidden = self.initial_hidden(B, observed.device, observed.dtype) if initial_hidden is None else initial_hidden
        terms = {key: [] for key in self.default_weights}
        metrics = (
            {"surface_height_mse_m2": []} if self.variant == "F" else {"depth_mse_m2": [], "near_depth_mse_m2": []}
        )
        phase = (lambda name: profile.phase(name, "forward")) if profile is not None else (lambda name: nullcontext())
        if profile is not None:
            profile.count("encoded_image_rows", int(T) * int(B) * (1 if self.variant == "F" else 3))
        for t in range(T):
            with phase("shared_cnn"):
                encoded = self.encode(
                    observed[t] if self.variant == "F" else torch.cat((observed[t], mapped[t], simulated[t]), 0)
                )
                tok, cov, clean = (encoded[key].split(B) for key in ("tokens", "coverage", "image"))
                ht = None if history_tokens is None else history_tokens[t]
                hv = None if history_valid is None else history_valid[t]
                ha = None if history_age is None else history_age[t]
                if history_images is not None:
                    raw = history_images[t]
                    encoded_history = self.encode_history(raw.flatten(0, 1))
                    ht = encoded_history["tokens"].reshape(B, raw.shape[1], self.history_token_count, self.channels)
                    token_valid = encoded_history["valid"].reshape(B, raw.shape[1], self.history_token_count)
                    if hv is not None:
                        if hv.ndim == 2:
                            hv = hv[..., None]
                        token_valid = token_valid & hv.bool()
                    hv = token_valid
            with phase("memory_readout_completion"):
                output = self._step_encoded(
                    {"tokens": tok[0], "coverage": cov[0]},
                    hidden,
                    reset[t] & active[t],
                    ht,
                    hv,
                    ha,
                    actor_features=False,
                )
                hidden = torch.where(active[t, :, None], output["hidden"], hidden)
            if self.variant == "F":
                with phase("decoder"):
                    bev_terms = self.bev_backend.auxiliary_terms(output, observed[t], mapped[t], simulated[t])
                for name in terms:
                    terms[name].append(bev_terms[name])
                if "surface_height_mse_m2" in bev_terms:
                    metrics["surface_height_mse_m2"].append(bev_terms["surface_height_mse_m2"])
                continue
            with phase("decoder"):
                predictions = self.decode(torch.cat((output["e"], *tok, output["z"]), 0)).split(B)
            with phase("loss_regularizer_mask"):
                terms["obs"].append(self._reconstruction(predictions[0], clean[0], predict_mask=False))
                terms["codec"].append(sum(self._reconstruction(predictions[1 + i], clean[i]) for i in range(3)) / 3)
                terms["reg"].append(sum(self._regularizer(tok[i], cov[i]) for i in range(3)) / 3)
                recovery = cov[1]
                if support is not None:
                    supported = support[t] & (clean[1][:, 1:2] > 0.5)
                    recovery = (
                        F.adaptive_avg_pool2d(supported.to(observed.dtype), (self.token_height, self.token_width))
                        .flatten(2)
                        .transpose(1, 2)
                    )
                terms["mem"].append(
                    self._weighted((output["e"] - tok[1].detach()).square().mean(-1), recovery.squeeze(-1))
                )
                terms["full"].append(
                    self._weighted((output["z"] - tok[2].detach()).square().mean(-1), cov[2].squeeze(-1))
                )
                terms["range"].append(self._reconstruction(predictions[4], clean[2]))
                error = ((predictions[4][:, :1] - clean[2][:, :1]) * self.max_range).square()
                valid = clean[2][:, 1:2]
                metrics["depth_mse_m2"].append(self._weighted(error, valid))
                metrics["near_depth_mse_m2"].append(
                    self._weighted(error, valid * (clean[2][:, :1] * self.max_range < self.near_range_m))
                )
        denominator = active.sum().clamp_min(1)
        result = {name: (torch.stack(values) * active).sum() / denominator for name, values in terms.items()}
        for name, values in metrics.items():
            if values:
                result[name] = (torch.stack(values) * active).sum() / denominator
        result.update(
            loss=sum(w[name] * result[name] for name in terms),
            hidden=hidden,
            anchors=active.sum(),
            anchor_count=active.sum(),
        )
        return result
