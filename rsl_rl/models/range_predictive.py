# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""G: compact spatial prediction/correction for sparse panoramic observations.

This is an observation-only predictive state estimator, not an MBRL controller.
Only depth/mask enters inference. Map/full images are training targets, and PPO
consumes the same detached 512 features as A0. No pose, action, teacher or EMA.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .range_jepa import RangeJEPA


class RangePredictiveJEPA(RangeJEPA):
    """Spatial latent prior -> masked measurement correction -> A0 readouts.

    Memory is flattened [B,4*8*32] for the existing episode-aware cache/storage.
    Its layout is angular, not a world map. One transition per genuine capture;
    held observations are handled by RangeJEPAInferencePolicy without advancing
    memory. Unobserved slots advance the prior, never absorb invalid CNN tokens.
    """

    uses_temporal_prediction = True
    metric_names = ("future", "future_pair_fraction")

    def __init__(
        self,
        image_height=40,
        image_width=96,
        channels=32,
        hidden_size=1024,
        bank_size=32,
        sectors=8,
        vertical_fov=(52.0, -90.0),
        max_range=40.0,
        attention_heads=4,
        input_normalized=True,
        variance_floor=0.5,
        covariance_weight=0.01,
        variant="G",
        memory_grid=(4, 8),
        capture_dt=0.1,
        future_weight=0.1,
    ):
        if variant != "G":
            raise ValueError("RangePredictiveJEPA is experiment G")
        if len(memory_grid) != 2 or min(memory_grid) < 1 or math.prod(memory_grid) * channels != hidden_size:
            raise ValueError("hidden_size must equal memory_grid area times channels")
        if math.prod(memory_grid) != bank_size or capture_dt <= 0 or not math.isfinite(future_weight) or future_weight < 0:
            raise ValueError("Require bank_size=memory area, positive capture period and nonnegative future weight")
        super().__init__(image_height, image_width, channels, hidden_size, bank_size, sectors,
                         vertical_fov, max_range, attention_heads, input_normalized,
                         variance_floor, covariance_weight)
        del self.gru
        del self.memory_bank
        self.variant = variant
        self.memory_grid = tuple(memory_grid)
        self.capture_dt = float(capture_dt)
        self.feature_size = 512
        self.default_weights = {**RangeJEPA.default_weights, "future": float(future_weight)}
        self.transition_depthwise = nn.Conv2d(channels, channels, 3, groups=channels)
        self.transition_pointwise = nn.Conv2d(channels, channels, 1)
        self.transition_norm = nn.LayerNorm(channels)
        self.correction_gate = nn.Linear(2 * channels, channels)
        self.correction_value = nn.Linear(2 * channels, channels)
        directions = self.query_directions.transpose(1, 2).reshape(1, 3, self.token_height, self.token_width)
        directions = F.adaptive_avg_pool2d(directions, self.memory_grid).flatten(2).transpose(1, 2)
        self.register_buffer("memory_directions", F.normalize(directions, dim=-1))

    def _pool(self, tokens, coverage):
        spatial = tokens.transpose(1, 2).reshape(-1, self.channels, self.token_height, self.token_width)
        valid = coverage.transpose(1, 2).reshape(-1, 1, self.token_height, self.token_width)
        pooled_valid = F.adaptive_avg_pool2d(valid, self.memory_grid)
        pooled = F.adaptive_avg_pool2d(spatial * valid, self.memory_grid) / pooled_valid.clamp_min(1e-6)
        return pooled.flatten(2).transpose(1, 2), pooled_valid.flatten(2).transpose(1, 2)

    def predict_prior(self, hidden):
        """No current measurement input: also usable for causal prediction tests."""
        state = hidden.reshape(-1, self.bank_size, self.channels)
        spatial = state.transpose(1, 2).reshape(-1, self.channels, *self.memory_grid)
        padded = F.pad(F.pad(spatial, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
        delta = self.transition_pointwise(F.silu(self.transition_depthwise(padded)))
        return self.transition_norm(state + delta.flatten(2).transpose(1, 2))

    def _step_encoded(self, encoded, hidden, reset=None):
        tokens, coverage = encoded["tokens"], encoded["coverage"]
        if hidden.shape != (tokens.shape[0], self.hidden_size):
            raise ValueError("Hidden shape differs from spatial memory contract")
        if reset is not None:
            if reset.numel() != tokens.shape[0]:
                raise ValueError("One reset flag per environment is required")
            hidden = torch.where(reset.reshape(-1, 1).bool(), torch.zeros_like(hidden), hidden)
        prior = self.predict_prior(hidden)
        current, valid = self._pool(tokens, coverage)
        combined = torch.cat((prior, current), -1)
        gate = torch.sigmoid(self.correction_gate(combined))
        corrected = prior + gate * (self.correction_value(combined) - prior)
        posterior = torch.where(valid > 0, corrected, prior)
        query = self.query(self.query_directions).expand(tokens.shape[0], -1, -1)
        memory_position = self.query(self.memory_directions)
        recovered = self.readout(query, posterior + memory_position, posterior, need_weights=False)[0]
        e = self.recovery_norm(tokens * coverage + recovered + query)
        context = F.adaptive_avg_pool1d(e.transpose(1, 2), self.bank_size).transpose(1, 2)
        z = self.completion_norm(e + self.completion(e + query, context, context, need_weights=False)[0])
        z = z + self.completion_mlp(z)
        return dict(e=e, z=z, hidden=posterior.flatten(1), prior=prior,
                    features=torch.cat((self.pool_sectors(e), self.pool_sectors(z)), -1).flatten(1))

    def _auxiliary_extra_terms(self, output, targets, coverage, temporal_valid):
        # Reuse the full-view encoding already required for JEPA/codec. No
        # extra CNN, image decoder, raycast or imagined image rollout is added.
        target, support = self._pool(targets[2].detach(), coverage[2])
        error = self._weighted((output["prior"] - target).square().mean(-1), support.squeeze(-1))
        if temporal_valid is None:
            raise ValueError("G requires explicit consecutive-capture eligibility during training")
        # Normalize by all active anchors (same weighting across DDP batches),
        # log pair fraction to disclose how many eligible pairs were available.
        return {"future": error * temporal_valid}
