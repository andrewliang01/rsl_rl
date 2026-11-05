"""F2: condition F's observed tokens on detached PPO proprioceptive features.

The actor owns the only proprioceptive MLP. This model owns only its small
64 -> channels projection; no actor module or parameters are registered here.
Target encoders and every F geometry objective remain unchanged.
"""
from __future__ import annotations

from torch import nn

from rsl_rl.models.range_experiments import RangeExperiment


class RangeBodyConditionedBEV(RangeExperiment):
    requires_body_features = True

    def __init__(self, variant="F2", **kwargs):
        if variant != "F2":
            raise ValueError("RangeBodyConditionedBEV requires variant='F2'.")
        # Internal geometry stays F, including the existing four-view inspector.
        super().__init__(variant="F", **kwargs)
        self.body_projection = nn.Linear(64, self.channels)
        # A zero residual initially reproduces F for identical base weights.
        nn.init.zeros_(self.body_projection.weight)
        nn.init.zeros_(self.body_projection.bias)

    def condition_context(self, encoded, body_features=None):
        if body_features is None or body_features.shape != (encoded["tokens"].shape[0], 64):
            raise ValueError("F2 requires capture-aligned actor body features [B,64].")
        return {
            **encoded,
            "tokens": encoded["tokens"] + self.body_projection(body_features.detach())[:, None],
        }
