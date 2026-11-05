"""HV: retain vertical layout in a zero-initialized PPO-only residual.

The observed CNN/GRU/cache and auxiliary three-view objective are unchanged.
This is an interface ablation, not a claim of a new memory mechanism.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.models.range_evidence import RangeEvidenceActor


class RangeEvidenceVerticalActor(RangeEvidenceActor):
    """Add 552 policy parameters without another reader or encoder forward."""

    vertical_grid = (4, 8)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reader.channels != 32:
            raise ValueError('HV requires the unchanged 32-channel H interface.')
        # Module construction must not advance the seed used for the Critic.
        with torch.random.fork_rng(devices=[]):
            self.vertical_residual = nn.Linear(68, 8)
            nn.init.zeros_(self.vertical_residual.weight)
            nn.init.zeros_(self.vertical_residual.bias)
        directions = self._vertical_pool(self.reader.directions)
        self.register_buffer('vertical_directions', directions.contiguous(), persistent=False)

    def _vertical_pool(self, tokens):
        image = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], *self.reader.grid)
        return F.adaptive_avg_pool2d(image, self.vertical_grid).flatten(2).transpose(1, 2)

    def _read_visual(self, cache, prop):
        # Exactly the original H read/fusion; a second call would waste attention
        # and could make a supposedly shared capture disagree with its decoder.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.reader_ppo_gradient):
            with torch.autocast(device_type=prop.device.type, dtype=torch.bfloat16,
                                enabled=self.reader_amp and prop.is_cuda):
                output = self.reader(cache, prop)
        evidence, residual = self.reader.pool(output['e']), self.reader.pool(output['delta'])
        support = self.reader.pool(output['support_logits'].sigmoid()).detach()
        dirs = self.sector_directions.expand(len(prop), -1, -1)
        base = self.fusion(torch.cat((evidence, residual, support, dirs), -1)).flatten(1)

        cells = torch.cat((self._vertical_pool(output['e']),
                           self._vertical_pool(output['delta']),
                           self._vertical_pool(output['support_logits'].sigmoid()).detach(),
                           self.vertical_directions.expand(len(prop), -1, -1)), -1)
        return base + self.vertical_residual(cells).flatten(1)
