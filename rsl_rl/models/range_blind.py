"""Matched A0 Actor control with range features withheld in every forward path.

The original perception model and auxiliary losses may still run for a matched
training workload, but the locomotion actor sees only proprioception and fixed
sector directions. This is a controlled blind baseline, not an inference-time
ablation of a vision-trained policy. Its parameters are state-dict compatible
with RangeJEPAActor; its exports also enforce the blind input contract.
"""

import torch
import torch.nn.functional as F

from .range_jepa import RangeJEPAActor, _RangeActorExport


class BlindRangeJEPAActor(RangeJEPAActor):
    """Use A0 proprioception and architecture, while always masking range features."""

    def get_latent(self, obs, masks=None, hidden_state=None):
        masked = obs.clone(recurse=False)
        masked.set(self.range_features_set, torch.zeros_like(obs[self.range_features_set]))
        return super().get_latent(masked, masks=masks, hidden_state=hidden_state)

    def as_jit(self):
        return _BlindRangeActorExport(self)

    def as_onnx(self, verbose=False):
        return _BlindRangeActorExport(self, verbose)


class _BlindRangeActorExport(_RangeActorExport):
    """Preserve the two-input interface but ignore its visual input by construction."""

    def forward(self, proprio_obs: torch.Tensor, range_features: torch.Tensor) -> torch.Tensor:
        prop = self.prop_mlp(self.obs_normalizer(proprio_obs))
        sectors = torch.zeros_like(range_features).reshape(-1, 8, 64)
        directions = self.sector_directions.expand(sectors.shape[0], -1, -1)
        visual = F.silu(self.fusion(torch.cat((sectors, directions), -1))).flatten(1)
        latent = torch.cat((prop, self.range_adapter(visual)), -1)
        return self.deterministic_output(self.mlp(latent))
