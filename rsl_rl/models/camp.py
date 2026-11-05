"""HDirect recovery with confidence-aware local terrain and direct joint control.

The old observed CNN/GRU/angular recovery stays intact. A small XY reader adds
surface geometry and confidence supervision beyond the physical sensor FOV.
No reference image, world pose, future foot target or simulator map is an Actor input.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .range_evidence import EvidenceReader, RangeEvidenceActor, RangeEvidenceJEPA
from .range_direct_objective import direct_geometry_loss


class CampTerrainReader(EvidenceReader):
    def __init__(self, bev_grid_size=32, bev_extent_m=2.4, source_priors=(1., .85, .35), **kwargs):
        super().__init__(**kwargs)
        if bev_grid_size != 32 or bev_extent_m <= 0:
            raise ValueError('CAMP decoder uses a 32-cell local surface grid and positive extent.')
        if len(source_priors) != 3 or not 1. >= source_priors[0] > source_priors[1] > source_priors[2] >= 0.:
            raise ValueError('Evidence priors must decrease from current to history to inferred.')
        self.settings.update(bev_grid_size=bev_grid_size, bev_extent_m=bev_extent_m,
                             source_priors=tuple(source_priors))
        unit = (torch.arange(8) + .5) / 8 - .5
        y, x = torch.meshgrid(unit, unit, indexing='ij')
        self.register_buffer('xy_queries', torch.stack((x, y, torch.ones_like(x)), -1).reshape(1, 64, 3))
        self.register_buffer('source_priors', torch.tensor(source_priors).reshape(1, 3, 1, 1))
        self.xy_encoder = nn.Sequential(nn.Linear(3, self.channels), nn.SiLU(), nn.Linear(self.channels, self.channels))
        self.terrain_attention = nn.MultiheadAttention(self.channels, self.settings['attention_heads'], batch_first=True)
        self.terrain_norm = nn.LayerNorm(self.channels)
        self.terrain_completion = nn.Sequential(nn.Linear(self.channels, self.channels * 2), nn.SiLU(),
                                                nn.Linear(self.channels * 2, self.channels))
        self.surface_decoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 6, 3, padding=1))

    def forward(self, cache, body_features, *, actor_only=False):
        output = super().forward(cache, body_features, recover_only=actor_only)
        if actor_only:
            memory = output.pop('reader_memory')
            body_query = output.pop('reader_body_query')
        else:
            memory = self.bank(cache[:, :self.hidden_size]).reshape(-1, self.bank_size, self.channels)
            body_query = self.body_query(body_features)
        query = self.xy_encoder(self.xy_queries) + body_query[:, None]
        keys = torch.cat((output['e'], memory), 1)
        recovered = self.terrain_norm(query + self.terrain_attention(query, keys, keys, need_weights=False)[0])
        completed = recovered + self.terrain_completion(recovered)
        n = len(cache)
        if actor_only:
            # Supervised learning still decodes both branches. Control only
            # consumes the completed terrain; avoid the unused mapped decoder.
            full = self.surface_decoder(completed.transpose(1, 2).reshape(n, self.channels, 8, 8))
        else:
            decoded = self.surface_decoder(torch.cat((recovered, completed)).transpose(1, 2).reshape(2 * n, self.channels, 8, 8))
            mapped, full = decoded.split(n)
            output['terrain_mapped'] = mapped[:, :2]
        probability = full[:, 2:5].softmax(1)
        confidence = full[:, 5:6].sigmoid() * (probability * self.source_priors).sum(1, keepdim=True) * full[:, 1:2].sigmoid()
        output.update(terrain_e=recovered, terrain_z=completed,
                      terrain_full=full[:, :2], terrain_source_logits=full[:, 2:5],
                      terrain_accuracy_logits=full[:, 5:6], terrain_confidence=confidence)
        return output


class CampTerrainMemory(RangeEvidenceJEPA):
    has_terrain_head = True
    terrain_metric_names = ('terrain_rmse_m', 'confidence_brier', 'provenance_accuracy')

    def __init__(self, variant='CAMP', bev_grid_size=32, bev_extent_m=2.4,
                 source_priors=(1., .85, .35), confidence_tau_m=.05, **kwargs):
        if variant != 'CAMP' or kwargs.get('auxiliary_objective') != 'direct_geometry':
            raise ValueError('CAMP retains the HDirect three-view recovery objective.')
        super().__init__(variant='H', **kwargs)
        if confidence_tau_m <= 0:
            raise ValueError('Positive geometric confidence scale required.')
        self.confidence_tau_m = confidence_tau_m
        self.reader = CampTerrainReader(**self.reader.settings, bev_grid_size=bev_grid_size,
                                        bev_extent_m=bev_extent_m, source_priors=source_priors)
        self.bind_reader(self.reader)
        self.default_weights = dict(self.default_weights, terrain_map=.5, terrain_full=1., terrain_known=.5,
                                    terrain_confidence=.25, terrain_source=.1)
        self.metric_names = (*self.metric_names, *self.terrain_metric_names,
                             'terrain_map', 'terrain_full', 'terrain_known', 'terrain_confidence', 'terrain_source')

    def auxiliary_loss(self, *args, **kwargs):
        return direct_geometry_loss(self, *args, **kwargs)

    def terrain_loss(self, output, targets):
        if targets.ndim != 5 or targets.shape[1:] != (3, 2, 32, 32):
            raise ValueError('Expected current/history/complete [B,3,2,32,32] terrain targets.')
        observed, mapped, full = targets.unbind(1)
        pred_map, pred_full = output['terrain_mapped'].float(), output['terrain_full'].float()
        valid = full[:, 1:2]

        def geometry(pred, target):
            error = F.smooth_l1_loss(pred[:, :1], target[:, :1], beta=.05, reduction='none')
            return self._weighted(error, target[:, 1:2]) + F.binary_cross_entropy_with_logits(
                pred[:, 1:2], target[:, 1:2], reduction='none').mean((1, 2, 3))

        true_current = (observed[:, 1:2] > .5) & ((observed[:, :1] - full[:, :1]).abs() <= .04)
        true_history = (mapped[:, 1:2] > .5) & ((mapped[:, :1] - full[:, :1]).abs() <= .04)
        source = torch.where(true_current[:, 0], 0, torch.where(true_history[:, 0], 1, 2))
        error = (pred_full[:, :1] - full[:, :1]).abs()
        accuracy = torch.exp(-error.detach() / self.confidence_tau_m) * valid
        quality_loss = F.binary_cross_entropy_with_logits(output['terrain_accuracy_logits'].float(), accuracy,
                                                          reduction='none').mean((1, 2, 3))
        source_loss = F.cross_entropy(output['terrain_source_logits'].float(), source, reduction='none')
        priors = self.reader.source_priors.flatten()[source][:, None]
        confidence_target = accuracy * priors
        known = true_current.to(valid.dtype) * valid
        known_error = F.smooth_l1_loss(pred_full[:, :1], observed[:, :1], beta=.05, reduction='none')
        return dict(terrain_map=geometry(pred_map, mapped), terrain_full=geometry(pred_full, full),
                    terrain_known=self._weighted(known_error, known), terrain_confidence=quality_loss,
                    terrain_source=self._weighted(source_loss, valid[:, 0]),
                    terrain_rmse_m=self._weighted(error.square(), valid).sqrt(),
                    confidence_brier=self._weighted((output['terrain_confidence'].float() - confidence_target).square(), valid),
                    provenance_accuracy=self._weighted((output['terrain_source_logits'].argmax(1) == source).float(), valid[:, 0]))


class CampActor(RangeEvidenceActor):
    """Spatial terrain features + proprioception -> the 29 joint position actions."""

    def __init__(self, *args, reader_cfg=None, terrain_cfg=None, fast_terrain_read=True, **kwargs):
        if type(fast_terrain_read) is not bool:
            raise TypeError('fast_terrain_read must be bool.')
        super().__init__(*args, reader_cfg=reader_cfg, **kwargs)
        self.fast_terrain_read = fast_terrain_read
        self.reader = CampTerrainReader(**(reader_cfg or {}), **(terrain_cfg or {}))
        del self.fusion
        self.range_adapter = nn.Identity()
        self.terrain_encoder = nn.Sequential(nn.Conv2d(2 * self.reader.channels + 3, 32, 3, padding=1), nn.ELU(),
                                             nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 64), nn.ELU())

    def _read_visual(self, cache, prop):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.reader_ppo_gradient):
            with torch.autocast(device_type=prop.device.type, dtype=torch.bfloat16,
                                enabled=self.reader_amp and prop.is_cuda):
                output = self.reader(cache, prop, actor_only=self.fast_terrain_read)
        confidence = F.adaptive_avg_pool2d(output['terrain_confidence'].float().detach(), (8, 8))
        evidence = output['terrain_e'].float().transpose(1, 2).reshape(-1, self.reader.channels, 8, 8)
        residual = (output['terrain_z'] - output['terrain_e']).float().transpose(1, 2).reshape(-1, self.reader.channels, 8, 8)
        height = F.adaptive_avg_pool2d(output['terrain_full'][:, :1].float(), (8, 8))
        valid = F.adaptive_avg_pool2d(output['terrain_full'][:, 1:2].float().sigmoid(), (8, 8))
        spatial = torch.cat((evidence * (.2 + .8 * confidence), residual * confidence, height, valid, confidence), 1)
        return self.terrain_encoder(spatial)
