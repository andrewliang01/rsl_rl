"""Exp-H: bounded observed memory, control-conditioned reading and separate completion.

PPO replays a detached pre-reader cache, never images or live recurrent state.
The shared reader belongs to the actor optimizer; the algorithm partitions its
auxiliary optimizer and constrains auxiliary changes to the policy interface.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.range_jepa import RangeJEPA
from rsl_rl.modules import MLP


class EvidenceReader(nn.Module):
    """A small shared query interface; support is an observation fraction, not certainty."""

    def __init__(self, image_height=40, image_width=96, channels=32, hidden_size=128,
                 bank_size=32, context_grid=(4, 8), vertical_fov=(52., -90.), attention_heads=4):
        super().__init__()
        self.settings = dict(image_height=image_height, image_width=image_width, channels=channels,
                             hidden_size=hidden_size, bank_size=bank_size, context_grid=tuple(context_grid),
                             vertical_fov=tuple(vertical_fov), attention_heads=attention_heads)
        self.channels, self.hidden_size = channels, hidden_size
        self.bank_size, self.context_grid = bank_size, tuple(context_grid)
        self.grid = ((image_height + 3) // 4, (image_width + 3) // 4)
        self.context_size = math.prod(self.context_grid)
        self.cache_size = hidden_size + self.context_size * (channels + 1)
        self.bank = nn.Linear(hidden_size, bank_size * channels)
        self.query = nn.Sequential(nn.Linear(3, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.body_query = nn.Linear(64, channels, bias=False)
        self.readout = nn.MultiheadAttention(channels, attention_heads, batch_first=True, dropout=0.)
        self.norm = nn.LayerNorm(channels)
        self.completion_mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(),
                                            nn.Linear(channels * 2, channels))
        self.support_head = nn.Linear(channels, 1)
        az = (torch.arange(image_width) + .5) * (2 * math.pi / image_width) - math.pi
        el = torch.deg2rad(vertical_fov[0] + (torch.arange(image_height) + .5) / image_height
                          * (vertical_fov[1] - vertical_fov[0]))
        el, az = torch.meshgrid(el, az, indexing='ij')
        rays = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), 0)[None]
        for name, grid in [('directions', self.grid), ('context_directions', self.context_grid)]:
            d = F.adaptive_avg_pool2d(rays, grid).flatten(2).transpose(1, 2)
            self.register_buffer(name, F.normalize(d, dim=-1).contiguous())

    def forward(self, cache, body_features, *, recover_only=False):
        if cache.ndim != 2 or cache.shape[-1] != self.cache_size:
            raise ValueError(f'Expected immutable pre-reader cache [B,{self.cache_size}].')
        if body_features.shape != (len(cache), 64):
            raise ValueError('Capture/control-aligned body features must be [B,64].')
        h, tokens, coverage = cache.split((self.hidden_size, self.context_size * self.channels,
                                            self.context_size), dim=-1)
        current = tokens.reshape(-1, self.context_size, self.channels)
        current = current + self.query(self.context_directions)
        memory = self.bank(h).reshape(-1, self.bank_size, self.channels)
        keys = torch.cat((memory, current), 1)
        invalid = torch.cat((torch.zeros(len(cache), self.bank_size, device=cache.device, dtype=torch.bool),
                             coverage <= 0), 1)
        body_query = self.body_query(body_features)
        query = self.query(self.directions) + body_query[:, None]
        # Memory slots always remain present, including empty starts: no all-masked softmax.
        e = self.norm(query + self.readout(query, keys, keys, key_padding_mask=invalid,
                                         need_weights=False)[0])
        if recover_only:
            # CAMP's Actor does not consume the angular completion/support
            # heads. Return the shared projections for its terrain query too.
            return dict(e=e, reader_memory=memory, reader_body_query=body_query)
        delta = self.completion_mlp(e)
        logits = self.support_head(e)
        return dict(e=e, delta=delta, z=e + delta, support_logits=logits)

    def pool(self, tokens):
        image = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], *self.grid)
        return F.adaptive_avg_pool2d(image, (1, 8)).squeeze(2).transpose(1, 2)


class RangeEvidenceJEPA(RangeJEPA):
    """Three-view auxiliary learning; runtime state is only the observed GRU summary."""

    requires_body_features = True
    history_offsets = ()
    default_weights = dict(obs=1., codec=1., reg=.01, mem=1., full=1., range=1.,
                           map_range=1., support=.1, known=.5)
    metric_names = ('map_range', 'support', 'known', 'rmse_m', 'near_rmse_m',
                    'map_coverage', 'support_brier')

    def __init__(self, image_height=40, image_width=96, channels=32, hidden_size=128,
                 bank_size=32, sectors=8, vertical_fov=(52., -90.), max_range=40.,
                 attention_heads=4, input_normalized=True, variance_floor=.5, covariance_weight=.01,
                 variant='H', context_grid=(4, 8), capture_dt=.1, huber_delta_m=.1, near_range_m=3., decoder_kind='linear',
                 image_encoding='depth_valid', directional_memory_write=False, auxiliary_objective='hybrid'):
        if auxiliary_objective not in ('hybrid', 'direct_geometry'):
            raise ValueError('auxiliary_objective must be hybrid or direct_geometry.')
        if type(directional_memory_write) is not bool:
            raise TypeError('directional_memory_write must be bool.')
        super().__init__(image_height, image_width, channels, hidden_size, bank_size, sectors,
                         vertical_fov, max_range, attention_heads, input_normalized,
                         variance_floor, covariance_weight, image_encoding=image_encoding)
        if variant != 'H' or len(context_grid) != 2 or any(type(v) is not int or v <= 0 for v in context_grid):
            raise ValueError('Exp-H requires a positive two-dimensional context grid.')
        if min(capture_dt, huber_delta_m, near_range_m) <= 0:
            raise ValueError('Capture period and metric depth-loss scales must be positive.')
        self.variant, self.capture_dt = variant, capture_dt
        self.auxiliary_objective = auxiliary_objective
        if auxiliary_objective == 'direct_geometry':
            self.default_weights = dict(self.default_weights, mem=0., full=0.)
        self.huber_delta_m, self.near_range_m = huber_delta_m, near_range_m
        # Retain the shared CNN and original small GRU; remove unused A0 predictors.
        for name in ('memory_bank', 'query', 'readout', 'recovery_norm', 'completion',
                     'completion_norm', 'completion_mlp', 'decoder'):
            delattr(self, name)
        self.reader = EvidenceReader(image_height, image_width, channels, hidden_size, bank_size,
                                     context_grid, vertical_fov, attention_heads)
        self.feature_size = self.reader.cache_size
        if decoder_kind != 'linear':
            raise ValueError('Exp-H speed baseline uses the original linear decoder.')
        self.decoder = nn.Linear(channels, 2)
        self.bind_reader(self.reader)
        # Zero initialization is an exact identity and consumes no random draws.
        # The flag is opt-in: legacy checkpoints retain exactly their old keys.
        self.directional_memory_write = directional_memory_write
        self.register_parameter('directional_write_weight',
            nn.Parameter(torch.zeros(channels, 3)) if directional_memory_write else None)

    def bind_reader(self, reader):
        if reader.settings != self.reader.settings:
            raise ValueError('Actor and auxiliary reader geometry/dimensions differ.')
        self.reader = reader
        # Play's existing observer hooks the actual residual, without a second inference.
        # This alias must not register duplicate parameters in the same state_dict.
        object.__setattr__(self, 'completion_mlp', reader.completion_mlp)

    def condition_context(self, encoded, body_features=None):
        # Body conditions querying, never writes inferred information into observed memory.
        return encoded

    def _pool_memory_input(self, tokens, coverage):
        """Bind token content to its fixed local direction before spatial pooling.

        This changes only the observed-memory write. Current context and shared
        target encodings keep the original tokens; coverage stays an observation
        fraction. It adds neither pose input nor a confidence/time channel.
        """
        if self.directional_write_weight is not None:
            scale = 1. + torch.tanh(F.linear(self.reader.directions, self.directional_write_weight))
            tokens = tokens * scale
        weight = self.pool_sectors(coverage.expand(-1, -1, self.channels))
        return (self.pool_sectors(tokens * coverage) / weight.clamp_min(1e-6)).flatten(1)

    def _step_encoded(self, encoded, hidden, reset=None, body_features=None):
        tokens, coverage = encoded['tokens'], encoded['coverage']
        if hidden.shape != (len(tokens), self.hidden_size):
            raise ValueError('Hidden state must match the selected capture rows.')
        if reset is not None:
            hidden = torch.where(reset.reshape(-1, 1).bool(), torch.zeros_like(hidden), hidden)
        proposal = self.gru(self._pool_memory_input(tokens, coverage), hidden)
        hidden = torch.where(coverage.sum(1) > 0, proposal, hidden)
        shape = (-1, self.channels, self.token_height, self.token_width)
        cov_image = coverage.transpose(1, 2).reshape(-1, 1, self.token_height, self.token_width)
        cov = F.adaptive_avg_pool2d(cov_image, self.reader.context_grid)
        context = F.adaptive_avg_pool2d((tokens * coverage).transpose(1, 2).reshape(shape),
                                       self.reader.context_grid) / cov.clamp_min(1e-6)
        cache = torch.cat((hidden, context.flatten(2).transpose(1, 2).flatten(1), cov.flatten(1)), -1)
        return dict(hidden=hidden, features=cache)

    def step(self, image, hidden, reset=None, body_features=None):
        return self._step_encoded(self.encode(image), hidden, reset)

    def decode(self, tokens):
        spatial = self.decoder(tokens).transpose(1, 2).reshape(-1, 2, self.token_height, self.token_width)
        return F.interpolate(spatial.float(), size=(self.image_height, self.image_width),
                             mode='bilinear', align_corners=False)

    def _reconstruction(self, prediction, target, *, predict_mask=True):
        error = F.smooth_l1_loss(prediction[:, :1] * self.max_range,
                                 target[:, :1] * self.max_range, beta=self.huber_delta_m, reduction='none')
        # Focus metric geometry on nearby surfaces without dropping distant supervision.
        weight = target[:, 1:2] * (1. + (target[:, :1] * self.max_range < self.near_range_m))
        result = self._weighted(error, weight)
        if predict_mask:
            result = result + F.binary_cross_entropy_with_logits(prediction[:, 1:2], target[:, 1:2],
                                                                  reduction='none').mean((1, 2, 3))
        return result

    def auxiliary_loss(self, observed, mapped, simulated, initial_hidden=None, reset=None, support=None,
                       active=None, weights=None, profile=None, body_features=None):
        if self.auxiliary_objective == 'direct_geometry':
            from rsl_rl.models.range_direct_objective import direct_geometry_loss
            return direct_geometry_loss(self, observed, mapped, simulated, initial_hidden=initial_hidden,
                reset=reset, support=support, active=active, weights=weights,
                profile=profile, body_features=body_features)
        if observed.ndim == 4:
            observed, mapped, simulated = observed[None], mapped[None], simulated[None]
            body_features = None if body_features is None else body_features[None]
        if observed.ndim != 5 or observed.shape != mapped.shape or observed.shape != simulated.shape:
            raise ValueError('Three aligned [T,B,2,H,W] views are required.')
        T, B = observed.shape[:2]
        if not T or not B or body_features is None or body_features.shape != (T, B, 64):
            raise ValueError('Auxiliary captures require paired [T,B,64] body features.')
        def flags(x, default):
            return torch.full((T, B), default, device=observed.device, dtype=torch.bool) if x is None else x.reshape(T, B).bool()
        reset, active = flags(reset, False), flags(active, True)
        w = dict(self.default_weights)
        if weights:
            if set(weights) - set(w) or any(not math.isfinite(float(v)) or v < 0 for v in weights.values()):
                raise ValueError('Unknown or invalid auxiliary weight.')
            w.update(weights)
        hidden = self.initial_hidden(B, observed.device, observed.dtype) if initial_hidden is None else initial_hidden
        terms = {k: [] for k in (*w, 'rmse_m', 'near_rmse_m', 'map_coverage', 'support_brier')}
        phase = lambda name: profile.phase(name, 'forward') if profile is not None else nullcontext()
        if profile is not None:
            profile.count('encoded_image_rows', T * B * 3)
        for t in range(T):
            with phase('shared_cnn'):
                enc = self.encode(torch.cat((observed[t], mapped[t], simulated[t]), 0))
                tok, cov, clean = (enc[k].split(B) for k in ('tokens', 'coverage', 'image'))
            with phase('memory_readout_completion'):
                state = self._step_encoded(dict(tokens=tok[0], coverage=cov[0]), hidden, reset[t] & active[t])
                hidden = torch.where(active[t, :, None], state['hidden'], hidden)
                output = self.reader(state['features'], body_features[t].detach())
            with phase('decoder'):
                pred = self.decode(torch.cat((output['e'], *tok, output['z']), 0)).split(B)
            with phase('loss_regularizer_mask'):
                observed_loss = (self._reconstruction(pred[0], clean[0], predict_mask=False)
                                 + self._reconstruction(pred[4], clean[0], predict_mask=False)) / 2
                terms['obs'].append(observed_loss)
                terms['codec'].append(sum(self._reconstruction(pred[1+i], clean[i]) for i in range(3)) / 3)
                terms['reg'].append(sum(self._regularizer(tok[i], cov[i]) for i in range(3)) / 3)
                valid = clean[1][:, 1:2]
                if support is not None:
                    valid = valid * support.reshape(T, B, 1, self.image_height, self.image_width)[t].bool()
                support_fraction = F.adaptive_avg_pool2d(valid, (self.token_height, self.token_width)).flatten(2).transpose(1, 2)
                terms['mem'].append(self._weighted((output['e'] - tok[1].detach()).square().mean(-1), support_fraction.squeeze(-1)))
                terms['full'].append(self._weighted((output['z'] - tok[2].detach()).square().mean(-1), cov[2].squeeze(-1)))
                terms['support'].append(F.binary_cross_entropy_with_logits(output['support_logits'], support_fraction,
                                                                           reduction='none').mean((1, 2)))
                terms['map_range'].append(self._reconstruction(pred[0], clean[1]))
                terms['range'].append(self._reconstruction(pred[4], clean[2]))
                consistent = valid * clean[2][:, 1:2] * (((clean[1][:, :1] - clean[2][:, :1]).abs() * self.max_range) <= self.huber_delta_m)
                known_target = torch.cat((clean[1][:, :1], consistent), 1)
                terms['known'].append(self._reconstruction(pred[4], known_target, predict_mask=False))
                squared = ((pred[4][:, :1] - clean[2][:, :1]) * self.max_range).square()
                terms['rmse_m'].append(self._weighted(squared, clean[2][:, 1:2]).sqrt())
                terms['near_rmse_m'].append(self._weighted(squared, clean[2][:, 1:2] * (clean[2][:, :1] * self.max_range < self.near_range_m)).sqrt())
                terms['map_coverage'].append(valid.mean((1, 2, 3)))
                terms['support_brier'].append((output['support_logits'].sigmoid() - support_fraction).square().mean((1, 2)))
        denominator = active.sum().clamp_min(1)
        result = {k: (torch.stack(v) * active).sum() / denominator for k, v in terms.items()}
        result.update(loss=sum(w[k] * result[k] for k in w), hidden=hidden, anchor_count=active.sum())
        return result


class RangeEvidenceActor(MLPModel):
    """PPO trains the reader on frozen capture candidates, not a cached final latent."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation='elu',
                 obs_normalization=True, distribution_cfg=None, reader_cfg=None, reader_amp=True, reader_chunk_size=4096, reader_checkpoint=True, reader_ppo_gradient=True, range_features_set='range_features'):
        self.range_features_set = range_features_set
        groups = dict(obs_groups)
        groups[obs_set] = [k for k in groups[obs_set] if k != range_features_set]
        super().__init__(obs, groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         copy.deepcopy(distribution_cfg))
        self.prop_mlp = MLP(self.obs_dim, 64, (128,), activation)
        if type(reader_amp) is not bool:
            raise TypeError('reader_amp must be boolean.')
        self.reader_amp = reader_amp
        if type(reader_chunk_size) is not int or reader_chunk_size < 1 or type(reader_checkpoint) is not bool:
            raise ValueError('Reader chunk size must be positive and checkpoint flag boolean.')
        self.reader_chunk_size, self.reader_checkpoint = reader_chunk_size, reader_checkpoint
        if type(reader_ppo_gradient) is not bool:
            raise TypeError('reader_ppo_gradient must be boolean.')
        self.reader_ppo_gradient = reader_ppo_gradient
        self.reader = EvidenceReader(**(reader_cfg or {}))
        if obs[range_features_set].shape[-1] != self.reader.cache_size:
            raise ValueError('Actor cache dimensions differ from the configured reader.')
        self.fusion = nn.Sequential(nn.Linear(67, 32), nn.ELU())
        self.range_adapter = nn.Sequential(nn.Linear(256, 64), nn.ELU())
        angles = (torch.arange(8) + .5) * (2 * math.pi / 8) - math.pi
        self.register_buffer('sector_directions', torch.stack((angles.cos(), angles.sin()), -1)[None])

    def _get_latent_dim(self):
        return 128

    def proprioceptive_observations(self, obs):
        return torch.cat([obs[k] for k in self.obs_groups], -1)

    def encode_proprioception(self, observations):
        return self.prop_mlp(self.obs_normalizer(observations))

    def _read_visual(self, cache, prop):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.reader_ppo_gradient):
            with torch.autocast(device_type=prop.device.type, dtype=torch.bfloat16,
                                enabled=self.reader_amp and prop.is_cuda):
                output = self.reader(cache, prop)
        evidence, residual = self.reader.pool(output['e']), self.reader.pool(output['delta'])
        support = self.reader.pool(output['support_logits'].sigmoid()).detach()
        dirs = self.sector_directions.expand(len(prop), -1, -1)
        return self.fusion(torch.cat((evidence, residual, support, dirs), -1)).flatten(1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        prop = self.encode_proprioception(self.proprioceptive_observations(obs))
        cache = obs[self.range_features_set].detach()
        parts = []
        for start in range(0, len(prop), self.reader_chunk_size):
            args = cache[start:start+self.reader_chunk_size], prop[start:start+self.reader_chunk_size]
            if self.reader_checkpoint and self.reader_ppo_gradient and torch.is_grad_enabled():
                # Checkpoint the small pooled result, not hundreds of angular tokens.
                # No RNG or mutable sensor state occurs inside the reader.
                parts.append(checkpoint(self._read_visual, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                parts.append(self._read_visual(*args))
        visual = torch.cat(parts, 0)
        return torch.cat((prop, self.range_adapter(visual)), -1)

    def as_jit(self):
        raise NotImplementedError('Exp-H needs explicit raw-range state export; use --skip_export for Play.')

    def as_onnx(self, **kwargs):
        raise NotImplementedError('Exp-H needs explicit raw-range state export; use --skip_export for Play.')
