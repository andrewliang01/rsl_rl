"""Direct two-level depth supervision for the existing small H model.

Only observed scans are CNN inputs. Map/full images remain detached metric
targets; neither target is encoded. Runtime, reader, memory and PPO are shared
with H. The observed codec/regularizer remain; target codecs and latent matching
are absent, so this is a supervision-package experiment, not a JEPA-only ablation.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F


def direct_geometry_loss(model, observed, mapped, simulated, initial_hidden=None,
                         reset=None, support=None, active=None, weights=None,
                         profile=None, body_features=None, terrain_targets=None):
    if observed.ndim == 4:
        observed, mapped, simulated = observed[None], mapped[None], simulated[None]
        body_features = None if body_features is None else body_features[None]
        terrain_targets = None if terrain_targets is None else terrain_targets[None]
    if observed.ndim != 5 or observed.shape != mapped.shape or observed.shape != simulated.shape:
        raise ValueError('Three aligned [T,B,2,H,W] views are required.')
    time_steps, batch_size = observed.shape[:2]
    if not time_steps or not batch_size or body_features is None or body_features.shape != (time_steps, batch_size, 64):
        raise ValueError('Auxiliary captures require paired [T,B,64] body features.')

    def flags(value, default):
        if value is None:
            return torch.full((time_steps, batch_size), default, device=observed.device, dtype=torch.bool)
        return value.reshape(time_steps, batch_size).bool()

    reset, active = flags(reset, False), flags(active, True)
    loss_weights = dict(model.default_weights)
    if weights:
        if set(weights) - set(loss_weights) or any(not math.isfinite(float(v)) or v < 0 for v in weights.values()):
            raise ValueError('Unknown or invalid auxiliary weight.')
        loss_weights.update(weights)
    if loss_weights['mem'] != 0 or loss_weights['full'] != 0:
        raise ValueError('direct_geometry excludes mem/full latent losses; their weights must be zero.')
    hidden = model.initial_hidden(batch_size, observed.device, observed.dtype) if initial_hidden is None else initial_hidden
    names = tuple(dict.fromkeys((*loss_weights, 'rmse_m', 'near_rmse_m', 'map_coverage', 'support_brier',
                                *getattr(model, 'terrain_metric_names', ()))))
    terms = {name: [] for name in names}
    phase = lambda name: profile.phase(name, 'forward') if profile is not None else nullcontext()
    if profile is not None:
        profile.count('encoded_image_rows', time_steps * batch_size)
    for t in range(time_steps):
        with phase('shared_cnn'):
            encoded = model.encode(observed[t])
            tokens, coverage, clean_obs = (encoded[k] for k in ('tokens', 'coverage', 'image'))
            clean_map = model._clean(mapped[t].detach())
            clean_sim = model._clean(simulated[t].detach())
        with phase('memory_readout_completion'):
            state = model._step_encoded(encoded, hidden, reset[t] & active[t])
            hidden = torch.where(active[t, :, None], state['hidden'], hidden)
            output = model.reader(state['features'], body_features[t].detach())
        with phase('decoder'):
            pred_map, pred_obs, pred_full = model.decode(torch.cat((output['e'], tokens, output['z']), 0)).split(batch_size)
        with phase('loss_regularizer_mask'):
            terms['obs'].append((model._reconstruction(pred_map, clean_obs, predict_mask=False)
                                 + model._reconstruction(pred_full, clean_obs, predict_mask=False)) / 2)
            terms['codec'].append(model._reconstruction(pred_obs, clean_obs))
            terms['reg'].append(model._regularizer(tokens, coverage))
            zero = tokens.new_zeros(batch_size)
            terms['mem'].append(zero)
            terms['full'].append(zero)
            valid = clean_map[:, 1:2]
            if support is not None:
                valid = valid * support.reshape(time_steps, batch_size, 1, model.image_height, model.image_width)[t].bool()
            support_fraction = F.adaptive_avg_pool2d(valid, (model.token_height, model.token_width)).flatten(2).transpose(1, 2)
            terms['support'].append(F.binary_cross_entropy_with_logits(output['support_logits'], support_fraction,
                                                                       reduction='none').mean((1, 2)))
            terms['map_range'].append(model._reconstruction(pred_map, clean_map))
            terms['range'].append(model._reconstruction(pred_full, clean_sim))
            consistent = valid * clean_sim[:, 1:2] * (((clean_map[:, :1] - clean_sim[:, :1]).abs()
                                                       * model.max_range) <= model.huber_delta_m)
            known_target = torch.cat((clean_map[:, :1], consistent), 1)
            terms['known'].append(model._reconstruction(pred_full, known_target, predict_mask=False))
            squared = ((pred_full[:, :1] - clean_sim[:, :1]) * model.max_range).square()
            terms['rmse_m'].append(model._weighted(squared, clean_sim[:, 1:2]).sqrt())
            near = clean_sim[:, 1:2] * (clean_sim[:, :1] * model.max_range < model.near_range_m)
            terms['near_rmse_m'].append(model._weighted(squared, near).sqrt())
            terms['map_coverage'].append(valid.mean((1, 2, 3)))
            terms['support_brier'].append((output['support_logits'].sigmoid() - support_fraction).square().mean((1, 2)))
            if getattr(model, 'has_terrain_head', False):
                if terrain_targets is None or terrain_targets.shape[:2] != (time_steps, batch_size):
                    raise ValueError('CAMP requires capture-aligned three-source terrain targets.')
                for name, value in model.terrain_loss(output, terrain_targets[t].detach()).items():
                    terms[name].append(value)
    denominator = active.sum().clamp_min(1)
    result = {name: (torch.stack(values) * active).sum() / denominator for name, values in terms.items()}
    result.update(loss=sum(loss_weights[name] * result[name] for name in loss_weights),
                  hidden=hidden, anchor_count=active.sum())
    return result
