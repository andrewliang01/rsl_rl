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


def _domain_reconstruction(model, prediction, target, domain=None, *, predict_mask=True):
    """Ignore out-of-domain pixels, including validity logits, without relabeling them."""
    if domain is None:
        return model._reconstruction(prediction, target, predict_mask=predict_mask)
    local_target = torch.cat((target[:, :1], target[:, 1:2] * domain), 1)
    result = model._reconstruction(prediction, local_target, predict_mask=False)
    if predict_mask:
        mask_error = F.binary_cross_entropy_with_logits(
            prediction[:, 1:2], target[:, 1:2], reduction='none')
        result = result + model._weighted(mask_error, domain.to(mask_error.dtype))
    return result


def direct_geometry_loss(model, observed, mapped, simulated, initial_hidden=None,
                         reset=None, support=None, active=None, weights=None,
                         profile=None, body_features=None, terrain_targets=None, map_domains=None, batch_time=False):
    if observed.ndim == 4:
        observed, mapped, simulated = observed[None], mapped[None], simulated[None]
        body_features = None if body_features is None else body_features[None]
        terrain_targets = None if terrain_targets is None else terrain_targets[None]
        map_domains = None if map_domains is None else map_domains[None]
    if observed.ndim != 5 or observed.shape != mapped.shape or observed.shape != simulated.shape:
        raise ValueError('Three aligned [T,B,2,H,W] views are required.')
    time_steps, batch_size = observed.shape[:2]
    if not time_steps or not batch_size or body_features is None or body_features.shape != (time_steps, batch_size, 64):
        raise ValueError('Auxiliary captures require paired [T,B,64] body features.')
    if map_domains is not None:
        if (map_domains.shape != observed.shape or map_domains.dtype != torch.bool
                or map_domains.device != observed.device):
            raise ValueError('Map domains must be bool [T,B,2,H,W], aligned with capture images.')
        map_domains = map_domains.detach()

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
    if batch_time:
        return _batched_geometry_sequence(model, observed, mapped, simulated, hidden, reset, active,
                                          loss_weights, body_features, terrain_targets, map_domains,
                                          support, phase)
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
            per_image = _direct_geometry_terms(
                model, tokens, coverage, clean_obs, clean_map, clean_sim, output,
                pred_map, pred_obs, pred_full,
                domain=None if map_domains is None else map_domains[t, :, 0:1],
                obs_domain=None if map_domains is None else map_domains[t, :, 1:2],
                support=None if support is None else support.reshape(time_steps, batch_size, 1,
                    model.image_height, model.image_width)[t],
                terrain_targets=None if terrain_targets is None else terrain_targets[t])
            for name, value in per_image.items():
                terms[name].append(value)
    denominator = active.sum().clamp_min(1)
    result = {name: (torch.stack(values) * active).sum() / denominator for name, values in terms.items()}
    result.update(loss=sum(loss_weights[name] * result[name] for name in loss_weights),
                  hidden=hidden, anchor_count=active.sum())
    return result


def _direct_geometry_terms(model, tokens, coverage, clean_obs, clean_map, clean_sim,
                           output, pred_map, pred_obs, pred_full, *, domain=None,
                           obs_domain=None, support=None, terrain_targets=None):
    """One loss/metric vector per image; rows never interact across the batch."""
    batch_size = len(tokens)
    terms = {}
    terms['obs'] = (_domain_reconstruction(model, pred_map, clean_obs, obs_domain, predict_mask=False) + model._reconstruction(pred_full, clean_obs, predict_mask=False)) / 2
    terms['codec'] = model._reconstruction(pred_obs, clean_obs)
    terms['reg'] = model._regularizer(tokens, coverage)
    zero = tokens.new_zeros(batch_size)
    terms['mem'] = zero
    terms['full'] = zero
    valid = clean_map[:, 1:2]
    if domain is not None:
        valid = valid * domain
    if support is not None:
        valid = valid * support.bool()
    support_fraction = F.adaptive_avg_pool2d(valid, (model.token_height, model.token_width)).flatten(2).transpose(1, 2)
    token_domain = None
    if domain is not None:
        token_domain = F.adaptive_avg_pool2d(domain.to(valid.dtype), (model.token_height, model.token_width)).flatten(2).transpose(1, 2)
        support_fraction = support_fraction / token_domain.clamp_min(1e-06)
    support_error = F.binary_cross_entropy_with_logits(output['support_logits'], support_fraction, reduction='none')
    terms['support'] = support_error.mean((1, 2)) if token_domain is None else model._weighted(support_error, token_domain)
    terms['map_range'] = _domain_reconstruction(model, pred_map, clean_map, domain)
    terms['range'] = model._reconstruction(pred_full, clean_sim)
    consistent = valid * clean_sim[:, 1:2] * ((clean_map[:, :1] - clean_sim[:, :1]).abs() * model.max_range <= model.huber_delta_m)
    known_target = torch.cat((clean_map[:, :1], consistent), 1)
    terms['known'] = model._reconstruction(pred_full, known_target, predict_mask=False)
    squared = ((pred_full[:, :1] - clean_sim[:, :1]) * model.max_range).square()
    terms['rmse_m'] = model._weighted(squared, clean_sim[:, 1:2]).sqrt()
    near = clean_sim[:, 1:2] * (clean_sim[:, :1] * model.max_range < model.near_range_m)
    terms['near_rmse_m'] = model._weighted(squared, near).sqrt()
    terms['map_coverage'] = valid.mean((1, 2, 3)) if domain is None else model._weighted(valid, domain.to(valid.dtype))
    support_squared = (output['support_logits'].sigmoid() - support_fraction).square()
    terms['support_brier'] = support_squared.mean((1, 2)) if token_domain is None else model._weighted(support_squared, token_domain)
    if getattr(model, 'has_terrain_head', False):
        if terrain_targets is None:
            raise ValueError('CAMP requires capture-aligned three-source terrain targets.')
        for (name, value) in model.terrain_loss(output, terrain_targets.detach()).items():
            terms[name] = value
    return terms


def _batched_geometry_sequence(model, observed, mapped, simulated, hidden, reset, active,
                               weights, body_features, terrain_targets, map_domains, support, phase):
    """Batch independent image work over time; retain the exact GRU recurrence.

    The shared encoder uses per-image GroupNorm, and reader/loss reductions are
    per image. Their time and sequence axes can therefore be flattened without
    changing TBPTT boundaries, active-row weighting, or optimizer update count.
    """
    time_steps, batch_size = observed.shape[:2]
    rows = time_steps * batch_size
    def flatten(image):
        return image.reshape(rows, *image.shape[2:])
    with phase('shared_cnn'):
        encoded = model.encode(flatten(observed))
        clean_map = model._clean(flatten(mapped).detach())
        clean_sim = model._clean(flatten(simulated).detach())
    with phase('memory_readout_completion'):
        sequential = {key: value.reshape(time_steps, batch_size, *value.shape[1:])
                      for key, value in encoded.items()}
        features = []
        for t in range(time_steps):
            frame = {key: value[t] for key, value in sequential.items()}
            state = model._step_encoded(frame, hidden, reset[t] & active[t])
            hidden = torch.where(active[t, :, None], state['hidden'], hidden)
            features.append(state['features'])
        output = model.reader(torch.cat(features, 0), body_features.reshape(rows, 64).detach())
    with phase('decoder'):
        predictions = model.decode(torch.cat((output['e'], encoded['tokens'], output['z']), 0)).split(rows)
    with phase('loss_regularizer_mask'):
        domains = None if map_domains is None else flatten(map_domains)
        per_image = _direct_geometry_terms(
            model, encoded['tokens'], encoded['coverage'], encoded['image'], clean_map, clean_sim,
            output, *predictions,
            domain=None if domains is None else domains[:, 0:1],
            obs_domain=None if domains is None else domains[:, 1:2],
            support=None if support is None else support.reshape(rows, 1, model.image_height, model.image_width),
            terrain_targets=None if terrain_targets is None else flatten(terrain_targets))
        denominator = active.sum().clamp_min(1)
        result = {name: (value.reshape(time_steps, batch_size) * active).sum() / denominator
                  for name, value in per_image.items()}
        result.update(loss=sum(weights[name] * result[name] for name in weights),
                      hidden=hidden, anchor_count=active.sum())
    return result
