"""Exp-H keeps RSL's AMP/PPO loops and the existing fresh-capture auxiliary loop."""
from copy import deepcopy
import math

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO
from rsl_rl.models.range_evidence import RangeEvidenceActor, RangeEvidenceJEPA


class RangeEvidenceAMPPPO(RangeJEPAAMPPPO):
    def __init__(self, actor, critic, storage, perception_cfg=None, auxiliary_policy_kl_limit=.01,
                 auxiliary_policy_probe_size=128, **kwargs):
        if not math.isfinite(auxiliary_policy_kl_limit) or auxiliary_policy_kl_limit <= 0:
            raise ValueError('A positive auxiliary policy KL limit is required.')
        if type(auxiliary_policy_probe_size) is not int or auxiliary_policy_probe_size < 1:
            raise ValueError('A positive auxiliary policy probe size is required.')
        if not isinstance(actor, RangeEvidenceActor):
            raise TypeError('Exp-H requires RangeEvidenceActor.')
        super().__init__(actor, critic, storage, perception_cfg=perception_cfg, **kwargs)
        if not isinstance(self.perception, RangeEvidenceJEPA):
            raise TypeError('Exp-H requires variant H perception.')
        self.auxiliary_policy_kl_limit = auxiliary_policy_kl_limit
        self.auxiliary_policy_probe_size = auxiliary_policy_probe_size
        self.perception.bind_reader(self.actor.reader)
        actor_ids = {id(p) for p in self.actor.parameters()}
        aux_params = [p for p in self.perception.parameters() if id(p) not in actor_ids]
        lr = self.perception_optimizer.param_groups[0]['lr']
        self.perception_optimizer = torch.optim.Adam(aux_params, lr=lr)
        own = {id(p) for g in self.perception_optimizer.param_groups for p in g['params']}
        policy = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        if own & policy or not {id(p) for p in self.perception.parameters()} <= own | policy:
            raise RuntimeError('Each trainable perception parameter must have exactly one optimizer owner.')

    def _zero_perception_grad(self):
        # Clear stale Actor/Critic gradients before an auxiliary-only reader step.
        self.optimizer.zero_grad(set_to_none=True)
        super()._zero_perception_grad()

    def _step_perception_optimizer(self):
        # Reader/support gradients use their sole PPO Adam state. Body/Actor/Critic
        # have grad=None here: auxiliary losses receive detached body conditions.
        self.optimizer.step()
        super()._step_perception_optimizer()

    @torch.no_grad()
    def _policy_probe(self):
        observations = self.storage.observations
        total = math.prod(observations.batch_size)
        idx = torch.linspace(0, total - 1, min(total, self.auxiliary_policy_probe_size),
                             device=self.device).long()
        keys = list(self.actor.obs_groups) + [self.actor.range_features_set]
        return TensorDict({k: observations[k].reshape(total, -1).index_select(0, idx) for k in keys},
                          batch_size=[len(idx)], device=self.device)

    @torch.no_grad()
    def _distribution_params(self, obs):
        # Update distribution without sampling or advancing the random generator.
        self.actor.distribution.update(self.actor.mlp(self.actor.get_latent(obs)))
        return tuple(p.detach().clone() for p in self.actor.output_distribution_params)

    @torch.inference_mode(False)
    @torch.enable_grad()
    def _update_perception(self):
        probe = self._policy_probe()
        params = list(self.actor.reader.parameters())
        before = deepcopy(self.actor.reader.state_dict())
        old_optimizer = {p: deepcopy(self.optimizer.state[p]) if p in self.optimizer.state else None
                         for p in params}
        old_distribution = self._distribution_params(probe)
        result = super()._update_perception()
        with torch.no_grad():
            new_distribution = self._distribution_params(probe)
            kl = self.actor.get_kl_divergence(old_distribution, new_distribution).mean()
            if self.is_multi_gpu:
                # One rank must not retain a reader update rejected by another.
                torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.MAX)
            rejected = not bool(torch.isfinite(kl)) or float(kl) > self.auxiliary_policy_kl_limit
            if rejected:
                self.actor.reader.load_state_dict(before)
                for p, state in old_optimizer.items():
                    if state is None:
                        self.optimizer.state.pop(p, None)
                    else:
                        self.optimizer.state[p] = state
                self._distribution_params(probe)
        self.optimizer.zero_grad(set_to_none=True)
        self.perception_optimizer.zero_grad(set_to_none=True)
        result.update({'range_h/aux_policy_kl_proposed': float(kl),
                       'range_h/aux_reader_rejected': float(rejected),
                       'range_h/ppo_cache_entries': self.perception.feature_size})
        self.last_perception_receipt.update(ppo_uses_cached_features_only=False,
            ppo_cache='detached capture-time hidden/current tokens/coverage; reader recomputed',
            auxiliary_policy_kl_proposed=float(kl), auxiliary_reader_rejected=rejected,
            optimizer_ownership='reader/support: PPO Adam; CNN/GRU/decoder: auxiliary Adam')
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if strict and (load_cfg is None or load_cfg.get('actor', False)) and (
                load_cfg is None or load_cfg.get('perception', load_cfg.get('actor', False))):
            # Shared tensors are serialized from both roots: reject contradictory copies.
            actor = loaded_dict.get('actor_state_dict', {})
            perception = loaded_dict.get('perception_state_dict', {})
            for name, value in actor.items():
                if name.startswith('reader.') and name in perception and not torch.equal(value, perception[name]):
                    raise ValueError(f'Checkpoint contains inconsistent shared reader tensor: {name}')
        return super().load(loaded_dict, load_cfg=load_cfg, strict=strict)
