"""F2 ownership, aligned replay, gradient boundary and actual runner contracts."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAInferencePolicy, _model_configuration
from rsl_rl.models.range_bev import RangeBEVActor
from rsl_rl.models.range_body import RangeBodyConditionedBEV
from rsl_rl.models.range_experiments import RangeExperiment
from rsl_rl.storage.perception_storage import PerceptionStorage
from rsl_rl.storage.range_history import RangeHistory

torch.set_num_threads(1)


def image(n=2):
    x = torch.ones(n, 2, 8, 16)
    x[:, 0] = torch.linspace(.01, .15, 128).reshape(8, 16)
    return x


def model():
    return RangeBodyConditionedBEV(image_height=8, image_width=16)


def actor(n=2, inputs=96):
    obs = TensorDict({'policy': torch.randn(n, inputs), 'range_features': torch.zeros(n, 4096)}, [n])
    a = RangeBEVActor(obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                      representation='latent', hidden_dims=[32], obs_normalization=True)
    return a, obs


def test_zero_residual_matches_f_and_only_projection_parameters_are_added():
    torch.manual_seed(7)
    parent = RangeExperiment(image_height=8, image_width=16, variant='F')
    child = model()
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {'body_projection.weight', 'body_projection.bias'} and not unexpected
    assert sum(p.numel() for p in child.body_projection.parameters()) == 2080
    x = image()
    with torch.no_grad():
        a = parent.step(x, parent.initial_hidden(2))
        b = child.step(x, child.initial_hidden(2), body_features=torch.randn(2, 64))
    torch.testing.assert_close(a['features'], b['features'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='capture-aligned'):
        child.step(x, child.initial_hidden(2))


def test_auxiliary_updates_projection_but_not_actor_body_encoder():
    a, obs = actor()
    m = model()
    body = a.encode_proprioception(a.proprioceptive_observations(obs))
    out = m.auxiliary_loss(image(), image(), image(), body_features=body)
    out['loss'].backward()
    assert m.body_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in a.parameters())
    assert not ({id(p) for p in a.parameters()} & {id(p) for p in m.parameters()})
    a.zero_grad(set_to_none=True)
    m.zero_grad(set_to_none=True)
    a(obs).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.prop_mlp.parameters())
    assert all(p.grad is None for p in m.parameters())


def test_conditioning_preserves_depth_coverage_and_target_encoder_inputs():
    m = model()
    torch.nn.init.normal_(m.body_projection.weight)
    x = image()
    x[:, 1, :3] = 0
    enc = m.encode(x)
    conditioned = m.condition_context(enc, torch.randn(2, 64))
    assert conditioned['coverage'] is enc['coverage']
    assert conditioned['image'] is enc['image']
    assert not torch.equal(conditioned['tokens'], enc['tokens'])
    with patch.object(m.bev_backend, 'encode_target', wraps=m.bev_backend.encode_target) as target:
        m.auxiliary_loss(x, x, x, body_features=torch.randn(2, 64))
    assert target.call_count > 0
    for call in target.call_args_list:
        torch.testing.assert_close(call.args[0], torch.cat((x, x, x)), rtol=0, atol=0)


def raw_obs(policy):
    n = policy.shape[0]
    return TensorDict(dict(policy=policy, range_obs=image(n), range_map=image(n), range_sim=image(n),
                          range_capture_id=torch.zeros(n, 1, dtype=torch.long),
                          range_episode_id=torch.zeros(n, 1, dtype=torch.long)), [n])


@pytest.mark.parametrize('profiled', [False, True])
def test_fresh_pair_ownership_held_capture_and_reset(profiled):
    from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler
    a, obs = actor()
    m = model()
    policy = RangeJEPAInferencePolicy(a, m)
    storage = PerceptionStorage()
    raw = raw_obs(obs['policy'])
    expected = raw['policy'].clone()
    profiler = RangeOnlineProfiler(profiled, 'cpu') if profiled else None
    first = policy.prepare(raw, record_indexed=storage.add_indexed, profile=profiler)
    assert storage.num_anchors == 2
    snap = storage._batches[0]
    torch.testing.assert_close(snap['body_observations'], expected)
    chunk_body = policy.history.body_observations[0]
    torch.testing.assert_close(chunk_body, expected)
    raw['policy'].add_(123)
    held = policy.prepare(raw, record_indexed=storage.add_indexed, profile=profiler)
    torch.testing.assert_close(first['range_features'], held['range_features'], rtol=0, atol=0)
    assert storage.num_anchors == 2 and policy.history.next_chunk == 1
    torch.testing.assert_close(snap['body_observations'], expected)
    torch.testing.assert_close(chunk_body, expected)
    policy.reset(torch.tensor([True, False]))
    assert (policy.history.ids[0] == -1).all()
    assert not policy.features[0].any()
    batches = list(storage.sequence_plan(2, 2))
    eager = storage.sequence_batches(2, 2)
    torch.testing.assert_close(batches[0].body_observations, eager[0].body_observations)


def test_history_replay_reencodes_past_body_with_current_weights_and_prunes():
    class BodyEncoder:
        scale = 1.
        def __call__(self, x):
            return x[:, :1].expand(-1, 64) * self.scale
    class CaptureEncoder:
        requires_body_features = True
        def encode_history(self, images, body_features):
            return {'tokens': body_features[:, :1, None], 'valid': torch.ones(len(images), 1, dtype=torch.bool)}
    h = RangeHistory(2, [1, 2], .1, 'cpu')
    encoder, m = BodyEncoder(), CaptureEncoder()
    for t in range(5):
        raw = torch.full((2, 96), float(t + 1))
        h.append(torch.arange(2), torch.full((2,), t), image(), m.encode_history(image(), encoder(raw)), raw)
        raw.zero_()
    ids, age, _ = h.select(torch.arange(2), torch.full((2,), 5))
    encoder.scale = 7.
    h.begin_replay_pass()
    out = h.replay(ids[None], age[None], m, 1, body_encoder=encoder)
    torch.testing.assert_close(out['history_tokens'][0, 0, :, 0, 0], torch.tensor([35., 28.]))
    assert not out['history_tokens'].requires_grad
    h.end_replay_pass()
    h.prune()
    assert len(h.chunks) == len(h.body_observations) == 3
    assert set(h.chunks) == set(h.body_observations)
    with pytest.raises(ValueError, match='body encoder'):
        h.replay(ids, age, m)


def test_actual_runner_two_updates_gradients_checkpoint_and_target_free_play(monkeypatch):
    from test_range_jepa_amp_ppo import TinyEnv, cfg
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms.amp_ppo import AMPPPO
    c = cfg()
    c['obs_groups']['actor'] = ['policy', 'range_features']
    c['actor']['class_name'] = 'rsl_rl.models.range_bev:RangeBEVActor'
    c['actor']['representation'] = 'latent'
    c['algorithm']['perception_cfg'].update(variant='F2', bank_size=32, num_learning_epochs=1)
    runner = OnPolicyRunner(TinyEnv(), c, device='cpu')
    alg = runner.alg
    phase = {'ppo': False}
    parent = AMPPPO.update
    original_encode = alg.perception.encode
    def parent_update(self):
        phase['ppo'] = True
        try:
            return parent(self)
        finally:
            phase['ppo'] = False
    def encode(*args, **kwargs):
        assert not phase['ppo'], 'PPO must use cached features, not encode depth'
        return original_encode(*args, **kwargs)
    monkeypatch.setattr(AMPPPO, 'update', parent_update)
    monkeypatch.setattr(alg.perception, 'encode', encode)
    aux = alg._update_perception
    calls = []
    def checked_aux():
        state = {k: v.detach().clone() for k, v in alg.actor.state_dict().items()}
        result = aux()
        for k, value in alg.actor.state_dict().items():
            torch.testing.assert_close(value, state[k], rtol=0, atol=0)
        calls.append(result)
        return result
    monkeypatch.setattr(alg, '_update_perception', checked_aux)
    runner.learn(2)
    assert len(calls) == 2 and alg.perception_update_count == 2
    assert alg._live_policy.history.body_observations
    assert alg.storage.observations['range_features'].shape[-1] == 4096
    saved = deepcopy(alg.save())
    assert saved['perception_model_cfg']['variant'] == 'F2'
    clone = OnPolicyRunner(TinyEnv(), c, device='cpu')
    clone.alg.load(saved, None, True)
    assert clone.alg._live_policy.history is None
    for k, value in alg.perception.state_dict().items():
        torch.testing.assert_close(value, clone.alg.perception.state_dict()[k], rtol=0, atol=0)
    inference = runner.get_inference_policy('cpu')
    raw = runner.env.get_observations().exclude('range_map', 'range_sim')
    assert inference(raw).shape == (4, 2)
    assert _model_configuration(c['algorithm']['perception_cfg'])['variant'] == 'F2'
    parent_cfg = deepcopy(c)
    parent_cfg['algorithm']['perception_cfg']['variant'] = 'F'
    old = OnPolicyRunner(TinyEnv(), parent_cfg, device='cpu')
    with pytest.raises(ValueError, match='configuration differs'):
        old.alg.load(saved, None, True)
