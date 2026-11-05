"""Exp-H gradient, immutable-cache, source-isolation and actual AMP/PPO integration."""
from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.range_evidence_amp_ppo import RangeEvidenceAMPPPO
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAInferencePolicy, _model_configuration
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.range_evidence import EvidenceReader, RangeEvidenceActor, RangeEvidenceJEPA
from rsl_rl.storage import RolloutStorage

torch.set_num_threads(1)
CFG = dict(image_height=8, image_width=16, context_grid=(2, 4))


def image(n=2):
    x = torch.ones(n, 2, 8, 16)
    x[:, 0] = torch.rand(n, 8, 16) * .1 + .02
    return x


def setup(n=2, kl=.01):
    model = RangeEvidenceJEPA(**CFG)
    obs = TensorDict(dict(policy=torch.randn(n, 96), critic=torch.randn(n, 16), amp=torch.zeros(n, 4),
                          range_features=torch.zeros(n, model.feature_size)), [n])
    groups = {'actor': ['policy', 'range_features'], 'critic': ['critic'], 'amp': ['amp']}
    actor = RangeEvidenceActor(obs, groups, 'actor', 3, hidden_dims=[32], reader_cfg=CFG,
        distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution', init_std=1., std_type='scalar'))
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[32])
    storage = RolloutStorage('rl', n, 4, obs, [3], device='cpu')
    algo = RangeEvidenceAMPPPO(actor, critic, storage,
        perception_cfg=dict(variant='H', **CFG, num_learning_epochs=1, tbptt_steps=2, sequence_batch_size=2),
        num_learning_epochs=1, num_mini_batches=1, auxiliary_policy_kl_limit=kl)
    return algo, obs


def raw(obs, capture=0):
    n = len(obs)
    return TensorDict(dict(policy=obs['policy'].clone(), critic=obs['critic'].clone(), amp=obs['amp'].clone(),
        range_obs=image(n), range_map=image(n), range_sim=image(n),
        range_capture_id=torch.full((n, 1), capture, dtype=torch.long),
        range_episode_id=torch.zeros(n, 1, dtype=torch.long)), [n])


def collect(algo, obs):
    with torch.inference_mode():
        for t in range(4):
            algo.act(raw(obs, t // 2))
            algo.process_env_step(raw(obs, (t + 1) // 2), torch.ones(len(obs)), torch.zeros(len(obs), dtype=torch.bool), {})
        algo.compute_returns(raw(obs, 2))


def test_ppo_trains_small_reader_and_body_but_never_encoder_write_or_support():
    algo, obs = setup()
    model, actor = algo.perception, algo.actor
    assert model.reader is actor.reader
    frame = model.step(image(), model.initial_hidden(2))
    obs['range_features'] = frame['features'].detach().clone().requires_grad_()
    actor(obs).square().mean().backward()
    for module in (actor.reader.readout, actor.reader.completion_mlp, actor.prop_mlp, actor.reader.body_query):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for module in (model.encoder, model.gru, model.decoder, actor.reader.support_head)
               for p in module.parameters())
    assert obs['range_features'].grad is None


def test_auxiliary_body_stops_and_jepa_targets_stop():
    algo, obs = setup()
    m, a = algo.perception, algo.actor
    x, mapped, sim = (image().requires_grad_() for _ in range(3))
    body = a.encode_proprioception(a.proprioceptive_observations(obs))
    weights = {k: 0. for k in m.default_weights}
    weights.update(mem=1., full=1.)
    m.auxiliary_loss(x, mapped, sim, body_features=body, weights=weights)['loss'].backward()
    assert x.grad.abs().sum() > 0
    assert not mapped.grad.any() and not sim.grad.any()
    assert all(p.grad is None for p in a.prop_mlp.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.gru.parameters())
    algo._zero_perception_grad()
    m.auxiliary_loss(x.detach(), mapped.detach(), sim.detach(), body_features=body.detach())['loss'].backward()
    assert m.reader.support_head.weight.grad.abs().sum() > 0


def test_unique_optimizer_ownership_and_no_explicit_history_growth():
    algo, _ = setup()
    actor = {id(p) for g in algo.optimizer.param_groups for p in g['params']}
    aux = {id(p) for g in algo.perception_optimizer.param_groups for p in g['params']}
    assert not actor & aux
    assert {id(p) for p in algo.perception.parameters()} <= actor | aux
    assert not algo.perception.history_offsets
    assert _model_configuration(dict(variant='H', **CFG))['context_grid'] == (2, 4)


@pytest.mark.parametrize('profiled', [False, True])
def test_capture_cache_held_reset_and_targets_never_reach_actor(profiled):
    from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler
    algo, obs = setup()
    policy = algo._live_policy
    source = raw(obs)
    profile = RangeOnlineProfiler(profiled, 'cpu') if profiled else None
    first = policy.prepare(source, record_indexed=algo.perception_storage.add_indexed, profile=profile)
    snapshot = first['range_features'].clone()
    assert 'range_map' not in first and 'range_sim' not in first and 'range_obs' not in first
    assert algo.perception_storage.num_anchors == 2
    old = policy.hidden.clone()
    altered = source.clone()
    altered['range_map'].fill_(123)
    altered['range_sim'].fill_(-34)
    held = policy.prepare(altered, record_indexed=algo.perception_storage.add_indexed, profile=profile)
    torch.testing.assert_close(held['range_features'], snapshot, rtol=0, atol=0)
    torch.testing.assert_close(policy.hidden, old, rtol=0, atol=0)
    torch.testing.assert_close(algo.actor(first), algo.actor(held), rtol=0, atol=0)
    source['range_capture_id'].fill_(1)
    policy.prepare(source, record_indexed=algo.perception_storage.add_indexed, profile=profile)
    torch.testing.assert_close(first['range_features'], snapshot, rtol=0, atol=0)
    policy.reset(torch.tensor([True, False]))
    assert not policy.features[0].any() and not policy.hidden[0].any()
    assert policy.history is None


def test_invalid_frame_holds_state_and_reset_erases_it():
    m = RangeEvidenceJEPA(**CFG)
    h = torch.randn(2, 128)
    out = m.step(torch.zeros(2, 2, 8, 16), h)
    torch.testing.assert_close(out['hidden'], h)
    reset = m.step(torch.zeros(2, 2, 8, 16), h, torch.tensor([True, False]))
    assert not reset['hidden'][0].any()
    torch.testing.assert_close(reset['hidden'][1], h[1])
    result = m.reader(out['features'], torch.zeros(2, 64))
    assert torch.isfinite(result['z']).all()


def test_masked_padded_auxiliary_and_temporal_gradient():
    m = RangeEvidenceJEPA(**CFG)
    obs = torch.stack([image(), image()]).requires_grad_()
    target = torch.stack([image(), image()])
    body = torch.randn(2, 2, 64)
    result = m.auxiliary_loss(obs, target, target, body_features=body,
        active=torch.tensor([[True, True], [True, False]]), support=torch.zeros(2, 2, 8, 16))
    assert result['anchor_count'] == 3 and result['mem'] == 0
    assert torch.isfinite(result['loss'])
    grad = torch.autograd.grad(result['hidden'].sum(), obs)[0]
    assert grad[0].abs().sum() > 0
    empty = m.auxiliary_loss(obs, target, target, body_features=body, active=torch.zeros(2, 2, dtype=torch.bool))
    assert empty['loss'] == 0


def test_real_rollout_update_resume_and_strict_shared_checkpoint(tmp_path):
    torch.manual_seed(19)
    algo, obs = setup()
    collect(algo, obs)
    result = algo.update()
    assert all(torch.isfinite(torch.as_tensor(v)) for v in result.values())
    assert algo.last_perception_receipt['global_fresh_anchors'] == 4
    assert algo.perception_storage.num_anchors == 0
    assert not algo.last_perception_receipt['ppo_uses_cached_features_only']
    path = tmp_path / 'checkpoint.pt'
    torch.save(algo.save(), path)
    state = torch.load(path, weights_only=False)
    resumed, _ = setup()
    resumed.load(state, None, True)
    assert resumed.perception.reader is resumed.actor.reader
    for k, v in algo.actor.state_dict().items():
        torch.testing.assert_close(v, resumed.actor.state_dict()[k], rtol=0, atol=0)
    collect(resumed, obs)
    resumed.update()
    wrong = deepcopy(state)
    wrong['perception_state_dict']['reader.bank.weight'] = wrong['perception_state_dict']['reader.bank.weight'].clone() + 1
    with pytest.raises(ValueError, match='inconsistent'):
        resumed.load(wrong, None, True)


def test_auxiliary_reader_kl_guard_restores_reader_and_its_adam_state():
    torch.manual_seed(29)
    algo, obs = setup(kl=1e-16)
    collect(algo, obs)
    before = deepcopy(algo.actor.reader.state_dict())
    body_before = deepcopy(algo.actor.prop_mlp.state_dict())
    cnn_before = algo.perception.encoder[0].conv.weight.detach().clone()
    result = algo._update_perception()
    assert result['range_h/aux_reader_rejected'] == 1
    for key, value in before.items():
        torch.testing.assert_close(value, algo.actor.reader.state_dict()[key], rtol=0, atol=0)
    for key, value in body_before.items():
        torch.testing.assert_close(value, algo.actor.prop_mlp.state_dict()[key], rtol=0, atol=0)
    assert not torch.equal(cnn_before, algo.perception.encoder[0].conv.weight)
    assert all(p not in algo.optimizer.state for p in algo.actor.reader.parameters())


def test_chunk_checkpoint_preserves_outputs_and_all_gradients():
    torch.manual_seed(54)
    algo, obs = setup()
    a = algo.actor
    b = deepcopy(a)
    a.reader_checkpoint, a.reader_chunk_size = False, 4096
    b.reader_checkpoint, b.reader_chunk_size = True, 1
    obs['range_features'] = algo.perception.step(image(), algo.perception.initial_hidden(2))['features'].detach()
    out_a, out_b = a(obs), b(obs)
    torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-6)
    out_a.square().sum().backward()
    out_b.square().sum().backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        if pa.grad is None:
            assert pb.grad is None
        else:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=1e-6)


def test_stop_gradient_ablation_preserves_forward_and_actor_fusion_learning():
    algo, obs = setup()
    a = algo.actor
    obs['range_features'] = algo.perception.step(image(), algo.perception.initial_hidden(2))['features'].detach()
    expected = a(obs).detach()
    a.reader_ppo_gradient = False
    out = a(obs)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    out.square().sum().backward()
    assert all(p.grad is None for p in a.reader.parameters())
    for module in (a.prop_mlp, a.fusion, a.range_adapter):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
