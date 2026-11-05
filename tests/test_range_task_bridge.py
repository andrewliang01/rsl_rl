"""Task alignment, cache ownership and unchanged legacy inference contracts."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO
from rsl_rl.algorithms.range_task_bridge_amp_ppo import (
    RangeTaskBridgeAMPPPO, frozen_actor, validate_task_bridge_config,
)
from rsl_rl.models.range_jepa import RangeJEPA
from rsl_rl.runners import OnPolicyRunner
from test_range_jepa_amp_ppo import TinyEnv, TinyExpert, cfg

torch.set_num_threads(1)


def configuration(coefficient=.05, amp=False):
    result = cfg(amp=amp)
    result['algorithm']['class_name'] = 'rsl_rl.algorithms.range_task_bridge_amp_ppo:RangeTaskBridgeAMPPPO'
    result['algorithm']['perception_cfg']['num_learning_epochs'] = 1
    result['algorithm']['task_bridge_cfg'] = {'version': 1, 'coefficient': coefficient}
    return result


def construct(coefficient=.05, no_fresh=False):
    env = TinyEnv(no_fresh=no_fresh)
    alg = RangeTaskBridgeAMPPPO.construct_algorithm(env.get_observations(), env, configuration(coefficient), 'cpu')
    return env, alg


def collect(env, alg):
    obs = env.get_observations()
    rows = []
    with torch.inference_mode():
        for t in range(alg.storage.num_transitions_per_env):
            obs['policy'] = torch.arange(env.num_envs)[:, None].expand(-1, 5).float() + t * 10
            action = alg.act(obs)
            rows.append((obs['policy'].clone(), action.clone(), alg.transition.actions_log_prob.clone()))
            obs, rewards, dones, extras = env.step(action)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    return rows


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('bad', [False, [], '0.1', {'coefficient': True}, {'coefficient': -1},
    {'coefficient': float('nan')}, {'coefficient': float('inf')}, {'version': 2}, {'unknown': 1}])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_task_bridge_config(bad)


def test_fresh_capture_control_step_episode_and_owned_scalar_labels():
    env, alg = construct()
    expected = collect(env, alg)
    snapshot = alg._snapshot_task_rollout()
    assert not snapshot.proprioception.is_inference()
    assert alg.perception_storage.num_anchors == 13  # 0/2/4 for all + env0 reset at3
    seen = []
    for batch in alg.perception_storage.sequence_plan(2, 2):
        for t, j in batch.active.nonzero().tolist():
            step = batch.control_steps[t, j].item()
            env_id = batch.env_ids[t, j].item()
            episode = batch.episode_ids[t, j].item()
            seen.append((step, env_id, episode))
            torch.testing.assert_close(snapshot.proprioception[step, env_id], expected[step][0][env_id], rtol=0, atol=0)
            torch.testing.assert_close(snapshot.actions[step, env_id], expected[step][1][env_id], rtol=0, atol=0)
            torch.testing.assert_close(snapshot.old_log_prob[step, env_id], expected[step][2][env_id].squeeze(), rtol=0, atol=0)
    assert len(set(seen)) == 13
    assert (3, 0, 1) in seen and not any(t == 1 for t, _, _ in seen)
    assert not any(e == 0 and t >= 3 and ep == 0 for t, e, ep in seen)
    before = deepcopy(snapshot)
    alg.storage.clear()
    alg.storage.observations['policy'].fill_(-123.)
    alg.storage.actions.fill_(-123.)
    alg.storage.actions_log_prob.fill_(-123.)
    with torch.inference_mode():
        alg.storage.weighted_advantages.fill_(-123.)
    for name in before.__dataclass_fields__:
        assert_tree_equal(getattr(before, name), getattr(snapshot, name))


def test_task_surrogate_is_exact_and_does_not_mutate_gaussian_or_actor():
    env, alg = construct()
    collect(env, alg)
    alg._task_rollout = alg._snapshot_task_rollout()
    # TinyEnv does not publish the real environment's reward-group extras;
    # its weighted advantages are zero. Set a known nonzero policy label to
    # test the task gradient itself, independently of that env fixture.
    alg._task_rollout.advantages.fill_(1.)
    batch = next(iter(alg.perception_storage.sequence_plan(2, 2)))
    kwargs = alg._perception_auxiliary_kwargs(batch)
    features = torch.randn(batch.active.shape[1], 512, requires_grad=True)
    actor = alg.actor
    values = deepcopy(actor.state_dict())
    actor.distribution.std_param.data[0] = -1.  # must not be clamped in-place by task branch
    values = deepcopy(actor.state_dict())
    distribution_cache = actor.distribution._distribution
    with frozen_actor(actor), patch.object(actor.distribution, 'update', side_effect=AssertionError('mutated cache')):
        actual = kwargs['feature_loss_fn'](features, 0)
        steps, ids = batch.control_steps[0], batch.env_ids[0]
        snap = alg._task_rollout
        mean = actor.task_bridge_mean(snap.proprioception[steps, ids], features)
        std = actor.distribution.std_param.clamp(*actor.distribution.std_range)
        log_prob = torch.distributions.Normal(mean, std).log_prob(snap.actions[steps, ids]).sum(-1)
        ratio = (log_prob - snap.old_log_prob[steps, ids]).exp()
        advantage = snap.advantages[steps, ids]
        reference = -torch.minimum(advantage * ratio, advantage * ratio.clamp(1-alg.clip_param, 1+alg.clip_param))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert_tree_equal(values, actor.state_dict())
    assert actor.distribution._distribution is distribution_cache


def test_task_only_gradient_reaches_memory_and_cnn_actor_is_frozen():
    torch.manual_seed(17)
    env, alg = construct()
    collect(env, alg)
    alg._task_rollout = alg._snapshot_task_rollout()
    alg._task_rollout.advantages.fill_(1.)  # known nonzero task-gradient probe
    batch = next(iter(alg.perception_storage.sequence_plan(2, 2)))
    before = deepcopy(alg.actor.state_dict())
    for p in alg.actor.parameters():
        p.grad = torch.full_like(p, .123)
    grads = [p.grad.clone() for p in alg.actor.parameters()]
    modes = [m.training for m in alg.actor.modules()]
    requires = [p.requires_grad for p in alg.actor.parameters()]
    calls = []
    encode = alg.perception.encode
    def counted(image):
        calls.append(image.shape[0])
        return encode(image)
    with frozen_actor(alg.actor), patch.object(alg.perception, 'encode', side_effect=counted):
        output = alg.perception.auxiliary_loss(batch.observed, batch.mapped, batch.simulated,
            initial_hidden=batch.initial_hidden, reset=batch.reset, active=batch.active,
            weights={name: 0. for name in alg.perception.default_weights}, **alg._perception_auxiliary_kwargs(batch))
        output['loss'].backward()
    assert len(calls) == batch.observed.shape[0] and all(n == 3*batch.observed.shape[1] for n in calls)
    nonzero = {name for name, p in alg.perception.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any('encoder' in name for name in nonzero), nonzero
    assert any('gru' in name for name in nonzero), nonzero
    assert any('readout' in name or 'cross' in name for name in nonzero), nonzero
    assert_tree_equal(before, alg.actor.state_dict())
    assert_tree_equal(grads, [p.grad for p in alg.actor.parameters()])
    assert modes == [m.training for m in alg.actor.modules()]
    assert requires == [p.requires_grad for p in alg.actor.parameters()]


def test_padding_cannot_contribute_task_loss_or_gradient():
    model = RangeJEPA(image_height=8, image_width=16)
    x = torch.rand(2, 2, 2, 8, 16)
    x[:, :, 1] = 1
    active = torch.tensor([[True, True], [True, False]])
    scale = torch.tensor(2., requires_grad=True)
    def loss(features, t):
        return scale * torch.tensor([1., 999. if t else 1.])
    output = model.auxiliary_loss(x, x, x, active=active, feature_loss_fn=loss,
        feature_loss_weight=.05, weights={name: 0. for name in model.default_weights})
    torch.testing.assert_close(output['task'], torch.tensor(2.))
    output['loss'].backward()
    torch.testing.assert_close(scale.grad, torch.tensor(.05))


def test_zero_coefficient_matches_legacy_losses_gradients_and_two_updates():
    saves = []
    losses = []
    for bridge in [False, True]:
        torch.manual_seed(71)
        config = configuration(0.)
        if not bridge:
            config['algorithm'].pop('task_bridge_cfg')
            config['algorithm']['class_name'] = 'rsl_rl.algorithms.range_jepa_amp_ppo:RangeJEPAAMPPPO'
        runner = OnPolicyRunner(TinyEnv(), config, device='cpu')
        collected = []
        update = runner.alg.update
        def observed_update():
            result = update()
            collected.append({k: v for k, v in result.items() if not k.endswith('_s')})
            return result
        with patch.object(runner.alg, 'update', side_effect=observed_update):
            runner.learn(2)
        saved = deepcopy(runner.alg.save())
        saved.pop('task_bridge_cfg', None)
        saves.append(saved)
        losses.append(collected)
        assert all('control_steps' not in b for b in runner.alg.perception_storage._batches)
        if bridge:
            assert runner.alg._task_rollout is None
    assert_tree_equal(saves[0], saves[1])
    assert_tree_equal(losses[0], losses[1])


def test_true_runner_snapshots_before_parent_clear_and_never_encodes_in_ppo(monkeypatch):
    import rsl_rl.algorithms.amp_ppo as amp_module
    monkeypatch.setattr(amp_module, 'AMPLoader', TinyExpert)
    torch.manual_seed(73)
    runner = OnPolicyRunner(TinyEnv(), configuration(.05, amp=True), device='cpu')
    alg = runner.alg
    parent = AMPPPO.update
    phase = {'ppo': False}
    original_encode = alg.perception.encode
    def encode(*args, **kwargs):
        assert not phase['ppo']
        return original_encode(*args, **kwargs)
    def ppo_update(self):
        assert self._task_rollout is not None and self.storage.step == 6
        phase['ppo'] = True
        try:
            result = parent(self)
            assert self.storage.step == 0
            return result
        finally:
            phase['ppo'] = False
    receipts = []
    original_aux = alg._update_perception
    def aux():
        actor = deepcopy(alg.actor.state_dict())
        grads = [None if p.grad is None else p.grad.clone() for p in alg.actor.parameters()]
        result = original_aux()
        assert_tree_equal(actor, alg.actor.state_dict())
        assert_tree_equal(grads, [p.grad for p in alg.actor.parameters()])
        assert result['range_jepa/optimizer_steps'] == 1
        assert 'range_jepa/task' in result
        receipts.append(deepcopy(alg.last_perception_receipt))
        return result
    with patch.object(AMPPPO, 'update', ppo_update), patch.object(alg.perception, 'encode', encode), patch.object(alg, '_update_perception', aux):
        runner.learn(2)
    assert len(receipts) == 2 and alg._task_rollout is None
    assert all(r['task_bridge']['extra_image_forwards'] == 0 for r in receipts)
    assert not alg.perception_storage.num_anchors


def test_checkpoint_inheritance_override_and_unchanged_inference_export():
    torch.manual_seed(13)
    env, alg = construct()
    collect(env, alg)
    saved = deepcopy(alg.save())
    legacy_cfg = cfg()
    legacy_cfg['algorithm']['perception_cfg']['num_learning_epochs'] = 1
    legacy = RangeJEPAAMPPPO.construct_algorithm(env.get_observations(), env, legacy_cfg, 'cpu')
    legacy.load(saved, None, True)
    clone_cfg = configuration()
    clone_cfg['algorithm'].pop('task_bridge_cfg')
    clone = RangeTaskBridgeAMPPPO.construct_algorithm(env.get_observations(), env, clone_cfg, 'cpu')
    clone.load(saved, None, True)
    assert clone.task_bridge_coefficient == .05
    assert_tree_equal(legacy.perception.state_dict(), clone.perception.state_dict())
    assert_tree_equal(legacy.actor.state_dict(), clone.actor.state_dict())
    x = torch.randn(3, clone.actor.obs_dim)
    features = torch.randn(3, 512)
    torch.testing.assert_close(legacy.actor.as_jit()(x, features), clone.actor.as_jit()(x, features), rtol=0, atol=0)
    export = torch.jit.script(clone.actor.as_jit())
    torch.testing.assert_close(export(x, features), clone.actor.as_jit()(x, features), rtol=0, atol=0)
    _, zero = construct(0.)
    zero.load(saved, None, True)
    assert zero.task_bridge_coefficient == 0.
    bad = deepcopy(saved)
    bad['perception_model_cfg']['image_width'] = 20
    with pytest.raises(ValueError, match='configuration differs'):
        clone.load(bad, None, True)


def test_no_fresh_rollout_does_not_update_perception():
    env, alg = construct(no_fresh=True)
    collect(env, alg)
    before = deepcopy(alg.perception.state_dict())
    result = alg.update()
    assert result['range_jepa/task'] == result['range_jepa/optimizer_steps'] == 0
    assert_tree_equal(before, alg.perception.state_dict())


def test_freeze_restores_modes_and_requires_grad_on_error():
    _, alg = construct()
    actor = alg.actor
    actor.prop_mlp.eval()
    original = [m.training for m in actor.modules()]
    with pytest.raises(RuntimeError, match='probe'):
        with frozen_actor(actor):
            assert not any(p.requires_grad for p in actor.parameters())
            raise RuntimeError('probe')
    assert original == [m.training for m in actor.modules()]
    assert all(p.requires_grad for p in actor.parameters())


def test_legacy_module_default_auxiliary_loss_and_gradients_are_bitwise_equal():
    baseline_dir = os.environ.get('TB1_BASELINE_DIR')
    if baseline_dir is None:
        pytest.skip('Exact pre-change source probe is opt-in via TB1_BASELINE_DIR; zero-mode runner regression remains unconditional.')
    baseline = Path(baseline_dir) / 'range_jepa.py'
    spec = importlib.util.spec_from_file_location('legacy_range_jepa_tb1', baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(27)
    legacy = module.RangeJEPA(image_height=8, image_width=16)
    current = RangeJEPA(image_height=8, image_width=16)
    current.load_state_dict(legacy.state_dict())
    x = torch.rand(3, 2, 2, 8, 16)
    x[:, :, 1] = (x[:, :, 1] > .3).float()
    a = legacy.auxiliary_loss(x, x, x)
    b = current.auxiliary_loss(x, x, x)
    assert_tree_equal(a, b)
    a['loss'].backward(); b['loss'].backward()
    for p, q in zip(legacy.parameters(), current.parameters()):
        assert_tree_equal(p.grad, q.grad)
