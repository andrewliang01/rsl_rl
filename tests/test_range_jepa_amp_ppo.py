from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import io
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import torch
from tensordict import TensorDict
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO, RangeJEPAInferencePolicy, RAW_RANGE_KEYS
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage.perception_storage import PerceptionStorage


torch.set_num_threads(1)


class TinyEnv:
    def __init__(self, n=4, no_fresh=False):
        self.num_envs, self.num_actions, self.device = n, 2, 'cpu'
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.cfg = SimpleNamespace(seed=7)
        self.step_count = 0
        self.episode = torch.zeros(n, 1, dtype=torch.long)
        self.no_fresh = no_fresh
        self.image = torch.zeros(n, 2, 8, 16)
        self.image[:, 0] = torch.linspace(.01, .9, 128).reshape(1,8,16)
        self.image[:, 1, 2:7, :10] = 1

    @property
    def unwrapped(self): return self

    def get_observations(self):
        n = self.num_envs
        return TensorDict(dict(policy=torch.randn(n, 5), critic=torch.randn(n, 7), amp=torch.randn(n, 4),
            range_obs=self.image.clone(), range_map=self.image.clone(), range_sim=self.image.clone(),
            range_capture_id=torch.full((n,1), -1 if self.no_fresh else self.step_count // 2, dtype=torch.long),
            range_episode_id=self.episode.clone()), batch_size=[n])

    def step(self, action):
        self.step_count += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.step_count % 3 == 0:
            done[0] = True
            self.episode[0] += 1
        return self.get_observations(), torch.randn(self.num_envs, 2), done, {'amp_next_obs':torch.randn(self.num_envs,4)}


def cfg(amp=False):
    return dict(num_steps_per_env=6, save_interval=50, obs_groups={'actor':['policy'],'critic':['critic'],'amp':['amp']},
        actor=dict(class_name='rsl_rl.models.range_jepa:RangeJEPAActor', hidden_dims=[16],obs_normalization=True,
                   distribution_cfg=dict(class_name='GaussianDistribution', init_std=0.5)),
        critic=dict(class_name='MLPModel',hidden_dims=[16],obs_normalization=True),
        algorithm=dict(class_name='rsl_rl.algorithms.range_jepa_amp_ppo:RangeJEPAAMPPPO',num_critics=2,
            num_learning_epochs=5,num_mini_batches=4, learning_rate=.0003,schedule='fixed',rnd_cfg=None,
            reward_group_names=['task','regularization'], reward_group_weights=[1,.1],
            amp_cfg=dict(motion_files=[],discriminator_hidden_dims=[16,8],discriminator_update_decimation=4) if amp else None,
            perception_cfg=dict(image_height=8,image_width=16,bank_size=4,sequence_batch_size=2,tbptt_steps=2)))


class TinyExpert:
    def __init__(self,**kwargs): self.amp_obs_dim=4
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch): yield torch.randn(mini_batch_size,4),torch.randn(mini_batch_size,4)


def test_true_runner_two_updates_multicritic_amp_and_no_ppo_encoder(monkeypatch):
    import rsl_rl.algorithms.amp_ppo as amp_module
    monkeypatch.setattr(amp_module, 'AMPLoader', TinyExpert)
    runner = OnPolicyRunner(TinyEnv(),cfg(amp=True),device='cpu')
    alg = runner.alg
    assert not (set(alg.storage.observations.keys()) & RAW_RANGE_KEYS)
    assert alg.storage.observations['range_features'].shape == (6,4,512)
    ppo_params={id(p) for g in alg.optimizer.param_groups for p in g['params']}
    aux_params={id(p) for p in alg.perception.parameters()}
    assert not ppo_params & aux_params
    phase={'ppo':False}; counts={'ppo_actor':0,'online':0,'amp_steps':0}; receipts=[]
    original_update=AMPPPO.update
    def parent_update(self):
        phase['ppo']=True
        try:return original_update(self)
        finally:phase['ppo']=False
    monkeypatch.setattr(AMPPPO,'update',parent_update)
    encode=alg.perception.encode
    def guarded_encode(*a,**k):
        assert not phase['ppo'], 'PPO must never encode images'
        return encode(*a,**k)
    monkeypatch.setattr(alg.perception,'encode',guarded_encode)
    alg.actor.register_forward_hook(lambda *_:counts.__setitem__('ppo_actor',counts['ppo_actor']+int(phase['ppo'])))
    step=alg.perception.step
    def online_step(*a,**k):
        counts['online']+=1
        return step(*a,**k)
    monkeypatch.setattr(alg.perception,'step',online_step)
    amp_step=alg.amp_optimizer.step
    def amp_optimizer_step(*a,**k):
        counts['amp_steps']+=1
        return amp_step(*a,**k)
    monkeypatch.setattr(alg.amp_optimizer,'step',amp_optimizer_step)
    update=alg.update
    def observed_update():
        result=update()
        receipts.append(dict(alg.last_perception_receipt))
        return result
    monkeypatch.setattr(alg,'update',observed_update)
    runner.learn(2)
    assert counts['ppo_actor']==40
    assert counts['amp_steps']==10
    assert len(receipts)==2
    assert all(r['auxiliary_optimizer_steps']==5 for r in receipts)
    assert all(r['anchors_per_local_pass']==r['local_fresh_anchors']>0 for r in receipts)
    assert alg.perception_update_count==10
    assert not alg.perception_storage.num_anchors
    # All five fresh auxiliary passes ran, while only fresh live captures stepped online.
    assert counts['online'] < 12
    assert not alg._live_policy.features.is_inference()
    saved=deepcopy(alg.save())
    assert all(key in saved for key in ('critic_0_state_dict','critic_1_state_dict','amp_optimizer_state_dict','perception_optimizer_state_dict'))
    clone=OnPolicyRunner(TinyEnv(),cfg(amp=True),device='cpu')
    assert clone.alg.load(saved,None,True)
    for a,b in zip(alg.perception.parameters(),clone.alg.perception.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert clone.alg.perception_update_count==10
    assert clone.alg._live_policy.hidden is None
    infer=runner.get_inference_policy('cpu')
    raw=runner.env.get_observations().exclude('range_map','range_sim')
    assert infer(raw).shape==(4,2)
    hidden=infer.hidden.clone()
    infer(raw)
    torch.testing.assert_close(hidden,infer.hidden,rtol=0,atol=0)
    infer.reset(torch.tensor([True,False,False,False]))
    assert not infer.hidden[0].any()


def test_fresh_only_state_and_target_ownership():
    env=TinyEnv();alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    obs=env.get_observations()
    with torch.inference_mode():alg.act(obs)
    h=alg._live_policy.hidden.clone(); f=alg.transition.observations['range_features'].clone()
    assert alg.perception_storage.num_anchors==4
    # The following frame is handled by process_env_step, not consumed until next act.
    nxt,reward,done,extras=env.step(torch.zeros(4,2))
    with torch.inference_mode():alg.process_env_step(nxt,reward,done,extras)
    assert alg.perception_storage.num_anchors==4
    with torch.inference_mode():alg.act(nxt)
    torch.testing.assert_close(h,alg._live_policy.hidden,rtol=0,atol=0)
    assert alg.perception_storage.num_anchors==4
    # Source mutation and a reset cannot corrupt retained PPO caches/aux images.
    obs['range_obs'].zero_()
    assert alg.perception_storage._batches[0]['observed'][:,1].any()
    alg._live_policy.reset_memory(torch.ones(4,dtype=torch.bool))
    torch.testing.assert_close(alg.storage.observations['range_features'][0],f,rtol=0,atol=0)
    assert all(not v.is_inference() for v in alg.perception_storage._batches[0].values())


def test_aux_only_cannot_change_policy_or_critics():
    env=TinyEnv();alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    with torch.inference_mode():alg.act(env.get_observations())
    before=[p.detach().clone() for g in alg.optimizer.param_groups for p in g['params']]
    aux_before=[p.detach().clone() for p in alg.perception.parameters()]
    alg._update_perception()
    after=[p for g in alg.optimizer.param_groups for p in g['params']]
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    assert all(p.grad is None for p in after)
    assert any(not torch.equal(a,b) for a,b in zip(aux_before,alg.perception.parameters()))


def test_storage_allfresh_sequences_reset_padding_prehidden_and_bad_ids():
    store=PerceptionStorage()
    for t in range(5):
        ids=torch.tensor([0,1]);ep=torch.tensor([int(t>=2),0]);cap=torch.tensor([t if t<2 else t-2,t])
        x=torch.ones(2,2,8,16)*t;h=torch.ones(2,128)*t
        store.add(x,x,x,h,ids,ep,cap)
    batches=store.sequence_batches(2,2)
    assert sum(b.anchor_count for b in batches)==10
    seen=[]
    for b in batches:
        assert not b.reset.any()
        for j in range(b.active.shape[1]):
            active=b.active[:,j]
            assert b.episode_ids[active,j].unique().numel()==1
            assert b.env_ids[active,j].unique().numel()==1
        for key in ('observed','mapped','simulated'):
            assert not getattr(b,key).is_inference()
        for t,j in b.active.nonzero().tolist():seen.append((int(b.env_ids[t,j]),int(b.episode_ids[t,j]),int(b.capture_ids[t,j])))
    assert len(set(seen))==10
    with unittest.TestCase().assertRaises(ValueError):store.sequence_batches(0,2)
    with unittest.TestCase().assertRaises(ValueError):store.add(x,x,x,h,ids,ep,torch.tensor([-1,0]))


def test_backwards_capture_rejected_and_episode_change_resets():
    env=TinyEnv();alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    policy=alg.get_policy(); obs=env.get_observations();obs['range_capture_id'].fill_(3)
    policy(obs)
    obs['range_capture_id'].fill_(2)
    with unittest.TestCase().assertRaises(ValueError):policy(obs)
    obs['range_capture_id'].zero_();obs['range_episode_id']+=1
    policy(obs)
    torch.testing.assert_close(policy.hidden,alg.perception.step(obs['range_obs'],alg.perception.initial_hidden(4))['hidden'])


def test_no_captures_produces_no_aux_steps():
    env=TinyEnv(no_fresh=True);alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    with torch.inference_mode():alg.act(env.get_observations())
    assert not alg._live_policy.features.any()
    assert alg.perception_storage.num_anchors==0
    assert alg._update_perception()['range_jepa/optimizer_steps']==0


def test_strict_checkpoint_rejects_same_shape_semantic_drift():
    env=TinyEnv();alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    saved=deepcopy(alg.save())
    for name,value in [('max_range',20.),('input_normalized',False),('vertical_fov',(45.,-90.))]:
        changed=cfg();changed['algorithm']['perception_cfg'][name]=value
        clone=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,changed,'cpu')
        with unittest.TestCase().assertRaises(ValueError):clone.load(saved,None,True)
    missing=deepcopy(saved);missing.pop('perception_model_cfg')
    with unittest.TestCase().assertRaises(KeyError):alg.load(missing,None,True)


def test_environment_metadata_contract_and_normalized_range():
    env=TinyEnv()
    env.range_image_spec=dict(image_height=8,image_width=16,vertical_fov=[52.,-90.],max_range=40.,input_normalized=True)
    RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    for name,value in [('image_height',9),('image_width',17),('vertical_fov',[40.,-90.]),('max_range',20.),('input_normalized',False)]:
        saved=env.range_image_spec[name];env.range_image_spec[name]=value
        with unittest.TestCase().assertRaises(ValueError):RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
        env.range_image_spec[name]=saved
    del env.range_image_spec['input_normalized']
    with unittest.TestCase().assertRaises(ValueError):RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')


def test_cached_old_probability_and_perception_weight_changes():
    env=TinyEnv();config=cfg();config['actor']['obs_normalization']=False
    alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,config,'cpu')
    obs=env.get_observations()
    with torch.inference_mode():
        for _ in range(6):
            action=alg.act(obs)
            obs,reward,done,extras=env.step(action)
            alg.process_env_step(obs,reward,done,extras)
        alg.compute_returns(obs)
    cached=alg.storage.observations.flatten(0,1)
    actions=alg.storage.actions.flatten(0,1)
    with torch.no_grad():
        alg.actor(cached,stochastic_output=True)
        old=alg.actor.get_output_log_prob(actions)
        torch.testing.assert_close(old,alg.storage.actions_log_prob.flatten(0,1).squeeze(-1),atol=1e-6,rtol=1e-6)
        before=alg.actor(cached).clone()
        for p in alg.perception.parameters():p.add_(torch.randn_like(p)*10.)
        after=alg.actor(cached)
    torch.testing.assert_close(before,after,atol=0,rtol=0)


def test_nccl_style_broadcast_noncontiguous_buffers_preserve_storage(monkeypatch):
    env=TinyEnv();alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
    alg.is_multi_gpu=True
    original_called=[]
    monkeypatch.setattr(AMPPPO,'broadcast_parameters',lambda self: original_called.append(True))
    tensors=list(alg.perception.parameters())+list(alg.perception.buffers())
    original=[(id(t),t.data_ptr(),t.stride(),t.requires_grad,t.detach().clone()) for t in tensors]
    # Exercise the actual transposed query buffer which Gloo accepts and NCCL rejects.
    assert any(not t.is_contiguous() for t in tensors)
    calls=[]
    def contiguous_only_rank1_receive(tensor,src):
        assert src==0 and tensor.is_contiguous()
        sentinel=17+len(calls)
        calls.append(sentinel)
        tensor.fill_(sentinel)
    monkeypatch.setattr(torch.distributed,'broadcast',contiguous_only_rank1_receive)
    alg.broadcast_parameters()
    assert original_called==[True] and len(calls)==len(tensors)
    for index,(tensor,old) in enumerate(zip(tensors,original)):
        assert (id(tensor),tensor.data_ptr(),tensor.stride(),tensor.requires_grad)==old[:4]
        torch.testing.assert_close(tensor,torch.full_like(tensor,calls[index]),rtol=0,atol=0)
        assert not torch.equal(tensor,old[4])



def test_indexed_storage_snapshots_and_legacy_add_ownership():
    store = PerceptionStorage()
    image = torch.rand(4, 2, 8, 16)
    hidden = torch.rand(4, 128)
    ids = torch.tensor([1, 3]); episode = torch.tensor([0, 1, 2, 3]); capture = torch.tensor([3, 4, 5, 6])
    with torch.inference_mode():
        owned = store.add_indexed(image, image, image, hidden, ids, episode, capture)
    expected_image = image[ids].clone(); expected_hidden = hidden[ids].clone()
    assert owned is store._batches[0]
    assert all(not value.is_inference() for value in owned.values())
    assert owned['observed'].data_ptr() != image.data_ptr()
    assert owned['pre_hidden'].data_ptr() != hidden.data_ptr()
    image.zero_(); hidden.zero_(); ids.zero_(); episode.zero_(); capture.zero_()
    torch.testing.assert_close(owned['observed'], expected_image, rtol=0, atol=0)
    torch.testing.assert_close(owned['pre_hidden'], expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(owned['env_ids'], torch.tensor([1, 3]), rtol=0, atol=0)
    torch.testing.assert_close(owned['episode_ids'], torch.tensor([1, 3]), rtol=0, atol=0)
    torch.testing.assert_close(owned['capture_ids'], torch.tensor([4, 6]), rtol=0, atol=0)
    # Public add still owns caller-selected tensors through independent clones.
    selected = expected_image.clone(); h = expected_hidden.clone(); e = torch.tensor([0, 1]); c = torch.tensor([2, 3])
    store.add(selected, selected, selected, h, e, e, c)
    selected.zero_(); h.zero_(); e.zero_(); c.zero_()
    torch.testing.assert_close(store._batches[1]['observed'], expected_image, rtol=0, atol=0)
    torch.testing.assert_close(store._batches[1]['pre_hidden'], expected_hidden, rtol=0, atol=0)


def test_indexed_prepare_matches_selected_path_loss_gradients_and_held_frames():
    env = TinyEnv()
    alg = RangeJEPAAMPPPO.construct_algorithm(env.get_observations(), env, cfg(), 'cpu')
    original = RangeJEPAInferencePolicy(torch.nn.Identity(), deepcopy(alg.perception))
    indexed = RangeJEPAInferencePolicy(torch.nn.Identity(), deepcopy(alg.perception))
    a, b = PerceptionStorage(), PerceptionStorage()
    for tick in range(6):
        obs = env.get_observations()
        obs['range_capture_id'] = torch.tensor([[tick // 2], [tick // 3], [-1 if tick == 0 else tick], [tick // 2]])
        obs['range_episode_id'] = torch.tensor([[0], [0], [int(tick >= 3)], [0]])
        if tick >= 3:
            obs['range_capture_id'][2] = tick - 3
        x = original.prepare(obs, record=a.add)
        y = indexed.prepare(obs, record_indexed=b.add_indexed)
        torch.testing.assert_close(x['range_features'], y['range_features'], rtol=0, atol=0)
        torch.testing.assert_close(original.hidden, indexed.hidden, rtol=0, atol=0)
        assert a.num_anchors == b.num_anchors
        for old, new in zip(a._batches, b._batches):
            for key in old:
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        # Env is free to overwrite buffers after the actor has consumed them.
        for key in ('range_obs', 'range_map', 'range_sim'):
            obs[key].zero_()
    old_batches, new_batches = a.sequence_batches(8, 128), b.sequence_batches(8, 128)
    assert len(old_batches) == len(new_batches)
    for old, new in zip(old_batches, new_batches):
        old_loss = original.perception.auxiliary_loss(old.observed, old.mapped, old.simulated,
                        old.initial_hidden, old.reset, active=old.active)['loss']
        new_loss = indexed.perception.auxiliary_loss(new.observed, new.mapped, new.simulated,
                        new.initial_hidden, new.reset, active=new.active)['loss']
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        old_loss.backward(); new_loss.backward()
    for old, new in zip(original.perception.parameters(), indexed.perception.parameters()):
        if old.grad is None:
            assert new.grad is None
        else:
            torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)


def test_indexed_storage_invalid_targets_ids_and_empty_selection():
    store = PerceptionStorage(); image = torch.zeros(4, 2, 8, 16); hidden = torch.zeros(4, 128)
    ids = torch.tensor([1, 3]); ep = torch.zeros(4, dtype=torch.long); cap = torch.arange(4)
    assert store.add_indexed(image, image, image, hidden, ids[:0], ep, cap) is None
    assert store.num_anchors == 0
    for args in ((image, image[:1], image, hidden, ids, ep, cap),
                 (image, image, image, hidden[:1], ids, ep, cap),
                 (image, image, image, hidden, ids.float(), ep, cap),
                 (image, image, image, hidden, ids, ep.float(), cap)):
        with unittest.TestCase().assertRaises(ValueError): store.add_indexed(*args)
    cap[3] = -1
    with unittest.TestCase().assertRaises(ValueError): store.add_indexed(image, image, image, hidden, ids, ep, cap)
    assert store.num_anchors == 0


class _MonkeyPatch:
    def __init__(self): self.stack=ExitStack()
    def setattr(self,obj,name,value): self.stack.enter_context(patch.object(obj,name,value))


class RangeJEPAAlgorithmTests(unittest.TestCase):
    pass


def _make_unittest(func):
    def test(self):
        if func.__code__.co_argcount:
            m=_MonkeyPatch()
            with m.stack: func(m)
        else: func()
    return test


for _name,_func in list(globals().items()):
    if _name.startswith('test_') and callable(_func):setattr(RangeJEPAAlgorithmTests,_name,_make_unittest(_func))

if __name__=='__main__':unittest.main(verbosity=2)
