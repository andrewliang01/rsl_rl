"""CPU invariants for an opt-in directional writer; no simulation or optimizer steps."""
from pathlib import Path
import hashlib,json,os,unittest

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms.range_evidence_amp_ppo import RangeEvidenceAMPPPO
from rsl_rl.algorithms.range_jepa_amp_ppo import _canonical_perception_contract
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.range_evidence import RangeEvidenceActor,RangeEvidenceJEPA
from rsl_rl.storage import RolloutStorage

torch.set_num_threads(1)
CFG=dict(image_height=8,image_width=16,context_grid=(2,4))

def image(n=2):
    x=torch.ones(n,2,8,16);x[:,0]=torch.rand(n,8,16)*.2+.02
    return x

def algorithm(enabled=False):
    model=RangeEvidenceJEPA(**CFG)
    obs=TensorDict(dict(policy=torch.randn(2,96),critic=torch.randn(2,16),amp=torch.zeros(2,4),
                       range_features=torch.zeros(2,model.feature_size)),[2])
    groups={'actor':['policy','range_features'],'critic':['critic'],'amp':['amp']}
    actor=RangeEvidenceActor(obs,groups,'actor',3,hidden_dims=[32],reader_cfg=CFG,
        distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=1.,std_type='scalar'))
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=[32])
    storage=RolloutStorage('rl',2,4,obs,[3],device='cpu')
    algo=RangeEvidenceAMPPPO(actor,critic,storage,
        perception_cfg=dict(variant='H',**CFG,directional_memory_write=enabled,num_learning_epochs=1,tbptt_steps=2,sequence_batch_size=2),
        num_learning_epochs=1,num_mini_batches=1)
    return algo,obs

class DirectionalWriteTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv('DIRECTIONAL_WRITE_FIXTURE'), 'Optional pre-edit fixture is supplied by the implementation audit.')
    def test_default_and_zero_optin_equal_captured_legacy_outputs_and_RNG(self):
        root=Path(os.environ['DIRECTIONAL_WRITE_FIXTURE'])
        original=json.loads((root/'before.json').read_text())
        with np.load(root/'legacy_outputs.npz') as a:
            for enabled in [False,True]:
                torch.manual_seed(original['seed'])
                m=RangeEvidenceJEPA(directional_memory_write=enabled).eval()
                self.assertTrue(np.array_equal(torch.get_rng_state().numpy(),a['constructor_rng']))
                common={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for k,v in m.state_dict().items() if k!='directional_write_weight'}
                self.assertEqual(common,original['state_sha256'])
                with torch.inference_mode():
                    out=m.step(torch.from_numpy(a['image']),torch.from_numpy(a['hidden']),torch.from_numpy(a['reset']))
                    read=m.reader(out['features'],torch.from_numpy(a['body']))
                for key,actual in [('expected_hidden',out['hidden']),('expected_cache',out['features']),
                                   ('expected_e',read['e']),('expected_z',read['z'])]:
                    self.assertTrue(np.array_equal(actual.numpy(),a[key]),(enabled,key))
                self.assertEqual(m.feature_size,1184)
                self.assertEqual(sum(p.numel() for p in m.parameters()),original['parameter_count']+(96 if enabled else 0))

    def test_zero_optin_matches_default_without_external_fixture(self):
        torch.manual_seed(440);old=RangeEvidenceJEPA(**CFG).eval();rng=torch.get_rng_state().clone()
        torch.manual_seed(440);new=RangeEvidenceJEPA(**CFG,directional_memory_write=True).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        old_state=old.state_dict();new_state=new.state_dict()
        self.assertEqual(set(new_state)-set(old_state),{'directional_write_weight'})
        for key,value in old_state.items():self.assertTrue(torch.equal(value,new_state[key]),key)
        x=image();h=old.initial_hidden(2)
        a,b=old.step(x,h),new.step(x,h)
        self.assertTrue(torch.equal(a['hidden'],b['hidden']))
        self.assertTrue(torch.equal(a['features'],b['features']))

    def test_sector_content_swap_can_be_distinguished_by_nonzero_direction_binding(self):
        m=RangeEvidenceJEPA(**CFG,directional_memory_write=True)
        f=torch.zeros(1,8,32);w=torch.zeros(1,8,1)
        f[:,1]=1;f[:,5]=2;w[:,[1,5]]=1
        other=f.clone();other[:,1]=f[:,5];other[:,5]=f[:,1]
        self.assertTrue(torch.equal(m._pool_memory_input(f,w),m._pool_memory_input(other,w)))
        with torch.no_grad():m.directional_write_weight[:,2]=.25
        self.assertGreater(float((m._pool_memory_input(f,w)-m._pool_memory_input(other,w)).detach().abs().max()),1e-3)

    def test_current_context_and_target_encoder_are_unchanged_by_direction_weights(self):
        m=RangeEvidenceJEPA(**CFG,directional_memory_write=True);x=image();h=m.initial_hidden(2)
        before=m.step(x,h);target=m.encode_target(x)['tokens'].clone()
        with torch.no_grad():m.directional_write_weight[:,2]=.25
        after=m.step(x,h)
        self.assertTrue(torch.equal(before['features'][:,128:],after['features'][:,128:]))
        self.assertTrue(torch.equal(target,m.encode_target(x)['tokens']))
        self.assertGreater(float((before['hidden']-after['hidden']).detach().abs().max()),1e-6)

    def test_empty_capture_holds_memory_and_episode_reset_clears_it(self):
        m=RangeEvidenceJEPA(**CFG,directional_memory_write=True)
        with torch.no_grad():m.directional_write_weight.fill_(.3)
        h=torch.randn(2,128);empty=torch.zeros(2,2,8,16)
        self.assertTrue(torch.equal(m.step(empty,h)['hidden'],h))
        after=m.step(empty,h,torch.tensor([True,False]))
        self.assertEqual(int(torch.count_nonzero(after['hidden'][0])),0)
        self.assertTrue(torch.equal(after['hidden'][1],h[1]))

    def test_auxiliary_gradient_reaches_writer_and_history_not_stop_gradient_targets(self):
        m=RangeEvidenceJEPA(**CFG,directional_memory_write=True)
        observed=torch.stack([image(),image()]).requires_grad_()
        mapped=torch.stack([image(),image()]).requires_grad_()
        simulated=torch.stack([image(),image()]).requires_grad_()
        body=torch.randn(2,2,64,requires_grad=True)
        weights={k:0. for k in m.default_weights};weights.update(mem=1.,full=1.)
        out=m.auxiliary_loss(observed,mapped,simulated,body_features=body,weights=weights)
        history_grad=torch.autograd.grad(out['hidden'].sum(),observed,retain_graph=True)[0]
        self.assertGreater(float(history_grad[0].abs().sum()),0.)
        out['loss'].backward()
        self.assertTrue(torch.isfinite(m.directional_write_weight.grad).all())
        self.assertGreater(float(m.directional_write_weight.grad.abs().sum()),0.)
        self.assertIsNone(body.grad)
        for target in [mapped,simulated]:
            self.assertTrue(target.grad is None or not bool(target.grad.any()))

    def test_writer_has_only_auxiliary_optimizer_ownership_and_no_PPO_gradient(self):
        algo,obs=algorithm(True);m=algo.perception
        actor_ids={id(p) for g in algo.optimizer.param_groups for p in g['params']}
        aux_ids={id(p) for g in algo.perception_optimizer.param_groups for p in g['params']}
        self.assertFalse(actor_ids&aux_ids)
        self.assertIn(id(m.directional_write_weight),aux_ids)
        self.assertNotIn(id(m.directional_write_weight),actor_ids)
        obs['range_features']=m.step(image(),m.initial_hidden(2))['features'].detach()
        algo.actor(obs).square().mean().backward()
        self.assertIsNone(m.directional_write_weight.grad)
        self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in algo.actor.reader.parameters()))

    def test_legacy_contract_loads_and_cross_writer_contract_is_rejected(self):
        old,_=algorithm(False);legacy=old.save();legacy['perception_model_cfg'].pop('directional_memory_write')
        resumed,_=algorithm(False);resumed.load(legacy,None,True)
        enabled,_=algorithm(True)
        with self.assertRaises(ValueError):enabled.load(legacy,None,True)
        state=enabled.save();same,_=algorithm(True);same.load(state,None,True)
        self.assertTrue(state['perception_model_cfg']['directional_memory_write'])
        self.assertTrue(torch.equal(enabled.perception.directional_write_weight,same.perception.directional_write_weight))

    def test_nonstrict_cross_writer_rejects_without_mutating_the_target(self):
        for source_mode,target_mode in [(False,True),(True,False)]:
            source,_=algorithm(source_mode);saved=source.save()
            for strict in [False,True]:
                target,_=algorithm(target_mode)
                before={k:v.clone() for k,v in target.perception.state_dict().items()}
                with self.assertRaisesRegex(ValueError,'directional_memory_write'):
                    target.load(saved,None,strict)
                for key,value in before.items():self.assertTrue(torch.equal(value,target.perception.state_dict()[key]))

    def test_explicit_actor_only_transfer_preserves_new_writer(self):
        source,_=algorithm(False);target,_=algorithm(True)
        before=target.perception.directional_write_weight.detach().clone()
        target.load(source.save(),{'actor':True,'perception':False,'critic':False,'optimizer':False},True)
        self.assertTrue(torch.equal(before,target.perception.directional_write_weight))

    def test_boolean_and_backend_contracts_are_explicit(self):
        for value in [None,0,1,'true']:
            with self.assertRaises(TypeError):RangeEvidenceJEPA(**CFG,directional_memory_write=value)
            with self.assertRaises(TypeError):_canonical_perception_contract(dict(variant='H',directional_memory_write=value))
        with self.assertRaises(ValueError):_canonical_perception_contract(dict(variant='A0',directional_memory_write=True))
        self.assertFalse(_canonical_perception_contract(dict(variant='H'))['directional_memory_write'])

    def test_input_encodings_do_not_change_writer_interface(self):
        for encoding in ['depth_valid','depth_center','depth_elevation']:
            m=RangeEvidenceJEPA(**CFG,directional_memory_write=True,image_encoding=encoding)
            x=image();x[:,1]=dict(depth_valid=1.,depth_center=.5,depth_elevation=.625)[encoding]
            out=m.step(x,m.initial_hidden(2))
            self.assertEqual(out['hidden'].shape,(2,128))
            self.assertEqual(out['features'].shape,(2,m.feature_size))
            self.assertTrue(torch.isfinite(out['features']).all())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DirectionalWriteTests))
    if 'DIRECTIONAL_WRITE_FIXTURE' in os.environ:
        root=Path(os.environ['DIRECTIONAL_WRITE_FIXTURE'])
        (root/'candidate_CPU.json').write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),
          failures=len(result.failures),errors=len(result.errors),GPU=False,simulation=False,optimizer_steps=0,
          no_control_efficacy_or_speed_claim=True),indent=2))
    raise SystemExit(not result.wasSuccessful())
