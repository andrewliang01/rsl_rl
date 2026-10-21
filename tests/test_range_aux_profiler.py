from copy import deepcopy
from contextlib import ExitStack
import time
import unittest
from unittest.mock import patch
import torch
from rsl_rl.models.range_jepa import RangeJEPA
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO
from rsl_rl.utils.range_aux_profiler import RangeAuxProfiler
from test_range_jepa_amp_ppo import TinyEnv, cfg

torch.set_num_threads(1)


class RangeAuxProfilerTests(unittest.TestCase):
    def test_cpu_top_additive_nested_forward_not_double_counted(self):
        p=RangeAuxProfiler(True,'cpu')
        with p.phase('pack_sequences'):time.sleep(.001)
        with p.phase('forward_loss'):
            for name in p.FORWARD_PHASES:
                with p.phase(name,'forward'):time.sleep(.001)
        with p.phase('backward'):time.sleep(.001)
        with p.phase('gradient_sync_layout'):time.sleep(.001)
        with p.phase('clip_adam'):time.sleep(.001)
        out=p.finish()
        self.assertAlmostEqual(sum(out[n+'_s'] for n in p.TOP_PHASES)+out['other_s'],out['aux_wall_s'],places=12)
        self.assertAlmostEqual(sum(out['forward/'+n+'_s'] for n in p.FORWARD_PHASES)+out['forward/other_s'],out['forward_loss_s'],places=12)
        self.assertTrue(all(out[n+'_s']>0 for n in p.TOP_PHASES))
        self.assertGreaterEqual(out['other_s'],0)
        self.assertEqual(out,p.finish())

    def test_cuda_event_mock_no_per_phase_sync_one_finish_sync(self):
        sync=[]
        class Event:
            def __init__(self,enable_timing):self.t=None
            def record(self,stream):self.t=time.perf_counter()
            def elapsed_time(self,other):return (other.t-self.t)*1000
        with patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'current_stream',lambda device:object()),patch.object(torch.cuda,'synchronize',lambda device:sync.append(str(device))):
            p=RangeAuxProfiler(True,'cuda:3')
            with p.phase('forward_loss'):
                with p.phase('shared_cnn','forward'):time.sleep(.001)
            with p.phase('backward'):time.sleep(.001)
            self.assertEqual(sync,[])
            m=p.finish()
            self.assertEqual(sync,['cuda:3'])
            self.assertEqual(m['cuda_event_timing'],1.)
            self.assertEqual(p.finish(),m)
            self.assertEqual(len(sync),1)
            self.assertGreaterEqual(m['cuda_stream_other_s'],0)
            self.assertAlmostEqual(sum(m[n+'_s'] for n in p.TOP_PHASES)+m['other_s'],m['aux_wall_s'],places=12)

    def test_disabled_and_overlapping_phase_admission(self):
        with patch.object(torch.cuda,'Event',side_effect=AssertionError('no CUDA event allowed')):
            p=RangeAuxProfiler(False,'cuda:0')
            with p.phase('forward_loss'):pass
            self.assertEqual(p.finish(),{})
        p=RangeAuxProfiler(True,'cpu')
        with self.assertRaises(ValueError):
            with p.phase('forward_loss'):
                with p.phase('backward'):pass
        with self.assertRaises(ValueError):
            with p.phase('decoder','forward'):pass

    def test_profile_on_off_model_values_and_all_gradients_exact(self):
        torch.manual_seed(45)
        a=RangeJEPA(image_height=8,image_width=16,bank_size=4)
        b=deepcopy(a)
        obs=torch.rand(2,3,2,8,16);obs[:,:,1]=(obs[:,:,1]>.3).float()
        mapped=torch.rand_like(obs);mapped[:,:,1]=(mapped[:,:,1]>.2).float()
        sim=torch.rand_like(obs);sim[:,:,1]=1
        h1=torch.randn(3,128,requires_grad=True);h2=h1.detach().clone().requires_grad_(True)
        reset=torch.tensor([[False,False,False],[True,False,False]])
        active=torch.tensor([[True,True,True],[True,True,False]])
        x=a.auxiliary_loss(obs,mapped,sim,initial_hidden=h1,reset=reset,active=active)
        x['loss'].backward()
        prof=RangeAuxProfiler(True,'cpu')
        with prof.phase('forward_loss'):
            y=b.auxiliary_loss(obs,mapped,sim,initial_hidden=h2,reset=reset,active=active,profile=prof)
        with prof.phase('backward'):y['loss'].backward()
        prof.count('active_image_rows',15)
        m=prof.finish()
        self.assertEqual(m['encoded_image_rows'],18)
        self.assertEqual(m['active_image_rows'],15)
        self.assertEqual(m['padding_image_rows'],3)
        self.assertEqual(m['padding_ratio'],1/6)
        for name in ('shared_cnn','memory_readout_completion','decoder'):self.assertEqual(m['forward/'+name+'_calls'],2)
        self.assertEqual(m['forward/loss_regularizer_mask_calls'],3)
        for key in x:torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
        torch.testing.assert_close(h1.grad,h2.grad,atol=0,rtol=0)
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        self.assertTrue(all(m['forward/'+n+'_s']>0 for n in prof.FORWARD_PHASES))

    def test_full_update_profile_on_off_exact_parameters_adam_and_losses(self):
        env=TinyEnv();base=cfg();profiled=deepcopy(base)
        profiled['algorithm']['perception_cfg']['profile_aux_breakdown']=True
        a=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,base,'cpu')
        b=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,profiled,'cpu')
        b.load(deepcopy(a.save()),None,True)
        obs=env.get_observations()
        for t in range(6):
            with torch.inference_mode():
                torch.manual_seed(100+t);aa=a.act(obs.clone())
                torch.manual_seed(100+t);bb=b.act(obs.clone())
                torch.testing.assert_close(aa,bb,atol=0,rtol=0)
                nxt,rewards,dones,extras=env.step(aa)
                a.process_env_step(nxt.clone(),rewards.clone(),dones.clone(),deepcopy(extras))
                b.process_env_step(nxt.clone(),rewards.clone(),dones.clone(),deepcopy(extras))
                obs=nxt
        with torch.inference_mode():a.compute_returns(obs);b.compute_returns(obs)
        packed=b.perception_storage.sequence_batches(b.perception_tbptt_steps,b.perception_sequence_batch_size)
        expected_encoded=sum(int(x.observed.shape[0])*int(x.observed.shape[1])*3 for x in packed)*5
        expected_active=b.perception_storage.num_anchors*3*5
        expected_cnn_calls=sum(int(x.observed.shape[0]) for x in packed)*5
        expected_loss_calls=sum(int(x.observed.shape[0])+1 for x in packed)*5
        torch.manual_seed(77);la=a.update()
        torch.manual_seed(77);lb=b.update()
        for module in ('actor','critic','perception'):
            for x,y in zip(getattr(a,module).parameters(),getattr(b,module).parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
        for left,right in zip(a.perception_optimizer.state.values(),b.perception_optimizer.state.values()):
            for name in left:torch.testing.assert_close(left[name],right[name],atol=0,rtol=0)
        for key in ('value','surrogate','entropy','range_jepa/loss','range_jepa/obs','range_jepa/codec','range_jepa/reg','range_jepa/mem','range_jepa/full','range_jepa/range'):
            if key in la:self.assertEqual(la[key],lb[key])
        self.assertEqual(a.perception_update_count,b.perception_update_count)
        self.assertEqual(b.last_perception_receipt['auxiliary_optimizer_steps'],5)
        prefix='range_profile/rank0/'
        self.assertFalse(any(k.startswith('range_profile/') for k in la))
        self.assertEqual(lb[prefix+'encoded_image_rows'],expected_encoded)
        self.assertEqual(lb[prefix+'active_image_rows'],expected_active)
        self.assertEqual(lb[prefix+'padding_image_rows'],expected_encoded-expected_active)
        self.assertEqual(lb[prefix+'padding_ratio'],(expected_encoded-expected_active)/expected_encoded)
        for name in ('shared_cnn','memory_readout_completion','decoder'):
            self.assertEqual(lb[prefix+'forward/'+name+'_calls'],expected_cnn_calls)
        self.assertEqual(lb[prefix+'forward/loss_regularizer_mask_calls'],expected_loss_calls)
        self.assertAlmostEqual(sum(lb[prefix+n+'_s'] for n in RangeAuxProfiler.TOP_PHASES)+lb[prefix+'other_s'],lb[prefix+'aux_wall_s'],places=12)
        self.assertEqual(lb[prefix+'aux_wall_s'],lb['range_jepa/aux_total_s'])

if __name__=='__main__':unittest.main(verbosity=2)
