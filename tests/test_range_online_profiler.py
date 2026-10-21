from copy import deepcopy
import time
import unittest
from unittest.mock import patch
import torch
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO, RangeJEPAInferencePolicy
from rsl_rl.storage.perception_storage import PerceptionStorage
from rsl_rl.utils.range_online_profiler import RangeOnlineProfiler
from test_range_jepa_amp_ppo import TinyEnv, cfg


torch.set_num_threads(1)


class RangeOnlineProfilerTests(unittest.TestCase):
    def test_disjoint_local_scopes_and_reset(self):
        p=RangeOnlineProfiler(True,'cpu')
        for name in p.PHASES:p.mark(name);time.sleep(.0001)
        p.mark();out=p.finish()
        self.assertAlmostEqual(sum(out[n+'_s'] for n in p.PHASES),out['scoped_stream_s'],places=12)
        self.assertTrue(all(out[n+'_calls']==1 for n in p.PHASES))
        self.assertTrue(all(p.finish()[n+'_calls']==0 for n in p.PHASES))

    def test_no_event_sync_per_phase_and_only_one_boundary_wait(self):
        waits=[]
        class Event:
            def __init__(self,enable_timing):self.t=None
            def record(self,stream):self.t=time.perf_counter()
            def synchronize(self):waits.append(self)
            def elapsed_time(self,other):return (other.t-self.t)*1000
        with patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'current_stream',lambda device:object()):
            p=RangeOnlineProfiler(True,'cuda:0')
            for name in p.PHASES:p.mark(name)
            p.mark();self.assertFalse(waits)
            out=p.finish();self.assertEqual(len(waits),1)
            self.assertEqual(out['cuda_event_timing'],1.)
            p.mark('publish');p.mark();p.finish(synchronize=False)
            self.assertEqual(len(waits),1)
        with patch.object(torch.cuda,'Event',side_effect=AssertionError('disabled profiler created event')):
            p=RangeOnlineProfiler(False,'cuda:0');p.mark('publish');p.mark();self.assertEqual(p.finish(),{})

    def test_prepare_new_held_reset_and_recorded_samples_exact(self):
        env=TinyEnv();alg=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,cfg(),'cpu')
        a=RangeJEPAInferencePolicy(torch.nn.Identity(),deepcopy(alg.perception))
        b=RangeJEPAInferencePolicy(torch.nn.Identity(),deepcopy(alg.perception))
        xstore,ystore=PerceptionStorage(),PerceptionStorage();p=RangeOnlineProfiler(True,'cpu')
        for tick in range(6):
            obs=env.get_observations();obs['range_capture_id'].fill_(tick//2)
            obs['range_episode_id'][0]=int(tick>=3)
            with torch.inference_mode():
                x=a.prepare(obs,record_indexed=xstore.add_indexed)
                y=b.prepare(obs,record_indexed=ystore.add_indexed,profile=p)
            torch.testing.assert_close(x['range_features'],y['range_features'],rtol=0,atol=0)
            torch.testing.assert_close(a.hidden,b.hidden,rtol=0,atol=0)
            for old,new in zip(xstore._batches,ystore._batches):
                for key in old:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        out=p.finish()
        self.assertEqual(out['validate_select_calls'],6)
        self.assertEqual(out['publish_calls'],6)
        self.assertEqual(out['online_cnn_calls'],4)
        self.assertEqual(out['memory_attention_calls'],4)

    def test_rollout_update_profile_flag_keeps_actions_losses_adam_exact(self):
        env=TinyEnv();base=cfg();base['algorithm']['perception_cfg']['num_learning_epochs']=1
        profiled=deepcopy(base);profiled['algorithm']['perception_cfg']['profile_online_breakdown']=True
        a=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,base,'cpu')
        b=RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,profiled,'cpu')
        b.load(deepcopy(a.save()),None,True);obs=env.get_observations()
        for tick in range(6):
            with torch.inference_mode():
                torch.manual_seed(400+tick);aa=a.act(obs.clone())
                torch.manual_seed(400+tick);bb=b.act(obs.clone())
                torch.testing.assert_close(aa,bb,rtol=0,atol=0)
                nxt,rewards,dones,extras=env.step(aa)
                a.process_env_step(nxt.clone(),rewards.clone(),dones.clone(),deepcopy(extras))
                b.process_env_step(nxt.clone(),rewards.clone(),dones.clone(),deepcopy(extras));obs=nxt
        with torch.inference_mode():a.compute_returns(obs);b.compute_returns(obs)
        torch.manual_seed(77);left=a.update()
        torch.manual_seed(77);right=b.update()
        for name in ('actor','critic','perception'):
            for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for x,y in zip(a.perception_optimizer.state.values(),b.perception_optimizer.state.values()):
            for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        for key in ('value','surrogate','entropy','range_jepa/loss','range_jepa/obs','range_jepa/codec','range_jepa/reg','range_jepa/mem','range_jepa/full','range_jepa/range'):
            if key in left:self.assertEqual(left[key],right[key])
        prefix='range_profile/rank0/online/'
        self.assertEqual(right[prefix+'parent_act_calls'],6)
        self.assertEqual(right[prefix+'validate_select_calls'],6)
        self.assertFalse(any(k.startswith(prefix) for k in left))
        self.assertAlmostEqual(sum(right[prefix+n+'_s'] for n in RangeOnlineProfiler.PHASES),right[prefix+'scoped_stream_s'],places=12)
        self.assertEqual(b._online_profile.records,[])


if __name__=='__main__':unittest.main(verbosity=2)
