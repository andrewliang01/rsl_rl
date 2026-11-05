"""Input, gradient and export contract for the matched blind A0 actor."""
import importlib.util
import unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models.range_jepa import RangeJEPAActor

from rsl_rl.models.range_blind import BlindRangeJEPAActor



class BlindActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.obs = TensorDict({"policy":torch.randn(4,96),
                               "range_features":torch.randn(4,512,requires_grad=True)},[4])
        cfg={"class_name":"GaussianDistribution","init_std":1.,"std_type":"scalar"}
        args=(self.obs,{"actor":["policy","range_features"]},"actor",29)
        self.actor=BlindRangeJEPAActor(*args,hidden_dims=[64,32],obs_normalization=True,distribution_cfg=cfg)
        self.normal=RangeJEPAActor(*args,hidden_dims=[64,32],obs_normalization=True,distribution_cfg=cfg)
        self.normal.load_state_dict(self.actor.state_dict(),strict=True)

    def test_visual_cannot_change_action_and_original_is_not_mutated(self):
        altered=self.obs.clone();altered["range_features"][:]=1000
        original=self.obs["range_features"].clone()
        torch.testing.assert_close(self.actor(self.obs),self.actor(altered),rtol=0,atol=0)
        self.assertGreater((self.normal(self.obs)-self.normal(altered)).abs().max().item(),1e-5)
        torch.testing.assert_close(self.obs["range_features"],original)

    def test_body_changes_action_and_actor_gradients_remain(self):
        changed=self.obs.clone();changed["policy"]+=1
        self.assertGreater((self.actor(self.obs)-self.actor(changed)).abs().max().item(),1e-5)
        self.actor(self.obs).square().mean().backward()
        self.assertIsNone(self.obs["range_features"].grad)
        for module in (self.actor.prop_mlp,self.actor.fusion,self.actor.range_adapter,self.actor.mlp):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))

    def test_export_is_also_blind_and_matches_actor(self):
        expected=self.actor(self.obs)
        export=self.actor.as_jit()
        torch.testing.assert_close(export(self.obs["policy"],self.obs["range_features"]),expected)
        scripted=torch.jit.script(export)
        torch.testing.assert_close(scripted(self.obs["policy"],self.obs["range_features"]*1000),expected)
        onnx=self.actor.as_onnx()
        self.assertEqual(onnx.input_names,["proprio_obs","range_features"])
        torch.testing.assert_close(onnx(self.obs["policy"],self.obs["range_features"]*1000),expected)

if __name__ == "__main__":
    torch.set_num_threads(1);unittest.main(verbosity=2)
