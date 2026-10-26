"""Causality, masks, temporal identity, PPO isolation and checkpoint tests for G."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from rsl_rl.models.range_predictive import RangePredictiveJEPA
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms.amp_ppo import AMPPPO
from test_range_jepa_amp_ppo import TinyEnv, cfg


def images(t=3, b=2):
    x = torch.rand(t, b, 2, 8, 16)
    x[:, :, 0] = 0.03 + x[:, :, 0] * 0.2
    x[:, :, 1] = (x[:, :, 1] > 0.6).float()
    return x


def model():
    return RangePredictiveJEPA(image_height=8, image_width=16, memory_grid=(2, 4), bank_size=8, hidden_size=256)


class PredictiveTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(731)

    def test_prior_causal_current_corrects_and_reset_removes_previous_episode(self):
        net = model()
        x = images()[0]
        hidden = torch.randn(2, net.hidden_size)
        a = net.step(x, hidden)
        b = net.step(x.roll(5, -1), hidden)
        torch.testing.assert_close(a["prior"], b["prior"], rtol=0, atol=0)
        self.assertGreater((a["hidden"] - b["hidden"]).abs().max().item(), 1e-5)
        reset = net.step(x, hidden, reset=torch.ones(2, dtype=torch.bool))
        fresh = net.step(x, net.initial_hidden(2))
        torch.testing.assert_close(reset["features"], fresh["features"])
        self.assertEqual(a["features"].shape, (2, 512))

    def test_invalid_measurements_do_not_write_memory(self):
        net = model()
        x = torch.zeros(2, 2, 8, 16)
        x[:, 0] = float("nan")
        hidden = torch.randn(2, net.hidden_size)
        out = net.step(x, hidden)
        torch.testing.assert_close(out["hidden"], out["prior"].flatten(1))
        self.assertTrue(torch.isfinite(out["features"]).all())

    def test_future_loss_learns_history_not_current_or_target_pixels(self):
        net = model()
        observed = images(t=2).requires_grad_()
        target = images(t=2).requires_grad_()
        eligible = torch.tensor([[False, False], [True, True]])
        weights = {k: float(k == "future") for k in net.default_weights}
        out = net.auxiliary_loss(observed, target, target, weights=weights, temporal_valid=eligible)
        out["loss"].backward()
        self.assertGreater(observed.grad[0].abs().sum().item(), 0)
        self.assertEqual(observed.grad[1].abs().sum().item(), 0)
        self.assertTrue(target.grad is None or target.grad.abs().sum() == 0)
        self.assertGreater(net.transition_pointwise.weight.grad.abs().sum().item(), 0)
        self.assertAlmostEqual(out["future_pair_fraction"].item(), .5)

    def test_padding_reset_and_boundary_do_not_create_future_pairs(self):
        net = model()
        x = images()
        active = torch.tensor([[True, True], [True, True], [True, False]])
        reset = torch.tensor([[False, False], [True, False], [False, False]])
        out = net.auxiliary_loss(x, x, x, active=active, reset=reset, temporal_valid=torch.ones(3, 2, dtype=torch.bool))
        self.assertAlmostEqual(out["future_pair_fraction"].item(), 2 / 5)
        out["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_runner_caches_features_and_loads_complete_predictive_checkpoint(self):
        settings = cfg()
        settings["algorithm"]["perception_cfg"].update(
            variant="G", bank_size=8, hidden_size=256, memory_grid=(2, 4), num_learning_epochs=1)
        runner = OnPolicyRunner(TinyEnv(), settings, device="cpu")
        alg = runner.alg
        phase = {"ppo": False}
        parent = AMPPPO.update
        encode = alg.perception.encode
        def guarded_encode(*args, **kwargs):
            self.assertFalse(phase["ppo"])
            return encode(*args, **kwargs)
        def parent_update(this):
            phase["ppo"] = True
            try:
                return parent(this)
            finally:
                phase["ppo"] = False
        with patch.object(AMPPPO, "update", parent_update), patch.object(alg.perception, "encode", guarded_encode):
            runner.learn(2)
        self.assertEqual(alg.perception_update_count, 2)
        saved = deepcopy(alg.save())
        clone = OnPolicyRunner(TinyEnv(), settings, device="cpu")
        clone.alg.load(saved, None, True)
        for a, b in zip(alg.perception.parameters(), clone.alg.perception.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        infer = runner.get_inference_policy("cpu")
        raw = runner.env.get_observations().exclude("range_map", "range_sim")
        infer(raw)
        memory = infer.hidden.clone()
        infer(raw)
        torch.testing.assert_close(infer.hidden, memory, atol=0, rtol=0)

    def test_gapped_capture_ids_disable_prediction_supervision(self):
        settings = cfg()
        settings["algorithm"]["perception_cfg"].update(
            variant="G", bank_size=8, hidden_size=256, memory_grid=(2, 4), num_learning_epochs=1)
        alg = OnPolicyRunner(TinyEnv(), settings, device="cpu").alg
        x = images(t=1, b=1)[0]
        for capture in (0, 3):
            alg.perception_storage.add(x, x, x, alg.perception.initial_hidden(1),
                                      torch.tensor([0]), torch.tensor([0]), torch.tensor([capture]))
        result = alg._update_perception()
        self.assertEqual(result["range_jepa/future_pair_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
