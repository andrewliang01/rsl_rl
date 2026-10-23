"""E/F integration on actual 40x96 rays; target branches are training-only."""

import inspect
import unittest
from unittest.mock import patch

import torch

from rsl_rl.models.range_experiments import RangeExperiment


class RangeBEVIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        torch.set_num_threads(1)
        self.image = torch.ones(2, 2, 40, 96)
        self.image[:, 0] = 0.03

    def test_f_step_uses_only_observed_history_not_target_branch(self):
        model = RangeExperiment(variant="F")
        self.assertEqual(model.history_offsets[-1], 50)
        self.assertEqual(model.bev_backend.projector.rays.shape, (1, 3840, 3))
        prior = model.encode_history(self.image)
        history = prior["tokens"][:, None]
        valid = prior["valid"][:, None]
        with patch.object(
            model.bev_backend, "encode_target", side_effect=AssertionError("training target accessed in rollout")
        ):
            output = model.step(
                self.image,
                model.initial_hidden(2),
                history_tokens=history,
                history_valid=valid,
                history_age=torch.ones(2, 1),
            )
        self.assertEqual(output["e"].shape, (2, 128, 32))
        self.assertEqual(output["features"].shape, (2, 4096))
        self.assertEqual(model.decode(output["z"]).shape, (2, 2, 2, 32, 32))
        self.assertNotIn("mapped", inspect.signature(model.step).parameters)
        self.assertNotIn("simulated", inspect.signature(model.step).parameters)
        self.assertNotIn("pose", inspect.signature(model.step).parameters)
        expected = output["z"].reshape(2, 2, 8, 8, 32).permute(0, 1, 4, 2, 3).reshape(2, -1)
        torch.testing.assert_close(output["features"], expected)

    def test_f_joint_training_and_shared_target_codec_gradient(self):
        model = RangeExperiment(variant="F")
        angular_params = {id(p) for p in model.encoder.parameters()}
        target_params = {id(p) for p in model.bev_backend.target_encoder.parameters()}
        self.assertFalse(angular_params & target_params)
        result = model.auxiliary_loss(self.image, self.image, self.image)
        (result["mem"] + result["full"]).backward(retain_graph=True)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.bev_backend.target_encoder.parameters()))
        model.zero_grad()
        result["codec"].backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.bev_backend.target_encoder.parameters())
        )
        self.assertTrue(all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.encoder.parameters()))
        result = model.auxiliary_loss(
            self.image,
            self.image,
            self.image,
            history_images=self.image[:, None],
            history_valid=torch.ones(2, 1, dtype=torch.bool),
            history_age=torch.full((2, 1), 0.1),
        )
        model.zero_grad()
        result["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_f_reset_ignores_previous_episode_context(self):
        model = RangeExperiment(variant="F")
        hidden = model.initial_hidden(2)
        tokens = torch.randn(2, 3, 32, 32)
        kwargs = dict(history_valid=torch.ones(2, 3, 32, dtype=torch.bool), history_age=torch.ones(2, 3))
        first = model.step(self.image, hidden, reset=torch.ones(2, dtype=torch.bool), history_tokens=tokens, **kwargs)
        second = model.step(
            self.image, hidden, reset=torch.ones(2, dtype=torch.bool), history_tokens=tokens + 100, **kwargs
        )
        torch.testing.assert_close(first["features"], second["features"])

    def test_e_projects_decoded_normalized_metres_once(self):
        model = RangeExperiment(variant="E", input_normalized=False)
        predicted = self.image.clone()
        predicted[:, 1] = 10  # mask logits, not a target mask
        with patch.object(model, "decode", return_value=predicted):
            out = model.step(torch.cat((self.image[:, :1] * 40, self.image[:, 1:]), 1), model.initial_hidden(2))
        expected = model.bev_projector(predicted, mask_logits=True)
        torch.testing.assert_close(out["features"], expected.flatten(1))
        self.assertEqual(out["features"].shape, (2, 4096))
        self.assertGreater(expected[:, :, 1].sum().item(), 0)
        self.assertGreater(expected[:, :, 0].abs().max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
