"""Surface geometry, independent target learning, and PPO stop-gradient."""

import unittest

import torch
from tensordict import TensorDict

from rsl_rl.models.range_bev import RangeBEVActor, RangeBEVBackend, RangeBEVProjector


class RangeBEVTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_cell_centres_direction_and_periodic_seam(self):
        projector = RangeBEVProjector(1, 4, (-44.0, -46.0), input_normalized=False, grid_size=4, extents=(4.0,))
        image = torch.ones(1, 2, 1, 4)
        grid = projector(image)
        # Azimuth centres -135,-45,+45,+135; four distinct quadrants.
        valid = grid[0, 0, 1]
        self.assertEqual(valid.sum().item(), 4)
        for y, x in ((1, 1), (1, 2), (2, 1), (2, 2)):
            self.assertEqual(valid[y, x], 1)
            self.assertAlmostEqual(grid[0, 0, 0, y, x].item(), -(2**-0.5), places=5)
        wide = RangeBEVProjector(1, 96, (-1.0, 1.0), input_normalized=False)
        # First and last angular columns physically adjacent near -X.
        self.assertLess(torch.linalg.vector_norm(wide.rays[0, 0] - wide.rays[0, -1]).item(), 0.07)

    def test_nohit_mask_and_invalid_predictions_not_clipped(self):
        projector = RangeBEVProjector(1, 4, (-44.0, -46.0), grid_size=4, extents=(4.0,), height_band=(-2.0, -0.1))
        image = torch.zeros(1, 2, 1, 4)
        image[:, 0] = torch.tensor([float("nan"), -1.0, 2.0, 0.025])
        image[:, 1] = 1
        grid = projector(image)
        self.assertEqual(grid[:, :, 1].sum().item(), 1)
        self.assertTrue(torch.isfinite(grid).all())
        image[:, 1] = 0
        self.assertEqual(projector(image).count_nonzero().item(), 0)
        image[:, 1] = -1
        self.assertEqual(projector(image, mask_logits=True).count_nonzero().item(), 0)

    def test_height_band_extent_and_max_surface(self):
        projector = RangeBEVProjector(
            2,
            1,
            (-10.0, -80.0),
            max_range=40,
            input_normalized=False,
            grid_size=2,
            extents=(4.0,),
            height_band=(-2.5, 0.5),
        )
        image = torch.ones(1, 2, 2, 1)
        grid = projector(image)
        z = projector.rays[0, :, 2]
        self.assertEqual(grid[:, :, 1].sum().item(), 1)
        self.assertAlmostEqual(grid[0, 0, 0, 1, 1].item(), z.max().item(), places=5)
        narrow = RangeBEVProjector(2, 1, (-10.0, -80.0), input_normalized=False, grid_size=2, extents=(0.1,))
        self.assertEqual(narrow(image).count_nonzero().item(), 0)
        band = RangeBEVProjector(
            2, 1, (-10.0, -80.0), input_normalized=False, grid_size=2, extents=(4.0,), height_band=(-0.1, 0.1)
        )
        self.assertEqual(band(image).count_nonzero().item(), 0)

    def test_f_jepa_targets_detached_but_codec_trains_target_encoder(self):
        backend = RangeBEVBackend(image_height=8, image_width=16)
        image = torch.ones(2, 2, 8, 16)
        image[:, 0] = 0.03
        context = torch.randn(2, 24, 32, requires_grad=True)
        out = backend.predict(context)
        self.assertEqual(out["features"].shape, (2, 4096))
        terms = backend.auxiliary_terms(out, image, image, image)
        (terms["mem"] + terms["full"]).mean().backward(retain_graph=True)
        self.assertIsNotNone(context.grad)
        self.assertTrue(all(p.grad is None for p in backend.target_encoder.parameters()))
        backend.zero_grad()
        (terms["codec"] + terms["reg"]).mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in backend.target_encoder.parameters()))
        self.assertTrue(all(torch.isfinite(v).all() for v in terms.values()))

    def test_actor_ppo_detaches_perception_and_exports_equivalently(self):
        for representation in ("height", "latent"):
            feats = torch.randn(2, 4096, requires_grad=True)
            obs = TensorDict({"policy": torch.randn(2, 12), "range_features": feats}, batch_size=[2])
            actor = RangeBEVActor(
                obs,
                {"actor": ["policy", "range_features"]},
                "actor",
                3,
                hidden_dims=(16,),
                representation=representation,
            )
            output = actor(obs)
            output.square().mean().backward()
            self.assertIsNone(feats.grad)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.bev_cnn.parameters()))
            export = actor.as_onnx()
            torch.testing.assert_close(export(obs["policy"], feats), output)


if __name__ == "__main__":
    unittest.main()
