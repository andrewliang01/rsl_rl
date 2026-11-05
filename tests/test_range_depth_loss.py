"""Physical depth precision and unchanged inference/checkpoint contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import unittest

import torch
from torch.nn import functional as F
from rsl_rl.models.range_jepa import RangeJEPA
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO, _model_configuration
from test_range_jepa_amp_ppo import TinyEnv, cfg


def images():
    depth = torch.rand(3, 2, 1, 8, 16) * .8 + .05
    mask = (torch.rand_like(depth) > .3).float()
    return torch.cat((depth * mask, mask), 2)


class DepthPrecisionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(178)
        self.model = RangeJEPA(image_height=8, image_width=16)

    def test_default_matches_legacy_loss_and_gradients(self):
        prediction = torch.randn(2, 2, 8, 16, requires_grad=True)
        target = images()[0]
        actual = self.model._reconstruction(prediction, target)
        valid = target[:, 1:2]
        legacy = ((F.smooth_l1_loss(prediction[:, :1], target[:, :1], reduction='none') * valid).sum((1, 2, 3))
                  / valid.sum((1, 2, 3)).clamp_min(1e-6)
                  + F.binary_cross_entropy_with_logits(prediction[:, 1:2], valid, reduction='none').mean((1, 2, 3)))
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        a = torch.autograd.grad(actual.sum(), prediction, retain_graph=True)[0]
        b = torch.autograd.grad(legacy.sum(), prediction)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_physical_beta_controls_the_analytic_depth_gradient(self):
        for maximum in [20., 40.]:
            model = RangeJEPA(image_height=8, image_width=16, max_range=maximum)
            model.set_depth_huber_beta_m(.1)
            self.assertAlmostEqual(model.depth_huber_beta_normalized, .1 / maximum)
            target = torch.tensor([[[[.3]], [[1.]]]], dtype=torch.float64)
            prediction = torch.tensor([[[[.31]], [[0.]]]], dtype=torch.float64, requires_grad=True)
            loss = model._reconstruction(prediction, target, predict_mask=False).sum()
            loss.backward()
            # A 0.2/0.4m error exceeds the requested 0.1m transition.
            self.assertAlmostEqual(prediction.grad[0, 0, 0, 0].item(), 1., places=12)
            self.assertEqual(prediction.grad[0, 1, 0, 0].item(), 0.)
            model.set_depth_huber_beta_m(None)
            prediction.grad = None
            model._reconstruction(prediction, target, predict_mask=False).sum().backward()
            self.assertAlmostEqual(prediction.grad[0, 0, 0, 0].item(), .01, places=12)

    def test_invalid_target_pixels_do_not_gain_depth_gradients(self):
        self.model.set_depth_huber_beta_m(.1)
        prediction = torch.randn(2, 2, 8, 16, requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = self.model._reconstruction(prediction, target, predict_mask=False)
        self.assertEqual(loss.sum().item(), 0.)
        loss.sum().backward()
        self.assertEqual(prediction.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(loss).all())

    def test_invalid_beta_fails_without_changing_effective_criterion(self):
        self.model.set_depth_huber_beta_m(.1)
        for value in [0, -1, float('nan'), float('inf'), True, '0.1', [], {}]:
            with self.subTest(value=str(value)):
                with self.assertRaisesRegex(ValueError, 'finite, positive'):
                    self.model.set_depth_huber_beta_m(value)
                self.assertEqual(self.model.depth_huber_beta_m, .1)

    def test_loss_option_does_not_change_inference_or_state_tensor_layout(self):
        modified = deepcopy(self.model)
        modified.set_depth_huber_beta_m(.1)
        x = images()[0]; hidden = torch.randn(2, 128)
        a = self.model.step(x, hidden); b = modified.step(x, hidden)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(self.model.decode(a['z']), modified.decode(b['z']), rtol=0, atol=0)
        self.assertEqual(set(self.model.state_dict()), set(modified.state_dict()))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), sum(p.numel() for p in modified.parameters()))

    def test_only_reconstruction_terms_change_in_the_complete_objective(self):
        modified = deepcopy(self.model); modified.set_depth_huber_beta_m(.1)
        x, mapped, simulated = images(), images(), images()
        a = self.model.auxiliary_loss(x, mapped, simulated)
        b = modified.auxiliary_loss(x, mapped, simulated)
        for key in ['mem', 'full', 'reg', 'hidden', 'anchor_count']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key in ['obs', 'codec', 'range']:
            self.assertGreater(abs((a[key] - b[key]).item()), 1e-6)
        b['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in modified.parameters()))


class DepthCriterionCheckpointTest(unittest.TestCase):
    def construct(self, beta=None, variant=None):
        configuration = cfg()
        configuration['algorithm']['perception_cfg']['depth_huber_beta_m'] = beta
        if variant is not None:
            configuration['algorithm']['perception_cfg']['variant'] = variant
        env = TinyEnv()
        with redirect_stdout(io.StringIO()):
            return RangeJEPAAMPPPO.construct_algorithm(env.get_observations(), env, configuration, 'cpu')

    def test_option_is_training_only_and_model_contract_is_unchanged(self):
        a = _model_configuration({'image_height': 8, 'image_width': 16})
        b = _model_configuration({'image_height': 8, 'image_width': 16, 'depth_huber_beta_m': .1})
        self.assertEqual(a, b)
        alg = self.construct(.1)
        self.assertNotIn('depth_huber_beta_m', alg.perception_model_cfg)
        self.assertEqual(alg.perception_cfg['depth_huber_beta_m'], .1)
        self.assertEqual(alg.last_perception_receipt['depth_huber_beta_normalized'], .0025)

    def test_new_checkpoint_preserves_criterion_on_default_resume_or_play(self):
        original = self.construct(.1); saved = deepcopy(original.save())
        self.assertEqual(saved['perception_loss_cfg'], {'version': 1, 'depth_huber_beta_m': .1})
        clone = self.construct()
        self.assertTrue(clone.load(saved, None, True))
        self.assertEqual(clone.perception.depth_huber_beta_m, .1)
        self.assertEqual(clone.perception.depth_huber_beta_normalized, .0025)
        self.assertEqual(clone.save()['perception_cfg']['depth_huber_beta_m'], .1)
        self.assertEqual(clone.last_perception_receipt['depth_loss_config_source'], 'checkpoint_inherited')
        for a, b in zip(original.perception.parameters(), clone.perception.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_old_checkpoint_can_load_legacy_or_explicit_precision_override(self):
        saved = deepcopy(self.construct().save())
        saved.pop('perception_loss_cfg')
        saved['perception_cfg'].pop('depth_huber_beta_m')
        for value, expected in [(None, 1.), (.1, .0025)]:
            clone = self.construct(value)
            self.assertTrue(clone.load(saved, None, True))
            self.assertEqual(clone.perception.depth_huber_beta_normalized, expected)

    def test_explicit_legacy_scale_can_override_new_checkpoint_intentionally(self):
        saved = deepcopy(self.construct(.1).save())
        clone = self.construct(40.)
        self.assertTrue(clone.load(saved, None, True))
        self.assertEqual(clone.perception.depth_huber_beta_normalized, 1.)
        self.assertEqual(clone.perception_cfg['depth_huber_beta_m'], 40.)

    def test_fov_and_normalization_mismatch_still_fail_strict_load(self):
        saved = deepcopy(self.construct(.1).save())
        for key, value in [('vertical_fov', [51., -90.]), ('max_range', 20.), ('input_normalized', False)]:
            changed = deepcopy(saved); changed['perception_model_cfg'][key] = value
            clone = self.construct()
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                clone.load(changed, None, True)
            self.assertIsNone(clone.perception.depth_huber_beta_m)

    def test_unsupported_variants_and_saved_loss_versions_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'only the A0'):
            self.construct(.1, 'A1')
        saved = deepcopy(self.construct(.1).save())
        saved['perception_loss_cfg']['version'] = 2
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.construct().load(saved, None, True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
