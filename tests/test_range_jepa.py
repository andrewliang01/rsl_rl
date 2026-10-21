"""CPU tests of the actual main-branch model; no simulator dependencies."""
import copy
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.models.range_jepa import RangeJEPA, RangeJEPAActor


def image(T=1, B=2, H=12, W=32):
    depth = torch.rand(T, B, 1, H, W) * .9 + .05
    mask = (torch.rand_like(depth) > .35).float()
    return torch.cat((depth * mask, mask), 2)


class RangeJEPATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(347)
        torch.set_num_threads(1)
        self.model = RangeJEPA(image_height=12, image_width=32)

    def loss(self, **kwargs):
        x, m, s = image(3), image(3), image(3)
        return self.model.auxiliary_loss(x, m, s, **kwargs)

    def test_step_shapes_and_independent_cache(self):
        x = image()[0]
        out = self.model.step(x, self.model.initial_hidden(2))
        self.assertEqual(out['e'].shape, (2, 24, 32))
        self.assertEqual(out['z'].shape, (2, 24, 32))
        self.assertEqual(out['features'].shape, (2, 512))
        expected = torch.cat((self.model.pool_sectors(out['e']), self.model.pool_sectors(out['z'])), -1).flatten(1)
        torch.testing.assert_close(out['features'], expected)
        old = out['features'].clone()
        self.model.step(image()[0], out['hidden'])
        torch.testing.assert_close(out['features'], old)

    def test_normalized_input_is_not_divided_twice(self):
        x = torch.ones(2, 2, 12, 32)
        x[:, 0] = .5
        torch.testing.assert_close(self.model.encode(x)['image'][:, 0], x[:, 0])
        metres = RangeJEPA(image_height=12, image_width=32, input_normalized=False)
        x[:, 0] *= 40
        torch.testing.assert_close(metres.encode(x)['image'][:, 0], torch.full_like(x[:, 0], .5))

    def test_invalid_values_and_all_invalid_hold_memory(self):
        x = torch.zeros(2, 2, 12, 32)
        x[:, 0, 0, :4] = torch.tensor([float('nan'), float('inf'), -1., 2.])
        x[:, 1, 0, :4] = 1
        h = torch.randn(2, 128)
        out = self.model.step(x, h)
        torch.testing.assert_close(out['hidden'], h)
        self.assertTrue(torch.isfinite(out['features']).all())
        reset = self.model.step(x, h, reset=torch.tensor([True, False]))
        torch.testing.assert_close(reset['hidden'][0], torch.zeros(128))
        torch.testing.assert_close(reset['hidden'][1], h[1])

    def test_history_preserves_azimuth(self):
        front = torch.zeros(1, 2, 12, 32)
        front[:, 0, 3:9, 2:6] = .3
        front[:, 1, 3:9, 2:6] = 1
        back = front.roll(16, dims=-1)
        a = self.model.step(front, self.model.initial_hidden(1))
        b = self.model.step(back, self.model.initial_hidden(1))
        self.assertGreater((a['hidden'] - b['hidden']).abs().max().item(), 1e-4)
        empty = torch.zeros_like(front)
        ra = self.model.step(empty, a['hidden'])
        rb = self.model.step(empty, b['hidden'])
        self.assertGreater((ra['e'] - rb['e']).abs().max().item(), 1e-4)

    def test_reset_matches_new_episode(self):
        x = image()[0]
        out = self.model.step(x, torch.randn(2, 128), reset=torch.ones(2, dtype=torch.bool))
        expected = self.model.step(x, self.model.initial_hidden(2))
        for key in out:
            torch.testing.assert_close(out[key], expected[key])

    def test_full_loss_all_modules_and_fresh_five_adam(self):
        x, m, s = image(3), image(3), image(3)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for _ in range(5):
            opt.zero_grad()
            result = self.model.auxiliary_loss(x, m, s)
            self.assertEqual(result['anchor_count'].item(), 6)
            result['loss'].backward()
            for module in (self.model.encoder, self.model.gru, self.model.readout, self.model.completion, self.model.decoder):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
            opt.step()
        self.assertTrue(all(state['step'] == 5 for state in opt.state.values()))

    def test_jepa_targets_stop_but_prediction_CNN_GRU_learn(self):
        x, m, s = (image(3).requires_grad_() for _ in range(3))
        weights = {k: 0. for k in self.model.default_weights}
        weights.update(mem=1., full=1.)
        result = self.model.auxiliary_loss(x, m, s, weights=weights)
        result['loss'].backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertEqual(m.grad.abs().sum().item(), 0)
        self.assertEqual(s.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.encoder[0].conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.gru.weight_hh.grad.abs().sum().item(), 0)

    def test_codec_regularizer_train_shared_map_and_sim_encoder(self):
        x, m, s = (image().requires_grad_() for _ in range(3))
        w = {k: 0. for k in self.model.default_weights}
        w.update(codec=1., reg=.1)
        self.model.auxiliary_loss(x, m, s, weights=w)['loss'].backward()
        self.assertGreater(m.grad.abs().sum().item(), 0)
        self.assertGreater(s.grad.abs().sum().item(), 0)
        collapsed = torch.ones(2, 24, 32)
        self.assertGreater(self.model._regularizer(collapsed, torch.ones(2, 24, 1)).min().item(), 0)

    def test_temporal_gradient_and_reset_boundary(self):
        x = image(2).requires_grad_()
        m, s = image(2), image(2)
        h0 = torch.randn(2, 128, requires_grad=True)
        result = self.model.auxiliary_loss(x, m, s, initial_hidden=h0)
        grad = torch.autograd.grad(result['hidden'].sum(), (x, h0), retain_graph=True)
        self.assertGreater(grad[0][0].abs().sum().item(), 0)
        self.assertGreater(grad[1].abs().sum().item(), 0)
        result2 = self.model.auxiliary_loss(x, m, s, initial_hidden=h0, reset=torch.tensor([[False, False], [True, True]]))
        grad2 = torch.autograd.grad(result2['hidden'].sum(), (x, h0))
        self.assertEqual(grad2[0][0].abs().sum().item(), 0)
        self.assertEqual(grad2[1].abs().sum().item(), 0)

    def test_active_anchor_mean_and_padding_hold(self):
        x, m, s = image(3), image(3), image(3)
        active = torch.tensor([[True, True], [True, False], [False, False]])
        result = self.model.auxiliary_loss(x, m, s, active=active)
        a = self.model.auxiliary_loss(x[:2, :1], m[:2, :1], s[:2, :1])
        b = self.model.auxiliary_loss(x[:1, 1:], m[:1, 1:], s[:1, 1:])
        for key in self.model.default_weights:
            torch.testing.assert_close(result[key], (a[key]*2 + b[key])/3, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(result['hidden'][0], a['hidden'][0])
        torch.testing.assert_close(result['hidden'][1], b['hidden'][0])

    def test_support_and_empty_anchor_loss(self):
        x, m, s = image(2), image(2), image(2)
        empty = self.model.auxiliary_loss(x, m, s, support=torch.zeros(2, 2, 12, 32))
        self.assertEqual(empty['mem'].item(), 0)
        inactive = self.model.auxiliary_loss(x, m, s, active=torch.zeros(2, 2))
        self.assertEqual(inactive['anchor_count'].item(), 0)
        self.assertEqual(inactive['loss'].item(), 0)
        inactive['loss'].backward()
        self.assertTrue(all(p.grad is None or p.grad.abs().sum() == 0 for p in self.model.parameters()))

    def test_nonmultiple_resolution_and_query_directions(self):
        model = RangeJEPA(image_height=13, image_width=35)
        out = model.step(image(H=13, W=35)[0], model.initial_hidden(2))
        self.assertEqual(out['e'].shape, (2, 36, 32))
        self.assertEqual(model.decode(out['z']).shape, (2, 2, 13, 35))
        rays = model.query_directions.reshape(4, 9, 3)
        self.assertGreater(rays[0, 0, 2].item(), 0)
        self.assertLess(rays[-1, 0, 2].item(), 0)
        torch.testing.assert_close(rays.norm(dim=-1), torch.ones(4, 9))

    def test_bad_shapes_fail(self):
        with self.assertRaises(ValueError):
            self.model.step(torch.zeros(2, 3, 12, 32), self.model.initial_hidden(2))
        with self.assertRaises(ValueError):
            self.loss(weights={'teacher': 1})
        with self.assertRaises(ValueError):
            self.model.auxiliary_loss(image(), image(), image(), active=torch.ones(7))


class ActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.obs = TensorDict({'policy': torch.randn(4, 76), 'range_features': torch.randn(4, 512, requires_grad=True)}, [4])
        self.cfg = dict(class_name='GaussianDistribution', init_std=1., std_type='scalar')
        self.actor = RangeJEPAActor(self.obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                                    hidden_dims=[64, 32], obs_normalization=True, distribution_cfg=self.cfg)

    def test_actual_normal_MLP_forward_detaches_before_fusion(self):
        out = self.actor(self.obs, stochastic_output=True)
        self.assertEqual(out.shape, (4, 29))
        self.actor(self.obs).sum().backward()
        self.assertIsNone(self.obs['range_features'].grad)
        for module in (self.actor.fusion, self.actor.range_adapter, self.actor.prop_mlp, self.actor.mlp):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertFalse(self.actor.is_recurrent)
        self.assertEqual(self.actor.obs_dim, 76)
        self.assertIn('class_name', self.cfg)
        self.assertEqual(self.actor.get_output_log_prob(out).shape, (4,))

    def test_PPO_actor_cannot_backprop_to_actual_perception(self):
        model = RangeJEPA(image_height=12, image_width=32)
        out = model.step(image(B=4)[0], model.initial_hidden(4))
        obs = self.obs.clone()
        obs['range_features'] = out['features']
        self.actor(obs).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(self.actor.fusion.weight.grad.abs().sum().item(), 0)

    def test_normalization_excludes_cache(self):
        other = copy.deepcopy(self.actor)
        b = self.obs.clone()
        b['range_features'] = torch.full((4, 512), 100000.)
        self.actor.update_normalization(self.obs)
        other.update_normalization(b)
        for left, right in zip(self.actor.obs_normalizer.buffers(), other.obs_normalizer.buffers()):
            torch.testing.assert_close(left, right)

    def test_split_export_matches_actor_and_scripts(self):
        expected = self.actor(self.obs)
        exporter = self.actor.as_jit()
        torch.testing.assert_close(exporter(self.obs['policy'], self.obs['range_features']), expected)
        scripted = torch.jit.script(exporter)
        torch.testing.assert_close(scripted(self.obs['policy'], self.obs['range_features']), expected)
        onnx = self.actor.as_onnx(False)
        self.assertEqual(onnx.input_names, ['proprio_obs', 'range_features'])
        torch.testing.assert_close(onnx(*onnx.get_dummy_inputs()), self.actor(TensorDict({'policy': torch.zeros(1,76), 'range_features': torch.zeros(1,512)}, [1])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
