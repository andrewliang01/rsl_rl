"""CAMP neural gradients, reference isolation and supervised sequence alignment."""

import unittest
import torch
from tensordict import TensorDict

from rsl_rl.models.camp import CampActor, CampTerrainMemory
from rsl_rl.algorithms.range_jepa_amp_ppo import _model_configuration, compact_range_observations, RangeJEPAInferencePolicy
from rsl_rl.storage.perception_storage import PerceptionStorage


class CampModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = dict(image_height=8, image_width=16, channels=32, hidden_size=16, bank_size=8,
                        context_grid=(2, 4), vertical_fov=(52., -7.), auxiliary_objective='direct_geometry')

    def test_fast_control_read_preserves_actions_and_trainable_gradients(self):
        reader_cfg = {k: v for k, v in self.cfg.items() if k != 'auxiliary_objective'}
        obs = TensorDict({'policy': torch.randn(3, 12), 'range_features': torch.randn(3, 280)}, batch_size=[3])
        def make(fast):
            return CampActor(obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                             hidden_dims=(32,), obs_normalization=False, reader_cfg=reader_cfg,
                             reader_amp=False, reader_checkpoint=False, fast_terrain_read=fast)
        reference, fast = make(False), make(True)
        fast.load_state_dict(reference.state_dict())
        self.assertEqual(set(reference.state_dict()), set(fast.state_dict()))
        output, fast_output = reference(obs), fast(obs)
        torch.testing.assert_close(fast_output, output, rtol=2e-5, atol=2e-6)
        weight = torch.randn_like(output)
        (output * weight).sum().backward()
        (fast_output * weight).sum().backward()
        for (name, before), (other, after) in zip(reference.named_parameters(), fast.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(before.grad is None, after.grad is None, name)
            if before.grad is not None:
                torch.testing.assert_close(after.grad, before.grad, rtol=2e-4, atol=2e-5, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for BF16 control path')
    def test_fast_read_preserves_cuda_bfloat16_actions_and_gradients(self):
        reader_cfg = {k: v for k, v in self.cfg.items() if k != 'auxiliary_objective'}
        obs = TensorDict({'policy': torch.randn(7, 12, device='cuda'),
                          'range_features': torch.randn(7, 280, device='cuda')}, batch_size=[7])
        def make(fast, recompute):
            return CampActor(obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                             hidden_dims=(32,), obs_normalization=False, reader_cfg=reader_cfg,
                             reader_amp=True, reader_checkpoint=recompute,
                             reader_chunk_size=4, fast_terrain_read=fast).cuda()
        reference, fast = make(False, True), make(True, False)
        fast.load_state_dict(reference.state_dict())
        output, fast_output = reference(obs), fast(obs)
        torch.testing.assert_close(fast_output, output, rtol=2e-3, atol=2e-4)
        weight = torch.randn_like(output)
        (output * weight).sum().backward()
        (fast_output * weight).sum().backward()
        for (name, before), (other, after) in zip(reference.named_parameters(), fast.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(before.grad is None, after.grad is None, name)
            if before.grad is not None:
                torch.testing.assert_close(after.grad, before.grad, rtol=2e-2, atol=2e-3, msg=name)

    def test_all_learning_paths_and_no_gradient_to_reference_labels(self):
        model = CampTerrainMemory(**self.cfg)
        image = torch.zeros(2, 2, 8, 16)
        image[:, 0], image[:, 1] = .025, 1.
        target = torch.zeros(2, 3, 2, 32, 32)
        target[:, :, 0], target[:, :, 1] = -1., 1.
        target.requires_grad_()
        result = model.auxiliary_loss(image, image, image, body_features=torch.randn(2, 64), terrain_targets=target)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertIsNone(target.grad)
        for prefix in ['encoder.', 'gru.', 'reader.terrain_attention.', 'reader.surface_decoder.']:
            grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), prefix)
        self.assertEqual(result['mem'].item(), 0.)
        self.assertEqual(result['full'].item(), 0.)

    def test_actor_action_does_not_depend_on_privileged_labels_or_cache_gradients(self):
        reader_cfg = {k: v for k, v in self.cfg.items() if k != 'auxiliary_objective'}
        cache_size = 16 + 8 * 33
        cache = torch.randn(2, cache_size, requires_grad=True)
        obs = TensorDict({'policy': torch.randn(2, 12), 'range_features': cache,
                          'camp_terrain_targets': torch.randn(2, 3, 2, 32, 32)}, batch_size=[2])
        actor = CampActor(obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                          hidden_dims=(32,), obs_normalization=False, reader_cfg=reader_cfg, reader_amp=False)
        a = actor(obs)
        changed = obs.clone()
        changed['camp_terrain_targets'] = torch.full_like(changed['camp_terrain_targets'], 999.)
        torch.testing.assert_close(a, actor(changed), rtol=0, atol=0)
        a.square().mean().backward()
        self.assertIsNone(cache.grad)
        compact = compact_range_observations(obs, cache)
        self.assertNotIn('camp_terrain_targets', compact)

    def test_extra_terrain_labels_follow_exact_episode_capture_sequences(self):
        storage = PerceptionStorage()
        image = torch.ones(2, 2, 8, 16)
        hidden = torch.zeros(2, 16)
        for capture in (0, 1):
            saved = storage.add_indexed(image, image, image, hidden, torch.tensor([0, 1]),
                                        torch.tensor([0, 0]), torch.tensor([capture, capture]))
            labels = torch.empty(2, 3, 2, 32, 32)
            labels[0] = capture
            labels[1] = capture + 10
            saved['terrain_targets'] = labels
        count = 0
        for batch in storage.sequence_plan(3, 1):
            expected = batch.capture_ids + 10 * batch.env_ids
            torch.testing.assert_close(batch.terrain_targets[:, :, 0, 0, 0, 0], expected.float())
            count += batch.anchor_count
        self.assertEqual(count, 4)

    def test_deployment_memory_runs_from_only_current_scan_and_proprioception(self):
        model = CampTerrainMemory(**self.cfg)
        reader_cfg = {k: v for k, v in self.cfg.items() if k != 'auxiliary_objective'}
        obs = TensorDict({'policy': torch.randn(2, 12), 'range_features': torch.zeros(2, 280)}, batch_size=[2])
        actor = CampActor(obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                          hidden_dims=(32,), obs_normalization=False, reader_cfg=reader_cfg, reader_amp=False)
        policy = RangeJEPAInferencePolicy(actor, model)
        image = torch.zeros(2, 2, 8, 16)
        image[:, 0], image[:, 1] = .025, 1.
        raw = TensorDict({'policy': obs['policy'], 'range_obs': image,
                          'range_capture_id': torch.zeros(2, 1, dtype=torch.long),
                          'range_episode_id': torch.zeros(2, 1, dtype=torch.long)}, batch_size=[2])
        output = policy(raw)
        self.assertEqual(tuple(output.shape), (2, 29))
        self.assertTrue(torch.isfinite(output).all())
        raw['range_capture_id'] += 1
        self.assertTrue(torch.isfinite(policy(raw)).all())
        self.assertTrue((policy.last_capture == 1).all())

    def test_factory_preserves_native_fov_and_explicit_terrain_metadata(self):
        cfg = _model_configuration(dict(variant='CAMP', **self.cfg))
        self.assertEqual(cfg['vertical_fov'], (52., -7.))
        self.assertEqual(cfg['bev_grid_size'], 32)
        self.assertEqual(cfg['source_priors'], (1., .85, .35))


if __name__ == '__main__':
    unittest.main()
