"""Time batching preserves recurrent state, supervision, and trainable gradients."""

import copy
import unittest
from unittest.mock import patch

import torch

from rsl_rl.models.camp import CampTerrainMemory
from rsl_rl.models.range_direct_objective import direct_geometry_loss


class CampBatchedAuxTests(unittest.TestCase):
    def compare_sequence(self, device):
        torch.manual_seed(197)
        reference = CampTerrainMemory(
            image_height=8, image_width=16, hidden_size=16, bank_size=8,
            context_grid=(2, 4), auxiliary_objective='direct_geometry',
            vertical_fov=(52., -7.),
        ).to(device)
        batched = copy.deepcopy(reference)
        t, b = 4, 3
        views = [torch.rand(t, b, 2, 8, 16, device=device) for _ in range(3)]
        for image in views:
            image[:, :, 1] = image[:, :, 1] > .25
        # Labels and body features must stay on the detached supervision side.
        for image in views[1:]:
            image.requires_grad_()
        terrain = torch.rand(t, b, 3, 2, 32, 32, device=device)
        terrain[:, :, :, 0] -= 1.
        terrain[:, :, :, 1] = terrain[:, :, :, 1] > .2
        terrain.requires_grad_()
        body = torch.randn(t, b, 64, device=device, requires_grad=True)
        reset = torch.zeros(t, b, dtype=torch.bool, device=device)
        reset[2, 1] = True
        active = torch.ones_like(reset)
        active[1, 2] = False
        active[-1, -1] = False
        initial = torch.randn(b, 16, device=device, requires_grad=True)
        kwargs = dict(
            body_features=body, terrain_targets=terrain, initial_hidden=initial,
            reset=reset, active=active,
            map_domains=torch.rand_like(views[0]) > .3,
            support=torch.rand(t, b, 1, 8, 16, device=device) > .2,
        )
        old = direct_geometry_loss(reference, *views, batch_time=False, **kwargs)
        new = direct_geometry_loss(batched, *views, batch_time=True, **kwargs)
        self.assertEqual(set(new), set(old))
        self.assertEqual(new['anchor_count'].item(), active.sum().item())
        for key in old:
            torch.testing.assert_close(new[key], old[key], rtol=2e-4, atol=2e-5, msg=key)
        old['loss'].backward()
        initial_gradient = initial.grad.clone()
        initial.grad = None
        new['loss'].backward()
        torch.testing.assert_close(initial.grad, initial_gradient, rtol=1e-3, atol=3e-5)
        for (name, before), (other, after) in zip(reference.named_parameters(), batched.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(before.grad is None, after.grad is None, name)
            if before.grad is not None:
                torch.testing.assert_close(after.grad, before.grad, rtol=1e-3, atol=3e-5, msg=name)
        for target in (*views[1:], terrain, body):
            self.assertIsNone(target.grad)

    def test_cpu_sequence_loss_state_and_all_gradients(self):
        self.compare_sequence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_sequence_loss_state_and_all_gradients(self):
        self.compare_sequence('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_variable_online_capture_disables_search_and_restores_flags(self):
        model = CampTerrainMemory(image_height=8, image_width=16,
                                  auxiliary_objective='direct_geometry').cuda()
        image = torch.zeros(2, 2, 8, 16, device='cuda')
        states = []
        original = type(model).__mro__[1].encode

        def encode(instance, value):
            states.append((torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled))
            return original(instance, value)

        with torch.backends.cudnn.flags(enabled=True, benchmark=True):
            with patch.object(type(model).__mro__[1], 'encode', encode), torch.no_grad():
                model.encode(image)
            self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertEqual(states, [(False, True)])


if __name__ == '__main__':
    unittest.main()
