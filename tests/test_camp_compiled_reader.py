"""Compiled PPO reader preserves parameters, gradients, and in-place reloads."""
import copy
import unittest

import torch
from tensordict import TensorDict

from rsl_rl.models.camp import CampActor


class CampCompiledReaderTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_multiple_live_forwards_gradients_and_reload(self):
        torch.manual_seed(91)
        obs = TensorDict({'policy': torch.randn(7, 12, device='cuda'),
                          'range_features': torch.randn(7, 280, device='cuda')}, batch_size=[7])
        cfg = dict(image_height=8, image_width=16, channels=32, hidden_size=16,
                   bank_size=8, context_grid=(2, 4), vertical_fov=(52., -7.))
        reference = CampActor(obs, {'actor': ['policy', 'range_features']}, 'actor', 29,
                              hidden_dims=(32,), reader_cfg=cfg, reader_chunk_size=4,
                              reader_checkpoint=False, reader_amp=True).cuda()
        compiled = copy.deepcopy(reference)
        compiled.compile_visual_reader = True
        for step in range(2):
            reference.zero_grad(set_to_none=True)
            compiled.zero_grad(set_to_none=True)
            first, other = reference(obs), compiled(obs)
            second, last = reference(obs), compiled(obs)
            torch.testing.assert_close(other, first, rtol=3e-3, atol=3e-4)
            torch.testing.assert_close(last, second, rtol=3e-3, atol=3e-4)
            (first.square().mean() + second.mean()).backward()
            (other.square().mean() + last.mean()).backward()
            self.assertEqual(set(reference.state_dict()), set(compiled.state_dict()))
            for (name, before), (other_name, after) in zip(reference.named_parameters(), compiled.named_parameters()):
                self.assertEqual(name, other_name)
                self.assertEqual(before.grad is None, after.grad is None, name)
                if before.grad is not None:
                    torch.testing.assert_close(after.grad, before.grad, rtol=3e-2, atol=3e-4, msg=name)
            # The shared reader is updated in place by the auxiliary step and
            # strict checkpoint loading; compiled code must observe new values.
            if step == 0:
                with torch.no_grad():
                    reference.reader.terrain_norm.bias.add_(.2)
                compiled.load_state_dict(reference.state_dict(), strict=True)
        self.assertIsNotNone(compiled._compiled_visual_reader)
        with torch.no_grad():
            torch.testing.assert_close(compiled(obs), reference(obs), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
