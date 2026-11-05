"""Real two-process checks for shared observation statistics and AMP updates."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.camp_amp_ppo import CampAMPPPO
from rsl_rl.models.amp_discriminator import AMPDiscriminator
from rsl_rl.modules import EmpiricalNormalization


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        norm = EmpiricalNormalization(3, synchronize_distributed=True)
        expected = EmpiricalNormalization(3)
        # Unequal batch sizes and separated rank means exercise weighted
        # counts and between-rank variance, including a second running update.
        for step in range(2):
            batches = [torch.arange(6).reshape(2, 3).float() + step,
                       torch.arange(15).reshape(5, 3).float() + 100 - step]
            norm.update(batches[rank])
            expected.update(torch.cat(batches))
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(norm.state_dict()[key], value, rtol=1e-6, atol=1e-5)
        norm.eval()
        frozen = {key: value.clone() for key, value in norm.state_dict().items()}
        norm.update(torch.randn(4, 3))
        for key, value in frozen.items():
            torch.testing.assert_close(norm.state_dict()[key], value, rtol=0, atol=0)

        # Deliberately different initializations and data. Initial broadcast,
        # gradient reduction before clipping and Adam updates must converge
        # to the same discriminator, including its running statistics.
        torch.manual_seed(10 + rank)
        critic = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1))
        algo = AMPPPO(torch.nn.Linear(3, 2), critic,
                      SimpleNamespace(num_envs=4),
                      multi_gpu_cfg=dict(global_rank=rank, world_size=2))
        algo.amp_discriminator = AMPDiscriminator(6, hidden_dims=(8, 4), normalize=True)
        algo.amp_discriminator.obs_normalizer.synchronize_distributed = True
        algo.amp_optimizer = torch.optim.Adam(algo.amp_discriminator.parameters(), lr=1e-3)
        algo.broadcast_parameters()
        for _ in range(3):
            policy = (torch.randn(4, 3) + rank, torch.randn(4, 3) + rank)
            expert = (torch.randn(4, 3) - rank, torch.randn(4, 3) - rank)
            metrics = algo._update_discriminator(policy, expert)
            assert all(bool(torch.isfinite(value)) for value in metrics)
        sample = torch.randn(4, 3) + rank
        before = algo.critic(sample).detach()
        CampAMPPPO._sync_critic_batchnorm_buffers(algo)
        torch.testing.assert_close(algo.critic(sample).detach(), before, rtol=0, atol=0)
        CampAMPPPO._sync_critic_batchnorm_buffers(algo)
        torch.save(dict(normalizer=norm.state_dict(), discriminator=algo.amp_discriminator.state_dict(),
                        critic=algo.critic.state_dict(), optimizer=algo.amp_optimizer.state_dict()),
                   Path(output) / f'rank_{rank}.pt')
    finally:
        dist.destroy_process_group()


class CampDistributedTests(unittest.TestCase):
    def test_global_moments_and_amp_adam_are_identical_across_real_ranks(self):
        with TemporaryDirectory() as directory:
            mp.spawn(_worker, args=('file://' + directory + '/rendezvous', directory), nprocs=2, join=True)
            states = [torch.load(Path(directory) / f'rank_{rank}.pt', weights_only=False) for rank in range(2)]
            def compare(first, second):
                if isinstance(first, torch.Tensor):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                elif isinstance(first, dict):
                    self.assertEqual(first.keys(), second.keys())
                    for key in first:
                        compare(first[key], second[key])
                elif isinstance(first, (list, tuple)):
                    self.assertEqual(len(first), len(second))
                    for one, two in zip(first, second):
                        compare(one, two)
                else:
                    self.assertEqual(first, second)
            compare(*states)


if __name__ == '__main__':
    unittest.main()
