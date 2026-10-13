"""Two-process global-batch equivalence checks with Gloo and optional NCCL."""
import copy
from datetime import timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rsl_rl.models.mid360_gru_heightmap import (
    Mid360GRUHeightmap, broadcast_estimator, update_estimator,
)


def _losses(estimator, rank, iteration):
    generator = torch.Generator().manual_seed(100 + iteration)
    hidden = None
    losses = []
    for step in range(3):
        inputs = (
            torch.rand(4, 10, 16, 96, generator=generator) * 40,
            torch.randn(4, 5, 96, generator=generator),
            torch.randn(4, 10, 96, generator=generator),
            torch.rand(4, 10, generator=generator),
        )
        target = torch.randn(4, 28, 20, generator=generator) * 10
        if rank is not None:
            inputs = tuple(value[rank * 2:(rank + 1) * 2] for value in inputs)
            target = target[rank * 2:(rank + 1) * 2]
        parameter = next(estimator.parameters())
        inputs = tuple(value.to(parameter) for value in inputs)
        target = target.to(parameter)
        prediction, hidden = estimator(*inputs, hidden)
        losses.append(F.mse_loss(prediction, target))
        if step == 1:
            mask = torch.tensor([False, True, True, False])
            if rank is not None:
                mask = mask[rank * 2:(rank + 1) * 2]
            hidden = hidden * mask.to(hidden.device)[None, :, None]
    return losses


def _worker(rank, rendezvous, directory, backend):
    torch.set_num_threads(1)
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42 + rank)
        # Fixed BatchNorm stats let CPU/Gloo isolate gradient averaging against
        # a concatenated batch. Production uses SyncBatchNorm on CUDA.
        estimator = Mid360GRUHeightmap(hidden_size=16).to(device=device, dtype=torch.float64)
        estimator.train(backend == "nccl")
        if backend == "nccl":
            estimator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(estimator)
        broadcast_estimator(estimator)
        reference = Mid360GRUHeightmap(hidden_size=16).to(device=device, dtype=torch.float64)
        reference.train(backend == "nccl")
        reference.load_state_dict(estimator.state_dict())
        actor = torch.nn.Linear(2, 2).requires_grad_(False)
        frozen_state = copy.deepcopy(actor.state_dict())
        policy = SimpleNamespace(estimator=estimator, actor=actor, detach_hidden=lambda: None)
        baseline = SimpleNamespace(estimator=reference, actor=actor, detach_hidden=lambda: None)
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
        for iteration in range(3):
            update_estimator(policy, optimizer, _losses(estimator, rank, iteration), distributed=True)
            update_estimator(baseline, reference_optimizer, _losses(reference, None, iteration))
            for actual, expected in zip(estimator.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
            for actual, expected in zip(optimizer.state.values(), reference_optimizer.state.values()):
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], atol=1e-7, rtol=1e-4,
                                               check_device=False)
            for actual, expected in zip(estimator.buffers(), reference.buffers()):
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)
            if iteration == 0:
                checkpoint = Path(directory) / "resume.pt"
                if rank == 0:
                    torch.save({"model": estimator.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
                dist.barrier()
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                estimator.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                broadcast_estimator(estimator)
        flat = torch.cat([p.detach().flatten() for p in estimator.parameters()])
        copies = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(copies, flat)
        torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, frozen_state[key], rtol=0, atol=0)
        assert all(p.grad is None for p in actor.parameters())
        bad_loss = next(estimator.parameters()).sum() * (float("nan") if rank == 1 else 1)
        with unittest.TestCase().assertRaisesRegex(FloatingPointError, "Non-finite"):
            update_estimator(policy, optimizer, [bad_loss], distributed=True)
    finally:
        dist.destroy_process_group()


class DistributedEstimatorTest(unittest.TestCase):
    def test_global_batch_equivalence_resume_and_nonfinite(self):
        with TemporaryDirectory() as directory:
            rendezvous = "file://" + os.path.join(directory, "rendezvous")
            mp.spawn(_worker, args=(rendezvous, directory, "gloo"), nprocs=2, join=True)

    @unittest.skipUnless(os.environ.get("MID360_TEST_NCCL") == "1", "Set MID360_TEST_NCCL=1 for two GPUs")
    def test_nccl_sync_batchnorm_global_batch_equivalence(self):
        with TemporaryDirectory() as directory:
            rendezvous = "file://" + os.path.join(directory, "rendezvous")
            mp.spawn(_worker, args=(rendezvous, directory, "nccl"), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
