"""Contracts for reusable UniFP algorithm naming and validation."""

from __future__ import annotations

import os
from itertools import chain

import torch
import torch.multiprocessing as mp
from torch import nn

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.unifp_adaptation_ppo import UniFPAdaptationPPO
from rsl_rl.algorithms.unifp_adaptation_ppo import _adaptation_part_names


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.decoder = nn.Linear(2, 2)
        self.body = nn.Linear(2, 1)


def _algorithm(*, multi_gpu: bool = False) -> UniFPAdaptationPPO:
    algorithm = UniFPAdaptationPPO.__new__(UniFPAdaptationPPO)
    algorithm.actor = _Actor()
    algorithm.critic = nn.Linear(2, 1)
    algorithm.optimizer = torch.optim.Adam(
        chain(algorithm.actor.parameters(), algorithm.critic.parameters()), lr=1.0e-3
    )
    algorithm.rnd = None
    algorithm.rnd_optimizer = None
    algorithm.dwaq = None
    algorithm.dwaq_optimizer = None
    algorithm._adaptation_params = list(algorithm.actor.encoder.parameters()) + list(
        algorithm.actor.decoder.parameters()
    )
    algorithm.adaptation_optimizer = torch.optim.Adam(algorithm._adaptation_params, lr=5.0e-6)
    algorithm.freeze_adaptation_after_iter = 4000
    algorithm.adaptation_frozen = False
    algorithm._update_count = 0
    algorithm.is_multi_gpu = multi_gpu
    algorithm.gpu_world_size = 2 if multi_gpu else 1
    algorithm.device = "cpu"
    return algorithm


def _distributed_gradient_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        algorithm = _algorithm(multi_gpu=True)
        for param in algorithm._adaptation_params:
            param.grad = torch.full_like(param, float(rank + 1))
        algorithm._reduce_adaptation_gradients()
        flattened = torch.cat([param.grad.view(-1) for param in algorithm._adaptation_params])
        torch.save(flattened, f"{output_dir}/rank_{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def test_legacy_adaptation_names_remain_backward_compatible():
    assert _adaptation_part_names(2) == (
        "adaptation_base_velocity",
        "adaptation_gripper_pos",
    )
    assert _adaptation_part_names(3) == (
        "adaptation_base_velocity",
        "adaptation_gripper_pos",
        "adaptation_gripper_orn",
    )
    assert _adaptation_part_names(4)[2:] == (
        "adaptation_force_ee",
        "adaptation_force_base",
    )


def test_adaptation_gradients_are_averaged_across_ranks(monkeypatch):
    algorithm = _algorithm(multi_gpu=True)
    for index, param in enumerate(algorithm._adaptation_params):
        param.grad = torch.full_like(param, float(index + 1))

    remote_grads = torch.cat(
        [torch.full_like(param, float(index + 3)).view(-1) for index, param in enumerate(algorithm._adaptation_params)]
    )

    def fake_all_reduce(flat_grads, op):
        assert op == torch.distributed.ReduceOp.SUM
        flat_grads.add_(remote_grads)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    algorithm._reduce_adaptation_gradients()

    for index, param in enumerate(algorithm._adaptation_params):
        expected = float(index + 2)
        assert torch.allclose(param.grad, torch.full_like(param, expected))


def test_adaptation_gradients_match_in_real_two_process_group(tmp_path):
    init_file = tmp_path / "dist_init"
    mp.spawn(
        _distributed_gradient_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    rank_one = torch.load(tmp_path / "rank_1.pt", weights_only=True)
    expected = torch.full_like(rank_zero, 1.5)
    assert torch.equal(rank_zero, rank_one)
    assert torch.allclose(rank_zero, expected)


def test_adaptation_logs_are_averaged_across_ranks(monkeypatch):
    algorithm = _algorithm(multi_gpu=True)
    logs = {"adaptation": 2.0, "adaptation_rmse_base_velocity": 4.0, "adaptation_frozen": 0.0}

    def fake_all_reduce(values, op):
        assert op == torch.distributed.ReduceOp.SUM
        values.add_(torch.tensor([6.0, 2.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    algorithm._average_adaptation_logs(logs)

    assert logs == {
        "adaptation": 4.0,
        "adaptation_rmse_base_velocity": 3.0,
        "adaptation_frozen": 0.0,
    }


def test_optional_freeze_broadcasts_then_freezes_adaptation(monkeypatch):
    algorithm = _algorithm(multi_gpu=True)
    broadcasts = []

    def fake_broadcast(tensor, src):
        assert src == 0
        broadcasts.append(tensor)

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    algorithm._freeze_adaptation()

    expected_tensors = sum(
        len(list(module.parameters())) + len(list(module.buffers()))
        for module in (algorithm.actor.encoder, algorithm.actor.decoder)
    )
    assert len(broadcasts) == expected_tensors
    assert algorithm.adaptation_frozen is True
    assert all(not param.requires_grad for param in algorithm._adaptation_params)


def test_adaptation_checkpoint_restores_optimizer_count_and_frozen_state():
    source = _algorithm()
    loss = sum(param.square().sum() for param in source._adaptation_params)
    source.adaptation_optimizer.zero_grad()
    loss.backward()
    source.adaptation_optimizer.step()
    source._update_count = 4321
    source.adaptation_frozen = True
    saved = source.save()

    target = _algorithm()
    loaded_iteration = target.load(saved, load_cfg=None, strict=True)

    assert loaded_iteration is True
    assert target._update_count == 4321
    assert target.adaptation_frozen is True
    assert all(not param.requires_grad for param in target._adaptation_params)
    assert len(target.adaptation_optimizer.state) == len(source.adaptation_optimizer.state)
    target.train_mode()
    assert target.actor.encoder.training is False
    assert target.actor.decoder.training is False


def test_legacy_checkpoint_uses_runner_iteration_for_freeze_state():
    source = _algorithm()
    saved = PPO.save(source)
    saved["iter"] = 4500

    target = _algorithm()
    target.load(saved, load_cfg=None, strict=True)

    assert target._update_count == 4500
    assert target.adaptation_frozen is True
