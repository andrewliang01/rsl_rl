# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical and optimizer-isolation contracts for dual-rate DeltaNet training."""

import copy
import torch

import pytest
from test_mid360_deltanet import actor, critic, observations

from rsl_rl.algorithms import DeltaNetDualRatePPO
from rsl_rl.modules import DeltaNetBlock


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("length,conv_size", [(1, 4), (9, 4), (9, 1), (320, 4)])
def test_parallel_matches_recurrent_outputs_states_and_gradients(length, conv_size):
    block = DeltaNetBlock(dim=8, heads=2, head_dim=4, ffn_dim=12, conv_size=conv_size, chunk_size=3).double()
    sequential = copy.deepcopy(block)
    x = torch.randn(length, 3, 8, dtype=torch.double, requires_grad=True)
    state = torch.randn(3, block.state_size, dtype=torch.double, requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    other_state = state.detach().clone().requires_grad_()
    resets = torch.rand(length, 3) < 0.35
    resets[:, 2] = False
    result, final = block(x, state, resets)
    memory, outputs = other_state, []
    for step in range(length):
        memory = memory * (~resets[step])[:, None]
        out, memory = sequential(other_x[step : step + 1], memory)
        outputs.append(out)
    expected = torch.cat(outputs)
    torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(final, memory, rtol=1e-7, atol=1e-8)
    (result[-1].square().sum() + final.square().sum()).backward()
    (expected[-1].square().sum() + memory.square().sum()).backward()
    torch.testing.assert_close(x.grad, other_x.grad, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(state.grad, other_state.grad, rtol=1e-6, atol=1e-8)
    assert x.grad[0, 2].abs().sum() > 0
    for a, b in zip(block.parameters(), sequential.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-8)


def make_algorithm(distributed=None):
    obs = observations(4)
    policy, value = actor(obs, 2), critic(obs)
    storage = DeltaNetDualRatePPO._make_storage(
        "rl",
        4,
        3,
        obs,
        [29],
        "cpu",
        tbptt_steps=8,
        actor=policy,
        critic=value,
    )
    return DeltaNetDualRatePPO(
        policy,
        value,
        storage,
        estimator_window_steps=8,
        tbptt_steps=8,
        estimator_num_learning_epochs=1,
        estimator_num_mini_batches=2,
        num_learning_epochs=1,
        num_mini_batches=2,
        device="cpu",
        multi_gpu_cfg=distributed,
        profile_learning=distributed is None,
    )


def collect(alg, rollout_id=0, rank=0):
    obs = observations(4)
    with torch.inference_mode():
        for step in range(3):
            alg.act(obs)
            obs = observations(4)
            dones = torch.tensor([step % 2 == rank, False, step == 2, (rollout_id + step) % 3 == 0])
            alg.process_env_step(obs, torch.randn(4), dones, {})
        alg.compute_returns(obs)


def test_dual_rate_preserves_samples_and_isolates_gradients_and_replay():
    alg = make_algorithm()
    initial_policy = copy.deepcopy(alg.actor.mlp.state_dict())
    expected_tails = [3, 6, 1, 4, 7, 2, 5, 0]
    expected_updates = [0, 0, 1, 1, 1, 2, 2, 3]
    for rollout in range(8):
        before = copy.deepcopy(alg.actor.estimator.state_dict())
        collect(alg, rollout)
        batch = next(alg._batches())
        alg._forward_actor(batch)
        torch.testing.assert_close(
            alg.actor.get_output_log_prob(batch.actions), batch.old_actions_log_prob.squeeze(-1)
        )
        assert alg.storage.next_observations is None
        assert alg.storage.saved_hidden_state_a is None
        assert "height_scan_policy" not in alg.storage.observations
        tail = alg.estimator_storage.step - 8
        if tail >= 0:
            expected_data = alg.estimator_storage.data[8 : alg.estimator_storage.step].clone()
            data, resets, state = next(alg.estimator_storage.batches(2, 1))
            assert data.batch_size == (8, 2)
        metrics = alg.update()
        assert metrics["estimator_buffer_steps"] == expected_tails[rollout]
        assert metrics["estimator_updates"] == expected_updates[rollout]
        assert alg.ppo_updates * 3 == alg.estimator_updates * 8 + alg.estimator_storage.step
        if tail >= 0:
            for key in expected_data.keys():
                torch.testing.assert_close(alg.estimator_storage.data[:tail][key], expected_data[key])
            assert any(not torch.equal(v, alg.actor.estimator.state_dict()[k]) for k, v in before.items())
            assert "map_rmse_m" in metrics
        else:
            for key, value in before.items():
                torch.testing.assert_close(value, alg.actor.estimator.state_dict()[key], rtol=0, atol=0)
            assert "map_rmse_m" not in metrics
        assert all(p.grad is None for p in alg.actor.estimator.parameters())
        assert alg.actor.get_hidden_state().grad_fn is None
    assert any(not torch.equal(v, alg.actor.mlp.state_dict()[k]) for k, v in initial_policy.items())


def test_estimator_isolation_and_resume_restores_both_optimizers():
    alg = make_algorithm()
    for i in range(3):
        collect(alg, i)
        if i < 2:
            alg.update()
        else:
            super(DeltaNetDualRatePPO, alg).update()
    before_actor = copy.deepcopy(alg.actor.mlp.state_dict())
    before_critic = copy.deepcopy(alg.critic.state_dict())
    alg._update_estimator()
    for key, value in before_actor.items():
        torch.testing.assert_close(value, alg.actor.mlp.state_dict()[key], rtol=0, atol=0)
    for key, value in before_critic.items():
        torch.testing.assert_close(value, alg.critic.state_dict()[key], rtol=0, atol=0)
    saved = copy.deepcopy(alg.save())
    restored = make_algorithm()
    assert restored.load(saved)
    assert restored.estimator_updates == alg.estimator_updates
    assert restored.estimator_storage.step == 0
    assert restored.actor.get_hidden_state() is None
    assert restored.optimizer.state_dict()["state"]
    assert restored.estimator_optimizer.state_dict()["state"]
    del saved["deltanet_dual_rate_config"]
    with pytest.raises(ValueError, match="dual-rate schedule"):
        restored.load(saved)


def distributed_worker(rank, rendezvous):
    from datetime import timedelta

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        alg = make_algorithm({"global_rank": rank, "local_rank": rank, "world_size": 2})
        alg.broadcast_parameters()
        torch.manual_seed(100 + rank)
        for rollout in range(3):
            collect(alg, rollout, rank)
            metrics = alg.update()
        assert metrics["estimator_updates"] == 1
        flat = torch.cat([p.detach().flatten() for p in alg.actor.parameters()])
        flat = torch.cat((flat, *(p.detach().flatten() for p in alg.critic.parameters())))
        other = [torch.zeros_like(flat) for _ in range(2)]
        torch.distributed.all_gather(other, flat)
        torch.testing.assert_close(other[0], other[1], atol=0, rtol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_updates_with_different_resets_remain_synchronized(tmp_path):
    torch.multiprocessing.spawn(distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
