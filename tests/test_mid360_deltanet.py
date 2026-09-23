# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for MID360 DeltaNet models, storage and PPO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import DeltaNetPPO
from rsl_rl.models import MID360DeltaNetActor, SequenceElevationHistoryCritic
from rsl_rl.modules import delta_rule_chunkwise
from rsl_rl.storage import DeltaNetRolloutStorage


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(previous)


def observations(batch: int = 3) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch, 96),
            "height_scan_policy": torch.rand(batch, 1, 16, 96) * 1.8 + 0.05,
            "critic": torch.randn(batch, 5 * 99),
            "height_scan_critic": torch.randn(batch, 5, 28, 20),
            "deltanet_velocity_target": torch.randn(batch, 3),
        },
        batch_size=[batch],
    )


OBS_GROUPS = {
    "actor": ["policy", "height_scan_policy"],
    "critic": ["critic", "height_scan_critic"],
}


def actor(obs: TensorDict, layers: int = 3) -> MID360DeltaNetActor:
    return MID360DeltaNetActor(
        obs,
        OBS_GROUPS,
        "actor",
        29,
        dim=32,
        num_delta_layers=layers,
        heads=2,
        head_dim=8,
        ffn_dim=48,
        chunk_size=3,
        hidden_dims=(32,),
        obs_normalization=False,
        cnn_circular_azimuth=True,
    )


def critic(obs: TensorDict) -> SequenceElevationHistoryCritic:
    return SequenceElevationHistoryCritic(
        obs,
        OBS_GROUPS,
        "critic",
        1,
        expected_proprio_dim=495,
        elevation_history_length=5,
        vision_spatial_size=(28, 20),
        hidden_dims=(32,),
        cnn_hidden_dims=(4,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(2,),
        prop_hidden_dims=(16,),
        prop_feature_dim=8,
        vision_feature_dim=8,
    )


def test_delta_rule_matches_stepwise_recurrence_and_has_gradients():
    q = torch.randn(2, 2, 7, 4, requires_grad=True)
    k = torch.randn(2, 2, 7, 4, requires_grad=True)
    v = torch.randn(2, 2, 7, 4, requires_grad=True)
    beta = torch.rand(2, 2, 7, requires_grad=True)
    initial = torch.randn(2, 2, 4, 4, requires_grad=True)

    actual, actual_state = delta_rule_chunkwise(q, k, v, beta, initial, chunk_size=3)
    state = initial
    expected = []
    for step in range(q.shape[2]):
        key = k[:, :, step]
        error = v[:, :, step] - (state @ key.unsqueeze(-1)).squeeze(-1)
        state = state + beta[:, :, step, None, None] * error.unsqueeze(-1) * key.unsqueeze(-2)
        expected.append((state @ q[:, :, step].unsqueeze(-1)).squeeze(-1))
    expected = torch.stack(expected, dim=2)

    torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-5)
    torch.testing.assert_close(actual_state, state, rtol=1.0e-4, atol=1.0e-5)
    (actual.square().mean() + actual_state.square().mean()).backward()
    for tensor in (q, k, v, beta, initial):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_current_frame_only_circular_cnn_and_selectable_delta_layers():
    obs = observations()
    model = actor(obs, layers=3).eval()
    assert len(model.estimator.blocks) == 3
    assert model.model_config["num_delta_layers"] == 3
    assert model.model_config["cnn_circular_azimuth"] is True
    assert model.estimator.vision.circular_azimuth is True
    assert model(obs).shape == (3, 29)
    assert model.last_map.shape == (3, 28 * 20)
    assert model.last_velocity.shape == (3, 3)
    with pytest.raises(ValueError, match="one current frame"):
        bad = obs.clone()
        bad["height_scan_policy"] = torch.zeros(3, 2, 16, 96)
        actor(bad)


def test_memory_is_implicit_and_resets_per_environment():
    obs = observations()
    model = actor(obs, layers=2).eval()
    first = model(obs)
    second = model(obs)
    assert not torch.allclose(first, second)
    old_state = model.get_hidden_state().clone()
    model.reset(torch.tensor([True, False, False]))
    assert torch.count_nonzero(model.get_hidden_state()[:, 0]) == 0
    torch.testing.assert_close(model.get_hidden_state()[:, 1:], old_state[:, 1:])
    reset_step = model(obs)
    torch.testing.assert_close(reset_step[0], first[0])


def test_actor_cannot_read_critic_history_or_supervision_targets():
    obs = observations()
    model = actor(obs, layers=2).eval()
    expected = model(obs)
    changed = obs.clone()
    changed["critic"].fill_(float("nan"))
    changed["height_scan_critic"].fill_(float("nan"))
    changed["deltanet_velocity_target"].fill_(float("nan"))
    model.reset()
    torch.testing.assert_close(model(changed), expected)


def test_explicit_state_deployment_matches_actor_step():
    obs = observations(batch=2)
    model = actor(obs, layers=2).eval()
    deployment = model.as_jit().eval()
    state = model.estimator.initial_state(2, obs["policy"])
    expected = model(obs)
    actual, height, velocity, next_state = deployment(
        obs["policy"],
        obs["height_scan_policy"],
        state,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(height, model.last_map)
    torch.testing.assert_close(velocity, model.last_velocity)
    torch.testing.assert_close(next_state, model.get_hidden_state())


@pytest.mark.parametrize("distributed", [False, True])
def test_history_critic_and_real_ppo_update(distributed, monkeypatch):
    obs = observations()
    value_model = critic(obs)
    assert value_model(obs).shape == (3, 1)
    cfg = {
        "num_steps_per_env": 4,
        "obs_groups": OBS_GROUPS,
        "actor": {
            "class_name": "rsl_rl.models:MID360DeltaNetActor",
            "expected_proprio_dim": 96,
            "expected_action_dim": 29,
            "dim": 32,
            "num_delta_layers": 2,
            "heads": 2,
            "head_dim": 8,
            "ffn_dim": 48,
            "hidden_dims": (32,),
            "obs_normalization": False,
        },
        "critic": {
            "class_name": "rsl_rl.models:SequenceElevationHistoryCritic",
            "expected_proprio_dim": 495,
            "elevation_history_length": 5,
            "vision_spatial_size": (28, 20),
            "hidden_dims": (32,),
            "cnn_hidden_dims": (4,),
            "cnn_kernel_sizes": (3,),
            "cnn_strides": (2,),
            "prop_hidden_dims": (16,),
            "prop_feature_dim": 8,
            "vision_feature_dim": 8,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:DeltaNetPPO",
            "map_target_set": "height_scan_critic",
            "map_target_history_index": -1,
            "velocity_target_set": "deltanet_velocity_target",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.008,
        },
        "multi_gpu": (
            {"global_rank": 0, "local_rank": 0, "world_size": 2}
            if distributed
            else None
        ),
    }
    collective_calls = []
    if distributed:
        def fake_all_reduce(tensor, op):
            assert op == torch.distributed.ReduceOp.SUM
            collective_calls.append(("all_reduce", tensor.numel()))

        def fake_broadcast(tensor, src):
            assert src == 0
            collective_calls.append(("broadcast", tensor.numel()))

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    algorithm = DeltaNetPPO.construct_algorithm(
        obs,
        SimpleNamespace(num_envs=3, num_actions=29),
        cfg,
        "cpu",
    )
    assert isinstance(algorithm.storage, DeltaNetRolloutStorage)
    assert algorithm.is_multi_gpu is distributed
    before = algorithm.actor.estimator.map_head[-1].weight.detach().clone()
    with torch.no_grad():
        for step in range(4):
            algorithm.act(obs)
            obs = observations()
            algorithm.process_env_step(
                obs,
                torch.randn(3),
                torch.tensor([step == 1, False, False]),
                {},
            )
        algorithm.compute_returns(obs)
    metrics = algorithm.update()
    assert all(value == value for value in metrics.values())
    assert not torch.equal(before, algorithm.actor.estimator.map_head[-1].weight)
    assert algorithm.storage.initial_actor_state is None
    if distributed:
        assert ("broadcast", 1) in collective_calls
        assert any(name == "all_reduce" and size > 1000 for name, size in collective_calls)
        assert ("all_reduce", 6) in collective_calls
