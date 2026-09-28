# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from tensordict import TensorDict
from types import SimpleNamespace

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.rollout_profiler import RolloutProfiler


def test_disabled_profile_does_not_synchronize(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Disabled profiler must not synchronize CUDA.")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    profiler = RolloutProfiler(False, "cuda:0")
    profiler.mark()
    profiler.mark("env_step")
    assert profiler.metrics(distributed=True) == {}


def test_rollout_nan_interval_crosses_update_boundaries_and_profiles(monkeypatch):
    import rsl_rl.runners.on_policy_runner as module

    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    obs = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])
    steps, checks, logs = [], [], []

    def step(actions):
        steps.append(1)
        return obs, torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}

    runner.env = SimpleNamespace(get_observations=lambda: obs, step=step, device="cpu")
    runner.device = "cpu"
    runner.is_distributed = False
    runner.current_learning_iteration = 0
    runner._formal_training_io = None
    runner.cfg = {
        "num_steps_per_env": 24,
        "check_for_nan_interval": 32,
        "profile_collection": True,
        "algorithm": {"rnd_cfg": None},
    }
    runner.alg = SimpleNamespace(
        train_mode=lambda: None,
        act=lambda obs: torch.zeros(2, 1),
        process_env_step=lambda *args: None,
        compute_returns=lambda obs: None,
        update=lambda: {},
        learning_rate=1e-3,
        get_policy=lambda: SimpleNamespace(output_std=torch.ones(1)),
    )
    runner.logger = SimpleNamespace(
        init_logging_writer=lambda: None,
        writer=None,
        process_env_step=lambda *args: None,
        log=lambda **kwargs: logs.append(kwargs),
    )
    monkeypatch.setattr(module, "check_nan", lambda *args: checks.append(len(steps) - 1))
    runner.learn(3)
    assert checks == [0, 32, 64]
    assert len(steps) == 72
    for log in logs:
        assert len(log["loss_dict"]) == 5
        assert all(value >= 0 for value in log["loss_dict"].values())
