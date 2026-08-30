# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


def test_rollout_storage_preserves_each_observation_dtype() -> None:
    num_envs = 3
    obs = TensorDict(
        {
            "policy": torch.randn(num_envs, 12, dtype=torch.float32),
            "ray_history": torch.randn(
                num_envs,
                5,
                2,
                4,
                8,
                dtype=torch.float16,
            ),
            "ray_state": torch.randint(
                0,
                2,
                (num_envs, 5, 4, 8),
                dtype=torch.uint8,
            ),
        },
        batch_size=[num_envs],
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=4,
        obs=obs,
        actions_shape=(2,),
        device="cpu",
    )

    assert storage.observations["policy"].dtype == torch.float32
    assert storage.observations["ray_history"].dtype == torch.float16
    assert storage.observations["ray_state"].dtype == torch.uint8
    assert storage.next_observations["policy"].dtype == torch.float32
    assert storage.next_observations["ray_history"].dtype == torch.float16
    assert storage.next_observations["ray_state"].dtype == torch.uint8

    for transition_index in range(storage.num_transitions_per_env):
        transition = RolloutStorage.Transition()
        transition.observations = obs.clone()
        transition.observations["ray_history"].fill_(transition_index + 1)
        transition.next_observations = obs.clone()
        transition.next_observations["ray_history"].fill_(
            transition_index + 1.5
        )
        transition.actions = torch.randn(num_envs, 2)
        transition.rewards = torch.randn(num_envs)
        transition.dones = torch.zeros(num_envs, dtype=torch.bool)
        transition.values = torch.randn(num_envs, 1)
        transition.actions_log_prob = torch.randn(num_envs, 1)
        transition.distribution_params = (
            torch.randn(num_envs, 2),
            torch.rand(num_envs, 2),
        )
        storage.add_transition(transition)

    assert storage.observations["ray_history"].dtype == torch.float16
    assert storage.next_observations["ray_history"].dtype == torch.float16
    for transition_index in range(storage.num_transitions_per_env):
        torch.testing.assert_close(
            storage.observations["ray_history"][transition_index],
            torch.full_like(
                storage.observations["ray_history"][transition_index],
                transition_index + 1,
            ),
        )
        torch.testing.assert_close(
            storage.next_observations["ray_history"][transition_index],
            torch.full_like(
                storage.next_observations["ray_history"][transition_index],
                transition_index + 1.5,
            ),
        )

    batches = list(
        storage.mini_batch_generator(
            num_mini_batches=2,
            num_epochs=1,
        )
    )
    assert len(batches) == 2
    for batch in batches:
        assert batch.observations is not None
        assert batch.next_observations is not None
        assert batch.observations["policy"].dtype == torch.float32
        assert batch.observations["ray_history"].dtype == torch.float16
        assert batch.observations["ray_state"].dtype == torch.uint8
        assert batch.next_observations["policy"].dtype == torch.float32
        assert batch.next_observations["ray_history"].dtype == torch.float16
        assert batch.next_observations["ray_state"].dtype == torch.uint8
        assert batch.dones is not None
        assert batch.dones.dtype == torch.uint8
