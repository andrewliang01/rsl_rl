"""Transition-storage contracts shared by MultiPPO-derived algorithms."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.multi_ppo import MultiPPO
from rsl_rl.storage.rollout_storage import RolloutStorage


class _NormalizationStub:
    def update_normalization(self, _observations: TensorDict) -> None:
        pass

    def reset(self, _dones: torch.Tensor) -> None:
        pass


def test_process_env_step_stores_the_true_next_observation() -> None:
    num_envs = 2
    current_observations = TensorDict(
        {"reconstruction_target": torch.full((num_envs, 3), -1.0)},
        batch_size=[num_envs],
    )
    next_observations = TensorDict(
        {"reconstruction_target": torch.full((num_envs, 3), 2.0)},
        batch_size=[num_envs],
    )
    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=1,
        obs=current_observations,
        actions_shape=(1,),
        device="cpu",
    )

    algorithm = MultiPPO.__new__(MultiPPO)
    algorithm.actor = _NormalizationStub()
    algorithm._first_critic = _NormalizationStub()
    algorithm.shared_critic = True
    algorithm.is_multi_critic = False
    algorithm.num_critics = 1
    algorithm.rnd = None
    algorithm.device = "cpu"
    algorithm.storage = storage
    algorithm.transition = RolloutStorage.Transition()
    algorithm.transition.observations = current_observations
    algorithm.transition.actions = torch.zeros(num_envs, 1)
    algorithm.transition.values = torch.zeros(num_envs, 1)
    algorithm.transition.actions_log_prob = torch.zeros(num_envs, 1)
    algorithm.transition.distribution_params = (torch.zeros(num_envs, 1),)

    algorithm.process_env_step(
        next_observations,
        rewards=torch.zeros(num_envs),
        dones=torch.zeros(num_envs, dtype=torch.bool),
        extras={},
    )

    torch.testing.assert_close(storage.observations[0], current_observations)
    torch.testing.assert_close(storage.next_observations[0], next_observations)
    assert not torch.equal(
        storage.next_observations[0]["reconstruction_target"],
        storage.observations[0]["reconstruction_target"],
    )
