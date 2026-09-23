# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Memory-efficient rollout storage for DeltaNet actors."""

from __future__ import annotations

from collections.abc import Generator

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class DeltaNetRolloutStorage(RolloutStorage):
    """Store only the rollout-boundary DeltaNet state.

    Episode boundaries within a rollout start from zero state, so retaining a
    state tensor for every time step would duplicate data that sequence replay
    can reconstruct causally.
    """

    initial_actor_state: torch.Tensor | None = None

    def _save_hidden_states(self, hidden_states) -> None:
        actor, critic = hidden_states
        if critic is not None:
            raise ValueError("DeltaNetRolloutStorage requires a feed-forward critic.")
        if self.step == 0:
            if actor is None:
                raise ValueError("DeltaNet actor state must be initialized before collection.")
            self.initial_actor_state = actor.detach().clone()

    def clear(self) -> None:
        super().clear()
        self.initial_actor_state = None

    def recurrent_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[RolloutStorage.Batch, None, None]:
        if not 1 <= num_mini_batches <= self.num_envs:
            raise ValueError("num_mini_batches must be between one and num_envs.")
        if self.step != self.num_transitions_per_env:
            raise ValueError("DeltaNet update requires a complete rollout window.")
        if self.initial_actor_state is None:
            raise ValueError("The rollout-boundary DeltaNet state is missing.")
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * self.num_envs // num_mini_batches
                stop = (index + 1) * self.num_envs // num_mini_batches
                done = self.dones[:, start:stop]
                obs, masks = split_and_pad_trajectories(self.observations[:, start:stop], done)
                starts = torch.ones_like(done.squeeze(-1), dtype=torch.bool)
                starts[1:] = done[:-1].squeeze(-1).bool()
                env_ids, times = starts.transpose(0, 1).nonzero(as_tuple=True)
                initial = self.initial_actor_state[:, start:stop][:, env_ids]
                initial = initial * (times == 0)[None, :, None]
                yield RolloutStorage.Batch(
                    observations=obs,
                    masks=masks,
                    hidden_states=(initial, None),
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(
                        parameter[:, start:stop] for parameter in self.distribution_params
                    ),
                )


__all__ = ["DeltaNetRolloutStorage"]
