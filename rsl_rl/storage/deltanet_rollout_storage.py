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
    """Store DeltaNet state only at fixed truncated-BPTT boundaries.

    Fixed chunks start from the online state captured during collection.
    Episode boundaries inside a chunk start from zero state, so retaining a
    state tensor for every time step would still duplicate data that sequence
    replay can reconstruct causally.
    """

    initial_actor_state: torch.Tensor | None = None
    actor_boundary_states: torch.Tensor | None = None

    def __init__(self, *args, **kwargs) -> None:
        tbptt_steps = kwargs.pop("tbptt_steps", None)
        if kwargs.pop("store_next_observations", False):
            raise ValueError("DeltaNetRolloutStorage does not store unused next observations.")
        super().__init__(*args, store_next_observations=False, **kwargs)
        self.tbptt_steps = self.num_transitions_per_env if tbptt_steps is None else int(tbptt_steps)
        if self.tbptt_steps <= 0:
            raise ValueError("DeltaNet tbptt_steps must be positive.")
        if self.num_transitions_per_env % self.tbptt_steps:
            raise ValueError("DeltaNet rollout length must be divisible by tbptt_steps.")
        self.num_tbptt_chunks = self.num_transitions_per_env // self.tbptt_steps

    def _save_hidden_states(self, hidden_states) -> None:
        actor, critic = hidden_states
        if critic is not None:
            raise ValueError("DeltaNetRolloutStorage requires a feed-forward critic.")
        if actor is None:
            raise ValueError("DeltaNet actor state must be initialized before collection.")
        if self.step % self.tbptt_steps:
            return
        if self.actor_boundary_states is None:
            self.actor_boundary_states = actor.new_zeros(
                self.num_tbptt_chunks,
                *actor.shape,
            )
            self.initial_actor_state = self.actor_boundary_states[0]
        boundary = self.step // self.tbptt_steps
        self.actor_boundary_states[boundary].copy_(actor.detach())

    def clear(self) -> None:
        super().clear()
        self.initial_actor_state = None
        self.actor_boundary_states = None

    @staticmethod
    def _concatenate_time_chunks(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Move fixed time chunks into the batch dimension without copying history order."""
        chunks = tuple(
            tensor[start : start + chunk_size]
            for start in range(0, tensor.shape[0], chunk_size)
        )
        return torch.cat(chunks, dim=1)

    def recurrent_mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[RolloutStorage.Batch, None, None]:
        if not 1 <= num_mini_batches <= self.num_envs:
            raise ValueError("num_mini_batches must be between one and num_envs.")
        if self.step != self.num_transitions_per_env:
            raise ValueError("DeltaNet update requires a complete rollout window.")
        if self.actor_boundary_states is None:
            raise ValueError("DeltaNet TBPTT boundary states are missing.")
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * self.num_envs // num_mini_batches
                stop = (index + 1) * self.num_envs // num_mini_batches
                observations = []
                masks = []
                initial_states = []
                for chunk, chunk_start in enumerate(
                    range(0, self.num_transitions_per_env, self.tbptt_steps)
                ):
                    chunk_stop = chunk_start + self.tbptt_steps
                    done = self.dones[chunk_start:chunk_stop, start:stop]
                    obs, mask = split_and_pad_trajectories(
                        self.observations[chunk_start:chunk_stop, start:stop],
                        done,
                    )
                    starts = torch.ones_like(done.squeeze(-1), dtype=torch.bool)
                    starts[1:] = done[:-1].squeeze(-1).bool()
                    env_ids, times = starts.transpose(0, 1).nonzero(as_tuple=True)
                    initial = self.actor_boundary_states[chunk, :, start:stop][:, env_ids]
                    initial = initial * (times == 0)[None, :, None]
                    observations.append(obs)
                    masks.append(mask)
                    initial_states.append(initial)
                yield RolloutStorage.Batch(
                    observations=torch.cat(observations, dim=1),
                    masks=torch.cat(masks, dim=1),
                    hidden_states=(torch.cat(initial_states, dim=1), None),
                    actions=self._concatenate_time_chunks(
                        self.actions[:, start:stop], self.tbptt_steps
                    ),
                    values=self._concatenate_time_chunks(
                        self.values[:, start:stop], self.tbptt_steps
                    ),
                    advantages=self._concatenate_time_chunks(
                        self.advantages[:, start:stop], self.tbptt_steps
                    ),
                    returns=self._concatenate_time_chunks(
                        self.returns[:, start:stop], self.tbptt_steps
                    ),
                    old_actions_log_prob=self._concatenate_time_chunks(
                        self.actions_log_prob[:, start:stop], self.tbptt_steps
                    ),
                    old_distribution_params=tuple(
                        self._concatenate_time_chunks(
                            parameter[:, start:stop], self.tbptt_steps
                        )
                        for parameter in self.distribution_params
                    ),
                )


__all__ = ["DeltaNetRolloutStorage"]
