# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent supervised sequences for a slowly updated DeltaNet estimator."""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class DeltaNetPolicyStorage(RolloutStorage):
    """Small PPO buffer with cached actor inputs and no recurrent state copies."""

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Yield flat PPO batches without allocating unused next observations."""
        size = self.num_envs * self.num_transitions_per_env
        if not 1 <= num_mini_batches <= size or size % num_mini_batches:
            raise ValueError("PPO mini-batches must evenly divide the transition count.")
        flat = self.observations.flatten(0, 1)
        for _ in range(num_epochs):
            for ids in torch.randperm(size, device=self.device).chunk(num_mini_batches):
                yield self.Batch(
                    observations=flat[ids],
                    actions=self.actions.flatten(0, 1)[ids],
                    values=self.values.flatten(0, 1)[ids],
                    returns=self.returns.flatten(0, 1)[ids],
                    advantages=self.advantages.flatten(0, 1)[ids],
                    old_actions_log_prob=self.actions_log_prob.flatten(0, 1)[ids],
                    old_distribution_params=tuple(p.flatten(0, 1)[ids] for p in self.distribution_params),
                )


class DeltaNetEstimatorStorage:
    """Preserve exact windows plus the tail up to the next PPO boundary.

    Only normalized proprioception, panorama and supervised labels are retained.
    Boundary states are detached snapshots, as in truncated recurrent PPO.
    """

    def __init__(
        self,
        template: TensorDict,
        initial_state: torch.Tensor,
        window: int,
        tbptt_steps: int,
        ppo_steps: int,
    ) -> None:
        """Allocate one supervised window, its PPO surplus, and boundary states."""
        if not 0 < ppo_steps <= window or tbptt_steps <= 0 or window % tbptt_steps:
            raise ValueError("Estimator window must divide into TBPTT chunks and be at least one PPO rollout.")
        self.window = window
        self.tbptt_steps = tbptt_steps
        self.capacity = window + ppo_steps
        self.step = 0
        self.data = TensorDict(
            {key: value.new_zeros((self.capacity, *value.shape)) for key, value in template.items()},
            batch_size=[self.capacity, *template.batch_size],
        )
        self.dones = torch.zeros(self.capacity, template.batch_size[0], dtype=torch.bool, device=initial_state.device)
        self.states = initial_state.new_zeros(((self.capacity - 1) // tbptt_steps + 1, *initial_state.shape))

    def record_inputs(self, inputs: TensorDict, state: torch.Tensor) -> None:
        """Snapshot inputs before stepping the environment."""
        if self.step >= self.capacity:
            raise OverflowError("Estimator must be updated at the first PPO boundary after its window fills.")
        self.data[self.step].copy_(inputs)
        if self.step % self.tbptt_steps == 0:
            self.states[self.step // self.tbptt_steps].copy_(state.detach())

    def finish_step(self, dones: torch.Tensor) -> None:
        """Complete the pending sample using its post-action done flags."""
        self.dones[self.step].copy_(dones.flatten().bool())
        self.step += 1

    def reset_before(
        self,
        start: int,
        stop: int,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return resets before each input; the initial state already handles index zero."""
        if env_ids is None:
            env_ids = slice(None)
        resets = torch.zeros_like(self.dones[start:stop, env_ids])
        resets[1:] = self.dones[start : stop - 1, env_ids]
        return resets

    def batches(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Generator[tuple[TensorDict, torch.Tensor, torch.Tensor], None, None]:
        """Yield fixed-length sequences without episode-dependent padding."""
        num_envs = self.data.batch_size[1]
        if self.step < self.window:
            raise ValueError("Estimator window is not complete.")
        if not 1 <= num_mini_batches <= num_envs or num_envs % num_mini_batches:
            raise ValueError("Estimator mini-batches must evenly divide the per-rank environment count.")
        for _ in range(num_epochs):
            for ids in torch.randperm(num_envs, device=self.dones.device).chunk(num_mini_batches):
                chunks, resets, states = [], [], []
                for start in range(0, self.window, self.tbptt_steps):
                    stop = start + self.tbptt_steps
                    chunks.append(self.data[start:stop, ids])
                    resets.append(self.reset_before(start, stop, ids))
                    states.append(self.states[start // self.tbptt_steps, :, ids])
                yield torch.cat(chunks, dim=1), torch.cat(resets, dim=1), torch.cat(states, dim=1)

    def consume(self, tail_initial_state: torch.Tensor) -> None:
        """Carry every surplus sample into the next supervised window."""
        tail = self.step - self.window
        if tail:
            self.data[:tail].copy_(self.data[self.window : self.step].clone())
            self.dones[:tail].copy_(self.dones[self.window : self.step].clone())
            self.states[0].copy_(tail_initial_state)
        self.step = tail
