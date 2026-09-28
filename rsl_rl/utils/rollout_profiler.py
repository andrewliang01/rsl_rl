# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in synchronized rollout phase timings; disabled during normal training."""

import time
import torch


class RolloutProfiler:
    """Accumulate synchronized wall-clock durations for rollout phases."""

    phases = ("actor_critic", "env_step", "nan_check", "storage", "bookkeeping")

    def __init__(self, enabled: bool, device: str) -> None:
        """Create a disabled or synchronized profiler for a single rollout."""
        self.enabled = enabled
        self.device = device
        self.sums = dict.fromkeys(self.phases, 0.0)
        self.last = 0.0

    def mark(self, phase: str | None = None) -> None:
        """Finish one phase, or initialize the timing boundary."""
        if not self.enabled:
            return
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        if phase is not None:
            self.sums[phase] += now - self.last
        self.last = now

    def metrics(self, distributed: bool = False) -> dict[str, float]:
        """Return optional per-phase maxima across ranks for logging."""
        if not self.enabled:
            return {}
        values = torch.tensor(list(self.sums.values()), device=self.device)
        if distributed:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        return {f"profile_collection_{name}_s": value for name, value in zip(self.phases, values.tolist())}
