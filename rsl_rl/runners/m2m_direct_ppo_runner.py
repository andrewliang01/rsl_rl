# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""F14 runner with unambiguous completed-update resume semantics."""

from __future__ import annotations

from typing import Any

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


_PROGRESS_SCHEMA = "m2m_direct_ppo_runner_progress_v1"


class M2MDirectPpoRunner(OnPolicyRunner):
    """Run F14 to an absolute update target without replaying a saved update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._formal_training_io is not None:
            raise ValueError(
                "M2MDirectPpoRunner does not support formal-training v1; "
                "use ordinary F14 OnPolicy IO."
            )
        self._m2m_updates_completed = 0

    @staticmethod
    def _exact_nonnegative_int(value: object, *, field: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be an exact non-negative integer.")
        return value

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save zero-indexed iteration plus its exact completed-update count."""
        if self._formal_training_io is not None:
            raise ValueError("F14 direct PPO cannot use formal-training save.")
        iteration = self._exact_nonnegative_int(
            self.current_learning_iteration,
            field="current_learning_iteration",
        )
        completed = self._exact_nonnegative_int(
            getattr(self.alg, "completed_updates", None),
            field="algorithm.completed_updates",
        )
        if completed < 1 or completed != iteration + 1:
            raise ValueError(
                "F14 save is allowed only immediately after a completed update: "
                f"iter={iteration}, algorithm_updates_completed={completed}."
            )
        saved_dict = self.alg.save()
        if type(saved_dict) is not dict:
            raise TypeError("M2M direct PPO algorithm save must return an exact dictionary.")
        saved_dict["m2m_direct_ppo_progress"] = {
            "schema": _PROGRESS_SCHEMA,
            "iter": iteration,
            "updates_completed": completed,
            "resume_starts_at_update": completed,
        }
        saved_dict["iter"] = iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        """Restore F14 and position the next loop at the first unseen update."""
        if self._formal_training_io is not None:
            raise ValueError("F14 direct PPO cannot use formal-training load.")
        if load_cfg is not None:
            raise ValueError("F14 resume rejects partial load_cfg.")
        if strict is not True:
            raise ValueError("F14 resume requires strict=True.")
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if type(loaded_dict) is not dict:
            raise TypeError("F14 checkpoint root must be an exact dictionary.")
        progress = loaded_dict.get("m2m_direct_ppo_progress")
        if type(progress) is not dict:
            raise ValueError("F14 checkpoint lacks its direct-PPO progress receipt.")
        if progress.get("schema") != _PROGRESS_SCHEMA:
            raise ValueError("F14 checkpoint progress schema differs.")
        iteration = self._exact_nonnegative_int(loaded_dict.get("iter"), field="iter")
        receipt_iteration = self._exact_nonnegative_int(progress.get("iter"), field="progress.iter")
        completed = self._exact_nonnegative_int(
            progress.get("updates_completed"),
            field="progress.updates_completed",
        )
        next_update = self._exact_nonnegative_int(
            progress.get("resume_starts_at_update"),
            field="progress.resume_starts_at_update",
        )
        if receipt_iteration != iteration or completed != iteration + 1 or next_update != completed:
            raise ValueError("F14 checkpoint progress receipt is internally inconsistent.")
        if self.alg.load(loaded_dict, None, True) is not True:
            raise RuntimeError("M2M direct PPO algorithm refused iteration restoration.")
        if getattr(self.alg, "completed_updates", None) != completed:
            raise RuntimeError("Algorithm/runner completed-update receipts disagree after load.")
        self._m2m_updates_completed = completed
        # OnPolicyRunner's loop starts at current_learning_iteration.  Store
        # the first unseen update, not the last completed zero-based index.
        self.current_learning_iteration = next_update
        return loaded_dict.get("infos")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Interpret max_iterations as one absolute update target on resume."""
        target = self._exact_nonnegative_int(
            num_learning_iterations,
            field="num_learning_iterations",
        )
        completed = self._exact_nonnegative_int(
            self._m2m_updates_completed,
            field="updates_completed",
        )
        if getattr(self.alg, "completed_updates", None) != completed:
            raise RuntimeError("F14 runner and algorithm update counters disagree.")
        if completed > target:
            raise ValueError(
                f"F14 checkpoint already completed {completed} updates, beyond target {target}."
            )
        if completed == target:
            return
        self.current_learning_iteration = completed
        super().learn(
            num_learning_iterations=target - completed,
            init_at_random_ep_len=init_at_random_ep_len,
        )
        if getattr(self.alg, "completed_updates", None) != target:
            raise RuntimeError("F14 algorithm did not reach the requested absolute update target.")
        self._m2m_updates_completed = target

    def progress_audit(self) -> dict[str, Any]:
        """Return the runner's completed/next update convention."""
        return {
            "schema": _PROGRESS_SCHEMA,
            "updates_completed": self._m2m_updates_completed,
            "next_update": self._m2m_updates_completed,
            "checkpoint_iter": (
                self._m2m_updates_completed - 1
                if self._m2m_updates_completed > 0
                else None
            ),
            "absolute_target_semantics": True,
            "formal_training_v1_supported": False,
        }


__all__ = ["M2MDirectPpoRunner"]
