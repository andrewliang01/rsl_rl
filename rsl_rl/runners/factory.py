# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact runner factory shared by Isaac-facing entry points."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from rsl_rl.env import VecEnv

from .distillation_runner import DistillationRunner
from .m2m_distillation_runner import M2MDistillationRunner
from .on_policy_runner import OnPolicyRunner

_RUNNERS = MappingProxyType({
    "OnPolicyRunner": OnPolicyRunner,
    "DistillationRunner": DistillationRunner,
    "M2MDistillationRunner": M2MDistillationRunner,
})


def resolve_runner_class(class_name: str) -> type[OnPolicyRunner]:
    """Resolve only the three audited built-in runner names."""
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("runner class_name must be a non-empty string.")
    try:
        return _RUNNERS[class_name]
    except KeyError as error:
        raise ValueError(f"Unsupported runner class {class_name!r}; expected one of {sorted(_RUNNERS)}.") from error


def make_runner(
    class_name: str,
    env: VecEnv,
    train_cfg: dict[str, Any],
    *,
    log_dir: str | None = None,
    device: str = "cpu",
) -> OnPolicyRunner:
    """Construct one audited runner without algorithm-name side channels."""
    runner_class = resolve_runner_class(class_name)
    return runner_class(env, train_cfg, log_dir=log_dir, device=device)


__all__ = ["make_runner", "resolve_runner_class"]
