# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "swanlab is required for SwanLab logging. Install it with: python -m pip install swanlab"
    ) from None


class SwanLabSummaryWriter(SummaryWriter):
    """Write standard TensorBoard events locally and mirror scalars to SwanLab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs=flush_secs)
        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError("Please specify swanlab_project in the runner config or with --log_project_name.") from None

        workspace = os.environ.get("SWANLAB_WORKSPACE")
        self.run = swanlab.init(
            project=project,
            workspace=workspace,
            name=os.path.basename(log_dir),
            log_dir=log_dir,
            config={"log_dir": log_dir},
        )

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration."""
        if isinstance(env_cfg, dict):
            env_dict = env_cfg
        elif hasattr(env_cfg, "to_dict"):
            env_dict = env_cfg.to_dict()  # type: ignore
        else:
            env_dict = asdict(env_cfg)  # type: ignore[arg-type]
        self.run.config.update({"train_cfg": train_cfg, "env_cfg": env_dict})

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both the local TensorBoard file and SwanLab."""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        value = scalar_value.item() if hasattr(scalar_value, "item") else scalar_value
        self.run.log({tag: value}, step=global_step)

    def stop(self) -> None:
        """Flush TensorBoard and finish the active SwanLab run."""
        super().close()
        self.run.finish()

    def save_file(self, path: str) -> None:
        """Upload a small provenance file such as a git diff."""
        self.run.save(path, base_path=os.path.dirname(path), policy="now")
