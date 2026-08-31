# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP-PPO with UniFP history adaptation supervision."""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.ppo_builders import construct_single_critic_algorithm
from rsl_rl.algorithms.unifp_adaptation_ppo import UniFPAdaptationMixin
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


class UniFPAMPAdaptationPPO(UniFPAdaptationMixin, AMPPPO):
    """Joint PPO, AMP-discriminator and UniFP auxiliary optimization."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        freeze_adaptation_after_iter: int | None = 4000,
        adaptation_learning_rate: float = 5.0e-6,
        obs_pred_group: str = "obs_pred",
        adaptation_weights: tuple[float, ...] = (0.6, 0.8, 1.0, 1.0),
        adaptation_part_names: tuple[str, ...] | None = None,
        adaptation_part_dims: tuple[int, ...] | None = None,
        adaptation_loss_types: tuple[str, ...] | None = None,
        reconstruction_obs_group: str | None = None,
        reconstruction_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self._init_adaptation(
            freeze_adaptation_after_iter=freeze_adaptation_after_iter,
            adaptation_learning_rate=adaptation_learning_rate,
            obs_pred_group=obs_pred_group,
            adaptation_weights=adaptation_weights,
            adaptation_part_names=adaptation_part_names,
            adaptation_part_dims=adaptation_part_dims,
            adaptation_loss_types=adaptation_loss_types,
            reconstruction_obs_group=reconstruction_obs_group,
            reconstruction_weight=reconstruction_weight,
        )

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
    ) -> "UniFPAMPAdaptationPPO":
        return construct_single_critic_algorithm(
            UniFPAMPAdaptationPPO,
            obs,
            env,
            cfg,
            device,
            include_amp_obs=True,
        )
