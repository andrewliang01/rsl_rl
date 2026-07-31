# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import os
import time
import torch
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.formal_training_io import FormalTrainingIO, FormalTrainingIOError
from rsl_rl.utils.logger import Logger
from rsl_rl.utils.training_receipt import (
    build_embedded_checkpoint_receipt,
    canonical_training_receipt_json_bytes,
    derive_checkpoint_progress,
    validate_embedded_checkpoint_receipt,
    validate_training_launch_receipt,
)


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_cfg = self.cfg["algorithm"]
        self._validate_explicit_ppo_variant(alg_cfg)
        alg_class: type[PPO] = resolve_callable(alg_cfg["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self._formal_training_io: FormalTrainingIO | None = None
        self._formal_launch_receipt: dict[str, Any] | None = None
        self._formal_launch_receipt_bytes: bytes | None = None
        self._formal_schedule: Mapping[str, Any] | None = None
        self._formal_parent_checkpoint: dict[str, Any] | None = None
        self._formal_last_local_embedded_receipt: dict[str, Any] | None = None
        self._formal_updates_completed = 0
        self._formal_resume_loaded = False

    @staticmethod
    def _formal_exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise FormalTrainingIOError(
                f"{field} must be an exact integer >= {minimum}."
            )
        return value

    @staticmethod
    def _formal_env_seed(env_cfg: object) -> Any:
        if isinstance(env_cfg, dict):
            return env_cfg.get("seed")
        return getattr(env_cfg, "seed", None)

    @staticmethod
    def _formal_finite_float(value: Any, *, field: str) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise FormalTrainingIOError(
                f"{field} must be an exact finite float."
            )
        return value

    @classmethod
    def _formal_optimizer_lrs_from_state(
        cls,
        saved_dict: Mapping[str, Any],
    ) -> tuple[float, ...]:
        optimizer_state = saved_dict.get("optimizer_state_dict")
        if type(optimizer_state) is not dict:
            raise FormalTrainingIOError(
                "Formal checkpoint requires an exact optimizer_state_dict."
            )
        param_groups = optimizer_state.get("param_groups")
        if type(param_groups) is not list or not param_groups:
            raise FormalTrainingIOError(
                "Formal optimizer state requires non-empty param_groups."
            )
        learning_rates: list[float] = []
        for index, group in enumerate(param_groups):
            if type(group) is not dict or "lr" not in group:
                raise FormalTrainingIOError(
                    "Formal optimizer state param group lacks an exact LR."
                )
            learning_rates.append(
                cls._formal_finite_float(
                    group["lr"],
                    field=f"optimizer_state_dict.param_groups[{index}].lr",
                )
            )
        if any(rate != learning_rates[0] for rate in learning_rates[1:]):
            raise FormalTrainingIOError(
                "Formal optimizer param-group learning rates disagree."
            )
        return tuple(learning_rates)

    def _formal_live_optimizer_lrs(self) -> tuple[float, ...]:
        optimizer = getattr(self.alg, "optimizer", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if type(param_groups) is not list or not param_groups:
            raise FormalTrainingIOError(
                "Formal PPO requires an optimizer with non-empty param_groups."
            )
        learning_rates = tuple(
            self._formal_finite_float(
                group.get("lr") if type(group) is dict else None,
                field=f"algorithm.optimizer.param_groups[{index}].lr",
            )
            for index, group in enumerate(param_groups)
        )
        if any(rate != learning_rates[0] for rate in learning_rates[1:]):
            raise FormalTrainingIOError(
                "Live optimizer param-group learning rates disagree."
            )
        return learning_rates

    def _assert_formal_optimizer_live_consistency(self) -> None:
        live_rates = self._formal_live_optimizer_lrs()
        algorithm_rate = self._formal_finite_float(
            getattr(self.alg, "learning_rate", None),
            field="algorithm.learning_rate",
        )
        if any(rate != algorithm_rate for rate in live_rates):
            raise FormalTrainingIOError(
                "Algorithm learning_rate differs from its optimizer state."
            )

    def _validate_formal_optimizer_save_state(
        self,
        saved_dict: Mapping[str, Any],
    ) -> None:
        saved_rates = self._formal_optimizer_lrs_from_state(saved_dict)
        live_rates = self._formal_live_optimizer_lrs()
        algorithm_rate = self._formal_finite_float(
            getattr(self.alg, "learning_rate", None),
            field="algorithm.learning_rate",
        )
        if saved_rates != live_rates or any(
            rate != algorithm_rate for rate in saved_rates
        ):
            raise FormalTrainingIOError(
                "Saved, live, and algorithm learning rates disagree."
            )

    def _restore_formal_optimizer_learning_rate(
        self,
        loaded_dict: Mapping[str, Any],
    ) -> None:
        """Restore adaptive PPO's Python LR field from optimizer state."""
        saved_rates = self._formal_optimizer_lrs_from_state(loaded_dict)
        live_rates = self._formal_live_optimizer_lrs()
        if live_rates != saved_rates:
            raise FormalTrainingIOError(
                "Algorithm load did not restore the exact optimizer LR."
            )
        self.alg.learning_rate = saved_rates[0]
        self._assert_formal_optimizer_live_consistency()

    def _assert_formal_supported_algorithm(self) -> None:
        alg_cfg = self.cfg.get("algorithm")
        if type(alg_cfg) is not dict:
            raise FormalTrainingIOError(
                "Formal training requires an exact algorithm configuration."
            )
        class_name = alg_cfg.get("class_name")
        if type(class_name) is not str:
            raise FormalTrainingIOError(
                "Formal training requires an explicit PPO class_name."
            )
        short_name = class_name.replace(":", ".").rsplit(".", 1)[-1]
        if short_name not in {"PPO", "MultiPPO"}:
            raise FormalTrainingIOError(
                "Formal training v1 supports only PPO or MultiPPO."
            )
        unsupported = [
            key
            for key in ("rnd_cfg", "dwaq_cfg", "amp_cfg")
            if alg_cfg.get(key) is not None
        ]
        live_unsupported = [
            name
            for name in ("rnd", "dwaq", "amp_discriminator")
            if getattr(self.alg, name, None) is not None
        ]
        if unsupported or live_unsupported:
            names = ", ".join([*unsupported, *live_unsupported])
            raise FormalTrainingIOError(
                "Formal training v1 rejects extensions with unproven "
                f"complete resume state: {names}."
            )
        self._assert_formal_optimizer_live_consistency()

    def _assert_formal_configuration_unchanged(self) -> None:
        launch = self._formal_launch_receipt
        launch_bytes = self._formal_launch_receipt_bytes
        schedule = self._formal_schedule
        formal_io = self._formal_training_io
        if (
            launch is None
            or launch_bytes is None
            or schedule is None
            or formal_io is None
        ):
            raise FormalTrainingIOError(
                "Formal training configuration was not frozen."
            )
        validated_launch = validate_training_launch_receipt(launch)
        if (
            canonical_training_receipt_json_bytes(validated_launch)
            != launch_bytes
        ):
            raise FormalTrainingIOError(
                "Formal launch receipt changed after configuration."
            )
        actual = (
            self._formal_exact_int(
                self.env.num_envs,
                field="env.num_envs",
                minimum=1,
            ),
            self._formal_exact_int(
                self.cfg.get("num_steps_per_env"),
                field="cfg.num_steps_per_env",
                minimum=1,
            ),
            self._formal_exact_int(
                self.cfg.get("max_iterations"),
                field="cfg.max_iterations",
                minimum=1,
            ),
            self._formal_exact_int(
                self.cfg.get("save_interval"),
                field="cfg.save_interval",
                minimum=1,
            ),
            self._formal_exact_int(
                self.cfg.get("seed"),
                field="cfg.seed",
            ),
            self._formal_exact_int(
                self._formal_env_seed(self.env.cfg),
                field="env.cfg.seed",
            ),
        )
        expected = (
            schedule["num_envs"],
            schedule["num_steps_per_env"],
            schedule["max_iterations"],
            schedule["save_interval"],
            schedule["seed"],
            schedule["seed"],
        )
        if actual != expected:
            raise FormalTrainingIOError(
                "Formal runner/env schedule or seed changed after configuration."
            )
        if (
            schedule["transitions_per_update"]
            != schedule["num_envs"] * schedule["num_steps_per_env"]
            or schedule["transition_budget"]
            != schedule["transitions_per_update"]
            * schedule["max_iterations"]
            or self.device != schedule["device"]
            or Path(self.logger.log_dir) != Path(schedule["run_dir"])
            or self.is_distributed is not False
            or self.gpu_world_size != 1
            or self.gpu_global_rank != 0
        ):
            raise FormalTrainingIOError(
                "Formal runtime topology changed after configuration."
            )
        if self.device.startswith("cuda"):
            if (
                os.environ.get("CUDA_VISIBLE_DEVICES")
                != schedule["cuda_visible_devices"]
                or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1
            ):
                raise FormalTrainingIOError(
                    "Formal CUDA binding changed after configuration."
                )
        self._assert_formal_supported_algorithm()

    def configure_formal_training(self, context: dict[str, Any]) -> None:
        """Explicitly enable fail-closed formal checkpoint semantics.

        ``context`` has exactly two keys: a validated launch receipt and the
        absolute run directory that must also be used by the logger.
        """
        if self._formal_training_io is not None:
            raise FormalTrainingIOError(
                "Formal training is already configured for this runner."
            )
        if type(context) is not dict or set(context) != {
            "launch_receipt",
            "run_dir",
        }:
            raise FormalTrainingIOError(
                "Formal training context must contain exactly "
                "launch_receipt and run_dir."
            )
        if (
            self.is_distributed is not False
            or type(self.gpu_world_size) is not int
            or self.gpu_world_size != 1
            or type(self.gpu_global_rank) is not int
            or self.gpu_global_rank != 0
        ):
            raise FormalTrainingIOError(
                "Formal training v1 requires one non-distributed process."
            )
        launch = validate_training_launch_receipt(
            context["launch_receipt"]
        )
        schedule = launch["payload"]["schedule"]
        num_envs = self._formal_exact_int(
            self.env.num_envs,
            field="env.num_envs",
            minimum=1,
        )
        steps = self._formal_exact_int(
            self.cfg.get("num_steps_per_env"),
            field="cfg.num_steps_per_env",
            minimum=1,
        )
        maximum = self._formal_exact_int(
            self.cfg.get("max_iterations"),
            field="cfg.max_iterations",
            minimum=1,
        )
        save_interval = self._formal_exact_int(
            self.cfg.get("save_interval"),
            field="cfg.save_interval",
            minimum=1,
        )
        actual_schedule = (
            num_envs,
            steps,
            maximum,
            save_interval,
            num_envs * steps,
            num_envs * steps * maximum,
        )
        receipt_schedule = (
            schedule["num_envs"],
            schedule["num_steps_per_env"],
            schedule["max_iterations"],
            schedule["save_interval"],
            schedule["transitions_per_update"],
            schedule["transition_budget"],
        )
        if actual_schedule != receipt_schedule:
            raise FormalTrainingIOError(
                "Formal launch schedule differs from runner/env configuration."
            )
        seed = self._formal_exact_int(
            self.cfg.get("seed"),
            field="cfg.seed",
        )
        env_seed = self._formal_exact_int(
            self._formal_env_seed(self.env.cfg),
            field="env.cfg.seed",
        )
        if seed != launch["payload"]["seed"] or env_seed != seed:
            raise FormalTrainingIOError(
                "Formal launch seed differs from runner or environment seed."
            )
        runtime = launch["payload"]["runtime"]
        if runtime["device"] != self.device:
            raise FormalTrainingIOError(
                "Formal launch runtime device differs from runner device."
            )
        if self.device.startswith("cuda"):
            visible_devices = runtime["cuda"]["cuda_visible_devices"]
            actual_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if (
                self.device != "cuda:0"
                or actual_visible_devices is None
                or visible_devices != actual_visible_devices
                or len(visible_devices.split(",")) != 1
                or not visible_devices.strip()
                or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1
            ):
                raise FormalTrainingIOError(
                    "Formal CUDA training requires exactly one visible GPU "
                    "addressed as cuda:0."
                )
        try:
            run_dir = Path(context["run_dir"])
        except TypeError as error:
            raise FormalTrainingIOError(
                "Formal run_dir must be a path-like value."
            ) from error
        if self.logger.log_dir is None or Path(self.logger.log_dir) != run_dir:
            raise FormalTrainingIOError(
                "Formal run_dir must exactly equal the logger log_dir."
            )
        check_for_nan = self.cfg.get("check_for_nan", True)
        if type(check_for_nan) is not bool:
            raise FormalTrainingIOError(
                "cfg.check_for_nan must be an exact boolean."
            )
        self._assert_formal_supported_algorithm()
        formal_io = FormalTrainingIO(
            run_dir=run_dir,
            launch_receipt=launch,
        )
        # Mark the runner as formal before any recovery can mutate algorithm
        # state.  A failed recovery therefore remains fail-closed instead of
        # silently falling back to the legacy branch.
        self._formal_training_io = formal_io
        self._formal_launch_receipt = launch
        self._formal_launch_receipt_bytes = (
            canonical_training_receipt_json_bytes(launch)
        )
        self._formal_schedule = MappingProxyType(
            {
                "num_envs": schedule["num_envs"],
                "num_steps_per_env": schedule["num_steps_per_env"],
                "max_iterations": schedule["max_iterations"],
                "save_interval": schedule["save_interval"],
                "transitions_per_update": schedule[
                    "transitions_per_update"
                ],
                "transition_budget": schedule["transition_budget"],
                "seed": seed,
                "device": self.device,
                "run_dir": str(run_dir),
                "cuda_visible_devices": runtime["cuda"][
                    "cuda_visible_devices"
                ],
                "check_for_nan": check_for_nan,
            }
        )
        self._formal_parent_checkpoint = None
        self._formal_last_local_embedded_receipt = None
        self._formal_updates_completed = 0
        self._formal_resume_loaded = not launch["payload"]["resume"][
            "is_resume"
        ]
        recovery_publication: dict[str, Any] | None = None
        try:
            recovery = formal_io.recover_interrupted_checkpoint(
                map_location=self.device,
            )
            if recovery is not None:
                self._formal_optimizer_lrs_from_state(
                    recovery["loaded_dict"]
                )
                load_iteration = self.alg.load(
                    recovery["loaded_dict"],
                    None,
                    True,
                )
                if load_iteration is not True:
                    raise FormalTrainingIOError(
                        "Algorithm refused recovered formal iteration state."
                    )
                self._restore_formal_optimizer_learning_rate(
                    recovery["loaded_dict"]
                )
                recovery_publication = formal_io.commit_recovery(recovery)
        except Exception:
            formal_io.close()
            raise
        if recovery is not None:
            assert recovery_publication is not None
            embedded = recovery["embedded_receipt"]
            progress = embedded["checkpoint_progress"]
            self._formal_parent_checkpoint = recovery_publication[
                "parent_record"
            ]
            self._formal_last_local_embedded_receipt = embedded
            self._formal_updates_completed = progress["updates_completed"]
            self.current_learning_iteration = progress["iter"]
            self._formal_resume_loaded = True

    @property
    def formal_external_parent_load_required(self) -> bool:
        """Return whether the bound external parent must still be loaded.

        Formal configuration attempts same-run local recovery before this
        property can return.  A resumed launch therefore returns ``True`` only
        when no local checkpoint was recovered and the algorithm is still in
        its unmodified, pre-parent state.
        """
        formal_io = self._formal_training_io
        launch = self._formal_launch_receipt
        if formal_io is None or launch is None:
            raise FormalTrainingIOError(
                "Formal external-parent requirement needs a configured runner."
            )
        if not formal_io.lock_held:
            raise FormalTrainingIOError(
                "Formal external-parent requirement needs a held run lock."
            )
        if type(self._formal_resume_loaded) is not bool:
            raise FormalTrainingIOError(
                "Formal external-parent requirement state is inconsistent."
            )
        resume = launch["payload"]["resume"]
        if self._formal_resume_loaded:
            if (
                resume["is_resume"] is True
                and (
                    self._formal_parent_checkpoint is None
                    or type(self._formal_updates_completed) is not int
                    or self._formal_updates_completed
                    < resume["parent_updates_completed"]
                )
            ):
                raise FormalTrainingIOError(
                    "Formal external-parent requirement state is inconsistent."
                )
            return False
        if (
            resume["is_resume"] is not True
            or type(self._formal_updates_completed) is not int
            or self._formal_updates_completed != 0
            or type(self.current_learning_iteration) is not int
            or self.current_learning_iteration != 0
            or self._formal_parent_checkpoint is not None
            or self._formal_last_local_embedded_receipt is not None
        ):
            raise FormalTrainingIOError(
                "Formal external-parent requirement state is inconsistent."
            )
        return True

    def close_formal_training(self) -> None:
        """Release the formal run lock, if configured."""
        formal_io = self._formal_training_io
        if formal_io is not None:
            formal_io.close()

    def _validate_explicit_ppo_variant(self, alg_cfg: dict) -> None:
        """Catch PPO-family configs that need an explicit algorithm class."""
        class_name = alg_cfg.get("class_name", "")
        short_name = class_name.replace(":", ".").rsplit(".", maxsplit=1)[-1]
        uses_amp = alg_cfg.get("amp_cfg") is not None
        uses_multi_critic = alg_cfg.get("num_critics", 1) > 1

        if short_name == "PPO" and uses_amp:
            raise ValueError('PPO config has amp_cfg. Set algorithm.class_name="AMPPPO" explicitly.')
        if short_name == "PPO" and uses_multi_critic:
            raise ValueError('PPO config has num_critics > 1. Set algorithm.class_name="MultiPPO" explicitly.')
        if short_name == "MultiPPO" and uses_amp:
            raise ValueError('MultiPPO config has amp_cfg. Use algorithm.class_name="AMPPPO" for AMP training.')
        if short_name == "AMPPPO" and not uses_amp:
            raise ValueError('AMPPPO config requires algorithm.amp_cfg.')

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        if self._formal_training_io is not None:
            self._learn_formal(
                target_updates=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
            return

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"[DIST] rank {self.gpu_global_rank} entering parameter sync...", flush=True)
            self.alg.broadcast_parameters()
            print(f"[DIST] rank {self.gpu_global_rank} finished parameter sync.", flush=True)

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode(): #禁用梯度计算相关的开销，从而提升性能。
                for _ in range(self.cfg["num_steps_per_env"]): 
                # 每一次的学习迭代中，算法会在环境中执行cfg["num_steps_per_env"]步。
                # 这意味着智能体将与环境进行多次交互，收集状态、奖励和其他相关信息，以用于后续的学习更新。
                # 那为什么是环境收集24步信息，算法才优化迭代一次？
                    
                    # actor和critic的推理，更新分布参数（有了新的action的分布），存储obs、actions和values，返回actions
                    actions = self.alg.act(obs)
                    
                    # 给定actions，环境去交互，得到obs、rewards、dones和extras
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # 检查环境返回的有没有异常值（NaN），如果有则抛出错误。这是为了确保训练过程中的数值稳定性和正确性。
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    
                    # 转移到GPU上进行后续处理
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    # 基于最新的obs做归一化处理，记录reward、dones，self.storage.add_transition(self.transition)
                    # 对time_outs的情况做特殊处理，因为没有下一步的$V(s_{t+1})$, 得换成 $V(s_t)$
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Extract intrinsic rewards if RND is used (only for logging)
                    # TODO： RND没学过不懂啊，下次学了再来看吧
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    # Handle multi-critic rewards: use the same group weights as advantage aggregation for logging.
                    if rewards.dim() > 1 and rewards.shape[-1] > 1:
                        if hasattr(self.alg, "reward_group_weights"):
                            weights = self.alg.reward_group_weights.to(device=rewards.device, dtype=rewards.dtype)
                            rewards_for_log = torch.sum(rewards * weights.view(1, -1), dim=-1)
                        else:
                            rewards_for_log = rewards.sum(dim=-1)
                    else:
                        rewards_for_log = rewards.squeeze(-1) if rewards.dim() > 1 else rewards

                    # 记录AMP的奖励啊
                    amp_rewards = None
                    if hasattr(self.alg, 'amp_discriminator') and self.alg.amp_discriminator is not None:
                        amp_rewards = {
                            'task': getattr(self.alg, 'task_rewards', None),
                            'style': getattr(self.alg, 'style_rewards', None),
                            'final': getattr(self.alg, 'final_rewards', None),
                        }

                    self.logger.process_env_step(rewards_for_log, dones, extras, intrinsic_rewards, amp_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                # 用优势函数估计每一步的return
                self.alg.compute_returns(obs)

            # Update policy
            # TODO： 时序网络的recurrent，MLP 路径下 24 步和 4096 envs 是全部拍扁打乱在一起切的，不是"24 步之间切"——这是 MLP 不依赖时序带来的便利，而 RNN 只能沿 env 切来保留时序完整性。
            # TODO： symmetry我同样不太懂，先不管了。总之就是在切数据的时候要保证对称性的数据在一起被切到同一个batch里。

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def _learn_formal(
        self,
        *,
        target_updates: int,
        init_at_random_ep_len: bool,
    ) -> None:
        """Run to one absolute update target under formal save semantics."""
        formal_io = self._formal_training_io
        launch = self._formal_launch_receipt
        schedule = self._formal_schedule
        if formal_io is None:
            raise FormalTrainingIOError(
                "Formal training is not configured with a held run lock."
            )
        if launch is None or schedule is None or not formal_io.lock_held:
            formal_io.close()
            raise FormalTrainingIOError(
                "Formal training is not configured with a held run lock."
            )
        try:
            self._assert_formal_configuration_unchanged()
            target_updates = self._formal_exact_int(
                target_updates,
                field="formal target_updates",
                minimum=1,
            )
            configured_target = schedule["max_iterations"]
            if target_updates != configured_target:
                raise FormalTrainingIOError(
                    "Formal learn argument is an absolute target and must equal "
                    "the launch receipt max_iterations."
                )
            if not self._formal_resume_loaded:
                raise FormalTrainingIOError(
                    "Formal resume launch must load its exact parent before learn."
                )
            if self._formal_updates_completed > target_updates:
                raise FormalTrainingIOError(
                    "Formal completed updates exceed the configured target."
                )
        except Exception:
            formal_io.close()
            raise
        try:
            if self._formal_updates_completed == target_updates:
                self.save(
                    os.path.join(
                        formal_io.run_dir,
                        f"model_{self.current_learning_iteration}.pt",
                    )
                )
                return
            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf,
                    high=int(self.env.max_episode_length),
                )

            obs = self.env.get_observations().to(self.device)
            self.alg.train_mode()
            self.logger.init_logging_writer()

            start_it = self._formal_updates_completed
            total_it = target_updates
            for it in range(start_it, total_it):
                self._assert_formal_configuration_unchanged()
                start = time.time()
                with torch.inference_mode():
                    for _ in range(schedule["num_steps_per_env"]):
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(
                            actions.to(self.env.device)
                        )
                        if schedule["check_for_nan"]:
                            check_nan(obs, rewards, dones)
                        obs, rewards, dones = (
                            obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        self.alg.process_env_step(
                            obs,
                            rewards,
                            dones,
                            extras,
                        )
                        intrinsic_rewards = None
                        if rewards.dim() > 1 and rewards.shape[-1] > 1:
                            if hasattr(self.alg, "reward_group_weights"):
                                weights = self.alg.reward_group_weights.to(
                                    device=rewards.device,
                                    dtype=rewards.dtype,
                                )
                                rewards_for_log = torch.sum(
                                    rewards * weights.view(1, -1),
                                    dim=-1,
                                )
                            else:
                                rewards_for_log = rewards.sum(dim=-1)
                        else:
                            rewards_for_log = (
                                rewards.squeeze(-1)
                                if rewards.dim() > 1
                                else rewards
                            )
                        amp_rewards = None
                        if (
                            hasattr(self.alg, "amp_discriminator")
                            and self.alg.amp_discriminator is not None
                        ):
                            amp_rewards = {
                                "task": getattr(
                                    self.alg,
                                    "task_rewards",
                                    None,
                                ),
                                "style": getattr(
                                    self.alg,
                                    "style_rewards",
                                    None,
                                ),
                                "final": getattr(
                                    self.alg,
                                    "final_rewards",
                                    None,
                                ),
                            }
                        self.logger.process_env_step(
                            rewards_for_log,
                            dones,
                            extras,
                            intrinsic_rewards,
                            amp_rewards,
                        )

                    stop = time.time()
                    collect_time = stop - start
                    start = stop
                    self.alg.compute_returns(obs)

                loss_dict = self.alg.update()
                self._assert_formal_optimizer_live_consistency()
                stop = time.time()
                learn_time = stop - start

                # Advance formal progress only after one successful optimizer
                # update.  The zero-indexed checkpoint name is therefore
                # updates_completed - 1.
                self._formal_updates_completed = it + 1
                self.current_learning_iteration = it

                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collect_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=None,
                )

                if it % schedule["save_interval"] == 0:
                    self.save(
                        os.path.join(
                            formal_io.run_dir,
                            f"model_{it}.pt",
                        )
                    )

            self.save(
                os.path.join(
                    formal_io.run_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            if self.logger.writer is not None:
                self.logger.stop_logging_writer()
        finally:
            formal_io.close()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        if self._formal_training_io is not None:
            try:
                self._save_formal(path, infos=infos)
            except Exception:
                self._formal_training_io.close()
                raise
            return

        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def _save_formal(self, path: str, *, infos: dict | None) -> None:
        formal_io = self._formal_training_io
        launch = self._formal_launch_receipt
        if formal_io is None or launch is None or not formal_io.lock_held:
            raise FormalTrainingIOError(
                "Formal save requires a configured runner with a held lock."
            )
        self._assert_formal_configuration_unchanged()
        schedule = self._formal_schedule
        assert schedule is not None
        if not self._formal_resume_loaded:
            raise FormalTrainingIOError(
                "Formal resume parent has not been loaded."
            )
        completed = self._formal_exact_int(
            self._formal_updates_completed,
            field="formal updates_completed",
            minimum=1,
        )
        iteration = completed - 1
        if self.current_learning_iteration != iteration:
            raise FormalTrainingIOError(
                "Runner iteration differs from completed formal updates."
            )
        progress = derive_checkpoint_progress(
            filename=f"model_{iteration}.pt",
            iteration=iteration,
            num_envs=schedule["num_envs"],
            num_steps_per_env=schedule["num_steps_per_env"],
            configured_target_updates=schedule["max_iterations"],
        )
        latest_parent = self._formal_parent_checkpoint
        if (
            latest_parent is not None
            and latest_parent["updates_completed"] == completed
        ):
            embedded = self._formal_last_local_embedded_receipt
            if embedded is None:
                raise FormalTrainingIOError(
                    "Formal save cannot republish an external resume parent."
                )
            embedded = validate_embedded_checkpoint_receipt(
                embedded,
                checkpoint_filename=f"model_{iteration}.pt",
            )
            if embedded["checkpoint_progress"] != progress:
                raise FormalTrainingIOError(
                    "Repeated formal save progress differs from local receipt."
                )
        else:
            embedded = build_embedded_checkpoint_receipt(
                launch_receipt=launch,
                checkpoint_progress=progress,
                parent_checkpoint=latest_parent,
            )

        def saved_dict_factory() -> dict[str, Any]:
            saved_dict = self.alg.save()
            if type(saved_dict) is not dict:
                raise FormalTrainingIOError(
                    "Algorithm save must return an exact dictionary."
                )
            self._validate_formal_optimizer_save_state(saved_dict)
            result = dict(saved_dict)
            result["iter"] = iteration
            result["infos"] = infos
            result["training_receipt"] = embedded
            return result

        publication = formal_io.publish_checkpoint(
            checkpoint_path=path,
            embedded_receipt=embedded,
            saved_dict_factory=saved_dict_factory,
        )
        self._formal_parent_checkpoint = publication["parent_record"]
        self._formal_last_local_embedded_receipt = publication[
            "embedded_receipt"
        ]
        if publication["status"] == "committed":
            self.logger.save_model(path, iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        if self._formal_training_io is not None:
            return self._load_formal(
                path,
                load_cfg=load_cfg,
                strict=strict,
                map_location=map_location,
            )

        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def _load_formal(
        self,
        path: str,
        *,
        load_cfg: dict | None,
        strict: bool,
        map_location: str | None,
    ) -> dict | None:
        formal_io = self._formal_training_io
        if formal_io is None or not formal_io.lock_held:
            raise FormalTrainingIOError(
                "Formal load requires a configured runner with a held lock."
            )
        if load_cfg is not None:
            formal_io.close()
            raise FormalTrainingIOError(
                "Formal resume forbids partial load_cfg."
            )
        if strict is not True:
            formal_io.close()
            raise FormalTrainingIOError(
                "Formal resume requires strict=True."
            )
        if map_location is not None and (
            type(map_location) is not str or map_location != self.device
        ):
            formal_io.close()
            raise FormalTrainingIOError(
                "Formal resume map_location must be None or the configured device."
            )
        if self._formal_resume_loaded:
            formal_io.close()
            raise FormalTrainingIOError(
                "Formal resume checkpoint was already loaded."
            )
        try:
            self._assert_formal_configuration_unchanged()
            loaded_dict, parent = formal_io.load_resume_checkpoint(
                path,
                map_location=self.device,
            )
            embedded = validate_embedded_checkpoint_receipt(
                loaded_dict["training_receipt"],
                checkpoint_filename=Path(path).name,
            )
            if embedded["checkpoint_progress"]["updates_completed"] != parent[
                "updates_completed"
            ]:
                raise FormalTrainingIOError(
                    "Loaded checkpoint progress differs from parent head."
                )
            self._formal_optimizer_lrs_from_state(loaded_dict)
            load_iteration = self.alg.load(
                loaded_dict,
                None,
                True,
            )
            if load_iteration is not True:
                raise FormalTrainingIOError(
                    "Algorithm refused to restore formal iteration state."
                )
            self._restore_formal_optimizer_learning_rate(loaded_dict)
            self._formal_parent_checkpoint = parent
            self._formal_updates_completed = parent["updates_completed"]
            self.current_learning_iteration = (
                self._formal_updates_completed - 1
            )
            self._formal_resume_loaded = True
            return loaded_dict.get("infos")
        except Exception:
            formal_io.close()
            raise

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
        input_mode: str = "split",
    ) -> None:
        """Export the model into an ONNX file."""
        try:
            onnx_model = self.alg.get_policy().as_onnx(verbose=verbose, input_mode=input_mode)
        except TypeError:
            onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            external_data=False,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
            dynamic_axes=getattr(onnx_model, "dynamic_axes", None),
        )
        self._rewrite_onnx_ir_version(save_path)

    @staticmethod
    def _rewrite_onnx_ir_version(save_path: str, target_ir_version: int = 8) -> None:
        """Rewrite ONNX IR version for older deployment runtimes."""
        try:
            import onnx
        except ImportError:
            print("[WARNING]: Could not rewrite ONNX IR version because onnx is not installed.")
            return

        model = onnx.load(save_path)
        if model.ir_version <= target_ir_version:
            return

        original_ir_version = model.ir_version
        model.ir_version = target_ir_version
        onnx.save(model, save_path, save_as_external_data=False)
        print(
            f"[INFO]: Rewrote ONNX IR version from {original_ir_version} to {target_ir_version}: {save_path}"
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
        print(
            f"[DIST] init_process_group ready: "
            f"rank={self.gpu_global_rank}, local_rank={self.gpu_local_rank}, "
            f"world_size={self.gpu_world_size}, device={self.device}",
            flush=True,
        )
