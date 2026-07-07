# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


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

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

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
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

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
