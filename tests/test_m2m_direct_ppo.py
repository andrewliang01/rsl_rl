# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import M2MDirectPPO
from rsl_rl.models import PropMLPElevationFusionModel
from rsl_rl.runners import M2MDirectPpoRunner, resolve_runner_class


_NUM_ENVS = 3
_NUM_STEPS = 4
_ACTION_DIM = 29
_STRICT_SET = "m2m_student_current_frame"
_M90_SET = "height_scan_policy"
_OBS_GROUPS = {
    "actor": ["policy", _STRICT_SET],
    "critic": ["critic", "height_scan_critic"],
}


def _frozen_actor_cfg() -> dict[str, Any]:
    return {
        "hidden_dims": [32, 16],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.7,
            "std_type": "scalar",
        },
        "elevation_set": _M90_SET,
        "cnn_observation_type": "depthcamera",
        "depth_camera_near": 0.05,
        "depth_camera_far": 6.0,
        "vision_spatial_size": (16, 96),
        "vision_feature_dim": 64,
        "elevation_history_length": 1,
        "cnn_hidden_dims": [4, 8],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [16],
        "use_prop_encoder": True,
    }


def _make_frozen_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    torch.manual_seed(101)
    obs = TensorDict(
        {
            "policy": torch.randn(2, 96),
            _M90_SET: torch.rand(2, 1, 16, 96) + 0.05,
        },
        batch_size=[2],
    )
    actor = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups={"actor": ["policy", _M90_SET]},
        obs_set="actor",
        output_dim=_ACTION_DIM,
        **_frozen_actor_cfg(),
    )
    path = tmp_path / "synthetic_m90.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "iter": 71}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_frame(batch_size: int, *, step: int) -> torch.Tensor:
    frame = torch.zeros(batch_size, 1, 4, 16, 96)
    ranges = torch.linspace(0.1, 5.9, 16 * 96).reshape(1, 16, 96)
    frame[:, 0, 0] = (ranges + 0.01 * step).clamp(0.1, 6.0)
    frame[:, 0, 1] = 1.0
    frame[:, 0, 2] = float(step % 5) * 0.02
    frame[:, 0, 3] = float(step % 5 == 0)
    return frame


def _observations(*, step: int) -> TensorDict:
    generator = torch.Generator(device="cpu").manual_seed(200 + step)
    return TensorDict(
        {
            "policy": torch.randn(_NUM_ENVS, 96, generator=generator),
            _STRICT_SET: _strict_frame(_NUM_ENVS, step=step),
            "critic": torch.randn(_NUM_ENVS, 99, generator=generator),
            "height_scan_critic": torch.randn(
                _NUM_ENVS, 1, 28, 20, generator=generator
            ),
        },
        batch_size=[_NUM_ENVS],
    )


def _runner_cfg(
    checkpoint: Path,
    digest: str,
    *,
    latent_hidden_dim: int = 24,
) -> dict[str, Any]:
    return {
        "num_steps_per_env": _NUM_STEPS,
        "obs_groups": copy.deepcopy(_OBS_GROUPS),
        "actor": {
            "class_name": "M2MMapFreeRecurrentStudent",
            "strict_frame_set": _STRICT_SET,
            "proprio_sets": ["policy"],
            "frozen_ecmm_checkpoint_path": str(checkpoint),
            "frozen_ecmm_expected_sha256": digest,
            "frozen_ecmm_actor_state_dict_key": "actor_state_dict",
            "frozen_ecmm_actor_cfg": _frozen_actor_cfg(),
            "frame_near_range_m": 0.1,
            "frame_far_range_m": 6.0,
            "frame_message_period_s": 0.1,
            "frame_max_age_s": 0.5,
            "frame_age_semantics": "uniform_message_age_s",
            "tokenizer_hidden_channels": [4, 8],
            "tokenizer_dim": 24,
            "tokenizer_pooled_spatial_size": (2, 4),
            "temporal_mode": "gru",
            "gru_hidden_dim": 20,
            "gru_num_layers": 1,
            "latent_hidden_dim": latent_hidden_dim,
        },
        "critic": {
            "class_name": "M2MSequenceCompatibleCritic",
            "proprio_set": "critic",
            "expected_proprio_dim": 99,
            "elevation_set": "height_scan_critic",
            "hidden_dims": [16],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": None,
            "cnn_observation_type": "elevationmap",
            "elevation_encoder_type": "cnn",
            "vision_spatial_size": (28, 20),
            "vision_feature_dim": 8,
            "elevation_history_length": 1,
            "cnn_hidden_dims": [4],
            "cnn_kernel_sizes": [3],
            "cnn_strides": [2],
            "prop_feature_dim": 8,
            "prop_hidden_dims": [16],
            "use_prop_encoder": True,
        },
        "algorithm": {
            "class_name": "M2MDirectPPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.008,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": None,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "dwaq_cfg": None,
            "amp_cfg": None,
            "num_critics": 1,
            "shared_critic": False,
            "share_cnn_encoders": False,
        },
    }


def _construct_algorithm(
    tmp_path: Path,
    *,
    checkpoint: Path | None = None,
    digest: str | None = None,
    latent_hidden_dim: int = 24,
) -> tuple[M2MDirectPPO, TensorDict, Path, str]:
    if checkpoint is None or digest is None:
        checkpoint, digest = _make_frozen_checkpoint(tmp_path)
    obs = _observations(step=0)
    env = SimpleNamespace(num_envs=_NUM_ENVS, num_actions=_ACTION_DIM)
    algorithm = M2MDirectPPO.construct_algorithm(
        obs,
        env,
        _runner_cfg(checkpoint, digest, latent_hidden_dim=latent_hidden_dim),
        "cpu",
    )
    return algorithm, obs, checkpoint, digest


def _clone_parameters(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    live = dict(module.named_parameters())
    return any(
        not torch.equal(value, live[name].detach())
        for name, value in before.items()
    )


def test_standard_builder_recurrent_rollout_update_and_gradient_whitelist(
    tmp_path: Path,
) -> None:
    assert resolve_runner_class("M2MDirectPpoRunner") is M2MDirectPpoRunner
    algorithm, obs, checkpoint, digest = _construct_algorithm(tmp_path)
    assert type(algorithm).__name__ == "M2MDirectPPO"
    assert algorithm.actor.is_recurrent is True
    assert algorithm.critic.is_recurrent is False
    assert type(algorithm.critic).__name__ == "M2MSequenceCompatibleCritic"

    # The normal PPO API calls this policy the "actor", whereas C12 calls the
    # same deployment role "student".  F14 explicitly selects the latter at
    # construction, so the actual C07 receipt is byte-for-byte equal to F09.
    f09_actor_cfg = _runner_cfg(checkpoint, digest)["actor"].copy()
    f09_actor_cfg.pop("class_name")
    f09_twin = type(algorithm.actor)(
        obs=obs,
        obs_groups={"student": ["policy", _STRICT_SET]},
        obs_set="student",
        output_dim=_ACTION_DIM,
        **f09_actor_cfg,
    )
    assert algorithm.actor.architecture_receipt() == f09_twin.architecture_receipt()
    assert (
        algorithm._configuration_receipt()["actor_architecture"]
        == f09_twin.architecture_receipt()
    )

    frozen_before = _clone_parameters(algorithm.actor.ecmm_core)
    student_before = {
        name: parameter.detach().clone()
        for name, parameter in algorithm.actor.named_parameters()
        if parameter.requires_grad
    }
    critic_before = _clone_parameters(algorithm.critic)

    for step in range(_NUM_STEPS):
        with torch.inference_mode():
            actions = algorithm.act(obs)
        assert actions.shape == (_NUM_ENVS, _ACTION_DIM)
        next_obs = _observations(step=step + 1)
        rewards = torch.linspace(-0.1, 0.9, _NUM_ENVS) + 0.01 * step
        dones = torch.zeros(_NUM_ENVS, dtype=torch.uint8)
        if step == 1:
            dones[0] = 1
        algorithm.process_env_step(next_obs, rewards, dones, extras={})
        obs = next_obs

    algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert set(losses) == {"value", "surrogate", "entropy"}
    assert all(math.isfinite(value) for value in losses.values())

    optimizer_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for module in (algorithm.actor, algorithm.critic)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    assert optimizer_ids == expected_ids
    assert all(
        name.startswith(("frame_tokenizer.", "gru.", "latent_head."))
        for name in student_before
    )
    assert set(student_before) == set(algorithm.actor_trainable_parameter_names)
    assert _changed(student_before, algorithm.actor)
    assert _changed(critic_before, algorithm.critic)
    for name, parameter in algorithm.actor.ecmm_core.named_parameters():
        torch.testing.assert_close(parameter, frozen_before[name], rtol=0.0, atol=0.0)
        assert parameter.grad is None

    audit = algorithm.audit()
    assert audit["standard_ppo_objective"] is True
    assert audit["optimizer_matches_trainable_boundary"] is True
    assert audit["frozen_actor_parameters_in_optimizer"] == 0
    assert audit["frozen_actor_parameters_with_grad"] == 0


def test_student_only_checkpoint_roundtrip_and_config_mismatch_rejection(
    tmp_path: Path,
) -> None:
    source, _, checkpoint, digest = _construct_algorithm(tmp_path)
    # Execute ten real updates so iter=9 truthfully means ten completed
    # optimizer steps rather than a counter fabricated by the test.
    for update_index in range(10):
        obs = _observations(step=update_index * 10)
        for step in range(_NUM_STEPS):
            with torch.inference_mode():
                source.act(obs)
            next_obs = _observations(step=update_index * 10 + step + 1)
            source.process_env_step(
                next_obs,
                torch.linspace(0.0, 1.0, _NUM_ENVS),
                torch.zeros(_NUM_ENVS, dtype=torch.uint8),
                extras={},
            )
            obs = next_obs
        source.compute_returns(obs)
        source.update()
    assert source.completed_updates == 10

    source_runner = object.__new__(M2MDirectPpoRunner)
    source_runner.alg = source
    source_runner.current_learning_iteration = 9
    source_runner._m2m_updates_completed = 10
    source_runner._formal_training_io = None
    source_runner.logger = SimpleNamespace(save_model=lambda path, iteration: None)
    checkpoint_path = tmp_path / "f14_runner_checkpoint.pt"
    source_runner.save(str(checkpoint_path), infos={"run": "F14"})
    payload = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    assert payload["schema"] == "m2m_direct_ppo_v2"
    assert payload["iter"] == 9
    assert payload["algorithm_updates_completed"] == 10
    assert payload["infos"] == {"run": "F14"}
    assert set(payload["actor_trainable_state_dict"]) == set(
        source.actor_trainable_parameter_names
    )
    assert not any(
        name.startswith("ecmm_core.")
        for name in payload["actor_trainable_state_dict"]
    )
    assert "actor_state_dict" not in payload
    assert "checkpoint_path" not in payload["frozen_artifact_receipt"]
    assert str(checkpoint.resolve()) not in repr(payload["config_receipt"])
    assert payload["frozen_artifact_receipt"]["checkpoint_bytes_saved"] is False

    target, _, _, _ = _construct_algorithm(
        tmp_path,
        checkpoint=checkpoint,
        digest=digest,
    )
    target_runner = object.__new__(M2MDirectPpoRunner)
    target_runner.alg = target
    target_runner.current_learning_iteration = 0
    target_runner._formal_training_io = None
    target_runner.logger = SimpleNamespace(save_model=lambda path, iteration: None)
    assert target_runner.load(str(checkpoint_path), map_location="cpu") == {"run": "F14"}
    assert target_runner.current_learning_iteration == 10
    assert target_runner._m2m_updates_completed == 10
    for name, parameter in target.actor.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter,
                payload["actor_trainable_state_dict"][name],
                rtol=0.0,
                atol=0.0,
            )
    for name, value in target.critic.state_dict().items():
        torch.testing.assert_close(
            value,
            payload["critic_state_dict"][name],
            rtol=0.0,
            atol=0.0,
        )
    assert target.optimizer.state_dict()["state"]

    class FakeEnv:
        device = "cpu"
        max_episode_length = 100

        def __init__(self) -> None:
            self.episode_length_buf = torch.zeros(_NUM_ENVS, dtype=torch.long)
            self.steps = 0

        def get_observations(self) -> TensorDict:
            return _observations(step=100 + self.steps)

        def step(self, actions: torch.Tensor):
            assert actions.shape == (_NUM_ENVS, _ACTION_DIM)
            self.steps += 1
            return (
                _observations(step=100 + self.steps),
                torch.ones(_NUM_ENVS),
                torch.zeros(_NUM_ENVS, dtype=torch.uint8),
                {},
            )

    class FakeLogger:
        writer = None

        def __init__(self) -> None:
            self.logged_iterations: list[int] = []

        def init_logging_writer(self) -> None:
            pass

        def process_env_step(self, *args, **kwargs) -> None:
            del args, kwargs

        def log(self, *, it: int, **kwargs) -> None:
            del kwargs
            self.logged_iterations.append(it)

        def save_model(self, path: str, iteration: int) -> None:
            del path, iteration

    fake_env = FakeEnv()
    fake_logger = FakeLogger()
    target_runner.env = fake_env
    target_runner.logger = fake_logger
    target_runner.device = "cpu"
    target_runner.is_distributed = False
    target_runner.cfg = {
        "num_steps_per_env": _NUM_STEPS,
        "save_interval": 50,
        "check_for_nan": False,
        "algorithm": {"rnd_cfg": None},
    }
    # The checkpoint contains ten completed updates (iter=9).  Absolute
    # target 11 must execute only update index 10, never replay index 9.
    target_runner.learn(num_learning_iterations=11)
    assert fake_env.steps == _NUM_STEPS
    assert fake_logger.logged_iterations == [10]
    assert target_runner.current_learning_iteration == 10
    assert target_runner._m2m_updates_completed == 11
    resumed_path = tmp_path / "f14_after_one_resumed_update.pt"
    target_runner.save(str(resumed_path))
    resumed = torch.load(resumed_path, weights_only=False, map_location="cpu")
    assert resumed["iter"] == 10
    assert resumed["m2m_direct_ppo_progress"] == {
        "schema": "m2m_direct_ppo_runner_progress_v1",
        "iter": 10,
        "updates_completed": 11,
        "resume_starts_at_update": 11,
    }

    # A malformed Adam moment must be rejected before any live actor/critic
    # tensor is copied.  This is the fail-atomic resume boundary.
    victim, _, _, _ = _construct_algorithm(
        tmp_path,
        checkpoint=checkpoint,
        digest=digest,
    )
    victim_actor_before = {
        name: parameter.detach().clone()
        for name, parameter in victim.actor.named_parameters()
        if parameter.requires_grad
    }
    victim_critic_before = _clone_parameters(victim.critic)
    malformed = copy.deepcopy(payload)
    malformed["optimizer_state_dict"]["param_groups"][0]["params"] = malformed[
        "optimizer_state_dict"
    ]["param_groups"][0]["params"][:-1]
    with pytest.raises(ValueError, match="step-safe"):
        victim.load(malformed, None, True)
    for name, parameter in victim.actor.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter,
                victim_actor_before[name],
                rtol=0.0,
                atol=0.0,
            )
    for name, parameter in victim.critic.named_parameters():
        torch.testing.assert_close(
            parameter,
            victim_critic_before[name],
            rtol=0.0,
            atol=0.0,
        )
    assert victim.optimizer.state_dict()["state"] == {}

    mismatched, _, _, _ = _construct_algorithm(
        tmp_path,
        checkpoint=checkpoint,
        digest=digest,
        latent_hidden_dim=25,
    )
    with pytest.raises(ValueError, match="configuration receipt differs"):
        mismatched.load(payload, None, True)
    with pytest.raises(ValueError, match="always strict"):
        target.load(payload, None, False)
    with pytest.raises(ValueError, match="partial"):
        target.load(payload, {"actor": True}, True)


def test_direct_ppo_rejects_non_gru_actor_and_auxiliary_extensions(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _make_frozen_checkpoint(tmp_path)
    fresh, _, _, _ = _construct_algorithm(
        tmp_path,
        checkpoint=checkpoint,
        digest=digest,
    )
    with pytest.raises(ValueError, match="at least one completed update"):
        fresh.save()
    fresh_runner = object.__new__(M2MDirectPpoRunner)
    fresh_runner.alg = fresh
    fresh_runner.current_learning_iteration = 0
    fresh_runner._m2m_updates_completed = 0
    fresh_runner._formal_training_io = None
    fresh_runner.logger = SimpleNamespace(save_model=lambda path, iteration: None)
    with pytest.raises(ValueError, match="immediately after a completed update"):
        fresh_runner.save(str(tmp_path / "forbidden_preupdate.pt"))

    cfg = _runner_cfg(checkpoint, digest)
    cfg["actor"]["temporal_mode"] = "current"
    env = SimpleNamespace(num_envs=_NUM_ENVS, num_actions=_ACTION_DIM)
    with pytest.raises(ValueError, match="temporal_mode='gru'"):
        M2MDirectPPO.construct_algorithm(_observations(step=0), env, cfg, "cpu")

    cfg = _runner_cfg(checkpoint, digest)
    cfg["algorithm"]["rnd_cfg"] = {"weight": 1.0}
    with pytest.raises(ValueError, match="rnd_cfg=None"):
        M2MDirectPPO.construct_algorithm(_observations(step=0), env, cfg, "cpu")
