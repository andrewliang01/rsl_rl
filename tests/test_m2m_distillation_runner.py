# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import hashlib
import torch
import torch.nn as nn
from pathlib import Path
from tensordict import TensorDict
from typing import Any

import pytest

from rsl_rl.runners import (
    DistillationRunner,
    M2MDistillationRunner,
    OnPolicyRunner,
    make_runner,
    resolve_runner_class,
)
from rsl_rl.storage import M2MSequenceRolloutStorage, RolloutStorage


class _RunnerFrozenCore(nn.Module):
    def __init__(self, checkpoint_sha256: str, actor_state_dict_key: str) -> None:
        super().__init__()
        self.action_head = nn.Linear(64, 29, bias=False)
        self.action_head.requires_grad_(False)
        self.checkpoint_sha256 = checkpoint_sha256
        self.actor_state_dict_key = actor_state_dict_key

    def action_mean(self, latent: torch.Tensor) -> torch.Tensor:
        return self.action_head(latent)

    def parameter_audit(self) -> dict[str, Any]:
        return {
            "contract": {
                "latent_a_dim": 64,
                "action_dim": 29,
                "proprio_dim": 2,
            }
        }


class _RunnerStudent(nn.Module):
    is_recurrent = True
    temporal_mode = "gru"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        frozen_ecmm_checkpoint_path: str,
        frozen_ecmm_expected_sha256: str,
        frozen_ecmm_actor_state_dict_key: str,
        gru_num_layers: int,
        gru_hidden_dim: int,
    ) -> None:
        super().__init__()
        assert Path(frozen_ecmm_checkpoint_path).is_file()
        assert obs_set == "student"
        assert obs_groups[obs_set] == ["policy", "strict_frame"]
        assert output_dim == 29
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_dim = gru_hidden_dim
        self.frame_tokenizer = nn.Sequential(nn.Linear(5, 8), nn.Tanh())
        self.gru = nn.GRU(8, gru_hidden_dim, num_layers=gru_num_layers)
        self.current_encoder = None
        self.latent_head = nn.Linear(gru_hidden_dim, 64)
        self.ecmm_core = _RunnerFrozenCore(
            frozen_ecmm_expected_sha256,
            frozen_ecmm_actor_state_dict_key,
        )
        self._hidden_state: torch.Tensor | None = None
        self.rollout_calls = 0

    @staticmethod
    def _input(obs: TensorDict) -> torch.Tensor:
        return torch.cat((obs["policy"], obs["strict_frame"]), dim=-1)

    def forward_with_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del masks, hidden_state, stochastic_output
        self.rollout_calls += 1
        token = self.frame_tokenizer(self._input(obs))
        output, self._hidden_state = self.gru(token.unsqueeze(0), self._hidden_state)
        latent = self.latent_head(output.squeeze(0))
        return self.ecmm_core.action_mean(latent), latent

    def predict_padded_latent_and_action_mean(
        self,
        obs: TensorDict,
        masks: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.frame_tokenizer(self._input(obs)) * masks.to(dtype=torch.float32)
        output, _ = self.gru(token, hidden_state)
        latent = self.latent_head(output)
        return latent, self.ecmm_core.action_mean(latent)

    def predict_latent_and_action_mean(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del masks, hidden_state
        token = self.frame_tokenizer(self._input(obs))
        latent = self.latent_head(torch.tanh(token[..., : self.gru_hidden_dim]))
        return latent, self.ecmm_core.action_mean(latent)

    def get_hidden_state(self) -> torch.Tensor | None:
        return self._hidden_state

    def reset(self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> None:
        if hidden_state is not None:
            self._hidden_state = hidden_state
        if dones is not None and self._hidden_state is not None:
            done = dones.reshape(-1).to(dtype=torch.bool)
            self._hidden_state[:, done] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    @property
    def output_std(self) -> torch.Tensor:
        return torch.ones(29)

    def architecture_receipt(self) -> dict[str, Any]:
        """Return a stable JSON-safe identity for C11 checkpoint binding."""
        return {
            "schema": "test_c07_runner_student_v1",
            "temporal_mode": self.temporal_mode,
            "gru_num_layers": self.gru_num_layers,
            "gru_hidden_dim": self.gru_hidden_dim,
            "deployment_groups": ["policy", "strict_frame"],
            "latent_dim": 64,
            "action_dim": 29,
        }


class _RunnerTeacher(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        shared_ecmm_core: nn.Module,
    ) -> None:
        super().__init__()
        assert obs_set == "teacher"
        assert obs_groups[obs_set] == ["policy", "teacher_map"]
        assert output_dim == 29
        assert "teacher_map" in obs
        self.ecmm_core = shared_ecmm_core
        self.map_encoder = nn.Linear(4, 64)
        self.loaded_value: float | None = None

    def load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        assert checkpoint["schema"] == "test_c06_map_only_v1"
        self.map_encoder.load_state_dict(checkpoint["map_encoder_state_dict"], strict=True)
        self.loaded_value = float(checkpoint["loaded_value"])

    def predict_latent_and_action_mean(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.map_encoder(obs["teacher_map"])
        return latent, self.ecmm_core.action_mean(latent)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones


class _RunnerEnv:
    def __init__(self, num_envs: int = 64) -> None:
        self.num_envs = num_envs
        self.num_actions = 29
        self.device = "cpu"
        self.cfg: dict[str, Any] = {"name": "synthetic_m2m_runner"}
        self.max_episode_length = 50
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self._step = 0

    def _obs(self) -> TensorDict:
        env = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(-1)
        return TensorDict(
            {
                "policy": torch.cat((env / self.num_envs, torch.full_like(env, self._step / 10.0)), dim=-1),
                "strict_frame": torch.cat(
                    (
                        torch.full_like(env, 0.25),
                        torch.full_like(env, 1.0),
                        torch.full_like(env, self._step / 100.0),
                    ),
                    dim=-1,
                ),
                "teacher_map": torch.full((self.num_envs, 4), 0.1 * (self._step + 1)),
                "ground_truth_pose": torch.full((self.num_envs, 7), 1000.0 + self._step),
            },
            batch_size=[self.num_envs],
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert actions.shape == (self.num_envs, self.num_actions)
        self._step += 1
        self.episode_length_buf += 1
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._step == 2:
            dones[0] = True
        return self._obs(), torch.zeros(self.num_envs), dones, {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path) -> tuple[Path, str, Path, str]:
    frozen = tmp_path / "frozen-m90.pt"
    torch.save({"actor_state_dict": {}}, frozen)
    encoder = nn.Linear(4, 64)
    with torch.no_grad():
        encoder.weight.fill_(0.125)
        encoder.bias.fill_(-0.25)
    teacher = tmp_path / "c06-map-only.pt"
    torch.save(
        {
            "schema": "test_c06_map_only_v1",
            "map_encoder_state_dict": encoder.state_dict(),
            "loaded_value": 7.0,
        },
        teacher,
    )
    return frozen, _sha256(frozen), teacher, _sha256(teacher)


def _config(tmp_path: Path) -> dict[str, Any]:
    frozen, frozen_sha, teacher, teacher_sha = _artifacts(tmp_path)
    return {
        "seed": 42,
        "num_steps_per_env": 2,
        "save_interval": 1,
        "check_for_nan": True,
        "logger": "tensorboard",
        "obs_groups": {
            "student": ["policy", "strict_frame"],
            "teacher": ["policy", "teacher_map"],
        },
        "frozen_ecmm_artifact": {
            "checkpoint_path": str(frozen),
            "expected_sha256": frozen_sha,
            "actor_state_dict_key": "actor_state_dict",
        },
        "teacher_artifact": {
            "checkpoint_path": str(teacher),
            "expected_sha256": teacher_sha,
        },
        "student": {
            "class_name": f"{__name__}:_RunnerStudent",
            "obs_set": "student",
            "gru_num_layers": 1,
            "gru_hidden_dim": 6,
        },
        "teacher": {
            "class_name": f"{__name__}:_RunnerTeacher",
            "obs_set": "teacher",
        },
        "storage": {
            "allowed_student_keys": ["policy", "strict_frame"],
            "hidden_state_shape": [1, 6],
            "target_dtype": "float32",
            "hidden_state_dtype": "float32",
            "student_obs_storage_dtypes": {"strict_frame": "float16"},
        },
        "algorithm": {
            "class_name": "M2MLatentActionDistillation",
            "loss_config": {
                "objective_mode": "joint",
                "latent_smooth_l1_weight": 1.0,
                "latent_cosine_weight": 0.2,
                "action_mse_weight": 2.0,
                "smooth_l1_beta": 1.0,
                "latent_normalization": "l2",
                "normalization_eps": 1.0e-8,
            },
            "learning_rate": 1.0e-2,
            "optimizer": "adam",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "sequence_length": 2,
            "max_grad_norm": 1.0,
            "rollout_action_source": "student_mean",
            "strict_teacher_label_checks": False,
            "shuffle_sequences": False,
            "sequence_seed": 13,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "dwaq_cfg": None,
            "amp_cfg": None,
        },
    }


def test_exact_runner_factory_preserves_legacy_names_and_adds_m2m() -> None:
    """The shared factory remains an exact append-only runner-name mapping."""
    assert resolve_runner_class("OnPolicyRunner") is OnPolicyRunner
    assert resolve_runner_class("DistillationRunner") is DistillationRunner
    assert resolve_runner_class("M2MDistillationRunner") is M2MDistillationRunner
    with pytest.raises(ValueError, match="Unsupported runner class"):
        resolve_runner_class("M2MByAlgorithmSideChannel")


def test_factory_runs_64_env_update_with_c10_storage_and_logger_contract(tmp_path: Path) -> None:
    """A full synthetic collection/update uses C10 and standard logger fields."""
    runner = make_runner(
        "M2MDistillationRunner",
        _RunnerEnv(),
        _config(tmp_path),
        log_dir=None,
        device="cpu",
    )
    assert isinstance(runner, M2MDistillationRunner)
    assert isinstance(runner.alg.storage, M2MSequenceRolloutStorage)
    assert not isinstance(runner.alg.storage, RolloutStorage)
    assert runner.alg.teacher_loaded is True
    assert runner.alg.rnd is None
    assert runner.cfg["algorithm"]["rnd_cfg"] is None
    assert runner.alg.get_policy().output_std.shape == (29,)
    assert runner.alg.teacher.loaded_value == 7.0
    assert runner.alg.teacher.ecmm_core is runner.alg.student.ecmm_core
    assert all(not parameter.requires_grad for parameter in runner.alg.teacher.parameters())
    assert runner.alg.storage.student_observations["strict_frame"].dtype == torch.float16
    assert not hasattr(runner.alg.storage, "teacher_map")
    assert not hasattr(runner.alg.storage, "teacher_observations")

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert runner.alg.num_updates == 1
    assert runner.alg.storage.step == 0
    assert runner.alg.student.rollout_calls == 2
    assert runner.audit()["runner_factory_integrated"] is True
    assert runner.audit()["storage"]["legacy_rollout_storage"] is False


def test_runner_checkpoint_roundtrip_binds_both_artifacts_and_rejects_artifact_as_resume(
    tmp_path: Path,
) -> None:
    """Resume binds M90 and C06 hashes without persisting either teacher."""
    config = _config(tmp_path)
    source = make_runner("M2MDistillationRunner", _RunnerEnv(), config, device="cpu")
    source.learn(num_learning_iterations=1)
    checkpoint = tmp_path / "m2m-resume.pt"
    source.save(str(checkpoint), infos={"purpose": "resume"})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert payload["schema"] == "m2m_latent_action_distillation_v1"
    assert "teacher_state_dict" not in payload
    assert "teacher_map" not in payload
    assert payload["frozen_artifact_receipt"]["checkpoint_sha256"] == config["frozen_ecmm_artifact"]["expected_sha256"]
    assert (
        payload["frozen_artifact_receipt"]["teacher_artifact"]["checkpoint_sha256"]
        == config["teacher_artifact"]["expected_sha256"]
    )
    assert payload["frozen_artifact_receipt"]["checkpoint_bytes_saved"] is False
    assert payload["frozen_artifact_receipt"]["teacher_artifact"]["checkpoint_bytes_saved"] is False
    assert payload["config_receipt"]["student"]["architecture_receipt"] == source.alg.student.architecture_receipt()

    restored = make_runner(
        "M2MDistillationRunner",
        _RunnerEnv(),
        copy.deepcopy(config),
        device="cpu",
    )
    infos = restored.load(str(checkpoint))
    assert infos == {"purpose": "resume"}
    assert payload["iter"] + 1 == payload["algorithm_iteration"] == 1
    assert restored.current_learning_iteration == 1
    assert restored.alg.num_updates == source.alg.num_updates
    for name, value in source.alg.student.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value, dict(restored.alg.student.named_parameters())[name])

    with pytest.raises(ValueError, match="not a teacher/frozen artifact"):
        restored.load(config["teacher_artifact"]["checkpoint_path"])
    with pytest.raises(ValueError, match="not a teacher/frozen artifact"):
        restored.load(config["frozen_ecmm_artifact"]["checkpoint_path"])
    with pytest.raises(ValueError, match="map_location"):
        restored.load(str(checkpoint), map_location="meta")

    # Resume starts at the next zero-based runner index, performs exactly one
    # additional update, and emits a self-consistent second receipt.
    restored.learn(num_learning_iterations=1)
    assert restored.alg.num_updates == 2
    assert restored.current_learning_iteration == 1
    second_checkpoint = tmp_path / "m2m-resume-second.pt"
    restored.save(str(second_checkpoint))
    second = torch.load(second_checkpoint, map_location="cpu", weights_only=True)
    assert second["iter"] + 1 == second["algorithm_iteration"] == 2


def test_teacher_sha_is_checked_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong teacher digest stops construction before torch.load sees it."""
    config = _config(tmp_path)
    teacher_path = Path(config["teacher_artifact"]["checkpoint_path"]).resolve()
    config["teacher_artifact"]["expected_sha256"] = "0" * 64
    original_load = torch.load
    loaded_paths: list[Path] = []

    def recording_load(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        loaded_paths.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    with pytest.raises(ValueError, match="teacher_artifact SHA-256 mismatch"):
        make_runner("M2MDistillationRunner", _RunnerEnv(), config, device="cpu")
    assert teacher_path not in loaded_paths


def test_nested_f07_teacher_training_checkpoint_is_selected_by_explicit_key(tmp_path: Path) -> None:
    """A standard F07 PPO checkpoint can expose its nested C06 map artifact."""
    config = _config(tmp_path)
    raw_path = Path(config["teacher_artifact"]["checkpoint_path"])
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    nested_path = tmp_path / "f07-teacher-training.pt"
    torch.save(
        {
            "schema": "m2m_observed_history_teacher_ppo_v1",
            "teacher_artifact": raw,
            "critic_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 17,
            "infos": None,
        },
        nested_path,
    )
    config["teacher_artifact"] = {
        "checkpoint_path": str(nested_path),
        "expected_sha256": _sha256(nested_path),
        "checkpoint_state_key": "teacher_artifact",
    }

    runner = make_runner("M2MDistillationRunner", _RunnerEnv(), config, device="cpu")

    receipt = runner.alg.frozen_artifact_receipt["teacher_artifact"]
    assert runner.alg.teacher.loaded_value == 7.0
    assert receipt["checkpoint_sha256"] == _sha256(nested_path)
    assert receipt["checkpoint_state_key"] == "teacher_artifact"
    assert receipt["container_schema"] == "m2m_observed_history_teacher_ppo_v1"
    assert receipt["schema"] == "test_c06_map_only_v1"


@pytest.mark.parametrize(
    ("section", "key", "match"),
    [
        ("teacher", "resume_path", "runner-owned"),
        ("student", "load_checkpoint", "runner-owned"),
        ("student", "frozen_ecmm_expected_sha256", "runner-owned"),
        ("teacher_artifact", "resume_path", "key mismatch"),
        ("storage", "legacy_storage", "Unsupported M2M storage"),
    ],
)
def test_config_rejects_teacher_resume_aliases_and_legacy_storage(
    tmp_path: Path,
    section: str,
    key: str,
    match: str,
) -> None:
    """Teacher/resume aliases and generic storage switches fail closed."""
    config = _config(tmp_path)
    config[section][key] = "/tmp/forbidden-side-channel.pt"
    with pytest.raises(ValueError, match=match):
        make_runner("M2MDistillationRunner", _RunnerEnv(), config, device="cpu")


def test_checkpoint_requires_cleared_rollout_and_generic_export_is_blocked(tmp_path: Path) -> None:
    """Partial rollout state and pre-C13 generic exports cannot escape."""
    runner = make_runner("M2MDistillationRunner", _RunnerEnv(), _config(tmp_path), device="cpu")
    runner.alg.act(runner.env.get_observations())
    with pytest.raises(RuntimeError, match="cleared rollout boundary"):
        runner.save(str(tmp_path / "mid-rollout.pt"))
    with pytest.raises(NotImplementedError, match="C13 student-only artifact"):
        runner.export_policy_to_jit(str(tmp_path))
    with pytest.raises(NotImplementedError, match="C13 student-only artifact"):
        runner.export_policy_to_onnx(str(tmp_path))


def test_resume_rejects_inconsistent_completed_update_receipt_before_mutation(tmp_path: Path) -> None:
    """Progress tampering is detected before student or optimizer restoration."""
    config = _config(tmp_path)
    source = make_runner("M2MDistillationRunner", _RunnerEnv(), config, device="cpu")
    source.learn(num_learning_iterations=1)
    checkpoint = tmp_path / "valid.pt"
    source.save(str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["iter"] = 9
    tampered = tmp_path / "bad-progress.pt"
    torch.save(payload, tampered)

    restored = make_runner("M2MDistillationRunner", _RunnerEnv(), copy.deepcopy(config), device="cpu")
    before = {
        name: value.detach().clone() for name, value in restored.alg.student.named_parameters() if value.requires_grad
    }
    with pytest.raises(ValueError, match=r"iter \+ 1 == algorithm_iteration"):
        restored.load(str(tampered))
    assert restored.current_learning_iteration == 0
    assert restored.alg.num_updates == 0
    for name, value in restored.alg.student.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value, before[name])


def test_base_runner_fields_cover_add_git_and_formal_boundary(tmp_path: Path) -> None:
    """The manual constructor initializes every inherited operational field."""
    runner = make_runner("M2MDistillationRunner", _RunnerEnv(), _config(tmp_path), device="cpu")
    assert runner.env.num_envs == 64
    assert runner.device == "cpu"
    assert runner.is_distributed is False
    assert runner.gpu_world_size == runner.gpu_global_rank + 1 == 1
    assert runner.current_learning_iteration == 0
    assert runner._formal_training_io is None
    initial_repos = tuple(runner.logger.git_status_repos)
    runner.add_git_repo_to_log(__file__)
    assert tuple(runner.logger.git_status_repos) == (*initial_repos, __file__)
    with pytest.raises(NotImplementedError, match="Formal PPO"):
        runner.configure_formal_training({})
