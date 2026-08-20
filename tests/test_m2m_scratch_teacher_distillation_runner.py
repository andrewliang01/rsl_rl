"""End-to-end C12 construction from one ordinary scratch-teacher checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import M2MObservedHistoryScratchTeacher
from rsl_rl.runners import M2MDistillationRunner, make_runner
from rsl_rl.utils.m2m_student_artifact import (
    build_m2m_scratch_student_artifact_payload,
    load_m2m_student_artifact,
    main as student_artifact_main,
    write_m2m_student_artifact,
)


_MAP = "teacher_map"
_FRAME = "strict_frame"


def _contract() -> dict[str, object]:
    return {
        "source": "observed_m52_history",
        "alignment": "gt_pose_training_only",
        "target_grid": "m90_spherical_16x96",
        "uses_future_frames": False,
        "uses_privileged_terrain_mesh": False,
        "uses_synthetic_fill": False,
        "near_range_m": 0.05,
        "far_range_m": 1.85699,
        "storage_backend": "voxel_hash_2p5d",
        "retention_mode": "episode",
        "voxel_size_m": 0.05,
        "hash_capacity": 1024,
        "hash_max_probes": 16,
    }


def _teacher_cfg() -> dict[str, Any]:
    return {
        "class_name": "M2MObservedHistoryScratchTeacher",
        "obs_set": "teacher",
        "teacher_map_set": _MAP,
        "proprio_sets": ["policy"],
        "map_contract": _contract(),
        "encoder_hidden_channels": [8, 16],
        "encoder_pooled_spatial_size": [1, 3],
        "encoder_mlp_hidden_dim": 32,
        "prop_feature_dim": 64,
        "prop_hidden_dims": [32],
        "fusion_hidden_dims": [64, 32],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "strict_runtime_value_checks": False,
    }


class _ScratchEnv:
    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.num_actions = 29
        self.device = "cpu"
        self.cfg = {"name": "scratch-teacher-distillation-test"}
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.step_count = 0

    def _obs(self) -> TensorDict:
        batch = self.num_envs
        frame = torch.empty(batch, 1, 4, 16, 96)
        frame[:, :, 0] = 1.0
        frame[:, :, 1] = 1.0
        frame[:, :, 2] = 0.0
        frame[:, :, 3] = float(self.step_count % 5 == 0)
        teacher_map = torch.ones(batch, 1, 2, 16, 96)
        teacher_map[:, :, 0] = 0.5 + self.step_count * 0.01
        return TensorDict(
            {
                "policy": torch.randn(batch, 96),
                _FRAME: frame,
                _MAP: teacher_map,
                "privileged_pose_must_not_be_stored": torch.full((batch, 7), 999.0),
            },
            batch_size=[batch],
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, 29)
        self.step_count += 1
        self.episode_length_buf += 1
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.step_count == 2:
            dones[0] = True
        return self._obs(), torch.zeros(self.num_envs), dones, {}


def _artifact(tmp_path: Path, env: _ScratchEnv) -> tuple[Path, str]:
    obs = env.get_observations()
    cfg = _teacher_cfg()
    cfg.pop("class_name")
    cfg.pop("obs_set")
    teacher = M2MObservedHistoryScratchTeacher(
        obs,
        {"teacher": ["policy", _MAP]},
        "teacher",
        29,
        **cfg,
    )
    path = tmp_path / "f07-full-teacher.pt"
    torch.save(
        {
            "actor_state_dict": teacher.state_dict(),
            "critic_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 19999,
        },
        path,
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, env: _ScratchEnv, *, temporal_mode: str = "gru") -> dict[str, Any]:
    path, digest = _artifact(tmp_path, env)
    recurrent = temporal_mode == "gru"
    return {
        "seed": 42,
        "num_steps_per_env": 2,
        "save_interval": 1,
        "check_for_nan": True,
        "logger": "tensorboard",
        "obs_groups": {
            "student": ["policy", _FRAME],
            "teacher": ["policy", _MAP],
        },
        "scratch_teacher_artifact": {
            "checkpoint_path": str(path),
            "expected_sha256": digest,
            "actor_state_dict_key": "actor_state_dict",
        },
        "frozen_ecmm_artifact": None,
        "teacher_artifact": None,
        "student": {
            "class_name": "M2MMapFreeRecurrentStudent",
            "obs_set": "student",
            "strict_frame_set": _FRAME,
            "proprio_sets": ["policy"],
            "frame_near_range_m": 0.05,
            "frame_far_range_m": 1.85699,
            "frame_message_period_s": 0.1,
            "frame_max_age_s": 0.5,
            "frame_age_semantics": "uniform_message_age_s",
            "tokenizer_hidden_channels": [4, 8],
            "tokenizer_dim": 16,
            "tokenizer_pooled_spatial_size": [1, 3],
            "temporal_mode": temporal_mode,
            "gru_hidden_dim": 16,
            "gru_num_layers": 1,
            "latent_hidden_dim": 16,
        },
        "teacher": _teacher_cfg(),
        "storage": {
            "allowed_student_keys": ["policy", _FRAME],
            "hidden_state_shape": [1, 16] if recurrent else [1, 1],
            "target_dtype": "float32",
            "hidden_state_dtype": "float32",
            "student_obs_storage_dtypes": {_FRAME: "float16"},
        },
        "algorithm": {
            "class_name": "M2MLatentActionDistillation",
            "loss_config": {
                "objective_mode": "joint",
                "latent_smooth_l1_weight": 1.0,
                "latent_cosine_weight": 0.1,
                "action_mse_weight": 1.0,
                "smooth_l1_beta": 1.0,
                "latent_normalization": "l2",
                "normalization_eps": 1.0e-8,
            },
            "learning_rate": 1.0e-3,
            "optimizer": "adam",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "sequence_length": 2,
            "max_grad_norm": 1.0,
            "rollout_action_source": "student_mean",
            "strict_teacher_label_checks": True,
            "shuffle_sequences": False,
            "sequence_seed": 7,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "dwaq_cfg": None,
            "amp_cfg": None,
        },
    }


def _export_config(config: dict[str, Any]) -> dict[str, Any]:
    student = copy.deepcopy(config["student"])
    student.pop("class_name")
    student.pop("obs_set")
    teacher = copy.deepcopy(config["teacher"])
    teacher.pop("proprio_sets")
    return {
        "schema": "m2m_student_export_scratch_control_v1",
        "obs_set": "student",
        "output_dim": 29,
        "strict_frame_set": student.pop("strict_frame_set"),
        "proprio_sets": student.pop("proprio_sets"),
        "proprio_group_dims": {"policy": 96},
        "scratch_teacher": teacher,
        **student,
    }


@pytest.mark.parametrize("temporal_mode", ["current", "gru"])
def test_single_full_teacher_artifact_runs_online_distillation_update(
    tmp_path: Path,
    temporal_mode: str,
) -> None:
    env = _ScratchEnv()
    config = _config(tmp_path, env, temporal_mode=temporal_mode)
    runner = make_runner("M2MDistillationRunner", env, config, device="cpu")
    assert isinstance(runner, M2MDistillationRunner)
    assert runner.alg.student.ecmm_core.actor is runner.alg.teacher
    assert all(not parameter.requires_grad for parameter in runner.alg.teacher.parameters())
    assert set(runner.alg.storage.student_observations.keys()) == {"policy", _FRAME}
    assert not hasattr(runner.alg.storage, _MAP)
    receipt = runner.alg.frozen_artifact_receipt
    assert receipt["artifact_kind"] == "scratch_teacher_ordinary_ppo_full_actor"
    assert receipt["checkpoint_sha256"] == config["scratch_teacher_artifact"]["expected_sha256"]
    assert receipt["teacher_artifact"]["checkpoint_sha256"] == receipt["checkpoint_sha256"]

    before = {
        name: parameter.detach().clone()
        for name, parameter in runner.alg.student.named_parameters()
        if parameter.requires_grad
    }
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    assert runner.alg.num_updates == 1
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in runner.alg.student.named_parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in runner.alg.teacher.parameters())
    assert runner.audit()["artifacts"]["mode"] == "scratch_teacher_full_actor"


def test_scratch_mode_rejects_legacy_sidecars_before_model_construction(tmp_path: Path) -> None:
    env = _ScratchEnv()
    config = _config(tmp_path, env)
    config["frozen_ecmm_artifact"] = copy.deepcopy(config["scratch_teacher_artifact"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_runner("M2MDistillationRunner", env, config, device="cpu")


@pytest.mark.parametrize("temporal_mode", ["current", "gru"])
def test_scratch_distillation_checkpoint_exports_standalone_student(
    tmp_path: Path,
    temporal_mode: str,
) -> None:
    env = _ScratchEnv()
    config = _config(tmp_path, env, temporal_mode=temporal_mode)
    runner = make_runner("M2MDistillationRunner", env, config, device="cpu")
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    distillation = tmp_path / f"c11-{temporal_mode}.pt"
    torch.save(runner.alg.save(), distillation)
    distillation_sha = hashlib.sha256(distillation.read_bytes()).hexdigest()
    teacher_path = Path(config["scratch_teacher_artifact"]["checkpoint_path"])
    teacher_sha = config["scratch_teacher_artifact"]["expected_sha256"]
    export_config = _export_config(config)
    payload = build_m2m_scratch_student_artifact_payload(
        distillation_checkpoint_path=distillation,
        expected_distillation_sha256=distillation_sha,
        scratch_teacher_checkpoint_path=teacher_path,
        expected_scratch_teacher_sha256=teacher_sha,
        construction_config=export_config,
    )
    assert payload["source_receipt"]["control_artifact_kind"] == (
        "scratch_teacher_ordinary_ppo_full_actor"
    )
    assert payload["source_receipt"]["scratch_teacher_checkpoint_sha256"] == teacher_sha
    assert payload["dependency_receipt"]["constructs_teacher"] is False
    assert payload["dependency_receipt"]["constructs_mapper"] is False
    assert all("map_encoder" not in key for key in payload["student_state_dict"])

    artifact = tmp_path / f"student-{temporal_mode}.pt"
    receipt = write_m2m_student_artifact(payload, artifact)
    policy = load_m2m_student_artifact(
        artifact,
        expected_sha256=receipt["artifact_sha256"],
        expected_frame_age_semantics="uniform_message_age_s",
    )
    assert policy.temporal_mode == temporal_mode
    assert policy.artifact_receipt["source_receipt"] == payload["source_receipt"]

    cli_config = tmp_path / f"scratch-export-{temporal_mode}.json"
    cli_config.write_text(json.dumps(export_config), encoding="utf-8")
    cli_artifact = tmp_path / f"student-cli-{temporal_mode}.pt"
    assert student_artifact_main(
        [
            "export-scratch",
            "--distillation-checkpoint",
            str(distillation),
            "--distillation-sha256",
            distillation_sha,
            "--scratch-teacher-checkpoint",
            str(teacher_path),
            "--scratch-teacher-sha256",
            teacher_sha,
            "--construction-config",
            str(cli_config),
            "--output",
            str(cli_artifact),
        ]
    ) == 0
    assert cli_artifact.is_file()
