# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.m2m_frozen_ecmm import M2MFrozenECMMCore
from rsl_rl.models.m2m_observed_history_formal_teacher import (
    M2MObservedHistoryFormalTeacher,
)
from rsl_rl.models.m2m_observed_history_teacher import ObservedHistoryMapContract
from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel


_BATCH = 2
_HEIGHT = 16
_WIDTH = 96
_NEAR_M = 0.05
_FAR_M = 1.8569868066305693
_MAX_AGE_S = 1.6
_MAP_SET = "m2m_teacher_observed_history"


def _contract(**overrides: Any) -> ObservedHistoryMapContract:
    values: dict[str, Any] = {
        "source": "observed_m52_history",
        "alignment": "gt_pose_training_only",
        "target_grid": "m90_spherical_16x96",
        "uses_future_frames": False,
        "uses_privileged_terrain_mesh": False,
        "uses_synthetic_fill": False,
        "near_range_m": _NEAR_M,
        "far_range_m": _FAR_M,
        "max_age_s": _MAX_AGE_S,
    }
    values.update(overrides)
    return ObservedHistoryMapContract(**values)


def test_map_contract_binds_spatial_backend_and_keeps_legacy_default() -> None:
    legacy = _contract()
    assert legacy.storage_backend == "frame_ring"
    assert legacy.audit()["voxel_size_m"] is None

    spatial = _contract(
        storage_backend="voxel_hash_2p5d",
        voxel_size_m=0.05,
        hash_capacity=32768,
        hash_max_probes=8,
    )
    audit = spatial.audit()
    assert audit["storage_backend"] == "voxel_hash_2p5d"
    assert audit["voxel_size_m"] == 0.05
    assert audit["hash_capacity"] == 32768
    assert audit["hash_max_probes"] == 8

    with pytest.raises(ValueError, match="cannot declare voxel-hash"):
        _contract(voxel_size_m=0.05)


def _teacher_map(
    *,
    batch_size: int = _BATCH,
    dtype: torch.dtype = torch.float32,
    range_m: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    age_s: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = (batch_size, 1, _HEIGHT, _WIDTH)
    if range_m is None:
        range_m = torch.empty(shape).uniform_(_NEAR_M, _FAR_M)
    if valid is None:
        valid = torch.ones(shape)
    if age_s is None:
        age_s = torch.rand(shape) * _MAX_AGE_S
        age_s = torch.where(valid == 1.0, age_s, torch.full_like(age_s, _MAX_AGE_S))
    return torch.stack((range_m, valid, age_s), dim=2).to(dtype=dtype)


def _observations(teacher_map: torch.Tensor, *, extra_privileged: bool = False) -> TensorDict:
    batch_size = teacher_map.shape[0]
    values: dict[str, torch.Tensor] = {
        "policy": torch.randn(batch_size, 96),
        _MAP_SET: teacher_map,
    }
    if extra_privileged:
        values.update(
            {
                "future_map": torch.randn(batch_size, 5),
                "m90_scan": torch.randn(batch_size, 5),
                "terrain_mesh": torch.randn(batch_size, 5),
            }
        )
    return TensorDict(values, batch_size=[batch_size])


def _actor(obs: TensorDict, **overrides: Any) -> PropMLPElevationFusionModel:
    height = torch.full((_BATCH, 1, _HEIGHT, _WIDTH), _FAR_M)
    actor_obs = TensorDict(
        {"policy": obs["policy"], "height_scan_policy": height},
        batch_size=[_BATCH],
    )
    values: dict[str, Any] = {
        "obs": actor_obs,
        "obs_groups": {"actor": ["policy", "height_scan_policy"]},
        "obs_set": "actor",
        "output_dim": 29,
        "hidden_dims": [32, 16],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "elevation_set": "height_scan_policy",
        "cnn_observation_type": "depthcamera",
        "depth_camera_near": _NEAR_M,
        "depth_camera_far": _FAR_M,
        "vision_spatial_size": (_HEIGHT, _WIDTH),
        "vision_feature_dim": 64,
        "elevation_history_length": 1,
        "cnn_hidden_dims": [8, 16],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [16],
        "use_prop_encoder": True,
    }
    values.update(overrides)
    return PropMLPElevationFusionModel(**values)


def _actor_cfg() -> dict[str, Any]:
    return {
        "hidden_dims": [32, 16],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "elevation_set": "height_scan_policy",
        "cnn_observation_type": "depthcamera",
        "depth_camera_near": _NEAR_M,
        "depth_camera_far": _FAR_M,
        "vision_spatial_size": (_HEIGHT, _WIDTH),
        "vision_feature_dim": 64,
        "elevation_history_length": 1,
        "cnn_hidden_dims": [8, 16],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [16],
        "use_prop_encoder": True,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(tmp_path: Path, obs: TensorDict) -> Path:
    checkpoint = tmp_path / "synthetic-m90.pt"
    if not checkpoint.exists():
        torch.manual_seed(100)
        torch.save({"actor_state_dict": _actor(obs).state_dict()}, checkpoint)
    return checkpoint


def _core(tmp_path: Path, obs: TensorDict) -> M2MFrozenECMMCore:
    checkpoint = _checkpoint(tmp_path, obs)
    torch.manual_seed(200)
    return M2MFrozenECMMCore(
        _actor(obs),
        checkpoint_path=checkpoint,
        expected_sha256=_sha256(checkpoint),
    )


def _teacher(
    tmp_path: Path,
    obs: TensorDict,
    *,
    production_constructor: bool = False,
    **overrides: Any,
) -> M2MObservedHistoryFormalTeacher:
    values: dict[str, Any] = {
        "obs": obs,
        "obs_groups": {"actor": ["policy", _MAP_SET]},
        "obs_set": "actor",
        "output_dim": 29,
        "teacher_map_set": _MAP_SET,
        "proprio_sets": ["policy"],
        "map_contract": _contract(),
        "encoder_hidden_channels": [8, 16],
        "encoder_pooled_spatial_size": (2, 4),
        "encoder_mlp_hidden_dim": 32,
        # Unit tests exercise the synchronized fail-closed preflight path.
        # Formal F07 rollout configuration explicitly selects the fast path.
        "strict_runtime_value_checks": True,
    }
    if production_constructor:
        checkpoint = _checkpoint(tmp_path, obs)
        values.update(
            {
                "frozen_ecmm_checkpoint_path": str(checkpoint),
                "frozen_ecmm_expected_sha256": _sha256(checkpoint),
                "frozen_ecmm_actor_cfg": _actor_cfg(),
            }
        )
    else:
        values["shared_ecmm_core"] = _core(tmp_path, obs)
    values.update(overrides)
    return M2MObservedHistoryFormalTeacher(**values)


def test_production_constructor_requires_external_checkpoint_path_hash_and_config(tmp_path: Path) -> None:
    obs = _observations(_teacher_map())
    teacher = _teacher(tmp_path, obs, production_constructor=True)

    audit = teacher.parameter_audit()
    assert audit["frozen_ecmm"]["teacher_loaded"] is True
    assert audit["checkpoint_contract"]["frozen_ecmm_weights_saved"] is False
    assert audit["checkpoint_contract"]["frozen_ecmm_path_and_hash_source"] == (
        "external_constructor_configuration"
    )
    assert teacher.ecmm_core.checkpoint_sha256 == _sha256(_checkpoint(tmp_path, obs))

    with pytest.raises(ValueError, match="path, expected SHA-256, and actor config"):
        M2MObservedHistoryFormalTeacher(
            obs,
            {"actor": ["policy", _MAP_SET]},
            "actor",
            29,
            teacher_map_set=_MAP_SET,
            proprio_sets=["policy"],
            map_contract=_contract(),
        )


def test_range_valid_and_age_each_reach_latent_encoder(tmp_path: Path) -> None:
    ranges = torch.full((_BATCH, 1, _HEIGHT, _WIDTH), 0.7)
    valid = torch.ones_like(ranges)
    ages = torch.zeros_like(ranges)
    base_map = _teacher_map(range_m=ranges, valid=valid, age_s=ages)
    obs = _observations(base_map)
    teacher = _teacher(tmp_path, obs)

    base_a = teacher.predict_latent(obs)

    range_obs = obs.clone()
    range_obs[_MAP_SET] = _teacher_map(
        range_m=torch.full_like(ranges, 1.5),
        valid=valid,
        age_s=ages,
    )
    assert not torch.allclose(base_a, teacher.predict_latent(range_obs))

    age_obs = obs.clone()
    age_obs[_MAP_SET] = _teacher_map(
        range_m=ranges,
        valid=valid,
        age_s=torch.full_like(ages, _MAX_AGE_S),
    )
    assert not torch.allclose(base_a, teacher.predict_latent(age_obs))

    far_ranges = torch.full_like(ranges, _FAR_M)
    max_ages = torch.full_like(ages, _MAX_AGE_S)
    valid_base = obs.clone()
    valid_base[_MAP_SET] = _teacher_map(range_m=far_ranges, valid=valid, age_s=max_ages)
    invalid = valid.clone()
    invalid[..., ::2] = 0.0
    invalid_obs = obs.clone()
    invalid_obs[_MAP_SET] = _teacher_map(range_m=far_ranges, valid=invalid, age_s=max_ages)
    assert not torch.allclose(
        teacher.predict_latent(valid_base),
        teacher.predict_latent(invalid_obs),
    )


def test_action_loss_backpropagates_through_frozen_C_only_into_map_encoder(tmp_path: Path) -> None:
    obs = _observations(_teacher_map())
    teacher = _teacher(tmp_path, obs)
    teacher.train()
    _, action_mean = teacher.predict_latent_and_action_mean(obs)
    action_mean.square().mean().backward()

    map_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in teacher.map_encoder.parameters()
        if parameter.grad is not None
    )
    assert map_grad > 0.0
    assert all(parameter.requires_grad for parameter in teacher.map_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.ecmm_core.parameters())
    assert all(parameter.grad is None for parameter in teacher.ecmm_core.parameters())
    batch_norms = [
        module
        for module in teacher.ecmm_core.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)

    audit = teacher.parameter_audit()
    assert audit["only_map_encoder_trainable"] is True
    assert audit["unexpected_trainable_parameter_names"] == []
    assert audit["components"]["map_encoder"]["trainable"] > 0
    assert audit["components"]["frozen_ecmm"]["trainable"] == 0
    assert audit["frozen_ecmm_batch_norm"]["training_count"] == 0


def test_standard_actor_distribution_api_uses_frozen_distribution(tmp_path: Path) -> None:
    obs = _observations(_teacher_map())
    teacher = _teacher(tmp_path, obs)

    sampled = teacher.get_actions(obs)
    deterministic = teacher.evaluate(obs)
    assert sampled.shape == (_BATCH, 29)
    assert deterministic.shape == (_BATCH, 29)
    assert teacher.distribution is teacher.ecmm_core.actor.distribution
    assert teacher.mean.shape == (_BATCH, 29)
    assert teacher.std.shape == (_BATCH, 29)
    assert teacher.entropy.shape == (_BATCH,)
    assert teacher.output_mean is teacher.mean
    assert teacher.output_std is teacher.std
    torch.testing.assert_close(teacher.output_entropy, teacher.entropy)
    assert teacher.get_output_log_prob(sampled).shape == (_BATCH,)
    assert len(teacher.output_distribution_params) == 2
    assert teacher.get_hidden_state() is None


def test_map_only_checkpoint_round_trip_is_elementwise_exact(tmp_path: Path) -> None:
    obs = _observations(_teacher_map())
    torch.manual_seed(300)
    first = _teacher(tmp_path, obs)
    with torch.no_grad():
        for parameter in first.map_encoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    expected_a, expected_action = first.predict_latent_and_action_mean(obs)
    payload = first.checkpoint_state()

    assert set(payload) == {"schema", "config_receipt", "map_encoder_state_dict"}
    assert payload["config_receipt"]["frozen_ecmm"]["weights_in_teacher_checkpoint"] is False
    assert "checkpoint_path" not in payload["config_receipt"]["frozen_ecmm"]
    assert all("ecmm" not in key and "actor" not in key for key in payload["map_encoder_state_dict"])

    checkpoint = tmp_path / "formal-teacher-map-only.pt"
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.manual_seed(999)
    restored = _teacher(tmp_path, obs)
    restored.load_checkpoint_state(loaded)
    actual_a, actual_action = restored.predict_latent_and_action_mean(obs)

    for expected, actual in zip(first.map_encoder.parameters(), restored.map_encoder.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_a, expected_a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_action, expected_action, rtol=0.0, atol=0.0)

    tampered = first.checkpoint_state()
    tampered["config_receipt"]["frozen_ecmm"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="receipt differs"):
        restored.load_checkpoint_state(tampered)


def test_extra_future_m90_and_mesh_tensors_are_not_actor_inputs(tmp_path: Path) -> None:
    obs = _observations(_teacher_map(), extra_privileged=True)
    teacher = _teacher(tmp_path, obs)
    expected = teacher.predict_latent_and_action_mean(obs)
    perturbed = obs.clone()
    for group in ("future_map", "m90_scan", "terrain_mesh"):
        perturbed[group].normal_(mean=1000.0, std=1000.0)
    actual = teacher.predict_latent_and_action_mean(perturbed)

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    actor_inputs = teacher.parameter_audit()["actor_inputs"]
    assert actor_inputs["ordered_groups"] == ["policy", _MAP_SET]
    assert actor_inputs["uses_future_frames"] is False
    assert actor_inputs["uses_m90_observation"] is False
    assert actor_inputs["uses_terrain_mesh"] is False
    assert actor_inputs["uses_synthetic_fill"] is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_map_storage_dtypes_are_promoted_to_encoder_dtype(tmp_path: Path, dtype: torch.dtype) -> None:
    teacher_map = _teacher_map(dtype=dtype)
    obs = _observations(teacher_map)
    teacher = _teacher(tmp_path, obs)
    latent_a = teacher.predict_latent(obs)
    assert latent_a.dtype == next(teacher.map_encoder.parameters()).dtype == torch.float32
    assert torch.isfinite(latent_a).all()


def test_fast_runtime_path_sanitizes_corrupt_values_without_host_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_map = _teacher_map()
    teacher_map[0, 0, 0, 0, 0] = float("nan")
    teacher_map[0, 0, 0, 0, 1] = float("inf")
    teacher_map[0, 0, 0, 0, 2] = float("-inf")
    teacher_map[0, 0, 1, 0, 0] = float("nan")
    teacher_map[0, 0, 1, 0, 1] = 0.75
    teacher_map[0, 0, 1, 0, 2] = float("inf")
    teacher_map[0, 0, 2, 0, 0] = float("nan")
    teacher_map[0, 0, 2, 0, 1] = float("inf")
    teacher_map[0, 0, 2, 0, 2] = float("-inf")
    obs = _observations(teacher_map)
    teacher = _teacher(tmp_path, obs, strict_runtime_value_checks=False)

    def _forbid_host_scalar(self: torch.Tensor, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("Fast formal-teacher rollout attempted a Tensor-to-host conversion.")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", _forbid_host_scalar)
        patch.setattr(torch.Tensor, "item", _forbid_host_scalar)
        patch.setattr(torch.Tensor, "tolist", _forbid_host_scalar)
        normalized = teacher._validate_and_normalize_map(obs[_MAP_SET])
        latent_a, action_mean = teacher.predict_latent_and_action_mean(obs)

    assert normalized.shape == (_BATCH, 3, _HEIGHT, _WIDTH)
    assert latent_a.shape == (_BATCH, 64)
    assert action_mean.shape == (_BATCH, 29)
    assert torch.isfinite(normalized).all()
    assert torch.isfinite(latent_a).all()
    assert torch.isfinite(action_mean).all()
    assert teacher.checkpoint_state()["config_receipt"]["strict_runtime_value_checks"] is False
    audit_inputs = teacher.parameter_audit()["actor_inputs"]
    assert audit_inputs["strict_runtime_value_checks"] is False
    assert audit_inputs["fast_path_nonfinite_policy"] == "gpu_side_sanitize_without_host_sync"


def test_strict_runtime_checks_remain_fail_closed(tmp_path: Path) -> None:
    teacher_map = _teacher_map()
    obs = _observations(teacher_map)
    teacher = _teacher(tmp_path, obs, strict_runtime_value_checks=True)
    obs[_MAP_SET][0, 0, 1, 0, 0] = 0.75
    with pytest.raises(ValueError, match="binary"):
        teacher.predict_latent(obs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_shape", "shape"),
        ("wrong_dtype", "storage dtype"),
        ("nonfinite_valid", "valid channel must be finite"),
        ("nonbinary_valid", "binary"),
        ("nonfinite_valid_range", "valid teacher range must be finite"),
        ("below_near", "below near_range_m"),
        ("above_far", "exceeds far_range_m"),
        ("nonfinite_age", "age channel must be finite"),
        ("unknown_age", "Unknown cells"),
    ],
)
def test_map_shape_dtype_unknown_and_nonfinite_inputs_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    teacher_map = _teacher_map()
    obs = _observations(teacher_map)
    teacher = _teacher(tmp_path, obs)
    candidate = teacher_map.clone()
    if mutation == "wrong_shape":
        candidate = candidate[..., :-1]
    elif mutation == "wrong_dtype":
        candidate = candidate.double()
    elif mutation == "nonfinite_valid":
        candidate[0, 0, 1, 0, 0] = float("nan")
    elif mutation == "nonbinary_valid":
        candidate[0, 0, 1, 0, 0] = 0.5
    elif mutation == "nonfinite_valid_range":
        candidate[0, 0, 0, 0, 0] = float("inf")
    elif mutation == "below_near":
        candidate[0, 0, 0, 0, 0] = _NEAR_M - 0.01
    elif mutation == "above_far":
        candidate[0, 0, 0, 0, 0] = _FAR_M + 0.01
    elif mutation == "nonfinite_age":
        candidate[0, 0, 2, 0, 0] = float("nan")
    elif mutation == "unknown_age":
        candidate[0, 0, 1, 0, 0] = 0.0
        candidate[0, 0, 2, 0, 0] = 0.0
    else:
        raise AssertionError(mutation)
    broken_obs = obs.clone()
    broken_obs[_MAP_SET] = candidate
    with pytest.raises(ValueError, match=message):
        teacher.predict_latent(broken_obs)


@pytest.mark.parametrize("forbidden_name", ["future_map", "m90_map", "terrain_mesh", "oracle_map"])
def test_actor_group_names_cannot_claim_privileged_sources(tmp_path: Path, forbidden_name: str) -> None:
    teacher_map = _teacher_map()
    obs = TensorDict(
        {"policy": torch.randn(_BATCH, 96), forbidden_name: teacher_map},
        batch_size=[_BATCH],
    )
    with pytest.raises(ValueError, match="forbidden tokens"):
        _teacher(
            tmp_path,
            obs,
            teacher_map_set=forbidden_name,
            obs_groups={"actor": ["policy", forbidden_name]},
        )
