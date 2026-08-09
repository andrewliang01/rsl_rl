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

from rsl_rl.models import (
    M2MFrozenECMMCore,
    M2MObservedHistoryProxyTeacher,
    ObservedHistoryMapContract,
    PropMLPElevationFusionModel,
)


_BATCH = 3
_HEIGHT = 16
_WIDTH = 96
_NEAR_M = 0.05
_FAR_M = 1.8569868066305693
_MAX_AGE_S = 1.6
_MAP_SET = "m2m_teacher_observed_history"


def _contract(**overrides: Any) -> ObservedHistoryMapContract:
    kwargs: dict[str, Any] = {
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
    kwargs.update(overrides)
    return ObservedHistoryMapContract(**kwargs)


def _teacher_map(
    *,
    batch_size: int = _BATCH,
    valid: torch.Tensor | None = None,
    age_s: torch.Tensor | None = None,
    range_m: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = (batch_size, 1, _HEIGHT, _WIDTH)
    if range_m is None:
        range_m = torch.empty(shape).uniform_(_NEAR_M, _FAR_M)
    if valid is None:
        valid = torch.ones(shape)
    if age_s is None:
        age_s = torch.rand(shape) * _MAX_AGE_S
        age_s = torch.where(valid == 1.0, age_s, torch.full_like(age_s, _MAX_AGE_S))
    return torch.stack((range_m, valid, age_s), dim=2)


def _observations(teacher_map: torch.Tensor) -> TensorDict:
    batch_size = teacher_map.shape[0]
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "height_scan_policy": teacher_map[:, 0, 0:1].clone(),
            _MAP_SET: teacher_map,
        },
        batch_size=[batch_size],
    )


def _actor(obs: TensorDict, **overrides: Any) -> PropMLPElevationFusionModel:
    kwargs: dict[str, Any] = {
        "obs": obs,
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
    kwargs.update(overrides)
    return PropMLPElevationFusionModel(**kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core(tmp_path: Path, obs: TensorDict) -> M2MFrozenECMMCore:
    torch.manual_seed(50)
    source = _actor(obs)
    checkpoint = tmp_path / "synthetic-m90.pt"
    torch.save({"actor_state_dict": source.state_dict()}, checkpoint)
    torch.manual_seed(99)
    return M2MFrozenECMMCore(
        _actor(obs),
        checkpoint_path=checkpoint,
        expected_sha256=_sha256(checkpoint),
    )


def _proxy(
    tmp_path: Path,
    obs: TensorDict,
    *,
    contract: ObservedHistoryMapContract | None = None,
) -> tuple[M2MObservedHistoryProxyTeacher, M2MFrozenECMMCore]:
    core = _core(tmp_path, obs)
    teacher = M2MObservedHistoryProxyTeacher(
        core,
        map_set=_MAP_SET,
        contract=contract or _contract(),
    )
    return teacher, core


def test_contract_audit_binds_observed_causal_source_and_fixed_layout() -> None:
    audit = _contract().audit()
    assert audit["source"] == "observed_m52_history"
    assert audit["alignment"] == "gt_pose_training_only"
    assert audit["target_grid"] == "m90_spherical_16x96"
    assert audit["uses_future_frames"] is False
    assert audit["uses_privileged_terrain_mesh"] is False
    assert audit["uses_synthetic_fill"] is False
    assert audit["history_layout"] == "current_and_past_only"
    assert audit["tensor_layout"] == "B_K_C_H_W"
    assert audit["shape_without_batch"] == [1, 3, 16, 96]
    assert audit["channels"] == ["range_m", "valid", "age_s"]
    assert audit["unknown_encoder_value"] == "far_range_m"
    assert audit["valid_semantics"] == "finite_exact_binary_0_or_1"
    assert audit["valid_range_semantics"] == "finite_and_within_near_far"
    assert audit["age_semantics"] == "seconds_in_closed_interval_0_to_max_age"
    assert audit["unknown_age_semantics"] == "exactly_max_age_s"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "full_terrain_oracle", "observed M52 history"),
        ("alignment", "deployable_odometry", "GT pose"),
        ("target_grid", "m52_grid", "M90 16x96"),
        ("uses_future_frames", True, "Future-frame leakage"),
        ("uses_privileged_terrain_mesh", True, "Terrain-mesh leakage"),
        ("uses_synthetic_fill", True, "Synthetic or oracle filling"),
    ],
)
def test_contract_rejects_source_and_privilege_leakage(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _contract(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uses_future_frames", 0),
        ("uses_privileged_terrain_mesh", None),
        ("uses_synthetic_fill", "false"),
    ],
)
def test_contract_requires_explicit_boolean_leakage_flags(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="explicit bool"):
        _contract(**{field: value})


def test_all_unknown_cells_become_far_without_completion_and_report_zero_coverage(tmp_path: Path) -> None:
    valid = torch.zeros(_BATCH, 1, _HEIGHT, _WIDTH)
    age = torch.full_like(valid, _MAX_AGE_S)
    unknown_values = torch.full_like(valid, float("nan"))
    teacher_map = _teacher_map(valid=valid, age_s=age, range_m=unknown_values)
    obs = _observations(teacher_map)
    teacher, _ = _proxy(tmp_path, obs)

    m90_range, diagnostics = teacher.prepare_m90_range(teacher_map)
    latent_a, action, label_diagnostics = teacher.teacher_labels(obs)

    torch.testing.assert_close(m90_range, torch.full_like(m90_range, _FAR_M), rtol=0.0, atol=0.0)
    assert latent_a.shape == (_BATCH, 64)
    assert action.shape == (_BATCH, 29)
    assert torch.isfinite(latent_a).all()
    assert torch.isfinite(action).all()
    for result in (diagnostics, label_diagnostics):
        torch.testing.assert_close(result["observed_coverage"], torch.zeros(_BATCH))
        torch.testing.assert_close(result["unknown_fraction"], torch.ones(_BATCH))
        torch.testing.assert_close(result["observed_count"], torch.zeros(_BATCH))
        torch.testing.assert_close(result["observed_mean_age_s"], torch.zeros(_BATCH))
        torch.testing.assert_close(result["observed_max_age_s"], torch.zeros(_BATCH))


def test_full_valid_same_range_is_exactly_equivalent_to_frozen_m90_core(tmp_path: Path) -> None:
    teacher_map = _teacher_map()
    obs = _observations(teacher_map)
    teacher, core = _proxy(tmp_path, obs)

    expected_a, expected_action = core.teacher_labels(obs)
    actual_a, actual_action, diagnostics = teacher.teacher_labels(obs)

    torch.testing.assert_close(actual_a, expected_a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_action, expected_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(diagnostics["observed_coverage"], torch.ones(_BATCH))
    assert not actual_a.requires_grad
    assert not actual_action.requires_grad


def test_age_diagnostics_cannot_change_A_or_action(tmp_path: Path) -> None:
    ranges = torch.empty(_BATCH, 1, _HEIGHT, _WIDTH).uniform_(_NEAR_M, _FAR_M)
    valid = torch.ones_like(ranges)
    young_map = _teacher_map(valid=valid, range_m=ranges, age_s=torch.zeros_like(ranges))
    old_map = _teacher_map(valid=valid, range_m=ranges, age_s=torch.full_like(ranges, _MAX_AGE_S))
    young_obs = _observations(young_map)
    old_obs = young_obs.clone()
    old_obs[_MAP_SET] = old_map
    teacher, _ = _proxy(tmp_path, young_obs)

    young_a, young_action, young_diagnostics = teacher.teacher_labels(young_obs)
    old_a, old_action, old_diagnostics = teacher.teacher_labels(old_obs)

    torch.testing.assert_close(young_a, old_a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(young_action, old_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(young_diagnostics["observed_mean_age_s"], torch.zeros(_BATCH))
    torch.testing.assert_close(old_diagnostics["observed_mean_age_s"], torch.full((_BATCH,), _MAX_AGE_S))


def test_partial_unknown_values_are_discarded_before_frozen_encoder(tmp_path: Path) -> None:
    valid = torch.ones(_BATCH, 1, _HEIGHT, _WIDTH)
    valid[..., ::2] = 0.0
    age = torch.rand_like(valid) * _MAX_AGE_S
    age = torch.where(valid == 1.0, age, torch.full_like(age, _MAX_AGE_S))
    ranges = torch.empty_like(valid).uniform_(_NEAR_M, _FAR_M)
    ranges[valid == 0.0] = float("nan")
    teacher_map = _teacher_map(valid=valid, age_s=age, range_m=ranges)
    obs = _observations(teacher_map)
    teacher, core = _proxy(tmp_path, obs)

    sanitized, diagnostics = teacher.prepare_m90_range(teacher_map)
    proxy_a, _, _ = teacher.teacher_labels(obs)
    expected_a = core.encode_teacher_A(sanitized)

    assert torch.all(sanitized[valid == 0.0] == _FAR_M)
    torch.testing.assert_close(proxy_a, expected_a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(diagnostics["observed_coverage"], torch.full((_BATCH,), 0.5))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_shape", "shape"),
        ("wrong_dtype", "float32"),
        ("nonbinary_valid", "exact \\{0,1\\}"),
        ("valid_nan_range", "valid teacher range must be finite"),
        ("valid_below_near", "below near_range_m"),
        ("valid_above_far", "exceeds far_range_m"),
        ("negative_age", "\\[0,max_age_s\\]"),
        ("excess_age", "\\[0,max_age_s\\]"),
        ("nan_age", "age channel must be finite"),
        ("unknown_age_not_max", "Unknown cells"),
    ],
)
def test_map_shape_valid_range_and_age_contracts_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    teacher_map = _teacher_map()
    obs = _observations(teacher_map)
    teacher, _ = _proxy(tmp_path, obs)
    candidate = teacher_map.clone()

    if mutation == "wrong_shape":
        candidate = candidate[..., :-1]
    elif mutation == "wrong_dtype":
        candidate = candidate.double()
    elif mutation == "nonbinary_valid":
        candidate[0, 0, 1, 0, 0] = 0.5
    elif mutation == "valid_nan_range":
        candidate[0, 0, 0, 0, 0] = float("nan")
    elif mutation == "valid_below_near":
        candidate[0, 0, 0, 0, 0] = _NEAR_M - 0.01
    elif mutation == "valid_above_far":
        candidate[0, 0, 0, 0, 0] = _FAR_M + 0.01
    elif mutation == "negative_age":
        candidate[0, 0, 2, 0, 0] = -0.01
    elif mutation == "excess_age":
        candidate[0, 0, 2, 0, 0] = _MAX_AGE_S + 0.01
    elif mutation == "nan_age":
        candidate[0, 0, 2, 0, 0] = float("nan")
    elif mutation == "unknown_age_not_max":
        candidate[0, 0, 1, 0, 0] = 0.0
        candidate[0, 0, 2, 0, 0] = 0.0
    else:
        raise AssertionError(f"Unhandled mutation {mutation}")

    with pytest.raises(ValueError, match=message):
        teacher.prepare_m90_range(candidate)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("near_range_m", 0.1, "near range mismatch"),
        ("far_range_m", 2.0, "far range mismatch"),
    ],
)
def test_proxy_rejects_map_and_frozen_encoder_range_mismatch(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    teacher_map = _teacher_map()
    obs = _observations(teacher_map)
    core = _core(tmp_path, obs)
    with pytest.raises(ValueError, match=message):
        M2MObservedHistoryProxyTeacher(
            core,
            map_set=_MAP_SET,
            contract=_contract(**{field: value}),
        )


def test_proxy_is_fully_frozen_and_train_preserves_batch_norm_eval(tmp_path: Path) -> None:
    teacher_map = _teacher_map()
    obs = _observations(teacher_map)
    teacher, core = _proxy(tmp_path, obs)
    teacher.train()

    batch_norms = [
        module
        for module in teacher.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert not core.training
    assert not core.actor.training
    assert all(not module.training for module in batch_norms)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    audit = teacher.parameter_audit()
    assert audit["phase"] == "phase0_observed_history_proxy"
    assert audit["trainable_adapter_present"] is False
    assert audit["parameter_count"] > 0
    assert audit["trainable_parameter_count"] == 0
    assert audit["batch_norm"]["training_count"] == 0
    assert audit["frozen_ecmm"]["teacher_loaded"] is True
