# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import (
    M2MFrozenECMMCore,
    PropMLPElevationFusionModel,
    load_frozen_m90_ecmm_core,
)


_BATCH = 3
_PROPRIO_DIM = 96
_ACTION_DIM = 29
_SPATIAL_SIZE = (16, 24)


def _observations(batch_size: int = _BATCH, spatial_size: tuple[int, int] = _SPATIAL_SIZE) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, _PROPRIO_DIM),
            "height_scan_policy": torch.rand(batch_size, 1, *spatial_size) + 0.1,
        },
        batch_size=[batch_size],
    )


def _actor(
    obs: TensorDict,
    *,
    spatial_size: tuple[int, int] = _SPATIAL_SIZE,
    production_width: bool = False,
    **overrides: Any,
) -> PropMLPElevationFusionModel:
    kwargs: dict[str, Any] = {
        "obs": obs,
        "obs_groups": {"actor": ["policy", "height_scan_policy"]},
        "obs_set": "actor",
        "output_dim": _ACTION_DIM,
        "hidden_dims": [512, 256, 128] if production_width else [32, 16],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "elevation_set": "height_scan_policy",
        "cnn_observation_type": "depthcamera",
        "depth_camera_near": 0.05,
        "depth_camera_far": 1.8569868066305693,
        "vision_spatial_size": spatial_size,
        "vision_feature_dim": 64,
        "elevation_history_length": 1,
        "cnn_hidden_dims": [8, 16],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [128] if production_width else [16],
        "use_prop_encoder": True,
    }
    kwargs.update(overrides)
    return PropMLPElevationFusionModel(**kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(path: Path, actor: PropMLPElevationFusionModel, *, key: str = "actor_state_dict") -> str:
    torch.save({key: actor.state_dict(), "iter": 123}, path)
    return _sha256(path)


def _load_synthetic(tmp_path: Path) -> tuple[M2MFrozenECMMCore, PropMLPElevationFusionModel, TensorDict]:
    torch.manual_seed(12)
    obs = _observations()
    source = _actor(obs).eval()
    path = tmp_path / "m90.pt"
    digest = _write_checkpoint(path, source)
    torch.manual_seed(91)
    core = load_frozen_m90_ecmm_core(
        _actor(obs),
        checkpoint_path=path,
        expected_sha256=digest,
    )
    return core, source, obs


def test_strict_load_reproduces_actor_and_exposes_B_A_C(tmp_path: Path) -> None:
    core, source, obs = _load_synthetic(tmp_path)

    with torch.no_grad():
        expected = source(obs)
        b = core.encode_proprio(obs)
        a = core.encode_teacher_A(obs)
        actual = core.action_mean_from_A(b, a)
        label_a, label_action = core.teacher_labels(obs)

    assert b.shape == (_BATCH, 64)
    assert a.shape == (_BATCH, 64)
    assert actual.shape == (_BATCH, _ACTION_DIM)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(label_a, a)
    torch.testing.assert_close(label_action, expected)
    assert not label_a.requires_grad
    assert not label_action.requires_grad
    assert core.teacher_loaded


def test_frozen_C_backpropagates_to_student_A_only(tmp_path: Path) -> None:
    core, _, obs = _load_synthetic(tmp_path)
    with torch.no_grad():
        b = core.encode_proprio(obs)
    student_a = torch.randn(_BATCH, 64, requires_grad=True)

    action = core.action_mean_from_A(b, student_a)
    action.square().mean().backward()

    assert student_a.grad is not None
    assert torch.isfinite(student_a.grad).all()
    assert float(student_a.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in core.parameters())
    assert all(not parameter.requires_grad for parameter in core.parameters())


def test_train_keeps_all_frozen_batch_norm_in_eval_and_audit_is_zero_trainable(tmp_path: Path) -> None:
    core, _, _ = _load_synthetic(tmp_path)
    core.train()

    batch_norms = [
        module
        for module in core.actor.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert not core.actor.training
    assert all(not module.training for module in batch_norms)

    audit = core.parameter_audit()
    assert audit["teacher_loaded"] is True
    assert audit["contract"] == {
        "proprio_dim": 96,
        "proprio_feature_dim": 64,
        "latent_a_dim": 64,
        "action_dim": 29,
        "history_length": 1,
        "spatial_size": list(_SPATIAL_SIZE),
        "observation_type": "depthcamera",
        "depth_camera_near": 0.05,
        "depth_camera_far": 1.8569868066305693,
    }
    assert audit["batch_norm"]["count"] == len(batch_norms)
    assert audit["batch_norm"]["training_count"] == 0
    assert audit["components"]["actor"]["total"] > 0
    assert all(component["trainable"] == 0 for component in audit["components"].values())


def test_hash_mismatch_fails_before_deserialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    obs = _observations()
    path = tmp_path / "m90.pt"
    _write_checkpoint(path, _actor(obs))

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load must not run before hash validation")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        M2MFrozenECMMCore(_actor(obs), checkpoint_path=path, expected_sha256="0" * 64)


def test_missing_actor_key_fails_closed(tmp_path: Path) -> None:
    obs = _observations()
    path = tmp_path / "wrong-key.pt"
    digest = _write_checkpoint(path, _actor(obs), key="teacher_state_dict")

    with pytest.raises(KeyError, match="actor_state_dict"):
        M2MFrozenECMMCore(_actor(obs), checkpoint_path=path, expected_sha256=digest)


def test_state_key_and_shape_mismatches_fail_closed(tmp_path: Path) -> None:
    obs = _observations()
    actor = _actor(obs)
    state = dict(actor.state_dict())

    missing_state = dict(state)
    missing_state.pop(next(iter(missing_state)))
    missing_path = tmp_path / "missing-state-key.pt"
    torch.save({"actor_state_dict": missing_state}, missing_path)
    with pytest.raises(ValueError, match="key mismatch"):
        M2MFrozenECMMCore(
            _actor(obs), checkpoint_path=missing_path, expected_sha256=_sha256(missing_path)
        )

    shape_state = dict(state)
    shaped_key = next(key for key, value in shape_state.items() if value.ndim > 0 and value.shape[0] > 1)
    shape_state[shaped_key] = shape_state[shaped_key][:-1]
    shape_path = tmp_path / "wrong-shape.pt"
    torch.save({"actor_state_dict": shape_state}, shape_path)
    with pytest.raises(ValueError, match="shape mismatch"):
        M2MFrozenECMMCore(
            _actor(obs), checkpoint_path=shape_path, expected_sha256=_sha256(shape_path)
        )


def test_wrong_ecmm_architecture_is_rejected_before_file_access(tmp_path: Path) -> None:
    obs = _observations()
    actor = _actor(obs, elevation_history_length=2)
    with pytest.raises(ValueError, match="H1 actor"):
        M2MFrozenECMMCore(
            actor,
            checkpoint_path=tmp_path / "does-not-exist.pt",
            expected_sha256="0" * 64,
        )


@pytest.mark.skipif(
    not (os.environ.get("M2M_M90_CHECKPOINT") and os.environ.get("M2M_M90_SHA256")),
    reason="set M2M_M90_CHECKPOINT and M2M_M90_SHA256 for the optional read-only artifact test",
)
def test_optional_real_m90_artifact_receipt() -> None:
    """Read-only integration check; the candidate path is never a source default."""
    checkpoint_path = os.environ["M2M_M90_CHECKPOINT"]
    expected_sha256 = os.environ["M2M_M90_SHA256"]
    obs = _observations(batch_size=2, spatial_size=(16, 96))
    actor = _actor(obs, spatial_size=(16, 96), production_width=True)

    core = M2MFrozenECMMCore(
        actor,
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
    )
    audit = core.parameter_audit()
    assert audit["checkpoint_sha256"] == expected_sha256.lower()
    assert audit["components"]["actor"]["trainable"] == 0
    assert core(obs).shape == (2, 29)
