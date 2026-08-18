# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: D103, N802

from __future__ import annotations

import copy
import hashlib
import json
import torch
from pathlib import Path
from tensordict import TensorDict
from typing import Any

import onnx
import pytest
from onnx.reference import ReferenceEvaluator

import rsl_rl.utils.m2m_student_artifact as student_artifact_module
from rsl_rl.models.m2m_recurrent_student import M2MMapFreeRecurrentStudent
from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.utils.m2m_student_artifact import (
    build_m2m_student_artifact_payload,
    inspect_m2m_student_artifact,
    load_m2m_student_artifact,
    main,
    write_m2m_student_artifact,
)

_PROPRIO_DIM = 96
_ACTION_DIM = 29
_STRICT_SET = "mid360_strict_frame"
_M90_SET = "height_scan_policy"
_FRAME_AGE_SEMANTICS = "uniform_message_age_s"
_LIVOX_FRAME_AGE_SEMANTICS = "winning_subframe_age_20ms"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_path_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if Path(value).is_absolute() else []
    if isinstance(value, dict):
        return [path for item in value.values() for path in _absolute_path_values(item)]
    if isinstance(value, (list, tuple)):
        return [path for item in value for path in _absolute_path_values(item)]
    return []


def _actor_cfg() -> dict[str, Any]:
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
        "depth_camera_far": 1.8569868066305693,
        "vision_spatial_size": [16, 96],
        "vision_feature_dim": 64,
        "elevation_history_length": 1,
        "cnn_hidden_dims": [4, 8],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [16],
        "use_prop_encoder": True,
    }


def _construction_config(
    *,
    temporal_mode: str = "gru",
    frame_age_semantics: str = _FRAME_AGE_SEMANTICS,
) -> dict[str, Any]:
    return {
        "schema": "m2m_student_export_construction_v2",
        "obs_set": "actor",
        "output_dim": _ACTION_DIM,
        "strict_frame_set": _STRICT_SET,
        "proprio_sets": ["policy"],
        "proprio_group_dims": {"policy": _PROPRIO_DIM},
        "frozen_ecmm_actor_cfg": _actor_cfg(),
        "frame_near_range_m": 0.1,
        "frame_far_range_m": 6.0,
        "frame_message_period_s": 0.1,
        "frame_max_age_s": 0.5,
        "frame_age_semantics": frame_age_semantics,
        "tokenizer_hidden_channels": [4, 8],
        "tokenizer_dim": 24,
        "tokenizer_pooled_spatial_size": [2, 4],
        "temporal_mode": temporal_mode,
        "gru_hidden_dim": 20,
        "gru_num_layers": 1,
        "latent_hidden_dim": 24,
    }


def _m90_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "synthetic-m90.pt"
    if not path.exists():
        torch.manual_seed(100)
        obs = TensorDict(
            {
                "policy": torch.randn(2, _PROPRIO_DIM),
                _M90_SET: torch.rand(2, 1, 16, 96) + 0.05,
            },
            batch_size=[2],
        )
        actor = PropMLPElevationFusionModel(
            obs,
            {"actor": ["policy", _M90_SET]},
            "actor",
            _ACTION_DIM,
            **copy.deepcopy(_actor_cfg()),
        )
        torch.save({"actor_state_dict": actor.state_dict(), "iter": 101}, path)
    return path, _sha256(path)


def _strict_frame(batch_size: int, *, step: int = 0) -> torch.Tensor:
    frame = torch.zeros(batch_size, 1, 4, 16, 96)
    ranges = torch.linspace(0.1, 6.0, 16 * 96).reshape(1, 16, 96)
    frame[:, 0, 0] = (ranges + 0.01 * step).clamp(0.1, 6.0)
    frame[:, 0, 1] = 1.0
    frame[:, 0, 2] = 0.02 * (step + 1)
    frame[:, 0, 3] = float(step % 5 == 0)
    frame[:, 0, 0, 0, 0] = float("nan")
    frame[:, 0, 1, 0, 0] = 0.0
    return frame


def _observations(batch_size: int = 3, *, step: int = 0) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.linspace(-1.0, 1.0, batch_size * _PROPRIO_DIM).reshape(
                batch_size,
                _PROPRIO_DIM,
            )
            + 0.005 * step,
            _STRICT_SET: _strict_frame(batch_size, step=step),
        },
        batch_size=[batch_size],
    )


def _training_student(
    tmp_path: Path,
    *,
    temporal_mode: str = "gru",
    frame_age_semantics: str = _FRAME_AGE_SEMANTICS,
) -> M2MMapFreeRecurrentStudent:
    config = _construction_config(
        temporal_mode=temporal_mode,
        frame_age_semantics=frame_age_semantics,
    )
    m90, digest = _m90_checkpoint(tmp_path)
    torch.manual_seed(300)
    return M2MMapFreeRecurrentStudent(
        _observations(),
        {"actor": ["policy", _STRICT_SET]},
        "actor",
        _ACTION_DIM,
        strict_frame_set=_STRICT_SET,
        proprio_sets=["policy"],
        frozen_ecmm_checkpoint_path=str(m90),
        frozen_ecmm_expected_sha256=digest,
        frozen_ecmm_actor_cfg=copy.deepcopy(config["frozen_ecmm_actor_cfg"]),
        frame_near_range_m=config["frame_near_range_m"],
        frame_far_range_m=config["frame_far_range_m"],
        frame_message_period_s=config["frame_message_period_s"],
        frame_max_age_s=config["frame_max_age_s"],
        frame_age_semantics=config["frame_age_semantics"],
        tokenizer_hidden_channels=config["tokenizer_hidden_channels"],
        tokenizer_dim=config["tokenizer_dim"],
        tokenizer_pooled_spatial_size=tuple(config["tokenizer_pooled_spatial_size"]),
        temporal_mode=temporal_mode,
        gru_hidden_dim=config["gru_hidden_dim"],
        gru_num_layers=config["gru_num_layers"],
        latent_hidden_dim=config["latent_hidden_dim"],
    )


def _c11_checkpoint(
    tmp_path: Path,
    *,
    frozen_receipt_sha: str | None = None,
    frame_age_semantics: str = _FRAME_AGE_SEMANTICS,
) -> tuple[Path, str, dict[str, Any]]:
    student = _training_student(tmp_path, frame_age_semantics=frame_age_semantics)
    torch.manual_seed(400)
    with torch.no_grad():
        for parameter in student.parameters():
            if parameter.requires_grad:
                parameter.copy_(torch.randn_like(parameter) * 0.05)
    trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    _m90, m90_sha = _m90_checkpoint(tmp_path)
    payload: dict[str, Any] = {
        "schema": "m2m_latent_action_distillation_v1",
        "config_receipt": {
            "algorithm": "M2MLatentActionDistillation",
            "student": {
                "class": "rsl_rl.models.m2m_recurrent_student.M2MMapFreeRecurrentStudent",
                "is_recurrent": True,
                "temporal_mode": "gru",
                "trainable_parameter_names": list(trainable_state),
                "allowed_observation_keys": ["policy", _STRICT_SET],
                "hidden_state_shape": [1, 20],
                "architecture_receipt": student.architecture_receipt(),
            },
        },
        "student_trainable_state_dict": trainable_state,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "algorithm_iteration": 17,
        "frozen_artifact_receipt": {
            "checkpoint_sha256": frozen_receipt_sha or m90_sha,
            "actor_state_dict_key": "actor_state_dict",
            "contract": {"latent_a_dim": 64, "action_dim": 29},
            "checkpoint_path_source": "external_constructor_configuration",
            "checkpoint_bytes_saved": False,
        },
    }
    path = tmp_path / "c11-distillation.pt"
    torch.save(payload, path)
    return path, _sha256(path), payload


def _build_payload(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    m90, m90_sha = _m90_checkpoint(tmp_path)
    c11, c11_sha, c11_payload = _c11_checkpoint(tmp_path)
    payload = build_m2m_student_artifact_payload(
        distillation_checkpoint_path=c11,
        expected_distillation_sha256=c11_sha,
        frozen_m90_checkpoint_path=m90,
        expected_frozen_m90_sha256=m90_sha,
        construction_config=_construction_config(),
    )
    return payload, c11_payload


def _write_artifact(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    payload, c11_payload = _build_payload(tmp_path)
    path = tmp_path / "student-only.pt"
    receipt = write_m2m_student_artifact(payload, path)
    return path, receipt, c11_payload


def test_payload_contains_only_student_B_C_distribution_and_strict_receipts(tmp_path: Path) -> None:
    payload, _c11 = _build_payload(tmp_path)
    assert payload["schema"] == "m2m_student_only_artifact_v2"
    assert payload["network_config"]["schema"] == "m2m_student_only_network_v2"
    assert payload["network_config"]["frame_age_semantics"] == _FRAME_AGE_SEMANTICS
    assert payload["input_receipt"]["frame_age_semantics"] == _FRAME_AGE_SEMANTICS
    assert payload["input_receipt"]["strict_frame_channels"] == [
        "range_m",
        "valid",
        "frame_age_s",
        "new_frame",
    ]
    state_keys = set(payload["student_state_dict"])
    assert any(key.startswith("frame_tokenizer.") for key in state_keys)
    assert any(key.startswith("gru.") for key in state_keys)
    assert any(key.startswith("latent_head.") for key in state_keys)
    assert any(key.startswith("obs_normalizer.") for key in state_keys)
    assert any(key.startswith("prop_mlp.") for key in state_keys)
    assert any(key.startswith("action_head.") for key in state_keys)
    assert any(key.startswith("distribution.") for key in state_keys)
    forbidden = ("teacher", "mapper", "rolling", "elevation_encoder", "height_scan", "m90")
    assert not [key for key in state_keys if any(token in key.lower() for token in forbidden)]
    assert payload["dependency_receipt"] == {
        "constructs_teacher": False,
        "constructs_mapper": False,
        "constructs_teacher_observation": False,
        "contains_m90_perception_encoder": False,
        "contains_optimizer_state": False,
        "contains_teacher_latent_labels": False,
        "contains_teacher_action_labels": False,
        "external_checkpoint_required_at_runtime": False,
    }
    assert payload["source_receipt"]["source_paths_embedded"] is False
    assert payload["source_receipt"]["source_checkpoint_bytes_embedded"] is False
    assert "frozen_ecmm_actor_cfg" not in payload["network_config"]
    assert "elevation_set" not in repr(payload["network_config"])
    artifact_metadata = {key: value for key, value in payload.items() if key != "student_state_dict"}
    assert _absolute_path_values(artifact_metadata) == []
    m90, _ = _m90_checkpoint(tmp_path)
    c11, _, _ = _c11_checkpoint(tmp_path)
    assert str(m90) not in repr(payload)
    assert str(c11) not in repr(payload)


def test_isaac_free_roundtrip_matches_training_student_over_recurrent_steps(tmp_path: Path) -> None:
    artifact, receipt, c11 = _write_artifact(tmp_path)
    policy = load_m2m_student_artifact(
        artifact,
        expected_sha256=receipt["artifact_sha256"],
        expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
    )
    source = _training_student(tmp_path)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(c11["student_trainable_state_dict"][name])
    source.eval()
    source.reset()
    policy.reset()
    for step in range(4):
        obs = _observations(step=step)
        source_hidden_before = source.get_hidden_state()
        if source_hidden_before is not None:
            source_hidden_before = source_hidden_before.clone()
        source_action, source_latent = source.forward_with_latent(obs)
        policy_action = policy(obs)
        policy_hidden = policy.get_hidden_state()
        assert policy_hidden is not None
        policy.reset()
        direct_action, direct_latent, direct_hidden = policy.step_tensors(
            obs["policy"],
            obs[_STRICT_SET],
            source_hidden_before,
        )
        policy._hidden_state = policy_hidden
        torch.testing.assert_close(policy_action, source_action, rtol=0.0, atol=0.0)
        torch.testing.assert_close(direct_action, source_action, rtol=0.0, atol=0.0)
        torch.testing.assert_close(direct_latent, source_latent, rtol=0.0, atol=0.0)
        torch.testing.assert_close(direct_hidden, source.get_hidden_state(), rtol=0.0, atol=0.0)

    assert all(not parameter.requires_grad for parameter in policy.parameters())
    audit = policy.dependency_audit()
    assert audit["forbidden_state_keys"] == []
    assert audit["forbidden_module_names"] == []
    assert audit["external_checkpoint_required_at_runtime"] is False
    privileged = _observations()
    privileged["teacher_map"] = torch.randn(3, 1, 3, 16, 96)
    with pytest.raises(ValueError, match="unexpected"):
        policy(privileged)


def test_artifact_load_does_not_construct_C07_M90_teacher_or_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, receipt, _c11 = _write_artifact(tmp_path)

    def _forbid_training_student(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Artifact load attempted to construct the C07/M90 training model.")

    monkeypatch.setattr(M2MMapFreeRecurrentStudent, "__init__", _forbid_training_student)
    policy = load_m2m_student_artifact(
        artifact,
        expected_sha256=receipt["artifact_sha256"],
        expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
    )
    assert policy(_observations()).shape == (3, _ACTION_DIM)


def test_runtime_age_semantics_mismatch_rejects_before_any_state_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, receipt, _c11 = _write_artifact(tmp_path)

    def _forbid_state_copy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Frame-age mismatch reached deployment state copying.")

    monkeypatch.setattr(
        student_artifact_module.M2MStudentOnlyPolicy,
        "load_state_dict",
        _forbid_state_copy,
    )
    with pytest.raises(ValueError, match="explicitly selected runtime contract"):
        load_m2m_student_artifact(
            artifact,
            expected_sha256=receipt["artifact_sha256"],
            expected_frame_age_semantics=_LIVOX_FRAME_AGE_SEMANTICS,
        )
    with pytest.raises(ValueError, match="explicitly selected runtime contract"):
        inspect_m2m_student_artifact(
            artifact,
            expected_sha256=receipt["artifact_sha256"],
            expected_frame_age_semantics=_LIVOX_FRAME_AGE_SEMANTICS,
        )


def test_legacy_artifact_missing_frame_age_semantics_is_rejected_before_state_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _receipt, _c11 = _write_artifact(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    del payload["network_config"]["frame_age_semantics"]
    legacy = tmp_path / "legacy-without-frame-age-semantics.pt"
    torch.save(payload, legacy)

    def _forbid_state_copy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Legacy artifact reached deployment state copying.")

    monkeypatch.setattr(
        student_artifact_module.M2MStudentOnlyPolicy,
        "load_state_dict",
        _forbid_state_copy,
    )
    with pytest.raises(ValueError, match=r"network config key mismatch.*frame_age_semantics"):
        load_m2m_student_artifact(
            legacy,
            expected_sha256=_sha256(legacy),
            expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
        )


def test_torchscript_and_onnx_wrapper_match_tensor_policy(tmp_path: Path) -> None:
    artifact, receipt, _c11 = _write_artifact(tmp_path)
    policy = load_m2m_student_artifact(
        artifact,
        expected_sha256=receipt["artifact_sha256"],
        expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
    )
    wrapper = policy.as_jit().eval()
    scripted = torch.jit.script(wrapper)
    assert wrapper.frame_age_semantics == _FRAME_AGE_SEMANTICS
    assert scripted.frame_age_semantics == _FRAME_AGE_SEMANTICS
    assert wrapper.input_names == ["proprio", "strict_frame", "hidden_state"]
    assert wrapper.output_names == ["action", "latent_A", "next_hidden_state"]
    assert policy.as_onnx().input_names == wrapper.input_names
    hidden = torch.zeros(1, 3, 20)
    for step in range(3):
        obs = _observations(step=step)
        expected = wrapper(obs["policy"], obs[_STRICT_SET], hidden)
        actual = scripted(obs["policy"], obs[_STRICT_SET], hidden)
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)
        hidden = actual[2]

    onnx_wrapper = policy.as_onnx().eval()
    onnx_path = tmp_path / "student-only.onnx"
    torch.onnx.export(
        onnx_wrapper,
        onnx_wrapper.get_dummy_inputs(),
        onnx_path,
        export_params=True,
        opset_version=18,
        external_data=False,
        input_names=onnx_wrapper.input_names,
        output_names=onnx_wrapper.output_names,
        dynamic_axes=onnx_wrapper.dynamic_axes,
    )
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    reference_obs = _observations()
    reference_inputs = {
        "proprio": reference_obs["policy"].numpy(),
        "strict_frame": reference_obs[_STRICT_SET].numpy(),
        "hidden_state": torch.zeros(1, 3, 20).numpy(),
    }
    reference_tensors = tuple(torch.from_numpy(value) for value in reference_inputs.values())
    expected_reference = wrapper(*reference_tensors)
    scripted_reference = scripted(*reference_tensors)
    export_wrapper_reference = onnx_wrapper(
        torch.from_numpy(reference_inputs["proprio"]),
        torch.from_numpy(reference_inputs["strict_frame"]),
        torch.from_numpy(reference_inputs["hidden_state"]),
    )
    for expected, scripted_value, export_value in zip(
        expected_reference,
        scripted_reference,
        export_wrapper_reference,
    ):
        torch.testing.assert_close(scripted_value, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(export_value, expected, rtol=0.0, atol=0.0)
    onnx_outputs = ReferenceEvaluator(onnx_model).run(None, reference_inputs)
    assert len(onnx_outputs) == len(expected_reference)
    for actual, expected in zip(onnx_outputs, expected_reference):
        torch.testing.assert_close(
            torch.from_numpy(actual),
            expected.detach().cpu(),
            rtol=1.0e-4,
            atol=1.0e-5,
        )


def test_artifact_hash_state_digest_and_dependency_corruption_fail_closed(tmp_path: Path) -> None:
    artifact, receipt, _c11 = _write_artifact(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_m2m_student_artifact(
            artifact,
            expected_sha256="0" * 64,
            expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
        )

    corrupted = tmp_path / "corrupt-bytes.pt"
    value = bytearray(artifact.read_bytes())
    value[len(value) // 2] ^= 0x01
    corrupted.write_bytes(value)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_m2m_student_artifact(
            corrupted,
            expected_sha256=receipt["artifact_sha256"],
            expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
        )

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    first_key = sorted(payload["student_state_dict"])[0]
    payload["student_state_dict"][first_key].view(-1)[0] += 1.0
    tampered_state = tmp_path / "tampered-state.pt"
    torch.save(payload, tampered_state)
    with pytest.raises(ValueError, match="state content digest differs"):
        load_m2m_student_artifact(
            tampered_state,
            expected_sha256=_sha256(tampered_state),
            expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
        )

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["dependency_receipt"]["constructs_teacher"] = True
    tampered_dependency = tmp_path / "tampered-dependency.pt"
    torch.save(payload, tampered_dependency)
    with pytest.raises(ValueError, match="dependency receipt differs"):
        load_m2m_student_artifact(
            tampered_dependency,
            expected_sha256=_sha256(tampered_dependency),
            expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
        )


def test_source_hashes_and_C11_M90_binding_are_verified_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m90, m90_sha = _m90_checkpoint(tmp_path)
    c11, c11_sha, _payload = _c11_checkpoint(tmp_path)
    kwargs = {
        "distillation_checkpoint_path": c11,
        "expected_distillation_sha256": c11_sha,
        "frozen_m90_checkpoint_path": m90,
        "expected_frozen_m90_sha256": m90_sha,
        "construction_config": _construction_config(),
    }
    missing_semantics = _construction_config()
    del missing_semantics["frame_age_semantics"]
    with pytest.raises(ValueError, match=r"construction config key mismatch.*frame_age_semantics"):
        build_m2m_student_artifact_payload(**{**kwargs, "construction_config": missing_semantics})
    invalid_semantics = _construction_config()
    invalid_semantics["frame_age_semantics"] = "message_age_s"
    with pytest.raises(ValueError, match="frame_age_semantics"):
        build_m2m_student_artifact_payload(**{**kwargs, "construction_config": invalid_semantics})
    with pytest.raises(ValueError, match="C11 distillation checkpoint SHA-256 mismatch"):
        build_m2m_student_artifact_payload(**{**kwargs, "expected_distillation_sha256": "0" * 64})
    with pytest.raises(ValueError, match="frozen M90 checkpoint SHA-256 mismatch"):
        build_m2m_student_artifact_payload(**{**kwargs, "expected_frozen_m90_sha256": "0" * 64})

    mismatched_c11, mismatched_sha, _ = _c11_checkpoint(tmp_path, frozen_receipt_sha="f" * 64)
    with pytest.raises(ValueError, match=r"frozen receipt.*differ"):
        build_m2m_student_artifact_payload(
            **{
                **kwargs,
                "distillation_checkpoint_path": mismatched_c11,
                "expected_distillation_sha256": mismatched_sha,
            }
        )

    def _forbid_deserialize(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Hash mismatch reached torch.load.")

    monkeypatch.setattr(torch, "load", _forbid_deserialize)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_m2m_student_artifact_payload(**{**kwargs, "expected_distillation_sha256": "0" * 64})


@pytest.mark.parametrize(
    ("semantic_change", "changed_value"),
    (
        ("frame_near_range_m", 0.2),
        ("frame_far_range_m", 5.5),
        ("frame_age_semantics", _LIVOX_FRAME_AGE_SEMANTICS),
        ("tokenizer_pooled_spatial_size", [1, 8]),
        ("frozen_actor_activation", "relu"),
    ),
)
def test_export_rejects_same_shape_but_different_C07_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_change: str,
    changed_value: object,
) -> None:
    m90, m90_sha = _m90_checkpoint(tmp_path)
    c11, c11_sha, _payload = _c11_checkpoint(tmp_path)
    config = _construction_config()
    if semantic_change == "frozen_actor_activation":
        config["frozen_ecmm_actor_cfg"]["activation"] = changed_value
    else:
        config[semantic_change] = changed_value

    def _forbid_trainable_weight_load(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Architecture mismatch reached C11 trainable-weight loading.")

    monkeypatch.setattr(
        student_artifact_module,
        "_load_c11_trainable_state",
        _forbid_trainable_weight_load,
    )

    # All four edits preserve parameter shapes: the two pooling layouts both
    # contain eight cells, range limits are normalization-only, and activation
    # is stateless.  The receipt must reject them before any C11 parameter copy.
    with pytest.raises(ValueError, match=r"architecture receipt.*differs"):
        build_m2m_student_artifact_payload(
            distillation_checkpoint_path=c11,
            expected_distillation_sha256=c11_sha,
            frozen_m90_checkpoint_path=m90,
            expected_frozen_m90_sha256=m90_sha,
            construction_config=config,
        )


def test_export_rejects_legacy_C11_without_C07_architecture_receipt(tmp_path: Path) -> None:
    m90, m90_sha = _m90_checkpoint(tmp_path)
    _c11, _c11_sha, source = _c11_checkpoint(tmp_path)
    legacy = copy.deepcopy(source)
    del legacy["config_receipt"]["student"]["architecture_receipt"]
    legacy_path = tmp_path / "legacy-c11-without-architecture-receipt.pt"
    torch.save(legacy, legacy_path)

    with pytest.raises(ValueError, match="architecture receipt is missing"):
        build_m2m_student_artifact_payload(
            distillation_checkpoint_path=legacy_path,
            expected_distillation_sha256=_sha256(legacy_path),
            frozen_m90_checkpoint_path=m90,
            expected_frozen_m90_sha256=m90_sha,
            construction_config=_construction_config(),
        )


def test_export_rejects_legacy_C11_without_frame_age_semantics(tmp_path: Path) -> None:
    m90, m90_sha = _m90_checkpoint(tmp_path)
    _c11, _c11_sha, source = _c11_checkpoint(tmp_path)
    legacy = copy.deepcopy(source)
    del legacy["config_receipt"]["student"]["architecture_receipt"]["strict_frame"][
        "frame_age_semantics"
    ]
    legacy_path = tmp_path / "legacy-c11-without-frame-age-semantics.pt"
    torch.save(legacy, legacy_path)

    with pytest.raises(ValueError, match=r"architecture receipt.*differs"):
        build_m2m_student_artifact_payload(
            distillation_checkpoint_path=legacy_path,
            expected_distillation_sha256=_sha256(legacy_path),
            frozen_m90_checkpoint_path=m90,
            expected_frozen_m90_sha256=m90_sha,
            construction_config=_construction_config(),
        )


def test_cli_exports_create_only_artifact_and_strict_inspection(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    m90, m90_sha = _m90_checkpoint(tmp_path)
    c11, c11_sha, _payload = _c11_checkpoint(tmp_path)
    config_path = tmp_path / "construction.json"
    config_path.write_text(json.dumps(_construction_config()), encoding="utf-8")
    output = tmp_path / "cli-student-only.pt"
    arguments = [
        "export",
        "--distillation-checkpoint",
        str(c11),
        "--distillation-sha256",
        c11_sha,
        "--frozen-m90-checkpoint",
        str(m90),
        "--frozen-m90-sha256",
        m90_sha,
        "--construction-config",
        str(config_path),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["artifact_path"] == str(output.resolve())
    assert exported["artifact_sha256"] == _sha256(output)
    with pytest.raises(FileExistsError):
        main(arguments)

    assert main(
        [
            "inspect",
            "--artifact",
            str(output),
            "--sha256",
            exported["artifact_sha256"],
            "--frame-age-semantics",
            _FRAME_AGE_SEMANTICS,
        ]
    ) == 0
    inspected_stdout = json.loads(capsys.readouterr().out)
    inspected = inspect_m2m_student_artifact(
        output,
        expected_sha256=exported["artifact_sha256"],
        expected_frame_age_semantics=_FRAME_AGE_SEMANTICS,
    )
    assert inspected_stdout == inspected
    assert inspected["dependency_audit"]["constructs_teacher"] is False

    torchscript_output = tmp_path / "cli-student-only.jit"
    onnx_output = tmp_path / "cli-student-only.onnx"
    compile_arguments = [
        "compile",
        "--artifact",
        str(output),
        "--sha256",
        exported["artifact_sha256"],
        "--frame-age-semantics",
        _FRAME_AGE_SEMANTICS,
        "--torchscript-output",
        str(torchscript_output),
        "--onnx-output",
        str(onnx_output),
    ]
    assert main(compile_arguments) == 0
    compiled = json.loads(capsys.readouterr().out)
    assert set(compiled["backends"]) == {"torchscript", "onnx"}
    assert compiled["frame_age_semantics"] == _FRAME_AGE_SEMANTICS
    assert compiled["backends"]["torchscript"]["interface"]["frame_age_semantics"] == (
        _FRAME_AGE_SEMANTICS
    )
    assert compiled["backends"]["onnx"]["interface"]["frame_age_semantics"] == (
        _FRAME_AGE_SEMANTICS
    )
    assert compiled["backends"]["torchscript"]["sha256"] == _sha256(torchscript_output)
    assert compiled["backends"]["onnx"]["sha256"] == _sha256(onnx_output)
    compiled_onnx = onnx.load(onnx_output)
    onnx.checker.check_model(compiled_onnx)
    assert {entry.key: entry.value for entry in compiled_onnx.metadata_props}[
        "m2m.frame_age_semantics"
    ] == _FRAME_AGE_SEMANTICS
    scripted = torch.jit.load(str(torchscript_output))
    assert scripted.frame_age_semantics == _FRAME_AGE_SEMANTICS
    scripted_outputs = scripted(
        torch.zeros(1, _PROPRIO_DIM),
        torch.zeros(1, 1, 4, 16, 96),
        torch.zeros(1, 1, 20),
    )
    assert [tuple(value.shape) for value in scripted_outputs] == [(1, 29), (1, 64), (1, 1, 20)]
    with pytest.raises(FileExistsError):
        main(compile_arguments)
