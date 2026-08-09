# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build, load, inspect, and export a standalone C13 M2M student artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import torch
from collections.abc import Mapping
from pathlib import Path
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.m2m_recurrent_student import M2MMapFreeRecurrentStudent, M2MStrictFrameTokenizer
from rsl_rl.models.m2m_student_only import (
    M2MStudentOnlyPolicy,
    normalize_m2m_student_network_config,
)

_ARTIFACT_SCHEMA = "m2m_student_only_artifact_v1"
_CONSTRUCTION_SCHEMA = "m2m_student_export_construction_v1"
_C11_SCHEMA = "m2m_latent_action_distillation_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TRAINABLE_PREFIXES = ("frame_tokenizer.", "gru.", "current_encoder.", "latent_head.")
_STATE_PREFIXES = (
    "frame_tokenizer.",
    "gru.",
    "current_encoder.",
    "latent_head.",
    "obs_normalizer.",
    "prop_mlp.",
    "action_head.",
    "distribution.",
)
_DEPENDENCY_RECEIPT = {
    "constructs_teacher": False,
    "constructs_mapper": False,
    "constructs_teacher_observation": False,
    "contains_m90_perception_encoder": False,
    "contains_optimizer_state": False,
    "contains_teacher_latent_labels": False,
    "contains_teacher_action_labels": False,
    "external_checkpoint_required_at_runtime": False,
}


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly 64 lowercase hexadecimal characters.")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: object, *, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON-compatible metadata.") from error
    return _sha256_bytes(encoded)


def _read_verified_bytes(path: str | Path, expected_sha256: str, *, label: str) -> tuple[Path, bytes]:
    digest = _validate_sha256(expected_sha256, label=f"{label} expected_sha256")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    value = resolved.read_bytes()
    actual = _sha256_bytes(value)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: expected={digest}, actual={actual}.")
    return resolved, value


def _weights_only_mapping(value: bytes, *, label: str) -> dict[str, Any]:
    loaded = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
    if type(loaded) is not dict:
        raise ValueError(f"{label} root must be an exact dictionary.")
    return loaded


def _state_content_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Student-only state must map string keys to tensors.")
        contiguous = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_export_construction_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("C13 export construction config must be a mapping.")
    expected = {
        "schema",
        "obs_set",
        "output_dim",
        "strict_frame_set",
        "proprio_sets",
        "proprio_group_dims",
        "frozen_ecmm_actor_cfg",
        "frame_near_range_m",
        "frame_far_range_m",
        "frame_message_period_s",
        "frame_max_age_s",
        "tokenizer_hidden_channels",
        "tokenizer_dim",
        "tokenizer_pooled_spatial_size",
        "temporal_mode",
        "gru_hidden_dim",
        "gru_num_layers",
        "latent_hidden_dim",
    }
    missing = expected.difference(config)
    unexpected = set(config).difference(expected)
    if missing or unexpected:
        raise ValueError(
            "C13 export construction config key mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    if config["schema"] != _CONSTRUCTION_SCHEMA:
        raise ValueError(f"Unsupported C13 construction schema {config['schema']!r}.")
    actor_cfg = config["frozen_ecmm_actor_cfg"]
    if not isinstance(actor_cfg, Mapping):
        raise ValueError("frozen_ecmm_actor_cfg must be a mapping.")
    required_actor = {
        "hidden_dims",
        "activation",
        "obs_normalization",
        "distribution_cfg",
        "prop_feature_dim",
        "prop_hidden_dims",
        "use_prop_encoder",
        "vision_feature_dim",
    }
    missing_actor = required_actor.difference(actor_cfg)
    if missing_actor:
        raise ValueError(f"frozen_ecmm_actor_cfg is missing control fields {sorted(missing_actor)}.")

    # Round-tripping through strict JSON removes tuples and rejects Python
    # objects, NaN, or Infinity before config becomes artifact metadata.
    try:
        normalized = json.loads(json.dumps(copy.deepcopy(dict(config)), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("C13 export construction config must be finite JSON metadata.") from error
    network = _network_config_from_construction(normalized)
    normalize_m2m_student_network_config(network)
    return normalized


def _network_config_from_construction(config: Mapping[str, Any]) -> dict[str, Any]:
    actor_cfg = config["frozen_ecmm_actor_cfg"]
    return normalize_m2m_student_network_config(
        {
            "schema": "m2m_student_only_network_v1",
            "obs_set": config["obs_set"],
            "output_dim": config["output_dim"],
            "strict_frame_set": config["strict_frame_set"],
            "proprio_sets": config["proprio_sets"],
            "proprio_group_dims": config["proprio_group_dims"],
            "frame_near_range_m": config["frame_near_range_m"],
            "frame_far_range_m": config["frame_far_range_m"],
            "frame_message_period_s": config["frame_message_period_s"],
            "frame_max_age_s": config["frame_max_age_s"],
            "tokenizer_hidden_channels": config["tokenizer_hidden_channels"],
            "tokenizer_dim": config["tokenizer_dim"],
            "tokenizer_pooled_spatial_size": config["tokenizer_pooled_spatial_size"],
            "temporal_mode": config["temporal_mode"],
            "gru_hidden_dim": config["gru_hidden_dim"],
            "gru_num_layers": config["gru_num_layers"],
            "latent_hidden_dim": config["latent_hidden_dim"],
            "control": {
                "obs_normalization": actor_cfg["obs_normalization"],
                "activation": actor_cfg["activation"],
                "prop_feature_dim": actor_cfg["prop_feature_dim"],
                "prop_hidden_dims": actor_cfg["prop_hidden_dims"],
                "use_prop_encoder": actor_cfg["use_prop_encoder"],
                "vision_feature_dim": actor_cfg["vision_feature_dim"],
                "fusion_hidden_dims": actor_cfg["hidden_dims"],
                "distribution_cfg": actor_cfg["distribution_cfg"],
            },
        }
    )


def _construction_observations(config: Mapping[str, Any]) -> TensorDict:
    data = {
        group: torch.zeros(1, int(config["proprio_group_dims"][group]), dtype=torch.float32)
        for group in config["proprio_sets"]
    }
    data[config["strict_frame_set"]] = torch.zeros(
        1,
        *M2MStrictFrameTokenizer.frame_shape,
        dtype=torch.float32,
    )
    return TensorDict(data, batch_size=[1])


def _validate_c11_source(
    checkpoint: dict[str, Any],
    *,
    student: M2MMapFreeRecurrentStudent,
    expected_m90_sha256: str,
    actor_state_dict_key: str,
) -> dict[str, torch.Tensor]:
    required = {
        "schema",
        "config_receipt",
        "student_trainable_state_dict",
        "optimizer_state_dict",
        "algorithm_iteration",
        "frozen_artifact_receipt",
    }
    allowed_extras = {"iter", "infos", "training_receipt"}
    missing = required.difference(checkpoint)
    unexpected = set(checkpoint).difference(required | allowed_extras)
    if missing or unexpected:
        raise ValueError(
            f"C11 source key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    if checkpoint["schema"] != _C11_SCHEMA:
        raise ValueError(f"Unsupported C11 source schema {checkpoint['schema']!r}.")
    iteration = checkpoint["algorithm_iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("C11 algorithm_iteration must be an exact non-negative integer.")

    frozen_receipt = checkpoint["frozen_artifact_receipt"]
    if not isinstance(frozen_receipt, Mapping):
        raise ValueError("C11 frozen_artifact_receipt must be a mapping.")
    if frozen_receipt.get("checkpoint_sha256") != expected_m90_sha256:
        raise ValueError("C11 frozen receipt and explicitly selected M90 SHA-256 differ.")
    if frozen_receipt.get("actor_state_dict_key") != actor_state_dict_key:
        raise ValueError("C11 frozen receipt and selected M90 actor state key differ.")
    if frozen_receipt.get("checkpoint_bytes_saved") is not False:
        raise ValueError("C11 frozen receipt must state checkpoint_bytes_saved=False.")

    config_receipt = checkpoint["config_receipt"]
    if not isinstance(config_receipt, Mapping):
        raise ValueError("C11 config_receipt must be a mapping.")
    student_receipt = config_receipt.get("student")
    if not isinstance(student_receipt, Mapping):
        raise ValueError("C11 config receipt is missing its student contract.")
    expected_class = "rsl_rl.models.m2m_recurrent_student.M2MMapFreeRecurrentStudent"
    if student_receipt.get("class") != expected_class:
        raise ValueError("C11 source was not trained with the formal C07 student class.")
    if student_receipt.get("temporal_mode") != student.temporal_mode:
        raise ValueError("C11 source temporal mode differs from export construction config.")
    source_architecture = student_receipt.get("architecture_receipt")
    live_architecture = student.architecture_receipt()
    if not isinstance(source_architecture, Mapping) or source_architecture != live_architecture:
        raise ValueError(
            "C11 source C07 architecture receipt is missing or differs from export construction config."
        )
    allowed_keys = student_receipt.get("allowed_observation_keys")
    if not isinstance(allowed_keys, list) or set(allowed_keys) != set(student.obs_groups):
        raise ValueError("C11 source deployment observation allowlist differs from C13 config.")

    state = checkpoint["student_trainable_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("C11 student_trainable_state_dict must be a mapping.")
    live = {name: value for name, value in student.named_parameters() if value.requires_grad}
    if set(state) != set(live):
        raise ValueError("C11 trainable state keys differ from the reconstructed C07 student.")
    receipt_names = student_receipt.get("trainable_parameter_names")
    if not isinstance(receipt_names, list) or set(receipt_names) != set(live):
        raise ValueError("C11 trainable-parameter receipt differs from its state dictionary.")
    for name, parameter in live.items():
        if not name.startswith(_TRAINABLE_PREFIXES):
            raise ValueError(f"C11 source contains non-student trainable parameter {name!r}.")
        saved = state[name]
        if not isinstance(saved, torch.Tensor):
            raise ValueError(f"C11 trainable state {name!r} must be a tensor.")
        if saved.shape != parameter.shape or saved.dtype != parameter.dtype or not torch.isfinite(saved).all():
            raise ValueError(f"C11 trainable state {name!r} shape/dtype/finiteness differs.")
    return dict(state)


def _copy_training_student_to_deployment(
    student: M2MMapFreeRecurrentStudent,
    network_config: Mapping[str, Any],
) -> M2MStudentOnlyPolicy:
    policy = M2MStudentOnlyPolicy(network_config)
    component_pairs = (
        (policy.frame_tokenizer, student.frame_tokenizer, "frame_tokenizer"),
        (policy.obs_normalizer, student.ecmm_core.actor.obs_normalizer, "obs_normalizer_B"),
        (policy.prop_mlp, student.ecmm_core.actor.prop_mlp, "proprio_encoder_B"),
        (policy.action_head, student.ecmm_core.actor.mlp, "fusion_action_head_C"),
        (policy.distribution, student.ecmm_core.actor.distribution, "action_distribution"),
        (policy.latent_head, student.latent_head, "latent_head"),
    )
    for target, source, label in component_pairs:
        if source is None:
            raise ValueError(f"Training student is missing required deployment component {label}.")
        target.load_state_dict(source.state_dict(), strict=True)
    if policy.is_recurrent:
        if policy.gru is None or student.gru is None:
            raise ValueError("GRU source/deployment components are inconsistent.")
        policy.gru.load_state_dict(student.gru.state_dict(), strict=True)
    else:
        if policy.current_encoder is None or student.current_encoder is None:
            raise ValueError("Current-frame source/deployment components are inconsistent.")
        policy.current_encoder.load_state_dict(student.current_encoder.state_dict(), strict=True)
    policy.requires_grad_(False)
    policy.eval()
    audit = policy.dependency_audit()
    if audit["forbidden_state_keys"] or audit["forbidden_module_names"]:
        raise RuntimeError("Standalone policy unexpectedly retained a training-only dependency.")
    return policy


def _load_c11_trainable_state(
    student: M2MMapFreeRecurrentStudent,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Apply already-validated C11 tensors only after provenance validation."""
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(state[name].to(device=parameter.device))


def build_m2m_student_artifact_payload(
    *,
    distillation_checkpoint_path: str | Path,
    expected_distillation_sha256: str,
    frozen_m90_checkpoint_path: str | Path,
    expected_frozen_m90_sha256: str,
    construction_config: Mapping[str, Any],
    actor_state_dict_key: str = "actor_state_dict",
) -> dict[str, Any]:
    """Build an in-memory standalone payload from verified C11 and M90 bytes."""
    if not isinstance(actor_state_dict_key, str) or not actor_state_dict_key:
        raise ValueError("actor_state_dict_key must be a non-empty string.")
    m90_digest = _validate_sha256(expected_frozen_m90_sha256, label="frozen M90 expected SHA-256")
    normalized_config = _normalize_export_construction_config(construction_config)
    network_config = _network_config_from_construction(normalized_config)
    _distill_path, distill_bytes = _read_verified_bytes(
        distillation_checkpoint_path,
        expected_distillation_sha256,
        label="C11 distillation checkpoint",
    )
    m90_path, _m90_bytes = _read_verified_bytes(
        frozen_m90_checkpoint_path,
        m90_digest,
        label="frozen M90 checkpoint",
    )
    c11 = _weights_only_mapping(distill_bytes, label="C11 distillation checkpoint")
    obs = _construction_observations(normalized_config)
    student = M2MMapFreeRecurrentStudent(
        obs,
        {normalized_config["obs_set"]: [
            *normalized_config["proprio_sets"],
            normalized_config["strict_frame_set"],
        ]},
        normalized_config["obs_set"],
        normalized_config["output_dim"],
        strict_frame_set=normalized_config["strict_frame_set"],
        proprio_sets=normalized_config["proprio_sets"],
        frozen_ecmm_checkpoint_path=str(m90_path),
        frozen_ecmm_expected_sha256=m90_digest,
        frozen_ecmm_actor_cfg=normalized_config["frozen_ecmm_actor_cfg"],
        frozen_ecmm_actor_state_dict_key=actor_state_dict_key,
        frame_near_range_m=normalized_config["frame_near_range_m"],
        frame_far_range_m=normalized_config["frame_far_range_m"],
        frame_message_period_s=normalized_config["frame_message_period_s"],
        frame_max_age_s=normalized_config["frame_max_age_s"],
        tokenizer_hidden_channels=normalized_config["tokenizer_hidden_channels"],
        tokenizer_dim=normalized_config["tokenizer_dim"],
        tokenizer_pooled_spatial_size=tuple(normalized_config["tokenizer_pooled_spatial_size"]),
        temporal_mode=normalized_config["temporal_mode"],
        gru_hidden_dim=normalized_config["gru_hidden_dim"],
        gru_num_layers=normalized_config["gru_num_layers"],
        latent_hidden_dim=normalized_config["latent_hidden_dim"],
    )
    trainable_state = _validate_c11_source(
        c11,
        student=student,
        expected_m90_sha256=m90_digest,
        actor_state_dict_key=actor_state_dict_key,
    )
    _load_c11_trainable_state(student, trainable_state)
    student.eval()
    policy = _copy_training_student_to_deployment(student, network_config)

    # Verify that stripping the unused M90 encoder did not alter one-step
    # student behavior.  Both paths start with a zero recurrent state.
    student.reset()
    policy.reset()
    with torch.no_grad():
        expected_action, expected_latent = student.forward_with_latent(obs)
        actual_action = policy(obs)
        actual_hidden = policy.get_hidden_state()
        policy.reset()
        proprio = torch.cat([obs[group] for group in policy.proprio_sets], dim=-1)
        direct_action, actual_latent, direct_hidden = policy.step_tensors(
            proprio,
            obs[policy.strict_frame_set],
        )
    torch.testing.assert_close(actual_action, expected_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(direct_action, expected_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_latent, expected_latent, rtol=0.0, atol=0.0)
    if policy.is_recurrent:
        torch.testing.assert_close(actual_hidden, direct_hidden, rtol=0.0, atol=0.0)

    state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    invalid_state_keys = [key for key in state if not key.startswith(_STATE_PREFIXES)]
    if invalid_state_keys:
        raise RuntimeError(f"Student-only state contains unexpected component keys {invalid_state_keys}.")
    state_sha = _state_content_sha256(state)
    config_sha = _canonical_json_sha256(network_config, label="student-only network config")
    c11_config_sha = _canonical_json_sha256(c11["config_receipt"], label="C11 config receipt")
    source_receipt = {
        "distillation_checkpoint_sha256": _sha256_bytes(distill_bytes),
        "distillation_schema": c11["schema"],
        "distillation_algorithm_iteration": c11["algorithm_iteration"],
        "distillation_config_receipt_sha256": c11_config_sha,
        "frozen_m90_checkpoint_sha256": m90_digest,
        "frozen_m90_actor_state_dict_key": actor_state_dict_key,
        "source_paths_embedded": False,
        "source_checkpoint_bytes_embedded": False,
    }
    input_receipt = {
        "ordered_groups": list(policy.obs_groups),
        "proprio_sets": list(policy.proprio_sets),
        "proprio_group_dims": dict(policy.proprio_group_dims),
        "strict_frame_set": policy.strict_frame_set,
        "strict_frame_shape": list(M2MStrictFrameTokenizer.frame_shape),
        "strict_frame_channels": list(M2MStrictFrameTokenizer.channels),
        "recurrent_hidden_shape": (
            [policy.gru_num_layers, "batch", policy.gru_hidden_dim] if policy.is_recurrent else False
        ),
    }
    return {
        "schema": _ARTIFACT_SCHEMA,
        "network_config": network_config,
        "network_config_sha256": config_sha,
        "input_receipt": input_receipt,
        "dependency_receipt": copy.deepcopy(_DEPENDENCY_RECEIPT),
        "source_receipt": source_receipt,
        "state_receipt": {
            "content_sha256": state_sha,
            "keys": sorted(state),
            "tensor_count": len(state),
        },
        "student_state_dict": state,
    }


def _validate_artifact_payload(payload: dict[str, Any]) -> M2MStudentOnlyPolicy:
    expected = {
        "schema",
        "network_config",
        "network_config_sha256",
        "input_receipt",
        "dependency_receipt",
        "source_receipt",
        "state_receipt",
        "student_state_dict",
    }
    if set(payload) != expected:
        raise ValueError("Student-only artifact top-level keys differ from the C13 schema.")
    if payload["schema"] != _ARTIFACT_SCHEMA:
        raise ValueError(f"Unsupported C13 artifact schema {payload['schema']!r}.")
    network_config = normalize_m2m_student_network_config(payload["network_config"])
    expected_config_sha = _canonical_json_sha256(network_config, label="artifact network config")
    if payload["network_config_sha256"] != expected_config_sha:
        raise ValueError("Student-only artifact network config digest differs.")
    if payload["dependency_receipt"] != _DEPENDENCY_RECEIPT:
        raise ValueError("Student-only artifact dependency receipt differs or claims training-only dependencies.")
    source_receipt = payload["source_receipt"]
    if not isinstance(source_receipt, Mapping):
        raise ValueError("Student-only source_receipt must be a mapping.")
    expected_source_keys = {
        "distillation_checkpoint_sha256",
        "distillation_schema",
        "distillation_algorithm_iteration",
        "distillation_config_receipt_sha256",
        "frozen_m90_checkpoint_sha256",
        "frozen_m90_actor_state_dict_key",
        "source_paths_embedded",
        "source_checkpoint_bytes_embedded",
    }
    if set(source_receipt) != expected_source_keys:
        raise ValueError("Student-only source_receipt keys differ from the C13 schema.")
    _validate_sha256(source_receipt.get("distillation_checkpoint_sha256"), label="source C11 SHA-256")
    _validate_sha256(source_receipt.get("frozen_m90_checkpoint_sha256"), label="source M90 SHA-256")
    _validate_sha256(source_receipt.get("distillation_config_receipt_sha256"), label="C11 config SHA-256")
    if source_receipt.get("distillation_schema") != _C11_SCHEMA:
        raise ValueError("Student-only source receipt has an unsupported distillation schema.")
    source_iteration = source_receipt.get("distillation_algorithm_iteration")
    if type(source_iteration) is not int or source_iteration < 0:
        raise ValueError("Student-only source receipt iteration must be a non-negative integer.")
    source_state_key = source_receipt.get("frozen_m90_actor_state_dict_key")
    if not isinstance(source_state_key, str) or not source_state_key:
        raise ValueError("Student-only source receipt requires the external M90 actor state key.")
    if source_receipt.get("source_paths_embedded") is not False:
        raise ValueError("Student-only artifact must not embed source paths.")
    if source_receipt.get("source_checkpoint_bytes_embedded") is not False:
        raise ValueError("Student-only artifact must not embed source checkpoint bytes.")

    state = payload["student_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("student_state_dict must be a mapping.")
    if any(not isinstance(key, str) or not key.startswith(_STATE_PREFIXES) for key in state):
        raise ValueError("Student-only artifact state contains a forbidden component key.")
    receipt = payload["state_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != {"content_sha256", "keys", "tensor_count"}:
        raise ValueError("Student-only state receipt is malformed.")
    if receipt["keys"] != sorted(state) or receipt["tensor_count"] != len(state):
        raise ValueError("Student-only state key/count receipt differs.")
    if receipt["content_sha256"] != _state_content_sha256(state):
        raise ValueError("Student-only artifact state content digest differs.")

    policy = M2MStudentOnlyPolicy(network_config)
    expected_state = policy.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("Student-only artifact state keys differ from its network config.")
    for key, expected_value in expected_state.items():
        saved = state[key]
        if not isinstance(saved, torch.Tensor):
            raise ValueError(f"Student-only state {key!r} must be a tensor.")
        if saved.shape != expected_value.shape or saved.dtype != expected_value.dtype:
            raise ValueError(f"Student-only state {key!r} shape/dtype differs from config.")
        if saved.is_floating_point() and not torch.isfinite(saved).all():
            raise ValueError(f"Student-only state {key!r} contains non-finite values.")
    policy.load_state_dict(state, strict=True)
    policy.requires_grad_(False)
    policy.eval()
    audit = policy.dependency_audit()
    if audit["forbidden_state_keys"] or audit["forbidden_module_names"]:
        raise ValueError("Loaded student-only policy contains a forbidden training-only dependency.")

    expected_input = {
        "ordered_groups": list(policy.obs_groups),
        "proprio_sets": list(policy.proprio_sets),
        "proprio_group_dims": dict(policy.proprio_group_dims),
        "strict_frame_set": policy.strict_frame_set,
        "strict_frame_shape": list(M2MStrictFrameTokenizer.frame_shape),
        "strict_frame_channels": list(M2MStrictFrameTokenizer.channels),
        "recurrent_hidden_shape": (
            [policy.gru_num_layers, "batch", policy.gru_hidden_dim] if policy.is_recurrent else False
        ),
    }
    if payload["input_receipt"] != expected_input:
        raise ValueError("Student-only artifact input receipt differs from the reconstructed policy.")
    policy.artifact_receipt = {
        "schema": payload["schema"],
        "network_config_sha256": payload["network_config_sha256"],
        "input_receipt": copy.deepcopy(payload["input_receipt"]),
        "dependency_receipt": copy.deepcopy(payload["dependency_receipt"]),
        "source_receipt": copy.deepcopy(payload["source_receipt"]),
        "state_receipt": copy.deepcopy(payload["state_receipt"]),
    }
    return policy


def serialize_m2m_student_artifact(payload: dict[str, Any]) -> bytes:
    """Validate and serialize an artifact using weights-only-safe values."""
    _validate_artifact_payload(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def write_m2m_student_artifact(payload: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Create a new artifact file without overwriting an existing result."""
    value = serialize_m2m_student_artifact(payload)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as artifact_file:
        artifact_file.write(value)
        artifact_file.flush()
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": _sha256_bytes(value),
        "artifact_bytes": len(value),
        "schema": _ARTIFACT_SCHEMA,
    }


def load_m2m_student_artifact(
    artifact_path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> M2MStudentOnlyPolicy:
    """Hash-verify, strict-load, and return a standalone deployment policy."""
    _path, value = _read_verified_bytes(artifact_path, expected_sha256, label="C13 student-only artifact")
    payload = _weights_only_mapping(value, label="C13 student-only artifact")
    policy = _validate_artifact_payload(payload)
    return policy.to(device).eval()


def inspect_m2m_student_artifact(
    artifact_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Return verified artifact and dependency receipts without running policy inference."""
    policy = load_m2m_student_artifact(artifact_path, expected_sha256=expected_sha256, device="cpu")
    return {
        "artifact_sha256": expected_sha256,
        "artifact_receipt": copy.deepcopy(policy.artifact_receipt),
        "dependency_audit": policy.dependency_audit(),
    }


def export_m2m_student_backends(
    artifact_path: str | Path,
    *,
    expected_sha256: str,
    torchscript_output: str | Path | None = None,
    onnx_output: str | Path | None = None,
    onnx_opset_version: int = 18,
) -> dict[str, Any]:
    """Create standalone TorchScript/ONNX files from a verified artifact."""
    if torchscript_output is None and onnx_output is None:
        raise ValueError("At least one TorchScript or ONNX output path is required.")
    if type(onnx_opset_version) is not int or onnx_opset_version < 18:
        raise ValueError("C13 ONNX opset_version must be an integer >= 18.")
    outputs = [Path(value).expanduser() for value in (torchscript_output, onnx_output) if value is not None]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("TorchScript and ONNX output paths must be distinct.")
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"C13 backend outputs already exist: {existing}.")

    policy = load_m2m_student_artifact(
        artifact_path,
        expected_sha256=expected_sha256,
        device="cpu",
    )
    generated: list[tuple[str, Path, bytes, dict[str, Any]]] = []
    if torchscript_output is not None:
        wrapper = policy.as_jit().eval()
        scripted = torch.jit.script(wrapper)
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        generated.append(
            (
                "torchscript",
                Path(torchscript_output).expanduser(),
                buffer.getvalue(),
                {
                    "input_names": list(wrapper.input_names),
                    "output_names": list(wrapper.output_names),
                    "dynamic_axes": copy.deepcopy(wrapper.dynamic_axes),
                },
            )
        )
    if onnx_output is not None:
        wrapper = policy.as_onnx().eval()
        buffer = io.BytesIO()
        torch.onnx.export(
            wrapper,
            wrapper.get_dummy_inputs(),
            buffer,  # pyright: ignore[reportArgumentType] - PyTorch accepts file-like buffers.
            export_params=True,
            opset_version=onnx_opset_version,
            external_data=False,
            input_names=wrapper.input_names,
            output_names=wrapper.output_names,
            dynamic_axes=wrapper.dynamic_axes,
        )
        generated.append(
            (
                "onnx",
                Path(onnx_output).expanduser(),
                buffer.getvalue(),
                {
                    "opset_version": onnx_opset_version,
                    "input_names": list(wrapper.input_names),
                    "output_names": list(wrapper.output_names),
                    "dynamic_axes": copy.deepcopy(wrapper.dynamic_axes),
                },
            )
        )

    backend_receipts: dict[str, Any] = {}
    for backend, path, value, interface in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output_file:
            output_file.write(value)
            output_file.flush()
        backend_receipts[backend] = {
            "path": str(path.resolve()),
            "sha256": _sha256_bytes(value),
            "bytes": len(value),
            "interface": interface,
        }
    return {
        "source_artifact_sha256": expected_sha256,
        "backends": backend_receipts,
    }


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if type(loaded) is not dict:
        raise ValueError("C13 construction config JSON root must be an exact object.")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or inspect a standalone M2M student artifact.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Build a create-only C13 artifact.")
    export.add_argument("--distillation-checkpoint", required=True)
    export.add_argument("--distillation-sha256", required=True)
    export.add_argument("--frozen-m90-checkpoint", required=True)
    export.add_argument("--frozen-m90-sha256", required=True)
    export.add_argument("--actor-state-dict-key", default="actor_state_dict")
    export.add_argument("--construction-config", required=True)
    export.add_argument("--output", required=True)
    inspect = subparsers.add_parser("inspect", help="Strictly inspect an existing C13 artifact.")
    inspect.add_argument("--artifact", required=True)
    inspect.add_argument("--sha256", required=True)
    compile_parser = subparsers.add_parser(
        "compile",
        help="Export create-only TorchScript and/or ONNX deployment files.",
    )
    compile_parser.add_argument("--artifact", required=True)
    compile_parser.add_argument("--sha256", required=True)
    compile_parser.add_argument("--torchscript-output")
    compile_parser.add_argument("--onnx-output")
    compile_parser.add_argument("--onnx-opset-version", type=int, default=18)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        payload = build_m2m_student_artifact_payload(
            distillation_checkpoint_path=args.distillation_checkpoint,
            expected_distillation_sha256=args.distillation_sha256,
            frozen_m90_checkpoint_path=args.frozen_m90_checkpoint,
            expected_frozen_m90_sha256=args.frozen_m90_sha256,
            construction_config=_load_json_mapping(args.construction_config),
            actor_state_dict_key=args.actor_state_dict_key,
        )
        result = write_m2m_student_artifact(payload, args.output)
    elif args.command == "inspect":
        result = inspect_m2m_student_artifact(args.artifact, expected_sha256=args.sha256)
    else:
        result = export_m2m_student_backends(
            args.artifact,
            expected_sha256=args.sha256,
            torchscript_output=args.torchscript_output,
            onnx_output=args.onnx_output,
            onnx_opset_version=args.onnx_opset_version,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_m2m_student_artifact_payload",
    "export_m2m_student_backends",
    "inspect_m2m_student_artifact",
    "load_m2m_student_artifact",
    "serialize_m2m_student_artifact",
    "write_m2m_student_artifact",
]
