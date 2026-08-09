# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed deployment manifest for the G1 MID-360 ray-time policy.

The exported TorchScript/ONNX graph only describes tensor shapes.  It cannot
describe the semantic order of the 96 proprioceptive values, the G1 joint
order, or how raw policy actions become joint targets.  This module stores
those assumptions in a versioned JSON sidecar and validates every field before
deployment.

The implementation intentionally depends only on the Python standard library.
It can therefore be reused by a robot-side launcher without importing
IsaacLab, PyTorch, or the training task.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RAY_TIME_MANIFEST_SCHEMA_NAME = "g1-mid360-ray-time-policy"
RAY_TIME_MANIFEST_SCHEMA_VERSION = 2
RAY_TIME_PROVENANCE_REPOSITORIES = ("lab_pro", "rsl_rl", "IsaacLab")

_SHA256_HEX_LENGTH = 64
_LEGACY_MANIFEST_SCHEMA_VERSION = 1
_FUSION_MODE_TO_VARIANT = {
    "attention": "Attention",
    "global": "Global",
    "query_global": "QueryGlobal",
}
_FUSION_MODE_TO_LEGACY_BOOL = {
    "attention": True,
    "global": False,
    # QueryGlobal was introduced with the legacy flag left enabled so that
    # old checkpoint/module layouts remain unchanged.  The explicit mode is
    # authoritative, but the redundant boolean must still be self-consistent.
    "query_global": True,
}
_ENCODER_VARIANTS = tuple(_FUSION_MODE_TO_VARIANT.values())
_LEGACY_ENCODER_VARIANTS = ("Global", "Attention")
_LEGACY_UNOBSERVED_RAY_SEMANTIC = (
    "[0.0, 0.0]; schema v1 intentionally conflates these cases"
)
_UNOBSERVED_RAY_SEMANTIC = (
    "[0.0, 0.0]; the ray-time contract intentionally conflates these cases"
)
_PROPRIO_TERM_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "velocity_commands",
    "previous_environment_action",
)
_PROPRIO_TERM_SIZES = (3, 3, 29, 29, 3, 29)
_PROPRIO_TERM_SEMANTICS = (
    "base angular velocity in the robot base frame, xyz",
    "gravity direction projected into the robot base frame, xyz",
    "joint position minus default joint position",
    "joint velocity relative to the default joint velocity",
    "commanded base linear x, linear y, and angular z velocity",
    "previous a_env after the optional actor-output clip and before action scale and offset",
)
_PROPRIO_TERM_INDEX_REFERENCES = (
    None,
    None,
    "action.joint_order",
    "action.joint_order",
    None,
    "action.joint_order",
)
_RAY_CHANNELS = (
    (0, "range_m", "metric ray range; zero exactly when the ray is unknown"),
    (1, "hit_mask", "binary validity mask; one means range_m is observed"),
)
_ENVIRONMENT_TO_JOINT_FORMULA = (
    "q_des[i] = clamp(default_joint_positions[i] + "
    "per_joint_action_scale[i] * a_env[i], "
    "processed_target_clip_rad[0], processed_target_clip_rad[1])"
)
_PREVIOUS_ACTION_SEMANTICS = (
    "the previous control step's 29-dimensional a_env passed into the action "
    "manager, after the optional RslRlVecEnvWrapper actor-output clip and "
    "before per-joint scale and default-position offset"
)
_TRAINING_POLICY_OBSERVATION_TERMS = (
    (
        "base_ang_vel",
        "isaaclab.envs.mdp.observations.base_ang_vel",
        False,
        {},
    ),
    (
        "projected_gravity",
        "isaaclab.envs.mdp.observations.projected_gravity",
        False,
        {},
    ),
    (
        "joint_pos",
        "isaaclab.envs.mdp.observations.joint_pos_rel",
        True,
        {"asset_name": "robot"},
    ),
    (
        "joint_vel",
        "isaaclab.envs.mdp.observations.joint_vel_rel",
        True,
        {"asset_name": "robot"},
    ),
    (
        "velocity_commands",
        "isaaclab.envs.mdp.observations.generated_commands",
        False,
        {"command_name": "base_velocity"},
    ),
    (
        "actions",
        "isaaclab.envs.mdp.observations.last_action",
        False,
        {"action_name": None},
    ),
)
_EXPECTATION_UNSET = object()


class RayTimeManifestError(ValueError):
    """Raised when a ray-time deployment manifest is missing or inconsistent."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a regular file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Expected a regular file to hash: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically, rejecting NaN and non-JSON values."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RayTimeManifestError(
            f"Manifest contains a non-canonical JSON value: {exc}"
        ) from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def collect_git_provenance(
    repositories: Mapping[str, str | os.PathLike[str]],
) -> dict[str, dict[str, Any]]:
    """Collect commit and worktree provenance for the three source repositories.

    The worktree digest binds the tracked diff against ``HEAD`` plus the
    contents of untracked files.  A dirty source tree is allowed, but it cannot
    be mistaken for the recorded clean commit.
    """
    if set(repositories) != set(RAY_TIME_PROVENANCE_REPOSITORIES):
        raise RayTimeManifestError(
            "Git provenance must contain exactly "
            f"{RAY_TIME_PROVENANCE_REPOSITORIES}, got {tuple(repositories)}."
        )

    result: dict[str, dict[str, Any]] = {}
    for name in RAY_TIME_PROVENANCE_REPOSITORIES:
        root = Path(repositories[name]).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{name} repository does not exist: {root}")

        def git(*args: str) -> bytes:
            try:
                return subprocess.check_output(
                    ["git", "-C", str(root), *args],
                    stderr=subprocess.STDOUT,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                output = getattr(exc, "output", b"")
                detail = output.decode("utf-8", errors="replace").strip()
                raise RayTimeManifestError(
                    f"Could not collect Git provenance for {name} at {root}: {detail}"
                ) from exc

        commit = git("rev-parse", "HEAD").decode("ascii").strip()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").decode("utf-8").strip()
        status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        tracked_diff = git("diff", "--binary", "HEAD", "--")
        untracked = git(
            "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0")

        snapshot = hashlib.sha256()
        snapshot.update(b"tracked-diff\0")
        snapshot.update(tracked_diff)
        snapshot.update(b"\0untracked-files\0")
        for raw_relative in sorted(item for item in untracked if item):
            relative = raw_relative.decode("utf-8", errors="surrogateescape")
            candidate = root / relative
            snapshot.update(raw_relative)
            snapshot.update(b"\0")
            if candidate.is_symlink():
                snapshot.update(b"symlink\0")
                snapshot.update(os.readlink(candidate).encode("utf-8"))
            elif candidate.is_file():
                snapshot.update(b"file\0")
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        snapshot.update(chunk)
            else:
                snapshot.update(b"missing-or-nonregular\0")
            snapshot.update(b"\0")

        result[name] = {
            "repository_root": str(root),
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "worktree_snapshot_sha256": snapshot.hexdigest(),
        }
    return result


def build_ray_time_deployment_manifest(
    *,
    encoder_variant: str,
    fusion_mode: str | None = None,
    history_length: int,
    proprio_terms: Sequence[Mapping[str, Any]],
    ray_shape: Sequence[int],
    ray_dtype: str,
    ray_channels: Sequence[Mapping[str, Any]],
    mount_geometry: Mapping[str, Any],
    row_elevation_degrees: Mapping[str, Any],
    column_azimuth_degrees: Mapping[str, Any],
    tensorization: Mapping[str, Any],
    history_order: str,
    packet_interval_control_steps: int,
    control_period_s: float,
    scan_fraction_per_packet: float,
    min_range_m: float,
    max_range_m: float,
    joint_order: Sequence[str],
    default_joint_positions: Sequence[float],
    per_joint_action_scale: Sequence[float],
    raw_actor_output_clip: Sequence[float] | None,
    processed_target_clip_rad: Sequence[float],
    previous_action_semantics: str,
    checkpoint_path: str | os.PathLike[str],
    training_agent_yaml_path: str | os.PathLike[str],
    training_env_yaml_path: str | os.PathLike[str],
    training_contract: Mapping[str, Any],
    torchscript_path: str | os.PathLike[str],
    onnx_path: str | os.PathLike[str],
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build, seal, and validate a deployment manifest.

    Schema v2 persists the canonical fusion mode independently of parameter
    names in the checkpoint.  ``fusion_mode=None`` remains a source-compatible
    builder fallback for the pre-QueryGlobal ``Global``/``Attention`` variants.
    """
    resolved_fusion_mode = _resolve_encoder_fusion_mode(
        encoder_variant=encoder_variant,
        fusion_mode=fusion_mode,
        path="encoder",
    )
    checkpoint = Path(checkpoint_path).resolve()
    training_agent_yaml = Path(training_agent_yaml_path).resolve()
    training_env_yaml = Path(training_env_yaml_path).resolve()
    torchscript = Path(torchscript_path).resolve()
    onnx = Path(onnx_path).resolve()
    payload: dict[str, Any] = {
        "schema": {
            "name": RAY_TIME_MANIFEST_SCHEMA_NAME,
            "version": RAY_TIME_MANIFEST_SCHEMA_VERSION,
        },
        "contract": {
            "encoder": {
                "type": "ray_time",
                "variant": encoder_variant,
                "fusion_mode": resolved_fusion_mode,
                "history_length": history_length,
            },
            "actor_proprioception": {
                "shape": [96],
                "dtype": "float32",
                "concatenation_order": [dict(term) for term in proprio_terms],
            },
            "ray_history": {
                "shape": list(ray_shape),
                "dtype": ray_dtype,
                "channels": [dict(channel) for channel in ray_channels],
                "mount_geometry": dict(mount_geometry),
                "row_elevation_degrees": dict(row_elevation_degrees),
                "column_azimuth_degrees": dict(column_azimuth_degrees),
                "tensorization": dict(tensorization),
                "history_order": history_order,
                "packet_interval_control_steps": packet_interval_control_steps,
                "control_period_s": control_period_s,
                "scan_fraction_per_packet": scan_fraction_per_packet,
                "valid_range_m": {
                    "min": min_range_m,
                    "max": max_range_m,
                },
                "unknown_ray_encoding": {
                    "range_m": 0.0,
                    "hit_mask": 0.0,
                },
            },
            "action": {
                "shape": [29],
                "dtype": "float32",
                "joint_order": list(joint_order),
                "default_joint_positions": list(default_joint_positions),
                "per_joint_action_scale": list(per_joint_action_scale),
                "raw_actor_output_clip": (
                    None
                    if raw_actor_output_clip is None
                    else list(raw_actor_output_clip)
                ),
                "processed_target_clip_rad": list(processed_target_clip_rad),
                "previous_action_semantics": previous_action_semantics,
                "decoding_formula": ray_time_action_formula(
                    raw_actor_output_clip
                ),
            },
            "export_interfaces": _build_export_interfaces(history_length),
        },
        "checkpoint": {
            "file_name": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "training_metadata": {
                "agent_yaml": _file_record(training_agent_yaml),
                "env_yaml": _file_record(training_env_yaml),
                "resolved_contract": dict(training_contract),
            },
        },
        "export_artifacts": {
            "torchscript": _file_record(
                torchscript,
                interface="$.contract.export_interfaces.torchscript",
            ),
            "onnx": _file_record(
                onnx,
                interface="$.contract.export_interfaces.onnx",
            ),
        },
        "provenance": {
            name: dict(provenance[name])
            for name in RAY_TIME_PROVENANCE_REPOSITORIES
            if name in provenance
        },
    }
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "RFC8259 JSON; UTF-8; sorted keys; no whitespace; no NaN",
        "payload_scope": "all top-level fields except integrity",
        "payload_sha256": canonical_json_sha256(payload),
    }
    validate_ray_time_deployment_manifest(
        payload,
        checkpoint_path=checkpoint,
        torchscript_path=torchscript,
        onnx_path=onnx,
    )
    return payload


def validate_ray_time_deployment_manifest(
    manifest: Mapping[str, Any],
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,
    torchscript_path: str | os.PathLike[str] | None = None,
    onnx_path: str | os.PathLike[str] | None = None,
    require_export_artifact: bool = True,
    expected_encoder_variant: str | None = None,
    expected_fusion_mode: str | None = None,
    expected_history_length: int | None = None,
    expected_proprio_shape: Sequence[int] | None = None,
    expected_proprio_order: Sequence[str] | None = None,
    expected_ray_shape: Sequence[int] | None = None,
    expected_ray_dtype: str | None = None,
    expected_ray_channels: Sequence[str] | None = None,
    expected_joint_order: Sequence[str] | None = None,
    expected_default_joint_positions: Sequence[float] | None = None,
    expected_action_scale: Sequence[float] | None = None,
    expected_raw_actor_output_clip: Sequence[float] | None | object = (
        _EXPECTATION_UNSET
    ),
    expected_contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate schema, integrity, checkpoint, and optional runtime expectations.

    Every optional ``expected_*`` argument is compared exactly.  Robot-side
    code should supply all expectations available from its sensor/controller
    configuration and the selected policy binary.
    """
    if not isinstance(manifest, Mapping):
        raise RayTimeManifestError("Manifest root must be a JSON object.")
    _require_exact_keys(
        manifest,
        (
            "schema",
            "contract",
            "checkpoint",
            "export_artifacts",
            "provenance",
            "integrity",
        ),
        "$",
    )

    schema = _mapping(manifest["schema"], "$.schema")
    _require_exact_keys(schema, ("name", "version"), "$.schema")
    _expect(schema["name"], RAY_TIME_MANIFEST_SCHEMA_NAME, "$.schema.name")
    schema_version = schema["version"]
    if schema_version not in (
        _LEGACY_MANIFEST_SCHEMA_VERSION,
        RAY_TIME_MANIFEST_SCHEMA_VERSION,
    ):
        raise RayTimeManifestError(
            "$.schema.version must be one of "
            f"{(_LEGACY_MANIFEST_SCHEMA_VERSION, RAY_TIME_MANIFEST_SCHEMA_VERSION)}, "
            f"got {schema_version!r}."
        )

    contract = _mapping(manifest["contract"], "$.contract")
    _require_exact_keys(
        contract,
        (
            "encoder",
            "actor_proprioception",
            "ray_history",
            "action",
            "export_interfaces",
        ),
        "$.contract",
    )
    _validate_encoder(
        contract["encoder"],
        schema_version=schema_version,
    )
    _validate_proprioception(contract["actor_proprioception"])
    _validate_ray_history(
        contract["ray_history"],
        schema_version=schema_version,
    )
    _validate_action(contract["action"])
    _validate_export_interfaces(
        contract["export_interfaces"],
        history_length=contract["encoder"]["history_length"],
    )
    _validate_checkpoint(
        manifest["checkpoint"],
        contract=contract,
        schema_version=schema_version,
    )
    _validate_export_artifacts(manifest["export_artifacts"])
    _validate_provenance(manifest["provenance"])
    _validate_integrity(manifest)

    encoder = _mapping(contract["encoder"], "$.contract.encoder")
    proprio = _mapping(
        contract["actor_proprioception"],
        "$.contract.actor_proprioception",
    )
    ray = _mapping(contract["ray_history"], "$.contract.ray_history")
    action = _mapping(contract["action"], "$.contract.action")
    _expect(
        ray["shape"][0],
        encoder["history_length"],
        "$.contract.ray_history.shape[0]",
    )

    _expect_optional(
        encoder["variant"],
        expected_encoder_variant,
        "$.contract.encoder.variant",
    )
    _expect_optional(
        _effective_encoder_fusion_mode(
            encoder,
            schema_version=schema_version,
        ),
        expected_fusion_mode,
        "$.contract.encoder.fusion_mode",
    )
    _expect_optional(
        encoder["history_length"],
        expected_history_length,
        "$.contract.encoder.history_length",
    )
    _expect_optional(
        list(proprio["shape"]),
        _optional_list(expected_proprio_shape),
        "$.contract.actor_proprioception.shape",
    )
    _expect_optional(
        [term["name"] for term in proprio["concatenation_order"]],
        _optional_list(expected_proprio_order),
        "$.contract.actor_proprioception.concatenation_order[*].name",
    )
    _expect_optional(
        list(ray["shape"]),
        _optional_list(expected_ray_shape),
        "$.contract.ray_history.shape",
    )
    _expect_optional(
        ray["dtype"],
        expected_ray_dtype,
        "$.contract.ray_history.dtype",
    )
    _expect_optional(
        [channel["name"] for channel in ray["channels"]],
        _optional_list(expected_ray_channels),
        "$.contract.ray_history.channels[*].name",
    )
    _expect_optional(
        list(action["joint_order"]),
        _optional_list(expected_joint_order),
        "$.contract.action.joint_order",
    )
    _expect_optional(
        list(action["default_joint_positions"]),
        _optional_float_list(expected_default_joint_positions),
        "$.contract.action.default_joint_positions",
    )
    _expect_optional(
        list(action["per_joint_action_scale"]),
        _optional_float_list(expected_action_scale),
        "$.contract.action.per_joint_action_scale",
    )
    if expected_raw_actor_output_clip is not _EXPECTATION_UNSET:
        expected_clip = (
            None
            if expected_raw_actor_output_clip is None
            else list(expected_raw_actor_output_clip)
        )
        _expect(
            action["raw_actor_output_clip"],
            expected_clip,
            "$.contract.action.raw_actor_output_clip",
        )
    if expected_contract is not None:
        _expect(
            contract,
            expected_contract,
            "$.contract",
        )

    if checkpoint_path is not None:
        checkpoint = _mapping(manifest["checkpoint"], "$.checkpoint")
        _validate_file_record_against_path(
            checkpoint,
            checkpoint_path,
            "$.checkpoint",
        )
    export_artifacts = _mapping(
        manifest["export_artifacts"],
        "$.export_artifacts",
    )
    if torchscript_path is not None:
        _validate_file_record_against_path(
            _mapping(
                export_artifacts["torchscript"],
                "$.export_artifacts.torchscript",
            ),
            torchscript_path,
            "$.export_artifacts.torchscript",
        )
    if onnx_path is not None:
        _validate_file_record_against_path(
            _mapping(export_artifacts["onnx"], "$.export_artifacts.onnx"),
            onnx_path,
            "$.export_artifacts.onnx",
        )
    if require_export_artifact and torchscript_path is None and onnx_path is None:
        raise RayTimeManifestError(
            "Deployment validation requires at least one explicit export "
            "artifact path (torchscript_path or onnx_path)."
        )


def serialize_ray_time_deployment_manifest(
    manifest: Mapping[str, Any],
) -> bytes:
    """Validate and return the deterministic on-disk JSON representation."""
    validate_ray_time_deployment_manifest(
        manifest,
        require_export_artifact=False,
    )
    return canonical_json_bytes(manifest) + b"\n"


def write_ray_time_deployment_manifest(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> Path:
    """Write a manifest atomically without replacing a different sidecar."""
    destination = Path(path)
    encoded = serialize_ray_time_deployment_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != encoded:
            raise RayTimeManifestError(
                "Refusing to overwrite a different deployment manifest: "
                f"{destination}"
            )
        return destination

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Fail closed if another process produced the sidecar concurrently.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise RayTimeManifestError(
                    "A different deployment manifest appeared concurrently: "
                    f"{destination}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_ray_time_deployment_manifest(
    path: str | os.PathLike[str],
    **validation_expectations: Any,
) -> dict[str, Any]:
    """Read a JSON sidecar and fail closed on any schema or expectation error."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise RayTimeManifestError(
            f"Could not read deployment manifest {source}: {exc}"
        ) from exc
    try:
        manifest = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RayTimeManifestError(
            f"Deployment manifest is not valid UTF-8 JSON: {source}"
        ) from exc
    validate_ray_time_deployment_manifest(
        manifest,
        **validation_expectations,
    )
    return manifest


def default_ray_time_proprio_terms() -> list[dict[str, Any]]:
    """Return the shared schema-v1/v2 96-value proprioceptive layout."""
    result = []
    offset = 0
    for name, size, semantic, indexed_by in zip(
        _PROPRIO_TERM_NAMES,
        _PROPRIO_TERM_SIZES,
        _PROPRIO_TERM_SEMANTICS,
        _PROPRIO_TERM_INDEX_REFERENCES,
    ):
        result.append(
            {
                "name": name,
                "offset": offset,
                "size": size,
                "semantic": semantic,
                "indexed_by": indexed_by,
            }
        )
        offset += size
    return result


def default_ray_time_channels() -> list[dict[str, Any]]:
    """Return the shared schema-v1/v2 ray channel layout."""
    return [
        {"index": index, "name": name, "semantic": semantic}
        for index, name, semantic in _RAY_CHANNELS
    ]


def default_ray_time_mount_geometry() -> dict[str, Any]:
    """Return the shared schema-v1/v2 mount and alignment."""
    return {
        "parent_link": "torso_link",
        "translation_m": [0.0002835, 0.00003, 0.41618],
        "original_rotation_xyzw": [0.0, 0.020069072, 0.0, 0.9997986],
        "local_upside_down_rotation_xyzw": [1.0, 0.0, 0.0, 0.0],
        "composed_rotation_xyzw": [0.9997986, 0.0, -0.020069072, 0.0],
        "rotation_composition": (
            "original_rotation_xyzw right-multiplied by the 180-degree "
            "local-x upside-down rotation"
        ),
        "ray_alignment": "base",
        "ray_alignment_semantic": (
            "rays follow torso_link full roll, pitch, and yaw; no gravity or "
            "yaw-only alignment is applied"
        ),
    }


def default_ray_time_tensorization() -> dict[str, Any]:
    """Return the current point/ray-to-packet conversion semantics."""
    return {
        "pattern_sensor_vertical_fov_degrees": [52.0, -7.0],
        "policy_body_row_vertical_fov_degrees": [-52.0, 7.0],
        "upside_down_width_reorder": {
            "operation_order": ["flip_width", "circular_roll_width"],
            "circular_roll_bins": 1,
            "resulting_azimuth_order": (
                "ascending body-frame azimuth on [-180, 180)"
            ),
        },
        "simulation_packet_mask": (
            "synthetic non-repetitive phase mask at the configured "
            "scan_fraction_per_packet"
        ),
        "real_packet_mask": (
            "use actual sensor-observed rays; never apply the simulation "
            "synthetic phase mask a second time"
        ),
        "range_encoding": {
            "finite_below_min_range": "[0.0, 0.0] (invalid)",
            "finite_at_or_above_min_range": (
                "[clamp(range_m, min_range_m, max_range_m), 1.0]"
            ),
            "finite_above_max_range": (
                "[max_range_m, 1.0] when the ray is actually observed"
            ),
            "unobserved_or_no_return": (
                _UNOBSERVED_RAY_SEMANTIC
            ),
        },
    }


def _ray_time_tensorization_for_schema(
    schema_version: int,
) -> dict[str, Any]:
    tensorization = default_ray_time_tensorization()
    if schema_version == _LEGACY_MANIFEST_SCHEMA_VERSION:
        tensorization["range_encoding"][
            "unobserved_or_no_return"
        ] = _LEGACY_UNOBSERVED_RAY_SEMANTIC
    return tensorization


def ray_time_action_formula(
    raw_actor_output_clip: Sequence[float] | None,
) -> list[str]:
    """Return the shared schema-v1/v2 action decoding formula."""
    if raw_actor_output_clip is None:
        actor_to_environment = (
            "a_env[i] = a_actor[i] (raw_actor_output_clip is null)"
        )
    else:
        actor_to_environment = (
            "a_env[i] = clamp(a_actor[i], raw_actor_output_clip[0], "
            "raw_actor_output_clip[1])"
        )
    return [actor_to_environment, _ENVIRONMENT_TO_JOINT_FORMULA]


def ray_time_previous_action_semantics() -> str:
    """Return the shared schema-v1/v2 previous-action definition."""
    return _PREVIOUS_ACTION_SEMANTICS


def _resolve_encoder_fusion_mode(
    *,
    encoder_variant: Any,
    fusion_mode: Any,
    path: str,
) -> str:
    if fusion_mode is None:
        for candidate, variant in _FUSION_MODE_TO_VARIANT.items():
            if variant not in _LEGACY_ENCODER_VARIANTS:
                continue
            if encoder_variant == variant:
                return candidate
        if encoder_variant == _FUSION_MODE_TO_VARIANT["query_global"]:
            raise RayTimeManifestError(
                f"{path}.fusion_mode='query_global' is required when "
                f"{path}.variant='QueryGlobal'; it cannot be inferred from "
                "legacy metadata."
            )
        raise RayTimeManifestError(
            f"{path}.variant must be one of {_ENCODER_VARIANTS}, "
            f"got {encoder_variant!r}."
        )
    if not isinstance(fusion_mode, str) or fusion_mode not in (
        _FUSION_MODE_TO_VARIANT
    ):
        raise RayTimeManifestError(
            f"{path}.fusion_mode must be one of "
            f"{tuple(_FUSION_MODE_TO_VARIANT)}, got {fusion_mode!r}."
        )
    expected_variant = _FUSION_MODE_TO_VARIANT[fusion_mode]
    if encoder_variant != expected_variant:
        raise RayTimeManifestError(
            f"{path}.variant={encoder_variant!r} conflicts with "
            f"{path}.fusion_mode={fusion_mode!r}; expected "
            f"variant {expected_variant!r}."
        )
    return fusion_mode


def _effective_encoder_fusion_mode(
    encoder: Mapping[str, Any],
    *,
    schema_version: int,
) -> str:
    if schema_version == _LEGACY_MANIFEST_SCHEMA_VERSION:
        return _resolve_encoder_fusion_mode(
            encoder_variant=encoder["variant"],
            fusion_mode=None,
            path="$.contract.encoder",
        )
    return _resolve_encoder_fusion_mode(
        encoder_variant=encoder["variant"],
        fusion_mode=encoder["fusion_mode"],
        path="$.contract.encoder",
    )


def _validate_encoder(value: Any, *, schema_version: int) -> None:
    encoder = _mapping(value, "$.contract.encoder")
    expected_keys = ["type", "variant", "history_length"]
    if schema_version == RAY_TIME_MANIFEST_SCHEMA_VERSION:
        expected_keys.insert(2, "fusion_mode")
    _require_exact_keys(
        encoder,
        expected_keys,
        "$.contract.encoder",
    )
    _expect(encoder["type"], "ray_time", "$.contract.encoder.type")
    allowed_variants = (
        _LEGACY_ENCODER_VARIANTS
        if schema_version == _LEGACY_MANIFEST_SCHEMA_VERSION
        else _ENCODER_VARIANTS
    )
    if encoder["variant"] not in allowed_variants:
        raise RayTimeManifestError(
            "$.contract.encoder.variant must be one of "
            f"{allowed_variants}, got {encoder['variant']!r}."
        )
    _effective_encoder_fusion_mode(
        encoder,
        schema_version=schema_version,
    )
    _positive_int(
        encoder["history_length"],
        "$.contract.encoder.history_length",
    )


def _validate_proprioception(value: Any) -> None:
    proprio = _mapping(value, "$.contract.actor_proprioception")
    _require_exact_keys(
        proprio,
        ("shape", "dtype", "concatenation_order"),
        "$.contract.actor_proprioception",
    )
    _expect(proprio["shape"], [96], "$.contract.actor_proprioception.shape")
    _expect(
        proprio["dtype"],
        "float32",
        "$.contract.actor_proprioception.dtype",
    )
    terms = _list(
        proprio["concatenation_order"],
        "$.contract.actor_proprioception.concatenation_order",
    )
    if len(terms) != len(_PROPRIO_TERM_NAMES):
        raise RayTimeManifestError(
            "$.contract.actor_proprioception.concatenation_order must contain "
            f"{len(_PROPRIO_TERM_NAMES)} terms, got {len(terms)}."
        )
    offset = 0
    for index, raw_term in enumerate(terms):
        path = f"$.contract.actor_proprioception.concatenation_order[{index}]"
        term = _mapping(raw_term, path)
        _require_exact_keys(
            term,
            ("name", "offset", "size", "semantic", "indexed_by"),
            path,
        )
        _expect(term["name"], _PROPRIO_TERM_NAMES[index], f"{path}.name")
        _expect(term["offset"], offset, f"{path}.offset")
        _expect(term["size"], _PROPRIO_TERM_SIZES[index], f"{path}.size")
        _expect(
            term["semantic"],
            _PROPRIO_TERM_SEMANTICS[index],
            f"{path}.semantic",
        )
        _expect(
            term["indexed_by"],
            _PROPRIO_TERM_INDEX_REFERENCES[index],
            f"{path}.indexed_by",
        )
        offset += _PROPRIO_TERM_SIZES[index]
    _expect(offset, 96, "$.contract.actor_proprioception.shape[0]")


def _validate_ray_history(value: Any, *, schema_version: int) -> None:
    path = "$.contract.ray_history"
    ray = _mapping(value, path)
    _require_exact_keys(
        ray,
        (
            "shape",
            "dtype",
            "channels",
            "mount_geometry",
            "row_elevation_degrees",
            "column_azimuth_degrees",
            "tensorization",
            "history_order",
            "packet_interval_control_steps",
            "control_period_s",
            "scan_fraction_per_packet",
            "valid_range_m",
            "unknown_ray_encoding",
        ),
        path,
    )
    shape = _list(ray["shape"], f"{path}.shape")
    if len(shape) != 4 or any(
        not _is_positive_int(value) for value in shape
    ):
        raise RayTimeManifestError(
            f"{path}.shape must be four positive integers, got {shape!r}."
        )
    _expect(ray["dtype"], "float16", f"{path}.dtype")

    channels = _list(ray["channels"], f"{path}.channels")
    if len(channels) != len(_RAY_CHANNELS):
        raise RayTimeManifestError(
            f"{path}.channels must contain exactly two channels."
        )
    for index, raw_channel in enumerate(channels):
        channel_path = f"{path}.channels[{index}]"
        channel = _mapping(raw_channel, channel_path)
        _require_exact_keys(
            channel,
            ("index", "name", "semantic"),
            channel_path,
        )
        expected = _RAY_CHANNELS[index]
        _expect(channel["index"], expected[0], f"{channel_path}.index")
        _expect(channel["name"], expected[1], f"{channel_path}.name")
        _expect(channel["semantic"], expected[2], f"{channel_path}.semantic")

    mount_geometry = _mapping(
        ray["mount_geometry"],
        f"{path}.mount_geometry",
    )
    _expect(
        mount_geometry,
        default_ray_time_mount_geometry(),
        f"{path}.mount_geometry",
    )

    row = _mapping(ray["row_elevation_degrees"], f"{path}.row_elevation_degrees")
    _require_exact_keys(
        row,
        ("count", "first", "last", "sampling"),
        f"{path}.row_elevation_degrees",
    )
    _expect(row["count"], 16, f"{path}.row_elevation_degrees.count")
    _expect(row["first"], -52.0, f"{path}.row_elevation_degrees.first")
    _expect(row["last"], 7.0, f"{path}.row_elevation_degrees.last")
    _expect(
        row["sampling"],
        "linear_inclusive",
        f"{path}.row_elevation_degrees.sampling",
    )

    column = _mapping(
        ray["column_azimuth_degrees"],
        f"{path}.column_azimuth_degrees",
    )
    _require_exact_keys(
        column,
        ("count", "first", "last_exclusive", "step", "sampling"),
        f"{path}.column_azimuth_degrees",
    )
    _expect(column["count"], 96, f"{path}.column_azimuth_degrees.count")
    _expect(column["first"], -180.0, f"{path}.column_azimuth_degrees.first")
    _expect(
        column["last_exclusive"],
        180.0,
        f"{path}.column_azimuth_degrees.last_exclusive",
    )
    _expect(column["step"], 3.75, f"{path}.column_azimuth_degrees.step")
    _expect(
        column["sampling"],
        "uniform_half_open",
        f"{path}.column_azimuth_degrees.sampling",
    )

    tensorization = _mapping(ray["tensorization"], f"{path}.tensorization")
    _expect(
        tensorization,
        _ray_time_tensorization_for_schema(schema_version),
        f"{path}.tensorization",
    )

    _expect(ray["history_order"], "oldest_to_newest", f"{path}.history_order")
    _expect(
        ray["packet_interval_control_steps"],
        5,
        f"{path}.packet_interval_control_steps",
    )
    _expect(ray["control_period_s"], 0.02, f"{path}.control_period_s")
    _expect(
        ray["scan_fraction_per_packet"],
        0.2,
        f"{path}.scan_fraction_per_packet",
    )
    valid_range = _mapping(ray["valid_range_m"], f"{path}.valid_range_m")
    _require_exact_keys(valid_range, ("min", "max"), f"{path}.valid_range_m")
    _expect(valid_range["min"], 0.1, f"{path}.valid_range_m.min")
    _expect(valid_range["max"], 6.0, f"{path}.valid_range_m.max")
    unknown = _mapping(
        ray["unknown_ray_encoding"],
        f"{path}.unknown_ray_encoding",
    )
    _require_exact_keys(
        unknown,
        ("range_m", "hit_mask"),
        f"{path}.unknown_ray_encoding",
    )
    _expect(unknown["range_m"], 0.0, f"{path}.unknown_ray_encoding.range_m")
    _expect(unknown["hit_mask"], 0.0, f"{path}.unknown_ray_encoding.hit_mask")

    _positive_int(shape[0], f"{path}.shape[0]")
    _expect(shape[1], len(channels), f"{path}.shape[1]")
    _expect(shape[2], row["count"], f"{path}.shape[2]")
    _expect(shape[3], column["count"], f"{path}.shape[3]")


def _validate_action(value: Any) -> None:
    path = "$.contract.action"
    action = _mapping(value, path)
    _require_exact_keys(
        action,
        (
            "shape",
            "dtype",
            "joint_order",
            "default_joint_positions",
            "per_joint_action_scale",
            "raw_actor_output_clip",
            "processed_target_clip_rad",
            "previous_action_semantics",
            "decoding_formula",
        ),
        path,
    )
    _expect(action["shape"], [29], f"{path}.shape")
    _expect(action["dtype"], "float32", f"{path}.dtype")
    joint_order = _list(action["joint_order"], f"{path}.joint_order")
    defaults = _list(
        action["default_joint_positions"],
        f"{path}.default_joint_positions",
    )
    scales = _list(
        action["per_joint_action_scale"],
        f"{path}.per_joint_action_scale",
    )
    raw_actor_clip = action["raw_actor_output_clip"]
    target_clip = _list(
        action["processed_target_clip_rad"],
        f"{path}.processed_target_clip_rad",
    )
    if len(joint_order) != 29 or len(set(joint_order)) != 29:
        raise RayTimeManifestError(
            f"{path}.joint_order must contain 29 unique names."
        )
    for index, name in enumerate(joint_order):
        if not isinstance(name, str) or not name:
            raise RayTimeManifestError(
                f"{path}.joint_order[{index}] must be a non-empty string."
            )
    _finite_number_list(defaults, 29, f"{path}.default_joint_positions")
    _finite_number_list(scales, 29, f"{path}.per_joint_action_scale")
    if any(float(scale) <= 0.0 for scale in scales):
        raise RayTimeManifestError(
            f"{path}.per_joint_action_scale values must all be positive."
        )
    if raw_actor_clip is not None:
        raw_actor_clip = _list(
            raw_actor_clip,
            f"{path}.raw_actor_output_clip",
        )
        _finite_number_list(
            raw_actor_clip,
            2,
            f"{path}.raw_actor_output_clip",
        )
        if (
            float(raw_actor_clip[0]) >= 0.0
            or float(raw_actor_clip[1]) <= 0.0
            or float(raw_actor_clip[0]) != -float(raw_actor_clip[1])
        ):
            raise RayTimeManifestError(
                f"{path}.raw_actor_output_clip must be symmetric [-x, x]."
            )
    _finite_number_list(target_clip, 2, f"{path}.processed_target_clip_rad")
    _expect(target_clip, [-50.0, 50.0], f"{path}.processed_target_clip_rad")
    _expect(
        action["previous_action_semantics"],
        _PREVIOUS_ACTION_SEMANTICS,
        f"{path}.previous_action_semantics",
    )
    decoding_formula = _list(
        action["decoding_formula"],
        f"{path}.decoding_formula",
    )
    _expect(
        decoding_formula,
        ray_time_action_formula(raw_actor_clip),
        f"{path}.decoding_formula",
    )


def _build_export_interfaces(history_length: int) -> dict[str, Any]:
    flat_input_size = 96 + int(history_length) * 2 * 16 * 96
    return {
        "torchscript": {
            "input_mode": "split",
            "inputs": [
                {
                    "name": "proprio_obs",
                    "dtype": "float32",
                    "shape": ["B", 96],
                },
                {
                    "name": "ray_history",
                    "allowed_dtypes": ["float16", "float32"],
                    "shape": ["B", int(history_length), 2, 16, 96],
                },
            ],
            "output": {
                "name": "actions",
                "dtype": "float32",
                "shape": ["B", 29],
                "semantic": "raw_action",
            },
        },
        "onnx": {
            "input_mode": "single",
            "input": {
                "name": "obs",
                "dtype": "float32",
                "shape": ["B", flat_input_size],
                "layout": (
                    "proprio_obs[96] followed by row-major flattened "
                    "ray_history[K,2,16,96]"
                ),
            },
            "output": {
                "name": "actions",
                "dtype": "float32",
                "shape": ["B", 29],
                "semantic": "raw_action",
            },
        },
    }


def _validate_export_interfaces(value: Any, *, history_length: int) -> None:
    path = "$.contract.export_interfaces"
    interfaces = _mapping(value, path)
    _require_exact_keys(interfaces, ("torchscript", "onnx"), path)
    _expect(
        interfaces,
        _build_export_interfaces(history_length),
        path,
    )


def _validate_checkpoint(
    value: Any,
    *,
    contract: Mapping[str, Any],
    schema_version: int,
) -> None:
    path = "$.checkpoint"
    checkpoint = _mapping(value, path)
    _require_exact_keys(
        checkpoint,
        ("file_name", "size_bytes", "sha256", "training_metadata"),
        path,
    )
    _validate_file_record_fields(checkpoint, path)
    metadata_path = f"{path}.training_metadata"
    metadata = _mapping(checkpoint["training_metadata"], metadata_path)
    _require_exact_keys(
        metadata,
        ("agent_yaml", "env_yaml", "resolved_contract"),
        metadata_path,
    )
    _validate_file_record(metadata["agent_yaml"], f"{metadata_path}.agent_yaml")
    _validate_file_record(metadata["env_yaml"], f"{metadata_path}.env_yaml")
    _validate_training_contract(
        metadata["resolved_contract"],
        contract=contract,
        schema_version=schema_version,
        path=f"{metadata_path}.resolved_contract",
    )


def _validate_export_artifacts(value: Any) -> None:
    path = "$.export_artifacts"
    artifacts = _mapping(value, path)
    _require_exact_keys(artifacts, ("torchscript", "onnx"), path)
    _validate_file_record(
        artifacts["torchscript"],
        f"{path}.torchscript",
        expected_interface="$.contract.export_interfaces.torchscript",
    )
    _validate_file_record(
        artifacts["onnx"],
        f"{path}.onnx",
        expected_interface="$.contract.export_interfaces.onnx",
    )


def _file_record(
    path: Path,
    *,
    interface: str | None = None,
) -> dict[str, Any]:
    record = {
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if interface is not None:
        record["interface"] = interface
    return record


def _validate_file_record(
    value: Any,
    path: str,
    *,
    expected_interface: str | None = None,
) -> None:
    record = _mapping(value, path)
    expected_keys = ["file_name", "size_bytes", "sha256"]
    if expected_interface is not None:
        expected_keys.append("interface")
    _require_exact_keys(
        record,
        expected_keys,
        path,
    )
    _validate_file_record_fields(record, path)
    if expected_interface is not None:
        _expect(record["interface"], expected_interface, f"{path}.interface")


def _validate_file_record_fields(
    record: Mapping[str, Any],
    path: str,
) -> None:
    if not isinstance(record["file_name"], str) or not record["file_name"]:
        raise RayTimeManifestError(f"{path}.file_name must be non-empty.")
    if not _is_positive_int(record["size_bytes"]):
        raise RayTimeManifestError(f"{path}.size_bytes must be positive.")
    _sha256(record["sha256"], f"{path}.sha256")


def _validate_training_contract(
    value: Any,
    *,
    contract: Mapping[str, Any],
    schema_version: int,
    path: str,
) -> None:
    training = _mapping(value, path)
    expected_keys = [
        "encoder_variant",
        "history_length",
        "use_query_attention",
        "actor_spatial_size",
        "actor_valid_range_m",
        "actor_vertical_fov_degrees",
        "env_history_length",
        "env_image_size",
        "env_packet_interval_control_steps",
        "env_valid_range_m",
        "env_scan_fraction_per_packet",
        "raw_actor_output_clip",
        "control_decimation",
        "simulation_dt_s",
        "control_period_s",
        "action_joint_names",
        "action_preserve_order",
        "action_use_default_offset",
        "action_default_joint_positions",
        "action_per_joint_scale",
        "action_processed_target_clip_rad",
        "policy_observation_terms",
    ]
    if schema_version == RAY_TIME_MANIFEST_SCHEMA_VERSION:
        expected_keys.insert(1, "fusion_mode")
    _require_exact_keys(
        training,
        expected_keys,
        path,
    )
    encoder = _mapping(contract["encoder"], "$.contract.encoder")
    ray = _mapping(contract["ray_history"], "$.contract.ray_history")
    action = _mapping(contract["action"], "$.contract.action")
    fusion_mode = _effective_encoder_fusion_mode(
        encoder,
        schema_version=schema_version,
    )
    expected = {
        "encoder_variant": encoder["variant"],
        "history_length": encoder["history_length"],
        "use_query_attention": _FUSION_MODE_TO_LEGACY_BOOL[fusion_mode],
        "actor_spatial_size": [16, 96],
        "actor_valid_range_m": {"min": 0.1, "max": 6.0},
        "actor_vertical_fov_degrees": [-52.0, 7.0],
        "env_history_length": encoder["history_length"],
        "env_image_size": [16, 96],
        "env_packet_interval_control_steps": 5,
        "env_valid_range_m": {"min": 0.1, "max": 6.0},
        "env_scan_fraction_per_packet": 0.2,
        "raw_actor_output_clip": action["raw_actor_output_clip"],
        "control_decimation": 4,
        "simulation_dt_s": 0.005,
        "control_period_s": ray["control_period_s"],
        "action_joint_names": action["joint_order"],
        "action_preserve_order": True,
        "action_use_default_offset": True,
        "action_default_joint_positions": action[
            "default_joint_positions"
        ],
        "action_per_joint_scale": action["per_joint_action_scale"],
        "action_processed_target_clip_rad": [
            action["processed_target_clip_rad"]
            for _ in action["joint_order"]
        ],
        "policy_observation_terms": [
            {
                "name": name,
                "callable": callable_name,
                "joint_names": (
                    action["joint_order"] if has_joint_order else None
                ),
                "preserve_order": True if has_joint_order else None,
                "parameters": parameters,
            }
            for name, callable_name, has_joint_order, parameters in (
                _TRAINING_POLICY_OBSERVATION_TERMS
            )
        ],
    }
    if schema_version == RAY_TIME_MANIFEST_SCHEMA_VERSION:
        # Preserve insertion order only for readability; canonical JSON sorts
        # keys before hashing.
        expected = {
            "encoder_variant": expected.pop("encoder_variant"),
            "fusion_mode": fusion_mode,
            **expected,
        }
    _expect(training, expected, path)
    _expect(
        ray["shape"],
        [
            training["history_length"],
            2,
            *training["env_image_size"],
        ],
        "$.contract.ray_history.shape",
    )


def _validate_file_record_against_path(
    record: Mapping[str, Any],
    file_path: str | os.PathLike[str],
    manifest_path: str,
) -> None:
    resolved = Path(file_path).resolve()
    if not resolved.is_file():
        raise RayTimeManifestError(
            f"Artifact supplied for validation is not a file: {resolved}"
        )
    _expect(resolved.name, record["file_name"], f"{manifest_path}.file_name")
    _expect(
        resolved.stat().st_size,
        record["size_bytes"],
        f"{manifest_path}.size_bytes",
    )
    _expect(
        sha256_file(resolved),
        record["sha256"],
        f"{manifest_path}.sha256",
    )


def _validate_provenance(value: Any) -> None:
    provenance = _mapping(value, "$.provenance")
    _require_exact_keys(
        provenance,
        RAY_TIME_PROVENANCE_REPOSITORIES,
        "$.provenance",
    )
    for repository in RAY_TIME_PROVENANCE_REPOSITORIES:
        path = f"$.provenance.{repository}"
        entry = _mapping(provenance[repository], path)
        _require_exact_keys(
            entry,
            (
                "repository_root",
                "commit",
                "branch",
                "dirty",
                "worktree_snapshot_sha256",
            ),
            path,
        )
        if not isinstance(entry["repository_root"], str) or not entry["repository_root"]:
            raise RayTimeManifestError(f"{path}.repository_root must be non-empty.")
        if not isinstance(entry["commit"], str) or len(entry["commit"]) != 40:
            raise RayTimeManifestError(
                f"{path}.commit must be a 40-character Git object id."
            )
        if not isinstance(entry["branch"], str) or not entry["branch"]:
            raise RayTimeManifestError(f"{path}.branch must be non-empty.")
        if not isinstance(entry["dirty"], bool):
            raise RayTimeManifestError(f"{path}.dirty must be boolean.")
        _sha256(
            entry["worktree_snapshot_sha256"],
            f"{path}.worktree_snapshot_sha256",
        )


def _validate_integrity(manifest: Mapping[str, Any]) -> None:
    integrity = _mapping(manifest["integrity"], "$.integrity")
    _require_exact_keys(
        integrity,
        ("algorithm", "canonicalization", "payload_scope", "payload_sha256"),
        "$.integrity",
    )
    _expect(integrity["algorithm"], "sha256", "$.integrity.algorithm")
    _expect(
        integrity["canonicalization"],
        "RFC8259 JSON; UTF-8; sorted keys; no whitespace; no NaN",
        "$.integrity.canonicalization",
    )
    _expect(
        integrity["payload_scope"],
        "all top-level fields except integrity",
        "$.integrity.payload_scope",
    )
    _sha256(integrity["payload_sha256"], "$.integrity.payload_sha256")
    payload = {
        key: value for key, value in manifest.items() if key != "integrity"
    }
    _expect(
        integrity["payload_sha256"],
        canonical_json_sha256(payload),
        "$.integrity.payload_sha256",
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RayTimeManifestError(f"{path} must be a JSON object.")
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RayTimeManifestError(
                f"Deployment manifest contains duplicate JSON key {key!r}."
            )
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> Any:
    raise RayTimeManifestError(
        f"Deployment manifest contains non-finite JSON constant {value!r}."
    )


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise RayTimeManifestError(f"{path} must be a JSON array.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    path: str,
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise RayTimeManifestError(
            f"{path} fields differ from schema; missing={missing}, extra={extra}."
        )


def _expect(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise RayTimeManifestError(
            f"{path} mismatch: expected {expected!r}, got {actual!r}."
        )


def _expect_optional(actual: Any, expected: Any, path: str) -> None:
    if expected is not None:
        _expect(actual, expected, path)


def _optional_list(value: Sequence[Any] | None) -> list[Any] | None:
    return None if value is None else list(value)


def _optional_float_list(
    value: Sequence[float] | None,
) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def _positive_int(value: Any, path: str) -> None:
    if not _is_positive_int(value):
        raise RayTimeManifestError(f"{path} must be a positive integer.")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number_list(value: list[Any], length: int, path: str) -> None:
    if len(value) != length:
        raise RayTimeManifestError(
            f"{path} must contain {length} values, got {len(value)}."
        )
    for index, item in enumerate(value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise RayTimeManifestError(
                f"{path}[{index}] must be a finite JSON number."
            )


def _sha256(value: Any, path: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RayTimeManifestError(
            f"{path} must be a lowercase {_SHA256_HEX_LENGTH}-character SHA-256."
        )
