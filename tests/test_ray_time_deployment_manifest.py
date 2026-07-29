from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from rsl_rl.utils.ray_time_deployment_manifest import (
    RayTimeManifestError,
    build_ray_time_deployment_manifest,
    canonical_json_bytes,
    collect_git_provenance,
    default_ray_time_channels,
    default_ray_time_mount_geometry,
    default_ray_time_proprio_terms,
    default_ray_time_tensorization,
    read_ray_time_deployment_manifest,
    ray_time_action_formula,
    ray_time_previous_action_semantics,
    serialize_ray_time_deployment_manifest,
    validate_ray_time_deployment_manifest,
    write_ray_time_deployment_manifest,
)


_JOINT_ORDER = tuple(f"joint_{index:02d}" for index in range(29))
_DEFAULTS = tuple(index * 0.01 for index in range(29))
_SCALES = tuple(0.1 + index * 0.001 for index in range(29))


def _training_contract(history_length: int, variant: str) -> dict:
    joint_terms = {"joint_pos", "joint_vel"}
    term_callables = (
        ("base_ang_vel", "isaaclab.envs.mdp.observations.base_ang_vel"),
        (
            "projected_gravity",
            "isaaclab.envs.mdp.observations.projected_gravity",
        ),
        ("joint_pos", "isaaclab.envs.mdp.observations.joint_pos_rel"),
        ("joint_vel", "isaaclab.envs.mdp.observations.joint_vel_rel"),
        (
            "velocity_commands",
            "isaaclab.envs.mdp.observations.generated_commands",
        ),
        ("actions", "isaaclab.envs.mdp.observations.last_action"),
    )
    term_parameters = {
        "joint_pos": {"asset_name": "robot"},
        "joint_vel": {"asset_name": "robot"},
        "velocity_commands": {"command_name": "base_velocity"},
        "actions": {"action_name": None},
    }
    return {
        "encoder_variant": variant,
        "history_length": history_length,
        "use_query_attention": variant == "Attention",
        "actor_spatial_size": [16, 96],
        "actor_valid_range_m": {"min": 0.1, "max": 6.0},
        "actor_vertical_fov_degrees": [-52.0, 7.0],
        "env_history_length": history_length,
        "env_image_size": [16, 96],
        "env_packet_interval_control_steps": 5,
        "env_valid_range_m": {"min": 0.1, "max": 6.0},
        "env_scan_fraction_per_packet": 0.2,
        "raw_actor_output_clip": [-50.0, 50.0],
        "control_decimation": 4,
        "simulation_dt_s": 0.005,
        "control_period_s": 0.02,
        "action_joint_names": list(_JOINT_ORDER),
        "action_preserve_order": True,
        "action_use_default_offset": True,
        "action_default_joint_positions": list(_DEFAULTS),
        "action_per_joint_scale": list(_SCALES),
        "action_processed_target_clip_rad": [
            [-50.0, 50.0] for _ in _JOINT_ORDER
        ],
        "policy_observation_terms": [
            {
                "name": name,
                "callable": callable_name,
                "joint_names": (
                    list(_JOINT_ORDER) if name in joint_terms else None
                ),
                "preserve_order": True if name in joint_terms else None,
                "parameters": term_parameters.get(name, {}),
            }
            for name, callable_name in term_callables
        ],
    }


def _provenance() -> dict:
    return {
        name: {
            "repository_root": f"/workspace/{name}",
            "commit": character * 40,
            "branch": "main",
            "dirty": False,
            "worktree_snapshot_sha256": character * 64,
        }
        for name, character in (
            ("lab_pro", "1"),
            ("rsl_rl", "2"),
            ("IsaacLab", "3"),
        )
    }


def _manifest(
    checkpoint: Path,
    *,
    history_length: int,
    variant: str,
) -> dict:
    torchscript = checkpoint.with_name(f"{checkpoint.stem}.jit.pt")
    onnx = checkpoint.with_name(f"{checkpoint.stem}.onnx")
    agent_yaml = checkpoint.with_name("agent.yaml")
    env_yaml = checkpoint.with_name("env.yaml")
    torchscript.write_bytes(b"torchscript artifact")
    onnx.write_bytes(b"onnx artifact")
    agent_yaml.write_bytes(b"agent metadata")
    env_yaml.write_bytes(b"environment metadata")
    return build_ray_time_deployment_manifest(
        encoder_variant=variant,
        history_length=history_length,
        proprio_terms=default_ray_time_proprio_terms(),
        ray_shape=(history_length, 2, 16, 96),
        ray_dtype="float16",
        ray_channels=default_ray_time_channels(),
        mount_geometry=default_ray_time_mount_geometry(),
        row_elevation_degrees={
            "count": 16,
            "first": -52.0,
            "last": 7.0,
            "sampling": "linear_inclusive",
        },
        column_azimuth_degrees={
            "count": 96,
            "first": -180.0,
            "last_exclusive": 180.0,
            "step": 3.75,
            "sampling": "uniform_half_open",
        },
        tensorization=default_ray_time_tensorization(),
        history_order="oldest_to_newest",
        packet_interval_control_steps=5,
        control_period_s=0.02,
        scan_fraction_per_packet=0.2,
        min_range_m=0.1,
        max_range_m=6.0,
        joint_order=_JOINT_ORDER,
        default_joint_positions=_DEFAULTS,
        per_joint_action_scale=_SCALES,
        raw_actor_output_clip=(-50.0, 50.0),
        processed_target_clip_rad=(-50.0, 50.0),
        previous_action_semantics=ray_time_previous_action_semantics(),
        checkpoint_path=checkpoint,
        training_agent_yaml_path=agent_yaml,
        training_env_yaml_path=env_yaml,
        training_contract=_training_contract(history_length, variant),
        torchscript_path=torchscript,
        onnx_path=onnx,
        provenance=_provenance(),
    )


@pytest.mark.parametrize(
    ("history_length", "variant"),
    ((1, "Global"), (5, "Global"), (5, "Attention")),
)
def test_k1_k5_global_attention_manifests_are_self_consistent(
    tmp_path: Path,
    history_length: int,
    variant: str,
) -> None:
    checkpoint = tmp_path / f"k{history_length}_{variant}.pt"
    checkpoint.write_bytes(b"checkpoint payload")
    manifest = _manifest(
        checkpoint,
        history_length=history_length,
        variant=variant,
    )

    validate_ray_time_deployment_manifest(
        manifest,
        checkpoint_path=checkpoint,
        torchscript_path=checkpoint.with_name(
            f"{checkpoint.stem}.jit.pt"
        ),
        onnx_path=checkpoint.with_name(f"{checkpoint.stem}.onnx"),
        expected_encoder_variant=variant,
        expected_history_length=history_length,
        expected_proprio_shape=(96,),
        expected_proprio_order=(
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel_rel",
            "velocity_commands",
            "previous_environment_action",
        ),
        expected_ray_shape=(history_length, 2, 16, 96),
        expected_ray_dtype="float16",
        expected_ray_channels=("range_m", "hit_mask"),
        expected_joint_order=_JOINT_ORDER,
        expected_default_joint_positions=_DEFAULTS,
        expected_action_scale=_SCALES,
    )
    assert manifest["contract"]["ray_history"]["history_order"] == (
        "oldest_to_newest"
    )
    assert manifest["contract"]["action"]["decoding_formula"] == (
        ray_time_action_formula((-50.0, 50.0))
    )
    assert manifest["contract"]["export_interfaces"]["onnx"]["input"][
        "shape"
    ] == ["B", 96 + history_length * 2 * 16 * 96]


def test_serialization_and_hash_are_deterministic_and_round_trip(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Attention")

    first = serialize_ray_time_deployment_manifest(manifest)
    second = serialize_ray_time_deployment_manifest(deepcopy(manifest))
    assert first == second
    assert first == canonical_json_bytes(manifest) + b"\n"

    destination = tmp_path / "model.manifest.json"
    assert write_ray_time_deployment_manifest(destination, manifest) == destination
    assert write_ray_time_deployment_manifest(destination, manifest) == destination
    loaded = read_ray_time_deployment_manifest(
        destination,
        checkpoint_path=checkpoint,
        torchscript_path=checkpoint.with_name("model.jit.pt"),
        expected_encoder_variant="Attention",
        expected_history_length=5,
    )
    assert loaded == manifest


@pytest.mark.parametrize(
    ("expectation", "value", "match"),
    (
        ("expected_encoder_variant", "Attention", "encoder.variant"),
        ("expected_history_length", 1, "encoder.history_length"),
        ("expected_proprio_shape", (95,), "actor_proprioception.shape"),
        (
            "expected_proprio_order",
            (
                "projected_gravity",
                "base_ang_vel",
                "joint_pos_rel",
                "joint_vel_rel",
                "velocity_commands",
                "previous_environment_action",
            ),
            "concatenation_order",
        ),
        ("expected_ray_shape", (1, 2, 16, 96), "ray_history.shape"),
        (
            "expected_ray_channels",
            ("hit_mask", "range_m"),
            "channels",
        ),
        (
            "expected_joint_order",
            tuple(reversed(_JOINT_ORDER)),
            "joint_order",
        ),
        (
            "expected_default_joint_positions",
            (999.0, *_DEFAULTS[1:]),
            "default_joint_positions",
        ),
        (
            "expected_action_scale",
            (999.0, *_SCALES[1:]),
            "per_joint_action_scale",
        ),
        (
            "expected_raw_actor_output_clip",
            (-1.0, 1.0),
            "raw_actor_output_clip",
        ),
    ),
)
def test_runtime_contract_mismatches_fail_closed(
    tmp_path: Path,
    expectation: str,
    value,
    match: str,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Global")

    with pytest.raises(RayTimeManifestError, match=match):
        validate_ray_time_deployment_manifest(
            manifest,
            **{expectation: value},
        )


def test_missing_or_extra_fields_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Global")

    missing = deepcopy(manifest)
    del missing["contract"]["action"]["joint_order"]
    with pytest.raises(RayTimeManifestError, match="missing=.*joint_order"):
        validate_ray_time_deployment_manifest(missing)

    extra = deepcopy(manifest)
    extra["contract"]["encoder"]["unversioned_hint"] = "unsafe"
    with pytest.raises(RayTimeManifestError, match="extra=.*unversioned_hint"):
        validate_ray_time_deployment_manifest(extra)


def test_checkpoint_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"original")
    manifest = _manifest(checkpoint, history_length=1, variant="Global")
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(RayTimeManifestError, match="checkpoint.sha256"):
        validate_ray_time_deployment_manifest(
            manifest,
            checkpoint_path=checkpoint,
        )


def test_export_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Attention")
    torchscript = checkpoint.with_name("model.jit.pt")
    torchscript.write_bytes(b"x" * len(b"torchscript artifact"))

    with pytest.raises(
        RayTimeManifestError,
        match=r"export_artifacts.torchscript.sha256",
    ):
        validate_ray_time_deployment_manifest(
            manifest,
            torchscript_path=torchscript,
        )


def test_export_artifact_missing_or_swapped_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Global")
    torchscript = checkpoint.with_name("model.jit.pt")
    onnx = checkpoint.with_name("model.onnx")

    with pytest.raises(RayTimeManifestError, match="at least one explicit"):
        validate_ray_time_deployment_manifest(manifest)
    with pytest.raises(RayTimeManifestError, match="not a file"):
        validate_ray_time_deployment_manifest(
            manifest,
            torchscript_path=tmp_path / "missing.pt",
        )
    with pytest.raises(
        RayTimeManifestError,
        match=r"export_artifacts.torchscript.file_name",
    ):
        validate_ray_time_deployment_manifest(
            manifest,
            torchscript_path=onnx,
        )
    validate_ray_time_deployment_manifest(
        manifest,
        torchscript_path=torchscript,
    )


def test_integrity_hash_detects_contract_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Global")
    manifest["contract"]["action"]["per_joint_action_scale"][0] += 0.001
    manifest["checkpoint"]["training_metadata"]["resolved_contract"][
        "action_per_joint_scale"
    ][0] += 0.001

    with pytest.raises(RayTimeManifestError, match="payload_sha256"):
        validate_ray_time_deployment_manifest(manifest)


def test_different_existing_sidecar_is_not_overwritten(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    manifest = _manifest(checkpoint, history_length=5, variant="Global")
    destination = tmp_path / "model.manifest.json"
    destination.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RayTimeManifestError, match="Refusing to overwrite"):
        write_ray_time_deployment_manifest(destination, manifest)
    assert destination.read_text(encoding="utf-8") == "{}\n"


def test_git_provenance_binds_commit_and_dirty_worktree(tmp_path: Path) -> None:
    repositories = {}
    for name in ("lab_pro", "rsl_rl", "IsaacLab"):
        root = tmp_path / name
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Manifest Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "config",
                "user.email",
                "manifest@example.invalid",
            ],
            check=True,
        )
        (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(root), "add", "tracked.txt"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "commit", "-qm", "initial"],
            check=True,
        )
        repositories[name] = root

    clean = collect_git_provenance(repositories)
    assert all(not entry["dirty"] for entry in clean.values())
    assert all(len(entry["commit"]) == 40 for entry in clean.values())

    (repositories["lab_pro"] / "tracked.txt").write_text(
        "modified\n",
        encoding="utf-8",
    )
    (repositories["rsl_rl"] / "untracked.txt").write_text(
        "untracked\n",
        encoding="utf-8",
    )
    dirty = collect_git_provenance(repositories)
    assert dirty["lab_pro"]["dirty"]
    assert dirty["rsl_rl"]["dirty"]
    assert not dirty["IsaacLab"]["dirty"]
    assert (
        dirty["lab_pro"]["worktree_snapshot_sha256"]
        != clean["lab_pro"]["worktree_snapshot_sha256"]
    )
    assert (
        dirty["rsl_rl"]["worktree_snapshot_sha256"]
        != clean["rsl_rl"]["worktree_snapshot_sha256"]
    )


def test_reader_rejects_duplicate_keys_and_non_finite_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":{},"schema":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RayTimeManifestError, match="duplicate JSON key"):
        read_ray_time_deployment_manifest(duplicate)

    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(RayTimeManifestError, match="non-finite JSON"):
        read_ray_time_deployment_manifest(non_finite)
