# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict, CPU-only replay contract for recorded Livox ``CustomMsg`` data.

The artifact described here is a sealed extraction boundary, not a ROS bag
reader.  It preserves the exact source-recording bytes while storing only the
two numeric arrays needed by :func:`point_packet_from_livox_custom_msg_arrays`
for each policy window.  Every extracted point must be accounted for by the
``CustomMsg.point_num`` total declared in the window metadata.

The loader deliberately rejects permissive NumPy object archives, non-
canonical JSON, undeclared files, changed source/manifest bytes, cross-clock
windows, and ambiguous units.  It imports neither ROS, Isaac nor CUDA.
"""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .mid360_ray_time_builder import (
    MID360_NORMALIZED_SENSOR_FRAME,
    Mid360PacketStats,
    Mid360RayTimeTensorBuilder,
    point_packet_from_livox_custom_msg_arrays,
)
from .ray_time_deployment_manifest import (
    canonical_json_bytes,
    validate_ray_time_deployment_manifest,
)

LIVOX_REPLAY_SCHEMA_NAME = "rsl_rl.livox_custom_msg_replay_artifact"
LIVOX_REPLAY_SCHEMA_VERSION = 1
LIVOX_REPLAY_RECEIPT_SCHEMA_NAME = "rsl_rl.livox_custom_msg_replay_receipt"
LIVOX_REPLAY_RECEIPT_SCHEMA_VERSION = 1
LIVOX_REPLAY_METADATA_FILE = "replay_manifest.json"
LIVOX_REPLAY_SOURCE_FILE = "source_recording.bin"
LIVOX_REPLAY_DEPLOYMENT_MANIFEST_FILE = "deployment_manifest.json"
LIVOX_REPLAY_XYZ_UNIT = "metre"
LIVOX_REPLAY_OFFSET_TIME_UNIT = "nanosecond"
LIVOX_REPLAY_CAPTURE_TIME_UNIT = "nanosecond"
LIVOX_REPLAY_SYNTHETIC_FORMAT = "synthetic_contract_fixture"
LIVOX_REPLAY_MAX_WINDOWS = 100_000
LIVOX_REPLAY_MAX_POINTS_PER_WINDOW = 10_000_000
LIVOX_REPLAY_MAX_INFERRED_GAP = 10_000

_WINDOW_ARRAY_KEYS = ("offset_time_ns", "xyz_m")
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_INT64_MAX = int(np.iinfo(np.int64).max)


class LivoxCustomMsgReplayError(ValueError):
    """Raised when a replay artifact is incomplete, ambiguous, or changed."""


@dataclass(frozen=True)
class LivoxCustomMsgReplayWindow:
    """One extracted policy window from one or more Livox ``CustomMsg`` values.

    ``source_point_count`` must be the original ``point_num`` field before any
    array slicing. Equality with both array lengths is mandatory; this is the
    fail-closed boundary against silent point truncation. Schema v1 accepts
    exactly one native ``CustomMsg`` per policy window because it preserves one
    message ``timebase``. Multi-message aggregation requires an absolute-time
    adapter and is intentionally outside this schema.
    """

    window_index: int
    timebase_ns: int
    offset_time_ns: np.ndarray
    xyz_m: np.ndarray
    capture_end_time_ns: int
    received_time_ns: int
    source_point_count: int
    source_message_count: int = 1


@dataclass(frozen=True)
class LoadedLivoxReplayWindow:
    """Validated numeric arrays and metadata for one replay window."""

    metadata: Mapping[str, Any]
    xyz_m: np.ndarray
    offset_time_ns: np.ndarray
    archive_sha256: str


@dataclass(frozen=True)
class LoadedLivoxReplayArtifact:
    """A fully byte-validated replay artifact loaded into CPU memory."""

    root: Path
    metadata: Mapping[str, Any]
    metadata_bytes_sha256: str
    deployment_manifest: Mapping[str, Any]
    windows: tuple[LoadedLivoxReplayWindow, ...]


@dataclass(frozen=True)
class LivoxReplayWindowResult:
    """Production-builder output and receipt evidence for one window."""

    window_index: int
    policy_tensor: np.ndarray
    tensor_sha256: str
    stats: Mid360PacketStats


@dataclass(frozen=True)
class LivoxReplayResult:
    """All replayed tensors plus their canonical evidence receipt."""

    receipt: Mapping[str, Any]
    windows: tuple[LivoxReplayWindowResult, ...]


def write_livox_custom_msg_replay_artifact(
    artifact_dir: str | os.PathLike[str],
    *,
    raw_recording_path: str | os.PathLike[str],
    deployment_manifest_path: str | os.PathLike[str],
    windows: Sequence[LivoxCustomMsgReplayWindow],
    monotonic_clock_domain: str,
    source_recording_format: str,
    real_data_present: bool,
) -> Path:
    """Write one sealed replay directory without overwriting an existing path."""
    target = Path(artifact_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Replay artifact path already exists: {target}")
    parent = target.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)

    clock_domain = _nonempty_string(monotonic_clock_domain, "monotonic_clock_domain")
    source_format = _nonempty_string(source_recording_format, "source_recording_format")
    if type(real_data_present) is not bool:
        raise TypeError("real_data_present must be bool.")
    if real_data_present and source_format == LIVOX_REPLAY_SYNTHETIC_FORMAT:
        raise LivoxCustomMsgReplayError("Real data cannot declare the synthetic fixture recording format.")
    if not real_data_present and source_format != LIVOX_REPLAY_SYNTHETIC_FORMAT:
        raise LivoxCustomMsgReplayError("A non-real artifact must use synthetic_contract_fixture.")

    raw_path = _regular_file(raw_recording_path, "raw_recording_path")
    manifest_path = _regular_file(deployment_manifest_path, "deployment_manifest_path")
    raw_size, raw_sha256 = _file_size_sha256(raw_path)
    if real_data_present and raw_size == 0:
        raise LivoxCustomMsgReplayError("A real source recording cannot be empty.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _parse_canonical_deployment_manifest(manifest_bytes)

    validated_windows = _validate_input_windows(windows)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        (staging / "windows").mkdir()
        shutil.copyfile(raw_path, staging / LIVOX_REPLAY_SOURCE_FILE)
        (staging / LIVOX_REPLAY_DEPLOYMENT_MANIFEST_FILE).write_bytes(manifest_bytes)
        window_records: list[dict[str, Any]] = []
        for window in validated_windows:
            archive_name = f"windows/window_{window.window_index:020d}.npz"
            archive_path = staging / archive_name
            np.savez(
                archive_path,
                offset_time_ns=window.offset_time_ns,
                xyz_m=window.xyz_m,
            )
            archive_size, archive_sha256 = _file_size_sha256(archive_path)
            window_records.append({
                "archive_file": archive_name,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_size,
                "capture_end_time_ns": window.capture_end_time_ns,
                "clock_domain": clock_domain,
                "coordinate_frame": MID360_NORMALIZED_SENSOR_FRAME,
                "offset_time_unit": LIVOX_REPLAY_OFFSET_TIME_UNIT,
                "point_count": int(window.xyz_m.shape[0]),
                "received_time_ns": window.received_time_ns,
                "source_message_count": window.source_message_count,
                "source_point_count": window.source_point_count,
                "timebase_ns": window.timebase_ns,
                "window_index": window.window_index,
                "xyz_unit": LIVOX_REPLAY_XYZ_UNIT,
            })

        payload: dict[str, Any] = {
            "coordinate_contract": {
                "coordinate_frame": MID360_NORMALIZED_SENSOR_FRAME,
                "xyz_unit": LIVOX_REPLAY_XYZ_UNIT,
            },
            "data_status": {
                "dataset_class": ("recorded_livox_custom_msg" if real_data_present else "synthetic_contract_fixture"),
                "real_data_present": real_data_present,
                "real_replay_verified": False,
                "training_ready": False,
            },
            "deployment_manifest": {
                "file_name": LIVOX_REPLAY_DEPLOYMENT_MANIFEST_FILE,
                "file_sha256": _sha256(manifest_bytes),
                "payload_sha256": manifest["integrity"]["payload_sha256"],
                "size_bytes": len(manifest_bytes),
            },
            "schema": {
                "name": LIVOX_REPLAY_SCHEMA_NAME,
                "version": LIVOX_REPLAY_SCHEMA_VERSION,
            },
            "source_recording": {
                "artifact_file_name": LIVOX_REPLAY_SOURCE_FILE,
                "original_file_name": raw_path.name,
                "recording_format": source_format,
                "sha256": raw_sha256,
                "size_bytes": raw_size,
            },
            "time_contract": {
                "capture_time_unit": LIVOX_REPLAY_CAPTURE_TIME_UNIT,
                "clock_domain": clock_domain,
                "offset_time_unit": LIVOX_REPLAY_OFFSET_TIME_UNIT,
                "point_time_formula": "timebase_ns + offset_time_ns",
            },
            "windows": window_records,
        }
        payload["integrity"] = _integrity(payload)
        (staging / LIVOX_REPLAY_METADATA_FILE).write_bytes(canonical_json_bytes(payload) + b"\n")
        # Re-read through the public strict loader before publishing the path.
        load_livox_custom_msg_replay_artifact(staging)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_livox_custom_msg_replay_artifact(
    artifact_dir: str | os.PathLike[str],
) -> LoadedLivoxReplayArtifact:
    """Validate every byte and load numeric windows with ``allow_pickle=False``."""
    supplied_root = Path(artifact_dir)
    if supplied_root.is_symlink():
        raise LivoxCustomMsgReplayError(f"Replay artifact root cannot be a symlink: {supplied_root}")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise LivoxCustomMsgReplayError(f"Replay artifact must be a real directory: {root}")
    metadata_path = _regular_file(root / LIVOX_REPLAY_METADATA_FILE, "metadata")
    metadata_bytes = metadata_path.read_bytes()
    metadata = _parse_canonical_metadata(metadata_bytes)
    _validate_metadata(metadata)

    expected_files = {
        LIVOX_REPLAY_METADATA_FILE,
        LIVOX_REPLAY_SOURCE_FILE,
        LIVOX_REPLAY_DEPLOYMENT_MANIFEST_FILE,
        *(record["archive_file"] for record in metadata["windows"]),
    }
    actual_files: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise LivoxCustomMsgReplayError(f"Replay artifacts cannot contain symlinks: {candidate}")
        if candidate.is_file():
            actual_files.add(candidate.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise LivoxCustomMsgReplayError(
            "Replay artifact file set differs from its sealed metadata: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}."
        )

    source = metadata["source_recording"]
    _verify_file_record(
        root / LIVOX_REPLAY_SOURCE_FILE,
        size_bytes=source["size_bytes"],
        sha256=source["sha256"],
        label="source recording",
    )
    manifest_record = metadata["deployment_manifest"]
    manifest_path = root / LIVOX_REPLAY_DEPLOYMENT_MANIFEST_FILE
    _verify_file_record(
        manifest_path,
        size_bytes=manifest_record["size_bytes"],
        sha256=manifest_record["file_sha256"],
        label="deployment manifest",
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest = _parse_canonical_deployment_manifest(manifest_bytes)
    if manifest["integrity"]["payload_sha256"] != manifest_record["payload_sha256"]:
        raise LivoxCustomMsgReplayError("Deployment manifest payload SHA-256 differs from replay metadata.")

    loaded_windows: list[LoadedLivoxReplayWindow] = []
    for record in metadata["windows"]:
        archive_path = root / record["archive_file"]
        _verify_file_record(
            archive_path,
            size_bytes=record["archive_size_bytes"],
            sha256=record["archive_sha256"],
            label=f"window {record['window_index']} archive",
        )
        xyz, offsets = _load_numeric_archive(
            archive_path,
            point_count=record["point_count"],
            source_point_count=record["source_point_count"],
        )
        _validate_window_times(record, offsets)
        loaded_windows.append(
            LoadedLivoxReplayWindow(
                metadata=record,
                xyz_m=xyz,
                offset_time_ns=offsets,
                archive_sha256=record["archive_sha256"],
            )
        )
    return LoadedLivoxReplayArtifact(
        root=root,
        metadata=metadata,
        metadata_bytes_sha256=_sha256(metadata_bytes),
        deployment_manifest=manifest,
        windows=tuple(loaded_windows),
    )


def replay_livox_custom_msg_artifact(
    artifact_dir: str | os.PathLike[str],
    *,
    max_packet_age_s: float,
    timestamp_tolerance_s: float = 1.0e-6,
    packet_cadence_tolerance_s: float = 0.02,
) -> LivoxReplayResult:
    """Replay all windows through the production builder and seal a receipt."""
    artifact = load_livox_custom_msg_replay_artifact(artifact_dir)
    clock_domain = artifact.metadata["time_contract"]["clock_domain"]
    builder = Mid360RayTimeTensorBuilder(
        artifact.deployment_manifest,
        max_packet_age_s=max_packet_age_s,
        timestamp_tolerance_s=timestamp_tolerance_s,
        packet_cadence_tolerance_s=packet_cadence_tolerance_s,
        monotonic_clock_domain=clock_domain,
    )
    results: list[LivoxReplayWindowResult] = []
    receipt_windows: list[dict[str, Any]] = []
    inferred_indices: list[int] = []
    previous_index: int | None = None
    for loaded in artifact.windows:
        record = loaded.metadata
        if previous_index is not None:
            inferred_indices.extend(range(previous_index + 1, record["window_index"]))
        previous_index = record["window_index"]
        capture_end_s = np.float64(record["capture_end_time_ns"] * 1.0e-9)
        received_time_s = np.float64(record["received_time_ns"] * 1.0e-9)
        packet = point_packet_from_livox_custom_msg_arrays(
            xyz_m=loaded.xyz_m,
            timebase_ns=record["timebase_ns"],
            offset_time_ns=loaded.offset_time_ns,
            coordinate_frame=record["coordinate_frame"],
            window_index=record["window_index"],
            capture_end_s=capture_end_s,
            received_time_s=received_time_s,
            monotonic_clock_domain=record["clock_domain"],
        )
        stats = builder.ingest_point_packet(packet)
        tensor = builder.policy_tensor(now_s=received_time_s)
        tensor_hash = _numeric_array_sha256(tensor)
        results.append(
            LivoxReplayWindowResult(
                window_index=record["window_index"],
                policy_tensor=tensor,
                tensor_sha256=tensor_hash,
                stats=stats,
            )
        )
        receipt_windows.append({
            "archive_sha256": loaded.archive_sha256,
            "input_point_count": record["point_count"],
            "policy_tensor": {
                "dtype": tensor.dtype.str,
                "sha256": tensor_hash,
                "shape": list(tensor.shape),
            },
            "stats": asdict(stats),
            "window_index": record["window_index"],
        })

    real_data_present = artifact.metadata["data_status"]["real_data_present"]
    real_replay_verified = bool(real_data_present)
    receipt: dict[str, Any] = {
        "artifact": {
            "deployment_manifest_file_sha256": artifact.metadata["deployment_manifest"]["file_sha256"],
            "deployment_manifest_payload_sha256": artifact.metadata["deployment_manifest"]["payload_sha256"],
            "metadata_bytes_sha256": artifact.metadata_bytes_sha256,
            "source_recording_sha256": artifact.metadata["source_recording"]["sha256"],
            "source_recording_size_bytes": artifact.metadata["source_recording"]["size_bytes"],
        },
        "dropped_window_accounting": {
            "accounting_scope": "strictly_between_first_and_last_recorded_window",
            "builder_explicit_dropped_total": builder.explicit_dropped_packets_total,
            "builder_implicit_missing_total": builder.implicit_missing_packets_total,
            "inferred_missing_window_count": len(inferred_indices),
            "inferred_missing_window_indices": inferred_indices,
        },
        "gate": {
            "REAL-LIVOX-REPLAY-001": ("verified" if real_replay_verified else "open_no_real_recording"),
            "engineering_contract_verified": True,
            "real_data_present": real_data_present,
            "real_replay_verified": real_replay_verified,
            "training_ready": False,
        },
        "replay_config": {
            "clock_domain": clock_domain,
            "max_packet_age_s": float(max_packet_age_s),
            "packet_cadence_tolerance_s": float(packet_cadence_tolerance_s),
            "timestamp_tolerance_s": float(timestamp_tolerance_s),
        },
        "schema": {
            "name": LIVOX_REPLAY_RECEIPT_SCHEMA_NAME,
            "version": LIVOX_REPLAY_RECEIPT_SCHEMA_VERSION,
        },
        "windows": receipt_windows,
    }
    receipt["integrity"] = _integrity(receipt)
    validate_livox_custom_msg_replay_receipt(receipt)
    return LivoxReplayResult(receipt=receipt, windows=tuple(results))


def validate_livox_custom_msg_replay_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Validate exact receipt structure and detached payload integrity."""
    _require_exact_keys(
        receipt,
        ("artifact", "dropped_window_accounting", "gate", "integrity", "replay_config", "schema", "windows"),
        "$",
    )
    schema = _mapping(receipt["schema"], "$.schema")
    _require_exact_keys(schema, ("name", "version"), "$.schema")
    if schema != {
        "name": LIVOX_REPLAY_RECEIPT_SCHEMA_NAME,
        "version": LIVOX_REPLAY_RECEIPT_SCHEMA_VERSION,
    }:
        raise LivoxCustomMsgReplayError("Replay receipt schema is unsupported.")
    gate = _mapping(receipt["gate"], "$.gate")
    _require_exact_keys(
        gate,
        (
            "REAL-LIVOX-REPLAY-001",
            "engineering_contract_verified",
            "real_data_present",
            "real_replay_verified",
            "training_ready",
        ),
        "$.gate",
    )
    for name in ("engineering_contract_verified", "real_data_present", "real_replay_verified", "training_ready"):
        if type(gate[name]) is not bool:
            raise LivoxCustomMsgReplayError(f"$.gate.{name} must be bool.")
    if gate["training_ready"]:
        raise LivoxCustomMsgReplayError("Replay evidence cannot promote training_ready.")
    if gate["real_replay_verified"] != gate["real_data_present"]:
        raise LivoxCustomMsgReplayError("real_replay_verified must exactly follow sealed real_data_present.")
    expected_gate = "verified" if gate["real_data_present"] else "open_no_real_recording"
    if gate["REAL-LIVOX-REPLAY-001"] != expected_gate:
        raise LivoxCustomMsgReplayError("REAL-LIVOX-REPLAY-001 status is inconsistent.")

    artifact = _mapping(receipt["artifact"], "$.artifact")
    _require_exact_keys(
        artifact,
        (
            "deployment_manifest_file_sha256",
            "deployment_manifest_payload_sha256",
            "metadata_bytes_sha256",
            "source_recording_sha256",
            "source_recording_size_bytes",
        ),
        "$.artifact",
    )
    for name in (
        "deployment_manifest_file_sha256",
        "deployment_manifest_payload_sha256",
        "metadata_bytes_sha256",
        "source_recording_sha256",
    ):
        _sha256_string(artifact[name], f"$.artifact.{name}")
    _nonnegative_int(
        artifact["source_recording_size_bytes"],
        "$.artifact.source_recording_size_bytes",
    )

    config = _mapping(receipt["replay_config"], "$.replay_config")
    _require_exact_keys(
        config,
        (
            "clock_domain",
            "max_packet_age_s",
            "packet_cadence_tolerance_s",
            "timestamp_tolerance_s",
        ),
        "$.replay_config",
    )
    _nonempty_string(config["clock_domain"], "$.replay_config.clock_domain")
    if _finite_float(config["max_packet_age_s"], "$.replay_config.max_packet_age_s") <= 0.0:
        raise LivoxCustomMsgReplayError("max_packet_age_s must be positive.")
    for name in ("packet_cadence_tolerance_s", "timestamp_tolerance_s"):
        if _finite_float(config[name], f"$.replay_config.{name}") < 0.0:
            raise LivoxCustomMsgReplayError(f"$.replay_config.{name} must be non-negative.")

    accounting = _mapping(
        receipt["dropped_window_accounting"],
        "$.dropped_window_accounting",
    )
    _require_exact_keys(
        accounting,
        (
            "accounting_scope",
            "builder_explicit_dropped_total",
            "builder_implicit_missing_total",
            "inferred_missing_window_count",
            "inferred_missing_window_indices",
        ),
        "$.dropped_window_accounting",
    )
    if accounting["accounting_scope"] != "strictly_between_first_and_last_recorded_window":
        raise LivoxCustomMsgReplayError("Dropped-window accounting scope is unsupported.")
    if (
        _nonnegative_int(
            accounting["builder_explicit_dropped_total"],
            "$.dropped_window_accounting.builder_explicit_dropped_total",
        )
        != 0
    ):
        raise LivoxCustomMsgReplayError("This replay schema never inserts explicit drops.")
    implicit_total = _nonnegative_int(
        accounting["builder_implicit_missing_total"],
        "$.dropped_window_accounting.builder_implicit_missing_total",
    )
    missing_count = _nonnegative_int(
        accounting["inferred_missing_window_count"],
        "$.dropped_window_accounting.inferred_missing_window_count",
    )
    missing = accounting["inferred_missing_window_indices"]
    if not isinstance(missing, list) or any(
        _nonnegative_int(item, "inferred_missing_window_indices[*]") != item for item in missing
    ):
        raise LivoxCustomMsgReplayError("Missing-window indices must be integers.")
    if missing != sorted(set(missing)) or len(missing) != missing_count or implicit_total != missing_count:
        raise LivoxCustomMsgReplayError("Dropped-window counts and indices disagree.")

    window_values = receipt["windows"]
    if not isinstance(window_values, list) or not window_values:
        raise LivoxCustomMsgReplayError("Receipt windows must be a non-empty array.")
    expected_stats_keys = tuple(field.name for field in fields(Mid360PacketStats))
    previous: int | None = None
    derived_missing: list[int] = []
    stats_implicit_total = 0
    for position, raw in enumerate(window_values):
        window = _mapping(raw, f"$.windows[{position}]")
        _require_exact_keys(
            window,
            ("archive_sha256", "input_point_count", "policy_tensor", "stats", "window_index"),
            f"$.windows[{position}]",
        )
        index = _window_index(window["window_index"], f"$.windows[{position}].window_index")
        if previous is not None:
            if index <= previous:
                raise LivoxCustomMsgReplayError("Receipt window indices must strictly increase.")
            derived_missing.extend(range(previous + 1, index))
        previous = index
        _sha256_string(window["archive_sha256"], f"$.windows[{position}].archive_sha256")
        point_count = _nonnegative_int(window["input_point_count"], f"$.windows[{position}].input_point_count")
        tensor = _mapping(window["policy_tensor"], f"$.windows[{position}].policy_tensor")
        _require_exact_keys(tensor, ("dtype", "sha256", "shape"), f"$.windows[{position}].policy_tensor")
        if tensor["dtype"] != "<f2":
            raise LivoxCustomMsgReplayError("Receipt policy tensors must be little-endian float16.")
        shape = tensor["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 5
            or shape[0] != 1
            or shape[2] != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        ):
            raise LivoxCustomMsgReplayError("Receipt policy tensor shape is invalid.")
        _sha256_string(tensor["sha256"], f"$.windows[{position}].policy_tensor.sha256")
        stats = _mapping(window["stats"], f"$.windows[{position}].stats")
        _require_exact_keys(stats, expected_stats_keys, f"$.windows[{position}].stats")
        if stats["window_index"] != index or stats["input_return_points"] != point_count:
            raise LivoxCustomMsgReplayError("Receipt window input counts disagree with builder stats.")
        if stats["manifest_payload_sha256"] != artifact["deployment_manifest_payload_sha256"]:
            raise LivoxCustomMsgReplayError("Builder stats use a different deployment manifest.")
        if stats["explicit_drop"] is not False:
            raise LivoxCustomMsgReplayError("Recorded replay windows cannot be explicit drops.")
        stats_implicit_total += _nonnegative_int(
            stats["implicit_missing_packets_inserted"],
            f"$.windows[{position}].stats.implicit_missing_packets_inserted",
        )
    if derived_missing != missing or stats_implicit_total != implicit_total:
        raise LivoxCustomMsgReplayError("Receipt windows do not reproduce dropped-window accounting.")
    _validate_integrity(receipt)


def write_livox_custom_msg_replay_receipt(path: str | os.PathLike[str], receipt: Mapping[str, Any]) -> Path:
    """Write a validated canonical receipt without replacing existing evidence."""
    validate_livox_custom_msg_replay_receipt(receipt)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Receipt path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return destination


def _validate_input_windows(
    windows: Sequence[LivoxCustomMsgReplayWindow],
) -> tuple[LivoxCustomMsgReplayWindow, ...]:
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise TypeError("windows must be a sequence of replay windows.")
    if not windows:
        raise LivoxCustomMsgReplayError("Replay artifact requires at least one window.")
    if len(windows) > LIVOX_REPLAY_MAX_WINDOWS:
        raise LivoxCustomMsgReplayError("Replay artifact contains too many windows.")
    validated: list[LivoxCustomMsgReplayWindow] = []
    previous: int | None = None
    previous_timebase: int | None = None
    previous_capture_end: int | None = None
    previous_received: int | None = None
    for position, window in enumerate(windows):
        if not isinstance(window, LivoxCustomMsgReplayWindow):
            raise TypeError(f"windows[{position}] must be LivoxCustomMsgReplayWindow.")
        index = _window_index(window.window_index, f"windows[{position}].window_index")
        if previous is not None:
            if index <= previous:
                raise LivoxCustomMsgReplayError("Window indices must be strictly increasing and unique.")
            if index - previous - 1 > LIVOX_REPLAY_MAX_INFERRED_GAP:
                raise LivoxCustomMsgReplayError("Window gap exceeds the audited limit.")
        previous = index
        xyz = _canonical_xyz(window.xyz_m, f"windows[{position}].xyz_m")
        offsets = _canonical_offsets(window.offset_time_ns, f"windows[{position}].offset_time_ns")
        if xyz.shape[0] != offsets.shape[0]:
            raise LivoxCustomMsgReplayError("xyz_m and offset_time_ns must have exactly the same point count.")
        source_count = _nonnegative_int(window.source_point_count, f"windows[{position}].source_point_count")
        if source_count != xyz.shape[0]:
            raise LivoxCustomMsgReplayError(
                "source_point_count must equal both extracted array lengths; silent point truncation is forbidden."
            )
        source_messages = _positive_int(window.source_message_count, f"windows[{position}].source_message_count")
        if source_messages != 1:
            raise LivoxCustomMsgReplayError("Replay schema v1 requires exactly one source CustomMsg per window.")
        timebase = _uint64(window.timebase_ns, f"windows[{position}].timebase_ns")
        capture_end = _uint64(window.capture_end_time_ns, f"windows[{position}].capture_end_time_ns")
        received = _uint64(window.received_time_ns, f"windows[{position}].received_time_ns")
        record = LivoxCustomMsgReplayWindow(
            window_index=index,
            timebase_ns=timebase,
            offset_time_ns=offsets,
            xyz_m=xyz,
            capture_end_time_ns=capture_end,
            received_time_ns=received,
            source_point_count=source_count,
            source_message_count=source_messages,
        )
        _validate_window_times(
            {
                "timebase_ns": timebase,
                "capture_end_time_ns": capture_end,
                "received_time_ns": received,
            },
            offsets,
        )
        if previous_timebase is not None:
            assert previous_capture_end is not None
            assert previous_received is not None
            if timebase <= previous_timebase or capture_end <= previous_capture_end or received <= previous_received:
                raise LivoxCustomMsgReplayError(
                    "Window timebase, capture end, and received time must be strictly increasing with window_index."
                )
        previous_timebase = timebase
        previous_capture_end = capture_end
        previous_received = received
        validated.append(record)
    return tuple(validated)


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    _require_exact_keys(
        metadata,
        (
            "coordinate_contract",
            "data_status",
            "deployment_manifest",
            "integrity",
            "schema",
            "source_recording",
            "time_contract",
            "windows",
        ),
        "$",
    )
    schema = _mapping(metadata["schema"], "$.schema")
    if schema != {"name": LIVOX_REPLAY_SCHEMA_NAME, "version": LIVOX_REPLAY_SCHEMA_VERSION}:
        raise LivoxCustomMsgReplayError("Replay artifact schema is unsupported.")
    coordinate = _mapping(metadata["coordinate_contract"], "$.coordinate_contract")
    if coordinate != {"coordinate_frame": MID360_NORMALIZED_SENSOR_FRAME, "xyz_unit": LIVOX_REPLAY_XYZ_UNIT}:
        raise LivoxCustomMsgReplayError("Coordinate frame/unit contract is not canonical.")
    timing = _mapping(metadata["time_contract"], "$.time_contract")
    _require_exact_keys(
        timing, ("capture_time_unit", "clock_domain", "offset_time_unit", "point_time_formula"), "$.time_contract"
    )
    if (
        timing["capture_time_unit"] != LIVOX_REPLAY_CAPTURE_TIME_UNIT
        or timing["offset_time_unit"] != LIVOX_REPLAY_OFFSET_TIME_UNIT
        or timing["point_time_formula"] != "timebase_ns + offset_time_ns"
    ):
        raise LivoxCustomMsgReplayError("Time unit/formula contract is not canonical.")
    clock = _nonempty_string(timing["clock_domain"], "$.time_contract.clock_domain")
    status = _mapping(metadata["data_status"], "$.data_status")
    _require_exact_keys(
        status, ("dataset_class", "real_data_present", "real_replay_verified", "training_ready"), "$.data_status"
    )
    for name in ("real_data_present", "real_replay_verified", "training_ready"):
        if type(status[name]) is not bool:
            raise LivoxCustomMsgReplayError(f"$.data_status.{name} must be bool.")
    if status["real_replay_verified"] or status["training_ready"]:
        raise LivoxCustomMsgReplayError("An input artifact cannot pre-claim replay verification or training readiness.")
    expected_class = "recorded_livox_custom_msg" if status["real_data_present"] else "synthetic_contract_fixture"
    if status["dataset_class"] != expected_class:
        raise LivoxCustomMsgReplayError("Dataset class disagrees with real_data_present.")

    source = _mapping(metadata["source_recording"], "$.source_recording")
    _require_exact_keys(
        source,
        ("artifact_file_name", "original_file_name", "recording_format", "sha256", "size_bytes"),
        "$.source_recording",
    )
    if source["artifact_file_name"] != LIVOX_REPLAY_SOURCE_FILE:
        raise LivoxCustomMsgReplayError("Source recording file name is not canonical.")
    original_name = _nonempty_string(source["original_file_name"], "$.source_recording.original_file_name")
    if Path(original_name).name != original_name or original_name in (".", ".."):
        raise LivoxCustomMsgReplayError("Original recording name must be a basename.")
    source_format = _nonempty_string(source["recording_format"], "$.source_recording.recording_format")
    if status["real_data_present"] == (source_format == LIVOX_REPLAY_SYNTHETIC_FORMAT):
        raise LivoxCustomMsgReplayError("Recording format disagrees with real-data status.")
    _sha256_string(source["sha256"], "$.source_recording.sha256")
    _nonnegative_int(source["size_bytes"], "$.source_recording.size_bytes")
    if status["real_data_present"] and source["size_bytes"] == 0:
        raise LivoxCustomMsgReplayError("A real source recording cannot be empty.")

    manifest = _mapping(metadata["deployment_manifest"], "$.deployment_manifest")
    _require_exact_keys(manifest, ("file_name", "file_sha256", "payload_sha256", "size_bytes"), "$.deployment_manifest")
    if manifest["file_name"] != LIVOX_REPLAY_DEPLOYMENT_MANIFEST_FILE:
        raise LivoxCustomMsgReplayError("Deployment manifest file name is not canonical.")
    _sha256_string(manifest["file_sha256"], "$.deployment_manifest.file_sha256")
    _sha256_string(manifest["payload_sha256"], "$.deployment_manifest.payload_sha256")
    _positive_int(manifest["size_bytes"], "$.deployment_manifest.size_bytes")

    records = metadata["windows"]
    if not isinstance(records, list) or not records or len(records) > LIVOX_REPLAY_MAX_WINDOWS:
        raise LivoxCustomMsgReplayError("windows must be a non-empty bounded JSON array.")
    previous: int | None = None
    previous_timebase: int | None = None
    previous_capture_end: int | None = None
    previous_received: int | None = None
    for position, raw in enumerate(records):
        record = _mapping(raw, f"$.windows[{position}]")
        _require_exact_keys(
            record,
            (
                "archive_file",
                "archive_sha256",
                "archive_size_bytes",
                "capture_end_time_ns",
                "clock_domain",
                "coordinate_frame",
                "offset_time_unit",
                "point_count",
                "received_time_ns",
                "source_message_count",
                "source_point_count",
                "timebase_ns",
                "window_index",
                "xyz_unit",
            ),
            f"$.windows[{position}]",
        )
        index = _window_index(record["window_index"], f"$.windows[{position}].window_index")
        if previous is not None:
            if index <= previous:
                raise LivoxCustomMsgReplayError("Window indices must be strictly increasing and unique.")
            if index - previous - 1 > LIVOX_REPLAY_MAX_INFERRED_GAP:
                raise LivoxCustomMsgReplayError("Window gap exceeds the audited limit.")
        previous = index
        expected_archive = f"windows/window_{index:020d}.npz"
        if record["archive_file"] != expected_archive:
            raise LivoxCustomMsgReplayError("Window archive name is not canonical.")
        _sha256_string(record["archive_sha256"], f"$.windows[{position}].archive_sha256")
        _positive_int(record["archive_size_bytes"], f"$.windows[{position}].archive_size_bytes")
        point_count = _nonnegative_int(record["point_count"], f"$.windows[{position}].point_count")
        if point_count > LIVOX_REPLAY_MAX_POINTS_PER_WINDOW:
            raise LivoxCustomMsgReplayError("Window point count exceeds the audited limit.")
        if _nonnegative_int(record["source_point_count"], f"$.windows[{position}].source_point_count") != point_count:
            raise LivoxCustomMsgReplayError("source_point_count mismatch forbids silent truncation.")
        if _positive_int(record["source_message_count"], f"$.windows[{position}].source_message_count") != 1:
            raise LivoxCustomMsgReplayError("Replay schema v1 requires exactly one source CustomMsg per window.")
        timebase = _uint64(record["timebase_ns"], f"$.windows[{position}].timebase_ns")
        capture_end = _uint64(record["capture_end_time_ns"], f"$.windows[{position}].capture_end_time_ns")
        received = _uint64(record["received_time_ns"], f"$.windows[{position}].received_time_ns")
        if previous_timebase is not None:
            assert previous_capture_end is not None
            assert previous_received is not None
            if timebase <= previous_timebase or capture_end <= previous_capture_end or received <= previous_received:
                raise LivoxCustomMsgReplayError(
                    "Window timebase, capture end, and received time must be strictly increasing with window_index."
                )
        previous_timebase = timebase
        previous_capture_end = capture_end
        previous_received = received
        if record["clock_domain"] != clock:
            raise LivoxCustomMsgReplayError("Cross-clock replay windows are forbidden.")
        if (
            record["coordinate_frame"] != MID360_NORMALIZED_SENSOR_FRAME
            or record["xyz_unit"] != LIVOX_REPLAY_XYZ_UNIT
            or record["offset_time_unit"] != LIVOX_REPLAY_OFFSET_TIME_UNIT
        ):
            raise LivoxCustomMsgReplayError("Per-window coordinate/unit declaration is not canonical.")
    _validate_integrity(metadata)


def _load_numeric_archive(path: Path, *, point_count: int, source_point_count: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            names = tuple(sorted(item.filename for item in members))
            if names != ("offset_time_ns.npy", "xyz_m.npy"):
                raise LivoxCustomMsgReplayError(f"Window NPZ must contain exactly {_WINDOW_ARRAY_KEYS}; got {names}.")
            for member in members:
                if member.is_dir() or member.flag_bits & 0x1:
                    raise LivoxCustomMsgReplayError("Encrypted/directory NPZ members are forbidden.")
                if member.file_size > 3 * LIVOX_REPLAY_MAX_POINTS_PER_WINDOW * 8 + 1_000_000:
                    raise LivoxCustomMsgReplayError("NPZ member exceeds the audited size bound.")
    except (OSError, zipfile.BadZipFile) as exc:
        raise LivoxCustomMsgReplayError(f"Invalid or truncated window NPZ: {path}") from exc
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if tuple(sorted(arrays.files)) != _WINDOW_ARRAY_KEYS:
                raise LivoxCustomMsgReplayError("Window NPZ keys are not canonical.")
            xyz = arrays["xyz_m"]
            offsets = arrays["offset_time_ns"]
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise LivoxCustomMsgReplayError("Window archive is truncated, object/pickle based, or invalid.") from exc
    xyz = _canonical_xyz(xyz, "archive.xyz_m")
    offsets = _canonical_offsets(offsets, "archive.offset_time_ns")
    if xyz.shape[0] != point_count or offsets.shape[0] != point_count or source_point_count != point_count:
        raise LivoxCustomMsgReplayError(
            "Archive array lengths, point_count, and source_point_count must match exactly."
        )
    return xyz, offsets


def _parse_canonical_metadata(raw: bytes) -> Mapping[str, Any]:
    value = _parse_json_no_duplicates(raw, "replay metadata")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise LivoxCustomMsgReplayError("Replay metadata must be canonical UTF-8 JSON with one final newline.")
    return _mapping(value, "$")


def _parse_canonical_deployment_manifest(raw: bytes) -> Mapping[str, Any]:
    value = _parse_json_no_duplicates(raw, "deployment manifest")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise LivoxCustomMsgReplayError("Deployment manifest bytes must use canonical JSON plus one newline.")
    manifest = _mapping(value, "deployment_manifest")
    validate_ray_time_deployment_manifest(manifest, require_export_artifact=False)
    return manifest


def _parse_json_no_duplicates(raw: bytes, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise LivoxCustomMsgReplayError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LivoxCustomMsgReplayError(f"Invalid UTF-8 JSON in {label}.") from exc


def _verify_file_record(path: Path, *, size_bytes: int, sha256: str, label: str) -> None:
    checked = _regular_file(path, label)
    actual_size, actual_sha256 = _file_size_sha256(checked)
    if actual_size != size_bytes:
        raise LivoxCustomMsgReplayError(f"{label} byte length changed or was truncated.")
    if actual_sha256 != sha256:
        raise LivoxCustomMsgReplayError(f"{label} SHA-256 mismatch.")


def _file_size_sha256(path: Path) -> tuple[int, str]:
    """Hash a regular file without loading an entire sensor recording."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_window_times(record: Mapping[str, Any], offsets: np.ndarray) -> None:
    timebase = _uint64(record["timebase_ns"], "timebase_ns")
    capture_end = _uint64(record["capture_end_time_ns"], "capture_end_time_ns")
    received = _uint64(record["received_time_ns"], "received_time_ns")
    max_offset = 0 if offsets.size == 0 else int(offsets.max())
    if timebase > _UINT64_MAX - max_offset:
        raise LivoxCustomMsgReplayError("timebase_ns + offset_time_ns overflows uint64.")
    if capture_end < timebase + max_offset:
        raise LivoxCustomMsgReplayError("capture_end_time_ns precedes a point acquisition time.")
    if received < capture_end:
        raise LivoxCustomMsgReplayError("received_time_ns precedes capture_end_time_ns.")


def _canonical_xyz(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if value.dtype != np.dtype("<f4"):
        raise LivoxCustomMsgReplayError(f"{name} must use canonical little-endian float32.")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise LivoxCustomMsgReplayError(f"{name} must have shape [N,3].")
    if value.shape[0] > LIVOX_REPLAY_MAX_POINTS_PER_WINDOW:
        raise LivoxCustomMsgReplayError(f"{name} exceeds the point-count bound.")
    if not value.flags.c_contiguous:
        raise LivoxCustomMsgReplayError(f"{name} must be C-contiguous.")
    if not np.isfinite(value).all():
        raise LivoxCustomMsgReplayError(f"{name} contains NaN or Inf.")
    return value.copy()


def _canonical_offsets(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if value.dtype != np.dtype("<u4"):
        raise LivoxCustomMsgReplayError(f"{name} must use canonical little-endian uint32.")
    if value.ndim != 1:
        raise LivoxCustomMsgReplayError(f"{name} must have shape [N].")
    if value.shape[0] > LIVOX_REPLAY_MAX_POINTS_PER_WINDOW:
        raise LivoxCustomMsgReplayError(f"{name} exceeds the point-count bound.")
    if not value.flags.c_contiguous:
        raise LivoxCustomMsgReplayError(f"{name} must be C-contiguous.")
    return value.copy()


def _numeric_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"rsl_rl_numeric_array_v1\0")
    digest.update(canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)}))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _integrity(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "canonicalization": "RFC8259 JSON; UTF-8; sorted keys; no whitespace; no NaN",
        "payload_scope": "all top-level fields except integrity",
        "payload_sha256": _sha256(canonical_json_bytes(value)),
    }


def _validate_integrity(value: Mapping[str, Any]) -> None:
    integrity = _mapping(value["integrity"], "$.integrity")
    _require_exact_keys(integrity, ("algorithm", "canonicalization", "payload_scope", "payload_sha256"), "$.integrity")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    expected = _integrity(payload)
    if integrity != expected:
        raise LivoxCustomMsgReplayError("Canonical payload SHA-256 is invalid.")


def _regular_file(path: str | os.PathLike[str], label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise LivoxCustomMsgReplayError(f"{label} must be a regular, non-symlink file: {candidate}")
    return candidate


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LivoxCustomMsgReplayError(f"{name} must be a JSON object.")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    if set(value) != set(keys):
        raise LivoxCustomMsgReplayError(f"{name} must contain exactly {tuple(keys)}, got {tuple(value)}.")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or not value.isprintable():
        raise LivoxCustomMsgReplayError(f"{name} must be a non-empty printable canonical string.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise LivoxCustomMsgReplayError(f"{name} must be non-negative.")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise LivoxCustomMsgReplayError(f"{name} must be positive.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise LivoxCustomMsgReplayError(f"{name} must be finite.")
    return result


def _uint64(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result > _UINT64_MAX:
        raise LivoxCustomMsgReplayError(f"{name} exceeds uint64.")
    return result


def _window_index(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result > _INT64_MAX:
        raise LivoxCustomMsgReplayError(f"{name} exceeds signed int64.")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_string(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LivoxCustomMsgReplayError(f"{name} must be a lowercase SHA-256 hex digest.")
    return value


__all__ = [
    "LIVOX_REPLAY_CAPTURE_TIME_UNIT",
    "LIVOX_REPLAY_OFFSET_TIME_UNIT",
    "LIVOX_REPLAY_SCHEMA_NAME",
    "LIVOX_REPLAY_SCHEMA_VERSION",
    "LIVOX_REPLAY_SYNTHETIC_FORMAT",
    "LIVOX_REPLAY_XYZ_UNIT",
    "LivoxCustomMsgReplayError",
    "LivoxCustomMsgReplayWindow",
    "LivoxReplayResult",
    "LivoxReplayWindowResult",
    "LoadedLivoxReplayArtifact",
    "LoadedLivoxReplayWindow",
    "load_livox_custom_msg_replay_artifact",
    "replay_livox_custom_msg_artifact",
    "validate_livox_custom_msg_replay_receipt",
    "write_livox_custom_msg_replay_artifact",
    "write_livox_custom_msg_replay_receipt",
]
