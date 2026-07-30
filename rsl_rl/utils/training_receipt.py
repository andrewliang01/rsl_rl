# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict, pickle-safe provenance receipts for formal training checkpoints.

The receipt deliberately separates three hash domains:

* a launch receipt binds the immutable training inputs;
* an embedded checkpoint receipt binds launch provenance, progress, and its
  parent, but never attempts to contain the checkpoint file's own hash;
* a sidecar binds the completed checkpoint file bytes to the embedded receipt.

All values intended for checkpoint embedding are JSON-native types supported
by ``torch.load(..., weights_only=True)``.  File reads use a no-follow
descriptor chain and verify the visible file identity again after reading.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch


TRAINING_LAUNCH_RECEIPT_CONTRACT = "rsl_rl_formal_training_launch_receipt_v1"
CHECKPOINT_EMBEDDED_RECEIPT_CONTRACT = "rsl_rl_formal_checkpoint_embedded_receipt_v1"
CHECKPOINT_SIDECAR_CONTRACT = "rsl_rl_formal_checkpoint_sidecar_v1"
TRAINING_RECEIPT_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CHECKPOINT_NAME_RE = re.compile(r"^model_(0|[1-9][0-9]*)\.pt$")
_GIT_REPOSITORIES = ("lab_pro", "rsl_rl", "IsaacLab")

_LAUNCH_PAYLOAD_KEYS = {
    "task",
    "seed",
    "training_started_at_utc",
    "argv",
    "git",
    "configs",
    "runtime",
    "schedule",
    "selector_protocol",
    "resume",
}
_GIT_RECORD_KEYS = {
    "repository_root",
    "head",
    "tree",
    "branch",
    "clean",
    "source_state_sha256",
}
_TEXT_PAYLOAD_KEYS = {"format", "encoding", "payload_utf8", "sha256", "bytes"}
_CONFIG_KEYS = {"agent", "env", "resume_compatibility_sha256"}
_RUNTIME_KEYS = {"python", "cuda", "physics", "headless", "device"}
_PYTHON_KEYS = {"executable", "version", "implementation"}
_CUDA_KEYS = {
    "cuda_visible_devices",
    "torch_version",
    "torch_cuda_version",
    "cudnn_version",
    "device_name",
    "device_uuid",
    "compute_capability",
}
_SCHEDULE_KEYS = {
    "training_schedule_id",
    "num_envs",
    "num_steps_per_env",
    "max_iterations",
    "save_interval",
    "transitions_per_update",
    "transition_budget",
}
_SELECTOR_KEYS = {"contract", "encoding", "payload_utf8", "sha256", "bytes"}
_RESUME_KEYS = {
    "is_resume",
    "parent_checkpoint_sha256",
    "parent_embedded_receipt_sha256",
    "parent_sidecar_payload_sha256",
    "parent_updates_completed",
    "parent_consumed_transitions",
}
_PROGRESS_KEYS = {
    "filename",
    "iter",
    "updates_completed",
    "num_envs",
    "num_steps_per_env",
    "transitions_per_update",
    "consumed_transitions",
    "configured_target_updates",
}
_PARENT_KEYS = {
    "checkpoint_file_name",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "embedded_receipt_sha256",
    "sidecar_payload_sha256",
    "launch_payload_sha256",
    "updates_completed",
    "transitions_per_update",
    "consumed_transitions",
}
_FILE_IDENTITY_KEYS = {
    "device",
    "inode",
    "mode",
    "size",
    "mtime_ns",
    "ctime_ns",
}
_CHECKPOINT_FILE_KEYS = {
    "file_name",
    "sha256",
    "bytes",
}


class TrainingReceiptError(RuntimeError):
    """Raised when formal training provenance is incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise TrainingReceiptError(message)


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _validate_json_native(value: Any, *, location: str = "$", seen: set[int] | None = None) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"Non-finite float at {location}.")
        return
    if type(value) not in {dict, list}:
        _fail(
            f"Non JSON-native value at {location}: "
            f"{type(value).__name__}."
        )
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        _fail(f"Cyclic JSON value at {location}.")
    seen.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    _fail(
                        f"JSON object key at {location} must be exact str, "
                        f"not {type(key).__name__}."
                    )
                _validate_json_native(
                    item,
                    location=f"{location}.{key}",
                    seen=seen,
                )
        else:
            for index, item in enumerate(value):
                _validate_json_native(
                    item,
                    location=f"{location}[{index}]",
                    seen=seen,
                )
    finally:
        seen.remove(identity)


def canonical_training_receipt_json_bytes(value: Any) -> bytes:
    """Serialize one finite JSON-native value into the receipt encoding."""
    _validate_json_native(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_training_receipt_sha256(value: Any) -> str:
    """Return SHA-256 over :func:`canonical_training_receipt_json_bytes`."""
    return hashlib.sha256(canonical_training_receipt_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    _fail(f"Non-standard JSON numeric constant {value!r}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Duplicate JSON object key {key!r}.")
        result[key] = value
    return result


def parse_canonical_training_receipt_json(
    payload: bytes | bytearray | memoryview,
) -> Any:
    """Parse strict receipt JSON and reject non-canonical or duplicate input."""
    if type(payload) not in {bytes, bytearray, memoryview}:
        _fail("Canonical JSON payload must be bytes-like.")
    raw = bytes(payload)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TrainingReceiptError("Receipt JSON is not UTF-8.") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except TrainingReceiptError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise TrainingReceiptError("Receipt JSON is invalid.") from error
    _validate_json_native(value)
    if canonical_training_receipt_json_bytes(value) != raw:
        _fail("Receipt JSON is not in canonical byte encoding.")
    return value


def _exact_mapping(value: Any, keys: set[str], *, location: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        _fail(f"{location} keys must be exactly {sorted(keys)}, got {actual}.")
    return value


def _exact_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{location} must be an exact integer >= {minimum}.")
    return value


def _text(value: Any, *, location: str, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        _fail(f"{location} must be a{'n' if allow_empty else ' non-empty'} string.")
    return value


def _utc_timestamp(value: Any, *, location: str) -> str:
    timestamp = _text(value, location=location)
    if "T" not in timestamp or not (
        timestamp.endswith("Z") or timestamp.endswith("+00:00")
    ):
        _fail(f"{location} must be an ISO-8601 UTC timestamp.")
    normalized = (
        f"{timestamp[:-1]}+00:00" if timestamp.endswith("Z") else timestamp
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise TrainingReceiptError(
            f"{location} must be an ISO-8601 UTC timestamp."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        _fail(f"{location} must have an explicit UTC offset.")
    return timestamp


def _hash(value: Any, *, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{location} must be a lowercase SHA-256 digest.")
    return value


def _git_object(value: Any, *, location: str) -> str:
    if type(value) is not str or _GIT_OBJECT_RE.fullmatch(value) is None:
        _fail(f"{location} must be a lowercase 40- or 64-character Git object ID.")
    return value


def _clone_native(value: Any) -> Any:
    return parse_canonical_training_receipt_json(
        canonical_training_receipt_json_bytes(value)
    )


def _validate_text_payload(
    value: Any,
    *,
    location: str,
    expected_format: str | None = None,
) -> dict[str, Any]:
    record = _exact_mapping(value, _TEXT_PAYLOAD_KEYS, location=location)
    payload_format = _text(record["format"], location=f"{location}.format")
    if expected_format is not None and payload_format != expected_format:
        _fail(f"{location}.format must be {expected_format!r}.")
    if record["encoding"] != "utf-8":
        _fail(f"{location}.encoding must be 'utf-8'.")
    payload = _text(
        record["payload_utf8"],
        location=f"{location}.payload_utf8",
        allow_empty=True,
    ).encode("utf-8")
    byte_count = _exact_int(
        record["bytes"],
        location=f"{location}.bytes",
        minimum=1,
    )
    digest = _hash(record["sha256"], location=f"{location}.sha256")
    if len(payload) != byte_count or hashlib.sha256(payload).hexdigest() != digest:
        _fail(f"{location} byte count or SHA-256 does not match its payload.")
    return record


def _validate_schedule(value: Any) -> dict[str, Any]:
    schedule = _exact_mapping(value, _SCHEDULE_KEYS, location="launch.payload.schedule")
    _text(
        schedule["training_schedule_id"],
        location="launch.payload.schedule.training_schedule_id",
    )
    num_envs = _exact_int(
        schedule["num_envs"],
        location="launch.payload.schedule.num_envs",
        minimum=1,
    )
    steps = _exact_int(
        schedule["num_steps_per_env"],
        location="launch.payload.schedule.num_steps_per_env",
        minimum=1,
    )
    maximum = _exact_int(
        schedule["max_iterations"],
        location="launch.payload.schedule.max_iterations",
        minimum=1,
    )
    _exact_int(
        schedule["save_interval"],
        location="launch.payload.schedule.save_interval",
        minimum=1,
    )
    expected_per_update = num_envs * steps
    expected_budget = expected_per_update * maximum
    if (
        _exact_int(
            schedule["transitions_per_update"],
            location="launch.payload.schedule.transitions_per_update",
            minimum=1,
        )
        != expected_per_update
    ):
        _fail("schedule.transitions_per_update is not num_envs*num_steps_per_env.")
    if (
        _exact_int(
            schedule["transition_budget"],
            location="launch.payload.schedule.transition_budget",
            minimum=1,
        )
        != expected_budget
    ):
        _fail("schedule.transition_budget is not transitions_per_update*max_iterations.")
    return schedule


def validate_training_launch_receipt(value: Any) -> dict[str, Any]:
    """Validate and return a detached formal launch receipt envelope."""
    envelope = _exact_mapping(
        value,
        {"schema_version", "contract", "payload", "payload_sha256"},
        location="launch_receipt",
    )
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != TRAINING_RECEIPT_SCHEMA_VERSION
        or envelope["contract"] != TRAINING_LAUNCH_RECEIPT_CONTRACT
    ):
        _fail("Unsupported launch receipt schema or contract.")
    payload = _exact_mapping(
        envelope["payload"],
        _LAUNCH_PAYLOAD_KEYS,
        location="launch_receipt.payload",
    )
    _text(payload["task"], location="launch_receipt.payload.task")
    _exact_int(payload["seed"], location="launch_receipt.payload.seed")
    _utc_timestamp(
        payload["training_started_at_utc"],
        location="launch_receipt.payload.training_started_at_utc",
    )
    if type(payload["argv"]) is not list or not payload["argv"] or not all(
        type(item) is str for item in payload["argv"]
    ):
        _fail("launch_receipt.payload.argv must be a non-empty list of strings.")

    repositories = _exact_mapping(
        payload["git"],
        set(_GIT_REPOSITORIES),
        location="launch_receipt.payload.git",
    )
    for name in _GIT_REPOSITORIES:
        record = _exact_mapping(
            repositories[name],
            _GIT_RECORD_KEYS,
            location=f"launch_receipt.payload.git.{name}",
        )
        root = _text(
            record["repository_root"],
            location=f"launch_receipt.payload.git.{name}.repository_root",
        )
        if not Path(root).is_absolute():
            _fail(f"launch_receipt.payload.git.{name}.repository_root must be absolute.")
        _git_object(record["head"], location=f"launch_receipt.payload.git.{name}.head")
        _git_object(record["tree"], location=f"launch_receipt.payload.git.{name}.tree")
        _text(record["branch"], location=f"launch_receipt.payload.git.{name}.branch")
        if record["clean"] is not True:
            _fail(f"launch_receipt.payload.git.{name}.clean must be true.")
        _hash(
            record["source_state_sha256"],
            location=f"launch_receipt.payload.git.{name}.source_state_sha256",
        )

    configs = _exact_mapping(
        payload["configs"],
        _CONFIG_KEYS,
        location="launch_receipt.payload.configs",
    )
    _validate_text_payload(
        configs["agent"],
        location="launch_receipt.payload.configs.agent",
        expected_format="canonical_yaml_v1",
    )
    _validate_text_payload(
        configs["env"],
        location="launch_receipt.payload.configs.env",
        expected_format="canonical_yaml_v1",
    )
    _hash(
        configs["resume_compatibility_sha256"],
        location="launch_receipt.payload.configs.resume_compatibility_sha256",
    )

    runtime = _exact_mapping(
        payload["runtime"],
        _RUNTIME_KEYS,
        location="launch_receipt.payload.runtime",
    )
    python = _exact_mapping(
        runtime["python"],
        _PYTHON_KEYS,
        location="launch_receipt.payload.runtime.python",
    )
    for key in _PYTHON_KEYS:
        _text(python[key], location=f"launch_receipt.payload.runtime.python.{key}")
    cuda = _exact_mapping(
        runtime["cuda"],
        _CUDA_KEYS,
        location="launch_receipt.payload.runtime.cuda",
    )
    for key in _CUDA_KEYS:
        if cuda[key] is not None:
            _text(cuda[key], location=f"launch_receipt.payload.runtime.cuda.{key}")
    _text(runtime["physics"], location="launch_receipt.payload.runtime.physics")
    if type(runtime["headless"]) is not bool:
        _fail("launch_receipt.payload.runtime.headless must be boolean.")
    _text(runtime["device"], location="launch_receipt.payload.runtime.device")

    schedule = _validate_schedule(payload["schedule"])
    selector = _exact_mapping(
        payload["selector_protocol"],
        _SELECTOR_KEYS,
        location="launch_receipt.payload.selector_protocol",
    )
    _text(
        selector["contract"],
        location="launch_receipt.payload.selector_protocol.contract",
    )
    if selector["encoding"] != "canonical-json-utf8-v1":
        _fail("selector_protocol.encoding must be 'canonical-json-utf8-v1'.")
    selector_payload = _text(
        selector["payload_utf8"],
        location="launch_receipt.payload.selector_protocol.payload_utf8",
    ).encode("utf-8")
    parse_canonical_training_receipt_json(selector_payload)
    if (
        _exact_int(
            selector["bytes"],
            location="launch_receipt.payload.selector_protocol.bytes",
            minimum=1,
        )
        != len(selector_payload)
        or _hash(
            selector["sha256"],
            location="launch_receipt.payload.selector_protocol.sha256",
        )
        != hashlib.sha256(selector_payload).hexdigest()
    ):
        _fail("selector_protocol byte count or SHA-256 does not match its payload.")

    resume = _exact_mapping(
        payload["resume"],
        _RESUME_KEYS,
        location="launch_receipt.payload.resume",
    )
    if type(resume["is_resume"]) is not bool:
        _fail("launch_receipt.payload.resume.is_resume must be boolean.")
    parent_values = (
        resume["parent_checkpoint_sha256"],
        resume["parent_embedded_receipt_sha256"],
        resume["parent_sidecar_payload_sha256"],
        resume["parent_updates_completed"],
        resume["parent_consumed_transitions"],
    )
    if resume["is_resume"]:
        _hash(parent_values[0], location="launch_receipt.payload.resume.parent_checkpoint_sha256")
        _hash(
            parent_values[1],
            location="launch_receipt.payload.resume.parent_embedded_receipt_sha256",
        )
        _hash(
            parent_values[2],
            location="launch_receipt.payload.resume.parent_sidecar_payload_sha256",
        )
        parent_updates = _exact_int(
            parent_values[3],
            location="launch_receipt.payload.resume.parent_updates_completed",
            minimum=1,
        )
        parent_consumed = _exact_int(
            parent_values[4],
            location="launch_receipt.payload.resume.parent_consumed_transitions",
            minimum=1,
        )
        expected_parent_consumed = (
            parent_updates * schedule["transitions_per_update"]
        )
        if parent_consumed != expected_parent_consumed:
            _fail(
                "Resume parent consumed transitions are not "
                "parent_updates*transitions_per_update."
            )
        if parent_updates >= schedule["max_iterations"]:
            _fail("Resume parent must precede the absolute max_iterations target.")
    elif any(item is not None for item in parent_values):
        _fail("Fresh launch receipt must have null parent resume fields.")

    expected_hash = canonical_training_receipt_sha256(payload)
    if _hash(envelope["payload_sha256"], location="launch_receipt.payload_sha256") != expected_hash:
        _fail("Launch receipt payload SHA-256 is stale or forged.")
    return _clone_native(envelope)


def build_training_launch_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a strict launch receipt from its exact payload fields."""
    detached = _clone_native(dict(payload))
    envelope = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "contract": TRAINING_LAUNCH_RECEIPT_CONTRACT,
        "payload": detached,
        "payload_sha256": canonical_training_receipt_sha256(detached),
    }
    return validate_training_launch_receipt(envelope)


def derive_checkpoint_progress(
    *,
    filename: str,
    iteration: int,
    num_envs: int,
    num_steps_per_env: int,
    configured_target_updates: int,
) -> dict[str, Any]:
    """Derive unambiguous zero-indexed checkpoint progress."""
    iteration = _exact_int(iteration, location="iteration")
    num_envs = _exact_int(num_envs, location="num_envs", minimum=1)
    num_steps_per_env = _exact_int(
        num_steps_per_env,
        location="num_steps_per_env",
        minimum=1,
    )
    target = _exact_int(
        configured_target_updates,
        location="configured_target_updates",
        minimum=1,
    )
    if type(filename) is not str or _CHECKPOINT_NAME_RE.fullmatch(filename) is None:
        _fail("Checkpoint filename must be exactly model_<nonnegative integer>.pt.")
    filename_iteration = int(_CHECKPOINT_NAME_RE.fullmatch(filename).group(1))  # type: ignore[union-attr]
    if filename_iteration != iteration:
        _fail("Checkpoint filename iteration differs from stored iter.")
    updates = iteration + 1
    if updates > target:
        _fail("Checkpoint updates_completed exceeds configured target.")
    per_update = num_envs * num_steps_per_env
    return {
        "filename": filename,
        "iter": iteration,
        "updates_completed": updates,
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "transitions_per_update": per_update,
        "consumed_transitions": updates * per_update,
        "configured_target_updates": target,
    }


def validate_checkpoint_progress(value: Any) -> dict[str, Any]:
    """Validate progress by re-deriving every redundant field."""
    progress = _exact_mapping(value, _PROGRESS_KEYS, location="checkpoint_progress")
    expected = derive_checkpoint_progress(
        filename=progress["filename"],
        iteration=progress["iter"],
        num_envs=progress["num_envs"],
        num_steps_per_env=progress["num_steps_per_env"],
        configured_target_updates=progress["configured_target_updates"],
    )
    if not _strict_equal(progress, expected):
        _fail("Checkpoint progress differs from zero-indexed derived values.")
    return _clone_native(progress)


def _validate_parent(value: Any, *, allow_none: bool) -> dict[str, Any] | None:
    if value is None:
        if allow_none:
            return None
        _fail("Checkpoint parent is required.")
    parent = _exact_mapping(value, _PARENT_KEYS, location="parent_checkpoint")
    name = _text(parent["checkpoint_file_name"], location="parent_checkpoint.checkpoint_file_name")
    if _CHECKPOINT_NAME_RE.fullmatch(name) is None:
        _fail("Parent checkpoint filename is invalid.")
    for key in (
        "checkpoint_sha256",
        "embedded_receipt_sha256",
        "sidecar_payload_sha256",
        "launch_payload_sha256",
    ):
        _hash(parent[key], location=f"parent_checkpoint.{key}")
    _exact_int(parent["checkpoint_bytes"], location="parent_checkpoint.checkpoint_bytes", minimum=1)
    updates = _exact_int(
        parent["updates_completed"],
        location="parent_checkpoint.updates_completed",
        minimum=1,
    )
    consumed = _exact_int(
        parent["consumed_transitions"],
        location="parent_checkpoint.consumed_transitions",
        minimum=1,
    )
    transitions_per_update = _exact_int(
        parent["transitions_per_update"],
        location="parent_checkpoint.transitions_per_update",
        minimum=1,
    )
    if int(_CHECKPOINT_NAME_RE.fullmatch(name).group(1)) + 1 != updates:  # type: ignore[union-attr]
        _fail("Parent filename is inconsistent with parent updates_completed.")
    if consumed != updates * transitions_per_update:
        _fail(
            "Parent consumed_transitions is not "
            "updates_completed*transitions_per_update."
        )
    return _clone_native(parent)


def _validate_launch_progress_binding(
    launch: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> None:
    schedule = launch["payload"]["schedule"]
    expected = (
        schedule["num_envs"],
        schedule["num_steps_per_env"],
        schedule["transitions_per_update"],
        schedule["max_iterations"],
    )
    actual = (
        progress["num_envs"],
        progress["num_steps_per_env"],
        progress["transitions_per_update"],
        progress["configured_target_updates"],
    )
    if not _strict_equal(actual, expected):
        _fail("Checkpoint progress differs from its launch training schedule.")


def build_embedded_checkpoint_receipt(
    *,
    launch_receipt: Mapping[str, Any],
    checkpoint_progress: Mapping[str, Any],
    parent_checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the receipt embedded in a checkpoint.

    The checkpoint file hash is intentionally absent because embedding it would
    create a circular digest.  It is added by :func:`build_checkpoint_sidecar`.
    """
    launch = validate_training_launch_receipt(dict(launch_receipt))
    progress = validate_checkpoint_progress(dict(checkpoint_progress))
    _validate_launch_progress_binding(launch, progress)
    parent = _validate_parent(
        None if parent_checkpoint is None else dict(parent_checkpoint),
        allow_none=True,
    )
    if parent is not None:
        if parent["updates_completed"] >= progress["updates_completed"]:
            _fail("Checkpoint progress must be strictly newer than its parent.")
        if parent["consumed_transitions"] >= progress["consumed_transitions"]:
            _fail("Checkpoint consumed transitions must increase over its parent.")
        if parent["transitions_per_update"] != progress["transitions_per_update"]:
            _fail("Checkpoint transitions_per_update differs from its parent.")
    core = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "contract": CHECKPOINT_EMBEDDED_RECEIPT_CONTRACT,
        "launch_receipt": launch,
        "checkpoint_progress": progress,
        "parent_checkpoint": parent,
    }
    result = {
        **core,
        "embedded_receipt_sha256": canonical_training_receipt_sha256(core),
    }
    return validate_embedded_checkpoint_receipt(result)


def validate_embedded_checkpoint_receipt(
    value: Any,
    *,
    checkpoint_filename: str | None = None,
) -> dict[str, Any]:
    """Validate a checkpoint-embedded receipt without loading executable objects."""
    receipt = _exact_mapping(
        value,
        {
            "schema_version",
            "contract",
            "launch_receipt",
            "checkpoint_progress",
            "parent_checkpoint",
            "embedded_receipt_sha256",
        },
        location="embedded_receipt",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != TRAINING_RECEIPT_SCHEMA_VERSION
        or receipt["contract"] != CHECKPOINT_EMBEDDED_RECEIPT_CONTRACT
    ):
        _fail("Unsupported embedded receipt schema or contract.")
    launch = validate_training_launch_receipt(receipt["launch_receipt"])
    progress = validate_checkpoint_progress(receipt["checkpoint_progress"])
    _validate_launch_progress_binding(launch, progress)
    parent = _validate_parent(receipt["parent_checkpoint"], allow_none=True)
    if checkpoint_filename is not None and progress["filename"] != checkpoint_filename:
        _fail("Embedded checkpoint filename differs from the actual filename.")
    if parent is not None:
        if parent["updates_completed"] >= progress["updates_completed"]:
            _fail("Embedded receipt does not advance beyond its parent.")
        if parent["consumed_transitions"] >= progress["consumed_transitions"]:
            _fail("Embedded receipt consumed transitions do not advance.")
        if parent["transitions_per_update"] != progress["transitions_per_update"]:
            _fail("Embedded receipt transitions_per_update differs from its parent.")
    core = {
        "schema_version": receipt["schema_version"],
        "contract": receipt["contract"],
        "launch_receipt": launch,
        "checkpoint_progress": progress,
        "parent_checkpoint": parent,
    }
    expected = canonical_training_receipt_sha256(core)
    if (
        _hash(
            receipt["embedded_receipt_sha256"],
            location="embedded_receipt.embedded_receipt_sha256",
        )
        != expected
    ):
        _fail("Embedded receipt SHA-256 is stale or forged.")
    return _clone_native(receipt)


def _identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
    }


def _open_regular_file_no_follow(path: Path) -> tuple[int, Path]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        _fail("Formal retained reads require O_NOFOLLOW and O_DIRECTORY.")
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts[1:]
    if not parts:
        _fail("Retained file path must identify a leaf.")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(os.sep, directory_flags)
    file_fd: int | None = None
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                directory_flags | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            _fail(f"Retained path is not a regular file: {absolute}")
        result = file_fd
        file_fd = None
        return result, absolute
    except TrainingReceiptError:
        raise
    except OSError as error:
        raise TrainingReceiptError(
            f"Cannot safely open retained file {absolute}: {error}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def retain_regular_file(path: str | os.PathLike[str]) -> tuple[bytes, dict[str, Any]]:
    """Retain one regular file and reject symlink and TOCTOU identity changes."""
    descriptor, absolute = _open_regular_file_no_follow(Path(path))
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or size != after.st_size:
        _fail(f"Retained file changed while it was read: {absolute}")

    visible_descriptor, visible_absolute = _open_regular_file_no_follow(absolute)
    try:
        visible = os.fstat(visible_descriptor)
    finally:
        os.close(visible_descriptor)
    if visible_absolute != absolute or _identity(visible) != _identity(after):
        _fail(f"Visible retained file identity changed after read: {absolute}")

    payload = b"".join(chunks)
    record = {
        "path": str(absolute),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "identity": _identity(after),
    }
    return payload, record


def _validate_file_identity(value: Any, *, location: str) -> dict[str, int]:
    identity = _exact_mapping(value, _FILE_IDENTITY_KEYS, location=location)
    for key in _FILE_IDENTITY_KEYS:
        _exact_int(identity[key], location=f"{location}.{key}")
    if not stat.S_ISREG(identity["mode"]):
        _fail(f"{location}.mode does not describe a regular file.")
    return identity


def _checkpoint_file_record(retained: Mapping[str, Any], *, filename: str) -> dict[str, Any]:
    record = _exact_mapping(
        dict(retained),
        {"path", "sha256", "bytes", "identity"},
        location="retained_checkpoint",
    )
    if Path(_text(record["path"], location="retained_checkpoint.path")).name != filename:
        _fail("Retained checkpoint basename differs from embedded progress.")
    identity = _validate_file_identity(
        record["identity"],
        location="retained_checkpoint.identity",
    )
    byte_count = _exact_int(
        record["bytes"],
        location="retained_checkpoint.bytes",
        minimum=1,
    )
    if identity["size"] != byte_count:
        _fail("Retained checkpoint identity size differs from byte count.")
    return {
        "file_name": filename,
        "sha256": _hash(record["sha256"], location="retained_checkpoint.sha256"),
        "bytes": byte_count,
    }


def _validate_checkpoint_payload_binding(
    payload: bytes,
    *,
    embedded_receipt: Mapping[str, Any],
) -> None:
    embedded = validate_embedded_checkpoint_receipt(dict(embedded_receipt))
    progress = embedded["checkpoint_progress"]
    try:
        loaded = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise TrainingReceiptError(
            "Checkpoint is not a weights-only-safe Torch checkpoint."
        ) from error
    if type(loaded) is not dict:
        _fail("Formal checkpoint top-level payload must be an exact dictionary.")
    if "training_receipt" not in loaded:
        _fail("Formal checkpoint lacks its embedded training_receipt.")
    if "iter" not in loaded:
        _fail("Formal checkpoint lacks its zero-indexed iter.")
    loaded_receipt = validate_embedded_checkpoint_receipt(
        loaded["training_receipt"],
        checkpoint_filename=progress["filename"],
    )
    if not _strict_equal(loaded_receipt, embedded):
        _fail(
            "Checkpoint-embedded training_receipt differs from the "
            "receipt supplied for its sidecar."
        )
    loaded_iteration = _exact_int(
        loaded["iter"],
        location="checkpoint.iter",
    )
    if loaded_iteration != progress["iter"]:
        _fail("Checkpoint iter differs from its embedded receipt progress.")


def build_checkpoint_sidecar(
    *,
    checkpoint_path: str | os.PathLike[str],
    embedded_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash a completed checkpoint and build its non-circular sidecar."""
    embedded = validate_embedded_checkpoint_receipt(dict(embedded_receipt))
    progress = embedded["checkpoint_progress"]
    payload, retained = retain_regular_file(checkpoint_path)
    checkpoint = _checkpoint_file_record(
        retained,
        filename=progress["filename"],
    )
    _validate_checkpoint_payload_binding(
        payload,
        embedded_receipt=embedded,
    )
    core = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "contract": CHECKPOINT_SIDECAR_CONTRACT,
        "checkpoint": checkpoint,
        "embedded_receipt_sha256": embedded["embedded_receipt_sha256"],
        "launch_payload_sha256": embedded["launch_receipt"]["payload_sha256"],
        "checkpoint_progress": progress,
        "parent_checkpoint": embedded["parent_checkpoint"],
    }
    sidecar = {
        **core,
        "sidecar_payload_sha256": canonical_training_receipt_sha256(core),
    }
    return validate_checkpoint_sidecar(
        sidecar,
        embedded_receipt=embedded,
    )


def validate_checkpoint_sidecar(
    value: Any,
    *,
    embedded_receipt: Mapping[str, Any],
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate a sidecar against embedded metadata and optional live bytes."""
    sidecar = _exact_mapping(
        value,
        {
            "schema_version",
            "contract",
            "checkpoint",
            "embedded_receipt_sha256",
            "launch_payload_sha256",
            "checkpoint_progress",
            "parent_checkpoint",
            "sidecar_payload_sha256",
        },
        location="checkpoint_sidecar",
    )
    if (
        type(sidecar["schema_version"]) is not int
        or sidecar["schema_version"] != TRAINING_RECEIPT_SCHEMA_VERSION
        or sidecar["contract"] != CHECKPOINT_SIDECAR_CONTRACT
    ):
        _fail("Unsupported checkpoint sidecar schema or contract.")
    embedded = validate_embedded_checkpoint_receipt(dict(embedded_receipt))
    checkpoint = _exact_mapping(
        sidecar["checkpoint"],
        _CHECKPOINT_FILE_KEYS,
        location="checkpoint_sidecar.checkpoint",
    )
    file_name = _text(
        checkpoint["file_name"],
        location="checkpoint_sidecar.checkpoint.file_name",
    )
    if _CHECKPOINT_NAME_RE.fullmatch(file_name) is None:
        _fail("Sidecar checkpoint filename is invalid.")
    _hash(checkpoint["sha256"], location="checkpoint_sidecar.checkpoint.sha256")
    bytes_count = _exact_int(
        checkpoint["bytes"],
        location="checkpoint_sidecar.checkpoint.bytes",
        minimum=1,
    )
    progress = validate_checkpoint_progress(sidecar["checkpoint_progress"])
    parent = _validate_parent(sidecar["parent_checkpoint"], allow_none=True)
    expected_bindings = (
        file_name,
        sidecar["embedded_receipt_sha256"],
        sidecar["launch_payload_sha256"],
        progress,
        parent,
    )
    actual_bindings = (
        embedded["checkpoint_progress"]["filename"],
        embedded["embedded_receipt_sha256"],
        embedded["launch_receipt"]["payload_sha256"],
        embedded["checkpoint_progress"],
        embedded["parent_checkpoint"],
    )
    if not _strict_equal(expected_bindings, actual_bindings):
        _fail("Sidecar bindings differ from the embedded checkpoint receipt.")
    core = {
        "schema_version": sidecar["schema_version"],
        "contract": sidecar["contract"],
        "checkpoint": checkpoint,
        "embedded_receipt_sha256": sidecar["embedded_receipt_sha256"],
        "launch_payload_sha256": sidecar["launch_payload_sha256"],
        "checkpoint_progress": progress,
        "parent_checkpoint": parent,
    }
    expected_hash = canonical_training_receipt_sha256(core)
    if (
        _hash(
            sidecar["sidecar_payload_sha256"],
            location="checkpoint_sidecar.sidecar_payload_sha256",
        )
        != expected_hash
    ):
        _fail("Checkpoint sidecar payload SHA-256 is stale or forged.")
    if checkpoint_path is not None:
        payload, retained = retain_regular_file(checkpoint_path)
        actual_checkpoint = _checkpoint_file_record(
            retained,
            filename=progress["filename"],
        )
        if not _strict_equal(checkpoint, actual_checkpoint):
            _fail("Checkpoint bytes differ from the sidecar.")
        _validate_checkpoint_payload_binding(
            payload,
            embedded_receipt=embedded,
        )
    return _clone_native(sidecar)


def checkpoint_parent_record(
    *,
    embedded_receipt: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact parent record required by the next checkpoint."""
    embedded = validate_embedded_checkpoint_receipt(dict(embedded_receipt))
    validated_sidecar = validate_checkpoint_sidecar(
        dict(sidecar),
        embedded_receipt=embedded,
    )
    progress = embedded["checkpoint_progress"]
    checkpoint = validated_sidecar["checkpoint"]
    return {
        "checkpoint_file_name": checkpoint["file_name"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_bytes": checkpoint["bytes"],
        "embedded_receipt_sha256": embedded["embedded_receipt_sha256"],
        "sidecar_payload_sha256": validated_sidecar["sidecar_payload_sha256"],
        "launch_payload_sha256": embedded["launch_receipt"]["payload_sha256"],
        "updates_completed": progress["updates_completed"],
        "transitions_per_update": progress["transitions_per_update"],
        "consumed_transitions": progress["consumed_transitions"],
    }


def validate_checkpoint_receipt_chain(
    entries: Sequence[Mapping[str, Any]],
    *,
    external_parent: Mapping[str, Any] | None = None,
    resume_parent_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an ordered checkpoint chain and return its unique latest head.

    Each entry must contain exactly ``embedded_receipt`` and ``sidecar``.  A
    resumed segment must receive the exact ``latest_head`` returned by
    validation of its parent chain; a boolean bypass is intentionally absent.
    Supplying ``resume_parent_checkpoint_sha256`` additionally proves that the
    requested resume parent is the latest validated head, rejecting rollback.
    """
    if type(entries) not in {list, tuple} or not entries:
        _fail("Checkpoint receipt chain must be a non-empty list or tuple.")
    validated_external_parent = (
        None
        if external_parent is None
        else _validate_parent(dict(external_parent), allow_none=False)
    )
    normalized: list[dict[str, Any]] = []
    previous_parent: dict[str, Any] | None = None
    previous_launch_hash: str | None = None
    seen_checkpoint_hashes: set[str] = set()
    seen_updates: set[int] = set()
    for index, raw_entry in enumerate(entries):
        entry = _exact_mapping(
            dict(raw_entry) if isinstance(raw_entry, Mapping) else raw_entry,
            {"embedded_receipt", "sidecar"},
            location=f"chain[{index}]",
        )
        embedded = validate_embedded_checkpoint_receipt(entry["embedded_receipt"])
        sidecar = validate_checkpoint_sidecar(
            entry["sidecar"],
            embedded_receipt=embedded,
        )
        parent = embedded["parent_checkpoint"]
        launch = embedded["launch_receipt"]
        launch_hash = launch["payload_sha256"]
        resume = launch["payload"]["resume"]
        if index == 0:
            if parent is None and validated_external_parent is not None:
                _fail("A genesis chain cannot receive an external parent.")
            if parent is not None and validated_external_parent is None:
                _fail(
                    "A resumed chain segment requires its validated external "
                    "parent head."
                )
            if parent is not None and not _strict_equal(
                parent,
                validated_external_parent,
            ):
                _fail(
                    "Resumed chain parent differs from the validated external "
                    "parent head."
                )
            if parent is None and resume["is_resume"]:
                _fail("A genesis checkpoint cannot claim a resume parent.")
            if parent is not None:
                expected_resume = (
                    parent["checkpoint_sha256"],
                    parent["embedded_receipt_sha256"],
                    parent["sidecar_payload_sha256"],
                    parent["updates_completed"],
                    parent["consumed_transitions"],
                )
                actual_resume = (
                    resume["parent_checkpoint_sha256"],
                    resume["parent_embedded_receipt_sha256"],
                    resume["parent_sidecar_payload_sha256"],
                    resume["parent_updates_completed"],
                    resume["parent_consumed_transitions"],
                )
                if resume["is_resume"] is not True or not _strict_equal(
                    actual_resume,
                    expected_resume,
                ):
                    _fail(
                        "External-parent chain segment lacks an exact launch "
                        "resume binding."
                    )
        elif not _strict_equal(parent, previous_parent):
            _fail(f"Checkpoint chain parent mismatch at index {index}.")
        elif launch_hash != previous_launch_hash:
            assert previous_parent is not None
            expected_resume = (
                previous_parent["checkpoint_sha256"],
                previous_parent["embedded_receipt_sha256"],
                previous_parent["sidecar_payload_sha256"],
                previous_parent["updates_completed"],
                previous_parent["consumed_transitions"],
            )
            actual_resume = (
                resume["parent_checkpoint_sha256"],
                resume["parent_embedded_receipt_sha256"],
                resume["parent_sidecar_payload_sha256"],
                resume["parent_updates_completed"],
                resume["parent_consumed_transitions"],
            )
            if resume["is_resume"] is not True or not _strict_equal(
                actual_resume,
                expected_resume,
            ):
                _fail(
                    "A changed launch receipt does not bind the previous "
                    "checkpoint chain head."
                )
        current_parent = checkpoint_parent_record(
            embedded_receipt=embedded,
            sidecar=sidecar,
        )
        checkpoint_hash = current_parent["checkpoint_sha256"]
        updates = current_parent["updates_completed"]
        if checkpoint_hash in seen_checkpoint_hashes:
            _fail("Checkpoint chain contains a duplicate checkpoint hash.")
        if updates in seen_updates:
            _fail("Checkpoint chain contains duplicate updates_completed.")
        if previous_parent is not None:
            if updates <= previous_parent["updates_completed"]:
                _fail("Checkpoint chain updates_completed did not increase.")
            if (
                current_parent["consumed_transitions"]
                <= previous_parent["consumed_transitions"]
            ):
                _fail("Checkpoint chain consumed_transitions did not increase.")
        seen_checkpoint_hashes.add(checkpoint_hash)
        seen_updates.add(updates)
        previous_parent = current_parent
        previous_launch_hash = launch_hash
        normalized.append(
            {"embedded_receipt": embedded, "sidecar": sidecar}
        )
    assert previous_parent is not None
    if resume_parent_checkpoint_sha256 is not None:
        requested = _hash(
            resume_parent_checkpoint_sha256,
            location="resume_parent_checkpoint_sha256",
        )
        if requested != previous_parent["checkpoint_sha256"]:
            _fail(
                "Requested resume parent is not the latest validated "
                "checkpoint chain head (rollback rejected)."
            )
    return {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "contract": "rsl_rl_formal_checkpoint_chain_validation_v1",
        "entry_count": len(normalized),
        "latest_head": previous_parent,
        "entries": normalized,
    }


__all__ = [
    "CHECKPOINT_EMBEDDED_RECEIPT_CONTRACT",
    "CHECKPOINT_SIDECAR_CONTRACT",
    "TRAINING_LAUNCH_RECEIPT_CONTRACT",
    "TRAINING_RECEIPT_SCHEMA_VERSION",
    "TrainingReceiptError",
    "build_checkpoint_sidecar",
    "build_embedded_checkpoint_receipt",
    "build_training_launch_receipt",
    "canonical_training_receipt_json_bytes",
    "canonical_training_receipt_sha256",
    "checkpoint_parent_record",
    "derive_checkpoint_progress",
    "parse_canonical_training_receipt_json",
    "retain_regular_file",
    "validate_checkpoint_progress",
    "validate_checkpoint_receipt_chain",
    "validate_checkpoint_sidecar",
    "validate_embedded_checkpoint_receipt",
    "validate_training_launch_receipt",
]
