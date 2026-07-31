# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed filesystem transactions for formal training checkpoints."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import re
import secrets
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from .training_receipt import (
    build_checkpoint_sidecar,
    canonical_training_receipt_json_bytes,
    canonical_training_receipt_sha256,
    checkpoint_parent_record,
    parse_canonical_training_receipt_json,
    retain_regular_file,
    validate_checkpoint_receipt_chain,
    validate_checkpoint_sidecar,
    validate_embedded_checkpoint_receipt,
    validate_training_launch_receipt,
)


FORMAL_CHECKPOINT_HEAD_CONTRACT = "rsl_rl_formal_checkpoint_head_v1"
FORMAL_CHECKPOINT_HEAD_SCHEMA_VERSION = 1
FORMAL_ANCESTOR_CHAIN_PROOF_CONTRACT = (
    "rsl_rl_formal_ancestor_chain_proof_v1"
)
FORMAL_ANCESTOR_CHAIN_PROOF_SCHEMA_VERSION = 1
FORMAL_LAUNCH_RECEIPT_NAME = "launch_receipt.json"

_CHECKPOINT_NAME_RE = re.compile(r"^model_(0|[1-9][0-9]*)\.pt$")
_CHECKPOINT_LIKE_RE = re.compile(r"^model_.*\.pt$")
_SIDECAR_NAME_RE = re.compile(r"^model_(0|[1-9][0-9]*)\.json$")
_SIDECAR_LIKE_RE = re.compile(r"^model_.*\.json$")
_HEAD_NAME_RE = re.compile(r"^head_([0-9]{20})\.json$")
_HEAD_TEMP_NAME_RE = re.compile(
    r"^\.head_[0-9]{20}\.json\.[0-9]+\.[0-9a-f]{16}\.tmp$"
)
_ANCESTOR_CHAIN_PROOF_NAME = "ancestor_chain.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEAD_KEYS = {
    "schema_version",
    "contract",
    "checkpoint_parent_record",
    "sidecar",
    "ancestor_chain_proof",
    "previous_head_payload_sha256",
    "head_payload_sha256",
}
_HEAD_SIDECAR_KEYS = {"file_name", "sha256", "bytes"}
_ANCESTOR_CHAIN_PROOF_KEYS = {
    "schema_version",
    "contract",
    "entry_count",
    "entries",
    "latest_parent",
    "proof_payload_sha256",
}


class FormalTrainingIOError(RuntimeError):
    """Raised when a formal checkpoint transaction is not auditable."""


def _fail(message: str) -> None:
    raise FormalTrainingIOError(message)


def _fcntl_module() -> Any:
    try:
        return importlib.import_module("fcntl")
    except ImportError as error:  # pragma: no cover - exercised off POSIX.
        raise FormalTrainingIOError(
            "Formal training requires POSIX flock support."
        ) from error


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], item)
            for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _exact_mapping(
    value: Any,
    expected_keys: set[str],
    *,
    location: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected_keys:
        _fail(f"{location} does not have the exact formal keyset.")
    return value


def _exact_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{location} must be an exact integer >= {minimum}.")
    return value


def _hash(value: Any, *, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{location} must be a lowercase SHA-256 digest.")
    return value


def _canonical_absolute(path: str | os.PathLike[str], *, location: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        _fail(f"{location} must be absolute.")
    normalized = Path(os.path.abspath(os.fspath(candidate)))
    if candidate != normalized:
        _fail(f"{location} must be lexically canonical.")
    return normalized


def _assert_directory_no_symlinks(
    path: Path,
    *,
    location: str,
    create: bool,
) -> None:
    path = _canonical_absolute(path, location=location)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if not create:
                _fail(f"{location} does not exist: {path}")
            current.mkdir(mode=0o700)
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"{location} contains a symlink component: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{location} contains a non-directory component: {current}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_existing_run_lock(run_dir: Path) -> int:
    """Lock an existing formal run without creating or changing any file."""
    flock_module = _fcntl_module()
    _assert_directory_no_symlinks(
        run_dir,
        location="formal parent run directory",
        create=False,
    )
    lock_path = run_dir / ".formal_training.lock"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as error:
        raise FormalTrainingIOError(
            f"Cannot safely open existing formal training lock: {lock_path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("Formal parent training lock is not a regular file.")
        _assert_lock_identity(
            lock_path,
            metadata,
            location="formal parent training lock",
        )
        try:
            flock_module.flock(
                descriptor,
                flock_module.LOCK_EX | flock_module.LOCK_NB,
            )
        except BlockingIOError as error:
            raise FormalTrainingIOError(
                f"Formal parent run is still locked: {run_dir}"
            ) from error
        _assert_lock_identity(
            lock_path,
            metadata,
            location="formal parent training lock",
        )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_lock_identity(
    lock_path: Path,
    metadata: os.stat_result,
    *,
    location: str,
) -> None:
    try:
        visible = os.lstat(lock_path)
    except OSError as error:
        raise FormalTrainingIOError(
            f"{location} disappeared during lock acquisition."
        ) from error
    if (
        visible.st_dev != metadata.st_dev
        or visible.st_ino != metadata.st_ino
        or not stat.S_ISREG(visible.st_mode)
    ):
        _fail(f"{location} visible identity changed.")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )


def _publish_new_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes once using a same-directory durable hard-link commit."""
    _assert_directory_no_symlinks(
        path.parent,
        location=f"parent of {path.name}",
        create=True,
    )
    if path.exists() or path.is_symlink():
        _fail(f"Refusing to overwrite immutable formal artifact: {path}")
    temporary = _temporary_sibling(path)
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FormalTrainingIOError(
                f"Refusing to overwrite immutable formal artifact: {path}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _assert_retained_checkpoint_unchanged(
    path: Path,
    *,
    expected_payload: bytes,
    expected_record: Mapping[str, Any],
    location: str,
) -> tuple[bytes, dict[str, Any]]:
    payload, record = retain_regular_file(path)
    if payload != expected_payload or not _strict_equal(
        record,
        dict(expected_record),
    ):
        _fail(f"{location} checkpoint path or bytes changed.")
    return payload, record


def _publish_new_torch_checkpoint(
    path: Path,
    value: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Serialize and publish a Torch checkpoint without replacing a path."""
    _assert_directory_no_symlinks(
        path.parent,
        location="formal run directory",
        create=False,
    )
    if path.exists() or path.is_symlink():
        _fail(f"Refusing to overwrite immutable formal checkpoint: {path}")
    temporary = _temporary_sibling(path)
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            torch.save(dict(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
        retained_payload, _retained_record = retain_regular_file(temporary)
        _safe_checkpoint_receipt(
            retained_payload,
            checkpoint_filename=path.name,
        )
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FormalTrainingIOError(
                f"Refusing to overwrite immutable formal checkpoint: {path}"
            ) from error
        linked_payload, linked_record = retain_regular_file(temporary)
        visible_payload, visible_record = retain_regular_file(path)
        if (
            linked_payload != retained_payload
            or visible_payload != retained_payload
            or not _strict_equal(
                linked_record["identity"],
                visible_record["identity"],
            )
        ):
            _fail(
                "Formal checkpoint visible path differs from its linked "
                "serialization inode."
            )
        temporary.unlink()
        _fsync_directory(path.parent)
        published_payload, published_record = retain_regular_file(path)
        if (
            published_payload != retained_payload
            or published_record["identity"]["device"]
            != linked_record["identity"]["device"]
            or published_record["identity"]["inode"]
            != linked_record["identity"]["inode"]
        ):
            _fail("Published formal checkpoint differs from serialized bytes.")
        return published_payload, published_record
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload, _record = retain_regular_file(path)
    value = parse_canonical_training_receipt_json(payload)
    if type(value) is not dict:
        _fail(f"Formal JSON root must be an object: {path}")
    return value, payload


def _safe_checkpoint_receipt(
    payload: bytes,
    *,
    checkpoint_filename: str,
    map_location: str | None = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        loaded = torch.load(
            io.BytesIO(payload),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise FormalTrainingIOError(
            "Formal checkpoint is not weights-only safe."
        ) from error
    if type(loaded) is not dict:
        _fail("Formal checkpoint root must be an exact dictionary.")
    if "training_receipt" not in loaded or "iter" not in loaded:
        _fail("Formal checkpoint lacks receipt or iter.")
    embedded = validate_embedded_checkpoint_receipt(
        loaded["training_receipt"],
        checkpoint_filename=checkpoint_filename,
    )
    if (
        type(loaded["iter"]) is not int
        or loaded["iter"] != embedded["checkpoint_progress"]["iter"]
    ):
        _fail("Formal checkpoint iter differs from its embedded receipt.")
    return loaded, embedded


def _sidecar_name(checkpoint_name: str) -> str:
    return f"{Path(checkpoint_name).stem}.json"


def _head_name(updates_completed: int) -> str:
    return f"head_{updates_completed:020d}.json"


def _validate_generation_compatibility(
    current_launch: Mapping[str, Any],
    parent_launch: Mapping[str, Any],
    parent_record: Mapping[str, Any],
) -> None:
    """Validate every immutable field at one changed-launch boundary."""
    current = validate_training_launch_receipt(dict(current_launch))
    parent = validate_training_launch_receipt(dict(parent_launch))
    current_payload = current["payload"]
    parent_payload = parent["payload"]
    for field in (
        "task",
        "seed",
        "git",
        "runtime",
        "schedule",
        "selector_protocol",
    ):
        if not _strict_equal(
            current_payload[field],
            parent_payload[field],
        ):
            _fail(f"Formal resume changed immutable launch field {field}.")
    if (
        current_payload["configs"]["resume_compatibility_sha256"]
        != parent_payload["configs"]["resume_compatibility_sha256"]
    ):
        _fail("Formal resume configuration compatibility hash differs.")
    resume = current_payload["resume"]
    expected = (
        parent_record["checkpoint_sha256"],
        parent_record["embedded_receipt_sha256"],
        parent_record["sidecar_payload_sha256"],
        parent_record["updates_completed"],
        parent_record["consumed_transitions"],
    )
    actual = (
        resume["parent_checkpoint_sha256"],
        resume["parent_embedded_receipt_sha256"],
        resume["parent_sidecar_payload_sha256"],
        resume["parent_updates_completed"],
        resume["parent_consumed_transitions"],
    )
    if resume["is_resume"] is not True or not _strict_equal(actual, expected):
        _fail("Formal launch does not bind the exact resume parent head.")


def _validate_chain_generation_compatibility(
    entries: list[Mapping[str, Any]],
) -> None:
    for index in range(1, len(entries)):
        previous = entries[index - 1]
        current = entries[index]
        previous_embedded = previous["embedded_receipt"]
        current_embedded = current["embedded_receipt"]
        if (
            previous_embedded["launch_receipt"]["payload_sha256"]
            == current_embedded["launch_receipt"]["payload_sha256"]
        ):
            continue
        previous_parent = checkpoint_parent_record(
            embedded_receipt=previous_embedded,
            sidecar=previous["sidecar"],
        )
        _validate_generation_compatibility(
            current_embedded["launch_receipt"],
            previous_embedded["launch_receipt"],
            previous_parent,
        )


def _validate_full_formal_chain(
    entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    chain = validate_checkpoint_receipt_chain(entries)
    _validate_chain_generation_compatibility(chain["entries"])
    return chain


def _build_ancestor_chain_proof(
    entries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    chain = _validate_full_formal_chain(entries)
    core = {
        "schema_version": FORMAL_ANCESTOR_CHAIN_PROOF_SCHEMA_VERSION,
        "contract": FORMAL_ANCESTOR_CHAIN_PROOF_CONTRACT,
        "entry_count": chain["entry_count"],
        "entries": chain["entries"],
        "latest_parent": chain["latest_head"],
    }
    return _validate_ancestor_chain_proof(
        {
            **core,
            "proof_payload_sha256": canonical_training_receipt_sha256(core),
        }
    )


def _validate_ancestor_chain_proof(value: Any) -> dict[str, Any]:
    proof = _exact_mapping(
        value,
        _ANCESTOR_CHAIN_PROOF_KEYS,
        location="ancestor_chain_proof",
    )
    if (
        type(proof["schema_version"]) is not int
        or proof["schema_version"]
        != FORMAL_ANCESTOR_CHAIN_PROOF_SCHEMA_VERSION
        or proof["contract"] != FORMAL_ANCESTOR_CHAIN_PROOF_CONTRACT
    ):
        _fail("Unsupported ancestor chain proof schema or contract.")
    entry_count = _exact_int(
        proof["entry_count"],
        location="ancestor_chain_proof.entry_count",
        minimum=1,
    )
    if type(proof["entries"]) is not list:
        _fail("ancestor_chain_proof.entries must be an exact list.")
    chain = _validate_full_formal_chain(proof["entries"])
    if entry_count != chain["entry_count"]:
        _fail("Ancestor chain proof entry_count differs from its entries.")
    if not _strict_equal(proof["latest_parent"], chain["latest_head"]):
        _fail("Ancestor chain proof latest_parent is stale or forked.")
    core = {
        "schema_version": proof["schema_version"],
        "contract": proof["contract"],
        "entry_count": entry_count,
        "entries": chain["entries"],
        "latest_parent": chain["latest_head"],
    }
    if (
        _hash(
            proof["proof_payload_sha256"],
            location="ancestor_chain_proof.proof_payload_sha256",
        )
        != canonical_training_receipt_sha256(core)
    ):
        _fail("Ancestor chain proof payload SHA-256 is stale or forged.")
    return parse_canonical_training_receipt_json(
        canonical_training_receipt_json_bytes(
            {
                **core,
                "proof_payload_sha256": proof["proof_payload_sha256"],
            }
        )
    )


def _ancestor_proof_file_record(payload: bytes) -> dict[str, Any]:
    return {
        "file_name": _ANCESTOR_CHAIN_PROOF_NAME,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _build_head(
    *,
    current_parent: Mapping[str, Any],
    sidecar_name: str,
    sidecar_payload: bytes,
    ancestor_chain_proof: Mapping[str, Any] | None,
    previous_head_payload_sha256: str | None,
) -> dict[str, Any]:
    if previous_head_payload_sha256 is not None:
        _hash(
            previous_head_payload_sha256,
            location="previous_head_payload_sha256",
        )
    core = {
        "schema_version": FORMAL_CHECKPOINT_HEAD_SCHEMA_VERSION,
        "contract": FORMAL_CHECKPOINT_HEAD_CONTRACT,
        "checkpoint_parent_record": dict(current_parent),
        "sidecar": {
            "file_name": sidecar_name,
            "sha256": hashlib.sha256(sidecar_payload).hexdigest(),
            "bytes": len(sidecar_payload),
        },
        "ancestor_chain_proof": (
            None
            if ancestor_chain_proof is None
            else dict(ancestor_chain_proof)
        ),
        "previous_head_payload_sha256": previous_head_payload_sha256,
    }
    return {
        **core,
        "head_payload_sha256": canonical_training_receipt_sha256(core),
    }


def _validate_head(value: Any, *, expected_name: str | None = None) -> dict[str, Any]:
    head = _exact_mapping(value, _HEAD_KEYS, location="checkpoint_head")
    if (
        type(head["schema_version"]) is not int
        or head["schema_version"] != FORMAL_CHECKPOINT_HEAD_SCHEMA_VERSION
        or head["contract"] != FORMAL_CHECKPOINT_HEAD_CONTRACT
    ):
        _fail("Unsupported formal checkpoint head schema or contract.")
    parent = head["checkpoint_parent_record"]
    if type(parent) is not dict:
        _fail("checkpoint_head.checkpoint_parent_record must be an object.")
    sidecar = _exact_mapping(
        head["sidecar"],
        _HEAD_SIDECAR_KEYS,
        location="checkpoint_head.sidecar",
    )
    if (
        type(sidecar["file_name"]) is not str
        or not sidecar["file_name"].endswith(".json")
        or Path(sidecar["file_name"]).name != sidecar["file_name"]
    ):
        _fail("checkpoint_head.sidecar.file_name is invalid.")
    _hash(sidecar["sha256"], location="checkpoint_head.sidecar.sha256")
    _exact_int(
        sidecar["bytes"],
        location="checkpoint_head.sidecar.bytes",
        minimum=1,
    )
    ancestor = head["ancestor_chain_proof"]
    if ancestor is not None:
        ancestor = _exact_mapping(
            ancestor,
            _HEAD_SIDECAR_KEYS,
            location="checkpoint_head.ancestor_chain_proof",
        )
        if ancestor["file_name"] != _ANCESTOR_CHAIN_PROOF_NAME:
            _fail("Formal head ancestor proof filename is invalid.")
        _hash(
            ancestor["sha256"],
            location="checkpoint_head.ancestor_chain_proof.sha256",
        )
        _exact_int(
            ancestor["bytes"],
            location="checkpoint_head.ancestor_chain_proof.bytes",
            minimum=1,
        )
    previous = head["previous_head_payload_sha256"]
    if previous is not None:
        _hash(previous, location="checkpoint_head.previous_head_payload_sha256")
    core = {
        "schema_version": head["schema_version"],
        "contract": head["contract"],
        "checkpoint_parent_record": parent,
        "sidecar": sidecar,
        "ancestor_chain_proof": ancestor,
        "previous_head_payload_sha256": previous,
    }
    if (
        _hash(
            head["head_payload_sha256"],
            location="checkpoint_head.head_payload_sha256",
        )
        != canonical_training_receipt_sha256(core)
    ):
        _fail("Formal checkpoint head hash is stale or forged.")
    updates = _exact_int(
        parent.get("updates_completed"),
        location="checkpoint_head.updates_completed",
        minimum=1,
    )
    if expected_name is not None and expected_name != _head_name(updates):
        _fail("Formal checkpoint head filename differs from updates_completed.")
    return parse_canonical_training_receipt_json(
        canonical_training_receipt_json_bytes(head)
    )


class FormalTrainingIO:
    """Own one formal run lock and publish/validate immutable checkpoint commits."""

    def __init__(
        self,
        *,
        run_dir: str | os.PathLike[str],
        launch_receipt: Mapping[str, Any],
    ) -> None:
        self.run_dir = _canonical_absolute(run_dir, location="formal run_dir")
        self.launch_receipt = validate_training_launch_receipt(
            dict(launch_receipt)
        )
        _assert_directory_no_symlinks(
            self.run_dir,
            location="formal run_dir",
            create=True,
        )
        self.receipts_dir = self.run_dir / "checkpoint_receipts"
        self.heads_dir = self.receipts_dir / "heads"
        _assert_directory_no_symlinks(
            self.receipts_dir,
            location="checkpoint receipt directory",
            create=True,
        )
        _assert_directory_no_symlinks(
            self.heads_dir,
            location="checkpoint head directory",
            create=True,
        )
        self._lock_descriptor: int | None = None
        self._resume_parent: dict[str, Any] | None = None
        self._ancestor_chain_proof: dict[str, Any] | None = None
        self._pending_recovery: dict[str, Any] | None = None
        self._acquire_lock()
        try:
            self._bind_launch_receipt()
        except Exception:
            self.close()
            raise

    @property
    def lock_held(self) -> bool:
        return self._lock_descriptor is not None

    @property
    def resume_parent(self) -> dict[str, Any] | None:
        return self._resume_parent

    @property
    def ancestor_chain_proof(self) -> dict[str, Any] | None:
        return self._ancestor_chain_proof

    def _acquire_lock(self) -> None:
        flock_module = _fcntl_module()
        lock_path = self.run_dir / ".formal_training.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise FormalTrainingIOError(
                f"Cannot safely open formal training lock: {lock_path}"
            ) from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            _fail("Formal training lock is not a regular file.")
        try:
            _assert_lock_identity(
                lock_path,
                metadata,
                location="formal training lock",
            )
        except Exception:
            os.close(descriptor)
            raise
        try:
            flock_module.flock(
                descriptor,
                flock_module.LOCK_EX | flock_module.LOCK_NB,
            )
        except BlockingIOError as error:
            os.close(descriptor)
            raise FormalTrainingIOError(
                f"Formal training run is already locked: {self.run_dir}"
            ) from error
        try:
            _assert_lock_identity(
                lock_path,
                metadata,
                location="formal training lock",
            )
        except Exception:
            flock_module.flock(descriptor, flock_module.LOCK_UN)
            os.close(descriptor)
            raise
        self._lock_descriptor = descriptor

    def _read_bound_launch_receipt(self) -> dict[str, Any]:
        launch_path = self.run_dir / FORMAL_LAUNCH_RECEIPT_NAME
        document, payload = _read_canonical_json(launch_path)
        validated = validate_training_launch_receipt(document)
        if payload != canonical_training_receipt_json_bytes(validated):
            _fail("Formal launch receipt artifact is not canonical.")
        return validated

    def _bind_launch_receipt(self) -> None:
        """Atomically bind one immutable launch receipt to this run directory."""
        if not self.lock_held:
            _fail("Formal training lock is not held.")
        launch_path = self.run_dir / FORMAL_LAUNCH_RECEIPT_NAME
        expected_payload = canonical_training_receipt_json_bytes(
            self.launch_receipt
        )
        if launch_path.exists() or launch_path.is_symlink():
            existing = self._read_bound_launch_receipt()
            retained_payload, _record = retain_regular_file(launch_path)
            if (
                retained_payload != expected_payload
                or not _strict_equal(existing, self.launch_receipt)
            ):
                _fail(
                    "Formal run directory is already bound to a different "
                    "launch receipt."
                )
        else:
            _publish_new_bytes(launch_path, expected_payload)
        final = self._read_bound_launch_receipt()
        final_payload, _record = retain_regular_file(launch_path)
        if (
            final_payload != expected_payload
            or not _strict_equal(final, self.launch_receipt)
        ):
            _fail("Formal launch receipt binding changed during publication.")

    def close(self) -> None:
        self._pending_recovery = None
        descriptor = self._lock_descriptor
        self._lock_descriptor = None
        if descriptor is not None:
            flock_module = _fcntl_module()
            try:
                flock_module.flock(descriptor, flock_module.LOCK_UN)
            finally:
                os.close(descriptor)

    def __enter__(self) -> "FormalTrainingIO":
        if not self.lock_held:
            _fail("Formal training lock has already been released.")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def checkpoint_path(self, iteration: int) -> Path:
        iteration = _exact_int(iteration, location="checkpoint iteration")
        return self.run_dir / f"model_{iteration}.pt"

    def _assert_checkpoint_target(self, path: Path, *, iteration: int) -> Path:
        expected = self.checkpoint_path(iteration)
        candidate = _canonical_absolute(path, location="checkpoint path")
        if candidate != expected:
            _fail(
                f"Formal checkpoint path must be exactly {expected}, got {candidate}."
            )
        return candidate

    def _read_ancestor_chain_proof(
        self,
        *,
        expected_record: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        proof_path = self.receipts_dir / _ANCESTOR_CHAIN_PROOF_NAME
        if not proof_path.exists() and not proof_path.is_symlink():
            if expected_record is not None:
                _fail("Formal ancestor chain proof file is missing.")
            return None
        proof_document, proof_payload = _read_canonical_json(proof_path)
        proof = _validate_ancestor_chain_proof(proof_document)
        actual_record = _ancestor_proof_file_record(proof_payload)
        if expected_record is not None and not _strict_equal(
            actual_record,
            dict(expected_record),
        ):
            _fail("Formal ancestor chain proof differs from its head binding.")
        return proof, actual_record

    def _publish_ancestor_chain_proof(
        self,
        proof: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validated = _validate_ancestor_chain_proof(dict(proof))
        payload = canonical_training_receipt_json_bytes(validated)
        expected_record = _ancestor_proof_file_record(payload)
        existing = self._read_ancestor_chain_proof()
        if existing is None:
            _publish_new_bytes(
                self.receipts_dir / _ANCESTOR_CHAIN_PROOF_NAME,
                payload,
            )
        else:
            existing_proof, existing_record = existing
            if (
                not _strict_equal(existing_proof, validated)
                or not _strict_equal(existing_record, expected_record)
            ):
                _fail(
                    "Existing immutable ancestor chain proof differs from "
                    "the validated resume lineage."
                )
        final = self._read_ancestor_chain_proof(
            expected_record=expected_record,
        )
        assert final is not None
        return final

    def _head_paths(self) -> list[Path]:
        _assert_directory_no_symlinks(
            self.heads_dir,
            location="checkpoint head directory",
            create=False,
        )
        paths: list[Path] = []
        for entry in os.scandir(self.heads_dir):
            if entry.is_symlink():
                _fail(f"Symlink is forbidden in checkpoint heads: {entry.path}")
            if not entry.is_file(follow_symlinks=False):
                _fail(f"Non-file entry is forbidden in checkpoint heads: {entry.path}")
            if _HEAD_TEMP_NAME_RE.fullmatch(entry.name) is not None:
                continue
            match = _HEAD_NAME_RE.fullmatch(entry.name)
            if match is None:
                _fail(f"Unexpected file in checkpoint heads: {entry.name}")
            paths.append(Path(entry.path))
        paths.sort(key=lambda item: item.name)
        return paths

    def _load_committed_chain(self) -> dict[str, Any] | None:
        bound_launch = self._read_bound_launch_receipt()
        if self.launch_receipt:
            if not _strict_equal(bound_launch, self.launch_receipt):
                _fail(
                    "Formal run launch artifact differs from the configured "
                    "launch receipt."
                )
        else:
            self.launch_receipt = bound_launch
        head_paths = self._head_paths()
        if not head_paths:
            return None
        entries: list[dict[str, Any]] = []
        previous_head_hash: str | None = None
        latest_head: dict[str, Any] | None = None
        unset_ancestor_record = object()
        ancestor_record: (
            dict[str, Any] | None | object
        ) = unset_ancestor_record
        for head_path in head_paths:
            head_document, _head_payload = _read_canonical_json(head_path)
            head = _validate_head(
                head_document,
                expected_name=head_path.name,
            )
            if head["previous_head_payload_sha256"] != previous_head_hash:
                _fail("Formal checkpoint head append chain is broken.")
            if ancestor_record is unset_ancestor_record:
                ancestor_record = head["ancestor_chain_proof"]
            elif not _strict_equal(
                head["ancestor_chain_proof"],
                ancestor_record,
            ):
                _fail(
                    "Formal checkpoint heads disagree on ancestor chain proof."
                )
            parent_record = head["checkpoint_parent_record"]
            checkpoint_name = parent_record.get("checkpoint_file_name")
            if (
                type(checkpoint_name) is not str
                or _CHECKPOINT_NAME_RE.fullmatch(checkpoint_name) is None
            ):
                _fail("Formal head checkpoint filename is invalid.")
            if head["sidecar"]["file_name"] != _sidecar_name(
                checkpoint_name
            ):
                _fail("Formal head sidecar filename differs from checkpoint.")
            checkpoint_path = self.run_dir / checkpoint_name
            checkpoint_payload, _checkpoint_record = retain_regular_file(
                checkpoint_path
            )
            _safe_loaded, embedded = _safe_checkpoint_receipt(
                checkpoint_payload,
                checkpoint_filename=checkpoint_name,
            )
            if not _strict_equal(
                embedded["launch_receipt"],
                bound_launch,
            ):
                _fail(
                    "Formal checkpoint launch differs from the immutable run "
                    "launch artifact."
                )
            sidecar_path = self.receipts_dir / head["sidecar"]["file_name"]
            sidecar, sidecar_payload = _read_canonical_json(sidecar_path)
            if (
                len(sidecar_payload) != head["sidecar"]["bytes"]
                or hashlib.sha256(sidecar_payload).hexdigest()
                != head["sidecar"]["sha256"]
            ):
                _fail("Formal head sidecar file binding differs.")
            validated_sidecar = validate_checkpoint_sidecar(
                sidecar,
                embedded_receipt=embedded,
                checkpoint_path=checkpoint_path,
            )
            actual_parent = checkpoint_parent_record(
                embedded_receipt=embedded,
                sidecar=validated_sidecar,
            )
            if not _strict_equal(actual_parent, parent_record):
                _fail("Formal head differs from checkpoint/sidecar identity.")
            entries.append(
                {
                    "embedded_receipt": embedded,
                    "sidecar": validated_sidecar,
                }
            )
            previous_head_hash = head["head_payload_sha256"]
            latest_head = head
        if [path.name for path in self._head_paths()] != [
            path.name for path in head_paths
        ]:
            _fail("Formal checkpoint head set changed during validation.")
        first_parent = entries[0]["embedded_receipt"]["parent_checkpoint"]
        ancestor_entries: list[dict[str, Any]] = []
        ancestor_proof: dict[str, Any] | None = None
        if first_parent is None:
            if ancestor_record is not None:
                _fail("Fresh formal run head must not bind an ancestor proof.")
            if self._read_ancestor_chain_proof() is not None:
                _fail("Fresh formal run must not contain an ancestor proof.")
        else:
            if type(ancestor_record) is not dict:
                _fail("Resumed formal run lacks a bound ancestor proof.")
            retained_proof = self._read_ancestor_chain_proof(
                expected_record=ancestor_record,
            )
            assert retained_proof is not None
            ancestor_proof, _validated_record = retained_proof
            if not _strict_equal(
                ancestor_proof["latest_parent"],
                first_parent,
            ):
                _fail(
                    "Ancestor chain proof does not end at the local resume parent."
                )
            ancestor_entries = ancestor_proof["entries"]
        full_entries = [*ancestor_entries, *entries]
        chain = _validate_full_formal_chain(full_entries)
        assert latest_head is not None
        if not _strict_equal(
            chain["latest_head"],
            latest_head["checkpoint_parent_record"],
        ):
            _fail("Formal latest head differs from the validated receipt chain.")
        return {
            "head": latest_head,
            "head_path": head_paths[-1],
            "chain": chain,
            "entries": entries,
            "ancestor_entries": ancestor_entries,
            "full_entries": chain["entries"],
            "ancestor_chain_proof": ancestor_proof,
            "ancestor_chain_proof_record": (
                None
                if ancestor_record is unset_ancestor_record
                else ancestor_record
            ),
        }

    def _checkpoint_files(self) -> dict[str, Path]:
        checkpoints: dict[str, Path] = {}
        for entry in os.scandir(self.run_dir):
            if _CHECKPOINT_NAME_RE.fullmatch(entry.name) is not None:
                if entry.is_symlink() or not entry.is_file(
                    follow_symlinks=False
                ):
                    _fail(
                        "Formal checkpoint candidate is not a regular file: "
                        f"{entry.name}"
                    )
                checkpoints[entry.name] = Path(entry.path)
            elif _CHECKPOINT_LIKE_RE.fullmatch(entry.name) is not None:
                _fail(f"Malformed formal checkpoint filename: {entry.name}")
        return checkpoints

    def _sidecar_files(self) -> dict[str, Path]:
        sidecars: dict[str, Path] = {}
        for entry in os.scandir(self.receipts_dir):
            if _SIDECAR_NAME_RE.fullmatch(entry.name) is not None:
                if entry.is_symlink() or not entry.is_file(
                    follow_symlinks=False
                ):
                    _fail(
                        "Formal sidecar candidate is not a regular file: "
                        f"{entry.name}"
                    )
                sidecars[entry.name] = Path(entry.path)
            elif _SIDECAR_LIKE_RE.fullmatch(entry.name) is not None:
                _fail(f"Malformed formal sidecar filename: {entry.name}")
        return sidecars

    @staticmethod
    def _validate_launch_resume_parent(
        launch: Mapping[str, Any],
        parent: Mapping[str, Any],
    ) -> None:
        validated_launch = validate_training_launch_receipt(dict(launch))
        resume = validated_launch["payload"]["resume"]
        expected = (
            parent["checkpoint_sha256"],
            parent["embedded_receipt_sha256"],
            parent["sidecar_payload_sha256"],
            parent["updates_completed"],
            parent["consumed_transitions"],
        )
        actual = (
            resume["parent_checkpoint_sha256"],
            resume["parent_embedded_receipt_sha256"],
            resume["parent_sidecar_payload_sha256"],
            resume["parent_updates_completed"],
            resume["parent_consumed_transitions"],
        )
        if resume["is_resume"] is not True or not _strict_equal(
            actual,
            expected,
        ):
            _fail(
                "Interrupted formal checkpoint does not bind its launch "
                "resume parent."
            )

    def recover_interrupted_checkpoint(
        self,
        *,
        map_location: str,
    ) -> dict[str, Any] | None:
        """Retain and load one orphan without publishing its commit marker."""
        if not self.lock_held:
            _fail("Formal training lock is not held.")
        if self._pending_recovery is not None:
            _fail("Formal recovery already has a retained pending candidate.")
        if type(map_location) is not str or not map_location:
            _fail("Formal recovery map_location must be a non-empty string.")
        committed = self._load_committed_chain()
        if committed is not None:
            self._ancestor_chain_proof = committed[
                "ancestor_chain_proof"
            ]
        else:
            standalone_proof = self._read_ancestor_chain_proof()
            launch_resume = self.launch_receipt["payload"]["resume"]
            if launch_resume["is_resume"] is False:
                if standalone_proof is not None:
                    _fail("Fresh formal run must not contain an ancestor proof.")
            elif standalone_proof is not None:
                proof, _record = standalone_proof
                self._validate_launch_resume_parent(
                    self.launch_receipt,
                    proof["latest_parent"],
                )
                _validate_generation_compatibility(
                    self.launch_receipt,
                    proof["entries"][-1]["embedded_receipt"][
                        "launch_receipt"
                    ],
                    proof["latest_parent"],
                )
                self._ancestor_chain_proof = proof
                self._resume_parent = proof["latest_parent"]
        committed_checkpoint_names = (
            set()
            if committed is None
            else {
                entry["embedded_receipt"]["checkpoint_progress"]["filename"]
                for entry in committed["entries"]
            }
        )
        committed_sidecar_names = {
            _sidecar_name(checkpoint_name)
            for checkpoint_name in committed_checkpoint_names
        }
        checkpoints = self._checkpoint_files()
        sidecars = self._sidecar_files()
        orphan_checkpoint_names = (
            set(checkpoints) - committed_checkpoint_names
        )
        orphan_sidecar_names = set(sidecars) - committed_sidecar_names
        if not orphan_checkpoint_names:
            if orphan_sidecar_names:
                _fail(
                    "Formal run has an orphan sidecar without its checkpoint."
                )
            if committed is not None:
                _fail(
                    "Formal run already has a committed checkpoint and no "
                    "recoverable interrupted save; resume into a distinct run."
                )
            return None
        if len(orphan_checkpoint_names) != 1:
            _fail("Formal run has multiple orphan checkpoint candidates.")
        checkpoint_name = next(iter(orphan_checkpoint_names))
        expected_sidecar_name = _sidecar_name(checkpoint_name)
        if orphan_sidecar_names - {expected_sidecar_name}:
            _fail("Formal run has a sidecar for a different orphan checkpoint.")
        checkpoint_path = checkpoints[checkpoint_name]
        retained_payload, retained_record = retain_regular_file(
            checkpoint_path
        )
        loaded, embedded = _safe_checkpoint_receipt(
            retained_payload,
            checkpoint_filename=checkpoint_name,
            map_location=map_location,
        )
        if not _strict_equal(
            embedded["launch_receipt"],
            self.launch_receipt,
        ):
            _fail(
                "Interrupted checkpoint launch differs from current formal launch."
            )
        expected_parent = (
            None
            if committed is None
            else committed["chain"]["latest_head"]
        )
        if (
            expected_parent is None
            and self.launch_receipt["payload"]["resume"]["is_resume"]
        ):
            if (
                self._ancestor_chain_proof is None
                or self._resume_parent is None
            ):
                _fail(
                    "Interrupted resumed checkpoint lacks its validated "
                    "ancestor chain proof."
                )
            expected_parent = self._resume_parent
        if not _strict_equal(
            embedded["parent_checkpoint"],
            expected_parent,
        ):
            _fail(
                "Interrupted checkpoint does not extend the latest formal head."
            )

        if expected_sidecar_name in orphan_sidecar_names:
            sidecar_path = sidecars[expected_sidecar_name]
            sidecar_document, _sidecar_payload = _read_canonical_json(
                sidecar_path
            )
            validated_sidecar = validate_checkpoint_sidecar(
                sidecar_document,
                embedded_receipt=embedded,
                checkpoint_path=checkpoint_path,
            )
            if (
                validated_sidecar["checkpoint"]["sha256"]
                != retained_record["sha256"]
                or validated_sidecar["checkpoint"]["bytes"]
                != retained_record["bytes"]
            ):
                _fail(
                    "Interrupted sidecar differs from the first retained "
                    "checkpoint bytes."
                )
        _assert_retained_checkpoint_unchanged(
            checkpoint_path,
            expected_payload=retained_payload,
            expected_record=retained_record,
            location="Interrupted recovery candidate",
        )
        candidate_token = secrets.token_hex(32)
        candidate = {
            "candidate_token": candidate_token,
            "loaded_dict": loaded,
            "embedded_receipt": embedded,
        }
        self._pending_recovery = {
            "candidate_token": candidate_token,
            "candidate_object": candidate,
            "loaded_dict_object": loaded,
            "checkpoint_path": checkpoint_path,
            "checkpoint_payload": retained_payload,
            "checkpoint_record": retained_record,
            "embedded_receipt": embedded,
        }
        return candidate

    def commit_recovery(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit a retained orphan only after runner state validation."""
        if not self.lock_held:
            _fail("Formal training lock is not held.")
        pending = self._pending_recovery
        if pending is None:
            _fail("Formal recovery has no retained pending candidate.")
        if (
            type(candidate) is not dict
            or set(candidate)
            != {
                "candidate_token",
                "loaded_dict",
                "embedded_receipt",
            }
            or candidate is not pending["candidate_object"]
            or candidate["candidate_token"] != pending["candidate_token"]
            or candidate["loaded_dict"]
            is not pending["loaded_dict_object"]
            or not _strict_equal(
                candidate["embedded_receipt"],
                pending["embedded_receipt"],
            )
        ):
            _fail("Formal recovery candidate identity changed before commit.")
        checkpoint_path = pending["checkpoint_path"]
        _assert_retained_checkpoint_unchanged(
            checkpoint_path,
            expected_payload=pending["checkpoint_payload"],
            expected_record=pending["checkpoint_record"],
            location="Pre-commit recovery candidate",
        )

        def unexpected_factory() -> Mapping[str, Any]:
            _fail(
                "Interrupted checkpoint disappeared before formal recovery."
            )

        publication = self._publish_checkpoint(
            checkpoint_path=checkpoint_path,
            embedded_receipt=pending["embedded_receipt"],
            saved_dict_factory=unexpected_factory,
            expected_checkpoint_payload=pending["checkpoint_payload"],
            expected_checkpoint_record=pending["checkpoint_record"],
        )
        _assert_retained_checkpoint_unchanged(
            checkpoint_path,
            expected_payload=pending["checkpoint_payload"],
            expected_record=pending["checkpoint_record"],
            location="Committed recovery candidate",
        )
        self._pending_recovery = None
        return publication

    @classmethod
    def _read_only_view(cls, run_dir: Path) -> "FormalTrainingIO":
        view = cls.__new__(cls)
        view.run_dir = run_dir
        view.receipts_dir = run_dir / "checkpoint_receipts"
        view.heads_dir = view.receipts_dir / "heads"
        view.launch_receipt = {}
        view._resume_parent = None
        view._ancestor_chain_proof = None
        view._pending_recovery = None
        view._lock_descriptor = None
        return view

    def _prepare_ancestor_binding(
        self,
        *,
        embedded: Mapping[str, Any],
        committed: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if committed is not None:
            self._ancestor_chain_proof = committed[
                "ancestor_chain_proof"
            ]
            return committed["ancestor_chain_proof_record"]
        parent = embedded["parent_checkpoint"]
        existing = self._read_ancestor_chain_proof()
        if parent is None:
            if existing is not None or self._ancestor_chain_proof is not None:
                _fail("Fresh formal run must not contain an ancestor proof.")
            return None
        self._validate_launch_resume_parent(
            embedded["launch_receipt"],
            parent,
        )
        proof = self._ancestor_chain_proof
        if proof is None:
            if existing is None:
                _fail(
                    "Resumed formal run requires a validated ancestor proof "
                    "before its first local checkpoint."
                )
            proof = existing[0]
        proof = _validate_ancestor_chain_proof(dict(proof))
        if not _strict_equal(proof["latest_parent"], parent):
            _fail(
                "Ancestor chain proof is stale, forked, or does not end at "
                "the exact resume parent."
            )
        _validate_generation_compatibility(
            embedded["launch_receipt"],
            proof["entries"][-1]["embedded_receipt"]["launch_receipt"],
            parent,
        )
        validated_proof, record = self._publish_ancestor_chain_proof(proof)
        self._ancestor_chain_proof = validated_proof
        return record

    def publish_checkpoint(
        self,
        *,
        checkpoint_path: str | os.PathLike[str],
        embedded_receipt: Mapping[str, Any],
        saved_dict_factory: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Publish checkpoint, portable sidecar, then immutable head marker."""
        if self._pending_recovery is not None:
            _fail(
                "Pending formal recovery must use commit_recovery before "
                "normal checkpoint publication."
            )
        return self._publish_checkpoint(
            checkpoint_path=checkpoint_path,
            embedded_receipt=embedded_receipt,
            saved_dict_factory=saved_dict_factory,
            expected_checkpoint_payload=None,
            expected_checkpoint_record=None,
        )

    def _publish_checkpoint(
        self,
        *,
        checkpoint_path: str | os.PathLike[str],
        embedded_receipt: Mapping[str, Any],
        saved_dict_factory: Callable[[], Mapping[str, Any]],
        expected_checkpoint_payload: bytes | None,
        expected_checkpoint_record: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Publish while retaining one checkpoint identity across all stages."""
        if not self.lock_held:
            _fail("Formal training lock is not held.")
        if (expected_checkpoint_payload is None) != (
            expected_checkpoint_record is None
        ):
            _fail("Formal checkpoint retention expectation is incomplete.")
        embedded = validate_embedded_checkpoint_receipt(
            dict(embedded_receipt)
        )
        progress = embedded["checkpoint_progress"]
        target = self._assert_checkpoint_target(
            Path(checkpoint_path),
            iteration=progress["iter"],
        )
        committed = self._load_committed_chain()
        if committed is not None:
            latest_updates = committed["chain"]["latest_head"][
                "updates_completed"
            ]
            if progress["updates_completed"] < latest_updates:
                _fail("Formal checkpoint publication would roll back the head.")
            if progress["updates_completed"] == latest_updates:
                latest_entry = committed["entries"][-1]
                if not _strict_equal(
                    latest_entry["embedded_receipt"],
                    embedded,
                ):
                    _fail("Existing formal head differs at the same progress.")
                return {
                    "status": "already_committed_idempotent",
                    "embedded_receipt": latest_entry["embedded_receipt"],
                    "sidecar": latest_entry["sidecar"],
                    "parent_record": committed["chain"]["latest_head"],
                    "head": committed["head"],
                }
        ancestor_chain_proof_record = self._prepare_ancestor_binding(
            embedded=embedded,
            committed=committed,
        )
        previous_parent = (
            self._resume_parent
            if committed is None
            else committed["chain"]["latest_head"]
        )
        if not _strict_equal(
            embedded["parent_checkpoint"],
            previous_parent,
        ):
            _fail("Embedded checkpoint parent is not the latest formal head.")

        if target.exists() or target.is_symlink():
            if expected_checkpoint_payload is None:
                _fail(
                    "Unexpected pre-existing checkpoint requires formal "
                    "recovery validation."
                )
            checkpoint_payload, checkpoint_record = retain_regular_file(target)
            if (
                checkpoint_payload != expected_checkpoint_payload
                or not _strict_equal(
                    checkpoint_record,
                    dict(expected_checkpoint_record),
                )
            ):
                _fail(
                    "Recovery checkpoint differs from its first retained "
                    "path or bytes."
                )
            _loaded, existing_embedded = _safe_checkpoint_receipt(
                checkpoint_payload,
                checkpoint_filename=target.name,
            )
            if not _strict_equal(existing_embedded, embedded):
                _fail("Existing incomplete checkpoint receipt differs.")
        else:
            if expected_checkpoint_payload is not None:
                _fail("Retained recovery checkpoint disappeared before commit.")
            saved_dict = dict(saved_dict_factory())
            if (
                "training_receipt" not in saved_dict
                or not _strict_equal(saved_dict["training_receipt"], embedded)
                or saved_dict.get("iter") != progress["iter"]
            ):
                _fail("Checkpoint factory did not bind formal receipt and iter.")
            checkpoint_payload, checkpoint_record = (
                _publish_new_torch_checkpoint(target, saved_dict)
            )

        _assert_retained_checkpoint_unchanged(
            target,
            expected_payload=checkpoint_payload,
            expected_record=checkpoint_record,
            location="Pre-sidecar formal publication",
        )

        sidecar_path = self.receipts_dir / _sidecar_name(target.name)
        expected_sidecar = build_checkpoint_sidecar(
            checkpoint_path=target,
            embedded_receipt=embedded,
        )
        expected_sidecar_payload = canonical_training_receipt_json_bytes(
            expected_sidecar
        )
        if (
            expected_sidecar["checkpoint"]["sha256"]
            != checkpoint_record["sha256"]
            or expected_sidecar["checkpoint"]["bytes"]
            != checkpoint_record["bytes"]
        ):
            _fail(
                "Formal sidecar was built from a replaced checkpoint path."
            )
        _assert_retained_checkpoint_unchanged(
            target,
            expected_payload=checkpoint_payload,
            expected_record=checkpoint_record,
            location="Post-sidecar formal publication",
        )
        if sidecar_path.exists() or sidecar_path.is_symlink():
            existing_sidecar, existing_payload = _read_canonical_json(
                sidecar_path
            )
            validated_sidecar = validate_checkpoint_sidecar(
                existing_sidecar,
                embedded_receipt=embedded,
                checkpoint_path=target,
            )
            if (
                existing_payload != expected_sidecar_payload
                or not _strict_equal(validated_sidecar, expected_sidecar)
            ):
                _fail("Existing incomplete checkpoint sidecar differs.")
        else:
            _publish_new_bytes(sidecar_path, expected_sidecar_payload)
            validated_sidecar = expected_sidecar
        _assert_retained_checkpoint_unchanged(
            target,
            expected_payload=checkpoint_payload,
            expected_record=checkpoint_record,
            location="Pre-head formal publication",
        )

        current_parent = checkpoint_parent_record(
            embedded_receipt=embedded,
            sidecar=validated_sidecar,
        )
        previous_head_hash = (
            None
            if committed is None
            else committed["head"]["head_payload_sha256"]
        )
        head = _build_head(
            current_parent=current_parent,
            sidecar_name=sidecar_path.name,
            sidecar_payload=expected_sidecar_payload,
            ancestor_chain_proof=ancestor_chain_proof_record,
            previous_head_payload_sha256=previous_head_hash,
        )
        head_path = self.heads_dir / _head_name(
            progress["updates_completed"]
        )
        head_payload = canonical_training_receipt_json_bytes(head)
        if head_path.exists() or head_path.is_symlink():
            existing_head, existing_head_payload = _read_canonical_json(
                head_path
            )
            validated_head = _validate_head(
                existing_head,
                expected_name=head_path.name,
            )
            if (
                existing_head_payload != head_payload
                or not _strict_equal(validated_head, head)
            ):
                _fail("Existing formal checkpoint head differs.")
        else:
            _publish_new_bytes(head_path, head_payload)
        _assert_retained_checkpoint_unchanged(
            target,
            expected_payload=checkpoint_payload,
            expected_record=checkpoint_record,
            location="Post-head formal publication",
        )
        final = self._load_committed_chain()
        _assert_retained_checkpoint_unchanged(
            target,
            expected_payload=checkpoint_payload,
            expected_record=checkpoint_record,
            location="Final formal publication",
        )
        if (
            final is None
            or not _strict_equal(final["chain"]["latest_head"], current_parent)
        ):
            _fail("Formal checkpoint commit marker did not validate.")
        return {
            "status": "committed",
            "embedded_receipt": embedded,
            "sidecar": validated_sidecar,
            "parent_record": current_parent,
            "head": head,
        }

    @staticmethod
    def _validate_resume_compatibility(
        current_launch: Mapping[str, Any],
        parent_launch: Mapping[str, Any],
        parent_record: Mapping[str, Any],
    ) -> None:
        _validate_generation_compatibility(
            current_launch,
            parent_launch,
            parent_record,
        )

    def load_resume_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        map_location: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate a parent latest head before loading algorithm state."""
        if not self.lock_held:
            _fail("Formal training lock is not held.")
        candidate = _canonical_absolute(
            checkpoint_path,
            location="formal resume checkpoint",
        )
        if candidate.parent == self.run_dir:
            _fail("Formal resume must write to a distinct run directory.")
        if _CHECKPOINT_NAME_RE.fullmatch(candidate.name) is None:
            _fail("Formal resume checkpoint filename is invalid.")
        parent_io = FormalTrainingIO._read_only_view(candidate.parent)
        parent_io._lock_descriptor = _acquire_existing_run_lock(
            parent_io.run_dir
        )
        try:
            parent_chain = parent_io._load_committed_chain()
            if parent_chain is None:
                _fail("Formal parent run has no committed checkpoint head.")
            latest_parent = parent_chain["chain"]["latest_head"]
            if latest_parent["checkpoint_file_name"] != candidate.name:
                _fail("Requested formal resume checkpoint is not the latest head.")
            retained_payload, retained_record = retain_regular_file(candidate)
            if (
                retained_record["sha256"]
                != latest_parent["checkpoint_sha256"]
                or retained_record["bytes"] != latest_parent["checkpoint_bytes"]
            ):
                _fail("Requested resume checkpoint differs from latest head.")
            loaded, embedded = _safe_checkpoint_receipt(
                retained_payload,
                checkpoint_filename=candidate.name,
                map_location=map_location,
            )
            self._validate_resume_compatibility(
                self.launch_receipt,
                embedded["launch_receipt"],
                latest_parent,
            )
            ancestor_proof = _build_ancestor_chain_proof(
                parent_chain["full_entries"]
            )
            existing_proof = self._read_ancestor_chain_proof()
            if existing_proof is not None and not _strict_equal(
                existing_proof[0],
                ancestor_proof,
            ):
                _fail(
                    "Existing child ancestor proof differs from the fully "
                    "validated resume parent chain."
                )
            self._resume_parent = latest_parent
            self._ancestor_chain_proof = ancestor_proof
            return loaded, latest_parent
        finally:
            parent_io.close()


def inspect_formal_resume_parent(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate and describe an exact latest resume parent without mutation.

    This preflight intentionally has no current-launch argument.  It permits a
    caller to obtain the exact hashes needed to build that launch receipt.
    ``FormalTrainingIO.load_resume_checkpoint`` repeats the validation after
    configuration and before the algorithm is mutated.
    """
    candidate = _canonical_absolute(
        checkpoint_path,
        location="formal resume checkpoint",
    )
    if _CHECKPOINT_NAME_RE.fullmatch(candidate.name) is None:
        _fail("Formal resume checkpoint filename is invalid.")
    parent_io = FormalTrainingIO._read_only_view(candidate.parent)
    parent_io._lock_descriptor = _acquire_existing_run_lock(
        parent_io.run_dir
    )
    try:
        parent_chain = parent_io._load_committed_chain()
        if parent_chain is None:
            _fail("Formal parent run has no committed checkpoint head.")
        latest_parent = parent_chain["chain"]["latest_head"]
        if latest_parent["checkpoint_file_name"] != candidate.name:
            _fail("Requested formal resume checkpoint is not the latest head.")
        retained_payload, retained_record = retain_regular_file(candidate)
        if (
            retained_record["sha256"]
            != latest_parent["checkpoint_sha256"]
            or retained_record["bytes"]
            != latest_parent["checkpoint_bytes"]
        ):
            _fail("Requested resume checkpoint differs from latest head.")
        _safe_loaded, embedded = _safe_checkpoint_receipt(
            retained_payload,
            checkpoint_filename=candidate.name,
        )
        if not _strict_equal(
            checkpoint_parent_record(
                embedded_receipt=embedded,
                sidecar=parent_chain["entries"][-1]["sidecar"],
            ),
            latest_parent,
        ):
            _fail("Inspected resume checkpoint differs from its committed head.")
        ancestor_chain_proof = _build_ancestor_chain_proof(
            parent_chain["full_entries"]
        )
        return {
            "parent_record": parse_canonical_training_receipt_json(
                canonical_training_receipt_json_bytes(latest_parent)
            ),
            "parent_launch_receipt": embedded["launch_receipt"],
            "ancestor_chain_proof": ancestor_chain_proof,
        }
    finally:
        parent_io.close()


__all__ = [
    "FORMAL_CHECKPOINT_HEAD_CONTRACT",
    "FORMAL_CHECKPOINT_HEAD_SCHEMA_VERSION",
    "FORMAL_ANCESTOR_CHAIN_PROOF_CONTRACT",
    "FORMAL_ANCESTOR_CHAIN_PROOF_SCHEMA_VERSION",
    "FORMAL_LAUNCH_RECEIPT_NAME",
    "FormalTrainingIO",
    "FormalTrainingIOError",
    "inspect_formal_resume_parent",
]
