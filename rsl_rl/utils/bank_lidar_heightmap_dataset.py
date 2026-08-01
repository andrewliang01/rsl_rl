# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Packed, deduplicated offline data contract for the H0b baseline.

The on-disk representation owns one table of unique LiDAR packets.  Anchors
reference five causal packet rows instead of copying dense K=5 windows.  This
module is offline-only: it does not participate in PPO collection or deploy
forward paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import torch
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rsl_rl.modules.bank_lidar_heightmap import (
    validate_manifest_target_contract_binding,
)


H0B_PACKED_SHARD_SCHEMA_VERSION = 1
H0B_PACKET_SPATIAL_SIZE = (16, 96)
H0B_TARGET_SPATIAL_SIZE = (28, 20)
H0B_HISTORY_SLOTS = 5
H0B_MAX_RANGE_M = 6.0

_PACKET_MASK_BYTES = H0B_PACKET_SPATIAL_SIZE[0] * H0B_PACKET_SPATIAL_SIZE[1] // 8
_TARGET_MASK_BYTES = H0B_TARGET_SPATIAL_SIZE[0] * H0B_TARGET_SPATIAL_SIZE[1] // 8
_SHARD_KEYS = {
    "schema_version",
    "packet_range_m",
    "packet_valid_bits",
    "packet_capture_step",
    "packet_trajectory_id",
    "anchor_id",
    "anchor_trajectory_id",
    "anchor_capture_step",
    "anchor_frame_ids",
    "target_height_m",
    "target_valid_bits",
}
_MANIFEST_SPLITS = ("train", "val", "test")
_MANIFEST_KEYS = {
    "schema_version",
    "classification",
    "target_contract_payload_sha256",
    "collector_receipt_payload_sha256",
    "source_commits",
    "k1_k5_shared_anchor_set",
    "shards",
    "split_anchor_counts",
    "split_trajectory_ids",
    "manifest_payload_sha256",
}
_MANIFEST_SHARD_KEYS = {
    "split",
    "relative_path",
    "file_size_bytes",
    "file_sha256",
    "payload_sha256",
    "num_unique_packets",
    "num_anchors",
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _require_cpu_tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape_tail: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must reside on CPU in a packed shard.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}.")
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have tail shape {shape_tail}.")
    return value


def pack_bool_mask(mask: torch.Tensor, *, spatial_size: tuple[int, int]) -> torch.Tensor:
    """Pack the final two boolean dimensions into little-bit-order bytes."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.device.type != "cpu" or mask.dtype != torch.bool:
        raise TypeError("mask must be a CPU boolean tensor.")
    if mask.ndim < 2 or tuple(mask.shape[-2:]) != tuple(spatial_size):
        raise ValueError("mask spatial shape differs from its declared contract.")
    num_bits = spatial_size[0] * spatial_size[1]
    if num_bits % 8 != 0:
        raise ValueError("Packed H0b masks require a spatial size divisible by 8.")
    flat = mask.contiguous().reshape(*mask.shape[:-2], num_bits)
    bit_weights = (1 << torch.arange(8, dtype=torch.uint8)).reshape(
        *((1,) * (flat.ndim - 1)), 8
    )
    grouped = flat.reshape(*flat.shape[:-1], num_bits // 8, 8).to(torch.uint8)
    return (grouped * bit_weights).sum(dim=-1).to(torch.uint8)


def unpack_bool_mask(
    packed: torch.Tensor,
    *,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    """Invert :func:`pack_bool_mask` without NumPy or serialized padding."""
    if not isinstance(packed, torch.Tensor):
        raise TypeError("packed must be a torch.Tensor.")
    if packed.device.type != "cpu" or packed.dtype != torch.uint8:
        raise TypeError("packed must be a CPU uint8 tensor.")
    num_bits = spatial_size[0] * spatial_size[1]
    if num_bits % 8 != 0 or packed.shape[-1] != num_bits // 8:
        raise ValueError("Packed mask byte count differs from its spatial contract.")
    shifts = torch.arange(8, dtype=torch.uint8).reshape(
        *((1,) * packed.ndim), 8
    )
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
    return bits.reshape(*packed.shape[:-1], *spatial_size)


def _tensor_payload_sha256(shard: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(H0B_PACKED_SHARD_SCHEMA_VERSION).encode("ascii"))
    for name in sorted(_SHARD_KEYS - {"schema_version"}):
        value = shard[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Packed shard field {name} must be a tensor.")
        contiguous = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_packed_h0b_shard(shard: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on layout, causal provenance, and masked-value semantics."""
    if not isinstance(shard, Mapping) or set(shard) != _SHARD_KEYS:
        raise ValueError("Packed H0b shard keys differ from schema version 1.")
    if type(shard["schema_version"]) is not int or shard["schema_version"] != 1:
        raise ValueError("Packed H0b shard schema_version must be exactly 1.")

    packet_range = _require_cpu_tensor(
        shard["packet_range_m"],
        name="packet_range_m",
        dtype=torch.float16,
        shape_tail=H0B_PACKET_SPATIAL_SIZE,
    )
    num_packets = packet_range.shape[0]
    if num_packets <= 0:
        raise ValueError("Packed H0b shard must contain at least one packet.")
    packet_valid_bits = _require_cpu_tensor(
        shard["packet_valid_bits"],
        name="packet_valid_bits",
        dtype=torch.uint8,
        shape_tail=(_PACKET_MASK_BYTES,),
    )
    packet_step = _require_cpu_tensor(
        shard["packet_capture_step"],
        name="packet_capture_step",
        dtype=torch.int64,
        shape_tail=(),
    )
    packet_trajectory = _require_cpu_tensor(
        shard["packet_trajectory_id"],
        name="packet_trajectory_id",
        dtype=torch.int64,
        shape_tail=(),
    )
    if not all(
        value.shape[0] == num_packets
        for value in (packet_valid_bits, packet_step, packet_trajectory)
    ):
        raise ValueError("Packet table fields must have one shared row count.")

    anchor_id = _require_cpu_tensor(
        shard["anchor_id"],
        name="anchor_id",
        dtype=torch.int64,
        shape_tail=(),
    )
    num_anchors = anchor_id.shape[0]
    if num_anchors <= 0:
        raise ValueError("Packed H0b shard must contain at least one anchor.")
    anchor_trajectory = _require_cpu_tensor(
        shard["anchor_trajectory_id"],
        name="anchor_trajectory_id",
        dtype=torch.int64,
        shape_tail=(),
    )
    anchor_step = _require_cpu_tensor(
        shard["anchor_capture_step"],
        name="anchor_capture_step",
        dtype=torch.int64,
        shape_tail=(),
    )
    frame_ids = _require_cpu_tensor(
        shard["anchor_frame_ids"],
        name="anchor_frame_ids",
        dtype=torch.int64,
        shape_tail=(H0B_HISTORY_SLOTS,),
    )
    target_height = _require_cpu_tensor(
        shard["target_height_m"],
        name="target_height_m",
        dtype=torch.float16,
        shape_tail=H0B_TARGET_SPATIAL_SIZE,
    )
    target_valid_bits = _require_cpu_tensor(
        shard["target_valid_bits"],
        name="target_valid_bits",
        dtype=torch.uint8,
        shape_tail=(_TARGET_MASK_BYTES,),
    )
    if not all(
        value.shape[0] == num_anchors
        for value in (
            anchor_trajectory,
            anchor_step,
            frame_ids,
            target_height,
            target_valid_bits,
        )
    ):
        raise ValueError("Anchor table fields must have one shared row count.")
    if torch.unique(anchor_id).numel() != num_anchors:
        raise ValueError("anchor_id must be unique within a packed shard.")
    if bool((anchor_id < 0).any()) or bool((anchor_trajectory < 0).any()):
        raise ValueError("Anchor and trajectory identifiers must be non-negative.")
    if bool((anchor_step < 0).any()) or bool((packet_step < 0).any()):
        raise ValueError("Packet and anchor capture steps must be non-negative.")
    if bool((frame_ids < 0).any()) or bool((frame_ids >= num_packets).any()):
        raise ValueError("Every anchor frame id must reference the packet table.")
    current_frame_ids = frame_ids[:, -1]
    if torch.unique(current_frame_ids).numel() != num_anchors:
        raise ValueError(
            "Each H0b packet-anchor window must own one unique current packet."
        )

    referenced_steps = packet_step[frame_ids]
    referenced_trajectories = packet_trajectory[frame_ids]
    if bool((referenced_steps > anchor_step.unsqueeze(1)).any()):
        raise ValueError("H0b source packet timestamps must not exceed anchor time.")
    if bool((referenced_trajectories != anchor_trajectory.unsqueeze(1)).any()):
        raise ValueError("H0b source packets and anchor must share one trajectory.")
    if bool((referenced_steps[:, 1:] < referenced_steps[:, :-1]).any()):
        raise ValueError("H0b source frame ids must be oldest-to-newest in time.")
    if bool((referenced_steps[:, -1] != anchor_step).any()):
        raise ValueError("The newest H0b source packet must be the anchor packet.")
    packet_valid = unpack_bool_mask(
        packet_valid_bits,
        spatial_size=H0B_PACKET_SPATIAL_SIZE,
    )
    packet_float = packet_range.to(torch.float32)
    if not bool(torch.isfinite(packet_float).all()):
        raise ValueError("Serialized H0b packet ranges must all be finite.")
    if bool((packet_float[~packet_valid] != 0.0).any()):
        raise ValueError("Serialized unknown LiDAR cells must have zero range.")
    valid_range = packet_float[packet_valid]
    if bool(((valid_range <= 0.0) | (valid_range > H0B_MAX_RANGE_M)).any()):
        raise ValueError("Valid serialized LiDAR ranges must lie in (0,6] metres.")

    target_valid = unpack_bool_mask(
        target_valid_bits,
        spatial_size=H0B_TARGET_SPATIAL_SIZE,
    )
    target_float = target_height.to(torch.float32)
    if not bool(torch.isfinite(target_float).all()):
        raise ValueError("Serialized H0b target heights must all be finite.")
    if bool((target_float[~target_valid] != 0.0).any()):
        raise ValueError("Serialized unknown target cells must have zero height.")
    if not bool(target_valid.any()):
        raise ValueError("Packed H0b shard must contain at least one valid target cell.")

    dense_k5_range_bytes = (
        num_anchors
        * H0B_HISTORY_SLOTS
        * H0B_PACKET_SPATIAL_SIZE[0]
        * H0B_PACKET_SPATIAL_SIZE[1]
        * 2
    )
    packet_table_range_bytes = packet_range.numel() * packet_range.element_size()
    return {
        "schema_version": H0B_PACKED_SHARD_SCHEMA_VERSION,
        "num_unique_packets": num_packets,
        "num_anchors": num_anchors,
        "packet_valid_count": int(packet_valid.sum().item()),
        "target_valid_count": int(target_valid.sum().item()),
        "payload_sha256": _tensor_payload_sha256(shard),
        "dense_k5_range_bytes": dense_k5_range_bytes,
        "packet_table_range_bytes": packet_table_range_bytes,
        "deduplicated_range_byte_ratio": (
            packet_table_range_bytes / dense_k5_range_bytes
        ),
        "k1_k5_shared_anchor_contract": True,
        "one_target_per_unique_current_packet": True,
        "history_order": "oldest_to_newest",
        "unknown_serialization": "zero_value_plus_bitpacked_validity",
    }


def materialize_h0b_batch(
    shard: Mapping[str, Any],
    anchor_rows: Sequence[int] | torch.Tensor,
    *,
    history_length: int,
) -> dict[str, torch.Tensor]:
    """Materialize K=1 or K=5 tensors from the same frozen anchor rows."""
    validate_packed_h0b_shard(shard)
    if history_length not in (1, 5):
        raise ValueError("history_length must be exactly 1 or 5.")
    rows = torch.as_tensor(anchor_rows, dtype=torch.int64, device="cpu")
    if rows.ndim != 1 or rows.numel() <= 0:
        raise ValueError("anchor_rows must be a non-empty one-dimensional index.")
    num_anchors = shard["anchor_id"].shape[0]
    if bool((rows < 0).any()) or bool((rows >= num_anchors).any()):
        raise IndexError("anchor_rows contains an out-of-range index.")

    frame_ids = shard["anchor_frame_ids"][rows]
    if history_length == 1:
        frame_ids = frame_ids[:, -1:]
    packet_range = shard["packet_range_m"][frame_ids]
    packet_valid_all = unpack_bool_mask(
        shard["packet_valid_bits"],
        spatial_size=H0B_PACKET_SPATIAL_SIZE,
    )
    packet_valid = packet_valid_all[frame_ids]
    ray_history = torch.stack(
        (packet_range, packet_valid.to(torch.float16)),
        dim=2,
    )
    target_valid_all = unpack_bool_mask(
        shard["target_valid_bits"],
        spatial_size=H0B_TARGET_SPATIAL_SIZE,
    )
    return {
        "anchor_id": shard["anchor_id"][rows].clone(),
        "ray_history": ray_history.contiguous(),
        "target_height_m": shard["target_height_m"][rows]
        .to(torch.float32)
        .unsqueeze(1),
        "target_valid": target_valid_all[rows].unsqueeze(1),
    }


def save_packed_h0b_shard(
    path: str | Path,
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and atomically create one immutable packed shard."""
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite packed H0b shard: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit = validate_packed_h0b_shard(shard)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary H0b shard path already exists: {temporary}")
    try:
        torch.save(dict(shard), temporary)
        with temporary.open("rb") as stream:
            file_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **audit,
        "path": str(destination),
        "file_size_bytes": destination.stat().st_size,
        "file_sha256": file_sha256,
    }


def load_packed_h0b_shard(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load with ``weights_only`` and verify both file and tensor payload hashes."""
    source = Path(path).resolve(strict=True)
    with source.open("rb") as stream:
        file_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if file_sha256 != expected_file_sha256:
        raise ValueError("Packed H0b shard file SHA-256 mismatch.")
    loaded = torch.load(source, map_location="cpu", weights_only=True)
    audit = validate_packed_h0b_shard(loaded)
    if audit["payload_sha256"] != expected_payload_sha256:
        raise ValueError("Packed H0b shard tensor payload SHA-256 mismatch.")
    return dict(loaded), {
        **audit,
        "path": str(source),
        "file_size_bytes": source.stat().st_size,
        "file_sha256": file_sha256,
    }


def _load_unbound_packed_h0b_shard(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("rb") as stream:
        file_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    audit = validate_packed_h0b_shard(loaded)
    return dict(loaded), {
        **audit,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": file_sha256,
    }


def _normalize_source_commits(source_commits: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(source_commits, Mapping) or set(source_commits) != {
        "lab_pro",
        "rsl_rl",
        "isaaclab",
    }:
        raise ValueError("H0b source commits must bind lab_pro, rsl_rl, and isaaclab.")
    normalized = dict(source_commits)
    if any(
        not isinstance(value, str) or _HEX40.fullmatch(value) is None
        for value in normalized.values()
    ):
        raise ValueError("Every H0b source commit must be lowercase 40-hex.")
    return normalized


def create_h0b_dataset_manifest(
    dataset_root: str | Path,
    *,
    split_shards: Mapping[str, Sequence[str | Path]],
    target_contract_payload_sha256: str,
    collector_receipt_payload_sha256: str,
    source_commits: Mapping[str, str],
    classification: str = "engineering_dataset",
    target_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash all shards and create a grouped-split manifest without copying data."""
    root = Path(dataset_root).resolve(strict=True)
    if not isinstance(split_shards, Mapping) or set(split_shards) != set(
        _MANIFEST_SPLITS
    ):
        raise ValueError("H0b dataset manifest requires train/val/test shard lists.")
    for digest_name, digest in (
        ("target contract", target_contract_payload_sha256),
        ("collector receipt", collector_receipt_payload_sha256),
    ):
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ValueError(f"H0b {digest_name} SHA-256 must be lowercase 64-hex.")
    if target_contract is not None:
        validate_manifest_target_contract_binding(
            target_contract,
            target_contract_payload_sha256,
        )
    if not isinstance(classification, str) or not classification:
        raise ValueError("H0b dataset classification must be non-empty.")

    shard_records: list[dict[str, Any]] = []
    split_anchor_counts: dict[str, int] = {}
    split_trajectory_ids: dict[str, list[int]] = {}
    all_anchor_ids: set[int] = set()
    split_trajectory_sets: dict[str, set[int]] = {}
    seen_paths: set[str] = set()
    for split in _MANIFEST_SPLITS:
        paths = split_shards[split]
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
            raise ValueError(f"H0b split {split} must contain at least one shard.")
        split_anchor_count = 0
        trajectories: set[int] = set()
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            path = path.resolve(strict=True)
            try:
                relative_path = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError("H0b shard path must remain under dataset_root.") from exc
            if relative_path in seen_paths:
                raise ValueError("One H0b shard path cannot appear more than once.")
            seen_paths.add(relative_path)
            shard, audit = _load_unbound_packed_h0b_shard(path)
            anchor_ids = set(map(int, shard["anchor_id"].tolist()))
            if all_anchor_ids.intersection(anchor_ids):
                raise ValueError("H0b anchor IDs must be globally unique across shards.")
            all_anchor_ids.update(anchor_ids)
            trajectories.update(map(int, shard["anchor_trajectory_id"].tolist()))
            split_anchor_count += audit["num_anchors"]
            shard_records.append(
                {
                    "split": split,
                    "relative_path": relative_path,
                    "file_size_bytes": audit["file_size_bytes"],
                    "file_sha256": audit["file_sha256"],
                    "payload_sha256": audit["payload_sha256"],
                    "num_unique_packets": audit["num_unique_packets"],
                    "num_anchors": audit["num_anchors"],
                }
            )
        split_anchor_counts[split] = split_anchor_count
        split_trajectory_sets[split] = trajectories
        split_trajectory_ids[split] = sorted(trajectories)

    for index, left in enumerate(_MANIFEST_SPLITS):
        for right in _MANIFEST_SPLITS[index + 1 :]:
            if split_trajectory_sets[left].intersection(split_trajectory_sets[right]):
                raise ValueError(
                    f"H0b trajectory leakage exists between {left} and {right}."
                )

    payload = {
        "schema_version": 1,
        "classification": classification,
        "target_contract_payload_sha256": target_contract_payload_sha256,
        "collector_receipt_payload_sha256": collector_receipt_payload_sha256,
        "source_commits": _normalize_source_commits(source_commits),
        "k1_k5_shared_anchor_set": True,
        "shards": shard_records,
        "split_anchor_counts": split_anchor_counts,
        "split_trajectory_ids": split_trajectory_ids,
    }
    return {**payload, "manifest_payload_sha256": _canonical_json_sha256(payload)}


def validate_h0b_dataset_manifest(
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    require_formal_400k: bool = False,
) -> dict[str, Any]:
    """Re-hash every shard and prove grouped split isolation."""
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("H0b dataset manifest keys differ from schema version 1.")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("H0b dataset manifest schema_version must be exactly 1.")
    if manifest["k1_k5_shared_anchor_set"] is not True:
        raise ValueError("H0b K1 and K5 must share one frozen anchor set.")
    for field in (
        "target_contract_payload_sha256",
        "collector_receipt_payload_sha256",
        "manifest_payload_sha256",
    ):
        if not isinstance(manifest[field], str) or _HEX64.fullmatch(manifest[field]) is None:
            raise ValueError(f"H0b manifest field {field} must be lowercase 64-hex.")
    payload = {key: manifest[key] for key in _MANIFEST_KEYS - {"manifest_payload_sha256"}}
    if _canonical_json_sha256(payload) != manifest["manifest_payload_sha256"]:
        raise ValueError("H0b dataset manifest payload SHA-256 mismatch.")
    normalized_commits = _normalize_source_commits(manifest["source_commits"])

    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("H0b dataset manifest must list at least one shard.")
    reconstructed_paths: dict[str, list[str]] = {split: [] for split in _MANIFEST_SPLITS}
    for record in shards:
        if not isinstance(record, Mapping) or set(record) != _MANIFEST_SHARD_KEYS:
            raise ValueError("H0b dataset shard record keys changed.")
        split = record["split"]
        if split not in _MANIFEST_SPLITS:
            raise ValueError("H0b dataset shard split is invalid.")
        relative_path = record["relative_path"]
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError("H0b dataset shard path must be a safe relative path.")
        reconstructed_paths[split].append(relative_path)

    rebuilt = create_h0b_dataset_manifest(
        dataset_root,
        split_shards=reconstructed_paths,
        target_contract_payload_sha256=manifest["target_contract_payload_sha256"],
        collector_receipt_payload_sha256=manifest[
            "collector_receipt_payload_sha256"
        ],
        source_commits=normalized_commits,
        classification=manifest["classification"],
    )
    if rebuilt != dict(manifest):
        raise ValueError("H0b dataset manifest differs from live shard evidence.")
    if require_formal_400k:
        expected_counts = {"train": 280_000, "val": 60_000, "test": 60_000}
        if manifest["classification"] != "formal_400k_grouped_v1":
            raise ValueError("Formal H0b data requires formal_400k_grouped_v1 classification.")
        if manifest["split_anchor_counts"] != expected_counts:
            raise ValueError("Formal H0b data must use the frozen 280k/60k/60k split.")
    return {
        "schema_version": 1,
        "classification": manifest["classification"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "split_anchor_counts": dict(manifest["split_anchor_counts"]),
        "split_trajectory_counts": {
            split: len(manifest["split_trajectory_ids"][split])
            for split in _MANIFEST_SPLITS
        },
        "num_shards": len(shards),
        "formal_400k_validated": require_formal_400k,
    }


__all__ = [
    "H0B_HISTORY_SLOTS",
    "H0B_MAX_RANGE_M",
    "H0B_PACKED_SHARD_SCHEMA_VERSION",
    "H0B_PACKET_SPATIAL_SIZE",
    "H0B_TARGET_SPATIAL_SIZE",
    "create_h0b_dataset_manifest",
    "load_packed_h0b_shard",
    "materialize_h0b_batch",
    "pack_bool_mask",
    "save_packed_h0b_shard",
    "unpack_bool_mask",
    "validate_h0b_dataset_manifest",
    "validate_packed_h0b_shard",
]
