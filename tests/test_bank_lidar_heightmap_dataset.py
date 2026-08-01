from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from rsl_rl.utils.bank_lidar_heightmap_dataset import (
    H0B_PACKET_SPATIAL_SIZE,
    H0B_TARGET_SPATIAL_SIZE,
    create_h0b_dataset_manifest,
    load_packed_h0b_shard,
    materialize_h0b_batch,
    pack_bool_mask,
    save_packed_h0b_shard,
    unpack_bool_mask,
    validate_h0b_dataset_manifest,
    validate_packed_h0b_shard,
)


def _shard() -> dict:
    packet_range = torch.zeros(6, *H0B_PACKET_SPATIAL_SIZE, dtype=torch.float16)
    packet_valid = torch.zeros(6, *H0B_PACKET_SPATIAL_SIZE, dtype=torch.bool)
    packet_valid[:, 2:5, 7:13] = True
    packet_range[packet_valid] = torch.linspace(
        0.25,
        5.75,
        int(packet_valid.sum()),
        dtype=torch.float16,
    )
    target_height = torch.zeros(2, *H0B_TARGET_SPATIAL_SIZE, dtype=torch.float16)
    target_valid = torch.zeros(2, *H0B_TARGET_SPATIAL_SIZE, dtype=torch.bool)
    target_valid[:, 3:20, 2:18] = True
    target_height[target_valid] = torch.linspace(
        -0.4,
        1.2,
        int(target_valid.sum()),
        dtype=torch.float16,
    )
    return {
        "schema_version": 1,
        "packet_range_m": packet_range,
        "packet_valid_bits": pack_bool_mask(
            packet_valid,
            spatial_size=H0B_PACKET_SPATIAL_SIZE,
        ),
        "packet_capture_step": torch.tensor([0, 1, 2, 3, 4, 8]),
        "packet_trajectory_id": torch.tensor([10, 10, 10, 10, 10, 11]),
        "anchor_id": torch.tensor([100, 101]),
        "anchor_trajectory_id": torch.tensor([10, 11]),
        "anchor_capture_step": torch.tensor([4, 8]),
        "anchor_frame_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]],
            dtype=torch.int64,
        ),
        "target_height_m": target_height,
        "target_valid_bits": pack_bool_mask(
            target_valid,
            spatial_size=H0B_TARGET_SPATIAL_SIZE,
        ),
    }


def _shift_identity(shard: dict, *, anchor_delta: int, trajectory_delta: int) -> dict:
    shifted = copy.deepcopy(shard)
    shifted["anchor_id"] += anchor_delta
    shifted["anchor_trajectory_id"] += trajectory_delta
    shifted["packet_trajectory_id"] += trajectory_delta
    return shifted


@pytest.mark.parametrize("spatial_size", (H0B_PACKET_SPATIAL_SIZE, H0B_TARGET_SPATIAL_SIZE))
def test_mask_pack_round_trip(spatial_size: tuple[int, int]) -> None:
    torch.manual_seed(17)
    mask = torch.rand(3, *spatial_size) > 0.35
    packed = pack_bool_mask(mask, spatial_size=spatial_size)
    restored = unpack_bool_mask(packed, spatial_size=spatial_size)
    assert torch.equal(restored, mask)
    assert packed.numel() * 8 == mask.numel()


def test_shard_audit_proves_packet_reuse_and_payload_hash() -> None:
    audit = validate_packed_h0b_shard(_shard())
    assert audit["num_unique_packets"] == 6
    assert audit["num_anchors"] == 2
    assert audit["packet_table_range_bytes"] < audit["dense_k5_range_bytes"]
    assert audit["k1_k5_shared_anchor_contract"] is True
    assert audit["one_target_per_unique_current_packet"] is True
    assert len(audit["payload_sha256"]) == 64


def test_k1_and_k5_materialize_from_identical_anchor() -> None:
    shard = _shard()
    k1 = materialize_h0b_batch(shard, [0, 1], history_length=1)
    k5 = materialize_h0b_batch(shard, [0, 1], history_length=5)
    assert k1["ray_history"].shape == (2, 1, 2, 16, 96)
    assert k5["ray_history"].shape == (2, 5, 2, 16, 96)
    assert torch.equal(k1["ray_history"][:, 0], k5["ray_history"][:, -1])
    assert torch.equal(k1["target_height_m"], k5["target_height_m"])
    assert torch.equal(k1["target_valid"], k5["target_valid"])
    assert torch.equal(k1["anchor_id"], k5["anchor_id"])


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda value: value["packet_trajectory_id"].__setitem__(4, 11), "trajectory"),
        (lambda value: value["anchor_capture_step"].__setitem__(0, 2), "timestamps"),
        (lambda value: value["packet_range_m"].__setitem__((0, 0, 0), 1.0), "unknown"),
        (lambda value: value["target_height_m"].__setitem__((0, 0, 0), 1.0), "unknown target"),
        (lambda value: value["anchor_frame_ids"].__setitem__((1, 4), 4), "unique current"),
    ),
)
def test_shard_rejects_provenance_or_mask_leak(mutation, match: str) -> None:
    shard = _shard()
    mutation(shard)
    with pytest.raises(ValueError, match=match):
        validate_packed_h0b_shard(shard)


def test_save_load_is_create_only_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "h0b_shard.pt"
    receipt = save_packed_h0b_shard(path, _shard())
    loaded, loaded_receipt = load_packed_h0b_shard(
        path,
        expected_file_sha256=receipt["file_sha256"],
        expected_payload_sha256=receipt["payload_sha256"],
    )
    assert loaded_receipt == receipt
    assert torch.equal(loaded["anchor_frame_ids"], _shard()["anchor_frame_ids"])
    with pytest.raises(FileExistsError, match="overwrite"):
        save_packed_h0b_shard(path, _shard())
    with pytest.raises(ValueError, match="file SHA"):
        load_packed_h0b_shard(
            path,
            expected_file_sha256="0" * 64,
            expected_payload_sha256=receipt["payload_sha256"],
        )


def test_payload_hash_rejects_semantic_tensor_change(tmp_path: Path) -> None:
    path = tmp_path / "h0b_shard.pt"
    receipt = save_packed_h0b_shard(path, _shard())
    changed = copy.deepcopy(_shard())
    changed["anchor_id"][0] += 99
    changed_path = tmp_path / "changed.pt"
    changed_receipt = save_packed_h0b_shard(changed_path, changed)
    with pytest.raises(ValueError, match="payload SHA"):
        load_packed_h0b_shard(
            changed_path,
            expected_file_sha256=changed_receipt["file_sha256"],
            expected_payload_sha256=receipt["payload_sha256"],
        )


def _manifest(tmp_path: Path) -> dict:
    split_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for index, split in enumerate(("train", "val", "test")):
        relative = f"{split}/part-000.pt"
        save_packed_h0b_shard(
            tmp_path / relative,
            _shift_identity(
                _shard(),
                anchor_delta=index * 1000,
                trajectory_delta=index * 100,
            ),
        )
        split_paths[split].append(relative)
    return create_h0b_dataset_manifest(
        tmp_path,
        split_shards=split_paths,
        target_contract_payload_sha256="1" * 64,
        collector_receipt_payload_sha256="2" * 64,
        source_commits={
            "lab_pro": "3" * 40,
            "rsl_rl": "4" * 40,
            "isaaclab": "5" * 40,
        },
    )


def test_dataset_manifest_rehashes_shards_and_proves_group_isolation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    audit = validate_h0b_dataset_manifest(manifest, tmp_path)
    assert audit["split_anchor_counts"] == {"train": 2, "val": 2, "test": 2}
    assert audit["split_trajectory_counts"] == {"train": 2, "val": 2, "test": 2}
    assert audit["num_shards"] == 3
    assert audit["formal_400k_validated"] is False


def test_dataset_manifest_rejects_cross_split_trajectory_leak(tmp_path: Path) -> None:
    for index, split in enumerate(("train", "val", "test")):
        save_packed_h0b_shard(
            tmp_path / f"{split}.pt",
            _shift_identity(_shard(), anchor_delta=index * 1000, trajectory_delta=0),
        )
    with pytest.raises(ValueError, match="trajectory leakage"):
        create_h0b_dataset_manifest(
            tmp_path,
            split_shards={
                "train": ["train.pt"],
                "val": ["val.pt"],
                "test": ["test.pt"],
            },
            target_contract_payload_sha256="1" * 64,
            collector_receipt_payload_sha256="2" * 64,
            source_commits={
                "lab_pro": "3" * 40,
                "rsl_rl": "4" * 40,
                "isaaclab": "5" * 40,
            },
        )


def test_dataset_manifest_rejects_tamper_and_unearned_formal_label(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["split_anchor_counts"]["train"] += 1
    with pytest.raises(ValueError, match="payload SHA"):
        validate_h0b_dataset_manifest(tampered, tmp_path)
    with pytest.raises(ValueError, match="formal_400k"):
        validate_h0b_dataset_manifest(manifest, tmp_path, require_formal_400k=True)
    create_h0b_dataset_manifest,
