from __future__ import annotations

import copy
import hashlib
import json
import torch
from pathlib import Path

import pytest

from rsl_rl.modules.bank_lidar_heightmap import (
    load_frozen_reconstructor_artifact,
    load_frozen_reconstructor_checkpoint,
    normalize_heightmap_target_contract,
)
from rsl_rl.utils.bank_lidar_heightmap_dataset import (
    H0B_PACKET_SPATIAL_SIZE,
    H0B_TARGET_SPATIAL_SIZE,
    create_h0b_dataset_manifest,
    pack_bool_mask,
    save_packed_h0b_shard,
)
from rsl_rl.utils.bank_lidar_heightmap_pretrain import (
    H0bOfflinePretrainer,
    H0bPretrainConfig,
    load_h0b_pretrain_checkpoint,
    main,
)


def _canonical_sha(value: dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _target_contract() -> dict:
    return {
        "schema_version": 1,
        "target_definition": "synthetic local support height for CPU tests",
        "height_unit": "metre",
        "height_sign": "positive_up",
        "grid_shape": [28, 20],
        "grid_axis_order": ["x", "y"],
        "grid_axis_directions": ["forward", "left"],
        "flatten_order": "C_contiguous_row_major",
        "coordinate_frame": "gravity_aligned_base_yaw",
        "origin": "base projection",
        "resolution_m": 0.05,
        "unknown_cell_policy": "zero plus validity mask",
        "contract_source_sha256": "a" * 64,
    }


def _shard(*, anchor_offset: int, trajectory_offset: int) -> dict:
    packet_range = torch.zeros(6, *H0B_PACKET_SPATIAL_SIZE, dtype=torch.float16)
    packet_valid = torch.zeros(6, *H0B_PACKET_SPATIAL_SIZE, dtype=torch.bool)
    packet_valid[:, 2:14, 3:93] = True
    for row in range(6):
        packet_range[row][packet_valid[row]] = 1.0 + 0.1 * row
    target_height = torch.zeros(2, *H0B_TARGET_SPATIAL_SIZE, dtype=torch.float16)
    target_valid = torch.zeros(2, *H0B_TARGET_SPATIAL_SIZE, dtype=torch.bool)
    target_valid[:, 2:26, 2:18] = True
    target_height[0][target_valid[0]] = 0.20
    target_height[1][target_valid[1]] = -0.10
    return {
        "schema_version": 1,
        "packet_range_m": packet_range,
        "packet_valid_bits": pack_bool_mask(packet_valid, spatial_size=H0B_PACKET_SPATIAL_SIZE),
        "packet_capture_step": torch.tensor([0, 1, 2, 3, 4, 8]),
        "packet_trajectory_id": torch.tensor([
            trajectory_offset,
            trajectory_offset,
            trajectory_offset,
            trajectory_offset,
            trajectory_offset,
            trajectory_offset + 1,
        ]),
        "anchor_id": torch.tensor([anchor_offset, anchor_offset + 1]),
        "anchor_trajectory_id": torch.tensor([trajectory_offset, trajectory_offset + 1]),
        "anchor_capture_step": torch.tensor([4, 8]),
        "anchor_frame_ids": torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]], dtype=torch.int64),
        "target_height_m": target_height,
        "target_valid_bits": pack_bool_mask(target_valid, spatial_size=H0B_TARGET_SPATIAL_SIZE),
    }


def _dataset(tmp_path: Path) -> tuple[dict, dict]:
    split_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for index, split in enumerate(("train", "val", "test")):
        relative = f"{split}/part-000.pt"
        save_packed_h0b_shard(
            tmp_path / relative,
            _shard(
                anchor_offset=1000 * (index + 1),
                trajectory_offset=100 * (index + 1),
            ),
        )
        split_paths[split].append(relative)
    contract = normalize_heightmap_target_contract(_target_contract())
    manifest = create_h0b_dataset_manifest(
        tmp_path,
        split_shards=split_paths,
        target_contract_payload_sha256=_canonical_sha(contract),
        collector_receipt_payload_sha256="b" * 64,
        source_commits={
            "lab_pro": "c" * 40,
            "rsl_rl": "d" * 40,
            "isaaclab": "e" * 40,
        },
    )
    return manifest, contract


def _trainer(
    tmp_path: Path,
    *,
    config: H0bPretrainConfig,
    manifest: dict | None = None,
    contract: dict | None = None,
    resume: dict | None = None,
) -> H0bOfflinePretrainer:
    if manifest is None or contract is None:
        manifest, contract = _dataset(tmp_path)
    return H0bOfflinePretrainer(
        config=config,
        dataset_root=tmp_path,
        manifest=manifest,
        target_contract=contract,
        resume_checkpoint=resume,
    )


@pytest.mark.parametrize("history_length", [1, 5])
def test_two_losses_descend_after_one_cpu_step_and_share_anchor_order(tmp_path: Path, history_length: int) -> None:
    """Both stages learn once and K changes never alter anchor permutation."""
    manifest, contract = _dataset(tmp_path)
    ae_config = H0bPretrainConfig(
        history_length=history_length,
        seed=73,
        batch_size=2,
        stage1_epochs=2,
        stage2_epochs=1,
        learning_rate=1.0e-3,
    )
    ae = _trainer(tmp_path, config=ae_config, manifest=manifest, contract=contract)
    initial_order_sha = ae._data_order_payload()["permutation_sha256"]
    ae_entries = ae.dataset.entries("train")
    ae_batch = ae.dataset.materialize(
        "train",
        ae_entries,
        history_length=history_length,
        include_target=False,
    )
    before_ae = float(ae._loss(ae_batch).detach())
    event = ae.train_one_step()
    after_ae = float(ae._loss(ae_batch).detach())
    assert event["gradient_audit"]["active_parameter_tensors"] > 0
    assert after_ae < before_ae
    assert ae.dataset.materialized_target_rows["train"] == 0

    height_config = H0bPretrainConfig(
        history_length=history_length,
        seed=73,
        batch_size=2,
        stage1_epochs=0,
        stage2_epochs=2,
        learning_rate=1.0e-3,
    )
    height = _trainer(tmp_path, config=height_config, manifest=manifest, contract=contract)
    height_entries = height.dataset.entries("train")
    height_batch = height.dataset.materialize(
        "train",
        height_entries,
        history_length=history_length,
        include_target=True,
    )
    before_height = float(height._loss(height_batch).detach())
    height.train_one_step()
    after_height = float(height._loss(height_batch).detach())
    assert after_height < before_height

    other_k = 5 if history_length == 1 else 1
    other = _trainer(
        tmp_path,
        config=H0bPretrainConfig(
            history_length=other_k,
            seed=73,
            batch_size=2,
            stage1_epochs=2,
            stage2_epochs=1,
        ),
        manifest=manifest,
        contract=contract,
    )
    assert initial_order_sha == other._data_order_payload()["permutation_sha256"]


def test_resume_is_exact_and_test_samples_are_never_materialized(tmp_path: Path) -> None:
    """Interrupted CPU training is exact and never materializes test samples."""
    manifest, contract = _dataset(tmp_path)
    config = H0bPretrainConfig(
        history_length=1,
        seed=91,
        batch_size=1,
        stage1_epochs=1,
        stage2_epochs=1,
        learning_rate=3.0e-4,
    )
    uninterrupted = _trainer(tmp_path, config=config, manifest=manifest, contract=contract)
    uninterrupted.run()

    interrupted = _trainer(tmp_path, config=config, manifest=manifest, contract=contract)
    interrupted.run(max_steps=1)
    checkpoint_path = tmp_path / "resume-step-1.pt"
    checkpoint_receipt = interrupted.save_checkpoint(checkpoint_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        interrupted.save_checkpoint(checkpoint_path)
    with pytest.raises(ValueError, match="file SHA"):
        load_h0b_pretrain_checkpoint(
            checkpoint_path,
            expected_file_sha256="0" * 64,
        )
    resumed = _trainer(
        tmp_path,
        config=config,
        manifest=manifest,
        contract=contract,
        resume=load_h0b_pretrain_checkpoint(
            checkpoint_path,
            expected_file_sha256=checkpoint_receipt["file_sha256"],
        ),
    )
    resumed.run()

    assert resumed.stage == "complete"
    assert resumed.global_step == uninterrupted.global_step
    assert resumed.epoch_metrics == uninterrupted.epoch_metrics
    for name, expected in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0.0, atol=0.0)
    for name, expected in uninterrupted.autoencoder_head.state_dict().items():
        torch.testing.assert_close(
            resumed.autoencoder_head.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    audit = resumed.dataset.access_audit()
    assert audit["manifest_integrity_validation_rehashed_all_splits"] is True
    assert audit["materialized_batches"]["test"] == 0
    assert audit["materialized_anchor_rows"]["test"] == 0
    assert audit["materialized_target_rows"]["test"] == 0
    assert audit["cached_sample_shard_counts"]["test"] == 0


def test_tamper_formal_gate_and_strict_resume_rejected(tmp_path: Path) -> None:
    """Dataset, target, resume, and formal-label mutations all fail closed."""
    manifest, contract = _dataset(tmp_path)
    config = H0bPretrainConfig(history_length=1, batch_size=2, stage1_epochs=1, stage2_epochs=1)
    with pytest.raises(ValueError, match="formal_400k"):
        H0bOfflinePretrainer(
            config=config,
            dataset_root=tmp_path,
            manifest=manifest,
            target_contract=contract,
            require_formal_400k=True,
        )

    changed_contract = copy.deepcopy(contract)
    changed_contract["height_sign"] = "positive_down"
    with pytest.raises(ValueError, match="target contract payload SHA"):
        _trainer(
            tmp_path,
            config=config,
            manifest=manifest,
            contract=changed_contract,
        )

    trainer = _trainer(tmp_path, config=config, manifest=manifest, contract=contract)
    trainer.run(max_steps=1)
    checkpoint = trainer.checkpoint()
    tampered_checkpoint = copy.deepcopy(checkpoint)
    tampered_checkpoint["bindings"]["collector_receipt_payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="payload SHA"):
        _trainer(
            tmp_path,
            config=config,
            manifest=manifest,
            contract=contract,
            resume=tampered_checkpoint,
        )

    shard_path = tmp_path / "train/part-000.pt"
    with shard_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match=r"manifest differs|file SHA"):
        _trainer(tmp_path, config=config, manifest=manifest, contract=contract)


def test_complete_export_is_loadable_create_only_and_has_separate_receipt(
    tmp_path: Path,
) -> None:
    """Final output is deploy-loadable and separately hash-attested."""
    manifest, contract = _dataset(tmp_path)
    trainer = _trainer(
        tmp_path,
        config=H0bPretrainConfig(
            history_length=1,
            batch_size=2,
            stage1_epochs=0,
            stage2_epochs=1,
        ),
        manifest=manifest,
        contract=contract,
    )
    trainer.run()
    export_path = tmp_path / "deploy/frozen_reconstructor.pt"
    receipt_path = tmp_path / "deploy/frozen_reconstructor.receipt.json"
    export = trainer.export_frozen_reconstructor(export_path, receipt_path=receipt_path)
    loaded_payload = torch.load(export_path, map_location="cpu", weights_only=True)
    restored = load_frozen_reconstructor_checkpoint(loaded_payload)
    production_restored, production_receipt = load_frozen_reconstructor_artifact(
        export_path,
        expected_file_sha256=export["checkpoint"]["file_sha256"],
        receipt_path=receipt_path,
    )
    assert restored.history_length == 1
    assert production_restored.history_length == 1
    assert production_receipt == export["receipt_payload"]
    assert set(loaded_payload) == {
        "schema",
        "schema_sha256",
        "state_dict",
        "state_sha256",
    }
    assert export["receipt_payload"]["checkpoint_file_sha256"] == (export["checkpoint"]["file_sha256"])
    assert export["receipt_payload"]["checkpoint_schema_sha256"] == (loaded_payload["schema_sha256"])
    assert export["receipt_payload"]["checkpoint_state_sha256"] == (loaded_payload["state_sha256"])
    assert export["receipt_payload"]["autoencoder_head_in_deploy_checkpoint"] is False
    receipt_without_digest = dict(export["receipt_payload"])
    receipt_digest = receipt_without_digest.pop("receipt_payload_sha256")
    assert receipt_digest == _canonical_sha(receipt_without_digest)
    assert json.loads(receipt_path.read_text()) == export["receipt_payload"]
    with pytest.raises(FileExistsError, match="overwrite"):
        trainer.export_frozen_reconstructor(export_path, receipt_path=tmp_path / "deploy/second.json")


def test_cli_runs_bounded_cpu_step_and_writes_create_only_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI is bounded, CPU-labelled, and emits an immutable checkpoint."""
    manifest, contract = _dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    contract_path = tmp_path / "target-contract.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    checkpoint_path = tmp_path / "cli-step-1.pt"
    arguments = [
        "--dataset-root",
        str(tmp_path),
        "--manifest",
        str(manifest_path),
        "--target-contract",
        str(contract_path),
        "--checkpoint",
        str(checkpoint_path),
        "--history-length",
        "1",
        "--batch-size",
        "1",
        "--stage1-epochs",
        "1",
        "--stage2-epochs",
        "1",
        "--max-steps",
        "1",
    ]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["device"] == "cpu"
    assert output["training_ready_claim"] is False
    assert output["steps_executed"] == 1
    assert checkpoint_path.is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        main(arguments)
