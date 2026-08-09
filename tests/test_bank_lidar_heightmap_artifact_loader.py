# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import hashlib
import json
import torch
from collections.abc import Callable
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.models.prop_mlp_elevation_fusion_model import (
    PropMLPElevationFusionModel,
)
from rsl_rl.modules.bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    create_frozen_reconstructor_checkpoint,
    freeze_reconstructor,
    load_frozen_reconstructor_artifact,
)


def _contract() -> dict:
    return {
        "schema_version": 1,
        "target_definition": "synthetic_test_height_target",
        "height_unit": "metre",
        "height_sign": "synthetic_test_positive_direction",
        "grid_shape": [28, 20],
        "grid_axis_order": ["synthetic_row", "synthetic_column"],
        "grid_axis_directions": [
            "synthetic_row_direction",
            "synthetic_column_direction",
        ],
        "flatten_order": "C_contiguous_row_major",
        "coordinate_frame": "synthetic_test_frame",
        "origin": "synthetic_test_origin",
        "resolution_m": 0.05,
        "unknown_cell_policy": "synthetic_dense_reconstruction",
        "contract_source_sha256": "a" * 64,
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_receipt_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_receipt(path: Path, receipt: dict) -> None:
    path.write_bytes(_canonical_receipt_bytes(receipt))


def _artifact(
    root: Path,
    *,
    history_length: int,
    seed: int,
) -> tuple[Path, str, Path, dict, dict]:
    root.mkdir(parents=True)
    torch.manual_seed(seed)
    model = BankLidarHeightmapReconstructor(
        history_length=history_length,
        target_contract=_contract(),
    )
    freeze_audit = freeze_reconstructor(model)
    checkpoint = create_frozen_reconstructor_checkpoint(model)
    checkpoint_path = root / "frozen_reconstructor.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    receipt = {
        "schema_version": 1,
        "classification": "h0b_frozen_reconstructor_export_v1",
        "training_ready_claim": False,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_size_bytes": len(checkpoint_bytes),
        "checkpoint_file_sha256": checkpoint_sha256,
        "frozen_payload_sha256": _json_sha256({
            "schema_sha256": checkpoint["schema_sha256"],
            "state_sha256": checkpoint["state_sha256"],
        }),
        "checkpoint_schema_sha256": checkpoint["schema_sha256"],
        "checkpoint_state_sha256": checkpoint["state_sha256"],
        "history_length": history_length,
        "dataset_manifest_payload_sha256": "b" * 64,
        "target_contract_payload_sha256": _json_sha256(_contract()),
        "target_contract_source_sha256": _contract()["contract_source_sha256"],
        "source_commits": {
            "lab_pro": "1" * 40,
            "rsl_rl": "2" * 40,
            "isaaclab": "3" * 40,
        },
        "freeze_audit": freeze_audit,
        "autoencoder_head_in_deploy_checkpoint": False,
    }
    receipt["receipt_payload_sha256"] = _json_sha256(receipt)
    receipt_path = root / "frozen_reconstructor.receipt.json"
    _write_receipt(receipt_path, receipt)
    return checkpoint_path, checkpoint_sha256, receipt_path, checkpoint, receipt


def _observations(history_length: int) -> TensorDict:
    torch.manual_seed(7100 + history_length)
    range_m = torch.rand(2, history_length, 16, 96) * 5.8 + 0.1
    valid = torch.rand_like(range_m) > 0.2
    ray = torch.stack(
        (torch.where(valid, range_m, torch.zeros_like(range_m)), valid.float()),
        dim=2,
    )
    return TensorDict(
        {"policy": torch.randn(2, 96), "ray_policy": ray},
        batch_size=[2],
    )


def _production_model(
    observations: TensorDict,
    *,
    history_length: int,
    checkpoint_path: Path | None,
    checkpoint_sha256: str | None,
    receipt_path: Path | None,
    internal_checkpoint: dict | None = None,
) -> PropMLPElevationFusionModel:
    return PropMLPElevationFusionModel(
        obs=observations,
        obs_groups={"actor": ["policy", "ray_policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=[64],
        distribution_cfg=None,
        elevation_encoder_type="bank_lidar_heightmap",
        ray_time_set="ray_policy",
        ray_time_history_length=history_length,
        ray_time_spatial_size=(16, 96),
        vision_feature_dim=32,
        prop_feature_dim=32,
        prop_hidden_dims=[64],
        bank_heightmap_target_contract=_contract(),
        bank_downstream_heightmap_contract=_contract(),
        bank_reconstructor_checkpoint_path=(None if checkpoint_path is None else str(checkpoint_path)),
        bank_reconstructor_checkpoint_expected_file_sha256=checkpoint_sha256,
        bank_reconstructor_receipt_path=(None if receipt_path is None else str(receipt_path)),
        _bank_reconstructor_checkpoint_for_testing=internal_checkpoint,
    )


@pytest.mark.parametrize("history_length", [1, 5])
def test_receipt_backed_loader_and_production_model_succeed(
    tmp_path: Path,
    history_length: int,
) -> None:
    """Load both supported history lengths through the production triplet."""
    checkpoint_path, digest, receipt_path, _, receipt = _artifact(
        tmp_path / f"k{history_length}",
        history_length=history_length,
        seed=7200 + history_length,
    )
    loaded, validated_receipt = load_frozen_reconstructor_artifact(
        checkpoint_path,
        expected_file_sha256=digest,
        receipt_path=receipt_path,
    )
    assert loaded.history_length == history_length
    assert validated_receipt == receipt
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

    observations = _observations(history_length)
    model = _production_model(
        observations,
        history_length=history_length,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=digest,
        receipt_path=receipt_path,
    ).eval()
    assert model.bank_reconstructor_artifact_receipt == receipt
    assert model(observations).shape == (2, 29)


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("checkpoint", "complete.*triplet"),
        ("sha256", "complete.*triplet"),
        ("receipt", "complete.*triplet"),
    ],
)
def test_production_config_requires_exact_triplet(
    tmp_path: Path,
    missing_field: str,
    message: str,
) -> None:
    """Reject every partial production artifact configuration."""
    checkpoint_path, digest, receipt_path, _, _ = _artifact(
        tmp_path / missing_field,
        history_length=1,
        seed=7300,
    )
    values = {
        "checkpoint": checkpoint_path,
        "sha256": digest,
        "receipt": receipt_path,
    }
    values[missing_field] = None
    with pytest.raises(ValueError, match=message):
        _production_model(
            _observations(1),
            history_length=1,
            checkpoint_path=values["checkpoint"],
            checkpoint_sha256=values["sha256"],
            receipt_path=values["receipt"],
        )


def test_production_config_rejects_internal_mapping_mix(tmp_path: Path) -> None:
    """Keep the internal mapping hook disjoint from production loading."""
    checkpoint_path, digest, receipt_path, checkpoint, _ = _artifact(
        tmp_path / "mixed",
        history_length=1,
        seed=7400,
    )
    with pytest.raises(ValueError, match="mixed production"):
        _production_model(
            _observations(1),
            history_length=1,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=digest,
            receipt_path=receipt_path,
            internal_checkpoint=checkpoint,
        )


def test_loader_rejects_symlink_files(tmp_path: Path) -> None:
    """Reject a symbolic link used for either retained input file."""
    checkpoint_path, digest, receipt_path, _, _ = _artifact(
        tmp_path / "real",
        history_length=1,
        seed=7500,
    )
    checkpoint_link = tmp_path / "checkpoint-link.pt"
    receipt_link = tmp_path / "receipt-link.json"
    checkpoint_link.symlink_to(checkpoint_path)
    receipt_link.symlink_to(receipt_path)
    with pytest.raises(RuntimeError, match="safely open"):
        load_frozen_reconstructor_artifact(
            checkpoint_link,
            expected_file_sha256=digest,
            receipt_path=receipt_path,
        )
    with pytest.raises(RuntimeError, match="safely open"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=digest,
            receipt_path=receipt_link,
        )


def test_loader_rejects_checkpoint_tamper_and_wrong_expected_hash(
    tmp_path: Path,
) -> None:
    """Use the configured checkpoint digest as the byte-level trust root."""
    checkpoint_path, digest, receipt_path, _, _ = _artifact(
        tmp_path / "tamper",
        history_length=1,
        seed=7600,
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256="f" * 64,
            receipt_path=receipt_path,
        )
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=digest,
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"extra": True}), "receipt keys changed"),
        (
            lambda value: value.update({"checkpoint_file_sha256": "f" * 64}),
            "file SHA-256 differs from receipt",
        ),
        (
            lambda value: value.update({"checkpoint_schema_sha256": "f" * 64}),
            "schema/state SHA-256 differs",
        ),
        (
            lambda value: value.update({"checkpoint_state_sha256": "f" * 64}),
            "schema/state SHA-256 differs",
        ),
        (
            lambda value: value.update({"frozen_payload_sha256": "f" * 64}),
            "payload SHA-256 differs",
        ),
        (
            lambda value: value.update({"history_length": 5}),
            "history_length differs",
        ),
        (
            lambda value: value.update({"target_contract_payload_sha256": "f" * 64}),
            "target contract payload SHA-256 differs",
        ),
        (
            lambda value: value.update({"target_contract_source_sha256": "f" * 64}),
            "target contract source SHA-256 differs",
        ),
        (
            lambda value: value["source_commits"].update({"rsl_rl": "bad"}),
            "source commits must",
        ),
        (
            lambda value: value["freeze_audit"].update({"parameter_count": 1}),
            "freeze audit is inconsistent",
        ),
        (
            lambda value: value.update({"autoencoder_head_in_deploy_checkpoint": True}),
            "Autoencoder head is forbidden",
        ),
    ],
)
def test_loader_rejects_receipt_tamper(
    tmp_path: Path,
    mutation: Callable[[dict], object],
    message: str,
) -> None:
    """Reject independently rehashed receipts with inconsistent semantics."""
    checkpoint_path, digest, receipt_path, _, receipt = _artifact(
        tmp_path / str(abs(hash(message))),
        history_length=1,
        seed=7700,
    )
    changed = copy.deepcopy(receipt)
    mutation(changed)
    changed_without_digest = dict(changed)
    changed_without_digest.pop("receipt_payload_sha256", None)
    changed["receipt_payload_sha256"] = _json_sha256(changed_without_digest)
    _write_receipt(receipt_path, changed)
    with pytest.raises(ValueError, match=message):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=digest,
            receipt_path=receipt_path,
        )


def test_loader_rejects_noncanonical_and_duplicate_key_receipts(
    tmp_path: Path,
) -> None:
    """Require one duplicate-free canonical JSON receipt object."""
    checkpoint_path, digest, receipt_path, _, receipt = _artifact(
        tmp_path / "json",
        history_length=1,
        seed=7800,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="ascii")
    with pytest.raises(ValueError, match="not canonical"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=digest,
            receipt_path=receipt_path,
        )
    receipt_path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="ascii")
    with pytest.raises(ValueError, match="repeats key"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=digest,
            receipt_path=receipt_path,
        )


def test_loader_rejects_valid_looking_source_tamper_without_fresh_digest(
    tmp_path: Path,
) -> None:
    """Detect a valid-shaped provenance edit through the receipt payload digest."""
    checkpoint_path, digest, receipt_path, _, receipt = _artifact(
        tmp_path / "source-tamper",
        history_length=1,
        seed=7850,
    )
    receipt["source_commits"]["rsl_rl"] = "e" * 40
    _write_receipt(receipt_path, receipt)
    with pytest.raises(ValueError, match="receipt payload SHA-256 mismatch"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=digest,
            receipt_path=receipt_path,
        )


def test_loader_rejects_cross_spliced_receipt(tmp_path: Path) -> None:
    """Reject a valid receipt taken from a different checkpoint export."""
    checkpoint_a, digest_a, _, _, _ = _artifact(tmp_path / "a", history_length=1, seed=7901)
    _, _, receipt_b, _, _ = _artifact(tmp_path / "b", history_length=1, seed=7902)
    with pytest.raises(ValueError, match="file SHA-256 differs from receipt"):
        load_frozen_reconstructor_artifact(
            checkpoint_a,
            expected_file_sha256=digest_a,
            receipt_path=receipt_b,
        )


def test_loader_rejects_extra_checkpoint_key_and_autoencoder_head(
    tmp_path: Path,
) -> None:
    """Reject a deployment mapping contaminated by the pretraining head."""
    checkpoint_path, _, receipt_path, checkpoint, receipt = _artifact(
        tmp_path / "extra-key",
        history_length=1,
        seed=8000,
    )
    checkpoint["autoencoder_head_state_dict"] = {"weight": torch.ones(1)}
    torch.save(checkpoint, checkpoint_path)
    changed_bytes = checkpoint_path.read_bytes()
    changed_digest = hashlib.sha256(changed_bytes).hexdigest()
    receipt["checkpoint_file_sha256"] = changed_digest
    receipt["checkpoint_file_size_bytes"] = len(changed_bytes)
    _write_receipt(receipt_path, receipt)
    with pytest.raises(ValueError, match="checkpoint keys changed"):
        load_frozen_reconstructor_artifact(
            checkpoint_path,
            expected_file_sha256=changed_digest,
            receipt_path=receipt_path,
        )


def test_production_model_rejects_wrong_k_artifact(tmp_path: Path) -> None:
    """Bind the loaded artifact history length to the actor configuration."""
    checkpoint_path, digest, receipt_path, _, _ = _artifact(
        tmp_path / "k5",
        history_length=5,
        seed=8100,
    )
    with pytest.raises(ValueError, match="history/target contract mismatch"):
        _production_model(
            _observations(1),
            history_length=1,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=digest,
            receipt_path=receipt_path,
        )
