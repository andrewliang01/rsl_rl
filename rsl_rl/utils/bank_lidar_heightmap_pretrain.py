# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auditable, CPU-only two-stage offline pretraining for the H0b baseline.

This module is intentionally separate from PPO, Gym, and Isaac Lab.  It first
trains the shared spherical frame encoder with the no-skip autoencoder head,
then trains the deployable LiDAR-to-height reconstructor with supervised local
height targets.  Dataset integrity is delegated to the frozen packed-shard
manifest contract before any sample is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import torch
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsl_rl.modules.bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    SphericalAutoencoderPretrainHead,
    create_frozen_reconstructor_checkpoint,
    freeze_reconstructor,
    spherical_valid_bce,
    supervised_height_valid_mse,
    validate_manifest_target_contract_binding,
    valid_masked_range_mse,
)
from rsl_rl.utils.bank_lidar_heightmap_dataset import (
    H0B_PACKET_SPATIAL_SIZE,
    H0B_TARGET_SPATIAL_SIZE,
    load_packed_h0b_shard,
    unpack_bool_mask,
    validate_h0b_dataset_manifest,
)

H0B_PRETRAIN_CHECKPOINT_SCHEMA_VERSION = 1
_STAGE_AE = "stage1_ae"
_STAGE_HEIGHT = "stage2_height"
_STAGE_COMPLETE = "complete"
_STAGES = (_STAGE_AE, _STAGE_HEIGHT, _STAGE_COMPLETE)
_HEX64 = frozenset("0123456789abcdef")
_CHECKPOINT_KEYS = {
    "schema_version",
    "config",
    "bindings",
    "model_state_dict",
    "autoencoder_head_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "progress",
    "data_order",
    "rng_state",
    "epoch_metrics",
    "access_audit",
    "payload_sha256",
}


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json_mapping(path: str | Path, *, description: str) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain one JSON object.")
    return dict(value)


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX64 for character in value):
        raise ValueError(f"{name} must be lowercase 64-hex.")
    return value


def _structured_sha256(value: Any) -> str:
    """Hash nested checkpoint primitives and tensor bytes without pickle."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"none;")
        elif type(item) is bool:
            digest.update(b"bool:1;" if item else b"bool:0;")
        elif type(item) is int:
            digest.update(f"int:{item};".encode("ascii"))
        elif type(item) is float:
            digest.update(b"float:" + struct.pack(">d", item) + b";")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(f"str:{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
            digest.update(b";")
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(b"tensor:")
            update(str(tensor.dtype))
            update(list(tensor.shape))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
            digest.update(b";")
        elif isinstance(item, Mapping):
            digest.update(f"map:{len(item)}:".encode("ascii"))
            ordered_keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            for key in ordered_keys:
                update(key)
                update(item[key])
            digest.update(b";")
        elif isinstance(item, (list, tuple)):
            digest.update(f"seq:{len(item)}:".encode("ascii"))
            for element in item:
                update(element)
            digest.update(b";")
        else:
            raise TypeError(
                f"Checkpoint integrity supports only tensors and primitive containers; received {type(item).__name__}."
            )

    update(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class H0bPretrainConfig:
    """Frozen hyperparameters shared by a run and every strict resume."""

    history_length: int
    seed: int = 42
    batch_size: int = 32
    stage1_epochs: int = 10
    stage2_epochs: int = 20
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    range_loss_weight: float = 1.0
    validity_loss_weight: float = 1.0
    max_grad_norm: float = 10.0

    def normalized(self) -> dict[str, Any]:
        """Validate and return a JSON-stable hyperparameter payload."""
        payload = asdict(self)
        if self.history_length not in (1, 5):
            raise ValueError("history_length must be exactly 1 or 5.")
        for field in ("seed", "batch_size", "stage1_epochs", "stage2_epochs"):
            value = payload[field]
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer.")
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive.")
        if self.stage1_epochs == 0 and self.stage2_epochs == 0:
            raise ValueError("At least one pretraining stage must have an epoch.")
        if type(self.plateau_patience) is not int or self.plateau_patience < 0:
            raise ValueError("plateau_patience must be a non-negative integer.")
        for field in (
            "learning_rate",
            "weight_decay",
            "range_loss_weight",
            "validity_loss_weight",
            "max_grad_norm",
        ):
            value = payload[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field} must be finite and non-negative.")
            payload[field] = float(value)
        if self.learning_rate == 0.0 or self.max_grad_norm == 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive.")
        if (
            isinstance(self.plateau_factor, bool)
            or not isinstance(self.plateau_factor, (int, float))
            or not math.isfinite(float(self.plateau_factor))
            or not 0.0 < float(self.plateau_factor) < 1.0
        ):
            raise ValueError("plateau_factor must lie strictly in (0,1).")
        payload["plateau_factor"] = float(self.plateau_factor)
        return payload


class _ValidatedPackedDataset:
    """Lazy train/val materializer bound to one fully validated manifest."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest: Mapping[str, Any],
        target_contract: Mapping[str, Any],
        require_formal_400k: bool,
    ) -> None:
        self.root = Path(dataset_root).resolve(strict=True)
        self.manifest = dict(manifest)
        expected_target_digest = _validate_sha256(
            self.manifest.get("target_contract_payload_sha256"),
            name="manifest target_contract_payload_sha256",
        )
        self.target_contract = validate_manifest_target_contract_binding(
            target_contract,
            expected_target_digest,
        )
        self.manifest_audit = validate_h0b_dataset_manifest(
            self.manifest,
            self.root,
            require_formal_400k=require_formal_400k,
        )
        self._records = {
            split: [dict(record) for record in self.manifest["shards"] if record["split"] == split]
            for split in ("train", "val")
        }
        if not self._records["train"] or not self._records["val"]:
            raise ValueError("H0b pretraining requires non-empty train and val shards.")
        self._shard_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._split_entries: dict[str, list[tuple[int, int]]] = {}
        for split in ("train", "val"):
            entries: list[tuple[int, int]] = []
            for shard_index, record in enumerate(self._records[split]):
                entries.extend((shard_index, row) for row in range(int(record["num_anchors"])))
            self._split_entries[split] = entries
        self.materialized_batches = {"train": 0, "val": 0, "test": 0}
        self.materialized_anchor_rows = {"train": 0, "val": 0, "test": 0}
        self.materialized_target_rows = {"train": 0, "val": 0, "test": 0}

    @property
    def bindings(self) -> dict[str, Any]:
        return {
            "dataset_manifest_payload_sha256": self.manifest["manifest_payload_sha256"],
            "target_contract_payload_sha256": self.manifest["target_contract_payload_sha256"],
            "target_contract_source_sha256": self.target_contract["contract_source_sha256"],
            "collector_receipt_payload_sha256": self.manifest["collector_receipt_payload_sha256"],
            "source_commits": dict(self.manifest["source_commits"]),
            "train_shard_payload_sha256": [record["payload_sha256"] for record in self._records["train"]],
            "val_shard_payload_sha256": [record["payload_sha256"] for record in self._records["val"]],
            "target_contract": dict(self.target_contract),
            "formal_400k_validated": self.manifest_audit["formal_400k_validated"],
        }

    def entries(self, split: str) -> list[tuple[int, int]]:
        if split not in ("train", "val"):
            raise ValueError("Pretraining may materialize only train or val.")
        return list(self._split_entries[split])

    def _load_shard(self, split: str, shard_index: int) -> dict[str, Any]:
        key = (split, shard_index)
        if key not in self._shard_cache:
            record = self._records[split][shard_index]
            shard, _ = load_packed_h0b_shard(
                self.root / record["relative_path"],
                expected_file_sha256=record["file_sha256"],
                expected_payload_sha256=record["payload_sha256"],
            )
            self._shard_cache[key] = shard
        return self._shard_cache[key]

    def materialize(
        self,
        split: str,
        entries: Sequence[tuple[int, int]],
        *,
        history_length: int,
        include_target: bool,
    ) -> dict[str, torch.Tensor]:
        if split not in ("train", "val"):
            raise ValueError("Pretraining may materialize only train or val.")
        if history_length not in (1, 5) or not entries:
            raise ValueError("A non-empty K=1/K=5 pretraining batch is required.")
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_row, (shard_index, anchor_row) in enumerate(entries):
            grouped[int(shard_index)].append((output_row, int(anchor_row)))

        batch_size = len(entries)
        ray_history = torch.empty(
            batch_size,
            history_length,
            2,
            *H0B_PACKET_SPATIAL_SIZE,
            dtype=torch.float16,
        )
        anchor_id = torch.empty(batch_size, dtype=torch.int64)
        target_height = torch.empty(batch_size, 1, *H0B_TARGET_SPATIAL_SIZE) if include_target else None
        target_valid = (
            torch.empty(
                batch_size,
                1,
                *H0B_TARGET_SPATIAL_SIZE,
                dtype=torch.bool,
            )
            if include_target
            else None
        )
        for shard_index, output_and_anchor_rows in grouped.items():
            output_rows = torch.tensor([item[0] for item in output_and_anchor_rows], dtype=torch.int64)
            anchor_rows = torch.tensor([item[1] for item in output_and_anchor_rows], dtype=torch.int64)
            shard = self._load_shard(split, shard_index)
            frame_ids = shard["anchor_frame_ids"][anchor_rows]
            if history_length == 1:
                frame_ids = frame_ids[:, -1:]
            packet_range = shard["packet_range_m"][frame_ids]
            packet_valid = unpack_bool_mask(
                shard["packet_valid_bits"][frame_ids],
                spatial_size=H0B_PACKET_SPATIAL_SIZE,
            )
            ray_history[output_rows] = torch.stack((packet_range, packet_valid.to(torch.float16)), dim=2)
            anchor_id[output_rows] = shard["anchor_id"][anchor_rows]
            if include_target:
                assert target_height is not None and target_valid is not None
                target_height[output_rows] = shard["target_height_m"][anchor_rows].to(torch.float32).unsqueeze(1)
                target_valid[output_rows] = unpack_bool_mask(
                    shard["target_valid_bits"][anchor_rows],
                    spatial_size=H0B_TARGET_SPATIAL_SIZE,
                ).unsqueeze(1)

        self.materialized_batches[split] += 1
        self.materialized_anchor_rows[split] += batch_size
        if include_target:
            self.materialized_target_rows[split] += batch_size
        result = {"anchor_id": anchor_id, "ray_history": ray_history}
        if include_target:
            assert target_height is not None and target_valid is not None
            result["target_height_m"] = target_height
            result["target_valid"] = target_valid
        return result

    def access_audit(self) -> dict[str, Any]:
        return {
            "manifest_integrity_validation_rehashed_all_splits": True,
            "test_sample_materialization_forbidden": True,
            "materialized_batches": dict(self.materialized_batches),
            "materialized_anchor_rows": dict(self.materialized_anchor_rows),
            "materialized_target_rows": dict(self.materialized_target_rows),
            "cached_sample_shard_counts": {
                split: sum(1 for key in self._shard_cache if key[0] == split) for split in ("train", "val", "test")
            },
        }


def _data_order_seed(seed: int, stage: str, epoch: int) -> int:
    stage_offset = {_STAGE_AE: 0x1A2B3C, _STAGE_HEIGHT: 0x4D5E6F}[stage]
    return (int(seed) + stage_offset + 1_000_003 * int(epoch)) % (2**63 - 1)


def _epoch_order(
    num_rows: int,
    *,
    seed: int,
    stage: str,
    epoch: int,
) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_data_order_seed(seed, stage, epoch))
    return torch.randperm(num_rows, generator=generator).tolist()


def _order_sha256(order: Sequence[int]) -> str:
    tensor = torch.tensor(order, dtype=torch.int64).contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _python_rng_payload() -> dict[str, Any]:
    version, state, gaussian = random.getstate()
    return {"version": version, "state": list(state), "gaussian": gaussian}


def _restore_python_rng(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "version",
        "state",
        "gaussian",
    }:
        raise ValueError("Python RNG checkpoint schema changed.")
    random.setstate((int(payload["version"]), tuple(payload["state"]), payload["gaussian"]))


def _atomic_create_torch(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination),
        "file_size_bytes": destination.stat().st_size,
        "file_sha256": digest,
    }


def _atomic_create_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary receipt already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination),
        "file_size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


class H0bOfflinePretrainer:
    """Stateful, resumable CPU trainer with explicit split boundaries."""

    def __init__(
        self,
        *,
        config: H0bPretrainConfig,
        dataset_root: str | Path,
        manifest: Mapping[str, Any],
        target_contract: Mapping[str, Any],
        require_formal_400k: bool = False,
        resume_checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate immutable inputs and initialize or strictly resume on CPU."""
        self.config = config
        self.config_payload = config.normalized()
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.dataset = _ValidatedPackedDataset(
            dataset_root=dataset_root,
            manifest=manifest,
            target_contract=target_contract,
            require_formal_400k=require_formal_400k,
        )
        self.model = BankLidarHeightmapReconstructor(
            history_length=config.history_length,
            target_contract=self.dataset.target_contract,
        ).cpu()
        self.autoencoder_head = SphericalAutoencoderPretrainHead().cpu()
        self.optimizer = torch.optim.AdamW(
            [*self.model.parameters(), *self.autoencoder_head.parameters()],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._new_scheduler()
        self.stage = self._first_stage()
        self.epoch = 0
        self.row_cursor = 0
        self.global_step = 0
        self.epoch_metrics: list[dict[str, Any]] = []
        if resume_checkpoint is not None:
            self._restore(resume_checkpoint)
        self._set_stage_trainability()

    def _new_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Create one objective-local plateau scheduler on the shared optimizer."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.plateau_factor,
            patience=self.config.plateau_patience,
        )

    def _first_stage(self) -> str:
        return _STAGE_AE if self.config.stage1_epochs > 0 else _STAGE_HEIGHT

    def _stage_epoch_budget(self) -> int:
        if self.stage == _STAGE_AE:
            return self.config.stage1_epochs
        if self.stage == _STAGE_HEIGHT:
            return self.config.stage2_epochs
        return 0

    def _set_stage_trainability(self) -> None:
        self.model.requires_grad_(False)
        self.autoencoder_head.requires_grad_(False)
        if self.stage == _STAGE_AE:
            self.model.frame_encoder.requires_grad_(True)
            self.autoencoder_head.requires_grad_(True)
            self.model.train()
            self.autoencoder_head.train()
        elif self.stage == _STAGE_HEIGHT:
            self.model.requires_grad_(True)
            self.model.train()
            self.autoencoder_head.eval()
        elif self.stage == _STAGE_COMPLETE:
            self.model.eval()
            self.autoencoder_head.eval()
        else:
            raise RuntimeError("Unknown H0b pretraining stage.")

    def _current_order(self) -> list[int]:
        if self.stage == _STAGE_COMPLETE:
            return []
        return _epoch_order(
            len(self.dataset.entries("train")),
            seed=self.config.seed,
            stage=self.stage,
            epoch=self.epoch,
        )

    def _data_order_payload(self) -> dict[str, Any]:
        order = self._current_order()
        return {
            "contract": "same_anchor_rows_for_k1_k5_seeded_randperm_v1",
            "history_length_excluded_from_permutation_seed": True,
            "stage": self.stage,
            "epoch": self.epoch,
            "row_cursor": self.row_cursor,
            "num_train_rows": len(self.dataset.entries("train")),
            "permutation_seed": (
                None if self.stage == _STAGE_COMPLETE else _data_order_seed(self.config.seed, self.stage, self.epoch)
            ),
            "permutation_sha256": (None if self.stage == _STAGE_COMPLETE else _order_sha256(order)),
        }

    def _loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        ray_history = batch["ray_history"]
        if self.stage == _STAGE_AE:
            if "target_height_m" in batch or "target_valid" in batch:
                raise RuntimeError("Stage 1 must not materialize height targets.")
            encoded = self.model.encode_frame_history(ray_history)
            reconstruction = self.autoencoder_head(encoded)
            target_range = ray_history[:, :, 0].to(torch.float32)
            target_valid = ray_history[:, :, 1] == 1.0
            return self.config.range_loss_weight * valid_masked_range_mse(
                reconstruction.range_m, target_range, target_valid
            ) + self.config.validity_loss_weight * spherical_valid_bce(reconstruction.valid_logits, target_valid)
        if self.stage == _STAGE_HEIGHT:
            return supervised_height_valid_mse(
                self.model(ray_history),
                batch["target_height_m"],
                batch["target_valid"],
            )
        raise RuntimeError("A completed trainer cannot compute training loss.")

    def _audit_gradients(self, loss: torch.Tensor) -> dict[str, Any]:
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("H0b loss must be one finite scalar.")
        active = [
            (name, parameter)
            for name, parameter in (
                [(f"model.{name}", parameter) for name, parameter in self.model.named_parameters()]
                + [
                    (f"autoencoder_head.{name}", parameter)
                    for name, parameter in self.autoencoder_head.named_parameters()
                ]
            )
            if parameter.requires_grad
        ]
        missing = [name for name, parameter in active if parameter.grad is None]
        if missing:
            raise RuntimeError(f"Active H0b parameters lack gradients: {missing}")
        if any(not bool(torch.isfinite(parameter.grad).all()) for _, parameter in active):
            raise FloatingPointError("H0b gradient contains NaN or infinity.")
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in active],
            max_norm=self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        return {
            "active_parameter_tensors": len(active),
            "preclip_grad_norm": float(norm),
            "max_grad_norm": self.config.max_grad_norm,
        }

    def train_one_step(self) -> dict[str, Any]:
        """Run one audited train-split optimizer step."""
        if self.stage == _STAGE_COMPLETE:
            raise RuntimeError("H0b pretraining is already complete.")
        order = self._current_order()
        end = min(self.row_cursor + self.config.batch_size, len(order))
        selected = order[self.row_cursor : end]
        entries = self.dataset.entries("train")
        batch = self.dataset.materialize(
            "train",
            [entries[index] for index in selected],
            history_length=self.config.history_length,
            include_target=self.stage == _STAGE_HEIGHT,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(batch)
        loss.backward()
        gradient_audit = self._audit_gradients(loss)
        self.optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()) or any(
            not bool(torch.isfinite(parameter).all()) for parameter in self.autoencoder_head.parameters()
        ):
            raise FloatingPointError("H0b optimizer produced a non-finite parameter.")
        self.row_cursor = end
        self.global_step += 1
        result = {
            "stage": self.stage,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss": float(loss.detach()),
            "anchor_ids": batch["anchor_id"].tolist(),
            "gradient_audit": gradient_audit,
        }
        if self.row_cursor == len(order):
            result["epoch_end"] = self._finish_epoch()
        return result

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Evaluate the active stage on val without backward or optimizer use."""
        if self.stage == _STAGE_COMPLETE:
            raise RuntimeError("A completed trainer has no active validation stage.")
        model_training = self.model.training
        head_training = self.autoencoder_head.training
        self.model.eval()
        self.autoencoder_head.eval()
        entries = self.dataset.entries("val")
        weighted_loss = 0.0
        row_count = 0
        for begin in range(0, len(entries), self.config.batch_size):
            batch_entries = entries[begin : begin + self.config.batch_size]
            batch = self.dataset.materialize(
                "val",
                batch_entries,
                history_length=self.config.history_length,
                include_target=self.stage == _STAGE_HEIGHT,
            )
            loss = self._loss(batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("H0b validation loss must be finite.")
            weighted_loss += float(loss) * len(batch_entries)
            row_count += len(batch_entries)
        self.model.train(model_training)
        self.autoencoder_head.train(head_training)
        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "loss": weighted_loss / row_count,
            "num_rows": row_count,
            "backward": False,
        }

    def _finish_epoch(self) -> dict[str, Any]:
        validation = self.validate()
        self.scheduler.step(validation["loss"])
        record = {
            **validation,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        self.epoch_metrics.append(record)
        self.epoch += 1
        self.row_cursor = 0
        if self.epoch >= self._stage_epoch_budget():
            if self.stage == _STAGE_AE and self.config.stage2_epochs > 0:
                self.stage = _STAGE_HEIGHT
                self.epoch = 0
                # AE and height MSE have incomparable scales; never carry the
                # plateau best/bad-epoch state across objectives.
                self.scheduler = self._new_scheduler()
            else:
                self.stage = _STAGE_COMPLETE
        self._set_stage_trainability()
        return record

    def run(self, *, max_steps: int | None = None) -> list[dict[str, Any]]:
        """Advance deterministically until completion or the optional step bound."""
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError("max_steps must be a non-negative integer or None.")
        events: list[dict[str, Any]] = []
        while self.stage != _STAGE_COMPLETE and (max_steps is None or len(events) < max_steps):
            events.append(self.train_one_step())
        return events

    def checkpoint(self) -> dict[str, Any]:
        """Build an integrity-bound, weights-only-compatible resume payload."""
        payload = {
            "schema_version": H0B_PRETRAIN_CHECKPOINT_SCHEMA_VERSION,
            "config": dict(self.config_payload),
            "bindings": self.dataset.bindings,
            "model_state_dict": {
                name: value.detach().contiguous().cpu().clone() for name, value in self.model.state_dict().items()
            },
            "autoencoder_head_state_dict": {
                name: value.detach().contiguous().cpu().clone()
                for name, value in self.autoencoder_head.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "progress": {
                "stage": self.stage,
                "epoch": self.epoch,
                "row_cursor": self.row_cursor,
                "global_step": self.global_step,
            },
            "data_order": self._data_order_payload(),
            "rng_state": {
                "torch_cpu": torch.get_rng_state().clone(),
                "python": _python_rng_payload(),
                "cuda_state_absent": True,
            },
            "epoch_metrics": list(self.epoch_metrics),
            "access_audit": self.dataset.access_audit(),
        }
        return {**payload, "payload_sha256": _structured_sha256(payload)}

    def save_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Create one immutable resume checkpoint and return its file receipt."""
        return _atomic_create_torch(path, self.checkpoint())

    def _restore(self, checkpoint: Mapping[str, Any]) -> None:
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_KEYS:
            raise ValueError("H0b pretraining checkpoint keys changed.")
        if checkpoint["schema_version"] != H0B_PRETRAIN_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("H0b pretraining checkpoint schema version changed.")
        payload = {key: checkpoint[key] for key in _CHECKPOINT_KEYS - {"payload_sha256"}}
        digest = checkpoint["payload_sha256"]
        if not isinstance(digest, str) or _structured_sha256(payload) != digest:
            raise ValueError("H0b pretraining checkpoint payload SHA-256 mismatch.")
        if checkpoint["config"] != self.config_payload:
            raise ValueError("H0b resume config differs from checkpoint.")
        if checkpoint["bindings"] != self.dataset.bindings:
            raise ValueError("H0b resume dataset/source/target bindings differ.")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.autoencoder_head.load_state_dict(checkpoint["autoencoder_head_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        if not isinstance(progress, Mapping) or set(progress) != {
            "stage",
            "epoch",
            "row_cursor",
            "global_step",
        }:
            raise ValueError("H0b resume progress schema changed.")
        self.stage = progress["stage"]
        self.epoch = progress["epoch"]
        self.row_cursor = progress["row_cursor"]
        self.global_step = progress["global_step"]
        if (
            self.stage not in _STAGES
            or type(self.epoch) is not int
            or type(self.row_cursor) is not int
            or type(self.global_step) is not int
            or min(self.epoch, self.row_cursor, self.global_step) < 0
        ):
            raise ValueError("H0b resume progress values are invalid.")
        if self.stage == _STAGE_COMPLETE:
            final_epoch_budget = (
                self.config.stage2_epochs if self.config.stage2_epochs > 0 else self.config.stage1_epochs
            )
            if self.epoch != final_epoch_budget or self.row_cursor != 0:
                raise ValueError("Completed H0b progress has invalid epoch/cursor state.")
        elif self.epoch >= self._stage_epoch_budget():
            raise ValueError("H0b resume epoch exceeds its configured stage budget.")
        expected_order = self._data_order_payload()
        if checkpoint["data_order"] != expected_order:
            raise ValueError("H0b resume data-order state differs from regeneration.")
        rng = checkpoint["rng_state"]
        if (
            not isinstance(rng, Mapping)
            or set(rng)
            != {
                "torch_cpu",
                "python",
                "cuda_state_absent",
            }
            or rng["cuda_state_absent"] is not True
        ):
            raise ValueError("H0b resume RNG schema changed.")
        torch_state = rng["torch_cpu"]
        if (
            not isinstance(torch_state, torch.Tensor)
            or torch_state.device.type != "cpu"
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1
        ):
            raise ValueError("H0b resume Torch CPU RNG state is invalid.")
        torch.set_rng_state(torch_state)
        _restore_python_rng(rng["python"])
        metrics = checkpoint["epoch_metrics"]
        if not isinstance(metrics, list):
            raise ValueError("H0b resume epoch metrics must be a list.")
        self.epoch_metrics = list(metrics)
        access = checkpoint["access_audit"]
        if not isinstance(access, Mapping):
            raise ValueError("H0b resume access audit must be a mapping.")
        if access.get("materialized_anchor_rows", {}).get("test") != 0:
            raise ValueError("H0b checkpoint claims forbidden test materialization.")
        for field, destination in (
            ("materialized_batches", self.dataset.materialized_batches),
            ("materialized_anchor_rows", self.dataset.materialized_anchor_rows),
            ("materialized_target_rows", self.dataset.materialized_target_rows),
        ):
            counts = access.get(field)
            if (
                not isinstance(counts, Mapping)
                or set(counts) != {"train", "val", "test"}
                or any(type(value) is not int or value < 0 for value in counts.values())
            ):
                raise ValueError(f"H0b resume {field} audit is invalid.")
            destination.update(counts)
        if self.stage != _STAGE_COMPLETE:
            num_rows = len(self.dataset.entries("train"))
            if self.row_cursor >= num_rows:
                raise ValueError("H0b resume row cursor must remain inside its epoch.")

    def export_frozen_reconstructor(
        self,
        path: str | Path,
        *,
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Create a deploy-compatible frozen checkpoint and independent receipt."""
        if self.stage != _STAGE_COMPLETE:
            raise RuntimeError("Frozen H0b export requires both configured stages complete.")
        if self.config.stage2_epochs <= 0:
            raise RuntimeError("Frozen H0b export requires supervised height training.")
        checkpoint_destination = Path(path).resolve()
        receipt_destination = Path(receipt_path).resolve()
        if checkpoint_destination == receipt_destination:
            raise ValueError("Frozen checkpoint and receipt paths must differ.")
        if checkpoint_destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_destination}")
        if receipt_destination.exists():
            raise FileExistsError(f"Refusing to overwrite receipt: {receipt_destination}")
        freeze_audit = freeze_reconstructor(self.model)
        deploy_checkpoint = create_frozen_reconstructor_checkpoint(self.model)
        checkpoint_receipt = _atomic_create_torch(path, deploy_checkpoint)
        bound_payload = {
            "schema_version": 1,
            "classification": "h0b_frozen_reconstructor_export_v1",
            "training_ready_claim": False,
            "checkpoint_path": checkpoint_receipt["path"],
            "checkpoint_file_size_bytes": checkpoint_receipt["file_size_bytes"],
            "checkpoint_file_sha256": checkpoint_receipt["file_sha256"],
            "frozen_payload_sha256": _canonical_json_sha256({
                "schema_sha256": deploy_checkpoint["schema_sha256"],
                "state_sha256": deploy_checkpoint["state_sha256"],
            }),
            "checkpoint_schema_sha256": deploy_checkpoint["schema_sha256"],
            "checkpoint_state_sha256": deploy_checkpoint["state_sha256"],
            "history_length": self.config.history_length,
            "dataset_manifest_payload_sha256": self.dataset.bindings["dataset_manifest_payload_sha256"],
            "target_contract_payload_sha256": self.dataset.bindings["target_contract_payload_sha256"],
            "target_contract_source_sha256": self.dataset.bindings["target_contract_source_sha256"],
            "source_commits": self.dataset.bindings["source_commits"],
            "freeze_audit": freeze_audit,
            "autoencoder_head_in_deploy_checkpoint": False,
        }
        bound_payload["receipt_payload_sha256"] = _canonical_json_sha256(bound_payload)
        receipt_receipt = _atomic_create_json(receipt_path, bound_payload)
        return {
            "checkpoint": checkpoint_receipt,
            "receipt": receipt_receipt,
            "receipt_payload": bound_payload,
        }


def load_h0b_pretrain_checkpoint(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Load an offline resume checkpoint with PyTorch's restricted unpickler."""
    source = Path(path).resolve(strict=True)
    expected_digest = _validate_sha256(
        expected_file_sha256,
        name="resume checkpoint expected_file_sha256",
    )
    with source.open("rb") as stream:
        actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("H0b resume checkpoint file SHA-256 mismatch.")
    loaded = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError("H0b pretraining checkpoint must contain a mapping.")
    return dict(loaded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-contract", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--resume-expected-file-sha256")
    parser.add_argument("--export")
    parser.add_argument("--export-receipt")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--history-length", type=int, choices=(1, 5), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--range-loss-weight", type=float, default=1.0)
    parser.add_argument("--validity-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU-only CLI and print one compact engineering receipt."""
    args = _parser().parse_args(argv)
    if (args.resume is None) != (args.resume_expected_file_sha256 is None):
        raise ValueError("--resume and --resume-expected-file-sha256 must be supplied together.")
    if (args.export is None) != (args.export_receipt is None):
        raise ValueError("--export and --export-receipt must be supplied together.")
    manifest = _read_json_mapping(args.manifest, description="H0b manifest")
    target_contract = _read_json_mapping(args.target_contract, description="H0b height target contract")
    config = H0bPretrainConfig(
        history_length=args.history_length,
        seed=args.seed,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        range_loss_weight=args.range_loss_weight,
        validity_loss_weight=args.validity_loss_weight,
        max_grad_norm=args.max_grad_norm,
    )
    resume = (
        None
        if args.resume is None
        else load_h0b_pretrain_checkpoint(
            args.resume,
            expected_file_sha256=args.resume_expected_file_sha256,
        )
    )
    trainer = H0bOfflinePretrainer(
        config=config,
        dataset_root=args.dataset_root,
        manifest=manifest,
        target_contract=target_contract,
        require_formal_400k=args.formal,
        resume_checkpoint=resume,
    )
    events = trainer.run(max_steps=args.max_steps)
    checkpoint_receipt = trainer.save_checkpoint(args.checkpoint)
    result: dict[str, Any] = {
        "classification": "engineering_offline_pretraining_only",
        "training_ready_claim": False,
        "device": "cpu",
        "stage": trainer.stage,
        "global_step": trainer.global_step,
        "steps_executed": len(events),
        "checkpoint": checkpoint_receipt,
        "access_audit": trainer.dataset.access_audit(),
    }
    if args.export is not None:
        result["export"] = trainer.export_frozen_reconstructor(
            args.export,
            receipt_path=args.export_receipt,
        )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "H0B_PRETRAIN_CHECKPOINT_SCHEMA_VERSION",
    "H0bOfflinePretrainer",
    "H0bPretrainConfig",
    "load_h0b_pretrain_checkpoint",
    "main",
]
