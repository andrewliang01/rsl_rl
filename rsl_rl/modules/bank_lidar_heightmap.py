# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Bank-style LiDAR-only local-heightmap reconstruction.

The deployable reconstructor consumes only a fixed window of spherical range
images.  It deliberately owns no odometry, proprioception, critic input,
recurrent carry, previous prediction, or supervised target interface.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ray_time_attention_encoder import CircularAzimuthConv2d


_INPUT_SPATIAL_SIZE = (16, 96)
_OUTPUT_SPATIAL_SIZE = (28, 20)
_FRAME_CHANNELS = (16, 24, 32)
_ENCODED_SPATIAL_SIZE = (2, 12)
_GRU_HIDDEN_SIZE = 128
_MAX_RANGE_M = 6.0
_CHECKPOINT_SCHEMA_VERSION = 2
_DEPLOY_VALIDATION_MODE = "trusted_no_sync"
_FROZEN_EXPORT_RECEIPT_KEYS = {
    "schema_version",
    "classification",
    "training_ready_claim",
    "checkpoint_path",
    "checkpoint_file_size_bytes",
    "checkpoint_file_sha256",
    "frozen_payload_sha256",
    "checkpoint_schema_sha256",
    "checkpoint_state_sha256",
    "history_length",
    "dataset_manifest_payload_sha256",
    "target_contract_payload_sha256",
    "target_contract_source_sha256",
    "source_commits",
    "freeze_audit",
    "autoencoder_head_in_deploy_checkpoint",
    "receipt_payload_sha256",
}
_FROZEN_EXPORT_FREEZE_AUDIT_KEYS = {
    "schema",
    "schema_sha256",
    "parameter_count",
    "trainable_parameter_count",
    "training",
    "state_sha256",
}
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_CONTRACT = {
    "range_channel": "range_m; ignored and zeroed where valid=0",
    "valid_channel": "finite exact binary {0,1}",
    "valid_range_m": "finite and in (0,6]",
    "content_preflight": "preflight_validate_lidar_history",
}
_TARGET_CONTRACT_KEYS = {
    "schema_version",
    "target_definition",
    "height_unit",
    "height_sign",
    "grid_shape",
    "grid_axis_order",
    "grid_axis_directions",
    "flatten_order",
    "coordinate_frame",
    "origin",
    "resolution_m",
    "unknown_cell_policy",
    "contract_source_sha256",
}


def _group_count(num_channels: int, maximum: int = 8) -> int:
    for num_groups in range(min(maximum, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


def normalize_heightmap_target_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize semantic metadata without guessing geometry."""
    if not isinstance(contract, Mapping) or set(contract) != _TARGET_CONTRACT_KEYS:
        raise ValueError("Heightmap target contract must contain the exact semantic key set.")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        raise ValueError("Heightmap target contract schema_version must be 1.")
    for field in (
        "target_definition",
        "height_sign",
        "coordinate_frame",
        "origin",
        "unknown_cell_policy",
    ):
        if not isinstance(contract[field], str) or not contract[field]:
            raise ValueError(f"Heightmap target contract {field} must be non-empty.")
    if contract["height_unit"] != "metre":
        raise ValueError("Heightmap target height_unit must be 'metre'.")
    grid_shape = contract["grid_shape"]
    if not isinstance(grid_shape, (list, tuple)) or list(grid_shape) != list(_OUTPUT_SPATIAL_SIZE):
        raise ValueError("Heightmap target grid_shape must be [28,20].")
    for field in ("grid_axis_order", "grid_axis_directions"):
        values = contract[field]
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"Heightmap target contract {field} must contain two labels.")
    if len(set(contract["grid_axis_order"])) != 2:
        raise ValueError("Heightmap target grid_axis_order labels must be distinct.")
    if contract["flatten_order"] != "C_contiguous_row_major":
        raise ValueError("Heightmap target flatten_order must be C_contiguous_row_major.")
    resolution_m = contract["resolution_m"]
    if (
        isinstance(resolution_m, bool)
        or not isinstance(resolution_m, (int, float))
        or not math.isfinite(float(resolution_m))
        or resolution_m <= 0.0
    ):
        raise ValueError("Heightmap target resolution_m must be positive.")
    source_sha256 = contract["contract_source_sha256"]
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("Heightmap target contract_source_sha256 must be lowercase 64-hex.")
    return {
        "schema_version": 1,
        "target_definition": contract["target_definition"],
        "height_unit": "metre",
        "height_sign": contract["height_sign"],
        "grid_shape": list(_OUTPUT_SPATIAL_SIZE),
        "grid_axis_order": list(contract["grid_axis_order"]),
        "grid_axis_directions": list(contract["grid_axis_directions"]),
        "flatten_order": "C_contiguous_row_major",
        "coordinate_frame": contract["coordinate_frame"],
        "origin": contract["origin"],
        "resolution_m": float(resolution_m),
        "unknown_cell_policy": contract["unknown_cell_policy"],
        "contract_source_sha256": source_sha256,
    }


def _validate_history_structure(
    ray_history: torch.Tensor,
    *,
    history_length: int,
    expected_device: torch.device | None = None,
) -> None:
    """Validate metadata without reading or scalarizing tensor contents."""
    if not isinstance(ray_history, torch.Tensor):
        raise TypeError("ray_history must be a torch.Tensor.")
    expected_tail = (history_length, 2, *_INPUT_SPATIAL_SIZE)
    if ray_history.ndim != 5 or tuple(ray_history.shape[1:]) != expected_tail:
        raise ValueError(f"LiDAR history must have exact shape [B,{history_length},2,16,96].")
    if ray_history.dtype not in (torch.float16, torch.float32):
        raise TypeError("LiDAR history dtype must be torch.float16 or torch.float32.")
    if ray_history.shape[0] <= 0:
        raise ValueError("LiDAR history batch dimension must be non-empty.")
    if expected_device is not None and ray_history.device != expected_device:
        raise ValueError("LiDAR history and reconstructor parameters must share one device.")


def preflight_validate_lidar_history(
    ray_history: torch.Tensor,
    *,
    history_length: int,
) -> dict[str, Any]:
    """Synchronously validate the trusted source contract before actor use.

    This explicit preflight is content-strict and may synchronize an accelerator.
    It must be run outside collection/deploy forward loops.
    """
    if history_length not in (1, 5):
        raise ValueError("history_length must be exactly 1 or 5.")
    _validate_history_structure(
        ray_history,
        history_length=history_length,
    )
    value = ray_history.to(dtype=torch.float32)
    range_m = value[:, :, 0]
    valid_value = value[:, :, 1]
    if not bool(torch.isfinite(valid_value).all()):
        raise ValueError("LiDAR valid channel must be finite.")
    if not bool(((valid_value == 0.0) | (valid_value == 1.0)).all()):
        raise ValueError("LiDAR valid channel must be exactly binary.")
    valid = valid_value == 1.0
    bad_valid_range = valid & (~torch.isfinite(range_m) | (range_m <= 0.0) | (range_m > _MAX_RANGE_M))
    if bool(bad_valid_range.any()):
        raise ValueError("Every valid LiDAR return must be finite and in (0, 6] metres.")
    valid_return_count = int(valid.sum().item())
    unknown = ~valid
    unknown_nonfinite_count = int((unknown & ~torch.isfinite(range_m)).sum().item())
    valid_ranges = range_m[valid]
    return {
        "validation_mode": "strict_content_preflight_sync",
        "deploy_validation_mode": _DEPLOY_VALIDATION_MODE,
        "source_contract": dict(_SOURCE_CONTRACT),
        "shape": list(ray_history.shape),
        "dtype": str(ray_history.dtype),
        "device": str(ray_history.device),
        "valid_return_count": valid_return_count,
        "unknown_count": int(unknown.sum().item()),
        "unknown_nonfinite_range_count": unknown_nonfinite_count,
        "valid_range_min_m": (float(valid_ranges.amin().item()) if valid_return_count > 0 else None),
        "valid_range_max_m": (float(valid_ranges.amax().item()) if valid_return_count > 0 else None),
    }


class _CircularFrameBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = CircularAzimuthConv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 5),
            stride=(2, 2),
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(value)))


class SphericalRangeFrameEncoder(nn.Module):
    """Shared per-frame CNN with circular azimuth boundaries."""

    def __init__(self) -> None:
        super().__init__()
        channels = (2, *_FRAME_CHANNELS)
        self.blocks = nn.Sequential(
            *(
                _CircularFrameBlock(in_channels, out_channels)
                for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=True)
            )
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor):
            raise TypeError("Sanitized spherical frames must be a tensor.")
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (2, *_INPUT_SPATIAL_SIZE) or frames.dtype != torch.float32:
            raise ValueError("Sanitized spherical frames must have shape [N,2,16,96] and dtype torch.float32.")
        encoded = self.blocks(frames)
        if tuple(encoded.shape[1:]) != (
            _FRAME_CHANNELS[-1],
            *_ENCODED_SPATIAL_SIZE,
        ):
            raise RuntimeError("Spherical frame encoder geometry changed.")
        return encoded


class _HeightmapDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(
            _GRU_HIDDEN_SIZE,
            _FRAME_CHANNELS[-1] * 7 * 5,
        )
        self.first_conv = nn.Conv2d(32, 24, kernel_size=3, padding=1)
        self.second_conv = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, temporal_feature: torch.Tensor) -> torch.Tensor:
        value = self.projection(temporal_feature).reshape(-1, 32, 7, 5)
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        value = F.silu(self.first_conv(value), inplace=True)
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        value = F.silu(self.second_conv(value), inplace=True)
        heightmap_m = self.output_conv(value)
        if tuple(heightmap_m.shape[1:]) != (1, *_OUTPUT_SPATIAL_SIZE):
            raise RuntimeError("Heightmap decoder geometry changed.")
        return heightmap_m


@dataclass(frozen=True)
class SphericalAutoencoderOutput:
    range_m: torch.Tensor
    valid_logits: torch.Tensor


class _CircularUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = CircularAzimuthConv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 5),
            stride=(1, 1),
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return F.silu(self.norm(self.conv(value)), inplace=True)


class SphericalAutoencoderPretrainHead(nn.Module):
    """No-skip decoder used only for spherical frame pretraining."""

    def __init__(self) -> None:
        super().__init__()
        self.skip_connections = False
        self.blocks = nn.Sequential(
            _CircularUpsampleBlock(32, 24),
            _CircularUpsampleBlock(24, 16),
        )
        self.output_conv = CircularAzimuthConv2d(
            16,
            2,
            kernel_size=(3, 5),
            stride=(1, 1),
            bias=True,
        )

    def forward(self, encoded_frames: torch.Tensor) -> SphericalAutoencoderOutput:
        if not isinstance(encoded_frames, torch.Tensor):
            raise TypeError("Encoded frame history must be a tensor.")
        if (
            encoded_frames.ndim != 5
            or tuple(encoded_frames.shape[2:]) != (_FRAME_CHANNELS[-1], *_ENCODED_SPATIAL_SIZE)
            or encoded_frames.dtype != torch.float32
        ):
            raise ValueError("Encoded frame history must have shape [B,K,32,2,12] and dtype torch.float32.")
        batch_size, history_length = encoded_frames.shape[:2]
        if batch_size <= 0 or history_length not in (1, 5):
            raise ValueError("Encoded frame history requires non-empty B and K in {1,5}.")
        value = encoded_frames.flatten(0, 1)
        value = self.blocks(value)
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        decoded = self.output_conv(value).reshape(
            batch_size,
            history_length,
            2,
            *_INPUT_SPATIAL_SIZE,
        )
        return SphericalAutoencoderOutput(
            range_m=F.softplus(decoded[:, :, 0]),
            valid_logits=decoded[:, :, 1],
        )


class BankLidarHeightmapReconstructor(nn.Module):
    """Reconstruct ``[B,1,28,20]`` local height from LiDAR history only."""

    def __init__(
        self,
        *,
        history_length: int,
        target_contract: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if history_length not in (1, 5):
            raise ValueError("history_length must be exactly 1 or 5.")
        self.history_length = int(history_length)
        self.target_contract = None if target_contract is None else normalize_heightmap_target_contract(target_contract)
        self.frame_encoder = SphericalRangeFrameEncoder()
        frame_feature_dim = _FRAME_CHANNELS[-1] * _ENCODED_SPATIAL_SIZE[0] * _ENCODED_SPATIAL_SIZE[1]
        self.temporal_gru = nn.GRU(
            input_size=frame_feature_dim,
            hidden_size=_GRU_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.heightmap_decoder = _HeightmapDecoder()

    def _trusted_no_sync_sanitize(
        self,
        ray_history: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the frozen upstream mask contract without content syncs."""
        encoder_weight = self.frame_encoder.blocks[0].conv.conv.weight
        if encoder_weight.dtype != torch.float32:
            raise RuntimeError("Reconstructor parameters must remain torch.float32.")
        _validate_history_structure(
            ray_history,
            history_length=self.history_length,
            expected_device=encoder_weight.device,
        )
        value = ray_history.to(dtype=torch.float32)
        range_m = value[:, :, 0]
        valid_value = value[:, :, 1]
        # Preflight owns content validation. The actor hot path trusts the
        # frozen binary-mask source contract and performs tensor operations only.
        valid = valid_value == 1.0
        safe_range_m = torch.where(valid, range_m, torch.zeros_like(range_m))
        return torch.stack(
            (safe_range_m / _MAX_RANGE_M, valid.to(torch.float32)),
            dim=2,
        )

    def encode_frame_history(self, ray_history: torch.Tensor) -> torch.Tensor:
        sanitized = self._trusted_no_sync_sanitize(ray_history)
        batch_size = sanitized.shape[0]
        encoded = self.frame_encoder(sanitized.flatten(0, 1))
        return encoded.reshape(
            batch_size,
            self.history_length,
            _FRAME_CHANNELS[-1],
            *_ENCODED_SPATIAL_SIZE,
        )

    def forward(self, ray_history: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_frame_history(ray_history)
        batch_size = encoded.shape[0]
        sequence = encoded.flatten(start_dim=2)
        # The temporal state is local to this fixed window and never persisted.
        initial_hidden = sequence.new_zeros(1, batch_size, _GRU_HIDDEN_SIZE)
        _, final_hidden = self.temporal_gru(sequence, initial_hidden)
        heightmap_m = self.heightmap_decoder(final_hidden[0])
        if heightmap_m.dtype != torch.float32:
            raise RuntimeError("Heightmap output must remain torch.float32.")
        return heightmap_m


def _validate_masked_loss_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    expected_tail: tuple[int, ...],
) -> torch.Tensor:
    """Strict synchronous validation for offline supervised losses only."""
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or not isinstance(valid, torch.Tensor)
    ):
        raise TypeError("prediction, target, and valid must be tensors.")
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target, and valid must have identical shapes.")
    if prediction.shape[0] <= 0:
        raise ValueError("Masked loss batch dimension must be non-empty.")
    if prediction.ndim != len(expected_tail) + 1 or tuple(prediction.shape[1:]) != expected_tail:
        raise ValueError(f"Masked loss tensor tail must be {expected_tail}.")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("Masked loss prediction and target must be torch.float32.")
    if valid.dtype != torch.bool:
        raise TypeError("Masked loss valid tensor must be torch.bool.")
    if prediction.device != target.device or prediction.device != valid.device:
        raise ValueError("Masked loss tensors must share one device.")
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    if not bool(torch.isfinite(safe_prediction).all()) or not bool(torch.isfinite(safe_target).all()):
        raise ValueError("Masked loss has a non-finite value in a valid cell.")
    return (safe_prediction - safe_target).square().sum() / valid.sum().clamp_min(1)


def valid_masked_range_mse(
    predicted_range_m: torch.Tensor,
    target_range_m: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Offline range reconstruction MSE over valid target returns only."""
    if not isinstance(predicted_range_m, torch.Tensor):
        raise TypeError("predicted_range_m must be a tensor.")
    if predicted_range_m.ndim != 4:
        raise ValueError("Range reconstruction must have shape [B,K,16,96].")
    history_length = predicted_range_m.shape[1]
    if history_length not in (1, 5):
        raise ValueError("Range reconstruction history length must be 1 or 5.")
    return _validate_masked_loss_inputs(
        predicted_range_m,
        target_range_m,
        target_valid,
        expected_tail=(history_length, *_INPUT_SPATIAL_SIZE),
    )


def spherical_valid_bce(
    valid_logits: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Offline binary-validity reconstruction loss over every spherical cell."""
    if not isinstance(valid_logits, torch.Tensor) or not isinstance(target_valid, torch.Tensor):
        raise TypeError("Validity logits and target must be tensors.")
    if (
        valid_logits.ndim != 4
        or valid_logits.shape != target_valid.shape
        or valid_logits.shape[0] <= 0
        or valid_logits.shape[1] not in (1, 5)
        or tuple(valid_logits.shape[2:]) != _INPUT_SPATIAL_SIZE
    ):
        raise ValueError("Validity tensors must have shape [B,K,16,96], K in {1,5}.")
    if valid_logits.dtype != torch.float32 or target_valid.dtype != torch.bool:
        raise TypeError("Validity logits must be float32 and target_valid boolean.")
    if valid_logits.device != target_valid.device:
        raise ValueError("Validity logits and target must share one device.")
    if not bool(torch.isfinite(valid_logits).all()):
        raise ValueError("Validity logits must be finite.")
    return F.binary_cross_entropy_with_logits(
        valid_logits,
        target_valid.to(torch.float32),
        reduction="mean",
    )


def supervised_height_valid_mse(
    predicted_height_m: torch.Tensor,
    target_height_m: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Offline supervised height MSE over valid target map cells only."""
    return _validate_masked_loss_inputs(
        predicted_height_m,
        target_height_m,
        target_valid,
        expected_tail=(1, *_OUTPUT_SPATIAL_SIZE),
    )


def _state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Reconstructor state_dict values must all be tensors.")
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_schema(model: BankLidarHeightmapReconstructor) -> dict[str, Any]:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in model.state_dict().items()}


def _schema_sha256(schema: Mapping[str, Any]) -> str:
    payload = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reconstructor_checkpoint_schema(
    model: BankLidarHeightmapReconstructor,
) -> dict[str, Any]:
    if not isinstance(model, BankLidarHeightmapReconstructor):
        raise TypeError("model must be BankLidarHeightmapReconstructor.")
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "class_name": "BankLidarHeightmapReconstructor",
        "history_length": model.history_length,
        "input_tail": [model.history_length, 2, *_INPUT_SPATIAL_SIZE],
        "output_tail": [1, *_OUTPUT_SPATIAL_SIZE],
        "accepted_input_dtypes": ["torch.float16", "torch.float32"],
        "internal_dtype": "torch.float32",
        "deploy_validation_mode": _DEPLOY_VALIDATION_MODE,
        "source_contract": dict(_SOURCE_CONTRACT),
        "strict_preflight_helper": "preflight_validate_lidar_history",
        "offline_losses_in_deploy_forward": False,
        "max_range_m": _MAX_RANGE_M,
        "frame_channels": list(_FRAME_CHANNELS),
        "encoded_spatial_size": list(_ENCODED_SPATIAL_SIZE),
        "gru_hidden_size": _GRU_HIDDEN_SIZE,
        "fixed_window_zero_hidden": True,
        "persistent_recurrent_state": False,
        "shared_frame_encoder": True,
        "previous_prediction_feedback": False,
        "odometry_input": False,
        "proprioception_input": False,
        "critic_input": False,
        "pretrain_head_in_deploy_checkpoint": False,
        "height_unit": "metre",
        "target_contract": (None if model.target_contract is None else dict(model.target_contract)),
        "deploy_inputs": ["ray_history"],
        "state_schema": _state_schema(model),
    }


def freeze_reconstructor(
    model: BankLidarHeightmapReconstructor,
) -> dict[str, Any]:
    """Freeze a deployable reconstructor and return an explicit audit."""
    if not isinstance(model, BankLidarHeightmapReconstructor):
        raise TypeError("model must be BankLidarHeightmapReconstructor.")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("Reconstructor parameters must remain torch.float32.")
    model.eval()
    model.requires_grad_(False)
    state = model.state_dict()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    schema = reconstructor_checkpoint_schema(model)
    return {
        "schema": schema,
        "schema_sha256": _schema_sha256(schema),
        "parameter_count": parameter_count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "training": model.training,
        "state_sha256": _state_sha256(state),
    }


def create_frozen_reconstructor_checkpoint(
    model: BankLidarHeightmapReconstructor,
) -> dict[str, Any]:
    """Create a strict in-memory checkpoint for an already-frozen model."""
    if not isinstance(model, BankLidarHeightmapReconstructor):
        raise TypeError("model must be BankLidarHeightmapReconstructor.")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("Reconstructor parameters must remain torch.float32.")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Reconstructor must be explicitly frozen before checkpointing.")
    state = {name: value.detach().contiguous().cpu().clone() for name, value in model.state_dict().items()}
    schema = reconstructor_checkpoint_schema(model)
    return {
        "schema": schema,
        "schema_sha256": _schema_sha256(schema),
        "state_dict": state,
        "state_sha256": _state_sha256(state),
    }


def load_frozen_reconstructor_checkpoint(
    checkpoint: Mapping[str, Any],
) -> BankLidarHeightmapReconstructor:
    """Strictly reconstruct and freeze a deployable H0b module."""
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "schema",
        "schema_sha256",
        "state_dict",
        "state_sha256",
    }:
        raise ValueError("Frozen reconstructor checkpoint keys changed.")
    schema = checkpoint["schema"]
    schema_digest = checkpoint["schema_sha256"]
    state = checkpoint["state_dict"]
    digest = checkpoint["state_sha256"]
    if not isinstance(schema, Mapping) or not isinstance(state, Mapping):
        raise TypeError("Checkpoint schema and state_dict must be mappings.")
    if not isinstance(schema_digest, str) or _schema_sha256(schema) != schema_digest:
        raise ValueError("Frozen reconstructor checkpoint schema mismatch: digest.")
    history_length = schema.get("history_length")
    if history_length not in (1, 5):
        raise ValueError("Checkpoint history_length must be exactly 1 or 5.")
    target_contract = schema.get("target_contract")
    model = BankLidarHeightmapReconstructor(
        history_length=history_length,
        target_contract=target_contract,
    )
    expected_schema = reconstructor_checkpoint_schema(model)
    if dict(schema) != expected_schema:
        raise ValueError("Frozen reconstructor checkpoint schema mismatch.")
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("Frozen reconstructor state_dict keys mismatch.")
    for name, expected in expected_state.items():
        candidate = state[name]
        if (
            not isinstance(candidate, torch.Tensor)
            or candidate.shape != expected.shape
            or candidate.dtype != expected.dtype
        ):
            raise ValueError(f"Frozen reconstructor tensor schema mismatch: {name}.")
    if not isinstance(digest, str) or _state_sha256(state) != digest:
        raise ValueError("Frozen reconstructor state digest mismatch.")
    model.load_state_dict(state, strict=True)
    freeze_reconstructor(model)
    return model


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase 64-hex.")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
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


def _decode_canonical_receipt(payload: bytes) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Frozen reconstructor receipt repeats key: {key}.")
            result[key] = value
        return result

    try:
        text = payload.decode("ascii")
        loaded = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {value}.")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Frozen reconstructor receipt is not canonical JSON.") from error
    if not isinstance(loaded, dict):
        raise TypeError("Frozen reconstructor receipt must contain one JSON object.")
    if _canonical_json_bytes(loaded) != payload:
        raise ValueError("Frozen reconstructor receipt bytes are not canonical.")
    return loaded


def _validate_frozen_export_receipt(
    receipt: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_file_sha256: str,
    checkpoint_file_size_bytes: int,
    model: BankLidarHeightmapReconstructor,
) -> dict[str, Any]:
    if set(receipt) != _FROZEN_EXPORT_RECEIPT_KEYS:
        raise ValueError("Frozen reconstructor receipt keys changed.")
    receipt_without_digest = dict(receipt)
    receipt_payload_sha256 = receipt_without_digest.pop("receipt_payload_sha256")
    if (
        _require_sha256(
            receipt_payload_sha256,
            field="receipt receipt_payload_sha256",
        )
        != hashlib.sha256(
            json.dumps(
                receipt_without_digest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
    ):
        raise ValueError("Frozen reconstructor receipt payload SHA-256 mismatch.")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("Frozen reconstructor receipt schema_version must be 1.")
    if receipt["classification"] != "h0b_frozen_reconstructor_export_v1":
        raise ValueError("Frozen reconstructor receipt classification changed.")
    if receipt["training_ready_claim"] is not False:
        raise ValueError("Frozen reconstructor receipt must not claim training readiness.")
    reported_path = receipt["checkpoint_path"]
    if not isinstance(reported_path, str) or not Path(reported_path).is_absolute():
        raise ValueError("Frozen reconstructor receipt checkpoint_path must be absolute metadata.")
    reported_size = receipt["checkpoint_file_size_bytes"]
    if type(reported_size) is not int or reported_size <= 0:
        raise ValueError("Frozen reconstructor receipt checkpoint size is invalid.")
    if reported_size != checkpoint_file_size_bytes:
        raise ValueError("Frozen reconstructor checkpoint file size differs from receipt.")
    if (
        _require_sha256(
            receipt["checkpoint_file_sha256"],
            field="receipt checkpoint_file_sha256",
        )
        != checkpoint_file_sha256
    ):
        raise ValueError("Frozen reconstructor checkpoint file SHA-256 differs from receipt.")

    schema_sha256 = _require_sha256(checkpoint["schema_sha256"], field="checkpoint schema_sha256")
    state_sha256 = _require_sha256(checkpoint["state_sha256"], field="checkpoint state_sha256")
    if (
        _require_sha256(
            receipt["checkpoint_schema_sha256"],
            field="receipt checkpoint_schema_sha256",
        )
        != schema_sha256
        or _require_sha256(
            receipt["checkpoint_state_sha256"],
            field="receipt checkpoint_state_sha256",
        )
        != state_sha256
    ):
        raise ValueError("Frozen reconstructor schema/state SHA-256 differs from receipt.")
    expected_frozen_payload_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema_sha256": schema_sha256,
                "state_sha256": state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if (
        _require_sha256(
            receipt["frozen_payload_sha256"],
            field="receipt frozen_payload_sha256",
        )
        != expected_frozen_payload_sha256
    ):
        raise ValueError("Frozen reconstructor payload SHA-256 differs from checkpoint.")

    if type(receipt["history_length"]) is not int or receipt["history_length"] != model.history_length:
        raise ValueError("Frozen reconstructor receipt history_length differs from checkpoint.")
    target_contract = model.target_contract
    if target_contract is None:
        raise ValueError("Frozen reconstructor production artifact requires a target contract.")
    target_payload_sha256 = hashlib.sha256(
        json.dumps(
            target_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if (
        _require_sha256(
            receipt["target_contract_payload_sha256"],
            field="receipt target_contract_payload_sha256",
        )
        != target_payload_sha256
    ):
        raise ValueError("Frozen reconstructor target contract payload SHA-256 differs.")
    if (
        _require_sha256(
            receipt["target_contract_source_sha256"],
            field="receipt target_contract_source_sha256",
        )
        != target_contract["contract_source_sha256"]
    ):
        raise ValueError("Frozen reconstructor target contract source SHA-256 differs.")
    _require_sha256(
        receipt["dataset_manifest_payload_sha256"],
        field="receipt dataset_manifest_payload_sha256",
    )
    _require_sha256(
        receipt["target_contract_payload_sha256"],
        field="receipt target_contract_payload_sha256",
    )
    source_commits = receipt["source_commits"]
    if not isinstance(source_commits, Mapping) or set(source_commits) != {
        "lab_pro",
        "rsl_rl",
        "isaaclab",
    }:
        raise ValueError("Frozen reconstructor receipt source commits changed.")
    if any(not isinstance(value, str) or _HEX40_RE.fullmatch(value) is None for value in source_commits.values()):
        raise ValueError("Frozen reconstructor source commits must be lowercase 40-hex.")

    freeze_audit = receipt["freeze_audit"]
    if not isinstance(freeze_audit, Mapping) or set(freeze_audit) != (_FROZEN_EXPORT_FREEZE_AUDIT_KEYS):
        raise ValueError("Frozen reconstructor receipt freeze audit keys changed.")
    if (
        freeze_audit["schema"] != checkpoint["schema"]
        or freeze_audit["schema_sha256"] != schema_sha256
        or freeze_audit["state_sha256"] != state_sha256
    ):
        raise ValueError("Frozen reconstructor freeze audit differs from checkpoint.")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        type(freeze_audit["parameter_count"]) is not int
        or freeze_audit["parameter_count"] != parameter_count
        or type(freeze_audit["trainable_parameter_count"]) is not int
        or freeze_audit["trainable_parameter_count"] != 0
        or freeze_audit["training"] is not False
    ):
        raise ValueError("Frozen reconstructor receipt freeze audit is inconsistent.")
    if receipt["autoencoder_head_in_deploy_checkpoint"] is not False:
        raise ValueError("Autoencoder head is forbidden in a deploy checkpoint.")
    return dict(receipt)


def load_frozen_reconstructor_artifact(
    checkpoint_path: str | Path,
    *,
    expected_file_sha256: str,
    receipt_path: str | Path,
) -> tuple[BankLidarHeightmapReconstructor, dict[str, Any]]:
    """Load one production H0b artifact through an independently retained receipt.

    Both paths are opened without following symbolic links.  The explicitly
    configured checkpoint file digest is the trust root; the receipt's reported
    path is metadata only and is never used to select the file to load.
    """
    expected_digest = _require_sha256(
        expected_file_sha256,
        field="bank_reconstructor_checkpoint_expected_file_sha256",
    )
    checkpoint_source = Path(checkpoint_path)
    receipt_source = Path(receipt_path)
    if checkpoint_source.absolute() == receipt_source.absolute():
        raise ValueError("Frozen reconstructor checkpoint and receipt paths must differ.")

    # Runtime import avoids a modules<->utils package initialization cycle.
    from rsl_rl.utils.training_receipt import retain_regular_file

    checkpoint_bytes, checkpoint_record = retain_regular_file(checkpoint_source)
    if checkpoint_record["sha256"] != expected_digest:
        raise ValueError("Frozen reconstructor checkpoint file SHA-256 mismatch.")
    receipt_bytes, _ = retain_regular_file(receipt_source)
    receipt = _decode_canonical_receipt(receipt_bytes)
    loaded = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(loaded, Mapping):
        raise TypeError("Frozen reconstructor checkpoint must contain a mapping.")
    checkpoint = dict(loaded)
    model = load_frozen_reconstructor_checkpoint(checkpoint)
    validated_receipt = _validate_frozen_export_receipt(
        receipt,
        checkpoint=checkpoint,
        checkpoint_file_sha256=checkpoint_record["sha256"],
        checkpoint_file_size_bytes=checkpoint_record["bytes"],
        model=model,
    )
    return model, validated_receipt


__all__ = [
    "BankLidarHeightmapReconstructor",
    "SphericalAutoencoderOutput",
    "SphericalAutoencoderPretrainHead",
    "SphericalRangeFrameEncoder",
    "create_frozen_reconstructor_checkpoint",
    "freeze_reconstructor",
    "load_frozen_reconstructor_artifact",
    "load_frozen_reconstructor_checkpoint",
    "normalize_heightmap_target_contract",
    "preflight_validate_lidar_history",
    "reconstructor_checkpoint_schema",
    "spherical_valid_bce",
    "supervised_height_valid_mse",
    "valid_masked_range_mse",
]
