# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict deployment receipt for the five-channel Ray-Event actor input."""

from __future__ import annotations

import copy
import math
from typing import Any, Final

from .ray_event_observation import RAY_EVENT_CHANNELS
from .raw_event_pies import PIES_EVENT_WINDOW_S


RAY_EVENT_DEPLOYMENT_CONTRACT: Final[str] = "g1_ray_event_actor_v2"
RAY_EVENT_CHANNEL_SEMANTICS: Final[dict[str, str]] = {
    "range_m": "same_winner_metric_range_or_zero_when_invalid",
    "return_valid": "binary_observed_return_mask",
    "return_age_s": "same_winner_action_time_age_or_zero_when_invalid",
    "packet_age_s": "packet_capture_age_floor_constant_within_input_frame",
    "frame_valid": "binary_history_slot_validity_constant_within_input_frame",
}
RAY_EVENT_ACTOR_EXPORT_CONTRACT: Final[dict[str, Any]] = {
    "observation_group": "ray_event_policy",
    "logical_layout": "B,K,C,H,W",
    "torchscript_input": "ray_event_policy",
    "onnx_input": "ray_event_policy",
    "dynamic_batch": True,
}


def build_ray_event_deployment_receipt(
    *,
    history_length: int,
    spatial_size: tuple[int, int],
    source: str,
    temporal_baseline: str,
    history_reduction: str = "history",
    geometry: str = "native",
    packet_time_quantization_upper_bound_s: float | None = None,
    training_ready: bool = False,
    smoke_receipt_sha256: str | None = None,
    self_return_filter: str = "unknown",
    self_return_filter_config_sha256: str | None = None,
    self_return_filtered_count: int | None = None,
    event_union_stage: str | None = None,
    packetization_invariance_proof_sha256: str | None = None,
    real_tensor_manifest_sha256: str | None = None,
    clock_alignment_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    deployment_scope = (
        "deployment_candidate"
        if self_return_filter in ("upstream_static_mask", "urdf_kinematic")
        else "synthetic_conformance_only"
    )
    if event_union_stage is None:
        if history_reduction == "history":
            event_union_stage = "none"
        elif history_reduction == "pies_latest_event_k1":
            event_union_stage = "raw_event"
        else:
            event_union_stage = "post_packet_raster"
    invariance_proven = (
        event_union_stage == "raw_event"
        and isinstance(packetization_invariance_proof_sha256, str)
        and len(packetization_invariance_proof_sha256) == 64
    )
    real_source_authenticated = (
        source == "livox_per_return"
        and _is_sha256(real_tensor_manifest_sha256)
        and _is_sha256(clock_alignment_receipt_sha256)
    )
    receipt = {
        "contract": RAY_EVENT_DEPLOYMENT_CONTRACT,
        "channels": list(RAY_EVENT_CHANNELS),
        "channel_semantics": dict(RAY_EVENT_CHANNEL_SEMANTICS),
        "actor_export_contract": dict(RAY_EVENT_ACTOR_EXPORT_CONTRACT),
        "input_shape": [history_length, 5, *spatial_size],
        "history_length": history_length,
        "spatial_size": list(spatial_size),
        "source": source,
        "temporal_baseline": temporal_baseline,
        "history_reduction": history_reduction,
        "event_union_stage": event_union_stage,
        "event_window_s": (
            PIES_EVENT_WINDOW_S
            if history_reduction in (
                "raster_latest_event_prototype",
                "pies_latest_event_k1",
            )
            else None
        ),
        "packetization_invariance_proven": invariance_proven,
        "packetization_invariance_proof_sha256": (
            packetization_invariance_proof_sha256
        ),
        "real_tensor_manifest_sha256": real_tensor_manifest_sha256,
        "clock_alignment_receipt_sha256": clock_alignment_receipt_sha256,
        "acquisition_delta_proprio_contract": (
            "raw_conformance_same_winner_actor_not_wired"
        ),
        "pies_full_contract_ready": False,
        "geometry": geometry,
        "simulator_time_capability": "packet_capture_only",
        "packet_time_quantization_upper_bound_s": (
            packet_time_quantization_upper_bound_s
        ),
        "training_ready": training_ready,
        "smoke_receipt_sha256": smoke_receipt_sha256,
        "legacy_ray_time_tasks_unchanged": True,
        "per_return_claim_allowed": real_source_authenticated,
        "self_return_filter": self_return_filter,
        "self_return_filter_config_sha256": self_return_filter_config_sha256,
        "self_return_filtered_count": self_return_filtered_count,
        "filtered_return_semantics": "removed_observation_not_emitted_no_return",
        "deployment_scope": deployment_scope,
    }
    validate_ray_event_deployment_receipt(receipt)
    return receipt


def validate_ray_event_deployment_receipt(receipt: dict[str, Any]) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("Ray-Event deployment receipt must be a dictionary.")
    required = {
        "contract",
        "channels",
        "channel_semantics",
        "actor_export_contract",
        "input_shape",
        "history_length",
        "spatial_size",
        "source",
        "temporal_baseline",
        "history_reduction",
        "event_union_stage",
        "event_window_s",
        "packetization_invariance_proven",
        "packetization_invariance_proof_sha256",
        "real_tensor_manifest_sha256",
        "clock_alignment_receipt_sha256",
        "acquisition_delta_proprio_contract",
        "pies_full_contract_ready",
        "geometry",
        "simulator_time_capability",
        "packet_time_quantization_upper_bound_s",
        "training_ready",
        "smoke_receipt_sha256",
        "legacy_ray_time_tasks_unchanged",
        "per_return_claim_allowed",
        "self_return_filter",
        "self_return_filter_config_sha256",
        "self_return_filtered_count",
        "filtered_return_semantics",
        "deployment_scope",
    }
    if set(receipt) != required:
        raise ValueError(
            "Ray-Event receipt fields differ from the strict contract: "
            f"missing={sorted(required - set(receipt))}, "
            f"extra={sorted(set(receipt) - required)}."
        )
    if receipt["contract"] != RAY_EVENT_DEPLOYMENT_CONTRACT:
        raise ValueError("Unknown Ray-Event deployment contract.")
    if receipt["channels"] != list(RAY_EVENT_CHANNELS):
        raise ValueError("Ray-Event channel order changed.")
    if receipt["channel_semantics"] != RAY_EVENT_CHANNEL_SEMANTICS:
        raise ValueError("Ray-Event channel semantics changed.")
    if receipt["actor_export_contract"] != RAY_EVENT_ACTOR_EXPORT_CONTRACT:
        raise ValueError("Ray-Event actor/export interface changed.")
    history_length = receipt["history_length"]
    spatial_size = receipt["spatial_size"]
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length <= 0:
        raise ValueError("history_length must be a positive integer.")
    if (
        not isinstance(spatial_size, list)
        or len(spatial_size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in spatial_size)
    ):
        raise ValueError("spatial_size must contain two positive integers.")
    if receipt["input_shape"] != [history_length, 5, *spatial_size]:
        raise ValueError("input_shape does not match history/channels/spatial size.")
    source = receipt["source"]
    baseline = receipt["temporal_baseline"]
    if source not in (
        "raycaster_packet",
        "raycaster_quantized_event",
        "livox_per_return",
    ):
        raise ValueError("Unsupported Ray-Event source.")
    if baseline not in (
        "packet_age",
        "age_zero",
        "per_return_age",
        "quantized_event_age",
    ):
        raise ValueError("Unsupported temporal baseline.")
    if source == "raycaster_packet" and baseline == "per_return_age":
        raise ValueError("RayCaster cannot claim per-return acquisition time.")
    if (
        source == "raycaster_quantized_event"
        and baseline != "quantized_event_age"
    ):
        raise ValueError(
            "Quantized RayCaster events require quantized_event_age."
        )
    if baseline == "quantized_event_age" and source != "raycaster_quantized_event":
        raise ValueError(
            "quantized_event_age requires raycaster_quantized_event source."
        )
    real_manifest_sha = receipt["real_tensor_manifest_sha256"]
    clock_receipt_sha = receipt["clock_alignment_receipt_sha256"]
    if source != "livox_per_return":
        if real_manifest_sha is not None or clock_receipt_sha is not None:
            raise ValueError("RayCaster sources reject real-sensor provenance hashes.")
        expected_per_return_claim = False
    else:
        provenance_values = (real_manifest_sha, clock_receipt_sha)
        if any(value is not None for value in provenance_values) and not all(
            _is_sha256(value) for value in provenance_values
        ):
            raise ValueError(
                "Livox provenance requires both lowercase real-tensor and "
                "clock-alignment SHA-256 receipts."
            )
        expected_per_return_claim = all(
            _is_sha256(value) for value in provenance_values
        )
    if receipt["per_return_claim_allowed"] is not expected_per_return_claim:
        raise ValueError("Per-return claim conflicts with authenticated source provenance.")
    quantization = receipt["packet_time_quantization_upper_bound_s"]
    if source in ("raycaster_packet", "raycaster_quantized_event"):
        if (
            isinstance(quantization, bool)
            or not isinstance(quantization, (int, float))
            or not math.isfinite(float(quantization))
            or float(quantization) <= 0.0
        ):
            raise ValueError(
                "RayCaster packet time requires a positive quantization upper bound."
            )
    elif quantization is not None:
        raise ValueError("Livox per-return time rejects a packet quantization claim.")
    if receipt["history_reduction"] not in (
        "history",
        "exact_union_k1",
        "raster_latest_event_prototype",
        "pies_latest_event_k1",
    ):
        raise ValueError("Unsupported history reduction.")
    union_stage = receipt["event_union_stage"]
    reduction = receipt["history_reduction"]
    if reduction == "history" and union_stage != "none":
        raise ValueError("History input requires event_union_stage='none'.")
    if reduction == "exact_union_k1" and union_stage != "post_packet_raster":
        raise ValueError("Nearest-range coverage oracle is post-packet-raster.")
    if (
        reduction == "raster_latest_event_prototype"
        and union_stage != "post_packet_raster"
    ):
        raise ValueError("Raster latest-event prototype must remain post-raster.")
    if reduction == "pies_latest_event_k1" and union_stage != "raw_event":
        raise ValueError("PIES latest-event K1 must reduce raw events.")
    event_window_s = receipt["event_window_s"]
    if reduction in (
        "raster_latest_event_prototype",
        "pies_latest_event_k1",
    ):
        if event_window_s != PIES_EVENT_WINDOW_S:
            raise ValueError("Latest-event window must remain exactly 0.5 s.")
    elif event_window_s is not None:
        raise ValueError(
            "Only the raster prototype or raw PIES may declare an event window."
        )
    if not isinstance(receipt["packetization_invariance_proven"], bool):
        raise ValueError("packetization_invariance_proven must be boolean.")
    proof_sha = receipt["packetization_invariance_proof_sha256"]
    if receipt["packetization_invariance_proven"]:
        if union_stage != "raw_event":
            raise ValueError("Packetization proof applies only before rasterization.")
        if source != "livox_per_return" or baseline != "per_return_age":
            raise ValueError(
                "Raw-event packetization proof requires authenticated Livox "
                "per-return timing."
            )
        if not isinstance(proof_sha, str) or len(proof_sha) != 64:
            raise ValueError("Packetization proof requires a SHA-256 evidence hash.")
    elif proof_sha is not None:
        raise ValueError("Unproven packetization invariance rejects a proof hash.")
    if receipt["acquisition_delta_proprio_contract"] != (
        "raw_conformance_same_winner_actor_not_wired"
    ):
        raise ValueError("Acquisition delta-proprio contract changed.")
    if receipt["pies_full_contract_ready"] is not False:
        raise ValueError(
            "The five-channel actor does not yet consume acquisition delta-proprio."
        )
    if receipt["geometry"] not in ("native", "rerender"):
        raise ValueError("Unsupported geometry.")
    if receipt["simulator_time_capability"] != "packet_capture_only":
        raise ValueError("Simulator capability must remain packet_capture_only.")
    if receipt["legacy_ray_time_tasks_unchanged"] is not True:
        raise ValueError("Legacy Ray-Time compatibility attestation is required.")
    filter_mode = receipt["self_return_filter"]
    if filter_mode not in (
        "unknown",
        "disabled",
        "upstream_static_mask",
        "urdf_kinematic",
        "simulator_geometry_excludes_robot",
    ):
        raise ValueError("Unsupported self-return filter provenance.")
    filter_hash = receipt["self_return_filter_config_sha256"]
    filtered_count = receipt["self_return_filtered_count"]
    if filter_mode in (
        "upstream_static_mask",
        "urdf_kinematic",
        "simulator_geometry_excludes_robot",
    ):
        if not isinstance(filter_hash, str) or len(filter_hash) != 64:
            raise ValueError("Authenticated self-return filtering requires a config SHA-256.")
        if (
            isinstance(filtered_count, bool)
            or not isinstance(filtered_count, int)
            or filtered_count < 0
        ):
            raise ValueError("Authenticated self-return filtering requires a count.")
    elif filter_hash is not None or filtered_count is not None:
        raise ValueError("Unknown/disabled self-return filtering rejects fake evidence.")
    if receipt["filtered_return_semantics"] != (
        "removed_observation_not_emitted_no_return"
    ):
        raise ValueError("Filtered returns must not become emitted no-return evidence.")
    expected_scope = (
        "deployment_candidate"
        if filter_mode in ("upstream_static_mask", "urdf_kinematic")
        else "synthetic_conformance_only"
    )
    if receipt["deployment_scope"] != expected_scope:
        raise ValueError("Deployment scope conflicts with self-return provenance.")
    training_ready = receipt["training_ready"]
    smoke_sha = receipt["smoke_receipt_sha256"]
    if not isinstance(training_ready, bool):
        raise ValueError("training_ready must be boolean.")
    if training_ready:
        if receipt["deployment_scope"] != "deployment_candidate":
            raise ValueError(
                "training_ready requires authenticated real self-return filtering."
            )
        if source == "livox_per_return" and not receipt["per_return_claim_allowed"]:
            raise ValueError(
                "training_ready requires bound real-tensor and clock-alignment receipts."
            )
        if receipt["history_reduction"] == "raster_latest_event_prototype":
            raise ValueError("Raster latest-event prototype can never be training-ready.")
        if (
            receipt["history_reduction"] == "pies_latest_event_k1"
            and not receipt["packetization_invariance_proven"]
        ):
            raise ValueError(
                "Latest-event training requires a raw-event packetization "
                "invariance proof."
            )
        if receipt["history_reduction"] == "pies_latest_event_k1":
            raise ValueError(
                "PIES training remains blocked until acquisition delta-proprio "
                "is wired into the actor/export contract."
            )
        if not isinstance(smoke_sha, str) or len(smoke_sha) != 64:
            raise ValueError(
                "training_ready requires a 64-character smoke receipt SHA-256."
            )
    elif smoke_sha is not None:
        raise ValueError("Unready configuration must not carry a smoke receipt.")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def copy_validated_ray_event_deployment_receipt(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    validate_ray_event_deployment_receipt(receipt)
    return copy.deepcopy(receipt)


__all__ = [
    "RAY_EVENT_ACTOR_EXPORT_CONTRACT",
    "RAY_EVENT_CHANNEL_SEMANTICS",
    "RAY_EVENT_DEPLOYMENT_CONTRACT",
    "build_ray_event_deployment_receipt",
    "copy_validated_ray_event_deployment_receipt",
    "validate_ray_event_deployment_receipt",
]
