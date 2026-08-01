# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rsl_rl.utils.ray_time_actor_complexity import (
    ComplexityReceiptError,
    ModuleMacCounter,
    _canonical_bytes,
    _sha256,
    build_ray_time_actor_complexity_receipt,
    construct_global_ray_time_actor,
    write_complexity_receipt_create_once,
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_ray_time_actor_complexity_receipt()


def test_receipt_binds_fixed_cpu_schema_source_and_partial_claim(receipt: dict) -> None:
    assert receipt["schema"] == {
        "name": "ray-time-actor-complexity-receipt",
        "version": 1,
    }
    assert receipt["contract"] == "ray_time_actor_complexity_partial_v1"
    assert receipt["scope"]["device"] == "cpu"
    assert receipt["scope"]["batch_size"] == 1
    assert receipt["scope"]["warmup"]["applicable"] is False
    assert receipt["scope"]["latency"] == {
        "measured": False,
        "target_platform_profiled": False,
        "p50_p95_p99_reported": False,
        "claim_boundary": "This receipt cannot support latency or real-time claims.",
    }
    assert receipt["formal_eligibility"]["eligible"] is False
    assert receipt["formal_eligibility"]["status"] == "partial"
    assert len(receipt["source"]["git"]["head"]) == 40
    assert set(receipt["source"]["files"]) == {
        "complexity_tool",
        "actor_model",
        "ray_time_encoder",
        "matched_union_router",
    }
    for binding in receipt["source"]["files"].values():
        assert len(binding["sha256"]) == 64
        assert binding["bytes"] > 0
    unsigned = dict(receipt)
    stored = unsigned.pop("payload_sha256")
    assert stored == _sha256(_canonical_bytes(unsigned))


def test_real_variants_freeze_parameter_and_known_mac_subtotals(receipt: dict) -> None:
    variants = {item["variant"]: item for item in receipt["variants"]}
    assert set(variants) == {
        "global_k1",
        "global_k5",
        "matched_post_raster_nearest_union_k1",
    }
    for item in variants.values():
        params = item["parameters"]
        assert params["total_trainable_parameters"] == 177_850
        assert params["active_trainable_parameters"] == 136_186
        assert params["inactive_trainable_parameters"] == 41_664
        assert params["active_trainable_parameters"] + params[
            "inactive_trainable_parameters"
        ] == params["total_trainable_parameters"]
        assert params["duplicate_parameters_counted_once"] is True
        inactive = {
            record["name"]: record["numel"]
            for record in params["inactive_parameter_names"]
        }
        assert inactive["elevation_encoder.query_bias"] == 256
        assert "elevation_encoder.global_ablation_adapter.0.weight" not in inactive

        mac = item["mac"]
        assert mac["coverage"]["status"] == "partial"
        assert mac["coverage"]["formal_eligible"] is False
        assert mac["coverage"]["total_mac_reported"] is False
        assert "total_mac" not in mac
        assert mac["dispatcher_crosscheck"]["exact_match"] is True

    assert variants["global_k1"]["mac"]["known_mac_subtotal"] == 3_267_392
    assert variants["global_k5"]["mac"]["known_mac_subtotal"] == 15_936_320
    assert variants["matched_post_raster_nearest_union_k1"]["mac"][
        "known_mac_subtotal"
    ] == 3_267_392
    assert variants["global_k5"]["mac"]["known_mac_subtotal"] > variants[
        "global_k1"
    ]["mac"]["known_mac_subtotal"]
    assert variants["global_k1"]["model_state_sha256"] == variants[
        "matched_post_raster_nearest_union_k1"
    ]["model_state_sha256"]


def test_input_schema_distinguishes_k5_and_matched_union(receipt: dict) -> None:
    variants = {item["variant"]: item for item in receipt["variants"]}
    k1 = variants["global_k1"]["input_schema"]
    k5 = variants["global_k5"]["input_schema"]
    matched = variants["matched_post_raster_nearest_union_k1"]["input_schema"]
    assert k1["observation_groups"]["policy"]["shape"] == [1, 96]
    assert k1["observation_groups"]["policy"]["dtype"] == "torch.float32"
    assert k1["observation_groups"]["mid360_policy"]["shape"] == [
        1,
        1,
        2,
        16,
        96,
    ]
    assert k5["observation_groups"]["mid360_policy"]["shape"] == [
        1,
        5,
        2,
        16,
        96,
    ]
    assert matched["observation_groups"]["mid360_policy"]["shape"] == [
        1,
        1,
        2,
        16,
        96,
    ]
    assert matched["observation_groups"]["mid360_policy"]["dtype"] == "torch.float16"
    producer = matched["producer_audit"]
    assert producer["source_history_length"] == 5
    assert producer["actor_history_length"] == 1
    assert len(producer["source_window_sha256"]) == 64
    reduction = producer["post_raster_reduction"]
    assert reduction["history_reduction"] == "exact_union_k1"
    assert reduction["temporal_baseline"] == "age_zero"
    assert reduction["winner_age_and_source_are_actor_invisible"] is True
    assert reduction["reducer_parameters"] == 0
    assert reduction["reducer_mac_included_in_actor_subtotal"] is False


def test_dispatcher_unconverted_operators_remain_fail_visible(receipt: dict) -> None:
    for item in receipt["variants"]:
        mac = item["mac"]
        unconverted = {
            record["operator"]
            for record in mac["coverage"]["unconverted_operator_census"]
        }
        assert {"aten.add.Tensor", "aten.mul.Tensor"}.issubset(unconverted)
        census = mac["dispatcher_census"]
        assert census["accelerator_queries"] is False
        assert census["supported_mac_subtotal"] == mac["known_mac_subtotal"]
        assert "aten.conv1d.default" in census["supported_mac_by_operator"]
        assert "Conv1d" in mac["known_mac_by_module_type"]
        assert "not be renamed total MAC" in mac["coverage"]["reason"]


class _HookFormulaFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(2, 3, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
        self.linear = nn.Linear(4, 5, bias=False)

    def forward(
        self,
        one_d: torch.Tensor,
        two_d: torch.Tensor,
        vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.conv1(one_d), self.conv2(two_d), self.linear(vector)


def test_module_hooks_use_output_shape_mac_formulas() -> None:
    model = _HookFormulaFixture()
    with ModuleMacCounter(model) as counter:
        model(
            torch.zeros(1, 2, 8),
            torch.zeros(1, 2, 4, 5),
            torch.zeros(1, 4),
        )
    summary = counter.summary()
    assert summary["known_mac_by_module_type"] == {
        "Conv1d": 1 * 3 * 8 * 2 * 3,
        "Conv2d": 1 * 3 * 4 * 5 * 2 * 3 * 3,
        "Linear": 1 * 5 * 4,
    }
    assert summary["known_mac_subtotal"] == 1_244
    assert all(record["bias_additions_included"] is False for record in summary["calls"])


def test_fixed_model_and_inputs_are_reproducible() -> None:
    first = construct_global_ray_time_actor("global_k1")
    second = construct_global_ray_time_actor("global_k1")
    torch.testing.assert_close(
        first.observations["policy"],
        second.observations["policy"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.observations["mid360_policy"],
        second.observations["mid360_policy"],
        rtol=0.0,
        atol=0.0,
    )
    for (_, first_parameter), (_, second_parameter) in zip(
        first.model.named_parameters(),
        second.model.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)
    with pytest.raises(ComplexityReceiptError, match="Unknown variant"):
        construct_global_ray_time_actor("attention_k5")


def test_receipt_write_is_create_once(receipt: dict, tmp_path: Path) -> None:
    output = tmp_path / "complexity.json"
    write_complexity_receipt_create_once(output, receipt)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["payload_sha256"] == receipt["payload_sha256"]
    with pytest.raises(ComplexityReceiptError, match="overwrite"):
        write_complexity_receipt_create_once(output, receipt)
