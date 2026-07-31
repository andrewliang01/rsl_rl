# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import numpy as np
import pytest

from rsl_rl.utils.uaba_budget_allocator import (
    UABA_MAX_TOTAL_BUDGET,
    UABA_RECEIPT_SCHEMA,
    UABA_ROLE_NAMES,
    UabaContractError,
    allocate_uaba_role_budgets,
)


def _ample_candidates(batch_size: int, width: int = 40):
    return [
        [
            [batch * 10000 + role * 1000 + index for index in range(width)]
            for role in range(4)
        ]
        for batch in range(batch_size)
    ]


def test_batched_largest_remainder_conserves_every_total():
    demand = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0e308, 1.0e-300, 0.0, 0.0],
        ]
    )
    result = allocate_uaba_role_budgets(
        demand,
        _ample_candidates(3),
        total_budget=np.asarray([7, 6, 8]),
    )

    assert result.target.tolist() == [
        [3, 2, 1, 1],
        [2, 2, 1, 1],
        [8, 0, 0, 0],
    ]
    assert np.array_equal(result.target.sum(axis=1), [7, 6, 8])
    assert np.array_equal(result.realized, result.target)
    assert np.all(result.shortfall == 0)


def test_min_max_prevent_starvation_without_breaking_total_budget():
    result = allocate_uaba_role_budgets(
        [[100.0, 0.0, 0.0, 0.0]],
        _ample_candidates(1),
        total_budget=7,
        per_role_min=[1, 1, 1, 1],
        per_role_max=[2, 2, 2, 2],
    )

    assert result.target.tolist() == [[2, 2, 2, 1]]
    assert int(result.target.sum()) == 7
    assert np.all(result.target >= result.per_role_min)
    assert np.all(result.target <= result.per_role_max)


def test_equal_remainders_use_stable_role_order():
    first = allocate_uaba_role_budgets(
        [[1.0, 1.0, 1.0, 1.0]],
        _ample_candidates(1),
        total_budget=2,
    )
    second = allocate_uaba_role_budgets(
        [[1.0, 1.0, 1.0, 1.0]],
        _ample_candidates(1),
        total_budget=2,
    )

    assert first.target.tolist() == [[1, 1, 0, 0]]
    assert np.array_equal(first.target, second.target)
    assert first.receipt_sha256 == second.receipt_sha256


def test_all_censored_zero_demand_and_no_candidates_are_explicit():
    result = allocate_uaba_role_budgets(
        np.zeros((2, 4)),
        [[[], [], [], []], [[], [], [], []]],
        total_budget=[8, 0],
    )

    assert result.target.tolist() == [[2, 2, 2, 2], [0, 0, 0, 0]]
    assert result.realized.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert result.shortfall.tolist() == [[2, 2, 2, 2], [0, 0, 0, 0]]
    assert result.shortfall_reason == (
        ("no_candidates",) * 4,
        ("target_zero",) * 4,
    )


def test_candidate_shortage_never_reallocates_the_strict_target():
    result = allocate_uaba_role_budgets(
        [[9.0, 1.0, 1.0, 1.0]],
        [[[10], [20, 21], [30], [40]]],
        total_budget=8,
        per_role_min=1,
    )

    assert result.target.tolist() == [[4, 2, 1, 1]]
    assert result.realized.tolist() == [[1, 2, 1, 1]]
    assert result.shortfall.tolist() == [[3, 0, 0, 0]]
    assert result.shortfall_reason[0][0] == "candidate_count_below_target"
    assert int(result.target.sum()) == 8
    assert int(result.realized.sum()) == 5


def test_duplicate_ids_are_deduplicated_and_matching_uses_alternatives():
    result = allocate_uaba_role_budgets(
        [[1.0, 1.0, 1.0, 1.0]],
        [[[1, 1, 1], [1, 2, 2], [3], [4]]],
        total_budget=4,
        per_role_min=1,
        per_role_max=1,
    )

    assert result.target.tolist() == [[1, 1, 1, 1]]
    assert result.realized.tolist() == [[1, 1, 1, 1]]
    assert result.selected_event_ids == (((1,), (2,), (3,), (4,)),)
    flattened = [event for role in result.selected_event_ids[0] for event in role]
    assert len(flattened) == len(set(flattened))
    assert result.within_role_duplicate_count.tolist() == [[2, 1, 0, 0]]
    assert result.candidate_count.tolist() == [[1, 2, 1, 1]]
    assert result.global_unique_candidate_count.tolist() == [4]
    assert result.cross_role_overlap_count.tolist() == [1]


def test_unavoidable_cross_role_duplicate_conflict_has_specific_reason():
    result = allocate_uaba_role_budgets(
        [[1.0, 1.0, 1.0, 1.0]],
        [[[7], [7], [7], [7]]],
        total_budget=4,
        per_role_min=1,
        per_role_max=1,
    )

    assert int(result.realized.sum()) == 1
    assert len({event for role in result.selected_event_ids[0] for event in role}) == 1
    assert result.shortfall_reason[0].count("complete") == 1
    assert result.shortfall_reason[0].count("global_duplicate_conflict") == 3


def test_role_permutation_is_equivariant_when_no_tie_is_present():
    demand = np.asarray([[8.0, 4.0, 2.0, 1.0]])
    candidates = _ample_candidates(1)
    reference = allocate_uaba_role_budgets(
        demand, candidates, total_budget=8
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = allocate_uaba_role_budgets(
        demand[:, permutation],
        [[candidates[0][index] for index in permutation]],
        total_budget=8,
    )
    restored = np.empty(4, dtype=np.int64)
    restored[permutation] = permuted.target[0]

    assert np.array_equal(restored, reference.target[0])


def test_receipts_store_demands_targets_realized_reasons_and_hashes():
    result = allocate_uaba_role_budgets(
        [[0.25, 0.5, 0.75, 1.0]],
        [[[10], [], [30, 31], [40, 41, 42]]],
        total_budget=6,
    )
    receipt = result.receipt(0)

    assert receipt["schema"] == UABA_RECEIPT_SCHEMA
    assert receipt["role_names"] == list(UABA_ROLE_NAMES)
    assert receipt["demand"] == [0.25, 0.5, 0.75, 1.0]
    assert len(receipt["demand_binary64"]) == 4
    assert receipt["target"] == result.target[0].tolist()
    assert receipt["realized"] == result.realized[0].tolist()
    assert receipt["shortfall_reason"] == list(result.shortfall_reason[0])
    assert len(result.receipt_sha256[0]) == 64
    assert len(result.aggregate_sha256) == 64
    json.dumps(receipt, sort_keys=True)


def test_hash_is_independent_of_byte_order_strides_and_sequence_container():
    native = np.asarray(
        [[0.125, 0.25, 0.5, 1.0], [1.5, 1.0, 0.5, 0.25]],
        dtype=np.float64,
    )
    non_native = np.asfortranarray(native.astype(">f8"))
    list_candidates = _ample_candidates(2)
    tuple_candidates = tuple(
        tuple(tuple(role) for role in batch) for batch in list_candidates
    )

    first = allocate_uaba_role_budgets(
        native, list_candidates, total_budget=[7, 9], per_role_min=1
    )
    second = allocate_uaba_role_budgets(
        non_native, tuple_candidates, total_budget=np.asarray([7, 9]), per_role_min=np.int64(1)
    )

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.aggregate_sha256 == second.aggregate_sha256


def test_hash_changes_when_a_semantic_input_changes():
    candidates = _ample_candidates(1)
    first = allocate_uaba_role_budgets(
        [[1.0, 2.0, 3.0, 4.0]], candidates, total_budget=8
    )
    second = allocate_uaba_role_budgets(
        [[1.0, 2.0, 3.0, 4.5]], candidates, total_budget=8
    )

    assert first.receipt_sha256 != second.receipt_sha256
    assert first.aggregate_sha256 != second.aggregate_sha256


def test_result_arrays_are_read_only_and_not_views_of_inputs():
    demand = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    result = allocate_uaba_role_budgets(
        demand, _ample_candidates(1), total_budget=8
    )
    demand[:] = 99.0

    assert result.demand.tolist() == [[1.0, 2.0, 3.0, 4.0]]
    with pytest.raises(ValueError):
        result.target[0, 0] = 7


def test_cpu_first_status_is_explicit_and_not_training_ready():
    result = allocate_uaba_role_budgets(
        [[1.0, 1.0, 1.0, 1.0]], _ample_candidates(1), total_budget=4
    )

    assert result.causal_only is True
    assert result.training_ready is False
    assert result.actor_integrated is False
    assert result.gpu_required is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"total_budget": UABA_MAX_TOTAL_BUDGET + 1}, "total_budget"),
        ({"total_budget": 3, "per_role_min": 1}, "minimums"),
        (
            {"total_budget": 8, "per_role_max": [1, 1, 1, 1]},
            "maximums",
        ),
        (
            {
                "total_budget": 4,
                "per_role_min": [2, 0, 0, 0],
                "per_role_max": [1, 4, 4, 4],
            },
            "cannot exceed",
        ),
    ],
)
def test_infeasible_budget_contracts_are_rejected(kwargs, match):
    with pytest.raises(UabaContractError, match=match):
        allocate_uaba_role_budgets(
            [[1.0, 1.0, 1.0, 1.0]], _ample_candidates(1), **kwargs
        )


@pytest.mark.parametrize(
    "demand",
    [
        [[1.0, 2.0, 3.0]],
        [[1.0, -1.0, 2.0, 3.0]],
        [[1.0, np.nan, 2.0, 3.0]],
        [[1.0, np.inf, 2.0, 3.0]],
        [["future", 1.0, 2.0, 3.0]],
    ],
)
def test_invalid_causal_demands_are_rejected(demand):
    with pytest.raises(UabaContractError):
        allocate_uaba_role_budgets(
            demand, _ample_candidates(1), total_budget=4
        )


@pytest.mark.parametrize(
    "candidates",
    [
        [[[1], [2], [3]]],
        [[[1], [2], [3], [-1]]],
        [[[1], [2], [3], [True]]],
        [[[1], [2], [3], [1.5]]],
    ],
)
def test_invalid_candidate_event_ids_are_rejected(candidates):
    with pytest.raises(UabaContractError):
        allocate_uaba_role_budgets(
            [[1.0, 1.0, 1.0, 1.0]], candidates, total_budget=4
        )
