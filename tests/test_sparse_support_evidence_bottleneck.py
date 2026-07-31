# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from rsl_rl.modules import SparseSupportEvidenceBottleneck
from rsl_rl.modules.sparse_support_evidence_bottleneck import (
    NUM_QUERIES,
    QUERY_NAMES,
)


SCORE_FEATURE_DIM = 5
TERRAIN_VALUE_DIM = 7
PROPRIO_DIM = 6
OUTPUT_DIM = 12
SCORE_DIM = 8
VALUE_EMBEDDING_DIM = 9


def _model(
    *,
    total_budget: int | str = 8,
    mode: str = "selected_only",
    selector_gradient: str = "hard",
) -> SparseSupportEvidenceBottleneck:
    return SparseSupportEvidenceBottleneck(
        score_feature_dim=SCORE_FEATURE_DIM,
        terrain_value_dim=TERRAIN_VALUE_DIM,
        proprio_dim=PROPRIO_DIM,
        output_dim=OUTPUT_DIM,
        total_budget=total_budget,
        score_dim=SCORE_DIM,
        value_embedding_dim=VALUE_EMBEDDING_DIM,
        mode=mode,
        selector_gradient=selector_gradient,
    )


def _inputs(
    *,
    batch_size: int = 2,
    num_tokens: int = 40,
    invalid_tail: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    score_features = torch.randn(
        batch_size,
        num_tokens,
        SCORE_FEATURE_DIM,
    )
    terrain_values = torch.randn(
        batch_size,
        num_tokens,
        TERRAIN_VALUE_DIM,
    )
    proprio = torch.randn(batch_size, PROPRIO_DIM)
    token_valid = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    if invalid_tail:
        token_valid[:, -invalid_tail:] = False
    eligibility = torch.zeros(
        batch_size,
        NUM_QUERIES,
        num_tokens,
        dtype=torch.bool,
    )
    token_ids = torch.arange(num_tokens)
    for query_index in range(NUM_QUERIES):
        eligibility[:, query_index] = (
            token_ids % NUM_QUERIES == query_index
        )[None, :]
    eligibility &= token_valid[:, None, :]
    return (
        score_features,
        terrain_values,
        proprio,
        token_valid,
        eligibility,
    )


def _all_eligible_inputs(
    *,
    batch_size: int = 2,
    num_tokens: int = 16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    score_features, terrain_values, proprio, token_valid, _ = _inputs(
        batch_size=batch_size,
        num_tokens=num_tokens,
    )
    eligibility = token_valid[:, None, :].expand(
        -1,
        NUM_QUERIES,
        -1,
    ).clone()
    return (
        score_features,
        terrain_values,
        proprio,
        token_valid,
        eligibility,
    )


def _zero_scores(model: SparseSupportEvidenceBottleneck) -> None:
    with torch.no_grad():
        model.score_key_projection.weight.zero_()
        model.score_query_projection.weight.zero_()
        model.score_query_projection.bias.zero_()
        model.query_embedding.zero_()


@pytest.mark.parametrize(
    ("total_budget", "quota"),
    ((4, 1), (8, 2), (16, 4), (32, 8)),
)
def test_fixed_total_budget_is_split_across_four_queries(
    total_budget: int,
    quota: int,
) -> None:
    torch.manual_seed(3001 + total_budget)
    model = _model(total_budget=total_budget).eval()
    inputs = _inputs(batch_size=3, num_tokens=40)

    with torch.inference_mode():
        output, diagnostics = model.forward_with_diagnostics(*inputs)

    assert output.shape == (3, OUTPUT_DIM)
    assert diagnostics["selection_indices"].shape == (
        3,
        NUM_QUERIES,
        quota,
    )
    torch.testing.assert_close(
        diagnostics["quota_per_query"],
        torch.full((3, NUM_QUERIES), quota),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["realized_per_query"],
        torch.full((3, NUM_QUERIES), quota),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["realized_slot_count"],
        torch.full((3,), total_budget),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["unique_selected_count"],
        torch.full((3,), total_budget),
        rtol=0.0,
        atol=0.0,
    )
    assert not diagnostics["overlap_count"].any()
    assert bool(
        (
            diagnostics["unique_selected_count"]
            <= total_budget
        ).all()
    )
    assert torch.count_nonzero(
        diagnostics["aggregation_weights"].masked_select(
            ~diagnostics["selection_query_mask"]
        )
    ) == 0
    weight_sum = diagnostics["aggregation_weights"].sum(dim=-1)
    assert bool((weight_sum > 0.0).all())
    assert bool((weight_sum < 1.0).all())


def test_overlapping_queries_report_slots_unique_and_overlap_separately() -> None:
    model = _model(total_budget=8).eval()
    _zero_scores(model)
    inputs = _all_eligible_inputs(batch_size=2, num_tokens=12)

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(*inputs)

    torch.testing.assert_close(
        diagnostics["realized_slot_count"],
        torch.full((2,), 8),
    )
    torch.testing.assert_close(
        diagnostics["unique_selected_count"],
        torch.full((2,), 2),
    )
    torch.testing.assert_close(
        diagnostics["overlap_count"],
        torch.full((2,), 6),
    )
    torch.testing.assert_close(
        diagnostics["pairwise_query_overlap_count"],
        torch.full((2, NUM_QUERIES, NUM_QUERIES), 2),
    )
    expected = torch.tensor([0, 1]).expand(2, NUM_QUERIES, -1)
    torch.testing.assert_close(
        diagnostics["selection_indices"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_insufficient_query_evidence_pads_indices_and_reports_realized() -> None:
    model = _model(total_budget=16).eval()
    score_features = torch.randn(1, 6, SCORE_FEATURE_DIM)
    terrain_values = torch.randn(1, 6, TERRAIN_VALUE_DIM)
    proprio = torch.randn(1, PROPRIO_DIM)
    token_valid = torch.ones(1, 6, dtype=torch.bool)
    eligibility = torch.zeros(1, NUM_QUERIES, 6, dtype=torch.bool)
    eligibility[0, 0, 0] = True
    eligibility[0, 1, 1:3] = True
    eligibility[0, 3, 3:6] = True

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            eligibility,
        )

    assert diagnostics["selection_indices"].shape == (1, NUM_QUERIES, 4)
    torch.testing.assert_close(
        diagnostics["realized_per_query"],
        torch.tensor([[1, 2, 0, 3]]),
    )
    assert bool((diagnostics["selection_indices"][0, 0, 1:] == -1).all())
    assert bool((diagnostics["selection_indices"][0, 2] == -1).all())
    assert diagnostics["aggregation_weights"][0, 2].count_nonzero() == 0
    torch.testing.assert_close(
        diagnostics["unique_selected_count"],
        torch.tensor([6]),
    )


def test_quota_larger_than_token_count_retains_nominal_padded_width() -> None:
    model = _model(total_budget=32).eval()
    inputs = _all_eligible_inputs(batch_size=1, num_tokens=3)

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(*inputs)

    assert diagnostics["selection_indices"].shape == (1, NUM_QUERIES, 8)
    assert bool((diagnostics["selection_indices"][..., 3:] == -1).all())
    torch.testing.assert_close(
        diagnostics["realized_per_query"],
        torch.full((1, NUM_QUERIES), 3),
    )
    torch.testing.assert_close(
        diagnostics["realized_slot_count"],
        torch.tensor([12]),
    )
    torch.testing.assert_close(
        diagnostics["unique_selected_count"],
        torch.tensor([3]),
    )


def test_invalid_high_score_token_is_never_selected() -> None:
    model = _model(total_budget=4).eval()
    with torch.no_grad():
        model.score_key_projection.weight.zero_()
        model.score_key_projection.weight[0, 0] = 1.0
        model.score_query_projection.weight.zero_()
        model.score_query_projection.bias.zero_()
        model.query_embedding.zero_()
        model.query_embedding[..., 0] = 1.0

    inputs = list(_inputs(batch_size=1, num_tokens=8, invalid_tail=1))
    inputs[0].zero_()
    inputs[0][0, -1, 0] = 1.0e6

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(*inputs)

    assert not bool(diagnostics["selection_unique_mask"][0, -1])
    assert bool(torch.isneginf(diagnostics["selection_scores"][0, :, -1]).all())


@pytest.mark.parametrize(
    "mode",
    (
        "full",
        "selected_only",
        "delete_selected",
        "delete_random",
        "zero_terrain",
    ),
)
def test_blackout_returns_proprio_only_stability_without_nan(mode: str) -> None:
    torch.manual_seed(3021)
    model = _model(mode=mode).eval()
    inputs = list(_inputs(batch_size=2, num_tokens=8))
    inputs[3].zero_()
    inputs[4].zero_()
    kwargs = {"random_seed": 71} if mode == "delete_random" else {}

    with torch.inference_mode():
        output, diagnostics = model.forward_with_diagnostics(
            *inputs,
            **kwargs,
        )
        expected = model.output_norm(
            model.stability_projection(inputs[2])
        )

    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(output).all()
    assert diagnostics["selection_unique_mask"].count_nonzero() == 0
    assert diagnostics["effective_unique_mask"].count_nonzero() == 0
    assert diagnostics["aggregation_weights"].count_nonzero() == 0
    assert diagnostics["terrain_contribution"].count_nonzero() == 0
    assert diagnostics["unique_selected_count"].count_nonzero() == 0
    if mode == "delete_random":
        assert diagnostics["random_not_applicable"].all()
        assert not diagnostics["random_primary_matched"].any()
        assert not diagnostics["random_unmatched"].any()
        assert not diagnostics[
            "random_delete_exactly_matches_selected"
        ].any()
    else:
        assert not diagnostics["random_not_applicable"].any()


def test_ties_are_stable_and_choose_lowest_eligible_token_ids() -> None:
    model = _model(total_budget=8).eval()
    _zero_scores(model)
    inputs = _inputs(batch_size=1, num_tokens=16)

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(*inputs)

    expected = torch.tensor(
        [[[0, 4], [1, 5], [2, 6], [3, 7]]],
    )
    torch.testing.assert_close(
        diagnostics["selection_indices"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_all_interventions_freeze_the_same_clean_scores_and_ids() -> None:
    torch.manual_seed(3031)
    model = _model(total_budget=8).eval()
    inputs = _inputs(batch_size=2, num_tokens=20)
    diagnostics_by_mode: dict[str, dict[str, torch.Tensor]] = {}

    with torch.inference_mode():
        for mode in (
            "full",
            "selected_only",
            "delete_selected",
            "delete_random",
            "zero_terrain",
        ):
            kwargs = {"random_seed": 97} if mode == "delete_random" else {}
            _, diagnostics_by_mode[mode] = model.forward_with_diagnostics(
                *inputs,
                mode=mode,
                **kwargs,
            )

    reference = diagnostics_by_mode["selected_only"]
    for diagnostics in diagnostics_by_mode.values():
        torch.testing.assert_close(
            diagnostics["selection_scores"],
            reference["selection_scores"],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            diagnostics["selection_indices"],
            reference["selection_indices"],
            rtol=0.0,
            atol=0.0,
        )
        assert diagnostics["clean_selection_frozen"].all()
        assert diagnostics["no_reselection"].all()

    full_valid = inputs[3][:, None, :].expand(-1, NUM_QUERIES, -1)
    torch.testing.assert_close(
        diagnostics_by_mode["full"]["effective_query_mask"],
        full_valid,
    )
    torch.testing.assert_close(
        diagnostics_by_mode["selected_only"]["effective_query_mask"],
        reference["selection_query_mask"],
    )
    expected_delete_selected = (
        full_valid
        & ~reference["selection_unique_mask"][:, None, :]
    )
    torch.testing.assert_close(
        diagnostics_by_mode["delete_selected"]["effective_query_mask"],
        expected_delete_selected,
    )
    assert (
        diagnostics_by_mode["zero_terrain"]["effective_query_mask"]
        .count_nonzero()
        == 0
    )


def test_delete_selected_and_random_use_same_unique_budget_without_reselection() -> None:
    torch.manual_seed(3037)
    model = _model(total_budget=16).eval()
    inputs = _all_eligible_inputs(batch_size=3, num_tokens=40)

    with torch.inference_mode():
        _, selected = model.forward_with_diagnostics(
            *inputs,
            mode="delete_selected",
        )
        _, random = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            random_seed=811,
        )

    torch.testing.assert_close(
        selected["deletion_unique_budget"],
        random["deletion_unique_budget"],
    )
    torch.testing.assert_close(
        selected["deletion_unique_realized_count"],
        selected["deletion_unique_budget"],
    )
    torch.testing.assert_close(
        random["deletion_unique_realized_count"],
        random["deletion_unique_budget"],
    )
    assert selected["deletion_unique_budget_match"].all()
    assert random["deletion_unique_budget_match"].all()
    # Nominal query slots are audit metadata, not a deletion constraint.
    assert bool(
        (
            selected["realized_slot_count"]
            >= selected["deletion_unique_budget"]
        ).all()
    )
    assert not bool(
        (
            random["random_delete_mask"]
            & ~inputs[3]
        ).any()
    )
    for batch_index in range(3):
        valid_indices = random["random_delete_indices"][
            batch_index,
            random["random_delete_slot_valid"][batch_index],
        ]
        assert valid_indices.unique().numel() == valid_indices.numel()
    expected_intersection = (
        random["random_delete_mask"] & random["selection_unique_mask"]
    ).sum(dim=-1)
    torch.testing.assert_close(
        random["random_selected_intersection_count"],
        expected_intersection,
    )
    assert random["random_primary_matched"].all()
    assert not random["random_unmatched"].any()
    assert random["random_selected_intersection_count"].count_nonzero() == 0


def test_deletion_matches_unique_not_nominal_slots_when_queries_overlap() -> None:
    model = _model(total_budget=8).eval()
    _zero_scores(model)
    inputs = _all_eligible_inputs(batch_size=2, num_tokens=20)

    with torch.inference_mode():
        _, selected = model.forward_with_diagnostics(
            *inputs,
            mode="delete_selected",
        )
        _, random = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            random_seed=919,
        )

    torch.testing.assert_close(
        selected["nominal_total_slot_budget"],
        torch.full((2,), 8),
    )
    torch.testing.assert_close(
        selected["realized_slot_count"],
        torch.full((2,), 8),
    )
    torch.testing.assert_close(
        selected["unique_selected_count"],
        torch.full((2,), 2),
    )
    torch.testing.assert_close(
        selected["deletion_unique_budget"],
        torch.full((2,), 2),
    )
    torch.testing.assert_close(
        selected["deletion_unique_realized_count"],
        torch.full((2,), 2),
    )
    torch.testing.assert_close(
        random["deletion_unique_budget"],
        torch.full((2,), 2),
    )
    torch.testing.assert_close(
        random["deletion_unique_realized_count"],
        torch.full((2,), 2),
    )
    assert selected["deletion_unique_budget_match"].all()
    assert random["deletion_unique_budget_match"].all()


def test_seeded_random_deletion_is_reproducible_and_seed_sensitive() -> None:
    torch.manual_seed(3041)
    model = _model(total_budget=8).eval()
    inputs = _inputs(batch_size=2, num_tokens=40)

    with torch.inference_mode():
        _, first = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            random_seed=1234,
        )
        _, repeated = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            random_seed=1234,
        )
        _, different = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            random_seed=1235,
        )

    torch.testing.assert_close(
        first["random_delete_mask"],
        repeated["random_delete_mask"],
    )
    torch.testing.assert_close(
        first["random_delete_indices"],
        repeated["random_delete_indices"],
    )
    assert not torch.equal(
        first["random_delete_mask"],
        different["random_delete_mask"],
    )
    torch.testing.assert_close(
        first["random_seed"],
        torch.full((2,), 1234),
    )


def test_explicit_generator_state_is_used_and_a_fresh_generator_replays() -> None:
    torch.manual_seed(3049)
    model = _model(total_budget=8).eval()
    inputs = _inputs(batch_size=2, num_tokens=40)
    generator = torch.Generator().manual_seed(991)

    with torch.inference_mode():
        _, first = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            generator=generator,
        )
        _, advanced = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            generator=generator,
        )
        _, replayed = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            generator=torch.Generator().manual_seed(991),
        )

    assert not torch.equal(
        first["random_delete_mask"],
        advanced["random_delete_mask"],
    )
    torch.testing.assert_close(
        first["random_delete_mask"],
        replayed["random_delete_mask"],
    )
    assert bool((first["random_seed"] == -1).all())


def test_random_shortfall_is_unmatched_without_touching_selected() -> None:
    model = _model(total_budget=4).eval()
    inputs = _inputs(batch_size=2, num_tokens=4)

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(
            *inputs,
            mode="delete_random",
            random_seed=17,
        )

    assert not diagnostics["random_delete_exactly_matches_selected"].any()
    torch.testing.assert_close(
        diagnostics["random_selected_intersection_count"],
        torch.zeros(2, dtype=torch.long),
    )
    torch.testing.assert_close(
        diagnostics["deletion_unique_realized_count"],
        torch.zeros(2, dtype=torch.long),
    )
    assert not diagnostics["deletion_unique_budget_match"].any()
    assert not diagnostics["random_primary_matched"].any()
    assert diagnostics["random_unmatched"].all()
    torch.testing.assert_close(
        diagnostics["random_candidate_shortfall"],
        torch.full((2,), 4),
    )


def test_all_budget_selected_only_equals_full_upper_bound() -> None:
    torch.manual_seed(3061)
    model = _model(total_budget="all").eval()
    inputs = _inputs(batch_size=2, num_tokens=13, invalid_tail=1)

    with torch.inference_mode():
        selected_output, selected = model.forward_with_diagnostics(
            *inputs,
            mode="selected_only",
        )
        full_output, full = model.forward_with_diagnostics(
            *inputs,
            mode="full",
        )

    torch.testing.assert_close(
        selected_output,
        full_output,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        selected["selection_unique_mask"],
        inputs[3],
    )
    full_valid = inputs[3][:, None, :].expand(-1, NUM_QUERIES, -1)
    torch.testing.assert_close(
        selected["selection_query_mask"],
        full_valid,
    )
    torch.testing.assert_close(
        selected["quota_per_query"],
        full_valid.sum(dim=-1),
    )
    assert selected["selection_indices"].shape == (2, NUM_QUERIES, 13)
    torch.testing.assert_close(
        selected["aggregation_weights"],
        full["aggregation_weights"],
        rtol=0.0,
        atol=0.0,
    )


def test_full_and_delete_operate_on_all_valid_not_only_support_eligible() -> None:
    torch.manual_seed(3063)
    model = _model(total_budget=4).eval()
    inputs = list(_inputs(batch_size=2, num_tokens=12))
    uncovered_token = 11
    inputs[4][:, :, uncovered_token] = False

    with torch.inference_mode():
        selected_output, selected = model.forward_with_diagnostics(
            *inputs,
            mode="selected_only",
        )
        full_output, full = model.forward_with_diagnostics(
            *inputs,
            mode="full",
        )
        deleted_output, deleted = model.forward_with_diagnostics(
            *inputs,
            mode="delete_selected",
        )
        changed_values = inputs[1].clone()
        changed_values[:, uncovered_token] += 100.0
        selected_changed = model(
            inputs[0],
            changed_values,
            inputs[2],
            inputs[3],
            inputs[4],
            mode="selected_only",
        )
        full_changed = model(
            inputs[0],
            changed_values,
            inputs[2],
            inputs[3],
            inputs[4],
            mode="full",
        )

    assert not bool(selected["selection_candidate_mask"][:, :, uncovered_token].any())
    assert not bool(selected["selection_unique_mask"][:, uncovered_token].any())
    assert full["effective_query_mask"][:, :, uncovered_token].all()
    assert deleted["effective_query_mask"][:, :, uncovered_token].all()
    torch.testing.assert_close(
        selected_changed,
        selected_output,
        rtol=0.0,
        atol=0.0,
    )
    assert float((full_changed - full_output).abs().max()) > 1.0e-6
    assert torch.isfinite(deleted_output).all()


def test_uncovered_valid_score_controls_full_and_delete_but_not_sparse() -> None:
    model = _model(total_budget=4).eval()
    with torch.no_grad():
        model.score_key_projection.weight.zero_()
        model.score_key_projection.weight[0, 0] = 1.0
        model.score_query_projection.weight.zero_()
        model.score_query_projection.bias.zero_()
        model.query_embedding.zero_()
        model.query_embedding[..., 0] = 1.0
    inputs = list(_inputs(batch_size=1, num_tokens=12))
    uncovered_token = 11
    inputs[4][:, :, uncovered_token] = False
    inputs[0].zero_()

    with torch.inference_mode():
        sparse_output, sparse = model.forward_with_diagnostics(
            *inputs,
            mode="selected_only",
        )
        full_output, full = model.forward_with_diagnostics(
            *inputs,
            mode="full",
        )
        delete_output, deleted = model.forward_with_diagnostics(
            *inputs,
            mode="delete_selected",
        )
        changed_scores = inputs[0].clone()
        changed_scores[:, uncovered_token, 0] = 8.0
        sparse_changed, sparse_after = model.forward_with_diagnostics(
            changed_scores,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            mode="selected_only",
        )
        full_changed, full_after = model.forward_with_diagnostics(
            changed_scores,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            mode="full",
        )
        delete_changed, deleted_after = model.forward_with_diagnostics(
            changed_scores,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            mode="delete_selected",
        )

    assert not bool(sparse["selection_candidate_mask"][..., uncovered_token].any())
    assert full["effective_query_mask"][..., uncovered_token].all()
    assert deleted["effective_query_mask"][..., uncovered_token].all()
    assert torch.equal(sparse_changed, sparse_output)
    assert torch.equal(
        sparse_after["aggregation_weights"],
        sparse["aggregation_weights"],
    )
    assert not torch.equal(
        full_after["aggregation_weights"],
        full["aggregation_weights"],
    )
    assert not torch.equal(
        deleted_after["aggregation_weights"],
        deleted["aggregation_weights"],
    )
    assert float((full_changed - full_output).abs().max()) > 1.0e-6
    assert float((delete_changed - delete_output).abs().max()) > 1.0e-6


@pytest.mark.parametrize("selector_gradient", ("hard", "straight_through"))
def test_unselected_terrain_values_have_no_forward_leakage(
    selector_gradient: str,
) -> None:
    torch.manual_seed(3067)
    model = _model(
        total_budget=4,
        selector_gradient=selector_gradient,
    ).eval()
    inputs = list(_inputs(batch_size=2, num_tokens=24))

    with torch.inference_mode():
        original_output, original = model.forward_with_diagnostics(*inputs)
        changed_values = inputs[1].clone()
        unselected = ~original["selection_unique_mask"]
        changed_values[unselected] += 1.0e4
        changed_output, changed = model.forward_with_diagnostics(
            inputs[0],
            changed_values,
            inputs[2],
            inputs[3],
            inputs[4],
        )

    torch.testing.assert_close(
        changed["selection_indices"],
        original["selection_indices"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        changed_output,
        original_output,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(
        original["aggregation_weights"].masked_select(
            ~original["selection_query_mask"]
        )
    ) == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    ("mode", "random_seed"),
    (
        ("selected_only", None),
        ("full", None),
        ("delete_selected", None),
        ("delete_random", 707),
    ),
)
def test_extreme_finite_ineffective_values_are_zeroed_before_projection(
    dtype: torch.dtype,
    mode: str,
    random_seed: int | None,
) -> None:
    torch.manual_seed(3069)
    model = _model(total_budget=8).to(dtype=dtype).eval()
    with torch.no_grad():
        model.value_projection.weight.fill_(2.0)
    raw_inputs = _inputs(batch_size=2, num_tokens=24, invalid_tail=1)
    inputs = (
        raw_inputs[0].to(dtype=dtype),
        raw_inputs[1].to(dtype=dtype),
        raw_inputs[2].to(dtype=dtype),
        raw_inputs[3],
        raw_inputs[4],
    )
    kwargs = {} if random_seed is None else {"random_seed": random_seed}

    with torch.inference_mode():
        original_output, diagnostics = model.forward_with_diagnostics(
            *inputs,
            mode=mode,
            **kwargs,
        )
        ineffective = ~diagnostics["effective_unique_mask"]
        assert bool(ineffective.any())
        changed_values = inputs[1].clone()
        changed_values[ineffective] = torch.finfo(dtype).max
        changed_output = model(
            inputs[0],
            changed_values,
            inputs[2],
            inputs[3],
            inputs[4],
            mode=mode,
            **kwargs,
        )

    assert torch.equal(changed_output, original_output)


def test_extreme_finite_effective_value_projection_fails_closed() -> None:
    model = _model(total_budget=4).eval()
    with torch.no_grad():
        model.value_projection.weight.fill_(2.0)
    inputs = list(_inputs(batch_size=1, num_tokens=12))

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(*inputs)
    effective = diagnostics["effective_unique_mask"]
    inputs[1][effective] = torch.finfo(torch.float32).max

    with pytest.raises(ValueError, match="terrain value projection"):
        model(*inputs)


def test_extreme_finite_candidate_score_projection_fails_closed() -> None:
    model = _model(total_budget=4).eval()
    with torch.no_grad():
        model.score_key_projection.weight.fill_(2.0)
        model.score_query_projection.weight.zero_()
        model.score_query_projection.bias.zero_()
        model.query_embedding.fill_(1.0)
    inputs = list(_all_eligible_inputs(batch_size=1, num_tokens=8))
    inputs[0].fill_(torch.finfo(torch.float32).max)

    with pytest.raises(ValueError, match="score projection"):
        model(*inputs)


def test_extreme_finite_terrain_output_projection_fails_closed() -> None:
    model = _model(total_budget=4).eval()
    inputs = list(_all_eligible_inputs(batch_size=1, num_tokens=8))
    inputs[1].fill_(1.0)
    with torch.no_grad():
        model.value_projection.weight.fill_(1.0)
        model.terrain_output_projection.weight.fill_(
            torch.finfo(torch.float32).max
        )

    with pytest.raises(ValueError, match="Terrain output projection"):
        model(*inputs)


def test_extreme_finite_stability_projection_fails_closed() -> None:
    model = _model(total_budget=4).eval()
    inputs = list(_inputs(batch_size=1, num_tokens=8))
    inputs[2].fill_(1.0)
    with torch.no_grad():
        model.stability_projection[0].weight.fill_(
            torch.finfo(torch.float32).max
        )

    with pytest.raises(ValueError, match="Stability projection"):
        model(*inputs, mode="zero_terrain")


def test_extreme_finite_output_normalization_fails_closed() -> None:
    model = _model(total_budget=4).eval()
    inputs = _inputs(batch_size=1, num_tokens=8)
    with torch.no_grad():
        model.output_norm.weight.fill_(torch.finfo(torch.float32).max)
        model.output_norm.bias.fill_(torch.finfo(torch.float32).max)

    with pytest.raises(ValueError, match="Output normalization"):
        model(*inputs, mode="zero_terrain")


def test_selected_value_change_affects_selected_only_output() -> None:
    torch.manual_seed(3071)
    model = _model(total_budget=4).eval()
    inputs = list(_inputs(batch_size=2, num_tokens=24))

    with torch.inference_mode():
        original_output, diagnostics = model.forward_with_diagnostics(*inputs)
        changed_values = inputs[1].clone()
        changed_values[diagnostics["selection_unique_mask"]] += 5.0
        changed_output = model(
            inputs[0],
            changed_values,
            inputs[2],
            inputs[3],
            inputs[4],
        )

    assert float((changed_output - original_output).abs().max()) > 1.0e-6


def test_straight_through_unselected_scores_do_not_leak_into_forward() -> None:
    model = _model(
        total_budget=4,
        selector_gradient="straight_through",
    ).eval()
    with torch.no_grad():
        model.score_key_projection.weight.zero_()
        model.score_key_projection.weight[0, 0] = 1.0
        model.score_query_projection.weight.zero_()
        model.score_query_projection.bias.zero_()
        model.query_embedding.zero_()
        model.query_embedding[..., 0] = 1.0

    inputs = list(_all_eligible_inputs(batch_size=2, num_tokens=12))
    inputs[0].zero_()
    descending_scores = 10.0 * torch.arange(12, 0, -1).float()
    inputs[0][:, :, 0] = descending_scores

    with torch.inference_mode():
        original_output, original = model.forward_with_diagnostics(*inputs)
        changed_scores = inputs[0].clone()
        unselected = ~original["selection_unique_mask"]
        changed_scores[..., 0][unselected] += 0.25
        changed_output, changed = model.forward_with_diagnostics(
            changed_scores,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        )

    assert torch.equal(
        changed["selection_indices"],
        original["selection_indices"],
    )
    assert not torch.equal(
        changed["score_probabilities"],
        original["score_probabilities"],
    )
    assert torch.equal(changed_output, original_output)
    assert torch.equal(
        changed["aggregation_weights"],
        original["aggregation_weights"],
    )


def test_zero_terrain_ignores_both_scores_and_values() -> None:
    torch.manual_seed(3079)
    model = _model().eval()
    inputs = list(_inputs(batch_size=2, num_tokens=20))

    with torch.inference_mode():
        first = model(*inputs, mode="zero_terrain")
        modified_scores = inputs[0] + 1000.0 * torch.randn_like(inputs[0])
        modified_values = inputs[1] + 1000.0 * torch.randn_like(inputs[1])
        modified = model(
            modified_scores,
            modified_values,
            inputs[2],
            inputs[3],
            inputs[4],
            mode="zero_terrain",
        )

    torch.testing.assert_close(modified, first, rtol=0.0, atol=0.0)


def test_straight_through_matches_hard_forward_and_extends_score_gradient() -> None:
    torch.manual_seed(3081)
    hard = _model(
        total_budget=4,
        selector_gradient="hard",
    ).train()
    straight_through = _model(
        total_budget=4,
        selector_gradient="straight_through",
    ).train()
    straight_through.load_state_dict(hard.state_dict(), strict=True)
    base_inputs = _all_eligible_inputs(batch_size=2, num_tokens=12)
    target = torch.randn(2, OUTPUT_DIM)

    hard_scores = base_inputs[0].clone().requires_grad_()
    hard_values = base_inputs[1].clone().requires_grad_()
    hard_output, hard_diagnostics = hard.forward_with_diagnostics(
        hard_scores,
        hard_values,
        base_inputs[2],
        base_inputs[3],
        base_inputs[4],
    )
    F.mse_loss(hard_output, target).backward()

    st_scores = base_inputs[0].clone().requires_grad_()
    st_values = base_inputs[1].clone().requires_grad_()
    st_output, st_diagnostics = straight_through.forward_with_diagnostics(
        st_scores,
        st_values,
        base_inputs[2],
        base_inputs[3],
        base_inputs[4],
    )
    F.mse_loss(st_output, target).backward()

    assert torch.equal(st_output, hard_output)
    assert torch.equal(
        st_diagnostics["aggregation_weights"],
        hard_diagnostics["aggregation_weights"],
    )
    torch.testing.assert_close(
        st_diagnostics["selection_indices"],
        hard_diagnostics["selection_indices"],
    )
    unselected = ~hard_diagnostics["selection_unique_mask"]
    assert hard_scores.grad is not None
    assert st_scores.grad is not None
    assert hard_values.grad is not None
    assert st_values.grad is not None
    selected = hard_diagnostics["selection_unique_mask"]
    assert float(hard_scores.grad[selected].abs().sum()) > 0.0
    assert torch.count_nonzero(hard_scores.grad[unselected]) == 0
    assert float(st_scores.grad[unselected].abs().sum()) > 0.0
    assert torch.count_nonzero(hard_values.grad[unselected]) == 0
    assert torch.count_nonzero(st_values.grad[unselected]) == 0
    assert torch.count_nonzero(
        st_diagnostics["aggregation_weights"].masked_select(
            ~st_diagnostics["selection_query_mask"]
        )
    ) == 0


def test_selected_only_backward_reaches_score_value_stability_and_output() -> None:
    torch.manual_seed(3089)
    model = _model(total_budget=8).train()
    inputs = list(_inputs(batch_size=3, num_tokens=24))
    inputs[0].requires_grad_()
    inputs[1].requires_grad_()
    inputs[2].requires_grad_()
    target = torch.randn(3, OUTPUT_DIM)

    output = model(*inputs)
    F.mse_loss(output, target).backward()

    for tensor in inputs[:3]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0
    prefixes = (
        "score_key_projection",
        "score_query_projection",
        "query_embedding",
        "value_projection",
        "terrain_output_projection",
        "stability_projection",
        "output_norm",
    )
    named_parameters = dict(model.named_parameters())
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in named_parameters.items()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_score_entropy_matches_uniform_ties_and_empty_query() -> None:
    model = _model(total_budget=4).eval()
    _zero_scores(model)
    score_features = torch.randn(1, 4, SCORE_FEATURE_DIM)
    terrain_values = torch.randn(1, 4, TERRAIN_VALUE_DIM)
    proprio = torch.randn(1, PROPRIO_DIM)
    token_valid = torch.ones(1, 4, dtype=torch.bool)
    eligibility = torch.zeros(1, NUM_QUERIES, 4, dtype=torch.bool)
    eligibility[0, 0] = True
    eligibility[0, 1, :2] = True
    eligibility[0, 2, 2:] = True

    with torch.inference_mode():
        _, diagnostics = model.forward_with_diagnostics(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            eligibility,
        )

    torch.testing.assert_close(
        diagnostics["score_entropy"],
        torch.tensor([[math.log(4.0), math.log(2.0), math.log(2.0), 0.0]]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        diagnostics["score_probabilities"].sum(dim=-1),
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    )


def test_diagnostics_have_auditable_shapes_and_counts() -> None:
    model = _model(total_budget=8).eval()
    inputs = _inputs(batch_size=3, num_tokens=17, invalid_tail=1)

    with torch.inference_mode():
        output, diagnostics = model.forward_with_diagnostics(*inputs)

    expected_keys = {
        "selection_indices",
        "selection_slot_valid",
        "selection_query_mask",
        "selection_unique_mask",
        "selection_candidate_mask",
        "selection_scores",
        "score_probabilities",
        "score_entropy",
        "quota_per_query",
        "nominal_total_slot_budget",
        "realized_per_query",
        "realized_slot_count",
        "nominal_slot_shortfall",
        "unique_selected_count",
        "overlap_count",
        "pairwise_query_overlap_count",
        "eligible_count_per_query",
        "valid_token_count",
        "effective_query_mask",
        "effective_unique_mask",
        "aggregation_weights",
        "deleted_mask",
        "deletion_unique_budget",
        "deletion_unique_realized_count",
        "deletion_unique_budget_match",
        "random_delete_mask",
        "random_delete_indices",
        "random_delete_slot_valid",
        "random_selected_intersection_count",
        "random_preferred_candidate_count",
        "random_candidate_shortfall",
        "random_not_applicable",
        "random_primary_matched",
        "random_unmatched",
        "random_delete_exactly_matches_selected",
        "random_seed",
        "clean_selection_frozen",
        "no_reselection",
        "query_values",
        "terrain_contribution",
        "stability_contribution",
    }
    assert set(diagnostics) == expected_keys
    assert output.shape == (3, OUTPUT_DIM)
    assert diagnostics["selection_scores"].shape == (3, NUM_QUERIES, 17)
    assert diagnostics["score_entropy"].shape == (3, NUM_QUERIES)
    assert diagnostics["query_values"].shape == (
        3,
        NUM_QUERIES,
        VALUE_EMBEDDING_DIM,
    )
    assert diagnostics["terrain_contribution"].shape == (3, OUTPUT_DIM)
    assert diagnostics["stability_contribution"].shape == (3, OUTPUT_DIM)
    torch.testing.assert_close(
        diagnostics["realized_slot_count"],
        diagnostics["realized_per_query"].sum(dim=-1),
    )
    torch.testing.assert_close(
        diagnostics["unique_selected_count"],
        diagnostics["selection_unique_mask"].sum(dim=-1),
    )
    torch.testing.assert_close(
        diagnostics["overlap_count"],
        diagnostics["realized_slot_count"]
        - diagnostics["unique_selected_count"],
    )


def test_state_dict_round_trip_and_eval_determinism() -> None:
    torch.manual_seed(3101)
    first = _model(total_budget=16).eval()
    second = _model(total_budget=16).eval()
    second.load_state_dict(first.state_dict(), strict=True)
    inputs = _inputs(batch_size=3, num_tokens=24)

    with torch.inference_mode():
        first_output, first_diagnostics = first.forward_with_diagnostics(*inputs)
        repeated_output, repeated = first.forward_with_diagnostics(*inputs)
        second_output, second_diagnostics = second.forward_with_diagnostics(*inputs)

    torch.testing.assert_close(repeated_output, first_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second_output, first_output, rtol=0.0, atol=0.0)
    for key in (
        "selection_indices",
        "selection_scores",
        "selection_unique_mask",
        "aggregation_weights",
        "score_entropy",
    ):
        torch.testing.assert_close(
            repeated[key],
            first_diagnostics[key],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            second_diagnostics[key],
            first_diagnostics[key],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


def test_double_dtype_batch_and_parameter_device_contract() -> None:
    model = _model().double().eval()
    inputs = _inputs(batch_size=4, num_tokens=19)
    double_inputs = (
        inputs[0].double(),
        inputs[1].double(),
        inputs[2].double(),
        inputs[3],
        inputs[4],
    )
    with torch.inference_mode():
        output = model(*double_inputs)
    assert output.dtype == torch.float64
    assert output.shape == (4, OUTPUT_DIM)
    assert torch.isfinite(output).all()

    meta_model = _model().to("meta")
    with pytest.raises(ValueError, match="module parameters"):
        meta_model(*inputs)


def test_invalid_input_contracts_fail_closed() -> None:
    model = _model()
    score_features, terrain_values, proprio, token_valid, eligibility = _inputs(
        batch_size=2,
        num_tokens=12,
    )

    with pytest.raises(ValueError, match=r"shape \[B, N, S\]"):
        model(
            score_features[..., :-1],
            terrain_values,
            proprio,
            token_valid,
            eligibility,
        )
    with pytest.raises(ValueError, match=r"shape \[B, N, V\]"):
        model(
            score_features,
            terrain_values[..., :-1],
            proprio,
            token_valid,
            eligibility,
        )
    with pytest.raises(ValueError, match=r"shape \[B, P\]"):
        model(
            score_features,
            terrain_values,
            proprio[..., :-1],
            token_valid,
            eligibility,
        )
    with pytest.raises(ValueError, match="token_valid must be boolean"):
        model(
            score_features,
            terrain_values,
            proprio,
            token_valid.float(),
            eligibility,
        )
    with pytest.raises(ValueError, match="query_eligibility must be boolean"):
        model(
            score_features,
            terrain_values,
            proprio,
            token_valid,
            eligibility.float(),
        )

    invalid_eligibility = eligibility.clone()
    token_valid_with_invalid = token_valid.clone()
    token_valid_with_invalid[:, -1] = False
    with pytest.raises(ValueError, match="cannot include invalid"):
        model(
            score_features,
            terrain_values,
            proprio,
            token_valid_with_invalid,
            invalid_eligibility,
        )

    nonfinite_scores = score_features.clone()
    nonfinite_scores[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(
            nonfinite_scores,
            terrain_values,
            proprio,
            token_valid,
            eligibility,
        )

    with pytest.raises(ValueError, match="same device"):
        model(
            score_features,
            terrain_values.to("meta"),
            proprio,
            token_valid,
            eligibility,
        )


def test_delete_random_requires_exactly_one_explicit_rng_source() -> None:
    model = _model()
    inputs = _inputs()

    with pytest.raises(ValueError, match="exactly one"):
        model(*inputs, mode="delete_random")
    with pytest.raises(ValueError, match="exactly one"):
        model(
            *inputs,
            mode="delete_random",
            random_seed=1,
            generator=torch.Generator(),
        )
    with pytest.raises(ValueError, match="random_seed must be an integer"):
        model(*inputs, mode="delete_random", random_seed=True)
    with pytest.raises(ValueError, match="0 <= seed"):
        model(*inputs, mode="delete_random", random_seed=-1)
    with pytest.raises(ValueError, match="only valid"):
        model(*inputs, mode="selected_only", random_seed=1)


def test_construction_and_mode_contracts() -> None:
    model = _model()
    assert model.query_names == (
        "left_near",
        "left_far",
        "right_near",
        "right_far",
    )
    assert QUERY_NAMES == model.query_names
    assert model.per_query_quota == 2
    assert _model(total_budget="all").per_query_quota is None
    assert _model(total_budget=64).per_query_quota == 16

    for bad_budget in (0, 5, 12, 60, True, 4.0, "everything"):
        with pytest.raises(ValueError, match="total_budget"):
            _model(total_budget=bad_budget)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_dim must be at least 2"):
        SparseSupportEvidenceBottleneck(
            SCORE_FEATURE_DIM,
            TERRAIN_VALUE_DIM,
            PROPRIO_DIM,
            1,
        )
    with pytest.raises(ValueError, match="must be an integer"):
        SparseSupportEvidenceBottleneck(
            SCORE_FEATURE_DIM,
            TERRAIN_VALUE_DIM,
            PROPRIO_DIM,
            8.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="mode must be one of"):
        _model(mode="unknown")
    with pytest.raises(ValueError, match="selector_gradient must be one of"):
        _model(selector_gradient="soft")

    inputs = _inputs()
    with pytest.raises(ValueError, match="mode must be one of"):
        model(*inputs, mode="unknown")


def test_no_named_global_or_full_terrain_bypass_parameters_exist() -> None:
    model = _model()
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert not any("global" in name for name in parameter_names)
    assert not any("full" in name for name in parameter_names)
    assert model.terrain_output_projection.in_features == (
        NUM_QUERIES * VALUE_EMBEDDING_DIM
    )
    assert model.terrain_output_projection.bias is None
