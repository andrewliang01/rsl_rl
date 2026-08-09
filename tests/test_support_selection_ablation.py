import pytest
import torch

from rsl_rl.modules.support_selection_ablation import (
    FixedBudgetSupportSelector,
)


def _inputs(num_tokens: int = 80):
    scores = torch.arange(
        4 * num_tokens, dtype=torch.float32
    ).reshape(1, 4, num_tokens)
    valid = torch.ones(1, num_tokens, dtype=torch.bool)
    roles = torch.zeros(1, 4, num_tokens, dtype=torch.bool)
    roles[:, 0, :20] = True
    roles[:, 1, 20:40] = True
    roles[:, 2, 40:60] = True
    roles[:, 3, 60:] = True
    return scores, valid, roles


def test_full_keeps_every_valid_token_for_every_query():
    scores, valid, roles = _inputs()
    valid[:, 7] = False
    selector = FixedBudgetSupportSelector(strategy="full", total_budget="all")
    mask, diagnostics = selector(scores, valid, roles)
    assert torch.equal(mask, valid[:, None, :].expand_as(mask))
    assert diagnostics["realized_unique_count"].item() == 79
    assert diagnostics["requested_unique_budget"].item() == 79


@pytest.mark.parametrize("budget", [8, 16, 32, 64])
def test_glad_top_m_spends_one_shared_unique_budget(budget):
    scores, valid, roles = _inputs()
    roles.zero_()  # The global control must not depend on role calibration.
    selector = FixedBudgetSupportSelector(
        strategy="glad_top_m", total_budget=budget
    )
    mask, diagnostics = selector(scores, valid, roles)
    assert mask.any(dim=1).sum().item() == budget
    assert mask.sum().item() == budget
    expected = torch.arange(80 - budget, 80)
    assert torch.equal(
        torch.where(mask.any(dim=1)[0])[0], expected
    )
    assert diagnostics["unique_budget_shortfall"].item() == 0
    assert diagnostics["selector_score_entropy_applicable"].item()
    assert torch.isfinite(diagnostics["selector_score_entropy"]).all()


def test_role_shared_total_m_is_unique_and_role_eligible():
    scores, valid, roles = _inputs()
    # Add overlap; it must not spend a second slot on the same token.
    roles[:, 0, 20:30] = True
    selector = FixedBudgetSupportSelector(
        strategy="role_shared_total_m", total_budget=32
    )
    mask, diagnostics = selector(scores, valid, roles)
    assert mask.sum().item() == 32
    assert mask.any(dim=1).sum().item() == 32
    assert not bool((mask & ~roles).any())
    pairwise = diagnostics["pairwise_role_overlap_count"][0]
    assert torch.equal(pairwise - torch.diag(torch.diag(pairwise)), torch.zeros_like(pairwise))
    assert diagnostics["candidate_opportunity_matched_to_role_selector"].item()


def test_role_shared_total_m_reports_candidate_shortfall():
    scores, valid, roles = _inputs()
    roles.zero_()
    roles[:, 2, :5] = True
    selector = FixedBudgetSupportSelector(
        strategy="role_shared_total_m", total_budget=8
    )
    mask, diagnostics = selector(scores, valid, roles)
    assert mask.sum().item() == 5
    assert diagnostics["unique_budget_shortfall"].item() == 3


def test_matched_random_m_is_seeded_count_and_opportunity_matched():
    scores, valid, roles = _inputs()
    roles[:, :, 64:] = False
    first = FixedBudgetSupportSelector(
        strategy="matched_random_m", total_budget=16
    )
    second = FixedBudgetSupportSelector(
        strategy="matched_random_m", total_budget=16
    )
    mask_a, diag_a = first(scores, valid, roles, random_seed=17)
    mask_b, diag_b = second(scores, valid, roles, random_seed=17)
    assert torch.equal(mask_a, mask_b)
    assert mask_a.sum().item() == 16
    assert not bool((mask_a.any(dim=1) & ~roles.any(dim=1)).any())
    assert diag_a["random_seed"].item() == 17
    assert diag_b["candidate_opportunity_matched_to_role_selector"].item()
    assert not diag_a["selector_score_entropy_applicable"].item()


def test_randomness_contract_is_fail_closed():
    scores, valid, roles = _inputs()
    random_selector = FixedBudgetSupportSelector(
        strategy="matched_random_m", total_budget=8
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        random_selector(scores, valid, roles)
    deterministic_selector = FixedBudgetSupportSelector(
        strategy="role_shared_total_m", total_budget=8
    )
    with pytest.raises(ValueError, match="only"):
        deterministic_selector(scores, valid, roles, random_seed=3)


def test_invalid_scores_outside_valid_support_do_not_matter():
    scores, valid, roles = _inputs()
    valid[:, -1] = False
    scores[:, :, -1] = torch.nan
    selector = FixedBudgetSupportSelector(
        strategy="role_shared_total_m", total_budget=8
    )
    mask, _ = selector(scores, valid, roles)
    assert not mask[:, :, -1].any()
