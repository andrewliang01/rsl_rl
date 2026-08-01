from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from rsl_rl.modules.cteq_dual_event_hazard import (
    CteqAdministrativeSurvivalLoss,
)
from rsl_rl.utils.cteq_administrative_censor import (
    administrative_censor_survival_loss,
    build_cteq_administrative_censor_batch,
    cteq_administrative_loss_status,
)
from rsl_rl.utils.cteq_contact_timing import (
    CTEQ_BIN_WIDTH_S,
    CTEQ_NUM_BINS,
    CteqContractError,
    build_independent_event_labels,
    debounce_contact_trace,
    dual_event_hazard_from_logits,
    dual_event_survival_loss,
)


def _boundary_kwargs(exposure, reason):
    exposure_array = np.asarray(exposure, dtype=np.int64)
    batch_size = exposure_array.size
    timeout = np.asarray([item == "time_limit" for item in reason], dtype=np.bool_)
    base = np.asarray([item == "base_contact" for item in reason], dtype=np.bool_)
    other = np.asarray([item == "other" for item in reason], dtype=np.bool_)
    return {
        "fully_observed_bins": exposure_array,
        "episode_done": timeout | base | other,
        "time_limit": timeout,
        "base_contact_termination": base,
        "other_early_termination": other,
    }


def _mixed_labels():
    event_bin = np.full((3, 2, 2), -1, dtype=np.int64)
    event_bin[0, 0, 0] = 4
    event_bin[0, 1, 1] = 8
    event_bin[1, 0, 1] = 2
    return build_cteq_administrative_censor_batch(
        event_bin,
        **_boundary_kwargs(
            [CTEQ_NUM_BINS, 6, 0],
            ["none", "time_limit", "base_contact"],
        ),
    )


def _torch_labels(labels):
    return (
        torch.from_numpy(labels.event_bin.copy()),
        torch.from_numpy(labels.event_observed.copy()),
        torch.from_numpy(labels.censor_after_bin.copy()),
        torch.from_numpy(labels.loss_eligible.copy()),
    )


def test_constant_hazard_uses_event_bin_and_strict_censor_prefix():
    labels = _mixed_labels()
    logits = np.zeros((3, 2, 2, CTEQ_NUM_BINS), dtype=np.float64)
    loss = administrative_censor_survival_loss(
        dual_event_hazard_from_logits(logits), labels
    )

    log_two = math.log(2.0)
    assert loss.per_target_nll[0, 0, 0] == pytest.approx(5 * log_two)
    assert loss.per_target_nll[0, 0, 1] == pytest.approx(25 * log_two)
    assert loss.per_target_nll[0, 1, 1] == pytest.approx(9 * log_two)
    assert loss.per_target_nll[1, 0, 0] == pytest.approx(6 * log_two)
    assert loss.per_target_nll[1, 0, 1] == pytest.approx(3 * log_two)
    prefix_probability = np.power(0.5, np.arange(1, 7))
    expected_censor_brier = np.square(prefix_probability).sum() + np.square(
        np.power(0.5, 6) - 1.0
    )
    assert loss.per_target_brier[1, 0, 0] == pytest.approx(
        expected_censor_brier
    )
    assert not loss.eligible_mask[2].any()
    assert np.count_nonzero(loss.per_target_nll[2]) == 0
    assert np.count_nonzero(loss.per_target_brier[2]) == 0
    assert loss.eligible_target_count == 8
    assert loss.excluded_target_count == 4
    assert loss.per_role_eligible_count.shape == (2, 2)
    assert loss.per_role_eligible_count.sum() == 8
    assert loss.td_event_count == 1
    assert loss.lo_event_count == 2
    assert loss.td_censored_count == 3
    assert loss.lo_censored_count == 2
    assert loss.mean_nll == pytest.approx(loss.per_target_nll.sum() / 8)
    assert loss.mean_brier == pytest.approx(loss.per_target_brier.sum() / 8)


def test_full_horizon_admin_loss_exactly_matches_legacy_nll_and_brier():
    contact = np.zeros((40, 2), dtype=np.bool_)
    contact[5:10, 0] = True
    contact[7:14, 1] = True
    trace = debounce_contact_trace(
        contact,
        sample_period_s=CTEQ_BIN_WIDTH_S,
        min_stable_steps=1,
    )
    legacy_labels = build_independent_event_labels(trace, anchor_indices=[0])
    admin_labels = build_cteq_administrative_censor_batch(
        legacy_labels.event_bin,
        **_boundary_kwargs([CTEQ_NUM_BINS], ["none"]),
    )
    rng = np.random.default_rng(912)
    distribution = dual_event_hazard_from_logits(
        rng.normal(size=(1, 2, 2, CTEQ_NUM_BINS))
    )
    legacy = dual_event_survival_loss(distribution, legacy_labels)
    admin = administrative_censor_survival_loss(distribution, admin_labels)
    np.testing.assert_allclose(admin.per_target_nll, legacy.per_target_nll)
    np.testing.assert_allclose(admin.per_target_brier, legacy.per_target_brier)
    assert admin.mean_nll == pytest.approx(legacy.mean_nll)
    assert admin.mean_brier == pytest.approx(legacy.mean_brier)


def test_numpy_and_torch_match_for_events_admin_censors_and_zero_exposure():
    labels = _mixed_labels()
    rng = np.random.default_rng(913)
    logits_np = rng.normal(size=(3, 2, 2, CTEQ_NUM_BINS))
    expected = administrative_censor_survival_loss(
        dual_event_hazard_from_logits(logits_np), labels
    )
    logits = torch.from_numpy(logits_np.copy())
    actual = CteqAdministrativeSurvivalLoss()(logits, *_torch_labels(labels))

    assert actual.mean_nll.item() == pytest.approx(expected.mean_nll, abs=1.0e-12)
    assert actual.mean_brier.item() == pytest.approx(
        expected.mean_brier, abs=1.0e-12
    )
    np.testing.assert_allclose(actual.per_target_nll.numpy(), expected.per_target_nll)
    np.testing.assert_allclose(
        actual.per_target_brier.numpy(), expected.per_target_brier
    )
    np.testing.assert_array_equal(
        actual.per_role_eligible_count.numpy(), expected.per_role_eligible_count
    )
    np.testing.assert_allclose(
        actual.per_role_nll_sum.numpy(), expected.per_role_nll_sum
    )
    np.testing.assert_allclose(
        actual.per_role_brier_sum.numpy(), expected.per_role_brier_sum
    )
    assert actual.eligible_target_count.item() == expected.eligible_target_count
    assert actual.excluded_target_count.item() == expected.excluded_target_count
    assert actual.td_event_count.item() == expected.td_event_count
    assert actual.lo_event_count.item() == expected.lo_event_count
    assert actual.td_censored_count.item() == expected.td_censored_count
    assert actual.lo_censored_count.item() == expected.lo_censored_count


def test_backward_excludes_zero_exposure_batch_rows():
    labels = _mixed_labels()
    torch.manual_seed(914)
    logits = torch.randn(
        3, 2, 2, CTEQ_NUM_BINS, dtype=torch.float64, requires_grad=True
    )
    loss = CteqAdministrativeSurvivalLoss()(logits, *_torch_labels(labels))
    (loss.mean_nll + loss.mean_brier).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:2]) > 0
    assert torch.count_nonzero(logits.grad[2]) == 0


def test_torchscript_matches_eager_for_dynamic_batch_sizes():
    module = CteqAdministrativeSurvivalLoss()
    scripted = torch.jit.script(module)
    labels = _mixed_labels()
    torch.manual_seed(915)
    logits = torch.randn(3, 2, 2, CTEQ_NUM_BINS)
    eager = module(logits, *_torch_labels(labels))
    actual = scripted(logits, *_torch_labels(labels))
    torch.testing.assert_close(actual.mean_nll, eager.mean_nll)
    torch.testing.assert_close(actual.mean_brier, eager.mean_brier)
    torch.testing.assert_close(actual.per_target_nll, eager.per_target_nll)
    assert actual.eligible_target_count == eager.eligible_target_count

    single_labels = build_cteq_administrative_censor_batch(
        np.full((1, 2, 2), -1, dtype=np.int64),
        **_boundary_kwargs([7], ["other"]),
    )
    single = scripted(
        torch.zeros(1, 2, 2, CTEQ_NUM_BINS),
        *_torch_labels(single_labels),
    )
    assert single.eligible_target_count.item() == 4


def test_zero_denominator_and_malformed_masks_fail_closed_numpy_and_torch():
    zero = build_cteq_administrative_censor_batch(
        np.full((1, 2, 2), -1, dtype=np.int64),
        **_boundary_kwargs([0], ["base_contact"]),
    )
    distribution = dual_event_hazard_from_logits(
        np.zeros((1, 2, 2, CTEQ_NUM_BINS))
    )
    with pytest.raises(CteqContractError, match="zero eligible"):
        administrative_censor_survival_loss(distribution, zero)
    with pytest.raises(ValueError, match="zero eligible"):
        CteqAdministrativeSurvivalLoss()(
            torch.zeros(1, 2, 2, CTEQ_NUM_BINS), *_torch_labels(zero)
        )

    mixed = _mixed_labels()
    bad_eligible = mixed.loss_eligible.copy()
    bad_eligible[2, 0, 0] = True
    malformed = replace(mixed, loss_eligible=bad_eligible)
    with pytest.raises(CteqContractError, match="loss_eligible"):
        administrative_censor_survival_loss(
            dual_event_hazard_from_logits(
                np.zeros((3, 2, 2, CTEQ_NUM_BINS))
            ),
            malformed,
        )
    torch_labels = list(_torch_labels(mixed))
    torch_labels[-1][2, 0, 0] = True
    with pytest.raises(ValueError, match="loss_eligible"):
        CteqAdministrativeSurvivalLoss()(
            torch.zeros(3, 2, 2, CTEQ_NUM_BINS), *torch_labels
        )


def test_loss_status_keeps_runner_and_training_gates_closed():
    status = cteq_administrative_loss_status()
    assert status["numpy_torch_interfaces_closed"] is True
    assert status["zero_exposure_excluded"] is True
    assert status["denominator"] == "count_nonzero(loss_eligible)"
    assert status["future_truth_consumers"] == ["loss", "evaluator"]
    assert status["runner_termination_provenance_authenticated"] is False
    assert status["actor_integrated"] is False
    assert status["gym_task_registered"] is False
    assert status["training_ready"] is False
