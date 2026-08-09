from __future__ import annotations

import copy

import numpy as np
import pytest

from rsl_rl.utils.cteq_administrative_censor import (
    CTEQ_FOOT_EVENT_SOURCE_CONTRACT,
    CteqCensorReason,
    build_cteq_administrative_censor_batch,
    validate_cteq_administrative_censor_receipt,
)
from rsl_rl.utils.cteq_contact_timing import (
    CTEQ_NUM_BINS,
    CteqContractError,
    PrivilegedLabelLeakageError,
    cteq_pr01_status,
    validate_causal_observation,
)


def _boundaries(
    observed_bins,
    *,
    time_limit=None,
    base_contact=None,
    other=None,
):
    exposure = np.asarray(observed_bins, dtype=np.int64)
    batch_size = exposure.size
    timeout = np.zeros(batch_size, dtype=np.bool_)
    base = np.zeros(batch_size, dtype=np.bool_)
    other_early = np.zeros(batch_size, dtype=np.bool_)
    if time_limit is not None:
        timeout[np.asarray(time_limit, dtype=np.int64)] = True
    if base_contact is not None:
        base[np.asarray(base_contact, dtype=np.int64)] = True
    if other is not None:
        other_early[np.asarray(other, dtype=np.int64)] = True
    done = timeout | base | other_early
    return {
        "fully_observed_bins": exposure,
        "episode_done": done,
        "time_limit": timeout,
        "base_contact_termination": base,
        "other_early_termination": other_early,
    }


def test_complete_window_separates_true_events_from_natural_censoring():
    event_bin = np.full((1, 2, 2), -1, dtype=np.int64)
    event_bin[0, 0, 0] = 4
    event_bin[0, 1, 1] = 8
    labels = build_cteq_administrative_censor_batch(
        event_bin,
        **_boundaries([CTEQ_NUM_BINS]),
    )

    assert labels.event_observed[0, 0, 0]
    assert labels.event_observed[0, 1, 1]
    assert labels.reason_code[0, 0, 0] == CteqCensorReason.OBSERVED_TOUCHDOWN
    assert labels.reason_code[0, 1, 1] == CteqCensorReason.OBSERVED_LIFTOFF
    natural = CteqCensorReason.NATURAL_HORIZON_RIGHT_CENSOR
    assert labels.reason_code[0, 0, 1] == natural
    assert labels.reason_code[0, 1, 0] == natural
    assert labels.censor_after_bin[0, 0, 1] == CTEQ_NUM_BINS
    assert labels.censor_after_bin[0, 1, 0] == CTEQ_NUM_BINS
    assert labels.censor_after_bin[0, 0, 0] == -1
    assert labels.loss_eligible.all()


def test_time_limit_censors_only_absent_targets_at_observed_boundary():
    event_bin = np.full((1, 2, 2), -1, dtype=np.int64)
    event_bin[0, 0, 0] = 5
    labels = build_cteq_administrative_censor_batch(
        event_bin,
        **_boundaries([6], time_limit=[0]),
    )

    assert labels.event_observed.sum() == 1
    assert labels.reason_code[0, 0, 0] == CteqCensorReason.OBSERVED_TOUCHDOWN
    absent = labels.right_censored
    assert np.all(
        labels.reason_code[absent]
        == CteqCensorReason.TIME_LIMIT_ADMINISTRATIVE_CENSOR
    )
    assert np.all(labels.censor_after_bin[absent] == 6)
    assert labels.reason_counts["episode_time_limit"] == 1
    assert labels.reason_counts["observed_touchdown"] == 1
    assert labels.reason_counts["time_limit_administrative_censor"] == 3
    # The episode boundary did not manufacture a touchdown/liftoff target.
    assert labels.event_observed.sum() == np.count_nonzero(event_bin >= 0)


def test_base_contact_and_other_early_termination_have_distinct_reasons():
    labels = build_cteq_administrative_censor_batch(
        np.full((2, 2, 2), -1, dtype=np.int64),
        **_boundaries([3, 9], base_contact=[0], other=[1]),
    )
    assert np.all(
        labels.reason_code[0]
        == CteqCensorReason.BASE_CONTACT_ADMINISTRATIVE_CENSOR
    )
    assert np.all(
        labels.reason_code[1]
        == CteqCensorReason.OTHER_EARLY_TERMINATION_ADMINISTRATIVE_CENSOR
    )
    assert np.all(labels.censor_after_bin[0] == 3)
    assert np.all(labels.censor_after_bin[1] == 9)
    assert labels.reason_counts["episode_base_contact"] == 1
    assert labels.reason_counts["episode_other_early_termination"] == 1


def test_zero_exposure_is_audited_but_not_loss_eligible():
    labels = build_cteq_administrative_censor_batch(
        np.full((1, 2, 2), -1, dtype=np.int64),
        **_boundaries([0], base_contact=[0]),
    )
    assert labels.right_censored.all()
    assert not labels.loss_eligible.any()
    assert labels.reason_counts["zero_exposure_ineligible_target"] == 4
    assert np.all(labels.censor_after_bin == 0)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("ambiguous_reason", "mutually exclusive"),
        ("missing_reason", "exhaustive union"),
        ("partial_nonterminal", "Partial nonterminal"),
        ("event_after_boundary", "strictly before"),
        ("termination_as_event_source", "termination flags are not"),
    ],
)
def test_ambiguous_or_future_boundary_inputs_fail_closed(mutation, match):
    event_bin = np.full((1, 2, 2), -1, dtype=np.int64)
    kwargs = _boundaries([5], base_contact=[0])
    source = CTEQ_FOOT_EVENT_SOURCE_CONTRACT
    if mutation == "ambiguous_reason":
        kwargs["time_limit"][0] = True
    elif mutation == "missing_reason":
        kwargs["base_contact_termination"][0] = False
    elif mutation == "partial_nonterminal":
        kwargs = _boundaries([5])
    elif mutation == "event_after_boundary":
        event_bin[0, 0, 0] = 5
    elif mutation == "termination_as_event_source":
        source = "episode_done_as_touchdown"
    with pytest.raises(CteqContractError, match=match):
        build_cteq_administrative_censor_batch(
            event_bin,
            foot_event_source_contract=source,
            **kwargs,
        )


def test_receipt_is_counted_hashed_unready_and_runner_unprovenance_is_visible():
    labels = build_cteq_administrative_censor_batch(
        np.full((2, 2, 2), -1, dtype=np.int64),
        **_boundaries([25, 4], time_limit=[1]),
    )
    receipt = labels.receipt()
    validate_cteq_administrative_censor_receipt(receipt)
    assert receipt["reason_counts"] == labels.reason_counts
    assert receipt["termination_is_event"] is False
    assert receipt["runner_termination_provenance_sha256"] is None
    assert receipt["runner_termination_provenance_receipted"] is False
    assert receipt["runner_termination_provenance_authenticated"] is False
    assert receipt["existing_full_horizon_loss_supports_early_censor"] is False
    assert receipt["administrative_censor_loss_interface_closed"] is True
    assert receipt["loss_interface_closed"] is True
    assert receipt["actor_integrated"] is False
    assert receipt["gym_task_registered"] is False
    assert receipt["training_ready"] is False
    assert len(receipt["label_tensor_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64

    tampered = copy.deepcopy(receipt)
    tampered["reason_counts"]["episode_time_limit"] = 9
    with pytest.raises(CteqContractError, match="digest mismatch"):
        validate_cteq_administrative_censor_receipt(tampered)

    with pytest.raises(CteqContractError, match="lowercase SHA-256"):
        build_cteq_administrative_censor_batch(
            np.full((1, 2, 2), -1, dtype=np.int64),
            runner_termination_provenance_sha256="not-a-runner-receipt",
            **_boundaries([25]),
        )

    synthetically_receipted = build_cteq_administrative_censor_batch(
        np.full((1, 2, 2), -1, dtype=np.int64),
        runner_termination_provenance_sha256="a" * 64,
        **_boundaries([25]),
    ).receipt()
    assert synthetically_receipted["runner_termination_provenance_receipted"] is True
    assert synthetically_receipted[
        "runner_termination_provenance_authenticated"
    ] is False
    assert synthetically_receipted["training_ready"] is False
    validate_cteq_administrative_censor_receipt(synthetically_receipted)


def test_administrative_truth_cannot_enter_actor_observation():
    labels = build_cteq_administrative_censor_batch(
        np.full((1, 2, 2), -1, dtype=np.int64),
        **_boundaries([25]),
    )
    assert labels.for_consumer("loss")["censor_after_bin"].shape == (1, 2, 2)
    assert labels.for_consumer("evaluator")["reason_code"].shape == (1, 2, 2)
    with pytest.raises(PrivilegedLabelLeakageError):
        labels.for_consumer("actor")
    with pytest.raises(PrivilegedLabelLeakageError):
        labels.observation_payload()
    with pytest.raises(PrivilegedLabelLeakageError):
        validate_causal_observation({"nested": labels})
    with pytest.raises(PrivilegedLabelLeakageError):
        validate_causal_observation({"censor_after_bin": np.zeros(1)})

    status = cteq_pr01_status()
    assert "administrative_censor_label_adapter" in status["implemented"]
    assert "administrative_censor_numpy_torch_survival_loss" in status["implemented"]
    assert status["training_ready"] is False
