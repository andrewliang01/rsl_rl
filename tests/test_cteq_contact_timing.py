import numpy as np
import pytest

from rsl_rl.utils.cteq_contact_timing import (
    CTEQ_BIN_WIDTH_S,
    CTEQ_NUM_BINS,
    CteqContractError,
    EventIndex,
    PrivilegedLabelLeakageError,
    build_independent_event_labels,
    cteq_pr01_status,
    cteq_role_time_weights,
    debounce_contact_trace,
    dual_event_hazard_from_logits,
    dual_event_hazard_from_probabilities,
    dual_event_survival_loss,
    hazard_calibration_diagnostics,
    validate_causal_observation,
)


def _trace(contact, *, stable_steps=1):
    return debounce_contact_trace(
        np.asarray(contact, dtype=np.bool_),
        sample_period_s=CTEQ_BIN_WIDTH_S,
        min_stable_steps=stable_steps,
    )


def _constant_contact(length, left=False, right=False):
    return np.tile(np.asarray([left, right], dtype=np.bool_), (length, 1))


def _peaked_logits(event_bins):
    """Build low-background hazards peaked at optional event-bin indices."""

    logits = np.full((1, 2, 2, CTEQ_NUM_BINS), -30.0)
    for foot in range(2):
        for event in range(2):
            event_bin = event_bins[foot][event]
            if event_bin is not None:
                logits[0, foot, event, event_bin] = 30.0
    return logits


def test_debounce_removes_glitches_and_backdates_confirmed_transitions():
    raw = _constant_contact(14)
    raw[2, 0] = True  # one-sample touchdown glitch
    raw[5:8, 0] = True  # confirmed touchdown starts at sample 5
    raw[8, 0] = False  # one-sample liftoff glitch
    raw[9, 0] = True
    raw[10:, 0] = False  # confirmed liftoff starts at sample 10

    trace = _trace(raw, stable_steps=2)

    assert not trace.stable_contact[:5, 0].any()
    assert trace.stable_contact[5:10, 0].all()
    assert not trace.stable_contact[10:, 0].any()
    assert [
        (event.sample_index, event.event_index) for event in trace.events
    ] == [
        (5, EventIndex.TOUCHDOWN),
        (10, EventIndex.LIFTOFF),
    ]
    assert trace.truth_only
    assert not trace.training_ready


def test_horizon_is_right_closed_and_anchor_event_is_not_future():
    contact = _constant_contact(55)
    contact[25:, 0] = True
    contact[26:, 1] = True
    trace = _trace(contact)

    anchor_zero = build_independent_event_labels(trace, anchor_indices=[0])
    assert anchor_zero.event_observed[0, 0, EventIndex.TOUCHDOWN]
    assert anchor_zero.event_bin[0, 0, EventIndex.TOUCHDOWN] == 24
    assert anchor_zero.time_to_event_s[0, 0, EventIndex.TOUCHDOWN] == pytest.approx(
        0.5
    )
    assert anchor_zero.right_censored[0, 1, EventIndex.TOUCHDOWN]
    assert anchor_zero.event_bin[0, 1, EventIndex.TOUCHDOWN] == -1

    anchor_at_event = build_independent_event_labels(trace, anchor_indices=[25])
    assert anchor_at_event.right_censored[0, 0, EventIndex.TOUCHDOWN]


def test_td_and_lo_are_independent_labels_after_lo_then_td():
    contact = _constant_contact(40, left=True)
    contact[5:10, 0] = False
    trace = _trace(contact)
    labels = build_independent_event_labels(trace, anchor_indices=[0])

    assert labels.event_observed[0, 0, EventIndex.LIFTOFF]
    assert labels.event_observed[0, 0, EventIndex.TOUCHDOWN]
    assert labels.event_bin[0, 0, EventIndex.LIFTOFF] == 4
    assert labels.event_bin[0, 0, EventIndex.TOUCHDOWN] == 9
    assert not labels.right_censored[0, 0].any()


def test_double_support_and_flight_preserve_per_foot_independent_censoring():
    double_support = _constant_contact(40, left=True, right=True)
    double_support[5:, 0] = False
    double_support[7:, 1] = False
    support_labels = build_independent_event_labels(
        _trace(double_support), anchor_indices=[0]
    )
    assert support_labels.event_observed[0, :, EventIndex.LIFTOFF].all()
    assert support_labels.right_censored[0, :, EventIndex.TOUCHDOWN].all()

    flight = _constant_contact(40)
    flight[5:, 0] = True
    flight[7:, 1] = True
    flight_labels = build_independent_event_labels(_trace(flight), anchor_indices=[0])
    assert flight_labels.event_observed[0, :, EventIndex.TOUCHDOWN].all()
    assert flight_labels.right_censored[0, :, EventIndex.LIFTOFF].all()


def test_all_four_targets_remain_in_loss_when_right_censored():
    labels = build_independent_event_labels(
        _trace(_constant_contact(30)), anchor_indices=[0]
    )
    distribution = dual_event_hazard_from_logits(
        np.full((1, 2, 2, CTEQ_NUM_BINS), -8.0)
    )

    loss = dual_event_survival_loss(distribution, labels)

    assert labels.right_censored.all()
    assert not labels.event_observed.any()
    assert np.isfinite(loss.mean_nll)
    assert np.isfinite(loss.mean_brier)
    assert loss.td_event_count == 0
    assert loss.lo_event_count == 0
    assert loss.td_censored_count == 2
    assert loss.lo_censored_count == 2
    assert loss.per_target_nll.shape == (1, 2, 2)


def test_dual_hazards_are_two_independent_survival_distributions():
    hazards = np.zeros((1, 2, 2, CTEQ_NUM_BINS), dtype=np.float64)
    hazards[..., 0] = 0.9
    distribution = dual_event_hazard_from_probabilities(hazards)

    total = distribution.event_mass.sum(axis=-1) + distribution.censor_mass
    assert np.allclose(total, 1.0)
    # TD and LO can each occur with high probability; they do not share a
    # single competing-risk probability simplex.
    assert distribution.event_mass[0, 0].sum() > 1.7

    extreme = dual_event_hazard_from_logits(
        np.linspace(-1000.0, 1000.0, 100).reshape(1, 2, 2, 25)
    )
    assert np.isfinite(extreme.hazard).all()
    assert np.isfinite(extreme.event_mass).all()
    assert np.isfinite(extreme.censor_mass).all()


def test_survival_nll_and_brier_reward_correct_event_or_censor_mass():
    contact = _constant_contact(40, left=True)
    contact[5:10, 0] = False
    labels = build_independent_event_labels(_trace(contact), anchor_indices=[0])

    correct_bins = (
        (9, 4),
        (None, None),
    )
    correct = dual_event_hazard_from_logits(_peaked_logits(correct_bins))
    wrong = dual_event_hazard_from_logits(
        _peaked_logits(((0, 0), (0, 0)))
    )
    correct_loss = dual_event_survival_loss(correct, labels)
    wrong_loss = dual_event_survival_loss(wrong, labels)

    assert correct_loss.mean_nll < wrong_loss.mean_nll
    assert correct_loss.mean_brier < wrong_loss.mean_brier


def test_role_weights_use_causal_contact_lo_survival_and_td_event_mass():
    hazards = np.full((2, 2, 2, CTEQ_NUM_BINS), 0.1)
    distribution = dual_event_hazard_from_probabilities(hazards)
    contact_now = np.asarray([[True, False], [False, True]], dtype=np.bool_)

    weights = cteq_role_time_weights(distribution, contact_now)

    assert weights.current.shape == (2, 2, CTEQ_NUM_BINS)
    assert weights.landing.shape == (2, 2, CTEQ_NUM_BINS)
    assert np.allclose(
        weights.landing,
        distribution.event_mass[..., EventIndex.TOUCHDOWN, :],
    )
    assert np.allclose(
        weights.current[0, 0],
        distribution.survival_before[0, 0, EventIndex.LIFTOFF],
    )
    assert np.count_nonzero(weights.current[0, 1]) == 0
    assert np.count_nonzero(weights.current[1, 0]) == 0

    with pytest.raises(CteqContractError, match="causal bool"):
        cteq_role_time_weights(distribution, contact_now.astype(np.float32))


def test_order_and_calibration_diagnostics_distinguish_valid_order():
    contact = _constant_contact(40, left=True, right=False)
    contact[3:9, 0] = False  # contact: LO before TD
    contact[3:9, 1] = True  # flight: TD before LO
    labels = build_independent_event_labels(_trace(contact), anchor_indices=[0])

    valid = dual_event_hazard_from_logits(
        _peaked_logits(((8, 2), (2, 8)))
    )
    invalid = dual_event_hazard_from_logits(
        _peaked_logits(((2, 8), (8, 2)))
    )
    valid_diag = hazard_calibration_diagnostics(valid, labels)
    invalid_diag = hazard_calibration_diagnostics(invalid, labels)

    assert valid_diag["actual_order_comparable_count"] == 2
    assert valid_diag["actual_order_violation_count"] == 0
    assert valid_diag["mean_order_violation_probability"] < 1.0e-8
    assert invalid_diag["mean_order_violation_probability"] > 1.0 - 1.0e-8
    assert valid_diag["observed_ttc_mae_s"] is not None
    assert valid_diag["reliability_input"][
        "td_predicted_within_horizon"
    ].shape == (1, 2)


def test_future_truth_is_loss_evaluator_only_and_status_is_not_training_ready():
    labels = build_independent_event_labels(
        _trace(_constant_contact(30)), anchor_indices=[0]
    )
    assert labels.for_consumer("loss")["event_bin"].shape == (1, 2, 2)
    assert labels.for_consumer("evaluator")["event_bin"].shape == (1, 2, 2)
    with pytest.raises(PrivilegedLabelLeakageError):
        labels.for_consumer("actor")
    with pytest.raises(PrivilegedLabelLeakageError):
        labels.observation_payload()
    with pytest.raises(PrivilegedLabelLeakageError):
        validate_causal_observation({"nested": [labels]})
    with pytest.raises(PrivilegedLabelLeakageError):
        validate_causal_observation({"future_event_labels": np.zeros(1)})

    validate_causal_observation(
        {"proprioception": np.zeros(8), "current_contact": [True, False]}
    )
    status = cteq_pr01_status()
    assert not status["training_ready"]
    assert not status["actor_integrated"]
    assert not status["gym_task_registered"]
    assert not status["gpu_required"]
    assert status["future_truth_allowed_consumers"] == ("loss", "evaluator")


def test_incomplete_horizon_and_non_boolean_trace_are_rejected():
    with pytest.raises(CteqContractError, match="boolean dtype"):
        debounce_contact_trace(
            np.zeros((30, 2), dtype=np.float32),
            sample_period_s=CTEQ_BIN_WIDTH_S,
            min_stable_steps=2,
        )

    trace = _trace(_constant_contact(30))
    with pytest.raises(CteqContractError, match="complete horizon"):
        build_independent_event_labels(trace, anchor_indices=[5])


def test_initial_contact_glitch_requires_stable_preroll():
    raw = _constant_contact(12)
    raw[0, 0] = True
    with pytest.raises(CteqContractError, match="stable pre-roll"):
        _trace(raw, stable_steps=3)

    too_short = _constant_contact(2)
    with pytest.raises(CteqContractError, match="initial pre-roll"):
        _trace(too_short, stable_steps=3)
