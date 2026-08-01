from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rsl_rl.utils.cteq_administrative_censor import (
    build_cteq_administrative_censor_batch,
    validate_cteq_administrative_censor_receipt,
)
from rsl_rl.utils.cteq_contact_timing import (
    CteqContractError,
    PrivilegedLabelLeakageError,
    cteq_pr01_status,
    validate_causal_observation,
)
from rsl_rl.utils.cteq_runner_termination_provenance import (
    CTEQ_BASE_CONTACT_KEY,
    CTEQ_EPISODE_ID_BEFORE_KEY,
    CTEQ_OTHER_TERMINATION_KEY,
    CTEQ_POST_STEP_EPISODE_ID_KEY,
    CTEQ_REQUIRED_EXTRAS_KEYS,
    CTEQ_TERMINAL_CONTACT_KEY,
    CTEQ_TERMINAL_CONTACT_VALID_KEY,
    CTEQ_TERMINAL_EPISODE_ID_KEY,
    CTEQ_TIME_LIMIT_KEY,
    build_cteq_on_policy_termination_provenance,
    cteq_current_on_policy_runner_provenance_status,
    validate_cteq_runner_termination_provenance_receipt,
)


SOURCE_SHA = "7" * 64


def _canonical_step():
    done = np.asarray([False, True, True, True], dtype=np.bool_)
    extras = {
        CTEQ_TIME_LIMIT_KEY: np.asarray(
            [False, True, False, False], dtype=np.bool_
        ),
        CTEQ_BASE_CONTACT_KEY: np.asarray(
            [False, False, True, False], dtype=np.bool_
        ),
        CTEQ_OTHER_TERMINATION_KEY: np.asarray(
            [False, False, False, True], dtype=np.bool_
        ),
        CTEQ_TERMINAL_CONTACT_KEY: np.asarray(
            [[False, False], [True, True], [True, False], [False, True]],
            dtype=np.bool_,
        ),
        CTEQ_TERMINAL_CONTACT_VALID_KEY: done.copy(),
        CTEQ_EPISODE_ID_BEFORE_KEY: np.asarray(
            [10, 20, 30, 40], dtype=np.int64
        ),
        CTEQ_TERMINAL_EPISODE_ID_KEY: np.asarray(
            [-1, 20, 30, 40], dtype=np.int64
        ),
        CTEQ_POST_STEP_EPISODE_ID_KEY: np.asarray(
            [10, 21, 35, 41], dtype=np.int64
        ),
    }
    return done, extras


def _build():
    done, extras = _canonical_step()
    return build_cteq_on_policy_termination_provenance(
        done,
        extras,
        terminal_contact_source_sha256=SOURCE_SHA,
    )


def test_canonical_step_receipts_all_three_termination_sources_and_reset_ids():
    provenance = _build()
    receipt = provenance.receipt()
    validate_cteq_runner_termination_provenance_receipt(receipt)

    assert receipt["required_extras_keys"] == list(CTEQ_REQUIRED_EXTRAS_KEYS)
    assert receipt["counts"] == {
        "done": 3,
        "time_limit": 1,
        "base_contact": 1,
        "other_termination": 1,
        "terminal_contact_valid": 3,
        "reset_boundary": 3,
    }
    assert receipt["terminal_contact_stage"] == (
        "pre_reset_terminal_transition"
    )
    assert receipt["terminal_contact_authenticated"] is True
    assert receipt["episode_reset_boundary_authenticated"] is True
    assert receipt["provenance_authenticated"] is True
    assert receipt["termination_is_foot_event"] is False
    assert receipt["actor_observation_access"] is False
    assert receipt["critic_hidden_access"] is False
    assert receipt["gpu_required"] is False
    assert receipt["training_ready"] is False


def test_on_policy_cpu_tensors_are_accepted_without_actor_or_gpu_path():
    done, extras = _canonical_step()
    tensor_extras = {
        key: torch.from_numpy(value.copy()) for key, value in extras.items()
    }
    provenance = build_cteq_on_policy_termination_provenance(
        torch.from_numpy(done.copy()),
        tensor_extras,
        terminal_contact_source_sha256=SOURCE_SHA,
    )
    assert provenance.receipt()["gpu_required"] is False
    np.testing.assert_array_equal(provenance.episode_done, done)


def test_stock_runner_fields_fail_closed_without_terminal_contact_and_ids():
    done = np.asarray([True, False], dtype=np.bool_)
    stock_extras = {
        "time_outs": np.asarray([True, False], dtype=np.bool_),
    }
    with pytest.raises(CteqContractError, match="incomplete; missing") as error:
        build_cteq_on_policy_termination_provenance(
            done,
            stock_extras,
            terminal_contact_source_sha256=SOURCE_SHA,
        )
    for key in CTEQ_REQUIRED_EXTRAS_KEYS:
        if key != CTEQ_TIME_LIMIT_KEY:
            assert key in str(error.value)

    status = cteq_current_on_policy_runner_provenance_status()
    assert status["stock_runner_visible_fields"] == [
        "dones",
        "extras.time_outs_optional",
    ]
    assert status["terminal_contact_authoritative"] is False
    assert status["episode_reset_boundary_authenticated"] is False
    assert status["provenance_authenticated"] is False
    assert "environment_pre_reset_terminal_contact_export" in status[
        "remaining_interfaces"
    ]
    assert status["training_ready"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("overlap_reason", "mutually exclusive"),
        ("missing_reason", "exhaustive union"),
        ("missing_terminal_sample", "Every done row"),
        ("latent_nondone_contact", "exactly false"),
        ("reset_contact_as_terminal", "pre-reset episode"),
        ("done_without_new_episode", "strictly newer"),
        ("nondone_crosses_reset", "cannot cross"),
    ],
)
def test_ambiguous_termination_or_episode_reset_boundary_fails_closed(
    mutation, match
):
    done, extras = _canonical_step()
    if mutation == "overlap_reason":
        extras[CTEQ_BASE_CONTACT_KEY][1] = True
    elif mutation == "missing_reason":
        extras[CTEQ_TIME_LIMIT_KEY][1] = False
    elif mutation == "missing_terminal_sample":
        extras[CTEQ_TERMINAL_CONTACT_VALID_KEY][1] = False
    elif mutation == "latent_nondone_contact":
        extras[CTEQ_TERMINAL_CONTACT_KEY][0, 0] = True
    elif mutation == "reset_contact_as_terminal":
        extras[CTEQ_TERMINAL_EPISODE_ID_KEY][1] = extras[
            CTEQ_POST_STEP_EPISODE_ID_KEY
        ][1]
    elif mutation == "done_without_new_episode":
        extras[CTEQ_POST_STEP_EPISODE_ID_KEY][1] = extras[
            CTEQ_EPISODE_ID_BEFORE_KEY
        ][1]
    elif mutation == "nondone_crosses_reset":
        extras[CTEQ_POST_STEP_EPISODE_ID_KEY][0] += 1
    with pytest.raises(CteqContractError, match=match):
        build_cteq_on_policy_termination_provenance(
            done,
            extras,
            terminal_contact_source_sha256=SOURCE_SHA,
        )


def test_validated_provenance_authenticates_matching_admin_labels_only():
    provenance = _build()
    event_bin = np.full((4, 2, 2), -1, dtype=np.int64)
    labels = build_cteq_administrative_censor_batch(
        event_bin,
        fully_observed_bins=np.asarray([25, 5, 3, 7], dtype=np.int64),
        episode_done=provenance.episode_done,
        time_limit=provenance.time_limit,
        base_contact_termination=provenance.base_contact_termination,
        other_early_termination=provenance.other_early_termination,
        runner_termination_provenance=provenance,
    )
    receipt = labels.receipt()
    validate_cteq_administrative_censor_receipt(receipt)
    assert not labels.event_observed.any()
    assert receipt["runner_termination_provenance_receipted"] is True
    assert receipt["runner_termination_provenance_authenticated"] is True
    assert receipt["runner_termination_provenance_receipt"] == (
        provenance.receipt()
    )
    assert receipt["training_ready"] is False

    mismatched_base = provenance.base_contact_termination.copy()
    mismatched_base[2] = False
    with pytest.raises(CteqContractError, match="provenance differ"):
        build_cteq_administrative_censor_batch(
            event_bin,
            fully_observed_bins=np.asarray([25, 5, 3, 7], dtype=np.int64),
            episode_done=provenance.episode_done,
            time_limit=provenance.time_limit,
            base_contact_termination=mismatched_base,
            other_early_termination=provenance.other_early_termination,
            runner_termination_provenance=provenance,
        )


def test_provenance_truth_is_never_actor_observation_or_critic_hidden_state():
    provenance = _build()
    assert provenance.for_consumer("label_builder")[
        "terminal_foot_contact"
    ].shape == (4, 2)
    assert provenance.for_consumer("loss")["time_limit"].shape == (4,)
    with pytest.raises(PrivilegedLabelLeakageError):
        provenance.for_consumer("actor")
    with pytest.raises(PrivilegedLabelLeakageError):
        provenance.for_consumer("critic")
    with pytest.raises(PrivilegedLabelLeakageError):
        provenance.observation_payload()
    with pytest.raises(PrivilegedLabelLeakageError):
        provenance.critic_hidden_payload()
    with pytest.raises(PrivilegedLabelLeakageError):
        validate_causal_observation({"runner_provenance": provenance})

    status = cteq_pr01_status()
    assert "on_policy_runner_termination_provenance_contract" in status[
        "implemented"
    ]
    assert "environment_extras_binding_for_runner_termination_provenance" in (
        status["blocked_next_steps"]
    )
    assert status["training_ready"] is False


def test_runner_receipt_tampering_and_unreceipted_contact_source_are_rejected():
    receipt = _build().receipt()
    tampered = copy.deepcopy(receipt)
    tampered["tensor_sha256"] = "8" * 64
    with pytest.raises(CteqContractError, match="digest mismatch"):
        validate_cteq_runner_termination_provenance_receipt(tampered)

    done, extras = _canonical_step()
    with pytest.raises(CteqContractError, match="lowercase SHA-256"):
        build_cteq_on_policy_termination_provenance(
            done,
            extras,
            terminal_contact_source_sha256=None,
        )
