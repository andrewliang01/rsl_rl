from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from rsl_rl.utils.cteq_contact_timing import (
    CteqContractError,
    PrivilegedLabelLeakageError,
    validate_causal_observation,
)
from rsl_rl.utils.cteq_isaaclab_extras_bridge import (
    CTEQ_BRIDGE_METADATA_KEY,
    CTEQ_FULLY_OBSERVED_BINS_KEY,
    CTEQ_FULLY_OBSERVED_BINS_SOURCE_RECEIPT_KEY,
    CTEQ_FULLY_OBSERVED_BINS_VALID_KEY,
    CTEQ_ISAACLAB_EXTRAS_BRIDGE_SCHEMA,
    CTEQ_ISAACLAB_TIME_LIMIT_KEY,
    CTEQ_RAW_TIME_LIMIT_KEY,
    CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY,
    build_cteq_isaaclab_termination_batch,
    cteq_isaaclab_extras_bridge_status,
    validate_cteq_isaaclab_extras_receipt,
)
from rsl_rl.utils.cteq_runner_termination_provenance import (
    CTEQ_BASE_CONTACT_KEY,
    CTEQ_EPISODE_ID_BEFORE_KEY,
    CTEQ_OTHER_TERMINATION_KEY,
    CTEQ_POST_STEP_EPISODE_ID_KEY,
    CTEQ_TERMINAL_CONTACT_KEY,
    CTEQ_TERMINAL_CONTACT_VALID_KEY,
    CTEQ_TERMINAL_EPISODE_ID_KEY,
)


def _hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _terminal_receipt() -> dict:
    receipt = {
        "schema": "cteq-terminal-contact-source-receipt-v1",
        "contract": "environment_pre_reset_terminal_foot_contact_v1",
        "capture_hook": "isaaclab_record_post_step_before_record_pre_reset_v1",
        "sensor_name": "contact_forces",
        "foot_order": ["left", "right"],
        "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "source_field": "net_forces_w",
        "force_reduction": "vector_l2_norm",
        "contact_rule": "force_norm_strictly_greater_than_threshold",
        "contact_force_threshold_n": 1.0,
        "terminal_rows_only": True,
        "truth_only": True,
        "allowed_consumers": ["label_builder", "loss", "evaluator"],
        "actor_observation_access": False,
        "critic_hidden_access": False,
        "reward_access": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    return receipt


def _bins_receipt() -> dict:
    receipt = {
        "schema": "cteq-fully-observed-bins-source-receipt-v1",
        "contract": "cteq_anchor_to_boundary_fully_observed_bins_v1",
        "capture_hook": "isaaclab_record_post_step_before_record_pre_reset_v1",
        "provider_id": "test_anchor_collector",
        "provider_source_sha256": "2" * 64,
        "sample_period_s": 0.02,
        "max_bins": 25,
        "count_semantics": "complete_post_anchor_contact_samples_before_boundary",
        "truth_only": True,
        "allowed_consumers": ["label_builder", "loss", "evaluator"],
        "training_ready": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    return receipt


def _extras() -> dict:
    return {
        # Raw timeout remains unchanged for PPO; row 2 overlaps base contact.
        "time_outs": torch.tensor([False, False, True, False]),
        CTEQ_RAW_TIME_LIMIT_KEY: torch.tensor([False, False, True, False]),
        CTEQ_ISAACLAB_TIME_LIMIT_KEY: torch.tensor([False, False, False, False]),
        CTEQ_BASE_CONTACT_KEY: torch.tensor([False, True, True, False]),
        CTEQ_OTHER_TERMINATION_KEY: torch.tensor([False, False, False, True]),
        CTEQ_TERMINAL_CONTACT_KEY: torch.tensor(
            [[False, False], [True, False], [False, True], [True, True]]
        ),
        CTEQ_TERMINAL_CONTACT_VALID_KEY: torch.tensor([False, True, True, True]),
        CTEQ_EPISODE_ID_BEFORE_KEY: torch.tensor([4, 8, 12, 16]),
        CTEQ_TERMINAL_EPISODE_ID_KEY: torch.tensor([-1, 8, 12, 16]),
        CTEQ_POST_STEP_EPISODE_ID_KEY: torch.tensor([4, 9, 13, 17]),
        CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY: _terminal_receipt(),
        CTEQ_FULLY_OBSERVED_BINS_KEY: torch.tensor([0, 5, 7, 2]),
        CTEQ_FULLY_OBSERVED_BINS_VALID_KEY: torch.tensor([False, True, True, True]),
        CTEQ_FULLY_OBSERVED_BINS_SOURCE_RECEIPT_KEY: _bins_receipt(),
        CTEQ_BRIDGE_METADATA_KEY: {
            "schema": CTEQ_ISAACLAB_EXTRAS_BRIDGE_SCHEMA,
            "capture_hook": "isaaclab_record_post_step_before_record_pre_reset_v1",
            "truth_only": True,
            "allowed_consumers": ["label_builder", "loss", "evaluator"],
            "actor_observation_access": False,
            "critic_hidden_access": False,
            "reward_access": False,
            "training_ready": False,
        },
    }


def test_bridge_authenticates_autoreset_boundary_and_exposure() -> None:
    batch = build_cteq_isaaclab_termination_batch(
        torch.tensor([0, 1, 1, 1], dtype=torch.long), _extras()
    )
    assert np.array_equal(batch.fully_observed_bins, [0, 5, 7, 2])
    truth = batch.for_consumer("label_builder")
    assert np.array_equal(truth["termination"]["time_limit"], [False] * 4)
    assert np.array_equal(
        truth["termination"]["base_contact_termination"],
        [False, True, True, False],
    )
    receipt = batch.receipt()
    validate_cteq_isaaclab_extras_receipt(receipt)
    assert receipt["autoreset_boundary_authenticated"] is True
    assert receipt["raw_timeout_cross_checked"] is True
    assert receipt["simultaneous_timeout_mdp_priority"] == "mdp_termination"
    assert receipt["device_transfer_authorized"] is False
    assert receipt["device_transfer_fields"] == []
    assert receipt["training_ready"] is False


def test_bridge_truth_cannot_enter_actor_critic_or_reward() -> None:
    batch = build_cteq_isaaclab_termination_batch(
        np.asarray([False, True, True, True]), _extras()
    )
    for method in (
        batch.observation_payload,
        batch.critic_hidden_payload,
        batch.reward_payload,
    ):
        with pytest.raises(PrivilegedLabelLeakageError):
            method()
    with pytest.raises(PrivilegedLabelLeakageError):
        batch.for_consumer("actor")
    with pytest.raises(PrivilegedLabelLeakageError):
        validate_causal_observation({"policy": batch})


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_bins",
        "raw_timeout_mismatch",
        "invalid_bins_row",
        "tampered_terminal_receipt",
        "post_reset_id_not_newer",
    ),
)
def test_bridge_fails_closed_on_incomplete_or_inconsistent_truth(mutation: str) -> None:
    extras = _extras()
    if mutation == "missing_bins":
        del extras[CTEQ_FULLY_OBSERVED_BINS_KEY]
    elif mutation == "raw_timeout_mismatch":
        extras["time_outs"] = torch.zeros(4, dtype=torch.bool)
    elif mutation == "invalid_bins_row":
        extras[CTEQ_FULLY_OBSERVED_BINS_KEY][0] = 3
    elif mutation == "tampered_terminal_receipt":
        extras[CTEQ_TERMINAL_CONTACT_SOURCE_RECEIPT_KEY]["sensor_name"] = "other"
    elif mutation == "post_reset_id_not_newer":
        extras[CTEQ_POST_STEP_EPISODE_ID_KEY][1] = 8
    with pytest.raises(CteqContractError):
        build_cteq_isaaclab_termination_batch(
            torch.tensor([0, 1, 1, 1]), extras
        )


def test_status_keeps_training_and_registration_disabled() -> None:
    status = cteq_isaaclab_extras_bridge_status()
    assert status["opt_in_only"] is True
    assert status["stock_runner_modified"] is False
    assert status["registered_task_modified"] is False
    assert status["fully_observed_bins_provider_implemented"] is False
    assert status["actor_integrated"] is False
    assert status["critic_hidden_integrated"] is False
    assert status["reward_integrated"] is False
    assert status["gym_task_registered"] is False
    assert status["training_ready"] is False
