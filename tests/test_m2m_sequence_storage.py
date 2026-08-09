# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D103

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import (
    M2MSequenceRolloutStorage,
    M2MSequenceTransition,
    RolloutStorage,
)

STUDENT_KEYS = ("proprioception", "observed_history", "history_valid")


def make_obs(num_envs: int, step: int = 0) -> TensorDict:
    env_ids = torch.arange(num_envs, dtype=torch.float32)
    return TensorDict(
        {
            "proprioception": torch.stack((env_ids, torch.full_like(env_ids, float(step))), dim=-1),
            "observed_history": torch.full((num_envs, 1, 2, 3), float(step), dtype=torch.float16),
            "history_valid": torch.full((num_envs, 1, 2, 3), step % 2, dtype=torch.uint8),
        },
        batch_size=[num_envs],
    )


def make_storage(*, num_envs: int = 3, rollout_length: int = 5) -> M2MSequenceRolloutStorage:
    return M2MSequenceRolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=rollout_length,
        student_obs=make_obs(num_envs),
        allowed_student_keys=STUDENT_KEYS,
        hidden_state_shape=(2, 4),
        device="cpu",
    )


def add_step(
    storage: M2MSequenceRolloutStorage,
    step: int,
    *,
    dones: torch.Tensor | None = None,
    episode_starts: torch.Tensor | None = None,
) -> None:
    num_envs = storage.num_envs
    if dones is None:
        dones = torch.zeros(num_envs, dtype=torch.bool)
    hidden = torch.full((num_envs, *storage.hidden_state_shape), float(step + 1), dtype=torch.float32)
    env_ids = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)
    latent = torch.full((num_envs, 64), float(step), dtype=torch.float32)
    latent[:, :1] = env_ids * 100.0 + step
    actions = torch.full((num_envs, 29), float(step), dtype=torch.float32)
    actions[:, :1] = env_ids * 100.0 + step
    storage.add_transition(
        M2MSequenceTransition(
            student_observations=make_obs(num_envs, step),
            teacher_latent_A=latent,
            teacher_action_mean=actions,
            dones=dones,
            student_hidden_state=hidden,
            episode_starts=episode_starts,
        )
    )


def test_allocates_only_deployable_sequence_fields_and_preserves_dtypes() -> None:
    storage = make_storage()

    assert storage.student_observations.batch_size == torch.Size([5, 3])
    assert storage.student_observations["proprioception"].dtype == torch.float32
    assert storage.student_observations["observed_history"].dtype == torch.float16
    assert storage.student_observations["history_valid"].dtype == torch.uint8
    assert storage.teacher_latent_A.shape == (5, 3, 64)
    assert storage.teacher_action_mean.shape == (5, 3, 29)
    assert storage.dones.dtype == torch.bool
    assert storage.student_hidden_states.shape == (5, 3, 2, 4)

    assert not hasattr(storage, "next_observations")
    assert not hasattr(storage, "teacher_observations")
    assert not hasattr(storage, "teacher_map")
    assert not hasattr(storage, "rewards")


@pytest.mark.parametrize(
    "forbidden_key",
    ["teacher_map", "privileged_height", "ground_truth_pose", "oracle_scan"],
)
def test_rejects_non_deployable_observation_keys(forbidden_key: str) -> None:
    observations = TensorDict({forbidden_key: torch.zeros(2, 4)}, batch_size=[2])
    with pytest.raises(ValueError, match="privileged/non-deployable"):
        M2MSequenceRolloutStorage(
            num_envs=2,
            num_transitions_per_env=3,
            student_obs=observations,
            allowed_student_keys=(forbidden_key,),
            hidden_state_shape=(1, 8),
        )


def test_exact_allowlist_rejects_disguised_extra_key() -> None:
    observations = TensorDict(
        {
            "policy": torch.zeros(2, 4),
            # This name avoids every defensive forbidden fragment.  It must
            # still fail because deployability is an explicit key contract.
            "context_cache": torch.zeros(2, 8),
        },
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="exactly match allowed_student_keys"):
        M2MSequenceRolloutStorage(
            num_envs=2,
            num_transitions_per_env=3,
            student_obs=observations,
            allowed_student_keys=("policy",),
            hidden_state_shape=(1, 8),
        )


def test_done_and_explicit_partial_reset_zero_only_affected_hidden_states() -> None:
    storage = make_storage()
    add_step(storage, 0, dones=torch.tensor([False, True, False]))
    add_step(storage, 1)
    storage.reset_envs(torch.tensor([2]))
    add_step(storage, 2)

    assert storage.episode_starts[0, :, 0].tolist() == [False, False, False]
    assert storage.episode_starts[1, :, 0].tolist() == [False, True, False]
    assert storage.episode_starts[2, :, 0].tolist() == [False, False, True]
    assert torch.count_nonzero(storage.student_hidden_states[1, 1]) == 0
    assert torch.count_nonzero(storage.student_hidden_states[2, 2]) == 0
    torch.testing.assert_close(storage.student_hidden_states[1, 0], torch.full((2, 4), 2.0))
    torch.testing.assert_close(storage.student_hidden_states[2, 1], torch.full((2, 4), 3.0))


def test_clear_preserves_cross_rollout_done_boundary_unless_explicitly_reset() -> None:
    storage = make_storage(rollout_length=2)
    add_step(storage, 0, dones=torch.tensor([False, True, False]))
    storage.clear()
    add_step(storage, 10)
    assert storage.episode_starts[0, :, 0].tolist() == [False, True, False]
    assert torch.count_nonzero(storage.student_hidden_states[0, 1]) == 0

    storage.clear(reset_episode_state=True)
    add_step(storage, 20)
    assert not torch.any(storage.episode_starts[0])


def test_sequence_batches_never_cross_done_or_partial_reset_boundaries() -> None:
    storage = make_storage(num_envs=2, rollout_length=5)
    add_step(storage, 0)
    add_step(storage, 1, dones=torch.tensor([True, False]))
    add_step(storage, 2)
    storage.reset_envs([1])
    add_step(storage, 3)
    add_step(storage, 4)

    batches = list(
        storage.sequence_mini_batch_generator(
            num_mini_batches=2,
            sequence_length=2,
            num_epochs=1,
            shuffle=False,
        )
    )
    descriptors: list[tuple[int, int, int]] = []
    for batch in batches:
        assert batch.teacher_latent_A.shape[0] == 2
        assert batch.teacher_action_mean.shape[0] == 2
        assert batch.masks.dtype == torch.bool
        for column in range(batch.env_ids.numel()):
            env_id = int(batch.env_ids[column])
            start = int(batch.start_steps[column])
            length = int(batch.sequence_lengths[column])
            descriptors.append((env_id, start, start + length))
            assert batch.masks[:, column, 0].tolist() == [True] * length + [False] * (2 - length)
            expected_steps = torch.arange(start, start + length, dtype=torch.float32)
            expected_labels = env_id * 100.0 + expected_steps
            torch.testing.assert_close(batch.teacher_latent_A[:length, column, 0], expected_labels)
            torch.testing.assert_close(batch.teacher_action_mean[:length, column, 0], expected_labels)
            # A terminal transition may only be the final valid item.  A new
            # episode marker may only appear at the first valid item.
            assert not torch.any(batch.dones[: max(length - 1, 0), column])
            assert not torch.any(batch.episode_starts[1:length, column])
            if bool(storage.episode_starts[start, env_id]):
                assert torch.count_nonzero(batch.initial_student_hidden[column]) == 0

    assert descriptors == [
        (0, 0, 2),
        (0, 2, 4),
        (0, 4, 5),
        (1, 0, 2),
        (1, 2, 3),
        (1, 3, 5),
    ]


def test_padding_is_zero_and_gru_hidden_conversion_is_explicit() -> None:
    storage = make_storage(num_envs=1, rollout_length=3)
    for step in range(3):
        add_step(storage, step)
    batches = list(
        storage.sequence_mini_batch_generator(
            num_mini_batches=1,
            sequence_length=2,
            shuffle=False,
        )
    )
    batch = batches[0]
    assert batch.sequence_lengths.tolist() == [2, 1]
    assert not batch.masks[1, 1, 0]
    assert torch.count_nonzero(batch.teacher_latent_A[1, 1]) == 0
    assert torch.count_nonzero(batch.student_observations["proprioception"][1, 1]) == 0
    assert batch.gru_initial_hidden_state().shape == (2, 2, 4)


def test_4096_environment_capacity_estimate_does_not_allocate_capacity() -> None:
    sample = TensorDict(
        {
            "proprioception": torch.zeros(1, 48, dtype=torch.float32),
            "observed_history": torch.zeros(1, 5, 4, 16, 96, dtype=torch.float16),
            "history_valid": torch.zeros(1, 5, 16, 96, dtype=torch.uint8),
        },
        batch_size=[1],
    )
    estimate = M2MSequenceRolloutStorage.estimate_memory(
        num_envs=4096,
        num_transitions_per_env=24,
        student_obs=sample,
        allowed_student_keys=STUDENT_KEYS,
        hidden_state_shape=(1, 128),
    )
    time_env = 4096 * 24
    expected = {
        "student_observations.proprioception": time_env * 48 * 4,
        "student_observations.observed_history": time_env * 5 * 4 * 16 * 96 * 2,
        "student_observations.history_valid": time_env * 5 * 16 * 96,
        "teacher_latents_A64": time_env * 64 * 4,
        "teacher_action_means_29": time_env * 29 * 4,
        "dones": time_env,
        "episode_starts": time_env,
        "student_hidden_states": time_env * 128 * 4,
        "pending_episode_starts": 4096,
    }
    assert dict(estimate.field_bytes) == expected
    assert estimate.total_bytes == sum(expected.values())
    assert math.isclose(estimate.total_gib, estimate.total_bytes / 1024**3)
    audit = estimate.audit()
    assert audit["num_envs"] == 4096
    assert audit["includes_teacher_map"] is False
    assert audit["includes_next_observation"] is False


def test_float_frame_compression_is_explicit_auditable_and_optionally_restored() -> None:
    num_envs = 2
    observations = TensorDict(
        {
            "policy": torch.arange(num_envs * 3, dtype=torch.float32).view(num_envs, 3),
            "m2m_student_frame": torch.linspace(0.0, 1.0, num_envs * 1 * 4 * 2 * 3, dtype=torch.float32).view(
                num_envs, 1, 4, 2, 3
            ),
        },
        batch_size=[num_envs],
    )
    storage = M2MSequenceRolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=1,
        student_obs=observations,
        allowed_student_keys=("policy", "m2m_student_frame"),
        hidden_state_shape=(1, 8),
        student_obs_storage_dtypes={"m2m_student_frame": torch.float16},
    )
    storage.add_transition(
        M2MSequenceTransition(
            student_observations=observations,
            teacher_latent_A=torch.zeros(num_envs, 64),
            teacher_action_mean=torch.zeros(num_envs, 29),
            dones=torch.zeros(num_envs, dtype=torch.bool),
            student_hidden_state=torch.zeros(num_envs, 1, 8),
        )
    )
    assert storage.student_observations["policy"].dtype == torch.float32
    assert storage.student_observations["m2m_student_frame"].dtype == torch.float16
    assert storage.observation_storage_audit() == {
        "policy": {
            "source_dtype": "torch.float32",
            "storage_dtype": "torch.float32",
            "compressed": False,
        },
        "m2m_student_frame": {
            "source_dtype": "torch.float32",
            "storage_dtype": "torch.float16",
            "compressed": True,
        },
    }

    stored_batch = next(storage.sequence_mini_batch_generator(num_mini_batches=1, sequence_length=1, shuffle=False))
    assert stored_batch.student_observations["m2m_student_frame"].dtype == torch.float16
    restored_batch = next(
        storage.sequence_mini_batch_generator(
            num_mini_batches=1,
            sequence_length=1,
            shuffle=False,
            restore_observation_dtypes=True,
        )
    )
    assert restored_batch.student_observations["m2m_student_frame"].dtype == torch.float32
    torch.testing.assert_close(
        restored_batch.student_observations["m2m_student_frame"][0],
        observations["m2m_student_frame"],
        rtol=1e-3,
        atol=5e-4,
    )


def test_actual_strict_frame_4096_memory_receipt_and_legacy_comparison() -> None:
    # C02 strict student output: float32 [N,1,4,16,96].  Keeping this sample
    # float32 is intentional: the receipt must expose, not hide, its 2.25 GiB
    # uncompressed rollout cost at T=24 and N=4096.
    sample = TensorDict(
        {
            "policy": torch.zeros(1, 96, dtype=torch.float32),
            "m2m_student_frame": torch.zeros(1, 1, 4, 16, 96, dtype=torch.float32),
        },
        batch_size=[1],
    )
    kwargs = {
        "num_envs": 4096,
        "num_transitions_per_env": 24,
        "student_obs": sample,
        "allowed_student_keys": ("policy", "m2m_student_frame"),
        "hidden_state_shape": (1, 128),
    }
    uncompressed = M2MSequenceRolloutStorage.estimate_memory(**kwargs)
    compressed = M2MSequenceRolloutStorage.estimate_memory(
        **kwargs,
        student_obs_storage_dtypes={"m2m_student_frame": torch.float16},
    )
    frame_elements = 4096 * 24 * 1 * 4 * 16 * 96
    assert uncompressed.field_bytes["student_observations.m2m_student_frame"] == frame_elements * 4
    assert math.isclose(frame_elements * 4 / 1024**3, 2.25)
    assert compressed.field_bytes["student_observations.m2m_student_frame"] == frame_elements * 2
    assert compressed.field_dtypes["student_observations.m2m_student_frame"] == (
        "source=torch.float32,storage=torch.float16"
    )

    comparison = M2MSequenceRolloutStorage.compare_memory_with_legacy_distillation(
        **kwargs,
        legacy_obs=sample,
        student_obs_storage_dtypes={"m2m_student_frame": torch.float16},
    )
    assert comparison.legacy_field_bytes["observations.m2m_student_frame"] == frame_elements * 4
    assert comparison.legacy_field_bytes["next_observations.m2m_student_frame"] == frame_elements * 4
    report = comparison.audit()
    assert report["m2m_sequence_storage"]["total_bytes"] == compressed.total_bytes
    assert report["legacy_distillation_rollout"]["total_bytes"] == (comparison.legacy_total_bytes)
    assert comparison.difference_bytes > 0


def test_actual_memory_report_matches_allocated_tensors() -> None:
    storage = make_storage(num_envs=2, rollout_length=3)
    estimate = storage.memory_estimate()
    tensors = [
        *storage.student_observations.values(include_nested=True, leaves_only=True),
        storage.teacher_latent_A,
        storage.teacher_action_mean,
        storage.dones,
        storage.episode_starts,
        storage.student_hidden_states,
        storage._next_episode_starts,
    ]
    expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    assert estimate.total_bytes == expected_bytes


def test_validation_and_overflow_fail_closed() -> None:
    storage = make_storage(num_envs=2, rollout_length=1)
    add_step(storage, 0)
    with pytest.raises(OverflowError, match="overflow"):
        add_step(storage, 1)
    with pytest.raises(IndexError, match="env_ids"):
        storage.reset_envs([2])
    with pytest.raises(TypeError, match="integers"):
        storage.reset_envs([0.5])
    with pytest.raises(ValueError, match="exceeds"):
        list(storage.sequence_mini_batch_generator(num_mini_batches=3, sequence_length=4))


def test_old_rollout_storage_remains_available_and_behavior_is_untouched() -> None:
    observations = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    old_storage = RolloutStorage(
        training_type="distillation",
        num_envs=2,
        num_transitions_per_env=4,
        obs=observations,
        actions_shape=(29,),
    )
    assert old_storage.next_observations.shape == torch.Size([4, 2])
    assert old_storage.privileged_actions.shape == (4, 2, 29)
    assert not isinstance(old_storage, M2MSequenceRolloutStorage)
