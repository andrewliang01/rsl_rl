"""Contract tests for the map-free deformable Livox memory actor."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models.m2m_livox_deformable_memory_actor import (
    M2MLivoxDeformableMemoryActor,
)
from rsl_rl.models.m2m_sequence_compatible_critic import M2MSequenceCompatibleCritic
from rsl_rl.storage import RolloutStorage


def _obs(batch: int, *, new_frame: bool, value: float = 0.8) -> TensorDict:
    frame = torch.zeros(batch, 1, 4, 16, 96)
    frame[:, 0, 0] = value
    frame[:, 0, 1] = 1.0
    frame[:, 0, 2] = 0.02
    frame[:, 0, 3] = float(new_frame)
    return TensorDict(
        {"policy": torch.randn(batch, 96), "strict_livox": frame},
        batch_size=[batch],
    )


def _actor(batch: int = 3) -> M2MLivoxDeformableMemoryActor:
    return M2MLivoxDeformableMemoryActor(
        _obs(batch, new_frame=False),
        {"actor": ["policy", "strict_livox"]},
        "actor",
        29,
        strict_frame_set="strict_livox",
        proprio_sets=("policy",),
        frame_near_range_m=0.1,
        frame_far_range_m=1.85699,
        frame_max_age_s=10.0,
        frame_age_semantics="winning_subframe_age_20ms",
        spatial_hidden_channels=(8, 8),
        deformable_samples=2,
        motion_dim=16,
        gru_hidden_dim=32,
        latent_hidden_dim=32,
    )


def _full_obs(batch: int, *, new_frame: bool, seed: int) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    obs = _obs(batch, new_frame=new_frame)
    obs["critic"] = torch.randn(batch, 99, generator=generator)
    obs["height_scan_critic"] = torch.randn(batch, 1, 28, 20, generator=generator)
    return obs


def test_state_updates_only_on_new_packet_and_reset_is_per_environment() -> None:
    torch.manual_seed(4)
    actor = _actor()
    held_obs = _obs(3, new_frame=False)
    action_before = actor(held_obs)
    state_before = actor.get_hidden_state().clone()
    action_after = actor(_obs(3, new_frame=False, value=1.2))
    state_after = actor.get_hidden_state().clone()
    assert torch.equal(state_before, state_after)
    assert not torch.equal(action_before, action_after)  # live 50 Hz proprioception branch

    actor(_obs(3, new_frame=True, value=1.2))
    updated = actor.get_hidden_state().clone()
    assert not torch.equal(updated, state_after)
    actor.reset(torch.tensor([False, True, False]))
    reset = actor.get_hidden_state()
    assert torch.count_nonzero(reset[:, 1]) == 0
    assert torch.equal(reset[:, 0], updated[:, 0])
    assert torch.equal(reset[:, 2], updated[:, 2])


def test_spatial_evidence_changes_latent_and_offsets_are_bounded() -> None:
    torch.manual_seed(5)
    actor = _actor(batch=2)
    latent_a = actor.predict_latent(_obs(2, new_frame=True, value=0.3))
    actor.reset()
    latent_b = actor.predict_latent(_obs(2, new_frame=True, value=1.5))
    assert latent_a.shape == (2, 64)
    assert not torch.allclose(latent_a, latent_b)
    offsets = actor._last_offsets
    assert offsets is not None
    assert offsets[:, :, 0].abs().max() <= actor.max_elevation_offset_cells + 1.0e-6
    assert offsets[:, :, 1].abs().max() <= actor.max_azimuth_offset_cells + 1.0e-6


def test_padded_recurrent_forward_matches_stepwise_valid_prefixes() -> None:
    torch.manual_seed(7)
    actor = _actor(batch=2)
    steps = [_obs(2, new_frame=flag) for flag in (True, False, True)]
    stacked = TensorDict.stack(steps, dim=0)
    masks = torch.ones(3, 2, dtype=torch.bool)
    initial = torch.zeros(1, 2, actor.hidden_state_dim)

    padded_actions = actor(stacked, masks=masks, hidden_state=initial)
    actor.reset(hidden_state=initial.clone())
    expected = []
    for step in steps:
        expected.append(actor(step))
    expected_time_major = torch.stack(expected)
    assert padded_actions.shape == expected_time_major.shape
    torch.testing.assert_close(padded_actions, expected_time_major)


def test_receipt_declares_no_map_pose_or_future_input() -> None:
    receipt = _actor(batch=1).architecture_receipt()
    assert receipt["actor_inputs"]["uses_map"] is False
    assert receipt["actor_inputs"]["uses_ground_truth_pose"] is False
    assert receipt["actor_inputs"]["uses_future_frames"] is False
    assert receipt["actor_inputs"]["updates_only_on_new_10hz_packet"] is True
    assert receipt["temporal_model"].endswith("persistent_gru")


def test_actual_recurrent_ppo_collection_and_update_runs_end_to_end() -> None:
    num_envs, time_steps = 4, 4
    obs = _full_obs(num_envs, new_frame=True, seed=30)
    actor = M2MLivoxDeformableMemoryActor(
        obs,
        {"actor": ["policy", "strict_livox"]},
        "actor",
        29,
        strict_frame_set="strict_livox",
        proprio_sets=("policy",),
        frame_near_range_m=0.1,
        frame_far_range_m=1.85699,
        frame_max_age_s=10.0,
        frame_age_semantics="winning_subframe_age_20ms",
        spatial_hidden_channels=(8, 8),
        deformable_samples=2,
        motion_dim=16,
        gru_hidden_dim=32,
        latent_hidden_dim=32,
    )
    critic = M2MSequenceCompatibleCritic(
        obs=obs,
        obs_groups={"critic": ["critic", "height_scan_critic"]},
        obs_set="critic",
        output_dim=1,
        hidden_dims=(16,),
        obs_normalization=False,
        vision_feature_dim=8,
        cnn_hidden_dims=(4,),
        cnn_kernel_sizes=(3,),
        cnn_strides=(2,),
        prop_feature_dim=8,
        prop_hidden_dims=(16,),
    )
    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=time_steps,
        obs=obs,
        actions_shape=(29,),
        device="cpu",
    )
    algorithm = PPO(
        actor=actor,
        critic=critic,
        storage=storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        desired_kl=None,
        device="cpu",
    )
    for step in range(time_steps):
        with torch.no_grad():
            algorithm.act(obs)
        next_obs = _full_obs(num_envs, new_frame=step % 2 == 1, seed=31 + step)
        rewards = torch.linspace(0.0, 1.0, num_envs)
        dones = torch.zeros(num_envs, dtype=torch.uint8)
        if step == 1:
            dones[0] = 1
        algorithm.process_env_step(next_obs, rewards, dones, extras={})
        obs = next_obs
    algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert set(losses) == {"value", "surrogate", "entropy"}
    assert all(math.isfinite(value) for value in losses.values())
