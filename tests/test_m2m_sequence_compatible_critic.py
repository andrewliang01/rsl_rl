from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models.m2m_sequence_compatible_critic import M2MSequenceCompatibleCritic
from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


_OBS_GROUPS = {
    "actor": ["policy"],
    "critic": ["critic", "height_scan_critic"],
}
_MODEL_KWARGS = {
    "hidden_dims": (16,),
    "obs_normalization": False,
    "vision_feature_dim": 8,
    "cnn_hidden_dims": (4,),
    "cnn_kernel_sizes": (3,),
    "cnn_strides": (2,),
    "prop_feature_dim": 8,
    "prop_hidden_dims": (16,),
}


def _make_obs(batch_size: tuple[int, ...], *, seed: int = 0) -> TensorDict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return TensorDict(
        {
            "policy": torch.randn(*batch_size, 6, generator=generator),
            "critic": torch.randn(*batch_size, 99, generator=generator),
            "height_scan_critic": torch.randn(*batch_size, 1, 28, 20, generator=generator),
        },
        batch_size=batch_size,
        device="cpu",
    )


def _make_critic(obs: TensorDict) -> M2MSequenceCompatibleCritic:
    return M2MSequenceCompatibleCritic(
        obs=obs,
        obs_groups=_OBS_GROUPS,
        obs_set="critic",
        output_dim=1,
        **_MODEL_KWARGS,
    )


def _trajectory_batch() -> tuple[TensorDict, TensorDict, torch.Tensor]:
    time_steps, num_envs = 4, 3
    original = _make_obs((time_steps, num_envs), seed=17)
    dones = torch.zeros(time_steps, num_envs, 1, dtype=torch.uint8)
    dones[1, 0] = 1
    dones[2, 1] = 1
    padded, masks = split_and_pad_trajectories(original, dones)
    return original, padded, masks


def test_constructor_enforces_exact_g1_critic_contract() -> None:
    obs = _make_obs((4,))

    with pytest.raises(ValueError, match="must be exactly"):
        M2MSequenceCompatibleCritic(
            obs=obs,
            obs_groups={**_OBS_GROUPS, "critic": ["height_scan_critic", "critic"]},
            obs_set="critic",
            output_dim=1,
            **_MODEL_KWARGS,
        )

    bad_prop = obs.clone()
    bad_prop["critic"] = torch.zeros(4, 98)
    with pytest.raises(ValueError, match="exact shape"):
        _make_critic(bad_prop)

    bad_height = obs.clone()
    bad_height["height_scan_critic"] = torch.zeros(4, 1, 27, 20)
    with pytest.raises(ValueError, match="exact shape"):
        _make_critic(bad_height)

    with pytest.raises(ValueError, match="output_dim"):
        M2MSequenceCompatibleCritic(
            obs=obs,
            obs_groups=_OBS_GROUPS,
            obs_set="critic",
            output_dim=2,
            **_MODEL_KWARGS,
        )


def test_forward_rejects_runtime_contract_drift_and_hidden_state() -> None:
    obs = _make_obs((4,))
    critic = _make_critic(obs)

    bad_obs = obs.clone()
    bad_obs["critic"] = torch.zeros(4, 100)
    with pytest.raises(ValueError, match="exact shape"):
        critic(bad_obs)

    with pytest.raises(ValueError, match="rejects hidden_state"):
        critic(obs, hidden_state=torch.zeros(1, 4, 8))


def test_step_path_is_state_dict_and_output_compatible_with_legacy_critic() -> None:
    obs = _make_obs((5,), seed=4)
    torch.manual_seed(41)
    legacy = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups=_OBS_GROUPS,
        obs_set="critic",
        output_dim=1,
        elevation_set="height_scan_critic",
        vision_spatial_size=(28, 20),
        elevation_history_length=1,
        cnn_observation_type="elevationmap",
        elevation_encoder_type="cnn",
        distribution_cfg=None,
        **_MODEL_KWARGS,
    )
    replacement = _make_critic(obs)

    assert tuple(legacy.state_dict()) == tuple(replacement.state_dict())
    incompatible = replacement.load_state_dict(legacy.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    legacy.eval()
    replacement.eval()
    with torch.no_grad():
        legacy_values = legacy(obs)
        replacement_values = replacement(obs)
    torch.testing.assert_close(replacement_values, legacy_values, rtol=0.0, atol=0.0)


def test_sequence_and_step_paths_are_elementwise_equivalent() -> None:
    original, padded, masks = _trajectory_batch()
    critic = _make_critic(original[0]).eval()

    with torch.no_grad():
        expected = torch.stack([critic(original[step]) for step in range(original.batch_size[0])])
        direct_sequence = critic(original)
        recurrent_ppo_sequence = critic(padded, masks=masks, hidden_state=None)

    assert recurrent_ppo_sequence.shape == (4, 3, 1)
    torch.testing.assert_close(direct_sequence, expected, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(recurrent_ppo_sequence, expected, rtol=1.0e-6, atol=1.0e-7)


def test_padded_values_do_not_affect_output_or_gradient() -> None:
    _, padded, masks = _trajectory_batch()
    critic = _make_critic(padded[0]).eval()

    clean = padded.clone()
    poisoned = padded.clone()
    tracked: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key in ("critic", "height_scan_critic"):
        value = poisoned[key].detach().clone()
        expanded_mask = masks.reshape(*masks.shape, *([1] * (value.ndim - masks.ndim))).expand_as(value)
        value = torch.where(expanded_mask, value, torch.full_like(value, 1.0e6)).requires_grad_(True)
        poisoned[key] = value
        tracked[key] = (value, expanded_mask)

    with torch.no_grad():
        clean_values = critic(clean, masks=masks)
    poisoned_values = critic(poisoned, masks=masks)
    torch.testing.assert_close(poisoned_values.detach(), clean_values, rtol=1.0e-6, atol=1.0e-7)
    poisoned_values.square().sum().backward()

    for value, expanded_mask in tracked.values():
        assert value.grad is not None
        assert torch.count_nonzero(value.grad[~expanded_mask]) == 0


def test_mask_contract_is_fail_closed() -> None:
    _, padded, masks = _trajectory_batch()
    critic = _make_critic(padded[0])

    with pytest.raises(TypeError, match="torch.bool"):
        critic(padded, masks=masks.float())

    non_right_padded = masks.clone()
    non_right_padded[:, 0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="right-padded"):
        critic(padded, masks=non_right_padded)


def test_actual_recurrent_ppo_update_signature_runs_end_to_end() -> None:
    """Exercise the exact PPO actor/critic call, not only a direct wrapper call."""
    num_envs, time_steps, num_actions = 4, 4, 2
    obs = _make_obs((num_envs,), seed=101)
    actor = RNNModel(
        obs=obs,
        obs_groups=_OBS_GROUPS,
        obs_set="actor",
        output_dim=num_actions,
        hidden_dims=(16,),
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        rnn_type="gru",
        rnn_hidden_dim=8,
        rnn_num_layers=1,
    )
    critic = _make_critic(obs)
    storage = RolloutStorage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=time_steps,
        obs=obs,
        actions_shape=(num_actions,),
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
        next_obs = _make_obs((num_envs,), seed=102 + step)
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
