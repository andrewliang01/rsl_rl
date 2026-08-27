"""Numerical and interface checks for the ECMM + UniFP fusion actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import PropMLPElevationUniFPFusionModel


def _actor(
    batch_size: int = 3, num_pred_obs: int = 6
) -> tuple[PropMLPElevationUniFPFusionModel, TensorDict]:
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 42),
            "estimator_history": torch.randn(batch_size, 32 * 42),
            "height_scan_policy": torch.rand(batch_size, 5, 24, 32) * 5.95 + 0.05,
        },
        batch_size=[batch_size],
    )
    actor = PropMLPElevationUniFPFusionModel(
        obs=obs,
        obs_groups={"actor": ["policy", "estimator_history", "height_scan_policy"]},
        obs_set="actor",
        output_dim=12,
        history_set="estimator_history",
        history_length=32,
        estimator_latent_dim=16,
        estimator_hidden_dims=(32, 24),
        decoder_hidden_dims=(16,),
        num_pred_obs=num_pred_obs,
        history_normalization=False,
        elevation_set="height_scan_policy",
        cnn_observation_type="depthcamera",
        depth_camera_near=0.05,
        depth_camera_far=6.0,
        vision_spatial_size=(24, 32),
        vision_feature_dim=16,
        elevation_history_length=5,
        cnn_hidden_dims=(4, 8),
        cnn_kernel_sizes=(3, 3),
        cnn_strides=(2, 2),
        prop_feature_dim=16,
        prop_hidden_dims=(24,),
        hidden_dims=(32, 24),
        activation="elu",
        obs_normalization=False,
    )
    actor.eval()
    return actor, obs


def test_ecmm_unifp_actor_predicts_twist6_and_leg_actions():
    actor, obs = _actor()

    twist = actor.predict_obs_pred(obs)
    actions = actor(obs)

    assert twist.shape == (3, 6)
    assert actions.shape == (3, 12)
    assert torch.isfinite(twist).all()
    assert torch.isfinite(actions).all()


def test_ecmm_unifp_actor_can_predict_only_base_linear_velocity():
    actor, obs = _actor(num_pred_obs=3)

    base_linear_velocity = actor.predict_obs_pred(obs)
    actions = actor(obs)

    assert base_linear_velocity.shape == (3, 3)
    assert actions.shape == (3, 12)
    assert torch.isfinite(base_linear_velocity).all()
    assert torch.isfinite(actions).all()


def test_deployment_wrapper_matches_actor_and_exposes_three_inputs():
    actor, obs = _actor(batch_size=2)
    exported = actor.as_jit().eval()

    expected = actor(obs)
    actual = exported(obs["policy"], obs["estimator_history"], obs["height_scan_policy"])

    assert exported.input_names == ["current_proprio", "proprio_history", "depth_history"]
    assert exported.output_names == ["actions"]
    assert actual.shape == (2, 12)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_history_contract_and_twist_width_are_rejected_early():
    actor, obs = _actor(batch_size=1)
    assert actor.history_dim == 32 * 42
    assert actor.num_history_frame_obs == 42

    bad_obs = obs.clone()
    bad_obs["estimator_history"] = torch.randn(1, 32 * 42 + 1)
    try:
        PropMLPElevationUniFPFusionModel(
            obs=bad_obs,
            obs_groups={"actor": ["policy", "estimator_history", "height_scan_policy"]},
            obs_set="actor",
            output_dim=12,
            history_length=32,
            num_pred_obs=6,
            elevation_set="height_scan_policy",
            vision_spatial_size=(24, 32),
            elevation_history_length=5,
        )
    except ValueError as exc:
        assert "not divisible" in str(exc)
    else:
        raise AssertionError("invalid flattened history width was accepted")

    try:
        _actor(batch_size=1, num_pred_obs=4)
    except ValueError as exc:
        assert "three-dimensional chunks" in str(exc)
    else:
        raise AssertionError("non-3D-aligned prediction width was accepted")
