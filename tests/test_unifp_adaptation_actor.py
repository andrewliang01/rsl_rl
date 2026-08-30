"""Regression tests for the reusable legacy UniFP history actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import UniFPAdaptationActor


def test_history_actor_prediction_action_and_export_contract():
    batch_size = 3
    frame_dim = 48
    history_length = 32
    obs = TensorDict(
        {"policy": torch.randn(batch_size, history_length * frame_dim)},
        batch_size=[batch_size],
    )
    actor = UniFPAdaptationActor(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=18,
        history_length=history_length,
        num_pred_obs=12,
        encoder_hidden_dims=(64, 32),
        decoder_hidden_dims=(32,),
        hidden_dims=(64, 32),
        obs_normalization=False,
    ).eval()

    prediction = actor.predict_obs_pred(obs)
    actions = actor(obs)
    exported = actor.as_jit().eval()
    exported_actions = exported(obs["policy"])

    assert prediction.shape == (batch_size, 12)
    assert actions.shape == (batch_size, 18)
    assert exported.input_names == ["obs"]
    assert exported.output_names == ["actions"]
    assert torch.allclose(exported_actions, actions, atol=1.0e-6, rtol=1.0e-6)


def test_separated_history_actor_exposes_explicit_heads_and_reconstruction():
    batch_size = 4
    history_length = 32
    obs = TensorDict(
        {
            "history": torch.randn(batch_size, history_length * 42),
            "current": torch.randn(batch_size, 47),
        },
        batch_size=[batch_size],
    )
    actor = UniFPAdaptationActor(
        obs=obs,
        obs_groups={"actor": ["current", "history"]},
        obs_set="actor",
        output_dim=12,
        history_length=history_length,
        history_set="history",
        current_obs_set="current",
        latent_dim=64,
        num_pred_obs=11,
        use_prediction_in_actor=True,
        prediction_part_dims=(3, 4, 4),
        prediction_part_activations=("identity", "identity", "sigmoid"),
        num_reconstruction_obs=30,
        encoder_hidden_dims=(64, 32),
        decoder_hidden_dims=(32,),
        hidden_dims=(64, 32),
        obs_normalization=False,
    ).eval()

    raw_prediction = actor.predict_obs_pred(obs)
    reconstruction = actor.predict_reconstruction(obs)
    policy_input = actor.get_latent(obs)
    actions = actor(obs)
    exported = actor.as_jit().eval()
    scripted = torch.jit.script(exported)
    exported_actions = scripted(obs["history"], obs["current"])

    assert raw_prediction.shape == (batch_size, 11)
    assert reconstruction.shape == (batch_size, 30)
    assert policy_input.shape == (batch_size, 47 + 64 + 11)
    contact_probability = policy_input[:, -4:]
    assert torch.all((0.0 <= contact_probability) & (contact_probability <= 1.0))
    assert actions.shape == (batch_size, 12)
    assert exported.input_names == ["history", "current_obs"]
    assert torch.allclose(exported_actions, actions, atol=1.0e-6, rtol=1.0e-6)
