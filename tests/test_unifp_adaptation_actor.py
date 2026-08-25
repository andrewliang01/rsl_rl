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
