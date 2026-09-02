"""Contracts for the unified fixed-window UniFP GRU actor."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import UniFPGRUHistoryActor


def _term_major_observation() -> tuple[TensorDict, torch.Tensor]:
    term_a = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
    )
    term_b = torch.tensor([[[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]])
    stacked = torch.cat((term_a.flatten(start_dim=1), term_b.flatten(start_dim=1)), dim=-1)
    chronological = torch.cat((term_a, term_b), dim=-1)
    return TensorDict({"policy": stacked}, batch_size=[1]), chronological


def _make_actor(obs: TensorDict) -> UniFPGRUHistoryActor:
    return UniFPGRUHistoryActor(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=3,
        history_length=6,
        history_term_dims=(2, 1),
        latent_dim=5,
        gru_hidden_dim=7,
        gru_num_layers=1,
        num_pred_obs=4,
        prediction_part_dims=(2, 2),
        prediction_part_activations=("identity", "sigmoid"),
        num_reconstruction_obs=5,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        hidden_dims=(8,),
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ).eval()


def test_gru_history_uses_one_complete_term_major_window_and_exports():
    obs, chronological = _term_major_observation()
    actor = _make_actor(obs)

    frames = actor._term_major_to_frames(obs["policy"])
    history, current = actor._history_and_current(obs)
    prediction = actor.predict_obs_pred(obs)
    reconstruction = actor.predict_reconstruction(obs)
    policy_input = actor.get_latent(obs)
    actions = actor(obs)
    exported = actor.as_jit().eval()
    scripted = torch.jit.script(exported)

    torch.testing.assert_close(frames, chronological)
    torch.testing.assert_close(history, chronological)
    torch.testing.assert_close(current, chronological[:, -1])
    assert not hasattr(actor, "short_encoder")
    assert not hasattr(actor, "long_encoder")
    assert prediction.shape == (1, 4)
    assert reconstruction.shape == (1, 5)
    assert policy_input.shape == (1, 3 + 5 + 4)
    assert actions.shape == (1, 3)
    assert exported.input_names == ["obs"]
    torch.testing.assert_close(scripted(obs["policy"]), actions)


def test_gru_history_rejects_a_policy_buffer_with_the_wrong_size():
    obs = TensorDict({"policy": torch.zeros(2, 17)}, batch_size=[2])
    with pytest.raises(ValueError, match="does not match"):
        UniFPGRUHistoryActor(
            obs=obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=3,
            history_length=6,
            history_term_dims=(2, 1),
            encoder_hidden_dims=(8,),
            decoder_hidden_dims=(8,),
            hidden_dims=(8,),
        )
