"""Contracts for the short/long-history UniFP actor."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import UniFPDualHistoryActor


def _term_major_observation() -> tuple[TensorDict, torch.Tensor]:
    term_a = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
    )
    term_b = torch.tensor([[[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]])
    stacked = torch.cat((term_a.flatten(start_dim=1), term_b.flatten(start_dim=1)), dim=-1)
    chronological = torch.cat((term_a, term_b), dim=-1)
    return TensorDict({"policy": stacked}, batch_size=[1]), chronological


def _make_actor(obs: TensorDict, encoder_type: str) -> UniFPDualHistoryActor:
    return UniFPDualHistoryActor(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=3,
        history_length=6,
        short_history_length=3,
        history_term_dims=(2, 1),
        exclude_current_from_history=True,
        short_latent_dim=4,
        long_latent_dim=5,
        long_encoder_type=encoder_type,
        gru_hidden_dim=7,
        gru_num_layers=1,
        num_pred_obs=4,
        prediction_part_dims=(2, 2),
        prediction_part_activations=("identity", "sigmoid"),
        num_reconstruction_obs=5,
        short_encoder_hidden_dims=(8,),
        long_encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        hidden_dims=(8,),
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ).eval()


@pytest.mark.parametrize("encoder_type", ["mlp", "gru"])
def test_dual_history_reorders_term_major_frames_and_exports(encoder_type: str):
    obs, chronological = _term_major_observation()
    actor = _make_actor(obs, encoder_type)

    frames = actor._term_major_to_frames(obs["policy"])
    short_history, long_history, current = actor._histories_and_current(obs)
    prediction = actor.predict_obs_pred(obs)
    reconstruction = actor.predict_reconstruction(obs)
    policy_input = actor.get_latent(obs)
    actions = actor(obs)
    exported = actor.as_jit().eval()
    scripted = torch.jit.script(exported)

    torch.testing.assert_close(frames, chronological)
    torch.testing.assert_close(long_history, chronological[:, :-1])
    torch.testing.assert_close(short_history, chronological[:, 3:5])
    torch.testing.assert_close(current, chronological[:, -1])
    assert prediction.shape == (1, 4)
    assert reconstruction.shape == (1, 5)
    assert policy_input.shape == (1, 3 + 4 + 5 + 4)
    assert actions.shape == (1, 3)
    assert exported.input_names == ["obs"]
    torch.testing.assert_close(scripted(obs["policy"]), actions)


def test_dual_history_rejects_a_policy_buffer_with_the_wrong_size():
    obs = TensorDict({"policy": torch.zeros(2, 17)}, batch_size=[2])
    with pytest.raises(ValueError, match="does not match"):
        UniFPDualHistoryActor(
            obs=obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=3,
            history_length=6,
            short_history_length=3,
            history_term_dims=(2, 1),
            short_encoder_hidden_dims=(8,),
            long_encoder_hidden_dims=(8,),
            decoder_hidden_dims=(8,),
            hidden_dims=(8,),
        )
