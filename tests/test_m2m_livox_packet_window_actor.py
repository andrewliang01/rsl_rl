from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.m2m_livox_packet_window_actor import M2MLivoxPacketWindowActor


def _obs(history_packets: int, batch: int = 3) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch, 96),
            "m2m_livox_packet_history": torch.zeros(
                batch, history_packets, 1, 4, 16, 96, dtype=torch.float16
            ),
        },
        batch_size=[batch],
    )


def _actor(history_packets: int) -> M2MLivoxPacketWindowActor:
    obs = _obs(history_packets)
    return M2MLivoxPacketWindowActor(
        obs,
        {"actor": ["policy", "m2m_livox_packet_history"]},
        "actor",
        29,
        packet_window_set="m2m_livox_packet_history",
        proprio_sets=["policy"],
        history_packets=history_packets,
        frame_near_range_m=0.1,
        frame_far_range_m=1.85699,
        frame_message_period_s=0.1,
        frame_max_age_s=1.0,
        frame_age_semantics="winning_subframe_age_20ms",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


@pytest.mark.parametrize("history_packets", [1, 3, 5, 10])
def test_packet_window_actor_forward_and_receipt(history_packets: int) -> None:
    actor = _actor(history_packets)
    obs = _obs(history_packets)
    action = actor(obs)
    assert action.shape == (3, 29)
    assert torch.isfinite(action).all()
    receipt = actor.architecture_receipt()
    assert receipt["actor_inputs"]["history_packets"] == history_packets
    assert receipt["actor_inputs"]["history_span_s"] == pytest.approx(
        (history_packets - 1) * 0.1
    )
    assert receipt["actor_inputs"]["uses_map"] is False
    assert receipt["actor_inputs"]["uses_ground_truth_pose"] is False


def test_all_horizons_have_identical_parameter_count() -> None:
    counts = {
        history: sum(parameter.numel() for parameter in _actor(history).parameters())
        for history in (1, 3, 5, 10)
    }
    assert len(set(counts.values())) == 1, counts


def test_packet_window_actor_rejects_wrong_history_shape() -> None:
    actor = _actor(5)
    with pytest.raises(ValueError, match="packet history shape"):
        actor(_obs(3))
