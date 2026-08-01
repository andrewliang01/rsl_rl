from __future__ import annotations

import torch
from dataclasses import replace

import pytest

from rsl_rl.utils.raw_event_pies import (
    LatestEventRaster,
    RawRayEventPacket,
    reduce_raw_event_packets_to_latest_raster,
)
from rsl_rl.utils.ray_event_observation import (
    pack_same_winner_pies_age_control_pair,
)


def _packet(indices: list[int]) -> RawRayEventPacket:
    cell = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    range_m = torch.tensor([[3.0, 2.0, 4.0, 1.5]])
    age_s = torch.tensor([[0.30, 0.10, 0.20, 0.20]])
    event_id = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    valid = torch.ones_like(cell, dtype=torch.bool)
    delta = torch.tensor(
        [[[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]]
    )
    index = torch.tensor(indices, dtype=torch.long)
    return RawRayEventPacket(
        cell_index=cell[:, index],
        range_m=range_m[:, index],
        return_age_s=age_s[:, index],
        event_id=event_id[:, index],
        return_valid=valid[:, index],
        acquisition_delta_proprio=delta[:, index],
    )


def _raster(packetization: list[list[int]]) -> LatestEventRaster:
    return reduce_raw_event_packets_to_latest_raster(
        [_packet(indices) for indices in packetization],
        spatial_size=(1, 2),
    )


def test_paired_arms_share_winner_range_validity_and_frame_exactly() -> None:
    """The paired actor views may differ only in visible return age."""
    raster = _raster([[0, 1], [2, 3]])
    pair = pack_same_winner_pies_age_control_pair(
        raster,
        frame_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    correct = pair.correct_age_observation
    zero = pair.age_zero_observation

    assert correct.shape == zero.shape == (1, 1, 5, 1, 2)
    for channel in (0, 1, 3, 4):
        assert torch.equal(correct[:, :, channel], zero[:, :, channel])
    assert torch.equal(correct[:, :, 2], raster.return_age_s)
    assert not torch.count_nonzero(zero[:, :, 2])
    assert torch.equal(pair.winner_event_id, raster.event_id)
    assert pair.winner_event_id.data_ptr() != raster.event_id.data_ptr()


def test_control_pair_is_packetization_invariant() -> None:
    """Repartitioning the same events cannot change either paired arm."""
    frame_valid = torch.ones(1, 1, dtype=torch.bool)
    forward = pack_same_winner_pies_age_control_pair(
        _raster([[0, 1], [2, 3]]),
        frame_valid=frame_valid,
    )
    permuted = pack_same_winner_pies_age_control_pair(
        _raster([[3], [1, 2], [], [0]]),
        frame_valid=frame_valid,
    )
    assert torch.equal(
        forward.correct_age_observation,
        permuted.correct_age_observation,
    )
    assert torch.equal(forward.age_zero_observation, permuted.age_zero_observation)
    assert torch.equal(forward.winner_event_id, permuted.winner_event_id)


@pytest.mark.parametrize("valid_id,invalid_id", [(-1, -1), (11, 4)])
def test_control_pair_rejects_winner_id_contract_breaks(
    valid_id: int,
    invalid_id: int,
) -> None:
    """Winner proof ids fail closed for valid and removed observations."""
    raster = _raster([[0, 1, 2, 3]])
    bad_ids = raster.event_id.clone()
    bad_ids[0, 0, 0, 0] = valid_id
    bad_raster = replace(raster, event_id=bad_ids)
    if invalid_id >= 0:
        bad_valid = raster.return_valid.clone()
        bad_valid[0, 0, 0, 1] = False
        bad_ids[0, 0, 0, 1] = invalid_id
        bad_raster = replace(
            raster,
            return_valid=bad_valid,
            event_id=bad_ids,
        )
    with pytest.raises(ValueError, match="winner id"):
        pack_same_winner_pies_age_control_pair(
            bad_raster,
            frame_valid=torch.ones(1, 1, dtype=torch.bool),
        )
