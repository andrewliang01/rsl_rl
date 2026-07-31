import torch

from rsl_rl.utils.raw_event_pies import (
    PIES_EVENT_WINDOW_S,
    RawRayEventPacket,
    latest_event_raster_sha256,
    reduce_raw_event_packets_to_latest_raster,
)


def _packet(indices: list[int]) -> RawRayEventPacket:
    cell = torch.tensor([[0, 0, 1, 1, 2, 2, 3, -1, 3]], dtype=torch.long)
    distance = torch.tensor(
        [[4.0, 2.0, 3.0, 1.5, 5.0, 1.0, 8.0, 0.0, 8.0]]
    )
    age = torch.tensor(
        [[0.30, 0.10, 0.20, 0.20, 0.60, 0.05, 0.12, 0.00, 0.12]]
    )
    event_id = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16, 17, 2]], dtype=torch.long
    )
    valid = torch.tensor(
        [[True, True, True, True, True, True, True, False, True]]
    )
    delta = torch.tensor(
        [[[float(i), float(-i)]] for i in range(9)], dtype=torch.float32
    ).transpose(0, 1)
    index = torch.tensor(indices, dtype=torch.long)
    return RawRayEventPacket(
        cell_index=cell[:, index],
        range_m=distance[:, index],
        return_age_s=age[:, index],
        event_id=event_id[:, index],
        return_valid=valid[:, index],
        acquisition_delta_proprio=delta[:, index],
    )


def test_raw_latest_event_is_bitwise_and_hash_invariant_for_k1_k3_k5_irregular():
    # Every scheme carries the exact same event union. It covers same-cell
    # collisions, equal-age/different-range, equal-age/range/different-id, an
    # expired event, an invalid observation, an empty packet, and packet/order
    # permutations.
    packetizations = {
        "k1": [_packet([0, 1, 2, 3, 4, 5, 6, 7, 8])],
        "k3": [_packet([0, 3, 6]), _packet([8, 5, 2]), _packet([7, 4, 1])],
        "k5": [
            _packet([8, 0]),
            _packet([]),
            _packet([7, 2, 5]),
            _packet([6]),
            _packet([4, 3, 1]),
        ],
        "irregular_permuted": [
            _packet([7]),
            _packet([5, 8, 1, 4]),
            _packet([]),
            _packet([3, 0]),
            _packet([6, 2]),
        ],
    }
    rasters = {
        name: reduce_raw_event_packets_to_latest_raster(
            packets, spatial_size=(2, 2)
        )
        for name, packets in packetizations.items()
    }
    raster_a = rasters["k1"]
    fields = (
        "range_m",
        "return_valid",
        "return_age_s",
        "event_id",
        "acquisition_delta_proprio",
    )
    hashes = set()
    for raster in rasters.values():
        for field in fields:
            assert torch.equal(getattr(raster_a, field), getattr(raster, field))
        hashes.add(latest_event_raster_sha256(raster))
    assert hashes == {
        "e8d461a7b4e0d14bfa00bc6bfd95cd6e85a0c0f79f9a87b2007b838fb0b8c332"
    }
    # Cell 0: newest event. Cell 1: exact-age tie chooses nearest range.
    # Cell 2: the 0.60 s event is outside the fixed window; 0.05 s wins.
    assert torch.equal(
        raster_a.range_m,
        torch.tensor([[[[2.0, 1.5], [1.0, 8.0]]]]),
    )
    assert torch.equal(
        raster_a.event_id,
        torch.tensor([[[[11, 13], [15, 2]]]], dtype=torch.long),
    )
    assert torch.equal(
        raster_a.acquisition_delta_proprio,
        torch.tensor(
            [[[[1.0, 3.0], [5.0, 8.0]], [[-1.0, -3.0], [-5.0, -8.0]]]]
        ),
    )
    assert torch.equal(
        raster_a.return_age_s,
        torch.tensor([[[[0.10, 0.20], [0.05, 0.12]]]]),
    )
    assert PIES_EVENT_WINDOW_S == 0.5


def test_raw_latest_event_tie_break_uses_stable_event_id_not_packet_order():
    first = RawRayEventPacket(
        cell_index=torch.tensor([[0]], dtype=torch.long),
        range_m=torch.tensor([[2.0]]),
        return_age_s=torch.tensor([[0.1]]),
        event_id=torch.tensor([[9]], dtype=torch.long),
        return_valid=torch.tensor([[True]]),
        acquisition_delta_proprio=torch.tensor([[[9.0, -9.0]]]),
    )
    second = RawRayEventPacket(
        cell_index=torch.tensor([[0]], dtype=torch.long),
        range_m=torch.tensor([[2.0]]),
        return_age_s=torch.tensor([[0.1]]),
        event_id=torch.tensor([[3]], dtype=torch.long),
        return_valid=torch.tensor([[True]]),
        acquisition_delta_proprio=torch.tensor([[[3.0, -3.0]]]),
    )
    forward = reduce_raw_event_packets_to_latest_raster(
        [first, second], spatial_size=(1, 1)
    )
    reverse = reduce_raw_event_packets_to_latest_raster(
        [second, first], spatial_size=(1, 1)
    )
    assert forward.event_id.item() == 3
    assert torch.equal(forward.event_id, reverse.event_id)
    assert torch.equal(
        forward.acquisition_delta_proprio,
        torch.tensor([[[[3.0]], [[-3.0]]]]),
    )
    assert latest_event_raster_sha256(forward) == latest_event_raster_sha256(
        reverse
    )
