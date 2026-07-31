import pytest
import torch

from rsl_rl.modules.ray_event_ablation import RayEventAblationRouter


def _batch():
    range_m = torch.zeros(1, 3, 2, 3)
    valid = torch.zeros_like(range_m, dtype=torch.bool)
    age = torch.zeros_like(range_m)
    frame_valid = torch.tensor([[True, True, True]])
    packet_age = torch.tensor([[0.01, 0.11, 0.21]])
    values = (
        (0, 0, 0, 2.0, 0.04),
        (0, 0, 1, 3.0, 0.05),
        (1, 0, 0, 1.5, 0.14),
        (1, 1, 1, 4.0, 0.16),
        (2, 0, 0, 1.5, 0.28),  # equal range; earlier acquisition wins
        (2, 1, 2, 2.5, 0.29),
    )
    for frame, row, col, distance, return_age in values:
        range_m[0, frame, row, col] = distance
        valid[0, frame, row, col] = True
        age[0, frame, row, col] = return_age
    return range_m, valid, age, packet_age, frame_valid


def test_native_correct_is_identity_and_rejects_unused_rerender():
    inputs = _batch()
    router = RayEventAblationRouter()
    output = router(*inputs)
    assert torch.equal(output.range_m, inputs[0])
    assert torch.equal(output.return_valid, inputs[1])
    assert torch.equal(output.return_age_s, inputs[2])
    with pytest.raises(ValueError, match="rejects unused"):
        router(*inputs, rerender_range_m=inputs[0])


def test_rerender_requires_and_uses_one_aligned_triplet():
    inputs = _batch()
    router = RayEventAblationRouter(geometry="rerender")
    with pytest.raises(ValueError, match="requires aligned"):
        router(*inputs)
    rerender_range = inputs[0] + inputs[1].to(torch.float32) * 0.25
    rerender_age = inputs[2] + inputs[1].to(torch.float32) * 0.01
    output = router(
        *inputs,
        rerender_range_m=rerender_range,
        rerender_return_valid=inputs[1],
        rerender_return_age_s=rerender_age,
    )
    assert torch.equal(output.range_m, rerender_range)
    assert torch.equal(output.return_age_s, rerender_age)


def test_rerender_age_contract_is_checked_against_packet_time():
    inputs = _batch()
    rerender_age = inputs[2].clone()
    rerender_age[0, 0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="selected valid return age"):
        RayEventAblationRouter(geometry="rerender")(
            *inputs,
            rerender_range_m=inputs[0].clone(),
            rerender_return_valid=inputs[1].clone(),
            rerender_return_age_s=rerender_age,
        )


@pytest.mark.parametrize("geometry", ["native", "rerender"])
def test_shuffled_preserves_each_frame_multiset_and_changes_association(geometry):
    inputs = _batch()
    kwargs = {}
    if geometry == "rerender":
        kwargs = {
            "rerender_range_m": inputs[0].clone(),
            "rerender_return_valid": inputs[1].clone(),
            "rerender_return_age_s": inputs[2].clone(),
        }
    router = RayEventAblationRouter(
        geometry=geometry,
        time_association="shuffled",
        shuffle_seed=7,
    )
    output = router(*inputs, **kwargs)
    assert output.diagnostics["shuffled_multiset_conserved"].all()
    assert output.diagnostics["changed_age_association_count"].sum() > 0
    assert torch.equal(output.return_valid, inputs[1])


def test_exact_union_k1_keeps_range_and_age_from_same_winner():
    inputs = _batch()
    output = RayEventAblationRouter(
        history_reduction="exact_union_k1"
    )(*inputs)
    assert output.range_m.shape == (1, 1, 2, 3)
    assert output.range_m[0, 0, 0, 0].item() == pytest.approx(1.5)
    # Tie at 1.5: frame 2 age 0.28 is earlier than frame 1 age 0.14.
    assert output.return_age_s[0, 0, 0, 0].item() == pytest.approx(0.28)
    assert output.diagnostics["exact_union_winner_frame_index"][0, 0, 0].item() == 2
    assert output.diagnostics["exact_union_collision_cell_count"].item() == 1
    assert output.packet_age_s.item() == pytest.approx(0.05)


def test_raster_latest_prototype_is_distinct_from_nearest_range_oracle():
    range_m = torch.tensor([[[[1.0]], [[3.0]], [[2.0]]]])
    valid = torch.ones_like(range_m, dtype=torch.bool)
    age = torch.tensor([[[[0.30]], [[0.10]], [[0.20]]]])
    packet_age = torch.tensor([[0.25, 0.05, 0.15]])
    frame_valid = torch.ones(1, 3, dtype=torch.bool)
    nearest = RayEventAblationRouter(
        history_reduction="exact_union_k1"
    )(range_m, valid, age, packet_age, frame_valid)
    latest = RayEventAblationRouter(
        history_reduction="raster_latest_event_prototype"
    )(range_m, valid, age, packet_age, frame_valid)
    assert nearest.range_m.item() == pytest.approx(1.0)
    assert nearest.return_age_s.item() == pytest.approx(0.30)
    assert latest.range_m.item() == pytest.approx(3.0)
    assert latest.return_age_s.item() == pytest.approx(0.10)
    assert latest.diagnostics["history_reduction_winner_frame_index"].item() == 1


def test_raster_latest_prototype_excludes_returns_older_than_half_second():
    range_m = torch.tensor([[[[4.0]], [[2.0]]]])
    valid = torch.ones_like(range_m, dtype=torch.bool)
    age = torch.tensor([[[[0.60]], [[0.50]]]])
    packet_age = torch.tensor([[0.55, 0.45]])
    frame_valid = torch.ones(1, 2, dtype=torch.bool)
    latest = RayEventAblationRouter(
        history_reduction="raster_latest_event_prototype"
    )(range_m, valid, age, packet_age, frame_valid)
    assert latest.return_valid.item()
    assert latest.range_m.item() == pytest.approx(2.0)
    assert latest.return_age_s.item() == pytest.approx(0.50)


def test_packet_age_and_age_zero_controls():
    inputs = _batch()
    packet = RayEventAblationRouter(temporal_baseline="packet_age")(*inputs)
    expected = inputs[3][:, :, None, None].expand_as(inputs[2])
    expected = torch.where(inputs[1], expected, torch.zeros_like(expected))
    assert torch.equal(packet.return_age_s, expected)
    zero = RayEventAblationRouter(temporal_baseline="age_zero")(*inputs)
    assert not zero.return_age_s.any()
    assert not zero.packet_age_s.any()
    assert torch.equal(zero.return_valid, inputs[1])


def test_router_rejects_invalid_range_age_contract():
    inputs = list(_batch())
    inputs[2][0, 0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="must be >="):
        RayEventAblationRouter()(*inputs)


def test_shuffle_seed_contract_is_fail_closed():
    with pytest.raises(ValueError, match="requires"):
        RayEventAblationRouter(time_association="shuffled")
    with pytest.raises(ValueError, match="only"):
        RayEventAblationRouter(shuffle_seed=3)
