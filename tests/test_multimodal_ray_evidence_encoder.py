# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import MultimodalRayEvidenceEncoder
from rsl_rl.modules.multimodal_ray_evidence_encoder import (
    DepthRayEvidenceStem,
    LidarRayEvidenceStem,
    build_depth_raster_coordinates,
    build_lidar_raster_coordinates,
)
from rsl_rl.modules.ray_time_attention_encoder import CircularAzimuthConv2d


PROPRIO_DIM = 11
OUTPUT_DIM = 16
NUM_QUERIES = 3
NUM_HEADS = 4


def _encoder(mode: str = "reliability") -> MultimodalRayEvidenceEncoder:
    return MultimodalRayEvidenceEncoder(
        proprio_dim=PROPRIO_DIM,
        lidar_max_range=6.0,
        depth_max_range=4.0,
        output_dim=OUTPUT_DIM,
        stem_channels=8,
        private_dim=6,
        shared_dim=6,
        token_dim=16,
        num_heads=NUM_HEADS,
        num_queries=NUM_QUERIES,
        age_time_scale=0.4,
        mode=mode,
    )


def _history(
    batch_size: int,
    history_length: int,
    height: int,
    width: int,
) -> torch.Tensor:
    observed = (torch.rand(batch_size, history_length, height, width) > 0.25).float()
    returned = observed * (
        torch.rand(batch_size, history_length, height, width) > 0.30
    ).float()
    metric_range = (0.2 + 4.0 * torch.rand_like(returned)) * returned

    # Guarantee evidence from each batch element and preserve one explicit
    # observed-no-return ray.
    observed[:, -1, 0, 0] = 1.0
    returned[:, -1, 0, 0] = 1.0
    metric_range[:, -1, 0, 0] = 1.0
    observed[:, 0, -1, -1] = 1.0
    returned[:, 0, -1, -1] = 0.0
    metric_range[:, 0, -1, -1] = 0.0
    return torch.stack((metric_range, returned, observed), dim=2)


def _inputs(
    batch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lidar = _history(batch_size, 3, 6, 10)
    depth = _history(batch_size, 2, 8, 8)
    lidar_ages = torch.tensor([0.35, 0.15, 0.0]).expand(batch_size, -1).clone()
    depth_ages = torch.tensor([0.08, 0.0]).expand(batch_size, -1).clone()
    proprio = torch.randn(batch_size, PROPRIO_DIM)
    return lidar, depth, lidar_ages, depth_ages, proprio


def _blackout(history: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(history)


@pytest.mark.parametrize(
    "mode",
    ("reliability", "concat", "lidar_only", "depth_only", "no_reliability"),
)
def test_modes_return_fixed_embedding_and_complete_diagnostics(mode: str) -> None:
    torch.manual_seed(1801)
    encoder = _encoder(mode).eval()
    inputs = _inputs(batch_size=3)

    with torch.inference_mode():
        embedding, diagnostics = encoder.forward_with_diagnostics(*inputs)
        ordinary = encoder(*inputs)

    assert embedding.shape == (3, OUTPUT_DIM)
    torch.testing.assert_close(ordinary, embedding, rtol=0.0, atol=0.0)
    assert torch.isfinite(embedding).all()
    assert diagnostics["query_gates"].shape == (3, NUM_QUERIES, 2)
    torch.testing.assert_close(
        diagnostics["query_gates"].sum(dim=-1),
        torch.ones(3, NUM_QUERIES),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert diagnostics["lidar_attention"].shape[:3] == (
        3,
        NUM_HEADS,
        NUM_QUERIES,
    )
    assert diagnostics["depth_attention"].shape[:3] == (
        3,
        NUM_HEADS,
        NUM_QUERIES,
    )
    assert diagnostics["lidar_frame_coverage"].shape == (3, 3)
    assert diagnostics["depth_frame_coverage"].shape == (3, 2)
    assert diagnostics["lidar_frame_validity"].shape == (3, 3)
    assert diagnostics["depth_frame_validity"].shape == (3, 2)
    assert diagnostics["lidar_global_quality"].shape == (3, 4)
    assert diagnostics["depth_global_quality"].shape == (3, 4)
    assert diagnostics["lidar_query_quality"].shape == (3, NUM_QUERIES, 4)
    assert diagnostics["depth_query_quality"].shape == (3, NUM_QUERIES, 4)
    assert diagnostics["lidar_token_quality"].shape[-1] == 4
    assert diagnostics["depth_token_quality"].shape[-1] == 4
    assert diagnostics["lidar_available"].dtype == torch.bool
    assert diagnostics["depth_available"].dtype == torch.bool
    assert diagnostics["terrain_available"].all()
    assert not diagnostics["both_modalities_missing"].any()
    assert torch.isfinite(diagnostics["lidar_attention"]).all()
    assert torch.isfinite(diagnostics["depth_attention"]).all()

    if mode == "lidar_only":
        torch.testing.assert_close(
            diagnostics["query_gates"][..., 0],
            torch.ones(3, NUM_QUERIES),
        )
        assert torch.count_nonzero(diagnostics["query_gates"][..., 1]) == 0
    elif mode == "depth_only":
        assert torch.count_nonzero(diagnostics["query_gates"][..., 0]) == 0
        torch.testing.assert_close(
            diagnostics["query_gates"][..., 1],
            torch.ones(3, NUM_QUERIES),
        )
    elif mode == "concat":
        torch.testing.assert_close(
            diagnostics["query_gates"],
            torch.full((3, NUM_QUERIES, 2), 0.5),
        )
    elif mode == "no_reliability":
        torch.testing.assert_close(
            diagnostics["query_gates"],
            torch.full((3, NUM_QUERIES, 2), 0.5),
        )


def test_minimum_declared_spatial_shape_supports_single_batch_and_frame() -> None:
    encoder = _encoder().eval()
    lidar = torch.zeros(1, 1, 3, 2, 2)
    depth = torch.zeros(1, 1, 3, 2, 2)
    lidar[:, :, 2] = 1.0
    depth[:, :, 2] = 1.0
    ages = torch.zeros(1, 1)
    proprio = torch.zeros(1, PROPRIO_DIM)

    with torch.inference_mode():
        output, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            ages,
            ages,
            proprio,
        )

    assert output.shape == (1, OUTPUT_DIM)
    assert torch.isfinite(output).all()
    assert torch.isfinite(diagnostics["lidar_attention"]).all()
    assert torch.isfinite(diagnostics["depth_attention"]).all()


def test_available_modality_attends_all_states_and_no_return_is_observed() -> None:
    torch.manual_seed(1807)
    encoder = _encoder().eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs(batch_size=1)
    lidar.zero_()
    depth.zero_()

    # One return and one emitted ray without a return in distant pooling cells.
    lidar[0, 2, :, 0, 0] = torch.tensor([1.2, 1.0, 1.0])
    lidar[0, 0, :, -1, -1] = torch.tensor([0.0, 0.0, 1.0])
    depth[0, 1, :, 0, 0] = torch.tensor([0.8, 1.0, 1.0])
    depth[0, 0, :, -1, -1] = torch.tensor([0.0, 0.0, 1.0])

    with torch.inference_mode():
        _, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            lidar_ages,
            depth_ages,
            proprio,
        )

    for sensor in ("lidar", "depth"):
        observed = diagnostics[f"{sensor}_token_observed"]
        returned = diagnostics[f"{sensor}_token_return_valid"]
        attention = diagnostics[f"{sensor}_attention"]
        unobserved = ~observed[:, None, None, :]
        assert bool(unobserved.any())
        # Missing rays are an explicit spatial state, not a numerical mask.
        assert bool((attention.masked_select(unobserved) > 0.0).all())
        assert bool((observed & ~returned).any())
        assert not bool((returned & ~observed).any())
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(1, NUM_HEADS, NUM_QUERIES),
            rtol=0.0,
            atol=1.0e-6,
        )


def test_query_quality_is_attention_weighted_and_query_local() -> None:
    torch.manual_seed(1809)
    encoder = _encoder().eval()
    inputs = _inputs(batch_size=2)

    with torch.inference_mode():
        _, diagnostics = encoder.forward_with_diagnostics(*inputs)

    for sensor in ("lidar", "depth"):
        mean_attention = diagnostics[f"{sensor}_attention"].mean(dim=1)
        expected = torch.einsum(
            "bqn,bnf->bqf",
            mean_attention,
            diagnostics[f"{sensor}_token_quality"],
        )
        torch.testing.assert_close(
            diagnostics[f"{sensor}_query_quality"],
            expected,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        assert diagnostics[f"{sensor}_query_quality"].shape == (
            2,
            NUM_QUERIES,
            4,
        )


def test_lidar_raster_coordinates_are_periodic_at_azimuth_seam() -> None:
    coordinates = build_lidar_raster_coordinates(
        3,
        8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert coordinates.shape == (3, 8, 3)
    torch.testing.assert_close(
        coordinates[..., :2].square().sum(dim=-1),
        torch.ones(3, 8, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    seam_step = torch.linalg.vector_norm(
        coordinates[1, 0, :2] - coordinates[1, -1, :2]
    )
    ordinary_step = torch.linalg.vector_norm(
        coordinates[1, 1, :2] - coordinates[1, 0, :2]
    )
    torch.testing.assert_close(seam_step, ordinary_step, rtol=1.0e-12, atol=1.0e-12)

    depth_coordinates = build_depth_raster_coordinates(
        3,
        8,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert depth_coordinates.shape == (3, 8, 2)
    assert bool((depth_coordinates[:, 1:, 0] > depth_coordinates[:, :-1, 0]).all())


def test_lidar_seam_ray_marks_both_receptive_field_sides_observed() -> None:
    encoder = _encoder("lidar_only").eval()
    lidar = torch.zeros(1, 1, 3, 8, 16)
    lidar[0, 0, :, 4, -1] = torch.tensor([2.0, 1.0, 1.0])
    depth = torch.zeros(1, 1, 3, 8, 8)
    ages = torch.zeros(1, 1)
    proprio = torch.zeros(1, PROPRIO_DIM)

    with torch.inference_mode():
        _, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            ages,
            ages,
            proprio,
        )

    observed = diagnostics["lidar_token_observed_fraction"].reshape(1, 4, 4)
    returned = diagnostics["lidar_token_return_fraction"].reshape(1, 4, 4)
    assert bool((observed[..., 0] > 0.0).any())
    assert bool((observed[..., -1] > 0.0).any())
    assert bool((returned[..., 0] > 0.0).any())
    assert bool((returned[..., -1] > 0.0).any())


def test_depth_boundary_ray_marks_every_overlapping_token_quality() -> None:
    encoder = _encoder("depth_only").eval()
    lidar = torch.zeros(1, 1, 3, 8, 8)
    depth = torch.zeros(1, 1, 3, 8, 16)
    depth[0, 0, :, 3, 3] = torch.tensor([2.0, 1.0, 1.0])
    ages = torch.zeros(1, 1)
    proprio = torch.zeros(1, PROPRIO_DIM)

    with torch.inference_mode():
        _, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            ages,
            ages,
            proprio,
        )

    observed = diagnostics["depth_token_observed_fraction"].reshape(1, 4, 4)
    returned = diagnostics["depth_token_return_fraction"].reshape(1, 4, 4)
    assert int(torch.count_nonzero(observed)) > 1
    assert torch.equal(observed > 0.0, returned > 0.0)
    assert bool((observed[..., 0] > 0.0).any())
    assert bool((observed[..., 1] > 0.0).any())


@pytest.mark.parametrize(
    ("mode", "sensor"),
    (("lidar_only", "lidar"), ("depth_only", "depth")),
)
def test_moving_a_return_in_sensor_raster_changes_embedding(
    mode: str,
    sensor: str,
) -> None:
    torch.manual_seed(1810)
    encoder = _encoder(mode).eval()
    lidar = torch.zeros(1, 1, 3, 12, 16)
    depth = torch.zeros(1, 1, 3, 16, 16)
    moved_lidar = lidar.clone()
    moved_depth = depth.clone()
    if sensor == "lidar":
        lidar[0, 0, :, 4, 0] = torch.tensor([2.0, 1.0, 1.0])
        moved_lidar[0, 0, :, 4, 4] = torch.tensor([2.0, 1.0, 1.0])
    else:
        depth[0, 0, :, 4, 4] = torch.tensor([2.0, 1.0, 1.0])
        moved_depth[0, 0, :, 8, 8] = torch.tensor([2.0, 1.0, 1.0])
    lidar_ages = torch.zeros(1, 1)
    depth_ages = torch.zeros(1, 1)
    proprio = torch.zeros(1, PROPRIO_DIM)

    with torch.inference_mode():
        first = encoder(lidar, depth, lidar_ages, depth_ages, proprio)
        moved = encoder(
            moved_lidar,
            moved_depth,
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert float((first - moved).abs().max()) > 1.0e-6


def test_equal_coverage_unobserved_hole_patterns_remain_distinguishable() -> None:
    torch.manual_seed(1810)
    encoder = _encoder("lidar_only").eval()
    first = torch.zeros(1, 1, 3, 8, 16)
    second = torch.zeros_like(first)
    first[:, :, 2] = 1.0
    second[:, :, 2] = 1.0
    first[:, :, 2, 2:4, 0:4] = 0.0
    second[:, :, 2, 2:4, 4:8] = 0.0
    depth = torch.zeros(1, 1, 3, 8, 8)
    ages = torch.zeros(1, 1)
    proprio = torch.zeros(1, PROPRIO_DIM)

    with torch.inference_mode():
        first_embedding, first_diagnostics = encoder.forward_with_diagnostics(
            first,
            depth,
            ages,
            ages,
            proprio,
        )
        second_embedding, second_diagnostics = encoder.forward_with_diagnostics(
            second,
            depth,
            ages,
            ages,
            proprio,
        )

    torch.testing.assert_close(
        first_diagnostics["lidar_frame_coverage"],
        second_diagnostics["lidar_frame_coverage"],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        first_diagnostics["lidar_token_observed_fraction"],
        second_diagnostics["lidar_token_observed_fraction"],
    )
    assert float((first_embedding - second_embedding).abs().max()) > 1.0e-6


def test_range_normalization_has_explicit_independent_scale_and_clamps() -> None:
    encoder = _encoder()
    valid = torch.ones(4)
    lidar_ranges = torch.tensor([0.0, 3.0, 6.0, 12.0])
    depth_ranges = torch.tensor([0.0, 2.0, 4.0, 8.0])

    lidar_normalized = encoder._normalize_metric_range(
        lidar_ranges,
        valid,
        minimum_range=encoder.lidar_min_range,
        maximum_range=encoder.lidar_max_range,
    )
    depth_normalized = encoder._normalize_metric_range(
        depth_ranges,
        valid,
        minimum_range=encoder.depth_min_range,
        maximum_range=encoder.depth_max_range,
    )
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0])
    torch.testing.assert_close(lidar_normalized, expected)
    torch.testing.assert_close(depth_normalized, expected)

    returned = torch.tensor([0.0, 1.0, 1.0])
    with_minimum = encoder._normalize_metric_range(
        torch.tensor([0.0, 0.5, 10.0]),
        returned,
        minimum_range=1.0,
        maximum_range=5.0,
    )
    torch.testing.assert_close(with_minimum, torch.tensor([0.0, 0.0, 1.0]))


def test_sensor_specific_range_scales_reach_the_correct_stem_end_to_end() -> None:
    encoder = _encoder().eval()
    lidar = torch.zeros(1, 1, 3, 4, 4)
    depth = torch.zeros(1, 1, 3, 4, 4)
    lidar[:, :, 2] = 1.0
    depth[:, :, 2] = 1.0
    lidar[0, 0, :, 0, 0] = torch.tensor([3.0, 1.0, 1.0])
    lidar[0, 0, :, 0, 1] = torch.tensor([12.0, 1.0, 1.0])
    depth[0, 0, :, 0, 0] = torch.tensor([2.0, 1.0, 1.0])
    depth[0, 0, :, 0, 1] = torch.tensor([8.0, 1.0, 1.0])
    ages = torch.zeros(1, 1)
    proprio = torch.zeros(1, PROPRIO_DIM)
    prepared: dict[str, torch.Tensor] = {}

    def capture(sensor: str) -> Callable[[nn.Module, tuple[torch.Tensor]], None]:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor]) -> None:
            prepared[sensor] = inputs[0].detach().clone()

        return hook

    handles = (
        encoder.lidar_stem.register_forward_pre_hook(capture("lidar")),
        encoder.depth_stem.register_forward_pre_hook(capture("depth")),
    )
    try:
        with torch.inference_mode():
            encoder(lidar, depth, ages, ages, proprio)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(
        prepared["lidar"][0, 0, 0, :2],
        torch.tensor([0.5, 1.0]),
    )
    torch.testing.assert_close(
        prepared["depth"][0, 0, 0, :2],
        torch.tensor([0.5, 1.0]),
    )


def test_single_modality_blackouts_force_exact_gates_without_nan() -> None:
    torch.manual_seed(1811)
    encoder = _encoder().eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()

    with torch.inference_mode():
        lidar_missing_output, lidar_missing = encoder.forward_with_diagnostics(
            _blackout(lidar),
            depth,
            lidar_ages,
            depth_ages,
            proprio,
        )
        depth_missing_output, depth_missing = encoder.forward_with_diagnostics(
            lidar,
            _blackout(depth),
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert torch.isfinite(lidar_missing_output).all()
    assert torch.isfinite(depth_missing_output).all()
    assert torch.count_nonzero(lidar_missing["query_gates"][..., 0]) == 0
    torch.testing.assert_close(
        lidar_missing["query_gates"][..., 1],
        torch.ones(2, NUM_QUERIES),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(lidar_missing["lidar_attention"]) == 0
    torch.testing.assert_close(
        depth_missing["query_gates"][..., 0],
        torch.ones(2, NUM_QUERIES),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(depth_missing["query_gates"][..., 1]) == 0
    assert torch.count_nonzero(depth_missing["depth_attention"]) == 0


@pytest.mark.parametrize(
    ("mode", "missing_sensor", "expected_gates"),
    (
        ("reliability", "lidar", (0.0, 1.0)),
        ("reliability", "depth", (1.0, 0.0)),
        ("concat", "lidar", (0.0, 1.0)),
        ("concat", "depth", (1.0, 0.0)),
        ("no_reliability", "lidar", (0.0, 1.0)),
        ("no_reliability", "depth", (1.0, 0.0)),
        ("lidar_only", "depth", (1.0, 0.0)),
        ("depth_only", "lidar", (0.0, 1.0)),
    ),
)
def test_every_mode_uses_exact_single_modality_gates(
    mode: str,
    missing_sensor: str,
    expected_gates: tuple[float, float],
) -> None:
    encoder = _encoder(mode).eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    if missing_sensor == "lidar":
        lidar = _blackout(lidar)
    else:
        depth = _blackout(depth)

    with torch.inference_mode():
        output, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert torch.isfinite(output).all()
    expected = torch.tensor(expected_gates).expand(2, NUM_QUERIES, -1)
    torch.testing.assert_close(
        diagnostics["query_gates"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_observed_no_return_is_not_treated_as_modality_blackout() -> None:
    encoder = _encoder().eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    lidar.zero_()
    depth.zero_()
    lidar[:, :, 2] = 1.0

    with torch.inference_mode():
        output, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert torch.isfinite(output).all()
    assert diagnostics["lidar_available"].all()
    assert not diagnostics["depth_available"].any()
    torch.testing.assert_close(
        diagnostics["query_gates"][..., 0],
        torch.ones(2, NUM_QUERIES),
        rtol=0.0,
        atol=0.0,
    )
    assert not diagnostics["lidar_token_return_valid"].any()
    assert diagnostics["lidar_token_observed"].all()


@pytest.mark.parametrize(
    "mode",
    ("reliability", "concat", "lidar_only", "depth_only", "no_reliability"),
)
def test_both_modalities_empty_returns_explicit_zero_terrain(mode: str) -> None:
    encoder = _encoder(mode).eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()

    with torch.inference_mode():
        output, diagnostics = encoder.forward_with_diagnostics(
            _blackout(lidar),
            _blackout(depth),
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(diagnostics["query_gates"]) == 0
    assert torch.count_nonzero(diagnostics["lidar_attention"]) == 0
    assert torch.count_nonzero(diagnostics["depth_attention"]) == 0
    assert diagnostics["both_modalities_missing"].all()
    assert not diagnostics["terrain_available"].any()


@pytest.mark.parametrize(
    ("mode", "missing_sensor"),
    (
        ("lidar_only", "lidar"),
        ("depth_only", "depth"),
    ),
)
def test_single_sensor_modes_return_zero_when_selected_sensor_is_missing(
    mode: str,
    missing_sensor: str,
) -> None:
    encoder = _encoder(mode).eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    if missing_sensor == "lidar":
        lidar = _blackout(lidar)
    else:
        depth = _blackout(depth)

    with torch.inference_mode():
        output, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(diagnostics["query_gates"]) == 0
    assert not diagnostics["terrain_available"].any()
    assert not diagnostics["both_modalities_missing"].any()


SemanticMutator = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


def _return_without_observation(
    lidar: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lidar[0, 0, :, 0, 0] = torch.tensor([1.0, 1.0, 0.0])
    return lidar, depth


def _range_without_return(
    lidar: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lidar[0, 0, :, 0, 0] = torch.tensor([1.0, 0.0, 1.0])
    return lidar, depth


def _zero_range_with_return(
    lidar: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth[0, 0, :, 0, 0] = torch.tensor([0.0, 1.0, 1.0])
    return lidar, depth


def _nonbinary_observation(
    lidar: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth[0, 0, 2, 0, 0] = 0.5
    return lidar, depth


def _negative_range(
    lidar: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lidar[0, 0, :, 0, 0] = torch.tensor([-1.0, 1.0, 1.0])
    return lidar, depth


def _nonfinite_range(
    lidar: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth[0, 0, 0, 0, 0] = float("nan")
    return lidar, depth


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (_return_without_observation, "return_valid <= ray_observed"),
        (_range_without_return, "exactly zero"),
        (_zero_range_with_return, "strictly positive"),
        (_nonbinary_observation, "exact 0/1"),
        (_negative_range, "non-negative"),
        (_nonfinite_range, "finite"),
    ),
)
def test_illegal_ray_semantics_are_rejected(
    mutator: SemanticMutator,
    match: str,
) -> None:
    encoder = _encoder()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    lidar, depth = mutator(lidar, depth)

    with pytest.raises(ValueError, match=match):
        encoder(lidar, depth, lidar_ages, depth_ages, proprio)


def test_shape_dtype_device_and_mode_validation() -> None:
    encoder = _encoder()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()

    with pytest.raises(ValueError, match=r"shape \[B, T, 3, H, W\]"):
        encoder(lidar[:, :, :2], depth, lidar_ages, depth_ages, proprio)
    with pytest.raises(ValueError, match="lidar_frame_ages"):
        encoder(lidar, depth, lidar_ages[:, :-1], depth_ages, proprio)
    with pytest.raises(ValueError, match=r"shape \[B, P\]"):
        encoder(lidar, depth, lidar_ages, depth_ages, proprio[:, :-1])
    with pytest.raises(ValueError, match="floating-point"):
        encoder(lidar.long(), depth, lidar_ages, depth_ages, proprio)
    with pytest.raises(ValueError, match="same device"):
        encoder(
            lidar,
            depth.to("meta"),
            lidar_ages,
            depth_ages.to("meta"),
            proprio,
        )
    with pytest.raises(ValueError, match="mode must be one of"):
        encoder(lidar, depth, lidar_ages, depth_ages, proprio, mode="unknown")


def test_inputs_must_match_parameter_device_and_half_inputs_are_supported() -> None:
    encoder = _encoder().eval()
    inputs = _inputs()
    with torch.inference_mode():
        output = encoder(*(tensor.half() for tensor in inputs))
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    meta_encoder = _encoder().to("meta")
    with pytest.raises(ValueError, match="encoder parameters"):
        meta_encoder(*inputs)


@pytest.mark.parametrize("sensor", ("lidar", "depth"))
@pytest.mark.parametrize("bad_age", (-0.01, float("nan"), float("inf")))
def test_invalid_ages_are_rejected(sensor: str, bad_age: float) -> None:
    encoder = _encoder()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    if sensor == "lidar":
        lidar_ages[0, 0] = bad_age
    else:
        depth_ages[0, 0] = bad_age

    with pytest.raises(ValueError, match=f"{sensor}_frame_ages"):
        encoder(lidar, depth, lidar_ages, depth_ages, proprio)


def test_nonfinite_proprio_is_rejected() -> None:
    encoder = _encoder()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    proprio[0, 0] = float("nan")
    with pytest.raises(ValueError, match="proprio must contain only finite"):
        encoder(lidar, depth, lidar_ages, depth_ages, proprio)


@pytest.mark.parametrize("field", ("age", "proprio", "range"))
def test_finite_float64_values_that_overflow_compute_dtype_are_rejected(
    field: str,
) -> None:
    encoder = _encoder()
    lidar, depth, lidar_ages, depth_ages, proprio = (
        tensor.double() for tensor in _inputs()
    )
    if field == "age":
        lidar_ages[0, 0] = 1.0e300
    elif field == "proprio":
        proprio[0, 0] = 1.0e300
    else:
        lidar[0, 0, :, 0, 0] = torch.tensor(
            [1.0e300, 1.0, 1.0],
            dtype=torch.float64,
        )

    with pytest.raises(ValueError, match="not representable as finite"):
        encoder(lidar, depth, lidar_ages, depth_ages, proprio)


def test_maximum_finite_compute_dtype_age_uses_bounded_encoding() -> None:
    encoder = _encoder().eval()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    lidar_ages.fill_(torch.finfo(torch.float32).max)
    depth_ages.fill_(torch.finfo(torch.float32).max)

    with torch.inference_mode():
        output, diagnostics = encoder.forward_with_diagnostics(
            lidar,
            depth,
            lidar_ages,
            depth_ages,
            proprio,
        )

    assert torch.isfinite(output).all()
    assert torch.isfinite(diagnostics["lidar_query_quality"]).all()
    assert torch.isfinite(diagnostics["depth_query_quality"]).all()
    torch.testing.assert_close(
        diagnostics["lidar_token_quality"][..., 2],
        torch.zeros_like(diagnostics["lidar_token_quality"][..., 2]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["lidar_token_quality"][..., 3],
        torch.ones_like(diagnostics["lidar_token_quality"][..., 3]),
        rtol=0.0,
        atol=0.0,
    )


def test_scalar_contract_must_be_representable_in_compute_dtype() -> None:
    inputs = _inputs()
    huge_range = MultimodalRayEvidenceEncoder(
        proprio_dim=PROPRIO_DIM,
        lidar_max_range=1.0e300,
        depth_max_range=4.0,
        output_dim=OUTPUT_DIM,
    )
    with pytest.raises(ValueError, match="Range limits are not representable"):
        huge_range(*inputs)

    tiny_age_scale = MultimodalRayEvidenceEncoder(
        proprio_dim=PROPRIO_DIM,
        lidar_max_range=6.0,
        depth_max_range=4.0,
        output_dim=OUTPUT_DIM,
        age_time_scale=1.0e-300,
    )
    with pytest.raises(ValueError, match="age_time_scale is not representable"):
        tiny_age_scale(*inputs)


def test_reliability_gate_has_gradient_dependency_on_both_sensor_ages() -> None:
    torch.manual_seed(1823)
    encoder = _encoder().train()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs(batch_size=3)
    lidar_ages.requires_grad_()
    depth_ages.requires_grad_()

    _, diagnostics = encoder.forward_with_diagnostics(
        lidar,
        depth,
        lidar_ages,
        depth_ages,
        proprio,
    )
    lidar_gate_objective = diagnostics["query_gates"][..., 0].square().sum()
    lidar_gate_objective.backward()

    assert lidar_ages.grad is not None
    assert depth_ages.grad is not None
    assert torch.isfinite(lidar_ages.grad).all()
    assert torch.isfinite(depth_ages.grad).all()
    assert float(lidar_ages.grad.abs().sum()) > 0.0
    assert float(depth_ages.grad.abs().sum()) > 0.0
    assert encoder.lidar_quality_projection.weight.grad is not None
    assert encoder.depth_quality_projection.weight.grad is not None
    assert float(encoder.lidar_quality_projection.weight.grad.abs().sum()) > 0.0
    assert float(encoder.depth_quality_projection.weight.grad.abs().sum()) > 0.0


def test_lidar_stem_is_periodic_but_depth_stem_is_not() -> None:
    lidar_stem = LidarRayEvidenceStem(out_channels=2)
    depth_stem = DepthRayEvidenceStem(out_channels=2)
    assert isinstance(lidar_stem.first_conv, CircularAzimuthConv2d)
    assert isinstance(depth_stem.first_conv, nn.Conv2d)
    assert depth_stem.first_conv.padding_mode == "zeros"

    with torch.no_grad():
        lidar_stem.first_conv.conv.weight.zero_()
        lidar_stem.first_conv.conv.weight[0, 0, 1, 0] = 1.0
        depth_stem.first_conv.weight.zero_()
        depth_stem.first_conv.weight[0, 0, 1, 0] = 1.0

    inputs = torch.zeros(1, 3, 3, 6)
    inputs[0, 0, 0, -1] = 2.5
    lidar_output = lidar_stem.first_conv(inputs)
    depth_output = depth_stem.first_conv(inputs)

    torch.testing.assert_close(lidar_output[0, 0, 0, 0], torch.tensor(2.5))
    torch.testing.assert_close(depth_output[0, 0, 0, 0], torch.tensor(0.0))


@pytest.mark.parametrize(
    "mode",
    ("reliability", "concat", "lidar_only", "depth_only", "no_reliability"),
)
def test_eval_forward_and_diagnostics_are_deterministic(mode: str) -> None:
    torch.manual_seed(1829)
    encoder = _encoder(mode).eval()
    inputs = _inputs(batch_size=3)

    with torch.inference_mode():
        first_output, first = encoder.forward_with_diagnostics(*inputs)
        second_output, second = encoder.forward_with_diagnostics(*inputs)

    torch.testing.assert_close(second_output, first_output, rtol=0.0, atol=0.0)
    for key in (
        "query_gates",
        "lidar_attention",
        "depth_attention",
        "lidar_global_quality",
        "depth_global_quality",
        "lidar_query_quality",
        "depth_query_quality",
    ):
        torch.testing.assert_close(second[key], first[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "mode",
    ("reliability", "concat", "lidar_only", "depth_only", "no_reliability"),
)
def test_all_modes_support_finite_backward(mode: str) -> None:
    torch.manual_seed(1830)
    encoder = _encoder(mode).train()
    inputs = _inputs(batch_size=2)
    target = torch.randn(2, OUTPUT_DIM)

    output = encoder(*inputs)
    loss = F.mse_loss(output, target)
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    output_gradients = [
        parameter.grad
        for name, parameter in encoder.named_parameters()
        if name.startswith("output_projection")
    ]
    assert output_gradients
    assert all(gradient is not None for gradient in output_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in output_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in output_gradients) > 0.0


def test_reliability_backward_reaches_both_stems_queries_gate_and_inputs() -> None:
    torch.manual_seed(1831)
    encoder = _encoder().train()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs(batch_size=3)
    lidar.requires_grad_()
    depth.requires_grad_()
    lidar_ages.requires_grad_()
    depth_ages.requires_grad_()
    proprio.requires_grad_()
    target = torch.randn(3, OUTPUT_DIM)

    output = encoder(lidar, depth, lidar_ages, depth_ages, proprio)
    F.mse_loss(output, target).backward()

    for name, tensor in (
        ("lidar", lidar),
        ("depth", depth),
        ("lidar_ages", lidar_ages),
        ("depth_ages", depth_ages),
        ("proprio", proprio),
    ):
        assert tensor.grad is not None, f"{name} has no gradient."
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0

    parameter_prefixes = (
        "lidar_stem",
        "depth_stem",
        "lidar_private_projection",
        "depth_private_projection",
        "shared_projection",
        "lidar_token_projection",
        "depth_token_projection",
        "lidar_position_projection",
        "depth_position_projection",
        "lidar_state_projection",
        "depth_state_projection",
        "lidar_age_projection",
        "depth_age_projection",
        "state_query_projection",
        "lidar_attention",
        "depth_attention",
        "gate_query_projection",
        "lidar_gate_evidence_projection",
        "depth_gate_evidence_projection",
        "lidar_quality_projection",
        "depth_quality_projection",
        "lidar_gate_score",
        "depth_gate_score",
        "output_projection",
    )
    named_parameters = dict(encoder.named_parameters())
    for prefix in parameter_prefixes:
        gradients = [
            parameter.grad
            for name, parameter in named_parameters.items()
            if name.startswith(prefix)
        ]
        assert gradients, f"No parameters found for {prefix}."
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


@pytest.mark.parametrize(
    ("mode", "active_sensor"),
    (("lidar_only", "lidar"), ("depth_only", "depth")),
)
def test_single_sensor_backward_reaches_the_applicable_branch(
    mode: str,
    active_sensor: str,
) -> None:
    torch.manual_seed(1837)
    encoder = _encoder(mode).train()
    lidar, depth, lidar_ages, depth_ages, proprio = _inputs()
    lidar.requires_grad_()
    depth.requires_grad_()

    encoder(lidar, depth, lidar_ages, depth_ages, proprio).square().mean().backward()

    active = lidar if active_sensor == "lidar" else depth
    inactive = depth if active_sensor == "lidar" else lidar
    assert active.grad is not None
    assert float(active.grad.abs().sum()) > 0.0
    assert inactive.grad is None or float(inactive.grad.abs().sum()) == 0.0


def test_construction_contract_and_independent_sensor_parameters() -> None:
    encoder = _encoder()
    assert (
        encoder.lidar_stem.first_conv.conv.weight.data_ptr()
        != encoder.depth_stem.first_conv.weight.data_ptr()
    )
    assert (
        encoder.lidar_private_projection.weight.data_ptr()
        != encoder.depth_private_projection.weight.data_ptr()
    )
    assert not any("align" in name.lower() for name, _ in encoder.named_parameters())

    with pytest.raises(TypeError, match="lidar_max_range"):
        MultimodalRayEvidenceEncoder(proprio_dim=4)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=0,
            lidar_max_range=6.0,
            depth_max_range=4.0,
        )
    with pytest.raises(ValueError, match="stem_channels must be at least two"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            stem_channels=1,
        )
    with pytest.raises(ValueError, match="token_dim must be at least two"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            token_dim=1,
            num_heads=1,
        )
    with pytest.raises(ValueError, match="output_dim must be at least two"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            output_dim=1,
        )
    with pytest.raises(ValueError, match="dimensions must be integers"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            output_dim=7.9,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="dimensions must be integers"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            num_queries=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="divisible"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            token_dim=10,
            num_heads=4,
        )
    with pytest.raises(ValueError, match="age_time_scale"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            age_time_scale=0.0,
        )
    with pytest.raises(ValueError, match="mode must be one of"):
        MultimodalRayEvidenceEncoder(
            proprio_dim=4,
            lidar_max_range=6.0,
            depth_max_range=4.0,
            mode="bad",
        )
    for kwargs in (
        {"lidar_max_range": 0.0, "depth_max_range": 4.0},
        {"lidar_max_range": float("inf"), "depth_max_range": 4.0},
        {
            "lidar_min_range": 6.0,
            "lidar_max_range": 6.0,
            "depth_max_range": 4.0,
        },
        {"lidar_max_range": 6.0, "depth_max_range": float("nan")},
    ):
        with pytest.raises(ValueError, match="range limits"):
            MultimodalRayEvidenceEncoder(proprio_dim=4, **kwargs)
