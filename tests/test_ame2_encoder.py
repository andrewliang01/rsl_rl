# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import torch

from rsl_rl.modules.ame2_encoder import AME2Encoder


def test_forward_shape_attention_sum_and_finiteness() -> None:
    torch.manual_seed(7)
    encoder = AME2Encoder().eval()
    height_history = torch.randn(3, 5, 28, 20)
    proprio_features = torch.randn(3, 64)

    terrain_embedding, attention_weights = encoder.forward_with_attention(height_history, proprio_features)

    assert terrain_embedding.shape == (3, 64)
    assert attention_weights.shape == (3, 4, 1, 35)
    assert torch.isfinite(terrain_embedding).all()
    assert torch.isfinite(attention_weights).all()
    torch.testing.assert_close(
        attention_weights.sum(dim=-1),
        torch.ones(3, 4, 1),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_point_fusion_global_pool_and_intermediate_shapes() -> None:
    torch.manual_seed(9)
    encoder = AME2Encoder().eval()
    height_history = torch.randn(3, 5, 28, 20)
    proprio_features = torch.randn(3, 64)

    (
        terrain_embedding,
        point_local_features,
        global_feature,
        query,
        keys,
        values,
        weighted_local_feature,
        attention_weights,
    ) = encoder.forward_with_intermediates(height_history, proprio_features)

    assert terrain_embedding.shape == (3, 64)
    assert point_local_features.shape == (3, 35, 64)
    assert global_feature.shape == (3, 32)
    assert query.shape == (3, 4, 1, 8)
    assert keys.shape == (3, 4, 35, 8)
    assert values.shape == (3, 4, 35, 8)
    assert weighted_local_feature.shape == (3, 32)
    assert attention_weights.shape == (3, 4, 1, 35)

    expected_global = encoder.global_encoder(point_local_features).max(dim=1).values
    torch.testing.assert_close(global_feature, expected_global, rtol=0.0, atol=0.0)


def test_attention_softmax_is_independent_across_batch_samples() -> None:
    torch.manual_seed(10)
    encoder = AME2Encoder().eval()
    height_history = torch.randn(3, 5, 28, 20)
    proprio_features = torch.randn(3, 64)

    batched_output, batched_attention = encoder.forward_with_attention(height_history, proprio_features)
    for batch_index in range(3):
        single_output, single_attention = encoder.forward_with_attention(
            height_history[batch_index : batch_index + 1],
            proprio_features[batch_index : batch_index + 1],
        )
        torch.testing.assert_close(
            batched_output[batch_index : batch_index + 1],
            single_output,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            batched_attention[batch_index : batch_index + 1],
            single_attention,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_xyz_coordinates_axis_order_endpoints_and_upward_z_sign() -> None:
    encoder = AME2Encoder().eval()
    height_history = torch.zeros(2, 5, 28, 20)
    height_history[0, -1, 0, 0] = 0.25
    height_history[1, -1, 27, 19] = -0.4

    elevation_xyz = encoder.build_elevation_xyz(height_history)

    assert elevation_xyz.shape == (2, 28, 20, 3)
    torch.testing.assert_close(elevation_xyz[0, 0, 0], torch.tensor([-0.675, -0.475, -0.25]))
    torch.testing.assert_close(elevation_xyz[1, 27, 19], torch.tensor([0.675, 0.475, 0.4]))

    # Axis 1 is x: x varies while y remains fixed.
    torch.testing.assert_close(elevation_xyz[0, :, 0, 0], torch.linspace(-0.675, 0.675, 28))
    torch.testing.assert_close(elevation_xyz[0, :, 0, 1], torch.full((28,), -0.475))
    # Axis 2 is y: y varies while x remains fixed.
    torch.testing.assert_close(elevation_xyz[0, 0, :, 0], torch.full((20,), -0.675))
    torch.testing.assert_close(elevation_xyz[0, 0, :, 1], torch.linspace(-0.475, 0.475, 20))


def test_local_and_position_feature_shapes() -> None:
    encoder = AME2Encoder().eval()
    height_history = torch.randn(3, 5, 28, 20)
    elevation_xyz = encoder.build_elevation_xyz(height_history)

    pooled_xyz = encoder.pool_elevation_xyz(elevation_xyz)
    local_features, position_features = encoder.extract_local_position_features(elevation_xyz)

    assert pooled_xyz.shape == (3, 35, 3)
    assert local_features.shape == (3, 35, 16)
    assert position_features.shape == (3, 35, 16)
    torch.testing.assert_close(pooled_xyz[0, 0, :2], torch.tensor([-0.6, -0.4]))
    torch.testing.assert_close(pooled_xyz[0, 4, :2], torch.tensor([-0.6, 0.4]))
    torch.testing.assert_close(pooled_xyz[0, 30, :2], torch.tensor([0.6, -0.4]))
    torch.testing.assert_close(pooled_xyz[0, 34, :2], torch.tensor([0.6, 0.4]))
    assert torch.isfinite(local_features).all()
    assert torch.isfinite(position_features).all()
    assert "xy_grid" not in encoder.state_dict()


def test_backward_reaches_local_position_and_qkv_parameters() -> None:
    torch.manual_seed(11)
    encoder = AME2Encoder().train()
    height_history = torch.randn(4, 5, 28, 20, requires_grad=True)
    proprio_features = torch.randn(4, 64, requires_grad=True)
    target = torch.randn(4, 64)

    terrain_embedding = encoder(height_history, proprio_features)
    torch.nn.functional.mse_loss(terrain_embedding, target).backward()

    assert height_history.grad is not None
    assert proprio_features.grad is not None
    assert torch.isfinite(height_history.grad).all()
    assert torch.isfinite(proprio_features.grad).all()
    torch.testing.assert_close(
        height_history.grad[:, :-1],
        torch.zeros_like(height_history.grad[:, :-1]),
        rtol=0.0,
        atol=0.0,
    )
    assert float(height_history.grad[:, -1].abs().sum()) > 0.0

    gradient_prefixes = (
        "local_encoder",
        "positional_encoder",
        "point_fusion",
        "global_encoder",
        "query_projection",
        "key_projection",
        "value_projection",
    )
    named_parameters = dict(encoder.named_parameters())
    for prefix in gradient_prefixes:
        gradients = [
            parameter.grad for name, parameter in named_parameters.items() if name.startswith(prefix)
        ]
        assert gradients, f"No parameters found for {prefix}."
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_only_latest_frame_is_used_and_only_local_height_is_mean_centered() -> None:
    torch.manual_seed(17)
    encoder = AME2Encoder().eval()
    height_history = torch.randn(2, 5, 28, 20)
    proprio_features = torch.randn(2, 64)

    reference = encoder(height_history, proprio_features)

    changed_old_frames = height_history.clone()
    changed_old_frames[:, :-1] = torch.randn_like(changed_old_frames[:, :-1]) * 100.0
    torch.testing.assert_close(
        encoder(changed_old_frames, proprio_features),
        reference,
        rtol=0.0,
        atol=0.0,
    )

    shifted_latest_frame = height_history.clone()
    shifted_latest_frame[:, -1] += 2.0
    reference_local, reference_position = encoder.extract_local_position_features(
        encoder.build_elevation_xyz(height_history)
    )
    shifted_local, shifted_position = encoder.extract_local_position_features(
        encoder.build_elevation_xyz(shifted_latest_frame)
    )
    torch.testing.assert_close(shifted_local, reference_local, rtol=1.0e-5, atol=1.0e-5)
    assert not torch.allclose(shifted_position, reference_position)
    assert not torch.allclose(encoder(shifted_latest_frame, proprio_features), reference)

    changed_latest_shape = height_history.clone()
    changed_latest_shape[:, -1, 4:12, 5:13] += 1.0
    assert not torch.allclose(encoder(changed_latest_shape, proprio_features), reference)


@pytest.mark.parametrize(
    ("height_shape", "proprio_shape"),
    (
        ((2, 4, 28, 20), (2, 64)),
        ((2, 5, 27, 20), (2, 64)),
        ((2, 5, 28, 20), (2, 63)),
        ((2, 5, 28, 20), (3, 64)),
    ),
)
def test_rejects_invalid_input_shapes(
    height_shape: tuple[int, ...], proprio_shape: tuple[int, ...]
) -> None:
    encoder = AME2Encoder()

    with pytest.raises(ValueError):
        encoder(torch.randn(height_shape), torch.randn(proprio_shape))


def test_rejects_attention_dimension_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        AME2Encoder(attention_dim=30, num_heads=4)


def test_rejects_invalid_map_extent_and_output_contract() -> None:
    with pytest.raises(ValueError, match="map_extent"):
        AME2Encoder(map_extent=(1.35, 0.0))
    with pytest.raises(ValueError, match=r"global_feature_dim \+ attention_dim"):
        AME2Encoder(global_feature_dim=32, attention_dim=16, output_dim=64)


def test_eager_eval_is_deterministic() -> None:
    torch.manual_seed(29)
    encoder = AME2Encoder().eval()
    height_history = torch.randn(3, 5, 28, 20)
    proprio_features = torch.randn(3, 64)

    with torch.inference_mode():
        first = encoder(height_history, proprio_features)
        second = encoder(height_history, proprio_features)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
