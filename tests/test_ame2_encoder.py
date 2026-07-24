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
    assert attention_weights.shape == (3, 1, 35)
    assert torch.isfinite(terrain_embedding).all()
    assert torch.isfinite(attention_weights).all()
    torch.testing.assert_close(
        attention_weights.sum(dim=-1),
        torch.ones(3, 1),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_backward_reaches_local_and_qkv_parameters() -> None:
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

    gradient_prefixes = (
        "local_encoder",
        "point_projection",
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


def test_only_latest_frame_is_used_and_height_is_mean_centered() -> None:
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
    torch.testing.assert_close(
        encoder(shifted_latest_frame, proprio_features),
        reference,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

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


def test_phase_one_rejects_multi_head_configuration() -> None:
    with pytest.raises(ValueError, match="exactly one attention head"):
        AME2Encoder(num_heads=4)
