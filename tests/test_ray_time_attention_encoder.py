# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules.ray_time_attention_encoder import (
    CircularAzimuthConv2d,
    RayTimeAttentionEncoder,
)


def _make_ray_history(
    batch_size: int = 2,
    history_length: int = 5,
    spatial_size: tuple[int, int] = (16, 96),
) -> torch.Tensor:
    metric_range = torch.rand(batch_size, history_length, *spatial_size) * 5.5 + 0.2
    hit_mask = (torch.rand_like(metric_range) > 0.25).to(metric_range.dtype)
    metric_range = metric_range * hit_mask
    return torch.stack((metric_range, hit_mask), dim=2)


def test_forward_shapes_attention_normalization_and_fixed_encodings() -> None:
    torch.manual_seed(701)
    encoder = RayTimeAttentionEncoder().eval()
    ray_history = _make_ray_history(batch_size=3)
    proprio_features = torch.randn(3, 64)

    output, attention, token_valid = encoder.forward_with_attention(
        ray_history,
        proprio_features,
    )

    assert encoder.token_spatial_size == (4, 12)
    assert encoder.num_spatial_tokens == 48
    assert encoder.num_tokens == 240
    assert output.shape == (3, 64)
    assert attention.shape == (3, 4, 4, 240)
    assert token_valid.shape == (3, 240)
    assert token_valid.dtype == torch.bool
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention).all()
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones(3, 4, 4),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert "spherical_position_encoding" not in encoder.state_dict()
    assert "time_encoding" not in encoder.state_dict()


def test_backward_reaches_every_history_frame_and_attention_parameters() -> None:
    torch.manual_seed(709)
    encoder = RayTimeAttentionEncoder().train()
    ray_history = _make_ray_history(batch_size=3).requires_grad_()
    proprio_features = torch.randn(3, 64, requires_grad=True)
    target = torch.randn(3, 64)

    output = encoder(ray_history, proprio_features)
    F.mse_loss(output, target).backward()

    assert ray_history.grad is not None
    assert proprio_features.grad is not None
    assert torch.isfinite(ray_history.grad).all()
    assert torch.isfinite(proprio_features.grad).all()
    for history_index in range(5):
        assert float(ray_history.grad[:, history_index, 0].abs().sum()) > 0.0
    assert float(proprio_features.grad.abs().sum()) > 0.0

    parameter_groups = (
        "frame_encoder",
        "spatial_projection",
        "temporal_depthwise",
        "temporal_pointwise",
        "query_projection",
        "query_bias",
        "key_projection",
        "value_projection",
        "attended_projection",
        "global_projection",
        "output_projection",
    )
    named_parameters = dict(encoder.named_parameters())
    for prefix in parameter_groups:
        gradients = [
            parameter.grad
            for name, parameter in named_parameters.items()
            if name.startswith(prefix)
        ]
        assert gradients, f"No parameters found for {prefix}."
        assert all(gradient is not None for gradient in gradients)
        assert all(
            torch.isfinite(gradient).all()
            for gradient in gradients
            if gradient is not None
        )
        assert (
            sum(
                float(gradient.abs().sum())
                for gradient in gradients
                if gradient is not None
            )
            > 0.0
        )


def test_unknown_ranges_are_ignored_and_all_unknown_input_is_finite() -> None:
    torch.manual_seed(719)
    encoder = RayTimeAttentionEncoder().eval()
    ray_history = _make_ray_history()
    proprio_features = torch.randn(2, 64)
    ray_history[0, 0, 0, 0, 0] = 0.0
    ray_history[0, 0, 1, 0, 0] = 0.0

    unknown = ray_history[:, :, 1] == 0.0
    changed_unknown = ray_history.clone()
    changed_unknown[:, :, 0][unknown] = 1.0e6
    changed_unknown[0, 0, 0, 0, 0] = float("nan")
    changed_unknown[0, 0, 1, 0, 0] = 0.0

    with torch.inference_mode():
        reference = encoder(ray_history, proprio_features)
        changed = encoder(changed_unknown, proprio_features)
        all_unknown = encoder(
            torch.zeros_like(ray_history),
            proprio_features,
        )

    torch.testing.assert_close(changed, reference, rtol=0.0, atol=0.0)
    assert torch.isfinite(all_unknown).all()


def test_circular_azimuth_convolution_connects_last_and_first_columns() -> None:
    layer = CircularAzimuthConv2d(
        1,
        1,
        kernel_size=(1, 3),
        stride=(1, 1),
        bias=False,
    )
    with torch.no_grad():
        layer.conv.weight.zero_()
        # Select the left neighbor of each azimuth cell.
        layer.conv.weight[0, 0, 0, 0] = 1.0

    inputs = torch.zeros(1, 1, 1, 7)
    inputs[0, 0, 0, -1] = 2.5
    output = layer(inputs)

    assert output.shape == inputs.shape
    torch.testing.assert_close(output[0, 0, 0, 0], torch.tensor(2.5))
    torch.testing.assert_close(output[0, 0, 0, 1:], torch.zeros(6))


def test_global_only_ablation_disables_attention_but_preserves_contract() -> None:
    torch.manual_seed(727)
    encoder = RayTimeAttentionEncoder(use_query_attention=False).eval()
    ray_history = _make_ray_history()
    proprio_features = torch.randn(2, 64)

    output, attention, token_valid = encoder.forward_with_attention(
        ray_history,
        proprio_features,
    )
    changed_proprio, changed_attention, changed_valid = encoder.forward_with_attention(
        ray_history,
        proprio_features + 100.0,
    )

    assert output.shape == (2, 64)
    assert torch.count_nonzero(attention) == 0
    assert torch.count_nonzero(changed_attention) == 0
    torch.testing.assert_close(changed_proprio, output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(changed_valid, token_valid, rtol=0.0, atol=0.0)


def test_attention_and_global_ablation_have_matched_active_capacity_and_gradients() -> None:
    torch.manual_seed(729)
    ray_history = _make_ray_history(batch_size=3)
    proprio_features = torch.randn(3, 64)
    target = torch.randn(3, 64)

    attention_encoder = RayTimeAttentionEncoder(use_query_attention=True).train()
    global_encoder = RayTimeAttentionEncoder(use_query_attention=False).train()

    attention_loss = F.mse_loss(
        attention_encoder(ray_history, proprio_features),
        target,
    )
    global_loss = F.mse_loss(
        global_encoder(ray_history, proprio_features),
        target,
    )
    attention_loss.backward()
    global_loss.backward()

    attention_prefixes = (
        "query_projection",
        "query_bias",
        "key_projection",
        "value_projection",
        "attended_projection",
    )
    global_prefix = "global_ablation_adapter"

    def _parameters_with_prefix(
        encoder: RayTimeAttentionEncoder,
        prefixes: tuple[str, ...],
    ) -> list[torch.nn.Parameter]:
        return [
            parameter
            for name, parameter in encoder.named_parameters()
            if name.startswith(prefixes)
        ]

    attention_branch = _parameters_with_prefix(
        attention_encoder,
        attention_prefixes,
    )
    inactive_global_branch = _parameters_with_prefix(
        attention_encoder,
        (global_prefix,),
    )
    global_branch = _parameters_with_prefix(global_encoder, (global_prefix,))
    inactive_attention_branch = _parameters_with_prefix(
        global_encoder,
        attention_prefixes,
    )

    for branch in (attention_branch, global_branch):
        assert branch
        assert all(parameter.grad is not None for parameter in branch)
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in branch
            if parameter.grad is not None
        )
        assert (
            sum(
                float(parameter.grad.abs().sum())
                for parameter in branch
                if parameter.grad is not None
            )
            > 0.0
        )
    assert all(parameter.grad is None for parameter in inactive_global_branch)
    assert all(parameter.grad is None for parameter in inactive_attention_branch)

    attention_total = sum(
        parameter.numel() for parameter in attention_encoder.parameters()
    )
    global_total = sum(parameter.numel() for parameter in global_encoder.parameters())
    assert attention_total == global_total

    attention_active = sum(
        parameter.numel()
        for parameter in attention_encoder.parameters()
        if parameter.grad is not None
    )
    global_active = sum(
        parameter.numel()
        for parameter in global_encoder.parameters()
        if parameter.grad is not None
    )
    assert abs(attention_active - global_active) == 320
    assert abs(attention_active - global_active) / attention_active < 0.01


def test_fusion_model_ray_time_aliases_actor_contract_and_export_error() -> None:
    torch.manual_seed(733)
    batch_size = 3
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "mid360_policy": _make_ray_history(batch_size=batch_size).half(),
        },
        batch_size=[batch_size],
    )
    model = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups={"actor": ["policy", "mid360_policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        elevation_encoder_type="ray_time",
        ray_time_set="mid360_policy",
        ray_time_history_length=5,
        ray_time_spatial_size=(16, 96),
        ray_time_use_query_attention=True,
        prop_feature_dim=64,
        prop_hidden_dims=[128],
        vision_feature_dim=64,
    ).eval()

    output = model(obs)
    assert model.elevation_set == "mid360_policy"
    assert model.elevation_history_length == 5
    assert model.vision_spatial_size == (16, 96)
    assert output.shape == (batch_size, 29)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    with pytest.raises(NotImplementedError, match="ray_time"):
        model.as_jit()
    with pytest.raises(NotImplementedError, match="ray_time"):
        model.as_onnx(verbose=False)


@pytest.mark.parametrize(
    "ray_shape",
    (
        (5, 16, 96),
        (4, 2, 16, 96),
        (5, 3, 16, 96),
        (5, 2, 15, 96),
    ),
)
def test_fusion_model_rejects_wrong_ray_time_shapes(
    ray_shape: tuple[int, ...],
) -> None:
    obs = TensorDict(
        {
            "policy": torch.randn(2, 96),
            "mid360_policy": torch.randn(2, *ray_shape),
        },
        batch_size=[2],
    )
    with pytest.raises(ValueError, match="perception branch|shape mismatch"):
        PropMLPElevationFusionModel(
            obs=obs,
            obs_groups={"actor": ["policy", "mid360_policy"]},
            obs_set="actor",
            output_dim=29,
            elevation_encoder_type="ray_time",
            ray_time_set="mid360_policy",
            ray_time_history_length=5,
            ray_time_spatial_size=(16, 96),
        )
