# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np
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


def _make_ray_time_actor(
    *,
    history_length: int,
    use_query_attention: bool,
    fusion_mode: str | None = None,
    batch_size: int = 3,
) -> tuple[PropMLPElevationFusionModel, TensorDict]:
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "mid360_policy": _make_ray_history(
                batch_size=batch_size,
                history_length=history_length,
            ).half(),
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
        ray_time_history_length=history_length,
        ray_time_spatial_size=(16, 96),
        ray_time_use_query_attention=use_query_attention,
        ray_time_fusion_mode=fusion_mode,
        prop_feature_dim=64,
        prop_hidden_dims=[128],
        vision_feature_dim=64,
    )
    calibration_obs = TensorDict(
        {"policy": 2.0 * torch.randn(11, 96) + 1.5},
        batch_size=[11],
    )
    model.update_normalization(calibration_obs)
    with torch.no_grad():
        model.distribution.std_param.fill_(3.0)
    return model.eval(), obs


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
    assert encoder.hit_pool_kernel == (4, 8)
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
    assert "hit_pool_kernel" not in encoder.state_dict()

    flattened_hit = ray_history[:, :, 1].float().flatten(0, 1)
    torch.testing.assert_close(
        F.max_pool2d(
            flattened_hit,
            kernel_size=encoder.hit_pool_kernel,
            stride=encoder.hit_pool_kernel,
        ),
        F.adaptive_max_pool2d(flattened_hit, encoder.token_spatial_size),
        rtol=0.0,
        atol=0.0,
    )


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


def test_explicit_fusion_modes_preserve_legacy_numerics_and_checkpoint_schema() -> None:
    torch.manual_seed(728)
    ray_history = _make_ray_history(batch_size=3)
    proprio_features = torch.randn(3, 64)

    legacy_attention = RayTimeAttentionEncoder(use_query_attention=True).eval()
    explicit_attention = RayTimeAttentionEncoder(
        use_query_attention=False,
        fusion_mode="attention",
    ).eval()
    explicit_attention.load_state_dict(legacy_attention.state_dict(), strict=True)

    legacy_global = RayTimeAttentionEncoder(use_query_attention=False).eval()
    explicit_global = RayTimeAttentionEncoder(
        use_query_attention=True,
        fusion_mode="global",
    ).eval()
    explicit_global.load_state_dict(legacy_global.state_dict(), strict=True)

    with torch.inference_mode():
        legacy_attention_output = legacy_attention.forward_with_attention(
            ray_history,
            proprio_features,
        )
        explicit_attention_output = explicit_attention.forward_with_attention(
            ray_history,
            proprio_features,
        )
        legacy_global_output = legacy_global.forward_with_attention(
            ray_history,
            proprio_features,
        )
        explicit_global_output = explicit_global.forward_with_attention(
            ray_history,
            proprio_features,
        )

    for legacy, explicit in zip(
        legacy_attention_output,
        explicit_attention_output,
        strict=True,
    ):
        torch.testing.assert_close(explicit, legacy, rtol=0.0, atol=0.0)
    for legacy, explicit in zip(
        legacy_global_output,
        explicit_global_output,
        strict=True,
    ):
        torch.testing.assert_close(explicit, legacy, rtol=0.0, atol=0.0)

    query_global = RayTimeAttentionEncoder(fusion_mode="query_global")
    legacy_state = legacy_attention.state_dict()
    query_global_state = query_global.state_dict()
    assert query_global_state.keys() == legacy_state.keys()
    for name in legacy_state:
        assert query_global_state[name].shape == legacy_state[name].shape
    query_global.load_state_dict(legacy_state, strict=True)

    legacy_actor, _ = _make_ray_time_actor(
        history_length=5,
        use_query_attention=True,
    )
    query_global_actor, _ = _make_ray_time_actor(
        history_length=5,
        use_query_attention=True,
        fusion_mode="query_global",
    )
    legacy_actor_state = legacy_actor.state_dict()
    query_global_actor_state = query_global_actor.state_dict()
    assert query_global_actor_state.keys() == legacy_actor_state.keys()
    for name in legacy_actor_state:
        assert query_global_actor_state[name].shape == legacy_actor_state[name].shape
    query_global_actor.load_state_dict(legacy_actor_state, strict=True)
    assert query_global_actor.elevation_encoder.fusion_mode == "query_global"
    assert not query_global_actor.elevation_encoder.use_query_attention


def test_query_global_is_proprio_conditioned_without_spatial_token_selection() -> None:
    torch.manual_seed(729)
    encoder = RayTimeAttentionEncoder(fusion_mode="query_global").eval()
    ray_history = _make_ray_history(batch_size=3)
    proprio_features = torch.randn(3, 64)

    with torch.inference_mode():
        output, gate_weights, token_valid = encoder.forward_with_attention(
            ray_history,
            proprio_features,
        )
        changed_output, changed_gate_weights, changed_valid = (
            encoder.forward_with_attention(
                ray_history,
                proprio_features + 3.0,
            )
        )
        all_unknown, unknown_gate_weights, unknown_valid = (
            encoder.forward_with_attention(
                torch.zeros_like(ray_history),
                proprio_features,
            )
        )

    assert output.shape == (3, 64)
    assert gate_weights.shape == (3, 4, 4, 240)
    assert token_valid.shape == (3, 240)
    assert not torch.equal(changed_output, output)
    assert not torch.equal(changed_gate_weights, gate_weights)
    torch.testing.assert_close(changed_valid, token_valid, rtol=0.0, atol=0.0)

    # QueryGlobal exposes each sigmoid gate uniformly over valid diagnostic
    # tokens; therefore it cannot encode a spatial/time selection.
    safe_valid = torch.where(
        token_valid.any(dim=1, keepdim=True),
        token_valid,
        torch.ones_like(token_valid),
    )
    expected_uniform = safe_valid.to(gate_weights.dtype)
    expected_uniform = expected_uniform / expected_uniform.sum(
        dim=1,
        keepdim=True,
    )
    recovered_gates = gate_weights.sum(dim=-1)
    torch.testing.assert_close(
        gate_weights,
        recovered_gates.unsqueeze(-1)
        * expected_uniform[:, None, None, :],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert torch.all(recovered_gates > 0.0)
    assert torch.all(recovered_gates < 2.0)

    assert torch.isfinite(all_unknown).all()
    assert torch.isfinite(unknown_gate_weights).all()
    assert torch.count_nonzero(unknown_valid) == 0


def test_query_global_zero_logit_gate_has_unit_attention_scale() -> None:
    encoder = RayTimeAttentionEncoder(fusion_mode="query_global").eval()
    with torch.no_grad():
        encoder.query_projection.weight.zero_()
        encoder.query_projection.bias.zero_()
        encoder.query_bias.zero_()
        encoder.key_projection.weight.zero_()
        encoder.key_projection.bias.zero_()

    global_token = torch.randn(7, encoder.token_dim)
    proprio_features = torch.randn(7, encoder.proprio_feature_dim)
    with torch.inference_mode():
        _, query_gates = encoder._query_global(
            global_token,
            proprio_features,
        )

    torch.testing.assert_close(
        query_gates,
        torch.ones_like(query_gates),
        rtol=0.0,
        atol=0.0,
    )


def test_global_encoder_supports_parameter_matched_k1_and_k5_histories() -> None:
    encoders: dict[int, RayTimeAttentionEncoder] = {}
    state_dicts: dict[int, dict[str, torch.Tensor]] = {}

    for history_length in (1, 5):
        torch.manual_seed(728)
        encoder = RayTimeAttentionEncoder(
            history_length=history_length,
            use_query_attention=False,
        ).train()
        ray_history = _make_ray_history(
            batch_size=3,
            history_length=history_length,
        ).requires_grad_()
        proprio_features = torch.randn(3, 64, requires_grad=True)

        output, attention, token_valid = encoder.forward_with_attention(
            ray_history,
            proprio_features,
        )
        output.square().mean().backward()

        assert output.shape == (3, 64)
        assert attention.shape == (
            3,
            4,
            4,
            history_length * encoder.num_spatial_tokens,
        )
        assert token_valid.shape == (
            3,
            history_length * encoder.num_spatial_tokens,
        )
        assert ray_history.grad is not None
        assert torch.isfinite(ray_history.grad).all()
        assert torch.isfinite(output).all()

        encoders[history_length] = encoder
        state_dicts[history_length] = {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }

    assert sum(
        parameter.numel() for parameter in encoders[1].parameters()
    ) == sum(parameter.numel() for parameter in encoders[5].parameters())
    assert state_dicts[1].keys() == state_dicts[5].keys()
    for name in state_dicts[1]:
        assert state_dicts[1][name].shape == state_dicts[5][name].shape

    encoders[1].load_state_dict(state_dicts[5], strict=True)
    encoders[5].load_state_dict(state_dicts[1], strict=True)


def test_attention_and_global_ablation_have_matched_active_capacity_and_gradients() -> None:
    torch.manual_seed(729)
    ray_history = _make_ray_history(batch_size=3)
    proprio_features = torch.randn(3, 64)
    target = torch.randn(3, 64)

    attention_encoder = RayTimeAttentionEncoder(use_query_attention=True).train()
    global_encoder = RayTimeAttentionEncoder(use_query_attention=False).train()
    query_global_encoder = RayTimeAttentionEncoder(
        fusion_mode="query_global",
    ).train()

    attention_loss = F.mse_loss(
        attention_encoder(ray_history, proprio_features),
        target,
    )
    global_loss = F.mse_loss(
        global_encoder(ray_history, proprio_features),
        target,
    )
    query_global_loss = F.mse_loss(
        query_global_encoder(ray_history, proprio_features),
        target,
    )
    attention_loss.backward()
    global_loss.backward()
    query_global_loss.backward()

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
    query_global_branch = _parameters_with_prefix(
        query_global_encoder,
        attention_prefixes,
    )
    inactive_query_global_adapter = _parameters_with_prefix(
        query_global_encoder,
        (global_prefix,),
    )

    for branch in (attention_branch, global_branch, query_global_branch):
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
    assert all(parameter.grad is None for parameter in inactive_query_global_adapter)

    attention_total = sum(
        parameter.numel() for parameter in attention_encoder.parameters()
    )
    global_total = sum(parameter.numel() for parameter in global_encoder.parameters())
    query_global_total = sum(
        parameter.numel() for parameter in query_global_encoder.parameters()
    )
    assert attention_total == global_total
    assert attention_total == query_global_total

    attention_active_names = {
        name
        for name, parameter in attention_encoder.named_parameters()
        if parameter.grad is not None
    }
    query_global_active_names = {
        name
        for name, parameter in query_global_encoder.named_parameters()
        if parameter.grad is not None
    }
    assert query_global_active_names == attention_active_names

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
    query_global_active = sum(
        parameter.numel()
        for parameter in query_global_encoder.parameters()
        if parameter.grad is not None
    )
    assert abs(attention_active - global_active) == 320
    assert abs(attention_active - global_active) / attention_active < 0.01
    assert query_global_active == attention_active


def test_fusion_model_ray_time_aliases_actor_contract() -> None:
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

    assert model.as_jit() is not None
    assert model.as_onnx(verbose=False, input_mode="split") is not None
    assert model.as_onnx(verbose=False, input_mode="single") is not None


@pytest.mark.parametrize("history_length", (1, 5))
@pytest.mark.parametrize(
    ("use_query_attention", "fusion_mode"),
    (
        pytest.param(False, None, id="global"),
        pytest.param(True, None, id="attention"),
        pytest.param(True, "query_global", id="query_global"),
    ),
)
def test_ray_time_jit_matches_eager_for_k1_k5_all_fusion_modes(
    history_length: int,
    use_query_attention: bool,
    fusion_mode: str | None,
) -> None:
    torch.manual_seed(
        739
        + history_length
        + int(use_query_attention)
        + int(fusion_mode == "query_global")
    )
    model, obs = _make_ray_time_actor(
        history_length=history_length,
        use_query_attention=use_query_attention,
        fusion_mode=fusion_mode,
    )
    jit_wrapper = model.as_jit().eval()

    assert int(jit_wrapper.obs_normalizer.count) == 11
    torch.testing.assert_close(
        jit_wrapper.obs_normalizer._mean,
        model.obs_normalizer._mean,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        jit_wrapper.obs_normalizer._std,
        model.obs_normalizer._std,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        model.obs_normalizer._mean,
        torch.zeros_like(model.obs_normalizer._mean),
    )

    scripted_model = torch.jit.script(jit_wrapper)
    with torch.inference_mode():
        eager_output = model(obs, stochastic_output=False)
        scripted_half_output = scripted_model(
            obs["policy"],
            obs["mid360_policy"],
        )
        scripted_float_output = scripted_model(
            obs["policy"],
            obs["mid360_policy"].float(),
        )

    assert obs["mid360_policy"].dtype == torch.float16
    assert eager_output.dtype == torch.float32
    torch.testing.assert_close(scripted_half_output, eager_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scripted_float_output, eager_output, rtol=0.0, atol=0.0)
    scripted_model.reset()


@pytest.mark.parametrize("history_length", (1, 5))
@pytest.mark.parametrize(
    ("use_query_attention", "fusion_mode"),
    (
        pytest.param(False, None, id="global"),
        pytest.param(True, None, id="attention"),
        pytest.param(True, "query_global", id="query_global"),
    ),
)
@pytest.mark.parametrize("input_mode", ("split", "single"))
def test_ray_time_onnx_dynamic_batch_three_matches_eager(
    tmp_path: Any,
    history_length: int,
    use_query_attention: bool,
    fusion_mode: str | None,
    input_mode: str,
) -> None:
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(
        751
        + history_length
        + int(use_query_attention)
        + int(fusion_mode == "query_global")
    )
    model, obs = _make_ray_time_actor(
        history_length=history_length,
        use_query_attention=use_query_attention,
        fusion_mode=fusion_mode,
    )
    onnx_model = model.as_onnx(verbose=False, input_mode=input_mode).eval()
    mode_name = fusion_mode or ("attention" if use_query_attention else "global")
    export_path = tmp_path / (
        f"ray_time_k{history_length}_"
        f"{mode_name}_{input_mode}.onnx"
    )

    dummy_inputs = onnx_model.get_dummy_inputs()
    assert all(dummy.dtype == torch.float32 for dummy in dummy_inputs)
    assert dummy_inputs[0].shape[0] == 1
    if input_mode == "split":
        assert dummy_inputs[0].shape == (1, 96)
        assert dummy_inputs[1].shape == (1, history_length, 2, 16, 96)
        runtime_inputs = (
            obs["policy"],
            obs["mid360_policy"].float(),
        )
    else:
        assert dummy_inputs[0].shape == (
            1,
            96 + history_length * 2 * 16 * 96,
        )
        flat_obs = torch.cat(
            (
                obs["policy"],
                obs["mid360_policy"].float().flatten(start_dim=1),
            ),
            dim=-1,
        )
        runtime_inputs = (flat_obs,)

    with torch.inference_mode():
        eager_output = model(obs, stochastic_output=False)
        wrapper_output = onnx_model(*runtime_inputs)
    torch.testing.assert_close(wrapper_output, eager_output, rtol=0.0, atol=0.0)

    torch.onnx.export(
        onnx_model,
        dummy_inputs,
        export_path,
        export_params=True,
        opset_version=18,
        external_data=False,
        verbose=False,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes=onnx_model.dynamic_axes,
    )

    graph = onnx.load(export_path)
    onnx.checker.check_model(graph)
    assert all(
        graph_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for graph_input in graph.graph.input
    )
    assert all(
        graph_input.type.tensor_type.shape.dim[0].dim_param
        for graph_input in graph.graph.input
    )

    if input_mode == "split":
        feeds = {
            "proprio_obs": obs["policy"].detach().cpu().numpy(),
            "ray_history": obs["mid360_policy"].float().detach().cpu().numpy(),
        }
    else:
        feeds = {"obs": runtime_inputs[0].detach().cpu().numpy()}

    try:
        import onnxruntime
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        onnx_output = ReferenceEvaluator(graph).run(None, feeds)[0]
    else:
        session = onnxruntime.InferenceSession(
            str(export_path),
            providers=["CPUExecutionProvider"],
        )
        onnx_output = session.run(None, feeds)[0]

    assert onnx_output.shape == (3, 29)
    np.testing.assert_allclose(
        onnx_output,
        eager_output.detach().cpu().numpy(),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_ray_time_onnx_single_input_layout_is_strict() -> None:
    model, _ = _make_ray_time_actor(
        history_length=5,
        use_query_attention=True,
    )
    onnx_model = model.as_onnx(verbose=False, input_mode="single").eval()
    expected_size = 96 + 5 * 2 * 16 * 96

    with pytest.raises(ValueError, match=r"layout.*proprio.*flatten"):
        onnx_model(torch.zeros(3, expected_size + 1))
    with pytest.raises(ValueError, match=r"layout.*proprio.*flatten"):
        onnx_model(torch.zeros(3, expected_size - 1))

    with pytest.raises(ValueError, match="Unsupported ONNX input mode"):
        model.as_onnx(verbose=False, input_mode="combined")


def test_fusion_model_accepts_single_packet_ray_time_history() -> None:
    batch_size = 3
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "mid360_policy": _make_ray_history(
                batch_size=batch_size,
                history_length=1,
            ).half(),
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
        ray_time_history_length=1,
        ray_time_spatial_size=(16, 96),
        ray_time_use_query_attention=False,
        prop_feature_dim=64,
        prop_hidden_dims=[128],
        vision_feature_dim=64,
    ).eval()

    output = model(obs)
    assert model.elevation_history_length == 1
    assert model.elevation_encoder.history_length == 1
    assert output.shape == (batch_size, 29)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


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
