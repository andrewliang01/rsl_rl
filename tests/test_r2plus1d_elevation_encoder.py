# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import (
    PropMLPElevationFusionModel,
)
from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder
from rsl_rl.modules.r2plus1d_elevation_encoder import (
    R2Plus1DBlock,
    R2Plus1DElevationEncoder,
)


def _observations(
    *,
    batch_size: int = 3,
    history_length: int = 5,
) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "height_scan_policy": torch.randn(
                batch_size,
                history_length,
                28,
                20,
            ),
        },
        batch_size=[batch_size],
    )


def _model(
    observations: TensorDict,
    *,
    history_length: int = 5,
) -> PropMLPElevationFusionModel:
    return PropMLPElevationFusionModel(
        obs=observations,
        obs_groups={"actor": ["policy", "height_scan_policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=(128, 64),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        elevation_set="height_scan_policy",
        cnn_observation_type="elevationmap",
        vision_spatial_size=(28, 20),
        vision_feature_dim=64,
        elevation_history_length=history_length,
        cnn_hidden_dims=(8, 16),
        cnn_kernel_sizes=(3, 3),
        cnn_strides=(2, 2),
        r2plus1d_hidden_dims=(8, 16),
        r2plus1d_spatial_kernel_sizes=(3, 3),
        r2plus1d_temporal_kernel_sizes=(3, 3),
        r2plus1d_spatial_strides=(2, 2),
        prop_feature_dim=64,
        prop_hidden_dims=(128,),
        use_prop_encoder=True,
        elevation_encoder_type="r2plus1d",
    )


@pytest.mark.parametrize("history_length", [1, 5])
def test_encoder_shape_dtype_and_backward(history_length: int) -> None:
    torch.manual_seed(1000 + history_length)
    encoder = R2Plus1DElevationEncoder(
        history_length=history_length,
        hidden_dims=(8, 16),
        spatial_kernel_sizes=(3, 3),
        temporal_kernel_sizes=(3, 3),
        spatial_strides=(2, 2),
        out_dim=32,
        vision_spatial_size=(28, 20),
    ).train()
    value = torch.randn(
        4,
        history_length,
        28,
        20,
        requires_grad=True,
    )

    output = encoder(value)
    assert output.shape == (4, 32)
    assert output.dtype == value.dtype
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert float(value.grad.abs().sum()) > 0.0
    temporal_parameters = [
        parameter
        for name, parameter in encoder.named_parameters()
        if ".temporal." in name and parameter.requires_grad
    ]
    assert temporal_parameters
    assert all(parameter.grad is not None for parameter in temporal_parameters)
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in temporal_parameters
        if parameter.grad is not None
    ) > 0.0


def test_factorization_uses_spatial_then_temporal_convolutions() -> None:
    block = R2Plus1DBlock(
        4,
        12,
        temporal_kernel_size=3,
        spatial_kernel_size=3,
        spatial_stride=2,
    )
    assert block.spatial.kernel_size == (1, 3, 3)
    assert block.spatial.stride == (1, 2, 2)
    assert block.temporal.kernel_size == (3, 1, 1)
    assert block.temporal.stride == (1, 1, 1)
    assert block.spatial.bias is None
    assert block.temporal.bias is None

    dense_parameter_count = 4 * 12 * 3 * 3 * 3
    factorized_parameter_count = (
        block.spatial.weight.numel() + block.temporal.weight.numel()
    )
    one_intermediate_channel_cost = 4 * 3 * 3 + 12 * 3
    assert factorized_parameter_count <= dense_parameter_count
    assert (
        dense_parameter_count - factorized_parameter_count
        < one_intermediate_channel_cost
    )


def test_default_encoder_is_parameter_matched_to_five_frame_cnn() -> None:
    r2plus1d = R2Plus1DElevationEncoder(
        history_length=5,
        out_dim=64,
        vision_spatial_size=(28, 20),
    )
    cnn5 = Elevation2DCNNEncoder(
        in_channels=5,
        hidden_dims=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        out_dim=64,
        vision_spatial_size=(28, 20),
    )
    r2plus1d_parameters = sum(
        parameter.numel() for parameter in r2plus1d.parameters()
    )
    cnn5_parameters = sum(parameter.numel() for parameter in cnn5.parameters())
    assert r2plus1d_parameters == 73_251
    assert cnn5_parameters == 73_312
    assert abs(r2plus1d_parameters - cnn5_parameters) / cnn5_parameters < 0.001


def test_model_uses_ordered_history_and_preserves_actor_contract() -> None:
    torch.manual_seed(1201)
    observations = _observations()
    model = _model(observations).eval()
    assert isinstance(model.elevation_encoder, R2Plus1DElevationEncoder)
    assert model.elevation_encoder_type == "r2plus1d"
    assert model._cnn_use_single_frame is False
    assert isinstance(model.mlp[0], nn.Linear)
    assert model.mlp[0].in_features == 128

    with torch.inference_mode():
        reference = model(observations)
        reversed_observations = observations.clone()
        reversed_observations["height_scan_policy"] = observations[
            "height_scan_policy"
        ].flip(dims=(1,))
        reversed_output = model(reversed_observations)

    assert reference.shape == (3, 29)
    assert torch.isfinite(reference).all()
    assert not torch.allclose(reference, reversed_output)


def test_model_strict_checkpoint_reload_and_torchscript_match() -> None:
    torch.manual_seed(1301)
    observations = _observations(batch_size=2)
    model = _model(observations).eval()
    reloaded = _model(observations).eval()
    reloaded.load_state_dict(model.state_dict(), strict=True)

    proprio = observations["policy"]
    elevation = observations["height_scan_policy"]
    with torch.inference_mode():
        expected = model(observations)
        actual = reloaded(observations)
        exported = torch.jit.script(model.as_jit())
        scripted = exported(proprio, elevation)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scripted, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("input_mode", ["split", "single"])
def test_model_onnx_dynamic_batch_matches_eager(
    tmp_path: Path,
    input_mode: str,
) -> None:
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(1351)
    observations = _observations(batch_size=3)
    model = _model(observations).eval()
    exported = model.as_onnx(verbose=False, input_mode=input_mode).eval()
    export_path = tmp_path / f"r2plus1d_{input_mode}.onnx"
    dummy_inputs = exported.get_dummy_inputs()
    torch.onnx.export(
        exported,
        dummy_inputs,
        export_path,
        export_params=True,
        opset_version=18,
        external_data=False,
        verbose=False,
        input_names=exported.input_names,
        output_names=exported.output_names,
        dynamic_axes=exported.dynamic_axes,
    )
    graph = onnx.load(export_path)
    onnx.checker.check_model(graph)
    with torch.inference_mode():
        expected = model(observations).cpu().numpy()
    if input_mode == "split":
        feeds = {
            "proprio_obs": observations["policy"].cpu().numpy(),
            "elevation_obs": observations["height_scan_policy"].cpu().numpy(),
        }
    else:
        feeds = {
            "obs": torch.cat(
                (
                    observations["policy"],
                    observations["height_scan_policy"].flatten(start_dim=1),
                ),
                dim=-1,
            )
            .cpu()
            .numpy()
        }
    try:
        import onnxruntime
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        actual = ReferenceEvaluator(graph).run(None, feeds)[0]
    else:
        session = onnxruntime.InferenceSession(
            str(export_path),
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(None, feeds)[0]
    assert actual.shape == (3, 29)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5)


def test_model_rejects_single_frame_selector_and_shape_mismatch() -> None:
    observations = _observations()
    with pytest.raises(
        ValueError,
        match="cnn_history_index is only supported",
    ):
        PropMLPElevationFusionModel(
            obs=observations,
            obs_groups={"actor": ["policy", "height_scan_policy"]},
            obs_set="actor",
            output_dim=29,
            elevation_set="height_scan_policy",
            vision_spatial_size=(28, 20),
            elevation_history_length=5,
            elevation_encoder_type="r2plus1d",
            cnn_history_index=-1,
        )

    wrong_history = _observations(history_length=4)
    with pytest.raises(
        ValueError,
        match="R\\(2\\+1\\)D elevation observation shape mismatch",
    ):
        _model(wrong_history, history_length=5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"history_length": 0}, "history_length"),
        ({"hidden_dims": ()}, "hidden_dims"),
        ({"spatial_kernel_sizes": (2, 3)}, "kernel sizes"),
        (
            {"temporal_kernel_sizes": (3,)},
            "temporal_kernel_sizes must contain",
        ),
        ({"spatial_strides": (2, 0)}, "spatial_strides"),
    ],
)
def test_encoder_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    config: dict[str, object] = {
        "history_length": 5,
        "hidden_dims": (8, 16),
        "spatial_kernel_sizes": (3, 3),
        "temporal_kernel_sizes": (3, 3),
        "spatial_strides": (2, 2),
        "out_dim": 32,
        "vision_spatial_size": (28, 20),
    }
    config.update(kwargs)
    with pytest.raises(ValueError, match=message):
        R2Plus1DElevationEncoder(**config)  # type: ignore[arg-type]


def test_encoder_rejects_runtime_shape_and_integer_input() -> None:
    encoder = R2Plus1DElevationEncoder(
        history_length=5,
        hidden_dims=(8,),
        spatial_kernel_sizes=(3,),
        temporal_kernel_sizes=(3,),
        spatial_strides=(2,),
        out_dim=16,
        vision_spatial_size=(28, 20),
    )
    with pytest.raises(ValueError, match="must have shape"):
        encoder(torch.zeros(2, 4, 28, 20))
    with pytest.raises(TypeError, match="floating point"):
        encoder(torch.zeros(2, 5, 28, 20, dtype=torch.int64))
