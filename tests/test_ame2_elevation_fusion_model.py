# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder


_BATCH_SIZE = 4
_PROPRIO_DIM = 96
_ACTION_DIM = 29
_HEIGHT_HISTORY_SHAPE = (5, 28, 20)


def _make_observations(
    *,
    batch_size: int = _BATCH_SIZE,
    height_history_shape: tuple[int, int, int] = _HEIGHT_HISTORY_SHAPE,
    requires_grad: bool = False,
) -> TensorDict:
    proprio = torch.randn(batch_size, _PROPRIO_DIM, requires_grad=requires_grad)
    height_history = torch.randn(batch_size, *height_history_shape, requires_grad=requires_grad)
    return TensorDict(
        {
            "policy": proprio,
            "height_scan_policy": height_history,
        },
        batch_size=[batch_size],
    )


def _make_actor(obs: TensorDict, **overrides: Any) -> PropMLPElevationFusionModel:
    kwargs: dict[str, Any] = {
        "obs": obs,
        "obs_groups": {"actor": ["policy", "height_scan_policy"]},
        "obs_set": "actor",
        "output_dim": _ACTION_DIM,
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "elevation_set": "height_scan_policy",
        "cnn_observation_type": "elevationmap",
        "vision_spatial_size": (28, 20),
        "vision_feature_dim": 64,
        "elevation_history_length": 5,
        "cnn_hidden_dims": [8, 16],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [128],
        "use_prop_encoder": True,
    }
    kwargs.update(overrides)
    return PropMLPElevationFusionModel(**kwargs)


def _ame2_overrides() -> dict[str, Any]:
    return {
        "elevation_encoder_type": "ame2",
        "ame2_map_extent": (1.35, 0.95),
        "ame2_history_index": -1,
        "ame2_token_spatial_size": (7, 5),
        "ame2_local_channels": (8, 16),
        "ame2_position_feature_dim": 16,
        "ame2_point_feature_dim": 64,
        "ame2_global_feature_dim": 32,
        "ame2_attention_dim": 32,
        "ame2_num_heads": 4,
        "ame2_height_scale": 0.6,
    }


def _cnn1_overrides() -> dict[str, Any]:
    return {
        "elevation_encoder_type": "cnn",
        "cnn_history_index": -1,
    }


def _make_critic(batch_size: int = _BATCH_SIZE) -> tuple[PropMLPElevationFusionModel, TensorDict]:
    obs = TensorDict(
        {
            "critic": torch.randn(batch_size, 99),
            "height_scan_critic": torch.randn(batch_size, *_HEIGHT_HISTORY_SHAPE),
        },
        batch_size=[batch_size],
    )
    model = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups={"critic": ["critic", "height_scan_critic"]},
        obs_set="critic",
        output_dim=1,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        elevation_set="height_scan_critic",
        cnn_observation_type="elevationmap",
        vision_spatial_size=(28, 20),
        vision_feature_dim=64,
        elevation_history_length=5,
        cnn_hidden_dims=[8, 16],
        cnn_kernel_sizes=[3, 3],
        cnn_strides=[2, 2],
        prop_feature_dim=64,
        prop_hidden_dims=[128],
        use_prop_encoder=True,
    )
    return model, obs


def _assert_finite_nonzero_gradients(module: nn.Module) -> None:
    named_parameters = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    assert named_parameters
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    assert not missing, f"Missing gradients for: {missing}"
    gradients = [parameter.grad for _, parameter in named_parameters if parameter.grad is not None]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_default_encoder_is_exactly_equivalent_to_explicit_cnn() -> None:
    torch.manual_seed(101)
    obs = _make_observations()

    torch.manual_seed(211)
    default_model = _make_actor(obs).eval()
    torch.manual_seed(211)
    explicit_cnn_model = _make_actor(obs, elevation_encoder_type="cnn").eval()

    default_state = default_model.state_dict()
    explicit_state = explicit_cnn_model.state_dict()
    assert list(default_state) == list(explicit_state)
    assert [tensor.shape for tensor in default_state.values()] == [
        tensor.shape for tensor in explicit_state.values()
    ]
    for key in default_state:
        torch.testing.assert_close(default_state[key], explicit_state[key], rtol=0.0, atol=0.0)

    with torch.inference_mode():
        default_output = default_model(obs)
        explicit_output = explicit_cnn_model(obs)
    torch.testing.assert_close(default_output, explicit_output, rtol=0.0, atol=0.0)


def test_cnn1_uses_only_latest_frame_and_preserves_actor_contract() -> None:
    torch.manual_seed(251)
    obs = _make_observations(requires_grad=True)
    model = _make_actor(obs, **_cnn1_overrides()).eval()

    first_conv = model.elevation_encoder.conv[0]
    assert isinstance(first_conv, nn.Conv2d)
    assert first_conv.in_channels == 1
    assert model._cnn_use_single_frame is True
    assert model._cnn_history_index == 4
    assert isinstance(model.mlp[0], nn.Linear)
    assert model.mlp[0].in_features == 128

    with torch.no_grad():
        reference = model(obs)
        changed_old_history = obs.clone()
        changed_old_history["height_scan_policy"][:, :-1] += 100.0
        old_history_output = model(changed_old_history)
        changed_latest = obs.clone()
        changed_latest["height_scan_policy"][:, -1] += torch.linspace(
            -2.0,
            2.0,
            _HEIGHT_HISTORY_SHAPE[1] * _HEIGHT_HISTORY_SHAPE[2],
        ).reshape(_HEIGHT_HISTORY_SHAPE[1:])
        latest_output = model(changed_latest)

    assert reference.shape == (_BATCH_SIZE, _ACTION_DIM)
    torch.testing.assert_close(old_history_output, reference, rtol=0.0, atol=0.0)
    assert not torch.allclose(latest_output, reference)

    target = torch.randn_like(reference)
    F.mse_loss(model(obs), target).backward()
    height_gradient = obs["height_scan_policy"].grad
    assert height_gradient is not None
    assert torch.count_nonzero(height_gradient[:, :-1]) == 0
    assert torch.isfinite(height_gradient[:, -1]).all()
    assert float(height_gradient[:, -1].abs().sum()) > 0.0


@pytest.mark.parametrize("history_index", (5, -6))
def test_cnn1_rejects_out_of_range_history_index(history_index: int) -> None:
    obs = _make_observations()
    with pytest.raises(ValueError, match="cnn_history_index is out of range"):
        _make_actor(obs, cnn_history_index=history_index)


def test_cnn1_checkpoint_strict_reload_and_output_match() -> None:
    torch.manual_seed(263)
    obs = _make_observations()
    model = _make_actor(obs, **_cnn1_overrides()).eval()
    model.update_normalization(obs)

    with torch.inference_mode():
        reference = model(obs)
    checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}

    reloaded = _make_actor(obs, **_cnn1_overrides()).eval()
    incompatible = reloaded.load_state_dict(checkpoint, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    with torch.inference_mode():
        restored = reloaded(obs)
    torch.testing.assert_close(restored, reference, rtol=0.0, atol=0.0)


def test_ame2_actor_forward_latent_decoder_and_backward_chain() -> None:
    torch.manual_seed(307)
    obs = _make_observations(requires_grad=True)
    model = _make_actor(obs, **_ame2_overrides()).train()

    with torch.no_grad():
        latent = model.get_latent(obs)
    assert latent.shape == (_BATCH_SIZE, 128)
    assert isinstance(model.mlp[0], nn.Linear)
    assert model.mlp[0].in_features == 128
    assert model.elevation_encoder.num_heads == 4
    assert model.elevation_encoder.head_dim == 8

    output = model(obs)
    assert output.shape == (_BATCH_SIZE, _ACTION_DIM)
    target = torch.randn_like(output)
    F.mse_loss(output, target).backward()

    _assert_finite_nonzero_gradients(model.elevation_encoder)
    _assert_finite_nonzero_gradients(model.prop_mlp)
    _assert_finite_nonzero_gradients(model.mlp)

    proprio_gradient = obs["policy"].grad
    height_gradient = obs["height_scan_policy"].grad
    assert proprio_gradient is not None
    assert height_gradient is not None
    assert torch.isfinite(proprio_gradient).all()
    assert torch.isfinite(height_gradient).all()
    assert float(proprio_gradient.abs().sum()) > 0.0
    assert float(height_gradient.abs().sum()) > 0.0


def test_ame2_checkpoint_strict_save_reload_and_output_match() -> None:
    torch.manual_seed(313)
    obs = _make_observations()
    model = _make_actor(obs, **_ame2_overrides()).eval()
    model.update_normalization(obs)

    with torch.inference_mode():
        reference = model(obs)
    checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}

    reloaded = _make_actor(obs, **_ame2_overrides()).eval()
    incompatible = reloaded.load_state_dict(checkpoint, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    with torch.inference_mode():
        restored = reloaded(obs)
    torch.testing.assert_close(restored, reference, rtol=0.0, atol=0.0)


def test_critic_remains_cnn_and_outputs_scalar_value() -> None:
    torch.manual_seed(317)
    critic, obs = _make_critic()

    assert critic.elevation_encoder_type == "cnn"
    assert isinstance(critic.elevation_encoder, Elevation2DCNNEncoder)
    output = critic(obs)
    assert output.shape == (_BATCH_SIZE, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("height_history_shape", "error_match"),
    [
        ((4, 28, 20), "history mismatch"),
        ((5, 27, 20), "spatial shape mismatch"),
    ],
)
def test_ame2_rejects_mismatched_observation_shape_at_construction(
    height_history_shape: tuple[int, int, int],
    error_match: str,
) -> None:
    obs = _make_observations(height_history_shape=height_history_shape)
    with pytest.raises(ValueError, match=error_match):
        _make_actor(obs, elevation_encoder_type="ame2")


def test_ame2_rejects_unsupported_modality_and_runtime_shape() -> None:
    obs = _make_observations()
    with pytest.raises(ValueError, match="only supports cnn_observation_type='elevationmap'"):
        _make_actor(
            obs,
            elevation_encoder_type="ame2",
            cnn_observation_type="depthcamera",
        )

    model = _make_actor(obs, **_ame2_overrides()).eval()
    wrong_runtime_obs = _make_observations(height_history_shape=(5, 28, 19))
    with pytest.raises(ValueError, match=r"height_history must have shape \[B, T, H, W\]"):
        model(wrong_runtime_obs)


def test_cnn_jit_matches_eager_output_exactly() -> None:
    torch.manual_seed(401)
    obs = _make_observations()
    torch.manual_seed(409)
    model = _make_actor(obs, elevation_encoder_type="cnn").eval()
    scripted_model = torch.jit.script(model.as_jit().eval())

    with torch.inference_mode():
        eager_output = model(obs)
        scripted_output = scripted_model(obs["policy"], obs["height_scan_policy"])

    assert eager_output.shape == (_BATCH_SIZE, _ACTION_DIM)
    assert scripted_output.shape == eager_output.shape
    torch.testing.assert_close(scripted_output, eager_output, rtol=0.0, atol=0.0)


def test_cnn1_jit_matches_eager_output_exactly_with_batch_three() -> None:
    torch.manual_seed(413)
    obs = _make_observations(batch_size=3)
    model = _make_actor(obs, **_cnn1_overrides()).eval()
    scripted_model = torch.jit.script(model.as_jit().eval())

    with torch.inference_mode():
        eager_output = model(obs)
        scripted_output = scripted_model(obs["policy"], obs["height_scan_policy"])

    assert eager_output.shape == (3, _ACTION_DIM)
    torch.testing.assert_close(scripted_output, eager_output, rtol=0.0, atol=0.0)


def test_ame2_jit_matches_eager_output_exactly_with_batch_three() -> None:
    torch.manual_seed(419)
    obs = _make_observations(batch_size=3)
    model = _make_actor(obs, **_ame2_overrides()).eval()
    scripted_model = torch.jit.script(model.as_jit().eval())

    with torch.inference_mode():
        eager_output = model(obs)
        scripted_output = scripted_model(obs["policy"], obs["height_scan_policy"])

    assert eager_output.shape == (3, _ACTION_DIM)
    assert scripted_output.shape == eager_output.shape
    torch.testing.assert_close(scripted_output, eager_output, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("input_mode", ("split", "single"))
def test_ame2_onnx_dynamic_batch_three_matches_eager(
    tmp_path: Any, input_mode: str
) -> None:
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(431)
    obs = _make_observations(batch_size=3)
    model = _make_actor(obs, **_ame2_overrides()).eval()
    onnx_model = model.as_onnx(verbose=False, input_mode=input_mode).eval()
    export_path = tmp_path / f"ame2_{input_mode}.onnx"

    dummy_inputs = onnx_model.get_dummy_inputs()
    assert dummy_inputs[0].shape[0] == 1
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
    with torch.inference_mode():
        eager_output = model(obs).cpu().numpy()

    if input_mode == "split":
        feeds = {
            "proprio_obs": obs["policy"].detach().cpu().numpy(),
            "elevation_obs": obs["height_scan_policy"].detach().cpu().numpy(),
        }
    else:
        flat_obs = torch.cat(
            (obs["policy"], obs["height_scan_policy"].flatten(start_dim=1)),
            dim=-1,
        )
        feeds = {"obs": flat_obs.detach().cpu().numpy()}

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

    assert onnx_output.shape == (3, _ACTION_DIM)
    np.testing.assert_allclose(onnx_output, eager_output, rtol=1.0e-4, atol=1.0e-5)


@pytest.mark.parametrize("input_mode", ("split", "single"))
def test_cnn1_onnx_dynamic_batch_three_matches_eager(
    tmp_path: Any, input_mode: str
) -> None:
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(439)
    obs = _make_observations(batch_size=3)
    model = _make_actor(obs, **_cnn1_overrides()).eval()
    onnx_model = model.as_onnx(verbose=False, input_mode=input_mode).eval()
    export_path = tmp_path / f"cnn1_{input_mode}.onnx"

    dummy_inputs = onnx_model.get_dummy_inputs()
    assert dummy_inputs[0].shape[0] == 1
    if input_mode == "single":
        assert dummy_inputs[0].shape == (1, _PROPRIO_DIM + 5 * 28 * 20)
    else:
        assert dummy_inputs[1].shape == (1, 5, 28, 20)
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
    with torch.inference_mode():
        eager_output = model(obs).cpu().numpy()

    if input_mode == "split":
        feeds = {
            "proprio_obs": obs["policy"].detach().cpu().numpy(),
            "elevation_obs": obs["height_scan_policy"].detach().cpu().numpy(),
        }
    else:
        flat_obs = torch.cat(
            (obs["policy"], obs["height_scan_policy"].flatten(start_dim=1)),
            dim=-1,
        )
        feeds = {"obs": flat_obs.detach().cpu().numpy()}

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

    assert onnx_output.shape == (3, _ACTION_DIM)
    np.testing.assert_allclose(onnx_output, eager_output, rtol=1.0e-4, atol=1.0e-5)
