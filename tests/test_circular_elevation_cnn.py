"""Panorama seam, checkpoint, and deployment regression tests."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from rsl_rl.modules.elevation_2D_cnn_encoder import Elevation2DCNNEncoder


def _encoder(circular=False):
    return Elevation2DCNNEncoder(
        in_channels=5, hidden_dims=(8, 16), kernel_sizes=(3, 3),
        strides=(2, 2), vision_spatial_size=(16, 96), circular_azimuth=circular,
    )


@pytest.mark.parametrize("layer_index,height,width", [(0, 16, 96), (3, 8, 48)])
def test_each_layer_wraps_seam_but_not_vertical_boundary(layer_index, height, width):
    layer = _encoder(True).conv[layer_index]
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.zero_()
    value = torch.zeros(1, layer.in_channels, height, width, requires_grad=True)
    with torch.no_grad():
        value[0, 0, 0, -1] = 1
        value[0, 0, -1, 0] = 10
    output = layer(value)
    # Top-left sees the last column, but must not see the bottom row.
    assert output[0, 0, 0, 0].item() == 1
    output[0, 0, 0, 0].backward()
    assert value.grad[0, 0, 0, -1].item() == 1
    assert value.grad[0, 0, -1, 0].item() == 0


def test_checkpoint_shapes_and_legacy_zero_padding_are_preserved():
    legacy, circular = _encoder(), _encoder(True)
    circular.load_state_dict(legacy.state_dict(), strict=True)
    assert list(legacy.state_dict()) == list(circular.state_dict())
    x = torch.randn(3, 5, 16, 96, requires_grad=True)
    assert legacy(x).shape == circular(x).shape == (3, 64)
    circular(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in circular.parameters())
    seam = torch.zeros(1, 5, 16, 96)
    seam[:, :, :, -1] = 1
    with torch.no_grad():
        legacy.conv[0].weight.fill_(1)
        legacy.conv[0].bias.zero_()
    assert legacy.conv[0](seam)[0, 0, 0, 0].item() == 0


def _actor_and_obs():
    obs = TensorDict({
        "policy": torch.randn(3, 96),
        "height_scan_policy": torch.rand(3, 5, 16, 96) + 0.05,
    }, batch_size=[3])
    actor = PropMLPElevationFusionModel(
        obs=obs, obs_groups={"actor": ["policy", "height_scan_policy"]},
        obs_set="actor", output_dim=29, elevation_set="height_scan_policy",
        cnn_observation_type="depthcamera", depth_camera_far=1.857,
        vision_spatial_size=(16, 96), elevation_history_length=5,
        cnn_hidden_dims=(8, 16), cnn_kernel_sizes=(3, 3), cnn_strides=(2, 2),
        cnn_circular_azimuth=True,
    ).eval()
    return actor, obs


def test_actor_wiring_and_torchscript_match():
    actor, obs = _actor_and_obs()
    # Match the standalone seam-tested encoder to prove the model forwards the option.
    encoder = _encoder(True).eval()
    encoder.load_state_dict(actor.elevation_encoder.state_dict(), strict=True)
    depth = obs["height_scan_policy"]
    torch.testing.assert_close(actor.elevation_encoder(depth), encoder(depth))
    scripted = torch.jit.script(actor.as_jit().eval())
    torch.testing.assert_close(
        scripted(obs["policy"], depth), actor(obs), rtol=0, atol=0,
    )


@pytest.mark.parametrize("input_mode", ["split", "single"])
def test_actor_onnx_dynamic_batch_matches(tmp_path, input_mode):
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    actor, obs = _actor_and_obs()
    exported = actor.as_onnx(verbose=False, input_mode=input_mode).eval()
    path = tmp_path / "circular_actor.onnx"
    torch.onnx.export(
        exported, exported.get_dummy_inputs(), path, opset_version=18,
        input_names=exported.input_names, output_names=exported.output_names,
        dynamic_axes=exported.dynamic_axes,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    if input_mode == "split":
        feeds = {"proprio_obs": obs["policy"].numpy(),
                 "elevation_obs": obs["height_scan_policy"].numpy()}
    else:
        feeds = {"obs": torch.cat((obs["policy"], obs["height_scan_policy"].flatten(1)), -1).numpy()}
    actual = ReferenceEvaluator(graph).run(None, feeds)[0]
    np.testing.assert_allclose(actual, actor(obs).detach().numpy(), rtol=1e-4, atol=1e-5)
