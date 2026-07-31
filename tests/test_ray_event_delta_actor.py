from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import (
    PropMLPElevationFusionModel,
)
from rsl_rl.utils.ray_event_observation import (
    pack_acquisition_delta_proprio_observation,
    pack_ray_event_observation,
)
from rsl_rl.utils.ray_event_deployment_receipt import (
    build_ray_event_deployment_receipt,
    validate_ray_event_deployment_receipt,
)


DELTA_DIM = 6
DELTA_SEMANTICS = (
    "base_linear_velocity_x_delta_mps",
    "base_linear_velocity_y_delta_mps",
    "base_angular_velocity_z_delta_radps",
    "gravity_projection_x_delta",
    "gravity_projection_y_delta",
    "gait_phase_delta_rad",
)


def _inputs(
    *, batch_size: int = 3, history_length: int = 1
) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
    torch.manual_seed(8101 + batch_size + history_length)
    valid = torch.rand(batch_size, history_length, 16, 96) > 0.55
    range_m = (torch.rand_like(valid, dtype=torch.float32) * 4.0 + 0.2) * valid
    packet_age = torch.zeros(batch_size, history_length)
    return_age = torch.where(
        valid,
        torch.rand_like(range_m) * 0.45 + 0.01,
        torch.zeros_like(range_m),
    )
    frame_valid = torch.ones(batch_size, history_length, dtype=torch.bool)
    event = pack_ray_event_observation(
        range_m,
        valid,
        return_age,
        packet_age,
        frame_valid,
        source="livox_per_return",
        temporal_baseline="per_return_age",
    )
    raw_delta = torch.randn(
        batch_size, history_length, DELTA_DIM, 16, 96
    )
    raw_delta = torch.where(
        valid[:, :, None], raw_delta, torch.zeros_like(raw_delta)
    )
    cell_count = history_length * 16 * 96
    winner_id = torch.arange(cell_count, dtype=torch.long).reshape(
        1, history_length, 16, 96
    ).expand(batch_size, -1, -1, -1)
    winner_id = torch.where(valid, winner_id, torch.full_like(winner_id, -1))
    delta = pack_acquisition_delta_proprio_observation(
        raw_delta,
        valid,
        winner_id,
        winner_id.clone(),
    )
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "ray_event_policy": event,
            "ray_event_delta_proprio": delta,
        },
        batch_size=[batch_size],
    )
    return obs, valid, winner_id


def _actor(
    *, batch_size: int = 3, history_length: int = 1
) -> tuple[PropMLPElevationFusionModel, TensorDict]:
    obs, _, _ = _inputs(
        batch_size=batch_size,
        history_length=history_length,
    )
    model = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups={
            "actor": [
                "policy",
                "ray_event_policy",
                "ray_event_delta_proprio",
            ]
        },
        obs_set="actor",
        output_dim=29,
        hidden_dims=[64],
        elevation_encoder_type="ray_event_time_delta",
        ray_time_set="ray_event_policy",
        ray_time_history_length=history_length,
        ray_time_spatial_size=(16, 96),
        ray_event_time_mode="per_return_age",
        ray_event_time_source="livox_per_return",
        ray_event_delta_proprio_set="ray_event_delta_proprio",
        ray_event_delta_proprio_dim=DELTA_DIM,
        ray_event_training_ready=False,
        prop_feature_dim=64,
        prop_hidden_dims=[64],
        vision_feature_dim=64,
    )
    return model, obs


def test_delta_actor_eager_backward_and_torchscript_consume_separate_tensor():
    model, obs = _actor(history_length=1)
    model.train()
    delta = obs["ray_event_delta_proprio"].detach().clone().requires_grad_(True)
    obs["ray_event_delta_proprio"] = delta
    output = model(obs)
    output.square().mean().backward()
    assert output.shape == (3, 29)
    assert delta.grad is not None and torch.isfinite(delta.grad).all()
    projection_grad = model.elevation_encoder.delta_proprio_projection.weight.grad
    assert projection_grad is not None
    assert torch.count_nonzero(projection_grad) > 0

    model.eval()
    with torch.no_grad():
        eager = model(obs)
        scripted = torch.jit.script(model.as_jit())
        exported = scripted(
            obs["policy"],
            obs["ray_event_policy"],
            obs["ray_event_delta_proprio"],
        )
    torch.testing.assert_close(exported, eager, rtol=1.0e-5, atol=1.0e-6)


def test_same_winner_packer_rejects_wrong_winner_and_invalid_leakage():
    obs, valid, winner_id = _inputs(batch_size=1)
    delta = obs["ray_event_delta_proprio"]
    wrong = winner_id.clone()
    first_valid = valid.nonzero(as_tuple=False)[0]
    wrong[tuple(first_valid)] += 1
    with pytest.raises(ValueError, match="same winner id"):
        pack_acquisition_delta_proprio_observation(
            delta,
            valid,
            winner_id,
            wrong,
        )

    leaked = delta.clone()
    first_invalid = (~valid).nonzero(as_tuple=False)[0]
    leak_index = (
        first_invalid[0],
        first_invalid[1],
        0,
        first_invalid[2],
        first_invalid[3],
    )
    leaked[leak_index] = 2.0
    with pytest.raises(ValueError, match="Invalid returns"):
        pack_acquisition_delta_proprio_observation(
            leaked,
            valid,
            winner_id,
            winner_id,
        )


def test_delta_actor_rejects_shape_nonfinite_and_invalid_leakage():
    model, obs = _actor(batch_size=1)
    wrong_shape = obs.clone()
    wrong_shape["ray_event_delta_proprio"] = obs[
        "ray_event_delta_proprio"
    ][:, :, :-1]
    with pytest.raises(ValueError, match=r"shape \[B,K,D,H,W\]"):
        model(wrong_shape)

    nonfinite = obs.clone()
    nonfinite_delta = obs["ray_event_delta_proprio"].clone()
    valid_index = (obs["ray_event_policy"][:, :, 1] > 0.5).nonzero()[0]
    nonfinite_delta[
        valid_index[0], valid_index[1], 0, valid_index[2], valid_index[3]
    ] = torch.nan
    nonfinite["ray_event_delta_proprio"] = nonfinite_delta
    with pytest.raises(ValueError, match="must be finite"):
        model(nonfinite)

    leaked = obs.clone()
    leaked_delta = obs["ray_event_delta_proprio"].clone()
    invalid_index = (obs["ray_event_policy"][:, :, 1] <= 0.5).nonzero()[0]
    leaked_delta[
        invalid_index[0], invalid_index[1], 0, invalid_index[2], invalid_index[3]
    ] = 1.0
    leaked["ray_event_delta_proprio"] = leaked_delta
    with pytest.raises(ValueError, match="Invalid returns"):
        model(leaked)


def test_scripted_valid_mask_prevents_invalid_delta_influence():
    model, obs = _actor(batch_size=1)
    scripted = torch.jit.script(model.eval().as_jit())
    clean = obs["ray_event_delta_proprio"]
    leaked = clean.clone()
    invalid = obs["ray_event_policy"][:, :, 1] <= 0.5
    leaked = torch.where(
        invalid[:, :, None],
        torch.full_like(leaked, 123.0),
        leaked,
    )
    with torch.no_grad():
        clean_output = scripted(obs["policy"], obs["ray_event_policy"], clean)
        leaked_output = scripted(obs["policy"], obs["ray_event_policy"], leaked)
    torch.testing.assert_close(leaked_output, clean_output, rtol=0.0, atol=0.0)


def test_old_two_and_five_channel_state_and_export_have_no_delta_keys():
    from tests.test_ray_event_actor_contract import _actor as old_actor
    from tests.test_ray_time_attention_encoder import _make_ray_time_actor

    model, obs = old_actor(
        history_length=1,
        temporal_baseline="per_return_age",
        source="livox_per_return",
    )
    assert not any("delta_proprio" in key for key in model.state_dict())
    wrapper = model.as_onnx(verbose=False, input_mode="split")
    assert wrapper.input_names == ["proprio_obs", "ray_history"]
    assert len(wrapper.get_dummy_inputs()) == 2
    with torch.no_grad():
        assert model(obs).shape == (2, 29)

    legacy_model, legacy_obs = _make_ray_time_actor(
        history_length=1,
        use_query_attention=True,
    )
    assert not any("delta_proprio" in key for key in legacy_model.state_dict())
    legacy_wrapper = legacy_model.as_onnx(verbose=False, input_mode="split")
    assert legacy_wrapper.input_names == ["proprio_obs", "ray_history"]
    assert len(legacy_wrapper.get_dummy_inputs()) == 2
    with torch.no_grad():
        assert legacy_model(legacy_obs).shape == (3, 29)


def test_delta_actor_onnx_dynamic_batch_three_matches_eager(tmp_path: Any):
    onnx = pytest.importorskip("onnx")
    model, obs = _actor(batch_size=3, history_length=1)
    model.eval()
    wrapper = model.as_onnx(verbose=False, input_mode="split").eval()
    dummy_inputs = wrapper.get_dummy_inputs()
    assert [tuple(item.shape) for item in dummy_inputs] == [
        (1, 96),
        (1, 1, 5, 16, 96),
        (1, 1, DELTA_DIM, 16, 96),
    ]
    runtime_inputs = (
        obs["policy"],
        obs["ray_event_policy"],
        obs["ray_event_delta_proprio"],
    )
    with torch.inference_mode():
        eager = model(obs)
        wrapper_output = wrapper(*runtime_inputs)
    torch.testing.assert_close(wrapper_output, eager, rtol=0.0, atol=0.0)

    path = tmp_path / "ray_event_delta_dynamic.onnx"
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        path,
        export_params=True,
        opset_version=18,
        external_data=False,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
        dynamic_axes=wrapper.dynamic_axes,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert [item.name for item in graph.graph.input] == [
        "proprio_obs",
        "ray_history",
        "acquisition_delta_proprio",
    ]
    assert all(
        item.type.tensor_type.shape.dim[0].dim_param
        for item in graph.graph.input
    )
    feeds = {
        "proprio_obs": runtime_inputs[0].detach().numpy(),
        "ray_history": runtime_inputs[1].detach().numpy(),
        "acquisition_delta_proprio": runtime_inputs[2].detach().numpy(),
    }
    try:
        import onnxruntime
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        onnx_output = ReferenceEvaluator(graph).run(None, feeds)[0]
    else:
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run(None, feeds)[0]
    assert onnx_output.shape == (3, 29)
    np.testing.assert_allclose(
        onnx_output,
        eager.detach().numpy(),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_delta_receipt_binds_semantics_dimension_and_export_but_stays_unready():
    receipt = build_ray_event_deployment_receipt(
        history_length=1,
        spatial_size=(16, 96),
        source="livox_per_return",
        temporal_baseline="per_return_age",
        history_reduction="pies_latest_event_k1",
        packetization_invariance_proof_sha256=(
            "e8d461a7b4e0d14bfa00bc6bfd95cd6e85a0c0f79f9a87b2007b838fb0b8c332"
        ),
        real_tensor_manifest_sha256="1" * 64,
        clock_alignment_receipt_sha256="2" * 64,
        self_return_filter="upstream_static_mask",
        self_return_filter_config_sha256="3" * 64,
        self_return_filtered_count=9,
        delta_proprio_observation_group="ray_event_delta_proprio",
        delta_proprio_dim=DELTA_DIM,
        delta_proprio_semantics=DELTA_SEMANTICS,
        delta_actor_interface_closed=True,
        delta_export_interface_closed=True,
    )
    assert receipt["delta_proprio_input_shape"] == [1, DELTA_DIM, 16, 96]
    assert receipt["delta_proprio_semantics"] == list(DELTA_SEMANTICS)
    assert receipt["delta_proprio_semantics_sha256"] == (
        "20da705d5ab66e9b99514184e0f1f8d94bba6bc3217a6c03f69dea5b0b0b7512"
    )
    assert receipt["delta_actor_export_contract"] == {
        "observation_group": "ray_event_delta_proprio",
        "logical_layout": "B,K,D,H,W",
        "torchscript_input": "acquisition_delta_proprio",
        "onnx_input": "acquisition_delta_proprio",
        "dynamic_batch": True,
    }
    assert receipt["delta_proprio_source_authenticated"] is False
    assert receipt["acquisition_delta_proprio_contract"] == (
        "actor_export_wired_source_unreceipted"
    )
    assert receipt["pies_full_contract_ready"] is False
    assert receipt["training_ready"] is False
    validate_ray_event_deployment_receipt(receipt)

    tampered = dict(receipt)
    tampered["delta_proprio_semantics"] = list(reversed(DELTA_SEMANTICS))
    with pytest.raises(ValueError, match="semantic hash"):
        validate_ray_event_deployment_receipt(tampered)
