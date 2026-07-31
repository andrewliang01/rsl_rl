from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import (
    PropMLPElevationFusionModel,
)
from rsl_rl.utils.mid360_ray_time_builder import Mid360AlignedRayTimeHistory
from rsl_rl.utils.ray_event_deployment_receipt import (
    build_ray_event_deployment_receipt,
    validate_ray_event_deployment_receipt,
)
from rsl_rl.utils.ray_event_observation import (
    aligned_history_to_ray_event_observation,
    pack_ray_event_observation,
)
from tests.test_mid360_ray_time_builder import (
    _event_time_builder,
    _timed_packet,
    _xyz_from_spherical,
)


def _event_observation(
    *,
    batch_size: int = 2,
    history_length: int = 5,
    temporal_baseline: str = "packet_age",
) -> torch.Tensor:
    range_m = torch.rand(batch_size, history_length, 16, 96) * 5.0 + 0.2
    valid = torch.rand_like(range_m) > 0.4
    range_m = range_m * valid
    packet_age = torch.arange(
        history_length - 1,
        -1,
        -1,
        dtype=torch.float32,
    )[None].expand(batch_size, -1) * 0.1
    frame_valid = torch.ones(batch_size, history_length, dtype=torch.bool)
    return_age = packet_age[:, :, None, None].expand_as(range_m)
    if temporal_baseline == "per_return_age":
        return_age = return_age + 0.02
    return_age = torch.where(valid, return_age, torch.zeros_like(return_age))
    source = {
        "per_return_age": "livox_per_return",
        "quantized_event_age": "raycaster_quantized_event",
    }.get(temporal_baseline, "raycaster_packet")
    return pack_ray_event_observation(
        range_m,
        valid,
        return_age,
        packet_age,
        frame_valid,
        source=source,
        temporal_baseline=temporal_baseline,
    )


def _actor(
    *,
    history_length: int,
    temporal_baseline: str,
    source: str,
    batch_size: int = 2,
) -> tuple[PropMLPElevationFusionModel, TensorDict]:
    event = _event_observation(
        batch_size=batch_size,
        history_length=history_length,
        temporal_baseline=temporal_baseline,
    )
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "ray_event_policy": event,
        },
        batch_size=[batch_size],
    )
    model = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups={"actor": ["policy", "ray_event_policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=[64],
        elevation_encoder_type="ray_event_time",
        ray_time_set="ray_event_policy",
        ray_time_history_length=history_length,
        ray_time_spatial_size=(16, 96),
        ray_event_time_mode=temporal_baseline,
        ray_event_time_source=source,
        ray_event_training_ready=False,
        prop_feature_dim=64,
        prop_hidden_dims=[64],
        vision_feature_dim=64,
    ).eval()
    return model, obs


@pytest.mark.parametrize("history_length", [1, 5])
@pytest.mark.parametrize("baseline", ["packet_age", "age_zero"])
def test_packet_level_actor_forward_and_torchscript(history_length, baseline):
    model, obs = _actor(
        history_length=history_length,
        temporal_baseline=baseline,
        source="raycaster_packet",
    )
    with torch.no_grad():
        eager = model(obs)
        scripted = torch.jit.script(model.as_jit())
        exported = scripted(obs["policy"], obs["ray_event_policy"])
    assert eager.shape == (2, 29)
    torch.testing.assert_close(exported, eager, rtol=1.0e-5, atol=1.0e-6)


def test_authenticated_per_return_actor_forward():
    model, obs = _actor(
        history_length=5,
        temporal_baseline="per_return_age",
        source="livox_per_return",
    )
    output = model(obs)
    assert output.shape == (2, 29)
    assert torch.isfinite(output).all()


def test_quantized_raster_event_actor_is_named_separately_from_per_return():
    model, obs = _actor(
        history_length=1,
        temporal_baseline="quantized_event_age",
        source="raycaster_quantized_event",
    )
    output = model(obs)
    assert output.shape == (2, 29)
    assert model.elevation_encoder.event_time_mode == "quantized_event_age"
    assert model.elevation_encoder.event_time_source == "raycaster_quantized_event"


def test_raycaster_per_return_claim_and_training_ready_are_fail_closed():
    event = _event_observation(history_length=5, temporal_baseline="packet_age")
    obs = TensorDict(
        {"policy": torch.randn(2, 96), "ray_event_policy": event},
        batch_size=[2],
    )
    kwargs = dict(
        obs=obs,
        obs_groups={"actor": ["policy", "ray_event_policy"]},
        obs_set="actor",
        output_dim=29,
        elevation_encoder_type="ray_event_time",
        ray_time_set="ray_event_policy",
        ray_time_history_length=5,
        ray_time_spatial_size=(16, 96),
        ray_event_time_mode="per_return_age",
        ray_event_time_source="raycaster_packet",
    )
    with pytest.raises(ValueError, match="RayCaster exposes packet-level"):
        PropMLPElevationFusionModel(**kwargs)
    kwargs["ray_event_time_mode"] = "packet_age"
    kwargs["ray_event_training_ready"] = True
    with pytest.raises(ValueError, match="64-environment smoke"):
        PropMLPElevationFusionModel(**kwargs)


def test_onnx_wrapper_uses_five_channels_in_split_and_single_modes():
    model, obs = _actor(
        history_length=5,
        temporal_baseline="packet_age",
        source="raycaster_packet",
    )
    split = model.as_onnx(verbose=False, input_mode="split")
    single = model.as_onnx(verbose=False, input_mode="single")
    with torch.no_grad():
        expected = model(obs)
        split_output = split(obs["policy"], obs["ray_event_policy"])
        flat = torch.cat(
            (obs["policy"], obs["ray_event_policy"].flatten(start_dim=1)),
            dim=1,
        )
        single_output = single(flat)
    assert split.get_dummy_inputs()[1].shape == (1, 5, 5, 16, 96)
    assert single.single_input_size == 96 + 5 * 5 * 16 * 96
    torch.testing.assert_close(split_output, expected)
    torch.testing.assert_close(single_output, expected)


@pytest.mark.parametrize("history_length", [1, 5])
def test_packet_age_actor_exports_dynamic_batch_to_onnx(history_length):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    model, _ = _actor(
        history_length=history_length,
        temporal_baseline="packet_age",
        source="raycaster_packet",
    )
    wrapper = model.as_onnx(verbose=False, input_mode="split").eval()
    exported = torch.onnx.export(
        wrapper,
        wrapper.get_dummy_inputs(),
        f=None,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
        dynamic_axes=wrapper.dynamic_axes,
        dynamo=True,
    )
    inputs = exported.model_proto.graph.input
    assert [item.name for item in inputs] == ["proprio_obs", "ray_history"]
    ray_shape = inputs[1].type.tensor_type.shape.dim
    assert ray_shape[1].dim_value == history_length
    assert ray_shape[2].dim_value == 5


def test_aligned_builder_snapshot_packs_range_valid_and_age_without_reassociation():
    ray_history = np.zeros((2, 2, 16, 96), dtype=np.float32)
    valid = np.zeros((2, 16, 96), dtype=np.bool_)
    valid[0, 3, 4] = True
    valid[1, 5, 7] = True
    ray_history[:, 1] = valid
    ray_history[0, 0, 3, 4] = 1.5
    ray_history[1, 0, 5, 7] = 2.5
    return_age = np.zeros((2, 16, 96), dtype=np.float32)
    return_age[0, 3, 4] = 0.23
    return_age[1, 5, 7] = 0.04
    snapshot = Mid360AlignedRayTimeHistory(
        ray_history=ray_history,
        return_valid=valid,
        return_age_s=return_age,
        packet_age_s=np.asarray((0.2, 0.0), dtype=np.float32),
        frame_valid=np.asarray((True, True), dtype=np.bool_),
        window_indices=np.asarray((8, 9), dtype=np.int64),
        capture_end_times_s=np.asarray((1.8, 2.0), dtype=np.float64),
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:test",
    )
    packed = aligned_history_to_ray_event_observation(snapshot)
    assert packed.shape == (2, 5, 16, 96)
    assert packed[0, 0, 3, 4] == pytest.approx(1.5)
    assert packed[0, 1, 3, 4] == 1.0
    assert packed[0, 2, 3, 4] == pytest.approx(0.23)
    assert np.all(packed[0, 3] == np.float32(0.2))
    assert np.all(packed[:, 4] == 1.0)


def test_real_builder_to_observation_to_actor_and_torchscript(tmp_path):
    builder = _event_time_builder(tmp_path / "builder", history_length=2)
    builder.ingest_point_packet(
        _timed_packet(
            points=np.stack(
                (
                    _xyz_from_spherical(1.4, -10.0, -20.0),
                    _xyz_from_spherical(2.2, 5.0, 30.0),
                )
            ),
            point_timestamps_s=np.asarray((1.02, 1.08), dtype=np.float64),
            window_index=0,
            capture_start_s=1.0,
            capture_end_s=1.1,
        )
    )
    builder.ingest_point_packet(
        _timed_packet(
            points=np.stack(
                (
                    _xyz_from_spherical(1.1, -12.0, -25.0),
                    _xyz_from_spherical(2.8, 8.0, 40.0),
                )
            ),
            point_timestamps_s=np.asarray((1.14, 1.19), dtype=np.float64),
            window_index=1,
            capture_start_s=1.1,
            capture_end_s=1.2,
        )
    )
    aligned = builder.aligned_event_time_history(
        now_s=1.2,
        monotonic_clock_domain="CLOCK_MONOTONIC_RAW:boot-7",
    )
    packed = torch.from_numpy(
        aligned_history_to_ray_event_observation(aligned)
    ).unsqueeze(0)
    model, obs = _actor(
        history_length=2,
        temporal_baseline="per_return_age",
        source="livox_per_return",
        batch_size=1,
    )
    obs["ray_event_policy"] = packed
    with torch.no_grad():
        eager = model(obs)
        scripted = torch.jit.script(model.as_jit())
        exported = scripted(obs["policy"], packed)
    assert aligned.return_valid.any()
    assert eager.shape == (1, 29)
    torch.testing.assert_close(exported, eager, rtol=1.0e-5, atol=1.0e-6)


def test_ray_event_deployment_receipt_records_capability_and_gate():
    receipt = build_ray_event_deployment_receipt(
        history_length=5,
        spatial_size=(16, 96),
        source="raycaster_packet",
        temporal_baseline="packet_age",
        packet_time_quantization_upper_bound_s=0.1,
    )
    assert receipt["training_ready"] is False
    assert receipt["per_return_claim_allowed"] is False
    assert receipt["input_shape"] == [5, 5, 16, 96]
    assert receipt["self_return_filter"] == "unknown"
    assert receipt["deployment_scope"] == "synthetic_conformance_only"
    assert receipt["filtered_return_semantics"] == (
        "removed_observation_not_emitted_no_return"
    )
    assert receipt["channel_semantics"]["return_age_s"].startswith(
        "same_winner"
    )
    assert receipt["actor_export_contract"] == {
        "observation_group": "ray_event_policy",
        "logical_layout": "B,K,C,H,W",
        "torchscript_input": "ray_event_policy",
        "onnx_input": "ray_event_policy",
        "dynamic_batch": True,
    }
    assert receipt["event_window_s"] is None
    validate_ray_event_deployment_receipt(receipt)
    with pytest.raises(ValueError, match="cannot claim"):
        build_ray_event_deployment_receipt(
            history_length=5,
            spatial_size=(16, 96),
            source="raycaster_packet",
            temporal_baseline="per_return_age",
            packet_time_quantization_upper_bound_s=0.1,
        )


def test_latest_receipt_separates_raster_prototype_from_raw_pies_proof():
    prototype = build_ray_event_deployment_receipt(
        history_length=1,
        spatial_size=(16, 96),
        source="raycaster_quantized_event",
        temporal_baseline="quantized_event_age",
        history_reduction="raster_latest_event_prototype",
        packet_time_quantization_upper_bound_s=0.1,
    )
    assert prototype["event_window_s"] == 0.5
    assert prototype["event_union_stage"] == "post_packet_raster"
    assert prototype["packetization_invariance_proven"] is False
    assert prototype["packetization_invariance_proof_sha256"] is None

    proof_sha = (
        "e8d461a7b4e0d14bfa00bc6bfd95cd6e85a0c0f79f9a87b2007b838fb0b8c332"
    )
    raw_event = build_ray_event_deployment_receipt(
        history_length=1,
        spatial_size=(16, 96),
        source="livox_per_return",
        temporal_baseline="per_return_age",
        history_reduction="pies_latest_event_k1",
        event_union_stage="raw_event",
        packetization_invariance_proof_sha256=proof_sha,
        self_return_filter="upstream_static_mask",
        self_return_filter_config_sha256="b" * 64,
        self_return_filtered_count=7,
    )
    assert raw_event["packetization_invariance_proven"] is True
    assert raw_event["packetization_invariance_proof_sha256"] == proof_sha
    validate_ray_event_deployment_receipt(raw_event)
    assert raw_event["pies_full_contract_ready"] is False
    assert raw_event["acquisition_delta_proprio_contract"].endswith(
        "actor_not_wired"
    )


def test_training_ready_fails_closed_without_real_self_return_filter():
    with pytest.raises(ValueError, match="self-return filtering"):
        build_ray_event_deployment_receipt(
            history_length=5,
            spatial_size=(16, 96),
            source="raycaster_packet",
            temporal_baseline="packet_age",
            packet_time_quantization_upper_bound_s=0.1,
            training_ready=True,
            smoke_receipt_sha256="c" * 64,
        )


def test_raw_pies_training_stays_blocked_without_actor_delta_proprio():
    with pytest.raises(ValueError, match="delta-proprio"):
        build_ray_event_deployment_receipt(
            history_length=1,
            spatial_size=(16, 96),
            source="livox_per_return",
            temporal_baseline="per_return_age",
            history_reduction="pies_latest_event_k1",
            packetization_invariance_proof_sha256=(
                "e8d461a7b4e0d14bfa00bc6bfd95cd6e85a0c0f79f9a87b2007b838fb0b8c332"
            ),
            training_ready=True,
            smoke_receipt_sha256="d" * 64,
            self_return_filter="upstream_static_mask",
            self_return_filter_config_sha256="e" * 64,
            self_return_filtered_count=4,
        )


def test_raster_latest_prototype_can_never_be_training_ready():
    with pytest.raises(ValueError, match="can never"):
        build_ray_event_deployment_receipt(
            history_length=1,
            spatial_size=(16, 96),
            source="raycaster_quantized_event",
            temporal_baseline="quantized_event_age",
            history_reduction="raster_latest_event_prototype",
            packet_time_quantization_upper_bound_s=0.1,
            training_ready=True,
            smoke_receipt_sha256="f" * 64,
            self_return_filter="upstream_static_mask",
            self_return_filter_config_sha256="0" * 64,
            self_return_filtered_count=0,
        )
