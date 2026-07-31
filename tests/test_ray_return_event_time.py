from __future__ import annotations

import copy

import pytest
import torch

from rsl_rl.modules.ray_return_event_time import (
    SUPPORTED_EVENT_TIME_MODES,
    RayReturnEventTimeEncoder,
)


def _inputs(
    *,
    batch_size: int = 2,
    history_length: int = 3,
    spatial_size: tuple[int, int] = (4, 8),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return_valid = torch.zeros(
        batch_size,
        history_length,
        *spatial_size,
        dtype=torch.bool,
    )
    return_valid[:, :, 0, 0] = True
    return_valid[:, :, 1, 1] = True
    return_valid[:, :, 3, 7] = True
    packet_age = torch.tensor(
        [[0.30, 0.20, 0.10], [0.35, 0.25, 0.15]],
        dtype=torch.float32,
    )[:batch_size, :history_length]
    frame_valid = torch.ones(
        batch_size,
        history_length,
        dtype=torch.bool,
    )
    return_age = torch.zeros_like(return_valid, dtype=torch.float32)
    return_age[return_valid] = (
        packet_age[:, :, None, None].expand_as(return_age)[return_valid]
        + torch.linspace(
            0.001,
            0.030,
            int(return_valid.sum()),
        )
    )
    return return_valid, packet_age, frame_valid, return_age


def _encoder(
    *,
    mode: str = "per_return_age",
    motion_delta_dim: int = 0,
) -> RayReturnEventTimeEncoder:
    torch.manual_seed(7)
    return RayReturnEventTimeEncoder(
        history_length=3,
        input_spatial_size=(4, 8),
        token_spatial_size=(2, 2),
        token_dim=12,
        mode=mode,
        age_time_scale_s=0.5,
        num_fourier_frequencies=3,
        hidden_dim=16,
        motion_delta_dim=motion_delta_dim,
        motion_hidden_dim=9,
    )


@pytest.mark.parametrize("mode", SUPPORTED_EVENT_TIME_MODES)
def test_modes_share_shape_and_blackout_is_exact_zero(mode: str) -> None:
    encoder = _encoder(mode=mode).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    output, diagnostics = encoder.forward_with_diagnostics(
        return_valid,
        packet_age,
        frame_valid,
        return_age,
    )

    assert output.shape == (2, 3, 4, 12)
    assert diagnostics["token_valid"].shape == (2, 3, 4)
    assert torch.isfinite(output).all()

    blackout = torch.zeros_like(return_valid)
    blackout_age = torch.zeros_like(return_age)
    blackout_output, blackout_diagnostics = encoder.forward_with_diagnostics(
        blackout,
        packet_age,
        frame_valid,
        blackout_age,
    )
    assert torch.equal(blackout_output, torch.zeros_like(blackout_output))
    assert not blackout_diagnostics["token_valid"].any()
    assert torch.equal(
        blackout_diagnostics["return_count"],
        torch.zeros_like(blackout_diagnostics["return_count"]),
    )


@pytest.mark.parametrize("mode", SUPPORTED_EVENT_TIME_MODES)
def test_compiled_paths_mask_residual_returns_from_invalid_frames(
    mode: str,
) -> None:
    encoder = _encoder(mode=mode).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    frame_valid[0, 1] = False
    packet_age[0, 1] = 0.0
    return_valid[0, 1] = True
    return_age[0, 1] = 9.0

    scripted = torch.jit.script(copy.deepcopy(encoder))
    exported = torch.export.export(
        copy.deepcopy(encoder),
        (return_valid, packet_age, frame_valid, return_age),
    ).module()
    with torch.no_grad():
        scripted_output = scripted(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        exported_output = exported(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )

    assert torch.equal(
        scripted_output[0, 1],
        torch.zeros_like(scripted_output[0, 1]),
    )
    assert torch.equal(
        exported_output[0, 1],
        torch.zeros_like(exported_output[0, 1]),
    )


@pytest.mark.parametrize("bad_value", (float("nan"), -0.5))
@pytest.mark.parametrize("mode", SUPPORTED_EVENT_TIME_MODES)
def test_compiled_paths_fail_safe_bad_age_to_unknown(
    mode: str,
    bad_value: float,
) -> None:
    encoder = _encoder(mode=mode).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    if mode == "per_return_age":
        frame_returns = return_valid[0, 1]
        return_age[0, 1][frame_returns] = bad_value
    else:
        packet_age[0, 1] = bad_value

    scripted = torch.jit.script(copy.deepcopy(encoder))
    exported = torch.export.export(
        copy.deepcopy(encoder),
        (return_valid, packet_age, frame_valid, return_age),
    ).module()
    with torch.no_grad():
        scripted_output = scripted(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        exported_output = exported(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )

    assert torch.isfinite(scripted_output).all()
    assert torch.isfinite(exported_output).all()
    assert torch.equal(
        scripted_output[0, 1],
        torch.zeros_like(scripted_output[0, 1]),
    )
    assert torch.equal(
        exported_output[0, 1],
        torch.zeros_like(exported_output[0, 1]),
    )


def test_compiled_paths_fail_safe_return_younger_than_packet() -> None:
    encoder = _encoder(mode="per_return_age").eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    frame_returns = return_valid[0, 1]
    return_age[0, 1][frame_returns] = packet_age[0, 1] - 0.01

    scripted = torch.jit.script(copy.deepcopy(encoder))
    exported = torch.export.export(
        copy.deepcopy(encoder),
        (return_valid, packet_age, frame_valid, return_age),
    ).module()
    with torch.no_grad():
        scripted_output = scripted(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        exported_output = exported(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )

    assert torch.isfinite(scripted_output).all()
    assert torch.isfinite(exported_output).all()
    assert torch.equal(
        scripted_output[0, 1],
        torch.zeros_like(scripted_output[0, 1]),
    )
    assert torch.equal(
        exported_output[0, 1],
        torch.zeros_like(exported_output[0, 1]),
    )


def test_per_return_summaries_preserve_within_packet_spread() -> None:
    encoder = _encoder(mode="per_return_age").eval()
    return_valid = torch.zeros(1, 3, 4, 8, dtype=torch.bool)
    return_valid[0, 2, 0, 0] = True
    return_valid[0, 2, 0, 1] = True
    packet_age = torch.tensor([[0.3, 0.2, 0.1]])
    frame_valid = torch.ones(1, 3, dtype=torch.bool)
    return_age = torch.zeros(1, 3, 4, 8)
    return_age[0, 2, 0, 0] = 0.11
    return_age[0, 2, 0, 1] = 0.17

    _, diagnostics = encoder.forward_with_diagnostics(
        return_valid,
        packet_age,
        frame_valid,
        return_age,
    )

    assert diagnostics["return_count"][0, 2, 0].item() == 2
    assert diagnostics["valid_fraction"][0, 2, 0].item() == 0.25
    assert diagnostics["mean_age_s"][0, 2, 0].item() == pytest.approx(
        0.14
    )
    assert diagnostics["min_age_s"][0, 2, 0].item() == pytest.approx(
        0.11
    )
    assert diagnostics["max_age_s"][0, 2, 0].item() == pytest.approx(
        0.17
    )
    assert diagnostics["age_span_s"][0, 2, 0].item() == pytest.approx(
        0.06
    )


def test_per_return_encoding_preserves_within_token_angular_assignment() -> None:
    encoder = _encoder(mode="per_return_age").eval()
    return_valid = torch.zeros(1, 3, 4, 8, dtype=torch.bool)
    return_valid[0, 2, 0, 0] = True
    return_valid[0, 2, 0, 1] = True
    packet_age = torch.tensor([[0.3, 0.2, 0.1]])
    frame_valid = torch.ones(1, 3, dtype=torch.bool)
    first_age = torch.zeros(1, 3, 4, 8)
    first_age[0, 2, 0, 0] = 0.11
    first_age[0, 2, 0, 1] = 0.19
    swapped_age = first_age.clone()
    swapped_age[0, 2, 0, 0] = 0.19
    swapped_age[0, 2, 0, 1] = 0.11

    with torch.no_grad():
        first, first_diagnostics = encoder.forward_with_diagnostics(
            return_valid,
            packet_age,
            frame_valid,
            first_age,
        )
        swapped, swapped_diagnostics = encoder.forward_with_diagnostics(
            return_valid,
            packet_age,
            frame_valid,
            swapped_age,
        )

    for key in (
        "return_count",
        "valid_fraction",
        "mean_age_s",
        "min_age_s",
        "max_age_s",
        "age_span_s",
    ):
        torch.testing.assert_close(
            first_diagnostics[key],
            swapped_diagnostics[key],
            rtol=0.0,
            atol=0.0,
        )
    assert not torch.equal(first[0, 2, 0], swapped[0, 2, 0])


def test_packet_age_control_discards_within_packet_return_timing() -> None:
    encoder = _encoder(mode="packet_age").eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    changed_return_age = return_age.clone()
    changed_return_age[return_valid] += 5.0

    with torch.no_grad():
        reference = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        changed = encoder(
            return_valid,
            packet_age,
            frame_valid,
            changed_return_age,
        )
    torch.testing.assert_close(reference, changed, rtol=0.0, atol=0.0)


def test_index_only_discards_all_measured_age_values() -> None:
    encoder = _encoder(mode="index_only").eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    changed_packet_age = packet_age + 10.0
    changed_return_age = return_age.clone()
    changed_return_age[return_valid] += 20.0

    with torch.no_grad():
        reference = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        changed = encoder(
            return_valid,
            changed_packet_age,
            frame_valid,
            changed_return_age,
        )
    torch.testing.assert_close(reference, changed, rtol=0.0, atol=0.0)


def test_per_return_mode_is_sensitive_to_intra_packet_timing() -> None:
    encoder = _encoder(mode="per_return_age").eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    changed_return_age = return_age.clone()
    changed_return_age[return_valid] += 0.07

    with torch.no_grad():
        reference = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        changed = encoder(
            return_valid,
            packet_age,
            frame_valid,
            changed_return_age,
        )
    assert not torch.equal(reference, changed)


def test_packet_age_mode_is_sensitive_to_packet_latency() -> None:
    encoder = _encoder(mode="packet_age").eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    changed_packet_age = packet_age + 0.08
    changed_return_age = return_age.clone()
    changed_return_age[return_valid] += 0.08

    with torch.no_grad():
        reference = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        changed = encoder(
            return_valid,
            changed_packet_age,
            frame_valid,
            changed_return_age,
        )
    assert not torch.equal(reference, changed)


def test_index_only_encodes_oldest_to_newest_slot_order() -> None:
    encoder = _encoder(mode="index_only").eval()
    return_valid = torch.ones(1, 3, 4, 8, dtype=torch.bool)
    packet_age = torch.zeros(1, 3)
    frame_valid = torch.ones(1, 3, dtype=torch.bool)

    output, diagnostics = encoder.forward_with_diagnostics(
        return_valid,
        packet_age,
        frame_valid,
    )

    assert diagnostics["slot_coordinate"][0, 0, 0].item() == 1.0
    assert diagnostics["slot_coordinate"][0, 1, 0].item() == 0.5
    assert diagnostics["slot_coordinate"][0, 2, 0].item() == 0.0
    assert torch.equal(
        diagnostics["mean_age_s"],
        torch.zeros_like(diagnostics["mean_age_s"]),
    )
    assert not diagnostics["time_value_is_seconds"].any()
    assert diagnostics["event_time_mode_id"].tolist() == [0]
    assert not torch.equal(output[:, 0], output[:, 2])


def test_motion_delta_branch_is_independently_switchable() -> None:
    encoder = _encoder(
        mode="packet_age",
        motion_delta_dim=6,
    ).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    motion = torch.zeros(2, 3, 6)
    changed_motion = motion.clone()
    changed_motion[:, 0, 0] = 0.4

    with torch.no_grad():
        reference = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            motion,
        )
        changed = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            changed_motion,
        )
    assert not torch.equal(reference[:, 0], changed[:, 0])
    torch.testing.assert_close(
        reference[:, 1:],
        changed[:, 1:],
        rtol=0.0,
        atol=0.0,
    )


def test_motion_blackout_and_compiled_nonfinite_are_exact_zero() -> None:
    encoder = _encoder(
        mode="per_return_age",
        motion_delta_dim=6,
    ).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    motion = torch.randn(2, 3, 6)
    blackout = torch.zeros_like(return_valid)
    blackout_age = torch.zeros_like(return_age)
    with torch.no_grad():
        blackout_output = encoder(
            blackout,
            packet_age,
            frame_valid,
            blackout_age,
            motion,
        )
    assert torch.equal(
        blackout_output,
        torch.zeros_like(blackout_output),
    )

    motion[0, 1, 0] = torch.nan
    scripted = torch.jit.script(copy.deepcopy(encoder))
    exported = torch.export.export(
        copy.deepcopy(encoder),
        (return_valid, packet_age, frame_valid, return_age, motion),
    ).module()
    with torch.no_grad():
        scripted_output = scripted(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            motion,
        )
        exported_output = exported(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            motion,
        )
    assert torch.isfinite(scripted_output).all()
    assert torch.isfinite(exported_output).all()
    assert torch.equal(
        scripted_output[0, 1],
        torch.zeros_like(scripted_output[0, 1]),
    )
    assert torch.equal(
        exported_output[0, 1],
        torch.zeros_like(exported_output[0, 1]),
    )


def test_backward_reaches_time_and_motion_projections() -> None:
    encoder = _encoder(
        mode="per_return_age",
        motion_delta_dim=6,
    ).train()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    return_age.requires_grad_()
    motion = torch.randn(2, 3, 6, requires_grad=True)

    output = encoder(
        return_valid,
        packet_age,
        frame_valid,
        return_age,
        motion,
    )
    output.square().mean().backward()

    assert return_age.grad is not None
    assert torch.isfinite(return_age.grad).all()
    assert float(return_age.grad[return_valid].abs().sum()) > 0.0
    assert torch.equal(
        return_age.grad[~return_valid],
        torch.zeros_like(return_age.grad[~return_valid]),
    )
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all()
    assert float(motion.grad.abs().sum()) > 0.0
    assert encoder.ray_time_projection.weight.grad is not None
    assert (
        float(
            encoder.ray_time_projection.weight.grad.abs().sum()
        )
        > 0.0
    )
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_scripted_grad_path_masks_invalid_frame_age_gradient() -> None:
    encoder = _encoder(mode="per_return_age").train()
    scripted = torch.jit.script(copy.deepcopy(encoder))
    return_valid, packet_age, frame_valid, return_age = _inputs()
    frame_valid[0, 0] = False
    packet_age[0, 0] = 0.0
    return_valid[0, 0] = True
    return_age[0, 0] = 9.0
    return_age.requires_grad_()

    output = scripted(
        return_valid,
        packet_age,
        frame_valid,
        return_age,
    )
    output.square().mean().backward()

    assert return_age.grad is not None
    assert torch.isfinite(return_age.grad).all()
    assert torch.equal(
        return_age.grad[0, 0],
        torch.zeros_like(return_age.grad[0, 0]),
    )
    assert float(return_age.grad[1].abs().sum()) > 0.0


def test_state_dict_round_trip_is_exact() -> None:
    encoder = _encoder(mode="per_return_age").eval()
    restored = _encoder(mode="per_return_age").eval()
    restored.load_state_dict(copy.deepcopy(encoder.state_dict()))
    return_valid, packet_age, frame_valid, return_age = _inputs()

    with torch.no_grad():
        reference = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        replay = restored(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
    torch.testing.assert_close(reference, replay, rtol=0.0, atol=0.0)


def test_modes_have_identical_parameter_schema_and_cross_load_state() -> None:
    encoders = {
        mode: _encoder(mode=mode)
        for mode in SUPPORTED_EVENT_TIME_MODES
    }
    schemas = {
        mode: {
            name: tuple(tensor.shape)
            for name, tensor in encoder.state_dict().items()
        }
        for mode, encoder in encoders.items()
    }
    assert schemas["index_only"] == schemas["packet_age"]
    assert schemas["index_only"] == schemas["per_return_age"]
    reference_state = copy.deepcopy(
        encoders["per_return_age"].state_dict()
    )
    for encoder in encoders.values():
        encoder.load_state_dict(reference_state, strict=True)


@pytest.mark.parametrize("mode", SUPPORTED_EVENT_TIME_MODES)
def test_torchscript_matches_eager(mode: str) -> None:
    encoder = _encoder(mode=mode).eval()
    scripted = torch.jit.script(copy.deepcopy(encoder))
    return_valid, packet_age, frame_valid, return_age = _inputs()
    with torch.no_grad():
        eager = encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
        compiled = scripted(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )
    torch.testing.assert_close(eager, compiled, rtol=0.0, atol=0.0)


def test_per_return_mode_exports_to_onnx() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    encoder = _encoder(mode="per_return_age").eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()

    exported = torch.onnx.export(
        encoder,
        (return_valid, packet_age, frame_valid, return_age),
        f=None,
        input_names=[
            "return_valid",
            "packet_age_s",
            "frame_valid",
            "return_age_s",
        ],
        output_names=["event_time"],
        dynamo=True,
        dynamic_axes={
            "return_valid": {0: "batch_size"},
            "packet_age_s": {0: "batch_size"},
            "frame_valid": {0: "batch_size"},
            "return_age_s": {0: "batch_size"},
            "event_time": {0: "batch_size"},
        },
    )

    assert exported.model_proto.graph.output[0].name == "event_time"
    assert (
        exported.model_proto.graph.output[0]
        .type.tensor_type.shape.dim[0]
        .dim_param
        == "batch_size"
    )
    assert len(exported.model_proto.SerializeToString()) > 0


def test_motion_enabled_mode_exports_dynamic_batch_to_onnx() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    encoder = _encoder(
        mode="per_return_age",
        motion_delta_dim=6,
    ).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    motion = torch.randn(2, 3, 6)

    exported = torch.onnx.export(
        encoder,
        (return_valid, packet_age, frame_valid, return_age, motion),
        f=None,
        input_names=[
            "return_valid",
            "packet_age_s",
            "frame_valid",
            "return_age_s",
            "frame_motion_delta",
        ],
        output_names=["event_time"],
        dynamo=True,
        dynamic_axes={
            "return_valid": {0: "batch_size"},
            "packet_age_s": {0: "batch_size"},
            "frame_valid": {0: "batch_size"},
            "return_age_s": {0: "batch_size"},
            "frame_motion_delta": {0: "batch_size"},
            "event_time": {0: "batch_size"},
        },
    )

    assert (
        exported.model_proto.graph.output[0]
        .type.tensor_type.shape.dim[0]
        .dim_param
        == "batch_size"
    )
    assert len(exported.model_proto.SerializeToString()) > 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_float_dtype_and_module_conversion(dtype: torch.dtype) -> None:
    encoder = _encoder(mode="per_return_age").to(dtype=dtype).eval()
    return_valid, packet_age, frame_valid, return_age = _inputs()
    output = encoder(
        return_valid,
        packet_age.to(dtype),
        frame_valid,
        return_age.to(dtype),
    )
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("return_dtype", "return_valid must be boolean"),
        ("packet_shape", "packet_age_s must have shape"),
        ("frame_shape", "frame_valid must have shape"),
        ("return_age_shape", "return_age_s must match"),
        ("return_nan", "return_age_s must contain only finite"),
        ("return_negative", "return_age_s must be non-negative"),
        ("unknown_age", "without a successful return"),
        ("younger_than_packet", "cannot be younger"),
        ("invalid_frame_age", "Invalid history frames"),
        ("invalid_frame_return", "cannot belong to an invalid frame"),
    ],
)
def test_invalid_event_time_contracts_fail_closed(
    mutation: str,
    message: str,
) -> None:
    encoder = _encoder(mode="per_return_age")
    return_valid, packet_age, frame_valid, return_age = _inputs()
    if mutation == "return_dtype":
        return_valid = return_valid.float()
    elif mutation == "packet_shape":
        packet_age = packet_age[:, :-1]
    elif mutation == "frame_shape":
        frame_valid = frame_valid[:, :-1]
    elif mutation == "return_age_shape":
        return_age = return_age[..., :-1]
    elif mutation == "return_nan":
        return_age[0, 0, 0, 0] = torch.nan
    elif mutation == "return_negative":
        return_age[0, 0, 0, 0] = -1.0
    elif mutation == "unknown_age":
        return_age[0, 0, 2, 2] = 0.3
    elif mutation == "younger_than_packet":
        return_age[0, 0, 0, 0] = packet_age[0, 0] - 0.1
    elif mutation == "invalid_frame_age":
        frame_valid[0, 0] = False
    elif mutation == "invalid_frame_return":
        frame_valid[0, 0] = False
        packet_age[0, 0] = 0.0
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
        )


def test_invalid_frame_requires_zero_motion_and_packet_age() -> None:
    encoder = _encoder(
        mode="packet_age",
        motion_delta_dim=6,
    )
    return_valid, packet_age, frame_valid, return_age = _inputs()
    return_valid[0, 0] = False
    frame_valid[0, 0] = False
    motion = torch.zeros(2, 3, 6)
    motion[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="zero packet age"):
        encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            motion,
        )
    packet_age[0, 0] = 0.0
    return_age[0, 0] = 0.0
    with pytest.raises(ValueError, match="zero motion"):
        encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            motion,
        )


def test_motion_input_is_rejected_when_branch_is_absent() -> None:
    encoder = _encoder(mode="packet_age", motion_delta_dim=0)
    return_valid, packet_age, frame_valid, return_age = _inputs()
    with pytest.raises(ValueError, match="only valid"):
        encoder(
            return_valid,
            packet_age,
            frame_valid,
            return_age,
            torch.zeros(2, 3, 6),
        )


def test_per_return_mode_requires_return_age() -> None:
    encoder = _encoder(mode="per_return_age")
    return_valid, packet_age, frame_valid, _ = _inputs()
    with pytest.raises(ValueError, match="requires return_age_s"):
        encoder(return_valid, packet_age, frame_valid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"history_length": 0}, "history_length"),
        ({"input_spatial_size": (4, 7)}, "divisible"),
        ({"token_spatial_size": (0, 2)}, "positive"),
        ({"token_dim": 0}, "token_dim"),
        ({"mode": "future"}, "mode must be"),
        ({"age_time_scale_s": 0.0}, "age_time_scale_s"),
        ({"num_fourier_frequencies": 0}, "num_fourier"),
        ({"motion_delta_dim": -1}, "motion_delta_dim"),
    ],
)
def test_constructor_contracts(
    kwargs: dict[str, object],
    message: str,
) -> None:
    base: dict[str, object] = {
        "history_length": 3,
        "input_spatial_size": (4, 8),
        "token_spatial_size": (2, 2),
        "token_dim": 12,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RayReturnEventTimeEncoder(**base)
