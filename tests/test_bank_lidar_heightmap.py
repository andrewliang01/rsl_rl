# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import inspect

import pytest
import torch
import torch.nn.functional as F

from rsl_rl.modules.bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    SphericalAutoencoderPretrainHead,
    create_frozen_reconstructor_checkpoint,
    freeze_reconstructor,
    load_frozen_reconstructor_checkpoint,
    preflight_validate_lidar_history,
    reconstructor_checkpoint_schema,
    spherical_valid_bce,
    supervised_height_valid_mse,
    valid_masked_range_mse,
)
from rsl_rl.modules.ray_time_attention_encoder import CircularAzimuthConv2d


def _history(
    history_length: int,
    *,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    torch.manual_seed(4100 + history_length + batch_size)
    range_m = torch.rand(batch_size, history_length, 16, 96) * 5.8 + 0.1
    valid = torch.rand(batch_size, history_length, 16, 96) > 0.3
    range_m = torch.where(valid, range_m, torch.zeros_like(range_m))
    return torch.stack((range_m, valid.to(range_m.dtype)), dim=2).to(dtype)


@pytest.mark.parametrize("history_length", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_strict_input_and_output_contract(
    history_length: int,
    dtype: torch.dtype,
) -> None:
    model = BankLidarHeightmapReconstructor(
        history_length=history_length
    ).eval()
    output = model(_history(history_length, batch_size=3, dtype=dtype))

    assert output.shape == (3, 1, 28, 20)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    schema = reconstructor_checkpoint_schema(model)
    assert schema["input_tail"] == [history_length, 2, 16, 96]
    assert schema["output_tail"] == [1, 28, 20]
    assert schema["internal_dtype"] == "torch.float32"
    assert schema["deploy_validation_mode"] == "trusted_no_sync"
    assert schema["strict_preflight_helper"] == (
        "preflight_validate_lidar_history"
    )
    assert schema["source_contract"]["valid_channel"] == (
        "finite exact binary {0,1}"
    )
    assert schema["offline_losses_in_deploy_forward"] is False
    assert schema["deploy_inputs"] == ["ray_history"]
    assert schema["fixed_window_zero_hidden"] is True
    assert schema["persistent_recurrent_state"] is False
    assert schema["shared_frame_encoder"] is True
    assert schema["previous_prediction_feedback"] is False
    assert schema["odometry_input"] is False
    assert schema["proprioception_input"] is False
    assert schema["critic_input"] is False
    assert schema["pretrain_head_in_deploy_checkpoint"] is False
    assert schema["height_unit"] == "metre"


def test_k1_k5_change_only_window_contract_not_parameter_architecture() -> None:
    k1 = BankLidarHeightmapReconstructor(history_length=1)
    k5 = BankLidarHeightmapReconstructor(history_length=5)
    k1_state = k1.state_dict()
    k5_state = k5.state_dict()

    assert set(k1_state) == set(k5_state)
    assert {
        name: (tuple(value.shape), value.dtype)
        for name, value in k1_state.items()
    } == {
        name: (tuple(value.shape), value.dtype)
        for name, value in k5_state.items()
    }
    assert sum(parameter.numel() for parameter in k1.parameters()) == sum(
        parameter.numel() for parameter in k5.parameters()
    )
    k5.load_state_dict(k1_state, strict=True)

    with pytest.raises(ValueError, match="exactly 1 or 5"):
        BankLidarHeightmapReconstructor(history_length=3)
    with pytest.raises(ValueError, match="exact shape"):
        k5(torch.zeros(2, 1, 2, 16, 96))
    with pytest.raises(TypeError, match="float16 or torch.float32"):
        k1(torch.zeros(2, 1, 2, 16, 96, dtype=torch.float64))


def test_unknown_range_is_sanitized_without_value_or_gradient_leakage() -> None:
    torch.manual_seed(4201)
    model = BankLidarHeightmapReconstructor(history_length=5).eval()
    reference_input = _history(5).requires_grad_()
    unknown = reference_input[:, :, 1] == 0.0
    changed_input = reference_input.detach().clone()
    changed_range = changed_input[:, :, 0]
    unknown_ids = unknown.nonzero(as_tuple=True)
    changed_range[unknown_ids] = float("nan")
    changed_range[
        unknown_ids[0][::2],
        unknown_ids[1][::2],
        unknown_ids[2][::2],
        unknown_ids[3][::2],
    ] = float("inf")

    reference = model(reference_input)
    preflight = preflight_validate_lidar_history(
        changed_input,
        history_length=5,
    )
    changed = model(changed_input)
    torch.testing.assert_close(changed, reference, rtol=0.0, atol=0.0)
    assert preflight["validation_mode"] == "strict_content_preflight_sync"
    assert preflight["deploy_validation_mode"] == "trusted_no_sync"
    assert preflight["unknown_nonfinite_range_count"] > 0

    reference.sum().backward()
    assert reference_input.grad is not None
    assert torch.count_nonzero(reference_input.grad[:, :, 0][unknown]) == 0

    invalid_valid_range = changed_input.clone()
    valid_id = (invalid_valid_range[:, :, 1] == 1.0).nonzero()[0]
    invalid_valid_range[
        valid_id[0], valid_id[1], 0, valid_id[2], valid_id[3]
    ] = float("nan")
    with pytest.raises(ValueError, match="valid LiDAR return"):
        preflight_validate_lidar_history(
            invalid_valid_range,
            history_length=5,
        )

    invalid_mask = changed_input.clone()
    invalid_mask[0, 0, 1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="valid channel must be finite"):
        preflight_validate_lidar_history(invalid_mask, history_length=5)


@pytest.mark.parametrize("bad_range", [0.0, -0.1, 6.1, float("inf")])
def test_valid_return_and_binary_mask_contract_fails_closed(
    bad_range: float,
) -> None:
    model = BankLidarHeightmapReconstructor(history_length=1).eval()
    history = _history(1)
    history[0, 0, 1, 0, 0] = 1.0
    history[0, 0, 0, 0, 0] = bad_range
    with pytest.raises(ValueError, match="valid LiDAR return"):
        preflight_validate_lidar_history(history, history_length=1)

    non_binary = _history(1)
    non_binary[0, 0, 1, 0, 0] = 0.25
    with pytest.raises(ValueError, match="exactly binary"):
        preflight_validate_lidar_history(non_binary, history_length=1)


@pytest.mark.parametrize("history_length", [1, 5])
def test_deploy_forward_never_scalarizes_or_copies_to_host(
    history_length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BankLidarHeightmapReconstructor(
        history_length=history_length
    ).eval()
    history = _history(history_length, dtype=torch.float16)
    preflight_validate_lidar_history(
        history,
        history_length=history_length,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Deploy forward attempted tensor scalarization or host copy."
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        output = model(history)

    assert output.shape == (2, 1, 28, 20)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_every_spherical_path_uses_circular_azimuth_and_wraps_seam() -> None:
    model = BankLidarHeightmapReconstructor(history_length=1)
    pretrain_head = SphericalAutoencoderPretrainHead()
    frame_convs = [block.conv for block in model.frame_encoder.blocks]
    decoder_convs = [block.conv for block in pretrain_head.blocks]
    assert all(
        isinstance(layer, CircularAzimuthConv2d)
        for layer in (*frame_convs, *decoder_convs, pretrain_head.output_conv)
    )

    first = frame_convs[0]
    with torch.no_grad():
        first.conv.weight.zero_()
        # Output azimuth zero reads the wrapped input column -1.
        first.conv.weight[0, 0, 1, 1] = 1.0
    frame = torch.zeros(1, 2, 16, 96)
    frame[0, 0, 0, -1] = 2.5
    convolved = first(frame)
    torch.testing.assert_close(convolved[0, 0, 0, 0], torch.tensor(2.5))


def test_gru_receives_fresh_zero_hidden_and_forward_has_no_persistent_state() -> None:
    torch.manual_seed(4301)
    model = BankLidarHeightmapReconstructor(history_length=5).train()
    history = _history(5)
    captured_hidden: list[torch.Tensor] = []

    def capture_hidden(_module, arguments):
        captured_hidden.append(arguments[1].detach().clone())

    hook = model.temporal_gru.register_forward_pre_hook(capture_hidden)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    first = model(history)
    intervening = history.clone()
    intervening[:, :, 0] *= 0.5
    model(intervening)
    repeated = model(history)
    hook.remove()

    torch.testing.assert_close(repeated, first, rtol=0.0, atol=0.0)
    assert len(captured_hidden) == 3
    assert all(hidden.shape == (1, 2, 128) for hidden in captured_hidden)
    assert all(torch.count_nonzero(hidden) == 0 for hidden in captured_hidden)
    assert list(model.named_buffers()) == []
    assert set(before) == set(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    forbidden = ("previous", "feedback", "odometry", "odom", "proprio", "critic")
    assert not any(
        token in name.lower()
        for name, _ in model.named_modules()
        for token in forbidden
    )


@pytest.mark.parametrize("history_length", [1, 5])
def test_no_skip_spherical_autoencoder_head_and_pretrain_losses(
    history_length: int,
) -> None:
    torch.manual_seed(4400 + history_length)
    model = BankLidarHeightmapReconstructor(history_length=history_length)
    head = SphericalAutoencoderPretrainHead()
    ray_history = _history(history_length)
    encoded = model.encode_frame_history(ray_history)
    reconstruction = head(encoded)

    assert encoded.shape == (2, history_length, 32, 2, 12)
    assert reconstruction.range_m.shape == (2, history_length, 16, 96)
    assert reconstruction.valid_logits.shape == (2, history_length, 16, 96)
    assert reconstruction.range_m.dtype == torch.float32
    assert reconstruction.valid_logits.dtype == torch.float32
    assert head.skip_connections is False
    assert tuple(inspect.signature(head.forward).parameters) == (
        "encoded_frames",
    )

    target_range = ray_history[:, :, 0].float()
    target_valid = ray_history[:, :, 1] == 1.0
    loss = valid_masked_range_mse(
        reconstruction.range_m,
        target_range,
        target_valid,
    ) + spherical_valid_bce(reconstruction.valid_logits, target_valid)
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.frame_encoder.parameters()
    )
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_masked_losses_use_valid_count_and_never_read_unknown_targets() -> None:
    range_prediction = torch.full((1, 1, 16, 96), float("inf"))
    range_target = torch.full((1, 1, 16, 96), float("nan"))
    range_valid = torch.zeros((1, 1, 16, 96), dtype=torch.bool)
    range_prediction[0, 0, 0, 0] = 2.0
    range_target[0, 0, 0, 0] = 1.0
    range_valid[0, 0, 0, 0] = True
    range_prediction[0, 0, 0, 1] = 5.0
    range_target[0, 0, 0, 1] = 2.0
    range_valid[0, 0, 0, 1] = True
    torch.testing.assert_close(
        valid_masked_range_mse(range_prediction, range_target, range_valid),
        torch.tensor(5.0),
    )

    height_prediction = torch.full((1, 1, 28, 20), float("inf"))
    height_target = torch.full((1, 1, 28, 20), float("nan"))
    height_valid = torch.zeros((1, 1, 28, 20), dtype=torch.bool)
    height_prediction[0, 0, 0, 0] = -1.0
    height_target[0, 0, 0, 0] = -2.0
    height_valid[0, 0, 0, 0] = True
    height_prediction[0, 0, 0, 1] = 4.0
    height_target[0, 0, 0, 1] = 1.0
    height_valid[0, 0, 0, 1] = True
    torch.testing.assert_close(
        supervised_height_valid_mse(
            height_prediction,
            height_target,
            height_valid,
        ),
        torch.tensor(5.0),
    )

    zero_prediction = torch.randn(1, 1, 28, 20, requires_grad=True)
    zero_valid = torch.zeros_like(zero_prediction, dtype=torch.bool)
    zero_target = torch.full_like(zero_prediction, float("nan"))
    zero_loss = supervised_height_valid_mse(
        zero_prediction,
        zero_target,
        zero_valid,
    )
    torch.testing.assert_close(zero_loss, torch.tensor(0.0))
    zero_loss.backward()
    assert torch.count_nonzero(zero_prediction.grad) == 0


def test_validity_bce_uses_every_cell_denominator() -> None:
    logits = torch.zeros(1, 1, 16, 96)
    logits[0, 0, 0, 0] = 2.0
    target = torch.zeros_like(logits, dtype=torch.bool)
    target[0, 0, 0, 0] = True
    expected = F.binary_cross_entropy_with_logits(
        logits,
        target.float(),
        reduction="mean",
    )
    torch.testing.assert_close(spherical_valid_bce(logits, target), expected)


def test_deploy_forward_cannot_receive_supervised_target_or_other_modalities() -> None:
    model = BankLidarHeightmapReconstructor(history_length=1).eval()
    assert tuple(inspect.signature(model.forward).parameters) == ("ray_history",)
    history = _history(1)
    target = torch.randn(2, 1, 28, 20)
    target_valid = torch.ones_like(target, dtype=torch.bool)
    deploy_output = model(history)
    supervised_height_valid_mse(deploy_output, target, target_valid)
    changed_target = target + 1000.0
    supervised_height_valid_mse(deploy_output, changed_target, target_valid)
    torch.testing.assert_close(model(history), deploy_output, rtol=0.0, atol=0.0)
    with pytest.raises(TypeError):
        model(history, target)


@pytest.mark.parametrize("history_length", [1, 5])
def test_freeze_audit_checkpoint_and_strict_reload(history_length: int) -> None:
    torch.manual_seed(4500 + history_length)
    model = BankLidarHeightmapReconstructor(history_length=history_length)
    with pytest.raises(ValueError, match="explicitly frozen"):
        create_frozen_reconstructor_checkpoint(model)

    half_model = BankLidarHeightmapReconstructor(
        history_length=history_length
    ).half()
    with pytest.raises(TypeError, match="must remain torch.float32"):
        freeze_reconstructor(half_model)

    audit = freeze_reconstructor(model)
    assert audit["parameter_count"] > 0
    assert audit["trainable_parameter_count"] == 0
    assert audit["training"] is False
    assert len(audit["state_sha256"]) == 64
    assert all(not parameter.requires_grad for parameter in model.parameters())

    checkpoint = create_frozen_reconstructor_checkpoint(model)
    restored = load_frozen_reconstructor_checkpoint(checkpoint)
    history = _history(history_length)
    torch.testing.assert_close(restored(history), model(history), rtol=0.0, atol=0.0)
    assert restored.training is False
    assert all(not parameter.requires_grad for parameter in restored.parameters())

    bad_digest = copy.deepcopy(checkpoint)
    first_name = next(iter(bad_digest["state_dict"]))
    bad_digest["state_dict"][first_name].flatten()[0] += 1.0
    with pytest.raises(ValueError, match="digest mismatch"):
        load_frozen_reconstructor_checkpoint(bad_digest)

    bad_schema = copy.deepcopy(checkpoint)
    bad_schema["schema"]["history_length"] = 5 if history_length == 1 else 1
    with pytest.raises(ValueError, match="schema mismatch"):
        load_frozen_reconstructor_checkpoint(bad_schema)

    missing_state = copy.deepcopy(checkpoint)
    missing_state["state_dict"].pop(next(iter(missing_state["state_dict"])))
    with pytest.raises(ValueError, match="keys mismatch"):
        load_frozen_reconstructor_checkpoint(missing_state)

    extra_top_level = copy.deepcopy(checkpoint)
    extra_top_level["unexpected"] = True
    with pytest.raises(ValueError, match="keys changed"):
        load_frozen_reconstructor_checkpoint(extra_top_level)
