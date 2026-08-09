# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.prop_mlp_elevation_fusion_model import (
    PropMLPElevationFusionModel,
)
from rsl_rl.modules.bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    create_frozen_reconstructor_checkpoint,
    freeze_reconstructor,
)


_AUTO_CHECKPOINT = object()


def _contract(**overrides) -> dict:
    contract = {
        "schema_version": 1,
        "target_definition": "synthetic_test_height_target",
        "height_unit": "metre",
        "height_sign": "synthetic_test_positive_direction",
        "grid_shape": [28, 20],
        "grid_axis_order": ["synthetic_row", "synthetic_column"],
        "grid_axis_directions": [
            "synthetic_row_direction",
            "synthetic_column_direction",
        ],
        "flatten_order": "C_contiguous_row_major",
        "coordinate_frame": "synthetic_test_frame",
        "origin": "synthetic_test_origin",
        "resolution_m": 0.05,
        "unknown_cell_policy": "synthetic_dense_reconstruction",
        "contract_source_sha256": "a" * 64,
    }
    contract.update(overrides)
    return contract


def _ray_history(
    history_length: int,
    *,
    batch_size: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    torch.manual_seed(6100 + history_length + batch_size)
    range_m = torch.rand(batch_size, history_length, 16, 96) * 5.8 + 0.1
    valid = torch.rand_like(range_m) > 0.25
    range_m = torch.where(valid, range_m, torch.zeros_like(range_m))
    return torch.stack((range_m, valid.to(range_m.dtype)), dim=2).to(dtype)


def _observations(
    history_length: int,
    *,
    batch_size: int = 3,
    dtype: torch.dtype = torch.float32,
) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 96),
            "ray_policy": _ray_history(
                history_length,
                batch_size=batch_size,
                dtype=dtype,
            ),
        },
        batch_size=[batch_size],
    )


def _model(
    observations: TensorDict,
    *,
    history_length: int,
    target_contract: dict | None = None,
    downstream_contract: dict | None = None,
    checkpoint: dict | None | object = _AUTO_CHECKPOINT,
) -> PropMLPElevationFusionModel:
    target_contract = _contract() if target_contract is None else target_contract
    downstream_contract = copy.deepcopy(target_contract) if downstream_contract is None else downstream_contract
    if checkpoint is _AUTO_CHECKPOINT:
        reconstructor = BankLidarHeightmapReconstructor(
            history_length=history_length,
            target_contract=target_contract,
        )
        freeze_reconstructor(reconstructor)
        checkpoint = create_frozen_reconstructor_checkpoint(reconstructor)
    return PropMLPElevationFusionModel(
        obs=observations,
        obs_groups={"actor": ["policy", "ray_policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=[64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        elevation_encoder_type="bank_lidar_heightmap",
        ray_time_set="ray_policy",
        ray_time_history_length=history_length,
        ray_time_spatial_size=(16, 96),
        vision_feature_dim=32,
        prop_feature_dim=32,
        prop_hidden_dims=[64],
        bank_heightmap_target_contract=target_contract,
        bank_downstream_heightmap_contract=downstream_contract,
        _bank_reconstructor_checkpoint_for_testing=checkpoint,
    )


@pytest.mark.parametrize("history_length", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_explicit_bank_branch_preserves_ray_layout_and_forward_contract(
    history_length: int,
    dtype: torch.dtype,
) -> None:
    observations = _observations(history_length, dtype=dtype)
    model = _model(observations, history_length=history_length).eval()
    audit = model.preflight_bank_heightmap_observation(observations)
    output = model(observations)

    assert output.shape == (3, 29)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert model.elevation_encoder_type == "bank_lidar_heightmap"
    assert model.elevation_set == "ray_policy"
    assert model.elevation_history_length == history_length
    assert model.vision_spatial_size == (16, 96)
    assert model.bank_heightmap_spatial_size == (28, 20)
    assert audit["actor_observation_contract"] == {
        "layout": "B_K_C_H_W",
        "flatten_order": "C_contiguous_row_major_K_C_H_W",
        "channels": ["range_m", "valid"],
        "range_unit": "metre",
        "valid_semantics": "finite exact binary {0,1}",
        "history_order": "oldest_to_newest",
        "spatial_shape": [16, 96],
    }
    assert audit["heightmap_contract"] == _contract()

    ray = observations["ray_policy"]
    flat = ray.flatten(start_dim=1)
    reconstructed_layout = flat.reshape(
        -1,
        history_length,
        2,
        16,
        96,
    )
    torch.testing.assert_close(reconstructed_layout, ray, rtol=0.0, atol=0.0)
    remapped = observations.clone()
    remapped["ray_policy"] = reconstructed_layout
    torch.testing.assert_close(model(remapped), output, rtol=0.0, atol=0.0)


def test_bank_branch_fails_closed_without_static_height_contract_evidence() -> None:
    observations = _observations(5)
    with pytest.raises(ValueError, match="fails closed"):
        PropMLPElevationFusionModel(
            obs=observations,
            obs_groups={"actor": ["policy", "ray_policy"]},
            obs_set="actor",
            output_dim=29,
            elevation_encoder_type="bank_lidar_heightmap",
            ray_time_set="ray_policy",
            ray_time_history_length=5,
            ray_time_spatial_size=(16, 96),
        )

    with pytest.raises(ValueError, match="strict frozen reconstructor"):
        _model(
            observations,
            history_length=5,
            checkpoint=None,
        )

    lab_style = _contract(
        target_definition="base_z_minus_hit_z",
        height_sign="positive_when_hit_is_below_base",
    )
    opposite_sign = _contract(
        target_definition="hit_z_minus_base_z",
        height_sign="positive_when_hit_is_above_base",
    )
    with pytest.raises(ValueError, match="contracts differ"):
        _model(
            observations,
            history_length=5,
            target_contract=lab_style,
            downstream_contract=opposite_sign,
        )

    incompatible_grid = _contract(grid_shape=[25, 17])
    with pytest.raises(ValueError, match="grid_shape"):
        _model(
            observations,
            history_length=5,
            target_contract=incompatible_grid,
            downstream_contract=incompatible_grid,
        )


@pytest.mark.parametrize("history_length", [1, 5])
def test_frozen_bank_branch_backpropagates_only_to_inputs_and_downstream(
    history_length: int,
) -> None:
    torch.manual_seed(6200 + history_length)
    observations = _observations(history_length, dtype=torch.float32)
    ray = observations["ray_policy"].detach().clone().requires_grad_()
    proprio = observations["policy"].detach().clone().requires_grad_()
    observations["ray_policy"] = ray
    observations["policy"] = proprio
    model = _model(observations, history_length=history_length).train()

    model(observations).square().mean().backward()

    assert ray.grad is not None
    assert proprio.grad is not None
    assert torch.isfinite(ray.grad).all()
    assert torch.isfinite(proprio.grad).all()
    for frame_index in range(history_length):
        assert torch.count_nonzero(ray.grad[:, frame_index, 0]) > 0
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in model.heightmap_reconstructor.parameters()
    )
    assert all(parameter.grad is not None for parameter in model.elevation_encoder.parameters())


@pytest.mark.parametrize("history_length", [1, 5])
def test_full_actor_forward_has_no_tensor_scalarization_or_host_copy(
    history_length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = _observations(history_length, dtype=torch.float16)
    model = _model(observations, history_length=history_length).eval()
    model.preflight_bank_heightmap_observation(observations)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Actor collection forward synchronized or copied.")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        output = model(observations)

    assert output.shape == (3, 29)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("history_length", [1, 5])
def test_frozen_reconstructor_checkpoint_and_parameter_audit(
    history_length: int,
) -> None:
    contract = _contract()
    reconstructor = BankLidarHeightmapReconstructor(
        history_length=history_length,
        target_contract=contract,
    )
    freeze_reconstructor(reconstructor)
    checkpoint = create_frozen_reconstructor_checkpoint(reconstructor)
    observations = _observations(history_length)
    model = _model(
        observations,
        history_length=history_length,
        target_contract=contract,
        checkpoint=checkpoint,
    )
    audit = model.bank_heightmap_parameter_audit()

    assert audit["reconstructor_parameter_count"] == 517_769
    assert audit["reconstructor_trainable_parameter_count"] == 0
    assert audit["reconstructor_training"] is False
    assert audit["reconstructor_loaded_frozen"] is True
    assert audit["primary_contract"] == "strict_frozen_reconstructor"
    assert audit["joint_training_authorized"] is False
    assert audit["downstream_elevation_encoder_parameter_count"] == sum(
        parameter.numel() for parameter in model.elevation_encoder.parameters()
    )
    assert audit["total_model_parameter_count"] == sum(parameter.numel() for parameter in model.parameters())
    assert audit["reconstructor_checkpoint_schema"] == checkpoint["schema"]
    assert checkpoint["schema"]["schema_version"] == 2
    assert checkpoint["schema"]["target_contract"] == contract
    assert len(checkpoint["schema_sha256"]) == 64

    model(observations).sum().backward()
    assert all(parameter.grad is None for parameter in model.heightmap_reconstructor.parameters())
    assert all(parameter.grad is not None for parameter in model.elevation_encoder.parameters())
    model.train()
    assert model.training is True
    assert model.heightmap_reconstructor.training is False

    wrong_history = 5 if history_length == 1 else 1
    with pytest.raises(ValueError, match="history/target contract mismatch"):
        _model(
            _observations(wrong_history),
            history_length=wrong_history,
            target_contract=contract,
            checkpoint=checkpoint,
        )

    tampered_contract = copy.deepcopy(checkpoint)
    tampered_contract["schema"]["target_contract"]["target_definition"] = "tampered_definition"
    with pytest.raises(ValueError, match="schema mismatch"):
        _model(
            observations,
            history_length=history_length,
            target_contract=contract,
            checkpoint=tampered_contract,
        )


def test_default_cnn_path_state_and_forward_remain_identical() -> None:
    torch.manual_seed(6301)
    observations = TensorDict(
        {
            "policy": torch.randn(3, 96),
            "height_scan_actor": torch.randn(3, 5, 25, 17),
        },
        batch_size=[3],
    )
    kwargs = {
        "obs": observations,
        "obs_groups": {"actor": ["policy", "height_scan_actor"]},
        "obs_set": "actor",
        "output_dim": 29,
        "distribution_cfg": None,
    }
    torch.manual_seed(6302)
    default_model = PropMLPElevationFusionModel(**kwargs).eval()
    torch.manual_seed(6302)
    explicit_model = PropMLPElevationFusionModel(
        **kwargs,
        elevation_encoder_type="cnn",
        bank_heightmap_target_contract=None,
        bank_downstream_heightmap_contract=None,
        bank_reconstructor_checkpoint_path=None,
        bank_reconstructor_checkpoint_expected_file_sha256=None,
        bank_reconstructor_receipt_path=None,
        _bank_reconstructor_checkpoint_for_testing=None,
    ).eval()

    assert set(default_model.state_dict()) == set(explicit_model.state_dict())
    for name, value in default_model.state_dict().items():
        torch.testing.assert_close(
            explicit_model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        explicit_model(observations),
        default_model(observations),
        rtol=0.0,
        atol=0.0,
    )


def test_bank_export_remains_fail_closed() -> None:
    observations = _observations(5)
    model = _model(observations, history_length=5)
    with pytest.raises(RuntimeError, match="deployment manifest"):
        model.as_jit()
    with pytest.raises(RuntimeError, match="deployment manifest"):
        model.as_onnx(verbose=False)
