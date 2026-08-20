"""Contracts for the from-scratch privileged-map teacher."""

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import M2MObservedHistoryScratchTeacher


_BATCH = 4
_MAP_GROUP = "m2m_teacher_observed_history"


def _obs(batch: int = _BATCH) -> TensorDict:
    teacher_map = torch.empty(batch, 1, 2, 16, 96)
    teacher_map[:, :, 0] = 1.0
    teacher_map[:, :, 1] = 1.0
    return TensorDict(
        {
            "policy": torch.randn(batch, 96),
            _MAP_GROUP: teacher_map,
        },
        batch_size=[batch],
    )


def _contract() -> dict[str, object]:
    return {
        "source": "observed_m52_history",
        "alignment": "gt_pose_training_only",
        "target_grid": "m90_spherical_16x96",
        "uses_future_frames": False,
        "uses_privileged_terrain_mesh": False,
        "uses_synthetic_fill": False,
        "near_range_m": 0.05,
        "far_range_m": 1.85699,
        "storage_backend": "voxel_hash_2p5d",
        "retention_mode": "episode",
        "voxel_size_m": 0.05,
        "hash_capacity": 131072,
        "hash_max_probes": 16,
    }


def _model(**overrides) -> M2MObservedHistoryScratchTeacher:
    values = {
        "teacher_map_set": _MAP_GROUP,
        "proprio_sets": ["policy"],
        "map_contract": _contract(),
        "encoder_hidden_channels": [16, 32, 64],
        "encoder_pooled_spatial_size": (2, 6),
        "encoder_mlp_hidden_dim": 128,
        "prop_feature_dim": 64,
        "prop_hidden_dims": [128],
        "fusion_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "strict_runtime_value_checks": True,
    }
    values.update(overrides)
    return M2MObservedHistoryScratchTeacher(
        _obs(),
        {"actor": ["policy", _MAP_GROUP]},
        "actor",
        29,
        **values,
    )


def test_all_a_b_c_and_distribution_parameters_train_from_random_initialization() -> None:
    torch.manual_seed(7)
    model = _model()
    audit = model.parameter_audit()
    assert audit["pretrained_policy_loaded"] is False
    assert audit["checkpoint_path_fields_accepted"] is False
    assert audit["all_actor_parameters_trainable"] is True
    assert all(audit["required_trainable_components_present"].values())
    assert audit["architecture"]["training_initialization"] == "random_no_pretrained_policy"
    assert audit["architecture"]["map_contract"]["channels"] == ["range_m", "valid"]
    assert audit["architecture"]["map_contract"]["retention_mode"] == "episode"
    assert audit["architecture"]["map_contract"]["query_frame"] == "full_pose"
    assert audit["architecture"]["map_contract"]["timestamp_visibility"] == "mapper_internal_only"
    assert model.map_encoder.spatial_encoder[0].conv.in_channels == 2
    assert audit["checkpoint_contract"]["owner"] == "ordinary_PPO_full_actor_state_dict"
    assert set(audit["checkpoint_contract"]["saved_components"]) == {
        "map_encoder_A",
        "proprio_encoder_B",
        "control_head_C",
        "distribution",
    }


def test_query_frame_is_receipted_and_invalid_frame_fails_closed() -> None:
    contract = _contract()
    contract["query_frame"] = "gravity_yaw"
    model = _model(map_contract=contract)
    assert model.architecture_receipt()["map_contract"]["query_frame"] == "gravity_yaw"

    contract["query_frame"] = "body_roll_pitch"
    with pytest.raises(ValueError, match="query_frame"):
        _model(map_contract=contract)


def test_action_distribution_must_resolve_to_rsl_distribution() -> None:
    with pytest.raises(ValueError, match="Distribution subclass"):
        _model(
            distribution_cfg={
                "class_name": "torch.nn.Linear",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )


def test_forward_and_joint_gradients_cover_a_b_c() -> None:
    model = _model()
    obs = _obs()
    action = model(obs, stochastic_output=True)
    assert action.shape == (_BATCH, 29)
    assert model.output_mean.shape == (_BATCH, 29)
    latent, mean = model.predict_latent_and_action_mean(obs)
    assert latent.shape == (_BATCH, 64)
    assert mean.shape == (_BATCH, 29)

    loss = model.get_output_log_prob(action).mean() + latent.square().mean() + mean.square().mean()
    loss.backward()
    for prefix in ("map_encoder.", "prop_mlp.", "mlp.", "distribution."):
        assert any(
            name.startswith(prefix) and parameter.grad is not None
            for name, parameter in model.named_parameters()
        )


def test_no_checkpoint_constructor_fields_exist_and_seeds_change_weights() -> None:
    torch.manual_seed(1)
    first = _model()
    torch.manual_seed(2)
    second = _model()
    assert any(
        not torch.equal(first.state_dict()[key], second.state_dict()[key])
        for key in first.state_dict()
        if first.state_dict()[key].is_floating_point()
    )

    with pytest.raises(TypeError, match="unexpected keyword"):
        _model(frozen_ecmm_checkpoint_path="/tmp/forbidden.pt")


def test_state_dict_roundtrip_contains_complete_actor() -> None:
    model = _model()
    state = copy.deepcopy(model.state_dict())
    assert any(key.startswith("map_encoder.") for key in state)
    assert any(key.startswith("prop_mlp.") for key in state)
    assert any(key.startswith("mlp.") for key in state)
    assert any(key.startswith("distribution.") for key in state)

    restored = _model()
    restored.load_state_dict(state, strict=True)
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_causal_contract_and_strict_map_checks_fail_closed() -> None:
    bad_contract = _contract()
    bad_contract["uses_future_frames"] = True
    with pytest.raises(ValueError, match="Future-frame"):
        _model(map_contract=bad_contract)

    model = _model()
    obs = _obs()
    obs[_MAP_GROUP][:, :, 1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="exact binary"):
        model(obs)
