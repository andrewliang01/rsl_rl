# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: D103, N802

from __future__ import annotations

import copy
import hashlib
import json
import torch
from pathlib import Path
from tensordict import TensorDict
from typing import Any

import pytest

from rsl_rl.models import M2MMapFreeRecurrentStudent, PropMLPElevationFusionModel
from rsl_rl.utils import resolve_callable, split_and_pad_trajectories

_PROPRIO_DIM = 96
_ACTION_DIM = 29
_FRAME_SHAPE = (1, 4, 16, 96)
_M90_SET = "height_scan_policy"
_STRICT_SET = "mid360_strict_frame"
_FRAME_AGE_SEMANTICS = "uniform_message_age_s"


def _actor_cfg() -> dict[str, Any]:
    return {
        "hidden_dims": [32, 16],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.7,
            "std_type": "scalar",
        },
        "elevation_set": _M90_SET,
        "cnn_observation_type": "depthcamera",
        "depth_camera_near": 0.05,
        "depth_camera_far": 1.8569868066305693,
        "vision_spatial_size": (16, 96),
        "vision_feature_dim": 64,
        "elevation_history_length": 1,
        "cnn_hidden_dims": [4, 8],
        "cnn_kernel_sizes": [3, 3],
        "cnn_strides": [2, 2],
        "prop_feature_dim": 64,
        "prop_hidden_dims": [16],
        "use_prop_encoder": True,
    }


def _checkpoint(tmp_path: Path) -> tuple[Path, str]:
    torch.manual_seed(123)
    obs = TensorDict(
        {
            "policy": torch.randn(2, _PROPRIO_DIM),
            _M90_SET: torch.rand(2, 1, 16, 96) + 0.05,
        },
        batch_size=[2],
    )
    actor = PropMLPElevationFusionModel(
        obs=obs,
        obs_groups={"actor": ["policy", _M90_SET]},
        obs_set="actor",
        output_dim=_ACTION_DIM,
        **copy.deepcopy(_actor_cfg()),
    )
    path = tmp_path / "synthetic_m90.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "iter": 77}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_frame(batch_size: int, *, offset: float = 0.0) -> torch.Tensor:
    frame = torch.zeros(batch_size, *_FRAME_SHAPE, dtype=torch.float32)
    ranges = torch.linspace(0.1, 5.9, 16 * 96).reshape(1, 16, 96)
    frame[:, 0, 0] = (ranges + offset).clamp(0.1, 6.0)
    frame[:, 0, 1] = 1.0
    frame[:, 0, 2] = 0.02
    frame[:, 0, 3] = 1.0
    # Exercise the invalid-range masking path.
    frame[:, 0, 0, 0, 0] = float("nan")
    frame[:, 0, 1, 0, 0] = 0.0
    return frame


def _observations(batch_size: int = 3, *, offset: float = 0.0, include_privileged: bool = True) -> TensorDict:
    data: dict[str, torch.Tensor] = {
        "policy": torch.randn(batch_size, _PROPRIO_DIM),
        _STRICT_SET: _strict_frame(batch_size, offset=offset),
    }
    if include_privileged:
        data["teacher_map"] = torch.randn(batch_size, 1, 3, 16, 96)
        data["ground_truth_pose"] = torch.randn(batch_size, 7)
        data["terrain_metadata"] = torch.randn(batch_size, 13)
    return TensorDict(data, batch_size=[batch_size])


def _student(
    tmp_path: Path,
    obs: TensorDict | None = None,
    *,
    temporal_mode: str = "gru",
    frame_age_semantics: str = _FRAME_AGE_SEMANTICS,
) -> M2MMapFreeRecurrentStudent:
    checkpoint, digest = _checkpoint(tmp_path)
    if obs is None:
        torch.manual_seed(456)
        obs = _observations()
    torch.manual_seed(789)
    return M2MMapFreeRecurrentStudent(
        obs,
        {"actor": ["policy", _STRICT_SET]},
        "actor",
        _ACTION_DIM,
        strict_frame_set=_STRICT_SET,
        proprio_sets=["policy"],
        frozen_ecmm_checkpoint_path=str(checkpoint),
        frozen_ecmm_expected_sha256=digest,
        frozen_ecmm_actor_cfg=_actor_cfg(),
        frame_near_range_m=0.1,
        frame_far_range_m=6.0,
        frame_message_period_s=0.1,
        frame_max_age_s=0.5,
        frame_age_semantics=frame_age_semantics,
        tokenizer_hidden_channels=[4, 8],
        tokenizer_dim=24,
        tokenizer_pooled_spatial_size=(2, 4),
        temporal_mode=temporal_mode,
        gru_hidden_dim=20,
        gru_num_layers=1,
        latent_hidden_dim=24,
    )


def _sequence(base: TensorDict, length: int) -> TensorDict:
    frames = []
    policies = []
    teacher_maps = []
    poses = []
    terrains = []
    for step in range(length):
        frames.append(_strict_frame(base.batch_size[0], offset=0.03 * step))
        policies.append(base["policy"] + 0.01 * step)
        teacher_maps.append(base["teacher_map"] + 100.0 * step)
        poses.append(base["ground_truth_pose"] - 50.0 * step)
        terrains.append(base["terrain_metadata"] + 25.0 * step)
    return TensorDict(
        {
            "policy": torch.stack(policies),
            _STRICT_SET: torch.stack(frames),
            "teacher_map": torch.stack(teacher_maps),
            "ground_truth_pose": torch.stack(poses),
            "terrain_metadata": torch.stack(terrains),
        },
        batch_size=[length, base.batch_size[0]],
    )


def test_deployment_contract_freezes_B_C_distribution_and_exports_model(tmp_path: Path) -> None:
    model = _student(tmp_path)
    model.train()

    assert resolve_callable("M2MMapFreeRecurrentStudent") is M2MMapFreeRecurrentStudent
    assert model.distribution is model.ecmm_core.actor.distribution
    assert model.is_recurrent
    assert not model.ecmm_core.actor.training
    assert all(not parameter.requires_grad for parameter in model.ecmm_core.parameters())
    frozen_batch_norms = [
        module
        for module in model.ecmm_core.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert frozen_batch_norms
    assert all(not module.training for module in frozen_batch_norms)

    audit = model.parameter_audit()
    assert audit["temporal_mode"] == "gru"
    assert audit["is_recurrent"] is True
    assert audit["deployment_inputs"] == {
        "ordered_groups": ["policy", _STRICT_SET],
        "proprio_sets": ["policy"],
        "proprio_group_dims": {"policy": 96},
        "proprio_dim": 96,
        "proprio_feature_to_temporal_dim": 64,
        "strict_frame_set": _STRICT_SET,
        "strict_frame_shape": [1, 4, 16, 96],
        "strict_frame_channels": ["range_m", "valid", "frame_age_s", "new_frame"],
        "strict_frame_storage_dtypes": ["float16", "bfloat16", "float32"],
        "strict_frame_compute_dtype": "float32",
        "strict_frame_near_range_m": 0.1,
        "strict_frame_far_range_m": 6.0,
        "strict_frame_message_period_s": 0.1,
        "strict_frame_max_age_s": 0.5,
        "strict_frame_age_semantics": _FRAME_AGE_SEMANTICS,
        "recurrent_state": [1, "batch", 20],
        "teacher_map": False,
        "ground_truth_pose": False,
        "terrain_metadata": False,
        "future_frames": False,
    }
    assert audit["components"]["frozen_ecmm"]["trainable"] == 0
    assert audit["temporal_input"] == {
        "sources": ["strict_frame_token", "frozen_proprio_B64"],
        "dimension": 88,
        "uses_frozen_teacher_encoder_A": False,
    }
    assert all(
        audit["components"][name]["trainable"] > 0
        for name in ("frame_tokenizer", "gru", "latent_head")
    )


def test_architecture_receipt_binds_all_same_shape_semantics_without_checkpoint_path(
    tmp_path: Path,
) -> None:
    obs = _observations()
    checkpoint, digest = _checkpoint(tmp_path)
    actor_cfg = _actor_cfg()
    model = M2MMapFreeRecurrentStudent(
        obs,
        {"actor": ["policy", _STRICT_SET]},
        "actor",
        _ACTION_DIM,
        strict_frame_set=_STRICT_SET,
        proprio_sets=["policy"],
        frozen_ecmm_checkpoint_path=str(checkpoint),
        frozen_ecmm_expected_sha256=digest,
        frozen_ecmm_actor_cfg=actor_cfg,
        frame_near_range_m=0.1,
        frame_far_range_m=6.0,
        frame_message_period_s=0.1,
        frame_max_age_s=0.5,
        frame_age_semantics=_FRAME_AGE_SEMANTICS,
        tokenizer_hidden_channels=[4, 8],
        tokenizer_dim=24,
        tokenizer_pooled_spatial_size=(2, 4),
        temporal_mode="gru",
        gru_hidden_dim=20,
        gru_num_layers=1,
        latent_hidden_dim=24,
    )

    receipt = model.architecture_receipt()
    assert receipt["schema"] == "m2m_map_free_student_architecture_v2"
    assert receipt["actor_interface"]["ordered_groups"] == ["policy", _STRICT_SET]
    assert receipt["actor_interface"]["proprio_group_dims"] == {"policy": 96}
    assert receipt["strict_frame"] == {
        "shape_without_batch": [1, 4, 16, 96],
        "channels": ["range_m", "valid", "frame_age_s", "new_frame"],
        "near_range_m": 0.1,
        "far_range_m": 6.0,
        "message_period_s": 0.1,
        "max_age_s": 0.5,
        "frame_age_semantics": _FRAME_AGE_SEMANTICS,
        "accepted_storage_dtypes": ["float16", "bfloat16", "float32"],
        "compute_dtype": "float32",
    }
    assert receipt["tokenizer"]["hidden_channels"] == [4, 8]
    assert receipt["tokenizer"]["token_dim"] == 24
    assert receipt["tokenizer"]["pooled_spatial_size"] == [2, 4]
    assert receipt["temporal"] == {
        "mode": "gru",
        "input_dim": 88,
        "gru_hidden_dim": 20,
        "gru_num_layers": 1,
        "latent_hidden_dim": 24,
        "proprio_feature_in_temporal_input": True,
    }
    assert receipt["frozen_ecmm_actor_cfg"]["activation"] == "elu"
    assert receipt["frozen_ecmm_actor_cfg"]["distribution_cfg"]["class_name"] == (
        "GaussianDistribution"
    )
    assert receipt["frozen_ecmm_actor_cfg"]["ray_time_vertical_fov_degrees"] == [-52.0, 7.0]
    assert not [
        key
        for key in receipt["frozen_ecmm_actor_cfg"]
        if "path" in key.lower()
    ]
    assert receipt["artifact_binding"]["checkpoint_path_embedded"] is False
    assert str(checkpoint.resolve()) not in json.dumps(receipt, sort_keys=True, allow_nan=False)

    receipt["strict_frame"]["near_range_m"] = 999.0
    assert model.architecture_receipt()["strict_frame"]["near_range_m"] == 0.1


@pytest.mark.parametrize(
    "frame_age_semantics",
    ["uniform_message_age_s", "winning_subframe_age_20ms"],
)
def test_frame_age_semantics_is_required_and_exactly_bound(
    tmp_path: Path,
    frame_age_semantics: str,
) -> None:
    model = _student(tmp_path, frame_age_semantics=frame_age_semantics)

    assert model.frame_tokenizer.frame_age_semantics == frame_age_semantics
    assert model.architecture_receipt()["strict_frame"]["frame_age_semantics"] == frame_age_semantics
    assert model.parameter_audit()["deployment_inputs"]["strict_frame_age_semantics"] == (
        frame_age_semantics
    )


def test_forward_uses_predicted_A_with_exact_frozen_B_C_and_runner_distribution_api(tmp_path: Path) -> None:
    obs = _observations()
    model = _student(tmp_path, obs)

    deterministic, latent_a = model.forward_with_latent(obs)
    with torch.no_grad():
        proprio = model.ecmm_core.encode_proprio(obs)
        expected = model.ecmm_core.action_mean_from_A(proprio, latent_a)
    torch.testing.assert_close(deterministic, expected)
    assert latent_a.shape == (obs.batch_size[0], 64)
    assert deterministic.shape == (obs.batch_size[0], 29)

    model.reset()
    paired_latent, paired_mean = model.predict_latent_and_action_mean(obs)
    torch.testing.assert_close(paired_latent, latent_a)
    torch.testing.assert_close(paired_mean, deterministic)

    model.reset()
    sampled = model(obs, stochastic_output=True)
    assert sampled.shape == deterministic.shape
    assert model.output_mean.shape == deterministic.shape
    assert model.output_std.shape == deterministic.shape
    assert model.output_entropy.shape == (obs.batch_size[0],)
    params = model.output_distribution_params
    assert len(params) == 2
    assert model.get_output_log_prob(sampled).shape == (obs.batch_size[0],)
    torch.testing.assert_close(model.get_kl_divergence(params, params), torch.zeros(obs.batch_size[0]))
    assert model.get_hidden_state().shape == (1, obs.batch_size[0], 20)  # type: ignore[union-attr]


def test_step_and_dense_sequence_paths_are_numerically_equivalent(tmp_path: Path) -> None:
    base = _observations(batch_size=2)
    sequence = _sequence(base, length=5)
    model = _student(tmp_path, base).eval()
    masks = torch.ones(5, 2, dtype=torch.bool)
    initial_hidden = torch.zeros(1, 2, 20)

    sequence_action, sequence_latent = model.forward_with_latent(
        sequence,
        masks=masks,
        hidden_state=initial_hidden,
    )
    model.reset()
    step_results = [model.forward_with_latent(sequence[step]) for step in range(5)]
    step_action = torch.stack([result[0] for result in step_results])
    step_latent = torch.stack([result[1] for result in step_results])

    torch.testing.assert_close(sequence_action, step_action, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sequence_latent, step_latent, rtol=1e-5, atol=1e-6)


def test_proprioception_B64_is_part_of_temporal_input_and_changes_Ahat(tmp_path: Path) -> None:
    obs = _observations(batch_size=2)
    model = _student(tmp_path, obs).eval()
    changed_proprio = obs.clone()
    changed_proprio["policy"] = changed_proprio["policy"] + 2.5

    model.reset()
    original_latent = model.predict_latent(obs)
    model.reset()
    changed_latent = model.predict_latent(changed_proprio)

    assert model.temporal_input_dim == model.frame_tokenizer.token_dim + 64
    assert not torch.allclose(original_latent, changed_latent)


@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.bfloat16])
def test_low_precision_frame_storage_is_converted_to_float32_compute(
    tmp_path: Path,
    storage_dtype: torch.dtype,
) -> None:
    obs = _observations(batch_size=2)
    model = _student(tmp_path, obs).eval()
    low_precision_obs = obs.clone()
    low_precision_obs[_STRICT_SET] = low_precision_obs[_STRICT_SET].to(storage_dtype)

    model.reset()
    reference_action, reference_latent = model.forward_with_latent(obs)
    model.reset()
    actual_action, actual_latent = model.forward_with_latent(low_precision_obs)

    assert actual_action.dtype == torch.float32
    assert actual_latent.dtype == torch.float32
    torch.testing.assert_close(actual_action, reference_action, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(actual_latent, reference_latent, rtol=2e-3, atol=2e-4)


def test_message_age_uses_independent_maximum_not_publish_period(tmp_path: Path) -> None:
    obs = _observations(batch_size=1)
    model = _student(tmp_path, obs).eval()
    frame = obs[_STRICT_SET].clone()
    frame[:, 0, 2] = 0.3  # Three missed 10 Hz publications is still representable.
    normalized = model.frame_tokenizer._normalize(frame)
    torch.testing.assert_close(normalized[:, 2], torch.full_like(normalized[:, 2], 0.6))

    frame[:, 0, 2] = 0.9
    clipped = model.frame_tokenizer._normalize(frame)
    torch.testing.assert_close(clipped[:, 2], torch.ones_like(clipped[:, 2]))


def test_current_mode_is_memoryless_and_uses_same_deployment_contract(tmp_path: Path) -> None:
    obs = _observations(batch_size=2)
    model = _student(tmp_path, obs, temporal_mode="current").eval()

    assert model.temporal_mode == "current"
    assert model.is_recurrent is False
    assert model.gru is None
    assert model.get_hidden_state() is None
    first_action, first_latent = model.forward_with_latent(obs)
    second_action, second_latent = model.forward_with_latent(obs)
    torch.testing.assert_close(second_action, first_action)
    torch.testing.assert_close(second_latent, first_latent)
    assert model.get_hidden_state() is None

    sequence = _sequence(obs, length=3)
    sequence_action = model(sequence)
    step_action = torch.stack([model(sequence[step]) for step in range(3)])
    torch.testing.assert_close(sequence_action, step_action)
    audit = model.parameter_audit()
    assert audit["is_recurrent"] is False
    assert audit["deployment_inputs"]["recurrent_state"] is False
    assert audit["components"]["gru"] == {"total": 0, "trainable": 0}
    assert audit["components"]["current_encoder"]["trainable"] > 0


def test_padded_bptt_matches_step_rollout_with_done_resets(tmp_path: Path) -> None:
    base = _observations(batch_size=2)
    sequence = _sequence(base, length=5)
    dones = torch.zeros(5, 2, 1, dtype=torch.uint8)
    dones[1, 0] = 1
    dones[2, 1] = 1
    padded_obs, masks = split_and_pad_trajectories(sequence, dones)
    model = _student(tmp_path, base).eval()
    initial_hidden = torch.zeros(1, masks.shape[1], 20)

    padded_action = model(
        padded_obs,
        masks=masks,
        hidden_state=initial_hidden,
    )
    model.reset()
    rollout_actions = []
    for step in range(5):
        rollout_actions.append(model(sequence[step]))
        model.reset(dones[step])
    rollout_action = torch.stack(rollout_actions)

    torch.testing.assert_close(padded_action, rollout_action, rtol=1e-5, atol=1e-6)
    model.reset(torch.ones(2, dtype=torch.uint8))
    hidden = model.get_hidden_state()
    assert hidden is not None
    assert torch.count_nonzero(hidden) == 0


def test_arbitrary_padded_tbptt_api_preserves_layout_and_matches_valid_steps(tmp_path: Path) -> None:
    base = _observations(batch_size=2)
    padded_obs = _sequence(base, length=2)
    # Two independent C10 chunks with lengths 2 and 1: there are three valid
    # items, which is not divisible by L=2 and therefore violates the generic
    # unpad_trajectories reshape assumption by construction.
    padded_obs["policy"][1, 1].zero_()
    padded_obs[_STRICT_SET][1, 1].zero_()
    masks = torch.tensor(
        [
            [[True], [True]],
            [[True], [False]],
        ]
    )
    initial_hidden = torch.zeros(1, 2, 20)
    model = _student(tmp_path, base).eval()

    padded_latent, padded_mean = model.predict_padded_latent_and_action_mean(
        padded_obs,
        masks,
        initial_hidden,
    )
    assert padded_latent.shape == (2, 2, 64)
    assert padded_mean.shape == (2, 2, 29)
    assert model.get_hidden_state() is None

    # Both accepted mask layouts describe the same C10 batch.
    latent_2d, mean_2d = model.predict_padded_latent_and_action_mean(
        padded_obs,
        masks.squeeze(-1),
        initial_hidden,
    )
    torch.testing.assert_close(latent_2d, padded_latent)
    torch.testing.assert_close(mean_2d, padded_mean)

    for column, sequence_length in ((0, 2), (1, 1)):
        model.reset(hidden_state=torch.zeros(1, 1, 20))
        for step in range(sequence_length):
            step_latent, step_mean = model.predict_latent_and_action_mean(
                padded_obs[step, column : column + 1]
            )
            torch.testing.assert_close(padded_latent[step, column : column + 1], step_latent)
            torch.testing.assert_close(padded_mean[step, column : column + 1], step_mean)


def test_bptt_gradients_update_only_tokenizer_gru_and_latent_head(tmp_path: Path) -> None:
    base = _observations(batch_size=2)
    sequence = _sequence(base, length=4)
    model = _student(tmp_path, base).train()
    masks = torch.ones(4, 2, dtype=torch.bool)

    action, latent_a = model.forward_with_latent(
        sequence,
        masks=masks,
        hidden_state=torch.zeros(1, 2, 20),
    )
    (action.square().mean() + 0.1 * latent_a.square().mean()).backward()

    assert model.gru is not None
    for module in (model.frame_tokenizer, model.gru, model.latent_head):
        grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert grads
        assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in grads)
    assert all(not parameter.requires_grad for parameter in model.ecmm_core.parameters())
    assert all(parameter.grad is None for parameter in model.ecmm_core.parameters())


def test_teacher_pose_and_terrain_values_cannot_change_deployment_action(tmp_path: Path) -> None:
    full_obs = _observations(batch_size=2)
    model = _student(tmp_path, full_obs).eval()
    changed_privileged = full_obs.clone()
    changed_privileged["teacher_map"] = torch.full_like(changed_privileged["teacher_map"], 1.0e9)
    changed_privileged["ground_truth_pose"] = torch.full_like(changed_privileged["ground_truth_pose"], -1.0e8)
    changed_privileged["terrain_metadata"] = torch.full_like(changed_privileged["terrain_metadata"], 3.0e7)

    model.reset()
    expected = model(full_obs)
    model.reset()
    actual = model(changed_privileged)
    torch.testing.assert_close(actual, expected)

    deploy_only = TensorDict(
        {"policy": full_obs["policy"], _STRICT_SET: full_obs[_STRICT_SET]},
        batch_size=full_obs.batch_size,
    )
    model.reset()
    torch.testing.assert_close(model(deploy_only), expected)


def test_constructor_and_runtime_fail_closed_on_input_contract_violations(tmp_path: Path) -> None:
    obs = _observations()
    checkpoint, digest = _checkpoint(tmp_path)
    kwargs = {
        "strict_frame_set": _STRICT_SET,
        "proprio_sets": ["policy"],
        "frozen_ecmm_checkpoint_path": str(checkpoint),
        "frozen_ecmm_expected_sha256": digest,
        "frozen_ecmm_actor_cfg": _actor_cfg(),
        "frame_near_range_m": 0.1,
        "frame_far_range_m": 6.0,
        "frame_message_period_s": 0.1,
        "frame_max_age_s": 0.5,
        "frame_age_semantics": _FRAME_AGE_SEMANTICS,
    }

    missing_semantics = dict(kwargs)
    del missing_semantics["frame_age_semantics"]
    with pytest.raises(TypeError, match="frame_age_semantics"):
        M2MMapFreeRecurrentStudent(
            obs,
            {"actor": ["policy", _STRICT_SET]},
            "actor",
            _ACTION_DIM,
            **missing_semantics,
        )
    for invalid_semantics in ("message_age_s", "", None, True):
        invalid_semantics_kwargs = {**kwargs, "frame_age_semantics": invalid_semantics}
        with pytest.raises(ValueError, match="frame_age_semantics"):
            M2MMapFreeRecurrentStudent(
                obs,
                {"actor": ["policy", _STRICT_SET]},
                "actor",
                _ACTION_DIM,
                **invalid_semantics_kwargs,
            )

    with pytest.raises(ValueError, match="exactly the explicit original proprio"):
        M2MMapFreeRecurrentStudent(
            obs,
            {"actor": ["policy", _STRICT_SET, "ground_truth_pose"]},
            "actor",
            _ACTION_DIM,
            **kwargs,
        )
    with pytest.raises(ValueError, match="output_dim=29"):
        M2MMapFreeRecurrentStudent(
            obs,
            {"actor": ["policy", _STRICT_SET]},
            "actor",
            28,
            **kwargs,
        )
    invalid_age_kwargs = dict(kwargs)
    invalid_age_kwargs["frame_max_age_s"] = 0.05
    with pytest.raises(ValueError, match="max_age_s"):
        M2MMapFreeRecurrentStudent(
            obs,
            {"actor": ["policy", _STRICT_SET]},
            "actor",
            _ACTION_DIM,
            **invalid_age_kwargs,
        )

    model = _student(tmp_path, obs)
    wrong_frame = obs.clone()
    wrong_frame[_STRICT_SET] = torch.zeros(obs.batch_size[0], 1, 4, 15, 96)
    with pytest.raises(ValueError, match="Strict MID-360 frame"):
        model(wrong_frame)
    sequence = _sequence(obs, length=2)
    with pytest.raises(ValueError, match="requires trajectory masks"):
        model(sequence)
    with pytest.raises(ValueError, match="Trajectory masks"):
        model(sequence, masks=torch.ones(2, obs.batch_size[0]))


def test_partial_done_reset_zeros_only_selected_hidden_state(tmp_path: Path) -> None:
    obs = _observations(batch_size=3)
    model = _student(tmp_path, obs)
    model(obs)
    before = model.get_hidden_state()
    assert before is not None
    before = before.clone()

    model.reset(torch.tensor([1, 0, 1], dtype=torch.uint8))
    after = model.get_hidden_state()
    assert after is not None
    torch.testing.assert_close(after[:, 0], torch.zeros_like(after[:, 0]))
    torch.testing.assert_close(after[:, 2], torch.zeros_like(after[:, 2]))
    torch.testing.assert_close(after[:, 1], before[:, 1])
