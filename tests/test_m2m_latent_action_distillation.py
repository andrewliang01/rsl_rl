# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import (
    M2MDistillationLossConfig,
    M2MLatentActionDistillation,
    M2MMaskedLatentActionLoss,
)
from rsl_rl.storage import M2MSequenceRolloutStorage


class _DummyFrozenCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_head = nn.Linear(64, 29, bias=False)
        self.action_head.requires_grad_(False)
        self.checkpoint_sha256 = "a" * 64
        self.actor_state_dict_key = "actor_state_dict"

    def action_mean(self, latent: torch.Tensor) -> torch.Tensor:
        return self.action_head(latent)

    def parameter_audit(self) -> dict[str, Any]:
        return {
            "contract": {
                "latent_a_dim": 64,
                "action_dim": 29,
                "proprio_dim": 2,
            }
        }


class _DummyC07Student(nn.Module):
    is_recurrent = True
    temporal_mode = "gru"

    def __init__(self) -> None:
        super().__init__()
        self.frame_tokenizer = nn.Sequential(nn.Linear(5, 8), nn.Tanh())
        self.gru = nn.GRU(8, 6, num_layers=1)
        self.current_encoder = None
        self.latent_head = nn.Linear(6, 64)
        self.ecmm_core = _DummyFrozenCore()
        self._hidden_state: torch.Tensor | None = None
        self.rollout_forward_calls = 0
        self.padded_predict_calls = 0
        self.last_padded_batch_size: torch.Size | None = None

    @staticmethod
    def _inputs(obs: TensorDict) -> torch.Tensor:
        return torch.cat((obs["policy"], obs["strict_frame"]), dim=-1)

    def forward_with_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del masks, hidden_state, stochastic_output
        self.rollout_forward_calls += 1
        token = self.frame_tokenizer(self._inputs(obs))
        output, self._hidden_state = self.gru(token.unsqueeze(0), self._hidden_state)
        latent = self.latent_head(output.squeeze(0))
        return self.ecmm_core.action_mean(latent), latent

    def predict_padded_latent_and_action_mean(
        self,
        obs: TensorDict,
        masks: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.padded_predict_calls += 1
        self.last_padded_batch_size = obs.batch_size
        token = self.frame_tokenizer(self._inputs(obs))
        token = token * masks.to(token.dtype)
        output, _ = self.gru(token, hidden_state)
        latent = self.latent_head(output)
        return latent, self.ecmm_core.action_mean(latent)

    def predict_latent_and_action_mean(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del masks, hidden_state
        token = self.frame_tokenizer(self._inputs(obs))
        output = torch.tanh(token[..., :6])
        latent = self.latent_head(output)
        return latent, self.ecmm_core.action_mean(latent)

    def get_hidden_state(self) -> torch.Tensor | None:
        return self._hidden_state

    def reset(self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> None:
        if hidden_state is not None:
            self._hidden_state = hidden_state
        if dones is not None and self._hidden_state is not None:
            done = dones.reshape(-1).to(dtype=torch.bool, device=self._hidden_state.device)
            self._hidden_state[:, done] = 0.0

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    @property
    def output_std(self) -> torch.Tensor:
        return torch.ones(29)


class _DummyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.map_encoder = nn.Linear(4, 64)
        self.action_head = nn.Linear(64, 29)
        self.label_calls = 0
        self.grad_enabled_during_call: list[bool] = []

    def predict_latent_and_action_mean(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        self.label_calls += 1
        self.grad_enabled_during_call.append(torch.is_grad_enabled())
        latent = self.map_encoder(obs["teacher_map"])
        return latent, self.action_head(latent)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones


def _student_obs(num_envs: int, step: int = 0) -> TensorDict:
    base = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)
    return TensorDict(
        {
            "policy": torch.cat((base, torch.full_like(base, float(step))), dim=-1),
            "strict_frame": torch.cat(
                (
                    base + 0.1 * step,
                    torch.full_like(base, 0.25 + 0.01 * step),
                    torch.full_like(base, 1.0),
                ),
                dim=-1,
            ),
        },
        batch_size=[num_envs],
    )


def _full_obs(num_envs: int, step: int = 0) -> TensorDict:
    student = _student_obs(num_envs, step)
    return TensorDict(
        {
            "policy": student["policy"],
            "strict_frame": student["strict_frame"],
            "teacher_map": torch.full((num_envs, 4), 0.2 * (step + 1), dtype=torch.float32),
            "ground_truth_pose": torch.full((num_envs, 7), 100.0 + step),
        },
        batch_size=[num_envs],
    )


def _loss_config(**overrides: Any) -> M2MDistillationLossConfig:
    values: dict[str, Any] = {
        "objective_mode": "joint",
        "latent_smooth_l1_weight": 1.0,
        "latent_cosine_weight": 0.2,
        "action_mse_weight": 2.0,
        "smooth_l1_beta": 1.0,
        "latent_normalization": "l2",
        "normalization_eps": 1.0e-8,
    }
    values.update(overrides)
    return M2MDistillationLossConfig(**values)


def _algorithm(
    *,
    num_envs: int = 2,
    rollout_length: int = 4,
    learning_rate: float = 1.0e-2,
    loss_config: M2MDistillationLossConfig | None = None,
    frozen_artifact_receipt: Mapping[str, Any] | None = None,
    rollout_action_source: str = "student_mean",
    strict_teacher_label_checks: bool = False,
) -> M2MLatentActionDistillation:
    student = _DummyC07Student()
    storage = M2MSequenceRolloutStorage(
        num_envs=num_envs,
        num_transitions_per_env=rollout_length,
        student_obs=_student_obs(num_envs),
        allowed_student_keys=("policy", "strict_frame"),
        hidden_state_shape=(1, 6),
        device="cpu",
    )
    return M2MLatentActionDistillation(
        student,
        _DummyTeacher(),
        storage,
        loss_config=loss_config or _loss_config(),
        frozen_artifact_receipt=frozen_artifact_receipt,
        learning_rate=learning_rate,
        optimizer="adam",
        num_learning_epochs=1,
        num_mini_batches=1,
        sequence_length=3,
        max_grad_norm=1.0,
        rollout_action_source=rollout_action_source,
        strict_teacher_label_checks=strict_teacher_label_checks,
        shuffle_sequences=False,
        sequence_seed=7,
        device="cpu",
    )


def _collect_rollout(algorithm: M2MLatentActionDistillation, steps: int) -> None:
    for step in range(steps):
        algorithm.act(_full_obs(algorithm.storage.num_envs, step))
        dones = torch.zeros(algorithm.storage.num_envs, dtype=torch.bool)
        if step == 1:
            dones[0] = True
        algorithm.process_env_step(
            _full_obs(algorithm.storage.num_envs, step + 1),
            torch.zeros(algorithm.storage.num_envs),
            dones,
            {},
        )


def test_known_analytic_masked_loss_and_padding_nan_invariance() -> None:
    config = M2MDistillationLossConfig(
        objective_mode="joint",
        latent_smooth_l1_weight=2.0,
        latent_cosine_weight=3.0,
        action_mse_weight=4.0,
        smooth_l1_beta=1.0,
        latent_normalization="l2",
        normalization_eps=1.0e-8,
    )
    loss_fn = M2MMaskedLatentActionLoss(config)
    student_latent = torch.full((2, 2, 64), float("nan"), requires_grad=True)
    teacher_latent = torch.full((2, 2, 64), float("inf"))
    student_action = torch.full((2, 2, 29), float("nan"), requires_grad=True)
    teacher_action = torch.full((2, 2, 29), float("inf"))
    with torch.no_grad():
        student_latent[0, 0].zero_()
        student_latent[0, 0, 0] = 1.0
        teacher_latent[0, 0].zero_()
        teacher_latent[0, 0, 1] = 1.0
        student_action[0, 0].zero_()
        teacher_action[0, 0].fill_(2.0)
    masks = torch.zeros(2, 2, 1, dtype=torch.bool)
    masks[0, 0] = True

    total, components = loss_fn(
        student_latent,
        teacher_latent,
        student_action,
        teacher_action,
        masks,
    )
    # Two unit coordinate differences contribute SmoothL1 0.5 each,
    # followed by the required feature mean over A64.
    assert components["latent_smooth_l1"].item() == pytest.approx(1.0 / 64.0)
    assert components["latent_cosine"].item() == pytest.approx(1.0)
    assert components["action_mean_mse"].item() == pytest.approx(4.0)
    assert components["valid_steps"].item() == 1.0
    assert total.item() == pytest.approx(2.0 / 64.0 + 3.0 + 16.0)

    total.backward()
    assert torch.count_nonzero(student_latent.grad[~masks.squeeze(-1)]) == 0
    assert torch.count_nonzero(student_action.grad[~masks.squeeze(-1)]) == 0
    assert torch.isfinite(student_latent.grad[masks.squeeze(-1)]).all()
    assert torch.isfinite(student_action.grad[masks.squeeze(-1)]).all()


def test_loss_configuration_makes_action_only_explicit_and_rejects_zero_objective() -> None:
    action_only = M2MDistillationLossConfig(
        objective_mode="action_only",
        latent_smooth_l1_weight=0.0,
        latent_cosine_weight=0.0,
        action_mse_weight=1.0,
    )
    assert action_only.receipt()["objective_mode"] == "action_only"
    with pytest.raises(ValueError, match="action_only"):
        M2MDistillationLossConfig(
            objective_mode="action_only",
            latent_smooth_l1_weight=0.1,
            latent_cosine_weight=0.0,
            action_mse_weight=1.0,
        )
    with pytest.raises(ValueError, match="joint"):
        M2MDistillationLossConfig(
            objective_mode="joint",
            latent_smooth_l1_weight=0.0,
            latent_cosine_weight=0.0,
            action_mse_weight=0.0,
        )


def test_online_rollout_advances_student_once_and_never_stores_teacher_map() -> None:
    algorithm = _algorithm(rollout_action_source="teacher_mean")
    full_obs = _full_obs(2)
    expected_teacher_action = algorithm.teacher.predict_latent_and_action_mean(full_obs)[1].detach()
    algorithm.teacher.label_calls = 0
    algorithm.teacher.grad_enabled_during_call.clear()

    action = algorithm.act(full_obs)
    assert algorithm.student.rollout_forward_calls == 1
    assert algorithm.teacher.label_calls == 1
    assert algorithm.teacher.grad_enabled_during_call == [False]
    torch.testing.assert_close(action, expected_teacher_action)
    assert algorithm._pending is not None
    assert set(algorithm._pending.student_observations.keys()) == {"policy", "strict_frame"}
    assert not algorithm._pending.teacher_latent_A.requires_grad
    assert not algorithm._pending.teacher_action_mean.requires_grad

    algorithm.process_env_step(
        _full_obs(2, 1),
        torch.zeros(2),
        torch.tensor([True, False]),
        {},
    )
    assert algorithm.storage.step == 1
    assert not hasattr(algorithm.storage, "teacher_map")
    assert not hasattr(algorithm.storage, "teacher_observations")
    assert not hasattr(algorithm.storage, "next_observations")
    assert algorithm.student.rollout_forward_calls == 1
    assert torch.count_nonzero(algorithm.student.get_hidden_state()[:, 0]) == 0
    assert all(not parameter.requires_grad for parameter in algorithm.teacher.parameters())
    assert all(parameter.grad is None for parameter in algorithm.teacher.parameters())


def test_default_teacher_label_hot_path_has_no_tensor_host_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    algorithm = _algorithm()
    observations = _full_obs(2)

    def forbidden_sync(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("default C11 teacher-label path attempted a Tensor host synchronization")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_sync)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_sync)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden_sync)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden_sync)

    action = algorithm.act(observations)

    assert action.shape == (2, 29)
    assert algorithm._pending is not None


def test_corrupt_teacher_labels_are_device_sanitized_by_default_and_strict_mode_fails() -> None:
    default_algorithm = _algorithm(rollout_action_source="teacher_mean")
    with torch.no_grad():
        default_algorithm.teacher.map_encoder.weight.fill_(float("nan"))
        default_algorithm.teacher.action_head.bias.fill_(float("inf"))

    action = default_algorithm.act(_full_obs(2))

    assert default_algorithm._pending is not None
    torch.testing.assert_close(
        default_algorithm._pending.teacher_latent_A,
        torch.zeros_like(default_algorithm._pending.teacher_latent_A),
    )
    torch.testing.assert_close(
        default_algorithm._pending.teacher_action_mean,
        torch.zeros_like(default_algorithm._pending.teacher_action_mean),
    )
    torch.testing.assert_close(action, torch.zeros_like(action))
    integrity = default_algorithm.audit()["teacher_label_integrity"]
    assert integrity == {
        "owner": "C11_online_collection_boundary",
        "strict_device_sync_checks": False,
        "default_hot_path_host_sync": False,
        "default_nonfinite_replacement": 0.0,
        "strict_mode_fail_closed": True,
    }

    strict_algorithm = _algorithm(strict_teacher_label_checks=True)
    with torch.no_grad():
        strict_algorithm.teacher.map_encoder.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        strict_algorithm.act(_full_obs(2))
    assert strict_algorithm._pending is None


def test_c10_batch_flows_directly_to_c07_tbptt_and_gradient_whitelist() -> None:
    torch.manual_seed(11)
    algorithm = _algorithm(rollout_length=4)
    _collect_rollout(algorithm, 4)
    frozen_before = copy.deepcopy(algorithm.student.ecmm_core.state_dict())
    teacher_before = copy.deepcopy(algorithm.teacher.state_dict())
    trainable_before = {
        name: value.detach().clone()
        for name, value in algorithm.student.named_parameters()
        if value.requires_grad
    }

    metrics = algorithm.update()

    assert algorithm.student.padded_predict_calls == 1
    assert algorithm.student.last_padded_batch_size == torch.Size([3, 4])
    assert algorithm.storage.step == 0
    assert metrics["valid_steps"] == 4 * 2
    assert metrics["optimizer_steps"] == 1.0
    assert metrics["algorithm_updates"] == 1.0
    assert all(math_value >= 0.0 for math_value in metrics.values())
    assert any(
        not torch.equal(trainable_before[name], value)
        for name, value in algorithm.student.named_parameters()
        if value.requires_grad
    )
    for name, value in algorithm.student.ecmm_core.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name])
    for name, value in algorithm.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name])
    assert all(value.grad is None for value in algorithm.student.ecmm_core.parameters())
    assert all(value.grad is None for value in algorithm.teacher.parameters())
    assert all(
        value.grad is None
        for name, value in algorithm.student.named_parameters()
        if not name.startswith(("frame_tokenizer.", "gru.", "current_encoder.", "latent_head."))
    )
    assert all(
        name.startswith(("frame_tokenizer.", "gru.", "latent_head."))
        for name in algorithm.trainable_parameter_names
    )


def test_action_only_ablation_runs_with_exact_zero_latent_weights() -> None:
    config = M2MDistillationLossConfig(
        objective_mode="action_only",
        latent_smooth_l1_weight=0.0,
        latent_cosine_weight=0.0,
        action_mse_weight=1.5,
    )
    algorithm = _algorithm(rollout_length=3, loss_config=config)
    _collect_rollout(algorithm, 3)

    metrics = algorithm.update()

    assert metrics["loss"] == pytest.approx(1.5 * metrics["action_mean_mse"])
    assert metrics["valid_steps"] == 6.0
    assert algorithm.audit()["configuration"]["loss"]["objective_mode"] == "action_only"


def test_checkpoint_roundtrip_saves_only_trainable_student_and_rejects_mismatch() -> None:
    torch.manual_seed(21)
    source = _algorithm(rollout_length=3)
    _collect_rollout(source, 3)
    source.update()
    payload = source.save()

    assert set(payload) == {
        "schema",
        "config_receipt",
        "student_trainable_state_dict",
        "optimizer_state_dict",
        "algorithm_iteration",
        "frozen_artifact_receipt",
    }
    assert "teacher_state_dict" not in payload
    assert payload["frozen_artifact_receipt"]["checkpoint_bytes_saved"] is False
    assert set(payload["student_trainable_state_dict"]) == set(source.trainable_parameter_names)
    assert all(not key.startswith("ecmm_core.") for key in payload["student_trainable_state_dict"])

    torch.manual_seed(99)
    restored = _algorithm(rollout_length=3)
    assert restored.load(copy.deepcopy(payload), strict=True)
    assert restored.num_updates == source.num_updates
    for name, value in source.student.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value, dict(restored.student.named_parameters())[name])
    assert restored.optimizer.state_dict()["state"]

    wrong_config = _algorithm(rollout_length=3, learning_rate=2.0e-2)
    with pytest.raises(ValueError, match="configuration receipt"):
        wrong_config.load(copy.deepcopy(payload))
    strict_label_config = _algorithm(rollout_length=3, strict_teacher_label_checks=True)
    with pytest.raises(ValueError, match="configuration receipt"):
        strict_label_config.load(copy.deepcopy(payload))
    wrong_artifact = _algorithm(
        rollout_length=3,
        frozen_artifact_receipt={
            "checkpoint_sha256": "b" * 64,
            "actor_state_dict_key": "actor_state_dict",
            "checkpoint_bytes_saved": False,
        },
    )
    with pytest.raises(ValueError, match="frozen-artifact receipt"):
        # Keep the live config identical so artifact validation is reached.
        mismatched = copy.deepcopy(payload)
        mismatched["config_receipt"] = wrong_artifact._configuration_receipt()
        wrong_artifact.load(mismatched)


def test_constructor_rejects_legacy_storage_and_unexpected_trainable_core() -> None:
    student = _DummyC07Student()
    student.ecmm_core.action_head.weight.requires_grad_(True)
    storage = M2MSequenceRolloutStorage(
        num_envs=2,
        num_transitions_per_env=2,
        student_obs=_student_obs(2),
        allowed_student_keys=("policy", "strict_frame"),
        hidden_state_shape=(1, 6),
    )
    with pytest.raises(ValueError, match="outside tokenizer/temporal/A-head"):
        M2MLatentActionDistillation(
            student,
            _DummyTeacher(),
            storage,
            loss_config=_loss_config(),
            learning_rate=1.0e-3,
            num_mini_batches=1,
            sequence_length=2,
        )


def test_audit_marks_runner_factory_boundary_and_checkpoint_exclusions() -> None:
    audit = _algorithm().audit()
    assert audit["runner_factory_integrated"] is False
    assert audit["rollout_storage"]["stores_teacher_map"] is False
    assert audit["rollout_storage"]["stores_next_observation"] is False
    assert audit["checkpoint"]["teacher_state_saved"] is False
    assert audit["checkpoint"]["teacher_map_saved"] is False
    assert audit["gradient_boundary"]["frozen_ecmm_trainable_count"] == 0
    assert audit["gradient_boundary"]["teacher_trainable_count"] == 0
