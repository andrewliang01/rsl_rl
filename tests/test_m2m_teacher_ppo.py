# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn as nn

from rsl_rl.algorithms import M2MObservedHistoryTeacherPPO, PPO


class _DummyFormalTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.map_encoder = nn.Linear(3, 4)
        self.frozen_ecmm = nn.Linear(4, 2)
        self.frozen_ecmm.requires_grad_(False)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": "dummy_map_only_v1",
            "config_receipt": {
                "frozen_sha256": "a" * 64,
                "frozen_ecmm_weights_saved": False,
            },
            "map_encoder_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.map_encoder.state_dict().items()
            },
        }

    def load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        if set(checkpoint) != {"schema", "config_receipt", "map_encoder_state_dict"}:
            raise ValueError("dummy teacher artifact key mismatch")
        if checkpoint["schema"] != "dummy_map_only_v1":
            raise ValueError("dummy teacher artifact schema mismatch")
        if checkpoint["config_receipt"] != self.checkpoint_state()["config_receipt"]:
            raise ValueError("dummy teacher artifact receipt mismatch")
        self.map_encoder.load_state_dict(checkpoint["map_encoder_state_dict"], strict=True)

    def parameter_audit(self) -> dict[str, Any]:
        trainable = [name for name, value in self.named_parameters() if value.requires_grad]
        return {
            "only_map_encoder_trainable": bool(trainable)
            and all(name.startswith("map_encoder.") for name in trainable)
        }


def _algorithm(*, learning_rate: float = 1.0e-3) -> M2MObservedHistoryTeacherPPO:
    return M2MObservedHistoryTeacherPPO(
        _DummyFormalTeacher(),
        nn.Sequential(nn.Linear(5, 8), nn.ELU(), nn.Linear(8, 1)),
        object(),  # PPO initialization does not inspect storage without an extension.
        num_learning_epochs=2,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        optimizer="adam",
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        normalize_advantage_per_mini_batch=False,
        device="cpu",
        rnd_cfg=None,
        symmetry_cfg=None,
        dwaq_cfg=None,
        multi_gpu_cfg=None,
    )


def _clone_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def test_optimizer_contains_only_map_encoder_and_critic_parameters() -> None:
    algorithm = _algorithm()
    optimizer_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for parameter in (*algorithm.actor.map_encoder.parameters(), *algorithm.critic.parameters())
    }
    frozen_ids = {id(parameter) for parameter in algorithm.actor.frozen_ecmm.parameters()}

    assert optimizer_ids == expected_ids
    assert optimizer_ids.isdisjoint(frozen_ids)
    audit = algorithm.audit()
    assert audit["optimizer_matches_trainable_boundary"] is True
    assert audit["frozen_actor_parameters_in_optimizer"] == 0


def test_checkpoint_roundtrip_restores_map_and_critic_but_not_frozen_ecmm() -> None:
    torch.manual_seed(11)
    source = _algorithm()
    map_before = _clone_parameters(source.actor.map_encoder)
    critic_before = _clone_parameters(source.critic)
    frozen_before = _clone_parameters(source.actor.frozen_ecmm)
    payload = source.save()

    assert set(payload) == {
        "schema",
        "config_receipt",
        "teacher_artifact",
        "critic_state_dict",
        "optimizer_state_dict",
    }
    assert set(payload["teacher_artifact"]) == {
        "schema",
        "config_receipt",
        "map_encoder_state_dict",
    }
    assert "actor_state_dict" not in payload
    assert "frozen_ecmm" not in payload["teacher_artifact"]

    with torch.no_grad():
        for parameter in source.actor.parameters():
            parameter.add_(3.0)
        for parameter in source.critic.parameters():
            parameter.sub_(4.0)
    frozen_mutated = _clone_parameters(source.actor.frozen_ecmm)
    payload_with_runner = {**payload, "iter": 3, "infos": None}
    assert source.load(payload_with_runner, load_cfg=None, strict=True) is True

    for name, parameter in source.actor.map_encoder.named_parameters():
        torch.testing.assert_close(parameter, map_before[name])
    for name, parameter in source.critic.named_parameters():
        torch.testing.assert_close(parameter, critic_before[name])
    for name, parameter in source.actor.frozen_ecmm.named_parameters():
        torch.testing.assert_close(parameter, frozen_mutated[name])
        assert not torch.equal(parameter, frozen_before[name])


def test_adaptive_runtime_learning_rate_roundtrips_without_changing_immutable_receipt() -> None:
    source = _algorithm(learning_rate=1.0e-3)
    source.learning_rate = 2.5e-4
    for group in source.optimizer.param_groups:
        group["lr"] = 2.5e-4
    payload = source.save()

    target = _algorithm(learning_rate=1.0e-3)
    target.load(payload, load_cfg=None, strict=True)

    assert target.initial_learning_rate == 1.0e-3
    assert target.learning_rate == 2.5e-4
    assert {group["lr"] for group in target.optimizer.param_groups} == {2.5e-4}


def test_checkpoint_and_actor_contract_mismatches_fail_closed() -> None:
    algorithm = _algorithm()
    payload = algorithm.save()

    changed_config = copy.deepcopy(payload)
    changed_config["config_receipt"]["gamma"] = 0.5
    with pytest.raises(ValueError, match="configuration receipt"):
        algorithm.load(changed_config, None, True)

    changed_teacher = copy.deepcopy(payload)
    changed_teacher["teacher_artifact"]["schema"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        algorithm.load(changed_teacher, None, True)

    with pytest.raises(ValueError, match="partial"):
        algorithm.load(payload, {"actor": True}, True)
    with pytest.raises(ValueError, match="always strict"):
        algorithm.load(payload, None, False)


def test_generic_ppo_save_contract_is_unchanged() -> None:
    generic = PPO(nn.Linear(2, 2), nn.Linear(2, 1), object(), device="cpu")
    payload = generic.save()
    assert set(payload) == {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    assert "teacher_artifact" not in payload


def test_constructor_rejects_extension_and_non_map_actor_gradients() -> None:
    with pytest.raises(ValueError, match="rnd_cfg"):
        M2MObservedHistoryTeacherPPO(
            _DummyFormalTeacher(),
            nn.Linear(2, 1),
            object(),
            rnd_cfg={"learning_rate": 1.0e-3},
        )

    bad_actor = _DummyFormalTeacher()
    bad_actor.frozen_ecmm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="only map_encoder"):
        M2MObservedHistoryTeacherPPO(
            bad_actor,
            nn.Linear(2, 1),
            object(),
            learning_rate=1.0e-3,
        )
