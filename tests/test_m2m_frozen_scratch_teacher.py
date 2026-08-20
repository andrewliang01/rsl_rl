"""Full-checkpoint and frozen-control contracts for scratch M2M teachers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import M2MFrozenScratchTeacherCore, M2MObservedHistoryScratchTeacher


_MAP = "teacher_map"


def _obs(batch: int = 3) -> TensorDict:
    teacher_map = torch.ones(batch, 1, 2, 16, 96)
    teacher_map[:, :, 0] = torch.linspace(0.1, 1.8, 16 * 96).reshape(1, 1, 16, 96)
    return TensorDict(
        {"policy": torch.randn(batch, 96), _MAP: teacher_map},
        batch_size=[batch],
    )


def _model(obs: TensorDict | None = None) -> M2MObservedHistoryScratchTeacher:
    sample = _obs() if obs is None else obs
    return M2MObservedHistoryScratchTeacher(
        sample,
        {"teacher": ["policy", _MAP]},
        "teacher",
        29,
        teacher_map_set=_MAP,
        proprio_sets=["policy"],
        map_contract={
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
            "hash_capacity": 1024,
            "hash_max_probes": 16,
        },
    )


def _artifact(tmp_path: Path, model: M2MObservedHistoryScratchTeacher) -> tuple[Path, str]:
    path = tmp_path / "scratch-teacher.pt"
    torch.save(
        {
            "actor_state_dict": model.state_dict(),
            "critic_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 19999,
        },
        path,
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_actor_load_is_exact_and_frozen_b_c_preserve_latent_gradient(tmp_path: Path) -> None:
    torch.manual_seed(11)
    obs = _obs()
    source = _model(obs).eval()
    expected_a, expected_action = source.predict_latent_and_action_mean(obs)
    path, digest = _artifact(tmp_path, source)

    restored = _model(obs)
    core = M2MFrozenScratchTeacherCore(
        restored,
        checkpoint_path=path,
        expected_sha256=digest,
    )
    actual_a, actual_action = core.actor.predict_latent_and_action_mean(obs)
    torch.testing.assert_close(actual_a, expected_a)
    torch.testing.assert_close(actual_action, expected_action)
    assert core.checkpoint_sha256 == digest
    assert core.actor_state_dict_key == "actor_state_dict"
    assert not core.actor.training
    assert all(not parameter.requires_grad for parameter in core.parameters())

    latent = actual_a.detach().clone().requires_grad_(True)
    proprio = core.encode_proprio(obs)
    core.action_mean_from_A(proprio, latent).square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert all(parameter.grad is None for parameter in core.parameters())

    core.train(True)
    assert not core.actor.training
    assert core.parameter_audit()["all_parameters_frozen"] is True


def test_digest_is_verified_before_torch_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    path, _ = _artifact(tmp_path, model)
    loaded: list[Path] = []
    original = torch.load

    def recording_load(candidate: str | Path, *args: Any, **kwargs: Any) -> Any:
        loaded.append(Path(candidate).resolve())
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        M2MFrozenScratchTeacherCore(
            _model(),
            checkpoint_path=path,
            expected_sha256="0" * 64,
        )
    assert path.resolve() not in loaded


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "dtype"])
def test_full_actor_state_is_strictly_checked(tmp_path: Path, mutation: str) -> None:
    model = _model()
    state = dict(model.state_dict())
    key = next(iter(state))
    if mutation == "missing":
        state.pop(key)
    elif mutation == "unexpected":
        state["forbidden.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = torch.zeros(*state[key].shape, 1, dtype=state[key].dtype)
    else:
        state[key] = state[key].to(torch.float64)
    path = tmp_path / f"bad-{mutation}.pt"
    torch.save({"actor_state_dict": state}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, TypeError), match="state"):
        M2MFrozenScratchTeacherCore(
            _model(),
            checkpoint_path=path,
            expected_sha256=digest,
        )
