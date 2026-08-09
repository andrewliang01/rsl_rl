# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rsl_rl.utils.formal_training_io as formal_io_module
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.formal_training_io import (
    FormalTrainingIOError,
    inspect_formal_resume_parent,
)
from rsl_rl.utils.training_receipt import (
    TrainingReceiptError,
    build_embedded_checkpoint_receipt,
    build_training_launch_receipt,
    canonical_training_receipt_json_bytes,
    canonical_training_receipt_sha256,
    checkpoint_parent_record,
    parse_canonical_training_receipt_json,
)


FORMAL_FAILURES = (FormalTrainingIOError, TrainingReceiptError)


def _text_record(payload: str) -> dict[str, Any]:
    encoded = payload.encode("utf-8")
    return {
        "format": "canonical_yaml_v1",
        "encoding": "utf-8",
        "payload_utf8": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def _selector_record() -> dict[str, Any]:
    payload = canonical_training_receipt_json_bytes(
        {
            "candidate_protocol": "fixed_v1",
            "ranking_protocol": ["worst_three", "worst", "macro"],
        }
    )
    return {
        "contract": "ray_time_selector_protocol_v1",
        "encoding": "canonical-json-utf8-v1",
        "payload_utf8": payload.decode("utf-8"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _fresh_resume() -> dict[str, Any]:
    return {
        "is_resume": False,
        "parent_checkpoint_sha256": None,
        "parent_embedded_receipt_sha256": None,
        "parent_sidecar_payload_sha256": None,
        "parent_updates_completed": None,
        "parent_consumed_transitions": None,
    }


def _resume_record(parent: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_resume": True,
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "parent_embedded_receipt_sha256": parent[
            "embedded_receipt_sha256"
        ],
        "parent_sidecar_payload_sha256": parent[
            "sidecar_payload_sha256"
        ],
        "parent_updates_completed": parent["updates_completed"],
        "parent_consumed_transitions": parent["consumed_transitions"],
    }


def _launch_receipt(
    *,
    target: int,
    save_interval: int,
    resume: dict[str, Any] | None = None,
    started_at: str = "2026-07-31T04:00:00+00:00",
    compatibility_hash: str = "d" * 64,
) -> dict[str, Any]:
    git = {
        name: {
            "repository_root": f"/workspace/{name}",
            "head": character * 40,
            "tree": character * 40,
            "branch": "main",
            "clean": True,
            "source_state_sha256": character * 64,
        }
        for name, character in (
            ("lab_pro", "a"),
            ("rsl_rl", "b"),
            ("IsaacLab", "c"),
        )
    }
    payload = {
        "task": "Formal-Runner-Test",
        "seed": 42,
        "training_started_at_utc": started_at,
        "argv": ["python", "train.py", "--headless"],
        "git": git,
        "configs": {
            "agent": _text_record("seed: 42\n"),
            "env": _text_record("scene:\n  num_envs: 2\n"),
            "resume_compatibility_sha256": compatibility_hash,
        },
        "runtime": {
            "python": {
                "executable": "/opt/python",
                "version": "3.11.9",
                "implementation": "CPython",
            },
            "cuda": {
                "cuda_visible_devices": "cpu",
                "torch_version": "2.7.0",
                "torch_cuda_version": "12.8",
                "cudnn_version": "91002",
                "device_name": "CPU test double",
                "device_uuid": "none",
                "compute_capability": "none",
            },
            "physics": "physx",
            "headless": True,
            "device": "cpu",
        },
        "schedule": {
            "training_schedule_id": (
                f"formal_test_e2_s1_i{target}_save{save_interval}"
            ),
            "num_envs": 2,
            "num_steps_per_env": 1,
            "max_iterations": target,
            "save_interval": save_interval,
            "transitions_per_update": 2,
            "transition_budget": 2 * target,
        },
        "selector_protocol": _selector_record(),
        "resume": resume or _fresh_resume(),
    }
    return build_training_launch_receipt(payload)


class _FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.cfg = SimpleNamespace(seed=42)
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.max_episode_length = 10
        self.step_calls = 0

    def get_observations(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 1))

    def step(
        self,
        _actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.step_calls += 1
        return (
            torch.zeros((self.num_envs, 1)),
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.long),
            {},
        )


class _FakeOptimizer:
    def __init__(self, learning_rate: float) -> None:
        self.param_groups = [{"lr": learning_rate}]

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": {},
            "param_groups": [
                {"lr": group["lr"]} for group in self.param_groups
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.param_groups = [
            {"lr": group["lr"]} for group in state["param_groups"]
        ]


class _FakeAlgorithm:
    def __init__(
        self,
        *,
        fail_update: bool = False,
        fail_save: bool = False,
        fail_load: bool = False,
        load_result: bool = True,
    ) -> None:
        self.fail_update = fail_update
        self.fail_save = fail_save
        self.fail_load = fail_load
        self.load_result = load_result
        self.update_calls = 0
        self.state_value = 0.0
        self.save_calls = 0
        self.load_calls = 0
        self.learning_rate = 1.0e-3
        self.optimizer = _FakeOptimizer(self.learning_rate)
        self.update_entry_learning_rates: list[float] = []
        self.rnd = None
        self.dwaq = None
        self.amp_discriminator = None
        self.loaded: dict[str, Any] | None = None
        self._policy = SimpleNamespace(output_std=torch.ones(1))

    def train_mode(self) -> None:
        pass

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros((observations.shape[0], 1))

    def process_env_step(self, *_args: Any) -> None:
        pass

    def compute_returns(self, _observations: torch.Tensor) -> None:
        pass

    def update(self) -> dict[str, float]:
        if self.fail_update:
            raise RuntimeError("algorithm update failed")
        self.update_entry_learning_rates.append(self.learning_rate)
        self.update_calls += 1
        self.state_value += 1.0
        return {"loss": float(self.update_calls)}

    def get_policy(self) -> SimpleNamespace:
        return self._policy

    def save(self) -> dict[str, Any]:
        if self.fail_save:
            raise RuntimeError("algorithm save failed")
        self.save_calls += 1
        return {
            "model_state_dict": {
                "weight": torch.tensor([self.state_value])
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(
        self,
        loaded: dict[str, Any],
        _load_cfg: dict[str, Any] | None,
        _strict: bool,
    ) -> bool:
        self.load_calls += 1
        if self.fail_load:
            raise RuntimeError("algorithm load failed")
        self.loaded = loaded
        self.state_value = float(
            loaded["model_state_dict"]["weight"].item()
        )
        self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        return self.load_result


class _FakeLogger:
    def __init__(
        self,
        log_dir: Path,
        *,
        writer_enabled: bool = False,
        fail_at: str | None = None,
    ) -> None:
        self.log_dir = str(log_dir)
        self.writer_enabled = writer_enabled
        self.fail_at = fail_at
        self.writer: object | None = None
        self.log_calls: list[int] = []
        self.saved_models: list[tuple[str, int]] = []
        self.stopped = False

    def init_logging_writer(self) -> None:
        if self.fail_at == "init":
            raise RuntimeError("logger init failed")
        self.writer = object() if self.writer_enabled else None

    def process_env_step(self, *_args: Any) -> None:
        pass

    def log(self, *, it: int, **_kwargs: Any) -> None:
        if self.fail_at == "log":
            raise RuntimeError("logger log failed")
        self.log_calls.append(it)

    def save_model(self, path: str, iteration: int) -> None:
        if self.fail_at == "save_model":
            raise RuntimeError("logger save_model failed")
        self.saved_models.append((path, iteration))

    def stop_logging_writer(self) -> None:
        if self.fail_at == "stop":
            raise RuntimeError("logger stop failed")
        self.stopped = True


def _runner(
    run_dir: Path,
    *,
    target: int,
    save_interval: int,
    writer_enabled: bool = False,
    logger_failure: str | None = None,
    algorithm: _FakeAlgorithm | None = None,
) -> OnPolicyRunner:
    runner = object.__new__(OnPolicyRunner)
    runner.env = _FakeEnv()
    runner.cfg = {
        "seed": 42,
        "num_steps_per_env": 1,
        "max_iterations": target,
        "save_interval": save_interval,
        "check_for_nan": False,
        "algorithm": {
            "class_name": "PPO",
            "rnd_cfg": None,
            "dwaq_cfg": None,
            "amp_cfg": None,
        },
    }
    runner.device = "cpu"
    runner.is_distributed = False
    runner.gpu_world_size = 1
    runner.gpu_global_rank = 0
    runner.alg = algorithm or _FakeAlgorithm()
    runner.logger = _FakeLogger(
        run_dir,
        writer_enabled=writer_enabled,
        fail_at=logger_failure,
    )
    runner.current_learning_iteration = 0
    runner._formal_training_io = None
    runner._formal_launch_receipt = None
    runner._formal_launch_receipt_bytes = None
    runner._formal_schedule = None
    runner._formal_parent_checkpoint = None
    runner._formal_last_local_embedded_receipt = None
    runner._formal_updates_completed = 0
    runner._formal_resume_loaded = False
    return runner


def _configure(
    runner: OnPolicyRunner,
    launch: dict[str, Any],
    run_dir: Path,
) -> None:
    runner.configure_formal_training(
        {
            "launch_receipt": launch,
            "run_dir": str(run_dir),
        }
    )


def _save_partial_parent(
    run_dir: Path,
    *,
    target: int = 3,
    save_interval: int = 1,
) -> tuple[OnPolicyRunner, dict[str, Any]]:
    runner = _runner(
        run_dir,
        target=target,
        save_interval=save_interval,
    )
    launch = _launch_receipt(
        target=target,
        save_interval=save_interval,
    )
    _configure(runner, launch, run_dir)
    runner._formal_updates_completed = 1
    runner.current_learning_iteration = 0
    runner.save(str(run_dir / "model_0.pt"))
    runner.close_formal_training()
    return runner, launch


def _leave_orphan_before_head(
    *,
    run_dir: Path,
    launch: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    algorithm: _FakeAlgorithm | None = None,
) -> Path:
    runner = _runner(
        run_dir,
        target=1,
        save_interval=1,
        algorithm=algorithm,
    )
    _configure(runner, launch, run_dir)
    runner._formal_updates_completed = 1
    runner.current_learning_iteration = 0
    original_publish = formal_io_module._publish_new_bytes

    def fail_before_head(path: Path, payload: bytes) -> None:
        if path.parent.name == "heads":
            raise RuntimeError("simulated orphan before head")
        original_publish(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_before_head,
    )
    checkpoint = run_dir / "model_0.pt"
    with pytest.raises(RuntimeError, match="orphan before head"):
        runner.save(str(checkpoint))
    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish,
    )
    return checkpoint


def _save_resumed_generation(
    *,
    parent_checkpoint: Path,
    run_dir: Path,
    started_at: str,
    target: int = 4,
) -> Path:
    inspected = inspect_formal_resume_parent(parent_checkpoint)
    runner = _runner(run_dir, target=target, save_interval=1)
    launch = _launch_receipt(
        target=target,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at=started_at,
    )
    _configure(runner, launch, run_dir)
    runner.load(str(parent_checkpoint))
    next_updates = inspected["parent_record"]["updates_completed"] + 1
    runner.alg.state_value += float(next_updates)
    runner._formal_updates_completed = next_updates
    runner.current_learning_iteration = next_updates - 1
    checkpoint = run_dir / f"model_{next_updates - 1}.pt"
    runner.save(str(checkpoint))
    runner.close_formal_training()
    return checkpoint


def _three_generation_chain(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir(parents=True)
    parent = _runner(parent_dir, target=4, save_interval=1)
    _configure(
        parent,
        _launch_receipt(target=4, save_interval=1),
        parent_dir,
    )
    parent._formal_updates_completed = 1
    parent.current_learning_iteration = 0
    parent_checkpoint = parent_dir / "model_0.pt"
    parent.save(str(parent_checkpoint))
    parent.close_formal_training()

    child_checkpoint = _save_resumed_generation(
        parent_checkpoint=parent_checkpoint,
        run_dir=tmp_path / "child",
        started_at="2026-07-31T04:01:00+00:00",
    )
    grandchild_checkpoint = _save_resumed_generation(
        parent_checkpoint=child_checkpoint,
        run_dir=tmp_path / "grandchild",
        started_at="2026-07-31T04:02:00+00:00",
    )
    return parent_checkpoint, child_checkpoint, grandchild_checkpoint


def _rewrite_bound_ancestor_proof(
    run_dir: Path,
    proof: dict[str, Any],
) -> None:
    rewritten = deepcopy(proof)
    core = {
        key: value
        for key, value in rewritten.items()
        if key != "proof_payload_sha256"
    }
    rewritten["proof_payload_sha256"] = (
        canonical_training_receipt_sha256(core)
    )
    payload = canonical_training_receipt_json_bytes(rewritten)
    proof_path = run_dir / "checkpoint_receipts" / "ancestor_chain.json"
    proof_path.write_bytes(payload)
    proof_record = {
        "file_name": "ancestor_chain.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    previous_head_hash: str | None = None
    heads_dir = run_dir / "checkpoint_receipts" / "heads"
    for head_path in sorted(heads_dir.iterdir()):
        head = parse_canonical_training_receipt_json(
            head_path.read_bytes()
        )
        head["ancestor_chain_proof"] = proof_record
        head["previous_head_payload_sha256"] = previous_head_hash
        head_core = {
            key: value
            for key, value in head.items()
            if key != "head_payload_sha256"
        }
        head["head_payload_sha256"] = (
            canonical_training_receipt_sha256(head_core)
        )
        head_path.write_bytes(canonical_training_receipt_json_bytes(head))
        previous_head_hash = head["head_payload_sha256"]


def _rebuild_proof_entry_with_launch(
    entry: dict[str, Any],
    launch: dict[str, Any],
) -> dict[str, Any]:
    previous_embedded = entry["embedded_receipt"]
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=launch,
        checkpoint_progress=previous_embedded["checkpoint_progress"],
        parent_checkpoint=previous_embedded["parent_checkpoint"],
    )
    previous_sidecar = entry["sidecar"]
    sidecar_core = {
        "schema_version": previous_sidecar["schema_version"],
        "contract": previous_sidecar["contract"],
        "checkpoint": deepcopy(previous_sidecar["checkpoint"]),
        "embedded_receipt_sha256": embedded[
            "embedded_receipt_sha256"
        ],
        "launch_payload_sha256": launch["payload_sha256"],
        "checkpoint_progress": embedded["checkpoint_progress"],
        "parent_checkpoint": embedded["parent_checkpoint"],
    }
    sidecar = {
        **sidecar_core,
        "sidecar_payload_sha256": canonical_training_receipt_sha256(
            sidecar_core
        ),
    }
    return {
        "embedded_receipt": embedded,
        "sidecar": sidecar,
    }


def test_legacy_save_load_and_writer_none_learning_are_unchanged(
    tmp_path: Path,
) -> None:
    save_runner = _runner(tmp_path, target=1, save_interval=1)
    save_runner.current_learning_iteration = 7
    checkpoint = tmp_path / "legacy.pt"
    save_runner.save(str(checkpoint), infos={"legacy": True})

    loaded = torch.load(checkpoint, weights_only=False)
    assert set(loaded) == {
        "model_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    assert loaded["iter"] == 7
    assert loaded["infos"] == {"legacy": True}
    assert save_runner.logger.saved_models == [(str(checkpoint), 7)]

    load_runner = _runner(tmp_path, target=1, save_interval=1)
    assert load_runner.load(str(checkpoint)) == {"legacy": True}
    assert load_runner.current_learning_iteration == 7
    assert load_runner.alg.load_calls == 1

    learn_dir = tmp_path / "legacy-learn"
    learn_dir.mkdir()
    learn_runner = _runner(learn_dir, target=1, save_interval=1)
    learn_runner.learn(1)
    assert learn_runner.alg.update_calls == 1
    assert learn_runner.alg.save_calls == 0
    assert list(learn_dir.iterdir()) == []


def test_formal_fresh_target_model_zero_and_writer_none_save(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, target=1, save_interval=1)
    launch = _launch_receipt(target=1, save_interval=1)
    _configure(runner, launch, tmp_path)

    runner.learn(1)

    checkpoint = tmp_path / "model_0.pt"
    loaded = torch.load(checkpoint, weights_only=True)
    progress = loaded["training_receipt"]["checkpoint_progress"]
    assert progress["iter"] == 0
    assert progress["updates_completed"] == 1
    assert progress["configured_target_updates"] == 1
    assert runner.current_learning_iteration == 0
    assert runner._formal_updates_completed == 1
    assert runner.alg.save_calls == 1
    assert runner.logger.saved_models == [(str(checkpoint), 0)]
    assert runner.logger.writer is None
    assert runner._formal_training_io.lock_held is False
    assert (tmp_path / "checkpoint_receipts" / "model_0.json").is_file()
    assert len(
        list((tmp_path / "checkpoint_receipts" / "heads").iterdir())
    ) == 1


def test_formal_resume_uses_absolute_target_without_repeating_update(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    parent_checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(parent_checkpoint)

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, child_launch, child_dir)
    assert child.load(str(parent_checkpoint)) is None
    assert child._formal_updates_completed == 1
    assert child.current_learning_iteration == 0

    child.learn(3)

    assert child.alg.update_calls == 2
    assert child._formal_updates_completed == 3
    assert child.current_learning_iteration == 2
    assert not (child_dir / "model_0.pt").exists()
    assert (child_dir / "model_1.pt").is_file()
    assert (child_dir / "model_2.pt").is_file()
    terminal = torch.load(child_dir / "model_2.pt", weights_only=True)
    assert terminal["iter"] == 2
    assert terminal["training_receipt"]["checkpoint_progress"][
        "updates_completed"
    ] == 3


def test_formal_external_parent_requirement_needs_live_configuration(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="configured runner"):
        _ = runner.formal_external_parent_load_required

    _configure(
        runner,
        _launch_receipt(target=1, save_interval=1),
        tmp_path,
    )
    assert runner.formal_external_parent_load_required is False
    with pytest.raises(AttributeError):
        runner.formal_external_parent_load_required = True

    runner.close_formal_training()
    with pytest.raises(FormalTrainingIOError, match="held run lock"):
        _ = runner.formal_external_parent_load_required


def test_formal_resumed_launch_requires_parent_until_exact_load(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    parent_checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(parent_checkpoint)

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:00:30+00:00",
    )
    _configure(child, child_launch, child_dir)

    child._formal_updates_completed = 1
    with pytest.raises(FormalTrainingIOError, match="state is inconsistent"):
        _ = child.formal_external_parent_load_required
    child._formal_updates_completed = 0
    child._formal_resume_loaded = True
    with pytest.raises(FormalTrainingIOError, match="state is inconsistent"):
        _ = child.formal_external_parent_load_required
    child._formal_resume_loaded = False
    assert child.formal_external_parent_load_required is True
    child.load(str(parent_checkpoint))
    assert child.formal_external_parent_load_required is False
    child.close_formal_training()


def test_formal_local_recovery_satisfies_external_parent_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    parent_checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(parent_checkpoint)

    child_dir = tmp_path / "child"
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:00:40+00:00",
    )
    first = _runner(child_dir, target=3, save_interval=1)
    _configure(first, child_launch, child_dir)
    assert first.formal_external_parent_load_required is True
    first.load(str(parent_checkpoint))
    assert first.formal_external_parent_load_required is False
    first._formal_updates_completed = 2
    first.current_learning_iteration = 1

    original_publish = formal_io_module._publish_new_bytes

    def fail_before_head(path: Path, payload: bytes) -> None:
        if path.parent.name == "heads":
            raise RuntimeError("simulated resumed orphan before head")
        original_publish(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_before_head,
    )
    with pytest.raises(RuntimeError, match="resumed orphan before head"):
        first.save(str(child_dir / "model_1.pt"))
    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish,
    )

    recovery = _runner(child_dir, target=3, save_interval=1)
    _configure(recovery, child_launch, child_dir)
    assert recovery.alg.load_calls == 1
    assert recovery._formal_updates_completed == 2
    assert recovery.formal_external_parent_load_required is False
    recovery.close_formal_training()


@pytest.mark.parametrize("iteration", [0, 2000, 19999])
def test_formal_checkpoint_filename_matches_zero_index_iteration(
    tmp_path: Path,
    iteration: int,
) -> None:
    run_dir = tmp_path / f"iteration-{iteration}"
    runner = _runner(run_dir, target=20000, save_interval=500)
    launch = _launch_receipt(target=20000, save_interval=500)
    _configure(runner, launch, run_dir)
    runner._formal_updates_completed = iteration + 1
    runner.current_learning_iteration = iteration

    checkpoint = run_dir / f"model_{iteration}.pt"
    runner.save(str(checkpoint))
    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded["iter"] == iteration
    assert loaded["training_receipt"]["checkpoint_progress"][
        "updates_completed"
    ] == iteration + 1
    runner.close_formal_training()


def test_repeated_terminal_save_is_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, target=1, save_interval=1)
    _configure(
        runner,
        _launch_receipt(target=1, save_interval=1),
        tmp_path,
    )
    runner._formal_updates_completed = 1
    runner.current_learning_iteration = 0
    checkpoint = tmp_path / "model_0.pt"

    runner.save(str(checkpoint))
    before = checkpoint.stat()
    before_bytes = checkpoint.read_bytes()
    runner.save(str(checkpoint))

    after = checkpoint.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert checkpoint.read_bytes() == before_bytes
    assert runner.alg.save_calls == 1
    assert runner.logger.saved_models == [(str(checkpoint), 0)]
    assert len(
        list((tmp_path / "checkpoint_receipts" / "heads").iterdir())
    ) == 1
    runner.close_formal_training()

    replay = _runner(tmp_path, target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="already has a committed"):
        _configure(
            replay,
            _launch_receipt(target=1, save_interval=1),
            tmp_path,
        )


def test_preexisting_checkpoint_is_never_overwritten(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"preexisting")
    runner = _runner(tmp_path, target=1, save_interval=1)

    with pytest.raises(FORMAL_FAILURES):
        _configure(
            runner,
            _launch_receipt(target=1, save_interval=1),
            tmp_path,
        )

    assert checkpoint.read_bytes() == b"preexisting"
    assert runner.alg.save_calls == 0
    assert runner._formal_training_io is not None
    assert runner._formal_training_io.lock_held is False


def test_checkpoint_save_recovers_only_consistent_pre_head_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    first = _runner(tmp_path, target=1, save_interval=1)
    _configure(first, launch, tmp_path)
    first._formal_updates_completed = 1
    first.current_learning_iteration = 0
    original_publish = formal_io_module._publish_new_bytes

    def fail_before_head(path: Path, payload: bytes) -> None:
        if path.parent.name == "heads":
            raise RuntimeError("simulated crash before commit marker")
        original_publish(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_before_head,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        first.save(str(tmp_path / "model_0.pt"))
    assert (tmp_path / "model_0.pt").is_file()
    assert (tmp_path / "checkpoint_receipts" / "model_0.json").is_file()
    assert list((tmp_path / "checkpoint_receipts" / "heads").iterdir()) == []
    assert first._formal_training_io.lock_held is False

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish,
    )
    recovery = _runner(tmp_path, target=1, save_interval=1)
    _configure(recovery, launch, tmp_path)
    assert recovery.alg.load_calls == 1
    assert recovery._formal_updates_completed == 1
    assert recovery.current_learning_iteration == 0
    recovery.learn(1)

    assert recovery.alg.save_calls == 0
    assert recovery.alg.update_calls == 0
    assert recovery.logger.saved_models == []
    assert len(
        list((tmp_path / "checkpoint_receipts" / "heads").iterdir())
    ) == 1
    assert recovery._formal_training_io.lock_held is False


def test_recovery_after_existing_head_restores_before_continuing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch_receipt(target=3, save_interval=1)
    first = _runner(tmp_path, target=3, save_interval=1)
    _configure(first, launch, tmp_path)
    first._formal_updates_completed = 1
    first.current_learning_iteration = 0
    first.save(str(tmp_path / "model_0.pt"))
    first._formal_updates_completed = 2
    first.current_learning_iteration = 1
    original_publish = formal_io_module._publish_new_bytes

    def fail_second_head(path: Path, payload: bytes) -> None:
        if path.parent.name == "heads" and "00000000000000000002" in path.name:
            raise RuntimeError("simulated second checkpoint crash")
        original_publish(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_second_head,
    )
    with pytest.raises(RuntimeError, match="second checkpoint crash"):
        first.save(str(tmp_path / "model_1.pt"))
    assert first._formal_training_io.lock_held is False

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish,
    )
    recovery = _runner(tmp_path, target=3, save_interval=1)
    _configure(recovery, launch, tmp_path)
    assert recovery.alg.load_calls == 1
    assert recovery._formal_updates_completed == 2
    assert recovery.current_learning_iteration == 1

    recovery.learn(3)

    assert recovery.alg.update_calls == 1
    assert recovery._formal_updates_completed == 3
    assert recovery.current_learning_iteration == 2
    assert (tmp_path / "model_2.pt").is_file()
    terminal = torch.load(tmp_path / "model_2.pt", weights_only=True)
    torch.testing.assert_close(
        terminal["model_state_dict"]["weight"],
        torch.tensor([1.0]),
    )
    assert len(
        list((tmp_path / "checkpoint_receipts" / "heads").iterdir())
    ) == 3


@pytest.mark.parametrize("load_failure", ["raises", "returns-false"])
def test_recovery_load_failure_cannot_fall_back_to_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_failure: str,
) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    first = _runner(tmp_path, target=1, save_interval=1)
    _configure(first, launch, tmp_path)
    first._formal_updates_completed = 1
    first.current_learning_iteration = 0
    original_publish = formal_io_module._publish_new_bytes

    def fail_before_head(path: Path, payload: bytes) -> None:
        if path.parent.name == "heads":
            raise RuntimeError("simulated orphan")
        original_publish(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_before_head,
    )
    with pytest.raises(RuntimeError, match="simulated orphan"):
        first.save(str(tmp_path / "model_0.pt"))
    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish,
    )

    algorithm = _FakeAlgorithm(
        fail_load=load_failure == "raises",
        load_result=load_failure != "returns-false",
    )
    recovery = _runner(
        tmp_path,
        target=1,
        save_interval=1,
        algorithm=algorithm,
    )
    with pytest.raises((RuntimeError, FormalTrainingIOError)):
        _configure(recovery, launch, tmp_path)

    assert recovery._formal_training_io is not None
    assert recovery._formal_training_io.lock_held is False
    with pytest.raises(FormalTrainingIOError):
        recovery.learn(1)
    assert recovery.alg.update_calls == 0
    assert recovery.alg.save_calls == 0
    assert list((tmp_path / "checkpoint_receipts" / "heads").iterdir()) == []
    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(tmp_path / "model_0.pt")


def test_recovery_rejects_path_replacement_after_first_retained_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    first = _runner(tmp_path, target=1, save_interval=1)
    _configure(first, launch, tmp_path)
    first._formal_updates_completed = 1
    first.current_learning_iteration = 0
    original_publish_bytes = formal_io_module._publish_new_bytes

    def fail_before_sidecar(path: Path, payload: bytes) -> None:
        if (
            path.parent.name == "checkpoint_receipts"
            and path.name == "model_0.json"
        ):
            raise RuntimeError("simulated pre-sidecar crash")
        original_publish_bytes(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_before_sidecar,
    )
    checkpoint = tmp_path / "model_0.pt"
    with pytest.raises(RuntimeError, match="pre-sidecar"):
        first.save(str(checkpoint))
    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish_bytes,
    )

    replacement_dict = torch.load(checkpoint, weights_only=True)
    replacement_dict["model_state_dict"]["weight"] = torch.tensor([999.0])
    replacement = tmp_path / "replacement.pt"
    torch.save(replacement_dict, replacement)
    algorithm = _FakeAlgorithm()
    original_load = algorithm.load

    def replace_after_candidate_load(
        loaded: dict[str, Any],
        load_cfg: dict[str, Any] | None,
        strict: bool,
    ) -> bool:
        result = original_load(loaded, load_cfg, strict)
        replacement.replace(checkpoint)
        return result

    monkeypatch.setattr(
        algorithm,
        "load",
        replace_after_candidate_load,
    )
    recovery = _runner(
        tmp_path,
        target=1,
        save_interval=1,
        algorithm=algorithm,
    )
    with pytest.raises(FormalTrainingIOError, match="changed"):
        _configure(recovery, launch, tmp_path)

    assert algorithm.loaded is not None
    torch.testing.assert_close(
        algorithm.loaded["model_state_dict"]["weight"],
        torch.tensor([0.0]),
    )
    assert list((tmp_path / "checkpoint_receipts" / "heads").iterdir()) == []
    assert recovery._formal_training_io.lock_held is False
    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(checkpoint)


def test_normal_publish_rejects_checkpoint_replacement_before_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, target=1, save_interval=1)
    _configure(
        runner,
        _launch_receipt(target=1, save_interval=1),
        tmp_path,
    )
    runner._formal_updates_completed = 1
    runner.current_learning_iteration = 0
    checkpoint = tmp_path / "model_0.pt"
    original_build_sidecar = formal_io_module.build_checkpoint_sidecar

    def replace_before_sidecar(**kwargs: Any) -> dict[str, Any]:
        replacement_dict = torch.load(checkpoint, weights_only=True)
        replacement_dict["model_state_dict"]["weight"] = torch.tensor(
            [999.0]
        )
        replacement = tmp_path / "replacement.pt"
        torch.save(replacement_dict, replacement)
        replacement.replace(checkpoint)
        return original_build_sidecar(**kwargs)

    monkeypatch.setattr(
        formal_io_module,
        "build_checkpoint_sidecar",
        replace_before_sidecar,
    )
    with pytest.raises(FormalTrainingIOError, match="replaced"):
        runner.save(str(checkpoint))

    assert not (tmp_path / "checkpoint_receipts" / "model_0.json").exists()
    assert list((tmp_path / "checkpoint_receipts" / "heads").iterdir()) == []
    assert runner._formal_training_io.lock_held is False


@pytest.mark.parametrize("corruption", ["missing-model", "invalid-lr"])
def test_invalid_orphan_never_publishes_a_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    checkpoint = _leave_orphan_before_head(
        run_dir=tmp_path,
        launch=launch,
        monkeypatch=monkeypatch,
    )
    sidecar = tmp_path / "checkpoint_receipts" / "model_0.json"
    sidecar.unlink()
    payload = torch.load(checkpoint, weights_only=True)
    if corruption == "missing-model":
        del payload["model_state_dict"]
    else:
        payload["optimizer_state_dict"]["param_groups"][0]["lr"] = float(
            "nan"
        )
    torch.save(payload, checkpoint)

    recovery = _runner(tmp_path, target=1, save_interval=1)
    with pytest.raises((KeyError, FormalTrainingIOError)):
        _configure(recovery, launch, tmp_path)

    assert recovery.alg.load_calls == (
        1 if corruption == "missing-model" else 0
    )
    assert list((tmp_path / "checkpoint_receipts" / "heads").iterdir()) == []
    assert recovery._formal_training_io.lock_held is False
    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(checkpoint)


@pytest.mark.parametrize("artifact", ["checkpoint", "sidecar", "head"])
def test_tampered_parent_is_rejected_before_algorithm_load(
    tmp_path: Path,
    artifact: str,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(checkpoint)
    artifact_path = {
        "checkpoint": checkpoint,
        "sidecar": parent_dir / "checkpoint_receipts" / "model_0.json",
        "head": next(
            (parent_dir / "checkpoint_receipts" / "heads").iterdir()
        ),
    }[artifact]
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, launch, child_dir)

    with pytest.raises(FORMAL_FAILURES):
        child.load(str(checkpoint))

    assert child.alg.load_calls == 0
    assert child._formal_training_io.lock_held is False


@pytest.mark.parametrize(
    ("load_cfg", "strict"),
    [
        ({"model": True}, True),
        (None, False),
    ],
)
def test_formal_resume_rejects_partial_or_non_strict_before_algorithm(
    tmp_path: Path,
    load_cfg: dict[str, Any] | None,
    strict: bool,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(checkpoint)

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, launch, child_dir)

    with pytest.raises(FormalTrainingIOError):
        child.load(str(checkpoint), load_cfg=load_cfg, strict=strict)

    assert child.alg.load_calls == 0
    assert child._formal_training_io.lock_held is False


def test_resume_configuration_tamper_is_rejected_before_algorithm(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(checkpoint)

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    changed_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
        compatibility_hash="e" * 64,
    )
    _configure(child, changed_launch, child_dir)

    with pytest.raises(FORMAL_FAILURES):
        child.load(str(checkpoint))

    assert child.alg.load_calls == 0
    assert child._formal_training_io.lock_held is False


def test_formal_resume_rejects_non_latest_checkpoint_before_algorithm(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent = _runner(parent_dir, target=3, save_interval=1)
    launch = _launch_receipt(target=3, save_interval=1)
    _configure(parent, launch, parent_dir)
    parent._formal_updates_completed = 1
    parent.current_learning_iteration = 0
    parent.save(str(parent_dir / "model_0.pt"))
    parent._formal_updates_completed = 2
    parent.current_learning_iteration = 1
    parent.save(str(parent_dir / "model_1.pt"))
    parent.close_formal_training()
    inspected = inspect_formal_resume_parent(parent_dir / "model_1.pt")

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, child_launch, child_dir)

    with pytest.raises(FORMAL_FAILURES, match="latest"):
        child.load(str(parent_dir / "model_0.pt"))

    assert child.alg.load_calls == 0
    assert child._formal_training_io.lock_held is False


def test_formal_resume_never_uses_unsafe_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(checkpoint)
    original_load = formal_io_module.torch.load
    load_calls = 0

    def guarded_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_calls
        assert kwargs.get("weights_only") is True
        load_calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(formal_io_module.torch, "load", guarded_load)
    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, child_launch, child_dir)
    child.load(str(checkpoint))

    assert load_calls > 0
    assert child.alg.load_calls == 1
    child.close_formal_training()


@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(_FakeAlgorithm(fail_load=True), id="raises"),
        pytest.param(_FakeAlgorithm(load_result=False), id="returns-false"),
    ],
)
def test_formal_algorithm_load_failure_releases_lock(
    tmp_path: Path,
    algorithm: _FakeAlgorithm,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(checkpoint)
    child_dir = tmp_path / "child"
    child = _runner(
        child_dir,
        target=3,
        save_interval=1,
        algorithm=algorithm,
    )
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, child_launch, child_dir)

    with pytest.raises((RuntimeError, FormalTrainingIOError)):
        child.load(str(checkpoint))

    assert algorithm.load_calls == 1
    assert child._formal_updates_completed == 0
    assert child._formal_training_io.lock_held is False


def test_formal_map_location_rejected_before_algorithm_load(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(checkpoint)
    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    _configure(child, child_launch, child_dir)

    with pytest.raises(FormalTrainingIOError, match="map_location"):
        child.load(str(checkpoint), map_location="cuda:0")

    assert child.alg.load_calls == 0
    assert child._formal_training_io.lock_held is False


def test_formal_run_flock_excludes_second_runner(tmp_path: Path) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    first = _runner(tmp_path, target=1, save_interval=1)
    second = _runner(tmp_path, target=1, save_interval=1)
    _configure(first, launch, tmp_path)

    with pytest.raises(FormalTrainingIOError, match="already locked"):
        _configure(second, launch, tmp_path)

    first.close_formal_training()


def test_formal_run_and_lock_symlinks_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    launch = _launch_receipt(target=1, save_interval=1)
    linked_runner = _runner(linked, target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="symlink"):
        _configure(linked_runner, launch, linked)

    run_dir = tmp_path / "lock-link"
    run_dir.mkdir()
    lock_target = tmp_path / "lock-target"
    lock_target.write_bytes(b"")
    (run_dir / ".formal_training.lock").symlink_to(lock_target)
    lock_runner = _runner(run_dir, target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="lock"):
        _configure(lock_runner, launch, run_dir)


@pytest.mark.parametrize(
    ("algorithm_failure", "logger_failure", "expected_updates"),
    [
        ("update", None, 0),
        (None, "init", 0),
        (None, "log", 1),
        ("save", None, 1),
        (None, "save_model", 1),
    ],
)
def test_formal_algorithm_and_logger_exceptions_release_lock_without_fake_head(
    tmp_path: Path,
    algorithm_failure: str | None,
    logger_failure: str | None,
    expected_updates: int,
) -> None:
    algorithm = _FakeAlgorithm(
        fail_update=algorithm_failure == "update",
        fail_save=algorithm_failure == "save",
    )
    runner = _runner(
        tmp_path,
        target=1,
        save_interval=1,
        logger_failure=logger_failure,
        algorithm=algorithm,
    )
    _configure(
        runner,
        _launch_receipt(target=1, save_interval=1),
        tmp_path,
    )

    with pytest.raises(RuntimeError):
        runner.learn(1)

    assert runner._formal_updates_completed == expected_updates
    assert runner._formal_training_io.lock_held is False
    heads = list(
        (tmp_path / "checkpoint_receipts" / "heads").iterdir()
    )
    if logger_failure == "save_model":
        assert len(heads) == 1
    else:
        assert heads == []


def test_formal_invalid_absolute_target_releases_lock(tmp_path: Path) -> None:
    runner = _runner(tmp_path, target=3, save_interval=1)
    _configure(
        runner,
        _launch_receipt(target=3, save_interval=1),
        tmp_path,
    )

    with pytest.raises(FormalTrainingIOError, match="absolute target"):
        runner.learn(2)

    assert runner._formal_updates_completed == 0
    assert runner._formal_training_io.lock_held is False


def test_formal_logger_stop_exception_keeps_committed_progress(
    tmp_path: Path,
) -> None:
    runner = _runner(
        tmp_path,
        target=1,
        save_interval=1,
        writer_enabled=True,
        logger_failure="stop",
    )
    _configure(
        runner,
        _launch_receipt(target=1, save_interval=1),
        tmp_path,
    )

    with pytest.raises(RuntimeError, match="logger stop failed"):
        runner.learn(1)

    assert runner._formal_updates_completed == 1
    assert runner._formal_training_io.lock_held is False
    assert (tmp_path / "model_0.pt").is_file()
    assert len(
        list((tmp_path / "checkpoint_receipts" / "heads").iterdir())
    ) == 1


def test_formal_context_is_exact_single_gpu_and_absolute(
    tmp_path: Path,
) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    runner = _runner(tmp_path, target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="exactly"):
        runner.configure_formal_training(
            {
                "launch_receipt": launch,
                "run_dir": str(tmp_path),
                "extra": True,
            }
        )

    distributed = _runner(tmp_path, target=1, save_interval=1)
    distributed.is_distributed = True
    with pytest.raises(FormalTrainingIOError, match="one non-distributed"):
        _configure(distributed, launch, tmp_path)

    nonzero_rank = _runner(tmp_path, target=1, save_interval=1)
    nonzero_rank.gpu_global_rank = 1
    with pytest.raises(FormalTrainingIOError, match="one non-distributed"):
        _configure(nonzero_rank, launch, tmp_path)

    relative = _runner(Path("relative"), target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError):
        _configure(relative, launch, Path("relative"))


def test_inspect_parent_is_read_only_and_requires_latest_head(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir)
    checkpoint = parent_dir / "model_0.pt"
    before = sorted(path.relative_to(parent_dir) for path in parent_dir.rglob("*"))

    inspected = inspect_formal_resume_parent(checkpoint)

    after = sorted(path.relative_to(parent_dir) for path in parent_dir.rglob("*"))
    assert before == after
    assert inspected["parent_record"]["checkpoint_file_name"] == "model_0.pt"
    assert (
        inspected["parent_launch_receipt"]["payload"]["resume"]["is_resume"]
        is False
    )

    stale_name = parent_dir / "model_1.pt"
    stale_name.write_bytes(checkpoint.read_bytes())
    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(stale_name)


def test_formal_launch_receipt_is_bound_before_model_zero_and_immutable(
    tmp_path: Path,
) -> None:
    first_launch = _launch_receipt(target=2, save_interval=1)
    first = _runner(tmp_path, target=2, save_interval=1)
    _configure(first, first_launch, tmp_path)
    launch_path = tmp_path / "launch_receipt.json"
    expected = canonical_training_receipt_json_bytes(first_launch)
    assert launch_path.read_bytes() == expected
    assert not (tmp_path / "model_0.pt").exists()
    first.close_formal_training()

    different_launch = _launch_receipt(
        target=2,
        save_interval=1,
        started_at="2026-07-31T04:10:00+00:00",
    )
    second = _runner(tmp_path, target=2, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="different launch"):
        _configure(second, different_launch, tmp_path)
    assert launch_path.read_bytes() == expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("num_steps_per_env", 2),
        ("max_iterations", 3),
        ("save_interval", 2),
        ("seed", 43),
    ],
)
def test_formal_schedule_drift_fails_before_first_env_step(
    tmp_path: Path,
    field: str,
    replacement: int,
) -> None:
    runner = _runner(tmp_path, target=2, save_interval=1)
    _configure(
        runner,
        _launch_receipt(target=2, save_interval=1),
        tmp_path,
    )
    runner.cfg[field] = replacement

    with pytest.raises(FormalTrainingIOError, match="changed"):
        runner.learn(2)

    assert runner.env.step_calls == 0
    assert runner.alg.update_calls == 0
    assert runner._formal_training_io.lock_held is False


def test_formal_save_rechecks_frozen_schedule_before_algorithm_save(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, target=2, save_interval=1)
    _configure(
        runner,
        _launch_receipt(target=2, save_interval=1),
        tmp_path,
    )
    runner._formal_updates_completed = 1
    runner.current_learning_iteration = 0
    runner.cfg["save_interval"] = 2

    with pytest.raises(FormalTrainingIOError, match="changed"):
        runner.save(str(tmp_path / "model_0.pt"))

    assert runner.alg.save_calls == 0
    assert not (tmp_path / "model_0.pt").exists()


@pytest.mark.parametrize("extension", ["rnd_cfg", "dwaq_cfg", "amp_cfg"])
def test_formal_v1_rejects_unproven_extension_resume_state(
    tmp_path: Path,
    extension: str,
) -> None:
    runner = _runner(tmp_path, target=1, save_interval=1)
    runner.cfg["algorithm"][extension] = {"enabled": True}

    with pytest.raises(FormalTrainingIOError, match="unproven"):
        _configure(
            runner,
            _launch_receipt(target=1, save_interval=1),
            tmp_path,
        )

    assert not (tmp_path / "launch_receipt.json").exists()


def test_formal_adaptive_lr_restores_from_optimizer_before_first_update(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent = _runner(parent_dir, target=2, save_interval=1)
    parent_launch = _launch_receipt(target=2, save_interval=1)
    _configure(parent, parent_launch, parent_dir)
    restored_rate = 2.5e-4
    parent.alg.learning_rate = restored_rate
    parent.alg.optimizer.param_groups[0]["lr"] = restored_rate
    parent._formal_updates_completed = 1
    parent.current_learning_iteration = 0
    checkpoint = parent_dir / "model_0.pt"
    parent.save(str(checkpoint))
    parent.close_formal_training()
    inspected = inspect_formal_resume_parent(checkpoint)

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=2, save_interval=1)
    child_launch = _launch_receipt(
        target=2,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:11:00+00:00",
    )
    _configure(child, child_launch, child_dir)
    child.load(str(checkpoint))

    assert child.alg.learning_rate == restored_rate
    assert child.alg.optimizer.param_groups[0]["lr"] == restored_rate
    child.learn(2)
    assert child.alg.update_entry_learning_rates == [restored_rate]


def test_formal_crash_recovery_restores_adaptive_lr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch_receipt(target=1, save_interval=1)
    first = _runner(tmp_path, target=1, save_interval=1)
    _configure(first, launch, tmp_path)
    restored_rate = 1.25e-4
    first.alg.learning_rate = restored_rate
    first.alg.optimizer.param_groups[0]["lr"] = restored_rate
    first._formal_updates_completed = 1
    first.current_learning_iteration = 0
    original_publish = formal_io_module._publish_new_bytes

    def fail_before_head(path: Path, payload: bytes) -> None:
        if path.parent.name == "heads":
            raise RuntimeError("simulated LR recovery crash")
        original_publish(path, payload)

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        fail_before_head,
    )
    with pytest.raises(RuntimeError, match="LR recovery crash"):
        first.save(str(tmp_path / "model_0.pt"))
    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_bytes",
        original_publish,
    )

    recovery = _runner(tmp_path, target=1, save_interval=1)
    _configure(recovery, launch, tmp_path)
    assert recovery.alg.learning_rate == restored_rate
    assert recovery.alg.optimizer.param_groups[0]["lr"] == restored_rate
    recovery.close_formal_training()


def test_formal_save_rejects_python_optimizer_lr_divergence(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, target=1, save_interval=1)
    _configure(
        runner,
        _launch_receipt(target=1, save_interval=1),
        tmp_path,
    )
    runner._formal_updates_completed = 1
    runner.current_learning_iteration = 0
    runner.alg.learning_rate = 5.0e-4

    with pytest.raises(FormalTrainingIOError, match="learning_rate"):
        runner.save(str(tmp_path / "model_0.pt"))

    assert runner.alg.save_calls == 0
    assert not (tmp_path / "model_0.pt").exists()


def test_ancestor_chain_proof_is_recursive_portable_and_complete(
    tmp_path: Path,
) -> None:
    parent, child, grandchild = _three_generation_chain(tmp_path)
    assert not (
        parent.parent / "checkpoint_receipts" / "ancestor_chain.json"
    ).exists()
    child_proof = parse_canonical_training_receipt_json(
        (
            child.parent
            / "checkpoint_receipts"
            / "ancestor_chain.json"
        ).read_bytes()
    )
    grandchild_proof = parse_canonical_training_receipt_json(
        (
            grandchild.parent
            / "checkpoint_receipts"
            / "ancestor_chain.json"
        ).read_bytes()
    )
    assert child_proof["entry_count"] == 1
    assert grandchild_proof["entry_count"] == 2

    inspected = inspect_formal_resume_parent(grandchild)
    assert inspected["ancestor_chain_proof"]["entry_count"] == 3
    assert inspected["ancestor_chain_proof"]["latest_parent"] == inspected[
        "parent_record"
    ]

    copied_dir = tmp_path / "copied-grandchild"
    shutil.copytree(grandchild.parent, copied_dir)
    copied = inspect_formal_resume_parent(copied_dir / grandchild.name)
    assert copied["parent_record"] == inspected["parent_record"]
    assert copied["ancestor_chain_proof"] == inspected[
        "ancestor_chain_proof"
    ]


def test_resume_binds_ancestor_proof_before_first_local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _save_partial_parent(parent_dir, target=3)
    parent_checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(parent_checkpoint)
    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    child_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:09:00+00:00",
    )
    _configure(child, child_launch, child_dir)
    child.load(str(parent_checkpoint))
    child._formal_updates_completed = 2
    child.current_learning_iteration = 1

    def fail_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("simulated checkpoint serialization failure")

    monkeypatch.setattr(
        formal_io_module,
        "_publish_new_torch_checkpoint",
        fail_checkpoint,
    )
    with pytest.raises(RuntimeError, match="serialization failure"):
        child.save(str(child_dir / "model_1.pt"))

    proof_path = (
        child_dir / "checkpoint_receipts" / "ancestor_chain.json"
    )
    assert proof_path.is_file()
    proof = parse_canonical_training_receipt_json(proof_path.read_bytes())
    assert proof["latest_parent"] == inspected["parent_record"]
    assert not (child_dir / "model_1.pt").exists()


@pytest.mark.parametrize("tamper", ["missing", "bytes"])
def test_ancestor_chain_proof_missing_or_byte_tamper_is_rejected(
    tmp_path: Path,
    tamper: str,
) -> None:
    _parent, _child, grandchild = _three_generation_chain(tmp_path)
    proof_path = (
        grandchild.parent
        / "checkpoint_receipts"
        / "ancestor_chain.json"
    )
    if tamper == "missing":
        proof_path.unlink()
    else:
        proof_path.write_bytes(proof_path.read_bytes() + b" ")

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(grandchild)


@pytest.mark.parametrize("tamper", ["truncated", "reordered", "stale"])
def test_ancestor_chain_semantic_tamper_is_rejected_even_after_reseal(
    tmp_path: Path,
    tamper: str,
) -> None:
    _parent, _child, grandchild = _three_generation_chain(tmp_path)
    proof_path = (
        grandchild.parent
        / "checkpoint_receipts"
        / "ancestor_chain.json"
    )
    proof = parse_canonical_training_receipt_json(proof_path.read_bytes())
    if tamper == "truncated":
        proof["entries"] = proof["entries"][1:]
        proof["entry_count"] = 1
        proof["latest_parent"] = deepcopy(
            proof["entries"][-1]["embedded_receipt"][
                "parent_checkpoint"
            ]
        )
    elif tamper == "reordered":
        proof["entries"] = list(reversed(proof["entries"]))
    else:
        proof["entries"] = proof["entries"][:1]
        proof["entry_count"] = 1
        proof["latest_parent"] = checkpoint_parent_record(
            embedded_receipt=proof["entries"][0]["embedded_receipt"],
            sidecar=proof["entries"][0]["sidecar"],
        )
    _rewrite_bound_ancestor_proof(grandchild.parent, proof)

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(grandchild)


def test_ancestor_chain_forked_proof_is_rejected_even_after_reseal(
    tmp_path: Path,
) -> None:
    parent, _child, grandchild = _three_generation_chain(tmp_path)
    sibling_checkpoint = _save_resumed_generation(
        parent_checkpoint=parent,
        run_dir=tmp_path / "sibling-child",
        started_at="2026-07-31T04:03:00+00:00",
    )
    sibling_proof = inspect_formal_resume_parent(sibling_checkpoint)[
        "ancestor_chain_proof"
    ]
    _rewrite_bound_ancestor_proof(grandchild.parent, sibling_proof)

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_resume_parent(grandchild)


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("task", "field task"),
        ("seed", "field seed"),
        ("git", "field git"),
        ("runtime", "field runtime"),
        ("schedule", "field schedule"),
        ("selector", "field selector_protocol"),
        ("config", "compatibility hash"),
    ],
)
def test_detached_ancestor_proof_rejects_incompatible_middle_generation(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    _parent, _child, grandchild = _three_generation_chain(tmp_path)
    proof_path = (
        grandchild.parent
        / "checkpoint_receipts"
        / "ancestor_chain.json"
    )
    proof = parse_canonical_training_receipt_json(proof_path.read_bytes())
    child_entry = proof["entries"][1]
    child_payload = deepcopy(
        child_entry["embedded_receipt"]["launch_receipt"]["payload"]
    )
    if mutation == "task":
        child_payload["task"] = "Incompatible-Task"
    elif mutation == "seed":
        child_payload["seed"] = 43
    elif mutation == "git":
        child_payload["git"]["lab_pro"]["branch"] = "incompatible"
    elif mutation == "runtime":
        child_payload["runtime"]["python"]["version"] = "3.12.1"
    elif mutation == "schedule":
        child_payload["schedule"]["training_schedule_id"] = (
            "incompatible_schedule"
        )
    elif mutation == "selector":
        selector_payload = canonical_training_receipt_json_bytes(
            {
                "candidate_protocol": "fixed_v1",
                "ranking_protocol": ["macro", "worst"],
            }
        )
        child_payload["selector_protocol"] = {
            "contract": "ray_time_selector_protocol_v1",
            "encoding": "canonical-json-utf8-v1",
            "payload_utf8": selector_payload.decode("utf-8"),
            "sha256": hashlib.sha256(selector_payload).hexdigest(),
            "bytes": len(selector_payload),
        }
    else:
        child_payload["configs"]["resume_compatibility_sha256"] = (
            "e" * 64
        )
    changed_launch = build_training_launch_receipt(child_payload)
    changed_entry = _rebuild_proof_entry_with_launch(
        child_entry,
        changed_launch,
    )
    proof["entries"][1] = changed_entry
    proof["latest_parent"] = checkpoint_parent_record(
        embedded_receipt=changed_entry["embedded_receipt"],
        sidecar=changed_entry["sidecar"],
    )
    proof_core = {
        key: value
        for key, value in proof.items()
        if key != "proof_payload_sha256"
    }
    proof["proof_payload_sha256"] = canonical_training_receipt_sha256(
        proof_core
    )

    with pytest.raises(FORMAL_FAILURES, match=error_match):
        formal_io_module._validate_ancestor_chain_proof(proof)


def test_fresh_run_rejects_preexisting_ancestor_proof(
    tmp_path: Path,
) -> None:
    _parent, child, _grandchild = _three_generation_chain(
        tmp_path / "source"
    )
    source_proof = (
        child.parent / "checkpoint_receipts" / "ancestor_chain.json"
    )
    fresh_dir = tmp_path / "fresh"
    fresh = _runner(fresh_dir, target=1, save_interval=1)
    launch = _launch_receipt(target=1, save_interval=1)
    _configure(fresh, launch, fresh_dir)
    fresh.close_formal_training()
    receipts = fresh_dir / "checkpoint_receipts"
    shutil.copy2(source_proof, receipts / "ancestor_chain.json")

    retry = _runner(fresh_dir, target=1, save_interval=1)
    with pytest.raises(FormalTrainingIOError, match="Fresh"):
        _configure(retry, launch, fresh_dir)


def test_launch_resume_binding_rejects_latest_parent_mismatch(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    parent = _runner(parent_dir, target=3, save_interval=1)
    parent_launch = _launch_receipt(target=3, save_interval=1)
    _configure(parent, parent_launch, parent_dir)
    parent._formal_updates_completed = 1
    parent.current_learning_iteration = 0
    first_checkpoint = parent_dir / "model_0.pt"
    parent.save(str(first_checkpoint))
    first_head = parse_canonical_training_receipt_json(
        (
            parent_dir
            / "checkpoint_receipts"
            / "heads"
            / "head_00000000000000000001.json"
        ).read_bytes()
    )
    stale_parent = first_head["checkpoint_parent_record"]
    parent._formal_updates_completed = 2
    parent.current_learning_iteration = 1
    latest_checkpoint = parent_dir / "model_1.pt"
    parent.save(str(latest_checkpoint))
    parent.close_formal_training()

    child_dir = tmp_path / "child"
    child = _runner(child_dir, target=3, save_interval=1)
    stale_launch = _launch_receipt(
        target=3,
        save_interval=1,
        resume=_resume_record(stale_parent),
        started_at="2026-07-31T04:12:00+00:00",
    )
    _configure(child, stale_launch, child_dir)

    with pytest.raises(FORMAL_FAILURES, match="exact resume parent"):
        child.load(str(latest_checkpoint))

    assert child.alg.load_calls == 0
    assert child._formal_training_io.lock_held is False
