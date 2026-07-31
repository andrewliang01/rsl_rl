# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

import rsl_rl.utils.formal_training_io as formal_io_module
from rsl_rl.utils import (
    FORMAL_CHECKPOINT_RUN_INSPECTION_CONTRACT,
    inspect_formal_checkpoint_run as exported_inspector,
)
from rsl_rl.utils.formal_training_io import (
    FORMAL_CHECKPOINT_RUN_INSPECTION_SCHEMA_VERSION,
    FormalTrainingIO,
    FormalTrainingIOError,
    inspect_formal_checkpoint_run,
    inspect_formal_resume_parent,
)
from rsl_rl.utils.training_receipt import (
    TrainingReceiptError,
    build_embedded_checkpoint_receipt,
    build_training_launch_receipt,
    canonical_training_receipt_json_bytes,
    canonical_training_receipt_sha256,
    derive_checkpoint_progress,
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
            "candidate_protocol": "formal_inspection_test_v1",
            "ranking_protocol": ["worst", "macro"],
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
    resume: dict[str, Any] | None = None,
    started_at: str = "2026-07-31T04:00:00+00:00",
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
    return build_training_launch_receipt(
        {
            "task": "Formal-Checkpoint-Inspection-Test",
            "seed": 42,
            "training_started_at_utc": started_at,
            "argv": ["python", "train.py", "--headless"],
            "git": git,
            "configs": {
                "agent": _text_record("seed: 42\n"),
                "env": _text_record("scene:\n  num_envs: 2\n"),
                "resume_compatibility_sha256": "d" * 64,
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
                    f"formal_inspection_e2_s1_i{target}_save1"
                ),
                "num_envs": 2,
                "num_steps_per_env": 1,
                "max_iterations": target,
                "save_interval": 1,
                "transitions_per_update": 2,
                "transition_budget": 2 * target,
            },
            "selector_protocol": _selector_record(),
            "resume": resume or _fresh_resume(),
        }
    )


def _publish_iterations(
    formal_io: FormalTrainingIO,
    launch: dict[str, Any],
    iterations: list[int],
    *,
    parent: dict[str, Any] | None = None,
    include_tensor: bool = False,
) -> dict[str, Any]:
    current_parent = parent
    publication: dict[str, Any] | None = None
    target = launch["payload"]["schedule"]["max_iterations"]
    for iteration in iterations:
        progress = derive_checkpoint_progress(
            filename=f"model_{iteration}.pt",
            iteration=iteration,
            num_envs=2,
            num_steps_per_env=1,
            configured_target_updates=target,
        )
        embedded = build_embedded_checkpoint_receipt(
            launch_receipt=launch,
            checkpoint_progress=progress,
            parent_checkpoint=current_parent,
        )

        def saved_dict_factory(
            *,
            receipt: dict[str, Any] = embedded,
            stored_iteration: int = iteration,
        ) -> dict[str, Any]:
            result: dict[str, Any] = {
                "iter": stored_iteration,
                "training_receipt": receipt,
                "infos": {"inspection_fixture": True},
            }
            if include_tensor:
                result["model_state_dict"] = {
                    "weight": torch.arange(8, dtype=torch.float32),
                    "quantized_weight": torch.quantize_per_tensor(
                        torch.arange(8, dtype=torch.float32),
                        scale=0.1,
                        zero_point=0,
                        dtype=torch.qint8,
                    ),
                }
            return result

        publication = formal_io.publish_checkpoint(
            checkpoint_path=formal_io.run_dir / f"model_{iteration}.pt",
            embedded_receipt=embedded,
            saved_dict_factory=saved_dict_factory,
        )
        current_parent = publication["parent_record"]
    assert publication is not None
    return publication["parent_record"]


def _fresh_run(
    run_dir: Path,
    *,
    target: int = 3,
    iterations: list[int] | None = None,
    include_tensor: bool = False,
) -> dict[str, Any]:
    launch = _launch_receipt(target=target)
    with FormalTrainingIO(
        run_dir=run_dir,
        launch_receipt=launch,
    ) as formal_io:
        _publish_iterations(
            formal_io,
            launch,
            list(range(target)) if iterations is None else iterations,
            include_tensor=include_tensor,
        )
    return launch


def _resumed_terminal_run(tmp_path: Path) -> Path:
    parent_dir = tmp_path / "parent"
    parent_launch = _launch_receipt(target=4)
    with FormalTrainingIO(
        run_dir=parent_dir,
        launch_receipt=parent_launch,
    ) as parent_io:
        _publish_iterations(parent_io, parent_launch, [0])
    parent_checkpoint = parent_dir / "model_0.pt"
    inspected = inspect_formal_resume_parent(parent_checkpoint)

    child_dir = tmp_path / "child"
    child_launch = _launch_receipt(
        target=4,
        resume=_resume_record(inspected["parent_record"]),
        started_at="2026-07-31T04:01:00+00:00",
    )
    with FormalTrainingIO(
        run_dir=child_dir,
        launch_receipt=child_launch,
    ) as child_io:
        _loaded, parent_record = child_io.load_resume_checkpoint(
            parent_checkpoint,
            map_location="cpu",
        )
        _publish_iterations(
            child_io,
            child_launch,
            [1, 2, 3],
            parent=parent_record,
        )
    return child_dir


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_inspection_is_canonical_deterministic_read_only_and_exported(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    launch = _fresh_run(run_dir)
    before = _tree_snapshot(run_dir)

    first = inspect_formal_checkpoint_run(
        run_dir,
        ["model_0.pt", "model_2.pt"],
    )
    second = exported_inspector(
        run_dir,
        ("model_0.pt", "model_2.pt"),
    )

    assert first == second
    assert _tree_snapshot(run_dir) == before
    assert set(first) == {
        "schema_version",
        "contract",
        "payload",
        "payload_sha256",
    }
    assert (
        first["schema_version"]
        == FORMAL_CHECKPOINT_RUN_INSPECTION_SCHEMA_VERSION
    )
    assert first["contract"] == FORMAL_CHECKPOINT_RUN_INSPECTION_CONTRACT
    assert first["payload_sha256"] == canonical_training_receipt_sha256(
        first["payload"]
    )
    assert (
        parse_canonical_training_receipt_json(
            canonical_training_receipt_json_bytes(first)
        )
        == first
    )

    payload = first["payload"]
    assert payload["required_checkpoint_names"] == [
        "model_0.pt",
        "model_2.pt",
    ]
    assert payload["launch_receipt"]["document"] == launch
    assert payload["chain"]["entry_count"] == 3
    assert payload["chain"]["local_entry_count"] == 3
    assert payload["chain"]["ancestor_entry_count"] == 0
    assert payload["chain"]["ancestor_chain_proof_file"] is None
    assert payload["chain"]["latest_parent_record"][
        "checkpoint_file_name"
    ] == "model_2.pt"
    assert len(payload["chain"]["head_files"]) == 3
    assert [
        item["iteration"] for item in payload["selected_checkpoints"]
    ] == [0, 2]
    for selected in payload["selected_checkpoints"]:
        checkpoint = Path(selected["checkpoint"]["path"])
        sidecar = Path(selected["sidecar"]["path"])
        assert selected["checkpoint"]["sha256"] == hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        assert selected["checkpoint"]["bytes"] == checkpoint.stat().st_size
        assert selected["sidecar"]["sha256"] == hashlib.sha256(
            sidecar.read_bytes()
        ).hexdigest()


def test_inspection_snapshot_is_explicitly_location_bound(
    tmp_path: Path,
) -> None:
    original_dir = tmp_path / "original"
    _fresh_run(original_dir, target=1)
    copied_dir = tmp_path / "copied"
    shutil.copytree(original_dir, copied_dir)

    original = inspect_formal_checkpoint_run(
        original_dir,
        ["model_0.pt"],
    )
    copied = inspect_formal_checkpoint_run(
        copied_dir,
        ["model_0.pt"],
    )

    assert original["payload_sha256"] != copied["payload_sha256"]
    assert original["payload"]["run_dir"] == str(original_dir)
    assert copied["payload"]["run_dir"] == str(copied_dir)
    assert (
        original["payload"]["chain"]["entries_sha256"]
        == copied["payload"]["chain"]["entries_sha256"]
    )
    assert (
        original["payload"]["selected_checkpoints"][0]["checkpoint"][
            "sha256"
        ]
        == copied["payload"]["selected_checkpoints"][0]["checkpoint"][
            "sha256"
        ]
    )


@pytest.mark.parametrize(
    "required",
    [
        [],
        ["model_0.pt", "model_0.pt"],
        ["model_2.pt", "model_0.pt"],
        ["model_00.pt"],
        ["../model_0.pt"],
        ["model_3.pt"],
    ],
)
def test_required_checkpoint_names_are_exact_sorted_unique_and_local(
    tmp_path: Path,
    required: list[str],
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_checkpoint_run(run_dir, required)


def test_partial_run_is_not_admitted_as_terminal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir, target=3, iterations=[0, 1])

    with pytest.raises(
        FormalTrainingIOError,
        match="configured terminal",
    ):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("model_1.pt"),
        Path("checkpoint_receipts/model_1.json"),
        Path(
            "checkpoint_receipts/heads/"
            "head_00000000000000000002.json"
        ),
    ],
)
def test_tamper_of_any_committed_artifact_fails_even_when_unselected(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    target = run_dir / relative_path
    original = target.read_bytes()
    target.write_bytes(original + b" ")

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_checkpoint_run(
            run_dir,
            ["model_0.pt", "model_2.pt"],
        )

    target.write_bytes(original)
    assert inspect_formal_checkpoint_run(
        run_dir,
        ["model_0.pt", "model_2.pt"],
    )["payload"]["chain"]["entry_count"] == 3


@pytest.mark.parametrize(
    "orphan_kind",
    ["checkpoint", "sidecar", "both"],
)
def test_orphan_checkpoint_or_sidecar_is_never_admitted(
    tmp_path: Path,
    orphan_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    if orphan_kind in {"checkpoint", "both"}:
        shutil.copyfile(
            run_dir / "model_2.pt",
            run_dir / "model_3.pt",
        )
    if orphan_kind in {"sidecar", "both"}:
        shutil.copyfile(
            run_dir / "checkpoint_receipts/model_2.json",
            run_dir / "checkpoint_receipts/model_3.json",
        )

    with pytest.raises(
        FormalTrainingIOError,
        match="missing or orphaned",
    ):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("model_1.pt"),
        Path("checkpoint_receipts/model_1.json"),
        Path(
            "checkpoint_receipts/heads/"
            "head_00000000000000000002.json"
        ),
    ],
)
def test_missing_committed_artifact_fails(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    (run_dir / relative_path).unlink()

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


def test_inspection_excludes_live_writer_and_releases_its_own_lock(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    launch = _launch_receipt(target=1)
    owner = FormalTrainingIO(
        run_dir=run_dir,
        launch_receipt=launch,
    )
    try:
        _publish_iterations(owner, launch, [0])
        with pytest.raises(FormalTrainingIOError, match="still locked"):
            inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])
    finally:
        owner.close()

    assert inspect_formal_checkpoint_run(
        run_dir,
        ["model_0.pt"],
    )["payload"]["chain"]["entry_count"] == 1


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("model_1.pt"),
        Path("checkpoint_receipts/model_1.json"),
        Path(
            "checkpoint_receipts/heads/"
            "head_00000000000000000002.json"
        ),
        Path(".formal_training.lock"),
    ],
)
def test_symlinked_committed_artifact_or_lock_fails(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    target = run_dir / relative_path
    backup = target.with_name(f"{target.name}.retained-backup")
    shutil.copyfile(target, backup)
    target.unlink()
    target.symlink_to(backup)

    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


def test_symlinked_run_directory_component_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    alias = tmp_path / "run-alias"
    alias.symlink_to(run_dir, target_is_directory=True)

    with pytest.raises(FormalTrainingIOError, match="symlink"):
        inspect_formal_checkpoint_run(alias, ["model_0.pt"])


def test_drift_after_locked_chain_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    sidecar_path = run_dir / "checkpoint_receipts/model_1.json"
    original_load = FormalTrainingIO._load_committed_chain
    drifted = False

    def load_then_drift(
        self: FormalTrainingIO,
    ) -> dict[str, Any] | None:
        nonlocal drifted
        result = original_load(self)
        if not drifted:
            sidecar_path.write_bytes(sidecar_path.read_bytes() + b" ")
            drifted = True
        return result

    monkeypatch.setattr(
        FormalTrainingIO,
        "_load_committed_chain",
        load_then_drift,
    )
    with pytest.raises(FORMAL_FAILURES):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


def test_inspection_rechecks_visible_lock_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    lock_path = run_dir / ".formal_training.lock"
    original_load = FormalTrainingIO._load_committed_chain
    replaced = False

    def load_then_replace_lock(
        self: FormalTrainingIO,
    ) -> dict[str, Any] | None:
        nonlocal replaced
        result = original_load(self)
        if not replaced:
            lock_path.unlink()
            lock_path.write_bytes(b"replacement lock inode")
            replaced = True
        return result

    monkeypatch.setattr(
        FormalTrainingIO,
        "_load_committed_chain",
        load_then_replace_lock,
    )
    with pytest.raises(
        FormalTrainingIOError,
        match="inspection lock visible identity changed",
    ):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


def test_fresh_ancestor_proof_cannot_appear_during_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir)
    proof_path = run_dir / "checkpoint_receipts/ancestor_chain.json"
    original_load = FormalTrainingIO._load_committed_chain
    appeared = False

    def load_then_publish_ancestor(
        self: FormalTrainingIO,
    ) -> dict[str, Any] | None:
        nonlocal appeared
        result = original_load(self)
        if not appeared:
            assert result is not None
            proof = formal_io_module._build_ancestor_chain_proof(
                result["full_entries"]
            )
            proof_path.write_bytes(
                canonical_training_receipt_json_bytes(proof)
            )
            appeared = True
        return result

    monkeypatch.setattr(
        FormalTrainingIO,
        "_load_committed_chain",
        load_then_publish_ancestor,
    )
    with pytest.raises(
        FormalTrainingIOError,
        match="ancestor chain proof appeared",
    ):
        inspect_formal_checkpoint_run(run_dir, ["model_0.pt"])


def test_inspection_decodes_receipts_on_cpu_for_meta_incompatible_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _fresh_run(run_dir, target=1, include_tensor=True)
    original_torch_load = torch.load
    observed_map_locations: list[Any] = []

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        observed_map_locations.append(kwargs.get("map_location"))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(formal_io_module.torch, "load", recording_load)
    result = inspect_formal_checkpoint_run(
        run_dir,
        ["model_0.pt"],
    )

    assert observed_map_locations
    assert set(observed_map_locations) == {"cpu"}
    assert canonical_training_receipt_json_bytes(result)


def test_resumed_chain_is_described_but_only_local_checkpoints_are_admitted(
    tmp_path: Path,
) -> None:
    child_dir = _resumed_terminal_run(tmp_path)

    inspected = inspect_formal_checkpoint_run(
        child_dir,
        ["model_1.pt", "model_3.pt"],
    )
    chain = inspected["payload"]["chain"]
    assert chain["entry_count"] == 4
    assert chain["ancestor_entry_count"] == 1
    assert chain["local_entry_count"] == 3
    assert chain["ancestor_chain_proof_file"] is not None
    assert chain["latest_parent_record"][
        "checkpoint_file_name"
    ] == "model_3.pt"
    assert inspected["payload"]["launch_receipt"]["document"]["payload"][
        "resume"
    ]["is_resume"] is True

    with pytest.raises(
        FormalTrainingIOError,
        match="committed local checkpoint",
    ):
        inspect_formal_checkpoint_run(
            child_dir,
            ["model_0.pt"],
        )
