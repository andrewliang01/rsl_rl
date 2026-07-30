# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

import rsl_rl.utils.training_receipt as receipt_module
from rsl_rl.utils.training_receipt import (
    TrainingReceiptError,
    build_checkpoint_sidecar,
    build_embedded_checkpoint_receipt,
    build_training_launch_receipt,
    canonical_training_receipt_json_bytes,
    canonical_training_receipt_sha256,
    checkpoint_parent_record,
    derive_checkpoint_progress,
    parse_canonical_training_receipt_json,
    retain_regular_file,
    validate_checkpoint_progress,
    validate_checkpoint_receipt_chain,
    validate_checkpoint_sidecar,
    validate_embedded_checkpoint_receipt,
    validate_training_launch_receipt,
)


class _UnsafeCheckpointObject:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        expression = (
            f"open({str(self.marker)!r}, 'w', encoding='utf-8')"
            ".write('executed')"
        )
        return eval, (expression,)


def _text_record(payload: str, *, payload_format: str = "canonical_yaml_v1") -> dict:
    encoded = payload.encode("utf-8")
    return {
        "format": payload_format,
        "encoding": "utf-8",
        "payload_utf8": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def _selector_record() -> dict:
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


def _launch_payload(*, resume: dict | None = None) -> dict:
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
    return {
        "task": "MID360-RayTime-Global-K1-Unitree-G1-29dof",
        "seed": 42,
        "training_started_at_utc": "2026-07-31T04:00:00+00:00",
        "argv": [
            "python",
            "scripts/rsl_rl/train.py",
            "physics=physx",
            "--headless",
        ],
        "git": git,
        "configs": {
            "agent": _text_record("seed: 42\n"),
            "env": _text_record("scene:\n  num_envs: 1024\n"),
            "resume_compatibility_sha256": "d" * 64,
        },
        "runtime": {
            "python": {
                "executable": "/opt/python",
                "version": "3.11.9",
                "implementation": "CPython",
            },
            "cuda": {
                "cuda_visible_devices": "0",
                "torch_version": "2.7.0",
                "torch_cuda_version": "12.8",
                "cudnn_version": "91002",
                "device_name": "NVIDIA RTX 5090",
                "device_uuid": "GPU-example",
                "compute_capability": "12.0",
            },
            "physics": "physx",
            "headless": True,
            "device": "cuda:0",
        },
        "schedule": {
            "training_schedule_id": "ray_time_s42_e1024_i20000_v1",
            "num_envs": 1024,
            "num_steps_per_env": 24,
            "max_iterations": 20000,
            "save_interval": 500,
            "transitions_per_update": 1024 * 24,
            "transition_budget": 1024 * 24 * 20000,
        },
        "selector_protocol": _selector_record(),
        "resume": resume
        or {
            "is_resume": False,
            "parent_checkpoint_sha256": None,
            "parent_embedded_receipt_sha256": None,
            "parent_sidecar_payload_sha256": None,
            "parent_updates_completed": None,
            "parent_consumed_transitions": None,
        },
    }


def _launch_receipt() -> dict:
    return build_training_launch_receipt(_launch_payload())


def _progress(iteration: int, *, target: int = 20000) -> dict:
    return derive_checkpoint_progress(
        filename=f"model_{iteration}.pt",
        iteration=iteration,
        num_envs=1024,
        num_steps_per_env=24,
        configured_target_updates=target,
    )


def _chain_entry(
    tmp_path: Path,
    *,
    iteration: int,
    launch: dict,
    parent: dict | None,
) -> tuple[dict, dict]:
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=launch,
        checkpoint_progress=_progress(iteration),
        parent_checkpoint=parent,
    )
    checkpoint = tmp_path / f"model_{iteration}.pt"
    torch.save(
        {
            "training_receipt": embedded,
            "iter": iteration,
            "model_state_dict": {"weight": torch.tensor([float(iteration)])},
        },
        checkpoint,
    )
    sidecar = build_checkpoint_sidecar(
        checkpoint_path=checkpoint,
        embedded_receipt=embedded,
    )
    return embedded, sidecar


def test_canonical_json_round_trip_and_hash_are_stable() -> None:
    value = {"z": [1, True, None, 1.25], "a": {"text": "中"}}
    payload = canonical_training_receipt_json_bytes(value)

    assert payload == b'{"a":{"text":"\\u4e2d"},"z":[1,true,null,1.25]}\n'
    assert parse_canonical_training_receipt_json(payload) == value
    assert canonical_training_receipt_sha256(value) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "payload, match",
    [
        (b'{"a":1,"a":2}\n', "Duplicate"),
        (b'{"a": NaN}\n', "Non-standard"),
        (b'{ "a":1}\n', "canonical"),
        (b'{"a":1}', "canonical"),
    ],
)
def test_canonical_json_rejects_duplicates_nonfinite_and_noncanonical(
    payload: bytes,
    match: str,
) -> None:
    with pytest.raises(TrainingReceiptError, match=match):
        parse_canonical_training_receipt_json(payload)


def test_canonical_json_rejects_non_native_and_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(TrainingReceiptError, match="Non JSON-native"):
        canonical_training_receipt_json_bytes({"bad": (1, 2)})
    with pytest.raises(TrainingReceiptError, match="Cyclic"):
        canonical_training_receipt_json_bytes(cyclic)
    with pytest.raises(TrainingReceiptError, match="Non-finite"):
        canonical_training_receipt_json_bytes({"bad": float("inf")})


def test_launch_receipt_binds_exact_payload_and_schedule() -> None:
    launch = _launch_receipt()

    assert validate_training_launch_receipt(launch) == launch
    assert launch["payload"]["schedule"]["transition_budget"] == 491_520_000

    tampered = copy.deepcopy(launch)
    tampered["payload"]["seed"] = 43
    with pytest.raises(TrainingReceiptError, match="stale or forged"):
        validate_training_launch_receipt(tampered)

    wrong_budget = _launch_payload()
    wrong_budget["schedule"]["transition_budget"] += 1
    with pytest.raises(TrainingReceiptError, match="transition_budget"):
        build_training_launch_receipt(wrong_budget)


def test_launch_receipt_rejects_bool_as_integer_and_unknown_keys() -> None:
    bool_seed = _launch_payload()
    bool_seed["seed"] = True
    with pytest.raises(TrainingReceiptError, match="exact integer"):
        build_training_launch_receipt(bool_seed)

    extra = _launch_payload()
    extra["undeclared"] = "not allowed"
    with pytest.raises(TrainingReceiptError, match="keys must be exactly"):
        build_training_launch_receipt(extra)


def test_launch_receipt_rejects_selector_and_config_byte_tampering() -> None:
    selector = _launch_payload()
    selector["selector_protocol"]["payload_utf8"] = '{"different":true}\n'
    with pytest.raises(TrainingReceiptError, match="byte count or SHA"):
        build_training_launch_receipt(selector)

    config = _launch_payload()
    config["configs"]["agent"]["payload_utf8"] = "seed: 43\n"
    with pytest.raises(TrainingReceiptError, match="byte count or SHA"):
        build_training_launch_receipt(config)


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-utc",
        "2026-07-31T04:00:00",
        "2026-07-31T12:00:00+08:00",
        "2026-07-31 04:00:00+00:00",
    ],
)
def test_launch_receipt_requires_explicit_iso_utc_timestamp(
    timestamp: str,
) -> None:
    payload = _launch_payload()
    payload["training_started_at_utc"] = timestamp

    with pytest.raises(TrainingReceiptError, match="UTC timestamp"):
        build_training_launch_receipt(payload)


def test_resume_parent_transitions_must_match_current_schedule() -> None:
    payload = _launch_payload(
        resume={
            "is_resume": True,
            "parent_checkpoint_sha256": "a" * 64,
            "parent_embedded_receipt_sha256": "b" * 64,
            "parent_sidecar_payload_sha256": "c" * 64,
            "parent_updates_completed": 1000,
            "parent_consumed_transitions": 1001,
        }
    )

    with pytest.raises(
        TrainingReceiptError,
        match=r"parent_updates\*transitions_per_update",
    ):
        build_training_launch_receipt(payload)


def test_zero_index_progress_and_terminal_transition_accounting() -> None:
    first = _progress(0)
    terminal = _progress(19999)

    assert first["iter"] == 0
    assert first["updates_completed"] == 1
    assert first["consumed_transitions"] == 1024 * 24
    assert terminal["updates_completed"] == 20000
    assert terminal["consumed_transitions"] == 491_520_000
    assert validate_checkpoint_progress(terminal) == terminal

    with pytest.raises(TrainingReceiptError, match="exceeds configured target"):
        _progress(20000)
    with pytest.raises(TrainingReceiptError, match="exact integer"):
        derive_checkpoint_progress(
            filename="model_1.pt",
            iteration=True,
            num_envs=1,
            num_steps_per_env=1,
            configured_target_updates=2,
        )


def test_progress_rejects_filename_and_derived_field_tampering() -> None:
    with pytest.raises(TrainingReceiptError, match="filename iteration"):
        derive_checkpoint_progress(
            filename="model_2.pt",
            iteration=1,
            num_envs=1,
            num_steps_per_env=1,
            configured_target_updates=2,
        )

    tampered = _progress(10)
    tampered["consumed_transitions"] += 1
    with pytest.raises(TrainingReceiptError, match="derived values"):
        validate_checkpoint_progress(tampered)


def test_embedded_progress_must_match_launch_schedule() -> None:
    mismatched_progress = derive_checkpoint_progress(
        filename="model_0.pt",
        iteration=0,
        num_envs=64,
        num_steps_per_env=24,
        configured_target_updates=20000,
    )

    with pytest.raises(TrainingReceiptError, match="launch training schedule"):
        build_embedded_checkpoint_receipt(
            launch_receipt=_launch_receipt(),
            checkpoint_progress=mismatched_progress,
        )


def test_embedded_receipt_is_weights_only_safe_and_has_no_self_file_hash(
    tmp_path: Path,
) -> None:
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=_launch_receipt(),
        checkpoint_progress=_progress(0),
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"training_receipt": embedded, "iter": 0}, checkpoint)

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert validate_embedded_checkpoint_receipt(
        loaded["training_receipt"],
        checkpoint_filename=checkpoint.name,
    ) == embedded
    assert "checkpoint" not in embedded
    assert "checkpoint_sha256" not in embedded


def test_embedded_receipt_rejects_hash_progress_and_parent_tampering(
    tmp_path: Path,
) -> None:
    launch = _launch_receipt()
    first_embedded, first_sidecar = _chain_entry(
        tmp_path,
        iteration=0,
        launch=launch,
        parent=None,
    )
    parent = checkpoint_parent_record(
        embedded_receipt=first_embedded,
        sidecar=first_sidecar,
    )
    second = build_embedded_checkpoint_receipt(
        launch_receipt=launch,
        checkpoint_progress=_progress(1),
        parent_checkpoint=parent,
    )

    tampered = copy.deepcopy(second)
    tampered["checkpoint_progress"]["updates_completed"] = 3
    with pytest.raises(TrainingReceiptError):
        validate_embedded_checkpoint_receipt(tampered)

    stale = copy.deepcopy(second)
    stale["parent_checkpoint"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(TrainingReceiptError, match="stale or forged"):
        validate_embedded_checkpoint_receipt(stale)


def test_retained_read_rejects_leaf_and_parent_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.pt"
    target.write_bytes(b"checkpoint")
    leaf_link = tmp_path / "leaf.pt"
    leaf_link.symlink_to(target)

    with pytest.raises(TrainingReceiptError, match="Cannot safely open"):
        retain_regular_file(leaf_link)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    nested = real_parent / "nested.pt"
    nested.write_bytes(b"checkpoint")
    parent_link = tmp_path / "linked"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(TrainingReceiptError, match="Cannot safely open"):
        retain_regular_file(parent_link / "nested.pt")


def test_retained_read_rejects_toctou_visible_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"before")
    original = receipt_module._open_regular_file_no_follow
    calls = 0

    def replace_before_visible_reopen(path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"after!")
            replacement.replace(checkpoint)
        return original(path)

    monkeypatch.setattr(
        receipt_module,
        "_open_regular_file_no_follow",
        replace_before_visible_reopen,
    )
    with pytest.raises(TrainingReceiptError, match="identity changed"):
        retain_regular_file(checkpoint)


def test_sidecar_binds_live_checkpoint_bytes_and_is_archive_portable(
    tmp_path: Path,
) -> None:
    embedded, sidecar = _chain_entry(
        tmp_path,
        iteration=0,
        launch=_launch_receipt(),
        parent=None,
    )
    checkpoint = tmp_path / "model_0.pt"

    assert (
        validate_checkpoint_sidecar(
            sidecar,
            embedded_receipt=embedded,
            checkpoint_path=checkpoint,
        )
        == sidecar
    )
    assert sidecar["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert set(sidecar["checkpoint"]) == {"file_name", "sha256", "bytes"}

    archive = tmp_path / "archive"
    archive.mkdir()
    archived_checkpoint = archive / checkpoint.name
    archived_checkpoint.write_bytes(checkpoint.read_bytes())
    assert (
        validate_checkpoint_sidecar(
            sidecar,
            embedded_receipt=embedded,
            checkpoint_path=archived_checkpoint,
        )
        == sidecar
    )

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(TrainingReceiptError, match="differ from the sidecar"):
        validate_checkpoint_sidecar(
            sidecar,
            embedded_receipt=embedded,
            checkpoint_path=checkpoint,
        )


def test_sidecar_rejects_checkpoint_without_matching_embedded_receipt(
    tmp_path: Path,
) -> None:
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=_launch_receipt(),
        checkpoint_progress=_progress(0),
    )
    changed_payload = _launch_payload()
    changed_payload["seed"] = 43
    changed_payload["training_started_at_utc"] = (
        "2026-07-31T04:00:01+00:00"
    )
    different_embedded = build_embedded_checkpoint_receipt(
        launch_receipt=build_training_launch_receipt(changed_payload),
        checkpoint_progress=_progress(0),
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save(
        {
            "training_receipt": embedded,
            "iter": 0,
            "model_state_dict": {},
        },
        checkpoint,
    )

    with pytest.raises(
        TrainingReceiptError,
        match="differs from the receipt supplied",
    ):
        build_checkpoint_sidecar(
            checkpoint_path=checkpoint,
            embedded_receipt=different_embedded,
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"not a torch checkpoint",
        b"",
    ],
)
def test_sidecar_rejects_non_torch_checkpoint_bytes(
    tmp_path: Path,
    payload: bytes,
) -> None:
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(payload)

    with pytest.raises(TrainingReceiptError):
        build_checkpoint_sidecar(
            checkpoint_path=checkpoint,
            embedded_receipt=build_embedded_checkpoint_receipt(
                launch_receipt=_launch_receipt(),
                checkpoint_progress=_progress(0),
            ),
        )


def test_sidecar_rejects_checkpoint_iter_mismatch(tmp_path: Path) -> None:
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=_launch_receipt(),
        checkpoint_progress=_progress(0),
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save(
        {
            "training_receipt": embedded,
            "iter": 1,
            "model_state_dict": {},
        },
        checkpoint,
    )

    with pytest.raises(TrainingReceiptError, match="Checkpoint iter differs"):
        build_checkpoint_sidecar(
            checkpoint_path=checkpoint,
            embedded_receipt=embedded,
        )


def test_sidecar_weights_only_load_never_executes_unsafe_reducer(
    tmp_path: Path,
) -> None:
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=_launch_receipt(),
        checkpoint_progress=_progress(0),
    )
    marker = tmp_path / "unsafe-reducer-executed"
    checkpoint = tmp_path / "model_0.pt"
    torch.save(
        {
            "training_receipt": embedded,
            "iter": 0,
            "unsafe": _UnsafeCheckpointObject(marker),
        },
        checkpoint,
    )

    with pytest.raises(TrainingReceiptError, match="weights-only-safe"):
        build_checkpoint_sidecar(
            checkpoint_path=checkpoint,
            embedded_receipt=embedded,
        )
    assert not marker.exists()


def test_sidecar_build_binds_single_retained_payload_if_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded = build_embedded_checkpoint_receipt(
        launch_receipt=_launch_receipt(),
        checkpoint_progress=_progress(0),
    )
    checkpoint = tmp_path / "model_0.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save(
        {
            "training_receipt": embedded,
            "iter": 0,
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        checkpoint,
    )
    torch.save(
        {
            "training_receipt": embedded,
            "iter": 0,
            "model_state_dict": {"weight": torch.tensor([2.0])},
        },
        replacement,
    )
    original_retain = receipt_module.retain_regular_file
    retained_sha256: str | None = None
    calls = 0

    def retain_then_replace(path):
        nonlocal calls, retained_sha256
        calls += 1
        payload, record = original_retain(path)
        retained_sha256 = record["sha256"]
        replacement.replace(checkpoint)
        return payload, record

    monkeypatch.setattr(
        receipt_module,
        "retain_regular_file",
        retain_then_replace,
    )
    sidecar = build_checkpoint_sidecar(
        checkpoint_path=checkpoint,
        embedded_receipt=embedded,
    )
    assert calls == 1
    assert sidecar["checkpoint"]["sha256"] == retained_sha256

    monkeypatch.setattr(
        receipt_module,
        "retain_regular_file",
        original_retain,
    )
    with pytest.raises(TrainingReceiptError, match="bytes differ"):
        validate_checkpoint_sidecar(
            sidecar,
            embedded_receipt=embedded,
            checkpoint_path=checkpoint,
        )


def test_sidecar_rejects_metadata_tampering_without_live_file(tmp_path: Path) -> None:
    embedded, sidecar = _chain_entry(
        tmp_path,
        iteration=0,
        launch=_launch_receipt(),
        parent=None,
    )
    tampered = copy.deepcopy(sidecar)
    tampered["checkpoint"]["sha256"] = "f" * 64

    with pytest.raises(TrainingReceiptError, match="stale or forged"):
        validate_checkpoint_sidecar(
            tampered,
            embedded_receipt=embedded,
        )


def test_complete_chain_and_latest_head_reject_rollback(tmp_path: Path) -> None:
    launch = _launch_receipt()
    embedded0, sidecar0 = _chain_entry(
        tmp_path,
        iteration=0,
        launch=launch,
        parent=None,
    )
    embedded1, sidecar1 = _chain_entry(
        tmp_path,
        iteration=1,
        launch=launch,
        parent=checkpoint_parent_record(
            embedded_receipt=embedded0,
            sidecar=sidecar0,
        ),
    )
    entries = [
        {"embedded_receipt": embedded0, "sidecar": sidecar0},
        {"embedded_receipt": embedded1, "sidecar": sidecar1},
    ]

    validated = validate_checkpoint_receipt_chain(
        entries,
        resume_parent_checkpoint_sha256=sidecar1["checkpoint"]["sha256"],
    )
    assert validated["entry_count"] == 2
    assert validated["latest_head"]["updates_completed"] == 2

    with pytest.raises(TrainingReceiptError, match="rollback rejected"):
        validate_checkpoint_receipt_chain(
            entries,
            resume_parent_checkpoint_sha256=sidecar0["checkpoint"]["sha256"],
        )


def test_chain_rejects_missing_link_reordering_and_duplicate(
    tmp_path: Path,
) -> None:
    launch = _launch_receipt()
    embedded0, sidecar0 = _chain_entry(
        tmp_path,
        iteration=0,
        launch=launch,
        parent=None,
    )
    embedded1, sidecar1 = _chain_entry(
        tmp_path,
        iteration=1,
        launch=launch,
        parent=checkpoint_parent_record(
            embedded_receipt=embedded0,
            sidecar=sidecar0,
        ),
    )
    first = {"embedded_receipt": embedded0, "sidecar": sidecar0}
    second = {"embedded_receipt": embedded1, "sidecar": sidecar1}

    with pytest.raises(TrainingReceiptError, match="external parent head"):
        validate_checkpoint_receipt_chain([second])
    with pytest.raises(TrainingReceiptError, match="external parent head"):
        validate_checkpoint_receipt_chain([second, first])
    with pytest.raises(TrainingReceiptError, match="parent mismatch"):
        validate_checkpoint_receipt_chain([first, first])


def test_chain_rejects_unbound_changed_launch(tmp_path: Path) -> None:
    launch = _launch_receipt()
    embedded0, sidecar0 = _chain_entry(
        tmp_path,
        iteration=0,
        launch=launch,
        parent=None,
    )
    parent = checkpoint_parent_record(
        embedded_receipt=embedded0,
        sidecar=sidecar0,
    )
    changed_payload = _launch_payload()
    changed_payload["training_started_at_utc"] = "2026-07-31T05:00:00+00:00"
    changed_launch = build_training_launch_receipt(changed_payload)
    embedded1, sidecar1 = _chain_entry(
        tmp_path,
        iteration=1,
        launch=changed_launch,
        parent=parent,
    )

    with pytest.raises(
        TrainingReceiptError,
        match="changed launch receipt",
    ):
        validate_checkpoint_receipt_chain(
            [
                {"embedded_receipt": embedded0, "sidecar": sidecar0},
                {"embedded_receipt": embedded1, "sidecar": sidecar1},
            ]
        )


def test_external_parent_chain_requires_explicit_opt_in(tmp_path: Path) -> None:
    launch = _launch_receipt()
    embedded0, sidecar0 = _chain_entry(
        tmp_path,
        iteration=0,
        launch=launch,
        parent=None,
    )
    parent = checkpoint_parent_record(
        embedded_receipt=embedded0,
        sidecar=sidecar0,
    )
    resumed_payload = _launch_payload(
        resume={
            "is_resume": True,
            "parent_checkpoint_sha256": parent["checkpoint_sha256"],
            "parent_embedded_receipt_sha256": parent[
                "embedded_receipt_sha256"
            ],
            "parent_sidecar_payload_sha256": parent[
                "sidecar_payload_sha256"
            ],
            "parent_updates_completed": parent["updates_completed"],
            "parent_consumed_transitions": parent[
                "consumed_transitions"
            ],
        }
    )
    resumed_payload["training_started_at_utc"] = "2026-07-31T05:00:00+00:00"
    resumed_launch = build_training_launch_receipt(resumed_payload)
    embedded1, sidecar1 = _chain_entry(
        tmp_path,
        iteration=1,
        launch=resumed_launch,
        parent=parent,
    )
    segment = [{"embedded_receipt": embedded1, "sidecar": sidecar1}]

    with pytest.raises(TrainingReceiptError, match="external parent head"):
        validate_checkpoint_receipt_chain(segment)
    wrong_parent = copy.deepcopy(parent)
    wrong_parent["checkpoint_sha256"] = "f" * 64
    with pytest.raises(TrainingReceiptError, match="differs from"):
        validate_checkpoint_receipt_chain(
            segment,
            external_parent=wrong_parent,
        )
    validated = validate_checkpoint_receipt_chain(
        segment,
        external_parent=parent,
    )
    assert validated["latest_head"]["updates_completed"] == 2
