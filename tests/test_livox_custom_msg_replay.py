# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import json
import numpy as np
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from rsl_rl.utils.livox_custom_msg_replay import (
    LIVOX_REPLAY_METADATA_FILE,
    LIVOX_REPLAY_SYNTHETIC_FORMAT,
    LivoxCustomMsgReplayError,
    LivoxCustomMsgReplayWindow,
    load_livox_custom_msg_replay_artifact,
    replay_livox_custom_msg_artifact,
    validate_livox_custom_msg_replay_receipt,
    write_livox_custom_msg_replay_artifact,
)
from rsl_rl.utils.ray_time_deployment_manifest import (
    canonical_json_bytes,
    serialize_ray_time_deployment_manifest,
)
from tests.test_ray_time_deployment_manifest import _manifest

_CLOCK = "CLOCK_MONOTONIC_RAW:boot-fixture"


def _window(index: int, timebase_ns: int) -> LivoxCustomMsgReplayWindow:
    xyz = np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.25)), dtype="<f4")
    offsets = np.asarray((0, 50_000_000), dtype="<u4")
    return LivoxCustomMsgReplayWindow(
        window_index=index,
        timebase_ns=timebase_ns,
        offset_time_ns=offsets,
        xyz_m=xyz,
        capture_end_time_ns=timebase_ns + 100_000_000,
        received_time_ns=timebase_ns + 110_000_000,
        source_point_count=2,
        source_message_count=1,
    )


def _artifact(
    tmp_path: Path,
    *,
    windows: Sequence[LivoxCustomMsgReplayWindow] | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = _manifest(checkpoint, history_length=5, variant="Global")
    manifest_path = tmp_path / "deployment-input.json"
    manifest_path.write_bytes(serialize_ray_time_deployment_manifest(manifest))
    source = tmp_path / "synthetic-custom-msg.capture"
    source.write_bytes(b"synthetic fixture; this is not a real Livox recording")
    return write_livox_custom_msg_replay_artifact(
        tmp_path / "replay",
        raw_recording_path=source,
        deployment_manifest_path=manifest_path,
        windows=windows or (_window(0, 1_000_000_000), _window(2, 1_200_000_000)),
        monotonic_clock_domain=_CLOCK,
        source_recording_format=LIVOX_REPLAY_SYNTHETIC_FORMAT,
        real_data_present=False,
    )


def _read_metadata(root: Path) -> dict:
    return json.loads((root / LIVOX_REPLAY_METADATA_FILE).read_text())


def _reseal(root: Path, value: dict) -> None:
    payload = {key: item for key, item in value.items() if key != "integrity"}
    value["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "RFC8259 JSON; UTF-8; sorted keys; no whitespace; no NaN",
        "payload_scope": "all top-level fields except integrity",
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    (root / LIVOX_REPLAY_METADATA_FILE).write_bytes(canonical_json_bytes(value) + b"\n")


def _replace_archive(root: Path, position: int, **arrays: np.ndarray) -> None:
    metadata = _read_metadata(root)
    record = metadata["windows"][position]
    archive = root / record["archive_file"]
    np.savez(archive, **arrays)
    raw = archive.read_bytes()
    record["archive_size_bytes"] = len(raw)
    record["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    _reseal(root, metadata)


def test_synthetic_artifact_replays_each_window_and_keeps_real_gate_open(
    tmp_path: Path,
) -> None:
    """Synthetic replay produces tensors but cannot close the real-data gate."""
    root = _artifact(tmp_path)
    loaded = load_livox_custom_msg_replay_artifact(root)
    result = replay_livox_custom_msg_artifact(root, max_packet_age_s=0.5)

    assert len(loaded.windows) == 2
    assert loaded.metadata["data_status"] == {
        "dataset_class": "synthetic_contract_fixture",
        "real_data_present": False,
        "real_replay_verified": False,
        "training_ready": False,
    }
    assert [item.window_index for item in result.windows] == [0, 2]
    assert all(item.policy_tensor.shape == (1, 5, 2, 16, 96) for item in result.windows)
    assert all(item.policy_tensor.dtype == np.float16 for item in result.windows)
    assert all(len(item.tensor_sha256) == 64 for item in result.windows)
    assert result.windows[1].stats.implicit_missing_packets_inserted == 1
    assert result.receipt["dropped_window_accounting"] == {
        "accounting_scope": "strictly_between_first_and_last_recorded_window",
        "builder_explicit_dropped_total": 0,
        "builder_implicit_missing_total": 1,
        "inferred_missing_window_count": 1,
        "inferred_missing_window_indices": [1],
    }
    assert result.receipt["gate"] == {
        "REAL-LIVOX-REPLAY-001": "open_no_real_recording",
        "engineering_contract_verified": True,
        "real_data_present": False,
        "real_replay_verified": False,
        "training_ready": False,
    }


def test_source_recording_and_manifest_bytes_are_sha_bound(tmp_path: Path) -> None:
    """Any change to source or deployment-manifest bytes is rejected."""
    root = _artifact(tmp_path)
    source = root / "source_recording.bin"
    raw = bytearray(source.read_bytes())
    raw[0] ^= 1
    source.write_bytes(raw)
    with pytest.raises(LivoxCustomMsgReplayError, match="source recording SHA-256"):
        load_livox_custom_msg_replay_artifact(root)

    root = _artifact(tmp_path / "manifest")
    manifest = root / "deployment_manifest.json"
    raw = bytearray(manifest.read_bytes())
    raw[0] ^= 1
    manifest.write_bytes(raw)
    with pytest.raises(LivoxCustomMsgReplayError, match="deployment manifest SHA-256"):
        load_livox_custom_msg_replay_artifact(root)


def test_truncated_or_object_pickle_npz_fails_closed(tmp_path: Path) -> None:
    """Truncated archives and pickle-capable object arrays are forbidden."""
    root = _artifact(tmp_path / "truncated")
    metadata = _read_metadata(root)
    archive = root / metadata["windows"][0]["archive_file"]
    archive.write_bytes(archive.read_bytes()[:31])
    with pytest.raises(LivoxCustomMsgReplayError, match=r"byte length changed|truncated"):
        load_livox_custom_msg_replay_artifact(root)

    root = _artifact(tmp_path / "object")
    offsets = np.asarray((0, 1), dtype="<u4")
    objects = np.asarray([{"x": 1}, {"x": 2}], dtype=object)
    _replace_archive(root, 0, offset_time_ns=offsets, xyz_m=objects)
    with pytest.raises(LivoxCustomMsgReplayError, match="object/pickle"):
        load_livox_custom_msg_replay_artifact(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda data: data["windows"][0].update(xyz_unit="millimetre"), "coordinate/unit"),
        (lambda data: data["windows"][0].update(clock_domain="CLOCK_REALTIME"), "Cross-clock"),
        (lambda data: data["windows"].reverse(), "strictly increasing"),
        (lambda data: data["windows"][1].update(timebase_ns=1_000_000_000), "strictly increasing"),
        (lambda data: data["windows"][0].pop("timebase_ns"), "exactly"),
        (lambda data: data["windows"][0].update(source_point_count=1), "silent truncation"),
    ),
)
def test_unit_clock_order_missing_field_and_count_tampering_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    """Resealing metadata cannot legalize invalid semantic declarations."""
    root = _artifact(tmp_path)
    metadata = _read_metadata(root)
    mutation(metadata)
    _reseal(root, metadata)
    with pytest.raises(LivoxCustomMsgReplayError, match=message):
        load_livox_custom_msg_replay_artifact(root)


def test_duplicate_window_and_noncanonical_metadata_fail_closed(tmp_path: Path) -> None:
    """Duplicate indices and pretty/noncanonical JSON fail closed."""
    root = _artifact(tmp_path / "duplicate")
    metadata = _read_metadata(root)
    metadata["windows"][1]["window_index"] = 0
    _reseal(root, metadata)
    with pytest.raises(LivoxCustomMsgReplayError, match="strictly increasing"):
        load_livox_custom_msg_replay_artifact(root)

    root = _artifact(tmp_path / "pretty")
    metadata = _read_metadata(root)
    (root / LIVOX_REPLAY_METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(LivoxCustomMsgReplayError, match="canonical UTF-8 JSON"):
        load_livox_custom_msg_replay_artifact(root)


def test_undeclared_files_and_symlink_root_fail_closed(tmp_path: Path) -> None:
    """The sealed file set cannot be extended or reached through a symlink."""
    root = _artifact(tmp_path / "extra")
    (root / "undeclared.txt").write_text("not sealed")
    with pytest.raises(LivoxCustomMsgReplayError, match="file set differs"):
        load_livox_custom_msg_replay_artifact(root)

    root = _artifact(tmp_path / "symlink")
    alias = tmp_path / "replay-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(LivoxCustomMsgReplayError, match="root cannot be a symlink"):
        load_livox_custom_msg_replay_artifact(alias)


def test_nan_inf_and_array_length_truncation_fail_closed(tmp_path: Path) -> None:
    """Non-finite coordinates and shortened point arrays are rejected."""
    root = _artifact(tmp_path / "nan")
    metadata = _read_metadata(root)
    original = np.load(root / metadata["windows"][0]["archive_file"], allow_pickle=False)
    xyz = original["xyz_m"].copy()
    offsets = original["offset_time_ns"].copy()
    original.close()
    xyz[0, 0] = np.nan
    _replace_archive(root, 0, offset_time_ns=offsets, xyz_m=xyz)
    with pytest.raises(LivoxCustomMsgReplayError, match="NaN or Inf"):
        load_livox_custom_msg_replay_artifact(root)

    root = _artifact(tmp_path / "short")
    _replace_archive(
        root,
        0,
        offset_time_ns=np.asarray((0,), dtype="<u4"),
        xyz_m=np.asarray(((1.0, 0.0, 0.0),), dtype="<f4"),
    )
    with pytest.raises(LivoxCustomMsgReplayError, match="array lengths"):
        load_livox_custom_msg_replay_artifact(root)


def test_writer_rejects_unsorted_duplicate_windows_and_noncanonical_arrays(
    tmp_path: Path,
) -> None:
    """The writer validates ordering and canonical numeric dtypes up front."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        serialize_ray_time_deployment_manifest(_manifest(checkpoint, history_length=5, variant="Global"))
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic")
    common = dict(
        raw_recording_path=source,
        deployment_manifest_path=manifest_path,
        monotonic_clock_domain=_CLOCK,
        source_recording_format=LIVOX_REPLAY_SYNTHETIC_FORMAT,
        real_data_present=False,
    )
    with pytest.raises(LivoxCustomMsgReplayError, match="strictly increasing"):
        write_livox_custom_msg_replay_artifact(
            tmp_path / "order", windows=(_window(1, 1_100_000_000), _window(1, 1_100_000_000)), **common
        )
    bad = deepcopy(_window(0, 1_000_000_000))
    object.__setattr__(bad, "xyz_m", bad.xyz_m.astype(np.float64))
    with pytest.raises(LivoxCustomMsgReplayError, match="float32"):
        write_livox_custom_msg_replay_artifact(tmp_path / "dtype", windows=(bad,), **common)


def test_receipt_cannot_promote_synthetic_or_training_ready(tmp_path: Path) -> None:
    """Receipt validation preserves both evidence and training gates."""
    receipt = dict(replay_livox_custom_msg_artifact(_artifact(tmp_path), max_packet_age_s=0.5).receipt)
    tampered = deepcopy(receipt)
    tampered["gate"]["training_ready"] = True
    payload = {key: item for key, item in tampered.items() if key != "integrity"}
    tampered["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "RFC8259 JSON; UTF-8; sorted keys; no whitespace; no NaN",
        "payload_scope": "all top-level fields except integrity",
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    with pytest.raises(LivoxCustomMsgReplayError, match="cannot promote"):
        validate_livox_custom_msg_replay_receipt(tampered)
