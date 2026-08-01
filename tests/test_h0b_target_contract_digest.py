from __future__ import annotations

import hashlib
import json

import pytest

from rsl_rl.modules import (
    H0B_MANIFEST_TARGET_CONTRACT_ENCODING,
    audit_heightmap_target_contract_json_bytes,
    manifest_target_contract_json_bytes,
    manifest_target_contract_payload_sha256,
    validate_manifest_target_contract_binding,
)

_MANIFEST_DIGEST = "048e649f1c7866e7dcb0f75536f41a21f4c91ca9046e7dd1d48083fdad231b3e"
_LAB_FILE_DIGEST = "afc45ff19d6a611de220f45e4af65d7e7190eb2a609fe4a715c937b80a1025d5"


def _frozen_lab_contract() -> dict:
    """Return the 13-key Lab H0b contract frozen on 2026-08-01."""
    return {
        "contract_source_sha256": ("0fa3e0dae7d33dee536c342d1f6ac7450c842ec732450609ccc78ae78053d8f5"),
        "coordinate_frame": "world_z_difference_on_yaw_aligned_sensor_xy_grid",
        "flatten_order": "C_contiguous_row_major",
        "grid_axis_directions": [
            "negative_to_positive",
            "negative_to_positive",
        ],
        "grid_axis_order": ["x", "y"],
        "grid_shape": [28, 20],
        "height_sign": "positive_when_ray_hit_is_below_sensor_origin",
        "height_unit": "metre",
        "origin": ("grid_center_at_height_scanner_sensor_origin; first_cell_xy_m=(-0.675,-0.475)"),
        "resolution_m": 0.05,
        "schema_version": 1,
        "target_definition": ("sensor.data.pos_w[:,2]-sensor.data.ray_hits_w[...,2]"),
        "unknown_cell_policy": (
            "valid iff all ray_hits_w xyz are finite; RayCaster miss +inf is "
            "invalid; training adapter zero-fills invalid target only with "
            "separate target_valid mask and zero is not an observed-height claim"
        ),
    }


def test_real_manifest_and_lab_file_digests_differ_only_by_final_lf() -> None:
    """Fix both real hashes and prove their preimages differ by one LF."""
    contract = _frozen_lab_contract()
    no_lf = manifest_target_contract_json_bytes(contract)
    with_lf = no_lf + b"\n"

    assert H0B_MANIFEST_TARGET_CONTRACT_ENCODING.endswith("no_trailing_lf_v1")
    assert len(no_lf) == 827
    assert hashlib.sha256(no_lf).hexdigest() == _MANIFEST_DIGEST
    assert manifest_target_contract_payload_sha256(contract) == _MANIFEST_DIGEST
    assert len(with_lf) == 828
    assert hashlib.sha256(with_lf).hexdigest() == _LAB_FILE_DIGEST

    no_lf_audit = audit_heightmap_target_contract_json_bytes(
        no_lf,
        expected_manifest_payload_sha256=_MANIFEST_DIGEST,
    )
    with_lf_audit = audit_heightmap_target_contract_json_bytes(
        with_lf,
        expected_manifest_payload_sha256=_MANIFEST_DIGEST,
        expected_file_sha256=_LAB_FILE_DIGEST,
    )
    assert no_lf_audit["normalized_contract"] == with_lf_audit["normalized_contract"]
    assert no_lf_audit["encoding_relation"] == ("manifest_preimage_no_trailing_lf")
    assert with_lf_audit["encoding_relation"] == ("manifest_preimage_plus_one_trailing_lf")


def test_reordered_keys_keep_one_normalized_semantic_digest() -> None:
    """Prove JSON presentation cannot alter the normalized contract digest."""
    contract = _frozen_lab_contract()
    reordered = dict(reversed(tuple(contract.items())))
    reordered_bytes = json.dumps(
        reordered,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    audit = audit_heightmap_target_contract_json_bytes(
        reordered_bytes,
        expected_manifest_payload_sha256=_MANIFEST_DIGEST,
    )

    assert audit["normalized_contract"] == contract
    assert audit["manifest_target_contract_payload_sha256"] == _MANIFEST_DIGEST
    assert audit["encoding_relation"] == "semantic_equivalent_noncanonical_json"


def test_alternate_contract_and_raw_semantic_cross_splice_fail_closed() -> None:
    """Reject alternate meanings and mixed raw/semantic digest evidence."""
    canonical = manifest_target_contract_json_bytes(_frozen_lab_contract())
    alternate = _frozen_lab_contract()
    alternate["target_definition"] = "alternate target definition"
    alternate_bytes = manifest_target_contract_json_bytes(alternate) + b"\n"
    alternate_file_digest = hashlib.sha256(alternate_bytes).hexdigest()

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        validate_manifest_target_contract_binding(alternate, _MANIFEST_DIGEST)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        audit_heightmap_target_contract_json_bytes(
            alternate_bytes,
            expected_manifest_payload_sha256=_MANIFEST_DIGEST,
            expected_file_sha256=alternate_file_digest,
        )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        audit_heightmap_target_contract_json_bytes(
            canonical,
            expected_manifest_payload_sha256=_MANIFEST_DIGEST,
            expected_file_sha256=_LAB_FILE_DIGEST,
        )


def test_duplicate_key_is_not_an_equivalent_encoding() -> None:
    """Reject ambiguous JSON objects before semantic normalization."""
    duplicate = manifest_target_contract_json_bytes(_frozen_lab_contract())
    duplicate = duplicate[:-1] + b',"schema_version":1}'
    with pytest.raises(ValueError, match="repeats key"):
        audit_heightmap_target_contract_json_bytes(duplicate)
