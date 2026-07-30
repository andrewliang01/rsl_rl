# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import rsl_rl.utils.ray_time_export_attestation as attestation_module
from rsl_rl.utils.ray_time_export_attestation import (
    RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME,
    RayTimeExportAttestationError,
    build_ray_time_export_attestation,
    capture_ray_time_checkpoint_snapshot,
    validate_ray_time_export_attestation,
)
from tests.test_ray_time_attention_encoder import _make_ray_time_actor


@dataclass
class _ExportBundle:
    model: torch.nn.Module
    eager_wrapper: torch.nn.Module
    checkpoint: Path
    torchscript: Path
    onnx: Path
    split_onnx: Path | None


def _export_onnx(
    wrapper: torch.nn.Module,
    destination: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    torch.onnx.export(
        wrapper,
        wrapper.get_dummy_inputs(),  # type: ignore[attr-defined]
        destination,
        export_params=True,
        opset_version=18,
        external_data=False,
        verbose=False,
        input_names=wrapper.input_names,  # type: ignore[attr-defined]
        output_names=wrapper.output_names,  # type: ignore[attr-defined]
        dynamic_axes=wrapper.dynamic_axes,  # type: ignore[attr-defined]
    )
    graph = onnx.load(destination, load_external_data=False)
    graph.ir_version = 8
    onnx.save(graph, destination, save_as_external_data=False)


def _make_bundle(
    root: Path,
    *,
    seed: int,
    include_split_onnx: bool,
) -> _ExportBundle:
    root.mkdir(parents=True)
    torch.manual_seed(seed)
    model, _ = _make_ray_time_actor(
        history_length=1,
        use_query_attention=False,
        batch_size=1,
    )
    checkpoint = root / "model.pt"
    torch.save({"actor_state_dict": model.state_dict()}, checkpoint)

    scripted = torch.jit.script(model.as_jit().eval())
    torchscript = root / "policy.pt"
    torch.jit.save(scripted, torchscript)

    onnx_path = root / "policy.onnx"
    _export_onnx(
        model.as_onnx(verbose=False, input_mode="single").eval(),
        onnx_path,
    )
    split_path: Path | None = None
    if include_split_onnx:
        split_path = root / "policy_split.onnx"
        _export_onnx(
            model.as_onnx(verbose=False, input_mode="split").eval(),
            split_path,
        )
    return _ExportBundle(
        model=model,
        eager_wrapper=model.as_jit().eval(),
        checkpoint=checkpoint,
        torchscript=torchscript,
        onnx=onnx_path,
        split_onnx=split_path,
    )


@pytest.fixture(scope="module")
def export_bundles(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[_ExportBundle, _ExportBundle]:
    root = tmp_path_factory.mktemp("ray_time_attestation")
    return (
        _make_bundle(root / "actor_a", seed=901, include_split_onnx=True),
        _make_bundle(root / "actor_b", seed=902, include_split_onnx=False),
    )


def _build(bundle: _ExportBundle) -> dict[str, Any]:
    return build_ray_time_export_attestation(
        checkpoint_path=bundle.checkpoint,
        checkpoint_before=capture_ray_time_checkpoint_snapshot(
            bundle.checkpoint,
        ),
        torchscript_path=bundle.torchscript,
        onnx_path=bundle.onnx,
        eager_wrapper=bundle.eager_wrapper,
        history_length=1,
    )


def _copy_bundle(bundle: _ExportBundle, root: Path) -> _ExportBundle:
    root.mkdir()
    checkpoint = root / "model.pt"
    torchscript = root / "policy.pt"
    onnx = root / "policy.onnx"
    shutil.copyfile(bundle.checkpoint, checkpoint)
    shutil.copyfile(bundle.torchscript, torchscript)
    shutil.copyfile(bundle.onnx, onnx)
    return _ExportBundle(
        model=bundle.model,
        eager_wrapper=bundle.eager_wrapper,
        checkpoint=checkpoint,
        torchscript=torchscript,
        onnx=onnx,
        split_onnx=None,
    )


def test_build_and_validate_ray_time_export_attestation(
    export_bundles: tuple[_ExportBundle, _ExportBundle],
) -> None:
    actor_a, _ = export_bundles
    attestation = _build(actor_a)

    assert attestation["schema"]["name"] == RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME
    assert attestation["checkpoint"]["stable"] is True
    assert attestation["artifacts"]["torchscript"]["stable"] is True
    assert attestation["artifacts"]["onnx"]["stable"] is True
    assert attestation["artifacts"]["onnx"]["interface"]["checker"] == "passed"
    torchscript_inputs = attestation["artifacts"]["torchscript"]["interface"][
        "inputs"
    ]
    assert torchscript_inputs[1]["allowed_dtypes"] == ["float16", "float32"]
    assert attestation["state_binding"]["bit_exact"] is True
    assert attestation["state_binding"]["checkpoint_only_allowlist"] == [
        "distribution.std_param"
    ]
    assert set(attestation["state_binding"]["analytic_buffers"]) == {
        "elevation_encoder.spherical_position_encoding",
        "elevation_encoder.time_encoding",
    }
    assert [case["name"] for case in attestation["verification"]["cases"]] == [
        "unknown",
        "structured",
    ]
    assert all(
        comparison["allclose"]
        for case in attestation["verification"]["cases"]
        for comparison in case["comparisons"].values()
    )
    structured = attestation["verification"]["cases"][1]
    assert structured["inputs"]["ray_history_fp16"]["dtype"] == "float16"
    assert structured["outputs"]["torchscript_fp16_ray"]["shape"] == [3, 29]
    assert structured["comparisons"]["eager_vs_torchscript_fp16_ray"][
        "allclose"
    ]

    fresh = validate_ray_time_export_attestation(
        attestation,
        checkpoint_path=actor_a.checkpoint,
        torchscript_path=actor_a.torchscript,
        onnx_path=actor_a.onnx,
        eager_wrapper=actor_a.eager_wrapper,
        history_length=1,
    )
    assert fresh["artifacts"] == attestation["artifacts"]


def test_external_data_hidden_in_unused_onnx_function_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    obs = helper.make_tensor_value_info(
        "obs",
        TensorProto.FLOAT,
        ["batch_size", 2],
    )
    actions = helper.make_tensor_value_info(
        "actions",
        TensorProto.FLOAT,
        ["batch_size", 2],
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["obs"], ["actions"])],
        "main",
        [obs],
        [actions],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
    )
    hidden = TensorProto(
        name="hidden",
        data_type=TensorProto.FLOAT,
        dims=[1],
        data_location=TensorProto.EXTERNAL,
    )
    hidden.external_data.add(key="location", value="external.bin")
    hidden.external_data.add(key="offset", value="0")
    hidden.external_data.add(key="length", value="4")
    function = helper.make_function(
        "local",
        "Hidden",
        [],
        ["z"],
        [helper.make_node("Constant", [], ["z"], value=hidden)],
        [helper.make_opsetid("", 18)],
    )
    model.functions.append(function)
    (tmp_path / "external.bin").write_bytes(b"1234")
    monkeypatch.chdir(tmp_path)
    onnx.checker.check_model(model, full_check=True)

    with pytest.raises(
        RayTimeExportAttestationError,
        match="external tensor data is forbidden",
    ):
        attestation_module._load_onnx(
            model.SerializeToString(),
            flat_input_size=2,
            action_size=2,
            expected_opset_version=18,
            expected_ir_version=8,
        )


@pytest.mark.parametrize("artifact", ("torchscript", "onnx"))
def test_fake_artifact_is_rejected(
    tmp_path: Path,
    export_bundles: tuple[_ExportBundle, _ExportBundle],
    artifact: str,
) -> None:
    actor_a, _ = export_bundles
    copied = _copy_bundle(actor_a, tmp_path / artifact)
    getattr(copied, artifact).write_bytes(b"not a policy export")

    with pytest.raises(
        RayTimeExportAttestationError,
        match="Could not load TorchScript|Could not parse ONNX",
    ):
        _build(copied)


def test_wrong_onnx_interface_is_rejected(
    tmp_path: Path,
    export_bundles: tuple[_ExportBundle, _ExportBundle],
) -> None:
    actor_a, _ = export_bundles
    assert actor_a.split_onnx is not None
    copied = _copy_bundle(actor_a, tmp_path / "wrong_interface")
    shutil.copyfile(actor_a.split_onnx, copied.onnx)

    with pytest.raises(
        RayTimeExportAttestationError,
        match="exactly one graph input named 'obs'",
    ):
        _build(copied)


def test_checkpoint_changed_after_pre_export_snapshot_is_rejected(
    tmp_path: Path,
    export_bundles: tuple[_ExportBundle, _ExportBundle],
) -> None:
    actor_a, _ = export_bundles
    copied = _copy_bundle(actor_a, tmp_path / "changed_checkpoint")
    before = capture_ray_time_checkpoint_snapshot(copied.checkpoint)
    with copied.checkpoint.open("ab") as stream:
        stream.write(b"changed during export")

    with pytest.raises(
        RayTimeExportAttestationError,
        match="no longer matches its pre-export snapshot",
    ):
        build_ray_time_export_attestation(
            checkpoint_path=copied.checkpoint,
            checkpoint_before=before,
            torchscript_path=copied.torchscript,
            onnx_path=copied.onnx,
            eager_wrapper=copied.eager_wrapper,
            history_length=1,
        )


def test_exports_from_another_same_shape_model_are_rejected_by_state_binding(
    export_bundles: tuple[_ExportBundle, _ExportBundle],
) -> None:
    actor_a, actor_b = export_bundles
    with pytest.raises(
        RayTimeExportAttestationError,
        match="not bit-exact between checkpoint and TorchScript",
    ):
        build_ray_time_export_attestation(
            checkpoint_path=actor_a.checkpoint,
            checkpoint_before=capture_ray_time_checkpoint_snapshot(
                actor_a.checkpoint,
            ),
            torchscript_path=actor_b.torchscript,
            onnx_path=actor_b.onnx,
            eager_wrapper=actor_a.eager_wrapper,
            history_length=1,
        )


def test_onnx_from_another_same_shape_model_is_rejected_by_execution(
    export_bundles: tuple[_ExportBundle, _ExportBundle],
) -> None:
    actor_a, actor_b = export_bundles
    with pytest.raises(
        RayTimeExportAttestationError,
        match="eager and onnx outputs differ",
    ):
        build_ray_time_export_attestation(
            checkpoint_path=actor_a.checkpoint,
            checkpoint_before=capture_ray_time_checkpoint_snapshot(
                actor_a.checkpoint,
            ),
            torchscript_path=actor_a.torchscript,
            onnx_path=actor_b.onnx,
            eager_wrapper=actor_a.eager_wrapper,
            history_length=1,
        )
