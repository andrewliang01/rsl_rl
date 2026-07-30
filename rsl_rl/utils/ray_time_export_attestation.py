# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent, fail-closed attestation for Ray-Time policy exports.

The deployment manifest binds files by hash, but a hash alone does not prove
that a checkpoint, TorchScript export, and ONNX export implement the same
policy.  This module closes that gap without importing IsaacLab:

* the checkpoint actor tensors are bound bit-for-bit to the live eager export
  wrapper and the serialized TorchScript module;
* TorchScript and ONNX interfaces are checked and executed;
* deterministic known-answer cases compare eager, TorchScript, and ONNX
  outputs, including the rollout-time float16 Ray-Time input contract;
* file snapshots are rechecked after validation to detect path replacement or
  mutation while the attestation is being built.

These checks provide strong export-integrity evidence; like any finite
known-answer suite, they are not a formal proof that arbitrary programs are
equivalent on every possible input.  The returned dictionary is canonical-JSON
compatible and contains its own payload digest.  It is a separate sidecar and
intentionally does not change the deployment-manifest schema.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .ray_time_deployment_manifest import canonical_json_sha256


RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME = "ray-time-export-attestation"
RAY_TIME_EXPORT_ATTESTATION_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_ONLY_KEYS = ("distribution.std_param",)
_TORCHSCRIPT_ONLY_KEYS = (
    "elevation_encoder.spherical_position_encoding",
    "elevation_encoder.time_encoding",
)
_INTEGRITY_CANONICALIZATION = (
    "UTF-8 JSON; sorted keys; compact separators; no NaN"
)
_INTEGRITY_PAYLOAD_SCOPE = "all top-level fields except integrity"


class RayTimeExportAttestationError(ValueError):
    """Raised when policy exports fail the required integrity evidence."""


def _fail(message: str, exc: BaseException | None = None) -> None:
    error = RayTimeExportAttestationError(message)
    if exc is None:
        raise error
    raise error from exc


def _file_record(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    """Read one regular file through a stable descriptor and hash those bytes."""
    if path.is_symlink():
        _fail(f"{label} must not be a symbolic link: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(f"Could not open {label}: {path}: {exc}", exc)

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} must be a regular file: {path}")
        if before.st_size <= 0:
            _fail(f"{label} must not be empty: {path}")

        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
        _fail(f"{label} changed while it was being read: {path}")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        _fail(
            f"{label} size changed while it was being read: "
            f"read {len(data)} bytes, stat reports {after.st_size}: {path}"
        )
    return (
        {
            "file_name": path.name,
            "size_bytes": len(data),
            "sha256": digest.hexdigest(),
        },
        data,
    )


def _validate_file_record(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "file_name",
        "size_bytes",
        "sha256",
    }:
        _fail(f"{label} snapshot has an invalid structure.")
    file_name = value["file_name"]
    size_bytes = value["size_bytes"]
    digest = value["sha256"]
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
    ):
        _fail(f"{label}.file_name must be one non-empty base name.")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        _fail(f"{label}.size_bytes must be a positive integer.")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label}.sha256 must be a lowercase SHA-256 digest.")
    return {
        "file_name": file_name,
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def capture_ray_time_checkpoint_snapshot(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Capture the checkpoint before loading/export begins.

    Pass this exact record to :func:`build_ray_time_export_attestation`.  A
    checkpoint change between capture and attestation is rejected.
    """
    record, _ = _file_record(Path(checkpoint_path), label="checkpoint")
    return record


def _same_record(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    parsed_expected = _validate_file_record(expected, label=f"expected {label}")
    if dict(actual) != parsed_expected:
        _fail(
            f"{label} no longer matches its pre-export snapshot: "
            f"expected {parsed_expected}, got {dict(actual)}."
        )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.layout != torch.strided:
        _fail(f"Only strided tensors can be attested, got {tensor.layout}.")
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _tensor_digest(tensor: torch.Tensor) -> str:
    payload = {
        "dtype": _dtype_name(tensor),
        "shape": list(tensor.shape),
        "data_sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    }
    return canonical_json_sha256(payload)


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    finite = (
        bool(torch.isfinite(tensor).all())
        if tensor.is_floating_point() or tensor.is_complex()
        else True
    )
    return {
        "dtype": _dtype_name(tensor),
        "shape": list(tensor.shape),
        "finite": finite,
        "sha256": _tensor_digest(tensor),
    }


def _state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        metadata = (
            f"{name}\0{_dtype_name(tensor)}\0"
            f"{','.join(str(value) for value in tensor.shape)}\0"
        ).encode("utf-8")
        data = _tensor_bytes(tensor)
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _require_tensor_state(
    state: Any,
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        _fail(f"{label} must be a non-empty tensor mapping.")
    parsed: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            _fail(f"{label} contains an invalid tensor name.")
        if not isinstance(value, torch.Tensor):
            _fail(f"{label}.{name} is not a tensor.")
        tensor = value.detach().cpu()
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all()):
                _fail(f"{label}.{name} contains NaN or infinity.")
        parsed[name] = tensor
    return parsed


def _load_checkpoint_actor(checkpoint_bytes: bytes) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        _fail(f"Could not safely load the checkpoint: {exc}", exc)
    if not isinstance(checkpoint, Mapping) or "actor_state_dict" not in checkpoint:
        _fail("Checkpoint must contain actor_state_dict.")
    return _require_tensor_state(
        checkpoint["actor_state_dict"],
        label="checkpoint.actor_state_dict",
    )


def _build_spherical_encoding(
    spatial_size: tuple[int, int],
    feature_dim: int,
    vertical_fov_degrees: tuple[float, float],
) -> torch.Tensor:
    if feature_dim <= 0 or feature_dim % 4:
        _fail(
            "TorchScript spherical encoding feature dimension must be a "
            f"positive multiple of four, got {feature_dim}."
        )
    height, width = spatial_size
    elevation = torch.linspace(
        math.radians(vertical_fov_degrees[0]),
        math.radians(vertical_fov_degrees[1]),
        height,
    )
    azimuth = -math.pi + (torch.arange(width, dtype=torch.float32) + 0.5) * (
        2.0 * math.pi / width
    )
    elevation_grid, azimuth_grid = torch.meshgrid(
        elevation,
        azimuth,
        indexing="ij",
    )
    frequencies = torch.arange(1, feature_dim // 4 + 1, dtype=torch.float32)
    return torch.cat(
        (
            torch.sin(azimuth_grid[..., None] * frequencies),
            torch.cos(azimuth_grid[..., None] * frequencies),
            torch.sin(elevation_grid[..., None] * frequencies),
            torch.cos(elevation_grid[..., None] * frequencies),
        ),
        dim=-1,
    ).flatten(start_dim=0, end_dim=1)


def _build_time_encoding(history_length: int, feature_dim: int) -> torch.Tensor:
    if feature_dim <= 0 or feature_dim % 2:
        _fail(
            "TorchScript time encoding feature dimension must be a positive "
            f"even integer, got {feature_dim}."
        )
    positions = (
        torch.zeros(1)
        if history_length == 1
        else torch.linspace(-1.0, 0.0, history_length)
    )
    frequencies = torch.arange(1, feature_dim // 2 + 1, dtype=torch.float32)
    phase = positions[:, None] * math.pi * frequencies[None, :]
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)


def _compare_state_dicts(
    checkpoint_state: Mapping[str, torch.Tensor],
    eager_state: Mapping[str, torch.Tensor],
    jit_state: Mapping[str, torch.Tensor],
    *,
    action_size: int,
    history_length: int,
    image_height: int,
    image_width: int,
    vertical_fov_degrees: tuple[float, float],
    jit_module: torch.jit.ScriptModule,
) -> dict[str, Any]:
    checkpoint_names = set(checkpoint_state)
    eager_names = set(eager_state)
    jit_names = set(jit_state)
    checkpoint_only_eager = sorted(checkpoint_names - eager_names)
    eager_only = sorted(eager_names - checkpoint_names)
    checkpoint_only_jit = sorted(checkpoint_names - jit_names)
    jit_only = sorted(jit_names - checkpoint_names)
    if checkpoint_only_eager != list(_CHECKPOINT_ONLY_KEYS) or eager_only:
        _fail(
            "Checkpoint and eager export wrapper state keys do not match the "
            f"fixed allowlist: checkpoint-only={checkpoint_only_eager}, "
            f"eager-only={eager_only}."
        )
    if checkpoint_only_jit != list(_CHECKPOINT_ONLY_KEYS) or jit_only != list(
        _TORCHSCRIPT_ONLY_KEYS
    ):
        _fail(
            "Checkpoint and TorchScript state keys do not match the fixed "
            f"allowlist: checkpoint-only={checkpoint_only_jit}, "
            f"TorchScript-only={jit_only}."
        )

    std = checkpoint_state[_CHECKPOINT_ONLY_KEYS[0]]
    if tuple(std.shape) != (action_size,) or std.dtype != torch.float32:
        _fail(
            "checkpoint distribution.std_param must be float32 with shape "
            f"[{action_size}], got {std.dtype} {tuple(std.shape)}."
        )

    matched_names = sorted(checkpoint_names & jit_names)
    for name in matched_names:
        checkpoint_tensor = checkpoint_state[name]
        eager_tensor = eager_state[name]
        jit_tensor = jit_state[name]
        for other_label, other in (
            ("eager", eager_tensor),
            ("TorchScript", jit_tensor),
        ):
            if checkpoint_tensor.dtype != other.dtype:
                _fail(
                    f"{name} dtype differs between checkpoint and {other_label}: "
                    f"{checkpoint_tensor.dtype} != {other.dtype}."
                )
            if tuple(checkpoint_tensor.shape) != tuple(other.shape):
                _fail(
                    f"{name} shape differs between checkpoint and {other_label}: "
                    f"{tuple(checkpoint_tensor.shape)} != {tuple(other.shape)}."
                )
            if not torch.equal(checkpoint_tensor, other):
                _fail(
                    f"{name} is not bit-exact between checkpoint and "
                    f"{other_label} export state."
                )

    try:
        encoder = jit_module.elevation_encoder
        jit_history = int(encoder.history_length)
        jit_image_size = tuple(int(value) for value in encoder.vision_spatial_size)
        token_spatial_size = tuple(
            int(value) for value in encoder.token_spatial_size
        )
        token_dim = int(encoder.token_dim)
    except Exception as exc:
        _fail(f"TorchScript Ray-Time encoder metadata is unavailable: {exc}", exc)
    if jit_history != history_length or jit_image_size != (
        image_height,
        image_width,
    ):
        _fail(
            "TorchScript Ray-Time layout disagrees with attestation layout: "
            f"K={jit_history}, image={jit_image_size}."
        )

    spherical = jit_state[_TORCHSCRIPT_ONLY_KEYS[0]]
    time_encoding = jit_state[_TORCHSCRIPT_ONLY_KEYS[1]]
    expected_spherical = _build_spherical_encoding(
        token_spatial_size,
        token_dim,
        vertical_fov_degrees,
    )
    expected_time = _build_time_encoding(history_length, token_dim)
    for name, actual, expected in (
        (_TORCHSCRIPT_ONLY_KEYS[0], spherical, expected_spherical),
        (_TORCHSCRIPT_ONLY_KEYS[1], time_encoding, expected_time),
    ):
        if actual.dtype != torch.float32 or actual.shape != expected.shape:
            _fail(
                f"{name} has wrong dtype/shape: {actual.dtype} "
                f"{tuple(actual.shape)}, expected float32 {tuple(expected.shape)}."
            )
        if not bool(torch.isfinite(actual).all()):
            _fail(f"{name} contains NaN or infinity.")
        if not torch.equal(actual, expected):
            max_error = float(torch.max(torch.abs(actual - expected)))
            _fail(
                f"{name} does not match its analytic construction "
                f"(max_abs_error={max_error})."
            )

    return {
        "status": "passed",
        "bit_exact": True,
        "matched_tensor_count": len(matched_names),
        "checkpoint_only_allowlist": list(_CHECKPOINT_ONLY_KEYS),
        "torchscript_only_allowlist": list(_TORCHSCRIPT_ONLY_KEYS),
        "checkpoint_actor_state": {
            "tensor_count": len(checkpoint_state),
            "sha256": _state_dict_digest(checkpoint_state),
        },
        "eager_export_state": {
            "tensor_count": len(eager_state),
            "sha256": _state_dict_digest(eager_state),
        },
        "torchscript_state": {
            "tensor_count": len(jit_state),
            "sha256": _state_dict_digest(jit_state),
        },
        "analytic_buffers": {
            _TORCHSCRIPT_ONLY_KEYS[0]: {
                **_tensor_record(spherical),
                "analytic_match": True,
                "token_spatial_size": list(token_spatial_size),
                "vertical_fov_degrees": list(vertical_fov_degrees),
            },
            _TORCHSCRIPT_ONLY_KEYS[1]: {
                **_tensor_record(time_encoding),
                "analytic_match": True,
                "history_length": history_length,
            },
        },
    }


def _load_torchscript(
    serialized: bytes,
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    try:
        module = torch.jit.load(io.BytesIO(serialized), map_location="cpu").eval()
    except Exception as exc:
        _fail(f"Could not load TorchScript artifact: {exc}", exc)
    schema = module.forward.schema
    arguments = list(schema.arguments)
    returns = list(schema.returns)
    if len(arguments) != 3 or [arg.name for arg in arguments[1:]] != [
        "proprio_obs",
        "ray_history",
    ]:
        _fail(
            "TorchScript forward must accept exactly "
            "(proprio_obs: Tensor, ray_history: Tensor)."
        )
    if any(str(argument.type) != "Tensor" for argument in arguments[1:]):
        _fail("TorchScript forward inputs must both be Tensor.")
    if len(returns) != 1 or str(returns[0].type) != "Tensor":
        _fail("TorchScript forward must return exactly one Tensor.")
    return module, {
        "schema_argument_names": ["proprio_obs", "ray_history"],
        "inputs": [
            {"name": "proprio_obs", "dtype": "float32", "shape": ["B", 96]},
            {
                "name": "ray_history",
                "allowed_dtypes": ["float16", "float32"],
                "shape": ["B", "K", 2, "H", "W"],
            },
        ],
        "outputs": [
            {"name": "actions", "dtype": "float32", "shape": ["B", 29]}
        ],
    }


def _onnx_model_tensors(model: Any) -> list[Any]:
    """Find every TensorProto anywhere in a ModelProto.

    Walking the protobuf rather than only ``model.graph`` also covers local
    functions and training-info graphs, even when they are not called by the
    inference graph.
    """
    tensors: list[Any] = []

    def visit(message: Any) -> None:
        descriptor = getattr(message, "DESCRIPTOR", None)
        if descriptor is None:
            return
        if descriptor.full_name == "onnx.TensorProto":
            tensors.append(message)
            return
        for field, value in message.ListFields():
            if field.type != field.TYPE_MESSAGE:
                continue
            if field.is_repeated:
                for item in value:
                    visit(item)
            else:
                visit(value)

    visit(model)
    return tensors


def _onnx_shape(value_info: Any, *, label: str) -> tuple[str, int]:
    tensor_type = value_info.type.tensor_type
    dimensions = tensor_type.shape.dim
    if len(dimensions) != 2:
        _fail(f"ONNX {label} must be rank two.")
    if dimensions[0].WhichOneof("value") != "dim_param":
        _fail(f"ONNX {label} batch dimension must be dynamic.")
    batch_symbol = dimensions[0].dim_param
    if not batch_symbol:
        _fail(f"ONNX {label} dynamic batch symbol must not be empty.")
    if dimensions[1].WhichOneof("value") != "dim_value":
        _fail(f"ONNX {label} feature dimension must be fixed.")
    return batch_symbol, int(dimensions[1].dim_value)


def _load_onnx(
    serialized: bytes,
    *,
    flat_input_size: int,
    action_size: int,
    expected_opset_version: int,
    expected_ir_version: int,
) -> tuple[Any, dict[str, Any]]:
    try:
        import onnx
    except ImportError as exc:
        _fail("ONNX is required for Ray-Time export attestation.", exc)
    try:
        model = onnx.load_model_from_string(serialized)
    except Exception as exc:
        _fail(f"Could not parse ONNX artifact: {exc}", exc)

    external_tensors = [
        tensor.name or "<unnamed>"
        for tensor in _onnx_model_tensors(model)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        or len(tensor.external_data) > 0
    ]
    if external_tensors:
        _fail(
            "ONNX external tensor data is forbidden: "
            f"{sorted(external_tensors)}."
        )
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        _fail(f"ONNX checker rejected the artifact: {exc}", exc)

    opsets = sorted((item.domain, int(item.version)) for item in model.opset_import)
    if opsets != [("", expected_opset_version)]:
        _fail(
            "ONNX must import exactly the default-domain opset "
            f"{expected_opset_version}, got {opsets}."
        )
    if int(model.ir_version) != expected_ir_version:
        _fail(
            f"ONNX IR version must be {expected_ir_version}, "
            f"got {model.ir_version}."
        )
    if len(model.graph.input) != 1 or model.graph.input[0].name != "obs":
        _fail("ONNX must have exactly one graph input named 'obs'.")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "actions":
        _fail("ONNX must have exactly one graph output named 'actions'.")
    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    if graph_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        _fail("ONNX obs input must be float32.")
    if graph_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        _fail("ONNX actions output must be float32.")
    input_batch, input_features = _onnx_shape(graph_input, label="obs input")
    output_batch, output_features = _onnx_shape(
        graph_output,
        label="actions output",
    )
    if input_batch != output_batch:
        _fail(
            "ONNX obs and actions must use the same dynamic batch symbol, got "
            f"{input_batch!r} and {output_batch!r}."
        )
    if input_features != flat_input_size:
        _fail(
            f"ONNX obs feature size must be {flat_input_size}, "
            f"got {input_features}."
        )
    if output_features != action_size:
        _fail(
            f"ONNX actions feature size must be {action_size}, "
            f"got {output_features}."
        )
    return model, {
        "checker": "passed",
        "external_data": False,
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {"domain": domain, "version": version} for domain, version in opsets
        ],
        "inputs": [
            {
                "name": "obs",
                "dtype": "float32",
                "shape": [input_batch, input_features],
            }
        ],
        "outputs": [
            {
                "name": "actions",
                "dtype": "float32",
                "shape": [output_batch, output_features],
            }
        ],
    }


def _deterministic_cases(
    *,
    history_length: int,
    image_height: int,
    image_width: int,
    proprio_size: int,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    unknown = (
        "unknown",
        torch.zeros(1, proprio_size, dtype=torch.float32),
        torch.zeros(
            1,
            history_length,
            2,
            image_height,
            image_width,
            dtype=torch.float32,
        ),
    )
    batch_size = 3
    proprio = torch.linspace(
        -1.25,
        1.75,
        batch_size * proprio_size,
        dtype=torch.float32,
    ).reshape(batch_size, proprio_size)
    ray_count = batch_size * history_length * image_height * image_width
    metric_range = torch.linspace(
        0.1,
        6.0,
        ray_count,
        dtype=torch.float32,
    ).reshape(batch_size, history_length, image_height, image_width)
    indices = torch.arange(ray_count).reshape_as(metric_range)
    hit_mask = ((indices % 5 != 0) & (indices % 11 != 0)).to(torch.float32)
    metric_range = metric_range * hit_mask
    ray_history = torch.stack((metric_range, hit_mask), dim=2)
    return [unknown, ("structured", proprio, ray_history)]


def _numpy_record(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "finite": bool(np.isfinite(contiguous).all()),
        "sha256": digest.hexdigest(),
    }


def _max_errors(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    max_absolute = float(difference.max(initial=0.0))
    denominator = np.maximum(
        np.maximum(np.abs(reference.astype(np.float64)), np.abs(actual.astype(np.float64))),
        np.finfo(np.float32).tiny,
    )
    max_relative = float((difference / denominator).max(initial=0.0))
    return max_absolute, max_relative


def _onnx_runner(model: Any, serialized: bytes) -> tuple[str, Any]:
    try:
        import onnxruntime
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        evaluator = ReferenceEvaluator(model)
        return "onnx.reference", lambda inputs: evaluator.run(None, inputs)[0]
    try:
        session = onnxruntime.InferenceSession(
            serialized,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        _fail(f"ONNX Runtime could not load the artifact: {exc}", exc)
    return "onnxruntime-cpu", lambda inputs: session.run(None, inputs)[0]


def _run_cases(
    *,
    eager_wrapper: torch.nn.Module,
    jit_module: torch.jit.ScriptModule,
    onnx_model: Any,
    onnx_bytes: bytes,
    history_length: int,
    image_height: int,
    image_width: int,
    proprio_size: int,
    action_size: int,
    rtol: float,
    atol: float,
) -> tuple[str, list[dict[str, Any]]]:
    for name, tensor in eager_wrapper.state_dict().items():
        if tensor.device.type != "cpu":
            _fail(f"eager export wrapper state {name} must be on CPU.")
    eager_wrapper.eval()
    backend, run_onnx = _onnx_runner(onnx_model, onnx_bytes)
    results: list[dict[str, Any]] = []

    for case_name, proprio, ray_history in _deterministic_cases(
        history_length=history_length,
        image_height=image_height,
        image_width=image_width,
        proprio_size=proprio_size,
    ):
        flat_obs = torch.cat((proprio, ray_history.flatten(start_dim=1)), dim=1)
        with torch.inference_mode():
            try:
                eager_output = eager_wrapper(proprio, ray_history)
                jit_output = jit_module(proprio, ray_history)
            except Exception as exc:
                _fail(
                    f"Eager/TorchScript execution failed for {case_name}: {exc}",
                    exc,
                )
        try:
            onnx_output = np.asarray(
                run_onnx({"obs": flat_obs.numpy()}),
            )
        except Exception as exc:
            _fail(f"ONNX execution failed for {case_name}: {exc}", exc)

        eager_array = eager_output.detach().cpu().numpy()
        jit_array = jit_output.detach().cpu().numpy()
        outputs = {
            "eager": eager_array,
            "torchscript": jit_array,
            "onnx": onnx_output,
        }
        if case_name == "structured":
            half_ray_history = ray_history.to(torch.float16)
            with torch.inference_mode():
                try:
                    eager_half_output = eager_wrapper(
                        proprio,
                        half_ray_history,
                    )
                    jit_half_output = jit_module(
                        proprio,
                        half_ray_history,
                    )
                except Exception as exc:
                    _fail(
                        "Eager/TorchScript float16 Ray-Time execution failed "
                        f"for {case_name}: {exc}",
                        exc,
                    )
            outputs["eager_fp16_ray"] = (
                eager_half_output.detach().cpu().numpy()
            )
            outputs["torchscript_fp16_ray"] = (
                jit_half_output.detach().cpu().numpy()
            )
        expected_shape = (proprio.shape[0], action_size)
        for output_name, output in outputs.items():
            if output.dtype != np.float32:
                _fail(
                    f"{case_name} {output_name} output must be float32, "
                    f"got {output.dtype}."
                )
            if output.shape != expected_shape:
                _fail(
                    f"{case_name} {output_name} output shape must be "
                    f"{expected_shape}, got {output.shape}."
                )
            if not np.isfinite(output).all():
                _fail(f"{case_name} {output_name} output is non-finite.")

        comparisons: dict[str, Any] = {}
        for output_name, output in (
            ("torchscript", jit_array),
            ("onnx", onnx_output),
        ):
            max_absolute, max_relative = _max_errors(eager_array, output)
            allclose = bool(
                np.allclose(eager_array, output, rtol=rtol, atol=atol)
            )
            if not allclose:
                _fail(
                    f"{case_name} eager and {output_name} outputs differ: "
                    f"max_abs_error={max_absolute}, "
                    f"max_rel_error={max_relative}, rtol={rtol}, atol={atol}."
                )
            comparisons[f"eager_vs_{output_name}"] = {
                "allclose": True,
                "max_abs_error": max_absolute,
                "max_rel_error": max_relative,
            }
        if case_name == "structured":
            eager_half_array = outputs["eager_fp16_ray"]
            jit_half_array = outputs["torchscript_fp16_ray"]
            max_absolute, max_relative = _max_errors(
                eager_half_array,
                jit_half_array,
            )
            allclose = bool(
                np.allclose(
                    eager_half_array,
                    jit_half_array,
                    rtol=rtol,
                    atol=atol,
                )
            )
            if not allclose:
                _fail(
                    "structured eager and TorchScript outputs differ for "
                    "float16 Ray-Time input: "
                    f"max_abs_error={max_absolute}, "
                    f"max_rel_error={max_relative}, rtol={rtol}, atol={atol}."
                )
            comparisons["eager_vs_torchscript_fp16_ray"] = {
                "allclose": True,
                "max_abs_error": max_absolute,
                "max_rel_error": max_relative,
            }

        input_records = {
            "proprio_obs": _numpy_record(proprio.numpy()),
            "ray_history": _numpy_record(ray_history.numpy()),
            "flat_obs": _numpy_record(flat_obs.numpy()),
        }
        if case_name == "structured":
            input_records["ray_history_fp16"] = _numpy_record(
                ray_history.to(torch.float16).numpy()
            )
        results.append(
            {
                "name": case_name,
                "batch_size": int(proprio.shape[0]),
                "inputs": input_records,
                "outputs": {
                    name: _numpy_record(output) for name, output in outputs.items()
                },
                "comparisons": comparisons,
            }
        )
    return backend, results


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite non-negative number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        _fail(f"{label} must be a finite non-negative number.")
    return resolved


def build_ray_time_export_attestation(
    *,
    checkpoint_path: str | os.PathLike[str],
    checkpoint_before: Mapping[str, Any],
    torchscript_path: str | os.PathLike[str],
    onnx_path: str | os.PathLike[str],
    eager_wrapper: torch.nn.Module,
    history_length: int,
    image_height: int = 16,
    image_width: int = 96,
    proprio_size: int = 96,
    action_size: int = 29,
    vertical_fov_degrees: Sequence[float] = (-52.0, 7.0),
    expected_opset_version: int = 18,
    expected_ir_version: int = 8,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    """Build machine-auditable evidence that all three policy forms agree."""
    history_length = _positive_int(history_length, label="history_length")
    image_height = _positive_int(image_height, label="image_height")
    image_width = _positive_int(image_width, label="image_width")
    proprio_size = _positive_int(proprio_size, label="proprio_size")
    action_size = _positive_int(action_size, label="action_size")
    expected_opset_version = _positive_int(
        expected_opset_version,
        label="expected_opset_version",
    )
    expected_ir_version = _positive_int(
        expected_ir_version,
        label="expected_ir_version",
    )
    rtol = _finite_nonnegative(rtol, label="rtol")
    atol = _finite_nonnegative(atol, label="atol")
    if (
        not isinstance(vertical_fov_degrees, Sequence)
        or isinstance(vertical_fov_degrees, (str, bytes))
        or len(vertical_fov_degrees) != 2
    ):
        _fail("vertical_fov_degrees must contain exactly two numbers.")
    vertical_fov = (
        float(vertical_fov_degrees[0]),
        float(vertical_fov_degrees[1]),
    )
    if (
        not all(math.isfinite(value) for value in vertical_fov)
        or vertical_fov[1] <= vertical_fov[0]
    ):
        _fail("vertical_fov_degrees must be finite and strictly increasing.")
    if not isinstance(eager_wrapper, torch.nn.Module):
        _fail("eager_wrapper must be a torch.nn.Module.")

    checkpoint_path = Path(checkpoint_path)
    torchscript_path = Path(torchscript_path)
    onnx_path = Path(onnx_path)
    resolved_paths = {
        checkpoint_path.resolve(),
        torchscript_path.resolve(),
        onnx_path.resolve(),
    }
    if len(resolved_paths) != 3:
        _fail("Checkpoint, TorchScript, and ONNX paths must be distinct.")

    checkpoint_record, checkpoint_bytes = _file_record(
        checkpoint_path,
        label="checkpoint",
    )
    _same_record(
        checkpoint_record,
        checkpoint_before,
        label="checkpoint",
    )
    torchscript_record, torchscript_bytes = _file_record(
        torchscript_path,
        label="TorchScript artifact",
    )
    onnx_record, onnx_bytes = _file_record(onnx_path, label="ONNX artifact")

    checkpoint_state = _load_checkpoint_actor(checkpoint_bytes)
    jit_module, jit_interface = _load_torchscript(torchscript_bytes)
    eager_state = _require_tensor_state(
        eager_wrapper.state_dict(),
        label="eager_export_wrapper.state_dict",
    )
    jit_state = _require_tensor_state(
        jit_module.state_dict(),
        label="torchscript.state_dict",
    )
    state_binding = _compare_state_dicts(
        checkpoint_state,
        eager_state,
        jit_state,
        action_size=action_size,
        history_length=history_length,
        image_height=image_height,
        image_width=image_width,
        vertical_fov_degrees=vertical_fov,
        jit_module=jit_module,
    )

    flat_input_size = (
        proprio_size
        + history_length * 2 * image_height * image_width
    )
    onnx_model, onnx_interface = _load_onnx(
        onnx_bytes,
        flat_input_size=flat_input_size,
        action_size=action_size,
        expected_opset_version=expected_opset_version,
        expected_ir_version=expected_ir_version,
    )
    runtime_backend, cases = _run_cases(
        eager_wrapper=eager_wrapper,
        jit_module=jit_module,
        onnx_model=onnx_model,
        onnx_bytes=onnx_bytes,
        history_length=history_length,
        image_height=image_height,
        image_width=image_width,
        proprio_size=proprio_size,
        action_size=action_size,
        rtol=rtol,
        atol=atol,
    )

    checkpoint_after, _ = _file_record(checkpoint_path, label="checkpoint")
    torchscript_after, _ = _file_record(
        torchscript_path,
        label="TorchScript artifact",
    )
    onnx_after, _ = _file_record(onnx_path, label="ONNX artifact")
    _same_record(checkpoint_after, checkpoint_record, label="checkpoint")
    _same_record(
        torchscript_after,
        torchscript_record,
        label="TorchScript artifact",
    )
    _same_record(onnx_after, onnx_record, label="ONNX artifact")

    payload: dict[str, Any] = {
        "schema": {
            "name": RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME,
            "version": RAY_TIME_EXPORT_ATTESTATION_SCHEMA_VERSION,
        },
        "checkpoint": {
            "before_export": dict(checkpoint_record),
            "after_validation": dict(checkpoint_after),
            "stable": True,
        },
        "layout": {
            "history_length": history_length,
            "ray_channels": 2,
            "image_height": image_height,
            "image_width": image_width,
            "proprio_size": proprio_size,
            "action_size": action_size,
            "onnx_flat_input_size": flat_input_size,
            "vertical_fov_degrees": list(vertical_fov),
        },
        "state_binding": state_binding,
        "artifacts": {
            "torchscript": {
                "file": dict(torchscript_record),
                "after_validation": dict(torchscript_after),
                "stable": True,
                "interface": {
                    **jit_interface,
                    "inputs": [
                        {
                            "name": "proprio_obs",
                            "dtype": "float32",
                            "shape": ["B", proprio_size],
                        },
                        {
                            "name": "ray_history",
                            "allowed_dtypes": ["float16", "float32"],
                            "shape": [
                                "B",
                                history_length,
                                2,
                                image_height,
                                image_width,
                            ],
                        },
                    ],
                    "outputs": [
                        {
                            "name": "actions",
                            "dtype": "float32",
                            "shape": ["B", action_size],
                        }
                    ],
                },
            },
            "onnx": {
                "file": dict(onnx_record),
                "after_validation": dict(onnx_after),
                "stable": True,
                "interface": onnx_interface,
            },
        },
        "verification": {
            "test_suite": "deterministic-unknown-and-structured-v1",
            "onnx_runtime_backend": runtime_backend,
            "rtol": rtol,
            "atol": atol,
            "cases": cases,
        },
    }
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": _INTEGRITY_CANONICALIZATION,
        "payload_scope": _INTEGRITY_PAYLOAD_SCOPE,
        "payload_sha256": canonical_json_sha256(payload),
    }
    return payload


def _validate_expected_document(attestation: Mapping[str, Any]) -> None:
    if not isinstance(attestation, Mapping):
        _fail("Expected attestation must be a mapping.")
    required = {
        "schema",
        "checkpoint",
        "layout",
        "state_binding",
        "artifacts",
        "verification",
        "integrity",
    }
    if set(attestation) != required:
        _fail(
            "Expected attestation has wrong top-level fields: "
            f"{sorted(attestation)}."
        )
    if attestation["schema"] != {
        "name": RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME,
        "version": RAY_TIME_EXPORT_ATTESTATION_SCHEMA_VERSION,
    }:
        _fail("Expected attestation schema is unsupported.")
    integrity = attestation["integrity"]
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "algorithm",
        "canonicalization",
        "payload_scope",
        "payload_sha256",
    }:
        _fail("Expected attestation integrity record is invalid.")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != _INTEGRITY_CANONICALIZATION
        or integrity["payload_scope"] != _INTEGRITY_PAYLOAD_SCOPE
    ):
        _fail("Expected attestation integrity contract is unsupported.")
    digest = integrity["payload_sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        _fail("Expected attestation payload digest is invalid.")
    payload = {key: attestation[key] for key in attestation if key != "integrity"}
    if canonical_json_sha256(payload) != digest:
        _fail("Expected attestation integrity digest does not match its payload.")


def _validate_attested_file_pair(
    value: Any,
    *,
    label: str,
    path: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "after_validation",
        "stable",
        "interface",
    }:
        _fail(f"{label} attestation record has an invalid structure.")
    before = _validate_file_record(value["file"], label=f"{label}.file")
    after = _validate_file_record(
        value["after_validation"],
        label=f"{label}.after_validation",
    )
    if value["stable"] is not True or before != after:
        _fail(f"{label} attestation does not record one stable artifact.")
    if not isinstance(value["interface"], Mapping):
        _fail(f"{label}.interface must be a mapping.")
    if path is not None:
        actual, _ = _file_record(Path(path), label=label)
        if actual != before:
            _fail(
                f"{label} does not match its attested file record: "
                f"expected {before}, got {actual}."
            )
    return before


def _validate_passed_verification(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "test_suite",
        "onnx_runtime_backend",
        "rtol",
        "atol",
        "cases",
    }:
        _fail("Expected verification record has an invalid structure.")
    if value["test_suite"] != "deterministic-unknown-and-structured-v1":
        _fail("Expected attestation deterministic test suite is unsupported.")
    if value["onnx_runtime_backend"] not in (
        "onnx.reference",
        "onnxruntime-cpu",
    ):
        _fail("Expected attestation ONNX runtime backend is unsupported.")
    _finite_nonnegative(value["rtol"], label="expected rtol")
    _finite_nonnegative(value["atol"], label="expected atol")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) != 2:
        _fail("Expected attestation must contain exactly two verification cases.")
    expected_cases = (
        (
            "unknown",
            1,
            ("eager_vs_torchscript", "eager_vs_onnx"),
        ),
        (
            "structured",
            3,
            (
                "eager_vs_torchscript",
                "eager_vs_onnx",
                "eager_vs_torchscript_fp16_ray",
            ),
        ),
    )
    for case, (name, batch_size, comparison_names) in zip(
        cases,
        expected_cases,
    ):
        if not isinstance(case, Mapping) or set(case) != {
            "name",
            "batch_size",
            "inputs",
            "outputs",
            "comparisons",
        }:
            _fail(f"Expected {name} verification case is malformed.")
        if case["name"] != name or case["batch_size"] != batch_size:
            _fail(f"Expected {name} verification case identity is invalid.")
        if not isinstance(case["inputs"], Mapping) or not isinstance(
            case["outputs"],
            Mapping,
        ):
            _fail(f"Expected {name} verification tensors are malformed.")
        comparisons = case["comparisons"]
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(
            comparison_names
        ):
            _fail(f"Expected {name} verification comparisons are malformed.")
        for comparison_name in comparison_names:
            comparison = comparisons[comparison_name]
            if not isinstance(comparison, Mapping) or set(comparison) != {
                "allclose",
                "max_abs_error",
                "max_rel_error",
            }:
                _fail(
                    f"Expected {name}.{comparison_name} comparison is malformed."
                )
            if comparison["allclose"] is not True:
                _fail(
                    f"Expected {name}.{comparison_name} did not pass allclose."
                )
            _finite_nonnegative(
                comparison["max_abs_error"],
                label=f"{name}.{comparison_name}.max_abs_error",
            )
            _finite_nonnegative(
                comparison["max_rel_error"],
                label=f"{name}.{comparison_name}.max_rel_error",
            )


def validate_ray_time_export_attestation_document(
    attestation: Mapping[str, Any],
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,
    torchscript_path: str | os.PathLike[str] | None = None,
    onnx_path: str | os.PathLike[str] | None = None,
) -> None:
    """Validate a sealed attestation document and its optional file bindings.

    This is a document-integrity and byte-binding preflight.  It deliberately
    does not claim graph equivalence: consumers that have the loaded eager
    policy must additionally call :func:`validate_ray_time_export_attestation`
    to re-run state binding and deterministic cross-backend execution.
    """
    _validate_expected_document(attestation)

    checkpoint = attestation["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "before_export",
        "after_validation",
        "stable",
    }:
        _fail("Expected checkpoint attestation record is malformed.")
    checkpoint_before = _validate_file_record(
        checkpoint["before_export"],
        label="checkpoint.before_export",
    )
    checkpoint_after = _validate_file_record(
        checkpoint["after_validation"],
        label="checkpoint.after_validation",
    )
    if checkpoint["stable"] is not True or checkpoint_before != checkpoint_after:
        _fail("Expected attestation does not record one stable checkpoint.")
    if checkpoint_path is not None:
        actual_checkpoint, _ = _file_record(
            Path(checkpoint_path),
            label="checkpoint",
        )
        if actual_checkpoint != checkpoint_before:
            _fail(
                "Checkpoint does not match its attested file record: "
                f"expected {checkpoint_before}, got {actual_checkpoint}."
            )

    layout = attestation["layout"]
    if not isinstance(layout, Mapping) or set(layout) != {
        "history_length",
        "ray_channels",
        "image_height",
        "image_width",
        "proprio_size",
        "action_size",
        "onnx_flat_input_size",
        "vertical_fov_degrees",
    }:
        _fail("Expected attestation layout is malformed.")
    history_length = _positive_int(
        layout["history_length"],
        label="layout.history_length",
    )
    if layout["ray_channels"] != 2:
        _fail("layout.ray_channels must be exactly two.")
    image_height = _positive_int(
        layout["image_height"],
        label="layout.image_height",
    )
    image_width = _positive_int(
        layout["image_width"],
        label="layout.image_width",
    )
    proprio_size = _positive_int(
        layout["proprio_size"],
        label="layout.proprio_size",
    )
    action_size = _positive_int(
        layout["action_size"],
        label="layout.action_size",
    )
    expected_flat_size = (
        proprio_size + history_length * 2 * image_height * image_width
    )
    if layout["onnx_flat_input_size"] != expected_flat_size:
        _fail(
            "layout.onnx_flat_input_size does not match the attested tensor "
            "layout."
        )
    vertical_fov = layout["vertical_fov_degrees"]
    if (
        not isinstance(vertical_fov, list)
        or len(vertical_fov) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in vertical_fov
        )
        or float(vertical_fov[1]) <= float(vertical_fov[0])
    ):
        _fail("layout.vertical_fov_degrees must be finite and increasing.")

    state_binding = attestation["state_binding"]
    if not isinstance(state_binding, Mapping):
        _fail("Expected state_binding must be a mapping.")
    if state_binding.get("status") != "passed" or state_binding.get(
        "bit_exact"
    ) is not True:
        _fail("Expected state binding did not pass bit-exact validation.")
    if state_binding.get("checkpoint_only_allowlist") != list(
        _CHECKPOINT_ONLY_KEYS
    ):
        _fail("Expected checkpoint-only state allowlist is unsupported.")
    if state_binding.get("torchscript_only_allowlist") != list(
        _TORCHSCRIPT_ONLY_KEYS
    ):
        _fail("Expected TorchScript-only state allowlist is unsupported.")
    analytic_buffers = state_binding.get("analytic_buffers")
    if not isinstance(analytic_buffers, Mapping) or set(
        analytic_buffers
    ) != set(_TORCHSCRIPT_ONLY_KEYS):
        _fail("Expected analytic buffer evidence is incomplete.")
    for name in _TORCHSCRIPT_ONLY_KEYS:
        record = analytic_buffers[name]
        if not isinstance(record, Mapping) or record.get(
            "analytic_match"
        ) is not True:
            _fail(f"Expected analytic buffer {name} did not pass validation.")

    artifacts = attestation["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "torchscript",
        "onnx",
    }:
        _fail("Expected artifact attestation record is malformed.")
    _validate_attested_file_pair(
        artifacts["torchscript"],
        label="TorchScript artifact",
        path=torchscript_path,
    )
    _validate_attested_file_pair(
        artifacts["onnx"],
        label="ONNX artifact",
        path=onnx_path,
    )
    _validate_passed_verification(attestation["verification"])


def validate_ray_time_export_attestation(
    attestation: Mapping[str, Any],
    *,
    checkpoint_path: str | os.PathLike[str],
    torchscript_path: str | os.PathLike[str],
    onnx_path: str | os.PathLike[str],
    eager_wrapper: torch.nn.Module,
    history_length: int,
    image_height: int = 16,
    image_width: int = 96,
    proprio_size: int = 96,
    action_size: int = 29,
    vertical_fov_degrees: Sequence[float] = (-52.0, 7.0),
    expected_opset_version: int = 18,
    expected_ir_version: int = 8,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    """Re-run live validation and verify it against an expected attestation.

    The returned value is the fresh live attestation.  ONNX numerical metrics
    are deliberately not required to be byte-identical across ONNX Runtime and
    the reference evaluator; artifact identity, interfaces, eager/JIT output
    identities, deterministic inputs, and all tolerance checks are exact.
    """
    validate_ray_time_export_attestation_document(
        attestation,
        checkpoint_path=checkpoint_path,
        torchscript_path=torchscript_path,
        onnx_path=onnx_path,
    )
    checkpoint = attestation["checkpoint"]
    verification = attestation["verification"]
    if not isinstance(checkpoint, Mapping) or not isinstance(
        verification,
        Mapping,
    ):
        _fail("Expected checkpoint/verification records are invalid.")
    expected_before = checkpoint.get("before_export")
    expected_rtol = verification.get("rtol")
    expected_atol = verification.get("atol")
    if not isinstance(expected_before, Mapping):
        _fail("Expected checkpoint.before_export record is invalid.")
    rtol = _finite_nonnegative(rtol, label="rtol")
    atol = _finite_nonnegative(atol, label="atol")
    if (
        _finite_nonnegative(expected_rtol, label="expected rtol") != rtol
        or _finite_nonnegative(expected_atol, label="expected atol") != atol
    ):
        _fail("Expected attestation tolerances do not match live requirements.")
    if verification.get("test_suite") != (
        "deterministic-unknown-and-structured-v1"
    ):
        _fail("Expected attestation deterministic test suite is unsupported.")

    actual = build_ray_time_export_attestation(
        checkpoint_path=checkpoint_path,
        checkpoint_before=expected_before,
        torchscript_path=torchscript_path,
        onnx_path=onnx_path,
        eager_wrapper=eager_wrapper,
        history_length=history_length,
        image_height=image_height,
        image_width=image_width,
        proprio_size=proprio_size,
        action_size=action_size,
        vertical_fov_degrees=vertical_fov_degrees,
        expected_opset_version=expected_opset_version,
        expected_ir_version=expected_ir_version,
        rtol=rtol,
        atol=atol,
    )

    for section in ("schema", "checkpoint", "layout", "state_binding", "artifacts"):
        if actual[section] != attestation[section]:
            _fail(f"Live {section} does not match the expected attestation.")

    expected_cases = verification.get("cases")
    actual_cases = actual["verification"]["cases"]
    if not isinstance(expected_cases, list) or len(expected_cases) != len(
        actual_cases
    ):
        _fail("Expected deterministic verification cases are invalid.")
    for expected_case, actual_case in zip(expected_cases, actual_cases):
        if not isinstance(expected_case, Mapping):
            _fail("Expected deterministic case must be a mapping.")
        for field in ("name", "batch_size", "inputs"):
            if expected_case.get(field) != actual_case[field]:
                _fail(
                    f"Live deterministic case {actual_case['name']} field "
                    f"{field} does not match the expected attestation."
                )
        expected_outputs = expected_case.get("outputs")
        if not isinstance(expected_outputs, Mapping):
            _fail("Expected deterministic outputs record is invalid.")
        portable_backends = ["eager", "torchscript"]
        if actual_case["name"] == "structured":
            portable_backends.extend(
                ("eager_fp16_ray", "torchscript_fp16_ray")
            )
        for backend in portable_backends:
            if expected_outputs.get(backend) != actual_case["outputs"][backend]:
                _fail(
                    f"Live {backend} output for case {actual_case['name']} "
                    "does not match the expected attestation."
                )
    return actual
