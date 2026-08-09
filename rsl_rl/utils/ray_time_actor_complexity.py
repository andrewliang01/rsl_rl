# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-visible CPU complexity receipts for the frozen Global Ray-Time actor.

The primitive constructs the real :class:`PropMLPElevationFusionModel` rather
than a shape-only surrogate.  It reports exact trainable-parameter counts and
an analytically verified subtotal of multiply-accumulate pairs executed by
``Linear``, ``Conv1d``, and ``Conv2d`` modules.  An independent CPU dispatcher
census cross-checks that subtotal without initializing or querying an
accelerator.  Unconverted normalization, pooling, reductions, and elementwise
operations keep the receipt fail-visible as ``partial``; the tool never exposes
a fabricated ``total_mac``.

This is an operation-count tool.  It performs no timing warm-up, latency
measurement, GPU query, or target-platform P99 claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils._python_dispatch import TorchDispatchMode

from rsl_rl.models.prop_mlp_elevation_fusion_model import (
    PropMLPElevationFusionModel,
)
from rsl_rl.modules.ray_event_ablation import RayEventAblationRouter
from rsl_rl.modules.ray_time_attention_encoder import RayTimeAttentionEncoder


SCHEMA_NAME: Final[str] = "ray-time-actor-complexity-receipt"
SCHEMA_VERSION: Final[int] = 1
TOOL_CONTRACT: Final[str] = "ray_time_actor_complexity_partial_v1"
MODEL_SEED: Final[int] = 20_260_801
BATCH_SIZE: Final[int] = 1
PROPRIO_DIM: Final[int] = 96
ACTION_DIM: Final[int] = 29
IMAGE_HEIGHT: Final[int] = 16
IMAGE_WIDTH: Final[int] = 96
RAY_CHANNELS: Final[tuple[str, str]] = ("range_m", "return_valid")
VARIANTS: Final[tuple[str, ...]] = (
    "global_k1",
    "global_k5",
    "matched_post_raster_nearest_union_k1",
)
COUNTED_MODULE_TYPES: Final[tuple[type[nn.Module], ...]] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Linear,
)


class ComplexityReceiptError(RuntimeError):
    """Fail-closed model, schema, counting, or artifact error."""


@dataclass(frozen=True)
class ConstructedVariant:
    variant: str
    model: PropMLPElevationFusionModel
    observations: TensorDict
    input_audit: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    header = _canonical_bytes(
        {
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
        }
    )
    return _sha256(header + b"\0" + detached.numpy().tobytes(order="C"))


def _state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_bytes(list(detached.shape)))
        digest.update(b"\0")
        digest.update(detached.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ComplexityReceiptError(f"Cannot read source file {resolved}: {error}") from error
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _git_binding(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComplexityReceiptError(
            f"Cannot bind source repository {repository}: {error}"
        ) from error
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ComplexityReceiptError(f"Invalid Git HEAD at {repository}: {head!r}")
    return {
        "path": str(repository),
        "head": head,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _fixed_proprio() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, PROPRIO_DIM, dtype=torch.float32).reshape(
        BATCH_SIZE,
        PROPRIO_DIM,
    )


def _fixed_native_returns(
    history_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if history_length <= 0:
        raise ComplexityReceiptError("history_length must be positive.")
    rows = torch.arange(IMAGE_HEIGHT, dtype=torch.int64)[None, None, :, None]
    columns = torch.arange(IMAGE_WIDTH, dtype=torch.int64)[None, None, None, :]
    frames = torch.arange(history_length, dtype=torch.int64)[None, :, None, None]
    code = rows * 97 + columns * 17 + frames * 29
    valid = (code.remainder(7) != 0).expand(
        BATCH_SIZE,
        history_length,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    metric_range = 0.2 + code.remainder(560).to(torch.float32) / 100.0
    metric_range = metric_range.expand_as(valid).clone()
    metric_range = torch.where(valid, metric_range, torch.zeros_like(metric_range))
    packet_age = (
        torch.arange(history_length - 1, -1, -1, dtype=torch.float32)
        * 0.1
    ).reshape(1, history_length).expand(BATCH_SIZE, history_length)
    cell_age_offset = columns.to(torch.float32).remainder(5) * 0.001
    return_age = packet_age[:, :, None, None] + cell_age_offset
    return_age = return_age.expand_as(metric_range).clone()
    return_age = torch.where(valid, return_age, torch.zeros_like(return_age))
    frame_valid = torch.ones(BATCH_SIZE, history_length, dtype=torch.bool)
    return metric_range, valid, return_age, packet_age, frame_valid


def _ordinary_ray_history(history_length: int) -> tuple[torch.Tensor, dict[str, Any]]:
    metric_range, valid, _, _, frame_valid = _fixed_native_returns(history_length)
    history = torch.stack((metric_range, valid.to(metric_range.dtype)), dim=2).to(
        torch.float16
    )
    return history, {
        "producer": "fixed_native_packet_raster",
        "source_history_length": history_length,
        "actor_history_length": history_length,
        "frame_valid": frame_valid.tolist(),
        "source_window_sha256": _sha256(
            _tensor_sha256(metric_range).encode("ascii")
            + _tensor_sha256(valid).encode("ascii")
        ),
        "post_raster_reduction": None,
    }


def _matched_union_ray_history() -> tuple[torch.Tensor, dict[str, Any]]:
    metric_range, valid, return_age, packet_age, frame_valid = _fixed_native_returns(5)
    router = RayEventAblationRouter(
        geometry="native",
        time_association="correct",
        temporal_baseline="age_zero",
        history_reduction="exact_union_k1",
    )
    reduced = router(metric_range, valid, return_age, packet_age, frame_valid)
    if reduced.range_m.shape != (BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ComplexityReceiptError("Matched reducer did not emit one range frame.")
    if bool(reduced.return_age_s.any()) or bool(reduced.packet_age_s.any()):
        raise ComplexityReceiptError("Matched age-zero reducer leaked time into its output.")
    history = torch.stack(
        (reduced.range_m, reduced.return_valid.to(reduced.range_m.dtype)),
        dim=2,
    ).to(torch.float16)
    source_window_hash = _sha256(
        _tensor_sha256(metric_range).encode("ascii")
        + _tensor_sha256(valid).encode("ascii")
        + _tensor_sha256(return_age).encode("ascii")
        + _tensor_sha256(packet_age).encode("ascii")
        + _tensor_sha256(frame_valid).encode("ascii")
    )
    return history, {
        "producer": "RayEventAblationRouter",
        "source_history_length": 5,
        "actor_history_length": 1,
        "source_window_sha256": source_window_hash,
        "post_raster_reduction": {
            "history_reduction": "exact_union_k1",
            "winner_policy": "nearest_range_then_largest_age_then_lowest_frame_index",
            "temporal_baseline": "age_zero",
            "actor_channels": list(RAY_CHANNELS),
            "winner_age_and_source_are_actor_invisible": True,
            "collision_cell_count": int(
                reduced.diagnostics["exact_union_collision_cell_count"].sum()
            ),
            "winner_frame_index_sha256": _tensor_sha256(
                reduced.diagnostics["exact_union_winner_frame_index"]
            ),
            "reducer_parameters": 0,
            "reducer_mac_included_in_actor_subtotal": False,
        },
    }


def construct_global_ray_time_actor(variant: str) -> ConstructedVariant:
    """Construct one frozen actor and deterministic batch-one observation."""
    if variant not in VARIANTS:
        raise ComplexityReceiptError(
            f"Unknown variant {variant!r}; expected one of {VARIANTS}."
        )
    if variant == "global_k1":
        history_length = 1
        ray_history, producer_audit = _ordinary_ray_history(1)
    elif variant == "global_k5":
        history_length = 5
        ray_history, producer_audit = _ordinary_ray_history(5)
    else:
        history_length = 1
        ray_history, producer_audit = _matched_union_ray_history()

    proprio = _fixed_proprio()
    observations = TensorDict(
        {
            "policy": proprio,
            "mid360_policy": ray_history,
        },
        batch_size=[BATCH_SIZE],
    )
    torch.manual_seed(MODEL_SEED)
    model = PropMLPElevationFusionModel(
        obs=observations,
        obs_groups={"actor": ["policy", "mid360_policy"]},
        obs_set="actor",
        output_dim=ACTION_DIM,
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        elevation_encoder_type="ray_time",
        ray_time_set="mid360_policy",
        ray_time_history_length=history_length,
        ray_time_spatial_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        ray_time_use_query_attention=False,
        ray_time_fusion_mode="global",
        prop_feature_dim=64,
        prop_hidden_dims=[128],
        vision_feature_dim=64,
    ).cpu()
    model.eval()
    input_audit = {
        "batch_size": BATCH_SIZE,
        "observation_groups": {
            "policy": {
                "shape": list(proprio.shape),
                "dtype": str(proprio.dtype),
                "sha256": _tensor_sha256(proprio),
                "semantics": "proprioception",
            },
            "mid360_policy": {
                "shape": list(ray_history.shape),
                "dtype": str(ray_history.dtype),
                "sha256": _tensor_sha256(ray_history),
                "channel_order": list(RAY_CHANNELS),
                "unknown_encoding": [0.0, 0.0],
                "spatial_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
            },
        },
        "actor_output": {
            "shape": [BATCH_SIZE, ACTION_DIM],
            "dtype": "torch.float32",
            "semantics": "deterministic_action_mean",
        },
        "producer_audit": producer_audit,
    }
    return ConstructedVariant(variant, model, observations, input_audit)


def _unique_named_trainable_parameters(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=True)
        if parameter.requires_grad
    ]


def audit_trainable_parameters(
    model: PropMLPElevationFusionModel,
    observations: TensorDict,
) -> dict[str, Any]:
    """Count total and PPO-active trainable parameters without branch guessing."""
    parameters = _unique_named_trainable_parameters(model)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(MODEL_SEED + 1)
    action_sample = model(observations, stochastic_output=True)
    if tuple(action_sample.shape) != (BATCH_SIZE, ACTION_DIM):
        raise ComplexityReceiptError(
            f"Actor emitted unexpected shape {tuple(action_sample.shape)}."
        )
    fixed_action = torch.zeros_like(action_sample)
    objective = -model.get_output_log_prob(fixed_action).sum()
    objective.backward()

    active: list[tuple[str, nn.Parameter]] = []
    inactive: list[tuple[str, nn.Parameter]] = []
    for name, parameter in parameters:
        if parameter.grad is None:
            inactive.append((name, parameter))
        else:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ComplexityReceiptError(
                    f"Active parameter {name!r} has a non-finite gradient."
                )
            active.append((name, parameter))
    model.zero_grad(set_to_none=True)
    total_numel = sum(parameter.numel() for _, parameter in parameters)
    active_numel = sum(parameter.numel() for _, parameter in active)
    inactive_numel = sum(parameter.numel() for _, parameter in inactive)
    if active_numel + inactive_numel != total_numel:
        raise ComplexityReceiptError("Active/inactive parameter partition is inconsistent.")
    return {
        "definition": (
            "Unique requires_grad parameters; active means grad is not None after "
            "the real stochastic actor forward and PPO log-probability path on "
            "the fixed batch-one observation."
        ),
        "total_trainable_parameters": int(total_numel),
        "active_trainable_parameters": int(active_numel),
        "inactive_trainable_parameters": int(inactive_numel),
        "trainable_tensor_count": len(parameters),
        "active_trainable_tensor_count": len(active),
        "inactive_trainable_tensor_count": len(inactive),
        "active_parameter_names_sha256": _sha256(
            _canonical_bytes([name for name, _ in active])
        ),
        "inactive_parameter_names": [
            {"name": name, "numel": int(parameter.numel())}
            for name, parameter in inactive
        ],
        "duplicate_parameters_counted_once": True,
    }


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor(item)
            except ComplexityReceiptError:
                continue
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except ComplexityReceiptError:
                continue
    raise ComplexityReceiptError("Hook output contains no tensor.")


class ModuleMacCounter(AbstractContextManager["ModuleMacCounter"]):
    """Count exact Conv/Linear MAC pairs from real module output shapes."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.handles: list[Any] = []
        self.records: list[dict[str, Any]] = []
        self._module_names = {id(item): name for name, item in module.named_modules()}

    def __enter__(self) -> "ModuleMacCounter":
        for submodule in self.module.modules():
            if isinstance(submodule, COUNTED_MODULE_TYPES):
                self.handles.append(
                    submodule.register_forward_hook(self._hook)
                )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return None

    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        output_tensor = _first_tensor(output)
        input_tensor = _first_tensor(inputs)
        if isinstance(module, nn.Linear):
            mac = output_tensor.numel() * module.in_features
            formula = "output_numel * in_features"
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            kernel_elements = 1
            for value in module.kernel_size:
                kernel_elements *= int(value)
            mac = (
                output_tensor.numel()
                * (module.in_channels // module.groups)
                * kernel_elements
            )
            formula = "output_numel * (in_channels/groups) * product(kernel_size)"
        else:
            raise ComplexityReceiptError(
                f"Unexpected counted module type {type(module).__qualname__}."
            )
        self.records.append(
            {
                "module": self._module_names[id(module)],
                "type": type(module).__qualname__,
                "input_shape": list(input_tensor.shape),
                "output_shape": list(output_tensor.shape),
                "mac": int(mac),
                "formula": formula,
                "bias_additions_included": False,
            }
        )

    def summary(self) -> dict[str, Any]:
        by_type: defaultdict[str, int] = defaultdict(int)
        for record in self.records:
            by_type[record["type"]] += record["mac"]
        return {
            "known_mac_subtotal": int(sum(by_type.values())),
            "known_mac_by_module_type": dict(sorted(by_type.items())),
            "counted_module_calls": len(self.records),
            "calls": self.records,
        }


class OperatorCensusMode(TorchDispatchMode):
    """Independently count executed CPU operators and dense Conv/Linear MACs."""

    _SUPPORTED_OPERATORS: Final[tuple[str, ...]] = (
        "aten.conv1d.default",
        "aten.conv2d.default",
        "aten.linear.default",
    )

    def __init__(self) -> None:
        super().__init__()
        self.operator_calls: Counter[str] = Counter()
        self.supported_records: list[dict[str, Any]] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        operator = str(func)
        self.operator_calls[operator] += 1
        output = func(*args, **dict(kwargs or {}))
        if operator not in self._SUPPORTED_OPERATORS:
            return output

        output_tensor = _first_tensor(output)
        input_tensor = _first_tensor(args[0])
        weight = _first_tensor(args[1])
        if operator == "aten.linear.default":
            if weight.ndim != 2 or input_tensor.shape[-1] != weight.shape[-1]:
                raise ComplexityReceiptError("Dispatcher observed an unsupported Linear schema.")
            mac = output_tensor.numel() * weight.shape[-1]
            formula = "output_numel * weight_in_features"
        else:
            expected_rank = 3 if operator == "aten.conv1d.default" else 4
            if input_tensor.ndim != expected_rank or weight.ndim != expected_rank:
                raise ComplexityReceiptError(
                    f"Dispatcher observed an unsupported {operator} tensor rank."
                )
            kernel_elements = 1
            for extent in weight.shape[2:]:
                kernel_elements *= int(extent)
            # Conv weight is [out_channels, in_channels/groups, *kernel].
            mac = output_tensor.numel() * weight.shape[1] * kernel_elements
            formula = "output_numel * weight_in_channels_per_group * product(kernel)"
        self.supported_records.append(
            {
                "operator": operator,
                "input_shape": list(input_tensor.shape),
                "weight_shape": list(weight.shape),
                "output_shape": list(output_tensor.shape),
                "mac": int(mac),
                "formula": formula,
            }
        )
        return output

    def summary(self) -> dict[str, Any]:
        by_operator: defaultdict[str, int] = defaultdict(int)
        for record in self.supported_records:
            by_operator[record["operator"]] += record["mac"]
        unsupported = [
            {"operator": operator, "calls": int(calls)}
            for operator, calls in sorted(self.operator_calls.items())
            if operator not in self._SUPPORTED_OPERATORS
        ]
        return {
            "backend": "torch.utils._python_dispatch.TorchDispatchMode",
            "device": "cpu",
            "accelerator_queries": False,
            "supported_mac_subtotal": int(sum(by_operator.values())),
            "supported_mac_by_operator": dict(sorted(by_operator.items())),
            "supported_calls": self.supported_records,
            "operator_census": [
                {"operator": operator, "calls": int(calls)}
                for operator, calls in sorted(self.operator_calls.items())
            ],
            "unconverted_operators": unsupported,
        }


def _dispatch_census_forward(
    model: PropMLPElevationFusionModel,
    observations: TensorDict,
) -> dict[str, Any]:
    census = OperatorCensusMode()
    with census, torch.inference_mode():
        output = model(observations, stochastic_output=False)
    if tuple(output.shape) != (BATCH_SIZE, ACTION_DIM):
        raise ComplexityReceiptError("Dispatcher-counted actor output shape drifted.")
    return census.summary()


def audit_known_mac_subtotal(
    model: PropMLPElevationFusionModel,
    observations: TensorDict,
) -> dict[str, Any]:
    with ModuleMacCounter(model) as counter:
        with torch.inference_mode():
            output = model(observations, stochastic_output=False)
    if tuple(output.shape) != (BATCH_SIZE, ACTION_DIM):
        raise ComplexityReceiptError("Hook-counted actor output shape drifted.")
    hook_summary = counter.summary()
    dispatcher = _dispatch_census_forward(model, observations)
    dispatcher_crosscheck = {
        "scope": "All executed nn.Linear/Conv1d/Conv2d calls",
        "hook_known_mac_subtotal": hook_summary["known_mac_subtotal"],
        "dispatcher_supported_mac_subtotal": dispatcher["supported_mac_subtotal"],
        "exact_match": (
            hook_summary["known_mac_subtotal"]
            == dispatcher["supported_mac_subtotal"]
        ),
    }
    if not dispatcher_crosscheck["exact_match"]:
        raise ComplexityReceiptError(
            "CPU dispatcher disagrees with module hooks for Conv/Linear MACs."
        )
    return {
        "definition": (
            "One MAC is one multiplication accumulated into an output element. "
            "The known subtotal counts executed nn.Linear/Conv1d/Conv2d modules "
            "from their real output shapes; bias additions are excluded."
        ),
        **hook_summary,
        "dispatcher_crosscheck": dispatcher_crosscheck,
        "coverage": {
            "status": "partial",
            "formal_eligible": False,
            "total_mac_reported": False,
            "reason": (
                "The hook and CPU dispatcher subtotal exactly covers executed "
                "Linear/Conv1d/Conv2d only. Normalization, pooling, reductions, "
                "padding, indexing, and elementwise operators are inventoried but "
                "not converted to MACs, so the subtotal must not be renamed total MAC."
            ),
            "counted_module_types": [
                "torch.nn.Linear",
                "torch.nn.Conv1d",
                "torch.nn.Conv2d",
            ],
            "unconverted_operator_census": dispatcher["unconverted_operators"],
            "explicitly_uncovered_operator_classes": [
                "normalization",
                "activation",
                "pooling",
                "elementwise arithmetic",
                "reductions",
                "padding/indexing/view/cast",
                "bias additions",
            ],
        },
        "dispatcher_census": dispatcher,
    }


def _variant_receipt(variant: str) -> dict[str, Any]:
    constructed = construct_global_ray_time_actor(variant)
    model = constructed.model
    observations = constructed.observations
    with torch.inference_mode():
        output = model(observations, stochastic_output=False)
    if output.dtype != torch.float32 or tuple(output.shape) != (BATCH_SIZE, ACTION_DIM):
        raise ComplexityReceiptError(
            f"Variant {variant} output schema is {output.dtype}/{tuple(output.shape)}."
        )
    return {
        "variant": variant,
        "actor_family": "PropMLPElevationFusionModel",
        "encoder": "RayTimeAttentionEncoder",
        "fusion_mode": "global",
        "model_seed": MODEL_SEED,
        "model_state_sha256": _state_dict_sha256(model),
        "input_schema": constructed.input_audit,
        "parameters": audit_trainable_parameters(model, observations),
        "mac": audit_known_mac_subtotal(model, observations),
    }


def build_ray_time_actor_complexity_receipt() -> dict[str, Any]:
    """Build the fixed three-arm CPU receipt as a JSON-compatible object."""
    source_root = Path(__file__).resolve().parents[2]
    source_files = {
        "complexity_tool": Path(__file__),
        "actor_model": Path(sys.modules[PropMLPElevationFusionModel.__module__].__file__),
        "ray_time_encoder": Path(sys.modules[RayTimeAttentionEncoder.__module__].__file__),
        "matched_union_router": Path(sys.modules[RayEventAblationRouter.__module__].__file__),
    }
    variants = [_variant_receipt(variant) for variant in VARIANTS]
    receipt = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "contract": TOOL_CONTRACT,
        "created_at": _utc_now(),
        "scope": {
            "device": "cpu",
            "batch_size": BATCH_SIZE,
            "actor_action_scope": "deterministic_action_mean",
            "parameter_activity_scope": "stochastic_actor_plus_PPO_log_probability",
            "warmup": {
                "applicable": False,
                "reason": "Operation and parameter counting only; no timing is performed.",
            },
            "latency": {
                "measured": False,
                "target_platform_profiled": False,
                "p50_p95_p99_reported": False,
                "claim_boundary": "This receipt cannot support latency or real-time claims.",
            },
        },
        "source": {
            "git": _git_binding(source_root),
            "files": {
                name: _source_binding(path) for name, path in source_files.items()
            },
        },
        "tool_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensordict": importlib.metadata.version("tensordict"),
            "rsl_rl_distribution": importlib.metadata.version("rsl-rl-lib"),
            "platform": platform.platform(),
            "operator_census": (
                "torch.utils._python_dispatch.TorchDispatchMode CPU execution census"
            ),
        },
        "variants": variants,
        "formal_eligibility": {
            "eligible": False,
            "status": "partial",
            "reason": (
                "Trainable-parameter counts are exact, but MAC coverage is only "
                "an exact Conv/Linear subtotal. No total MAC is reported."
            ),
        },
    }
    receipt["payload_sha256"] = _sha256(_canonical_bytes(receipt))
    return receipt


def write_complexity_receipt_create_once(path: Path, receipt: Mapping[str, Any]) -> None:
    """Write one retained receipt without overwriting files or symlinks."""
    target = path.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(receipt),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except FileExistsError as error:
        raise ComplexityReceiptError(f"Refusing to overwrite receipt: {target}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The create-once file remains fail-visible if an I/O failure occurs;
        # never remove or retry over a possibly partial evidence artifact.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_ray_time_actor_complexity_receipt()
        write_complexity_receipt_create_once(args.output, receipt)
        print(
            json.dumps(
                {
                    "output": str(args.output.absolute()),
                    "payload_sha256": receipt["payload_sha256"],
                    "formal_eligible": False,
                    "mac_coverage": "partial",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except ComplexityReceiptError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
