from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any

from rsl_rl.modules import (
    CalibratedSphericalSupportRoleGeometry,
    CausalCommandFootSupportProjector,
    CausalSphericalSupportEvidencePipeline,
    SharedUniqueSupportActorAdapter,
)


class _RejectLocalScalarMode(TorchDispatchMode):
    """Reject implicit tensor-to-Python scalar extraction end to end."""

    def __torch_dispatch__(
        self,
        func: Any,
        _types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("Tensor-to-host scalar extraction detected")
        return func(*args, **({} if kwargs is None else kwargs))


def _pipeline() -> CausalSphericalSupportEvidencePipeline:
    rays = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ]
    )
    geometry = CalibratedSphericalSupportRoleGeometry(
        rays,
        torch.eye(3),
        torch.zeros(3),
        external_calibration_sha256="d" * 64,
        current_radius=0.08,
        landing_radius=0.08,
        vertical_half_extent=0.05,
        min_range=0.1,
        max_range=5.0,
        range_strata_edges=(0.5, 1.5, 3.0),
        age_strata_edges=(0.01, 0.05, 0.2),
    )
    projector = CausalCommandFootSupportProjector(
        gait_cycle_s=0.8,
        min_horizon_s=0.05,
        max_horizon_s=0.8,
    )
    return CausalSphericalSupportEvidencePipeline(projector, geometry)


def _inputs() -> tuple[torch.Tensor, ...]:
    rotation = torch.eye(3).view(1, 1, 3, 3)
    return (
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4, dtype=torch.bool),
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1),
        rotation,
        torch.zeros(1, 1, 3),
        torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
        torch.zeros(1, 3),
        torch.tensor([[0.0, 1.0]]),
    )


def test_native_pipeline_and_actor_have_no_host_scalar_and_match_audit() -> None:
    """Cover event, projection, geometry, unique selection, action, and value."""
    torch.manual_seed(8101)
    pipeline = _pipeline()
    actor = SharedUniqueSupportActorAdapter(
        score_feature_dim=pipeline.geometry.score_feature_dim,
        terrain_value_dim=pipeline.geometry.terrain_value_dim,
        proprio_dim=6,
        action_dim=3,
        total_budget=8,
    ).eval()
    audited = pipeline(*_inputs())
    with torch.inference_mode():
        audited_action, audited_value = actor(
            audited.geometry.score_features,
            audited.geometry.terrain_values,
            torch.zeros(1, 6),
            audited.geometry.token_valid,
            audited.geometry.role_eligibility,
            mask_provenance=pipeline.geometry.provenance(),
        )

    with torch.inference_mode(), _RejectLocalScalarMode():
        fast = pipeline.forward_native_training(*_inputs())
        action, value, actor_gate = actor.forward_native_training(
            fast.geometry.score_features,
            fast.geometry.terrain_values,
            torch.zeros(1, 6),
            fast.geometry.token_valid,
            fast.geometry.role_eligibility,
            mask_provenance=pipeline.geometry.provenance(),
        )
        combined_gate = fast.finite_gate & actor_gate

    assert torch.equal(action, audited_action)
    assert torch.equal(value, audited_value)
    assert combined_gate.all()
    assert fast.geometry.candidate_mask.sum().item() == 2


def test_native_pipeline_propagates_projector_failure_as_tensor_gate() -> None:
    """Keep an undefined phase finite but mark the combined row invalid."""
    pipeline = _pipeline()
    inputs = list(_inputs())
    inputs[-1] = torch.zeros(1, 2)

    with _RejectLocalScalarMode():
        output = pipeline.forward_native_training(*inputs)

    assert torch.equal(output.finite_gate, torch.tensor([False]))
    assert torch.isfinite(output.geometry.score_features).all()


def test_pipeline_receipt_keeps_training_and_gpu_claims_false() -> None:
    """Do not promote CPU component closure to a task or performance result."""
    receipt = _pipeline().receipt()

    assert receipt["selected_only_actor_connected"] is False
    assert receipt["registered_lab_task"] is False
    assert receipt["gpu_latency_measured"] is False
    assert receipt["training_ready"] is False
    assert receipt["g1_closed_loop_validated"] is False
