import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any

import pytest

from rsl_rl.modules import CausalCommandFootSupportProjector


class _RejectLocalScalarMode(TorchDispatchMode):
    """Reject implicit tensor-to-Python scalar extraction in a fast path."""

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


def _projector() -> CausalCommandFootSupportProjector:
    return CausalCommandFootSupportProjector(
        gait_cycle_s=0.8,
        min_horizon_s=0.05,
        max_horizon_s=0.8,
    )


def test_phase_anchors_produce_left_full_and_right_half_cycle() -> None:
    """Project the next, not current, anchor when phase is exactly aligned."""
    projector = _projector()
    feet = torch.tensor([[[0.0, 0.1, -0.8], [0.0, -0.1, -0.8]]])
    command = torch.tensor([[1.0, 0.0, 0.0]])
    phase = torch.tensor([[0.0, 1.0]])
    output = projector(feet, command, phase)

    torch.testing.assert_close(
        output.support_horizon_s,
        torch.tensor([[0.8, 0.4]]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert output.finite_gate.all()
    torch.testing.assert_close(
        output.future_centres_body,
        torch.tensor([[[0.8, 0.1, -0.8], [0.4, -0.1, -0.8]]]),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_constant_twist_uses_stable_exact_se2_displacement() -> None:
    """Match analytic circular-arc displacement without a small-yaw branch."""
    projector = _projector()
    feet = torch.zeros(1, 2, 3)
    angular_rate = math.pi / 0.8
    output = projector(
        feet,
        torch.tensor([[1.0, 0.0, angular_rate]]),
        torch.tensor([[0.0, 1.0]]),
    )

    torch.testing.assert_close(
        output.future_centres_body[0, 0, :2],
        torch.tensor([0.0, 1.6 / math.pi]),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        output.future_centres_body[0, 1, :2],
        torch.tensor([0.8 / math.pi, 0.8 / math.pi]),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_projection_retains_finite_command_and_foot_gradients() -> None:
    """Keep the causal query generator differentiable for joint training."""
    projector = _projector()
    feet = torch.tensor(
        [[[0.1, 0.1, -0.8], [0.1, -0.1, -0.8]]],
        requires_grad=True,
    )
    command = torch.tensor([[0.5, -0.2, 0.3]], requires_grad=True)
    output = projector(feet, command, torch.tensor([[1.0, 0.0]]))
    loss = output.future_centres_body.square().sum()
    loss.backward()

    assert feet.grad is not None and torch.isfinite(feet.grad).all()
    assert command.grad is not None and torch.isfinite(command.grad).all()
    assert feet.grad.abs().sum() > 0.0
    assert command.grad.abs().sum() > 0.0


def test_receipt_rejects_terrain_planning_and_real_accuracy_claims() -> None:
    """Describe future centres as causal queries, not planned touchdowns."""
    receipt = _projector().receipt()

    assert receipt["terrain_aware_footstep_planner"] is False
    assert receipt["simulator_contact_truth"] is False
    assert receipt["simulator_terrain_truth"] is False
    assert receipt["predicted_touchdown_accuracy_validated"] is False
    assert receipt["training_ready"] is False
    assert receipt["g1_closed_loop_validated"] is False


def test_native_training_has_no_host_scalar_and_matches_audited_path() -> None:
    """Return identical projection and a tensor gate without host extraction."""
    projector = _projector()
    feet = torch.tensor([[[0.1, 0.1, -0.8], [0.1, -0.1, -0.8]]])
    command = torch.tensor([[0.5, -0.2, 0.3]])
    phase = torch.tensor([[1.0, 0.0]])
    audited = projector(feet, command, phase)

    with _RejectLocalScalarMode():
        fast = projector.forward_native_training(feet, command, phase)

    assert torch.equal(fast.future_centres_body, audited.future_centres_body)
    assert torch.equal(fast.support_horizon_s, audited.support_horizon_s)
    assert torch.equal(fast.finite_gate, audited.finite_gate)
    assert fast.finite_gate.all()


def test_native_training_reports_zero_phase_with_tensor_gate() -> None:
    """Keep an undefined phase on device and mark its row invalid."""
    projector = _projector()
    with _RejectLocalScalarMode():
        output = projector.forward_native_training(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 2),
        )

    assert torch.equal(output.finite_gate, torch.tensor([False]))
    assert torch.isfinite(output.future_centres_body).all()


def test_rejects_zero_phase_vector_and_mixed_horizon_order() -> None:
    """Fail closed on an undefined clock phase or inverted horizon bounds."""
    with pytest.raises(ValueError, match="max_horizon"):
        CausalCommandFootSupportProjector(
            gait_cycle_s=0.8,
            min_horizon_s=0.5,
            max_horizon_s=0.1,
        )

    with pytest.raises(ValueError, match="nonzero phase"):
        _projector()(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 2),
        )
