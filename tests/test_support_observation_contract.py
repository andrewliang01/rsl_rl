from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any

from rsl_rl.modules import (
    unpack_ray_event_support_observation,
    unpack_support_motion_observation,
)


class _RejectLocalScalarMode(TorchDispatchMode):
    """Reject tensor-to-Python scalar extraction while unpacking observations."""

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


def test_ray_event_unpack_has_no_host_scalar_and_accepts_packet_age() -> None:
    """Accept honest RayCaster packet age with zero per-return age."""
    event = torch.zeros(2, 3, 5, 4, 8)
    event[:, :, 3] = 0.1
    event[:, :, 4] = 1.0
    event[:, :, 0, 0, 0] = 1.0
    event[:, :, 1, 0, 0] = 1.0

    with _RejectLocalScalarMode():
        unpacked = unpack_ray_event_support_observation(event)

    assert unpacked.range_m.shape == (2, 3, 4, 8)
    assert unpacked.return_valid[:, :, 0, 0].all()
    assert torch.equal(unpacked.packet_age_s, torch.full((2, 3), 0.1))
    assert unpacked.finite_gate.all()


def test_ray_event_unpack_reports_channel_contract_failures_as_gate() -> None:
    """Report nonbinary, nonbroadcast, and invalid-cell mutation per row."""
    event = torch.zeros(3, 1, 5, 2, 2)
    event[:, :, 4] = 1.0
    event[0, :, 1, 0, 0] = 0.5
    event[1, :, 3, 0, 1] = 0.1
    event[2, :, 0, 0, 0] = 1.0

    with _RejectLocalScalarMode():
        unpacked = unpack_ray_event_support_observation(event)

    assert torch.equal(unpacked.finite_gate, torch.zeros(3, dtype=torch.bool))


def test_support_motion_unpack_is_exact_and_tensor_gated() -> None:
    """Recover K transforms, two feet, and phase from one compact vector."""
    history_length = 2
    rotation = torch.eye(3).reshape(1, 1, 9).expand(2, history_length, -1)
    translation = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    feet = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    phase = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    observation = torch.cat(
        (
            rotation.flatten(start_dim=1),
            translation.flatten(start_dim=1),
            feet.flatten(start_dim=1),
            phase,
        ),
        dim=-1,
    )

    with _RejectLocalScalarMode():
        unpacked = unpack_support_motion_observation(
            observation,
            history_length=history_length,
        )

    assert torch.equal(unpacked.history_body_to_current_rotation, rotation.reshape(2, 2, 3, 3))
    assert torch.equal(unpacked.history_body_to_current_translation, translation)
    assert torch.equal(unpacked.current_foot_centres_body, feet)
    assert torch.equal(unpacked.gait_phase_sin_cos, phase)
    assert unpacked.finite_gate.all()


def test_support_motion_nonfinite_row_sets_gate_without_host_error() -> None:
    """Keep malformed dynamic input on device for the model fail gate."""
    observation = torch.zeros(2, 20)
    observation[0, 0] = torch.nan

    with _RejectLocalScalarMode():
        unpacked = unpack_support_motion_observation(
            observation,
            history_length=1,
        )

    assert torch.equal(unpacked.finite_gate, torch.tensor([False, True]))
