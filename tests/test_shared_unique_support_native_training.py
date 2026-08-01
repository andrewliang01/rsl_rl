import copy
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any, NoReturn

import pytest

import rsl_rl.modules.shared_unique_support_actor as actor_module
from rsl_rl.modules import (
    SharedUniqueSupportActorAdapter,
    SupportMaskProvenance,
)

SCORE_DIM = 5
VALUE_DIM = 3
PROPRIO_DIM = 7
ACTION_DIM = 4


class _RejectLocalScalarMode(TorchDispatchMode):
    """Reject implicit Tensor-to-Python scalar extraction during forward."""

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


def _provenance() -> SupportMaskProvenance:
    return SupportMaskProvenance(
        geometry_source="calibrated_lidar_ray_geometry",
        uses_proprioception=True,
        uses_gait_phase=True,
    )


def _inputs(
    batch_size: int = 3,
    num_tokens: int = 24,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(7101)
    score_features = torch.randn(batch_size, num_tokens, SCORE_DIM, generator=generator)
    terrain_values = torch.randn(batch_size, num_tokens, VALUE_DIM, generator=generator)
    proprio = torch.randn(batch_size, PROPRIO_DIM, generator=generator)
    token_valid = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    role_eligibility = torch.zeros(batch_size, 4, num_tokens, dtype=torch.bool)
    role_eligibility[:, 0, 0:6] = True
    role_eligibility[:, 1, 6:12] = True
    role_eligibility[:, 2, 12:18] = True
    role_eligibility[:, 3, 18:24] = True
    return (
        score_features,
        terrain_values,
        proprio,
        token_valid,
        role_eligibility,
    )


def _model() -> SharedUniqueSupportActorAdapter:
    return SharedUniqueSupportActorAdapter(
        SCORE_DIM,
        VALUE_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        total_budget=8,
    )


def test_native_training_has_no_host_scalar_hash_or_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep production native forward free of audit-only host work."""
    torch.manual_seed(7103)
    model = _model().eval()

    def forbidden_hash(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("Native training called tensor SHA")

    def forbidden_intervention(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("Native training called offline intervention")

    monkeypatch.setattr(actor_module, "_tensor_sha256", forbidden_hash)
    monkeypatch.setattr(model, "_apply_intervention", forbidden_intervention)

    with torch.inference_mode(), _RejectLocalScalarMode():
        action, value, finite_gate = model.forward_native_training(*_inputs(), mask_provenance=_provenance())

    assert action.shape == (3, ACTION_DIM)
    assert value.shape == (3, 1)
    assert finite_gate.dtype == torch.bool
    assert finite_gate.shape == (3,)
    assert finite_gate.all()


def test_native_training_is_bitwise_equal_to_diagnostic_native() -> None:
    """Match the retained offline native path on finite CPU inputs."""
    torch.manual_seed(7109)
    model = _model().eval()
    inputs = _inputs()

    with torch.inference_mode():
        fast_action, fast_value, finite_gate = model.forward_native_training(*inputs, mask_provenance=_provenance())
        audited_action, audited_value, _ = model.forward_with_diagnostics(
            *inputs,
            mask_provenance=_provenance(),
            intervention_mode="native",
        )

    assert torch.equal(fast_action, audited_action)
    assert torch.equal(fast_value, audited_value)
    assert finite_gate.all()


def test_native_training_returns_per_batch_finite_contract_gate() -> None:
    """Report invalid rows as a tensor without a host-side exception."""
    torch.manual_seed(7111)
    model = _model().eval()
    inputs = list(_inputs())

    # Row zero contains a non-finite value only in an invalid, unselected cell.
    inputs[3][0, -1] = False
    inputs[4][0, :, -1] = False
    inputs[1][0, -1, 0] = torch.nan
    # Row one violates the role/token contract, also outside selection.
    inputs[3][1, -1] = False
    inputs[4][1, 0, -1] = True

    with torch.inference_mode(), _RejectLocalScalarMode():
        action, value, finite_gate = model.forward_native_training(*inputs, mask_provenance=_provenance())

    assert torch.isfinite(action).all()
    assert torch.isfinite(value).all()
    assert torch.equal(
        finite_gate,
        torch.tensor([False, False, True]),
    )


def test_native_training_preserves_native_outputs_and_gradients() -> None:
    """Preserve native CPU results and trainable parameter gradients."""
    torch.manual_seed(7117)
    fast_model = _model()
    audited_model = copy.deepcopy(fast_model)
    inputs = _inputs()

    fast_action, fast_value, finite_gate = fast_model.forward_native_training(*inputs, mask_provenance=_provenance())
    audited_action, audited_value, _ = audited_model.forward_with_diagnostics(
        *inputs,
        mask_provenance=_provenance(),
        intervention_mode="native",
    )
    fast_loss = fast_action.square().mean() + fast_value.square().mean()
    audited_loss = audited_action.square().mean() + audited_value.square().mean()
    fast_loss.backward()
    audited_loss.backward()

    assert finite_gate.all()
    assert torch.equal(fast_action, audited_action)
    assert torch.equal(fast_value, audited_value)
    audited_parameters = dict(audited_model.named_parameters())
    for name, parameter in fast_model.named_parameters():
        fast_gradient = parameter.grad
        audited_gradient = audited_parameters[name].grad
        assert fast_gradient is not None, name
        assert audited_gradient is not None, name
        assert torch.isfinite(fast_gradient).all(), name
        assert torch.equal(fast_gradient, audited_gradient), name


def test_native_actor_training_skips_critic_and_host_audit_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove actor-only native forward never enters critic or audit paths."""
    torch.manual_seed(7121)
    model = _model().eval()

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("Actor-only native forward entered forbidden path")

    monkeypatch.setattr(actor_module, "_tensor_sha256", forbidden)
    monkeypatch.setattr(model, "_apply_intervention", forbidden)
    monkeypatch.setattr(model.critic_backbone, "forward", forbidden)
    monkeypatch.setattr(model.value_head, "forward", forbidden)

    with torch.inference_mode(), _RejectLocalScalarMode():
        action, finite_gate = model.forward_native_actor_training(*_inputs(), mask_provenance=_provenance())

    assert action.shape == (3, ACTION_DIM)
    assert finite_gate.dtype == torch.bool
    assert finite_gate.shape == (3,)
    assert finite_gate.all()


def test_native_actor_training_matches_action_and_actor_gradients() -> None:
    """Match the full native action graph while leaving critic gradients absent."""
    torch.manual_seed(7127)
    actor_model = _model()
    full_model = copy.deepcopy(actor_model)
    inputs = _inputs()

    actor_action, actor_gate = actor_model.forward_native_actor_training(*inputs, mask_provenance=_provenance())
    full_action, _, full_gate = full_model.forward_native_training(*inputs, mask_provenance=_provenance())
    actor_action.square().mean().backward()
    full_action.square().mean().backward()

    assert torch.equal(actor_action, full_action)
    assert torch.equal(actor_gate, full_gate)
    full_parameters = dict(full_model.named_parameters())
    for name, parameter in actor_model.named_parameters():
        full_gradient = full_parameters[name].grad
        if name.startswith(("critic_backbone.", "value_head.")):
            assert parameter.grad is None, name
            assert full_gradient is None, name
            continue
        assert parameter.grad is not None, name
        assert full_gradient is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.equal(parameter.grad, full_gradient), name


def test_native_actor_gate_is_independent_of_critic_parameters() -> None:
    """Exclude non-finite critic parameters from actor-only outputs and gate."""
    torch.manual_seed(7129)
    model = _model().eval()
    inputs = _inputs()
    with torch.no_grad():
        for parameter in model.critic_backbone.parameters():
            parameter.fill_(torch.nan)
        for parameter in model.value_head.parameters():
            parameter.fill_(torch.nan)

    with torch.inference_mode():
        actor_action, actor_gate = model.forward_native_actor_training(*inputs, mask_provenance=_provenance())
        full_action, full_value, full_gate = model.forward_native_training(*inputs, mask_provenance=_provenance())

    assert torch.equal(actor_action, full_action)
    assert actor_gate.all()
    assert not torch.isfinite(full_value).any()
    assert not full_gate.any()
