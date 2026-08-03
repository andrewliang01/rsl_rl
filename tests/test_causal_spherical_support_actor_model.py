from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any

import pytest

from rsl_rl.models import CausalSphericalSupportActorModel
from rsl_rl.modules import CausalSphericalSupportEvidenceBatch


class _RejectLocalScalarMode(TorchDispatchMode):
    """Reject tensor-to-Python scalar extraction over the whole actor path."""

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


def _observations(batch_size: int = 2) -> TensorDict:
    event = torch.zeros(batch_size, 1, 5, 1, 4)
    event[:, :, 0] = 1.0
    event[:, :, 1] = 1.0
    event[:, :, 4] = 1.0
    rotation = torch.eye(3).reshape(1, 9).expand(batch_size, -1)
    translation = torch.zeros(batch_size, 3)
    feet = torch.tensor(
        [[1.0, 0.0, 0.0, -1.0, 0.0, 0.0]],
    ).expand(batch_size, -1)
    phase = torch.tensor([[0.0, 1.0]]).expand(batch_size, -1)
    motion = torch.cat((rotation, translation, feet, phase), dim=-1)
    return TensorDict(
        {
            "policy_a": torch.zeros(batch_size, 4),
            "policy_b": torch.zeros(batch_size, 2),
            "ray_event": event,
            "support_motion": motion,
        },
        batch_size=[batch_size],
    )


def _model(obs: TensorDict, **overrides: object) -> CausalSphericalSupportActorModel:
    kwargs: dict[str, object] = {
        "obs": obs,
        "obs_groups": {
            "actor": ["policy_a", "ray_event", "policy_b", "support_motion"]
        },
        "obs_set": "actor",
        "output_dim": 3,
        "ray_event_set": "ray_event",
        "support_motion_set": "support_motion",
        "command_indices": (1, 3, 5),
        "external_calibration_sha256": "a" * 64,
        "history_length": 1,
        "ray_spatial_size": (1, 4),
        "vertical_fov_degrees": (0.0, 1.0),
        "current_support_radius": 0.15,
        "future_support_radius": 0.15,
        "support_vertical_half_extent": 0.05,
        "total_budget": 8,
        "score_dim": 8,
        "value_embedding_dim": 8,
        "hidden_dim": 16,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.7,
            "std_type": "scalar",
        },
    }
    kwargs.update(overrides)
    return CausalSphericalSupportActorModel(**kwargs)


def test_actor_model_runs_distribution_contract_without_host_scalar() -> None:
    """Connect observation unpacking, geometry, unique selection, and PPO API."""
    torch.manual_seed(8201)
    obs = _observations()
    model = _model(obs).eval()

    with _RejectLocalScalarMode():
        deterministic = model(obs)
        stochastic = model(obs, stochastic_output=True)
        log_prob = model.get_output_log_prob(stochastic)

    assert deterministic.shape == (2, 3)
    assert stochastic.shape == (2, 3)
    assert log_prob.shape == (2,)
    assert torch.isfinite(deterministic).all()
    assert torch.isfinite(stochastic).all()
    assert model.last_finite_gate.all()
    assert model.output_mean.shape == (2, 3)
    assert model.output_std.shape == (2, 3)


def test_actor_model_uses_actor_only_path_and_freezes_legacy_value_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the RSL actor wrapper cannot enter the embedded value path."""
    obs = _observations()
    model = _model(obs)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Actor model entered embedded value path")

    monkeypatch.setattr(model.support_actor.critic_backbone, "forward", forbidden)
    monkeypatch.setattr(model.support_actor.value_head, "forward", forbidden)
    output = model(obs)
    output.square().mean().backward()

    assert all(
        not parameter.requires_grad
        for parameter in model.support_actor.critic_backbone.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.support_actor.value_head.parameters()
    )
    assert model.support_actor.action_head.weight.grad is not None


def test_actor_model_extracts_command_from_concatenated_proprioception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind command indices to raw proprio group order, not normalized input."""
    obs = _observations()
    obs["policy_a"][:, 1] = 0.4
    obs["policy_a"][:, 3] = -0.2
    obs["policy_b"][:, 1] = 0.3
    model = _model(obs)
    captured: list[torch.Tensor] = []
    original = model.support_pipeline.forward_native_training

    def capture(
        *args: torch.Tensor,
        **kwargs: object,
    ) -> CausalSphericalSupportEvidenceBatch:
        captured.append(args[-2].detach().clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(model.support_pipeline, "forward_native_training", capture)
    model(obs)

    assert len(captured) == 1
    assert torch.equal(
        captured[0],
        torch.tensor([[0.4, -0.2, 0.3], [0.4, -0.2, 0.3]]),
    )


def test_actor_model_poison_invalid_event_row_and_keeps_valid_row() -> None:
    """Fail closed per row when the observation channel contract is violated."""
    obs = _observations()
    obs["ray_event"][0, :, 1, 0, 0] = 0.5
    model = _model(obs).eval()

    output = model(obs)

    assert torch.isnan(output[0]).all()
    assert torch.isfinite(output[1]).all()
    assert torch.equal(model.last_finite_gate, torch.tensor([False, True]))


def test_actor_model_receipt_and_exports_remain_conservative() -> None:
    """Keep Lab, latency, training, and export claims false before closure."""
    model = _model(_observations())
    receipt = model.receipt()

    assert receipt["embedded_legacy_value_modules_frozen"] is True
    assert receipt["external_critic_configuration_unchanged"] is False
    assert receipt["registered_lab_task"] is False
    assert receipt["gpu_latency_measured"] is False
    assert receipt["training_ready"] is False
    with pytest.raises(RuntimeError, match="not yet authorized"):
        model.as_jit()
    with pytest.raises(RuntimeError, match="not yet authorized"):
        model.as_onnx(verbose=False)


def test_actor_model_rejects_structured_distribution_input() -> None:
    """Reject heteroscedastic output until the actor head supports shaped output."""
    with pytest.raises(TypeError, match="one integer input dimension"):
        _model(
            _observations(),
            distribution_cfg={
                "class_name": "HeteroscedasticGaussianDistribution",
                "init_std": 0.7,
                "std_type": "scalar",
            },
        )
