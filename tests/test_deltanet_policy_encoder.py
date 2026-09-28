"""Detached policy inputs, privileged-data isolation and deployment contracts."""

import numpy as np
import torch
from collections.abc import Generator
from pathlib import Path
from tensordict import TensorDict

import pytest
from test_mid360_deltanet import actor, observations

from rsl_rl.models import MID360DeltaNetActor


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    """Keep small CPU comparisons deterministic and inexpensive."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(before)


def encoded_actor(obs: TensorDict) -> MID360DeltaNetActor:
    """Construct the policy-side encoder variant of the small DeltaNet actor."""
    return actor(obs, layers=2, use_actor_input_encoders=True)


def test_policy_encoders_receive_gradients_but_inputs_and_estimator_do_not() -> None:
    """Policy losses train both encoders while stopping at their input values."""
    model = encoded_actor(observations(2))
    assert model._get_latent_dim() == 128
    assert model.mlp[0].in_features == 128
    inputs = torch.randn(3, 2, model.policy_input_dim, requires_grad=True)
    features = model.encode_policy_input(inputs)
    assert features.shape == (3, 2, 128)
    features.square().mean().backward()
    assert inputs.grad is None
    for branch in (model.policy_encoder.map_conv, model.policy_encoder.map_projection, model.policy_encoder.state_mlp):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    model.zero_grad(set_to_none=True)
    model(observations(2)).square().mean().backward()
    assert all(p.grad is None for p in model.estimator.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy_encoder.parameters())
    assert model.last_map.requires_grad  # Supervised training can still use the prediction graph.


def test_policy_encoders_are_independent_of_minibatch_composition_and_privileged_targets() -> None:
    """Shuffling and privileged labels cannot alter behavior-policy replay."""
    obs = observations(4)
    model = encoded_actor(obs)
    inputs = torch.randn(4, model.policy_input_dim)
    expected = model.encode_policy_input(inputs)
    ids = torch.tensor([3, 0])
    torch.testing.assert_close(model.encode_policy_input(inputs[ids]), expected[ids])
    action = model(obs)
    changed = obs.clone()
    for key in ("critic", "height_scan_critic", "deltanet_velocity_target"):
        changed[key].fill_(float("nan"))
    model.reset()
    torch.testing.assert_close(model(changed), action)


def test_encoded_actor_explicit_state_and_traced_deployment_match_across_resets(tmp_path: Path) -> None:
    """Exported actions and recurrent state match online control across resets."""
    obs = observations(2)
    model = encoded_actor(obs).eval()
    core = model.as_jit().eval()
    state = model.estimator.initial_state(2, obs["policy"])
    example = (obs["policy"], obs["height_scan_policy"], state)
    traced = torch.jit.trace(core, example, check_trace=False)
    path = tmp_path / "policy.pt"
    traced.save(str(path))
    traced = torch.jit.load(str(path))
    for step in range(3):
        expected = model(obs)
        actual = traced(obs["policy"], obs["height_scan_policy"], state)
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], model.last_map)
        torch.testing.assert_close(actual[2], model.last_velocity)
        torch.testing.assert_close(actual[3], model.get_hidden_state())
        state = actual[3]
        if step == 1:
            dones = torch.tensor([True, False])
            model.reset(dones)
            state = state * (~dones)[None, :, None]
        obs = observations(2)


def test_encoded_actor_onnx_dynamic_batch_matches(tmp_path: Path) -> None:
    """An ONNX export made for batch one supports a different batch size."""
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    obs = observations(2)
    model = encoded_actor(obs).eval()
    exported = model.as_onnx(verbose=False).eval()
    path = tmp_path / "policy.onnx"
    torch.onnx.export(
        exported,
        exported.get_dummy_inputs(),
        path,
        opset_version=18,
        input_names=exported.input_names,
        output_names=exported.output_names,
        dynamic_axes=exported.dynamic_axes,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    state = model.estimator.initial_state(2, obs["policy"])
    feeds = {"proprio": obs["policy"].numpy(), "panorama": obs["height_scan_policy"].numpy(), "state": state.numpy()}
    actual = ReferenceEvaluator(graph).run(None, feeds)
    expected = exported(obs["policy"], obs["height_scan_policy"], state)
    for value, target in zip(actual, expected):
        np.testing.assert_allclose(value, target.detach().numpy(), rtol=1e-4, atol=1e-5)
