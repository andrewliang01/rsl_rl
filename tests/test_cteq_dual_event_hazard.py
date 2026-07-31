from __future__ import annotations

import numpy as np
import pytest
import torch

from rsl_rl.modules.cteq_dual_event_hazard import (
    CteqDualEventHazardHead,
    CteqIndependentSurvivalLoss,
    cteq_distribution_from_logits,
    cteq_role_time_weights_from_logits,
)
from rsl_rl.utils.cteq_contact_timing import (
    CTEQ_BIN_WIDTH_S,
    build_independent_event_labels,
    debounce_contact_trace,
    dual_event_hazard_from_logits,
    dual_event_survival_loss,
)


def _labels():
    contact = np.zeros((40, 2), dtype=np.bool_)
    contact[5:10, 0] = True
    contact[7:14, 1] = True
    trace = debounce_contact_trace(
        contact,
        sample_period_s=CTEQ_BIN_WIDTH_S,
        min_stable_steps=1,
    )
    return build_independent_event_labels(trace, anchor_indices=[0])


def test_torch_distribution_matches_numpy_and_sums_to_one():
    torch.manual_seed(701)
    logits = torch.randn(3, 2, 2, 25, dtype=torch.float64)
    actual = cteq_distribution_from_logits(logits)
    expected = dual_event_hazard_from_logits(logits.numpy())
    torch.testing.assert_close(
        actual.event_mass,
        torch.from_numpy(expected.event_mass.copy()),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        actual.censor_mass,
        torch.from_numpy(expected.censor_mass.copy()),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        actual.event_mass.sum(-1) + actual.censor_mass,
        torch.ones_like(actual.censor_mass),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_torch_loss_matches_numpy_for_observed_and_censored_targets():
    labels = _labels()
    torch.manual_seed(702)
    logits = torch.randn(1, 2, 2, 25, dtype=torch.float64)
    expected = dual_event_survival_loss(
        dual_event_hazard_from_logits(logits.numpy()),
        labels,
    )
    actual = CteqIndependentSurvivalLoss()(
        logits,
        torch.from_numpy(labels.event_bin.copy()),
        torch.from_numpy(labels.event_observed.copy()),
        torch.from_numpy(labels.right_censored.copy()),
    )
    assert actual.mean_nll.item() == pytest.approx(expected.mean_nll, abs=1.0e-12)
    assert actual.mean_brier.item() == pytest.approx(expected.mean_brier, abs=1.0e-12)
    np.testing.assert_allclose(
        actual.per_target_nll.numpy(), expected.per_target_nll, atol=1.0e-12
    )


def test_head_and_survival_loss_have_finite_nonzero_gradients():
    labels = _labels()
    torch.manual_seed(703)
    head = CteqDualEventHazardHead(64, hidden_dims=(48,))
    features = torch.randn(1, 64, requires_grad=True)
    logits = head(features)
    loss = CteqIndependentSurvivalLoss()(
        logits,
        torch.from_numpy(labels.event_bin.copy()),
        torch.from_numpy(labels.event_observed.copy()),
        torch.from_numpy(labels.right_censored.copy()),
    )
    (loss.mean_nll + loss.mean_brier).backward()
    assert logits.shape == (1, 2, 2, 25)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_role_weights_use_current_contact_and_independent_event_masses():
    logits = torch.zeros(2, 2, 2, 25)
    contact = torch.tensor([[True, False], [False, True]])
    current, landing = cteq_role_time_weights_from_logits(logits, contact)
    assert current.shape == landing.shape == (2, 2, 25)
    assert torch.equal(current[0, 1], torch.zeros(25))
    assert torch.equal(current[1, 0], torch.zeros(25))
    assert torch.count_nonzero(current[0, 0]) == 25
    assert torch.count_nonzero(current[1, 1]) == 25
    torch.testing.assert_close(
        landing,
        cteq_distribution_from_logits(logits).event_mass[:, :, 0],
    )


def test_future_labels_are_loss_only_and_malformed_censoring_fails_closed():
    labels = _labels()
    logits = torch.zeros(1, 2, 2, 25)
    bins = torch.from_numpy(labels.event_bin.copy())
    observed = torch.from_numpy(labels.event_observed.copy())
    censored = torch.from_numpy(labels.right_censored.copy())
    bad = censored.clone()
    bad[0, 0, 0] = observed[0, 0, 0]
    with pytest.raises(ValueError, match="complement"):
        CteqIndependentSurvivalLoss()(logits, bins, observed, bad)

    assert all(
        "label" not in name and "truth" not in name
        for name, _ in CteqDualEventHazardHead(16).named_parameters()
    )


def test_hazard_head_torchscript_and_onnx_use_dynamic_batch(tmp_path):
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(704)
    head = CteqDualEventHazardHead(32, hidden_dims=(24,)).eval()
    scripted = torch.jit.script(head)
    features = torch.randn(3, 32)
    with torch.no_grad():
        torch.testing.assert_close(scripted(features), head(features))

    path = tmp_path / "cteq_hazard.onnx"
    torch.onnx.export(
        head,
        (torch.zeros(1, 32),),
        path,
        opset_version=18,
        input_names=["causal_features"],
        output_names=["dual_event_logits"],
        dynamic_axes={
            "causal_features": {0: "batch_size"},
            "dual_event_logits": {0: "batch_size"},
        },
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
