import ast
import inspect
import itertools
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any, NoReturn

import pytest

from rsl_rl.modules.support_selection_ablation import (
    FixedBudgetSupportSelector,
)


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


def _assert_exact_reference_parity(
    scores: torch.Tensor,
    roles: torch.Tensor,
    budget: int,
) -> None:
    tensor_result = FixedBudgetSupportSelector._role_quota_unique_tensor(scores, roles, budget)
    reference_result = FixedBudgetSupportSelector._role_quota_unique_reference(scores, roles, budget)
    for tensor_value, reference_value in zip(tensor_result, reference_result, strict=True):
        assert torch.equal(tensor_value, reference_value)


def _brute_force_maximum_cardinality(
    role_candidates: torch.Tensor,
    quota: int,
) -> int:
    num_roles, num_tokens = role_candidates.shape
    maximum = 0
    # -1 means unowned; 0..Q-1 are the only possible owners.
    for assignment in itertools.product(range(-1, num_roles), repeat=num_tokens):
        counts = [0] * num_roles
        feasible = True
        cardinality = 0
        for token, role in enumerate(assignment):
            if role < 0:
                continue
            if not role_candidates[role, token]:
                feasible = False
                break
            counts[role] += 1
            if counts[role] > quota:
                feasible = False
                break
            cardinality += 1
        if feasible:
            maximum = max(maximum, cardinality)
    return maximum


def test_tensor_matcher_has_no_host_scalar_or_batch_loop() -> None:
    """Keep the production matcher free of tensor-to-host synchronization."""
    source = inspect.getsource(FixedBudgetSupportSelector._role_quota_unique_tensor)
    tree = ast.parse(inspect.cleandoc(source))
    forbidden_names = {"bool", "int"}
    forbidden_attributes = {"cpu", "item", "numpy", "tolist"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_attributes
        if isinstance(node, (ast.For, ast.AsyncFor)):
            assert not (isinstance(node.target, ast.Name) and "batch" in node.target.id)


def test_batched_random_tensor_matcher_exactly_matches_cpu_reference() -> None:
    """Compare exact masks and stable indices over many batched random cases."""
    generator = torch.Generator().manual_seed(6201)
    scores = torch.randn(32, 4, 7, generator=generator)
    roles = torch.rand(32, 4, 7, generator=generator) < 0.48
    _assert_exact_reference_parity(scores, roles, budget=8)


def test_tensor_matcher_matches_reference_for_larger_quota() -> None:
    """Exercise repeated role slots rather than only the minimum quota."""
    generator = torch.Generator().manual_seed(6203)
    scores = torch.randn(3, 4, 12, generator=generator)
    roles = torch.rand(3, 4, 12, generator=generator) < 0.55
    _assert_exact_reference_parity(scores, roles, budget=16)


def test_adversarial_overlap_scarcity_and_ties_match_reference() -> None:
    """Cover augmenting chains, scarce roles, and stable equal-score ties."""
    scores = torch.zeros(3, 4, 8)
    roles = torch.zeros(3, 4, 8, dtype=torch.bool)

    # Dense total overlap forces repeated ownership reassignment.
    roles[0] = True
    # A Hall-style chain requires an augmenting path to preserve cardinality.
    roles[1, 0, [0, 1]] = True
    roles[1, 1, [0, 2]] = True
    roles[1, 2, [2, 3, 4]] = True
    roles[1, 3, [4, 5, 6, 7]] = True
    # One scarce role must shortfall while other roles cannot backfill it.
    roles[2, 0, 0] = True
    roles[2, 1, 1:4] = True
    roles[2, 2, 4:6] = True
    roles[2, 3, 6:8] = True

    _assert_exact_reference_parity(scores, roles, budget=8)
    first = FixedBudgetSupportSelector._role_quota_unique_tensor(scores, roles, 8)
    repeated = FixedBudgetSupportSelector._role_quota_unique_tensor(scores, roles, 8)
    for first_value, repeated_value in zip(first, repeated, strict=True):
        assert torch.equal(first_value, repeated_value)


def test_small_n_tensor_matcher_reaches_brute_force_maximum() -> None:
    """Prove maximum cardinality against exhaustive assignment enumeration."""
    generator = torch.Generator().manual_seed(6211)
    scores = torch.randn(6, 4, 6, generator=generator)
    roles = torch.rand(6, 4, 6, generator=generator) < 0.5

    _, _, query_mask, unique_mask = FixedBudgetSupportSelector._role_quota_unique_tensor(scores, roles, 8)

    for batch_index in range(scores.shape[0]):
        expected = _brute_force_maximum_cardinality(roles[batch_index], quota=2)
        assert unique_mask[batch_index].sum().item() == expected
        assert query_mask[batch_index].sum().item() == expected
        assert torch.all(query_mask[batch_index].sum(dim=-1) <= 2)
        assert not bool((query_mask[batch_index] & ~roles[batch_index]).any())


def test_forward_uses_tensor_matcher_and_marks_gpu_latency_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent accidental fallback to CPU reference or latency claims."""
    selector = FixedBudgetSupportSelector(strategy="role_quota_shared_unique_m", total_budget=8)
    scores = torch.randn(4, 4, 10)
    valid = torch.ones(4, 10, dtype=torch.bool)
    roles = torch.rand(4, 4, 10) < 0.5

    def forbidden_reference(
        *_args: object,
        **_kwargs: object,
    ) -> NoReturn:
        raise AssertionError("CPU reference was called by forward")

    monkeypatch.setattr(
        FixedBudgetSupportSelector,
        "_role_quota_unique_reference",
        staticmethod(forbidden_reference),
    )
    with _RejectLocalScalarMode():
        _, diagnostics = selector(scores, valid, roles)

    assert diagnostics["gpu_latency_unmeasured"].all()
    assert not diagnostics["matcher_performance_claimed"].any()
    assert diagnostics["valid_scores_finite"].all()
