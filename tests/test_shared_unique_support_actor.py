import pytest
import torch

from rsl_rl.modules import (
    SharedUniqueSupportActorAdapter,
    SupportMaskProvenance,
)
from rsl_rl.modules.support_selection_ablation import (
    FixedBudgetSupportSelector,
)


SCORE_DIM = 5
VALUE_DIM = 3
PROPRIO_DIM = 7
ACTION_DIM = 4


def _provenance() -> SupportMaskProvenance:
    return SupportMaskProvenance(
        geometry_source="calibrated_lidar_ray_geometry",
        uses_proprioception=True,
        uses_gait_phase=True,
    )


def _actor_inputs(batch_size: int = 2, num_tokens: int = 24):
    generator = torch.Generator().manual_seed(5101)
    score_features = torch.randn(
        batch_size, num_tokens, SCORE_DIM, generator=generator
    )
    terrain_values = torch.randn(
        batch_size, num_tokens, VALUE_DIM, generator=generator
    )
    proprio = torch.randn(
        batch_size, PROPRIO_DIM, generator=generator
    )
    token_valid = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    roles = torch.zeros(batch_size, 4, num_tokens, dtype=torch.bool)
    roles[:, 0, 0:6] = True
    roles[:, 1, 6:12] = True
    roles[:, 2, 12:18] = True
    roles[:, 3, 18:24] = True
    return score_features, terrain_values, proprio, token_valid, roles


def test_role_quota_selector_spends_unique_m_with_overlap() -> None:
    scores = torch.tensor(
        [
            [
                [9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [9.0, 0.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 8.0],
            ]
        ]
    )
    valid = torch.ones(1, 8, dtype=torch.bool)
    roles = torch.zeros(1, 4, 8, dtype=torch.bool)
    roles[:, 0, [0, 1]] = True
    roles[:, 1, [0, 2, 3]] = True
    roles[:, 2, [4, 5]] = True
    roles[:, 3, [6, 7]] = True
    selector = FixedBudgetSupportSelector(
        strategy="role_quota_shared_unique_m", total_budget=8
    )

    owner_mask, diagnostics = selector(scores, valid, roles)

    assert owner_mask.sum().item() == 8
    assert owner_mask.any(dim=1).sum().item() == 8
    assert torch.equal(
        diagnostics["realized_per_role"], torch.full((1, 4), 2)
    )
    assert diagnostics["unique_budget_shortfall"].item() == 0
    assert diagnostics["per_role_shortfall"].count_nonzero() == 0
    # Token zero costs one owner slot but is available to both eligible roles.
    assert diagnostics["selection_unique_mask"][0, 0]
    assert diagnostics["selected_eligible_role_mask"][0, 0, 0]
    assert diagnostics["selected_eligible_role_mask"][0, 1, 0]


def test_role_shortfall_is_not_backfilled_by_other_roles() -> None:
    scores = torch.randn(1, 4, 16)
    valid = torch.ones(1, 16, dtype=torch.bool)
    roles = torch.zeros(1, 4, 16, dtype=torch.bool)
    roles[:, 0, 0] = True
    roles[:, 1, 1:6] = True
    roles[:, 2, 6:11] = True
    roles[:, 3, 11:16] = True
    selector = FixedBudgetSupportSelector(
        strategy="role_quota_shared_unique_m", total_budget=8
    )

    owner_mask, diagnostics = selector(scores, valid, roles)

    assert owner_mask.sum().item() == 7
    assert torch.equal(
        diagnostics["realized_per_role"], torch.tensor([[1, 2, 2, 2]])
    )
    assert torch.equal(
        diagnostics["per_role_shortfall"], torch.tensor([[1, 0, 0, 0]])
    )
    assert diagnostics["unique_budget_shortfall"].item() == 1
    assert not bool((owner_mask & ~roles).any())


def test_actor_projects_only_selected_width_and_shares_overlap() -> None:
    torch.manual_seed(5107)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM,
        VALUE_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        total_budget=8,
    ).eval()
    inputs = list(_actor_inputs(batch_size=1, num_tokens=24))
    # Eight feasible unique candidates with token zero shared by two roles.
    inputs[4].zero_()
    inputs[4][0, 0, [0, 1]] = True
    inputs[4][0, 1, [0, 2, 3]] = True
    inputs[4][0, 2, [4, 5]] = True
    inputs[4][0, 3, [6, 7]] = True
    seen_shapes: list[tuple[int, ...]] = []

    def record_projection_shape(
        _module: torch.nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        seen_shapes.append(tuple(arguments[0].shape))

    hook = model.selected_value_projection.register_forward_pre_hook(
        record_projection_shape
    )
    try:
        with torch.inference_mode():
            action, value, diagnostics = model.forward_with_diagnostics(
                *inputs, mask_provenance=_provenance()
            )
    finally:
        hook.remove()

    assert action.shape == (1, ACTION_DIM)
    assert value.shape == (1, 1)
    assert seen_shapes == [(1, 8, VALUE_DIM)]
    assert diagnostics["realized_unique_count"].item() == 8
    assert diagnostics["projected_selected_token_count"].item() == 8
    assert diagnostics["post_bottleneck_token_width"].item() == 8
    assert diagnostics["selection_unique_mask"][0, 0]
    selected_positions = diagnostics["selection_indices"][0]
    position = torch.where(selected_positions == 0)[0].item()
    assert diagnostics["role_consumption_mask"][0, 0, position]
    assert diagnostics["role_consumption_mask"][0, 1, position]


def test_actor_unselected_value_perturbation_changes_neither_action_nor_value() -> None:
    torch.manual_seed(5113)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM,
        VALUE_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        total_budget=8,
    ).eval()
    inputs = list(_actor_inputs())

    with torch.inference_mode():
        action, value, diagnostics = model.forward_with_diagnostics(
            *inputs, mask_provenance=_provenance()
        )
        changed_values = inputs[1].clone()
        unselected = ~diagnostics["selection_unique_mask"]
        changed_values[unselected] += 1.0e6
        changed_action, changed_value = model(
            inputs[0],
            changed_values,
            inputs[2],
            inputs[3],
            inputs[4],
            mask_provenance=_provenance(),
        )

    assert torch.equal(changed_action, action)
    assert torch.equal(changed_value, value)


def test_actor_reports_role_shortfall_without_cross_role_backfill() -> None:
    torch.manual_seed(5119)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM,
        VALUE_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        total_budget=8,
    ).eval()
    inputs = list(_actor_inputs(batch_size=1, num_tokens=24))
    inputs[4][0, 0].zero_()
    inputs[4][0, 0, 0] = True

    with torch.inference_mode():
        _, _, diagnostics = model.forward_with_diagnostics(
            *inputs, mask_provenance=_provenance()
        )

    assert diagnostics["realized_unique_count"].item() == 7
    assert torch.equal(
        diagnostics["realized_per_role"], torch.tensor([[1, 2, 2, 2]])
    )
    assert torch.equal(
        diagnostics["per_role_shortfall"], torch.tensor([[1, 0, 0, 0]])
    )


def test_actor_and_value_paths_are_trainable_through_selected_evidence() -> None:
    torch.manual_seed(5123)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM,
        VALUE_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        total_budget=8,
    )
    inputs = _actor_inputs()

    action, value = model(*inputs, mask_provenance=_provenance())
    (action.square().mean() + value.square().mean()).backward()

    for name in (
        "score_key_projection.weight",
        "score_query_projection.weight",
        "selected_value_projection.weight",
        "action_head.weight",
        "value_head.weight",
    ):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_provenance_is_mandatory_and_rejects_simulator_truth() -> None:
    with pytest.raises(ValueError, match="contact truth"):
        SupportMaskProvenance(
            geometry_source="calibrated_lidar_ray_geometry",
            uses_proprioception=True,
            uses_gait_phase=True,
            uses_simulator_contact_truth=True,
        )
    with pytest.raises(ValueError, match="terrain truth"):
        SupportMaskProvenance(
            geometry_source="calibrated_lidar_ray_geometry",
            uses_proprioception=True,
            uses_gait_phase=True,
            uses_simulator_terrain_truth=True,
        )
    with pytest.raises(ValueError, match="deployment-observable"):
        SupportMaskProvenance(
            geometry_source="simulator_height_field",
            uses_proprioception=True,
            uses_gait_phase=True,
        )

    receipt = _provenance().receipt()
    assert receipt["deployment_observable_only"] is True
    assert receipt["uses_simulator_contact_truth"] is False
    assert receipt["uses_simulator_terrain_truth"] is False

    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    )
    with pytest.raises(ValueError, match="mask_provenance"):
        model.forward_with_diagnostics(
            *_actor_inputs(batch_size=1), mask_provenance=None
        )
