import torch

import pytest

from rsl_rl.modules import (
    MatchedSubstitutionMetadata,
    MatchedSubstitutionShortfallError,
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


def _actor_inputs(
    batch_size: int = 2,
    num_tokens: int = 24,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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


def _matched_metadata(
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    candidate_mask: torch.Tensor | None = None,
) -> MatchedSubstitutionMetadata:
    token_valid = inputs[3]
    roles = inputs[4]
    batch_size, num_tokens = token_valid.shape
    if candidate_mask is None:
        candidate_mask = token_valid.clone()
    # Disjoint six-cell role groups use exact, nontrivial discrete strata.
    role_stratum = (
        torch.arange(num_tokens, dtype=torch.long) // 6
    )[None, :].expand(batch_size, -1).clone()
    priority = torch.arange(
        num_tokens, dtype=torch.long
    )[None, :].expand(batch_size, -1).clone()
    return MatchedSubstitutionMetadata.register(
        candidate_mask=candidate_mask,
        role_eligibility=roles,
        range_stratum=role_stratum,
        angle_stratum=role_stratum + 10,
        age_stratum=role_stratum + 20,
        candidate_priority=priority,
    )


def test_role_quota_selector_spends_unique_m_with_overlap() -> None:
    """Charge an overlapping token once while exposing both eligible roles."""
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
    """Leave a scarce role slot empty instead of spending it elsewhere."""
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
    """Project exactly M values and share an eligible overlapping token."""
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
    """Prove that unselected terrain values have no action/value bypass."""
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
    """Expose actor-level per-role shortfall without cross-role replacement."""
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
    """Retain finite gradients through selected values and their scores."""
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


def test_explicit_native_is_bitwise_identical_to_legacy_default() -> None:
    """Keep the explicit native intervention bitwise backward compatible."""
    torch.manual_seed(5127)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs()

    with torch.inference_mode():
        default_action, default_value, default_diagnostics = (
            model.forward_with_diagnostics(
                *inputs, mask_provenance=_provenance()
            )
        )
        native_action, native_value, native_diagnostics = (
            model.forward_with_diagnostics(
                *inputs,
                mask_provenance=_provenance(),
                intervention_mode="native",
            )
        )

    assert torch.equal(native_action, default_action)
    assert torch.equal(native_value, default_value)
    assert torch.equal(
        native_diagnostics["selection_indices"],
        default_diagnostics["selection_indices"],
    )
    assert (
        native_diagnostics["clean_membership_sha256"]
        == default_diagnostics["clean_membership_sha256"]
    )
    assert native_diagnostics["clean_selection_frozen"].all()
    assert native_diagnostics["no_reselection"].all()
    assert native_diagnostics["selection_recomputed_count"].count_nonzero() == 0


def test_all_interventions_preserve_clean_membership_bitwise() -> None:
    """Freeze identical clean membership for every causal intervention."""
    torch.manual_seed(5131)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs()
    batch_size = inputs[0].shape[0]
    role_permutation = torch.tensor([[1, 0, 3, 2]]).expand(
        batch_size, -1
    ).clone()
    cell_permutation = torch.arange(7, -1, -1).long()[None, :].expand(
        batch_size, -1
    ).clone()
    metadata = _matched_metadata(inputs)

    with torch.inference_mode():
        _, _, native = model.forward_with_diagnostics(
            *inputs, mask_provenance=_provenance()
        )
        variants = []
        for mode, kwargs in (
            ("zero_selected", {}),
            ("matched_substitution", {"matched_substitution": metadata}),
            ("role_shuffle", {"role_permutation": role_permutation}),
            ("cell_shuffle", {"cell_permutation": cell_permutation}),
        ):
            _, _, diagnostics = model.forward_with_diagnostics(
                *inputs,
                mask_provenance=_provenance(),
                intervention_mode=mode,
                **kwargs,
            )
            variants.append(diagnostics)

    for diagnostics in variants:
        assert torch.equal(
            diagnostics["selection_indices"], native["selection_indices"]
        )
        assert torch.equal(
            diagnostics["selection_unique_mask"],
            native["selection_unique_mask"],
        )
        assert torch.equal(
            diagnostics["effective_selection_indices"],
            native["selection_indices"],
        )
        assert (
            diagnostics["clean_membership_sha256"]
            == native["clean_membership_sha256"]
        )
        assert diagnostics["clean_selection_frozen"].all()
        assert diagnostics["no_reselection"].all()
        assert diagnostics["selection_recomputed_count"].count_nonzero() == 0
        assert not diagnostics["fair_performance_claim"].any()
        assert not diagnostics["real_geometry_connected"].any()


def test_exact_matched_substitution_uses_registered_role_and_strata() -> None:
    """Match substitutes on role signature plus range, angle, and age."""
    torch.manual_seed(5137)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs()
    metadata = _matched_metadata(inputs)
    projected_inputs: list[torch.Tensor] = []

    def capture_substitutes(
        _module: torch.nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        projected_inputs.append(arguments[0].detach().clone())

    hook = model.selected_value_projection.register_forward_pre_hook(
        capture_substitutes
    )
    try:
        with torch.inference_mode():
            _, _, diagnostics = model.forward_with_diagnostics(
                *inputs,
                mask_provenance=_provenance(),
                intervention_mode="matched_substitution",
                matched_substitution=metadata,
            )
    finally:
        hook.remove()

    assert diagnostics["matched_substitution_slot_valid"].all()
    assert torch.equal(
        diagnostics["matched_substitution_target_count"],
        torch.full((2,), 8),
    )
    assert torch.equal(
        diagnostics["matched_substitution_realized_count"],
        torch.full((2,), 8),
    )
    assert diagnostics["matched_substitution_shortfall"].count_nonzero() == 0
    assert diagnostics["intervention_applicable"].all()
    assert not diagnostics["matched_substitution_global_random_fallback"].any()
    assert (
        diagnostics["matched_substitution_registration_sha256"]
        == metadata.registration_sha256
    )
    selected = diagnostics["selection_indices"]
    substitutes = diagnostics["matched_substitution_indices"]
    expected_values = torch.gather(
        inputs[1],
        1,
        substitutes[:, :, None].expand(-1, -1, VALUE_DIM),
    )
    assert torch.equal(projected_inputs[0], expected_values)
    for batch_index in range(selected.shape[0]):
        assert substitutes[batch_index].unique().numel() == 8
        assert not bool(
            diagnostics["selection_unique_mask"][
                batch_index, substitutes[batch_index]
            ].any()
        )
        for slot in range(selected.shape[1]):
            source = selected[batch_index, slot]
            substitute = substitutes[batch_index, slot]
            assert torch.equal(
                inputs[4][batch_index, :, source],
                inputs[4][batch_index, :, substitute],
            )
            for stratum in (
                metadata.range_stratum,
                metadata.angle_stratum,
                metadata.age_stratum,
            ):
                assert stratum[batch_index, source] == stratum[
                    batch_index, substitute
                ]


def test_zero_selected_is_exactly_the_zero_value_counterfactual() -> None:
    """Zero selected values without changing selection or score weights."""
    torch.manual_seed(5139)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = list(_actor_inputs())

    with torch.inference_mode():
        zero_action, zero_value, zero_diagnostics = (
            model.forward_with_diagnostics(
                *inputs,
                mask_provenance=_provenance(),
                intervention_mode="zero_selected",
            )
        )
        globally_zeroed = inputs.copy()
        globally_zeroed[1] = torch.zeros_like(inputs[1])
        expected_action, expected_value, expected_diagnostics = (
            model.forward_with_diagnostics(
                *globally_zeroed,
                mask_provenance=_provenance(),
            )
        )

    assert torch.equal(zero_action, expected_action)
    assert torch.equal(zero_value, expected_value)
    assert torch.equal(
        zero_diagnostics["selection_indices"],
        expected_diagnostics["selection_indices"],
    )
    assert zero_diagnostics["intervention_applicable"].all()


def test_matched_substitution_shortfall_fails_with_frozen_audit() -> None:
    """Fail closed with membership evidence when exact substitutes run out."""
    torch.manual_seed(5141)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs(batch_size=1)
    metadata = _matched_metadata(
        inputs,
        candidate_mask=torch.zeros_like(inputs[3]),
    )

    with pytest.raises(MatchedSubstitutionShortfallError) as caught:
        model.forward_with_diagnostics(
            *inputs,
            mask_provenance=_provenance(),
            intervention_mode="matched_substitution",
            matched_substitution=metadata,
        )

    audit = caught.value.audit
    assert torch.equal(
        audit["matched_substitution_target_count"], torch.tensor([8])
    )
    assert torch.equal(
        audit["matched_substitution_realized_count"], torch.tensor([0])
    )
    assert torch.equal(
        audit["matched_substitution_shortfall"], torch.tensor([8])
    )
    assert audit["clean_selection_frozen"].all()
    assert audit["no_reselection"].all()
    assert audit["matched_substitution_global_random_fallback"] is False
    assert len(audit["clean_membership_sha256"][0]) == 64


def test_role_shuffle_only_reorders_clean_consumption_relationship() -> None:
    """Shuffle role consumption while keeping token identities and values."""
    torch.manual_seed(5147)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs(batch_size=1)
    permutation = torch.tensor([[1, 0, 3, 2]], dtype=torch.long)

    with torch.inference_mode():
        _, _, native = model.forward_with_diagnostics(
            *inputs, mask_provenance=_provenance()
        )
        _, _, shuffled = model.forward_with_diagnostics(
            *inputs,
            mask_provenance=_provenance(),
            intervention_mode="role_shuffle",
            role_permutation=permutation,
        )

    expected = torch.gather(
        native["clean_role_consumption_mask"],
        1,
        permutation[:, :, None].expand(-1, -1, 8),
    )
    assert torch.equal(shuffled["role_consumption_mask"], expected)
    assert torch.equal(
        shuffled["clean_role_consumption_mask"],
        native["clean_role_consumption_mask"],
    )
    assert torch.equal(
        shuffled["selection_indices"], native["selection_indices"]
    )
    assert shuffled["role_shuffle_permutation_sha256"] is not None


def test_cell_shuffle_only_reorders_values_across_selected_cells() -> None:
    """Shuffle selected values while preserving frozen cell identities."""
    torch.manual_seed(5153)
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs(batch_size=1)
    permutation = torch.arange(7, -1, -1).long()[None, :]
    projected_inputs: list[torch.Tensor] = []

    def capture_values(
        _module: torch.nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        projected_inputs.append(arguments[0].detach().clone())

    hook = model.selected_value_projection.register_forward_pre_hook(
        capture_values
    )
    try:
        with torch.inference_mode():
            _, _, native = model.forward_with_diagnostics(
                *inputs, mask_provenance=_provenance()
            )
            _, _, shuffled = model.forward_with_diagnostics(
                *inputs,
                mask_provenance=_provenance(),
                intervention_mode="cell_shuffle",
                cell_permutation=permutation,
            )
    finally:
        hook.remove()

    expected_values = torch.gather(
        projected_inputs[0],
        1,
        permutation[:, :, None].expand(-1, -1, VALUE_DIM),
    )
    assert torch.equal(projected_inputs[1], expected_values)
    assert torch.equal(
        shuffled["role_consumption_mask"], native["role_consumption_mask"]
    )
    assert torch.equal(
        shuffled["selection_indices"], native["selection_indices"]
    )
    assert shuffled["cell_shuffle_permutation_sha256"] is not None


def test_intervention_metadata_contract_is_fail_closed() -> None:
    """Reject missing, cross-mode, mutated, or invalid intervention metadata."""
    model = SharedUniqueSupportActorAdapter(
        SCORE_DIM, VALUE_DIM, PROPRIO_DIM, ACTION_DIM, total_budget=8
    ).eval()
    inputs = _actor_inputs(batch_size=1)
    with pytest.raises(ValueError, match="requires explicit"):
        model.forward_with_diagnostics(
            *inputs,
            mask_provenance=_provenance(),
            intervention_mode="matched_substitution",
        )

    metadata = _matched_metadata(inputs)
    metadata.candidate_priority[0, 0] += 100
    with pytest.raises(ValueError, match="changed after registration"):
        model.forward_with_diagnostics(
            *inputs,
            mask_provenance=_provenance(),
            intervention_mode="matched_substitution",
            matched_substitution=metadata,
        )

    with pytest.raises(ValueError, match=r"permute 0\.\.3"):
        model.forward_with_diagnostics(
            *inputs,
            mask_provenance=_provenance(),
            intervention_mode="role_shuffle",
            role_permutation=torch.tensor([[0, 0, 1, 2]]),
        )


def test_provenance_is_mandatory_and_rejects_simulator_truth() -> None:
    """Reject undeclared, privileged, or simulator-truth role geometry."""
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
