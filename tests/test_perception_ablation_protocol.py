import pytest

from rsl_rl.utils.perception_ablation_protocol import (
    H1AblationSpec,
    H2AblationSpec,
    build_h1_protocol,
    build_h2_protocol,
    perception_ablation_receipt,
)


def test_h1_protocol_has_full_and_all_budgeted_selected_only_arms():
    specs = build_h1_protocol(random_seed=19)
    assert len(specs) == 13
    assert "h1_full" in specs
    for budget in (8, 16, 32, 64):
        for label in ("glad", "role", "random"):
            name = f"h1_{label}_m{budget:02d}_selected_only"
            assert name in specs
            assert not specs[name].training_ready
            assert any(
                f"support_total_budget={budget}" in value
                for value in specs[name].reserved_overrides
            )


def test_h2_protocol_has_causal_2x2_exact_union_and_age_controls():
    specs = build_h2_protocol(shuffle_seed=23)
    assert len(specs) == 8
    for geometry in ("native", "rerender"):
        for association in ("correct", "shuffled"):
            name = f"h2_{geometry}_{association}_history_per_return_age"
            assert name in specs
    assert "h2_native_correct_exact_union_k1_age_zero" in specs
    assert (
        "h2_native_correct_raster_latest_event_prototype_per_return_age"
        in specs
    )
    assert "h2_native_correct_history_packet_age" in specs
    assert "h2_native_correct_history_age_zero" in specs


def test_protocol_receipt_does_not_claim_registered_training_tasks():
    receipt = perception_ablation_receipt()
    assert receipt["training_task_registration"] == "intentionally_deferred"
    assert receipt["schema"] == "g1_perception_ablation_protocol_v3"
    assert receipt["h1_role_semantics"] == "cteq_current_landing_v2"
    assert receipt["incompatible_role_migration"]["automatic_aliasing"] is False
    assert "causal/action_kl_clean_vs_intervention" in receipt["metric_keys"]["paired_causal"]
    assert all(not row["training_ready"] for row in receipt["h1"].values())
    assert all(not row["training_ready"] for row in receipt["h2"].values())


def test_specs_reject_false_training_readiness_and_invalid_randomness():
    with pytest.raises(ValueError, match="cannot be marked"):
        H1AblationSpec(
            name="bad",
            selector="full",
            total_budget="all",
            value_intervention="full",
            random_seed=None,
            training_ready=True,
        )
    with pytest.raises(ValueError, match="requires"):
        H2AblationSpec(
            name="bad",
            geometry="native",
            time_association="shuffled",
        )
    with pytest.raises(ValueError, match="must use age_zero"):
        H2AblationSpec(
            name="bad_exact_union_with_time",
            geometry="native",
            time_association="correct",
            temporal_baseline="per_return_age",
            history_reduction="exact_union_k1",
        )
