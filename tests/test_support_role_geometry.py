import torch

import pytest

from rsl_rl.modules import (
    CalibratedSphericalSupportRoleGeometry,
    SharedUniqueSupportActorAdapter,
)


def _geometry() -> CalibratedSphericalSupportRoleGeometry:
    rays = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ]
    )
    return CalibratedSphericalSupportRoleGeometry(
        rays,
        torch.eye(3),
        torch.zeros(3),
        external_calibration_sha256="a" * 64,
        current_radius=0.08,
        landing_radius=0.08,
        vertical_half_extent=0.05,
        min_range=0.1,
        max_range=5.0,
        range_strata_edges=(0.5, 1.5, 3.0),
        age_strata_edges=(0.01, 0.05, 0.2),
        azimuth_strata=8,
        elevation_strata=4,
    )


def _centres() -> tuple[torch.Tensor, torch.Tensor]:
    current = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    landing = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
    return current, landing


def _transforms(
    history_length: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(
        1, history_length, -1, -1
    ).clone()
    translation = torch.zeros(1, history_length, 3)
    return rotation, translation


def test_four_physical_support_roles_and_actor_wiring() -> None:
    """Map four calibrated rays to four physical roles and run the H1 actor."""
    geometry = _geometry()
    current, landing = _centres()
    rotation, translation = _transforms()
    batch = geometry(
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4, dtype=torch.bool),
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1),
        rotation,
        translation,
        current,
        landing,
    )

    assert batch.score_features.shape == (1, 4, 9)
    assert batch.terrain_values.shape == (1, 4, 5)
    assert torch.equal(batch.role_eligibility[0], torch.eye(4, dtype=torch.bool))
    assert batch.candidate_mask.all()
    assert torch.equal(batch.body_points[0], torch.cat((current[0], landing[0]))[[0, 2, 1, 3]])

    actor = SharedUniqueSupportActorAdapter(
        score_feature_dim=geometry.score_feature_dim,
        terrain_value_dim=geometry.terrain_value_dim,
        proprio_dim=6,
        action_dim=3,
        total_budget=8,
    ).eval()
    with torch.inference_mode():
        action, value, diagnostics = actor.forward_with_diagnostics(
            batch.score_features,
            batch.terrain_values,
            torch.zeros(1, 6),
            batch.token_valid,
            batch.role_eligibility,
            mask_provenance=geometry.provenance(),
        )
    assert action.shape == (1, 3)
    assert value.shape == (1, 1)
    assert diagnostics["realized_unique_count"].item() == 4


def test_invalid_return_is_unknown_and_cannot_enter_support_roles() -> None:
    """Keep invalid, NaN, too-near, and too-far cells as masked unknowns."""
    geometry = _geometry()
    current, landing = _centres()
    rotation, translation = _transforms()
    ranges = torch.tensor([[[[1.0, float("nan"), 0.0, 6.0]]]])
    declared_valid = torch.ones_like(ranges, dtype=torch.bool)
    batch = geometry(
        ranges,
        declared_valid,
        torch.zeros_like(ranges),
        torch.zeros(1, 1),
        rotation,
        translation,
        current,
        landing,
    )

    assert torch.equal(
        batch.token_valid, torch.tensor([[True, False, False, False]])
    )
    assert torch.equal(
        batch.candidate_mask, torch.tensor([[True, False, False, False]])
    )
    assert batch.score_features[0, 1:].count_nonzero() == 0
    assert batch.terrain_values[0, 1:].count_nonzero() == 0
    assert torch.equal(batch.range_stratum[0, 1:], torch.full((3,), -1))
    assert not batch.role_eligibility[:, :, 1:].any()


def test_history_registration_has_exact_strata_and_stable_cell_ids() -> None:
    """Expose deterministic temporal IDs and hash exact matching metadata."""
    geometry = _geometry()
    current, landing = _centres()
    rotation, translation = _transforms(history_length=2)
    ranges = torch.ones(1, 2, 1, 4)
    return_age = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0]], [[0.1, 0.1, 0.1, 0.1]]]]
    )
    batch = geometry(
        ranges,
        torch.ones_like(ranges, dtype=torch.bool),
        return_age,
        torch.tensor([[0.0, 0.1]]),
        rotation,
        translation,
        current,
        landing,
    )
    metadata = batch.register_matched_substitution()

    assert torch.equal(batch.history_slot, torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]))
    assert torch.equal(batch.cell_index, torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))
    assert torch.equal(batch.candidate_priority, torch.arange(8)[None, :])
    assert torch.equal(batch.range_stratum, torch.ones(1, 8, dtype=torch.long))
    assert torch.equal(batch.age_stratum[0, :4], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch.age_stratum[0, 4:], torch.full((4,), 2))
    assert batch.score_features[0, 4, 7].item() == pytest.approx(0.1)
    assert batch.score_features[0, 4, 8].item() == pytest.approx(0.1)
    assert batch.terrain_values[0, 4, 4].item() == pytest.approx(0.1)
    assert metadata.registration_sha256 == metadata.computed_sha256()
    assert torch.equal(metadata.role_eligibility, batch.role_eligibility)


def test_history_points_require_causal_acquisition_to_current_transform() -> None:
    """Align an older return with current support geometry using causal motion."""
    geometry = _geometry()
    current, landing = _centres()
    landing[:, 0] = torch.tensor([1.0, 1.0, 0.0])
    rotation, translation = _transforms(history_length=2)
    translation[:, 1, 1] = 1.0
    ranges = torch.ones(1, 2, 1, 4)
    batch = geometry(
        ranges,
        torch.ones_like(ranges, dtype=torch.bool),
        torch.zeros_like(ranges),
        torch.zeros(1, 2),
        rotation,
        translation,
        current,
        landing,
    )

    assert torch.equal(batch.body_points[0, 4], torch.tensor([1.0, 1.0, 0.0]))
    assert batch.role_eligibility[0, 1, 4]
    assert not batch.role_eligibility[0, 1, 0]


def test_packet_age_is_effective_age_when_raycaster_has_no_return_time() -> None:
    """Use packet age for honest RayCaster packet-time training observations."""
    geometry = _geometry()
    current, landing = _centres()
    rotation, translation = _transforms()
    batch = geometry(
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4, dtype=torch.bool),
        torch.zeros(1, 1, 1, 4),
        torch.tensor([[0.1]]),
        rotation,
        translation,
        current,
        landing,
    )

    assert batch.token_valid.all()
    assert torch.allclose(batch.score_features[..., 7], torch.full((1, 4), 0.1))
    assert torch.allclose(batch.score_features[..., 8], torch.full((1, 4), 0.1))
    assert torch.allclose(batch.terrain_values[..., 4], torch.full((1, 4), 0.1))
    assert torch.equal(batch.age_stratum, torch.full((1, 4), 2))


def test_receipt_does_not_overclaim_external_calibration_or_training() -> None:
    """Keep calibration, formal-task, training, and real-robot claims false."""
    geometry = _geometry()
    receipt = geometry.receipt()

    assert receipt["external_calibration_sha256"] == "a" * 64
    assert receipt["external_calibration_verified_by_component"] is False
    assert receipt["invalid_cell_semantics"] == "unknown_not_free_space"
    assert receipt["evidence_age_semantics"] == "max(return_age_s,packet_age_s)"
    assert receipt["simulator_contact_truth"] is False
    assert receipt["simulator_terrain_truth"] is False
    assert receipt["training_ready"] is False
    assert receipt["g1_closed_loop_validated"] is False


def test_rejects_invalid_calibration_and_negative_packet_age() -> None:
    """Fail closed on improper extrinsics and noncausal packet ages."""
    with pytest.raises(ValueError, match="proper rotation"):
        CalibratedSphericalSupportRoleGeometry(
            torch.tensor([[[1.0, 0.0, 0.0]]]),
            torch.diag(torch.tensor([1.0, 1.0, -1.0])),
            torch.zeros(3),
            external_calibration_sha256="b" * 64,
            current_radius=0.1,
            landing_radius=0.1,
            vertical_half_extent=0.1,
            min_range=0.1,
            max_range=5.0,
            range_strata_edges=(1.0,),
            age_strata_edges=(0.1,),
        )

    geometry = _geometry()
    current, landing = _centres()
    rotation, translation = _transforms()
    with pytest.raises(ValueError, match="packet_age"):
        geometry(
            torch.ones(1, 1, 1, 4),
            torch.ones(1, 1, 1, 4, dtype=torch.bool),
            torch.zeros(1, 1, 1, 4),
            torch.tensor([[-0.1]]),
            rotation,
            translation,
            current,
            landing,
        )
