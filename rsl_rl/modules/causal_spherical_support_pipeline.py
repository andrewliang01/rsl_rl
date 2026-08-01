# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end causal support evidence assembly before unique selection."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

from .causal_foot_support_projector import (
    CausalCommandFootSupportProjector,
    CausalFootSupportProjection,
)
from .support_role_geometry import (
    CalibratedSphericalSupportRoleGeometry,
    SupportRoleGeometryBatch,
)


@dataclass(frozen=True)
class CausalSphericalSupportEvidenceBatch:
    """Future query projection, physical token geometry, and health gate."""

    projection: CausalFootSupportProjection
    geometry: SupportRoleGeometryBatch
    finite_gate: torch.Tensor


class CausalSphericalSupportEvidencePipeline(nn.Module):
    """Compose causal command projection with calibrated spherical geometry."""

    def __init__(
        self,
        projector: CausalCommandFootSupportProjector,
        geometry: CalibratedSphericalSupportRoleGeometry,
    ) -> None:
        """Register already configured, independently auditable components."""
        super().__init__()
        if not isinstance(projector, CausalCommandFootSupportProjector):
            raise TypeError("projector must be CausalCommandFootSupportProjector.")
        if not isinstance(geometry, CalibratedSphericalSupportRoleGeometry):
            raise TypeError("geometry must be calibrated spherical support geometry.")
        self.projector = projector
        self.geometry = geometry

    def receipt(self) -> dict[str, object]:
        """Return component hashes and strict unvalidated evidence boundaries."""
        geometry_receipt = self.geometry.receipt()
        return {
            "schema": "causal_spherical_support_evidence_pipeline_v1",
            "projector": self.projector.receipt(),
            "geometry_configuration_sha256": geometry_receipt[
                "configuration_sha256"
            ],
            "selected_only_actor_connected": False,
            "registered_lab_task": False,
            "gpu_latency_measured": False,
            "training_ready": False,
            "g1_closed_loop_validated": False,
        }

    def forward(
        self,
        range_history: torch.Tensor,
        return_valid_history: torch.Tensor,
        return_age_history: torch.Tensor,
        packet_age: torch.Tensor,
        history_body_to_current_rotation: torch.Tensor,
        history_body_to_current_translation: torch.Tensor,
        current_foot_centres_body: torch.Tensor,
        commanded_planar_twist_body: torch.Tensor,
        gait_phase_sin_cos: torch.Tensor,
    ) -> CausalSphericalSupportEvidenceBatch:
        """Run the synchronized audited component path."""
        projection = self.projector(
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
        )
        geometry = self.geometry(
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            projection.future_centres_body,
        )
        return CausalSphericalSupportEvidenceBatch(
            projection=projection,
            geometry=geometry,
            finite_gate=projection.finite_gate & geometry.finite_gate,
        )

    def forward_native_training(
        self,
        range_history: torch.Tensor,
        return_valid_history: torch.Tensor,
        return_age_history: torch.Tensor,
        packet_age: torch.Tensor,
        history_body_to_current_rotation: torch.Tensor,
        history_body_to_current_translation: torch.Tensor,
        current_foot_centres_body: torch.Tensor,
        commanded_planar_twist_body: torch.Tensor,
        gait_phase_sin_cos: torch.Tensor,
    ) -> CausalSphericalSupportEvidenceBatch:
        """Run the tensor-gated training path without value-based host checks."""
        projection = self.projector.forward_native_training(
            current_foot_centres_body,
            commanded_planar_twist_body,
            gait_phase_sin_cos,
        )
        geometry = self.geometry.forward_native_training(
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            projection.future_centres_body,
        )
        return CausalSphericalSupportEvidenceBatch(
            projection=projection,
            geometry=geometry,
            finite_gate=projection.finite_gate & geometry.finite_gate,
        )


__all__ = [
    "CausalSphericalSupportEvidenceBatch",
    "CausalSphericalSupportEvidencePipeline",
]
