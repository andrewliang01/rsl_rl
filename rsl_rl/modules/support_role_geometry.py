# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment-observable support-role geometry for spherical range history.

The module converts calibrated spherical return cells into body-frame surface
points, then constructs four support-role masks around the current and causal
landing centres of the left and right feet.  It consumes no simulator contact
or terrain truth.  Invalid cells remain unknown; they are never interpreted as
free space or a maximum-range return.
"""

from __future__ import annotations

import hashlib
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final

from .shared_unique_support_actor import (
    MatchedSubstitutionMetadata,
    SupportMaskProvenance,
)

SUPPORT_ROLE_NAMES: Final[tuple[str, ...]] = (
    "left_current",
    "left_landing",
    "right_current",
    "right_landing",
)


def _configuration_sha256(
    text_fields: tuple[str, ...], tensor_fields: tuple[torch.Tensor, ...]
) -> str:
    digest = hashlib.sha256(b"support_role_geometry_config_v1")
    for value in text_fields:
        digest.update(value.encode("utf-8"))
    for tensor in tensor_fields:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


def _strict_edges(name: str, values: tuple[float, ...]) -> torch.Tensor:
    edges = torch.as_tensor(values, dtype=torch.float32)
    if edges.ndim != 1 or edges.numel() == 0:
        raise ValueError(f"{name} must contain at least one boundary.")
    if not bool(torch.isfinite(edges).all()):
        raise ValueError(f"{name} must be finite.")
    if not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError(f"{name} must be strictly increasing.")
    return edges


@dataclass(frozen=True)
class SupportRoleGeometryBatch:
    """Flattened geometry tensors ready for the selected-only H1 actor."""

    score_features: torch.Tensor
    terrain_values: torch.Tensor
    token_valid: torch.Tensor
    role_eligibility: torch.Tensor
    candidate_mask: torch.Tensor
    range_stratum: torch.Tensor
    angle_stratum: torch.Tensor
    age_stratum: torch.Tensor
    candidate_priority: torch.Tensor
    body_points: torch.Tensor
    history_slot: torch.Tensor
    cell_index: torch.Tensor
    finite_gate: torch.Tensor

    def register_matched_substitution(self) -> MatchedSubstitutionMetadata:
        """Freeze exact role/range/angle/age matching metadata by SHA-256."""
        return MatchedSubstitutionMetadata.register(
            candidate_mask=self.candidate_mask,
            role_eligibility=self.role_eligibility,
            range_stratum=self.range_stratum,
            angle_stratum=self.angle_stratum,
            age_stratum=self.age_stratum,
            candidate_priority=self.candidate_priority,
        )


class CalibratedSphericalSupportRoleGeometry(nn.Module):
    """Build four physical support-query masks from calibrated return rays."""

    score_feature_dim: Final[int] = 9
    terrain_value_dim: Final[int] = 5
    score_feature_names: Final[tuple[str, ...]] = (
        "point_x_body_m",
        "point_y_body_m",
        "point_z_body_m",
        "ray_direction_x_body",
        "ray_direction_y_body",
        "ray_direction_z_body",
        "range_m",
        "evidence_age_s",
        "packet_age_s",
    )
    terrain_value_names: Final[tuple[str, ...]] = (
        "point_x_body_m",
        "point_y_body_m",
        "point_z_body_m",
        "range_m",
        "evidence_age_s",
    )

    def __init__(
        self,
        unit_ray_directions_sensor: torch.Tensor,
        sensor_to_body_rotation: torch.Tensor,
        sensor_origin_body: torch.Tensor,
        *,
        external_calibration_sha256: str,
        current_radius: float,
        landing_radius: float,
        vertical_half_extent: float,
        min_range: float,
        max_range: float,
        range_strata_edges: tuple[float, ...],
        age_strata_edges: tuple[float, ...],
        azimuth_strata: int = 8,
        elevation_strata: int = 4,
    ) -> None:
        """Register calibrated rays, extrinsics, support volumes, and strata."""
        super().__init__()
        rays = torch.as_tensor(unit_ray_directions_sensor, dtype=torch.float32)
        rotation = torch.as_tensor(sensor_to_body_rotation, dtype=torch.float32)
        origin = torch.as_tensor(sensor_origin_body, dtype=torch.float32)
        if rays.ndim != 3 or rays.shape[-1] != 3:
            raise ValueError("unit_ray_directions_sensor must have shape [H,W,3].")
        if rotation.shape != (3, 3):
            raise ValueError("sensor_to_body_rotation must have shape [3,3].")
        if origin.shape != (3,):
            raise ValueError("sensor_origin_body must have shape [3].")
        if not bool(torch.isfinite(rays).all()):
            raise ValueError("unit_ray_directions_sensor must be finite.")
        if not bool(torch.isfinite(rotation).all()) or not bool(
            torch.isfinite(origin).all()
        ):
            raise ValueError("Sensor extrinsics must be finite.")
        ray_norm = torch.linalg.vector_norm(rays, dim=-1)
        if not torch.allclose(ray_norm, torch.ones_like(ray_norm), atol=1.0e-5):
            raise ValueError("Every spherical ray direction must be unit length.")
        identity = rotation @ rotation.transpose(0, 1)
        if not torch.allclose(identity, torch.eye(3), atol=1.0e-5):
            raise ValueError("sensor_to_body_rotation must be orthonormal.")
        if not torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1.0e-5):
            raise ValueError("sensor_to_body_rotation must be a proper rotation.")
        if (
            not isinstance(external_calibration_sha256, str)
            or len(external_calibration_sha256) != 64
        ):
            raise ValueError("external_calibration_sha256 must be a SHA-256 hex string.")
        try:
            int(external_calibration_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "external_calibration_sha256 must be a SHA-256 hex string."
            ) from error

        self.height, self.width = int(rays.shape[0]), int(rays.shape[1])
        self.current_radius = _positive_real("current_radius", current_radius)
        self.landing_radius = _positive_real("landing_radius", landing_radius)
        self.vertical_half_extent = _positive_real(
            "vertical_half_extent", vertical_half_extent
        )
        self.min_range = _positive_real("min_range", min_range)
        self.max_range = _positive_real("max_range", max_range)
        if self.max_range <= self.min_range:
            raise ValueError("max_range must be greater than min_range.")
        self.azimuth_strata = _positive_integer(
            "azimuth_strata", azimuth_strata
        )
        self.elevation_strata = _positive_integer(
            "elevation_strata", elevation_strata
        )
        self.external_calibration_sha256 = external_calibration_sha256.lower()

        body_rays = torch.einsum("ij,hwj->hwi", rotation, rays)
        self.register_buffer("body_ray_directions", body_rays.contiguous())
        self.register_buffer("sensor_origin_body", origin.contiguous())
        self.register_buffer(
            "range_strata_edges",
            _strict_edges("range_strata_edges", range_strata_edges),
        )
        self.register_buffer(
            "age_strata_edges",
            _strict_edges("age_strata_edges", age_strata_edges),
        )

    @property
    def num_cells(self) -> int:
        """Return the number of angular cells in one spherical frame."""
        return self.height * self.width

    def provenance(self) -> SupportMaskProvenance:
        """Declare the only admissible sources used by the role masks."""
        return SupportMaskProvenance(
            geometry_source="calibrated_lidar_ray_geometry",
            uses_proprioception=True,
            uses_gait_phase=True,
        )

    def receipt(self) -> dict[str, object]:
        """Return strict evidence boundaries for this component."""
        return {
            "schema": "calibrated_spherical_support_role_geometry_v1",
            "external_calibration_sha256": self.external_calibration_sha256,
            "external_calibration_verified_by_component": False,
            "role_order": SUPPORT_ROLE_NAMES,
            "invalid_cell_semantics": "unknown_not_free_space",
            "evidence_age_semantics": "max(return_age_s,packet_age_s)",
            "score_feature_names": self.score_feature_names,
            "terrain_value_names": self.terrain_value_names,
            "geometry_inputs": (
                "calibrated_spherical_return_rays",
                "causal_acquisition_body_to_current_body_transform",
                "joint_encoder_foot_fk_centres",
                "causal_command_gait_phase_landing_centres",
            ),
            "simulator_contact_truth": False,
            "simulator_terrain_truth": False,
            "registered_lab_task": False,
            "training_ready": False,
            "real_calibration_evidence_bound": False,
            "g1_closed_loop_validated": False,
            "global_map_or_long_horizon_odometry_required": False,
            "configuration_sha256": _configuration_sha256(
                (
                    self.external_calibration_sha256,
                    str((self.height, self.width)),
                    str(self.current_radius),
                    str(self.landing_radius),
                    str(self.vertical_half_extent),
                    str(self.min_range),
                    str(self.max_range),
                    str(self.azimuth_strata),
                    str(self.elevation_strata),
                ),
                (
                    self.body_ray_directions,
                    self.sensor_origin_body,
                    self.range_strata_edges,
                    self.age_strata_edges,
                ),
            ),
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
        landing_foot_centres_body: torch.Tensor,
    ) -> SupportRoleGeometryBatch:
        """Convert ``[B,K,H,W]`` return history into flattened H1 inputs."""
        self._validate_inputs(
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            landing_foot_centres_body,
            validate_values=True,
        )
        return self._forward_impl(
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            landing_foot_centres_body,
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
        landing_foot_centres_body: torch.Tensor,
    ) -> SupportRoleGeometryBatch:
        """Build geometry without tensor-to-host checks and return a tensor gate."""
        self._validate_inputs(
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            landing_foot_centres_body,
            validate_values=False,
        )
        return self._forward_impl(
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            landing_foot_centres_body,
        )

    def _forward_impl(
        self,
        range_history: torch.Tensor,
        return_valid_history: torch.Tensor,
        return_age_history: torch.Tensor,
        packet_age: torch.Tensor,
        history_body_to_current_rotation: torch.Tensor,
        history_body_to_current_translation: torch.Tensor,
        current_foot_centres_body: torch.Tensor,
        landing_foot_centres_body: torch.Tensor,
    ) -> SupportRoleGeometryBatch:
        """Run the shared tensor-only geometry implementation."""
        batch_size, history_length = range_history.shape[:2]
        compute_dtype = self.body_ray_directions.dtype
        ranges = range_history.to(dtype=compute_dtype)
        return_age = return_age_history.to(dtype=compute_dtype)
        packet_age_value = packet_age.to(dtype=compute_dtype)
        history_rotation = history_body_to_current_rotation.to(dtype=compute_dtype)
        history_translation = history_body_to_current_translation.to(
            dtype=compute_dtype
        )
        current_centres = current_foot_centres_body.to(dtype=compute_dtype)
        landing_centres = landing_foot_centres_body.to(dtype=compute_dtype)

        packet_age_grid = packet_age_value[:, :, None, None].expand(
            -1, -1, self.height, self.width
        )
        finite = torch.isfinite(ranges) & torch.isfinite(return_age)
        range_admissible = (ranges >= self.min_range) & (ranges <= self.max_range)
        age_admissible = return_age >= 0.0
        token_valid_grid = (
            return_valid_history & finite & range_admissible & age_admissible
        )
        valid_event_gate = (
            ~return_valid_history
            | (finite & range_admissible & age_admissible)
        ).all(dim=(-1, -2, -3))
        invalid_event_gate = (
            return_valid_history
            | ((ranges == 0.0) & (return_age == 0.0))
        ).all(dim=(-1, -2, -3))
        packet_gate = (
            torch.isfinite(packet_age_value) & (packet_age_value >= 0.0)
        ).all(dim=-1)
        body_gate = (
            torch.isfinite(history_rotation).all(dim=(-1, -2, -3))
            & torch.isfinite(history_translation).all(dim=(-1, -2))
            & torch.isfinite(current_centres).all(dim=(-1, -2))
            & torch.isfinite(landing_centres).all(dim=(-1, -2))
        )
        rotation_identity = torch.matmul(
            history_rotation,
            history_rotation.transpose(-1, -2),
        )
        expected_identity = torch.eye(
            3,
            device=history_rotation.device,
            dtype=history_rotation.dtype,
        )
        rotation_gate = (
            (rotation_identity - expected_identity).abs().amax(dim=(-1, -2))
            <= 1.0e-4
        ).all(dim=-1) & (
            (torch.linalg.det(history_rotation) - 1.0).abs() <= 1.0e-4
        ).all(dim=-1)
        finite_gate = (
            valid_event_gate
            & invalid_event_gate
            & packet_gate
            & body_gate
            & rotation_gate
        )
        safe_ranges = torch.where(token_valid_grid, ranges, torch.zeros_like(ranges))
        acquisition_rays = self.body_ray_directions[None, None, :, :, :]
        acquisition_points = (
            safe_ranges[..., None] * acquisition_rays
            + self.sensor_origin_body[None, None, None, None, :]
        )
        points = torch.einsum(
            "bkij,bkhwj->bkhwi", history_rotation, acquisition_points
        ) + history_translation[:, :, None, None, :]
        directions = torch.einsum(
            "bkij,bkhwj->bkhwi",
            history_rotation,
            acquisition_rays.expand(batch_size, history_length, -1, -1, -1),
        )

        points_flat = points.flatten(1, 3)
        valid_flat = token_valid_grid.flatten(1, 3)
        ranges_flat = safe_ranges.flatten(1, 3)
        evidence_age_grid = torch.maximum(return_age, packet_age_grid)
        evidence_age_flat = torch.where(
            token_valid_grid,
            evidence_age_grid,
            torch.zeros_like(evidence_age_grid),
        ).flatten(1, 3)
        packet_age_flat = torch.where(
            token_valid_grid, packet_age_grid, torch.zeros_like(packet_age_grid)
        ).flatten(1, 3)
        directions_flat = directions.flatten(1, 3)

        centres = torch.stack(
            (
                current_centres[:, 0],
                landing_centres[:, 0],
                current_centres[:, 1],
                landing_centres[:, 1],
            ),
            dim=1,
        )
        radii = ranges_flat.new_tensor(
            (
                self.current_radius,
                self.landing_radius,
                self.current_radius,
                self.landing_radius,
            )
        )
        delta = points_flat[:, None, :, :] - centres[:, :, None, :]
        horizontal_distance_sq = delta[..., 0].square() + delta[..., 1].square()
        inside_horizontal = horizontal_distance_sq <= radii[None, :, None].square()
        inside_vertical = delta[..., 2].abs() <= self.vertical_half_extent
        role_eligibility = (
            inside_horizontal & inside_vertical & valid_flat[:, None, :]
        )
        candidate_mask = role_eligibility.any(dim=1)

        range_stratum = torch.bucketize(
            ranges_flat.contiguous(), self.range_strata_edges
        )
        age_stratum = torch.bucketize(
            evidence_age_flat.contiguous(), self.age_strata_edges
        )
        azimuth = torch.atan2(directions_flat[..., 1], directions_flat[..., 0])
        elevation = torch.asin(directions_flat[..., 2].clamp(-1.0, 1.0))
        azimuth_id = torch.floor(
            (azimuth + math.pi) * self.azimuth_strata / (2.0 * math.pi)
        ).to(torch.long).clamp_(0, self.azimuth_strata - 1)
        elevation_id = torch.floor(
            (elevation + math.pi / 2.0) * self.elevation_strata / math.pi
        ).to(torch.long).clamp_(0, self.elevation_strata - 1)
        angle_stratum = elevation_id * self.azimuth_strata + azimuth_id

        history_slot = torch.arange(
            history_length, device=ranges.device, dtype=torch.long
        ).view(1, history_length, 1).expand(batch_size, -1, self.num_cells)
        history_slot = history_slot.flatten(1, 2)
        cell_index = torch.arange(
            self.num_cells, device=ranges.device, dtype=torch.long
        ).view(1, 1, self.num_cells).expand(batch_size, history_length, -1)
        cell_index = cell_index.flatten(1, 2)
        candidate_priority = history_slot * self.num_cells + cell_index

        score_features = torch.cat(
            (
                points_flat,
                directions_flat,
                ranges_flat[..., None],
                evidence_age_flat[..., None],
                packet_age_flat[..., None],
            ),
            dim=-1,
        )
        terrain_values = torch.cat(
            (
                points_flat,
                ranges_flat[..., None],
                evidence_age_flat[..., None],
            ),
            dim=-1,
        )
        score_features = torch.where(
            valid_flat[..., None], score_features, torch.zeros_like(score_features)
        )
        terrain_values = torch.where(
            valid_flat[..., None], terrain_values, torch.zeros_like(terrain_values)
        )
        range_stratum = torch.where(
            valid_flat, range_stratum, torch.full_like(range_stratum, -1)
        )
        angle_stratum = torch.where(
            valid_flat, angle_stratum, torch.full_like(angle_stratum, -1)
        )
        age_stratum = torch.where(
            valid_flat, age_stratum, torch.full_like(age_stratum, -1)
        )

        return SupportRoleGeometryBatch(
            score_features=score_features,
            terrain_values=terrain_values,
            token_valid=valid_flat,
            role_eligibility=role_eligibility,
            candidate_mask=candidate_mask,
            range_stratum=range_stratum,
            angle_stratum=angle_stratum,
            age_stratum=age_stratum,
            candidate_priority=candidate_priority,
            body_points=torch.where(
                valid_flat[..., None], points_flat, torch.zeros_like(points_flat)
            ),
            history_slot=history_slot,
            cell_index=cell_index,
            finite_gate=finite_gate,
        )

    def _validate_inputs(
        self,
        range_history: torch.Tensor,
        return_valid_history: torch.Tensor,
        return_age_history: torch.Tensor,
        packet_age: torch.Tensor,
        history_body_to_current_rotation: torch.Tensor,
        history_body_to_current_translation: torch.Tensor,
        current_foot_centres_body: torch.Tensor,
        landing_foot_centres_body: torch.Tensor,
        *,
        validate_values: bool,
    ) -> None:
        expected_tail = (self.height, self.width)
        if range_history.ndim != 4 or tuple(range_history.shape[-2:]) != expected_tail:
            raise ValueError("range_history must have shape [B,K,H,W].")
        if return_valid_history.shape != range_history.shape:
            raise ValueError("return_valid_history must match range_history shape.")
        if return_valid_history.dtype != torch.bool:
            raise TypeError("return_valid_history must have bool dtype.")
        if return_age_history.shape != range_history.shape:
            raise ValueError("return_age_history must match range_history shape.")
        batch_size, history_length = range_history.shape[:2]
        if packet_age.shape != (batch_size, history_length):
            raise ValueError("packet_age must have shape [B,K].")
        if history_body_to_current_rotation.shape != (
            batch_size,
            history_length,
            3,
            3,
        ):
            raise ValueError(
                "history_body_to_current_rotation must have shape [B,K,3,3]."
            )
        if history_body_to_current_translation.shape != (
            batch_size,
            history_length,
            3,
        ):
            raise ValueError(
                "history_body_to_current_translation must have shape [B,K,3]."
            )
        expected_centres = (batch_size, 2, 3)
        if current_foot_centres_body.shape != expected_centres:
            raise ValueError("current_foot_centres_body must have shape [B,2,3].")
        if landing_foot_centres_body.shape != expected_centres:
            raise ValueError("landing_foot_centres_body must have shape [B,2,3].")
        for name, value in (
            ("range_history", range_history),
            ("return_age_history", return_age_history),
            ("packet_age", packet_age),
            ("history_body_to_current_rotation", history_body_to_current_rotation),
            (
                "history_body_to_current_translation",
                history_body_to_current_translation,
            ),
            ("current_foot_centres_body", current_foot_centres_body),
            ("landing_foot_centres_body", landing_foot_centres_body),
        ):
            if not value.dtype.is_floating_point:
                raise TypeError(f"{name} must have floating-point dtype.")
        tensors = (
            range_history,
            return_valid_history,
            return_age_history,
            packet_age,
            history_body_to_current_rotation,
            history_body_to_current_translation,
            current_foot_centres_body,
            landing_foot_centres_body,
        )
        if any(value.device != range_history.device for value in tensors):
            raise ValueError("All support-role geometry inputs must share one device.")
        if range_history.device != self.body_ray_directions.device:
            raise ValueError("Geometry inputs and calibrated buffers must share one device.")
        if not validate_values:
            return
        if not bool(torch.isfinite(packet_age).all()) or bool((packet_age < 0).any()):
            raise ValueError("packet_age must be finite and non-negative.")
        if not bool(torch.isfinite(history_body_to_current_rotation).all()) or not bool(
            torch.isfinite(history_body_to_current_translation).all()
        ):
            raise ValueError("History-to-current body transforms must be finite.")
        rotation_identity = torch.matmul(
            history_body_to_current_rotation,
            history_body_to_current_rotation.transpose(-1, -2),
        )
        expected_identity = torch.eye(
            3,
            device=rotation_identity.device,
            dtype=rotation_identity.dtype,
        ).expand_as(rotation_identity)
        determinant = torch.linalg.det(history_body_to_current_rotation)
        if not torch.allclose(rotation_identity, expected_identity, atol=1.0e-4):
            raise ValueError("History-to-current rotations must be orthonormal.")
        if not torch.allclose(determinant, torch.ones_like(determinant), atol=1.0e-4):
            raise ValueError("History-to-current rotations must be proper rotations.")
        if not bool(torch.isfinite(current_foot_centres_body).all()) or not bool(
            torch.isfinite(landing_foot_centres_body).all()
        ):
            raise ValueError("Foot centres must be finite.")


__all__ = [
    "SUPPORT_ROLE_NAMES",
    "CalibratedSphericalSupportRoleGeometry",
    "SupportRoleGeometryBatch",
]
