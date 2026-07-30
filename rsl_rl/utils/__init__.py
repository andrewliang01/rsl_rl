# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions."""

from .motion_loader import AMPLoader
from .motion_loader_for_display import AMPLoaderDisplay
from .mid360_ray_time_builder import (
    MID360_NORMALIZED_SENSOR_FRAME,
    MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS,
    MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY,
    MID360_TIMESTAMP_LIVOX_CUSTOM_MSG,
    Mid360PacketAdapter,
    Mid360PacketStats,
    Mid360PointPacket,
    Mid360RayTimeBuilderError,
    Mid360RayTimeTensorBuilder,
    StaleMid360PacketError,
    point_packet_from_livox_custom_msg_arrays,
)
from .ray_time_deployment_manifest import (
    RayTimeManifestError,
    build_ray_time_deployment_manifest,
    canonical_json_bytes,
    canonical_json_sha256,
    collect_git_provenance,
    default_ray_time_channels,
    default_ray_time_mount_geometry,
    default_ray_time_proprio_terms,
    default_ray_time_tensorization,
    read_ray_time_deployment_manifest,
    serialize_ray_time_deployment_manifest,
    validate_ray_time_deployment_manifest,
    write_ray_time_deployment_manifest,
)
from .ray_time_export_attestation import (
    RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME,
    RAY_TIME_EXPORT_ATTESTATION_SCHEMA_VERSION,
    RayTimeExportAttestationError,
    build_ray_time_export_attestation,
    capture_ray_time_checkpoint_snapshot,
    validate_ray_time_export_attestation,
)

from .utils import (
    check_nan,
    get_param,
    resolve_callable,
    resolve_nn_activation,
    resolve_obs_groups,
    resolve_optimizer,
    split_and_pad_trajectories,
    unpad_trajectories,
)

__all__ = [
    "AMPLoader",
    "MID360_NORMALIZED_SENSOR_FRAME",
    "MID360_TIMESTAMP_ADAPTER_ABSOLUTE_POINTS",
    "MID360_TIMESTAMP_CAPTURE_WINDOW_ONLY",
    "MID360_TIMESTAMP_LIVOX_CUSTOM_MSG",
    "Mid360PacketAdapter",
    "Mid360PacketStats",
    "Mid360PointPacket",
    "Mid360RayTimeBuilderError",
    "Mid360RayTimeTensorBuilder",
    "RayTimeManifestError",
    "RayTimeExportAttestationError",
    "RAY_TIME_EXPORT_ATTESTATION_SCHEMA_NAME",
    "RAY_TIME_EXPORT_ATTESTATION_SCHEMA_VERSION",
    "StaleMid360PacketError",
    "build_ray_time_deployment_manifest",
    "build_ray_time_export_attestation",
    "capture_ray_time_checkpoint_snapshot",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "check_nan",
    "collect_git_provenance",
    "default_ray_time_channels",
    "default_ray_time_mount_geometry",
    "default_ray_time_proprio_terms",
    "default_ray_time_tensorization",
    "get_param",
    "point_packet_from_livox_custom_msg_arrays",
    "resolve_callable",
    "resolve_nn_activation",
    "resolve_obs_groups",
    "resolve_optimizer",
    "read_ray_time_deployment_manifest",
    "serialize_ray_time_deployment_manifest",
    "split_and_pad_trajectories",
    "unpad_trajectories",
    "validate_ray_time_deployment_manifest",
    "validate_ray_time_export_attestation",
    "write_ray_time_deployment_manifest",
]
