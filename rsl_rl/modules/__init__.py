# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .cnn import CNN
from .bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    H0B_MANIFEST_TARGET_CONTRACT_ENCODING,
    SphericalAutoencoderOutput,
    SphericalAutoencoderPretrainHead,
    SphericalRangeFrameEncoder,
    audit_heightmap_target_contract_json_bytes,
    create_frozen_reconstructor_checkpoint,
    freeze_reconstructor,
    load_frozen_reconstructor_checkpoint,
    manifest_target_contract_json_bytes,
    manifest_target_contract_payload_sha256,
    normalize_heightmap_target_contract,
    preflight_validate_lidar_history,
    reconstructor_checkpoint_schema,
    spherical_valid_bce,
    supervised_height_valid_mse,
    validate_manifest_target_contract_binding,
    valid_masked_range_mse,
)
from .distribution import Distribution, GaussianDistribution, HeteroscedasticGaussianDistribution
from .mlp import MLP
from .multimodal_ray_evidence_encoder import MultimodalRayEvidenceEncoder
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .r2plus1d_elevation_encoder import R2Plus1DBlock, R2Plus1DElevationEncoder
from .ray_time_attention_encoder import RayTimeAttentionEncoder
from .ray_return_event_time import RayReturnEventTimeEncoder
from .cteq_dual_event_hazard import (
    CteqAdministrativeSurvivalLoss,
    CteqDualEventHazardHead,
    CteqIndependentSurvivalLoss,
)
from .ray_event_ablation import RayEventAblationOutput, RayEventAblationRouter
from .rnn import RNN, HiddenState
from .sparse_support_evidence_bottleneck import SparseSupportEvidenceBottleneck
from .support_selection_ablation import FixedBudgetSupportSelector

__all__ = [
    "CNN",
    "BankLidarHeightmapReconstructor",
    "H0B_MANIFEST_TARGET_CONTRACT_ENCODING",
    "SphericalAutoencoderOutput",
    "SphericalAutoencoderPretrainHead",
    "SphericalRangeFrameEncoder",
    "audit_heightmap_target_contract_json_bytes",
    "create_frozen_reconstructor_checkpoint",
    "freeze_reconstructor",
    "load_frozen_reconstructor_checkpoint",
    "manifest_target_contract_json_bytes",
    "manifest_target_contract_payload_sha256",
    "normalize_heightmap_target_contract",
    "preflight_validate_lidar_history",
    "reconstructor_checkpoint_schema",
    "spherical_valid_bce",
    "supervised_height_valid_mse",
    "validate_manifest_target_contract_binding",
    "valid_masked_range_mse",
    "MLP",
    "MultimodalRayEvidenceEncoder",
    "R2Plus1DBlock",
    "R2Plus1DElevationEncoder",
    "RayTimeAttentionEncoder",
    "RayReturnEventTimeEncoder",
    "CteqDualEventHazardHead",
    "CteqAdministrativeSurvivalLoss",
    "CteqIndependentSurvivalLoss",
    "RayEventAblationOutput",
    "RayEventAblationRouter",
    "RNN",
    "SparseSupportEvidenceBottleneck",
    "FixedBudgetSupportSelector",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "GaussianDistribution",
    "HeteroscedasticGaussianDistribution",
    "HiddenState",
]
