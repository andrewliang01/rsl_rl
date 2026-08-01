# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .bank_lidar_heightmap import (
    BankLidarHeightmapReconstructor,
    SphericalAutoencoderOutput,
    SphericalAutoencoderPretrainHead,
    SphericalRangeFrameEncoder,
    create_frozen_reconstructor_checkpoint,
    freeze_reconstructor,
    load_frozen_reconstructor_checkpoint,
    normalize_heightmap_target_contract,
    preflight_validate_lidar_history,
    reconstructor_checkpoint_schema,
    spherical_valid_bce,
    supervised_height_valid_mse,
    valid_masked_range_mse,
)
from .cnn import CNN
from .cteq_dual_event_hazard import (
    CteqAdministrativeSurvivalLoss,
    CteqDualEventHazardHead,
    CteqIndependentSurvivalLoss,
)
from .distribution import Distribution, GaussianDistribution, HeteroscedasticGaussianDistribution
from .mlp import MLP
from .multimodal_ray_evidence_encoder import MultimodalRayEvidenceEncoder
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .r2plus1d_elevation_encoder import R2Plus1DBlock, R2Plus1DElevationEncoder
from .ray_event_ablation import RayEventAblationOutput, RayEventAblationRouter
from .ray_return_event_time import RayReturnEventTimeEncoder
from .ray_time_attention_encoder import RayTimeAttentionEncoder
from .rnn import RNN, HiddenState
from .shared_unique_support_actor import (
    MatchedSubstitutionMetadata,
    MatchedSubstitutionShortfallError,
    SharedUniqueSupportActorAdapter,
    SupportMaskProvenance,
)
from .sparse_support_evidence_bottleneck import SparseSupportEvidenceBottleneck
from .support_role_geometry import (
    SUPPORT_ROLE_NAMES,
    CalibratedSphericalSupportRoleGeometry,
    SupportRoleGeometryBatch,
)
from .support_selection_ablation import FixedBudgetSupportSelector

__all__ = [
    "CNN",
    "MLP",
    "RNN",
    "SUPPORT_ROLE_NAMES",
    "BankLidarHeightmapReconstructor",
    "CalibratedSphericalSupportRoleGeometry",
    "CteqAdministrativeSurvivalLoss",
    "CteqDualEventHazardHead",
    "CteqIndependentSurvivalLoss",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "FixedBudgetSupportSelector",
    "GaussianDistribution",
    "HeteroscedasticGaussianDistribution",
    "HiddenState",
    "MatchedSubstitutionMetadata",
    "MatchedSubstitutionShortfallError",
    "MultimodalRayEvidenceEncoder",
    "R2Plus1DBlock",
    "R2Plus1DElevationEncoder",
    "RayEventAblationOutput",
    "RayEventAblationRouter",
    "RayReturnEventTimeEncoder",
    "RayTimeAttentionEncoder",
    "SharedUniqueSupportActorAdapter",
    "SparseSupportEvidenceBottleneck",
    "SphericalAutoencoderOutput",
    "SphericalAutoencoderPretrainHead",
    "SphericalRangeFrameEncoder",
    "SupportMaskProvenance",
    "SupportRoleGeometryBatch",
    "create_frozen_reconstructor_checkpoint",
    "freeze_reconstructor",
    "load_frozen_reconstructor_checkpoint",
    "normalize_heightmap_target_contract",
    "preflight_validate_lidar_history",
    "reconstructor_checkpoint_schema",
    "spherical_valid_bce",
    "supervised_height_valid_mse",
    "valid_masked_range_mse",
]
