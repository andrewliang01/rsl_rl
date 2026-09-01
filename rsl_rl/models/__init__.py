# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .amp_discriminator import AMPDiscriminator
from .cnn_model import CNNModel
from .mlp_model import MLPModel
from .m2m_frozen_ecmm import M2MFrozenECMMCore, load_frozen_m90_ecmm_core
from .m2m_frozen_scratch_teacher import M2MFrozenScratchTeacherCore
from .m2m_observed_history_formal_teacher import (
    M2MObservedHistoryFormalTeacher,
    M2MObservedHistoryMapEncoder,
)
from .m2m_observed_history_scratch_teacher import (
    M2MObservedHistoryScratchTeacher,
    M2MScratchTeacherMapContract,
)
from .m2m_observed_history_teacher import M2MObservedHistoryProxyTeacher, ObservedHistoryMapContract
from .m2m_recurrent_student import M2MMapFreeRecurrentStudent, M2MStrictFrameTokenizer
from .m2m_livox_deformable_memory_actor import M2MLivoxDeformableMemoryActor
from .m2m_sequence_compatible_critic import M2MSequenceCompatibleCritic
from .m2m_student_only import M2MStudentOnlyPolicy, normalize_m2m_student_network_config
from .rnn_model import RNNModel
from .prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from .prop_mlp_elevation_unifp_fusion_model import PropMLPElevationUniFPFusionModel
from .shared_prop_mlp_elevation_multi_head_critic import SharedPropMLPElevationMultiHeadCritic
from .unifp_adaptation_actor import UniFPAdaptationActor
from .unifp_dual_history_actor import UniFPDualHistoryActor

__all__ = [
    "AMPDiscriminator",
    "CNNModel",
    "MLPModel",
    "M2MFrozenECMMCore",
    "M2MFrozenScratchTeacherCore",
    "load_frozen_m90_ecmm_core",
    "M2MObservedHistoryFormalTeacher",
    "M2MObservedHistoryMapEncoder",
    "M2MObservedHistoryScratchTeacher",
    "M2MScratchTeacherMapContract",
    "M2MObservedHistoryProxyTeacher",
    "ObservedHistoryMapContract",
    "M2MMapFreeRecurrentStudent",
    "M2MLivoxDeformableMemoryActor",
    "M2MStrictFrameTokenizer",
    "M2MSequenceCompatibleCritic",
    "M2MStudentOnlyPolicy",
    "normalize_m2m_student_network_config",
    "RNNModel",
    "PropMLPElevationFusionModel",
    "PropMLPElevationUniFPFusionModel",
    "SharedPropMLPElevationMultiHeadCritic",
    "UniFPAdaptationActor",
    "UniFPDualHistoryActor",
]
