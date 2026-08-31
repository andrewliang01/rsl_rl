# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .amp_ppo import AMPPPO
from .distillation import Distillation
from .m2m_distillation import M2MLatentActionDistillation
from .m2m_distillation_loss import M2MDistillationLossConfig, M2MMaskedLatentActionLoss
from .m2m_direct_ppo import M2MDirectPPO
from .m2m_teacher_ppo import M2MObservedHistoryTeacherPPO
from .multi_ppo import MultiPPO
from .ppo import PPO
from .unifp_adaptation_ppo import UniFPAdaptationPPO
from .unifp_amp_ppo import UniFPAMPAdaptationPPO

__all__ = [
    "AMPPPO",
    "Distillation",
    "M2MDistillationLossConfig",
    "M2MDirectPPO",
    "M2MLatentActionDistillation",
    "M2MMaskedLatentActionLoss",
    "M2MObservedHistoryTeacherPPO",
    "MultiPPO",
    "PPO",
    "UniFPAdaptationPPO",
    "UniFPAMPAdaptationPPO",
]
