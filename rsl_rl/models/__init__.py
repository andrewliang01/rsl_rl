# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .amp_discriminator import AMPDiscriminator
from .causal_spherical_support_actor_model import CausalSphericalSupportActorModel
from .cnn_model import CNNModel
from .mlp_model import MLPModel
from .prop_mlp_elevation_fusion_model import PropMLPElevationFusionModel
from .rnn_model import RNNModel
from .shared_prop_mlp_elevation_multi_head_critic import SharedPropMLPElevationMultiHeadCritic

__all__ = [
    "AMPDiscriminator",
    "CNNModel",
    "CausalSphericalSupportActorModel",
    "MLPModel",
    "PropMLPElevationFusionModel",
    "RNNModel",
    "SharedPropMLPElevationMultiHeadCritic",
]
