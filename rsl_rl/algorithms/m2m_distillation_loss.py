# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict masked latent/action objectives for MID-360 M2M distillation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class M2MDistillationLossConfig:
    """Fail-closed definition of one M2M distillation objective.

    ``objective_mode`` makes ablations explicit.  In particular, setting both
    latent weights to zero is accepted only for ``"action_only"``; an
    accidental all-zero or mislabeled objective is rejected at construction.
    """

    objective_mode: str = "joint"
    latent_smooth_l1_weight: float = 1.0
    latent_cosine_weight: float = 0.1
    action_mse_weight: float = 1.0
    smooth_l1_beta: float = 1.0
    latent_normalization: str = "l2"
    normalization_eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.objective_mode not in {"joint", "action_only", "latent_only"}:
            raise ValueError(
                "objective_mode must be one of 'joint', 'action_only', or "
                f"'latent_only', got {self.objective_mode!r}."
            )
        for name in (
            "latent_smooth_l1_weight",
            "latent_cosine_weight",
            "action_mse_weight",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be an exact finite float >= 0, got {value!r}.")
        if type(self.smooth_l1_beta) is not float or not math.isfinite(self.smooth_l1_beta):
            raise ValueError("smooth_l1_beta must be an exact finite float.")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError(f"smooth_l1_beta must be positive, got {self.smooth_l1_beta}.")
        if self.latent_normalization != "l2":
            raise ValueError(
                "The formal C11 objective currently supports only latent_normalization='l2', "
                f"got {self.latent_normalization!r}."
            )
        if type(self.normalization_eps) is not float or not math.isfinite(self.normalization_eps):
            raise ValueError("normalization_eps must be an exact finite float.")
        if self.normalization_eps <= 0.0:
            raise ValueError(f"normalization_eps must be positive, got {self.normalization_eps}.")

        latent_enabled = self.latent_smooth_l1_weight > 0.0 or self.latent_cosine_weight > 0.0
        action_enabled = self.action_mse_weight > 0.0
        if self.objective_mode == "joint" and not (latent_enabled and action_enabled):
            raise ValueError("objective_mode='joint' requires positive latent and action supervision weights.")
        if self.objective_mode == "action_only" and (latent_enabled or not action_enabled):
            raise ValueError(
                "objective_mode='action_only' requires both latent weights to be zero and action_mse_weight > 0."
            )
        if self.objective_mode == "latent_only" and (not latent_enabled or action_enabled):
            raise ValueError(
                "objective_mode='latent_only' requires a positive latent weight and action_mse_weight == 0."
            )

    def receipt(self) -> dict[str, Any]:
        """Return an exact JSON-compatible loss receipt."""
        return asdict(self)


class M2MMaskedLatentActionLoss(nn.Module):
    """Compute feature-mean, valid-step-mean M2M losses.

    Invalid padded positions are indexed out *before* any arithmetic.  This is
    stronger than multiplying by a zero mask: NaN/Inf or arbitrarily large
    padding therefore cannot contaminate the reported objective.
    """

    latent_dim: int = 64
    action_dim: int = 29

    def __init__(self, config: M2MDistillationLossConfig) -> None:
        super().__init__()
        if not isinstance(config, M2MDistillationLossConfig):
            raise TypeError("config must be M2MDistillationLossConfig.")
        self.config = config

    @staticmethod
    def _valid_mask(masks: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool:
            raise ValueError("M2M loss masks must be a bool tensor.")
        if tuple(masks.shape) == leading_shape:
            valid = masks
        elif tuple(masks.shape) == (*leading_shape, 1):
            valid = masks.squeeze(-1)
        else:
            raise ValueError(
                "M2M loss masks must match prediction leading dimensions, got "
                f"mask={tuple(masks.shape)}, leading={leading_shape}."
            )
        if not torch.any(valid):
            raise ValueError("M2M loss requires at least one valid (non-padding) transition.")
        return valid

    @staticmethod
    def _validate_pair(
        name: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        feature_dim: int,
    ) -> None:
        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError(f"{name} prediction and target must be tensors.")
        if prediction.shape != target.shape:
            raise ValueError(
                f"{name} prediction/target shapes differ: {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim < 2 or prediction.shape[-1] != feature_dim:
            raise ValueError(f"{name} tensors must end in dimension {feature_dim}, got {tuple(prediction.shape)}.")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError(f"{name} prediction and target must be floating-point tensors.")
        if prediction.device != target.device:
            raise ValueError(f"{name} prediction and target must share a device.")

    def forward(
        self,
        student_latent_A: torch.Tensor,  # noqa: N803 - paper/contract symbol is A
        teacher_latent_A: torch.Tensor,  # noqa: N803 - paper/contract symbol is A
        student_action_mean: torch.Tensor,
        teacher_action_mean: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted total and detached-free scalar components."""
        self._validate_pair("latent", student_latent_A, teacher_latent_A, self.latent_dim)
        self._validate_pair("action", student_action_mean, teacher_action_mean, self.action_dim)
        if student_latent_A.shape[:-1] != student_action_mean.shape[:-1]:
            raise ValueError("Latent and action predictions must share leading dimensions.")
        if student_latent_A.device != student_action_mean.device:
            raise ValueError("Latent and action predictions must share a device.")
        valid = self._valid_mask(masks, tuple(student_latent_A.shape[:-1]))

        # Select valid transitions before normalization/subtraction.  Padded
        # entries may contain any value without affecting loss or gradients.
        student_latent = student_latent_A[valid]
        teacher_latent = teacher_latent_A[valid].to(dtype=student_latent.dtype)
        student_action = student_action_mean[valid]
        teacher_action = teacher_action_mean[valid].to(dtype=student_action.dtype)

        student_latent_normalized = F.normalize(
            student_latent,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_eps,
        )
        teacher_latent_normalized = F.normalize(
            teacher_latent,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_eps,
        )
        latent_smooth_l1 = F.smooth_l1_loss(
            student_latent_normalized,
            teacher_latent_normalized,
            beta=self.config.smooth_l1_beta,
            reduction="none",
        ).mean(dim=-1).mean()
        latent_cosine = (
            1.0
            - F.cosine_similarity(
                student_latent,
                teacher_latent,
                dim=-1,
                eps=self.config.normalization_eps,
            )
        ).mean()
        action_mse = F.mse_loss(student_action, teacher_action, reduction="none").mean(dim=-1).mean()

        total = (
            self.config.latent_smooth_l1_weight * latent_smooth_l1
            + self.config.latent_cosine_weight * latent_cosine
            + self.config.action_mse_weight * action_mse
        )
        components = {
            "latent_smooth_l1": latent_smooth_l1,
            "latent_cosine": latent_cosine,
            "action_mean_mse": action_mse,
            "valid_steps": valid.sum().to(dtype=total.dtype),
        }
        return total, components


__all__ = ["M2MDistillationLossConfig", "M2MMaskedLatentActionLoss"]
