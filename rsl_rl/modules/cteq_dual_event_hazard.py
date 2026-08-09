"""Torch primitives for causal CTEQ touchdown/liftoff timing.

The deployable head consumes only causal features.  Future contact labels are
accepted by the separate loss module and therefore cannot enter actor forward.
"""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F


CTEQ_NUM_FEET = 2
CTEQ_NUM_EVENTS = 2
CTEQ_NUM_BINS = 25
CTEQ_TOUCHDOWN = 0
CTEQ_LIFTOFF = 1


class CteqTorchDistribution(NamedTuple):
    hazard: torch.Tensor
    survival_before: torch.Tensor
    survival_at_boundary: torch.Tensor
    event_mass: torch.Tensor
    censor_mass: torch.Tensor
    log_survival_at_boundary: torch.Tensor
    log_event_mass: torch.Tensor
    log_censor_mass: torch.Tensor


class CteqTorchLoss(NamedTuple):
    mean_nll: torch.Tensor
    mean_brier: torch.Tensor
    per_target_nll: torch.Tensor
    per_target_brier: torch.Tensor


class CteqAdministrativeTorchLoss(NamedTuple):
    mean_nll: torch.Tensor
    mean_brier: torch.Tensor
    per_target_nll: torch.Tensor
    per_target_brier: torch.Tensor
    eligible_mask: torch.Tensor
    eligible_target_count: torch.Tensor
    excluded_target_count: torch.Tensor
    per_role_eligible_count: torch.Tensor
    per_role_nll_sum: torch.Tensor
    per_role_brier_sum: torch.Tensor
    td_event_nll_sum: torch.Tensor
    lo_event_nll_sum: torch.Tensor
    td_censored_nll_sum: torch.Tensor
    lo_censored_nll_sum: torch.Tensor
    td_event_count: torch.Tensor
    lo_event_count: torch.Tensor
    td_censored_count: torch.Tensor
    lo_censored_count: torch.Tensor


def cteq_distribution_from_logits(logits: torch.Tensor) -> CteqTorchDistribution:
    """Convert ``[B,2,2,25]`` independent hazards to event/censor masses."""

    if (
        logits.ndim != 4
        or logits.shape[1] != 2
        or logits.shape[2] != 2
        or logits.shape[3] != 25
    ):
        raise ValueError("CTEQ logits must have shape [B,2,2,25].")
    if not logits.is_floating_point():
        raise ValueError("CTEQ logits must be floating point.")
    if not torch.jit.is_tracing() and not bool(torch.isfinite(logits).all()):
        raise ValueError("CTEQ logits must be finite.")
    log_hazard = F.logsigmoid(logits)
    log_one_minus = F.logsigmoid(-logits)
    log_survival_before = torch.cat(
        (
            torch.zeros_like(log_one_minus[..., :1]),
            torch.cumsum(log_one_minus[..., :-1], dim=-1),
        ),
        dim=-1,
    )
    log_event_mass = log_survival_before + log_hazard
    log_censor_mass = torch.sum(log_one_minus, dim=-1)
    log_survival_at_boundary = torch.cat(
        (
            torch.zeros_like(log_one_minus[..., :1]),
            torch.cumsum(log_one_minus, dim=-1),
        ),
        dim=-1,
    )
    return CteqTorchDistribution(
        hazard=torch.sigmoid(logits),
        survival_before=torch.exp(log_survival_before),
        survival_at_boundary=torch.exp(log_survival_at_boundary),
        event_mass=torch.exp(log_event_mass),
        censor_mass=torch.exp(log_censor_mass),
        log_survival_at_boundary=log_survival_at_boundary,
        log_event_mass=log_event_mass,
        log_censor_mass=log_censor_mass,
    )


class CteqDualEventHazardHead(nn.Module):
    """Predict independent next-TD/next-LO hazards from causal features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128,),
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, Integral) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        resolved_hidden = tuple(int(width) for width in hidden_dims)
        if any(width <= 0 for width in resolved_hidden):
            raise ValueError("hidden_dims must contain only positive widths.")
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in resolved_hidden:
            layers.extend((nn.Linear(previous, width), nn.ELU()))
            previous = width
        layers.append(
            nn.Linear(
                previous,
                CTEQ_NUM_FEET * CTEQ_NUM_EVENTS * CTEQ_NUM_BINS,
            )
        )
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(*layers)

    def forward(self, causal_features: torch.Tensor) -> torch.Tensor:
        if causal_features.ndim != 2 or causal_features.shape[1] != self.input_dim:
            raise ValueError("causal_features must have shape [B,input_dim].")
        if not causal_features.is_floating_point():
            raise ValueError("causal_features must be floating point.")
        logits = self.network(causal_features)
        return logits.reshape(
            causal_features.shape[0],
            2,
            2,
            25,
        )


class CteqIndependentSurvivalLoss(nn.Module):
    """Independent TD/LO survival NLL plus multiclass Brier diagnostic."""

    def forward(
        self,
        logits: torch.Tensor,
        event_bin: torch.Tensor,
        event_observed: torch.Tensor,
        right_censored: torch.Tensor,
    ) -> CteqTorchLoss:
        distribution = cteq_distribution_from_logits(logits)
        expected = logits.shape[:-1]
        if tuple(event_bin.shape) != tuple(expected) or event_bin.dtype != torch.long:
            raise ValueError("event_bin must be int64 with shape [B,2,2].")
        if (
            tuple(event_observed.shape) != tuple(expected)
            or event_observed.dtype != torch.bool
            or tuple(right_censored.shape) != tuple(expected)
            or right_censored.dtype != torch.bool
        ):
            raise ValueError("Observed/censored flags must be bool [B,2,2].")
        if any(
            tensor.device != logits.device
            for tensor in (event_bin, event_observed, right_censored)
        ):
            raise ValueError("CTEQ logits and labels must share one device.")
        if not torch.equal(right_censored, ~event_observed):
            raise ValueError("Independent right-censor flags must complement observed flags.")
        if bool((event_observed & ((event_bin < 0) | (event_bin >= CTEQ_NUM_BINS))).any()):
            raise ValueError("Observed events require a bin in [0,24].")
        if bool((right_censored & (event_bin != -1)).any()):
            raise ValueError("Right-censored events require event_bin=-1.")

        gather_index = event_bin.clamp(0, CTEQ_NUM_BINS - 1).unsqueeze(-1)
        selected_log_mass = torch.gather(
            distribution.log_event_mass,
            dim=-1,
            index=gather_index,
        ).squeeze(-1)
        per_target_nll = -torch.where(
            event_observed,
            selected_log_mass,
            distribution.log_censor_mass,
        )
        event_target = F.one_hot(
            event_bin.clamp(0, CTEQ_NUM_BINS - 1),
            num_classes=CTEQ_NUM_BINS,
        ).to(logits.dtype)
        event_target = event_target * event_observed.unsqueeze(-1).to(logits.dtype)
        target = torch.cat(
            (event_target, right_censored.unsqueeze(-1).to(logits.dtype)),
            dim=-1,
        )
        probability = torch.cat(
            (distribution.event_mass, distribution.censor_mass.unsqueeze(-1)),
            dim=-1,
        )
        per_target_brier = torch.sum(torch.square(probability - target), dim=-1)
        return CteqTorchLoss(
            mean_nll=torch.mean(per_target_nll),
            mean_brier=torch.mean(per_target_brier),
            per_target_nll=per_target_nll,
            per_target_brier=per_target_brier,
        )


class CteqAdministrativeSurvivalLoss(nn.Module):
    """Independent TD/LO loss with target-specific censor boundaries.

    Future labels enter only this loss call.  Censored target ``m`` contributes
    survival through bins ``[0,m)``; an observed event in bin ``j`` contributes
    survival through ``[0,j)`` and the hazard at ``j``.  Ineligible zero-
    exposure targets are exactly zeroed and excluded from the denominator.
    """

    def forward(
        self,
        logits: torch.Tensor,
        event_bin: torch.Tensor,
        event_observed: torch.Tensor,
        censor_after_bin: torch.Tensor,
        loss_eligible: torch.Tensor,
    ) -> CteqAdministrativeTorchLoss:
        num_bins = 25
        touchdown = 0
        liftoff = 1
        distribution = cteq_distribution_from_logits(logits)
        if (
            event_bin.ndim != 3
            or event_bin.shape[0] != logits.shape[0]
            or event_bin.shape[1] != 2
            or event_bin.shape[2] != 2
            or event_bin.dtype != torch.long
        ):
            raise ValueError("event_bin must be int64 with shape [B,2,2].")
        if (
            event_observed.shape != event_bin.shape
            or event_observed.dtype != torch.bool
            or censor_after_bin.shape != event_bin.shape
            or censor_after_bin.dtype != torch.long
            or loss_eligible.shape != event_bin.shape
            or loss_eligible.dtype != torch.bool
        ):
            raise ValueError(
                "event_observed/loss_eligible must be bool and "
                "censor_after_bin int64 with shape [B,2,2]."
            )
        if (
            event_bin.device != logits.device
            or event_observed.device != logits.device
            or censor_after_bin.device != logits.device
            or loss_eligible.device != logits.device
        ):
            raise ValueError("CTEQ logits and administrative labels must share one device.")
        censored = ~event_observed
        if bool(
            (
                event_observed
                & ((event_bin < 0) | (event_bin >= num_bins))
            ).any()
        ):
            raise ValueError("Observed events require a bin in [0,24].")
        if bool((event_observed & (censor_after_bin != -1)).any()):
            raise ValueError("Observed events require censor_after_bin=-1.")
        if bool((censored & (event_bin != -1)).any()):
            raise ValueError("Censored targets require event_bin=-1.")
        if bool(
            (
                censored
                & (
                    (censor_after_bin < 0)
                    | (censor_after_bin > num_bins)
                )
            ).any()
        ):
            raise ValueError("Censored targets require censor_after_bin in [0,25].")
        expected_eligible = event_observed | (censored & (censor_after_bin > 0))
        if not torch.equal(loss_eligible, expected_eligible):
            raise ValueError(
                "loss_eligible must include every event and positive-exposure "
                "censor, while excluding zero exposure."
            )
        eligible_target_count = torch.sum(loss_eligible)
        if not torch.jit.is_tracing() and bool(eligible_target_count == 0):
            raise ValueError("Administrative-censor loss has zero eligible targets.")

        event_index = event_bin.clamp(0, num_bins - 1).unsqueeze(-1)
        event_log_mass = torch.gather(
            distribution.log_event_mass,
            dim=-1,
            index=event_index,
        ).squeeze(-1)
        boundary_index = censor_after_bin.clamp(0, num_bins).unsqueeze(-1)
        censor_log_survival = torch.gather(
            distribution.log_survival_at_boundary,
            dim=-1,
            index=boundary_index,
        ).squeeze(-1)
        raw_nll = -torch.where(
            event_observed,
            event_log_mass,
            censor_log_survival,
        )

        full_probability = torch.cat(
            (distribution.event_mass, distribution.censor_mass.unsqueeze(-1)),
            dim=-1,
        )
        event_target = F.one_hot(
            event_bin.clamp(0, num_bins - 1),
            num_classes=num_bins,
        ).to(logits.dtype)
        event_target = event_target * event_observed.unsqueeze(-1).to(logits.dtype)
        full_target = torch.cat(
            (
                event_target,
                torch.zeros_like(event_observed.unsqueeze(-1), dtype=logits.dtype),
            ),
            dim=-1,
        )
        event_brier = torch.sum(
            torch.square(full_probability - full_target), dim=-1
        )

        bin_index = torch.arange(num_bins, device=logits.device)
        prefix_mask = bin_index < censor_after_bin.unsqueeze(-1)
        prefix_event_mass = distribution.event_mass * prefix_mask.to(logits.dtype)
        censor_survival = torch.gather(
            distribution.survival_at_boundary,
            dim=-1,
            index=boundary_index,
        ).squeeze(-1)
        censor_brier = torch.sum(torch.square(prefix_event_mass), dim=-1) + (
            torch.square(censor_survival - 1.0)
        )
        raw_brier = torch.where(event_observed, event_brier, censor_brier)

        eligible_float = loss_eligible.to(logits.dtype)
        per_target_nll = raw_nll * eligible_float
        per_target_brier = raw_brier * eligible_float
        denominator = eligible_target_count.to(logits.dtype)
        per_role_eligible_count = torch.sum(loss_eligible, dim=0)
        per_role_nll_sum = torch.sum(per_target_nll, dim=0)
        per_role_brier_sum = torch.sum(per_target_brier, dim=0)
        td_event = loss_eligible[..., touchdown] & event_observed[..., touchdown]
        lo_event = loss_eligible[..., liftoff] & event_observed[..., liftoff]
        td_censored = loss_eligible[..., touchdown] & censored[..., touchdown]
        lo_censored = loss_eligible[..., liftoff] & censored[..., liftoff]
        return CteqAdministrativeTorchLoss(
            mean_nll=torch.sum(per_target_nll) / denominator,
            mean_brier=torch.sum(per_target_brier) / denominator,
            per_target_nll=per_target_nll,
            per_target_brier=per_target_brier,
            eligible_mask=loss_eligible,
            eligible_target_count=eligible_target_count,
            excluded_target_count=torch.sum(~loss_eligible),
            per_role_eligible_count=per_role_eligible_count,
            per_role_nll_sum=per_role_nll_sum,
            per_role_brier_sum=per_role_brier_sum,
            td_event_nll_sum=torch.sum(
                per_target_nll[..., touchdown][td_event]
            ),
            lo_event_nll_sum=torch.sum(
                per_target_nll[..., liftoff][lo_event]
            ),
            td_censored_nll_sum=torch.sum(
                per_target_nll[..., touchdown][td_censored]
            ),
            lo_censored_nll_sum=torch.sum(
                per_target_nll[..., liftoff][lo_censored]
            ),
            td_event_count=torch.sum(td_event),
            lo_event_count=torch.sum(lo_event),
            td_censored_count=torch.sum(td_censored),
            lo_censored_count=torch.sum(lo_censored),
        )


def cteq_role_time_weights_from_logits(
    logits: torch.Tensor,
    contact_state_now: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal ``current=c_now*S_LO`` and ``landing=pi_TD`` weights."""

    distribution = cteq_distribution_from_logits(logits)
    if (
        tuple(contact_state_now.shape) != (logits.shape[0], CTEQ_NUM_FEET)
        or contact_state_now.dtype != torch.bool
        or contact_state_now.device != logits.device
    ):
        raise ValueError("contact_state_now must be bool [B,2] on the logits device.")
    current = (
        contact_state_now.unsqueeze(-1).to(logits.dtype)
        * distribution.survival_before[:, :, CTEQ_LIFTOFF]
    )
    landing = distribution.event_mass[:, :, CTEQ_TOUCHDOWN]
    return current, landing


__all__ = [
    "CTEQ_NUM_BINS",
    "CteqAdministrativeSurvivalLoss",
    "CteqAdministrativeTorchLoss",
    "CteqDualEventHazardHead",
    "CteqIndependentSurvivalLoss",
    "CteqTorchDistribution",
    "CteqTorchLoss",
    "cteq_distribution_from_logits",
    "cteq_role_time_weights_from_logits",
]
