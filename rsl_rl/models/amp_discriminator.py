# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd

from rsl_rl.modules import MLP, EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """Adversarial Motion Prior (AMP) Discriminator.

    This discriminator is used to provide style rewards for motion imitation tasks.
    It distinguishes between expert motion data and policy-generated motion data.

    Args:
        input_dim: Dimension of the input feature vector (concatenated state and next state).
        hidden_dims: Hidden layer dimensions for the discriminator MLP.
        activation: Activation function for the MLP.
        amp_reward_coef: Coefficient to scale the AMP reward.
        normalize: Whether to normalize inputs using EmpiricalNormalization.
        task_reward_lerp: Interpolation factor between AMP reward and task reward.
            0.0 = only AMP reward, 1.0 = only task reward.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        amp_reward_coef: float = 1.0,
        normalize: bool = False,
        task_reward_lerp: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        # Build discriminator network: MLP trunk + output layer
        self.trunk = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Observation normalizer (optional)
        self.normalize = normalize
        if normalize:
            # Normalize single state (input_dim is state + next_state)
            self.obs_normalizer = EmpiricalNormalization(input_dim // 2)
        else:
            self.obs_normalizer = nn.Identity()

        # Initialize output layer with small weights for stable training
        nn.init.uniform_(self.output_layer.weight, -0.01, 0.01)
        nn.init.uniform_(self.output_layer.bias, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            Discriminator logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.output_layer(h)
        return d

    def compute_grad_pen(
        self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
    ) -> torch.Tensor:
        """Compute gradient penalty for expert data (regularization).

        This penalizes the gradient norm to enforce Lipschitz constraint.

        Args:
            expert_state: Expert state samples.
            expert_next_state: Expert next state samples.
            lambda_: Gradient penalty coefficient.

        Returns:
            Scalar gradient penalty loss.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)

        disc = self.forward(expert_data)
        ones = torch.ones_like(disc)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Penalize deviation from zero gradient norm
        grad_pen = lambda_ * grad.norm(2, dim=1).pow(2).mean()
        return grad_pen


    def compute_grad_pen_from_disc(
        self, expert_data: torch.Tensor, disc: torch.Tensor, lambda_: float = 10.0
    ) -> torch.Tensor:
        """Compute gradient penalty from an existing discriminator forward pass."""
        ones = torch.ones_like(disc)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(
        self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Predict AMP reward from state transitions.

        Args:
            state: Current state tensor.
            next_state: Next state tensor.
            task_reward: Optional task reward for interpolation.

        Returns:
            Tuple containing:
                - final_reward: Combined reward after interpolation.
                - style_reward: Raw AMP style reward.
                - disc_logits: Raw discriminator output.
                - disc_reward: Discriminator component of final reward.
                - task_reward_component: Task component of final reward.
        """
        with torch.no_grad():
            self.eval()

            # Normalize if enabled
            if self.normalize:
                state = self.obs_normalizer(state)
                next_state = self.obs_normalizer(next_state)

            # Compute discriminator output
            disc_input = torch.cat([state, next_state], dim=-1)
            d = self.forward(disc_input)

            # Compute AMP style reward: r = -log(1 - sigmoid(D)) ≈ clamp(1 - 0.25 * (D-1)^2)
            # This formulation encourages the policy to match expert data
            style_reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)

            # Interpolate with task reward if specified
            if self.task_reward_lerp > 0 and task_reward is not None:
                final_reward, disc_r, task_r = self._lerp_reward(style_reward, task_reward.unsqueeze(-1))
            else:
                final_reward = style_reward
                disc_r = style_reward
                task_r = torch.zeros_like(style_reward)

            self.train()

        return final_reward.squeeze(), style_reward.squeeze(), d, disc_r.squeeze(), task_r.squeeze()

    def _lerp_reward(
        self, disc_r: torch.Tensor, task_r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linearly interpolate between discriminator and task rewards.

        Args:
            disc_r: Discriminator reward.
            task_r: Task reward.

        Returns:
            Tuple of (interpolated reward, disc component, task component).
        """
        final_r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        disc_component = (1.0 - self.task_reward_lerp) * disc_r
        task_component = self.task_reward_lerp * task_r
        return final_r, disc_component, task_component

    def update_normalization(self, obs: torch.Tensor) -> None:
        """Update observation normalizer with new observations.

        Args:
            obs: Observations to update with (single state, not concatenated).
        """
        if self.normalize:
            self.obs_normalizer.update(obs)

    def get_stats(self) -> dict[str, float]:
        """Return statistics for logging."""
        stats = {}
        if self.normalize and hasattr(self.obs_normalizer, 'mean'):
            stats['amp_obs_mean'] = self.obs_normalizer.mean.mean().item()
            stats['amp_obs_std'] = self.obs_normalizer.std.mean().item()
        return stats
