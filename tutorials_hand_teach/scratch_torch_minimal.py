from __future__ import annotations

import torch
import torch.nn as nn


class TinyActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(32, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.backbone(obs)
        mean = self.mean_head(latent)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class TinyCritic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main() -> None:
    torch.manual_seed(3)

    batch_size = 8
    obs_dim = 10
    action_dim = 3

    obs = torch.randn(batch_size, obs_dim)
    fake_returns = torch.randn(batch_size, 1)
    fake_advantages = torch.randn(batch_size)

    actor = TinyActor(obs_dim, action_dim)
    critic = TinyCritic(obs_dim)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=1e-3,
    )

    with torch.no_grad():
        old_mean, old_std = actor(obs)
        old_dist = torch.distributions.Normal(old_mean, old_std)
        actions = old_dist.sample()
        old_log_prob = old_dist.log_prob(actions).sum(dim=-1)

    mean, std = actor(obs)
    dist = torch.distributions.Normal(mean, std)
    new_log_prob = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()
    values = critic(obs)

    ratio = torch.exp(new_log_prob - old_log_prob)
    clip_param = 0.2
    surrogate = -fake_advantages * ratio
    surrogate_clipped = -fake_advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    policy_loss = torch.max(surrogate, surrogate_clipped).mean()
    value_loss = (values - fake_returns).pow(2).mean()
    loss = policy_loss + value_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("obs:", obs.shape)
    print("actions:", actions.shape)
    print("old_log_prob:", old_log_prob.shape)
    print("ratio:", ratio.shape, "mean =", ratio.mean().item())
    print("values:", values.shape)
    print("policy_loss:", policy_loss.item())
    print("value_loss:", value_loss.item())
    print("total_loss:", loss.item())
    print("actor params updated by optimizer:", sum(p.numel() for p in actor.parameters()))
    print("critic params updated by optimizer:", sum(p.numel() for p in critic.parameters()))


if __name__ == "__main__":
    main()
