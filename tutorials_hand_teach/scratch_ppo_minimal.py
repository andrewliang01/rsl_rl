from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


def make_obs(num_envs: int, obs_dim: int) -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(num_envs, obs_dim)},
        batch_size=[num_envs],
    )


def build_ppo(num_envs: int, num_steps: int, obs_dim: int, num_actions: int) -> tuple[PPO, TensorDict]:
    obs = make_obs(num_envs, obs_dim)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        num_actions,
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = MLPModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], device="cpu")
    ppo = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        schedule="fixed",
        device="cpu",
    )
    return ppo, obs


def main() -> None:
    torch.manual_seed(7)

    num_envs = 4
    num_steps = 8
    obs_dim = 10
    num_actions = 3

    ppo, obs = build_ppo(num_envs, num_steps, obs_dim, num_actions)
    print("obs['policy']:", obs["policy"].shape)

    for step in range(num_steps):
        actions = ppo.act(obs)
        next_obs = make_obs(num_envs, obs_dim)
        rewards = torch.randn(num_envs)
        dones = torch.zeros(num_envs)
        if step == num_steps - 1:
            dones[0] = 1.0

        ppo.process_env_step(next_obs, rewards, dones, extras={})
        obs = next_obs

        if step == 0:
            print("actions:", actions.shape)
            print("storage.actions:", ppo.storage.actions.shape)
            print("storage.rewards:", ppo.storage.rewards.shape)
            print("storage.values:", ppo.storage.values.shape)

    ppo.compute_returns(obs)
    print("returns:", ppo.storage.returns.shape)
    print("advantages:", ppo.storage.advantages.shape)
    print("first return row:", ppo.storage.returns[:, 0, 0])

    losses = ppo.update()
    print("losses:", losses)


if __name__ == "__main__":
    main()
