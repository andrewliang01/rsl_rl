from __future__ import annotations

import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


def _uses_amp(alg_cfg: dict) -> bool:
    return alg_cfg.get("amp_cfg") is not None


def _uses_multi_critic(alg_cfg: dict) -> bool:
    return alg_cfg.get("num_critics", 1) > 1


def select_ppo_algorithm_class(alg_cfg: dict):
    """Resolve the concrete PPO implementation from the composed algorithm config."""
    from .amp_ppo import AMPPPO
    from .multi_ppo import MultiPPO
    from .ppo import PPO

    if _uses_amp(alg_cfg):
        return AMPPPO
    if _uses_multi_critic(alg_cfg):
        return MultiPPO
    return PPO


def construct_ppo_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
    variant: str = "auto",
):
    """Construct the appropriate PPO variant from a composed config."""
    alg_cfg = cfg["algorithm"].copy()

    if variant == "auto":
        alg_class = select_ppo_algorithm_class(alg_cfg)
        if alg_class.__name__ == "AMPPPO":
            return _construct_amp_algorithm(obs, env, cfg, device)
        if alg_class.__name__ == "MultiPPO":
            return _construct_multi_algorithm(obs, env, cfg, device)
        return _construct_standard_algorithm(obs, env, cfg, device)
    if variant == "amp":
        return _construct_amp_algorithm(obs, env, cfg, device)
    if variant == "multi":
        return _construct_multi_algorithm(obs, env, cfg, device)
    if variant == "ppo":
        return _construct_standard_algorithm(obs, env, cfg, device)
    raise ValueError(f"Unknown PPO construction variant: {variant}")


def _construct_standard_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str):
    from .ppo import PPO

    return _construct_single_critic_algorithm(PPO, obs, env, cfg, device, include_amp_obs=False)


def _construct_amp_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str):
    from .amp_ppo import AMPPPO

    return _construct_multi_algorithm(obs, env, cfg, device, alg_class=AMPPPO, include_amp_obs=True)


def _construct_single_critic_algorithm(
    alg_class,
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
    *,
    include_amp_obs: bool,
):
    alg_cfg = cfg["algorithm"].copy()
    actor_cfg = cfg["actor"].copy()
    critic_cfg = cfg["critic"].copy()

    alg_cfg.pop("class_name", None)

    actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name", "MLPModel"))
    critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name", "MLPModel"))

    default_sets = ["actor", "critic"]
    if include_amp_obs:
        default_sets.append("amp")
    if alg_cfg.get("rnd_cfg") is not None:
        default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups"), default_sets)

    alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)

    # These knobs belong to optional PPO features. Keep them on the shared config
    # object, but strip anything that the single-critic constructor does not consume.
    alg_cfg.pop("num_critics", None)
    alg_cfg.pop("reward_group_names", None)
    alg_cfg.pop("reward_group_weights", None)
    alg_cfg.pop("shared_critic", None)
    if not include_amp_obs:
        alg_cfg.pop("amp_cfg", None)

    actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
    print(f"Actor Model: {actor}")
    if alg_cfg.pop("share_cnn_encoders", None):
        critic_cfg["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    return alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg.get("multi_gpu"))


def _construct_multi_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
    *,
    alg_class=None,
    include_amp_obs: bool = False,
):
    if alg_class is None:
        from .multi_ppo import MultiPPO

        alg_class = MultiPPO

    alg_cfg = cfg["algorithm"].copy()
    actor_cfg = cfg["actor"].copy()

    alg_cfg.pop("class_name", None)
    share_cnn_encoders = alg_cfg.pop("share_cnn_encoders", False)
    if not include_amp_obs:
        alg_cfg.pop("amp_cfg", None)

    amp_cfg = alg_cfg.get("amp_cfg")
    num_critics = alg_cfg.pop("num_critics", 1)
    reward_group_names = alg_cfg.pop("reward_group_names", None)
    reward_group_weights = alg_cfg.pop("reward_group_weights", None)
    shared_critic = alg_cfg.pop("shared_critic", False)

    use_amp_multi_critic = include_amp_obs and amp_cfg is not None and num_critics > 1
    if reward_group_names is None:
        reward_group_names = [f"critic_{i}" for i in range(num_critics)]
    else:
        reward_group_names = list(reward_group_names)
    if reward_group_weights is None:
        reward_group_weights = [1.0] * num_critics
    else:
        reward_group_weights = list(reward_group_weights)

    effective_num_critics = num_critics
    effective_reward_group_names = list(reward_group_names)
    effective_reward_group_weights = list(reward_group_weights)

    actor_class = resolve_callable(actor_cfg.pop("class_name", "MLPModel"))

    def resolve_critic_cfg(group_name: str | None = None) -> tuple[type[MLPModel], dict]:
        critic_key = f"critic_{group_name}" if group_name else "critic"
        if critic_key in cfg:
            critic_cfg = cfg[critic_key].copy()
        elif "critic" in cfg:
            critic_cfg = cfg["critic"].copy()
        else:
            raise KeyError(
                f"Missing critic config. Expected '{critic_key}'"
                + (" or 'critic'." if critic_key != "critic" else ".")
            )
        critic_class = resolve_callable(critic_cfg.pop("class_name", "MLPModel"))
        return critic_class, critic_cfg

    default_sets = ["actor", "critic"]
    if include_amp_obs:
        default_sets.append("amp")
    if alg_cfg.get("rnd_cfg") is not None:
        default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups"), default_sets)

    alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)

    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
    print(f"Actor Model: {actor}")

    if effective_num_critics > 1 and shared_critic:
        critic_class, critic_cfg = resolve_critic_cfg()
        critic_cfg.setdefault("num_heads", effective_num_critics)
        critic = critic_class(obs, cfg["obs_groups"], "critic", effective_num_critics, **critic_cfg).to(device)
        print(f"Created shared multi-head critic for groups: {effective_reward_group_names}")
    elif effective_num_critics > 1:
        critics = nn.ModuleList()
        for i in range(effective_num_critics):
            group_name = effective_reward_group_names[i]
            critic_class, critic_cfg = resolve_critic_cfg(group_name)
            critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
            critics.append(critic)
        print(f"Created {effective_num_critics} critics for multi-critic training: {effective_reward_group_names}")
        critic = critics
    else:
        critic_class, critic_cfg = resolve_critic_cfg()
        if share_cnn_encoders:
            critic_cfg["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

    storage = RolloutStorage(
        "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device, num_critics=effective_num_critics
    )

    return alg_class(
        actor,
        critic,
        storage,
        num_critics=effective_num_critics,
        reward_group_names=effective_reward_group_names,
        reward_group_weights=effective_reward_group_weights,
        shared_critic=shared_critic,
        device=device,
        **alg_cfg,
        multi_gpu_cfg=cfg.get("multi_gpu"),
    )
