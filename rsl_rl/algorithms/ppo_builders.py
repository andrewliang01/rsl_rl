from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_dwaq_config, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


def construct_single_critic_algorithm(
    alg_class,
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
    *,
    include_amp_obs: bool = False,
):
    """Construct a PPO-family algorithm with one critic."""
    alg_cfg = cfg["algorithm"].copy()
    actor_cfg = cfg["actor"].copy()
    critic_cfg = cfg["critic"].copy()

    alg_cfg.pop("class_name", None)
    alg_cfg.pop("num_critics", None)
    alg_cfg.pop("reward_group_names", None)
    alg_cfg.pop("reward_group_weights", None)
    alg_cfg.pop("shared_critic", None)
    if not include_amp_obs:
        alg_cfg.pop("amp_cfg", None)

    actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name", "MLPModel"))
    critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name", "MLPModel"))

    obs_groups = _resolve_ppo_obs_groups(obs, env, cfg, alg_cfg, include_amp_obs=include_amp_obs)
    alg_cfg = resolve_dwaq_config(alg_cfg, obs, obs_groups, env)
    alg_cfg = resolve_rnd_config(alg_cfg, obs, obs_groups, env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)

    actor_obs = _augment_actor_obs_sample_for_dwaq(obs, alg_cfg)
    actor: MLPModel = actor_class(actor_obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)
    print(f"Actor Model: {actor}")
    if alg_cfg.pop("share_cnn_encoders", None):
        critic_cfg["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic: MLPModel = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
    return alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg.get("multi_gpu"))


def construct_multi_critic_algorithm(
    alg_class,
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
    *,
    include_amp_obs: bool = False,
):
    """Construct a PPO-family algorithm with multi-critic configuration."""
    alg_cfg = cfg["algorithm"].copy()
    actor_cfg = cfg["actor"].copy()

    alg_cfg.pop("class_name", None)
    share_cnn_encoders = alg_cfg.pop("share_cnn_encoders", False)
    if not include_amp_obs:
        alg_cfg.pop("amp_cfg", None)

    num_critics = alg_cfg.pop("num_critics", 1)
    reward_group_names = _as_list_or_default(
        alg_cfg.pop("reward_group_names", None), [f"critic_{i}" for i in range(num_critics)]
    )
    reward_group_weights = _as_list_or_default(alg_cfg.pop("reward_group_weights", None), [1.0] * num_critics)
    shared_critic = alg_cfg.pop("shared_critic", False)

    actor_class = resolve_callable(actor_cfg.pop("class_name", "MLPModel"))
    obs_groups = _resolve_ppo_obs_groups(obs, env, cfg, alg_cfg, include_amp_obs=include_amp_obs)
    alg_cfg = resolve_dwaq_config(alg_cfg, obs, obs_groups, env)
    alg_cfg = resolve_rnd_config(alg_cfg, obs, obs_groups, env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)

    actor_obs = _augment_actor_obs_sample_for_dwaq(obs, alg_cfg)
    actor = actor_class(actor_obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)
    print(f"Actor Model: {actor}")

    if num_critics > 1 and shared_critic:
        critic_class, critic_cfg = _resolve_critic_cfg(cfg)
        critic_cfg.setdefault("num_heads", num_critics)
        critic = critic_class(obs, obs_groups, "critic", num_critics, **critic_cfg).to(device)
        print(f"Created shared multi-head critic for groups: {reward_group_names}")
    elif num_critics > 1:
        critics = nn.ModuleList()
        for group_name in reward_group_names:
            critic_class, critic_cfg = _resolve_critic_cfg(cfg, group_name)
            critics.append(critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device))
        critic = critics
        print(f"Created {num_critics} critics for multi-critic training: {reward_group_names}")
    else:
        critic_class, critic_cfg = _resolve_critic_cfg(cfg)
        if share_cnn_encoders:
            critic_cfg["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

    storage = RolloutStorage(
        "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device, num_critics=num_critics
    )
    return alg_class(
        actor,
        critic,
        storage,
        num_critics=num_critics,
        reward_group_names=reward_group_names,
        reward_group_weights=reward_group_weights,
        shared_critic=shared_critic,
        device=device,
        **alg_cfg,
        multi_gpu_cfg=cfg.get("multi_gpu"),
    )


def _resolve_ppo_obs_groups(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    alg_cfg: dict,
    *,
    include_amp_obs: bool,
) -> dict:
    default_sets = ["actor", "critic"]
    if include_amp_obs:
        default_sets.append("amp")
    if alg_cfg.get("rnd_cfg") is not None:
        default_sets.append("rnd_state")
    if alg_cfg.get("dwaq_cfg") is not None:
        default_sets.append(alg_cfg["dwaq_cfg"].get("input_obs_set", "actor"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups"), default_sets)
    return cfg["obs_groups"]


def _augment_actor_obs_sample_for_dwaq(obs: TensorDict, alg_cfg: dict) -> TensorDict:
    dwaq_cfg = alg_cfg.get("dwaq_cfg")
    if dwaq_cfg is None:
        return obs

    code_group = dwaq_cfg.get("code_append_obs_group", "policy")
    if code_group not in obs:
        raise ValueError(
            f"DWAQ code_append_obs_group '{code_group}' was not found in observations. "
            f"Available observations: {list(obs.keys())}"
        )

    code_dim = int(dwaq_cfg.get("num_latent", 19))
    obs_for_actor = obs.clone()
    code_template = torch.zeros(
        *obs[code_group].shape[:-1],
        code_dim,
        dtype=obs[code_group].dtype,
        device=obs[code_group].device,
    )
    obs_for_actor[code_group] = torch.cat((obs[code_group], code_template), dim=-1)
    return obs_for_actor


def _resolve_critic_cfg(cfg: dict, group_name: str | None = None) -> tuple[type[MLPModel], dict]:
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


def _as_list_or_default(value, default: list):
    if value is None:
        return list(default)
    return list(value)
