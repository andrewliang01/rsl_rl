import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.env import VecEnv

class DWAQ(torch.nn.Module):
    # TODO: DWAQ网络的构建
    # 两个loss
    # 
    def __init__(
        self,
        obs_groups: dict[str, list[str]],
        num_states: int, # actor的obs的维度
        num_target_states: int,
        input_obs_set: str = "actor",
        obs_noise_free_group: str = "policy_noise_free",
        vel_obs_group: str = "vel",
        code_append_obs_group: str = "policy",
        encoder_hidden_dims: tuple[int] | list[int] = [128, 64], # 第三层需要自己画
        decoder_hidden_dims: tuple[int] | list[int] = [64, 128],
        num_history_len: int = 5,
        num_latent: int = 19,  # 隐向量的长度 3(vel) + 16(z_t)
        num_decode: int | None = None,
        activation: str = "elu",
        VAE_beta: int = 1.0,
        use_adaboot: bool = False,
        adaboot_eps: float = 1.0e-8,
        state_normalization: bool = True,
        vel_loss_coef: float = 1.0,
        obs_loss_coef: float = 1.0,
        device: str = "cuda",
        **kwargs: dict[str, Any],
    ) -> None:
        """
        DWAQ module.

        """
        if kwargs:
            print(
                "DWAQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        self.beta = VAE_beta
        # 是否使用adaboot
        self.use_adaboot = use_adaboot
        self.adaboot_eps = adaboot_eps
        self.last_adaboot_probability: torch.Tensor | None = None
        self.num_history_len = num_history_len
        self.num_latent = num_latent
        # Get the observation dimensions
        self.obs_groups = obs_groups
        self.input_obs_set = input_obs_set
        self.obs_noise_free_group = obs_noise_free_group
        self.vel_obs_group = vel_obs_group
        self.code_append_obs_group = code_append_obs_group
        self.state_normalization = state_normalization
        self.vel_loss_coef = vel_loss_coef
        self.obs_loss_coef = obs_loss_coef
        self.device = device

        # 单帧obs长度
        if num_states % num_history_len != 0:
            raise ValueError(
                f"DWAQ encoder input dim ({num_states}) must be divisible by num_history_len ({num_history_len})."
            )
        self.obs_one_frame_len: int = int(num_states / num_history_len)
        # 记录decoder输出的维度
        self.num_decoder = num_target_states if num_decode is None else num_decode
        if self.num_decoder > num_target_states:
            raise ValueError(
                f"num_decode ({self.num_decoder}) can not be larger than DWAQ target obs dimension "
                f"({num_target_states})."
            )

        if state_normalization:
            if self.num_decoder > self.obs_one_frame_len:
                raise ValueError(
                    f"num_decode ({self.num_decoder}) can not be larger than the DWAQ one-frame obs dimension "
                    f"({self.obs_one_frame_len}) when state_normalization=True."
                )
            self.state_normalizer = EmpiricalNormalization(shape=[num_states], until=int(1.0e8)).to(self.device)
        else:
            self.state_normalizer = torch.nn.Identity()

        self.encoder_backbone = MLP(
            input_dim=num_states,
            output_dim=encoder_hidden_dims[-1],
            hidden_dims=encoder_hidden_dims[:-1],
            activation=activation,
            last_activation="elu",
        ).to(device)

        self.encoder_latent_mean = torch.nn.Linear(encoder_hidden_dims[-1], num_latent-3).to(device) # 隐向量均值层
        self.encoder_latent_logvar = torch.nn.Linear(encoder_hidden_dims[-1], num_latent-3).to(device) # 隐向量方差层
        self.encoder_vel_mean = torch.nn.Linear(encoder_hidden_dims[-1], 3).to(device) # 速度的均值层
        self.encoder_vel_logvar = torch.nn.Linear(encoder_hidden_dims[-1], 3).to(device) # 速度的方差层

        print(f"Encoder backbone MLP: {self.encoder_backbone}")
        print(f"Encoder latent mean: {self.encoder_latent_mean}")
        print(f"Encoder latent logvar: {self.encoder_latent_logvar}")
        print(f"Encoder velocity mean: {self.encoder_vel_mean}")
        print(f"Encoder velocity logvar: {self.encoder_vel_logvar}")

        self.decoder = MLP(
            input_dim=num_latent,
            output_dim=self.num_decoder,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        ).to(device)
        print(f"Decoder MLP: {self.decoder}")

    def encoder_forward(self, obs_history):
        """CENet 前向推理

        Args:
            obs_history (_type_): 历史观测值

        Returns:
            _type_: _description_
        """
        x = self.encoder_backbone(obs_history)
        latent_mean = self.encoder_latent_mean(x)
        latent_logvar = self.encoder_latent_logvar(x)
        vel_mean = self.encoder_vel_mean(x)
        vel_logvar = self.encoder_vel_logvar(x)
        
        # 对数方差限制在一定范围内，避免过大
        latent_logvar = torch.clip(latent_logvar,min=-10,max=10)
        vel_logvar = torch.clip(vel_logvar,min=-10,max=10)
        # 重参数化得到隐向量和速度的采样值
        latent_sample = self.reparameterize(latent_mean, latent_logvar)
        vel_sample = self.reparameterize(vel_mean, vel_logvar)

        # 将速度和隐向量拼接起来
        code = torch.cat((vel_sample,latent_sample),dim=-1)
        # 解码得到下一时刻观测值
        decode = self.decoder(code)

        return code, latent_sample, vel_sample, vel_mean, vel_logvar, latent_mean, latent_logvar,decode  

    def reparameterize(self,mean,logvar):
        """重参数化

        Args:
            mean (_type_): 均值
            logvar (_type_): 对数方差

        Returns:
            _type_: 隐向量
        """
        std = torch.exp(logvar*0.5) # 得到标准差
        code_temp = torch.randn_like(std)
        code = mean + std * code_temp
        return code

    def compute_loss(self, observations: TensorDict, next_observations: TensorDict) -> dict[str, torch.Tensor]:
        input_obs = self.get_input_obs(observations)
        next_obs_noise_free = self.get_obs_noise_free(next_observations)
        if self.state_normalization:
            input_obs = self.normalize_input_obs(input_obs)
            next_obs_noise_free = self.normalize_single_frame_obs(next_obs_noise_free)
        next_obs_noise_free = next_obs_noise_free[..., : self.num_decoder]
        vel_obs = self.get_vel_obs(observations)

        _, _, vel_sample, _, _, latent_mean, latent_logvar, decode = self.encoder_forward(input_obs)
        if next_obs_noise_free.shape[-1] != self.num_decoder:
            raise ValueError(
                f"DWAQ decoder output dim ({self.num_decoder}) must match obs_noise_free dim "
                f"({next_obs_noise_free.shape[-1]})."
            )

        if vel_obs.shape[-1] != 3:
            raise ValueError(f"DWAQ velocity target dim must be 3, got {vel_obs.shape[-1]}.")
        vel_target = vel_obs.detach()
        obs_target = next_obs_noise_free.detach()

        vel_loss = nn.functional.mse_loss(vel_sample, vel_target) * self.vel_loss_coef
        obs_loss = nn.functional.mse_loss(decode, obs_target) * self.obs_loss_coef
        dkl_loss = -0.5 * torch.mean(torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp(), dim=1))
        total_loss = vel_loss + obs_loss + self.beta * dkl_loss

        return {
            "total": total_loss,
            "vel": vel_loss,
            "obs": obs_loss,
            "dkl": dkl_loss,
        }

    def get_code(
        self,
        observations: TensorDict,
        deterministic: bool = True,
        bootstrap_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_obs = self.get_input_obs(observations)
        return self.get_code_from_input_obs(
            input_obs,
            observations,
            deterministic=deterministic,
            bootstrap_rewards=bootstrap_rewards,
        )

    def get_code_from_input_obs(
        self,
        input_obs: torch.Tensor,
        observations: TensorDict,
        deterministic: bool = True,
        bootstrap_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.state_normalization:
            input_obs = self.normalize_input_obs(input_obs)
        (
            code,
            latent_sample,
            vel_sample,
            vel_mean,
            _vel_logvar,
            latent_mean,
            _latent_logvar,
            _decode,
        ) = self.encoder_forward(input_obs)
        if deterministic:
            if self.use_adaboot and bootstrap_rewards is not None:
                return self.apply_adaboot(observations, vel_mean, latent_mean, bootstrap_rewards)
            return torch.cat((vel_mean, latent_mean), dim=-1)
        if self.use_adaboot and bootstrap_rewards is not None:
            return self.apply_adaboot(observations, vel_sample, latent_sample, bootstrap_rewards)
        return code

    def get_actor_observation(
        self,
        observations: TensorDict,
        deterministic: bool = True,
        bootstrap_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_obs = self.get_input_obs(observations)
        if self.state_normalization:
            normalized_input_obs = self.normalize_input_obs(input_obs)
        else:
            normalized_input_obs = input_obs
        current_obs = normalized_input_obs[:, -self.obs_one_frame_len :]
        code = self.get_code_from_input_obs(
            input_obs,
            observations,
            deterministic=deterministic,
            bootstrap_rewards=bootstrap_rewards,
        )
        return torch.cat((current_obs, code), dim=-1)

    def apply_adaboot(
        self,
        observations: TensorDict,
        estimated_vel: torch.Tensor,
        latent: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptively choose estimated velocity or simulator velocity for the actor code."""
        if estimated_vel.shape[-1] != 3:
            raise ValueError(f"DWAQ estimated velocity dim must be 3, got {estimated_vel.shape[-1]}.")
        real_vel = self.get_vel_obs(observations)
        if real_vel.shape[-1] != 3:
            raise ValueError(f"DWAQ AdaBoot velocity target dim must be 3, got {real_vel.shape[-1]}.")

        flat_rewards = rewards.float().reshape(-1)
        cv_rewards = torch.std(flat_rewards, unbiased=False) / (torch.abs(torch.mean(flat_rewards)) + self.adaboot_eps)
        p_boot = torch.clamp(1.0 - torch.tanh(cv_rewards), min=0.0, max=1.0)
        self.last_adaboot_probability = p_boot.detach()

        use_estimated = torch.rand((), device=estimated_vel.device) < p_boot
        selected_vel = torch.where(use_estimated, estimated_vel, real_vel)
        return torch.cat((selected_vel.detach(), latent.detach()), dim=-1)

    def update(
        self,
        observations: TensorDict,
        next_observations: TensorDict,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
    ) -> dict[str, float]:
        loss = self.compute_loss(observations, next_observations)
        optimizer.zero_grad()
        loss["total"].backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()
        return {key: value.item() for key, value in loss.items()}



    def _get_obs_by_set(self, obs: TensorDict, obs_set: str) -> torch.Tensor:
        if obs_set not in self.obs_groups:
            raise KeyError(f"Observation set '{obs_set}' is not defined in obs_groups: {list(self.obs_groups.keys())}")
        obs_list = [obs[obs_group] for obs_group in self.obs_groups[obs_set]]
        return torch.cat(obs_list, dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_obs_by_set(obs, "actor")

    def get_input_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_obs_by_set(obs, self.input_obs_set)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_obs_by_set(obs, "critic")

    def get_obs_noise_free(self, obs: TensorDict) -> torch.Tensor:
        return obs[self.obs_noise_free_group]

    def get_vel_obs(self, obs: TensorDict) -> torch.Tensor:
        return obs[self.vel_obs_group]

    def normalize_input_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.state_normalization:
            return obs
        return self.state_normalizer(obs)  # type: ignore[operator]

    def normalize_single_frame_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.state_normalization:
            return obs
        mean = self.state_normalizer._mean[..., -obs.shape[-1] :]  # type: ignore[attr-defined]
        std = self.state_normalizer._std[..., -obs.shape[-1] :]  # type: ignore[attr-defined]
        return (obs - mean) / (std + self.state_normalizer.eps)  # type: ignore[attr-defined]

    def update_normalization(self, obs: TensorDict, next_obs: TensorDict | None = None) -> None:
        if self.state_normalization:
            self.state_normalizer.update(self.get_input_obs(obs))  # type: ignore[attr-defined]


def resolve_dwaq_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]], env: VecEnv) -> dict:
    """Resolve the DWAQ configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs: Observation dictionary.
        obs_groups: Observation groups dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # Resolve dimension of dwaq gated state
    if "dwaq_cfg" in alg_cfg and alg_cfg["dwaq_cfg"] is not None:
        input_obs_set = alg_cfg["dwaq_cfg"].get("input_obs_set", "actor")
        obs_noise_free_group = alg_cfg["dwaq_cfg"].get("obs_noise_free_group", "policy_noise_free")
        vel_obs_group = alg_cfg["dwaq_cfg"].get("vel_obs_group", "vel")
        # Get dimension of dwaq encoder input state
        num_states = 0
        for obs_group in obs_groups[input_obs_set]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The DWAQ module only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
            num_states += obs[obs_group].shape[-1]

        for obs_group in (obs_noise_free_group, vel_obs_group):
            if obs_group not in obs:
                raise ValueError(
                    f"DWAQ requires observation group '{obs_group}', but it was not found. "
                    f"Available observations: {list(obs.keys())}"
                )
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The DWAQ module only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
        num_target_states = obs[obs_noise_free_group].shape[-1]
        # Add dwaq gated state to config

        alg_cfg["dwaq_cfg"]["num_states"] = num_states
        alg_cfg["dwaq_cfg"]["num_target_states"] = num_target_states
        alg_cfg["dwaq_cfg"]["obs_groups"] = obs_groups
    else:
        alg_cfg["dwaq_cfg"] = None
    return alg_cfg  
