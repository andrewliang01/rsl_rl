# 00. Repo Map

先把 rsl_rl 想成一条流水线:

```text
env -> runner -> algorithm -> model -> storage -> optimizer
```

每个文件的责任:

`rsl_rl/runners/on_policy_runner.py`

这是训练主循环。它不应该懂 PPO 数学细节, 只负责:

- 创建算法对象。
- 每轮 rollout 调 `alg.act()`。
- 把 env 返回的 `obs/rewards/dones/extras` 交给 `alg.process_env_step()`。
- rollout 后调用 `alg.compute_returns()`。
- 最后调用 `alg.update()`。

`rsl_rl/algorithms/ppo.py`

标准 PPO。你学习 rsl_rl 的第一主线。

核心函数:

- `__init__`: 接住 actor/critic/storage, 创建 optimizer, 保存超参数。
- `act`: 用 actor 采样 action, 用 critic 估 value, 临时存在 `self.transition`。
- `process_env_step`: env step 之后补齐 reward/done, 把 transition 写进 storage。
- `compute_returns`: 用 GAE 算 return 和 advantage。
- `update`: 从 storage 取 mini-batch, 算 PPO loss, backward, optimizer step。

`rsl_rl/storage/rollout_storage.py`

这不是普通 replay buffer。PPO 是 on-policy, 所以它只存一段刚采集的 rollout。

你要记住三个 shape:

```text
T = num_steps_per_env
N = num_envs
A = num_actions
```

常见 shape:

```text
observations: [T, N, obs_dim]
actions:      [T, N, A]
rewards:      [T, N, num_critics]
values:       [T, N, num_critics]
returns:      [T, N, num_critics]
advantages:   [T, N, num_critics]
```

`rsl_rl/models/mlp_model.py`

actor 和 critic 都可以是 `MLPModel`。区别不是类不同, 而是:

- actor 有 `distribution_cfg`, 输出 action distribution。
- critic 没有 `distribution_cfg`, 输出 value。

actor 调用:

```python
actions = actor(obs, stochastic_output=True)
log_prob = actor.get_output_log_prob(actions)
```

critic 调用:

```python
values = critic(obs)
```

`rsl_rl/algorithms/ppo_builders.py`

构造 helper。它不负责 PPO 数学, 只负责:

- 从 cfg 构造 actor。
- 从 cfg 构造 critic 或 critics。
- 构造 storage。
- 解析 obs groups。

`rsl_rl/algorithms/multi_ppo.py`

MultiPPO 只应该管 multi-critic:

- 多 value head 或多个 critic。
- 多 reward group。
- 每个 critic 单独 GAE。
- actor 用 weighted advantage。

`rsl_rl/algorithms/amp_ppo.py`

AMP-PPO 只应该管 AMP:

- discriminator。
- style reward。
- final reward。
- AMP replay buffer。
- discriminator update。

第一条铁律:

```text
model 管 forward, algorithm 管训练, storage 管数据, runner 管流程。
```
