# 03. Line By Line PPO

这份导读只盯 `rsl_rl/rsl_rl/algorithms/ppo.py` 主线。

## __init__

你读 `__init__` 时, 不要被参数吓到。它做四件事:

1. 保存 device / distributed 信息。
2. 创建 RND / symmetry 这些可选模块。
3. 保存 actor、critic、storage。
4. 创建 optimizer 和 PPO 超参数。

关键代码:

```python
self.actor = actor.to(self.device)
self.critic = critic.to(self.device)
```

含义: 网络搬到 CPU/GPU。

```python
self.optimizer = resolve_optimizer(optimizer)(
    chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
)
```

含义: optimizer 会同时更新 actor 和 critic。

你要问:

```text
哪些参数会被更新?
答案: actor.parameters() + critic.parameters()
```

## act

`act` 是 rollout 阶段的前半段。

```python
self.transition.actions = self.actor(obs, stochastic_output=True).detach()
```

actor 根据 obs 采样 action。`detach()` 表示 rollout 不训练。

```python
self.transition.values = self.critic(obs).detach()
```

critic 估当前状态价值。

```python
self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
```

存旧策略下 action 的 log probability。

```python
self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
```

存旧策略 distribution 参数, 后面算 KL 用。

```python
self.transition.observations = obs
```

暂存 step 前 obs。等 env 返回 reward/done 后一起写 storage。

## process_env_step

`process_env_step` 是 rollout 阶段的后半段。

```python
self.actor.update_normalization(obs)
self.critic.update_normalization(obs)
```

如果模型开了 obs normalization, 更新统计量。

```python
self.transition.rewards = rewards.clone()
self.transition.dones = dones
```

补上 env.step 返回的 reward/done。

```python
if "time_outs" in extras:
    self.transition.rewards += gamma * value * timeout
```

timeout 不是任务失败, 所以要 bootstrap。意思是 episode 因时间截断, 不是状态真的终止, 还要把后续价值补回来。

```python
self.storage.add_transition(self.transition)
self.transition.clear()
```

写入 storage, 清空临时 transition。

## compute_returns

这段是 PPO 灵魂之一。

```python
last_values = self.critic(obs).detach()
```

rollout 结束时, 还需要最后一个 next state 的 value。

```python
for step in reversed(range(st.num_transitions_per_env)):
```

倒序算 GAE。

```python
delta = reward + not_done * gamma * next_value - value
advantage = delta + not_done * gamma * lam * advantage
return = advantage + value
```

你要能手算一个 3-step 例子。不会手算, 就不算懂 GAE。

## update

`update` 做真正训练。

```python
generator = self.storage.mini_batch_generator(...)
```

把 `[T, N, ...]` flatten 成 `[T*N, ...]`, 再随机 mini-batch。

```python
self.actor(batch.observations, stochastic_output=True)
actions_log_prob = self.actor.get_output_log_prob(batch.actions)
values = self.critic(batch.observations)
```

注意: 这里重新 forward。因为 actor/critic 参数可能已经更新, 需要当前新策略的 log_prob 和 value。

```python
ratio = torch.exp(actions_log_prob - old_actions_log_prob)
```

PPO 的重要比值:

```text
new_prob / old_prob = exp(new_log_prob - old_log_prob)
```

```python
surrogate = -advantage * ratio
surrogate_clipped = -advantage * clamp(ratio)
surrogate_loss = max(surrogate, surrogate_clipped).mean()
```

这里前面是负号, 因为 PyTorch optimizer 默认最小化 loss。我们想最大化 PPO objective, 所以取负。

```python
loss = surrogate_loss + value_coef * value_loss - entropy_coef * entropy
```

三部分:

- policy loss: 推 actor。
- value loss: 推 critic。
- entropy: 鼓励探索, 所以前面是减号。

```python
loss.backward()
self.optimizer.step()
```

参数真的在这里更新。

## 你今天要做到

把 `PPO.act`、`process_env_step`、`compute_returns`、`update` 每个函数旁边写一行自己的中文注释。

不要写“这里调用 actor”。要写业务含义:

```text
这里保存旧策略采样 action 的 log_prob, update 时用来和新策略比较。
```
