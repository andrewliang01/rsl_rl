# 02. PPO Dataflow

一次 PPO iteration 分两段:

```text
rollout: 采数据
update:  学参数
```

## Rollout

入口在 `OnPolicyRunner.learn()`。

伪代码:

```python
obs = env.get_observations()

for step in range(num_steps_per_env):
    actions = alg.act(obs)
    obs, rewards, dones, extras = env.step(actions)
    alg.process_env_step(obs, rewards, dones, extras)

alg.compute_returns(obs)
losses = alg.update()
```

## alg.act(obs)

来自 `PPO.act()`:

```python
self.transition.actions = self.actor(obs, stochastic_output=True).detach()
self.transition.values = self.critic(obs).detach()
self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach()
self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
self.transition.observations = obs
```

它做了五件事:

- actor 采样 action。
- critic 估计当前 value。
- 记录这次 action 在旧策略下的 log_prob。
- 记录旧策略 distribution 参数。
- 暂存当前 obs。

为什么叫旧策略?

因为之后 `update()` 会改变 actor 参数。PPO loss 要比较:

```text
new_policy(action) / old_policy(action)
```

所以 rollout 时必须存旧 log_prob。

## env.step(actions)

环境返回:

```text
next_obs
reward
done
extras
```

注意: `alg.act(obs)` 里存的是 step 前的 obs。`process_env_step(next_obs, reward, done, extras)` 收到的是 step 后的 obs。

## process_env_step

它补齐 transition:

```python
self.transition.rewards = rewards.clone()
self.transition.dones = dones
self.storage.add_transition(self.transition)
self.transition.clear()
```

一句话:

```text
act 先存 obs/action/value/log_prob, env.step 后补 reward/done, 然后写入 storage。
```

## compute_returns

rollout 结束后, storage 里有:

```text
obs, action, reward, done, value, old_log_prob
```

但还没有:

```text
returns, advantages
```

GAE 公式:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
return_t = A_t + V(s_t)
```

代码从最后一步倒着算, 因为 `A_t` 依赖 `A_{t+1}`。

## update

`update()` 从 storage 取 mini-batch。

核心计算:

```python
ratio = exp(new_log_prob - old_log_prob)
surrogate = -advantage * ratio
surrogate_clipped = -advantage * clamp(ratio, 1-clip, 1+clip)
surrogate_loss = max(surrogate, surrogate_clipped).mean()
value_loss = mse(values, returns)
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

最后:

```python
optimizer.zero_grad()
loss.backward()
clip_grad_norm_
optimizer.step()
```

## 一张脑内图

```text
obs
  -> actor -> action -> env.step -> reward/done/next_obs
  -> critic -> value
  -> distribution -> old_log_prob

transition -> storage
storage -> compute_returns -> returns/advantages
storage mini-batch -> update -> loss -> backward -> optimizer.step
```
