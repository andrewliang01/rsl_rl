# 04. Experiments

你要交互式学习, 我建议三种方式混用。

## 1. 直接跑最小脚本

```bash
cd /home/andew/RL/new_lab_ws/rsl_rl
PYTHONPATH=/home/andew/RL/new_lab_ws/rsl_rl python3 tutorials_hand_teach/scratch_ppo_minimal.py
```

你应该看到:

```text
obs policy shape
actions shape
stored rewards shape
returns shape
loss dict
```

第一轮不用懂全部, 先确认每个 shape。

## 2. 用 debugger

推荐断点:

```text
rsl_rl/algorithms/ppo.py: act
rsl_rl/algorithms/ppo.py: process_env_step
rsl_rl/algorithms/ppo.py: compute_returns
rsl_rl/algorithms/ppo.py: update
rsl_rl/storage/rollout_storage.py: add_transition
rsl_rl/storage/rollout_storage.py: mini_batch_generator
rsl_rl/models/mlp_model.py: forward
```

每到一个断点, 看这些变量:

```text
obs.keys()
obs.batch_size
self.transition.actions.shape
self.transition.values.shape
self.storage.rewards.shape
batch.actions.shape
batch.advantages.shape
```

## 3. Jupyter

Jupyter 适合实验 tensor, 不适合完整训练大工程。

你可以在 notebook 里做:

```python
import torch

old_log_prob = torch.tensor([0.0, 0.0])
new_log_prob = torch.tensor([0.2, -0.5])
ratio = torch.exp(new_log_prob - old_log_prob)
print(ratio)
```

再做 GAE:

```python
rewards = torch.tensor([1.0, 2.0, 3.0])
values = torch.tensor([0.5, 1.0, 1.5])
last_value = torch.tensor(2.0)
gamma = 0.99
lam = 0.95

adv = 0.0
advantages = []
for step in reversed(range(3)):
    next_value = last_value if step == 2 else values[step + 1]
    delta = rewards[step] + gamma * next_value - values[step]
    adv = delta + gamma * lam * adv
    advantages.append(adv)

advantages = list(reversed(advantages))
print(advantages)
```

## 练习任务

练习 1:

在 `scratch_ppo_minimal.py` 里把 `num_envs` 从 4 改成 2, 观察所有 shape。

练习 2:

把 `num_steps` 从 8 改成 3, 手算 returns, 对照 storage。

练习 3:

把 actor hidden dims 从 `[32, 32]` 改成 `[64, 64]`, 看 `state_dict().keys()`。

练习 4:

在 `PPO.update()` 里临时打印:

```python
print(ratio.mean(), surrogate_loss.item(), value_loss.item())
```

跑完后删掉打印。

练习 5:

自己写一个 `TinyMLPModel`, 只支持 critic, 替换 scratch 里的 critic。

你做到练习 5, 就已经不是“看代码”, 是开始掌控代码了。
