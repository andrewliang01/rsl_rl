# 01. PyTorch Minimum For rsl_rl

你不用先学完整 PyTorch。为了手撕 rsl_rl, 先吃透这些。

## Tensor

`torch.Tensor` 是带 shape 的数组。

```python
x = torch.randn(32, 10)
print(x.shape)  # [32, 10]
```

在 rsl_rl 里, 第 0 维通常是 batch:

```text
[num_envs, obs_dim]
[batch_size, obs_dim]
[num_steps, num_envs, obs_dim]
```

## nn.Module

所有网络都是 `nn.Module`。

```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

net = TinyNet()
y = net(torch.randn(4, 10))
print(y.shape)  # [4, 1]
```

你要负责的事情:

- `__init__` 里定义层。
- `forward` 里定义数据怎么流过层。
- 不要手动更新参数, optimizer 会更新。

## Gradient

训练的四步:

```python
pred = net(x)
loss = ((pred - target) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义:

- `pred = net(x)`: 前向传播。
- `loss`: 一个标量, 告诉网络现在有多差。
- `zero_grad`: 清空上一次梯度。
- `backward`: 从 loss 反向算每个参数的梯度。
- `step`: optimizer 根据梯度改参数。

## detach

`detach()` 的意思是: 这个 tensor 的数值留下, 但梯度不要再往前传。

PPO 里常见:

```python
self.transition.actions = self.actor(obs, stochastic_output=True).detach()
self.transition.values = self.critic(obs).detach()
```

为什么?

rollout 阶段只是采数据。真正训练在 `update()`。如果 rollout 时不 detach, PyTorch 会试图记住整个 env rollout 的计算图, 内存爆炸, 语义也错。

## no_grad / inference_mode

rollout 时:

```python
with torch.inference_mode():
    actions = alg.act(obs)
```

意思是这段不建计算图。速度更快, 内存更小。

## state_dict

模型保存靠:

```python
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

rsl_rl 里 `save()` 会保存:

- actor 参数。
- critic 参数。
- optimizer 状态。
- RND / AMP 的额外状态。

## 你现在必须背下来的三句话

```text
forward 产生 tensor。
loss.backward 产生梯度。
optimizer.step 更新参数。
```

```text
detach 是切断梯度。
inference_mode 是整段不记录梯度。
```

```text
shape 不清楚, 代码一定看不懂。
```
