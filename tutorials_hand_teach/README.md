# rsl_rl Hand-Teach Tutorial

这个文件夹不是官方文档, 是给你从零拆 rsl_rl 用的。

目标很直接:

- 你能说清一次 PPO iteration 里每个 tensor 从哪里来, 到哪里去。
- 你能在 debugger/Jupyter 里暂停任意一行, 知道当前变量应该是什么 shape。
- 你能手写一个 toy PPO, 再回头看 rsl_rl 为什么这么工程化。
- 你能继续独立开发 MultiPPO / AMPPPO / 自定义 actor-critic。

推荐顺序:

1. `00_map.md`: 仓库地图, 先知道文件谁管什么。
2. `01_torch_minimum.md`: 只学 rsl_rl 需要的 PyTorch。
3. `02_ppo_dataflow.md`: 一次 PPO rollout + update 的数据流。
4. `03_line_by_line_ppo.md`: 对 `ppo.py` 主线逐段导读。
5. `04_experiments.md`: 怎么用脚本/Jupyter/断点交互式学。
6. `scratch_torch_minimal.py`: 纯 PyTorch 第一课, 不依赖 rsl_rl。
7. `scratch_ppo_minimal.py`: 一个不依赖 Isaac 的最小 PPO 实验脚本。

Jupyter 讲义:

- `notebooks/00_course_map.ipynb`: 课程地图和 rsl_rl 文件职责。
- `notebooks/01_torch_minimum.ipynb`: PyTorch 最小必备。
- `notebooks/02_ppo_dataflow_and_gae.ipynb`: PPO 数据流与 GAE。
- `notebooks/03_actor_critic_distribution.ipynb`: Actor/Critic/Gaussian distribution。
- `notebooks/04_rollout_storage_toy.ipynb`: 手写玩具 RolloutStorage。
- `notebooks/05_multi_ppo_concepts.ipynb`: MultiPPO 概念手写。
- `notebooks/06_amp_reward_flow.ipynb`: AMP reward flow。

学习方式:

- 不要只读。每读一段, 就运行一次 `scratch_ppo_minimal.py` 或下断点。
- 每看到一个 tensor, 先问 shape, 再问梯度, 最后问谁会更新它。
- 每看到一个 `.detach()`, 必须停下来问: 为什么这里不让梯度回去?

最短路径:

```bash
cd /home/andew/RL/new_lab_ws/rsl_rl
python3 tutorials_hand_teach/scratch_torch_minimal.py
```

进入 rsl_rl 代码实验:

```bash
cd /home/andew/RL/new_lab_ws/rsl_rl
python3 tutorials_hand_teach/scratch_ppo_minimal.py
```

如果 `scratch_ppo_minimal.py` import 失败, 先确认当前环境有 rsl_rl 依赖。当前 rsl_rl 需要:

```text
torch
tensordict
numpy
```

可以临时这样跑:

```bash
PYTHONPATH=/home/andew/RL/new_lab_ws/rsl_rl python3 tutorials_hand_teach/scratch_ppo_minimal.py
```

如果报 `ModuleNotFoundError: No module named 'tensordict'`, 说明你当前 shell 的 Python 不是训练环境。切到 Isaac/rsl_rl 训练用的 Python 环境后再跑。
