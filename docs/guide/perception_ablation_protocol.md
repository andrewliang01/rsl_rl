# H1/H2 感知消融协议（接线前版本）

本协议只冻结实验语义、CPU 路由、诊断张量和预留配置名，不注册 Gym
训练任务。当前策略观测只有 range 与 hit mask；H1 还缺标定后的四类足端角色
mask，H2 还缺逐回波采集时间与 rerender 三元组。把这些名称注册成可训练任务会
产生“参数生效但数据不存在”的伪实验，因此接口采用 fail-closed 设计。

H1 四个角色的唯一合法顺序为 `left_current_support`、
`left_landing_support`、`right_current_support`、`right_landing_support`。
旧的 near/far 标签已被不兼容地移除；二者不是同义词，不允许自动迁移。

## H1 名称

- `h1_full`
- `h1_{glad,role,random}_m{08,16,32,64}_selected_only`

其中 `glad` 是状态条件全局 top-M 重叠基线；`role` 是四类角色约束下共享一个
unique-token 总预算；`random` 从与 role 完全相同的候选并集无放回采样，匹配候选
机会和实际 unique 数量。它目前不声称匹配 range/angle/age 分层。

## H2 名称

因果 2x2：

- `h2_native_correct_history_per_return_age`
- `h2_native_shuffled_history_per_return_age`
- `h2_rerender_correct_history_per_return_age`
- `h2_rerender_shuffled_history_per_return_age`

时间/历史控制：

- `h2_native_correct_exact_union_k1_per_return_age`
- `h2_native_correct_raster_latest_event_prototype_per_return_age`
- `h2_native_correct_history_packet_age`
- `h2_native_correct_history_age_zero`

`exact_union_k1` 是 coverage oracle，在每个角度栅格选择全历史最近回波；距离相同时选择更早采集的
回波，range 与 return age 始终来自同一个 winner。`shuffled` 只在同一 packet 的
有效回波内部打乱时间，并验证每帧时间多重集守恒。

`raster_latest_event_prototype` 只在已按 packet 栅格化的历史上选择最新有效值，
不得称为 PIES；两者不得混名。

## 查看冻结配置

```bash
python -c "from pprint import pprint; from rsl_rl.utils.perception_ablation_protocol import perception_ablation_receipt; pprint(perception_ablation_receipt())"
```

输出中的 `reserved_overrides` 是后续环境和 actor 接线后使用的稳定 Hydra 名称，
目前故意不可直接传给训练入口。每个条目均携带 `training_ready=False` 和明确的
`blocking_inputs`。

## 必须记录的指标

H1 至少记录请求/实际 unique M、shortfall、每个角色实际数量、角色重叠、候选数、
有效 token 数和 selector entropy。H2 至少记录每帧有效回波数、时间均值/范围、
shuffle 守恒、被改变关联数、exact-union 冲突栅格数。配对干预必须在同一 checkpoint
与同一 episode/seed 上记录动作 L2、动作 KL、value delta、地形成功率、跌倒率、
最小边缘裕度和 unsafe-step 次数。

稳定的机器可读 metric key 位于
`rsl_rl.utils.perception_ablation_protocol` 的 `H1_METRIC_KEYS`、
`H2_METRIC_KEYS` 和 `PAIRED_CAUSAL_METRIC_KEYS`。
