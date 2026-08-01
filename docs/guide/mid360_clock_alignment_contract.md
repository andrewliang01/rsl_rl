# MID-360 传感器时钟到策略时钟合同

真实 `CustomMsg.timebase + offset_time` 不一定与 G1 策略调用的
`CLOCK_MONOTONIC_RAW` 属于同一数值时钟。ROS callback 的接收时刻还包含传输、调度和
队列延迟，不能用作采集时刻的替代或自动校准依据。

`Mid360ClockAlignment` 因此只接受外部测得的仿射映射：

```text
action_time_s = scale * sensor_time_s + offset_s
```

软件门固定要求：

- 两个显式且不同的 clock-domain identity；
- 传感器序列号与主机 boot id；
- 至少 32 个外部 cross-timestamp 样本和至少 30 s 校准跨度；
- 绝对 drift 不超过 100 ppm；
- residual P99 不超过 2 ms，maximum 不超过 5 ms；
- uncertainty 覆盖 maximum residual 且不超过 5 ms；
- 原始外部校准证据的 SHA-256；
- packet 的完整采集区间位于已校准区间，禁止无界外推。

`map_livox_packet_to_action_clock()` 只接受尚未写入 receive time 的原始 CustomMsg
packet。调用方另外传入 action-clock receive time；映射后才能计算 transport latency。
返回值携带 clock-alignment payload SHA，后续 tensor、export 和实机 trial receipt 必须
继续绑定该 SHA。

本模块不会验证 SHA 所指向的硬件证据，也不会从 ROS receive time 反推 offset。其
receipt 明确写入：

```text
external_evidence_verified_by_this_module = false
claim_scope = software_mapping_contract_only
```

因此 CPU 测试通过只能说明映射逻辑、边界和 builder 接线闭合；要形成真实 H2/PIES
证据，仍需保存原始 PTP/PPS/hardware cross-timestamp 记录、校准工具版本、设备序列号、
boot id、残差明细和长时漂移复测。
