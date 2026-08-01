# MID-360 真实消息到策略张量的软件集成合同

`ingest_livox_custom_msg_runtime_step()` 将此前独立验证的组件接成一条 CPU 可执行路径：

```text
Livox CustomMsg
  -> 官方字段/点数/frame/数值检查
  -> sensor-clock Mid360PointPacket
  -> 外部仿射时钟映射
  -> action-clock packet + transport latency
  -> manifest-bound Mid360RayTimeTensorBuilder
  -> [1,K,2,16,96] range history
  -> [1,K,5,16,96] range/event-time history
```

两个 actor view 的 range 与 return-valid 在函数返回前逐元素比对。收据绑定 deployment
manifest SHA、clock-alignment receipt SHA、transport latency、shape、dtype、两个
actor tensor 的完整字节 SHA-256 和当前碰撞裁决语义。
`validate_mid360_software_runtime_step()` 会从实时 tensor、packet stats 和 receipt
重建这些绑定；修改 tensor 或把 `training_ready` 改为 true 都会失败。

## 证据边界

这只是软件接口闭合，不认证物理传感器或外部校准证据，也不表示已经完成 G1 闭环。
尤其必须区分：

- 当前 production builder 在一个 packet 的同 cell 碰撞中先取最近距离，只有等距时才
  用更早 timestamp 破同距平局；
- PIES 要求在 raw-event union 上先取最新事件，再以距离和 stable event id 破同龄平局。

因此本路径可以为 Global K1/K5 以及普通 per-return-age actor 生成软件张量，但不能把
结果叫作 PIES。收据固定写入：没有 stable raw-event id、没有连接 raw-event PIES
reducer、same-winner 对照未就绪、training-ready=false、真实录包未认证。只有后续把
原始事件身份、分包前 union、真实 clock evidence 和录包 manifest 全部接入并通过，才
能另行提升这些门。

## CPU 验证

```bash
PYTHONPATH=. pytest -q \
  tests/test_mid360_runtime_integration.py \
  tests/test_mid360_ros_adapter.py \
  tests/test_mid360_clock_alignment.py \
  tests/test_mid360_ray_time_builder.py \
  tests/test_pies_same_winner_age_controls.py
```

测试同时构造同 cell 的早近/晚远两条返回，证明 production builder 保留近表面且收据
仍拒绝 PIES promotion，防止以后把两种碰撞语义误写成同一个方法。
