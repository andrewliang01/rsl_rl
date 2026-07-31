# H2 / PIES 五通道接线契约

新 actor 输入固定为 `[range_m, return_valid, return_age_s, packet_age_s,
frame_valid]`，逻辑形状为 `[B,K,5,H,W]`。`packet_age_s` 与
`frame_valid` 在空间上广播，actor 会验证同一帧内完全一致；未知回波的 range 和
return age 必须严格为零。

旧的 `ray_time` K1/K5 actor 仍使用 `[B,K,2,H,W]`，默认模块图、参数名、导出
wrapper 和输入大小均不改变。五通道路径使用独立的 `ray_event_time` encoder 类型。

## 时间真实性边界

当前 IsaacLab `RayCasterData` 只公开 sensor pose 与整批 `ray_hits_w`，没有每条
射线的发射/命中时间。现有 20% mask 是在一次整批 raycast 后按 packet index 取
子集，所有有效几何来自同一个 pose。因此：

- 仿真只允许 `raycaster_packet × packet_age` 与 `age_zero`；
- 仿真 `per_return_age` 在 observation、actor 和 receipt 三层均直接报错；
- 不允许用像素相位或列号生成伪逐回波时间；
- 当前 packet 时间量化上界为 `5 × 0.02 s = 0.1 s`，必须写入 receipt；
- `exact_union_k1` 是 coverage oracle：每个 cell 取最近 range，在仿真只允许
  age-zero；它不是 PIES。
- `raster_latest_event_prototype` 只是栅格级 latest prototype：每个 cell 取最新有效值并从
  同一 winner gather range/age。由于输入已经做过 packet 内栅格碰撞裁决，它尚未
  证明对任意原始事件重新分包保持不变，receipt 固定记录
  `event_union_stage=post_packet_raster` 与
`packetization_invariance_proven=false`，因此仍不可训练。

真正的 PIES 合约固定使用 `event_window_s=0.5`，必须在 packet rasterization **之前**
接收带稳定 event id 的原始回波集合。每个角度 cell 按“最小 age（最新）→同龄最近
range→最小稳定 event id”选 winner，并从同一 winner 输出 range/valid/age。仓库中的
`raw_event_pies.py` 对同一原始事件集合的 K=1、K=3、K=5 与 irregular/乱序分包验证
range/age/acquisition-delta-proprio 同 winner，输出 bitwise 相等且 SHA-256 相等。
这只证明 reducer 对同一原始事件集合的分包不变性；当前 RayCaster 接线仍处于
`post_packet_raster`，不得借用这份证明把它标成可训练或逐回波时间。

## 自回波边界

仿真 RayCaster 只 cast `/World/ground`，机器人自身不在 mesh 列表中。这是
`simulator_geometry_excludes_robot`，不是实机 self-return filter。部署 receipt 必须
记录 `self_return_filter`、filter config SHA-256 与 `self_return_filtered_count`；真实
MID-360 未绑定 `upstream_static_mask` 或 `urdf_kinematic` 证明时，scope 固定为
`synthetic_conformance_only`。被过滤回波的语义是 removed observation，禁止改写为
emitted no-return/free-space 证据。

actor/deployment receipt 同时固定记录五个通道的逐通道语义、0.5 s PIES 窗口、event
union 所在阶段、分包不变性证明哈希和 self-return-filter provenance，缺失任一训练
证据都 fail-close。

当前五通道 actor 尚未消费 acquisition-time delta-proprio。因此 raw-event reducer 只是
PIES 的 range-age-delta-proprio conformance primitive，receipt 固定写入
`pies_full_contract_ready=false`；即使合成分包证明通过，也不能据此宣称完整 PIES 或
开启 PIES 训练。

真实 Livox 路径由 `Mid360RayTimeTensorBuilder` 使用 CustomMsg 的
`timebase + offset_time` 生成同源 range/valid/return-age winner，随后通过
`aligned_history_to_ray_event_observation` 接入相同 actor。
仅把 `source` 字符串写成 `livox_per_return` 不构成认证。只有同时绑定 real-tensor
manifest 与 common-clock alignment receipt 的 SHA-256 后，receipt 才允许
`per_return_claim_allowed=true`；缺任一项时保持未认证并禁止 training promotion。

可选的未来控制是把 RayCaster 更新率提高到每个 control step（20 ms），每步只保留
一个稀疏子包，再按真实 capture step 聚合。它只能称为 “20 ms-quantized packet
event time”，不能称为连续逐回波时间；本提交没有注册或宣称该任务。

## 训练门禁

lab 侧提供四个未注册配置：K5 packet-age、K5 age-zero、coverage-nearest K1
age-zero，以及 `raster_latest_event_prototype`。
所有配置保持 `ray_event_training_ready=False`，Gym registry 中没有对应训练 task。
后续只有真实 64-env smoke 成功、receipt 绑定其 SHA-256 后，才允许单独提交训练就绪
的 promotion；不得直接修改本门禁。
