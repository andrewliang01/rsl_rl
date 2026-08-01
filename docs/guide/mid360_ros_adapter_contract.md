# MID-360 ROS 消息适配合同

本模块补齐真实消息进入冻结 tensor builder 前的数值边界，但**不声称已经完成实机
闭环**。实现不导入 ROS、Isaac 或 CUDA；ROS1/ROS2 callback 可以把原消息对象直接传给
以下函数：

- `livox_custom_msg_to_mid360_packet()`；
- `extract_livox_custom_msg()` 与
  `livox_custom_msg_to_sensor_clock_packet()`，用于后续显式跨时钟映射；
- `livox_pointcloud2_to_mid360_packet()`；
- `extract_livox_pointcloud2()`，用于逐字节审计 PointCloud2。

对应官方定义固定于 `livox_ros_driver2@13eb05e4e6dd7a765b934d0c5fd6236676a57b49`：

- [`CustomMsg.msg`](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/msg/CustomMsg.msg)；
- [`CustomPoint.msg`](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/msg/CustomPoint.msg)；
- [官方 README 的 PointXYZRTLT/CustomMsg 字段说明](https://github.com/Livox-SDK/livox_ros_driver2/blob/13eb05e4e6dd7a765b934d0c5fd6236676a57b49/README.md#32-livox-ros-driver-2-internal-main-parameter-configuration-instructions)。

官方同时说明该驱动主要面向调试而非量产。本项目因此把消息结构兼容和实机可靠性分成
两个门：本模块只关闭前者的 CPU 软件门，后者仍需真实 MID-360 录包、时钟审计和长时
运行。

## CustomMsg

适配器逐点读取 `x/y/z`（米）和 `offset_time`（纳秒），并用
`timebase + offset_time` 构造返回时间。严格不变量为：

1. `point_num == len(points)`，禁止无记录截尾或过滤；
2. `timebase` 为 `uint64` 范围，`offset_time` 为 `uint32` 范围；
3. 所有坐标有限；
4. `header.frame_id` 必须等于已标定的传感器 frame；
5. callback 显式提供 `window_index`、`capture_end_s`、`received_time_s` 和时钟域；
6. `received_time_s >= capture_end_s >= latest return time`；
7. 不用 `header.stamp` 替换或猜测上述时间。

若设备 `timebase` 与策略 action clock 不同，不得调用第一条路径伪称共同时钟。
`livox_custom_msg_to_sensor_clock_packet()` 会保留设备时钟并故意将 receive time 留为
`None`；随后必须通过 `Mid360ClockAlignment` 映射，才能写入 action-clock callback
receive time 并计算 transport latency。软件不会用 receive latency 反推 clock offset。

输出坐标合同固定为
`mid360_physical_sensor_frame_x_forward_y_left_z_up_metres`。调用方只有在确认消息已经位于
该物理传感器坐标系时才能使用适配器；外参变换、去畸变和 G1 body frame 变换不在此处
暗中执行。

## PointCloud2

适配器逐字节解释标准 PointCloud2 元数据，支持：

- organized cloud；
- 行尾 padding；
- little/big endian；
- 任意合法 field offset 和 point step。

但字段语义只允许两种显式模式：

1. `use_livox_point_timestamps=True`：`x/y/z` 必须为 scalar `float32`，
   `timestamp` 必须为官方 PointXYZRTLT 的 scalar `float64` 绝对秒；
2. `False`：只生成 range-only 的 capture-window packet，不声称有 per-return time。

时间模式默认拒绝常数时间戳、逆序时间戳、非有限时间戳和捕获窗外时间戳。这样可避免
把驱动异常误写成 H2 的真实时序证据。range-only 模式可用于当前 `[range,valid]` K1/K5
策略，但不能为 event-time actor 提供证据。

所有点都必须有限。PointCloud2 的 `is_dense=false` 不会触发静默删点；只要存在 NaN/Inf
整条消息就 fail closed。缺失返回仍保持 unknown，适配器不会把未收到的方向伪造成
no-return。

## CPU 验证

```bash
PYTHONPATH=. pytest -q \
  tests/test_mid360_ros_adapter.py \
  tests/test_mid360_ray_time_builder.py \
  tests/test_livox_custom_msg_replay.py
```

测试覆盖 CustomMsg 点数对账、两种 endian、organized row padding、字段类型/重叠/长度
篡改、坐标和时间异常，以及适配结果进入 manifest-bound builder 后生成
`[1,K,2,16,96] float16` 张量和严格 event-time 对齐。

## 尚未关闭的实机门

- 真实 MID-360 ROS bag/LVX 及逐窗口 `point_num` 对账；
- PTP/GPS/设备时钟到策略 action clock 的实测映射；
- 运动畸变和外参标定误差；
- callback 队列、丢包、延迟和 window index 状态机；
- export bundle/ONNX 与当前正式 winner；
- Jetson 推理、Unitree LowCmd 安全状态机和 G1 闭环。

因此当前证据表述只能是“ROS 消息数值适配器及 CPU 合同通过”，不能表述为“真实
MID-360 已验证”或“已完成 Sim-to-Real”。
