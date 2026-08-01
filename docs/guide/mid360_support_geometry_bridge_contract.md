# MID-360 到物理支撑证据桥接合同

本合同只闭合一条软件接口：

```text
Livox CustomMsg
  -> 外部标定的传感器时钟到策略时钟映射
  -> [B,K,5,16,96] return-event tensor
  -> 采集时机身到当前机身的短时因果变换
  -> 当前/预测落脚中心的四个物理支撑角色
  -> selected-only H1 actor inputs
```

五个通道固定为 `range_m, return_valid, return_age_s, packet_age_s,
frame_valid`。`return_age_s` 已经相对当前策略时刻定义，桥接层不会再叠加
`packet_age_s`。无效栅格严格表示 unknown：range 和 return age 必须为零，
不得解释成 free space、no-return 或最大量程回波。

历史回波不能默认处于当前机身坐标系。调用方必须提供
`history_body_to_current_rotation/translation`，以短时 IMU/本体状态将每个
采集时机身坐标系对齐到当前机身坐标系。该接口不需要全局地图或长期
里程计，但当前组件也不验证这些动态变换的外部证据来源。

四个角色顺序固定为：

1. left current support；
2. left causal landing support；
3. right current support；
4. right causal landing support。

角色掩码只允许使用标定射线、关节编码器 FK 足端中心、命令/步态相位
产生的因果落脚中心以及短时机身变换，不允许使用仿真 contact truth 或
terrain truth。range、angle、age strata 和 candidate priority 可注册为
matched-substitution 的不可变审计元数据。

当前成立的证据只有 CPU 软件接线与定向测试。以下声明仍为 false：真实
MID-360 录包认证、外部标定验证、动态机身变换字节绑定、正式 IsaacLab
任务接线、GPU P99、训练就绪以及 G1 实机闭环。
