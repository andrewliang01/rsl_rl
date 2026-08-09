# Livox CustomMsg 真实录包回放合同

`REAL-LIVOX-REPLAY-001` 的第一阶段实现是一个完全 CPU-only 的工程边界：它不导入
ROS、Isaac 或 CUDA，而是把录包字节、部署 manifest、逐窗口 `CustomMsg` 数组和
`Mid360RayTimeTensorBuilder` 的输出绑定成可复查证据。该实现**不是** ROS bag 解析器，
也不意味着仓库已经拥有真实 MID-360 数据。

## 封存目录

一个目录只允许包含以下文件，存在额外文件、符号链接或缺失文件都会拒绝加载：

```text
replay/
├── replay_manifest.json
├── source_recording.bin
├── deployment_manifest.json
└── windows/
    ├── window_00000000000000000000.npz
    └── window_00000000000000000001.npz
```

`source_recording.bin` 是采集时原始文件的逐字节副本；manifest 同时记录其长度和
SHA-256。`deployment_manifest.json` 必须是 canonical JSON，并同时绑定文件字节
SHA-256 和 manifest 自身的 payload SHA-256。复制到固定文件名不会改变其中字节。

每个窗口 NPZ 只允许两个非 object 数组：

- `xyz_m`: little-endian `float32`、`[N,3]`、C contiguous；
- `offset_time_ns`: little-endian `uint32`、`[N]`、C contiguous。

加载固定使用 `allow_pickle=False`。JSON 为排序键、无空白、禁止 NaN 的 UTF-8
canonical 表示，并且只有一个末尾换行。窗口元数据逐一绑定：

- `window_index`；
- `timebase_ns` 与每点 `offset_time_ns`；
- `capture_end_time_ns`、`received_time_ns`；
- 单调时钟域；
- `x-forward/y-left/z-up` 的传感器坐标系与米/纳秒单位；
- NPZ 文件长度与 SHA-256；
- 原消息 `point_num` 和抽取数组点数。

`source_point_count` 必须与两个数组的长度严格相等，禁止适配器在不留痕的情况下切掉
尾部点。该相等性必须在抽取前读取原始 `CustomMsg.point_num`；回放层不会从不透明的
ROS bag 字节中猜测该字段。因此，采集者仍需保留抽取程序版本和采集记录，真实数据
验收时会一并审计。

v1 的一个窗口严格对应一个原生 `CustomMsg`（`source_message_count=1`），因为这个
schema 只保存一个 `timebase_ns`。如果控制窗口需要聚合多个原生消息，必须先通过已有
absolute-time adapter 为每点保留绝对时间，不能把多组 timebase 偷换成一组 offset。

窗口索引必须严格递增且唯一；每个窗口必须属于同一单调时钟域。索引间的空洞由已有
builder 作为 unknown frame 插入，receipt 同时记录缺失索引、逐窗口
`implicit_missing_packets_inserted` 和 builder 累计值。首个窗口之前与末个窗口之后的
丢包无法由有限录包证明，所以 accounting scope 明确限定在首末已录窗口之间。

## CPU 回放

```python
from rsl_rl.utils.livox_custom_msg_replay import (
    replay_livox_custom_msg_artifact,
)

result = replay_livox_custom_msg_artifact(
    "/data/mid360/replay",
    max_packet_age_s=0.5,
)
```

每个窗口都调用现有 `point_packet_from_livox_custom_msg_arrays` 和
`Mid360RayTimeTensorBuilder.ingest_point_packet`，然后输出：

- `[1,K,2,H,W]` 的 `float16` policy tensor；
- 完整 `Mid360PacketStats`；
- 绑定 dtype、shape 和 C-order bytes 的 tensor SHA-256；
- 原窗口 archive SHA-256；
- 丢窗 accounting。

receipt 自身也是 canonical JSON payload SHA-256 封存，且 `training_ready` 永远为
`false`。录包回放证据只能关闭部署数据门禁，不能自动注册训练任务或改变正式训练配置。

## 当前证据边界

仓库没有真实 Livox 录包。自动测试生成的文件全部声明：

```json
{
  "REAL-LIVOX-REPLAY-001": "open_no_real_recording",
  "real_data_present": false,
  "real_replay_verified": false,
  "training_ready": false
}
```

这些 fixture 只证明格式、篡改检测、逐窗口 builder 调用和收据生成可运行，不能写成
实机结果。关闭 `REAL-LIVOX-REPLAY-001` 仍需要一份真实 MID-360 录包、原始文件 SHA、
采集时钟/坐标标定声明、逐窗口 `point_num` 对账，以及用该真实 artifact 产生的回放
receipt。
