# H0b 高程目标合同摘要边界

H0b 数据 manifest 的 `target_contract_payload_sha256` 不是合同 JSON 文件的
逐字节 SHA-256。它绑定的是经过 13-key 严格校验与规范化后的下列预映像：

- JSON 字符编码：ASCII；
- key：字典序排列；
- 分隔符：`,` 与 `:`，无空白；
- 非有限数：禁止；
- 末尾：**没有** `LF`。

该编码公开命名为
`canonical_compact_json_ascii_sorted_no_trailing_lf_v1`，由
`manifest_target_contract_json_bytes()` 与
`manifest_target_contract_payload_sha256()` 唯一实现。dataset manifest builder 在
获得合同对象时也调用同一函数核对调用方提供的摘要；pretrainer 不再维护私有、同名
不明的 JSON hash 实现。

## 2026-08-01 冻结实值

当前 Lab 13-key 合同的无换行 manifest 预映像为 827 字节：

```text
048e649f1c7866e7dcb0f75536f41a21f4c91ca9046e7dd1d48083fdad231b3e
```

Lab sidecar 在完全相同的 827 字节后只增加一个 `0a`，因此文件是 828 字节，
文件 SHA-256 为：

```text
afc45ff19d6a611de220f45e4af65d7e7190eb2a609fe4a715c937b80a1025d5
```

两者不是两个 target contract，也不能交换填写到 manifest 字段和文件证据字段中。
`audit_heightmap_target_contract_json_bytes()` 分别返回：

- `manifest_target_contract_payload_sha256`：规范化语义摘要；
- `file_sha256`：输入文件原始字节摘要；
- `normalized_contract`：严格的 13-key 对象；
- `encoding_relation`：无换行、单个末尾换行或其他语义等价 JSON。

提供预期 manifest 摘要和/或文件摘要时，验证器会 fail closed。key 重排及 JSON
空白不会改变规范化语义摘要；字段值改变、重复 key、alternate contract，以及把一个
文件摘要与另一个语义摘要 cross-splice 都会拒绝。

生产代码不读取或依赖 Lab 仓库。上面的冻结实值仅由独立 RSL 单元测试固定；跨仓
collector 应显式传递合同对象、manifest 语义摘要和（若需要审计文件编码）文件摘要。
