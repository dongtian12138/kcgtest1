# J599 完整原始证据归档

本目录保存 `evidence/` 在清理前的完整、逐字节可恢复副本。普通审阅继续使用上级目录中保留的最终摘要、静态验证和两组已接受报告；大体积的接触流、轨迹与临时物理场景放入压缩包，避免默认文件检索和上下文读取被历史运行数据占满。

## 文件

- `J599_25_35_FULL_EVIDENCE_20260820.tar.gz`：原始 `evidence/` 全目录，共 54 个文件、65 个归档成员。
- `ARCHIVE_SHA256.txt`：压缩包自身的 SHA-256。
- `EVIDENCE_ORIGINAL_SHA256.txt`：归档前 54 个文件的逐文件 SHA-256。

## 证据边界

归档不会改变任何运行结果或验收结论。压缩包内同时包含中间失败运行、已接受运行的原始接触/轨迹以及当时的授权文件；`passed=true`、退出码 0 或文件存在均不能脱离对应报告和门限解释为新的动态通过，更不构成真实硬件验收。

## 校验与恢复

从 `j599_25_35_standard_interface_v1/` 目录执行：

```bash
sha256sum -c evidence_archive/ARCHIVE_SHA256.txt
tar -tzf evidence_archive/J599_25_35_FULL_EVIDENCE_20260820.tar.gz

restore_dir="$(mktemp -d)"
tar -xzf evidence_archive/J599_25_35_FULL_EVIDENCE_20260820.tar.gz -C "$restore_dir"
(
  cd "$restore_dir"
  sha256sum -c "$OLDPWD/evidence_archive/EVIDENCE_ORIGINAL_SHA256.txt"
)
```

需要恢复到工作树时，先在临时目录完成上述校验，再复制所需文件；不要直接覆盖当前保留的接受证据。
