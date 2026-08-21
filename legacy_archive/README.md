# 历史源码恢复入口

`legacy_archive/` 不参加当前 ROS 2 构建、默认测试、Isaac 动态运行或正式验收。
已被 Git 跟踪过的旧源码、ROS 1 工程、旧文档和一次性工具不再在主分支保存第二份
散文件或源码 ZIP；统一从本地标签 `pre-contract-cleanup-20260821` 恢复。

## Git 历史恢复

查看标签中的历史文件清单：

```bash
git ls-tree -r --name-only pre-contract-cleanup-20260821 legacy_archive/
```

读取单个文件，不改工作树：

```bash
git show pre-contract-cleanup-20260821:legacy_archive/outdated_docs/README_before_cleanup_20260819.md
```

导出整个旧归档到临时 tar：

```bash
git archive --format=tar pre-contract-cleanup-20260821 legacy_archive/ \
  > /tmp/kcgtest1-legacy-before-contract-cleanup.tar
```

不要用 `git checkout --` 覆盖当前工作树；需要恢复某项实现时先在临时目录审计。

## 唯一额外归档

`RETIRED_B_GRASP_ROUTES_20260820.zip` 保存旧提交中没有的 B_GOAL_MODE、fast-pick、
minimal-v2 和 PAD 兼容性路线。它们只作论文/诊断 baseline，不是当前候选或运行授权。

- 成员数：138
- 大小：616538 bytes
- SHA-256：`626c55f8beeb2285bf19247766545356f31fc60641878fbd5d70c5cd1235a6f4`
- 已排除：`__pycache__/` 和 `*.pyc`

校验：

```bash
(cd legacy_archive && sha256sum -c RETIRED_B_GRASP_ROUTES_20260820.zip.sha256)
unzip -t legacy_archive/RETIRED_B_GRASP_ROUTES_20260820.zip
```

## 清理说明

先前临时生成的 `LEGACY_CODE_SNAPSHOT_20260821.zip` 已删除，因为它重复封装 Git
已有内容，并混入881个可再生成缓存条目。该包不包含需要单独保留的冻结运行证据；
唯一的未提交 B 路线已由上面的小包接替。

`artifacts/` 中的模型、动态日志、失败证据和交付归档没有移入或删除。ZIP 校验通过
也只证明归档完整，不证明任何抓取、装配或动态验收通过。
