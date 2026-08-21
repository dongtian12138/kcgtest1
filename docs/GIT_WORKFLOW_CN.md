# kcgtest1 本地版本控制指南

本仓库用 Git 保存源码、配置、测试和当前说明；大型运行证据仍按各自保留规则管理。提交是否已同步到远端必须用 `git status` 和 `git log` 实时确认，文档不写死“当前有几个提交”。

## 当前原则

1. 一次提交只包含一个明确主题，例如“CARTS 检查点”“J599 数据归档”或“陈旧代码清理”。
2. 只按精确路径暂存，避免 `git add .` 把并行任务、用户改动或大文件一起带入。
3. 删除前先建立标签；历史源码由 Git 对象和标签恢复，不在主分支重复保存解压副本。
4. 未经用户明确要求，不自动 `push`、不强制推送、不改远端历史。
5. `artifacts/` 的冻结证据和失败运行不能因为 Git 忽略而随意删除；其保留规则见 `docs/ARTIFACT_RETENTION_CN.md`。

## 一次安全提交

在仓库根目录执行：

```bash
git status --short
git diff --check

git add -- path/to/file_a path/to/file_b
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only

git commit -m "type: concise reason"
git show --stat --oneline HEAD
```

提交前必须确认 `git diff --cached --name-only` 只列出本次任务文件。工作树中出现不属于本次任务的修改时，原样保留，不回退、不顺带提交。

## 建立可恢复检查点

大规模清理或重构前：

```bash
git tag -a checkpoint-name -m "what this checkpoint preserves"
git show --no-patch --decorate checkpoint-name
```

查看被删除文件：

```bash
git show checkpoint-name:path/to/deleted_file
```

导出一组历史文件到临时归档：

```bash
git archive --format=tar checkpoint-name path/to/subtree > /tmp/subtree.tar
```

这些命令只读取历史。不要用 `git reset --hard` 或 `git checkout -- <path>` 处理含有未提交用户改动的工作树。

## 查看本地与远端差异

```bash
git branch --show-current
git remote -v
git status --short --branch
git log --oneline --decorate -10
git log --oneline origin/master..HEAD
```

`origin/master..HEAD` 有输出，表示这些提交只在本地；这不等于丢失，但另一台电脑暂时看不到。只有用户明确要求同步时才执行：

```bash
git push origin master
git push origin --tags
```

不得把访问令牌写进仓库、脚本、终端记录或说明文档。认证失败时先检查凭证管理器和远端地址，不把令牌发给 Agent。

## 换机与离线恢复

联网时从私有远端克隆；离线时使用经校验的 Git bundle。克隆后先核对标签和当前分支，再按仓库 README 构建。`artifacts/`、本地环境和未进入 Git 的外部证据必须通过独立备份恢复，不能假设它们存在于源码仓库。

## 当前清理检查点

- `pre-contract-cleanup-20260821`：第一轮合同清理前。
- `post-contract-cleanup-20260821`：第一轮合同清理后。
- `current-carts-j599-checkpoint-20260821`：当前 CARTS 源码与可恢复 J599 证据进入 Git 后、第二轮深度清理前。
- `post-deep-cleanup-20260821`：第二轮依赖清理和回归完成后。

标签只证明对应文件状态可恢复，不代表动态仿真、正式装配或真实硬件验收通过。
