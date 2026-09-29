# 2026-09-29 本地磁盘清理记录

## 基线与保留范围

用户指定的基线是 GitHub 分支 `codex/connector-four-camera-baseline-20260920`，当前远端提交为 `ca5ebc7913bc88190fc94d44084aef7cfcd5a13a`。固定发布页为 `assembly-four-camera-high-global1-20260920`。主目录 `/home/noob/WorkPlace/kcgtest1` 已改为从该提交创建的本地分支 `codex/cleanup-four-camera-baseline-20260929`；独立基线工作树 `/home/noob/WorkPlace/kcgtest1-baseline-20260920` 和远端分支均未改动。没有推送。

以下三份完整成功回合的原始逐帧数据没有删除：

- 最新四相机回合：`/home/noob/WorkPlace/kcgtest1-performance-20260917/artifacts/initial_contact_fix_20260919/balanced_band_full_01/run`，包含 5 分 13.8 秒五视角录像；原始 `whole_assembly_review.json` 的 `complete_visual_assembly_verified=true`。
- 两次重复装配：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01/run` 与 `contact_last_gc128_repeat02/run`；两个原始审查结果均为 true。

最新基线的 297 个发布资产和 2 个随 Git 提交的高位相机输入已经在主目录核验。独立基线工作树保留已校验的视觉模型源码与权重；规划环境按发布说明重建。主目录以符号链接复用规划环境、视觉源码和记录依赖，并在本地重新编译两个原生记录模块。主目录和独立基线工作树的 `scripts/run_current_hand_assembly.py --check` 均通过，核验 806 个源码与资产文件及运行入口。该检查只读，没有执行预检或完整物理回合；完整重跑不在本次清理范围内。

## 清理结果

磁盘占用的主因是旧回合的逐帧流、录像和重复检出目录。清理前 `/home/noob/WorkPlace` 约 960 GiB；清理后约 175 GiB，减少约 785 GiB。主目录从约 435 GiB 降至约 1 GiB；旧螺母研究工作树约 191 GiB，现已移除。清理选择表记录了其余两个工作树中删除约 109.6 GiB 与 42.0 GiB，同时保留约 26.9 GiB 的最新成功回合和约 51.9 GiB 的两次重复成功回合。完成时根文件系统约 277 GiB 已用、1.5 TiB 可用。数值来自 `du`、`df` 的四舍五入显示。

删除对象包括：

- 原主目录中约 413 GiB 的旧忽略实验产物；保留并重新核验发布基线的运行资产。
- 已冻结的 2026-09-21 螺母研究工作树的大型实验产物及该工作树本身；本地 Git 分支仍在。
- 高位四相机成功前的失败或被替代回合、速度试验和局部诊断的大型产物；两次重复通过工作树内的失败装配档案和临时复现克隆。
- 原主目录的旧 `.venv`、GraspGenX 嵌套克隆与缓存、旧构建和日志。GraspGenX 的小型本地覆盖内容单独封存。
- 旧 2026-09-16 基线工作树和 2026-09-20 独立 GitHub 回拉检出副本；前者的本地与远端 Git 分支仍在，后者对应的远端提交和本地发布核验记录仍在。两者的 SAM 本地补丁与保留的最新基线一致，补丁另行封存。

本次未按“代码文件越少越好”删除发布基线的源码或模型。现有运行入口及其 806 个绑定文件仍可通过只读检查；旧实验分支由 Git 保留，不再占用工作树的大型实验空间。

## 可追溯性与限制

删除前已将 49,323 份不超过 2 MB 的 JSON、YAML、文本、日志、SVG 和 CSV 结果压成 `/home/noob/WorkPlace/kcgtest1_cleanup_20260929/experiment_summaries.tar.gz`，压缩文件约 284 MB，并验证了归档目录可读取。大型逐帧原始流和视频没有包含在该压缩包；已删除的失败回合无法再做完整原始流复算。历史文档引用这些旧本机路径时须按此边界解释。

清理明细位于同目录的 `selected_deletions.json`、`root_artifacts_deleted.txt`、`nut_artifacts_deleted.txt` 和 `root_environment_deleted.txt`。旧主目录未提交改动另存为补丁与未跟踪文件包，并保留 Git stash `f3994fb71a00d9f920cbab47c9cc9dfa584bcb60`；螺母研究未提交改动另存并保留 stash `4536e94a1068b90f1f368549f34f2a9956eb063d`。原有 Git 分支均未删除。原基线分支和固定发布页可从 GitHub 获取。

清理前的活动说明备份为 `docs/history/CURRENT_CONTEXT_CN_before_disk_cleanup_20260929.md`。本次没有执行仿真、修改装配控制/几何/安全边界、推送或发布新的成果。

## 2026-09-29 腕部回转补核更正

清理时对整个螺母研究产物目录的删除范围过宽。该目录中还包含一个已验收通过的局部连续回合 `captured_to_release_02`，其回转策略已选择90°、0°、90°并完成剩余旋拧与3秒松手；它没有从桌面抓取重新执行全流程，但不能归为失败实验。整目录清理同时删除了该回合的大型原始流与 `video/assembly_five_view.mp4`。本轮在保留的工程目录、Git历史、常用下载/文档/缓存/回收站目录中按相关文件名检索，未找到该录像副本。

代码与小体积结果仍在Git中。本轮将提交 `1483051` 的原验收文件、执行绑定和末帧图片，以及实际执行提交 `60b8cbc` 的回转选择器，提取至 `artifacts/wrist_reindex_audit_20260929`；没有恢复已删除原始流或录像，也没有重新执行物理。公开完整四相机基线及其录像仍完好，但仍采用旧回转规则，不应将其称为回转问题修复后的完整验证。
