# 当前：四相机完整装配基线与磁盘清理后的本地状态

核验时间：2026-09-29（UTC）。纯仿真，`hardware_authorized=false`；没有运行中的装配实验。本机当前主目录 `/home/noob/WorkPlace/kcgtest1` 位于本地分支 `codex/cleanup-four-camera-baseline-20260929`，从已发布的 `ca5ebc7913bc88190fc94d44084aef7cfcd5a13a` 创建。GitHub 分支 `codex/connector-four-camera-baseline-20260920` 及其独立干净工作树 `/home/noob/WorkPlace/kcgtest1-baseline-20260920` 保持不变；此次清理没有推送。

## 当前唯一认可的完整装配基线

2026-09-19 高位全局相机 1 的四相机方案，在同一连续 Isaac Sim 回合中完成桌面抓取、两次观键、搬运、插入、螺母旋紧和最后 3 秒完全松手保持。原始物理回合位于 `/home/noob/WorkPlace/kcgtest1-performance-20260917/artifacts/initial_contact_fix_20260919/balanced_band_full_01/run`，其 `whole_assembly_review.json` 为 `complete_visual_assembly_verified=true`；原始逐帧档案和五视角录像均保留。详细数据、配置和证据边界见 `reproducibility/high_global1_acceptance_20260919/result_CN.md`。

末端软件原先误把松手前后的正常落座量当成松手后的不稳定。修复后的末端判定仅在本回合已在线获取的传感记录上重放通过；没有用修复版再跑一整轮动力学。原判定记录与修复后审查均保留，不把二者混写。

公开复现分支与固定发布页：

- `https://github.com/dongtian12138/kcgtest1/tree/codex/connector-four-camera-baseline-20260920`
- `https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-four-camera-high-global1-20260920`

主目录已从独立基线工作树补回运行资产，`scripts/prepare_assembly_reproduction.py --assets` 核验了 297 个发布资产及随 Git 保存的 2 个高位相机输入。规划 Python 环境已按基线声明在独立基线工作树中重建，主目录以符号链接复用该环境、视觉源码和记录依赖，并在本地重新编译两个原生记录模块。主目录与独立基线工作树的 `scripts/run_current_hand_assembly.py --check` 均通过，核验了 806 个源码与资产文件、原生记录模块及三个解释器。此次清理没有启动新物理回合。

2026-09-29 腕部回转补核：发布基线仍在提前停拧后采用固定90°空手回转；其成功回合中24.8145°、59.6676°旋拧指令之后均回转90°。后续本地分支 `codex/nut-grasp-efficiency-20260920` 已在执行提交 `60b8cbc` 实现0/±90°等效抓取相位选择，并检查下一段关节可行性。`captured_to_release_02` 的90°/0°/90°回转、剩余旋拧到位及3秒松手阶段曾通过验收，但从已插入状态初始化，`full_tabletop_visual_episode_retested=false`；没有包含该修正的新完整桌面装配基线。

## 本地保留与删除边界

另外保留 2026-09-17 的两次完整重复通过原始回合：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01` 与 `contact_last_gc128_repeat02`；两份 `whole_assembly_review.json` 均为 true。最新四相机回合、两次重复通过、模型运行输入、发布录像和小体积验收证据均未删除。

用户本轮明确要求大幅清理旧失败实验。旧主目录和 2026-09-21 螺母研究工作树的大型实验产物、其他失败或被替代回合的大型逐帧档案、旧 2026-09-16 基线工作树和独立 GitHub 回拉检出副本已删除。历史文档中相应的本地原始路径不再保证存在；不能再写“全部失败原始数据保留”。删除前的 49,323 份小型结果与日志压缩在 `/home/noob/WorkPlace/kcgtest1_cleanup_20260929/experiment_summaries.tar.gz`；大型原始流和媒体不在该压缩包中。未提交源码与文档另有补丁、未跟踪文件包和 Git stash。具体删除目录见同目录的 `selected_deletions.json`、`root_artifacts_deleted.txt`、`nut_artifacts_deleted.txt`。

旧螺母研究工作树已移除，其本地 Git 分支 `codex/connector-decision-research-20260921` 仍在；工作树未提交修改在 stash `4536e94a` 和独立备份中。旧主目录未提交修改在 stash `f3994fb7` 和独立备份中。不要把历史研究任务或动态实验授权从旧上下文自动恢复。详细清理记录见 `docs/DISK_CLEANUP_20260929_CN.md`；清理前原入口归档为 `docs/history/CURRENT_CONTEXT_CN_before_disk_cleanup_20260929.md`。

清理范围更正：上述整研究目录删除也包含已通过的局部回合 `captured_to_release_02` 的原始流和录像，不能全部称为失败产物。当前未找到其录像副本。Git提交 `1483051` 中的原验收文件、末帧图片及回转选择器已提取到 `artifacts/wrist_reindex_audit_20260929` 供审阅；不是重跑产生的新证据。现存完整五视角录像仍对应未含该回转修正的发布基线。
