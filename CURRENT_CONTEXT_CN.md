# 当前：接入已验证回转策略并执行完整四相机装配验收

核验时间：2026-09-29T11:39:38.464862+00:00。用户本轮已明确授权：把已验证回转策略接入完整基线，从桌面抓取实际执行到最终松手，完成同回合验收，形成新基线并上传GitHub，保存录像。仅仿真，hardware_authorized=false；没有子代理或硬件授权。当前处于运行前准备，尚未启动新物理回合。

## 目标与冻结边界

主目录 `/home/noob/WorkPlace/kcgtest1`，新分支 `codex/connector-adaptive-return-baseline-20260929`。原完整基线 `ca5ebc7` 及独立工作树 `kcgtest1-baseline-20260920` 保留。只移植已完成局部动态验证的 `60b8cbc` 中回转选择、下一段关节预览和重抓朝向保留；不引入研究分支的柔顺、抓位、预载或模型改动。

专用配置和绑定在 `reproducibility/adaptive_return_baseline_20260929`。与原高位四相机配置相比仅增加 `nut_reindex.phase_equivalent_return=true` 与首个回转的 `next_loaded_turn_deg=-90`；后续实际下一段角度由既有继续旋拧入口传入。四相机、首抓、物体模型、质量/材料/接触、960Hz/64/4、原力速/关节/验收限制相同。在线仍只使用当前视觉和机器人信号。原始接触/对象数据仅用于结束后的验收。

## 本轮执行与验收

复用 `scripts/run_current_hand_assembly.py --run`；先新的300秒预检，通过后完整回合沿用21600秒预算和300秒收尾。一次只运行一个主要整机回合；不延长已经开始的运行期限。程序退出和录像生成不能替代成功验收。检查桌面抓起、两次观键、插入、实际旋合、源面/键槽/关节/传动、金属止挡及128夹片、完整3秒零手部接触保持，以及同回合原视频。

原完整成功原始数据及录像仍在 `kcgtest1-performance-20260917/artifacts/initial_contact_fix_20260919/balanced_band_full_01/run`。新完整成功录像须另外复制到本机独立视频保存目录，并上传本次GitHub发布附件；原始完整运行目录也保留，不按清理缓存删除。后续局部回转验收与上轮误删研究录像的更正见 `docs/DISK_CLEANUP_20260929_CN.md` 和 `artifacts/wrist_reindex_audit_20260929`。

## 恢复后的下一步

先核对真实进程和最新运行文件。本轮尚需针对回转的测试、完整物理执行、结束后数值与影像验收、封存和GitHub发布。没有实际完整通过前，不宣称新基线完成。旧上下文归档为 `docs/history/CURRENT_CONTEXT_CN_before_adaptive_return_full_validation_20260929.md`。
