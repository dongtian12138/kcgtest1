# kcgtest1 当前轻量上下文

> 快照时间：2026-08-26T12:40:00Z
> 当前分支：`carts-grasp-surface-v2-fast6h-20260826`
> 原始运行产物、源码、哈希和进程事实优先于本摘要。

## 一句话状态

Surface V2 对象 B 的固定设计、原始网格精查、唯一轴向分层复查和两轮登记边界精修均未产生满足 1 mm 桌面余量的三指几何候选；当前停止算法修改和新 Isaac，只做六小时窗口证据收口，不能据此声称三指手机械结构无解。

## 六行恢复摘要

- 最初目标：A/B 使用同一无指甲手型、同一方法和主要参数，由三根真实内侧抓持面接触后离桌、抬升至少 50 mm、保持至少 2 s，且无未授权穿透。
- 当前已完成：对象表面 PRIMARY/SECONDARY/HARD 三角色、区域接触、对象 B 固定候选第一指 Isaac 区分、`7×72×3×27=40,824` 完整便宜搜索、两组各 24 项精查和两组登记的各 8 项精修。
- 当前真实物理结果：固定 Top-1 在 Isaac 运行 398 步/3.317 s；f1j2 峰值速度 0.164896 rad/s、最大绝对受力 0.037169 N·m，但 PhysX 手—物接触为 0，第二/三指及抬升未启动。
- 当前唯一主要阻塞：分层精修最佳三指精确重放的桌面间隙为 0.634889 mm，距冻结的 1 mm 操作余量仍差 0.365111 mm。
- 最近用户指令：`SURFACE_SEMANTICS_METHOD_CORRECTION + FEATURE_AWARE_FAST_SEARCH + REGION_CONTACT + SIX_HOUR_ISAAC_CLOSURE`；不降低 12 N、3 rad/s、50 mm、2 s 或真值隔离。
- 下一步：不再改变排序、扩大 24/8 预算或启动新 Isaac；保存全部负结果、完成定向回归、监督复核、提交和普通 push，到窗口截止后只交付事实。

## 当前窗口和边界

- 窗口：`SURFACE_V2_FAST6H`；开始 `2026-08-26T09:16:26Z`；硬截止 `2026-08-26T15:16:26Z`。
- 状态：`IMPLEMENTING`；里程碑：`SURFACE_V2_EVIDENCE_CLOSEOUT`。
- `hardware_authorized=false`、`formal_dynamic_pass=false`、`research_dynamic_pass=false`。
- 研究动态门失败关闭：当前没有通过采样原始网格几何的候选，故不能进入 12 N、IK、全路径或新 Isaac。
- 不覆盖旧带指甲模型，不修改对象物性，不缩小 1 mm 桌面余量，不降低速度/力量/50 mm/2 s 标准。

## Surface V2 当前有效数据

- 对象 B 面角色：PRIMARY `7269` 面/`31.440%`面积，SECONDARY `132` 面/`1.614%`，HARD `679635` 面/`66.946%`；`face21232` 为 SECONDARY。
- 固定姿态重评：8 个输入、7 个名义研究候选、6 个误差任务幸存；Top-3 第一指几何等价，PRIMARY 领先 HARD 仅约 `5.56 nm`。
- 固定 Top-1 第一指 Isaac：398 步/3.317 s，f1j2 峰值 `0.164896 rad/s`，峰值绝对受力 `0.037169 N·m`，对象最大移动 `0.135 µm`，PhysX 手—物接触 0；同第一指几何的固定 Top-3 因语义边界且无物理接触被淘汰。
- canonical 快搜：40,824 个组合、189 组 FK 缓存、85.072 s；中心代理为 4 个可能三指接触、58 个 HARD 先到、40,762 个缺少三指中心接触。
- 原全局排序：Patch 轴向层 `0/78/18`、精查 `0/20/4`；24/24 精查中 20 个低位三指见证投影后丢失接触，4 个最多只有 1/2 指，几何幸存为 0。
- 原登记精修：8 个候选、41 次三指精确重放、几何幸存为 0；最佳仍差 1 mm 桌面余量 `7.050 mm`。
- 唯一分层复查使用相同 40,824 ID 和层内排序；Patch 固定 `32/32/32`、精查固定 `8/8/8`，快搜 84.850 s。
- 分层精查：24/24 完成，低/中/高层三指见证分别 `3/3/7`，其余 `5/5/1`；几何幸存为 0，13 个桌面—接触冲突为 `2.892–19.112 mm`，中位 `8.766 mm`。
- 分层精修：8 个固定名额中 3 个与原登记种子/见证完全相同并复用有效结果，5 个新跑；8/8 几何幸存为 0。最佳候选 `opposition_p0959931_a46_z1__p212` 的三指精确重放桌面间隙 `0.634889 mm`，仍差 `0.365111 mm`。
- 没有可执行 Top-3，任务受力、IK和全路径没有对失败几何候选继续计算；新一轮 Isaac 没有启动。
- 当前三指真实接触、离桌、50 mm和2 s均为 false；对象 B 抬升 `0 mm`、保持 `0 s`，对象 A 本窗口未运行。

## 证据入口

- 快搜：`artifacts/carts_v2/surface_v2_fast6h/feature_search_B_run03.json`
- 分层快搜：`artifacts/carts_v2/surface_v2_fast6h/feature_search_B_stratified_run04.json`
- 分层精查：`exact_B_stratified_run08/09/10_*.json`
- 分层精修清单：`refinement_rescue_selection_B_stratified_run02.json`
- 新精修结果：`refinement15mm8deg_B_stratified_index00/04/05/06/07_run01.json`
- 固定候选动态评价：`object_b_top1_first_finger_run01/evaluation_run02.json`

## 恢复读取顺序

1. `AGENTS.md`
2. `docs/carts_v2/NORTH_STAR_CN.md`
3. 本文件
4. `docs/carts_v2/DECISIONS_CN.md` 最新部分
5. `artifacts/carts_v2/STATE.json`
6. `git status` 与 `git log -8`
7. 上述 Surface V2 原始产物和当前唯一相关源码

## 证据边界

- 快搜是排序代理；登记控制步上的原始网格精查不是连续路径数学证明。
- 离线三指见证不是 PhysX 接触，更不是抓取或抬升。
- 程序退出码、测试、文件或哈希不能替代三指接触、离桌、50 mm、2 s、滑移和穿透证据。
- 当前负结果只覆盖本轮固定离散设计与登记精修邻域，不能证明完整连续构型空间无解。
