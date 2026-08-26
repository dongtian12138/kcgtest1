# kcgtest1 当前轻量上下文

> 快照时间：2026-08-26T23:46:00Z
> 当前分支：`carts-grasp-contactopt-1488-fast6h-20260826`
> 原始源码、配置、产物、Git、进程和时间戳优先于本摘要。

## 当前唯一优先级

当前任务 `CONTACT_TELEMETRY_POSITIVE_CONTROL + REMAINING_CANDIDATE_TASK_FIRST_SELECTION`
已按用户要求停止。两项结论彼此独立冻结：

- `q09_a13 = REJECTED_NON_TASK_GEOMETRY_PRECEDES_TASK_CONTACT`，接触报告修复不得使它重新晋级。
- `CONTACT_TELEMETRY_UNVERIFIED`。A/B/C 最终正对照中，A 位于 offset 外；B 的 TASK 凸块正间隙
  0.156720 mm；C 第一步 TASK raw/凸块均相交，raw 非 TASK 仍有 0.477629 mm、非 TASK 凸块
  仍有 2.803126 mm。128/128 shape enabled、无 pair filter/group，两端动态刚体的
  `ContactReportAPI.threshold=0`，但 full event、basic event、full poll、ContactSensor raw 和项目
  `hand_object` 仍全部为 0；filtered reading 有效但 `in_contact=false`。该失败停留在 PhysX 原始
  报告导出层，不能把下游 0 单独解释成无接触。
- 因接触链未通过，q09 step 503 负对照没有运行。剩余 4 个输入已完成 raw 与复合凸资产几何
  粗采样+二分表；真实 `q_*_contact_physx` 保持 null，当前决策为 `PARKED_NO_DYNAMIC_SELECTION`。
- 当前只允许汇报和只读复核；不重跑 1488、不增加候选、不改 Surface V2、不启动第一指动态。

## 漏斗复盘的已验证事实

- 旧 1488 个规格：664 个轴向目标越界、567 个三点边长不兼容、194 个目标窗口无面、
  52 个特征方向不足、4 个点/法向对齐失败，只生成 7 个姿态。
- 旧生成器把 72×24 外轮廓的 1,543 个代表面继续当作三点目标搜索域；它只占 B0 外部
  承载面 687,036 个面的 0.2246%。这会制造“窗口无面”和边长假拒绝。
- 24 个分层边长失败样本在完整 B0 表面重评后有 10 个无需改变 1.5 mm 门限即可通过。
- 664 个轴向拒绝中有 541 个只需共同平移三指目标中心即可进入原抓持带；中位平移量
  2.79 mm。真正手形轴向跨度超出对象抓持带的为 123 个。
- 诊断产物：`artifacts/carts_v2/contactopt_1488_fast6h/generator_funnel_diagnosis_run01/`。

## 已实施的最小生成修正

- 1488 规格、手型、B0 对象语义、1.5 mm 边长门、碰撞门、12 N 和随机性约束均未改变。
- 三点目标改在完整 B0 外部承载面、完整 ±12.5° 窗口内搜索，每指固定 64 个确定性
  点/法向代表；72×24 外轮廓只保留为旧基线。
- 三指轴向偏移先保持相对关系，只对共同中心做有界投影；只有真实手形跨度超过对象抓持带
  才拒绝。
- 三边相容的三点组先按局部表面法向残差、再按点位残差选择，避免“点很近但闭合方向不对”。

## 同一 1488 规格重跑结果

- 完整表面 + 轴向投影：7 → 209 个六维姿态，耗时 183.80 s。
- 再采用法向优先选择：209 → 236 个姿态，耗时 194.11 s。
- 最大法向残差中位数从 1.1958 rad 降至 1.0214 rad，最小值从 0.7007 rad 降至
  0.2527 rad；这是生成覆盖改善，不是抓取成功。
- 236 个姿态经过便宜层级后有 4 个代理区间幸存；220 个完成原始网格/桌面/任务/有界 IK
  复核，得到 5 个 12 N 名义研究输入。五者全部未通过完整误差/六方向鲁棒门，也没有完成
  整臂路径碰撞检查。

## Top-3 与 Isaac 实际结果

- `q05_a13` 和 `q09_a08` 在第一指端点规划时失败：任务指腹没有在非任务几何边界前形成
  运动相容接触，因此未启动 Isaac。
- `q09_a13` 通过端点、0.5 s 初始状态和预构型检查；随后第一指以 0.18 rad/s 上限闭合，
  Isaac 物理推进 4.2 s，共 504 步。
- 第一指最大速度 0.17910 rad/s、最大等效关节力矩 0.05273 N·m、最大单周期目标变化
  0.0015 rad；未触发 3 rad/s、受力或跟随误差安全门。
- 原始网格评价在第 489 步开始出现运动相容 TASK 面近邻，但到最后安全端点仍有
  `hand_object=0` 个 PhysX 接触；第 503 步非任务手面进入不可执行边界。
- 第二、第三指、共同预紧和抬升命令均为 0；连接器最大位移仅 0.135 µm，离桌、50 mm
  抬升和 2 s 保持都没有发生。

## 证据边界和授权

- 当前证据：生成器漏斗诊断、离线原始网格/任务评价、有界 IK、第一指研究型动态失败。
- `research_dynamic_pass=false`、`formal_dynamic_pass=false`、`hardware_authorized=false`。
- 本轮没有磁吸、隐藏固定、物体位姿写入或在线对象/接触真值控制。
- q09_a13 的永久主分类是非 TASK 几何先于 TASK 接触；接触基础设施故障是独立结论。
  本次正对照失败、离线表和生成文件都不证明抓取、抬升、保持或正式动态成功。

## 验证、进程与 Git

- 本任务两个小脚本编译通过；最终正对照记录 6 个物理步，4 候选离线表已生成；未运行完整测试套件。
- 当前没有 Isaac、GraspGenX 或 CONTACTOPT 活动进程。
- `scripts/carts_v2/audit_nailfree_graspgenx_seed_reuse.py` 是无关未跟踪资产，不得暂存。
- 本任务结果：`artifacts/carts_v2/contactopt_1488_fast6h/contact_telemetry_positive_control/result.json`
  和 `artifacts/carts_v2/contactopt_1488_fast6h/remaining_candidate_task_first/result.json`。
- 本任务尚未推送；只允许暂存本任务两个脚本和本上下文，禁止包含上面的无关未跟踪资产。

## 最小读取路线

1. `AGENTS.md`
2. 本文件
3. `docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`
4. `docs/carts_v2/DECISIONS_CN.md`
5. `artifacts/carts_v2/STATE.json`
6. `artifacts/carts_v2/contactopt_1488_fast6h/MANIFEST.json`
7. 当前 Git 状态、进程和上面列出的原始产物
