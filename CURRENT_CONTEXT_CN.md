# kcgtest1 当前轻量上下文

> 快照时间：2026-08-27T07:51:42Z
> 当前分支：`carts-grasp-contactopt-1488-fast6h-20260826`
> 原始源码、配置、产物、Git、进程和时间戳优先于本摘要。

## 当前唯一优先级

当前任务 `PHYSX_PRIMITIVE_CONTACT_REPORT_ISOLATION + CONDITIONAL_REAL_ASSET_COMPARISON`
已达到 4 次定向 Isaac 运行上限并停止。结论与边界如下：

- `q09_a13` 永久保持 `REJECTED / NON_TASK_GEOMETRY_FIRST`，`task_first_margin_m =
  -0.0005071808379559516`；接触链修复不得使其重新晋级。
- 两个 40 mm box 在 120 Hz、GPU/CUDA 正式后端中自然相撞：动态 box 从 0.2 m/s 降至约
  0.000252 m/s，A/B/C 分别记录 59/1/9 步；两个 shape enabled、无 filter/group，实际每 shape
  `contactOffset=0.8 mm`、`restOffset=0`，未调用 offset setter。
- GPU 上 full callback、Isaac 6 core contact callback、physics-step 内 full poll 和 step 后 full poll
  均为 0。将订阅从未附着的旧接口切到当前 `omni.physics.core` 后，GPU 复跑仍为 0，分类为
  `PRIMITIVE_SOLVER_RESPONSE_WITHOUT_CONTACT_REPORT`。
- 完全相同 CPU 对照产生正确 box pair；full/basic/step 内 poll/step 后 poll 各有 9 个 header，首个
  header 有 4 个 contact data，路径、有限 separation 和非零 impulse 均可解码。因此最终根因层为
  `GPU_CONTACT_REPORT_CONFIGURATION_OR_BACKEND_SPECIFIC_FAILURE`；CPU PASS 不能替代 GPU。
- `CONTACT_TELEMETRY_UNVERIFIED` 继续保持。因 GPU primitive 未通过，ContactSensor raw、filtered
  reading、项目聚合、独立真实 hull 和 q09 step503 全部未运行；未生成 real-hull JSON。
- 当前不允许恢复候选筛选、q09、真实资产动态或控制器修改；不重跑 1488、不改 Surface V2、
  候选位姿、控制器或 offset。

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

- primitive 主脚本 300 行并通过 `py_compile`；GPU 修复后复跑与 CPU 对照均各记录 69 个物理步；
  未运行完整测试套件。
- 当前没有 Isaac、GraspGenX 或 CONTACTOPT 活动进程。
- `scripts/carts_v2/audit_nailfree_graspgenx_seed_reuse.py` 是无关未跟踪资产，不得暂存。
- 已有提交 `7cb9282` 已普通推送。当前结果为
  `artifacts/carts_v2/contactopt_1488_fast6h/primitive_contact_report_isolation/result.json`；该文件被
  `.gitignore` 忽略，提交时只可对这一文件使用 `git add -f`。
- 本轮新提交尚未创建/推送；只允许暂存 primitive 脚本、本上下文和上面的单个小型结果 JSON，
  禁止包含无关未跟踪资产或大型 Kit 日志。

## 最小读取路线

1. `AGENTS.md`
2. 本文件
3. `docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`
4. `docs/carts_v2/DECISIONS_CN.md`
5. `artifacts/carts_v2/STATE.json`
6. `artifacts/carts_v2/contactopt_1488_fast6h/MANIFEST.json`
7. 当前 Git 状态、进程和上面列出的原始产物
