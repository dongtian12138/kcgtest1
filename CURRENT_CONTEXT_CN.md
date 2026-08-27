# kcgtest1 当前轻量上下文

> 快照时间：2026-08-27T08:21:56Z
> 当前分支：`carts-grasp-contactopt-1488-fast6h-20260826`
> 原始源码、配置、产物、Git、进程和时间戳优先于本摘要。

## 当前唯一优先级

当前任务 `DIRECT_GPU_CONTACT_REPORT_CLASSIFICATION + GPU_NATIVE_RIGID_CONTACT_VIEW`
已达到本轮 2 次 Isaac 运行上限并停止。结论与边界如下：

- `q09_a13` 永久保持 `REJECTED_NON_TASK_GEOMETRY_PRECEDES_TASK_CONTACT`，
  `task_first_margin_m=-0.0005071808379559516`；本轮没有运行 q09 或候选扫描。
- 只读预检与 primitive 运行时回读一致：`/physics/suppressReadback=true`、
  `/physics/cudaDevice=0`、physics simulation device=`cuda:0`、
  `physxScene:enableGPUDynamics=true`、`physxScene:broadphaseType=GPU`。GPU device 与
  suppress-readback 在 `World` 构造前启用；tensor SimulationView 按 Isaac 生命周期在
  `world.reset()` warm-up 内创建。
- 唯一一次 GPU box primitive 使用 native `RigidContactView`，没有 callback、CPU poll 或
  ContactSensor：A/B/C 为 59/1/9 步，正确 sensor/filter pair 的 count 非零，position、normal、
  force、由 `force*dt` 得到的 impulse 和 separation 均为有限值，求解器碰撞响应同时成立。
- 当前最终分类为 `DIRECT_GPU_CPU_CONTACT_REPORT_UNAVAILABLE_EXPECTED`；这只证明当前
  suppress-readback/DirectGPU 配置下应走 GPU-native 张量路径。CPU PASS 仍不能替代正式 GPU，
  primitive PASS 也不证明抓取、真实资产接触、抬升或保持成功。
- 第 2 次尝试把冻结资产的 `f1Link3_compound_hull_63` 与对象 `Hull_062` 复制为两个独立刚体；
  进程在创建 Kit log 和 PhysX 实例前卡住，300 s 后退出 124。它没有改写结果 JSON，故分类为
  `ISAAC_STARTUP_TIMEOUT_BEFORE_PHYSX_INSTANCE_NOT_A_HULL_CONTACT_FAILURE`；真实 hull 接触仍未验证。
- 本轮预算 2/2 已耗尽，脚本入口已阻止第三次 GPU-native 运行。不得继续 callback/ContactSensor
  调试，不运行 q09、候选或 1488，不改 Surface V2、候选位姿、控制器、contactOffset/restOffset
  或物理标准。

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

- 当前脚本 420 行；相对 `c536a83` 新增 154 行、删除 34 行，满足新增代码不超过 200 行；
  `py_compile`、`git diff --check` 和结果 JSON 的 `jq` 解析通过，未运行完整测试套件。
- 本轮第 1 次 Isaac 运行的 GPU-native primitive gate 通过；第 2 次在 PhysX 实例前启动超时。
  当前没有 Isaac、GraspGenX 或 CONTACTOPT 活动进程，也不允许第三次运行。
- `scripts/carts_v2/audit_nailfree_graspgenx_seed_reuse.py` 是无关未跟踪资产，不得暂存。
- 当前 HEAD/远端均为 `c536a83`。结果仍只写入
  `artifacts/carts_v2/contactopt_1488_fast6h/primitive_contact_report_isolation/result.json`；该文件被
  `.gitignore` 忽略，提交时只可对这一文件使用 `git add -f`。
- 本轮新提交尚未创建/推送；只允许暂存本脚本、本上下文和上述单个结果 JSON，禁止包含无关
  未跟踪资产或 Kit 日志。

## 最小读取路线

1. `AGENTS.md`
2. 本文件
3. `docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`
4. `docs/carts_v2/DECISIONS_CN.md`
5. `artifacts/carts_v2/STATE.json`
6. `artifacts/carts_v2/contactopt_1488_fast6h/MANIFEST.json`
7. 当前 Git 状态、进程和上面列出的原始产物
