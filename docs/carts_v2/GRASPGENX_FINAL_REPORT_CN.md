# GraspGenX-CARTS 路线一事实报告

> 事实冻结时间：2026-08-24T21:12:36Z
> 分支：`carts-grasp-graspgenx-route1-20260824`
> 双对象同源离线运行提交：`db540b3e5b85fd33cbc03d3265aa22ef1372eaae`
> 当前证据提交：`ef03cb2`
> 本报告不把退出码、覆盖图或静态测试写成物理抓取成功。

## 一、最终一句话结论

官方 GraspGenX、GPU、5 个 KCG 三指手描述器和双对象六维提案链已经真实运行，但对象 A 的 3 个三指允许接触预测全部在闭合路径中碰桌，对象 B 没有形成三指允许接触预测，因此当前没有候选可安全进入机械臂规划或 Isaac，50 mm 抬升和 2 s 保持尚未发生。

## 二、机器人和连接器实际完成了什么

### 对象 A：`current_d38999_26kj61sn_public_spec`

- GraspGenX 对 5 个固定手型共提出 `17,480` 个原始六维姿态；工作区可见 `15,704` 个，每个描述器固定保留 128 个，再合并为 256 个正式离线提案。
- 256 个提案中包含 26 个 diffusion 提案和 230 个 OBB 提案；只有 3 个 OBB 提案预测出三块允许指腹接触。
- 其中两个在逐指闭合中间状态扫入桌面，最小间隙均约 `-4.058 mm`；另一个在预抓状态已经约 `-118.790 mm` 穿入桌面。
- 因而完整闭合桌面安全数为 0；全手—对象 FCL、12 N 名义任务、鲁棒任务、机械臂 IK 和 Isaac 都没有被允许继续。
- 没有机械臂运动，没有动态三指接触，没有离桌、抬升或保持数据。

### 对象 B：`te_deutsch_d38999_26fj35pn_step`

- 同一模型、5 个描述器、随机种子、预算和处理顺序同样得到 `17,480 → 15,701 → 640 → 256` 个提案。
- 最终 256 个提案同样由 26 个 diffusion 和 230 个 OBB 提案组成，但三指闭合预测存活为 0。
- 首个失败原因为：第一指没有允许接触 164 个、禁抓面先接触 51 个、第二指没有接触 39 个、第三指没有接触 2 个。
- 因此桌面扫掠、FCL、12 N 任务、机械臂 IK 和 Isaac 都未到达。
- 没有机械臂运动，没有动态三指接触，没有离桌、抬升或保持数据。

## 三、最优抓取算法现在怎样工作

1. 官方预训练 GraspGenX 在隔离 Python 环境中读取登记允许表面点云，并对 5 个对象无关 KCG 手型提出完整六维手掌姿态。
2. 完整对象网格没有从碰撞模型删除；允许表面点云只负责引导提案。
3. 每个手型先固定保留 128 个方向分层姿态，五个手型再按六维多样性合并为最多 256 个候选。
4. KCG 真实 mimic 关系和指腹面用于预测三根手指依次闭合、首次允许指腹接触即停。
5. 每个独立关节相邻检查状态不超过 `0.0015 rad`，整手每一步都检查有限桌面。
6. 后续 FCL 薄封装检查非相邻自碰和非指腹部位碰物，不重写碰撞引擎。
7. 只有几何门通过后，才求解 12 N 名义任务及固定误差场景下三个方向的力和三个方向的力矩平衡。
8. 排序先看物理任务余量、单指力、关节/腕部利用率和路径间隙，GraspGenX 分数只作后置并列信息。
9. 只有任务、IK、接近和抬升路径都通过的候选才进入可执行 Top-3。
10. 当前 A/B 都在这条链的更早几何门关闭，所以不存在“最不差失败项冒充第一名”的情况。

## 四、关键数值

| 指标 | 对象 A | 对象 B |
|---|---:|---:|
| 原始六维提案 | 17,480 | 17,480 |
| 每描述器保留后 | 640 | 640 |
| 合并候选 | 256 | 256 |
| diffusion / OBB | 26 / 230 | 26 / 230 |
| 三指闭合预测通过 | 3 | 0 |
| 完整闭合不碰桌 | 0 | 0（未到达） |
| 任务评价数 | 0 | 0 |
| 机械臂 IK/路径候选 | 0 | 0 |
| 可执行 Top-3 | 空 | 空 |
| 双对象同源离线内核时间 | 20.534 s | 37.183 s |
| 包含加载、报告和哈希的墙时 | 41.10 s | 141.96 s |
| 最差稳定余量 | 未评价 | 未评价 |
| 最大单指所需力 | 未评价 | 未评价 |
| Isaac 物理时间 | 未运行 | 未运行 |
| 抬升高度 | 不适用 | 不适用 |
| 保持时间 | 不适用 | 不适用 |
| 最大滑移/姿态变化 | 不适用 | 不适用 |
| 抬升后是否仍接触桌面 | 不适用 | 不适用 |

六维覆盖诊断在两对象上均通过：5 个描述器都有候选、手掌分布覆盖 4 个方位象限，roll/pitch/radial distance 均非恒定。它只证明搜索不再局限于旧轴对称姿态族，不证明候选抓得住。

## 五、证据等级

- 代码检查：通过定向接口、坐标变换、描述器、闭合和碰撞回归；不等于抓取成功。
- 收口复核：系统 Python 因没有 `python-fcl` 为 26/27；绑定项目现有 FCL 环境后同组为 27/27。缺包失败已保留，它不改变离线物理结论。
- 官方模型正对照：`robotiq_3f + banana` 返回 20/20 个有限六维姿态；证明模型、权重和 GPU 推理链可用。
- 离线算法：A/B 六维覆盖已验证；A 在桌面路径门失败，B 在三指闭合预测门失败。
- 研究型动态：未运行，`research_dynamic_pass=false`。
- 正式动态：未运行，`formal_dynamic_pass=false`。
- 硬件：未授权，`hardware_authorized=false`。

## 六、代码是否保持简洁

- 相对路线起点，生产源码物理新增 `2145` NLOC、删除 `76` NLOC、净增 `2069` NLOC，低于 `2200` 硬上限，余量 55 行；1500 目标例外已经书面记录。
- 本轮涉及的新/扩展源码最大物理行数为 `models.py=550`、`reporting.py=550`、`graspgenx_adapter.py=548`，均未超过 550 行硬上限。
- `controller.py` 现有 745 个物理行，基线已经是 728 行；本轮仅净增 17 NLOC，用于六维接近方向接线，没有借动态任务做无关重构。
- 复用官方 `run_planner_on_object`、现有 V2 FK/闭合/任务/selector、python-fcl 和既有 Isaac 控制器；没有复制 GraspGenX 或碰撞引擎。
- 没有新增 manager、contract、ledger、bridge、binder 或 certificate 框架；旧 H102 和大型连续碰撞器未修改。

## 七、学术价值

- 研究问题：跨机械手六维提案加真实三指手完整闭合和任务载荷重排，能否比轴对称窄搜索更容易找到双对象可执行抓法。
- 已形成的方法组件：对象无关的多预构型描述器、完整六维文件接口、真实 PAD 语义、完整控制步闭合筛选，以及研究/正式任务门分离。
- 已有对照：旧轴对称基线与 GraspGenX 的离线覆盖、闭合和路径结果；官方手对同一 A/B 点云的 diffusion 正对照。
- 当前观察：六维覆盖范围变宽，但未转化为路径安全候选；KCG 长指描述器超出本轮已审计官方描述器清单的尺度范围是一个受支持的可疑原因，但训练分布和唯一因果根源均未被证明。
- 不能声称：新算法、显著优于旧方法、GraspGenX 对 KCG 手零样本成功、任务鲁棒、双对象动态成功、P1–P4 已验证或任何硬件有效性。

## 八、Git 提交与远端

- `1552265` — `chore: open graspgenx route1 autonomous branch`，已 push。
- `dae886e` — `feat: add validated kcg three-finger graspgenx descriptors`，已 push。
- `247e159` — `build: bind official graspgenx inference environment`，已 push。
- `3621a59` — `fix: align graspgenx virtual base with proximal finger plane`，已 push。
- `db540b3` — `feat: replace production seeds with graspgenx 6d proposals`，已 push。
- `ef03cb2` — `test: record two-object graspgenx offline gate results`，已 push。
- 报告收口提交以本文件提交后的 `git log` 为准；禁止 force push。

远端分支：<https://github.com/dongtian12138/kcgtest1/tree/carts-grasp-graspgenx-route1-20260824>

## 九、仍未解决的问题

1. KCG 长指工作区明显超出本轮已审计的官方描述器尺度清单；这是否代表模型训练域外推、以及是否导致本轮失败，都仍未验证。
2. A 的少量三指接触提案仍让长指连杆扫桌；B 的提案不能形成三指允许接触。
3. 因几何门未通过，12 N 任务力学、机械臂 IK、接近/抬升路径、Isaac 动态和跨对象迁移都尚无本路线证据。
4. 当前离散碰撞步长与真实控制周期一致，但不是状态间连续数学证明。

## 十、本次只需理解的小知识

“六维覆盖”只说明手掌从更多位置和方向尝试，并不等于抓取可行。对应证据链是：候选姿态 → 关节闭合目标 → 登记网格的离线逐状态检查 → 指腹接触预测/连杆间隙 → 承重与抬升判据；本轮失败发生在离线网格检查这一环，所以后面的受力、IK 和 Isaac 必须保持未评价。

## 原始证据入口

- 对象 A：`artifacts/carts_v2/graspgenx/offline_A/CURRENT_FINAL_INDEX.json`
- 对象 B：`artifacts/carts_v2/graspgenx/offline_B/CURRENT_FINAL_INDEX.json`
- 动态未运行：`artifacts/carts_v2/graspgenx/dynamic_A/NOT_RUN.json`、`dynamic_B/NOT_RUN.json`
- 版本与哈希：`artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json`

## 复现命令与时间证据缺口

双对象运行时没有把实际 shell 命令和开始时间直接写入 RUN_RECORD；开始时刻只能由“完成时刻减墙时”推得，A 约为 `2026-08-24T21:01:17.304821648Z`，B 约为 `2026-08-24T21:01:39.342992609Z`。这是一项证据链缺口，不影响已哈希的 result/coverage 内容，但不得冒充直接记录。

下面是由已绑定参数重建的等价复现命令，不声称是原运行逐字符录制：

```bash
PYTHONPATH=src/kcg_connector python3 scripts/carts_v2/run_graspgenx_offline.py \
  --config src/kcg_connector/config/carts_graspgenx_route1.yaml \
  --baseline-config src/kcg_connector/config/carts_grasp_v2.yaml \
  --integration-manifest artifacts/carts_v2/graspgenx/INTEGRATION_MANIFEST.json \
  --object-manifest artifacts/carts_v2/graspgenx/objects/object_manifest.json \
  --proposal artifacts/carts_v2/graspgenx/proposals_allowed_surface_roi_v1/OBJECT.json \
  --object-id OBJECT_ID \
  --output-dir OUTPUT_DIRECTORY
```

其中 A/B 的 `OBJECT.json`、`OBJECT_ID` 和 `OUTPUT_DIRECTORY` 分别取最终索引登记的对象名与 `offline_final_db540b3_A/B`。由于当前 manifest 是运行后的更新版本，其自身哈希与 RUN_RECORD 记录的运行时 manifest 哈希不同；实际输入身份仍由 RUN_RECORD 中的 `integration_manifest_at_run=033d919d…` 及各文件 SHA-256 绑定。
