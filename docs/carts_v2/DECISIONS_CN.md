# CARTS-Grasp V2 决策记录

## 2026-08-23T14:41:38Z — NORTH_STAR_CHANGE

- 用户明确把当前阶段改为 CARTS-Grasp V2：真实表面候选 → 任务载荷鲁棒 Top-3 → Isaac 研究型动态。
- 物理成功改为离桌后抬升至少 50 mm、保持至少 2 s；旧 40 mm 口径退出当前主线。
- 旧 H102 路线冻结为 reference，严禁继续连续编号式局部修补。
- `hardware_authorized=false`、在线真值隔离、禁磁吸/隐藏固定/物体位姿写入保持不变。

## 2026-08-23T14:44:00Z — 证据线分离

- 旧正式控制面的 `dynamic_launch_allowed=false`、正式候选为空和 `NOT_CERTIFIABLE` 保持不变。
- 新 V2 只建立独立 `research_dynamic_gate`；它满足有限速度/力量、初始无明显穿透和在线真值隔离后，才允许研究运行。
- 研究运行无论成功与否都不回写为正式动态或硬件结论。

## 2026-08-23T14:52:11Z — ARCHITECTURE_CHANGE

- 问题：旧高层优化器对全部候选绑定严格 clearance，旧路线碰撞器还写死 40 mm，与 V2 顺序及 50 mm 冲突。
- 证据：两名监督分别核对公开力学、排序、碰撞和 Isaac 入口后均只作有条件批准。
- 替代：只复用公开模型、表面采样、Sobol、LP 与尾部统计；V2 自己做一次七项字典序调度。
- 允许面：`models.py` 维护唯一逐面语义掩码；映射未闭环则不生成候选，不能把全部 external 面当允许面。
- 碰撞：唯一薄适配器无法直接验证 50 mm 时返回 `UNRESOLVED_INTERFACE_MISMATCH`，正式线失败关闭，研究线按独立门推进。
- 力语义：另求 `lambda=1` 的最小单指负担；旧极限 lambda 处峰值力不得写成规定任务所需力。
- 时间影响：减少旧证书依赖，预计节省超过 60 分钟；不改变 50 mm、2 s、真值隔离和双对象同参数。
- 监督结论：Academic 与 Complexity 均 `APPROVE_WITH_REQUIRED_SIMPLIFICATIONS`。

## 2026-08-23T15:55:41Z — LOCAL_CORRECTION

- 高维 Sobol 不能把 `bits` 设成场景数的最小指数；该组合在新 SciPy 中会退化，旧离线报告已撤回并重算。
- 冻结数值环境为项目 `.venv`，26×16 设计完整写入结果 JSON，SHA-256 为 `6aea2d6e5bc384445e1accf7636435142356eaf96a425481d17299a5cdcd93c1`。
- “对象位姿误差”收窄为实际实现的共同接触点平移和重力方向误差，不声称完整手—物体刚体位姿误差。
- URDF `100 N·m` 归类为 `UNKNOWN_UNCALIBRATED_MODEL_VALUE`；只报告实际广义关节力矩，不声称关节或腕部硬件利用率。
- B 的 `candidate_46` 只通过离线任务门；IK、路径碰撞、初始穿透和控制器未闭合前，研究动态门保持 false。

## 2026-08-23T16:51:24Z — LOCAL_CORRECTION

- run04 在 Kp=400 时第二关节误差 0.12044 rad 对应 48.19 N·m；再等待 1 s 后仍为 0.11947 rad / 47.55 N·m，确认是重力下的刚度稳态误差。
- Academic 与 Complexity Supervisor 一致批准仅把 `arm_stiffness` 改为 2500；数值来自 48.19/0.020=2409.5 后向上圆整，不是盲扫参数。
- Kd=40、100 N·m 限力、速度/跟踪门、轨迹、100 mm 上方净空及 A/B 共用配置全部冻结。
- 2500 只归类为 `SIM_OPERATION_CAP`，仍是带阻尼和限力的位置/刚度控制，不声称重力补偿、硬件刚度或学术创新。
- 本次只允许一次 A 预飞；失败即停止 Kp/Kd 调参并转入公开重力前馈重设计。

## 2026-08-23T16:57:35Z — MILESTONE_REORDER

- run05 已让 A 到达上方稳定门：最大臂误差 0.01768 rad、最大臂力矩 45.38 N·m，未触及 100 N·m 上限。
- 下降第 146/240 步出现 f2Link2 凸包—桌面接触，随后 f2j1 速度 4.628 rad/s 触发安全急停；当前根因从控制稳态误差转为 A Top-1 路径几何冲突。
- A Top-2/3 的任务载荷余量为空，不能因为更高就替代任务质量 Top-1；该试验后置为路径消融。
- 两位监督者一致批准冻结 A 推导的所有主要参数，先集成唯一通过离线门的 B Top-1 自由 STEP 同源资产。
- B 只先做预飞；通过不等于抓取成功，失败不得回调高度、增益、门限或候选编号。
### 2026-08-23：PARK 三点刚体对齐候选族

- 类型：`LOCAL_CORRECTION`
- 证据：同一 384 样本/48 候选只读原型中，A/B 闭合幸存分别 5/4，但桌面快筛幸存均为 0；三点刚体拟合残差约 9.0--20.4 mm。
- 决定：保留高度条件化 yaw-only 生成器作为当前失败基线；停止倾角、预抓相位与三点匹配补丁，不做第三次候选姿态局部修改。
- 研究动态边界：当前 A/B Top-1 均可平衡名义重力与抬升峰值，只允许无扰动研究观察；完整误差/扰动任务门仍失败关闭。

## 2026-08-23T18:22:00Z — LOCAL_CORRECTION

- 用户插话要求澄清“是否只是碰撞检测没做好”；本次不改变总目标或成功标准。
- 旧 A `candidate_07` / B `candidate_46` 确实因快筛未覆盖真实手部网格—有限桌面路径而漏检；该缺口已由真实碰撞 STL 的接近、预抓和顺序停靠快筛补上。
- 当前 A `candidate_11` 规划净空约 `+0.705 mm`，但 Isaac 中约 `0.015 rad` 的重力稳态跟踪误差使手基实际低约 `9 mm`；5 个笛卡尔高度 IK 路点仍在同一下降末端碰桌。
- 当前根因因此是“已规划净空小于控制跟踪误差带”，不再是单纯的路径插值或快筛漏检。
- 下一唯一改变为公开姿态相关重力补偿；候选、路径、速度、力量上限、真值隔离和 A/B 同参数均保持。

## 2026-08-23T18:25:00Z — LOCAL_CORRECTION

- 用户询问 MoveIt 现成运动规划与碰撞检测能否解决当前困境；本次不改变总目标。
- 仓库已有 `kcg_moveit1` 的 ROS 2 / MoveIt 2 配置，但 V2 尚未把真实连接器、有限桌面、夹具和最新手部碰撞语义接入其规划场景。
- MoveIt 可作为后续 Home—预抓的全机器人避障路径后端，但不解决重力稳态跟踪误差、三指有效接触身份或任务承载排序。
- 当前先完成同参数重力补偿区分实验；若跟踪误差闭合后仍是路径中段冲突，再以真实 V2 规划场景接入 MoveIt，不复用旧圆柱场景。

## 2026-08-23T18:47:00Z — ARCHITECTURE_CHANGE

- 问题：公开 Newton `ArticulationActuators` 的两次运动前接线尝试分别失败于 DOF 顺序严格校验和 CPU 参数—`cuda:0` 物理视图设备不匹配；两次均未发送轨迹、未发生碰撞。
- 两次规则：停止 Newton 接线补丁，不修第三次设备配置。
- 最简单替代：删除 Newton actuator 路径，只保留一个公开 experimental `Articulation` 视图和原生有限力位置驱动；七轴命令为 `q_drive = q_nominal + g(q) / Kp`。
- 物理边界：只能写“控制律代数等价”，Isaac 离散实现待证；跟踪误差仍对比未偏置的 `q_nominal`，偏置、预测总力矩和关节限位分开记录。
- 冻结不变：前馈比例固定 1.0，A/B 同参数；Kp=2500、Kd=40、100 N·m、3 rad/s、轨迹、候选、场景、dt、seed 和真值隔离不变。
- 停止条件：只允许一次 A 有界预飞；若视图/shape/设备不一致、偏置越限、总力矩饱和、稳态误差不降或仍碰桌，立即 `PARK`，不再换 API 或调参。
- 监督结论：Academic 与 Complexity 均 `REDESIGN_APPROVE_WITH_FAIL_CLOSED_CONDITIONS`。

## 2026-08-23T20:34:00Z — PARK 动态对象运行

- 唯一 A 原生重力预飞真实推进 2.5667 s；机器人—桌面、夹具、连接器接触均为 0，机械臂最大跟踪误差 0.00735 rad。
- `f2j1` 在上方接近第 68 步由平滑目标 0.01250 rad 附近突增至 4.4916 rad/s，触发预登记 3 rad/s 急停；未进入预抓、闭指或抬升。
- 重力项最大 1.527 N·m、等效偏置最大 0.000611 rad、预测总力矩最大 12.26 N·m；无 100 N·m 饱和或关节限位越界。
- 本次启动时 NVIDIA 驱动不可枚举，PhysX 退到 CPU；旧比较运行使用不同后端，因此不能把机械臂误差差异严格归因于重力偏置。
- 失败类登记为 `HAND_PRESHAPE_SPEED_ABORT_WITH_CPU_FALLBACK`；PARK 重力接口、Kp/Kd、候选补丁以及当前条件下继续 A/B 对象运行。
- MoveIt 后置：当前触发点是自由空间手指动力学，不是机械臂路径或环境碰撞。
- GPU 恢复后唯一允许的区分实验为无对象、无桌面的同轨迹 `compare_cpu_vs_gpu_hand_drive`；通过前不得再次运行 A/B 候选。
- Academic 与 Complexity Supervisor 均 `VETO_PREFLIGHT_PASS_AND_PARK_OBJECT_DYNAMICS`。

## 2026-08-23T20:48:13Z — 动态失败证据边界更正

- run05 trace 直接证明的失败类别是 `HAND_PRESHAPE_SPEED_ABORT`：`f2j1=4.4916 rad/s` 超过 3 rad/s 并急停，且没有进入闭指、抬升或保持。
- NVIDIA 不可用和 PhysX CPU fallback 来自启动环境观察，未绑定进 trace；改记为 `CPU_FALLBACK_CONTEXT`，不得把它写成已证根因。
- 速度异常根因保持 `UNRESOLVED_HAND_MODEL_OR_PHYSICS_BACKEND`；GPU 恢复后的同轨迹无对象/无桌面对照仍是唯一允许的后续动态区分实验。
- 未归档的单一名义场景排序不作为 P3 证据；P1--P4 当前均未验证。

## 2026-08-23T21:01:32Z — 根因范围与迁移对象更正

- B 是见过几何的迁移对象，不是前瞻盲测或严格留出集。
- 根因更正为 `UNRESOLVED_HAND_DRIVE_MODEL_OR_BACKEND`，覆盖驱动接线、手模型/mimic 约束和物理后端。

## ARCHITECTURE_CHANGE — 共享 IK 的源码总量目标例外

- 问题：V2 core 为 3371 NLOC；完整计入共享 IK 后为 3622，超过 3500 目标 122 行。
- 证据：`candidate_joint_route.py` 1160→1004；aggregate 872→819；roster 803→854；共享 IK 251，robust 支撑净增 93。
- 最简单替代：一个公开 bounded IK 同时供 legacy 与 V2 调用，不保留两套求解器，也不再拆 bridge/state 模块。
- 时间影响：已完成，无新增动态运行；controller 597 NLOC 冻结，后续加功能前必须先删减。
- 监督结论：Complexity `APPROVE_WITH_WRITTEN_EXCEPTION`；未降低物理门、跨对象复用或真值隔离。

## 2026-08-23T21:42:11Z — Git 只读环境的集成交付

- 原工作区执行 `git add` 时失败：`.git/index.lock` 位于只读文件系统；未修改或破坏原 Git 数据。
- 从同一 `698a2bb` HEAD 建立 `/tmp/kcgtest1-carts-v2-integration-20260823`，以原工作区作为内容源精确暂存本轮路径。
- 代码与离线数据提交为 `4d90a52`；原工作区 HEAD 仍为 `698a2bb`，因此 IDE 会继续显示工作树修改。
- 远端推送结果在文档提交后记录；禁止把临时克隆机制解释成原工作区已同步。

## 2026-08-23T21:44:03Z — 远端失败与可恢复交付

- 文档提交为 `58c9503`；推送 `carts-grasp-v2-rebuild-20260823` 失败，原因为 `Couldn't connect to server`，未发生远端写入。
- 本地保留 `/tmp/kcgtest1-carts-v2-integration-20260823`；另生成增量 bundle，要求起点 `698a2bb`，包含到 `58c9503`。
- bundle 为 802,864 B，SHA-256 `b22d65dd19d7b468ac7a1d0cfe2a8c46f9b784d1473f884b93124a9f44c3a9d1`，已通过 `git bundle verify`。
- 网络恢复后只允许普通 push 同名 V2 分支，禁止 force push；原工作区同步仍需恢复 `.git` 写权限。

## 2026-08-24T06:42:18Z — STOP：12 小时硬截止

- 本轮硬截止为 `2026-08-24T02:41:38Z`；复核时已经超时 4 h 00 min 40 s，因此停止新增功能、Isaac 运行和 GPU/CPU 区分实验。
- 截止时只达到双对象离线排序和真实 Isaac 预飞失败证据；三指闭合、离桌、抬升 50 mm、保持 2 s 和扰动均未完成，不能写 `RESEARCH_DYNAMIC_PASS`。
- 截止后唯一执行的是交付修复：只读确认原工作区内容与临时集成 tip 完全一致，保留 `698a2bb` 恢复分支，再无历史改写快进至 `57efc16`。
- `57efc16` 已用普通 push 同步远端 `carts-grasp-v2-rebuild-20260823`；没有 force push，没有启动仿真，没有改变对象、参数、成功门或 `hardware_authorized=false`。

## 2026-08-24T07:05:10Z — MILESTONE_REORDER + EXECUTION_WINDOW_REOPEN

- 用户明确重开补足窗口 `V2_RECOVERY_1`，最长 5 小时，硬截止为 `2026-08-24T12:05:10Z`；这不是新的 12 小时窗口，也不是 `NORTH_STAR_CHANGE`。
- 北极星、两个对象身份、50 mm、2 s、真实指腹接触、3 rad/s 安全限速、在线真值隔离和禁止未授权穿透全部保持。
- `hardware_authorized=false`、`legacy_formal_dynamic_launch_allowed=false`、`formal_dynamic_pass=false` 保持失败关闭。
- 研究门先只开放 `ISOLATED_HAND_DIAGNOSTIC`：无对象、无桌面、同模型/初值/轨迹/dt/drive 的手部区分实验；在 `f2j1` 根因关闭前禁止对象 A/B 运行和候选算法修改。

## 2026-08-24T08:13:09Z — LOCAL_CORRECTION + EXECUTION_CONTINUATION

- 用户明确纠正：GPU found/lost aggregate-pair 与 total aggregate-pair 容量属于 Isaac/PhysX 运行资源，不是抓取算法、物理参数、控制增益或成功标准；`V2_RECOVERY_1` 因此属于过度保守停止。
- 开启补足窗口 `V2_RECOVERY_2`，有效预算 4 小时 15 分钟，硬截止 `2026-08-24T12:28:09Z`，不自动延长，也不重开 12 小时计划。
- 先修复预检查失败关闭漏洞，再审计场景结构与两类容量峰值；没有重复碰撞体或错误过滤时，容量按 `next_power_of_two(ceil(2 * observed_peak))` 分别确定。
- `candidate_11`、`candidate_33`、算法排序、物性、摩擦、增益、力量、3 rad/s、50 mm、2 s、指腹标准、真值防火墙及旧 H102 全部冻结。
- `hardware_authorized=false`、旧正式动态门、`formal_dynamic_pass=false` 和 `research_dynamic_pass=false` 保持；只有新评价的 `accepted_preflight_pass=true` 才能启动对象 A 抓取。

## 2026-08-24T09:36:16Z — V2_RECOVERY_2 失败收口

- run17 在第二次有证据修复后，仍与 run15 同一步、同阶段触发 `f1j3` 超过 3 rad/s；该根因的两次尝试已用尽。
- 决定：`status=PARKED`、V2 研究动态门关闭；禁止第三次直接补丁、调增益/阈值或提高限速。
- 对象 A 未形成三指真实 PAD 接触，抬升和保持均为 0；对象 B 不满足启动前提，因此本窗口不运行。
- GPU 容量已验证健康，不再列为当前阻塞；正式动态、研究动态和硬件通过均保持 false。

## 2026-08-24T10:16:54Z — SUPERVISION_RULE_CORRECTION + ROOT_CAUSE_REVIEW

- 用户纠正：同一根因两次实现失败只触发根因复盘，不自动触发项目 `PARKED`；恢复 V2_RECOVERY_2 剩余 10313 s，不创建新窗口。
- `03594b8` 只影响接触确认后的后续保持周期，不能影响 run17 接触确认当周期的目标阶跃，因此不计作消除该阶跃的独立因果修复。
- 当前改为 `IMPLEMENTING`，里程碑为接触状态切换根因复盘；先使用 run15/run17 原 trace，连续性测试通过前不启动 Isaac。
- 3 rad/s、50 mm、2 s、物性、候选、控制增益和真值隔离均冻结；硬件、研究动态和正式动态通过继续为 false。

## 2026-08-24T10:36:33Z — 接触切换采用关节力代理的有界位置修正

- 删除所有接触阶段的 `target=measured_position`；确认周期输出保持上一命令。
- CONTACT_SETTLE/HOLD 只复用既有检测阈值与手部刚度，把去 tare 关节力代理误差换成每周期受限的位置修正，不新增增益或力量门限。
- 该控制量包含惯性、阻尼和约束反力，只称“关节力代理”，不声称是真实指腹法向力或完整力控。
- 只有新源码绑定预检查和第一指 0.5 s 事后接触评价通过，才允许继续完整三指运行；整末端凸包路径不能升级为 PAD patch 身份。

## ARCHITECTURE_CHANGE — 完整场景接触见证的临时规模例外

- 问题：对象场景超速而无对象同目标回放稳定，旧步后轮询无法区分真实碰撞、未分类路径和 GPU 接触缓冲时机。
- 最简单替代：在现有 evaluator 内并行记录事件回调与步后轮询，未分类接触和通道分歧失败关闭；不建新文件或框架。
- 规模：控制修复仍约 119 NLOC；见证约增 67 NLOC，Recovery 2 四个生产文件累计约 186 NLOC，低于 250 行窗口额度。
- 影响：三个动态文件暂超原 600 行目标；动态前重构会改变已绑定路径，故仅批准一次预检查和第一指诊断，取得证据后禁止继续叠加功能。
- 监督结论：Academic 与 Complexity Supervisor 均批准；在线真值隔离、3 rad/s、对象/候选/物性和正式门均未改变。

## 2026-08-24T11:46:11Z — ROOT_CAUSE_REVIEW 关闭冻结动态路径

- 注册网格重放确认 `candidate_11` 第一指闭合中段由 `f1Link3` 先扫入桌面；代理触发时 PAD 距连接器仍约 41.873 mm。
- 快筛只检查预抓、接近和接触停止端点，漏掉最深约 1.838 mm 的非单调中间扫掠。
- 两位监督者一致认为继续运行会重复已知未授权穿透；filtered-pairs 会隐藏真实冲突，明确禁止。
- 由于候选、快筛和路径在本窗口冻结，研究动态门关闭；这不是预算耗尽，也不改变 NORTH_STAR、50 mm、2 s、正式门或硬件授权。
