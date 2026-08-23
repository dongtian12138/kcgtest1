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
