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
