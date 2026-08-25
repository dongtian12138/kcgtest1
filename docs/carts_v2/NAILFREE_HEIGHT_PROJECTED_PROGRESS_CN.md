# 无指甲高度投影路线当前进度

## 一句话结论

对象 A/B 已用同一源码和配置完成 91 角全池、12 N任务评价与 50 mm离散逆解；A 为 `40→32→12、IK 3/3`，B 为 `1→1→1、IK 1/1`。无指甲 PhysX 碰撞资产仍失败关闭，因此尚未启动任何连接器抓取，研究动态、正式动态和硬件结果全部为否。

## 三指手和连接器实际发生了什么

对象 A目前只在离线模型中发生了以下过程：程序改变手掌候选的世界 Z 高度，重新预测三指逐指闭合，在登记控制状态上检查整手与有限桌面、对象非允许面和手部自碰，然后计算接触力分配与机械臂离散逆解。

连接器没有在 Isaac 中被三指实际碰触，也没有离开桌面。50 mm只出现在机械臂逆解目标路点中，不是物体已经抬升 50 mm。

对象 B 也只经历了同一离线链；没有加载到 Isaac 抓取场景，也没有因 B 的结果改变候选、力上限、摩擦、高度或控制参数。

## 对象 A 的完整离线链

已验证计数为：

`5824 输入 → 40 高度投影后几何存活 → 32 保留 / 8 后置 → 32 个12 N名义合格 → 12 个误差场景合格 → Top-3 bounded IK 3/3可达`

逐项解释：

1. 91 个掌形角，每角固定 64 个 GraspGenX 输入，共 5824 个；全部候选先做高度可行性，不先截 Top-8。
2. 40 个候选在某个预登记预闭合组合下找到非空的 `H_table/H_reach_outer` 必要区间交集，并在投影后重新通过三指允许面接触预测和整手几何检查。
3. 每掌角最多保留 8 个后，共 32 个进入任务受力评价；另 8 个仅因预算被后置，不应称物理失败。
4. 32 个候选全部能在 12 N 单指操作上限下平衡名义重力与抬升任务。
5. 其中 12 个在当前 16 个预登记误差场景下仍达到最差任务余量不小于 1。
6. 按固定字典序得到的误差场景 Top-3 为 `graspgenx_726`、`graspgenx_498`、`graspgenx_874`。
7. 三个候选均生成了 5 个接近路点和 11 个 50 mm抬升路点，bounded IK 可达计数为 3/3。

## 对象 A 的关键数值

| 项目 | 当前数值 | 证据含义 |
|---|---:|---|
| 原始候选 | 5824 | 91角 × 每角64个 |
| 高度投影后几何存活 | 40 | 投影后重新接触预测和整手快筛存活 |
| 每角Top-8后保留 | 32 | 进入任务评价 |
| 预算后置 | 8 | 未证明失败，也未进入本轮任务评价 |
| 12 N名义任务合格 | 32 | 名义重力与抬升可平衡 |
| 误差场景合格 | 12 | 16个固定误差场景的最差余量不小于1 |
| Top-1最差任务余量 | 1.580186 | 离线接触力分配余量 |
| Top-1最大单指所需力 | 8.596684 N | 小于本轮12 N操作上限 |
| Top-3 bounded IK | 3/3 | 仅离散运动学路点可达 |
| 最大逆解位置误差 | 1.244e-12 m | 三条路由中的最大值 |
| 最大逆解姿态误差 | 1.014e-11 rad | 三条路由中的最大值 |
| 最大接近相邻关节步长 | 0.110362 rad | 离散路点差，不是控制周期指令 |
| 最大抬升相邻关节步长 | 0.017397 rad | 离散路点差，不是控制周期指令 |

Top-1 的已检查手部采样路径最小桌面净空为 0.005290 m，对应 `f3Link3` 的第三指闭合阶段。该值只覆盖报告明确列出的离散手部状态，不覆盖七轴机械臂完整路径或离散状态之间的连续碰撞。

## 对象 B 的完整离线链

`5824 输入 → 1 个高度投影后几何存活 → 1 个进入任务评价 → 1 个名义 12 N 合格 → 1 个误差场景合格 → bounded IK 1/1 可达`

唯一候选 `graspgenx_659` 的最差任务余量为 `1.012419`，较差 20% 场景平均余量 `1.095510`，最大单指所需力 `11.924125 N`，已检查手部采样路径最小桌面净空 `0.005293 m`。其 5 个接近与 11 个 50 mm 抬升离散路点可达，最大位置/姿态误差分别为 `7.48e-13 m` / `7.59e-12 rad`。

该余量接近 1，说明在当前离线误差范围内只剩很小的理论余量；它不是动态稳定或正式成功证据。

## PhysX 无指甲碰撞资产为什么仍失败关闭

无指甲末节的精确视觉网格已能被 Isaac 资产导入，但第一末节的 SDF 碰撞烹饪没有形成可接受的运行时碰撞绑定。PhysX 报告源网格不适合可靠 SDF：非流形、非水密、含重复三角面，自动修复失败，后续烹饪未得到可接受结果。

定向拓扑审计先移除 64 个完全重复面，剩余 11772 个面仍有 97 条非流形边：

- 56 条边连接 3 个面，无法在不删除或复制保留面的前提下唯一配对；
- 41 条边连接 4 个面；
- 因此不能把一次任意修补或凸包填槽当成原几何的等价碰撞资产。

随后按原始精确连通实体拆分，而不进行 round-weld。155 个组件中仅 152、154 两个组件闭合；承载全部任务抓持面的组件 0 和 75 分别仍有 43、19 条开放边。组件拆分能消除跨组件非流形边，却不能把开放表面变成具有确定内外的 SDF 实体，因此这条不近似路线也失败关闭。

最后一次有界边界环审查确认：组件0的43条开放边形成一个19边环和两个12边环，均不共享已删除指甲组件顶点；它们是原 STL 多壳/PAD—主体接缝。19边环跨度约20 mm且非平面，每条开放边都有两种方向相容源面，自动封口没有唯一 CAD 依据。组合看似相邻的组件后仍有9条非流形边和1380对非相邻面相交，因此没有生成生产碰撞资产。

当前事实字段保持：

- `f1_sdf_diagnostic_pass=false`；
- `collision_runtime_binding_accepted=false`；
- SDF 没有持久化为生产碰撞资产；
- 该失败是 Isaac 资产诊断，不是对象 A 抓取运行失败，因为对象 A 抓取尚未启动。

## 当前证据等级

| 层级 | 状态 | 已有证据 / 缺口 |
|---|---|---|
| 静态模型 | 部分完成 | 无指甲视觉网格和手侧语义已绑定；运行时碰撞资产未接受 |
| 离线几何 | 两对象完成 | A 40个几何幸存、B 1个；只覆盖登记的采样手部状态 |
| 离线任务 | 两对象完成 | A 32名义/12误差合格，B 1/1；12 N为仿真操作上限 |
| 离线机械臂运动学 | 两对象完成 | A Top-3为3/3，B Top-1为1/1离散路点可达 |
| 完整机械臂路径碰撞 | 未完成 | 手部快筛不覆盖七轴完整路径；旧后端只有40 mm且接口不匹配 |
| 研究型 Isaac 动态 | false | 没有连接器抓取、离桌、50 mm或2 s保持数据 |
| 正式动态 | false | 正式路径、重复、扰动和双对象证据未闭合 |
| 真实硬件 | false | `hardware_authorized=false` |

## 两对象高度投影汇总

| 指标 | 对象 A | 对象 B |
|---|---:|---:|
| 六维输入种子 | 5824 | 5824 |
| `H_table/H_reach_outer` 必要区间交集非空 | 170 | 88 |
| 必要区间交集为空 | 5654 | 5736 |
| 投影后精确复核失败 | 130 | 87 |
| 几何幸存 | 40 | 1 |
| 每角 Top-8 保留 | 32 | 1 |
| 保留候选平均上移 | 3.141 mm | 2.799 mm |

A 的几何幸存者只在约 0°–5°出现：约 1°为15个、约2°为9个，其余 0°/3°/4°/5°分别为2/5/4/5个；B 唯一幸存者在约4°。91个掌面角都运行了相同规则，不能把这个对象相关分布升级为普遍机械规律。

## 为什么 Top-3 仍不能称为可执行候选

当前 `result.json` 正确保持：

- `executable_top_k_count=0`；
- `selected_executable_candidate=null`；
- `offline_task_gate_passed=false`；
- `research_dynamic_gate_passed=false`；
- 严格路径后端没有被调用。

bounded IK 只回答“关节限位内能否到达这些离散位姿”。它没有回答：

- 机械臂各连杆沿接近路径是否碰桌、手、连接器或环境；
- 抓住连接器后，携带物体抬升 50 mm是否碰撞；
- 离散路点之间是否存在连续碰撞；
- 无指甲末节在 PhysX 中的实际接触形状是否与离线三角网格一致；
- 动态闭合时是否真的由三块允许内侧面接触并保持物体。

旧严格后端固定检查 40 mm抬升，而当前目标为50 mm，所以返回接口不匹配且没有执行；不能用旧40 mm证据补写当前门。

## 下一步及停止边界

1. 当前精确整网格 SDF、确定性局部拓扑归一化和精确组件分拆均失败关闭；后续需要回到 CAD/源拓扑级构造可追溯闭合碰撞实体。
2. 不能通过填回指甲拆卸槽、扩大容差或改用会覆盖禁区的粗凸包来跑通。
3. 只有运行时资产绑定、完整七轴机械臂接近/闭合/携带抬升路径和独立预检查全部通过，才允许创建真正的执行候选。
4. 之后才能在 Isaac 中观察三指实际接触、离桌、50 mm抬升和至少2 s保持；运行后评价与在线控制真值继续隔离。

## 状态快照与原始证据

本进度快照依据提交 `070ddca69d0c85b4e36f051ff64594f136ccf781` 及以下原始文件整理：

- `artifacts/carts_v2/STATE.json`
- `artifacts/carts_v2/nailfree_height_projected/offline_A/full_palm_cascade_audit.json`
- `artifacts/carts_v2/nailfree_height_projected/offline_A/result.json`
- `artifacts/carts_v2/nailfree_height_projected/offline_kinematic_routes/object_A_bounded_ik.json`
- `artifacts/carts_v2/nailfree_height_projected/offline_B/full_palm_cascade_audit.json`
- `artifacts/carts_v2/nailfree_height_projected/offline_B/result.json`
- `artifacts/carts_v2/nailfree_height_projected/offline_kinematic_routes/object_B_bounded_ik.json`
- `artifacts/carts_v2/nailfree_height_projected/OFFLINE_CROSS_OBJECT_SUMMARY.json`
- `artifacts/carts_v2/nailfree_height_projected/hand_model_audit/sdf_runtime_preparation/SDF_RUNTIME_F1_FAILURE_EVIDENCE.json`
- `artifacts/carts_v2/nailfree_height_projected/hand_model_audit/topology_normalization_f1/TOPOLOGY_NORMALIZATION_AUDIT.json`
- `artifacts/carts_v2/nailfree_height_projected/hand_model_audit/COMPONENT_SDF_READONLY_AUDIT.json`
- `artifacts/carts_v2/nailfree_height_projected/hand_model_audit/BOUNDARY_LOOP_CAP_FEASIBILITY.json`

授权和成功字段在本快照中保持：

- `hardware_authorized=false`
- `research_dynamic_pass=false`
- `formal_dynamic_pass=false`
- `legacy_formal_dynamic_launch_allowed=false`

这些字段只能由相应真实证据关闭后改变，不能由文档、离线结果、退出码或逆解成功改变。
