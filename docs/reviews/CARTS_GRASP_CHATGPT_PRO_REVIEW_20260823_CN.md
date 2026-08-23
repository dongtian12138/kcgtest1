# CARTS-Grasp 当前算法审查入口

> 截止时间：2026-08-23T12:47:40Z
>
> 当前任务：`CARTS-GRASP-CROSS-OBJECT-V1`
>
> 结论级别：源码、静态测试和离线计时快照；不是 Isaac 动态验收或硬件验证

## 请先看这一段

机器人现在还没有执行抓取。算法已经能用同一套方法为 public-spec D38999 和
TE/DEUTSCH J35 建立对象、三指手、接触范围、受力和 11 关节路线合同，但正式候选生成
仍停在完整连续碰撞检查。`selected_candidate=null`、
`dynamic_launch_allowed=false`、`hardware_authorized=false`。

当前最需要审查的不是“摩擦系数再调多少”，而是一个保守连续碰撞算法的正确性和计算量：

1. 三指闭合过程中，每个运动三角形与对象三角形先用 BVH/AABB 粗筛；
2. AABB 不能分开的三角形对，再用完整 17 轴分离轴检查；
3. 仍不能证明分离的时间区间二分，并把未解决的三角形对传给子区间；
4. 只有整段时间都被严格证明无禁止碰撞，候选才可继续；任何数值不确定都失败关闭。

这相当于控制系统里的“安全判据”计算太慢：传感器/模型输入已经准备好，但在发出动作前，
安全监控器需要反复证明上百万组三角形对不接触，导致候选无法在预算内完成。

## 已验证事实

| 项目 | 当前事实 | 证据边界 |
| --- | --- | --- |
| 便宜初筛 | 256 个候选约 11.99 秒，固定 Top‑4 为 155、174、14、198 | 离线排序，不证明抓取 |
| 精确接触 | Top‑4 单个精确接触约 3.8–5.5 秒；174 因第三指无首次 PAD 接触而拒绝 | 静态接触合同 |
| H93 | Top‑4 曾在 65.09 秒自然返回，candidate 14 的 `f1Link3` 为 2.84 秒 | 后续 H94 发现接触面身份不一致，因此不能作为正确实现的性能基线 |
| H94/H95 | V9 在 2479 面蓝色 PAD 表面选接触；旧终端碰撞检查只认可另一封闭表面的 16 个共享面 | 已定位为物理语义绑定错误 |
| H96 | 将“允许接触的 PAD skin”和“用于禁止穿透的 closed shell”分开；155 的 `f1Link3` 7.65 秒，14 为 36.85 秒 | 正确性保持，但 120 秒内 Top‑4 未完成 |
| H97 | 复用父区间未解决对；155 的 `f1Link3` 4.39 秒，14 为 21.22 秒 | 比 H96 快约 1.74 倍，仍未达 10 秒/90 秒目标 |
| H98 | 预计算静态轴并分三阶段压缩 survivor | 155 变慢到 5.18 秒，14 在剩余 27.39 秒内仍未完成 |
| H99 | 只有 AABB survivor 数量减半或到终点才运行完整窄相检查 | 155 单一 `f1Link3` 已超过 33.49 秒，比 H98 至少慢 6.47 倍 |
| H100 | 固定合成例证明：延迟窄相会把本可在父区间排除的对带入两个子区间，工作量从 1 次变 2 次 | 已识别结构性根因，H99 路线应放弃 |
| H101 | 已实现“每次细分前都完整窄相 + 预计算两侧自身轴投影 + 65,536 对大包”；相关 192 项测试通过 | 155 为 4.84 秒，14 为 24.65 秒；Top‑4 90 秒超时，性能目标未达 |

当前没有 Isaac、候选 runner 或硬件进程在运行。静态测试通过只说明代码合同没有在相应
测试中破坏，不能证明真实接触、离桌 40 mm、保持、滑移、桌面碰撞或力矩安全。

## 当前计算瓶颈

H97/H99/H101 的真实 Top‑4 对比显示，单次对比在开始候选碰撞前已有约 56.0 秒公共准备时间：
运行时和对象/手合同构建约 14.4 秒，四个精确接触约 19 秒，聚合碰撞输入约 15 秒，
对象 145,588 面静态表面准备约 6.4 秒。随后 `f1Link3` 成为首个主要热点。

在 H97 中，candidate 14 的这个热点处理 165 个时间区间，窄相检查 8,448,199 对，
最后仍剩 3 对落到相邻 binary64 时间端点，无法继续二分，只能返回 `UNRESOLVED`。
也就是说，程序花费大量时间把绝大多数对证明为分离，但最后的三对仍没有形成正式无碰撞证书。

H101 没有解决该瓶颈：candidate 14 仍检查 8,432,470 对，却由 BVH 叶节点分别触发
21,444 个窄相 packet，平均每包只有 393.23 对，远低于 65,536 上限。三组轴函数、索引收集
和校验因此在叶粒度反复调用，预计算收益被调度开销抵消。下一条已识别假设是先汇总 root
叶节点产生的三角形对，再做真正的大包窄相。H102 实现快照与假设文件随本提交保留，但
用户已在正式结果登记前叫停该轮；没有 H102 result，不能声称其测试或性能目标通过。

需要同时审查两类问题：

- 算法复杂度：怎样在不漏碰撞的前提下，避免“时间二分次数 × 未解决三角形对”重复工作；
- 物理语义：允许 PAD 接触的表面与禁止穿透的封闭外壳怎样形成可证明、不会互相矛盾的合同。

## 建议审查的源码顺序

1. [`continuous_collision.py`](../../src/kcg_connector/kcg_connector/grasp/robust/continuous_collision.py)
   - `_StaticTriangleBVH`
   - `_moving_triangle_triangle_strict_separation_mask`
   - `certify_moving_link_surface_separated_from_static_surface`
2. [`full_hand_collision.py`](../../src/kcg_connector/kcg_connector/grasp/robust/full_hand_collision.py)
   - `TerminalForbiddenSurface`
   - `certify_full_hand_contact_range_policy_closure`
3. [`aggregate_collision_inputs.py`](../../src/kcg_connector/kcg_connector/grasp/robust/aggregate_collision_inputs.py)
4. [`production_candidate_generation.py`](../../src/kcg_connector/kcg_connector/grasp/robust/production_candidate_generation.py)
5. [`multifidelity_candidate_rank.py`](../../src/kcg_connector/kcg_connector/grasp/robust/multifidelity_candidate_rank.py)
   与 [`bounded_topk_exact.py`](../../src/kcg_connector/kcg_connector/grasp/robust/bounded_topk_exact.py)
6. [`test_continuous_collision.py`](../../src/kcg_connector/test/robust_grasp/test_continuous_collision.py)
   与 [`test_full_hand_collision.py`](../../src/kcg_connector/test/robust_grasp/test_full_hand_collision.py)

本次提交还保留了 H93–H101 的小型原始 JSON。请以 JSON 内的状态、原始数值和
`formal_selected_candidate` 等字段为准，不要只从文件名里的 `PASS` 推断结果。

## 希望 ChatGPT Pro 回答的问题

请基于本提交的实际源码回答，优先给出可证伪的判断，不要先写代码：

1. H96 的 dual-surface 语义是否充分：接触 skin 用于证明“这是允许的指垫接触”，
   closed shell 用于证明“其余实体没有穿透”。还缺哪些集合包含、边界一致性或运动连续性证明？
2. 当前 BVH + 区间运动学 + 17 轴 SAT + 时间二分的最坏复杂度在哪里？H101 实测平均
   packet 只有 393.23 对；应如何跨 BVH 叶汇总又不丢失严格 pair accounting？
3. 父子 frontier 应怎样缓存或组织，才能保证每个子区间在细分前完成必要窄相，同时避免
   H99 那种虚假不确定性和重复终点计算？请比较至少两种架构。
4. 对最后 3 个 `ADJACENT_BINARY64_PHASE_ENDPOINTS` 未解决对，应该采用什么严格方法区分
   真接触、允许 PAD 接触和区间包络过松？不能用 epsilon 放宽、删面或忽略。
5. 约 56.0 秒的公共准备能否做成哈希绑定的可复用只读缓存，同时保证对象、手、几何、
   编译后端和配置任一变化都会失效？请指出建议缓存的对象和完整缓存键。
6. 请给出最多 5 个高信息量实验。每个实验写明唯一改变量、预期数据、失败判据，以及它会
   区分哪两种根因。不要给参数网格盲扫。
7. 请列出当前测试没有覆盖、但可能导致“错误认证为无碰撞”的 5 个最高风险点，并指出应加
   到哪个测试文件。

输出请分为：根因排序、正确性审查、性能审查、最多 5 个实验、推荐架构、禁止采用的捷径。

## 不允许通过审查改变的边界

- 不得放宽几何、碰撞、8 N、0.30 N·m、50 µm 或 5 µm 门限；
- 不得删掉真实面、改用圆柱近似、磁吸、隐藏固定或运行后写物体位姿；
- 不得把对象真实位姿、碰撞名称、接触点/法向或 PhysX 真值送入正式在线控制；
- 不得因单测通过、程序退出码 0、离线 `passed=true` 或生成文件而选择正式候选；
- 不得启动 Isaac、训练/RL 或真实硬件；本次请求只授权代码和算法审查。

## 原始证据索引与 SHA-256

这些 JSON 是本次 GitHub 审查特意纳入的小型证据；本地大型 `artifacts/`、轨迹、日志、
USD 和缓存仍不上传。

| 证据 | SHA-256 |
| --- | --- |
| [`H93 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H93_PACKET_STATIC_BVH_TRAVERSAL_RESULT_20260823T091408.446Z.json) | `5fc53ff48d5f5ca562c94b16acc0b71231d705a75a702392c732604b2dfa0ac0` |
| [`H94 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H94_F1LINK3_COMMON_BLOCKER_DIAGNOSTIC_RESULT_20260823T092914.615Z.json) | `79a21eabf79e96a6c21bb27a3fca0ad5c8c84d14ae0bf1122af6410932630e57` |
| [`H95 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H95_SAME_SOURCE_TERMINAL_SURFACE_ROUTE_AUDIT_RESULT_20260823T094429.687Z.json) | `be2f51ac829556026450d8278ad7790eb1f78da427673874635c0f862cc2d4e3` |
| [`H96 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H96_INDEPENDENT_CONTACT_SKIN_CLOSED_SHELL_NARROWPHASE_RESULT_20260823T101704.127Z.json) | `5d04de255d74384f48279139052e2e29859006b6924a0553717bce7a144f2650` |
| [`H97 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H97_PERSISTENT_UNRESOLVED_PAIR_FRONTIER_RESULT_20260823T105139.410Z.json) | `ae30f31b079028183e08cb90d199f36d3c816c2de02e54931facbac8cb15527c` |
| [`H98 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H98_STAGED_ANISOTROPIC_PRECOMPUTED_AXIS_NARROWPHASE_RESULT_20260823T111559.059Z.json) | `fb0951988b56c956bf88f708c34571d68e3fec71670bc6e807d45667f7c9fd42` |
| [`H99 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H99_AMORTIZED_HALVING_NARROWPHASE_DEFER_RESULT_20260823T115024.137Z.json) | `ef4e8836f92adf46979771a474d5fd66f6d947532c9c3e833545905a09a6450f` |
| [`H100 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H100_TERMINAL_FORCE_WORK_EXPLOSION_DIAGNOSTIC_RESULT_20260823T122115.231Z.json) | `82d4f709d36d2060895a2f1c5354bf8413a180b437b585952172d4268f0075fc` |
| [`H101 hypothesis`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H101_EAGER_PRECOMPUTED_AXIS_FAMILY_LARGE_PACKET_HYPOTHESIS_20260823T122530.001Z.json) | `1185a34066092df053fb387245532cc5e9c7c80df4be841ca144d99c54575d58` |
| [`H101 result`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/H101_EAGER_PRECOMPUTED_AXIS_FAMILY_LARGE_PACKET_RESULT_20260823T123547.024Z.json) | `45fd32dd7bd64b0d03f3cfadddf0e5217b0e65307cef085f1f77d1bc0ad21677` |
