# 001 响应：快照与多视角影子估计方案

## 结论

第一版只做 shadow，不产生任何插入命令、C2 选择或授权。在现有 `d38999_tabletop_pick_smoke.py` 的正式抓取 `unsupported_final_hold` 成功终点之后，插入一个 opt-in 的 `POST_GRASP_SHADOW` 阶段：先保存双文件快照，冻结现有手/臂命令再稳定 0.5 s，然后使用 2--3 个真实腕相机视角，联合估计 12 维状态 `x = [T_hand_plug(6), T_receptacle_plug(6)]`，对 C2 两个 yaw 分支分别求解、分别输出，不平均。`T_hand_plug` 由真实腕相机中 Plug 端点的 RGB/深度/法向/边缘 + 腕相机外参 + 手部 FK 约束；`T_receptacle_plug` 由同一批图像中的 Receptacle 端点观测 + 同一 `T_hand_plug` + 相机外参/FK 约束。C2 的 Rz 若不可观，输出 `observable_dofs=5` 与 `C2_UNRESOLVED`。协方差只使用白化后的、无 LM damping 的 `J^T J` 在可观子空间上的逆，并由离线的 bootstrap coverage 校准，不允许用在线残差任意缩放 Hessian。阶段 0 冒烟仍只运行现有逐指柔顺抓取，不改抓取、不跑 Isaac 之外的任何授权路径。

## 主要方案

### 1. 快照 schema（双文件，类型/文件级真值隔离）

在正式抓取 PASS 已按现有硬门计算完成后、进程退出前，调用真值授权的 `postgrasp_snapshot_truth` 写入两个独立文件：

`<output-dir>/snapshot/control_and_obs.json`（estimator/controller 可读）：
- `schema_version: kcg_d38999_postgrasp_snapshot_v1`
- `snapshot_id`、`episode`、`seed`、`global_step`、`phase = postgrasp_shadow_hold_ready`、`timestamp_utc`
- `source_hashes`：pick/tabletop/physical-grasp/runner/runtime/registration 的 SHA-256
- `control_frozen`：`arm_q_target_rad`、`hand_q_target_rad`、`kps`、`kds`、`physical_grasp_method`；这些都是现有 controller 输出，不是对象真值
- `robot_observed`：`arm_q_actual_rad`、`hand_q_actual_rad`、`joint_qd_rad_s`、`tcp_pose_from_actual_arm_fk`、`handbase_pose_from_robot_link_state`
- `wrist_ft`：`payload_reference`、`last_canonical`、`empty_baseline`；wrist-FT 只作为负载一致性诊断，不进入位姿残差
- `observation_manifest`：指向 `obs/view_*/` 的 rgb/depth/semantic/fk/camera_contract 清单及 manifest SHA-256
- `restore_policy`：`settle_steps=120`、`physics_rate_hz=240`（0.5 s）、`freeze_commands=true`、`capture_allowed_after_settle=true`

`<output-dir>/snapshot/truth_restore.posthoc.json`（只允许真值授权模块读写）：
- `role: truth_restore`、`scope: snapshot_restore_and_posthoc_evaluation_only`
- `object_dynamic_truth`：`plug_body` 与 `coupling_nut` 的 `position_m`、`orientation_wxyz`、`linear_velocity_m_s`、`angular_velocity_rad_s`；`fixed_receptacle` 的 `position_m/orientation_wxyz`（漂移审计）
- `robot_dynamic_truth`：仅用于 restore 的关节 `q/qd` 冗余副本
- `posthoc_truth`：`T_hand_plug_actual`、`T_receptacle_plug_actual`、`body_nut_separation_change_m` 等，只由 posthoc 评价器读取

文件/API 防火墙：
- 只有 `postgrasp_snapshot_truth.py` 能 import/调用对象 prim 的 `get_world_pose/get_linear_velocity/get_angular_velocity` 和 restore 用 setter；该模块的返回值在 online 路径中直接落到 `truth_restore.posthoc.json`，不进入内存中的 estimator 输入对象。
- `postgrasp_shadow_estimator.py` 不 import `postgrasp_snapshot_truth.py`，不接受对象 prim、`contact_report` 或任何 `truth_restore` 字段；其唯一输入是 `control_and_obs.json` + `obs/view_*/`。
- 观测加载器 `PostgraspObservationArchive.load()` 逐字段校验 `role/scope`，遇到 `truth_restore` 或未知字段抛 `TruthFirewallViolation`；即使把 truth 文件路径传给它也会因 `role != observation` 拒绝。
- 可测试防火墙：AST 测试断言 estimator 源码不含 `body.get_world_pose`、`nut.get_*`、`fixed_prim`、`contact_report`、`set_world_pose`；CI 注入伪造 `T_hand_plug_actual` 字段到观测 JSON，必须抛错；调用图测试证明 estimator 不 import 真值模块。

### 2. 恢复后的 0.5 s 稳定与一致性检查

Restore 仅用于 posthoc/replay，绝不在正式 episode 中对对象 pose teleport。顺序固定：
1. 在暂停的 World 中由真值模块恢复 `plug_body/coupling_nut` 的位姿与速度、机器人 `q/qd`；contact 不直接恢复，作为瞬态在后续物理步中重新求解。
2. 用快照中的冻结 `arm_q_target/hand_q_target/kps/kds` 恢复控制器命令。
3. 恢复 play 后先运行 `120` 个物理步（240 Hz 即 0.5 s），期间禁止任何相机 render/estimator 调用；记录 wrist-FT 与关节速度。
4. 之后再用 60 步（0.25 s）做一致性检查：尾窗关节速度低于现有 `maximum_final_observable_joint_speed_rad_s`；wrist-FT 与 `payload_reference` 差值的 force ≤ 1.0 N、moment ≤ 0.05 N·m（仅诊断，不是放宽 8 N / 0.30 N·m 硬门）。
5. Posthoc 真值检查：settle 后 `T_hand_plug` 与快照真值的平移差 ≤ 1.0 mm、旋转差 ≤ 2.0°；`T_receptacle_plug` 平移差 ≤ 1.0 mm、非 yaw 旋转差 ≤ 2.0°，yaw 按分支分别记录。任一失败输出 `SNAPSHOT_RESTORE_SETTLE_REJECTED`，该 replay 不产生测量结果。

### 3. Online / posthoc 数据流

Online（同一 Isaac World，不 reset）：
1. 现有逐指柔顺抓取完整跑完，`formal_lift_terminal`、`passed` 按现状计算。
2. `POST_GRASP_SHADOW` opt-in 块执行快照写入；不改任何现有门、不改变 `passed`。
3. 保持快照冻结命令 0.5 s；采样 wrist-FT/关节速度。
4. 视角规划器给出最多 3 个候选腕相机 TCP 位姿（见下）；每个视角在暂停 timeline 下用现有 `capture_d38999_rgbd_runtime.py` 捕获 RGB/深度/语义。
5. `shadow_capture` 包装器从 runtime 返回值中只抽取 `rgb/depth/semantic/id_to_labels/camera 内参/外参/FK/wrist_ft`，丢弃 `registered_truth_xy_m`、`xy_error_m`、`camera_projection.world_points_m` 等真值派生字段；runtime 的 `.passed` 也**不**作为正式质量门，正式门由 mask/深度/FK 重算。
6. estimator 对两个 C2 分支分别求解并写 `shadow_result.json`；`shadow_authorized=false`、`control_authorized=false` 恒成立，不生成任何运动命令。
7. 进程正常结束，不释放、不插入。

Posthoc（CPU 或独立 Isaac replay）：
- 用 `control_and_obs.json` 重放 estimator，结果必须与 online 结果逐 bit 可比（相同随机种子/相同采样顺序）或差异小于 `1e-12`。
- 用 `truth_restore.posthoc.json` 计算误差与覆盖率；该文件不进入 estimator。
- 需要物理一致性时运行独立 restore replay：恢复快照 → 0.5 s settle → 上述负载/相对位姿检查 → 才允许重捕获。

### 4. 2--3 视角选择评分（真实腕相机通过，旧合成只作先验）

候选来源：
- 复用 `d38999_wrist_camera_candidate_search.py` 生成的固定腕相机安装 `mount_eye/target` 与姿态族**只作为先验/种子**，不得作为真实通过证据。
- 在真实 post-grasp 实际 FK 位姿 `H0` 上，用 `solve_fixed_q7_tcp_pose`/现有 IK 生成 TCP 候选：`V0=H0`，`V1/V2/V3` 为 handbase/assembly-TCP 坐标系内 ≤ 40 mm 平移、≤ 12° 旋转的小偏移；所有偏移只由静态 camera-mount 与任务安装关系生成，不读对象真值。
- 候选硬门（全部满足才可捕获）：
  - IK 有限、关节不越限、最大关节增量 ≤ 0.05 rad；
  - 保守几何 clearance ≥ 10 mm；不使用 PhysX contact report/collider identity；
  - 光轴与插装轴夹角 `[25°, 70°]`，任意两视角光轴夹角 ≥ 20°；
  - 现有 wrist-FT/torque 门全程启用，任何触发即中止并返回 `H0`。
- 捕获后真实质量分（每个视角）：
  - `q_pix = min(1, min(endpoint_pixels)/150)`
  - `q_diam = min(1, min(endpoint_diameter_px)/120)`
  - `q_vis = 1.0` 当且仅当 Plug/Receptacle 两个真实语义 mask 同帧可见
  - `q_depth = min(1, depth_valid_fraction/0.10)`
  - `q_occ = max(0, predicted_occluder_fraction - 0.35)`
  - `q_coax = max(0, (25° - coaxial_offset_angle)/25°)`
  - `S_i = 2.0 q_pix + 2.0 q_diam + 1.5 q_vis + 1.0 q_depth - 2.0 q_occ - 0.5 q_coax`
- 视角集合分：`S(V) = min_i S_i + 0.8·I[|V|=2] + 1.2·I[|V|=3] - 1.0·max(0, log10(cond_obs_5d/1e6))`；其中 `cond_obs_5d` 是由捕获后真实 mask + CAD 重投影近似 Jacobian 计算的数据条件数，不是旧 `render_points` 合成图。
- 贪心选择：先捕获质量最高的 2 个视角；若硬门或 `cond_obs_5d > 1e6` 不满足，再捕获第 3 个。3 个仍不满足 → `VIEW_SET_REJECTED`。
- 旧 `d38999_wrist_multiview_evaluation.py` 只能产生 CPU 合成 accuracy 证据，永远不写入 `WRIST_CAMERA_REAL_PASS`；真实通过必须来自上述同一 World、真实 render product 的捕获。

### 5. 联合优化状态与残差

每 C2 分支 `b ∈ {0, π}` 独立求解 12 维状态：
`x_b = [t_hp_x,t_hp_y,t_hp_z,r_hp_x,r_hp_y,r_hp_z, t_rp_x,t_rp_y,t_rp_z,r_rp_x,r_rp_y,r_rp_z]`
- 前 6 维为 `T_hand_plug`（Plug 坐标系在手/`handbase_link` 中的位姿，RPY 顺序 xyz，沿用现有 `pose_matrix/matrix_pose`）。
- 后 6 维为 `T_receptacle_plug`（Plug 坐标系在 Receptacle 坐标系中的位姿）；C2 分支只指后者的 Rz，两分支不共享、不平均。
- 相机外参不优化；每个视角位姿由实际关节 FK 给定，不做 per-view 位姿估计。

真实观测约束（每视角 i）：
`T_WH_i = FK(actual_arm_q_i)`，`T_HC` 为标定腕相机外参，`T_WC_i = T_WH_i T_HC`；
- Plug 模型点：`p_cam = π( inv(T_WC_i) · T_WH_i · T_HP(x) · p_plug )`
- Receptacle 模型点：`p_cam = π( inv(T_WC_i) · T_WH_i · T_HP(x) · inv(T_RP_b) · p_receptacle )`
- 因此：
  - `T_hand_plug` 由 Plug 端点的 RGB 边缘/轮廓、真实深度尺度、深度法向、多视角视差约束；
  - `T_receptacle_plug` 由 Receptacle 端点的同批 RGB-D 约束 + 共享 `T_hand_plug` 约束；
  - 若某视角只有一个端点可见，该视角不能同时解耦 `T_HP` 与 `T_RP`，必须拒识该视角；两个 CAD 模型本身提供公制尺度，深度提供绝对尺度。
- wrist-FT 不进入位姿残差，只做抓持/恢复一致性门，避免把力信号伪装成几何约束。

残差组（复用 `d38999_cad_registration.py` 与 `d38999_inhand_multiview.py` 的现有核，不新造注册系统）：
`r = {plug_edge, plug_silhouette_DT, plug_depth, plug_normal, plug_occlusion, rec_edge, rec_silhouette_DT, rec_depth, rec_normal, rec_occlusion, fk_prior_hp, fk_prior_rp}`
- 真实语义只提供 Plug/Receptacle 根级 endpoint mask；`RECEPTACLE_MATING/SHELL_FLANGE` 等理想特征标签只允许出现在离线观测性诊断，不得作为真实通过输入。为此给现有核增加一个按 endpoint 角色传入 mask 的窄接口，不新建第三套注册器。
- 在线路径不得调用 `d38999_cad_registration.register_relative_pose_multiview(..., receptacle_pose_world, ...)`：该入口需要 Receptacle 世界位姿真值；在线只复用它的残差核与参数化方式，以及 `d38999_inhand_multiview` 的相机组合/联合单参考位姿思路。
- 边缘：真实 RGB 的 Canny 边 + mask 膨胀窗，CAD 边缘样本采样距离变换，尺度沿用 `1.25 px`。
- 轮廓：mask distance transform 在 CAD 采样点上的值，尺度 `1.5 px`。
- 深度：可见 CAD 点的预测深度与真实深度差，尺度 `0.75 mm`，z-buffer 可见性 `d_pred ≤ d_obs + 1.5 mm`。
- 法向：真实深度图局部法向与 CAD 法向点积，尺度 `0.08`；法向不可靠处自动零权，不硬凑。
- 遮挡：预测点在真实深度后方时给 0--4 的截断遮挡代价，避免把手指/nut 遮挡拉进估计。
- 先验：`fk_prior_hp` 用已验证的 30/30 post-grasp 分布均值与协方差（有限、明确标记 `population_prior`），`fk_prior_rp` 用静态任务几何先验；都不是当前 episode 对象真值，权重沿用现有 `0.12` 级别。
- 优化：`scipy.least_squares(..., method="trf", loss="linear")`；初值来自上述先验；参数坐标归一化为 `[1 mm, 1 mm, 1 mm, 0.05 rad, ...]`，沿用现有 `parameter_scale`。

### 6. C2 输出

`shadow_result.json` 固定输出两个分支：
```
c2: {
  retained_hypotheses: 2,
  averaged: false,
  selected_for_control: null,
  hypotheses: [
    {id: "YAW_0", yaw_rad: 0.0, T_hand_plug: [...], T_receptacle_plug: [...], cost: ...},
    {id: "YAW_PI", yaw_rad: pi, T_hand_plug: [...], T_receptacle_plug: [...], cost: ...}
  ],
  resolution: "C2_UNRESOLVED" | "C2_RESOLVED_BY_OBSERVATION",
  observable_dofs: 5 | 6,
  conditional_covariance_xyz_rx_ry_5x5_per_branch: [...]
}
```
- 任何情况下都不得输出两分支的算术平均作为 C2。
- `observable_dofs` 专指该 C2 分支 `T_receptacle_plug` 的可观自由度；`T_hand_plug` 的可观秩在 `hand_plug_observable_dofs` 中单独输出。
- 若数据 Hessian 中相对 Rz 的归一化特征值 `λ_rz / λ_max ≤ 1e-8`，或两个分支的非 yaw 结果横向差 > 0.50 mm 或轴差 > 1.0°，输出 `observable_dofs=5` 和 `C2_UNRESOLVED`。
- `T_hand_plug` 在两分支中共享 Plug 残差，因此应一致；若分支间 `T_hand_plug` 平移差 > 0.2 mm 或旋转差 > 0.2°，说明联合问题病态，输出 `JOINT_BRANCH_INCONSISTENT`，不得选择较小 cost 分支掩盖问题。
- 即使 Rz 可区分，第一版 shadow 仍不选分支、不授权控制；`C2_RESOLVED_BY_OBSERVATION` 只是观测结论，不是运动许可。

### 7. 协方差：白化、无 LM Hessian、可观子空间、bootstrap coverage

按以下顺序计算，禁止在最终 Hessian 上任意乘系数或加 `np.eye` damping：
1. 第一次 `loss="linear"` 求解后，按残差组计算稳健尺度 `σ_g = 1.4826·MAD(r_g)`，下限保护为初始组尺度的 `0.1`。
2. 用 `W_g = 1/σ_g²` 白化各组残差，重新 `loss="linear"` 求解；迭代至 `max|Δlog σ_g| < 1e-3`，最多 3 轮。
3. 最终轮取 `J = result.jac`（白化残差 Jacobian），`H = J^T J`；不采用 `least_squares` 内部 LM/TR 修正矩阵，不添加任何对角 damping，不用 Huber/soft_l1 的 robust loss 生成授权协方差。
4. 白化健全性：`χ²/DOF = ||r_w||²/(n - rank)` 必须在 `[0.25, 4.0]`，否则协方差状态为 `UNVALIDATED`；这只能触发拒识，不能反向缩放 H。
5. 可观子空间：对 H 做特征分解，`λ_cut = 1e-9·λ_max`；仅保留 `λ > λ_cut` 的特征方向，`C_obs = V_obs diag(1/λ_obs) V_obs^T`。总体输出 `joint_observable_dofs = rank(V_obs)`；C2 的 `observable_dofs` 取 `T_receptacle_plug` 后 6 维块在该可观子空间中的秩（相对 Rz 不可观时 = 5）。零/近零方向不赋伪有限方差；输出 `null_directions` 名称。
6. `T_receptacle_plug` 的 5x5 条件协方差取可观子空间逆后、后 6 维中 `x,y,z,rx,ry` 的边际块；`T_hand_plug` 同理给其可观维协方差。C2 Rz 不可观时该 5x5 必须存在。
7. Bootstrap coverage 校准：离线渲染/噪声基准 N ≥ 400 episodes；在可观子空间计算 Mahalanobis 误差与 90% 置信椭球覆盖率，并做逐 DOF 95% 检查。经验覆盖率目标 `[0.85, 0.98]`。若不在区间，用 split-half（各 200）搜索单个乘法因子 `γ ∈ [0.25, 4.0]`，只在另一半 held-out 上重新达到 `[0.85, 0.98]` 才认证 `coverage_calibrated=true` 并输出 `covariance_calibrated = γ·C_obs`。否则 `coverage_calibrated=false`，协方差仅作诊断。
8. `γ` 只来自离线 held-out bootstrap，版本化记录 `coverage_calibration_version`、`empirical_coverage`、`heldout_coverage`；在线残差永远不能用于缩放。

### 8. 拒识门

优先级从高到低，任一触发即 `shadow_status=REJECTED`；第一版所有结果都不得带 `control_authorized=true`：
1. 快照/观测不完整、SHA-256 不匹配、时间戳不单调、非有限数组、shape 不一致 → `SNAPSHOT_OR_ARCHIVE_INVALID`
2. 少于 2 个通过硬门的视角、任一端点不可见、mask 中心进入 16 px 边缘禁区、端点像素 < 100 或直径 < 80 px、depth 有效率 < 0.05 → `VIEW_SET_REJECTED`
3. 真实数据条件数 `cond_obs_5d > 1e6` 或近似 Jacobian 秩亏 → `OBSERVABILITY_REJECTED`
4. 优化不收敛、越界终止、残差非有限、白化健全性失败 → `OPTIMIZATION_REJECTED`
5. 分支间 `T_hand_plug` 不一致、非 yaw C2 不一致、Rz 不可观 → `C2_UNRESOLVED`；`observable_dofs=5`
6. `coverage_calibrated=false` → 协方差不能用于任何授权；shadow 报告仍可输出，标记 `COVARIANCE_NOT_AUTHORIZED`
7. 2 mm / 6 deg 只作为视场与初始化包络记录；`+X 0.55 mm` 只作为既有 Bcapture 证据记录；两者不得写成授权区，不得触发 `control_authorized`。

## 备选方案

- 备选 A：先 Plug-only 估计 `T_hand_plug`，再固定它分别估计两个 `T_receptacle_plug` 分支。调试更简单，但会隐藏 Plug/Receptacle 残差对 `T_HP` 的耦合，不满足“联合优化”要求；不选。
- 备选 B：3 视角默认全开。信息更足，但增加腕力矩/耗时与遮挡风险；选为条件触发——2 视角先过，只有质量/条件数不过才开第 3 视角。
- 备选 C：bundle adjustment 同时优化相机外参/每视角位姿。现有两条注册系统都冻结外参，且要求明确禁止 camera extrinsic 优化；不选。
- 备选 D：整体 USD session-layer 快照恢复。最保真但文件大、真值泄漏面大、与“对象动态真值”的最小 schema 目标冲突；不选，保留 minimal state + 0.5 s settle。
- 备选 E：用旧 `d38999_wrist_multiview_evaluation.py` 的合成 render 当作腕相机通过。明确不选，只作 CPU 回归。

## 失败点

- D38999 圆 mating rim/隐藏 key 导致相对 Rz 真不可观：预期常态是 `observable_dofs=5` + `C2_UNRESOLVED`，不得强造唯一 yaw。
- 手/指/coupling nut 遮挡 Plug 或 Receptacle 边缘，单视角只剩一个 endpoint：联合状态不可解耦，必须拒识。
- 腕相机离 mating 区近、两视角 TCP 偏移只有 12--40 mm，视差/条件数可能不足；`cond_obs_5d` 硬门会拒绝而非放宽。
- 真实语义只有根级 endpoint mask，没有理想 mating/feature 标签；若只用根 mask + CAD 采样仍系统性偏置，合成离线测试可能好、真实测试差，需要真实 mask 诊断优先。
- 快照 restore 后接触瞬态不同，settle 后的 `T_hand_plug` 超过 1.0 mm / 2.0° 一致性线：该 replay 被拒，不能把不一致数据送去覆盖校准。
- 深度法向在斜视/遮挡边界不可靠，normal 残差会被迫降权，信息下降。
- 离线 bootstrap 基准与真实照明/手指遮挡分布不一致，coverage 校准失败或过拟合；只能 `COVARIANCE_NOT_AUTHORIZED`。
- 现有 `capture_d38999_rgbd_runtime` 的 `.passed` 包含真值 XY 误差；若误把它作为正式质量门，会破坏真值防火墙。必须由 `shadow_capture` 过滤。
- 视角运动虽小仍可能触碰桌面/夹具；保守几何门和 wrist-FT 硬门缺一不可，不能用 contact report 选视角。

## 真值 / 坐标 / 安全风险

- 坐标：世界 W；`handbase_link` 为 `T_hand_plug` 的手坐标；Receptacle/Plug mating CAD 原点在 z=0 配对面上；RPY 为 xyz 顺序；`T_A_B` 表示 B 在 A 中的位姿（与现有 `posthoc_t_hand_plug_actual` 一致）。
- 真值允许范围：只允许真值授权模块（`postgrasp_snapshot_truth.py` 与独立 posthoc 评价器）读写 `truth_restore.posthoc.json` 用于 restore 与 posthoc 评价；estimator/controller/view-planner/C2 分支/拒识门/授权均不得读取。
- 安全：本阶段 shadow 不产生任何 TCP/关节命令，不插入、不释放；视角运动若实现也必须沿用现有 wrist-FT 8 N / 0.30 N·m 与指根 torque 硬门，速度/位移设上限，任何触发即回 `H0`。快照 restore 只在独立 replay 中允许对象 pose/velocity 写入，正式 episode 中继续禁止。
- 不修改抓取控制器、硬门、30/30 基线；`2 mm / 6 deg` 与 `+X 0.55 mm` 的既有结论不被本方案扩大解释。

## 预计修改文件

- 新：`src/kcg_connector/kcg_connector/postgrasp_snapshot_truth.py`（唯一真值授权捕获/恢复）
- 新：`src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py`（纯 CPU 联合估计、C2、协方差、拒识）
- 新：`src/kcg_connector/kcg_connector/postgrasp_shadow_view_planner.py`（视角评分，不含注册求解）
- 新：`src/kcg_connector/config/d38999_postgrasp_shadow_v1.yaml`（视角/协方差/一致性阈值契约）
- 改：`src/kcg_connector/kcg_connector/d38999_inhand_multiview.py`（增加 12 维 `register_postgrasp_hand_plug_and_receptacle_plug_multiview`，复用现有残差核）
- 改：`src/kcg_connector/kcg_connector/d38999_cad_registration.py`（只增加白化协方差/可观子空间工具，不建第三套注册）
- 改：`src/kcg_connector/kcg_connector/isaac_d38999_rgbd_runtime.py`（最小扩展：可选返回/落盘 `semantic.npy` 与原始数组；默认调用行为不变）
- 改：`src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py`（仅在 `passed` 计算后加入 opt-in `--postgrasp-shadow` 块；现有抓取/硬门零改动）
- 新测试：`test_postgrasp_snapshot_firewall.py`、`test_postgrasp_shadow_estimator.py`、`test_postgrasp_covariance_coverage.py`

## CPU / 离线验收

不启动 Isaac 的最小验收：
1. 快照防火墙：truth/observation 双文件 role 校验；truth 路径喂给 observation loader 必须抛 `TruthFirewallViolation`；缺失 SHA、非有限数组、错误 quaternion 范数必须失败。
2. 联合估计单元：用 `proxy_cad_points/render_points` 生成已知真值扰动 ±2 mm/±3° 内的 2/3 视角合成观测（只作离线基准），跑两个 C2 分支；断言两个分支都返回、`averaged=false`、残差 Jacobian 有限、`T_hand_plug` 分支差 ≤ 0.2 mm/0.2°。
3. Rz 不可观用例：只保留圆 mating rim/圆柱且不注入可区分 Rz 的特征，断言 `observable_dofs=5`、`C2_UNRESOLVED`、5x5 条件协方差 PSD；加入可区分 Rz 的角特征后允许 6 DOF。
4. 白化/无 damping：静态测试断言授权协方差路径含 `loss="linear"`、不含 Huber、不在 H 上加 `np.eye`；合成已知组噪声下，白化各组 RMSE 须回到 `[0.5, 2.0]`。
5. Bootstrap coverage：N ≥ 400 渲染 episodes，90% 椭球经验覆盖须在 `[0.85, 0.98]`；若用 `γ∈[0.25,4.0]` 校准，必须在 held-out 半集复现，否则 `coverage_calibrated=false`。
6. 视角评分：同一合成基准上，同轴/单 endpoint/低像素/高条件数候选必须被硬门拒绝；得分排序不得读 `truth_restore`。
7. 恢复 settle 逻辑：用录制的状态数组做 CPU mock，0.5 s 前任何 capture 调用必须抛 `SETTLE_NOT_COMPLETE`；负载或相对位姿超线输出 `SNAPSHOT_RESTORE_SETTLE_REJECTED`。
8. 旧合成评估回归：`d38999_wrist_multiview_evaluation.py` 可继续运行，但报告字段必须保持 `ideal_part_labels_offline_diagnostic=true`，且不得被 shadow 门读取为 `WRIST_CAMERA_REAL_PASS`。

## 需要 Codex 运行的唯一阶段 0 抓取冒烟命令

```bash
PYTHONPATH=src/kcg_connector src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/d38999_tabletop_physical_grasp_v1.py \
  --physical-grasp-method sequential-compliant \
  --formal-lift-mode staged \
  --seed 0 \
  --output-dir artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1/phase0_codex_smoke/seed000
```

通过标准：进程 exit code 0；stdout 出现 `ISAAC D38999 TABLETOP PHYSICAL GRASP V1 PASSED`；`nominal_physics_report.json` 中 `passed=true`、`formal_lift_stages` 长度为 3、`formal_acceptance.sensor_lift_gate=true`、`control_reads_object_truth=false`、`control_reads_contact_report=false`。本命令不改抓取、不启用 shadow、不插入。
