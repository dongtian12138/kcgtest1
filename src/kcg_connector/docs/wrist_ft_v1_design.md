# 虚拟腕部六维力/力矩边界与关节力矩估计 v1 设计

## 范围和兼容边界

本阶段不为末端六维 F/T 增加实体模型。现有机器人已经有零变换 fixed joint：

```text
iiwa_link_ee --hand2arm--> handbase_link --...--> grasp_tcp
```

因此直接把 `hand2arm` 定义为理想的 virtual measurement boundary：

- 测量边界的 parent 是 `iiwa_link_ee`，child 和原始输出坐标系都是
  `handbase_link`；
- 零厚度、零质量、零惯量，无 visual、collision、新 link 或新 joint；
- 不改变现有 Xacro、USD、TCP、关节种子、碰撞几何和动力学；
- 当前 Isaac 导入已设置 `merge_fixed_joints=False`，USD 中也保留了
  `PhysicsFixedJoint hand2arm`；
- 当前 `kcg_connector_twist_residual_v0` 的 4 维动作和 24 维观测保持不变。

`config/wrist_ft_v1_contract.yaml` 仅冻结未来 v1 接口，v1 consumer 默认关闭。虚拟
测量边界则随现有资产默认存在，不需要重新导入或覆盖 USD。

## 七个关节力矩能否换算末端六维力

可以，但它是依赖假设和模型的估计，不是无条件等价的腕部六维传感器。

机器人动力学可写成：

```text
tau_measured = M(q) qdd + C(q, qdot) qdot + g(q)
               + tau_friction + tau_bias + tau_external
```

先补偿机器人、手和工件的重力、惯性、科氏/离心、关节摩擦、迟滞、温漂及零偏，
得到七维外部关节力矩 `tau_external`。如果唯一外部作用是已知工具坐标系上的一个
合力/合力矩，则虚功关系为：

```text
tau_external = transpose(J_tool(q)) * wrench_tool
```

iiwa 是 7-DOF，工具几何 Jacobian `J_tool` 为 `6 x 7`。当它的秩为 6 时，理想的
最小二乘解为：

```text
wrench_tool = inverse(J * transpose(J)) * J * tau_external
```

仓库现在已有默认关闭的纯 NumPy 安全核心
`kcg_connector.joint_torque_wrench.estimate_tool_wrench()`。它使用带关节力矩可靠度
权重、任务尺度和阻尼的最小二乘，而不是直接求逆。令
`S=diag(wrench_scales)`、`wrench=S*u`，求解：

```text
argmin_u ||sqrt(W) (transpose(J) S u - tau_external)||^2
         + lambda^2 ||u||^2
```

力的单位是 N、力矩的单位是 Nm，不能直接对混合单位坐标计算一个看似客观的条件数。
因此 `wrench_scales=[Fx,Fy,Fz,Mx,My,Mz]` 必须按任务和标定给出；rank 与 condition
在 `sqrt(W) * transpose(J) * S` 上计算。`wrench_scales`、`lambda`、条件数上限和
七维投影 residual 上限任何一项缺失，当前实现都会返回 `valid=false` 和
`wrench=None`，不会用猜测值启用估计器。

### 7-DOF 冗余和零空间

当 `rank(J)=6` 时，七个传感器并不会让同一个工具 wrench 产生无穷多个答案；在
“单一已知末端接触”假设下，六维 wrench 是唯一可估的。多出的第七维提供一致性
检查：`tau_external` 中落在 `null(J)` 的分量不能由该末端 wrench 解释。

应持续计算：

```text
tau_residual = tau_external - transpose(J) * wrench_hat
```

该 residual 过大可能来自零空间控制力矩、动力学误差、摩擦、线缆力、其他连杆碰撞
或多个接触点，此时不得继续把 `wrench_hat` 当成可信的单一末端载荷。

### 奇异性和可观测性

当 Jacobian 降秩或条件数很大时，某些笛卡尔 wrench 方向不可区分，微小关节力矩
误差会被反解放大。阻尼伪逆只能给出平滑的近似值，不能恢复丢失的信息。因此精密
插入路径必须避开这类构型；越过秩/条件数门槛时应降低估计置信度或停止接触技能，
而不能仅依靠更大的阻尼继续运行。

### 接触位置和分布的限制

如果手、工件和环境只有一个已知工具基准处的合 wrench，以上关系成立。如果手指、
工件、法兰外壳或机械臂连杆同时与环境接触，实际关系是：

```text
tau_external = sum(transpose(J_i) * wrench_i)
```

仅凭七个关节力矩不能唯一恢复每个接触的位置和六维分布，只能在选定工具点上拟合
一个等效 wrench。它也不能判断力究竟作用在连接器哪一侧；这需要视觉、接触模型、
腕部实体传感器或其他局部传感器辅助。

## virtual hand2arm reaction wrench

Isaac 6.0.1 的 `get_measured_joint_forces()` 返回 link incoming joint reaction，排列
为 `[Fx, Fy, Fz, Tx, Ty, Tz]`，坐标系是 joint child frame。查询指定 joint 时，
articulation metadata 的 joint index 需要加 1，因为返回数组第 0 行属于 base link。

本项目因此读取 `hand2arm`，原始 frame 声明为 `handbase_link`。这是整个手以及所抓
工件传给机械臂的净反力，不是指尖接触真值。Isaac 原生符号保留在 boundary/raw
层；六方向物理单位载荷测试已经确认无轴交换，并且 canonical “环境作用于工具”
满足：

```text
wrench_canonical = -wrench_raw
```

随后再把 canonical wrench 完整变换到连接器任务系，并把旋紧阻力定义为正。

当前资产的短 smoke 已通过：metadata joint index 为 7，实际 reaction row 为 8，
输出 shape 为 `(1, 6)`。从零重力切到 `-9.81 m/s^2` 后，原始 wrench 的变化约为：

```text
[0.00191, -0.00000, 20.52195 N, 0.11246, 0.17904, -0.00001 Nm]
```

这证明该边界包含三指手下游的重力载荷，也说明策略使用前必须去重力。TCP 偏移仍为
`[0, 0, 0.4] m`，变化约 `1.1e-16 m`；机器人目录 9 个 USD 文件的组合 SHA256 在
运行前后均为 `65f14f08d7864b041b145e8582891043f171a7dde52656fe715ecd6451d39f86`。

六轴标定门在 `handbase_link` 局部原点分别施加 `+/-4 N`、`+/-0.4 Nm`，并增加
50% 量程。实测 raw→canonical 是 `-I6`；绝对增益范围
`0.9937990057..1.0000000001`，最大同类轴串扰 `0.00018902946`，最大奇对称误差
`0.00020546798`，最大半/全量程线性误差 `0.00017604281`。异类响应分别为最大
`0.00132971717 Nm/N`（力输入到力矩输出）和 `0.00883211736 N/Nm`（力矩输入到力
输出）；它们保留单位，不能冒充无量纲串扰比。`handarm.usda` SHA256 前后均为
`031f8241c9dd1e2af96d7b1dde7d2adda7744891832a05e2580fd3398da4216b`，TCP 变化
`1.65e-24 m`，关节创建/disjoint/teardown articulation assignment 警告均为 0。

复现命令：

```bash
cd ~/WorkPlace/kcgtest1
src/kcg_connector/isaac/run_isaac_python.sh \
  src/kcg_connector/isaac/virtual_wrist_ft_smoke.py
```

输出建议为：

- `/wrist_ft/boundary`：`handbase_link` 中的边界 reaction wrench；
- `/wrist_ft/compensated`：去偏置、重力、载荷和运动惯性后的 wrench；
- `/wrist_ft/task_wrench`：完整空间力变换到 `connector_task_frame` 后的 wrench。

从边界原点移动到啮合基准点时必须包含力臂项 `r x F`，不能只旋转六个数。

## 重力、惯性和摩擦补偿

虚拟边界的 reaction 本来就包含边界以下所有手指连杆及工件的重力和惯性。策略和
装配判据应使用：

```text
contact_wrench = boundary_wrench
                 - electronic_or_simulation_bias
                 - distal_hand_gravity
                 - grasped_payload_gravity
                 - dynamic_inertia
```

使用每个手指当前关节角、已辨识的各连杆质量/质心/惯量以及视觉估计的
`T_hand_connector` 计算载荷。电子零偏只允许在空手、无接触的
`HOME_FREE_SPACE_EMPTY_HAND` 更新；`POST_GRASP_FREE_SPACE` 只能采集/核验工件载荷，
不能把单姿态读数直接清零。`INSERT/ENGAGE/SCREW/HOLD` 中严禁 tare。

通过七个关节传感器估计 wrench 时，补偿范围更大：不仅是手和工件，还必须准确
补偿整条机械臂的 `M/C/g`、关节摩擦和传感器误差。低速准静态旋拧会减小惯性误差，
但不会消除重力、摩擦、温漂和 Jacobian 条件数造成的误差。

## 仿真与实机融合建议

建议同时保留三类互补信号：

1. Isaac 中的 `hand2arm` virtual reaction wrench；若实机以后有物理腕部 F/T，则
   映射为同一 canonical boundary wrench。
2. 七个 iiwa 关节外力矩经过动力学补偿和 Jacobian 反解得到的独立 wrench estimate。
3. `f1j2/f2j1/f3j2` 三路指根力矩，继续表征夹持分布、偏载和滑移趋势。

仿真可把 `hand2arm` reaction 作为主要传感器通道，并给它注入偏置、噪声、延迟、
量化、串扰和饱和；同时独立计算关节力矩估计，与 reaction 做差来训练健康度和置信度
逻辑。实机若有腕部实体 F/T，则以它为主要接触信号，关节估计做交叉校验；若没有，
可以退化到关节估计，但必须使用更保守阈值，并把精度等级明确标为较低。

典型融合解释：

- 腕部/边界 `Tz` 高、关节估计一致、深度不足且三路指根稳定：更像乱扣或卡滞；
- `Tz` 高且某路指根力矩突降、视觉看到手到工件相对转动：更像手内滑移；
- `Fx/Fy` 或 `Tx/Ty` 高而三路指根稳定：更像插入轴偏心或倾斜；
- 关节估计很高但 `hand2arm`/物理腕部 F/T 不高：检查机械臂其他连杆碰撞、零空间
  力矩或动力学补偿误差；
- 两个六维来源一致、转角/轴向进给正确、三路指根稳定并完成保持，才构成成功证据。

高 `Tz` 本身永远不能宣告成功。

## residual v1 契约

不要修改 `residual_rl.py` 中 v0 的常量、顺序或编码器。未来新增独立模块，保持原
4 维动作不变，在原 24 维观测末尾追加：

```text
wrist_force_x, wrist_force_y, wrist_force_z,
wrist_torque_x, wrist_torque_y, wrist_torque_z
```

因此 `kcg_connector_twist_residual_wrist_ft_v1` 是 4D action / 30D observation。
策略只接收补偿后、变换到连接器任务系、按型号尺度归一化并裁剪的六维 wrench。
原始值、Jacobian 条件数、rank、投影 residual、时间戳和饱和状态由独立安全监控以
更高频率处理，不允许 policy 绕过。

## 实施文件和测试门

当前只增加独立 smoke、标定分析、安全估计核心、设计配置和纯测试，不改
`hand.xacro`、USD 或 v0/backend，也没有把六维力接入 v0 policy。
以后按以下门控实施：

1. `virtual_wrist_ft_smoke.py` 验证现有 USD 的 `hand2arm` metadata、`index + 1`、
   六维 shape/finite、child frame、重力响应、TCP 偏移和资产 hash 不变。
2. 六方向单位载荷分别验证 `+/-Fx/Fy/Fz/Tx/Ty/Tz` 的实际符号、尺度和串扰。
3. 多姿态空手与已知工件载荷验证重力/惯性补偿；接触阶段 tare 必须失败关闭。
4. 关节估计器在每步检查 rank、条件数和 `tau_residual`；NaN、陈旧、条件恶化或
   residual 超限进入安全停止/降级，不能补零后继续。
5. 以 `hand2arm` reaction 为 ground truth 标定仿真关节估计；实机最好临时使用经
   标定的参考 F/T 或已知砝码/力矩工装校准六个方向。
6. 重新运行抓取、q7 twist、两个反事实、3x120 度、reset、动作因果、zero residual
   和 SAC smoke，证明 v0 行为和资产 hash 未变化。
7. 判据反事实：高 `Tz` 但无转角/进给不得成功；提前高 `Tz` 加进给停滞应判卡滞；
   横向力/弯矩超限应停止；只有进度、终拧区间、三路指根稳定和保持同时满足才能
   成功。

## 主要风险

- 把未补偿的 joint torque 直接乘 Jacobian 伪逆，会把机器人自身动力学误认为接触；
- 接近奇异位形时反解噪声会急剧放大，阻尼只能缓和数值，不能恢复不可观方向；
- 多点或未知位置接触不能从七维 torque 唯一分解；
- 零空间控制、摩擦、迟滞、线缆力、温漂和时序错位会进入估计 residual；
- 仿真 virtual wrench 过于干净会造成 sim-to-real 过拟合，必须随机化测量误差；
- 当前 synthetic rack/thread 仍不能验证真实乱扣、针脚损伤或真实终拧规格。

## 参考

- KUKA iiwa 的七关节力矩传感器与 Jacobian wrench 估计实验：
  <https://doi.org/10.3389/frobt.2022.892916>
- 关节传感器在特定构型中的任务空间可观测性限制：
  <https://arxiv.org/abs/2206.10798>
