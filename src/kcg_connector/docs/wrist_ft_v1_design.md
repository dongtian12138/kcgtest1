# 腕部三个方向力与三个方向力矩：当前边界

## 当前状态

`config/wrist_ft_v1_contract.yaml` 是 `design_only` 合同，`enabled=false`。它说明如何
读取和解释腕部三个方向的力与三个方向的力矩，但不授权 Isaac、真实硬件或在线策略。
本轮清理没有修改该合同、机器人 Xacro、USD、TCP、质量、质心或安全门限。

早期 residual v0/v1、q7 twist 和 SAC 训练设计已退出活动源码。需要审计原设计时读取：

```bash
git show pre-active-route-prune-20260823:src/kcg_connector/docs/wrist_ft_v1_design.md
```

## 仿真测量边界

现有运动链包含零变换固定关节：

```text
iiwa_link_ee --hand2arm--> handbase_link --...--> grasp_tcp
```

`hand2arm` 可作为虚拟腕部测量边界。它没有新增质量、惯量、视觉形状、碰撞形状、
link 或 joint，也不改变 TCP。Isaac 返回顺序为：

```text
[Fx, Fy, Fz, Tx, Ty, Tz]
```

原始坐标系是 joint child frame，也就是 `handbase_link`。项目标定约定为：

```text
wrench_canonical = -wrench_raw
```

从 `handbase_link` 转换到连接器任务坐标系时必须使用完整空间力变换；测量点发生平移
时，力矩需要包含 `r × F`，不能只旋转六个数。该边界测到的是整只手和所抓工件传给
机械臂的净载荷，不是指尖接触点、接触法向或碰撞体真值。

## 补偿链

在线可用量应来自：

```text
原始六维载荷
  - 空手零偏
  - 三指手重力
  - 工件重力
  - 运动惯性
  = 接触相关六维载荷
```

零偏只允许在空手、无接触的自由空间更新。抓取后自由空间可以识别工件载荷，但
`INSERT/ENGAGE/SCREW/HOLD` 等接触阶段禁止 tare，否则会把真实接触力消掉。

## 从七个关节力矩估计末端六维载荷

在“唯一外力作用于已知工具点”且工具 Jacobian 满秩时：

```text
tau_external = transpose(J_tool) * wrench_tool
```

可以用带权、带尺度、带阻尼的最小二乘估计 `wrench_tool`。这不是无条件等价的实体
腕部传感器，至少要同时检查：

- 机器人动力学、重力、摩擦、零偏和温漂补偿；
- Jacobian 的秩与条件数；
- `tau_external - transpose(J) * wrench_hat` 的七维残差；
- 是否存在机械臂连杆、手、工件和环境的多点接触。

多点或未知位置接触时，七个关节力矩不能唯一分解每个接触的六维载荷。当前
`kcg_connector.joint_torque_wrench.estimate_tool_wrench()` 在尺度、阻尼、条件数门限
或残差门限缺失时失败关闭，不返回猜测值。

## 判据边界

高 `Tz` 只能说明绕任务轴的阻力增大，不能单独证明插入或锁紧成功。成功至少还要
结合轴向位移、转角、进给、横向力与弯矩、三路指根载荷、滑移和稳定保持。仿真内部
对象位姿、碰撞名称、接触点和接触法向只能用于运行后独立评价，不能进入正式在线控制。

当前 CARTS 受力实现和正式门限以 `CURRENT_CONTEXT_CN.md` 指向的配置、源码和测试为准；
本文不是动态运行授权。
