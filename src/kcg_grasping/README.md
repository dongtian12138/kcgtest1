# KCG 圆柱体抓取任务

`kcg_grasping` 是 KUKA iiwa 14 + KCG 空间三指手的第一个可重复任务：

> 抓住圆柱体，垂直抬起，并稳定保持。

它只负责任务世界、复位、观测、成功判定和脚本化基线，不复制 MoveIt、控制器或
机器人描述。后续批量实验可以复用这里的接口。

## 一键诊断

默认是自然演示模式：红色圆柱从 Gazebo 启动开始就在黑色台座上，机械臂从零位先
运动到圆柱上方安全位，再下降进入张开的三指预抓位。整个接近过程以及 `RESET`
阶段都不会移动圆柱。

```bash
cd ~/WorkPlace/kcgtest1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch kcg_grasping cylinder_grasp.launch.py \
  gui:=false use_rviz:=false \
  run_baseline:=true shutdown_on_completion:=true
```

看到 `SCRIPTED GRASP PASSED` 才表示本轮通过；`FAILED` 会保留现场指标，不会伪造
成功。当前原始指尖碰撞网格下的脚本物理基线已经通过独立冷启动验收。带界面观察时
使用：

```bash
ros2 launch kcg_grasping cylinder_grasp.launch.py \
  gui:=true use_rviz:=false run_baseline:=true
```

自然演示的阶段顺序是 `OPEN → PRE_APPROACH → APPROACH → RESET → GRASP →
LIFT → HOLD`。终端中的 `RESET RESULT` 应包含
`Episode initialized without moving cylinder`，任务指标中的 `reset_mode` 应为
`stationary`。

包内仍保留 `fast_reset:=true` 和 `start_at_cylinder_pregrasp:=true` 的快速回归模式：
机器人从已验证的预抓位启动，圆柱先停放在工作区外，每次 episode reset 时再传送到
任务位。早期 `kcg_rl` 调用端已退役，快速模式本身仍可用于受控批量实验。

## 任务定义

- 物体：半径 `45 mm`、长度 `120 mm`、质量 `0.10 kg` 的竖直圆柱体。
- 演示初态：圆柱体中心 `z = 0.300 m`，从仿真启动开始位于台座上。
- 演示复位：不改动物体，只记录当时的真实位置作为抬升零点并对三路力矩去皮。
- 快速回归复位：保持固定高度和姿态，X/Y 对齐真实 `grasp_tcp` 后传送圆柱并清零速度。
- 抬升命令：TCP 从约 `z = 0.300 m` 垂直移动到约 `z = 0.400 m`。
- 成功条件：
  - 圆柱体净抬升不低于 `70 mm`（目标 `80 mm`，容差 `10 mm`）；
  - 圆柱体中心到 `grasp_tcp` 不超过 `75 mm`；
  - 上述条件连续成立至少 `3 s`。

成功评估使用 Gazebo 中的物体位姿作为训练/验收真值。三路力矩保留为策略观测和
夹紧控制反馈，但不会被误当成三个二值接触开关；姿态变化时，手指自重同样会改变
应变桥读数。

脚本化基线的动作顺序为：侧指预展开、关节轨迹到圆柱上方、下降到预抓位、建立
任务基准、三指径向闭合、控制器执行垂直抬升、保持、独立评估。抓取完全依靠
Gazebo 接触和摩擦，没有吸附插件，也没有把物体临时固定到机器人上。

## ROS 2 接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `/kcg_grasp/reset` | `std_srvs/srv/Trigger` | 演示模式不移动圆柱；快速回归模式复位位姿和速度；两者都重置任务状态并对三路力矩短时去皮 |
| `/kcg_grasp/evaluate` | `std_srvs/srv/Trigger` | 返回当前指标；物体掉落后成功状态会立即撤销 |
| `/kcg_grasp/observation` | `std_msgs/msg/Float64MultiArray` | 固定 27 维任务观测 |
| `/kcg_grasp/task_state` | `std_msgs/msg/String` | 便于调试的 JSON 指标 |
| `/kcg_grasp/phase` | `std_msgs/msg/String` | 脚本基线当前阶段 |
| `/finger_torque_broadcaster/joint_states` | `sensor_msgs/msg/JointState` | `f1j2, f2j1, f3j2` 三路单轴力矩，写入 `effort` 字段 |

实体手没有指尖触觉传感器。仿真内部在上述三个关节计算反力矩，只导出各关节局部
`z` 轴的一个标量，以对应三只惠斯通全桥应变片；其余五个力/矩分量不进入 ROS 观测。

手动调用任务接口：

```bash
ros2 service call /kcg_grasp/reset std_srvs/srv/Trigger '{}'
ros2 service call /kcg_grasp/evaluate std_srvs/srv/Trigger '{}'
```

## 27 维观测 `kcg_cylinder_observation_v1`

| 索引 | 数量 | 内容 | 坐标系/顺序 |
| ---: | ---: | --- | --- |
| 0–2 | 3 | 圆柱体相对位置 | `grasp_tcp` |
| 3–6 | 4 | 圆柱体相对姿态四元数 | `x, y, z, w`，相对 `grasp_tcp` |
| 7–9 | 3 | 圆柱体线速度 | Gazebo world |
| 10–12 | 3 | 圆柱体角速度 | Gazebo world |
| 13–16 | 4 | 主动手指关节位置 | `f1j1, f1j2, f2j1, f3j2` |
| 17–20 | 4 | 主动手指关节速度 | 同上 |
| 21–23 | 3 | 去皮后的单轴力矩增量 | `f1j2, f2j1, f3j2`，单位 `N·m` |
| 24 | 1 | 相对复位高度的抬升量 | m |
| 25 | 1 | 圆柱体中心到 TCP 的距离 | m |
| 26 | 1 | 当前连续满足保持条件的时间 | s |

## 与策略训练的边界

当前包只保留自然演示、确定性快速复位、三路单轴力矩观测和客观成功条件。早期
`kcg_rl` 策略、奖励与训练入口已退出活动树；任何新训练路线必须重新建立当前任务合同，
不能从这里的快速复位能力推断训练或动态验收已获授权。
