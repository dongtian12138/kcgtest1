# ROS 2 迁移与本机适配记录

日期：2026-08-11

## 范围与保留策略

- 只迁移 KUKA iiwa 14 + KCG 空间三指手系统。
- 原始压缩包未修改。
- 原 ROS 1 Catkin 工作空间完整移动到 `ros1_original/kcgtest1`。
- 原 ROS 1 配置分别保留在各包的 `ros1_config`、`ros1_launch` 和
  `ros1_urdf` 中，供追溯使用，不参与 ROS 2 构建。

## 已完成的迁移

1. 将 `iiwa_description` 和 `kcg_moveit1` 转换为 `ament_cmake` 包。
2. 将 ROS 1 XML launch 转换为 ROS 2 Python launch。
3. 建立统一的 `handarm.urdf.xacro`，同时支持：
   - `mock_components/GenericSystem` 模拟硬件；
   - `gazebo_ros2_control/GazeboSystem` Gazebo 物理仿真。
4. 使用 ROS 2 原生 mimic 配置替代旧 ROS 1 mimic Gazebo 插件，不再构建
   `roboticsgroup_upatras_gazebo_plugins`。
5. 配置两个 `JointTrajectoryController`，分别控制 7 轴 iiwa 和 4 个主动手指关节。
6. 迁移 MoveIt 2 的 SRDF、KDL、OMPL、关节约束及控制器映射，并增加
   `handarm` 联合规划组。
7. 将机械臂碰撞网格切换到已有简化网格；为三指手生成凸包碰撞网格，视觉网格保持不变。
   同时将 `link_7_s.stl` 从 ASCII STL 无损转换为 RViz2 可读取的二进制 STL。
8. 修正从动手指关节类型和限位，并为手指关节加入阻尼及摩擦。
9. 增加独立物理诊断状态流，避免 Humble Gazebo 的 `*_mimic` 接口名污染
   MoveIt 使用的标准 `/joint_states`。
10. 增加自动化物理稳定性检查与 MoveIt 规划/执行冒烟测试。
11. 新增独立 `kcg_grasping` 包，实现“抓住圆柱体—抬起—保持”任务世界、确定性
    reset、27 维观测和成功评估服务。
12. 为手部增加 `grasp_tcp`；保留三个原始末端网格，并在 `f1j2`、`f2j1`、`f3j2`
    处导出三个单轴关节反力矩，以对应实体手的惠斯通全桥应变片。没有增加指尖触觉、
    吸附或临时固定物体。
13. 增加脚本化基线以及控制器/关节状态启动门控，完整覆盖 MoveIt 接近、闭合、抬升、
    保持和结果验收。

## 碰撞网格简化结果

| 网格类别 | 原始三角面数 | 凸包三角面数 |
| --- | ---: | ---: |
| `handbase` | 732478 | 702 |
| `f1Link1` / `f3Link1` | 128942 | 300 |
| `f1Link2` / `f2Link1` / `f3Link2` | 20266 | 1032 |
| 末端指节 | 14192 | 1100 |

这样显著降低了 MoveIt 碰撞检测和 Gazebo 接触计算负担。视觉外观仍使用原始 STL。

## 本机验收结果

- `rosdep check --from-paths src --ignore-src -r`：通过。
- `colcon build --symlink-install`：3 个 ROS 2 包通过。
- Xacro 展开与 `check_urdf`：模拟硬件和 Gazebo 两种模型均通过。
- XML、YAML、Python 静态检查：通过。
- 模拟硬件直接控制：机械臂与手部轨迹均 `SUCCEEDED`。
- Gazebo 直接控制：机械臂与手部轨迹均 `SUCCEEDED`，4 个 mimic 关节正确跟随。
- MoveIt + OMPL + Gazebo：机械臂和手部均成功规划并执行。
- RViz2 图形实测：OpenGL 4.6 初始化、机器人模型和规划场景加载通过；碰撞网格无加载错误。
- 最终 30 秒无控制物理测试：通过。
  - 关节样本：2534
  - 连杆样本：1266
  - 最大绝对关节速度：0.049 rad/s
  - 最大手部连杆到基座距离：1.661 m
  - 最大 mimic 跟随误差：0.003669 rad
  - 未发现 NaN、关节缺失、手部断开或模型飞散
- 三路单轴力矩发布链路已验证：话题仅包含 `f1j2`、`f2j1`、`f3j2`，静止、夹紧与
  脱离阶段读数可区分，并在 episode 复位后自动短时去皮。
- 先前依赖额外刚性指腹盒体得到的抓取通过结果已经作废；恢复原始末端网格后，当前
  脚本可以形成三点接触并将圆柱体带离台面约 `10 mm`，但随后滑脱，因此尚未通过
  `70 mm + 3 s` 的任务验收。

## 为后续 RL 保留的边界

当前已经建立稳定仿真底座和可重复的第一个任务，但尚未加入训练框架。下一阶段建议按以下顺序扩展：

1. 基于现有 27 维观测确定第一版动作空间和奖励函数。
2. 增加固定频率的 `reset()` / `step()` RL 适配层以及终止、截断逻辑。
3. 加入初始位姿、质量、摩擦和传感噪声随机化，同时保留无随机化基准模式。
4. 用同一成功评估接口对比脚本基线与策略，再接入并行训练。
5. 对接实体前标定手指柔顺性、摩擦、关节力矩/速度和通信延迟。

## 已知限制

- 当前 Gazebo 使用位置接口，尚未建模真实执行器力矩环、延迟、噪声和安全控制器。
- Gazebo 模型级 self-collision 当前关闭；MoveIt 仍使用 SRDF 自碰撞矩阵。
- 未配置 3D 感知，因此不生成动态 Octomap。
- 实体只有三路惠斯通全桥单轴力矩，不存在指尖触觉。仿真已经匹配通道和测量轴，
  但仍需用实体数据标定桥路零点/增益、温漂、重力补偿、噪声和饱和范围。
- 原始末端网格下的圆柱抓取基线尚未通过，当前主要瓶颈是持续夹紧预载和滑脱；不得
  用额外指腹几何、吸附或成功锁存掩盖该问题。
- 本机 MoveIt 2.5.9 在停止 `move_group` / MoveIt RViz 插件时存在已复现的退出阶段段错误；
  运行、显示、规划和执行阶段未出现该问题。
