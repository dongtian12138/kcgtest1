# 工程结构与责任边界

## 主线

`src/kcg_connector/` 是电连接器任务唯一主包；`src/iiwa_description/` 和
`src/kcg_moveit1/` 提供机器人描述、ROS 2 与 MoveIt 支撑。抓取动态控制主要位于
`kcg_connector/grasp/`，共享 Isaac 入口位于 `isaac/d38999_tabletop_pick_smoke.py`；
具体当前模块必须从控制面反查。

主线的数据流应保持为：

```text
公开规格/冻结合同
  → 确定性模型生成
  → 相机与腕部力觉观测
  → 三指抓取/对准/插入/锁紧控制
  → 原始硬门
  → 运行后独立评估与证据
```

## 支撑包

- `src/kcg_grasping/`：早期圆柱抓取回归，用于防止机器人与三指手基础能力退化。
- `src/kcg_moveit_collision_audit/`：规划场景碰撞审计。
- `src/kcg_rl/`：早期 residual RL 研究代码。是否开放训练必须以最新章程为准，不能
  默认把它当成当前完成路径。

## 历史与数据

- `ros1_original/`：迁移来源，只读保留，不参加 ROS 2 构建。
- `artifacts/agent_control/`：当前任务、状态、队列、门和不可覆盖历史。
- `artifacts/kcg_connector/`：模型、轨迹、图像、运行日志、报告和交付包。
- `docs/archive/`：过去 README、计划和交接说明；用于追溯，不用于判断当前状态。

## 工具

- `tools/agent_control/`：受控运行、检查和证据构建工具。
- `tools/experiments/`：一次性或历史分析脚本；不应被产品运行时导入。
- `scripts/`：工作空间级构建/验证脚本。

## 新代码放置规则

1. 可复用控制/评估逻辑放在 `src/kcg_connector/kcg_connector/`，同时提供定向测试。
2. Isaac 专用世界、渲染或 PhysX 入口放在 `src/kcg_connector/isaac/`。
3. 参数放在 `src/kcg_connector/config/`，禁止把正式门限散落在脚本常量中。
4. 一次性分析放在 `tools/experiments/`；完成后要么提升为受测模块，要么保留为历史，
   不再堆回包根目录。
5. 运行输出只进入唯一的新 `artifacts/` 子目录，禁止覆盖旧 run_id。

## 不在本轮做的重构

活动 Isaac 运行器体积很大，且抓取主线仍在快速变化。为避免改变物理行为，本轮只清理
与它隔离的文档、工具和旧时间窗脚手架；运行器拆分应在抓取节点形成新的冻结动态基线后
单独进行。
