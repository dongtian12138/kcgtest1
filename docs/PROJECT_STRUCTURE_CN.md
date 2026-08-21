# 工程结构与责任边界

## 主线

`src/kcg_connector/` 是电连接器任务主包；`src/iiwa_description/` 和
`src/kcg_moveit1/` 提供机器人描述、ROS 2 与 MoveIt 支撑。当前抓取研究入口是
`kcg_connector/grasp/robust/`、`isaac/robust_grasp/` 和四份 `carts_*.yaml`
配置；具体任务和动态授权仍必须从 `CURRENT_CONTEXT_CN.md` 与控制面反查。

`isaac/d38999_tabletop_pick_smoke.py` 是旧 V2/B 路线仍需兼容的单体运行器，不是
当前 CARTS 入口。它暂时保留是因为现有验证脚本和下游兼容链仍有真实引用；在新的
CARTS 动态基线形成前，不把它拆分或误写成当前正式候选生成器。

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

- `artifacts/agent_control/`：当前任务、状态、队列、门和不可覆盖历史。
- `artifacts/kcg_connector/`：模型、轨迹、图像、运行日志、报告和交付包。
- `legacy_archive/`：统一保存退出主线的源码、旧文档、旧实验脚本和 ROS 1 迁移来源；
  整棵目录不参加当前构建、默认测试或运行。
- `j599_25_35_standard_interface_v1/`：独立的 J599 25-35 公共标准接口模型、生成
  资产、渲染和接受证据。大体积原始接触/轨迹已进入带逐文件哈希的可恢复压缩归档；
  该模型不等于厂商原始 CAD，也不构成真实硬件验收。

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

## 暂不执行的高风险重构

旧单体 Isaac 运行器体积很大，且抓取主线仍在快速变化。当前只删除没有入向引用的
一次性历史入口，并压缩静态合同数据；仍被兼容链引用的运行器拆分应在 CARTS 形成新
的冻结动态基线后单独进行。
