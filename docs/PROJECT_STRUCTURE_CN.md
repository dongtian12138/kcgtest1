# 工程结构与责任边界

## 主线

`src/kcg_connector/` 是电连接器任务主包；`src/iiwa_description/` 和
`src/kcg_moveit1/` 提供机器人描述、ROS 2 与 MoveIt 支撑。当前抓取研究入口是
`kcg_connector/grasp/robust/`、`isaac/robust_grasp/` 和四份 `carts_*.yaml`
配置；具体任务和动态授权仍必须从 `CURRENT_CONTEXT_CN.md` 与控制面反查。

旧 V2/B 单体运行器、B-V2/H1-H25 控制器、R12 multilayer 控制栈和 residual RL
已退出活动源码；需要历史审计时从 `pre-active-route-prune-20260823` 按路径读取。
冻结模型生成器和合同仍保留在活动树，用于复核模型来源，不能据此重新授权旧路线。

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

## 历史与数据

- `artifacts/agent_control/`：当前任务、状态、队列、门和不可覆盖历史。
- `artifacts/kcg_connector/`：模型、轨迹、图像、运行日志、报告和交付包。
- `legacy_archive/`：统一保存退出主线的源码、旧文档、旧实验脚本和 ROS 1 迁移来源；
  整棵目录不参加当前构建、默认测试或运行。
- `j599_25_35_standard_interface_v1/`：独立的 J599 25-35 公共标准接口模型、生成
  资产、渲染和接受证据。大体积原始接触/轨迹已进入带逐文件哈希的可恢复压缩归档；
  该模型不等于厂商原始 CAD，也不构成真实硬件验收。

## 工具

- `tools/deepseek_consult.py`：授权时使用的外部模型协作入口。
- `scripts/bootstrap.sh`：工作空间环境引导；它不构成动态运行授权。

## 新代码放置规则

1. 可复用控制/评估逻辑放在 `src/kcg_connector/kcg_connector/`，同时提供定向测试。
2. Isaac 专用世界、渲染或 PhysX 入口放在 `src/kcg_connector/isaac/`。
3. 参数放在 `src/kcg_connector/config/`，禁止把正式门限散落在脚本常量中。
4. 一次性分析放在 `tools/experiments/`；完成后要么提升为受测模块，要么保留为历史，
   不再堆回包根目录。
5. 运行输出只进入唯一的新 `artifacts/` 子目录，禁止覆盖旧 run_id。

## 退役路线恢复

活动树不再为旧执行路线保留兼容调用链。需要复核旧实现时使用 Git 标签读取单个文件
或导出限定路径，不要把整条历史路线恢复到当前 `PYTHONPATH`。
