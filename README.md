# kcgtest1：KUKA iiwa + 空间三指手电连接器装配

本工程只有一个主目标：在仿真中让 KUKA iiwa 机械臂与 KCG 空间三指手完成
D38999 类电连接器的识别、抓取、对准、插入、锁紧和回到 Home。

当前仍是 **simulation-only** 工程：`hardware_authorized=false`。代码、单元测试、
离线报告或某个进程退出码都不能替代真实 Isaac 动态验收。

## 当前从哪里开始

- 新对话轻量入口：[`CURRENT_CONTEXT_CN.md`](CURRENT_CONTEXT_CN.md)
- 人工查看当前状态：[`artifacts/agent_control/CURRENT_STATUS_CN.md`](artifacts/agent_control/CURRENT_STATUS_CN.md)
- 当前任务原始合同：[`artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json`](artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json)
- 完整历史恢复入口（按需）：[`artifacts/agent_control/AUTONOMOUS_RESUME_CN.md`](artifacts/agent_control/AUTONOMOUS_RESUME_CN.md)
- D38999 主包：[`src/kcg_connector/README.md`](src/kcg_connector/README.md)
- 工程目录说明：[`docs/PROJECT_STRUCTURE_CN.md`](docs/PROJECT_STRUCTURE_CN.md)
- 证据保留规则：[`docs/ARTIFACT_RETENTION_CN.md`](docs/ARTIFACT_RETENTION_CN.md)

普通工作先按轻量入口定向读取；历史审计、状态冲突、正式 gate 变更和动态启动仍以磁盘
中的原始控制文件与运行产物为准，不能从本 README 推断。

## 目录边界

| 路径 | 作用 | 当前定位 |
| --- | --- | --- |
| `src/kcg_connector/` | D38999 模型、控制、Isaac 运行器与测试 | 主线 |
| `src/iiwa_description/` | KUKA iiwa 与三指手描述 | 主线依赖 |
| `src/kcg_moveit1/` | ROS 2 / MoveIt 配置 | 主线依赖 |
| `src/kcg_grasping/` | 圆柱抓取回归 | 支撑回归，不是最终任务 |
| `src/kcg_moveit_collision_audit/` | 规划场景碰撞审计 | 支撑工具 |
| `src/kcg_rl/` | 早期 residual RL 实验 | 非当前动态主线 |
| `tools/agent_control/` | 受控验证与证据工具 | 运维工具 |
| `artifacts/` | 模型、运行日志、报告与冻结证据 | 数据区，不是源码区 |
| `legacy_archive/` | 旧源码、文档、实验脚本和 ROS 1 工程 | 统一历史归档，不参加当前运行 |

## 构建与纯 CPU 检查

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

一个不绑定历史运行器源码哈希的纯 CPU 基础回归示例：

```bash
PYTHONPATH=src/kcg_connector \
python3 -m pytest -q \
  src/kcg_connector/test/test_geometry.py
```

不要从 README 直接猜测并启动 Isaac。动态运行前必须先读取当前控制面、完成对应预检，
并确认没有重复 run_id 或活动进程。

## 版本控制原则

- Git 跟踪源码、配置、测试和当前说明；`build/`、`install/`、`log/`、`.venv/` 与
  运行证据不进入普通提交。
- `artifacts/` 体积会随运行增长，其中有不可替代的失败证据与冻结模型。可删除每次
  Isaac 独立环境自动生成的 cache，但不得据此删除轨迹、报告、日志或冻结资产。
- 本轮合同清理前的已提交源码由本地标签 `pre-contract-cleanup-20260821` 锚定；被移出
  主线的文件用 `git show <tag>:<path>` 或 `git archive` 恢复，不在主分支重复保存源码 ZIP。
- 当前有效分支以 `git branch --show-current` 的实时结果为准；未经明确要求不推送标签或提交。
- 禁止用 `git reset --hard`、强制推送或批量删除来“变干净”。
