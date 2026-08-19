# kcgtest1：KUKA iiwa + 空间三指手电连接器装配

本工程只有一个主目标：在仿真中让 KUKA iiwa 机械臂与 KCG 空间三指手完成
D38999 类电连接器的识别、抓取、对准、插入、锁紧和回到 Home。

当前仍是 **simulation-only** 工程：`hardware_authorized=false`。代码、单元测试、
离线报告或某个进程退出码都不能替代真实 Isaac 动态验收。

## 当前从哪里开始

- 人工查看当前状态：[`artifacts/agent_control/CURRENT_STATUS_CN.md`](artifacts/agent_control/CURRENT_STATUS_CN.md)
- Agent 恢复入口：[`artifacts/agent_control/AUTONOMOUS_RESUME_CN.md`](artifacts/agent_control/AUTONOMOUS_RESUME_CN.md)
- 当前任务与禁止项：[`artifacts/agent_control/CURRENT_TASK.md`](artifacts/agent_control/CURRENT_TASK.md)
- D38999 主包：[`src/kcg_connector/README.md`](src/kcg_connector/README.md)
- 工程目录说明：[`docs/PROJECT_STRUCTURE_CN.md`](docs/PROJECT_STRUCTURE_CN.md)
- 证据保留规则：[`docs/ARTIFACT_RETENTION_CN.md`](docs/ARTIFACT_RETENTION_CN.md)

状态以磁盘中的控制文件为准，不从本 README 推断。2026-08-19 清理时，A1/A2 已有
动态通过证据，当前前沿是 B-V3 三指抓取支撑转移；B 尚未动态通过，C 到 G 尚未进入。

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
| `ros1_original/` | 原 ROS 1 工程 | 只读迁移来源 |
| `docs/archive/` | 清理前的旧说明和交接文本 | 历史，不代表当前状态 |

## 构建与纯 CPU 检查

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

当前 B-V3 控制器的纯 CPU 定向测试：

```bash
PYTHONPATH=src/kcg_connector \
python3 -m pytest -q \
  src/kcg_connector/test/test_moment_constrained_support_transfer.py
```

不要从 README 直接猜测并启动 Isaac。动态运行前必须先读取当前控制面、完成对应预检，
并确认没有重复 run_id 或活动进程。

## 版本控制原则

- Git 跟踪源码、配置、测试和当前说明；`build/`、`install/`、`log/`、`.venv/` 与
  运行证据不进入普通提交。
- `artifacts/` 约 18 GB，其中有不可替代的失败证据与冻结模型。本轮整理不删除、
  不移动这些内容；仅跟踪其入口说明。
- 清理前完整源码状态保存在本地分支 `cleanup/project-structure-20260819` 的提交
  `c606865`，旧文件可从该提交逐项恢复。
- 禁止用 `git reset --hard`、强制推送或批量删除来“变干净”。
