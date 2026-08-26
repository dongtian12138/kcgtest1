# kcgtest1

KUKA iiwa、空间三指手与 D38999/J599 电连接器的仿真研究工程。工程当前仍是
simulation-only：`hardware_authorized=false`。静态测试、离线结果和程序退出码不能替代
Isaac 动态或真实硬件验证。

## 从这里开始

- 当前唯一任务入口：[`CURRENT_CONTEXT_CN.md`](CURRENT_CONTEXT_CN.md)
- 稳定协作与证据规则：[`AGENTS.md`](AGENTS.md)
- 当前抓取目标：[`docs/carts_v2/NORTH_STAR_CN.md`](docs/carts_v2/NORTH_STAR_CN.md)
- 当前 CONTACTOPT 方法：[`docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`](docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md)
- D38999 主包：[`src/kcg_connector/README.md`](src/kcg_connector/README.md)
- 目录边界：[`docs/PROJECT_STRUCTURE_CN.md`](docs/PROJECT_STRUCTURE_CN.md)
- 证据保留规则：[`docs/ARTIFACT_RETENTION_CN.md`](docs/ARTIFACT_RETENTION_CN.md)

历史报告、旧任务和旧运行结果不是当前指令。需要审计时从
`pre-clean-project-20260826` 等标签按单个路径读取，不恢复进活动树。

## 活动目录

| 路径 | 作用 |
| --- | --- |
| `src/kcg_connector/` | D38999 模型、抓取/装配逻辑、Isaac 接口与测试 |
| `src/iiwa_description/` | KUKA iiwa 与三指手描述 |
| `src/kcg_moveit1/` | ROS 2 / MoveIt 支撑 |
| `src/kcg_grasping/` | 基础圆柱抓取回归 |
| `scripts/carts_v2/` | 当前 CONTACTOPT 与局部手诊断入口 |
| `docs/carts_v2/` | 当前北极星、方法和精简决策 |
| `artifacts/` | 模型、日志、轨迹、报告和冻结证据；不是源码入口 |
| `j599_25_35_standard_interface_v1/` | 独立公共规格 J599 25-35 模型与证据 |

## 静态检查

```bash
PYTHONPATH=src/kcg_connector python3 -m pytest -q src/kcg_connector/test/carts_v2
```

动态运行前必须重新核对当前上下文、原始配置、资产哈希、进程和唯一输出路径；不要从
README 直接启动 Isaac。
