# kcg_connector

这是 kcgtest1 的 D38999 电连接器装配主包。它包含仿真模型合同、三指抓取与装配控制、
Isaac Sim 运行入口、独立评估器和纯 CPU 测试。

工程仍是 `simulation-only`，不能据此授权真实硬件，也不能把静态或离线结果写成
`DYNAMIC_PASS`。

## 当前入口

- 当前任务：[`../../artifacts/agent_control/CURRENT_TASK.md`](../../artifacts/agent_control/CURRENT_TASK.md)
- 恢复检查点：[`../../artifacts/agent_control/AUTONOMOUS_RESUME_CN.md`](../../artifacts/agent_control/AUTONOMOUS_RESUME_CN.md)
- 活动 Isaac 运行器：[`isaac/d38999_tabletop_pick_smoke.py`](isaac/d38999_tabletop_pick_smoke.py)
- B-V3 历史检查点控制器：[`kcg_connector/grasp/moment_constrained_support_transfer.py`](kcg_connector/grasp/moment_constrained_support_transfer.py)
- B-V3 历史检查点配置：[`config/d38999_moment_constrained_support_transfer_v1.yaml`](config/d38999_moment_constrained_support_transfer_v1.yaml)
- 对应定向回归：[`test/test_moment_constrained_support_transfer.py`](test/test_moment_constrained_support_transfer.py)

动态运行是否获准、允许的 run_id、冻结摘要和下一动作都以控制面为准。README 不作为
运行授权。

## 包内结构

| 路径 | 内容 |
| --- | --- |
| `kcg_connector/` | 与模拟器解耦的模型、控制器、评估和安全逻辑 |
| `kcg_connector/grasp/` | 当前与历史三指抓取、支撑转移控制 |
| `isaac/` | Isaac 场景、探针和动态验证入口 |
| `config/` | 模型、场景、抓取和安全配置 |
| `test/` | 纯 CPU 契约、回归和防火墙测试 |
| `assets/public_specs/` | 可公开追溯的规格来源，不是制造 CAD |
| `docs/` | 包级设计说明 |

`isaac/` 中存在按历史阶段保留的探针，`config/` 中也存在冻结旧版本。文件名带有
`v1/v2/r7/r12` 并不自动表示当前有效；应从控制面和证据引用反查，不能仅凭名称删除。

## 开发检查

```bash
PYTHONPATH=src/kcg_connector python3 -m pytest -q src/kcg_connector/test
```

上面的命令是纯 CPU 测试，不等于 Isaac 动态验收。启动 Isaac 前还必须执行当前任务
要求的源码、配置、冻结资产、进程空闲和输出唯一性预检。

## 证据边界

- B 阶段安全量的含义、阈值与软/硬门分类必须读取最新章程和当前任务；历史
  `0.30 N·m` 不得脱离最新授权解释为实体硬件硬门。任何当前原始硬门都不得用滤波
  延迟或离线重算替代。
- 在线控制不得读取对象真值位姿、碰撞对象名、接触法向、事件真值或语义实例真值。
- 仿真真值只可进入运行后独立评估。
- 冻结模型、失败证据和历史运行均保存在 `artifacts/`，不随普通源码清理移动。

清理前的长版阶段记录已原样保存为
[`../../legacy_archive/outdated_docs/KCG_CONNECTOR_README_before_cleanup_20260819.md`](../../legacy_archive/outdated_docs/KCG_CONNECTOR_README_before_cleanup_20260819.md)。
