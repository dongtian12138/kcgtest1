# kcg_connector

这是 kcgtest1 的 D38999 电连接器装配主包。它包含仿真模型合同、三指抓取与装配控制、
Isaac Sim 运行入口、独立评估器和纯 CPU 测试。

工程仍是 `simulation-only`，不能据此授权真实硬件，也不能把静态或离线结果写成
`DYNAMIC_PASS`。

## 当前入口

- 轻量任务上下文：[`../../CURRENT_CONTEXT_CN.md`](../../CURRENT_CONTEXT_CN.md)
- 当前任务合同：[`../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json`](../../artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TASK_SWITCH_PLAN.json)
- CARTS-Grasp 方法实现：[`kcg_connector/grasp/robust/`](kcg_connector/grasp/robust/)
- CARTS-Grasp Isaac 接口：[`isaac/robust_grasp/`](isaac/robust_grasp/)
- 当前配置：[`config/carts_grasp_v1.yaml`](config/carts_grasp_v1.yaml)
- 当前定向回归：[`test/robust_grasp/`](test/robust_grasp/)

旧 `d38999_tabletop_pick_smoke.py`、B-V3/H1-H25 和固定候选路线只作历史证据或
baseline。动态运行是否获准、允许的 run_id、冻结摘要和下一动作都以控制面为准；
README 不作为运行授权。

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

`isaac/` 中仍有被历史 V2 生成链引用的探针，`config/` 中也有冻结旧版本。文件名带有
`v1/v2/r7/r12` 并不自动表示当前有效；默认测试已移除只校验旧源码文本/哈希的合同，
但准备删除运行文件时仍应从当前任务和证据引用反查。

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

清理前的长版阶段记录由 Git 标签保存，不再占据活动路径。需要查看时执行：

```bash
git show pre-contract-cleanup-20260821:legacy_archive/outdated_docs/KCG_CONNECTOR_README_before_cleanup_20260819.md
```
