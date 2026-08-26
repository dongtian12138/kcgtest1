# kcg_connector

D38999/J599 电连接器仿真主包，包含冻结模型合同、CONTACTOPT 抓取逻辑、装配控制、
Isaac 接口和纯 CPU 回归。工程仍是 simulation-only。

## 当前入口

- 任务路由：[`../../CURRENT_CONTEXT_CN.md`](../../CURRENT_CONTEXT_CN.md)
- 抓取目标：[`../../docs/carts_v2/NORTH_STAR_CN.md`](../../docs/carts_v2/NORTH_STAR_CN.md)
- 当前方法：[`../../docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`](../../docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md)
- 当前实现：[`kcg_connector/grasp/carts_v2/`](kcg_connector/grasp/carts_v2/)
- 共享安全内核：[`kcg_connector/grasp/robust/`](kcg_connector/grasp/robust/)
- 当前配置：[`config/carts_contactopt_1488_fast6h.yaml`](config/carts_contactopt_1488_fast6h.yaml)
- 当前测试：[`test/carts_v2/`](test/carts_v2/)

`grasp/robust/` 中保留的是当前 CONTACTOPT 的传递依赖及通用安全能力；文件名中的历史
版本号和任务名只表示模型谱系，不能自行恢复成运行路线。

## 包内边界

| 路径 | 内容 |
| --- | --- |
| `kcg_connector/grasp/carts_v2/` | 当前候选、接触、筛选和评价逻辑 |
| `kcg_connector/grasp/robust/` | 几何、区间、碰撞、受力与 IK 依赖 |
| `isaac/carts_v2/` | Isaac 控制器、引擎健康和事后评价 |
| `config/` | 当前配置及冻结模型合同 |
| `test/carts_v2/` | 当前方法定向回归 |
| `assets/public_specs/` | 可公开追溯规格来源，不是制造 CAD |

## 检查

```bash
PYTHONPATH=src/kcg_connector python3 -m pytest -q src/kcg_connector/test/carts_v2
```

测试通过只属于静态证据。动态运行必须重新核对当前上下文、配置、冻结资产、进程和输出
路径；`hardware_authorized=false` 不因测试改变。
