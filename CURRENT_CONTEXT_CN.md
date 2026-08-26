# kcgtest1 当前轻量上下文

> 快照时间：2026-08-26T18:25:00Z
> 当前分支：`carts-grasp-contactopt-1488-fast6h-20260826`
> 原始源码、配置、产物、Git、进程和时间戳优先于本摘要。

## 当前唯一优先级

用户最新要求是清理活动工程包，移除旧版本代码、旧阶段 Markdown 和过时 Python，
并防止历史要求覆盖新要求。清理状态为 `IMPLEMENTING`。

清理期间不启动 Isaac、不继续第一指实验、不改变物理配置。清理前已建立恢复标签：
`pre-clean-project-20260826`。

## 受保护的当前工作

以下未提交文件属于当前/并行 WIP，清理不得修改、暂存或删除：

- `scripts/carts_v2/evaluate_opposition60_first_finger_trace.py`
- `scripts/carts_v2/plan_opposition60_physical_contact_endpoint.py`
- `scripts/carts_v2/run_contactopt_seed_generation.py`
- `scripts/carts_v2/run_opposition60_local_contact.py`
- `src/kcg_connector/kcg_connector/grasp/carts_v2/structured_seed_generator.py`
- `scripts/carts_v2/audit_nailfree_graspgenx_seed_reuse.py`

## 清理前保留的物理事实

- 对象 B 候选 `contactopt_g_q08_a03_z1_p0` 只完成了 0.5 s 初始状态动态检查；
  该次运行未发闭指或抬升指令，手—物接触数为 0。
- 第一指真实接触、三指接触、离桌、50 mm 抬升、2 s 保持、整臂路径和额外扰动均尚未验证。
- `hardware_authorized=false`、`formal_dynamic_pass=false`、
  `research_dynamic_pass=false`。
- 清理前未发现活动的 pytest、Isaac Sim 或 `/kit/kit` 进程。

## 清理保留边界

- 保留当前 CONTACTOPT-1488 入口、第一指诊断入口及它们的传递依赖。
- 保留冻结模型、对象/手身份、物性、安全门限和全部证据数据。
- 历史方法的源码与文档只由 Git 标签恢复，不在活动树保存第二份。
- `artifacts/` 是证据区，不是默认源码或指令入口；不得从旧产物恢复运行授权。

## 最小读取路线

普通源码或清理工作只读：

1. `AGENTS.md`
2. 本文件
3. 待改源码及其直接配置/测试

只有明确恢复 CONTACTOPT 研究时再读：

1. `docs/carts_v2/NORTH_STAR_CN.md`
2. `docs/carts_v2/CONTACTOPT_1488_FAST6H_PLAN_CN.md`
3. `artifacts/carts_v2/contactopt_1488_fast6h/MANIFEST.json`
4. 当前 Git 状态、进程和最新产物

旧动态不会在清理完成后自动续跑；需要用户新的直接指令。
