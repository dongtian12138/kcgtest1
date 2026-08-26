# 工程结构与责任边界

## 当前活动链

```text
公开规格与冻结模型
  → CONTACTOPT-1488 结构化接触初值
  → 代理筛选与原始网格复核
  → 局部手 Isaac 接触诊断
  → 完整机械臂路径与抓举验证（尚未完成）
  → 运行后独立评价
```

- `src/kcg_connector/kcg_connector/grasp/carts_v2/`：当前抓取方法。
- `src/kcg_connector/kcg_connector/grasp/robust/`：当前方法复用的几何、区间、碰撞和
  受力安全内核；它是依赖层，不是另一条自动运行路线。
- `src/kcg_connector/isaac/carts_v2/`：通用 Isaac 控制与事后评价。
- `scripts/carts_v2/`：当前 CONTACTOPT、初始状态和第一指诊断的薄入口。
- `src/iiwa_description/`、`src/kcg_moveit1/`：机器人描述与 ROS 2 / MoveIt 支撑。

## 配置与冻结来源

- 当前方法入口配置是 `carts_contactopt_1488_fast6h.yaml`。
- `carts_surface_v2_fast6h.yaml` 和 `carts_nailfree_height_projected.yaml` 仍由当前
  依赖/受保护 WIP 读取，不能因名称看起来像旧阶段而删除。
- D38999 keyed-v2/R12 等生成器和合同保留用于冻结模型来源复核；历史任务名属于证据谱系，
  不是活动运行授权。

## 文档与证据

- 根目录只保留 `AGENTS.md`、`CURRENT_CONTEXT_CN.md` 和 `README.md` 三个入口。
- `docs/carts_v2/` 只保留当前北极星、当前方法和精简决策。
- `artifacts/` 保存不可替代的成功/失败证据与冻结资产，不参加默认源码搜索和测试。
- 旧源码、旧阶段文档和旧入口由 Git 标签 `pre-clean-project-20260826` 恢复；活动树
  不再维护 `legacy_archive/` 或源码 ZIP。

## 新增文件规则

1. 可复用逻辑放入 `src/kcg_connector/kcg_connector/` 并配套测试。
2. Isaac 专用入口放入 `src/kcg_connector/isaac/` 或当前薄脚本目录。
3. 参数只放配置文件，不在脚本中复制物理门限。
4. 一次性实验完成后删除，由 Git 历史恢复；不要把新版本继续堆入活动树。
5. 运行输出只进入新的 `artifacts/` run_id，禁止覆盖旧证据。
