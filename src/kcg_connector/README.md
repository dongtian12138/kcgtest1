# kcg_connector

电连接器模型、三指手控制和 Isaac Sim 运行代码。

当前验收是同一回合的视觉定位、指甲抓本体、键槽对准插入、换抓螺母、旋拧到位及松手保持。
主线与组件状态见仓库根目录的 `CURRENT_CONTEXT_CN.md` 和 `docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md`。

复用的执行入口与共同控制：

- `isaac/run_body_assembly_with_video.py`：全链包装候选，视觉抓取和装配分支尚待接通。
- `isaac/carts_v2/run_grasp_lift.py`
- `isaac/carts_v2/controller.py`
- `isaac/carts_v2/evaluate_run.py`
- `isaac/carts_v2/engine_health.py`

`isaac/diagnose_saved_hand_wrench.py`是局部诊断入口；其预插入快照不能算视觉取件或完整装配成功。
当前目标和证据边界以仓库根目录 `CURRENT_CONTEXT_CN.md` 为准。历史搜索、离线优化和一次性
试验继续保留证据，不因此成为当前执行路线。
