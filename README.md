# kcgtest1

KUKA iiwa、空间三指手和电连接器的 ROS 2 / Isaac Sim 工程。

当前入口：

- [`AGENTS.md`](AGENTS.md)：协作、事实和最小实施规则。
- [`CURRENT_CONTEXT_CN.md`](CURRENT_CONTEXT_CN.md)：当前唯一物理目标与活动入口。
- [`第一版视觉全链路计划`](docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md)：从本回合视觉、指甲抓本体到对键插入、换抓旋拧和松手的顺序与验收边界。
- `src/kcg_connector/isaac/run_body_assembly_with_video.py`：全链执行包装候选；视觉抓取与后续装配尚待按计划接通，不能视为已成功的一键入口。
- `src/kcg_connector/isaac/carts_v2/run_grasp_lift.py`：复用的场景、视觉与抓取运行核心。
- `src/kcg_connector/isaac/diagnose_saved_hand_wrench.py`：局部接触/旋拧诊断，不能代替视觉全链验收。
- `src/kcg_connector/isaac/verify_connector_engagement.sh`：连接器首段旋合、侧推支撑及螺母往返短验证，启动到结束有 3 分钟硬上限。

工程当前为 simulation-only：`hardware_authorized=false`。程序退出、静态测试和离线结果
不能证明连接器已经被抓住或抬起。
