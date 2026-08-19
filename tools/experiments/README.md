# experiments

这里保存一次性分析与已结束阶段的辅助脚本。它们不属于 ROS 2/Isaac 产品运行时，
不应被 `src/` 导入，也不作为当前动态任务入口。

- `legacy_keyed_v2/`：2026-08-19 整理时从 `tools/` 根目录移入的视觉、相机、键位、
  传输姿态和临时补丁分析脚本。

若某个脚本重新成为主线依赖，应先整理接口、补测试，再移入
`src/kcg_connector/kcg_connector/` 或 `tools/agent_control/`，不要直接从本目录引用。
