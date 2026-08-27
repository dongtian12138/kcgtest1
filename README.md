# kcgtest1

KUKA iiwa、空间三指手和电连接器的 ROS 2 / Isaac Sim 工程。

当前入口：

- [`AGENTS.md`](AGENTS.md)：协作、事实和最小实施规则。
- [`CURRENT_CONTEXT_CN.md`](CURRENT_CONTEXT_CN.md)：当前唯一物理目标与活动入口。
- `src/kcg_connector/isaac/carts_v2/run_grasp_lift.py`：当前 Isaac 抓取运行入口。

工程当前为 simulation-only：`hardware_authorized=false`。程序退出、静态测试和离线结果
不能证明连接器已经被抓住或抬起。
