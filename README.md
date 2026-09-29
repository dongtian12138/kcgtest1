# kcgtest1 — 按关节余量选择回转的完整四相机装配基线

当前分支：`codex/connector-adaptive-return-baseline-20260929`。

本版已在2026-09-29的同一个连续仿真回合中实际完成桌面抓取、两次观键、搬运、插入、换抓旋拧和最终3秒完全松手，原物理、源面、键槽、影像及严格释放验收均通过。实际执行提交为`6df30f1`，在线到位判断也随该完整回合实际执行。

张手后依据当前姿态与下一段关节可达性选择0°或±90°，随后重抓保留所选朝向。本轮实际回转为 **90°、90°、0°、90°、0°**，两次省去不必要空转；其余控制与原安全边界保持。

- **[下载、环境、运行与验收说明](docs/REPRODUCE_ADAPTIVE_RETURN_BASELINE_20260929_CN.md)**
- **[完整原录像与发布附件](https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-four-camera-adaptive-return-20260929)**
- [本轮完整验收结果与证据](reproducibility/adaptive_return_baseline_20260929/result_CN.md)
- [当前上下文](CURRENT_CONTEXT_CN.md)
- [项目协作约定](AGENTS.md)

```bash
git clone --branch codex/connector-adaptive-return-baseline-20260929 \
  https://github.com/dongtian12138/kcgtest1.git kcgtest1-adaptive-return
cd kcgtest1-adaptive-return
python3 scripts/prepare_assembly_reproduction.py --assets
# 按说明准备三个环境、固定视觉权重、原生记录模块和ROS资源后：
python3 scripts/run_current_hand_assembly.py --check
python3 scripts/run_current_hand_assembly.py --run
```

新回合先执行新预检，结束后必须按说明重新验收。本轮原五视角录像5分4.8秒，计算用时2小时42分54秒，不含后评；最终深度约14.604611mm，3秒2880步手冲量严格为零、止挡和同128夹片持续承载。

原通用CARTS退出码2和旧全指腹分类报告仍如实保留，不能代替专用全流程验收。仅仿真、`hardware_authorized=false`；一个名义回合和软件回归不构成任意极端工况或硬件保证。原2026-09-20基线及历史研究继续保留为比较来源。
