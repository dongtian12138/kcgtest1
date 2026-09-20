# kcgtest1 — 四相机电连接器装配基线

当前分支：`codex/connector-four-camera-baseline-20260920`。

本版保存高位全局相机1、固定全局相机2、掌心相机和腕部相机的完整仿真装配。已实际完成桌面抓取、两次观键及中间绕轴调整、两段搬运、插入、换抓旋紧和最终3秒完全松手；原物理、接触、键槽几何和影像验收通过。

实际完整动力学执行提交为`686734f`。末端判定曾将卸力落座误判为释放后不稳定；`84e534c`已修正并通过同回合原在线传感记录重放验证，没有声称该最后判定修复又重新执行了一整轮。原“未确认”记录和原版审核均保留。

- **[从其他电脑复现：下载、环境、输入、运行与验收](docs/REPRODUCE_FOUR_CAMERA_BASELINE_20260920_CN.md)**
- **[全过程录像与发布附件](https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-four-camera-high-global1-20260920)**
- [实际验收结果与证据](reproducibility/high_global1_acceptance_20260919/result_CN.md)
- [当前上下文](CURRENT_CONTEXT_CN.md)
- [项目协作约定](AGENTS.md)

```bash
git clone --branch codex/connector-four-camera-baseline-20260920 \
  https://github.com/dongtian12138/kcgtest1.git kcgtest1-four-camera
cd kcgtest1-four-camera
python3 scripts/prepare_assembly_reproduction.py --assets
# 按复现说明准备三个独立环境、视觉权重和原生记录模块后：
python3 scripts/run_current_hand_assembly.py --check
python3 scripts/run_current_hand_assembly.py --run
```

默认启动参数已绑定本次成功配置，并且每个新回合先执行新预检。运行资产共299个：297个来自版本化Release包，2个高位相机输入随Git提供。新回合必须重新验收，不能沿用旧报告的通过标记。

本轮录像5分13.8秒，计算用时2小时51分38秒；该耗时不是5倍提速结果。只用于仿真，`hardware_authorized=false`。早期基线与历史研究文档继续保留，但本分支请使用上面的新复现入口。
