# 当前状态：用户要求停止，保留记录层优化

核验时间：2026-09-18T03:50:33.101270+00:00。

## 当前指令

用户最新要求：“把明确能够加快时间进展且不会对效果产生任何影响的效果保留，其他可能有潜在风险的回退，然后停止工作，我们来交流一下。”

**已停止。当前无本任务仿真进程。不得沿历史计划自动恢复实验或继续追求5倍；后续仅交流，等待新的明确执行指令。**

## 有效保留状态

- 优化工作树：`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/assembly-performance-5x-20260917`。主树及原成功基线的用户资产未改；没有远端推送。
- 活动保留配置：`reproducibility/performance_20260917/assembly_recording_only_960hz.yaml`；除recording字段外，与封存 `reproducibility/improvements_20260916/config/visual_contact_last_gc128_task.yaml` 全部值严格一致。配合原 `src/kcg_connector/config/visual_assembly_v1_body.yaml`。
- 保留只读原生接触复制/路径缓存、无损紧凑点编码、全量传感器历史紧凑存储、标准gzip兼容ISA-L压缩，及必要JSON兼容/归档索引/STOP_REQUEST/阶段计时。全部原生点与每物理步记录保留。
- 求解/控制/碰撞/视觉恢复原基线：CPU960Hz64/4、原contact-last、原CAD/SDF/材料/接触参数、原运动时序/相机采样/力速保护、原Nut阶段GC每128步。FK缓存与Fabric延迟发布亦已回退。
- 实验性降频、求解配置、GPU、碰撞组/几何/SDF/接触margin、轨迹重计时、相机采样及大间隔GC代码/配置已从活动入口移除。原始实验数据保留，代码在Git历史及 `artifacts/performance_5x/user_requested_conservative_stop/before_rollback_sources.zip` 可恢复。

## 验证与结果边界

- 同源480帧 `all_native_comparison.json` 证明记录优化候选与原始参考全部字段和物理轨迹逻辑字节一致；`header_copy_probe/float32_result.json` 证明18339点/1889头全部值、顺序和事件精确一致。
- 回退后11项记录/编码兼容测试、12项GC/收尾测试通过；语法检查通过。没有再启动物理回合。
- **完整至少5倍未达到，也未宣称达到。** 原完整成功参考为15909.994568975s，新回合目标3181.998913795s；保留方案尚无完整回合倍率验收。
- 最后运行`artifacts/performance_5x/full_balanced480_rgbd2x_01/run`已因用户请求受控停止，进程于2026-09-18T03:45:01.287296+00:00退出，960帧原始档案/索引匹配、视频关闭。提前中止导致旧评价器缺少settle样本报错，不是装配结果。
- 两个原成功回合和资产保持：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/run`等。只有仿真证据，hardware_authorized=false。

## 说明与历史

- 保留/回退详情：`reproducibility/performance_20260917/CONSERVATIVE_STOP_CN.md`。
- 回退配置比较与源码快照：`artifacts/performance_5x/user_requested_conservative_stop/`。
- 过程历史：`docs/history/CURRENT_CONTEXT_CN_20260918_before_balanced480.md`；停止前最新状态在上述回退目录`context_before_final_stop.md`。这些不是当前执行授权。
- Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`；原项目`.venv`用于不含SDK的单元测试。当前不需要运行它们。
