# 已按用户要求停止，仅保留记录层优化

状态核验：2026-09-18T03:50:33.101270+00:00。用户要求保留明确加速且不影响装配效果的改动，回退有潜在风险的路线并停止工作。当前无本任务仿真进程，未启动新的物理实验。至少5倍完整回合目标尚未达到。

保留的配置为 [assembly_recording_only_960hz.yaml](assembly_recording_only_960hz.yaml)。以封存配置 `reproducibility/improvements_20260916/config/visual_contact_last_gc128_task.yaml` 为依据，除 `recording` 字段外，全部配置值严格相同。使用原 `src/kcg_connector/config/visual_assembly_v1_body.yaml` 的960Hz时间步。

保留内容：

- 只读原生接触报告复制与不变路径名缓存，首次真实报告与原解码器交叉核验。
- 接触点无损紧凑编码、标准gzip兼容的ISA-L压缩，逐物理步保留全部原生接触点。
- 全量传感器历史的紧凑存储；读取接口保留原数值、顺序、切片和快照语义。
- 紧凑记录的JSON兼容写出修复、归档块索引、控制边界STOP_REQUEST及阶段耗时记录。

已回退：240/480Hz、迭代配置变更、GPU候选、碰撞排除组编译、碰撞几何/SDF候选、pin/socket接触margin变化、路径重计时、运动学缓存、Fabric延迟发布、初始相机分辨率变化，以及扩大GC间隔/GC阶段范围。相关实验配置和辅助模块已从活动实现中移除。

物理求解与控制恢复封存基线：CPU960Hz、64位置/4速度迭代、原接触后求解顺序、原完整源CAD/SDF与材料、原接触参数、原运动时序、原相机采样和原力/速度保护。GC也恢复原Nut阶段每128物理步全代回收及最终清理。

证据与边界：

- `artifacts/performance_5x/all_native_comparison.json`：同源480帧全部字段及物理轨迹逻辑字节一致。
- `artifacts/performance_5x/header_copy_probe/float32_result.json`：18339个真实SDK接触点、1889个头的值/顺序/事件精确一致；原SDK float32提升后的Python数值不变。
- 回退后的11项紧凑记录/归档兼容测试及12项GC/收尾测试通过；相关Python语法检查通过。未为此再跑完整回合。
- 这些证据支持保留纯记录层改动；不把局部等价和局部提速写成所有场景的完整验证，也不宣称已经达到5倍。

原成功基线、全部失败记录及同回合视频未删除。被回退的实现可从Git历史de10c99及 `artifacts/performance_5x/user_requested_conservative_stop/before_rollback_sources.zip` 恢复；归档不作为当前执行入口。

最后运行 `full_balanced480_rgbd2x_01` 已响应用户停止请求并退出，保存960帧封存原始档案及关闭后的视频。因在正式settle/抓取前中止，旧评价器报告缺少settle样本；该运行不计完成或成功。完整进程退出时间见其 `motion_process.json`。

后续仅交流。没有新明确执行指令前，不启动仿真、不继续优化、不自动恢复旧实验。
