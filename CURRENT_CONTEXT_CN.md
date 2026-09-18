# 当前任务：完整装配至少 5 倍性能优化

核验时间：2026-09-18T03:34:51.717977+00:00。**至少5倍尚未实现**。480Hz128/8局部旋拧与near-seat原精度验证通过，新完整候选正在运行。

## 验收与保留基线

- 封存成功对照：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/run`，295421步，收尾前15909.994568975s。新完整回合含记录/必要收尾/进程退出应<=3181.998913795s（53.03min），才是至少5倍。局部速度、失败回合、少走物理阶段不能代替完整验收。
- 保留实际14.605mm±10µm到位、源止挡正载、键槽侧壁<=2µm、原NAIL/PAD接触身份、末尾连续3s完全张手且原始手冲量严格0、同128簧套每帧正载、Body/Nut不休眠、后备轴向限位<0.5999mm且不承载及同回合影像。
- 原几何/质量惯量/材料/有限驱动与保护不放宽；仿真真值只用于事后评价。仅仿真，hardware_authorized=false。
- 原有效树`kcgtest1-improvements-20260916`、原始数据及主树未提交用户资产保留。优化树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/assembly-performance-5x-20260917`。未推送，无硬件/系统变更。

## 最新结果与当前运行

- 完整`artifacts/performance_5x/full_cpu960_software_fast_01/run`已结束：exit2，总1804.271685132s，148546步，物理及运输1656.5675s。原保护在编码插深0.69225mm时判定BLOCKED_WITHOUT_AXIAL_PROGRESS，真实最深0.696642mm，随后正常退至3.1018mm间隙。未进入旋拧；不是提速验收成果。原档案/索引/视频已封存。
- 此回合保留CPU960Hz64/4、原CAD/SDF/材料/控制限值、pin/Socket预测接触margin10µm；加入精确FK缓存、Fabric仅图像边界发布和自由直线路径重计时。最后视觉校正实际误差0.0296°，但其后长下降期间连接器相对原生手部发生约0.163°轴向转动；封存成功对照仅0.00329°。受阻处实际键方向约0.493°。`software_fast_alignment_posthoc.json`与`software_fast_grip_drift_posthoc.json`保留完整只读比较。尚不能把多项改变的全部差异归因到某一项；下一完整候选先关闭自由搬运重计时，恢复原路径时序。
- **first_turn_balanced480_retry02已结束并局部通过**：同封存第一轮旋拧源态211314，CPU480Hz128/8，原SDF/几何/力界、contact-last、pin margin10µm、f32无损记录/FK缓存/Fabric边界发布。完整90°指令、1861物理步，outer_abort无；键槽原标准通过，最差1.655694µm，末深9.238852mm。stepper39.6262s，对应960Hz64/4软件候选74.7848s；local含当前图像和收尾68.6192s vs111.2069s；进程164.729s vs207.533s。这些都不是完整5倍。原记录`artifacts/performance_5x/first_turn_balanced480_retry02/`。
- near_seat_balanced480已结束：240步=0.5s，按诊断步数正常停止(exit2/DIAGNOSTIC_STEP_BUDGET_REACHED)，local13.9993s；原键槽通过且无越界，末深14.6043126mm。与960Hz margin10同源local23.5359s相比约1.68倍，仅局部。`balanced480_local_comparison.json`记录两个窗口全部比较，注明near-seat另含FK/Fabric软件改动。
- 当前唯一主要物理运行：**artifacts/performance_5x/full_balanced480_01/run**，exec session55696。启动PID/UTC/完整argv在父目录motion_process.json；配置assembly_balanced_480hz.yaml与body_480hz.yaml，源码f939387。预检exit0/131.2955s，preflight/engine/identity均通过，实际CPU480Hz128/8读回通过。原搬运时序已恢复；保留pin margin10µm及记录/FK/Fabric优化。预算3600s、收尾240s，目标仍3181.9989s；额外记录preflight_plus_motion_s。恢复先核实实际进程/文件状态；安全停止写run/STOP_REQUEST。
- 新数值假设：每秒碰撞检测从960降到480，同时位置迭代每秒61440、速度迭代每秒3840不变。此前480Hz64/4完整中断前键槽4.92µm失败；960Hz16/4第一圈13.47µm失败而960Hz64/4同源1.17µm通过，说明更充分求解有直接精度依据。不是提高物理力或放宽容差。
- 193f548为局部诊断扩展，后续f939387接入显式480Hz128/8完整候选与margin许可。诊断部分：诊断CLI显式允许这一有界配置，并在contact-last/margin检查与实际作者记录中一致处理。7641ba0/b838170为前序提交；头两次first_turn_balanced480与retry01因旧CLI/交叉检查拒绝，尚未启动物理，原log保留。当前有效retry02已成功。

## 下一步

1. 核对full_balanced480_01的实际进程/当前阶段，继续观察初始视觉、真实抓取、对键插入、旋拧与最终松手承载。不凭旧状态等待。
2. 源码与参数在运行中冻结。结束后读取实际motion_process总时间，并按原body/nut/key/turns/terminal/band/whole审查。先核对实际结果，不能以程序exit或PASS代替完整装配。
3. 若完整物理通过且总耗时<=3181.9989s，才按5倍目标验收；连预检耗时也要明确报告。若失败或超时，定位最早实际原因后继续，不盲扫/调宽物理标准。
4. 全过程只一项主物理运行；独立离线分析可继续，但避免重CPU负载污染计时。

## 已验证计算改进与暂不采用路线

- 原生接触点复制+路径缓存；ext43保留SDK原float32全部原值，as_array仍升float64供后评。完整接触/事件/顺序保留；原始64帧分块MessagePack+标准gzip(ISA-L)及增量块索引。EncodedSensorHistory保存全样本字节，精确FK有限缓存返回独立矩阵。GC每步gen0、每1024步gen2及最终gen2。匹配真实数据/多项针对性测试已通过。
- 原CPU960Hz64/4、margin10µm匹配首90°有键槽证据；Fabric边界发布/FK缓存同源3721帧原生关节、力矩、Body/Nut位姿与前候选全部0差（software_optimization_physical_comparison.json）。这是局部物理等价，不是整轮等价。
- 先前完整full_fast240_01初始视觉拒绝；240Hz255/16仍拒绝。full_fast480_01完成前两圈及第三圈部分，45min中断，恢复121728完整帧；键槽4.92µm失败。full_cpu960_margin10_01约34.5min尚在key entry，受控停止；JSON紧凑contact见证写出错误已修复，不改原失败数据。
- GPU初始化根因已解决：166586排除关系导致逐对重传SDF。全有效pair编译30CollisionGroup、1513形状全部1143828对关系检查无差；但GPU480/960 near-seat键槽77/51µm失败。GPU非合格路线。CPU同组编译无性能增益。
- 原生CPU时间线显示窄相位接触为密集阶段主要成本。降位置迭代16、SDF8bit、CPU串行/核心绑定、四连杆少重配均未提供合格显著收益。
- 单180顶点pin合并实际SDKcook钳64顶点造成31µm外形损失，拒绝；精确轴向640凸块分解几何通过但收益小，未用于完整回合。
- 全指端改原凸分解后的真实重新抓握+90°，f2的327接触点超过原PAD投影0.5mm限（最大0.606mm），未采用。其他指端局部身份通过不证明全程。
- 当前离线网格候选未用于物理：nut_mesh_probe外Nut235728→124046三角面，但独立采样最大25.5µm且非闭合；thread_mesh_probe8个仅Nut接触的插座三角面约减25%，采样约30µm。QEM标称1µm不是Hausdorff保证；不作为无损优化或验收。原几何文件未改。

## 入口与环境

- Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`；规划 `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；视觉 `/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`；SAM根 `/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`。
- 运行环境ISAAC_ENV_PREFIX为Isaac conda目录，KCG_PLANNER_PYTHON/KCG_SAM6D_PYTHON/KCG_SAM6D_ROOT如上。Isaac6.0.1.0/PhysX110.1.13，RTX5070Ti16GB。不得在物理进程运行时重写已加载native so。
- 主回合安全停止写run/STOP_REQUEST；勿SIGINT（SDK可能跳过Python收尾）。functions.wait只用于exec返回的cell ID；shell会话用write_stdin。原始数据artifacts被忽略不代表可删。
- 源态：first_turn_source.json为211314/nut_rotation/nut_regrasp；near_seat_source.json为291580/continued_04；open_first_grip_source.json为206658/nut_regrasp。只能事前设置冷态用于局部诊断，不冒充连续装配。
- 原整体验收入口`reproducibility/improvements_20260916/review_completed_assembly.py`；key review脚本`src/kcg_connector/isaac/evaluate_source_key_containment.py`。必须读取实际原数据，保留失败。
- 详细优化历史见`docs/history/CURRENT_CONTEXT_CN_20260918_before_balanced480.md`；两次原成功与验收见`docs/history/CURRENT_CONTEXT_CN_20260917_before_5x_performance.md`。
