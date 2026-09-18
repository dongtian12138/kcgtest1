# 当前任务：完整装配至少 5 倍性能优化

核验时间：2026-09-18T02:52:47.077819+00:00。完整至少5倍尚未达成；第二个CPU960Hz软件优化完整候选正在运行。

## 验收

- 对照封存第二轮完整名义回合：295421 步，主程序收尾前 15909.994568975 秒，约265.17分钟。目标同一完整流程含记录与收尾 <=3181.998913795秒（约53.03分钟），至少5倍。不能以局部速度、提前失败、少执行步骤或移出必要收尾来验收。
- 原完整装配验收保留：实际到原止挡、原2 µm键槽标准、真实手接触、连续3秒完全松手、同128簧套承载且Body/Nut不休眠、后备限位不承载。原几何、质量惯量、材料事实、有限驱动和力速保护不放宽。
- 两次有效成功证据及参数见本文件归档 `docs/history/CURRENT_CONTEXT_CN_20260917_before_5x_performance.md`。硬件未验证。

## 当前状态（5倍尚未实现）

- 本地已提交bad29f6（原生点复制/紧凑记录/传感历史/ISA-L）和ab811e1（480Hz候选配置），均未推送。其后有未提交的原生报告头缓存、停止和索引改动。原成功树与原始数据不改。
- 密集0.5秒基线：481步合计61.511秒。原生点复制后约43秒，紧凑记录后约34.8秒。字节重编码核查均保留全部记录内容。
- 新原生报告头路径缓存已加入`native/contact_reports.cpp`。真实SDK向量离线回放：事件头+完整报告6.251→1.132ms。实际960Hz/480步`all_native_960`合计27.639秒，原生完整回调0.977秒、事件头0.0855秒，其余物理/SDK仍占主要成本。与contact_last_p64的480步所有raw字段/物理轨迹逻辑字节相同（all_native_comparison.json）。单靠软件记录优化当前约2.2倍，不是5倍。
- C++构建：Isaac Python运行`scripts/build_contact_copy.py`，生成两个本地so，不入Git。本地pybind11=2.13.6，SDK ABI v5/gcc/cxxabi1016；ContactData=64字节、Header=72字节，首两份真实报告与原Python参考核查。`.deps/recording`为isal1.8.0。
- `EncodedSensorHistory`保留全历史字节和512条解码缓存；32768行GC扫描0.04766→0.001665秒。ISA-L真实密集行压缩6.947→1.307ms，标准gzip解压完全相同。相关记录/归档/原生接触12项unittest已通过；后评selective reader已适配extension42并对真实数据比较一致。
- 240Hz/64/4中段120步=0.5秒约8.153秒，局部约7.5倍；近坐底冷诊断键槽通过但深度变为14.611286mm。冷诊断不能代替连续装配。
- 完整`full_fast240_01`：预检通过，正式85.44秒时初始视觉拒绝，未抓取。实际Body轴倾0.001082rad，原参考0.000034rad；图像倾角裕度0.011851>原0.0112，视觉界未改。
- 完整`full_fast480_01`：预检和初始视觉通过，执行抓取/搬运/入槽、前两次90度指令及第三次32.78度传动余量换抓。45分钟仍在转移，投影不满足53分钟目标，于2728秒请求停止，不计完成或提速成功。SDK信号处理直接退出导致无收尾索引/分项计时。原2.92GB压缩文件未改，恢复出1902个完整gzip块/121728帧，无文件尾截断，未刷出的内存帧数未知。外部`recovered_partial_index.json`明确标为中止回合，未伪造结束或成功marker。
- 已修正停止方式：完整RecordedStepper绑定run/STOP_REQUEST，后续在控制步边界停止；每块写`.blocks.jsonl`，在阶段切换写`phase_timing.jsonl`。这些是为保留失败证据和定位性能，不替代物理验收。
- 高起始迭代假设已否定：`initial_precision_probe`在240Hz/255位置/16速度下预检通过，但初始视觉仍拒绝（裕度0.012113>0.0112）。不继续调宽视觉阈值或盲扫此候选。
- 已查清并解决GPU初始化异常：CUPTI记录原始场景每秒数千次重复申请/释放50,230,784和63,479,296字节。原生调用栈指向PxgGeometryManager；官方公开InternalFilteredPairs.cpp证明每个排除对都调用resetFiltering整actor。模型166586条排除关系触发大量重复几何上传。只删除禁用API/只编译叶子对均未解决。
- 新runtime_collision_groups.py将包括刚体/关节树/目标子树的全部有效pair映射成30个CollisionGroup；1513碰撞形状全部1143828个不同形状对检查，图差异0，另用USD官方ComputeCollisionGroupTable检查。原几何/材料/质量/位姿/驱动/源资产不改。禁用API1047个、不可见无子且非刚体叶919个只在运行时省略。兼容问题：Kit USD25.11内置LoadUsdPhysicsFromRange在此组合场景段错误；已将只读所有权解析隔离到现有planner环境USD26.8，输入保留为运行目录collision_filter_ownership/composed_scene.usdc，结果ownership.json。完整编译后GPU约24秒完成reset、RSS峰约9GB。
- GPU速度与精度分开：gpu_all_filters_960_retry01中段480步总24.30秒（stepper24.06），CPU同优化约27.6秒；gpu_all_filters_480中段240步合计12.09秒/收尾12.38秒，按相同0.5秒模拟时间仅局部约5.08倍，不能当整回合验收。1024步一次全代GC用于后者，逐帧gen0及最终gen2仍执行。
- GPU路线当前不合格：gpu_near_seat_480近坐底240步键槽最差77.213µm越界、最终深度14.665795mm；只恢复960Hz的gpu_near_seat_960仍50.722µm越界、深度14.625085mm。原标准2µm/目标14.605mm(±10µm)保持。不得用这些候选启动接受性整回合或说完成。main已加入显式GPU-host诊断profile和配套后端身份检查，但未通过物理验收，非默认。
- CPU480也不满足原精度：对full_fast480_01中断文件恢复出的72071..121727帧只读复查，第一圈键槽最差4.922554µm，另一个键3.847689µm。partial_key_containment_review.json明确只覆盖恢复完整块，无新建完成marker，无修改原3GB数据。这否定直接降频交付。
- CPU全排除组编译(cpu_all_filters_960)480步27.97秒，无速度收益；与未编译源位置最大差2.265µm、关节位最大57.34µrad，并非逐字节轨迹相同（compiled_filter_cpu_state_equivalence.json）。静态排除关系等价不代表浮点轨迹相同。
- 只绑定8个P核心(cpu_pcores_960)480步26.67秒，局部约5%改善、启动更慢；未改系统governor/驱动，暂未采用。
- adaptive_fourbar_960结束：1440次检查只重配3次、闭合误差6.13pm，但480步27.75秒，无显著整体收益。未采用。
- pin几何优化：新fuse_pin_collision_quarters.py证明128根四象限凸块并集为凸体(最大体积相对差5.73e-16、坐标差1.73e-18m)。单180顶点凸体局部19.33秒，但后续实际cooking读回仅40顶点、外形内缩31.12µm，拒绝；官方ConvexMeshCookingTask.cpp证实SDK无条件把vertexLimit钳到64，设置255不起作用。此方案未进入主线。
- 改为沿真实CAD直杆/圆头交界分块：一个40顶点直杆+四个50顶点鼻部；640块actual cooking全部读回、原外包络偏差约1e-19m，质量/材料/源文件不动。axial_pin_near_seat_960静态480步35.20秒(旧同窗口40.38秒)，键槽通过、深度14.604849mm；收益有限，未用于完整回合。所有分块改动仅显式诊断，main尚未接入。
- 找到更直接开销：near_seat_960第300帧18339个点中14223个零冲量；其中12754个是pin—原插座刚性孔壁，全部零冲量且间距>20µm。pin—128簧套有5126点/3936正冲量。原Pin与Socket各50µm contactOffset产生100µm预测接触范围。
- 新pin_contact_margin.py仅将原512个pin凸块和插座OfficialVisual/Geometry的contactOffset从50µm改为10µm，restOffset=0、原网格/材料/簧套margin不变；保持CPU960Hz。mid pin_margin10_960(p64)480步19.56秒vs27.92软件优化基线；仍非5倍。
- pin_margin10_near_seat_960加无损f32记录/1024步全代GC：480步23.54秒vs旧同窗口40.38秒，键槽通过、深度14.60464045mm。原物理限值不放宽；完整回合未验证。
- 原生接触扩展43：SDK原本float32，直接储存10个f32，读回原Pythonfloat并将as_array提升f64，避免后评计算精度变化。真实SDK向量18339点回放全部字段/顺序/事件精确一致，点载荷1467120→733560字节。相关13项unittest通过。两个本地so已重新构建；不在物理进程运行时重写so。新增native_contact_float32选项默认false，main与source-probe已接入，当前margin-near实际f32运行首两报告与Python参考校验已通过。
- GC可显式1024步一次gen2(范围128..8192)，逐帧gen0和最终gen2仍保留；默认128未改。main另可defer_gc_only_nut_phases=false，使全阶段使用同记录GC策略。
- first_turn_source.json提取原成功repeat02第211314步/第一轮nut_rotation和nut_regrasp；四个电机输入状态从原hand_mechanism_samples.jsonl.gz匹配取得。first_turn_margin10_p64已完成同源冷态90°加载运动：3721步、wall215.26秒、local118.90秒(含当前RGBD等)，stepper约81.68秒。实际Nut相对Body转89.153°、Body深度9.242309mm，原2µm键槽标准通过(最差1.171276µm)。这是局部加载段，不是完整装配。
- first_turn_margin10_p16结束：与p64同样3721步/90度，physics63.03秒vs63.02秒，没有收益；原键槽标准失败，最差13.473947µm，拒绝16迭代，不用于完整候选。
- 官方Carbonite原生CPU时间线已采集：native_zone_profile首圈局部28个有效帧中main物理13.81ms，SolveIslandTask平均0.411ms；PxsContext.contactManagerDiscreteUpdate为关键路径。native_zone_dense中密集状态main38.47ms，Solver10.52ms，窄相位各线程累计34.04ms。线程累计含重叠且profiling本身有开销，不作为提速验收。SDK GuContactMeshMesh.cpp证实双SDF接触执行双向投影；指端14192三角面/42576点，NutExternal235728面/117866点。
- 网格简化未采用：fast_simplification已有0.2.0，及官方meshoptimizer-v1.2(9d9890c73011d75920af614485296d1e03e95448，本地.deps构建)均只做离线候选。原坐标为m时trimesh小三角形距离谓词有绝对容差问题，后改为µm坐标计算并换回m，保留旧结果。meshopt目标3/20/50µm各生成13888/11226/8520面，但双向采样实际最大24.45/59.41/141.13µm，超对应预算，均未用于仿真。误差仅有限采样，不冒充全局Hausdorff证明。原STL/质量/视觉不改。
- first_turn_sdf8仅外部Nut的SDF由16bit→8bit：CPU960Hz64/4与源网格/分辨率不变，完成3721步、physics62.95秒，与p64无显著速度差。键槽通过(最差0.989µm)、深度9.252203mm。精度变化不是无损，但该候选没有收益，未采用。
- first_turn_original_convex：省去Nut专用指端SDF，改用原资产已有指端convexDecomposition。复用旧SDF已抓握电机状态，481步后CAPACITY_TESTED_GRIP_PRELOAD_NOT_REACHED，未开始旋转；不是完整失败/成功性能结论。下一步必须用同力目标的真实重新抓握建立接触，不能加力或放松预载检查。
- 凸分解后续：first_turn_convex_fresh_grip复用闭合起点，前置路径检查报Nut与f1Link3冲突，未开始重新抓握；改用原回合206658步真实张手tare状态(open_first_grip_source.json)后convex_open_grip_turn完成真实抓握+90度指令，共10311步/wall358.69秒。后评f2原PAD投影有327点超过原0.5mm限，最大0.605744mm；键侧越界未超过2µm，但该局部结束深度7.764918mm、非所有键全部越口，不能写完整插合。凸分解路线未用于完整候选。原阈值未改。
- full_cpu960_margin10_01结束且不合格：约34.5分钟仍在对键插入阶段，按已测分段预测超出53.03分钟目标，通过run/STOP_REQUEST在控制边界停止。原始档案/索引/影像已封存，motion执行2045.612s；程序总wall2200.169s、exit1，收尾在序列化包含PackedNativeFloat32ContactPoints的评价见证时TypeError。未完成装配、未宣称倍率。JSON摘要/最终评价写出已补充原数值默认编码器，兼容42/43，相关回归通过；原失败证据不改。
- 新kinematic_result_cache.py仅缓存不可变运行模型的有限tuple输入，键保留signed-zero和limit模式；给每次调用独立可写矩阵，模型合同对象变化失效。真实模型400个重复查询全连杆逐字节一致，0.1087→0.0314s。两个缓存行为测试通过。默认关闭，不是物理/控制参数变化。
- 新CPU Fabric显示发布延迟：保持逐步原生传感读回，仅World.render前发布显示数据；_install_rgbd_resume_sync现在幂等，world._kcg_defer_fabric_until_render显式选用，默认仍原逐步输出。主入口和源态诊断均已接入。
- first_turn_deferred_fabric结束：原CPU960Hz64/4+margin10、相同源态/输入/90度，加入精确FK缓存与显示边界发布。3721帧的native关节位置/速度/力矩、Body/Nut位置/四元数与first_turn_margin10_p64全部逐值零差，键槽完全相同(最差1.171276µm)。physics63.02→57.96s、audit14.05→12.21s、local118.90→111.21s；不是整回合5倍证明。
- 新retime_transport.py只作用于经验证的单调关节直线路径；保留起终点/直线、原0.8速度余量、每关节峰值速度不增加和每关节峰值加速度不增加(容许浮点差)。复用已有ScalarMotion，非直线路径不修改。原主搬运动作48.636→32.970模拟秒，峰值速度仍0.12rad/s，峰值加速度约4.787→0.12rad/s²；原后续2s保持不删。te_body_assembly_motion仅在显式computation选项对自由搬运与侧观察路径应用，生成后仍走完整当前手+插头碰撞检查；接触插入/旋拧/末尾3s保持不改。
- 当前唯一物理运行：artifacts/performance_5x/full_cpu960_software_fast_01/run，exec session55060；实际PID/启动UTC/完整argv在motion_process.json。新预检通过(exit0,172.97s)。配置assembly_cpu960_software_fast.yaml在CPU960Hz64/4 margin10原SDF/原CAD上启用cpu_fabric_output、defer_fabric_until_render、cache_forward_kinematics、retime_straight_free_transport。预算3600秒、240秒收尾，目标仍3181.9989秒。恢复后先核实实际进程/文件状态。安全停止用run/STOP_REQUEST。
- 源码c776f5e保留；0b1c72c已提交精确FK缓存/Fabric显示发布/受限直线路径重计时/JSON写出修复及新候选。本轮使用0b1c72c，代码冻结不在运行中改变候选参数。无远端推送。所有失败证据保留。
- 目标仍是新完整回合含记录收尾<=3181.998913795秒并通过原实际到位、2µm键槽、源接触、3s全松手零手冲量、同128簧套逐帧承载、Body/Nut不休眠、后备限位不承载及同回合影像验收。任何局部成功或缩短路径数学检查均不能替代完整验收。
- 所有运行simulation-only；没有硬件、系统驱动变更、外部发布或新子代理。始终只有一个主要物理进程。

## 环境

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`；规划 `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；视觉 `/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`；SAM根 `/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`。Isaac6.0.1.0 / PhysX110.1.13，原CPU960Hz/64位置+4速度迭代，关节接触最后求解。所有运行simulation-only/hardware_authorized=false。
