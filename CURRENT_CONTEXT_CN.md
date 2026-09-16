# 当前任务：四项交付仍在执行，不能只交付第一项就结束

核验时间：2026-09-16T20:05:51.950708+00:00。用户要求完成截图全部四项：保存成功版/可复现交付、计算记录提速、第四段腕跟踪根因、原2µm键槽问题及少量完整重复。simulation-only，hardware_authorized=false。原2µm检查不放宽；一次只运行一个物理实验。

## 当前实际运行与下一步

- 当前唯一物理运行：改进工作树 artifacts/local_tracking/fourth_matched10ms，主exec session91231，PID1269292包装器；2026-09-16T20:06启动，最多7000物理拍，1500秒总预算/180秒收尾。恢复时先核对实际结束，不凭本句等待。
- 这是原完整14第四段前step258721冷初始化（不恢复原生接触warm-start，不是完整回合复现），保持4电机实际输入状态、当前手/连接器、64/4求解960Hz、全部原安全条件。控制设置直接取原第四段保存settings，再放入诊断入口实际使用的nut_rotation_after_index位置。运行前A/B整份配置比对已通过；当前10ms的rotation/coaxial_recipe.json和progress.json已实测横向.002m/s、轴向.003m/s、补偿.01s。
- 正确10ms候选已启动（见上一条）：artifacts/local_tracking/fourth_matched10ms_command.json、fourth_matched10ms_probe.json、fourth_matched10ms_task.yaml；只改负载前馈滤波50->10ms；原50ms力控制/保护信号、抓力、速度、所有硬限不变。候选设置也预校验了.002/.003/.01。启动前保留源码快照；不要与50ms并行。
- te_local_interface_following.py已加入可选contact_load_compensation_filter_time_constant_s与独立filtered_contact_load；缺省仍使用原filtered，原50ms算法和保护不变；新_update_contact_filters给adopt_existing_grip和update共用。匹配50ms已经退出0（旧session72515关闭），1824物理拍，命令30.80258°/实际Nut29.793825°/本体前进.557780mm，最大虚拟中心XY.579183mm，无硬abort、无4电机边界超出；源面和键槽全流未在该局部报告中认证。local_turn_physical_review.json已出，reproducibility/improvements_20260916/review_local_turn.py用于结束后的对应复核。
- 匹配控制A/B完成后，整合已验证的接触向量tuple优化，再做必要短物理等价/性能对照。不要在进行中的对照里改生产源。
- 键槽局部64迭代基线已准备（未启动）：artifacts/key_review/second_source_solver64_command.json，source step226737、原第二段实际settings、原4电机输入、最多4500拍；计划仅在64基线出现相关受载接触后比较位置128/速度4，保留几何/材料/控制/960Hz/2µm。当前不能并行启动。
- 随后仍须选择有局部证据的控制/效率/求解方案，少量同场景完整装配重复，再有限位姿扰动；这些完整重复尚未开始。

## 已完成的保存交付（第1项）

- 成功版分支codex/connector-assembly-baseline-20260916，提交d64fa4a0a1a394bc800f96191d23aecc3bdad45c已推送GitHub。Release https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-baseline-20260916 已公开。
- Release资产：assembly-runtime-assets.tar.gz 121458865B，SHA256195e378646663803872f527c17afb1ee3541ce4af95cdb533e629112832d9e70；原完整视频42428584B/291.4秒；原末段片段1491344B/9.6秒。资产包297文件，新改进工作树已从GitHub下载、解压逐文件校验通过。
- 保存版新目录完成配置/几何加载、156个组合机器人prim属性与关系等价、Tesseract资源展开、新Isaac预检。原两次路径预检失败也保留。没有把预检当完整复跑；跨机器新环境安装尚未验证。
- 原执行源码在reproducibility/assembly_20260916/execution_sources；原报告在evidence；运行/准备脚本scripts/run_current_hand_assembly.py、prepare_assembly_reproduction.py；说明docs/REPRODUCE_CURRENT_HAND_ASSEMBLY_CN.md。SAM固定提交1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2，两个推理补丁、四权重SHA、三环境清单已发布。保存版工作树保持封存。

## 当前提速证据（第2项）

- 改进源码carts_v2/evaluate_run.py已采用接触路径归属缓存：只缓存路径是否属robot/object/table/fixture/末节，不缓存接触数值。6实际阶段帧完整contacts MessagePack与原函数/原归档逐字节一致，NaN/Inf等故障保留；密集contact_counts函数约节约31–36%，不能说整场节约三成。根源函数占原总耗时11.3%。
- 独立审查完成tuple候选：三维position/normal/impulse从list保存为tuple（序列化仍完全一样），默认GC保持开启。每类3新进程、各120帧，总回放10.988->9.450s（13.99%），GC4.895->3.210s，完整411513840B消息流一致，源Float3改写后仍持有原值；额外转换.0389微秒/点。尚未整合生产。需要改test/test_nut_runtime_efficiency.py:32的list类型断言并保留内容/持有性验证，生产未发现list类型门或原地修改依赖。
- 实际归档64帧/1缓存，不是类默认512/2。缩16/无写缓存离线仅节约7.4%，未采用。
- 机器人/六维力self.samples确实全程保留；但主代理2万条历史+120密集帧对照，落盘没有提速（list10.403/10.517s，disk10.644/10.662s），不扩大生产改动。原型和负结果在reproducibility/improvements_20260916/benchmark_robot_history_storage.py与artifacts/performance_review/robot_history。
- 独立代理/root/independent_final_guard_audit只读任务均完成，报告都在原工作区artifacts/full_rotation_20260914/independent_final_guard_review_20260916；可复用已授权的独立只读审查，不另开代理。

## 控制排查事实与本轮纠错（第3项）

- 原完整14第四段1320控制拍由实际编码器重建全部吻合；速度求解最大平移残差仅.000366mm/s，无arm饱和。约26°指令向内2mm/s，实际前后位姿差分却向外近10mm/s；限速不是初始失跟的充分解释。
- 已提取原258600..260060机器人行，1461行6MB：artifacts/control_review/fourth_stroke_robot_window.json；重建脚本reproducibility/improvements_20260916/review_fourth_robot_tracking.py；结果fourth_decomposition。记录了原生速度和实际位姿差分；PhysX约束下二者不能当严格相等。
- 负载补偿使用50ms滤波；反推滞后的力矩量级可对应数百微米，但只是候选解释，未证明因果。手指输出角后段也变化（约-.15/+0.04/+.12°），原手腕固定抓取关系不能直接当真实Nu中心；3接触点事后刚性拟合末残差.282mm，不足证明纯弹性或纯滑动。
- 第一次fourth_loaded_baseline在481拍转动前失败：诊断runtime.robot_asset引用旧metadata路径，而物理选了新目录同模型；已改te_source_stage_probe.py使用实际args.robot_asset。原失败保留。
- fourth_loaded_baseline02退出0，1822拍、30.7045°正常换抓、最大XY.575557mm、无硬abort。但随后查明旧诊断入口默认选前两段配方，实际XY1mm/s/Z1.5mm/s；不是原第四段同参数对照，已向用户纠正。
- fourth_loaded_10ms同样误选前段，10ms未生效；在发现后写STOP_REQUEST结束，退出2，保留配置路由失败证据，不能当10ms比较结论。
- 当前fourth_matched50ms/待运行matched10ms已通过实际配置核对解决该问题，无需再改诊断入口来猜测阶段。此前source_stage_probe默认use_current_rotation_config选assembly[nut_rotation_after_index]的源码事实已确认。

## 键槽直接证据（第4项）

- 原最差raw228525，第三键−2.147659492µm，原2µm带超.147659492µm。
- 高精度(longdouble)重算−2.147659492351258µm，与原仅差5.92e-13µm；本体保存位置float32/归一化float32四元数的量化误差保守界约.014934µm，角点范围−2.160180..−2.135139µm。该界只涵盖保存本体位姿，不是PhysX全部内部浮点误差。
- 同时原生第三键接触6点，最小separation−2.011984634µm；key凸包与Socket静态三角网格（approximation none），两者restOffset0，不是这对接触的SDF网格问题。
- descriptor两槽壁平面与当前冻结Socket155742三角面对应，最大顶点差<.000576µm，排除明显旧描述几何错配。
- 数据/脚本：artifacts/key_review/key_precision_review.json、worst_key_contact_witness.json、key_plane_vs_current_socket_mesh.json；reproducibility/improvements_20260916/review_key_precision.py。
- 已把原key_backlash_dimensions.json逐字节复制到reproducibility/assembly_20260916，evaluate_source_key_containment.py加入原artifact不存在时的随版只读回退；数值/原阈值没有改变。

## 原成功回合与环境

- 原工作区/home/noob/WorkPlace/kcgtest1，codex/visual-assembly-v1，原脏文件保留。保存版/home/noob/WorkPlace/kcgtest1-baseline-20260916。当前/home/noob/WorkPlace/kcgtest1-improvements-20260916，分支codex/connector-assembly-improvements-20260916，从d64fa4a建立；本轮改进尚未提交推送。
- 原完整14：/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14，279682 raw/19.6GB已封存。单回合到14.604491442mm、最后2880拍3s真实松手，手冲量逐帧0，原止挡与128簧套持续承载；原数值总检查仍REVIEW_REQUIRED。
- 原各段真实Nut角89.195/90.804/67.858/28.732/76.984°，五段仿真共12.880s，全动作291.335s，墙钟15419.988s。contact_callback5155.943包含于native_physics9289.904，contact_counts1743.741，archive_append884.616。
- 三环境：ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env；KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python；KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python；当前局部KCG_SAM6D_ROOT=/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D。
- 当前portable_source_manifest仍绑定保存版；改进源有修改，因此正式使用check/run前应在确定版本后更新绑定，不绕过它。必要资产已在改进目录，ROS iiwa_description已单独构建。
- GitHub CLI路径保存在原工作区artifacts/reproducible_baseline_20260916/github_cli_path.txt；凭据仅内存使用不打印。Release发布回执published_release.json。不得强推/重写保存tag，不修改或删除原失败证据。

补充：project_truth_fields.py是新离线读已封存归档所需字段的原型，未接入生产。两个64帧块与完整解码所选字段MessagePack完全相同；密集块.8506->.4582秒，稀疏块.0232->.0154秒，不能当整场提速。原始全部接触数据仍保留。报告artifacts/performance_review/projected_archive_read.json。
