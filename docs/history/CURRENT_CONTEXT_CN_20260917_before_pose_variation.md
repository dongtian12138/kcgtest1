# 当前任务：完成用户截图的四项装配工作

核验时间：2026-09-17 11:55 UTC。第二轮完整运动已结束，295421帧/26530655597字节归档封存。Body最终14.604402035mm，原2µm键槽最差1.866570826µm通过；body/nut源面、实际图像、独立完整3秒松手/StopBox+同128簧套承载均通过。whole汇总session48801与主生产postprocessing session79305/PID443628尚未结束，当前尚不计第二次最终VERIFIED。

## 当前有效结果与执行位置

- 活动树 `/home/noob/WorkPlace/kcgtest1-improvements-20260916`，分支 `codex/connector-assembly-improvements-20260916`，最新baf6d4f5c2c7035b4fd17ba38e0a19bd04ea7455已推送，含首轮证据和收尾修正。首轮实际执行25a0d6f，第二轮执行baf6d4f，只有记录收尾异常处理不同，控制/物理不改。原目录脏资产保留；封存基线树不改。
- 当前完整回合 `artifacts/full_validation/contact_last_gc128_repeat01_restart01/run`：294560帧、25520469055字节归档，索引封存一致；motion_timing已生成。主程序session17980/PID21229已退出2；通用evaluation仅前35649帧pickup且旧全指腹门不适用，保留原报告。原引擎健康、身份和真值隔离均true。
- 实际最终Body深度14.604580849409077mm，欠目标0.419150591µm。独立完整3秒、2880帧每帧手冲量精确0，8组原StopBox正载，与同一套128簧套同时承载；Body/Nut均不休眠，深度保存精度下恒定。独立报告父目录review/independent_terminal_release.json与中文说明。
- `source_key_containment_review.json` accepted=true：五键最差间隙约−.910355/−.651432/−1.848420/+101.983432/−.859048µm，第三键raw230008最差，原2µm未改，余量约.151580µm。原源面Body/Nut检查均通过；原指甲抓起保持最低离桌56.109771mm，2秒桌面0、三甲100%。螺母真实区域为第一/第三指甲、第二指腹。
- 六段实际Nu角89.047246/89.549457/31.160159/70.687890/73.206586/.037954°。对应腕指令90/90/33.030902/71.152244/75.068092/.748762°，总约360。第三/第五f1传动余量正常换抓，第四腕弯矩正常换抓，无硬abort。第五已经到位，不能说第六补转导致到位。
- 实际三张末段图片已亲自看过、红带未露；1024原源面径向样点都被螺母遮挡。final_mating_visibility_review=true。实际视频run/video/assembly_four_view.mp4；连续21秒末段片段postrun_evidence/final_two_strokes_and_release_actual_clip.mp4已完成。
- 整体review session23420已退出0，whole_assembly_review.json=VERIFIED，所有mechanics/visual/additional项true；最大thrust.516563319mm<.5999mm。独立最差键三帧复算与主报告完全相同，源几何/原2µm绑定未变，独立报告review/independent_key_report_review_CN.md。
- 当前说明docs/ASSEMBLY_COMPLETE_VALIDATION_20260917_CN.md已更新为第一次完整通过，相关小证据已复制至reproducibility/improvements_20260916/evidence/contact_last_gc128_repeat01，保存原25a执行manifest。已随baf6d4f提交交付；仅修正后续收尾异常，不改控制/物理。

## 立即继续

1. 第一次完整验收已结束通过，不再等待旧session；新Release https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-first-verified-20260917 已发布307秒完整视频、21秒末段、1.76MB证据包，GitHub四资产SHA和本地均匹配。
2. 异常收尾已修复，12项针对性测试通过，768绑定检查通过，随baf6d4f推送。
3. 当前唯一主物理run是 artifacts/full_validation/contact_last_gc128_repeat02，session79305，物理PID443628，新预检退出0/四门true，前两段腕指令均90度完整，虚拟中心最大XY误差.080070/.219381mm，仅机器人信号，尚无对象后验；第三段40.011693度正常ELASTIC_RESERVE_EARLY_REGRASP，虚拟中心最大.618502mm；后验见证控制步246007、前拍246006，全运动结束后再核对真实Nut/抓持关系，记录third_control_interval_only.json。正在准备第四抓。轻量只读监控functions.exec cell115/122/150均已结束，不再等；motion已结束，主进程仅做postprocessing。保持生产源码冻结，只读机器人/录像元数据监控；未结束不得读其对象真值。一轮失败则先定位最早原因，不盲重跑。
4. 第二轮运动结束/封存后，按原标准完成body/nut/key/whole/完整3秒/同回合图像检查，并补核第三段control246007对应前拍246006的.618502mm虚拟偏差与真实Nu/抓持关系。主生产进程完全退出且第二轮验收通过后，可应用已准备的纯后处理字段读取和Native有限数流式检查，更新两源绑定、相关验证、提交推送；不改变控制、物理配方与数据。历史sensor tuple仍只是候选，当前不直接集成。两次名义完整通过后运行 `--pose-variation`，初始横移1mm、偏转1°，摩擦仍.45，尚未执行。
5. 完成重复和扰动后整理新视频/evidence/可复现提交及GitHub交付，不能只汇报状态就把四项当完成。

## 当前冻结配方与已准备修正

- id `load10ms_contact_last_gc128_nut_cpu960_64_4_six_grips`；配置reproducibility/improvements_20260916/config/visual_contact_last_gc128_task.yaml；BASE assembly_command指向对应候选。768绑定，执行manifest SHA e7cb9ca0901ee5ca6ca9d75290ebed41b7662ac03a8121b9f65d11255988e416。
- 10ms负载前馈、原50ms观测/导纳/停止、CPU960Hz、64/4、关节接触最后求解；最多6抓但总360度。源模型材料与力/速度/2µm不改。
- 记录GC仅Nut阶段帧末gen0、每128全局步gen2，尾段回收，原advance方法体不变。相关14测试通过。实际481密集步完整1421146150字节相同，72.585761→60.442991s（16.73%局部），不能外推整场。
- 本完整motion execution14572.931180s≈242.88min；sim306.833333s。stepper physics8173.564761s（含内部回调）、audit2843.337378s、GC2270.717461s、command395.511487s；不要把嵌套回调重复加总。GC最长3.660647s，尾2.893803s，无错误，完整计算速度仍慢。
- 收尾修正已于首轮主程序退出后应用：run_body_assembly_with_video.py独立尝试raw/GC审计/video收尾并保留首异常。test_recording_finalizer.py六项、原GC六项共12项通过；只有wrapper生产源变化，控制/物理源未改。recording_finalizer_repair.json记录前后hash；768绑定检查通过。

## 已定位的腕偏差和未解决范围

- artifacts/control_review/grasp_relation_decomposition：旧第四段最大腕推算误差.607844mm，实体Nut仅.029618mm；真实手/Nut关系相对最初刚性假设变化.530467mm；原生手姿态与编码器虚拟点只差.000168mm。不能将.53mm叫纯滑移，仍混有指节、柔顺、滚动；不能把虚拟误差说成实体偏心或纯机械臂错误。在线控制未用离线对象真值。
- 同源局部50→10ms负载补偿曾把第四段最大虚拟误差.579183降至.099038mm并完成90度，但不证明每种抓姿均改善。
- 接触顺序局部default/contact_last只差bool，最差键1.714/1.460µm，两者局部都通过；colddefault未重现全流2.476µm。不能孤立归因全部全场成功于此一项。

## 保留的失败与已撤回解释

- load10ms_repeat01执行c95785ca，277530帧：最后14.582080096mm，欠22.919904µm，完整松手3秒但原StopBox从未正载；全流第三键2.475998µm、第五2.141567µm，whole REVIEW_REQUIRED。保留docs/ASSEMBLY_FIRST_COMPLETE_REVIEW_20260917_CN.md和随版evidence。
- cold final_regrasp_remaining360：手未接触时Body已自行从14.582259到14.605773mm，故不能证明末5.25度/多换抓修复短缺；后续final_held主动停。证据evidence/cold_remainder_diagnostic。
- contact_last_gc128_repeat01在2026-09-17 02:41:52主机重启时仅到initial_rgbd_settle，没有抓取，未封存；标INTERRUPTED_AT_HOST_RESTART_NO_COMPLETE_ASSEMBLY_RESULT，返回码null。当前restart01不是丢弃失败重编成功。驱动未改。
- 原第三键2.147659µm不是浮点重算或SDF分辨率解释：longdouble同值，保存Body量化界约.014934µm；键凸包/静态槽三角，源平面匹配。原生接触与前一步步后姿态对齐。已撤回missing-wrench-origin-shift说法，不能恢复。

## 基线交付、命令和环境

- 不变基线 `/home/noob/WorkPlace/kcgtest1-baseline-20260916`，codex/connector-assembly-baseline-20260916，d64fa4a0a1a394bc800f96191d23aecc3bdad45c。
- Release https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-baseline-20260916 ，297运行资产/121MB和原视频已公开下载并逐文件验证；不是全新机器零安装保证。原baseline14到14.604491442mm但键2.147659µm未过，不能称全验收通过。
- 环境 ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env；KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python；KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python；KCG_SAM6D_ROOT=/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D。
- 入口python3 scripts/run_current_hand_assembly.py --run --output-root 新目录；--pose-variation只在名义通过后。--check不启动物理，--gui打开窗口。完整固定21600秒/收尾1200，不能运行时扩限。不要对bounded_experiment.py调用--help（会被当执行指令）。
- 后评reproducibility/improvements_20260916/review_completed_assembly.py，body/nut/key/turns/band/whole/terminal；terminal输出run外。必须motion结束且归档封存，不能读活体真值。whole以without_cyclic_gc包装运行较快。完整3秒每帧补核仍不可省略。
- 只读监控 artifacts/full_validation/read_active_progress.py 检查控制/录像/实际PID，不读liveGT。历史含docs/history/CURRENT_CONTEXT_CN_20260917_before_first_complete_postreview.md。

## 已完成的额外离线工作（生产尚未采用）

- relative_summary_projection（session51946已退出0）：首轮294560帧完整字段投影449.242秒，与原相对运动统计每个字段精确相同；同末1024帧原19.763秒、投影8.850秒。源码计算公式保持，生产未改。prepared_reader_change.json准备了_motion结束后_读取修正，主程序完全退出后再检查/应用。不能把原947秒混合汇总计时全部当成此函数时间。准备补丁的纯AST/合成存储4种调用与原结果一致（list/未关闭/无marker/封存marker），prepared_patch_probe.json；生产尚未应用。
- immutable_sensor_history（85168已退出0）：首轮step0..32767的32768条机器人记录，固定历史堆gen2稳态中位数.112760→.026396s，逐行msgpack字段相同；转换额外约13µs/行，摄入总时间上升，不是整段记录净提速。
- immutable_sensor_history_late（71085已退出0）：首轮step249000..281767，含晚段Nut旋拧/换抓/卸载32768条，同样字节精确一致；稳态gen2 .107628→.027487s，转换.419676s。仍仅离线内存扫描，未进行控制器/物理验证。
- 独立消费者审查完成：历史仍在线用于最新腕力、目标保持、去皮、夹持均值和预接触恢复；未发现依赖嵌套list可变性的消费者。四条去皮/lift/Nut/unload代表记录的现有JSON、msgpack、NumPy消费者兼容。不得称“纯离线日志”而直接改在线数据表示。当前生产baf6d4f保持冻结。
- 独立旧第四段分解完成：控制260041对应物理260040、前拍FK重建虚拟点精确一致；.607844/.029618/.530467/.000168mm同一时刻。约.078807mm初始标定和native/FK合并项仍存在，不等于.000168mm；不能把模长相加或称纯滑移、全部伺服错误或伺服完全正常。原始7帧只读复核，不读repeat02。
- 上述性能小证据已复制到reproducibility/improvements_20260916/evidence/postrun_and_history_performance；分解在evidence/grasp_relation_decomposition。文档与新证据未提交；src/及运行manifest相对baf6d4f没有修改。
- GitHub最新交付新目录检查完成：artifacts/reproduction_check/assembly-first-verified-20260917，公开tag克隆baf6d4f、资产重新公开下载297文件、SAM固定源/补丁/4权重已校验（权重复用本机cache）、ROS构建退出0、默认和pose-variation --check退出0。没有另外启动物理，没有验证新机器安装。报告first_verified_fresh_checkout_check.json，471项Git文件+297打包资产=768检查。

- finite_contact_validation离线检查完成：首轮末帧17901点、30次平衡顺序，此有限数检查4.939→3.524ms；4个封存帧整个contact_counts与原归档msgpack完全相同。prepared_streaming_guard_change.json尚未应用。独立SDK110.1.13实际契约/探针确认位置/法向/冲量为固定Float3，decoder复制3+3+3+1floats；只在该native契约内等价。任意混合9/10长度、数字字符串、None的Python注入字典与原np.asarray并非完全等价，报告保留反例；不能宣称通用格式校验等价。小优化不是整场28.6%提速。相关文件在artifacts/performance_review/finite_contact_validation及随版性能evidence中。

## 第二轮新增的控制核对（仍不读活体对象真值）

- 第4段仅13.417704°触原WRIST_BENDING_RESERVE_EARLY_REGRASP；第1轮同段71.152244°。第5段随后88.594837°正常ELASTIC_RESERVE_EARLY_REGRASP，总腕指令322.024234°，剩余37.975766°、还有最后第6抓；不能预告成功或增加预算。four_control_intervals_only.json保存机器人信号。
- fourth_control_wrench_comparison.json：两轮第4段初始弯矩分别.238/.203Nm，13.4°附近.815/.987Nm；横向合力78.07/84.96N。两轮既往插入/接触/抓握历史不同，非单变量因果对照。不是起始即接近1Nm，而是在旋拧中分叉。
- 明确代码事实：te_local_interface_following.py先计算XY顺应，随后planar_hold=true无条件将vxy清零。当前两轮实际coaxial_controller_report.grip_recipe确为true，sidewall_sequence缺省false；20µm/N和.1s不产生旋拧接口XY参考偏置，位置增益20/s和总接口XY2mm/s限制仍生效。横向力仍进入10ms全六维负载补偿和50ms保护，不能说六维力没用。
- 独立代理已完成artifacts/control_review/planar_hold_20260917核对，并复制随版evidence/planar_hold_review。用各自实际腕部视觉self.axes重建，offset[:2]全0；共同前938行已发角序列完全相同，首轮14/938、新轮11/938截速，新轮首次13.067807°，在10.002948°弯矩.789501Nm但接口XY仅.654653mm/s未限速。速度分配残差3.82e-7/2.58e-7m/s不是实物执行误差。新轮实际停止决定260260弯矩1.008883601Nm越原1.0正常线，无hard abort。两轮共有planar_hold，不是单变量因果实验。
- 这提出有界XY让位的一项候选原因，尚未单变量验证。当前物理继续baf6d4f固定版本，不临时切planar_hold。先完成第2轮真正验收；若失败，查最早实际失效再决定针对性局部/完整对照，注意冷重建不保留完整接触/簧套历史。若通过，优先按原顺序完成小位姿变化，不把未验证控制候选混入重复统计。

## 第二轮封存后的最新核验

- 六段腕指令90/90/40.011693/13.417704/88.594837/37.726809°，合359.751043°，余.248957°因第6段原腕合力正常换抓线停止；无hardabort。实际Nu/Body角分别89.051104/89.210522/38.072012/13.216402/88.164310/35.644015°。Body段末9.257418/11.012924/11.707110/12.007279/13.939453/14.604372mm，最终松手14.604402035mm，欠.597964525µm。
- 独立terminal：3840帧卸载张手+保持，最后2880帧3秒每帧hand原冲量精确0，原StopBox9–11组，同一套128簧套持续正载；Body/Nut均不睡眠、深度保存精度下恒定，末段thrust最大.511408260mm。父目录review/independent_terminal_release.json和中文说明已完成。
- SourceKey最差第三键raw229257为−1.8665708256µm；五键全部原2µm通过。body/nut已accepted=true。真实三个末段帧已看，红带未露，1024/1024源面径向样点遮住，final_mating_visibility_review=true。whole尚在运行，需读实际结果；不能只靠分项预告最终通过。
- 11:39启动reviews：body95389/nut19829/key24992/turns80708/band81394/frame42450均已退出0；whole48801仍处理。真实末两段连续录像提取session74385待收输出，文件postrun_evidence/final_two_strokes_and_release_actual_clip.mp4。
- 主程序79305/PID443628仍postprocess，src和绑定manifest自baf6d4f未变，暂不应用性能补丁。execution_through_transport14738.754247s（245.65min），sim307.730208s，GC2300.488453s；最终总计与legacy exit待主程序真正结束。
- 新后验artifacts/control_review/repeat02_grasp_relation/comparison.json：第三max虚拟.618502、实体Nu偏心.017307、固定手/Nu关系差.591691mm；第四.165387/.104507/.071770mm；第六.860838/.044767/.910480mm，模长不可相加。第三/sixth显示大虚拟误差不等于实体偏心；第四不能强行套同一种成分比例。root依原验证公式生成，仅已封存样本。独立代理正在以相同起点/最大点补充分解指末端运动，输出同目录independent_finger_motion_*，不作为等待whole的前置。
