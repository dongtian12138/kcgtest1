# 当前任务：完成用户截图的四项装配工作

核验时间：2026-09-17 12:05 UTC。两次同初始场景完整视觉装配均已VERIFIED，全部原2µm键槽、实际到位、原止挡/簧套和完整3秒真实松手检查通过。两个主程序均已结束，没有物理实验运行。当前保存第二轮和小型审计/后处理优化，准备按原顺序启动1mm/1°初始位姿变化；尚未完成该变化案例。

## 有效位置、代码与当前动作

- 活动树 /home/noob/WorkPlace/kcgtest1-improvements-20260916，分支codex/connector-assembly-improvements-20260916。HEAD baf6d4f5c2c7035b4fd17ba38e0a19bd04ea7455已推送，当前有本任务待提交改动。原树/home/noob/WorkPlace/kcgtest1用户原脏资产保留，仅镜像本上下文；不变基线树保持原样。
- 第1次完整通过：artifacts/full_validation/contact_last_gc128_repeat01_restart01/run，实际执行25a0d6f68771080fe60a7dde948cdaa4e8d5927f。
- 第2次完整通过：artifacts/full_validation/contact_last_gc128_repeat02/run，实际执行baf6d4f。session79305/PID443628已结束，whole48801已生成VERIFIED（可收工具最终结果，不再等待物理）。所有辅助body/nut/key/turns/band/terminal/影像核验已完成。监控cells115/122/150均结束，不能再等旧cell。
- baf6d4f相对25a只修复记录收尾异常，控制/物理一致。第二轮结束后已应用两个新小优化：carts_v2/run_grasp_lift.py只在已结束且封存的相对运动统计中读取所需位姿字段；carts_v2/evaluate_run.py只把native有限数检查的临时数组换为流式读取已有数值。reproducibility/.../review_completed_assembly.py的whole分支用原without_cyclic_gc包装。没有改控制、planar_hold、力、速度、模型、物理设置或验收公式。
- 两个生产源文件实际AST与先前准备补丁精确一致，2项绑定已明确更新；768启动/pose-variation绑定检查通过。Isaac Python3.12环境中native contact与codec测试8项、4子项通过。pytest9.1.1安装在artifacts/test_dependencies/pytest_9_1_1，用临时PYTHONPATH运行，仿真环境包未改。最初规划venv缺msgpack、Isaac env缺pytest的尝试已纠正，不是代码失败。
- 当前未提交资料含docs/ASSEMBLY_REPEATABILITY_20260917_CN.md、第二轮与控制诊断/性能小证据、first_verified_fresh_checkout_check.json、second_recording_optimization.json、上述2源+后评入口+manifest。下一步检查diff/提交推送本任务路径，然后从相同物理控制配置启动pose-variation新预检和完整动作。

## 两轮完整验收

| 项目 | 首轮restart01 | 第二轮repeat02 |
| --- | --- | --- |
| 最终Body深度mm | 14.604580849409077 | 14.60440203547475 |
| 欠14.605mm目标µm | .419150591 | .597964525 |
| 全程最差键穿入µm | 1.848419870833，raw230008第三键 | 1.866570825613，raw229257第三键 |
| 原2µm | 通过 | 通过 |
| 完全张手保持 | 每帧零手冲量，2880帧/3s | 同左 |
| 原止挡正载 | 每帧8组 | 每帧9–11组 |
| 簧套 | 每帧同一套128个，与原止挡同时承载 | 同左 |
| Body/Nut休眠 | 每帧false/false | 同左 |
| 源指甲/指腹、视觉参与、实际图片 | 全部通过 | 全部通过 |
| whole | VERIFIED | VERIFIED |

- 每次真实3秒由独立terminal补核覆盖，不能用旧whole的末2秒/any判断代替。独立报告在各case父目录review。原生点/通道/单步回调标志完整，末段仅f1甲/f2腹/f3甲对Nut；允许区外与未知点0。后备轴向.5999mm未达到（首轮全程.516563319、第二轮.511467930mm）。
- 第二轮六段腕指令90/90/40.011693/13.417704/88.594837/37.726809°，总359.751043°，余.248957°未发；第3/5手指余量、第4弯矩、第6合力原正常换抓线，均无hardabort。实际Nu/Body角89.051104/89.210522/38.072012/13.216402/88.164310/35.644015°，不能把腕指令当Nu角，不能把差全叫纯滑移。最后真实坐底，故未发尽指令不等于失败。
- 第二轮295421帧/26530655597字节归档，索引匹配；terminal[291581,295421)，完全张手[292541,295421)。第一轮294560帧/25520469055字节。原始档完整保留。
- 两主程序退出2保留：legacy evaluation只评初始pickup，并有旧full-pad等未满足字段；不能宣称所有JSON都PASS。两轮engine_health/identity/truth_isolation均true，适用源面/机械/whole验收为通过。
- 首轮运动执行14572.931s=242.88min，总关闭前15717.009s=261.95min；第二轮14738.754s=245.65min，总15909.995s=265.17min。路径不同，不把差异当提速归因；计算仍慢。
- 首轮真实307s视频与21s末段已公开Release。第二轮完整视频run/video/assembly_four_view.mp4，22.8s连续末两段片段run/postrun_evidence/final_two_strokes_and_release_actual_clip.mp4；实际三个末段图已看，红带未露，1024/1024径向样点遮住。没有合成图像或场景摆拍。

## 后续顺序与边界

1. 保存本次源/证据/文档，提交推送到现改进分支。首轮/第二轮原执行版本和manifest保持不被回写。
2. 用现有入口：python3 scripts/run_current_hand_assembly.py --run --pose-variation --output-root artifacts/full_validation/contact_last_gc128_pose_x1mm_yaw1deg（目录应不存在）。配置已准备，初始平移[.001,0,0]m、yawπ/180，摩擦仍.45，所有变化在物理启动前。先新预检，固定21600s墙钟/1200s收尾，不运行中扩限；不要对bounded_experiment.py调用--help。
3. 完整运动结束且归档封存后，独立核对该新回合所有源面/视觉/关键2µm/原止挡/3s完全零手承载/同128簧套/awake/后备轴向，以及真实视频。通过/失败均按实际数据，不能套用前两轮。
4. 如果失败，先定位最早失效再做有可检验差别的诊断；不盲增力/增次数/扫参数，不借冷重建后自发沉降证明连续装配成功。若当前控制已经通过，先完成这个既定变化案例，不把尚未验证的控制候选混入重复统计。
5. 完成后更新简明四项交付与新视频/证据/GitHub复现状态。不能只启动长实验、交一个状态就把四项算完成。

## 已查清的腕误差与控制事实

- 旧完整14第4段：control260041对应前拍260040，虚拟中心.607844mm，实体Nu偏心.029618mm，初始手/Nu关系预测误差.530467mm；native与encoder虚拟点只差.000168mm。另有约.078807mm初始标定与native/FK合并项，不能与.000168混同。独立前拍FK重建精确，已撤回missing-wrench-origin-shift说法，不要恢复。
- 第二轮第三/四/六段最大虚拟误差分别.618502/.165387/.860838mm；对应实体Nu偏心.017307/.104507/.044767mm，初始手/Nu关系预测差.591691/.071770/.910480mm。大虚拟误差不等于实体偏心；第4段成分不同，不能全套同一比例。数据artifacts/control_review/repeat02_grasp_relation及随版evidence。
- 独立六帧补充分解完成：把中心固定于任一单指末端仍有明显残差，向量会反向抵消，不能按指节角或模长算纯滑移/贡献率。固定初始中心关系比真实无滑动接触更强，滚动、接触斑迁移、相对转动/柔顺/滑移尚未分离。
- 当前planar_hold=true在te_local_interface_following.py计算横向顺应后把vxy清零；sidewall_sequence缺省false。20µm/N与.1s不产生旋拧接口XY偏置，两轮第4段offset[:2]全0。横向力仍进入10ms全六维负载补偿与原50ms保护，不能说六维力没用。位置反馈20/s、总接口横向限速2mm/s仍生效。
- 独立用各自当前腕部视觉self.axes重建，未混淆worldXY与interfaceXY。两轮第4段共同前938行已发角相同；首轮14/938、新轮11/938截速。第二轮10.002948°时弯矩.789501Nm，而横向请求.654653mm/s未截速；首次截速13.067807°。停止决定260260弯矩1.008883601Nm越原1.0正常线。速度分配残差3.82e-7/2.58e-7m/s不是实际执行误差。不能据末尾截速断言提限解决。
- planar_hold是两轮共有策略，既往抓持/接触历史不同；以上非单变量因果实验。有限XY让位是可检验方向，尚未做开关对照；当前不改它。完整资料evidence/planar_hold_review，Hubble独立只读任务均已完成，可继续用于新回合末段只读审查。

## 性能改动及证据范围

- 正式已有：路径角色缓存、独立tuple接触向量、Nut阶段帧尾gen0/每128全局步gen2及尾回收；控制advance原方法体未改。实际密集481步1421146150字节相同，72.585761→60.442991s=16.73%局部，不外推整场。
- 新已应用：相对运动收尾只投影封存位置/姿态；首轮294560帧重算每字段精确相等，1024帧19.763→8.850s；不得把原947s混合汇总全当此函数。普通list/未封存/无marker/封存marker四合成调用与原值一致。
- 新已应用：native有限数流式检查。17901点一帧30次平衡次序4.939→3.524ms，只是检查局部。四个真实阶段完整contact_counts及编码字节与原档一致。SDK110.1.13+不变decoder保证每点Float3/Float3/Float3/float，自有3+3+3+1值；NaN/±Inf等拒绝不变。不对任意畸形Python字典承诺通用等价，报告保留混合维度/字符串/None反例。
- sensor历史tuple仅离线候选：32768早段/晚段记录扫描有约75%下降，但转换13µs/行，摄入反而增加，历史仍在线用于力/目标/去皮/恢复。当前没有加入生产，不用这项宣称整场提速。
- 证据artifacts/performance_review及reproducibility/improvements_20260916/evidence/postrun_and_history_performance，改动记录second_recording_optimization.json。现生产源修改只有审计/后处理，控制/物理配方不变。

## 已交付、环境与历史

- 当前原物理/控制配置id load10ms_contact_last_gc128_nut_cpu960_64_4_six_grips：10ms负载前馈、原50ms导纳/保护、CPU960Hz、64/4、关节接触后求解，最多6抓/总360°原预算，原几何质量材料和力速保持。位姿变化用同配方，suffix_x1mm_yaw1deg。
- 首轮通过Release：https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-first-verified-20260917 ，tag baf6d4f；视频/片段/证据包/校验单四附件SHA与GitHub一致。干净新目录公开clone与资产下载、SAM固定源码补丁/四权重校验、ROS构建、nominal与variation启动检查通过；471项Git文件+297资产=768检查。未另跑物理、未验证新机器零安装。
- 原不变基线/home/noob/WorkPlace/kcgtest1-baseline-20260916，codex/connector-assembly-baseline-20260916，d64fa4a0a1a394bc800f96191d23aecc3bdad45c；Release assembly-baseline-20260916含121MB运行资产、旧视频。原完整14机械到位，但键2.147659µm不通过，保留原失败。
- 更早load10ms_repeat01执行c95785ca：最终14.582080096mm、欠22.919904µm，StopBox无正载，键2.475998µm；失败保留。冷重抓诊断在手未碰前自行从14.582到14.605，不能证明补转成功；后续已停。
- contact_last_gc128_repeat01初次在2026-09-17 02:41:52主机重启时仅initial_rgbd_settle，未完整抓取/封存；按中断留存、返回码null，不能计成功或物理失败。
- 环境：ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env；KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python；KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python；KCG_SAM6D_ROOT=/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D。运行始终simulation-only/hardware_authorized=false。
- 后评reproducibility/improvements_20260916/review_completed_assembly.py支持body/nut/key/turns/band/whole/terminal；terminal明确输出run外。extract_final_episode_frames.py提取真实帧，亲自查看后才能写visibility接受。阶段未结束前只读机器人/录像元数据，不读对象真值。
- 历史已归档docs/history/CURRENT_CONTEXT_CN_20260917_before_pose_variation.md，正文报告docs/ASSEMBLY_REPEATABILITY_20260917_CN.md。全部相关旧原始档保留。
