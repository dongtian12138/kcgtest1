# 当前任务结果：同回合机械装配与真实松手已实现，原数值检查未全通过

核验时间：2026-09-16T18:03:46.964077+00:00。用户首要目标是当前连接器配当前三指手先装配成功，并已允许独立只读复核。本次完整14已在同一回合实现机械到位与3秒真实松手保持；原五键2µm数值比较带有超带，完整原检查没有全部通过。不得把两类结果混为一谈，不改原failed，不称所有原指标通过。simulation-only，hardware_authorized=false。

本轮用户询问进度、完整视频、自行运行和GitHub复现。已核对完整视频291.4s/1457帧/5fps、初始视觉阶段也在同回合内；Git远端没有本地codex/visual-assembly-v1分支，HEAD409d5c2仍有大量未提交源码，关键artifacts配置/模型被忽略；本轮没有提交、建分支或推送。新增scripts/run_current_hand_assembly.py与docs/REPRODUCE_CURRENT_HAND_ASSEMBLY_CN.md，面向原本机目录，默认仅检查；--run先新预检再有界完整运行，--gui同时传给两阶段，自动新目录，保留所有原参数。54个已记录绑定文件检查一致；命令与原始记录逐项比对、拒绝覆盖和预检失败阻断/退出码保留的假执行检查通过，未启动新的物理实验；GUI完整复跑没有重复验证，且入口不是跨机器GitHub交付包。不能说现已上传或能任意机器一键复现。

当前没有物理运行或后评进程。完整14 session60711/monitor2008已结束；包装器2026-09-16T17:09:50.296847 UTC退出2。279682 raw样本封存，归档19618739915字节与索引末偏移相等；execution_source_snapshots共30文件（含附带记录）。post endpoints16006、body37806、source76766、transient93320、whole57085均结束0，band与视频提取也结束。独立代理本次收尾复核已完成。不要继续等待旧session/cell，不自动再开整机仿真或扩大模型/参数来追逐内部比较带。

唯一结果目录：artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14。新机械完整阶段基线是本回合的源码/配方快照与真实记录；不证明重复可靠性或硬件成功。配置visual_common_all_reserves_task.yaml，前段two_nail_all_reserves_recipe.json、后段wrist_all_reserves_no_angle_gate_recipe.json。所有段elastic0.5Nm/min0、合力90N正常/110N硬、弯矩1Nm/min0/1.2Nm硬；4Nm输出传动、8.8Nm输入、掌1Nm、臂100Nm、腕扭4.6Nm、累计360°/最多3附加段保持。2Nm且累计>=350°仅为到位候选，本次最终按正常弹性余量结束并执行原释放。未用真值在线控制。

已验证的实际结果：最终Body14.604491442mm，最终倾角0.001638632°；raw276802..279681共2880拍=3s，手部raw冲量范数和每帧精确0，Body深度范围14.604461640..14.604521245mm，跨度0.059604645µm（不是逐帧恒定）。每帧原StopBox6—10组正接触、同一套128簧套同时承载，Body/Nut未休眠；释放3840帧报告点保留/通道一致/一拍一回调。全回合最大绝对Thrust0.515937989mm<0.5999mm，末窗无人工扇区。源指甲抓Body保持2s、最小抬升56.054339mm、无桌面承载、3指甲每帧有接触。入槽后Body实际与手分离通过。抓持/旋拧原两NAIL加一PAD检查通过；最终卸载/张手源面另由独立补核通过。

五次腕指令/真实Nu转角/本体末深度：90/89.195361/9.325636；90/90.803772/11.075271；66.859245/67.857650/12.444777；29.679989/28.731501/12.983484；78.747889/76.983788/14.604402（角度deg、深度mm）。总已发355.287121882°。第3/4/5段分别f2/f1/f1余量约0.5Nm正常停，无硬abort；五抓图像校正都按原条件完成。各次之间有卸载回退，不能说未坐底阶段绝对不动。

原标准未通过：source_key_containment_review.json第三键min−2.147659492µm（raw228525、第二旋拧），超过2µm比较带0.147659492µm；其他key min为−0.668403/−0.901160/+104.336300/−1.018564µm。最差点附近±480共961帧中4帧超带，最长连续2帧≈2.08ms，不是全回合超带总数。当时深度10.145021mm，下一决定腕角53.687468°/合力48.620486N。源码明确2µm由历史成功参考1.706µm残差外取整，是后评数值带；独立复核未找到用户亲定物理安全上限或厂家额定依据。负几何间隙仍是观测，不能称纯浮点噪声，也不能改failed。whole_assembly_review机械与视觉条件全true，额外检查仅source_key_containment=false，因此status=REVIEW_REQUIRED、complete_visual_assembly_verified=false。这不否定已经发生的机械到位和真实松手；也不允许声称全部原数值审核通过。

相关报告：run14/whole_assembly_review.json、turn_progress_physical_review.json、source_nail_body_review.json、source_nut_pad_review.json、source_key_containment_review.json、key_borderline_window_review.json、final_mating_visibility_review.json。独立复核目录artifacts/full_rotation_20260914/independent_final_guard_review_20260916：full14_terminal_release_supplement.json、full14_terminal_release_review_CN.md、key_band_meaning_review_CN.md。原始评估与阈值均未修改。

第4段跟踪瞬态已对齐：completed_stroke_robot_tracking_review.json与fourth_tracking_transient_physical_review.json。前20°机器人枢轴XY误差约0.020mm；约25.6°开始被2mm/s纠偏上限截断，末0.607844mm，仅最后87条控制行截断。实际Nu侧向约0.030mm、本体约0.024mm/倾角0.035°，手/Nu相对角变化0.82°；该段实际Nu仍转28.73°/本体前进0.540mm。1320机器人帧无arm_control.saturated。接触负载补偿沿原源码路径传递，并未遗漏开关；不能归因于速度上限一个参数，也不能把相对角变化全部称滑动。未为此新增物理实验。

耗时：主动作279682/960=291.335417s，五段转动合计12.880208s；整次墙钟15419.988s≈257min。run_timing.json：contact_callback5155.943s包含在native_physics9289.904s内；contact_counts1743.741、archive_append884.616，不重复相加。sensor_archive_and_runtime_summary708.207s。整体效率仍待改进，不称此前Float3[:3]优化解决整体耗时。

用户说明已重写docs/GRASP_PHASE_ROOT_CAUSE_20260915_CN.md，先讲本次实际结果和未过项。视频postrun_evidence/final_turn_and_release_actual_clip.mp4为原完整视频帧1409..1456的9.6s剪辑、5fps，无改模型/位姿；溯源final_clip_provenance.json。rotation_end_actual_frame.png是旋拧末段尚未张手的近景，released_hold_actual_frame.png与final_released_actual_frame.jpg是实际松手宽视图，不混标。原红带终态1024几何径向样本均由原Nu遮挡，与实际图像一同支持可见性报告，不宣称所有相机角度证明。

历史与早前失败依据完整保存在docs/history/CURRENT_CONTEXT_CN_20260916_before_full14_mechanical_result.md及GRASP_PHASE_ROOT_CAUSE_20260915_CN_before_full14_mechanical_result.md。保留13/27与此前失败资产。13曾到14.6042但硬停无松手，27是局部冷初始化并真实松手，本次14才是同回合机械结果。旧17.191125°只是13最后成功归档/记账命令，故障拍273525已推进但缺raw对象状态，不误写为物理最后下发角。旧FT原点漏换诊断已撤回，勿恢复；active轴线加硬旧分支不因历史恢复。

恢复后先核实用户当前目标。若继续处理数值质量或提速，围绕已保存的具体窗口做最小可归因诊断，保留本回合机械成功基线及原failed；不默认新开完整5小时实验，不把历史2µm数值带变成硬件事实，不放宽用户明确冻结或已验证的实际安全边界，不作硬件动作。
