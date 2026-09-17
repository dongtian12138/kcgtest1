# 当前任务：用户截图四项工作的最终交付

核验时间：2026-09-17T12:39:09.344999+00:00。两次名义完整装配均已 VERIFIED；随后 1 mm/1° 初始位姿变化已执行，在初始抓取接触阶段停止。无主要物理进程，session40317/cell198 已结束。现在仅补独立失败审查、发布第二轮与变化案例视频/证据、核对 GitHub 交付；不得再按旧“运行中”等待，也不自动重跑四小时整场。

## 当前有效验收与结果

- 活动树 `/home/noob/WorkPlace/kcgtest1-improvements-20260916`，分支 `codex/connector-assembly-improvements-20260916`。生产源码 HEAD ab129ee333b573dda98c9582d947b506f5e1ea74 已推送。其后仅本次文档/证据更新，未改控制、物理、力速或源几何。原树 `/home/noob/WorkPlace/kcgtest1` 的原脏资产保留，仅镜像本上下文。
- 第一次完整成功：`artifacts/full_validation/contact_last_gc128_repeat01_restart01/run`，实际执行25a0d6f。294560帧/25520469055字节。Body深度14.604580849409077 mm，最差键1.848419870833 µm，原2 µm通过。2880帧完整3秒手原始冲量精确0，原止挡每帧8组，同一套128簧套逐帧承载，Body/Nut均醒着，后备轴向全流最大.516563319 mm<.5999 mm。
- 第二次完整成功：`artifacts/full_validation/contact_last_gc128_repeat02/run`，实际执行baf6d4f。295421帧/26530655597字节。Body14.60440203547475 mm，最差键1.866570825613 µm，原2 µm通过。完全张手[292541,295421)2880帧手原始冲量精确0，原止挡每帧9–11组，同128簧套承载且醒着；后备轴向全流.511467930 mm<.5999 mm。两轮实际末段图均已看，源指甲/指腹与视觉检查通过，whole VERIFIED。
- 两轮legacy通用拾取评分与主程序退出码2原样保留，专用实际验收通过；不能声称每个JSON全PASS。baf比25a仅修记录收尾异常，物理/控制一致。
- 变化案例：`artifacts/full_validation/contact_last_gc128_pose_x1mm_yaw1deg/run`，实际执行ab129ee；新预检0、主程序2，无主机重启。14720帧/35256838字节，索引匹配，归档封存。第14719步 `parallel_contact_approach`，f1j3=-4.000000477 rad/s > 原3 rad/s，JOINT_SPEED_ABORT，尚未抓起或装配。无motion_timing是因为未进入装配阶段，run_timing和最终封存齐全，不应伪造完成marker。
- 变化事实：初始X+1 mm、yaw+1°，摩擦.45且物理启动前布置；相机中心后验误差87.4086 µm，手指每步指令+0.0001875 rad、臂目标不跳。14716 F1→Body，14718 F2→Body，14719 F1末节速度/四杆约束残差突增；该窗口无Nu正载。主要假设为接触建立与四杆约束耦合瞬态，唯一主导点/约束未知。不是3 rad/s线上舍入误停，不是螺母旋拧失败，也不能把平移与转角影响拆开归因。

## 四项工作与证据入口

- 用户采用的四项顺序已执行：封存基线与复现、局部等价提速、腕误差排查、维持2 µm并做两次名义重复和一例变化。验证得出2次通过/1例更早失败；不能将有限验证说成全部初始位置鲁棒。最终中文入口 `docs/ASSEMBLY_FOUR_ITEMS_DELIVERY_20260917_CN.md`。
- 两轮报告 `docs/ASSEMBLY_REPEATABILITY_20260917_CN.md`；环境/下载/GUI和headless命令 `docs/REPRODUCE_CURRENT_HAND_ASSEMBLY_CN.md`。跨机器零安装与GUI完整重跑均未宣称已验证。
- 旧第四段虚拟中心.607844 mm不等于真实Nu偏心.029618 mm；初始手—Nu关系预测差.530467 mm。第二轮六段也有关系变化。滚动/接触迁移/柔顺/纯滑移未唯一分开。控制 `planar_hold=true` 将力顺应XY让位清零；XY力仍进10ms负载补偿与50ms保护。第二轮第4段达到1 Nm正常换抓线；早期弯矩增长时未触2 mm/s，不能断言提高限速会解决。
- 已验证局部性能：481密集物理步72.586→60.443 s且全部原始编码字节相同；1024帧收尾位置统计19.763→8.850 s，完整首轮结果相同；单帧17901点有限数检查4.939→3.524 ms。未删点、未降频。有限数等价限SDK Float3/固定decoder域，不对任意畸形Python字典承诺。sensor历史tuple未加入生产。
- 实际整场仍慢：运动242.88/245.65分钟，主程序关闭前261.95/265.17分钟。不同接触轨迹不能作为整场速度AB结论。
- 独立代理 `/root/independent_final_guard_audit` 是用户明确授权的现有只读复核，勿另起新代理。本次变化末段和名义对应有界窗口审查已完成，实际输出 `artifacts/control_review/pose_variation_early_abort/independent_*`。此前控制/指节/末3秒审查均已完成，随版evidence保留。

## 当前交付状态与收尾

- 原基线树 `/home/noob/WorkPlace/kcgtest1-baseline-20260916`，分支codex/connector-assembly-baseline-20260916，d64fa4a；Release `assembly-baseline-20260916`：121458865字节/297资产，旧291.4秒视频，旧键2.147659 µm失败保留。
- 首次完整通过 Release `assembly-first-verified-20260917` 已公开且4附件GitHub SHA匹配，tag baf6d4f，实际执行25a。干净下载目录已校验471 Git+297资产=768项，固定SAM源码补丁/4权重，ROS构建及nominal/variation --check通过，复用本机环境，不算额外物理重复。
- 正准备 Release `assembly-repeatability-20260917`，文件在 `artifacts/deliveries/assembly-repeatability-20260917/`：第二轮307.8秒完整视频、22.8秒连续末段、变化15.4秒提前停止录像、变化原始档69.7 MB、中文notes。独立报告和161项small evidence打包已完成，还需提交本次文档/证据、推送分支/标签和上传核对附件。未经这些实际完成不得说新Release已公开。
- 新变化小证据已复制至 `reproducibility/improvements_20260916/evidence/pose_variation_early_abort/`；独立结果已复制，review状态/清单已更新。原case记录不改、不删。
- 最短后续区分实验仅作为提案：同变化场景，初始合拢0.18→0.09 rad/s，其他参数和3 rad/s保护保持，到抓起/保持结束。不改交付配方，不盲扫，不自动开新的整场。用户续作若要求修复该最早问题再依现有证据执行。

## 运行环境与边界

`ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env`；`KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；`KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`；`KCG_SAM6D_ROOT=/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`。

Isaac6.0.1.0/Python3.12.13，OmniPhysX110.1.13，CPU960 Hz/64位置+4速度迭代，关节接触最后求解；RTX5070Ti/driver595.91.07。source control10 ms loadFF、原50 ms保护，最多6抓/腕指令360°。所有运行simulation-only/hardware_authorized=false；真值只在结束封存后审计，原始失败、用户脏文件不删。

更详尽历史见 `docs/history/CURRENT_CONTEXT_CN_20260917_before_four_item_delivery.md` 与 `...before_pose_variation.md`。旧load10ms失败、冷重建自行沉降（不可算连续成功）、早期主机重启中断均保留。当前没有应继续等待的旧物理进程。
