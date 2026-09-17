# 当前任务：完成用户截图的四项装配工作

核验时间：2026-09-17 07:27 UTC。当前第一次新候选完整验收已VERIFIED，原2µm键槽和完整3秒松手均通过；主程序和全部后评已结束，正在保存提交并准备第二次名义重复。simulation-only，hardware_authorized=false；同时一个主要物理实验。

## 当前有效结果与执行位置

- 活动树 `/home/noob/WorkPlace/kcgtest1-improvements-20260916`，分支 `codex/connector-assembly-improvements-20260916`，执行代码25a0d6f68771080fe60a7dde948cdaa4e8d5927f已推送。原目录脏资产保留；封存基线树不改。
- 当前完整回合 `artifacts/full_validation/contact_last_gc128_repeat01_restart01/run`：294560帧、25520469055字节归档，索引封存一致；motion_timing已生成。主程序session17980/PID21229已退出2；通用evaluation仅前35649帧pickup且旧全指腹门不适用，保留原报告。原引擎健康、身份和真值隔离均true。
- 实际最终Body深度14.604580849409077mm，欠目标0.419150591µm。独立完整3秒、2880帧每帧手冲量精确0，8组原StopBox正载，与同一套128簧套同时承载；Body/Nut均不休眠，深度保存精度下恒定。独立报告父目录review/independent_terminal_release.json与中文说明。
- `source_key_containment_review.json` accepted=true：五键最差间隙约−.910355/−.651432/−1.848420/+101.983432/−.859048µm，第三键raw230008最差，原2µm未改，余量约.151580µm。原源面Body/Nut检查均通过；原指甲抓起保持最低离桌56.109771mm，2秒桌面0、三甲100%。螺母真实区域为第一/第三指甲、第二指腹。
- 六段实际Nu角89.047246/89.549457/31.160159/70.687890/73.206586/.037954°。对应腕指令90/90/33.030902/71.152244/75.068092/.748762°，总约360。第三/第五f1传动余量正常换抓，第四腕弯矩正常换抓，无硬abort。第五已经到位，不能说第六补转导致到位。
- 实际三张末段图片已亲自看过、红带未露；1024原源面径向样点都被螺母遮挡。final_mating_visibility_review=true。实际视频run/video/assembly_four_view.mp4；连续21秒末段片段postrun_evidence/final_two_strokes_and_release_actual_clip.mp4已完成。
- 整体review session23420已退出0，whole_assembly_review.json=VERIFIED，所有mechanics/visual/additional项true；最大thrust.516563319mm<.5999mm。独立最差键三帧复算与主报告完全相同，源几何/原2µm绑定未变，独立报告review/independent_key_report_review_CN.md。
- 当前说明docs/ASSEMBLY_COMPLETE_VALIDATION_20260917_CN.md已更新为第一次完整通过，相关小证据已复制至reproducibility/improvements_20260916/evidence/contact_last_gc128_repeat01，保存原25a执行manifest。当前正在提交交付；仅修正后续收尾异常，不改控制/物理。

## 立即继续

1. 第一次完整验收已结束通过，不再等待旧session。
2. 异常收尾已修复，12项针对性测试通过，768绑定检查通过，待提交推送；不是已证实主机重启原因。
3. 用相同控制/物理配置开始第二次名义重复 `.../contact_last_gc128_repeat02`，新预检。一轮失败则先定位最早原因，不盲重跑。
4. 两次名义完整通过后运行已准备的 `--pose-variation`，初始横移1mm、偏转1°，摩擦仍.45。尚未执行。
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
