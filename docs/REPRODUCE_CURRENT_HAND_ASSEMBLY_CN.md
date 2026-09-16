# 查看与重跑当前三指手装配

核对日期：2026-09-16。参考回合是 `visual_complete_all_reserves14`：同一回合已完成视觉驱动的桌面抓取、搬运、入槽、旋拧到位和真实3秒松手保持，最终深度约14.60449 mm。原键槽2 µm数值比较带有一项超带，原整体评估仍为 `REVIEW_REQUIRED`。这是仿真结果，尚无多次重复成功率或硬件验证。

完整实际视频已经生成，长291.4秒（4分51.4秒）、1920×1080、5帧/秒，文件约42.4 MB。它按仿真时间播放，电脑实际计算和收尾用了约4小时17分钟。视频从本回合初始视觉阶段开始，包含抓取、搬运、对准、插入、换抓、旋拧和最终松手；画面主要是四视角场景，识别输入和位姿估计另存为文件，没有把算法的识别框与说明字幕合成为讲解片。

[完整实际视频](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/video/assembly_four_view.mp4) · [9.6秒末段旋拧与松手](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/postrun_evidence/final_turn_and_release_actual_clip.mp4) · [初始RGB图像](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/initial_rgbd/observation/rgb.png) · [初始视觉定位结果](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/initial_rgbd/body_localization.json)

直接播放已有完整视频：

```bash
ffplay -autoexit /home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/video/assembly_four_view.mp4
```

录像的大致位置如下。这些是视频时间，不是电脑计算耗时；视觉计算会在对应场景之间进行。

| 视频位置 | 场景 |
|---|---|
| 0:00—0:16 | 传感器参考、初始RGB-D识别、靠近并抓取 |
| 0:16—0:37 | 抬升、离桌保持 |
| 0:37—2:32 | 搬运、重新观察、对准插座 |
| 2:32—3:10 | 小力探测与入槽 |
| 3:10—3:13 | 松开插头本体 |
| 3:13—3:39 | 调整到螺母抓位并夹紧 |
| 3:39—4:47 | 五段旋拧及中间的卸力、回腕、视觉重抓 |
| 4:47—4:51 | 最终卸力、张手和保持 |

已经增加[本机重跑入口](/home/noob/WorkPlace/kcgtest1/scripts/run_current_hand_assembly.py)。先检查文件和命令，不启动物理仿真：

```bash
cd /home/noob/WorkPlace/kcgtest1
python3 scripts/run_current_hand_assembly.py --check --gui
```

实际启动 Isaac Sim 窗口并重跑：

```bash
cd /home/noob/WorkPlace/kcgtest1
python3 scripts/run_current_hand_assembly.py --run --gui
```

该命令先运行一次新预检，核对预检、控制预检、引擎状态和模型身份检查；通过后才启动完整装配。预检和装配使用同样的GUI选择。预检窗口结束后会进入正式运行，不能把预检结束当成完整装配结束。

若按原成功回合的无GUI方式运行并录制视频，去掉 `--gui`：

```bash
cd /home/noob/WorkPlace/kcgtest1
python3 scripts/run_current_hand_assembly.py --run
```

原记录采用无GUI方式。`--gui` 是现有Isaac入口支持的查看选项；本次已核对参数及干跑命令，没有再执行一次GUI完整物理实验，因此不声称GUI运行已经重复验证。

入口复用原命令和原物理参数，核对54个已记录绑定的源码、配置与场景文件。若这些文件发生变化，会列出差异并停止，不自动覆盖工作区。输出自动写到新的 `artifacts/reproductions/current_hand_<UTC时间>/` 目录；也可通过 `--output-root` 指定一个尚不存在的目录。原成功回合和失败证据不会被覆盖。

新目录中的 `preflight.log` 是预检日志，`assembly.log` 是正式运行日志，`run/video/assembly_four_view.mp4` 是本轮录制的视频。运行时可以在另一终端用 `tail -f` 查看脚本打印出的日志绝对路径。每次预检预算300秒，正式运行预算18000秒，其中保留1200秒收尾；这些沿用参考回合的有界预算。前台按Ctrl+C会通知原有包装器中止并保存记录，保存可能需要等待。

本机Isaac环境默认为 `/home/noob/WorkPlace/isaacsim/.conda-env`，GUI需要图形桌面。本次核对时图形显示为 `DISPLAY=:1`。现场运行仍可能耗时数小时；若只是想看已经完成的效果，播放保存的视频更直接。参考回合的原始物理档约19.6 GB，另有传感器、图像和日志，需要为每次重跑留出足够磁盘空间。

入口负责重跑和记录，尚未把所有专门后评及人工图像核对打包成自动验收。进程退出码不能代替机械结果：原回合的包装器返回2，但原始记录确认了机械到位与真实松手；原键槽数值比较带仍未全过。新运行必须检查自己的实际深度、松手保持、原止挡、簧套、源面和键槽，不能复用旧回合的通过结论。

GitHub状态也已只读核对：当前本地分支为 `codex/visual-assembly-v1`，HEAD为 `409d5c2`，远端没有同名分支；最新关键修改仍未提交。本回合用到的命令JSON、装配YAML、连接器模型和手模型受 `.gitignore` 的 `artifacts/*` 规则忽略。因此，这个入口目前依赖本机保留的数据，尚不是从GitHub重新克隆即可使用的交付包，也没有承诺每次重跑的数值完全一致。

后续封存需要把这次实际执行的源码和配置提交到明确的新分支，同时整理所有必要资产、环境版本、运行与后评入口，并给大文件提供可获取且可核对的版本信息。仅上传代码不足以复现。当前原始命令与执行快照仍保留，便于完成这一步。

[原完整命令](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14_command.json) · [原预检命令](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/preflight_all_reserves13_command.json) · [执行快照清单](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/execution_source_snapshots/manifest.json) · [实际结果和限制](/home/noob/WorkPlace/kcgtest1/docs/GRASP_PHASE_ROOT_CAUSE_20260915_CN.md)
