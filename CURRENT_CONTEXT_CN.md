# 当前状态：高位全局相机1整机验证未完成

核验时间：2026-09-19T13:09:25.019389+00:00。本轮用户要求“使用高位全局相机1验证，成功则生成只有主视角和相机名称的录像”。验证已结束，未获得高位完整装配成功，不能交付成功全过程录像。没有运行中的主实验，也未安排自动重跑。

## 有效验收与当前结果

完整验收仍为同回合视觉粗定位、抓取搬运、两次固定G2看键、插入、旋紧到位和最终3秒完全松手；原物理与安全边界不变。最近完整成功基线仍为`artifacts/two_key_20260919/socket_plus20_full_01/run`（低位G1，df1afac，完整及3秒释放均过），未被高位候选替换。

本次工作树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/high-global1-full-20260919`。最新候选回合`artifacts/two_key_20260919/high_global1_full_03/run`，执行源码8959158fe2925492647ef334c04c5535a3260a2a，2026-09-19T12:59:01.430368Z结束，6705.369秒、252522步、exit2。旧session95957及PID962132均已结束，不再等待。

高位G1同帧两件粗定位和初抓独立验收通过：实际最低抬升55.9825毫米、2秒保持、3指甲本体接触。两次固定G2观测通过，视觉粗转-19.294955°，实际本体绕轴-19.310270°，第二次视觉剩余-0.070624°；已插入，原键槽几何门通过，0.5秒本体松手支撑逐帧通过，已执行的原指腹螺母接触通过。已执行前三段旋紧，但第四次抓握前视觉复核STOPPED，未完成旋紧到位及最终3秒释放，whole状态INCOMPLETE_NO_TERMINAL_RELEASE_RECORD。

## 当前阻塞与证据

`nut_regrasp_continued_02`中，手修正-0.580131°后视觉角误差-0.521842°超过原0.5°门。两幅图像的螺母估计227.275→227.800°；结束后实际螺母几乎没转（约0.000000392°），同帧实际螺母/编码器手部角差-0.219339°。证据支持跨视角估计波动触发误停，不能据后验真值补算在线成功。当前实际终深11.9381毫米，未达到原14.605毫米到位参考。

报告`reproducibility/two_key_20260919/high_global1_full_result_CN.md`，机器状态`high_global1_full_status.json`与`delivery_status.json`，精简证据`evidence/high_global1_full_03/`。原始诊断在回合`postrun_evidence/nut_grasp_visual_stop_diagnosis.json`。完整过程归档`docs/history/CURRENT_CONTEXT_CN_20260919_high_global1_full_worklog.md`。

## 配置和录像

高位装配配置`reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml`；本次Body候选`visual_body_contact_0p09.yaml`仅初次接触合拢降为0.09rad/s，原0.18的`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`保留。修复高位SAM中央端面漏外圈时，使用同帧普通深度的唯一重叠连通物体区域，再跑原几何门。原模型/材料/质量/CPU960Hz64/4/力速停止门/360°/最多6次抓握保持。

本轮录像`artifacts/two_key_20260919/high_global1_full_03/run/video/assembly_five_view.mp4`为263.2秒、1920×1080、5fps，仅五个名称。它是未完成装配的诊断录像，不是成功交付；已检查实际末帧红带仍露出，ffmpeg为0、渲染原生状态差均0。禁止用旧label_five_view.py加回其他文字。

## 恢复后的最短路线与边界

若继续修复同一方案，先针对保存的螺母两帧观测检验跨视角估计不一致，或在原有限角度预算、避碰和0.5°验收内做有界再次视觉修正。先取得能区分原因的局部证据，再决定新的完整回合；不放宽门、不凭旧成功原样盲跑，不用真值控制或拼接成成功录像。尚未实施这项后续修复。

仅仿真、hardware_authorized=false，一次一个主物理实验，原始数据/失败证据/未提交用户资产保留，不推送；原项目只同步本入口。无子代理授权。Isaac Python为/home/noob/WorkPlace/isaacsim/.conda-env/bin/python，规划Python为/home/noob/WorkPlace/kcgtest1/.venv/bin/python。每次恢复先核对真实进程状态。后验审查用已有原脚本和without_cyclic_gc，不能以测试/退出/文件生成代替物理结果。
