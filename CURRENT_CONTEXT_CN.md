# 当前任务：高位全局相机1完整装配通过原验收

核验时间：2026-09-19T20:07:42.598204+00:00。用户最新明确要求“继续解决问题，直到成功完成验收”；不以局部通过、失败报告或程序结束代替完整结果。

## 当前运行与约束

当前无物理实验运行。balanced_band_full_01/run（源码686734f4d4324993e1fbd2024e95caf2ed437ec6）于2026-09-19T20:40:11.687208Z结束，10298.020652s，exit2，工具session92759已关闭。完整动作及3秒完全释放已实际执行；source_nail、highG1、source_nut、two_key、camera、native_support、source_key_containment、strict3s、source_band全部通过，三张同回合末段图像已实际查看并写visibility accepted。全局2视觉粗调-20.120149°、二次余差-0.037745°；Body松手峰0.491534Nm<0.9；第四次Nut视觉误差0.013331°<0.5。

唯一剩余审查：原版whole_assembly_review正在离线扫描同回合全部原始接触，工具session55826，脚本/tmp/kcg_original_whole_686734f.py。完成后会写whole_assembly_review_original.json；必须保留。已生成的其他审查无需重复。原全量审查约已十余分钟；不是物理运行，恢复核对真实进程。

实现工作树/home/noob/WorkPlace/kcgtest1-performance-20260917；分支codex/high-global1-full-20260919。原项目仅同步本入口。simulation-only，hardware_authorized=false；无子代理授权、不推送、不删除原始数据。一主物理实验，运行时加载源码配置冻结；停止用run/STOP_REQUEST，不在线延预算。原960Hz/64/4、几何/质量/材料、力速保护与验收不变。禁止在线对象/接触真值控制、启动后改物体位姿、隐藏固定和增力造成功。

当前配置reproducibility/initial_contact_fix_20260919/visual_body_balanced_band.yaml；装配reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml；相机four_camera_two_key_high_global1.yaml。+20°插座角仅启动前场景设置，不是在线角度输入。启动器/tmp/kcg_launch_balanced.py。新配置独立预检balanced_band_preflight_01已194.817s通过全部原门。完整预算21600s/300s收尾，录像5fps、只五名称。

## 有效验收与最近完整成功比较基线

验收：高位G1同帧两件粗定位；三原Nail抓Body并真实抬升≥50mm/保持2s/桌面脱离；仅四台功能相机；两段搬运；同一固定G2两次看键，中间视觉绕轴粗调；掌心5DOF与键方向记忆；腕部插座位姿/键槽；插入；螺母旋紧到原源几何止挡；最终3秒完全松手，逐原始帧手部接触冲量严格为0、同128夹片受载、金属止挡接触且Body/Nut清醒；原源键槽容纳和合法接触区门全部通过。主视角+四相机同回合录像，文字仅主视角/全局相机1/全局相机2/掌心相机/腕部相机。

保留的低位G1完整成功：artifacts/two_key_20260919/socket_plus20_full_01/run，源码df1afac5bd5d9bdb56dd49b58ef8b9728092ed08，10878.632s/300583步，whole VERIFIED且严格3秒释放通过。高位G1尚无完整成功；high_global1_full_03曾前三段旋紧通过、停于第四次Nut视觉误差门。不能把旧低位结果或局部试验写成本轮成功。

## 已保留修复与证据

1. 高G1仅识别插座内端面时，从同帧深度前景恢复外轮廓，仍过原几何门，4b07471；未读取真值。
2. Nut视觉从随机采样点距离改为原静态CAD三角面距离（Warp CPU BVH），a308389。16帧原质量门过；local_fourth_grasp_02真实局部重抓误差0.021783°<原0.5°，原接触区和键槽后评通过。仅局部验证。报告reproducibility/nut_phase_surface_20260919/result_CN.md。
3. 初抓0.09rad/s只降低接触接近速度，不能声称其根治：high_global1_full_04初触仍2.493Nm>0.9Nm。旧接触在螺纹斜面，原CAD平直带z约[-0.02995,-0.02805]m，半径17.729mm。原hand相对Body z=-0.45243714；+2mm候选=-0.45443714，避免了初触螺纹冲击。
4. 可选prelift_effort_settle_timeout_s=1.5（原默认0.5，允许≤2），其余夹紧目标/6连样/.01Nm/0.9Nm保护不变。smooth_band_grasp_02独立原Nail/抬升55.9805mm/保持2s/桌面0/最大0.520579Nm全部过；实际等待0.430208s，未用到扩展，不能归功于延长上限。报告reproducibility/initial_contact_fix_20260919/stage_result_CN.md。
5. 掌心端面圆拟合在原拟合失败后，按当前深度近前景排除遮挡切边，仍用原半径和原RMS门，另需≥100点/18角度箱/信息eig≥0.2；efb0599。旧失败帧RMS0.234555→0.041191mm，原门0.110925mm，排除161保留1074点/34箱；24旧帧位姿不变，8相关测试过。仅观测5DOF，不估绕轴角。报告reproducibility/initial_contact_fix_20260919/palm_occlusion/result_CN.md。前一完整候选已过搬运/两次键/插入，证明它在真实流程中生效。

## 本轮单变量与直接原因

前一smooth_band_full_03（efb0599）2026-09-19T17:21:42.334289Z结束，2776.934s，exit2，session96449已关闭。完成初抓/两次G2/搬运/插入：视觉粗调-20.297327°，第二次余差+0.002257°。随后Body松手减力第168203步因f1反力矩1.437649Nm>原0.9Nm停止，尚未抓Nut。

原松手控制已单调打开，不存在重夹/指令跳变。减力1.710s时约0.06Nm；Body正常下沉0.108mm，与旧成功的约0.111mm相近。异常帧f2出现0.323823Ns总冲量，Body额外下降0.518mm。用事件前一帧映射，异常点Body z=-0.029997419m，距CAD后倒角2.914微米；引擎法线轴向0.908与CAD约0.148不一致。结论是后倒角附近异常接触响应，不能宣称真实卡死或唯一引擎根因。

本轮只把手沿Body轴从+2mm回到+1.5mm（z=-0.45393714），向平直带中部回移0.5mm。原Nail/运动学预测首次接触z=-0.028296/-0.028452/-0.028281m仍在平直带内，增加后倒角余量0.5mm。其余配置不变。初抓已走过，但尚须验证松手及完整结果。

诊断/候选依据reproducibility/initial_contact_fix_20260919/body_release/diagnosis_CN.md、candidate_basis.json、contact_source_geometry.json。原始artifacts/initial_contact_fix_20260919/body_release_diagnosis/release_history.json保留。原CAD网格/tmp/kcg_body_exterior.npz；离线脚本/tmp/kcg_release_surface.py、kcg_predict_balanced_band.py。不要把单调卸力当新修复（原代码已有）。

## 当前末端判定修复与紧接工作

旧online_completion=false的原因是把夹持状态与松手状态的深度拿来做5微米稳定性比较。真实卸力落座7.361微米（14.597428→14.604789mm）；完全松手的2880步物理深度恒定，所有原3秒条件通过。该窗口15帧已在线消费的掌心观测深度极差0.259338微米<5。相机没有误估这次落座，错误在比较了不同载荷状态。

已改four_camera_online_completion及post_entry末尾，只对完整释放窗口内的实际已消费相机序列做最终判定，保留5微米/20微米/3秒及所有原物理门。按原请求周期0.2秒、实测处理延迟和最大观测年龄0.5秒核对连续覆盖与因果性。没有新增动作/物理步/图像/真值输入。11项相关边界测试过；本轮15帧及旧成功基线14帧重放都通过。报告reproducibility/terminal_completion_20260919/diagnosis_CN.md。原始控制器false不得改写，最终必须说明末端修正经同回合传感重放验证，没有再重跑整轮动力学。

下一步：等待session55826原版whole审核完成，先核对它是否仅online标记为false，其他所有物理、视觉、几何项均true。然后用当前evaluate_visual_assembly_v1.refresh_terminal_decision(RUN, RUN/whole_assembly_review_original.json)，只更新末端传感判据并绑定原审查摘要，复用已完成且不变的全量接触审查，避免再次扫描20多GB。新whole必须complete_visual_assembly_verified=true且原strict3s accepted。若还有其他真实失败继续解决，不改门限。

实际帧已抽取并查看：run/postrun_evidence/rotation_end_before_release.png、released_hold_midpoint.png、final.png，来源原始assembly_five_view.mp4。主视角和G2未见露出的红色指示带，1024/1024源带径向样本被Nut遮挡；原始录像313.8s（5分13.8秒）、1569帧、1920x1080、5fps，只五名称。

源代码与诊断文档提交后重新生成online_completion_sensor_replay.json使decision_code_commit绑定实际修复提交；核对原始历史false仍在。最后写完整验收报告、同步本入口、展示本轮实际五视角录像，说明物理全过程实际完成与末端判定重放两种证据的区别，不把新代码说成重跑过全程。不推送，不删除失败数据。

Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python。OPENBLAS_NUM_THREADS=1，PYTHONPATH=src/kcg_connector:src/kcg_connector/isaac:src/kcg_connector/isaac/carts_v2。离线原始审查gc.disable。当前工具session24916（各分项后评）和81112（帧抽取）均已关闭。
