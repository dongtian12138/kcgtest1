# 当前状态：高位全局相机1完整仿真装配已验收

核验时间：2026-09-20T07:47:05.339280+00:00。完整装配验收及用户授权的GitHub新分支发布已完成。当前没有物理实验或发布任务在运行。

## 当前有效结果与证据边界

完整物理回合：`artifacts/initial_contact_fix_20260919/balanced_band_full_01/run`。实际动力学源码`686734f4d4324993e1fbd2024e95caf2ed437ec6`；171分38秒、301175步，全部运动及最后3秒完全松手已实际执行。原退出码2及原在线完成false保留，因为旧末端判断把卸力前后7.361微米落座位移当成释放后不稳定。

末端判断修复`84e534c96760c59997fad938fbac07d16331188b`只在所有动作结束后执行，使用原已在线消费的释放窗口观测。15帧极差0.259338微米<原5微米，因果时刻及完整采样覆盖通过；11项边界测试及旧完整基线重放通过。**此修复的验证是同回合真实传感记录重放，没有声称新末端代码重新执行了一整轮动力学。**原始记录/影像/原审核不改写；新版整体审核明确记录该区别，并复用摘要绑定的同一回合全量物理审查。

`whole_assembly_review.json`: VERIFIED、complete_visual_assembly_verified=true；strict3s accepted=true。source_nail、highG1、source_nut、two_key、camera、native_support、source_key_containment、source_band、实际末段图像均通过。原版whole只因旧online判据为false而REVIEW_REQUIRED，保存为whole_assembly_review_original.json。

## 关键结果

高位G1真实同帧两件粗定位；恰好4台功能相机。两段搬运，同一固定G2两次观键，视觉粗调-20.120149°、二次细调-0.037745°（图像指令量，不冒充真实剩余角）。掌心5DOF持续更新与键方向记忆，Body释放后关系停用。三原Nail保持期最小抬升55.989876mm/2秒/桌面0；Body卸力峰0.491534Nm<0.9；第4次Nut重抓误差0.013331°<0.5；6次抓取完成原预算内旋紧。

最终2880步=3秒：手原始冲量每步严格0，源金属止挡每步正接触、同128夹片每步受载、Body/Nut均清醒，记录深度14.604789466mm。全程最大推力备用坐标0.520647mm<原0.5999mm；键侧壁最不利残差1.862230微米<原2微米。三张原视频帧已实看；红带未露出，1024/1024源带样本被Nut遮挡。

## 交付与工作树

验收报告：`reproducibility/high_global1_acceptance_20260919/result_CN.md`；同目录acceptance_summary.json及evidence。完整录像：`/home/noob/WorkPlace/kcgtest1-performance-20260917/artifacts/initial_contact_fix_20260919/balanced_band_full_01/run/video/assembly_five_view.mp4`，5分13.8秒/1920×1080/5fps，只五视角名称，不拼接。计算用时2小时51分38秒，不是5倍提速声明。

实现工作树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支codex/high-global1-full-20260919。原项目只同步本入口，原有未提交资产不动。配置visual_body_balanced_band.yaml（原抓取高度+1.5mm，平直带中部）；assembly_high_global1_plus20.yaml；four_camera_two_key_high_global1.yaml。源码修复与原始证据均保留；2026-09-20已另建GitHub复现分支，见下文。

## 历史索引与边界

修复证据：reproducibility/initial_contact_fix_20260919/body_release/diagnosis_CN.md（后倒角接触）、palm_occlusion/result_CN.md（掌心遮挡）；reproducibility/nut_phase_surface_20260919/result_CN.md（Nut三角面匹配）；reproducibility/terminal_completion_20260919/diagnosis_CN.md（卸力与释放状态区分）。低位完整基线socket_plus20_full_01/df1afac仍保留，全部失败原始数据保留。

仅仿真hardware_authorized=false；不得在线对象/接触真值控制、改物体位姿、隐藏固定或增力造成功。960Hz/64/4、源几何质量材料、原力速/键槽/释放验收边界不变。后续任务按用户新指令，恢复时核对实际进程，不凭历史“运行中”等待。

## 2026-09-20 GitHub复现封存

发布分支codex/connector-four-camera-baseline-20260920；发布标签assembly-four-camera-high-global1-20260920。工作目录/home/noob/WorkPlace/kcgtest1-baseline-20260920；本机已在该独立目录从公开Release恢复297输入，另有2个高位G1输入随Git保存，固定SAM源码/补丁/四权重校验、记录模块重新编译、当前ROS资源构建、USD引用闭合及规划FK只读加载、启动--check均通过。第三方Python环境复用，未做跨机器零安装或新完整物理回合。

本分支入口README.md和docs/REPRODUCE_FOUR_CAMERA_BASELINE_20260920_CN.md；默认scripts/run_current_hand_assembly.py已绑定本次高位四相机成功配置。历史本机路径仅保留为原证据，其他电脑应从当前克隆目录和自身环境启动。用户授权的发布不修改主分支或旧基线。新分支已推送GitHub并从远端重新克隆；已核对509个Git源码/高位输入绑定、恢复297个Release输入、固定SAM与四权重、重新编译记录模块和构建当前ROS资源，--check通过。Release已公开，录像/证据包/SHA256清单均已匿名下载并核对字节与摘要。固定标签指向d6b6fc5001edaebfd73145d3005ee55d721362fd；后续分支提交仅追加发布核验文档，控制和配置与标签相同。

发布地址：https://github.com/dongtian12138/kcgtest1/tree/codex/connector-four-camera-baseline-20260920
固定发布页：https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-four-camera-high-global1-20260920
复现说明：docs/REPRODUCE_FOUR_CAMERA_BASELINE_20260920_CN.md。核验文件：reproducibility/four_camera_baseline_20260920/publication_verification.json、github_clone_validation.json。主分支及旧分支未推送改动。

若在其他电脑或旧工作树阅读本入口，应以发布分支的当前克隆目录为运行根目录，并指定当地三个Python环境；上文历史本机路径不是跨机器依赖。当前发布核验没有重新跑整轮动力学，原完整成功和末端传感判定重放的证据边界不变。
