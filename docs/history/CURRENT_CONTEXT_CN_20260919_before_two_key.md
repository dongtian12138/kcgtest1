# 当前状态：四相机与单次键观测改造已完成，名义整机验收通过

核验时间：2026-09-18T19:30:49.435053+00:00。本次已授权实现与验证工作结束，目前无物理实验运行，也未安排自动后续实验。初抓重复性和原5倍整轮提速目标尚未解决，如后续继续须按用户当前请求选择任务，不自动恢复历史实验。

## 当前有效基线与证据

- 实现工作树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/four-camera-single-key-20260918`。完整已验证执行源码提交`84bef70924b7244f47493a75c38a17fade6c3b58`；后续提交只增加报告、诊断配置及证据，未改变该实现。原`/home/noob/WorkPlace/kcgtest1`除同步本入口外未改其脏资产，未推送。
- 主Body配置`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`，主装配配置`assembly_fixed_cameras.yaml`（同目录），原contact_endpoint_timeout_s=.5保留；调节deadband=.005，力目标、±.01Nm验收、6连续帧不变。1秒配置未采用。
- 完整回合`artifacts/four_camera_20260918/full_chain_04/run`，2026-09-18T15:28:37.959009Z至17:59:51.662992Z，9073.704s、273285物理步。`whole_assembly_review.status=VERIFIED`且`complete_visual_assembly_verified=true`，独立`three_second_release_review.accepted=true`。
- 原CPU960Hz、64/4、TGS/contact-last、CAD/SDF、质量/质心/惯量、材料、被动关节、力/速/行程限均保留；仅已有无损记录优化，不恢复已回退数值加速候选。仅仿真，hardware_authorized=false。
- 功能相机恰好4条路径：G1同初始RGBD帧粗定位两件；固定G2只观察一次插头底部键；掌心位置/轴线5DOF；固定腕部精测Socket位姿和5槽。G2 eye[.378,.319,.244]/target[.490,.185,.275]，大搬运之后在3mm预接触高度、Socket侧方60mm看键，再短归中。录像旁观镜头不进入控制。
- 实际插头搬运两段.3970812+.0603910=.4574722m；手基点.4431966m；搬运峰值arm.120988rad/s<原.15。旧主搬运约3.4746m。
- KeyDirectionMemory由编码器传播手运动，掌心只修正位置/轴线，忽略圆拟合任意横向基；Body卸载前retire。接触开始主键轴向误差.0421348°，整个抓持期最大.2726773°；掌心没有测出额外轴向转动。中心误差最大21.627µm，轴.009570°。硬件延迟未标定。
- 感知/规划结果的可用时间由受保护保持动作推进物理后满足。Nu首次规划8.628619s对应8284步；首次回转2.246249s对应2157步；Nu相位规划/验证同样有真实保持。te_visual_seating_axis可选分段分支修复了重复缓存帧确认，但当前配置未启用，只做软件检查，不称动态覆盖。
- 在线到位True：独立观测深度14.603779/14.603582mm，实际张手保持3s、加载扭矩2.003314Nm。第四段在累计356.248994°因弹性裕量换抓，第五段再转1.708548°后到位候选及最终释放；原360°上界不变，换抓停止本身不等于成功。
- 原始最终3s[270405,273285)共2880帧：手冲量严格零、源止挡和同128簧套每帧正载、Body/Nut醒着；实际深度14.603955mm恒定，满足14.605mm±10µm。全回合最大备用轴坐标.522465mm<原.5999mm。
- 源键对槽面最大残差1.175528µm<原2µm；三原NAIL/Body初抓通过，Nu保持原允许区域（本轮两甲一腹），无其它受载对象。原红带1024/1024径向样本遮挡，已查看同回合旋拧末/释放中/最终视频帧。引擎健康/身份/有限性/真值隔离均true，PhysX容量警告0。
- 原通用evaluation与exit2含旧全指腹/旧tensor通道的未通过标签，原报告保留；不得称所有JSON均通过。当前成功由适用源区域/源几何/原生接触/在线判断及完整汇总共同建立。
- 旧完整参考`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/run`耗时15909.99s；当前进程比约1.7534倍，**未达到5倍**。独立后验检查另计；约7.6倍路程比不是整轮速度比。

完整报告`reproducibility/four_camera_20260918/full_assembly_result_CN.md`；精简证据`evidence/full_chain_04/`，原始数据原位保留（truth msgpack.gz16342211301bytes、录像44762666bytes）。交付状态`delivery_status.json`。

## 初抓扰动与诊断：结果及边界

均只检查初抓/抬升/2s保持，不代表扰动下完整装配。

- `variation_plus_grasp_01`：+1mm/+1°，原.5s，2026-09-18T18:25:46.836303Z结束，496.825s。未重现旧被动关节超速，但在预载稳定门停止，未抬升。480帧最多5连续合格，要求6；末第一指**重力补偿后控制信号**误差-.0110755Nm超±.01。通用减零偏摘要的“末帧在带内”不是该信号，已更正解释。
- `variation_minus_grasp_01`：-1mm/-1°，原.5s，2026-09-18T18:52:39.146998Z结束，711.787s；345帧获得6连续合格，实际抬升55.9769mm/2s/三NAIL Body接触/保持桌面零载通过。
- `settling_1s_plus_grasp_01`：+1mm/+1°，仅最大等待.5→1s诊断，sourceecf7b0e，2026-09-18T19:14:54.066264Z结束，710.949s。358帧/.372917s已过门，早期接触303帧，均未用扩展窗口；实际抬升56.0074mm/2s/source_NAIL通过。
- 两个+方向回合初始视觉位姿和下发运动计划完全相同，但前置实际视觉等待25054与24438步，相差.641667s，之后力轨迹不同。不能据此确定唯一根因；也没有证据证明1s是修复，故**未采用1秒配置，不自动继续扫描超时**。重复性仍未解决。

详细记录`initial_grasp_robustness_CN.md`、`variation_plan.json`、`settling_diagnostic_plan.json`及evidence对应子目录（含机器人信号重放输入）。旧+1mm/+1°首次接触后三步f1j3=-4rad/s>3的原证据在`reproducibility/improvements_20260916/evidence/pose_variation_early_abort`；不要把当前未重现写成已证明彻底修复。

## 继续工作时的约束和入口

先核对用户当前任务；无新请求不启动新回合。一次只运行一个主物理实验，运行时冻结加载代码，用run/STOP_REQUEST受控停止，不SIGINT、不在线延长预算；没有子代理授权。真值仅后评，不进入在线控制；无硬件授权，不增力/放宽原物理验收，不删失败数据。

初抓后评`evaluate_visual_body_grasp.py`；Nu源面须`evaluate_source_nut_pad.py --geometry-plan artifacts/grasp_capacity_20260914/selected_two_nail_geometry.json`。键/3秒/原生支持/相机/箭头分别用已有review脚本。红带用`reproducibility/improvements_20260916/postreview/review_source_band_occlusion.py`加真实录像查看；evaluate_rear_face_visibility是合成遮挡消融，不是红带检查。完整汇总旧2s项须同时满足独立3s；本轮已全部完成，不重复重扫。

详细历史见docs/history。主要失败链：早看键搬运后误差.555°；高位50mm看键后长下探误差约.4°并停在.777mm；低位01键掠视、02 G1学习粗姿态错误、03腕部遮挡，04改为低位60mm侧距通过。full_chain_03因审计发现Nu规划漏计物理等待而受控停止，并非物理失败；完整full_chain_04随后通过。旧contact_integration_candidate/prepare已过时，不重新应用。

环境：Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同环境；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM Python `/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`，SAM根`/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`；OPENBLAS_NUM_THREADS=1。测试禁用pytest自动插件，sys.path加src/kcg_connector、isaac、carts_v2。普通rg被忽略时用限定目录--no-ignore。
