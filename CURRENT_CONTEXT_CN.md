# 当前任务：继续修复，直至高位方案完整通过验收

核验时间：2026-09-19T17:44:34.867301+00:00。用户明确要求“继续解决问题，直到成功完成验收”。必须持续推进，不以局部报告结束。当前抓取已通过，掌心遮挡修复已重放验证，准备新的完整回合。

工作树/home/noob/WorkPlace/kcgtest1-performance-20260917，分支codex/high-global1-full-20260919；原项目只同步本入口。仅仿真hardware_authorized=false，无子代理授权，不推送，不删除原数据。一主物理实验，加载源码配置冻结，停止用STOP_REQUEST，不在线延预算。

## 有效验收和已保留基线

验收保持：高位G1同帧两件粗定位、三原指甲抓Body抬升≥50mm保持2s、两段搬运、同一固定G2两次看键中间视觉粗转、掌心5DOF主键记忆、腕部插座/槽、插入、旋紧到位、最终3秒完全松手、原源几何/接触/受力门；主视角+四相机录像只显示名称。原低位G1完整成功socket_plus20_full_01/df1afac及源版本保留。

螺母视觉采样误停修复a308389已完成：原随机采样点距离改成静态CAD三角面距离（Warp CPU BVH）。16帧原质量门全过，局部local_fourth_grasp_02实际误差0.021783°<0.5°，夹持/原接触区/键槽几何后评通过。源码当前6b6c15f含该修复；没有改物理几何、力/速、5°修正上界等。报告reproducibility/nut_phase_surface_20260919/result_CN.md。该修复保留。

高位full03/8959158曾完成初抓到前三段旋紧，停于第四次螺母视觉门。full04/98ff9b2已在初抓停止（466.521s），f2j1峰2.493Nm>原0.9Nm，未抬升未执行新螺母算法。原0.09rad/s降速未根治接触冲击。所有旧session（43043、11300、38425）已关闭。

## 当前主要假设与最小候选

已对照同为0.09rad/s的full03/04原始初触：成功f2首次接触相对闭合527步，关节反力矩0.149Nm；失败531步初触即2.493Nm并伴大轴向冲量。两轮此前Body位置差约1微米、姿态和手指位置差很小；f2均先触，不能归结为第一指先推物体。

接触点落在Body局部z约-0.027m的螺纹段，源码外表面在该段法线含显著轴向分量。逐高度/周向72射线对照原CAD，在z约-0.0298..-0.0281m有连续平直圆环，半径17.729mm。现假设是原指尖落在螺纹/斜面交界引起卡挤/接触敏感性；不能宣称唯一数值根因。选择只将hand相对视觉Body的z目标减2mm（世界抓取位置上移约2mm），几何/材料/质量/碰撞表示/960Hz64/4/0.9Nm及其他门全部不变，接近速度仍0.09rad/s。

候选reproducibility/initial_contact_fix_20260919/visual_body_smooth_band.yaml。唯一物理变量object_from_hand_row_major[11]: -0.45243714 -> -0.45443714。原计划及候选原Nail/运动学/全周向外包络预测对照在candidate_basis.json，三候选接触z=-0.028796/-0.028952/-0.028781m均在平直圆环内。只是几何筛选，必须实际验证。原始诊断artifacts/initial_contact_fix_20260919，脚本/tmp/kcg_initial_contact_compare.py、kcg_contact_order_compare.py、kcg_body_contact_surface_check.py、kcg_body_profile.py、kcg_predict_smooth_grasp.py。原Body网格数据/tmp/kcg_body_exterior.npz。

## 已通过抓取，当前视觉修复与紧接步骤

平直圆环抓取+1.5s等待上限的smooth_band_grasp_02已通过原源Nail接触/55.9805mm抬升/2秒保持/桌面0/最大反力矩0.520579Nm；实际0.430208s达到力带，未用到额外上限。此前前段01仅因0.5s等待未6连样停止，接触冲击已未重现。该高度与等待配置继续保留。

smooth_band_full_02（a9673f2，session59769/PID1398099）已2026-09-19T16:04:27.707179Z退出1，674.322s；初抓、抬升、保持均控制完成，实际力带0.476042s通过。第一张掌心图像因手指遮挡切边被当圆轮廓而拒绝：RMS0.234555mm>原0.110925mm。未搬运，未对键。当前无主实验运行。

新视觉修复在te_plug_five_dof_geometry._coarse_face_center新增仅失败后使用的前景深度遮挡过滤：当前深度比当前端面平面更近>2倍原平面噪声带时识别遮挡，排除相邻边界后拟合实际可见圆弧。保留原圆半径、原RMS要求、原平面支持门；额外要求≥100点、至少18/36角度箱和信息最小特征值≥0.2。保留原完整轮廓RMS及被排除点数。原成功路径不变；不估计Body绕轴角、不读对象真值。session同时在测量前保存measurement_input.json，便于失败重放，不返回真值。

同一失败帧过滤161点、保留1074点、34箱覆盖，RMS0.041191mm通过；结束后独立真值位置误差8.54微米、轴误差0.000627°。24张旧观测位姿矩阵不变；同步G2/掌心宽ROI规则同帧也过；8项相关单测过。报告reproducibility/initial_contact_fix_20260919/palm_occlusion/result_CN.md，原始artifacts/initial_contact_fix_20260919/palm_occlusion_probe。临时重放/tmp/kcg_palm_occlusion_probe.py和/tmp/kcg_palm_circle_regression.py。

smooth_band_full_03/run（efb0599）已2026-09-19T17:21:42.334289Z结束，2776.934s，exit2，session96449已关闭。完成初抓/两次G2/搬运/插入；视觉粗调-20.297327°，二次余差+0.002257°。随后Body松手的单调减力阶段第168203步停止：f1关节反力矩1.437649Nm>原0.9Nm。此前1641步逐渐减力至约0.06Nm、手目标无跳变/重夹，Body下沉约0.108mm与旧成功松手约0.111mm相同；突发帧f2与Body后缘附近接触出现0.3238Ns总冲量和轴向法线，引起Body再下沉0.518mm。不是掌心、对键或螺母视觉门失败，尚未抓螺母。原始诊断artifacts/initial_contact_fix_20260919/body_release_diagnosis。

新主假设：+2mm避开螺纹的抓取位置太靠近后端倒角，松手时微小下沉使指尖边缘接近倒角。下一候选只把hand相对Body高度从-0.45443714改为-0.45393714，即从+2mm回到+1.5mm；静态CAD/原Nail运动学预测三指首次接触z=-0.028296/-0.028452/-0.028281m仍在平直带内，同时增加后端倒角间隙0.5mm。配置visual_body_balanced_band.yaml，其余物理/控制/保护不改。依据reproducibility/initial_contact_fix_20260919/body_release/candidate_basis.json。当前无主实验运行，下一步独立预检balanced_band_preflight_01，通过后balanced_band_full_01完整回合；启动器/tmp/kcg_launch_balanced.py。该候选未验证，不能声称解决。


用户明确要求直到完整成功验收，不能以本次局部视觉修复或失败报告结束。全过程完成后原source_nail_body、high_global1、source_nut_pad、two_key、camera、transport、native_support、source_key_containment、最终3秒、红带图像与几何、whole均须通过。可用/tmp/kcg_high_g1_postreview.py new_run（terminal存在且已结束才用）；再看同回合实际终态图像，写visibility和whole。不得拼接或用真值补成功。

仅仿真hardware_authorized=false，一次一主实验，控制源码配置冻结，停止用STOP_REQUEST，不在线延预算/改物体位姿/使用对象真值控制。原低位完整成功和全部失败证据保留，不推送，无子代理。Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python。OPENBLAS_NUM_THREADS=1，PYTHONPATH含src/kcg_connector、isaac、carts_v2。恢复核对真实进程。
