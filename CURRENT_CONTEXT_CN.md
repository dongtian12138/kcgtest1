# 当前任务：继续修复，直至高位方案完整通过验收

核验时间：2026-09-19T15:08:26.112198+00:00。用户明确要求“继续解决问题，直到成功完成验收”。必须持续推进，不以局部通过/失败报告结束。无主实验运行，准备平直圆环抓取候选预检。

工作树/home/noob/WorkPlace/kcgtest1-performance-20260917，分支codex/high-global1-full-20260919；原项目只同步本入口。仅仿真hardware_authorized=false，无子代理授权，不推送，不删除原数据。一主物理实验，加载源码配置冻结，停止用STOP_REQUEST，不在线延预算。

## 有效验收和已保留基线

验收保持：高位G1同帧两件粗定位、三原指甲抓Body抬升≥50mm保持2s、两段搬运、同一固定G2两次看键中间视觉粗转、掌心5DOF主键记忆、腕部插座/槽、插入、旋紧到位、最终3秒完全松手、原源几何/接触/受力门；主视角+四相机录像只显示名称。原低位G1完整成功socket_plus20_full_01/df1afac及源版本保留。

螺母视觉采样误停修复a308389已完成：原随机采样点距离改成静态CAD三角面距离（Warp CPU BVH）。16帧原质量门全过，局部local_fourth_grasp_02实际误差0.021783°<0.5°，夹持/原接触区/键槽几何后评通过。源码当前6b6c15f含该修复；没有改物理几何、力/速、5°修正上界等。报告reproducibility/nut_phase_surface_20260919/result_CN.md。该修复保留。

高位full03/8959158曾完成初抓到前三段旋紧，停于第四次螺母视觉门。full04/98ff9b2已在初抓停止（466.521s），f2j1峰2.493Nm>原0.9Nm，未抬升未执行新螺母算法。原0.09rad/s降速未根治接触冲击。所有旧session（43043、11300、38425）已关闭。

## 当前主要假设与最小候选

已对照同为0.09rad/s的full03/04原始初触：成功f2首次接触相对闭合527步，关节反力矩0.149Nm；失败531步初触即2.493Nm并伴大轴向冲量。两轮此前Body位置差约1微米、姿态和手指位置差很小；f2均先触，不能归结为第一指先推物体。

接触点落在Body局部z约-0.027m的螺纹段，源码外表面在该段法线含显著轴向分量。逐高度/周向72射线对照原CAD，在z约-0.0298..-0.0281m有连续平直圆环，半径17.729mm。现假设是原指尖落在螺纹/斜面交界引起卡挤/接触敏感性；不能宣称唯一数值根因。选择只将hand相对视觉Body的z目标减2mm（世界抓取位置上移约2mm），几何/材料/质量/碰撞表示/960Hz64/4/0.9Nm及其他门全部不变，接近速度仍0.09rad/s。

候选reproducibility/initial_contact_fix_20260919/visual_body_smooth_band.yaml。唯一物理变量object_from_hand_row_major[11]: -0.45243714 -> -0.45443714。原计划及候选原Nail/运动学/全周向外包络预测对照在candidate_basis.json，三候选接触z=-0.028796/-0.028952/-0.028781m均在平直圆环内。只是几何筛选，必须实际验证。原始诊断artifacts/initial_contact_fix_20260919，脚本/tmp/kcg_initial_contact_compare.py、kcg_contact_order_compare.py、kcg_body_contact_surface_check.py、kcg_body_profile.py、kcg_predict_smooth_grasp.py。原Body网格数据/tmp/kcg_body_exterior.npz。

## 紧接步骤

冻结候选后，用原高位assembly_high_global1_plus20.yaml做新预检（300s/30s）及仅初抓抬升保持前段（900s/90s）；与旧full04相比只改抓取高度。验证实际源Nail/Body接触、抬升2秒保持、无Nut/桌违规接触、力速未触发。通过后完整新回合（21600s/300s），不得仅凭前段或程序退出宣布完成。失败先查最早机制再选可区分的下一步，不盲扫。

原run/grasp入口run_body_assembly_with_video.py，参考high_global1_full_04与preflight04的process.json argv。前段只保留--visual-body-start，删掉--postgrasp-key-observation、--body-assembly-transport、--body-key-entry、--body-support-test、--body-nut-regrasp；Body换新候选，输出和preflight均新目录。录像5fps原names-only，不用label_five_view.py加额外字。

完整结束后可调用/tmp/kcg_high_g1_postreview.py new_run（只有terminal完整时），原source_nail_body、high_global1、source_nut_pad、two_key、camera、transport、native_support、source_key_containment、3秒、source_band；必须再看本回合实际终态图像，写visibility和whole，全部通过才生成成功结论。后评关闭cyclic GC不改变数据。已有数据仅结束后读真值；禁止真值在线控制、启动后改物体位姿、隐藏固定或增力制造成功。

Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python。ISAAC_ENV_PREFIX同Isaac环境，OPENBLAS_NUM_THREADS=1，PYTHONPATH含src/kcg_connector、isaac、carts_v2。恢复先核验实际进程。
