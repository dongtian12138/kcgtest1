# 当前任务：继续修复，直至高位方案完整通过验收

核验时间：2026-09-19T15:08:26.112198+00:00。用户明确要求“继续解决问题，直到成功完成验收”。必须持续推进，不以局部通过/失败报告结束。平直圆环前段smooth_band_grasp_01已结束，474.483s、exit2，session84529关闭，无主物理实验运行。初次接触未出现尖峰，最大手指反力矩0.514992Nm<0.9，但原半秒预紧等待未取得6个连续稳定采样，未抬升。

工作树/home/noob/WorkPlace/kcgtest1-performance-20260917，分支codex/high-global1-full-20260919；原项目只同步本入口。仅仿真hardware_authorized=false，无子代理授权，不推送，不删除原数据。一主物理实验，加载源码配置冻结，停止用STOP_REQUEST，不在线延预算。

## 有效验收和已保留基线

验收保持：高位G1同帧两件粗定位、三原指甲抓Body抬升≥50mm保持2s、两段搬运、同一固定G2两次看键中间视觉粗转、掌心5DOF主键记忆、腕部插座/槽、插入、旋紧到位、最终3秒完全松手、原源几何/接触/受力门；主视角+四相机录像只显示名称。原低位G1完整成功socket_plus20_full_01/df1afac及源版本保留。

螺母视觉采样误停修复a308389已完成：原随机采样点距离改成静态CAD三角面距离（Warp CPU BVH）。16帧原质量门全过，局部local_fourth_grasp_02实际误差0.021783°<0.5°，夹持/原接触区/键槽几何后评通过。源码当前6b6c15f含该修复；没有改物理几何、力/速、5°修正上界等。报告reproducibility/nut_phase_surface_20260919/result_CN.md。该修复保留。

高位full03/8959158曾完成初抓到前三段旋紧，停于第四次螺母视觉门。full04/98ff9b2已在初抓停止（466.521s），f2j1峰2.493Nm>原0.9Nm，未抬升未执行新螺母算法。原0.09rad/s降速未根治接触冲击。所有旧session（43043、11300、38425）已关闭。

## 当前主要假设与最小候选

已对照同为0.09rad/s的full03/04原始初触：成功f2首次接触相对闭合527步，关节反力矩0.149Nm；失败531步初触即2.493Nm并伴大轴向冲量。两轮此前Body位置差约1微米、姿态和手指位置差很小；f2均先触，不能归结为第一指先推物体。

接触点落在Body局部z约-0.027m的螺纹段，源码外表面在该段法线含显著轴向分量。逐高度/周向72射线对照原CAD，在z约-0.0298..-0.0281m有连续平直圆环，半径17.729mm。现假设是原指尖落在螺纹/斜面交界引起卡挤/接触敏感性；不能宣称唯一数值根因。选择只将hand相对视觉Body的z目标减2mm（世界抓取位置上移约2mm），几何/材料/质量/碰撞表示/960Hz64/4/0.9Nm及其他门全部不变，接近速度仍0.09rad/s。

候选reproducibility/initial_contact_fix_20260919/visual_body_smooth_band.yaml。唯一物理变量object_from_hand_row_major[11]: -0.45243714 -> -0.45443714。原计划及候选原Nail/运动学/全周向外包络预测对照在candidate_basis.json，三候选接触z=-0.028796/-0.028952/-0.028781m均在平直圆环内。只是几何筛选，必须实际验证。原始诊断artifacts/initial_contact_fix_20260919，脚本/tmp/kcg_initial_contact_compare.py、kcg_contact_order_compare.py、kcg_body_contact_surface_check.py、kcg_body_profile.py、kcg_predict_smooth_grasp.py。原Body网格数据/tmp/kcg_body_exterior.npz。

## 当前最早阻塞与紧接步骤

平直圆环前段原生接触完成，三指末帧均仍接触Body，最大反力矩0.514992Nm，原2.49Nm冲击未重现。结束原因PRELOAD_CONTACT_EFFORT_NOT_REACHED。按同一个FingerRootMomentObserver重放原tare/编码器/反力：目标[0.489391,0.432955,0.407838]Nm，最后误差[-0.010095,-0.006301,-0.007232]Nm，原容差0.01Nm且需6连样，本回合最多2连样，0.5s截止时尚未稳定。报告artifacts/initial_contact_fix_20260919/preload_settle_review.json和history。

下一候选visual_body_smooth_band_settle.yaml保留同一+2mm高度，只新增prelift_effort_settle_timeout_s=1.5（原0.5）；源controller.py只为该预紧检查读取独立等待时限，默认仍原值，允许有限≤2s。目标力、0.01容差、6连样、0.9瞬时停止与控制器都不变。新的预检smooth_band_preflight_02（300s/30s）、前段smooth_band_grasp_02（900s/90s）随后冻结执行。启动脚本/tmp/kcg_launch_smooth_settle.py preflight或grasp，原脚本/tmp/kcg_launch_smooth_body.py保留。旧前段和数据不覆盖。

前段只有初抓/抬升/保持，通过原实际接触/离桌/2秒等验收后，必须继续完整同回合至最终3秒松手成功及五名称录像；用户明确要求直到成功，不以局部报告结束。完整21600s/300s、原高位assembly_high_global1_plus20.yaml不变。若失败先找最早原因，不能无依据加力或放宽判据。

完整结束后可调用/tmp/kcg_high_g1_postreview.py new_run（只有terminal完整时），原source_nail_body、high_global1、source_nut_pad、two_key、camera、transport、native_support、source_key_containment、3秒、source_band；必须再看本回合实际终态图像，写visibility和whole，全部通过才生成成功结论。后评关闭cyclic GC不改变数据。已有数据仅结束后读真值；禁止真值在线控制、启动后改物体位姿、隐藏固定或增力制造成功。

Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python。ISAAC_ENV_PREFIX同Isaac环境，OPENBLAS_NUM_THREADS=1，PYTHONPATH含src/kcg_connector、isaac、carts_v2。恢复先核验实际进程；旧预检session17870及前段84529均已关闭。
