# 当前任务：四相机、一次键观测、短搬运改进已完成名义整机验证

核验时间：2026-09-18T18:56:58.952809+00:00。名义整机完整验收通过；两组初抓扰动完成，+方向预载停止、-方向实际抬升保持通过。准备一次1s等待诊断。

## 授权和不变边界

用户“可以按照你说的做”授权：四功能相机；抓后→固定G2观察位→插座上方两段短搬运；一次主键箭头加掌心位置/轴线5DOF反馈；视觉/规划结果延迟对应真实物理保持；到位与保护/换抓停止分开；核对旧+1mm/+1°初抓早停并做少量扰动。

- 仅仿真，hardware_authorized=false。一次只运行一个主物理回合；已结束数据可以独立离线审查。没有子代理授权，不生成代理。
- 主树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/four-camera-single-key-20260918`，实现提交`84bef70924b7244f47493a75c38a17fade6c3b58`。原`/home/noob/WorkPlace/kcgtest1`除同步本文件外未改其脏资产，未推送。
- 原CPU960Hz、64/4、TGS/contact-last、CAD/SDF、质量/质心/惯量、材料、被动关节、力/速/行程限制不变；只保留已有无损记录优化，不恢复已退回数值加速候选。
- 完整验收：14.605mm±10µm、源键顶点/槽面残差≤2µm、原接触区域、连续3s严格零原始手冲量、源止挡和同128簧套逐帧正载、Body/Nut醒着、备用轴限<.5999mm且不承载。真值只在运行后评价，绝不进入在线控制。
- 运行期间冻结加载源码；用`run/STOP_REQUEST`受控停止，不SIGINT、不在线延长预算。原始失败和成功数据不删。

## 唯一当前物理回合：恢复先核对实际状态

目前没有主物理回合运行。variation_minus_grasp_01已于2026-09-18T18:52:39.146998Z结束，711.787s，source84bef70；原力门通过，真实抬升55.9769mm、2s保持、三原NAIL/Body全程接触、保持桌面零载通过，source_nail_body_review accepted=true。session96363已结束。minus控制力重放仅是离线工具session56587，可能待收尾。

+1mm/+1°前缀variation_plus_grasp_01已结束：2026-09-18T18:25:46.836303Z、496.825s、40746帧，PRELOAD_CONTACT_EFFORT_NOT_REACHED，未抬升。旧首次接触后三步f1j3=-4rad/s>3的超速未重现，但不能称初抓成功。重放**控制实际使用的重力补偿指根力矩**：480帧/.5s检查最多5连续合格、要求6，末F1误差-.0110755Nm超±.01，末50帧均值[-.0068604,-.0032048,-.0036744]Nm。通用evaluation的末帧减零偏数值“带内”不是控制器的补偿信号，已向用户更正；不据此放宽门。提取与重放脚本/原机器人信号在artifacts/four_camera_20260918/preload_diagnostic，精简结果已复制reproducibility相应evidence。

下一步：已完成minus同配置检查。随后已向用户说明做**一次有界稳定等待时间诊断**，区分收敛时间不足和稳态仍不稳定；候选visual_body_settle_1s_diagnostic.yaml只将dynamic.contact_endpoint_timeout_s从.5→1.0s，保留6连续帧、±.01Nm验收、原目标力/速度/行程/保护，已创建但尚未执行，不盲扫。contact_endpoint_timeout_s还供接触协调器使用，若用此字段需核对早期接触是否本来就在.5s内完成；新配置须预检。名义完整成功配置仍原.5s，不在未验证前替换。可以把诊断独立于主基线保存，不把初抓前缀当成扰动下完整装配。

旧超速证据保留在reproducibility/improvements_20260916/evidence/pose_variation_early_abort，具体唯一机制仍不确定。原提出合拢.18→.09rad/s未执行，也不是当前这一预载不足故障的自动修复路线。

## 已通过的完整名义回合

`artifacts/four_camera_20260918/full_chain_04/run`，source84bef70。2026-09-18T15:28:37.959009Z至17:59:51.662992Z，9073.704s（2h31m14s），273285物理步，进程exit2为旧通用评价标签。**whole_assembly_review.status=VERIFIED / complete_visual_assembly_verified=true；three_second_release_review.accepted=true。** 所有已启动后评工具均已结束，没有其它主物理回合。

- 实际插头两段搬运：0.3970812+0.0603910=0.4574722m；手基坐标点0.4431966m；搬运峰值arm.120988rad/s<原.15。
- 功能感知路径恰好4条；G2键1次、掌心1109帧延迟消费；固定手眼外参和Global1同帧定位两件审查通过。本体卸载前retire，手回转不再传播Body坐标关系。
- 接触开始的主键轴向误差.0421348°；整个抓持生命周期最大.2726773°、中心最大21.627µm、轴.009570°。5DOF仍不观测额外轴转；不要把最大值说成首次入槽前误差。
- 最终3秒[270405,273285)共2880帧：原始手冲量严格零、源止挡每帧正载、同128簧套每帧正载、Body/Nut醒着。实际深度14.603955mm恒定。全回合最大备用轴坐标.522465mm<.5999mm。
- 源键最大残差1.175528µm<2µm；五键全部入槽。初抓三原NAIL/Body通过；Nut原区域通过，本轮f1/f3为NAIL、f2为PAD，无其它受载对象。原接触身份没有替换。
- 原红带径向1024/1024样本被Nu原始网格遮挡，已实际查看同回合旋拧末/释放中/最终视频帧，无露出红带。渲染没有推进物理或改变原生状态，图像未修改。
- 在线到位True：独立视觉深度14.603779/14.603582mm，实际张手3s，末次加载扭矩2.003314Nm。第四次在累计356.248994°因弹性裕量换抓，第五次再转1.708548°后到位候选及最终释放；原360°上界未放宽。
- 真实规划等待已动态验证：初次Nu转移8.628619s/8284物理步，首次回转2.246249s/2157步，Nu相位规划与验证亦有对应保持。原保护没有豁免。
- 引擎健康、身份、有限性、真值隔离、controller_nominal_physical_pass均true，PhysX容量警告0。旧通用全指腹/旧tensor通道报告有不适用的失败标签，原报告保留；不能说所有JSON都通过。
- 完整报告`reproducibility/four_camera_20260918/full_assembly_result_CN.md`；精简证据`reproducibility/four_camera_20260918/evidence/full_chain_04/`。原始raw msgpack.gz16342211301bytes，录像44762666bytes，仍在原位。

旧整机参考`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/run`耗时15909.99s；当前进程时间比约1.7534倍，**未达到5倍整轮提速**。独立后验检查另计。旧主搬运约3.4746m，约7.6倍路程比不能当作整轮速度比。当前只是一轮名义完整仿真成功。

## 当前实现与必要复核入口

G1固定粗定位两件：同初始RGBD帧SAM分割，Socket用上口沿几何估计中心/轴线，明确不测键yaw。G2固定eye[.378,.319,.244]/target[.490,.185,.275]；先做大搬运/转向，3mm低位、Socket侧距60mm处只看一次键，再短归中。掌心和腕部T_HC固定。录像旁观视角不进入控制。

KeyDirectionMemory在手系用最小轴旋转传播主键箭头，忽略圆拟合任意XY基，编码器传播手运动；掌心名义.2s采样，.05s传感延迟+实际估计时间后消费，Body主动卸载前retire。硬件延迟未标定。规划/感知等候通过原保持控制推进真实物理步。

原Body目标/±.01Nm验收/6连续帧/.5s检查窗口不变，仅`visual_body_centered_preload.yaml`的调节死区为.005Nm，避免过滤控制在验收边缘停调。不是旧超速故障的已证明机制修复。

`te_visual_seating_axis`可选分段分支已修复重复缓存帧确认，等待独立已消费帧，0.5s无新帧则停；当前config没有启用该分支，只做了软件检查，不称动态覆盖。实际运行是原coaxial控制与掌心Nu相位抓取。相关16项检查通过：延迟6、视觉分段7、session3；其它此前KeyMemory/到位负例检查保留。

后评仅对结束/封存数据：
- `evaluate_visual_body_grasp.py RUN`：三NAIL初抓；扰动前缀用它配合evaluation的原保护/引擎等字段。初抓前缀可能没有motion_timing，不伪造结束标记。
- `evaluate_source_nut_pad.py RUN --geometry-plan artifacts/grasp_capacity_20260914/selected_two_nail_geometry.json`：必须按本次原区域合同，默认全PAD不适用两甲一腹。
- `evaluate_source_key_containment.py`、`review_terminal_release.py`、`review_native_body_support.py`、`audit_camera_records.py`、`review_transport.py`。
- 到位红带：`reproducibility/improvements_20260916/postreview/review_source_band_occlusion.py`加实际视频帧查看。`evaluate_rear_face_visibility.py`是合成遮挡消融，**不是**到位红带检查。
- `evaluate_visual_assembly_v1.py`最后汇总（旧2s项之外必须另有严格3s通过）。本轮完整汇总已经通过，不要反复重扫。

## 历史及环境

详细旧过程已移至`docs/history/CURRENT_CONTEXT_CN_20260918_before_full_acceptance.md`和此前历史；最重要保留失败：早看键short_transport_01末额外轴误差.555°；高50mm看键后full_chain_02长下降又积累约.4°并停在.777mm；低站01键掠视、02 G1学习姿态错误、03腕部口沿遮挡，04低位/60mm侧距真实入槽支持成功。full_chain_03在发现后段漏计规划等待后STOP_REQUEST受控结束748.44s，不是物理失败。旧脚本contact_integration_candidate/prepare已过时，不再应用。

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同环境；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM Python `/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`；SAM根`/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`；OPENBLAS_NUM_THREADS=1。
离线规划LD_LIBRARY_PATH含原.venv的tesseract_robotics及Isaac lib，AMENT_PREFIX_PATH本树install/iiwa_description:/opt/ros/humble。测试禁用pytest自动插件，sys.path加src/kcg_connector、isaac、carts_v2。rg必要时--no-ignore限定目录。读取运行状态可用`artifacts/four_camera_20260918/latest_allowed_sensor_status.py RUN`，只读控制及已消费相机，不读运行中对象真值。
