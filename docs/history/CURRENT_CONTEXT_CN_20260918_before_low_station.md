# 当前任务：四相机、一次键观测与真实闭环装配改进

核验时间：2026-09-18T11:02:06.948885+00:00。用户已同意六项改进，任务仍在实施。full_chain_02是唯一主物理回合；完整新版装配尚未验证。

## 授权、验收与工作区

- 用户要求：①固定四功能相机（G1粗定位两件；G2只看一次插头底部键；掌心位置/轴线5DOF；腕部插座/键槽精定位）；②抓起→固定G2观察位→短路线至插座上方；③单次主键箭头由编码器传播、掌心修正平移/倾斜，不虚构轴向角测量；④视觉/规划结果按模拟时间延迟后消费，等待期间物理保持；⑤保护/换抓停止与在线到位分离；⑥核对旧1mm/+1°早停并做少量扰动验证。用户已授权，无需再次询问。
- 主实现树 `/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支 `codex/four-camera-single-key-20260918`，当前提交 97da4b1d56cd58bc542bb9451a80a5df7c39506e。原 `/home/noob/WorkPlace/kcgtest1` 脏资产不碰，仅同步本上下文。未推送。
- 仅仿真，hardware_authorized=false。一次只运行一个主物理实验；已结束数据的离线分析可并行。当前会话未授权子代理，不创建代理。
- 物理基线CPU960Hz64/4、原contact-last、CAD/SDF/质量/材料/原力速界保留；只沿用无损记录优化，不恢复已回退降频/GPU/contactOffset等数值性能候选。
- 完整验收保留14.605mm±10µm、源止挡正载、原2µm键槽穿透界、NAIL/PAD接触身份、连续3s全张手零原始手冲量、同128簧套逐帧正载、Body/Nut不休眠、备用限位不承载。程序退出/单元测试/局部阶段不是整机成功；真值只作结束后评价。
- 原完整成功：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/run`，295421步、15909.99s。原5倍整轮提速目标没有达到，不把搬运距离缩短误写成整轮5倍。

## 当前运行（恢复先核对process.json与实际进程）

- centered_preload_preflight_01已结束：194.715s、exit0、preflight/engine/identity通过。centered_preload_grasp_01已结束：715.293s，实际抬升56.00785mm/保持2s，3NAIL全程Body保持接触、source_nail_body_review accepted=true；原prelift±.01与6连续帧通过，其他物理/引擎/身份通过。当前唯一主物理回合 `artifacts/four_camera_20260918/full_chain_02`，bounded PID2927297，shell session51124，2026-09-18T10:29:59.177875Z开始，source97da4b1，21600s/360s收尾；完整4相机/晚看键/插入/松手/换抓/旋拧/终态候选。
- 当前新增的单一控制变量：Body抓持调节死区0.01→0.005N·m。目标[.489391,.432955,.407838]N·m、原±.01验收带、连续6帧、.5s检查时窗、全部力速/预紧行程/物理边界不变。
- 新抓取配置 `reproducibility/four_camera_20260918/visual_body_centered_preload.yaml` 是原visual_assembly_v1_body.yaml完整副本，仅dynamic.finger_effort_control_deadband_nm=.005一项新增；已结构化比较确认其余完全相同。旧原配置保留。
- controller._run_preload_lift_hold与held-path共用可选deadband，缺省仍原.01。新deadband显式记录在registered_grasp.finite_clamp_target。已语法核对；正在物理验证，不宣称修复成功。
- 预检通过后已启动**仅抓取抬升保持**的centered_preload_grasp_01：从full_chain_01 argv移除postgrasp-key-observation/body-assembly-transport/body-key-entry/body-support-test/body-nut-regrasp，保留visual-body-start、同IK seed、4相机assembly配置；--config改新抓取配置，预检指向当前新预检。明确900s/120s收尾。bounded_experiment已给该无postkey/无transport的visual-grasp-only分支900s上限。不要直接重跑整机碰运气。
- 当前动态源码冻结；只做已结束数据分析或独立工具准备。停止用run/STOP_REQUEST，勿SIGINT。

## 新观察顺序已验证，后续整机首次尝试抓取早停

- `near_socket_key_station_01`已结束：3efdabc，1444.85s，107771步，exit2仍旧评价标签；前缀到插座上方，原指甲接触/抬升/保持通过，source_nail_body_review accepted=true。
- 固定新G2 eye[.445,.345,.291]/target[.515,.185,.322]在**大搬运/大转向后**看一次键。前段只用明确“未测键向”的运输坐标系；随后G2+同刻掌心初始化真实箭头，腕部识别5槽，最后约35.8mm归中。两台外部相机始终固定。
- 到达后主键轴向误差0.041475°（最大0.045315°），中心20.14µm、轴0.000411°；相比早看键版本末轴向0.555°已改善。腕部测得Socket yaw约0.12273°，实际Body末yaw约0.169°。这是单个仿真前缀，不是已插入或硬件保证。
- camera contract审计通过：实际感知路径恰好4个，G2=1次，掌心230次延迟消费；固定T_HC误差数值约1e-16。报告位于run/four_camera_transport_posthoc.json、four_camera_contract_review.json、source_nail_body_review.json。
- 44961ad已接入完整后续：te_body_key_entry用掌心5DOF+单次箭头代替外部键重观测；_run_light_contact_key_search可实时取得relation并更新载荷COM；原低力律/保护不改。Body卸载前retire；RecordedStepper每步服务延迟掌心，Nut阶段只用不超过0.5s龄的已消费帧。完整post-entry支持/换抓/旋拧沿原流程。在线到位与原物理验收分开。
- full_chain_preflight_01通过（196.24s，exit0、三个门全true）。full_chain_01（PID2870150、session26656）**已结束失败**：2026-09-18T10:01:17.146903Z，481.08s，exit2，在初始prelift_effort_check停于PRELOAD_CONTACT_EFFORT_NOT_REACHED；尚未抬升、未启用four-camera session/后续代码。
- 已只读重放两回合机器人关节力信号和原FingerRootMomentObserver：成功前缀在368帧检查后获得原6连续帧；失败回合480帧只获得最多3连续帧。失败末50帧三指平均误差[-.008389,-.006833,-.008937]N·m，贴近-.01验收边缘；最后单帧[-.012245,-.006344,-.012758]。原控制对50ms过滤力使用±.01死区，而抬升对原始补偿力也要求±.01，存在同宽边缘停调问题。
- 诊断资料 `artifacts/four_camera_20260918/preload_diagnostic/`：extract.py（Isaac Python读结束原始数据）、review.py（规划.venv重放原机器人侧观察器），preload_settle_review.json。没有从真值修正控制。当前候选缩窄调节死区，验收/目标不放宽，待物理检查。
- 当前15项相关软件行为已检查；单个新测试放置错误已更正后3项session测试通过。覆盖延迟/初始化一次/掌心横向基不覆写键/松手不随手/到位判断，但不能代替动态验证。

## 上一轮的关键实测结果与路线修正

- `short_transport_01`已结束：accf8e2，1445.12s，exit2仍为旧整体评价标签；lift/hold/controller_nominal_physical/engine/identity/truth隔离通过。初始26s视觉+规划等待实际推进物理，完整前缀共106553步。
- 实际主搬运0.400737m，加局部归中0.035233m，合计0.435970m；旧主搬运3.47418m。原速度上限保留，实际峰值0.11498rad/s。
- G1同帧定位两件，G2恰好1次键，掌心225次延迟更新，腕部1次看到全部5槽；已到插座上方。source_nail_body_review accepted=true，原NAIL/Body接触区域保持。
- 但单次键在搬运前建立的方案**未通过角向可用性**：末中心误差17.49µm、轴0.000368°，比冻结记忆81.04µm/0.4393°明显改善；主键仍有0.5552°轴向误差，最大0.5589°。
- 初次G2本身旋转误差仅0.02786°，不是主误差来源。轴向误差随大搬运从约0.064°→0.399°(半程)→0.509°(终点)，归中后0.555°。因此**不能声称抓Body就不轴转，也不能用5DOF掩盖这一误差或从真值补偿**。
- 上一轮腕部槽yaw后验约+0.207°，Socket中心约[.550007,.184991,.271622]。固定腕部视角可用，但精度不是硬件保证；完整插入未运行。原小键/槽理想居中半角余量约0.398°，还需计横偏/倾斜，故未盲目启动插入。
- 已向用户明确说明上述实测优点与失败，并说明把大的转向安排在唯一键观测之前；新观测站在插座附近，剩余运动短。用户已有授权覆盖布置与两段搬运实现，无需再确认。
- 报告位于short_transport_01/run/four_camera_transport_posthoc.json、single_key_anchor_posthoc.json、source_nail_body_review.json；原影像/失败/原始数据全保留。session66353、28633、55788均已收尾。额外接触存在性复核shell session8701可能待收尾，纯结束数据分析，不是物理运行。

## 已验证结果与当前实现

- 13d777c：固定四相机rig、G2/掌心同状态取图前缀与2.4–2.5s因果物理保持。
- a9d762c：KeyDirectionMemory、工作区筛选、可选current-seed IK。KeyMemory使用手坐标下最小轴旋转搬运主键箭头，忽略掌心横向基；Body松手必须retire。
- accf8e2：short_body_motion沿本体直线+姿态插值，以原Dogbox/原软界求连续逆解；four_camera_body_transport已接入实际运行；four_camera_perception_session固定掌心/腕部、延迟队列、5DOF更新；Global1同帧socket工作区筛选+SAM6D后台job。初始视觉/规划延迟、搬运规划延迟真实保持。
- `grasp_visibility_02`523.45s：原姿态真实抬升56.11mm/保持2s/3指甲Body接触通过；固定G2键角误差0.10825°，掌心中心20.13µm/轴0.0000224°。
- `alternate_arm_grasp_01`521.50s：仅换初始arm IK支路（手—本体抓位不变），实际抬升55.98mm/保持2s/三NAIL Body全程接触/无桌接触，source_nail_body_review accepted=true；固定G2键误差0.010568°，2.51458模拟秒延迟保持通过。工具41959复核已完成。
- 新支路CLI必须保留：`--approach-high-seed-arm-positions-rad 0 0 0 -1.5707963267948966 0 1.5707963267948966 2.5`，预检和运行均同参。旧支路末端接近关节软限位，直线规划失败；新支路在同一抓位离线成功且实际抓取已验证。
- 原长弧本体路程3.47418m，直线距离0.40159m，最高升至1.493m。新支路离线253空间样本通过，第一次实际短搬运已完成；角向记忆问题见上。
- 当前9项键记忆/延迟/工作区测试通过；只证明对应软件行为。

## 本轮后续代码与复核工具

- f2d99fd的`src/kcg_connector/isaac/four_camera_online_completion.py`已在44961ad接入完整续段；5项针对性测试：高力但未到深度、重复同一张图、宣称3s但实际不足、偏心均不报完成。按现有视觉14.605mm±20µm/稳定5µm、腕扭矩≥0.4Nm、实际张手保持步数≥3s及轴/横向范围判断；物理原10µm等验收独立。
- 已在原结束成功回合允许传感器上回放：松手前后视觉深度14.604372/14.603995mm、相差0.377µm、腕扭矩1.316Nm、实际开放保持3s，得到在线到位；报告`completion_sensor_replay.json`。不是新的四相机装配成功。
- `reproducibility/four_camera_20260918/review_transport.py`：已评short_transport_01，待新回合结束后再运行，比较初始冻结记忆/延迟5DOF更新误差、实际路径与掌心单帧精度，导出30Hz轨迹。仅结束后真值评价；已增加忽略key初始化前掌心事件，适配新观察时机。
- `artifacts/four_camera_20260918/contact_integration_candidate/prepare*.py`是已应用候选的历史准备脚本；源码已在44961ad接入并进一步修正，**不要重新应用旧候选覆盖现代码**。
- 候选思路：原te_body_key_entry增加可选session，不再调用额外外部键相机；掌心更新后仍保留原3次有限纠偏/原力速旅行界。_run_light_contact_key_search添加body_relation_getter，仅按实时5DOF更新Body估计和载荷COM。
- 候选还从原流程提取four_camera_post_entry支持/换抓/旋拧续段；Body卸载前retire；RecordedStepper每步服务延迟掌心（只采样/排队，不额外推进物理）；observe_current_plug走已消费的掌心帧（最多0.5s龄）而非新建相机。已接入并通过相关软件检查，整机第一次尝试在更早的原抓取阶段失败，后续动态尚未验证。
- 候选prepare_session依赖session.last_observation，关键是不要用同一帧重复冒充两次确认。上述entry/continuation、在线终态、终端载荷记录和四相机视觉评价合同均已接入，完整动态验证仍待抓取前置通过。

## 其它已完成离线诊断

- `old_carry_ideal_five_dof_axial_slip.json`：在原已结束长弧回合48612帧中，用事后真值分解“若5DOF完全准确”的剩余误差，max轴向主键误差0.20265°、末0.20201°。仅误差分解，绝未返回控制；不能据此断言新短路径也相同。已告知用户本体夹持的额外轴转动需要实测验证。
- 原CAD小键宽4.02158°、槽宽4.817476°，居中理想角向半余量约0.397948°，不是之前误猜的1.07°；实际还需计横偏/倾斜/径向边界，保留原物理评价。
- 旧X+1mm/yaw+1°确实在首次Body接触后三步触发f1j3=-4rad/s>3原界，四杆残差突增，无Nut载荷/目标跳变。支持接触建立/四杆求解耦合瞬态，唯一主因未知。详见reproducibility/improvements_20260916/evidence/pose_variation_early_abort；可用原提出0.18→0.09rad/s局部合拢诊断区分，不升保护线，不盲扫。

## 下一步

1. 跟进当前full_chain_02，恢复先核对process.json/进程/阶段，不凭文档等待。若阶段失败先诊断最早原因，保持所有原安全/物理验收，不盲重跑。
2. 新抓取调节通过后使用已接入的完整四相机方案再跑一回合；保留原21600s有限预算，准备全部原物理/视觉/终态复核。
3. 必要源码预检后执行一次整机回归（若更早阶段失败就先诊断），再做有意义的小扰动验证。原始失败证据保留，未完成部分如实记录。

## 环境

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同目录；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM Python `/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`，SAM根 `/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`。启动argv/env参考各回合process.json和本轮shell命令。
离线规划LD_LIBRARY_PATH含原.venv/lib/python3.10/site-packages/tesseract_robotics及Isaac环境lib，AMENT_PREFIX_PATH=本树install/iiwa_description:/opt/ros/humble，OPENBLAS_NUM_THREADS=1。
测试PYTEST_DISABLE_PLUGIN_AUTOLOAD=1，sys.path加src/kcg_connector、src/kcg_connector/isaac及carts_v2。普通rg --files可能被.gitignore过滤，必要时--no-ignore限定目录。

新增结束后复核工具：reproducibility/four_camera_20260918/review_terminal_release.py逐帧验证真实末3s零手冲量/原止挡+同128簧套每帧正载/BodyNut醒着/原深度及备用限位；在原成功repeat02重放得到全通过，输出另存current artifacts/terminal_three_second_reference_replay.json，未改原基线。完整验收需与原source_key/nail/pad/visibility/whole_assembly审查同时成立。当前新整机未运行到终态。

搬运图已从旧成功与near_socket_key_station_01原始轨迹生成：artifacts/four_camera_20260918/measured_transport_comparison.png/json，旧主段3.4746m，新两段约0.436m，约7.96倍路程比；不是整轮提速。两幅图仍仅是搬运证据。绘图脚本在同目录，tool session86194可能待收尾（纯离线）。

新deadband抓取诊断：centered_preload_grasp_01在443检查帧后达原6连续帧，最后误差[-.005231,-.004145,+.003062]Nm；末50帧均值[-.008708,-.002649,-.003686]Nm，F1仍较靠近下边缘，不可声称所有条件已鲁棒。见preload_diagnostic/preload_settle_review_centered_preload_grasp_01.json。检查deadline仍原0.5s、全部力速/行程/目标未改。当前整机将检验重复性；若同类再失败，用该收敛证据定位，不自动扫描死区或增力。

当前full_chain_02最新：初始抓取控制与原预紧6帧门再次通过，抬升/保持控制段完成；已完成大搬运、固定G2一次键和腕部槽识别。两次有界对准后在线估计中心残差8.449µm/完整姿态残差0.003307°，通过原50µm/0.1°门。正在下探前的key_probe_entry_planning_hold（约136模拟秒），对应原逐物理采样避障计算的因果保持；尚未接触插入。下探路径检查无碰撞，规划执行时长44.06354模拟秒；不要把在线估计残差说成已完成的真实几何验收。未读取本回合对象真值用于判断控制。

后续复核review_transport.py已将统计起点改为key真实availability_step，并只重放key_update非空的掌心消费事件，避免把初始化前/不可用期的5DOF记录当成已被键闭环使用；Body松手retire后停止键记忆评价。新增visual_body_centered_preload_variation.yaml/variation_plan.json只定义±1mm/±1°两个后续小扰动，尚未执行，当前主回合仍只一个。
