# 当前任务：第一版视觉全链路装配

核验：2026-09-12 13:09 UTC。用户已明确授权按详细计划自主执行、可据证据修改安排并持续反馈。唯一目标仍为同一回合完成Isaac Sim视觉取件到完整装配；不在局部阶段结束任务。simulation-only，hardware_authorized=false。

## 用户最新验收与工作边界

唯一目标是：**本回合视觉识别桌面插头 → 原三指手用指甲抓本体并离桌 → 视觉识别本体键与插座键槽 → 搬运、对键插入 → 自然支承后真实换抓螺母 → 原臂旋拧及必要换抓 → 装配到位、松手保持。** 必须同一物理回合，不以已预插入初态、局部PASS、退出码或拼接片段代替。

用户明确“能否导纳”只是问题，并未强制必须采用导纳。控制方法服从上述链路，不把二阶导纳、导纳扰动试验、电路仿真或参数辨识另立为阻挡第一版的目标。保留有限驱动、原几何/质量/惯量/限位和实际安全边界；15N指—螺母法向上限已取消，16N提砝码与PWM600/750不等于通用接触容量或80%力矩。

详细计划：[FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md](/home/noob/WorkPlace/kcgtest1/docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md)。用户已批准执行该计划；当前已完成新版同回合视觉指甲取件、56 mm抬升/2 s保持及抓后识键；正在继续搬运、对键插入和自然支承验证，不在局部阶段结束。

旧“修复J599旋拧后仿真偏斜”任务01a088b0-0c88-76e0-ac61-1fc9e31682e7已停止交接，最近app核验completed/notLoaded，不自行恢复。本任务为唯一实施方；不主动创建子代理。硬件与推送未授权。

## 已核实的路线和缺口

正式入口候选：`src/kcg_connector/isaac/run_body_assembly_with_video.py`包装`carts_v2/run_grasp_lift.py`；复用已有视觉、Body抓取、搬运/重新观察、入槽/支承、Nut换抓/旋拧/续拧模块，在同一runtime、stepper和场景内连接。不要另写一套大型框架。

已定位缺口：
1. `vision-grasp-servo`分支完成后直接返回，未接后面的本体观察和装配；旧`grasp-lift`可进入装配，但名义抓取不能自动算视觉引导抓取。
2. 新四杆/自锁/电机PD仅在`diagnose_saved_hand_wrench.py`接入，`JointSignalStepper`仍直接驱动旧手目标。需统一给所有阶段使用。
3. 旧Body抓位70°，新Nut证据60°。当前局部掌面自锁是启动角零宽限位，不能在流程中主动70→60换位。需实现真实共同电机驱动与到位保持；若源几何证明同一布局可满足两种抓取，也可据证据简化，不能直接改锁定角跳过运动。
4. `te_nail_tip_body_grasp_v1.yaml`历史说明允许指腹/指甲任意抓Body，不足以证明满足当前指甲承载要求。需从源几何定义分阶段NAIL—Body与PAD—Nut接触；其他碰撞不能关闭。
5. 已找到的视觉抓取physical_rx180_run12…19均未离桌，因腕力矩停止。视觉工具可复用，但当前新模型的初始视觉指甲抓取必须重新取得证据。
6. 局部入口读取旧truth_samples和传感器快照恢复初态；LocalInterfaceFollowing还从9月7日旋拧结果加载settings。需提取带来源正式配置，不能作为最终全链在线初态来源。
7. 局部诊断用物体真值作位移越界中止；这不是机器人传感器闭环，正式链中须与控制/阶段决策分离，不能照搬“online truth=false”标志作为全程序保证。

## 当前执行进展与最早阻塞

R=`artifacts/visual_assembly_v1`。当前分支`codex/visual-assembly-v1`，先前基线提交`c928aa7`。无关用户修改、删除标记和原始数据保留。没有推送、硬件动作或删除旧artifacts。旧源码/工作区快照在R/checkpoints/20260912T112851Z。

**最近完成的有效阶段：`visual_body_grasp04`已实际完成本回合RGB-D定位→三指甲抓Body→抬升约56.02mm→保持2s→抓后新图像识别本体键。** 当前手机构、连接器、原指甲和有限驱动均保持。独立`source_nail_body_review.json`确认保持每一步三根指甲均接触Body；三指所有正载触点投影到原STL指甲shell，无PAD/安装柱/其他物体承载，源面投影最大约.291mm（当前烘焙表示近似，不冒称零几何误差）。保持阶段无桌面接触，最低抬升56.0199mm，手—Body相对变化约1.30µm/.00349°。抬升过程抓内变化约.351mm/2.197°，因此仍保留搬运后重观察。

04抓后识键误差仅用于独立评价：位置.02968mm，轴角约1.36e-5°，主键角.013509°。初始视觉仅用普通RGB-D、冻结背景/CAD与桌面支承先验，明确没有测键yaw；抓后才测键。初始226个离散检查姿态覆盖17link、所有Body yaw包络、桌面/夹具/插座搜索体积；没有读真实物体位姿来生成动作。

04程序外部900s超时发生在物理动作及独立评价已完成后的最后汇总阶段，退出124，**没有最终evaluation.json，不能伪造它为正常结束**。已保存完整35389步压缩truth、手传动记录、FT、RGB-D、成品mp4和日志中的NAIL_BODY_STABILITY；`physical_stage_review.json`明确记录从这些独立证据恢复阶段结论及超时限制。视频R/visual_body_grasp04/video/assembly_four_view.mp4已给用户打开。下一回合不再附加该不合适的总wall timeout。

已解决的前置故障：01/02按旧Cube桌面时手未靠近插头已漂移约8mm并摇动，下降提前碰到Body；源四杆闭环杆长误差<.3µm，原开手几何下降最小间隙约4.58mm，故没有盲改开手或提高腕门限。03仅换同尺寸、同材质、同变换的12三角面桌面，整个13.03s最大漂移约.007mm，空手接近无碰撞通过；04同样稳定。使用现有`visual_assembly_v1_table_mesh_trial.yaml`保留03物理配置身份，待整合时正式收敛到canonical task配置。不要误切回未启用该选项的旧表面。

已接入：
- `te_hand_mechanism_runtime.py`将4个电机和非线性四杆包在同一world.step前后，所有共同JointSignalStepper名义目标均由它驱动；初始化后不写实际q/qd或物体位姿。掌面保留全行程和1:1两个支路，已替换局部零宽限位；70→60主动换位仍待换抓时实证。
- `te_visual_body_start.py`在主`grasp-lift`流程的FTtare/桌面停稳后采图、生成抓取计划并继续原后段。`--visual-body-start`不能初始化到预抓/载入旧视觉。当前视觉轨迹有独立几何检查，03只作为名义物理比较，accepted_preflight_bound=false，不冒充旧preflight验证了新图像。
- Controller的mimic诊断已由q2−q1改为q2−f(q1)和实际局部导数速度误差。Body夹紧算法/目标保持原设定；原电机K264/D4.4+输出弹性120/黏性2，自锁c1.2等仍开发参考。Body主动cap1Nm，指传动边界3.5Nm、掌面1Nm。
- 指甲源面定义见`visual_assembly_v1_contact_regions.json`：[11836,12912)指甲shell，后续两安装柱不算NAIL；独立评价入口`evaluate_visual_body_grasp.py`只运行后读取实际体姿态/触点。
- LocalInterfaceFollowing已改为读正式`visual_assembly_v1_rotation_reference.json`中提取的旧settings，不再依赖9月7日大型结果作为执行输入。
- 后续Nut接口已修正两个确定性旧假设：开手60°时关闭目标不再携带Body70°；对于新约5°绕轴偏转的canonical抓位，先把平移转成Hand局部量再按当次观察轴/当前手横向基生成目标，不再要求canonical旋转为单位阵。这些仅逻辑/语法检查，尚未动态执行。

**当前唯一主要回合：`visual_to_key_entry05`**，exec session78879。已授权运行至自然支承/释放后观察，命令含visual-body-start、postgrasp-key-observation、body-assembly-transport、body-key-entry、body-support-test；尚未开启Nut复抓/旋拧，因为前段还未在新模型实证。恢复先核对实际进程/文件，不能凭本段“运行中”等待。日志R/visual_to_key_entry05.log，无外部wall timeout。若前段通过就接后续；失败按最早原因修复，不能拿预插入局部回合替代。

05新增的记录改动：`carts_v2/sample_store.py`将原始truth按512步普通gzip成员保存并以2块缓存读取，完整原始数据和随机访问保留；无损读取/切片/有界缓存/标准gzip兼容检查通过。仍是同一个truth_samples.jsonl.gz，未抽样、未删观测。原实现04物理37s内存从约12GB涨至19GB，完整装配有撑满62GiB风险，故在长流程前修正。录像包装异常退出也会关闭原始归档；初始化、抓取完成、录像关闭前先保存元数据/结果。旧pickup evaluator仅评价第一次抓起到hold的前缀，后续释放/换抓不能按Body-grasp规则误判；完整机械验收仍待全链评价。

下一步先核对05初始化/新的记录接口，继续实际搬运及视觉对鍵插入，检查新版自然支承。Nut阶段仍需统一3.5Nm主动cap/新根力矩目标及姿态相关重力扣除、选好最后松手间隙，并接真实分段旋拧到止挡/卸手保持。当前canonical task里的Nut旋拧仍有旧0.04Nm扭矩试探门等历史参数，不能直接冒称采用最新已验证局部方案。完整装配尚未发生。

## 有效资产与证据边界

M=`artifacts/kcg_connector/te_connector_contact_repaired_20260911`：connector_model.usdc、install_model.py、validation_manifest.json为有效连接器。CPU/TGS960Hz、64位置/4速度迭代、每迭代外力、至少32768接触记录；安装在第一次reset/张量读者创建前，保留原质量和位姿，不安装验收转台。模型已自包含USD且无未解析依赖。单体完整旋紧/旋开、键槽、止挡、簧套、保持与错误接触过滤证据可复用，未测参数边界仍保留。

正式机器人资产：`artifacts/kcg_connector/isaac/te_nail_tip_body_grasp_v1/handarm_original_nails_source_decomposition.usda`，另外依赖10个USD层，主要位于旧robot/handarm_keyed_v3_physical_r7目录，不能按日期整目录删。源手STL/URDF未改。

A=`artifacts/hand_mechanism_audit_20260912`。SW包完整，原件及提取在A/source/1。手指外部传动240:1、掌面每支66⅔:1；不代表额定或效率。四杆10–70–10–68mm，约0–80°累计角差f1 .471°/f2 .751°/f3 .472°，.876是局部导数。共享FK/Jacobian/速度和约束候选已有测试和单指实际运动证据；物理参数仍是开发参考，不是硬件辨识。

手指应变片在底座双桥，标定量为绕O外载力矩M=F*l。正反5N单指检查支持当前根力矩语义；无需ADC电路仿真。WormDrive的passive_split＋电机PD在局部整机whole_hand_worm_grip_trial25完成预插入起点5.7s抓持/保持/松手：正载接触均Nut原PAD，保持相对变化约0.025mm/0.054°，末0.5s手接触0。**这不是指甲抓Body、视觉取件或旋拧通过。** 该试验约1–3s更新手指力反馈目标，随后冻结；最终打开是预设轨迹，不能当导纳扰动退让验证。

旧Body证据：isaac/te_body_assembly_20260905/grasp_assembly_01抬升约56.27mm、保持2s；key_entry_02有当前图像对键并实际入槽证据。均旧模型条件，且初始抓取不能自动算视觉指甲抓取。旧transport_03抓内记忆漂移约1.2mm/6.3°，因此保留搬运后重新观察。

旧无指甲G/segmented_full_mating_release_trial33仅作分段流程对照；旧有指甲N/successful60_full_mating03虽至约353.64°/14.604856mm，但第三指碰插座5.649N、松手第二指回碰.752N且手机构旧，未验收。G=artifacts/kcg_connector/grasp_comparison_20260911；N=artifacts/kcg_connector/nail_present_grasp_20260912。

## 实施顺序

0. 固定审阅后的代码/配置/资产清单与可复现环境；新日志止增，旧压缩不成为前置大工程。
1. 将新机构及电机驱动放到正式公共步进器，解决主动掌面换位与分阶段接触面，衔接视觉抓取返回和后续装配。
2. 新图像定位→指甲抓Body→离桌/保持；失败就在此修复，不跳过验收第一段。
3. 抓后键位识别→搬运→本体重观察＋插座槽观测。
4. 当前图像对键→有界低速插入→确认自然支承。
5. 真实松Body/改布局/重观察→抓Nut，并预查完整轴向范围及最后松手的手—插座间隙。
6. 短段实际旋拧→必要分段/换抓→继续推进；主要未知是有限驱动下的后段扭矩传递与干涉，不凭腕角/力升高算成功。
7. 在线可用信号判断终止、实际松手保持；真值只作运行后完整机械验收。
8. 同一新回合从视觉起点贯通全部链路，冻结首次成功版本及原始证据。随机统计、泛化、硬件和论文均后置。

## 版本和存储核验

盘点开始时HEAD68b51f8（9月10日），分支codex/connector-assembly-recovery-20260909；16个tracked修改、2个tracked删除、113个untracked文件。本轮又新增计划/索引文档。新四杆、自锁和较新旋拧控制文件有未跟踪项，HEAD不能代表当前工作区。

artifacts实占596259520512字节≈555.31GiB、71940文件；JSON/JSONL逻辑大小≈439.24GiB。旧大trace和jsonl可能重复但未逐项证明；无损压缩归档/引用解除后才能按具体清单处理。磁盘约998GiB可用，不是当前执行硬阻塞。原数据、失败证据和未提交用户资产保留；当前只做了盘点。

审计文件：`docs/assembly_v1_route_manifest_20260912.json`（27项主组件、hash、git状态）；`assembly_v1_storage_version_inventory_20260912.json`（文件/空间/版本）；`assembly_v1_usd_dependencies_20260912.json`（USD依赖）。全部Python/配置的条件依赖与视觉模型环境版本尚未封闭，不冒称发布包已完成。

前一完整工作状态保存在`docs/history/CURRENT_CONTEXT_CN_20260912_before_full_visual_route_plan.md`；其他历史继续只作追溯，不恢复旧参数/局部目标为当前验收。
