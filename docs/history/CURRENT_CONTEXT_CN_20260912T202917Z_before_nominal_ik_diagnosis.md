# 当前任务：第一版视觉全链路装配

核验：2026-09-12 15:39 UTC。用户已明确授权按详细计划自主执行、可据证据修改安排并持续反馈。唯一目标仍为同一回合完成Isaac Sim视觉取件到完整装配；不在局部阶段结束任务。simulation-only，hardware_authorized=false。

## 用户最新验收与工作边界

唯一目标是：**本回合视觉识别桌面插头 → 原三指手用指甲抓本体并离桌 → 视觉识别本体键与插座键槽 → 搬运、对键插入 → 自然支承后真实换抓螺母 → 原臂旋拧及必要换抓 → 装配到位、松手保持。** 必须同一物理回合，不以已预插入初态、局部PASS、退出码或拼接片段代替。

用户明确“能否导纳”只是问题，并未强制必须采用导纳。控制方法服从上述链路，不把二阶导纳、导纳扰动试验、电路仿真或参数辨识另立为阻挡第一版的目标。保留有限驱动、原几何/质量/惯量/限位和实际安全边界；15N指—螺母法向上限已取消，16N提砝码与PWM600/750不等于通用接触容量或80%力矩。

详细计划：[FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md](/home/noob/WorkPlace/kcgtest1/docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md)。用户已批准执行该计划；当前已完成新版同回合视觉指甲取件、56 mm抬升/2 s保持及抓后识键；正在继续搬运、对键插入和自然支承验证，不在局部阶段结束。

旧“修复J599旋拧后仿真偏斜”任务01a088b0-0c88-76e0-ac61-1fc9e31682e7已停止交接，最近app核验completed/notLoaded，不自行恢复。本任务为唯一实施方；不主动创建子代理。硬件与推送未授权。

## 已核实的路线和缺口

正式入口候选：`src/kcg_connector/isaac/run_body_assembly_with_video.py`包装`carts_v2/run_grasp_lift.py`；复用已有视觉、Body抓取、搬运/重新观察、入槽/支承、Nut换抓/旋拧/续拧模块，在同一runtime、stepper和场景内连接。不要另写一套大型框架。

当前已接通视觉取件、共享机构、搬运和对键的正式路径；最初断点已移入历史。尚需解决：
1. 新模型无接触探入已到旧6mm深度上限，12mm总行程修改待验证。
2. 新Nut抓位已解决抽样末段几何干涉，但局部握持的横向合力导致导向摩擦承重、螺纹未充分承载；正在做有界XY力反馈对照。
3. 同一视觉回合中的主动掌面70→60、正式根力矩夹紧/保持、连续旋拧及安全松手仍待贯通。
4. 到位必须由实际原止挡、接触和松手保持独立验收；预插入局部试验和转角完成都不能替代。

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

**05已结束，当前最早全链阻塞是探入行程不足。** `visual_to_key_entry05`完成新视觉取件、抓后识键、48.69s原臂搬运、搬运后重观察、侧方插座键槽观察及两次图像对准；图像估计的最终对准误差约2.86µm/.0124°。低速探入30s后触发`AXIAL_TRAVEL_LIMIT`，并主动退回3mm前间隙；没有执行松Body或Nut换抓。总180903步，simulation时间188.44s；进程已退出2，最终evaluation、视频、完整压缩truth/FT均保存，identity_hash_check_pass与engine_health_pass都为true。旧evaluation中的controller_completed仅指初始抓起，旧transport.completed也曾错误保留carry成功，不能当全链完成；新代码已修正key-entry失败传播。

`probe_travel_boundary_review.json`独立确认：最大/最后实际Body深度5.9954mm、横向43.79µm、轴倾.1954°、键前端重叠5.17mm、最小槽角余量.1018°、径向余量.1698mm；整个探入段插头—插座正载接触0。旧9mm行程包含起始3mm间隙，因此只能到6mm。已有新模型局部试验初始支承深度约7.963mm。下一正式配置已将有限行程改为12mm（最多约9mm深），0.20N参考/.3mm每秒及其他试探限制不变；该改动待动态验证。

**末段几何阻塞已有候选修正，未改原物理几何。** 使用最新受载手指编码器和当前4bar/绑定烘焙凸块，对要求的旋转及松手姿态做离线CAD检查，原Nut抓位在到位深度第三指碰安装法兰；`loaded_nut_end_clearance_review.json`记录132/1656采样姿态碰撞。上移2.3mm可清除受载干涉，但在最不利轴向位置松手20–30%时仍有32处碰撞。上移3.5mm候选`R/nut_grasp_raise_3p5mm/geometry_plan.json`保持60°掌面、相同三指首次接触角，均为原PAD，Body最小间隙约3.755mm；完整抽样的±.5mm轴向位置均无碰撞，最小手—Socket间隙.7425mm（负轴端）/.9707mm（正轴端）。这是离散几何检查，不是连续或动态到位保证；动态腕偏移还需计入。

**06局部20°试验已完成，但不算螺纹推进通过。** `raised_nut_twist06`使用上述抬高抓位、正式`HandMechanismRuntime`四个电机（包含完整行程、非零宽限位的主动掌面）和已验证根力矩候选。22512步/23.45s，无传动失败，实际Nut约19.675°、Body约.375°；所有记录的手接触对象只有三末节—Nut，按每8物理步保存的正载触点全为源PAD（最大投影残差8.32µm），最终.5s所有手接触负载0。独立`local_physical_review.json`保存这些边界。

06揭示实际推进问题：转动期间Nut仅前进.1724mm，Body反而后退.0201mm；末段握持时Nut—Socket法向接触为0，松手后Body/Nut又自然下降约.252mm才形成Nut—Socket承载。腕部横向合力峰13.21N、末段约7.14N，轴向约.597N已接近自重参考.613N。主要假设是偏心夹持造成导向侧摩擦承重，使自重参考满足而螺母尚未充分入扣；不能把转角完成叫装配推进成功。

**07已完成，准备下一次正式全链执行。** `raised_nut_xy_follow07`仅启用现有有界XY力反馈（.15mm/s、.5mm范围），其他参数同06；无中止，实际Nut转19.643°，转动期间Body后退.0290mm、Nut前进.2230mm，Nut—Body轴向相对位移约.252mm。横向力峰由13.21N降至2.908N、末段由约7.14N降至.872N，实际腕偏移最大.286mm；握持末段Nut—Socket法向承载约2.862N，松手后约.663N。几何/短转允许继续检查间隙消除后的推进，但不能称Body装配推进已通过。结果见07/local_physical_review.json。两个局部试验都不是视觉验收回合。

**08完整候选已停止，未完成装配。** 已在同一回合完成新视觉取件/指甲抓Body、搬运/两端识键、实际探入7.91mm、卸指后的自然支承、主动掌面70→60、抬高3.5mm的NuT复抓、首段20°、松Nut/空手回腕/再次复抓。第二个90°段在elapsed21.8135s（仅转动指令约3°）因`bounded planar force-admittance travel exhausted`停止；尝试偏移[.4957,-.0663]mm越过.5mm边界，滤后横向力[.7876,-.3915]N。不是整项成功，未到位、未最终松手。

08物理控制停止后进程退出137。已找到收尾旧函数`_evaluate_key_entry_after_motion`把全部raw行建成by_step字典，绕过了磁盘缓存、可能耗尽内存。改成有序流合并，并在任何运行后评价前封存最后raw块。08保留269824条raw（最后step269823），相对控制结束270165缺341步尾段；各阶段FT/命令和结果均已保存，不能补造丢失真值。`interrupted_postprocessing_review.json`记录边界，尚未取得OS日志直接证明kill原因。

恢复后的独立评价已证明08实际键入和自然支承：`recovered_physical_key_entry_result.json`实际7.91056mm、五键入槽、手—Socket正载0，最大Body—Socket穿透约.394µm；流式评价峰RAM约290MB。`body_support`评价实际末7.965875mm，最后.5s全部手指接触0、深度范围0，五键前端始终在槽内。`source_key_containment_review.json`全过程已保存范围的源键侧壁最小残差约−.910µm，处于原交付模型数值范围。`actual_turn_endpoint_review.json`：首20°实际Nut19.4795°、Body−.0116°、手19.7109°；Body后退.1468mm、Nut前进.0912mm，Nut在手中相对转动约.584°。第二段已保存前缀：Nut2.3775°、Body.6787°、手3.098°，Body/Nut均略后退，尾段缺失明确标注。

**当前最早控制阻塞是扭转期间持续XY力积分耗尽有限偏移。** 原模式没有恢复刚度，离线固定信号回放加入2000N/m恢复项可把最大位移从约.50mm降到.21mm，但这不是新物理成功。现已建立`te_source_stage_probe.py`，复用正式`run_body_nut_rotation`、同一个`HandMechanismRuntime`及当前RGB-D/机器人信号，从已结束08第二次复抓前的源step249224冷初始化，仅作诊断，不当完整视觉验收。

诊断入口保留原空载零点（源FT行已保存run_specific_tare字段）及四个输入端角度/速度，在第一次测量前声明；之后不写实际关节/物体位姿。初始化时发现并修复了重复重力补偿：旧初始化映射把原生已加重力偏置的drive目标写入了nominal位置字段；局部Stepper现恢复源nominal参考。修复后预热手腕变化约8µm横向/39µm轴向、最大关节变化约4.96e-5rad。09/10/11仅为入口调试：FT空阶段合同、缺少照明导致黑RGB、供应观测source标签合同等问题均保留日志，未把它们算控制物理结论。

`source_stage_restoring12`已结束：采用恢复刚度2000N/m仍在elapsed23.6979s耗尽.5mm范围，滤后力[1.214,-.873]N；该候选未解决阻塞，不增加行程或再次增刚度。

**当前唯一主要物理回合为`source_stage_xy_hold13`，exec session15242。** 恢复时先检查真实进程/文件。新对照相对08仅改变`freeze_after_preparation=true`：先按原视觉/力反馈完成对准准备，之后固定该横向参考，旋拧期间仅保留轴向力跟随和原有受力监测；恢复刚度回到原0。源码实际复用正式控制器，90°轮廓、0.5mm范围、速度、力参考和电机上限保持。原回合状态是诊断初态，成功仍不能当视觉全链完成。

Canonical已包含待验证的12mm探入、抬高3.5mm的60°Nut抓位，以及现有20°首段＋90°分段＋最终释放；横向参数仍是08的原已冻结配置，等13结果再决定正式修正。NuT几何在`config/visual_assembly_v1_nut_geometry.json`，与原源面/计算记录绑定。

`diagnose_saved_hand_wrench.py`新增`--shared-hand-mechanism`以使用同一四电机模型；不再给这两个局部试验使用零宽掌面锁。局部入口仍有真值位移护栏和历史快照初始化，只允许诊断，正式路径不依赖它。06首次命令因本工具wall限制最大600s被参数校验拒绝、未启动物理；该日志独立保留。修正为600s内部循环限制后正常完成，没有外部强杀计时器。

**当前源码/配置未全部提交。** 最近提交`2f8f436`保存探入/Nut候选和输出修正；`744cdd1`保存共享机构＋视觉取件，`c928aa7`为更早检查点。之后仅新增独立运行后评价脚本，活动控制源码未修改。当前相关内容包括：
- Canonical`visual_assembly_v1_task.yaml`采用已通过的同形三角桌面、12mm探入、抬高3.5mm的60°Nut抓位，以及待验证的正式根力矩夹紧（1.7571/1.25/1.2535Nm，1.5s渐升/2s总时长，120Nm/rad换算、1/6s修正时间尺度、3.5Nm有限电机）。夹紧后冻结目标。未来20°首段＋90°分段＋最终松手，尚未实际贯通。
- 新的`run_body_nut_regrasp`根力矩分支已写好，使用底座力矩定义及姿态重力差；只在Nut阶段将旧.9Nm观测参考改为record_only，驱动仍有限。Nut受载阶段观测20N/.4Nm/4.6Nm取自已用局部参考和新模型记录，旧.04Nm源于已淘汰.02Nm阻力，不再作为新版拧紧门。均不是厂家手/传感器额定。实际正式夹紧/旋拧尚未运行。
- 原Nut接口旋转5°/掌面70→60错误已修正并提交；受载力矩新分支仍待验证。根据07结果再决定正式XY力反馈范围/启用时刻，不要盲加轴向力。
- `compare_nominal_scene_scope`允许canonical配置路径及后段控制设置更新时复用03的名义物理比较；严格核对collision、physics_numerics、passive_joint_solver、grounding_band_contact_model、validated_connector及所有其他绑定。新视觉轨迹依旧独立检查，accepted_preflight_bound=false；不伪造旧preflight覆盖新任务。
- 正式初始/抓后观测现在只保存RGB-D/编码器及对应步号，真值误差比较统一移到所有动作结束后从原始归档读取。旧04/05保存方式不回改。
- `trace_metadata.py`添加兼容普通/压缩truth的流读取及逐行gzip JSON数组写入；已有support/full-key评价器已适配压缩流。`GzipSampleStore`控制原始流内存已在05验证，取件轨迹与04完全重现。但05旧FT整体转换/汇总曾临时升至约49GB、随后回落；现改为逐条JSON序列化，避免整体深拷贝。旧evaluation内约399MB的大数组将单独无损压缩保存、汇总只留索引；既有05两个399MB文件保留。
- 保存后处理微基准发现GC扫描使读取2400条约.86s，对比关闭循环GC约.51s；只在运行后JSON树分析中临时关闭循环GC，不改变物理步或运行中GC。格式兼容/无损小检查已通过，相关源码语法通过，下一回合将验证长流程输出。

08当前进展：12mm有限行程已解决旧边界，probe以STABLE_AXIAL_CONTACT_REQUIRES_POSTRUN_EVALUATION结束；Body卸力/开手/保持控制完成，松手后的当前RGB-D估计Body深度约7.966mm，尚须运行后接触复核。随后进入60°掌面与抬高3.5mm Nut抓位，主动电机参考已连续切换为掌面1Nm、三指3.5Nm，未重置输入状态。下一步继续08同一回合，检查真实换抓、旋合和最终松手。新的运行后评价脚本检查原始NAIL/PAD接触、源键侧壁、原止挡、128簧套与松手保持；目前不影响在线控制。08的execution_source_manifest.json绑定了当前提交和实际控制文件。不要把局部诊断的初态、真值护栏或姿态读回接入正式控制；完整装配、终止/松手保持、主动70→60转换仍未取得同一视觉回合证据。

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
