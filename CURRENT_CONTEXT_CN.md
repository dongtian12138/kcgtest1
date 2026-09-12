# 当前任务：第一版视觉全链路装配

核验：2026-09-12 10:51 UTC。simulation-only，hardware_authorized=false。本轮仅完成源码/证据/版本/存储审计及详细计划，没有新物理实验、没有删除或迁移原数据、没有创建提交或分支。

## 用户最新验收与工作边界

唯一目标是：**本回合视觉识别桌面插头 → 原三指手用指甲抓本体并离桌 → 视觉识别本体键与插座键槽 → 搬运、对键插入 → 自然支承后真实换抓螺母 → 原臂旋拧及必要换抓 → 装配到位、松手保持。** 必须同一物理回合，不以已预插入初态、局部PASS、退出码或拼接片段代替。

用户明确“能否导纳”只是问题，并未强制必须采用导纳。控制方法服从上述链路，不把二阶导纳、导纳扰动试验、电路仿真或参数辨识另立为阻挡第一版的目标。保留有限驱动、原几何/质量/惯量/限位和实际安全边界；15N指—螺母法向上限已取消，16N提砝码与PWM600/750不等于通用接触容量或80%力矩。

详细计划：[FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md](/home/noob/WorkPlace/kcgtest1/docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md)。用户本轮要求先找到路线并列详细顺序；下次明确续作应从计划第0/1步执行，不自动沿旧摘要只做螺母或导纳。

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
