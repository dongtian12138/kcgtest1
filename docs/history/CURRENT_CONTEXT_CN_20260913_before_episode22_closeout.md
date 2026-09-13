# 当前任务：Isaac Sim第一版视觉全链路装配

核验：2026-09-13T03:49:34.825315+00:00。用户已授权执行`docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md`，可按证据修订并持续反馈。任务未完成，不在局部旋拧成功处结束。simulation-only，hardware_authorized=false。

## 用户最新停止安排

用户于本轮执行中要求：**当前这一轮仿真结束后停止工作并做好收尾，供用户检查进度。** 让当前22自然完成或按现有停止条件结束；之后只做本轮数据封存、独立结果核对、必要版本/状态记录和交付，不启动23或其他新动态实验，不继续试参数或功能修改。未完成装配也按该安排停下，不能把早先“持续到全链完成”的授权覆盖用户此次停止指令。等待用户检查后的新指示。

## 唯一验收边界

同一物理回合：当前桌面RGB-D识别 → 原三指甲抓Body、离桌和保持 → 当前本体键与插座槽视觉 → 搬运/重观察/对键 → 实际插入与自然支承 → 真实松Body、掌面换位、抓Nut → 原臂旋拧和必要换抓 → 原机械到位 → 手脱离并保持至少2s。

预插入初态、跨回合拼接、命令转角完成、PASS或程序退出均不等于完成。在线只用当前图像、编码器、腕部与底座应变桥对应信号；对象/接触真值只作后评估。禁止隐藏固定、磁吸、启动后改物体/关节实际状态、无限增力、盲扫。导纳不是额外验收要求。

旧任务“修复J599旋拧后仿真偏斜”01a088b0-0c88-76e0-ac61-1fc9e31682e7已停止，本任务为唯一实施方；不自行恢复旧任务，不主动派子代理。只有一个主要物理试验。无硬件、推送、删除原始artifacts授权。

## 已实际完成的最远阶段

R=`artifacts/visual_assembly_v1`。

**08 `visual_complete_candidate08`为完整流程的有效比较基线。** 同一回合已完成新视觉指甲取件、约56mm抬升、搬运、两端识键、实际插入7.91056mm、完全卸指后的自然支承、主动掌面70→60、抬高3.5mm的Nut抓位、首20°、松Nut/回腕/再次复抓。第二个90°段在elapsed21.8135s、仅转动约3°时耗尽0.5mm横向力反馈行程；未到位、未最终松手，不能验收。

08独立结果：
- `recovered_physical_key_entry_result.json`：真实深度7.91056mm，五键入槽，手—插座正载0，Body—Socket最大穿透约0.394µm。
- Body自然支承：最终深度7.965875mm，最后0.5s全部手接触0、深度范围0、五键前端在槽内。
- `source_key_containment_review.json`：已保存范围五键源侧壁最小残差约−0.910µm，在原模型1µm数值参考内。
- `actual_turn_endpoint_review.json`：首段实际Nut19.4795°、Body−0.0116°、手19.7109°；Body后退0.1468mm、Nut前进0.0912mm。第二段已保存前缀Nut2.3775°、Body0.6787°，均略后退。

08控制停机后退出137；旧后处理重建全raw字典可能耗尽内存，但未取得OS直接证明。已保存269824行至step269823，相对控制结束step270165缺341步尾段。各阶段FT/命令/结果完整，不能补造缺失真值。见`interrupted_postprocessing_review.json`。

04 `visual_body_grasp04`独立证明当前原指甲取件：保持每步三指甲承载Body、无桌面和非NAIL承载、最低抬升56.0199mm；2s保持相对变化1.30µm/0.00349°。抬升过程中抓内变动0.351mm/2.197°，故仍保留搬运后重新观察。抓后主键视觉误差0.013509°仅作后评价。04外部900s超时发生在物理和后评估结束后、最终汇总前，不能冒充正常evaluation；以后无该外部总超时。

05完成全前缀但旧9mm总探入包含3mm空隙，实际只到5.9954mm，探入段无插座接触。canonical改12mm总行程后已在08验证解决。0.20N/0.3mm/s没增加。

## 当前最早阻塞与唯一在跑试验

**目前阻塞在第二段旋拧的受载运动控制，尚无连续推进到位证据。** 不重建手或连接器。

12 `source_stage_restoring12`：相对08加入XY恢复刚度2000N/m，仍在23.6979s耗尽0.5mm，候选无效，不继续增刚度或行程。

13 `source_stage_xy_hold13`：相对08只在视觉/力准备结束后固定XY参考（偏移范数0.05844mm），恢复刚度回0；越过旧行程停止，29.5375s、命令9.37°时避碰停止，Fxy约10.25N，未完成90°。

**13最早原因已区分：实际第一指距原Body约4.44mm，距0.8mm保守包络约3.54mm；当前FK和原生位置相差小于0.2µm，手机构不是这次停止原因。名义臂位置与实际受载位置差最多0.00221rad，导致第一指的名义位置相对实际偏出约3.4mm，名义姿态进入Body避碰包络。** 见`body_bound_reference_review.json`和`target_body_bound_review.json`。原旋拧微分IK按受载编码器误差不断积分名义目标，积累对抗接触的目标偏移。不能删除避碰检查来绕过。

**14 `source_stage_nominal_ik14`已结束，未完成90°。** 以名义指令姿态积分IK后，原9.4°停止点已越过；同4.2°指令处横向力从13的5.05N降到1.92N。但16.50°指令又在“名义臂+实测手型”的Body包络检查停止。后验记录：实际手转16.56°、Nut8.18°、Body0.545°；Body后退0.434mm，Nut后退0.245mm，说明抓内滑移明显，不只是显示问题。Body估计横向偏差2.684mm、轴向0.688mm；原生Body处名义第一指距0.8mm包络仍有2.07mm，而错误地跟随手的虚拟Body发生碰撞。实际FK仍与原生姿态一致。见14/target_body_bound_review.json及local_motion_review.json。全过程35,485条记录完整，进程已退出0但控制未完成，不能以退出码称成功。

**15 `source_stage_xy_follow15`已结束，仍未完成。** 在名义姿态IK基础上恢复原有XY跟随，elapsed22.4552s、命令约3.15°时用尽0.5mm；滤后Fxy[0.603,-0.312]N，峰侧力约1.008N。实际手3.155°、Nut2.341°、Body0.644°，Body/Nut均退约0.115mm；Body估计横向误差约0.429mm。原生/名义指位对真实Body包络均仍有约3mm间隙，没有实际碰撞。完整23,477行，进程退出0但控制失败。

14离线进一步确认：准备结束实际Body倾角0.0817°（旧刚性关系估计0.00145°），转动末实际0.5176°。主要偏斜在受载旋拧中发展。当前实际PAD/Nut材料均µ0.45，combine=max（源配置嵌套历史估计1.4不代表当前材质）。14末三指正法向合量约[7.12,25.76,18.83]N，第三指传动3.470Nm逼近3.5边界。离线8接触点库仑锥、零合力/零弯矩与既有4驱动cap的静态纯扭容量：起始约0.702–0.739Nm，末约0.785–0.829Nm；限制到实测各指正常载荷时更低。只是当前接触布局的静态估计，不是全抓法不可能或硬件额定。记录在14/grasp_torque_capacity*.json与contact_material_review.json。

**16 `source_stage_grip_balance16`已被主动提前结束，不能说控制失败或完成。** 三指载荷分配已启用；至32s/12.36°指令，侧力约0.84N、模型增量[-8,2.23,5.77]N。随后发现更早的缺失轴向入扣辅助（下文），主动结束这个低轴向参考对照。SIGINT触发Isaac原生直接退出0，未执行Python最终封存：无最终probe结果/raw索引/视频汇总，保存35,328行至step35327，最后控制行35813，约0.5s尾段未封存。该有效前缀后验：实际手16.17°、Nut14.29°、Body0.643°，Body后退0.130mm、Nut前进0.131mm；相较14明显改善跟转，但不能称本体持续推进。详见local_motion_review.json、explicit_early_stop.json和interrupted_recording_review.json；不补造尾段。今后局部暂停写output/STOP_REQUEST，由控制器正常返回并封存，不再用SIGINT退出。

**新的主要机械假设：整机漏掉已验证模型的有限入扣轴向辅助。** 已结束15在转动末的三个键面正法向约6.94+8.28+1.66=16.87N，腕部净横向力却约0.68N；抵抗扭矩的力偶互相抵消，但不消除轴向摩擦。实际Body/Socket键面µ为0.45、max组合（不是内部Body/Nut传动摩擦0.2），上述法向对应摩擦锥上界约7.59N，不等于实测轴向摩擦。Nut—Socket接触间断、无持续轴向推进。只看净侧力并靠0.613N自重不足以说明能入扣。

直接对照已验收单体`artifacts/kcg_connector/contact_consistency_20260911/free_guide_32_final`：result.json明确engagement_axial_assist=true；driver_snapshot.py在couple且编码角<40°时向Nut施加额外3.0400615N轴向力，之后撤去。整机原控制没有这段参考，属于确切接口遗漏，而不是需要再次重建几何。

**17 `source_stage_engagement_assist17`已停止，未完成90°。** 相对16只补这段有界辅助，通过原机械臂和腕部力反馈实现，不给对象施加直接力。现有自重参考0.612729N加额外3.0400615N，最大参考3.652791N；沿既有0.5s曲线渐升，累计受载命令35–40°渐退。局部初始累计20°来自08先前完成的首段命令（不是Nut实际角），因此本段前20°内辅助，之后恢复自重参考；全90°轨迹仍保留。原0.3mm/s/4mm、2°/s、驱动cap、20N/0.4Nm/4.6Nm停止及原几何不变。额外3.04N是已接受模型试验输入，不是厂家额定。配置、代码、命令绑定在同名前缀文件，恢复核对实际进程。

17停止的进一步核验：不是已证实的3.5Nm机械超限，而是float32原生投影误差触发过紧容差。F3读数3.500000476837158，旧判断cap+1e-7；输入子步弹簧力矩3.492606Nm、终点弹性力矩约3.493Nm均未超限，elastic_effort_boundary_exceeded=false，仅drive_saturation=true。修正为观测比较使用8个float32 epsilon×力矩尺度（3.5Nm处约3.34e-6Nm）；不改native maxForce或原始读数，独立弹性边界保留，并单独记录达到有限上限。使用原失败前后两帧复现已通过：该舍入读数不误停，真实3.5001Nm仍拒绝，弹性过载仍拒绝；整个32项相关测试通过。

17保存34,995条truth到34994；34995的物理手传动完成后抛异常，来不及进入主auditor，因此最后这一个物理tick只有手传动记录，不能当raw完全覆盖。已保存转动前缀实际Hand15.725°、Nut10.413°，Body后退0.264mm、Nut后退0.149mm；尚无持续推进证明。进程正常结束0不代表成功。

**18 `source_stage_effort_tolerance18`已停止，数值误停修正起效，但触发独立弹性边界。** 控制和物理输入与17相同，只修正上述观测容差。保留载荷分配及入扣辅助3.04N、全部物理边界，继续检查更长旋合。命令/配方/含te_worm_drive的源码绑定均已保存；恢复核对实际状态。局部正常暂停采用output/STOP_REQUEST，不用SIGINT。

18停止事件34922：F3原生驱动力矩3.4999566078Nm未越界，修正后的drive_saturation=false；输入子步弹性力矩3.499845846Nm，物理步后K(z−q)=3.500239237Nm，elastic_effort_boundary_exceeded=true。不是上次float32读数误判，不能继续放宽读数容差处理。其余两个指传动力矩约1.373/1.787Nm，第三指承担较多扭矩。

**19 `source_stage_effort_rate19`已结束，在软余量保持条件停止。** 相对18只增加基于三指弹性力矩余量的旋拧参考时钟减速：余量低于0.5Nm开始减速，0.1Nm时保持角度；近停最多2s，总追加时间最多20s，未完成角度和保持不能报completed。力矩估计使用共享传动内部输入状态和输出关节编码器，不用对象/接触真值；原3.5Nm弹性边界和native力矩cap、2°/s最大转速、轴向输入、行程、碰撞与其他停止不变。参考加速度不再冒称原最小加加速度轮廓的数值；实际机器人仍有原速度/有限驱动力矩/FT保护。保留原腕力驱动的三指载荷分配与有限入扣预载。源码、配方、命令单独绑定；停止用output/STOP_REQUEST。该新控制尚未验证。

19在step37500、elapsed37.0625s停止：命令约14.94°，F3弹性力矩约3.399945Nm，硬3.5Nm未越界；参考速度降至近零后2s未恢复余量，触发正常的软保持超时。该结果不能证明用尽全部硬上限后仍不可能，只证明当前低初始夹紧加净腕侧力分配没有在该余量要求内恢复。角度时钟已增加约3.17s等待，不能误标完整90°。

**20 `source_stage_task_preload20`已完成整个90°控制段与末段保持，局部物理复核通过。** 更换的是抓持控制方案，不冒充单参数对照：从相同08源状态出发，经原始2s预热/当前图像/避碰准备，再用原基座桥标定语义将M目标从当前实测值渐升至[2.0855669,2.2503397,2.1457547]Nm（来源下节0.6Nm任务估算），1.5s渐升/3s总时长；位置修正仍1/6s、120Nm/rad、0.15rad/s和原位置界限。随后重新获取当前RGB-D轴线，固定这次建立的电机位置参考。关闭16的净腕侧力分配和19的软限速；保留名义姿态IK、固定准备后XY、原轴向入扣辅助3.04N、物理驱动上限3.5Nm和所有硬停止/几何。检查更均匀且与旋拧任务匹配的初始夹力能否避免F3单独过载与滑移；未修改材料或几何。新增预加载记录task_root_preload_samples.jsonl，结果和当前图像均保存。若预加载就失败，可能没有rotation目录，先读source_stage_probe_result.json及预加载/手传动日志。正常提前停止写output/STOP_REQUEST。

20独立结果：94,917条raw已封存；实际Nut87.8539°、Hand89.9796°、Body0.6393°；Body前进1.25599mm至9.392393mm，Nut前进1.51882mm。最大Nut-Body轴向坐标0.501722mm，未接近0.6mm备份限位；峰侧力1.6455N、峰扭矩0.55470Nm。三末节所有承载点均原PAD，无手部-Body/Socket/其他对象正载；PAD最大投影残差约58.63µm。局部不是完整视觉装配验收。

键槽原1µm后验带使20的1.159µm残差被标false；原结果保存在20/source_key_containment_review_initial_1um.json。用完全相同源顶点/侧壁平面指标复算已验收free_guide_32_final全旋合记录，得到1.706µm最大残差；因此把后验数值带取为2µm并保留所有原始最小值，模型几何/物理接触不变。新20/source_key_containment_review.json通过，全部94,917步五键完整轴向范围均在槽内。基准记录R/reference_full_mating_key_gap_review.json。

用20末0.5s实测手型复核末段和松手几何：正/负0.5mm位置偏差、0–355°每5°、原深度与松手采样均无干涉；最小间隙0.606815/0.714986mm。见R/task_preload20_end_clearance_plus.json与minus.json；仍是离散几何检查。

**现已将20方案接入canonical，准备新预检21后执行完整视觉回合22。** 公共NuT抓取保留原2s接触预载，再加同一1.5s渐升/3s总时长的任务预載，目标2.08557/2.25034/2.14575Nm，随后固定电机位置。两种旋拧段均采用名义姿态IK、准备后固定XY、无净腕侧力分配/软限速、前40°有限轴向辅助（整回合累计命令从0开始，不在每段重置）。原3.5Nm驱动/弹性边界、最大速度、行程及材料不变。

为避免到位时直接触发硬边界而无法松手，另设0.05Nm传动余量下的控制器试拧停止（局部20最大弹性力矩3.30223Nm，不会触发此阈值）。它只允许当前图像检查导向后按最后实际发送的目标单调卸载，不豁免硬关节/腕/碰撞/弹性故障，不宣称已经到位。原停止记录保留，最终是否装配成功由独立实际止挡/簧套/脱手保持判定。6项卸载准入检查已通过，硬故障、过期步号和错误阶段均不能进入该路径。

命令已准备：R/task_preload_preflight21_command.json、R/visual_complete_task_preload22_command.json。21为新的13s级名义预检，绑定当前代码后才能启动22；22必须从桌面新图像开始，不能继承局部状态。22现已启动。21在建场景、动作开始前退出：重建命令时多带了旧--free-split-object-manifest覆盖，令接触政策变成CouplingNut，与Body首抓冲突；failure.json保留。已移除这个多余覆盖，使用Body配置自身已注册的同一split模型。21b（task_preload_preflight21b）随后通过：preflight_pass、accepted_preflight_pass、identity_hash_check_pass、engine_health_pass均true，正常退出0。
**当前唯一主要物理回合22 `visual_complete_task_preload22`，exec session23426。** 从桌面原始初态和本回合新图像开始，首抓Body，包含全部视觉识键/搬运/插入/自然支承/两阶段任务NuT预载/旋拧续拧/可控终止卸载。没有旧快照接力，没有改材质、质量、几何或驱动上限。实际代码已提交8e92313；精确命令R/visual_complete_task_preload22_command.json，源码绑定R/visual_complete_task_preload22_source_manifest.json，视频和raw在同名目录。恢复先检查真实进程及最新阶段；不要因为入口或阶段完成就结束任务。局部诊断20不作为全链成功。

12–20使用`te_source_stage_probe.py`：从08已结束的step249224冷初始化作局部诊断，复用正式`run_body_nut_rotation`/共享机构/当前RGB-D/同一传感器，不可当完整视觉验收。保留源空载FT零点与四电机输入角/速度；初始化后不回写实际q/qd/对象姿态。09–11只是入口问题：FT阶段合同、缺照明、观测标签及重复重力补偿，不能作物理控制结论。修复后预热腕位置变动8µm横向/39µm轴向。

下一步：跟进22同一视觉回合至实际装配/松手结果；失败则按最早真实原因修复并继续；按真实螺母转动、本体推进、抓内滑移与源接触决定后续。14也证明旧手—Body刚性关系不能长时间用于避碰位置，应在需要时使用当前RGB-D更新，而不能放宽包络或读真值代替。尚未把候选提升到canonical；准备的完整回合命令仅为draft，未执行。

## 待核对材料与离线预加载估算

已只读核对原手USD：fingertip_pad静/动摩擦均1.4、combine=max；当前沿用旧friction_lower_0p45实验，通过_apply_contact_friction_perturbation将其覆盖到0.45。原值1.4本身也不是硬件标定，不因当前困难直接换高摩擦来声明成功。已用异步问题询问用户实物接触面是裸金属还是橡胶/硅胶软垫；尚无回复，当前20仍保持0.45。记录R/source_hand_material_reference.json。

已有一个现用于20受控验证的任务力矩预加载估算R/task_torque_preload_estimate.json：以原模型入扣期峰约0.592Nm取开发目标0.6Nm、µ0.45、源接触半径约23.67mm，等法向约18.78N/指，对应CAD基座桥力矩目标[2.08557,2.25034,2.14575]Nm和主动广义力矩约[2.867,3.214,2.991]Nm。忽略分布接触矩、轴向摩擦占用、接触迁移等，不能当已达到夹力或已验证抓法；20仅在局部候选中使用这些目标，canonical尚未更新。高静态预载不能直接沿用19的0.5Nm起减速阈值，否则会在起步就限速；20用固定位置保持和原硬保护验证新抓持方案。

## 主线和控制事实

正式入口：`src/kcg_connector/isaac/run_body_assembly_with_video.py` → `carts_v2/run_grasp_lift.py --visual-body-start`。复用同一world/stepper的视觉、Body抓取、搬运、观察、入槽、支承、换抓、旋拧、续拧。

- `visual_assembly_v1_body.yaml`和`visual_assembly_v1_task.yaml`为canonical。任务采用同形12三角桌面（Cube导致接近前漂移约8mm；同形mesh约0.007mm）、12mm总探入、3.5mm上移Nut抓位、首20°后90°分段。XY参数仍是08的原模式，尚未提升失败诊断候选。
- 原Body控制保留底座力矩误差修正位置参考；目标约[0.489391,0.432955,0.407838]Nm，掌面70°。NuT根力矩目标[1.7570975,1.25,1.2535247]Nm，1.5s渐升/2s夹紧，120Nm/rad换算、1/6s修正尺度、0.15rad/s；之后冻结位置目标。08夹紧实际根力矩约[1.62,1.14,1.15]Nm，不冒称精确力矩伺服。
- `te_hand_mechanism_runtime.py`共享4电机PD+有限passive_split蜗杆等效模型；4bar每步硬切线mimic，掌面保留完整限位及1:1两支路。Body输出等效主动cap1Nm，Nut掌1/指3.5Nm；传动结构边界掌1/指3.5Nm。输入等效cap=(1+c)*输出cap。Ktrans120、Dout2、PD264/4.4、c1.2均开发参考，未硬件辨识。初始化后不写实际状态。
- 力桥贴底座双桥，标定M=F*l绕根轴，不是电机转矩；无需先仿真ADC电路。原驱动模式仍可通过改位置参考主动退让，用户不要求强制导纳。
- NuT准备轴向0.20N，转动0.612729N对应原自重，未额外加压；轴向0.3mm/s/4mm范围，臂0.075rad/s，转动2°/s。Nut观测20N/弯矩0.4Nm/扭矩4.6Nm，源于当前模型和局部候选，不是厂家额定；原0.04Nm来自淘汰模型。
- 末段原抓位第三指碰Socket法兰。3.5mm上移只改抓位；离散覆盖旋角0–355°、深度4/9/14.605mm、轴向±0.5mm及松手路径，最小间隙0.7425mm。不是连续动态保证。canonical geometry：`src/kcg_connector/config/visual_assembly_v1_nut_geometry.json`。
- 06/07局部20°相同新抓位对照：启用XY力跟随后横向峰13.21→2.908N、末段7.14→0.872N，最大偏移0.286mm，握持末端产生Nut—Socket承载；Body尚未向前推进，不能冒称旋合完成。

## 当前有效模型与运行条件

M=`artifacts/kcg_connector/te_connector_contact_repaired_20260911`：自包含`connector_model.usdc`、安装器、validation_manifest；保留原几何、质量、质心、惯量、限位。单体原机构已完整旋紧/旋开，约355.7°/14.605mm/128簧套，正向峰3.0816Nm、反向4.2118Nm。原止挡`SourceMetalStopBox`，簧套按`/Leaf_`前唯一组计128。

Body—Nut工作轴隙±0.5mm，数值备份±0.6mm，后验拒绝≥0.5999mm；正常基线最大约0.553mm。原五键总角间隙约0.85118°，侧壁评价数值参考1µm。源内部摩擦0.20、外部0.45，不冒称厂家测定。

原指甲机器人：`artifacts/kcg_connector/isaac/te_nail_tip_body_grasp_v1/handarm_original_nails_source_decomposition.usda`，SHA256 4e220d97f3e12594c82dd6879cb584fd79cfd9264289a71e2383e43f53e6d46e；仍依赖10个旧层，见`docs/assembly_v1_usd_dependencies_20260912.json`，不能按旧目录删除。NAIL源面[11836,12912)，安装柱不算指甲。

SW完整包在`artifacts/hand_mechanism_audit_20260912/source/1`；四杆10–70–10–68mm，真实非线性函数在`finger_fourbar.py`和`hand_fourbar_20260912.json`。外部指传动240:1、掌面每支66⅔:1；不表示硬件效率/额定。

CPU/TGS960Hz、64位置/4速度迭代、每迭代外力、contact cap32768；原安装器在第一次reset前运行。启动：`PYTHONPATH=src/kcg_connector src/kcg_connector/isaac/run_isaac_python.sh ...`。离线环境`.venv/bin/python`有ijson/fcl等；不要用系统python假定有ijson。

## 记录、后评价与版本

- `GzipSampleStore`每512行独立gzip块，缓存2块，完整保留原始采样；长实验不整体复制raw。08后已修正`_evaluate_key_entry_after_motion`按step有序流合并，恢复08峰内存约290MB；物理结束立即封存raw，再进入后评价，避免再次丢尾段。
- 当前视觉在线记录仅图像、编码器和步号；GT误差分析统一在运动后读取归档。FT逐行gzip输出和大型评价数组外置无损gzip，旧证据不改。
- 独立脚本：`evaluate_visual_body_grasp.py`、`evaluate_body_support.py`、`evaluate_source_nut_pad.py`、`evaluate_source_key_containment.py`、`evaluate_visual_assembly_v1.py`。最终综合脚本尚未得到成功回合；最终红带/到位可见性审阅尚未落盘，不能假设已有。
- 当前分支`codex/visual-assembly-v1`，最近提交8e92313（任务预载/可控收尾），前f267403（入扣辅助/原生读数容差），前1165eb6（源阶段诊断/载荷分配候选），前0db8f15（封存raw/流式后评价），前2f8f436（探入/Nut候选），前744cdd1（共享机构/视觉取件）、c928aa7。之后后评价/流式收尾/局部诊断和此次IK修正未全部提交。仅暂存本任务源码和文档；无关修改/删除标记/未跟踪文件属于用户资产。
- 初始快照R/checkpoints/20260912T112851Z。路线清单`docs/assembly_v1_route_manifest_20260912.json`、存储版本盘点同目录。旧artifacts约555GiB，不作为当前前置清理任务；无删除。
- 本次收敛前的完整活动文档：docs/history/CURRENT_CONTEXT_CN_20260912T202917Z_before_nominal_ik_diagnosis.md。更早历史仅供追溯，不恢复旧参数/授权。
