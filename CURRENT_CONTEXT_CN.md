# 当前任务：Isaac Sim第一版视觉全链路装配

核验：2026-09-12T21:25:58.264884+00:00。用户已授权执行`docs/FIRST_VISUAL_ASSEMBLY_V1_PLAN_CN.md`，可按证据修订并持续反馈。任务未完成，不在局部旋拧成功处结束。simulation-only，hardware_authorized=false。

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

**当前唯一主要试验16 `source_stage_grip_balance16`，exec session40178。** 相对14固定XY/名义IK，只启用已有腕部横向力驱动三指载荷重分配分支，适配当前原PAD CAD点和共享机构。模型增量和为0，各指幅度8N，响应1/s；不声称实际总法向力严格恒定。以当前电机264、载荷摩擦1.2和传动K120推得闭合方向等效位置刚度60Nm/rad，替代旧分支误用的legacy K12换算；8N对应约0.020–0.023rad位置调整，分配矩阵条件数1.45。源驱动cap、夹紧预加载基准、力/速度/行程/几何边界不变。用于检验主动指间分配能否减少偏载和滑移，不扩大XY范围。源码/配方/命令均有同名绑定，恢复先核对实际进程。

12–16使用`te_source_stage_probe.py`：从08已结束的step249224冷初始化作局部诊断，复用正式`run_body_nut_rotation`/共享机构/当前RGB-D/同一传感器，不可当完整视觉验收。保留源空载FT零点与四电机输入角/速度；初始化后不回写实际q/qd/对象姿态。09–11只是入口问题：FT阶段合同、缺照明、观测标签及重复重力补偿，不能作物理控制结论。修复后预热腕位置变动8µm横向/39µm轴向。

下一步：读取16实际结果；按真实螺母转动、本体推进、抓内滑移与源接触决定后续。14也证明旧手—Body刚性关系不能长时间用于避碰位置，应在需要时使用当前RGB-D更新，而不能放宽包络或读真值代替。尚未把候选提升到canonical；准备的完整回合命令仅为draft，未执行。

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
- 当前分支`codex/visual-assembly-v1`，最近提交0db8f15（封存raw/流式后评价），前2f8f436（探入/Nut候选），前744cdd1（共享机构/视觉取件）、c928aa7。之后后评价/流式收尾/局部诊断和此次IK修正未全部提交。仅暂存本任务源码和文档；无关修改/删除标记/未跟踪文件属于用户资产。
- 初始快照R/checkpoints/20260912T112851Z。路线清单`docs/assembly_v1_route_manifest_20260912.json`、存储版本盘点同目录。旧artifacts约555GiB，不作为当前前置清理任务；无删除。
- 本次收敛前的完整活动文档：docs/history/CURRENT_CONTEXT_CN_20260912T202917Z_before_nominal_ik_diagnosis.md。更早历史仅供追溯，不恢复旧参数/授权。
