# 当前任务：四相机、一次键观测与真实闭环装配改进

核验时间：2026-09-18T15:04:13.300374+00:00。low_key_station_entry_04已结束并通过源几何/原始接触入槽支持复核；新版完整装配仍未完成。

## 当前授权及不变验收

用户“可以按照你说的做”授权继续改进并验证：四功能相机；抓起→固定G2观察站→插座上方两段短搬运；一次主键箭头加掌心5DOF修正；感知/规划结果可用前的物理等待；保护停止与到位判断分开；旧X+1mm/yaw+1°失败核对及少量扰动。

- G1同帧粗定位两件；G2固定且插头底部键只看一次；掌心固定于手、测Body位置和轴线，不测轴向角；腕部固定于手、测插座与键槽。额外录像视角不进控制。
- 实现树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/four-camera-single-key-20260918`，HEAD f170d6b。原`/home/noob/WorkPlace/kcgtest1`只同步本上下文，其它脏资产不碰。未推送。
- 仅仿真，无硬件授权。一次只运行一个主物理回合，运行期间冻结相关源码；已结束数据可离线分析。当前未授权子代理。
- 原CPU960Hz、64/4、TGS/contact-last、CAD/SDF/质量/材料/力速行程边界保留。只沿用无损记录优化；不恢复已退回数值加速候选。
- 完整验收仍为14.605mm±10µm、源止挡正载、源键槽穿透≤2µm、原NAIL/PAD接触身份、连续3s完全张手原始手冲量严格零、同128簧套逐帧正载、Body/Nut清醒、备用轴限<.5999mm且不承载。真值只在结束后评价，不进控制。
- 原成功reference：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/run`，295421步、15909.99s。5倍整轮加速未达成；短搬运距离改善不等于整轮提速。

## 当前状态与最近有效阶段

没有主物理回合运行。`artifacts/four_camera_20260918/low_key_station_entry_04`已于2026-09-18T14:58:02.129890Z结束，2694.3986s、158678物理步、source f170d6b、exit2旧通用评价标签。session12786已结束。范围到松手支持，没有Nut旋拧。

- 实际初始抬升55.9571mm、2s保持、三原NAIL接触全程成立，source_nail_body_review accepted=true。
- 实际两段搬运0.3971397+0.0603799=0.4575196m；峰值arm速度.1206714<原.15rad/s。原主段3.4746m，约7.6倍路程比，非整轮时间比。actual_transport_path_review.json。
- 功能相机恰好4个，G2键恰好1次，掌心512帧延迟消费，腕部实际看清全部5槽，固定安装审计通过。3mm低位/60mm侧距，G2 eye[.378,.319,.244]/target[.490,.185,.275]。
- 实际探入约7.85mm，松手后7.960928mm。源CAD全键顶点对真实槽面复核accepted=true，最大残差0.474573µm<原2µm；五键全部进入口内。source_key_containment_review.json。
- 中间换抓支持窗口仅0.5s/480帧（不是最终3s验收）：每帧原始手冲量严格零、插座接触正载、Body/Nut醒着，高度范围0，最大备用轴坐标0.268µm未靠限位。native_body_support_review.json。原始正载来自Nut源CAD/原插座NutContactSurface及源键，未修改物理模型。
- 注意旧physical_key_entry_result近似角边界给出yaw.424°、角余量-.02786°；旧evaluate_body_support用当前未记录的tensor_headers误报无支撑。**原报告保留**，不改阈值；实际源CAD槽面及原生poll_headers审查见上述独立报告，不以旧简化模型冒充源几何，也不从未记录通道推断零接触。
- 单键跟踪到主动松Body之前最大中心19.23µm、轴.009759°、轴向键误差.277116°；松手前轴向误差.275615°。5DOF仍不测轴向转动，物理入槽通过不代表该分量被观测。four_camera_transport_posthoc.json已按真实可用时刻开始、retire截止。

进入完整回归前发现并已修复后段缓存确认问题：te_visual_seating_axis四相机模式原会暂停后读两次同一缓存帧；现生成原有受控保持步骤，等不同采样时刻、已消费且因果可用的掌心图像；0.5s无新帧则停，原10µm/.005°一致性门不改。7项相关测试通过（含旧4项及重复缓存/无新帧/未来帧3项）。完整Nut动态待验证。review_transport原元数据误把recorded FK标为native，现更正为与在线相同的编码器FK，不改变已算出的误差。

接下来：完成当前改动记录后进行完整回归预检，再从初始状态跑一次含Nut的四相机整机；仅一个主回合、21600s/240s收尾，不在线延长。配置visual_body_centered_preload.yaml/assembly_fixed_cameras.yaml及原arm seed[0,0,0,-pi/2,0,pi/2,2.5]、friction_lower_0p45、parallel_contact_latch、msgpack/native-report/5fps保持。之后执行已定义的小扰动；当前扰动0次，完整整机0次成功。

## 最新阻塞及已结束证据

1. `low_key_station_entry_03`（5471449）1360.331s，2026-09-18T13:38:59Z结束，exit2。G1图像粗定位、抬升/搬运、低位G2一次键成功；G2初始角误差约-.070°（事后）。腕部槽被拒绝`SOCKET_MAIN_SLOT_PATTERN_NOT_UNIQUE`，尚未归中/接触。真实图像显示35mm侧距时插头遮住部分口沿；原外轮廓圆拟合残差.759mm，4弧宽角被扭曲。保存帧上的外圈/RANSAC诊断可恢复中心但yaw仍约.22–.25°，**没有采用这些估计器改动**。当前仅扩大侧距后用原估计器验证。
2. `low_key_station_entry_02`（7e28b6a）690.803s，12:38:04Z结束。G1原SAM6D学习姿态粗估严重歪轴，IK在执行前拒绝。5471449改为当前同帧SAM mask+depth上口沿几何粗定位，不用PEM；五个已结束初始帧重放通过，约.47mm中心误差属粗定位，腕部仍负责精度。没有读对象真值。报告`global1_coarse_geometry_replay.json`。
3. `low_key_station_entry_01`（51de77e）1307.615s，12:09:49Z结束。低位站G2窄键掠视、原宽度门拒绝。7e28b6a把固定G2方位转到主/窄键视向中间；03实际成功。原键宽/置信门没有放宽。
4. `full_chain_02`（97da4b1）3518.445s，11:28:37Z结束。50mm高位看键后，约37s规划物理保持+44s下探把键记忆轴向误差扩大到约.4°；13.175s低力探入触发原.3N横向力停止（.307159N），实际只入.776899mm，源键槽角余量-.1101°。安全退出，未松手/旋拧。因此唯一键改到真正接触前原3mm高度，去掉47mm长下探，不增加键观察次数。
5. `near_socket_key_station_01`（3efdabc）1444.85s，107771步。大搬运/转向后50mm高位单次看键，随后短归中，到达主键轴误差.041475°（最大.045315°），中心20.14µm，轴.000411°；真实抬升/三NAIL接触/四相机审计通过，仅搬运前缀。比早看键`short_transport_01`末.555°改善，但没有证明入槽。
6. 原长搬运主段约3.4746m。`short_transport_01`实际两段.435970m、原关节速度内；原NAIL抬升保持通过。掌心把中心17.49µm/轴.000368°，但无法消除额外轴转.555°。不能假定抓Body必然不转。
7. `full_chain_01`原抓取力门早停：过滤控制死区±.01与原始验收±.01同宽导致贴边。97da4b1新配置只把调节死区缩至.005，原目标/验收/6连续帧/.5s窗口/力速行程不变。`centered_preload_grasp_01`真实抬升56.00785mm/2s/三NAIL通过，此后full02、低位03/04初始力门也通过；不据此宣称鲁棒。

## 代码与针对性检查

- `four_camera_rig.py`及config固定四相机；Global1同帧粗定位两件（SAM分割+几何插座中心轴）；Global2一次键后`adopt_key_anchor`。
- `KeyDirectionMemory`用手坐标下最小轴旋转运输主键箭头；忽略轴对称圆的任意横向基；编码器传播手运动。掌心每.2s图像，.05s声明传感延迟+实际几何计算延迟后在物理步消费；硬件延迟未标定。Body主动松手前retire，之后Body不跟随Nut旋转。
- `short_body_motion.py`直线位置+姿态插值/原Dogbox软限位；离散完整17链/携物几何检查≤.01rad，不宣称连续碰撞证明；原960Hz实物理监控不变。规划/感知计算等待对应物理保持。
- `te_body_key_entry.py`保留原最多三次有限纠偏和低力探入，只用掌心5DOF+箭头，不再额外看键。`four_camera_post_entry.py`接回原支持/换抓/旋拧，动态后续尚未通过。
- `four_camera_online_completion.py`用独立两时刻视觉深度14.605±20µm/稳定5µm、原腕扭矩≥.4Nm、实际3s张手保持与对准条件；力停止单独不报成功。原物理10µm验收独立。成功旧回合传感器重放及失败用例测试通过，当前新系统未到达终态。
- 针对性key5/latency2/workspace2/session3/completion5检查已通过，不替代物理结果。未改动时不用反复跑。

## 结束后复核与后续顺序

1. 04阶段实际通过，准备完整Nut回归。恢复先查最新process.json及进程，不凭历史“运行中”等待。运行期间不改加载源码；失败则诊断最早原因，不盲扫或放宽门。
2. 复核工具：`reproducibility/four_camera_20260918/review_transport.py`从key availability_step起重放已消费5DOF，Body retire后截止；`audit_camera_records.py`从process.source_commit读取rig，核对四路径/G2一次/固定安装/因果延迟；原`evaluate_visual_body_grasp.py`、`evaluate_body_support.py`、`evaluate_source_key_containment.py`和完整装配原reviewers。
3. `review_terminal_release.py`逐帧检查末3s完全零手/源止挡/同128簧套/醒着/原深度/备用限位。旧成功重放通过，输出当前artifacts，原树未改。新整机尚未用到。
4. `visual_body_centered_preload_variation.yaml`及`variation_plan.json`仅定义+1mm/+1°和镜像-1mm/-1°；**未运行**。需对应配置预检后逐个执行。旧+1mm/+1°失败是最初接触后三步f1j3=-4rad/s>原3，非Nut载荷或目标跳变；不能声称新死区解决它。保留原提出闭合.18→.09rad/s可归因诊断，尚未实施。
5. 最终记录实际完成/未完成，不把exit0或前缀当作装配成功，不把路程减少当整轮5倍提速。

## 环境及历史

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同环境；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM Python `/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python`，SAM根 `/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D`。
规划LD_LIBRARY_PATH为原.venv的tesseract_robotics+Isaac lib；AMENT_PREFIX_PATH本树install/iiwa_description:/opt/ros/humble；OPENBLAS_NUM_THREADS=1。测试禁用pytest插件自动加载，sys.path加src/kcg_connector、isaac、carts_v2。rg必要时--no-ignore限定目录。
详细旧过程保存在`docs/history/CURRENT_CONTEXT_CN_20260918_before_low_station_60mm.md`以及之前历史文件。artifacts中的contact_integration_candidate/prepare脚本已应用且过时，不要重新执行覆盖当前源码。没有删除原始数据。
