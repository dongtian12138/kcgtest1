# 当前任务：两次键观测完整验证已结束，正在验证用户要求的高位全局相机1

核验时间：2026-09-19T09:23:04.895995+00:00。用户授权实现“两次看键，中间按视觉粗转，第二次重建主键记忆”及“主视角+四工作相机”录像，随后明确不认可近侧低位G1，偏好旧录像高位俯视位。继续完成必要验证，不能仅停在代码或报告。仅仿真、无硬件授权、无子代理授权、不推送、不删原始数据；一次只运行一个主要物理实验，离线分析可并行。原工作树脏资产不动，仅同步本入口。

## 已结束完整回合与已验证事实

实现树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/two-key-five-view-20260919`。完整回合`artifacts/two_key_20260919/socket_plus20_full_01/run`，执行源码df1afac5bd5d9bdb56dd49b58ef8b9728092ed08，2026-09-19T05:56:37.788492Z启动，已结束，10878.6321秒（3小时1分18.6秒）、300583物理步。exit2仍含旧通用全指腹/旧记录通道评价标签，不代表本轮所有JSON均通过。controller completed=True/failure_reason=None；engine/identity/finite为true；objects pose writes after start=0、online truth used=false、truth returned to controller=false。

该回合保持原Body抓取配置`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`，装配配置`reproducibility/two_key_20260919/assembly_socket_plus20.yaml`。只在reset前把Socket绕轴转20°，控制通过当前图像测量，物理参数/力速/总360°上界/原验收不变。G1仍低位near_side，不能把这份完整录像说成已采用高位G1。

- G2第一图t102.6520887s；腕部槽图t105.2625055s；由图像计算粗转−19.0022613°。原位只改绕轴的目标，实际Body轴向转−19.0284745°、J7转−18.9214361°；Body中心实移13.647µm、轴变化0.018119°。
- G2第二图t113.1708392s，t115.4510477s才消费；同固定安装位。新图相对旧预测更新0.0316586°，二次在线剩余修正−0.0286274°。不能称它为真实角差。开始入槽的真实绕轴差后验0.105913°。
- 两次G2/四功能相机/五录像视角/因果等待审计全部通过。掌心消费1240次，仅修位置和轴线。第二锚点后的最大跟踪误差：中心21.067µm、轴0.004902°、主键轴向0.090196°；Body释放前retire，掌心不伪造轴向观测。
- 在线到位确认True，独立图像深度14.604166/14.604015mm，实际3秒张手保持，先前加载扭矩2.001026Nm。六次抓握旋拧，累计实际下发357.871377°；第三段在236.819164°因腕力储备换抓，第五段在356.247276°因弹性储备换抓，均未当成到位。原最多6次、总360°保留。
- 独立末3s复核accepted=True：2880帧手冲量严格零、源止挡每帧正载、同128簧套每帧正载、Body/Nut醒着；实际深度14.604491–14.604521mm，符合原14.605±0.010mm；末窗备用坐标max0.519276mm<0.5999mm。
- 源键/槽面最大残差1.531157µm<原2µm，accepted=True。原三NAIL Body初抓/搬运审核通过，初始2s保持最低抬升55.9968mm。Nu原允许区域审核通过（实际f1/f3 NAIL、f2 PAD），无其它受载对象。本体卸载后0.5s真实原生支持审核通过。
- 原红带1024/1024径向样本被Nu遮挡，同回合旋拧末/释放中/最终原视频帧已实际看过，无露出红带。final_mating_visibility_review.json已写。
- 整轮汇总已结束：whole_assembly_review.status=VERIFIED、complete_visual_assembly_verified=true。整轮最大备用轴坐标0.519365mm<0.5999mm。全部适用独立检查通过；不存在仍在运行的旧回合或汇总任务。报告`reproducibility/two_key_20260919/two_key_full_result_CN.md`与精简evidence/socket_plus20_full_01已生成。

完整原视频`.../run/video/assembly_five_view.mp4`，中文明确历史读数版`assembly_five_view_CN.mp4`，1080p/5fps/1566帧/313.2秒。CN只替换顶部标题说明，原相机画面及原视频保留，字幕值只来自同回合在线frame JSON。已看过粗转与末帧CN画面排版；原版顶部“待修正角”在后半段可能被误解为实时残差，交付应使用CN版。原始渲染每帧native状态差均为0。postrun_evidence有原始抽帧与selected_actual_frames.json。

## 高位G1：当前唯一新物理实验

用户两次中途强调高位视角更清晰、已曾否定低位，不应因旧算法继续沿用低位。历史记录确实写“G1暂沿用近侧”。旧4分44秒右上为高位global_e50录像位，旧真实G1为低位near_side；此前回答“对应控制所以正确”没有解决选址问题，已纠正。

原始高位RGBD在原工作区`artifacts/kcg_connector/isaac/te_visual_ft_assembly_v1/workflow_a/te_rgbd_observe_only_v1/global_camera_e50_run23_v1`，已复制到本树`artifacts/two_key_20260919/global1_high_replay`。旧低环方法因474点<原1000而拒绝，失败证据保留，未降门。复用现有可见端面float-depth方法（原质量门）得到Plug位置后验误差24.066µm；原SAM6D+Socket粗几何得到34.230µm。估计保存后才比较历史真值；该静态历史图无机器人，不是新G1抓取或整机成功。

源码e77b86fa0388bd359b25da59d4c9f82b31c2723b已在完整物理回合结束后提交并启用高位接入（不是在旧回合中途换位）：
- 新`global1_visible_face.py`：当前普通深度/背景差分/声明工作区/CAD端面，保留轴符号、原3mm支持高度/5°倾斜界和图像质量门；不推断键yaw。
- `te_visual_body_start.py`与`four_camera_global_localization.py`从rig读G1相机及标定/背景，不再硬编码near_side；选高位时使用端面5DOF，初抓丢弃圆面任意横向基。
- `run_grasp_lift.py`确保visual_body_start也开启原照明，使独立初抓检查不依赖postgrasp-key标志；不增加相机/力/物理修改。
- 配置`reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml` + `four_camera_two_key_high_global1.yaml` + `global1_high_sources.json`。G1为`te_rgbd_camera_global_e50_v1.yaml`，eye[.952812,-.0125,.733740]；Body配置保持原版。
- 24项相关软件检查通过，高位标定/背景哈希及安装位姿核对通过。加载代码现在冻结，等待本轮预检结果。

高位预检`high_global1_preflight_01`已结束，208.403s，exit0，preflight/controller/engine/identity/finite全部通过。当前唯一物理实验`artifacts/two_key_20260919/high_global1_grasp_01`，09:26:05.096203Z启动，supervisor PID804581，工具session85449，900s上界、60s收尾保留，源码e77b86f。仅初始视觉抓取/抬升/2s保持，无postgrasp-key/transport/insertion/Nu参数。实际初始图像已确认高位TeGlobalE50相机，同帧定位Plug和Socket，Body端面5DOF已用于实际计划。已通过原夹持预载门，最新约55.8仿真秒仍在抬升，需恢复时核对真实进程。不得读取其运行中对象真值。

## 下一步与尚未完成

1. 旧+20°完整回合已全部验收并写报告/evidence。等待高位G1检查后统一更新交付状态和最终CURRENT_CONTEXT，保留原始与失败证据。
2. 跟进当前高位G1初抓回合，原900s上界不在线延长。结束后用原source_nail_body_review核对实际抬升/保持/接触，核对高位当前初始帧后验误差及同帧两件定位。它不代表高位G1完整装配已跑通，不可将旧低位完整视频换镜头冒充高位成功。
3. 高位初抓后用原source_nail_body_review核对实际抬升、保持、接触与来源，不能只凭程序结束。更新唯一CURRENT_CONTEXT入口（原工作树同文同步），必要文档和本任务文件提交，不推送。
4. 高位新回合未开始前，不再应用`high_global1_preparation/activate_after_current_run.patch`（已应用）；它只作历史准备证据。停止仿真用run/STOP_REQUEST，不SIGINT。无自动后续实验授权，不创建自动化。

## 环境

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同环境；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM运行时复用`te_runtime_paths`。OPENBLAS_NUM_THREADS=1。pytest禁用插件，PYTHONPATH加src/kcg_connector、isaac、carts_v2。离线视觉import完整runtime需LD_LIBRARY_PATH含规划环境tesseract_robotics和Isaac环境lib。普通rg受忽略时限定目录加--no-ignore，原始资产多在原工作区，不能因工作树没复制就判定数据不存在。

旧目标5倍提速尚未达到；这次+20°回合多一次换抓且加双观测，不能用与旧0°不同场景的耗时直接归因录像开销。旧初抓±1mm/±1°重复性仍未解决，当前不自动恢复无关扫描。详细历史见docs/history/CURRENT_CONTEXT_CN_20260919_before_two_key.md及before_high_global1。
