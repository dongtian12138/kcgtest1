# 当前任务：两次键位观测、真实视觉粗转与五画面录像

核验时间：2026-09-19T05:50:35.462787+00:00。用户已明确授权实现并动态验证：固定G2第一次看键后做绕轴粗调整，第二次重新观测建立正式插入键方向；录像为一个旁观主视角加G1/G2/掌心/腕部四个实际工作相机。+20°独立预检已结束并通过；当前唯一主回合 `artifacts/two_key_20260919/socket_plus20_full_01` 正在运行，须恢复时核实实际进程。不得仅做代码或文档后结束任务。

## 工作区和验收

实现树 `/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支 `codex/two-key-five-view-20260919`。原工作树脏资产不动，仅同步本入口。无硬件授权，无子代理授权，不推送、不删除原始证据，一次只运行一个主要物理实验。当前用户授权覆盖必要源码改动、受控仿真及对应验证。

保留原完整名义基线源码84bef70、证据`artifacts/four_camera_20260918/full_chain_04/run`。它证明固定名义场景完整装配，不能证明大角度未知错位的自主视觉对键。审计发现默认插头Rx180、插座零yaw，且预观测搬运将临时Y轴置为世界+Y，因此默认场景提前接近对中。没有在这段目标生成链发现在线键角真值读取，但固定场景与预设动作存在有利匹配。

新验证选插座初始绕轴+20°，保持原初抓和预观测搬运策略不变；角度只在第一次reset之前用于场景安装，控制必须从当前G2与腕部图像得到角差。配置`reproducibility/two_key_20260919/assembly_socket_plus20.yaml`；Body仍`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`。四相机定义`reproducibility/two_key_20260919/four_camera_two_key.yaml`。

## 已完成与下一步

已接入两阶段：纯函数由当前键/槽视觉位姿计算保持Body中心和轴线的绕轴粗转；实际粗转和原2s保持完成后，G2/掌心/编码器同步第二次取样，因果等待后显式更新键记忆。第二次之后允许的剩余绕轴修正上限1°只是小动作准入约定，不是键槽物理允差；超过即不插入。原力速/碰撞/行程和最终验收不放宽。

五画面直接用四台感知相机的prim及安装位姿，额外只有旁观主视角；新文件名`assembly_five_view.mp4`，附中文标签、图像采样时间、在线测得角差和编码器J7，不显示真值为在线测量。连续录像图像不返回控制。旧四画面为非四相机模式保留。

当前相关软件20项检查通过，包括第二次观测的时间/阶段约束、轴向滑转重测、保持位置/轴线的转动、视觉槽方向变化导致相反转向、原单次分支和收尾。尚未验证物理表现。

下一步：继续跟进当前唯一完整回合，不改其已加载源码/配置。该回合使用原21600s上界，保留全部原始记录和五画面；run/STOP_REQUEST用于受控停止，不SIGINT、不在线延预算。结束后逐项原完整验收，使用新增`review_two_key.py`核对两次真实G2观测、首次明显非零角差、实际粗转、第二次重新锚定及插入前误差；适配后的camera audit、箭头后评和whole review支持两次观测。固定插座后评使用冻结安装记录中的Socket变换。原始失败证据保留。

最新在线进度（仅传感器/控制记录，尚非物理验收）：G2首次样本t=102.6520887s，腕部槽样本t=105.2625055s；图像计算的粗转−19.0022613°已执行；G2第二次样本t=113.1708392s，t=115.4510477s消费；二次观测相对旧预测更新0.0316586°，剩余修正−0.0286274°。已完成在线入槽目标7.85mm、本体张手及支撑控制阶段，进入Nu旋拧。前两段各约90°；第三段在追加56.819164°时触发WRIST_FORCE_RESERVE_EARLY_REGRASP，累计命令236.819164°，当前继续换抓，视觉深度约12.117mm。尚未到位，不能把正常换抓停止说成成功。progress.json可能停留在上一更新点228.122°，已完成段的最终值应读nut_rotation_controller_result.json。

录像交付注意：当前原始五画面顶端“二次观测后待修正角”显示的是带采样时间的历史测量；结束后将文字明确为“第二次测得角差/历史读数”，避免进入后续阶段仍被误读成实时残差。保留原始录像，注释只用同回合在线帧记录；运行中不改录像源码。已实际查看五画面有效彩色图，G1/G2/掌心/腕部各有标签。

已完成的离线输入隔离：同一旧回合真实深度图，把初始猜测yaw改为0/+60/−80°，键识别结果不跟随猜测（Plug变化0，Socket仅约5e−9°数值差，不是精度声明）。报告`artifacts/two_key_20260919/offline_input_isolation/seed_yaw_invariance.json`，未运行物理、未使用真值。配置差异报告同目录`configuration_difference.json`确认原物理/力速/验收未变。

## 环境

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同环境；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM环境/依赖保持原入口。OPENBLAS_NUM_THREADS=1。pytest禁用插件，PYTHONPATH加src/kcg_connector、isaac、carts_v2。普通rg受忽略时限定目录加--no-ignore。

旧验收/未解决初抓重复性与5倍提速边界详见`docs/history/CURRENT_CONTEXT_CN_20260919_before_two_key.md`。不要因历史运行中描述自动恢复旧回合。

本轮运行：预检`socket_plus20_preflight_01`，源码2cd44f7，198.916s，preflight/controller/engine/identity通过；安装后Socket实际yaw20°，5视角渲染native状态差全零。完整回合`socket_plus20_full_01`源码df1afac，2026-09-19T05:56:37.788492Z启动，supervisor PID486560，工具session34131，原21600s预算。全过程保存`run/video/assembly_five_view.mp4`。仅运行结束后使用`reproducibility/two_key_20260919/review_two_key.py`以及适配后的`audit_camera_records.py`/`review_transport.py`。当前不可读取运行中物体真值作控制或决定调参。

## 用户中途明确的相机1偏好（当前试验不受打断）

用户指出并不认可当前低位近侧G1，记得以前已批评过这个视角，偏好旧4分44秒录像中的高位俯视视角。已解释并纠正：旧录像右上角是高位global_e50旁观位（eye[.952812,-.0125,.733740]），真实初始粗定位G1一直是near_side（eye[.690992,-.395170,.306815]）；新五画面直接用了真实G1。历史`CURRENT_CONTEXT_CN_20260918_before_short_transport.md:24`明确写“全局1暂沿用原近侧固定相机”，这是临时配置被沿用，不能据“与控制一致”认定安装位置合理。高位视角几何预检覆盖两件；原来小键像素不足不应否定它做粗定位。用户当前偏好明确，承诺把高位G1列为待修正，做必要定位验证；不在当前回合中切换、不用替换录像画面冒充控制切换。当前完整回合仍必须如实保留低位G1，不因此中断。

候选相机文件`src/kcg_connector/config/te_rgbd_camera_global_e50_v1.yaml`，observe-only覆盖配置`te_rgbd_observe_only_global_e50_v1.yaml`。初始识别`te_visual_body_start.py`目前硬编码near_side以及`te_same_reset_rgbd_observe_v1.json`的标定/静态背景，正式替代必须连同真实采图、标定和粗定位输入改正，不能只改rig或录像。高位RGBD现成证据尚在查找，`workflow_a/te_rgbd_observe_only_v1`当前只有near_side目录。已只读查本任务历史，未查到用户提到的更早原句，但当前偏好不需要重新确认。

高位G1最新离线结果：原始高位数据在原工作区`artifacts/kcg_connector/isaac/te_visual_ft_assembly_v1/workflow_a/te_rgbd_observe_only_v1/global_camera_e50_run23_v1`，已完整复制到本树`artifacts/two_key_20260919/global1_high_replay`。旧低环粗定位因474点<原1000点门而拒绝（plug_coarse_result.json保留，未降门）。复用现有`estimate_plug_rear_circle_from_float_depth`处理同一图像普通背景差分+声明工作区掩膜，端面定位成功，原质量门保留；后验位置误差24.066µm、轴0.000270°。Socket采用原SAM6D+global_socket_coarse_geometry，位置误差34.230µm。所有估计保存后才读取历史capture_report的truth比较；当前运行未改。结果rear_face_coarse_candidate.json / socket_coarse_result.json / coarse_localization_posthoc.json；此静态图无机器人，不能称已验证新G1抓取或整机。已告诉用户这些正面离线结果，需改初始识别以高位可见端面为特征，不只是替换rig/录像。

新增未接入运行的交付注释脚本`reproducibility/two_key_20260919/label_five_view.py`：结束后从同回合online frame JSON生成明确“历史读数”的中文顶栏，保持原视频。已经在已结束预检视频做注释/渲染检查（/tmp/kcg_five_view_caption_check.mp4/png）；预检未开视觉照明，背景黑属旧预检模式，仅用于字幕排版检查，不是正式交付录像。正式回合五画面彩色影像已看过正常。
