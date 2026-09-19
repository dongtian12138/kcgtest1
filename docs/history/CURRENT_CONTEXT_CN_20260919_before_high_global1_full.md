# 当前状态：两次键观测整机通过；高位全局相机1初抓通过

核验时间：2026-09-19T09:48:40.407410+00:00。本轮已授权实现和对应验证结束，当前无仿真实验或后评进程运行，未安排自动后续实验。用户偏好的高位G1已实际接入并通过初始粗定位/抓取/抬升/保持验证；不要把此前低位G1的完整成功视频说成高位G1整机成功。

## 有效实现与验收范围

实现工作树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/two-key-five-view-20260919`。原工作树`/home/noob/WorkPlace/kcgtest1`的脏资产未改，仅同步本入口；未推送。仅simulation-only，hardware_authorized=false。没有子代理授权。原模型、质量/质心/惯量、材料、CPU960Hz64/4 TGS/contact-last、力速/行程及验收限制保留。

方案：四工作相机；固定G2第一图+腕部槽图决定绕Body自身轴的粗转，保持目标中心/轴线；粗转与原保持完成后，固定G2第二图+同步掌心/编码器重新锚定主键。结果按模拟时间等待后消费。二次后的剩余绕轴修正须≤1°才进入精对准（这是小动作准入，不是键槽允差）。掌心持续修位置/轴线，不虚构轴向观测；Body释放前retire随手关系。录像是一个仅记录的主视角加四台实际工作相机，显示中文角色/阶段及带时刻的历史角读数。

## 完整装配：低位G1、插座独立+20°错位

回合`artifacts/two_key_20260919/socket_plus20_full_01/run`，执行源码`df1afac5bd5d9bdb56dd49b58ef8b9728092ed08`；装配配置`reproducibility/two_key_20260919/assembly_socket_plus20.yaml`，Body配置仍`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`。10878.632s、300583步。whole_assembly_review.status=VERIFIED、complete_visual_assembly_verified=true，独立3秒释放accepted=true。

- 插座yaw20°仅在reset前用于布置，未进入在线对键计算。G2第一图t102.652089s、腕部槽图t105.262505s，图像计算粗转−19.002261°；真实Body轴向转−19.028474°、J7转−18.921436°，中心移13.647µm、轴变化0.018119°。证明这轮并非搬运预设后直接对好。
- G2第二图t113.170839s，t115.451048s才消费；更新旧预测0.031659°，在线剩余修正−0.028627°。开始入槽的真实绕轴差后验0.105913°，不能把在线估计当真值。二次锚点后最大跟踪误差：中心21.067µm、轴0.004902°、主键轴向0.090196°。
- 两次G2同一固定安装位、四功能相机、手部固定安装关系和因果等待审核通过；掌心消费1240次。相机录像所有帧native状态差0。
- 实际末3s共2880帧：手原始冲量严格零、源止挡每帧正载、同128簧套每帧正载、Body/Nu清醒。深度14.604491～14.604521mm，满足原14.605±0.010mm；整轮备用轴坐标max0.519365mm<0.5999mm。
- 源键对槽面最大残差1.531157µm<原2µm。三NAIL Body初抓/搬运通过，初抓2s最低实际抬升55.9968mm。Nu原允许区域通过（f1/f3 NAIL、f2 PAD），无其它受载对象；Body释放原生支持通过。
- 在线到位True：独立视觉深度14.604166/14.604015mm、加载扭矩2.001026Nm、实际3s张手。六次抓握，累计下发357.871377°；第3段腕力储备、第5段弹性储备换抓未当成功，未增加原最多6次/总360°。
- 红带1024/1024径向样本被Nut遮挡，同回合原视频旋拧末/释放中/最终帧已查看，无露出红带。engine/identity/finite通过，启动后物体位姿改写0、在线对象/接触真值=false。真值仅记录及后评。
- 进程exit2与通用evaluation保留旧全指腹/旧记录通道/研究门的未通过标签，不得说所有JSON通过。完整成功由适用源接触、源几何、原生支撑、在线判断及整轮汇总建立。

完整视频`.../run/video/assembly_five_view_CN.mp4`，1080p/5fps/1566帧/313.2s（5分13秒）。原`assembly_five_view.mp4`保留。CN只改顶部说明，角读数明确标为历史，不替换任何工作相机视角。报告`reproducibility/two_key_20260919/two_key_full_result_CN.md`，精简证据`evidence/socket_plus20_full_01/`，原始数据全部保留。原完整旧单次键回合84bef70及full_chain_04也保留。

## 用户偏好的高位G1：已通过实际初抓，未跑高位整机

用户明确不认可低位近侧G1，偏好旧录像的高位俯视位。历史确有“全局1暂沿用近侧”记录；旧4分44秒右上是高位录像位，真实粗定位是低位。不能再用“对应控制相机”替代选址合理性说明，也不能把录像换位冒充识别已切换。

当前推荐G1采用`te_rgbd_camera_global_e50_v1.yaml`，eye[.952812,-.0125,.733740]m。使用配置`reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml`及同目录`four_camera_two_key_high_global1.yaml`/`global1_high_sources.json`；低位配置留作完整回合复现。源码`e77b86fa0388bd359b25da59d4c9f82b31c2723b`已把初始采图、标定/背景、同帧Socket粗定位及录像统一读取rig。Plug采用现有可见端面5DOF方法，原图像质量门、轴符号、3mm支持/5°倾斜界保持；丢弃圆面任意横向基，不推断键yaw。visual_body_start独立检查也开启原照明，无需增加额外采图相机。

- 高位静态历史图已复制到`artifacts/two_key_20260919/global1_high_replay`。旧低环474点<原1000门而拒绝，失败保留、未降门；改用现有端面方法，离线Plug/Socket位置误差24.066/34.230µm。该图无机器人，不是动态成功。
- 新预检`high_global1_preflight_01`208.403s、exit0，必要门全部通过。
- 实际初抓`artifacts/two_key_20260919/high_global1_grasp_01/run`，745.714s，源码e77b86f，high_global1_initial_review.accepted=true。同一当前高位RGBD定位两件，Body/Socket后验位置误差23.777/25.489µm；Body轴误差0.000248°。实际抬升55.9817mm、2s保持、三NAIL接触100%、保持桌面零接触通过；controller无失败，engine/identity/finite通过，真值未进入控制，录像native状态差0。
- 本回合明确仅初始定位/抓取/抬升/保持，未运行G2、搬运、插入或Nu，不能称高位G1完整装配已验证。视频`.../high_global1_grasp_01/run/video/assembly_five_view.mp4`63.8s，画面0/2键观测是范围在首次看键之前结束。

报告`reproducibility/two_key_20260919/high_global1_result_CN.md`，精简证据`evidence/high_global1_grasp_01/`，统一状态`delivery_status.json`。24项相关软件检查通过。准备补丁`high_global1_preparation/activate_after_current_run.patch`已应用，禁止再次应用覆盖现实现。

## 后续边界与入口

当前任务已交付对应验证，没有自动后续回合。用户若继续，可用已接入的高位G1配置做其余所需验证；未获得新结果前保持“高位仅初抓通过”的范围。旧5倍整轮提速尚未完成，±1mm/±1°初抓重复性仍未解决，不自动恢复无关扫描。

源NAIL初抓用evaluate_visual_body_grasp.py；Nu用evaluate_source_nut_pad.py --geometry-plan artifacts/grasp_capacity_20260914/selected_two_nail_geometry.json。两次键/高位初抓审核在reproducibility/two_key_20260919/review_two_key.py与review_high_global1.py；相机/箭头/3s/原生支持沿用已适配的four_camera_20260918脚本。SourceKey/Whole用对应isaac evaluator，红带用原postreview工具加实际同回合图像。只对已结束的运动做真值后评。

Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`，ISAAC_ENV_PREFIX同环境；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；SAM复用te_runtime_paths。OPENBLAS_NUM_THREADS=1；pytest禁用插件，PYTHONPATH加src/kcg_connector、isaac、carts_v2。普通rg受忽略时限定目录--no-ignore。一次只运行一个主物理实验，冻结加载代码；停止用run/STOP_REQUEST，不SIGINT、不在线延预算。

历史详见docs/history/CURRENT_CONTEXT_CN_20260919_before_two_key.md、before_high_global1.md、before_delivery.md。历史“运行中”不构成当前事实或授权。
