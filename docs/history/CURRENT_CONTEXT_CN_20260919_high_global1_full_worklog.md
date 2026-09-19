# 当前任务：高位全局相机1完整装配验证与仅相机名五画面录像

核验时间：2026-09-19T11:07:47.820985+00:00。用户明确授权“使用高位全局相机1，验证是否能够成功，成功生成录像，文字只保留相机名称和主视角”。前两轮均已停止，视觉修复已通过当前帧，初抓瞬时力矩故障正做单变量验证。必须执行完整装配及原物理验收，不能用此前仅高位初抓成功代替。

## 工作区与边界

工作树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/high-global1-full-20260919`；原`/home/noob/WorkPlace/kcgtest1`脏资产不动，仅同步本入口。仅仿真，hardware_authorized=false，无子代理授权，不推送、不删除原数据。一次只跑一个主物理实验，冻结加载代码；停止用run/STOP_REQUEST，不SIGINT、不在线延预算。

Body配置保留`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`，装配配置`reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml`。G1为高位TeGlobalE50，同帧两件粗定位；G2固定两次看键，中间视觉决定绕轴粗转，第二次重建主键；掌心位置/轴线，腕部插座/槽精定位。插座初始+20°仅场景布置，在线不读角度真值。原模型/质量/材料/CPU960Hz64/4/力速/360°上界/最多6次抓握/原验收保留。

录像刚局部修改`te_multiview_video.py`：仅显示主视角、全局相机1、全局相机2、掌心相机、腕部相机五个名称，不显示阶段/时间/角差/编码器。运行帧JSON仍保留在线观测与编码器用于审核。`assembly_five_view.mp4`将直接符合交付要求，不调用旧label_five_view.py加回额外说明。布局已用历史图片做离线合成检查（/tmp/kcg_camera_names_only_preview.png，仅排版检查，不是新回合证据）。

## 已验证比较基线

上轮低位G1+20°完整回合`artifacts/two_key_20260919/socket_plus20_full_01/run`执行df1afac，10878.632秒，whole VERIFIED及独立3秒释放通过。高位G1初抓`high_global1_grasp_01/run`执行e77b86f，745.714秒，实际55.9817mm抬升/2秒/3NAIL/table零通过，只证明初抓，不证明高位整机。详细前轮状态已归档before_high_global1_full，不覆盖原报告/数据。

## 当前已核验状态与下一步

截至 2026-09-19T11:02:48.531799+00:00，full_01 与 full_02 均已退出，无主实验运行。full_01 在完成初抓/抬升后，SAM 只分割中央端面而漏插座外圈，原几何门正确拒绝，707.946s 退出；局部修复用同帧普通背景差分与声明工作区恢复唯一重叠的深度连通区域，再跑原几何门，原失败/高低成功三帧重放通过，修复提交4b07471。

full_02源码d2fa140、491.350s退出2。新的插座定位通过，但初抓接近接触瞬间f2j1反力矩1.794573Nm超过原0.9Nm，停止未抬升。两回合视觉抓取目标完全一致、每步手指指令连续；物理视觉等待相差104步。不能断言唯一根因。关节尖峰前合拢速度约0.1775rad/s，末步2.7236rad/s；原接触记录未报告正手接触，记录时序/约束载荷仍待核实。原记录保留于high_global1_full_01/02，诊断high_global1_contact_diagnostic。

当前受控候选只将初次接触合拢速度0.18降为0.09rad/s，新增可选contact_approach_speed_rad_s，保留原预紧/保持控制速度、力和停止门。配置visual_body_contact_0p09.yaml，原配置保留。假设是减小接近接触时的动态载荷；不能把尚未验证的候选称成功。配置边界检查通过，预检high_global1_names_preflight_03已201.310s通过原必要门。唯一运行中主回合high_global1_full_03于2026-09-19T11:07:16.062969Z启动，源码8959158fe2925492647ef334c04c5535a3260a2a，supervisor PID962132，工具session95957，原21600s/300s收尾，names-only5fps。控制源码与配置冻结；只依据当前在线传感/控制日志跟踪进度，结束后再读真值验收。

完成后用原source_nail_body、source_nut_pad、review_two_key、camera audit、review_transport、原生支持、source_key_containment、独立3秒释放、红带几何及同回合图像检查，最后whole汇总。成功才交付录像，失败先定位最早原因，不盲扫。原始证据不删，不因程序退出或单项PASS认定装配成功。

环境Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python；ISAAC_ENV_PREFIX前者环境，OPENBLAS_NUM_THREADS=1，PYTHONPATH含src/kcg_connector、isaac、carts_v2。参考full_02及preflight_02的process.json argv，换新Body、output及preflight路径。执行期间源码/配置冻结，停止用STOP_REQUEST。后评使用without_cyclic_gc避免旧全量扫描GC开销，不改变数据或门。

本轮full_03已越过初次接触，pickup_controller_outcome.completed=true，抬升/保持控制阶段跑完，正在准备去固定G2。暂未做运行后真值验收，不称完整成功。实际preview已核对高位G1及仅五名称，原渲染和相机控制保持。后评助手/tmp/kcg_high_g1_postreview.py只允许运行结束且terminal记录存在后调用，逐项原review并关闭离线GC；随后仍必须实际看视频终态、写visibility复核和whole。

full_03最新在线进度：两次G2有效采样103.929172s/114.496881s，中间视觉粗转-19.294955°，第二次视觉剩余-0.070624°；已完成两段搬运及两次位置/轴线精对准，现key_probe_contact插入推进中（约160s仿真，掌心视觉深度约4mm）。无保护停止，不能据此称完整成功。工具session95957仍活动，候选源码8959158冻结。读取当前在线掌心/腕部图像结果可跟踪深度，禁止读取运行中对象真值做控制或调参。

full_03在线阶段已推进到约233s仿真：key_entry控制completed=true（视觉终深7.8508mm），body_support松手保持控制completed=true；已完成前两段螺母旋紧，当前nut_index_free_final_hold准备第三次抓握，尚未终态成功。实际preview已看本体松开且插头留在插座内；接触真实性/原始CAD面/3秒终态等仍须结束后原审查。不要读运行中truth。当前工具session95957仍活动，主进程962132。

full_03已结束（2026-09-19T12:59:01.430368Z），6705.369s、exit2、252522steps，无运行中主实验，session95957已关闭。第四次抓螺母 nut_regrasp_continued_02 STOPPED：校正-0.580131°后，新图像复核手与螺母外形角度误差-0.521842°，越过原0.5°门；前后Nu视觉估计227.275→227.800°（质量门均过）。前三段旋紧已执行，但未完成旋紧到位/最终释放，不能交付成功录像。视频已完成263.2s、1316帧、names-only。下一步只核对运行后阶段原始证据及该局部失败原因；禁止直接放宽阈值或原样盲重跑。
