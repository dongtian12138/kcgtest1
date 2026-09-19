# 当前任务：高位全局相机1完整装配验证与仅相机名五画面录像

核验时间：2026-09-19T10:22:05.263245+00:00。用户明确授权“使用高位全局相机1，验证是否能够成功，成功生成录像，文字只保留相机名称和主视角”。新预检已通过；首轮高位整机已在搬运前停止，局部视觉修复已通过原始同帧离线复核，准备重新预检及完整验证。必须执行完整装配及原物理验收，不能用此前仅高位初抓成功代替。

## 工作区与边界

工作树`/home/noob/WorkPlace/kcgtest1-performance-20260917`，分支`codex/high-global1-full-20260919`；原`/home/noob/WorkPlace/kcgtest1`脏资产不动，仅同步本入口。仅仿真，hardware_authorized=false，无子代理授权，不推送、不删除原数据。一次只跑一个主物理实验，冻结加载代码；停止用run/STOP_REQUEST，不SIGINT、不在线延预算。

Body配置保留`reproducibility/four_camera_20260918/visual_body_centered_preload.yaml`，装配配置`reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml`。G1为高位TeGlobalE50，同帧两件粗定位；G2固定两次看键，中间视觉决定绕轴粗转，第二次重建主键；掌心位置/轴线，腕部插座/槽精定位。插座初始+20°仅场景布置，在线不读角度真值。原模型/质量/材料/CPU960Hz64/4/力速/360°上界/最多6次抓握/原验收保留。

录像刚局部修改`te_multiview_video.py`：仅显示主视角、全局相机1、全局相机2、掌心相机、腕部相机五个名称，不显示阶段/时间/角差/编码器。运行帧JSON仍保留在线观测与编码器用于审核。`assembly_five_view.mp4`将直接符合交付要求，不调用旧label_five_view.py加回额外说明。布局已用历史图片做离线合成检查（/tmp/kcg_camera_names_only_preview.png，仅排版检查，不是新回合证据）。

## 已验证比较基线

上轮低位G1+20°完整回合`artifacts/two_key_20260919/socket_plus20_full_01/run`执行df1afac，10878.632秒，whole VERIFIED及独立3秒释放通过。高位G1初抓`high_global1_grasp_01/run`执行e77b86f，745.714秒，实际55.9817mm抬升/2秒/3NAIL/table零通过，只证明初抓，不证明高位整机。详细前轮状态已归档before_high_global1_full，不覆盖原报告/数据。

## 下一步

1. 冻结仅标签改动，使用高位同一配置做独立新预检（300秒有界），通过后启动高位完整回合（原21600秒/300秒收尾）。录像5fps同回合采集，原始真值完整保留，只在运动结束后评价。
2. 检查G1确为高位且同帧定位两件、两次固定G2及视觉非零粗转、插入/旋紧/到位；失败先定位最早原因，有可检验差别才调整，不盲扫。
3. 完成后用source_nail_body、source_nut_pad（geometry-plan为selected_two_nail_geometry.json）、review_two_key、camera audit、review_transport、原生支持、source_key_containment、3秒终态、红带几何及实际同回合图像检查，最后whole汇总。通用exit2可能仍有旧评价口径，不能冒充所有JSON通过。
4. 交付实际成功回合的五画面仅名称录像，报告实际计算时间与仿真录像长度，更新唯一CURRENT_CONTEXT及本任务证据并提交，不推送。若未成功如实保留失败和当前阻塞，不提前结束已授权必要工作。

环境Isaac Python `/home/noob/WorkPlace/isaacsim/.conda-env/bin/python`、ISAAC_ENV_PREFIX同路径；规划Python `/home/noob/WorkPlace/kcgtest1/.venv/bin/python`；OPENBLAS_NUM_THREADS=1；PYTHONPATH含src/kcg_connector、isaac、carts_v2。运行参考上一完整process.json，换高位assembly配置及新output/preflight。后评脚本位于reproducibility/two_key_20260919和four_camera_20260918，Whole/源键/接触在src/kcg_connector/isaac。禁止在运行中读取对象真值做控制/调参。

当前回合`artifacts/two_key_20260919/high_global1_full_01`，执行源码53ee996e8ab567b41338b43753dd93269ebf53e2，2026-09-19T10:26:20.851665Z启动，supervisor PID897863，工具session26069，21600s上界/300s收尾。预检`high_global1_names_preflight_01`201.274s、exit0，preflight/controller/engine/identity/finite全通过，五画面names-only、渲染native差全零。当前尚在启动，不称完整成功。控制源码与配置冻结。

高位full_01已结束：707.946s、exit1；抓取/抬升/保持完成，prekey_transport在WAITING_FOR_GLOBAL1_SOCKET停止，未搬运。SAM最高有效提议只包含中央端面3960pixels，漏外圈，原global_socket_coarse_geometry环半径检查正确拒绝。诊断同帧使用普通静态背景差分+声明Socket工作区，选择与原SAM seed唯一重叠的连通深度物体区域11681pixels；再跑同一个原几何估计，原半径/残差/倾斜/支持门全部保留并通过，后验中心误差25.77µm量级，报告见high_global1_socket_mask_diagnostic。原失败数据与mask保留。

最小修复：global_socket_coarse_geometry新增depth_component_from_image_seed，four_camera_global_localization仅在原几何估计ValueError时恢复当前深度component，再跑原估计；不得篡改SAM原掩膜或分数，不读对象真值。3项软件检查通过（同物体扩展、两个物体歧义拒绝、背景不充当物体），失败帧与之前高/低成功帧重放通过，两个成功帧不走fallback。接下来冻结修复，做新预检后再执行完整回合，不原样盲重跑。
