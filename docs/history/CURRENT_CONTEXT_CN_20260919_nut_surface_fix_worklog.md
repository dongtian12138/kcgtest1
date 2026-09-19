# 当前任务：修复螺母换抓的跨视角角度误停

核验时间：2026-09-19T14:28:36.122380+00:00。用户问“问题出在哪、能否解决”，正在落实根因修复及局部验证；局部验证及后评已完成，完整高位复验正在运行。工作树/home/noob/WorkPlace/kcgtest1-performance-20260917，分支codex/high-global1-full-20260919。仅仿真，无子代理，不推送，不动原项目脏资产，只同步本入口。

## 原结果与验收

原完整成功基线仍为低位G1 socket_plus20_full_01（df1afac），高位最新full_03（8959158，6705.369s）在第四次螺母换抓前停于视觉误差0.521842°>原0.5°，最终到位/3秒松手未完成。高位定位、实际56mm抬升2秒、两次固定G2看键、插入、0.5秒本体松手、已执行的指腹接触均已独立通过。前轮报告reproducibility/two_key_20260919/high_global1_full_result_CN.md及full_status.json保留，旧session95957已结束。

## 已定位根因和当前改动

原te_nut_phase_vision将CAD外表面随机采样后使用最近样本点距离。改变相机视角时像素采样变化，采样空隙改变角度代价最小点：旧两帧角227.275/227.800°，真实螺母几乎没动。当前候选改为直接对原CAD三角形连续表面求距离，使用现有Warp CPU BVH，仅静态源模型和图像/标定/编码器输入，不改变仿真物理/控制律/0.5°门/5°纠正上界。两帧新结果同为227.500°（0.025°搜索分辨率），原失败图像的手/螺母误差-0.221842°；16幅高/低已结束回合图像的原质量门全过，原校正后图像均在0.5°内。匹配核约0.058s，对比旧0.095s；包含深度处理的中位0.106s。不能据离线重放称整机成功。

SourceTriangleSurface位于src/kcg_connector/isaac/te_nut_phase_surface.py；只修改视觉距离。独立双精度穷举三角形投影/边距离与CPU BVH在540个点上最大差<2.8nm；Trimesh默认近表面容差并非精确参考。3项新单元测试通过，Isaac环境已安装与项目现有一致的pytest9.1.1。

诊断artifacts/nut_phase_surface_20260919；固定图像回放入口reproducibility/nut_phase_surface_20260919/replay_saved_frames.py；输出ended_frame_regression.json。此前失败真值只用于根因后评，未用于新估计。

## 当前验证状态与下一步

局部local_fourth_grasp_02/run已2026-09-19T14:17:13.333728Z结束，1175.338s、8503步、exit0，session11300已关闭。源接触区域及键槽几何后评均通过；视觉复核误差-0.021783°<原0.5°，实际重新抓握完成。local_01只有输出目录检查失败，未运行物理。控制修复已局部验证，完整高位成功仍未获得。

预检high_global1_names_preflight_04已195.684s通过全部必要门与视频隔离。唯一主回合high_global1_full_04（artifacts/two_key_20260919）于2026-09-19T14:26:51.279825Z启动，源码98ff9b213fef2003b1dfe43c61ca8b0d1cfb7274，PID1265467，工具session43043；预检session38425、局部session11300均已关闭。保持Body visual_body_contact_0p09.yaml、assembly_high_global1_plus20.yaml、原21600s/300s收尾、5fps仅五名称。只改变螺母视觉距离，控制/力/速/碰撞/5°修正/0.5°验收/360°/最多6次抓握门保留。一次一个主实验，加载源码配置冻结。停止用STOP_REQUEST，不在线改预算。

后评通用助手/tmp/kcg_high_g1_postreview.py需运行结束且terminal记录存在后调用（参数为新run路径），独立运行source_nail_body、high_global1、source_nut_pad、two_key、camera、transport、native_support、source_key_containment、3秒释放、source_band。随后必须提取并查看同回合最后旋紧/释放中间/最后帧，写final_mating_visibility_review，再用without_cyclic_gc做whole；不能拿局部成功代替全装配。旧label_five_view.py不得给录像加回其他文字。

本次修复报告reproducibility/nut_phase_surface_20260919/result_CN.md与status.json，证据evidence/。全轮成功后才更新two_key_20260919/delivery_status.json的高位完整成功标志，原失败证据不覆盖。

Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python，ISAAC_ENV_PREFIX同环境；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python。PYTHONPATH含src/kcg_connector、isaac、carts_v2；OPENBLAS_NUM_THREADS=1；离线后评用without_cyclic_gc。仅仿真hardware_authorized=false，无子代理、不推送、不删原数据。每次恢复先核对实际进程，不能等已退出的旧session。
