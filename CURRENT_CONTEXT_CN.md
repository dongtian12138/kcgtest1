# 当前任务：修复螺母换抓的跨视角角度误停

核验时间：2026-09-19T13:54:25.666388+00:00。用户问“问题出在哪、能否解决”，正在落实根因修复及局部验证；无主仿真实验运行。工作树/home/noob/WorkPlace/kcgtest1-performance-20260917，分支codex/high-global1-full-20260919。仅仿真，无子代理，不推送，不动原项目脏资产，只同步本入口。

## 原结果与验收

原完整成功基线仍为低位G1 socket_plus20_full_01（df1afac），高位最新full_03（8959158，6705.369s）在第四次螺母换抓前停于视觉误差0.521842°>原0.5°，最终到位/3秒松手未完成。高位定位、实际56mm抬升2秒、两次固定G2看键、插入、0.5秒本体松手、已执行的指腹接触均已独立通过。前轮报告reproducibility/two_key_20260919/high_global1_full_result_CN.md及full_status.json保留，旧session95957已结束。

## 已定位根因和当前改动

原te_nut_phase_vision将CAD外表面随机采样后使用最近样本点距离。改变相机视角时像素采样变化，采样空隙改变角度代价最小点：旧两帧角227.275/227.800°，真实螺母几乎没动。当前候选改为直接对原CAD三角形连续表面求距离，使用现有Warp CPU BVH，仅静态源模型和图像/标定/编码器输入，不改变仿真物理/控制律/0.5°门/5°纠正上界。两帧新结果同为227.500°（0.025°搜索分辨率），原失败图像的手/螺母误差-0.221842°；16幅高/低已结束回合图像的原质量门全过，原校正后图像均在0.5°内。匹配核约0.058s，对比旧0.095s；包含深度处理的中位0.106s。不能据离线重放称整机成功。

SourceTriangleSurface位于src/kcg_connector/isaac/te_nut_phase_surface.py；只修改视觉距离。独立双精度穷举三角形投影/边距离与CPU BVH在540个点上最大差<2.8nm；Trimesh默认近表面容差并非精确参考。3项新单元测试通过，Isaac环境已安装与项目现有一致的pytest9.1.1。

诊断artifacts/nut_phase_surface_20260919；固定图像回放入口reproducibility/nut_phase_surface_20260919/replay_saved_frames.py；输出ended_frame_regression.json。此前失败真值只用于根因后评，未用于新估计。

## 下一步

复用现有diagnose_saved_hand_wrench/te_source_stage_probe，新增显式stop_after_current_regrasp，只局部运行原第四次换抓并在完成后收尾，不进入旋紧。候选recipe在reproducibility/nut_phase_surface_20260919/local_fourth_grasp.json；source run为full_03，source step252521（已有开放手指的最后物理帧），sensor/grip stage nut_regrasp_continued_02，source rotation nut_rotation_continued_01。初始已发角度应以该旋转段终结记录核对。局部场景只在reset前使用封存初态，是诊断，不是同回合完整装配。模型/质量/材料/CPU960Hz64/4、原力/速/几何门保持。已有local source runner最大1500s适用于>180°阶段、无最终释放，计划20000步上界。必须先冻结提交，然后运行；只一个主实验。

局部通过后还需要完整高位同回合验收及仅五名称录像，不能拼接。完整仍需原source_nail_body、source_nut_pad、two_key、camera、transport、native_support、source_key_containment、最终3秒、红带图像+几何、whole通过。原低位成功不被候选替换。原始数据/失败证据保留，运行中代码配置冻结、停止用STOP_REQUEST、不改物体位姿/不使用真值控制。

Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python，ISAAC_ENV_PREFIX同环境；规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python。PYTHONPATH含src/kcg_connector、isaac、carts_v2；OPENBLAS_NUM_THREADS=1；离线后评用without_cyclic_gc。恢复先核对实际进程。
