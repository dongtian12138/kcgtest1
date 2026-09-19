# 当前状态：螺母视觉误停已局部修复，整机初抓接触冲击仍未解决

核验时间：2026-09-19T14:46:34.521026+00:00。本轮用户问“问题出在哪、能否解决”。已定位并修复螺母角度估计的离散采样偏差，通过固定图像回放和实际局部抓握验证；随后新整机复验在更早的初次抓取触发手指力矩保护，未到螺母阶段。当前没有运行中的实验，不存在新的高位完整成功录像。

## 有效验收与基线

原完整验收仍为高位G1实际粗定位、抓取搬运、两次固定G2看键、插入、旋紧到位、最终3秒完全松手，只有同回合原物理门全部通过才可称成功。最近完整成功记录仍是低位G1的socket_plus20_full_01（df1afac）；原始数据和源码版本保留。高位full_03（8959158）曾通过初抓、两次对键、插入、本体松手和前三段旋紧，但在第四次螺母视觉复核停止。

工作树/home/noob/WorkPlace/kcgtest1-performance-20260917，分支codex/high-global1-full-20260919。修复代码a308389，完整复验执行98ff9b2；原项目只同步本入口，不动其他脏资产。

## 已完成的修复与验证

原te_nut_phase_vision把原CAD表面采成随机点，用最近样本点距离找角度；换相机视角时深度采样落点改变，点间空隙造成角度代价偏差。原静止Nu两帧227.275/227.800°相差0.525°。

候选改为到原CAD三角形连续表面距离，使用已有Warp CPU BVH。仅静态CAD、当前深度、相机标定/编码器和图像Body位置/轴线输入，不读在线对象真值。不改物理模型/材料/质量/960Hz64/4求解/控制律/力/速/5°纠正上界/0.5°视觉验收门。独立双精度穷举三角形投影/边距离与CPU BVH在540点上最大差<2.8nm。三项几何单测在Isaac Python实际通过；普通规划环境无Warp时按环境跳过这些Isaac依赖测试，必要验证已在真实依赖环境完成。

16张高/低已结束回合图像重放的原质量门全部通过，原失败两帧在0.025°搜索分辨率下都为227.500°，原校正后图像误差-0.221842°；所有校正后图像均<0.5°。匹配核约0.058s（旧0.095s），含深度处理新中位0.106s，不等于整机提速承诺。

局部local_fourth_grasp_02/run以封存开放手指状态仅在reset前初始化，实际完成重新观测/校正/抓握：1175.338s、8503步、exit0，新图像复核误差-0.021783°，原接触区域及键槽几何后评都过；不包含后续旋紧或最终释放，不冒充完整装配。该局部session11300已结束。

## 新整机复验与当前阻塞

预检high_global1_names_preflight_04已195.684s通过。high_global1_full_04于2026-09-19T14:34:37.799010Z结束，466.521s、39843步、exit2；session43043/PID1265467与预检session38425均已关闭，不再等待。

full_04初抓parallel_contact_approach末步39842，第二指f2j1反力矩2.493268Nm>原0.9Nm，触发HAND_MEASURED_EFFORT_ABORT，未抬升，未运行螺母估计。原生接触记录f2/Body正冲量约0.182597N·s，部分点负间隙约0.106mm。只能确认接触冲击，唯一根因尚未知；不能径称纯数值问题或靠增力绕过。

full_03/full_04初始图像抓取位姿完全相同、均0.09rad/s；一轮初抓通过、一轮失败。物理视觉等待分别25874/24881步，是可核对差别但不是已证根因。此前降速没有根治初抓敏感性。初抓控制相对8959158未改，新螺母函数尚未执行。

## 证据与恢复路线

修复和验证报告reproducibility/nut_phase_surface_20260919/result_CN.md，status.json和evidence/包含16帧、独立距离检查、局部原始门和full04停止证据。原始局部artifacts/nut_phase_surface_20260919/local_fourth_grasp_02/run；新整机artifacts/two_key_20260919/high_global1_full_04/run。原高位full03失败数据及其报告保留。工作过程索引docs/history/CURRENT_CONTEXT_CN_20260919_nut_surface_fix_worklog.md。

继续整机修复时，最早阻塞是初抓第二指第一次碰到Body的冲击；先对照成功/失败的实际接触几何、间隙、法向与指令/实测响应，找到可归因原因再单变量验证。不直接放宽0.9Nm、0.5°或几何门，不继续无依据降速/盲扫/原样整机重跑。新完整高位成功仍需原各阶段后评、三秒释放与同回合仅五名称录像。

配置仍为reproducibility/two_key_20260919/visual_body_contact_0p09.yaml与assembly_high_global1_plus20.yaml，固定四台功能相机+记录主视角。源模型、360°总预算、最多6次抓握及用户冻结边界保留。仅仿真hardware_authorized=false，一次一主实验，不使用在线真值、不在启动后改物体位姿、不删原证据、不推送、无子代理授权。

Isaac Python /home/noob/WorkPlace/isaacsim/.conda-env/bin/python，规划Python /home/noob/WorkPlace/kcgtest1/.venv/bin/python，OPENBLAS_NUM_THREADS=1，PYTHONPATH含src/kcg_connector、isaac、carts_v2。后评用without_cyclic_gc。每次恢复核对真实进程，不凭旧运行文字继续等待。
