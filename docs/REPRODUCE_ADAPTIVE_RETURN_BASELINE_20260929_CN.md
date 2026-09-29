# 按关节余量选择回转的四相机完整装配基线

当前状态：**VERIFIED**。本轮实际完整执行、全部数值/影像分项及整体审查已通过。详细结果见 `reproducibility/adaptive_return_baseline_20260929/result_CN.md`，录像与证据包随本次固定发布提供。

本分支：`codex/connector-adaptive-return-baseline-20260929`。实际完整回合执行提交：`6df30f18308bcad767bc0b5f46ef80089f7a6bf8`。本次从桌面初始场景重新运行，没有恢复已插入的记录状态。

## 1. 下载与准备

```bash
git clone --branch codex/connector-adaptive-return-baseline-20260929 \
  https://github.com/dongtian12138/kcgtest1.git kcgtest1-adaptive-return
cd kcgtest1-adaptive-return
python3 scripts/prepare_assembly_reproduction.py --assets
```

继续使用版本化发布包中的297个运行输入和Git中的2个高位相机输入。原模型、原几何/质量/材料/驱动限制均保持。准备脚本会核验资产，不能把本机旧实验目录当成新机器的依赖。

三个独立Python环境、固定SAM源码/权重、记录模块编译和ROS机器人资源构建步骤沿用[环境与依赖准备说明](REPRODUCE_FOUR_CAMERA_BASELINE_20260920_CN.md)第2、3节。本次实际环境版本另见 `reproducibility/adaptive_return_baseline_20260929/environment_versions.json`。使用已存在且匹配的环境时，设置：

```bash
export ISAAC_ENV_PREFIX=/实际的/isaac环境
export KCG_PLANNER_PYTHON=/实际的/规划环境/bin/python
export KCG_SAM6D_PYTHON=/实际的/视觉环境/bin/python
```

规划环境使用Python3.10、NumPy1.23.5、SciPy1.10.1及Tesseract0.35.0.7。Isaac和SAM环境要求见上述说明。跨机器应重新编译两个原生记录模块并构建当前克隆的ROS资源；不能照搬另一台电脑的二进制或绝对路径。本次没有执行第二台机器上的完整物理回合。

## 2. 本次改动

默认启动配置为 `reproducibility/adaptive_return_baseline_20260929/assembly_task.yaml`。与原高位四相机基线相比，只启用此前已做局部动态验证的回转选择和重抓朝向保留：

- 张手后评价0°、+90°、−90°三种与源螺母抓取相位相容的候选。
- 检查回转路径的关节、限速与碰撞，并预览实际下一段旋拧的关节可达性。
- 可不回转时优先省去空转；需要回转时保留原关节边界，并检查其他候选。
- 下一次抓握保留已选相位，继续执行原视觉复核和原受力保护。

下一段的预览只判断关节可达性，不能代替实际重抓、碰撞和受载控制。0/±90°是当前源螺母外形对应的有限候选，不是所有物体或机械臂的通用最优策略。

四相机、两次观键、原抓位/预载/柔顺控制、960Hz/64/4求解、360°总指令预算、最多6次抓握以及原接触/键槽/力速/释放标准保持。集成来源与变更范围见 `reproducibility/adaptive_return_baseline_20260929/integration_scope.json`。

## 3. 检查与新运行

```bash
python3 scripts/run_current_hand_assembly.py --check
python3 scripts/run_current_hand_assembly.py --run
```

`--run`先执行本次新预检，原四项预检条件全部通过才开始完整装配。默认结果目录是独立时间戳目录；可用 `--output-root`指定尚不存在的目录。预检300秒、收尾30秒；完整回合21600秒、收尾300秒。不要在运行中延长预算或覆盖旧结果。

五视角原视频在 `本次目录/run/video/assembly_five_view.mp4`，帧与物理步绑定记录在同目录的 `assembly_five_view_frames.jsonl`。需要停止时在 `本次目录/run/STOP_REQUEST` 写入说明，等待原机制收尾。

## 4. 必须按新回合验收

```bash
python3 scripts/review_current_hand_assembly.py /本次目录/run
```

数值分项通过后，亲自查看 `postrun_evidence/` 中的末次旋拧、释放保持中段和末帧，确认主视角/全局2没有露出的红色指示带，再执行：

```bash
python3 scripts/review_current_hand_assembly.py /本次目录/run --confirm-band-hidden
"$KCG_PLANNER_PYTHON" \
  reproducibility/adaptive_return_baseline_20260929/review_return_policy.py /本次目录/run
```

验收要求至少同时满足：`whole_assembly_review.json` 的 `complete_visual_assembly_verified=true`、`three_second_release_review.json` 的 `accepted=true`、`adaptive_return_policy_review.json` 的 `accepted=true`，并保留实际查看的同回合影像。回转审查还会报告本轮是否实际选择过0°、实际转角和回转期间的机械臂余量。

底层通用CARTS评估仍保留其旧的全指腹分类字段，本轮与旧完整基线都曾返回退出码2。原输出不改写。不能仅凭退出码2判定物理装配失败，也不能仅凭控制器`completed`判定成功；必须执行上面的当前源指甲/指腹、键槽、支撑、释放与全程独立审查。

## 5. 证据与保留

本轮记录位于 `artifacts/adaptive_return_20260929/full01/run`，小体积验收证据随本分支保存于 `reproducibility/adaptive_return_baseline_20260929/evidence/`。完整原始逐帧档案保留在本机；它不是启动新回合的输入，也不包含在小体积发布证据包中。

原视频另有本机独立副本 `/home/noob/Videos/kcgtest1/20260929_adaptive_return/assembly_five_view_full01.mp4`；具体字节数和校验摘要见 `video_preservation.json`。复制没有剪辑或拼接。后续清理应将此副本、成功回合原始数据和已发布基线识别为保留资产。

本结果仅涉及一个固定名义场景中的仿真。没有执行硬件测试，也不把一次完整通过或软件短行程回归当作任意极端工况的成功保证。
