# kcgtest1 当前上下文

## 当前实际结果

2026-08-28 在真实 Isaac Sim GPU physics 中，旧版带指甲三指手对
`current_d38999_26kj61sn_public_spec` 连接器完成了一次固定场景名义抓取：

- 三个末节碰撞体都与连接器接触，接触记录数分别为 5160、10660、10273；
- 连接器离开桌面，最大抬升 50.889 mm；
- 保持 2.0 s，保持期间没有重新接触桌面；
- 最大相对滑移 0.709 mm，最大姿态变化 0.0722 rad；
- 手—桌、手—夹具和未授权手—物接触记录均为 0；
- 控制器完成且没有触发安全失败。

以上是一次特定手模型、特定连接器和冻结初始条件下的 simulation-only 名义结果。
原始评价见
[`evaluation.json`](artifacts/kcg_connector/isaac/carts_v2_goal_20260827/visual_delivery_20260828/grasp_lift/evaluation.json)，
四张 Isaac 过程图见
[`visuals/`](artifacts/kcg_connector/isaac/carts_v2_goal_20260827/visual_delivery_20260828/grasp_lift/visuals/)，
学术风格汇总图见
[`figures/`](artifacts/kcg_connector/isaac/carts_v2_goal_20260827/visual_delivery_20260828/figures/)。

## 必须同时保留的证据边界

### 1. 这不是无指甲抓取

本次运行使用
`artifacts/kcg_connector/isaac/robot/handarm_keyed_v3_physical_r7/handarm.usda`。
三个末节碰撞体仍来自原始 `f1Link3_convex.stl`、`f2Link2_convex.stl`、
`f3Link3_convex.stl`，包含原有指甲几何。此前制作的无指甲网格因非流形、非水密，
没有成为可接受的 PhysX 运行碰撞体，也没有完成动态抓取。

因此，本次结果只能证明“三个旧版末节凸包接触并抬起”，不能证明“无指甲真实指腹抓取”。
评价器的 `pad_surface_identity_verified=true` 只证明接触来自末节上唯一启用且被命名为
pad 的碰撞体，不能证明接触点位于已经去掉指甲的表面。

### 2. 另一款连接器没有完成抓取

TE/DEUTSCH `D38999/26FJ35PN` 只做过自由落桌稳定性和装配 smoke，没有做三指动态抓取。
当前登记抓法也只包含本次连接器。换型号后的几何、质心和接触位置不同，不能直接复制本次
位姿和关节目标。

### 3. 当前方法不是最优算法，也没有完成鲁棒性验证

本次只实际比较了三个固定配置；当前方案只是“本次实际比较的三种配置中最佳名义仿真抓法”，
不是连续空间最优、全局最优或跨型号最优。

同一冻结条件下的获胜配置有 3/3 重复成功，但这只说明确定性重复性。没有系统执行初始位姿、
摩擦、质量/质心、关节误差、观测噪声、外部扰动或连接器型号变化。两个单变量对照还对很小的
横向位移和较低预载表现出敏感性，因此不能宣称鲁棒。

## 结论状态

- `nominal_research_dynamic_pass=true`：只指上述固定场景名义物理结果；
- `research_dynamic_pass=false`：现有评价合同中的完整研究门没有闭合；
- `formal_dynamic_pass=false`；
- `hardware_authorized=false`；
- 无指甲抓取：尚未验证；
- TE/DEUTSCH 跨型号抓取：尚未验证；
- 最优性：尚未证明；
- 扰动鲁棒性：尚未验证。

## 直接复现实验

下面两条命令会重新运行 Isaac，而不是播放已有图片。第二条会保存四个实际阶段画面，
但图片和物体真值不会进入在线控制。

```bash
cd /home/noob/WorkPlace/kcgtest1
export PYTHONPATH="$PWD/src/kcg_connector${PYTHONPATH:+:$PYTHONPATH}"
CARTS_ISAAC_PY=/home/noob/WorkPlace/isaacsim/.conda-env/bin/python
CARTS_REPLAY_DIR=/tmp/carts_nominal_replay

"$CARTS_ISAAC_PY" src/kcg_connector/isaac/carts_v2/run_grasp_lift.py \
  --mode preflight \
  --object-id current_d38999_26kj61sn_public_spec \
  --output-directory "$CARTS_REPLAY_DIR/preflight" \
  --omit-trace-json

"$CARTS_ISAAC_PY" src/kcg_connector/isaac/carts_v2/run_grasp_lift.py \
  --mode grasp-lift \
  --object-id current_d38999_26kj61sn_public_spec \
  --preflight-evaluation "$CARTS_REPLAY_DIR/preflight/evaluation.json" \
  --output-directory "$CARTS_REPLAY_DIR/grasp_lift" \
  --capture-visual-evidence \
  --omit-trace-json
```

复现后先检查连接器是否真实离桌、是否达到 50 mm、是否保持 2 s，再看程序状态字段。

## 下一项最短研究动作

若用户继续授权，先从可靠的闭合 CAD 同时生成无指甲可见网格和无指甲碰撞体，保持当前控制律
不变，只对当前连接器做一次直接复验。该步骤成功后，冻结算法和超参数，再对 TE/DEUTSCH
型号做一次零调参留出验证；最后才进入受控扰动实验。不得在这些直接证据之前恢复批量搜索、
新优化器、管理器或认证框架。
