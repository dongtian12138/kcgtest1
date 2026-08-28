# 无指甲三指双型号名义抓取基线

## 结论

**无指甲三指名义抓取在两个型号上动态成功。**

这里的“成功”只表示已有 Isaac Sim 动态运行中，三块完整蓝色指腹都与连接器接触，连接器离开桌面，抬升超过 50 mm，并保持至少 2 s。当前抓法只是 baseline，不是全局最优；已有扰动结果只是探索数据，不是正式鲁棒性结论；全部证据都是 simulation-only，不是硬件验证。

TE 型号已经参与过方法调整，因此这里只称为双型号动态基线和跨型号验证，不称为严格盲测或零调参泛化。

## 两型号原始结果

| 型号 | 三块完整指腹接触 | 离桌 | 最大抬升 | 保持 | 最大相对滑移 | 最大姿态变化 | 最大手指关节力矩 | 未授权接触 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 D38999/26KJ61SN | 是 | 是 | 55.732 mm | 2.000 s | 0.407 mm | 4.422° | 0.656 Nm | 0 |
| TE/DEUTSCH D38999/26FJ35PN | 是 | 是 | 55.379 mm | 2.000 s | 0.430 mm | 0.078° | 0.644 Nm | 0 |

两个运行都满足三指真实接触、完整指腹身份核对、离桌、50 mm 抬升和 2 s 保持。当前型号最大桌面穿透为 0.001239 mm，TE 为 0.000280 mm。

TE 运行的控制器完成了全部动作，没有报告动作失败；但它的实测抬升峰值加速度为 0.04643 m/s²，而运行前登记值为 0.000832 m/s²，因此“预登记加速度一致性”这一研究 gate 未通过，顶层 `formal_dynamic_pass` 仍为 `false`。这不否定已直接观测到的接触、离桌、55.379 mm 抬升和 2 s 保持，但限制了结论强度。

原始评价文件：

- [当前型号预抓评价](../nailfree_current_com_height_preload050_lift056_regression_run02/preflight/evaluation.json)
- [当前型号抓取抬升评价](../nailfree_current_com_height_preload050_lift056_regression_run02/grasp_lift/evaluation.json)
- [TE 预抓评价](../te_d38999_26fj35pn_final_shared_method_regression_run01/preflight/evaluation.json)
- [TE 抓取抬升评价](../te_d38999_26fj35pn_final_shared_method_regression_run01/grasp_lift/evaluation.json)

## 真实 Isaac Sim 图片

当前型号有一组更早的真实成功截图；它使用同一用户无指甲 STL、同样的三指完整指腹语义，但碰撞资产是早期 v1，而不是最终定量运行绑定的 v2 shrink-wrap 版本。该次运行抬升 51.353 mm 并保持 2 s。它只证明这次早期成功运行的实际画面，表中的定量值仍来自上面的最终共享方法回归运行。

- [三指有限夹持](../nailfree_three_current_nominal_run01/grasp_lift/visuals/02_three_finger_clamp.png)
- [离桌 20 mm](../nailfree_three_current_nominal_run01/grasp_lift/visuals/03_table_released_20mm.png)
- [最终抬升并保持](../nailfree_three_current_nominal_run01/grasp_lift/visuals/04_final_hold.png)
- [截图时刻与仿真真值](../nailfree_three_current_nominal_run01/grasp_lift/visual_evidence.json)
- [该截图运行的完整评价](../nailfree_three_current_nominal_run01/grasp_lift/evaluation.json)

**TE 成功运行没有保存图片。** 两次最终 TE 成功运行都使用了 `--omit-trace-json`，且没有使用 `--capture-visual-evidence`。现存 TE 图片来自首次失败的预抓诊断，不能作为成功证据，因此未收入本里程碑。按当前指令，本次封存没有重跑 Isaac 补拍，也没有用生成图或失败图冒充。这是当前里程碑唯一明确的证据产物缺口；TE 的动态数值证据仍保存在上述原始评价中。

## 可复现命令

下面的命令使用与两次成功运行相同的对象、无指甲手资产和控制入口，并把新结果写到 `/tmp`，不会覆盖已封存证据。加入 `--capture-visual-evidence` 会在复现实验时保存真实 Isaac 帧；这些命令在本次封存中没有重新执行。

```bash
REPO=/home/noob/WorkPlace/kcgtest1
ISAAC_PY=/home/noob/WorkPlace/isaacsim/.conda-env/bin/python
ROBOT_ASSET="$REPO/artifacts/kcg_connector/isaac/cad_robust_grasp_goal_20260828/nailfree_three_finger_direct_v2_shrinkwrap/handarm_nailfree_three_direct.usda"

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode preflight \
  --object-id current_d38999_26kj61sn_public_spec \
  --robot-asset "$ROBOT_ASSET" \
  --output-directory /tmp/kcg_nominal_current/preflight \
  --omit-trace-json

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode grasp-lift \
  --object-id current_d38999_26kj61sn_public_spec \
  --robot-asset "$ROBOT_ASSET" \
  --preflight-evaluation /tmp/kcg_nominal_current/preflight/evaluation.json \
  --output-directory /tmp/kcg_nominal_current/grasp_lift \
  --capture-visual-evidence \
  --omit-trace-json

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode preflight \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --robot-asset "$ROBOT_ASSET" \
  --output-directory /tmp/kcg_nominal_te/preflight \
  --omit-trace-json

env PYTHONPATH="$REPO/src/kcg_connector" "$ISAAC_PY" \
  "$REPO/src/kcg_connector/isaac/carts_v2/run_grasp_lift.py" \
  --mode grasp-lift \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --robot-asset "$ROBOT_ASSET" \
  --preflight-evaluation /tmp/kcg_nominal_te/preflight/evaluation.json \
  --output-directory /tmp/kcg_nominal_te/grasp_lift \
  --capture-visual-evidence \
  --omit-trace-json
```

## 与原始运行绑定的实现

两次最终运行绑定到同一组源文件和无指甲手资产：

| 内容 | SHA-256 |
|---|---|
| 抓法配置 | `079f2ddc802ef810a6ef989c0795cbecbdac22e5e128c30af1ae7b4a3335329c` |
| 控制器 | `5bc80caa5106d9647fad9945d80ffc076e3c9fd11e48c83e134e52a78803b14e` |
| Isaac 运行入口 | `5cf8ace7b91add7166814d66b1290c5f9c76aa7de97adc318000b797b9748850` |
| 运行后评价器 | `4582b1bc262f683dc5fe6b0f241732f75a559bad4b14a291721abdd040ae281c` |
| 三指无指甲 USD 资产 | `d4ec8ada91bf948824bdf2c50d007ef86e03834c234059e45ef4785493781747` |

配置中的合法接触语义是用户确认的**整块蓝色指腹**，不是两个三角片。程序中的接触点投影也对整块指腹表面执行。后续有限搜索与这个 baseline 是两个问题：本里程碑证明可执行名义抓法存在，不证明搜索器已经能发现它，更不证明连续抓取空间的全局最优。
