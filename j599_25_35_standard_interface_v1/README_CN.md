# J599/26FJ35PN + J599/20FJ35SN 隔离模型

本目录只服务于以下一对现实采购型号的公开标准接口建模：

- 活动端：`J599/26FJ35PN`，直式插头、25 号壳体（代码 J）、25-35 排列、128 个 22D 插针、N 键位。
- 固定端：`J599/20FJ35SN`，方盘壁装插座、25 号壳体（代码 J）、25-35 排列、128 个 22D 插孔、N 键位。

它不是仓库原有 25-61、61 芯模型的修改版。本目录不会引用或覆盖旧 USD，也不包含机械臂、三指手、桌面抓取或手指碰撞逻辑。

## 真实性边界

这是 `GJB599/J599` 与 `MIL-DTL-38999 Series III` 公共接口资料驱动的仿真模型。用户尚未提供制造商、实物测量或厂商原始 CAD，因此：

- 标准接口身份、128 个孔位、五键 N 键位和三头螺纹关系可追溯；
- 外壳非接口小倒角、滚花细节、内部材料分层、质心和惯量属于明确标注的仿真假设；
- 不能把仿真通过解释成用户手中产品的尺寸检验、质量检验或硬件装配验收；
- `hardware_authorized` 和 `hardware_exact_fidelity` 始终为 `false`。

## 预期输出

生成器将输出：

- `generated/j599_25_35_pair_visual.usda`：保留两个连接器、128 对触点和外观细节的视觉层；
- `generated/j599_25_35_pair_assembly.usda`：用于 Isaac Sim 的壳体导向、五键/五键槽、止挡及独立连接环物理层；
- `generated/j599_25_35_pair_assembly.usdc`：同一装配资产的二进制 USD；
- `generated/build_report.json` 与后续 `evidence/`：静态和 Isaac 动态原始证据。

动态验证只使用透明测试夹具的关节驱动，不使用机器人手指，不在物理开始后写连接器世界位姿，不使用磁吸或隐藏固定活动端。

当前权威输入是 [模型合同](config/model_contract.json)、[128 孔位表](data/contact_positions_25_35.csv) 与 [来源说明](docs/SOURCES.md)。

## 在 Isaac Sim 中使用

物理装配应导入：

```text
generated/j599_25_35_pair_assembly.usdc
```

只做外观检查时可导入：

```text
generated/j599_25_35_pair_visual.usda
```

装配资产的根 Prim 是 `/World/J599_25_35_N_Pair`。固定端配合面中心为装配坐标原点，`+Z` 指向尚未插入的活动端，主键中心线为 `+X`；名义装配时保持 N 键位对齐并沿 `-Z` 方向接近。`assembly.usdc` 已包含壳体导向、五键/五键槽、金属终止面和连接环物理代理；128 对电触点是视觉几何，不承担微观插针变形或电接触仿真。

验证器会在运行时创建透明的 connector-only 关节夹具。该夹具只用于证明资产能够在 Isaac 物理步进中旋合到位，不属于需要集成到机器人场景的连接器资产。

## 已完成验证

- 静态检查：`STATIC_PASS`；128 个插针与 128 个插孔编号均精确为 1–128，五键/五键槽齐全，触点碰撞 API 数为 0，机械臂/手指 Prim 数为 0。
- 名义 N 键位装配：`DYNAMIC_PASS`；端面误差 3.72 µm，7.62 mm/rev 螺纹关系误差 40.25 µm，保持漂移 21.35 µm，金属终止面接触样本 14,544，求解器和 PhysicsUSD 错误均为 0。
- 3° 错键负例：`DYNAMIC_PASS`；活动端被五键碰撞挡在终止面上方 7.240 mm，错键接触样本 1,248，金属终止面接触样本 0，求解器和 PhysicsUSD 错误均为 0。

错键负例使用上限 7.833 N 的独立 PrismaticJoint drive，并明确隔离螺纹 rack。报告中的 25.12 N 是瞬态接触反力峰值，不是驱动限力；该负例只证明五键碰撞会阻止错误全插入。最终接受证据见 [总摘要](evidence/final_validation_summary.json)、[名义装配报告](evidence/isaac_nominal_run_07/report.json)、[错键报告](evidence/isaac_wrong_key_run_03/report.json) 和 [接受文件哈希清单](evidence/SHA256SUMS.accepted.txt)。

## 复现命令

以下命令从仓库根目录运行。Isaac Conda 的库目录必须放在现有 ROS/Gazebo 路径之前，否则 Ubuntu 22.04 的系统 `libstdc++` 可能导致 `CXXABI_1.3.15` 启动错误。

```bash
isaac_prefix=/home/noob/WorkPlace/isaacsim/.conda-env
export LD_LIBRARY_PATH="$isaac_prefix/lib:${LD_LIBRARY_PATH-}"

"$isaac_prefix/bin/python" j599_25_35_standard_interface_v1/src/build_j599_25_35_assets.py
"$isaac_prefix/bin/python" j599_25_35_standard_interface_v1/src/validate_static.py
"$isaac_prefix/bin/python" -m unittest discover \
  -s j599_25_35_standard_interface_v1/tests -p 'test_*.py'

"$isaac_prefix/bin/python" \
  j599_25_35_standard_interface_v1/src/validate_isaac_assembly.py \
  --case nominal \
  --output-dir /tmp/j599_nominal_recheck

"$isaac_prefix/bin/python" \
  j599_25_35_standard_interface_v1/src/validate_isaac_assembly.py \
  --case wrong_key_3deg \
  --output-dir /tmp/j599_wrong_key_recheck
```

每个动态用例必须在独立 Isaac 进程中运行，并以对应 `report.json` 的 `passed` 和全部 gates 为准；不能以生成文件存在或进程退出码 0 代替动态判定。

## 不包含的验收

当前结果证明的是“公开标准接口模型在 connector-only Isaac 夹具中能够装配”。它不证明制造商专有 CAD 一致性、用户实物尺寸一致性、真实电接触性能，也不包含机械臂抓取、手指碰撞或真实硬件装配验收。制造商或实物测量数据到手后，应另开版本校准外形、质量、惯量和厂商专有细节，不能静默覆盖本公共标准版本。
