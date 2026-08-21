# 来源、推导与未知项

## 1. 型号解释与 J599 对应关系

用户给出的原始型号是 `J599/26FJ35PN` 与 `J599/20FJ35SN`。

公开的 J599 技术资料说明：J599 是 GJB599 系列主称；`20` 为方盘插座，`26` 为插头；`F` 为铝合金化学镀镍；`J` 为 25 号壳体；`35` 为 128 芯 22D 接点排列；`P/S` 分别为插针/插孔；`N` 为正常键位。资料还明确该系列使用三头螺纹和五条导向键。

- J599 使用说明书（厂商技术资料，制造商与用户实物是否相同未知）：https://ht693.com.cn/home/a/4/56pspa/resource/2024/08/20/66c40d1e2286b.pdf
- GJB599B/MIL-DTL-38999 互配与型号语法交叉核对：https://www.zsx-connector.com/upload/other/20241113/d71aef3d365209227cb1f9078503bbb7.pdf

结论等级：型号语法与公共接口系列可用于建模；厂商小外形、内部结构、批次公差不能从型号唯一确定。

## 2. 128 个触点坐标

`MIL-STD-1560C with Change 3` 图 9、印刷页 154–156 明确给出 25-35 排列的 1–128 号 X/Y 坐标，英寸值为控制值；第 156 页还给出 128 个 22D 接触件、工作等级 M。

- DLA 官方 PDF：https://weaponssupportapps.dla.mil/Downloads/MilSpec/Docs/MIL-STD-1560/std1560.pdf
- DLA 文档状态页：https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=36980
- 本次取得文件 SHA-256：`45edb45edeed726c1a7836c145ba9210a074da3765903f2c32d6717a296f9290`

逐孔转录结果保存在 `data/contact_positions_25_35.csv`。生成器使用英寸列乘以精确换算系数 0.0254；毫米列只作人工交叉核对。

## 3. `/20` 与 `/26` 外形和接口

- `MIL-DTL-38999/20H w/Amendment 1`：壁装法兰插座，DLA 状态页 https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=70615
- `MIL-DTL-38999/26G w/Amendment 4`：直式插头，DLA 状态页 https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=70618
- `MIL-DTL-38999` 一般接口要求和 Series III 壳体/键位图，DLA 状态页 https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=22497

仓库已有两份公开图纸 PDF，只读复用其标准尺寸，不复用旧 61 芯几何：

- `src/kcg_connector/assets/public_specs/mil_dtl_38999/dtl38999ss20.pdf`
- `src/kcg_connector/assets/public_specs/mil_dtl_38999/dtl38999ss26.pdf`

## 4. 质量

一份公开 J599 使用说明书列出：`J599/20FJ35SN = 96.5 g`、`J599/26FJ35PN = 75.4 g`。由于用户实物制造商未知，这两个数只作为代表性仿真质量，不作为购买件实测值。连接环与插头本体之间的质量拆分、质心和惯量由仿真几何计算，属于等效假设。

## 5. 明确未知或等效处理

- 制造商、批次、序列号与用户实物尺寸：未知。
- 厂商原始 STEP：未取得，不能声称使用厂商精确 CAD。
- 绝缘体内部台阶、橡胶密封具体截面、插孔簧片、压接尾端：只做视觉简化。
- 128 个触点不进入全组合刚体微碰撞；它们保留逐号视觉几何与同号配对。装配碰撞由连续壳体、五键/五键槽和止挡承担。
- 三头螺纹在装配控制层使用明确的 `7.62 mm/rev` 转角—轴向关系；不把简化关系说成真实牙面摩擦或锁紧扭矩验证。

因此最终结果只能称为“公开标准接口、连接器独立、Isaac Sim 仿真装配验证”。
