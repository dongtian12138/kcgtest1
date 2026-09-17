独立末段只读复核（2026-09-17 UTC）：**本回合末段已进入原机械到位范围，原止挡承载，并真正松手保持完整 3 秒。** 该结论来自实际归档深度、原始手部冲量及原机械接触，不以 controller completed 或单独“稳定”代替坐底。本报告不代替全过程 Body/Nut 源面、键槽、逐段进展与影像检查，也不声明所有原检查已通过。

运行目录：`/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01/run`。读取前已核对 motion_timing.json 存在；归档为 294560 帧，索引末偏移 25520469055 与文件字节数相等。最终释放结果 completed=true、outer_abort_reason=null、failure_reason=null。使用原末段补核入口，输出只写 run 外的 review；未启动物理，未改源码、原始数据或验收规则。

| 末段范围与物理记录 | 实测 |
|---|---|
| 审核范围 | [290720,294560)，3840 帧 |
| 卸载 | [290720,290912)，192 帧 / 0.2 s |
| 张手 | [290912,291680)，768 帧 / 0.8 s |
| 完全张手保持 | [291680,294560)，2880 帧 / 3.0 s，dt=1/960 s |
| 原始手部冲量范数之和 | 保持区间每帧精确为 0，没有冲量阈值过滤 |
| Body 深度范围 | [14.604580849409077,14.604580849409077] mm，保存精度下恒定 |
| 距 14.605 mm | 欠 0.419150590923 µm，处于原 10 µm 到位范围内 |
| 原 SourceMetalStopBox 正载 | 每帧 8 组接触报告；any/every 均为 true |
| 原簧套承载 | 每帧同一组 128 个，集合种类数为 1 |
| 原止挡与 128 簧套同时承载 | 全部 2880 帧均为 true |
| Body/Nut 休眠标记 | 全部 2880 帧为 [false,false] |

全部 3840 帧的 all_provided_native_report_points_retained 与 contact_report_channels_agree 均记录为 true，physics_step_callback_count 均为 1，逐帧编号连续。止挡/簧套正载沿用原接触组冲量范数和 >1e-10 N·s 判据；没有将这个阈值用于“手部精确零冲量”的检查。“8 组”指接触报告分组，不是声称存在 8 个独立止挡零件。

卸载/张手 960 帧仅观察到下列正载手部演员对 CouplingNut 接触；没有意外演员、未分类正载点或允许区域外点。

| 手部演员 | 实际源面 | 正载点数 / 涉及帧数 | 最后正载帧 | 最大源面投影残差 |
|---|---|---:|---:|---:|
| f1Link3 | 原指甲 | 358 / 358 | 291077 | 2.956510 µm |
| f2Link2 | 原指腹 | 378 / 332 | 291075 | 10.614268 µm |
| f3Link3 | 原指甲 | 542 / 404 | 291126 | 30.183568 µm |

三者都没有超过原 500 µm 投影范围的未确定点。源网格哈希与原接触区域合同一致，分类继续使用 selected_two_nail_geometry 的允许区域；没有改为更宽的手部源面。所有最后正载点都发生在保持起点 291680 之前。

末段被动推力轴坐标绝对最大值为 0.516086508873 mm，低于原后备限位检查线 0.5999 mm。这一结果仅覆盖被审核的末段，不能替代全回合后备限位/其他接触检查。

原 any/max 检查只证明存在某帧承载或某帧达到最大簧套数；本次补核实际遍历了完整 2880 帧，并证明每帧原止挡与同一组 128 簧套同时正载。它补充覆盖而没有另改验收标准。没有因缺记录或休眠推断承载；本次相应记录完整且对象未休眠。

本次执行命令（工作目录为改进树）：

```text
PYTHONDONTWRITEBYTECODE=1 /home/noob/WorkPlace/isaacsim/.conda-env/bin/python reproducibility/improvements_20260916/review_completed_assembly.py artifacts/full_validation/contact_last_gc128_repeat01_restart01/run --part terminal --output artifacts/full_validation/contact_last_gc128_repeat01_restart01/review/independent_terminal_release.json
```

精确结果见 [独立末段 JSON](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01/review/independent_terminal_release.json)。该调用通过索引读取本次最终释放区间，未重新遍历整份 25.52 GB 归档；没有使用旧回合或冷重建状态补足本次证据。全过程与实际视频可见性由并行审查另外判断。
