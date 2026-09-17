第二轮独立末段复核（2026-09-17 UTC）：**本轮末段进入原机械到位范围，原 StopBox 与同一组 128 个簧套逐帧承载，并真正松手保持完整 3 秒。** 此结果独立来自第二轮归档，没有沿用首轮通过结论；不以 controller completed 或单独稳定保持代替坐底。

读档前确认 motion_timing.json 存在，295421 帧，归档字节数和索引末偏移均为 26530655597。最终 release completed=true、outer_abort_reason=null、failure_reason=null。现有审核入口只读取本轮索引末段，不重跑物理，不改源码、阈值或原始记录。

| 核验项 | 本轮实测 |
|---|---|
| 审核范围 | [291581,295421)，3840 帧 |
| 卸载 | [291581,291773)，192 帧 / 0.2 s |
| 张手 | [291773,292541)，768 帧 / 0.8 s |
| 完全张手保持 | [292541,295421)，2880 帧，dt=1/960 s，恰为 3 s |
| 保持阶段原始手部冲量范数和 | 每帧精确为 0，没有阈值过滤 |
| Body 深度范围 | [14.60440203547475,14.60440203547475] mm，保存精度下恒定 |
| 距 14.605 mm | 欠 0.597964525 µm，处于原 10 µm 到位范围 |
| 原 StopBox 正载接触报告 | 每帧 9–11 组，any/every 均为 true |
| 正载簧套 | 每帧 128 个，始终同一组；集合种类数为 1 |
| 止挡与 128 簧套同时正载 | 全部 2880 帧成立 |
| Body/Nut 休眠 | 全部保持帧均为 [false,false] |

全部 3840 帧连续，原生点完整保留、接触通道一致、一次物理回调的记录标志均为 3840/3840。止挡/簧套正载沿用原接触组冲量范数和 >1e-10 N·s 判据；该阈值没有用于手部“精确零冲量”判断。9–11 指接触报告分组数，不是新增止挡零件数。原 any/max 只证明存在承载；本次实际检查了完整 2880 帧的同时承载和同组簧套，未另改验收标准。

卸载及张手的 960 帧中，正载手接触演员仅为下列三个原源面，对方均为 CouplingNut：

| 演员 | 实际源面 | 正载点数 / 涉及帧数 | 最后正载帧 | 最大源面投影残差 |
|---|---|---:|---:|---:|
| f1Link3 | 原指甲 | 446 / 376 | 291960 | 9.735847 µm |
| f2Link2 | 原指腹 | 868 / 289 | 291872 | 4.574807 µm |
| f3Link3 | 原指甲 | 772 / 407 | 291988 | 26.176558 µm |

允许区域外点、超过原 500 µm 投影范围的不确定点、意外演员和未分类正载点均为 0。所有最后正载帧均早于完全张手保持起点 292541。

本次末段被动推力轴坐标绝对最大值为 **0.511408259642488 mm**，低于原后备限位检查线 **0.5999 mm**。它证明末段未达到该检查线，不能替代全回合的后备限位审查。

精确数据见 [本轮独立末段 JSON](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat02/review/independent_terminal_release.json)。执行命令（工作目录为改进树）：

```text
PYTHONDONTWRITEBYTECODE=1 /home/noob/WorkPlace/isaacsim/.conda-env/bin/python reproducibility/improvements_20260916/review_completed_assembly.py artifacts/full_validation/contact_last_gc128_repeat02/run --part terminal --output artifacts/full_validation/contact_last_gc128_repeat02/review/independent_terminal_release.json
```

结论范围仅为最终卸载、张手与 3 秒保持。全过程 Body/Nut 源面、原 2 µm 键槽、转角进展、影像/红带和 whole 汇总仍由并行审查决定，不能把本次末段通过当成第二轮全部检查已经通过。
