独立有界复核（2026-09-17 UTC）：**本回合 source_key_containment_review.json 的 accepted=true 有一致的输入绑定和最差见证支持，原 2 µm 数值后评带没有放宽。** 仅重新读取最差见证及相邻三帧，没有重扫 25.52 GB 全流、启动物理或修改原报告/验收规则。本说明生成时 whole_assembly_review.json 尚未出现，整体验收待定。

本回合封存索引为 294560 帧，末偏移 25520469055 与原始文件大小相等，motion_timing.json 存在。键报告起点 146350 与本回合 key_entry_controller_result.contact_first_step 一致；报告末帧 294559 正好是索引末帧，末深度 14.604580849409077 mm 与独立最终释放补核一致。

评估器使用 iter_truth_fields 从 146350 连续读到 294559，读取器核对封存大小、字段完整性、逐帧顺序及最终计数。原来的有效域保持为 key_probe_/nut_index_ 相位，以及已进入槽口之后的键顶点；不是对从桌面开始的全部 294560 帧、每个键都机械套用槽内条件。原报告每键有效样本数为 135615/135551/135589/135726/135760，五键完整轴向进入的样本数为 113350。本次核对入口和这些保存计数，没有独立重算全流计数。

源模型 SHA256 为 `234b7c0f14caeae1b8dffeabb8bae5a63e8b05d2463f3a7ff3e0c0d1d53f9eb6`：同时匹配本回合 trace_metadata.evidence_binding、frozen_model_installation 指定文件及原完整 14 的源模型文件。实际使用的回退槽壁尺寸文件为 reproducibility/assembly_20260916/key_backlash_dimensions.json，SHA256 `aa09cd6133f4a54cc3aca731e9c723826d67432eea43325dde8a24e7024e6dc0`；其 JSON 内容与原始尺寸文件一致。该尺寸文件和当前键评估源码均匹配本回合绑定的 portable manifest，manifest 自身哈希也匹配 reproduction_plan。当前与原完整 14 报告的 tolerance 都为 0.000002 m。

| 键（从 1 编号） | 原报告全有效域最小源侧壁间隙 |
|---|---:|
| 1 | −0.910355160 µm |
| 2 | −0.651431701 µm |
| 3 | −1.848419871 µm |
| 4 | +101.983431955 µm |
| 5 | −0.859048214 µm |

用原五个源键网格顶点、原槽壁平面和本回合归档 Body 位姿重算最差位置附近，第三键结果为：

| 原始帧 | 相位 | 重算第三键最小间隙 |
|---|---|---:|
| 230007 | key_probe_nut_rotation_turn | −1.742738799 µm |
| 230008 | key_probe_nut_rotation_turn | −1.8484198708327293 µm |
| 230009 | key_probe_nut_rotation_turn | −1.643789927 µm |

230008 的值与原报告浮点值完全相等，仍有 0.151580129 µm 到原 2 µm 边界。负间隙原样保留；通过的是既有数值后评带，不能表述为源几何严格零穿透。原报告本身没有 run_path 或整份归档 digest 字段，本补证通过同回合起止索引、模型/尺寸/源码绑定和精确见证复算建立关联，没有声称重新验证了全部 25 GB 数据内容。

对当前汇总结论的检查：专用 source_nail_body_review、source_nut_pad_review、final_mating_visibility_review 均 accepted=true，键进入报告为 DYNAMIC_PASS；本代理上一项独立末段审核已经证明 3 秒逐帧零手冲量、原 StopBox 与同 128 簧套承载及实际深度到位。未在这些已完成的适用报告中发现被忽略的失败项，但仍不提前宣布完整汇总通过。

新生成的通用 evaluation.json 必须保留其不同范围：pickup_evaluation_sample_scope 明确只评前 35649 帧，whole_assembly_acceptance=false。它仍有 full-pad 身份未提供、旧三阶段/研究门未通过，以及 NO_TERMINAL_CONTACT、nail_body_stability=false 等与专用原指甲源面报告不一致的摘要；因此不能说“所有 JSON 都 PASS”。这些通用字段不能直接用来替代当前原指甲/两甲一腹的专用源面验收，也不宜从摘要反推实际没有抓住。其实际有限性、通道/API、真值隔离、身份绑定、引擎健康及 controller_nominal_physical_pass 均为 true，原始碰撞计数四类均为 0，PhysX error 列表为空。本有界审查没有继续追溯通用适配器的全部字段；整体结论由当前 whole_assembly_review 与适用专用证据合并判断，旧未通过标记保留并明确范围。

文件：

- [输入绑定、三帧重算及汇总快照](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01/review/independent_key_witness_check.json)
- [有界复算脚本](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01/review/independent_key_witness_check.py)，本次由 Isaac Python 离线运行，只读 230007..230009。原模型和尺寸文件只做输入绑定检查，无 SimulationApp。
- [同回合独立末段补核](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/full_validation/contact_last_gc128_repeat01_restart01/review/independent_terminal_release.json)
