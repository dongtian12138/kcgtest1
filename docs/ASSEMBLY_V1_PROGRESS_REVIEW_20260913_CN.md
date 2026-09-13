# 第一版视觉装配进度检查

核验日期：2026-09-13。**已按用户要求在本轮结束后停止工作并完成收尾，未启动新回合。** 主程序已退出，物理动作、独立复核与录像封存均已结束。

**结论：尚未完成电连接器的全过程装配。** 这次已从桌面新图像开始，在同一回合实际完成指甲取件、搬运和对键插入、松手自然支承、换抓螺母、分段旋合。最终本体深度 **10.148 mm**，距原 **14.605 mm** 止挡还差 **4.457 mm**；停止后已张开手，独立2秒窗口保持通过。

## 本轮完成到了哪里

| 环节 | 本轮实际结果 | 证据 |
|---|---|---|
| 当前视觉引导原三指甲抓本体 | 保持时抬升至少56.020 mm；2秒内三根指甲每步都承载本体，桌面接触为0；源指甲面复核通过 | [指甲抓取复核](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/source_nail_body_review.json) |
| 抓后识键、搬运后重新观察、观察插座槽并对准 | 本回合图像用于动作；实际低力入槽到7.922 mm，插入过程中无手—插座正载 | [插入物理结果](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/physical_key_entry_result.json)、[本轮视觉与运输记录](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/socket_transport/transport_and_observation.json) |
| 完全松开本体后的自然支承 | 最终深度7.965 mm；保持期间手指无接触、深度不变，五键前端仍在槽内 | [自然支承复核](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/support_posthoc_v3.json) |
| 掌面换位、指腹抓螺母及两次重新换抓 | 实际掌面70°→60°；已执行首抓和两次松开、回腕、重新观察、再夹持。夹持/旋拧承载点均为原PAD，无手—Body/Socket/其他对象正载 | [原指腹复核](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/source_nut_pad_review.json) |
| 真实分段旋合 | 前两段完成，第三段在传动余量停止；实际转角和推进见下表 | [真实运动端点](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/motion_endpoints_review.json) |
| 源键槽约束 | 复核通过。最大侧壁数值残差1.979 µm，处于此前固定的2 µm后评价带内；未更改几何或该判据 | [源键槽复核](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/source_key_containment_review.json) |
| 停止后的张手与保持 | 最终连续2秒手接触为0，深度变化0.477 µm；停留在未完成的中间插入状态 | [最终保持窗口](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/terminal_hold_window_review.json) |
| 原机械完整到位 | **未通过。** 最终止挡接触为0，最终窗口内同时承载的簧套组数为0，未达到128组完整装配条件 | [最终保持窗口](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/terminal_hold_window_review.json) |

“指腹复核”的范围是螺母接触、夹持和旋拧阶段；最终脱手窗口另行核对。[整轮独立汇总](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/whole_assembly_review.json)确认437275步连续、各视觉阶段成立、源面/键槽复核通过，同时明确完整装配为false。不能把源面通过、单个控制段完成或最初抓取的PASS，当成完整装配成功。

## 三段旋拧的真实结果

转角来自运行后读取的实际物体姿态；下列本体推进量覆盖各段转动及其末段保持。没有用手腕指令替代螺母转角。

| 段落 | 手腕指令 | 螺母实际转角 | 手部实际转角 | 本体实际推进 | 本体段末深度 |
|---|---:|---:|---:|---:|---:|
| 首段 | 20.000° | 19.979° | 19.991° | 0.032 mm | 8.002 mm |
| 第一次换抓后的90°段 | 90.000° | 87.805° | 89.977° | 1.402 mm | 9.400 mm |
| 第二次换抓后的续拧段 | 37.470°后停止 | 35.453° | 37.357° | 0.749 mm | 10.146 mm |

最终卸载、张手后深度为10.148 mm。从首段转动起点到最终脱手，螺母净转角约142.427°；它与三段转角直接相加略有不同，因为换抓和卸载期间也有小幅运动。

**这轮确认了持续旋合推进这一实际进展。** 之前完整回合08停在第二段约3°附近；局部诊断20虽然完成过90°，采用的是已插入源状态。本轮22从桌面新视觉开始，实际接续完成了90°段，并进一步推进到第三段。没有跨回合拼接。

## 卡在什么地方

直接停止事件发生在step432029：第三指模型弹性传动力矩为 **3.450013 N·m**，距离3.5 N·m硬边界只剩 **0.049987 N·m**，触发本轮预设的0.05 N·m余量停止。

同一时刻三指模型传动力矩约为 **[2.6600, 3.2349, 3.4500] N·m**。控制器保留了停止原因，随后依据当前图像检查导向，按最后实际发送的目标卸载并张手；没有越过硬边界，也没有恢复旋拧。

第三段腕部滤后横向力峰值约 **2.971 N**，扭矩峰值约 **0.658 N·m**。这表明当前握持/载荷分配和有限传动条件下，第三指先触及保留余量的停止线。为何它承担了更多载荷，以及怎样继续完成更深的旋合，尚未在本轮收尾中进一步试验。

这些传动上限、刚度及摩擦参数是当前仿真设定，尚未全部完成硬件辨识；不能据此宣布真实三指手做不到，也不能直接通过提高上限宣称问题已解决。当前证据不要求再重建整只手或连接器。

停止原件：[旋拧停止记录](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/socket_transport/nut_rotation_continued_01/nut_rotation_controller_result.json)、[按停止状态卸载的记录](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/socket_transport/nut_terminal_release/nut_reindex_controller_result.json)。

## 到最终目标仍缺少什么

1. 在现有机构和受控载荷条件下越过本次第三指余量停止，继续产生真实螺母转动和本体推进。
2. 实际到达14.605 mm原止挡，并满足原簧套等完整机械状态。最终高负载阶段仍未由三指手整机验证。
3. 在**完整到位状态**再完成松手和至少2秒保持。目前已证明的是中间插入状态下的脱手保持。
4. 补齐能清楚检查最终红带的侧向近景证据。本轮最终四视角画面不足以独立判定红带是否完全遮挡；机械深度已经明确未到位。

后续若用户要求恢复，首先复核本次停止附近的实际接触、抓位和三指传动负载，区分载荷分配、抓内迁移和传动边界的作用，再决定一个有依据的局部改动。**本次不执行这些后续工作。**

## 检查视频

[本轮完整四视角原始视频](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/video/assembly_four_view.mp4)；[停止后的同回合画面](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/video/stopped_state_review.jpg)。

| 视频位置 | 内容 |
|---|---|
| 00:16–00:37 | 指甲抬升本体和保持 |
| 00:37–01:43 | 搬运、重新识键、观察插座槽并对准 |
| 01:43–03:07 | 接近与低力入槽 |
| 03:07–03:10 | 卸载本体、张手、自然支承 |
| 03:10–03:43 | 掌面换位与首次指腹抓螺母 |
| 03:46–04:06 | 首段20°旋拧和保持 |
| 04:06–04:31 | 第一次松开、回腕、再夹持和准备 |
| 04:31–05:57 | 完整90°旋拧段和保持 |
| 05:57–06:52 | 第二次松开、回腕、再夹持和准备 |
| 06:52–07:30 | 第三段旋拧，随后触发余量停止 |
| 07:30–07:36 | 最终卸载、张手与保持 |

时标来自同回合逐帧记录，视频不拼接不同回合。精确阶段时标见[视频阶段索引](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/video_phase_timeline.json)。最终红带审阅的限制见[影像复核](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/final_mating_visibility_review.json)。

## 版本和数据在哪里

- 分支：`codex/visual-assembly-v1`。本轮运行代码：**8e92313**；后续收尾文档提交不改变该运行版本。
- 正式入口：`src/kcg_connector/isaac/run_body_assembly_with_video.py`；当前Body、任务和Nut抓位配置均由本轮清单绑定。
- [精确运行命令](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22_command.json)、[运行源码清单](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22_source_manifest.json)、[启动后源码一致性复核](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/source_binding_closeout_review.json)。
- [本轮全部数据目录](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22)：437275条原始物理记录已封存，末条step437274；压缩真值约12.2 GB。原始失败和停止记录保留，不删旧artifacts，不推送远端。
- [收尾清单](/home/noob/WorkPlace/kcgtest1/artifacts/visual_assembly_v1/visual_complete_task_preload22/closeout_manifest.json)绑定本轮代码、各独立复核、录像完整性及停止状态；运行和复核进程均已结束。
- 完整录像1920×1080、5帧/秒、2278帧、455.6秒，逐帧索引连续，完整解码检查通过。渲染前后被记录的关节、物体和腕力状态变化均为0，物理时钟逐帧不变检查通过。
- 主程序退出码为2；引擎健康与身份一致性检查均为true。顶层退出码仍由旧抓取研究评价决定，本次新视觉轨迹明确不冒称由名义预检直接认证，因此旧的accepted_preflight_bound为false；这不是引擎崩溃。最初抓取的物理检查通过，完整装配则按独立机械汇总明确未通过。保留这些原始标记，后续入口汇总语义需与完整装配验收统一。
- 当前活动入口：[CURRENT_CONTEXT_CN.md](/home/noob/WorkPlace/kcgtest1/CURRENT_CONTEXT_CN.md)。旧活动文本已保存在[收尾前历史](/home/noob/WorkPlace/kcgtest1/docs/history/CURRENT_CONTEXT_CN_20260913_before_episode22_closeout.md)。
- 旧资产依赖仍按[USD依赖清单](/home/noob/WorkPlace/kcgtest1/docs/assembly_v1_usd_dependencies_20260912.json)保留。未对旧目录进行清理。
