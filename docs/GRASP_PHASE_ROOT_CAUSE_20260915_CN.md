# 本次仿真装配结果与数值检查说明

2026-09-16T17:30:56.551522+00:00 更新。**当前连接器与有限四杆、自锁三指手，已经在同一次连续仿真中完成桌面抓取、入槽、旋拧到位和真实松手保持。原键槽数值比较带仍有一项未通过，因此不能说所有原检查全部通过。**

本次不是冷初始化末段，也不是拼接几次实验。从桌面开始的279,682个物理记录连续保存，模型、材料、质量及既定硬力矩上限保持原配置。

最直接的结果是：最终本体深度 **14.604491 mm**，目标14.605 mm。最终张手后的连续3秒，全部手部原始接触冲量逐帧为零；本体深度范围为 **14.604461640—14.604521245 mm**，变化约0.000060 mm，可以说稳定保持，不能说数值逐帧完全不变。每一帧均有原金属止挡正接触，且同一套128个簧套同时承载。全回合最大轴向后备限位坐标0.515938 mm，未达到原0.5999 mm检查值；最终没有人工扇区接触。

[查看最后旋拧和松手的9.6秒实际视频](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/postrun_evidence/final_turn_and_release_actual_clip.mp4) · [查看完整4分51秒视频](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/video/assembly_four_view.mp4) · [独立逐帧松手复核](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/full14_terminal_release_review_CN.md)

桌面抓取也已按原始指甲表面检查：保持2秒期间，本体相对初始高度至少抬升 **56.05 mm**，无桌面承载，三只原指甲持续接触本体。入槽后，本体与手的真实分离也通过了整回合原始接触核验。抓螺母与旋拧期间，两只原指甲加一处原指腹的接触区域检查通过；最终卸载和张手阶段另经独立补核通过。

| 抓取旋拧次数 | 手腕已发指令 | 螺母相对本体实际转角 | 本体深度变化 |
|---|---:|---:|---|
| 第1次 | 90.00° | 89.20° | 7.917 → 9.326 mm |
| 第2次 | 90.00° | 90.80° | 9.057 → 11.075 mm |
| 第3次 | 66.86° | 67.86° | 10.937 → 12.445 mm |
| 第4次 | 29.68° | 28.73° | 12.443 → 12.983 mm |
| 第5次 | 78.75° | 76.98° | 12.984 → 14.604 mm |

各次之间实际执行了卸力、张手、回腕和重新抓取，所以表中上一段结束与下一段开始的深度会有差别。例如第一次换抓发生约0.269 mm卸载回退；未坐底阶段不能说本体绝对不动。最后到位后的3秒保持已单独逐帧核对。

此次已经证实能起作用的控制修正，是每次换抓依据当前螺母图像重新对准，并让所有旋拧段一致使用正常卸力规则；0.5 N·m手指传动余量不再要求先旋转60°。第3、4、5段分别在第二指、第一指、第一指余量达到约0.5 N·m时正常结束，顺利接上释放或下一抓。第4段只转了29.68°也能正常卸力。没有为本次成功提高4 N·m手指传动、8.8 N·m输入、110 N腕合力等既定硬上限。

这与上次末段只计划转19.25°、却要求先转满60°才能启用余量保护的错误不同；那个条件使正常停止根本无法触发。旧约14.605 mm记录虽然使用当前修复连接器，但手仍是旧1:1联动；本次使用的是当前有限四杆、自锁三指手。

**保留的未通过项是第三个键的几何数值残差。** 第二次旋拧的raw228525，原键顶点对原槽侧壁平面的最小间隙为−2.147659 μm，超出原2 μm比较带0.147659 μm。当时本体深度10.145021 mm，下一控制决定的腕合力48.6205 N。在该最差点前后共961帧的局部窗口中，4帧超带，最长连续2帧，约2.08毫秒；这不是全回合所有超带帧的总数统计。

源码说明2 μm来自历史已接受旋合仿真约1.706 μm残差向外取整的后评比较带，不是厂家机械安全上限，也没有改变真实几何间隙。负残差是实测的同源几何投影结果，不能直接称为浮点噪声，其具体成因仍未分清。原`source_key_containment_review.json`保持`accepted=false`，整体`whole_assembly_review.json`保持`REVIEW_REQUIRED`及`complete_visual_assembly_verified=false`。**机械到位与真实松手已经发生，全部原数值检查通过则没有发生。**

[原五键检查](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/source_key_containment_review.json) · [最差点窗口](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/key_borderline_window_review.json) · [独立核对比较带的来源与意义](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/key_band_meaning_review_CN.md) · [整体原始核验](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/whole_assembly_review.json)

第4段约25°以后还记录到手侧跟踪瞬态：按机器人编码器计算的旋拧中心偏差增至约0.608 mm，横向纠偏请求受2 mm/s上限截断；此时螺母实际横向位置仍在插座中心附近约0.030 mm，本体倾角约0.035°，手与螺母相对转角变化约0.82°。该段机械臂没有驱动饱和，螺母仍实转28.73°并推进0.540 mm。现有数据支持受力和跟踪需要协调，不能据此断言提高速度上限就能解决全部问题，也不能把手与螺母的相对角变化全部当作接触滑动。

耗时仍是明显不足。本次电脑从启动到收尾约 **257.0分钟（4小时17分钟）**；记录的物理动作约 **291.34秒**，其中五段旋拧合计约 **12.88秒**。接触计算、读取与保存记录占用大量电脑时间，搬运和探测流程本身也较保守。此前向量读取的小优化不能称作整体效率问题已经解决。

包装器退出码为2，通用`evaluation.json`仍是旧指腹／顺序抓取用途的初始拾取评估，覆盖前35,684帧，并未代表整条装配验收；其中指腹身份缺失等判定不能替代本回合原指甲接触的直接核验。这里保留原文件与退出码，以逐阶段实际证据说明结果，不修改旧报告来制造通过。所有结论限于本次仿真，不是硬件验证。

[逐次实际转角与深度](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/turn_progress_physical_review.json) · [原指甲抓取与离桌保持](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/source_nail_body_review.json) · [旋拧期间原指甲／指腹](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/source_nut_pad_review.json) · [第4段机器人与物体对齐记录](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/fourth_tracking_transient_physical_review.json) · [耗时记录](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14/run_timing.json)

此前分析保存在[历史说明](/home/noob/WorkPlace/kcgtest1/docs/history/GRASP_PHASE_ROOT_CAUSE_20260915_CN_before_full14_mechanical_result.md)，不再用旧回合角度和深度代替本次结果。
