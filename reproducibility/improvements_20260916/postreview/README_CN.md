# 结束后的复核辅助脚本

这些脚本复用原完整14和独立最终释放复核的算法。仅在新回合物理结束、原始归档索引封存后使用；不进入在线控制。

- review_episode_turn_progress：本回合各段实际Nut角、Body深度。
- review_source_band_occlusion：原红色标记的末态径向遮挡；仍须查看本回合真实视频，不能凭射线布尔自动接受可见性。
- supplement_terminal_release及verify_sealed_final_evidence：本回合完整最终卸载/张手源面、实际保持时长、逐帧零手冲量/止挡/128簧套、记录覆盖和休眠。后者只保留共享函数，未带入旧13/27专属main。
- selective_postrun_reader：原已验证的手接触选择性读取器，额外保留Body复核所需原时间与桌面冲量字段。它会跳过内部Clip/止挡点，不能直接用于整体机械复核。
- review_ending_mating：旧2秒窗口、止挡any/128max规则；不是完整3秒逐帧同时承载的替代证据。

原模型、原接触点和原始数据不改。所有路径基于当前仓库；不得沿用旧回合帧号、步号或accepted结果。
