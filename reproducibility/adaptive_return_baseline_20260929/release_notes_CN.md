# 四相机完整装配：按关节余量选择空手回转

本次已在同一个连续仿真回合中实际完成桌面抓取、两次观键、搬运、插入、换抓旋拧和最终3秒完全松手，独立全程验收为VERIFIED。

回转选择为 **90°、90°、0°、90°、0°**。两次省去回转均有实际运动记录，后续装配完成；需要回转时检查原关节边界和下一段旋拧可达性。原模型、原控制限制和原物理验收标准保持。

最终2880个释放保持步手部冲量严格为零；源金属止挡和同组128夹片逐帧受载；深度约14.604611mm。完整原五视角录像 **5分4.8秒**，未剪辑或拼接。实际执行提交为 `6df30f18308bcad767bc0b5f46ef80089f7a6bf8`；后续发布提交整理证据与复现入口。

- [源代码分支](https://github.com/dongtian12138/kcgtest1/tree/codex/connector-adaptive-return-baseline-20260929)
- [复现与验收说明](https://github.com/dongtian12138/kcgtest1/blob/codex/connector-adaptive-return-baseline-20260929/docs/REPRODUCE_ADAPTIVE_RETURN_BASELINE_20260929_CN.md)
- [详细结果](https://github.com/dongtian12138/kcgtest1/blob/codex/connector-adaptive-return-baseline-20260929/reproducibility/adaptive_return_baseline_20260929/result_CN.md)

附件包含原完整录像、验收证据包和SHA256校验表。运行资产复用已有版本化发布包，准备脚本校验297个资产及2个Git相机输入。完整原始逐帧档案保留本机，不包含在小体积附件中。

原底层通用CARTS进程返回码2及旧分类字段均保留，原因和正确的独立验收入口见说明。新回合必须重新验收。仅仿真，未执行硬件试验，也不宣称任意极端短旋拧条件下的成功保证。
