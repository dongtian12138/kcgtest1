# 当前：新回转策略的完整四相机装配已验收，正在发布

核验时间：2026-09-29T14:52:34.127330+00:00。用户本轮明确授权完整仿真、新基线、GitHub发布和录像保存。全部物理动作及后评已完成，没有运行中的物理实验；hardware_authorized=false。

## 当前有效完整基线

工作目录 `/home/noob/WorkPlace/kcgtest1`，分支 `codex/connector-adaptive-return-baseline-20260929`。实际完整执行提交 `6df30f18308bcad767bc0b5f46ef80089f7a6bf8`。全新桌面场景连续完成抓取、两次观键、插入、六次螺母抓握旋拧和最终3秒完全松手。全程292567步连续；整体VERIFIED、严格3秒释放accepted=true、回转专项accepted=true。在线到位判定也在本轮实际执行中确认。

实际空手回转为90/90/0/90/0度，两个0度实测也为0，后续重抓保留所选朝向。回转期间最小臂关节余量13.17921度。最终2880步手冲量逐帧零、源止挡和同128夹片逐帧承载，深度14.604610652mm。最不利键侧壁数值残差1.975941µm仍在原2µm带内；原几何/质量/材料/驱动和全部原边界不变。仅此名义仿真回合，不宣称极端工况统计或硬件保证。

本轮底层退出码2来自继承的通用CARTS nominal_research_dynamic_pass旧分类路径，与旧完整基线一样；原evaluation.json不改写。独立源指甲/指腹、键槽、四相机、松手与全程实际物理验收均通过，不能把旧分类或退出码单独当作装配结论。

## 必须保留的资产

- 新完整原始数据与原视频：`artifacts/adaptive_return_20260929/full01/run`。
- 本机独立原视频副本：`/home/noob/Videos/kcgtest1/20260929_adaptive_return/assembly_five_view_full01.mp4`，304.8秒、1524帧、1920×1080、5fps，复制后字节摘要一致，无剪辑拼接。
- 源码、配置、执行绑定与小型证据：`reproducibility/adaptive_return_baseline_20260929`。结果见 `result_CN.md`、`acceptance_summary.json`；复现见 `docs/REPRODUCE_ADAPTIVE_RETURN_BASELINE_20260929_CN.md`。
- 原2026-09-20基线独立工作树、其完整成功原始数据和2026-09-17两次重复通过原始数据继续保留。

本次新成功原始数据和独立视频副本不属于缓存或失败实验，不因artifacts被Git忽略而删除。上次删除研究局部成功录像的更正见 `docs/DISK_CLEANUP_20260929_CN.md`；旧活动入口见 `docs/history/CURRENT_CONTEXT_CN_during_adaptive_return_full01_20260929.md`。

## 剩余工作

尚需提交并推送本次新分支、固定发布 `assembly-four-camera-adaptive-return-20260929`，上传完整原录像及证据包，验证远端代码与附件可取用。发布已获用户授权，不再重复确认。GitHub现有账号写入权限已核对。所有运行控制源码仍是已实际执行的6df30f1，之后只整理后评与发布信息。
