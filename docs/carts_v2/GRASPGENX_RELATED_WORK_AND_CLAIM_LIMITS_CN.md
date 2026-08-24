# GraspGenX-CARTS 相关工作与结论边界

## 官方方法与本项目的边界

GraspGen-X 是跨机械手六维抓取生成方法：用手的 swept-volume（开合扫掠体）表示来条件化一个跨形态模型，并同时提供 diffusion 与 OBB 提案的 GraspMoE 路线。官方论文和仓库报告它在程序化手型与大规模模拟抓取数据上训练，并面向未见手型做零样本泛化。

本项目只调用官方预训练推理，不训练、不微调、不复制核心实现。GraspGenX 输出的是“值得验证的六维手掌姿态”，不是 KCG 三指手的指腹接触、承重、机械臂可达性或 Isaac 动态证明。

## 可写入论文的方法部分

- 从 KCG 手真实 URDF、mimic 关系和注册 PAD 几何自动建立固定多预构型描述器库。
- 把跨机械手六维提案映射回真实 `handbase_link`，并绑定完整坐标链与哈希。
- 用真实三指顺序闭合、完整控制步手—桌检查、非 PAD 碰撞和任务载荷重排淘汰生成器假阳性。
- 分开“12 N 名义任务合格”“预登记误差鲁棒合格”“机械臂路径可执行”和“Isaac 动态成功”。
- 在两种连接器上冻结同一主要参数并保存全部失败。

## 本轮不能声称

- 不能声称本项目发明了 GraspGenX 或跨机械手六维生成。
- 不能声称 GraspGenX 对当前 KCG 长指手零样本泛化成功。
- 不能声称比旧轴对称生成器显著更好：目前只证明姿态覆盖更广，没有路径安全或动态成功优势。
- 不能声称任务鲁棒、跨对象成功或论文假设 P1–P4 已验证。
- 不能把官方 banana 正对照、GPU 推理、退出码 0、覆盖图或离线测试当成抓取成功。
- 不能提出真实硬件结论；`hardware_authorized=false`。

## 当前可证伪结论

如果同一预训练模型、描述器库、预算和物理门在对象 A/B 都产生路径安全且任务合格候选，并在冻结参数的 Isaac 运行中完成三指允许接触、离桌、50 mm、2 s，则支持本路线。当前结果与该假设不一致：A 的三指接触候选扫桌，B 没有三指闭合候选。

当前数据支持的限制解释之一是预训练描述器域外推，而不是“三指手物理上不能抓”。该解释仍需未来用训练域匹配的描述器/模型或预先登记的替代全局提案器进行独立验证，不能在本轮通过继续调箱体、偏移或面标签来证实。

## 官方来源

- <https://github.com/NVlabs/GraspGenX>
- <https://arxiv.org/abs/2606.00998>
- <https://huggingface.co/adithyamurali/GraspGenXModel>
- <https://huggingface.co/datasets/adithyamurali/gripper_descriptions>

官方仓库代码许可为 Apache-2.0；checkpoint 使用 NVIDIA Open Model License；本轮实际 revision 与文件哈希记录在 `INTEGRATION_MANIFEST.json`。
