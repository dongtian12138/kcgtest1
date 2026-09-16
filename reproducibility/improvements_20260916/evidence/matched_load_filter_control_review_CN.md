matched50ms／matched10ms的独立只读控制复核。

核对时点：2026-09-16 20:22:55 UTC。结论：**在所审源码、开跑配置、实际selected settings、实际coaxial_recipe及已写机器人控制记录范围内，A/B的有效控制差别只有接触负载前馈的滤波由原50 ms变为10 ms。原50 ms导纳／停止信号和原硬界保留，没有再次误选前段1／1.5 mm/s配方。** 未读取10 ms回合的物体或接触truth，未启动物理、改生产源码或配置。

新增字段的路由明确：[te_local_interface_following.py:121](/home/noob/WorkPlace/kcgtest1-improvements-20260916/src/kcg_connector/isaac/te_local_interface_following.py:121)始终按原`dt/(0.05+dt)`更新`self.filtered`；只有显式配置新字段时，才另更新`filtered_contact_load`。后者在[358行的原接触负载补偿支路](/home/noob/WorkPlace/kcgtest1-improvements-20260916/src/kcg_connector/isaac/te_local_interface_following.py:358)进入力／矩换点及`-Jᵀwrench`前馈，另有日志输出。导纳及后续观察继续从`wrench=self.filtered.copy()`取得原信号；正常90 N／1 Nm余量门读取的`interface_wrench`也来自该原信号。110 N／1.2 Nm／4.6 Nm硬门及2 Nm坐底候选仍用它，相关stop／advance顺序没有改动。

“只改前馈”应理解为新增信号的路由范围：补偿支路原有的加载参考迁移仍会使用补偿向量，不是保证所有情况下名义arm参考完全不受前馈影响。本匹配入口通过`adopt_existing_grip`建立原当前受载编码器参考，设置`load_compensation_active=True`与已有reference-transfer记录；没有新增独立位置反馈或把10 ms信号送入原导纳／保护。更改前馈会改变后续真实运动和测量，因而不承诺两回合保护读数或停止时刻相同。

默认None保持原算式：提取新纯滤波方法，在2048个六维有限输入、dt=1/960 s上核对，None分支的原50 ms状态逐位等于旧表达式；显式50 ms的附加分支亦逐位一致；启用10 ms时，原50 ms状态仍逐位不变，新分支逐位等于独立10 ms递推。这是纯数值方法核对，没有创建机器人或仿真。默认新增了两个日志字段，不能把“原算式相同”说成整个旧日志文件字节相同。

| 实际有效项 | matched50ms | matched10ms |
|---|---:|---:|
| 新前馈滤波字段 | 未配置／日志null，沿用原50 ms | 0.01 s |
| 原导纳／停止滤波 | 0.05 s | 0.05 s |
| 横向／轴向速度上限 | 0.002／0.003 m/s | 0.002／0.003 m/s |
| 腕合力／弯矩／扭矩硬门 | 110 N／1.2 Nm／4.6 Nm | 相同 |
| 正常合力／弯矩／弹性余量 | 90 N／1 Nm(min0)／0.5 Nm(min0) | 相同 |
| 原手主动上限 | 掌1、三指各4 Nm | 相同 |
| 原输入折算边界 | 4 Nm输出参考对应8.8 Nm | 相同 |
| CPU／物理频率／求解迭代 | CPU／960 Hz／64、4 | 相同 |

两份实际`rotation/coaxial_recipe.json`完整递归比较，仅有`contact_load_compensation_filter_time_constant_s`缺省与0.01的区别。两份task YAML仅改变`nut_rotation_after_index.coaxial_interface_recipe`指向；probe仅改变assembly配置文件名，四电机输入角／速度、原source_step258721、warmup0.5 s、90°请求、累计起点246.8592448503°、单次诊断与7000拍预算均相同。命令仅有输出目录、probe路径和task路径三处命名差异。

两份实际controller result的settings均与执行入口分支重建值完全一致。相对prelaunch的`local_selected_settings`，共同的`single_attempt_diagnostic`会把recovery改为`{'enabled':False}`，所以去掉了原先同样禁用状态下的`maximum_regrasps=1`；这是两个回合同有的诊断整理，不是A/B额外控制差异。实际配方的坐底minimum_command共同由350变为103.1407551497°，等于350−246.8592448503，是原累计角到局部角的换算，未降低累计350°要求或2 Nm门。

读取到的50 ms控制记录为1343行、step481—1823，全部新tau列为null，负载前馈信号前三个力分量逐行等于原50 ms控制力；10 ms为3240行、step481—3720，tau全部0.01，独立前馈力信号已实际生效。只比较前三个力分量是因为两份日志的力矩所在原点不同：负载信号在固定滤波原点，`interface_wrench`已移到当前枢轴，不能把这种原点差误判成滤波差。

A/B开跑快照中的`te_local_interface_following.py`、`te_source_stage_probe.py`及`carts_v2/evaluate_run.py`三份代码逐字相同；各列入manifest的文件都匹配其摘要。本次读取时亦与当时工作文件相同。**这项一致性限于A/B执行快照和上述审查时点**：主代理已说明B结束后再整合接触向量tuple及测试，不应把随后磁盘上的evaluate_run.py变化误判为A/B执行时改源。

唯一快照覆盖小缺口：50 ms的prelaunch_sources没有独立复制其所引用的原wrist recipe文件；10 ms复制了新recipe。但50 ms的prelaunch_config_check完整保存了selected_recipe，与该回合实际coaxial_recipe交叉核对，只有上述原有累计角换算差异，因此这不是实际错选配方或对照无效的证据，也无需据此修改原运行。

完整证据：[matched_load_filter_control_review.json](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/matched_load_filter_control_review.json)。只读核对脚本：[audit_matched_load_filter.py](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/audit_matched_load_filter.py)。本审查没有据控制日志宣布机械效果，封存后的实际运动与接触效果由主代理的本回合后评承担。
