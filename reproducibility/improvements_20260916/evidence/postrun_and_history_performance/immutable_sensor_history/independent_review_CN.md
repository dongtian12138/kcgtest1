独立只读复核（2026-09-17 UTC）：**当前已检查的腕部传感历史消费者，没有发现必须保留嵌套数值 list 可变性的实际用法；把已构建记录内的数组转为拥有副本的 tuple，具有静态兼容依据。离线数据只证明固定历史堆的 gen2 扫描显著变快，还不能证明完整记录或整机净提速。** 生产 baf6d4f 保持不变，没有读取 repeat02、运行物理或更改原始记录。

审查范围为 `_HighObservationWristFtAuditor` 生产者、当前 Body/键进入/Nut/换抓与释放路径、必要的同类 FT 历史入口和现有序列化工具。这里只改变“每行记录里面的数组”这一候选含义；外层 samples 仍须支持 append 的 list，每行仍为 dict，不能将整个 auditor 或所有在线可变状态冻结。

| 实际消费者 | 在线用途及 tuple 兼容性 |
|---|---|
| run_grasp_lift.py:5348 的 capture 末尾 | 先完成当前 FT 去皮/过滤、峰值、first_ft_stop 与 stepper.abort_reason 更新，再追加记录；这些在线计算状态独立于保存数组。不能把转换移到原停止计算之前，或顺带冻结 `_tare_sensor_samples`、动态惯量预测器、连续过滤状态。|
| te_coaxial_nut_interval.py:152 → te_local_interface_following.py:187 | 每拍读取最新 hand2arm_raw_wrench，用切片和 NumPy 矩阵运算；tuple 可供相同运算，不依赖 list 方法或原地写入原记录。|
| te_body_nut_reindex.py:218/339、te_body_nut_regrasp.py:399、te_body_support_release.py:61 | 从最新 active_targets_rad 取最后真实已发目标，转 np.asarray 后复制/生成可变工作数组，用于卸载、保持和重新抓握。没有对历史数组本身赋值；这一路不能误改为未发的候选目标。|
| te_foundationpose_handoff_runtime.py:290、te_body_support_release.py:72 | 在线遍历早先 phase=tare 的 active_efforts_nm 求均值；旧记录仍会被读，不能仅保留最新一行。切片及 np.mean 接受 tuple。|
| te_body_nut_rotation.py:613–647 | 旧控制分支读取最近一段夹持历史，构建手位姿、初始化因果过滤器；赋值目标是新建 NumPy 矩阵，并未改旧记录。|
| te_coaxial_nut_interval.py:49–56、te_local_interface_following.py:133 | 当前同轴主路径从本回合 regrasp 导出的 JSON 重新读取开手去皮与最后约 0.5 秒已夹持历史，用于标定/载荷过滤。导出后 JSON 数组会重新解码为 list；记录含义与样本范围仍须保持。|
| te_body_nut_continuation.py:59/189 | 最新 phase/step 用于正常换抓/原停止后的释放资格；数组类型无关。该文件 `sample.update(...)` 修改的是从控制 JSONL 新读出的控制行，不是 ft.samples 的传感行。|
| run_grasp_lift.py:7241 的预接触恢复分支 | 可遍历全部已发送目标历史再倒序运行；先转 NumPy，tuple 数值数组兼容。这不是当前完整主线每拍都做的工作，但也说明历史不能以“只是日志”为由删除。|
| run_grasp_lift.py:10862 与各阶段 joint_ft 导出 | samples 的长度、负索引、外层切片与逐行迭代都保留。当前 run_grasp_lift._json_ready 和 te_foundationpose_handoff_runtime._json_ready 均明确处理 list/tuple；fast_json.dump_array 使用的 ujson 也在本次小样本验证中输出相同数组。此前 GzipStore/自定义切片的兼容问题不适用于这里保持外层 list 的候选。|

生产记录的向量/矩阵大多已经通过 `.tolist()` 获得 Python 数值副本。仅 `arm_control.drive_target_rad` 和 `gravity_compensation_nm` 保存了上游列表引用；controller.py:257/270 每拍通过 `.tolist()` 新建这两个列表，在本次必要消费者中未发现后续原地修改它们的用途。候选递归转换会再拥有一份 tuple 容器，切断这些列表别名。若以后为了省转换而直接改成 `tuple(numpy_array)`，需另确认标量类型和导出行为；它不自动等价于当前“对已 .tolist() 的记录递归转换”的试验。

独立小型兼容探针使用同一已封存回合的 4 条记录：step0 去皮、32767 抬升、290424 晚段 Nut 旋拧、290720 卸载。MessagePack、现有 _json_ready+标准 JSON、fast_json/dump_array 的结果均相同；NumPy 形状/类型/值、矩阵赋值和直接 tuple 切片矩阵乘法一致。修改解析出的源 list 或新建 NumPy 工作数组，不影响转换后记录。另用小型合成值检查 None/bool/−0.0/±Inf/NaN 的现有编码行为保持。没有执行控制器；这不能替代真实 capture 位置和完整控制调用链的等价验证。

离线试验的精确范围：

| 指标 | 原 list 数组 | tuple 数组 |
|---|---:|---:|
| 固定历史条数 | 32768 | 32768 |
| gen2 稳态中位数 | 0.1127595895 s | 0.0263962605 s |
| 摄入、解析、双编码和检查总时间 | 1.425435904 s | 1.845572113 s |
| 转换循环计时 | 0.001708116 s | 0.423717263 s |
| 记录峰值 RSS | 394.625 MiB | 371.4375 MiB |
| 6 次回收后的受追踪外层行 dict | 32768 | 32768 |

扫描中位数下降约 **76.59%**，但六次回收是每模式同一个固定堆上的连续操作，稳态中位数实际取后 **4** 次；每模式只有一个新进程，原 list 后 tuple 的固定顺序，不是六次独立重复。两路回收对象数均为 0，说明这里主要测量活对象扫描成本，没有验证循环垃圾释放。外层行 dict 仍全部受追踪，不能说“所有传感历史都已被 GC 取消追踪”。历史条数也仍持续增长，tuple 没有使其变成有界存储。

递归转换额外约 0.422009 s，即 **12.879 µs/行**；摄入区间增加约 29.47%。峰值 RSS 减少约 23.19 MiB（5.88%），这个数含本进程解析/检查等开销，不是纯历史对象的精确内存占用。完整逻辑 MessagePack 流 SHA 一致，且每行直接比较相等；但生产主输出是 JSON gzip，原试验没有测量实际 capture 每帧转换成本、生产归档写入、混合动态堆或整机总时间。

本次额外按流读取确认：原性能样本是 **step0..32767**，覆盖去皮、接近、抓取及抬升，尚未包含 Nut 段。它不是从 29 万帧中均匀抽样，也没有测量实际每 128 步新增记录、原始接触缓存、控制/渲染对象并存时的混合堆。晚段数据的小型类型/导出检查已经补做，但没有把它冒充晚段性能测试。

结论可表述为“现有消费者未见嵌套数组可变性/list 类型阻塞；固定 32768 行机器人历史的稳态老代扫描成本下降，数据与小样本现有导出形式一致”。不应表述为“FT 只是离线日志”“整机可提速 76.6%”或“可以在冻结回合中直接采用”。若后续采用，最小变动应仅发生在保存已完成记录的位置，保留所有字段、原在线状态和停止顺序，再把转换、回收、导出一起计入局部净成本；本次未实施。

证据：[独立兼容与阶段覆盖 JSON](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/performance_review/immutable_sensor_history/independent_compatibility_probe.json)；[小型探针脚本](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/performance_review/immutable_sensor_history/independent_compatibility_probe.py)。原 review.py、comparison.json 与生产文件均保持不变。
