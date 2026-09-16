三份接触向量的tuple存储对照，2026-09-16。

结论：**候选能在保留全部归档数值、字段顺序和MessagePack字节的同时，使三份向量及其点dict在GC后解除跟踪；120帧、64／1存储、默认GC的六个独立进程中，总墙钟中位数降低约14%，GC时间降低约34%。** 这是有依据的候选，不是完整物理运行提速结论。内存中向量类型从list变为tuple，确实影响一个现有类型敏感测试，不能声称所有Python对象行为完全不变。

只把原`_decode_full_report`中`position[:3]`、`normal[:3]`、`impulse[:3]`的结果分别包装为tuple；所有源字段读取、范围检查、点数、offset、顺序、separation和后续统计代码保持。候选只动态装入离线子进程的类，生产文件未修改。原GC始终开启，阈值不变为`(700,10,10)`；记录仍为64帧块、1个已封块缓存，每个独立进程120帧。

使用完整14封存raw279680、279681，分别为17384、17380接触点。两帧原始完整poll_headers、经原contact_counts得到的完整contacts，以及JSON编码均与tuple候选相同；MessagePack按array编码list和tuple，字节完全一致。改写并清空重建的原生Float3缓冲后，两种输出均保持原字节，确认tuple仍持有独立的Python数值。六个120帧回放的完整解压MessagePack流均为411,513,840字节，摘要相同，压缩档大小均为57,495,982字节；各块索引完整。没有删点、精度压缩或丢弃零冲量／负separation。

单独的、计时外的跟踪状态探针明确显示：在一次正常`gc.collect()`后，原17380／17384个点dict以及各自三份list仍全部受跟踪；tuple候选中的这些点dict和三个向量tuple的受跟踪数量全部为0。刚解码完、尚未显式collect时，有些新tuple和点dict仍被跟踪，不能把探针结果误写成“所有时刻都立即untracked”。实际计时回放中没有手动collect；GC时间也并未归零。

先完成两轮正／逆顺序配对，再补一轮独立进程记录CPU。下表按每类三个进程取中位数；GC时间已包含在各阶段和总墙钟中：

| 指标 | 原list | tuple候选 |
|---|---:|---:|
| 总墙钟 | 10.988 s | 9.450 s，下降13.99% |
| 其中循环GC | 4.895 s | 3.210 s，下降34.43% |
| 创建／解码阶段 | 6.034 s | 3.709 s |
| 原contact_counts统计 | 2.807 s | 3.581 s |
| append | 1.042 s | 1.052 s |
| 最终close | 1.090 s | 1.080 s |
| 峰值RSS | 1837.5 MiB | 1645.1 MiB |
| 第2代GC总时间 | 4.412 s | 2.810 s |
| 第0／1／2代次数，每次重复相同 | 15530／1411／18 | 15513／1410／30 |
| 第2代最长暂停，各进程范围 | 0.811—0.816 s | 0.203—0.233 s |

六个进程各代收集的循环垃圾对象和不可回收对象均为0。候选减少了扫描开销，也改变GC触发阶段：创建阶段的GC中位数从3.450降到1.020 s，统计阶段反从1.445增到2.190 s。应看整个记录管线的净收益，不能只报告创建阶段下降而忽略统计阶段增加。

追加第三轮的**进程CPU时间**为list10.9359 s、tuple9.5379 s，下降12.78%，与该轮墙钟接近；此前两轮只计墙钟，未把它们冒充CPU测量。最早两轮的总墙钟中位数为11.296→9.343 s（17.28%），追加CPU轮后本报告使用所有三个进程的中位数13.99%，完整原始结果均保留。

tuple转换并非免费。单独保持GC开启，对真实Float3执行“三份切片”与“三份切片再tuple”各100000次，交替8次：中位0.076912→0.080800 s，三份转换合计约增加0.0389 μs／点，按17380点粗算约0.676 ms／帧。这个短向量循环只衡量额外转换成本，不含整个decoder或大堆GC；实际全回放已包含转换，并仍有净收益。

**消费者兼容性。** 检查了`src/kcg_connector/{isaac,kcg_connector,test}`中的直接字段访问、类型检查、列表比较及向量原地写，并阅读了原统计和主要后评路径。生产消费者主要通过索引／解包、`math.hypot`或`np.asarray`读取；`_finite_vector`也是NumPy转换后检查形状和有限值，接受tuple。msgpack／JSON重新读取仍得到list，已有磁盘报告格式不变。没有发现所审生产路径要求这三个内存向量为list或向其元素原地写入，但本静态检查不证明任意外部脚本或任意别名修改兼容。

明确的两项测试依赖不能隐藏：

- [test_nut_runtime_efficiency.py:32](/home/noob/WorkPlace/kcgtest1/src/kcg_connector/test/test_nut_runtime_efficiency.py:32)直接断言`position_m == [1.,2.,3.]`。在子进程装入tuple候选后，该现有测试确实失败；数值及编码检查通过。若采用新内存表示，应使这个断言检查数值／序列或归档字节，而不是继续假称原list类型保留。本次未改测试。
- [test_native_contact_audit.py:35](/home/noob/WorkPlace/kcgtest1/src/kcg_connector/test/test_native_contact_audit.py:35)通过`impulse_n_s[0]=NaN`注入异常。该fixture自行构造list、未经过decoder，因此本候选不会改变它；对真实tuple输出做类似故障注入时应替换完整向量字段，不能原地改tuple元素。独立有限值拒绝检查没有放宽。

候选源文件：[tuple_contact_decode_candidate.py](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/tuple_contact_decode_candidate.py)。[六进程汇总](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/tuple_storage_gc_summary.json) · [完整结果与跟踪探针](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/tuple_storage_gc_replay_results.json) · [转换微基准](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/tuple_conversion_microbenchmark.json) · [静态消费者审查](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/tuple_vector_consumer_static_audit.json)。

复现脚本为[benchmark_tuple_contact_storage.py](/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/independent_final_guard_review_20260916/benchmark_tuple_contact_storage.py)，共用此前有界回放脚本。默认入口执行前两轮，`--worker --mode list/tuple --repeat 2 --frames 120`分别执行追加CPU轮，`--conversion-only`执行短向量微基准；已有同名进程结果目录拒绝覆盖，可使用新的repeat编号。

边界：回放仍只有两个密集末帧的120份新容器，不包含SimulationApp本身、C++报告取得／字段绑定成本、完整运动接触分布或279682条不断增长的机器人／六维力历史。因此它验证了“不可变数值容器可减少无循环数据的GC扫描”以及本离线管线的净收益，不能直接预测完整14提速14%。此次不与路径分类缓存或更小归档块混合，以保留归因；生产采用仍应在既有受控小窗口检查真实运行时间和相同记录语义，不需要也不建议关闭全局GC。
