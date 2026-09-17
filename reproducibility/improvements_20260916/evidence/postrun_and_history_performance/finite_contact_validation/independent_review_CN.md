独立只读复核（2026-09-17 UTC）：**在当前已安装 Native ContactData SDK 和未变 `_decode_full_report` 的输入范围内，流式 finite 检查有充分的形状/类型依据，没有发现它会漏过当前原生路径可能提供的非有限浮点值；值得保留为一个小型候选。它不是任意 Python 容器上的完全等价替换。** 生产仍保持 baf6d4f，本次未启动 SimulationApp/物理、未读 repeat02 或其他原始运行档，也未修改原候选、报告或生产源码。

依据来自本机实际 SDK，而不只来自健康帧：

- 已安装 omni.physx **110.1.13** 的 [ContactData 定义](/home/noob/WorkPlace/isaacsim/.conda-env/lib/python3.12/site-packages/isaacsim/extscache/omni.physx-110.1.13+110.1.2.lx64.r.cp312.u7f4/omni/physx/bindings/_physx.pyi:442) 将 position、normal、impulse 声明为 carb.Float3，separation 为 float；ContactDataVector 的整数索引返回 ContactData。
- [Carbonite C++ Float3 存储定义](/home/noob/WorkPlace/isaacsim/.conda-env/lib/python3.12/site-packages/isaacsim/kit/dev/include/carb/Types.h:369) 是 x/y/z 三个 C float；[Python 绑定说明](/home/noob/WorkPlace/isaacsim/.conda-env/lib/python3.12/site-packages/isaacsim/kit/kernel/py/carb/_carb.pyi:443) 明确为三分量、整数索引返回 float、切片返回 list。本次仅导入轻量 carb 类型，确认真实 Float3 的 len 和 [:3] 长度始终为 3，元素都是 Python float；两元素构造抛 IndexError，四元素构造仍只形成三分量，不能形成不齐长的原生向量。
- 当前 [解码器](/home/noob/WorkPlace/kcgtest1-improvements-20260916/src/kcg_connector/isaac/carts_v2/evaluate_run.py:426) 先验证 header 数据范围，再把三个 [:3] 复制成自有 tuple，separation 显式转 float。生产 `_physics_step_reports` 的填充点是 [原生回调](/home/noob/WorkPlace/kcgtest1-improvements-20260916/src/kcg_connector/isaac/carts_v2/evaluate_run.py:471)，随后被 `_contact_counts` 消费，没有发现中间将其替换为任意形状/类型的生产写入。单测的 SimpleNamespace/手工报告注入是另一个输入范围。

因此当前 native-report 分支的每点由 **3+3+3+1 个 Python float** 组成。旧 `np.asarray(...,dtype=float)` 的矩形性和数值类型在此已经由上游契约保证；任一分量为 NaN/±Inf，两种有限检查均拒绝，有限值/±0 均接受。解码在检查前已经复制全部原生点，流式迭代只遍历 Python 自有值，不会把原生缓冲读取推迟到下一物理步。原 missing-callback 检查、其他 header/通道/归类逻辑及非 native-report 的 tensor 分支保持原样。

确实存在的非通用差异已独立复现：

| 人工 Python 输入 | 原数组检查 | 流式检查 |
|---|---|---|
| 一行正常 10 元素，另一行少一个位置分量 | ValueError，矩形构造失败 | 所有现有值有限时返回 true |
| 唯一一行位置只有 2 分量 | 返回 true | 返回 true |
| 分量为数值字符串 "1.0" | 转 float 后接受 | TypeError |
| 分量为 None | 转 NaN 后 finite=false | TypeError |
| 第一行 NaN、后面另有不齐长行 | 先因数组形状失败 | 短路返回 finite=false |

这也说明旧 NumPy 检查并没有严格验证“每向量必须为 3、每点必须为 10”；它只是附带拒绝某些混合长度或类型。上述畸形容器不能由当前 SDK 的三个 Float3 和当前复制规则产生。不能据此声称流式检查保留了任意损坏 Python 对象的所有异常类型/先后顺序。若以后加入任意字典/外部文件作为生产输入，或更换不再保证 Float3 的后端，应在那个输入边界保留形状/类型验证；没有必要为本次已被 SDK 保证的每个正常分量再加一套泛化热路径框架。

性能与现有证据的范围：

- 当前生产源哈希与 prepared_against_sha256 完全一致，准备补丁只替换原有限值检查块。
- 17901 点的同一密集帧，检查函数中位数 4.9388225→3.5243090 ms，节省 **1.4145135 ms/该类帧（28.64%）**。这是 finite 检查局部耗时，不是完整 `_contact_counts`、接触回调或整机提速。
- 30 次测量在同一进程、同一已解码帧、自动 GC 关闭条件下进行。代码交替正反三种方法顺序，streamed 始终位于中间；不是 30 个独立原生/物理样本，也不是所有位置完全随机平衡的重复。
- 现有四帧 prepared_patch_probe 检查完整 `_contact_counts` 输出及其 MessagePack 与原保存接触记录一致，覆盖接近、保持、旋拧和密集自由保持；本次审阅其代码/结果，没有重读原档重做该回放。NaN/±Inf/±1e308 的十分量测试提供布尔值证据，其中 ±1e308 是普通 Python double 的额外边界，不是 Float3 可保存的有限幅值保证。
- 新增小探针还验证了实际 carb.Float3 的 NaN/±Inf、±0 和 3e38，两个 guard 的结果一致，并确认改写源 Float3 不改变解码式 tuple 副本。无物理对象或接触接口被创建/查询。

建议按“**当前安装 SDK + 当前解码器输入范围内的有限值检查等价**”描述候选，不把 prepared JSON 中的 invalid-value 保留标志解释成任意畸形 Python 输入上的全面等价。现有证据支持以后在冻结回合结束后的局部集成候选评估，没有必要据健康样本承诺全域等价或提前写入当前运行。收益真实但局部较小，不宜将它当作当前整机耗时的主要解决项。

证据：[独立 SDK/反例探针 JSON](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/performance_review/finite_contact_validation/independent_contract_probe.json)；[探针脚本](/home/noob/WorkPlace/kcgtest1-improvements-20260916/artifacts/performance_review/finite_contact_validation/independent_contract_probe.py)。原 review.py、prepared_patch_probe.json、prepared_streaming_guard_change.json 和生产文件均未改动。
