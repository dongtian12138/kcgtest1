本目录保存初始位姿变化失败的证据副本。原始回合不被改写。首次正载接触的时间序列限定在14700–14719窗口，接触回调和步后姿态不被视为严格同时采样。独立结论不把时间相关当成唯一因果。

independent_extract_window.py 是按原执行位置归档的只读分析脚本，保留其目录假设，不是完整装配入口。若需重算，将该脚本单独放在工程下 artifacts/control_review/<新目录>/ 中，准备两回合原始归档在 artifacts/full_validation/ 对应案例目录；使用带 NumPy/msgpack 的 Isaac Python执行。脚本拒绝覆盖同名结果。公开原始包仅含本次变化失败，完整名义26.53 GB归档在本机保留；也可运行同版流程产生新的可比较数据，但新回合不是旧回合的字节副本。

当前装配复现入口是 scripts/run_current_hand_assembly.py，按 docs/REPRODUCE_CURRENT_HAND_ASSEMBLY_CN.md 准备环境与资产。改变初始位姿的选项已观察到本次失败，不保证成功。
