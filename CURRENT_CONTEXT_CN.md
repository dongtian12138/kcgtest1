# kcgtest1 当前上下文

## 当前唯一目标

在 Isaac Sim 的 simulation-only 场景中，让三根真实指腹接触电连接器，保持有限夹持，
把连接器从桌面抬升 50 mm，并保持至少 2 s。

## 已验证事实

- Isaac 能加载现有机器人、三指手、连接器和 GPU physics。
- 手指能够执行闭合运动，DirectGPU 接触数据通路可用。
- 现有运行尚未形成一次真实手—连接器接触；三指抓取、离桌、50 mm 抬升和 2 s 保持
  均未验证。
- 四个保留入口当前可完成 Python 导入，但旧离线候选结果已经删除；
  `run_grasp_lift.py` 尚未解耦该输入，因此当前不能直接启动抓取。
- `research_dynamic_pass=false`、`formal_dynamic_pass=false`、
  `hardware_authorized=false`。

## 唯一活动动作链

```text
加载现有场景
  → 把手放到三指包围连接器的预抓姿态
  → 三指低速闭合并形成真实接触
  → 保持有限夹力
  → 手上移 50 mm
  → 保持 2 s
```

只保留现有运行入口：

- `src/kcg_connector/isaac/carts_v2/run_grasp_lift.py`
- `src/kcg_connector/isaac/carts_v2/controller.py`
- `src/kcg_connector/isaac/carts_v2/evaluate_run.py`
- `src/kcg_connector/isaac/carts_v2/engine_health.py`

## 实施边界

- 不恢复已清理的候选扫描、离线搜索、优化或方法认证路线。
- 不新增 optimizer、planner、manager、telemetry、方法认证、候选 ledger 或一次性 runner。
- 不调用外部执行代理。
- 优先直接调整预抓手位姿、闭合目标和有限夹力；一次失败只处理一个实际物理原因。
- 不以代码量、测试数量、候选数量、退出码或报告数量冒充物理进展。
- 不启动真实硬件，不使用磁吸、隐藏固定、运行后写物体位姿或无限增力。

## 下一步

先原位修改现有 runner/controller，使其直接读取一个手工登记的预抓关节姿态和三指闭合
目标，不再读取候选报告。随后执行一次直接三指闭合—抬升尝试。判断顺序固定为：

1. 三根指腹是否真实接触连接器；
2. 连接器是否离开桌面；
3. 是否上升 50 mm；
4. 是否保持 2 s 且不掉落。
