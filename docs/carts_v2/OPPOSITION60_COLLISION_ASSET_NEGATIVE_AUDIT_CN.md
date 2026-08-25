# 60°复合凸碰撞资产负结果与源码边界

一句话结论：现有 CoACD、PhysX shrink-wrap 与删块后处理结果都保持失败关闭；它们能说明哪些几何方案失败，但不能直接绑定 Isaac 运行时。

## 历史证据状态

| 结果 | manifest 记录的执行 SHA | 当前对应源码 SHA | 结论 |
|---|---|---|---|
| CoACD 64 块/末节 | `ce2947be…c5b` | builder `9d36a9d7…796c` | `EXECUTED_SOURCE_SNAPSHOT_UNAVAILABLE` |
| PhysX VHACD shrink-wrap | builder `b0e8cd75…bf3`；runner SHA 未记录 | builder `9d36a9d7…796c`；runner `03941694…77b` | `EXECUTED_SOURCE_SNAPSHOT_UNAVAILABLE` |
| prune + convex-hull 规范化 | pruner `73c8d78b…8fd` | pruner `4e46700c…372` | `EXECUTED_SOURCE_SNAPSHOT_UNAVAILABLE` |

旧产物没有被覆盖或重写，且三个结果的 `runtime_binding_accepted` 都是 `false`。由于仓库中没有与记录 SHA 完全一致的执行源码快照，这些数据只保留为历史负结果和方法排除证据，不作为可复现的当前资产。

## 未来失败关闭规则

- runner、builder、pruner 的新输出必须分别记录实际执行脚本 SHA；请求链或源码 SHA 缺失、不一致时拒绝继续。
- 静态几何通过只能写 `STATIC_GEOMETRY_ASSET_CANDIDATE`。
- `runtime_binding_accepted` 保持 `false`，直到运行时导入、初始穿透检查、60°闭合重放和 PhysX 健康四门在独立运行证据中全部通过。
- prune 的实际删除块数量必须与预登记期望一致；不一致时，即使距离和凸性检查通过，也不能成为静态几何候选。

这些规则不改变碰撞容差、2.0 mm 研究上限、相对基线 0.25 mm 门、删除区零占据门，也不产生动态、正式或硬件成功声明。
