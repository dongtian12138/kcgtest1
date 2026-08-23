# tools

- `deepseek_consult.py`：外部模型协作入口；仅在当前授权允许时使用。

旧 TASK-R12 runner、报告器和一次性实验脚本已由
`pre-active-route-prune-20260823` 保留，不再占据活动工具目录。

新的可复用逻辑应进入 `src/kcg_connector/kcg_connector/` 并配套测试，不要继续把
无测试的临时代码堆在 `tools/` 根目录。
