# tools

- `agent_control/`：受控检查、运行和证据工具；修改前必须核对当前任务授权。
- `experiments/`：一次性或已结束阶段的分析脚本，不属于产品运行时。
- `deepseek_consult.py`：外部模型协作入口；仅在当前授权允许时使用。

新的可复用逻辑应进入 `src/kcg_connector/kcg_connector/` 并配套测试，不要继续把
无测试的临时代码堆在 `tools/` 根目录。
