# 当前任务：完成用户截图的四项装配工作

核验时间：2026-09-17 02:32 UTC。用户最新指令“继续”；原请求是全部完成：保存可复现基线、计算/记录提速、腕部跟踪根因、原2µm数值检查及完整重复。不要只做报告或部分交付就结束。simulation-only，hardware_authorized=false；一次只运行一个物理实验；原2µm不放宽，原始失败不删除。

## 当前执行位置

- 活动工作树：/home/noob/WorkPlace/kcgtest1-improvements-20260916，分支codex/connector-assembly-improvements-20260916。此前已推送2ee7778（首次完整失败和后评工具）；新数值/记录实现准备提交后启动完整验证。原目录/home/noob/WorkPlace/kcgtest1的原脏资产不动。
- 所有局部物理实验当前均已结束：final_regrasp_remaining360(60515)、final_held_remaining360(86761)、接触顺序default(57303)/contact_last(5453)、third_contact_last_gc_prefix(11757)、dense_gc_prefix(8072)。独立代理/root/independent_final_guard_audit本轮只读任务已全部完成。
- 下一步：提交并推送已准备代码，运行scripts/run_current_hand_assembly.py --run --output-root artifacts/full_validation/contact_last_gc128_repeat01。入口先做新预检，四项通过后才启动完整回合。全程上限21600秒、收尾1200秒，按已测深插成本增加电脑时间预算，不改物理力/速度/角度边界；不运行中延长。
- 当前默认配置id：load10ms_contact_last_gc128_nut_cpu960_64_4_six_grips。配置reproducibility/improvements_20260916/config/visual_contact_last_gc128_task.yaml，命令contact_last_gc128_candidate_command.json；BASE assembly_command已更新。768个源码/资产绑定检查通过，后续改动绑定源须明确更新manifest，不能绕过。
- 正式候选保留10ms负载前馈、50ms原观测/导纳/停止、960Hz、64/4、原模型材料和有限抓力/速度。新增关节接触最后求解、仅螺母阶段GC调度；最多6次抓握（原5次）但总指令仍360度。尚无新候选完整通过回合。
- 新候选完整通过后，再同初始场景完整重复一次，然后做1mm平移+1度偏角。入口已准备--pose-variation，当前不得提前当成已跑；其命令与配置已绑定。第一轮失败后仍须先定位最早问题，不盲重跑。

## 首轮完整改进版：已核验但失败

- artifacts/full_validation/load10ms_repeat01/run使用c95785ca3d81798146ef7b464c86b840c722f3ab；277530帧、19,090,801,262字节原始归档封存。墙钟12859.883秒（214.33分钟），物理289.09375秒、五段旋拧13.565625秒。原257分钟与此接触历史/终点不同，不能把全部差值归因提速。
- 五段已发角90/46.874383/90/90/37.876059度，累计354.750442度，尚余5.249558度但五次预算用尽。第二/第五段正常余量换抓，无硬abort。第五段f1弹性3.501318Nm、余量.498682Nm触线；这是手指传动，不是螺母轴向拧矩。末条腕部观测扭矩.803025Nm，早停止决定一拍。
- 最终Body深度14.582080096mm，差22.919904µm。完整3秒每帧手冲量精确0、同128簧套承载、Body/Nut未休眠，但原StopBox每帧零正载，未坐底。
- 全流第三键最差−2.475998µm(raw240346)，第五键−2.141567µm(raw225936)。原生接触与前一拍步后姿态对齐吻合。原2µm失败；第三键最差时腕虚拟中心误差仅.108mm，非最大误差时刻。
- whole_assembly_review=REVIEW_REQUIRED：全部视觉阶段、源指甲/指腹、真实松手、备份轴向限位通过；机械坐底/止挡和键槽未通过。最大被动轴向坐标.506907mm<.5999mm。实际末段图片已看，红带遮住不能代替坐底。
- 报告docs/ASSEMBLY_FIRST_COMPLETE_REVIEW_20260917_CN.md；小证据随版evidence/load10ms_repeat01，独立补核在该run父目录review/。程序原退出2及未更新plan状态均保留/解释，不能把退出码当物理结论。

## 最近局部数值与记录证据

- 同源第三段238926，default与contact_last只差solve_articulation_contact_last布尔，代码快照相同。19项Actor/API均64/4；local scene位置范围1..255、速度4..4；原模型文件未改。
- default：指令82.744°后90N正常换抓，实际Nut82.197°、Body+1.889mm、最大虚拟中心XY .273672mm，最差键1.714076µm。
- contact_last：90°完成，实际Nut90.583°、Body+2.050mm、最大XY .404106mm，最差键1.459824µm。均无硬abort，均局部2µm通过。冷baseline未重现整场2.476µm，且腕误差未全面改善；只能据此选完整验证候选。证据evidence/contact_order_comparison。
- GC离线120帧仅collect(0)有约31%收益，但独立发现老代循环不会回收，未照搬。新256帧每128步collect(2)：19.251→14.634秒（23.98%），877896192字节一致，跨129/193缓存淘汰，gen2哨兵128帧释放。
- 实际较浅481步38,759,566字节完全相同；实际密集481步1,421,146,150字节完全相同，SHA6f9fb4799734d3789764b2efaecee65592fec49569c40627072b366715e6fae0，物理/传感/记录/计划GC72.585761→60.442991秒（16.73%）。参考为先前已结束相同窗口，非同分钟成对计时；不能外推整场。最长GC暂停.633372秒，尾段GC正常，无错误。
- GC实现保留原advance方法体（AST一致），只延后帧内回收；跨阶段按总步数计数，每128步完整回收并有结束/中止尾段处理，恢复原GC状态，保留原始错误/停止。全程仅Nut阶段启用，早期桌面/搬运保持默认。14项相关测试通过。源码controller.py、te_source_stage_probe.py、run_body_assembly_with_video.py；GC总时间单列，尾GC在closeout计时中。

## 必须保留的诊断纠错

- final_regrasp_remaining360在8641步/2102.68秒触墙钟保护，夹持8504已完成、仅补.208748°。但重建后手还没接触时Body已从14.582259（step0）到14.597607（480）、14.605773mm（2000），Nu角仅变约.002°，StopBox已承载。
- 因此冷重建未恢复原接触历史，不能证明“重抓/补5.25度解决原欠行程”。后续final_held_remaining360从8503夹持末态出发，被主动STOP_REQUEST，739步/278.18秒退出2。两者未计成功；不能继续该无效解释。evidence/cold_remainder_diagnostic保存纠正。
- 原第四段有效50→10ms局部对照：指令30.80258°/90°，最大XY .579183/.099038mm；独立确认其因果范围。原完整/新完整其它抓姿仍可能大误差（新第二段.693mm）。不能恢复已撤回的missing-wrench-origin-shift解释。
- 原key longdouble重算同值，保存Body量化界仅约.014934µm；源平面与模型一致；凸包键/静态三角槽壁，不是该接触对的SDF网格误差。128位置迭代先前局部无收益，未采用。

## 保存版和环境

- 原成功到位基线树/home/noob/WorkPlace/kcgtest1-baseline-20260916，分支codex/connector-assembly-baseline-20260916，提交d64fa4a0a1a394bc800f96191d23aecc3bdad45c。
- Release https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-baseline-20260916，含297运行资产和原291.4秒视频；新目录公开下载/逐文件核验、预检完成。未验证全新机器从零安装。
- 原完整14到14.604491442mm、3秒松手、StopBox/128簧套持续承载，但原第三键−2.147659492µm，整体仍REVIEW_REQUIRED；不可称全部数值检查通过。
- 环境：ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env；KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python；KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python；KCG_SAM6D_ROOT=/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D。
- 后评入口reproducibility/improvements_20260916/review_completed_assembly.py，支持body/nut/key/turns/terminal/band/whole；terminal须输出run外，Nut须选两NAIL+PAD。已封存后才能读真值。extract_final_episode_frames.py从本回合视频选图，需亲自看图再写visibility判定。完整3秒每帧承载由独立terminal补核，不由旧whole的2秒/any规则替代。
- 当前说明docs/ASSEMBLY_CONTACT_ORDER_AND_RECORDING_20260917_CN.md。历史入口docs/history，不重复恢复过期动态任务。
