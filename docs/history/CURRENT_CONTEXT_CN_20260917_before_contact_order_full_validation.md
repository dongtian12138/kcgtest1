# 当前任务：完成用户截图的四项装配改进与交付

核验时间：2026-09-17 01:48 UTC。用户要求保存可复现基线、减少计算/记录耗时、查清腕部跟踪偏差、保持原2µm键槽检查并做完整重复与小姿态变化。simulation-only，hardware_authorized=false；同一时间只运行一个主物理实验。

## 当前实际结果与下一步

- 改进目录：/home/noob/WorkPlace/kcgtest1-improvements-20260916，分支codex/connector-assembly-improvements-20260916。首轮完整实验使用已推送提交c95785ca3d81798146ef7b464c86b840c722f3ab。
- 首轮artifacts/full_validation/load10ms_repeat01/run已经结束，无物理进程。277530物理帧，motion_timing存在；原始归档19090801262字节，与索引末端一致。包装器记录退出2，墙钟12859.883秒（214.33分钟）；旧plan状态未更新，已按实际日志补记结束，保留补记前副本。
- 五段指令分别90、46.874383、90、90、37.876059度，累计354.750442度。第2、第5段为ELASTIC_RESERVE_EARLY_REGRASP正常余量换抓，均无outer_abort。第4段实际Nut转90.112785度、Body前进2.014607mm。
- 最终松手后的Body深度14.582080096mm，距离14.605mm还差22.919904µm，超过原10µm到位范围，不能算完整成功。最终release独立核验：2880帧/3秒逐帧零手冲量、同128簧套加载、Body/Nut未休眠；原StopBox每帧零正载。因此稳定但未坐底。
- 首轮完整后评全部结束：whole_assembly_review为REVIEW_REQUIRED，源指甲/指腹与所有视觉阶段通过；第三键最大穿透2.475998µm(raw240346)，第五键2.141567µm(raw225936)，保持原2µm失败。原生接触与前一拍后姿态对齐吻合。最大被动轴向游隙.506907mm<.5999mm。独立代理本次只读补核已结束。
- 当前唯一物理实验：artifacts/local_tracking/repeat01_third_solver_order_default，exec session57303，2026-09-17 01:45 UTC左右启动；900秒墙钟、120秒收尾、5000步上限。取首轮完整运行238926第三段夹持末态，10ms/90度/CPU960/64/4，显式solve_articulation_contact_last=false。读回：reset前后均false，19项Actor/API均64/4，scene位置范围1..255、速度4..4。当前无其他物理实验。
- 对应candidate是repeat01_third_solver_order_contact_last，除solve_articulation_contact_last=true以外配方完全一致，尚未运行。新的局部选项已在diagnose_saved_hand_wrench.py实现，限制CPU TGS960/64/4并保存配置读回；改动尚未提交。A结束先后评，再决定B；勿同时运行，勿把冷起点对照当连续全程。
- 下一步完成第三段原/改求解顺序的有因果对照。用最小局部诊断区分“剩余有限角度需再换抓”与“受载腕跟踪/数值接触”问题。未解决前不盲跑第二个完整回合。通过完整验收后才做第二次同场景重复，再做1mm+1度的明确初始姿态变化。仍不能宣称四项全部完成。

- 末尾局部诊断已纠正：final_regrasp_remaining360在8641步/2102.68秒触及墙钟保护，夹持完成8504，补转仅.208748度。后评发现在没有任何手冲量的重建场景中，Body已从14.582259mm（step0）变到14.597607（480）、14.605773（2000）；Nu角仅变化约.002度，原止挡已承载。因此它不能证明“重抓/补5.25度解决原回合欠行程”，重建没有恢复原接触历史。详见该目录pre_turn_depth_progress_review.json和cold_initialization_interpretation_correction.json。
- 随后的final_held_remaining360（从完成夹持8503起、3600秒预算）已因上述解释失效主动STOP_REQUEST，在739步/278.18秒退出2。没有完整释放或完整装配成功。它的source_stage_probe_result和归档保留，session86761已结束。原第一局部session60515、后评60872/89763也均结束。
- 记录GC仅有离线新候选，尚未接生产：120帧只collect(0)从8.944到6.184秒但老代循环不回收，独立复核指出风险；补成每128帧collect(2)的256帧对照19.251→14.634秒（约24%）、全流哈希一致、缓存跨129淘汰，老代哨兵128帧释放、RSS约1.72GiB。仍需真实受载前缀及完整计时/状态恢复检查，不能把这约24%称为整场提速。路径artifacts/performance_review/gc_frame_boundary和gc_bounded_generations。独立代理这两项只读任务均已结束。

## 已封存交付

- 原工作区/home/noob/WorkPlace/kcgtest1，原脏修改全部保留；保存版/home/noob/WorkPlace/kcgtest1-baseline-20260916不再改动。
- 基线分支codex/connector-assembly-baseline-20260916，提交d64fa4a0a1a394bc800f96191d23aecc3bdad45c已推送。
- Release：https://github.com/dongtian12138/kcgtest1/releases/tag/assembly-baseline-20260916；有297文件运行资产包、原完整291.4秒视频与末段实际片段。新目录从公开Release下载、全部资产校验及Isaac预检通过；全新机器环境安装尚未验证。
- 原完整14达到14.604491442mm，3秒真实松手保持、止挡与128簧套持续承载；但原第三键最差−2.147659492µm，整体仍REVIEW_REQUIRED。不得把基线说成所有数值检查均通过。

## 已验证改进与边界

- 当前正式候选：独立10ms负载前馈滤波，原50ms导纳/观测/停止信号保持；CPU960Hz、64位置/4速度迭代、原有限抓力/速度/硬限、2µm阈值保持。全部native接触点保留，只用路径归属缓存和拥有副本的tuple三向量减少开销。
- 481步匹配受载保持完整1421146150字节MessagePack完全相同；callback减少13.27%，物理/传感/记录合计减少5.42%。局部回放和函数收益不能直接当整场收益。原整场257分钟与本轮214分钟路径/接触历史并不完全相同。
- 第四段同源有效A/B仅改变滤波50→10ms：前者30.80258度指令就换抓，最大虚拟中心XY .579183mm；后者90度完成、Nut90.029598度、最大XY .099038mm。独立审查确认。局部不能保证全程：本轮第2段最大XY仍.692971mm。
- 第二段同源局部50ms/64、10ms/64、10ms/128的最差键穿透为1.678369/.896779/1.188415µm，三者均局部通过2µm；128迭代未改善，正式仍64。冷初始化没有复现原完整14的2.148µm，所以不能当原键槽问题已解决。
- 原键槽超带不是后处理四舍五入：longdouble同值、已保存Body位姿量化界约.014934µm（仅保存输入）。原生接触与归档位姿相差一拍；正确对齐后差约.008µm。源槽壁描述与模型一致，这对接触是凸包/静态三角网格而非SDF。原2µm不放宽。
- 早期配置路由错误的fourth_loaded_baseline02、fourth_loaded_10ms是无效A/B，失败证据保留。原missing-wrench-origin-shift诊断已撤回，勿恢复。

## 运行与后评入口

- 默认scripts/run_current_hand_assembly.py --check/--run/--gui/--preflight-only，绑定762个文件，配置id load10ms_native_lossless_tuple_cpu960_64_4。改动运行源/配置后需明确更新绑定，不能绕过。
- 环境：ISAAC_ENV_PREFIX=/home/noob/WorkPlace/isaacsim/.conda-env；KCG_PLANNER_PYTHON=/home/noob/WorkPlace/kcgtest1/.venv/bin/python；KCG_SAM6D_PYTHON=/home/noob/.cache/kcgtest1-sam6d/.venv/bin/python；KCG_SAM6D_ROOT=/home/noob/WorkPlace/kcgtest1-baseline-20260916/.deps/SAM-6D/SAM-6D。
- 后评reproducibility/improvements_20260916/review_completed_assembly.py，支持body/nut/key/turns/terminal/band/whole；terminal输出须在原run之外。Nut必须使用selected_two_nail_geometry。选择性手接触读取器不能供内部止挡/簧套全验收。
- 实际视频抽帧extract_final_episode_frames.py已用于本轮，图片在run/postrun_evidence；须亲自看图后写可见性结论，不能只根据红带射线推断完整装配。
- 首轮后评工具、报告已提交并推送2ee7778；诊断接触顺序选项、当前说明更新与小姿态准备尚未提交。物理源/配置再次修改后需要更新portable_source_manifest绑定，不能绕过。pose_variation_plan仍待名义场景验证。
- 历史过程详见docs/history/CURRENT_CONTEXT_CN_20260917_before_first_complete_review.md以及reproducibility/improvements_20260916/evidence；原始失败与所有大归档保留。
