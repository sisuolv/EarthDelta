# P0_SURVIVAL_EXPERIMENTS

依据紧前一轮复核，仅三个实验。任务ID属于 `round2-next-iteration-20260921`：本包P0-03对应历史P0-05的oracle问题；本包P0-04对应历史P0-06；本包P0-05对应历史P0-07。不要把旧任务完成记录套在新编号上。

## 公共 estimand 与合法信息

参考模型 F_ref 是已冻结F0或Fs；选择后不得替换。对固定输出映射D和Q_eff：e0=D(Y)-D(F_ref(x))；u_a=D(F_a(x))-D(F_ref(x))；g_a=2 e0^T Q_eff u_a-u_a^T Q_eff u_a。主目标是所有注册起报机会的等权平均收益，过程只是相关性/重采样单位；不要悄悄变成每个过程等权的另一个总体。

I包括签发可用历史、已申明和计费的参考预览、注册动作及预算。I不包括未来truth、真实未来误差、实际全部编辑输出、oracle b/H、未来已核验memory。数据是回顾性open-loop研究；不冒称ERA5在签发时可即时取得。标签可在离线训练/评分侧使用，所有serving对象须与标签对象分离。

端点恒等式对实际非线性候选成立；u=Ra、singleton叠加是额外假设。当前主line使用实际有限响应。H非对角是响应重叠，不等于非线性混合干预；无组合实验就不写组合贡献。

Q先固定变量scale、lead权重、真实面积/空间权重和投影。Q_eff=(q/s²)/sum(q)只按H,S,F归一化一次。batch不是目标维。perlead/pervariable诊断另列；RMSE先汇总平方误差再开方。加性校准off，ridge/成本项如使用与Q单位同步记录。

## 预算与 no-edit

每一可部署方法计 C_context+C_plan+C_selected_rollout+C_decoder+C_retry/fallback。no-edit的raw gain为0，不代表总成本为0。绝对预算过滤 total_cost<=B；可选净效用用 g-lambda*(C_a-C_noedit)，其中lambda有明确单位和预登记。候选采集时运行全部分支属于离线仿真成本，不冒充低成本serving；真u诊断同样单列。

## E1：冻结字典的完整候选余量（P0-02/03）

**假设：**合理训练的最小非零bank存在有用的oracle相对F_ref和强静态差距。

**比较：**F0信息项、F_ref/no-edit、dev-best-fixed candidate、冻结regime、hindsight oracle。所有方法同registry和合法成本。强静态适配已在bank阶段建立，不能只拿原始F0作薄弱对照。

**估计：**V_oracle=E[max_a g_a]；报告V_oracle相对reference与已冻结static policy的gap。V_static<=V_I<=V_oracle，其中V_I=E[max_a E[g_a|I]]。反例e=±1独立I，u∈{0,±0.5}：oracle .75、合法最优0；PASS仅支持尝试E2。

**实现：**使用一个清单内所有候选的实际非线性rollout，保留全部lead与返回轨迹。top3复验或缺候选表不称complete oracle。每个候选运行从相同初始状态开始。已有cached_responses可复用，API结果需加实际调用/成本记录。

**oracle FAIL范围：**仅当前bank/registry/Q/lead/cap。singleton-only不能否证未测组合。多候选使事后max上升是estimand性质；固定registry后不是自动统计假阳性，但不能因此声称有可用信息，也不能看结果后扩候选“捞”收益。打乱对照若已有cap可作诊断，绝不机械扣除打乱oracle。

**产物：**candidate_outcomes.parquet、predictions_index.json、failure_ledger.jsonl、oracle_static_gaps.json、bootstrap_intervals.json、variance_power.json、cost_report.json。

## E2：合法策略与监督分解（P0-04）

**假设：**I中的可用信号能实现一部分动态价值；paired在同资格/资源下有不同于direct-gain的有限资源优点。

**比较：**best-static、regime、direct-gain(I,a)、paired_e_u(I,a)、direct-edited-forecast(I,a)。同缓存可选对比只学习误差在响应span内的投影/b；不得为了该变体另开大规模仿真。

**合法资源：**相同历史/context/candidate descriptors、相同仿真资格与标签子集，公开参数/epoch/HPO/seed/仿真成本。每起报一份truth足以构造所有候选gain，不能把其重复K次计成K倍标签。direct-gain允许同样仿真辅助任务；direct-edited-forecast的端点目标本身也无需未来truth。

**训练/评分：**小型heads在过去的fit/dev_select内训练，按真实过程向前交叉拟合至dev_screen。同issue全部候选和lead同fold，数据统计不跨折。先保存无标签action_selection，再由offline scorer join候选结果。每格oracle替换必须标OFFLINE_DIAGNOSTIC。

**四格：**真e真u、真e预测u、预测e真u、预测e预测u；全部用同真实候选结果结算。真u是额外仿真，不是免费预测。还报告du方向/范数、响应span内误差、gain/ranking、实际regret、harmful/noedit与总成本，但不凭整体e0 MSE决定生死。

**判决：**部署策略对static的主收益必须在CI下达到预登记价值。paired与direct-gain等价且无登记资源优势，撤下分解主贡献。oracle好合法policy弱，到cap停止扩张；不以更大编码器、新memory或calibration续命。

**产物：**crossfit_predictions.parquet、action_selection.jsonl、four_cell_scorecard.json、direct_gain_comparison.json、label_simulation_compute_ledger.json、models/folds.json。

## E3：参数执行必要性（条件性P0-05）

**触发：**E2有合法收益及继续理由，且单独批准训练预算。否则不实施。

**比较：**实际编辑；virtual(reference+predicted-u)；一个轻量joint multilead corrector；一个feedback corrector。每步feedback将自身修正后的完整状态输入下一步，不读新真值。无需新的backbone。

**可比性：**压缩D下的virtual仅在该D内比较；需要全场输出就付出解码器成本和误差，不能用不存在的逆D。feedback无限容量等价性不是廉价构造，不预设谁赢。所有方法同信息、目标和可比资源，单列训练与推理开销。

**估计：**编辑相对输出纠错的配对净价值、资源前沿和保护时效。只在实际已注册lead内比较；当前起点6/24/72h，未预登记不得事后选择lead或扩大到10天。

**产物：**comparison.parquet、resource_frontier.json、rollout_protection.json、final/DECISION.md、final/decision.json。

## Dev / confirm 与统计预登记

1. bank_fit与policy_fit/dev_select/dev_screen按实际索引声明；有资源可复用历史fit，但各角色及暴露状态必须明确。所有样本的历史和未来端点都要通过admission。
2. 当前只读dev。E1的DEV_PROMISING可申请E2/E3，不必先消耗confirm。最终模型、registry、metric、阈值、比较与失败规则全部冻结后，明确授权才打开独立confirm一次。
3. paired bootstrap按weather process或预登记的连续起报块抽样，块内保持完整candidate/lead向量；每次重新计算所有注册起报的总体mean，不把候选数/网格点当独立N。过程划分不得使用哪个候选赢作为分组依据。
4. 指定主comparison及必要保护项；多次looks/多重主检验的规则先写，不能随CI反复窥视。先以pilot估计方差和成本，再确认alpha/power/N及cap；delta_min的价值依据独立写清。
5. 所有delta字段当前为 `TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。它不表示delta_min=MDE；未填certificate时仅descriptive，不签科学PASS。资源不足达到足够精度时保持INCONCLUSIVE。

## 失败分母与表schema

必需列：iteration_id,run_id,issue_id,process_id,role,fold_id,plan_id,valid_times,reference_hash,bank_hash,registry_hash,metric_hash,input_hash,status,failure_reason,loss_ref,loss_edit,raw_gain,total_cost,lead_metrics,prediction_ref。

每个registry×issue必须有行。不可用真值/输入在开始前按共同admission排除并记录；运行后失败不能从分母删除。基础设施缺候选阻断complete；数值失败仅按预先定义且真实可执行的回退/惩罚规则结算，否则保留BLOCKED。在失败规则未冻结前不得汇总成胜利表。

## 决策范围

E1 PASS只支持E2；E2能支持合法收益但不自动证明执行必要；E3才能支持参数执行相对纠错的有限资源价值。所有阶段通过仍仅允许提出下一迭代，不自动进入P1/P2。完整停止条款见 ../STOP_CONDITIONS.md。
