# P0 实验执行合同

本文件由同一机器任务清单展开，与主合同一致。

### TASK P0-01 — 只读冻结当前版本与本地资产

**Prerequisites**：无；初始唯一允许项。

#### Scientific purpose
建立可复核起点；区分 source implemented、repo reported、实际可读与实际执行，防止把旧kit测试当当前模型结果。

#### Files to inspect
- README.md
- pyproject.toml
- earthdelta/
- tests/
- scripts/s0_gate.py
- reference/_manifest.json
- plans/plans_v1_0919/v6_draft/research_spec_v6.yaml

#### Files to modify
无。

#### Files to create
无。

#### Implementation requirements
- 读取 git HEAD/status，记录与审查提交的差异；有变动先逐文件对照本包 A01–A13，不 reset/revert。
- 运行本包只读 preflight；只记录 checkpoint 和数据路径的存在/大小，不自动下载、反序列化权重或输出环境密钥。
- 先收集 tests/；在现有依赖齐备、CPU测试限额确定后运行非 slow 测试，保留通过/跳过/失败的实际分母。缺依赖不自动 pip。
- 人工核对本机数据年份、真实时间轴、权重来源、资源上限和运行许可；可读取小型元数据，不先扫描全部数据。
- 后续资源 caps、主指标、校准/确认角色仍为 null 的字段保留 BLOCKED；本任务完成只允许提出 P0-02 放行，不自动执行。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 不新增研究源码；验证预检在非git路径失败而不创建伪baseline；输出不含凭据；输出路径已存在时拒绝覆盖。

#### Command
```bash
python "$PACKAGE/tools/repo_preflight.py" --repo "$REPO" --out "$RUN_DIR/preflight.json"
cd "$REPO" && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/ --collect-only -q > "$RUN_DIR/test_collection.log" 2>&1
# 依赖与CPU限额确认后才执行：
cd "$REPO" && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/ -m "not slow" -q > "$RUN_DIR/pytest_cpu.log" 2>&1
```

#### Expected artifact
- preflight.json
- test_collection.log
- pytest_cpu.log（确有运行）
- asset_inventory.json
- HEAD_RECONCILIATION.md
- decision.json

#### Success criteria
- 实际HEAD/dirty状态和审查SHA均保存；所需目录无猜测；数据与checkpoint状态分别枚举。
- 无资源下也可完成只读清单；测试未运行必须为 NOT_RUN，不填历史192。

#### Failure criteria
- 仓库无法定位或HEAD漂移未经审阅；实际资产状态被猜测；无权限读取。

#### Decision after completion
READ_ONLY_PASS → 记录 P0-02 可放行建议并停止本次执行；未解决版本差异 → BLOCKED。

#### Commit suggestion
`docs(p0): freeze audited baseline and local asset inventory`


### TASK P0-02 — 统一数学、指标、可行域与时间合同

**Prerequisites**：P0-01。

#### Scientific purpose
先修复可能导致不公平分数、非法候选或伪可用性的确定性问题，避免将工程错误解释为科学结论。

#### Files to inspect
- earthdelta/paired.py
- earthdelta/geometry.py
- earthdelta/heads.py
- earthdelta/selection.py
- earthdelta/contracts.py
- earthdelta/data/make_splits.py

#### Files to modify
- earthdelta/paired.py
- earthdelta/geometry.py
- earthdelta/heads.py
- earthdelta/selection.py
- earthdelta/contracts.py
- earthdelta/data/make_splits.py

#### Files to create
- earthdelta/metrics_contract.py
- tests/test_metric_contract.py
- tests/test_selection_domain.py
- tests/test_time_provenance.py
- scripts/audit_contracts.py

#### Implementation requirements
- 新增 MetricSpec：schema、变量/单位/scale、lead、区域/面积、D标识、Q归一化、missing policy。e[B,H,S,F]、du[B,K,H,S,F]，公共 gain reducer，权重按声明的全维度一次归一化。
- paired默认真值减参考；geometry不得隐式另归一化；head返回 gain_analytic 与 gain_calibrated，默认校准关闭；旧 gain 兼容路径明确版本。
- 修正 paired 文档：变换后各端点作差的二次恒等式不要求变换线性；D作用于原始差及线性组合响应需要线性。主线仍固定线性D，不偷换目标。
- finite候选共享 bound/max_active/max_candidates/finite/shape验证，拒绝非法而非悄悄裁剪；explicit no-edit，费用和并列规则一致。
- 所有时间用 timezone.utc；availability_source=reanalysis_retrospective|observed_first_seen|scenario，6h情景不称真实ERA5可用性；事件行ID与天气过程group分离。
- 默认禁用 memory/JEPA/校准；不必删除已有模块。实际强制内容hash的执行入口在P0-03完成。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- e0正负号反例；非均匀Q跨三个实现一致；summary损失差等于gain；FP32真值不先转BF16。
- finite幅度越界、超active、NaN、empty、重复ID与超candidate数拒绝；零编辑总可选。
- 更换TZ得到相同UTC索引；缺历史/未来端点拒绝；版本情景字段缺失不得进入formal。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_metric_contract.py tests/test_selection_domain.py tests/test_time_provenance.py -q
cd "$REPO" && python scripts/audit_contracts.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- contract_checks.json
- metric_spec.json
- schema_migration.md
- pytest.log
- decision.json

#### Success criteria
- 所有新增反例及原相关回归通过；同一float64输入的三条gain路径按预登记数值tol一致。
- 正式字段不再接受placeholder；未定统计阈值不影响代数修复，但阻断后续生死判断。

#### Failure criteria
- 非法候选仍可进入选择；跨时区变动；旧接口静默换指标；修复后未说明历史结果影响。

#### Decision after completion
PASS → 解锁 P0-03；FAIL → STOP_IMPLEMENTATION，不运行任何天气收益比较。

#### Commit suggestion
`fix(p0): align metric sign domains and UTC provenance`


### TASK P0-03 — 修复并执行独立真实 Stormer S0

**Prerequisites**：P0-02。

#### Scientific purpose
证明被编辑的确实是指定checkpoint的预报系统，而不是两个有相同错误的本地路径。

#### Files to inspect
- earthdelta/bridge/stormer_arch.py
- earthdelta/bridge/stormer_bridge.py
- scripts/s0_gate.py
- tests/test_bridge.py
- reference/stormer/inference.py
- reference/stormer/stormer/models/iterative_module.py

#### Files to modify
- earthdelta/bridge/stormer_bridge.py
- scripts/s0_gate.py
- tests/test_bridge.py
- earthdelta/contracts.py

#### Files to create
- scripts/export_upstream_reference.py
- tests/test_upstream_parity.py
- tests/test_differentiable_rollout.py
- tests/test_s0_fail_closed.py

#### Implementation requirements
- pinned official实现为独立参考进程/命名空间，必要时隔离环境。不得把本地函数改名 official；xformers不可用时写 upstream_unavailable，不伪造独立验证。
- NormalizationContract显式policy；official_inference使用零diff_mean，legacy保持只读。核查真实npz与变量顺序、坐标、常量、padding、interval/10。
- checkpoint可信来源+SHA256，谨慎处理 weights_only=False 反序列化；禁止下载后自动不可信pickle。模型加载与正式保存使用明确来源许可。
- 装入已知非零合成adapter测试零系数（仅结构检查），另测非零→零与异常cleanup；S0本身不要求先训练科学bank。
- 版本入口绑定checkpointbytes、adapter/bankhash、normalizationpolicy与数组键、gridcoords、D/Q和continuation。任何不匹配拒绝。
- controlled_rollout保留原末态API，新增显式tensorcoeffs[B,K]及 return_trajectory；参数梯度穿过冻结算子但冻结权重不更新；tuple推理路径不冒充可微。
- gate按 checkpoint、upstream parity、norm、zero-edit、finite数据、targetalignment、stateisol全部判断。删除猜测paper30/80门槛、自动pip及绝对路径；小样例论文RMSE只作另列诊断。
- 优先manifest声明的一个checkpoint，另一个为独立扩展；按每变量标准化空间测误差，dtype-specific tol必须来自重复运行并远小于最小科学效应。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 非零diff_mean文件不会改变officialpolicy；normhash绑定名字/intervalkeys。
- mock一个parity失败、NaN评分、checkpoint不符和资源缺失，finalpass必须false且exit非零。
- 梯度从末时效损失传至coeffs/head与bank，基模型无梯度；实际非零bank零编辑与原模型一致。
- 真实futurepayload poisoning隔离；并发或checkpointrecompute未支持则明确拒绝，不仅改变unused局部变量。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_upstream_parity.py tests/test_differentiable_rollout.py tests/test_s0_fail_closed.py -q
cd "$REPO" && python scripts/export_upstream_reference.py --config "$CONFIG" --out "$RUN_DIR/upstream"
cd "$REPO" && python scripts/s0_gate.py --config "$CONFIG" --upstream-reference "$RUN_DIR/upstream" --out "$RUN_DIR/s0"
```

#### Expected artifact
- s0/config.json
- s0/metrics.json
- s0/S0_GATE_REPORT.md
- s0/checkpoint_manifest.json
- s0/decision.json
- upstream/prediction_identity.json

#### Success criteria
- 真实checkpoint、数据、独立上游全部可核实，所有必需gate通过；skip不算通过。
- 每条parity轨迹标识、误差和tol来源可复算；不存在仅local-local而记official的成绩。

#### Failure criteria
- 缺权重/GPU/数据/上游依赖 → BLOCKED；路径不等价/单位错/NaN → FAIL_S0。

#### Decision after completion
只有真实 PASS 才允许 P0-04 的模型工作；单元测试PASS不解锁天气实验。

#### Commit suggestion
`fix(s0): require independent checkpoint parity and fail-closed gates`


### TASK P0-04 — 冻结合格参考、非零字典和候选集合

**Prerequisites**：P0-03。

#### Scientific purpose
让oracle ceiling评价一个已经合理训练的编辑空间，避免零初始化或随机字典导致假否证。

#### Files to inspect
- earthdelta/lowrank.py
- earthdelta/contracts.py
- earthdelta/bridge/stormer_bridge.py
- earthdelta/data/pull_wb2.py
- earthdelta/data/make_splits.py
- plans/plans_v1_0919/v6_draft/research_spec_v6.yaml

#### Files to modify
- earthdelta/data/make_splits.py
- earthdelta/contracts.py

#### Files to create
- scripts/prepare_survival_spec.py
- scripts/train_reference_bank.py
- earthdelta/data/paired_index.py
- tests/test_bank_manifest.py
- tests/test_candidate_registry.py

#### Implementation requirements
- 优先复用已训练但需验证的Fs/bank；若没有，只在批准预算内训练静态Fs及小K bank。F0与Fs分开报告。
- 确认checkpoint预训练/选择年份；按实际时间轴分配fit/dev/calibration/confirm，不随误差重选天气事件。已经用于S0和debug的2020片段标dev/exposed。
- Finite registry先用 no-op + 单专家有界强度；强度在fit/dev固定。若允许负幅度，训练覆盖/诊断适用域；不能以未训练外推域的失败杀整个方向。
- K和rank为可校准规模参数，不新增网络。至少检验bank非零、每专家作用、响应多样性以及同总参数的强静态参考。
- 冻结完整候选hash、hold窗口、参考之后继续规则、主lead与Q；先phase小lead6/24且72guard，不承诺十天训练。
- prepare_survival_spec须把资源caps、δ_gain/δ_dynamic、noninferiority margins、最大确认样本数的依据登记。字段未定则暂停，不能默认2–3%。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 零B不得进入oracleheadroom判定；nonzeroedit引起真实可重复差异；zeroedit恒等。
- 候选构造一致、train/dev/confirm不按candidate随机拆分；同init各lead归同processblock。
- 数据缺时次、边界未来不足、重复坐标/单位不匹配必须报错或显式缺失，不生成空假目标。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_bank_manifest.py tests/test_candidate_registry.py -q
cd "$REPO" && python scripts/prepare_survival_spec.py --config "$CONFIG" --out "$RUN_DIR/spec"
cd "$REPO" && python scripts/train_reference_bank.py --config "$RUN_DIR/spec/frozen_config.json" --out "$RUN_DIR/bank"
```

#### Expected artifact
- spec/frozen_config.json
- spec/threshold_certificate.json
- bank/bank_manifest.json
- bank/training_log.jsonl
- bank/checkpoint_identity.json
- decision.json

#### Success criteria
- bank合格且非零；同预算static已训练；split实际可取数据成立；已冻结候选与阈值依据。

#### Failure criteria
- 零或未训练bank → BLOCKED_INVALID_DICTIONARY，不等于科学STOP；达到cap仍无可用bank → STOP_CURRENT_SETUP。

#### Decision after completion
PASS → P0-05；资源未批准不自动下载/训练。

#### Commit suggestion
`feat(p0): freeze qualified reference bank and survival registry`


### TASK P0-05 — 完整有限候选 oracle ceiling 与静态差距

**Prerequisites**：P0-04。

#### Scientific purpose
第一项决定方向是否值得继续的天气实验；区分普遍后处理收益和真正逐状态选编辑空间。

#### Files to inspect
- earthdelta/probe.py
- earthdelta/paired.py
- earthdelta/teacher.py
- earthdelta/selection.py

#### Files to modify
无。

#### Files to create
- scripts/p0_oracle_ceiling.py
- earthdelta/evaluation/survival.py
- tests/test_oracle_table.py
- tests/test_block_statistics.py

#### Implementation requirements
- 复用 cached_responses 全部预登记候选的实际非线性响应，不用中心导数替代；不把 verify_candidates top3称全库oracle。
- 每起报一次reference，候选共享输入/目标/边界。保存所有候选成本、gain、失败和无编辑；缺候选不能当completeoracle。
- best-static/regime只在fit/dev选定后应用于confirm；noedit/random固定seed；hindsightoracle单列，不作为可部署模型。
- 主headline检验oracle相对Fs和相对best-static两种gap；init/process配对bootstrap，按约定主指标，未有足够功效 INCONCLUSIVE。
- 保留global rawMSE、perlead/pervariable、sqrt-after-aggregate RMSE、正负收益分布。用同注册机会分母；失败策略预登记，不静默剔除。
- 先dev，冻结后confirm一次；缺足够独立样本不得改用网格点扩大N。δ的确认只依赖pilot和业务/研究价值，不依据confirm选择。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 含noedit则completeoraclegain>=0；oracle选项可复算；validationbeststatic不是testargmin。
- top3/subsample输出必须标tested-candidate ceiling而非full ceiling；nonfinite候选阻断complete。
- 同weather过程不同lead聚合/配对一致；全体损失先聚合后sqrt，统计不把候选或网格当独立样本。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_oracle_table.py tests/test_block_statistics.py -q
cd "$REPO" && python scripts/p0_oracle_ceiling.py --config "$CONFIG" --role dev --all-registered-candidates --out "$RUN_DIR"
# 独立确认角色只在预登记冻结后开启：
python scripts/p0_oracle_ceiling.py --config "$CONFIG" --role confirm --all-registered-candidates --out "$CONFIRM_DIR"
```

#### Expected artifact
- candidate_outcomes.parquet
- oracle_static_gaps.json
- bootstrap_intervals.json
- metrics.json
- predictions/index.json
- decision.json

#### Success criteria
- 完整注册集、正确bank、无违规后，主oraclegap和dynamicgap的lowerCI超过各自已注册最小有用差异。
- 若只有dev过关，只标 DEV_PROMISING，不称confirm成功。

#### Failure criteria
- 充分功效下upperCI低于δ → STOP_CURRENT_DICTIONARY；dynamicgap不足 → PIVOT_STATIC_ADAPTATION。
- 缺样本/候选失败/CI跨阈值 → INCONCLUSIVE或BLOCKED，不继续大训练。

#### Decision after completion
PASS或明确批准的DEV_PROMISING → 有界 P0-06/P0-07；STOP不解锁；确认结论在P0-08统合。

#### Commit suggestion
`feat(p0): measure exhaustive finite-edit headroom and static gap`


### TASK P0-06 — 交叉拟合双头、直接gain与oracle分解

**Prerequisites**：P0-05。

#### Scientific purpose
判断瓶颈在e0方向、du响应还是选择，以及分解是否比直接收益预测更有价值。

#### Files to inspect
- earthdelta/heads.py
- earthdelta/paired.py
- earthdelta/selection.py

#### Files to modify
- earthdelta/heads.py

#### Files to create
- scripts/p0_predictability.py
- earthdelta/baselines/direct_gain.py
- earthdelta/evaluation/predictability.py
- tests/test_oracle_decomposition.py
- tests/test_head_contracts.py

#### Implementation requirements
- 保留现有多lead ComposedPredictionHead，默认memory为显式零输入并锁对应层偏置行为、JEPA off、calibration off；不要另造相同head。
- 同history/context/candidate descriptors/HPO预算训练：e0/du、直接scalar gain、直接edited forecast、mean response/no-effect。
- 所有heads和normalization只从fit，超参dev，按起报/过程crossfit生成验证预测；绝不让同issue不同candidate散到train/test。
- 执行4格：真e真du、真e预测du、预测e真du、预测e预测du，同一actual outcome结算。oracledu为额外运行结果、只能诊断；不能把它作为低成本serving。
- 报告whole-e0和响应span内e0准确度；du relative weighted error、cos、R²、gain误差、ranking、regret、harmful/noedit；nearnull响应指标显式N/A。
- 新增no-op结构零：u_hat(x,a)=r(x,a)-r(x,0)或显式noeditmask；默认无额外gain校准。实际未经训练不声称低误差。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- zeroeditdu/gain精确零；candidate置换同步输出；oracle标签poison仅改变评分不改变pred输出。
- 4格真正替换相应量；normalization和Q一致；truth-shuffled sanity；nullresponsecos不虚报1。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_oracle_decomposition.py tests/test_head_contracts.py -q
cd "$REPO" && python scripts/p0_predictability.py --config "$CONFIG" --paired-table "$ORACLE_TABLE" --out "$RUN_DIR"
```

#### Expected artifact
- crossfit_predictions.parquet
- four_cell_scorecard.json
- direct_gain_comparison.json
- response_metrics.json
- fit_budget.json
- decision.json

#### Success criteria
- pred/pred相对static的lowerCI达到已注册价值门槛；du比zero/mean具有决策相关信号。
- 分解相对directgain若无优势，不判科学FAIL，但必须降级factorization贡献并按已注册label/candidate实验检验。

#### Failure criteria
- du无信息 → STOP_RESPONSE_SCALE_OR_DESCRIPTOR；e0失效 → PIVOT_ERROR_ESTIMATION；oracle两项好但planner差 → FIX_METRIC_OR_SELECTION。
- 达到预算后pred/pred不优于static → STOP_CURRENT_PLANNER；不允许用校准项隐藏失败。

#### Decision after completion
完成4格后不自行添加JEPA/memory；与P0-07共同交P0-08决定。

#### Commit suggestion
`feat(p0): cross-fit paired response and direct-gain diagnostics`


### TASK P0-07 — 参数编辑对输出反馈纠错挑战

**Prerequisites**：P0-05, P0-06。

#### Scientific purpose
直接检验参数干预是否有必要；多步传播和多变量一致性并非参数编辑独占。

#### Files to inspect
- earthdelta/bridge/stormer_bridge.py
- earthdelta/heads.py
- earthdelta/lowrank.py

#### Files to modify
无。

#### Files to create
- earthdelta/baselines/output_correction.py
- scripts/p0_output_challenge.py
- tests/test_feedback_corrector.py

#### Implementation requirements
- 实现三类baseline：联合multi-lead末端纠错；每步将全变量修正反馈到下一步的feedback corrector；使用同候选响应预测的virtual edited forecast = reference + predicteddu。
- 使用同合法历史/预览信息、目标变量、训练样本、损失及HPObudget；分别报告同参数机制比较和同实测GPU成本前沿，不强行同时相等。
- feedback路径起报后只消费自己的预测；与参数编辑共用hold窗口/时效；不能称其只能改一步。
- 参数编辑与输出纠错都要在6/24/72/120h报告变化，早期改善晚期回退独列；240h只在资源准入后。
- 预设primary/guard变量，不只展示最有利变量。若virtualprediction已与actualediting等效且便宜，削弱实执行编辑必要性。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- corrector=0严格还原Fref；第一步修正会通过后续Fref传播；目标数据不进入forward。
- 相同预算/输入role守卫；actualedited与virtual prediction分别保存、不能把后者当真实执行。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_feedback_corrector.py -q
cd "$REPO" && python scripts/p0_output_challenge.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- correction_frontier.json
- lead_variable_metrics.parquet
- rollout_stability.json
- information_cost_ledger.json
- decision.json

#### Success criteria
- 参数编辑在预登记的指标/资源约束下有可重复优势或明确的有限样本效率优势，不必虚构全面支配。

#### Failure criteria
- feedback/output在关键效果与成本上统计支持的支配 → PIVOT_OUTPUT_TRAJECTORY_REPAIR。
- onlyone-stepgain、longleadupperharm beyondmargin → STOP_ROLLOUT_CLAIM；差异不清 → INCONCLUSIVE。

#### Decision after completion
与P0-06一起进入P0-08；禁止以“输出不改动力学”回避负结果。

#### Commit suggestion
`feat(p0): challenge edits with feedback and virtual-output correction`


### TASK P0-08 — 生存门审议与主线冻结

**Prerequisites**：P0-06, P0-07。

#### Scientific purpose
把工程合格、干预有空间、可预测、分解有用、参数必要性分开做决定，避免无限加模块。

#### Files to inspect
- artifacts/p0/
- STOP_CONDITIONS.md
- plans/

#### Files to modify
无。

#### Files to create
- scripts/p0_decision.py
- tests/test_research_gate.py
- plans/current/FINAL_RESEARCH_DECISION.md

#### Implementation requirements
- 生成每个门的status、effect、CI、δ依据、样本功效、失败/资源覆盖；独立确认不足不能用dev代替。
- 区分CONTINUE、INCONCLUSIVE_WITHIN_CAP、BLOCKED、STOP_CURRENT_DICTIONARY、PIVOT；所有critical门必须成立才解锁P1。
- 冻结最多三条贡献和一条P2增强；若directgain等价不再把identityfactor当创新；若outputdominate转向另立计划不暗改目标。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 任何critical missing/NaN/S0fail不可CONTINUE；CIupper小于δ和CI跨δ分别STOP/INCONCLUSIVE。
- 预算达到cap禁止自动增样；新确认数据开封后禁止调δ。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_research_gate.py -q
cd "$REPO" && python scripts/p0_decision.py --runs "$P0_RUN_INDEX" --thresholds "$THRESHOLDS" --out "$RUN_DIR"
```

#### Expected artifact
- FINAL_RESEARCH_DECISION.md
- gate_table.json
- UNLOCK_P1.json
- decision.json

#### Success criteria
- 判定和证据一一对应，critical gate皆PASS且确认角色合法。

#### Failure criteria
- 核心收益不成立、资源未核实或不公平比较 → 不解锁P1。

#### Decision after completion
CONTINUE → 按依赖解锁P1；其他状态停止当前执行并输出精确原因。

#### Commit suggestion
`docs(p0): freeze survival decision and paper claims`
