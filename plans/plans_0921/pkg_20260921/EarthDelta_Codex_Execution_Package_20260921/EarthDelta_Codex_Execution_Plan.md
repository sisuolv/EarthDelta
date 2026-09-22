# EarthDelta Codex Execution Plan

## 0. Mission

当前目标是对**冻结参考天气模型的有限、可逆参数编辑**学习多时效输出响应，并检验 `e0/du` 分解相对直接gain与反馈式输出纠错是否具有同资源下的真实价值。本轮先做 P0 生存实验，不启动完整大训练。

本合同基于实际审查 main@4fe55a7af90ea92f62a3232a571af92bfbd6114d。当前代码不是旧 Notion 的 Complexity Atlas；也不是旧 ResponseKit 的未接入原型。本包是研究决策与工程任务，不是已经应用到仓库的补丁。

## 1. Repository Ground Truth

- 分支main；审查提交见上。工作机HEAD如已变化必须P0-01对照，而非reset到旧提交。
- 主要目录：earthdelta/、bridge/、data/、scripts/s0_gate.py、tests/、plans/plans_v1_0919/、reference/_manifest.json。
- 已读源码实现：paired有限du、FD响应、局部geometry/teacher、finite/QPselection、多leadheads、K-expert dense/sparse、localStormerbridge、dataindex、memory/spectral部分。
- README报告192通过3跳过（195用例），本轮**未在完整repo重跑**。已有随机smallmodel测试不能替代realcheckpointparity。
- README报告2020 ERA5 1464×69×128×256；本次没有本地bytes验证，标 REPOSITORY_REPORTED_ONLY。
- 两个大型checkpoint存在/加载、GPU S0与所有天气gain：本次UNVERIFIED；README仍S0 pending、S1+notstarted。
- 当前阅读未找到实际oracle/head训练/长期确认结果；不填写任何天气准确率。
- JVP、band_energy、single_mode_energy明确占位；memory包含异常归零风险。已有正确代码不要推翻。

完整静态发现见 research/CURRENT_METHOD_AUDIT.md、evidence/audit_findings.json；“未发现结果”仅指所读取范围，不代表检查了用户所有远端磁盘。

## 2. Scientific Contract

**必须保留**：
1. F0与静态Fs身份分离；S0对官方F0，主干预zeroedit对已声明Fref。冻结权重内容hash不变。
2. e0=truth−reference，du=edited−reference；同D/Q下gain=2<e0,du>−||du||²。主线D固定线性，变量尺度/面积/lead权重锁定。
3. finite response是实际端点差；R是局部导数；预测head是近似。三个量不同，必须有response_kind字段。
4. serving只读合法历史、明确付费的referencepreview、已训练模型和候选描述。未来真值、实际allcandidateoutputs、oracleteacher只在离线/训练。
5. 无编辑始终可行且解析gain=0；误修正/失败/缺失保留在共同注册分母。
6. 完全自主滚动不读取后续分析；prequential新分析属于另一个预登记协议。
7. 候选字典训练、超参/阈值、确认数据角色隔离；同issue/过程所有候选在同split。
8. 精度、变换、预测路径、窗口、参考模型和代价合同在所有方法一致。普通输出反馈基线不得被削弱。
9. 跨预算/时效/新专家泛化必须测试；新专家内容不在当前descriptor中，不能宣称zero-shotbyconstruction。
10. 预算区分系数域、实际adapter算子成本、端到端时间与训练teacher成本。已有稀疏分支真的跳过，但不等于整体明显加速。

## 3. Non-goals

不新增JEPA、gated memory、dynamicrank、复杂谱训练、RL、第二backbone；不重写已有多lead头/ExpertLoRA；不把synthetic实验计天气结果；不重构无关下载代码。外部源码许可不明不复制，旧kit不覆盖当前repo。

## 4. Current Blocking Questions

Q1 独立真实Stormer parity和norm是否成立？Q2 合格非零bank的完整oracle与beststatic差距是否足够？Q3 du在同状态、未见样本上有决策相关预测信号吗？Q4 e0沿response方向可预测吗？Q5 分解是否比同信息directgain有价值？Q6 参数编辑是否胜过feedback/virtualoutput？Q7 短期收益是否保留到未来且值得preview/执行成本？

## 5. Execution DAG

```text
P0-01 只读基线（初始仅允许此项）
  ↓
P0-02 数学/指标/候选/时间合同
  ↓
P0-03 独立真实S0 ─失败/缺资产→ FAIL_IMPLEMENTATION / BLOCKED
  ↓
P0-04 合格非零bank + frozen finite registry + δ证书
  ↓
P0-05 完整oracle与static gap ─不足→ STOP_CURRENT_DICTIONARY / PIVOT_STATIC
  ↓
P0-06 4格+directgain
  ↓
P0-07 feedback/output挑战（使用P0-06的实际部署结果）
  ↓
P0-08 统一生存门 ─不通过→ 停止/有界补证
          ↓ 仅全部critical通过
        P1-01 成对轨迹数据
          ├────────────────┐
          ↓                ↓
        P1-02 双头serving  P1-03 强基线
          └────────┬───────┘
                   ↓
        P1-04 组合/近似诊断 → P1-05 独立确认
                               ├→ P2-01 未见程序轨迹泛化（主要增强）
                               └→ P2-02 选择性校准（仅有瓶颈才做）
P3模块永久DEFER，不能由P1自动触发。
```

阶段转移不仅检查文件存在：`decision.json`必须绑定数据/权重/confighash、必需测试和效应证据。当前一次执行完成P0-01后停止；其后每次沿允许DAG推进，不自动跨phase。

## 6. 命令与目录约定

```bash
# 由执行者设置为真实路径，禁止猜测或直接覆盖现有文件。
export REPO=/absolute/path/to/EarthDelta
export PACKAGE=/absolute/path/to/EarthDelta_Codex_Execution_Package_20260921
export CONFIG=/absolute/path/to/frozen_run_config.json
# RUN_DIR 每次须全新；正式任务可位于 REPO/artifacts/<phase>/<run-id>/。
export RUN_DIR=/absolute/path/to/new_run_directory
mkdir -p "$RUN_DIR"
```

只有 `tools/repo_preflight.py`、`tools/check_math.py`、`tools/validate_package.py` 是本包已提供可运行工具。下文 `scripts/p0_*.py` 等标于 **Files to create** 的命令是**实现后的验收CLI合同**，并非声明当前repo已有。Codex必须先创建该脚本并测试，再运行它。`--config/--out/--role` 必须按下面规格实现。输入不齐时非零退出，写BLOCKED状态。

额外路径如 ORACLE_TABLE、P0_RUN_INDEX、HEAD_CHECKPOINT、THRESHOLDS、CONFIRM_DIR 均从上游任务manifest解析，不凭文件名猜测。所有新脚本禁止import时执行pip、下载、训练或改环境。

## 7. Detailed Codex Tasks
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


### TASK P1-01 — 构建完整成对轨迹数据与来源链

**Prerequisites**：P0-08。

#### Scientific purpose
复用训练侧无需未来标签的du与需要核验的e0，形成可重建、非泄漏的数据集。

#### Files to inspect
- earthdelta/probe.py
- earthdelta/bridge/stormer_bridge.py
- earthdelta/data/paired_index.py

#### Files to modify
无。

#### Files to create
- earthdelta/data/response_dataset.py
- scripts/collect_paired_trajectories.py
- tests/test_response_dataset.py

#### Implementation requirements
- 输出 reference/edited trajectory及固定线性D的summary；时效axis显式；给e0标签与du模拟标签分别标role。
- du虽然无futuretruth标签仍消耗backbone运行；所有候选模拟数、GPU时间、缓存共享条件计账。
- bank变更自动失效；禁止开发后缓存跨bank复用；filehash+keymap+operationlineage；流式采集不复制多份history。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- split按issue/process不是按candidate；empty/missinglabels不被填未来；缓存篡改拒绝；端点差identity复算。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_response_dataset.py -q
cd "$REPO" && python scripts/collect_paired_trajectories.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- response_index.parquet
- response_dataset_manifest.json
- trajectory_store/
- cost_ledger.json
- decision.json

#### Success criteria
- 全部登记init/candidate/lead可追溯，标签来源明确，缺测保留且不夸完整率。

#### Failure criteria
- 任何train/test泄漏、跨bank旧cache → STOP_DATASET；资源上限到达 → BLOCKED或完整已界定子集。

#### Decision after completion
PASS→P1-02与P1-03；不新增数据灾种或backbone。

#### Commit suggestion
`feat(data): build versioned finite-edit trajectory pairs`


### TASK P1-02 — 训练双头并建立无标签 serving 接口

**Prerequisites**：P1-01。

#### Scientific purpose
将已有组件串成真正的部署系统，而不是训练侧oracle展示。

#### Files to inspect
- earthdelta/heads.py
- earthdelta/selection.py
- earthdelta/bridge/stormer_bridge.py

#### Files to modify
- earthdelta/heads.py
- earthdelta/selection.py

#### Files to create
- earthdelta/serving.py
- scripts/train_paired_heads.py
- scripts/run_serving.py
- tests/test_serving_isolation.py

#### Implementation requirements
- 同MetricSpec训练e0/du及可选gain辅助；纯解析score与校准score分榜。输出额外返回预测状态/候选cost记录。
- predict(history,legal_context,plans,metric)->pred e/du; select->plan; executeactualplan。不能在select内部调用teacher/evaluateallcandidates或readfuture。
- 选择step0后段features时计referencepreview与回放；默认共同完整preview以公平，不承诺零成本。
- 不通过tuple/tensor重建切断需要的梯度；trainingforward和inferenceplan分开；参数/数据flowtyped但不把typehints称sandbox。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- futurestore不可挂载/读取；poison标签不改变plan；候选置换等变；explicitnoeditgain0；冻结weightsSHA不变。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_serving_isolation.py -q
cd "$REPO" && python scripts/train_paired_heads.py --config "$CONFIG" --out "$RUN_DIR/train"
cd "$REPO" && python scripts/run_serving.py --config "$CONFIG" --model "$HEAD_CHECKPOINT" --out "$RUN_DIR/serve"
```

#### Expected artifact
- train/model_manifest.json
- serve/plans.parquet
- serve/predictions/
- serve/information_access.jsonl
- serve/cost_ledger.json
- decision.json

#### Success criteria
- 完整serving不接触真值，训练实际完成有权重；actual回放结果与plan一致；内容身份守卫生效。

#### Failure criteria
- 访问未来/候选oracle或计费漏preview → STOP_SERVING_CLAIM。

#### Decision after completion
PASS→P1-04；模型分数只来自actualrun。

#### Commit suggestion
`feat(serving): deploy analytic paired-response selection`


### TASK P1-03 — 训练最强同条件基线并做预算前沿

**Prerequisites**：P1-01。

#### Scientific purpose
排除条件网络、常规适配、一般辅助监督和额外计算的解释。

#### Files to inspect
- earthdelta/heads.py
- earthdelta/lowrank.py
- earthdelta/baselines/
- reference/_manifest.json

#### Files to modify
无。

#### Files to create
- scripts/train_baselines.py
- configs/baselines.json
- tests/test_baseline_fairness.py

#### Implementation requirements
- 至少F0、Fs、beststatic、largerstatic、samecontexttemporalrouter+multistep、directgain、feedbackoutput、virtualoutput。random和oracle免训练但角色独立。
- 可参考DISeL/WeatherPEFT/CoMoL等论文；许可未核实不vendor，改写标adapted且不得继承其官方结论。
- 各组相同data/targets/earlystop/HPOtrial上限；报告总参数/训练和教师生成成本，不仅rank。
- L2损失/summary/全场目标一致；closedloop基线按相同hold后reference规则，输出反馈不削弱。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 每组schema自动检查共享信息和预算；seed/run/config缺失不能入主表；virtual和actual标签不混。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_baseline_fairness.py -q
cd "$REPO" && python scripts/train_baselines.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- baseline_manifests.json
- fairness_matrix.csv
- trained_models/index.json
- budget_frontier.csv
- decision.json

#### Success criteria
- 全部必要强基线真的完成或明确BLOCKED；非matching部分解释，未run不填0。

#### Failure criteria
- 必要directgain/feedback未跑则主novelty比较BLOCKED。

#### Decision after completion
PASS→与P1-02共同进入P1-04。

#### Commit suggestion
`feat(baselines): enforce same-information edit and correction comparisons`


### TASK P1-04 — 有限组合交互与局部近似核查

**Prerequisites**：P1-02, P1-03。

#### Scientific purpose
区分全程非线性交互和由向量内积自然产生的cross terms。

#### Files to inspect
- earthdelta/probe.py
- earthdelta/geometry.py
- earthdelta/selection.py
- earthdelta/heads.py

#### Files to modify
无。

#### Files to create
- scripts/evaluate_edit_composition.py
- tests/test_composition_residual.py

#### Implementation requirements
- 计算I(a,b)=du(a+b)-du(a)-du(b)，所有响应与a=0同参考；pair实际跑而非仅Σdu代替。
- 比较finitewholeplan预测、linearR、single-response-sum及full/diagH；同等候选和budget。
- 只在已登记小幅度域使用R；epsilon及mixed校验、fp32/64诊断；零bank不做“rank发现”。
- 完整矩阵不优于对角则删除“交互贡献”，不为了保住结论改pair筛选。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 线性toy I=0而quadraticcross可能非零；非线性toy I非零；分母接近0标清。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_composition_residual.py -q
cd "$REPO" && python scripts/evaluate_edit_composition.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- composition_errors.parquet
- finite_vs_linear_scorecard.json
- full_diag_comparison.json
- decision.json

#### Success criteria
- 近似有效域与失效情况可复算；必须有actual组合而非只有代数证明。

#### Failure criteria
- 混合近似失效则改finite-only，不因此杀所有finite方向；full≈diag则降级交互声明。

#### Decision after completion
完成明确范围后→P1-05。

#### Commit suggestion
`feat(eval): separate exact edit composition from quadratic overlap`


### TASK P1-05 — 独立确认、成本与论文证据发布

**Prerequisites**：P1-04。

#### Scientific purpose
在未参与选择的数据上建立可投稿证据，而不是不断迭代开发分数。

#### Files to inspect
- earthdelta/evaluation/
- reference/weatherbenchX/
- artifacts/p1/

#### Files to modify
无。

#### Files to create
- scripts/export_wbx.py
- scripts/evaluate_confirm.py
- scripts/build_paper_artifacts.py
- tests/test_evaluation_contract.py

#### Implementation requirements
- 锁方法/weights/metric/candidate/splits/HPO后一次确认；多lead6/24/72/120，240为预登记extension。
- WeatherBench-X作为评分候选，优先用已锁本地版本；独立小数组手算验证聚合，不把官方库自动当科学正确保证。
- aggregateMSE→sqrt，ACC用训练climatology；变量/过程分组CI；配对方法共同validmask；主榜failure不静默排除。
- GPUwarmup、synchronize、batch一致、多次测量median/p95；初始化/preview/head/planner/editedrollout单列，训练teacher成本另列。
- 输出oracle分布、predresponse真实性、static/dynamic/oracle、directgain/outputchallenge、longrollout与efficiency，并附全部负结果。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- WBX与手算一致；候选失败分母保留；confirmationhash变化拒绝；原论文表缺真实记录则禁止生成。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_evaluation_contract.py -q
cd "$REPO" && python scripts/evaluate_confirm.py --config "$CONFIG" --out "$RUN_DIR"
cd "$REPO" && python scripts/build_paper_artifacts.py --run "$RUN_DIR" --out "$RUN_DIR/paper"
```

#### Expected artifact
- confirmation_manifest.json
- metrics.json
- bootstrap_intervals.json
- paper/figures/
- paper/tables/
- cost_profile.json
- decision.json

#### Success criteria
- 核心假设在独立数据证据支持，范围与声明一致；所有主表可从冻结记录重建。

#### Failure criteria
- 确认失败保留；不能重命名test为dev再申请同一结论；重大回退STOP或修改主张。

#### Decision after completion
PASS可选择P2之一；P2不是P1成功的事后补救。

#### Commit suggestion
`feat(eval): publish frozen confirmation and full cost evidence`


### TASK P2-01 — 未见修正组合与时窗的轨迹响应泛化

**Prerequisites**：P1-05。

#### Scientific purpose
唯一主要novelty增强：把候选向量预测变成对未见有限修正程序的可验证泛化，而非泛泛world model命名。

#### Files to inspect
- earthdelta/contracts.py
- earthdelta/heads.py
- earthdelta/bridge/stormer_bridge.py

#### Files to modify
- earthdelta/contracts.py
- earthdelta/heads.py
- earthdelta/bridge/stormer_bridge.py

#### Files to create
- scripts/p2_action_generalization.py
- tests/test_action_holdout.py

#### Implementation requirements
- 优先同bank未见amplitude/hold/window/组合；分别冻结actionholdout与atmospheretimeholdout。
- 利用已有ProgramSpec/leadhead，必要时加明确作用时窗描述符，不另起JEPA/新Jacobian系统。
- 同样给directgain与directrouter候选/预算/lead descriptors；不能说其按构造不能新预算。
- longlead外推与已有lead插值分开；新bank需weight/behavior descriptors及独立标定，明确不是当前zero-shot主张。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- train中无holdoutprogram及近重复；候选置换；同描述符不同内容bank拒绝旧cache。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_action_holdout.py -q
cd "$REPO" && python scripts/p2_action_generalization.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- action_split.json
- generalization_scorecard.json
- composition_rollouts/
- decision.json

#### Success criteria
- 在预登记未见程序上有actualtrajectory校验，相比同描述符基线的收益有CI支持。

#### Failure criteria
- 仅已见candidate有效→收窄lookupsurrogate定位；不称通用interventionworldmodel。

#### Decision after completion
保留有效的一条增强；失败不自动加memory。

#### Commit suggestion
`feat(p2): test held-out intervention trajectory generalization`


### TASK P2-02 — 可校准的选择性不编辑

**Prerequisites**：P1-05。

#### Scientific purpose
处理选择器从多个噪声预测中挑最大值的过度乐观风险，提供经验性可靠性而非无条件安全承诺。

#### Files to inspect
- earthdelta/selection.py
- earthdelta/heads.py

#### Files to modify
- earthdelta/selection.py

#### Files to create
- earthdelta/calibration.py
- scripts/p2_selective_editing.py
- tests/test_calibration_isolation.py

#### Implementation requirements
- 只有P1观察到gain校准/误修正问题时启用；单独校准时间块、候选数量/预算conditioning，threshold不从test定。
- mean-uncertainty penalty只是选择规则；经验coverage/riskcurves、under-shift失效一起报告，不照搬VI-MoLE证书。
- 两个upperrisk差不是普遍gainlowerbound；采用直接gain误差校准或有效的jointassumption证明，并审查所需exchangeability。
- noedit也计controller成本；回退不是零成本。不得调高拒绝率后只报被编辑子集精度。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- calib/test隔离，testtruth变动不改变threshold；allopportunityrisk与conditionalrisk分开。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_calibration_isolation.py -q
cd "$REPO" && python scripts/p2_selective_editing.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- calibration_manifest.json
- risk_coverage.csv
- harmful_edit_rate.json
- decision.json

#### Success criteria
- holdout经验误修正/覆盖改善且净收益在预登记范围内；条件与效度范围明确。

#### Failure criteria
- 跨shift校准无效→不称safe；coverage低不能隐藏totalutility差。

#### Decision after completion
不与P2-01同时盲目启动；解决已观测瓶颈。

#### Commit suggestion
`feat(p2): calibrate selective editing on held-out blocks`


### TASK P3-01 — 登记延期项目但不实现

**Prerequisites**：P1-05。

#### Scientific purpose
防止模块堆叠与无关工程扩张。

#### Files to inspect
- earthdelta/memory.py
- earthdelta/spectral.py
- earthdelta/probe.py

#### Files to modify
无。

#### Files to create
无。

#### Implementation requirements
- DEFER：JEPA/gated memory、dynamic rank、完整SHT正则、第二backbone、JVP、RL和大规模10天极端专项。
- 任何重启需新增明确bottleneck、同预算baseline、许可与资源预算，另行研究决策。
- memory重启前修窄异常/zip长度/decay；spectral重启前验证真实SHT节点与线性操作，不把logenergy直接放线性gain。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 本任务不执行模型测试。

#### Command
```bash
# DEFERRED: no execution command
```

#### Expected artifact
- DEFERRED_BACKLOG.md

#### Success criteria
- 延期状态保留，未把占位模块列为论文贡献。

#### Failure criteria
- 绕过P0/P1扩展即SCOPE_VIOLATION。

#### Decision after completion
保持DEFERRED，需独立授权与新计划。

#### Commit suggestion
`docs: record evidence-gated future work`

## 8. P0 — Project Survival Tests
P0-01至P0-08是唯一当前执行主线。最早科学判定是P0-05；其前不是冗长“合规工程”，而是避免错误参考/零字典/非法候选直接污染结论的必要门。P0-06/07为有界pilot，不等于大规模paper训练。

## 9. P1 — Core Paper Implementation
只有P0-08正式放行才执行P1-01至05。优先完整paired finite候选，不同时另建连续QP、JEPA、memory等多条方法主线。

## 10. P2 — Novelty Enhancement
最多两条：主要为未见程序的多时效响应泛化；次要为有独立校准集的选择性不编辑。需要P1证据支持，不作为负结果后无限补模块。

## 11. P3 — Optional / Future Work
见P3-01。所有模块保持DEFER，不自动执行。

## 12. Baseline Implementation Plan
必需baseline与公平性见 experiments/BASELINES.md。F0、Fs、random、beststatic、regime、oracle、true/pred4格、directgain、same-historyrouter、反馈/末端/virtualoutput均单列。统计分母与信息权限相同；oracle分数不列部署主榜。

## 13. Evaluation Contract
详见 implementation/EVALUATION_CONTRACT.md。gain是同Q的平方损失差，不是RMSE差。按注册总体先聚合MSE再开方；ACCclimatology固定。含有效数、失败数、被编辑率、伤害率、regret、response方向/幅度/R²、分变量/时效和端到端成本。任何比率分母接近0记N/A，不用epsilon掩盖不可解释比率。

## 14. Artifact Contract
每次 artifacts/<phase>/<run-id>/ 必有 config.json、manifest.json、metrics.json、summary.md、stdout.log、decision.json。实际prediction有 predictions/索引；figure只从实际结果生成。oracle与serving分开路径与role。单次报告包含完整command、code/checkpoint/bank/data/metric/split身份。

## 15. Reproducibility Contract
固定seed和随机状态，保留算法确定性范围、dtype/backend、原生网格和变量顺序；新旧分数分目录，不覆写失败。新脚本仅优化requires_grad=True参数。checkpoint和teacher必须来自已核查来源。正式结果绑定commit，包括未提交diffhash但不公开密钥。详见 implementation/REPRODUCIBILITY.md。

## 16. Literature-to-Code Mapping
见 implementation/LITERATURE_TO_CODE.md 和 research/RELATED_WORK.md。仓库46项manifest不是46项都已许可审定；标NONE的代码不直接vendor。主线只需现有EarthDelta、Stormer、SciPy/PyTorch与评估库，避免为论文名增加依赖。

## 17. Expected Paper Artifacts
Figure1 完整oracle/static gap分布；Figure2 predicted/actual响应与4格分解；Figure3 static/router/paired/oracle共同预算；Figure4 paired/directgain/feedback/virtualoutput；Figure5 多lead退化/改善与实际成本；Table1主forecast、Table2关键机制、Table3端到端资源。没有真实结果时输出NOT_RUN占位metadata，不画模拟科学曲线。

## 18. STOP Conditions
严格使用 STOP_CONDITIONS.md。old2–3%阈值不是本合同默认。未完成pilot与SESOI理由前不得宣称directionPASS或STOP。阶段处理器不得以缺文件静默跳过，也不得自动开启新候选来规避STOP。

## 19. 自含规范附录

以下为执行所必需的规范副本；即使不打开其他文件，也不得绕过这些门。详细研究报告不在此重复。


### STOP / PIVOT / BLOCKED 合同

#### 先分清四种情况
- **BLOCKED**：资产、许可、环境、数据角色、效应阈值或预算未确认。缺 GPU 不是科学失败。
- **FAIL_IMPLEMENTATION**：错误符号、指标、无效候选、独立S0不等价、泄漏等。先修复，已有数字不可作为正式结论。
- **INCONCLUSIVE**：误差区间跨过最小有用差异、独立过程不足或功效不足。在预登记cap内补样；cap到达即停止，不擅自改阈值。
- **STOP / PIVOT**：一个明确且合格的设定在充分精度下被否证。停止的是当前字典/当前分解/参数编辑主张，不声称所有可能方法都无价值。

#### 阈值不能照抄2–3%
主损失先定 L（非混合变量原始单位的随意均值）。报告配对变化 Δ=L_base−L_method；相对量仅当基线分母稳定且非零时使用。主参考为 Fs，另比较 best-static 的增量 Δ_dynamic。

P0-04建立 threshold_certificate.json，必须含：
1. 固定主variable/lead与全局辅助指标；建议先24h主、72h guard，是否采用由实际数据/成本确定。
2. 数值重复运行/实现parity产生的误差范围 δ_numeric，不用增大tol掩盖模型差异。
3. 最小有用效应 δ_practical：解释在同成本下为何值得，或通过静态模型的成本–技能曲线估计达到同效应所需成本。没有外部业务标尺则写“研究性SESOI”，不能冒称业务阈值。
4. 最终 δ_min 至少高于数值地板；δ_dynamic另定。**不得把标准误本身当业务效应阈值**。不能由独立确认结果选δ。
5. Pilot过程级paired variance、独立时间块定义、预期功效和最大样本/计算cap。可预设alpha=.05、power=.8作为分析政策并说明；用pilot方差规划样本，最终使用配对块bootstrap及敏感性分析。
6. Per-variable/long-lead non-inferiority margins：用相同校准逻辑，避免全局均值掩盖某关键变量恶化。

`configs/survival_template.json` 的效应和资源字段故意为 null：没有本地pilot，不能制造精确阈值。Codex须通过P0-04的实际测量和研究理由填写；未填写时所有科学放行均为BLOCKED_THRESHOLDS。

#### 判定函数
设 [lo,hi] 为预登记比较的过程级置信区间，δ为事前SESOI：

    if resource_missing or threshold_unset: BLOCKED
    elif invalid_contract_or_incomplete_comparison: FAIL_IMPLEMENTATION
    elif lo > delta: PASS_EFFECT
    elif hi < delta and planned_precision_met: STOP_CURRENT_CLAIM
    else: INCONCLUSIVE_WITHIN_CAP

这不是把 p>.05 当无效。若确实显著恶化（hi<0），可先暂停部署；无足够精度时不宣称等价。

#### 各门的科学停止条件
| ID | 条件 | 行动 |
|---|---|---|
| STOP-0 | 独立S0/metric/合法候选/时间合同未通过 | STOP_FORMAL_EXPERIMENT；修复不计科学成果 |
| STOP-1 | 合格非零字典、全部注册候选、确认精度足够时 oracle-Fs 的 upperCI<δ_gain | STOP_CURRENT_DICTIONARY；不训练复杂响应模型 |
| STOP-2 | oracle-beststatic 的 upperCI<δ_dynamic | PIVOT_STATIC_ADAPTATION；不要用接近0的“解释百分比”阈值 |
| STOP-3 | predicted du不优于zero/mean且4格定位响应为瓶颈，达到预登记fitcap | STOP_CURRENT_RESPONSE_REPRESENTATION；仅准一项事先界定的尺度/描述符诊断 |
| STOP-4 | true du也救不了pred e0；可部署收益不足 | PIVOT_ERROR_ESTIMATION或缩小可预测目标；不继续增加编辑器 |
| STOP-5 | 同teacher/data/budget directgain与双头无实质区别 | 删除分解独立novelty声明；只有预登记的label/action迁移成立才保留相关价值 |
| STOP-6 | feedback或virtual输出纠错在效应与成本上支配参数编辑且CI支持 | PIVOT_OUTPUT_TRAJECTORY_REPAIR，不再宣称参数必需 |
| STOP-7 | 长时效关键变量的退化超过margin，反复独立过程出现 | STOP_LONG_HORIZON_CLAIM；修训练分布后用新确认数据，不覆盖旧结果 |
| STOP-8 | 实际成本含preview/运行后无收益或超过已定资源cap | 停止“高效”主张，保留机制诊断范围 |

STOP后输出：失败证据、已执行量、未执行量、当前待办锁定状态。不得自动重启第二backbone、JEPA、memory、谱损失或扩大数据。


### 统一评测合同

#### 数据/输出
Serving输入I须包含可核验history timestamps、Fref身份、可选referencepreview及其费用；truth不可见。真实输出张量保留完整[init,lead,var,lat,lon]，summary为[B,H,S,F]。每个candidate都有明确bank、系数、hold、层、continuation。prefix一旦含学生预测，不能复用真值prefix上的response标签。

#### MetricSpec
令s_v为训练期固定变量scale，q包含区域cellarea、variableweight、leadweight；Q在全目标轴一次归一化。D可先进行固定线性降采样/投影；scale只应用一次。对无缺失情况：

    L = Σ_h,s,f q_hsf ((Y-F)/s_f)^2 / Σ_h,s,f q_hsf
    gain = L_reference - L_edited.

若已在D中除s，则Q中不再重复除s²。perlead统计使用该lead的声明条件归一化；全局统计不能再无权mean各lead代替。weight/scale/lead列表/投影均哈希入模型和运行manifest。缺测权重与共同有效mask在看到某方法表现前冻结；不按方法丢点造成不同分母。

RMSE：先在共同样本/空间聚合MSE，再sqrt。mean(per-init RMSE)是不同统计量，若保留必须明确命名。ACC用训练期climatology，含变量/lead维度。确定性单样本CRPS退化为MAE，不称概率校准。

#### Intervention metrics
- actual_gain：真正执行后的完整Q损失差；summary_gain另列，不混成fullfield。
- oracle_gain：完整已注册且可行有限集合上的逐样本最优，无编辑保证非负。任何候选缺失则complete_oracle=false。
- static_gain：只在fit/dev选择一个固定候选，再应用未见样本。
- dynamic_gap：hindsightoracle−beststatic；不是所有参数空间的上界。
- regret：同真实结果、同可行集下best−selected；缺失/不可行状态不可静默替换。
- ranking：Spearman与top-k best-action recall；平分和所有响应零时N/A并记录。
- harmful_edit_rate：全部已执行nonzero edit中 actual_gain<0 的比例，同时报告占全部注册起报的比例，不能只看conditional。
- no-edit率、失败率、未结算率全部保留；当oracle接近0时不报告“已实现oracle百分比”。

#### Response metrics
weighted NMSE/relative error、cosine、normbias、R²，在每个lead/变量/动作族单列。nullresponse作为单独组：du=0时cos未定义，不自动当perfect或0。no-effect、fit平均响应和descriptor-shuffle必测。全空间误差与响应span中e0误差分开，span只用于离线诊断，不假装推理可获得真实R。

#### 同一表的4格
OO：truee0+actualdu；OP：truee0+preddu；PO：prede0+actualdu；PP：prede0+preddu。四者选择后都在同一realized table结算。OO是离线参照；PO实际跑了所有candidate，不能列低成本部署。PP与directgain同算力/输入比较。选择器保持不变才可解释瓶颈。

#### Physical structure
全变量评分之外可报告风场涡度/散度、位置/强度和合法SHT系数诊断，但需要真实网格与单位。谱功率、动能、logenergy是非线性函数：可分别变换各端点后做平方差，但不能将T(Yedit−Yref)当T(Yedit)−T(Yref)。不以物理图好看证明参数编辑更物理。

#### 成本
分别记录初始化、referencepreview、historyencoder、allcandidatehead、planner、选择后的真正rollout、读取/导出、训练/teacher生成成本。对同等batch/dtype/backend暖机后，CUDA synchronize计时，多重复报告median/p95。稀疏slotcount为抽象预算；elapsed/GPUmemory为实际成本，两者不混称。

默认不把所有K个候选的真实天气运行隐藏在“label-free du”里。作为response数据采集可以；作为deployment必须明确付费compare。

#### 统计
按起报时间块或独立天气过程配对bootstrap，同一个过程不同lead/网格不是独立N。预先冻结主指标、SESOI、guardmargins和最大确认样本；多重比较标明主/次目标，研究性分层不等于独立确认。所有置信区间同时报告样本组数和有效覆盖。


### Baseline Implementation Contract

| ID | 实现复用与wrapper | 输入/输出 | 关键公平限制 |
|---|---|---|---|
| F0 | pinned Stormer +独立official对照 | 原生当前场→原始forecast | 不减变量；不把多intervalensemble只给某方法 |
| Fs | 现有ExpertLoRA或许可明确静态LoRA，训练后冻结 | 同当前/历史权限→forecast | 与intervention相同训练资料；F0/Fs明确分开 |
| Random | 固定seed从同可行registry抽样 | 同candidate→actualforecast | 含noedit，预算分布匹配；不挑最佳seed |
| BestStatic | dev集固定最优candidate | 所有确认issue用同edit | 不能按test选择；regimeonly先在fit定分组规则 |
| HighCapacityStatic | 调整静态rank使总trainable量匹配 | 与主方法相同信息 | 同rank与同总参数两个视角都报；不继承成本相等 |
| DirectRouter | 保留相同encoder/bank，直接boundedcoeffs +multisteploss | history/preview/budget→edit | 给予候选/预算/lead描述符，允许合理泛化；不要稻草人 |
| DirectGain | 新baseline浅MLP/相同backbonehead | I,a,lead→scalar gain | 同candidate标签、同HPO、同split；目标同Q |
| DirectEditedForecast | 与双头相近总宽度 | I,a→D(Fa) | 与主方法同有限仿真训练输出，不多偷未来truth |
| PairedAnalytic | 现有ComposedPredictionHead补统一metric | ehat,duhat→analyticgain | calibration off；JEPA/memory off同权限 |
| PairedCalibrated | 同上＋单独校准项 | analytic＋calibration | 独立消融，不把它的胜利都归于identity |
| ResidualPost | 共同context，多变量多lead decoder | Fref+predresidual | 不仅one-lead标量；训练成本匹配 |
| ResidualFeedback | Solver-in-the-Loop式协议、PyTorch本地实现 | 每步共同input→correctedstate→Fref | 能影响后续动力学；与参数方法同hold/rolloutloss |
| VirtualEdit | 共享du预测 | Fref+duhat(selected) | 数值近似不是实际模型编辑；保留误差和额外Fref成本 |
| OracleEdit | 全部注册candidateactualrun | offline truth→bestcandidate | 仅当前有限集上限；不列serving榜 |
| OracleE / OracleDu | 同一paired表4格替换 | 精确对应量 | actualdu花了K条forecast，不能藏成本 |
| Finite vs R/QP | 现有probe/geometry/teacher/selection | fixed finiteeffects vs localJacobian | 近似有效域、预算与候选必须一致 |

#### 外部实现
Stormer/DISeL/WeatherBench-X优先复用已核实版本与许可；GEPS/WeatherPEFT/CoMoL/W2T当前清单有NONE或未确认，不直接复制。理论上的VI-MoLE-style marginalgain baseline应标明independentreimplementation，不冒称原作完整认证复现。CLAW/CCM作者代码本轮未核实，因此不作为强制安装依赖。

#### 标签效率
每个起报的e0只算一份verification label，不因K编辑重复计K份。du是仿真label，不需要未来truth但有计算成本。可以另采无未来truth的训练issue响应，所有方法给予相同仿真输出训练资格；设计same-verification-label与same-total-compute两张表。只给双头额外仿真而不给directedited/gain可利用数据，不能直接归因factorization。

#### 候选描述符
当前activebit+coefficient+hold只标当前bank索引。新bank即使K一样也不是同一action；需要内容绑定与行为/权重descriptor。P2优先同bank未见幅度/窗口/组合，不做未经标定的跨模型zero-shot。


### Literature-to-Code Mapping / provenance

本表刻意区分“本轮已读取源码/历史已读取”与“仅仓库manifest报告”。没有独立核实完整SHA或许可就写UNVERIFIED，不编造完整hash。以下短SHA可由Codex在已有本地clone用git rev-parse解析，之后把完整结果写external_lock.json。**不得先重新下载46个库。**

| External repo | Pinned commit / provenance | License状态 | Reused idea/code | EarthDelta位置 | Modification |
|---|---|---|---|---|---|
| tung-nd/stormer | 58dfee5a6037399a40fefd492bc00421e0c885a8；官方inference已重读 | MIT，manifest报告/既有会话检查 | 原生forecast与零diffmean还原 | bridge/stormer_bridge.py；S0独立参考 | 保留原始语义，显式编辑与trace |
| alizindari/DISeL | b6e543a13bd79caca75655cbae9b616011fe2d71；会话已读gate/variant | Apache-2.0；会话读LICENSE | 输入相关lowrank gate baseline | baselines/direct_router | 不依赖不稳定PEFT内部variant，移植标adapted |
| google-research/weatherbenchX | manifest短SHA964a35e；完整SHA UNVERIFIED | Apache-2.0，manifest报告/官方README已读 | 标准weightedmetric/aggregation | evaluation；export_wbx.py | 独立手算对照；数据合同仍自行核查 |
| scipy/scipy | 当前用户环境version与hash待P0-01记录 | BSD-3-Clause（安装许可须核对） | lsq_linear / minimize | teacher.py / selection.py | 优先复用现有，不写新QPsolver |
| pytorch/pytorch | 当前用户环境version待P0-01 | BSD-style（具体版本核对） | autograd、tensor，JVP后续 | bridge/heads/lowrank | 无全网络no_grad；当前JVP不启动 |
| tum-pbs/Solver-in-the-Loop | manifest短SHAF514fcf；完整SHA UNVERIFIED | MIT，manifest报告；旧官方README已读 | feedback纠错与多步训练协议 | baselines/output_correction.py | PyTorch独立实现，不混TF1环境 |
| ShileiCao/WeatherPEFT | b2cdc25cbbb78de272a6f6e0d532c2fe2006a15c；会话源码已读 | **UNVERIFIED / manifest NONE** | 领域PEFT对照，参数选择概念 | 可选adapted baseline | 许可明确前不得vendor，不能冒称完整复现 |
| DCDmllm/CoMoL | manifest短SHA011306a；完整SHA UNVERIFIED | **UNVERIFIED / NONE** | shared factors/core mixture | lowrank.py未来对照 | P0不换bank结构；未获许可不复制 |
| itsakk/geps | e9a865218ecffacb7007ac7d719f3741afcf8c02；会话已读layers/kolmo | **UNVERIFIED / NONE** | 受控PDE/context低秩近邻 | 独立toy诊断 | 不按默认1024网格运行；许可先查 |
| xiaolonghan2000/Weight2Token | manifest短SHA4979345；完整SHA UNVERIFIED | **UNVERIFIED / NONE** | adapter内容与规范表示 | descriptor未来研究 | 首轮同bank组合holdout；不复制weightsencoder |
| VI-MoLE / CLAW / CCM | 论文L01/L06/L05 | 官方code未核实 | counterfactualrisk/directcontext/weightdirections | 强baseline设计 | 数学独立实现须标记非官方、无原文安全保证 |

#### 使用步骤
P0-01查已有clone及许可证文件；P0-02/03仅复用已明确许可部分；写入sourcefile路径、原版权、修改摘要和diffhash。代码许可不自动覆盖模型checkpoint或数据；拒绝把ROOT LICENSE=null解读为“随便复制”。

主线启动的必要工具已经在现有repo内：ExpertLoRA、paired、cached_responses、SciPyteacher与heads。文献越多不等于依赖越多。
