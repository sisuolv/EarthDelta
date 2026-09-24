# 后续执行计划：只恢复原计划的关键依赖链

状态：所有任务结果均为 **TO_BE_RUN**。这是待执行计划，不是新的实验报告。

审查实现 `a3596e6b804e9d23b72d1247b08c47129d6b50b1`；实际分支头 `fdd92d79b1b03e8397d4c487d7fa4a347ae6fc37`；原比较基线 `d749a1c62521226df857587e08f7d067b0f15355`。
本轮不是重排全项目P0/P1/P2，也不以新增shared-F0分支替代原fitted-Fs合同。

## 必要输入与命令规则

设 `PKG` 为解压目录、`REPO` 为现有工作区、`RUN` 为全新run目录、`ASSETS` 为真实checkpoint/NPZ/Zarr根目录，`OVERLAY_RUN` 为已批准的CUDA依赖overlay。
所有变量在本机确定；不要照抄报告中的/mnt/afs路径当成必然存在。

`plan_followup_runner.py`、`plan_dev_evaluate.py` 及标注NEW的测试是本轮 **需要创建的薄入口**，尚不存在，不得现在把它们的命令写成已执行。
复用 `static_adapter.py` / `bank_training.py` / existing S0 / native Q / registry / checkpoint primitives；不要重写模型或引入新backbone。

现有 `r2_fs_bank_train.py --stage all` 尚有前驱失败继续调用的缺陷，修复并通过CLI反例前禁止用于真实训练。
S0现有CLI可用于诊断，但必须先准备冻结config、实际源码环境并使用新的输出目录。

## 时间和资源

每个任务的 `estimated_minutes` 暂为null：缺少当前工程进度、GPU/IO吞吐和排队信息。先测实值，不虚构完成时间。
原计划J0 60分钟、J1 75分钟、J2最多约4小时为管理cap；实际新任务须在用户批准的预算内登记，不能据此保证时间。
训练16–32 updates是诊断规模，500是起始上限；没有理由要求跑满500才检查质量。

## 任务

## FP-00 — 只读核验最新分支/工作区、原计划、原始日志与实际资产；不训练、不提交GPU作业。

**依赖：** 无  
**状态：** TO_BE_RUN；READY  
**计算：** CPU only  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `evidence/REQUIRED_INPUTS.md`
- `tools/collect_entry_evidence.py`
- `plans/plans_v3_0922/`
- `codex_audit_round4/evidence/`

### 实施步骤
1. 记录实际HEAD、代码树、未提交diff及module文件hash；保留用户改动。
2. 检查pt-7ant09uv原始证书/张量和两个formal Fs作业资料；缺失项逐项MISSING，禁止据摘要升级。
3. 冻结新run目录、asset map、曝光台账与原fitted-Fs协议；只读收集不产生任何parity PASS。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python "$PKG/tools/validate_delivery.py" --package "$PKG"
python "$PKG/tools/collect_entry_evidence.py" --repo "$REPO" --out "$RUN/entry"
```

### 所需证据／输出
- `entry/entry_evidence.json`
- `entry/worktree.diff`
- `entry/source_files.json`
- `required_inputs_resolved.json`
- `authorization.json`

### PASS
- source与实现差异清单可复核；没有覆盖原证据。
- 列全S0/Fs/data缺项，清楚区分可执行CPU工作与缺资产的GPU工作。
- FP-01所需实际路径与S0-only预算均明确，方可请求其解锁。

### FAIL / BLOCKED
- 代码或资产身份未确定；禁止修改证书填PASS。
- 任何要求提供账号密钥/修改用户无关作业的步骤。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。

## FP-01 — 定位并修复官方/bridge多步差异；冻结1e-5不变，1/4/12步独立官方S0全部通过才闭合。

**依赖：** FP-00  
**状态：** TO_BE_RUN；BLOCKED_BY_FP00  
**计算：** CPU regression + real GPU S0  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `earthdelta/bridge/stormer_bridge.py`
- `scripts/r4_s0_parity_diagnose.py`
- `scripts/export_upstream_reference.py`
- `scripts/s0_gate.py`
- `scripts/r4_s0_rerun_existing_env.py`
- `tests/test_normalization_policy.py`
- `tests/test_s0_gate_end_to_end.py`
- `tests/test_input_inverse_source_precision.py (NEW)`

### 实施步骤
1. 检查执行机真实NPZ dtype与hash。比较input inverse在源dtype先求倒数再cast、先float再求倒数；必要时保留raw input常量/预构造buffers，更新算术身份。
2. 修复diagnostic后处理模型条件遗漏/10的问题；实现official-model×official-transform、official×bridge、bridge×official、bridge×bridge四臂逐step trace。
3. 共同输入局部算子比较与自由rollout分开；记录inverse/constants/pad/constant mask/model/raw add/renormalize差异；不要用双边共错修补。
4. 以实际导入源码hash+config+raw/normalized输入+各时效输出绑定独立export/gate；修改后建立新证据目录，不复用旧PASS。
5. 独立真实export和gate输出全部注册步数<=1e-5；缺CUDA/xformers/权重/NPZ则BLOCKED，CPU synthetic只能证明测试逻辑。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python -m pytest -q tests/test_normalization_policy.py tests/test_s0_gate_identity.py tests/test_s0_fail_closed.py tests/test_s0_gate_end_to_end.py tests/test_input_inverse_source_precision.py
python scripts/r4_s0_parity_diagnose.py --run "$RUN" --assets "$ASSETS" --out "$RUN/s0/four_arm_trace.json"
python scripts/r4_s0_rerun_existing_env.py --run "$RUN" --assets "$ASSETS" --overlay-run "$OVERLAY_RUN"
```

### 所需证据／输出
- `s0/gate_config.json`
- `s0/four_arm_trace.json`
- `s0/official_manifest.json`
- `s0/official_output_6h_1step.pt`
- `s0/official_output_6h_4step.pt`
- `s0/official_output_6h_12step.pt`
- `s0/s0_gate_result.json`
- `s0/job_result.json`
- `s0/stdout.log`
- `s0/source_manifest.json`

### PASS
- 四臂/逐算子数据解释首个差异；若未解释则PLAUSIBLE_UNVERIFIED而不是写root cause fixed。
- 相同冻结ps4、原生输入、FP32、official_zero_diff_mean、全部1/4/12步max_abs_diff<=1e-5。
- gate结果committed PASS、job exit成功、身份/hash/环境完整；真实原始输出可独立复算。

### FAIL / BLOCKED
- 任一注册时效超1e-5或输入/源码/资产绑定失败。
- 不得放宽容差、减少时效/变量/空间、用SDPA冒充official xformers。
- 任何GPU资产/数值根因证据缺失；到达已批准cap。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。

## FP-02 — 只修复原Fs计划的真实入口消费链：STOP、admission、Q、registry、profile不可变性。

**依赖：** FP-01  
**状态：** TO_BE_RUN；LOCKED  
**计算：** CPU; no expert training  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `scripts/r2_fs_bank_train.py`
- `scripts/r2_admission_gate.py`
- `earthdelta/static_adapter.py`
- `earthdelta/pilot_contract.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/registry.py`
- `earthdelta/bank_training.py`
- `scripts/plan_followup_runner.py (NEW THIN WRAPPER)`
- `tests/test_plan_gate_chain.py (NEW)`
- `tests/test_loaded_admission_binding.py (NEW)`
- `tests/test_profile_immutability.py (NEW)`

### 实施步骤
1. stage-all任一步False/exception则后继BLOCKED且未调用；单阶段也必须验证前驱证书，不能绕过。
2. 消费者验证外层PASS、全部行内容hash、实际UTC t-12/t-6/t以及+6/+24/+72、norm/source/config和角色；不得以issue_index替代实际时间身份。
3. 同一Q用于训练loss/端点gain/选择评分，端点先double再差分；legacy仅诊断，校准项默认关闭且单独输出。
4. profile使用临时clone或完整恢复，参数/buffers/模式/梯度状态前后一致，绝不改变已冻结产物。
5. 正式registry非空唯一明确no-edit，必须绑定实际Fs/bank/normalization/protocol；构造测试直接调用真正消费入口。
6. 创建薄runner复用现有函数与序列化，不重写气象模型；其CLI按本任务文档新增后才可调用。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python -m pytest -q tests/test_metric_contract.py tests/test_content_verification_b08.py tests/test_pilot_admission_b09.py tests/test_candidate_registry_b15.py tests/test_plan_gate_chain.py tests/test_loaded_admission_binding.py tests/test_profile_immutability.py
python scripts/plan_followup_runner.py verify-contracts --run "$RUN"
```

### 所需证据／输出
- `contracts/contract_gate.json`
- `contracts/pytest.log`
- `contracts/junit.xml`
- `contracts/source_manifest.json`
- `contracts/admission_manifest.json`
- `contracts/rejected_cases.json`
- `contracts/profile_immutability.json`

### PASS
- 注入前驱FAIL/MISSING、错norm、错时次、缺任一内容证书、6h准入供24h训练时，实际入口拒绝且trainer spy调用数0。
- native endpoint identity通过FP32源端点反例；invalid mask/scale拒绝。
- profile前后bank hash相同；critical source更新影响S0时返回FP-01重新验证。

### FAIL / BLOCKED
- 任何消费链可fail-open，或变更了S0认证的数值路径未复验。
- 冻结split/目标/候选/阈值不能从后续结果倒推修改。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。

## FP-03 — 先解决fitted-Fs退化，再训练/选定并冻结唯一共同Fs；不能直接启动动态expert。

**依赖：** FP-02  
**状态：** TO_BE_RUN；LOCKED  
**计算：** CPU regression + one bounded GPU Fs path  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `earthdelta/static_adapter.py`
- `scripts/r2_fs_bank_train.py`
- `scripts/plan_followup_runner.py (NEW)`
- `tests/test_fs_quality_gate.py (NEW)`
- `tests/test_fs_static_adapter.py`

### 实施步骤
1. 建立真实质量资格门，与finite/update count/merge equivalence分开。固定初始化、训练侧面板、候选checkpoint、允许retry/LR选择及预算；规则在观察候选质量前冻结。
2. 先16–32 update诊断；finite grad和loss仅必要条件。500保留为上限，非强制目标；不盲目重复已退化0.01/500设置。
3. 在共同初始/候选最终checkpoint上复评同一面板，并单列逐issue首次/末次访问loss；实际8/8大退化案例必须拒绝，不能只做格式化。
4. 依据训练侧预登记规则选一次Fs，保存adapter及合并backbone、optimizer/seed/data/source证据、nonfinite和失败尝试。
5. 独立进程重载并核验merged/always-on、no-edit/continuation从编辑状态继续Fs；所有后续worker只加载这份内容身份。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python -m pytest -q tests/test_fs_static_adapter.py tests/test_fs_quality_gate.py tests/test_plan_gate_chain.py
python scripts/plan_followup_runner.py fs-diagnose --run "$RUN"
python scripts/plan_followup_runner.py fs-fit-freeze --run "$RUN"
python scripts/plan_followup_runner.py verify-fs --run "$RUN"
```

### 所需证据／输出
- `fs/quality_rule.json`
- `fs/training_records.json`
- `fs/panel_initial_final.json`
- `fs/fs_adapter.pt`
- `fs/fs_merged_backbone.pt`
- `fs/qualification.json`
- `fs/independent_reload.json`
- `fs/reference_manifest.json`
- `fs/job_result.json`

### PASS
- 真实固定面板质量满足预登记标准；没有将数值merge PASS当质量PASS。
- 重载后Fs身份/行为一致；退化候选不能发布合格reference_manifest。
- 得到唯一Fs certificate，包含实际权重hash、训练侧选择轨迹和资格限制。

### FAIL / BLOCKED
- S0不再有效；缺训练侧质量阈值/预算；Fs仍明显退化；没有独立可重载产物。
- 禁止以F0冒充合格Fs、按dev赢家挑Fs、无限重试或漏报坏run。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。

## FP-04 — 在唯一合格Fs上训练K4/rank4，并建立持久化、独立qualification、组装与registry证书。

**依赖：** FP-03  
**状态：** TO_BE_RUN；LOCKED  
**计算：** GPU independent processes + CPU artifact validation  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `earthdelta/bank_training.py`
- `earthdelta/value_pilot.py (reuse persistence only with fitted-Fs schema)`
- `scripts/r2_fs_bank_train.py`
- `scripts/plan_followup_runner.py (NEW)`
- `earthdelta/registry.py`
- `tests/test_fitted_fs_bank_roundtrip.py (NEW)`

### 实施步骤
1. 所有worker从同一Fs artifact启动并记录加载后state hash；不允许worker本地再fit Fs。
2. K4/rank4 blocks18–23、singleton .25、hold4×6h固定；profile若要求K2只能在看dev前登记新身份。
3. 先短步验证，再按预算训练；每个expert有自己的训练来源、梯度、非零响应、资格与可重载checkpoint。
4. 保存并重载组装bank；同输入逐k比较源expert与assembled singleton 6/24/72h输出，zero-edit=Fs、窗外继续Fs。
5. 冻结registry绑定完整bank/Fs/norm/inputs/protocol/source与资格证书；不得仅保存JSON中的参数摘要。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python -m pytest -q tests/test_bank_training.py tests/test_bank_gradients.py tests/test_fitted_fs_bank_roundtrip.py tests/test_candidate_registry_b15.py
python scripts/plan_followup_runner.py bank-train --run "$RUN"
python scripts/plan_followup_runner.py bank-qualify-assemble --run "$RUN"
```

### 所需证据／输出
- `bank/expert_0.pt`
- `bank/expert_1.pt`
- `bank/expert_2.pt`
- `bank/expert_3.pt`
- `bank/qualification.json`
- `bank/bank.pt`
- `bank/assembly_equivalence.json`
- `bank/registry.json`
- `bank/bank_manifest.json`
- `bank/job_results.json`

### PASS
- 4个worker的reference hash唯一；4个expert全有合格且可重载证据。
- 组装前后行为等价与reference continuation通过冻结数值规则；不得套用S0以外容差给S0放行。
- 被消费registry精确绑定最终dynamic权重；profile不会改变它。

### FAIL / BLOCKED
- 任何reference不同/专家不合格/动态权重缺失/行为不等价/metadata不匹配。
- 不按dev收益删掉坏expert再称原K4合格。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。

## FP-05 — 按原计划构建五候选cache与廉价时间隔离OOF开发表；不训练dual-head，不打开confirm。

**依赖：** FP-04  
**状态：** TO_BE_RUN；LOCKED  
**计算：** GPU cached rollouts + CPU cheap heads/statistics  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `scripts/plan_followup_runner.py (NEW)`
- `earthdelta/value_pilot.py (reuse cache primitives, explicit fitted-Fs identity)`
- `earthdelta/metrics_contract.py`
- `scripts/plan_dev_evaluate.py (NEW)`
- `tests/test_plan_cache_evaluation.py (NEW)`

### 实施步骤
1. 8个exposed起报全候选debug；通过后按实际吞吐/可用独立过程/磁盘冻结32/64/128 dev规模及采样顺序，非依据收益。
2. 复用cache机制但明确reference=fitted-Fs；候选严格5=zero+4singleton .25，一条12步轨迹存6/24/72端点；保留全部失败分母与成本。
3. 跨时间purged OOF，以fold-train选择静态专家/拟合regime和ridge direct-gain；scaler与HPO均仅fold-train；每issue所有candidate同角色。
4. 合法policy读取t-12/t-6/t和已声明的便宜信息，不读验证truth/真实candidate response/oracle。预测/动作先保存hash，再由评分器结算。
5. oracle从完整实际结果表计算，明确hindsight非可部署；不把完整表预先泄漏到模型训练；confirm保持封闭。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python -m pytest -q tests/test_plan_cache_evaluation.py tests/test_metric_contract.py tests/test_plan_gate_chain.py
python scripts/plan_followup_runner.py cache-debug --run "$RUN"
python scripts/plan_followup_runner.py cache-dev --run "$RUN"
python scripts/plan_dev_evaluate.py --run "$RUN"
```

### 所需证据／输出
- `cache/debug_manifest.json`
- `cache/dev_protocol.json`
- `cache/cache_manifest.json`
- `cache/candidate_results.parquet`
- `policies/folds.json`
- `policies/oof_predictions.json`
- `policies/prediction_freeze.json`
- `evaluation/dev_results.csv`
- `evaluation/paired_block_bootstrap.json`
- `evaluation/costs.json`
- `evaluation/failures.json`

### PASS
- 事先冻结的完整issue×5候选矩阵可从hash绑定端点独立复算。
- OOF动作在标签揭示前提交，fold无历史/未来支持交叠；不把grid/candidate/lead当独立n。
- 24h主loss、gain Fs/static、72h guard、harm/noedit、成本及失败可复核；宽CI保留INCONCLUSIVE。

### FAIL / BLOCKED
- 任何前驱不合格；矩阵不完整或只有成功subset；发现oracle/validation信息进入fit。
- dev规模/专家/目标按结果改变；读取confirm。
- 不得把oracle PASS当合法policy或novelty PASS。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。

## FP-06 — 独立核验本迭代结果并决定STOP/CONTINUE/PIVOT/INCONCLUSIVE；不自动开启下一研究阶段。

**依赖：** FP-05  
**状态：** TO_BE_RUN；LOCKED  
**计算：** CPU only  
**预计分钟：** 待实际profile，不作数值承诺。

### 输入／文件
- `scripts/plan_dev_evaluate.py (NEW)`
- `scripts/plan_followup_runner.py (NEW)`
- `CURRENT_GAP_ANALYSIS.md`
- `evidence/PLAN_AUDIT_RESULT.json`

### 实施步骤
1. 从冻结矩阵重算各方法，核对issue/candidate/seed/时间块分母与全部资源。
2. 分别评估oracle headroom、oracle-static、legal-static、ridge/regime；阈值与MDE分开，不凭空写收益百分比。
3. 若Fs/bank资格不足，停止当前资产继续扩张但不判idea科学失败；若CI跨阈值到达cap则INCONCLUSIVE停止追加。
4. 发布机器decision与HANDOFF，明确代码、CPU、GPU、parity、实验、科学支持各层；original confirm/dualhead仍未解锁。

### 命令（NEW入口建成且gate解锁后执行）
```bash
python scripts/plan_dev_evaluate.py --run "$RUN" --verify-only
python scripts/plan_followup_runner.py decision --run "$RUN"
```

### 所需证据／输出
- `decision.json`
- `HANDOFF.md`
- `AUDIT_RESULT_AFTER_RUN.json`
- `evaluation/independent_recompute.json`
- `claims.json`

### PASS
- 每条外部声明能追到正确commit、job、artifact、metric。
- 停止/继续结论与置信区间、最低有用效应、资格和成本相符；无阈值回改。
- 保存新PLAN_AUDIT_RESULT和HANDOFF而不篡改本包历史审查。

### FAIL / BLOCKED
- 证据仍缺失/不可重算；效应不够或INCONCLUSIVE且预算到cap。
- 任何宣称新颖性成立/正式holdout完成却无对应实验。
- 已有合法输入下断言失败为FAIL；缺资源、文件、授权或依赖为BLOCKED。未来文件不存在记MISSING/TO_BE_RUN，不生成假PASS。

### 完成后
保存task_result.json、所有原始日志/hash；只为后继提供解锁资格，不能自动越过额外资源授权。
