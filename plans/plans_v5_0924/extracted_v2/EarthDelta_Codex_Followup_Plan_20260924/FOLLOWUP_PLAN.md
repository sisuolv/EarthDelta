# 后续执行计划：沿用 FP 编号，不重复已完成工作

验收源码：`e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`；观察HEAD：`08093650ba56e8cf709d75a61c26c50a0c35f6f7`。
**当前状态：PARTIALLY_EXECUTED。** S0数值阻塞已解除；Fs是预登记容忍内的弱参考，不是样本外有效静态模型；bank只有训练资格和单探针精确组装证据。
本文件的 NEW CLI 是交给 Codex 实现的接口；本次没有修改仓库、执行这些命令、启动GPU或训练模型。
所有未来任务 `estimated_minutes=null` / `MISSING_NEEDS_MEASURED_PROFILE`。历史766/861项测试耗时、bank profile仅是历史观测，不能当本轮完成承诺。

## 已完成部分的处理
FP00/01/03/04的DONE限定详见审查JSON。FP02部分完成不升级；FP05a/b只补本阶段消费所需残留。保留全部v1 STOP、X1 EXPLORATORY和DEV偏差记录。

## 协议调整（必须在DEV outcomes前）
1. 2020H2改为EXPOSED_NOT_CONFIRM；本轮confirm不指定、不读取。2019H2是retrospective DEV池，净化完整support与全项目曝光。
2. 0.0034只保留为Fs-relative工程screen；同一DEV的F0/ Fs/ policy配对盈亏独立报告。72h允许损害及其价值依据必须另行事前冻结。
3. 固定一次N：debug/profile与完整性只能决定事先列出的32/64/128之一，DEV结果不能决定追加。五候选不变，另有一个背景F0。
4. 双向blocked CV仅是回顾性转移；inner HPO和所有data-dependent transforms要在相应train内。
5. 两个科学GPU jobs + 一次infra retry，总3次；原计划上限4不解释为任意可用的第四次科学机会。预算变化需要新授权。

## 任务细节

## FP-05a — 补原始回执、重算现有证书、恢复完整曝光台账

**依赖：** FP-00, FP-01, FP-03, FP-04  
**状态：** READY / TO_BE_RUN  
**CPU/GPU：** CPU only; read-only existing files  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
只读维护已有认证的可信度；识别 FP02 残留和 X1 曝光。不运行训练或天气推理。

### 输入／文件
- `evidence/EXPECTED_INPUTS.json`
- `tools/read_only_preflight.py`
- `plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/s0-gate-20260923t174135587010z/s0_gate_result.json`
- `plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/certify/fs/`
- `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank/`
- `plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/admission/`
- `plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/results/`
- `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/protocol/bank_protocol_v1.json`

### Implementation
1. 读取实际HEAD及diff，保留未提交变更。核对17个core pin；新文件hash另建manifest。不要只认当前commit名。
2. 补齐job_result/invocation/source_manifest/stdout，按job_id/run_id和时间串联S0、Fs v2、bank。验算所有protocol/decision/certificate链，receipt缺失记MISSING而不是补造。
3. 从原始 Fs panel 和 bank verify 数字重算资格；要求完整0..3专家、1/4/12键以及8个hold后比较。对权重以只读路径重新hash，不因缺附件重训。
4. 建包含所有旧Fs、v1筛选、X1 N32/N64、v2资格、bank和debug的曝光台账。区分只看完整性与已训练/看效果。X1失败或未选中的模型也记入。
5. 标2020H2 EXPOSED_NOT_CONFIRM。2019H2只作为待验证DEV池；样本角色必须看完整支持窗口与24h buffer。记录原FP02尚缺哪些入口。

### Tests
- 对收据缺字段、错job映射、protocol hash改变、证书缺step键的合成输入，预检不能生成PASS。
- 核算X1 56个时间点及其连续支持覆盖；不得只列最后选中资产的训练数据。

### Run command（实现NEW入口并解锁后）
```bash
python "$PKG/tools/validate_package.py" --package "$PKG"
python "$PKG/tools/read_only_preflight.py" --repo "$REPO" --out "$RUN/entry/evidence_inventory.json"
python "$PKG/tools/recompute_report_arithmetic.py" --package "$PKG" --out "$RUN/entry/arithmetic.json"
```

### Required evidence / outputs
- `entry/evidence_inventory.json`
- `entry/job_receipts_index.json`
- `entry/certificate_recompute.json`
- `entry/core_source_pin_check.json`
- `entry/exposure_ledger.json`
- `entry/fp02_remaining.json`
- `entry/decision.json`

### PASS
- 可区分现有数值证书通过与平台completion原件未齐；所有新run前必需输入均定位/校验。
- 曝光账本包含X1 H2实际训练；2020H2不能再标untouched；2019DEV合法池经过审查。
- FP00/01/03/04不重训，FP02不虚标全DONE；只给FP05b解锁资格。

### FAIL / BLOCKED / STOP
- 真实资产hash或证书绑定不符；缺原始完成回执无法对齐；出现新source drift。
- 任何缺项必须BLOCKED，允许保存已完成只读清单；不得触发GPU。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-05a): 补原始回执、重算现有证书、恢复完整曝光台账`。

## FP-05b — 实现 split/消费链、CPU测试与结果揭示前协议

**依赖：** FP-05a  
**状态：** LOCKED / TO_BE_RUN  
**CPU/GPU：** CPU tests + bounded integrity I/O; no weather forward  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
以新文件补齐 FP02 的 FP05 消费边界，冻结更正后的协议；不改17个已认证核心文件。

### 输入／文件
- `earthdelta/split_freeze.py (NEW)`
- `earthdelta/candidate_cache.py (NEW)`
- `earthdelta/policy_oof.py (NEW)`
- `scripts/plan_followup_runner.py (NEW thin wrapper)`
- `scripts/r4_candidate_cache.py (NEW)`
- `scripts/r4_cache_decide.py (NEW)`
- `scripts/r4_policy_oof.py (NEW)`
- `tests/test_split_freeze.py (NEW)`
- `tests/test_plan_cache_evaluation.py (NEW)`
- `tests/test_fp05_consumption_chain.py (NEW)`
- `tests/test_policy_oof_isolation.py (NEW)`

### Implementation
1. 实现ledger+role capability，admission、worker launch、真实读取、resume/merge、scaler/PCA/HPO、fit、predict、score全部调用；不只在最外层写标签。
2. 先冻结选点/替换规则，再做完整性扫描。按同一已冻结7天区块就近替换，tie更早、禁止重复和跨块追好天气。扫描不运行模型、不选择效果；精确issue列表在正式forecast前冻结。
3. thin runner只串接现有认证加载器和新增cache/OOF；缺前驱/任一identity即拒绝，下游spy调用数0。冻结core17及新call graph源码。
4. 五候选固定Fs/no-edit+4个singleton .25；另存F0背景。native loss由同一Q/scale计算，端点先转FP64再差分；禁用旧带calibration的head路径。
5. 协议模板中的0.0034保留为Fs-normalized工程screen；另冻72h允许损害及理由、统计方法、嵌套超参数搜索上限、资源上限。无依据的安全阈值不能填数字冒充已定。
6. 实现候选完整性、绝对时间、所有15个6h支持时次的内容校验、不可覆盖产物和逐文件hash。debug与DEV证据目录分离。
7. OOF静态/regime/ridge实现就绪但不读DEV标签；冻结所需纯函数、折规则和参数网格。all-fold模型不能复用包含外折validation的数据拟合的scaler/PCA。

### Tests
- 训练/校准/探索/被拒模型的曝光均阻断；t在窗口内但support越界也阻断；2020H2禁止读取。
- 对一个外折validation标签毒化，该折模型与预测不变；其他折若将这些样本用作训练不要求不变。
- 缺candidate、extra/duplicate row、换Fs/coeff/objective、缺冻结文件、错step键皆拒绝。
- FP32端点下native端点损失差与gain一致；禁止以放宽S0阈值掩盖评分精度问题。
- 每个真实入口必须有negative spy，确认拒绝发生在数据读取/模型前向前。

### Run command（实现NEW入口并解锁后）
```bash
python -m pytest -q tests/test_split_freeze.py tests/test_fp05_consumption_chain.py tests/test_plan_cache_evaluation.py tests/test_policy_oof_isolation.py
python scripts/plan_followup_runner.py verify-contracts --run "$RUN"
python scripts/plan_followup_runner.py freeze-dev --run "$RUN" --declaration-only
```

### Required evidence / outputs
- `contracts/pytest.log`
- `contracts/junit.xml`
- `contracts/negative_cases.json`
- `protocol/fp05_protocol.json`
- `protocol/fp05_protocol.sha256`
- `protocol/sampling_declaration.json`
- `protocol/split_freeze.json`
- `protocol/new_source_manifest.json`
- `protocol/authorization.json`
- `contracts/decision.json`

### PASS
- CPU完整入口测试通过，不只helper测试；没有修改认证core17。
- 协议/曝露/统计/阈值/预算一一冻结；confirm=UNASSIGNED_NO_ACCESS且2020H2为已曝光。
- 纯输入/完整性筛查和模型结果筛选严格分开；新CLI标明版本并通过--help/参数测试。

### FAIL / BLOCKED / STOP
- 阈值/预算/原始证据仍未定；split存在遗漏；保护源码改变；任何对未来/validation标签的越权读取。
- 不得因CPU通过自动提交GPU；GPU仅在本节点验收及明确授权后。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-05b): 实现 split/消费链、CPU测试与结果揭示前协议`。

## FP-05c — C-J1 调试复现、只测成本、一次冻结N

**依赖：** FP-05b  
**状态：** LOCKED / TO_BE_RUN  
**CPU/GPU：** Real GPU C-J1 + CPU verification; explicit new authorization required  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
先在8个已曝光issue检验新cache路径和成本，不产生DEV效果。

### 输入／文件
- `scripts/r4_candidate_cache.py (NEW)`
- `scripts/r4_cache_decide.py (NEW)`
- `scripts/plan_followup_runner.py (NEW)`
- `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank/`
- `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/admission/bank_fit_grouping.json`

### Implementation
1. 固定每组首尾各一条，共8个已曝光debug issue；每条由两个独立worker重复，保持官方backend/FP32/TF32-off。
2. 每条五候选+独立F0背景，全部12步只存6/24/72端点；交叉重复state要求bitwise相同。
3. 与FP04已存在的面板只比较可比cell：每条的Fs与所属专家own-group结果；其他专家该issue没有旧panel，不能伪造历史锚点。
4. CPU对保存端点独立FP64计分，numerical reduction tolerance事前从算术测试冻结，不要求不同reduction backend天然bitwise同分。
5. 记录实测step、I/O、peak memory、存储和失败成本；按冻结规则一次确定N及exact DEV list，不能查看DEV任何candidate效果。
6. 保存C-J1原始job receipt和复算结果，profile不允许更新已认证bank。

### Tests
- 小模型/合成测试先拒绝anchor缺失、错误所属expert、背景F0混入oracle等。
- 同一状态输出/权重全生命周期hash不变，拒绝profile原地优化。

### Run command（实现NEW入口并解锁后）
```bash
python scripts/plan_followup_runner.py cache-debug --run "$RUN"
python scripts/r4_cache_decide.py --run "$RUN" --phase debug --verify-only
python scripts/plan_followup_runner.py freeze-dev --run "$RUN" --finalize-from-profile
```

### Required evidence / outputs
- `jobs/C-J1/job_result.json`
- `jobs/C-J1/invocation.json`
- `jobs/C-J1/stdout.log`
- `cache/debug_manifest.json`
- `cache/debug_anchor_check.json`
- `costs/debug_profile.json`
- `protocol/dev_issue_ids.json`
- `protocol/dev_size_decision.json`

### PASS
- 全部预定debug重复与现有可比anchor通过；bank/Fs/source不变。
- 按实测且不按效果选择N一次；剩余预算足够完整矩阵和验收。

### FAIL / BLOCKED / STOP
- 任何差异/identity失配；缺GPU或预算；无法完成合格样本量；擅自放宽数值规则。
- C-J1不能充当DEV样本，也不能在看过DEV结果后重跑用于扩样。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-05c): C-J1 调试复现、只测成本、一次冻结N`。

## FP-05d — C-J2 完整五候选缓存与独立F0背景

**依赖：** FP-05c  
**状态：** LOCKED / TO_BE_RUN  
**CPU/GPU：** Real GPU C-J2 only + CPU I/O/recompute  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
只运行一次冻结DEV矩阵；未来标签封存在评价/训练分区，不发给selector预测路径。

### 输入／文件
- `earthdelta/candidate_cache.py (NEW)`
- `scripts/r4_candidate_cache.py (NEW)`
- `scripts/r4_cache_decide.py (NEW)`
- `scripts/plan_followup_runner.py (NEW)`

### Implementation
1. 每个worker消费同一Fs/bank/source/protocol/split和确定分片；按id对齐而非完成顺序。
2. 读取前再次验证完整支持窗口、角色、原始字节证书。每issue五候选×3端点+F0背景单独存放；完整候选不重标为六个。
3. 合法features独立文件，只用t-12/t-6/t与登记日历。truth、real response、oracle/loss不可进入serving feature artifact。
4. 保留全部planned/attempted/completed/failed。任何candidate缺失使issue/shard不合格，禁止成功子集计分或填零；按事前infra retry规则处理。
5. merge重hash所有文件，核对N×5候选交叉积、F0覆盖、源row和3个lead，发布前独立CPU重算native loss/gain。

### Tests
- 缺失/重复/多余候选、错误role/source/lead、损坏tensor、同id错content全部拒绝。
- 扰动truth不能改变feature哈希；F0不是reference或候选；重试不得覆盖原尝试。

### Run command（实现NEW入口并解锁后）
```bash
python scripts/plan_followup_runner.py cache-dev --run "$RUN"
python scripts/r4_cache_decide.py --run "$RUN" --phase dev --verify-only
```

### Required evidence / outputs
- `jobs/C-J2/job_result.json`
- `jobs/C-J2/invocation.json`
- `jobs/C-J2/stdout.log`
- `cache/cache_manifest.json`
- `cache/issue_manifests/`
- `cache/failures.json`
- `cache/native_recompute.json`
- `costs/cache_costs.json`
- `cache/decision.json`

### PASS
- 精确冻结矩阵完整且可复算，所有候选/背景同一分母；权重未变。
- 模型与标签分开保存，fold访问接口可验证；未读confirm。

### FAIL / BLOCKED / STOP
- 任何前驱过期或样本/候选身份不匹配；任何失败被删掉；重试超1；看结果后改N/时段/强度。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-05d): C-J2 完整五候选缓存与独立F0背景`。

## FP-05e — 嵌套purged OOF：cheap policies与预测冻结

**依赖：** FP-05d  
**状态：** LOCKED / TO_BE_RUN  
**CPU/GPU：** CPU; actual runtime measured, no GPU weather calls  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
训练no-edit、fold-train static、辅助regime和主ridge；不训练双头，不执行新天气rollout。

### 输入／文件
- `earthdelta/policy_oof.py (NEW)`
- `scripts/r4_policy_oof.py (NEW)`
- `tests/test_policy_oof_isolation.py (NEW)`

### Implementation
1. 按冻结时间blocks构建双向outer5fold，所有issue候选同折；完整support+buffer从train净化。inner3fold用于ridge超参数，scaler/PCA/kmeans都只拟合相应train。
2. no-edit固定gain0；static仅按outertrain最优五候选选，tie先no-edit；regime不根据外折结果调整分组。
3. ridge可multi-output四gain或5含零，但no-edit输出固定0，选择规则冻结。只用合法features，不能访问验证真实candidate forecast/response/oracle。
4. 每个issue每方法恰好一份OOF动作/预测，保存fold-model/scaler/HPO输入identity和training-issue集。
5. 训练/预测完成后先写模型和全OOF动作冻结manifest，再允许score揭示该动作的验证结果。保持本轮retrospective DEV措辞。

### Tests
- 外折validation标签毒化仅检查本折预测不变，inner/outerrole边界检查先于I/O。
- 整dataset拟合scaler/PCA、outervalidation参与lambda选择、候选跨折等反例必须失败。
- 无prediction freeze或缺issue时scorer拒绝。

### Run command（实现NEW入口并解锁后）
```bash
python -m pytest -q tests/test_policy_oof_isolation.py tests/test_plan_cache_evaluation.py
python scripts/r4_policy_oof.py fit-freeze --run "$RUN"
```

### Required evidence / outputs
- `policies/folds.json`
- `policies/inner_folds.json`
- `policies/fit_records.json`
- `policies/models/`
- `policies/oof_predictions.json`
- `policies/prediction_freeze.json`
- `policies/access_log.json`
- `policies/decision.json`

### PASS
- 全部fold访问和HPO合法；每issue一次预测；模型/动作先于验证评分冻结。
- 允许真实train数据带标签，但测试heldout-fold标签不影响对应模型。

### FAIL / BLOCKED / STOP
- 泄漏、fold交叠、验证标签影响本折预测、missing模型/动作/hash、事后挑seed/超参数。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-05e): 嵌套purged OOF：cheap policies与预测冻结`。

## FP-05f — 固定DEV结算、配对块统计与分层判决输入

**依赖：** FP-05e  
**状态：** LOCKED / TO_BE_RUN  
**CPU/GPU：** CPU only  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
回答同一认证bank在新时段有没有价值、是否需要动态selector；绝不将oracle或训练资格当效用。

### 输入／文件
- `earthdelta/policy_oof.py (NEW)`
- `scripts/r4_policy_oof.py (NEW)`
- `scripts/r4_cache_decide.py (NEW)`

### Implementation
1. 重新核验prediction freeze和全部cache文件；执行后才读取OOF actions的outcomes。
2. 报告native24h、gain_vs_Fs、gain_vs_OOF_static、same-panel F0 contrast、72h损害、harm/noeditrate、失败与全成本。oracle在完整五候选上独立诊断。
3. paired UTC block resampling同时抽所有方法，以Fs同一重采样均值归一化；不把grid/candidate/lead/重复设备当n。保存seed与draw指标。
4. 区分条件于本次已拟合OOF策略的CI与完整学习流程不确定性；本轮只是DEV，绝不标confirm。
5. 固定0.0034工程screen和预先冻结72h规则；不得用MDE替代价值。CI跨阈值或块数不足保留INCONCLUSIVE，不追加样本。

### Tests
- 代数恒等式G_F0=G_Fs-h_Fs_on_same_panel；F0背景不进argmax；每个bootstrapdraw归一化正确。
- 异常宽CI、无效块、不同方法分母、状态缺失不得变PASS；oracle常大但合法policy零收益的合成反例。

### Run command（实现NEW入口并解锁后）
```bash
python scripts/r4_policy_oof.py score --run "$RUN"
python scripts/r4_cache_decide.py --run "$RUN" --phase dev --verify-only
```

### Required evidence / outputs
- `evaluation/dev_results.csv`
- `evaluation/paired_block_bootstrap.json`
- `evaluation/costs.json`
- `evaluation/failures.json`
- `evaluation/same_panel_F0.json`
- `evaluation/independent_recompute.json`
- `evaluation/claims.json`

### PASS
- 结果全部可从冻结端点和动作重算，给出效应/CI/有效blocks/限定。
- 没有结果导向阈值/分组/N改动，unknown bank价值由本实验回答而非先假定。

### FAIL / BLOCKED / STOP
- 结果不可重算、数据泄漏、方法分母不同、阈值回改、样本追加或confirm读取。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-05f): 固定DEV结算、配对块统计与分层判决输入`。

## FP-06 — 独立最终DEV决策与handoff

**依赖：** FP-05f  
**状态：** LOCKED / TO_BE_RUN  
**CPU/GPU：** CPU only  
**预计耗时：** MISSING_NEEDS_MEASURED_PROFILE

### Goal
在完整证据后选择CONTINUE/PIVOT/STOP/INCONCLUSIVE；不自动开启confirm或新方法训练。

### 输入／文件
- `scripts/plan_followup_runner.py (NEW)`
- `evaluation/`
- `protocol/`
- `evidence/PLAN_AUDIT_RESULT.json`

### Implementation
1. 只消费已冻结规则与FP05f结果，不允许按期望方向改变解释。
2. oracle不足只停止当前bank；static足够就转static；oracle有而cheap合法策略不够则报告限定的选择失败/不确定性；不要把未显著视为等效。
3. Ridge合法增益只解锁未来研究设计，不证明response factorization新颖或有效。
4. 输出新PLAN_AUDIT_RESULT/HANDOFF和下一预算要买的证据；不覆盖本次历史审查/原v1失败。

### Tests
- decision纯函数对所有STOP分支和CI边界测试；对缺少FP05证据的输入拒绝positive结论。

### Run command（实现NEW入口并解锁后）
```bash
python scripts/r4_policy_oof.py verify-results --run "$RUN"
python scripts/plan_followup_runner.py decision --run "$RUN"
```

### Required evidence / outputs
- `decision.json`
- `HANDOFF.md`
- `PLAN_AUDIT_RESULT_AFTER_FP05.json`
- `claims.json`
- `evidence_manifest.json`

### PASS
- 每个结论可追至同一source/protocol/asset/job/样本/动作/指标；明确DEV与confirm差异。

### FAIL / BLOCKED / STOP
- 缺任何前驱或证据、无OOS结果却宣称bank value、试图把2020H2称untouchedconfirm。
- 有效输入下数值/契约不符为FAIL；缺资产、回执、授权或未定阈值为BLOCKED；统计CI不够为INCONCLUSIVE。它们不是同一种科学失败。

### Decision after completion
保存本节点decision及证据哈希，按DAG申请后继。不得自动越过资源授权。Suggested commit: `feat(fp-06): 独立最终DEV决策与handoff`。

## 防止旧入口改标签绕过
新cache不能把policy_dev重新标为bank_fit来复用仅接受bank_fit的旧训练loader。应使用新role-aware只读样本视图与纯数值函数，并在任何数据读取前消费冻结split capability。
