# STOP_CONDITIONS — 一个迭代的停止合同

迭代：`round2-next-iteration-20260921`。所有判断限于冻结 reference、bank、registry、合法信息、Q、lead 和预算域。不得把当前配置失败推广成所有参数编辑不可能。

## 1. 状态与自动化边界

| 状态 | 含义 | 是否允许下任务 |
|---|---|---|
| PASS_MEASUREMENT / PASS_BANK_AND_DATA | 技术测量路径／字典资产合格 | 仅可提出授权请求，不自动执行 |
| DEV_PROMISING / LEGAL_POLICY_DEV_PASS | 开发证据达到已登记价值门槛 | 仅可申请相应小型后续任务，不等于 confirm |
| BLOCKED | 缺资产、许可、资源、阈值或可信测量 | 停止受影响执行；不得生成合成天气结果补位 |
| FAIL_IMPLEMENTATION | 确定实现错误让测量不可信 | 修当前允许的最小路径；无科学 STOP 结论 |
| INCONCLUSIVE | CI 跨价值阈值／独立样本不足 | 仅按已登记采样与cap补证；到cap停止追加 |
| STOP_CURRENT_* | 在规定范围内没有值得继续的证据 | 不解锁后续；保留负结果和完整分母 |
| PIVOT_* | 支持更窄或更便宜的命题 | 提交证据和转向建议；不自动实现新项目 |
| CONTINUE_BOUNDED_NEXT_ITERATION | 当前窄命题获得继续理由 | 结束本迭代，下一迭代仍须新授权 |

`codex_tasks.json` 初始只开放 P0-01。PASS 只改变执行者写出的结果，不会自动改变授权。技术依赖满足与预算/任务被授权是两把独立锁。

## 2. 阈值登记，不预填百分比

以下字段当前统一为：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。

- delta_min_vs_reference：oracle/策略相对固定参考的最小有用改善。
- delta_min_dynamic：相对 dev-best-static 的最小有用动态余量。
- delta_min_legal_policy：合法可部署策略相对强静态的最小有用改善。
- delta_min_factorization：资源／标签对齐后分解相对直接模型的最小有用差异。
- noninferiority_margin_long_lead：保护时效可接受的最大退化。

`threshold_certificate.json` 必须分别保存：指标单位与总体、delta_min及其价值/成本依据、数值误差地板、开发过程级方差、alpha、power、N或样本上限、MDE、预定looks、bootstrap规则、资源cap和批准记录。**delta_min 不由 pilot 方差/MDE 自动产生；“先估方差”只帮助判断可测性和所需成本。** 若没有价值依据，可输出效应和CI，但不能签科学PASS。

近似规划公式：MDE_0=(z_(1-alpha)+z_(1-beta))*sigma/sqrt(n)。这只是独立过程近似下相对0的可检测效果；检验超过delta_min时需对超过该阈值的备择设计功效。真实时间相关性应按过程块模拟，不把网格或候选计入n。

数值容差用独立重复/精度检查确定并先冻结，不由科学比较的胜负反推。未经批准的补样、加候选或改Q禁止。

## 3. 正式停止条件

### STOP-CORRECTNESS / PROVENANCE

身份、官方policy、实际所用输入/目标、共同Q或合法信息边界未确定：`BLOCKED_CORRECTNESS_OR_PROVENANCE`。缺GPU/checkpoint/data是BLOCKED，不是novelty失败。验证过的CPU原语不替代真实S0。

### STOP-ORACLE

完整、合格registry的oracle相对参考或强静态gap，若上CI小于其delta_min：`STOP_CURRENT_DICTIONARY`；仅动态gap不足则 `PIVOT_STATIC_ADAPTATION`。singleton-only只否定当前singleton域；未测组合H不得被顺带否定。无合格非零bank/缺候选则不能使用此规则。

### ORACLE-CONTINUE（仅必要条件）

两个预登记gap的下CI高于各自delta_min，只意味着hindsight空间值得做小型可预测性筛查。不能把E[max g]当E[max E[g|I]]，不能在此处声称deployable、factorization或parameter execution成立。

### STOP-LEGAL-POLICY

在预定信息、数据、模型/HPO和资源cap内，合法策略相对强静态的改善上CI低于delta_min_legal_policy：`STOP_CURRENT_PLANNER`。若全局static或小regime已经解释有用收益：`PIVOT_STATIC_ADAPTATION`。禁止凭“样本不显著”替代该判定；不是不可预测性定理。

### STOP-FACTORIZATION

paired相对direct-gain/直接候选端点模型，在同标签与同仿真资格、可比总资源下，上CI低于预登记有用差异，且没有已登记标签效率/动作复用优势：`PIVOT_NARROW_CLAIM`。不以加性calibration、更多hidden宽度、更多仿真或新模块隐藏失败。允许报告其它合法策略的价值，但不自动扩大它。

### STOP-PARAMETER-EXECUTION

feedback/joint output或virtual，在同信息与资源约束下稳定达到预登记的支配/非劣标准：`PIVOT_OUTPUT_CORRECTION`。virtual仅有summary预测时，不得用它声称全场支配；没有可比资源时标INCOMPARABLE_RESOURCE/BLOCKED，而非科学胜负。

### STOP-LONG-LEAD

保护时效退化 h=L_method-L_reference，若其下CI超过预登记noninferiority margin：`STOP_CURRENT_CONFIGURATION`。若上CI小于等于margin，可通过保护检查；否则INCONCLUSIVE。不得用平均主指标掩盖明确长时效失稳。

### INCONCLUSIVE / CAP

对收益delta，L>delta为支持继续，U<delta为支持当前配置停止；其余含边界相等均INCONCLUSIVE。阈值仍为占位符则BLOCKED_THRESHOLDS。允许的补样必须在原cap/looks内，所有结果仍保留。达到cap即结束追加，不能改口为确认零效应。

## 4. 失败与确认集

每个注册issue和candidate都有行。基础设施缺行阻断complete oracle；数值失败如需回退，必须事先登记可执行规则且算回退成本。禁止dropna、只报选中成功或仅保留合适天气过程。

本轮默认dev-only。confirm仅在全部registry、模型、阈值和比较方式冻结且有明确权限后读一次。已读confirm用于调整后续方法则立即降级EXPOSED_DEV；不能继续沿用其确认身份。

## 5. 每任务结尾必须写的记录

`decision.json` 至少包含 iteration_id、task_id、code_commit、parent_artifact_hashes、implementation_status、scientific_status、scope、effect/CI/threshold_certificate、cost_used/cap、failure_counts、allowed_next_task=null、recommended_next_task、requires_owner_authorization=true、reason。字段缺失不能默认为PASS。
