# EarthDelta 原计划执行审查报告

审查日期：2026-09-23。**overall_verdict：PARTIALLY_EXECUTED。**  
计划对象：`plans/plans_v3_0922`；受审实现：`a3596e6b804e9d23b72d1247b08c47129d6b50b1`；比较基线：`d749a1c62521226df857587e08f7d067b0f15355`。
实际分支HEAD：`fdd92d79b1b03e8397d4c487d7fa4a347ae6fc37`，其earthdelta/scripts/tests代码树与受审实现完全相同，plans改变；没有新的实现修复可以据此认定已发生。

## 1. 决策摘要

原计划**部分执行、有真实工程和历史GPU运行进展，但关键科学依赖未完成**。这不是“只写了方案”，也不是“已证明novelty或weather utility”。

- 当前S0：提交内最新记录为1-step PASS、4/12-step FAIL，固定1e-5；最新原始GPU结果/张量尚未提供，因此本审查没有独立复算最新数值。
- 原Stage 3b：static/Fs/expert模块、历史GPU更新确实存在；部分formal Fs同配对样本8/8退化、仍eligible；worker各自Fs不同，不能组成common-reference bank。
- B13/B14代码修复对症，不能再次当成未修；B05/B06/B08/B09/B12/B15存在不同程度的真实入口消费或范围缺口。
- 新shared-F0 pipeline是另一个明确研究设置，不能给原fitted-Fs目标销账；它有checkpoint/qualification/cache实现进展，但最新sprint下游未启动。
- Stage4原计划要求的完整缓存、廉价OOF策略开发表无合格执行证据。dual-head、E3、confirm在原计划明确暂缓，未做它们不是本窗口的违约。
- 存在一项明确的**局部FAILED_PLAN_COMPLIANCE**：旧stage-all在前驱False/exception后仍继续调用后继。整体选择PARTIALLY_EXECUTED反映已经存在的工作，不代表对此违规放行。

## 2. 阅读及证据边界

附件16文件均已检查；source_plan全部五文件完整阅读，包含NEXT_STEPS、CLAUDE prompt、三份Stage1A review；然后阅读CURRENT_STATUS/EVIDENCE_MANIFEST和reference_reports。三份review面对同一旧packet，不算三次独立实跑。

随后通过授权GitHub连接重新核对branch/commit/tree、当前代码关键范围与提交内日志。没有逐行读完整仓库或巨型commit diff，也没有运行当前仓库测试/GPU/训练。ZIP无法由Files解析，已使用本地zipfile安全解压，记录输入hash。

证据四层不混用：

| 层 | 本次可核实什么 | 不能推出什么 |
|---|---|---|
| L1代码 | 固定commit上的函数、入口和具体缺口，标STATIC_CONFIRMED | 不能推出测试已运行 |
| L2CPU测试 | Round4原始两组日志307 passed、127 passed；有当时source快照索引，标EXECUTION_CONFIRMED且限历史工作区 | 不是本次当前HEAD全仓测试；66/3最新组原日志MISSING |
| L3GPU | 历史集合内嵌job_result与run记录证明曾执行；最新S0只有提交内摘要，未取得原始证书 | 不能把SUCCEEDED、finite或准备argv当official parity/资格PASS |
| L4科学 | 需要合格共同reference/bank、完整候选、合法策略预测与结果 | 当前无此闭环；novelty/weather utility均INCONCLUSIVE |

更完整证据路径、读取范围和hash见evidence/EVIDENCE_MANIFEST.json。原始计划副本与提交内报告也在包内保留；CURRENT_STATUS仅索引，不作为单独成功证明。

## 3. 原计划的真实范围

`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:11–19`将目标分为保底、正常条件目标和扩展；允许交付具体S0阻塞。`:99`要求影响bridge/norm/source/rollout的改动重新验证。`:180`要求每阶段消费前驱、失败不启动依赖。

`:229–239`明确是bank_fit训练的**Fs**、K4/rank4/blocks18–23、5候选、a0=rho=.25、hold4×6h后继续Fs；不是后来的F0+9/13候选幅度插值协议。`:165`要求t-12/t-6/t历史。`:237`明确暂缓dual-head/E3/confirm。

因此本次不以“没完成128例/双头/holdout”自动判违约，也不以“新F0路线可运行”宣布原Fs路线达标。

## 4. 逐项计划对照

下表CPU栏的“历史套件”只指已读两组日志；具体最新源码逐测试绑定仍需FP-00/回归，不能把它当成新的PASS。

| 项目 | 代码/真实入口 | CPU/测试证据 | GPU/实际执行 | 目标状态/缺口 |
|---|---|---|---|---|
| B05 | native权重/scale/分母校验已实现；旧诊断零F切片仍可除0 | 历史套件 | 未单独认证此旧边界 | 部分完成，主路径限native |
| B06 | 同Q_eff的loss/gain，FP64 reducer进入训练 | 历史套件 | 旧Fs确有native训练 | 部分完成；旧head/新调用侧FP32先差分残留 |
| B08 | 内容验证producer及新F0再读存在 | 历史套件 | 旧admission8行仅2证书；新616仅摘要 | 旧消费者未完整核验外层/行证书 |
| B09 | 真time join入口存在，旧loader按index消费 | 历史套件 | 旧declared6h而actualtrain24h记录 | 完整时次/历史/目标消费链未闭合 |
| B12 | controlled rollout模型级锁和checkpoint flag检查 | 历史套件 | 串行运行可行，通用共享安全无证据 | plain forward未受同锁，禁止共享并发 |
| B13 | 必要检查后committed PASS；critical exception撤销；cleanup另记 | 历史套件 | 历史失败证书确实false；最新完整bundle缺失 | 代码缺陷修复，不等于S0成功 |
| B14 | 安全format_finite进入report | 历史套件 | 无需以完整天气实验验证格式化 | 代码修复，非科学完成 |
| B15 | registry值域校验、选择入口均存在 | 历史套件 | 旧训练有registry但缺动态权重产物 | 原Fs/bank artifact身份绑定未闭合 |
| H100环境 | 运行脚本存在 | 最新66/3原日志缺失 | 历史job records真实；最新xformers probe仅摘要 | 环境与official parity分开 |
| S0 | actual/expected、raw-normalized、1/4/12多步入口存在 | 历史逻辑测试 | 最新摘要：1step过，4/12失败 | BLOCKED；不能复活旧Stage1A自比较指控 |
| Fs fit | static训练/合并/曲线存在 | 历史套件 | 三作业，formal中4个worker各8/8退化 | 质量门未消费，UNRESOLVED |
| K4/rank4 | 动态训练与分组存在 | 历史套件 | 4个不同Fs/每作业；旧dynamic未持久化 | 非共同合格bank |
| Qualification/registry | 旧eligible检查有限性，新F0独立qualify代码 | 新F0测试定义不是运行日志 | 新路径NOT_STARTED；旧质量未合格 | 未完成原目标 |
| Cache | 新F0 cache/merge代码完整阶段入口 | 测试定义存在 | 正式产物MISSING/新sprint NOT_STARTED | 未完成原Fs五候选缓存 |
| Cheap policy开发 | 原语存在，无完整fit/freeze/eval CLI | 无端到端拟合日志 | MISSING | 原条件性目标未实现 |
| Holdout/dual-head | 原计划本窗明确不做 | 不适用 | 未执行 | 不是本窗缺交；也没有holdout科学支持 |
| STOP与source | 新S0 fail-closed改进；旧stage-all继续 | 部分逻辑测试 | 最新sprint报告停住；旧流水线不严格 | 局部FAILED_PLAN_COMPLIANCE |
| novelty/weather utility | 无合格科学产物闭环 | 单元测试不能证明 | 无完整合法策略表 | INCONCLUSIVE |

### 逐项可定位证据

**B05 — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`earthdelta/metrics_contract.py:100–160`；`earthdelta/metrics_contract.py:199–215`；`earthdelta/metrics_contract.py:557–617`。native目标路径已有校验；legacy quadratic_gain 仍先验全局权重和、再除逐F切片和，零切片仍可能NaN。主pilot禁止使用该legacy评分，非主路径残留不等于天气结果已被污染。

**B06 — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`earthdelta/metrics_contract.py:652–833`；`earthdelta/static_adapter.py:948–954`；`earthdelta/value_pilot.py:170–183`。full_objective_loss/gain实现并进入训练；旧head端到端原生目标尚未闭合。shared-F0 component_gains在进入double scorer前先作FP32差分，须单独修复；不要机械归咎native reducer。

**B08 — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/r2_admission_gate.py:176–216`；`earthdelta/static_adapter.py:1809–1939`；`earthdelta/value_pilot.py:79–151`。旧入口可只抽样验内容；load_admitted_samples不强制消费外层PASS/每行证书。新F0有更严格再读hash，但不是旧Fs入口自动修复。

**B09 — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/r2_admission_gate.py:276–311`；`earthdelta/static_adapter.py:1850–1913`；`scripts/r4_freeze_sprint.py:57–107`。producer真实time join存在；旧loader按保存index再取目标，未完整核验外层合同/绝对时次/证书。原计划t-12,t-6,t三帧不能被新F0只读t-6,t悄悄替代。

**B12 — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`earthdelta/bridge/stormer_bridge.py:848–881`；`earthdelta/bridge/stormer_bridge.py:889–1057`。controlled_rollout已有模型级进程内互斥与标志检查；普通forward_validation不进入同一锁。当前严格串行路线可用，不应宣称通用并发安全，也不需要先扩建并发框架。

**B13 — COMPLETE / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/s0_gate.py:2407–2470`。原代码缺陷已对症修复，不能继续照抄旧finding；整个S0真实发布成功仍未成立。

**B14 — COMPLETE / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/s0_gate.py:2470–2515`。format_finite及generate_report真实调用已存在；报告可生成不是官方parity或科学目标通过。

**B15 — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/r2_admission_gate.py:220–250`；`earthdelta/selection.py:365–395`；`scripts/r2_fs_bank_train.py:545–581`。registry值域验证和选择入口存在；旧流程没有合格共同Fs/动态checkpoint的完整消费链。近零候选映射问题只在相应值域触发，固定0.25不是它的实际反例。

**ENV — PARTIAL / PARTIALLY_ACHIEVED / EXECUTION_CONFIRMED**  
`scripts/r4_prepare_environment.py:1`；`scripts/r4_run_strict_fp32.py:1`。历史GPU可运行已支持；最新official环境细节不能由旧SDPA训练任务或自述替代。环境可运行不等于S0通过。

**S0 — BLOCKED / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/s0_gate.py:2334–2382`；`earthdelta/bridge/stormer_bridge.py:236–370`；`scripts/r4_s0_parity_diagnose.py:112–148`。真实身份比较/raw-normalized绑定/多步入口已实现，旧Stage1A自比较不能再照搬。最新多步记录FAIL；input inverse源精度是假设，未取得完整GPU trace不能宣称唯一根因。

**FS_FIT — BLOCKED / PARTIALLY_ACHIEVED / EXECUTION_CONFIRMED**  
`earthdelta/static_adapter.py:891–994`。真实训练发生，但fit仍仅以有限loss/grad及updates判eligible。历史8/8退化未关闭；同面板最终Fs/F0亦恶化，不能用合并数值一致性替代训练质量。

**COMMON_FS_BANK — BLOCKED / PARTIALLY_ACHIEVED / EXECUTION_CONFIRMED**  
`scripts/r2_fs_bank_train.py:298–349`；`scripts/r2_fs_bank_train.py:520–624`；`earthdelta/bank_training.py:1297–1355`。K4/rank4模块与GPU更新存在，但不是共同参考的可重载合格bank；profile会在训练后的同bank上optimizer.step，训练记录与后续参数可能错位。

**STAGE4_CACHE — PARTIAL / NOT_ESTABLISHED / STATIC_CONFIRMED**  
`scripts/r4_value_pilot.py:319–446`。新shared-F0 cache/merge代码存在但正式产物未运行，且动作与参考协议不同；不能认作原Fs五候选缓存完成。

**STAGE4_POLICY_DEV — NOT_STARTED / NOT_ESTABLISHED / MISSING**  
`scripts/r4_value_pilot.py:449–493`；`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:259–274`。已有head原语不等于fit/prediction-freeze/evaluation CLI；未见拟合权重、OOF predictions、结果表与成本。

**DEFERRED_HOLDOUT — NOT_APPLICABLE / NOT_ESTABLISHED / STATIC_CONFIRMED**  
`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:235–237`。未做这些延期项目本身不违反原计划。也不能反向宣称已证明holdout无泄漏；本包仍不解锁确认集。

**GATE_CHAIN — PARTIAL / PARTIALLY_ACHIEVED / STATIC_CONFIRMED**  
`scripts/r2_fs_bank_train.py:963–1005`；`scripts/s0_gate.py:1865–1946`；`scripts/r4_value_pilot.py:59–77`。旧runner仅在末尾all(results)，中途继续调用；新路线停住不修复旧入口。source hash记录未完整作为消费者准入合同核验。

**F0_AMENDMENT — PARTIAL / NOT_ESTABLISHED / STATIC_CONFIRMED**  
`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:229–239`；`scripts/r4_value_pilot.py:109–145`；`scripts/r4_freeze_sprint.py:92–123`。shared-F0是公开另案，不应机械套用旧worker-Fs缺陷；但F0参考、9/13候选、两帧历史不是原计划完成，原Fs阻塞不能被它清账。

**SCIENCE — NOT_STARTED / NOT_ESTABLISHED / MISSING**  
`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:278–288`；`plans/novelty_review_20260922/SPRINT_8H_DECISION.json:1`。没有合格共同参考bank上的完整合法策略结果；novelty与weather utility均INCONCLUSIVE。此处不重新评估文献价值。


## 5. 特别判决

### S0 parity

最新提交内证据（`plans/novelty_review_20260922/SPRINT_8H_DECISION.json`；本包副本`evidence/reference_reports/SPRINT_8H_DECISION.json`）对应 **pt-7ant09uv**：

| rollout | max_abs_diff | frozen tolerance | 记录判定 |
|---|---:|---:|---|
|6h×1|5.9604644775390625e-06|1e-5|PASS|
|6h×4|3.933906555175781e-05|1e-5|FAIL|
|6h×12|1.3911724090576172e-04|1e-5|FAIL|

`STATIC_CONFIRMED`是指这份提交内摘要可读；最新原始job_result、gate JSON完整criteria、输入/输出tensors与producer环境在本次为MISSING。没有新artifact改变这些状态，故保持BLOCKED，而不是自行推断“已经修好”。

当前代码`scripts/s0_gate.py:2334–2382`已实际调用expected-version/input binding/multistep parity。旧“版本自比较、4步完全不加载”的判断不是当前事实。剩余源码真实性消费问题见source identity检查，以及input inverse精度的待验证假设。

input inverse在`.float()`后求倒数的差异可作为FP-01假设；不能只凭其数学可能性宣布是完整H100多步误差的唯一原因。`r4_s0_parity_diagnose.py`后处理分支必须与真正模型条件一样除10，并记录四臂每一步，不放宽容差。

### Fs-fit divergence

历史执行集合是`codex_audit_round4/evidence/gpu_records_recomputed.json`，不是只准备了argv。它包含job_result、原始run_record路径和hash。

| job | worker | paired improved | 最终Fs/F0同面板loss比（报告重算） |
|---|---|---:|---:|
|pt-3x63g0c6|expert1|0/8|4.644734|
|pt-3x63g0c6|expert3|0/8|1.219248|
|pt-cdj1s2le|expert2|0/8|4.066871|
|pt-cdj1s2le|expert3|0/8|1.070957|

精确原始目录、记录hash和核验范围见EVIDENCE_MANIFEST。这里的first/last按同issue每次访问配对，但不是同一初始/最终checkpoint共同面板；Round4另算最终Fs/F0也退化，故不是仅“比较两个不同样本”造成的假象。

结论：**UNRESOLVED_MUST_FIX_BEFORE_STAGE4**。现代码`static_adapter.py:891–994`仍未把质量退化用于eligible；seed相同不等于四个Fs权重相同，四个source摘要实际不同。不能只剔除两个坏worker后把剩下两个称共同Fs。

这是旧设置下的有限但显著训练退化，不是证明无限发散、不是什么硬件故障定理，也不直接否定EarthDelta研究价值。

### Stage4 readiness

**BLOCKED_BY_S0_AND_QUALIFIED_COMMON_FS_BANK**。

新F0脚本`scripts/r4_value_pilot.py:449–493`只有reference、train-bank、qualify-bank、assemble-bank、cache、merge-cache。没有已经执行的head fitting、prediction freeze或holdout evaluation证据。新sprint的“gate失败后未启动”与旧historical Fs/expert job已执行是不同时间/路线，不能合并成一句“从未训练过”。

### Novelty / weather utility

均为 **INCONCLUSIVE_NOT_EXPERIMENTALLY_ESTABLISHED**。本次只查是否有科学证据，不重新做新颖性文献裁决。原计划延期的dualhead/confirm不能作为违规，但其效果也不能先宣布成功。

## 6. 阻塞项、最小修复与优先级

### PA-01 [P0] S0最新提交记录多步失败，原始最新证书缺失

证据标记：STATIC_CONFIRMED / OBSERVED。`plans/novelty_review_20260922/SPRINT_8H_DECISION.json:10–23`。

影响：BLOCKED；不能认证official rollout或解锁训练。 最小处置：补齐pt-7ant09uv原始bundle；四臂逐步定位，不放宽1e-5；新source重新导出并复验。

### PA-02 [P0] Fs质量信号未进入eligible/冻结准入

证据标记：STATIC_CONFIRMED / OBSERVED。`earthdelta/static_adapter.py:891–994`；`gpu_records_recomputed.json: pt-3x63g0c6/expert1,expert3; pt-cdj1s2le/expert2,expert3`。

影响：旧Fs-fit divergence仍UNRESOLVED。 最小处置：固定初始/候选终态与共同bank_fit面板；冻结训练侧资格规则，差的Fs禁止发布及后续使用。

### PA-03 [P0] stage-all失败不阻断依赖阶段

证据标记：STATIC_CONFIRMED / OBSERVED。`scripts/r2_fs_bank_train.py:963–1005`；`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:180`。

影响：局部FAILED_PLAN_COMPLIANCE；最终returncode非零不足以撤销已经执行的训练。 最小处置：每阶段读取前驱证据；False/exception/MISSING则后继BLOCKED未调用；单阶段入口同样强制检查。

### PA-04 [P0] 历史四worker未共享同一个Fs

证据标记：EXECUTION_CONFIRMED / OBSERVED。`codex_audit_round4/evidence/artifact_crosschecks.json: distinct_fs_digests`；`scripts/r2_fs_bank_train.py:298–447`。

影响：不构成common-reference bank。 最小处置：只拟合/资格化/冻结一次Fs，所有worker加载并核对该状态内容hash。

### PA-05 [P0] admission producer与training consumer未闭环

证据标记：STATIC_CONFIRMED / OBSERVED。`scripts/r2_admission_gate.py:176–216`；`earthdelta/static_adapter.py:1809–1939`。

影响：6h准入可被24h训练消费，抽样证书不能覆盖全部训练行。 最小处置：校验外层PASS及spec/source/normalization/实际时次/全部历史和目标切片hash；任何未证端点拒绝。

### PA-06 [P1] 旧动态权重未落盘且profile改变训练后bank

证据标记：STATIC_CONFIRMED / OBSERVED。`scripts/r2_fs_bank_train.py:541–624`；`earthdelta/bank_training.py:1297–1355`。

影响：无法独立恢复评分所用bank，记录可能与实际权重错位。 最小处置：复用现有dynamic序列化形成fitted-Fs身份版本；profile克隆/恢复且前后hash一致，组装后逐expert行为等价。

### PA-07 [P1] source身份记录尚不是完整准入约束

证据标记：STATIC_CONFIRMED / OBSERVED。`scripts/s0_gate.py:1865–1946`；`scripts/r4_value_pilot.py:30–77`。

影响：不能只凭pinned字符串/同config给修改后源码延续旧证书。 最小处置：核验实际导入文件的冻结hash和环境；改变bridge/norm/rollout时重新签发S0，消费者复核证书链。

### PA-08 [P1] 原生评分修复不覆盖所有旧调用及新FP32端点差分

证据标记：STATIC_CONFIRMED / OBSERVED。`earthdelta/metrics_contract.py:199–215`；`earthdelta/metrics_contract.py:652–833`；`earthdelta/value_pilot.py:170–183`。

影响：不能由native reducer正确推断全部heads/cache gain正确。 最小处置：主路径只用共同Q；端点先double再差分；legacy零切片拒绝；校准独立保存且默认关闭。

### PA-09 [P1] 新F0路线不能冲抵原Fs计划与Stage4交付缺口

证据标记：STATIC_CONFIRMED / OBSERVED。`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:229–239`；`scripts/r4_value_pilot.py:449–493`。

影响：存在新实现进展，但原Fs完整cache/OOF决策表仍缺。 最小处置：本包固定原计划；若另选F0必须独立协议/身份/授权，不得报告原计划完成。

### PA-10 [P0] 无可支持科学目标的最终产物

证据标记：MISSING / MISSING。`required: qualified_fs + common_fs_bank + complete_5_candidate_cache + OOF_predictions + native_results + costs`。

影响：novelty/weather utility=INCONCLUSIVE。 最小处置：保持阻塞；不能先写新颖性胜利或收益结论。


## 7. 不能对外宣称的结论

- 不能声称plans_v3_0922全部执行或目标全部实现。
- 不能把历史307/127两组测试或CURRENT_STATUS的66/3宣称为本次完整HEAD测试。
- 不能声称官方Stormer 4-step/12-step parity已通过。
- 不能把job SUCCEEDED、finite或合并一致性称为Fs训练质量达标。
- 不能声称旧四worker构成共同Fs上的合格K4专家库。
- 不能声称Stage4 cache、policy fitting、OOF/holdout evaluation已完成。
- 不能用shared-F0代码代替原计划的fitted-Fs完成证明。
- 不能声称双头/响应表征具有novelty收益或提高weather utility；目前无合格实验。
- 不能说本次独立重跑GPU/训练/全仓测试；本次仅进行了材料、代码和交付检查。

## 8. 本轮建议与原计划的关系

下一笔资源只买S0定位/独立复验及必要CPU消费门修复。S0通过以后先解决单一Fs质量与冻结，再进入共同Fs专家、缓存、廉价OOF策略表。不能绕过Fs退化用F0路线证明原计划完成。

原计划是条件性的：S0未过时不完成cache是遵守门控，不是自动违约；旧runner前驱失败仍调用下游则是实质违约。两者要同时记录。

若合格bank的完整oracle余量不足，停止当前bank；若static解释主要收益，转向static；若合法策略仍无收益，停止当前selector扩张；CI跨阈值则INCONCLUSIVE，在预登记cap内补证。没有必要为“看起来工作更多”加新模块。

本包不执行这些未来阶段，所有任务结果仍TO_BE_RUN。实际观测、历史摘要与缺失输入在manifest逐项分开。
