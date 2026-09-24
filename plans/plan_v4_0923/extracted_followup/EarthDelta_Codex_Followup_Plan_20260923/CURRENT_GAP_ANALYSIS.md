# 当前缺口分析（与审查JSON同源）

`overall_verdict=PARTIALLY_EXECUTED`；实现`a3596e6b804e9d23b72d1247b08c47129d6b50b1`；branch HEAD`fdd92d79b1b03e8397d4c487d7fa4a347ae6fc37`，代码树一致。
原计划路径`plans/plans_v3_0922`；不是新版F0计划的验收。

| 项目 | 实现 | 目标 | 证据级别 | 精确位置 | 未完成/阻塞 |
|---|---|---|---|---|---|
|B05|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`earthdelta/metrics_contract.py:100–160`；`earthdelta/metrics_contract.py:199–215`；`earthdelta/metrics_contract.py:557–617`|native目标路径已有校验；legacy quadratic_gain 仍先验全局权重和、再除逐F切片和，零切片仍可能NaN。主pilot禁止使用该legacy评分，非主路径残留不等于天气结果已被污染。|
|B06|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`earthdelta/metrics_contract.py:652–833`；`earthdelta/static_adapter.py:948–954`；`earthdelta/value_pilot.py:170–183`|full_objective_loss/gain实现并进入训练；旧head端到端原生目标尚未闭合。shared-F0 component_gains在进入double scorer前先作FP32差分，须单独修复；不要机械归咎native reducer。|
|B08|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/r2_admission_gate.py:176–216`；`earthdelta/static_adapter.py:1809–1939`；`earthdelta/value_pilot.py:79–151`|旧入口可只抽样验内容；load_admitted_samples不强制消费外层PASS/每行证书。新F0有更严格再读hash，但不是旧Fs入口自动修复。|
|B09|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/r2_admission_gate.py:276–311`；`earthdelta/static_adapter.py:1850–1913`；`scripts/r4_freeze_sprint.py:57–107`|producer真实time join存在；旧loader按保存index再取目标，未完整核验外层合同/绝对时次/证书。原计划t-12,t-6,t三帧不能被新F0只读t-6,t悄悄替代。|
|B12|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`earthdelta/bridge/stormer_bridge.py:848–881`；`earthdelta/bridge/stormer_bridge.py:889–1057`|controlled_rollout已有模型级进程内互斥与标志检查；普通forward_validation不进入同一锁。当前严格串行路线可用，不应宣称通用并发安全，也不需要先扩建并发框架。|
|B13|COMPLETE|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/s0_gate.py:2407–2470`|原代码缺陷已对症修复，不能继续照抄旧finding；整个S0真实发布成功仍未成立。|
|B14|COMPLETE|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/s0_gate.py:2470–2515`|format_finite及generate_report真实调用已存在；报告可生成不是官方parity或科学目标通过。|
|B15|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/r2_admission_gate.py:220–250`；`earthdelta/selection.py:365–395`；`scripts/r2_fs_bank_train.py:545–581`|registry值域验证和选择入口存在；旧流程没有合格共同Fs/动态checkpoint的完整消费链。近零候选映射问题只在相应值域触发，固定0.25不是它的实际反例。|
|ENV|PARTIAL|PARTIALLY_ACHIEVED|EXECUTION_CONFIRMED|`scripts/r4_prepare_environment.py:1`；`scripts/r4_run_strict_fp32.py:1`|历史GPU可运行已支持；最新official环境细节不能由旧SDPA训练任务或自述替代。环境可运行不等于S0通过。|
|S0|BLOCKED|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/s0_gate.py:2334–2382`；`earthdelta/bridge/stormer_bridge.py:236–370`；`scripts/r4_s0_parity_diagnose.py:112–148`|真实身份比较/raw-normalized绑定/多步入口已实现，旧Stage1A自比较不能再照搬。最新多步记录FAIL；input inverse源精度是假设，未取得完整GPU trace不能宣称唯一根因。|
|FS_FIT|BLOCKED|PARTIALLY_ACHIEVED|EXECUTION_CONFIRMED|`earthdelta/static_adapter.py:891–994`|真实训练发生，但fit仍仅以有限loss/grad及updates判eligible。历史8/8退化未关闭；同面板最终Fs/F0亦恶化，不能用合并数值一致性替代训练质量。|
|COMMON_FS_BANK|BLOCKED|PARTIALLY_ACHIEVED|EXECUTION_CONFIRMED|`scripts/r2_fs_bank_train.py:298–349`；`scripts/r2_fs_bank_train.py:520–624`；`earthdelta/bank_training.py:1297–1355`|K4/rank4模块与GPU更新存在，但不是共同参考的可重载合格bank；profile会在训练后的同bank上optimizer.step，训练记录与后续参数可能错位。|
|STAGE4_CACHE|PARTIAL|NOT_ESTABLISHED|STATIC_CONFIRMED|`scripts/r4_value_pilot.py:319–446`|新shared-F0 cache/merge代码存在但正式产物未运行，且动作与参考协议不同；不能认作原Fs五候选缓存完成。|
|STAGE4_POLICY_DEV|NOT_STARTED|NOT_ESTABLISHED|MISSING|`scripts/r4_value_pilot.py:449–493`；`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:259–274`|已有head原语不等于fit/prediction-freeze/evaluation CLI；未见拟合权重、OOF predictions、结果表与成本。|
|DEFERRED_HOLDOUT|NOT_APPLICABLE|NOT_ESTABLISHED|STATIC_CONFIRMED|`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:235–237`|未做这些延期项目本身不违反原计划。也不能反向宣称已证明holdout无泄漏；本包仍不解锁确认集。|
|GATE_CHAIN|PARTIAL|PARTIALLY_ACHIEVED|STATIC_CONFIRMED|`scripts/r2_fs_bank_train.py:963–1005`；`scripts/s0_gate.py:1865–1946`；`scripts/r4_value_pilot.py:59–77`|旧runner仅在末尾all(results)，中途继续调用；新路线停住不修复旧入口。source hash记录未完整作为消费者准入合同核验。|
|F0_AMENDMENT|PARTIAL|NOT_ESTABLISHED|STATIC_CONFIRMED|`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:229–239`；`scripts/r4_value_pilot.py:109–145`；`scripts/r4_freeze_sprint.py:92–123`|shared-F0是公开另案，不应机械套用旧worker-Fs缺陷；但F0参考、9/13候选、两帧历史不是原计划完成，原Fs阻塞不能被它清账。|
|SCIENCE|NOT_STARTED|NOT_ESTABLISHED|MISSING|`plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md:278–288`；`plans/novelty_review_20260922/SPRINT_8H_DECISION.json:1`|没有合格共同参考bank上的完整合法策略结果；novelty与weather utility均INCONCLUSIVE。此处不重新评估文献价值。|

## 先后关系

FP-00证据→FP-01 S0→FP-02真实入口→FP-03唯一合格Fs→FP-04共同专家库→FP-05完整缓存/廉价OOF表→FP-06决策。

B13/B14无需为关闭同一历史问题重新造框架；未启用的legacy诊断、near-zero候选和共享并发问题按实际路径处理。Fs质量和S0是硬前置，不能因其他任务已完成而越过。
