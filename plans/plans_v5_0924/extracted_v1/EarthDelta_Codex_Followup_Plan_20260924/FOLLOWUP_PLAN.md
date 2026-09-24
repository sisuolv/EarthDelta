# 下一迭代执行计划

本包新增任务从FP-05a开始。上一轮已完成项目不重新编号、重训或重复获取科学结果。
所有新CLI均是**待实现接口合同**，不是宣称当前仓库已存在；先按FP-05b实现并通过测试再使用。

## 统一执行要求

- 当前只允许FP-05a（CPU）。后续状态均TO_BE_RUN且未授权；依赖PASS不是自动预算许可。
- 包内 `evidence/PROPOSED_PROTOCOL.json` 是草案，不是可直接执行的冻结协议。null字段不得默认为通过。
- 必要小型原始job receipts和本地大权重hash在FP05a补；不强制上传多GB数据，不因为只有缺上传就重跑已过认证。
- 避免修改17个受保护源文件。新thin runner负责FP05真实入口，Fs/bank认证通过原接口消费。
- 当前原计划是fitted-Fs五候选；不能借用shared-F0脚本而改成另一reference或样本数/幅度域。
- 时间预算尚无新cache/profile实测，所有estimated_minutes=null。旧job耗时仅历史证据，不能当本轮保证。
- 预计两项forecast job（debug、完整cache），一次infra重试。全局上限4不是自动授权第四个数值试验。

## 新协议的冻结顺序

先重建历史曝光和候选/替换规则 → 只读取完整性 → 冻结合格候选及新代码/规则 → exposed debug盲profile →
按预先机械规则确定一次N和final roster → 提交正式cache → outer/inner OOF → 冻结预测 → 评分 → FP06。

这允许debug决定**成本能力**，不允许debug/正式dev的性能决定样本区间、阈值、候选或扩样。
所有“先后”保存UTC、file SHA、实际提交argv/hash；本地timestamp不是可信时间戳服务，仍要保留原始调度receipt。


## FP-05a · 证据补链、曝光重建与计划更正

**依赖：** FP-00, FP-01, FP-03, FP-04。**范围：** CPU only。修复FP05台账而不重训，不跑policy_dev forecast。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. 复用现有哈希/证书读取器；不伪造不存在的job_result。读取旧输出属于审查，不新增曝光试验。
2. 新增thin runner的FP05任务白名单与前驱验证；不要给legacy --stage all提供绕过功能。
3. 区分项目设计曝光、最终模型梯度输入、完整性-only读取；废弃模型的设计曝光不能删除。

**命令：**
```text
python "$PKG/tools/validate_package.py" --package "$PKG"
python "$PKG/tools/check_task_evidence.py" --repo "$REPO" --out "$RUN/entry_receipts.json"
TO_BE_IMPLEMENTED: python scripts/plan_followup_runner.py --task FP-05a --protocol "$RUN/protocol/fp05.json" --evidence-root "$RUN" --dry-run
```

**测试与验收：**
- 核对真实HEAD、17个pin、S0/Fs/bank JSON与本地权重hash；补小型job receipts或明确阻塞范围
- 从actual training/panel/read-set重建完整曝光，包含X1_N32/N64、失败/探索以及旧Round4；每段有来源
- 2020H2清洁confirm撤销；confirm=null时所有confirm入口拒绝；2019H2dev按完整support筛候选
- 输出更正后的FP05协议草案与需要协调者确认的72h损害、block/fold、HPO、数值容差字段；不以dev收益填值

**输出：** `entry_receipts.json`、`exposure_ledger.json`、`exposure_reconstruction_report.json`、`deviations/FP05_AMENDMENT.json`、`protocol/fp05.proposed.json`、`task_result_FP-05a.json`

**失败处理：** 新源码/资产与证书不符；遗漏X1_H2曝光或仍将整段2020H2标untouched；原始receipt与已提交摘要冲突；任何正式dev/cache/confirm执行要求。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。


## FP-05b · 新增split/消费者/OOF代码与完整预注册

**依赖：** FP-05a。**范围：** CPU implementation + synthetic regression。只加新文件，不改17个pin。补齐FP02余项及FP05方法合同。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. 新增earthdelta/split_freeze.py、candidate_cache.py、policy_oof.py；脚本plan_followup_runner/r4_candidate_cache/r4_cache_decide/r4_policy_oof。接口为本任务实施目标，不是假称已存在。
2. reuse low-level content validator and certified Fs/bank loaders; implement policy_dev consumer separately from Fs-only bank_fit certifier.
3. 先固定候选/替换/预算规则；绝不根据候选loss决定保留哪些天气。

**命令：**
```text
TO_BE_IMPLEMENTED: python -m pytest -q tests/test_split_freeze.py tests/test_plan_cache_evaluation.py tests/test_policy_oof_isolation.py
TO_BE_IMPLEMENTED: python scripts/plan_followup_runner.py --task FP-05b --protocol "$RUN/protocol/fp05.json" --evidence-root "$RUN" --dry-run
```

**测试与验收：**
- split在admit/launch/workerload/merge/fit/innerHPO/transforms/freeze/score所有实际入口生效
- registry精确5候选且唯一Fs，F0仅报告；policy_dev不得改标签为bank_fit来通过旧certifier
- outer fold F毒化测试只比较F的OOF预测；checksum早退与算法不读标签的测试分开
- 先选点/替换规则，再完整性扫描，最后final名单/一次N；不声称后写声明早于已完成读取
- protocol规定delta=.0034工程用途、same-issue F0对照、OOFstatic比较、72h损害准则、nested HPO、block/CI解释、blind规模规则、失败分母、预算；所有必填字段无null后冻结

**输出：** `protocol/fp05.json`、`protocol/fp05.sha256`、`protocol/PREREGISTRATION.md`、`splits/debug_admission.json`、`splits/dev_candidate_manifest.json`、`splits/exposure_certificate.json`、`tests/cpu.log`、`tests/cpu.xml`、`task_result_FP-05b.json`

**失败处理：** protected file changed；wrong/silent role reassignment；missing harm72/inference/numeric bound at freeze；validation/confirm labels consumed by feature/preprocess/fit；unsigned or late protocol modification。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。


## FP-05c · 8个exposed调试与盲吞吐定额

**依赖：** FP-05b。**范围：** 第一项新GPU工作。仅已曝光8例debug，验证当前bundle与新cache路径。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. 每条debug由两个worker重算；保存预算计费全部尝试。
2. CPU native reduction与GPU结果用预冻结数值界限校验；不能把该界限放宽S0或bank exact gate。

**命令：**
```text
TO_BE_IMPLEMENTED: python scripts/r4_candidate_cache.py --stage debug --protocol "$RUN/protocol/fp05.json" --out "$RUN/debug"
TO_BE_IMPLEMENTED: python scripts/r4_cache_decide.py --stage debug --protocol "$RUN/protocol/fp05.json" --cache "$RUN/debug" --out "$RUN/debug_decision.json"
```

**测试与验收：**
- 实际job_result含ID/rc/elapsed/source/env，4独立worker各自模型，无共享hook并发
- 先保存debug anchor map；仅已有历史own-group单元格要求精确一致，缺少golden不伪造
- 双worker重复状态精确相等；no-edit=Fs、后续Fs、5actions+F0用途正确
- 只用吞吐、存储、显存、可用时次机械选一次N<=128；保存final roster freeze在正式dev结果前
- debug不进入效应估计；固定32/64/128由机械规则一次选择而不是看到CI后扩样

**输出：** `jobs/C-J1/job_result.json`、`debug/anchor_coverage.json`、`debug/pairwise_repeat.json`、`debug/independent_reduction_check.json`、`debug/profile.json`、`debug_decision.json`、`protocol/sample_size_decision.json`、`splits/dev_final_roster.json`、`task_result_FP-05c.json`

**失败处理：** any golden/repeat/identity failure；model/probe labels used to choose dev scale；unexpected memory/code branch；unregistered retry or extra job。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。


## FP-05d · 一次性完整五候选缓存与F0背景

**依赖：** FP-05c。**范围：** GPU cache only on frozen dev roster。全部5候选和F0背景完整保留。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. 优先复用FP04 bank_panel_losses/controlled_rollout的相同native scoring，不换shared-F0硬编码pipeline。
2. 每个完整12-step轨迹同时收集6/24/72h，不按lead重复无必要rollout；F0计实际资源，不预填20%。

**命令：**
```text
TO_BE_IMPLEMENTED: python scripts/r4_candidate_cache.py --stage cache --protocol "$RUN/protocol/fp05.json" --roster "$RUN/splits/dev_final_roster.json" --shard "$SHARD" --shards 4 --out "$RUN/cache/shard_${SHARD}"
TO_BE_IMPLEMENTED: python scripts/r4_cache_decide.py --stage merge --protocol "$RUN/protocol/fp05.json" --cache "$RUN/cache" --out "$RUN/cache_manifest.json"
```

**测试与验收：**
- N*5*3候选标量单元完整无重复；F0 N*3报告数据分开、无候选ID
- 每issue绑定完整UTC支持、内容证书、Fs/bank/registry/Q/source/hash；scoring前重新校验
- 特征文件与sealed outcome隔离；日志不揭示dev loss或oracle；不按效果修补
- CPU独立重算满足冻结数值合同；矩阵missing/invalid/failed都在分母账本，不进行successful subset scoring

**输出：** `jobs/C-J2/job_result.json`、`cache_manifest.json`、`cache/coverage.json`、`cache/failure_ledger.json`、`cache/costs.json`、`cache/access_log.jsonl`、`task_result_FP-05d.json`

**失败处理：** source/weight mutation；missing/duplicate/unregistered rows；nonfinite output；confirm or exposure intersection；budget exhausted; preserve incomplete state, no partial scientific table。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。


## FP-05e · Purged nested OOF与不可变预测冻结

**依赖：** FP-05d。**范围：** CPU fitting：主ridge_direct_gain、次regime、OOF static。不得开启dual-head/E3。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. primary ridge不必PCA；若启用PCA需每fold训练内拟合。每候选直接预测gain；Fs/no-edit固定gain0。
2. bidirectional CV只支持回顾性setting，不能通过命名把future-labeled trainfold变成实时信息。

**命令：**
```text
TO_BE_IMPLEMENTED: python scripts/r4_policy_oof.py --stage fit-freeze --protocol "$RUN/protocol/fp05.json" --cache "$RUN/cache_manifest.json" --out "$RUN/predictions"
TO_BE_IMPLEMENTED: python scripts/r4_policy_oof.py --stage audit-isolation --protocol "$RUN/protocol/fp05.json" --cache "$RUN/cache_manifest.json" --predictions "$RUN/predictions" --out "$RUN/isolation.json"
```

**测试与验收：**
- fold只从UTC/元数据生成，完整support净化，train/val全部候选同fold；预留confirm拒绝
- scaler/PCA/kmeans/lambda选择仅inner/outer train；static也从outer train选含no-edit
- 对每个outer F，内部自洽标签毒化fixture改变F标签而F预测逐位不变；不错误要求其它fold预测不变
- 每个dev issue恰好一条合法OOF选择；NO EDIT tie规则预先固定；所有模型/超参/输入哈希/预测先封存
- reference、actual candidate outputs、oracle标签不输入selector；没有确认数据读取

**输出：** `predictions/prediction_freeze.json`、`predictions/oof_predictions.npz`、`predictions/fold_manifest.json`、`predictions/models_and_transforms.json`、`predictions/hpo_inner_results.json`、`isolation.json`、`tests/oof_test_log.txt`、`task_result_FP-05e.json`

**失败处理：** validation leakage or exposure；no nested HPO evidence；changing seed/grid after scored dev results；missing/duplicate OOF predictions；confirm access。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。


## FP-05f · 冻结后评分、配对区块统计与证据输出

**依赖：** FP-05e。**范围：** CPU evaluation。只产生冻结规则下的ABOVE/BELOW/STRADDLE与成本/损害表，不先写最终贡献。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. 0.0034仅工程目标且保持结果前设定；同批F0证据决定是否真实补回损害。
2. retrospective H2负结果限于当前bank/setting，不把季节迁移的失败外推为普遍方法无效。

**命令：**
```text
TO_BE_IMPLEMENTED: python scripts/r4_policy_oof.py --stage score --protocol "$RUN/protocol/fp05.json" --cache "$RUN/cache_manifest.json" --predictions "$RUN/predictions/prediction_freeze.json" --out "$RUN/evaluation"
```

**测试与验收：**
- oracle只是离线全候选上限；static选择不看对应fold核验；同一完整issue/lead分母
- 同时报告gain_vs_Fs、dynamic_increment_vs_OOF_static、net_vs_F0与72h损害；明确各分母/权重
- 固定OOF paired block bootstrap每次保持candidate/lead/method配对；日期block非候选/网格/seed；记录实际block数
- conditional-on-fitted-model CI与算法训练不确定性分开；依计划做block敏感性，不能挑显著结果
- 独立numpy/第二实现数值交叉验证；保存全失败和成本；没有confirm结果/novelty成功字段

**输出：** `evaluation/results.csv`、`evaluation/paired_effects.json`、`evaluation/bootstrap.json`、`evaluation/independent_recompute.json`、`evaluation/costs.json`、`evaluation/access_audit.json`、`task_result_FP-05f.json`

**失败处理：** no valid prediction freeze；missing harm72 rule；data-dependent delta/N/block change；candidate/issue omission；insufficient precision: emit INCONCLUSIVE and stop additional sampling。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。


## FP-06 · 独立开发迭代决策与交接

**依赖：** FP-05f。**范围：** 只在真实FP05结果后作STOP/CONTINUE/PIVOT；不是新确认实验。

**输入：** 上游指定evidence_outputs、被冻结协议及其身份；FP-05a从既有S0/Fs/bank证书和原始receipt开始。缺失不得猜测。

**实施：**
1. 若static足够，转static；若oracle不到工程目标，停当前bank；若ridge有效，才考虑有界下一轮，不自动开启dual-head。

**命令：**
```text
TO_BE_IMPLEMENTED: python scripts/plan_followup_runner.py --task FP-06 --protocol "$RUN/protocol/fp05.json" --evidence-root "$RUN" --dry-run
TO_BE_IMPLEMENTED: python scripts/r4_policy_oof.py --stage decision --protocol "$RUN/protocol/fp05.json" --results "$RUN/evaluation/paired_effects.json" --out "$RUN/decision.json"
```

**测试与验收：**
- 先完整性/损害/成本再oracle/static/legal增益，严格用预注册映射
- bank OOS有证据前不允许任何utility正面结论；只完成有用性开发也不等于response novelty
- 2020H2不作清洁confirm；未来确认窗口仍需独立立项，不自动下新job
- 负/不确定结果、协议偏差、未授权事项与剩余输入全部记录

**输出：** `decision.json`、`RESULTS.md`、`NEXT_HANDOFF.md`、`task_result_FP-06.json`

**失败处理：** prerequisite missing/fail；positive narrative unsupported by effects；confirm exposure disguised as clean；proposal to add modules to avoid fixed STOP。

**耗时：** `MISSING_NEEDS_MEASURED_PROFILE`；不得自动把null替换为经验分钟数。
