你是 EarthDelta 项目的独立计划执行审查者。这是对上一轮审查（`plans/plans_v5_0924/`，两份不完全相同的 zip，均产出于 `EarthDelta_Codex_Followup_Plan_20260924.zip`）的**后续一轮**。上一轮审查判定 `FP05_PLAN.md` 里"2020 下半年是完全未触碰的 confirm 预留区"这个判断有误（FP-03 的 X1_N64 探索性实验其实用过 2020 下半年数据），并要求先做一次只读曝光台账审计（FP-05a）再继续。本轮请你判断：(1) FP-05a 的审计、FP-05b 的候选缓存+样本外策略评估、以及新增的 WeatherBench-X 评测接入这三项工作是否被真正、诚实地执行；(2) 我方给出的核心科学结论（动态专家库未证明样本外收益）是否站得住、有没有被我方自己的验证掩盖的问题；(3) 我方新写的 `FOLLOWUP_PLAN_20260924_ROUND2.md` 是否合理、需不需要在执行前修改；(4) 在此基础上生成新一轮供 Codex 执行的后续计划 zip。

## 仓库与证据位置

仓库：`https://github.com/sisuolv/EarthDelta`，分支 `audit/round2-review-20260921`，当前 HEAD `45edd51`。上一轮审查时的基线是 `e0136d4`；本轮新增的相关提交包括 `0809365`（FP-05 计划 + 上一轮审查 prompt）、`4289985`（README 更新 + FP-05a 计划）、`45edd51`（FP-05a 完整产出 + FP-05b 完整产出 + WeatherBench-X 集成，本轮的主要审查对象）。GPU 权重、数据集和完整 `artifacts/` 目录按 `.gitignore` 排除，没有推上 GitHub；如需具体某个 job 的原始权重/日志核实，告诉我路径，我单独提供 hash 或精简摘要。

请先完整阅读，按此顺序：

1. `plans/plans_v5_0924/extracted_v2/EarthDelta_Codex_Followup_Plan_20260924/`——上一轮的验收基准，尤其是 `FULL_REVIEW.md`、`FOLLOWUP_PLAN.md`、`STOP_CONDITIONS.md`。
2. `plans/plan_v4_0923/FP05A_PLAN.md`——FP-05a 的设计（本轮已执行完毕）。
3. `plans/plan_v4_0923/run_20260924T093810Z_fp05a/`——FP-05a 实际产出：`entry/exposure_ledger.json`、`entry/exposure_reconstruction_report.json`、`entry/core_source_pin_check_before.json`/`_after.json`、`entry/certificate_recompute.json`、`protocol/fp05_protocol.DRAFT.json`、`task_result_FP-05a.json`。
4. `plans/plan_v4_0923/run_20260924T104725Z_fp05b/`——FP-05b 完整证据链，这是本轮审查重点，请特别关注以下文件（不要只看汇总，要核对原始数字）：
   - `admission/policy_dev_selection_STOP.json` + `admission/policy_dev_selection_coordinator_resolution.json`：一次真实触发的预注册 STOP（16 个目标点找不到候选，超过预注册的 8 个丢弃上限），协调者决定按原规则接受结果（N=112，2 个分层零候选）而不是切换到三个备选规则中的任何一个。请独立评估这个决定本身是否合理，以及是否存在"挑对结果最有利的规则"这种变相调参的嫌疑（即使我方记录的理由是"两个分层结构性没有候选，任何同层规则都填不上"）。
   - `tests/mutation_checks.json`：12 项人为注入的 bug，是否真的全部被测试"杀死"（KILLED），还是有形式主义走过场的嫌疑。
   - `protocol/cache_protocol_v1.json` + `protocol/PREREGISTRATION.md`：协议是否真的在任何 GPU 作业提交前冻结（核对 `.sha256` 时间戳和 job 提交时间的先后关系）。
   - `decisions/CJ1_debug_decision.json`、`analysis/cj1_independent_verify.json`、`analysis/cj2_independent_verify_noloss.json`：debug 批次的 anchor 数字是否真的和 FP-04 已有的 `bank_panels_expert{k}.json` 逐位相等（请自己独立抽查至少 2-3 条，不要只信任我方脚本的判定）。
   - `policies/folds.json`、`policies/cp3/`（8 次中毒验证标签重跑）、`analysis/phasec_independent_recompute.json`：净化交叉验证和"没有偷看验证标签"的证据是否扎实。
   - `evaluation/dev_results.csv`、`evaluation/paired_block_bootstrap.json`：**核心结论的原始数字**——M3（ridge，主策略）vs Fs 在 24h 的点估计/CI、oracle vs Fs 的点估计/CI、H_Fs、以及 **M3 vs M1（"智能路由" vs "永远选同一个专家"）几乎为零**这个发现。请独立核实这些数字，并给出你自己对"这个结果说明了什么"的判断，不要只复述我方的解读。
5. `artifacts/wbx_smoke/20260924T135007Z_tierB/tierb_summary.json`（若 GitHub 上因为路径不在 `artifacts/` 排除规则里而不可见，告诉我，我单独提供）——Tier B 的 104×3 逐位对账表，请抽查至少 2 条和 FP-03/FP-04 原始记录（`plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/certify/fs/panel_initial_final.json` 等）核对。
6. `README.md` 的 Status & roadmap 一节——检查这次的状态描述是否如实（尤其是"BELOW/STRADDLE"这个负面结论有没有被淡化或包装得比实际更乐观）。
7. `plans/plan_v4_0923/FOLLOWUP_PLAN_20260924_ROUND2.md`——本轮新写的下一步计划草案，尚未执行。

不得只根据计划文字、README 或本 prompt 的自述作结论，必须核对实际 JSON、实际 log、实际 job 记录。

## 逐项核查：FP-05a / FP-05b / WeatherBench-X 是否真正、诚实地执行

对每一项区分四个层次，不能把前两层自动升级为后两层：

1. 代码/文档存在；
2. CPU/单元测试通过；
3. 真实 GPU job 执行并产出证据（要有真实 job ID、`job_result.json`、`returncode`、`elapsed_seconds`）；
4. 科学问题得到了真实回答（不管答案是正是负）。

具体检查：

- **FP-05a**：曝光台账重建是否真的纠正了原计划的 D1 错误（2020 下半年不能当 confirm），台账里 2019/2020 两年的曝光区间划分是否和 FP-03/FP-04 的真实 job 记录吻合。
- **FP-05b 的数据隔离闸门**（`earthdelta/split_freeze.py`）：是否真的在每个消费入口都生效（准入脚本、`r4_candidate_cache.py` 的 cache/merge 阶段、`r4_policy_oof.py` 的 folds/fit-predict/score），还是只在部分路径生效、留了绕过口子。`tests/mutation_checks.json` 里 `scripts/r4_candidate_cache.py`/`scripts/r4_policy_oof.py` 的几条注入 bug 是核查这个的关键证据。
- **FP-05b 的核心结论本身**：主策略（M3）BELOW、oracle STRADDLE、H_Fs 为正（Fs 自己有真实样本外伤害）、M3≈M1（路由没有增量价值）——这四条放在一起，你的独立判断是什么？是否支持我方在 `FOLLOWUP_PLAN_20260924_ROUND2.md` 里给出的"路由机制本身可能没有捕捉到真实信号，而不只是样本量不够"这个诊断假设？有没有其他更可能的解释（比如季节错配、K=4 本身太小、特征工程不够、delta_min 本身定得是否合理）？
- **WeatherBench-X Tier A/B**：`earthdelta/wbx/` 是否真的复用了官方 `weatherbenchX`/`weatherbench2` 代码而不是重新实现了一遍 RMSE/regrid（检查 `earthdelta/wbx/_vendor.py` 和 `evaluate.py` 的实际 import）；Tier B 的"和 FP-03/FP-04 逐位对账"是否真实、有没有可能是巧合或选择性展示；年份闸门（拒绝 2021/2022/confirm 数据）是否真的在代码层面强制，不只是文档承诺。
- **2022 数据校验失败**（`checkpoints/pull_2022.log`，`status: verification_failed`）：这是本轮新发现、尚未根因的问题，请指出这对 `FOLLOWUP_PLAN_20260924_ROUND2.md` 里"2022 优先、2021 备选"这个此前的 confirm 年份建议有什么影响，是否应该在阶段排序上更早处理。

## 逐项核查：`FOLLOWUP_PLAN_20260924_ROUND2.md` 是否合理

该文件末尾已经列出我方自己认为需要审的 5 个问题（阶段排序、FP-07 诊断设计是否够、FP-07 有没有事实上的偷看风险、阶段 5 决策门时机、是否有遗漏的优先事项），请逐条给出你自己独立的评估，而不是照搬我方给出的理由。特别关注：

1. FP-07（诊断性复盘）声称"只产出诊断信息，不能倒回去改已经冻结的 FP-05b 结果"——这条防线在实际执行层面是否可信？有没有具体的、可执行的隔离手段（比如新开一个只读 run 目录、不允许修改 `evaluation/` 下任何已有文件）需要加进协议里，而不是只在文字上承诺？
2. FP-06（机械套用预注册判定规则）是否真的是"零判断"的——预注册的 FP06-RULES-v1 规则文本本身是否已经完整、没有需要临场解释的模糊地带？
3. 阶段 5 的"重新设计 vs 收尾 vs confirm"决策门，是否应该现在就先给出一个初步的框架/判断标准（即使不执行），而不是完全留白到 FP-07 做完之后？

## 证据标注规则（沿用上一轮）

每条 finding 必须给出精确 `file:line`、commit 或 artifact 路径，并标记：

- `STATIC_CONFIRMED`：直接逐行核实；
- `EXECUTION_CONFIRMED`：真实测试或 GPU 记录核实；
- `PLAUSIBLE_UNVERIFIED`：合理但证据不足；
- `MISSING`：所需证据不存在。

## 输出要求

先输出结构化 JSON（字段至少包含：`overall_verdict`、`fp05a_verdict`、`fp05b_execution_verdict`、`fp05b_headline_result_verdict`、`fp05b_data_isolation_verdict`、`wbx_tierA_verdict`、`wbx_tierB_verdict`、`year2022_finding_severity`、`followup_plan_soundness_verdict`、`blocking_items`），再输出 Markdown 分析，至少包含：

1. `overall_verdict`，只能是 `PLAN_EXECUTED`、`PARTIALLY_EXECUTED`、`BLOCKED`、`FAILED_PLAN_COMPLIANCE` 或 `INSUFFICIENT_EVIDENCE`；
2. FP-05a / FP-05b / WeatherBench-X 三项的对照表：计划要求、实际实现、测试、真实 GPU 执行、目标状态、缺口；
3. 对 FP-05b 核心结论（BELOW/STRADDLE/H_Fs 为正/M3≈M1）的独立科学评估，包括你认为最可能的解释和最值得优先排查的方向；
4. `FOLLOWUP_PLAN_20260924_ROUND2.md` 的独立评审意见，逐条回答文件末尾的 5 个问题，指出你认为需要修改的地方；
5. 按优先级排列所有阻塞项和最小后续行动；
6. 明确列出当前仍然不能对外宣称的结论（尤其是：动态专家库目前没有证明任何样本外收益；FP-06 正式判定尚未做出；2022 数据是否可用尚未确认）。

最终必须严格区分：代码实现完成、测试通过、GPU 执行、独立数字核对通过、diagnostic 分析完成，以及"动态编辑这条技术路线是否值得继续投入"这个更大的问题得到了回答。证据不足时写 `INCONCLUSIVE`，不要用推测填补缺失证据。

## 分析结束后的强制交付物

分析完成后，请根据你的审查结果，生成一个供 Codex 继续执行的后续计划包，并打包为：

`EarthDelta_Codex_Followup_Plan_20260925.zip`

zip 内必须有以下结构（和上一轮一致）：

```text
EarthDelta_Codex_Followup_Plan_20260925/
├── START_HERE_FOR_CODEX.md
├── FOLLOWUP_PLAN.md
├── EXECUTION_DAG.md
├── STOP_CONDITIONS.md
├── CODEX_TASKS.json
├── ARTIFACT_CONTRACT.md
├── TEST_PLAN.md
├── REPRODUCIBILITY.md
├── CURRENT_GAP_ANALYSIS.md
├── evidence/
│   ├── PLAN_AUDIT_RESULT.json
│   ├── EVIDENCE_MANIFEST.json
│   └── REQUIRED_INPUTS.md
└── prompts/
    └── CODEX_EXECUTION_PROMPT.md
```

这些文件必须满足：

1. `CURRENT_GAP_ANALYSIS.md` 逐项引用本次审查的结论、`file:line`、commit 或 artifact，明确 FP-05a/FP-05b/WeatherBench-X 里哪些目标已完成、部分完成、阻塞或尚未开始，`FOLLOWUP_PLAN_20260924_ROUND2.md` 哪些部分可以直接采用、哪些需要修改；
2. `CODEX_TASKS.json` 的任务编号请延续 `FP-` 前缀（`FP-06`/`FP-07`/`FP-08`/`WBX-02` 起步，可以进一步拆分子任务），不要重新从 `FP-00` 编号——FP-00~FP-05b、WeatherBench-X Tier A/B 已完成的部分请在任务列表里标注为 `status: DONE`、附上对应的真实证据路径，而不是重新列为待执行；
3. `FOLLOWUP_PLAN.md` 给出有顺序的任务计划，包含每项任务的输入、命令、预期输出、成功标准、失败处理和预计耗时（不虚构耗时，标 `MISSING_NEEDS_MEASURED_PROFILE` 也可以）；如果你认为 `FOLLOWUP_PLAN_20260924_ROUND2.md` 的阶段排序需要调整，请在这里给出你修改后的版本并说明理由；
4. `EXECUTION_DAG.md` 明确任务依赖关系，禁止在前置 gate 失败时启动 downstream scientific run，也禁止 FP-07 的诊断发现被用来倒回修改 FP-05b 已冻结的结果；
5. `STOP_CONDITIONS.md` 复制所有必须保持的阈值、准入条件、冻结 split 和禁止放宽的门槛，明确写出"FP-05b 的 BELOW/STRADDLE 结论不得因为 FP-07 诊断结果而被重新解读为达标"这类禁止事项；
6. `CODEX_TASKS.json` 每项至少包含 `id`、`priority`、`depends_on`、`scope`、`commands`、`acceptance_criteria`、`evidence_outputs`、`stop_if`、`status`、`estimated_minutes`；
7. `ARTIFACT_CONTRACT.md` 规定每个实验必须保存的 JSON、日志、hash、job ID 和指标字段；
8. `TEST_PLAN.md` 把 CPU 测试、GPU gate、回归测试和科学结果验证分开；
9. `CODEX_EXECUTION_PROMPT.md` 必须是一段可以直接复制给 Codex 的执行指令，要求它先读取 `START_HERE_FOR_CODEX.md`，再按 DAG 顺序工作，持续保存 evidence，并在 STOP 条件触发时停止；
10. 如果你认为阶段 5（重新设计 vs 收尾 vs confirm）需要提前给出判断框架，请把框架写进 `FOLLOWUP_PLAN.md`，但明确标注"框架"而不是"决定"，不得让 Codex 替代人工做这个战略决策；
11. zip 中不得伪造尚未运行的结果。已完成的部分标 `OBSERVED`/`DONE`，未来结果标 `TO_BE_RUN`，缺失证据标 `MISSING`。

最后请同时返回：

- 完整审查报告；
- zip 文件名；
- zip 内文件清单；
- zip 的 SHA-256；
- Codex 执行该计划前必须补充上传的文件清单（例如具体某个 job 的原始日志、某个权重文件的 hash）。
