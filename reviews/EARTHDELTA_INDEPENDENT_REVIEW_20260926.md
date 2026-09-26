# EarthDelta 独立复查与后续研究建议

**复查日期：** 2026-09-26
**复查仓库：** `/mnt/afs/260010168/EarthDelta`
**复查时 HEAD：** `8ab914b877fcf2ace8c0e8da28cd8b0c47f22714`
**复查分支：** `audit/round2-review-20260921`
**复查依据：** v7 计划包、当前工作树、历史与最新执行 receipt、当前代码和 targeted tests

这份文档是对当前项目状态的独立审计。它区分四个层次：工程代码是否正确、实验是否真的执行、方法是否产生有效结果、结果是否足以支持 novelty/value。测试通过或脚本正常退出，不等价于科学假设成立。

## 结论摘要

当前项目处于“工程审计接近闭合、科学验证尚未闭合”的状态。

在冻结的 2019 policy-DEV 数据上，当前动态路由器 M3 没有相对于静态基线 M1 的 24h 可检测收益，FP06 规则给出 `PIVOT_STATIC`。这表示当前配置下应优先保留 M1/static route；它不表示动态编辑路线整体失败，也不表示 novelty 已经建立。

GitHub HEAD 仍然落后于当前本地 dirty worktree 中的安全修复。当前正式 FP06 输入还绑定旧 commit，WBX 当前 Tier-A receipt 缺失，且没有合法 fresh/confirm split。因此不能把当前结果写成“最新 GitHub 代码已经严格执行全部计划并完成样本外验证”。

## 实际代码和仓库状态

- `origin/audit/round2-review-20260921` 与复查时 HEAD 相同。
- 工作树存在 10 个已修改文件以及多个未跟踪脚本、测试和计划包；这些修改没有提交到 GitHub。
- 当前本地代码是 `HEAD + dirty overlay`，不是一个可由 GitHub HEAD 直接复现的快照。
- v7 计划实际采用的 authoritative ZIP 为：
  `plans/plans_v7_0925/EarthDelta_Codex_Followup_Plan_20260925_v2(1).zip`
  ，SHA-256：`59af3cee41735744ce84fdb6d19b749e03f749f81f1de72e61ea271438f1bbf7`。
  同目录另一个同前缀 ZIP 的 SHA 不同，后续必须固定上述文件。
- v7 执行没有启动新的 GPU、API、Stage4 或 confirm 作业，也没有读取新的天气/模型数组。

GitHub HEAD 中仍可见的关键差异包括：

1. `scripts/r5_audit_runner.py` 的旧版本把缺失的 `evidence_valid` 默认成有效值，并没有当前 dirty 版本中的严格布尔和证据检查。
2. `earthdelta/wbx/evaluate.py` 的旧版本只检查 issue year，没有完整检查 lead 后的 valid year，也允许调用方覆盖 authorized years。
3. 当前 dirty worktree 中的这些修复没有形成 release commit，因此测试本地工作树不能证明 GitHub 版本安全。

## 按 v7 计划的执行状态

| 计划分支 | 复查结论 |
|---|---|
| FP-00/01/03/04/05a/05b、WBX-01A/B | 历史材料已只读盘点；不能作为当前代码证明 |
| FP-06a-R2 | 工作区、receipt 和 source overlay 已封存；历史 S0 receipt 中 `stormer_bridge.py` 与当前工作树不同，当前 source 资格未闭合 |
| FP-06b-R2 | legacy 500-update 资产能被拒绝；H32 资产被观察但 Stage4 readiness 保持 false，符合安全停止条件 |
| FP-06c-R2 | 类型检查和数值 CI 检查已修复，但 source/hash/threshold 绑定仍不完整，应重新标为部分完成 |
| FP-06d-R2 | 冻结标量重算与历史 FP05b 一致，历史决策为 `PIVOT_STATIC`；输入仍绑定旧 commit `45edd51`，不是当前 HEAD |
| WBX-02d | 历史 Tier-B receipt 的 312 条 anchored rows 加 21 条 informational rows 已审计；当前 HEAD 的 Tier-A receipt 没有运行 |
| FP-07d | 只读机制诊断完成；只解释冻结 DEV，不能外推到 confirm |
| FP-08c | 2022 仅完成日志/元数据审计，保持 BLOCKED；2021 metadata 也没有内容完整性或 split assignment 证明 |
| FP-09 | 决策框架和下一实验草案已生成；human decision 仍缺失 |
| FP-10 | handoff、manifest、保护快照和日志已封存 |

因此，执行边界总体遵守了计划，但若把“计划验收”理解为当前代码和科学结论都已经闭合，答案是否定的。

## 真实实验结果

### 冻结 DEV 诊断

- 112 issues；23 个有效 7-day blocks；10,000 次 paired block bootstrap。
- M3 相对 M1 有 39 次 action switches。
- M3 vs Fs，24h：点估计 `0.00261143`，即 `0.2611%`；95% CI 为约 `[0.2389%, 0.2844%]`。
- 原计划的 minimum effect 为 `0.0034`，即 `0.34%`。M3 没有达到该阈值。
- M3 vs M1，24h：约 `-0.000302%`；95% CI `[-0.0204%, 0.0185%]`，跨 0。
- direct expert headroom vs M1：`0.05356%`，95% CI `[0.04184%, 0.06644%]`。
- routing selection error：`0.05386%`，95% CI `[0.04008%, 0.06928%]`。
- 72h M3 vs Fs：`0.43895%`，95% CI `[0.3805%, 0.4967%]`，harm guard 通过；这不是主要 24h 成功标准。

direct headroom 与 routing selection error 几乎完全抵消，解释了 M3 相对 M1 没有收益。更重要的是，即使完美选择当前候选专家，额外 headroom 的上界仍显著低于 `0.34%`。如果目标阈值保持不变，继续增加 selector 复杂度不是最有希望的路线。

### selector sensitivity

固定 threshold grid 在同一 DEV 上重放时，所有 24h 点估计都不为正，置信区间跨 0。threshold 为 0 时恰好复现 M3 的 39 次切换。该结果是描述性 DEV 诊断，不是 fresh split 或 novelty 证据。

### WBX 和未来年份

- 历史 Tier-B receipt：312 条 anchored rows、21 条 informational rows，312 项 family/lead 对账通过。
- 当前 Tier-A receipt：`NOT_RUN_AS_CLEAN_PINNED_RECEIPT`。
- 2021 metadata 显示 shape `[1460, 69, 128, 256]`，但内容未验证，也没有 confirm/fresh assignment。
- 2022 final store metadata 为空；日志显示 network、object/manifest、schema/time 和 verification 多类失败。
- 这些信息不能支持 fresh split、confirm 或模型 skill 结论。

## 关键实现和审计问题

### P0：GitHub 版本和实际修复版本不一致

当前 dirty 版本已经加入 fail-closed runner、年份边界修复和诊断输入绑定，但 GitHub HEAD 没有这些修复。历史 S0 receipt 的 canonical source 中，`stormer_bridge.py` 与当前工作树不同；FP06 scalar recompute 和正式 decision 仍绑定 `45edd51`。

修复要求：形成干净 release commit；为所有 canonical source 保存 commit 和文件 hash；区分 `historical_replay` 与 `current_release`；在 release commit 上重新运行最小 receipt。

### P0：WBX `evaluate_single_chunk` 仍可能绕过 valid-time 边界

当前函数的 `valid_times` 仍是可选参数，而数组在调用函数前已经加载。调用者可将未授权年份的 truth 数组传入，同时只提供 2020 issue time，或者省略 `valid_times`。

修复要求：强制 valid-time/content certificate；校验长度、顺序、shape 和内容 hash；没有证明的 single-chunk helper 改为内部接口；增加遗漏 valid-time 的负向测试。

### P1：FP06 source/threshold/input manifest 绑定不完整

当前 runner 检查 source manifest 是否为 dict、input manifest 是否为真值，并检查 CI 与内嵌 threshold 的一致性，但没有验证 source hash、当前 HEAD、input 文件 hash、历史/当前角色，也没有强制 CLI threshold 等于 bundle threshold。

临时 mutation 检查表明：将 source head 改成 `deadbeef` 仍可返回 `PIVOT_STATIC`；用不同 CLI threshold 也仍然成功退出并产生不一致 receipt。

修复要求：递归重算 manifest hash；current release 强制绑定当前 commit；threshold 严格相等；所有 mutation 必须非零退出。

### P1：erratum gate 只保护 24h

`r5_score_erratum.py` 会计算 6h/24h/72h，但总的 `gate_passed` 只检查 24h 与旧结果完全一致。6h/72h 的 corrected rows 仍需要独立完整性、有限性和 source hash gate。

### P1：诊断 consumer 依赖上游 manifest 的完整性声明

`r5_diagnose.py` 已验证顶层和 child hashes，但没有在入口直接重验完整的 `112 × 5 × 3` 矩阵、candidate 唯一性和 row completeness。当前 cache 的 1680 条记录全部 PASS，没有发现实际损坏；该问题属于防御式合同缺口。

### P2：科学覆盖不足

当前只有冻结 DEV，没有合法 fresh/confirm split。112 issues 和 23 blocks 的结果不能支持跨年份、跨状态或部署后泛化。没有 fresh split 之前，不能把 `PIVOT_STATIC` 写成动态编辑路线的普遍否定。

## 当前结论的可信边界

| 层次 | 可以说什么 | 不能说什么 |
|---|---|---|
| 工程代码 | 当前 dirty worktree 的 targeted contracts 大多通过；cache、OOF 和只读诊断链有较强约束 | GitHub HEAD 已包含全部修复；当前 release 与历史 GPU receipt 完全一致 |
| 实验执行 | 冻结 DEV CPU 诊断真实执行，且没有越过 confirm/GPU/API 边界 | fresh split、confirm、当前 Tier-A benchmark 已执行 |
| 方法效果 | 当前候选集上 M3 没有相对 M1 的 24h 可检测增益；M1 更适合作为当前默认 | 动态路线普遍无效；72h 安全指标证明动态方法有价值 |
| novelty/value | 研究问题仍可检验 | novelty 已建立；结果已足够发表或支持样本外结论 |

指标实现上，`policy_oof.py` 的 paired bootstrap 使用完整 7-day blocks，效果分母是 M0；M0 在协议中就是 no-edit Fs，因此数值内部一致，但未来 receipt 应显式写出该 estimand，避免把 M0、Fs 和 F0 混淆。

## 下一阶段计划

### 1. Release/provenance closure

**目标：** 让代码、receipt、source hash 和 GitHub 指向同一个不可变版本。

**成功标准：** clean release commit；所有 source/input hash 可重新计算；历史 replay/current release 角色明确；修改 head、threshold 或 input path 后非零退出。

**失败标准：** 继续把 FP06 标记为 historical-only，不对 current release 做科学宣称。

### 2. WBX exposure gate closure

**目标：** 关闭已加载数组缺少合法 valid-time/content 证明的入口。

**成功标准：** unauthorized valid year 在 loader 前拒绝；伪造坐标和 omitted valid-times 测试失败；若当前 Tier-A receipt 缺失则继续明确标记 BLOCKED。

**失败标准：** WBX 只能保留为历史 Tier-B 证据。

### 3. 合法 fresh split 上的一次 selector-only intervention

**前提：** fresh split 必须有独立 assignment/access certificate；不能把已曝光的 2020H2 充当 confirm；candidate set、预算、feature 和 threshold 在读取 label 前冻结；不新增 expert、不重新训练 bank。

**成功标准：** M3 vs M1 的 24h CI 下界大于 0；M3 vs Fs 达到预注册的 `0.34%` threshold；72h harm guard 通过。

**失败标准：** CI 仍跨 0、效果仍低于阈值，或无法取得合法 fresh split。失败后保留 M1，停止当前 dynamic-routing 线。

### 4. 只有 fresh split 出现正向证据后才使用 GPU

固定 candidate bank、预算、指标和多个预注册 seed，对比 M1、M3 和一个简单 utility baseline。若第 3 步没有正向证据，不应启动新的 bank 训练、Stage4、API inference 或大规模 H100 实验。

## 暂时不值得继续的实验

- 在同一个 DEV 上继续搜索更多 selector threshold、HPO 或 feature 组合；
- 继续增加 dynamic routing 模块数量；
- 没有 fresh split 就训练新的 expert bank；
- 将 2021 metadata shape 当作 confirm 数据；
- 读取或下载空的 2022 store 以补齐实验；
- 反复重跑历史 S0，只为了替换缺失的 current receipt；
- 把历史 Tier-B WBX receipt 写成当前 Tier-A benchmark 成功；
- 用更多单元测试替代样本外实验。

## 复查证据索引

主要执行目录在仓库外，未把大体积数据、模型或历史 receipt 复制进 Git：

- `/mnt/afs/260010168/earthdelta_v7_execution_20260925T171700Z/`
- `/mnt/afs/260010168/earthdelta_followup_execution_20260925T182900Z/`
- `/mnt/afs/260010168/earthdelta_followup_execution_20260925T184000Z/`
- `/mnt/afs/260010168/earthdelta_followup_execution_20260925T185716Z_metadata/`

关键文件包括 `TASK_RESULTS.json`、`HANDOFF.md`、`FINAL_MANIFEST.json`、`verified_fp06_inputs.json`、`fp06_v2/decision.json`、`wbx_v3/decision.json`、`year2022/findings.json` 和 `future_metadata_v2/metadata_only_audit.json`。

## 可交给 ChatGPT 的后续分析 Prompt

```text
请对 GitHub 仓库 EarthDelta 做一次独立、证据优先的研究与工程审查。

先读取当前 HEAD、分支、工作树状态和最近提交，不要默认 README 或旧报告仍然正确。重点阅读：

1. reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md
2. plans/plans_v7_0925/EarthDelta_Codex_Followup_Plan_20260925_v2(1).zip（若仓库中可见）
3. earthdelta/、scripts/、tests/ 中与 FP06、policy OOF、WBX、candidate cache、selector 和 source manifest 相关的实现

请把结论分成四层：

- 工程代码是否正确；
- 实验是否真的执行且数据边界是否合规；
- 方法是否产生有效效果；
- 结果是否足以支持 novelty/value。

必须独立验证，而不是复述 review：

- GitHub HEAD 是否包含当前 dirty worktree 的修复；
- FP06 runner 是否真正绑定 source/input manifest、threshold 和 current release；
- WBX 是否可能通过已加载数组、缺失 valid-time 或 authorized override 绕过年份边界；
- 24h/6h/72h 的 metric、分母、bootstrap 和 erratum gate 是否一致；
- 当前结果是否只是冻结 DEV，是否存在 fresh/confirm/Tier-A 证据；
- 当前最强基线是否仍是 M1/static。

请输出：

1. 已完成、部分完成、未完成和偏离计划的逐项表格；
2. 按 P0/P1/P2 排序的代码、数据、指标和科学风险；
3. 每个风险的最小修复和可验证成功/失败标准；
4. 下一阶段 3–5 个任务，每个任务写清目标、前置条件、成功标准、失败后的停止规则；
5. 明确哪些实验不值得继续；
6. 最终给出“当前可以相信什么、仍未证明什么、下一步是否应该保留 dynamic-routing”的明确结论。

不要把测试通过、脚本 rc=0、历史 Tier-B receipt 或 metadata shape 当成科学成功。不要读取或建议读取未授权的 confirm/holdout 数据。若需要新的实验，请先定义冻结协议、数据资格、指标、预算和停止规则。
```
