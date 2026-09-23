你是 EarthDelta 项目的独立计划执行审查者。请判断当前最新代码是否真正执行了
`plans_v3_0922` 中的原计划，以及计划目标是否已经实现。

请先完整阅读本包中的：

1. `source_plan/` 下全部文件，尤其是 `NEXT_STEPS_10H_PLAN.md`、`CLAUDE_10H_PROMPT.md`
   和三个 Stage 1A review 文件；
2. `CURRENT_STATUS.md`、`EVIDENCE_MANIFEST.md`、`reference_reports/` 下的报告；
3. 我随后提供的私有仓库代码、git diff、测试日志和 GPU artifacts。

当前实现提交是 `a3596e6b804e9d23b72d1247b08c47129d6b50b1`，比较基线是
`d749a1c62521226df857587e08f7d067b0f15355`。不得只根据计划、README 或
`CURRENT_STATUS.md` 的自述作结论，必须核对实际代码和实际证据。

请逐项检查：

- B05/B06/B08/B09/B12/B13/B14/B15 是否实现并被真实入口调用；
- Stage 3b 的 static adapter、Fs fit、K=4/rank=4 expert bank、qualification、registry；
- S0 官方 Stormer parity、identity、normalization、checkpoint binding 和 multistep rollout；
- Stage 4 candidate cache、head/policy fitting、holdout evaluation 是否实际执行；
- 计划要求的 STOP 条件、准入门、冻结 split、证据绑定是否真正生效；
- Round 4 的 Fs-fit divergence finding 是否解决，尤其是 formal 模式下部分专家在同一配对样本上
  出现 8/8 loss deterioration 的问题；
- novelty 和 weather utility 是否有真实、可复核的实验结果支持。

对每项区分四个层次：

1. 代码存在；
2. CPU/单元测试通过；
3. 真实 GPU job 执行并产出证据；
4. 科学目标已经被实验支持。

不能把前两个层次自动升级为后两个层次。不能把准备好的 argv、job manifest、测试定义或报告中的计划表当成执行证据。

特别核查当前已知事实：H100 环境本身可以运行，最新修复后 S0 一步 parity 通过，但 4-step 和 12-step
parity 仍超过冻结的 `1e-5` 容差；downstream expert/cache/evaluation 在该 gate 失败后没有正式启动。
如果你从新 artifacts 中发现这些事实已改变，必须给出 job ID、artifact 路径和数值证据。

每个 finding 必须给出精确 `file:line`、commit 或 artifact 证据，并标记：

- `STATIC_CONFIRMED`：直接逐行核实；
- `EXECUTION_CONFIRMED`：真实测试或 GPU 记录核实；
- `PLAUSIBLE_UNVERIFIED`：合理但证据不足；
- `MISSING`：所需证据不存在。

请先输出符合 `OUTPUT_SCHEMA.json` 的 JSON，再输出 Markdown 解释。Markdown 至少包含：

1. `overall_verdict`，只能是 `PLAN_EXECUTED`、`PARTIALLY_EXECUTED`、`BLOCKED`、
   `FAILED_PLAN_COMPLIANCE` 或 `INSUFFICIENT_EVIDENCE`；
2. 逐项计划对照表，列出计划要求、实现、测试、真实执行、目标状态和缺口；
3. 单独给出 `s0_parity_verdict`、`fs_fit_divergence_verdict`、`stage4_readiness`、
   `novelty_claim_status`、`weather_utility_claim_status`；
4. 按优先级排列所有阻塞项和最小后续行动；
5. 明确列出当前不能对外宣称的结论。

最终必须严格区分：工程环境可运行、代码实现完成、测试通过、GPU 执行、官方 parity 通过、
downstream scientific experiment 完成，以及 novelty/weather utility 得到证明。证据不足时写
`INCONCLUSIVE`，不要用推测填补缺失证据。

## 分析结束后的强制交付物

分析完成后，请不要只返回文字建议。请根据你的审查结果，生成一个供 Codex 继续执行的
后续计划包，并将它打包为：

`EarthDelta_Codex_Followup_Plan_<YYYYMMDD>.zip`

zip 内必须有以下结构：

```text
EarthDelta_Codex_Followup_Plan_<YYYYMMDD>/
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

1. `CURRENT_GAP_ANALYSIS.md` 逐项引用本次审查的结论、`file:line`、commit 或 artifact，
   明确哪些目标已完成、部分完成、阻塞或尚未开始；
2. `FOLLOWUP_PLAN.md` 给出有顺序的修复与实验计划，包含每项任务的输入、命令、预期输出、
   成功标准、失败处理和预计耗时；
3. `EXECUTION_DAG.md` 明确任务依赖关系，禁止在前置 gate 失败时启动 downstream scientific run；
4. `STOP_CONDITIONS.md` 复制所有必须保持的阈值、准入条件、冻结 split 和禁止放宽的门槛；
5. `CODEX_TASKS.json` 必须是机器可读任务列表，每项至少包含 `id`、`priority`、`depends_on`、
   `scope`、`commands`、`acceptance_criteria`、`evidence_outputs`、`stop_if` 和 `estimated_minutes`；
6. `ARTIFACT_CONTRACT.md` 规定每个实验必须保存的 JSON、日志、hash、job ID 和指标字段，避免
   Codex 只得到口头结论；
7. `TEST_PLAN.md` 把 CPU 测试、GPU gate、回归测试和科学结果验证分开；
8. `CODEX_EXECUTION_PROMPT.md` 必须是一段可以直接复制给 Codex 的执行指令，要求它先读取
   `START_HERE_FOR_CODEX.md`，再按 DAG 顺序工作，持续保存 evidence，并在 STOP 条件触发时停止；
9. 如果 S0 multistep parity 仍失败或 Fs-fit divergence 仍未解决，计划必须把它们列为前置阻塞项，
   不得把 expert training、Stage 4 cache 或 novelty conclusion 安排在其前面；
10. zip 中不得伪造尚未运行的结果。所有未来结果必须标为 `TO_BE_RUN`，当前已知结果必须标为
    `OBSERVED`，缺失证据必须标为 `MISSING`。

最后请同时返回：

- 完整审查报告；
- zip 文件名；
- zip 内文件清单；
- zip 的 SHA-256；
- Codex 执行该计划前必须补充上传的文件清单。
