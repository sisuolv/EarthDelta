你是 EarthDelta 项目的独立计划执行审查者。这是对上一轮审查（`plans/chatgpt_pro_plan_audit_20260923/`，产出了 `EarthDelta_Codex_Followup_Plan_20260923.zip`，即本仓库里的
`plans/plan_v4_0923/extracted_followup/EarthDelta_Codex_Followup_Plan_20260923/`）的**后续一轮**。上一轮审查判定当时的实现是 `PARTIALLY_EXECUTED`，科学路线 `BLOCKED`：S0 官方 multistep parity 未过、fitted-Fs 的退化/共同参考问题未关闭。本轮请你判断：(1) 上一轮 zip 里 `CODEX_TASKS.json` 定义的 FP-00~FP-04 是否被真正执行、上一轮的阻塞项是否真正解决；(2) 我方（Claude Code 会话）新写的 FP-05 计划是否合理、完整、没有漏洞；(3) 在此基础上生成新一轮供 Codex 执行的后续计划 zip。

## 仓库与证据位置

仓库：`https://github.com/sisuolv/EarthDelta`，分支 `audit/round2-review-20260921`，当前 HEAD `e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`。上一轮审查时的实现提交是 `a3596e6b804e9d23b72d1247b08c47129d6b50b1`，比较基线 `d749a1c62521226df857587e08f7d067b0f15355`；本轮新增的两个 commit 是 `bc57a70`（FP-00~FP-04 全部代码改动 + 真实 GPU 作业证据）和 `e0136d4`（FP-05 计划文档）。GPU 权重、数据集和完整 `artifacts/` 目录按 `.gitignore` 排除，没有推上 GitHub；如果需要具体某个 job 的权重文件核实，告诉我路径，我单独提供 hash 或精简摘要。

请先完整阅读，按此顺序：

1. `plans/chatgpt_pro_plan_audit_20260923/` 下的上一轮审查材料，尤其是 `EarthDelta_Codex_Followup_Plan_20260923.zip` 解压后的 `START_HERE_FOR_CODEX.md`、`FOLLOWUP_PLAN.md`、`STOP_CONDITIONS.md`、`CODEX_TASKS.json`——这是本轮的验收基准，不是背景资料。
2. `plans/plan_v4_0923/CLAUDE_OPUSPLAN_EXECUTION_PROMPT.md`——本轮实际驱动执行的指令。
3. 按顺序阅读四个真实 run 目录（每个都有 `protocol/`、`decisions/`、真实 GPU `jobs/` 提交记录，`certify/` 或 `task_result_*.json`）：
   - `plans/plan_v4_0923/run_20260923T_fp01_trace/`（S0 gate 真实 GPU 复验）
   - `plans/plan_v4_0923/run_20260923T192759Z_fp03_fs/` 和 `run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/` 和 `run_20260924T013959Z_fp03_v2/`（Fs 认证的完整过程，含协议 v1 到 v2 的演变、两次探索性诊断、DEV-001/DEV-002 偏差记录）
   - `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/`（K=4 bank 认证的完整过程，含 DEV-FP04-001 偏差记录）
4. `plans/plan_v4_0923/FP05_PLAN.md`——尚未实现的下一阶段设计。

不得只根据计划文字、README 或本 prompt 的自述作结论，必须核对实际代码、实际 diff、实际 JSON 证据。

## 逐项核查：FP-00~FP-04 是否真正执行

对每一项区分四个层次，不能把前两层自动升级为后两层：

1. 代码存在；
2. CPU/单元测试通过；
3. 真实 GPU job 执行并产出证据（要有真实 job ID、`job_result.json`、`returncode`、`elapsed_seconds`）；
4. 科学目标已经被实验支持。

具体检查：

- **S0 官方 multistep parity**：上一轮明确记录 1 步通过、4 步和 12 步超出冻结的 `1e-5` 容差。本轮声称已经全部修复（真实 job `pt-g3344e9z`，`S0_GATE_REPORT.md` 显示三个注册 rollout 全部 `max_abs_diff=0.0`）。请核实这个修复是真的定位并修正了根因（`earthdelta/bridge/stormer_bridge.py` 的 inverse-transform 精度问题 + `scripts/s0_gate.py` 的 TF32 问题），还是变相放宽了容差或用别的手段让检查通过。容差数值本身有没有被改动？
- **Fs-fit 退化（round-4/round-3 的核心遗留问题）**：本轮声称已定位真正根因（训练当时跑在未认证的模型后端上，且 TF32 设置不一致，导致看似"GPU 随机性"决定哪个专家训练发散；修复后训练变成逐位确定性），并通过一次完整的预注册协议（`fs_protocol_v2.json`）在一个全新、从未使用过的 2019 留出集上正式验证通过（`verdict=FS_SELECTED`）。请核实：(a) 这个根因诊断有没有实际代码证据支持，不是事后合理化；(b) 协议 v2 的预注册流程是否真的在提交任何 GPU 作业前就冻结了规则（检查 `protocol/*.sha256` 的时间戳和 job 提交时间的先后关系）；(c) 最终认证的 Fs 在留出集上的真实数字是什么（据我方记录：训练集内改善 2.6%，留出集实际还差 0.34%，落在预注册的 2% 容忍区间内）——这是否真的达到了"合格"，还是只是"没有严重伤害"？
- **K=4 动态专家库认证（本轮新增，不在上一轮范围内）**：请核实历史专家库训练"每个 worker 各自重新拟合一份不同 Fs"这个缺陷描述是否属实（检查 `run_20260924T033627Z_fp04_bank/analysis/` 和历史 job 记录里的 `fs_backbone_digest`），以及本轮认证（`verdict=BANK_CERTIFIED`）是否真的堵住了这个漏洞（4 个 worker 应该有同一个 Fs 引用哈希）。组装等价性验证（A1-A5）是否真的做到逐位精确，还是设了容差？专家资格判定（own-group direction 的硬门槛）有没有被事后放宽？
- **FP-02（原计划要求的入口消费链）**：据我方记录，这一项只做了和 FP-03 直接相关的子集（`test_plan_gate_chain.py`、`test_loaded_admission_binding.py`），原计划要求的 `scripts/plan_followup_runner.py` 薄封装 CLI 和 profile 不可变性测试当初没做（profile 不可变性后来在 FP-04 里补了，`tests/test_profile_immutability.py`）。请如实标注这一项为部分完成，不要因为其它阶段推进了就默认它也完成了。
- **数据 split 冻结**：请核实截至本轮结束，是否存在任何真正强制执行的 bank_fit/policy_dev/confirm 曝光窗口隔离机制，还是仍然只是文档里记录"以后要排除这些窗口"的意图（`earthdelta/pilot_contract.py` 里 `DataRole` 的文档字符串明确写"admission 层只打标签，下游消费者自己负责校验"）。

## 逐项核查：FP-05 计划是否合理

`FP05_PLAN.md` 记录了六个已被接受的判断，请逐条给出你自己独立的评估，而不是照搬我方给出的理由：

1. policy_dev = 2019 下半年，confirm 预留 = 2020 下半年；
2. delta_min = 0.34%（取自 Fs 自己实测的样本外伤害幅度）；
3. 额外加一条仅报告用的 F0 背景轨迹；
4. 一次性定样本量（不做分阶段扩样，避免 optional stopping）；
5. 主策略 ridge_direct_gain，regime 分组为次要；
6. 双向阻塞交叉验证 + 支持窗口净化。

请特别关注：delta_min 的推导逻辑（"动态层至少要能补回 Fs 自己造成的伤害才算有用"）是否站得住，有没有更好的锚点；policy_dev 窗口和已训练窗口的季节错配（专家只在 1-6 月数据上训练过，评估在 7-12 月）会不会让整个 FP-05 从一开始就测不出真实效果，如果会，本轮计划有没有充分承认这个限制并做相应的解释（不是隐藏它）；`earthdelta/split_freeze.py` 这个新闸门的设计是否足够堵住数据泄漏，有没有遗漏的入口点。

## 证据标注规则（沿用上一轮）

每条 finding 必须给出精确 `file:line`、commit 或 artifact 路径，并标记：

- `STATIC_CONFIRMED`：直接逐行核实；
- `EXECUTION_CONFIRMED`：真实测试或 GPU 记录核实；
- `PLAUSIBLE_UNVERIFIED`：合理但证据不足；
- `MISSING`：所需证据不存在。

## 输出要求

先输出结构化 JSON（字段至少包含：`overall_verdict`、`fp00_04_disposition`（每个子阶段的独立判定）、`s0_parity_verdict`、`fs_fit_divergence_verdict`、`bank_certification_verdict`、`fp05_plan_soundness_verdict`、`novelty_claim_status`、`weather_utility_claim_status`、`blocking_items`），再输出 Markdown 分析，至少包含：

1. `overall_verdict`，只能是 `PLAN_EXECUTED`、`PARTIALLY_EXECUTED`、`BLOCKED`、`FAILED_PLAN_COMPLIANCE` 或 `INSUFFICIENT_EVIDENCE`；
2. 逐阶段（FP-00 到 FP-04）对照表：上一轮计划要求、本轮实现、测试、真实 GPU 执行、目标状态、缺口；
3. FP-05 计划的独立评审意见，逐条判定六个已接受判断是否合理，指出你认为需要修改的地方；
4. 按优先级排列所有阻塞项和最小后续行动；
5. 明确列出当前仍然不能对外宣称的结论（尤其是：K=4 bank 的样本外价值目前完全没有证据，Fs 本身样本外只是"无害"不是"有益"）。

最终必须严格区分：代码实现完成、测试通过、GPU 执行、官方 parity 通过、downstream scientific experiment 完成，以及 novelty/weather utility 得到证明。证据不足时写 `INCONCLUSIVE`，不要用推测填补缺失证据。

## 分析结束后的强制交付物

分析完成后，请根据你的审查结果，生成一个供 Codex 继续执行的后续计划包，并打包为：

`EarthDelta_Codex_Followup_Plan_20260924.zip`

zip 内必须有以下结构（和上一轮一致）：

```text
EarthDelta_Codex_Followup_Plan_20260924/
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

1. `CURRENT_GAP_ANALYSIS.md` 逐项引用本次审查的结论、`file:line`、commit 或 artifact，明确 FP-00~FP-04 里哪些目标已完成、部分完成、阻塞或尚未开始，FP-05 计划哪些部分可以直接采用、哪些需要修改；
2. `CODEX_TASKS.json` 的任务编号请延续 `FP-` 前缀（`FP-05` 起步，可以拆分成更细的子任务，例如 `FP-05a`/`FP-05b`），不要重新从 `FP-00` 编号——FP-00~FP-04 已完成的部分请在任务列表里标注为 `status: DONE`、附上对应的真实证据路径，而不是重新列为待执行；
3. `FOLLOWUP_PLAN.md` 给出有顺序的任务计划，包含每项任务的输入、命令、预期输出、成功标准、失败处理和预计耗时（不虚构耗时，标 `MISSING_NEEDS_MEASURED_PROFILE` 也可以）；
4. `EXECUTION_DAG.md` 明确任务依赖关系，禁止在前置 gate 失败时启动 downstream scientific run；
5. `STOP_CONDITIONS.md` 复制所有必须保持的阈值、准入条件、冻结 split 和禁止放宽的门槛——如果你认为 `FP05_PLAN.md` 里的 delta_min、split 窗口等判断需要修改，请在这里给出你修改后的版本并说明理由，而不是简单地照抄；
6. `CODEX_TASKS.json` 每项至少包含 `id`、`priority`、`depends_on`、`scope`、`commands`、`acceptance_criteria`、`evidence_outputs`、`stop_if`、`status`、`estimated_minutes`；
7. `ARTIFACT_CONTRACT.md` 规定每个实验必须保存的 JSON、日志、hash、job ID 和指标字段；
8. `TEST_PLAN.md` 把 CPU 测试、GPU gate、回归测试和科学结果验证分开；
9. `CODEX_EXECUTION_PROMPT.md` 必须是一段可以直接复制给 Codex 的执行指令，要求它先读取 `START_HERE_FOR_CODEX.md`，再按 DAG 顺序工作，持续保存 evidence，并在 STOP 条件触发时停止；
10. 如果你认为 K=4 bank 的样本外价值仍是未知数、或 FP-05 的设计有需要收紧的地方，必须把它列为前置阻塞项或强制修改项，不得让 Codex 在这些问题解决之前就开始 FP-06 的最终决策；
11. zip 中不得伪造尚未运行的结果。已完成的部分标 `OBSERVED`/`DONE`，未来结果标 `TO_BE_RUN`，缺失证据标 `MISSING`。

最后请同时返回：

- 完整审查报告；
- zip 文件名；
- zip 内文件清单；
- zip 的 SHA-256；
- Codex 执行该计划前必须补充上传的文件清单（例如具体某个 job 的原始日志、某个权重文件的 hash）。
