# ChatGPT Pro prompt — independent review of the latest EarthDelta state

你是 EarthDelta 的独立研究与工程审查者。请基于我提供的最新仓库快照、计划包、CPU 测试、ACP GPU 回执和审计材料，判断当前代码是否真正执行了原计划、哪些目标已经被真实证据支持，以及接下来是否值得继续投入。不要把程序启动、测试通过或 S0 数值等价性误写成动态编辑的科学价值或 novelty 已经成立。

## 输入与阅读顺序

请先确认仓库实际 HEAD、分支、工作区状态和远端同步情况，然后按顺序完整阅读：

1. `README.md`；
2. `plans/plan_v6_0925/README.md`、`EXECUTION_STATUS_20260925.md`、`GPU_RECHECK_EVIDENCE_20260925.md`；
3. `plans/plan_v6_0925/` 中的计划 ZIP（解压后阅读 `START_HERE_FOR_CODEX.md`、`FOLLOWUP_PLAN.md`、`EXECUTION_DAG.md`、`STOP_CONDITIONS.md`、`CODEX_TASKS.json`、`CURRENT_GAP_ANALYSIS.md`）；
4. `codex_audit_round4/` 的完整审计材料；
5. `plans/plan_v4_0923/` 中 FP-05a、FP-05b、WeatherBench-X 和 FP-06 的协议、决策、结果和测试证据；
6. 实际代码与当前 diff，特别是 `earthdelta/bridge/stormer_bridge.py`、`scripts/s0_gate.py`、Fs 训练/质量门、policy OOF、candidate cache 和 WeatherBench-X 入口；
7. 若我同时上传了 AFS 原始材料，再读取：
   - `/mnt/afs/260010168/acp_runs/ed-r6-s0-overlay-recheck-20260925T115020Z/s0/gate_output/s0-gate-20260925t115202730947z/s0_gate_result.json`；
   - `/mnt/afs/260010168/acp_runs/ed-r6-h100-overlay-20260925T114520Z/result.json`；
   - `/mnt/afs/260010168/acp_runs/ed-r6-h100-4single-20260925T113532Z/shard*/result.json`；
   - FP-05b 的原始 GPU job、policy、bootstrap 和 WeatherBench-X Tier-B 回执。

如果 ChatGPT Pro 无法访问私有 GitHub 或 `/mnt/afs`，请把这些 Markdown/JSON 文件作为附件提供；不得用本 prompt 的摘要替代原始证据。

## 已知但必须独立核实的事实

- 最新 4 卡 H100 S0 作业 `pt-lujnmfhg` 在固定 overlay（Torch `2.3.1+cu121`、xformers `0.0.27`）下通过；1/4/12 步官方 parity 的 `max_abs_diff` 均为 `0.0`，冻结容差仍为 `1e-5`。
- 4 个原生标准镜像单卡任务均拿到 H100，但因镜像没有 `xformers` 失败。请区分调度/硬件可用和官方依赖合同可用。
- 旧的 Fs formal 训练发散 finding 仍需独立处理；merge-equivalence 通过不能代替训练稳定性门。
- FP-05b 的动态 bank 样本外主策略没有达到预注册 minimum-effect bar；不能因为 S0 通过就改写这个负面/INCONCLUSIVE 结果。
- 当前没有足够证据宣称动态编辑带来天气预报收益或 novelty 已经被实验证明。

## 必须回答的问题

### A. 计划执行与工程状态

1. `plans/plan_v6_0925` 的每一项任务实际处于 `DONE`、`PARTIALLY_DONE`、`BLOCKED`、`INCONCLUSIVE` 还是 `NOT_STARTED`？分别给出代码、测试、真实 job 和科学目标四层判定。
2. S0 通过是否可信？请核对源代码 hash、checkpoint/normalization binding、官方参考 manifest、1/4/12 步差异和冻结容差；明确说明 overlay 与原生镜像的差别。
3. 最新修改是否引入新的依赖、可复现性或发布问题？

### B. Fs 与动态 bank 的科学有效性

1. 从 `per_sample_loss_direction`、`FsFitRecord`、训练循环、学习率/调度、梯度裁剪、种子、专家数据切分和质量门逐行核对 Fs divergence 是否是真问题。
2. 判断当前 Stage 3b/FP-04 状态应称为“数值管线通过但训练稳定性未解决”“可用但需排除发散专家”还是更严重；给出明确 verdict。
3. 评估 FP-05b 的 BELOW/STRADDLE、`H_Fs` 和 M3≈M1 结果分别意味着什么，是否支持继续扩大实验，还是应优先重设计。
4. 明确回答 Stage 4/candidate cache 是否可以启动。除非存在可审计的 fail-closed divergence gate，否则不要建议把发散 Fs 送入 Stage 4。

### C. novelty/value 与相关工作

请结合近期 weather foundation model、test-time adaptation、parameter-efficient editing、adapter routing、mixture-of-experts、forecast sensitivity/observation impact、online model selection 和 WeatherBench-X 相关论文，给出：

- EarthDelta 当前最可信的 novelty 表述；
- 哪些部分只是已有思想的组合或工程实现；
- 哪些主张已有实验支持，哪些仍是 `INCONCLUSIVE`；
- 若动态 bank 没有样本外收益，最值得保留和重构的核心贡献是什么；
- 最少需要哪些对照实验才能把“有价值的研究结果”与“只是实现完整”区分开。

不要要求“完全没有重叠”，但必须避免把组合性贡献、工程可靠性和性能 novelty 混成同一件事。

### D. 后续 8–10 小时计划

请按信息价值排序设计下一阶段计划。优先解决会改变研究决策的阻塞项，保留真实可复现的 CPU/GPU 证据。每项任务必须有：输入、命令或伪命令、预计资源、输出 artifact、成功标准、STOP 条件和失败后的 pivot。要求：

- 先固定 S0 的 overlay/source/hash 合同；
- Fs divergence 必须有根因证据或 fail-closed 排除门；
- 不得因为诊断结果倒写 FP-05b 已冻结的数字；
- 不得在 S0 或训练稳定性门失败时启动 Stage 4；
- 结果必须分别标记 `ENGINEERING_PASS`、`EXPERIMENT_EXECUTED`、`SCIENTIFIC_SUPPORT` 和 `NOVELTY_SUPPORT`。

## 输出格式

先输出 JSON，至少包含：

```json
{
  "overall_verdict": "...",
  "s0_verdict": "...",
  "fs_divergence_verdict": "...",
  "fp05b_value_verdict": "...",
  "novelty_verdict": "...",
  "stage4_authorization": "...",
  "blocking_items": [],
  "next_plan_priority": []
}
```

然后输出 Markdown 审核报告，必须包含：

1. 当前 HEAD、工作区和证据边界；
2. 代码/测试/GPU/科学目标四层状态表；
3. S0 可信性核查；
4. Fs divergence 独立判断及 Stage 3b/Stage 4 建议；
5. FP-05b 与 WeatherBench-X 的数字复核和科学解释；
6. novelty/value 的保守表述与最强可辩护贡献；
7. 排序后的 8–10 小时执行计划和 STOP 条件；
8. 不能对外宣称的结论。

每条 finding 给出精确 `file:line`、commit 或 artifact 路径，并标记 `STATIC_CONFIRMED`、`EXECUTION_CONFIRMED`、`PLAUSIBLE_UNVERIFIED` 或 `MISSING`。禁止凭摘要补全缺失证据，禁止伪造未运行的 GPU 或科学结果。

## 强制生成后续计划 ZIP

审核完成后，请生成一个新的 Codex 执行包：

`EarthDelta_Codex_Followup_Plan_20260925_v2.zip`

ZIP 根目录必须包含：

```text
EarthDelta_Codex_Followup_Plan_20260925_v2/
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

ZIP 内必须：

- 把已执行结果标为 `OBSERVED`/`DONE`，未执行内容标为 `TO_BE_RUN`，缺失材料标为 `MISSING`；
- 把 S0 overlay 合同、源 hash、job ID、冻结 `1e-5` 容差写入 artifact contract；
- 把 Fs divergence 和 FP-05b 负面/INCONCLUSIVE 结果作为前置阻塞，不得让 Codex 自动跳入 Stage 4；
- 明确禁止 FP-07/诊断实验倒写 FP-05b 已冻结数字；
- 对每个任务列出 `id`、`priority`、`depends_on`、`scope`、`commands`、`acceptance_criteria`、`evidence_outputs`、`stop_if`、`status`、`estimated_minutes`；
- 最终返回 JSON 审核结果、Markdown 报告、ZIP 文件名、ZIP 文件清单、ZIP SHA-256，以及执行前仍需补充的外部文件清单。
