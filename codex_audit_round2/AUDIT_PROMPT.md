# Codex 审计 Prompt — Round 2（2026-09-21）

复制本文件全文作为对 Codex 的任务指令。配套上下文见同目录 `AUDIT_BRIEF.md`（先读它，
它包含了本轮需要核查的具体改动、commit、文件行数和已知限制——不要跳过）。

---

## 角色与边界

你是独立审计者，正在对 EarthDelta 仓库做**第二轮**审计。第一轮（你自己产出的
`EarthDelta_Codex_Execution_Package_20260921`）审计了 commit
`4fe55a7af90ea92f62a3232a571af92bfbd6114d`，判定 `CONDITIONAL_CONTINUE_P0_ONLY`，
只解锁 P0-01。此后我们按你自己的任务 DAG（`codex_tasks.json`）依序执行并各自独立验证了
P0-01 → P0-02（commit `607fad5`）→ P0-03（commit `fb767f7`），当前 HEAD 为
`fb767f7f6efbc428be39c9ad84f5905331d6e40f`。

**本轮任务分两部分，第零部分优先级最高，请先做**：

- **第零部分（最重要，是整个项目的地基）**：独立重新评估这个研究 idea 本身是否真的
  novel、是否真的有价值——不是评估代码有没有正确实现设计，而是评估设计本身站不站得住。
  见下方"零、idea 本身的 novelty 与价值判断"一节。这部分如果给出负面或存疑结论，后面
  第二部分的代码审计结果如何都不改变结论——请不要把它当成走过场的开场白，如果你的分析
  导向"这个前提有问题"，直接明确说出来，不要为了配合后续代码审计的篇幅而弱化这个结论。
- **第二部分**：核实 P0-02/P0-03 是否真正修复了你在第一轮报告的
  A01/A03/A04/A05/A06/A11，并对新增/重写的代码本身做独立代码审查（修复本身也可能引入
  新问题）。不要重复第一轮已完成的工作（包完整性、DAG 结构、任务分解本身），除非你发现
  这次的改动与之产生了结构性矛盾。

## 零、idea 本身的 novelty 与价值判断（最高优先级，先于代码审计）

详细背景见 `AUDIT_BRIEF.md` 的"Novelty and value re-assessment"一节——那里写明了
idea 的具体主张（签发时刻联合学习 e0 与 du，用精确 FSO 二次型 `gain=2⟨e0,du⟩-‖du‖²`
组合，在预算约束+交互项 H 下做编辑选择）、我们内部（非独立）已做过的 novelty 结论
（surveys A-E + 两次外部检索，结论：novelty 成立，风险在 headroom 不在 occupancy）、
以及四个最近邻工作（Aurora LoRA / WeatherPEFT 2509.22020 / Adapter Banks 2609.17042 /
VI-MoLE 2608.02528）。请不要重复我们的结论——独立重新推导。

请给出：

1. 你自己独立判断：是否真的没有已发表工作占据这条"签发时刻联合学习 e0+du、精确 FSO
   组合、预算约束+交互项选择"的管线？如果你的知识截止日期或检索能力有限制导致无法充分
   核实较新的工作，明确声明这个限制，不要在不确定的情况下给出确定性措辞。
2. 独立核实我们列的四个"最近邻"工作是否真的不占据这条管线——如果你认真读过后发现其中
   某个工作实际上比我们以为的更接近，直接指出。
3. 超越"有没有人做过"这个层面，评估**机制上的合理性**：联合学习 e0 和 du 并用精确 FSO
   二次型组合，是否有先验理由认为这比简单的直接/静态基线（或 `experiments/BASELINES.md`
   /`research/OUTPUT_CORRECTION_CHALLENGE.md` 里已经列出的输出纠错挑战者基线）能捕获
   更多可用信号？还是说这个 idea 的成立与否完全取决于还没跑的 P0-05 oracle 天花板实验，
   在那之前纯属未经检验的假设？
4. 评估 **P0-05 实验设计本身**（`experiments/P0_SURVIVAL_EXPERIMENTS.md` 里"全量有限
   候选 oracle 天花板"go/no-go 实验）——即使它真的跑出一个干净的 PASS/FAIL，这个实验设计
   本身是否真的能公正、决定性地回答"这个 idea 有没有价值"这个问题？还是你发现了会让这个
   实验结果本身变得没有信息量的设计缺陷？
5. 给出明确建议：应该按现有前提继续投入 P0-04（生存规格+参照库训练），还是这个前提本身
   需要先被重新审视、或项目需要暂停/转向，才该继续投入工程资源？不要给模糊的"看情况"式
   回答——如果你确实无法确定，明确说清楚"需要什么具体证据才能解决这个不确定性、以及如何
   在投入更多资源之前拿到这个证据"。

## 方法要求（与第一轮同一纪律，务必遵守）

1. **必须读实际代码/实际 diff，不得只信 commit message 或我们的自述。**
   用 `git diff 4fe55a7..fb767f7`（或分别看两个 commit）拿到真实改动。
2. 每条 finding 必须给出精确 `file:line` 引用。
3. 用 `evidence_status` 字段区分：`STATIC_CONFIRMED`（你亲自读代码确认属实）
   vs `PLAUSIBLE_UNVERIFIED`（合理但未逐行核实）。不允许笼统断言。
4. 可以且应该运行仓库自带的 CPU 可跑测试来辅助判断（`pytest tests/ -q -rs`），但**不要**
   尝试真的跑 GPU/xformers 依赖的路径——你没有这个硬件，我们也没有，这些路径的真实结果本来
   就还没产生，属已知限制，见 AUDIT_BRIEF.md 末尾说明，不要因为"没跑通"而把这个标记为新
   finding。
5. 沿用你自己第一轮的 STOP 词汇表：`BLOCKED` / `FAIL_IMPLEMENTATION` / `INCONCLUSIVE` /
   `STOP` / `PIVOT`，以及 `PASS`。本轮请给出一个总体 `round2_verdict`，取值集合与第一轮
   `CONDITIONAL_CONTINUE_P0_ONLY` 同一体系（例如：`CONFIRM_P0_02_03_PASS_CONTINUE_P0_04` /
   `PARTIAL_PASS_WITH_OPEN_FINDINGS` / `FAIL_REOPEN_P0_02` / `FAIL_REOPEN_P0_03` /
   `BLOCKED_INSUFFICIENT_INFO`）。这个 `round2_verdict` 只覆盖代码审计部分；第零部分
   （novelty/价值判断）请单独给出 `novelty_value_verdict`，两者不要合并成一个字段——
   即使代码审计全绿，novelty/价值判断也可能是负面或存疑的，两个结论必须能独立呈现，不能
   互相掩盖。

## 一到六、具体代码审计范围（六项，逐项给结论）

对 A01/A03/A04/A05/A06/A11 各给一个明确判定：`CLOSED`（确认修复到位）/
`PARTIALLY_CLOSED`（修了但有残留问题，需说明）/ `NOT_CLOSED`（修复无效或未触及）。

1. **A01（最高优先级）**：`scripts/s0_gate.py` 重写 + 新增
   `scripts/export_upstream_reference.py` 是否真正做到"不自我对照"？重点检查：
   - `export_upstream_reference.py` 是否真的调用了 `reference/stormer/` 里带真实
     xformers 的官方代码路径，而不是又绕回本地 SDPA 实现或某种命名空间替换。
   - 无 xformers/GPU 时是否确实 `BLOCKED` 退出而非静默降级或伪造结果（我们已自测退出码为
     2，但请你独立核实脚本逻辑本身，不要只信我们的测试结果）。
   - `gate_criteria` 的 7 项是否全部 fail-closed（初始 `False`，异常路径不得变 `True`）。
   - `upstream_parity` 判据在参照产物缺失时的行为是否清晰阻断，而不是被其他判据的
     `all(...)` 逻辑意外掩盖。

2. **A03**：`earthdelta/selection.py` 的 `_plan_from_finite_candidates` 新增的
   finiteness / bound / max_active / max_candidates 四项校验是否都真正生效（不是加了参数
   但从未在判断逻辑里使用）。检查 `bound=0.5` 的那处测试改动是否合理（我们认为合理，请独立
   复核，不要预设我们是对的）。

3. **A04**：核实 `earthdelta/metrics_contract.py` 是否被 `paired.py` / `heads.py` /
   `geometry.py` / `selection.py` 真正调用（而不是新建了模块但原有 5 处实现仍留着死代码
   或仍被使用）。重点核实数学上 sum vs mean 的取舍是否合理——不只是"内部一致"，而是"对
   FSO/编辑选择这个应用场景来说，这个取舍在数学上站得住脚"。

4. **A05**：checkpoint SHA-256 是否真的接入了某个会被检查/绑定的地方（而非算出来又没人用，
   重蹈覆辙）；`NormalizationContract.digest` 是否真的把变量名序、interval 键、shape 都
   纳入了摘要计算（不是加了参数但摘要算法本身没变）。

5. **A06**：`earthdelta/data/make_splits.py` 是否所有 datetime 构造路径都已 UTC 感知，
   有没有漏网的 naive datetime；新增的 `availability_source` 字段是否在每条实际产生
   split 记录的代码路径里都被正确赋值（不是只在某个分支里赋值，其他分支留空/默认值不当）。

6. **A11**：`plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md`（新增，未改写原
   `research_spec_v6.yaml`）这种"指针说明"处置方式，你认为是否足够，还是必须原地修改
   `research_spec_v6.yaml` 本身才算真正 CLOSED。

## 额外要检查的新引入风险（不在原 A01-A13 列表里，需要你新开 finding ID，例如 B01, B02...）

- `earthdelta/metrics_contract.py`（新文件，297 行）、
  `scripts/export_upstream_reference.py`（新文件，459 行）、
  重写后的 `scripts/s0_gate.py`（约 1059 行）——作为全新代码独立审查，而不仅仅对照旧
  finding 打勾。
- `earthdelta/bridge/stormer_bridge.py` 里新增的 `controlled_rollout` 并发复用防护
  ——检查这个防护对应的到底是不是这份代码里真实存在的并发调用场景，还是防御了一个实际上
  不会发生的假想情况（过度设计）；反过来也检查是否有真实并发路径没被这个防护覆盖到
  （防护不足）。
- 是否存在任何一处，为了让新增测试通过而**削弱**了某个已有测试的断言强度（而不是因为新增
  校验逻辑，测试需要提供更合规的输入）——请全量扫描 `607fad5` 和 `fb767f7` 两个 commit
  里 `tests/` 目录下的每一处改动，逐条判断"是合理适配"还是"是变相放水"。
- `earthdelta/contracts.py`、`earthdelta/data/pull_wb2.py`、split/manifest 相关代码
  里，有没有会污染或偏置 P0-04（2015-2018 数据训练参照专家字典）的问题——P0-04 还没开始，
  这是在它开始前最后的检查窗口。

## 输出格式

### 零、novelty/价值判断的输出（独立于代码 finding，优先给出）

```json
{
  "novelty_value_assessment": {
    "agrees_no_prior_occupancy": "YES | NO | UNCERTAIN",
    "occupancy_reasoning": "你独立推导的理由，不是复述我们的结论",
    "knowledge_cutoff_limitation_disclosed": "true | false，以及具体说明你的检索/知识\n      能覆盖到什么时间点，超出部分如何处理",
    "nearest_neighbor_recheck": [
      {"work": "Aurora LoRA", "still_non_occupying": "YES | NO | UNCERTAIN", "note": "..."},
      {"work": "WeatherPEFT 2509.22020", "still_non_occupying": "YES | NO | UNCERTAIN", "note": "..."},
      {"work": "Adapter Banks 2609.17042", "still_non_occupying": "YES | NO | UNCERTAIN", "note": "..."},
      {"work": "VI-MoLE 2608.02528", "still_non_occupying": "YES | NO | UNCERTAIN", "note": "..."}
    ],
    "mechanism_plausibility": "对'联合学习e0/du+精确FSO组合是否有先验理由优于基线'的判断与理由",
    "p0_05_experiment_design_critique": "对 oracle 天花板实验设计本身是否公正/决定性的评估，若发现设计缺陷请具体指出",
    "novelty_value_verdict": "PROCEED_TO_P0_04 | PAUSE_FOR_LITERATURE_RECHECK | REDESIGN_ORACLE_CEILING_EXPERIMENT | FUNDAMENTAL_CONCERN_STOP",
    "if_uncertain_what_would_resolve_it": "需要什么具体证据、如何在投入更多资源前拿到它（仅当 verdict 含不确定性时必填）"
  }
}
```

### 一到六、代码审计 finding 的输出

请仿照你自己 `evidence/audit_findings.json` 的 schema 追加新一轮结果，字段：

```json
{
  "id": "B01",
  "priority": "P0 | P1 | P2",
  "files": "path/to/file.py:line",
  "symbols": "函数/类名",
  "finding": "问题描述",
  "evidence_status": "STATIC_CONFIRMED | PLAUSIBLE_UNVERIFIED",
  "minimal_action": "建议的最小修复动作",
  "closes_prior_finding": "A01 | null",
  "disposition": "CLOSED | PARTIALLY_CLOSED | NOT_CLOSED | null"
}
```

顶层再给一份 `round2_summary.json`：

```json
{
  "audited_commit_range": "4fe55a7..fb767f7",
  "prior_findings_disposition": {
    "A01": "CLOSED|PARTIALLY_CLOSED|NOT_CLOSED",
    "A03": "...", "A04": "...", "A05": "...", "A06": "...", "A11": "..."
  },
  "new_findings_count": 0,
  "round2_verdict": "见上文取值集合",
  "blocking_items_before_P0_04": ["..."]
}
```

以及一份人类可读的 `ROUND2_REPORT.md`，按严重度从高到低列出全部 finding，每条给出
file:line、结论、依据。

## 明确不在本轮范围内

- **不需要重新评估 P0/P1/P2 整体任务分解与排布顺序**（谁在谁之前、门控依赖关系本身）
  ——那部分不变。**例外**：第零部分第 4 条明确要求你评估 P0-05 oracle 天花板实验*设计*
  本身能否公正回答 novelty/价值问题，这一条仍在范围内，不受本条排除；这里排除的是"任务
  排布顺序是否合理"，不是"某个具体实验设计是否有逻辑漏洞"。
- 不需要、也不可能真的执行 GPU/xformers 路径——静态代码审查 + CPU 测试套件运行结果即为
  本轮预期的完整验证面。
- 不需要对 T-P0.4（生存规格/阈值证书/参照库训练）的具体实现给出实质意见——它还没开始写
  代码，等它完成后会有独立的第三轮审计请求。
