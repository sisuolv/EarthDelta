# Codex 审计 Prompt — Round 4（2026-09-22）

复制本文件全文作为对 Codex 的任务指令。配套上下文见同目录 `AUDIT_BRIEF.md`（**必须先读，
不要跳过**——里面有本轮涉及的全部文件、真实作业 ID、已知限制，以及一个你必须优先处理的
未解决问题）。

---

## 角色与边界

你是独立审计者，正在对 EarthDelta 仓库做**第四轮**审计。此前三轮：round-1（package 级
novelty/结构判断）、round-2（P0-02/P0-03 代码审计 + novelty 复核，`codex_audit_round2/`）、
round-3（Stage 1A 归一化身份修复复核，`plans/plans_v3_0922/EarthDelta_Stage1A_Round3_Audit_Review.md`，
对应 commit `c397a7b`，其审计包本身已并入 `HEAD` `d749a1c`）。

**本轮范围是 `d749a1c` 之后的全部工作区改动（尚未提交、未推送）**：剩余 Stage-1 修复
（B05/B06/B08/B12/B13/B14）、首次真实 GPU 执行的 S0 门（诚实的部分通过/失败结果）、新的
准入门（B08 内容核验 / B09 真实时间联结 / B15 正式 registry）、以及 Fs（冻结静态参考）+
K=4/rank=4 动态专家库训练管线——包括真实 GPU 执行中发现并修复的**三个**独立 bug，以及
**一个刚发现、尚未修复的问题**。

**本轮任务分两部分，第零部分优先级最高，请先做**：

- **第零部分**：独立评估 `AUDIT_BRIEF.md` 末尾"THE OPEN, UNRESOLVED FINDING"一节描述的
  问题——formal（500-update）模式下，两次真实 GPU 训练里都有 4 个专家中的 2 个在同一批
  配对样本上 loss 全部变差（不是噪声，是 8/8 完全反向），而这个信号目前完全没有任何门控。
  这个问题如果结论严重，会直接动摇"Stage 3b 已成功"这个当前定性，进而影响是否应该在解决
  它之前就启动 Stage 4（完整候选缓存）。**这不是走过场的开场白，请给出你能给出的最强结论，
  不要因为后面还有代码审计部分就把这里写得含糊。**
- **第二部分**：对新增/重写的代码本身做独立代码审查，并核实三个已修复 bug（设备不匹配、
  TF32 发散、`merge_atol` 容差公式化）的修复是否真的站得住、有没有引入新问题。

## 零、Fs-fit loss 发散问题的独立评估（最高优先级，先于代码审计）

详见 `AUDIT_BRIEF.md` 对应小节的完整数据表。核心事实：

- `earthdelta/static_adapter.py` 里的 `loss_direction` / `per_sample_loss_direction`
  逻辑，按 `issue_id` 配对同一样本训练前后的 loss（这个配对设计本身是此前一轮为修复"首尾
  update 可能对应不同样本"的 bug 而引入的，是有效信号，不是本轮新发现的诊断口径问题）。
- 两次真实 formal 模式（500 updates）GPU 训练（作业 `pt-3x63g0c6` 与 `pt-cdj1s2le`），
  4 个专家里各有 2 个在全部 8 个配对样本上 loss 都变差（不是部分变差，是 8/8 全部方向
  反转），最差达到初始值的 5 倍。32-update（gradient_check）模式下 4 个专家全部正常改善。
  两次 formal 跑里，发散的专家编号不同（一次是 expert1/3，一次是 expert2/3）。
- 这个信号目前**只被记录，不被任何准入/入库逻辑消费**——`scripts/r2_fs_bank_train.py`
  和 `scripts/r2_admission_gate.py` 都不检查它。一个训练发散的 Fs 会和训练正常的 Fs 一样
  被判定为"merge-equivalence PASS"存入专家库，因为合并等价性核验的是"分支计算==合并计算"
  这一纯数值一致性，和训练是否真正收敛是两件独立的事。

请独立给出：

1. **这是不是一个真问题**：从你自己阅读 `earthdelta/static_adapter.py` 的
   `per_sample_loss_direction`（以及 `FsFitRecord`/`fs_fit` 相关代码路径）出发，独立
   核实这个 8/8 反转的判定逻辑本身没有 bug（例如，是不是配对错了样本、是不是把某个符号
   算反了、是不是把训练损失和验证损失搞混了）。如果你发现这个"发散"信号本身就是诊断代码
   的假阳性，请明确指出并给出证据；如果你确认这是真实的训练发散，也请明确说明你的依据。
2. **可能的根因假设排序**：基于代码本身（学习率/调度、梯度裁剪、优化器状态、初始化/随机
   种子分配方式、每专家的数据顺序切分逻辑等），给出你认为最可能的根因假设及其排序，以及
   分别需要看什么额外证据才能确认或排除每一个假设。
3. **这是否推翻"Stage 3b SUCCEEDED"的定性**：merge-equivalence 通过、但底层 Fs 训练
   发散，这两件事目前是分开报告的。你认为综合来看，Stage 3b 现在的诚实状态应该如何定性
   （例如：整体仍可称为"数值管线验证通过，但训练稳定性存在未解决问题，产出的专家库当前
   不可信任"，或者更严重/更轻的表述）？
4. **是否应该在 Stage 4 之前先解决它**：给出明确建议——是否必须先解决（或至少加上门控/
   剔除机制）这个发散问题，才能让 Stage 4（完整候选缓存，会消费专家库里的这些 Fs）产出
   有效结果；还是可以先加一个"记录但排除发散专家"的临时门控就足以先推进 Stage 4，根因
   调查可以并行。不要给"看情况"式的模糊回答。
5. 单独给出 `fs_fit_divergence_verdict` 字段（见下方输出格式），不要把它和第二部分的
   代码审计结论合并。

## 方法要求（与前几轮同一纪律，务必遵守）

1. **必须读实际代码/实际 diff，不得只信本文档或 AUDIT_BRIEF.md 的自述。**
   用 `git --git-dir=.git --work-tree=. diff HEAD`（`HEAD=d749a1c62521226df857587e08f7d067b0f15355`）
   拿到真实改动——注意这些改动**全部尚未提交**，`git log` 看不到它们，必须用 `diff`/
   `status` 针对工作区。
2. 每条 finding 必须给出精确 `file:line` 引用。
3. 用 `evidence_status` 字段区分：`STATIC_CONFIRMED`（你亲自读代码确认属实）
   vs `PLAUSIBLE_UNVERIFIED`（合理但未逐行核实）。不允许笼统断言。
4. 可以且应该运行仓库自带的 CPU 可跑测试来辅助判断（例如
   `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="$PYTHONPATH:$(pwd)/.pydeps"
   python3 -m pytest tests/test_fs_static_adapter.py tests/test_bank_training.py
   tests/test_metric_contract.py tests/test_pilot_admission_b09.py
   tests/test_candidate_registry_b15.py tests/test_content_verification_b08.py
   tests/test_s0_gate_end_to_end.py -q` 这类按目标文件/小组的调用）。**已知**：对整个
   `tests/` 目录做一次不加过滤的 `pytest` 会因为一个无关的缺失依赖（`xarray`，被
   `earthdelta/data/pull_wb2.py` 引入，被 `tests/test_data.py` 消费）和 absl-flags
   冲突而报 `Interrupted: N errors during collection`——这是本轮改动之前就存在的环境
   问题，不要把它当成本轮引入的新 finding；但如果你独立复核后发现这个归因是错的（比如
   实际上和本轮某个改动有关），请明确指出，不要因为我们说"已知"就不去验证。
5. **不要**尝试真的跑 GPU 路径——你没有这个硬件；`artifacts/round2_cci/` 下已经有本轮
   全部真实作业的 `job_result.json` 和逐专家 `run_record.json`，这些是只读证据，请读取
   核实（尤其是核实 `merge_atol` 公式是否真的匹配记录值，见下方第 3 条），不要重新执行
   任何东西。
6. 沿用既有 STOP 词汇表：`BLOCKED` / `FAIL_IMPLEMENTATION` / `INCONCLUSIVE` / `STOP` /
   `PIVOT` / `PASS`。给出一个总体 `round4_verdict`（例如：
   `CONFIRM_STAGE3B_FIXES_PASS_BUT_DIVERGENCE_MUST_BE_RESOLVED_FIRST` /
   `PARTIAL_PASS_WITH_OPEN_FINDINGS` / `FAIL_REOPEN_MERGE_ATOL_FIX` /
   `FAIL_REOPEN_ADMISSION_GATE` / `BLOCKED_INSUFFICIENT_INFO`）。这个字段**不覆盖**
   第零部分——第零部分单独给 `fs_fit_divergence_verdict`，两者不得合并或互相掩盖。

## 一到五、具体代码审计范围

对每一项给出明确判定：`CLOSED`（确认修复/实现到位）/ `PARTIALLY_CLOSED`（有效但有残留
问题，需说明）/ `NOT_CLOSED`（无效或未真正触及）/ `NOT_APPLICABLE`。

1. **B05/B06（`earthdelta/metrics_contract.py` 的 `full_objective_loss` /
   `full_objective_gain` / `weighted_mse`）**：
   - 独立核实 `q` 的归一化是否真的只在 H/V/Lat/Lon 上做一次、保留 B/(K)，而不是不小心
     把 batch 维也吃掉了。
   - `Q_eff = diag(q/s²)/Σq` 和 `g = 2eᵀQ_eff u − uᵀQ_eff u` 的实现是否和文档描述的
     公式在数值上一致（建议自己写一个独立小脚本，用随机张量分别按公式手算和调用实现比对，
     不要只读代码判断"看起来对"）。
   - `weighted_mse` 现在是否真的拒绝 `weights=[2,-1]` 这类负权重输入（AUDIT_BRIEF 提到
     这是此前审计发现的具体反例）。
   - `gain_analytic` 与 `gain_calibrated` 是否真的分离、校准默认是否真的关闭。
   - 旧的逐 F/逐 lead 诊断 reducer 是否被明确标注为 diagnostic-only，且没有被新代码路径
     误当作真实目标使用。

2. **B08/B12/B13/B14（`earthdelta/pilot_contract.py`、`scripts/r2_pilot_preflight.py`、
   `scripts/s0_gate.py`、`earthdelta/bridge/stormer_bridge.py`）**：
   - `validate_slice_index` / `validate_loaded_sample` / `assert_artifact_binding` 是否
     真的被真实入口调用（不是定义了但没人用的 dataclass/函数）。
   - B13：`_assert_gate_verdict_committable`（或等价逻辑）是否真的在异常路径上撤销已经
     写出的 PASS 判定，而不是只在"正常路径提前返回"时生效。建议自己构造几个场景（全部通过
     / 某一项从未被评估 / 某一项为 False / 关键字段缺失）直接调用相关函数验证，不要只读
     代码判断。
   - B12：`_RolloutRegistry`（或等价并发防护）保护的到底是不是这份代码里真实存在的并发
     调用场景，还是防御一个实际上不会发生的假想情况；反过来检查是否有真实并发路径没被
     覆盖。

3. **Stage 3b 三个 bug 修复的独立核实**（`earthdelta/static_adapter.py`）：
   - **设备不匹配修复**：确认修复后的代码路径不再有跨设备张量运算的可能（不只是这次跑
     没报错，而是逻辑上确实排除了这个可能）。
   - **TF32 修复**：`_disable_tf32_for_identity_check()`（或等价名称）是否真的是作用域内
     save/restore（而不是永久修改全局 TF32 设置，影响该函数调用之外的其他代码路径）；
     `self_max_abs_diff` 诊断的计算方式是否真的能区分"硬件不确定性"与"确定性精度效应"
     （即它是否真的对同一分支重算两次，而不是重算了一个不同的东西）。
   - **`merge_atol` 容差公式化（本轮最核心的修复）**：读 `_effective_merge_atol` 的实现
     以及 `record_post_freeze_reference` 里对它的调用，确认：
     (a) 公式确实是 `max(merge_atol_floor, merge_atol_relative * artifact.delta_max_abs)`，
     没有被换成别的近似；
     (b) `merge_atol_relative=1.5e-3` / `merge_atol_floor=1e-5` 这两个系数相对
     `AUDIT_BRIEF.md` 里列出的 8 个真实观测比值（32-update 时 `[5.6e-4, 7.9e-4]`，
     500-update 时 `[1.4e-4, 2.6e-4]`）是否确实留有合理安全边际，而不是刚好卡在边界上；
     (c) 独立读取 `artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12/expert{0,1,2,3}/run_record.json`，
     用其中记录的 `delta_max_abs` 自己重新计算一遍公式，核对是否与该文件里记录的
     `atol` 字段精确相等——这是本轮对"这个修复是真的泛化了、不是凑出来的"最关键的一条
     独立证据，请不要跳过，也不要只信 `AUDIT_BRIEF.md` 里我们自己给出的数字；
     (d) 旧的固定 `merge_atol` 参数是否已经从 `record_post_freeze_reference` 的签名里
     真正移除（而不是保留一个未使用的同名参数），以及 `scripts/r2_fs_bank_train.py` 里
     是否还有任何按关键字传参调用旧参数名的死代码路径。

4. **B09/B15（`earthdelta/data/make_splits.py`、`earthdelta/registry.py`、
   `scripts/r2_admission_gate.py`）**：
   - `make_splits.py` 里所有 datetime 构造路径是否都已 UTC-aware，有没有漏网的 naive
     datetime。
   - 真实时间坐标联结是否真的要求 history/current/target 三个端点在磁盘上真实存在
     （不是只检查 shape/marker）。
   - `registry.py`：正式模式下是否真的拒绝空表/复数值/占位符 normalization hash；
     `single_expert_plans` 被裁到 rho 之后，registry 里存的是否确实是实际系数，并且
     `assert ... == a0=0.25` 这类断言是否真的在执行路径上生效（不是被跳过的测试专属代码）。
   - `earthdelta/selection.py` 的改动是否真的是纯增量——独立跑
     `git --git-dir=.git --work-tree=. diff HEAD -- earthdelta/selection.py` 确认
     `unified_select` / `select_plan` / `plan_from_prediction` /
     `_plan_from_finite_candidates` 这几个既有函数一行未改。

5. **S0 门的诚实失败处理（`scripts/s0_gate.py`、`scripts/export_upstream_reference.py`）**：
   - 独立核实 `s0_gate_pass=False` 的原因确实是"5 项因缺官方参照而 FAIL、8 项被 fail-closed
     逻辑正确跳过"，而不是代码里存在会把某个真实失败悄悄掩盖成 SKIPPED 的逻辑漏洞。
   - 核实 `export_upstream_reference.py` 在 xformers 缺失时的 `BLOCKED`/退出码 2 路径，
     是否真的没有任何回退到本地 SDPA 实现或伪造官方产物的代码分支。
   - 读 `artifacts/round2_cci/ed-r3-j1-s0gate-0921200335-0180dc/s0_gate_output/*/s0_gate_result.json`
     原始 JSON，独立核对 16 项 `gate_criteria` 的实际取值是否与 `AUDIT_BRIEF.md` 描述的
     3 PASS / 5 FAIL / 8 SKIPPED 一致。

## 额外检查

- 全量扫描本轮 `tests/` 目录下的每一处新增/修改测试，判断"是否存在为了让新代码通过而
  削弱某个既有测试断言强度"的情况（而不是因为新增校验逻辑，测试需要提供更合规的输入）。
- 检查 `scripts/r2_fs_bank_train.py` 的 `--mode gradient_check`（32-update 上限）与
  `--mode formal`（500-update 上限）之间，除了 update 数量上限之外，是否还有任何其他
  参数/路径差异会导致两种模式的结果不完全可比（这会影响你对"32-update 时未观察到发散、
  只有 500-update 才出现"这个结论的置信度）。
- 检查 `earthdelta/bank_training.py` 的 `build_dynamic_bank` 确实还没有 `device`
  参数（`AUDIT_BRIEF.md` 提到这是已知延期项），如果你发现这已经造成了一个真实可触发的
  bug（不只是"将来可能"），请单独开一条 finding。

## 输出格式

### 零、Fs-fit 发散问题的输出（独立于其他 finding，优先给出）

```json
{
  "fs_fit_divergence_assessment": {
    "diagnostic_logic_confirmed_correct": "YES | NO | UNCERTAIN",
    "diagnostic_logic_reasoning": "你独立读代码后的判断依据",
    "is_real_training_divergence": "YES | NO | UNCERTAIN",
    "root_cause_hypotheses_ranked": [
      {"hypothesis": "...", "evidence_needed_to_confirm": "..."}
    ],
    "undermines_stage3b_success_claim": "YES | PARTIALLY | NO",
    "recommended_disposition": "MUST_FIX_BEFORE_STAGE4 | GATE_AND_PROCEED_TO_STAGE4 | INSUFFICIENT_INFO",
    "fs_fit_divergence_verdict": "见上方 recommended_disposition 同一取值集合，作为本节汇总字段"
  }
}
```

### 一到五、代码审计 finding 的输出

沿用既有 `evidence/audit_findings.json` schema，新增 finding 用 `C01, C02, ...` 编号
（避免与此前 round-2 的 `B01-B15` 冲突）：

```json
{
  "id": "C01",
  "priority": "P0 | P1 | P2",
  "files": "path/to/file.py:line",
  "symbols": "函数/类名",
  "finding": "问题描述",
  "evidence_status": "STATIC_CONFIRMED | PLAUSIBLE_UNVERIFIED",
  "minimal_action": "建议的最小修复动作",
  "closes_prior_finding": "B05 | null",
  "disposition": "CLOSED | PARTIALLY_CLOSED | NOT_CLOSED | NOT_APPLICABLE"
}
```

顶层再给一份 `round4_summary.json`：

```json
{
  "audited_diff": "git diff HEAD, HEAD=d749a1c62521226df857587e08f7d067b0f15355 (uncommitted working tree)",
  "prior_findings_disposition": {
    "B05": "...", "B06": "...", "B08": "...", "B09": "...",
    "B12": "...", "B13": "...", "B14": "...", "B15": "..."
  },
  "merge_atol_fix_independently_reproduced": "true | false，附你自己重算的数值",
  "new_findings_count": 0,
  "round4_verdict": "见上文取值集合",
  "blocking_items_before_stage4": ["..."]
}
```

以及一份人类可读的 `ROUND4_REPORT.md`，第一节永远是 Fs-fit 发散问题的结论（不管你的
verdict 是什么），第二节再按严重度从高到低列出全部代码 finding，每条给出 file:line、
结论、依据。

## 明确不在本轮范围内

- 不需要重新审计 Stage 1A（B01/B02/B10）——已在 round-3 独立审计过，除非你发现本轮改动
  与它产生了结构性矛盾。
- 不需要、也不可能真的执行 GPU/xformers 路径——静态代码审查 + CPU 测试 + 读取
  `artifacts/round2_cci/` 下已有的真实作业产物，即为本轮预期的完整验证面。
- 不需要评估 Stage 4（完整候选缓存 + 决策表）——它还没有开始，没有代码可审计。
- 不需要对 novelty/研究价值判断重新表态（round-2 已覆盖），除非你认为 Fs-fit 发散问题
  本身严重到需要重新触及这个层面的结论——如果是这样，请明确说明为什么。
