# CODEX_TASKS

Iteration: `round2-next-iteration-20260921`。本表P0编号不是历史DAG中的同号任务。

## P0 — 一个迭代

- [ ] P0-01 修复最小correctness/身份/共同指标路径，完成真实S0（当前唯一授权；CPU补丁不等于真实S0）
- [ ] P0-02 实际数据准入、合格非零小字典与冻结registry（LOCKED）
- [ ] P0-03 完整候选dev表与oracle/static余量（LOCKED）
- [ ] P0-04 合法策略、direct-gain、双头与四格筛查（LOCKED）
- [ ] P0-05 条件性feedback/output/virtual挑战并结束本迭代（LOCKED）

Current task: P0-01

Do not proceed beyond: P0-01

Unlock condition: P0-01真实PASS_MEASUREMENT产物完成，并收到明确P0-02范围/资源授权；仅CPU通过或工具exit 0均不得解锁。

每项完成只在执行副本勾选并写decision.json；原交付包作为只读基准保存。状态变化不得抹掉失败、跳过和BLOCKED。下一任务的allowed_tasks/current_task只有新授权后才更新，Codex不得为赶进度自行改。

若无GPU/checkpoint/data，P0-01完成允许的CPU补丁与证据后写PATCH_CPU_PASS_REAL_S0_BLOCKED并停止。不进行bank训练。所有后续任务现在未执行。
