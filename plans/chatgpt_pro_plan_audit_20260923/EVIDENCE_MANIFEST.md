# Evidence manifest

The reviewer should inspect these paths when they are available:

| Question | Evidence to inspect |
|---|---|
| What was planned? | `source_plan/NEXT_STEPS_10H_PLAN.md`, `source_plan/CLAUDE_10H_PROMPT.md`, the three Stage 1A review files |
| What code exists? | `earthdelta/`, `scripts/`, `tests/`, `README.md`, `pyproject.toml` |
| What did Round 4 find? | `reference_reports/ROUND4_REPORT.md`, `codex_audit_round4/round4_summary.json`, `codex_audit_round4/fs_fit_divergence_assessment.json` |
| What happened in the sprint? | `reference_reports/SPRINT_8H_RESULTS.md`, `reference_reports/SPRINT_8H_DECISION.json` |
| Did S0 pass? | Latest `s0_gate_result.json`, `job_result.json`, parity diagnostic JSON under local `artifacts/` |
| Did downstream science run? | Expert run records, qualification records, registry, candidate cache, evaluation and holdout artifacts |

For each claim, label evidence as one of:

- `STATIC_CONFIRMED`: verified directly from code or a supplied artifact;
- `EXECUTION_CONFIRMED`: verified from a real job record or test log;
- `PLAUSIBLE_UNVERIFIED`: reasonable but not directly evidenced;
- `MISSING`: the required evidence was not supplied.

Do not treat a generated plan, an argv file, a test definition, or a prepared job manifest as
evidence that the corresponding experiment ran.
