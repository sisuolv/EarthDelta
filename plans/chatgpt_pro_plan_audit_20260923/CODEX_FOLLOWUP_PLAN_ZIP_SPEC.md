# Codex follow-up plan zip specification

ChatGPT Pro 的最终交付物必须是一个真正可解压、可检查、可交给 Codex 的 zip 文件，
而不是 Markdown 中伪造的附件链接。

## Required package

The archive root must be `EarthDelta_Codex_Followup_Plan_<YYYYMMDD>/` and contain:

- `START_HERE_FOR_CODEX.md`: entry point and exact reading order;
- `FOLLOWUP_PLAN.md`: ordered work plan with commands and acceptance criteria;
- `EXECUTION_DAG.md`: dependencies and gates;
- `STOP_CONDITIONS.md`: hard stop rules and frozen thresholds;
- `CODEX_TASKS.json`: machine readable tasks;
- `ARTIFACT_CONTRACT.md`: required evidence outputs;
- `TEST_PLAN.md`: CPU/GPU/scientific validation;
- `REPRODUCIBILITY.md`: environment, commit, seed, data, and asset bindings;
- `CURRENT_GAP_ANALYSIS.md`: evidence-backed gap list;
- `evidence/PLAN_AUDIT_RESULT.json`;
- `evidence/EVIDENCE_MANIFEST.json`;
- `evidence/REQUIRED_INPUTS.md`;
- `prompts/CODEX_EXECUTION_PROMPT.md`.

## Status vocabulary

Use `OBSERVED` only for facts supported by supplied code, logs, tests, or job artifacts.
Use `TO_BE_RUN` for future work. Use `MISSING` when the input or evidence was unavailable.
Use `BLOCKED` when a prerequisite gate failed. Do not turn a prepared command into an
execution result.

## Required task fields

Every `CODEX_TASKS.json` item must contain:

```json
{
  "id": "S0-DIAG-001",
  "priority": "P0",
  "depends_on": [],
  "scope": "",
  "status": "TO_BE_RUN",
  "commands": [],
  "acceptance_criteria": [],
  "evidence_outputs": [],
  "stop_if": [],
  "estimated_minutes": 30
}
```

The plan must keep official parity, fitted-Fs stability, expert qualification, cache
construction, and holdout utility as separate gates. A downstream task may depend on a
gate but may not silently bypass it.
