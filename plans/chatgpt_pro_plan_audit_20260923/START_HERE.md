# 给 ChatGPT Pro 的使用顺序

1. 上传本 zip 包。
2. 发送 `CHATGPT_PRO_PLAN_AUDIT_PROMPT.md` 的完整内容。
3. 如果 ChatGPT Pro 无法读取私有仓库，再补充上传 `EVIDENCE_MANIFEST.md` 中列出的代码、测试日志和 JSON 证据。
4. 要求它先输出结构化 JSON，再输出人类可读的 Markdown 分析。
5. 审查结论必须以当前证据为准；不得因为计划写了某一阶段就把该阶段判定为已完成。
6. 最后要求它按照 `CODEX_FOLLOWUP_PLAN_ZIP_SPEC.md` 生成真正的后续计划 zip，并返回 zip 的 SHA-256、
   文件清单以及 Codex 执行前仍需补充的输入文件。

推荐补充上传的最小文件是：

- `earthdelta/`、`scripts/`、`tests/`；
- `README.md`、`.gitignore`、`pyproject.toml`；
- `codex_audit_round4/ROUND4_REPORT.md` 及其 `round4_summary.json`；
- `plans/novelty_review_20260922/SPRINT_8H_RESULTS.md` 和 `SPRINT_8H_DECISION.json`；
- 最新 S0 gate 的 `s0_gate_result.json`、parity diagnostic JSON 和 job result JSON。

ChatGPT Pro 生成的后续 zip 应由 Codex 解压后先读取 `START_HERE_FOR_CODEX.md`，再按
`EXECUTION_DAG.md` 执行。任何 `BLOCKED` 或 `STOP` 条件都必须保留，不能由 Codex 自行放宽。
