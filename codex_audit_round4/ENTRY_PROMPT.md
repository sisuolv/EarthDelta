你现在是 EarthDelta 仓库的独立审计者（Round 4）。请先后完整阅读以下两个文件，
不要跳过或只读摘要：

1. `/mnt/afs/260010168/EarthDelta/codex_audit_round4/AUDIT_BRIEF.md` —— 背景与范围
2. `/mnt/afs/260010168/EarthDelta/codex_audit_round4/AUDIT_PROMPT.md` —— 你的完整任务指令

读完 `AUDIT_PROMPT.md` 后，严格按照其中的要求执行审计（尤其是"第零部分"——
Fs-fit loss 发散问题的独立评估，优先级最高，必须先给出结论），并按其指定的
JSON schema + `ROUND4_REPORT.md` 格式输出结果到
`/mnt/afs/260010168/EarthDelta/codex_audit_round4/` 目录下。

仓库根目录：`/mnt/afs/260010168/EarthDelta`（当前分支
`audit/round2-review-20260921`，`HEAD=d749a1c`，所有本轮改动均为未提交的工作区
状态，需用 `git diff HEAD` 而非 commit range 查看）。
