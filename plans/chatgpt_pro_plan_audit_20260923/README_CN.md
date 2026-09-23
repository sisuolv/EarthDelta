# EarthDelta 计划执行复核包

这个压缩包用于交给 ChatGPT Pro，复核当前 EarthDelta 代码是否执行了
`plans_v3_0922` 中的原计划，并判断计划目标是否已经实现。审查结束后，ChatGPT Pro
还必须根据缺口生成一个新的 `EarthDelta_Codex_Followup_Plan_<YYYYMMDD>.zip`，
供 Codex 继续执行；具体交付标准见 `CODEX_FOLLOWUP_PLAN_ZIP_SPEC.md`。

建议先上传整个压缩包，再把 `CHATGPT_PRO_PLAN_AUDIT_PROMPT.md` 作为首条消息发送。
如果界面不支持 zip，按 `START_HERE.md` 中的顺序上传文件夹内的 Markdown、JSON 和报告文件。

包内包含：

- `CHATGPT_PRO_PLAN_AUDIT_PROMPT.md`：主审查 prompt；
- `source_plan/`：原始计划与前一轮复核材料；
- `reference_reports/`：当前 Round 4 审计和 8 小时实验的已知状态；
- `OUTPUT_SCHEMA.json`：要求 ChatGPT Pro 返回的结构化结果模板；
- `CODEX_FOLLOWUP_PLAN_ZIP_SPEC.md`：要求 ChatGPT Pro 生成后续计划 zip 的格式；
- `FOLLOWUP_PLAN_MANIFEST.template.json`：后续计划包的机器可读模板；
- `EVIDENCE_MANIFEST.md`：需要从私有仓库补充上传的代码、测试和 GPU 证据清单；
- `CURRENT_STATUS.md`：基线、最新提交和当前阻塞状态。

GPU 权重、数据集、CUDA 环境和完整 `artifacts/` 目录没有放进包内。它们体积很大，
并且其中的 job artifact 需要从本地仓库单独上传或提供精简 JSON 摘要。
