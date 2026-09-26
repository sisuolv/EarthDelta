# ChatGPT 后续独立分析 Prompt

请对 GitHub 仓库 EarthDelta 做一次独立、证据优先的研究与工程审查。

先读取当前 HEAD、分支、工作树状态和最近提交，不要默认 README 或旧报告仍然正确。重点阅读：

1. `reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md`
2. `plans/plans_v7_0925/EarthDelta_Codex_Followup_Plan_20260925_v2(1).zip`（若仓库中可见）
3. `earthdelta/`、`scripts/`、`tests/` 中与 FP06、policy OOF、WBX、candidate cache、selector 和 source manifest 相关的实现

请把结论分成四层：

- 工程代码是否正确；
- 实验是否真的执行且数据边界是否合规；
- 方法是否产生有效效果；
- 结果是否足以支持 novelty/value。

必须独立验证，而不是复述 review：

- GitHub HEAD 是否包含当前 dirty worktree 的修复；
- FP06 runner 是否真正绑定 source/input manifest、threshold 和 current release；
- WBX 是否可能通过已加载数组、缺失 valid-time 或 authorized override 绕过年份边界；
- 24h/6h/72h 的 metric、分母、bootstrap 和 erratum gate 是否一致；
- 当前结果是否只是冻结 DEV，是否存在 fresh/confirm/Tier-A 证据；
- 当前最强基线是否仍是 M1/static。

请输出：

1. 已完成、部分完成、未完成和偏离计划的逐项表格；
2. 按 P0/P1/P2 排序的代码、数据、指标和科学风险；
3. 每个风险的最小修复和可验证成功/失败标准；
4. 下一阶段 3–5 个任务，每个任务写清目标、前置条件、成功标准、失败后的停止规则；
5. 明确哪些实验不值得继续；
6. 最终给出“当前可以相信什么、仍未证明什么、下一步是否应该保留 dynamic-routing”的明确结论。

不要把测试通过、脚本 rc=0、历史 Tier-B receipt 或 metadata shape 当成科学成功。不要读取或建议读取未授权的 confirm/holdout 数据。若需要新的实验，请先定义冻结协议、数据资格、指标、预算和停止规则。

如果分析工具支持生成文件，请同时输出一份机器可读的后续执行计划（JSON/YAML 均可），包括任务依赖、命令、输入、输出、停止条件和数据访问边界，便于后续 Codex 执行。
