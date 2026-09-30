# 给 ChatGPT Pro：独立复查并制定下一阶段执行包

请担任 EarthDelta 项目的独立研究审查者和下一阶段计划负责人，用中文回答。请基于代码和真实证据判断，不机械接受旧计划或本轮报告的解释。

仓库：https://github.com/sisuolv/EarthDelta
本轮固定工作分支：https://github.com/sisuolv/EarthDelta/tree/online-stage0-20260930
审阅入口：`reviews/online_stage0_20260930/README.md`。
如果我同时提供审阅 ZIP，请先核对其中 `REVIEW_PACKAGE.json` 的提交与文件清单；不要误用缓存的旧 main。先列明实际读到的 HEAD、文件和无法访问的材料。GitHub 读不到时使用 ZIP，不得假装已经读取。

请依次阅读：

1. `reviews/online_stage0_20260930/{FINAL_REPORT.md,DECISION.json,STATUS.json,SESSION_AUTHORIZATION.json}`；这些是待复核的执行记录，不是权威结论。
2. `plans/plan_v9_2_online_20260930/` 的机器合同、统计计划、数据规则、任务计划及修订记录；同时参考 `prior_review/`、`prior_t0/` 和 `prior_plan/V91_SOURCE.zip`。
3. `earthdelta/online/`、`scripts/online_*.py`、对应测试。重点看 `metrics.py`、`coverage.py`、`power.py`、`output_fit.py`、`directions.py`、`grad_utils.py` 及实际 worker。
4. 审阅目录中的覆盖率、功效、参数/输出选择收据，以及 `fit_influence_002/FIT_PAIRED_CELLS.npz`（80×9×24 的小型拟合 MSE 表，非原始天气场）。能用 Python 复核的请实际复核；模型权重和原始天气数据未公开，不能声称重跑了天气模型。

本轮已完成真实拟合和参数校准，但正式分析期尚未运行。合成覆盖率验证采用 2 倍区间宽度后通过；注入 2pp 效应时，完整 Gate 的功效在估计期来源为 0.2%，在选择期来源为 97.6%，所以当前停在 `STATISTICAL_INCONCLUSIVE`。**请独立判断这个停止决定是否合理，尤其不要把这个巨大差异直接当作真实季节外功效的事实。**

请回答：

- 实现是否忠实执行计划？逐项区分完成、部分完成、未完成和偏离，给出文件/函数证据。
- 是否存在数据泄漏、因果时钟、训练/选择污染、F0 分母、物理单位、面积权重、梯度/编辑、配对统计、bootstrap 或功效代码的错误？按严重性排序，给出最小修复和可证伪的验收标准。
- 重点复查覆盖模拟与经验零效应功效模拟是否相容；独立按臂归一化是否改变关键协方差；短拟合期、OCL 训练乐观性、区块数和完整 Gate 交集如何影响功效。哪些是代码问题，哪些是统计设计问题，哪些只是证据不足？
- 当前结果支持哪些有限结论？哪些仍未证明？明确分开 ENGINEERING_PASS、EXPERIMENT_EXECUTED、SCIENTIFIC_SUPPORT、NOVELTY_SUPPORT。
- 下一步应继续、修改还是放弃什么？不要默认必须沿用参数编辑路线，也不要仅因旧 K4 或未完成探针就判定整个方向无效。比较修复功效估计、时间块外预测、冻结配置的描述性分析、输出订正/强静态基线和暂停参数投入等选项；挑出最能改变研究决策的 3–5 项。
- 评价研究价值与 novelty 时核实最接近的原始论文/官方实现并给出来源；DABC 或普通在线输出订正本身不能直接算新意。

请据此设计下一轮约 9 小时的执行方案。资源假设为最多 4 张标准 H100，另有 spot H100/5090；提出明确的并发、卡时和墙钟上限，包含失败与抢占成本。CPU 重计算放 ACP worker，CCI 只做编辑、提交和监控；5090 推理需引擎资格检查，梯度用 H100。API 凭证只从环境变量读取。

每项任务必须写：要解决的问题、前置证据、具体文件/函数、测试与命令、输入/输出、数据角色和访问边界、成功/失败/停止标准、预算，以及失败后转向。优先获得最小真实证据，避免新增大量文档、无效审计循环或 HPO。不能自动访问 2021/2022、修改原冻结结果、放宽原阈值或把未授权的描述性分析当正式 GO；需要改变这些边界时，另列待批准的新协议。

**最终请生成并附上可下载的 `EarthDelta_Next_Execution_Plan.zip`**，方便交给 Codex。至少包含：

- `START_HERE.md`、`INDEPENDENT_REVIEW.md`、`FINDINGS.json`；
- `PLAN.md`、`TASKS.json`（唯一机器任务合同）、`DECISIONS.md`；
- `DATA_METRIC_CONTRACT.json`、`TEST_PLAN.md`、`STOP_CONDITIONS.md`；
- `CODEX_EXECUTE_PROMPT.md`、`EVIDENCE_INDEX.json`、`MANIFEST.sha256`。

尚未批准的权限明确标记 MISSING，不代签。新计划不能冒充已完成实验。对话最后只需概述“当前到哪 → 主要问题 → 已可信/未证明 → 下一步顺序”，并给出 ZIP 下载链接；若环境不能生成附件，请明确说明并提供完整文件内容，不能编造下载链接。
