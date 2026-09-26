# ChatGPT Pro 审核 Prompt（2026-09-26）

用法：给 ChatGPT Pro 开通 GitHub 私有仓库 `sisuolv/EarthDelta` 的读取权限，然后把下面代码块里的全部内容粘贴进去。它取代同目录下较早的 [`CHATGPT_FOLLOWUP_ANALYSIS_PROMPT_20260926.md`](CHATGPT_FOLLOWUP_ANALYSIS_PROMPT_20260926.md)：那份只针对 Codex 复查，这份要求裁决 Claude 与 Codex 两份复查的分歧。

```text
你是一名独立审稿人，同时具备机器学习天气预报（ML-NWP）、统计评估和研究工程经验。请对 GitHub 私有仓库 sisuolv/EarthDelta 做一次证据优先的独立审核。用中文回答。

==================== 0. 固定版本与可见范围 ====================
- 审核对象：分支 main。代码与实验证据以提交 9e91191 为准；其后的提交只新增本 prompt、调整 README 链接，并让 reviews/claude_independent_review_20260926/scalar_recheck.py 自动定位仓库根目录。先报告你实际读到的 HEAD，并用 git log 确认 9e91191 之后的提交只改动了 README.md 和 reviews/；如果还改动了别的文件，列出差异再继续。
- 以下内容不在 git 中，你看不到：ERA5 zarr（data/）、Stormer 检查点、GPU 产物与端点数组（artifacts/）、数据拉取日志（checkpoints/*.log）、仓库外的 Codex 执行目录、reference/ 下的第三方克隆。Stormer 官方代码请直接看 https://github.com/tung-nd/stormer（项目固定在 commit 58dfee5）。
- 凡结论依赖不可见内容，标 UNVERIFIABLE，不要猜，也不要把文档里的叙述当成已核实的事实。

==================== 1. 背景（只作导航，不作结论） ====================
EarthDelta 的研究问题：冻结一个 ML 天气模型（Stormer，1.40625°，69 通道，6h 步长），准备一组低秩参数编辑（LoRA 专家）；在起报时刻只用合法历史，预测参考误差 e0 和每个编辑的响应 du，用 gain = 2<e0,du> − ||du||² 在预算内选择编辑。
目前真实做过的实验：一个静态适配器 Fs + 4 个 rank-4 专家（K4 bank），在 2019 年下半年 112 个起报（DEV）上用 ridge 路由（M3）、最佳静态（M1）、k-means 路由（M2）和事后 oracle 做了一次离线评估（FP-05b）。
同日有两份独立复查，结论冲突：
- Codex 复查：当前最强基线是 M1/static；72h 护栏通过；下一步在 fresh split 上做一次 selector-only 干预。
- Claude 复查：改用原始 Stormer（F0）作基线后，所有编辑方案（含 oracle）在 6h/72h 显著差于 F0，24h 只好 0.07%；按变量看 M1 让 Z500/T850/MSLP 变差，只有 50–100 hPa 风变好；原因是目标函数权重错配、训练样本太少；数据管线有空年份和 NaN 空洞；建议先修指标与数据，再做"正确版 headroom 实验"，不做 selector 工作。
你的任务是独立核实，而不是复述或折中这两份复查。

==================== 2. 阅读顺序 ====================
1) README.md 中 "Independent reviews (2026-09-26)" 一节
2) reviews/CLAUDE_INDEPENDENT_REVIEW_20260926.md 与 reviews/claude_independent_review_20260926/（脚本 + 输出）
3) reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md（Codex）
4) 原始研究计划 plans/plans_v1_0919/EarthDelta_v6_Review_and_Plan_CN.md，重点 §0、§4.2（P1 headroom 门）、§4.6（S0–S7）、§4.7（必做基线）、§6（风险与否证）
5) 实验预注册与证据：
   - Fs：plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/protocol/PREREGISTRATION_v2.md，以及 run_20260923T192759Z_fp03_fs/admission/bank_fit_manifest.json（8 个训练起报）
   - Bank：plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/protocol/PREREGISTRATION.md、admission/bank_fit_grouping.json
   - FP-05b：plans/plan_v4_0923/run_20260924T104725Z_fp05b/ 下的 protocol/PREREGISTRATION.md、cache/dev/{candidate_results.json, background_f0.json, cache_manifest.json}、policies/{oof_predictions.json, folds.json}、evaluation/{dev_results.csv, paired_block_bootstrap.json}
   - 数据完整性：plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/holdout/finiteness_scan_{2015,2018,2019}.json、run_20260924T104725Z_fp05b/splits/finiteness_scan_2019_H2.json
6) 代码：
   - 目标函数：earthdelta/static_adapter.py（build_objective_spec 约 L327–360；area_weight_q；objective_loss_for_sample）
   - 策略与评分：earthdelta/policy_oof.py（折与 purge 约 L90–150；M1/M2/M3 约 L270–360；realized_losses 与 paired_block_bootstrap 约 L372–450）
   - 特征：earthdelta/candidate_cache.py（L43 FEATURE_GRID；legal_features_np 约 L493–520）
   - 数据管线：earthdelta/data/pull_wb2.py（L398、L423–448、L677、L1103、L1147–1181、L1396–1399）、checkpoints/run_multi_year_pull.sh
   - 审计入口：scripts/r5_audit_runner.py、earthdelta/fp06_decision.py、earthdelta/wbx/evaluate.py、scripts/r5_score_erratum.py、scripts/r7_followup_decomposition.py、scripts/r7_selector_sensitivity.py
   - 从未在真实数据上运行的研究核心：earthdelta/{heads,teacher,probe,paired,selection}.py
   - 训练：scripts/r2_fs_bank_train.py、earthdelta/bank_training.py

==================== 3. 任务一：逐条裁决分歧（必须给证据） ====================
对每一条给出：裁决（CONFIRMED / REFUTED / PARTIAL / UNVERIFIABLE）、证据类型（RECOMPUTED 你自己重算 / CODE_READ / DOC_READ / INFERRED）、具体 file:line 或数字。

D1 最强基线是 F0 还是 M1。
   用 candidate_results.json（1680 行，字段 issue_id、candidate_id、lead_hours、cpu_loss）、background_f0.json（F0 的 cpu 损失）和 oof_predictions.json（M1/M3 的动作），以 F0 为分母自行重算 6/24/72h 的配对 7 天块 bootstrap（块起点 2019-07-06T12Z，23 个非空块，B ≥ 10000，95% 百分位区间），覆盖 Fs、M1、M3、事后 oracle（只在 5 个候选中选）和包含 F0 的 oracle。
   请自己写一份独立实现；可以再运行 reviews/claude_independent_review_20260926/scalar_recheck.py 交叉对照（它只依赖 git 中的文件）。
   还要判断：FP-05b 协议把 F0 排除在候选集和 oracle 之外、把 72h 护栏定义为相对 Fs，是合理的协议选择还是设计错误？

D2 目标函数权重。
   build_objective_spec 用状态标准差 inp_std 归一化、69 通道等权；Stormer 自身训练损失见官方仓库 stormer/utils/data_utils.py 的 WEIGHT_DICT 与 stormer/models/iterative_module.py（差分归一化 + 气压权重）。
   判断 Claude 复查中"湿度约 50%、风约 45%、Z/T/MSLP/T2m 合计不到 3%、50–150 hPa 约 29%"是否可信（见 per_variable_report.txt），以及这个错配对训练结果和评估结论的影响有多大。

D3 按变量的退化。
   端点数组不在 git 中，你无法重算，但请审查 reviews/claude_independent_review_20260926/per_variable.py、per_variable_report.py、per_variable_oracle.py 的正确性：纬度网格（−89.296875 到 89.296875，128 点）、cos 纬度权重、通道顺序、std 来源、"先对起报平均 MSE 再开方"的 RMSE 聚合、bootstrap 做法；并检查报告中"与存储损失相对误差 ≤3.5e-9"的校验逻辑是否真能证明计算正确。

D4 "72h 护栏通过"是否误导。

D5 scripts/r7_followup_decomposition.py 的结论"瓶颈是 selector 误差而不是缺少 headroom"是否成立（对照 oracle−M1 的点估计与区间、delta_min = 0.34%）。

D6 下一步该做什么。
   (a) Codex 建议的 fresh split 上 selector-only 干预（成功标准含 M3 vs Fs ≥ 0.34%）是否值得做、是否可能达标；
   (b) Claude 复查的 T3"正确版 headroom 实验"设计是否合理：三个臂（充分训练的静态 Fs*、逐起报事后最优连续编辑、EWMA 偏差订正）、Go ≥ 2% / Stop < 0.5% 的阈值、逐起报事后拟合的乐观偏差、编辑方向如何选、线性度、成本估计。指出你会怎么改。

D7 数据管线。
   核实 pull_wb2.py 中"时间步数为 0 仍标记完成""finalize 返回 verification_failed 但退出码为 0""完整性检查只看步数与通道数"的说法；根据 finiteness_scan_*.json 中空洞的形态（是否 8 步对齐、是否覆盖整个压力层变量），评估"代理丢块时 fsspec 把缺块当 missing key、zarr 填 NaN"这一推断是否站得住，有无其他解释。日志不在 git 中，相关引用标 UNVERIFIABLE。

D8 Fs 的"认证"。
   Fs 用 8 个起报 × 32 次更新训练，合格门槛 Q6 允许 holdout 上最多 +2% 伤害，实测 +0.34%。这是合理的工程门，还是协议缺陷？它对下游 bank 与结论有什么影响？

==================== 4. 任务二：两份复查都漏掉的问题 ====================
至少检查以下方面，发现问题就按 P0/P1/P2 列出，每条附证据和最小修复：
- 统计：23 个块的比值估计量 bootstrap、百分位区间是否合适；多重比较；OOF 策略拟合的方差是否被区间覆盖；效果以 M0 总和为分母是否一致。
- 泄漏与数据角色：特征、折、purge 是否只用合法信息；2019 是 Stormer 的验证年；2020 原是测试年却被用于训练。
- 代码正确性：policy_oof、candidate_cache、static_adapter、bank_training 中可能影响数字的错误。
- S0 等价性覆盖范围：它是否覆盖后续实验真正用到的路径（带 hook 的编辑前向、训练反传）。
- 可复现性：环境依赖靠 overlay 而非 pyproject；关键结果是否能从 git 中的文件复算。
- 流程：审计与 provenance 工作量与科学进展的比例是否构成主要风险。

==================== 5. 任务三：研究价值判断 ====================
- 核心假设（从合法历史预测 e0 与 du，用 2<e0,du> − ||du||² 在预算内选编辑）现在是否仍值得检验？当前证据是"该配置失败"，还是足以说明"假设不成立"？
- 对一个已充分训练的 ML-NWP 模型做状态依赖的低秩编辑，headroom 的先验有多大？只引用你能给出可点击链接并确认存在的文献；不确定就写"未知"。
- 唯一的正信号在 50–100 hPa 风。它更可能是慢变的环流型偏差（例如 QBO 相关），还是噪声/伪影？值得作为方向，还是会分散注意力？
- 给出能最快做出 Go/Stop 决定的最小实验。

==================== 6. 任务四：下一阶段计划 ====================
给出 3–5 个任务，每个写清：目标、前置条件、方法、数据角色（训练 / dev / confirm 的年份与冻结时点）、指标（必须以 F0 为基线，并报告 Z500、T850、T2m、MSLP、U850、Q700 在 6/24/72/120h 的 RMSE 或 ACC）、成功与失败阈值及理由、停止规则、成本（GPU 卡时 / 天数）、依赖顺序。
另列"暂时不值得投入的实验"。如果你与两份复查的计划都不同意，直接说明理由。

==================== 7. 硬约束 ====================
- 测试通过、脚本 rc=0、receipt 齐全、S0 max_abs_diff=0 都只是工程事实，不能当作科学成功。
- 不要建议在数据角色冻结之前读取任何 confirm/holdout 数据；不要看过技能后再挑年份。
- 区分"这个配置失败"和"这个假设不成立"。
- 不编造文献、数字或文件内容；数字给 3 位有效数字并注明来源（重算 / 文件 / 推断）。
- 不要把两份复查平均或折中；每条结论都要有你自己的依据。

==================== 8. 输出格式 ====================
1) 一句话总判断，随后按"做到哪里 → 哪些地方有问题 → 哪些结论已经可信 → 哪些仍未证明 → 接下来的顺序"五段给出结论
2) 分歧裁决表 D1–D8（裁决、证据类型、关键数字或 file:line、一句话理由）
3) 你重算得到的 F0 基线表（6/24/72h × Fs/M1/M3/oracle/含 F0 的 oracle，点估计与 95% 区间），并注明与 Claude 复查数字的差异
4) 两份复查都漏掉的问题（P0/P1/P2，含证据与最小修复）
5) 研究价值判断
6) 下一阶段计划与"不值得做"清单
7) 机器可读 JSON：{"tasks":[{"id","goal","depends_on","inputs","method","data_roles","metrics","success","failure","stop_rule","data_access","est_cost"}], "not_worth_doing":[...], "unverifiable":[...]}
8) 你无法核实的内容清单，以及要核实它们需要哪些文件
```
