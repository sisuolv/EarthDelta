# EarthDelta v9.2：公开审阅入口

本目录整理 2026-09-30 的实际执行，不重跑实验、不改写原收据。当前判定是 **STATISTICAL_INCONCLUSIVE，分析期尚未运行**。这既不是“参数编辑有效”，也不是“参数编辑已被科学证伪”。

## 阅读顺序

1. [FINAL_REPORT.md](FINAL_REPORT.md)：实际进展、结果、限制和下一步。
2. [DECISION.json](DECISION.json)、[STATUS.json](STATUS.json)：停在何处、四层证据结论。
3. [原合同](../../plans/plan_v9_2_online_20260930/PHASE0_CONTRACT_DRAFT.json)、[统计计划](../../plans/plan_v9_2_online_20260930/STATISTICAL_PLAN.md)、[任务计划](../../plans/plan_v9_2_online_20260930/TASKS.json)。计划文件保留原来的草案措辞与 MISSING 签署；后续用户执行指令记录在 [SESSION_AUTHORIZATION.json](SESSION_AUTHORIZATION.json)，没有代签历史文件。
4. [覆盖率校准](COVERAGE_CALIBRATION.json)、[独立验证](COVERAGE_VALIDATION.json)、[功效输入定义](power_001/MDE_INPUTS.json)、[完整功效表](power_001/MDE.json)。
5. [拟合期描述性指标](power_001/FIT_DESCRIPTIVE.json)、[小型逐日起报配对 MSE](fit_influence_002/FIT_PAIRED_CELLS.npz)、[参数选择](parameter_fit_001/PARAMETER_FIT_RECEIPT.json)、[输出订正选择](output_fit_002/SELECTION_SCORES.json)。
6. 当前 [online 实现](../../earthdelta/online/)、[worker 脚本](../../scripts/)、[测试](../../tests/)，以及 [旧审查](prior_review/REVIEW.md) 和 [T0 复用审计](prior_t0/FINAL_REPORT.md)。

## 统计复查重点

- 2pp 替代效应下，完整 Gate 功效的 0.2% / 97.6% 分歧究竟来自真实方差变化、训练/选择乐观性、零效应构造，还是实现问题？请重新核验，不能默认本轮解释正确。
- `coverage.py` 的合成依赖与 `power.py` 的经验零效应构造是不同模型。前者通过覆盖测试，不能直接为后者或真实非平稳数据保证覆盖。
- 54 / 26 天只有约 3.86 / 1.86 个非重叠 14 日块。功效外推到 270 天是否稳健，需要审查。
- 区间实际使用计划正文的估计值锚定扩宽公式。c=1 也可能包含原百分位区间之外的点估计，不能称作完全未修改的百分位区间。
- 所有天气技能数字来自估计/选择期，且选择期参与超参数选择。模型与方法之间的描述性差异不是独立测试结论。

## 包含什么、缺少什么

`SOURCE_COPY_INDEX.json` 逐文件绑定公开副本和本机原路径，副本逐字节不变。`PUBLICATION_MANIFEST.json` 记录本目录与相关代码/计划的校验和；`LOCAL_RUN_MANIFEST.json` 是原始运行清单，列出的完整本地环境不是全部随 Git 分发。

小型 NPZ **仅有已曝光的 2020 拟合指标**：`mse[80,9,24]` 为逐起报物理 MSE；`indices[80]` 为六小时时间索引；`is_select[80]` 区分估计期与选择期。臂序和格子序见 `FIT_INFLUENCE_RECEIPT.json`。使用 `numpy.load(..., allow_pickle=False)`；先池化 MSE 再开方，不平均逐日 RMSE。该文件 SHA256 与原收据一致。

原始天气场、模型权重、原始梯度、F0 全场缓存、大型 MC 中间分片与第三方克隆均留在 AFS。缺少这些材料时不能声称已独立重跑模型；可以用所附指标表和代码复算统计。绝对路径是历史身份，外部审阅需按 `SOURCE_COPY_INDEX.json` 映射到相对路径；不要直接执行仍指向原运行目录的 request。

`snapshots/` 记录实际执行时的源码哈希。`PUBLICATION_CODE_IDENTITY.json` 将本次提交的新代码与最后一份固定快照对账。提交和打包没有改动科学实现或冻结计划。

## 当前代码的停止边界

实现覆盖拟合与统计资格；完整分析期 actor 回放、T5、2021/2022 未获本轮放行。本次上传并不解除这一边界。下一阶段应由独立审阅决定最有价值的验证实验，保留原结果，并使用新的运行目录和明确的数据角色。

交给 ChatGPT 的完整指令见 [CHATGPT_NEXT_PLAN_PROMPT.md](CHATGPT_NEXT_PLAN_PROMPT.md)。要求它最后生成可供 Codex 接续执行的计划 ZIP；无法读取材料或产生附件时必须如实说明。
