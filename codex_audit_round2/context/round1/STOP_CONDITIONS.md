# STOP / PIVOT / BLOCKED 合同

## 先分清四种情况
- **BLOCKED**：资产、许可、环境、数据角色、效应阈值或预算未确认。缺 GPU 不是科学失败。
- **FAIL_IMPLEMENTATION**：错误符号、指标、无效候选、独立S0不等价、泄漏等。先修复，已有数字不可作为正式结论。
- **INCONCLUSIVE**：误差区间跨过最小有用差异、独立过程不足或功效不足。在预登记cap内补样；cap到达即停止，不擅自改阈值。
- **STOP / PIVOT**：一个明确且合格的设定在充分精度下被否证。停止的是当前字典/当前分解/参数编辑主张，不声称所有可能方法都无价值。

## 阈值不能照抄2–3%
主损失先定 L（非混合变量原始单位的随意均值）。报告配对变化 Δ=L_base−L_method；相对量仅当基线分母稳定且非零时使用。主参考为 Fs，另比较 best-static 的增量 Δ_dynamic。

P0-04建立 threshold_certificate.json，必须含：
1. 固定主variable/lead与全局辅助指标；建议先24h主、72h guard，是否采用由实际数据/成本确定。
2. 数值重复运行/实现parity产生的误差范围 δ_numeric，不用增大tol掩盖模型差异。
3. 最小有用效应 δ_practical：解释在同成本下为何值得，或通过静态模型的成本–技能曲线估计达到同效应所需成本。没有外部业务标尺则写“研究性SESOI”，不能冒称业务阈值。
4. 最终 δ_min 至少高于数值地板；δ_dynamic另定。**不得把标准误本身当业务效应阈值**。不能由独立确认结果选δ。
5. Pilot过程级paired variance、独立时间块定义、预期功效和最大样本/计算cap。可预设alpha=.05、power=.8作为分析政策并说明；用pilot方差规划样本，最终使用配对块bootstrap及敏感性分析。
6. Per-variable/long-lead non-inferiority margins：用相同校准逻辑，避免全局均值掩盖某关键变量恶化。

`configs/survival_template.json` 的效应和资源字段故意为 null：没有本地pilot，不能制造精确阈值。Codex须通过P0-04的实际测量和研究理由填写；未填写时所有科学放行均为BLOCKED_THRESHOLDS。

## 判定函数
设 [lo,hi] 为预登记比较的过程级置信区间，δ为事前SESOI：

    if resource_missing or threshold_unset: BLOCKED
    elif invalid_contract_or_incomplete_comparison: FAIL_IMPLEMENTATION
    elif lo > delta: PASS_EFFECT
    elif hi < delta and planned_precision_met: STOP_CURRENT_CLAIM
    else: INCONCLUSIVE_WITHIN_CAP

这不是把 p>.05 当无效。若确实显著恶化（hi<0），可先暂停部署；无足够精度时不宣称等价。

## 各门的科学停止条件
| ID | 条件 | 行动 |
|---|---|---|
| STOP-0 | 独立S0/metric/合法候选/时间合同未通过 | STOP_FORMAL_EXPERIMENT；修复不计科学成果 |
| STOP-1 | 合格非零字典、全部注册候选、确认精度足够时 oracle-Fs 的 upperCI<δ_gain | STOP_CURRENT_DICTIONARY；不训练复杂响应模型 |
| STOP-2 | oracle-beststatic 的 upperCI<δ_dynamic | PIVOT_STATIC_ADAPTATION；不要用接近0的“解释百分比”阈值 |
| STOP-3 | predicted du不优于zero/mean且4格定位响应为瓶颈，达到预登记fitcap | STOP_CURRENT_RESPONSE_REPRESENTATION；仅准一项事先界定的尺度/描述符诊断 |
| STOP-4 | true du也救不了pred e0；可部署收益不足 | PIVOT_ERROR_ESTIMATION或缩小可预测目标；不继续增加编辑器 |
| STOP-5 | 同teacher/data/budget directgain与双头无实质区别 | 删除分解独立novelty声明；只有预登记的label/action迁移成立才保留相关价值 |
| STOP-6 | feedback或virtual输出纠错在效应与成本上支配参数编辑且CI支持 | PIVOT_OUTPUT_TRAJECTORY_REPAIR，不再宣称参数必需 |
| STOP-7 | 长时效关键变量的退化超过margin，反复独立过程出现 | STOP_LONG_HORIZON_CLAIM；修训练分布后用新确认数据，不覆盖旧结果 |
| STOP-8 | 实际成本含preview/运行后无收益或超过已定资源cap | 停止“高效”主张，保留机制诊断范围 |

STOP后输出：失败证据、已执行量、未执行量、当前待办锁定状态。不得自动重启第二backbone、JEPA、memory、谱损失或扩大数据。
