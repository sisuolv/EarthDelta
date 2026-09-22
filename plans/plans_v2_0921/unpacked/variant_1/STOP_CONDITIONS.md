# STOP / CONTINUE / PIVOT 合同

本文件适用于一个迭代 R2I-01…R2I-05，不授权完整 P1/P2。不允许用新模块、临时候选、新指标或更换确认集规避停止条件。

## 0. 效应阈值与功效

所有尚未冻结的效应字段为：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。

**这个占位符不是“用pilot方差自动生成价值标准”。**delta_min表达增量值不值得承担复杂度/成本；其依据来自外部研究价值、比较对象与成本权衡，必须显式独立给出。MDE取决于alpha、power、N及相关性/方差。pilot只可帮助功效、资源和可分辨性估计。若无可辩护delta_min，只能给描述性结果，不虚构百分比。

阈值证书分别列：delta_oracle_vs_ref、delta_oracle_vs_static、delta_legal_vs_static、delta_factorization_vs_direct、output非劣性margin、long-lead guard、成本单位、价值依据；另列alpha/power/目标及最大N/block length/最大查看次数/MDE及模拟方法。不能以网格数、候选数或lead数当N。

对预登记增量 Δ 的区间 [L,U] 和价值门槛 δ：L>δ才达到该增量证据；U≤δ说明本设置没有达到门槛；其余INCONCLUSIVE。既定显著性/多比较/序贯规则不得在看confirm后修改。开发结果必须加DEV前缀，不冒称确认。

## 1. 决策表

| ID | 条件 | 正确决定 | 不能推出什么 |
|---|---|---|---|
| STOP-CORRECTNESS | 参考/归一化/输入/时效/Q错误；非有限值或数据不完整 | `FAIL_IMPLEMENTATION` 或 `BLOCKED_PROVENANCE`，先修入口；不可发布科学比较 | idea本身没有价值 |
| STOP-ASSETS | GPU、可信checkpoint、所需数据、上游backend、运行许可或cap缺失 | `BLOCKED`；CPU补丁可另报，后继不解锁 | GPU没跑是新代码bug |
| STOP-BANK | 未训练零bank，或达到已批准cap仍无法得到可执行且非退化的bank | `INVALID_BANK` / `STOP_CURRENT_SETUP` | 所有参数编辑无效 |
| STOP-ORACLE | 合格完整registry的oracle vs reference增量U≤δ | `STOP_CURRENT_DICTIONARY` | 尚未测试的不同字典/组合无效 |
| PIVOT-STATIC | oracle vs dev选定best-static动态增量U≤δ_dynamic，且静态训练资格成立 | `PIVOT_STATIC_ADAPTATION` | 用凭空的“解释收益X%”替代区间判断 |
| STOP-LEGAL | oracle有空间，但在预登记有限策略挑战与cap后，合法策略vs静态U≤δ_legal | `STOP_CURRENT_PLANNER` 或缩窄信息/场景命题 | 一次ridge失败证明Bayes策略上限为零 |
| DROP-FACTOR | 分解vs最强公平direct-gain无注册价值增量且无标签/成本优势 | `DROP_FACTORIZATION_CLAIM` / `PIVOT_SIMPLE_POLICY` | 简单有效策略也必然没用；仅“不显著”证明等效 |
| PIVOT-OUTPUT | 输出/feedback在注册margin内不劣且更省，或在共同成本下明显更好 | `DROP_PARAMETER_EXECUTION_NECESSITY` / `PIVOT_OUTPUT_CORRECTION` | 无限容量等价性已证明有限资源胜负 |
| STOP-LONG | 长时效/关键变量损害超过预先可容忍margin | `NARROW_HORIZON` 或 `STOP_EDIT_FAMILY` | 临时只报6h消除长期损害 |
| STOP-STATS | CI跨阈值、独立过程不足、缺候选、未冻结阈值或训练充分性不明 | `INCONCLUSIVE` / `INCOMPLETE` / `DESCRIPTIVE_ONLY` | PASS或确定科学FAIL |
| STOP-CONFIRM | confirm被查看后改bank/policy/Q/候选/阈值/数据选择 | `INVALID_CONFIRMATION`，记录曝光并停止 | 通过重命名同一数据恢复独立性 |
| STOP-BUDGET | 达到已授权训练/仿真/HPO/存储/查看次数cap | 停止，记录已获得证据与剩余不确定性 | 自动扩卡/下载/换更大模型 |

## 2. Oracle PASS 不等于研究 PASS

Oracle测 `E[max_a g]`，合法策略上限测 `E[max_a E[g|I]]`。前者通过只标 `HEADROOM_PASS_ONLY`，允许提出下一小型可预测性检验，不宣布双头/部署有效。

singleton-only registry的FAIL只覆盖singletons。若想检验H/联合组合，必须事先登记并实际执行；有限端点恒等式不要求线性，局部Ra才要求组合近似。

## 3. CI / 缺失 / 失败

不静默删除困难起报或崩溃candidate；reference失败保留在attempted分母。部分候选结果只能称tested subset lower bound，不能给complete oracle PASS。未知损失不填0。合法policy若有预登记fallback，记录其真实运行结果与成本。

最多按预登记扩样规则增加独立过程，达到cap仍不决就结束本迭代。不能把MDE降低当作收益忽然更有价值。

## 4. 执行权限与研究门槛相互独立

当前只有R2I-01有代码修改范围授权。每个任务结束保存decision.json、更新工作副本checkbox，然后停止。PASS/DEV_PROMISING只产生eligibility，后继还须owner明确批准、预算完整。修改自身状态文件不能构成自我授权。

任务R2I-03若oracle强而cheap无信号，只允许提出**一次**R2I-04有独立cap的更强检验；未授权不得执行。R2I-05结束一律停止本迭代。
