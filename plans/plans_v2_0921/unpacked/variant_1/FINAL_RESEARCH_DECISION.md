# FINAL_RESEARCH_DECISION

日期：2026-09-21。依据：上一轮已完成的独立研究审查；本文件是执行冻结，不是新一轮文献结论。

- **Research verdict：`BOUNDED_CONTINUE_WITH_NARROWED_CLAIMS`。**我会再投入一笔有限预算，但只购买小型真实 bank 的“oracle—强静态—轻量合法策略”证据。
- **Implementation readiness：`NOT_READY_AS_IS_FOR_VALIDATED_PILOT`。**必须补足所选模型/输入/多步参照绑定、同一 Q 与实际数据入口；不要求机械关闭 B01–B15。
- 当前交付分支：`audit/round2-review-20260921@403b55db65f4c35c1a85d0794ad0de2765b07d96`；被复核源码：`fb767f7f6efbc428be39c9ad84f5905331d6e40f`。交付 HEAD 不是新实验成功证据。

## 保留的研究命题（最多两条，均为假设）

1. 对固定参考和有限编辑库，分开利用仿真响应与核验误差信号，是否在相同合法信息、标签资格与总资源下，提高合法策略的有限样本/成本效率。
2. 响应表示在其覆盖的目标 Q/时效内是否可复用。当前迭代只保留为后续资格，不实施新动作零样本泛化或目标迁移大实验。

`e0/du + 二次型`、低秩编辑、Gram、预算本身不宣布新颖性已成立；不声称参数编辑原则上优于反馈纠正。

## 下一笔预算买什么

先解决必要入口错误，再复用或训练一个合格小型非零 bank，冻结 registry，获得完整实际非线性候选开发缓存。用同一缓存测量：有没有好编辑、相对最佳静态是否有差异、轻量合法策略能否利用差异。不是买更复杂架构。

## 三项生存实验

1. **R2I-02/03：**真实有限候选 oracle + dev 选定静态/简单 regime + 交叉拟合轻量 direct-gain。Oracle PASS 仅为 `HEADROOM_PASS_ONLY`。
2. **R2I-04（条件解锁）：**双头/决策投影/direct-gain/同响应辅助 direct-gain，公平标签与总成本对照，四格 e/du 诊断。
3. **R2I-05（条件解锁）：**实际参数执行 vs virtual response correction vs feedback output correction，连同长时效 guard。

## 暂缓

JEPA、学习式 memory、动态 rank、谱 loss、Complexity Atlas、新 backbone、完整 P1/P2、无关重构、并发设施。仅串行、固定候选、无 activation replay。默认不消费 confirm。

## 决策

- **CONTINUE：**有真实动态空间，并有合法 out-of-sample 信号；之后分解至少在一个预登记标签/成本维度提供增量，实际参数执行也有有限资源保留理由。
- **STOP_CURRENT_DICTIONARY：**合格冻结 bank 的 oracle 上界不足。只否定该 bank/Q/预算/时效；singleton-only 结果不否定未测组合。
- **PIVOT_STATIC：**最佳静态解释主要收益，以 dynamic 增量不足的区间证据判定，不发明“解释了X%”阈值。
- **STOP_CURRENT_PLANNER：**有界合法策略挑战后仍无可利用增量；小策略一次失败不等于 Bayes 上限为零。
- **PIVOT_SIMPLE_POLICY / OUTPUT：**分解被 direct-gain 支配，或实际参数执行被同信息纠正器在预登记价值/成本约束下支配。
- **BLOCKED / INCONCLUSIVE：**正确性/来源无法保证，或区间跨门槛/功效不足；不能当科学失败，也不能自动扩资源。

所有效应阈值：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。delta_min 的价值依据必须独立于 MDE；功效估计不能自动决定什么值得做。

**初始授权只到 R2I-01。PASS 只生成后继资格建议，不自动执行下一任务。**
