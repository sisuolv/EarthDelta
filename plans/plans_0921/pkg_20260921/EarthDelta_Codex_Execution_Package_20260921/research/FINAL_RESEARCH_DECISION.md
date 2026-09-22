# FINAL_RESEARCH_DECISION

Decision: CONTINUE_CONDITIONALLY_WITH_P0; NOT_A_NOVELTY_OR_ACCEPTANCE_GUARANTEE.

## KEEP — 只保留两项方法核心与一个实证贡献候选

C1. 学习固定天气预报器对明确、可逆、有限参数干预的条件轨迹响应；在执行候选前生成响应预测。贡献对象是该响应代理的泛化和决策效用，不是LoRA本身。

C2. 在共同线性物理输出空间中分离 e0 与 du，并使用统一指标进行选择；检验无需真值的 du 仿真监督是否带来标签效率或未见动作组合优势。分解的必要性是待验证假设，恒等式不作贡献。

C3. 在同信息、同参考、真实执行成本和强反馈式输出基线下，提供完整的可实现收益、误差来源及长时效稳定性证据。作为实证贡献，而不是第三个大型模块。

## HYPOTHESES

H1. 非零、训练期冻结字典有实质性可利用上限，而且相对验证期选定的静态/气候分区策略仍有动态余量。
H2. 仅凭允许的context能预测足够准确的候选响应与决策相关误差；双头在至少一个预登记维度优于同容量direct-gain（标签预算/未见组合/成本—技能）。
H3. 参数干预的归纳偏置相对强输出反馈，在预登记的有限样本/预算设定中有可重复价值，不牺牲后续时效与普通天气稳定性。

## DEFER / REMOVE FROM MAINLINE

JEPA、learned memory、dynamic rank、spectral training loss、Complexity Atlas、第二backbone、全物理约束、RL scheduler、在线JVP、全参数Jacobian、跨模型零样本迁移全部DEFER。保留源代码与已有测试，不删除历史。

已有简单历史context与zero-edit保留。memory输入首轮设零且所有方法一致；已核验近期误差可另列同权限简单对照，不能把6h情景标签当真实可得性。

## 新增方向只选一个

选择 trajectory-level response planning，并以同字典未见幅度/窗口/组合为可检验增强；不是再造一个名为world model的大骨干。风险校准仅为P2第二条可选路线。已经存在多lead head，因此新工作是实际trajectory采集、跨lead共同决策、持出动作验证，不是声称新造multi-head。

## 立即不再使用的声明

“首次先预测再编辑”；“新二次收益公式”；“参数编辑独占动态一致性”；“router按构造不能迁移新预算”；“不存在先例”；“半正定H保证安全”；“保存了数据/195 tests所以方法已验证”；“后段reference token零成本”。

## 阶段顺序

P0来源/代码合同 → 真实S0 → 非零bank → finite oracle/static余量 → 四格误差分解+direct-gain → 强残差挑战 → 生存决策。
只有核心P0通过才能启动P1完整训练/确认；P2只在P1后；P3登记但不执行。

## STOP / PIVOT

见根目录STOP_CONDITIONS.md。技术BLOCKED不是科学STOP。oracle不足先停止当前字典而非宣称整个理论方向无效。静态可解释全部实质收益则转静态适配。direct-gain无劣势则撤下分解为主贡献。强残差在共同成本前沿上占优则转输出订正。置信区间不够窄则INCONCLUSIVE，不用任意2–3%阈值强行通过。
