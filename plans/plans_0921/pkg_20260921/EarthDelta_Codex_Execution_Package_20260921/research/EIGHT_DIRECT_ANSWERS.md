# 用户最后八个问题的明确回答

1. **novelty是否明显更强？** 研究问题比physics-state LoRA明显更具体：候选编辑的效果先预测、再选择。真实方法优势尚未建立，不能称已经是足够投稿的新颖性。当前源码尚缺独立真实S0和合格天气实验。
2. **最有价值哪几个？** 第一是逐候选、多时效response prediction；第二是e0/du监督来源分离能否带来可测的label/action泛化价值；budgetedplanning是验证载体。LoRA、FSO代数、counterfactual命名不是创新本身。
3. **最容易被说组合？** VI-MoLE式价值路由 + FSO/PQC影响估计；或 Green’s-function响应 + learnedreferenceerror；或 AdapterBanks/CCM参数动作 + 通用surrogate。
4. **最大科学风险？** 先是可利用dynamicheadroom不足；之后是e0方向难预测，而一旦e0预测好，简单feedback输出订正同样受益，参数必要性未必存在。最大工程风险是S0假等价和metric/candidate合同漂移。
5. **立刻跑哪个？** 先修完真实S0与非零bank准入，然后完整有限候选oracle相对Fs和validationbeststatic的配对余量表。这比先训练JEPA或更大head决定性强。
6. **只保留一个创新？** 保留candidate-conditioned finite trajectory response surrogate，并用actualintervention验证泛化及选择效用。
7. **只新增一个？** 选trajectory-level response planning，重点是同bank未见幅度/时窗/组合与跨lead共同收益，不是另造已存在的multi-leadhead。比直接改名worldmodel/Jacobian更贴代码；不先加uncertainty与spectrum。
8. **投稿还缺什么？** 独立upstream真实bridge、非零字典上限和dynamicgap、跨fit/confirm的du/e0预测与4格、同信息directgain/router/feedback对照、actual组合和多lead稳定性、完整成本与未见动作证据。只有测试数量和已下载数据不够。

这些回答是依据定向源码审查和所列原始论文给出的研究判断，不是对投稿录用的保证。
