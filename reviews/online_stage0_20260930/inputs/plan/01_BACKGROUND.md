# 研究问题与结论边界

## 已有结果的身份

当前代码基准是 `probe-headroom-20260927@d55ad70`，不是 main@d55ad70。FP-05b 检验的是 Fs+4个 expert 的候选选择，Fs 只经历 8 个起报、32 次更新。含 F0 的按24h事后选择约 +0.16%、沿用同一选择到72h约 −0.44%，是旧状态标准化平方损失口径，不能写成标准变量 RMSE 改善。

这些历史数字来自已保存审计，不是本轮重新计算。它们只支持“在该候选库和训练强度下没有充分收益”。预注册的 2020 headroom 探针未完成，P4/Step2/P5 等仍有缺陷；本版正式将其退出当前执行范围，保留原件，不把未运行结果写成负证据。FP-05b 冻结结论不改写。

## 理论只能解释动机

平方损失下，有限参数模型的一阶驻点通常约束的是梯度投影，并不推出条件残差为零。例：X 为标准正态、Y=X²，仿射模型的总体最优预测为1，残差仍依赖 X。旧方向没有在原理上被排除。

若 r=Y−F0，I 是部署时可用的完整信息，m=E[r|I]，则任意 I 可测预测改变量 d 的条件期望收益为 `||m||²−||m−d||²`。这只给无限表达力输出订正的理论界，不等于 DABC/OCL 一定达到；也不能用只含 I 一部分的特征去替代 I。本协议实际残差定义为 e=F0−truth，订正时做减法，勿混淆两种符号。

## 可判别的假设

| 问题 | 预注册比较 | 能支持什么 |
|---|---|---|
| H1a 慢漂移可跟踪 | DABC vs OBC/F0；ewma_g vs static | 固定偏差会过时；不是短期记忆的独立证据 |
| H1b 新鲜反馈增量 | OCL-fresh−DABC；raw lag1−raw ewma_g；lag1c−同样中心化的delay30 | 在这些有限基线之外是否有增量；仍不穷尽所有漂移模型 |
| H2 参数空间价值 | lag1c vs OCL-fresh/DABC/OBC，且同样的标签可用性 | 当前编辑族对有限输出模型的额外价值；不是胜过无限输出订正 |
| H3 跨时效风险 | 6变量×4时效的共同掩码和非劣界 | 注册变量/时效上的有限证据，不证明所有通道或极端天气安全 |
| H4 在线 vs 近期重拟合 | 以后 T5 的 online vs sliding_refit | 阶段0单步编辑不能回答，暂不排期 |

主臂固定 lag1c；中心化仅表示减去冻结原始梯度均值，**不保证与 static 正交**。OCL-fresh 提供全部六变量的1–4步反馈，不只当前目标变量的残差。阈值、比较族及变体数在试点前冻结。

## 相关工作与业务边界

DABC 属于已有的衰减平均偏差订正思想。Cui、Toth、Zhu、Hou 的 *Bias Correction for Global Ensemble Forecast*（2012，DOI 10.1175/WAF-D-11-00011.1）描述了 NCEP/CMC 的衰减平均和 Kalman-type 校准；本文不能把该思想当新方法。[NCEP 保存的原论文](https://www.emc.ncep.noaa.gov/gmb/yzhu/gif/pub/WAF201204_Cui.pdf)

本实验只是再分析时钟回放。ECMWF 明确说明 ERA5T 日更新落后实时约5天；cutoff(t)=t 不代表业务标签到达。12h 缓冲也不消除输入再分析场的前瞻。[ECMWF 2026-09-21 说明](https://www.ecmwf.int/en/about/media-centre/news/2026/era5t-reanalysis-data)

TAFAS/OnlineTSF 的本地版本与 license 尚未核验，计划阶段只保留相关工作待办；PETSA、ORCA、INC 的已读设计见旧 T0 报告。不能抄用 PETSA 的 NC/SA 代码或无 license 的 ORCA 代码。T5 前做范围明确的相关工作核查，且仍不写“首次”。原 v9.1 表里未在本轮核实的其他论文不升级为已验证来源。

新意只能是待检验的比较性贡献：明确时间协议、对齐标签来源、与强输出/重拟合基线相比的增量证据。当前 SCIENTIFIC_SUPPORT 与 NOVELTY_SUPPORT 均未建立。
