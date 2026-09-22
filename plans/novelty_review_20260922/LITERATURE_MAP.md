# EarthDelta 文献地图：哪些方法值得借鉴，如何转成实验

检索日期：2026-09-22 UTC。检索使用 arXiv API 的关键词与论文 ID 查询，并取回原始 HTML/PDF。关键词涵盖天气基础模型适配、adapter utility、参数/干预响应、模型合并、world model、误差订正，以及 CV/NLP 的效果预测。主要检索窗口为 2025-01-01 至 2026-09-22，另有已知近邻查询。

这是围绕 EarthDelta 研究决策的定向检索，不是穷尽式综述。关键词查询存在噪声，返回的条目未全部精读。选入结构化索引的 26 篇中，18 篇阅读了方法或相关正文段落，8 篇仅用于摘要级定位；具体到章节的范围写在 [literature_review.json](literature_review.json)。下载全文不等于完成阅读，也不等于独立复现性能。本次没有核验每篇的会议接收状态，近期预印本不能直接视为已确立结果。

原始检索与下载记录保存在 [evidence/](evidence/)；JSON 保留版本、API 日期、来源、阅读范围和本地正文 SHA256。个别论文 ID 的数字顺序与 API 发布日期不直观，日期按源元数据记录，不从 ID 猜测。

## 1. 最建议精读和借鉴的六篇

### L01. DART，2026-09-17，CV / 视频扩散

[DART: Distillation-Aware Reparameterization for Training-Free LoRA Reuse in Few-Step Video Diffusion Models](https://arxiv.org/abs/2609.20051v1)

阅读：Section 4 完整方法，以及 Sections 3/5 的诊断、组件比较与限制。

它把 LoRA 的权重兼容与实际功能效果区分开：新生成日程下，静态几何相近也可能有不同响应。方法先做坐标运输，再用配对前向有限差分测量 channel response，以正则化最小二乘拟合目标日程的响应描述。探针无法识别的方向保留原值。

**借鉴：** EarthDelta 在幅度、hold 或 lead 改变后，用少量共同初态的实际响应测量判断能否复用原预测器；评价功能保留与负迁移，而不只检查 delta 范数或合并等价。

**范围：** 原文是视频日程迁移，不是天气；局部响应模型不能保证所有轨迹恢复。原文 calibration 贡献较多质量收益，但部分 adapter 仍有负功能效果，且校准不是免费计算。不能将其直接当作“无数据/无成本就能迁移”的证据。

### L02. L-State，2026-09-08，NLP / 训练响应

[Target-Independent Micro-Interventions for Predicting Training Response Across Language-Model Families](https://arxiv.org/abs/2609.08618v1)

阅读：Sections 3.1-3.3、3.6、4，以及相关结果描述。

它从同一 checkpoint 的独立副本执行四种标准化微训练，在共享能力空间记录响应。既比较自由 ridge 读出，也比较保留脉冲几何的算子读出；响应误差、方向与最终动作决策分开评价。跨 family 的共享动作坐标被明确当作需要检验的假设。

**借鉴：** 共用 Fs、共同探针、先冻结预测再揭示目标动作结果；用自由回归头挑战结构头，最终看动作收益。结构应在实验中证明价值。

**范围：** 原文预测的是短程训练后的能力改变，EarthDelta 预测的是固定编辑的天气轨迹改变。其个别 family/action 的方向反转很有提醒意义：跨域代数形式相同，不代表坐标语义必然相同。

### L03. W2T，2026-03-16，NLP / adapter 表征

[W2T: LoRA Weights Already Know What They Can Do](https://arxiv.org/abs/2603.15990v1)

阅读：canonicalization 与 encoder 相关方法。

对 LoRA 更新做 QR 加小矩阵 SVD，再编码 rank token 与位置，学习 adapter 行为/性能表征。核心启发是描述实际更新，避免学习因子分解的任意性。

**借鉴：** EarthDelta 先使用 `scale * B * (A * Z)`、模块位置、更新范数，以及共同探针响应作为动作描述。比较 ID-only、weight-only、behavior-only 与两者组合。

**范围：** 简单 sketch 是本计划的轻量实现建议，不是 W2T 原方法。四专家不足以训练并证明大型权重表征模型；SVD 的符号/重根约定也不能被一句“唯一规范化”掩盖。

### L04. LiST v2，2026-08-31 更新，NLP / VLM

[LiST: Local-Simplex Test-Time LoRA Fusion](https://arxiv.org/abs/2608.22370v2)

阅读：Sections 3.2-3.4、相关消融和限制。首次取回失败后重试成功，已不只是摘要级判断。

参数 PCA anchor 和 support prompt 的行为向量构成联合描述，用于局部 adapter 检索与先验。在线采用 branch-preserving fusion 和 CMA-ES 搜索，以代理能量、几何约束和随机一致性决定接受或回退。

**借鉴：** 接入新专家时比较联合描述；将 no-edit/回退作为显式动作；测退化频率而不只测均值。

**范围：** 在线搜索使用多个真实模型前向，与 EarthDelta 的无在线候选天气 rollout 目标不同。其“safe”是特定代理约束下的接受规则，不能拿来承诺时间分布漂移下无损害。

### L05. HyperFix，2026-08-11，CV / 模型合并

[HyperFix: Combinatorial Nonlinear Correction for Task Vector Merging](https://arxiv.org/abs/2608.11499v1)

阅读：完整 Method，部分局部理论与实验说明。

从 task-vector Gram 行构造集合 embedding，用小型 hypernetwork 预测低秩权重修正；训练 singleton/pair/triple，再测试更大子集。原文的对象是权重合并后的能力保留。

**借鉴：** 用少量组合先识别非加性，再决定是否需要交互模型；避免为每个组合单独调参。

**范围：** EarthDelta 的首选是输出响应残差，不是再生成新权重。原文局部平滑/小扰动分析并不意味着 72h 大气 rollout 可以由低阶组合精确决定。

### L06. Proxy Policy Steering v3，2026-09-13 更新，视觉机器人

[Proxy Policy Steering](https://arxiv.org/abs/2609.09148v3)

阅读：Sections 4.1-4.3、相关消融、限制和 Appendix D 的假设说明。

先把参考代理蒸馏到冻结基础模型的访问状态，再从参考代理初始化任务代理；二者的速度场差分用于逐步 steering。共享近似误差是构造动机，实际效果通过消融检验。

**借鉴：** 给 EarthDelta 增加真正有竞争力的差分蒸馏/逐步反馈订正；检验直接预测响应后是否还有必要执行真实权重编辑。

**范围：** flow-matching 推导不直接适用于 Stormer；共享误差抵消不是任意两个小网络的自动性质。API 摘要与取回正文中的总体增益数字不同，本文未转述该数字，也未将其作为本项目的性能预期。

## 2. 天气、PDE 与执行前价值预测的关键近邻

| ID / 论文 | 版本与日期 | 本次阅读范围 | 影响研究判断的内容 |
| --- | --- | --- | --- |
| L07 [WeatherPEFT](https://arxiv.org/abs/2509.22020v2) | v2，2026-06-17 更新；首次 2025-09-26 | Sections 4.1-4.2 | 从 embedding 权重生成 task prompt、Fisher/annealed noise 参数选择；天气 PEFT 有明确先例。本次已读到 v2 方法，补足旧复核的材料缺口 |
| L08 [TaCT](https://arxiv.org/abs/2603.19325v1) | 2026-03-17 | Sections 3.2-3.3 | SAE 概念激活条件化 adapter/LoRA；天气中的按状态干预已有近邻，可借作简单条件门 |
| L09 [VI-MoLE](https://arxiv.org/abs/2608.02528v1) | 2026-08-03 | Sections 3-4 | 激活专家前预测 prefix risk 并分配预算；需要突出响应复用的实证价值，而非泛泛“先预测收益” |
| L10 [CCM](https://arxiv.org/abs/2605.14546v1) | 2026-05-14 | Section 3 | 同一 anchor 的 PDE 专家方向与组合；支持共同 Fs 合同，不能为 arbitrary expert 自动赋予物理因果语义 |
| L11 [CLAW](https://arxiv.org/abs/2609.12278v1) | 2026-09-10 | Section 4 | transition context 条件化、分块生成 LoRA；可作为后续 direct hypernetwork 参考，当前不宜为它重训基础模型 |

VI-MoLE 的正文还包含评估/发布协议内容；本文没有把这些当作已复现的大规模数值成绩。对风险差的解释也需谨慎：两个上界的差不自动构成真实改进的下界。

## 3. 用来改进实验与诊断的工作

| ID / 论文 | 日期 | 本次阅读范围 | 具体用途与限制 |
| --- | --- | --- | --- |
| L12 [Sampling Headroom Is Not Selection Gain](https://arxiv.org/abs/2609.13257v1) | API：2026-09-06 | 选择器、成本与限制相关段落 | 区分候选空间和合法可实现收益；计全部计算；有回顾性分析边界 |
| L13 [Measuring the Value of World-Model Updates](https://arxiv.org/abs/2609.10954v1) | 2026-09-10 | Method Section 3 | 同状态/随机性下 update/hold 分支；失败尝试留在账本，适合 Fs 稳定性设计 |
| L14 [Sandwich-Residuals](https://arxiv.org/abs/2609.21740v1) | 2026-09-18 | 选读方法、预算对齐评估与限制 | 冻结视觉模型的小型残差适配提醒我们认真比较反馈订正；未逐式审核全篇 |
| L15 [Learned Atmospheric Critic](https://arxiv.org/abs/2609.18381v1) | 2026-09-16 | critic 构造、评估与限制 | 可作事后真实性诊断；边际真实性与逐样本准确率是不同对象，不能替代 native objective |
| L16 [MergeProbe](https://arxiv.org/abs/2606.19549v1) | 2026-06-17 | Sections 2.4-4、实验来源、部分附录 | 早期几何/梯度/激活预测合并效用，按 adapter/domain 切分；部分结果明示为受控 simulator/pilot，不能当成熟大规模性能证据 |
| L17 [Predicting Where Steering Vectors Succeed](https://arxiv.org/abs/2604.15557v1) | 2026-04-16 | 选读 formulation、4.2 与层深控制 | 读出对齐可帮助选择干预层；可作为训练侧 block 筛选思路，不能等同天气收益 |
| L18 [Look Before You Steer](https://arxiv.org/abs/2609.22782v1) | 2026-09-19 | Sections 3-4.1、边界说明 | 几何描述与实际行为探针对照；实验 feature 筛选含因果输出过滤，不能把整个流程描述为完全不用前向 |

## 4. 仅摘要级定位，不作为细节结论依据

| ID / 论文 | 日期/版本 | 可借鉴的方向 | 阅读边界 |
| --- | --- | --- | --- |
| L19 [ORCA](https://arxiv.org/abs/2606.14222v1) | 2026-06-12 | 黑盒误差上下文与在线残差适配 | 全文已下载；本轮未细审方法，实现前需读 |
| L20 [RATL](https://arxiv.org/abs/2609.03937v1) | 2026-09-03 | 训练侧 residual memory 与路由 | 全文已下载；因果可用性未逐路径核验 |
| L21 [Text-to-LoRA](https://arxiv.org/abs/2506.06105v2) | v2，2025-06-09 更新 | 描述符生成 LoRA 的经典近邻 | v1/v2 均取回；本轮只用摘要定位 |
| L22 [Stochastically Perturbed Weights](https://arxiv.org/abs/2609.08412v1) | 2026-09-08 | 天气权重扰动集合，作为以后概率预报方向 | 不把集合/CRPS 收益转述为确定性精度收益 |
| L23 [ARC-STAR](https://arxiv.org/abs/2605.22222v3) | v3，2026-05-25 更新 | 冻结 PDE host 的订正与预算路由 | HTML 未取回，只读 API 摘要 |
| L24 [Compact but Moving](https://arxiv.org/abs/2609.21787v1) | 2026-09-18 | rollout 中干预相关几何的运输 | PDF 已下载但未解析；原文干预是 latent state，不是 LoRA 参数 |
| L25 [Adapter Banks](https://arxiv.org/abs/2609.17042v1) | 2026-09-15 | recurrent adapter options 与组合运动控制 | 只用摘要定位，不简化成纯 RL 回报路由 |
| L26 [FARM](https://arxiv.org/abs/2609.11445v1) | 2026-09-10 | 冻结 predictive features 上的 failure readout | 只用摘要；完整特征提取费用须另算 |

这些条目可帮助后续选择具体基线，但未被用作当前核心方法设计的唯一依据。无需为补齐它们先停止天气 pilot。

## 5. 阅读后的行动优先级

1. 立即采用：共同初态与配对响应、自由/结构读出强对照、完整成本、动作结果与响应误差分报。
2. 主实验采用：决策分量表示、向量收益基线、目标/动作联合留出、virtual edit 与反馈订正。
3. 主线有效后采用：行为描述的新专家接入、按时效重校准、有限组合交互残差。
4. 暂缓：大型权重 encoder、端到端 hypernetwork 重训、JEPA/RL/memory 组合、learned critic 主目标、跨多个 backbone 的全面推广。

重叠用于定位继承关系和强对照。项目是否值得做，最终取决于这些实验能否产生有用的新能力、可靠的收益或可测的成本优势。
