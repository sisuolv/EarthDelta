# EarthDelta 深度研究审计与后续决策报告

**审计日期：2026-09-20（America/New_York；对应部分 UTC 记录为 2026-09-21）。**

**固定源码：** `sisuolv/EarthDelta` / `main` / `4fe55a7af90ea92f62a3232a571af92bfbd6114d`。

**结论先行：当前版本不宜直接以“新公式 + 新 QP + 动态 LoRA”写成强方法论文。值得保留的是一个尚待真实实验验证的科学问题：在未来核验未知时，能否学习足够支持编辑决策的轨迹响应结构，并以更少核验标签、可复用目标或更稳健的多步收益，胜过直接风险预测、条件路由和输出订正？**

本次通过连接器读取固定提交的核心源码、关键测试范围、v6 研究规范/评审，以及 Notion 主页面和相关容量笔记，并独立核查 20 项最近邻/理论先例。不是完整仓库逐行形式验证。远程工作区连接失败，本地无法取得可运行的完整仓库，因此**没有复跑全套 pytest、加载真实天气 checkpoint、扫描 ERA5 bytes 或开展任何天气实验**。下文区分代码事实、项目自行报告、外部文献、分析推论和下一阶段建议。详细读取范围在 `evidence/source_ledger.json`。

最终工程交付从 `START_HERE_FOR_CODEX.md` 进入；本轮唯一解锁任务为 P0-01。新增 CLI 是编码合同，不是已经在仓库存在或运行完成的程序。


## Part I — Current EarthDelta Reconstruction

现有代码是一组可组合原语，并存至少三条路线：

1. `cached_responses` 得到有限候选的真实非线性响应；`paired_targets` 产生 e0/du/gain 标签；未来可用 `ComposedPredictionHead` 预测 e0/du，再由 `select_plan` 选择。
2. `central_response` 测局部 R；`ResponseGeometry.from_error` 计算 b、H；`box_candidates` 提供离线教师；`BoundedProgramHead` 学系数。
3. `InteractionUtilityHead` 预测 b/H，再调用 `plan_from_prediction` 做支持枚举与盒约束优化。

路线 2/3 不能冒充路线 1 的有限非线性响应评估。`ComposedPredictionHead` 还包含 memory 接口和 score calibration，与“纯解析收益”的叙述不同。没有发现主线的真实 weather-head 训练脚本/训练权重/闭环选择结果。

#### 参数干预的实际语义

在 attention projection 的输入 z 上，输出增量是 sum_k a_k B_k A_k z。a 在该次执行内与 token 无关时，它等价于 W_eff=W+sum_k a_k B_k A_k；无需永久改写 checkpoint。默认作用前 4 个 6h 步，之后 a=0，但**从已经被改变的气象状态继续 rollout**，并非跳回未编辑轨迹。桥接默认返回末步。若使用旧 ProgramSpec 的 token/region 条件系数，则是局部条件适配，不能笼统称全局权重矩阵编辑。

#### 信息与组件边界

| 类别 | 现有原语 | 合法用途 |
|---|---|---|
| Offline，不用未来标签 | frozen forecast、编辑 forecast、cached_responses、central_response | 收集确定性模型响应；但耗计算并非免费 |
| Offline，用未来标签 | paired_targets、ResponseGeometry.from_error、teacher.verify_candidates | e0/收益标签与 oracle 分析 |
| Learned | ReferenceErrorHead、EditResponseHead、InteractionUtilityHead、BoundedProgramHead、组合头 | 类定义存在不代表已训练 |
| Analytic | gain、Gram geometry、盒约束/支持枚举 | 代理空间解析；不是未知真实收益的精确计算 |
| Runtime | controlled_rollout + 未来计划连接的 predictor/selector | issue-time 输入不含本次未来 truth |
| Memory | VerifiedRecord、eligible_records、EWMA/gated delta 原语 | 只能使用已经 available 的核验记录；P0 暂缓 |
| Physics | 系数谱诊断和节点检查；regridding | 不是已完成的 KE/anisotropy/SHT loss pipeline |

### 最准确的范式

**以 PEFT 作为动作空间、以 forecast sensitivity / response surrogate 作为模型、以 decision-time planning 作为推理规则的摊销控制原型。**

它不是通常意义的 test-time gradient adaptation，因为主线服务时不做梯度更新；不是传统 model editing，因为本次正确目标未来未知；可以视为 model-based planning，但“world model”只是描述，不自动带来新贡献；模型内干预不等于对真实大气的因果干预。直接 b/H 头接近 decision-focused prediction；直接系数头接近 conditional router/hypernetwork。

---

## Part II — GitHub vs Notion Evolution

Notion 主页面中的原始方案围绕 **physics-state-conditioned parameter-efficient adaptation**：先描述气象复杂度与状态，再调整适配容量/参数。当前代码核心转为 **在执行干预前，预测它如何改变预报，并据此比较收益**。这是研究对象的变化，不只是模块替换。

[原始 EarthDelta 页面](https://app.notion.com/p/EarthDelta-3dcff220ec698014882ec73e8d9b26ad) 是 idea provenance；其中阅读笔记不能替代原论文，也不能证明模块已经接入。`plans/` 的 v6 已承认 FSO、Green’s-function 和价值路由先例，但其“剩余空白”结论仍须独立检索，而不是引用自身计划完成 novelty 证明。

当前最重要的不同在于：旧方案生成适配参数，现方案尝试建模其**效果**；旧方案偏复杂度/容量匹配，现方案应聚焦**可预测收益与何时不编辑**；旧谱损失/神经记忆是可选设计，不是主线必须继承的研究承诺。



| 模块 | 对应明确 failure mode | 不加入缺什么 | prior-work / novelty 判断 | 决定 |
|---|---|---|---|---|
| ERA5 Complexity Atlas | 不知道 headroom 集中在哪些 regime | 缺细分误差分析，不妨碍主模型 | 数据分层/诊断不自动是方法创新 | Analysis only |
| Physics-state encoder | 简单上下文看不出 flow-dependent gain | 可能预测不足 | 条件适配已广泛存在；先冻结主干特征 | Minimal core input；新 encoder 后置 |
| Memory | 近期有合法核验误差却未利用 | 缺在线偏差反馈 | ML-PQC/校准已有相近思路 | EWMA baseline；神经 memory DEFER |
| JEPA / latent state | 输入表示限制响应预测 | 未确认前不能假设必须 | 潜预测不保证干预保真 [R18] | DEFER |
| Dynamic rank | 真正算力瓶颈和容量不匹配 | 仅固定容量，未必造成损失 | Conditional PEFT/库路由已有先例 | P3 |
| Spectral features | 小尺度误差/相位被普通特征漏掉 | 缺多尺度诊断 | 物理特征不等于新问题 | Analysis only，必要时 Optional |
| Spectral loss | 训练过度平滑且能证实 | 可能仍有高频损失 | 不能以新损失掩盖 oracle 不足 | DEFER |
| Anisotropy diagnostics | 编辑破坏方向结构 | 缺方向安全检查 | 属评测结构而非核心算法 | Analysis only |
| Kinetic-energy spectrum | 编辑破坏尺度能量分配 | 缺谱安全检查 | 能量不能替代相位/风场验证 | Analysis only |
| Weather-regime prototypes | 银行缺异质性 | oracle 可能不足 | CCM/条件专家已有谱系 | 可用于唯一预登记 bank fallback |
| Long-term memory | 长期漂移且短历史不足 | 当前未发现该 failure | 没有证据前是模块堆叠 | Future work |
| Hierarchical edit dictionary | 已证实大 bank 搜索/训练成本过高 | 小 bank 不受影响 | 层级检索是实现选择 | Future work |
| Gated incremental capacity | 容量过写/长期上下文瓶颈 | 当前主线没有该需求 | Notion Proteus 阅读笔记不是天气实验 | Remove from mainline |

这里的 DEFER 不要求删除已通过测试的通用模块；只是禁止把它们作为新一轮 P0 的前置或默认训练项。

---

## Part III — 2024–2026 SOTA Landscape

#### 最新威胁与历史基础应同时看

按首次公开时间，2026 年值得优先核查的是 CCM（5 月 14 日）、AdaWeather（6 月 1 日）、VI-MoLE（8 月 3 日）、GeoQ（8 月 21 日）、Intervention Gap（8 月 30 日）、SPW（9 月 8 日）和 CABRA（9 月 15 日）。这些工作的预印本身份与投稿备注在下表明确列出，不把尚未核验的录用状态写成会议事实。

其中，WeatherPEFT 已有 ICLR 2026 官方会议页面；Aurora 有 Nature 2025 原文及逐 rollout 步 LoRA 的实际公开实现。后两者直接阻止“天气 + PEFT + 多步作用就是 novelty”的论证。SPW 则说明推理期改变既有天气模型权重本身也不是空白。

仅检索 2024–2026 会遗漏数学最近邻。Green’s-function 模式参数校准早已采用扰动响应与加权最小二乘；FSO/EFSO 的预报误差影响分析亦有长期基础。SPO、PETS、影响函数和 Solver-in-the-Loop 分别覆盖决策导向学习、响应模型规划、局部影响估计与循环输出订正。它们不等于 EarthDelta 的全部设定，但足以否定泛化过度的“新范式”包装。

#### 开源实现核查的实际深度

| 实现 | 本次实际核查 | 可据此决定什么 | 不能据此声称什么 |
|---|---|---|---|
| Stormer | 固定提交的 `inference.py`、LICENSE | 独立官方参考；零增量均值；固定步长与多路径平均区分 | 已复现官方论文所有 skill 数值 |
| Aurora | 固定提交的 `aurora/model/lora.py` | 已有逐步 LoRA；默认/逐步调度语义；MIT 代码头 | 已在 EarthDelta 运行 Aurora checkpoint |
| WeatherPEFT | 官方树与 `aurora/model/fish.py` | 梯度重要性与随机 Fisher-mask 的实际机制 | 可以无条件复制整个仓库；许可仍需核验 |
| MEND / LoRAHub / AdaMerging / Solver-in-the-Loop 等 | 原论文及已定位的官方代码入口，未逐行运行 | 最近邻与基线设计来源 | 其代码已可直接套入天气任务且通过验证 |

完整提交/许可边界在 `implementation/LITERATURE_TO_CODE.md`。未核实许可时不复制代码；未定位官方代码时不抹去原论文的 prior-art 意义。GraphCast、Pangu、FuXi、FengWu、FourCastNet、GenCast 等可作为主干或集合预报背景，但“没有在其中某个模型上做过”不足以构成方法 novelty。

原始依据：[Stormer](https://arxiv.org/abs/2312.03876)、[FSO 原始工作](https://doi.org/10.1111/j.1600-0870.2004.00056.x)、以及下一部分 R01–R20。

---

## Part IV — Nearest Neighbor Matrix

| Work | Year / venue | Core mechanism | Predicts intervention effect? | Parameter intervention? | State conditioned? | Online selection? | Analytical utility | Weather? | 真正重叠 | 剩余候选贡献（尚未证明） |
|---|---|---|---|---|---|---|---|---|---|---|
| [R01](https://arxiv.org/abs/2608.02528) Uncertainty Is Not Enough: Value-of-Information Routing for Mixtures of LoRA Experts (VI-MoLE) | 2026-08-03 · arXiv | 预测专家前缀的剩余风险，按边际价值分配预算 | 标量风险，不是天气响应场 | 是，LoRA 库 | 输入 | 是 | 风险代理；不能当真实收益恒等式 | 否 | 最直接威胁“反事实价值驱动的预算 LoRA 路由” | 有符号多时效响应、延迟核验、可重加权目标的实证价值 |
| [R02](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL107938) Machine Learning Enables Real-Time Proactive Quality Control: A Proof-of-Concept Study | 2024 · GRL | 用机器学习参考分析支持实时主动观测质控 | EFSO 观测影响 | 否，观测/同化 | 分析历史 | 是 | 预报误差影响量；参考态是估计 | Lorenz-96 概念验证 | 未来真值不可得时仍做预报影响决策已有先例 | 参数编辑的响应学习与全球天气轨迹验证 |
| [R03](https://journals.ametsoc.org/abstract/journals/mwre/133/5/mwr2912.1.xml) Using Green’s Functions to Calibrate an Ocean General Circulation Model | 2005 · Monthly Weather Review | 参数扰动建立响应核，联合校准参数 | 是，响应核 | 物理模式参数 | 校准资料/模式 | 离线 | 加权最小二乘 | 海洋 | 有限差分—响应矩阵—加权拟合不是新算法 | 把流依赖响应摊销为逐起报预测，而非一次全局校准 |
| [R04](https://gmd.copernicus.org/articles/15/2309/2022/) Earth system model parameter adjustment using a Green’s functions approach | 2022 · GMD | ESM 多参数敏感度与多观测目标加权优化 | 是，敏感度实验 | 20 个物理控制参数 | 目标与权重 | 离线 | 加权最小二乘 | Earth system | 联合影响、目标权重与参数依赖均有先例 | 神经参数编辑的输入条件化响应预测及核验标签效率 |
| [R05](https://arxiv.org/abs/2605.14546) Discovering Physical Directions in Weight Space: Composing Neural PDE Experts (CCM) | 2026-05-14 · arXiv | 同锚点端点专家分解，物理坐标/观测前缀选择权重组合 | 权重方向，不等同输出 Jacobian | 是，权重合并 | 物理元数据/前缀 | 选定后部署 | 坐标读出/前缀校准 | PDE，非全球天气 | 物理条件化权重干预与长轨迹收益不独占 | 未知未来误差下对响应场进行选择，不依赖已知物理坐标 |
| [R06](https://arxiv.org/abs/2609.17042) Learning Options for Compositional Motor Control with Adapter Banks (CABRA) | 2026-09-15 · arXiv | 共享循环网络、适配器库、组合运动控制 | 不是先预测逐编辑天气响应 | 适配器；低秩为学习后结构 | 任务/轨迹 | 优化控制策略 | 任务轨迹目标 | 否 | 共享主干+适配器库+控制已存在 | 未知天气未来真值、响应代理与核验延迟；不能声称首个适配器控制 |
| [R07](https://arxiv.org/abs/2609.08412) Stochastically Perturbed Weights: Ensembles from Deterministic Machine-Learning Weather Models (SPW) | 2026-09-08 · arXiv；AIES 投稿非录用 | 已有确定性天气模型推理时权重扰动生成集合 | 通过实际采样，不是收益预测 | 是，原始权重张量 | 架构/注入位置 | 扰动推理 | 概率预报评价 | 是，四种主干 | 冻结天气模型上的推理期参数扰动已有直接先例 | 逐状态定向改善的可预测性，而不是生成不确定性集合 |
| [R08](https://www.nature.com/articles/s41586-025-09005-y) A foundation model for the Earth system (Aurora; LoRARollout implementation) | 2025 · Nature；公开代码核查至 2026-09-14 | Earth-system foundation model；实现含逐 rollout 步 LoRA | 不独立预测 edit effect | 是，LoRA | 步号/任务 | 预定步号调度 | 训练预报损失 | 是 | “改变多步天气动力学的 LoRA”不是新概念 | 逐起报有符号响应评估和主动不编辑 |
| [R09](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4e6219cc3913fd508dbd7f3f6d90cd4e-Abstract-Conference.html) Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models (WeatherPEFT) | 2025 arXiv / ICLR 2026 | 任务动态提示与随机 Fisher 引导参数选择 | 否 | 是，选参数/提示 | 任务/输入模式 | 前向条件化，非候选收益规划 | 训练目标 | 是 | 天气特定 PEFT 与动态条件化已被覆盖 | 效果建模、选择性干预；不能只比较普通 LoRA |
| [R10](https://arxiv.org/abs/2608.21652) GeoQ: Geometry-Aware Conditional Quantile Error Estimation for Scientific Surrogate Models | 2026-08-21 · arXiv | 锚点校准误差+几何条件分位数增量 | 预测误差而非编辑效果 | 否 | 查询/支持度 | 误差估计 | 条件分位数，非有符号二次收益 | 含天气实验 | 逐输入科学代理误差预测已有直接先例 | 误差方向与编辑响应对齐；误差幅值不够规划 |
| [R11](https://arxiv.org/abs/2606.02663) AdaWeather: Adaptively Mixing Probabilistic Weather Forecasts with Logarithmic Regret | 2026-06-01 · arXiv；NeurIPS 投稿非录用 | 监督学习+在线专家建议组合概率天气预报 | 不是参数效果预测 | 否，输出混合 | 上下文/反馈 | 是 | 预报混合损失与 regret | 温度预报 | 天气中的适应性选择、相对静态混合的价值需对照 | 参数干预而非输出融合，必须用实验证明必要性 |
| [R12](https://openreview.net/forum?id=0DcZxeWfOPt) Fast Model Editing at Scale (MEND) | 2022 · ICLR | 学习转换低秩分解梯度来快速编辑 | 不独立预测所有候选效果 | 是 | 给定输入—目标输出 | 是 | 编辑目标/局部性损失 | 否 | 学习参数编辑、低秩编辑、局部性不是新贡献 | 起报时没有目标未来答案，不可调用 edit-label 梯度 |
| [R13](https://proceedings.mlr.press/v70/koh17a.html) Understanding Black-box Predictions via Influence Functions | 2017 · ICML | 梯度/Hessian-vector 近似训练数据扰动影响 | 是，局部影响 | 由重加权训练诱导 | 训练点/测试点 | 诊断 | 局部影响近似 | 否 | 执行昂贵改动前估计输出/损失变化并非全新范式 | 有限神经编辑轨迹响应，不等同训练样本影响函数 |
| [R14](https://arxiv.org/abs/2307.13269) LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition | 2023 · arXiv | 用少量目标任务示例无梯度搜索 LoRA 组合 | 实际评估，不是响应代理 | 是 | 目标任务示例 | 适应后推理 | 示例损失 | 否 | 组合库+搜索已知；无梯度不代表无标签 | 不用未来真值，离线响应监督与合法历史核验分离 |
| [R15](https://arxiv.org/abs/2310.02575) AdaMerging: Adaptive Model Merging for Multi-Task Learning | 2023 arXiv / 2024 版本 | 无标签测试样本熵最小化调节合并系数 | 不独立建模响应场 | 是 | 测试分布 | 测试时适应 | 熵代理 | 否 | 无测试标签的权重组合已有先例 | 天气回归不能直接把分类熵当可靠误差代理 |
| [R16](https://arxiv.org/abs/2408.01415) Conditional LoRA Parameter Generation (COND P-DIFF) | 2024-08-02 · arXiv | 参数自编码器+任务条件潜空间扩散生成 LoRA | 否，生成参数 | 是 | 任务条件，非自动等同逐天气状态 | 参数生成 | 生成/适应训练目标 | 否 | 条件生成 LoRA 不构成 EarthDelta 独占创新 | 先评估响应再选择与直接生成的受控比较 |
| [R17](https://arxiv.org/abs/1805.12114) Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS) | 2018 · arXiv/NeurIPS | 概率动力学模型+轨迹采样规划 | 状态—动作效果 | 动作非网络权重 | 当前状态/动作 | 是 | 已知奖励下预期轨迹回报 | 否 | 学习响应再规划、风险传播来自模型预测控制谱系 | 模型自身参数为动作，天气真值和可核验标签的特殊信息结构 |
| [R18](https://arxiv.org/abs/2608.29998) The Intervention Gap in Latent World Models | 2026-08-30 · arXiv | 匹配真实干预，审计规划时的效果保真度 | 直接审计干预效果 | 环境动作，非参数编辑 | 任务/状态/候选 | 评测 | operator-error 与任务方向 | 否 | 预测准确/奖励准确不等于干预准确的区分已被明确提出 | 把该审计应用于天气参数干预并展示新的可重复经验规律 |
| [R19](https://arxiv.org/abs/1710.08005) Smart “Predict, then Optimize” (SPO) | 2017 · arXiv；后续正式发表 | 让预测损失关注下游优化决策误差 | 预测决策参数 | 不特指网络编辑 | 决策上下文 | 预测后求解 | 优化结构/决策损失 | 否 | 预测 b/H 再 QP、用 regret 训练有成熟谱系 | 天气响应监督与结构化信息分离必须提供额外价值 |
| [R20](https://arxiv.org/abs/2007.00016) Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers | 2020 · arXiv；2021 修订 | 把订正器放入可微分迭代求解器训练 | 学习订正，不独立逐候选响应 | 输出/状态订正 | 当前迭代状态 | 循环订正 | 多步物理求解损失 | PDE | 输出订正也能形成稳定、连贯多步反馈；不能弱化这类基线 | 参数通道是否比同信息循环输出订正更有价值需实测 |
### 最接近的五项：10 维逐项比较

以下比较是从原论文与当前代码推导的审计判断，不是外部方法的新实验结果。对方的优点不因为不在天气领域而被抹去。

| 维度 | VI-MoLE [R01] | ML-PQC [R02] | Green’s-function [R03/R04] | CCM [R05] | CABRA [R06] | EarthDelta 当前/目标 |
|---|---|---|---|---|---|---|
| Scientific question | 哪个专家值得追加 | 哪个观测值得保留 | 哪些参数联合降低模式偏差 | 专家权重是否包含物理方向 | 适配器能否复用组合运动技能 | 未知未来误差时是否能选择有效参数编辑 |
| Formulation | 条件剩余风险与增量价值 | ML 参考分析支持 EFSO | 参数响应核与加权最小二乘 | 同锚点共享更新+有符号端点方向 | 共享循环动力学+适配器代码 | e0、有限 du 或局部 R；二次几何 |
| Supervision | 委员会/任务风险 | 历史分析训练参考预测 | 历史观测目标及扰动模拟 | 端点训练资料/物理坐标/前缀 | 运动示范与轨迹目标 | du 无未来标签；e0/收益须核验 |
| Inference | 风险预测后预算分配 | 估计影响后质控 | 调参后部署 | 读出坐标后单一合并模型 | 优化/执行适配器策略 | 合法历史→预测→预先承诺编辑窗 |
| Intervention | LoRA 专家前缀/组合 | 观测/同化 | 物理参数 | 神经网络权重组合 | 残差适配器 | 注意力投影的低秩权重等价改动 |
| Objective | 单位成本风险下降 | 预报误差下降 | 多目标加权误差 | 物理族 OOD 轨迹精度 | 已知目标运动轨迹 | 固定物理指标的预报误差下降 |
| Optimization | 有条件的贪心分配/校准 | 质控决策 | 联合线性拟合 | 坐标校准/选择 | 可微目标下控制 | 有限选择或支持枚举+盒约束 QP |
| Serving info | 输入/专家状态 | 当前分析与过去资料 | 校准完成后的模型 | 已知元数据或已观察前缀 | 给定任务目标 | 不含本次 forecast 的未来验证真值 |
| Budget | 专家计算预算 | 同化/预报影响 | 离线模拟数量 | 单个部署模型 | 控制优化成本 | 目前多为声明的专家成本；须测真实开销 |
| Experiments | LoRA 路由任务 | Lorenz-96 概念验证 | 海洋/耦合 ESM | 三类 PDE | 运动控制 | 目前核心真实天气选择结果未证实 |

严格区别：CCM 的“有限差分探针”是端点**权重**的差，不自动等于 EarthDelta 在每个输入上测得的**输出**响应 Jacobian。CABRA 的适配器低秩结构不应擅自改写成固定 LoRA 参数化；其共享网络训练协议也不能简化为从头到尾冻结预训练主干。ML-PQC 是有力先例，但不能写成已在全球神经天气模型上完成相同参数编辑。

#### Reviewer 最可能的 X + Y

最有力的说法不是“LoRA + weather”，而是 **Green’s-function 参数校准 / forecast-impact analysis + VI-MoLE 式价值路由**；从系统架构看还可以说 **条件适配器库 + model-based planning**。该批评对算法原语基本成立。合理回应必须是新的受控发现：有限轨迹响应是否可预测、是否在少核验标签下产生优势、是否能复用同一代理更换指标/预算，以及是否比直接标量风险预测和循环输出订正更好。单靠换名为“counterfactual world model”不能回应。


### 论文与代码证据分开

“没有定位到官方代码”不等于作者没有代码，也不使先行论文失去 prior-art 意义。除下述明确说明外，本次未运行外部实现。预印本不能写成顶会录用。

- **R01** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R02** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R03** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R04** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R05** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R06** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R07** — 见原文官方 Code 链接；本次未审计其实现/许可
- **R08** — https://github.com/microsoft/aurora
- **R09** — https://github.com/ShileiCao/WeatherPEFT
- **R10** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R11** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R12** — https://github.com/eric-mitchell/mend
- **R13** — https://github.com/kohpangwei/influence-release
- **R14** — https://github.com/sail-sg/lorahub
- **R15** — https://github.com/EnnengYang/AdaMerging
- **R16** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R17** — https://github.com/kchua/handful-of-trials
- **R18** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R19** — 本次未核实可复用官方代码、固定提交和许可；仅用原文作方法比较。
- **R20** — https://github.com/tum-pbs/Solver-in-the-Loop

---

## Part V — Novelty Audit

结论：**当前创新强度不足以单凭方法结构支撑强顶会 claim；但保留一个值得低成本否证的交叉科学问题。** 尚未检索到与全部设置完全同构的公开方法，不能据此证明“first”。最近邻不是 GraphCast/Pangu，而是 VI-MoLE、ML-PQC、Green’s-function 校准、CCM 与 CABRA；参见 RELATED_WORK.md 的原文链接。

| 层面 | 最强可尝试的主张 | 审稿人反驳 | 必须证据 | 当前判断 |
|---|---|---|---|---|
| Problem | 起报时预测参数编辑对未知未来误差的价值 | ML-PQC 已处理不可见未来参考；VI-MoLE 已做价值路由 | 全球天气/延迟核验/真实 finite response 的统一受控结果 | 窄交叉空白有潜力，不是泛范式首创 |
| Formulation | e0/du 分离利用两类监督 | 平方误差展开与 response calibration 是旧方法 | 低标签/跨指标比 direct gain 更好 | 公式 novelty 低，信息结构值得验证 |
| Algorithm | 交互感知预算选择 | Gram 最小二乘/支持枚举/QP 已成熟 | composition holdout、非线性误差及真实收益 | 当前算法 novelty 低 |
| Weather | 流依赖多时效物理编辑 | Aurora/WeatherPEFT/SPW 已做天气适配/扰动 | 跨变量、极端、长时效，不依赖单一投影 | 目前 weather-specific 方法证据不足 |
| Systems | 可逆、可审计 issue-time edits | 工程纪律不是科学贡献 | 真实成本、安全回退、无泄漏、重放 | 工程有价值；不能替代科学发现 |
| Empirical | 发现可预测的干预子空间/其边界 | 只是挑选容易改善的模型/年份 | 预登记失败边界、强基线、独立年与负结果 | 最值得押注，但当前未有真实证据 |

---

## Part VI — Scientific Value Audit

e0 本身就是残差监督。如果能预测 e0，直接修正 forecast 是一个自然且强的对手。参数编辑可能限制输出在模型动力学允许的轨迹族中，降低多步订正的样本复杂度，并让同一低维动作影响多个变量/lead；但这些只是可检验归纳偏置。模型本身不是严格保守的物理积分器，编辑也不自动保持守恒或平衡。

“输出订正只会修一个点，参数编辑才有连贯 rollout”是假对照：循环 residual corrector 可以把订正状态反馈，trajectory corrector 也能共同预测多变量/时效。应当同时比较 terminal、trajectory 和 recurrent 三类输出订正；后两类进入强主表。只有参数编辑展示匹配资源下的 skill/稳定性/标签效率/可复用性优势，才有独立存在价值。

相比 ensemble/更大模型/全量重训，本项目的合理动机是重用既有资产、控制适应成本、检测何时不应调整；不是声称所有情形都更好。对 ensemble 比较真实调用次数与概率指标；对 full fine-tuning 比较总训练成本；对 retrieval/memory 比较同一可用资料；对 calibration 保留简单偏差订正。


#### 可接受的必要性论证

最好把“为什么 edit 参数”写成以下可证伪命题，而不是抽象的物理一致性口号：在同样合法信息和训练预算下，用少量低秩动作限制整条预测轨迹的变化，是否能降低核验标签需求，或改善关键变量/长 lead 的性能—成本权衡？若成立，限制假设空间是有价值的归纳偏置；若不成立，输出订正可能更简单、更有效。

反例也应被认真保留：好的静态 LoRA 可能已经消除大部分系统误差；直接 gain 可能绕开高维响应重建而更高效；高容量 trajectory residual 可能达到更好准确率；更大的预训练模型可能不再有编辑空间。任何一种都不能通过只换更弱 backbone、只挑极端个例或缩短 lead 来隐藏。

因此当前的 scientific value 是**研究问题值得检验**，不是**已证明 operational forecast system 需要逐起报改参数**。重新运行确定性 backbone 得到 du 不需要真实未来标签，但仍耗大量离线计算。若在线特征来自额外参考前向，必须纳入延迟；不能说“冻结特征免费”。

---

## Part VII — Mathematical / Methodological Audit

### 数学合同：什么精确，什么不精确

令 F_a(X) 为同一冻结参考模型在编辑计划 a 下的**完整有限 rollout**，Y 为仅离线可见的未来验证资料。对同一个固定 summary S 定义

    e = S(Y) − S(F_0(X))
    d(a) = S(F_a(X)) − S(F_0(X))
    L(a) = (e − d(a))^T Q (e − d(a)),  Q ⪰ 0
    G(a) = L(0) − L(a) = 2 e^T Q d(a) − d(a)^T Q d(a).

G>0 表示改善。这里有限 rollout 可以任意非线性，S 也可以非线性：只要始终先用同一个 S 再相减，`e−d=S(Y)−S(F_a(X))` 恒成立。线性 S 的必要性来自不同命题：`S(Y−F0)=S(Y)−S(F0)`、对原场固定二次范数的解释、局部响应叠加，不能混为一谈。若 S 取 kinetic-energy spectrum，则得到谱空间的损失恒等式，不是原始风场 MSE 的等价优化，方向/相位可能丢失。

固定仿射标准化的平移在差分中抵消，尺度会改变 Q。必须固定通道顺序、训练期尺度、纬度/lead/变量权重。候选相关的标准差、自适应归一化或不同 summary 会改变比较对象。代码里 `paired` 的归一化加权平均、`geometry` 的加权和、head 的 feature mean 不能混用；统一约定后再计算成本惩罚。首版优先将实测成本作为可行性约束；若用软惩罚 G−λc，λ 必须预登记并把成本换算到相同收益单位，不可直接把毫秒、rank 或 token 数与 MSE 相减。

#### 局部 QP

若 d(a)≈Ra，b=R^T Qe，H=R^T QR，则 surrogate gain=2b^T a−a^T Ha。
固定支持集时，H PSD 对应凸最小化 `a^T(H+λI)a−2b^Ta`，盒约束 |a_j|≤ρ。再加 support 数量、激活成本时，整个问题包含组合约束；当前支持枚举加数值盒优化，不是一个无条件的全局连续凸 QP。

H_ij 表示响应列在度量下的重叠。它不代表真实二阶动力学项。定义

    γ_ij = d(a_i+a_j) − d(a_i) − d(a_j).

即使 H=R^TQR 计算完全正确，γ_ij 仍可很大。应该实际缓存 pair intervention，或者使用局部可信域与非线性误差诊断。**有限候选数组输入到 b/H 分支，也不会自动把线性近似变成真实 finite-response 评估。**

`(a−a*)^T H(a−a*) = ||R(a−a*)||_Q²` 是输出响应度量下的系数模仿；有用但不应命名成新的蒸馏几何原理。

#### 真正需要预测多少 e0？

在固定线性响应空间中，决策只需要 b=R^TQe，而不是完美重建 e 的所有方向。设 Q=I、响应只有第一坐标 e1，则第二坐标的误差再大也不影响编辑排序。因此：完整 e0 的 R² 低**不能单独作为 kill criterion**；必须测 response-subspace error、b 误差、gain ranking 与实际 regret。反过来，完整 e0 的平均 R² 高也可能漏掉极小但关键的响应方向。

若 ê=e+η，d̂=d+ξ，则

    Ĝ−G = 2η^TQd + 2(e−d)^TQξ + 2η^TQξ − ξ^TQξ.

这给出明确分解，而不是笼统说两个 head 都要好。对同一有限可行集合，若所有候选 |Ĝ−G|≤ε，精确 argmax 选择的 regret≤2ε；近似求解器再加优化误差。该界是直接代数性质，不作为独立新定理贡献。

#### 不确定性不能只用 mean gain 减一个任意方差

给定 issue-time 信息 I，令 e、d 的条件均值为 μe、μd，则

    E[G|I] = 2μe^TQμd − μd^TQμd
             + 2 tr(Q Cov(d,e|I)) − tr(Q Cov(d|I)).

相关项不可默认忽略。固定确定性 backbone 的真实 d(X,a) 对完整 X 是确定的；代理模型/压缩 context 的不确定性不能自动称“大气随机性”。

首版不必训练复杂联合概率模型。可在独立校准块上直接计算每次起报的最大收益高估 `r_t=max_a(ĝ_t(a)−g_t(a))`，用固定分位数 q 得到 `LCB_t(a)=ĝ_t(a)−q`；只有 LCB 超过成本且保护条件满足才编辑。No-edit 的真实收益恒为 0，应特殊处理为精确零。

在 iid/exchangeable calibration 条件下可以讨论集合级覆盖；天气时间依赖、候选改变和分布漂移会破坏直接套用的保证。本项目默认报告时间块上的经验覆盖和 shift degradation，不声称无条件 finite-sample operational safety。两项风险上界之差也**不是**收益下界；不能不加证明地照搬“certified marginal gain”的措辞。



### 决策子空间与目标复用：必须遵守的附加合同

以下为建议实现，而非当前代码已经具有的能力。先将每个变量/lead 的格点场按固定训练期尺度与纬度面积权重白化。只在 train-only 的真实编辑响应上拟合固定正交基 U；不得用测试 truth，也不得在 serving 中先计算实际候选响应后把它当作免费的“预测”输入。

对同一白化空间，写 e=Ue_z+e_perp、d=Ud_z+d_perp，则

    G_full − G_projected = 2 e_perp^T d_perp − ||d_perp||².

因此低响应重建误差或高解释方差**不充分**：小 d_perp 仍可能与很大的 e_perp 对齐，改变收益和排序。基维数的选择应同时看开发集的实际 gain distortion、top-edit agreement、regret、完整场保护量，并把最终 basis 锁定后再测试。

为了有清晰的目标复用保证，首版使用按 variable×lead 分块的 U，并只允许对这些块施加非负标量权重。该类权重保持块内基空间不变。新空间区域掩码、任意非对角物理度量或新变量不自动满足这一条件；必须保留完整场预测，或重新定义/校准读出并计入成本。不能仅改变一个 Q 矩阵就声称可迁移到所有天气目标。

主实验至少比较：full-e/full-d；projected-e/projected-d；直接 b；action-conditioned direct gain。相同 metric/budget 输入必须给 direct-gain/router。对未见 bank 成员，现有专家 ID 和系数描述符不够；默认不宣称 zero-shot，新专家需独立权重描述或无标签 probe adaptation 协议。

运行时的保护条件只能依赖当前可用信息、模型预测或确定性有效性约束。真实未来保护变量损失只可由离线 evaluator 计算，不能暗中用其决定应用/撤销编辑。

#### 参考误差的四级评估与四象限分解

对 e0 分别评估 oracle、朴素预测、只在独立校准块上拟合的偏差/尺度校准预测，以及有明确校准边界的不确定性版本。四级不是要求直接堆四套模型，而是逐步判断误差来自不可预测性、简单失校准还是选择风险。

与响应预测交叉得到四象限：oracle e/oracle d 给出同一候选集的上界；oracle e/predicted d 隔离响应代理；predicted e/oracle d 隔离决策相关误差；predicted both 是实际可部署目标。oracle 分支不能进入实际服务路径。

高 oracle gain 并不保证有可学策略。事后在许多候选中取最小误差，天然产生选择优势；即使这一优势真实且非数值噪声，也可能依赖起报信息无法预测的未来误差。必须再看独立预测器的实际 regret，而不是将 oracle 提升直接称作可实现收益。

---

## Part VIII — Repository Audit

| ID | 优先级 | 位置/符号 | 观察 | 最小处理 | 证据边界 |
|---|---|---|---|---|---|
| A01 | P0/blocker | [scripts/s0_gate.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/scripts/s0_gate.py): `verify_zero_edit_equivalence / run_s0_gate` | “官方”比较实际是本地 bridge 的两条路径，传空 expert_loras；RMSE 写报告但不进入总体 gate。 | 重新独立调用 pinned upstream，非零 B 的零系数测试；RMSE/NaN/通道/网格条件全部列入 gate。 | 代码事实，不表示已观测到真实预报错误。 |
| A02 | P0/blocker | [earthdelta/bridge/stormer_bridge.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/bridge/stormer_bridge.py): `NormalizationContract.denormalize_diff` | 本地支持增量均值；官方 inference.py 构造增量变换时均值为零。 | 严格 official 模式使用零增量均值；输入非零 diff_mean 时拒绝或明确独立协议。 | 是否影响现有 checkpoint 的实际输入，取决于本地 npz；当前 UNVERIFIED。 |
| A03 | P0/blocker | [earthdelta/lowrank.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/lowrank.py): `ExpertLoRA.__init__` | B 初始化全零；未训练银行可能所有编辑都为零。 | 把专家构造与非退化检查置于 oracle 前；训练专家时系数须非零。 | 初始化为零本身是合理 LoRA 设计，不是代码错误；错误是拿它测 scientific ceiling。 |
| A04 | P0/correctness | [earthdelta/selection.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/selection.py): `plan_from_prediction finite candidate branch` | 有限候选分支没有继承 bound/max_active/max_candidates 的过滤。 | 所有分支共用 feasibility 检查，包括 finite/shape/box/support/cost/noop/稳定 tie-break。 | 静态代码审计发现；本次未运行完整项目测试。 |
| A05 | P0/metric | [earthdelta/paired.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/paired.py): `_assert_linear_operator_assumption / paired_targets` | 文档对 gain 的符号有误；把非线性 summary 一概视作破坏恒等式不正确；断言体为空。 | 统一正收益符号；解释任意共同 summary 的恒等式和原空间 metric 的区别。 | 代数结论可以独立验证，不依赖天气数据。 |
| A06 | P0/metric | [earthdelta/heads.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/heads.py): `ComposedPredictionHead.forward` | gain 默认包含可学习 calibration，且 mean-feature 归约；paired/geometry 权重归约并不一致。 | 输出 analytic_gain 与 corrected_score 两个不同字段；统一 MetricSpec；主线关闭任意校准偏置。 | 不能把可学习修正后的分数叫 exact analytical gain。 |
| A07 | P0/data | [earthdelta/data/make_splits.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/data/make_splits.py): `build_manifest / compute_guard_boundaries` | naive datetime.timestamp 受系统时区影响；归一化 hash 有 placeholder 默认值；默认核验延迟 6h。 | UTC aware；真实 hash 必填；实际资料 coverage；ERA5 回放与 operational availability 分开。 | 6h 不是已核验 ERA5 可用时延；retrospective 协议必须标注。 |
| A08 | P0/provenance | [earthdelta/contracts.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/contracts.py); [earthdelta/bridge/stormer_bridge.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/bridge/stormer_bridge.py): `ArtifactVersion / load_stormer_checkpoint / controlled_rollout` | 非空字符串即可；结构版本不等于 checkpoint 内容身份；版本检查工具未在 rollout 自动执行。 | 完整 checkpoint / bank / norm / metric / grid / split 内容指纹，公共入口 fail closed。 | 严格 state_dict load 只验证结构映射，不证明预测 parity。 |
| A09 | P1/training | [earthdelta/bridge/stormer_bridge.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/bridge/stormer_bridge.py): `controlled_rollout` | tuple→torch.tensor 切断系数梯度；一个 EditPlan 共用于 batch；默认只返回最终步。 | 保留旧 API；加 differentiable coefficient tensor path 与 trajectory return，逐样本计划。 | 梯度断开不影响离线枚举，但阻塞同信息 end-to-end router 强基线。 |
| A10 | P1/system | [earthdelta/bridge/stormer_bridge.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/bridge/stormer_bridge.py): `controlled_rollout hooks` | try/finally 清理合理，但 forward hooks 修改模块临时状态，不是并发安全的纯函数。 | 早期单进程显式限制；P1 新建显式低秩投影接口，保留旧路径 parity。 | 可逆不等于无运行时共享状态；不要无理由重写整个主干。 |
| A11 | P1/math | [earthdelta/geometry.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/geometry.py); [earthdelta/probe.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/probe.py): `ResponseGeometry / local_linearity_error` | RᵀQR 是线性响应重叠，不包含真实 mixed nonlinear interaction。已有测试明确展示失败例。 | 用实际 pair response 检验 gamma_ij；超出局部可信域转 finite exact response。 | 不将正确的一阶几何改成错误“全非线性交互”。 |
| A12 | optional/fail-closed | [earthdelta/memory.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/memory.py): `ewma_error_feature_batched` | 广泛捕获 ValueError 返回零可能掩盖版本/重复记录问题；zip 长度不一致可截断。 | 只允许明确 empty-history 回退；其他异常抛出；batch lengths 校验。 | memory 不进入 P0 主线，使用该分支前必须修复。 |
| A13 | status | [earthdelta/probe.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/probe.py); [earthdelta/spectral.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/spectral.py): `jvp_response / band_energy / single_mode_energy` | 存在 NotImplementedError；频谱诊断是系数级工具，不是完整天气球面谱流程。 | 如实标记未实现；JVP 与 SHT 暂缓，不作为 P0 前置任务。 | 已有 coefficient_diagnostics 可以保留但不可夸大。 |
| A14 | status | [README.md](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/README.md); [scripts/s0_gate_inputs/channels.json](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/scripts/s0_gate_inputs/channels.json): `README evidence vs tracked tree` | README 报告 2020 ERA5 和 checkpoint，本次能访问的树没有对应真实大文件或 S1+ 实验结果。 | 本地 inventory 核验后才能升级为 VERIFIED；报告不把“未见”写成“根本不存在”。 | 报告型证据≠本轮独立数据校验；本轮未跑 195 项测试。 |
| A15 | P1/claim | [earthdelta/contracts.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/contracts.py); [earthdelta/heads.py](https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/earthdelta/heads.py): `EditPlan.descriptor / EditResponseHead` | 描述符包含激活标记、系数、窗长，不包含新专家权重本身的可泛化表征。 | 新 bank 是新版本；未见专家实验需明示 descriptors 或无标签 probe adaptation。 | 不能声称当前按构造支持新专家 zero-shot。 |


### 完成度

- CODE_VERIFIED：低秩专家、成对目标、有限/局部探测、Gram 几何、离线教师、规划器和头类；桥接/归一化/数据下载/划分实现；相关合成测试。
- README_REPORTED：195 项总测试规模，192 passed / 3 skipped 的记录；2020 ERA5 `1464×69×128×256`；本地两种 Stormer checkpoint。
- UNVERIFIED：以上数据/权重的实际 bytes、SHA256、可读性、训练年覆盖、GPU 可用性、完整测试当前执行结果。
- 尚无本轮可核验的论文级 evidence：独立官方真实 parity，非零 bank oracle ceiling，du/e0 预测效果，动态 edit skill，跨年/长轨迹/成本收益。

不能从“GitHub 没有大文件”推断“本地没有数据”。但也不能把 README 的下载完成记录升级成已完成科学验证。

---

## Part IX — Fatal Risks & Kill Criteria

#### 按项目生死顺序排列

先处理 correctness / provenance，否则任何 oracle 或 skill 都不可信。之后的关键顺序不是“先训练表示，再看效果”，而是：实际非零动作 → 相对 no-edit 的 oracle 空间 → 相对静态方案的动态空间 → 响应方向/误差投影的可预测性 → 强输出订正与直接路由挑战 → 长轨迹与完整场保护 → 可推广性。

P0 资源缺失并不证明方法失败；一个全零 LoRA bank 也不能否定所有参数编辑。相反，若经一次预登记的合理 bank 重构后仍无可辨 oracle 空间，就不应继续靠增加 JEPA/memory 等模块救场。

### 状态不是二值

- `BLOCKED`：checkpoint / 数据 / 独立官方依赖 / 许可 / 配置 / 资源缺失。不得当科学失败，也不得用合成数据顶替。
- `INVALID`：泄漏、度量错、版本错、官方 parity 错、实际未来数据可达。停止这次实验，修复后新 run；历史结果作废但保留。
- `INCONCLUSIVE`：置信区间跨过决策阈值或样本不足。只按预登记规则增加样本，不得直接换指标/年份。
- `PASS_SCREEN`：通过开发集的继续投入门，不等于发表证据。
- `PASS`：对应任务全部工程条件与预先指定的科学条件满足；必须附 artifact/hash。
- `STOP` / `PIVOT`：当前主张失败或被更简单方案解释。不得通过加模块绕过。

### 固定度量

主损失 L_t(a)=Σ_h β_h Σ_v α_v Σ_i w_i[(F_a−Y)_{hvi}/s_v]^2，所有权重预先固定且各自归一化，s_v 来自训练期，不随候选或测试结果改变。主 screen 先用 72h 的同一固定多变量目标；Z500/T850/T2m/U10/V10/MSLP 分项和 24/120/240h 是保护/扩展，不准只选赢家。

主收益差异为 paired origin loss，不把网格点当独立样本。forecast 表逐变量使用 sqrt(聚合加权 MSE)，不是先开根号再平均。Z500 默认 geopotential m²/s²，若转 geopotential height 要明确除以固定 g0 并改单位。

### 阈值如何得到（不是随意指定 2%）

P0-01 在固定精度/硬件上重复无编辑参考推理，计算数值引起的 paired loss 差异，并冻结 δ_noise（相应绝对差的高分位数及置信上界）。这是数值可分辨阈值，不是业务意义阈值。

在独立开发 pilot 上估计 paired-loss 的时间块方差 σ_block、相关长度和可承担的最大独立块数 n_eff,max。样本量规划采用正态近似 δ_MDE=(z_0.975+z_0.8) σ_block/sqrt(n_eff,max)≈2.80 σ_block/sqrt(n_eff,max)，实际区间由 time-block bootstrap 给出。MDE 只表示该设计对效应的探测能力，不等于应用价值。

`decision_policy.json` 在后续 oracle/主比较前冻结：主指标、开发/校准/测试范围、block length、最大样本/最大 GPU-hours、δ_noise、δ_MDE、δ_practical（若有业务最低收益依据）、protected_harm_margin、alpha、最终单次检验方案。没有 operational relevance 依据时，`delta_practical=null` 并标记 `claim_scope=research_screen_only`；允许用 max(δ_noise,δ_MDE) 做资源投入 screen，但禁止称为业务收益已达标。不能把 δ_MDE 当事实上的 operational threshold。

如需要中途反复看结果，先指定 look 数与 alpha spending；默认只做一次固定样本量正式 gate。模型/字典/超参数试验预算同样登记。开发集选方案，测试集仅最后打开一次；统计结论以测试集为准。

### STOP-0 — Integrity

任何独立官方 parity、零系数+非零 bank、缓存指纹、未来标签隔离、UTC/coverage、finite constraints、metric identity 测试失败 → `INVALID`，不得训练主系统。官方不可运行 → `BLOCKED` 而不是“内部一致所以 PASS”。

### STOP-1 — No real edit / no oracle ceiling

首先检查 bank 存在非零 BA、非零 rollout response 且作用窗正确；全零银行是 `INVALID_DICTIONARY`，不是科学反证。

对冻结字典含 no-edit 的真实最优 a*_t 计算 I_oracle=E[L_t(0)−L_t(a*_t)]。若区间上界 U_oracle<δ_survival → STOP 当前字典。最多允许一次在方案中事先列明的 trained/structured bank fallback（只用 train/dev，不看最终 test）；仍失败则 STOP 参数编辑主线。若 L_oracle>δ_survival → PASS_SCREEN；否则 INCONCLUSIVE。

Oracle 是可用候选上的事后上限，不是服务结果；有限近似 QP teacher 不得标为 full oracle。

### STOP-2 — No dynamic headroom

从 train/dev 选 best-fixed 编辑 a_static 并锁定；不得用每个测试 origin 的 truth 选 static。计算 I_dynamic=E[L_t(a_static)−L_t(a*_t)]，以及相对不编辑的绝对收益，含成本。U_dynamic<δ_dynamic → PIVOT 静态校准并停止动态 planner 主张。不得使用“static 解释 80%”这类近零分母不稳定比率作为唯一条件。

### STOP-3 — No decision-relevant predictability

先分离 oracle-e+oracle-d、oracle-e+pred-d、pred-e+oracle-d、pred-both 四象限。与零响应、常数/平均响应、时间/季节均值和简单线性回归比较。以真实 gain、regret 与有 headroom 样本上的 ranking 为主，cosine/R² 只辅助。

在固定训练/调参预算内，若 predicted-both 相比 best-static 和强 direct gain 的改进上界不超过 δ_predict，且标签效率/目标转移也无可辨优势 → STOP factorized-planning 主张。完整 e0 R² 低不能单独触发停止；方向投影和 ranking 必须失败才有力。

### STOP-4 — Output correction dominates

匹配 context、标签、trainable parameters（同时报告总参数）、训练预算、候选响应 teacher access 与实测在线成本。比较 terminal、joint-trajectory、recurrent residual；后二者是强基线。

若参数编辑的收益不高于最强输出订正（改进区间上界≤0），成本不更低、保护变量/long rollout 不更好，且无预登记标签效率/目标转移优势 → STOP 参数编辑主线或 PIVOT 输出订正。成本不能只用 adapter rank 或 declared token cost 表示。

### STOP-5 — Unsafe trajectory / projection gaming

若 summary gain>0 而完整场/保护变量持续劣化，或 120/240h 的损害超过预登记 non-inferiority margin → STOP 该 summary/window/方法 claim。一次真实发散/NaN 立即触发工程停止并保存失败率，不能在统计表中删除该样本。

### STOP-6 — No reusable factorization advantage

在同信息、接收相同 metric/budget 输入的 direct-gain/router 下，响应路线在新权重/预算与核验标签受限设定均无改进 → 撤回“可复用”创新。不能把“不喂新目标给 baseline”造成的失效当作优势。

### STOP-7 — Calibration invalid

候选集合、metric、bank、backbone 或输入分布变化后不复核校准即声称安全 → INVALID claim。区间实际覆盖显著低于名义目标且无法通过预登记校准恢复 → no-edit 或放弃安全保证；不得把两项风险上界的差当作收益下界。

### 项目级决策

正确性门失败：修复，不宣称科学失败。资源门失败：BLOCKED。H1 失败：不投资主 head。H2 失败但 H3 成功：只可保留经证实的系统/经验贡献。H3 失败且 H2 无独立实证价值：STOP。第二主干未做不会抹去第一主干结果，但必须缩窄 generality claim。

---

## Part X — Experiments Required for Publication

#### 阶段、输入输出与下一步

| 阶段 | 最小输入 | 主要输出/指标 | 通过意味着什么 | 不通过后做什么 |
|---|---|---|---|---|
| P0-01 / S0 | 真 checkpoint、真 ERA5 子集、官方代码、归一化 | 独立 parity、同协议 RMSE、完整指纹 | 比较基础可信 | BLOCKED 或 INVALID，不能进入 oracle |
| P0-02 / S1–S2 | 冻结非零 bank、合法开发 origins | 实际候选响应、oracle 相对 no-edit 的分布与 CI | 动作空间可能有价值 | 一次预登记 bank fallback；仍无效 STOP |
| P0-03 / S3–S4 pilot | 同一 bank/cache、train/dev 分割 | static/dynamic 差异、四象限、简单线性预测 | 动态与可学空间可能存在 | PIVOT static 或 STOP 无可学主张 |
| P0-04 | 同信息简单 residual/direct gain | 初步强对照与成本 | 值得实现主系统 | 明显支配则早停；不以弱基线胜利结案 |
| P1-01–05 | 冻结数据合同、可微 bridge、真实缓存 | e/du、router、joint/recurrent residual、实际规划 | 形成真正可运行的比较实验 | 失败按瓶颈定位，不改测试协议 |
| P1-06 | 锁定方法、最终未使用年份 | 6/24/72/120/240h skill、保护量、效率与 CI | 才开始构成论文证据 | 撤回过强泛化/稳定性结论 |
| P2-01–02 | P1 有效，独立校准分块 | 标签效率、目标/预算复用、no-edit 校准 | 才可强化 V2 方法主张 | 保留 P1 有效结果，撤回增强主张 |
| P3 | 明确主干/极端/谱 failure mode | 第二主干、极端分层、物理诊断 | 扩展范围 | 缩小范围，不抹去已有有效结果 |

#### 必须比较的基线

第一组检查有没有必要动态选择：frozen F0、train/dev 选的 best-fixed edit、训练的 static LoRA、参数量对齐的更大 static LoRA、random edit（含相同幅度/成本）、nearest-regime lookup、oracle edit。

第二组直接攻击因子分解：同一 X/a 输入的 scalar gain head，直接 b/H head，直接 edited forecast predictor，同信息 conditional LoRA/router/hypernetwork。后者需要可微系数路径，不能因为现有 tuple 接口切断梯度就省略它。

第三组直接攻击参数干预：简单偏差校准、terminal residual、joint-trajectory residual、recurrent state correction。主表至少包含后二者。合法已核验误差可用时，为双方提供同样的 EWMA；不允许只给 EarthDelta 更多历史资料。不能使用未来标签驱动 TTA 作为可部署基线；无标签 TTA 必须有预先固定、合理的回归代理目标。

集合比较另报告 ensemble size、CRPS 等对应指标与真实推理开销。概率预报不能只与单轨迹 RMSE 进行片面对比；预算不匹配的更大模型/全量微调属于成本前沿背景，而非自动等价的对照。

#### 核心消融不要超过问题所需

1. Oracle 四象限，full-e 对 projected-e / direct b；识别可学瓶颈。
2. finite actual response 对 local linear response；single 对实际 pair composition；识别真实非线性。
3. analytic gain 对任意 learned score correction、direct gain；识别因子分解必要性。
4. 标签比例变化与固定 head 的 metric/budget 变化；所有基线收到相同条件。
5. no-edit、不同 hold window 与 5–10 日完整场保护；作用结束不重置状态。

标签比例必须按完整训练 origin/时间块采样，而不是让同一 origin 的其他 lead 泄漏监督。若更多无标签 response trajectories 给了 EarthDelta，也要让 direct-gain/其他基线在公平的辅助监督赛道使用它们；同时报告不共享辅助 teacher 的实际系统成本赛道。

#### 统一评价

Forecast：逐变量纬度加权 RMSE、ACC 与固定多变量标准化 MSE。Intervention：actual gain、predicted gain、oracle upper bound、regret、top-k recall、ranking correlation、负收益激活率与 no-edit 率。Response：有符号 cosine、R²、范数校准、投影损失与 lead 衰减。Efficiency：总/可训练参数、离线 probe 调用、总训练预算、在线额外前向、median/P95 latency、显存峰值。

ACC 的气候态定义、季节时间索引和所用年份应锁定；零范数 cosine/R² 退化应记录 NA 而非伪设满分。真正统计单元为起报时间块；多变量/多时效比较必须预先指定主比较与探索性结果，不用挑表格中最佳项证明全局有效。

---

## Part XI — EarthDelta V2

这是建议的下一阶段研究设计，不是对已经完成方法的重新命名。

### Problem statement / scientific question

给定冻结天气预报器、合法起报信息 X、冻结编辑集合 A 和有限核验标签，能否在不运行每条候选 forecast、不看未来真值的条件下，预测足以比较编辑收益的轨迹响应结构，并选择比 no-edit、最佳静态编辑及同信息输出订正更有价值的计划？

核心不应是“更复杂的 LoRA”，而应是：**真实天气中是否存在一个比完整未来误差更容易学习、又足够支持干预决策的响应子空间？**

### 两个核心贡献与可证伪结果

**C1：决策充分的轨迹响应学习。** 在 train-only 数据上定义固定 summary / basis U，保持度量可解释。输出 ê_h、d̂_h(a)，必要时投影到 span{d_h(a)}；比较 full-e、projected-e、直接 b、直接 gain。不是宣称投影公式新，而是检验哪些可预测信息真正驱动天气干预收益。

**C2：可重用、选择性的规划。** 一个预测器服务多个已声明变量/lead 权重和预算；校准候选集级收益高估，保留 no-edit。必须证明迁移无需重新训练 head 或说明少量再校准成本，并与 action/metric/budget-conditioned direct-gain/router 对照，不能用不接收新目标的弱 router。

### 输入输出和训练

首版 action descriptor 使用固定 bank 内身份/系数/作用窗，**不声称未见专家外推**。输入为过去 L 帧的冻结空间特征及可用时间元数据；memory 允许空，默认禁用。首版固定原场变量/纬度加权 summary；U 只能在训练分割拟合。只在 basis 覆盖的空间宣称目标重加权。

离线对同一 X 运行 reference 与所有候选，产生 d 标签；未来 Y 到达才产生 e/g 标签。损失采用

    L = λd ||d̂−d||_Q² + λe ||ê−e||_Q²
        + λg |ĝ−g|² + λr regret_surrogate,

其中 λg、λr 的项只能作用于有核验的训练样本；λr 为可选，先做朴素回归对照。所有 λ、basis 尺寸、K、rank、作用窗只能在训练/开发集选择。禁止利用测试 oracle 反向选 bank。主线中解析 ĝ=2ê^TQd̂−||d̂||_Q²；任意 learned score correction 必须单列 baseline。

### Inference

1. 校验模型、bank、normalization、grid、split、summary、cost profile 指纹。
2. 构建不含 TruthBatch 的 IssueContext；提取上下文特征，记录额外前向成本。
3. 同时预测候选完整多时效响应与参考误差；根据 MetricSpec 重加权。
4. 应用 feasible set（box/support/cost）及集合级收益高估校准；没有可信正收益则 no-edit。
5. 执行一次被选中的编辑 rollout；前 4 步编辑、随后回到 reference 参数，状态不重置。
6. 保存预测分数/决策/成本；真值之后到达时，在独立 evaluator 计算实际收益。

### 既有工作不能“自动覆盖”的部分

不是因为前人没有写相同模块名，而是因为需要验证联合能力：同一合法信息条件下的 signed trajectory response、有限核验标签下的决策充分性、重加权目标的可复用性与保护变量上的实际收益。每一项都可能被直接 gain 或条件 router 同样做到；因此目前仅是待检验差异，不是 exclusivity claim。

### Strongest baselines / decisive experiments

同一 bank 的 best-static、regime lookup、action-conditioned direct gain、b/H predictor、端到端条件 router；直接 trajectory residual correction 和 recurrent state correction；frozen / trained static LoRA / rank-matched larger static LoRA；known e0/du 的四象限 oracle。补充 SPW/ensemble 为不同成本的对照，不把概率集合与确定性单轨迹只比较 RMSE 而忽略 CRPS/预算。

决定性主图是“标签比例—真实增益”“固定 head 改目标/预算”“参数编辑 vs 输出订正的 skill—latency Pareto”“5–10 日收益保持及保护变量损失”，不是模块消融越多越好。

### Kill criteria

采用 STOP_CONDITIONS.md。若 oracle dynamic headroom 不足，不执行 V2；若 direct-gain/router 在公平数据成本下同样好且响应路线没有复用/标签效率优势，撤回 factorization 的方法贡献；若循环输出订正全面支配，停止参数编辑主张；若只有 summary 改善而完整场恶化，停止该 summary 作为主要目标。

### 必须做 vs 有结果后再做

必须做：P0 真实 gate；有限非零 bank；统一 metric；actual/oracle/static；4 象限分解；强 direct gain/router/residual；time-block uncertainty；6/24/72/120/240h。

有结果后再做：P2 决策子空间及目标转移；经验风险控制；若 pair composition 确有收益且线性误差小才恢复 QP。第二主干、神经 memory、JEPA、dynamic rank 不属于当前主线前置任务。

### 五个升级方向的选择

| 方向 | 实质增量 | 风险 | 决定 |
|---|---|---|---|
| Counterfactual intervention world model | 明确 state×edit→trajectory response | 只换名，MPC 早已有该范式 | 保留问题描述，不单列贡献 |
| Learned intervention Jacobian/basis | 从逐候选泛化到组合，降低 probe 成本 | 非线性和未见方向外推 | P2；先 finite exact 验证 |
| Trajectory-level intervention | 多变量/多时效收益与连贯性 | 输出循环订正也可做到 | 主实验设计必须包含 |
| Uncertainty-aware planning | 允许 no-edit，检验错误激活风险 | 校准漂移/假保证 | 集合级经验校准，小规模实现 |
| Physics-structured response space | 可解释尺度/风场方向与保护量 | 谱能量丢相位、投影掩盖损害 | 固定物理度量+诊断优先，spectral loss 暂缓 |



具体投影误差、可复用目标的闭包限制和 serving 保护边界见 Part VII 与 `research/EARTHDELTA_V2.md` 附加合同。

---

## Part XII — Implementation Plan

工程合同不是把研究报告再复制给 Codex。主执行文件把每项任务写成 Scientific purpose → Files to inspect/modify/create → 输入输出与函数 → Tests → Command → Artifacts → Success/failure → Decision → Commit suggestion。

**只解锁 P0-01。** Codex 必须先核查实际工作区版本；若已有更新，只评估相关 diff，不强制回退、不覆盖用户修改。配置中路径和实际资源由本地 inventory 解析；无法满足真实 gate 时报告具体 BLOCKED 项并保留已完成补丁，不编造实验。

| Task | 目标 | 依赖 | 主要触及路径（节选） | 必须产物（节选） |
|---|---|---|---|---|
| P0-01 | 独立官方 S0 与可追溯数据合同 | none | `scripts/s0_gate.py`<br>`earthdelta/bridge/stormer_bridge.py`<br>`earthdelta/data/make_splits.py` | `status.json`, `provenance.json` |
| P0-02 | 非退化编辑库与真实 oracle 上限 | P0-01 | `earthdelta/selection.py`<br>`earthdelta/contracts.py`<br>`earthdelta/lowrank.py` | `bank_manifest.json`, `bank_checkpoint.pt` |
| P0-03 | 静态之外的动态空间与瓶颈四象限 | P0-02 | `earthdelta/paired.py`<br>`earthdelta/analysis/headroom.py`<br>`earthdelta/context.py` | `static_choice.json`, `crossfit_splits.json` |
| P0-04 | 最小残差订正与直接收益挑战 | P0-03 | `earthdelta/baselines/simple.py`<br>`scripts/p0_baseline_challenge.py`<br>`configs/p0/baselines.json` | `baseline_results.parquet`, `fairness_ledger.json` |
| P1-01 | 真实成对轨迹数据集与可用信息接口 | P0-04 | `earthdelta/context.py`<br>`earthdelta/data/paired_dataset.py`<br>`scripts/build_response_dataset.py` | `paired_manifest.parquet`, `context_features/` |
| P1-02 | e0/du、bH、直接收益预测头训练 | P1-01 | `earthdelta/heads.py`<br>`earthdelta/paired.py`<br>`earthdelta/training/response.py` | `head_checkpoints/`, `training_history.json` |
| P1-03 | 可微逐样本计划与显式低秩注入 | P0-04 | `earthdelta/bridge/stormer_bridge.py`<br>`earthdelta/bridge/controlled_projection.py`<br>`tests/test_differentiable_rollout.py` | `pytest.log`, `gradient_audit.json` |
| P1-04 | 强 router、LoRA 和输出订正基线 | P1-01, P1-03 | `earthdelta/baselines/neural.py`<br>`earthdelta/baselines/peft.py`<br>`scripts/train_baselines.py` | `baseline_checkpoints/`, `training_costs.json` |
| P1-05 | 有限响应规划的实际执行闭环 | P1-02, P1-03, P1-04 | `earthdelta/selection.py`<br>`earthdelta/planner.py`<br>`scripts/evaluate_planner.py` | `serving_predictions/`, `decisions.parquet` |
| P1-06 | 长轨迹、独立年份和论文主表 | P1-05 | `earthdelta/evaluation.py`<br>`scripts/evaluate_publication.py`<br>`scripts/make_paper_tables.py` | `overall_results.csv`, `ablations.csv` |
| P2-01 | 决策子空间与目标/预算重用 | P1-06 | `earthdelta/decision_space.py`<br>`scripts/decision_transfer.py`<br>`configs/p2/transfer.json` | `decision_space.json`, `label_efficiency.csv` |
| P2-02 | 候选集级收益校准与不编辑策略 | P1-06 | `earthdelta/calibration.py`<br>`scripts/calibrate_intervention_risk.py`<br>`configs/p2/calibration.json` | `calibration_state.json`, `coverage.csv` |
| P3-01 | 第二 checkpoint / 第二 backbone | P1-06 | `configs/p3/generalization.json`<br>`scripts/evaluate_backbone_transfer.py`<br>`tests/test_backbone_contract.py` | `second_backbone_gate.json`, `generalization_results.csv` |
| P3-02 | 极端事件与物理诊断的定向扩展 | P1-06 | `configs/p3/diagnostics.json`<br>`scripts/analyze_extreme_regimes.py`<br>`tests/test_physical_diagnostics.py` | `event_results.csv`, `physical_diagnostics.csv` |

#### 主要最小补丁

`selection.py` 统一有限/连续分支的可行性检查，而非重写正确的 QP；`paired.py/geometry.py/heads.py` 统一 MetricSpec 和 actual/proxy/corrected score 语义；`stormer_bridge.py` 保留旧枚举 API，新增可微、逐样本、返回轨迹的显式系数入口；`make_splits.py` 修正 UTC、真实内容指纹和 coverage。旧 neural memory/spectral 工具保留文件，不成为当前主线依赖。

期望结果保存到 `artifacts/p0|p1|p2/<task>/<run_id>/`，至少 `config.json`、`status.json`、`metrics.json`、`summary.md`、`stdout.log`、provenance，预测与图片独立目录。任何结果不得覆盖历史版本；真实来源身份必须包含 checkpoint、bank、normalization、grid、summary/metric、split、continuation。

文件清单和完整符号/I/O 在 `implementation/FILE_BY_FILE_PLAN.md`、`implementation/TEST_PLAN.md`；任务图的机器可读版本在 `codex_tasks.json`。`UNVERIFIED` 依赖在导入/复制前锁定提交与许可，不用旧 manifest 的短 SHA 当完整证明。

---

## Part XIII — Paper Story & Claims

#### 三种强度的论文中心句

**Conservative — 当前可以准确说的版本：** EarthDelta 实现了冻结天气模型可逆低秩干预的响应建模与选择原型，并明确区分真实有限响应、局部几何和起报时收益预测，为检验参数编辑相对静态适配与输出订正的必要性建立可复现的验证合同。

这句话不声称提升准确率，也不把测试原语包装成已验证方法；作为当前工作进展介绍可以成立，作为顶会论文贡献仍不够。

**Strong — P1 核心实验成立后：** 针对起报时无法获得未来核验、盲目调整又可能损害多步预报的问题，EarthDelta 分离学习无需未来标签的参数干预响应与需要延迟核验的决策相关误差，在执行前选择低秩编辑，并通过独立年份、强直接路由与轨迹输出订正对照验证其实际增益及边界。

**Ambitious — P2 额外实验成立后：** EarthDelta 学习冻结天气预报器的决策充分轨迹响应子空间，在核验标签受限时支持可校准的不编辑/编辑选择与同一代理上的目标、预算复用，并揭示这种结构何时比直接风险预测和输出订正更有效。

“更少标签”“更有效”“可校准”都是需要结果后才能使用的限定，不是本报告确认的性能。

#### 最多三个 contribution

C1 方法/问题：分离模型可生成的响应监督与滞后核验的误差监督，并研究决策充分表示；不把恒等式当新算法。

C2 方法/能力：在明确目标族内复用响应预测并允许 no-edit，通过候选集级校准控制经验错误激活；不声称任意分布下安全。

C3 经验：系统测定 oracle/static/learnable 的差距，比较强轨迹订正与多步保护，发布可追溯结果与失败边界。这不是第三套额外模块。

#### 建议 paper story

已有预报模型可以继续训练或集成，不能说“所有部署模型永远冻结”。本研究选择重用一个冻结资产的受限场景。由于未来核验不可见，每次修改不一定有益。我们先证明一个具体编辑空间中存在超出静态方案的改善余量，再问起报信息能否预测该余量，并检验响应分解是否比更简单策略更值得。最终展示收益、计算代价、拒绝编辑和失效边界，而不是从算法包装直接跳到 operational claim。

#### Figure / table plan

| 产物 | 应回答的问题 | 不合格的替代 |
|---|---|---|
| Figure 1 oracle / best-static / no-edit 分布 | 动作与动态选择分别有多大空间？ | 只报 oracle 最大单例 |
| Figure 2 四象限与方向投影误差 | 误差、响应还是排序是瓶颈？ | 只报平均 R² |
| Figure 3 核验标签比例曲线 | 因子分解是否节省真实标签？ | 额外 teacher 只给本方法 |
| Figure 4 skill–latency Pareto | 参数干预是否比强输出订正值得？ | 仅比较参数个数 |
| Figure 5 6–240h 多变量保护 | 改善能否持续，有无隐藏损害？ | 只报有利 lead/summary |
| Figure 6 固定 head 的目标/预算变化 | 代理是否真正可复用？ | baseline 没有收到新目标 |
| Figure 7 校准覆盖/no-edit/增益 | 何时应当不编辑？ | 两个风险上界相减称收益保证 |
| Table 1/2/3 | 整体结果、必要消融、真实成本 | 混合 oracle 与可部署结果 |

---

## Part XIV — Final Assessment

#### 1. 当前版本直接写论文，novelty 是否足够？

**按强 ICLR/NeurIPS/ICML 方法论文标准，目前不足以给出有把握的肯定判断。** 公式、低秩库、支持枚举/QP、状态条件化、响应代理和可逆执行分别有成熟先例；仓库又尚未提供本次可核验的真实干预收益与强基线结果。不是因为项目“没用”，而是尚未证实剩余交叉差异有科学后果。

#### 2. 实际科学价值是否足够？

**足以值得做一个严格、有限预算的 P0，而不足以立即支撑 operational 部署价值。** 能否在未来真值未知时预测参数改动的效果，及其相对输出订正的适用边界，是清楚的可证伪问题。真实价值必须用完整轨迹收益、标签/计算效率或可复用性证明。

#### 3. 最容易被 reviewer 攻击的三点？

第一，已有 Green’s-function/FSO 与价值路由组合可以解释大部分方法；第二，e0 监督天然可以训练 residual corrector，为何还要绕到参数空间；第三，验证基础不够——内部 bridge 互比、oracle 与部署混淆、投影与完整场指标混淆、未有长期与公平强基线证据。第三点可修复，但不能靠更多单元测试数量回答。

#### 4. 当前最值得保留的创新候选？

**把无需未来标签的模型响应与需要延迟核验的决策相关误差分开，并检验这种结构能否带来可重复的标签效率和目标复用优势。** 其中最值得深入的是响应方向上的可预测性，而不是完整天气误差是否能被高精度重建。

#### 5. 最应该删除或降级的设计？

从主线删去“新二次恒等式”“QP 本身新”“world model 名称即新颖”“LoRA 自动保证物理一致”这些 claim。JEPA、神经 memory、dynamic rank、spectral loss 与复杂 Atlas 暂缓；将谱/各向异性留作诊断，将 EWMA 留作简单公平基线。不要删除已经可靠的通用代码来制造无谓重构。

#### 6. 下一阶段只能做三件事？

（1）重建独立官方 S0，同时冻结真实数据/归一化/单位/时间/版本合同。

（2）构造并核验实际非零编辑库，测 no-edit → best-static → per-origin oracle 的差距和不确定性。

（3）在同一缓存和信息条件下做四象限、简单 direct-gain、trajectory/recurrent residual 挑战，判断是否值得训练完整预测器。没有必要先研究新的 memory 架构。

#### 7. 什么结果应终止当前路线？

经一次预登记的合理银行重构仍无有意义 oracle 空间；或动态空间几乎不存在；或在固定训练预算内实际选择不优于静态且没有标签/复用优势；或强循环输出订正在精度、成本、稳定性上支配参数编辑；或改善只存在于 summary、短时效而完整场/5–10 日系统性恶化。使用置信区间与事先冻结的阈值，不用一个未经验证的 R² 阈值或任意 2% 决定项目命运。

#### 8. 什么结果值得继续投入并冲击高质量会议？

独立官方 gate 通过；动态 oracle 空间在独立时间块上稳定；服务可用信息的预测能兑现明显比例的空间；在同 teacher/标签/计算条件下胜过强 direct-gain/router 或提供清晰的标签效率/目标复用优势；对 joint/recurrent residual 仍有可解释的成本—效果优势；收益跨关键变量与长期 lead 保持，并公开负结果、选择性不编辑与成本。第二 backbone 能增强范围，但不能代替这些核心证据。

**最终建议：保留方向，降低现有 novelty 包装，先让 P0 决定是否继续。当前的最佳投入不是多训练一个新模块，而是完成能让这条路线被否定、也能让它被真正相信的三组关键对照。**

---

## 证据入口与执行包边界

核心源码链接已在 Part VIII 的逐项表中固定到审计 SHA。原论文与已定位代码在 Part IV；实际已核查的外部版本与许可在 `implementation/LITERATURE_TO_CODE.md`。ERA5 可用性说明依据 [Copernicus 官方 ERA5 说明](https://climate.copernicus.eu/what-copernicus-climate-change-services-era5-reanalysis-dataset)：快速 ERA5T 与最终 reanalysis 不是六小时就可核验的同一产品，回放协议须显式区分。

本交付包中的 `tools/audit_math.py` 只演示独立代数与反例；`tools/validate_package.py` 只检查工程合同文件和任务图。其通过结果不能升级为 repository pytest、官方 Stormer parity、真实 ERA5 质量验证或任何预报性能结果。
