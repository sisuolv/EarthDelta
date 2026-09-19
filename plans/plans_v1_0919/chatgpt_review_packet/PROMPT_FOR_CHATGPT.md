# 给 ChatGPT 的评审提示词：EarthDelta 研究计划的独立红队分析

> 使用方法：把本提示词整体粘贴给 ChatGPT，并上传同目录下的 `EARTHDELTA_REVIEW_PACKET.md`（已把全部材料按阅读顺序拼成一个文件，约 6 万中文字 + 英文调研报告）。如果模型上下文不够，优先上传主文档 `EarthDelta_v6_Review_and_Plan_CN.md` 与 `survey_D`、`survey_B`。

---

## 你的角色

你是同时具备两种背景的资深评审人：

1. NeurIPS / ICLR / ICML 领域主席级别的机器学习审稿人，熟悉参数高效微调（LoRA、hypernetwork、动态路由、混合专家）、决策导向学习（predict-then-optimize）、世界模型与 JEPA、模型合并（Fisher/RegMean/TIES）、摊销优化。
2. 数值天气预报与数据同化专家，熟悉 FSO/EFSO 预报敏感度、伴随与切线性模型、Green's-function 参数校准、WeatherBench-2/X 评测规范，以及 GraphCast、Aurora、Stormer、FuXi 等机器学习天气模型的训练与微调实践。

你的任务是**红队式审阅**：找出会让这篇论文被拒、或让实验白做的问题，并给出可操作的修改建议。不要复述材料，不要客套。

## 材料清单（附件）

1. `EarthDelta_v6_Review_and_Plan_CN.md` —— 合并后的 v6 方案与 novelty 分析（**主文档**，中文）。
2. `research_spec_v6.yaml`、`PACKAGE_MERGE_MAP.md` —— 研究规格与代码合并映射。
3. `survey_A` ～ `survey_E` —— 五路文献调研原始报告（英文，含 arXiv 号、日期、重叠等级）。
4. 两份原始计划：`IMPLEMENTATION_PLAN_CN.md`（v5 线：Paired-Response Latent Planning）与 `CODEX_PLAN_CN.md`（Response Kit 线：Interaction-Aware Trajectory Repair）。

如果你只收到本提示词而没有附件，请基于下面的"核心摘要"先给初步分析，并明确标出哪些判断必须看到附件才能确认。

## 核心摘要（英文，便于精确对齐术语）

**Setting.** Frozen ML weather backbone: Stormer (1.40625°, 128×256 grid, 69 channels, 6-hour steps; DiT-style ViT, 24 blocks, d=1024; official checkpoints exist for patch size 2 and 4). A static LoRA reference F_s (rank 16 on `attn.proj` of the last 4–6 blocks, multi-step loss) plus a small dictionary of K≈8 low-rank "edits" trained as heterogeneous experts (by flow regime or error mode) on top of F_s.

**Decision problem.** At each forecast issue time the controller sees only legal history (4 input frames, pooled token features from the reference model's step-0 forward pass, optionally an EWMA of recent *verified* errors filtered by availability time) and must choose which edits to apply during the first 4 rollout steps (coefficients a ∈ [−ρ, ρ]^K, at most m active), with no access to future truth. No-edit (= F_s) is always a candidate.

**Method (the claimed contribution).** Instead of mapping history → coefficients end-to-end (router / hypernetwork trained with a multi-step loss), learn two things separately: (i) the *label-free response* du_k = F_edit_k − F_s of the forecast to each edit (needs only two rollouts, no truth), and (ii) a *flow-dependent forecast of the reference error* e0 = Y − F_s (needs truth, available with delay). Select by the exact quadratic gain 2⟨e0, du⟩_W − ‖du‖²_W = ‖e0‖² − ‖e0 − du‖² (identical to the FSO/EFSO observation-impact measure). Continuous variant: an offline teacher builds a central-difference response matrix R over the edit coefficients, forms H = RᵀQR and b = RᵀQe, solves a box-constrained least-squares (the Green's-function-calibration normal equations), and verifies the top candidates with full nonlinear rollouts. Students are either *fully amortized* (predict a* with an H-weighted imitation loss, which equals output-space squared error because du is linear in a) or *semi-amortized* (predict (b, H) from history and solve a tiny QP online; off-diagonal H = interaction between simultaneously applied edits). After our literature survey, the claim was narrowed to: **"amortizing forecast sensitivity to parameter edits into an issue-time controller"**, positioned against adjoint parameter sensitivity (diagnostic, needs verifying analysis) and Green's-function calibration (offline, global, non-amortized).

**Main experiments.** P1 headroom gate: hindsight-best candidate vs best-fixed candidate vs F_s vs frozen F_0 at 24/72/120 h (go/no-go). P2: response route vs matched end-to-end route (regret, RMSE gain, harmful-edit rate, calibration) with identical context encoder, dictionary, data and compute. P3: zero-shot transfer to new budgets, held-out dictionary members (few-shot response probing for new experts), new lead times, and label-scarce e0 (du stays 100 % unlabeled). Mechanism: full vs diagonal H; imitation vs coefficient loss; memory-feature ablation; descriptor shuffle; same-rank-different-direction; equal-energy-opposite-phase.

## 我们已经找到的最近邻（请不要重复罗列；如果你认为我们对某篇的理解有误，请指出）

- 选择规则最近邻：VI-MoLE（arXiv 2608.02528）；MAPLE（2608.15299）；LiST（2608.22370）；CCM-LoRA（ACL 2026 Findings 1329）；DISeL（2605.19028）；CoMoL（2603.00573）；CLAW（2609.12278）；LoRA logit-shift 一阶分解技术笔记（2604.20313）；HyperFix（2608.11499）；LoraHub、AdaMerging。
- 架构最近邻：Adapter Banks for Compositional Motor Control（2609.17042）；Aurora 代码内置的按 rollout 步索引的 `LoRARollout`；WeatherPEFT（2509.22020）。
- 教师侧最近邻：Green's-function 参数校准（Menemenlis et al. 2005 MWR；Strobach et al. 2022 GMD）；CCM "Discovering Physical Directions in Weight Space"（2605.14546）；Vonich & Hakim（2504.20238，oracle 初值优化）。
- 无未来真值的敏感度选择：Honda & Yamazaki 2024 GRL（ML 学参考态以实现实时 proactive QC）；PEFSO（2609.12296）；FSO/EFSO（Langland & Baker 2004；Kalnay et al. 2012）；PEAR（2605.01361）。
- 误差预测 / 记忆：GeoQ（2608.21652）；HRRR 误差 LSTM（2512.14898 / 2606.19026）；ORCA（2606.14222）；RATL（2609.03937）；STEPS（2605.08005）；TEFL（2602.22520）；McCast（2605.13197）；HERA（2608.05523）；CRAFTER（2608.05207）。
- 天气领域控制器 / 编辑：ARROW（2510.09734）；ARC-STAR（2605.22222）；TaCT（2603.19325）；SPW 随机权重扰动（2609.08412）；FTAE-Weather（2608.09948）；AdaWeather（2606.02663）。
- JEPA 系：EPM-JEPA（2606.12979，空结果）；JEPA-Anything（2609.20800）；SG-JEPA（2609.10464）；Spectral-Target JEPA（2609.04264）；IMPLY（2609.12441）；Intervention Gap（2608.29998）；JEPA-x（2608.24044）；PhyLatent（2608.05720）；DA-LeWM（2608.18746）。
- 谱与合并：AMSE（2501.19374）；FastNet（2509.17601）；RegMean（2212.09849）；Fisher merging（2111.09832）；OBC/GPTQ。

## 请按下面的结构回答（每条给 1–5 的置信度，并区分"来自材料的事实"与"你的先验判断"）

**Q1 独立的 novelty 裁决。** 材料声称唯一未被占据的空白是："从合法历史同时学习 e0 与逐编辑响应 du，在起报时刻用精确二次收益在预算内选择参数空间编辑，并用学习到的非对角交互矩阵处理编辑重叠"。你是否同意？你是否知道任何**已经做了这件事**的工作（ML、NWP、控制、运筹任一领域）？请特别检查：集合权重学习（ensemble weighting）、多模型后处理、模型误差参数的在线估计、贝叶斯优化中的"预测干预效果"类工作、以及 LLM 领域的"预测微调/编辑效果"类工作。

**Q2 最强反对意见。** 请以审稿人身份写出三条最可能导致拒稿的意见，并为每条给出我们最好的回应与需要补的实验。至少要包含对"端到端 router/hypernetwork 调好了一样强"这条的最强 steelman，以及对"这只是 FSO + Green's function + LoRA 的组合"这条的回应。

**Q3 headroom 的先验估计。** 在 Stormer 1.4° 这样的强主干上，几个 rank-16 的 LoRA 专家之间"逐起报选择"能带来多大的 RMSE 改进？请给出你的数量级估计与依据（可参考已知的 LoRA 微调天气模型的收益幅度、多模型集合平均的收益、流依赖偏差订正的收益）。如果你认为 headroom 不足，请提出 2–3 个 headroom 更大的实验设定（例如分布偏移年份、极端事件、更弱主干、更长时效、区域下游任务），并评估每个设定对 novelty 主张的影响。

**Q4 实验设计审查。** 检查 P1/P2/P3 与基线清单：(a) 是否存在信息不对称或算力不对称使对比不公平？(b) 是否有泄漏风险（时间划分、记忆、归一化、专家训练集）？(c) 统计检验是否合适（按起报时间块的配对 bootstrap）？(d) 最小决定性实验集是什么——如果只能跑 5 个实验，跑哪 5 个？(e) 缺了哪些必要消融？

**Q5 被降级模块的价值。** 方案把 v5 的 JEPA 潜空间、EMA 目标编码器、gated-delta 记忆、谱训练目标全部降级为消融。这样做是否扔掉了有价值的东西？特别是：e0 的预测在高维场上是否需要潜空间表示才能做好？记忆机制是否是 e0 可预测性的关键？

**Q6 主张与标题。** 给出你认为最紧、最可辩护的一句话贡献陈述（中英各一版）和 2–3 个候选标题；并指出材料中哪些表述仍然过度声称。

**Q7 工程决策审查。** 评估以下决定的风险：数据路径（WB2 GCS 不可达，候选为 hf-mirror 的 1.5° 镜像 + Stormer 官方保守重网格、或 CDS 服务端插值到 1.40625°）；把 xformers 换成 SDPA；torch-harmonics 等角网格含极点与 Stormer 无极点网格的不匹配；第二主干选 ArchesWeather-M 或 ClimaX 而非 Aurora；控制器输入改用冻结主干第 0 步的池化特征。

**Q8 Top-10 修改清单。** 按"对录用概率的影响 × 实施成本"排序，给出十条最值得立即做的修改。

## 输出要求

- 用中文回答，术语可保留英文；每个判断给出理由，能给数字就给数字。
- **不要编造引用。** 只引用材料中出现的论文，或你确信存在的论文；对后者标注 [需核实]，不确定 arXiv 号就写"不确定"。
- 区分事实与推测；对每条结论给 1–5 的置信度。
- 优先给可操作的修改，不要泛泛而谈；总长度 3000–5000 字，可用表格。
