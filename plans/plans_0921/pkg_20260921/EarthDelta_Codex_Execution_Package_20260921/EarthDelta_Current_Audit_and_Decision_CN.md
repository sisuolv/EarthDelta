# EarthDelta 当前版本审查与研究决策

审查提交：main@4fe55a7af90ea92f62a3232a571af92bfbd6114d。正文是研究审查；执行请用主Codex合同。

# 当前方法审查：main@4fe55a7af90ea92f62a3232a571af92bfbd6114d

审查日期以 2026-09-21 UTC 记录（美国东部仍为 9 月 20 日）。依据两份用户任务文件、GitHub connector 读取的固定提交、论文/官方来源。没有写入 GitHub、没有启动用户 GPU、没有访问本地真实天气权重或数据。

## 当前 scientific core

不是旧 Notion 的 Complexity Atlas。当前代码形成两个相邻实验路径：

1. `paired.py + ComposedPredictionHead`：预测 reference error `e0` 与候选有限编辑响应 `du`，生成候选分数。
2. `probe.py + geometry.py + teacher.py + selection.py`：以有限差分响应 R 构造局部二次代理，离线解有界教师，在线可以用预测 b/H 规划。

两者尚未被证据证明为同一个已训练、同一指标下的部署系统。主线选择有限候选的真实 `du`，局部 R/QP 降为近似对照，不同时训练所有架构。

## 已有资产与证据级别

| 项目 | 状态 | 边界 |
|---|---|---|
| K-expert low-rank，dense/sparse 路径 | SOURCE_IMPLEMENTED | 稀疏路径确实跳过不活跃 expert/行；不是旧kit仅乘零。端到端GPU加速未验证 |
| paired target、finite cached responses、central differences、box teacher | SOURCE_IMPLEMENTED | 数值原语存在不等于真实天气闭环完成 |
| 多lead paired head，参考误差与编辑响应解码、gain calibration | IMPLEMENTED_NOT_WEATHER_TRAINED | README 未提供真实训练结果；不得要求从零另造相同多lead头 |
| 本地 Stormer 架构与 bridge | IMPLEMENTED_PARTIAL_VALIDATION | 官方独立真实权重 parity 尚未成立 |
| 测试 | REPO_REPORTS_192_PASS_3_SKIP | 本次未在完整仓库运行；读取了测试与门禁源码 |
| 2020 ERA5 数组1464×69×128×256 | REPOSITORY_REPORTED_ONLY | bytes、数据质量、实际 time axis 未在本次环境验证 |
| 两个 checkpoint 文件 | REPOSITORY_REPORTED_ONLY | 不把文件名/大小当成功加载或SHA核验 |
| S0 real checkpoint gate | NOT_ESTABLISHED | 源码存在不等于已真实通过 |
| oracle ceiling、模型预测收益、长期确认 | UNVERIFIED / NO_RESULT_FOUND_IN_READ_SCOPE | 本次不能报告数值天气增益 |
| memory/spectral | PARTIAL / DEFER | memory有校验问题；谱桥/JVP有明确占位 |

## 静态发现

### A01 — P0
`scripts/s0_gate.py` / `verify_zero_edit_equivalence / run_s0_gate`

本地 forward_validation 与本地 controlled_rollout 比较，后者传入空 expert_loras；不是独立官方实现对照。gate 总结没有将全部加载、有限数值和 RMSE 条件纳入。

证据：STATIC_CONFIRMED。最小处理：增加独立 upstream 路径、真实非零 bank 的零系数测试；失败/缺失条件阻断正式 S0；保留旧报告。

### A02 — P0
`earthdelta/bridge/stormer_bridge.py` / `NormalizationContract.from_npz_dir / denormalize_diff`

存在 diff_mean 文件时会使用它；已核查 upstream inference.py 显式使用零增量均值。

证据：STATIC_CONFIRMED_BEHAVIOR; HISTORICAL_RESULT_IMPACT_UNVERIFIED。最小处理：增加 normalization_policy，默认 pinned_inference_zero_diff_mean；legacy 模式只读复算；nonzero diff_mean 反例和真实官方 parity。

### A03 — P0
`earthdelta/selection.py` / `plan_from_prediction / _plan_from_finite_candidates`

有限候选分支没有执行 bound、max_active 和候选数量上限；缺少对候选有限值/类型的完整验证。

证据：STATIC_CONFIRMED。最小处理：公共 validate_feasible_candidates；非法候选显式拒绝；有限/连续共享域；保留 no-edit。

### A04 — P0
`earthdelta/paired.py; earthdelta/geometry.py; earthdelta/heads.py` / `quadratic_gain / from_error / ComposedPredictionHead.forward`

paired 对 F 权重归一化，geometry 用未归一化权重，head 用无权均值并叠加 trainable gain_calibration；不是同一个统一分数。

证据：STATIC_CONFIRMED。最小处理：MetricSpec 绑定变量尺度、面积、lead、投影和约简；gain_analytic 与 gain_calibrated 分开；默认禁用校准项。

### A05 — P0
`earthdelta/bridge/stormer_bridge.py; earthdelta/contracts.py` / `load_stormer_checkpoint / NormalizationContract.digest / ArtifactVersion`

backbone 身份主要是架构而非实际权重 SHA；norm digest 不绑定变量名/步长键；grid 名称不是实际坐标身份；版本检查未成为执行入口必经门。

证据：STATIC_CONFIRMED。最小处理：内容哈希与结构schema；字典、静态适配、projection、split、continuation 都在 serving/caching 前核验；兼容旧schema只读。

### A06 — P0
`earthdelta/data/make_splits.py` / `build_manifest / compute_available_time`

6h 核验延迟为手设情景；naive datetime.timestamp 依赖进程时区；event_id 是起报时效键而非独立天气过程；边界没有证明实际样本完整。

证据：STATIC_CONFIRMED。最小处理：显式 timezone.utc，available_time provenance/role，actual time index 完整性；独立 process_group_id；默认 retrospective_open_loop，memory 关闭。

### A07 — P0
`earthdelta/bridge/stormer_bridge.py` / `controlled_rollout`

目前仅返回末态；EditPlan 浮点 tuple 转 tensor，不能把该接口直接当控制器端到端可微输出通道。共享 forward_hooks 已有串行清理但并发/重算未证明。

证据：STATIC_CONFIRMED / CONCURRENCY_UNVERIFIED。最小处理：保留末态兼容，新增 return_trajectory 和系数张量路径；显式上下文与 checkpoint replay 测试，不随意全重写。

### A08 — P1
`earthdelta/heads.py; earthdelta/contracts.py` / `ComposedPredictionHead / EditPlan.descriptor`

多时效接口已存在；描述符是 active、coefficients、hold，没有新专家内容的表示；未证明新字典/新时效迁移。

证据：STATIC_CONFIRMED; TRAINING_UNVERIFIED。最小处理：保留已有多lead头；先做同字典未见幅度/窗口/组合；全新专家推迟并另设描述符/标定协议。

### A09 — DEFER
`earthdelta/memory.py` / `ewma_error_feature_batched`

捕获所有 ValueError 后归零，可掩盖版本不匹配或重复ID；zip 输入长度可截断；默认 EWMA 未充分验证 decay。

证据：STATIC_CONFIRMED。最小处理：本轮主线禁用 memory；重启该模块前窄化异常、长度和衰减验证、跨时区/来源测试。

### A10 — STATUS
`earthdelta/probe.py; earthdelta/spectral.py` / `cached_responses / central_response / jvp_response / band_energy`

有限候选 du 和局部导数均有实现，但 JVP、band_energy、single_mode_energy 明确为 NotImplementedError。

证据：STATIC_CONFIRMED。最小处理：准确标注实现状态；P0 优先 cached nonlinear du，不扩充谱/JVP。

### A11 — P0
`plans/plans_v1_0919/v6_draft/research_spec_v6.yaml` / `context_encoder / claim / falsification_gates`

将 step0 后段 reference tokens 称为零额外成本、预写 beats/zero-shot、固定2–3%门槛没有当前测量依据。

证据：STATIC_CONFIRMED_PLAN_NOT_RESULT。最小处理：计 reference preview 与必要重放，改可证伪假设，δ_min 从 pilot噪声/价值/功效预登记；不把 old claim 当事实。

### A12 — STATUS
`tests/test_bridge.py; README.md` / `CPU tests / real checkpoint skips`

README 同时写195 tests及192 passed 3 skipped；已读测试包含随机小模型、归一化零均值fixture、可跳过真实资源路径。

证据：REPOSITORY_REPORTED; NOT_RERUN_HERE。最小处理：保存现有测试全部；本次不宣称195通过，不把原kit21/50计入当前仓库独立天气证据。

### A13 — P0
`reference/_manifest.json` / `license / pins`

已有46项引用清单；WeatherPEFT、CoMoL、GEPS、W2T 等标NONE，短SHA不能替代完整锁和实际许可确认。

证据：REPOSITORY_REPORTED。最小处理：优先许可明确的基础代码；NONE视为复用阻塞，不重新发布源码；可独立实现数学基线并保留引用。

## 当前代码不能证明的事项

没有证据证明双头优于直接 gain；没有证据证明参数编辑优于多变量反馈式输出订正；没有证据证明 du 的训练集拟合可迁移到未见动作组合。不得由测试数量、模块数量或数据文件数量推断这些。

特别保留已有正确部分：`ExpertLoRA.forward_sparse` 的真实跳过、try/finally hook 清理、float64 teacher算术、no-op教师候选、finite响应缓存、按可用时间筛选记录。修改应围绕缺口进行。

## 源码定位

所有代码事实均对应固定提交目录：https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/
逐条机器可读证据见 `evidence/audit_findings.json`。本报告是定向静态审查而非逐行全仓形式化验证。大型历史 review packet、所有脚本和完整测试未逐行复核；当前外部部署资产未 materialize，未重跑当前仓库测试。


---

# Novelty判定

结论：相比最初physics-state-conditioned LoRA，科学问题更明确、可证伪性更强；实证novelty尚未建立。当前代码只是把“预测再编辑”的候选机制落实为若干可测试原语，没有给出真实天气优势。

## 可以争取，但尚未证明

- 逐大气状态、逐候选参数编辑的多时效响应预测是否有稳定、可泛化的规则。
- du的label-free仿真监督与e0的核验标签分离，是否比同样拥有这些数据的directgain/直接edited-forecast模型更省真实标签。
- 对持出的幅度、窗口或组合，响应结构是否提高同预算选择质量。

## 不成立的默认推理

“du是确定性函数→一定容易学”不成立；输入摘要可能丢失决定响应的信息。
“e0拟合得好→参数编辑必要”不成立；它同时增强了直接残差基线。
“正交参数→互补天气作用”不成立；应测实际响应。
“有限d的R低秩→天气修正低维”不成立；这是矩阵列数上界。
“更晚年份→物理OOD”不成立；最多先称时间外推。
“descriptor支持任意数目candidate→zero-shot新专家”不成立；新专家权重内容尚未编码。

## 五个增强方向的取舍

1. Intervention world model：可以作为解释性比喻；当前本质是model-response surrogate。若没有可组合transition、闭环状态更新与未见动作验证，不把它升级成完整世界模型。
2. Learned intervention Jacobian：参数敏感度/学习Jacobian有先例；代码JVP尚占位，且大幅编辑需要非线性响应。DEFER，不作为救novelty的改名。
3. Trajectory planning：KEEP（唯一主增强）。利用已有多lead头，但补actual rollout监督、共同目标和持出动作测试。
4. Uncertainty-aware：P2可选，校准的是预测收益/误修正风险，不能凭均值减方差宣称保证安全。
5. Physics-structured utility：先diagnostic，固定线性coefficient/vorticity算子可作后续；功率/动能非线性须端点分别变换。谱损失不进入P0。

## 模块裁决

| 模块 | 所针对瓶颈 | 本轮决定 |
|---|---|---|
| 短历史state encoder | context是否含du/e0信息 | KEEP现有，所有baseline同权限 |
| JEPA | 是否有可测响应表征样本效率瓶颈 | DEFER，无此证据 |
| memory / regime retrieval | 是否反复出现可复用误差模式 | DEFER；先静态均值/EWMA同信息对照 |
| dynamic rank / expert count | 实测质量成本曲线有无非平凡需求 | DEFER；有限候选与真实成本已够 |
| spectral state/loss | 主空间是否漏掉影响决策的结构 | DIAGNOSTIC_ONLY |
| hierarchy / layer routing | 候选空间是否明显不足 | DEFER，不能以范围增大替代P0 |
| global/context pooling | 避免过强压缩和混淆季节地域 | KEEP并加入context-only基线 |
| multi-edit interaction | 是否存在真实非加性或Gram响应重叠 | P1配对检验；不是默认第三创新 |

## 审稿防守的最小证据

真实bridge可复核；非零bank在独立dev有上限；动态相对静态有余量；du优于no-effect和bankmean；e0的决策投影可预测；双头胜directgain至少一个预登记维度；参数编辑不被feedback残差支配；多时效和真实成本可复算。缺任何关键一项，收窄相应贡献。


---

# 数学合同与方法边界

## 1. 统一参考与符号

F0 是原始冻结 checkpoint。Fs 是只在训练区间拟合并随后冻结的静态适配参考。S0 验证 F0；科学干预实验的 zero-edit 必须复现所声明的 Fs，不得在比较时偷换参考。没有 Fs 时可登记 reference=F0，但要单列轨道。

令 y0 = D(Rollout(Fref,x,0))，ya = D(Rollout(Fref,x,a))，y = D(Y)。定义：

    e0 = y - y0
    du(a) = ya - y0
    e_a = e0 - du(a)
    gain_Q(a) = ||e0||_Q² - ||e_a||_Q²
              = 2 e0ᵀ Q du(a) - du(a)ᵀ Q du(a).

这是平方损失的代数恒等式，不是新理论。`paired.py`、`geometry.py` 的主计算采用 truth-reference；附件中出现 reference-truth 的表述不能与正号公式混用。若坚持 error=prediction-truth，则式子第一项应改成 -2<error,du>。

## 2. 三类响应不能混称

- `cached_responses`：对有限 candidate 实际运行得到的端点差，是当前神经模型及 D 下的有限响应；不是实际大气干预。
- `central_response`：在 a0 邻域估计 Jacobian R；Ra 仅局部近似。
- head 输出：learned approximation，任何解析 gain 都是 predicted gain，非真实保证。

在固定 T 上分别计算 e=T(Y)-T(F0)、du=T(Fa)-T(F0)，上面的恒等式对任意固定 T 都成立。线性/仿射性是为了将 D(Y-F0) 与 D(Y)-D(F0) 互换、以及线性叠加/解析R路径；不是平方恒等式本身的必要条件。主协议继续用冻结线性 D，修正文档而不暗改实验。

## 3. MetricSpec

约定 decoded shape e:[B,H,S,F], du:[B,K,H,S,F]。保存变量次序、原始单位、只在允许训练期计算的 scale[F]、lead_hours[H]、spatial support与真实面积、D的内容hash及共同有效mask。

    Q[h,s,f] ∝ lead_weight[h] * cell_area[s] * variable_weight[f] / scale[f]²

若D已标准化变量，则Q不得再次除scale²。约定Q在完整 H×S×F 上总和为1，只归一化一次；报告每变量/每lead损失使用对应条件分母。所有方法共享有效mask，不根据预测好坏删点。MSE先累加再sqrt得到RMSE，禁止先逐起报开方再平均后称 pooled RMSE。ACC使用冻结训练气候态，零方差显式missing。

当前三个路径不一致：paired只归一化最后一维；geometry不归一化；head无权mean并加calibration。必须保留raw/calibrated两列，并默认 analytic-only。部署API显式返回 `gain_analytic`、可选`gain_calibrated`，不得共用一个含糊的`gain`。

## 4. 多编辑

如果 du(a)≈Ra，则 H=RᵀQR、b=RᵀQe0，gain≈2bᵀa-aᵀHa。非对角项来自平方范数，是已有二次代理的交叉项；H为响应Gram，不是完整非线性天气损失Hessian。

对真正的组合动作需另外量化：

    interaction(a,b) = du(a+b)-du(a)-du(b)  (零参考)

它不等于H_ij。finite动作模式直接采集组合端点、无需假设线性。对角H与完整H、线性合成与直接组合响应分别对照。

## 5. e0是否等于重新预报？

若允许信息I中Fref(x)已知，则

    E[e0 | I] = E[Y | I] - Fref(x).

因此在无限表达/足够训练及平方损失下，准确的条件均值误差订正本身就是Bayes最优预测。不能证明参数编辑普遍优于不受约束的输出纠错。实际价值只能来自有限样本、低参数、约束空间、跨lead复用与成本的归纳偏置。

但选择编辑只需要e0在候选响应张成空间中的投影：若e_perp与所有du在Q内积下正交，e_perp不影响gain。没必要要求小头重建所有不可预测噪声。首轮固定可解释响应空间，用oracle e0/predicted du四格实验检验瓶颈。

## 6. 均值代入不是自动得到期望收益

若只给压缩context c，e与du在条件c下仍有随机性，则

    E[g|c,a] = 2 μeᵀQ μd - μdᵀQ μd
               +2 tr(Q Cov(e,du|c,a)) - tr(Q Var(du|c,a)).

若完整初始输入、模型与动作固定，du是确定性模型输出；此时不存在这种物理条件随机性，但学习器误差仍存在。不要把预测头方差、模型误差和真实天气随机性混为一谈。P0不因此加入大型概率模型；先用direct-gain基线和校准诊断检验均值代入的代价。

## 7. 参数干预与输出反馈的可表达性

任意已编辑一步映射可以写作 F_a(s)=Fref(s)+C_a(s)，其中 C_a=F_a-Fref。迭代使用相同C_a即可复现同一轨迹。因此“只有参数编辑会改变未来动力学”“输出纠错只能改变末端”均不成立。

强基线至少包括：多lead联合后处理、逐步反馈残差模型、同控制器同动作预算的低秩输出/activation残差、virtual edit Fref+预测du。后者用于问实际执行参数编辑是否提供了代理无法替代的收益。没有实证优势就pivot到更简单订正，不在叙述上排除它。

## 8. oracle和部署分开

候选全集穷举的 best-of-registered-set 是该有限集合的事后参照。只用top3 nonlinear verification的teacher不是全集上限。oracle e0涉及未来真值；oracle du可由模型仿真获得、不需真值，但代价是多条rollout，不能算低成本部署。

所有论文中的“gain”需标记oracle/predicted/realized与summary/full-field。代理预测为正不能保证实际为正；no-op由真实标签挑出来也不能称serving-time安全。

## 8. gain误差的方向分解（代数诊断，不是新理论）

令预测偏差 εe=ehat−e、εd=dhat−d，则

    ghat−g = 2<εe,d>_Q + 2<e−d,εd>_Q
             + 2<εe,εd>_Q − ||εd||_Q².

所以只报告e0总体R²、du平均cosine不能保证选择正确；误差朝向和被选候选的尾部误差尤其重要。P0应对top候选单独检查误修正，但主分母仍是全部起报，不能只报被选子集。


---

# 为什么参数编辑不是天然必需？

对任意一个具体编辑后的单步映射Fa，都可以定义Ca(x)=Fa(x)−Fref(x)。那么Fa(x)=Fref(x)+Ca(x)。因此，表达能力不受限的、将修正反馈到下一步的输出/状态corrector可以完全复现同一轨迹。这是一个存在性恒等式，不代表低成本网络一定学得出Ca。

所以参数编辑可争取的是有限数据/算力下的归纳偏置、权重共享和表示效率，不是“只有它能改变动力学”。同一个输出corrector也可以跨步共享、联合预测所有变量，并通过后续Fref产生一致响应。

若I含参考forecast且平方误差目标固定，E[e0|I]+Fref=E[Y|I]。预测得足够好的e0直接产生强残差校正。又若e0在span(du)之外有不可预测噪声，选择只需要其相关投影，不必完整重建未来；P0四格因此需同时报告全场和响应方向误差。

必需挑战：
- 末端多lead多变量残差网络（不是弱的单变量one-step）；
- 迭代全变量feedbackcorrector（同history、hold、multisteploss、总成本）；
- virtual editedforecast=reference+preddu（检验是否还值得运行被编辑backbone）；
- 同状态编码器、同字典的directrouter。

参数方法赢的合格证据：相同总资源下在未见天气或未见动作有更好MSE/稳定性；或达到相同skill所需fitlabel显著更少；或一致的跨lead/变量收益而输出基线在充分调参后仍达不到。观察到风场图更平滑、某个变量更准、一次headroom更高，都不够。


---

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


---

# EarthDelta V2（研究合同，不是新版本已实现）

## 1 Core problem

同一冻结参考模型下，在运行多个昂贵候选预报前，能否预测有限编辑的完整未来响应，并选出值得执行的修正？

## 2 Core hypotheses

引用 FINAL_RESEARCH_DECISION 的H1/H2/H3。不重新扩展更多“创新”。

## 3 Formulation

输入I包含声明历史、可选一次reference preview及其成本；不包含未来truth、exact edited outputs、oracle b/H。动作a包含bank identity、系数、window、target layers、continuation。候选全集包含zero。

定义e0=D(Y)-D(Yref)，du(a)=D(Ya)-D(Yref)。主线S_phi(I,a)=(ehat,dhat_a)，预测g=2<ehat,dhat>-||dhat||²，同MetricSpec。首轮finite选择argmax(g-λC)，无改动候选允许停止。动作连续性/Jacobian只做诊断。

## 4 Training objective

同共同特征编码器，对e0和du分别使用Q加权回归；相同数据上增加可选gain回归/排序损失。raw geometric与加性gain校准是两个单独variant。训练/标签预算记录：一份e0标签不因K个candidate重复计数，du仿真轨迹总数也单列。du可以在没有未来标签的额外训练起报上采集，但必须独立时间范围、无test适配；其他方法应有same-label与same-total-data/compute两个比较轨。

当前ComposedPredictionHead已具备leads接口、reference/response解码；保留并补MetricSpec、zero-action结构约束与配置开关。不要默认启用JEPA或variance penalty后又称变化只来自e/du分解。

## 5 Inference procedure

(1) 校验checkpoint/bank/normalization/data identities；(2) 只读ServingContext；(3)一次编码context；(4)对全部声明candidate descriptor预测du和分数；(5)共同feasibility过滤，zero优先tie；(6)只运行选中动作的真实预报；(7)保存无标签决策日志。无oracle teacher/no per-candidate weather rollout。若context来自reference第0步后段，则先执行并计费该preview，再执行或合法重放选中预测，不能称零开销。

## 6 Main novelty candidate

可学习的逐状态、逐参数干预响应，及e/du监督非对称的可测价值。不是generic counterfactual world model的首次。

## 7 Why parameter intervention

没有一般必要性定理。用相同信息的多lead后处理、迭代feedback残差、virtual edit与同控制器输出动作去挑战。只在有限资源下证实更稳/更省/泛化更好才能保留。

## 8 Strongest baseline

同历史、同bank、同多步loss的direct router；同描述符/预算的direct-gain；同信息、同更新日程的feedback output correction。三个均必需，不相互替代。

## 9 Killer experiment

先做真实完整候选的oracle对静态余量。通过后同一候选表四格替换e/du，加directgain和反馈残差：一张共同样本、共同预算的表同时暴露字典、响应、误差、选择和执行的瓶颈。

## 10 Kill criterion

按STOP_CONDITIONS的预登记δ与置信区间决策。不能只统计“选中成功”的样本；保留所有起报、noedit、失败和最差lead。有限动作上限只是当前字典上限，不能据它否定所有参数编辑。


---

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


---

# 最近邻与检索范围

截至本轮 2026-09-21 UTC 核查。优先 2025–2026 原始论文，并加入不能省略的经典直接先例。下表是定向搜索，不是穷尽性首创证明。

## L01 VI-MoLE: Uncertainty Is Not Enough — 2026-08-03

来源：https://arxiv.org/abs/2608.02528

补充原始入口：https://arxiv.org/html/2608.02528v1

核查范围：已核查摘要与方法、实验章节；未核实作者代码。

已覆盖：在执行下一专家前预测剩余风险并分配适配预算。最直接威胁：预算内先预测价值再激活 LoRA。

本项目边界：保留差异应为多变量、多时效向量响应及其标签来源分离，不是预算或反事实命名。

## L02 Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects — 2026-06-06

来源：https://arxiv.org/abs/2606.08365

补充原始入口：https://arxiv.org/html/2606.08365v1

核查范围：已核查摘要与全文入口；未核实作者代码。

已覆盖：执行 steering 前预测稳定性与附带影响，包含未见特征筛选。

本项目边界：已经覆盖泛化的 predict-before-intervention；EarthDelta 要检验逐大气状态、逐编辑的天气轨迹响应。

## L03 Machine Learning Enables Real-Time Proactive Quality Control — 2024

来源：https://doi.org/10.1029/2023GL107938

补充原始入口：https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL107938

核查范围：期刊正文与数据声明已核查；Zenodo 8429402 未下载。

已覆盖：学习未来参考状态，避免依赖未来观测进行影响估计与观测剔除。

本项目边界：不能称首次无未来真值的影响驱动决策；不同之处是学习参数编辑的响应而不是观测同化。

## L04 Preemptive Ensemble Forecast Sensitivity to Observations — 2026-09-10

来源：https://arxiv.org/abs/2609.12296

补充原始入口：https://arxiv.org/html/2609.12296v1

核查范围：摘要/HTML 已核查；作者实现未核实。

已覆盖：用更强的切线近似，在不重新积分下估计观测影响与近似更新。

本项目边界：EarthDelta 是固定神经模型的有限参数编辑及其摊销预测；不可照搬近似有效范围或业务资格。

## L05 CCM: Discovering Physical Directions in Weight Space — 2026-05-14

来源：https://arxiv.org/abs/2605.14546

补充原始入口：https://arxiv.org/html/2605.14546v1

核查范围：摘要已核查；官方代码未核实。

已覆盖：PDE 专家端点形成权重方向，元数据或短轨迹前缀决定组合坐标。

本项目边界：权重方向、短历史、后续 rollout 都有先例；需要区别出候选条件化响应预测和决策收益。

## L06 CLAW: Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning — 2026-09-10

来源：https://arxiv.org/abs/2609.12278

补充原始入口：https://arxiv.org/html/2609.12278v1

核查范围：摘要和 HTML 已核查；官方代码未核实。

已覆盖：联合训练世界模型与 hypernetwork，从少量交互生成低秩适配。

本项目边界：必须比较同历史的直接 router；冻结第三方 backbone 是设定差异，不自动是方法创新。

## L07 W2T: LoRA Weights Already Know What They Can Do — 2026-03-16

来源：https://arxiv.org/abs/2603.15990

补充原始入口：https://github.com/xiaolonghan2000/Weight2Token

核查范围：摘要与作者代码入口已核查；代码许可证未核实。

已覆盖：从 LoRA 权重表示预测能力和表现，使用规范化处理因子不唯一。

本项目边界：当前 EarthDelta 描述符只有系数与窗口，不支持无条件宣称对全新专家泛化。不要立刻复制完整权重编码器。

## L08 Learning Options for Compositional Motor Control with Adapter Banks — 2026-09-15

来源：https://arxiv.org/abs/2609.17042

补充原始入口：https://arxiv.org/html/2609.17042v1

核查范围：摘要与 HTML 已核查；官方代码未核实。

已覆盖：冻结共享循环核心后由高层策略组合残差适配器，呈现低秩结构。

本项目边界：参数改变作为动作已有；EarthDelta 不是首次参数空间控制，而是预测候选的天气输出效果。

## L09 Earth system model parameter adjustment using a Green’s functions approach — 2022

来源：https://gmd.copernicus.org/articles/15/2309/2022/

补充原始入口：https://doi.org/10.5281/zenodo.5507631

核查范围：期刊正文与代码/数据入口已核查；归档未运行。

已覆盖：前向参数扰动得到响应核，再用加权最小二乘进行参数校准。

本项目边界：R→RᵀQR→求解不是新算法；贡献需在逐起报的可学习代理和可部署决策上。

## L10 Solver-in-the-Loop — 2020

来源：https://arxiv.org/abs/2007.00016

补充原始入口：https://github.com/tum-pbs/Solver-in-the-Loop

核查范围：论文及官方 README 在会话中核查；本轮未执行。

已覆盖：输出/状态纠错进入求解器滚动和多步训练，能影响后续轨迹。

本项目边界：强残差基线必须反馈到 rollout。参数编辑不独占长期动力学修正。

## L11 WeatherPEFT — 2025-09-26 / ICLR 2026

来源：https://arxiv.org/abs/2509.22020

补充原始入口：https://github.com/ShileiCao/WeatherPEFT

核查范围：论文与源码/仓库清单核查；根许可证未确认。

已覆盖：任务提示与参数选择；已有天气 PEFT 对照。

本项目边界：与完全冻结骨干、同任务内逐状态编辑不同；移植版必须标 adapted。

## L12 The Intervention Gap in Latent World Models — 2026-08-30

来源：https://arxiv.org/abs/2608.29998

补充原始入口：https://arxiv.org/html/2608.29998v1

核查范围：摘要已核查；实现未核实。

已覆盖：干预保真度需单独评估，不能由一般拟合或任务回报推断。

本项目边界：将 no-effect、平均响应、错误方向和决策 regret 纳入必测。

## L13 GeoQ — 2026-08-21

来源：https://arxiv.org/abs/2608.21652

补充原始入口：https://arxiv.org/html/2608.21652v1

核查范围：摘要核查；实现未核实。

已覆盖：非侵入式、依赖局部几何的条件误差分位数估计。

本项目边界：幅度/风险估计不是带符号残差方向；P0 要比较简单误差基线。

## L14 JacQuant — 2026-05-25

来源：https://arxiv.org/abs/2605.25469

补充原始入口：https://arxiv.org/html/2605.25469v1

核查范围：摘要核查；实现未核实。

已覆盖：学习参数变化的局部敏感度代理用于量化训练。

本项目边界：不能因改名 Learned Intervention Jacobian 就声称首创；有限动作响应更贴当前代码。

## L15 AdaWeather — 2026-06-01

来源：https://arxiv.org/abs/2606.02663

补充原始入口：https://arxiv.org/html/2606.02663v1

核查范围：摘要核查；实现未核实。

已覆盖：概率天气预报的自适应组合与相对静态混合的 regret。

本项目边界：组合预测与组合参数不相同；在线收到核验的协议也不同。

## Nearest-neighbor coverage matrix

Y=原文明确涉及，P=部分/不同对象，—=在所核查内容中未建立；不是对全部历史版本的“不存在”证明。

| 工作 | 状态条件适配 | 编辑响应/效果预测 | 干预选择 | 预算 | 多编辑作用 | 解析效用 |
|---|---|---|---|---|---|---|
| VI-MoLE | Y | Y（标量风险） | Y | Y | P（前缀） | P |
| SAE pre-intervention | P | Y（副作用/稳定性） | Y（筛选） | — | P | — |
| Honda 2024 | P | P（观测影响） | Y | — | P | Y |
| PEFSO | P | P（切线观测影响） | Y | P | P | Y |
| CCM | Y | P（权重方向） | Y | — | P | P |
| CLAW | Y | P（世界模型适配，不是编辑后果头） | P | — | — | — |
| W2T | —（权重为输入） | P（能力/表现） | P（检索） | — | — | — |
| Adapter Banks | Y | —（非显式候选效果头） | Y | — | Y（序列组合） | — |
| Green's functions | P（校准，不是逐起报控制器） | Y（计算响应） | Y | — | Y（局部联合） | Y |
| Solver-in-the-Loop | Y（纠错状态） | P | P | — | P（滚动） | — |

## 最危险的 X + Y

1. VI-MoLE 的候选价值路由 + FSO/PEFSO 的二次误差影响量。
2. Honda/Yamazaki 的无未来参考估计 + Green's-function 参数响应。
3. 候选响应 surrogate + Adapter Banks/CCM 的参数动作。

不能用“领域不同”直接回避。争取的差异必须通过 e/du 标签非对称、同信息direct-gain、未见动作组合与完整天气rollout形成证据。

## 新近论文不能无条件变成代码依赖

“有arXiv”≠同行评审通过，“文中说有release plan”≠代码已发布，“仓库可读”≠许可明确。VI-MoLE的风险证书也不自动是边际收益的下界：两个上界之差一般不是两真值之差的下界。本项目不继承未经单独验证的安全定理。该观察不改变其对候选价值路由这一动机的先例地位。

检索主题包括 weather adaptation、conditional/hypernetwork LoRA、pre-intervention steering、parameter-response surrogate、influence/Jacobian、amortized optimization、predict-then-optimize、FSO/PQC、反馈纠错。没有发现完整相同实现不构成“首次”的逻辑证明。
