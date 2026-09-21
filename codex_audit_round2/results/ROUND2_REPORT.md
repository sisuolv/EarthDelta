# EarthDelta 第二轮独立审计

审计日期：2026-09-21 UTC。实际范围：`4fe55a7..fb767f7`；HEAD：`fb767f7f6efbc428be39c9ad84f5905331d6e40f`。已按顺序完整阅读 AUDIT_BRIEF、AUDIT_PROMPT，先完成第零部分判断，再完成代码审计。研究结论与代码准入分别记录。

## 零、novelty 与价值判断

**独立建议：`novelty_value_verdict = PROCEED_TO_P0_04`，只支持有资源上限的最小生存规格与合格非零 bank 投入。`agrees_no_prior_occupancy = UNCERTAIN`。** 现有证据支持把它作为可证伪的研究假设继续检验，不能支持“novelty 已经成立、剩下只有 headroom 风险”这一更强前提。本建议不取决于后面的代码审计是否通过，也不代替真实 S0 准入。

### 0.1 检索能支持什么

我重新阅读了第一轮包的 NOVELTY_AUDIT、RELATED_WORK、EIGHT_DIRECT_ANSWERS、FINAL_RESEARCH_DECISION、OUTPUT_CORRECTION_CHALLENGE、literature_sources，以及 BASELINES、P0_SURVIVAL_EXPERIMENTS 和 v6 核心假设，并实际取回下面的原始文献。已读文献未展示完整的“签发时刻参考误差/候选响应双头 + 平方误差收益组合 + 带交互的预算参数编辑”管线；但“特定交集尚未检出”并不足以证明全领域无人先做，更不足以证明这个交集有科学价值。

此会话未提供可可靠声明的精确训练知识截止日。我不以参数记忆核实 2025–2026 年的新论文，而以本次成功取回的文本为依据。WeatherPEFT v2 正文请求超时，只有 v1 全文与 v2 摘要/版本信息；Honda 2024 期刊正文返回 403；扩展搜索受代理/TLS 失败限制。本轮是定向原文重读，不能冒称已完成截至当日的穷尽检索，也没有独立复现此前约 200 次搜索。访问结果与文本摘要哈希见 [literature_retrieval.json](evidence/literature_retrieval.json)。

| 最近邻 | 已核查内容 | 对“占据”的判断及必须修正的描述 |
| --- | --- | --- |
| [Aurora LoRA](https://arxiv.org/html/2405.13063v1) | Appendix D.4、G.6/G.7 | 已覆盖通过 LoRA 改变多步动力学、replay-buffer rollout fine-tuning 与指标权衡；未见逐起报 e0/du 双头预算选择。所读版本 `still_non_occupying=YES`，但不能把它简化为普通 one-step 微调。 |
| [WeatherPEFT v1](https://arxiv.org/html/2509.22020v1)、[v2 摘要](https://arxiv.org/abs/2509.22020) | v1 §4；v2 摘要与版本日期 | brief 的“PEFT benchmark”不准确：论文提出 TADP 与 SFAS；task prompt 来自任务编码器 embedding 权重，Fisher 自适应选择发生于训练。dynamic task prompt 不等于每起报的收益规划。已读方法 `YES`；v2 完整方法仍待核验。 |
| [Adapter Banks](https://arxiv.org/html/2609.17042v1) | §3.1.2、§3.2、§4.4 | brief 的“RL-return-based selection”错误：先由示范分段与模仿学习建立 bank，再冻结网络，针对目标手部轨迹，以 L1 损失优化 T×10 soft policy 350 步。它比 brief 描述更接近“冻结动力学、adapter 组合、轨迹优化”。未见合法天气历史下的 e0/du 收益预测与 FSO 预算二次型，故所读版本仍为 `YES`。 |
| [VI-MoLE](https://arxiv.org/html/2608.02528v1) | §3–5、实验协议 | 执行前预测候选 prefix 风险、按边际收益/成本共享适配预算，已经占据“先估值再执行 LoRA”的宽泛动机；beta=0 还有 committee-relative 无标签目标。未见 e0/du 向量分解与任意编辑组合的 Gram 二次目标，故精确管线为 `YES`。但标量条件风险同样可以编码 prefix 交互，不能把“标量”当成表达能力必然较弱的证明。 |

另外，[Strobach 2022](https://gmd.copernicus.org/articles/15/2309/2022/) 已提供参数响应与加权最小二乘的直接先例。二次恒等式、参数响应拟合、执行前估值与预算选择各自都不是新发明。较有希望的贡献应落在：**把不需要未来真值的响应仿真监督，与需要核验标签的参考误差监督分开，是否在同信息、同总资源条件下提高标签效率或未见动作泛化。**

这里四个 `YES` 的量词仅覆盖所读版本中的明确方法；不与全局 `UNCERTAIN` 矛盾。尚未核完的 WeatherPEFT v2 也不能由 v1 的 `YES` 自动覆盖。

### 0.2 机制成立，但学习后的优势不由恒等式保证

固定输出空间和同一半正定 Q，令 e 为真值减参考预测，u 为编辑预测减参考预测，则

```text
g = ||e||_Q^2 - ||e-u||_Q^2 = 2 e^T Q u - u^T Q u.
```

这对实际端点差是精确恒等式；把学习到的 e_hat、u_hat 代入后，结果是收益 surrogate，不再自动精确或无偏。即使两个头都是各自 MSE 意义下的最优条件均值，压缩 context I 仍可能丢失与决策有关的二阶量：

```text
E[g | I] = 2 mu_e^T Q mu_u - mu_u^T Q mu_u
           + 2 tr(Q Cov(u,e | I)) - tr(Q Var(u | I)).
```

因此只组合两个均值可能错误排序。若 I 含有确定 u 所需的全部状态与候选信息，响应不确定性项消失，此时预测 E[e|I] 足以计算条件期望收益；但直接 gain predictor 在同样信息和监督下也可以学习同一个量。联合训练可能减轻误差，却不会单凭“联合”一词消除这个差别。

双头的先验理由是可复用仿真数据、跨候选共享误差信号、PSD 几何、换 Q/预算时复用表示；这些是可检验的归纳偏置，不是增加信息量的定理。direct-gain、直接 edited-forecast、静态/regime 和输出纠错挑战者都应获得同等合法输入、仿真资格、标签与调参资源，不能只让双头使用廉价 du 数据再宣称机制胜出。

交互项也要准确界定：仅在 `u(a)=R a` 的域内，`b=R^T Q e`、`H=R^T Q R` 给出精确系数二次型。H 非对角首先表示响应方向的加权重叠，不等于已捕获非线性编辑相互作用。对每个实际非线性候选计算端点收益恒等式仍然精确，却不能据此证明 singleton 响应可线性叠加。固定线性 D 对这种响应组合有意义；对“变换后的各端点作差”本身，恒等式并不要求 D 线性。

参数编辑相对 feedback output correction 也没有无限表达能力上的独占优势：足够自由的状态反馈修正可以表示 `F_edit(x)-F_ref(x)` 并逐步传播。研究应争取有限样本、稳定性或实测成本上的优势，不能以“输出修正不改动力学”排除挑战者。

### 0.3 P0-05 能否决定这个 idea 有价值

**P0-05 对冻结 registry 的上限检验有信息量，设计主体可以保留；它不是这个 idea 的充分价值检验。** 全量实际非线性候选、合格非零 bank、dev 固定静态策略、confirm 一次、起报/天气过程配对 bootstrap、保留失败分母，都是正确的防偏设计。问题在结论的范围，而不是全量 oracle 的计算方式本身。

oracle 回答 `E[max_a g]`，合法签发时策略的上限是 `E[max_a E[g|I]]`；两者可以相差很大。例如 I 与 e 无关，e 以相等概率取 ±1，候选 u 为 0、+0.5、−0.5。oracle 每次顺着真误差选方向，gain=0.75；任何合法固定非零动作的期望 gain=−0.25，最佳合法策略是 no-edit，gain=0。oracle-static gap 的 CI 可以非常窄，却完全没有可部署动态价值。这不是 bootstrap 假阳性，而是测量对象不同。

所以 PASS 只说明“该候选库存在值得尝试预测的 hindsight 空间”，不证明 e0/du 可学习、不证明分解优于 direct-gain，也不证明参数编辑优于反馈纠错。FAIL 在充分功效下可以否证该 bank/预算/Q/lead 下的选择空间；不能否证所有参数编辑，尤其 singleton-only registry 的 FAIL 不能否证尚未测试的组合 H。

现有 P0-06、P0-07、P0-08 已经分别处理可预测性、输出挑战和总决策，因此无需重新安排 DAG，也无须因这一 estimand 差别整体推翻 P0-05。应在预登记中写清 PASS/FAIL 的不对称含义。用同一小 pilot 候选缓存增加 context/truth 打乱的 hindsight 参照、交叉拟合的小 direct-gain/regime 对照，可帮助辨别选择空间与可用信号；打乱只作诊断，不能视为技能或机械地从真实收益扣除。

此外，最小有用效应 delta_min 与在给定 alpha、power、N 下可检测的 MDE 是两件事。前者表达值得投入的收益，后者表达当前实验能否识别它。样本变多导致 MDE 变小，不能自动把更小效应变成有研究价值；CI 跨价值阈值时应为 `INCONCLUSIVE`。

### 0.4 明确投入建议与下一份决定性证据

继续的是**有 cap 的最小假设检验**，不是在“novelty 与价值已确定”的前提下扩大训练。先修正 Adapter Banks/WeatherPEFT 描述及贡献措辞；补齐 WeatherPEFT v2 方法与“parameter-response surrogate + reference error + predict-then-optimize”的定向检索。文献不确定性尚不构成停止小 pilot 的充分理由，也不要求继续无边界搜索。

如果不存在合格现成 bank，要得到真实 headroom/可预测性信息就必须支付最小 bank 的成本，无法由更多代数推理替代。该小 bank 的 dev 表先检查 oracle-static gap、响应方向上的误差可预测性和同信息的小 direct-gain；后续持续投资必须拿到 pred/pred 相对强静态的收益，并在标签效率、动作泛化或成本至少一个预登记维度证明分解的价值。oracle PASS 单独不够。独立机读结论见 [novelty_value_assessment.json](novelty_value_assessment.json)。

## 一、代码审计结论与六项 disposition

**`round2_verdict = FAIL_REOPEN_P0_03`，同时须补 P0-02 残留。** 共 15 条 finding：P0 两条、P1 十条、P2 三条。全部问题机制均为亲读源码确认的 `STATIC_CONFIRMED`；运行复现另列证据，不把尚未观察到的真实天气损害说成已发生。

当前不能解锁 P0-04 的模型工作，原因包括已复现的实现缺陷。另一个独立状态是：真实 GPU/xformers S0 尚无产物，仍为 `BLOCKED`；这是已知资源限制，不作为新 finding，也没有因本机未跑 GPU 宣判科学 `STOP` 或 `PIVOT`。

| 原 finding | disposition | 确认有效的改动 | 仍未关闭的范围 |
| --- | --- | --- | --- |
| A01 | PARTIALLY_CLOSED | 真实官方网络/xformers 导入；无资源明确 BLOCKED；七项 False 初始化；缺参照阻断；内部一致性重新标注 | 完整官方 rollout/归一化不独立；上游身份与多步未绑定；晚期异常保留 PASS。B01/B02/B10/B13。 |
| A03 | PARTIALLY_CLOSED | 四项点名校验全部实际生效；bound/support 超限跳过且计数，NaN/Inf/K 超限报错；0.5 测试适配合理 | 原 A03 的类型合同未完：复数会丢弃虚部再参与选择；空 registry 与显式 no-edit 未区分。仅 P2 残留 B15，不能解读成四项主要修复无效。 |
| A04 | PARTIALLY_CLOSED | 五个实现实际调用公共 helper；head 确实使用权重参数 | 全目标 MetricSpec 与默认关闭校准未落地；新增 batched Gram/权重问题。B04/B05/B06。 |
| A05 | PARTIALLY_CLOSED | SHA 完整值保存，前 16 位接入 backbone 版本；norm digest 确含变量序、interval 键、shape | 消费端只展示身份、未匹配期望内容；入口核验、实际 grid 身份仍缺。B02。 |
| A06 | PARTIALLY_CLOSED | UTC 构造路径完整；本轮真实切换 UTC/Honolulu/Shanghai 后时间戳一致；每条 builder 行都有来源字段 | 六小时假定来源、legacy 默认升级、actual time index 与 process group 不合格。B07/B09。 |
| A11 | PARTIALLY_CLOSED | 指针说明可接受，旧 YAML 可保留；beats/zero-shot 与 2–3% gate 已降级 | preview/replay 成本未处置；MDE 不可代替价值阈值。B11。 |

### 1.1 数学合同的独立结论

`paired.py:70`、`heads.py:351`、`heads.py:445`、`geometry.py:84`、`selection.py:99` 均进入公共实现，并非建了无人调用的模块。但公式代码复用与指标合同统一是两件事。

sum 和 mean 都可以是正确的平方损失定义。对同一完整目标，以冻结 schema 的 scale、D、面积/lead/变量权重定义 Q_eff，再构造 `b=R^T Q_eff e` 与 `H=R^T Q_eff R`，系数空间本来就无须再次除维数。如果 sum 与 mean 只差对所有候选相同的正数，无其它惩罚时 argmax 不变；存在固定 ridge 或 cost penalty 时，则必须一起变换单位。仅以“scipy 用 sum、训练用 mean”解释，不足以保证实际选择一致。

当前按 F 先归一化会消去每个 H/S 切片上的共同 lead/面积因子。CPU 两 lead 例子里，权重 100:1 的真实全目标收益分别为 100/101 和 1/101，按 F 化简再平均却得到 0.5、0.5。逐 lead/region 的诊断向量可以保留，但不能冒充声明的全目标得分。head 校准层也仍可训练并直接加到唯一 gain 输出，应与 analytic gain 分开，默认关闭。

“uniform weights 下 bit-identical”这一 commit 文字过强：`sum(x/F)` 与 `mean(x)` 在 FP32 中可有舍入差，本轮样例最大 5.96e-8。它们数学等价，此舍入差单独不是阻塞 finding。权重负值、零切片、batched Gram 的问题则是实际行为缺陷。

### 1.2 官方路径与并发边界

exporter 导入 `reference/stormer` 的 OfficialStormer，官方文件直接依赖 xformers，没有重新伪造命名空间或自动切换 SDPA。缺 xformers 在导入模型前退出码 2；已有 xformers 但缺 CUDA 则码 3；均明确 BLOCKED。七项 False 初始化和缺 manifest 返回 False 都有效，`all(...)` 不会掩盖缺参照。

但 exporter 只独立了网络部分，手写 rollout 仍复用 EarthDelta NormalizationContract；官方 iterative module 没有被调用。B01 的共同归一化偏差足以说明“真实官方网络”不等于“独立官方完整系统”。此外 B13 是总体异常闭合的问题，不应抹掉其它已确认有效的 fail-closed 分支。

当前生产调用图中没有发现并发 controlled_rollout 或 activation-checkpoint replay；串行训练/脚本是观察到的路径。thread-local guard 只解决同线程同 bridge 重入，无法提供 model 级并发隔离。本轮双线程只验证 guard 可被同时取得，未运行共享模型竞态，不宣称已发生输出污染。串行 P0 声明暂不支持并发/replay 即可，不要求现在建设复杂并发框架。

## 二、全部代码 findings（按严重度排序）

### P0 · B01 — 官方网络已经独立，完整 rollout 与归一化仍共享待审实现

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A01：`PARTIALLY_CLOSED`。**

定位：[scripts/export_upstream_reference.py:106](../../scripts/export_upstream_reference.py#L106)；[scripts/export_upstream_reference.py:110](../../scripts/export_upstream_reference.py#L110)；[scripts/export_upstream_reference.py:230](../../scripts/export_upstream_reference.py#L230)；[scripts/export_upstream_reference.py:248](../../scripts/export_upstream_reference.py#L248)；[earthdelta/bridge/stormer_bridge.py:251](../../earthdelta/bridge/stormer_bridge.py#L251)；[reference/stormer/inference.py:118](../context/upstream_stormer/inference.py#L118)；[reference/stormer/stormer/models/iterative_module.py:159](../context/upstream_stormer/stormer/models/iterative_module.py#L159)。

**结论与依据：** 网络架构确实来自官方 xformers Stormer，但 exporter 没有调用官方 GlobalForecastIterativeModule.forward_validation，而是重写 rollout，并直接复用待审 bridge 的 NormalizationContract。官方 inference 的差分均值固定为零，本地实现却加上非零 diff_mean；pinned 6h/24h 数组均有 69 个非零元素，max(abs(diff_mean/inp_std)) 分别约 1.66e-5/7.17e-5。故共享错误能通过当前所谓 upstream parity，完整官方路径仍未独立验证。该问题还保留第一轮 A02 的已知偏差，不能因 A02 未列入本轮六项就忽略其对 A01 的影响。上述数值是小型归一化资产检查，不是实际 GPU parity 结果。

**独立验证：** 独立官方 normalizer 明确零增量均值；真实小型 npz 的非零偏移见 contract_repros.json。不是由未运行 GPU 推导的 finding。

**最小动作：** 独立进程执行 pinned 官方 iterative module 和官方 transforms，直接从官方 npz/变量序构造零 diff_mean 推理 policy；bridge 增加显式 policy，legacy 仅复算旧结果。加入非零 diff_mean 的 CPU 反例，再由有资源的后续任务运行真实 GPU parity。

### P0 · B02 — 参照身份没有核验，错误标识与缺多步产物仍可通过 parity

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A05：`PARTIALLY_CLOSED`。**

定位：[scripts/s0_gate.py:186](../../scripts/s0_gate.py#L186)；[scripts/s0_gate.py:265](../../scripts/s0_gate.py#L265)；[scripts/s0_gate.py:273](../../scripts/s0_gate.py#L273)；[scripts/s0_gate.py:293](../../scripts/s0_gate.py#L293)；[scripts/export_upstream_reference.py:379](../../scripts/export_upstream_reference.py#L379)；[earthdelta/bridge/stormer_bridge.py:818](../../earthdelta/bridge/stormer_bridge.py#L818)；[earthdelta/contracts.py:28](../../earthdelta/contracts.py#L28)。

**结论与依据：** SHA 已进入 ArtifactVersion，norm digest 也已绑定变量名、interval 键和 shape，但 S0 只检查 SHA 字符串长 64；上游 manifest 的 checkpoint SHA 只展示、不与 bridge 比较，normalization_digest 不核验，x_raw 参数完全不用，也不核验原始输入/输出内容 hash、源码 pin 或精确 shape。导出的 6h_4step 文件甚至不加载。CPU 合成反例在错误 checkpoint/norm、不同 x_raw、缺 4-step 文件时仍 passed=True；任意存在的文件都能通过独立 hash gate。执行入口也未调用 check_version_match，grid 仍只是尺寸字符串，故 A05 没有完全关闭。

**独立验证：** cpu_repros.json 的 B02_unbound_reference 与 B02_hash_gate 均为 true。合成 identity bridge 用于隔离 manifest 校验逻辑，不代表真实 Stormer 输出。

**最小动作：** 在数值比较之前核对冻结预期 checkpoint 的完整 SHA、上游源码/配置 pin、归一化 policy/digest、变量序、真实坐标、输入 hash 与 exact shape；比较全部注册 rollout 输出，缺一项阻断。把版本核验接入实际执行/缓存入口，不能只提供可选 helper。

### P1 · B03 — 可微分支切断调用方系数与 controller 的梯度

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`。**

定位：[earthdelta/bridge/stormer_bridge.py:663](../../earthdelta/bridge/stormer_bridge.py#L663)；[earthdelta/bridge/stormer_bridge.py:742](../../earthdelta/bridge/stormer_bridge.py#L742)；[earthdelta/bridge/stormer_bridge.py:747](../../earthdelta/bridge/stormer_bridge.py#L747)；[earthdelta/contracts.py:51](../../earthdelta/contracts.py#L51)；[tests/test_differentiable_rollout.py:173](../../tests/test_differentiable_rollout.py#L173)；[tests/test_differentiable_rollout.py:201](../../tests/test_differentiable_rollout.py#L201)。

**结论与依据：** differentiable=True 用 torch.tensor(coeffs_at_step, requires_grad=True) 新建每步叶张量，既不接受调用方 [B,K] 系数，也不保留到 controller 的计算图；所有 batch 仍共用一份 tuple。即使 tuple 元素来自有梯度系数，输出 backward 后调用方 coeff.grad 仍为 None。轨迹返回功能成立，bank/input 梯度也可以存在，但这些不等于系数/head 梯度；新增测试只断言 x_norm.grad，未验证其标题所称性质。

**独立验证：** cpu_repros.json 中 caller_coefficient_grad_is_none=true、bank_has_nonzero_gradient=true、frozen_backbone_has_gradient=false。

**最小动作：** 提供显式 tensor_coefficients[B,K] 参数，通过保留 autograd 的 to()/窗口 mask 注入；保持旧 tuple 推理接口。以冻结骨干、非零合成 bank 验证末态损失到 caller coefficients 和 controller 参数的非零梯度，并核验 hold 之外梯度为零。

### P1 · B04 — 迁移公共 helper 后，共享 program 配批量 Gram 报错

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A04：`PARTIALLY_CLOSED`。**

定位：[earthdelta/metrics_contract.py:192](../../earthdelta/metrics_contract.py#L192)；[earthdelta/metrics_contract.py:194](../../earthdelta/metrics_contract.py#L194)；[earthdelta/heads.py:445](../../earthdelta/heads.py#L445)。

**结论与依据：** 公开支持的 benefit[B,d]、gram[B,d,d]、program[d] 组合走到 einsum('i,ij,j->...',...)，其中 ij 无法接收三维 Gram，运行时报错。第一轮 heads.py 的 ellipsis 公式支持该组合，因此这是迁移到 canonical helper 后的真实回归。benefit[d] 配 batched program/gram 的原广播能力也被当前分支限制。

**独立验证：** 同输入旧公式得到 [2,2]，新公式抛出 Gram 维数错误；见 B04_batched_gram。

**最小动作：** 统一使用正确的 ellipsis contraction，并校验最终 d 及可广播 batch 维；覆盖共享/逐样本 program 与共享/逐样本 Gram 的组合。

### P1 · B05 — 权重校验回退，有限输入可产生 NaN 或负 MSE

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`。**

定位：[earthdelta/metrics_contract.py:96](../../earthdelta/metrics_contract.py#L96)；[earthdelta/metrics_contract.py:145](../../earthdelta/metrics_contract.py#L145)；[earthdelta/metrics_contract.py:152](../../earthdelta/metrics_contract.py#L152)；[earthdelta/metrics_contract.py:250](../../earthdelta/metrics_contract.py#L250)；[earthdelta/metrics_contract.py:263](../../earthdelta/metrics_contract.py#L263)。

**结论与依据：** 原 paired._weights 检查每个 feature 切片的正权重和；新 helper 只检查整个 w.sum()>0，随后逐 F 除法。广播权重含一个全零 lead/空间切片时，有限输入得到 NaN，原先会拒绝。新 weighted_mse 又完全不复用权重/scale 校验：例如 prediction=[1,2]、target=0、weights=[2,-1] 返回 MSE=-2；零 scale、零权重和同样未拒绝。当前 MetricSpec 的 missing_policy 不会处理这些情况。

**独立验证：** B05_weights 记录 nonfinite gain 及 weighted_mse=-2。

**最小动作：** 广播并转换计算 dtype 后，按实际约简维检查非负、有限、正分母；missing_policy 明确决定拒绝还是合法 mask。MSE/RMSE 使用同一校验，并要求 scale 有限且正、convention 有效。

### P1 · B06 — 公式复用尚未形成同一指标执行合同，校准仍默认可训练

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A04：`PARTIALLY_CLOSED`。**

定位：[earthdelta/metrics_contract.py:41](../../earthdelta/metrics_contract.py#L41)；[earthdelta/metrics_contract.py:150](../../earthdelta/metrics_contract.py#L150)；[earthdelta/geometry.py:65](../../earthdelta/geometry.py#L65)；[earthdelta/heads.py:294](../../earthdelta/heads.py#L294)；[earthdelta/heads.py:356](../../earthdelta/heads.py#L356)；[earthdelta/heads.py:413](../../earthdelta/heads.py#L413)；[earthdelta/selection.py:180](../../earthdelta/selection.py#L180)。

**结论与依据：** 五处确实调用公共公式，但并未形成共同 MetricSpec 执行合同。MetricSpec 没有生产调用者；head 仍将可训练 calibration 加到唯一 gain 输出，默认未禁用且未暴露 gain_analytic/gain_calibrated。公共物理空间 reducer 只按 F 归一化，若将含 lead/area 权重的 Q 直接传入，逐切片标量权重会被消去；不能冒称已按 H,S,F 一次归一化。系数空间用 sum、训练用 mean 可同时成立，但须让 b/H 由同一个 Q_eff 构造；仅说 '权重已嵌入' 不确定其归一化。固定 ridge 或成本惩罚不随 Q 的缩放转换时，sum/mean 还会改变实际最优动作。

**独立验证：** B06_metric_scope 记录 F-only 结果 [0.5,0.5] 与全目标 [0.990099,0.00990099]。没有把所有使用 sum 的代码一律判错。

**最小动作：** 保留分 lead/region 的诊断 reducer，同时定义有 schema/D/scale/Q/missing policy 的全目标 reducer；在建 b/H 前统一 Q_eff 并约定 ridge/penalty 单位。分别返回 analytic/calibrated gain，默认关闭校准。用同一实际 e、R、a、非均匀 H/S/F 权重检查物理损失差、head gain 和 coefficient gain 的一致性。

### P1 · B07 — 六小时情景和缺来源的旧记录被赋予不正确 provenance

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A06：`PARTIALLY_CLOSED`。**

定位：[earthdelta/data/make_splits.py:86](../../earthdelta/data/make_splits.py#L86)；[earthdelta/data/make_splits.py:224](../../earthdelta/data/make_splits.py#L224)；[earthdelta/data/make_splits.py:306](../../earthdelta/data/make_splits.py#L306)；[earthdelta/data/make_splits.py:308](../../earthdelta/data/make_splits.py#L308)；[earthdelta/data/make_splits.py:378](../../earthdelta/data/make_splits.py#L378)；[tests/test_time_provenance.py:109](../../tests/test_time_provenance.py#L109)。

**结论与依据：** UTC 构造已修复，所有 builder 记录也确实赋值 availability_source，但语义仍错误：默认 delay=6h 的假定被标为 reanalysis_retrospective；未提供真实 first-seen 记录也可选择 observed_first_seen；旧 manifest 缺字段时会被自动补成 reanalysis_retrospective。枚举校验只能证明字符串合法，不能证明历史可得性。这会让 memory/近期核验特征沿用原来的过早可用时刻。

**独立验证：** B07_availability 记录默认与 legacy 补值。UTC 修复本身经三种真实进程 TZ 验证有效。

**最小动作：** 把 6h 固定延迟明确标 scenario，并限制到情景/retrospective open-loop；observed_first_seen 必须来自真实时间记录，ERA5 与 ERA5T/最终版区分 provenance。旧记录缺 provenance 保留 unknown/legacy 或阻断 formal，不能自动升级。P0 主线继续禁用 memory。

### P1 · B08 — 数据完整性只看形状，尚未写入的数据可被当作已完成

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`。**

定位：[earthdelta/data/pull_wb2.py:441](../../earthdelta/data/pull_wb2.py#L441)；[earthdelta/data/pull_wb2.py:472](../../earthdelta/data/pull_wb2.py#L472)；[earthdelta/data/pull_wb2.py:498](../../earthdelta/data/pull_wb2.py#L498)；[earthdelta/data/pull_wb2.py:705](../../earthdelta/data/pull_wb2.py#L705)；[earthdelta/data/pull_wb2.py:783](../../earthdelta/data/pull_wb2.py#L783)；[earthdelta/data/pull_wb2.py:1078](../../earthdelta/data/pull_wb2.py#L1078)；[earthdelta/data/pull_wb2.py:1180](../../earthdelta/data/pull_wb2.py#L1180)。

**结论与依据：** 完整性检查只看 time/channel 数量。create_intermediate_zarr 和 rechunk_to_final 在写任何天气数据之前就预建完整形状的 metadata；中断后的全 NaN/未写完 store 仍会被判 complete，pilot/legacy skip 会跳过它。markers 也只按 year/source_var 文件存在判断，没有绑定具体 store 或 chunk 内容；旧 marker 与新建空 store 可形成同类错误。源代码问题已确认，但本轮未扫描正在拉取的 2015-2018 资产，不能据此断言现有数据已污染。

**独立验证：** B08_unwritten_store 同时记录 declared_complete=true、first_field_all_nan=true；未读取或修改后台真实训练年份 store。

**最小动作：** 输出到 staging store，逐 source/chunk 核验写入和 finite/missing policy 后再原子发布 completion manifest；绑定真实 time/grid/channel/unit/hash 和 marker 身份。完成性不能由数组形状或两变量 nanmean 代替。训练准入读取该证书并拒绝残缺 store。

### P1 · B09 — 日历 manifest 未与真实 history/target 索引联结

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A06：`PARTIALLY_CLOSED`。**

定位：[earthdelta/data/make_splits.py:143](../../earthdelta/data/make_splits.py#L143)；[earthdelta/data/make_splits.py:153](../../earthdelta/data/make_splits.py#L153)；[earthdelta/data/make_splits.py:338](../../earthdelta/data/make_splits.py#L338)；[earthdelta/data/make_splits.py:352](../../earthdelta/data/make_splits.py#L352)；[earthdelta/data/make_splits.py:381](../../earthdelta/data/make_splits.py#L381)；[earthdelta/contracts.py:28](../../earthdelta/contracts.py#L28)。

**结论与依据：** split builder 仍按日历生成全年记录，既不读取实际 time index，也不验证 history/target 端点。build_manifest([2015]) 保留 2015-01-01 00UTC（24h history 会需要 2014），且年底目标伸入未传入的 2016；全量 [2015..2018] 也不能发现缺时次或重复时次。normalization_hash 缺失会成为 placeholder，ArtifactVersion 只要求非空字符串；event_id 含 lead，同一 issue 的各 lead 并非 process group。对 P0-04 不能把 validate_all 的时序不等式当训练样本完整性或独立统计分组证明。

**独立验证：** B09_calendar_index 给出 2015 首项、2016 末目标、placeholder norm 及分 lead 行 ID。日历有序不等于样本可取。

**最小动作：** 训练前以实际坐标索引建立 history/current/target 联结，明确首尾可用窗口和缺时次策略；formal 模式拒绝 placeholder，绑定实际 norm/grid；保存 issue_id 与独立 process_group_id。不要提前规定尚未实现的 P0-04 trainer，只需将这些作为其准入条件。

### P1 · B10 — gate 固定 ps2，无法验证冻结主线 ps4

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`。**

定位：[scripts/s0_gate.py:609](../../scripts/s0_gate.py#L609)；[scripts/s0_gate.py:620](../../scripts/s0_gate.py#L620)；[scripts/s0_gate.py:797](../../scripts/s0_gate.py#L797)；[scripts/export_upstream_reference.py:410](../../scripts/export_upstream_reference.py#L410)；[scripts/export_upstream_reference.py:429](../../scripts/export_upstream_reference.py#L429)；[plans/plans_v1_0919/v6_draft/research_spec_v6.yaml:33](../../plans/plans_v1_0919/v6_draft/research_spec_v6.yaml#L33)。

**结论与依据：** exporter 提供 --checkpoint ps2|ps4，但消费者固定加载 checkpoint_ps2 且 patch_size=2，没有选 ps4 或显式 upstream-reference 目录的 CLI。当前 research spec 的主 checkpoint 是 patch4。按计划导出 ps4 后，gate 实际比较 ps2 与 ps4；即使保留默认 ps2 成功，也不能验证后续使用的 ps4。两个 exporter 配置还默认写同一目录，易覆盖/混用。

**独立验证：** 静态对照 exporter CLI、gate 装载路径与 YAML 的 checkpoint 即可确认；没有加载大 checkpoint。

**最小动作：** exporter 与 gate 共用显式冻结 checkpoint/config/reference-dir 参数，默认主线与 spec 一致；按 checkpoint/输入身份隔离产物目录，并在加载前匹配 manifest。

### P1 · B11 — A11 的历史指针可用，但价值阈值和成本处置不完整

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A11：`PARTIALLY_CLOSED`。**

定位：[plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md:17](../../plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md#L17)；[plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md:38](../../plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md#L38)；[plans/plans_v1_0919/v6_draft/research_spec_v6.yaml:82](../../plans/plans_v1_0919/v6_draft/research_spec_v6.yaml#L82)。

**结论与依据：** 附加处置说明可以保留历史 YAML，无需强制原地重写；beats/zero-shot 已降级、旧 2-3% gate 也已撤销。但 note 将实际 go/no-go 阈值指向 bootstrap-derived delta_MDE，混淆最小有用效应 delta_min 与指定 alpha/power/N 下的最小可检测效应 MDE；还没有处置原 A11 的后段 reference tokens 'zero extra cost' 声明。MDE 随样本量改变，不能单独定义研究价值。

**独立验证：** note 已撤回旧百分比，但第 38 行仍将实际阈值导向 MDE；此问题与历史文件是否原地修改无关。

**最小动作：** 扩展 superseding note 或当前权威入口，明确 preview/必要重放的成本；证书分列 delta_min 的价值依据、pilot 方差、alpha/power/N 与 MDE，CI 跨 delta_min 时 INCONCLUSIVE。保留历史 YAML 即可，但新规格明确引用处置后的权威合同。

### P1 · B13 — 晚期异常不会撤销已经写入的总体 PASS

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A01：`PARTIALLY_CLOSED`。**

定位：[scripts/s0_gate.py:680](../../scripts/s0_gate.py#L680)；[scripts/s0_gate.py:685](../../scripts/s0_gate.py#L685)；[scripts/s0_gate.py:687](../../scripts/s0_gate.py#L687)；[scripts/s0_gate.py:864](../../scripts/s0_gate.py#L864)；[tests/test_s0_fail_closed.py:395](../../tests/test_s0_fail_closed.py#L395)；[tests/test_s0_fail_closed.py:414](../../tests/test_s0_fail_closed.py#L414)。

**结论与依据：** 七项初始化 False、缺 reference 阻断和 all(...) 汇总本身正确，但总体状态不是严格 exception-fail-closed：先写 s0_gate_pass=True/status=ok，再执行 empty_cache；若后者抛异常，except 只追加 error，不撤销 PASS，main 仍返回 0。CPU mock 实际 run_s0_gate 重现 {s0_gate_pass:true,status:ok,error:'cleanup failed'}。新增总 gate/exit code 测试只计算自己构造的 dict/表达式，未调用真实 orchestration，所以未覆盖该错误。

**独立验证：** B13_late_exception 记录真实 run_s0_gate 返回 {s0_gate_pass:true,status:ok,error:"RuntimeError: cleanup failed"}；七项结果和清理异常均由 CPU mock 提供。

**最小动作：** 在全部必要操作完成后再提交 PASS，或外层 except 无条件重置 verdict/status；用 mock 小模型驱动真实 run_s0_gate/main，覆盖晚期异常、mismatch、缺 reference、NaN 和退出码。

### P2 · B12 — thread-local 重入检查不等于共享模型并发安全

**状态：`STATIC_CONFIRMED`；`INCONCLUSIVE`。**

定位：[earthdelta/bridge/stormer_bridge.py:635](../../earthdelta/bridge/stormer_bridge.py#L635)；[earthdelta/bridge/stormer_bridge.py:647](../../earthdelta/bridge/stormer_bridge.py#L647)；[earthdelta/bridge/stormer_bridge.py:780](../../earthdelta/bridge/stormer_bridge.py#L780)；[earthdelta/bridge/stormer_bridge.py:790](../../earthdelta/bridge/stormer_bridge.py#L790)；[tests/test_differentiable_rollout.py:250](../../tests/test_differentiable_rollout.py#L250)。

**结论与依据：** threading.local 只阻止同线程同 bridge 重入，不会阻止两个线程同时使用同一 bridge，也不能发现两个 bridge 共享同一 model。CPU 双线程反例中两个调用均成功取得所谓 guard。forward_validation 不受它保护；若未来使用 activation checkpoint，backward replay 时 hook 已移除。当前生产调用图只发现串行脚本，未发现线程池/async rollout 或 checkpoint replay，因此不是已发生的并发污染，也不应为现有串行 P0 强制搭建并发框架；但不能称该 guard 已支持并发安全。

**独立验证：** B12_thread_guard 记录两个线程都进入 guard；仅证明机制缺口。真实并发输出损害为未观察到，故运行风险状态 INCONCLUSIVE。

**最小动作：** 明确声明仅支持串行、无 replay，必要时拒绝不支持模式；若确需并发，按底层 model 共享状态做互斥并覆盖 forward/replay 生命周期。测试实际重入/并发，而非仅顺序调用两次。

### P2 · B14 — 错误结果缺少数值字段时，Markdown 报告生成再次异常

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`。**

定位：[scripts/s0_gate.py:270](../../scripts/s0_gate.py#L270)；[scripts/s0_gate.py:301](../../scripts/s0_gate.py#L301)；[scripts/s0_gate.py:752](../../scripts/s0_gate.py#L752)；[scripts/s0_gate.py:770](../../scripts/s0_gate.py#L770)；[scripts/s0_gate.py:843](../../scripts/s0_gate.py#L843)。

**结论与依据：** manifest 存在但 tensor 缺失/损坏时 upstream_available 已为 True，错误结果没有 max_abs_diff；报告却对默认字符串 'N/A' 使用 :.2e，抛 ValueError。zero-edit 异常结果也有同样格式化问题。JSON 先保存所以仍可定位问题，但要求的 Markdown 报告和正常总结会缺失；CPU 反例已复现。

**独立验证：** B14_failure_report 记录 ValueError: Unknown format code e for object of type str。

**最小动作：** 只格式化已存在且有限的数值，缺失时直接输出 N/A 和 error；以缺 tensor 与模型异常生成真实报告的回归用例。

### P2 · B15 — 四项候选约束有效，但类型合同仍会静默改变候选

**状态：`STATIC_CONFIRMED`；`FAIL_IMPLEMENTATION`；原 A03：`PARTIALLY_CLOSED`。**

定位：[earthdelta/selection.py:248](../../earthdelta/selection.py#L248)；[earthdelta/selection.py:262](../../earthdelta/selection.py#L262)；[earthdelta/selection.py:272](../../earthdelta/selection.py#L272)。

**结论与依据：** 本轮点名的 finite/bound/max_active/max_candidates 四项都实际生效，但原 A03 的完整类型校验仍未关闭：candidate_offsets 不要求 real floating-point，有限复数通过 isfinite 后被 .double() 丢弃虚部，再按另一组实数候选选择。CPU 输入 [[0.1+100j,0]] 实际返回 [0.1,0]，只有 warning，没有拒绝。另一个已要求的合同用例 K=0 也返回正常 no-op；这可作为通用 planner 约定，但不能在正式 registry 中把空表与显式 no-edit 混为一谈。未发现正常实浮点候选绕过新增幅度/支持限制。

**独立验证：** contract_repros.json 的 A03_complex_cast 记录 [0.1+100j,0] 被选为 [0.1,0]；四项正常实浮点约束均独立复核通过。

**最小动作：** 转换前验证 candidate_offsets 为受支持的实浮点 Tensor；正式有限 registry 在入口拒绝空表，并显式注册 no-edit。保留现有四项校验与 bound=0.5 的合理测试适配。


## 三、tests/ 全量改动核查

两个 commit 的 tests/ 变化共九个文件：三个旧文件修改、六个全新文件。**没有发现为了通过新增测试而削弱旧断言。** 新测试覆盖不充分是另一项判断，不能据此声称开发者篡改旧测试。逐 test 定义及每个旧 diff hunk 的机读清单见 [test_change_review.json](evidence/test_change_review.json)。

| commit / 文件 | 逐项审查结论 |
| --- | --- |
| 607fad5 / test_data.py | timezone 导入；两个 guard 用例中的四个 datetime 加 UTC；valid/invalid row 与 VerifiedRecord 共三个 fixture 加来源。原 assert 均保留，属于接口/时区适配。来源默认值自身的语义问题见 B07。 |
| 607fad5 / test_earthdelta.py | finite candidate 用例设置 `bound=0.5`、`max_active=2`，原候选与选择断言保留。合法域变明确，合理；默认 0.25 超限行为另有新测试。 |
| fb767f7 / test_bridge.py | 函数签名预期增加两个新参数，仍精确比较全部参数并保留 mutable-default 检查，合理。 |
| 607fad5 / test_metric_contract.py，25 个新定义 | 有独立代数、非均匀 F 权重、RMSE 聚合测试；只测 batched program，遗漏 shared program/batched Gram、逐切片零权重与非法 MSE 输入。几个 wrapper 对 canonical 的比较只能验证转发，不能证明共同 Q_eff。 |
| 607fad5 / test_selection_domain.py，18 个新定义 | 确实检验 NaN/Inf、bound、support、K 与 budget，包括合法负幅度和 no-op；遗漏 complex dtype 与 K=0。`empty_support` 是零候选向量，不能当空 registry 测试。 |
| 607fad5 / test_time_provenance.py，22 个新定义 | 覆盖字段/序列化和 UTC 构造。所谓 timezone-independent 没有实际改变进程 TZ；所谓 semantic correctness 主要检查枚举。legacy 默认升级反而被测试固定为预期；不能证明来源真实。 |
| fb767f7 / test_differentiable_rollout.py，11 个新定义 | 轨迹、input/bank 梯度、异常释放有价值；系数梯度用例只检查 x.grad，reentrant 用例实际顺序调用两次。标题与验证对象不一致，漏 B03/B12。 |
| fb767f7 / test_s0_fail_closed.py，13 个新定义 | 缺路径、NaN、strict-load metadata 等部分测试真实执行函数；no-state-leak exception 仅检查字段存在；总 all-gate/exit-code 自己构造 dict/表达式；最后结构测试主要检查 callable。未覆盖真实晚期异常和错误报告。 |
| fb767f7 / test_upstream_parity.py，6 个新定义 | 缺 xformers 的 subprocess 阻断测试有用；GPU/attention 与参照产物测试本机跳过符合预期。它们没有替本轮生成任何真实 upstream parity 证据。 |

## 四、执行验证、证据与限制

### 4.1 仓库 CPU 套件

实际执行完成：**290 collected，283 passed，7 skipped，0 failed，58.28 秒，exit 0**。跳过：内存不足三项、CUDA 两项、缺 upstream 产物两项；另有一条 NVML warning。完整原始输出：[pytest_cpu.log](evidence/pytest_cpu.log)。

环境起初缺 timm/xarray 等依赖，首次 collection 与早期子集未能作为完整结果。为满足本轮明确要求，在 `/tmp` 的隔离 target 补依赖后完整重跑，未改仓库依赖文件或全局包。Python 3.10、系统 torch 2.3 开发版、timm 0.9.2、xarray 2023.1.0；版本全表与命令见 [validation_environment.json](evidence/validation_environment.json)。

```bash
PYTHONPATH=/tmp/earthdelta_round2_deps:/tmp/earthdelta_round2_xarray \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -m pytest tests/ -q -rs
```

未执行 GPU/xformers 模型路径、真实 checkpoint 反序列化、训练或天气数据拉取。没有触碰后台 2015–2018 拉取任务，也没有扫描其数据并推断污染。

### 4.2 CPU 审计反例

[cpu_repros.py](evidence/cpu_repros.py) 与 [contract_repros.py](evidence/contract_repros.py) 使用小型随机模型、真实公共 helper、mock orchestration、小型归一化数组与临时 metadata-only Zarr。结果分别见 [cpu_repros.json](evidence/cpu_repros.json)、[contract_repros.json](evidence/contract_repros.json)。合成资产仅存在于临时目录，不能当天气 skill 结果。

| 检查 | 实际观测 |
| --- | --- |
| B01 | 6h/12h/24h diff_mean 各 69 个非零；6h 与 24h 的最大标准化均值偏移约 1.66e-5、7.17e-5。只证明归一化 policy 不同，未声称真实 rollout 误差值。 |
| B02 | 错误 checkpoint/norm 标识、不同 x_raw、缺 four-step 输出的合成 reference 仍 parity PASS；任意文件的 SHA gate PASS。 |
| B03 | 非零 bank 接收非零梯度，冻结 backbone 无梯度，调用方系数 grad 为 None。 |
| B04 | 旧 ellipsis 公式返回 `[2,2]`，新 helper 对相同 batched Gram 抛 einsum 错误。 |
| B05/B06 | 全零权重切片产生非有限 gain；负权重产生 MSE=-2；全目标权重与 F-only 约简改变动作排序。 |
| B07/B09 | 默认 6h 被标 reanalysis；legacy 缺字段同样升级；2015 manifest 第一项缺 2014 history，末项 target 在 2016，norm 是 placeholder。 |
| B08 | 只有完整形状 metadata、第一字段全 NaN 的新 store，`is_year_complete=True`。 |
| B12/B13/B14 | 两线程同时取得 guard；真实 gate orchestration 经晚期 mock 异常仍 PASS；错误报告生成抛 ValueError。 |
| A03/A06 | 四项主要候选约束均生效；complex 候选被改成实数（B15）；三种进程 TZ 下 manifest 时间戳完全一致。 |

### 4.3 P0-04 开始之前必须满足的准入

实现方面先修复 B01/B02/B10 的独立完整推理与身份绑定，B13/B14 的异常闭合，B03 的 caller 系数梯度，以及 B04/B05/B06 的指标合同。B07/B08/B09 要求数据完成证书、真实 history/current/target 联结、真实 hash 与分组，避免在空 store 或日历假样本上训练；这些是训练准入条件，不是在评审尚未实现的 trainer。

规格方面完成 B11 的价值阈值/功效区分与成本计费、B15 的类型/registry 验证。B12 可以通过明确串行且不支持 replay 的边界处置。实现修复后，再由有资源的后续任务取得真实 S0 PASS；本轮 CPU PASS 不替代这一步。

本轮只创建审计结果和复现材料，未修改受审 tracked 源码、未提交或推送。实际 diff 保存在 [audited.diff](evidence/audited.diff)，起始 HEAD/status 与 20 个改动文件的哈希保存在 [repository_snapshot.json](evidence/repository_snapshot.json)，最终一致性复核见 [final_integrity.json](evidence/final_integrity.json)。机读 disposition、阻塞项及代码 verdict 见 [round2_summary.json](round2_summary.json)。
