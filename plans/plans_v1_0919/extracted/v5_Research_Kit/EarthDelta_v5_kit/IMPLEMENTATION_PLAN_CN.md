# EarthDelta v5：成对编辑响应预测与预算约束的天气模型修正

核查日期：2026-09-17。基于现有 v4 和实际挂载的旧 starter 继续开发，不覆盖任何旧文件。

**交付边界：** 本轮实现的是可单元测试的 PyTorch 核心原型和实现规格。新增 41 项 CPU 测试通过；旧 starter 的 15 项 CPU 测试也独立重跑通过。未加载真实 Stormer 权重，未训练或评测 ERA5，未完成球谐数据前端、GPU 集成、FLA 内核替换、WeatherBench-X 接入和多步循环 JEPA。不能把这些测试当作天气性能、速度或创新性验证。

## 1. 保留四个方向，但进一步明确它们共同解决的问题

v4 直接预测每个候选编辑之后的未来误差。v5 将它分为：

- Reference-error JEPA：利用当前天气和过去已核验的误差记忆，预测参考模型未来可能犯什么错误。
- Edit-response JEPA：给定同一个起点和明确的编辑计划，预测它相对参考模型会造成什么预报变化。
- 物理读出和收益校准：将潜表示对应到固定物理目标和真实配对收益。
- 有限候选动态 rank：在测得的额外预算内选择编辑，而不是按复杂度或惊讶度机械扩容。

工作标题：**EarthDelta: Paired-Response Latent Planning for Memory-Conditioned Weather Model Editing**。

“成对差分”是数学恒等式，不是新定理。研究贡献假设是：将有未来不确定性的参考误差与给定执行器的编辑响应分开学习，比单头候选误差预测、直接收益回归和简单四模块拼接更有效、更节省标签，并更容易诊断选错原因。

## 2. 最新论文如何改变设计

|论文|核查版本|借鉴与边界|
|---|---|---|
|Spectral-Target Physical Latent Structuring for JEPA-Style World Models|2026-09-02 首发，2026-09-16 v3|反坍塌不保证物理信息被保留，加入固定物理读出；最新版 Fourier 目标来自物体包围框，不是天气能谱，不能直接照抄|
|Semigroup-JEPA|2026-09-09 v1|把预测潜状态反馈给下一步，并让多步目标训练状态表示；不要凭标题虚构一个额外的 semigroup loss 或天气动力学定理|
|StreamTTT|2026-09-10 v4|分开近期状态与长期 fast-weight 记忆；其 VLM 组件不是天气插件|
|CCM-LoRA|Findings ACL 2026|输入相关秩方向和计算预算不是本方案首次贡献；须与同预算路由比较|
|LeWorldModel|已阅读 jepa.py；论文 2026 年|复用 action_encoder/predict/rollout 的职责分离；不复用需要未来目标图像的规划代价|
|Flow-JEPA|2026-08-29|为随机未来的联合轨迹建模提供后续方向；当前先完成确定性编辑选择，不额外堆流匹配模块|
|EPM-JEPA|2026-06|JEPA+经验缓冲+LoRA 已有近邻；不以三者组合本身作为 novelty|
|Modified Spherical Harmonic Loss / AMSE|2025 论文|把已有谱损失作为强 baseline，不把 FFT 后 MSE 叫新损失|

本表是定向核查，不是完整首次性证明。两篇 9 月新论文的方法已阅读，但本包未确认并移植其作者训练仓库。出处和已读文件标识见 SOURCES.json。

## 3. 方法核心：先分清误差和编辑响应

设 F0 为原始冻结模型，Fs 为一个强静态适配参考，Fu 为同样起点下采用候选编辑计划 u 的模型。候选计划规定适配位置、mask、系数、持续步数和之后采用的策略。

对同一未来真值 Y：

    e0 = Y - Fs(history)
    du = Fu(history) - Fs(history)
    eu = Y - Fu(history) = e0 - du

该关系在物理场或相同固定线性投影中精确成立。**一般不满足 E(e0-du)=E(e0)-E(du)**；非线性 JEPA latent 不可直接相减冒充误差恒等式。

du 可以只运行参考和候选预报得到，不需要未来 Y，但仍有候选执行成本。e0 需要未来核验数据。对训练起点可以缓存一次参考轨迹、一次 e0、多个 du；不减少 K 个候选的实际大模型执行，也不允许把测试期未来标签用于优化任何部件。

对同一正权二次误差：

    gain(u) = ||e0||_W^2 - ||e0-du||_W^2
            = 2 <e0,du>_W - ||du||_W^2

解释：更大修正或更多 rank 不等于更好，修正必须与实际误差对齐，且不能过量。

此公式不是 RMSE、AMSE 或 log-energy loss 的普适公式。谱收益另用实际配对预报和真值计算并监督。压缩后的投影收益不等于全网格收益；保留全场标签校准。

若只把 e0、du 的条件均值代入公式，通常不等于真实条件期望收益，尤其在压缩 context 后二者仍有不确定性时。最终评分头必须以真实配对收益训练，不能宣称仅凭恒等式就实现校准。

## 4. JEPA 结构和实际原型的区别

完整目标：

    c = HistoryEncoder(legal history, coordinates, known forcings)
    m = ReadVerifiedMemory(query=c, snapshot_at_origin)
    z_e = ReferencePredictor(c,m,horizon)
    z_d = EditResponsePredictor(c,edit_descriptor,horizon)
    e_hat = PhysicalDecoder_e(z_e)
    d_hat = PhysicalDecoder_d(z_d)
    gain_hat = GeometricQuadraticGain(e_hat,d_hat) + GainCalibration(...)

编辑响应在完整输入和固定执行器下是确定的；memory 可作为压缩 context 的辅助，但要比较 response 分支有/无 memory。包内小原型共享带 memory 的 context，未强制这一分离。

目标编码器对训练期真实 e0/du 编码，EMA 仅在训练更新。加入预测端与目标端的固定物理读出，防止 latent 忽略幅度、相位或极端值；避免借由目标漂移获得低 JEPA loss。

包内当前使用按时间 GRU、跨尺度均值交互、两个预测头、EMA target encoder、固定维数物理 decoder、gain calibration 和简单方差下限。**不是完整 LeWM/LeJEPA 的复现，不是 SIGReg 实现，也不是全球天气 encoder。**

首版先直接预测 6/12/24h 的目标。之后增加 autoregressive latent predictor，并允许多步梯度训练历史编码器，借鉴 SG-JEPA。不能仅对误差 latent 施加强制 Markov/semigroup 关系：误差受天气背景和编辑延续策略影响，递推输入必须包含足够状态。

no-edit 的 response 和 gain 强制为零，结构性零目标不参与 response 防坍塌统计。不要让 no-edit 有一个可以任意预测非零收益的偏置。

## 5. 冻结的 MVP 合同

|项目|首版选择|
|---|---|
|主干|官方 Stormer patch-size-4 checkpoint|
|原生输入|128×256，69 通道，完整官方变量和压力层顺序|
|步长|固定 6h；官方网络时间输入按小时/10|
|短历史|4 帧 t−18h,t−12h,t−6h,t|
|插入位置|延续旧 starter 的 post-block (20,23)，0-based；不是标准 attention-proj LoRA|
|静态基准|Fs 独立保留；no-edit=Fs，不能意外变回 F0|
|额外编辑|每个位置 4 个组、每组 rank4；最大附加 rank16/位置|
|候选|reference、g0、g1、g01、g23、all；同 rank 要有不同方向候选|
|执行语义|候选保持 4 步即 24h，之后 reference_after_hold|
|预热目标|6/12/24h；扩展评测 3d、5d|
|谱分析|先少量标量变量，3 个固定粗尺度带；网格先审计|
|memory|固定 Fs 的已到期、已可用历史残差，预测期间只读|

当编辑结束时，Fs 从**已编辑后的当前预测状态**继续运行，不能把轨迹拼回最初起点的原始 Fs 轨迹。未来改成逐步重规划，必须重新定义与生成延续策略一致的标签。

四个 rank 组不是四个物理频带；实际输出效应可能跨尺度。mask 的活跃组数仅给出适配矩阵秩上界。两处 post-block 的 rank 不可相加后称为整个网络响应的秩。

## 6. 数据、缓存和时序合同

### 6.1 原始数据记录

    issue_time / valid_time / available_time
    source_id / event_id / split_id
    variable_order / pressure_levels / latitude / longitude
    normalization_hash / grid_hash

任何在线可用性声明必须依据 available_time，而不只是 valid_time。ERA5 回放可明确采用理想化可获得分析场，但不能当作已证明真实业务延迟。

### 6.2 成对样本建议

    inputs.zarr       # 合法输入和静态坐标，按起点/历史/通道组织
    reference.zarr    # 固定 Fs 预测，按起点/时效
    responses.zarr    # 固定候选 du，按起点/候选/时效/投影
    error_targets.zarr# 仅离线训练可读 e0 及附加能量/相关性目标
    gains.parquet     # 真正全场/分带配对收益、候选成本和元数据
    manifest.json     # 所有模型/网格/投影/编辑计划/划分版本

这是建议存储规范，包内尚未实现 Zarr/Parquet I/O。优先缓存固定投影物理量，而不是长期缓存会随着 EMA 变化的 latent。

已实现 ArtifactVersion 会对 backbone、static_adapter、edit_bank、normalization、grid、projection、split、continuation 进行一致性检查。实际使用时，各字段应填真实内容哈希；只记录一个模型名字不足以使缓存可信。

### 6.3 划分与防泄漏

把 adapter-bank 训练、JEPA/selector 训练、模型选择验证和最终测试分开，或使用严格时间块交叉拟合。划分保护窗至少覆盖历史读取范围、记忆预热和最大结果标签时效，避免同一事件邻近片段跨集合。

使用测试期已到期标签的 prequential 评测是单独协议。它不等于各预报起点独立测试，不得把有利顺序的跨测试记忆当成独立泛化。

### 6.4 减少探测成本

先小范围测候选异质性：验证集选择的 best-fixed 候选和逐样本 hindsight best 的差距是否值得研究。若所有天气都喜欢同一候选，学习复杂 planner 可能没有足够空间。

每个训练起点先 6 候选/3时效，不展开无限连续动作搜索。先复用一条 Fs 轨迹，再记录 paired delta；仍需报告所有探测成本。增加无未来标签的状态用于 du 训练属于额外训练数据，所有对应基线需有公平访问和清楚划分。

## 7. Verified memory

固定 Fs 残差记忆比混用不同模型版本更容易审计。记录至少含：issue_time、valid_time、available_time、event_id、version、key、value。

写入需满足：issue<valid<=available<=origin。特征维度和版本不匹配时 fail closed。包内记录复制 payload，避免写入后外部张量被改；重复 record id 拒绝。

先用小型可微 gated-delta replay，将时间作为递归轴，空间用粗区域或全局池化。包内实现是独立简化 PyTorch 版本，不是 FLA/StreamTTT 的完整数学复现。之后再替换内核，并测试数值误差、反向、padding、reset、batch 分组和长序列稳定性。

写入 history 的未来目标不可进入 forward；预测期间 verified memory 快照校验和不变。可另设 predictive working memory，但不可把生成值冒充新证据。训练 shuffled batch 时应在每个 episode 的合法历史内重放，而不是跨 batch 随机延续 hidden state。

## 8. 频谱网格是阻断式验收项

Stormer 输入的 shape 为 128×256 不代表与 torch-harmonics 的 128×256 节点相同。

本轮阅读的 torch-harmonics：

- `precompute_latitudes` 实际返回**余纬弧度**；应转成 geographic latitude 再比较。
- `equiangular` 使用 Clenshaw-Curtis 节点，包含两极。
- `equiangular-trapezoidal` 不是“任意等距纬度格点”的通用替代。
- 当前 RealSHT 有三角截断默认行为；显式固定 lmax/mmax，不凭数组大小猜覆盖的物理波数。

首版保留 Stormer 原始预报网格，在**分析侧**建立经过验证的重网格或投影，不为配合谱工具改变 checkpoint 输入。相同线性重网格可保持 e0-du 恒等式，但会改变分析尺度与权重，需记录。

先检查 Z500/T850/T2m/MSLP 等标量；正式风矢量能谱用 VectorSHT 或一致的涡度/散度处理，不能把分别标量变换的 u/v 能量称为坐标无关动能谱。

包内 spectral.py 只实现匹配复系数的能量/相位敏感相关性、经纬节点一致性检查。**没有实现 SHT、AMSE 或重网格。**调用方需提供实际模式 mask、m>0 双边权重、正确归一化和有效模式范围。

谱验证至少包括常量、单球谐模式、频带重建、经度平移导致相位变化但能量不变、低能量稳定性及有限差分梯度。真正全频正交 L2 的 Parseval 等价不是独立新损失。

## 9. 代码复用地图

|上游|具体入口|复用策略|
|---|---|---|
|tung-nd/stormer|stormer/models/hub/stormer.py；stormer/models/iterative_module.py|固定 backbone 与归一化合同，保留旧 bridge，显式传 plan，不用全局 hook 状态|
|lucas-maes/le-wm|jepa.py|动作/计划、预测器和 rollout 职责分离；不搬机器人环境或需要真实目标图像的 cost|
|fla-org/flash-linear-attention|fla/layers/gated_deltanet.py|先用纯 Torch 参考，再测试 chunk/recurrent 内核替换；不照搬语言模型默认宽度|
|NVIDIA/torch-harmonics|torch_harmonics/sht.py、quadrature.py|SHT 与节点合同；分析前端单独桥接|
|google-research/weatherbenchX|README 指向的 quickstart 和模块接口|生成带 xarray 坐标的预测，再用独立评测流程；它替代的是 WB2 评测代码，不是否定 WB2 benchmark|
|csubich/graphcast 的 amse 分支|v4 已定位分支，核心尚未完整移植|后续官方 AMSE 强对照；JAX到PyTorch需等价性和梯度验证|

不要把 StreamTTT、FLA、JAX GraphCast、Stormer 的全套依赖混在一个环境。先锁定一个真实 Stormer 环境与此核心包，评测可独立环境。源码、权重、数据和论文许可分别核查。包中没有复制上游代码或权重。

## 10. 首版张量与接口

    history_physical: [B,L,C,H,W]           # 真正接数据后
    history_features:[B,L,S,F]             # 包内 predictor 接收这个
    memory_read:     [B,S,M]
    edit_descriptor: [K,A]
    e0_targets:      [B,T,S,D]
    du_targets:      [B,K,T,S,D]
    predicted_gain:  [B,K,T,S]

S 是尺度，K 是候选，G 是 rank 组，不要混用。

包内可运行核心示例：

```python
import torch
from earthdelta_v5 import PairedEditPredictor, pilot_plans, select_plan

plans = pilot_plans()
descriptors = torch.tensor([p.descriptor() for p in plans])
enabled = torch.tensor([any(p.groups) for p in plans])
model = PairedEditPredictor(6, 4, descriptors.shape[-1], 6, 16)
# 演示接口；随机数组不是天气样本。
history = torch.randn(2, 4, 3, 6)
memory = torch.randn(2, 3, 4)
leads = torch.tensor([6., 12., 24.])
prediction = model(history, memory, descriptors, leads, enabled)
# 下面的 cost 是示例单位，真实使用须替换测量值。
ids = select_plan(prediction['gain'].mean((-1, -2)),
                  torch.tensor([1., 2., 2., 3., 3., 5.]), 3.,
                  tuple(p.plan_id for p in plans))
```

future truth 只进入 `make_training_targets`/离线 loss，不能进入上述 forward 或选择函数。此片段尚未执行真实 Stormer。

## 11. 分阶段训练与 Codex 提交计划

### C0：审计和原始复现

读取旧 starter，不改变原文件。记录数据是否真实可用、权重SHA256、原始环境、坐标/变量/normalization 合同。先一小段连续数据，不能上来全量下载数十年ERA5。

验收：官方原始结果与 bridge 输出一致，关闭额外编辑仍为 Fs；完整前向/反向和单步/多步归一化一致。仅“模型成功加载”不算通过。

### C1：冻结静态参考与训练编辑组

静态 Fs 与额外编辑组分开。训练随机合理 mask 和所需候选，避免只有最大 rank 学过、其余是坏子网。组的缩放固定，不通过 rank 分母变化混淆强度与容量。先两处 post-block，标准 attention-proj LoRA 作为独立后续对照。

验收：每个候选有真实留出表现；全零 mask 与 Fs 等价；inactive group 不执行；冻结主干的梯度正确回传到编辑层。报告两阶段总参数与训练成本。

### C2：成对探测缓存

冻结 Fs 和编辑组；对每个训练起点运行各候选，固定 hold=4 / reference_after_hold。生成 e0、du、实际 full-grid 和 spectral gains，并保留所有负收益。

验收：identity 在数值容差内成立；baseline 只保存一次；缓存版本完整；若 continuation/normalization/basis 改动旧缓存立即拒绝；事后 oracle 与 validation-best-fixed 差距值得继续。

### C3：JEPA＋记忆＋收益

训练 target/predictor/物理锚定和 gain 校准。目标 latent 随EMA更新时从固定物理缓存重新编码。memory 在每个起点前合法重放，不能随 batch 顺序泄漏。

验收：比较 v4 单头、v5 成对分头、同输入 direct-gain MLP、无 memory 同历史 GRU、随机打乱编辑描述。至少不仅 latent loss 更小，还要选择 regret 更小。

### C4：预算决策和部署推理

对候选成本建立实测表。predictor 批量评分，执行器只运行选择后的路径；zero-edit 为基准参考动作，打分并列优先成本更低然后稳定ID。当前纯 Torch 分组循环是正确性实现，不能声称 GPU优化。

验收：同候选顺序置换不改变结果，预算不足明确报错；实际总时延包含 controller+memory+执行，而非只数 active rank。对比 best-fixed、不带JEPA selector、相同平均实际成本的静态方案。

### C5：多步预测和最终评测

在自身预测历史上训练 latent rollout；首版始终保持固定计划及其延续语义。逐步重规划是另一协议，必须重新生成对应标签或建模编辑序列。用 WeatherBench-X 输出标准 RMSE/ACC 等以及独立频谱和选择指标。

验收：未来标签更改不改变当前推理；verified memory 在整条 forecast 内只读；3d/5d收益和短期收益一起报告；按起点/天气过程做配对时间块 bootstrap。

### C6：第二主干/更强拓展

上述成立后才接 Aurora 等第二主干。跨模型需要重新生成 executor response，不声称参数基能直接跨架构复制。Flow-JEPA、概率轨迹、物理扰动响应、v3反问题机制是另行验证的增强，不作为当前起步依赖。

## 12. 方法与 novelty 必做对照

主对照：F0、Fs、static low-rank、高容量static、同历史普通dynamic、naive JEPA+memory+rank+spectral、v4直接候选残差、v5成对响应、同context直接gain、低维物理特征配普通回归器、输出纠错。

损失公平性：空间/谱损失向可比较基线开放；记忆的全部合法历史也给长窗口GRU等基线。按总训练数据、探测成本、trainable参数、活跃计算和真实墙钟分别对齐，不只对齐rank。

机制实验：

- same rank / different direction：预测器是否识别方向，而不只学大rank更好。
- equal energy / opposite phase：功率相同但位置错时是否误判。
- predictable repeat bias / independent noise：高误差不应总换来扩容。
- memory phase/region matched shuffle：排除季节地理捷径。
- candidate order/descriptor shuffle：区分利用编辑后果与单纯天气误差预测。
- held-out edit families：作为额外压力测试，不能仅用训练见过的ID却宣称组合泛化。
- projected analytic gain / direct full-field gain / calibrated gain：验证压缩损失没有掩盖实际失败。

没有显著候选异质性时，应缩减 planner 目标；JEPA 不超过同信息回归器时，不宣称 latent planning 必要；新增控制成本超过收益时，不称为效率突破。

## 13. 实际已实现与未实现

|文件|已实现范围|尚未覆盖|
|---|---|---|
|contracts.py|不可变编辑计划、版本指纹和失配拒绝|真实文件内容哈希自动采集|
|paired.py|成对误差/响应及正权二次收益恒等式|完整物理场I/O、AMSE收益|
|rank_groups.py|按有效样本行执行组，跳过关闭组，正确梯度|Stormer真实集成、融合GPU内核|
|memory.py|记录时间/版本/事件审核、小型gated-delta replay、只读读取|持久流式数据库、FLA替换、业务观测延迟|
|selection.py|预算候选选择、确定并列规则、事后regret|真实代价测量与校准、不确定性保证|
|spectral.py|复系数能量/相位相关性、纬度节点检查|SHT/重网格/向量谱/完整AMSE|
|model.py|预计算特征上的GRU、paired JEPA、EMA、物理读出、gain原型|原始天气编码、多步递归latent、SIGReg完整复现|
|tests/|41项CPU单元测试|上游checkpoint/GPU/ERA5/跨模型结果|
|examples/synthetic_smoke.py|随机投影数组上的优化与选择链路|真实天气预测与候选选择效果|

smoke 中所有随机候选都被选择为 reference，这不是预测收益结果。训练loss下降只说明梯度通路可用；不得用它宣传方法有效。

## 14. 快速运行与工具边界

```bash
cd EarthDelta_v5_kit
# 优先使用已验证的PyTorch环境；不要盲目覆盖既有Stormer环境。
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests
PYTHONPATH=. python examples/synthetic_smoke.py
```

Windows PowerShell 可先设置 `$env:PYTHONPATH='.'` 再执行 Python。requirements 是原型最低依赖范围，不是经过验证的 Stormer/CUDA 完整锁文件。执行记录中的 Python/PyTorch 版本写入 environment.json。

本包没有伪造 `train_weather.py` 或声称下载数据后即可完整训练。把 C0–C5 做成独立可验收提交，是下一阶段真实集成任务。

## 15. 最终研究表述

> EarthDelta 将未来参考误差与参数编辑响应分开建模，通过已核验误差记忆和物理目标约束的成对JEPA，预测有限低秩编辑计划的实际收益，并在额外预算内选择更有效的修正，以检验学习编辑后果是否优于直接状态路由和模块拼接。

这一表述是待验证的科学目标；不声称首次低秩适配、首次JEPA+memory、首次谱监督或保证预报永不退化。
