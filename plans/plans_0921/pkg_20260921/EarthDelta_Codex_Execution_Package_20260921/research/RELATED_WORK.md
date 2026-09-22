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
