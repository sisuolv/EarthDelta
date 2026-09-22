# Literature-to-Code Mapping / provenance

本表刻意区分“本轮已读取源码/历史已读取”与“仅仓库manifest报告”。没有独立核实完整SHA或许可就写UNVERIFIED，不编造完整hash。以下短SHA可由Codex在已有本地clone用git rev-parse解析，之后把完整结果写external_lock.json。**不得先重新下载46个库。**

| External repo | Pinned commit / provenance | License状态 | Reused idea/code | EarthDelta位置 | Modification |
|---|---|---|---|---|---|
| tung-nd/stormer | 58dfee5a6037399a40fefd492bc00421e0c885a8；官方inference已重读 | MIT，manifest报告/既有会话检查 | 原生forecast与零diffmean还原 | bridge/stormer_bridge.py；S0独立参考 | 保留原始语义，显式编辑与trace |
| alizindari/DISeL | b6e543a13bd79caca75655cbae9b616011fe2d71；会话已读gate/variant | Apache-2.0；会话读LICENSE | 输入相关lowrank gate baseline | baselines/direct_router | 不依赖不稳定PEFT内部variant，移植标adapted |
| google-research/weatherbenchX | manifest短SHA964a35e；完整SHA UNVERIFIED | Apache-2.0，manifest报告/官方README已读 | 标准weightedmetric/aggregation | evaluation；export_wbx.py | 独立手算对照；数据合同仍自行核查 |
| scipy/scipy | 当前用户环境version与hash待P0-01记录 | BSD-3-Clause（安装许可须核对） | lsq_linear / minimize | teacher.py / selection.py | 优先复用现有，不写新QPsolver |
| pytorch/pytorch | 当前用户环境version待P0-01 | BSD-style（具体版本核对） | autograd、tensor，JVP后续 | bridge/heads/lowrank | 无全网络no_grad；当前JVP不启动 |
| tum-pbs/Solver-in-the-Loop | manifest短SHAF514fcf；完整SHA UNVERIFIED | MIT，manifest报告；旧官方README已读 | feedback纠错与多步训练协议 | baselines/output_correction.py | PyTorch独立实现，不混TF1环境 |
| ShileiCao/WeatherPEFT | b2cdc25cbbb78de272a6f6e0d532c2fe2006a15c；会话源码已读 | **UNVERIFIED / manifest NONE** | 领域PEFT对照，参数选择概念 | 可选adapted baseline | 许可明确前不得vendor，不能冒称完整复现 |
| DCDmllm/CoMoL | manifest短SHA011306a；完整SHA UNVERIFIED | **UNVERIFIED / NONE** | shared factors/core mixture | lowrank.py未来对照 | P0不换bank结构；未获许可不复制 |
| itsakk/geps | e9a865218ecffacb7007ac7d719f3741afcf8c02；会话已读layers/kolmo | **UNVERIFIED / NONE** | 受控PDE/context低秩近邻 | 独立toy诊断 | 不按默认1024网格运行；许可先查 |
| xiaolonghan2000/Weight2Token | manifest短SHA4979345；完整SHA UNVERIFIED | **UNVERIFIED / NONE** | adapter内容与规范表示 | descriptor未来研究 | 首轮同bank组合holdout；不复制weightsencoder |
| VI-MoLE / CLAW / CCM | 论文L01/L06/L05 | 官方code未核实 | counterfactualrisk/directcontext/weightdirections | 强baseline设计 | 数学独立实现须标记非官方、无原文安全保证 |

## 使用步骤
P0-01查已有clone及许可证文件；P0-02/03仅复用已明确许可部分；写入sourcefile路径、原版权、修改摘要和diffhash。代码许可不自动覆盖模型checkpoint或数据；拒绝把ROOT LICENSE=null解读为“随便复制”。

主线启动的必要工具已经在现有repo内：ExpertLoRA、paired、cached_responses、SciPyteacher与heads。文献越多不等于依赖越多。
