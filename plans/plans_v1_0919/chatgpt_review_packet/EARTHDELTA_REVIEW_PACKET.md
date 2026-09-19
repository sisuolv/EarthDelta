# EarthDelta review packet (generated 2026-09-19, for external LLM review)

Reading order: (1) merged v6 review & plan [Chinese]; (2) research spec YAML; (3) package merge map; (4) literature surveys A–E [English]; (5) the two original plan documents [Chinese].


==================================================================
===== FILE: EarthDelta_v6_Review_and_Plan_CN.md
==================================================================

# EarthDelta：两份 v1 计划的评审、novelty 判断与 v6 合并方案（草案 v0.1）

日期：2026-09-19。评审对象：`EarthDelta_v5_Research_Kit.zip`（Paired-Response Latent Planning）与 `EarthDelta_ResponseKit_20260917.zip`（Interaction-Aware Trajectory Repair）。本文件由本轮评审生成；文献对照已整合 4 路调研（A：JEPA/误差记忆/谱；B：动态 LoRA/hypernetwork；C：天气模型适配；D：响应预测/决策导向学习/敏感度），逐篇明细见 `literature/`。第 5 路（E：工具链、主干、数据镜像核查）已整合到 §2、§4.5、§4.6、§5.4，明细见 `literature/survey_E_tooling_backbones_data.md`。

---

## 0. 结论摘要

1. **核心想法有 novelty，但属于"新框架 + 新机制"而非"新算法原语"。** "在冻结天气模型上，学习参数编辑的**响应** du（无需未来真值）与参考模型的**未来误差** e0，再用 gain = 2⟨e0,du⟩ − ‖du‖² 在预算内选择编辑"这一整体表述，经两路定向调研（约 60 次 arXiv 查询、80 篇摘要页逐一核对，明细见 `literature/survey_A/B`）确认：**没有已发表工作计算逐编辑响应矩阵 R 并构造 H=RᵀQR 做 QP 选择，没有 H 加权的响应蒸馏 (a−a*)ᵀH(a−a*)，也没有"从历史预测 (b,H) 后在线解小 QP"**。但最接近的先例比原计划估计的更近：**VI-MoLE**（arXiv 2608.02528，2026-08-03）已经"学习每个 LoRA 专家的反事实剩余风险，并按认证的单位成本边际风险降低，在全局适配器预算内贪心分配"，并给出贪心最优性与 regret 界。EarthDelta 的选择规则必须定位为它的**向量/场版本实例**（响应方向与误差方向的对齐、非对角交互项、未来不可观测的 e0、多步 rollout），而不能写成新发明。其余近邻：hypernetwork/动态 LoRA（CLAW、CoMoL、DISeL、CCM-LoRA、LiST、MAPLE）、Aurora 自带的逐 rollout 步 LoRA、SPW（冻结天气模型的推理期权重扰动）、GeoQ（代理模型逐样本误差估计，含天气）、EPM-JEPA（记忆→LoRA 增量→JEPA，空结果）、JEPA-Anything（JEPA+天气+干预效应，2026-09-17）、"LoRA logit shift 一阶分解"技术笔记（2604.20313）、决策导向学习（PEAR/SPO）、Gram/Fisher 型模型合并（RegMean/Fisher/MaTS）、Solver-in-the-Loop。审稿人会问的核心问题只有一个：**"响应路线"到底买到了什么端到端条件适配（history→系数的 router/hypernetwork + 多步 loss）买不到的东西？**
   **调研 C/D 追加的三条硬事实（明细见 `literature/survey_C/D`）：**(i) 收益恒等式 gain = 2⟨e0,du⟩−‖du‖² 就是 NWP 里 FSO/EFSO 的标准观测影响量（Langland & Baker 2004；Kalnay et al. 2012 的 Δe² = (e_a−e_b)ᵀC(e_a+e_b)），不能作为贡献；(ii) 离线教师"中心差分响应列 → 正规方程 GᵀWG、GᵀW(y−model) → 加权最小二乘"就是 **Green's-function 参数校准**（Menemenlis et al. 2005 MWR；Strobach et al. 2022 GMD），必须引用并说明本方案是它的"流依赖、逐起报、摊销"版本；(iii) "冻结主干 + 低秩适配器库 + 选择器"这一架构已被 **Adapter Banks for motor control**（2609.17042，2026-09-15）占据，"把微调端点当作权重空间方向的有限差分探针 + 用短前缀读出系数"已被 **CCM**（2605.14546，2026-05-14）占据，"用 ML 学出参考态以便在没有未来真值时做基于影响的选择"已被 **Honda & Yamazaki 2024（GRL）**在观测空间做过。**四路调研一致的空白只有一个：从合法历史同时学习 e0 与逐编辑响应 du，在起报时刻用该恒等式在预算内选择参数空间编辑，并用学习到的非对角交互矩阵处理编辑间重叠。** 论文主张应收窄为"把'预报对参数编辑的敏感度'摊销为起报时刻的控制器"，对标伴随参数敏感度（Daescu & Todling 2010；Shaw & Daescu 2017，事后诊断）与 Green's-function 校准（离线、全局、非摊销）。此外，"response distillation"一词已被类增量检测领域占用（指 logit 蒸馏）；且由于 du(a)=Ra 线性，(a−a*)ᵀH(a−a*) 恰等于输出空间平方误差 ‖du(a)−du(a*)‖²_Q，应改称"输出空间系数模仿"并引用 GGN/Fisher、OBC/GPTQ、RegMean、EWC 谱系。

2. **能回答这个问题的三件事，应当成为论文主表，而不是放在"机制实验"里：** (a) **零样本迁移**到新预算、新编辑字典成员、新时效——router 按构造做不到；(b) **标签效率**——du 不需要真值，e0 需要，这与业务数据可得性结构（响应即时可得、核验滞后数天）天然匹配；(c) **可诊断性**——选错时能分辨是误差预测错了还是响应预测错了。
3. **最大的科学风险不是 novelty，而是 headroom。** 如果每个起报时刻的事后最优候选与"验证集最优固定候选"差距很小，整个 planner 没有空间。这必须是 Phase 0 的 go/no-go 门，先于任何 JEPA/记忆/谱模块。
4. **两条线应合并为一条，以 Response Kit 为主干。** v5 的 JEPA + verified memory + rank-group + 谱目标四件套会让论文看起来像模块堆叠（EPM-JEPA 已经是 JEPA+缓冲+LoRA 的组合）。保留 v5 的成对恒等式与标签效率论证、可用时间（available_time）数据合同、候选置换不变性与 no-edit 结构零；把 JEPA 潜空间、gated-delta 记忆内核、谱训练目标降级为消融或后续工作。
5. **工程上需要立即修正的事实：** HF `tungnd/stormer` 同时提供 `stormer_1.40625_patch_size_2.ckpt` 与 `stormer_1.40625_patch_size_4.ckpt`（已经 hf-mirror 核实），官方 `inference.py` 用 patch 2，计划选 patch 4 合法且前向便宜约 7 倍；Aurora 代码已内置 `LoRARollout`（按 rollout 步索引的 LoRA，作用于 qkv/proj），必须作为先例引用并定位；WB2 数据所在的 GCS 从本机不可达，数据获取路径要另行落实；计划反复引用的旧 starter（StateGatedLinear/TemporalPatchState，21 项测试）不在本文件系统上。

---

## 1. 两份计划的对照与诊断

| 维度 | v5 Research Kit | Response Kit | 诊断 |
|---|---|---|---|
| 编辑参数化 | 有限候选：2 个 post-block 位置 × 4 个 rank-4 组，6 个候选计划，hold 4 步后回到 Fs | 连续程序 a∈R^8：4 层 × 2 个时间步，各一个 rank-16 方向，盒约束 \|a_j\|≤0.25 | 两者可统一：有限候选 = 连续程序在离散支持上的特例。建议字典改为"异质专家"（见 §4.3）以提高 headroom |
| 学习目标 | 双头 JEPA 潜空间预测 e0、du，EMA 目标编码器，物理读出，gain 校准 | 教师：中心差分得 R，H=RᵀQR，b=RᵀQe，盒约束 LSQ；学生：系数蒸馏 vs 响应蒸馏，或预测 (b,H) + 在线 QP | Response Kit 更可证伪；v5 的 JEPA 对主张不是必需的（gain 最终由解码后的物理量算出，潜空间只是表示选择） |
| 上下文 | 4 帧历史 GRU + verified memory（gated-delta） | "TemporalPatchState"（来自缺失的旧 starter） | 两者都没说清控制器怎么从 4×69×128×256 得到低维输出而不过拟合。建议直接复用**冻结主干第 0 步前向的池化 token 特征**作为上下文（零额外成本，见 §4.4） |
| 预算 | 候选成本表 + 预算可行集 + 确定性并列规则 | 加性单位成本，max_active=2 | 两者都承认成本是"声明的"。rank-16 LoRA 相对 1024 宽主干的 FLOPs 差异可忽略，"预算约束"叙事在推理侧站不住，见 §6 风险 3 |
| 评测纪律 | 严格（available_time、划分、prequential 单列、bootstrap 按时间块） | 严格（信息条件分开、开环/重规划分开、WB-X） | 都好，保留 |
| 否证门 | 6.4 候选异质性；C3 regret 对照 | 第 10 节 6 条 gate | 合并保留，且把 headroom 门提到最前 |
| 代码状态 | 41 项 CPU 测试通过（本机复跑 41/41） | 50 项 CPU 测试通过（本机复跑 50/50） | 均为合成原语，无天气集成；本机需设 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` |

**共同盲点：**
- 没有"最强直接基线"的具体定义。B2（同历史 router + 多步 loss）必须是精心调过的 hypernetwork/门控 LoRA，否则任何胜利都不可信。
- 没有把"迁移"设为主实验。这是响应路线唯一按构造成立的优势。
- e0 的预测才是难点（流依赖误差预报），du 相对容易（确定性函数）。两份计划对 e0 头的输入信息（尤其是近期已核验误差）与基线（自适应偏差订正 EWMA）都没有单列。
- 谱分析（SHT 节点、AMSE）是评测层的事，不应进入 v1 训练目标。

---

## 2. 事实起点（本轮实际核验）

**环境（/mnt/afs 节点）**
- python 3.10.12，NVIDIA 容器版 torch 2.3.0a0，本节点无可见 GPU（NVML 初始化失败），torch.cuda 不可用；训练需另行申请 GPU 节点。
- 缺：xarray、zarr、netCDF4、h5py、torch_harmonics、fla、peft、huggingface_hub、xformers、lightning、timm。Stormer 官方代码硬依赖 xformers（`memory_efficient_attention`）、timm、lightning。
- 网络：GitHub 直连 `git clone` 可用（慢），`gh-proxy.com`/`ghfast.top`/`ghproxy.net`/`codeload` 可用；arXiv、PyPI（清华/阿里镜像）、hf-mirror.com 可达；google.com、GCS（WB2 数据）、huggingface.co 不可达。
- protobuf 4.24.4 与容器内 onnx 生成代码冲突，导致导入 `torch.optim` 时触发 onnx 导入失败；两个 kit 各 1 项测试因此失败，设 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 后 41/41、50/50 全部通过。

**参考仓库（已克隆到 `/mnt/afs/260010168/EarthDelta/reference/`，清单见 `_manifest.json`）**

| 目录 | HEAD | 日期 | 许可 | 用途 |
|---|---|---|---|---|
| stormer | 58dfee5（=计划 pin，即 main 最新） | 2025-03-17 | MIT | 主干、归一化、数据预处理脚本 |
| aurora | 4765abc | 2026-09-14 | MIT | 第二主干；内置 `LoRARollout` 先例 |
| ClimaX | 6d5d354 | 2023-09-30 | MIT | 1.40625° ERA5 预处理（同作者） |
| weatherbenchX | 964a35e | 2026-09-17 | Apache-2.0 | 标准评测 |
| weatherbench2 | b5fc067 | 2026-09-10 | Apache-2.0 | 数据规范、旧评测 |
| torch-harmonics | 4ac8ed3 | 2026-09-18 | BSD-3 | SHT 谱诊断 |
| flash-linear-attention | 864a87f | 2026-09-18 | MIT | gated delta rule（含 `naive` 纯 torch 参考实现） |
| peft | 50a277e | 2026-09-18 | Apache-2.0 | LoRA 参考实现 |
| graphcast-amse（分支 amse） | 6ed80d8 | 2025-05-20 | Apache-2.0 | AMSE 谱损失参考 |
| graphcast | f2f2c51 | 2026-09-04 | Apache-2.0 | 与 amse 分支对 diff |
| le-wm | 8edfeb3 | 2026-05-22 | MIT | LeWorldModel（JEPA 动作编码/预测器分离） |
| sg-jepa | 1b7794b | 2026-09-09 | MIT | Semigroup-JEPA 官方代码（含 train/evaluate_control） |
| StreamTTT | 96aed21 | 2026-09-13 | Apache-2.0 | 流式 TTT/fast-weight 记忆 |
| CoMoL | 011306a | 2026-04-29 | 无 LICENSE 文件 | 共享 A/B + 小 core 的动态 LoRA 混合 |
| DISeL | b6e543a | 2026-05-15 | Apache-2.0 | 输入相关低秩门控（何时适配） |
| PEAR | bb00e7b | 2026-05-14 | MIT | 决策导向学习（切空间投影误差） |
| Solver-in-the-Loop | f514fcf | 2022-10-19 | MIT | 求解器内学习修正的实验协议 |
| geps | e9a8652 | 2024-11-19 | 无 LICENSE 文件 | PDE 上下文低秩适配 |

第二批（依据调研 A/B 推荐，已克隆）：JEPA-Anything、lejepa（SIGReg）、text-to-lora（D0 系数蒸馏基线）、lorahub（CMA-ES 系数搜索基线）、AdaMerging（无标签熵代理基线）、fusion_bench（Fisher/RegMean/TIES/DARE 实现）、CoDA、zebra、Weight2Token、SHINE。第四批（依据调研 E，已克隆）：geoarches（ArchesWeather-M，首选第二主干）、Flow-JEPA（官方代码）、ace（ACE2，备选第二主干）。第三批（依据调研 C/D 推荐，已克隆）：WeatherPEFT（Aurora 上的 PEFT 基线与数据准备）、ai-models-ensembles（SPW 冻结权重扰动代码）、weather-regional（区域 LoRA）、ORCA、PETSA、INC、ncflow、LEADS、PyEPO、cvxpylayers、qpth、amortized-optimization-tutorial、ties-merging、task_vectors、mergekit。完整清单与许可见 `reference/README.md` 与 `reference/_manifest.json`。

**Stormer 代码事实（对设计有直接影响）**
- `Block` = adaLN-Zero 调制（由时间间隔嵌入 c 产生 shift/scale/gate 各 6×hidden）+ `MemEffAttention`（融合 `qkv` + `proj`）+ `Mlp`。`FinalLayer` 也是 adaLN。→ 适配点候选：`attn.proj`（标准 LoRA）、post-block（旧 starter 做法）、**adaLN 调制向量的低秩扰动**（FiLM 式"转向"，无矩阵乘开销，是一个值得对照的更便宜编辑族）。
- 官方 `inference.py`：`in_img_size=[128,256]`，`patch_size=2`，`hidden 1024`，`depth 24`，`heads 16`，检查点 `https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_2.ckpt`。patch 2 → 64×128 = 8192 token；HF 仓库另有 `stormer_1.40625_patch_size_4.ckpt`（2048 token），已核实存在。粗略 FLOPs（d=1024、24 层、含注意力二次项）：patch 4 约 1.7 TFLOP/前向，patch 2 约 11.5 TFLOP/前向；A100 上量级分别约 0.01 s 与 0.08 s，20 步 rollout 约 0.2 s 与 1.5 s/样本。按 9 条轨迹（参考 + 8 候选）× 2 万起报 × 20 步估算，patch 4 的探测缓存约 10–15 GPU 小时，patch 2 约 80 GPU 小时，均可承受。
- 归一化合同：输入按变量 mean/std 归一化；网络输出"归一化的增量"，按间隔（6/12/24）用 `normalize_diff_std_{l}` 反归一化，加回原始物理状态，再归一化作为下一步输入；常量变量增量置零；`time_interval` 输入为小时/10。官方评测对 6/12/24 三条路径的预报取**集合平均**——计划里"只走 6h 路径"是合理简化，但报告数字时必须注明与官方数字不可直接比。
- 数据：官方从 WB2 GCS 下载 → `regrid_wb2.py` 到 1.40625° → `process_one_step_data.py` 转 h5（每个时刻一个文件）；归一化常数已随仓库提供（1979–2018）。GCS 本机不可达 → 数据路径见 §5.4。
- 69 通道 = t2m、u10、v10、mslp + z/u/v/t/q × 13 层，顺序如 `inference.py`。

**Aurora 代码事实**
- `aurora/model/lora.py`：`LoRA`（A kaiming、B 零初始化）与 `LoRARollout(max_steps=40, mode∈{single, from_second, all})`，按 rollout 步索引选择不同 LoRA；`swin3d.py` 在每个 Swin 块的 `qkv` 与 `proj` 上加 `lora_qkv(x, rollout_step)`、`lora_proj(x, rollout_step)`，rank 8、alpha 8；步索引由 `Batch.metadata.rollout_step` 携带。MLP、Perceiver 编解码器、patch embedding 均未适配。文档明确"开关 LoRA 会改变参数/行为"。→ **"冻结天气主干 + 按时间步索引的低秩编辑"已经是发布过的工程模式**；EarthDelta 的差异必须落在"按起报状态选择/组合、显式响应建模、预算/字典迁移"上。
- Aurora 只有 0.25°/0.1°/0.4° 检查点（4.4–4.8 GB；small 0.42 GB 也是 0.25°），推理需约 40 GB 显存，微调需 80 GB → **不适合作为有限 GPU 下的第二主干**。

**工具链事实（调研 E）**
- Stormer 两个检查点均为约 5.2 GB 的 Lightning `.ckpt`（含优化器状态，模型本身约 4 亿参数），权重许可 MIT；9 个公开 fork 无一去掉 xformers 依赖，SDPA 替换需自做（注意 xformers 的 (B,N,H,D) 与 SDPA 的 (B,H,N,D) 布局差异）。
- **torch-harmonics 无法精确表示 Stormer 的 128×256 网格**：其 `equiangular` 是含两极的 Clenshaw–Curtis 节点（nlat=128 时间距 180/127=1.4173°），Stormer 是不含极点的 1.40625° 格点中心；没有"不含极点的等角"选项，`legendre-gauss` 不含极点但非均匀；经度完全一致。谱诊断前必须显式选择缓解方案（见 §4.5）。PyPI 最新 0.9.2；裸源码树不能直接 import（需 `pip install`）。
- WeatherBench-X 只能从 git 安装（不在 PyPI），依赖 `apache_beam[gcp]`、`gcsfs`、`jax[cpu]`、`xarray>=2025.7`、`numpy>=2.1.3`；`GridAreaWeighting` 与网格无关，对不含极点的 128 纬度处理正确；预测与目标必须同网格。
- flash-linear-attention：`naive_recurrent_gated_delta_rule` 等纯 torch 参考实现可用；`GatedDeltaNet` 层本身需 Triton。
- 第二主干候选（按可行性）：**ArchesWeather-M**（INRIA/geoarches，1.5°、13×121×240、340 MB、代码与权重均 BSD-3、全 `nn.Linear`、活跃维护）> **ClimaX 1.40625°**（MIT/MIT，443 MB，与 Stormer 同网格，仓库 2023 年后冻结）> ACE2-ERA5（1°，Apache-2.0，MLP 为 1×1 Conv）。GraphCast_small/NeuralGCM 为 JAX；Pangu/FuXi 为 ONNX/PT2 冻结图。
- 邻居方法代码：Flow-JEPA 有官方代码（HuoYanchen/Flow-JEPA，MIT）；CLAW、Spectral-Target JEPA、EPM-JEPA 无代码；CCM-LoRA 在 arXiv 与 GitHub 均查无（仅 ACL Anthology 条目）。`csubich/graphcast` 默认分支是 `graphcast_train`，必须显式 pin `amse`（本地已如此）。`google-deepmind/graphcast` 已重定向到 `weathernext`，GraphCast 权重自 2026-08-06 起为 CC BY 4.0。

**缺失资产**
- 本机文件系统（深度 5 内）没有任何 ERA5/zarr/nc/h5 数据；`/mnt/afs` 下无共享数据目录。
- `EarthDelta_Starter_20260915.zip`（旧 starter）不在本机；Response Kit 计划的 R01–R04 依赖其 `StateGatedLinear`/`TemporalPatchState`。若找不到，需按 §4.4 重新定义控制器输入（建议本来就换）。

---

## 3. 新颖性分析

### 3.1 定位表（按类别；逐篇明细含 arXiv 号与日期见 `literature/survey_*.md`）

| 近邻类别 | 代表 | 它已经做了什么 | EarthDelta 与之的差异（必须用实验证明的部分） |
|---|---|---|---|
| Hypernetwork / 摊销 LoRA | CLAW（2609.12278）、HyperLoRA、Text-to-LoRA | 上下文 → 直接生成 LoRA 权重/系数，任务 loss 端到端 | 无显式响应模型；换预算/换字典/换时效要重训；全部监督依赖任务标签。EarthDelta 要证明迁移与标签效率 |
| 输入相关 / 动态秩 LoRA | CoMoL（2603.00573）、DISeL（2605.19028）、CCM-LoRA（ACL 26 Findings）、AdaLoRA/DyLoRA | 按输入路由或门控低秩分支；"何时适配" | 同上；此外它们的门控不建模候选之间的交互（H 非对角） |
| 天气主干上的 LoRA | Aurora `LoRARollout` | 静态、按 rollout 步索引的 LoRA，微调 rollout | 不按起报状态选择、无响应/误差预测。是最近的工程先例，必须引用 |
| NWP 预报敏感度 | FSO/EFSO、PEFSO（2609.12296） | 用伴随/集合估计观测对预报误差的影响；多为事后诊断，PEFSO 尝试"先发制人"估计 | 对象是**参数编辑**而非观测；把影响估计**摊销**成学习模型，在真值到达前用于决策 |
| 决策导向学习 | PEAR（2605.01361）、SPO+、OptNet | 用下游决策损失训练预测器；投影到决策相关方向 | 响应蒸馏 (a−a*)ᵀH(a−a*) 是 DFL 的一个实例，应如此引用而不宣称新损失 |
| Gram/曲率型合并与修剪 | RegMean、Fisher merging、TIES | 用输入 Gram/Fisher 决定参数合并权重 | H=RᵀQR 是**输出响应**的 Gram、随流型变化且被**预测**；概念相关，用途不同 |
| 学习修正嵌入求解器 | Solver-in-the-Loop、后续混合 NWP-ML 订正 | 多步训练的学习修正项 | EarthDelta 在多个修正中按状态选择；SOL 式多步训练是 B2/B3 基线 |
| JEPA 物理世界模型 | SG-JEPA（2609.10464）、Spectral-Target JEPA（2609.04264）、Flow-JEPA、EPM-JEPA、LeWM | 潜空间动作条件预测、反坍塌、记忆缓冲 | 对 EarthDelta 主张非必需；EPM-JEPA 已是 JEPA+缓冲+LoRA，"再组合"没有 novelty。建议只作表示消融 |
| 后处理 / 自适应偏差订正 | MOS、EWMA 在线偏差订正、学习后处理 | 用近期核验误差订正输出 | EarthDelta 在模型内部修正使订正随动力学传播；EWMA 订正与同信息输出订正器（O0）是必做基线 |

### 3.2 我对 novelty 的判断

- **成立的部分：** (i) 把"编辑后果"分解为标签无关的响应与需要真值的参考误差，并指出这与业务可得性结构一致；(ii) 用预测的 (b,H) 做可迁移的小规模规划，而非端到端路由；(iii) 在冻结天气模型上系统回答"何时、何处、哪种低秩修正值得做"。三者合起来是一个可辩护的研究问题。
- **不成立/不要宣称的部分：** 成对恒等式、Gauss–Newton 局部模型、盒约束 QP、LoRA、JEPA、gated-delta 记忆、球谐谱损失，任何一个都不是新贡献。
- **审稿风险：** "组合工程"印象；"router 调好了一样强"；"headroom 太小，改进在噪声内"；"预算叙事无实际成本差异"。§4 的设计针对这四条。

### 3.3 已核实的最近邻（调研 A/B/C/D 合并；E 为工具链核查）

| # | 论文 | 它已做到 | EarthDelta 仍需额外证明/新增 |
|---|---|---|---|
| 1 | **VI-MoLE** 2608.02528（2026-08-03） | 预测每个 LoRA 专家前缀后的反事实剩余风险；按认证的"单位成本边际风险降低"在全局预算内贪心分配；贪心最优与 regret 定理 | 标量风险 → 向量响应 du 与误差 e0 的内积；非对角交互 H；e0 是真正不可观测的未来量（VI-MoLE 在留出标签上校准）；多步 rollout 与逐步编辑 |
| 2 | **CLAW** 2609.12278（2026-09-10） | 冻结世界模型 + 测试期上下文→hypernetwork→LoRA，多步动力学 | 生成之后的一切：编辑库、预算选择、响应预测、QP 规划、非线性复验。其消融"收益来自表达力而非上下文条件化"可直接引用 |
| 3 | **LiST** 2608.22370（2026-08-23） | 冻结 LoRA 库上无标签的测试期融合权重搜索 + 安全接受规则 | 用学习到的响应模型替代手工能量；交互感知 H；搜索摊销进学生；预算；时间结构 |
| 4 | **CCM-LoRA** ACL 2026 Findings 1329 | 输入相关的秩方向子集 + 预算约束目标（期望有效秩/FLOPs） | 按预测收益而非任务 loss 选择；层×秩组×时间步联合动作空间；规划/复验；系数 vs 响应蒸馏 |
| 5 | **LoRA logit shift 技术笔记** 2604.20313 + **HyperFix** 2608.11499 | 多层 LoRA 效应的一阶可加分解 + 层间耦合余项（纯理论）；子集条件化的非线性合并修正 | 数值测量 R，组装 H，把 H 作为在线盒约束 QP 的算子；以起报历史为条件 |
| 6 | **EPM-JEPA** 2606.12979（2026-06-11） | 经验记忆→LoRA 权重增量→JEPA 预测器（预注册，主结果为空） | 在预建编辑库中**选择**而非生成一个增量；分别预测 e0 与 du；真实预报技巧指标；可引用其空结果说明"朴素 记忆→LoRA"为何失败 |
| 7 | **IMPLY** 2609.12441（2026-09-11） | 物理锚定的潜空间打分在无真值下在候选 rollout 集合中选择，接近 oracle | 候选是参数编辑而非 rollout；锚点是全球场统计；显式预测误差场 |
| 8 | **GeoQ** 2608.21652（2026-08-21） | 非侵入、逐样本的代理模型误差估计（表示空间位移 + 局部支撑密度），含中期天气 | 向量化、空间分辨的 e0；闭环"估计→选择→改进"。GeoQ 是 e0 头的必做基线 |
| 9 | **SPW** 2609.08412（2026-09-08） | 冻结 Aurora/GraphCast/SFNO/AIFS 的推理期权重扰动集合；发现有效注入位置随架构而异 | 用学习到的编辑响应替代逐架构手调注入位置——SPW 是 EarthDelta 最有力的动机引用 |
| 10 | **TEFL** 2602.22520（2026-02-26） | 滚动预报的历史残差作为输入 + 显式可观测性推理 + 低秩适配器 | "verified memory + available_time 纪律 + 低秩编辑"三者组合本身不再是新意；EarthDelta 的差异在于预测**未来**误差与编辑响应并选择 |
| 11 | **Green's-function 校准** Menemenlis et al. 2005（MWR）；Strobach et al. 2022（GMD） | 逐参数扰动前向 → 有限差分核矩阵 G → 加权最小二乘正规方程 GᵀWG、GᵀW(y−model)，含跨参数响应重叠 | 这就是离线教师。新增：响应对象是多步神经 rollout 对适配器系数；逐起报、流依赖；预算/稀疏约束；**从合法历史预测 (b,H) 的摊销学生** |
| 12 | **FSO/EFSO/PQC** Langland & Baker 2004；Kalnay et al. 2012；Hotta et al. 2017；**PEFSO** 2609.12296（2026-09-10） | 观测影响 = 两个二次误差范数之差（与本方案 gain 同一标量）；PEFSO 无需重积分即可预测干预后的预报 | 干预对象是参数编辑而非观测；du、e0 由学习得到而非切线性/集合代数；不受 ~2 天切线性窗口限制；有预算与学习的交互矩阵 |
| 13 | **Honda & Yamazaki 2024**（GRL, 10.1029/2023GL107938） | 用 ML 学出参考态，使基于影响的 proactive QC 无需未来观测即可实时进行 | 最接近"无未来真值的敏感度选择"的先例：它只学参考态（隐含 e0），不学各候选的响应 du；观测空间、玩具系统 |
| 14 | **CCM: Discovering Physical Directions in Weight Space** 2605.14546（2026-05-14） | 把微调端点专家解释为权重空间方向的有限差分探针；从短 rollout 前缀读出一个合成坐标并部署合并检查点 | 只推断一个方向上的一个标量坐标；不预测 du、不建模 e0、无多编辑预算 QP。必须显式区分 |
| 15 | **Adapter Banks for Compositional Motor Control** 2609.17042（2026-09-15） | 冻结循环核心 + 残差低秩适配器库 + 冻结网络上的高层策略选择/串接适配器 | 架构层面最接近；按 RL 回报选择，无响应预测、无 e0、无 H、无预算 |
| 16 | **ARROW** 2510.09734（ICLR 2026） | 天气模型中按当前状态用 Q-learning 选择下一步 rollout 间隔/专家以抑制误差累积 | 天气领域最接近的逐起报控制器；动作是步长/专家而非参数编辑；model-free 而非预测 du/e0 的 model-based |
| 17 | **ARC-STAR** 2605.22222（2026-05-21） | 冻结 PDE 基础模型 + 部署时用无标签风险分数在算力预算内把精修路由到高风险块 | 路由对象是空间块而非权重空间编辑；分数是风险代理而非预测收益 |
| 18 | **RATL** 2609.03937 / **ORCA** 2606.14222 / **STEPS** 2605.08005 | 冻结基模型 + 因果可得的历史残差记忆 + 路由/求解得到修正（输出空间） | 修正在参数空间、进入 rollout 动力学；多步而非单块；有响应预测/规划层。ORCA 是 e0 头假设的现成验证 |
| 19 | **TaCT** 2603.19325（2026-03-17） | 天气模型中按 SAE 概念激活门控地注入参数更新（只在特定情形应用编辑） | 门控是手工概念指示器，无对未来效果的预测，无候选库与预算 |
| 20 | **DISCO** 2504.19496 / **Test-time Operator Splitting** 2602.00884 | 历史→hypernetwork→算子参数；冻结算子字典 + 测试期按前缀拟合搜索组合 | 这是"直接路线"与"无标签搜索"基线；EarthDelta 以预测响应而非前缀拟合来选择 |

其他必须引用的近邻：JEPA-Anything 2609.20800（JEPA+天气+干预效应）、SG-JEPA 2609.10464、Spectral-Target JEPA 2609.04264、McCast 2605.13197（记忆驱动的潜空间漂移修正）、HERA 2608.05523、CRAFTER 2608.05207（冻结预报器残差挖掘 + 接受门）、MAPLE 2608.15299、DISeL 2605.19028、Ouroboros 2604.02051、DA-MergeLoRA 2607.17467、MergeProbe 2606.19549、AdaMerging 2310.02575、LoraHub 2307.13269、WeatherPEFT 2509.22020、ARROW 2510.09734、FTAE-Weather 2608.09948、AdaWeather 2606.02663、HRRR 误差预测 2512.14898/2606.19026、AMSE 2501.19374、FastNet 2509.17601、BSP 2502.00472。

### 3.4 文献强制新增的基线与诊断

- **du 头的"预测无效应"基线**（Intervention Gap 2608.29998 显示 LeWM 想象的干预效应比预测零效应还差）。
- **GeoQ 式误差估计器**作为 e0 头基线；**HRRR-LSTM 式直接误差回归**作为最简 e0 基线。
- **AdaWeather 式在线混合（对最佳静态混合有对数 regret）**作为控制器基线；**VI-MoLE 式贪心认证分配**作为选择规则基线；**LiST/LoraHub 式无标签系数搜索**与 **AdaMerging 熵代理**作为"非响应式无真值选择"基线。
- **Plan-Real Spearman**（DA-LeWM 2608.18746）：报告预测收益排序与真实收益排序的相关，证明打分确实能排序候选。
- **描述符置换 / 反事实塌缩检查**（PhyLatent 2608.05720）：du 头是否真的依赖编辑描述符。
- **SPW 式随机权重扰动**作为编辑库的对照（随机扰动 vs 学习的专家）。
- **Green's-function 静态教师**：用训练期全局 a*（不随起报变化）作为"非摊销、非流依赖"的对照，量化流依赖选择的增量。
- **ORCA/RATL 式输出空间残差记忆订正**与 **FFORMPP 式按历史特征预测误差后选模型**：作为"e0 预测有用但不进入参数空间"的对照。
- **ARROW 式 RL 调度器**（同动作空间、同信息）：作为 model-free 控制器对照。
- **Test-time Operator Splitting / LiST 式前缀拟合搜索**：作为"无标签在线搜索"对照。
- **TaCT 式手工门控编辑**：作为"条件化应用单一编辑"对照。

### 3.5 对 v5 设计的直接修正

- **EMA 目标编码器默认去掉**：SG-JEPA、LeWM、LeJEPA 都论证目标网络非必需；若保留必须给出消融理由。
- **物理读出头要证明提升的是 du 的"可预报性"而不只是"可解码性"**（JEPA-x 2608.24044）。
- **谱目标只做诊断不做训练 loss**：FastNet 报告 MSH 损失单独使用会略微恶化 RMSE；AMSE 是必比的强对照。
- **"记忆 + available_time"不作为贡献点**（TEFL、McCast、HERA 已覆盖），只作为数据合同与消融特征。
- **选择规则写成决策导向学习（SPO/DFL）与 VI-MoLE 的实例**，(a−a*)ᵀH(a−a*) 写成 DFL 代理损失；Gram 型算子引用 RegMean/Fisher/MaTS 的谱系。
- **恒等式用 FSO/EFSO 的语言陈述**（Kalnay et al. 2012 形式），教师写成 Green's-function 校准的"逐起报、流依赖、摊销"版本；引用 Daescu & Todling 2010 / Shaw & Daescu 2017 作为参数敏感度的伴随先例，引用 Vonich & Hakim（2504.20238）作为"oracle 存在、实时化是开放问题"的先例。
- **改名**：response distillation → 输出空间系数模仿（output-space coefficient imitation）；学生两个变体按 Amos 的摊销优化分类命名为"完全摊销（直接预测 a*）"与"半摊销（预测 (b,H) 再解小 QP）"。
- **学生实现**：在线 QP 可用 qpth/cvxpylayers 做成可微层（若需要端到端），预算/基数约束可参考 2607.00581 的可微 top-k。

---

## 4. v6 合并方案

### 4.1 一句话主张

> 对冻结的天气预报模型和一小组低秩编辑，我们把"预报对参数编辑的敏感度"（NWP 中需要伴随/切线性模型与事后核验分析场才能得到的量）**摊销**为一个只看合法历史的起报时刻控制器：它学习（i）编辑的无标签**响应** du 与（ii）参考模型的流依赖**误差预报** e0，用 FSO 形式的精确二次收益在预算内选择编辑；相比端到端条件适配，它以更低的 regret 选择编辑，并且**无需重训即可迁移到新的预算、新的编辑字典成员与新的时效**。

工作标题（二选一）：*EarthDelta: Response-Space Planning of Low-Rank Edits for Frozen Weather Models*；或 *Learning What an Edit Will Do: Amortized Response Prediction for Budgeted Adaptation of Frozen Forecasters*。

### 4.2 三大支柱实验（论文主表）

- **P1 Headroom 与线性度（go/no-go）。** 事后逐样本最优候选 vs 验证集最优固定候选 vs Fs vs F0，在 24/72/120h、Z500/T850/T2m/MSLP 上报告；同时报告 du 对系数的局部线性误差（ε/2、ε、2ε、混合方向）。门：oracle 相对 Fs 的 RMSE 改进在配对时间块 bootstrap 下显著且 ≥ 预设阈值（建议 Z500@72h ≥ 2–3%）。不过门则缩小目标（见 §6）。
- **P2 响应路线 vs 直接路线（信息与算力对齐）。** 同上下文编码器、同字典、同训练起报集合：B2 直接 router/hypernetwork（多步 loss）、直接 gain 回归、D0 系数蒸馏、D1 响应蒸馏、U0 对角 H、U1 全 H、响应路线的有限候选版。指标：regret、RMSE 改进、有害编辑率、无编辑率、校准。
- **P3 迁移（router 按构造做不到）。** (a) 预算迁移：训练 max_active=2，测试 1 与 3；(b) 字典迁移：训练时留出 2 个专家，测试时只对新专家做少量响应探测（或依赖描述符条件化）即可规划；(c) 时效迁移：24h 训练、72h 规划；(d) 标签稀缺：e0 头只用 10%/30% 真值、du 头用 100% 无标签起报；观察退化曲线。

机制实验（副表）：全 H vs 对角 H；响应 vs 系数蒸馏；记忆特征消融（无 / EWMA 近期误差 / 学习式 gated-delta）；同秩不同方向；描述符置换；等能量反相位。

### 4.3 编辑族与字典（提高 headroom 的关键改动）

- 冻结主干 F0；训练静态 LoRA 参考 Fs（`attn.proj`，rank 16，后 6 个块或 4 个块，多步 4 步 loss）。Fs 是 B1，也是 no-edit。
- 在 Fs 之上训练 **异质字典**（K=6–8）而非随机 mask 的 rank 组：按流型/季节/区域聚类的"regime 专家"，或按误差模式（热带 vs 中高纬、上层 vs 低层、特定谱带/变量）训练的"误差模式专家"，并加多样性正则。理由：headroom 来自候选之间**方向不同**，不是秩不同。
- 每个专家的作用窗：前 h 步（h=4，即 24h）后回到 Fs（沿用 reference_after_hold），或允许 step-indexed（借鉴 Aurora `all` 模式）作为扩展。
- 连续系数版本：a∈[−ρ,ρ]^K（K 专家各一个标量）先于"层×时间步"的 8 槽版本，因为专家字典的语义更清楚、与有限候选版天然统一。
- 可选更便宜的编辑族对照：adaLN 调制向量的低秩扰动（FiLM 式）。

### 4.4 控制器输入（替代缺失的 TemporalPatchState）

- 参考前向在起报时刻本来就要跑第 0 步：取 Fs 第 0 步前向中若干块（如 12/18/23）的 token 特征做面积加权池化 + 少量粗尺度池化（如 8×16 网格），拼接 4 帧历史的低维统计与已核验近期误差特征（EWMA 与最近 k 次残差在摘要基 D 上的投影，严格按 available_time 过滤）。
- 好处：零额外前向成本；特征天然编码流型；低维输出（K 维 b + K(K+1)/2 维 H）不易过拟合；支持后续逐步重规划。
- 学习式记忆（gated-delta，FLA 的 naive 实现即可）只作消融。

### 4.5 摘要空间 D 与目标

- D 为固定线性算子：面积加权粗网格化（如 1.40625°→5.625°）+ 若干标量变量（Z500、T850、T2m、MSLP、U/V850）+ 3 个粗谱带（用经过节点校验的 SHT，仅诊断/分带，不做训练 loss）。线性保证 e_u = e0 − du 在 D 空间严格成立。
- **SHT 网格缓解方案（必须二选一并记录）**：(a) 先用固定线性算子把 128 个不含极点的纬度插值到 torch-harmonics 的 129 点含极点 Clenshaw–Curtis 网格（或保守重网格到 `legendre-gauss` 节点），再做 SHT——该插值是线性的，恒等式仍成立，但改变了分析尺度与权重；(b) 直接按 128 纬度调用 `equiangular` 并接受最多半个格点的节点偏差，只用于分带诊断而不用于任何定量声明。推荐 (a)。
- 训练侧全场 gain 与 D 空间 gain 同时缓存；最终评分头用真实全场配对 gain 校准。

### 4.6 阶段与验收（合并 C0–C6 与 R00–R06）

| 阶段 | 内容 | 验收/门 |
|---|---|---|
| S0 事实与桥接（1–2 周） | GPU 环境；xformers 或把 `MemEffAttention` 换为 SDPA 并做数值一致性测试；加载官方检查点（确认 patch size）；在小测试片段复现官方 6/24/72/120h RMSE 量级；显式 `controlled_rollout(bridge, history, plan)`，真值不进参数 | 零编辑 = 官方输出（容差内）；归一化/常量变量/间隔嵌入一致；跨分支无状态串扰 |
| S1 数据（与 S0 并行） | 落实 1.40625° 69 变量数据源（§5.4）；先 2–3 年连续数据 + 一个测试年；建立 issue/valid/available_time 记录与划分 | 变量顺序、层顺序、归一化 hash、网格 hash 全部核验 |
| S2 Fs 与字典（2 周） | 训练 Fs；训练 K 个异质专家；每个候选独立留出评测 | 全零系数 = Fs；每个专家有真实留出表现；报告参数与训练成本 |
| S3 成对探测缓存 + P1（2 周） | 每个训练起报：参考 + K 候选（有限）+ 子集上的中心差分 R；缓存 e0、du、全场/分带 gain、线性误差；artifact 版本绑定 | 恒等式在容差内；**P1 门**；探测总成本如实记录 |
| S4 控制器（3–4 周） | 上下文编码器；B2/直接 gain/D0/D1/U0/U1/有限候选响应头；教师用盒约束 LSQ + 非线性复验 | **P2 表**；训练无未来标签进入 forward；候选顺序置换不变 |
| S5 迁移与机制（2–3 周） | P3 四项 + 机制实验；成本账本用实测算子/墙钟 | **P3 表**；不夸大预算叙事 |
| S6 评测与写作（2 周） | WB-X 导出（xarray：init_time/lead_time/lat/lon/level）；RMSE/ACC/谱；配对时间块 bootstrap；极端事件个案（可与 DisasterTrace 数据联动，但不改动其仓库） | 所有方法同有效样本；失败计数保留 |
| S7 扩展（可选） | 第二主干改为 **ArchesWeather-M**（1.5°，BSD-3，全 `nn.Linear`）或 **ClimaX 1.40625°**（与 Stormer 同网格）；Aurora 因无粗分辨率权重、需 40–80 GB 显存，只作为"逐步 LoRA 先例"引用与（若有大卡）对照；逐步重规划；Flow-JEPA 式随机轨迹（官方代码已克隆） | 需重新生成响应缓存，不宣称跨架构可复制 |

### 4.7 必做基线清单

F0；Fs（静态 LoRA）；高容量静态 LoRA（参数对齐）；B2 同历史 router/hypernetwork + 多步 loss；B3 残差辅助监督；直接 gain 回归；O0 同信息输出订正器；**EWMA 自适应偏差订正**；"全部候选跑一遍取平均"的算力上界；best-fixed 候选；事后 oracle（只报不比）。文献强制新增（§3.4）：du 的预测无效应基线；GeoQ 式 e0 估计器；AdaWeather 式在线混合；VI-MoLE 式贪心认证分配；LiST/LoraHub 式无标签系数搜索；AdaMerging 熵代理；SPW 随机权重扰动库。

---

## 5. 具体工程决策

### 5.1 Stormer 接入
- 不改动上游文件；在 EarthDelta 包内子类化/包装 `Block`，为 `attn.proj` 挂显式 `coefficients` 接口（无全局 hook、无可变模块属性），`forward(x, c, plan_step_coeffs)`。
- xformers 缺失时用 `F.scaled_dot_product_attention` 替换并写 parity 测试（fp32 下逐元素容差、fp16/bf16 下统计容差）。
- 稀疏执行：无激活专家时直接走 base 投影；部分样本激活时按支持分组 microbatch；训练走 dense 可微路径，推理走显式支持路径，做等价性测试。

### 5.2 探测
- 首版中心差分（2K+1 条轨迹/样本，含 1 条可重复性检查）；float64 累加；ε∈{0.01,0.02,0.04} 诊断。`torch.func.jvp` 作为后续加速（需确认 SDPA/xformers 算子支持）。
- 有限候选版直接缓存非线性 du，不依赖线性化。

### 5.3 成本核算
- 明确写出：rank-16 LoRA 在 4–6 层的额外 FLOPs 相对主干可忽略；因此"预算"在推理侧主要体现为专家数上限（max_active）作为稀疏/稳健正则，在训练侧体现为探测轨迹数。若要让预算真的有意义，可在候选集中加入**真正昂贵的候选**（如额外一次订正前向或更高秩专家），使"便宜编辑 vs 昂贵编辑 vs 不编辑"成为真实决策。

### 5.4 数据获取（GCS 不可达）
本机可达性实测（2026-09-19）：Copernicus CDS API（202）、ECMWF open data（200）、TUM WeatherBench-1 分享页（200）、ModelScope（200）、Zenodo（200）、hf-mirror（200，慢）可达；GCS（WB2 官方数据）、AWS ERA5 桶、NCAR RDA 不可达；本机无任何现成 ERA5。

候选路径按优先级：
1. **最忠实**：Copernicus CDS 下载 0.25° ERA5（69 变量对应的 4 个单层 + 5 个多层变量 × 13 层，6 小时），再用 Stormer 自带的 `regrid_wb2.py`（保守重网格，与官方预处理同一路径）到 1.40625°，用仓库自带 1979–2018 归一化常数。量大：0.25° 每年约数百 GB，先只下 1 个训练年段 + 1 个测试年；CDS 也支持服务端 `grid` 参数直接出粗网格，但那是双线性插值，与官方保守重网格不同，只能作为应急。
2. ~~TUM WeatherBench-1 服务器的 1.40625° 打包数据~~：已核对 WB1 官方 README 的变量树，**不含 mean_sea_level_pressure**（也无 surface_pressure 可供换算），因此无法构成 Stormer 的 69 通道输入；只可作为额外诊断数据源。
3. 镜像（已核查，均不够用）：ModelScope `OneScience/ERA5` 每年只有 **5 个时间步**（远程读 HDF5 头部确认 `fields.shape=(5,243,721,1440)`，是样例集，虽然变量齐全含 mslp 与 13 层 q）；`zhangminglang/ERA5_1p5deg_V2` 只有 t、z 两个高空变量；`hhs2000/WeatherBench` 为空仓。hf-mirror 数据集搜索超时，待调研代理补充。
4. 集群侧：询问是否有能访问 GCS 的代理/跳板；若有，直接按 Stormer README 下载 WB2 zarr（`download_wb2.py` 写死读取 `gs://weatherbench2/datasets/era5/`）是最省事的路径。

**调研 E 补充的数据事实：**
- 本机实测吞吐：ModelScope 约 7 MB/s > hf-mirror 约 2.5 MB/s（且 API 经常超时）>> GCS 0.1–0.5 MB/s（元数据可解析但批量传输不可用）。
- **hf-mirror 上有 WB2 1.5° 网格的 6 小时 ERA5 镜像**：`JleeOfficial/ERA5-240x121-1979-2018`（591 GB，1979–2019，13 个变量 + 常量/归一化统计，npy 形状 (1460,240,121) 与 (1460,13,240,121)，13 层）；同作者 `ERA5-64x32-1979-2015`（38.7 GB，5.625°）可用于快速迭代。**两者均未声明许可**。若变量集包含 mslp 与 z/u/v/t/q，则"1.5° 镜像 + Stormer 自带 `regrid_wb2.py` 保守重网格到 1.40625°"就是 Stormer 作者自己的数据路径，优先级应高于 CDS。**本轮未能核实其变量集**（hf-mirror 的 API 与 resolve 端点在 16:00 后持续超时；只确认了 `v_component_of_wind.npy` 与 `norm_stats.npz` 存在）。下次网络空闲时用一条命令核实：`curl -s https://hf-mirror.com/api/datasets/JleeOfficial/ERA5-240x121-1979-2018/tree/main/train/year_1979 | python3 -c "import json,sys;print([f['path'].split('/')[-1] for f in json.load(sys.stdin)])"`，看是否含 `mean_sea_level_pressure.npy`。
- TUM WB1 下载需用真正的 GET（HEAD/PROPFIND 返回 401）：`wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F1.40625deg%2Fgeopotential&files=geopotential_1.40625deg.zip"`；1.40625° 为逐小时数据，仅 z/t/q 三个变量就约 1.15 TB，且缺 mslp。
- AWS `s3://nsf-ncar-era5` 可匿名列目录（原生 0.25° ERA5 GRIB/netCDF），是无需 CDS 账号的 0.25° 备选源；AWS `era5-pds` 拒绝匿名访问。
- 其余 hf-mirror 数据集不够用：`thainamhoang/era5-climate-learn`（1.40625° 但只有 3 个地面变量，CC-BY-4.0）、`jasonjewik/climate-learn`（5.625°/2.8125°）、`TornikeO/era5-5.625deg`。

**建议的默认数据方案（在无 GCS 代理的前提下）：** 首选上面第 2 条（1.5° 镜像 + 保守重网格，待变量集核对通过）；否则 CDS API 以 `grid=[1.40625,1.40625]` 服务端插值直接下载 128×256 的 69 个通道（4 单层 + z/u/v/t/q × 13 层，6 小时），训练年段先取 2015–2018，验证 2019，测试 2020（可加 2021–2022 作分布偏移测试）；每年约 12 GB。另下 2020 年的 0.25° 子集（若干旬）用 `regrid_wb2.py` 做保守重网格，与服务端插值结果对比，并用官方检查点在两种数据上跑零编辑 RMSE 校验分布偏移是否可接受。CDS 需要免费账号与 API key（用户提供）；新版 CDS 有单请求体积限制与排队，需按月/按变量拆分请求。数据预处理复用 Stormer 的 `process_one_step_data.py`（h5，每时刻一文件）与仓库自带归一化常数；`regridding.py` 依赖 JAX（CPU 版即可）。

---

## 6. 风险与否证条件

1. **Headroom 不足**（最大风险）→ P1 门不过：转向"何时不编辑"（有害编辑检测）或换设定（分布偏移年份、极端事件、更弱主干、更长时效），并如实缩小标题。
2. **线性区间太窄**（72h+）→ 缩小 ρ、缩短作用窗、有限候选版不依赖线性化。
3. **预算叙事无实际成本差异** → 按 §5.3 处理，不称"效率突破"。
4. **B2 同样强** → 主张收窄到迁移与标签效率；若迁移也不成立，则只保留可诊断性与工程贡献。
5. **控制器过拟合** → 冻结主干特征 + 低维输出 + 时间块交叉验证。
6. **数据/权重不可得** → S1 先行，任何结论不建立在合成数据上。

---

## 7. 参考代码地图

`/mnt/afs/260010168/EarthDelta/reference/`：见 §2 表与 `_manifest.json`；`_clone_refs.sh` 可重复克隆（含代理回退）。重点入口：
- `stormer/stormer/models/hub/stormer.py`（Block/Attention/adaLN）、`stormer/stormer/models/iterative_module.py`（rollout 归一化）、`stormer/normalization_constants/`、`stormer/stormer/data_preprocessing/`。
- `aurora/aurora/model/lora.py`、`aurora/aurora/model/swin3d.py`（LoRARollout 用法）、`aurora/aurora/rollout.py`。
- `weatherbenchX/weatherbenchX/{data_loaders,metrics,aggregation.py,weighting.py,binning.py}`。
- `torch-harmonics/torch_harmonics/{quadrature.py,sht.py}`。
- `flash-linear-attention/fla/ops/gated_delta_rule/naive.py`（纯 torch 参考）。
- `peft/src/peft/tuners/lora/`（LoRA 参考实现与 merge/unmerge 语义）。
- `sg-jepa/`、`le-wm/jepa.py`、`StreamTTT/streamttt/`（仅作表示/记忆消融参考）。
- `PEAR/`、`CoMoL/src/mocorelora/layer.py`、`DISeL/`、`Solver-in-the-Loop/`、`geps/`（基线与定位）。

---

## 8. 下一步（本轮之后）

本轮已完成：文献表（`literature/survey_A–D`，四路合计约 200 次检索、约 150 篇摘要页核对）、§0/§3 的 novelty 判断、Stormer 权重与网格核实、数据源可达性核实、`v6_draft/research_spec_v6.yaml`、`v6_draft/PACKAGE_MERGE_MAP.md`、`v6_draft/cds_request_template.py`、43 个参考仓库克隆（`reference/`）。

需要用户决策/提供的事项：
1. **是否接受"以 Response Kit 为主干、v5 的 JEPA/记忆/谱降级为消融"的合并方向**，以及主张收窄为"把预报对参数编辑的敏感度摊销为起报时刻控制器"。
2. **数据路径**：提供 CDS 账号/API key 走 §5.4 默认方案；或告知是否有能访问 GCS 的代理（可直接下载 WB2 zarr）。
3. **GPU 资源**：本节点无 GPU；S0–S3 需要至少 1 张 40 GB 级 GPU（patch 4 检查点，探测缓存约 10–15 GPU 小时）。
4. **旧 starter**（`EarthDelta_Starter_20260915.zip`）是否还需要；若不再需要，控制器输入按 §4.4 重定义。

接下来的实现顺序（见 `PACKAGE_MERGE_MAP.md` 的 8 个提交）：先把 91 项既有测试迁入统一包 → S0 桥接与 SDPA parity → S1 数据 → S2 字典 → S3 探测缓存与 P1 门。


==================================================================
===== FILE: v6_draft/research_spec_v6.yaml
==================================================================

```yaml
# EarthDelta v6 research specification (DRAFT, 2026-09-19).
# Merges v5 Research Kit + Response Kit into one line. NOT a runnable CLI config.
# Status flags are honest: nothing below has been run on real weather yet.
status: proposal_merged_v5_and_response_kit_not_weather_trained

claim:
  one_sentence: >
    We AMORTIZE forecast sensitivity to parameter edits (in NWP obtainable only with adjoint /
    tangent-linear models and a verifying analysis) into an issue-time controller that sees only
    legal history: it learns the label-free RESPONSE du of a frozen weather forecaster to each
    low-rank edit and a flow-dependent ERROR FORECAST e0 of the reference model, and selects edits
    under a budget with the exact FSO-form quadratic gain; it beats end-to-end conditional
    adaptation on regret and transfers zero-shot to new budgets, dictionary members and lead times.
  positioning:
    identity_is_not_novel: FSO/EFSO observation impact (Langland & Baker 2004; Kalnay et al. 2012)
    teacher_is_not_novel: Green's-function calibration (Menemenlis et al. 2005; Strobach et al. 2022)
    architecture_is_not_novel: Adapter Banks for motor control (arXiv 2609.17042); Aurora LoRARollout
    closest_decision_rule: VI-MoLE (arXiv 2608.02528) — scalar risk, greedy; ours vector response + Gram interactions
    closest_teacher_side: CCM weight-space finite-difference directions (arXiv 2605.14546) — one scalar coordinate
    closest_no_future_truth: Honda & Yamazaki 2024 GRL (learns reference state only, observation space)
    unoccupied: learn BOTH e0 and per-edit du from legal history at issue time; combine via the exact quadratic; budgeted selection with learned off-diagonal interactions
  naming:
    metric_formerly_response_distillation: output-space coefficient imitation   # (a-a*)^T H (a-a*) == ||du(a)-du(a*)||_Q^2 (GGN/Fisher form; OBC/GPTQ/RegMean/EWC lineage)
    student_variants: {fully_amortized: predict_a_star, semi_amortized: predict_b_H_then_tiny_QP}   # Amos amortized-optimization taxonomy
  pillars:
    P1_headroom_and_linearity: go_no_go_gate
    P2_response_route_vs_direct_route: main_table
    P3_transfer_budget_dictionary_leadtime_labels: main_table

backbone:
  repository: tung-nd/stormer
  commit: 58dfee5a6037399a40fefd492bc00421e0c885a8   # == main HEAD as of 2025-03-17 (verified 2026-09-19)
  checkpoint: stormer_1.40625_patch_size_4.ckpt       # exists on HF tungnd/stormer (verified via hf-mirror); patch_size_2 also exists
  checkpoint_sha256: null                              # fill after download
  native_grid: [128, 256]
  channels: 69                                         # order as in inference.py
  patch_size: 4
  hidden: 1024
  depth: 24
  heads: 16
  step_hours: 6
  time_interval_input: hours_div_10
  rollout_path: fixed_6h_only                          # official eval averages 6/12/24 paths; report separately
  attention_impl: xformers_or_sdpa_with_parity_test    # xformers only used in MemEffAttention
  normalization: repo_normalization_constants_1979_2018

data:
  source_preference:
    - cds_api_server_side_grid_1p40625                 # default without a GCS proxy; ~12 GB/year
    - wb2_gcs_zarr_via_proxy_then_regrid_wb2           # official path if GCS becomes reachable
  parity_check: 0p25_subset_conservative_regrid_vs_server_bilinear_zero_edit_rmse
  years: {train: [2015, 2018], val: [2019], test: [2020], shift_test: [2021, 2022]}
  time_contract: [issue_time, valid_time, available_time, event_id, split_id, normalization_hash, grid_hash]
  splits: time_blocks_with_guard_windows                # guard >= history + memory warmup + max lead
  prequential_protocol: separate_not_default

reference_and_dictionary:
  static_reference_Fs:
    type: lora
    target: attn.proj
    blocks: [18, 19, 20, 21, 22, 23]
    rank: 16
    train_loss: multi_step_4_steps_lat_weighted_mse
  edit_dictionary:
    K: 8
    type: heterogeneous_experts                        # replaces random rank-group masks
    families:
      - regime_experts_by_backbone_feature_clusters
      - error_mode_experts_by_region_or_level_or_band
    diversity_regularizer: true
    action_window_steps: 4                             # 24h, then reference_after_hold
    step_indexed_variant: optional_aurora_all_mode_style
    cheaper_edit_family_control: adaLN_modulation_lowrank_perturbation   # optional ablation
  no_edit_is_Fs: true

program:
  continuous_version: {dimension: K, bound_rho: calibrate_on_validation, max_active: 2}
  finite_version: {candidates: no_edit_plus_K_singletons_plus_selected_pairs}
  unify: finite_is_discrete_support_of_continuous

context_encoder:
  primary_input: pooled_tokens_from_reference_step0_forward   # zero extra cost; blocks [12, 18, 23]
  pooling: [area_weighted_global, coarse_8x16_grid]
  history_stats: low_dim_from_4_frames
  verified_error_features:
    ewma_recent_errors_in_summary_basis: true
    availability_filter: available_time_le_issue_time
    learned_gated_delta_memory: ablation_only            # fla naive_recurrent_gated_delta_rule as reference
  no_future_labels_in_forward: true

summary_space_D:
  linear_operator: true                                  # keeps e_u = e0 - du exact
  components: [coarsen_to_5p625, scalars(Z500,T850,T2m,MSLP,U850,V850), sht_bands(3, diagnostics_only)]
  weights_Q: [area, variable_scale, lead_time]
  sht_grid_validation: required_before_use              # torch-harmonics precompute_latitudes returns colatitudes
  sht_grid_mismatch: torch_harmonics_equiangular_includes_poles_spacing_180_over_127_vs_stormer_polefree_1p40625
  sht_grid_mitigation: linear_interpolate_to_129_lat_clenshaw_curtis_or_conservative_regrid_to_legendre_gauss  # diagnostics only

probe_and_teacher:
  finite_candidates: cache_nonlinear_du_directly
  continuous: central_difference_R                       # 2K+1 trajectories + 1 repeatability check
  epsilon_sweep: [0.01, 0.02, 0.04]
  mixed_direction_linearity_check: true
  accumulate_dtype: float64
  jvp_upgrade: later_if_ops_supported
  teacher: {solver: scipy.optimize.lsq_linear, constraint: componentwise_box, ridge: 1e-4}
  nonlinear_verification: {top_k: 3, full_field: true, keep_failures_and_no_gain: true}
  allowed_split: train_only
  cost_estimate_gpu_hours: {patch4_9traj_20k_issues_20steps: 10_to_15, patch2: ~80}

controllers:
  B2_direct_router_hypernetwork_multistep_loss: required_strong_baseline
  direct_gain_regression: required
  D0_coefficient_distillation: required
  D1_response_distillation_gram_metric: required
  U0_predict_b_diag_H_plan: required
  U1_predict_b_full_H_plan: required
  finite_candidate_response_heads: required             # e0-head + du-head + calibration
  latent_jepa_heads: ablation_only
  online_planner: {solver: scipy_L_BFGS_B_per_support, max_supports: 37, no_truth_no_probe_at_runtime: true}

baselines:
  - F0_frozen
  - Fs_static_lora
  - high_capacity_static_lora_param_matched
  - B2_router_multistep
  - B3_residual_auxiliary_supervision
  - O0_output_corrector_same_information
  - ewma_adaptive_output_bias_correction
  - run_all_candidates_and_average_compute_upper_bound
  - best_fixed_candidate_on_validation
  - hindsight_oracle_report_only
  # forced by the literature survey (see plan section 3.4)
  - predict_no_effect_for_du                 # Intervention Gap 2608.29998
  - geoq_style_error_estimator_for_e0        # 2608.21652
  - adaweather_style_online_mixture          # 2606.02663
  - vi_mole_style_greedy_certified_allocation # 2608.02528
  - list_or_lorahub_label_free_coefficient_search  # 2608.22370 / 2307.13269
  - adamerging_entropy_surrogate             # 2310.02575
  - spw_random_weight_perturbation_bank      # 2609.08412
  - greens_function_static_global_teacher    # Menemenlis 2005 (non-amortized, non-flow-dependent a*)
  - orca_or_ratl_output_space_residual_memory # 2606.14222 / 2609.03937
  - arrow_style_rl_scheduler_same_action_space # 2510.09734
  - operator_splitting_prefix_fit_search     # 2602.00884
  - tact_style_hand_gated_single_edit        # 2603.19325
diagnostics_forced_by_literature:
  - plan_real_spearman_of_predicted_vs_realized_gain   # DA-LeWM 2608.18746
  - descriptor_shuffle_counterfactual_collapse_check   # PhyLatent 2608.05720
  - readouts_improve_du_forecastability_not_only_decodability  # JEPA-x 2608.24044

experiments:
  P1: {metric: oracle_vs_best_fixed_vs_Fs_vs_F0, leads_h: [24, 72, 120], vars: [Z500, T850, T2m, MSLP],
       gate: paired_block_bootstrap_significant_and_ge_2_to_3_percent_Z500_72h, also: local_linearity_error}
  P2: {matched: [context_encoder, dictionary, issue_set, compute], metrics: [regret, rmse_gain, harmful_edit_rate, no_edit_rate, calibration]}
  P3:
    budget_transfer: {train_max_active: 2, test_max_active: [1, 3]}
    dictionary_transfer: {held_out_experts: 2, few_shot_response_probe_for_new_experts: true}
    leadtime_transfer: {train_h: 24, plan_h: 72}
    label_scarcity: {e0_head_truth_fraction: [0.1, 0.3, 1.0], du_head_unlabeled_fraction: 1.0}
  mechanism: [full_vs_diag_H, response_vs_coefficient_distillation, memory_feature_ablation,
              same_rank_different_direction, descriptor_shuffle, equal_energy_opposite_phase]

evaluation:
  library: weatherbenchX
  export: xarray_zarr(init_time, lead_time, latitude, longitude, level)
  leads_h: [6, 24, 72, 120]
  aggregate_before_sqrt: true
  bootstrap_unit: issue_time_block_or_weather_process
  spectra: sht_band_energy_diagnostics
  extremes_case_study: optional_link_to_disastertrace_read_only

falsification_gates:
  - P1_fails -> shrink_to_when_not_to_edit_or_change_setting
  - linearity_fails_at_72h -> smaller_rho_or_shorter_window_or_finite_only
  - full_H_eq_diag_H -> drop_interaction_head
  - D1_eq_D0 -> shrink_response_distillation_claim
  - B2_matches -> claim_only_transfer_and_label_efficiency
  - summary_gain_good_but_full_field_bad -> fix_D_not_cherry_pick

stages:
  S0_facts_and_bridge: {weeks: 1-2, accept: [zero_edit_equals_official, normalization_parity, no_state_leak]}
  S1_data: {weeks: parallel, accept: [variable_level_order_hash, available_time_records]}
  S2_reference_and_dictionary: {weeks: 2, accept: [zero_coeff_equals_Fs, each_expert_heldout_score]}
  S3_probe_cache_and_P1: {weeks: 2, accept: [identity_within_tol, P1_gate, probe_cost_reported]}
  S4_controllers_and_P2: {weeks: 3-4, accept: [no_future_labels_in_forward, permutation_invariance]}
  S5_transfer_and_mechanism_P3: {weeks: 2-3, accept: [measured_costs_not_declared]}
  S6_eval_and_writing: {weeks: 2}
  S7_optional: [archesweather_m_or_climax_1p40625_second_backbone, aurora_only_if_80GB_gpu_available, stepwise_replanning, stochastic_trajectories]
second_backbone_candidates:
  - {name: ArchesWeather-M, repo: INRIA/geoarches, weights: gcouairon/ArchesWeather (340 MB), grid: 1.5deg_121x240, license: BSD-3_code_and_weights, adapter_targets: all_nn_Linear}
  - {name: ClimaX, repo: microsoft/ClimaX, weights: tungnd/climax 1.40625deg.ckpt (443 MB), grid: 128x256_same_as_stormer, license: MIT, caveat: repo_frozen_2023_patch_size_4}
  - {name: ACE2-ERA5, repo: ai2cm/ace, weights: allenai/ACE2-ERA5 (1.8 GB), grid: 1deg_180x360, license: Apache-2.0, caveat: MLPs_are_1x1_conv}
  - {name: Aurora, note: no_coarse_checkpoint_40GB_inference_80GB_finetune; cite LoRARollout as precedent}

do_not_claim:
  - first_lora_on_weather_models          # Aurora ships LoRARollout
  - new_quadratic_identity_or_new_loss
  - efficiency_breakthrough_from_budget    # LoRA FLOPs negligible vs backbone
  - operational_realtime_readiness
  - causal_atmospheric_discovery_from_H
```


==================================================================
===== FILE: v6_draft/PACKAGE_MERGE_MAP.md
==================================================================

# v6 统一包：两个 kit 的模块合并映射（草案，2026-09-19）

原则：不改动 `extracted/` 下两个 kit 的原始文件；新建 `earthdelta/` 统一包；两个 kit 现有的 91 项 CPU 测试（41 + 50）改 import 路径后全部保留，再加桥接/数据/评测的新测试。

| v6 模块 | 来源 | 保留 | 改动 |
|---|---|---|---|
| `earthdelta/contracts.py` | v5 `contracts.py` + Response `program.py` | `ArtifactVersion`（版本指纹与失配拒绝）、`EditPlan` 的不可变与结构校验、`Slot/ProgramSpec` | `EditPlan` 从"rank 组掩码"泛化为"专家字典系数 + 作用窗"；`ProgramSpec.coefficients_for` 保留 dense 展开，但显式标注不跳过矩阵乘 |
| `earthdelta/paired.py` | v5 `paired.py` | `edit_responses`、`quadratic_gain`、`make_training_targets`（离线专用） | 无；增加"D 必须是线性算子"的断言与文档 |
| `earthdelta/probe.py` | Response `probe.py` | `central_response`、`local_linearity_error` | 增加对有限候选的"直接缓存非线性 du"入口；预留 `torch.func.jvp` 路径 |
| `earthdelta/geometry.py` | Response `geometry.py` | `ResponseGeometry.from_error`、`response_distillation` | 无 |
| `earthdelta/teacher.py` | Response `teacher.py` | `box_candidates`（`lsq_linear`）、`verify_candidates` | 成本从"加性单位成本"改为可注入的实测成本表 |
| `earthdelta/selection.py` | v5 `selection.py` + Response `utility.plan_from_prediction` | `select_plan`（预算可行集、确定性并列）、`selection_regret`、`plan_from_prediction`（枚举支持 + L-BFGS-B 盒 QP） | 统一接口：有限候选 = 连续程序在离散支持上的特例 |
| `earthdelta/heads.py` | Response `student.py`、`utility.InteractionUtilityHead`、v5 `model.py` | `BoundedProgramHead`、`InteractionUtilityHead`（Cholesky PSD 的 (b,H)）、v5 的 e0 头 / du 头 / 物理读出 / gain 校准 | v5 的 EMA 目标编码器与潜空间 JEPA loss 改为可选分支（消融），默认直接在摘要空间回归 e0、du；控制器输入改为冻结主干第 0 步池化特征 |
| `earthdelta/memory.py` | v5 `memory.py` | `VerifiedRecord`（issue<valid<=available 校验）、`eligible_records`、`read_memory` | `gated_delta_replay` 降级为消融；默认用 EWMA 近期误差特征 |
| `earthdelta/spectral.py` | v5 `spectral.py` | `coefficient_diagnostics`、`assert_same_latitude_nodes` | 新增 torch-harmonics 桥（`RealSHT` 节点/lmax 显式化，colatitude→latitude 转换测试） |
| `earthdelta/lowrank.py` | v5 `rank_groups.py` | 按行执行、跳过关闭分支、冻结尾部梯度测试 | 泛化为 `ExpertLoRA(attn.proj)`：K 个专家共享或独立 A/B、显式 `coefficients` 参数、dense 可微路径与 sparse 推理路径的等价测试 |
| `earthdelta/bridge/stormer_bridge.py` | 新 | — | 加载官方 ckpt（strict 加载、参数名/shape 校验）；`MemEffAttention`→SDPA 可切换 + 数值一致性测试；`WeatherStepBridge` 封装 `iterative_module.forward_validation` 的归一化/常量/间隔合同；`controlled_rollout(bridge, history, plan)`，真值不进参数 |
| `earthdelta/data/` | 新（复用 Stormer 预处理脚本） | — | `cds_download.py`（1.40625 服务端网格、按月拆请求）、`make_splits.py`（issue/valid/available_time、保护窗）、`build_probe_cache.py`（Zarr/Parquet 存储规范来自 v5 §6.2） |
| `earthdelta/eval/` | 新 | — | `export_wbx.py`（xarray：init_time/lead_time/lat/lon/level）、`paired_analysis.py`（配对时间块 bootstrap、regret、有害编辑率、成本账本） |
| `tests/` | 两个 kit 的 91 项 | 全部 | 新增：桥接零编辑等价、归一化往返、跨分支无状态串扰、dense/sparse 等价、SHT 单模态/常量/平移测试 |

删除或不再作为主路径的内容：v5 的 `PairedEditPredictor` 作为整体模型（拆成 heads 的可选分支）；v5 的 rank-组"动态 rank"叙事；Response 的"层×时间步 8 槽"首版程序（改为"K 专家系数"，8 槽作为扩展实验）；对旧 starter `StateGatedLinear`/`TemporalPatchState` 的依赖（文件不在本机）。

首批提交顺序（对应方案 §4.6）：
1. `test: import both kits' 91 tests under earthdelta/ namespace`
2. `feat: stormer bridge with SDPA parity and controlled_rollout` (S0)
3. `feat: cds downloader + split manifest with available_time` (S1)
4. `feat: static reference LoRA and heterogeneous expert dictionary` (S2)
5. `feat: probe cache + headroom report (P1)` (S3)
6. `feat: controllers B2/D0/D1/U0/U1/finite response heads (P2)` (S4)
7. `feat: transfer experiments (P3) and mechanism ablations` (S5)
8. `eval: weatherbenchX export and paired audit` (S6)


==================================================================
===== FILE: literature/survey_A_jepa_error_memory_spectral.md
==================================================================

# Survey A — JEPA / latent world models for physics & weather; forecast-error prediction; memory / fast-weight / TTT; spectral losses

Source: background literature agent, 2026-09-19. Dates are v1 arXiv submission dates read from the abs page unless flagged. The arXiv API was rate-limited (429) during this run; coverage came from the arXiv search UI plus ~45 individually opened abstract pages. Items whose abstract page was not opened are flagged at the end.

## 1. JEPA / latent world models for physics & dynamics

- **Semigroup-JEPA (SG-JEPA)** | 2609.10464 | 2026-09-09 | Andy Zeyi Liu. LeWM + physics parameter concatenated to the action; encoder+predictor trained through a discounted autoregressive latent rollout; SIGReg, no target network. Overlap HIGH (latent rollout), MEDIUM (edit-response conditioning). No error prediction, memory, spectral targets or edit selection. Code https://github.com/sg-jepa/sg-jepa ; checkpoints on HF datasets sg-jepa/sg-jepa.
- **Spectral-Target Physical Latent Structuring for JEPA-Style World Models** | 2609.04264 | v1 2026-09-02, v3 2026-09-16 | Penghao Zhu. "Physical representation laziness" in LeWM fixed with a training-time Fourier auxiliary head. Overlap HIGH (physical readout anchors + spectral targets). No code found.
- **Flow-JEPA** | 2608.29029 | 2026-08-29 | Yanchen Huo. Conditional flow matching over whole future latent sequences. Overlap LOW-MEDIUM. No code found.
- **EPM-JEPA: Operator-Side Experience Modulation** | 2606.12979 | 2026-06-11 | Vedant Pandya. Experience memory -> low-rank (LoRA) weight deltas applied to the JEPA predictor vs operand-side injection; pre-registered; headline result is NULL (4.74%, n.s.); mechanism analysis of buffer cycling, EMA target drift, LoRA settling transient. Overlap HIGH (memory + LoRA + JEPA). It generates a delta; it does not predict the response of a forecaster to a candidate edit, nor the reference error, nor select under a budget. Promised successor "PEM-JEPA" does not exist on arXiv. No code found.
- **LeWorldModel (LeWM)** | 2603.19312 | v1 2026-03-13 | Maes, Le Lidec, Scieur, LeCun, Balestriero. Two-loss stable end-to-end JEPA, ~15M params. Code https://github.com/lucas-maes/le-wm . LeJEPA/SIGReg: 2511.08544, code https://github.com/rbalestr-lab/lejepa .
- **JEPA-Anything** | 2609.20800 | 2026-09-17 | Taoyong Cui. Orthogonal Predictive Factorization; seven domains incl. weather and physical fields; explicit intervention prediction. Overlap HIGH (JEPA + weather + intervention effects) but it predicts the latent effect of an intervention on the environment, not the response of a frozen forecaster's output to a weight edit. Code https://github.com/Gen-Verse/JEPA-Anything . Predecessor Orthogonal JEPA 2608.20065.
- **M-JEPA (Tracing the Unlabeled Storm)** | 2608.22358 | 2026-08-23 | K M Anirudh. Lagrangian monsoon JEPA; frozen representation transferred to precipitation; beats 51-member ECMWF ensemble on CRPS. Overlap MEDIUM-HIGH: the only genuinely atmospheric JEPA forecasting paper found.
- **Phys-JEPA** | 2606.16076 | 2026-06-15 | Weizhi Nie. Physical + residual latent decomposition on multivariate series. Overlap MEDIUM-HIGH (readout anchors).
- **JEPA-x** | 2608.24044 | 2026-08-25 | Kehan Wen. Privileged physical state as second view; ablation: direct physical-state regression improves decodability without improving forecastability. Overlap HIGH as a caveat for physical readout heads.
- **PhyLatent** | 2608.05720 | 2026-08-06 | Xi Zeng. Names "counterfactual dynamics collapse" (predictor ignores which action was applied). Overlap HIGH — the exact failure an edit-response head risks.
- **PSG-JEPA** | 2608.06799 | 2026-08-07 | Haodong Yan. Grounds latent pairs (differences) in physical deltas. Overlap HIGH (grounding a du-latent).
- **IMPLY** | 2609.12441 | 2026-09-11 | Aman Mehta. Physically anchored scoring selects among candidate rollout sets without truth; within 0.003 of oracle on V-JEPA 2-AC. Overlap HIGH (anchored selection). Companion CALIPER 2609.08250.
- **Goal-Agnostic Joint-Embedding Predictive Control of PDEs** | 2607.21644 | 2026-07-21 | Jonathan Gallagher. Frozen action-conditioned JEPA on Navier–Stokes; MPPI with a learned linear kinetic-energy probe as the objective. Overlap HIGH (physical readout used as the selection objective).
- **Decision-Metric Alignment in Latent World Models (DA-LeWM)** | 2608.18746 | 2026-08-19 | Jiawei Wang. Latent distance does not rank candidate action sequences by real progress; Plan-Real Spearman diagnostic. Overlap MEDIUM-HIGH (edit ranking validity).
- **The Intervention Gap in Latent World Models** | 2608.29998 | 2026-08 | Donna Vakalis. On LeWM checkpoints, imagined 5-step intervention effects are worse than predicting no effect. Overlap HIGH as a threat: a predict-no-effect baseline for du is mandatory.
- **Delta-JEPA** | 2606.31232 | 2026-06-30 | Zhenghao Zhang. Latent Difference Action Decoder makes latent displacement identify the action. Overlap MEDIUM-HIGH.
- **ACPC diagnostics** | 2608.12939 | 2026-08-13 | Guo An. Bisimulation-grounded robustness diagnostic. MEDIUM.
- **Branch-JEPA** (formerly MoP-JEPA) | 2607.05238 | 2026-07-06 | Zhi Song. Finite-support latent successors. LOW-MEDIUM.
- **UniJEPA** | 2608.07409 | 2026-08-07 | ICML 2026. No EMA/stop-grad. LOW-MEDIUM (argues against EMA targets).
- **ScaleAware-JEPA** | 2606.29723 | 2026-06-29 | Guang-Xing Li. Scale-band-aware masking for multiscale physical fields. MEDIUM-HIGH (spectral bands).
- Context: V-JEPA 2 2506.09985; DINO-WM 2411.04983; PLDM 2502.14819; Koopman Dreamer 2607.19719 (EMA teacher + multi-step rollout); "The Observer Effect in World Models" 2602.12218 (invasive fine-tuning corrupts latent physics: argument for a frozen backbone + low-capacity readouts).

## 2. Predicting the ERROR of a forecast model (no future truth)

- **Predicting Forecast Error for the HRRR Using LSTM** | 2512.14898 | 2025-12-16 | David Aaron Evans. Station-point error prediction; "predicted errors can be used to adjust deterministic HRRR forecasts at the point of use". Overlap HIGH (e0 prediction), scalar additive correction only.
- **Hybrid LSTM–ViT for HRRR forecast errors** | 2606.19026 | 2026-06-17 | David Aaron Evans. ~2x precipitation-error skill; gains at short leads / active PBL. HIGH.
- **GeoQ: Geometry-Aware Conditional Quantile Error Estimation for Scientific Surrogates** | 2608.21652 | 2026-08-21 | Khoa Nguyen (LANL). Input-dependent per-query error estimation from representation-space displacement + local support density; evaluated on medium-range weather. Overlap HIGH: closest competitor to the e0 head; scalar quantile, never used to act. Mandatory baseline.
- **Forecast error diagnostics in neural weather models** | 2506.11987 | 2025-06 | Uros Perkan. Where error sensitivity and skill gain overlap. MEDIUM (prior for where edits should act).
- **Error growth / predictability of TCs in MLWP** | 2603.26165 | 2026-03-27 | Jingchen Pu. MEDIUM.
- **Certified World Models** | 2606.13092 | 2026-06 | Hongbo Wang. A-priori predictability certificate from the model's Jacobian; couples to a budgeted re-observation decision. MEDIUM-HIGH.
- **Stochastically Perturbed Weights (SPW)** | 2609.08412 | 2026-09-08 | Simon Adamov (ETH/MeteoSwiss). Inference-time raw weight perturbations of frozen Aurora/GraphCast/SFNO/AIFS for zero-cost ensembles; finding: no injection site works across models, the productive tensor group is architecture-specific. Overlap MEDIUM-HIGH: the only weather paper treating weight-space perturbations of a frozen MLWP backbone as first-class objects; strong motivating citation for learning edit responses.
- Also: 2504.20238 (IC optimization with future truth; real-time determination flagged open); 2506.22450 (Arnoldi singular vectors for MLWP); 2601.17636 (HealDA).

## 3. Memory / fast-weight / TTT for spatiotemporal forecasting

- **StreamTTT** | 2608.13416 | v1 2026-08-13, v4 2026-09-10 | Joya Chen, Zeyun Zhong. Fast weights outside the attention context + short sliding cache. MEDIUM-HIGH (architectural pattern). Code https://github.com/zeyun-zhong/StreamTTT (default branch `master`).
- **McCast: Memory-Guided Latent Drift Correction for Precipitation Nowcasting** | 2605.13197 | 2026-05-13 | Penghui Wen. Drift-Corrective Memory Bank emits a latent correction term. Overlap HIGH: closest weather-domain prior art to a verified-error memory; memory holds rollout states, no availability discipline, no discrete edits.
- **TEFL: Prediction-Residual-Guided Rolling Forecasting** | 2602.22520 | 2026-02-26 | Xiannan Huang. Past rolling-forecast residuals as input with explicit observability reasoning, integrated through a lightweight low-rank adapter; two-stage training. Overlap HIGH: verified-residual input + availability reasoning + low-rank adapter already exist together (generic time series; conditions on residuals rather than predicting future error; one adapter, no bank).
- **HERA: Historical Evidence Routing Adapter** | 2608.05523 | 2026-08-06 | Ruyi Yuan. Routes historical evidence into a frozen latent predictor via register-routed patch memory. HIGH (frozen predictor + memory + adapter scaffolding).
- **CRAFTER: When Do Corrective Features Help?** | 2608.05207 | 2026-08-05 | Fangxin Wang. Frozen forecaster + residual mining + a single validation-grounded accept/reject gate; "corrective features model the model-failure process". HIGH (edit selection under a gate).
- FlashBack Memory 2606.16342; PARA-PV 2607.08079 (retrieval + frozen FM + residual adapter + gating); Reviving Error Correction 2605.21088; MemCast 2602.03164; ForecastCompass 2605.30858.
- Fast-weight lineage: Gated DeltaNet 2412.06464; DeltaNet parallelization 2406.06484; Titans 2501.00663; TTT for time series 2409.14012; MesaNet 2506.05233. No paper found applying gated fast weights to gridded weather forecasting.

## 4. Spectral / spherical-harmonic losses and diagnostics

- **AMSE (modified spherical harmonic loss)** | 2501.19374 | 2025-01-31 | Christopher Subich | ICML 2025. Separates decorrelation from spectral-amplitude error; GraphCast effective resolution 1250 km -> 160 km. Code https://github.com/csubich/graphcast branch `amse` (train.py --spectral-amse).
- **FastNet** | 2509.17601 | 2025-09-22 | Tom Dunstan et al. MSH loss + gradient + wind decoupling; MSH and gradient losses alone may slightly degrade RMSE. HIGH (tension a gain-targeted formulation must beat).
- **MOSAIC / (Sparse) Attention to the Details** | 2604.16429 | 2026-04 | ICML 2026. Damping and aliasing fixes on HEALPix. MEDIUM-HIGH (abs page not opened).
- **Binned Spectral Power (BSP) loss** | 2502.00472 | 2025-01-31 | Chakraborty, Mohan, Maulik. HIGH. Follow-up 2607.19387 (graph-Laplacian bands, 2026-07-01).
- Latent Structured Spectral Propagators 2605.10154; spectral nudging hybrid ensembles 2603.05570; PhysMetrics.Weather 2606.10642; WP-MIP 2604.16643; "The Recipe Matters More Than the Kitchen" 2604.01215 (FFT isotropic spectrum approximates true SH spectrum only to O(l^-1)); butterfly effect / KE cascade 2609.18489.

## 5. Loud flags — JEPA/latent prediction + adapter selection; "predict error then choose a correction"

No paper found that predicts a forecast model's future error in latent space and uses it to select a correction. Four papers own large pieces:
- **VI-MoLE** | 2608.02528 | 2026-08-03 | Tom Saliencro. Learns counterfactual risk remaining after each LoRA expert prefix, certifies it, spends a global adapter budget by certified marginal risk reduction per unit cost; greedy optimality and allocation-regret theorems. The abstract selection mathematics EarthDelta claims is already proven in LLM-land; EarthDelta's delta is vector-valued field response, unobserved future e0, spectral/physical anchoring, frozen weather backbone, verified-error memory.
- **EPM-JEPA** (memory -> LoRA deltas -> JEPA, null result).
- **IMPLY** (anchored latent scoring selects among candidates without truth).
- **JEPA-Anything** (JEPA + weather + intervention-effect prediction, 2026-09-17).
Also: ST-LoRA 2404.07919; GEPS 2410.23889; WeatherPEFT 2509.22020 (PEFT on Aurora still below full fine-tuning); AdaWeather 2606.02663 (online mixture with logarithmic regret vs best static mixture — a baseline reviewers will demand); MoWE 2509.09052; VA-MoE 2412.02503; SPECTRA 2608.01751 (band-routed embedding + stage-wise LoRA, remote sensing).

## (a) Five closest prior works and what EarthDelta still adds
1. VI-MoLE — vector/field formulation, frozen physical simulator, anticipatory e0, SH band structure, verified memory; reframe the selection rule as an instantiation.
2. EPM-JEPA — select among a bank rather than generate one delta; predict e0 and du separately; real skill metric; cite its null result as motivation.
3. IMPLY — candidates are parameter edits, anchors are global-field statistics, explicit predicted error field.
4. GeoQ — vector-valued spatially resolved e0; closes the loop estimate -> select -> improve; mandatory baseline.
5. SG-JEPA — conditioning on an edit descriptor, error/response target, real forecast system; note SG-JEPA/LeWM/LeJEPA argue EMA target encoders are unnecessary.

## (c) Will "JEPA + memory + LoRA selection" be seen as module combination?
Yes, as framed in v5. Every component has a 2025–2026 antecedent in its intended role, and several carry negative results (EPM-JEPA null; FastNet spectral loss degrades RMSE; JEPA-x decodability vs forecastability; Intervention Gap worse than no-effect; SPW architecture-specific sites). The unoccupied claim: predicting at issue time, from legal history only, the future error field of a frozen forecaster AND the counterfactual response of its output to each candidate parameter edit, and using their inner product to select. Advice: lead with the bilinear selection identity; add baselines predict-no-effect (du), GeoQ (e0), AdaWeather (controller), Plan-Real Spearman (ranking validity), SPW (edit bank); demote EMA/readouts/memory/spectral to ablations; show readouts improve forecastability of du, not just decodability.

## (d) Repos (existence verified by git ls-remote)
sg-jepa/sg-jepa; lucas-maes/le-wm; rbalestr-lab/lejepa; zeyun-zhong/StreamTTT (branch master); csubich/graphcast (amse); Gen-Verse/JEPA-Anything; NVIDIA/torch-harmonics; fla-org/flash-linear-attention; test-time-training/ttt-lm-pytorch; facebookresearch/vjepa2; gaoyuezhou/dino_wm; tung-nd/stormer; microsoft/aurora.
No public code: 2609.04264, 2608.29029, 2606.12979. "PEM-JEPA" does not exist.

## Uncertainty
arXiv API 429 throughout; 2607.05238 retitled (MoP-JEPA -> Branch-JEPA); 2608.29998 v1 day not confirmed; 2604.16429 abs page not opened; abs pages not opened for 2606.16342, 2602.03164, 2605.30858, 2605.21088, 2604.16643, 2602.15040, 2608.01751, 2509.09052, 2412.02503, 2404.07919, 2410.23889, 2603.26165, 2506.22450, 2601.17636 (IDs reliable, re-check dates before citing).


==================================================================
===== FILE: literature/survey_B_dynamic_lora_hypernetwork.md
==================================================================

# Survey B — context-conditioned / dynamic / hypernetwork LoRA and budget-aware adapter selection (2024-09 to 2026-09)

Source: background literature agent, 2026-09-19. ~45 arXiv API queries, ~17 web searches, ~35 abstract pages opened; dates read from the abs page unless flagged.

## 1. Hypernetwork-generated LoRA / amortized adaptation
- **CLAW** | 2609.12278 | 2026-09-10 | Fernando Palafox. Hypernetwork generates LoRA for a frozen world model from test-time transitions; jointly pretrained. Overlap HIGH: frozen backbone + context->LoRA for multi-step rollouts, but direct weight generation with a task loss; no response prediction, gain objective, budget or edit bank. Its own ablation: advantage comes from expressive adapters rather than context conditioning. No code.
- **Text-to-LoRA (T2L)** | 2506.06105 | 2025-06-06 | Rujikorn Charakorn. Hypernetwork emits LoRA from a task description; distills a bank of 9 LoRAs (coefficient/weight distillation = EarthDelta's D0 baseline). Code https://github.com/SakanaAI/text-to-lora .
- **Doc-to-LoRA** 2602.15902 (2026-02-13); **SHINE** 2602.06358 (2026-02-06, code https://github.com/MuLabPKU/SHINE ); **LoRA-Gen** 2506.11638; **MoEGen** 2608.03275 (2026-08-04; expert codes -> hypernetwork -> instance LoRA; closest to a continuous program a learned end-to-end); **Ouroboros** 2604.02051 (2026-04-02; controller emits per-step diagonal modulation over frozen SVD LoRA bases — structurally close to Variant B's parameterization, trained by task loss); **DA-MergeLoRA** 2607.17467 (2026-07-20; hypernetwork emits per-column merging factors over a frozen LoRA bank from an unlabeled target batch); **HyperFix** 2608.11499 (2026-08-11; subset-conditioned nonlinear corrections for task-vector merging with perturbation bounds beyond linear merging — closest treatment of non-additive edit interactions, offline, weight space, no Gram/QP).
- Crowded-field block: HyperDreamBooth 2307.06949; Trans-LoRA 2405.17258; In-Context Meta LoRA 2501.17635; Meta-LoRA 2503.22352 / 2608.12389; ParametricSkills 2606.30015; Compliance2LoRA 2607.27594; Code2LoRA 2606.06492; HyLoVQA 2605.22035 (alignment loss tying feature discrepancy to parameter functional change); federated hypernetwork LoRA 2606.06154; scaling laws for hypernetwork knowledge injection 2607.19604; HyperLoRA for PDEs 2308.09290.

## 2. Input-dependent / dynamic LoRA, routing, budget-aware compute
- **VI-MoLE** | 2608.02528 | 2026-08-03 | Tom Saliencro. Learns counterfactual risk remaining after each expert prefix; upper-risk certificates; spends a global adapter budget on the token-layer action with largest certified marginal risk reduction per unit cost; greedy optimality and regret bounds. Overlap HIGH — closest prior on "predict gain per candidate, allocate under budget". Differences: scalar risk not vector response; greedy over prefixes not QP over a Gram; no cross-edit interaction; per-token LLM not multi-step rollout.
- **LiST: Local-Simplex Test-Time LoRA Fusion** | 2608.22370 | 2026-08-23 | Yihua Shao. Label-free test-time search of sample-specific fusion weights over a LoRA bank with an energy + safe acceptance rule. HIGH structurally (bank + per-input coefficient search + verify gate); heuristic energy, no learned response, no H, no budget.
- **MAPLE** | 2608.15299 | 2026-08-15 | Lie Li. Probe each layer's response to expert count, closed-form budgeted allocation. MEDIUM-HIGH ("measure response, solve allocation" in miniature; static, scalar).
- **DISeL** | 2605.19028 | 2026-05-18 | Ali Zindari. Input-dependent gates over rank-one LoRA components; diagnostics of which layers/ranks matter. MEDIUM-HIGH (the end-to-end foil). Code https://github.com/alizindari/DISeL .
- **CCM-LoRA** | Findings of ACL 2026 (2026.findings-acl.1329, pp. 26670–26689) | Rifat Rafiuddin & Rafae Abdullah. Input-dependent subset of rank directions with a budget-constrained objective on expected effective rank/FLOPs. HIGH on budget-aware context-conditioned rank selection (end-to-end task loss). No arXiv preprint or code found.
- **CoMoL** 2603.00573 (2026-02-28); **CARE** 2607.26052 (budget thermostat); **Hard-Routed MoR-LoRA** 2606.31413 (frozen experts, hard top-1); **Budgeted LoRA** 2605.04341; **LD-MoLE** 2509.25684; **SpawnLoRA** 2609.03150 (adapter-gradient cosine similarity as pairwise interaction measure); **Not All Layers Need Tuning (VLA)** 2609.18084 (2026-09-16; unlabeled diagnostic -> per-region adaptation cost -> budgeted variable-rank LoRA; HIGH minus response/gain formalism and temporal dimension).
- Rank allocation baselines: AdaLoRA 2303.10512; DyLoRA 2210.07558; SoRA 2311.11696; DR-LoRA 2601.04823; FIM-LoRA 2605.16800 (Fisher scalar -> budget-constrained integer allocation); PARA 2604.27796; Aletheia 2604.15351; ReMix 2603.10160 (RL routing over LoRAs).

## 3. Selecting among adapters by predicted utility
- **Counterfactual Routing Analysis in MoE LMs** 2605.07260 (2026-05-08): oracle best route != router pick; measures counterfactual utility, does not learn it.
- **MergeProbe** 2606.19549 (2026-06-17): predicts pairwise/set-level retention of PEFT updates after merging and drives merge/reweight/prune/route decisions. MEDIUM-HIGH (predicted consequence incl. interactions; offline scalar).
- **W2T (Weight2Token)** 2603.15990 (2026-03-16): canonicalize LoRA (QR->SVD), predict adapter performance from weights. Code https://github.com/xiaolonghan2000/Weight2Token .
- **LORAUTER** 2601.21795; **LoraHub** 2307.13269 (CMA-ES over composition coefficients; code https://github.com/sail-sg/lorahub ); **LoraRetriever** 2402.09997; **LoRAverse** 2510.15022 (submodular selection); **AdaMerging** 2310.02575 (label-free coefficients via test-time entropy; code https://github.com/EnnengYang/AdaMerging ); Adaptive Minds 2510.15416; Spectral Geometry of LoRA Adapters 2604.08844.

## 4. Model editing with predicted effect / linearized fine-tuning
- **Formalising the Logit Shift Induced by LoRA** | 2604.20313 | 2026-04-22 | Xiang Shi. First-order Fréchet expansion: multi-layer LoRA effect = linear sum of layerwise contributions + higher-order inter-layer coupling remainder. HIGH: the published math behind additive responses + coupling term; no measured R, no H, no selection, no budget.
- Task Arithmetic in the Tangent Space 2305.12827; Distilling Linearized Behavior 2605.18993; Rank-Efficient LoRA via Tangent-Space Optimization 2609.12123 (2026-09-10); Curvature-Guided LoRA 2603.29824.
- **Predicting Where Steering Vectors Succeed** 2604.15557 (2026-04-16): predicts before intervening which layer an edit will work on. MEDIUM.
- **Pre-Intervention Prediction of SAE Steering Side Effects** 2606.08365 (2026-06-06): first-order expansion predicts collateral spread and ranks candidate features. MEDIUM-HIGH conceptually.

## 5. Meta-learning for PDE / physics adaptation
- GEPS 2410.23889 (NeurIPS 2024; project page https://geps-project.github.io ; repo path not resolved by the agent — note: itsakk/geps exists and is cloned locally); CoDA 2202.01889 (code https://github.com/yuan-yin/CoDA ); Neural Context Flows 2405.02154; Zebra 2410.03437 (code https://github.com/LouisSerrano/zebra ); Unsupervised Adaptation of PDE Foundation Models 2608.07053 (LoRA with PDE-residual objective, no ground truth); hypernetwork spatially adaptive neural operators 2609.20309; graph hypernetworks for PINNs 2609.19915; **WeatherPEFT** 2509.22020 (2025-09-26; dynamic prompting + Fisher-guided parameter selection on weather FMs); **ARROW** 2510.09734 (2025-10-10; shared-private MoE over time scales + RL adaptive rollout scheduler — chooses rollout configuration per issue time in weather); **FTAE-Weather** 2608.09948 (2026-07; RL agent assigns variable/horizon-specific fusion weights over a pool of pretrained forecasters from the current state).

## (a) Five closest prior works and what EarthDelta still adds
1. VI-MoLE — vector du and e0, explicit quadratic gain, Gram H with off-diagonal interactions, multi-step rollout with per-step edits, predictable e0 field.
2. CLAW — edit bank, budgeted selection, response prediction, QP planning, nonlinear verification.
3. LiST — learned response model with error-reduction objective instead of hand energy; interaction-aware H; amortized student; budget; temporal structure.
4. CCM-LoRA — selection by predicted gain rather than task loss; layers x rank-groups x time steps; planning/verification; coefficient-vs-response distillation.
5. Logit-shift note + HyperFix — measure R numerically, assemble H, use it as the online QP operator; condition on forecast history at issue time.
Honourable mentions: AdaMerging, MAPLE, Not All Layers Need Tuning, MergeProbe, DISeL, Ouroboros, DA-MergeLoRA.

## (b) Already done? — No paper found that
- computes a per-edit response matrix R over adapter candidates and forms H = R^T Q R, b = R^T Q e;
- selects adapter edits by maximizing 2<e0,du> - ||du||^2 or an equivalent predicted-error-reduction quadratic;
- trains a student with the H-weighted loss (a-a*)^T H (a-a*) (coefficient distillation alone IS published: T2L, MoEGen, DA-MergeLoRA);
- predicts (b,H) from history and solves a tiny QP online.
Nearest misses: VI-MoLE, MAPLE, 2604.20313, HyperFix, LoraHub, AdaMerging. Caveat: 2<e0,du> - ||du||^2 = ||e0||^2 - ||e0-du||^2 is the exact error reduction; reviewers will see a derived acceptance test, so the defensible novelty is the pipeline (measured R -> H with cross terms -> box QP -> nonlinear verification -> amortized student), not the algebra.

## (c) Name of "Gram matrix of responses + QP"
Model-merging lineage: **RegMean** 2212.09849 (input-activation Gram matrices, closed-form least squares; RegMean++ 2508.03121); **Fisher Merging** 2111.09832; **MaTS** 2312.04339 (unifies Fisher merging and RegMean as one linear system). "Surgery" (2402.02705, SurgeryV2 2410.14389) is a different thing. TIES/DARE are heuristic interference resolution, not QP. Training-objective half: decision-focused learning / Smart Predict-then-Optimize (1710.08005); (a-a*)^T H (a-a*) is a DFL surrogate loss; no prior work found applying DFL to adapter selection. Recent Gram merging: Bayesian Model Merging 2605.12843, ACE-Merging 2603.02945 (dates unverified).

## (d) Repos
SakanaAI/text-to-lora; alizindari/DISeL; xiaolonghan2000/Weight2Token; MuLabPKU/SHINE; sail-sg/lorahub; EnnengYang/AdaMerging; yuan-yin/CoDA; LouisSerrano/zebra; tanganke/fusion_bench (existence not verified by the agent). No code on abs pages for CLAW, CoMoL (note: DCDmllm/CoMoL exists and is cloned locally), LiST; CCM-LoRA has no arXiv ID or code.


==================================================================
===== FILE: literature/survey_C_weather_model_adaptation.md
==================================================================

# Survey C — adaptation / correction of ML weather & climate models (2024-09 to 2026-09)

Source: background literature agent, 2026-09-19. ~30 searches; every abstract page opened to confirm ID, v1 date, first author. Note: several 2026 arXiv IDs carry a month later than the abs-page "Submitted on" date (e.g. 2608.09948 submitted 2026-07-07); dates below are the abs-page dates.

## 1. PEFT / task adaptation of weather & climate foundation models
- **WeatherPEFT** | 2509.22020 | 2025-09-26 (ICLR 2026) | Shilei Cao. TADP (encoder-derived embedding injection) + SFAS (stochastic Fisher scoring of parameters to update) on Aurora / Prithvi-WxC; benchmarks LoRA/DoRA/AdaptFormer/VPT/etc. MEDIUM-HIGH: per-task offline adaptation, never chosen per issue time. Code https://github.com/ShileiCao/WeatherPEFT .
- **Efficient Localized Adaptation of Neural Weather Forecasting (MENA)** | 2409.07585 | 2024-09-11 | Muhammad Akhtar Munir. Static regional LoRA on a global forecaster. MEDIUM. Code https://github.com/akhtarvision/weather-regional .
- **Finetuning a Weather FM with Lightweight Decoders** | 2506.19088 | 2025-06-23 | Fanny Lehmann. Frozen Aurora + shallow decoders for unseen hydrological variables. MEDIUM-LOW.
- **Efficient fine-tuning of 37-level GraphCast (Canadian analysis)** | 2408.14587 | 2024-08-26 | Christopher Subich. LOW-MEDIUM.
- **FlowDA** | 2602.06800 | 2026-02-06 | Ran Cheng. Flow-matching DA fine-tuning Aurora; FlowDA-LoRA rank 60 (37M of 1.3B). MEDIUM.
- **From Global to Local (regional downscaling heads)** | 2607.03279 | 2026-07-03 | Wiktor Kamzela. MEDIUM-LOW.
- Gravity-wave parameterization from a weather FM 2509.03816; MarsCast 2608.05054; mechanistic interpretability of a fine-tuned FM 2607.20778 (LOW).

## 2. Test-time / online / continual adaptation of forecasters
- **PETSA** | 2506.23424 | 2025-06-29 | Heitor R. Medeiros. Low-rank adapters + dynamic gating updated at test time on a frozen forecaster. HIGH on mechanism, LOW on domain (gradient updates from revealed truth; generic time series). Code https://github.com/BorealisAI/PETSA .
- **ORCA** | 2606.14222 | 2026-06-12 | Xilin Dai. Base-model error conditioned on base input AND base output ("context of errors"); black-box residual adapter, no gradients into the frozen FM; 5 TSFMs x 8 datasets. HIGH (e0-head hypothesis validated); corrects output directly. Code https://github.com/Fifthky/ORCA .
- **STEPS** | 2605.08005 | 2026-05-08 | Jiaqi Liu. TTA as Dirichlet boundary problem: prefix error propagated; Global Solver retrieves cross-window "error memory". HIGH (error memory + future error field). No code.
- **FAC / principled TTA protocol** | 2605.17250 | 2026-05-17 | Haochun Wang. "Matured ground truth only" protocol; frequency-aware calibration. MEDIUM (legality of adaptation signals).
- **FORESEE** | 2602.21757 | 2026-02-25 | Xiannan Huang. Yesterday's error -> today's correction, MoE over error dynamics, no base updates. MEDIUM-HIGH.
- **VA-MoE** | 2412.02503 | 2024-12-03 | Hao Chen. Variable-adaptive experts for incremental weather learning. MEDIUM.
- REE-TTT 2601.01605 (radar nowcasting TTT); domain-adaptive downscaling 2607.05645 (LOW).

## 3. Learned error correction, error memory, error prediction
- **RATL** | 2609.03937 | 2026-09-03 | Yuchen He. Frozen base forecaster; historical residuals as a context-keyed memory; retrieval under causal-availability constraints; set-aware router over forecast blocks/variables. HIGH: closest realization of "verified error memory read at issue time under legality + router", in output space. No code.
- **HopCast** | 2501.16587 | 2025-01-27 | Muhammad Bilal Shahid. Modern Hopfield memory of past errors read by similarity for autoregressive dynamics models. HIGH (error memory mechanism). No code.
- **HRRR forecast-error LSTM** 2512.14898 (2025-12-16) and **LSTM-ViT** 2606.19026 (2026-06-17) | David Aaron Evans. MEDIUM-HIGH (e0 head prior art; station-level).
- **AIFS-TC** | 2608.09959 | 2026-07-24 | Anna Allen. Cheap correction on frozen AIFS for TC intensity (built by an LLM agent). MEDIUM.
- **SwAIther-Precip** | 2605.16163 | 2026-05-15 | Dan Assouline. Lead-time FiLM-conditioned residual bias correction of AIFS precipitation. MEDIUM.
- **Improving precipitation in an AI model with observations** | 2609.03210 | 2026-09-02 | Julian F. Schmitt. LOW-MEDIUM.
- **Forecast error diagnostics in neural weather models** | 2506.11987 | 2025-06-13 | Uros Perkan. Measured perturb-then-respond experiments via autodiff. MEDIUM (offline analogue of R).
- **Offline+online hybrid model error correction in IFS** | 2403.03702 | 2024-03-06 | Alban Farchi (QJRMS). MEDIUM.
- **Hybrid sea-ice thermodynamics with state-dependent error parameterization** | 2601.23190 | 2026-01-30 | Giovanni De Cillis. MEDIUM.

## 4. Learned correction inside / around the solver
- **INC: Indirect Neural Corrector** | 2511.12764 | 2025-11-16 (NeurIPS 2025) | Hao Wei. Direct state corrections amplify error O(dt^-1 + L); inject into the equations instead. MEDIUM (where to inject). Code https://github.com/tum-pbs/INC .
- **ARC-STAR** | 2605.22222 | 2026-05-21 | Chengze Li. Frozen PDE FM (Poseidon) + global corrector + blockwise refiner; at deployment a label-free score routes refinement to high-risk blocks under a compute budget. HIGH (frozen host + budget-aware routing by a label-free score; spatial blocks, risk proxy not gain). No code.
- **Online RL in the Met Office Unified Model** | 2609.02566 | 2026-09-02 | Pritthijit Nath. DDPG actor applies bounded tendency corrections; trained on nudged counterfactuals, deployed without analysis access. MEDIUM-HIGH (teacher->student structure).
- **Replacing tunable parameters with state-dependent functions via RL** | 2601.04268 | 2026-01-07 | Pritthijit Nath. HIGH (state-conditioned low-dim parameter program; RL, no response prediction).

## 5. Meta-learning / in-context / context-conditioned adaptation for PDEs
- **GEPS** | 2410.23889 | 2024-10-31 (NeurIPS 2024) | Armand Kassaï Koupaï. Shared + environment-specific low-rank context modulation. HIGH (the ordinary conditional adaptation baseline). Code: local clone itsakk/geps.
- **DISCO** | 2504.19496 | 2025-04-28 | Rudy Morel. Large hypernetwork reads a short trajectory and emits a small operator network. HIGH (canonical history -> parameters -> rollout baseline). No code.
- **Test-time Generalization via Neural Operator Splitting** | 2602.00884 | 2026-01-31 | Louis Serrano. Dictionary of frozen operators + test-time search over compositions, no weight change. HIGH (bank + inference-time composition search by prefix fit, not predicted response). No code.
- **CCM: Discovering Physical Directions in Weight Space** | 2605.14546 | 2026-05-14 | Pengkai Wang. Fine-tuned endpoint experts reinterpreted as finite-difference probes of a physical direction in weight space; Calibration-Conditioned Merge infers a composition coordinate from metadata / calibrated map / a short observed rollout prefix, then deploys one merged checkpoint for the rest of the rollout. HIGH — FLAG: closest to EarthDelta's teacher; does not predict du, model e0, or solve a budgeted QP over multiple edits (one scalar coordinate along one direction). No code.
- **Zebra** 2410.03437 (code https://github.com/LouisSerrano/zebra ); **Neural Context Flows** 2405.02154 (Taylor expansion in context space; code https://github.com/ddrous/ncflow ); **CoDA** 2202.01889 (code https://github.com/yuan-yin/CoDA ; LEADS https://github.com/yuan-yin/LEADS ); **CHOP** 2606.12318; graph ICON 2603.12725; VICON 2411.16063; ICON 2304.07993.

## 6. Steering / editing / selecting among variants of a frozen weather model
- **SPW** | 2609.08412 | 2026-09-08 | Simon Adamov. Random weight perturbations of frozen Aurora/GraphCast/etc. at inference; best injection site architecture-specific. HIGH (weight-space edits on frozen weather backbone). Code https://github.com/MeteoSwiss/ai-models-ensembles .
- **Rescene** | 2608.09971 | 2026-07-30 | Minjong Cheon. 0.4M-param wrapper around a frozen 1.5 deg 6-hourly ViT weather operator (slow-clock blend + spectral perturbations each step). MEDIUM-HIGH (same backbone class, tiny per-step external controller).
- **TaCT: Target Concept Tuning** | 2603.19325 | 2026-03-17 | Shijie Ren. SAE-discovered failure concepts gate an indicator-conditioned parameter update injected into a frozen weather model; baselines LoRA/Adapter/LoREFT. HIGH (conditional application of a parameter edit; hand-derived gate, no predicted effect, no bank, no budget). No code.
- **ARROW** | 2510.09734 | 2025-10-10 (ICLR 2026) | Jindong Tian. Shared-Private MoE + RL Adaptive Rollout Scheduler choosing the next forecast interval per weather state. HIGH (closest weather-domain per-issue-time controller; selects time steps/experts by Q-learning). No code.
- **FTAE-Weather** | 2608.09948 | 2026-07-07 | Qiang Wu. Weight-Agent reads initial state + 8 forecasters' outputs and emits variable/horizon fusion weights by RL. HIGH (state-conditioned selection among a bank of whole models). No code.
- **MoWE** | 2509.09052 | 2025-09-10 | Dibyajyoti Chakraborty. Per-grid-point lead-time-conditioned gating over AI models. MEDIUM-HIGH.
- **Semantic Adapter Routing (ARIADNE)** 2606.19079 (2026-06-17); **LoGo** 2511.07129 (ACL 2026); **W2T** 2603.15990 (predicts adapter performance from weights; code https://github.com/xiaolonghan2000/Weight2Token ); **RL for Neural Model Editing** 2606.13461 (2026-06-11; LoRA-parameterized edit actions, scalar reward).

## (a) Five closest prior works
1. CCM 2605.14546 — teacher-side finite-difference probes + prefix readout; EarthDelta adds learned du and e0 models combined via the gain, multi-dim program, budget/QP, memory, 69-channel backbone.
2. ARROW 2510.09734 — weather per-state rollout controller; EarthDelta adds parameter-edit action space, model-based (du,e0) controller instead of Q-learning, budgeted multi-edit planning, frozen backbone.
3. ARC-STAR 2605.22222 — frozen host + budget-aware label-free routing; EarthDelta routes in weight space with predicted gain not risk.
4. RATL + ORCA + STEPS — frozen base + causal residual memory read at inference; EarthDelta acts in parameter space inside the rollout.
5. GEPS + DISCO — the conditional low-rank / hypernetwork baselines; EarthDelta predicts responses and plans instead of regressing coefficients.
Honourable: SPW, TaCT, Test-time Operator Splitting, FTAE-Weather.

## (b) Already done? No full match. Flags: CCM 2605.14546 (finite-difference weight-space probes + prefix readout); W2T 2603.15990 (predict adapter quality from weights, static); ARIADNE / LoGo (per-input adapter selection by similarity). Not found: learned du over multi-step rollout; e0 + du combination; QP over Gram H; two heads in weather.

## (c) Repos
tung-nd/stormer; ShileiCao/WeatherPEFT; microsoft/aurora; MeteoSwiss/ai-models-ensembles; Fifthky/ORCA; BorealisAI/PETSA; tum-pbs/INC; xiaolonghan2000/Weight2Token; ddrous/ncflow; yuan-yin/CoDA, yuan-yin/LEADS; LouisSerrano/zebra; akhtarvision/weather-regional; google-deepmind/graphcast, NASA-IMPACT/Prithvi-WxC, NVlabs/FourCastNet (alternative backbones). Not found: GEPS GitHub URL (note: itsakk/geps exists locally), ARROW, ARC-STAR, TaCT, RATL, CCM, HopCast, Rescene, FTAE-Weather, MoWE, VA-MoE, DISCO, operator splitting.


==================================================================
===== FILE: literature/survey_D_response_prediction_dfl_sensitivity.md
==================================================================

# Survey D — predicting the effect of interventions; decision-focused & response-based learning; forecast sensitivity (NWP/DA); Gram/curvature metrics

Source: background literature agent, 2026-09-19. ~70 arXiv searches, 12 web searches, ~10 Crossref lookups; every arXiv ID/date confirmed on the abs page, journal DOIs via Crossref.

## 1. Decision-focused learning / predict-then-optimize
- **PEAR** | 2605.01361 | 2026-05-02 | Junhyeong Lee. Regret gradient = prediction error projected onto the tangent space of active constraints, scaled by curvature. MEDIUM-HIGH (decision-relevant projection; shapes a training gradient, not inference-time selection). Code https://github.com/FinJun/PEAR (needs Gurobi). Venue unverified.
- SPO/SPO+ 1710.08005 (Elmachtoub & Grigas); Task-based end-to-end learning 1703.04529 (Donti, Amos, Kolter); OptNet 1703.00443 / qpth; cvxpylayers 1910.12430; DFL survey 2307.13565 (Mandi et al., JAIR) — "predict the QP parameters (b,H) directly" is not one of its four gradient-based families.
- **Decision Geometry of Covariance Estimation (exact regret identity)** | 2606.27462 | 2026-06-25 | Xavier Fonseca. MEDIUM-HIGH (exact quadratic regret identity on a low-dim decision subspace).
- **Decision-focused Sparse Tangent Portfolio Optimization** | 2607.00581 | 2026-07-01 | Haeun Jeon. Differentiable exact cardinality-k selection. MEDIUM-HIGH (template for a differentiable budget).
- Differentiable knapsack / top-k via DP 2601.21775; **Forecast Skill Is Not Decision Skill** 2512.14779 (Raeth & Ludwig; motivation, lists "optimize MLWP for decisions" as open); solver-free PtO 2606.19587; cost-sensitive DFL 2605.18005; dual perspective 2511.04909.

## 2. Forecast sensitivity & impact estimation in NWP / DA
- **PEFSO** | 2609.12296 | 2026-09-10 | Fumitoshi Kawasaki, Shunji Kotsuki. Observation impact without reintegration (stronger tangent-linear approx than EFSO); ADD-SEL-PRE updates the forecast ensemble after denying observations. HIGH (predict consequence of an intervention cheaply, then act); observation-space, tangent-linear, needs verifying analysis, ~2-day validity.
- **FSO** Langland & Baker 2004 (Tellus A 56(3), DOI 10.3402/tellusa.v56i3.14413); **EFSO** Kalnay, Ota, Miyoshi, Liu 2012 (Tellus A 64:18462, DOI 10.3402/tellusa.v64i0.18462); Liu & Kalnay 2008 (QJRMS, DOI 10.1002/qj.280); Kotsuki, Kurosawa, Miyoshi 2019 (QJRMS, DOI 10.1002/qj.3534). HIGH on the gain identity (see below).
- **Proactive QC** Hotta, Chen, Kalnay, Ota, Miyoshi 2017 (MWR 145(8), DOI 10.1175/MWR-D-16-0290.1). Select interventions by estimated impact, then rerun; needs a 6-h-later analysis. HIGH.
- **ML enables real-time Proactive QC** Takumi Honda & Atsushi Yamazaki 2024 (GRL, DOI 10.1029/2023GL107938). ML trained on analyses supplies the reference state WITHOUT future observations, enabling real-time impact-based selection. HIGH — closest precedent for "sensitivity without future truth"; learns only the reference (implicitly e0), observations not weight edits, toy system, no bank/budget/interactions.
- Adjoint sensitivity to DA/model-error parameters: Daescu & Todling 2010 (QJRMS 136, DOI 10.1002/qj.693); Shaw & Daescu 2017 (JCP, DOI 10.1016/j.jcp.2017.04.050); **EFSR** Hotta, Kalnay, Ota, Miyoshi 2017 (MWR 145(12), DOI 10.1175/MWR-D-17-0122.1). HIGH on "forecast sensitivity to parameters" — diagnostic, not amortized, no competing-edit Gram.
- **Using DA tools to dissect GraphDOP** | 2510.27388 | 2025-10-31 | Laloyaux et al. (ECMWF). FSOI on an ML model via autodiff. MEDIUM.
- **Sparse Sensor Placement for Reducing Forecast Errors in EnKF** | 2606.27267 | 2026-06-25 | Takumi Saito, Shunji Kotsuki. Budgeted greedy selection maximizing predicted forecast-error reduction with information matrices (A/D/E-optimality). MEDIUM-HIGH.
- **Atmospheric Predictability Beyond 30 Days with ML** | 2504.20238 | 2025-04-28 | P. Trent Vonich, Gregory J. Hakim. Oracle IC optimization through GraphCast against known future; real-time identification left open. HIGH on oracle, zero on amortization.
- Arnoldi singular vectors 2506.22450; CNOP in FuXi 2603.26165; SPW 2609.08412 (MEDIUM on weight-space intervention premise).
- **Green's-function calibration**: Menemenlis, Fukumori, Lee 2005 (MWR 133:1224, DOI 10.1175/MWR2912.1); Strobach et al. 2022 (GMD 15:2309, DOI 10.5194/gmd-15-2309-2022). Perturb each parameter, run forward, assemble finite-difference kernel G, solve weighted least squares (G^T W G, G^T W (y - model)). HIGH — this IS EarthDelta's offline teacher (R, H = R^T Q R, b = R^T Q e), published 2005. Not flow-conditioned, not amortized, not per-forecast, no budget.

## 3. Learned surrogates of the effect of parameter/weight changes
- **Black Box Causal Inference** 2503.05985 (Bynum et al.) — amortized effect estimation. MEDIUM.
- **HyperSteer** 2506.03292 (Sun et al., Stanford) — hypernetwork emits steering vectors; predicts the intervention, not its consequence. MEDIUM. **HyperTransport** 2605.08254. LOW-MEDIUM.
- **FFORMPP** 1908.11500 (Talagala, Li, Kang) — predict forecast error from history features, select model/combination by minimum predicted error. MEDIUM-HIGH (controller in miniature).
- HRRR error prediction 2512.14898 / 2606.19026 (Evans). MEDIUM-HIGH.
- Transferability estimation: MetaRank 2511.21007; implicit modeling 2510.23145; topology-driven 2602.23916; Evidence > Intuition 2210.11255. MEDIUM.

## 4. Functional / response-weighted metrics
- EWC 1612.00796; Fisher merging 2111.09832; **OBC** 2208.11580 and **GPTQ** 2210.17323 (minimize dW^T (X X^T) dW — Gram-of-activations-weighted parameter distance, tightest analogue); RegMean 2212.09849 (RegMean++ 2508.03121); uncertainty-based gradient matching merging 2310.12808. HIGH on the FORM of (a-a*)^T H (a-a*).
- Terminology: "response distillation" already denotes logit distillation in class-incremental detection (Elastic Response Distillation 2204.02136; Refined Response Distillation 2305.00620) — rename. Because du(a) = R a is linear, (a-a*)^T H (a-a*) = ||du(a) - du(a*)||_Q^2 exactly: it is output-space squared error in coefficient coordinates (generalized Gauss–Newton / Fisher metric for a linear parameterization), not a curvature approximation.

## 5. Amortized optimization & learned planners
- **Tutorial on amortized optimization** 2202.00665 (Brandon Amos). EarthDelta's students are fully-amortized (predict a*) vs semi-amortized (predict (b,H), solve a tiny QP). Code https://github.com/facebookresearch/amortized-optimization-tutorial .
- Learning to warm-start fixed-point algorithms 2309.07835 (Sambharya, Hall, Amos, Stellato). MEDIUM. OSQP 1711.08013 (tooling).

## 6. Control-theoretic framing / adapter banks as actuators
- **Learning Options for Compositional Motor Control with Adapter Banks** | 2609.17042 | 2026-09-15 | Sreejan Kumar, Marcelo Mattar, Lea Duncker. Frozen recurrent core + bank of residual adapters (emergent low-rank perturbations) + high-level policy over options with the network frozen. HIGH on architecture (frozen backbone + bank of low-rank edits + selecting controller), LOW on mechanism (RL return, no response prediction, no e0, no H, no budget). Must cite.
- **LORAUTER** 2601.21795 (routing by task representations). MEDIUM-HIGH.
- **Chance-constrained selection of sequential interventions from counterfactual estimates** | 2608.13209 | 2026-08-13 | Minkyoung Kim. Budgeted selection from predicted counterfactual outcomes. MEDIUM-HIGH.
- WeatherPEFT 2509.22020; MENA LoRA 2409.07585 (baselines).

## 7. Model merging / task vectors with curvature
- Task arithmetic 2212.04089 (code mlfoundations/task_vectors); TIES 2306.01708 (code prateeky2806/ties-merging); task vectors and gradients 2508.16082; high-dimensional sparse disentanglement 2608.25354. MEDIUM-HIGH on edit interference; EarthDelta's interference is measured in response space over a rollout and predicted at issue time.

## (a) Five closest prior works
1. Green's-function calibration (Menemenlis et al. 2005; Strobach et al. 2022) — EarthDelta adds: response of a multi-step neural rollout w.r.t. adapter coefficients; per-forecast flow-dependent solve; box/budget best-subset; amortized student predicting (b,H) from legal history.
2. Honda & Yamazaki 2024 (GRL) — EarthDelta adds: learned du per candidate; weight edits not observations; bank with interactions and budget; nonlinear verification; modern MLWM.
3. PEFSO 2609.12296 + EFSO/PQC lineage — EarthDelta adds: du and e0 learned from history (not tangent-linear; not capped by the ~2-day window); discrete edit bank with budget; learned off-diagonal H.
4. Adapter Banks 2609.17042 — EarthDelta adds: explicit predicted quadratic gain rather than RL return; offline QP oracle with finite differences and verification; interaction-aware planning; physical error norm.
5. PEAR 2605.01361 (+ Fonseca 2606.27462) — EarthDelta applies the projection principle to choosing interventions at inference; the decision-relevant subspace (span of edit responses) is itself predicted.

## (b) Precedent for the identity and the decomposition
- Identity: YES, standard in FSO/EFSO since Langland & Baker 2004 and in the Kalnay et al. 2012 form Δe² = (e_a − e_b)^T C (e_a + e_b); with e_b = e0 and e_a = e0 − du this is exactly −(2<e0,du>_C − ||du||_C^2). Also the one-step gain of greedy least squares / matching pursuit, and the Green's-function normal equations. Do not claim as novel.
- Decomposition "predict e0 and du separately from legal history, then combine": NO explicit precedent found. Halves exist separately: e0 (Honda & Yamazaki 2024; Evans 2512.14898/2606.19026; FFORMPP 1908.11500); du computed (EFSO/PEFSO; Green's functions) but never learned from history. In all FSO/EFSO work e0 comes from a verifying analysis (post-hoc diagnostic).

## (c) Name for (a−a*)^T H (a−a*)
"Response distillation" is taken (logit distillation). The object is output-space squared error in coefficient coordinates = generalized Gauss–Newton / Fisher metric; precedents OBC, GPTQ, RegMean, Fisher merging, EWC, OBD/OBS lineage. Claim only that the responses over a forecast rollout, and their prediction from history, are new.

## (d) Repos
FinJun/PEAR; khalil-research/PyEPO; cvxgrp/cvxpylayers; locuslab/qpth (+ optnet); facebookresearch/amortized-optimization-tutorial; prateeky2806/ties-merging; mlfoundations/task_vectors; arcee-ai/mergekit; google-deepmind/graphcast. Not found: code for 2607.00581, SPW URL (note: MeteoSwiss/ai-models-ensembles per survey C), PEFSO, Honda & Yamazaki 2024, GMD 2022 Green's-function calibration.

## Bottom line
- Not novel (cite aggressively): the gain identity (FSO/EFSO); the offline oracle R -> b,H -> box-LSQ -> verify (Green's-function calibration); the (a−a*)^T H (a−a*) metric form (GGN/Fisher; OBC/GPTQ/RegMean/EWC); frozen backbone + low-rank adapter bank + selecting controller (2609.17042); differentiable QP layers; the name "response distillation".
- Unoccupied on ~70 searches: learning BOTH e0 and per-edit du from legal history at issue time and combining them through the exact quadratic to choose parameter-space edits under a budget with a learned off-diagonal interaction matrix.
- Sharpened claim: "amortizing forecast-sensitivity-to-parameter-edits into an issue-time controller", positioned against adjoint parameter sensitivity (Daescu & Todling 2010; Shaw & Daescu 2017; diagnostic) and Green's-function calibration (offline, global, non-amortized).


==================================================================
===== FILE: literature/survey_E_tooling_backbones_data.md
==================================================================

# Survey E — engineering landscape: backbones, tooling, data access (verified 2026-09-19)

Source: background tooling agent, 2026-09-19; verified against local clones under `/mnt/afs/260010168/EarthDelta/reference/`, hf-mirror API, PyPI, and live GET probes. Items marked UNVERIFIED were not confirmed.

## 1. Stormer (tung-nd/stormer, arXiv 2312.03876)
- Repo frozen: `main` HEAD `58dfee5a6037399a40fefd492bc00421e0c885a8` (2025-03-17, "update readme"); no tags, no newer commits; 9 forks, none removes the xformers dependency (`stormer/models/hub/stormer.py:4`).
- Checkpoints on HF `tungnd/stormer` (HF sha `2ceebb74…`, 2024-10-23, license MIT): `stormer_1.40625_patch_size_2.ckpt` 5,625,590,679 B; `stormer_1.40625_patch_size_4.ckpt` 5,570,407,547 B. Lightning `.ckpt` (loaded via `checkpoint["state_dict"]`), size includes optimizer state (~400 M params model). `inference.py` defaults to patch_size_2. Mirror: `https://hf-mirror.com/tungnd/stormer/resolve/main/<file>`.
- 69 channels = 4 surface (t2m, u10, v10, mslp) + 5 vars × 13 levels; grid 128×256 = pole-free cell centres (`regrid_wb2.py:65-68`: lat −89.296875…+89.296875, lon 0…358.59375).
- Normalization: 10 npz files in `normalization_constants/` (109 keys each, 1979–2018). `inference.py` expects copies inside the preprocessed h5 dir.
- Architecture: DiT-style; per-variable `PatchEmbed` + variable-aggregation `nn.MultiheadAttention`; 24 blocks with `MemEffAttention` (`qkv` d→3d, `proj` d→d), timm `Mlp` (fc1/fc2), `adaLN_modulation` (SiLU + Linear d→6d); `FinalLayer` (Linear + adaLN 2d). hidden 1024, depth 24, heads 16. All adapter targets are plain `nn.Linear`.
- env.yml: python 3.11.5, torch 2.1.0 cu118, xformers 0.0.22.post7+cu118, timm 0.9.2, lightning 2.2.1. SDPA swap needs layout change (xformers (B,N,H,D) vs SDPA (B,H,N,D)).
- No Stormer-preprocessed ERA5 on HF (`tungnd/*` has only models, `stormer_forecasts`, IndiaWeatherBench).

## 2. Aurora (microsoft/aurora)
- Local HEAD `4765abc` (2026-09-14). HF `microsoft/aurora` (MIT code + weights; commercial use: contact Microsoft): 9 checkpoints incl. `aurora-0.25-pretrained.ckpt` 4.68 GiB, `aurora-0.25-small-pretrained.ckpt` 0.42 GiB, `aurora-0.25-v1.5.ckpt`, `-v1.5-ensemble`, `aurora-0.1-finetuned`, `-air-pollution`, `-wave`, `-12h-pretrained`; loader `Aurora.load_checkpoint(repo, name, revision, strict=True)`.
- GPU: ~40 GB for 0.25° inference (Aurora 1.5 ~32 GB); fine-tuning tested on 80 GB A100.
- LoRA built in: `aurora/model/lora.py` (`LoRA`: A kaiming, B zero, scaling alpha/r; `LoRARollout(max_steps=40, mode∈{single, from_second, all})`), applied only to Swin3D attention `qkv` and `proj` (`swin3d.py:130-139, 159, 182`); step index carried on `Batch.metadata.rollout_step` (`batch.py:43`, threaded at `aurora.py:435`). Defaults r=8, alpha=8, dropout 0, steps 40, mode single. MLPs, Perceiver encoder/decoder, patch embeddings NOT adapted. Base `Aurora` use_lora=True; `AuroraPretrained`/`Small`/`12h`/`V1p5` use_lora=False; `AirPollution`/`Wave` mode from_second.
- No coarse-resolution Aurora checkpoint exists → not a practical second backbone on limited GPU.

## 3. Other open-weight coarse backbones (PyTorch-friendliness)
| Model | Repo / weights | Grid | Licenses | Notes |
|---|---|---|---|---|
| ClimaX | microsoft/ClimaX `6d5d354` (2023-09-30); HF `tungnd/climax` `1.40625deg.ckpt` 443 MB, `5.625deg.ckpt` 432 MB | 128×256 and 32×64 | MIT / MIT | same grid as Stormer; timm ViT; patch 4 for [128,256] |
| ArchesWeather / Gen | INRIA/geoarches `b2bdb07` (2026-09-18); HF `gcouairon/ArchesWeather` `archesweather-m-seed0_checkpoint.ckpt` 340 MB, `archesweathergen_checkpoint.ckpt` 1.9 GB | 1.5°, 13×121×240 | BSD-3 / BSD-3 | `nn.Linear`-dense (qkv, proj, fc1/fc2, adaLN); on PyPI; **preferred second backbone** |
| ACE2 | ai2cm/ace `b648495`; HF `allenai/ACE2-ERA5` 1.8 GB | ~1° Gaussian 180×360 | Apache-2.0 / Apache-2.0 | SFNO MLPs are Conv2d(1×1), adapter needs Conv variant |
| Prithvi-WxC | NASA-IMPACT/Prithvi-WxC; HF 28.4 GB | MERRA-2 0.5°×0.625°, 2.3 B params | MIT / CDLA-Permissive-2.0 | too large |
| SFNO/FCNv2, FourCastNet 3 | NVIDIA/makani, earth2studio; NGC / HF `nvidia/fourcastnet3` 2.8 GB | 0.25° | Apache-2.0 | needs torch-harmonics; FCN3 needs 80 GB |
| Pangu-Weather | 198808xc/Pangu-Weather; ONNX ~1.1 GB | 0.25° | none / CC BY-NC-SA 4.0 | ONNX only |
| FuXi | tpys/FuXi; Zenodo 8.6 GB ONNX; HF `tpys/fuxi-2.1` PT2 3.9 GB | 0.25° | none / CC-BY-4.0 | frozen graphs |
| GraphCast small | google-deepmind/graphcast → redirects to google-deepmind/weathernext `f2f2c51`; `gs://dm_graphcast/params/GraphCast_small…npz` 144 MB | 1°, 13 levels | Apache-2.0 / **CC BY 4.0 since 2026-08-06** | JAX/Haiku |
| NeuralGCM | neuralgcm/neuralgcm; `gs://neuralgcm/models/v1/deterministic_{2_8,1_4,0_7}_deg.pkl` | 2.8/1.4/0.7° | Apache-2.0 / CC BY-SA 4.0 | JAX |
| AIFS-single | ecmwf/anemoi-core; HF `ecmwf/aifs-single-1.0` 0.99 GB | n320 (~0.25°) data grid | Apache-2.0 / CC BY 4.0 | PyTorch, heavy |
| WeatherMesh | windborne/weathermesh-3 | — | Apache-2.0 | no weights released |
Recommendation: ArchesWeather-M (1.5°, BSD-3 both) first, ClimaX 1.40625° (MIT, identical grid) second; ACE2 runner-up.

## 4. WeatherBench-X and WB2 data
- weatherbenchX HEAD `964a35e` (2026-09-17), Apache-2.0; **git-install only** (`pip install git+https://github.com/google-research/weatherbenchX.git`); deps include `apache_beam[gcp]`, `xarray>=2025.7`, `zarr`, `gcsfs`, `xarray-beam`, `jax[cpu]`, `arch`, `numpy>=2.1.3`.
- APIs: `data_loaders/xarray_loaders.py` (`PredictionsFromXarray` L187, `TargetsFromXarray` L236, `ClimatologyFromXarray`, `PersistenceFromXarray`; `rename_dimensions='ecmwf'` maps time→init_time, prediction_timedelta→lead_time / time→valid_time); `weighting.py` `GridAreaWeighting` (L92, grid-agnostic, clamps outer bounds at ±90°); `metrics/deterministic.py` `RMSE` L312, `MSE`, `ACC` L374, `WindVectorRMSE`; `aggregation.py` `Aggregator(reduce_dims, weigh_by, bin_by)` L269; `time_chunks.py` `TimeChunks`; `beam_pipeline.py` `define_pipeline`; `binning.py` `Regions`, `LatitudeBins`, `LandSea`; `interpolations.py`; `metrics/spatial.py` FSS; `statistical_inference/` bootstrap/t-test. Runner: `--runner=DirectRunner` locally.
- 1.40625° / pole-free 128-lat grids handled correctly; predictions and targets must share a grid (no auto-regrid).
- WB2 GCS zarr paths: `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr`; `…/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr` (1.5°); `…/1959-2022-6h-64x32_equiangular_conservative.zarr`; `…/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr`; climatology `gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr`. Stormer's `download_wb2.py` reads `gs://weatherbench2/datasets/era5/` + file. GCS bulk transfer from this machine: 0.1–0.5 MB/s (unusable).

## 5. torch-harmonics
- Local HEAD `4ac8ed3` (2026-09-18) = `v0.9.3b1 (unreleased)`; PyPI 0.9.2 (2026-08-12), BSD-3.
- API: `RealSHT(nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True)`, `InverseRealSHT`, `RealVectorSHT`, `InverseRealVectorSHT`. Grids: `equiangular` (Clenshaw–Curtis), `equiangular-trapezoidal`, `legendre-gauss`, `lobatto`.
- `precompute_latitudes` returns **colatitudes in radians** (float64, lru_cached): `lats = flip(arccos(xlg))`.
- **Equiangular includes both poles**: nlat=128 → colat 0°, 1.4173°, …, 180° (spacing 180/127 = 1.4173°), NOT Stormer's pole-free 1.40625° cell centres. No pole-free equiangular option; `legendre-gauss` is pole-free but non-uniform. Longitudes match exactly. Mitigations: interpolate to a 129-lat pole-inclusive grid, or conservative regrid to legendre-gauss, or accept quadrature error (document it).
- v0.9.3b1 changes: theta_cutoff defaults derived from grid spacing; bilinear-spherical resampling fix; legpoly tables re-keyed on (nlat, grid); trapezoidal weights float64.
- Bare source tree does not import (`attention_helpers` extension); use `pip install -e .` or PyPI.

## 6. flash-linear-attention
- Local HEAD `864a87f` (`__version__` 0.6.0 dev); PyPI `flash-linear-attention` / `fla-core` 0.5.2 (2026-07-27), MIT.
- `fla/layers/gated_deltanet.py` `class GatedDeltaNet` (L30); kernels `fla/ops/gated_delta_rule/{chunk,fused_recurrent,naive,…}.py`. Pure-torch `naive_recurrent_gated_delta_rule` / `naive_chunk_gated_delta_rule` exist (fp32, Python loop over T); the `GatedDeltaNet` layer itself requires Triton (`mode in ['chunk','fused_recurrent']`, `ShortConvolution`, `FusedRMSNormGated`). Since v0.5 torch/triton are not base deps; extras `[cuda]` (torch>=2.7, triton>=3.3), `[cpu]`, etc.

## 7. Neighbour-method code
- CLAW 2609.12278: no code. CCM-LoRA: no arXiv record and no GitHub repo found (only the ACL Anthology entry from survey B). Spectral-Target JEPA 2609.04264: no code. EPM-JEPA 2606.12979: no code. **Flow-JEPA 2608.29029: code at https://github.com/HuoYanchen/Flow-JEPA (main `e9172e7`, 2026-08-27, MIT; LICENSE header still credits Lucas Maes).**
- Already-cloned neighbours (HEADs match remote): CoMoL `011306a` (no license); DISeL `b6e543a` (Apache-2.0; paper "coming soon"); PEAR `bb00e7b` (MIT, needs Gurobi); geps `e9a8652` (no license); Solver-in-the-Loop `f514fcf` (MIT); sg-jepa `1b7794b` (MIT; no arXiv ID, self-cites as software); le-wm `8edfeb3` (MIT); StreamTTT `96aed21` (Apache-2.0, default branch master); csubich/graphcast branch `amse` `6ed80d8` (default branch is `graphcast_train`, so pin `amse` explicitly); peft `50a277e` (0.21.1.dev0).

## 8. ERA5 access from this cluster (no GitHub raw, no usable GCS)
Throughput: ModelScope ~7.2 MB/s > hf-mirror ~2.5 MB/s >> GCS 0.1–0.5 MB/s. Reachable: CDS, TUM, AWS `s3://nsf-ncar-era5` (anonymous listing works), Planetary Computer.
1. **hf-mirror datasets**: `JleeOfficial/ERA5-240x121-1979-2018` — 591.43 GB, 1.5° (240×121, WB2 grid), 6-hourly, 13 pressure levels, years 1979–2019, 13 variables + constants/norm stats, npy `(1460,240,121)` / `(1460,13,240,121)`; **no license declared**. `JleeOfficial/ERA5-64x32-1979-2015` — 38.66 GB, 5.625°, same layout. `thainamhoang/era5-climate-learn` (cc-by-4.0) 1.40625° but only 3 surface variables. `jasonjewik/climate-learn` (cc-by-4.0) 5.625°/2.8125° multi-variable shards. `TornikeO/era5-5.625deg` raw WB1 nc, few variables.
2. **TUM WeatherBench-1**: live; download needs a real GET (HEAD/PROPFIND return 401): `wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F1.40625deg%2Fgeopotential&files=geopotential_1.40625deg.zip"`. 1.40625° sizes: t2m 35.5 GB, z 344 GB, t 435 GB, q 370 GB (hourly, 1979–2018). 5.625° `all_5.625deg.zip` 267.66 GB. **No mean_sea_level_pressure in WB1** → insufficient for Stormer's 69 channels.
3. **ModelScope**: `OneScience/ERA5` CC-BY-4.0, 0.25°, 243 vars, `data/{year}.h5` 5.05 GB each — but each file holds only T=5 time steps (verified separately: sample set). `LwojvzeL/ERA5_5p625` partial (3 surface vars). `zhangminglang/ERA5_1p5deg_V2` only t/z. `hhs2000/WeatherBench` empty.
4. Others: CDS (needs account; `cdsapi` 0.7.7 on Tsinghua mirror); ECMWF open data = forecasts only; AWS `s3://era5-pds` AccessDenied; AWS `s3://nsf-ncar-era5` anonymous OK (native 0.25° GRIB/netCDF); Planetary Computer STAC `era5-pds`; OpenDataLab / BAAI UNVERIFIED.
Recommendation: (a) `JleeOfficial/ERA5-240x121-1979-2018` via hf-mirror + Stormer's `regrid_wb2.py` (the authors' own path from WB2 240×121), subject to variable-set check and license caveat; (b) TUM 1.40625° only for diagnostics; (c) `JleeOfficial/ERA5-64x32` for quick iteration; (d) CDS or `nsf-ncar-era5` for faithful 0.25° subsets.

## Things that change the plan
1. torch-harmonics cannot represent the 128×256 pole-free grid exactly; choose a mitigation before spectral diagnostics.
2. Stormer requires xformers; budget an SDPA rewrite with layout change.
3. Aurora's LoRA touches only Swin attention qkv/proj; its per-step `lora_mode="all"` driven by `Batch.metadata.rollout_step` is the design to compare against. Aurora is not usable as a coarse second backbone.
4. WeatherBench-X is git-install only with heavy GCS/beam deps; handles 128×256 correctly.
5. No public 1.40625° ERA5 with all 69 variables; realistic path = 1.5° mirror + `regrid_wb2.py`.
6. GraphCast weights now CC BY 4.0; `google-deepmind/graphcast` redirects to `weathernext`.


==================================================================
===== FILE: extracted/v5_Research_Kit/EarthDelta_v5_kit/IMPLEMENTATION_PLAN_CN.md
==================================================================

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


==================================================================
===== FILE: extracted/ResponseKit/EarthDelta_ResponseKit_20260917/CODEX_PLAN_CN.md
==================================================================

# EarthDelta：响应交互驱动的轨迹修正——研究与代码实现方案

日期：2026-09-17。本文继续此前的响应空间路线，不再另起一个大型框架。

**状态：研究设计 + 已完成的基础代码原型。不是已经训练成功的天气方法，也没有证明首创性。**

## 0. 本轮决定

保持冻结天气模型、非零低秩修正字典和显式多步rollout。将“响应驱动”具体化为两件可测量的事情：

1. 修正的未来收益：给定当前合法历史，哪次修正可能改善后续多个时效？
2. 修正之间的响应重叠：两次修正是否重复、抵消、互补？预算有限时，不应只按单项分数各自取最大。

推荐工作表述：**EarthDelta: Interaction-Aware Forecast Trajectory Repair with Amortized Response Models**。

主贡献不是发明LoRA、Gram矩阵、Gauss–Newton或稀疏QP，而是研究：在冻结天气模型上，能否用受控的训练侧响应实验学习可部署的修正价值/组合模型，使少量修正比普通条件适配和同预算多步训练更有效。

两个学生路径共用同一基础，不串成更深系统：
- 最小验证路径：历史 -> 直接程序头 -> 有界修正；以同教师的系数/响应蒸馏比较验证响应监督。
- 拟议主方法路径：历史 -> 预测b/H -> 小型受约束规划器 -> 有界修正；以完整H/对角H比较验证修正交互。

若完整H并不比对角H或直接程序头更有效，则删掉交互复杂度。暂不开展跨backbone响应迁移、长期记忆、TTT、动态rank、额外JEPA或重型谱损失。

## 1. 研究近邻与复用分工

详细链接、日期和实际阅读范围见随包SOURCES.md。

| 来源 | 已有内容 | 本项目需要额外证明的内容 | 代码使用方式 |
|---|---|---|---|
| CLAW，2026-09-10 [S1] | 近期转换到hypernetwork生成LoRA的世界模型适配 | 不仅把历史变成参数；修正价值、组合与时机是否有增量 | 未确认可直接复用官方代码，不设为前置依赖 |
| PEFSO，2026-09-10 [S2] | 观测敏感度近似预报影响、减少重积分 | 内部参数修正的可部署选择；与观测同化任务分开 | 参考近似有效范围和非线性核验，不直接迁移气象结论 |
| PEAR，v2 2026-05-19 [S3] | 决策相关投影误差与约束/曲率 | 响应损失是否比系数误差有用，不能移植理论保证 | 作者仓库链接已找到，未逐文件审查，不作为已接入组件 |
| CoMoL，v2 2026-06-04 [S4] | 共享A/B与小core空间动态混合 | 不把便宜的动态混合作为主创新 | layer.py实际读取；可作第二阶段高效实现和近邻基线 |
| DISeL [S5] | 输入相关低秩门控 | 排除普通动态router的解释 | 参考门控与保存逻辑；保持移植/官方名称边界 |
| Solver-in-the-Loop [S6] | 修正后状态分布和多步训练 | 排除“多步训练本来就能学会”的解释 | 借鉴实验协议，不引入旧TF/PhiFlow环境 |
| GEPS [S7] | 物理系统上下文低秩适配 | 受控方程中的修正方向/时机机制 | 参考PDE生成器，独立审查与数值收敛测试 |
| Stormer [C1] | 完整天气预测底座 | 只实现新修正接口，不重写气象流水线 | 原权重、数据约定、增量还原直接复用 |
| WeatherBench-X [C2] | 权重/区域/指标/聚合与数据装载 | 自己只补修正收益、成本和配对统计 | 标准预测导出后独立评估，可本地运行 |

这是定向查重，不是穷尽式首创证明。仅改变领域、标题或加入多个现成模块不能建立novelty。

## 2. 定义研究对象，防止信息条件混淆

起报合法信息I_t包括X[t-12h:t]、静态场、已知步长与预算。模型输入保持Stormer原生变量；未来真值Y只进入训练标签/离线评估。

固定backbone参数theta0和已训练的非零bank psi。程序a属于R^d，描述预先定义的层/时间/区域槽位的系数。首版a0=0表示完全不执行该bank。

G_x(a) = D(Rollout(F_theta0,psi, I_t, a))。

D固定为多变量、多时效、多尺度的摘要或下采样向量；不是只优化最后一天，也不按测试误差挑区域。Q定义面积、变量尺度、时效等非负权重，各项只应用一次。

所有推理路径必须分开：
- 原模型基线：固定6h路线；不额外平均6/12/24h路线。
- 开环学生：起报时生成整段程序，后续只读自身预测。
- 预测前缀重规划：可以基于自己产生的前缀重新决策，但需要新的训练协议和计算账本。
- 重新获得真实分析场的滚动更新：另一任务，不混入开环成绩。

训练阶段使用ERA5的合法时间顺序，不等于证明业务时刻能够取得所有同化后的分析场；不要把回顾性ERA5设置称作已部署实时预报。

## 3. 低维响应与修正交互

在a0附近用中心差分求响应列：

R_j = (G_x(a0+epsilon_j e_j)-G_x(a0-epsilon_j e_j))/(2epsilon_j)。

e = D(Y)-G_x(a0)，b = R^T Q e，H = R^T Q R。

局部平方误差收益：

Gain(a) ≈ 2b^T a - a^T H a。

本文首版a0=0，a既是offset也是绝对系数。非零a0时公式用于delta=a-a0，必须同时约束a0+delta；原型盒约束只约束offset，不能静默改协议。

H_ii：当前单位修正响应的加权大小；b_i：该响应和训练残差的对齐；H_ij：两个响应重叠。符号与所选系数共同决定交互，不能笼统把所有非对角项解释成抑制。

重要边界：
- 这是线性化预测映射下的二次模型，不是新数学结论。
- H不等于真实非线性损失Hessian；完整二阶项还涉及残差加权的G二阶导数。
- 两列正交/相关是模型输出空间关系，不是大气过程独立/因果关系。
- d=8自然导致rank(R)<=8，不能据此宣布发现天气纠错本质低维。
- 响应R不需要未来真值，收益b需要；在线计算R也不能绕过未来未知的问题。

## 4. 首版程序：先让总d=8真的可执行

建议最小配置：4个已核查的后部层[18,20,22,23]，各一个rank16低秩方向，作用时刻为预测步0和2（6h固定步长下为第1次与第3次调用），全域支持，共8个标量自由度。

这一首版把where暂时限定为网络层，不宣称已经学会地理区域选择。地域支持扩展在第二阶段单列；不可同时把每patch8维展开成巨大程序。

为了让每个槽位成本清晰，初版可以将旧StateGatedLinear的groups设为1；rank仍为16。旧groups4的实验保留为另一配置，不覆盖旧结果。每个(layer,step)最多一项整分支调用，后续真正跳过该分支才有实际成本意义。

组件边界候选rho=0.25仅为原型默认值，不是气象最优值；在训练/验证侧结合已训练bank的函数输出尺度、epsilon稳定性选择并冻结。不得比较不同bank任意缩放下的系数幅度并称为公平。

## 5. 字典训练与冻结

最小策略：相同数据、相同多步规则先训练静态LoRA得到非零A/B。保留该静态模型为基线；响应与学生阶段固定bank，不更改基础模型或bank。

B零初始化能够保留基模型，但此时对系数的响应也是零。因此响应采集的前置条件是非零bank、有效输出变化及可重复前向。

已有旧层内部router是2*sigmoid，不能自然表示负系数。新学生使用有界tanh，经coefficients显式接口绕过旧router；教师和学生同域。

扩展时可用CoMoL的共享A/B与core C_k，使不同方向只增加小矩阵。但首轮保持旧bank，不同时更换字典和训练监督。C_k的r×r矩阵与程序H的d×d矩阵完全不同，不混同。

若在后续共同多步微调中解冻bank：停止复用旧R/H/b，或明确重新生成响应；不能以旧几何约束变化后的模型。

## 6. 离线教师：小型求解，完整非线性复验

### 6.1 连续参考先行

求解带ridge的加权最小二乘，并施加分量盒约束：

min_a ||sqrt(Q)(e-Ra)||^2 + lambda||a||^2，|a_j|<=rho_j。

实际复用SciPy lsq_linear；不要重写一个未经测试的优化器。这里是盒约束，不是旧提案中的L2信赖球；可通过局部幅度校准控制范围，但不能声称两者等价。

首先全部维度连续优化；确认教师有完整非线性收益后，再比较稀疏程序。

### 6.2 预算支持枚举

若总d=8，每次最多2个单位成本槽位，支持数为1+8+28=37，包含无修正。36个非空支持分别求小问题。这只是计算不同系数组合，不代表需要重新运行天气模型36次。

cost目前为声明的加性单位成本；真实运行必须替换为实际执行组成本。共用A/B或同一层/步多个方向的成本有共享，不能逐系数重复收费。

### 6.3 非线性核验

用共同未压缩天气指标重跑排名靠前的少数候选，例如3个，并保留无修正。摘要收益高但全场恶化时拒绝；求解失败、非有限输出、无收益都保留在清单。

真值协助的接受/拒绝只用于训练教师或离线诊断，不能用作部署时的“安全开关”。最多测试了几个候选的最佳结果称为“已测试候选参考”，不是全局oracle；学生超出这一有限候选参考并不矛盾。

### 6.4 探针预算与缓存

d=8中心差分17条完整轨迹；默认重复基线测试再加1，实际18。每条24h包含4次6h模型调用，因此完整从头重跑时为72次单步调用；top3复验的独立实现另外重跑基线+3=4条轨迹，即16次，合计22条/88次，不含epsilon与混合方向诊断。

当前原型重复基线而非自动复用。可以在冻结快照验证后复用基线，但必须如实更新成本；不能用17个单步描述17条轨迹。

先流式生成R与教师，再只保留训练需要的H,b、候选标签、标识和少量诊断R。按d8与m约几万，存所有样本R会明显放大磁盘；不要默认缓存全层激活或全部天气分支。

缓存至少绑定checkpoint、bank、输入样本与预处理、program spec、reference、D/Q、epsilon、dtype/backend；bank或任何科学字段变化即失效。

## 7. 学生：先直接蒸馏，再验证交互模型

### 7.1 直接程序头（快速路径）

旧TemporalPatchState产生z[B,N,128]，按真实面积加权形成z_pool[B,128]，BoundedProgramHead输出a[B,8]。比较同架构同教师：

L_coef = ||a-a*||^2；
L_resp = (a-a*)^T H (a-a*)。

教师系数可能不唯一，响应等效不应强制逐系数一样。仍需保留小型系数正则限制响应零空间中的大系数，以及共同多步预测损失。对H按训练数据固定尺度归一化，避免不同天气过程仅因原始误差量纲支配学习。

### 7.2 交互收益头（拟议主方法）

同样z预测b_hat[B,8]和H_hat[B,8,8]。使用L的36个下三角参数，H_hat=L L^T+eta I，总输出44个标量。softplus正对角只是保证PSD，不是准确性证明。

初步监督：b标签、H标签、在固定有界候选集合上的收益回归/排序。b/H标签生成仅在训练侧。更重要的是用已经非线性核验的候选实际收益做独立选择质量诊断，避免H拟合看起来好但决策差。

候选非线性收益并非总能由PSD二次模型表达。先测近似误差；不立即再叠大型score network。若线性区间不成立，缩小范围、缩短局部作用窗或回退直接多步控制器。

### 7.3 在线选择

plan_from_prediction只接受预测b/H、幅度与预算，枚举允许支持并求小型盒约束QP。程序实现复用SciPy L-BFGS-B及解析梯度。没有通过优化器反向传播，也没有在线调用教师或真实未来。

d8、max_active2最多36次很小的求解，但是否相对天气模型足够便宜必须实测。对batch可先逐样本CPU选择；不在起步阶段引入可微QP/强化学习/大型分布式搜索。

无修正候选始终存在。门槛可以在验证侧校准，报告误修正率与覆盖率，不宣称严格的未来风险保证。

### 7.4 开环与重规划

原型是起报时预提交程序。若升级成每步控制：状态必须包括自己预测的历史、剩余预算、已执行支持与时效；禁止按日真值补充。收集学生产生的前缀重新建立相应训练响应，或统一多步训练。基于真实分析场建立的缓存标签不能直接用到不同预测前缀。

## 8. Stormer代码接入：复用什么、必须改什么

保持上游提交58dfee5a6037399a40fefd492bc00421e0c885a8。本轮重新核对远端main一致。[C1]

### 8.1 加载与前向

原样加载checkpoint -> 完整校验参数名/shape -> 冻结backbone -> 插入模块。禁止插入后重新initialize_weights或用strict=False吞掉原模型未加载字段。

首版原生69变量、128×256、匹配patch4权重；确认权重来源与数据经历。展示7个变量不意味着只输入7通道。保持归一化增量还原、零增量均值和interval=hours/10。

`stormer/models/hub/stormer.py`：
- Stormer.forward接收新增显式program/step参数；无程序时原路径不变。
- Block.forward继续保留原time_interval adaLN；向Attention传递必要的系数/执行标识。
- MemEffAttention使用fused attn.qkv；适配目标为选定attn.proj，不套用LLM q_proj/v_proj名称。
- StateGatedLinear接收coefficients，不隐式读取全局变量或可变hook状态。

### 8.2 roll-out桥接

`stormer/models/iterative_module.py`中的padding、增量反归一化、加回当前物理状态与下一步归一化直接复用，封装为显式WeatherStepBridge。

未来目标只传给训练loss/evaluator，不是ControlledRollout参数。建议接口：

```python
# 待实现的天气桥接API，不是当前包已提供的函数
trajectory = controlled_rollout(
    bridge, initial_history, program_spec, coefficients,
    horizon_steps=4, return_leads=(1, 2, 4),
)
# labels 不允许出现在这一路参数中
```

每次中心差分从相同初始快照启动，不能接着前一分支的history/controller状态跑。除输出外，不修改输入tensor、随机状态约定或隐藏账本。

### 8.3 真正跳过分支

当前旧层即使coefficients全零仍计算A/B。新的执行器在无激活程序时调用base投影而不调用A/B。若batch只有部分样本激活，先选择同支持microbatch或显式gather/scatter；CPU/GPU同步与散射开销要计入。

训练阶段全部系数初始零时，不能沿硬跳过路径切掉学生梯度；训练用dense可微路径，正式推理用显式选定支持路径，做数值等价和实际算子计数测试。

### 8.4 梯度与缓存

冻结权重不等于整个forward可no_grad。第一处adapter前的纯冻结前缀在特定单步原始输入条件下可缓存；多步全反传时下一步输入依赖上一轮适配，需要保持相应梯度。截断策略应让所有基线一致。

使用checkpoint重算时必须显式传入当前step与coefficients；可变模块属性会导致重算使用错误状态。所有checkpoint保存bank、student、program和原模型identity，不复制巨大权重多份。

## 9. 标准评估复用WeatherBench-X

官方示例实际使用[C2]：
- xarray_loaders.PredictionsFromXarray / TargetsFromXarray；
- deterministic.RMSE() / MSE()；
- weighting.GridAreaWeighting()；
- binning.Regions；
- aggregation.Aggregator；
- 本地DirectRunner或后续分布式runner。

先把预测导出为规范xarray/Zarr：init_time、lead_time、latitude、longitude，以及对应变量的level轴。valid_time=init_time+lead_time。温度、位势、风和湿度单位与原始目标一致，不能把位势当位势高度混用。

先在小型数组上比对本地加权MSE与WB-X，再上真实数据。RMSE应对完整加权平方误差聚合后开方，不能用逐batch RMSE平均替代主指标。

自定义只补：真实净修正收益、误修正率、无修正率、候选参考差距、计量成本与配对bootstrap。WB-X不是训练loss模块，不为部署额外引入全部评估依赖。

缺测/非有限预测要保留分母与失败计数，不能各方法用不同有效样本悄悄比较。训练阈值定义的初始复杂区域与按未来结果定义的极端过程分开；不因高湿区有改善就称降水技巧改善。

## 10. 首轮实验与否证条件

共享：同checkpoint、变量、预处理、合法历史、bank容量、训练/验证/测试分期、共同多步loss、HPO预算和评估集合。

| ID | 方法 | 优先回答 |
|---|---|---|
| B0 | Frozen | 基线 |
| B1 | 静态LoRA | 普通适配是否足够 |
| B2 | 同历史temporal router + 多步loss | 普通条件适配和未来梯度是否足够 |
| B3 | 旧H+R + 同多步loss | 残差辅助监督是否足够 |
| D0 | 同教师 + 系数蒸馏 | 教师额外计算的作用 |
| D1 | 同教师 + 响应蒸馏 | 功能空间监督的必要性 |
| U0 | 预测b + 对角H规划 | 单独价值/幅度模型 |
| U1 | 预测b + 完整H规划 | 修正交互的必要性 |
| O0 | 相同历史的输出纠错器 | 内部参数修正是否必要 |

不要求首批同时跑全部九组。先共享bank做D0/D1以及用同一真值局部几何full/diag的诊断；通过后加U0/U1与完整强基线。教师参考只在独立离线表报告，不放入可部署模型主榜。

教师样本必须覆盖无收益案例，不按未来正收益筛掉困难样本。B0与B1训练不同是合理参考，但D0/D1/U0/U1必须匹配bank和信息。

统计按起报时间块/天气过程配对；不是按网格点独立bootstrap。日期外推不自动等于OOD。正式确认数据从未用于选bank、D/Q、阈值、边界、时刻、损失系数或超参数。

关键gate：
- 非线性教师无稳定全场收益 -> 不扩大学生，不继续加memory。
- R局部准确、学生差 -> 检查历史能否识别b/交互，避免归因给求解器。
- full/diag没有区别 -> 去掉交互头，保留更简单方法。
- D1与D0相当 -> 缩小“响应蒸馏独有收益”的主张。
- B2相当 -> 明确当前额外机制未证明价值。
- 摘要好全场差 -> 修正目标映射，不挑选有利变量隐藏失败。

## 11. 建议Codex任务与提交

任务序列是验收顺序，不是时间承诺。所有GPU调用和数据下载需先明确本地真实资产与预算；不得改动DisasterTrace仓库。

### R00：保存事实起点

重新运行旧21测试与本包测试；记录实际工作目录、依赖、checkpoint是否存在。两包合成测试通过不等于真实集成。

提交：`test: preserve starter and response-kit invariants`。

### R01：显式受控Stormer桥接

新增`stormer_bridge.py`、`controlled_rollout.py`，固定原归一化/变量和6h路径。原版预测与零程序逐元素容差等价；跨分支、跨batch、checkpoint重算不串state；真值无法进入rollout。

先仅用已有少量真实合法天气样例，不全量下载。

提交：`feat: add explicit controlled Stormer rollout`。

### R02：非零bank与程序支持

新增`fit_bank.py`和bank状态报告；检查B非零、冻结参数不变、selected layer名shape。让8维程序在对应层/步生效，外部系数与显式矩阵一致。

提交：`feat: freeze nonzero repair bank and eight-slot programs`。

### R03：响应采集与教师闭环

新增`collect_responses.py`、`collect_teacher.py`和最小缓存索引。调用本包central_response、box_candidates、verify_candidates，不重写求解器。

先少量跨季节起报，24h探针；epsilon/2、epsilon、2epsilon；至少混合方向检查；float64累计和dtype回执。教师完整全场收益、无收益、失败全部输出。

提交：`feat: add offline response collection and verified local teacher`。

### R04：直接学生与同教师对照

新增`train_direct_student.py`，复用旧TemporalPatchState与新BoundedProgramHead。相同数据、同教师分别系数蒸馏和响应蒸馏。真正推理函数无truth/teacher/H标签输入。

提交：`feat: train coefficient and response-distilled program heads`。

### R05：修正交互与部署规划

新增`train_utility.py`、`infer_program.py`，使用本包InteractionUtilityHead和plan_from_prediction。先验证预测b/H，再比较同预算U0/U1。用真实算子调用计数建立执行组账本，dense与sparse执行输出等价。

提交：`feat: add label-free interaction-aware repair planning`。

### R06：评估与冻结确认

新增`export_wbx.py`、`evaluate_wbx.py`、`paired_analysis.py`。6/24/72/120h全部方法共同评估，记录全部成本和失败。更长时效不能未经验证直接使用24h局部近似替代真实rollout。

提交：`eval: add WeatherBench-X export and paired repair audit`。

### 延后事项

CoMoL core实现、地理区域程序、每步重规划、第二backbone与响应迁移分别立实验，不让首批依赖它们。每项都需清晰未解决的问题和固定baseline。

## 12. 本轮真实代码状态

已提供七个模块及合成测试，实际模块表见README_CN.md。新增utility head和在线规划器是基础实现；没有训练student，也没有weather bridge。

原包21项CPU测试重跑通过。新包最终精确测试数、版本与运行状态以TEST_REPORT.txt和EXECUTION_STATUS.json为准。本轮测试包括BF16预测下保持真值精度的回归检查；不等于GPU BF16模型已验证。

本包没有复制上游代码或权重，无用户账号/密钥。引用、许可与具体版本分别见SOURCES.md、VERIFIED_UPSTREAM.json。

## 13. 新代码使用的真实示例

```python
# 本包已提供：offline_callback必须由未来的天气桥接实现。
from earthdelta_response.probe import central_response
from earthdelta_response.geometry import ResponseGeometry
from earthdelta_response.teacher import box_candidates, verify_candidates

probe = central_response(offline_callback, reference, epsilon=0.02)
geom = ResponseGeometry.from_error(
    probe.response, target_summary - probe.reference_output, weights
)
candidates = box_candidates(geom, bound=0.25, max_active=2, budget=2)
# full_callback返回固定全场校验向量；本段仅限训练/离线。
verified = verify_candidates(
    full_callback, reference, candidates, target_full, weights_full,
    max_nonzero=3,
)
```

```python
# 本包已提供：feature来自待实现数据/天气接口和已训练状态编码器。
from earthdelta_response.utility import InteractionUtilityHead, plan_from_prediction
head = InteractionUtilityHead(features=128, dimension=8)
# 真实部署之前必须加载已经训练并独立验证的head；下面仅说明API。
b_hat, h_hat = head(feature)
plan = plan_from_prediction(b_hat[0], h_hat[0], max_active=2, budget=2)
# plan.coefficients交给未来Stormer受控rollout。这里没有访问真实未来。
```

运行原型真实命令：`python -m pytest -q`、`PYTHONPATH=. python examples/synthetic_probe.py`。计划中的天气脚本尚不存在，不能宣称已可运行完整训练。

## 14. 论文中心句与声明边界

建议中心句：

> EarthDelta从合法大气历史中学习低秩修正对未来预报的收益与相互影响，并在共同执行预算下选择有界修正程序，检验针对性轨迹修正能否比普通条件适配更有效地改善多步天气预测。

若实验只支持响应蒸馏，不支持交互规划，则据实收窄标题与贡献；若只有受控PDE支持则不能扩大为地球系统发现。没有因果发现、真实天气控制、全局最优修正或通用安全保证的声明。

## 15. 来源

[S1] CLAW: https://arxiv.org/abs/2609.12278
[S2] PEFSO: https://arxiv.org/abs/2609.12296
[S3] PEAR: https://arxiv.org/abs/2605.01361 ; author code https://github.com/FinJun/PEAR （仅链接确认）
[S4] CoMoL: https://arxiv.org/abs/2603.00573 ; https://github.com/DCDmllm/CoMoL
[S5] DISeL: https://arxiv.org/abs/2605.19028 ; https://github.com/alizindari/DISeL
[S6] Solver-in-the-Loop: https://arxiv.org/abs/2007.00016 ; https://github.com/tum-pbs/Solver-in-the-Loop
[S7] GEPS: https://geps-project.github.io/ ; https://github.com/itsakk/geps
[C1] Stormer: https://github.com/tung-nd/stormer/tree/58dfee5a6037399a40fefd492bc00421e0c885a8
[C2] WB-X example: https://github.com/google-research/weatherbenchX/blob/19548af79e75190254bba273ff5e3b179265e0bd/evaluation_scripts/run_example_evaluation.py
[C3] CoMoL layer: https://github.com/DCDmllm/CoMoL/blob/main/src/mocorelora/layer.py （读取blob9aa7a4aa9d0a0e796115e722b2a552638741b931）
[C4] PyTorch JVP: https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html
[C5] SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
[C6] WeatherBench2: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
