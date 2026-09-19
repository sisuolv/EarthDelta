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
