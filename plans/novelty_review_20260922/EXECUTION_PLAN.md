# EarthDelta 后续执行计划：先取得可信天气结果，再验证响应复用价值

日期：2026-09-22 UTC。依据：[研究建议](NOVELTY_VALUE_REVIEW.md)、[Round 4 审计](../../codex_audit_round4/ROUND4_REPORT.md)。本文件是待执行方案；此次研究整理未修改生产代码、未提交 H100/5090 作业，也没有宣称已经得到新的天气实验结果。

## 1. 目标、边界与交付

**第一目标：让共同 Fs 上的真实专家库、完整候选缓存和合法选择器形成可信闭环。第二目标：检验响应分解在目标/动作变化时，是否能获得比强直接收益预测更好的效果或更低的再校准成本。**

第一轮约 10 小时包含编码、排队、运行、分析与交接。基础交付是可验收的最小管线修复、真实运行记录和清楚的阻断原因；条件允许时，增加小型天气效果表。完整迁移实验和 novelty 证据放在随后数日，不能在未经 profile 前承诺十小时完成。

执行仍采用一名主要实现者、后台 CCI 作业并行；不依赖额外代码代理。多个已经满足前驱的作业可以排队，不预先提交占着 GPU 等待代码补齐的任务。

本次的最终交付应包含：

- `run_manifest.json`：源码、环境、数据角色、checkpoint/norm/Fs/bank/Q 的真实身份。
- `fs_qualification.json`、动态专家 checkpoint、正式 registry、独立重载记录。
- `cache_manifest.json`：预先声明的起报 × 全部候选 × 时效覆盖及失败列表。
- `method_results.csv`：真实 loss/gain、固定候选与合法策略、长程退化、完整成本。
- `HANDOFF.md`：`PASS/BLOCKED/FAIL_IMPLEMENTATION/INCONCLUSIVE/STOP/PIVOT`，对应证据和下一步。

这些是执行时要产生的产物，目前不能由本文替代。

## 2. 最小修复包：直接服务于实验

| 优先级 | 对应代码/审计项 | 实现与验收要求 | 依赖它的阶段 |
| --- | --- | --- | --- |
| P0-A | `static_adapter.py`、`r2_admission_gate.py`；C01/C04/C06 | 固定训练侧资格面板；拒绝外层 FAIL、陈旧/不足的内容时间证书；前置失败令后继 BLOCKED；不发布合格标记 | Fs 与全部正式缓存 |
| P0-B | `r2_fs_bank_train.py`；C02/C03/C05 | 一份 Fs 拟合/冻结后可重载；四 worker 禁止再各拟合 Fs；真实保存动态权重；profiling 使用副本且正式状态 hash 不变 | 专家库交付 |
| P0-C | `registry.py` 与 loader；C09 | 正式 registry 绑定实际 Fs/norm/objective/动态 checkpoint；所有专家的共同参考摘要相同；独立进程重载输出一致 | 缓存消费 |
| P0-D | 官方 exporter、`s0_gate.py` 及环境 | 恢复与实际执行栈匹配的官方验证；认证实际使用的 1/4/12 步或对应注册 rollout | 声称官方等价的天气实验 |
| P1-A | `heads.py`、`paired.py`、`metrics_contract.py`；C07/C08 | 统一原生 Q；analytic/calibrated 分开且默认校准关；零权重切片合法处理或明确拒绝 | 所有新 head 比较 |
| P1-B | `selection.py`；C11 | 保留被选候选 ID/行索引，不通过近似系数回查制造身份错误 | 多幅度/连续描述的候选扩展 |
| 合同限制 | bridge；C10 | 当前保持一进程一个模型、串行前向；后续若引入共享模型线程再补完整保护 | 当前不建设并发服务 |

验收需要调用真实入口：资格失败不能开始后继训练；四 worker 加载同一 Fs；保存的是最后验收的动态参数；重启进程重载后同输入输出一致。只给 helper 加测试，不能代替这些结果。

Fs 资格使用共同初始/最终 checkpoint 在同一固定面板上的 native objective，并记录均值、分时效和退化尾部。允许的容差、选 checkpoint、早停和回退规则在 bank-fit 内冻结。不规定每个样本都必须改善，也不按 dev/confirm 收益挑选 Fs。

## 3. 第一轮约 10 小时安排

| 墙钟目标 | 主要工作 | CCI 并行工作 | 到时必须能回答的问题 |
| --- | --- | --- | --- |
| T+0:00–0:30 | 建立唯一运行目录；冻结工作区与实际数据角色；复用 Round 4 反例整理最小补丁验收 | 检查现有官方环境与队列；只读查询，不重跑泛用硬件探针 | 本次真实依赖和已知阻断是什么 |
| T+0:30–2:30 | P0-A/B/C：准入消费、依赖停止、共同 Fs 加载、动态保存/重载、profiling 隔离 | 完整环境任务就绪后提交 J0；CPU 写代码时不占卡等待 | 错误产物是否确实被拒绝 |
| T+2:30–4:00 | 入口集成测试与正式 registry；补 P1-A；准备固定数据面板、S0 配置与 LR 对照 | J1 真实 S0；匹配执行栈后进行有界训练诊断 | 实际官方 parity 和训练合同分别是否通过 |
| T+4:00–5:30 | Fs 短/长程质量对照；选择规则只用 bank-fit；发布一份合格 Fs | J2 四卡优化对照，见下一节 | 500-update 退化是否被拦截；能否得到合格共同参考 |
| T+5:30–6:45 | 共同 Fs 上重训四个专家；组装 registry；独立进程重载验证 | J3 四卡各一个动态专家 | 磁盘上的产物是否就是验收的同一专家库 |
| T+6:45–8:15 | 实现/检查完整候选缓存；8 个暴露样本调试；通过后扩大预注册开发子集 | J4 按起报分片，每个起报包含全部候选 | 真实候选有多少余量，是否存在系统性退化 |
| T+8:15–9:00 | 最佳固定、regime、ridge direct-gain、oracle 的初步表；检查标签与合法输入分离 | 必要的小容量训练可 CPU 执行 | 动态选择是否值得继续，数据是否足以判断 |
| T+9:00–10:00 | 完整性检查、相关回归、结果与状态交接；不再启动超出窗口的大任务 | 回收本轮已完成/仍排队作业的真实状态 | 哪些已证实，哪些仍 BLOCKED/INCONCLUSIVE |

这是预算分配，不是通过承诺。P0 修复或官方环境花费超过四小时，就缩减后面的研究扩展，保留完整收尾。若小缓存入口未实现，不能用假路径提交“研究任务”；先交付已经运行的 Fs/专家管线结果。

三个硬门：T+4 未有可用执行合同，就不提交正式训练；T+6:45 未有合格共同参考及可重载 bank，就不启动正式 Stage 4；T+8:15 缓存不完整，就不输出伪装完整的 oracle/策略表。工程诊断可继续，但标记范围。

## 4. Fs 优化失稳：用四卡做可解释对照

先保存同一初始 adapter、输入内容与样本顺序。四卡诊断建议如下，均限 500 updates，并允许预注册的失稳早停：

| GPU | 条件 | 用途 |
| --- | --- | --- |
| 0 | 常数 LR=1e-2，当前优化设置 | 对照，允许以失败结束；不自动发布 |
| 1 | 同 GPU0 配置，独立进程重复 | 区分单次随机轨迹与可重复退化 |
| 2 | 仅将 LR 改为 1e-3 | 较小步长是否足以改善 |
| 3 | 仅将 LR 改为 3e-4 | 进一步检查步长敏感性 |

这些数值是待验证的对照候选，不是已证明的最优超参数。先不同时改调度、clip、顺序和初始化；LR 证据明确后再单独比较 warmup/decay。训练异常或面板显著退化仍保留在诊断表中，不能因为资格门拒绝它就从结果中删除。

在更新 0/32/64/128/256/500，记录固定面板 native loss、逐样本配对、裁剪前后梯度范数、真实 optimizer 步长、LoRA delta 范数、参数/输入 hash、TF32/deterministic/环境设置。当前 first/last 访问损失可以保留，但共同 checkpoint 面板才用于资格判断。

通过的配置按预先规则选一份 Fs，冻结并发布；动态专家只重载它。四卡实验产生的四份候选 Fs 不能被拼成“共同 Fs 上的四专家”。若都失败，保留 FAIL_IMPLEMENTATION/INCONCLUSIVE 并调查；可另立 `Fs=F0` 的明确对照研究合同，但不能将其冒充已修复的静态适配结果。

历史 formal-v2 作业总耗时 490.26 秒，expert0 的 Fs fit 为 167.47 秒、动态训练为 172.28 秒。这说明八样本短目标诊断不必占用十小时全部预算；它们使用的是存在缺陷的旧流程，**不能作为新数据量、12 步 rollout 或修复后完整缓存的吞吐保证**。原始来源为 `artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12/`。

### 4.1 H100 与 5090

主资源仍用已验证的 4×H100 CCI 规格。现有证据不能认定 H100 硬件异常。5090 可以用于以下两个分支：官方环境在 H100 暂不可用而 5090 有兼容栈；或者要做同初始权重/输入的设备复验。

5090 的实际资源规格、可用显存、驱动、Torch/CUDA/xformers 支持需在任务中先检查；不能把当前 H100 镜像直接视为兼容。先测与模型有关的真实前向/反向再决定 batch/horizon。软件栈不同时，将结果称为“设备与执行栈对照”，不把差异全部归为 GPU 硬件。切卡不替代共同 Fs、质量门、保存重载和官方验证。

## 5. 后续分阶段研究，避免一次改太多

天数是工作量安排，排队和实测吞吐决定实际日期。免费 GPU 允许多种子和对照；论文中仍须记录计算成本。

| 阶段 | 建议工作量 | 实验/交付 | 继续条件 |
| --- | --- | --- | --- |
| R0 | 首个约 10h 窗口 | 最小修复、Fs/专家证书；条件允许得到 8/128 起报初表 | 正式前驱全部通过 |
| R1 | 接下来 1–2 天 | K=4 固定动作完整 pilot；足够 policy-fit 与 dev 时间块；cheap legal、固定与 oracle | 有动态余量，并能用合法信息恢复一部分；不确定时按预定预算扩样 |
| R2 | 再 2–3 天 | 原 e/u、决策分量、标量/向量 direct-gain、直接候选预测；配对 e/u 四格诊断 | 结构带来效果/样本效率信号，或明确支持更简单方法 |
| R3 | 再 2–3 天 | 目标与幅度联合留出；virtual edit、逐步 feedback；0/少量重校准成本曲线 | native gain、长程退化与总代价共同支持主张 |
| R4，可选 | 再 2–4 天 | 多个新专家库、行为描述接入；更多独立训练种子 | 主线已有信号，且新库接入是明确使用需求 |
| R5 | 由功效与新数据决定 | 新时间段确认、冻结分析、最终结果包 | 数据从未用于本项目选择；主张与结果一致 |

一个建议的 R1 起点是 512 个 policy-fit 起报、128 个开发起报；这是低容量模型的起步预算，不是“足够训练”或显著性的保证。依据时间块有效样本数、学习曲线和事先确定的扩样预算，扩大到 1024/2048 等规模。所有候选在同一起报上完整采集，不能按响应幅度或收益选择保留样本。

## 6. 数据角色与防止实验自欺

初始分工建议：2015/2018 bank-fit，2019 policy-fit，2020 development/calibration；最终以实际内容准入、历史使用账本和基础模型的预训练截止日期为准。2016/2017 当前 store 为空，不能列进可用训练量；2021 尚待完成和核验。2020 已经暴露，不是 confirm。

bank-fit 内部再区分更新样本和固定资格/探针面板。policy-fit 内做时间块交叉验证；dev 内区分模型选择和经验性回退校准，或采用完整时间块交叉拟合。训练、探针、校准、评估角色记录在 manifest 中，采样使用固定 seed 和真实 UTC 时间，而不是随机打散 cache 行。

边界检查依据每个样本 `[earliest_history_time, latest_target_time]`，拒绝跨角色复用支持范围。以 t-12h 历史、t+72h 目标为例，不能只确认起报时刻不同就宣布隔离。块 bootstrap 的长度还需要覆盖天气相关性，开发侧可以比较 7/14/28 天敏感性，但不根据 confirm 显著性选块长。

失败处理规则提前冻结：基础数据/身份失败会阻断整批发布；合法执行时的候选数值失败进入失败表，该候选的 oracle 行动视为不可用；策略若选择失败动作，按预定义的 no-edit 回退或失败代价计入。回退若需要额外运行 Fs，其成本照计。报告始终保留全部预定起报，不能只评价成功子集。

## 7. 第一张完整效果表与关键指标

主目标先固定 24h 的原生加权目标；6h 与 72h 分开报告，并使用预先定义的 72h 退化容差。多目标阶段才切入区域、变量和时效权重组合。可以使用温度、风或区域加权作为应用例子，但必须绑定实际变量/单位；没有业务效用模型时，不把它直接称为电力收益。

`method_results.csv` 至少包含：

```text
method, split, bank_hash, objective_id, action_set_hash, seed
N_issues, N_time_blocks, missing_rows, failures, no_edit_rate
native_loss_6h, native_loss_24h, native_loss_72h
gain_vs_Fs, gain_vs_best_static, paired_CI, oracle_regret
harm_rate, harm_tail, response_error, gain_calibration_error
offline_rollouts, offline_gpu_seconds, recalibration_rollouts
inference_gpu_seconds, latency_p50, latency_p95, peak_memory, cache_bytes
```

至少三种 head seed 用于最终神经模型比较；bank seed 也需重复，固定候选的选择只能使用训练/开发数据。时间相关样本的 CI 按完整时间块配对；多目标、多个基线的主/次比较提前声明。数据不足时输出描述性区间与 INCONCLUSIVE，别把八个 debug 样本作科学结论。

主判据建议在开发结束后冻结为：相对预选强基线的真实增益下置信界超过有意义阈值，且 72h 退化不超容差；或在效果等效带内明显降低重新仿真/训练/部署成本。阈值需要结合开发集波动与应用尺度定数，本文不凭空指定“0.1% 就够”。

## 8. 计算、缓存与公平预算

K=4 加 no-edit 共五候选。若一次轨迹同时产出 6/24/72h，则 128 个起报需要 640 条最长 72h 轨迹，而不是把三个端点重复当成三次独立训练样本。512 个 policy-fit 起报为 2560 条；若执行日程不同导致不能共享，按实际调用数计费。

仅保存 128×5×3×69×128×256 的 FP32 预报端点约 17.36 GB，即 16.17 GiB；相同三端点真值另约 3.47 GB。1024 起报仅这部分预报约 138.92 GB。这里不含历史、特征、临时副本和额外动作。按起报分片并流式写出，先 profile 8 个起报再扩张。

响应压缩可以降低 head 训练成本；研究不同区域/目标和完整反馈订正时，必须保留必要分辨率的数据或明确重算预算。不能为某方法保存更多模拟信息而不给对照同样访问权。

两条公平比较都要保留：相同缓存/标签下的效果比较，以及总计算受限时的方法比较。结构方法所用的额外无真值仿真、行为探针、feature preview、超参数搜索和重试均进入成本。直接收益头可利用相同模拟数据做辅助任务；如果不给，就另报这个限制。

serving 延迟测全路径：合法特征提取 + 可选 preview + 全候选小头打分 + 选中动作的真实天气 rollout + 回退。不要只报小头前向耗时。virtual edit 若需要完整 Fs 预报，也计完整成本；逐步反馈订正的每步小模型调用都计入。

## 9. 现有入口、待实现入口与实际启动方式

### 9.1 当前状态

| 入口 | 当前事实 | 下一步 |
| --- | --- | --- |
| `scripts/r2_fs_bank_train.py` | 已有 `fs_fit/fs_freeze/bank_train/profile/horizon_check/all`；当前没有可用的共同 Fs 加载流程 | 修复依赖与发布；添加真正可测试的 Fs checkpoint 加载和动态保存 |
| `scripts/r2_admission_gate.py`、`scripts/s0_gate.py` | 已有 CLI，仍有前述消费/环境缺口 | 在真实入口验证后使用 |
| `plans/plans_v2_0921/cci/submit_job.py` | 已验证通用 argv 提交、快照、dry-run | 复用，不另写调度平台 |
| 完整 candidate cache / head train / evaluation | 当前没有已验收的一体化研究 CLI | 建议新增小入口 `scripts/r4_response_pilot.py`，复用上述原语 |

建议新入口只包含 `cache`、`fit-heads`、`evaluate` 三个子命令，共享一份明确 config。不要同时新建通用 DAG、分布式训练平台、在线服务、自动文献代理。新模块只在方法比较确需时增加 `response_basis.py`、`adapter_descriptors.py` 等，首轮可以先在小脚本内实现简单回归。

以下为**待实现的接口合同，不是当前可运行命令**：

```text
scripts/r2_fs_bank_train.py --stage fs_freeze --fit-record <qualified-fit>
scripts/r2_fs_bank_train.py --stage bank_train --fs-artifact <shared-fs>
scripts/r4_response_pilot.py cache --config <frozen-config> --shard <i/4>
scripts/r4_response_pilot.py fit-heads --config <frozen-config>
scripts/r4_response_pilot.py evaluate --config <frozen-config>
```

实现者可采用同等明确的名字，但要在交付中保存实际 `--help`、解析测试和执行 argv。`--fs-artifact`/`--fit-record` 不能只出现在错误文案里。Fs-freeze 的 artifact 需绑定通过的资格结果；bank-train 入口应拒绝“无 Fs 时自动重新训练”的隐式行为。

### 9.2 提交器的可用命令

在 CLI 和配置已实现并测试后，建立本轮唯一目录，例如：

```bash
cd /mnt/afs/260010168/EarthDelta
ED_RUN="$(mktemp -d /mnt/afs/260010168/EarthDelta/artifacts/ed-value-20260922-XXXXXX)"
```

把完整 argv 写成 JSON 数组，保存在该目录。示例结构如下；`r4_response_pilot.py` 是上一节待实现入口，不能立即拿此示例当作就绪任务提交：

```json
[
  "python3",
  "-u",
  "@SNAPSHOT@/scripts/r4_response_pilot.py",
  "cache",
  "--config",
  "/absolute/path/to/frozen/research_config.json",
  "--shard",
  "0/4"
]
```

实际 Python 应使用 J0 验证的解释器/环境；示例的 `python3` 仅表示结构。配置必须在提交前存在，绑定全部实际身份并通过校验，不得带占位 hash。

```bash
python3 plans/plans_v2_0921/cci/submit_job.py \
  --label ed-value-cache0 --timeout-seconds 7200 \
  --argv-file "$ED_RUN/cache0.argv.json" --dry-run
```

检查快照与命令后，在前驱通过的情况下才执行提交：

```bash
python3 plans/plans_v2_0921/cci/submit_job.py \
  --label ed-value-cache0 --timeout-seconds 7200 \
  --argv-file "$ED_RUN/cache0.argv.json"
```

dry-run 会创建新的快照目录；实际提交也会生成新快照。二者之间不要再改实验源代码/配置；核对正式快照身份。四卡可以由一个已完成的小 runner 在四个独立进程中处理四个 shard，不用四个四卡节点各只跑一张卡。

查询实际返回的 JOB_ID：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs describe JOB_ID \
  --workspace-name=share-space -o json
```

读取对应 run_dir 中的 `worker.log`、`job_result.json` 和科学产物。提交超时先按唯一 run_id 查询，避免重复提交。平台 SUCCEEDED、returncode=0、科学门 PASS、策略收益成立分别记录。job timeout 不含排队，十小时窗口必须另记真实墙钟时间。

### 9.3 必要验证

修复后运行与变化相关的 CPU 测试，先复现 Round 4 反例，再确认反例变为拒绝；新功能测试聚焦实际边界。现有可用回归命令为：

```bash
CUDA_VISIBLE_DEVICES='' PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
PYTHONPATH="$(pwd):$(pwd)/.pydeps:${PYTHONPATH:-}" \
python3 -m pytest tests/test_fs_static_adapter.py tests/test_bank_training.py \
  tests/test_metric_contract.py tests/test_pilot_admission_b09.py \
  tests/test_candidate_registry_b15.py tests/test_content_verification_b08.py \
  tests/test_s0_gate_end_to_end.py -q
```

Round 4 的 434 个 CPU 测试通过是已有证据，不能给新补丁背书。缓存新增测试至少覆盖缺候选、重复起报、错 Fs/bank/Q、失败候选保留、重载不一致、合法输入混入 truth；评分再用真实小缓存作端到端验收。只有合成测试不能证明天气效果。

## 10. 交给实现者的最终决策次序

1. 按 P0 最小修复包恢复可信数据生产，不扩大系统范围。
2. 做真实 S0 与 Fs 对照，发布一份共同参考及可重载动态库。
3. 用固定小字典取得完整缓存和 cheap legal 表，决定是否值得训练复杂 head。
4. 让完整 e/u、决策分量和强向量收益方法在同一数据/预算上竞争。
5. 有信号后做联合目标/动作留出；virtual edit 与反馈订正必须接受同等机会。
6. 只有新库接入确实有价值时，扩大行为描述实验；最后用未触碰数据确认。

若第 4/5 步支持更简单方法，采用更简单方法并重写主张；若支持响应分解，则围绕复用效果和成本建立论文证据。不要为保住某个模块而不断改变评价目标。
