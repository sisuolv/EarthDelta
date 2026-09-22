# EarthDelta Round 4 独立审计报告

审计日期：2026-09-22 UTC。基准：`d749a1c62521226df857587e08f7d067b0f15355`，分支 `audit/round2-review-20260921`。对象为基准之后的实际工作区 diff 及未跟踪的新代码，不是 commit range。本文没有把作业终态、数值等价和训练质量合并为一个 PASS。

## 零、Fs-fit 退化问题：必须先处理，再启动正式 Stage 4

**`fs_fit_divergence_verdict = MUST_FIX_BEFORE_STAGE4`。** 配对诊断没有发现会制造此次假阳性的符号、样本配对或训练/验证混淆错误。两个 formal 作业各有两份 Fs 在同一批八个样本上全部退化；冻结后的共同面板评分也确认退化。**当前不能继续把 Stage 3b 定性为“合格专家库已经成功产出”。Stage 4 当前状态应为 `STOP`。** 此判断独立于后文三个局部 bug 修复是否成立。

诚实的 Stage 3b 状态是：**执行与部分数值一致性证据成立；训练质量准入失败，当前产物不能认定为合格的共同 Fs 上 K=4 专家库。** 最新作业 `SUCCEEDED` 是进程成功完成的事实，不是训练质量或专家库交付资格。

### 0.1 诊断逻辑及原始记录复算

`evidence_status: STATIC_CONFIRMED`。

- `earthdelta/static_adapter.py:842` 的 `per_sample_loss_direction` 按 `issue_id` 聚合；`:867` 用 `last < first` 判断改善，符号正确。
- `earthdelta/static_adapter.py:942` 从固定样本列表 round-robin 取样，`:954` 计算同一个训练目标；`:974` 执行更新后，`:976` 保存的是此前计算的 loss，`:979` 保存同次访问的样本 ID。这个 loss 是训练目标，不是验证损失。
- 从三个作业十二份 `run_record.json` 的原始 loss/issue-ID 序列重新聚合，配对结果与记录一致。实际这些记录没有长度错位或配对错样本现象。
- 必须保留一个口径限定：first/last 是各样本首次/末次被访问时的优化前损失，各自处在不同更新时刻；它不是“共同初始 checkpoint 对全部样本”和“共同最终 checkpoint 对全部样本”的严格面板比较。因此另外核对 `earthdelta/static_adapter.py:1519` 的固定样本均值，以及 `:1555`、`:1576` 分别计算的 F0 与冻结 Fs 评分。它们给出相同方向的结论。

以下数值由原始记录独立重算；完整逐样本值、梯度统计、记录 SHA256 在 [gpu_records_recomputed.json](evidence/gpu_records_recomputed.json)。表中“冻结 Fs/F0”是同八样本平均训练目标的比值，用来判定训练是否退化，不替代未来研究所需的 Fs-relative gain。

| 作业 | worker | 改善样本 | 配对首次 loss 均值 | 配对末次 loss 均值 | 冻结 Fs/F0 | 最大裁剪前梯度范数 |
|---|---|---:|---:|---:|---:|---:|
| `pt-3x63g0c6`，500 updates | expert0 | 8/8 | 0.03195348 | 0.02831234 | 0.885448 | 0.0341 |
| 同上 | expert1 | **0/8** | 0.03195336 | **0.15188166** | **4.644734** | **1934.19** |
| 同上 | expert2 | 8/8 | 0.03195372 | 0.02896663 | 0.907321 | 0.0943 |
| 同上 | expert3 | **0/8** | 0.03195344 | **0.03978072** | **1.219248** | **101.40** |
| `pt-cdj1s2le`，500 updates | expert0 | 8/8 | 0.03195338 | 0.03055698 | 0.957994 | 5.94 |
| 同上 | expert1 | 8/8 | 0.03195361 | 0.03116442 | 0.976921 | 30.58 |
| 同上 | expert2 | **0/8** | 0.03195359 | **0.13426245** | **4.066871** | **1771.61** |
| 同上 | expert3 | **0/8** | 0.03195359 | **0.03421454** | **1.070957** | **744.13** |

32-update 作业 `pt-2r6tbwu7` 的四个 worker 都是 8/8 改善，冻结 Fs/F0 为 0.96920 至 0.96930。两个 formal 作业的失败位置分别是 1/3 和 2/3；第一个 formal 作业自身因固定容差检查失败而终态 `FAILED`，第二个在新容差下终态 `SUCCEEDED`，两者的训练退化事实都独立成立。

这里确认的是**有限但显著的优化退化/失稳**，并不声称已经证明数学意义上的无限发散，也不把它直接上升为研究 idea 或 novelty 的否定。八个样本并不提供八次独立随机实验，但同面板全面反向且冻结评分大幅恶化，足以否决当前产物的训练质量资格。

### 0.2 根因假设排序及所缺证据

下列因果假设均为 `PLAUSIBLE_UNVERIFIED`；用于支持排序的配置、调用路径和统计值已静态或从记录确认。不能把“最可能”写成“已定位”。

| 排序 | 假设及现有支持 | 确认或排除所需证据 |
|---|---|---|
| 1 | 常数 Adam LR=0.01，在八样本上反复更新 500 次，没有 LR 调度、最佳 checkpoint 回退或质量停止。`static_adapter.py:928`、`:941`、`:963`、`:974`；退化 worker 有巨大梯度尖峰。clip=1 限制梯度范数，并不直接限制 Adam 的参数更新或累计 LoRA delta。 | 保存失稳前后 checkpoint，记录逐步 loss、裁剪前后范数、实际参数步长、delta 范数和固定面板评分；保持初始化与顺序相同，只改变 LR/调度，比较首次偏离时刻。 |
| 2 | 训练数值微扰被长程优化放大。四份 Fs 配置的 seed 都是 20260921，样本 ID 和顺序相同，但最终 Fs 不同；这比“专家天然用了不同 Fs seed”更符合记录。 | 先比较初始化和输入内容 hash，再做同设备重复训练及 deterministic/TF32 受控实验，定位首个前向或反向差异。相同 seed/ID 不等于已证明所有输入字节和训练内核相同。 |
| 3 | Adam 矩估计、大步更新与小样本周期顺序耦合。每次 fit 新建 Adam，未发现恢复旧 optimizer 或跨进程共享 optimizer 的路径。 | 保存首次失稳前后一/二阶矩与真实更新量；固定其他条件分别改变顺序和 clip。没有证据时不要优先归因于“旧优化器状态污染”。 |
| 4，当前证据不支持 | Fs 的专家专属 seed、月份分组或 target-block 分工导致某个 slot 必然坏。Fs 阶段四进程的 seed、样本顺序、目标块相同；`r2_fs_bank_train.py:475` 的 diversity grouping 在后续动态训练才发生。 | 若坚持此解释，须提供未记录的初始化、实际取样或 argv 差异；不能用后续动态专家分组解释此前 Fs 拟合。 |

`self_max_abs_diff=0` 仅证明指定冻结推理调用的两次输出一致，不能排除训练 backward、更新过程或跨设备重复运行的数值非确定性。现有证据不足以认定 H100 硬件异常。

### 0.3 为什么“临时剔除两个专家后继续”还不够

有一个比退化比例更基础的结构问题：每个 `--expert-index` 进程都重新 `fs_fit -> fs_freeze`，再训练本进程的动态专家。见 `scripts/r2_fs_bank_train.py:304`、`:369`、`:963`。从三个作业的实际记录读取到的 merged Fs digest，**每个作业均为四个不同值**。

因此，当前是四个不同 Fs 背景上的训练实例，并非一份共同 Fs 上的四专家库。直接留下两个“看起来正常”的实例，仍无法得到一个共同 no-edit 参考；把它们当作同一库会令候选差异混入参考模型差异。CLI 也没有真正实现加载既有 Fs 的工作流，尽管错误信息在 `scripts/r2_fs_bank_train.py:366` 提到了 `--fs-adapter`。

明确建议：**先修复资格门和共同 Fs 产物合同，再产生新的合格专家库，之后才启动正式 Stage 4。** 不要求等全部根因被因果实验彻底证明；根因调查可以并行。但当前只记录退化，或仅剔除两个 worker，都不足以放行。

最低要求是：在 bank_fit 侧预先冻结的面板上比较共同初始/候选最终 checkpoint，定义允许的退化容差、回退与选择规则；只冻结一份合格 Fs；所有动态专家加载同一内容摘要的 Fs；资格失败阻断发布及后继执行。不得按后续 dev/confirm 的收益挑选“好 Fs”，也不需要草率规定每个样本都必须改善。后文还要求保存并重载动态权重、修复准入消费与 profiling 的状态变更。

独立机器可读结论：[fs_fit_divergence_assessment.json](fs_fit_divergence_assessment.json)。

## 一、代码 findings，按严重度排序

本轮编号 C01–C11，共 11 条：P1 九条、P2 两条。全部问题事实为 `STATIC_CONFIRMED`；C07/C08/C10 是前轮问题的未关闭残留，不冒称本轮新引入了十一种回归。因果假设另列在第零部分。完整 schema 见 [audit_findings.json](evidence/audit_findings.json)。

### C01 — P1：真实训练退化不影响 Fs 资格

`evidence_status: STATIC_CONFIRMED`；`disposition: NOT_CLOSED`；当前 Stage 4 阻断项。

位置：`earthdelta/static_adapter.py:906`、`earthdelta/static_adapter.py:954`、`earthdelta/static_adapter.py:990`、`earthdelta/static_adapter.py:1644`；`scripts/r2_fs_bank_train.py:334`、`scripts/r2_fs_bank_train.py:357`。

训练资格只排除非有限 loss/梯度与零更新。`loss_direction` 被打印和保存，未进入资格判断；post-freeze `passed` 只组合 merge、zero-edit、静态身份。最新 formal 作业的退化 Fs 因而正常返回 PASS。问题的独立数据与结论见第零部分。最小修复是增加独立训练质量门，并令它真正约束冻结发布和后继执行。

### C02 — P1：四进程独立冻结四份 Fs，无法组成共同基线的 K=4 库

`evidence_status: STATIC_CONFIRMED`；`disposition: NOT_CLOSED`；当前 Stage 4 阻断项。

位置：`scripts/r2_fs_bank_train.py:304`、`scripts/r2_fs_bank_train.py:369`、`scripts/r2_fs_bank_train.py:475`、`scripts/r2_fs_bank_train.py:963`。

`--expert-index` 只限定动态专家训练索引，没有共享 Fs 的加载/核验机制。十二份真实记录支持三个作业各有四种 Fs 内容身份。最小修复是 Fs 单独合格后冻结一次，worker 显式加载同一 Fs，在组装时严格核对 Fs/norm/objective 身份。已有不同背景上的专家不能仅靠改 manifest 变成共同背景上的专家。

### C03 — P1：动态专家权重没有落盘，训练完成后无法交付或复用

`evidence_status: STATIC_CONFIRMED`；`disposition: NOT_CLOSED`；当前 Stage 4 阻断项。

位置：`scripts/r2_fs_bank_train.py:541`、`scripts/r2_fs_bank_train.py:564`、`scripts/r2_fs_bank_train.py:573`、`scripts/r2_fs_bank_train.py:1012`。

动态 bank 在内存更新，写出内容只有 grouping、registry 和训练记录 JSON。实际目录确实存在 `fs_adapter.pt`、`fs_merged_backbone.pt`，但它们是 Fs 的权重，不是动态 bank；未发现保存动态因子的路径或产物。见 [artifact_crosschecks.json](evidence/artifact_crosschecks.json)。最小修复是保存验收后的动态 state_dict 与内容身份，独立进程重载后复验输出、非零响应及共同参考绑定。

### C04 — P1：准入结果、内容证书及时间/归一化身份没有在训练消费处闭合

`evidence_status: STATIC_CONFIRMED`；B08/B09 `PARTIALLY_CLOSED`；当前 Stage 4 阻断项。

位置：`earthdelta/static_adapter.py:1817`、`earthdelta/static_adapter.py:1832`、`earthdelta/static_adapter.py:1867`、`earthdelta/static_adapter.py:1881`、`earthdelta/static_adapter.py:1925`；`scripts/r2_admission_gate.py:154`、`scripts/r2_admission_gate.py:185`；`earthdelta/data/make_splits.py:980`。

消费端信行级 `admitted`，不要求整体 `passed`，不核验内容证书覆盖、实际 normalization identity 或重新打开 store 后的绝对 issue time。CPU 反例确认：外层门失败仍加载一个样本；错误 normalization hash 被接受；将 store 时间轴整体后移 24h 后，仍以旧 issue-ID/issue_time 加载新时间的数据。formal admission 对缺失 grid stamp 也不拒绝。

真实训练用的 admission record 为八行，但只核验前两行的内容、行内证书全部为 null；行准入目标是 6h，实际 Fs/bank 训练目标为 24h。加载器确实检查请求索引范围和时间间隔，然而这不能证明旧证书和当前实际目标属于同一内容身份。该 finding 不声称现有天气数据已坏，确认的是错误/陈旧记录可穿过当前边界。

最小修复：正式加载只接受可验证整体 PASS，按绝对时间及请求 horizon 定位真实输入/目标，校验 norm/grid/channel/store 内容身份，内容证书覆盖全部实际消费范围；抽查不得冒称完整训练集认证。

### C05 — P1：profiling 对已训练专家继续更新，验收记录与最终状态分离

`evidence_status: STATIC_CONFIRMED`；`disposition: NOT_CLOSED`；当前 Stage 4 阻断项。

位置：`scripts/r2_fs_bank_train.py:584`、`scripts/r2_fs_bank_train.py:607`、`scripts/r2_fs_bank_train.py:963`；`earthdelta/bank_training.py:1332`、`earthdelta/bank_training.py:1353`。

`all` 在 bank_train 之后把同一个 bank 交给 profiler；后者新建 Adam 并实际 `optimizer.step()`，不复制、不恢复。CPU 一步即使活动参数变化约 0.00991–0.00997。训练资格、更新数与 nonzero-response 记录在这次变更之前，后续 horizon 却消费变更之后的权重；profiling 固定取全部样本中的第一份，还可能不属于该专家分组。

最小修复：在独立副本上测量或先测量再正式训练，要求测量前后正式 bank 的 state hash 相同，并确保保存的版本就是最后验收的版本。

### C06 — P1：前置阶段失败后仍执行依赖阶段并写出产物

`evidence_status: STATIC_CONFIRMED`；`disposition: NOT_CLOSED`；当前 Stage 4 阻断项。

位置：`scripts/r2_fs_bank_train.py:373`、`scripts/r2_fs_bank_train.py:440`、`scripts/r2_fs_bank_train.py:975`、`scripts/r2_fs_bank_train.py:992`、`scripts/r2_fs_bank_train.py:995`。

handler 返回 False 或异常被记录后，循环仍继续。Fs bridge 在核验前进入 ctx，核验失败仍保存 checkpoint。CPU 注入 `fs_fit=False` 后五个阶段都被调用；首个 formal GPU 记录也显示 freeze 失败后动态训练继续。最终总体 FAIL/rc=1 没有被掩盖，但它不能替代依赖阶段的执行门。

最小修复：明确阶段依赖，失败则将后继标为 BLOCKED/SKIPPED 并不执行；需要保留的诊断产物单独隔离，合格产物通过验收后再发布。

### C07 — P1：B05 的旧逐特征全零权重切片除零仍未修复

`evidence_status: STATIC_CONFIRMED`；B05 `PARTIALLY_CLOSED`；属于前轮残留。

位置：`earthdelta/metrics_contract.py:148`、`earthdelta/metrics_contract.py:215`；`earthdelta/paired.py:70`、`earthdelta/paired.py:103`。

新 `weighted_mse` 校验正确，但旧 `quadratic_gain` 只验证全局权重和，再逐 F 归一化。独立有限输入反例中，一个正常切片和一个全零切片得到 `[[[[1.0], [nan]]]]`。既有 paired/head 仍调用它。最小修复是在广播后逐实际约简轴验证正分母，或明确实现 mask 语义。它不阻断当前不调用该 reducer 的 native Fs/bank 目标，但必须在相关 head/selector 研究比较前关闭。

### C08 — P1：B06 的新无校准 API 尚未替代 head 的唯一混合 gain 输出

`evidence_status: STATIC_CONFIRMED`；B06 `PARTIALLY_CLOSED`；属于前轮残留。

位置：`earthdelta/heads.py:294`、`earthdelta/heads.py:351`、`earthdelta/heads.py:356`、`earthdelta/heads.py:359`；`earthdelta/paired.py:103`；`earthdelta/metrics_contract.py:793`。

新全目标 API 正确且校准默认关闭，但既有 head 仍无条件将 trainable calibration 叠加到唯一 `gain`，paired 标签仍是逐 F 诊断值。这两个旧文件未被本轮修改，因此不作为新回归，也不能算 B06 全部关闭。最小修复是在进入对应比较前统一 Q_eff、分开 analytic/calibrated 输出并默认关闭校准，用非均匀 lead/area 权重验证全链路目标一致性。

### C09 — P1：registry 校验系数，但不校验正式产物身份

`evidence_status: STATIC_CONFIRMED`；B15 `PARTIALLY_CLOSED`；当前 Stage 4 阻断项。

位置：`earthdelta/registry.py:275`、`earthdelta/registry.py:301`、`earthdelta/registry.py:342`、`earthdelta/registry.py:487`、`earthdelta/registry.py:609`；`scripts/r2_fs_bank_train.py:492`。

metadata 不解释、artifact_ref 可空、validate 不绑定 norm/Fs/动态权重/objective。CPU 构造带 `normalization_hash=placeholder_1979_2018` 的 pilot registry 仍通过。formal 样本准入确实拒绝 placeholder，但那个检查在 make_splits，不在 registry。当前 bank registry 在动态训练前建立，后续也未绑定动态产物。

最小修复是区分可宽松构建的候选系数表与正式可消费 registry；正式 freeze/load/assert-ready 边界拒绝缺失或占位身份，确认所有行共用 Fs 且引用真实可加载的专家权重。

### C10 — P2：普通 forward 不参加 controlled-rollout 的共享模型保护

`evidence_status: STATIC_CONFIRMED`；B12 `PARTIALLY_CLOSED`；当前串行流程非阻断项。

位置：`earthdelta/bridge/stormer_bridge.py:756`、`earthdelta/bridge/stormer_bridge.py:904`、`earthdelta/bridge/stormer_bridge.py:1083`。

新 registry 能防止两个受控 rollout 并发，但普通 `forward_validation` 可以在另一个线程挂有编辑 hook 时运行。CPU 双线程反例中普通前向偏移 `3.695487976074219e-05`，且等于编辑后前向。现有生产脚本为串行，四 GPU 进程分别持有自己的模型；没有证据该重叠实际发生在本轮 GPU 作业，因此不能据此解释 Fs 退化。

最小修复是把“普通与受控前向也必须串行”明确纳入调用合同；若未来开放共享模型并发，让所有相关前向参与同一生命周期保护。当前不应为不存在的生产线程池强制建设并发执行框架。

### C11 — P2：registry 的近似系数反查可以把真实非零动作标成 no-edit

`evidence_status: STATIC_CONFIRMED`；B15 `PARTIALLY_CLOSED`；当前固定 a0=0.25 表非阻断项。

位置：`earthdelta/selection.py:382`、`earthdelta/selection.py:385`、`earthdelta/selection.py:387`。

float32 分支按 `atol=1e-6` 匹配第一个 registry 行。合法单专家候选 `5e-7` 被 planner 选中，预测 gain 约 `1e-6`，却返回 `plan_id=reference`、coefficients=(0,)；surrogate 里仍是非零系数。当前五行 pilot 候选不会触发，但通用入口的身份对应关系已被反例否定。最小修复是直接保留规划器选中行索引，或同执行 dtype 精确匹配并拒绝不可区分候选。

## 二、逐项关闭判定与三个 Stage 3b 修复

### 2.1 B05/B06：全目标公式正确，但不等于旧合同全部关闭

新 full-objective 子项判定 **CLOSED**；完整 B05、B06 均为 **PARTIALLY_CLOSED**。

`earthdelta/metrics_contract.py:463` 只累计最后四个 H/V/Lat/Lon 维；`:557` 构造 `Q_eff = (q/s²)/sum(q)`，`:652`/`:713` 使用同一全目标约简。独立脚本以 Python/NumPy 显式循环逐 B/K/H/V/Lat/Lon 手算，没有调用实现内部 helper 充当 oracle；覆盖共享 q、按 batch 的 q、按 batch/expert 的 q 与空间广播，返回 B/K 均保留。

最大 gain 误差 `2.6645352591003757e-15`，最大 loss 误差 `4.440892098500626e-15`，Q_eff 最大误差 `1.1102230246251565e-16`。`weighted_mse(weights=[2,-1])` 抛出 `weights must be non-negative`。新 `gain_analytic` 与 `gain_calibrated` 分离，后者默认等于前者，独立探针验证为精确相等。

旧 reducer 的模块说明及 `quadratic_gain` 文档明确标为 diagnostic-only；新增 Fs/bank 的 `objective_loss_for_sample` 使用 full-objective，没有发现把旧 reducer 当新训练目标的路径。旧 paired/head 的使用与校准残留另由 C07/C08 说明，不能因文档标签而忽略实际消费者。

### 2.2 B08/B09/B12/B13/B14：接线与真实边界

| 项目 | 判定 | 独立核实 |
|---|---|---|
| B08 | PARTIALLY_CLOSED | `scripts/r2_pilot_preflight.py:204` 调用 `run_pilot_preflight`；`pilot_contract.py:1517`、`:1538`、`:1553` 真正执行 artifact binding、slice 与 loaded sample 校验，不是死代码。新 admission 路径与消费侧仍有 C04；旧 marker/rechunk 时间启发式仍未整体修复。 |
| B09 | PARTIALLY_CLOSED | `make_splits.py:257`、`:267`、`:520`、`:521` 的 datetime 构造，以及 `:734`、`:738` 的 fromtimestamp 都显式 UTC；`:757` 坐标转换为 UTC。`:949` 列出 history/issue/target，`:957` 逐磁盘时间索引查找，`:992` 要求真实存在。formal placeholder 拒绝在 `:911`。这些实现有效，消费侧身份闭合仍见 C04。 |
| B12 | PARTIALLY_CLOSED | model-id 全进程 registry + mutex 修复了受控对受控的跨线程/跨 bridge 共享模型竞争，并拒绝已声明 activation-checkpoint 模式。当前生产串行；未覆盖普通前向的残留见 C10。 |
| B13 | CLOSED | `scripts/s0_gate.py:1957` 的提交不变量拒绝未评估项、False 项及缺 config/source/details；`:2419` 的异常路径无条件撤销 PASS/committed。独立注入 provisional PASS 后异常，结果为 status=exception、pass=False、committed=False。`:2668` 起的产物发布采用 staging/原子目录发布且 PASS marker 最后写入。 |
| B14 | CLOSED | `scripts/s0_gate.py:2455` 安全格式化缺失/非有限数值；`:2841` 附近有报告降级路径。相关 CPU 缺字段/失败路径测试通过。 |

B13 的非关键 cleanup（释放模型引用/empty-cache）被明确定义为告警路径；这与关键计算/身份产物写出异常不同，不把告警误读成静默保留失败 PASS。

### 2.3 设备不匹配修复：CLOSED，限已审查的实际单设备路径

关键修复实际在 `scripts/r2_fs_bank_train.py:389`：把新建于 CPU 的全部动态 LoRA 移到 Fs model device 后才执行 zero-edit。训练、profiling 和 horizon 调用各自也把 bank/sample 移到 model device；`earthdelta/static_adapter.py:1166` 在合并时显式统一因子到 projection 的 dtype/device，`:646` 附近把目标和权重移到实际计算设备。

在当前声明的单设备 backbone 调用图中，没有发现残留的 CPU/CUDA 混算入口。`earthdelta/bank_training.py:721` 的 `build_dynamic_bank` 仍无 device 参数，这个 API 延期项本身并未构成上述实际路径中的新 bug。不据此承诺任意模型分片、多设备输入或外部自定义调用都安全。

### 2.4 TF32 scope 修复：CLOSED；诊断不能承担完整根因证明

`earthdelta/static_adapter.py:1290` 保存 matmul/cuDNN TF32 设置，`:1297` 在 finally 恢复。CPU 探针在作用域内注入异常，确认两个设置均恢复原值。`verify_merge_equivalence` 的 scope 包括 branch 和 merged 前向，`:1348`、`:1352` **重复同一个 merged-path 调用**，没有拿两个不同模型计算冒充自一致性。

十二份记录的 self-diff 都为零，这支持这些指定冻结前向的局部可重复性。它不排除训练非确定性，也不单独证明 branch-vs-merge 残差只来自 FP32 非结合性。brief 对“排除硬件非确定性”的表述应收窄。另一个边界是 `static_adapter.py:1168` 的合并 delta matmul 在这个检查 scope 之外；没有本轮 GPU 对照证据证明它导致观察到的误差，因此不把这个可能性写成已确认根因。TF32 设置属于进程全局状态，当前 save/restore 适用于已声明的串行使用，并非线程局部数值环境。

### 2.5 merge_atol 公式修复：CLOSED，已完成要求的独立逐记录复算

`earthdelta/static_adapter.py:1664` 的实际公式为：

```text
effective_atol = max(float(floor), float(relative) * float(delta_max_abs))
floor = 1e-5
relative = 1.5e-3
```

`record_post_freeze_reference` 在 `:1712`、`:1713` 使用这两个默认系数，`:1749` 传入实际 `artifact.delta_max_abs`。旧 `merge_atol` 参数已从签名删除，未发现训练脚本以旧 keyword 调用；函数内部局部变量同名并非遗留 API。

独立读取 `artifacts/round2_cci/ed-r3-j4-bank-formal-v2-0922011307-2d8e12/expert{0,1,2,3}/run_record.json` 后，四份均为 Python 浮点值**精确相等**，没有用近似容差替代这一步核对：

| expert | delta_max_abs | 独立计算 atol = 记录 atol | 记录 max_abs_diff | atol/diff |
|---|---:|---:|---:|---:|
| 0 | 0.5175191760063171 | 0.0007762787640094757 | 0.00010704994201660156 | 7.251557 |
| 1 | 0.8071883320808411 | 0.0012107824981212616 | 0.00013583898544311523 | 8.913365 |
| 2 | 1.5924005508422852 | 0.0023886008262634277 | 0.0007569789886474609 | 3.155439 |
| 3 | 2.1599972248077393 | 0.003239995837211609 | 0.0002353191375732422 | 13.768518 |

原八个观测的 `diff/delta`，32-update 为 0.000562579–0.000792959，500-update 为 0.000143877–0.000257910；新系数分别提供至少约 1.89 倍、5.82 倍安全边际。新 formal 作业的最大比值 0.000475370 超过旧 formal 的范围，但仍被同一已注册系数覆盖，最小绝对容差余量 3.16 倍。旧作业记录仍采用固定 1e-4，不应要求其 retroactively 等于新公式。表中余量只用 atol，实际检查还使用 `rtol=1e-5`。

这支持“新公式在所记录的不同训练长度和新 delta 上通过独立实例检验”，不支持“已证明对任意输入/模型/horizon 普适”。当前 `static_adapter.py:1758` 仍硬编码 merge-check `steps=1`，输入是脚本 `:399` 的第一份样本；另一个 zero-edit 检查才使用四步。正式扩大运行范围时应对注册 horizon 和固定多输入面板复验，不能用四步 zero-edit 冒充四步 branch/merge 等价证明。

### 2.6 B15 及 selection 的增量性

空表、复数和 realized-a0 子项 **CLOSED**；完整正式 registry 资格 **PARTIALLY_CLOSED**。

`earthdelta/registry.py:620` 拒绝空表，`:398` 经 `_coerce_real_float` 拒绝复数。独立直接向 registry 传入 raw plan 的复数系数，得到 `REGISTRY_COEFFICIENT_COMPLEX`，不是被 EditPlan 构造器提前挡住后误归给 registry。`:568` 实际调用 `single_expert_plans`，`:584`、`:585` 检查被 rho 截断后的实际系数等于 expected a0；默认 0.25 并非测试专用断言。

`normalization_hash` 占位拒绝没有在 registry 实现，见 C09；选中行与实际系数对应还有 C11。原低层通用 planner 的空表/复数行为没有被悄悄改掉，正式入口加了检查。独立实际 diff 和逐函数源文本比较确认 `unified_select`、`select_plan`、`plan_from_prediction`、`_plan_from_finite_candidates` 四个既有函数一字未改；新增 registry 选择路径确为增量。

## 三、S0 的真实结果与失败处理

代码失败处理子项 **CLOSED**；真实官方 parity 门依然 **BLOCKED**。

原始文件：`artifacts/round2_cci/ed-r3-j1-s0gate-0921200335-0180dc/s0_gate_output/s0-gate-20260921t200508599573z/s0_gate_result.json`。按 `gate_criteria` 与 `criteria_details` 重算得到：

| 状态 | 数量 | 条目 |
|---|---:|---|
| PASS | 3 | identity_config_bound；ckpt_sha256_bound；normalization_parity |
| FAIL | 5 | source_identity_match；variable_coordinate_identity_match；manifest_identity_match；raw_input_binding；multistep_reference_present |
| SKIPPED | 8 | strict_load_zero_diff；version_identity_match；input_norm_binding；upstream_parity；multistep_parity；zero_edit_equals_official；no_state_leak；outputs_finite |

八项 SKIPPED 在布尔表中均为 False，详情明确写出“identity binding failed before model load”。`scripts/s0_gate.py:2295` 起先判断身份失败，再决定不加载模型/不执行后续；没有发现这个具体路径将已经发生的模型运行错误改写成 SKIPPED。原始结果为 `s0_gate_pass=False`、`status=gate_failed`、`verdict_committed=False`。这里的三个 PASS 不等于完成了真实模型数值 parity。

`scripts/export_upstream_reference.py:113` 的环境门在 xformers 不可用时以 BLOCKED/rc=2 退出；CPU mock 独立验证了该分支。CLI 在 `:824` 调用该门，官方路径在 `:403` 导入实际上游实现，未发现切换到本地 SDPA 来生成“官方”参照的分支。CPU 的 S0 正向合成 fixture 验证的是编排和身份合同，不替代官方 xformers 数值证据。

因此，本轮支持 brief 的“3 PASS / 5 FAIL / 8 SKIPPED”事实与诚实失败处理，不支持把 S0 标为整体通过。官方依赖/ABI 问题应通过可验证的兼容软件环境处理；仅更换 GPU 型号并不会自动修复依赖问题。

## 四、测试、覆盖范围与证据边界

已先完整阅读本目录 `AUDIT_BRIEF.md`、`AUDIT_PROMPT.md`，再审查实际工作区代码。保存的 [working_tree.diff](evidence/working_tree.diff) 为实际 `git --git-dir=.git --work-tree=. diff HEAD`。diff 不包含未跟踪文件，因此另直接阅读七份新生产模块、九份新测试及三个修改测试的所有新增/修改处。只向审计目录写入报告、探针与证据；没有修改生产代码、提交、推送或重新执行 GPU 作业。

### 4.1 已执行的验证

| 验证 | 结果 | 证据 |
|---|---|---|
| 指定七组 CPU 测试 | 307 passed，0 failed，0 skipped，17.21s | [targeted_tests.log](evidence/targeted_tests.log)、[XML](evidence/targeted_tests.xml) |
| 六组补充 CPU 测试 | 127 passed，0 failed，0 skipped，11.29s | [additional_tests.log](evidence/additional_tests.log)、[XML](evidence/additional_tests.xml) |
| 全 tests 目录收集 | 686 tests collected，5.72s | [full_collection.log](evidence/full_collection.log) |
| 独立 CPU 反例与公式手算 | 完成；输出包含通过项及已复现缺陷 | [主探针](evidence/independent_cpu_probes.py)、[结果](evidence/independent_cpu_probes.json)、[补充探针](evidence/additional_cpu_probes.py)、[结果](evidence/additional_cpu_probes.json) |
| 原始 GPU 记录只读复算 | 三作业、十二份专家记录 | [GPU 重算](evidence/gpu_records_recomputed.json)、[交叉核对](evidence/artifact_crosschecks.json) |

合计实际执行 **434 个仓库测试全部通过**。686 是收集数量，没有执行完整 686 项测试。

测试环境采用 `CUDA_VISIBLE_DEVICES=''`、`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`、仓库与 `.pydeps` 加入 PYTHONPATH、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=2`。主要命令可在仓库根目录重现：

```bash
export CUDA_VISIBLE_DEVICES=''
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="$(pwd):$(pwd)/.pydeps:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
python3 -m pytest tests/test_fs_static_adapter.py tests/test_bank_training.py tests/test_metric_contract.py tests/test_pilot_admission_b09.py tests/test_candidate_registry_b15.py tests/test_content_verification_b08.py tests/test_s0_gate_end_to_end.py -q
python3 -m pytest tests/test_bank_gradients.py tests/test_checkpoint_compat_shim.py tests/test_pilot_contract.py tests/test_normalization_policy.py tests/test_s0_gate_identity.py tests/test_s0_fail_closed.py -q
python3 -m pytest tests/ --collect-only -q
python3 codex_audit_round4/evidence/independent_cpu_probes.py
python3 codex_audit_round4/evidence/additional_cpu_probes.py
```

brief 所称“当前整树 collection 必因 xarray/absl 失败”，在本轮已配置环境中**未复现**：xarray 可导入，整树收集成功。没有把它列成本轮回归，也不在缺少历史环境复现的情况下断言过去的归因必错。主探针第一次缺少必填 convention、补充探针第一次未把单专家 max_active 设为 1，均是审计脚本自身的构造错误；已修正后执行通过，没有冒列为产品 finding。

### 4.2 测试断言是否被削弱

逐一检查九份新测试和三份修改测试后，**未发现为新代码通过而削弱既有断言的证据**。metric 测试是增补；normalization 增加校验并调整解释；S0 identity 将旧裸 digest fixture 改为带 schema 的 identity digest，与 Stage 1A 合同一致，并新增 legacy 拒绝用例。

存在新测试的覆盖不足：`tests/test_bank_gradients.py:514`、`tests/test_bank_gradients.py:581` 的 `count_hooks(model) == count_hooks(model)` 是恒真表达式，不能证明恢复，但旁边其他 hook/异常测试不因此失效。`tests/test_content_verification_b08.py:175` 的 fixture 通道数本身就不满足旧 shape 检查，不能独立证明“旧 shape 门会放行、内容门才拦下”的区分性。上述不足不等于既有测试被弱化。

434 个绿色测试没有覆盖固定面板训练资格、四进程共用 Fs、动态权重独立重载、准入失败消费、profiling 不变性和本轮 registry 近邻反例；这解释了为什么测试通过与 C01–C11 可以同时成立。

### 4.3 两种模式的可比性

`scripts/r2_fs_bank_train.py:840` 默认用 `MODE_UPDATE_CAPS` 设置更新数，`earthdelta/static_adapter.py:153` 定义 32/500 上限；Fs 与 bank 配置另外保存 mode 标签及 cap。检查生产路径未发现 mode 会自动改变 LR、Fs seed、clip、目标块、目标函数、训练步长或样本分配。十二份实际 Fs config 中，除 mode/max_updates/update_cap 外相关训练参数一致，样本 ID/顺序一致。

三个作业源码快照中的 `fit_static_adapter` AST 摘要一致，值为 `ac3414a8ce2cd67ad24f2e95813b1834c010f32596609e12f15e597c246599dd`。后续容差/诊断修改不能解释 Fs fit 自身的模式差异。不过不同作业不是同一初始化状态文件的严格续跑；没有全部初始权重/输入字节和训练内核可重复性的证明。可说“32 次未观察到此退化，500 次两次出现”，不能把“训练长度是唯一因果变量”写成已证明结论。

本轮保留已知边界：data_role 在训练入口要求 bank_fit，但没有跨实验使用账本；history 主要作为 provenance，并未成为多历史输入模型；内容物理范围带不能穷尽单位误标；官方 parity 未完成。没有审计不存在的 Stage 4 缓存/决策表，也没有重新作 novelty 裁决。

## 五、修复顺序、复验与 H100/5090 对照

当前总体字段：`round4_verdict = PARTIAL_PASS_WITH_OPEN_FINDINGS`；代码资格状态 `FAIL_IMPLEMENTATION`；Fs 独立字段 `MUST_FIX_BEFORE_STAGE4`；正式 Stage 4 `STOP`；官方 S0 `BLOCKED`。三个局部 bug 的 CLOSED 不覆盖任何一个独立资格结论。

建议按以下次序执行后续工作，本轮未代替实现者改代码或提交 GPU 作业：

1. **先修资格边界与产物合同。** 实现 C01/C04/C06 的拒绝条件及对应 CPU 反例：失败/陈旧/范围不足的 admission 不得进入训练，资格失败不得执行依赖阶段或发布可消费产物。
2. **建立一份共同 Fs。** 固定 bank_fit 面板和选择/回退规则，拟合并冻结一次 Fs，记录原始 checkpoint、norm、objective、输入与 Fs 内容 hash。worker 只加载该产物；不接受四进程分别重拟合后冒称共享参考。
3. **修复动态专家交付。** 在共同 Fs 上重训专家，隔离 profiling，保存真正动态权重，补齐正式 registry 身份；在独立进程验证重载前后输出和状态一致，再验证注册 horizon 的 merge/zero-edit 与非零响应。
4. **并行恢复官方 S0。** 准备匹配 torch/CUDA/ABI 的官方 xformers 环境，运行真实 upstream export 与 S0。软件依赖检查、数值 parity、Fs 优化稳定性分别报告。正式上游等价性前提下的研究结论要等待 S0，通过前的工程诊断明确标注范围。
5. **最后放行缓存与研究比较。** C01–C06/C09 及所依赖资格关闭后再启动正式 Stage 4；C07/C08 在相关 head/selector 比较前关闭，C10 保持串行合同，扩展候选表前修 C11。任何阈值/候选剔除规则在看 dev/confirm 收益之前冻结。

对用户提出的 5090 备选，建议把它作为**受控复验设备**，而不是默认的修复方案：

| 对照 | 固定条件与记录 | 能回答什么 |
|---|---|---|
| H100 同设备重复 | 同一保存的初始 adapter、checkpoint、输入内容、样本顺序、LR；记录 torch/CUDA/内核、TF32/deterministic 设置；至少重复两次，在 0/32/64/128/256/500 保存面板评分与梯度/更新量。 | 相同环境下是否可重现、何时开始偏离；不能仅凭 seed 宣称完全相同。 |
| H100 较小 LR/调度对照 | 固定其他条件，只改变一项优化配置，预注册停止和回退规则。 | 优化参数是否足以消除失稳。 |
| H100 与 5090 对照 | 先验证所选 torch/CUDA 与各 GPU 架构及官方 xformers 内核可用；共用保存的输入/初始权重和精度设置，分别记录实际软件差异。 | 失稳是否与设备/内核环境相关；若软件栈不同，不能把差异全部归为 GPU 硬件。 |

若 H100 的官方依赖环境暂时无法建立、5090 上兼容环境可用，可在 5090 运行同一已修复且有质量门的实验。但某一张卡“跑完且 exit=0”仍不够：必须同时满足共同 Fs、质量门、身份绑定、持久化重载及相应 S0 证据。没有必要等待 GPU 对照才能先修复本报告已经静态确认的实现问题。

## 六、交付文件与完整性

- [fs_fit_divergence_assessment.json](fs_fit_divergence_assessment.json)：独立第零部分结论与假设。
- [evidence/audit_findings.json](evidence/audit_findings.json)：C01–C11，精确 source line、证据状态、最小动作及关闭判定。
- [round4_summary.json](round4_summary.json)：总体结论、B05–B15 disposition、四份 atol 精确复算和 Stage 4 阻断项。
- [evidence/final_validation.json](evidence/final_validation.json)：交付 schema/引用核验、源文件 SHA256、HEAD/diff 与只读记录的完整性复查。

本轮读到的原始 GPU 作业仅作为已有证据；本轮没有提交 H100/5090 任务，没有修改这些原始记录。最终代码状态与审计起点的 48 份 Python 源文件摘要及 HEAD 对照见完整性记录。
