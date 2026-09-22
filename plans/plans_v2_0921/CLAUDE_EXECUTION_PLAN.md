# EarthDelta 下一轮执行计划：交给 Claude 的实施合同

日期：2026-09-21。仓库：`/mnt/afs/260010168/EarthDelta`。

本计划把 [NEXT_STEPS_REVIEW.md](NEXT_STEPS_REVIEW.md) 转成可执行任务。与 [CLAUDE_PROMPT.md](CLAUDE_PROMPT.md) 和 [CCI 启动手册](cci/README.md) 一起使用。2026-09-21 用户已明确允许使用 CCI 任意资源，主要为提交任务形式的 4 卡 H100，允许多个任务排队，资源免费；同时允许按需使用所提供的 SiliconFlow API。本修订取消上一版“默认只完成 CPU 阶段，再等待 GPU 授权”的限制。

当前范围是把正确性修复、真实启动、训练、开发集效果评估和条件性后续实验执行到底。阶段推进取决于证据和可操作的运行护栏；不为已经授权的 GPU、依赖准备、模型加载、必要数据读取、训练或任务提交重复确认。确认集仍需在协议冻结后一次使用，不因资源免费而提前揭盲。真实实验结果按实际发生记录，不能把启动探针当作科学 S0。

## 1. 目标、依据与本轮边界

首个研究交付：在一个可信、冻结的小专家库上，得到包含 **Fs/no-edit、最佳固定候选、regime、direct-gain、完整 oracle** 的原生收益与成本表。先判断动态空间能否被合法信息利用，再支付双头与输出纠错实验成本。

首个工程检查点：完成 `R2-P0-01` 的最小正确性补丁和合成 CPU 验证，随后直接提交真实 S0 任务。它不要求关闭全部审计 finding，也不要求提前实现所有实验脚本；CPU 检查点不是本次任务终点。

来源及优先级：

1. 用户当前及后续明确指令，以及运行环境的实际约束。
2. 本计划和配套 prompt：规定本轮顺序、默认设计、阶段边界与验收。
3. `unpacked/variant_2/EarthDelta_Codex_Execution_Plan.md`、`experiments/BASELINES.md`、`experiments/P0_SURVIVAL_EXPERIMENTS.md`、`STOP_CONDITIONS.md`：提供完整科学合同。
4. `NEXT_STEPS_REVIEW.md`：解释取舍；variant_1 只吸收 cheap-policy 优先顺序和 Fs 执行说明。
5. 历史审计提供缺陷证据；旧规格保留为背景。

不合并三份原始任务 JSON，不修改 ZIP 或 `unpacked/` 内原始文件。本计划中的 A/B 是同一任务的内部检查点，不创建并行的第二套任务状态系统。

默认保持：一个 backbone、每个模型实例内串行执行、无 activation-checkpoint replay、memory/JEPA/gain calibration 关闭、有限离散候选、retrospective perfect-analysis。4 张 GPU 可由独立进程各自加载模型并行训练专家/seed 或模拟互不重叠的起报分片；不在多个线程间共享同一个带 hook 的 bridge。当前不做端到端系数控制器、连续 QP、新天气 encoder、动态 rank、谱训练、大范围文献重审或全量数据下载器重构。

## 2. 启动事实与工作方式

编写时核验：

- 分支：`audit/round2-review-20260921`。
- HEAD：`403b55db65f4c35c1a85d0794ad0de2765b07d96`。
- 已审源码：`fb767f7f6efbc428be39c9ad84f5905331d6e40f`；上述两个提交的 `earthdelta/`、`scripts/`、`tests/`、`research_spec_v6.yaml` 无差异。
- 工作树存在用户未跟踪内容：`artifacts/`、`checkpoints/run_multi_year_pull.sh`、`plans/plans_0921/`、`plans/plans_v2_0921/`。不得清理、覆盖或停止其中关联的用户进程。
- 历史 CPU 成绩为 283 passed / 7 skipped；这是旧日志，不是新代码验收结果。

Claude 每次开始先重新核验。新的 HEAD 或工作区变化先作差异归因，不自动 reset/checkout，不把旧成绩套给新源码。已知未跟踪文件不是重新请求开工的理由；执行期间出现无法归因的源代码变更，先暂停相关编辑并报告。

当前普通 Git 命令曾遇到 ownership 错误。在仓库中可用以下只读方式定位，不全局放宽配置：

```bash
git --git-dir=.git --work-tree=. rev-parse HEAD
git --git-dir=.git --work-tree=. status --short
git --git-dir=.git --work-tree=. diff --stat
```

按依赖顺序修改，优先复用已有符号。只有真实调用链需要时才新建小模块。每一检查点完成后自查 diff、运行相关验证并更新同一工作记录，不为每个 helper、测试或日志另行询问。

如果 Claude 本机启用了 `~/.claude/CLAUDE.md` 引用的 Fableplan 条件路由，按其实际适用条件工作；不宣称未发生的模型调用。依赖任务顺序执行，不让多个执行者同时修改同一文件。此项目计划不修改 Claude 的全局配置。

## 3. 阶段与准入

| 顺序 | 对应任务/检查点 | 主要交付 | 进入下一步的条件 |
| --- | --- | --- | --- |
| 0 | 启动核验 | 当前源码差异、资产元数据、执行记录目录 | 能明确本次改什么、验证什么 |
| 1 | R2-P0-01 A：CPU 正确性 | 最小补丁、实际 orchestration 反例测试、CPU 日志、待测真实配置 | 当前路径 CPU 验收通过；真实资源范围与 cap 明确 |
| 2 | R2-P0-01 B：真实 S0 | 独立官方参照、完整身份绑定、一步/多步对齐证书、推理成本 profile | 所选 F0 的真实 S0 通过；bank/cache 预算冻结 |
| 3 | R2-P0-02 A：Fs 与 bank | 强静态参考、小库、训练来源、非零/恢复/多样性资格 | 库合格，数据角色/registry/Q 冻结 |
| 4 | R2-P0-02 B：完整缓存 | 全候选实际结果、oracle/static 开发表、成本账本 | 有完整缓存与可信动态余量；为 cheap-policy 保留预算 |
| 5 | R2-P0-03 A：cheap legal | 独立评估的 regime/direct-gain、已保存动作、实际收益 | 合法信号值得继续，或明确限定的一次加强挑战 |
| 6 | R2-P0-03 B：dual 增量 | 压缩诊断、四格、公平 direct 对照、少量标签曲线 | 合法策略有价值；是否保留双头主张单独判断 |
| 7 | R2-P0-04：输出挑战 | actual/virtual/joint/feedback 的效果与成本 | 有待检验的合法动态策略；单列本阶段预算 |
| 8 | 最终确认与 R2-P0-05 | 一次确认结果，或提前停止的只读汇总 | 最终确认须先冻结全部比较；汇总可在任何终点进行 |

Claude 的执行范围覆盖完整路线。完成 0/1 后自行推进真实 S0、Fs/bank、缓存和 cheap-policy；dual/E3 按开发证据条件推进。原 variant_2 中“仅第一任务解锁、后续 GPU/安装/训练为 false”是旧授权快照，保留原文件，但在工作副本中依据本次用户指令登记新范围，不再把旧锁作为阻塞。

本地控制节点没有 GPU 不构成阻塞，应使用 ACP 提交队列。缺必要依赖时在专用环境中安装并记录版本，不能伪造模块或偷偷切换官方注意力实现。只有实际作业/资产失败、科学证据触发停止、无法安全修复的外部问题，才停止受影响路径；先完成其它可做工作再报告准确原因。

## 4. 阶段 1：最小正确性补丁，按以下顺序实施

### 1A. 显式归一化与统一身份入口

优先位置：

- `earthdelta/bridge/stormer_bridge.py`：`NormalizationContract`、`digest`、bridge 配置。
- `scripts/export_upstream_reference.py`：`run_official_inference`、`export_reference`、CLI。
- `scripts/s0_gate.py`：`get_paths`、`verify_ckpt_sha256`、`verify_upstream_parity`、`run_s0_gate`、CLI。

工作要求：

1. 增加显式 `official_zero_diff_mean` policy，独立测试非零差分均值输入仍按官方零均值语义执行。旧行为保留明确 legacy 标记和兼容回归，不能静默改变历史实验语义。
2. policy 进入 normalization 身份；新 pilot 入口必须显式选择 policy。变量序、单位、坐标、interval、形状和数值来源不能仅作为日志展示。
3. exporter 与 gate 消费同一配置和预期身份，消除 ps4 exporter / ps2 consumer 默认错配。
4. exporter 使用 pinned 官方网络与实际 `GlobalForecastIterativeModule.forward_validation` 路径，独立构造官方 transforms。不能导入受审 NormalizationContract 生成参照；缺依赖时如实 BLOCKED，CPU 阶段仍应完成可测的入口与身份校验。
5. 实际参与计算的 checkpoint 字节、raw/normalized input、官方 commit、norm、变量/坐标、dtype、exact shape、interval 和全部登记 rollout 输出均需绑定。随机 64 位字符串不能充当预期身份核验。
6. 新入口可复用原计划的 `--config/--authorization/--task-state`，但只实现必要字段/调用，不建设通用权限平台。原包初始状态应复制到工作目录，原文件不改。

验收：由真实 gate 调用链驱动的合成测试能拒绝 checkpoint/norm/grid/input/shape 任一错配和缺少任一多步结果。测试不能只在测试文件自建一个 PASS 字典。合成参考只验证 gate 控制流，输出必须标为 synthetic，不能生成可放行真实训练的 S0 证书。

### 1B. 原生评分合同

优先位置：`earthdelta/metrics_contract.py`；后续 head 校准在阶段 6 才修改。

对原生注册目标 `[B,H,V,Lat,Lon]`，定义：

```text
L(Y,F) = sum(q * ((Y-F)/s)^2) / sum(q)
e = Y-Fref
u = Fa-Fref
g(a) = L(Y,Fref)-L(Y,Fa) = 2 e^T Q_eff u-u^T Q_eff u
Q_eff = diag(q/s^2) / sum(q)
```

要求：q 在 H/V/Lat/Lon 上归一化一次，保留 B 和候选 K；真实浮点、有限值、非负 q、正 scale、显式广播和每个实际目标的正分母都在入口检查。先保留真值精度，再用 float64 形成误差/累加。missing 默认拒绝，不静默删候选或改变分母；合法零权重掩码允许，全零目标拒绝。

保留旧逐变量/逐时效诊断 API，不让其冒充全目标。最终 RMSE 在正确聚合平方误差后开根号。

验收：非均匀 lead/area/variable 权重下的端点损失差与解析 gain 一致；两个 lead 权重不等的测试能区分原先被错误平均的候选；非法 scale/weight/broadcast 被拒绝；FP64 参照独立计算。若实际有限端点路径不调用 batched Gram，B04 暂缓，不加连续优化器。

### 1C. 小切片准入、bank 梯度与 gate 收尾

优先复用现有 contract；必要时新增 `earthdelta/pilot_contract.py` 与 `scripts/r2_pilot_preflight.py`。

- 加载入口验证真实 time 精确索引、history/target 端点、通道/坐标顺序与有限内容；阶段 1 用临时合成资产测试，阶段 2/3 才读取批准的实际切片。证明 marker/shape 不足以放行，避免改造整个 Zarr 下载器。
- 正式 registry 验证非空、实浮点、唯一 ID、显式 no-edit、有限/幅度/support/count/窗口一致；可在后续首次消费 registry 前接入，但不能只有无人调用的校验类。
- 固定非零 tuple 系数下，非零合成 bank 的 loss 能反传到 A/B，骨干不更新；`zero -> nonzero -> zero` 恢复；正常与异常退出后 hook 清理。B03 caller 系数梯度暂缓，不能伪称修好。
- `run_s0_gate` 必需计算或身份产物写入失败不得保留 PASS；已分类的非必要清理警告独立记录；报告缺值安全输出 N/A。测试真实 orchestrator 和 report。

### 1D. 验证与交付

先读待跑测试及 fixture，使用明确的合成测试清单。`CUDA_VISIBLE_DEVICES=""` 或 `-m "not slow"` 不等于完全隔离真实资产：`tests/test_bridge.py::test_load_real_normalization` 当前会读真实 NPZ，三个 checkpoint 测试则已有 slow 标记。

建议先核查并选择 `test_metric_contract.py`、`test_s0_fail_closed.py`、`test_differentiable_rollout.py` 的相关节点，再加入新增的合同/orchestration 测试。新增测试文件可沿用原计划命名，也可放入已有测试文件。只有读过后确认合成且符合资源上限的节点才执行，不照抄全套测试命令。

记录命令、节点、退出码、通过/失败/跳过与原因。先运行相关回归；仅在新增改动或失败需要时扩大范围。不能删旧断言、降低容差或改变问题定义来让测试通过。旧测试若只证明较弱性质，可保留并新增真正反例，不以测试名称作验收。

阶段 1 完成条件：上列实际路径的必要反例通过，diff 自查完成，真实运行 CLI 已实现且缺项明确。此时的中间状态是 `code_status=CPU_VALIDATED`、真实 S0 `run_status=NOT_RUN`、`research_verdict=NOT_EVALUATED`。保存检查点后继续配置并提交真实 S0；只有实际缺项不能解决时才标 BLOCKED，不能因为完成 CPU 测试就结束整个任务。

## 5. 阶段 2：真实 S0 与资源冻结

真实运行前由 Claude 核验并填齐：所选设备/环境、可信 checkpoint 路径和来源、normalization 资产、非 confirm 的实际输入切片、一步/多步配置、dtype/tolerance，以及本阶段的调用数/耗时/内存/存储/重试护栏。这是执行准备，不是再次向用户索要 GPU 权限。

默认选择 ps4、官方 pin `58dfee5a6037399a40fefd492bc00421e0c885a8`、6h interval；核验 1 步和 4 步。若 pilot 的 12 步/72h 路径此前未覆盖，在首次正式 72h 评分前补齐该推进路径验证。不能用 ps2 成绩给 ps4 放行。

先做所选 F0 的独立官方对齐，再测推理耗时、峰值资源和缓存输出量。单次训练步 profile 单列为 bank 准备测量，记录显存/速度与更新内容；它也在当前资源授权内，不要再次停下来请求。

所有用于 S0/debug/profile 的起报标为 exposed，不进入最终 confirm。容差来自数值重复性、dtype 和规范要求；不能看偏差后任意调宽。

阶段 2 交付：实际输入证书、官方 reference manifest、真实 S0 数值/身份结果、推理/训练步 cost ledger，以及阶段 3–5 的运行配置。虽然无调用费用，仍须记录 GPU 时间、前向数、磁盘和失败重试以便比较方法；保证时间和存储投入能覆盖 cheap-policy，不只产出 oracle。

### 5.1 已核验的 CCI/ACP 启动方式

已实现的提交工具为 `plans/plans_v2_0921/cci/submit_job.py`；使用现有 SCO 认证、`share-space` 工作区、`share-cluster`、`n6ls.iu.i40.4.32c512g`、1 worker/4×H100、32 CPU/512 GiB 规格及共享 `/mnt/afs`。镜像和挂载参数在工具中固定，详细命令见 `cci/README.md`。

提交工具会生成独立源码快照、命令/源码哈希、作业 ID 和共享日志，避免排队时被后续编辑改变代码。它不自动实现任何天气实验，也不会将作业退出码 0 等同于 S0 或科学 PASS。新建实验入口后，用 JSON argv 提交该入口；默认调用仅为硬件/依赖/样本探针。

当前 SCO 普通分发路径下载组件返回 404；实际验证有效的局部调用是 `SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco ...`。这只是使用现有客户端的调度方式，不改全局配置，也不改变身份认证。提交工具已包含这一局部设置。

各 GPU 进程写独立 shard，CPU 控制进程验证完整性后合并。同一任务模型实例不共享；并行 job 可排队。独立专家、缓存分片、head seeds 可以并行；依赖 S0/Fs/bank 的后继任务只在前驱证据文件通过核验后提交，避免占用 GPU 等待前驱。

### 5.2 第一轮运行护栏与推进节奏

下列是防止挂起和无边界搜索的起始配置，不是 GPU 费用预算，也不是机器吞吐或统计功效的实测结论。Claude 在开发结果揭示前可按 profile 调整并记录依据，无需为同一已授权路线反复确认。

- 探针 15 分钟；S0/环境准备单作业 2 小时；bank/cache 作业分片控制在单作业 8 小时内，支持断点续跑。每种明确的基础设施失败最多自动重试 2 次，不对确定的正确性失败盲重试。
- 先完成 1–2 个起报的真实前向/训练步 profile，再做 8 个 exposed 起报的完整候选调试；不能用这 8 个样本宣称效果或作为 confirm。
- 流程通过后，在预先冻结的 policy_dev 上先取 128 个登记起报做开发筛查。若有效过程数或功效不足，按提前写好的开发采样方案扩到 256/512；样本是否独立由实际时间/过程决定。confirm 的 N 单独依据功效设计，不能照抄这些起报数。
- Fs 和每个专家先用至多 32 updates 检查梯度、吞吐和训练方向；正式小库起始上限为各 500 updates，开发结果未查看前依据训练侧收敛性与 profile 可调整。未充分训练只能报资格不足，不判科学 STOP。
- cheap 策略每方法起始最多 6 个 HPO 配置；先一个 seed 跑通，再用 3 个固定 seed 检查训练稳定性。双头若进入比较，使用一致的标签/仿真与 HPO 口径。
- 默认维持 2 个运行/排队作业的调度窗口，独立任务可按集群可用性增加。这个窗口便于管理和避免重复提交，不限制用户已授予的 CCI 资源范围。

同一 confirm 必须在阈值、模型和分析代码全部冻结后只开启一次。前面的开发扩样不属于确认检验，不能写成预注册 confirm PASS；达到既定科学停止条件后停止该路线，即便 GPU 免费也不靠无限扩库调参制造收益。

## 6. 阶段 3–5：优先完成第一张决策表

### 6.1 设计默认值与必须补测的字段

以下是建议的首轮设计值，尚非已执行配置。允许在首次开发评分前依据资源/数据证据作一次明确调整并冻结；看结果后修改则成为新实验。

| 项目 | 首轮设计默认值 | 冻结依据 |
| --- | --- | --- |
| Backbone | 一个 ps4 checkpoint，6h 推进 | 与真实 S0 相同身份 |
| 动态 bank | K=4、每专家 rank=4、目标块 18–23 | 先核对模型层与单训练步 profile；不足时在看开发结果前缩为 K=2 或降低 rank |
| 候选 | 显式 no-edit + K 个 singleton，max_active=1 | 不使用旧 `pilot_plans()` 的组合表 |
| 幅度 | `a0=0.25`、`rho=0.25`，训练与缓存一致 | 沿用当前受控接口的有界域；显式构造，不依赖 helper 静默裁剪 |
| Hold | 4 个 6h 步，之后从编辑状态继续 Fs | 始终使用冻结 Fs continuation |
| 主目标 | 24h 原生注册变量的面积加权、scale 标准化平方损失 | 变量/单位/scale 来源/Q 在结果前固定；不由赢家表现选择变量 |
| 诊断/guard | 6h 诊断；72h 后续损害 guard | 与 24h 分开报告，不暗中三时效平均成另一主目标 |
| 输入信息 | 当前、前 6h、前 12h 的登记历史；memory off | 实际时间端点存在，retrospective 条件明确 |
| Cheap context | 优先复用合格现有特征；否则各通道固定 2×4 空间池化并串联历史，训练侧标准化 | 所有方法相同；不使用未来真值或实际候选响应，不新增学习型 encoder |
| Cheap 方法 | no-edit、best-static、fit-only regime、ridge direct-gain | 全部共享 registry；HPO 在训练侧、统一上限 |
| 复杂选项 | 无 pair、负幅度外推、在线连续搜索或 reference-token preview | 首轮控制成本，结论仅覆盖测试动作集 |

现有 `single_expert_plans` 会把系数裁到 rho，因此必须在 registry 中保存实际系数，断言等于预期 a0。这里不要求为了 helper 风格重写整个公共接口。

不能凭空填写的字段：checkpoint 实际哈希、真实时间划分、过程定义/guard、GPU/存储 cap、训练 updates、N、HPO 次数、有效过程数、统计方差、delta_min、MDE、alpha/power 和正式 CI 规则。写入同一配置，注明 measured / chosen / pending。未知时可以做已许可的实现或描述性开发，不能给正式科学 PASS。

### 6.2 Fs、bank 与数据角色

先冻结真实时间角色：bank_fit、policy_dev（用于策略 crossfit 和有限调参）、confirm。必要的内部训练/验证块在各自角色内明确分开；bank 选择也不能利用外层策略评估块。history/target 重叠和时间依赖需要登记 purge/guard；不把候选、格点或不同 lead 当独立样本。

Fs 在 bank_fit 侧训练/选择并冻结；动态专家在其上训练，保留 F0 背景分数。静态 adapter 可在独立模型副本合并，或使用经过等价性验证的 always-on 分支。no-edit 和 hold 后必须是 Fs；冻结 Fs 后重新记录参考身份及零编辑等价性，不沿用 F0 的数值参照。

bank 先说明专家差异的训练依据，优先使用 fit-only 的合法 regime/数据分组与共同目标，不做一轮无边界专家构造搜索。每个专家记录训练数据、更新数、loss、非零响应与多样性；零 B 初始状态是正常初始化，未训练零库不是科学失败。只允许按事前资格规则处理无效库，不按评估收益删掉专家。

固定系数训练验证 A/B 梯度即可，不等待 B03。若 24h 反传在当前串行/no-replay 模式下不可行，先提供 6h 训练、24h/72h 评估的明确规格调整和成本；不能暗中开 activation checkpoint 或改变训练时域后仍引用旧实验身份。

### 6.3 缓存与 oracle

每个登记起报串行运行 Fs 和全部冻结候选，统一输入、原生目标和机会分母。保存全部候选状态、实际 loss/gain、耗时/调用/失败/回退成本及向量结果或可兑现的重算路径；winner-only 表不合格。

truth/outcomes 与 legal feature 文件分离。缓存 manifest 绑定源码、Fs/bank、registry、Q、split、输入与坐标。失败不能被忽略或填 0；缺任何必要候选则不允许完整 oracle PASS。

存储预算必须按实际维度计算。FP32、69×128×256、5 候选含 no-edit、3 个时效、1,000 起报，仅候选预测约 135.66 GB，真值另约 27.13 GB。它是算术示例，不是样本量要求或硬件测量。按起报/候选分片；压缩保真与重算开销需验证。

先在 policy_dev 生成 oracle/static 开发表，不打开 confirm。最佳固定候选在每个外折训练侧选择，对该外折评估起报固定；最终确认前再冻结用于 confirm 的单一策略。

### 6.4 Cheap legal screen

复用缓存，先跑低容量 regime 和 ridge/direct-gain。同一个外层被评估起报不能被 bank 看过；各折标准化、聚类、输出基和 HPO 只接触训练侧。默认采用有时间顺序和边界隔离的 crossfit，不能先全量拟合特征再称 OOF。

predict 阶段只读合法 features、模型和 registry，先持久化预测/动作及 hash；score 再读真实 outcomes。测试通过改变/污染离线未来 payload 不能改变已经产生的动作，并核查真实 predict 数据访问路径。

第一张表至少包含以下列：

```text
method | evidence_role | n_issues | n_processes | attempted/complete
native_primary_loss | gain_vs_Fs | gain_vs_frozen_static | paired_CI
guard_72h | harmful_edit_rate | no_edit_rate
offline_cost | online_cost | failure/fallback_cost
```

oracle 标为 hindsight；前三种合法静态/动态基线和 direct 分开。no-edit gain 为 0，但计算成本非零。按过程配对统计，同时预先声明点估计按起报等权还是过程等权。

判决：合格完整库无有用 oracle 空间，停止当前库；静态解释可用空间，转静态；oracle 有空间且 cheap 有可信合法信号，进入双头比较；oracle 好但 cheap 无信号，只在瓶颈诊断支持下给一次有 cap 的加强挑战。CI 宽则 INCONCLUSIVE，不用新增模块掩盖证据不足。

## 7. 阶段 6：分解价值，按诊断顺序投入

1. **先检查表示。** 在 fit-only 固定线性 D 下，用真实 e_D/du_D 选动作，再按原生结果结算，与原生 true/true oracle 比较。若压缩已丢掉选择空间，先标明表示瓶颈，不急于扩大 head。
2. **复用 head。** 在 `earthdelta/heads.py::ComposedPredictionHead` 及其真实 caller 中关闭 memory/JEPA/独立 gain calibration，显式区分 analytic 与 calibrated 输出；no-edit 响应和得分为结构零。仅在实际用到时修 B04。
3. **比较同资源 direct。** 输入、候选、标签 ID、仿真资格、输出读出与 HPO 额度一致。双头使用额外无真值响应预训练时，给 direct 实际可运行的同等辅助训练对照；一条 truth 给全部候选打标签，不按 K 倍惩罚 direct。
4. **四格实际选动作。** true/true、true/pred、pred/true、pred/pred 全部用真实候选结果结算。true 格只用于离线诊断，不能算可部署成绩。报告 response-span/排序误差，不以全场 e 的低 R² 单独停止。
5. **再做有限标签曲线。** 先一个预算点验证流程，再用少量预登记预算点、同 label IDs 和固定 seeds。分别计 given-bank 与含 Fs/bank 的 end-to-end 标签及计算成本。

均值双头在压缩 context 下可能遗漏条件协方差与响应方差；若诊断支持，可在同缓存中给一个条件矩/决策投影变体有限预算。不是首轮必做多架构清单，也不能用它无限救活失败主张。

结论分开写：合法动态策略是否有效；双头相对 direct 是否有最低有用的额外效果/标签/成本优势。差异不显著不等于无价值；只有预登记界限与 CI 足以排除增量才转 direct。若 direct 有效，缩窄主张后仍可进入 E3。

## 8. 阶段 7–8：输出挑战、确认与结束

只有前面有值得比较的合法策略才实现 E3。actual edit、virtual、联合输出纠错与逐步 feedback 使用同 I、Q、作用窗、机会分母和资源口径；feedback 仅消费自身预测状态，hold 外不继续注入修正，但从改变后的状态继续 Fs。

virtual 需要原生 du 或经过验证的固定 lift；summary-only 不冒充全场预报。计 reference preview、必要重放、bank/head 训练、缓存、adapter、decoder/lift 和失败回退成本。参数数目匹配与实测成本前沿分开报告。

最终确认前冻结全部模型、候选、阈值、主比较、guard、统计脚本与失败处理，再一次批量评估 confirm。若 E1 已经看过 confirm 并影响设计，则它变为 exposed，后续必须另留独立确认。不得在看到 CI 后不断补样直到显著。

正式阈值采用 variant_2 的 STOP_CONDITIONS：delta_min 有独立价值依据，MDE 描述辨别能力；下 CI 超过阈值才支持最低价值，上 CI 低于阈值可排除该价值，覆盖边界为 INCONCLUSIVE。明确一个 primary contrast 或多重比较规则。

任何提前停止都运行只读 R2-P0-05 汇总已获得证据，后续任务标未到达，不为“完成计划”继续花资源。输出包含继续/静态转向/direct 转向/输出纠错转向/证据不足的具体原因和适用范围。未验证组合不能声明组合失败，未完成 E3 不能声明实际编辑必要。

## 9. 运行记录与阶段交接

使用一个工作目录体系，例如 `artifacts/round2_next/<唯一运行ID>/`；原始交付文件不在此处更新。不要覆盖既有运行目录。

每阶段最低记录：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 当前阶段实填配置、值来源、仍未确定字段；后续冻结 hash |
| `source_state.json` | HEAD、基线、工作区差异、相关源码内容身份 |
| `commands.txt` 与日志 | 真正运行的命令、退出码、测试/模拟输出 |
| `decision.json` | task/substage、code/run/research 状态、范围、证据路径、缺项、下一步 |
| `HANDOFF.md` | 已完成、未完成、当前阻塞、下一条命令与资源需求 |

可复用原有等价文件，避免为了命名重复生成多套报告。测试日志不能伪装实验 metrics；未运行的值为 null。涉及真实执行时，再增加身份/S0/bank/cache/阈值等该阶段确需的 manifest，不提前建设完整报告平台。

code/run/research 三轴沿用主计划。阶段 1 是 `CPU_VALIDATED + NOT_RUN/BLOCKED + NOT_EVALUATED`；开发期标 DEV，确认结果与 DEV 分开。证据和人工资源范围是不同字段，不能用一个 PASS 合并。

首次真实运行前由 Claude 填写可审查的运行表并直接执行：设备、checkpoint 与输入路径、最大前向数、更新数、起报/候选数、墙钟时间、内存/显存、磁盘、重试和 E2 预留额度。已知字段填实际值，未知字段通过有界 profile 测得；不能编造硬件小时，也不能把填表转化成重复的 GPU 授权请求。

API 可按需用于代码分析、日志归纳或辅助交叉检查。入口为 `https://api.siliconflow.cn/v1`；模型候选为 `zai-org/GLM-5.3`、`deepseek-ai/DeepSeek-V4-Flash`、`Qwen/Qwen3.8-27B`。先核查实际可用性；不得把语言模型回答当作天气真值、实验结果或科学裁判。密钥通过私有环境变量 `SILICONFLOW_API_KEY` 提供，不能进入文件/命令行参数/日志或 Git；不因天气实验不需要 API 而强行引入调用。具体辅助示例见启动手册。

默认不自动 commit/push。需要提交时只列出本次文件，不 `git add .`；不得上传 checkpoint、天气缓存或未筛选 artifacts。既有会话若已授权相应 Git 操作，按该具体范围执行。

## 10. Claude 的阶段记录与最终回复验收

阶段记录持续写入；最终回复须包含：

1. 改了哪些使用路径，以及它们对应的缺陷；附文件位置。
2. 实际测试命令和结果、跳过项及理由。
3. 哪些 finding 已在本路径修复，哪些仍明确暂缓，不能把暂缓写成关闭。
4. 分开的 code/run/research 状态，真实 S0 是否执行。
5. 可点击的补丁/日志/配置/交接记录路径。
6. 实际 ACP job ID、队列/运行/结束状态、关键失败及处置；真实收益表与成本账本。若科学实验确实无法完成，提供可核验阻塞证据和下一条命令，不将“本地无 GPU”或旧模板锁作为理由。

不得只回复计划或建议，也不得提前声称 oracle、标签效率、双头优势或 novelty 成立。
