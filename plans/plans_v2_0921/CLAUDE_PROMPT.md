# 给 Claude Code 的执行 Prompt

适用于能读取仓库并通过 CCI/ACP 提交作业的 Claude Code。已按用户 2026-09-21 的新指令修订：允许使用 CCI 任意资源，主要是提交任务形式的 4 卡 H100，可提交多个任务排队，资源免费；按需可使用 SiliconFlow API。执行目标包括真实实验和效果评估，取消旧版“仅做 CPU 再等待 GPU 授权”的限制。

---

你是 EarthDelta 项目的实施负责人。此前已完成两轮审计和一次 Pro 方案整合，现在请依据已收敛的执行计划完成正确性修复、CCI 真实启动、训练与效果评估，直到得到有依据的继续/转向/停止结论。不要重新进行一轮宽泛审计，不要只给我另一份计划，也不要做到 CPU 测试就结束。

仓库路径：

`/mnt/afs/260010168/EarthDelta`

主执行计划：

`/mnt/afs/260010168/EarthDelta/plans/plans_v2_0921/CLAUDE_EXECUTION_PLAN.md`

## 一、先按顺序阅读并建立上下文

1. 完整阅读主执行计划 `CLAUDE_EXECUTION_PLAN.md`。
2. 阅读同目录 `NEXT_STEPS_REVIEW.md`，理解为何优先真实小库、完整候选和 cheap legal screen，以及哪些旧审计阻塞项已按使用路径收窄。
   再读 `cci/README.md` 和 `CCI_VALIDATION.md`，使用已实际验证的提交/监控/日志路径；后者只证明已列出的启动与回归事实，不证明天气技能。
3. 阅读 `unpacked/variant_2/IMPLEMENTATION_AUDIT_ACTIONS.md`，以及 `unpacked/variant_2/EarthDelta_Codex_Execution_Plan.md` 的 scientific contract、R2-P0-01、状态/资源部分。后续进入新任务前再读取该任务和对应实验/停止条件，不要求开工前重复通读三套 ZIP。
4. 按需读取 `codex_audit_round2/results/ROUND2_REPORT.md`、`audit_findings.json` 和相关反例源码；亲读将修改的代码与测试。历史报告是证据，不能代替当前调用链检查。

本次指令与 `CLAUDE_EXECUTION_PLAN.md` 是当前实施范围；variant_2 是补充科学合同。原包的“只有第一任务获批”“GPU/训练/依赖安装为 false”是旧授权快照，本次明确覆盖它们；在工作副本中登记此次授权，无需重复确认。保留原始 ZIP、解压包和历史审计，不能改写历史证据。科学准入与确认集一次使用要求仍然有效。

遵守你实际适用的本机/仓库指令。若本机条件性 Fableplan 路由生效，按其要求作必要的计划模式/任务路由，不虚报模型；仅为当前实施核对必要顺序，不重新设计研究。依赖任务串行执行，多个执行者不得同时改同一文件。如果环境强制审批造成等待，指出具体规则和已完成工作；不要添加规则没有要求的额外确认。

## 二、当前直接执行的范围

请先完成 **R2-P0-01 A：最小正确性补丁＋合成 CPU 验证**，随后连续推进真实 S0、Fs/专家库训练、完整候选缓存、cheap legal 效果评估，并按证据条件进入 dual 与输出纠错挑战。

我允许你使用 CCI 中任意可用资源，主要使用 4×H100 ACP 作业；可以提交多个任务排队，资源免费。自主完成必要代码修改、CPU/GPU 测试、可信 checkpoint 加载、实际天气数据读取、环境依赖准备、profile、训练、候选模拟和结果评估。无需逐个 helper/文件/测试/作业向我确认，也不要读完文档后仅问“是否开始”。确认集只有在完整协议冻结后才能一次性访问；这也是当前路线中的任务，不另设常规 GPU 授权障碍。

本地控制节点没有 GPU 时，通过队列提交任务，不把它报成整个项目无 GPU。依赖缺失时建立专用环境并安装必要包，记录版本和 CUDA ABI；不能伪造导入模块或换成 SDPA 后声称官方 xformers 已通过。优先复用现有 checkpoint 和数据，只按需要补齐资产，不能破坏用户正在进行的数据拉取。每作业设置超时、日志和有限重试，按 profile 填写运行护栏；免费资源不等于无限调参或可忽略计算成本。默认不自动 push。

这里的最低研究交付是：来自真实运行的 Fs/no-edit、最佳固定候选、regime、direct-gain、oracle 原生收益与成本表，或者足以解释为何未能生成该表的真实失败证据。不能用硬件 smoke、包校验或合成测试代替天气实验。后续 dual/E3 若被科学停止条件切断，直接汇总已有证据，不为完成清单继续无意义训练。

### 已提供的启动工具

在仓库根目录可直接执行下面命令提交独立源码快照的 4 卡检查任务；先阅读已有验收记录，同一环境没有变化时无需重复 smoke：

```bash
python3 plans/plans_v2_0921/cci/submit_job.py --label probe --timeout-seconds 900
```

已有的三文件回归作业可这样提交：

```bash
python3 plans/plans_v2_0921/cci/submit_job.py --label regression --timeout-seconds 900 \
  --argv-file plans/plans_v2_0921/cci/regression_argv.json
```

CLI 返回 run_dir 和 job ID。查询作业：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs describe JOB_ID \
  --workspace-name=share-space -o json
```

默认 probe 不运行科学 S0。完成实际实验脚本后，为每个阶段写 JSON argv 文件，使用同一提交工具。详细操作、参数、环境与 4 卡分工见 `cci/README.md`。不能调用尚未实现的脚本后就声称完成实验。

## 三、启动时必须核验的事实

计划编写时分支是 `audit/round2-review-20260921`，HEAD 是 `403b55db65f4c35c1a85d0794ad0de2765b07d96`，研究源码与受审 `fb767f7f6efbc428be39c9ad84f5905331d6e40f` 相同。先核验当前实际状态，不依赖这个快照。

已知用户未跟踪内容包括 `artifacts/`、`checkpoints/run_multi_year_pull.sh`、`plans/plans_0921/` 和 `plans/plans_v2_0921/`。保留它们，不清理、不覆盖、不停止数据拉取进程；已知未跟踪内容不构成暂停开工的理由。若工作期间出现无法归因的源码改动，暂停相关编辑并报告。

普通 Git 命令曾因 ownership 失败。在仓库中可使用 `git --git-dir=.git --work-tree=.` 进行所需操作，不全局放宽安全配置，不 reset/checkout 用户工作区。

283 passed / 7 skipped 是历史 CPU 日志，不能作为你的补丁已经通过的证据。

## 四、按这个顺序落实补丁

### A. 官方归一化与参照身份

- 在 `NormalizationContract` 加显式官方零 diff_mean policy，policy 纳入身份；legacy 行为有明确标识和回归，不能静默改写旧实验。
- exporter 使用 pinned 官方网络及实际官方 iterative forward 路径，独立构造 transforms，不使用 EarthDelta 待验证的 NormalizationContract 生成参照。官方 pin 为 `58dfee5a6037399a40fefd492bc00421e0c885a8`。
- exporter 与 S0 消费同一明确配置，绑定 checkpoint、patch size、normalization、变量/单位/坐标、实际 raw/normalized input、dtype、exact shape、interval 和全部登记多步结果。不能只核验 SHA 字符串长度，也不能只读一步输出。
- 在真实 gate 调用链上加入错配/缺项拒绝测试；合成参照只能证明控制流，不得产生可供真实训练准入的证书。

### B. 原生全目标评分

- 在 `earthdelta/metrics_contract.py` 实现或修正实际使用的原生 reducer：在 `[H,V,Lat,Lon]` 一次归一化，保留 B/K；同一 q/scale 下 `gain=loss_ref-loss_edit`。
- 检查 real floating、finite、非负 q、正 scale、广播及每个实际目标的有效分母。保留真值精度，用 float64 形成误差/累加；不能用 eps 掩盖空目标。
- 用非均匀时效/面积/变量权重的独立端点计算验收，避免逐特征先归一化消掉真实权重。旧逐变量诊断 API 可以保留。

### C. 实际入口校验与执行状态

- 实现最小切片/index/content 校验，并接到正式入口；本轮以临时合成资产验证缺端点、错坐标、非有限内容、stale marker 等反例，不扫全库。
- 固定非零系数下验证 bank A/B 梯度、骨干冻结、非零编辑响应、zero/nonzero/zero 恢复及串行 hook cleanup。B03 caller coefficient 图问题明确暂缓，不能称已修复或作为 bank 阻塞。
- B04 仅在当前路径确实使用 batched Gram 时处理；不新增连续优化器。保持串行、无 activation-checkpoint replay。
- 必需计算/身份结果写入失败不得留下 PASS。可确认的非必要清理 warning 分开记录，未知异常失败；修报告缺值格式化。测试实际 orchestrator/report，不能只测试手工创建的结果字典。

优先修改现有 bridge、metrics、exporter、S0 及测试；必要才新增 `pilot_contract`/preflight 小模块。不要顺手重构全仓库、重写数据下载器或提前实现完整 head/bank/output 管线。首次正式 registry 消费前必须拒绝复数/空表并含显式 no-edit，但不要求当前重写全部 selector。

## 五、验证与停止边界

先检查 test/fixture，再列出明确的合成测试节点。不要直接照抄 `pytest tests/ -m "not slow"`：当前 `test_load_real_normalization` 未标 slow，可能读真实 NPZ；checkpoint 测试已有 slow 标记。

测试结果必须来自本次运行，记录命令、退出码、通过/失败/跳过及理由。不得删旧断言、放宽容差或以改预期掩盖 regression。检查整体 diff 和入口是否真正调用新增逻辑。遇到同范围问题继续定位、修复、验证，直到完成或有准确的外部阻塞。

无需为修复数量凑齐 15 条 finding。保持 memory/JEPA/gain calibration 关闭的本轮设计；head 校准接线在后续双头阶段处理。若发现不可绕开的新问题，提供最小反例和影响路径，并完成同一路径的必要修复。运行护栏可在未看相应开发结果前依据实测调整并记录；不得事后改主指标、候选、分组或确认阈值来制造收益。

每阶段分别记录 `code_status`、`run_status`、`research_verdict`。CPU 阶段的 `CPU_VALIDATED/NOT_RUN/NOT_EVALUATED` 是中间检查点，保存后继续提交真实 S0。没有对应真实结果不能声称 S0、oracle、标签效率或双头优势已经通过。

### 四卡与多作业如何使用

- S0 保持同一参考路径独立比较；其它卡可并行做不依赖其结果的环境检查。
- Fs 冻结后，各专家可分配到不同 GPU 的独立进程；每个进程自己的模型、hook 和输出目录。
- 完整候选缓存按起报 ID 切成互不重叠的四个 shard，各 shard 运行全部候选，最后核验并合并。不能只存赢家，不能漏掉失败起报。
- head 的独立 seed/配置可并行，所有方法共享标签/仿真资格；不并发修改同一模型或输出文件。
- 后继任务以真实证书/manifest 和源码身份为依赖；不要让已分配 GPU 的任务无限轮询等待前驱。排队任务保存 job ID，状态不明确时先查询，避免重复提交。

### 可选 API

允许按需使用 `https://api.siliconflow.cn/v1`，模型候选为 `zai-org/GLM-5.3`、`deepseek-ai/DeepSeek-V4-Flash`、`Qwen/Qwen3.8-27B`。先查询可用性，使用 OpenAI 兼容接口即可。密钥只从私有环境变量 `SILICONFLOW_API_KEY` 或用户会话的安全配置读取，不复制进 prompt、源码、命令参数、输出或 Git。API 用于辅助代码/日志分析，不生成天气真值或替代实验裁决。天气主实验不需要这些模型时可以不调用，不以缺 API 环境变量阻塞 H100 工作。

## 六、交付与后续续接

把本次记录放入新的 `artifacts/round2_next/<唯一运行ID>/`，至少包含当前配置/缺项、source state、真实命令与日志、`decision.json` 和 `HANDOFF.md`。复用等价已有格式，不建设新的报告框架。

最终回复请用中文，包含：

1. 实际修改了什么、为什么，附文件位置。
2. 实际测试结果与覆盖的关键反例。
3. 已修复与暂缓项，分别列明，不能把暂缓记成关闭。
4. code/run/research 三轴状态，以及真实 S0 是否执行。
5. 结果文件路径和剩余限制。
6. 真实作业 ID、运行状态、原生效果表、配对区间、guard 和成本；若失败，给出具体日志及已尝试的解决方式。只有不可解决的实际阻塞才交回用户，不以旧模板锁、本地无 GPU 或缺少重复授权为由停止。

长任务持续更新 `HANDOFF.md`，上下文切换后从真实工作树和作业记录续接，已有授权持续有效。路线保持：真实 S0 → Fs/小 bank → 完整缓存与 oracle → cheap legal → 有条件的 dual → 有条件的输出挑战 → 一次最终确认/只读收尾。不要重做已完成工作，不绕过科学证据与确认集边界。

现在开始实施并持续推进到真实结果或有依据的停止结论，不要仅回复计划或停在 CPU 验证。
