# B01–B15 → 最小执行优先级

依据：上轮独立复核，位置固定在`fb767f7f6efbc428be39c9ad84f5905331d6e40f`。本包不要求把15条全部修掉。分类表示当前实际pilot路径的优先级，不将原报告P0/P1严重度机械等同于任务解锁。

- `BLOCKS_NEXT_PILOT`：不修复/不由可靠入口阻断就不能采信真实收益。
- `FIX_BEFORE_TRAINING`：在指定训练/数据消费模式前修复，不一定阻塞更早固定系数bank。
- `FIX_LATER`：未来真正触发该接口时再修，当前限制模式。
- `NOT_A_BLOCKER`：不单独阻止受控pilot，所列最小声明/入口限制仍需遵守。
- `AUDIT_FINDING_REJECTED`：本轮没有整条B finding被认定不存在。空集合是有意的；拒绝的是某些过宽推论，不抹掉已确认事实。

| ID / 复核判断 | 当前分类 | 文件:行 / 符号 | 最小处置与任务 | 何时才真正阻塞 / 不要做什么 |
|---|---|---|---|---|
| B01 同意核心事实 | BLOCKS_NEXT_PILOT | `scripts/export_upstream_reference.py:106–110,230–251`；`stormer_bridge.py:229–251`；官方`inference.py:114–119` | R2-P0-01：官方零diff_mean policy与独立rollout/transform；policy入hash。 | 共享代码本身不是罪；已知policy差异才是问题。不以偏移小推断无影响，也不声称已观察天气损害。 |
| B02 同意 | BLOCKS_NEXT_PILOT | `scripts/s0_gate.py:175–194,265–300`；`check_version_match` | R2-P0-01：checkpoint、norm、变量/坐标、raw input、shape、dtype、源码pin和全部注册多步输出绑定。 | 长64字符串仅是记录hash，不是指定模型校验。“任意文件过hash子门”不等于过整个S0。 |
| B03 bug成立，阻塞范围缩窄 | FIX_BEFORE_TRAINING（仅caller/controller端到端训练） | `stormer_bridge.py:742–750`，`controlled_rollout(differentiable=True)` | 本迭代固定系数bank只验证bank梯度；caller `[B,K]` 保图接口暂缓。若未来必须端到端controller才修并测试。 | 不阻塞R2-P0-02 bank或离线缓存；不得把bank/input梯度与caller梯度混为一谈。不借此弱化本轮有限候选direct-gain/regime对照。 |
| B04 同意广播回归 | FIX_LATER（实际进入batch Gram前） | `metrics_contract.py:180–195` | R2-P0-03若使用该分支，改正确ellipsis并覆盖共享/逐样本program/Gram。 | 原生端点损失主路径不依赖它；不因此重写geometry或新增continuous QP。 |
| B05 同意实际边界bug | BLOCKS_NEXT_PILOT（评分入口） | `metrics_contract.py:96–152,226–274` | R2-P0-01：全目标reducer统一real/finite/非负/scale/分母验证。 | 合法零权重mask允许，但全目标无有效权重拒绝；正常正权重调用不自动错误。 |
| B06 部分同意 | BLOCKS_NEXT_PILOT（Q一致性）；head校准在E2前 | `metrics_contract.py:145–157`；`heads.py:345–358`；`geometry.py` | R2-P0-01定义native Q_eff；R2-P0-03默认关闭校准、单列analytic/calibrated。 | sum/mean可共存，只要构造和惩罚单位一致。不要为了MetricSpec“完整”建新指标框架。 |
| B07 部分同意 | NOT_A_BLOCKER（memory off、retrospective pilot限定） | `make_splits.py:86,224–228,300–333,373–388` | R2-P0-01入口声明scenario/retrospective；禁止未证实近期核验特征；legacy不升级成observed_first_seen。 | 真实可用性/开启memory时才阻塞。UTC已修不重做。不开发全量first-seen采集服务。 |
| B08 同意 | BLOCKS_NEXT_PILOT（实际样本内容） | `pull_wb2.py:425–478,668–705,751–783` | R2-P0-01小输入guard；R2-P0-02对实际读的小子集逐批内容核验并绑定证书。 | 不相信shape/markers。无需先重写整个下载器事务发布；未扫描全库就不能称现有资产已污染。 |
| B09 同意准入风险 | FIX_BEFORE_TRAINING（任何真实pilot样本） | `make_splits.py:143–159,338–388`；`ArtifactVersion` | R2-P0-02实际time lookup联结history/target，issue/process分组，正式入口拒placeholder。 | 日历计划表本身可以保留；禁止把它当可训练样本证明，不需重写历史split全文。 |
| B10 同意 | BLOCKS_NEXT_PILOT | `s0_gate.py:609–620,797–817`；exporter CLI | R2-P0-01单一显式模型配置、独立reference目录、S0与实验同身份。 | ps2/ps4只验证实际使用的一个；换主模型先冻结规格，不用别的模型PASS背书。 |
| B11 部分同意 | NOT_A_BLOCKER（最小bank）；阻塞正式科学判决 | `A11_DISPOSITION_NOTE.md:17–41`；v6 context声明 | R2-P0-02证书分列delta_min/MDE，登记preview/重放费用；本包主合同覆盖历史门槛。 | 历史YAML可保留；已有上轮FINAL已撤下零成本主张。不要为了清理历史文件延误pilot；也不沿用2–3%或MDE=价值。 |
| B12 边界成立 | FIX_LATER | `stormer_bridge.py:635–657,780–793` | 当前强制串行、独立model实例、无activation-checkpoint replay；记录不支持模式。 | 没有实际并发调用，不建并发框架；未来需并发/replay再做model级隔离。 |
| B13 状态矛盾成立，科学危害需限定 | NOT_A_BLOCKER（独立数值事实）；同gate小修 | `s0_gate.py:680–692,864` | R2-P0-01真实orchestrator晚期异常测试；未知错误不保PASS；纯已分类cleanup warning另轴记录。 | cleanup失败不自动抹掉正确数值，也不得将未分类CUDA错误当无害。无需独立cleanup项目。 |
| B14 同意 | NOT_A_BLOCKER | `s0_gate.py:752,770,838–845` | R2-P0-01同文件安全格式化，缺数值输出N/A/error，JSON先保存。 | 不改变成功数值；阻塞依赖Markdown的自动化，不等于研究方向失败。 |
| B15 类型残留成立，A03主要修复有效 | FIX_BEFORE_TRAINING（正式registry消费前） | `selection.py:248–296` | R2-P0-02 registry入口要求实浮点、非空、显式no-edit；保留已有finite/bound/max_active/max_candidates校验。 | 通用planner空表回no-op可合法，正式registry不能这样混淆；不称四项已修失败，不扩大selector重构。 |

## 原A项的精确状态

A01/A05：必要官方路径/身份绑定尚未闭合，重新打开合理。A04：公共公式复用有效，但全目标与校准合同仍需落地。A06：UTC修复有效，真实数据端点与provenance消费要另核验。A03：四项主要校验CLOSED，类型/正式registry残留PARTIAL。A11：预写声明/旧百分比已降级，保留历史文件允许，新的阈值合同须修正。

没有GPU是已知资源限制，不新建finding。新增测试没覆盖某性质，不等于旧测试被削弱。本包没有重新复跑原CPU反例，不把原日志当本次执行。

## 对Codex的约束

以实际调用路径为准，只改分配到当前任务的最小部分。如果发现新的不可绕开的correctness问题，提供反例、影响路径和局部修复；影响科学目标/资源或超出scope时停止申请审阅，不自行增加第六、第七实验。已正确部分保留，不能“为统一风格”全面改写。

证据路径：`codex_audit_round2/results/audit_findings.json`、`results/evidence/cpu_repros.json`、`results/ROUND2_REPORT.md`；以上复核判断来自上轮独立审查，不要求原报告标签全盘正确。
