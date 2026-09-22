# IMPLEMENTATION_AUDIT_ACTIONS

本表继承上一轮复核，不新增代码finding；行号指fb767f7。当前分支403b55d与其研究源码树相同。分类针对**本轮最小实际路径**，不是承诺所有公共API均已修复。

- `BLOCKS_NEXT_PILOT`：不修实际使用路径，就无法得到可信实验；最小patch或入口验证即可。
- `FIX_BEFORE_TRAINING`：只在对应训练／计算路径被调用前处理。
- `FIX_LATER`：当前不触发或非科学核心，保留未闭合状态。
- `NOT_A_BLOCKER`：事实风险可能存在，但当前通过限制模式/正式入口排除，不为此停止所有实验。
- `AUDIT_FINDING_REJECTED`：本轮没有整体否定某个B finding；拒绝的是部分过宽的阻塞/科学后果，不伪造“15条中已修复/关闭多少”。

| ID | 执行分类 | 复核结论 | 受审file:line | 影响 | 何时处理 | 最小处置 |
|---|---|---|---|---|---|---|
| B01 | BLOCKS_NEXT_PILOT | 同意核心；共享实现本身不是错误 | scripts/export_upstream_reference.py:106-113,230-250；earthdelta/bridge/stormer_bridge.py:228-257 | 共享待审normalization会保留已知diff_mean偏离；官方policy应为零均值。 | P0-01 | 只对所用checkpoint建立独立policy/变换与真实参照；不重写网络，不把微小偏离直接称巨大天气损害。 |
| B02 | BLOCKS_NEXT_PILOT | 同意 | scripts/s0_gate.py:165-191,231-301；earthdelta/bridge/stormer_bridge.py:818 | 单纯SHA长度/只读1-step/未绑定输入和policy可能让不匹配参考通过。 | P0-01 | 绑定实际资产、输入、所有注册时效及exact shape；不扩为通用安全平台；单个hash gate失败不代表strict loader能加载任意垃圾。 |
| B03 | FIX_LATER | 同意caller梯度缺失；不作当前bank普遍阻塞 | earthdelta/bridge/stormer_bridge.py:742-750 | 新建叶tensor不接调用者图；bank/input梯度不因此消失。 | 仅启用端到端controller训练前 | 当前固定系数bank+缓存head学习无需caller图；那条路径启用时新增tensor_coefficients[B,K]并测试head/caller梯度。 |
| B04 | FIX_BEFORE_TRAINING | 同意真实广播回归；限定触发路径 | earthdelta/metrics_contract.py:181-195；earthdelta/heads.py:445 | batched Gram + shared program会报einsum维度错。 | P0-04若调用该系数helper | 最小ellipsis contraction和广播测试；当前finite端点路线不调用则不阻塞oracle。 |
| B05 | BLOCKS_NEXT_PILOT | 同意；先保证实际评分域 | earthdelta/metrics_contract.py:75-99,145-155,228-270 | 全局正和不保证逐切片分母正；weighted_mse缺负权重/scale边界。 | P0-01 | 全目标reducer和实际入口校验Q、scale、分母；目前missing=error，不建设复杂缺测系统。 |
| B06 | BLOCKS_NEXT_PILOT | 部分同意；sum/mean不自动是bug | earthdelta/metrics_contract.py:145-155；earthdelta/heads.py:350-366；earthdelta/geometry.py:65 | 目标作用域与校准未闭合；未正确嵌入Q_eff会消去时空权重。 | P0-01评分；P0-04校准关闭 | 固定全目标Q_eff；保留诊断reducer；校准off并分开输出；同Q缩放的ridge/成本项同时缩放。 |
| B07 | NOT_A_BLOCKER | 部分同意；当前retrospective且memory off | earthdelta/data/make_splits.py:84-97,222-228,300-327 | 6h场景/legacy被标为真实来源不可靠；当前不用核验memory时不产生那条输入泄漏。 | P0-02标注；真实first-seen路径后置 | 情景与回顾性明确，旧记录不自动升级；不为小pilot建立完整实时归档。 |
| B08 | BLOCKS_NEXT_PILOT | 同意检查器缺陷；实际污染未证实 | earthdelta/data/pull_wb2.py:428-498,705,783,1078,1180 | 数组shape/marker不能证明数据写入完整。 | P0-01所用S0切片；P0-02训练切片 | 逐一检查实际消费片段并绑定证书；不要求重构所有下载器/扫描全量数据。 |
| B09 | FIX_BEFORE_TRAINING | 部分同意；日历生成可作为计划 | earthdelta/data/make_splits.py:133-160,338-390；earthdelta/contracts.py:28 | 日历manifest未证明实际history/target存在；lead-specific ID不是独立过程分组。 | P0-02 | 在admission/Dataset入口join真实索引，另存过程ID并拒绝placeholder进入正式运行；不禁用日历计划器。 |
| B10 | BLOCKS_NEXT_PILOT | 同意 | scripts/s0_gate.py:608-623,797；scripts/export_upstream_reference.py:410-429 | export选择ps4而consumer固定ps2将验证错误对象。 | P0-01 | 统一一个实际checkpoint/config/reference目录；不要求两个模型都验证。 |
| B11 | NOT_A_BLOCKER | 部分同意；说明保留历史YAML合理 | plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md:17-48；research_spec_v6.yaml:82 | delta_MDE不能定义价值；preview零额外成本未获支持。 | P0-03判gate前 | 本包作为superseding合同；证书区分delta_min/MDE，计preview；不要求覆写历史YAML。 |
| B12 | NOT_A_BLOCKER | 同意仅支持范围；非已发生串行污染 | earthdelta/bridge/stormer_bridge.py:633-652,779-790 | thread-local不提供model级线程安全，也未支持checkpoint replay。 | 串行合同立即；框架后置 | 本轮serial/no-replay；未来确需并发或activation checkpoint时再补，不主动实现。 |
| B13 | NOT_A_BLOCKER | 部分同意；状态应一致，科学严重度下调 | scripts/s0_gate.py:679-692,863-865 | cleanup异常可能留下PASS+error；不说明已完成测量数值错误。 | P0-01最小状态分离；不单独阻断 | 必需计算异常使measurement false；非必要cleanup可warning；别因此重跑整套天气实验。 |
| B14 | FIX_LATER | 同意P2报告器缺陷 | scripts/s0_gate.py:748-771,843 | 对N/A使用指数格式会令Markdown失败；JSON先保存。 | 可随相邻改动一行修复，非必需 | 缺值安全格式化；完整JSON可复核就不影响科学判断，不借机重做报告框架。 |
| B15 | NOT_A_BLOCKER | 部分同意；A03四项核心校验已有效 | earthdelta/selection.py:247-309 | 实浮点dtype缺校验、空表返回no-op；并未推翻finite/bound/active/count四项修复。 | P0-02正式registry入口 | 拒绝复数/空registry并注册noedit；通用planner空表语义可保留，helper清理后置。 |

## 对原审计应保留的纠正

A01/A04/A05继续部分关闭合理；A06拆为“UTC已修复、availability语义未闭合”。A03应明确“四项主要可行域修复已关闭；类型和registry入口残留”；A11应明确“预写胜利/固定百分比已处置；价值阈值和preview成本待补”。无需修改历史YAML才算处置。

不接受以下扩大解释：caller系数图断等于bank无法训练；sum和mean字面不同必然是错误；缺并发框架阻塞串行pilot；非必要cleanup失败自动作废测量；B15意味着原四项修复无效。没有证据指认全部旧测试被削弱；新增测试覆盖不足与旧测试主动弱化是两件事。

本轮不得为“审计清零”建立并发调度、全局manifest平台、全量Zarr修复器、报表框架或新GPU后端。涉及同文件的廉价修复可以随任务完成，但必须列明它是否真实影响当前执行路径，不以新增测试数冒充研究进展。
