# STOP_CONDITIONS — 本迭代正式判决合同

Package ID：EarthDelta-R2-NextIteration-20260921。适用范围是冻结的 Fref/bank/registry、数据/信息条件、Q、lead、作用窗与资源上限，不自动外推整个参数编辑方向。

## 1. 尚未填写的统计阈值

下面所有 value 初始均为：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。

| 字段 | 含义 | 设定依据 |
|---|---|---|
| delta_gain_min | oracle相对Fref的最低有用物理收益 | 继续投资所需研究/资源价值，不由标准误自动计算 |
| delta_dynamic_min | oracle相对dev冻结强静态策略的最低有用余量 | 动态决策相对静态额外开销/价值 |
| delta_policy_min | 合法policy相对强静态的最低有用收益 | 有限资源下值得继续的实际效果 |
| delta_factorization_min | 分解相对direct-gain值得保留的额外价值 | 效果或达到同效所需标签/成本的改善；预先选一个主要量 |
| correction_noninferiority_margin | 纠错/virtual相对actual edit的效果非劣界 | 有实际意义的可接受损失，独立于confirm结果 |
| guard_harm_margin | 注册后续lead/变量可接受的最大损害 | 明确风险/研究要求，不事后调宽 |
| alpha / target_power / confirm_n_processes_max | 显著性/功效/独立过程上限 | pilot过程依赖与paired方差、cap |
| mde_vs_zero / planned_alternative | 当前设计分辨能力与检验备择 | 统计功效计算；不是价值阈值 |

该占位符表示必须在pilot方差取得后、confirm开启前完成注册，**不表示delta_min应当等于MDE**。价值阈值的来源可在pilot前确定；没有价值依据时，不替用户挑一个漂亮百分比。

数值parity tolerance也须单独登记：来源是相同合法实现的数值重复性、dtype与规范要求，且应小于拟研究的差异。不得扩大tol掩盖错误policy。parity tol不是科学gain阈值。

## 2. threshold_certificate 必须含什么

证书包含：schema/version、metric/Q/hash、Fref/bank/registry hashes、primary target、guard、point-estimand、paired unit、delta_min各值及value_rationale、pilot来源/方差、不包含confirm数据的声明、alpha/power、N上限、block长度/过程分组、MDE/功效计算方法、planned alternative、HPO/标签点与多重比较处理、retry/追加规则、创建时间、冻结hash、reviewer授权引用。

- 正式下CI > delta_min：支持超过最低价值。
- 正式上CI < delta_min：数据排除了该最低价值，适用STOP/PIVOT规则。
- CI覆盖边界（包括等于边界）：INCONCLUSIVE。不得以点估计大于阈值代替。
- 与零检验的“不显著”不能自动证明等效/非劣。
- MDE高于想检测的差异，不能为了凑PASS把delta_min设成MDE；要缩小结论或停止当前不足功效的支出。
- 追加confirm只能按事先注册的样本/alpha-spending规则。没有该规则，查看confirm后不允许继续增样直至显著。

## 3. 优先级最高：测量或权限失败

**C0 — INVALID / BLOCKED，不是科学STOP。**

触发：Fref或checkpoint配置错；normalization/raw input/multistep无法绑定；Q/scale/分母不一致；实际样本缺history/target；未来信息泄漏；registry内容与缓存不符；旧结果无法追溯；未授权使用confirm。

动作：停止评分/训练的受影响路径，保留日志，修复同一任务。不得用合成天气、历史README、CPU PASS或少数成功例子填科学metrics。无GPU/xformers/checkpoint/data/许可记BLOCKED，给出精确缺项。局部状态是`code_status=CPU_VALIDATED, run_status=BLOCKED`时，R2-P0-02仍锁定。

## 4. E1 — oracle和静态空间

**C1 — STOP_CURRENT_BANK。** 合格且完整的有限registry，oracle−Fref收益的上CI低于delta_gain_min。该库/预算/Q/lead连hindsight也不值得；不再为当前planner做大训练。

**C2 — PIVOT_STATIC。** 静态修正相对Fref有价值，而oracle−强dev-static的上CI低于delta_dynamic_min，动态选择没有足够额外空间。不采用“静态解释了X%”这种未注册比例阈值。

资格限制：零B初始化、未训练库、非法负幅度外推、部分候选未运行、缺数据和错误统计不能触发全方向科学否证。singleton-only FAIL只限制该注册动作集；若组合是关键假设，应事前注册少量组合。看到失败后扩库必须新run/确认集，且不是本包自动许可。

**C3 — HEADROOM_PASS只解锁取证资格。** 两项相关下CI超过各自delta，且正确性与guard合格。只支持存在hindsight空间，不能标deployable PASS。原反例：I无信息、e=±1、u∈{0,±0.5}，oracle=.75而合法最优=0。

## 5. E2 — 合法policy和分解价值

**C4 — STOP_CURRENT_PLANNER。** oracle空间有价值，但在已注册低容量模型/信息/资源范围内，最佳合法策略相对强静态的上CI低于delta_policy_min。不是“全场e0 R²低”或“还没赢过随机”单指标自动终止；需要实际结算与充分有界证据。

**C5 — PIVOT_DIRECT_GAIN / CLAIM_NARROWED。** direct-gain或regime策略有收益，但在相同数据资格、标签账本、HPO与成本下，双头最低额外价值被CI/预注册非劣判据排除，且注册标签/成本维度也无保留优势。撤下双头贡献，允许明确缩窄后申请E3；不擅自加入calibration/JEPA掩盖失败。

若双头与direct-gain差异不显著但CI很宽，只能INCONCLUSIVE，不写“被否定”。若预训练bank用了全量labels，而head少labels，仅能报given-bank/controller标签效率；不得报全系统标签节省。

**C6 — POLICY_SIGNAL。** 合法策略下CI超过delta_policy_min，原生注册目标结算、guard不越界。只证明该信息/库/预算下技能；双头额外价值独立判决。

## 6. E3 — 实际参数编辑必要性

**C7 — PIVOT_OUTPUT。** 反馈/联合输出纠错在注册效果非劣界内且成本更优（或效果更好且成本非劣），通过事先选择的配对统计/成本前沿判据。不能仅凭无限容量表达等价性触发此条。

**C8 — PIVOT_RESPONSE_SURROGATE。** virtual完整预报在注册效果上非劣、代价更低，则削弱必须实际编辑的主张；不否定响应模型本身。summary-only不具有效原生lift时记未完成对照，而非虚构virtual输出。

**C9 — STOP_WINDOW_OR_ROLLOUT_CLAIM。** 注册后续时效/变量损害的下CI超过guard_harm_margin（按预先定义的正损害方向）。停止该窗口/时效主张；不能删除guard或仅展示短期收益。若损害证据不确定，同样保持INCONCLUSIVE。

**C10 — CONTINUE_BOUNDED。** 在合法技能已成立的基础上，双头在至少一个注册有限资源维度保留明确增量价值，且actual edit必要性没有被强纠错挑战否定、guard合格。若E3尚未跑，只能建议完成E3，不能宣称参数编辑必要性已成立。所有CONTINUE都需下一次显式授权，不自动执行。

## 7. 统计不足、资源耗尽与运行失败

- CI跨阈值、MDE不够、过程样本不足：INCONCLUSIVE。可停止支出而不宣称科学命题错误；只按预注册追加方案和剩余cap获取更多证据。
- 资源cap达到：STOP_RESOURCE_CAP，保留已经实际完成部分；不转到未授权机器或扩大候选/epoch。
- 运行失败：失败机会保留统一分母；若有预注册no-edit回退，计入已花计算和回退成本。未知候选结果不是0收益。完整oracle任一必要候选缺失，不可获得完整headroom PASS。
- 防止确认集重复使用：若E1 confirm已用于后续方法设计，E2/E3必须另留独立confirm；不能“同一个集只运行各脚本一次”就声称整体没有重复适配。

## 8. 收尾

每个任务在任何终点保存decision.json并停止。负结论也是合格产物。R2-P0-05可经授权只读汇总；不得为了“完成计划”运行已被STOP/PIVOT切断的节点。
