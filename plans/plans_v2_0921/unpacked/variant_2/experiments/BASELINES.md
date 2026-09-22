# 本迭代最小baseline合同

不是最新论文实现大赛，不克隆全部近邻。复用当前numpy/torch、EarthDelta原语与已pin的官方Stormer；没有新增第三方实现复制要求。

| ID | 对应实验 | 实现与输入 | 输出/选择 | 公平性与用途 |
|---|---|---|---|---|
| REF | E1–E3 | 冻结Fref/Fs；合法初始化 | no-edit完整轨迹，物理gain=0 | 不是零计算成本；hash/归一化/continuation一致。F0另列背景，不替代强Fs。 |
| STATIC | E1–E3 | fit/dev选定固定候选；另有同资源训练的强静态Fs | confirm应用冻结ID，不重选 | 区分最佳已测试固定候选与总体理想静态最优；bank/static训练labels计账。 |
| REGIME | E2 | `earthdelta/baselines/r2_cached_policy.py`；与dual同合法context，fit-only原型/聚类 | 各regime在fit/dev选候选，sign-off后固定 | 低成本强挑战者；不得用confirm天气误差定义regime。 |
| ORACLE | E1/E2诊断 | 完整实际非线性候选+未来核验 | 每起报max实际gain | 不可部署；全部调用单列离线取证成本；候选不全不能报complete。 |
| DIRECT_GAIN | E2 | 同context+candidate descriptors；低容量ridge/MLP；同仿真资格 | 预测原生gain，预算内argmax含no-edit | 一条truth生成全部候选标签；相同HPO/labels，允许与dual同等响应预训练资格。 |
| DUAL | E2 | 复用ComposedPredictionHead，memory/JEPA/calibration off | e_hat、du_hat、analytic surrogate；native outcome结算 | no-edit结构零；fixed D需明确summary/native差距；不是默认赢家。 |
| EDITED_FORECAST | E2诊断 | 同仿真轨迹、同读出/容量预算，预测Fa或其注册读出 | 与预测参考误差信息/共同合法参考态组合的候选打分，native结算 | 验证增量学习是否比直接预测整轨迹更有效；本身也不需要未来核验来学Fa。不得把真目标输入预测器。 |
| MEAN/ZERO_DU | E2诊断 | fit平均响应或结构零响应 | 与同e_hat/选择规则结合 | 检查复杂响应头是否真的学到状态变化；near-null方向指标N/A。 |
| VIRTUAL | E3 | shared合法policy + 原生predicted_du或有效fixed lift | Fref+predicted_du，不运行选定真edit轨迹 | reference/preview/lift成本计账；summary-only无有效lift则BLOCKED该全场对照。 |
| JOINT_OUTPUT | E3 | 同I与明确计费reference preview；低容量联合残差head | 注册多个lead的完整目标残差 | 同labels/fit/dev/HPO与读出；不限制成只改单步弱对手。 |
| FEEDBACK | E3 | 每步自身预测状态+同合法context | hold内注入修正，反馈下一步Fref | 不能喂未来truth；zero还原；同窗口、原生状态传播、相同guard。 |

## 固定的四象限

true e / true du，true e / predicted du，predicted e / true du，predicted e / predicted du。前三格都只在离线诊断区，不能标serving；其选项均按实际candidate outcome评分。

## 两份标签/资源账本

**given-bank/controller账本**：条件在同一个已冻结bank上，比较policy需要的新增labels、仿真/训练/推理。

**end-to-end账本**：额外计入Fs/bank训练、特征/输出基训练、所有调参与候选仿真、失败重试和confirm成本。bank已看过全量labels时，少label head实验不能声称全系统少标签。

同参数和同实测成本是两种互补比较，不声称自动同时相等。新增任务或调参配置必须扣cap，不给dual专属数据/表示/预览权限。

## 实现复用与暂缓

- 使用已审 `earthdelta/heads.py`、`paired.py`、`probe.py`、`lowrank.py`、`bridge/stormer_bridge.py`；先最小补丁，不重新实现整个backbone。
- 官方Stormer参考pin：`tung-nd/stormer@58dfee5a6037399a40fefd492bc00421e0c885a8`。外部代码引用沿用现有来源与许可说明，不移除版权；本包不分发权重/数据。
- 不要求本迭代实现VI-MoLE全套、WeatherPEFT全部任务或新的hypernetwork。调用方coeff梯度只有端到端系数策略训练才是前提，本轮regime/direct-gain有限候选选择不需它。
- 不为本包重新做literature review；上一轮近邻结论只用于claim边界，不能当新增实验结果。
