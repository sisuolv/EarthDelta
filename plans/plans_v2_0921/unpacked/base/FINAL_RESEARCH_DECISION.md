# FINAL_RESEARCH_DECISION

日期：2026-09-21。迭代：`round2-next-iteration-20260921`。依据：紧前一轮独立研究复核；本包不重新做文献综述。

## 冻结决定

- **research_verdict：BOUNDED_CONTINUE。** 再买一次有上限的取证，不认定 novelty 已成立。
- **implementation_readiness：NOT_READY_FOR_CREDIBLE_WEATHER_COMPARISON。** 最小实际路径的参考、身份、数据与评分合同必须先成立；其余工程问题不一律阻塞。
- **下一笔预算：** 一个合格非零小字典、一张完整有限候选 dev 结果表，以及同表上的小型合法策略／direct-gain 筛查。不是只买 hindsight oracle，更不是完整 P1/P2。

## 保留的 scientific claim（均待验证）

1. 候选参数干预的有限轨迹响应可以从合法输入预测，并对实际选择有用。
2. 响应仿真与核验误差监督分离，可能在同标签／仿真资格和匹配总资源下提高标签效率或动作复用；不是信息增益定理。

“e0/du 双头＋二次式”、LoRA、预算、非对角 Gram 本身不算新原语；不声称参数编辑必然胜过 feedback correction。全新专家迁移、部署安全保证及真实大气因果发现均不在当前声明内。

## 仅保留三个生存实验

E1：合格小字典的完整有限候选 oracle／强静态差距（本包 P0-02、P0-03）。

E2：同缓存、同合法信息的交叉拟合策略比较与四格 e/u 诊断（P0-04）。

E3：仅在 E2 提供继续理由后，检验实际参数执行相对 virtual／输出反馈纠错的价值（条件性 P0-05）。

## 暂缓

大规模重构、新 backbone、大字典、JEPA、memory、dynamic rank、spectral loss、Complexity Atlas、RL、在线 JVP、并发与 activation-checkpoint 支持、全量 P1/P2、额外论文综述。保留已有模块但默认不调用。不得因某个辅助模块已写好而把它加回主线。

## 决策规则

- correctness／身份／实际数据不可信：`BLOCKED_CORRECTNESS_OR_PROVENANCE`；不产生科学 PASS/STOP。
- 合格 registry 的 oracle 余量上置信界仍低于预登记最小有用效应：`STOP_CURRENT_DICTIONARY`，不是停止所有参数编辑。
- oracle PASS：仅支持有界 E2；不证明 `E[max E[g|I]]` 已经为正。
- 在约定数据、模型和训练 cap 内，合法策略不能超过强静态：`STOP_CURRENT_PLANNER`；若静态解释主要收益，`PIVOT_STATIC_ADAPTATION`。
- direct-gain 已解释分解收益且无已登记资源优势：`PIVOT_NARROW_CLAIM`；不继续用校准或新模块包装双头。
- feedback／virtual 在相同限制下稳定支配真实编辑：`PIVOT_OUTPUT_CORRECTION`，保留有证据的响应代理价值。
- CI 跨价值阈值：`INCONCLUSIVE`；只按原 cap 补证，到 cap 后停止追加，不冒称零效应。
- 合法策略达到价值门槛、至少一个已登记资源／复用优势成立且保护指标不过度退化：`CONTINUE_BOUNDED_NEXT_ITERATION`，只提出下一次授权，不自动开跑。

所有科学阈值目前为 `TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。其中 delta_min 必须有独立价值／成本依据，不能由方差或 MDE 自动替代。资源 cap 为待实测、待批准，不虚构小时或收益。

**初始只允许 P0-01。每完成一项写 decision.json 并停止；机器状态不得自行解锁下一项。**
