# Baseline Implementation Contract

| ID | 实现复用与wrapper | 输入/输出 | 关键公平限制 |
|---|---|---|---|
| F0 | pinned Stormer +独立official对照 | 原生当前场→原始forecast | 不减变量；不把多intervalensemble只给某方法 |
| Fs | 现有ExpertLoRA或许可明确静态LoRA，训练后冻结 | 同当前/历史权限→forecast | 与intervention相同训练资料；F0/Fs明确分开 |
| Random | 固定seed从同可行registry抽样 | 同candidate→actualforecast | 含noedit，预算分布匹配；不挑最佳seed |
| BestStatic | dev集固定最优candidate | 所有确认issue用同edit | 不能按test选择；regimeonly先在fit定分组规则 |
| HighCapacityStatic | 调整静态rank使总trainable量匹配 | 与主方法相同信息 | 同rank与同总参数两个视角都报；不继承成本相等 |
| DirectRouter | 保留相同encoder/bank，直接boundedcoeffs +multisteploss | history/preview/budget→edit | 给予候选/预算/lead描述符，允许合理泛化；不要稻草人 |
| DirectGain | 新baseline浅MLP/相同backbonehead | I,a,lead→scalar gain | 同candidate标签、同HPO、同split；目标同Q |
| DirectEditedForecast | 与双头相近总宽度 | I,a→D(Fa) | 与主方法同有限仿真训练输出，不多偷未来truth |
| PairedAnalytic | 现有ComposedPredictionHead补统一metric | ehat,duhat→analyticgain | calibration off；JEPA/memory off同权限 |
| PairedCalibrated | 同上＋单独校准项 | analytic＋calibration | 独立消融，不把它的胜利都归于identity |
| ResidualPost | 共同context，多变量多lead decoder | Fref+predresidual | 不仅one-lead标量；训练成本匹配 |
| ResidualFeedback | Solver-in-the-Loop式协议、PyTorch本地实现 | 每步共同input→correctedstate→Fref | 能影响后续动力学；与参数方法同hold/rolloutloss |
| VirtualEdit | 共享du预测 | Fref+duhat(selected) | 数值近似不是实际模型编辑；保留误差和额外Fref成本 |
| OracleEdit | 全部注册candidateactualrun | offline truth→bestcandidate | 仅当前有限集上限；不列serving榜 |
| OracleE / OracleDu | 同一paired表4格替换 | 精确对应量 | actualdu花了K条forecast，不能藏成本 |
| Finite vs R/QP | 现有probe/geometry/teacher/selection | fixed finiteeffects vs localJacobian | 近似有效域、预算与候选必须一致 |

## 外部实现
Stormer/DISeL/WeatherBench-X优先复用已核实版本与许可；GEPS/WeatherPEFT/CoMoL/W2T当前清单有NONE或未确认，不直接复制。理论上的VI-MoLE-style marginalgain baseline应标明independentreimplementation，不冒称原作完整认证复现。CLAW/CCM作者代码本轮未核实，因此不作为强制安装依赖。

## 标签效率
每个起报的e0只算一份verification label，不因K编辑重复计K份。du是仿真label，不需要未来truth但有计算成本。可以另采无未来truth的训练issue响应，所有方法给予相同仿真输出训练资格；设计same-verification-label与same-total-compute两张表。只给双头额外仿真而不给directedited/gain可利用数据，不能直接归因factorization。

## 候选描述符
当前activebit+coefficient+hold只标当前bank索引。新bank即使K一样也不是同一action；需要内容绑定与行为/权重descriptor。P2优先同bank未见幅度/窗口/组合，不做未经标定的跨模型zero-shot。
