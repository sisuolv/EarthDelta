# STOP合同：冻结项、科学判断和缺证分开

状态：合同文本OBSERVED（来自计划及本次审查）；未来判定TO_BE_RUN。

## 1. 不得改变的原计划合同

原文：source_plan/NEXT_STEPS_10H_PLAN.md:99、116–163、165–182、215–245、257–288。

- official独立源码/数值transform；ps4，FP32，official_zero_diff_mean，6h step；注册1/4/12步；**max_abs_diff <= 1e-5**，比较normalized全字段原生网格，72h评分前12步必须通过。
- 其它容差（例如input-norm或merge检查）从真正冻结gate_config/原始证书逐字段恢复，缺失则BLOCKED，不默认猜数。
- 合并/常开等价的阈值与官方S0阈值不是一回事。历史报告中max(1e-5,1.5e-3*delta_max_abs)和rtol=1e-5是特定合并检查记录，**绝不能移用为S0放宽依据**。
- reference为唯一合格fitted-Fs；no-edit=Fs，hold后从编辑状态继续Fs，不能换F0。
- K4/rank4，blocks18–23，max_active1，a0=rho=0.25，hold4×6h；5个完整候选。K2仅可在看dev前依据profile登记新协议，不得原K4失败后按赢家删除。
- 原历史输入t-12,t-6,t；核验目标t+6,t+24,t+72；UTC真实索引/变量顺序/坐标/全部内容证书；bank_fit与policy_dev完整support隔离；所有candidate同issue同role。
- 所有S0/profile/debug曝光入账；confirm本窗口不读，dualhead/E3/JEPA/memory/dynamic-rank/spectral/new backbone均暂缓。
- 训练先16–32 update诊断，formal最多500作为原起始cap，非必须跑满；Fs/experts的LR/选择/重试在训练侧结果使用前冻结，禁止无限增加尝试。
- 8个exposed debug；dev32/64/128由profile/实际独立过程数预先选，不能按收益挑时段。
- native24h主目标，6h诊断，72h损害guard；Q只归一化一次，e=truth-Fs，du=edited-Fs；全端点先double差分；analytic/calibrated单列。
- 无未来truth、真实候选response或oracle输入serving；scaler/basis/HPO仅fold-train；prediction/动作先冻结再评分。
- 不允许共享hook并发、DDP/activation-checkpoint简化；不得用环境成功或finite替代官方parity。

## 2. 硬阻塞

S0任一时效/身份失败；Fs质量或共同参考未闭合；前驱MISSING/FAIL；content/registry/权重hash不匹配；bank恢复失败；profile修改冻结权重；实际入口可以绕过资格：**BLOCKED，不执行下游**。
缺GPU/权重/Zarr/原始日志不代表科学无效，也不准生成synthetic PASS冒充正式结果。
旧8/8大退化Fs及其派生专家隔离，不得因finite/eligible旧字段为true重新准入。

## 3. Fs质量与阈值

现有finite/update count仅为必要条件。资格质量规则在训练侧候选结果前冻结，比较同初始/终态checkpoint同面板；每次访问的first/last曲线另列。
Fs可接受退化/数值地板、72h损害上限尚无本次真实冻结输入，配置保留null并标TO_BE_PREREGISTERED。不能默默借用新F0 sprint的1.001/1.05代替原Fs资格标准。
允许训练不足=INCONCLUSIVE_QUALIFICATION，不自动判方法无价值；不过明确观察到的大退化仍禁止进入bank。

## 4. 科学STOP / CONTINUE / PIVOT

- 合格完整bank下oracle相对Fs的效应上界仍小于预登记delta_min：STOP_CURRENT_BANK。
- oracle相对best-static的上界也不够：PIVOT_STATIC或停止当前dynamic selector。
- oracle有余量但合法OOF policy在cap内无有用增量：STOP_CURRENT_SELECTOR；这不是证明任何未来表示都不可预测。
- ridge/regime解释主要可部署收益：窄化为廉价策略，不宣称双头/响应有额外价值。
- 明确正向合法收益、成本/损害合格：CONTINUE_NEXT_ITERATION，但confirm与response实验须另行立项。
- CI跨delta_min、统计功效不足：INCONCLUSIVE；仅在预登记采样/预算上限内补证，到cap停止追加。
- 任意失败/丢失candidate被删除、静态选择用了test标签、候选数按oracle收益调节：INVALID_EVIDENCE，不出科学决策。

## 5. delta_min不是MDE

配置标记：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。
这表示冻结流程仍待完成，不表示从方差直接推导应用价值。delta_min由价值/成本/数值地板依据决定；MDE由alpha、power、独立过程n、方差决定，分别保存。
不能凭空恢复2–3%，也不能把新sprint的0.001/0.005借给原协议。所有确认性threshold在对应数据揭示前锁定。
Bootstrap保持同一时间过程的candidate/lead/method成组配对；n不是网格数、candidate数或更新步数。

## 6. 状态与退出

OBSERVED=有材料支持的事实；TO_BE_RUN=未来任务；MISSING=必要输入未取得；BLOCKED=前驱阻塞。
返回码0只证明进程没有报错，不等于科学成功。必须检查artifact result、每项required gate、verdict_committed以及完整身份。
