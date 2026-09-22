# 文件级最小改动计划

以下是当前源码事实到实验的映射。SOURCE_IMPLEMENTED不意味着真实数据测试通过。完整符号/证据见evidence/audit_findings.json；每个当前文件的公开定位是审查提交的相同路径。

| 当前文件/符号 | 当前实现 | 问题与最小修改 | 对应实验 | 优先级 |
|---|---|---|---|---|
| paired.py / make_training_targets | e0=truth-reference、finite du、gain | 统一MetricSpec与promoted arithmetic；修订过强线性前提文档 | 4格、directgain | P0-02 |
| probe.py / cached_responses | 实际非线性有限候选响应；中心差分也在 | P0复用cached，不把Jacobian称finite effect；加运行manifest/callcost，不实现JVP | oracle ceiling、组合 | P0-05/P1-04 |
| geometry.py / ResponseGeometry | RᵀQR、RᵀQe、float64 | 保留原语；消费统一Q，禁止重复缩放；标local surrogate | local vs finite | P0-02/P1-04 |
| teacher.py / box_candidates、verify_candidates | SciPy box teacher+top候选非线性复验 | 保留；另外新建完整有限oracle评估，不把top3当全库上限 | A/B | P0-05 |
| selection.py / finite/QP | finite/QP分支和cost选择 | 有限路径补bound/count/finite/maxcandidate；真实noedit；共同fees | A–F | P0-02 |
| heads.py / ComposedPredictionHead | 已有GRU、leadcondition、e0/du/calibration | 不新造multi-lead头；公共gain、解析/校准开关、noedit零响应；公平directgain | C/D/F | P0-02/06 |
| heads.py / InteractionUtilityHead | 预测PSD H与b | 暂不主线扩建；与finite代理分开，H非非线性Hessian | 组合局部对照 | P1-04 |
| lowrank.py / ExpertLoRA | K专家，dense梯度、sparse真的跳过 | 复用；核对非零bank与总参数/真实执行成本，不误称只是乘零 | A/B/E | P0-04/P1 |
| bridge/stormer_bridge.py / normalization | 加载及使用diffmean | 显式official零均值policy，legacy只读；独立pinned路径对照 | S0 | P0-03 |
| bridge/stormer_bridge.py / controlled_rollout | hook cleanup、hold后关闭、末态返回 | 保留兼容；tensorcoeff梯度与trajectory返回；闭环重算/并发未支持则拒绝 | F/反馈 | P0-03 |
| bridge/stormer_bridge.py / checkpoint | strict参数名加载、冻结 | weightsSHA绑定、可信pickle来源、真实upstream一致性；not架构字符串identity | S0 | P0-03 |
| contracts.py / EditPlan、ProgramSpec | hold编辑与独立slot规格都存在 | 不另造第三套程序；descriptor不代表新专家权重，绑定bank哈希 | B/动作泛化 | P0/P2 |
| data/make_splits.py | 年份划分、手设6havailability、naive时间 | UTC、真实索引、scenario明确、processgroup、端点实际存在 | 全部 | P0-02/04 |
| data/pull_wb2.py | 9源变量抓取与保守重网格/缓存流程 | 不重写；先验实际坐标/变量/缺块/完整时间轴；header并行命令不能替代本机内存cap | 数据准入 | P0-01/04 |
| memory.py / ewma_error_feature_batched | 历史已核验记录/版本筛选与EWMA | 默认off；使用前修复捕获所有ValueError及zip截断，不能错版本归零 | 可选memory | P3 |
| spectral.py | 已计算复系数的诊断；SHT相关占位 | 不称完整SHT桥；linear vorticity/SHTcoeff可后续；energy是非线性端点指标 | 结构诊断 | P3 |
| scripts/s0_gate.py | 真实运行脚本但local-local、empty bank | 所有gatefailclosed，独立official、真实nonzero-bank zeroedit、禁autopip | S0 | P0-03 |
| tests/ | 随机小模型与skiprealassets测试 | 全部保留；新增针对确定性缺口的反例与actual入口测试 | 全部 | P0–P1 |
| plans/v6_draft | 多条候选+预写beats/zero-shot/2–3% | 新增当前decision与状态指针，不改写历史原文 | 研究决策 | P0-08 |
