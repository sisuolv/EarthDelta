# P0 生存取证：一个迭代、三个实验

本文件不是全项目DAG。E1/R2-P0-02、E2/R2-P0-03、E3/R2-P0-04按证据和授权逐项进行。R2-P0-01负责可信执行；R2-P0-05只读汇总。初始E1–E3全部LOCKED。

## 1. 固定统计对象

对注册起报i和候选a：
`g_i(a)=L_native(Y_i,Fref_i)−L_native(Y_i,Fa_i)`。
`L_native=Σq*((Y−F)/scale)^2/Σq`；q在注册lead/变量/原生lat/lon一次归一化，fit-only scale固定。

- Hindsight：`V_H=E_i[max_{a∈A_i} g_i(a)]`。
- 合法信息理想值：`V_I=E_I[max_{a∈A(I)} E[g_i(a)|I]]`。
- dev选定固定策略：`V_fixed=E_i[g_i(a_fixed_dev)]`；它不是已知总体最优静态值。
- 训练后的合法policy：`V_pi=E_i[g_i(pi_fit,dev(I_i))]`。
- 主要差值：`G_oracle_ref=V_H`、`G_oracle_static=V_H−V_fixed`、`G_policy_static=V_pi−V_fixed`。
- 分解增量：在同样输入/labels/仿真资格下的`V_dual−V_direct`，或达到已注册技能目标所需labels/cost的差。预先指定主量，不随结果挑维度。

同一A下`V_H>=V_I>=max_a E[g(a)]`。no-edit的物理g=0；若引入总成本净utility，上述值需全部用同一utility重定义，不能保留“no-edit无成本”的隐含前提。

## 2. 数据与legal information

输入清单必须逐字段声明：source、时间、availability basis、shape/units、内容hash、提取成本、是否各baseline可用。默认memory off/JEPA off；所有policy共享注册历史context。允许retrospective perfect-analysis，但标签可用性scenario不冒充真实运营时序。

原生缓存：`reference:[N,H,V,Lat,Lon]`、`edited:[N,K,H,V,Lat,Lon]`、truth同reference、status/cost表。可用按起报/候选分片与chunk节省内存，但不能只保存赢家。若因存储cap仅缓存可无损重算的摘要与指针，须保证完整原生评分、guard和E3所需数据仍可在授权内重算，重算计费。

head输入输出：legal context与训练labels分别存储。e/du可用预先声明的固定线性D降低输出维数，但最终policy按native目标结算。D下恒等式只等于D下损失，不自动覆盖完整场；记录summary/native gain与选项不一致率。虚拟全场输出无有效lift则BLOCKED该对照。

## 3. E1：小bank与完整有限候选上限

**假设**：合理非零库相对强静态仍有最低有用的hindsight空间。

**比较**：F0（背景）、冻结Fref/Fs的no-edit、同资源强静态适配、dev最佳固定候选、全注册候选hindsight。regime实现可留E2，但其分组必须fit-only。

**bank资格**：合法fit数据；标签/训练成本计账；非零可重复响应；零系数还原；冻结作用窗/幅度。singleton可做最小入口，但不验证组合H。若要组合，先注册有限pair并逐个模拟，禁止用singleton和加法代理冒充组合真响应。

**estimand**：上述G_oracle_ref/G_oracle_static，外加per-lead/variable guard和完整失败率。

**输出**：bank_manifest.json、candidate_registry.json、candidate_outcomes.parquet、prediction index、bootstrap_intervals.json、cost/label ledger、decision.json。

**映射**：下CI超过注册价值→HEADROOM_PASS，仅允许申请E2；上CI不足→STOP_CURRENT_BANK/PIVOT_STATIC；跨阈值→INCONCLUSIVE。没有阈值只出DEV_DESCRIPTIVE。reviewer可明确承担探索风险批准小額E2，标DEV_PROMISING，不伪装confirm。

## 4. E2：合法策略与分解区分

**假设**：I存在决策相关信号，双头在有限核验监督下有额外价值。

**核心比较**：no-edit、best-static、regime、direct-gain、dual e0/du；直接edited-forecast、mean/zero response作为同缓存低容量诊断。只有实现已存在或同预算允许才增加直接b/H诊断，不新建连续规划路线。

**四象限**：
| e | du | 角色 |
|---|---|---|
| true | true | 注册候选hindsight基准 |
| true | predicted | 响应瓶颈诊断 |
| predicted | true | 误差方向瓶颈诊断；真实du是额外模拟，不是免费serving |
| predicted | predicted | 合法双头策略 |

实际评分永远用选中的真实候选outcome。不能以predicted gain当实现收益；不能把真e/真u格放进deployable主表。

**公平性**：所有方法同I/registry/fit与dev机会、目标、标签子集、仿真资格及HPO额度。direct-gain的一条truth可产生全部候选标签；不按K倍惩罚它。若dual利用未核验样本的du预训练，对手也有同等仿真预训练/辅助目标资格。

**标签曲线**：只注册少量预算点；同点同label IDs、seed与可用simulation集合。分别报告given-bank与end-to-end成本，bank用过的labels不能从后者消失。若标签曲线置信区间不够，报告不足，不用更多候选冒充更多独立标签。

**指标**：V_pi与policy-static配对增益、regret=best feasible realized gain−selected realized gain、harmful-edit rate、no-edit rate、mean predicted-vs-realized gain；e0全场/响应span误差、du方向/大小仅诊断。无法稳定定义near-null响应cos时记N/A。

**输出**：crossfit_predictions、four_cell_scorecard、direct_gain_comparison、label_efficiency、information_ledger、selected_actions/action_commit、summary_fullfield_gap、decision。

**映射**：合法技能被有界证据排除→STOP_CURRENT_PLANNER；direct有效但分解增量被排除→PIVOT_DIRECT_GAIN；合法技能及分解增量成立→申请E3；CI跨阈值→INCONCLUSIVE。低全场e0 R²不单独kill。

## 5. E3：实际编辑是否必要（条件性）

**假设**：在特定有限资源内，真实参数编辑比同信息输出修正有明确优势。

**比较**：actual edit、virtual=reference+predicted_du、联合注册时效输出纠错、hold窗内反馈纠错。后两者不能只做最后一步弱订正；feedback起报后仅消费自身预测状态，零corrector还原Fref。

**公平性**：目标/labels/仿真资格/合法preview/HPO与窗口一致。参数数目对齐和实测成本前沿分别做；full-field虚拟输出需要明确lift；不以无限容量等价性替代有限成本测量。

**指标**：原生完整注册目标、后续guard、训练/推理/存储成本，所有失败机会计分；短期与晚期效果单列。

**输出**：correction_frontier、lead_variable_metrics、guard_harm、forecast manifests、资源账本、decision。

**映射**：反馈支配→PIVOT_OUTPUT；virtual非劣更省→收窄到response surrogate；编辑保留注册区间优势且guard合格→建议有界继续；不明确→INCONCLUSIVE。

## 6. dev / confirm、crossfit与一次性使用

1. fit用于训练bank、模型与归一化/聚类/固定输出基；dev用于预先有限HPO、profile/方差估计、冻结配置。已经S0/debug看过的片段标exposed/dev。固定bank训练标签不得包含任何外层policy评估块；最小版用bank未见的policy开发块做crossfit，不为每折重新训练bank。每外折的标准化/聚类/读出/HPO也只拟合该折训练侧。
2. 分组依据真实时间或天气过程，不按候选随机切分。同issue所有lead/candidate同fold。history/target跨fold的共享数据按注册guard处理，不能只有日历标签不同。
3. 默认把dev缓存用于E1/E2/E3探索，最终全部候选/政策、主指标/guard、代码、分析和阈值冻结后统一打开confirm一次。此时报告多方法比较要有预注册primary contrast或多重比较规则。
4. E1若先单独用confirm并影响后续方法选择，则该集变成exposed；E2/E3必须另有独立确认。不把“每个脚本只跑一次”当作整体一次性确认。
5. policy predict先持久化prediction/action hash，score随后加载targets；score不可修改模型、registry和actions。

## 7. paired block bootstrap与样本量

对同一注册机会下的两方法差值做配对，整个process/time block的全部起报/lead/candidate一起重采样。point estimator到底是issue平均还是process等权必须预注册；block bootstrap中保持同一目标权重，不因过程大小变动偷偷换estimand。

block长度/过程定义从依赖结构、最长history+forecast与pilot方差制定；不能直接宣称独立。报告有效过程数、每过程机会数、敏感性范围。候选/格点/高度不是独立N。

bootstrap次数、seed、CI类型、alpha、功效、最大confirm过程数均在证书登记。无必要先实施新的统计库，优先现有numpy/scipy/评测组件。CI程序需要合成可解例子与配对索引测试。

## 8. failure denominator与候选数问题

注册机会分母N固定。记录每条候选失败原因/重试次数/已耗成本。若预先规定deployable遇失败回退no-edit，其实际回退损失与已花成本进入policy指标；未规定则标INVALID或按缺失界限报告，不能默默丢弃。

完整oracle必须全部注册候选有有效结果。未知候选不能填0、忽略后仍称complete oracle；只报tested-subset lower ceiling需显著标记，不能用其失败杀死完整库。

增加K会提升hindsight挑选收益，可能完全不可利用。这不是可以随意扣掉的bootstrap偏差。冻结K/registry，报告事前嵌套子集；context/truth打乱维持时间依赖只作诊断，不从真实技能机械扣除。

## 9. delta_min、MDE与决策

全部待定值：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。

在简化独立过程近似下，检验零差异的MDE约为`(z_(1-alpha)+z_(1-beta))*sigma_paired/sqrt(N_process)`；实际非线性oracle及时间相关设计优先用pilot过程块模拟功效。若要证明超过delta_min，必须指定高于该边界的planned alternative。不能把MDE当delta_min，更不能在看confirm后据结果定阈值。

所有规则详见根目录STOP_CONDITIONS。缺证书只支持探索，不支持正式go/no-go。

## 10. 资源cap

profile得到单步、bank更新、候选轨迹、context preview、纠错与存储的实测成本；cap使用真实最大update/call/issue/candidate/重试与硬件资源。当前硬件小时、N、K、rank均不伪造。

预算分配必须保留E2份额。E3是条件支出，不在E2无信号时自动训练。必要依赖/资产缺失就BLOCKED，不自动下载或部署新服务。
