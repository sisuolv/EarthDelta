# P0_SURVIVAL_EXPERIMENTS — 只执行本迭代三项实验

ID说明：本包任务为R2I-01…05，不重编号、不替换历史P0-05/06/07。R2I-01为准入，R2I-02为共同资产；实验E1对应R2I-03，E2对应R2I-04，E3对应R2I-05。后三任务都初始锁定。

## 1. 共同对象与估计目标

冻结合法信息I、参考F_ref、bank、候选全集A（含no-edit）、窗口与continuation、固定半正定Q及实际成本C(a)。主目标优先使用已经声明的线性空间；精确full-field评分同时保存，不能用summary获益替代全场获益。

对起报i、候选a：e_i=Y_i-F_ref(I_i)，u_ia=F_a(I_i)-F_ref(I_i)，g_ia=2e_iᵀQu_ia-u_iaᵀQu_ia。成本可行集先冻结；若主目标是gain-λC而非单纯gain，λ、单位及所有比较必须共同冻结。原始gain和net utility都保存，不能只报告有利的那个。

- `V_PI = E[max_a g_ia]`：在完整有限库上的事后最优值。
- `V_I = E[max_a E[g_ia | I_i]]`：合法信息理想上限，不由oracle直接测得。
- `V_static = max_a E[g_ia]`：理想静态；实际基线是dev_select选定后冻结的a_static，报告其真实out-of-sample价值，不冒充已知population最优。
- `V_policy = E[g_i,π(I_i)]`：实际已训练合法策略价值。
- `Δ_PI = V_PI - V_dev_selected_static`：hindsight相对固定静态的增量。
- `Δ_legal = V_policy - V_dev_selected_static`：可部署策略增量。

比较策略若有不同额外成本，另以共同预算可行前沿或注册net utility评估，不把只运行一个选中候选的方法和oracle全库运行成本混为一谈。

## 2. Legal information / 数据权限

I包含在声明研究签发时点可用的历史与当前状态、固定candidate descriptor、允许的reference preview（实测计费）。不包含本起报未来truth、exact candidate outcomes、oracle e/du/b/H或真值选择后的ID。

默认是明确的retrospective initial-state实验，不宣称业务实时可用；memory=off、JEPA=off。归档ERA5初始场可作为统一实验输入，但不能把6h假定写成真实first_seen。响应仿真无需未来truth，但用来评测一个新起报的全库exact du仍是昂贵oracle诊断，不是免费serving输入。

固定ServingContext只含input路径/特征/候选及身份；训练fit与评分API读取labels。API隔离不是OS安全sandbox，必须加future-payload poisoning和无全局状态测试，不能凭函数名声称完全隔离。

## 3. Dev / confirm 与交叉拟合

### 角色

1. `fit`：可细分bank_fit/head_fit；bank训练与各head的标签来源和重叠必须记账。同一核验起点只算一份truth，bank用过的truth列入总成本。
2. `dev_select`：候选强度、static/regime策略、HPO、数值/方差校准与功效设计，只能在这里选择；必须在probe前冻结比较方案。
3. `dev_probe`：不与前两者重叠的时间过程，用于本轮有界开发判断；已查看后永久标exposed，不能后改名confirm。
4. `confirm`：封存，默认不生成含未来标签的分析表。最后方法全部冻结且owner明确授权才读取一次；可留下给下一迭代，不是本包强制消费资产。

数据较少时，可对开发区采用按过程的嵌套交叉拟合代替额外固定probe切分；每个外折的head/HPO/static/regime只用其训练/内层数据，候选库不能根据外折结果重训。跨origin的时间窗（history、最大目标、可选预览）要purge；同一过程/起报的candidate、变量和leads不跨fold。声称严格时间部署时采用时间前向训练，不用未来折训练过去起报。

R2I-03只输出开发结论。R2I-04/05使用相同开发缓存继续研究不会使其变成确认；所有确认性结论仅来自最后一次冻结运行。confirmation_ledger记录曝光、roles、所有方法hash与使用次数；数据已被看过就不得复用“唯一confirm”称号。

## 4. 实验 E1 — 选择空间与合法信号（R2I-02/03）

**Hypothesis：**一个合格小bank既有hindsight空间，也包含I可利用的编辑偏好。

**Comparison：**Fs/no-edit、dev-best-static、简单regime、cheap direct-gain、完整finite hindsight oracle。可加预登记随机策略作sanity，不是主胜利对象。

**Implementation：**全部预登记候选实际非线性执行；候选表冻结后，在相同cache上训练轻量ridge/小MLP；候选静态和HPO只用dev_select；主评分dev_probe。所有策略使用同可行候选集合，额外preview成本纳入。

**Estimands：**Δ_PI、oracle vs Fs、Δ_cheap，及各自配对区间。Oracle不报告为deployable模型。

**Result→decision：**oracle不足STOP_CURRENT_DICTIONARY；相对static不足PIVOT_STATIC；oracle强+合法信号→申请E2；oracle强+cheap失败→只允许一次另行批准的更强策略检验；CI跨阈值保留INCONCLUSIVE。

**Cap：**先测实际单轨迹/训练/caching成本，以预先K×起点数×lead数与内存/存储形成预算上界；不是凭空填GPU小时。资源不足时缩小的是预试验设计，不能看到确认收益后删候选/删事件。

## 5. 实验 E2 — 双头增量（R2I-04）

**Hypothesis：**仿真响应/核验误差分工在同信息同总资源下改善policy value或核验标签效率。

**Comparison：**完整e/du；决策投影/条件矩；direct-gain；direct-gain+相同仿真辅助；可在cap内用residual/skip版本的direct-edited-forecast。不得把对照故意实现为更重的裸全场预测网络。

**Four-cell：**true e/true du、true e/pred du、pred e/true du、pred e/pred du。同真实结果结算；前三含oracle信息的格子只诊断，不混进低成本部署成绩。

**Estimands：**相对best-static与最强direct-gain的策略价值、达到相同价值的核验origin数量/计算成本。辅助MSE/余弦只诊断。避免近零响应余弦虚报满分，分母不足记N/A。

**Fairness：**一份truth为所有已运行candidate产生gain，不按K倍算label；允许所有方法使用相同无标签响应模拟；报告条件于bank的controller标签曲线和包含bank训练标签的总账。label subsets按origin/过程选，绝不按candidate行抽取。

**Decision：**投影比完整e好→收窄为决策相关预测；direct-gain同好更省→撤回分解主张；所有合法策略均无实用信号→停止当前planner；若有可复核增量，才申请E3。

**Cap：**首先不新增仿真、不训练新的bank；head结构/优化步数/HPO数量预先记录。不允许通过添加JEPA或memory解释失败。

## 6. 实验 E3 — 参数执行必要性（R2I-05）

**Hypothesis：**在有限训练数据与真实运行成本下，actual parameter execution优于或比输出路径更有效。

**Comparison：**实际编辑；virtual=reference+predicteddu；反馈输出纠正；可同读出加入reference+predictederror。feedback逐步修正全变量并将修正状态反馈，不能做成弱的一步baseline。

**Estimands：**同Q主目标增益及资源前沿、guard lead/变量的非劣性。不能凭无限容量模拟等价性否定有限资源价值，也不能凭能改后续动力学声称参数独占。

**Fairness：**full-field输出分辨率一致；摘要du没有合理full-field readout时标NOT_COMPARABLE，不得宣布参数胜出。输入历史、hold与continuation、监督和HPO资格一致。参数量与实测成本分别做对照，不假装两者必能同时相等。

**Decision：**输出同好更省→PIVOT_OUTPUT；实际执行有注册增量且guard合格→建议下一迭代；长期损伤→缩窄/停止edit family；资源不足→BLOCKED。

**Cap：**owner独立批准反馈训练与多步评测预算后才执行。默认不增加240h和第二主干。

## 7. Paired bootstrap 与 MSE/RMSE

先按每个origin和每个冻结策略计算相同Q的损失，再按事先定义的过程/时间块共同重采样；同一次重采样对所有方法/candidates/leads使用同一块列表。重采样次数、block length、区间构造、alpha、power、主比较和多比较控制全部在confirm前冻结。块长参考开发数据误差/天气过程相关性，不按结果挑最有利块。

固定已训练策略的区间是条件于该训练/开发选择的评测区间；若要包括HPO/训练波动，需在重采样中重训/重选并另记成本。不能将简单evaluation bootstrap说成包含训练不确定性。

RMSE派生指标在重采样内先聚合weighted squared error再开方；不能平均逐像素/逐origin RMSE代替注册聚合。若主统计是MSE gain，所有门槛用其同一单位，RMSE另报不混。

## 8. Candidate-count / hindsight

K增大往往使E[max g]增大，包括不可预测噪声的选择收益。这不要求从oracle机械扣一个shuffled值，但必须冻结K、候选身份/支持、随机性和成本。singleton-only的失败不否定未覆盖联合组合；若要检验组合，提前登记且实际执行，不能事后救项目。

真值/上下文打乱诊断必须保留同origin的candidate配对和过程结构，标SANITY_ONLY。oracle=0.75/合法=0的反例必须写入自动测试，防止将HEADROOM_PASS_ONLY当最终CONTINUE。

## 9. Failure denominator

固定attempted origins及每个注册candidate状态。完成/有效计数双列；参考失败、candidate崩溃、nonfinite truth分别记录。缺候选时仅可报告tested subset的oracle下界或INCOMPLETE，不给完整oracle结论。若注册policy有fallback到Fs，执行并记其成本；缺loss不能填零；不得以“只看成功选择”删除困难事件。

模型无法运行是实现/资源结果，不是预报无收益；bank无效是INVALID_BANK；样本不足是INCONCLUSIVE。只有在合格设置中得到充足区间证据，才做scientific STOP/PIVOT。

## 10. Threshold certificate

`delta_min`与MDE均未指定：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。证书包含：价值依据、cost penalty单位、effect定义、开发方差/相关性、目标alpha/power/N/MDE、最大N、block length、查看规则、guard margin和确认角色。使用pilot variance选N或评估可行性，不以MDE替代价值。

若MDE大于delta_min，只能说明设计目前不足；不要自动提高delta_min来制造“可检测”。CI跨delta_min→INCONCLUSIVE。所有内容freeze后hash进入每个结果。

## 11. 必须输出的实验资产

每个任务：config.json、metrics.json、summary.md、stdout.log、decision.json、git/source/diff身份。
共同：candidate_registry.json、bank_manifest.json、split_roles.json、cache_manifest.json、candidate_outcomes.parquet、向量target索引、合法context索引、failure_denominator.json、cost_ledger.json。
统计：threshold_certificate.json、paired_intervals.json、确认使用ledger。decision必须同时有evidence_grade（DEV/CONFIRMED/STRUCTURAL_ONLY）、estimand、比较对象、区间和阈值来源、stop_reason、next_task_eligible、next_task_authorized=false。

所有值来自真实运行；未运行填写NOT_RUN，外部资产缺失填写BLOCKED，不填之前的CPU分数或合成天气收益。
