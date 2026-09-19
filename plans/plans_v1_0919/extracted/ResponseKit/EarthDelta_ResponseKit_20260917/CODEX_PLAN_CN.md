# EarthDelta：响应交互驱动的轨迹修正——研究与代码实现方案

日期：2026-09-17。本文继续此前的响应空间路线，不再另起一个大型框架。

**状态：研究设计 + 已完成的基础代码原型。不是已经训练成功的天气方法，也没有证明首创性。**

## 0. 本轮决定

保持冻结天气模型、非零低秩修正字典和显式多步rollout。将“响应驱动”具体化为两件可测量的事情：

1. 修正的未来收益：给定当前合法历史，哪次修正可能改善后续多个时效？
2. 修正之间的响应重叠：两次修正是否重复、抵消、互补？预算有限时，不应只按单项分数各自取最大。

推荐工作表述：**EarthDelta: Interaction-Aware Forecast Trajectory Repair with Amortized Response Models**。

主贡献不是发明LoRA、Gram矩阵、Gauss–Newton或稀疏QP，而是研究：在冻结天气模型上，能否用受控的训练侧响应实验学习可部署的修正价值/组合模型，使少量修正比普通条件适配和同预算多步训练更有效。

两个学生路径共用同一基础，不串成更深系统：
- 最小验证路径：历史 -> 直接程序头 -> 有界修正；以同教师的系数/响应蒸馏比较验证响应监督。
- 拟议主方法路径：历史 -> 预测b/H -> 小型受约束规划器 -> 有界修正；以完整H/对角H比较验证修正交互。

若完整H并不比对角H或直接程序头更有效，则删掉交互复杂度。暂不开展跨backbone响应迁移、长期记忆、TTT、动态rank、额外JEPA或重型谱损失。

## 1. 研究近邻与复用分工

详细链接、日期和实际阅读范围见随包SOURCES.md。

| 来源 | 已有内容 | 本项目需要额外证明的内容 | 代码使用方式 |
|---|---|---|---|
| CLAW，2026-09-10 [S1] | 近期转换到hypernetwork生成LoRA的世界模型适配 | 不仅把历史变成参数；修正价值、组合与时机是否有增量 | 未确认可直接复用官方代码，不设为前置依赖 |
| PEFSO，2026-09-10 [S2] | 观测敏感度近似预报影响、减少重积分 | 内部参数修正的可部署选择；与观测同化任务分开 | 参考近似有效范围和非线性核验，不直接迁移气象结论 |
| PEAR，v2 2026-05-19 [S3] | 决策相关投影误差与约束/曲率 | 响应损失是否比系数误差有用，不能移植理论保证 | 作者仓库链接已找到，未逐文件审查，不作为已接入组件 |
| CoMoL，v2 2026-06-04 [S4] | 共享A/B与小core空间动态混合 | 不把便宜的动态混合作为主创新 | layer.py实际读取；可作第二阶段高效实现和近邻基线 |
| DISeL [S5] | 输入相关低秩门控 | 排除普通动态router的解释 | 参考门控与保存逻辑；保持移植/官方名称边界 |
| Solver-in-the-Loop [S6] | 修正后状态分布和多步训练 | 排除“多步训练本来就能学会”的解释 | 借鉴实验协议，不引入旧TF/PhiFlow环境 |
| GEPS [S7] | 物理系统上下文低秩适配 | 受控方程中的修正方向/时机机制 | 参考PDE生成器，独立审查与数值收敛测试 |
| Stormer [C1] | 完整天气预测底座 | 只实现新修正接口，不重写气象流水线 | 原权重、数据约定、增量还原直接复用 |
| WeatherBench-X [C2] | 权重/区域/指标/聚合与数据装载 | 自己只补修正收益、成本和配对统计 | 标准预测导出后独立评估，可本地运行 |

这是定向查重，不是穷尽式首创证明。仅改变领域、标题或加入多个现成模块不能建立novelty。

## 2. 定义研究对象，防止信息条件混淆

起报合法信息I_t包括X[t-12h:t]、静态场、已知步长与预算。模型输入保持Stormer原生变量；未来真值Y只进入训练标签/离线评估。

固定backbone参数theta0和已训练的非零bank psi。程序a属于R^d，描述预先定义的层/时间/区域槽位的系数。首版a0=0表示完全不执行该bank。

G_x(a) = D(Rollout(F_theta0,psi, I_t, a))。

D固定为多变量、多时效、多尺度的摘要或下采样向量；不是只优化最后一天，也不按测试误差挑区域。Q定义面积、变量尺度、时效等非负权重，各项只应用一次。

所有推理路径必须分开：
- 原模型基线：固定6h路线；不额外平均6/12/24h路线。
- 开环学生：起报时生成整段程序，后续只读自身预测。
- 预测前缀重规划：可以基于自己产生的前缀重新决策，但需要新的训练协议和计算账本。
- 重新获得真实分析场的滚动更新：另一任务，不混入开环成绩。

训练阶段使用ERA5的合法时间顺序，不等于证明业务时刻能够取得所有同化后的分析场；不要把回顾性ERA5设置称作已部署实时预报。

## 3. 低维响应与修正交互

在a0附近用中心差分求响应列：

R_j = (G_x(a0+epsilon_j e_j)-G_x(a0-epsilon_j e_j))/(2epsilon_j)。

e = D(Y)-G_x(a0)，b = R^T Q e，H = R^T Q R。

局部平方误差收益：

Gain(a) ≈ 2b^T a - a^T H a。

本文首版a0=0，a既是offset也是绝对系数。非零a0时公式用于delta=a-a0，必须同时约束a0+delta；原型盒约束只约束offset，不能静默改协议。

H_ii：当前单位修正响应的加权大小；b_i：该响应和训练残差的对齐；H_ij：两个响应重叠。符号与所选系数共同决定交互，不能笼统把所有非对角项解释成抑制。

重要边界：
- 这是线性化预测映射下的二次模型，不是新数学结论。
- H不等于真实非线性损失Hessian；完整二阶项还涉及残差加权的G二阶导数。
- 两列正交/相关是模型输出空间关系，不是大气过程独立/因果关系。
- d=8自然导致rank(R)<=8，不能据此宣布发现天气纠错本质低维。
- 响应R不需要未来真值，收益b需要；在线计算R也不能绕过未来未知的问题。

## 4. 首版程序：先让总d=8真的可执行

建议最小配置：4个已核查的后部层[18,20,22,23]，各一个rank16低秩方向，作用时刻为预测步0和2（6h固定步长下为第1次与第3次调用），全域支持，共8个标量自由度。

这一首版把where暂时限定为网络层，不宣称已经学会地理区域选择。地域支持扩展在第二阶段单列；不可同时把每patch8维展开成巨大程序。

为了让每个槽位成本清晰，初版可以将旧StateGatedLinear的groups设为1；rank仍为16。旧groups4的实验保留为另一配置，不覆盖旧结果。每个(layer,step)最多一项整分支调用，后续真正跳过该分支才有实际成本意义。

组件边界候选rho=0.25仅为原型默认值，不是气象最优值；在训练/验证侧结合已训练bank的函数输出尺度、epsilon稳定性选择并冻结。不得比较不同bank任意缩放下的系数幅度并称为公平。

## 5. 字典训练与冻结

最小策略：相同数据、相同多步规则先训练静态LoRA得到非零A/B。保留该静态模型为基线；响应与学生阶段固定bank，不更改基础模型或bank。

B零初始化能够保留基模型，但此时对系数的响应也是零。因此响应采集的前置条件是非零bank、有效输出变化及可重复前向。

已有旧层内部router是2*sigmoid，不能自然表示负系数。新学生使用有界tanh，经coefficients显式接口绕过旧router；教师和学生同域。

扩展时可用CoMoL的共享A/B与core C_k，使不同方向只增加小矩阵。但首轮保持旧bank，不同时更换字典和训练监督。C_k的r×r矩阵与程序H的d×d矩阵完全不同，不混同。

若在后续共同多步微调中解冻bank：停止复用旧R/H/b，或明确重新生成响应；不能以旧几何约束变化后的模型。

## 6. 离线教师：小型求解，完整非线性复验

### 6.1 连续参考先行

求解带ridge的加权最小二乘，并施加分量盒约束：

min_a ||sqrt(Q)(e-Ra)||^2 + lambda||a||^2，|a_j|<=rho_j。

实际复用SciPy lsq_linear；不要重写一个未经测试的优化器。这里是盒约束，不是旧提案中的L2信赖球；可通过局部幅度校准控制范围，但不能声称两者等价。

首先全部维度连续优化；确认教师有完整非线性收益后，再比较稀疏程序。

### 6.2 预算支持枚举

若总d=8，每次最多2个单位成本槽位，支持数为1+8+28=37，包含无修正。36个非空支持分别求小问题。这只是计算不同系数组合，不代表需要重新运行天气模型36次。

cost目前为声明的加性单位成本；真实运行必须替换为实际执行组成本。共用A/B或同一层/步多个方向的成本有共享，不能逐系数重复收费。

### 6.3 非线性核验

用共同未压缩天气指标重跑排名靠前的少数候选，例如3个，并保留无修正。摘要收益高但全场恶化时拒绝；求解失败、非有限输出、无收益都保留在清单。

真值协助的接受/拒绝只用于训练教师或离线诊断，不能用作部署时的“安全开关”。最多测试了几个候选的最佳结果称为“已测试候选参考”，不是全局oracle；学生超出这一有限候选参考并不矛盾。

### 6.4 探针预算与缓存

d=8中心差分17条完整轨迹；默认重复基线测试再加1，实际18。每条24h包含4次6h模型调用，因此完整从头重跑时为72次单步调用；top3复验的独立实现另外重跑基线+3=4条轨迹，即16次，合计22条/88次，不含epsilon与混合方向诊断。

当前原型重复基线而非自动复用。可以在冻结快照验证后复用基线，但必须如实更新成本；不能用17个单步描述17条轨迹。

先流式生成R与教师，再只保留训练需要的H,b、候选标签、标识和少量诊断R。按d8与m约几万，存所有样本R会明显放大磁盘；不要默认缓存全层激活或全部天气分支。

缓存至少绑定checkpoint、bank、输入样本与预处理、program spec、reference、D/Q、epsilon、dtype/backend；bank或任何科学字段变化即失效。

## 7. 学生：先直接蒸馏，再验证交互模型

### 7.1 直接程序头（快速路径）

旧TemporalPatchState产生z[B,N,128]，按真实面积加权形成z_pool[B,128]，BoundedProgramHead输出a[B,8]。比较同架构同教师：

L_coef = ||a-a*||^2；
L_resp = (a-a*)^T H (a-a*)。

教师系数可能不唯一，响应等效不应强制逐系数一样。仍需保留小型系数正则限制响应零空间中的大系数，以及共同多步预测损失。对H按训练数据固定尺度归一化，避免不同天气过程仅因原始误差量纲支配学习。

### 7.2 交互收益头（拟议主方法）

同样z预测b_hat[B,8]和H_hat[B,8,8]。使用L的36个下三角参数，H_hat=L L^T+eta I，总输出44个标量。softplus正对角只是保证PSD，不是准确性证明。

初步监督：b标签、H标签、在固定有界候选集合上的收益回归/排序。b/H标签生成仅在训练侧。更重要的是用已经非线性核验的候选实际收益做独立选择质量诊断，避免H拟合看起来好但决策差。

候选非线性收益并非总能由PSD二次模型表达。先测近似误差；不立即再叠大型score network。若线性区间不成立，缩小范围、缩短局部作用窗或回退直接多步控制器。

### 7.3 在线选择

plan_from_prediction只接受预测b/H、幅度与预算，枚举允许支持并求小型盒约束QP。程序实现复用SciPy L-BFGS-B及解析梯度。没有通过优化器反向传播，也没有在线调用教师或真实未来。

d8、max_active2最多36次很小的求解，但是否相对天气模型足够便宜必须实测。对batch可先逐样本CPU选择；不在起步阶段引入可微QP/强化学习/大型分布式搜索。

无修正候选始终存在。门槛可以在验证侧校准，报告误修正率与覆盖率，不宣称严格的未来风险保证。

### 7.4 开环与重规划

原型是起报时预提交程序。若升级成每步控制：状态必须包括自己预测的历史、剩余预算、已执行支持与时效；禁止按日真值补充。收集学生产生的前缀重新建立相应训练响应，或统一多步训练。基于真实分析场建立的缓存标签不能直接用到不同预测前缀。

## 8. Stormer代码接入：复用什么、必须改什么

保持上游提交58dfee5a6037399a40fefd492bc00421e0c885a8。本轮重新核对远端main一致。[C1]

### 8.1 加载与前向

原样加载checkpoint -> 完整校验参数名/shape -> 冻结backbone -> 插入模块。禁止插入后重新initialize_weights或用strict=False吞掉原模型未加载字段。

首版原生69变量、128×256、匹配patch4权重；确认权重来源与数据经历。展示7个变量不意味着只输入7通道。保持归一化增量还原、零增量均值和interval=hours/10。

`stormer/models/hub/stormer.py`：
- Stormer.forward接收新增显式program/step参数；无程序时原路径不变。
- Block.forward继续保留原time_interval adaLN；向Attention传递必要的系数/执行标识。
- MemEffAttention使用fused attn.qkv；适配目标为选定attn.proj，不套用LLM q_proj/v_proj名称。
- StateGatedLinear接收coefficients，不隐式读取全局变量或可变hook状态。

### 8.2 roll-out桥接

`stormer/models/iterative_module.py`中的padding、增量反归一化、加回当前物理状态与下一步归一化直接复用，封装为显式WeatherStepBridge。

未来目标只传给训练loss/evaluator，不是ControlledRollout参数。建议接口：

```python
# 待实现的天气桥接API，不是当前包已提供的函数
trajectory = controlled_rollout(
    bridge, initial_history, program_spec, coefficients,
    horizon_steps=4, return_leads=(1, 2, 4),
)
# labels 不允许出现在这一路参数中
```

每次中心差分从相同初始快照启动，不能接着前一分支的history/controller状态跑。除输出外，不修改输入tensor、随机状态约定或隐藏账本。

### 8.3 真正跳过分支

当前旧层即使coefficients全零仍计算A/B。新的执行器在无激活程序时调用base投影而不调用A/B。若batch只有部分样本激活，先选择同支持microbatch或显式gather/scatter；CPU/GPU同步与散射开销要计入。

训练阶段全部系数初始零时，不能沿硬跳过路径切掉学生梯度；训练用dense可微路径，正式推理用显式选定支持路径，做数值等价和实际算子计数测试。

### 8.4 梯度与缓存

冻结权重不等于整个forward可no_grad。第一处adapter前的纯冻结前缀在特定单步原始输入条件下可缓存；多步全反传时下一步输入依赖上一轮适配，需要保持相应梯度。截断策略应让所有基线一致。

使用checkpoint重算时必须显式传入当前step与coefficients；可变模块属性会导致重算使用错误状态。所有checkpoint保存bank、student、program和原模型identity，不复制巨大权重多份。

## 9. 标准评估复用WeatherBench-X

官方示例实际使用[C2]：
- xarray_loaders.PredictionsFromXarray / TargetsFromXarray；
- deterministic.RMSE() / MSE()；
- weighting.GridAreaWeighting()；
- binning.Regions；
- aggregation.Aggregator；
- 本地DirectRunner或后续分布式runner。

先把预测导出为规范xarray/Zarr：init_time、lead_time、latitude、longitude，以及对应变量的level轴。valid_time=init_time+lead_time。温度、位势、风和湿度单位与原始目标一致，不能把位势当位势高度混用。

先在小型数组上比对本地加权MSE与WB-X，再上真实数据。RMSE应对完整加权平方误差聚合后开方，不能用逐batch RMSE平均替代主指标。

自定义只补：真实净修正收益、误修正率、无修正率、候选参考差距、计量成本与配对bootstrap。WB-X不是训练loss模块，不为部署额外引入全部评估依赖。

缺测/非有限预测要保留分母与失败计数，不能各方法用不同有效样本悄悄比较。训练阈值定义的初始复杂区域与按未来结果定义的极端过程分开；不因高湿区有改善就称降水技巧改善。

## 10. 首轮实验与否证条件

共享：同checkpoint、变量、预处理、合法历史、bank容量、训练/验证/测试分期、共同多步loss、HPO预算和评估集合。

| ID | 方法 | 优先回答 |
|---|---|---|
| B0 | Frozen | 基线 |
| B1 | 静态LoRA | 普通适配是否足够 |
| B2 | 同历史temporal router + 多步loss | 普通条件适配和未来梯度是否足够 |
| B3 | 旧H+R + 同多步loss | 残差辅助监督是否足够 |
| D0 | 同教师 + 系数蒸馏 | 教师额外计算的作用 |
| D1 | 同教师 + 响应蒸馏 | 功能空间监督的必要性 |
| U0 | 预测b + 对角H规划 | 单独价值/幅度模型 |
| U1 | 预测b + 完整H规划 | 修正交互的必要性 |
| O0 | 相同历史的输出纠错器 | 内部参数修正是否必要 |

不要求首批同时跑全部九组。先共享bank做D0/D1以及用同一真值局部几何full/diag的诊断；通过后加U0/U1与完整强基线。教师参考只在独立离线表报告，不放入可部署模型主榜。

教师样本必须覆盖无收益案例，不按未来正收益筛掉困难样本。B0与B1训练不同是合理参考，但D0/D1/U0/U1必须匹配bank和信息。

统计按起报时间块/天气过程配对；不是按网格点独立bootstrap。日期外推不自动等于OOD。正式确认数据从未用于选bank、D/Q、阈值、边界、时刻、损失系数或超参数。

关键gate：
- 非线性教师无稳定全场收益 -> 不扩大学生，不继续加memory。
- R局部准确、学生差 -> 检查历史能否识别b/交互，避免归因给求解器。
- full/diag没有区别 -> 去掉交互头，保留更简单方法。
- D1与D0相当 -> 缩小“响应蒸馏独有收益”的主张。
- B2相当 -> 明确当前额外机制未证明价值。
- 摘要好全场差 -> 修正目标映射，不挑选有利变量隐藏失败。

## 11. 建议Codex任务与提交

任务序列是验收顺序，不是时间承诺。所有GPU调用和数据下载需先明确本地真实资产与预算；不得改动DisasterTrace仓库。

### R00：保存事实起点

重新运行旧21测试与本包测试；记录实际工作目录、依赖、checkpoint是否存在。两包合成测试通过不等于真实集成。

提交：`test: preserve starter and response-kit invariants`。

### R01：显式受控Stormer桥接

新增`stormer_bridge.py`、`controlled_rollout.py`，固定原归一化/变量和6h路径。原版预测与零程序逐元素容差等价；跨分支、跨batch、checkpoint重算不串state；真值无法进入rollout。

先仅用已有少量真实合法天气样例，不全量下载。

提交：`feat: add explicit controlled Stormer rollout`。

### R02：非零bank与程序支持

新增`fit_bank.py`和bank状态报告；检查B非零、冻结参数不变、selected layer名shape。让8维程序在对应层/步生效，外部系数与显式矩阵一致。

提交：`feat: freeze nonzero repair bank and eight-slot programs`。

### R03：响应采集与教师闭环

新增`collect_responses.py`、`collect_teacher.py`和最小缓存索引。调用本包central_response、box_candidates、verify_candidates，不重写求解器。

先少量跨季节起报，24h探针；epsilon/2、epsilon、2epsilon；至少混合方向检查；float64累计和dtype回执。教师完整全场收益、无收益、失败全部输出。

提交：`feat: add offline response collection and verified local teacher`。

### R04：直接学生与同教师对照

新增`train_direct_student.py`，复用旧TemporalPatchState与新BoundedProgramHead。相同数据、同教师分别系数蒸馏和响应蒸馏。真正推理函数无truth/teacher/H标签输入。

提交：`feat: train coefficient and response-distilled program heads`。

### R05：修正交互与部署规划

新增`train_utility.py`、`infer_program.py`，使用本包InteractionUtilityHead和plan_from_prediction。先验证预测b/H，再比较同预算U0/U1。用真实算子调用计数建立执行组账本，dense与sparse执行输出等价。

提交：`feat: add label-free interaction-aware repair planning`。

### R06：评估与冻结确认

新增`export_wbx.py`、`evaluate_wbx.py`、`paired_analysis.py`。6/24/72/120h全部方法共同评估，记录全部成本和失败。更长时效不能未经验证直接使用24h局部近似替代真实rollout。

提交：`eval: add WeatherBench-X export and paired repair audit`。

### 延后事项

CoMoL core实现、地理区域程序、每步重规划、第二backbone与响应迁移分别立实验，不让首批依赖它们。每项都需清晰未解决的问题和固定baseline。

## 12. 本轮真实代码状态

已提供七个模块及合成测试，实际模块表见README_CN.md。新增utility head和在线规划器是基础实现；没有训练student，也没有weather bridge。

原包21项CPU测试重跑通过。新包最终精确测试数、版本与运行状态以TEST_REPORT.txt和EXECUTION_STATUS.json为准。本轮测试包括BF16预测下保持真值精度的回归检查；不等于GPU BF16模型已验证。

本包没有复制上游代码或权重，无用户账号/密钥。引用、许可与具体版本分别见SOURCES.md、VERIFIED_UPSTREAM.json。

## 13. 新代码使用的真实示例

```python
# 本包已提供：offline_callback必须由未来的天气桥接实现。
from earthdelta_response.probe import central_response
from earthdelta_response.geometry import ResponseGeometry
from earthdelta_response.teacher import box_candidates, verify_candidates

probe = central_response(offline_callback, reference, epsilon=0.02)
geom = ResponseGeometry.from_error(
    probe.response, target_summary - probe.reference_output, weights
)
candidates = box_candidates(geom, bound=0.25, max_active=2, budget=2)
# full_callback返回固定全场校验向量；本段仅限训练/离线。
verified = verify_candidates(
    full_callback, reference, candidates, target_full, weights_full,
    max_nonzero=3,
)
```

```python
# 本包已提供：feature来自待实现数据/天气接口和已训练状态编码器。
from earthdelta_response.utility import InteractionUtilityHead, plan_from_prediction
head = InteractionUtilityHead(features=128, dimension=8)
# 真实部署之前必须加载已经训练并独立验证的head；下面仅说明API。
b_hat, h_hat = head(feature)
plan = plan_from_prediction(b_hat[0], h_hat[0], max_active=2, budget=2)
# plan.coefficients交给未来Stormer受控rollout。这里没有访问真实未来。
```

运行原型真实命令：`python -m pytest -q`、`PYTHONPATH=. python examples/synthetic_probe.py`。计划中的天气脚本尚不存在，不能宣称已可运行完整训练。

## 14. 论文中心句与声明边界

建议中心句：

> EarthDelta从合法大气历史中学习低秩修正对未来预报的收益与相互影响，并在共同执行预算下选择有界修正程序，检验针对性轨迹修正能否比普通条件适配更有效地改善多步天气预测。

若实验只支持响应蒸馏，不支持交互规划，则据实收窄标题与贡献；若只有受控PDE支持则不能扩大为地球系统发现。没有因果发现、真实天气控制、全局最优修正或通用安全保证的声明。

## 15. 来源

[S1] CLAW: https://arxiv.org/abs/2609.12278
[S2] PEFSO: https://arxiv.org/abs/2609.12296
[S3] PEAR: https://arxiv.org/abs/2605.01361 ; author code https://github.com/FinJun/PEAR （仅链接确认）
[S4] CoMoL: https://arxiv.org/abs/2603.00573 ; https://github.com/DCDmllm/CoMoL
[S5] DISeL: https://arxiv.org/abs/2605.19028 ; https://github.com/alizindari/DISeL
[S6] Solver-in-the-Loop: https://arxiv.org/abs/2007.00016 ; https://github.com/tum-pbs/Solver-in-the-Loop
[S7] GEPS: https://geps-project.github.io/ ; https://github.com/itsakk/geps
[C1] Stormer: https://github.com/tung-nd/stormer/tree/58dfee5a6037399a40fefd492bc00421e0c885a8
[C2] WB-X example: https://github.com/google-research/weatherbenchX/blob/19548af79e75190254bba273ff5e3b179265e0bd/evaluation_scripts/run_example_evaluation.py
[C3] CoMoL layer: https://github.com/DCDmllm/CoMoL/blob/main/src/mocorelora/layer.py （读取blob9aa7a4aa9d0a0e796115e722b2a552638741b931）
[C4] PyTorch JVP: https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html
[C5] SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
[C6] WeatherBench2: https://weatherbench2.readthedocs.io/en/latest/data-guide.html
