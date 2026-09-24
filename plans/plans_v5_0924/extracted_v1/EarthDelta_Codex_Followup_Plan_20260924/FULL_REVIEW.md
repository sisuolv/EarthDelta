# EarthDelta FP-00～FP-04 执行复核及 FP-05 独立评审

日期：2026-09-24。**overall_verdict：PARTIALLY_EXECUTED。**

## 1. 决策摘要

本轮不是上一轮阻塞状态的重复：详细S0结果、Fs资格panel和bank独立验证，支持关闭对应的**有限技术准入目标**。
不应再安排重新训练Fs/专家作为默认下一步。但FP02的跨角色消费链未完，FP05曝光台账存在一个实质错误，现阶段不能直接提交正式缓存。

**最重要的新发现：X1-N64的实际训练已使用2020年7–12月56个新增issue。** 因而“2020H2是未曝光confirm”与执行证据矛盾。
其最终模型未被选中，不会消除其训练结果对研究设计选择的曝光。现在应保留2019H2作为受限定的回顾性开发评估，撤销整段2020H2的清洁confirm身份，未来确认窗口另行审定，不在本轮自动下载新年份。[E14/E15]

## 2. 读取与证据边界

分支最新为 `08093650ba56e8cf709d75a61c26c50a0c35f6f7`，核心源码树与指定 `e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a` 一致，只有后续文档不同。以e0136d4的实际代码/结果为受审范围。
已先完整读取上一轮四个验收文件和当前执行prompt，再检查FP01、Fs v1、探索、v2和bank过程，最后评审FP05。
核查实际源码、部分diff/树、测试全文和详细JSON，未逐行审查所有大型模块、每个大型admission或完整原始GPU目录。

没有在本轮运行仓库pytest或GPU。提交中的766 passed/7 skipped、861 passed/7 skipped为实际测试日志，不是本轮复验。
新S0/Fs/bank有真实数值检查产物，不能说只有argv；但多数正式job的原始`job_result.json`和完整logs在ignored artifacts中，尚未取得。
因此将“读到数值执行记录”和“独立校验调度receipt/权重字节”分列。X1-N64结果中明确嵌入真实job_result，包含returncode、耗时、主机与起止时间。[E00–E03/E13/E14]

## 3. 分阶段验收

| 阶段 | 原合同 | 代码/CPU | GPU与目标 | 判定 |
|---|---|---|---|---|
| FP00 | 真实起点、旧包合同、资产/证据清单 | entry命令及source清单存在 | 不涉及科学结果 | DONE库存范围 |
| FP01 | 独立官方1/4/12步parity，不放宽1e-5 | input inverse原dtype保留，TF32关闭；有针对性测试 | success gate三个max=0，四臂trace对应相同状态；receipt需补原件 | DONE注册数值范围 |
| FP02 | 真实fail-stop、消费链、薄runner、profile不变 | dispatch失败break；Fs认证loader有反例；profile深拷贝已FP04补做 | 已在后继运行消费；没有完整跨角色split机制 | PARTIALLY_DONE |
| FP03 | 唯一合格Fs，独立重载与共同参考 | fs_protocol/资格规则/认证load存在，766/7历史log | v2 H32过规则；训练-2.594%，16例资格集+0.3393% | DONE有限面板认证，不是OOS benefit |
| FP04 | 同Fs K4/r4、资格、持久化/组装/重载 | source/initialFs/slot绑定；861/7历史log | 同Fs，E3四专家严格<1，A1–A5全0注册probe | DONE有限bank认证，OOS utility缺失 |

### 3.1 S0不是放宽容差

`earthdelta/bridge/stormer_bridge.py:167-400`新增`_reverse_inp_mean/std`，保存源NPZ常量，再按源精度构造inverse；
`scripts/s0_gate.py:115-130`在加载前关闭两项TF32。详细gate中容差仍为1e-5，三时效max_abs_diff都0.0，committed=true。
四臂同过程trace的对应状态/模型差也为0。没有发现改小网格、移除注册rollout或把内部一致性冒充全部官方parity的证据。[E04/E05]

这是“声明的软件、模型、输入、时效”的通过，不是所有GPU环境永久逐位确定性的定理。
`test_input_inverse_source_precision.py`复用辅助inverse函数，单独看它只证明源精度路径被选中；独立官方的数值证据来自实际gate/trace，而不是这个测试名称。

### 3.2 Fs问题的真正关闭范围

旧v1长训失败被保留，正控制在认证后端、lr=.01下仍严重失稳。因此不接受“未认证后端/TF32是全部发散唯一原因”的叙事。
后端匹配和TF32设置减少了执行差异；学习率、重复拟合与训练长度仍影响稳定性/泛化。v2是根据已曝光v1/X1结果修改后的新协议，不是把旧失败翻成PASS。[E06/E07]

v2 JSON创建02:04:04Z，sha文件明确写02:04:13Z冻结，V2J3提交记录02:09:45Z，执行/decider引用同一个SHA。
这支持本地“先固定规则后提交”流程；不是外部可信时间戳证明。DEV001只修有效TF32分类，DEV002改绑定到同一原始panel的重判文件；原FAIL/STOP应永远保留。[E07/E08]

真实均值：训练24h `.03187907820466652 → .03105212451750749`，比值`.974059673813342`；
2019资格集16例 `.03127357741727903 → .03137967485354686`，比值`1.0033925583521255`，14/16例变差。
它满足预注册2%容忍范围，**只是有限资格面板内未越允许上限**，不是总体无害/有益、不是统计非劣效证明。
四份相同训练复制验证的是数值复现，不把资格样本量从16变为64。[E09]

### 3.3 Bank确实堵住共同Fs漏洞，但没有样本外价值结果

新worker直接授权/加载同一个certified Fs，merged digest一致。它们不再自行fit另一份Fs。
E3硬规则是各own-group平均24h loss ratio严格小于1；结果分别约`.990075/.990894/.984983/.980241`。
组数为15/16/7/6，不是四个独立留出组。E3不是事后从1%放宽到0%；FP04协议在提交前已说明它只是最弱非平凡方向门。
BJ3第一次因relative decision path失效而拒绝，保留原FAIL；仅修绝对路径并用已登记infra retry再组装，没有改变E3阈值。[E10/E11]

A1/A2独立源专家与组装输出1/4/12步相等；A3 zero==Fs且!=F0；A4步骤5–12继续Fs与F0可区分；A5重载sha/digests一致。
代码`compare_probe_states`要求shape/dtype相同、torch.equal且maxdiff=0，不是设置一个隐蔽容差。
但A1–A5只覆盖预注册一个probe issue。不能从中推导专家库对任意天气输入都无害，更不能从own-group训练改善推导OOS收益。[E11/E12]

## 4. FP05六项判断独立结论

| 判断 | 结论 | 修改 |
|---|---|---|
| 2019H2 dev / 2020H2 confirm | **前半有条件同意，后半不同意** | dev须按真实support净化；撤销2020H2清洁confirm，加入X1曝光。confirm=null/未授权 |
| delta_min=.34%来自Fs损害 | **部分同意数字可保留，不同意原推导** | 明确定义为Fs-relative工程目标，不是新季节必然回本线；增加同批F0与OOF static比较 |
| F0背景轨迹 | **同意** | 不进入五候选、不影响选点/HPO；20%只是轨迹数+1/5，不是实测GPU时长 |
| 一次性N，预计128 | **同意当前简化选择** | profile/完整性可在揭示前按机械规则确定一次N；不把128当独立样本数或功效保证 |
| 主ridge、次regime | **同意** | nested HPO，static在每个outer train选择；无全体dev最优静态标签泄漏 |
| 双向blocked cross-fit | **有条件同意** | 只称回顾性跨时间OOF；完整support净化、fold内变换、内层调参、正确CI解释 |

### 4.1 曝光遗漏的实证反例

`plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/results/x1_results_X1_N64.json:1-35`含job `pt-t3bfl3e9`，returncode=0，elapsed_seconds=719.624，以及64训练样本的结果。
其已绑定manifest builder定义56个新增点：`779 + round(i*(1451-779)/55)`，时轴每步6h，对应2020-07-13T18Z至12-28T18Z。[E14]
这些新增样本间隔72/78h；按完整`t-12..t+72`支持，区间连续重叠。加入FP05拟定24h外缓冲后，重建区间为2020-07-12T06Z至2021-01-01T18Z。
本包重算的是源代码定义的时间几何，不是新weather实验；FP05a仍须对actual read-set逐条认证。
即使不使用连续区间推导，56个真实H2训练样本本身已足以否定“整个2020H2未曝光”。

只写两段H1曝光并让新split_freeze检查，会**严格执行一个错误台账**。规则强不弥补输入台账缺漏。
必须记录失败、探索、未选中模型、调试、资格筛选的结果曝光；区别metadata/content-only读取与真正gradient/outcome/design-selection曝光。

### 4.2 delta的准确数学含义

旧面板损害`h=(L_Fs-L_F0)/L_F0=.0033925583521254854`。
即使在同一面板，若动态改善用Fs分母，则回本需要`g=h/(1+h)=.0033810878144214`，不恰好等于h。
更重要的是新H2分布的h未知；若讨论ridge相对best-static增益，该差异更不能直接等于Fs历史伤害。

修改后分别输出：
- `gain_vs_Fs = mean(L_Fs-L_policy)/mean(L_Fs)`；保留`.0034`作为已选工程目标，明确非应用价值证明。
- `dynamic_increment = mean(L_static_OOF-L_ridge_OOF)/mean(L_Fs)`；方向证据以0为参照，但0不是实际价值阈值。
- `net_vs_F0 = mean(L_F0-L_policy)/mean(L_F0)`；用同批配对数据判断是否真正超过未编辑骨干。
最小实际效应、数值地板、MDE、指定alpha/power是不同字段。未登记72h损害准则前不能签发最终科学CONTINUE。

### 4.3 季节错配和时间方向

最终选中Fs/bank主要在2020H1训练，2019H2作为dev是**逆年份的回顾性跨季节评估**。即便选择器fold不重叠，也不是当年真实签发时可执行的前向回测。
该设计仍可测冻结模型/库对该目标季节的迁移；不需要因担心负结果而改到训练季节。若失败，应停止或收窄**当前H2目标设置**，不能推出所有天气/所有bank都无价值。
2020H2因X1已用于探索，其“未来确认”的解释又须单独撤销。forward-chaining不能自动消除冻结Fs本身训练年份在dev之后的问题。

### 4.4 OOF设计必须补全的入口

split/admission在计划、提交、worker加载、merge之外，还必须约束**fit、scaler/PCA、inner HPO、regime、prediction freeze、score和任何重新读数组的路径**。
不能把`data_role=policy_dev`临时改成bank_fit，以复用只接受bank_fit的Fs certifier。

毒化outer fold F标签时，应断言**F的OOF预测**不变。其他fold模型合法地用F做训练，它们的预测可以变化；要求整个OOF文件不变会错误拒绝正确cross-fit。
毒化测试还须创建内部身份自洽的两个synthetic fixture，不能只靠checksum早退假装证明fit没有读标签。

主ridge的lambda在inner purged CV或预先固定；所有变换仅训练fold拟合。每个issue的全部候选留同fold，F0仅报告。
block数必须由实际时间轴和预定block长度算出，不能把约25个当既定有效样本量；14日block在半年度通常只有约12–13个日历block，7日才约25个，均不保证独立。
固定OOF预测后的paired block bootstrap必须标“条件于已拟合OOF模型”，不假装包括完整训练算法不确定性；需要强泛化声明时增加CPU重拟合敏感性或未来真正确认。

### 4.5 Debug与数据扫描

每组首尾8个debug有该组专家的历史panel；并非每个专家对全部8例都有旧记录。先冻结anchor coverage map，再要求已有锚点精确匹配；其它组合做双worker重复和独立数值重算。
“先完整性扫描，再冻结一个早于任何数值读取的声明”顺序不自洽。正确顺序是先固定候选/替换/排除规则，再完整性读取，再固定最终N/列表；任何模型收益揭示只能在之后。
新CPU评分与GPU归约的有限精度检验界限必须提前登记，不能用S0容差代替，也不能把本不存在的历史loss作为精确锚点。

## 5. 最小后续动作

1. FP05a补receipt/本地hash，重建全历史曝光，撤销2020H2 confirm，保存修订理由；不重训、不跑dev forecast。
2. FP05b新增薄runner/split/cache/OOF与真正的负向测试，冻结同Fs/五候选、指标、fold、N规则和72h损害判断；不改17个pin。
3. FP05c仅8个exposed debug，满足历史anchor/跨worker/绑定，按盲profile确定一次N。
4. FP05d一次性完整cache；任何缺行/坏身份/失败分母问题不得“只留下成功样本”。
5. FP05e nested OOF，冻结预测；FP05f冻结后评分和条件统计；FP06只作本开发迭代决策，不能开启confirm或声称response novelty。

## 6. 现在不能宣称

不能宣称FP02全部完成、TF32是历史失稳唯一原因、Fs样本外有益或总体无害、bank样本外无害或有价值、A1–A5覆盖任意输入、2020H2未曝光、2019双向OOF是实时部署回测、direct-gain开发结果证明response factorization novelty。

**结论：技术基础有真实进展，下一笔预算应买受控FP05价值证据；但应先修曝光账本和评估合同，不能直接照当前FP05计划提交缓存。**
