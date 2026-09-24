# EarthDelta：FP-00～FP-04执行复核与FP-05独立设计审查

## 1. 总体判决

**overall_verdict = PARTIALLY_EXECUTED。** 与上一轮不同，指定配置的S0数值阻塞已经解除；确有Fs v2资格与共同Fs K=4 bank计算证据，不能继续把旧S0 FAIL或“每个worker各训Fs”套到当前认证资产。另一方面，FP02完整消费链仍未完成，FP05仅为计划，bank样本外效用和novelty仍是INCONCLUSIVE。

最需要立即修正的新问题不是重训，而是 **FP05把2020H2预留为未触碰confirm，却遗漏了X1 N32/N64已经在该半年训练并分析过的事实**。[R18–R21、R41–R42；FP05_PLAN.md:19]

当前实际HEAD为08093650ba56e8cf709d75a61c26c50a0c35f6f7，较用户给的e0136d4只增加审查prompt。代码与计划审查固定到e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a，新增实现属于bc57a70；实际验收合同为仓库extracted_followup中的FP编号版，不拿更早聊天里的另一套编号替代。[R01–R04]

## 2. 实际证据边界

本次读取验收入口/任务/后续计划/STOP、实际执行prompt、S0原始计算JSON与trace片段、Fs v1/v2协议和资格、X1结果和manifest、bank协议/assembly/verify/decision、偏差文件、关键源码和历史CPU日志。SOURCE_INDEX逐项列出文件、固定commit与已读范围；没有逐行复核整个仓库、全部长trace或全部未上传artifacts。

766 passed/7 skipped和861 passed/7 skipped来自亲读历史pytest日志，非本次重跑。S0、Fs和bank的计算记录支持真实数值执行；当前S0/Fs-v2/bank的独立平台job_result原件未读到，部分rc/elapsed只有task_result摘要。X1的results JSON例外：内嵌原job_result含hostname、起止、returncode和elapsed，本次确已读到。[R27、R38、R41–R42]

未重hash大权重、未重跑官方GPU、未独立认证平台时间戳。因此精确区分“计算记录可支持”“原始平台回执完整”“本次独立复现”。缺收据先补只读小文件，不因此要求重训。

## 3. FP-00～FP-04四层对照

| 阶段 | 原要求 | 当前代码 / CPU | GPU / 计算证据 | 目标判定 / 限制 |
|---|---|---|---|---|
| FP00 | 只读身份和材料清单 | entry_evidence真实git命令rc0 | 不要求GPU | DONE于inventory；平台回执与全曝光账本还需补齐 |
| FP01 | 原ps4、policy、1e-5，1/4/12官方parity | 源精度input inverse、正式NPZ工厂、TF32设置与真实gate | pt-g3344e9z；s0_gate_result全部16 criteria true，1/4/12差0 | DONE于指定数值配置；不能泛化其它backend/所有初态 |
| FP02 | STOP/admission/Q/registry/profile/薄runner全部消费 | real stage-all禁止，失效后继spy0，Fs相关admission，FP04新增profile deepcopy | 子集实际在Fs/bank使用；完整统一门未交付 | PARTIAL；不能因FP03/04推进而全打勾 |
| FP03 | 质量门+唯一Fs+独立重载 | 协议CLI/source/cert binding、质量判据、冻结/重载 | v1两formal FAIL；v2 pt-fuiqqeau+pt-gr3rrmdg选Fs | DONE于v2 H32工程资格；样本外是轻微伤害在容忍内，不是有益/统计无害 |
| FP04 | 同一Fs的K4/rank4、独立专家资格、持久化和组装 | E0–E6、registry/bundle、source/probe/reload与不可变profile | B-J2四专家PASS；B-J3_retry独立组装/verify A1–A5 | DONE于训练方向和一个probe的精确机制；没有OOS bank value |

源码依据：S0 bridge:145–415、s0_gate:118–128及正式runtime；fs_protocol:60–218；test_plan_gate_chain:1–137；test_profile_immutability:1–178；bank_training:2730–3005；r2_fs_bank_train:1670–1780。[R08–R13、R34–R35]

### GPU作业层

| 作业 | 判决 | rc / elapsed的证据类型 |
|---|---|---|
| pt-g3344e9z | S0 committed PASS | 原始计算gate有，独立job_result未取得；不从gate起止倒造job耗时 |
| pt-fuiqqeau | v2 formal质量通过 | task_result称rc0 /306.8s；原平台回执MISSING |
| pt-gr3rrmdg | 独立Fs选择/冻结认证 | task_result称rc0 /263.7s；原平台回执MISSING |
| pt-ycpvyb6n | bank短步、profile/horizon | task_result称rc0 /561.7s |
| pt-vt003lhm | 四专家formal资格 | task_result称rc0 /364.4s |
| pt-m52qwunf | 组装授权路径fail-closed | task_result称rc1 /180.2s；旧失败保留 |
| pt-831c01g4 | 组装retry BANK_CERTIFIED | task_result称rc0 /357.3s |
| pt-jbhf3z73 / pt-t3bfl3e9 | X1探索N32/N64 | 已读内嵌job_result：rc0 /624.669s与719.624s |

这些层次见JOB_EVIDENCE_REGISTER；不得把submission_record单独当完成。

## 4. S0：真正修了什么

当前Normalizer保留raw input mean/std，inverse在源dtype上先构造再cast；正式gate改用NPZ构造路径，避免只修diagnose工厂而正式gate继续用截断NPY；TF32明确关闭。当前raw gate仍使用1e-5，全部注册1/4/12没有减少，差值均0；有独立official reference配置与checkpoint/norm/input绑定。因此**数值验收可关闭，不是靠放宽容差**。[R06、R08–R10]

四臂报告支持当前路径一致，但并不自动证明历史每个误差只由一个因素导致。不要从当前全零误差推出任意后端下都精确，也不要从一次forward的可重复性推出所有训练反向普遍确定。原始输出大tensor未在本次重算。

## 5. Fs：v1失败、探索、v2的正确解释

### 5.1 根因不能过度归因

认证官方后端、同一源代码和有效TF32设置下，已运行的四复制损失/梯度/权重记录一致；它纠正了历史不受认证执行路径的可重复性问题。但v1在此后端上H256仍未通过：lr .0025的holdout24 ratio1.038925，fallback .00125仍1.020438，超过1.02，原协议选择不了Fs。记录本身明确“哪个旧kernel backward不确定”未经定位。[R17]

因此当前成功由至少三项共同构成：受认证可重复执行、会真正拒绝退化的质量门、依据旧曝光数据选择并重新登记的H32训练。不能简化成“只有GPU随机性导致发散，现在TF32修好所有问题都解决”。

### 5.2 不是把2%门槛事后放宽才过

v1的Q6本来就<=1.02；v2仍保持该门。变更的是H256→H32及新16点资格面板，训练仍是8点。v1失败保留，X1全部标EXPLORATORY，未选/冻结X1权重；这些是可接受的研究迭代，不可把v1改写成PASS。[R14、R17–R22]

DEV-001修的是有效TF32判定器，不是降低质量门；DEV-002换的是携带相同F0参考面板的文件。对原始decision和修订decision都保留的做法认可，但必须把源码/carrier偏差记入链条。[R15–R16]
X1 N64在epoch8不适用于N32的诊断规则下曾触发STOP，之后经记录的override继续探索。这不是formal预登记成功，所产数据全部算已曝光。[R20]

### 5.3 预登记顺序能证明到哪里

fs_protocol_v2.sha256写02:04:13Z，V2_J3 submission记录02:09:45.019981Z，协议SHA一致；源码会在加载前消费协议/参数/证书。这支持“记录的v2规则先于该v2作业”。它不意味着v2早于所有v1/X1工作，也不能靠sha文件本身证明外部防篡改时间。[R11、R23–R24]

v2的2019面板可称未用于这轮参数更新的资格面板；源文档也承认2019与backbone/项目验证使用的历史关系，因此不称项目最终untouchedconfirm。

### 5.4 实际资格数值

| 面板24h | F0均值 | Fs均值 | ratio / 解释 |
|---|---:|---:|---|
| train8 | 0.03187907820466652 | 0.03105212451750749 | .974059673813342，即改善2.594% |
| qualification16 | .03127357741727903 | .03137967485354686 | 1.0033925583521255，即退化.3393%；14/16更差 |

Fs满足预登记2%点均值容忍，所以有资格作当前参考；**不能称样本外有益，也不能把有限面板PASS称统计上“无害已证明”**。Q5是训练72h门，不能把它说成所有heldout leads安全。[R25–R26]

## 6. Bank：共同参考与A1–A5确有证据，但限定非常重要

四专家现加载相同认证Fs，merged state digest为370bd3bcc7545755bcdb60e482e1ce1ebc201302dc710f7992a4d76c4915bb97，组装复制对应因子；不是各worker再fitFs。每专家32updates、rank4，group sizes15/16/7/6，E3 own-group24 ratios为.9900749/.9908936/.9849827/.9802412，严格<1.00。协议、代码和结果一致，未见事后放宽E3。[R28、R30–R35]

A1/A2在四专家的1/4/12 probe实际数值差0且torch_equal=true；A3 no-edit=Fs，A4hold后继续Fs，A5重载身份。当前bank证书是逐位门，不是设了非零容差。与Fs branch/merge本身容许小非零舍入误差的检查是不同对象，不能混淆。[R31–R35]

DEV-FP04-001最初相对路径使组装fail-closed，随后相同文件/质量结果改成绝对路径重新作决定，原失败保留，耗用一次infra retry；未见改门槛或拼接不同job的赢家。[R36–R37]

但这些probe只有一个已曝光issue，不能推广为所有天气初态的经验等价；E3是own训练组，不是留出集。generic assembly evaluator本身还缺精确step-key/长度的强制检查，当前数据完整因此不推翻本次PASS；新增消费者应拒绝缺1/4/12或缺8个continuation数值的未来证书。[R34]

## 7. FP05六项判断独立评审

### (1) DEV2019H2 / confirm2020H2 —— 必须修改

2019H2可以是经完整曝光净化后的回顾性DEV池；“不需逐条purge”与7月4日18Z的现有支持曝光矛盾。加24h buffer及12h历史后，6h网格中最早7月6日12Z才可能合法，仍要检查所有其它曝光。

更严重的是X1 N32/N64使用2020H2。其N64 manifest有56个新增索引779..1451，6h轴上为2020-07-13T18Z..12-28T18Z；相邻间隔最大78h，小于84h支持长度。完整支持连续覆盖7月13日06Z..12月31日18Z，加24h buffer覆盖7月12日06Z..次年1月1日18Z。此为根据实际manifest作的区间算术，执行存在又有X1数字结果和内嵌job_result支持。[R18–R21、R41–R42]

**修订：2020H2=EXPOSED_NOT_CONFIRM；confirm本轮UNASSIGNED_NO_ACCESS。** 不为了保留名称只挑少量剩余空隙，也不自动下载新年份。FP05的DEV本身不因此被取消。

此外，最终Fs/bank训练2020H1，评估2019H2是时间倒序加季节转移，只能标retrospective transfer；原plan已承认季节变化，缺少证据不能说必然无效果。失败结论只覆盖当前bank与这类shift，不证明全部动态编辑没价值。

### (2) delta_min=.34% —— 可保留为工程screen，推导必须纠正

历史伤害h0=(LFs−LF0)/LF0=.003392558。即使在同一面板，用Fs作分母的回本线也为h0/(1+h0)=.003381088；新DEV的Fs伤害更是未知。

保留.0034作为待冻结的Fs-relative工程筛查线可以，但必须另报同DEV的G_F0=G_Fs−H_Fs。还需比较OOF_static；仅补回Fs自己造成的损失不证明动态selector必要。MDE、value threshold和72h安全界分别登记，不能因power不够就回改效用线。[R25、R39]

### (3) F0旁路 —— 同意，限定成本与使用

F0不进入候选、不供ridge择优、不进入五候选oracle；仅在预测freeze之后同面板结算背景差。6条相对5条是轨迹数量多20%，不是GPU总耗时必然多20%；模型加载/存储/计分需profile。[R39]

### (4) 一次定N —— 同意，不等于功效已够

先冻结采样和缺失替换算法，完整性扫描不看模型效果，debug只测成本，然后一次选定32/64/128之一及精确名单。固定N之后不因CI或均值追加。约25个时间blocks仍可能不够识别.34%，需报告MDE、假设和有效blocks；宽CI为INCONCLUSIVE。
FP05文本“先扫描，再冻结早于任何数值读取的声明”自相矛盾：应改成规则声明在扫描前、最终内容合格名单在模型outcomes前。替换要有明确区块、tie和去重规则，不可只说“与以前一样”。[R39:29–33]

### (5) ridge主、regime辅 —— 同意，补上嵌套选择

ridge是合理低成本区分实验。必须保留每外折train选定的强static并包含no-edit；regime不能按外折结果改分组。lambda及scaler/PCA/kmeans用inner folds，不能先在整DEV拟合再cross-fit。无须本轮引入MLP、双头或响应模型。

### (6) 双向blocked cross-fit —— 回顾性DEV可接受，不是历史部署

每issue恰好一次OOF预测是优点。outer/inner train与validation净化完整support，候选同issue同折，所有预处理和超参数访问遵循fold capability。毒化测试要检查对应外折模型，不能要求被合法用作其它折train的标签变化后所有模型都不变。
CI应注明条件于本次OOF流程；finalconfirm与普遍可部署收益仍需后续独立证据。[R39]

## 8. 最小后续行动和不能宣称的内容

先FP05a补回执、权重/证书重hash及全曝光账本，再FP05b补新入口和统计协议；不重训已合格Fs/bank。通过后C-J1 debug/profile、一次C-J2完整缓存、CPU OOF fit/freeze、独立score、FP06 DEV判决。初始只解锁FP05a，GPU权限为false。

仍不能宣称：FP02全部完成；旧后端是所有Fs退化的唯一根因；Fs样本外有益或已统计证明无害；K4认证证明样本外天气效用；2020H2是未触碰confirm；双向CV等于历史部署；oracle headroom等于合法selector增益；ridge胜出证明response factorization或novelty。

**下一笔资源应买完整split/消费链和真实OOF开发证据，而不是重复认证或增加模块。Bank OOS未知阻止FP06正面结论，但不构成禁止FP05取证的循环前提。**

---
来源索引：`evidence/SOURCE_INDEX.json`；完整逐finding标签：`evidence/FINDINGS.json`；四层结构化结果：`evidence/PLAN_AUDIT_RESULT.json`。本报告没有把未来任务或准备好的命令当作已执行结果。
