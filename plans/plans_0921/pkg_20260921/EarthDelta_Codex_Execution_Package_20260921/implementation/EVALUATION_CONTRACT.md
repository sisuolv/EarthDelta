# 统一评测合同

## 数据/输出
Serving输入I须包含可核验history timestamps、Fref身份、可选referencepreview及其费用；truth不可见。真实输出张量保留完整[init,lead,var,lat,lon]，summary为[B,H,S,F]。每个candidate都有明确bank、系数、hold、层、continuation。prefix一旦含学生预测，不能复用真值prefix上的response标签。

## MetricSpec
令s_v为训练期固定变量scale，q包含区域cellarea、variableweight、leadweight；Q在全目标轴一次归一化。D可先进行固定线性降采样/投影；scale只应用一次。对无缺失情况：

    L = Σ_h,s,f q_hsf ((Y-F)/s_f)^2 / Σ_h,s,f q_hsf
    gain = L_reference - L_edited.

若已在D中除s，则Q中不再重复除s²。perlead统计使用该lead的声明条件归一化；全局统计不能再无权mean各lead代替。weight/scale/lead列表/投影均哈希入模型和运行manifest。缺测权重与共同有效mask在看到某方法表现前冻结；不按方法丢点造成不同分母。

RMSE：先在共同样本/空间聚合MSE，再sqrt。mean(per-init RMSE)是不同统计量，若保留必须明确命名。ACC用训练期climatology，含变量/lead维度。确定性单样本CRPS退化为MAE，不称概率校准。

## Intervention metrics
- actual_gain：真正执行后的完整Q损失差；summary_gain另列，不混成fullfield。
- oracle_gain：完整已注册且可行有限集合上的逐样本最优，无编辑保证非负。任何候选缺失则complete_oracle=false。
- static_gain：只在fit/dev选择一个固定候选，再应用未见样本。
- dynamic_gap：hindsightoracle−beststatic；不是所有参数空间的上界。
- regret：同真实结果、同可行集下best−selected；缺失/不可行状态不可静默替换。
- ranking：Spearman与top-k best-action recall；平分和所有响应零时N/A并记录。
- harmful_edit_rate：全部已执行nonzero edit中 actual_gain<0 的比例，同时报告占全部注册起报的比例，不能只看conditional。
- no-edit率、失败率、未结算率全部保留；当oracle接近0时不报告“已实现oracle百分比”。

## Response metrics
weighted NMSE/relative error、cosine、normbias、R²，在每个lead/变量/动作族单列。nullresponse作为单独组：du=0时cos未定义，不自动当perfect或0。no-effect、fit平均响应和descriptor-shuffle必测。全空间误差与响应span中e0误差分开，span只用于离线诊断，不假装推理可获得真实R。

## 同一表的4格
OO：truee0+actualdu；OP：truee0+preddu；PO：prede0+actualdu；PP：prede0+preddu。四者选择后都在同一realized table结算。OO是离线参照；PO实际跑了所有candidate，不能列低成本部署。PP与directgain同算力/输入比较。选择器保持不变才可解释瓶颈。

## Physical structure
全变量评分之外可报告风场涡度/散度、位置/强度和合法SHT系数诊断，但需要真实网格与单位。谱功率、动能、logenergy是非线性函数：可分别变换各端点后做平方差，但不能将T(Yedit−Yref)当T(Yedit)−T(Yref)。不以物理图好看证明参数编辑更物理。

## 成本
分别记录初始化、referencepreview、historyencoder、allcandidatehead、planner、选择后的真正rollout、读取/导出、训练/teacher生成成本。对同等batch/dtype/backend暖机后，CUDA synchronize计时，多重复报告median/p95。稀疏slotcount为抽象预算；elapsed/GPUmemory为实际成本，两者不混称。

默认不把所有K个候选的真实天气运行隐藏在“label-free du”里。作为response数据采集可以；作为deployment必须明确付费compare。

## 统计
按起报时间块或独立天气过程配对bootstrap，同一个过程不同lead/网格不是独立N。预先冻结主指标、SESOI、guardmargins和最大确认样本；多重比较标明主/次目标，研究性分层不等于独立确认。所有置信区间同时报告样本组数和有效覆盖。
