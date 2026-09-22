# EarthDelta V2（研究合同，不是新版本已实现）

## 1 Core problem

同一冻结参考模型下，在运行多个昂贵候选预报前，能否预测有限编辑的完整未来响应，并选出值得执行的修正？

## 2 Core hypotheses

引用 FINAL_RESEARCH_DECISION 的H1/H2/H3。不重新扩展更多“创新”。

## 3 Formulation

输入I包含声明历史、可选一次reference preview及其成本；不包含未来truth、exact edited outputs、oracle b/H。动作a包含bank identity、系数、window、target layers、continuation。候选全集包含zero。

定义e0=D(Y)-D(Yref)，du(a)=D(Ya)-D(Yref)。主线S_phi(I,a)=(ehat,dhat_a)，预测g=2<ehat,dhat>-||dhat||²，同MetricSpec。首轮finite选择argmax(g-λC)，无改动候选允许停止。动作连续性/Jacobian只做诊断。

## 4 Training objective

同共同特征编码器，对e0和du分别使用Q加权回归；相同数据上增加可选gain回归/排序损失。raw geometric与加性gain校准是两个单独variant。训练/标签预算记录：一份e0标签不因K个candidate重复计数，du仿真轨迹总数也单列。du可以在没有未来标签的额外训练起报上采集，但必须独立时间范围、无test适配；其他方法应有same-label与same-total-data/compute两个比较轨。

当前ComposedPredictionHead已具备leads接口、reference/response解码；保留并补MetricSpec、zero-action结构约束与配置开关。不要默认启用JEPA或variance penalty后又称变化只来自e/du分解。

## 5 Inference procedure

(1) 校验checkpoint/bank/normalization/data identities；(2) 只读ServingContext；(3)一次编码context；(4)对全部声明candidate descriptor预测du和分数；(5)共同feasibility过滤，zero优先tie；(6)只运行选中动作的真实预报；(7)保存无标签决策日志。无oracle teacher/no per-candidate weather rollout。若context来自reference第0步后段，则先执行并计费该preview，再执行或合法重放选中预测，不能称零开销。

## 6 Main novelty candidate

可学习的逐状态、逐参数干预响应，及e/du监督非对称的可测价值。不是generic counterfactual world model的首次。

## 7 Why parameter intervention

没有一般必要性定理。用相同信息的多lead后处理、迭代feedback残差、virtual edit与同控制器输出动作去挑战。只在有限资源下证实更稳/更省/泛化更好才能保留。

## 8 Strongest baseline

同历史、同bank、同多步loss的direct router；同描述符/预算的direct-gain；同信息、同更新日程的feedback output correction。三个均必需，不相互替代。

## 9 Killer experiment

先做真实完整候选的oracle对静态余量。通过后同一候选表四格替换e/du，加directgain和反馈残差：一张共同样本、共同预算的表同时暴露字典、响应、误差、选择和执行的瓶颈。

## 10 Kill criterion

按STOP_CONDITIONS的预登记δ与置信区间决策。不能只统计“选中成功”的样本；保留所有起报、noedit、失败和最差lead。有限动作上限只是当前字典上限，不能据它否定所有参数编辑。
