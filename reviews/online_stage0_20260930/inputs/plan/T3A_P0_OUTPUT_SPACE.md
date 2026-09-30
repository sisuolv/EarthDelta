# T3A：先输出空间，拟合与分析分开授权

目标：建立一次共享F0库，并与OBC/DABC/OCL-fresh比较；区分静态偏差、慢漂移和新鲜反馈。OCL未检出信号不能在逻辑上排除参数方法。

前置：T2验收；协议A段与A3签署；DATA_POLICY的复用/补QC身份通过；20步及梯度最小数值门通过。文件：`scripts/online_p0a_f0bank.py`、`online_p0a_outputspace.py`、`online_qc_prepare.py`、`online_engine_precheck.py`，以及相应 `tests/test_online_p0a.py`；全部尚待实现。

## F0库

名义361个起报，实际按DATA_ROSTER槽位记“发布/无输入”。缺真值不禁止F0发布，不以未来QC删除输入合法的起报。六变量保存步骤1,2,3,4,12,20；另为校准/诊断子集保存第4步全69通道预测，以支持统一效应比。无梯度20步库每issue只计算一次；梯度worker必须重新运行可微4步，不能从保存场反传。

拟合期A3只跑01-01..03-26，包含隔离带，共86名义槽位。允许的真值支持最晚03-31；先测加载/I/O、20步、4步梯度成本。拟合阶段不读取/运行分析期模型目标，仅指定17项数据QC可涉及其他2020时刻且只给scorer。

同库由T3A/T3B共用，manifest绑定每个槽位、dtype、lead/channel/coord、模型与归一化哈希；不可回写。不得复用旧probe的面积分母。

## 输出臂与选择

- F0：始终存在。
- OBC：estimate平均偏差，选择λ={.5,1}。
- DABC：已兑现同时效残差按日历EWMA，τ={3,7,15,30}、λ={.5,1}。
- OCL-fresh：六变量×前四步完整反馈包的EOF系数 + 本目标时效已兑现历史的τ7 EWMA；Ridge选择K={10,20}、λ={.001,.01,.1,1,10}。任何前提特征缺失，训练行剔除、分析预测回退F0；不用零填补或取最近有效日替代昨天。

01-01..02-24拟合变换/模型；03-01..03-26只选超参；选择固定后在估计+选择目标行重拟合一次，不含隔离带目标。全部候选按同一个6变量×四时效归一化MSE汇总选择，损失并列按机器合同网格顺序；不能按分析期更换配置。

预热只积累F0合法历史并在04-01重建已冻结模型的状态，不回写03-27的预测。最终模型及依赖版本必须在04-01可用。

## 分析前后

完成拟合后，先做T3B-fit以得到参数比较的MDE输入，再做统计校准/预算表，等待协议B段和A4。之后才运行预热与分析F0库/输出回放。即使输出结果先出来，也不能修改已锁定的参数臂。

输出：`f0bank/manifest.json`、不可变预报缓存、`cells.npz`（物理MSE、共同资格、逐issue日期）、`selection_scores.json`、`config_resolved.json`、`feedback_lineage.jsonl`、`timing.json`。工作进度、数据缺失与方法失败分开；不写虚假的EXPERIMENT_EXECUTED收据。

## 验收与命令契约

合成测试必须覆盖完整事件次序、fresh信息来源、120h反馈至少5日延迟、未来缺失不可见、gap仅排目标、03-31产物不倒灌预热、run写一次；真实结果要能从逐格MSE复算。

以下为拟实现CLI，不在本轮执行：

```bash
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -B scripts/online_qc_prepare.py --contract <signed_contract> --run <new_run>
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -B scripts/online_engine_precheck.py --contract <signed_contract> --run <new_run>
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -B scripts/online_p0a_f0bank.py --config <resolved_config> --phase fit --device cuda
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -B scripts/online_p0a_outputspace.py --config <resolved_config> --phase fit
# 仅协议B段/A4满足后
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -B scripts/online_p0a_f0bank.py --config <resolved_config> --phase warmup,analysis --device cuda
PYTHONPATH=.pydeps:. PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -B scripts/online_p0a_outputspace.py --config <resolved_config> --phase analysis
```

预算是阶段0总A3/A4共享上限，不给T3A另开8卡时。计时/数据/实现失败为技术BLOCKED，输出空间效应未检出为科学证据不足，不能混用。
