# 第二轮独立研究复核摘要

本文件压缩保存上一轮判断，不是新文献综述。代码受审点为 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`；当前可读交付分支头为 `403b55db65f4c35c1a85d0794ad0de2765b07d96`。两者现有根目录条目一致，后者仅新增 `codex_audit_round2/`，详见 ../evidence/REPOSITORY_FREEZE.json。

## 研究价值

决定：有界继续。科学问题已从 state-conditioned LoRA 收敛为候选模型干预的条件轨迹响应预测。其方法组成有大量先例；“没有检索到同时满足全部限定词的论文”不等于非平凡贡献。最多争取监督来源分离的有限资源优势、以及同字典未见幅度／窗口／组合的复用价值。下一轮首先检验前者。

## 文献判断（沿用前次已核查版本，不再次检索）

Aurora `2405.13063v3` 已有多步 LoRA；WeatherPEFT `2509.22020v2` 提出 TADP/SFAS，不是仅 benchmark；Adapter Banks `2609.17042v1` 的目标轨迹 soft-policy 优化不是 RL-return-based selection；VI-MoLE `2608.02528v1` 已覆盖执行前估值和预算分配，标量条件风险也能编码交互。Green’s-function 校准、forecast sensitivity、predict-then-optimize 与反馈求解器修正都是已有谱系。没有要求 Codex 重做文献工作。

## 数学边界

固定 Q 下，e=truth-reference，u=edited-reference，g=2 e^T Q u-u^T Q u 是实际端点平方误差差的恒等式。学习值 plug-in 是 surrogate；u=Ra 是额外条件，不因恒等式而成立。变换后端点作差的恒等式不要求变换线性；交换变换与差分／组合则需要相应条件。本轮仍固定线性 D。

压缩 I 下，E[g|I]=2 mu_e^T Q mu_u-mu_u^T Q mu_u+2 tr(Q Cov(u,e|I))-tr(Q Var(u|I))。完整确定性输入使响应条件方差项消失；联合训练可能缓解表示问题但不自动恢复缺失矩。只预测误差在响应张成空间中的投影可以足够，但预测子空间也须符合合法信息约束。

H=R^T Q R 描述局部响应重叠，不等于 u(a+b)-u(a)-u(b) 的非线性交互。singleton 实验不能否证未测组合。

## Oracle 与合法策略

V_static <= E[max_a E[g|I]] <= E[max_a g]。独立于 I 的 e=±1 与 u∈{0,±0.5} 给出 oracle=0.75、最佳合法策略=0；不是 bootstrap 假阳性。P0-05 原主体保留，PASS 只放行小型可预测性取证。原 P0-06/07 已存在，不能以此反例要求推翻全部 DAG。

## 参数执行的必要性

无限表达能力下 feedback 可以模拟 F_edit-F_ref；但等式不提供廉价模型，不足以否定有限资源参数编辑。反过来，参数编辑也不独占多步传播。virtual 响应若只在压缩空间存在，不能免费视作完整天气预报。

## 审计复核

参考路径、归一化 policy、真实资产绑定和实际数据／全目标评分是当前关键准入项。caller 系数图问题不等于 bank 没梯度；并发／replay 不阻塞串行路径；复数候选与空 registry 在正式入口校验即可；cleanup 非必要异常不自动作废已完成科学测量；A03 四项主要校验有效，A11 保留历史 YAML 合理。

## 证据纪律

前次日志：283 passed、7 skipped、0 failed；本包仅重取日志，没有重跑仓库测试。真实 GPU/xformers S0、训练及天气收益仍无已核实产物。本次实际动作仅是打包工程任务与校验交付物。
