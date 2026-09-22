# 第二轮独立复核摘要（本次仅压缩保存，不重做文献检索）

## 决策

有界继续。科学价值尚未由天气实验兑现；下一份证据应同时针对“可修正空间”和“合法信息能否识别”。不能让工程 bug 代替研究否证，也不能借研究有潜力跳过测量合同。

## novelty 的保留边界

LoRA、适配器库、参数响应矩阵、加权最小二乘、收益恒等式、执行前估值与预算选择均有先例；删去领域和公式包装后，当前管线是已知结构的组合。仅未检出所有限定词同时出现的论文，不能证明非平凡贡献。

上轮核验的纠正：WeatherPEFT 不只是 benchmark，包含 TADP/SFAS；Adapter Banks 的相关迁移使用已知轨迹上的优化，不是 brief 所称 RL-return-based selection；VI-MoLE 的 scalar risk 能表达 prefix 的非加性效应，并非天然弱于非对角 H。保留的贡献候选是公平资源下的核验标签效率，以及受限动作/目标复用，均待证。

## 数学与估计目标

同一端点空间：`g = ||e||_Q² − ||e−u||_Q² = 2eᵀQu−uᵀQu`。变换后各端点相减不要求变换线性；若写 `D(raw difference)` 或组合响应，则需对应线性条件。

给定固定候选且 Q 固定/由 I 确定、二阶矩存在：
`E[g|I] = 2μeᵀQμu − μuᵀQμu + 2 tr(Q Cov(u,e|I)) − tr(Q Var(u|I))`。
完整 I 确定响应时后二项消失；压缩 context 下分别回归两个均值未必足够。联合 MSE 训练不自动消除这个问题。Decision-focused 或直接 b/H 可作为不同机制，但本轮不另加复杂方法。

在线性响应空间中决策只需 `b=RᵀQe`，不是必须还原全 e0。低全场 R² 不是独立 kill criterion；应看实际收益、排序和 regret。H 是响应重叠，非线性组合残差需真实组合计算。

`E[max_a g] >= E[max_a E[g|I]] >= max_a E[g]`。I 与 e 独立、e=±1、u∈{0,±0.5} 时 oracle=0.75，最佳合法策略=0。P0-05 PASS 只授权可预测性试验；原 P0-06/07 已有该挑战，不需推翻完整 DAG。

## 公平性与参数编辑必要性

一个核验轨迹可生成所有候选 gain 标签。所有对手应有同仿真资格，bank 消耗的核验标签必须计入总账。另报给定现成 bank 的控制器标签效率，不能混称全系统标签效率。

无限容量 feedback corrector 可表示 edited−reference 的单步差并传播，但这种表达能力包含不等于同预算支配。需要 actual edit、virtual forecast、联合输出和 feedback 的有限资源比较；参数编辑不自动保证物理一致性。

## 审计的执行优先级

B01/B02/B10、实际 Q 和样本准入首先解决。B03 不阻塞固定系数 bank；B07 在无 memory 的 retrospective pilot 中以声明/入口限制处置；B08/B09 用实际小子集证书阻断，不先重写全数据系统。并发/replay、空 registry 通用约定、格式化问题不与错误参考同等阻塞。A03 四项核心修复有效；历史 YAML 可保留但执行入口须服从新合同。

## 来源与范围

本摘要来自上轮对话中的独立复核，不是原 ROUND2_REPORT 的自动采纳。工程位置固定在 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`，详见根目录 `IMPLEMENTATION_AUDIT_ACTIONS.md`。

以下链接是上轮核验来源的沿用索引，本次未重新浏览论文，也没有新增“最新论文”结论：
- Aurora v3，§12.4：https://arxiv.org/html/2405.13063v3
- WeatherPEFT v2，§3–4：https://arxiv.org/html/2509.22020v2
- Adapter Banks v1，§3.1.2、§3.2、§4.4：https://arxiv.org/html/2609.17042v1
- VI-MoLE v1，§3–4：https://arxiv.org/html/2608.02528v1
- Strobach 等，GMD 2022，§3：https://gmd.copernicus.org/articles/15/2309/2022/

本次只重新核对分支/源码对象身份及既有 CPU 日志，没有运行该仓库实验。
