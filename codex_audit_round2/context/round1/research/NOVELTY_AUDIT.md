# Novelty判定

结论：相比最初physics-state-conditioned LoRA，科学问题更明确、可证伪性更强；实证novelty尚未建立。当前代码只是把“预测再编辑”的候选机制落实为若干可测试原语，没有给出真实天气优势。

## 可以争取，但尚未证明

- 逐大气状态、逐候选参数编辑的多时效响应预测是否有稳定、可泛化的规则。
- du的label-free仿真监督与e0的核验标签分离，是否比同样拥有这些数据的directgain/直接edited-forecast模型更省真实标签。
- 对持出的幅度、窗口或组合，响应结构是否提高同预算选择质量。

## 不成立的默认推理

“du是确定性函数→一定容易学”不成立；输入摘要可能丢失决定响应的信息。
“e0拟合得好→参数编辑必要”不成立；它同时增强了直接残差基线。
“正交参数→互补天气作用”不成立；应测实际响应。
“有限d的R低秩→天气修正低维”不成立；这是矩阵列数上界。
“更晚年份→物理OOD”不成立；最多先称时间外推。
“descriptor支持任意数目candidate→zero-shot新专家”不成立；新专家权重内容尚未编码。

## 五个增强方向的取舍

1. Intervention world model：可以作为解释性比喻；当前本质是model-response surrogate。若没有可组合transition、闭环状态更新与未见动作验证，不把它升级成完整世界模型。
2. Learned intervention Jacobian：参数敏感度/学习Jacobian有先例；代码JVP尚占位，且大幅编辑需要非线性响应。DEFER，不作为救novelty的改名。
3. Trajectory planning：KEEP（唯一主增强）。利用已有多lead头，但补actual rollout监督、共同目标和持出动作测试。
4. Uncertainty-aware：P2可选，校准的是预测收益/误修正风险，不能凭均值减方差宣称保证安全。
5. Physics-structured utility：先diagnostic，固定线性coefficient/vorticity算子可作后续；功率/动能非线性须端点分别变换。谱损失不进入P0。

## 模块裁决

| 模块 | 所针对瓶颈 | 本轮决定 |
|---|---|---|
| 短历史state encoder | context是否含du/e0信息 | KEEP现有，所有baseline同权限 |
| JEPA | 是否有可测响应表征样本效率瓶颈 | DEFER，无此证据 |
| memory / regime retrieval | 是否反复出现可复用误差模式 | DEFER；先静态均值/EWMA同信息对照 |
| dynamic rank / expert count | 实测质量成本曲线有无非平凡需求 | DEFER；有限候选与真实成本已够 |
| spectral state/loss | 主空间是否漏掉影响决策的结构 | DIAGNOSTIC_ONLY |
| hierarchy / layer routing | 候选空间是否明显不足 | DEFER，不能以范围增大替代P0 |
| global/context pooling | 避免过强压缩和混淆季节地域 | KEEP并加入context-only基线 |
| multi-edit interaction | 是否存在真实非加性或Gram响应重叠 | P1配对检验；不是默认第三创新 |

## 审稿防守的最小证据

真实bridge可复核；非零bank在独立dev有上限；动态相对静态有余量；du优于no-effect和bankmean；e0的决策投影可预测；双头胜directgain至少一个预登记维度；参数编辑不被feedback残差支配；多时效和真实成本可复算。缺任何关键一项，收窄相应贡献。
