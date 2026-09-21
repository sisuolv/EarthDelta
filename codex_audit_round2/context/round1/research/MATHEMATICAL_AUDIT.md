# 数学合同与方法边界

## 1. 统一参考与符号

F0 是原始冻结 checkpoint。Fs 是只在训练区间拟合并随后冻结的静态适配参考。S0 验证 F0；科学干预实验的 zero-edit 必须复现所声明的 Fs，不得在比较时偷换参考。没有 Fs 时可登记 reference=F0，但要单列轨道。

令 y0 = D(Rollout(Fref,x,0))，ya = D(Rollout(Fref,x,a))，y = D(Y)。定义：

    e0 = y - y0
    du(a) = ya - y0
    e_a = e0 - du(a)
    gain_Q(a) = ||e0||_Q² - ||e_a||_Q²
              = 2 e0ᵀ Q du(a) - du(a)ᵀ Q du(a).

这是平方损失的代数恒等式，不是新理论。`paired.py`、`geometry.py` 的主计算采用 truth-reference；附件中出现 reference-truth 的表述不能与正号公式混用。若坚持 error=prediction-truth，则式子第一项应改成 -2<error,du>。

## 2. 三类响应不能混称

- `cached_responses`：对有限 candidate 实际运行得到的端点差，是当前神经模型及 D 下的有限响应；不是实际大气干预。
- `central_response`：在 a0 邻域估计 Jacobian R；Ra 仅局部近似。
- head 输出：learned approximation，任何解析 gain 都是 predicted gain，非真实保证。

在固定 T 上分别计算 e=T(Y)-T(F0)、du=T(Fa)-T(F0)，上面的恒等式对任意固定 T 都成立。线性/仿射性是为了将 D(Y-F0) 与 D(Y)-D(F0) 互换、以及线性叠加/解析R路径；不是平方恒等式本身的必要条件。主协议继续用冻结线性 D，修正文档而不暗改实验。

## 3. MetricSpec

约定 decoded shape e:[B,H,S,F], du:[B,K,H,S,F]。保存变量次序、原始单位、只在允许训练期计算的 scale[F]、lead_hours[H]、spatial support与真实面积、D的内容hash及共同有效mask。

    Q[h,s,f] ∝ lead_weight[h] * cell_area[s] * variable_weight[f] / scale[f]²

若D已标准化变量，则Q不得再次除scale²。约定Q在完整 H×S×F 上总和为1，只归一化一次；报告每变量/每lead损失使用对应条件分母。所有方法共享有效mask，不根据预测好坏删点。MSE先累加再sqrt得到RMSE，禁止先逐起报开方再平均后称 pooled RMSE。ACC使用冻结训练气候态，零方差显式missing。

当前三个路径不一致：paired只归一化最后一维；geometry不归一化；head无权mean并加calibration。必须保留raw/calibrated两列，并默认 analytic-only。部署API显式返回 `gain_analytic`、可选`gain_calibrated`，不得共用一个含糊的`gain`。

## 4. 多编辑

如果 du(a)≈Ra，则 H=RᵀQR、b=RᵀQe0，gain≈2bᵀa-aᵀHa。非对角项来自平方范数，是已有二次代理的交叉项；H为响应Gram，不是完整非线性天气损失Hessian。

对真正的组合动作需另外量化：

    interaction(a,b) = du(a+b)-du(a)-du(b)  (零参考)

它不等于H_ij。finite动作模式直接采集组合端点、无需假设线性。对角H与完整H、线性合成与直接组合响应分别对照。

## 5. e0是否等于重新预报？

若允许信息I中Fref(x)已知，则

    E[e0 | I] = E[Y | I] - Fref(x).

因此在无限表达/足够训练及平方损失下，准确的条件均值误差订正本身就是Bayes最优预测。不能证明参数编辑普遍优于不受约束的输出纠错。实际价值只能来自有限样本、低参数、约束空间、跨lead复用与成本的归纳偏置。

但选择编辑只需要e0在候选响应张成空间中的投影：若e_perp与所有du在Q内积下正交，e_perp不影响gain。没必要要求小头重建所有不可预测噪声。首轮固定可解释响应空间，用oracle e0/predicted du四格实验检验瓶颈。

## 6. 均值代入不是自动得到期望收益

若只给压缩context c，e与du在条件c下仍有随机性，则

    E[g|c,a] = 2 μeᵀQ μd - μdᵀQ μd
               +2 tr(Q Cov(e,du|c,a)) - tr(Q Var(du|c,a)).

若完整初始输入、模型与动作固定，du是确定性模型输出；此时不存在这种物理条件随机性，但学习器误差仍存在。不要把预测头方差、模型误差和真实天气随机性混为一谈。P0不因此加入大型概率模型；先用direct-gain基线和校准诊断检验均值代入的代价。

## 7. 参数干预与输出反馈的可表达性

任意已编辑一步映射可以写作 F_a(s)=Fref(s)+C_a(s)，其中 C_a=F_a-Fref。迭代使用相同C_a即可复现同一轨迹。因此“只有参数编辑会改变未来动力学”“输出纠错只能改变末端”均不成立。

强基线至少包括：多lead联合后处理、逐步反馈残差模型、同控制器同动作预算的低秩输出/activation残差、virtual edit Fref+预测du。后者用于问实际执行参数编辑是否提供了代理无法替代的收益。没有实证优势就pivot到更简单订正，不在叙述上排除它。

## 8. oracle和部署分开

候选全集穷举的 best-of-registered-set 是该有限集合的事后参照。只用top3 nonlinear verification的teacher不是全集上限。oracle e0涉及未来真值；oracle du可由模型仿真获得、不需真值，但代价是多条rollout，不能算低成本部署。

所有论文中的“gain”需标记oracle/predicted/realized与summary/full-field。代理预测为正不能保证实际为正；no-op由真实标签挑出来也不能称serving-time安全。

## 8. gain误差的方向分解（代数诊断，不是新理论）

令预测偏差 εe=ehat−e、εd=dhat−d，则

    ghat−g = 2<εe,d>_Q + 2<e−d,εd>_Q
             + 2<εe,εd>_Q − ||εd||_Q².

所以只报告e0总体R²、du平均cosine不能保证选择正确；误差朝向和被选候选的尾部误差尤其重要。P0应对top候选单独检查误修正，但主分母仍是全部起报，不能只报被选子集。
