# 来源与复用边界

核查日期：2026-09-17。论文日期按arXiv提交/修订记录；方法结论均归作者，不能据此推定EarthDelta收益。本包是原创研究代码，未复制第三方源文件。公开可读不等于已核查所有许可证或已复现结果。

## 本轮直接核查的论文

[S1] CLAW — Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning. 2026-09-10.
https://arxiv.org/abs/2609.12278
已读正文。近期交互 -> hypernetwork -> 低秩世界模型适配；联合预训练基础模型。没有确认可直接复用的官方实现，不作为安装依赖。

[S2] PEFSO — Preemptive Ensemble Forecast Sensitivity to Observations. 2026-09-10.
https://arxiv.org/abs/2609.12296
已读正文。研究额外观测的预报影响及无需重积分的近似；实验为Lorenz-96，受切线性近似范围限制。不是EarthDelta的内部参数编辑实现。

[S3] PEAR — Decision-Focused Learning via Tangent-Space Projection of Prediction Error. v2 2026-05-19.
https://arxiv.org/abs/2605.01361
正文附作者代码链接 https://github.com/FinJun/PEAR 。本轮未逐文件验证该仓库，不把它称作已接入。其投影误差/决策导向思想是近邻；定理假设不能直接套到天气闭环。

[S4] CoMoL — Efficient Mixture of LoRA Experts via Dynamic Core Space Merging. v2 2026-06-04.
https://arxiv.org/abs/2603.00573
https://github.com/DCDmllm/CoMoL
本轮实际读取README与src/mocorelora/layer.py。shared A/B与小r×r core是可复用工程思想，不是本项目新贡献。

[S5] DISeL — Learning When to Adapt. 2026-05-18.
https://arxiv.org/abs/2605.19028
https://github.com/alizindari/DISeL
论文及作者代码关系核查；门控源码在此前本对话读取。不得把本包原创分组实现称作官方DISeL等价复现。

[S6] Solver-in-the-Loop. NeurIPS2020.
https://arxiv.org/abs/2007.00016
https://github.com/tum-pbs/Solver-in-the-Loop
官方README此前读取，含历史TF/PhiFlow依赖。借鉴修正后状态分布与多步训练协议，不整体并入PyTorch环境。

[S7] GEPS. NeurIPS2024.
https://geps-project.github.io/
https://github.com/itsakk/geps
此前已读取geps/model/layers.py、geps/datasets/kolmo.py。可作受控PDE与上下文适配近邻；生成器默认高分辨率及显式CUDA，缩小配置需独立检查数值收敛。代码复制前另查许可证。

[S8] Semigroup-JEPA. 2026-09-09.
https://arxiv.org/abs/2609.10464
https://github.com/sg-jepa/sg-jepa
论文/仓库入口检索，不纳入当前首版依赖；不以新增JEPA模块替代响应机制验证。

## 实际读取的代码与官方工具

[C1] Stormer，已重新核对main提交：58dfee5a6037399a40fefd492bc00421e0c885a8。
https://github.com/tung-nd/stormer
https://github.com/tung-nd/stormer/blob/58dfee5a6037399a40fefd492bc00421e0c885a8/stormer/models/hub/stormer.py
https://github.com/tung-nd/stormer/blob/58dfee5a6037399a40fefd492bc00421e0c885a8/stormer/models/iterative_module.py
https://github.com/tung-nd/stormer/blob/58dfee5a6037399a40fefd492bc00421e0c885a8/inference.py
源文件此前对话读过，本轮复核分支身份。未下载权重、未运行真实天气模型。

[C2] WeatherBench-X，tree/main ref：19548af79e75190254bba273ff5e3b179265e0bd。
https://github.com/google-research/weatherbenchX
https://github.com/google-research/weatherbenchX/blob/19548af79e75190254bba273ff5e3b179265e0bd/evaluation_scripts/run_example_evaluation.py
本轮读取示例前190行，核查PredictionsFromXarray、TargetsFromXarray、GridAreaWeighting、RMSE/MSE、Aggregator与DirectRunner用法；尚未安装执行。

[C3] CoMoL source blob（不是commit）：9aa7a4aa9d0a0e796115e722b2a552638741b931。
https://github.com/DCDmllm/CoMoL/blob/main/src/mocorelora/layer.py
本轮实际读取。迁移需适配PEFT内部接口和张量约定；不得把LLM脚本直接当Stormer训练器。

[C4] PyTorch JVP官方文档。
https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html
前向AD可能遇到未覆盖算子；首版采用中心差分。容器实际测试torch2.10.0+cpu，不表示2.14已测试。

[C5] SciPy lsq_linear。
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
实际离线教师复用该有界线性最小二乘求解器。在线低维盒约束QP使用scipy.optimize.minimize的L-BFGS-B和显式梯度，不对求解过程反向传播。

[C6] WeatherBench2 data guide / evaluation migration。
https://weatherbench2.readthedocs.io/en/latest/data-guide.html
https://github.com/google-research/weatherbench2
1.5°、1.40625°网格与数据年份必须检查实际坐标；本计划不重新提供未经检查的数据下载URL。

## 继承材料

- EarthDelta_Codex_Plan_CN.md：旧ED00—ED06及归一化/变量合同。
- EarthDelta_Starter_20260915.zip：原始StateGatedLinear与TemporalPatchState，21项CPU测试本轮重跑。
- EarthDelta_Response_Space_Novelty_Plan_20260917_CN.md：响应空间提案，本轮细化到收益交互、在线接口与代码。
