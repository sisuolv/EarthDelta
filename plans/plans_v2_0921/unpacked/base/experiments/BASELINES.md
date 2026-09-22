# BASELINES — 当前一个迭代的最小集合

不复现十几种论文，不新增backbone。下列baseline直接在现有EarthDelta接口/缓存上实现。先检查仓库中是否已有等价实现，有则最小复用。

| Baseline | 实施路径 | 输入／监督 | 决定什么 | 本轮时机 |
|---|---|---|---|---|
| F_ref / no-edit | 原bridge、零编辑plan | 合法当前输入；Fs若训练须仅fit | 共同测量起点；F0与Fs分开 | P0-01/02 |
| Best-static | 对dev_select候选平均实际效用选一个并冻结 | 只用过去开发标签；confirm不argmax | 是否需要细粒度动态选择 | P0-03 |
| Regime router | `pilot_policies.py`：fit上冻结简单regime划分，dev_select内每regime选动作 | 共同历史context；聚类/阈值不看评估赢家 | 廉价状态策略是否已解释收益 | P0-03/04 |
| Hindsight oracle | 每issue完整真实候选表中最佳可行项 | 未来truth允许，仅离线 | 当前注册域的事后上限 | P0-03 |
| Direct gain | 共享context与动作描述符的小MLP | 实际gain监督；同样允许仿真辅助预训练 | 二头分解是否必要 | P0-04 |
| Paired e/u | 复用ComposedPredictionHead，默认calibration/JEPA/memory/variance off | e需truth，u来自模型仿真；noedit结构零 | 候选主方法的最小实现 | P0-04 |
| Direct edited-forecast | 复用response-head架构预测同D下edited端点 | 端点仿真目标；预测端点a与0作差，再用同合法参考误差信息形成分数 | 是否只是响应目标更容易而非分解必需 | P0-04 |
| Virtual edit | 已有reference summary加预测u | 无真实候选轨迹；全场需有效decoder | 是否值得再运行真实参数编辑 | 条件性P0-05 |
| Joint output correction | 一个轻量共享结构输出多lead全变量残差 | 同历史/预览、同fit标签和多步目标 | 后处理是否更便宜有效 | 条件性P0-05 |
| Feedback output correction | 同轻量纠错器按步作用，自身纠错状态回输入 | 仅签发历史与自身预测；无未来再分析 | 参数编辑不独占动态传播 | 条件性P0-05 |

## 比较边界

- 固定同一个F_ref和bank。oracle、静态和直接模型使用同可行候选域；任何额外候选、preview、teacher或真u访问单列成本。
- Best-static是dev选定策略，不是理论最优所有静态模型；同时保留合理静态适配Fs的真实训练日志和资源，避免只赢原始F0。
- 给direct-gain相同仿真资格不代表强迫它复制双头；允许同等额外仿真辅助任务，方法/超参清单及总预算事先固定。
- 一份核验truth支持同issue全部K个动作的gain标签，不计K倍独立监督；同一过程的lead不算独立样本。
- Direct-edited-forecast的模型输出不是现实truth；在对齐的输出空间中比较响应目标与端点目标，不宣称它直接预知未来现实。
- 校准后的gain如后续研究使用必须单列；当前不得启用以遮掩解析组合失效。
- 当前不要求重写端到端大router或复现外部RL方案。小型regime/direct-gain筛查不足以证明所有router都失败，结论受本次模型/资源范围限制。
- Sparse ExpertLoRA已经有真实跳过路径可复用，但省下的算子成本仍需实测；不能把小gate/低rank当成端到端加速。

## 可直接复用的工程基础

Stormer pinned接口/归一化；现有ExpertLoRA、paired/probe/selection、SciPy有界求解；已有xarray/parquet读写与评价工具若环境中可用则复用。WeatherBench-X可在已安装且现有导出兼容时使用，本轮不把新搭评估框架作为准入条件。标准主MSE/RMSE使用已验证共同reducer即可。

上一轮参考仓库的许可证信息有未确认项；不把WeatherPEFT/CoMoL/GEPS等“可读源码”当成可以直接复制。当前包没有复制外部源码，也不要求安装它们。若必须引入依赖，先核对原许可、版本与本地环境，再申请，不自动pip。
