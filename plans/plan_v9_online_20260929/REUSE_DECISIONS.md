# EarthDelta v9.1 / T0 复用决策

审计日期：2026-09-29 UTC。范围仅为 T0；没有实现 T1、T2 或后续任务。

代码基准：`d55ad70854bcf535b86c761b8d9cf7a7981374a0`，以实际本地 HEAD 为准。
输入 ZIP：仓库根目录 `plan_v9_1_online_20260929 (2).zip`，SHA256 为 `3310490bfacfae60734c0ae8f892bb06033ba1a4a88ad2add53d26c593c92b8b`。

运行记录：[T0 目录](../../runs/online/t0_reuse_audit_20260929T114614Z/)。下文 `RUN/` 均指该目录。

## 结论与完成边界

**建议复用现有模型、LoRA、逐 issue 指标、WBX 和 sklearn；新增代码集中在时间与访问边界、可微损失、统计口径和在线状态管理。** `arch` 的采样器值得采用，但其多重比较类不能直接代替 Gate 所需的同时置信区间。

清单中的 46 个参考目录全部存在，HEAD 与清单的短 SHA 前缀全部一致：缺失 0、版本不一致 0、不可核验 0；缺顶层 license 文件 6。逐项证据在 [reference inventory](../../runs/online/t0_reference_inventory_20260929T114614Z.json)。这里只核验清单版本与 license 是否存在，不把所有仓库视为已经过功能审查。

指定计划目录和根目录 `CLAUDE.md` 在开始时均不存在。已从 ZIP 完整读取对应文件，副本只放在 `RUN/inputs/`；本轮仅在目标计划目录新增本文件和 `t0_reuse.patch`，没有擅自部署其他计划文件或修改授权表。

授权事实：用户本轮明确授权 A1 的 T0 盘点和合成 CPU 核验；包内 T0 也写明盘点用 A1。通用授权表的 A1 行未列 T0，签署栏为空，这一措辞差异记录在 `RUN_CONFIG.json`，不能扩展解释为 T1/T2 开工。A2a、A2b 均为“待授权”、签署人为空；本轮没有安装任何依赖、没有克隆 batch 5，也没有改 reference 下的三个候选文件。

**T0 的已授权审计已形成决策，但全部验收仍未齐备：arch 的实际接口运行，以及 TAFAS/OnlineTSF 的本地固定版本审查等待签署。** 缺证据项明确保留为未核验，不视为通过。

## 环境和依赖决定

实际 Python 为 `/usr/bin/python` 3.10.12；torch 为 `2.3.0a0+6ddf5cf85e.nv24.04`。本轮禁用 CUDA，全部张量测试在 CPU 上完成。

系统默认 NumPy 为 1.24.4，直接运行 WBX 会被 `_vendor.py` 的 NumPy 最低版本检查拒绝。使用现有 `PYTHONPATH=.pydeps:.` 后，实际导入 NumPy 1.26.4、SciPy 1.12.0、pandas 2.3.3；WBX 按自己的 `ensure_vendored()` 验证官方源码，并把 `.pydeps_wbx` **追加**到路径，导入 CPU JAX 0.6.2。没有修改 vendor 加载规则。

`.pydeps` 同时残留 `numpy-1.26.4.dist-info` 和 `numpy-2.2.6.dist-info`，但运行时模块是 1.26.4。不能只凭 `pip freeze`/metadata 推断实际版本，也不能把 2.2.6 当作已经验证的工作环境。本轮保留两份元数据及实际导入路径，不清理、不升级。

以下是**签署用的具体版本清单**。保留已验证的版本，不要求无理由重装。“兼容”指 Python 元数据与当前依赖约束相容；只有实际跑过的部分才标运行核验。

| 包 / 建议版本 | Python 要求、license | 用途 | 当前结果与下一步 |
|---|---|---|---|
| `arch==7.2.0` | >=3.9；NCSA | Stationary/Circular bootstrap，StepM/SPA/MCS 辅助检验 | 当前缺失；5 个接口例子跳过。A2a 签署后安装并原样补跑，不能记为已验证 |
| `scikit-learn==1.2.0` | >=3.8；BSD-3-Clause | Ridge、RidgeCV、TimeSeriesSplit | 当前已安装；三个接口的合成拟合/分割均通过 |
| `zarr==2.18.3` | >=3.10；MIT | 未来 store I/O | `.pydeps` 已有且可导入；保留 Zarr v2，未读取真实 store |
| `xarray==2025.6.1` | >=3.10；Apache-2.0 | WBX 数据结构和坐标 | 已导入并参与随机场评分；需要 pandas>=2.1，使用已有 2.3.3 |
| `fsspec==2026.9.0` | >=3.10；BSD-3-Clause | 未来存储映射 | `.pydeps` 已有且可导入；本轮未核验远端缺块行为；系统另有 2024.2.0，勿混淆 |
| `matplotlib==3.8.4` | >=3.9；Matplotlib/PSF-based license（包元数据标 PSF） | 后续报告绘图 | 已安装、可导入；本轮没有科学图表或绘图正确性结论 |
| `jax==0.6.2`、`jaxlib==0.6.2` | >=3.10；Apache-2.0 | WBX 官方指标依赖，CPU only | 已在 `.pydeps_wbx`，通过 vendor 和随机场核验；无需重复安装 |
| `ml_dtypes==0.5.1` | >=3.9；Apache-2.0 | JAX 依赖 | 已在 `.pydeps_wbx`；该目录不得包含与 `.pydeps` 重叠的顶层包 |

`arch` 的必要传递依赖也应列入 A2a 审阅：建议 `statsmodels==0.14.4`（BSD-3-Clause、Python>=3.9）和 `patsy==1.0.1`（BSD-2-Clause、Python>=3.6），当前缺失，未安装。前者要求 NumPy>=1.22.3,<3、SciPy>=1.8 且不为 1.9.2、pandas>=1.4 且不为 2.1.0；与已实际导入的 1.26.4/1.12.0/2.3.3 元数据相容。NumPy、SciPy、pandas 和 numcodecs 0.13.1 保持当前工作版本；不要让解析器将 NumPy 自动换成 2.x。

来源：`RUN/ENVIRONMENT.json`、`MODULES_DEFAULT.json`、`MODULES_PYDEPS.json`、`DEPENDENCY_LOCAL_PROVENANCE.json`（含本地许可证/metadata 的路径和哈希）、`PYPI_RELEASE_METADATA.json`（11 个版本的官方 PyPI 元数据、Python 要求、依赖和 CPython 3.10/通用 wheel 列表）。例如 [arch 7.2.0 官方发行页](https://pypi.org/project/arch/7.2.0/)。所有 wheel 只核对清单，没有下载或安装。安装前后 freeze 对照结果见运行总结。

签署后若补装 JAX，仍须按 `earthdelta/wbx/_vendor.py:291`、`:329` 的隔离方式处理：只将 CPU jax/jaxlib/ml_dtypes 放入 `.pydeps_wbx`，追加路径，不带 CUDA extras，不覆盖 `.pydeps`。当前工作目录中的 JAX 已可用，因此默认不重装。

## 逐项复用决定

内部代码的项目 metadata 在 `pyproject.toml:11` 声明 MIT；仓库根目录未找到单独 LICENSE 文本。本轮内部 import 可执行，不把这条声明扩大为第三方代码可重新分发的许可。第三方 license 均核对本地文件；本轮 vendor 拷贝数量为 0。

| 组件 | 来源（路径::符号 或 pip） | 复用方式 | license | 核验证据 | 备注 |
|---|---|---|---|---|---|
| 固定 Stormer 生产加载与 rollout | `earthdelta/probe/rollout.py::load_bridge, rollout_trajectory`；`bridge/stormer_bridge.py::WeatherStepBridge` | import | 内部 MIT 声明；Stormer MIT | `rollout.py:33,49`；`SYMBOLS.json` | 生产路径固定 69×128×256、hidden=1024；T2 不能声称它是通用小模型接口，CPU 用注入的假 rollout。未加载真实 checkpoint |
| 全矩阵编辑 | `earthdelta/probe/edits.py::DirectWeightEditor, pristine_state_digest` | import；新写调用边界 | 内部 MIT 声明 | `edits.py:12,22,33`；`test_direct_editor_partial_entry_failure_is_not_safe` | 无共享模型并发保护；进入前校验全部 block/shape。第二个 block 失败时第一个 hook 会遗留，负例已复现。不能直接依赖 `with` 在 `__enter__` 失败时清理 |
| LoRA 与受控 rollout | `static_adapter.py::build_fs_adapter, fs_edit_plan`；`lowrank.py::ExpertLoRA`；`bridge/stormer_bridge.py::controlled_rollout` | import | 内部 MIT 声明 | `static_adapter.py:477,537`；`lowrank.py:17`；`stormer_bridge.py:1115`；合成零编辑通过 | 不重写 LoRA。不在 controlled_rollout 外包 checkpoint；新调用端管理每臂状态和冻结检查 |
| LoRA 设计对照 | `reference/peft`；`reference/aurora/aurora/model/lora.py::LoRA, LoRARollout` | 只读参考 | Apache-2.0 / MIT | 两个本地 LICENSE；`aurora/model/lora.py:15,67` | Aurora 有 single/from_second/all 步模式；不因能复用就引入额外算法变体 |
| 官方主指标与交叉核对 | `earthdelta/wbx/evaluate.py::evaluate, evaluate_single_chunk, standard_metrics, make_aggregator` | import | 内部 MIT 声明；WBX/WB2 Apache-2.0 | `evaluate.py:132,147,196,269`；两个 `WBX_*_CHECK.json` | 官方计算不重写。single_chunk 的 valid_times 仍可省略，且不绑定调用者给的时间与坐标；T2/T4 包装必须补校验，不能把它当完整数据访问边界 |
| 逐 issue MSE、pooled RMSE | `earthdelta/v8/standard_metrics.py::issue_mse, relative_rmse` | import；新写薄封装 | 内部 MIT 声明 | `standard_metrics.py:76,93`；宽度不变性和 WBX 对照 | 输入是单变量 `[N,L,H,W]`；6 变量需逐变量调用/堆叠。未来 `[N,V,L]` 已算 MSE 的接口只写 pooled reduction，百分比与百分点明确分开 |
| 可微短时效损失 | `static_adapter.py::area_weight_q` 的 torch 权重；v8/WBX 作核对 | import；新写 | 内部 MIT 声明 | `static_adapter.py:208`；`HELPER_SEMANTICS.json` | NumPy issue_mse 不可直接反传。补完整 H×W 归一化、任意短轨迹和版本化分母；不搬旧 l6/cell_mse 的数值 |
| bootstrap 与同时区间 | `arch.bootstrap::StationaryBootstrap, CircularBlockBootstrap`；`conf_int` | pip；新写充分统计量/掩码薄封装 | NCSA | 官方接口文档；本轮 5 个 arch 例子因未安装跳过 | 采样器可复用；pooled RMSE 的非线性统计量和同时覆盖仍须明确。见下一节；当前为有条件复用 |
| 多模型比较诊断 | `arch.bootstrap::StepM, SPA, MCS` | pip | NCSA | [StepM 官方接口](https://arch.readthedocs.io/en/stable/multiple-comparison/generated/arch.bootstrap.StepM.html)；[比较示例](https://arch.readthedocs.io/en/stable/multiple-comparison/multiple-comparison_examples.html) | 输出优胜模型、p 值或模型集合，不是 24 格 RMSE 同时区间；不能单独驱动 G-B/G-H |
| OCL 回归与基础分割 | `sklearn.linear_model::Ridge,RidgeCV`；`sklearn.model_selection::TimeSeriesSplit`；`numpy.linalg.svd` | pip；新写因果特征与 EOF 管线 | BSD-3-Clause / NumPy BSD-3-Clause | 3 项 sklearn 合成测试；T2/T3A 原计划的分割规则 | 不实现岭回归求解器。TimeSeriesSplit 的 gap 按行数，不懂 available_time；需按日期/支持窗 purge。每一折的 EOF/均值只在该折训练部分拟合 |
| DABC 的记录筛选 | `memory.py::VerifiedRecord, eligible_records, ewma_error_feature` | import；新写日历衰减 | 内部 MIT 声明 | `memory.py:18,47,128`；2 项合成检查 | 复用时间顺序/筛选；EWMA 按记录条数，不能用于不规则日历的正式 DABC。每天一条时 `decay=exp(-1/tau)` 的特例已对账 |
| 释放队列、Derived、actor/scorer | `memory.VerifiedRecord` 的时间规则；`probe/access.py::ProbeAccessController` 的台账字段 | 新写；import 已释放记录的筛选 | 内部 MIT 声明 | `memory.py:32`；`access.py:64,112,137`；字段见 `SYMBOL_REVIEW.md` | VerifiedRecord 的 tensor 内容可变、payload 仅向量；access 控制器绑定旧 spec，没有每次起报 cutoff。不得直接暴露 pending 列表或原始 store，必须写时钟和边界 |
| 全矩阵梯度/校准 | `scripts/probe_p3_directions.py::target_params,flat_delta,calibration_effect`；内部 effect/calibrate | 只读参考；新写通用短窗口封装 | 内部 MIT 声明 | `probe_p3_directions.py:71,78,111,197,204`；`HELPER_SEMANTICS.json` | effect/calibrate 是 main 局部函数，不能 import。target_params/flat_delta 写死 blocks 18–23，需新 metadata 驱动接口；效应比定义见下文 |
| 在线更新规则（T5） | `reference/PETSA/tta/petsa.py`；待 A2b 的 `reference/TAFAS`、`reference/OnlineTSF` | 只读参考 | PETSA CC-BY-NC-SA-4.0；另两项未核验 | 本地 PETSA LICENSE 与源码；inventory/授权状态 | PETSA 不拷贝；另两仓库本地缺失，不能声称已核对其 HEAD、日期、license 或协议 |
| 输出订正/门控设计 | `reference/ORCA/eval/online_training.py`；`core/refiner_bay.py` | 只读参考 | 未找到 LICENSE | `online_training.py:371`；`refiner_bay.py:153,283,383,510` | 只提炼设计，不复制代码，不直接引入整个框架 |
| 回路内订正（T5 可选） | `reference/INC/trainer/trainer_1d.py::simulate_rollout`；`reference/Solver-in-the-Loop` | 只读参考 | Apache-2.0 / MIT | `trainer_1d.py:154,699` 与本地 LICENSE | 此阶段无需 vendor；PDE solver 组件并不能直接成为 Stormer 的在线适配实现 |
| 数据修复（T6） | `v8/data_repair.py::repair_zarr_year,refetch_zarr_indices`；`data/pull_wb2.py::ConservativeRegridder` | import（未来 A6/A7 范围内） | 内部 MIT 声明 | `data_repair.py:95,164`；`pull_wb2.py:318` | 只核对源码；不读取 store。修复索引子集不等于全年认证，包装函数并非年份授权边界 |
| 其他原型 | `heads.ReferenceErrorHead`；`probe.estimators`；`probe.budget.BudgetLedger` | 只读参考 | 内部 MIT 声明 | `heads.py:106`；`estimators.py:18,30`；`budget.py:16` | head 不是现成 OCL；estimators 的逐 issue 相对 MSE 均值不作新主指标；budget 不作为已验证的全阶段账本 |

66 个源码符号均已定位，完整签名/行号在 `RUN/SYMBOLS.json` 和 `RUN/SYMBOL_REVIEW.md`。这是符号与契约审查，不代表 66 个函数全部完成数值验证。几个需要纠正的细节：`build_fs_adapter(seed=...)` 会把 down/A 替换成 std=0.02 的正态初始化（`:516`），不是始终 kaiming；`init_up_std=0` 的零编辑性质已验证。`issue_index` 只做 6h 对齐，跨年边界仍需新封装。

## 统计与指标的具体改动建议

1. **保持原 estimand。** 每格用共同评分掩码，先在 issue 上池化 MSE 再开方：`I_a=100*(1-sqrt(S_a/n)/sqrt(S_F0/n))`。差值 `I_a-I_b` 共用 F0 分母，单位为百分点；不是用 b 重新作分母，也不是逐 issue 改善率的平均。
2. **按周保留足够信息。** 周 w 保存各方法/格子的误差平方和 `S[w,a,c]` 与共同有效数 `n[w,c]`，用同一个采样索引重采样整张表；每次用 `sum(S)/sum(n)` 重建 pooled RMSE。仅保留“周平均损失差”会丢掉 F0 分母和缺失数量，不能恢复这个指标。空周保留日历位置，不把有缺失的周拼接成相邻日期。
3. **采样器和推断对象分开。** 建议优先采用 arch 的 circular sampler；若先聚合为 7 天点，7/14/28 天对应 1/2/4 个周单位，不能再次把 7 作为“周”块长度。周的日历锚点、端点周、空样本拒绝规则与随机种子在协议签署前冻结。StationaryBootstrap 可作为明确记录平均块长的备选，不能静默切换。
4. **同时区间需要另行明确。** 建议用 arch `conf_int(method='percentile')` 返回的逐统计量区间做 Bonferroni 家族调整：F_B/F_H/F_C 分别包含 8/24/12 项，对每族使用覆盖率 `1-0.05/M`；边际区间另用 0.95。所有对象包含主臂与基线的差值统计量；可共享一次 arch 采样/使用其复用选项。该方案是保守的近似 bootstrap 家族区间，不声称有限样本精确覆盖；10000 次抽样在 24 项族尾部约只有 10 个样本，需报告 Monte Carlo 限制、做合成覆盖诊断。此为待采纳的计划修订，尚未安装 arch、尚未验证覆盖率，更没有替人签署统计方法。
5. **StepM/SPA/MCS 用途单列。** 可以为固定的模型损失比较提供补充检验，但不能把 superior_models、MCS included 或 SPA pvalues 填进同时区间上下界。其输入损失均值与 RMSE 百分点阈值并不等价。[官方区间接口](https://arch.readthedocs.io/en/stable/bootstrap/generated/generated/arch.bootstrap.StationaryBootstrap.conf_int.html)与[多重比较接口](https://arch.readthedocs.io/en/stable/multiple-comparison/multiple-comparison_examples.html)分别记录了这两种输出。
6. **WBX 对照只在匹配网格上承诺。** 本轮使用全球等距、无极点的格心网格；WBX 用球面网格盒面积，v8 用 cos(lat) 归一化，两者在这个网格上相同。源码 `pull_wb2.py:165` 声明的 Stormer 目标网格也是该类型；本轮未读真实坐标数组，不能据此认证真实 store。任意不规则/极点网格不能无条件断言两套权重相同。NumPy 评分包装统一 float64、显式坐标和通道身份。

## 需要新写而不能直接复用的边界

- **不可变监督记录和隔离。** `VerifiedRecord` 防御性 clone 能阻断源 tensor 的修改，但不能阻断 `record.value.add_()`；`eligible_records` 在过滤可用时间前检查全体 ID，故 actor 也不能拿到包含 pending 的列表。新队列只暴露已释放快照，记录带 model/loss/norm/denominator 版本；copy + read-only payload，并避免把队列持有的可变引用暴露给调用者。没有记录时的维度须显式给出，不从未来记录推断。
- **安全编辑调用。** 进入 DirectWeightEditor 前检查所有目标层、形状、device、dtype；用独占模型或调用端互斥，不允许共享 backbone 并发。失败路径验证 hook 清零；不要全局清除他人 hook。backbone digest 不含 requires_grad、模式和 RNG，分别保存/恢复/校验。带编辑反传时禁用 checkpoint，或保证整个重算期 hook 有效并用有限差分验算；默认优先采用无 checkpoint 路径。
- **短窗口效应比。** 冻结 `calibration_effect` 实际在第 4 步，将所有状态标准化通道的平方误差先池化后开方，并返回 batch 内比值的 RMS；不是逐变量比值的算术平均。双通道合成反例：冻结函数得到 `sqrt(2/10)=0.4472136`，逐通道比值平均为 `0.6666667`。建议短窗口新函数保留前一种公式，batch=1 逐 issue 计算，再按 T3B 在校准 issue 上取中位数；第 4 步要求包含初始场的 T>=5，不再受完整 21 状态断言限制。零/非有限 F0 误差分母必须显式拒绝。若希望改成六变量逐比值平均，须另写协议变更，不能冒称与旧函数一致。
- **OCL 时间验证。** 时间排序不等于防泄漏。`TimeSeriesSplit(gap)` 以样本行数计算，不能识别 120h 标签延迟和缺失日期；需要按 available/支持窗过滤训练行。EOF、中心化、标准化必须在每折训练段内拟合，随后最后一次才用完整拟合期重新拟合；RidgeCV 的默认 cv=None 不是时间切分。[TimeSeriesSplit 官方说明](https://scikit-learn.org/1.2/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

## 参考实现要点

以下是固定本地 HEAD 的只读阅读结论，不是对这些第三方方法效果的验证；没有运行它们或拷贝其代码。

### PETSA（87853d8，CC-BY-NC-SA-4.0）

1. `tta/petsa.py:161,230` 维护 `cur_step` 和每批预测终点；完整标签到达 `cur_step >= pred_step_end` 后才消费历史批。另有当前批的部分真值更新。这里的时间单位是序列步，不是 ERA5 的发布日期。
2. `tta/petsa.py:174,217,230,270` 用固定 batch 或从输入 FFT 估计周期决定批跨度；每次完整/部分监督执行 `cfg.TTA.PETSA.STEPS` 次更新。`config.py:92,113` 的默认学习率 0.005、steps=1，只是该实现默认，不能直接迁移为 EarthDelta 参数。
3. `tta/petsa.py:337` 的 GCM 使用低秩 A/B、零初始化 B、tanh 门控和残差加法，作用于输入/输出校准；这个 gate 不是根据已兑现误差决定发布 F0 的保护门。
4. `tta/petsa.py:230` 消费并移除已经兑现的历史批，同时用当前部分观测适配；所读路径没有按实际日历年龄统一折扣的机制，EarthDelta 仍需自行定义过时监督的处理。
5. `tta/petsa.py:192,317` 可在部分真值更新后重新预测并替换尚未观测的后缀，再汇总 MSE/MAE。该协议不等同于一次发布后不可回写的完整 120h 预报；T5 只参考“已兑现部分”的思想，不能照搬评价流。

### ORCA（e00bc06，未找到 LICENSE，只读参考）

1. `eval/online_training.py:371` 中 pending snapshot 只有在 `t_start+expected_H <= global_t` 时才拼出完整标签；`:426` 又校验闭合长度。这是 horizon-paced 的闭合队列，不含 ERA5 analysis-window 可用性语义。
2. `core/refiner_bay.py:179,349,493` 使用持久 AdamW；收集足够窗口后，每 H 个闭合样本触发一次训练周期。默认 lr=1e-4、热身 50 epochs、之后每周期 10 步（`:20`），不是每个时刻无限更新。
3. `core/refiner_bay.py:153,449,510` 用已闭合窗口的 base/refined 误差 EMA 建门控，支持 boltzmann、比例或 hard；热身完成前返回原始预报，之后按 gate 混合修正。EarthDelta 应复用设计语义，独立实现自己的 F0 回退规则。
4. `core/refiner_bay.py:283` 的 replay 采样按记录年龄指数衰减，系数 2/4000；记录步数不等于实际日历时间，不能直接作为缺日数据的日历遗忘。
5. `eval/evaluator.py:1983,1997,2050,2291` 使用时间流上的 train/val/test 窗口划分、单流更新、只在 test 区间计分，每个模型/数据/时效重置状态。需要审查的是闭合队列和发布次序，不能把此实现当作 EarthDelta 因果正确性的证据。

### INC（16d29b6，Apache-2.0）

1. `trainer/trainer_1d.py:154` 每个自回归步从当前状态生成 correction，作为 PDE solver 的 source 项进入下一步；可借鉴“回路内订正”与事后输出订正的区别。
2. `trainer/trainer_1d.py:699` 用已备好的目标轨迹训练整段 rollout；这是离线监督，没有 delayed-label release clock。不能把它直接称为在线适配基线。
3. `trainer/trainer_1d.py:576,602,631` 分开训练、验证和测试；梯度更新发生在训练 batch 中，推理回路不会自动执行在线 optimizer step。
4. `trainer/trainer_1d.py:154` 的 `correction_term=None` 是明确的无订正对照，但该函数没有按兑现误差选择这一分支的 gate。T5 若需要保护门仍须新写。
5. `trainer/trainer_1d.py:699` 支持梯度/谱正则等选项，但所读路径没有监督年龄衰减或过时反馈筛选。它为回路位置提供参考，不解决 EarthDelta 的时间可用性问题；本轮不引入这些额外损失。

### TAFAS 与 OnlineTSF：待 A2b，尚未读固定版本

两个本地目录不存在，未执行克隆或下载替代代码，HEAD/日期/license 均为 MISSING。不能拿 PETSA 自带的 `tta/tafas.py` 当作已核对的 `kimanki/TAFAS`，也不能仅据计划中的 PROCEED 说明声称完成源码审查。

签署后各自补齐以下 5 点并附实际路径：真值释放时间和预测跨度；更新节奏/步数/学习率；门控及回退；陈旧监督的处理；训练/验证/在线测试与发布次序。该缺口阻塞 batch 5 的复用定案和 T5 设计对照，不阻塞已经得到的 WBX/sklearn/v8 核验结论。

## 对 T2–T5 的影响

| 任务 | 采纳的复用 | 必须补的薄封装/约束 | 当前放行边界 |
|---|---|---|---|
| T2 | v8 issue_mse、WBX、现有 LoRA、sklearn；arch 待签 | 时钟/队列/Derived/两类 store；可微短窗口损失；通用梯度元数据；共享掩码、周充分统计量和家族区间；时间 purge 与折内 EOF | 本轮不实现；arch 支路仍待 A2a，T0 全部验收尚未齐备 |
| T3A | 固定模型加载/rollout；Ridge/EOF 管线复用 T2 | F0 库只写一次；保留逐 issue MSE 与有效数，拟合折内重拟合 EOF；明确运行环境 | 不执行；GPU/分析期仍遵守 A3/A4 与原协议 |
| T3B | F0 库、模型和编辑器；短窗口效应比对账 | hook 失败预检；统一效应比公式；重新计算正确分母；未来真值只由 worker 经释放队列提供 | 不执行；不直接 import main 内部局部函数 |
| T4 | 官方 WBX 全链与 arch 采样器 | 从冻结比较族构造百分点差值/区间；时间坐标绑定；保留依赖跳过状态 | 不执行；不能用检验 p 值冒充同时区间，Gate 仍由人裁决 |
| T5 | build_fs_adapter、ExpertLoRA、controlled_rollout | 每臂独立状态、消费游标、日历衰减、已兑现 gate；参考协议差异要说明 | A5 仍待授权；TAFAS/OnlineTSF 的设计审查待 A2b |

补丁为同目录 [t0_reuse.patch](t0_reuse.patch)，只建议修改 T2、T3A、T3B、T4、T5 的任务文字，**未应用**。补丁以 ZIP 中原文为基准，不包含生产代码、阈值变更或任何批准。

## T2 新代码量估算

以下是实现前的人工工作量估算，按非测试 Python 行粗估；不是实际已删除/节省的行数。本轮 T2 实现行数为 0。模型/LoRA 本来就计划复用，不重复计算它们的整库行数。

| T2 模块 | 从零实现估计 | 复用后估计 | 减少 |
|---|---:|---:|---:|
| timeindex | 60 | 60 | 0 |
| clock / Derived | 170 | 150 | 20 |
| stores / QC 边界 | 220 | 180 | 40 |
| losses | 150 | 110 | 40 |
| metrics / bootstrap | 330 | 160 | 170 |
| grad_utils | 260 | 210 | 50 |
| dabc | 100 | 80 | 20 |
| outcorr | 260 | 120 | 140 |
| **合计** | **1550** | **1070** | **480（约 31%）** |

合理范围约减少 350–600 行，主要来自回归求解、采样器和指标归约。该估计以 A2a 补齐 arch 且统计修订被采纳为前提；安全边界、发布状态和因果测试不能为了减少代码而省略。若坚持原有未知“同时区间”接口而不修订，既无法验收，也不能承诺节省这部分代码。

## 核验、偏离和待签署事项

- 新增审计测试位于 `RUN/tools/test_t0_interfaces.py`，没有改 `tests/`。已完成 11 passed / 5 skipped，跳过全部是 arch 未安装；其中两个通过用例是**复现已有语义限制**，不应解读为这些限制已经修好。
- 第一次 WBX 测试在沙箱内因 `/proc/<pid>/stat` 不可见触发 psutil 错误，结果为 2 failed / 9 passed / 5 skipped；保留原始失败日志。在允许的主机 CPU 环境运行完全相同测试后得到上述 11/5，没有换掉 WBX、xarray 或指标实现。
- 手写公式对照最大绝对 RMSE 差为 `7.771561172376096e-16`；WBX/v8 相对改善率差为 `1.4210854715202004e-14` 个百分点。仅证明该合成网格和接口一致，不证明天气技能或研究价值。
- 补充 `helper_semantics.py` 运行成功：旧面积函数在 W=5、常数误差=3 时给出 MSE=45，正确值=9；校准公式差异也复现。只记录，未修冻结代码。
- 现有 CPU 测试选择 `not slow`，结果 **975 passed / 62 skipped / 4 deselected / 0 failed**，耗时约 766 秒。62 个跳过中，58 个由真实数组/归一化常数/检查点读取边界拦截，2 个需要 CUDA，2 个缺少已导出的上游参考输出。4 个 slow 测试未选中。不能把这次受限回归称为所有测试全部通过；命令、具体原因见 `RUN/FINAL_REPORT.md`、`EXISTING_CPU_JUNIT.xml`、`CPU_DATA_GUARD.json`。
- A2a 待签，阻塞 arch 的安装与五项运行验收；如批准，连同上文列出的必要传递依赖和实际环境约束一并确认。已有 WBX/JAX/sklearn 工作，不受此项阻塞。
- A2b 待签，阻塞 TAFAS/OnlineTSF 的固定版本、license 和设计审查。克隆脚本默认会在 reference 写 `_clone_log_batch5.txt`，超出本轮列出的三个可改 reference 文件；下一轮应在已获 A2b 的脚本副本中把日志改到 runs，保留已有目录，不执行针对已有参考目录的删除。
- 正式 `AUTHORIZATIONS.md` 尚未落到指定目录；需要人发布并填写签署，不能把本轮 inputs 副本或口头“继续”当成 A2 授权。本轮不代签。
- 本轮没有新增外部代码、没有 GPU 作业、没有天气数组读取，没有修改冻结源码、历史产物或 `$OUT`；不提交、不推送，runs 不暂存。数据全年 QC、T1 的 P5 修复和所有后续实现均留在原任务范围内。
