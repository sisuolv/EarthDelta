# 当前方法审查：main@4fe55a7af90ea92f62a3232a571af92bfbd6114d

审查日期以 2026-09-21 UTC 记录（美国东部仍为 9 月 20 日）。依据两份用户任务文件、GitHub connector 读取的固定提交、论文/官方来源。没有写入 GitHub、没有启动用户 GPU、没有访问本地真实天气权重或数据。

## 当前 scientific core

不是旧 Notion 的 Complexity Atlas。当前代码形成两个相邻实验路径：

1. `paired.py + ComposedPredictionHead`：预测 reference error `e0` 与候选有限编辑响应 `du`，生成候选分数。
2. `probe.py + geometry.py + teacher.py + selection.py`：以有限差分响应 R 构造局部二次代理，离线解有界教师，在线可以用预测 b/H 规划。

两者尚未被证据证明为同一个已训练、同一指标下的部署系统。主线选择有限候选的真实 `du`，局部 R/QP 降为近似对照，不同时训练所有架构。

## 已有资产与证据级别

| 项目 | 状态 | 边界 |
|---|---|---|
| K-expert low-rank，dense/sparse 路径 | SOURCE_IMPLEMENTED | 稀疏路径确实跳过不活跃 expert/行；不是旧kit仅乘零。端到端GPU加速未验证 |
| paired target、finite cached responses、central differences、box teacher | SOURCE_IMPLEMENTED | 数值原语存在不等于真实天气闭环完成 |
| 多lead paired head，参考误差与编辑响应解码、gain calibration | IMPLEMENTED_NOT_WEATHER_TRAINED | README 未提供真实训练结果；不得要求从零另造相同多lead头 |
| 本地 Stormer 架构与 bridge | IMPLEMENTED_PARTIAL_VALIDATION | 官方独立真实权重 parity 尚未成立 |
| 测试 | REPO_REPORTS_192_PASS_3_SKIP | 本次未在完整仓库运行；读取了测试与门禁源码 |
| 2020 ERA5 数组1464×69×128×256 | REPOSITORY_REPORTED_ONLY | bytes、数据质量、实际 time axis 未在本次环境验证 |
| 两个 checkpoint 文件 | REPOSITORY_REPORTED_ONLY | 不把文件名/大小当成功加载或SHA核验 |
| S0 real checkpoint gate | NOT_ESTABLISHED | 源码存在不等于已真实通过 |
| oracle ceiling、模型预测收益、长期确认 | UNVERIFIED / NO_RESULT_FOUND_IN_READ_SCOPE | 本次不能报告数值天气增益 |
| memory/spectral | PARTIAL / DEFER | memory有校验问题；谱桥/JVP有明确占位 |

## 静态发现

### A01 — P0
`scripts/s0_gate.py` / `verify_zero_edit_equivalence / run_s0_gate`

本地 forward_validation 与本地 controlled_rollout 比较，后者传入空 expert_loras；不是独立官方实现对照。gate 总结没有将全部加载、有限数值和 RMSE 条件纳入。

证据：STATIC_CONFIRMED。最小处理：增加独立 upstream 路径、真实非零 bank 的零系数测试；失败/缺失条件阻断正式 S0；保留旧报告。

### A02 — P0
`earthdelta/bridge/stormer_bridge.py` / `NormalizationContract.from_npz_dir / denormalize_diff`

存在 diff_mean 文件时会使用它；已核查 upstream inference.py 显式使用零增量均值。

证据：STATIC_CONFIRMED_BEHAVIOR; HISTORICAL_RESULT_IMPACT_UNVERIFIED。最小处理：增加 normalization_policy，默认 pinned_inference_zero_diff_mean；legacy 模式只读复算；nonzero diff_mean 反例和真实官方 parity。

### A03 — P0
`earthdelta/selection.py` / `plan_from_prediction / _plan_from_finite_candidates`

有限候选分支没有执行 bound、max_active 和候选数量上限；缺少对候选有限值/类型的完整验证。

证据：STATIC_CONFIRMED。最小处理：公共 validate_feasible_candidates；非法候选显式拒绝；有限/连续共享域；保留 no-edit。

### A04 — P0
`earthdelta/paired.py; earthdelta/geometry.py; earthdelta/heads.py` / `quadratic_gain / from_error / ComposedPredictionHead.forward`

paired 对 F 权重归一化，geometry 用未归一化权重，head 用无权均值并叠加 trainable gain_calibration；不是同一个统一分数。

证据：STATIC_CONFIRMED。最小处理：MetricSpec 绑定变量尺度、面积、lead、投影和约简；gain_analytic 与 gain_calibrated 分开；默认禁用校准项。

### A05 — P0
`earthdelta/bridge/stormer_bridge.py; earthdelta/contracts.py` / `load_stormer_checkpoint / NormalizationContract.digest / ArtifactVersion`

backbone 身份主要是架构而非实际权重 SHA；norm digest 不绑定变量名/步长键；grid 名称不是实际坐标身份；版本检查未成为执行入口必经门。

证据：STATIC_CONFIRMED。最小处理：内容哈希与结构schema；字典、静态适配、projection、split、continuation 都在 serving/caching 前核验；兼容旧schema只读。

### A06 — P0
`earthdelta/data/make_splits.py` / `build_manifest / compute_available_time`

6h 核验延迟为手设情景；naive datetime.timestamp 依赖进程时区；event_id 是起报时效键而非独立天气过程；边界没有证明实际样本完整。

证据：STATIC_CONFIRMED。最小处理：显式 timezone.utc，available_time provenance/role，actual time index 完整性；独立 process_group_id；默认 retrospective_open_loop，memory 关闭。

### A07 — P0
`earthdelta/bridge/stormer_bridge.py` / `controlled_rollout`

目前仅返回末态；EditPlan 浮点 tuple 转 tensor，不能把该接口直接当控制器端到端可微输出通道。共享 forward_hooks 已有串行清理但并发/重算未证明。

证据：STATIC_CONFIRMED / CONCURRENCY_UNVERIFIED。最小处理：保留末态兼容，新增 return_trajectory 和系数张量路径；显式上下文与 checkpoint replay 测试，不随意全重写。

### A08 — P1
`earthdelta/heads.py; earthdelta/contracts.py` / `ComposedPredictionHead / EditPlan.descriptor`

多时效接口已存在；描述符是 active、coefficients、hold，没有新专家内容的表示；未证明新字典/新时效迁移。

证据：STATIC_CONFIRMED; TRAINING_UNVERIFIED。最小处理：保留已有多lead头；先做同字典未见幅度/窗口/组合；全新专家推迟并另设描述符/标定协议。

### A09 — DEFER
`earthdelta/memory.py` / `ewma_error_feature_batched`

捕获所有 ValueError 后归零，可掩盖版本不匹配或重复ID；zip 输入长度可截断；默认 EWMA 未充分验证 decay。

证据：STATIC_CONFIRMED。最小处理：本轮主线禁用 memory；重启该模块前窄化异常、长度和衰减验证、跨时区/来源测试。

### A10 — STATUS
`earthdelta/probe.py; earthdelta/spectral.py` / `cached_responses / central_response / jvp_response / band_energy`

有限候选 du 和局部导数均有实现，但 JVP、band_energy、single_mode_energy 明确为 NotImplementedError。

证据：STATIC_CONFIRMED。最小处理：准确标注实现状态；P0 优先 cached nonlinear du，不扩充谱/JVP。

### A11 — P0
`plans/plans_v1_0919/v6_draft/research_spec_v6.yaml` / `context_encoder / claim / falsification_gates`

将 step0 后段 reference tokens 称为零额外成本、预写 beats/zero-shot、固定2–3%门槛没有当前测量依据。

证据：STATIC_CONFIRMED_PLAN_NOT_RESULT。最小处理：计 reference preview 与必要重放，改可证伪假设，δ_min 从 pilot噪声/价值/功效预登记；不把 old claim 当事实。

### A12 — STATUS
`tests/test_bridge.py; README.md` / `CPU tests / real checkpoint skips`

README 同时写195 tests及192 passed 3 skipped；已读测试包含随机小模型、归一化零均值fixture、可跳过真实资源路径。

证据：REPOSITORY_REPORTED; NOT_RERUN_HERE。最小处理：保存现有测试全部；本次不宣称195通过，不把原kit21/50计入当前仓库独立天气证据。

### A13 — P0
`reference/_manifest.json` / `license / pins`

已有46项引用清单；WeatherPEFT、CoMoL、GEPS、W2T 等标NONE，短SHA不能替代完整锁和实际许可确认。

证据：REPOSITORY_REPORTED。最小处理：优先许可明确的基础代码；NONE视为复用阻塞，不重新发布源码；可独立实现数学基线并保留引用。

## 当前代码不能证明的事项

没有证据证明双头优于直接 gain；没有证据证明参数编辑优于多变量反馈式输出订正；没有证据证明 du 的训练集拟合可迁移到未见动作组合。不得由测试数量、模块数量或数据文件数量推断这些。

特别保留已有正确部分：`ExpertLoRA.forward_sparse` 的真实跳过、try/finally hook 清理、float64 teacher算术、no-op教师候选、finite响应缓存、按可用时间筛选记录。修改应围绕缺口进行。

## 源码定位

所有代码事实均对应固定提交目录：https://github.com/sisuolv/EarthDelta/blob/4fe55a7af90ea92f62a3232a571af92bfbd6114d/
逐条机器可读证据见 `evidence/audit_findings.json`。本报告是定向静态审查而非逐行全仓形式化验证。大型历史 review packet、所有脚本和完整测试未逐行复核；当前外部部署资产未 materialize，未重跑当前仓库测试。
