# v6 统一包：两个 kit 的模块合并映射（草案，2026-09-19）

原则：不改动 `extracted/` 下两个 kit 的原始文件；新建 `earthdelta/` 统一包；两个 kit 现有的 91 项 CPU 测试（41 + 50）改 import 路径后全部保留，再加桥接/数据/评测的新测试。

| v6 模块 | 来源 | 保留 | 改动 |
|---|---|---|---|
| `earthdelta/contracts.py` | v5 `contracts.py` + Response `program.py` | `ArtifactVersion`（版本指纹与失配拒绝）、`EditPlan` 的不可变与结构校验、`Slot/ProgramSpec` | `EditPlan` 从"rank 组掩码"泛化为"专家字典系数 + 作用窗"；`ProgramSpec.coefficients_for` 保留 dense 展开，但显式标注不跳过矩阵乘 |
| `earthdelta/paired.py` | v5 `paired.py` | `edit_responses`、`quadratic_gain`、`make_training_targets`（离线专用） | 无；增加"D 必须是线性算子"的断言与文档 |
| `earthdelta/probe.py` | Response `probe.py` | `central_response`、`local_linearity_error` | 增加对有限候选的"直接缓存非线性 du"入口；预留 `torch.func.jvp` 路径 |
| `earthdelta/geometry.py` | Response `geometry.py` | `ResponseGeometry.from_error`、`response_distillation` | 无 |
| `earthdelta/teacher.py` | Response `teacher.py` | `box_candidates`（`lsq_linear`）、`verify_candidates` | 成本从"加性单位成本"改为可注入的实测成本表 |
| `earthdelta/selection.py` | v5 `selection.py` + Response `utility.plan_from_prediction` | `select_plan`（预算可行集、确定性并列）、`selection_regret`、`plan_from_prediction`（枚举支持 + L-BFGS-B 盒 QP） | 统一接口：有限候选 = 连续程序在离散支持上的特例 |
| `earthdelta/heads.py` | Response `student.py`、`utility.InteractionUtilityHead`、v5 `model.py` | `BoundedProgramHead`、`InteractionUtilityHead`（Cholesky PSD 的 (b,H)）、v5 的 e0 头 / du 头 / 物理读出 / gain 校准 | v5 的 EMA 目标编码器与潜空间 JEPA loss 改为可选分支（消融），默认直接在摘要空间回归 e0、du；控制器输入改为冻结主干第 0 步池化特征 |
| `earthdelta/memory.py` | v5 `memory.py` | `VerifiedRecord`（issue<valid<=available 校验）、`eligible_records`、`read_memory` | `gated_delta_replay` 降级为消融；默认用 EWMA 近期误差特征 |
| `earthdelta/spectral.py` | v5 `spectral.py` | `coefficient_diagnostics`、`assert_same_latitude_nodes` | 新增 torch-harmonics 桥（`RealSHT` 节点/lmax 显式化，colatitude→latitude 转换测试） |
| `earthdelta/lowrank.py` | v5 `rank_groups.py` | 按行执行、跳过关闭分支、冻结尾部梯度测试 | 泛化为 `ExpertLoRA(attn.proj)`：K 个专家共享或独立 A/B、显式 `coefficients` 参数、dense 可微路径与 sparse 推理路径的等价测试 |
| `earthdelta/bridge/stormer_bridge.py` | 新 | — | 加载官方 ckpt（strict 加载、参数名/shape 校验）；`MemEffAttention`→SDPA 可切换 + 数值一致性测试；`WeatherStepBridge` 封装 `iterative_module.forward_validation` 的归一化/常量/间隔合同；`controlled_rollout(bridge, history, plan)`，真值不进参数 |
| `earthdelta/data/` | 新（复用 Stormer 预处理脚本） | — | `cds_download.py`（1.40625 服务端网格、按月拆请求）、`make_splits.py`（issue/valid/available_time、保护窗）、`build_probe_cache.py`（Zarr/Parquet 存储规范来自 v5 §6.2） |
| `earthdelta/eval/` | 新 | — | `export_wbx.py`（xarray：init_time/lead_time/lat/lon/level）、`paired_analysis.py`（配对时间块 bootstrap、regret、有害编辑率、成本账本） |
| `tests/` | 两个 kit 的 91 项 | 全部 | 新增：桥接零编辑等价、归一化往返、跨分支无状态串扰、dense/sparse 等价、SHT 单模态/常量/平移测试 |

删除或不再作为主路径的内容：v5 的 `PairedEditPredictor` 作为整体模型（拆成 heads 的可选分支）；v5 的 rank-组"动态 rank"叙事；Response 的"层×时间步 8 槽"首版程序（改为"K 专家系数"，8 槽作为扩展实验）；对旧 starter `StateGatedLinear`/`TemporalPatchState` 的依赖（文件不在本机）。

首批提交顺序（对应方案 §4.6）：
1. `test: import both kits' 91 tests under earthdelta/ namespace`
2. `feat: stormer bridge with SDPA parity and controlled_rollout` (S0)
3. `feat: cds downloader + split manifest with available_time` (S1)
4. `feat: static reference LoRA and heterogeneous expert dictionary` (S2)
5. `feat: probe cache + headroom report (P1)` (S3)
6. `feat: controllers B2/D0/D1/U0/U1/finite response heads (P2)` (S4)
7. `feat: transfer experiments (P3) and mechanism ablations` (S5)
8. `eval: weatherbenchX export and paired audit` (S6)
