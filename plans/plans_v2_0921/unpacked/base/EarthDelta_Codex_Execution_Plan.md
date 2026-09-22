# EarthDelta_Codex_Execution_Plan

**版本：round2-independent-review / next-iteration。日期：2026-09-21。**

这是一个迭代、五个有依赖的工程任务，不是新研究综述或完整P0/P1/P2路线。唯一当前授权为本包P0-01；本文中的P0编号属于 `round2-next-iteration-20260921`，与仓库历史同号任务不同。即使旧P0-01已经完成，本包P0-01仍未执行。

## 0. Mission 与最终决定

`research_verdict=BOUNDED_CONTINUE`。下一笔预算购买：最小合格非零字典+完整候选开发缓存+小型合法策略筛查。仅在这两层证据支持后才做输出纠错挑战。

主张最多两条：候选条件化有限轨迹响应具有实际决策价值；响应仿真/核验误差监督分离有有限样本、资源或未见动作复用优势。二次恒等式/LoRA/预算本身不是创新。不能以“没有同标题论文”宣称novelty成立。

`implementation_readiness=NOT_READY_FOR_CREDIBLE_WEATHER_COMPARISON`：先闭合实际用到的参考/身份/数据/指标路径；不要求修完所有B项。

## 1. Current ground truth

| 字段 | 冻结事实与证据级别 |
|---|---|
| Repository | `sisuolv/EarthDelta`（私有，当前授权连接可读取） |
| Requested branch | `audit/round2-review-20260921` |
| 当前分支HEAD（本次元数据核对） | `403b55db65f4c35c1a85d0794ad0de2765b07d96` |
| 前轮受审源码HEAD | `fb767f7f6efbc428be39c9ad84f5905331d6e40f`，受审区间`4fe55a7..fb767f7` |
| 分支头与受审点关系 | ahead 1；根树比较显示原所有条目SHA相同，仅新增`codex_audit_round2/`。不得据新HEAD声称B项已修复 |
| 已有模块 | `paired/probe/geometry/teacher/selection/heads/lowrank/contracts/metrics_contract`，Stormer bridge、数据pull/split、S0/exporter、相关CPU测试；源码存在不代表天气验证 |
| 当前接口 | `controlled_rollout`已有`return_trajectory`；tuple系数和新建leaf不保留caller/controller图；bank/input可有梯度；`ExpertLoRA`已有dense/sparse路径 |
| 已有CPU证据 | 远端旧日志290项：283 passed、7 skipped、0 failed；本次重取日志，**未重跑仓库测试** |
| GPU / xformers真实S0 | `BLOCKED_UNVERIFIED_ASSETS`：无本次核实过的真实通过产物 |
| 本地checkpoint/ERA5/训练bank | 执行者机器上 `UNVERIFIED`，preflight记录存在性不等于合格；不得从仓库目录推断大文件存在 |
| 天气增益/合法策略/论文结论 | `NOT_ESTABLISHED`；本包没有运行训练或天气实验 |

元数据依据：GitHub branches、两点root tree和`codex_audit_round2/results/evidence/pytest_cpu.log`。完整链接/树SHA在`evidence/REPOSITORY_FREEZE.json`，日志副本`evidence/pytest_cpu_prior.log`。本轮不重新拉取或核验论文，研究判断继承刚完成的独立复核。

### 当前真正 blocking 的内容

B01/B02/B10：实际使用checkpoint的独立参考、正确零diff_mean policy、输入/输出/多步身份绑定。B05/B06：实际主评分的合法Q/scale/全目标归一化。B08/B09：实际消费窗口的内容和真实时间索引准入。

这些最小路径必须合格；不要求泛化修复全部public API。B03不阻塞固定系数bank或缓存heads；B12不阻塞串行无replay；B13/B14不让非必要清理/报告问题变成科学否证；B15正式实浮点registry入口防守即可。详细分类见`IMPLEMENTATION_AUDIT_ACTIONS.md`。

## 2. Scientific contract（不许偷偷更改）

### 2.1 reference、动作与信息

先核实公开F0的S0；若后续训练静态Fs，P0-02再冻结Fs为F_ref并重验zero-edit=Fs。F0与Fs分开报告。bank/静态适配/模型权重/输入scale/网格坐标/变量序/程序窗口/continuation均有内容身份；任何变化建立新run identity并保留旧证据。

动作是冻结registry中的有限参数编辑：plan_id、bank identity、实浮点系数、幅度域、层/窗口、后续回参考规则。显式no-edit总可选，但不是零运行成本。空registry与只含合法no-edit是不同状态。

ServingContext仅含签发时可用历史、可选且计费reference preview、候选描述符和预算；不含truth/未来残差/全部实际候选输出/oracle b,H或未来核验memory。标签仅在offline训练与评分使用。先输出不含标签的选择记录，再由独立scorer join结果；不要把“导入分文件”本身当作隔离。

本轮retrospective_open_loop，不声称真实业务可用时间；不在预测步后读取真实ERA5更新历史。memory显式关闭；固定延迟只标scenario。history默认是已声明的合法短历史，所有基线一致。

### 2.2 e0、du、gain与Q

取同一固定线性D：`e0=D(Y)-D(Yref)`，`du(a)=D(Ya)-D(Yref)`；`gain=2 e0^T Q_eff du-du^T Q_eff du`。该式是实际端点误差差，预测plug-in不是无偏/安全保证。

原始输出张量采用e[B,H,S,F]和du[B,K,H,S,F]。变量scale为s_F，目标权重为q_HSF：`Q_eff=(q_HSF/s_F²)/sum_HSF(q_HSF)`。Q/scale缺省值不得在不同模块独立猜。主gain约简H,S,F一次，batch保持样本独立；perlead/perregion输出单列。scale须有限正、Q须有限非负且有效分母正；当前missing=error。RMSE必须sqrt-after-aggregate。

若使用局部R：`b=R.T @ (Q_eff*e)`，`H=R.T @ diag(Q_eff) @ R`。u=Ra是额外假设；H响应重叠不是完整非线性混合效应。实际有限候选不用线性叠加替代。配对loss差、head gain、coefficient gain必须在同目标下对齐；不靠不同sum/mean或calibration制造收益。

变换后端点作差恒等式并不要求变换线性；但本轮D仍固定线性，谱/能量等非线性模块不进入主line。压缩I下双均值plug-in未必等于E[g|I]；direct-gain与投影/b对照用于检验，不预设二头有优势。

### 2.3 成本与统计

预算使用真实声明并实测的context/planner/selected-rollout/decoder/retry成本，不能把gate稀疏率当节省。候选仿真成本离线另列，truth标签按独立issue/过程计一次。direct-gain/端点模型有相同仿真使用资格。

主估计总体为所有注册起报的等权平均，过程块用于处理相关性；候选数、网格、lead不增加独立样本数。保留全部失败分母、未选候选和no-edit。改变候选数/幅度/过程筛选/Q/lead/时间划分/失败策略必须开新研究规格，不能覆盖旧结果。

E[max g]是oracle；E[max E[g|I]]是合法信息下的最优值。Oracle PASS只支持后续有界筛查；FAIL只针对当前合格registry。默认只读dev；confirm在模型与全部选择冻结且明确授权后读一次。

科学阈值目前均为 `TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。delta_min来自价值和成本；MDE来自方差、alpha/power/N；两者不得等同。可以先做获准开发方差/成本估计，不可以先签科学PASS。保护时效和bootstrap/looks也须预登记。

## 3. Non-goals

不再做literature review，不重写仓库，不新增backbone/大字典/JEPA/memory/dynamic-rank/spectral训练/RL/在线JVP，不做全量P1/P2，不扫描/下载全量ERA5，不搭并发或activation checkpoint框架，不做无关cleanup，不自动pip、下载权重或改凭据，不自动push。旧研究/审计/YAML/测试保留，失败不隐藏。

## 4. 如何开始（不依赖旧执行包）

当前只授权P0-01的最小源码实现与安全CPU检查；任何GPU实测还需本地资产、权限与cap。不要因为用户给了项目链接就代替用户申请GPU或启动训练。

在已有仓库环境中设置真实路径（不硬编码CCI或本机路径）：

```bash
export PACKAGE="/absolute/path/to/EarthDelta_Codex_Execution_Package_20260921"
# 先进入实际仓库；下面不会切换branch。
export REPO="$(git rev-parse --show-toplevel)"
export RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/earthdelta-round2-20260921.XXXXXXXX")"
export CONFIG="$RUN_DIR/pilot_config.json"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cp "$PACKAGE/configs/pilot_config.template.json" "$CONFIG"
python "$PACKAGE/tools/validate_package.py" --root "$PACKAGE"
python "$PACKAGE/tools/preflight.py" --repo "$REPO" --out "$RUN_DIR/preflight.json"
```

将config中的真实路径、资源cap和批准引用从已有本地证据填入，不猜。config是待填模板，不是现成可训练配置。`allow_training=false`、`allow_gpu_run=false`、`allow_confirm_read=false`是初始值，缺批准不能改true。

如果只收到本主计划和仓库而没有附工具：先用`git rev-parse HEAD`、`git status --short`、`git diff --name-status fb767f7f6efbc428be39c9ad84f5905331d6e40f HEAD -- earthdelta scripts tests pyproject.toml`记录preflight，按下述字段建立本地config；P0-01的接口/测试/验收已全部列在本文。不要回到旧16项DAG。

### 配置至少包含

execution：iteration_id、approved_task_ids、owner_approval_reference、allow_gpu_run/training/confirm、serial_only=true、activation_checkpointing=false。

model/s0：checkpoint_path+SHA+patch_size、trusted_source、upstream_pin、normalization_dir+policy+digest、raw_input+hash、变量/坐标身份、registered interval/steps、tol及来源。

data：实际store/index、角色时间窗、history/leads、process规则、retrospective/scenario来源与memory=false。

metric：D标识、scale、q、一次全目标weighted-mean、missing=error、calibration=false。

resources：CPU timeout/RSS、GPU秒、bank更新/尝试、issue/candidate/forward/HPO/seed/bootstrap/storage/retry上限（未定就是null/BLOCKED）。

statistics：各delta_min、保护margin、独立价值依据、方差、alpha/power/N/MDE、bootstrap/looks与cap（未定用上述占位符）。

### 命令和产物接口

除`tools/preflight.py`和`tools/validate_package.py`已在本包实现外，各任务标注的**新脚本和新CLI要由Codex先实现，再运行**。本包没有假装这些脚本已存在。

所有新脚本接受`--config`与`--out`；读取task授权与对应资源cap，未知则先写decision/blockers后退出，禁止自动安装或下载。单元测试资源由当前runner的批准CPU timeout/RSS限制；需要长GPU任务时外部调度器也要施加同cap。`--out`必须新目录或明确resume同identity，不能覆盖旧结果。

`$CERT`=审核后真实threshold_certificate路径；`$OUTCOME_TABLE`=P0-03产物；`$LEGAL_FEATURES`=同run合法context缓存；`$POLICY_MODELS`=P0-04模型目录；`$ITERATION_EVIDENCE`=前任务manifest索引。全由前任务产物解析/显式传入，不能猜文件名补不存在结果。

脚本exit：0=计算完成（仍要读scientific_status，descriptive不是PASS），1=实现错误，2=BLOCKED，3=INCONCLUSIVE，4=STOP/PIVOT。每个脚本先落地机器结果；任何exit code都不自动解锁后任务。

## 5. 决策DAG

```text
P0-01 最小correctness/身份/共同指标与真实S0（当前唯一授权）
  ├─ 缺资产/容差/资源 → BLOCKED；CPU补丁可完成，停止
  ├─ 真实错误 → FAIL_IMPLEMENTATION；只修当前最小路径
  └─ PASS_MEASUREMENT + 新授权
       ↓
P0-02 数据实际准入 + 合格非零bank + 冻结registry
  ├─ 未训练/零bank/数据不齐 → BLOCKED_INVALID_SETUP
  ├─ 到bank cap仍不合格 → STOP_CURRENT_SETUP
  └─ PASS_BANK_AND_DATA + 新授权
       ↓
P0-03 完整有限候选dev表 / oracle-static差距
  ├─ UCI < delta_min → STOP_CURRENT_DICTIONARY / PIVOT_STATIC
  ├─ CI跨delta/阈值未定 → INCONCLUSIVE / DESCRIPTIVE_ONLY
  └─ DEV_PROMISING + 新授权（只证明hindsight余量）
       ↓
P0-04 合法policy/direct-gain/双头/四格筛查
  ├─ 合法策略无有用收益 → STOP_CURRENT_PLANNER
  ├─ static/direct-gain解释收益 → PIVOT_STATIC / PIVOT_NARROW_CLAIM
  ├─ CI跨delta → INCONCLUSIVE到cap停止
  └─ LEGAL_POLICY_DEV_PASS + 有限资源继续理由 + 新授权
       ↓
P0-05 条件性output/feedback/virtual挑战；封存本迭代
  ├─ 纠错支配真实参数执行 → PIVOT_OUTPUT_CORRECTION
  ├─ 保护lead明确退化 → STOP_CURRENT_CONFIGURATION
  ├─ 不足/CI跨delta → INCONCLUSIVE；停止追加
  └─ 支持窄命题 → CONTINUE_BOUNDED_NEXT_ITERATION（本迭代结束）
```

每个节点的PASS/FAIL/CI解释见其gate段；所有箭头还需要明确新授权。初始`allowed_tasks=[P0-01]`，没有其它已解锁节点。

## 6. Detailed Codex Tasks
## TASK P0-01 — 修复一条最小可信测量路径并完成正确性准入

### Scientific purpose
使后续测量确实对应指定参考模型、输入和同一个收益目标；不为此修复所有公开 API。该任务允许最小源码修改，不是又一次只读审计。

### Depends on
无；本包当前唯一授权任务。

### Files to inspect
- `earthdelta/bridge/stormer_bridge.py`
- `earthdelta/bridge/stormer_arch.py`
- `scripts/export_upstream_reference.py`
- `scripts/s0_gate.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/paired.py`
- `earthdelta/geometry.py`
- `earthdelta/heads.py`
- `earthdelta/selection.py`
- `tests/test_upstream_parity.py`
- `tests/test_s0_fail_closed.py`
- `tests/test_metric_contract.py`
- `reference/stormer/inference.py`
- `reference/stormer/stormer/models/iterative_module.py`

### Files to modify
- `earthdelta/bridge/stormer_bridge.py`
- `scripts/export_upstream_reference.py`
- `scripts/s0_gate.py`
- `earthdelta/metrics_contract.py`

### Files to create
- `scripts/round2_iteration/check_correctness.py`
- `tests/test_r2_pilot_contract.py`

### Implementation steps
1. 先运行包内只读 preflight，核对本地 HEAD、dirty 状态、受审/分支源树；403b55d 只增加审计材料，不代表缺陷已修复。未知代码漂移先记录 HEAD_RECONCILIATION.md，禁止 reset、revert 或覆盖用户改动。预检通过后立即在已授权范围做本任务最小补丁，不另起审计项目。
2. 只选择配置中的一个 checkpoint（计划起点 patch4，但实际文件与架构必须匹配）；不要求同时验证 ps2。预期完整 SHA 来自明确可信来源/本地人工确认的资产清单，不以文件存在或 SHA 长度为身份验证。未被信任的 pickle 不反序列化。
3. NormalizationContract 增加显式 official_inference_zero_diff_mean policy：从 pinned 原版输入 mean/std 与分步长 diff std 建立变换，推理增量 mean 强制为零。legacy 非零均值路径保留只读复算并有不同 policy/digest。正常化 digest 同时绑定键、shape、变量顺序与 policy；不重写天气网络。
4. exporter 在独立命名空间运行 pinned 官方网络及官方 iterative_module.forward_validation/官方 transforms。禁止调用待审 NormalizationContract 来构造其预处理。若官方依赖不可用，输出 BLOCKED_UPSTREAM_UNAVAILABLE；不能把本地 rollout 更名 official。
5. 为 exporter 与 s0_gate 增加统一 --config、--out、--upstream-reference CLI（旧入口可保留兼容）。manifest 绑定 checkpoint/上游 pin/归一化 policy+digest/变量顺序/坐标内容/原始输入 hash+shape/输出 hash+shape/interval 与 steps。消费者比较全部注册路径（起点 6h×1、6h×4），不能只检查第一条，也不能拿 ps2 验证 ps4。
6. 在 metrics_contract.py 最小新增 full_objective_gain(error[B,H,S,F], response[B,K,H,S,F], q[H,S,F], scales[F]) -> [B,K] 与 full_objective_loss(pred,target,q,scales) -> [B]。一次按 H,S,F 归一化，先统一工作 dtype，再检查有限非负 Q、有限正 scale、正分母。只保留当前 missing_policy=error；原分 lead/区域 diagnostic helper 保留原 API。Q_eff=(q/s²)/sum(q)，损失和 gain、后续 b/H 使用同一 Q_eff。
7. check_correctness.py 的 --cpu-only 模式验证代数与配置边界，不读取 checkpoint 大文件/不启动 GPU。若 P0-03 仅走有限端点路径，不必修 B04 的未使用 batched QP；B04 若后续启用则以最小 ellipsis 修复。禁止为了统一 helper 重写所有模块。
8. 真实 S0 在批准资源内使用已检查有限值/实际时间与网格的最小本地输入。非零合成 adapter + 零系数测试 identity；非零→零、异常清理、返回轨迹时次对齐；S0 合成 adapter 不是科学 bank。固定 dtype，容差依据重复性/数值误差地板事先记录，不放宽容差来吞错。
9. 状态分离：measurement_pass 由所有必需计算和身份校验决定；必需项异常/缺失必须 false。必要结果持久化失败不得自动放行。非必要 empty_cache 失败可以 cleanup_warning，不必否定已完成测量。Markdown 缺失但 JSON 完整可供人工复核；不将 B13/B14 变成新的科学停止理由。
10. 记录只支持串行、无 activation checkpoint；不实现并发框架。未来真值只存在于检查/评分对象，不进入预测 callable。无 GPU/权重/数据/资源 cap 时完成已许可 CPU 补丁与日志，写 PATCH_CPU_PASS_REAL_S0_BLOCKED 并停止。

### Tests
- 非零 diff_mean 文件不改变 official policy；legacy 与 official digest 不同；真实旧文件不覆盖。
- wrong checkpoint、normalization、原始输入、坐标、exact shape、上游 pin、缺第4步输出任一错均不能 numeric parity 放行；禁止广播掩盖 shape mismatch。
- 非均匀 H/S/F 权重下：全目标损失差=全目标 gain；将 q 乘常数不变；若系数路径被调用，b/H 与 Q_eff 一致。负 Q/零 scale/非有限值与全零分母明确拒绝。
- mock 驱动实际 s0 orchestration，覆盖缺参照/关键计算异常/NaN、资源缺失、正常结果、非必要 cleanup warning；不要只对手造字典 all(...) 断言。
- 非零 adapter 的 zero-edit identity、串行重复调用及 hook cleanup 通过。记录旧相关测试真实通过/跳过/失败；不以新增测试标题代替性质。

### Run command
以下命令按顺序；新脚本/参数先实现，真实模型命令只有资产与对应cap获准后运行。

```bash
python "$PACKAGE/tools/preflight.py" --repo "$REPO" --out "$RUN_DIR/preflight.json"
git -C "$REPO" diff --stat && git -C "$REPO" status --short
cd "$REPO" && python -m pytest tests/test_r2_pilot_contract.py -q
cd "$REPO" && python scripts/round2_iteration/check_correctness.py --config "$CONFIG" --cpu-only --out "$RUN_DIR/contracts"
cd "$REPO" && python scripts/export_upstream_reference.py --config "$CONFIG" --out "$RUN_DIR/upstream"
cd "$REPO" && python scripts/s0_gate.py --config "$CONFIG" --upstream-reference "$RUN_DIR/upstream" --out "$RUN_DIR/s0"
```

### Expected artifacts
- `preflight.json`
- `HEAD_RECONCILIATION.md`
- `contracts/checks.json`
- `contracts/metric_spec.json`
- `s0/measurement.json`
- `s0/reference_manifest.json`
- `s0/independent_parity.json`
- `tests.log`
- `patch.diff`
- `decision.json`

### Success criteria
- 实际运行 commit 与所有输入身份已绑定；原/新 CPU 回归按实际分母记录。
- 实际所用一个 checkpoint 的独立真实 S0 通过全部注册轨迹，指标单元性质通过。
- 若只完成 CPU，则仅标 PATCH_CPU_PASS_REAL_S0_BLOCKED；不能将其写成任务完全 PASS。

### Failure criteria
- 未知代码漂移或来源未信任；必需参数不齐；身份错、数值不等价、非有限数据或指标不一致。
- 缺 GPU/xformers/权重/本地数据/资源批准属于 BLOCKED，不是方向被否证。

### Decision after completion
PASS_MEASUREMENT -> 写 P0-02 可放行建议并停止；BLOCKED/FAIL -> 停在 P0-01，不训练 bank。

**PASS能推出：** 这条受检模型/指标/数据输入路径可测量，不代表天气方法有效。

**FAIL能推出：** 实现/资产未合格，不能做科学判断。

**CI跨阈值：** 不适用统计 CI；数值容差未知则 BLOCKED_TOLERANCE。

### Suggested commit message
`fix(round2): bind minimal reference parity and pilot metric contract`

不得自动push；只在用户既定版本管理授权内提交。

---

## TASK P0-02 — 建立最小数据准入、非零参考字典与冻结 registry

### Scientific purpose
让 oracle 测的是合理训练的真实动作空间，而不是零初始化、随机字典或不存在的时间窗口。

### Depends on
P0-01

### Files to inspect
- `earthdelta/lowrank.py`
- `earthdelta/contracts.py`
- `earthdelta/data/make_splits.py`
- `earthdelta/data/pull_wb2.py`
- `earthdelta/bridge/stormer_bridge.py`
- `plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md`

### Files to modify
- `earthdelta/data/make_splits.py`

### Files to create
- `earthdelta/data/pilot_admission.py`
- `scripts/round2_iteration/prepare_pilot.py`
- `scripts/round2_iteration/train_minimal_bank.py`
- `tests/test_r2_pilot_admission.py`
- `tests/test_r2_bank.py`

### Implementation steps
1. 沿 P0-01 已通过的模型路径，准备实际 timestamp 索引，逐一 join 当前输入、全部历史帧与所有目标 lead；检查 UTC、间隔、重复时间、变量/层/网格/单位及每个会被消费切片的 finite/missing policy。只扫描本次明确窗口，不要求重建全量下载器。
2. is_year_complete/marker 不作为训练准入。对所用片段生成 data_admission.json，绑定 store identity、具体时次/切片、检查方式和摘要hash。残缺、全 NaN、缺帧、重复时次在 Dataset 入口拒绝。日历 generator 可继续作计划表，不能用其 validate_all 证明数据存在。
3. 区分 bank_fit/policy_fit/dev_select/dev_screen/confirm 的真实时间窗；同 issue 全 candidate 和 lead 共享 role/fold；同天气过程在同分组；guard 覆盖实际历史及最远标签。受过 S0/debug 暴露的片段仅作开发。核对 backbone 预训练/选择经历；不能把已用年份包装为全新分布。
4. 最小协议为 retrospective_open_loop：不使用近期核验 memory。固定延迟若出现仅标 scenario；first_seen 无记录不能声称 observed。旧无 provenance 记录保留 unknown/legacy 或仅进入非正式计划，不自动升级。修改这些 builder/loader 默认只为当前路径服务。
5. 优先复用有真实训练日志、匹配身份和 held-out 检查的现成 F_ref/bank。否则在批准 cap 内先训练一个合理静态适配参考 Fs（F0另报），冻结 Fs，然后训练最小数量额外 ExpertLoRA。每次固定一个已知非零系数，优化活跃 bank 参数；固定 tuple 系数并不需要 caller coefficient 梯度。全 backbone/Fs 不更新；不要把整段前向放 no_grad。
6. 以训练集定义的不同 regime 子集或已批准误差目标形成有限的专家差异；不按 confirm 结果选专家。记录各专家训练范围、是否收敛到非零效果、作用层、rank、共享参数和实际资源；不得用未训练随机 bank 的失败否定整个方向。所有 bank 训练从此任务的批准次数/更新数 cap 内完成。
7. freeze registry：显式 no-edit 与至少一个合格非零候选，实际候选数由 cap 决定。每候选有唯一 plan_id、bank hash、实浮点系数、已训练幅度范围、hold/window、target layers、continuation、成本口径。验证 finite、bound、active、max count；空表/复数在这个入口拒绝，保留已有 selection 四项校验。
8. 本轮默认 singleton registry；若要保留 H/组合主张，必须在看收益前追加少量且有预算的组合并单独注册。没有组合只限制结论域，不算失败。候选冻结后不得因为 oracle 不好再偷偷改幅度、候选或Q。
9. prepare_pilot.py 生成可执行 frozen_config.json 与 budget_manifest.json。先测单条轨迹成本/峰值/IO，再填写并批准资源cap；统计 delta_min 等此时可仍是指定占位符，因为此任务不是生死判定。未批准训练则只生成 BLOCKED 所缺字段清单。

### Tests
- 真实索引缺历史/目标、跨角色/缺帧/重复时次、空未写 store 均拒绝；不把 metadata shape 当完成证明。
- no-edit 复现同一 Fs；bank 非零效果可重放；backbone/Fs hash 不变、bank.grad 非零；不要求 caller 系数图。
- registry 空表、复数、非法幅度/窗口/active、版本错拒绝；同 issue 不得按候选拆分。
- retrospective/scenario 不被升级为真实 first-seen；future memory 默认关闭。

### Run command
以下命令按顺序；新脚本/参数先实现，真实模型命令只有资产与对应cap获准后运行。

```bash
cd "$REPO" && python -m pytest tests/test_r2_pilot_admission.py tests/test_r2_bank.py -q
cd "$REPO" && python scripts/round2_iteration/prepare_pilot.py --config "$CONFIG" --out "$RUN_DIR/spec"
cd "$REPO" && python scripts/round2_iteration/train_minimal_bank.py --config "$RUN_DIR/spec/frozen_config.json" --out "$RUN_DIR/bank"
```

### Expected artifacts
- `spec/frozen_config.json`
- `spec/data_admission.json`
- `spec/actual_index.parquet`
- `spec/split_manifest.json`
- `spec/budget_manifest.json`
- `bank/reference_manifest.json`
- `bank/bank_manifest.json`
- `bank/candidate_registry.json`
- `bank/qualification.json`
- `bank/train_log.jsonl`
- `decision.json`

### Success criteria
- 实际消费切片合格；冻结非零 bank 与 reference 身份、候选域、role和资源；每个专家作用可重放。
- 强静态参考/候选选择规则有 fit/dev-only 记录；未访问 confirm。

### Failure criteria
- 无权重/数据/许可或未批准 cap -> BLOCKED。
- 到批准 bank cap 仍无合格非零动作空间 -> STOP_CURRENT_SETUP，不称整个研究不可能。

### Decision after completion
PASS_BANK_AND_DATA -> 申请 P0-03 并停止；不能自动做 oracle。统计阈值未定不阻塞 bank 合格性，但不允许科学 PASS。

**PASS能推出：** 可以合法生成候选表，不意味着动态 headroom 已存在。

**FAIL能推出：** 数据/字典/设置不可用；未训练字典不能否证方法。

**CI跨阈值：** 不以科学 CI 判 bank 资格；资源/身份未知则 BLOCKED。

### Suggested commit message
`feat(round2): admit finite pilot windows and freeze a qualified edit registry`

不得自动push；只在用户既定版本管理授权内提交。

---

## TASK P0-03 — 生成完整候选 dev 表并判定 oracle／静态余量

### Scientific purpose
测量冻结候选域中“有无值得识别的事后空间”；只解锁有界合法策略实验，不把 oracle 当可部署结果。

### Depends on
P0-02

### Files to inspect
- `earthdelta/probe.py`
- `earthdelta/paired.py`
- `earthdelta/selection.py`
- `earthdelta/teacher.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/bridge/stormer_bridge.py`

### Files to modify
无；除非该任务步骤显式要求相邻最小修改。

### Files to create
- `earthdelta/evaluation/pilot_survival.py`
- `scripts/round2_iteration/collect_candidates.py`
- `scripts/round2_iteration/oracle_analysis.py`
- `tests/test_r2_candidate_table.py`
- `tests/test_r2_block_statistics.py`

### Implementation steps
1. collect_candidates.py 仅读已冻结 registry/config；固定最大 lead 一次 rollout 返回全轨迹。从同一起报计算 reference 和每个实际有限非线性候选，不用 Ra、singleton 相加或 top3 教师替代完整候选。复用 cached_responses/controlled_rollout，不扩 JVP。
2. 将合法 ServingContext/可用特征和 OfflineOutcomes 分开保存；合法特征缓存固定输出table/serving_features.npz（纯数值，读取allow_pickle=False）与table/feature_manifest.json，绑定feature_spec/时间范围/尺度/成本/父输入hash。forecast callable 仅收输入状态/plan，目标只进入离线评分。来自参考后段 tokens 的 context 需 preview/重放则计费，不能写零额外开销。
3. 每行至少保存 issue_id/process_id/role/plan_id/reference_hash/bank_hash/registry_hash/metric_hash、各lead损失、全目标loss_ref/loss_edit/raw_gain、total_cost、状态和失败原因。原始输出索引与配置可追溯。noedit raw_gain=0 但执行成本不为零。
4. 事先冻结 failure policy；所有起报和候选都保留。基础设施缺响应 -> BLOCKED_INCOMPLETE_TABLE；数值发散按预登记可部署回退/成本规则结算，否则不能称完整 oracle。不得 dropna 后提高收益；重试和回退算成本。
5. no-edit、best-static、regime 使用共同可行域。best-static/regime 仅在 dev_select/过去训练部分选定，dev_screen 不重新挑；确认时冻结。若要交叉拟合，保持过程级向前时间折，不让同issue候选串折。
6. oracle_analysis.py 计算 oracle 相对 F_ref 与 dev-best-static 的两个 gap，保留全目标MSE、先聚合后开方RMSE与 perlead/pervariable 诊断；CI 按真实过程块重采样，块内候选及 lead 完整配对。不得将 K 个候选或网格点计为 K 倍独立样本。
7. 先 --descriptive-only 得到开发方差/成本并登记 variance_power.json。delta_min 的价值依据必须独立于 MDE，不以当前均值恰好能过关来选。冻结 alpha/power/N/cap/预定looks与阈值证书后才允许 --evaluate-gates。阈值占位符仍在则仅 DESCRIPTIVE_ONLY/BLOCKED_THRESHOLDS。
8. 明确 estimands：oracle=E[max g]；最佳合法信息策略上限=E[max E[g|I]]。candidate 数量造成的 hindsight 增益属于这个estimand，不能当可预测技能；不机械减去打乱oracle。singleton-only 失败仅限该registry。
9. 本迭代默认只做开发筛查，不打开 confirm。DEV_PROMISING 仅可申请小型 P0-04；CI跨delta只按预登记追加上限补证，到cap停。最终 confirm 不应在选择模型前被 oracle 查看。

### Tests
- 显式noedit使完整oracle原始gain非负；e=±1独立I例 oracle=0.75、合法最优0，报告分开。
- 缺一个候选、重复候选、错误bank/hash、无目标、NaN不产生complete标记；所有失败计入登记分母。
- 相同loss表的 oracle、static和差值可复算；全候选与top3状态不同；代价预算约束生效。
- 同过程配对bootstrap保持所有lead与候选；sqrt-after-aggregate；CI跨delta和未定阈值不会得到PASS。

### Run command
以下命令按顺序；新脚本/参数先实现，真实模型命令只有资产与对应cap获准后运行。

```bash
cd "$REPO" && python -m pytest tests/test_r2_candidate_table.py tests/test_r2_block_statistics.py -q
cd "$REPO" && python scripts/round2_iteration/collect_candidates.py --config "$CONFIG" --role dev --all-registered-candidates --out "$RUN_DIR/table"
cd "$REPO" && python scripts/round2_iteration/oracle_analysis.py --config "$CONFIG" --table "$RUN_DIR/table/candidate_outcomes.parquet" --descriptive-only --out "$RUN_DIR/descriptive"
cd "$REPO" && python scripts/round2_iteration/oracle_analysis.py --config "$CONFIG" --table "$RUN_DIR/table/candidate_outcomes.parquet" --threshold-certificate "$CERT" --evaluate-gates --out "$RUN_DIR/gates"
```

### Expected artifacts
- `table/candidate_outcomes.parquet`
- `table/predictions_index.json`
- `table/failure_ledger.jsonl`
- `table/serving_features.npz`
- `table/feature_manifest.json`
- `descriptive/variance_power.json`
- `gates/threshold_certificate.json`
- `gates/oracle_static_gaps.json`
- `gates/bootstrap_intervals.json`
- `gates/cost_report.json`
- `decision.json`

### Success criteria
- 完整注册集、已资格bank、合法分母、共同Q下，两个主gap的lowerCI均高于各自预登记delta_min -> DEV_PROMISING。
- 未用confirm；oracle不被标为deployment skill。

### Failure criteria
- upperCI低于delta_min -> STOP_CURRENT_DICTIONARY；仅动态差距不足 -> PIVOT_STATIC_ADAPTATION。
- CI跨阈值 -> INCONCLUSIVE；到资源cap停止追加；缺候选/阈值 -> BLOCKED或DESCRIPTIVE_ONLY。

### Decision after completion
DEV_PROMISING -> 仅申请 P0-04 小型筛查并停止；不自动放行大训练或论文胜利。

**PASS能推出：** 存在值得识别的 hindsight 空间；非合法收益证明。

**FAIL能推出：** 在当前bank/动作域/Q/lead内余量不足；不能推广到未测组合。

**CI跨阈值：** INCONCLUSIVE；只按预登记cap补证，不改候选/阈值/分母。

### Suggested commit message
`feat(round2): cache exhaustive finite outcomes and separate oracle from legal value`

不得自动push；只在用户既定版本管理授权内提交。

---

## TASK P0-04 — 同缓存交叉拟合合法策略、direct-gain 与双头

### Scientific purpose
检验 oracle 余量是否可被合法信息利用，以及监督分解有没有超出直接决策学习的有限资源价值。

### Depends on
P0-03

### Files to inspect
- `earthdelta/heads.py`
- `earthdelta/paired.py`
- `earthdelta/selection.py`
- `earthdelta/metrics_contract.py`

### Files to modify
- `earthdelta/heads.py`

### Files to create
- `earthdelta/baselines/pilot_policies.py`
- `earthdelta/evaluation/pilot_predictability.py`
- `scripts/round2_iteration/fit_policies.py`
- `scripts/round2_iteration/score_policies.py`
- `tests/test_r2_legal_policy.py`
- `tests/test_r2_four_cells.py`

### Implementation steps
1. 复用已有 history GRU/多lead heads，不另建天气编码器。共享同合法context和候选描述符、数据资格、训练目标尺度、调参预算。先声明 context 生成是否需参考preview并计费；不得用实际全部候选输出作serving feature。
2. heads.py 返回 gain_analytic 与可选 gain_calibrated；新增显式 calibration_enabled=False，关闭相关参数训练；JEPA/memory/variance penalty 默认关闭。noedit u_hat=0 和raw gain=0通过结构保证（例如 response_raw(I,a)-response_raw(I,0)），不把enabled表示为noedit语义。全目标使用P0-01 reducer；不要只对F求均值后再随意平均时空。
3. pilot_policies.py 只实现小型直接gain MLP和简单regime selector，复用head结构形成 direct-edited-forecast 对照：动作条件head监督同D下实际候选端点，分别预测y_hat_a和y_hat_0，以两者之差形成u_hat，再使用相同参考误差信息和共同gain规则。不得免费读取精确未来参考轨迹；若使用实际reference preview必须计费。该对照检验端点/响应目标选择，direct-gain才是无显式分解的决策对照。原paired路径不重写。可选用相同缓存测试误差在响应子空间的投影/b版本，但需原cap内显式批准，不建额外geometry框架。
4. 使用向前时间/过程交叉拟合：编码统计、特征缩放、静态候选、regime规则和模型选择仅用每折过去的fit/dev_select；持出dev_screen全issue所有candidate不得用于训练。标签子集按独立起报/过程采样，一份真值覆盖所有候选，不将K份复制视作K份标签。
5. 仿真数据对各方法同等可用：direct-gain可用同样响应/端点辅助预训练，direct-edited-forecast可用无核验模型输出。记录同标签与总仿真/总计算两个视角，不只给双头额外数据。first trial限制在已登记method/hyperparameter/seed清单内。
6. score_policies.py 的部署路径只接 feature+registry+budget；锁定模型先输出 action_selection.jsonl，然后离线评分器join outcomes。truth poisoning后决策不变；独立未来真值只改变评分。
7. 四格替换：true_e+true_u、true_e+pred_u、pred_e+true_u、pred_e+pred_u。每格选择都在相同实际结果表结算，前3格标OFFLINE_DIAGNOSTIC；true_u虽然不需天气真值，也需要全部候选运行，不能当低成本serving。
8. 报告合法策略相对dev-best-static与regime的配对实际收益，再报告paired相对direct-gain/edited forecast的差异；辅以du方向/范数误差、误差响应投影、harmful/noedit、成本。不得仅凭whole-e0 MSE差宣布方向失败，也不得用联合训练自动抹去压缩context的条件矩问题。
9. 默认不需要caller系数梯度：这里训练缓存上的监督heads，再做离散finite选择。仅当显式选择端到端多步router训练时，B03必须先修、测试caller/head图；本任务不因其未实现而阻塞缓存筛查。

### Tests
- 同issue候选/lead不能跨fold；future/true_gain/true_du进入ServingContext被拒绝；改变离线目标不改变已保存选择。
- 四格真的只替换指定量，并以实际outcomes结算；无效/近零响应指标N/A不伪造完美cosine。
- noedit结构零、候选顺序置换一致；默认calibration/memory/JEPA/variance penalty未参与，成本记录含preview。
- 所有方法使用同标签索引和同仿真资格；实际训练次数/seed/HPO不越cap。

### Run command
以下命令按顺序；新脚本/参数先实现，真实模型命令只有资产与对应cap获准后运行。

```bash
cd "$REPO" && python -m pytest tests/test_r2_legal_policy.py tests/test_r2_four_cells.py -q
cd "$REPO" && python scripts/round2_iteration/fit_policies.py --config "$CONFIG" --table "$OUTCOME_TABLE" --features "$LEGAL_FEATURES" --out "$RUN_DIR/models"
cd "$REPO" && python scripts/round2_iteration/score_policies.py --config "$CONFIG" --models "$RUN_DIR/models" --table "$OUTCOME_TABLE" --features "$LEGAL_FEATURES" --threshold-certificate "$CERT" --out "$RUN_DIR/scores"
```

### Expected artifacts
- `models/model_manifest.json`
- `models/fit_log.jsonl`
- `models/folds.json`
- `scores/crossfit_predictions.parquet`
- `scores/action_selection.jsonl`
- `scores/four_cell_scorecard.json`
- `scores/direct_gain_comparison.json`
- `scores/label_simulation_compute_ledger.json`
- `scores/bootstrap_intervals.json`
- `decision.json`

### Success criteria
- 可部署pred/pred在预登记指标上相对强静态达到最小价值，保护lead符合预登记条件。
- 至少一个已登记有限资源/标签/动作复用比较对分解给出继续理由；只在开发集观察到时标DEV，不称论文确认。

### Failure criteria
- oracle强但合法策略到cap仍无有用优势 -> STOP_CURRENT_PLANNER（不是不可预测定理）。
- best-static/regime解释收益 -> PIVOT_STATIC_ADAPTATION；direct-gain/端点预测解释分解收益 -> PIVOT_NARROW_CLAIM。
- CI跨阈值或资源差不可比 -> INCONCLUSIVE，不加模块挽救。

### Decision after completion
仅当LEGAL_POLICY_DEV_PASS且已有资源对齐的继续依据时申请条件性P0-05；否则以STOP/PIVOT/INCONCLUSIVE结束本迭代。每种结论都须人工复核，不自动解锁。

**PASS能推出：** 此预算/信息下有可实现收益，且候选分解有待确认的资源价值。

**FAIL能推出：** 当前合法策略/分解未证明必要；支持收窄或停止该配置而不是无限加模型。

**CI跨阈值：** INCONCLUSIVE；只允许原定追加，不用oracle补成deployable PASS。

### Suggested commit message
`feat(round2): cross-fit legal policies and test response factorization against direct gain`

不得自动push；只在用户既定版本管理授权内提交。

---

## TASK P0-05 — 条件性输出纠错挑战与本迭代决策封存

### Scientific purpose
只在合法策略已有价值后，判断真正执行参数编辑是否仍值得，而非无限容量等价性或视觉案例论证。

### Depends on
P0-04

### Files to inspect
- `earthdelta/bridge/stormer_bridge.py`
- `earthdelta/heads.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/evaluation/pilot_predictability.py`

### Files to modify
无；除非该任务步骤显式要求相邻最小修改。

### Files to create
- `earthdelta/baselines/pilot_correction.py`
- `scripts/round2_iteration/compare_corrections.py`
- `scripts/round2_iteration/finalize_decision.py`
- `tests/test_r2_correction.py`

### Implementation steps
1. 只有人工确认P0-04具备有限资源继续理由且本任务资源已授权，才实现/训练一个轻量joint multi-lead residual corrector和一个feedback版本。共享小结构与合法context，不搜大量架构、不加第二backbone。
2. virtual baseline在同一D空间直接 reference_summary+predicted_du；预测du若不含完整场，禁止宣称它是全场预报或免费逆变换。全场virtual需另有已批准decoder并计费，否则将该比较明确标SUMMARY_ONLY。
3. feedback每步只从当前自身预测状态/起报历史生成全变量增量，修正后输入下一步。冻结参考参数、保留到纠错头的梯度；不能在起报之后读真实ERA5作为新的观测。训练多步课程与共同主指标匹配，差异和额外资源完整记录。
4. 真实编辑、joint correction、feedback、virtual均计reference preview、controller、decoder、planner、rollout及fallback成本，汇总参数/内存/训练计算和相同总预算下的损失。actual noedit不是零延迟。无可比预算的结果只标INCOMPARABLE_RESOURCE。
5. finalize_decision.py 读取已保存决策/模型/阈值证书和CI，不重新调参。主lead及保护lead(起点6/24/72h)使用同样机会；120h仅已在数据和预算中预注册才增加，240h本轮不做。
6. confirm默认sealed。本轮若确需confirm一次，必须有显式allow_confirm_read授权、独立时段、已冻结候选/模型/阈值/方法选择的hash和所需功效；之后不再调整同一确认实验。否则以DEV结论结束并给出下一迭代证据请求，不伪称独立确认。
7. 若feedback/virtual在等限制下稳定支配参数执行，PIVOT_OUTPUT_CORRECTION；若仅direct-gain解释双头则PIVOT_NARROW_CLAIM。合法收益、至少一项资源/复用优势和长lead保护同时支持才CONTINUE_BOUNDED_NEXT_ITERATION。无下一轮DAG、无自动P1/P2。

### Tests
- feedback改变后续自身状态但不读取未来真值；参考权重hash不变；纠错头梯度非零。
- virtual仅摘要时不能输出FULL_FIELD标签；错误目标shape/网格拒绝。
- 报告使用实际每次运行成本、统一分母；缺confirm授权时禁止打开确认文件；CI跨阈值或未填证书不判胜。

### Run command
以下命令按顺序；新脚本/参数先实现，真实模型命令只有资产与对应cap获准后运行。

```bash
cd "$REPO" && python -m pytest tests/test_r2_correction.py -q
cd "$REPO" && python scripts/round2_iteration/compare_corrections.py --config "$CONFIG" --models "$POLICY_MODELS" --out "$RUN_DIR/corrections"
cd "$REPO" && python scripts/round2_iteration/finalize_decision.py --config "$CONFIG" --evidence "$ITERATION_EVIDENCE" --threshold-certificate "$CERT" --out "$RUN_DIR/final"
```

### Expected artifacts
- `corrections/comparison.parquet`
- `corrections/resource_frontier.json`
- `corrections/rollout_protection.json`
- `final/DECISION.md`
- `final/decision.json`
- `final/iteration_receipt.json`

### Success criteria
- 预登记比较、资源与失败分母完整；决策是CONTINUE/STOP/PIVOT/INCONCLUSIVE之一并明确所适用bank/信息/Q域。
- 仅有DEV证据则结论带DEV限定；所有后续工作保持未授权。

### Failure criteria
- 反馈/virtual同限制下支配 -> PIVOT_OUTPUT_CORRECTION；长lead保护失败 -> STOP_CURRENT_CONFIGURATION。
- 无法保证身份/数据/资源公平 -> BLOCKED；CI跨delta -> INCONCLUSIVE到cap停止。

### Decision after completion
无论结果如何，本迭代结束。CONTINUE只解锁“下一轮计划的人工讨论”，不授权下一轮实验。

**PASS能推出：** 有证据继续窄命题；不是参数编辑的一般必要性定理。

**FAIL能推出：** 当前执行方式被更低成本纠错替代或没有稳健收益，转向/停止当前声明。

**CI跨阈值：** INCONCLUSIVE；到cap结束，不能以某个精选时效宣布胜利。

### Suggested commit message
`feat(round2): challenge parameter execution and seal the next-budget decision`

不得自动push；只在用户既定版本管理授权内提交。

---

## 7. 最小取证输出与复现合同

每任务运行目录至少有`config.json`、`stdout.log`、`metrics.json`（未运行则明确NOT_RUN）、`decision.json`和`artifact_manifest.json`。manifest保存命令、真实code commit/dirty patch、父产物hash、模型/静态参考/bank/registry/D/Q/数据角色身份、随机seed、成本cap与实际使用量、测试通过/跳过/失败分母。不得将旧283条测试复制为新回归结果。

`decision.json`必须分别记录：implementation_status、measurement_status、scientific_status、governance_status。GPU缺失即使代码补丁正确仍保持测量BLOCKED。无定义阈值或无足够CI时不得写科学PASS。`allowed_next_task`默认null；`recommended_next_task`可填写建议，不具有授权含义。

### 当前三份最重要的科研证据

E1（P0-02/03）：完整候选实际结果证明当前动作域是否有余量。

E2（P0-04）：交叉拟合合法策略与direct-gain分解比较证明是否可利用。

E3（条件性P0-05）：输出反馈/virtual挑战证明是否值得真正执行参数编辑。

没有必要为报告完整性强行跑第三项；若前两项已触发停止/转向，就以完整证据结束。停止本身是合格交付，不是执行失败。

## 8. 继续／停止的最终规则

只接受固定registry与共同分母上的配对证据。收益CI下界>已登记delta_min才支持价值；上界<delta_min支持停止当前配置；跨阈值按cap保持INCONCLUSIVE。oracle PASS不代替合法policy，direct-gain不劣时不硬守双头，feedback/virtual等限制支配时不硬守参数执行。缺身份/数据/资源是BLOCKED，不等于科学否定。

保护lead用退化h=L_method-L_reference：UCI<=margin支持非劣，LCI>margin明确失败，其余INCONCLUSIVE。delta_min和margin由独立价值依据定义，MDE只是功效，禁止恢复历史固定2–3%阈值。

**本轮没有任何P1/P2节点。P0-05结束或任一STOP/PIVOT出现就停止；CONTINUE仅意味着值得申请下一轮预算。**

## 9. 文件优先级和旧材料

当前用户要求与本包科学合同 > 本迭代状态/任务 > 已冻结运行规格 > 旧审计与历史YAML。B01–B15保留原证据并按本包分类处理，不进行审计清零。原历史YAML和旧zip不覆写。

本包与此前同名旧包不同：`package_revision=round2_independent_review_next_iteration`，任务只有5项，受审点fb767f7、分支头403b55d。交付包完整性由MANIFEST_SHA256.json验证；完成工作时更新执行副本而保留原包为只读基准。
