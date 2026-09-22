# EarthDelta Codex 主执行合同 — 一个迭代

**Date：2026-09-21 · Package：EarthDelta_Codex_Execution_Package_20260921**

当前任务：**R2I-01**。`Do not proceed beyond: R2I-01`。本文件包含开始执行所需的代码定位、合同、任务、命令和决策规则；其他文件是更短入口或专题细化，不需要重新做literature review。

## 0. Mission / 最终冻结

Research verdict：**BOUNDED_CONTINUE_WITH_NARROWED_CLAIMS**。
Implementation readiness：**NOT_READY_AS_IS_FOR_VALIDATED_PILOT**。

下一笔有限预算只购买：一个正确、非退化的小型候选库，在同一真实cache上究竟有没有超越强static的、合法可预测的选择空间。仅当证据支持，才继续比较双头/投影与direct-gain，再挑战参数执行是否优于输出纠正。

保留最多两条未证假设：仿真响应/核验误差分工的有限样本或成本优势；在响应覆盖目标内的复用潜力。第二条不是本轮额外迁移实验。二次恒等式、LoRA、预算、Gram不宣称新颖性已成立。

## 1. Current ground truth

| 项目 | 固定事实或明确边界 |
|---|---|
| Repo / branch | `sisuolv/EarthDelta` / `audit/round2-review-20260921` |
| 本轮实际重新查到HEAD | `403b55db65f4c35c1a85d0794ad0de2765b07d96`；commit时间2026-09-21T09:22:56Z，直接parent为受审HEAD |
| 审计适用源码 | `fb767f7f6efbc428be39c9ad84f5905331d6e40f`，受审范围 `4fe55a7..fb767f7` |
| 已有模块 | `paired.py`端点分解；`probe.py`实际有限响应和中心差分；`geometry.py`与`teacher.py`局部几何/离线求解；`heads.py`多lead/e/du/b/H与可选JEPA；`selection.py`有限/连续选择；`lowrank.py`专家适配；bridge和数据处理 |
| 已查看CPU证据 | `codex_audit_round2/results/evidence/pytest_cpu.log`记录 **283 passed / 7 skipped / 0 failed**；上一轮审查读过，不是本包重跑 |
| 未验证 | 真实GPU/xformers独立S0、真实bank与heads训练、天气收益/延迟、用户机器被git-ignore的checkpoint/data字节。缺资源时一律BLOCKED，不复用历史数填新结果 |
| 本包本轮执行 | 只核对分支身份、写交付文件、校验交付结构/工具；不修改远端仓库、不执行研究任务 |
| 科学上真正阻塞 | 动态headroom、合法可利用性、分解增量、参数执行的有限资源价值均未知 |
| 工程上当前必防 | B01/B02/B10模型/归一化/多步身份；B05/B06实际Q；B08/B09所选数据覆盖。B03 caller梯度不是固定bank训练必需；其余按调用路径隔离 |

本地开始时复核HEAD/dirty。分支可能继续变化；不得把这个表当永远有效。未解释漂移 → BLOCKED_RECONCILIATION；禁止自动reset到审计提交。

## 2. Scientific contract — 不可静默改变

### 2.1 Reference / action

F_ref=已经冻结并登记的强静态参考Fs，不是默认原始F0。F0仅另报背景。no-edit始终允许，严格复现**Fs**。静态adapter权重与动态bank分开hash；不能用动态零初始化冒称已经训练的静态参考。

候选a包含bank identity、固定实浮点系数、层/支持、窗口、step_hours和continuation。冻结registry后不能按probe/confirm成绩改K、幅度、字典或删负收益。若joint候选未登记，不宣称测试了H组合。

前hold_steps实施edit；之后从已受edit影响的状态继续Fs，不能拼接原参考轨迹。6h为当前pilot意向；若实际数据或已有合同冲突，先记录解决，不能默改。

### 2.2 信息隔离

ServingContext只读签发时合法历史/当前状态、固定描述符及显式计费的reference preview。不得读取future truth、exact candidate output、oracle e/du/b/H。oracle/训练标签只在offline fit/score路径。

完整响应是模型仿真目标，不是地球反事实。仿真不依赖truth，但在评估起报上跑全库得到exact du只能诊断，不能给它免费部署身份。后续状态只能用自身预测，不能将它写成核验memory。

默认memory/JEPA/dynamic-rank/spectral-training off、串行无replay。retrospective initial-state假设显式登记；“6h可用”只是scenario，未提供first-seen不能称observed。

### 2.3 数学与度量

同一固定端点空间及Q≥0：

```text
e0 = truth - reference
du(a) = edited - reference
gain(a) = ||e0||_Q² - ||e0-du(a)||_Q²
        = 2 e0ᵀ Q du(a) - du(a)ᵀ Q du(a)
```

实际端点差恒等式精确；预测ehat/duhat的plug-in近似不自动无偏。仅当du(a)=Ra成立时，b=RᵀQe0、H=RᵀQR给出系数二次型。H非对角表示响应重叠，不代替非线性组合测试。固定候选直接实际执行，不需要Ra假设。

本轮主路径固定线性D；若采用变换后端点差，恒等式不需D线性，但不许将非线性D作用于差当作端点之差，或把能谱能量冒充线性坐标。

主目标按H/S/F固定一次Q_eff归约，保留B、K。诊断可按F逐lead归约，但不能偷换主指标。scale、area/lead/variable权重、sum/mean、cost penalty/ridge单位共同冻结。MSE主目标与RMSE派生指标不混阈值；RMSE先聚合平方误差后sqrt。

全e0预测不是必需；可比较响应子空间投影或条件矩。完整确定性I下条件du二阶项消失；压缩context时两个均值plug-in不一定等于条件收益。联合训练可作为明确variant，不能未经测量称物理head精确。

### 2.4 实验不可变项

不能更换benchmark协议制造收益：所有方法相同候选资格、信息、核验origins、仿真数据权限及合理HPO预算；同时报告同标签与同总成本，不假称全部资源自动相等。失败机会保留；不把candidate/grid/lead当独立N。不把已看过的dev变成confirm。不按看见的收益改主Q/时效/预算/样本。

## 3. Non-goals

不做大规模重构、第二backbone、全P1/P2、论文整套表格、重新综述；不添加JEPA、学习式memory、动态rank、谱loss、Complexity Atlas或新求解器。并发框架、通用checkpoint-replay改造、不相关cleanup暂缓。已有正常模块尽量复用，受审历史文档不必重写。

本次生成的任务中所有新增训练/评测脚本都是**Codex待实现的目标接口**，不是声称已经存在或跑通的工具。包内只有 `tools/preflight.py`、`tools/validate_package.py` 是本次交付可直接运行的工具；它们不执行天气模型。

## 4. 授权、环境与分阶段准入

### 4.1 初始运行

设置下列路径为真实绝对路径，不照抄示例占位符。PACKAGE是本ZIP解压目录，REPO是已授权本地仓库，RUN_ROOT必须是新目录。

```bash
export PACKAGE="/absolute/path/EarthDelta_Codex_Execution_Package_20260921"
export REPO="/absolute/path/EarthDelta"
export RUN_ROOT="/absolute/path/earthdelta_runs/unique_iteration1_run"
export CONFIG="$RUN_ROOT/spec/working_config.json"

python "$PACKAGE/tools/validate_package.py" --root "$PACKAGE" --verify-manifest
# RUN_ROOT已存在就停止，不覆盖旧结果。
test ! -e "$RUN_ROOT" || { echo "RUN_ROOT exists; choose a new directory"; exit 2; }
mkdir -p "$RUN_ROOT/spec" "$RUN_ROOT/R2I-01"
cp "$PACKAGE/configs/iteration1.template.json" "$CONFIG"
python "$PACKAGE/tools/preflight.py" --repo "$REPO" --out "$RUN_ROOT/R2I-01/preflight.json"
```

之后Codex填写真实路径、已有资源许可与本任务必要字段。不能运行模板的null字段；也不能为了填满config虚构数字。工具不会下载、安装、训练、自动解锁或修改仓库。

### 4.2 分阶段检验，避免过度阻塞

- R2I-01代码/CPU小测试不需要delta_min，不应因P0统计尚未定而拒绝修bug。定向测试只在已知宿主限制内，先记录cap；不默认跑全套大测试。
- 真实S0必须有可信权重、上游backend、选定真实输入、数值tol与GPU/资源授权；缺失BLOCKED。不是所有S0都必须“GPU”，但当前官方xformers路径没成功执行就不能伪装通过。
- R2I-02探索bank/cache需明确训练/仿真cap与fit/dev角色；统计价值阈值可待开发方差估计，但不做正式研究PASS。
- R2I-03正式阈值决策必须有完整证书；此前仅DESCRIPTIVE/DEV。
- R2I-04/05需独立授权和剩余预算；confirm单独授权，默认关闭。

每个新脚本在main入口检查config.authorization.allowed_task_ids及相应mode许可、资源caps；缺失返回非零并保存BLOCKED。只“前继完成”不能替代明确授权。不得因为脚本运行命令已写在本计划中就直接执行它。

任务输出status只在工作副本更新。原始package和其hash保持不变；不得自行把LOCKED改成AUTHORIZED。Suggested commit只提供本地提交建议，不授权push、改分支历史或触碰其他仓库。所有可能导入旧脚本的检查先确认没有导入时安装、自动运行或大资产反序列化；优先静态检查，不能为测mock而意外触发真实模型。

## 5. 决策DAG

```text
R2I-01 必要正确性＋已选输入准入【当前唯一授权】
  ├─缺资产/资源/独立参照 → BLOCKED；可报告CPU补丁，但不训练
  └─真实必要证据PASS → 仅产生R2I-02资格，停止等授权
       ↓ owner授权＋训练/仿真cap
R2I-02 最小非零bank＋冻结registry＋开发cache
  ├─零/不合格bank → INVALID_BANK；达到cap → STOP_CURRENT_SETUP
  └─BANK_CACHE_READY → 仅产生R2I-03资格
       ↓ owner授权
R2I-03 Oracle＋best-static/regime＋cheap legal policy【实验E1】
  ├─oracle上界不足 → STOP_CURRENT_DICTIONARY
  ├─oracle-static动态空间不足 → PIVOT_STATIC
  ├─CI跨阈值/数据不全 → INCONCLUSIVE/INCOMPLETE
  ├─oracle强、cheap弱 → 仅可申请一次独立cap的R2I-04挑战
  └─HEADROOM_PASS_ONLY＋DEV_LEGAL_SIGNAL → 申请R2I-04
       ↓ owner授权，不是自动解锁
R2I-04 同资源双头/投影/direct-gain【实验E2】
  ├─legal仍不优于static → STOP_CURRENT_PLANNER（低功效除外）
  ├─direct-gain同好更省 → DROP_FACTORIZATION/PIVOT_SIMPLE_POLICY
  ├─CI不决 → INCONCLUSIVE
  └─有限样本/成本增量有证据 → 申请R2I-05
       ↓ owner授权＋输出挑战cap
R2I-05 actual edit / virtual / feedback【实验E3】
  ├─输出同好更省 → PIVOT_OUTPUT
  ├─长时效损害越界 → NARROW_HORIZON/STOP_EDIT_FAMILY
  ├─资源/统计不足 → BLOCKED/INCONCLUSIVE
  └─价值及guard成立 → 建议下一迭代；停止，不执行P1/P2
```

**所有箭头都要求两个独立条件：证据资格 + owner明确下一任务授权。**当前停止点R2I-01；成功后也停。CI跨门槛不沿PASS箭头走；可申请的扩样/挑战不得超预登记cap。

## 6. 共同数据、统计与artifact合同

### 6.1 角色与时间

fit（必要时bank_fit/head_fit）→dev_select→dev_probe→封存confirm。真实时间/过程窗口不重叠；同起报所有candidate/lead一起分组。已用于S0/debug的数据是exposed dev。数据不足可用分组嵌套crossfit，但每个外折的HPO/static/head选择只能用内层；声称严格时间部署时使用前向时间训练。

本迭代E1/E2先用development，不消费confirm。只有最终方法/超参/static/registry/Q/阈值/成本/失败规则全部冻结、确认数据未曝光、owner授权后，R2I-05允许一次confirm；否则结束时只保留开发证据。

### 6.2 Estimand与规则

oracle=E[max g]；合法信息上限=E[max E[g|I]]；实际策略值=E[g(π(I))]。oracle-static gap不是合法dynamic gap。反例e=±1,I无信息,u=0/±0.5，oracle0.75、最佳合法0。

主比较为完整oracle vs Fs及dev静态，cheap合法策略vsdev静态，条件解锁后分解vs公平direct-gain、实际执行vs输出纠正。按起报/天气过程paired bootstrap，相同resample索引用于所有方法，记录独立块数。不能无数据先填bootstrap显著性。

阈值状态：`TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`。delta_min的价值依据独立于MDE；证书分别登记alpha/power/N/block length/variance/MDE和价值/非劣性阈值。CI[L,U]：L>δ达到注册门；U≤δ未达门；其余INCONCLUSIVE。缺阈值允许描述，禁止正式PASS/STOP。

### 6.3 失败分母与费用

注册attempted origins/candidates；reference失败和缺候选全部保留。partial表不是complete oracle；未知loss不填0。fallback只在预登记且实际执行后记账。预算包括bank、仿真、核验origin（不按K倍算）、HPO、preview、controller、重算、选中预报和fallback。低秩活跃数不是GPU时间。

### 6.4 每次运行的文件

新建、不覆盖：`config.json / metrics.json / summary.md / stdout.log / decision.json`，加本任务Expected artifacts。数据较大只记录带内容身份的相对路径索引，不把全量预测塞JSON。所有时间UTC，seed与git diff/status、环境、权重/数据身份都保存。不要输出token、key或带凭据remote URL。

`decision.json`最低字段：

```json
{
  "task_id": "R2I-01",
  "status": "BLOCKED",
  "evidence_grade": "STRUCTURAL_ONLY",
  "audited_source_head": "fb767f7f6efbc428be39c9ad84f5905331d6e40f",
  "executed_head": null,
  "config_sha256": null,
  "actual_artifacts": [],
  "metrics": null,
  "estimand": null,
  "threshold_certificate": null,
  "reason": "NOT_RUN_OR_MISSING_REQUIRED_EVIDENCE",
  "next_task_eligible": false,
  "next_task_authorized": false
}
```

这是结果schema示例，不是本包伪造的任务结果。脚本需保存真实exit code；不能用`|| true`掩盖失败。tests命令非零时先记录FAIL与日志，再停止，不继续后面的天气命令。

## 7. Detailed tasks

以下命令除包内tools外均在**对应文件实现后、任务已授权、该mode资源条件满足后**才可执行。若§4已成功生成preflight.json，则R2I-01命令表中的同一条preflight不重复执行，直接引用并核对该不可覆盖产物。各任务开始前建自己的新输出目录；测试文件不存在不等于测试通过。创建同名功能前检查仓库是否已有，已有则最小适配，不复制第二套并行实现。

## TASK ID: R2I-01 — 最小 pilot 的模型、度量与数据入口正确性

### Scientific purpose
保证下一份天气结果测量声明的参考、候选、时效与固定 Q；只解决本次入口会触发的 correctness blocker。

### Depends on
无。初始唯一允许任务；后续没有自动权限。

### Files to inspect
- `scripts/export_upstream_reference.py`
- `scripts/s0_gate.py`
- `earthdelta/bridge/stormer_bridge.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/data/pull_wb2.py`
- `earthdelta/data/make_splits.py`
- `earthdelta/contracts.py`
- `tests/test_upstream_parity.py`
- `tests/test_s0_fail_closed.py`
- `reference/stormer/inference.py`
- `reference/stormer/stormer/models/iterative_module.py`
- `codex_audit_round2/results/ROUND2_REPORT.md`
- `codex_audit_round2/results/evidence/pytest_cpu.log`

### Files to modify/create
**Modify（已有）**
- `scripts/export_upstream_reference.py`
- `scripts/s0_gate.py`
- `earthdelta/bridge/stormer_bridge.py`
- `earthdelta/metrics_contract.py`
- `tests/test_upstream_parity.py`
- `tests/test_s0_fail_closed.py`

**Create（本任务待实现，不是包内已完成研究代码）**
- `earthdelta/data/pilot_guard.py`
- `scripts/iteration1_guard.py`
- `tests/test_iteration1_guards.py`

### Implementation steps
1. 先运行包内 stdlib preflight。记录本地 branch/HEAD/dirty、与受审提交的研究源码差异。当前交付 HEAD 为 403b55d，不要求回滚到 fb767f7。存在未解释的研究源码漂移或重叠未提交修改时，只记录并停止，禁止 reset/stash/revert。
2. 从既有权重、归一化、数据元信息建立 working_config.json；复制配置模板到全新 RUN_ROOT/spec 后填写真实路径。只登记已存在资产；无资产、不可信 checkpoint 来源或缺资源授权则 BLOCKED，不自动下载/安装。CPU 定向测试不要求先有 GPU；真实 S0 单列状态。
3. 为 NormalizationContract 增加显式 diff_mean_policy；official_inference_zero_mean 使用零差分均值和官方 diff std。legacy_policy 只用于复算旧结果，不能混入新 cache。digest 绑定变量顺序、interval 键、policy、数组 shape/dtype/content。保留旧参数接口但禁止 formal 模式静默猜 policy。
4. exporter 不再导入受审 NormalizationContract。优先调用 pinned 上游 transforms/rollout；也可用独立最小实现逐句核对官方 raw→normalize→diff/std→raw accumulation，保留来源与对应行。无需强制迁入完整 Lightning 训练框架。网络仍使用真实 upstream 实现；缺 xformers/真实 backend 时返回 BLOCKED，不使用本地 SDPA 冒充。
5. exporter 与 gate 增加一致的 --config、--out；gate 增加 --upstream-reference。只验所选一个 checkpoint，默认意向 ps4。数值比较前绑定完整 checkpoint SHA、官方源码 pin、config、norm policy/digest、变量顺序、实际 lat/lon、原始输入 SHA/shape/dtype、预登记 step/lead。逐一加载并比较所有 required parity trajectories；缺文件、shape 广播、非有限值或身份不同均失败。
6. 在 metrics_contract.py 增加 full_target_loss 和 full_target_gain：e[B,H,S,F]、du[B,K,H,S,F]；Q 按 H/S/F 一次固定归约，保持 B、K；统一 scale 与 Q_eff。保留现有按 F 的诊断 API 及名字，不能把其结果当全目标。先验证实际 broadcast 后非负有限权重、每起点正分母、正有限 scale。一次有效零权重 slice 可作为全目标 mask；若独立 slice mean 无正分母则拒绝或显式 missing，不返回 NaN 假数。
7. pilot_guard 在实际选定窗口上核对 time 唯一/递增、6h 间隔、history/current/targets 实际索引、变量/level/lat/lon/单位、所读取各 channel 数值有限与写入完成凭证。只看 Zarr shape 不合格；对本次用到的每个 chunk 做可追溯读取和核验，不必扫描全库。输出 issue_id 与独立 process_group_id，不能用含 lead 的行 ID 充当独立组。
8. 无 memory/近期核验输入时明确 retrospective_initial_state_assumption；固定 6h availability 只能标 scenario，不能标 observed_first_seen。formal pilot manifest 不自动升级 legacy provenance。当前无需重做全数据拉取器；在唯一 pilot 数据入口 fail closed。
9. S0 status 区分 required_validation、runtime_status、cleanup_warning。必要步骤异常不能 PASS；仅非必要 empty_cache 失败可保留已经完成的有效数值验证但明确 warning，不能留下 status=ok 且未分类 error。报告缺数字用 N/A，不能格式化崩溃。上述是少量局部补丁，不建设通用异常框架。
10. 用 mock/stub CPU 测试真实 gate orchestration；获得人工授权与明确资源 cap 后才运行真实选定 checkpoint S0。tol 必须来自事先冻结的数值校准，缺 tol 时只输出 CALIBRATION_ONLY，不能自动 PASS。独立 real S0 不可用时，本任务保留 CPU_PATCH_PASS + real_s0=BLOCKED，后继不放行。

### Tests
- 非零假 diff_mean 在 official policy 下不影响输出；legacy 明确不同；norm digest 随 policy/变量序/interval 改变。
- 错误 checkpoint/norm/input/grid/source pin、缺注册多步文件、非有限输出和 shape 广播均不能 passed；测试调用实际 verify_upstream_parity/run_s0_gate，而非自造 dict。
- 非均匀 H/S/F 权重下 full-field loss difference 与 gain 一致；零/负权重、非法 scale 和 missing policy 行为有确定期望。
- 预建空 Zarr、缺时次/重复时次、缺历史/目标、错误 channel/单位拒绝；只扫描选定 pilot 窗口。
- 非零合成 bank + 零系数还原参考；非零→零串行调用不泄漏 hook。此时不要求 caller 系数梯度。
- 必要步骤晚期失败撤销验收；仅 cleanup 异常有明确独立状态；错误结果仍能生成 JSON/Markdown。

### Run command
逐项执行，不是全自动脚本。`--role confirm`默认禁止；包中写出命令不是授权。

```bash
python "$PACKAGE/tools/preflight.py" --repo "$REPO" --out "$RUN_ROOT/R2I-01/preflight.json"
cd "$REPO" && python -m pytest tests/test_iteration1_guards.py tests/test_upstream_parity.py tests/test_s0_fail_closed.py -m "not slow" -q > "$RUN_ROOT/R2I-01/pytest_targeted.log" 2>&1
cd "$REPO" && python scripts/iteration1_guard.py --config "$CONFIG" --out "$RUN_ROOT/R2I-01/input_guard"
cd "$REPO" && python scripts/export_upstream_reference.py --config "$CONFIG" --out "$RUN_ROOT/R2I-01/upstream"
cd "$REPO" && python scripts/s0_gate.py --config "$CONFIG" --upstream-reference "$RUN_ROOT/R2I-01/upstream" --out "$RUN_ROOT/R2I-01/s0"
```

### Expected artifacts
相对本任务输出根目录；仅列真实产生的资产，缺失保留NOT_RUN/BLOCKED。
- `preflight.json`
- `HEAD_RECONCILIATION.md`
- `asset_inventory.json`
- `metric_spec.json`
- `pilot_input_manifest.json`
- `pytest_targeted.log`
- `upstream/manifest.json`
- `s0/metrics.json`
- `s0/decision.json`
- `decision.json`
- `summary.md`

### Success criteria
- 限定源码补丁完成且相关旧回归未削弱；定向 CPU 实测日志保留 passed/skipped/failed 分母。
- 所选模型的真实独立 S0、required multistep、数据入口与实际全目标 Q 全部通过；不能以 unit test 或 hash 长度代替。
- owner 授权、资源 cap、数值容差来源和输入角色清楚；只生成下一任务放行建议，不自动继续。

### Failure criteria
- 缺 GPU/checkpoint/data/upstream/cap/tol：BLOCKED；数学或路径错：FAIL_IMPLEMENTATION。
- 源码漂移未解释、输入来源或 frozen identity 无法核对：BLOCKED_PROVENANCE。

### Decision after completion
全部必需证据 PASS → ELIGIBLE_FOR_R2I-02；仅 CPU 通过 → CPU_PATCH_PASS_REAL_S0_BLOCKED；无论何种结果，本次均停止等待明确授权。

### Suggested commit message
`fix(pilot): bind independent reference and guard metric and data inputs`

## TASK ID: R2I-02 — 冻结最小非零 bank、候选 registry 与共同仿真缓存

### Scientific purpose
支付一次最小动作空间建设成本，得到可复用的完整非线性结果，避免零初始化造成假否证。

### Depends on
R2I-01

### Files to inspect
- `earthdelta/lowrank.py`
- `earthdelta/contracts.py`
- `earthdelta/bridge/stormer_bridge.py`
- `earthdelta/probe.py`
- `earthdelta/data/pilot_guard.py`
- `plans/plans_v1_0919/v6_draft/research_spec_v6.yaml`

### Files to modify/create
**Modify（已有）**
- `earthdelta/contracts.py`

**Create（本任务待实现，不是包内已完成研究代码）**
- `scripts/iteration1_bank_cache.py`
- `earthdelta/data/pilot_cache.py`
- `tests/test_iteration1_bank_cache.py`

### Implementation steps
1. 优先查验已存在 Fs/bank，不能把 README 资产声明当文件存在。若不存在，先提交本配置下的训练/缓存资源测算与 cap，请求授权后仅训练最小 bank；禁止因训练失败自动扩 K/rank/年份/候选幅度。
2. 冻结 bank_fit/head_fit/dev_select/dev_probe/confirm 的真实时间/过程角色；bank 与候选强度只能用 bank_fit/dev_select 定义。已用于 S0/debug 的片段标 exposed/dev。所有候选、同起报各 lead 属同一组；fit/dev_probe/confirm 不交叉。
3. 复用 ExpertLoRA 与当前受控 rollout。训练 Fs 时冻结原始 backbone，只训练静态 Adapter；训练 bank 时冻结 Fs，只训练各专家 A/B，使用预登记固定非零系数和相同多步全目标 loss。固定 bank 训练不要求 caller 系数梯度；测试必须证明 loss 到 bank 非零且 backbone 不更新。
4. F0 与 Fs 分开：静态 adapter 保存为独立小权重，可在新加载的模型副本中确定性合并到 attn.proj，或以经过等价性测试的 always-on 分支执行；不能修改官方 checkpoint 文件。动态编辑结束后继续 Fs，而非回到 F0。测试 always-on 与合并路径的一致性并将 Fs hash 绑定 static adapter。
5. 候选先 no-edit + 已有合格 bank 的有界单专家计划；若要检验联合候选必须在看 dev_probe/confirm 前预登记并实际执行。K、rank、幅度、hold 和目标由批准配置确定。至少包含不同方向，不要求每个专家都在 dev 上获益；保留无效/负收益候选，不以 confirm 结果删库。
6. 新增 registry validator：唯一 ID、real floating dtype、非空、显式 no-edit、有限幅度、support、窗口、单步时间和 content hash；不能使用通用 QP 的空集 no-op 作为 registry。非法候选拒绝且记录，不裁剪不静默丢虚部。
7. 非零资格仅证明候选响应高于数值重复噪声、能够真实执行且训练预算充分；零 B/全零响应记 INVALID_BANK，不是 scientific STOP。比较训练充分 Fs 与 dev_select 选定的 best-static，记录静态训练过程与训练成本，防止弱静态人为放大 gap。
8. cache 阶段对每个合法起报从同一状态串行执行 Fs 和所有冻结候选，保存注册多 lead 轨迹或受批准的线性摘要；优先复用 cached_responses 思想，不用 singleton 导数替代联合响应。在线选择尚不运行；此处都是离线仿真。
9. 为每个 issue/candidate/lead 保存预测路径、向量响应、真实全目标 loss/gain（仅有真值的离线目标文件）、执行状态、实测前向次数/耗时/显存。参考失败仍保留机会记录；候选失败不得偷偷删除该起点。
10. 缓存写入全新 RUN_ROOT，按 reference/bank/Q/coordinates/normalization/candidate/window/continuation hash 绑定。不要缓存会变的 latent。分别输出合法 ServingContext 和 labels/outcomes，控制器 forward 不接整个 outcome table。

### Tests
- 固定 tuple 系数下 bank 有非零梯度、冻结 reference 不更新；zero-edit 等于 Fs，且 Fs 不被误当 F0。
- 静态 adapter 重构/合并等价；hold 结束后从编辑后的状态继续 Fs；不能拼回原参考轨迹。
- 未训练零 bank、complex/NaN/空 registry、重复 ID、超幅度/支持/窗口、stale hash 均拒绝。
- cache 中 e_a=e0-du 和完整 Q 的 loss difference/gain 一致；候选置换同步结果；失败行保留。
- 同 issue 不同候选/lead 不能跨 fit/dev_probe/confirm；未来标签不参与 ServingContext 构造。

### Run command
逐项执行，不是全自动脚本。`--role confirm`默认禁止；包中写出命令不是授权。

```bash
cd "$REPO" && python -m pytest tests/test_iteration1_bank_cache.py -q > "$RUN_ROOT/R2I-02/pytest.log" 2>&1
cd "$REPO" && python scripts/iteration1_bank_cache.py --config "$CONFIG" --phase inventory --out "$RUN_ROOT/R2I-02/inventory"
cd "$REPO" && python scripts/iteration1_bank_cache.py --config "$CONFIG" --phase train-or-reuse --out "$RUN_ROOT/R2I-02/bank"
cd "$REPO" && python scripts/iteration1_bank_cache.py --config "$CONFIG" --phase cache-development --bank "$RUN_ROOT/R2I-02/bank/bank_manifest.json" --out "$RUN_ROOT/R2I-02/cache"
```

### Expected artifacts
相对本任务输出根目录；仅列真实产生的资产，缺失保留NOT_RUN/BLOCKED。
- `training_budget.json`
- `bank/bank_manifest.json`
- `bank/reference_identity.json`
- `bank/candidate_registry.json`
- `bank/training_log.jsonl`
- `cache/cache_manifest.json`
- `cache/candidate_outcomes.parquet`
- `cache/serving_context/index.json`
- `cache/vector_targets/index.json`
- `cache/cost_ledger.json`
- `split_roles.json`
- `decision.json`

### Success criteria
- 一个预登记且非退化的 Fs/bank/registry 已固定；每个候选有实际运行记录。
- 开发缓存中各起报的注册候选是否完整可直接计算；失败与空目标不被包装为成功。
- 训练、仿真、标签和已知未来信息的角色分别记账；confirm 未读。

### Failure criteria
- 缺资源或资产：BLOCKED；达到训练 cap 仍无可用 bank：STOP_CURRENT_SETUP，不声称否定所有编辑。
- 冻结后改候选/参考/主指标或时间泄漏：INVALID_EVIDENCE，停止并记录需要全新预登记。

### Decision after completion
BANK_CACHE_READY → 仅建议解锁 R2I-03；不以非零响应、训练 loss 下降或缓存完成宣布科研有效。

### Suggested commit message
`feat(pilot): freeze a minimal edit bank and paired development cache`

## TASK ID: R2I-03 — 同一缓存区分 oracle 空间与合法可预测增量

### Scientific purpose
直接购买最重要的研究证据：动作库是否有空间，以及空间是否包含可由合法信息利用的部分。该任务承接旧 P0-05 和 P0-06 的轻量部分，不修改旧编号。

### Depends on
R2I-02

### Files to inspect
- `earthdelta/data/pilot_cache.py`
- `earthdelta/paired.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/selection.py`
- `codex_audit_round2/context/round1/experiments/P0_SURVIVAL_EXPERIMENTS.md`

### Files to modify/create
**Modify（已有）**
无；优先只新增窄接口。

**Create（本任务待实现，不是包内已完成研究代码）**
- `scripts/iteration1_headroom.py`
- `earthdelta/evaluation/pilot_statistics.py`
- `earthdelta/baselines/cheap_legal_policy.py`
- `tests/test_iteration1_headroom.py`

### Implementation steps
1. 先只用 dev_select 的过程级 loss/gain 方差和已测成本准备 threshold_certificate.json。delta_min 的价值依据独立于 MDE；同时冻结 alpha/power/N/block_length/主目标/长时效 guard/最大查看次数。未定字段保持指定 sentinel，允许描述性 DEV 输出，不生成正式科研 PASS/STOP。
2. no-edit 恒为 Fs；best-static 只在 dev_select 选定后冻结；regime router 的划分与每组选择仅用 fit/dev_select，默认简单低容量（例如固定元信息分组），不新增复杂 state encoder。
3. 离线 oracle 对每个 issue 从全部注册且完整的实际非线性结果中选择同一预算可行集最优；不称 top-k 或部分响应为完整 oracle。所有候选使用相同 Q、hold/continuation 和目标时间。
4. cheap direct-gain 用合法低维特征/固定已授权前缀特征＋候选描述，先 ridge 或小 MLP；按过程做交叉拟合或严格独立 dev_probe。超参及 best-static 在内层训练/选择角色确定，不能先看全部 dev_probe 再挑模型。预览晚层 reference 必须计费，不能记零额外前向。
5. 输出 Δ_PI=oracle_value-dev_selected_static_value 与 Δ_cheap=legal_policy_value-dev_selected_static_value；同时报告相对 Fs。oracle PASS 只命名 HEADROOM_PASS_ONLY；不是可部署成功。cheap 失败只否定这类已训练策略，不自动证明 Bayes 合法上限为零。
6. 按起报/天气过程共同重采样所有候选和方法，块长由开发相关结构确定；同 lead/grid 不增加独立 N。固定策略的主 CI 条件于已经训练/选择的策略；需研究训练算法方差时另行 bootstrap 重训并计成本，不把两种 CI 混用。
7. 开发阶段做可选 paired block label/context 打乱诊断：同起报所有 candidate 保持完整，不打散单列标签。仅作为无信息参照，不能机械从 oracle 扣掉 shuffled 分数，也不能称合法技能。
8. 对缺候选、reference 失败、非有限目标按预登记失败策略输出 complete_count/attempted_count；缺失表不作完整 oracle 结论。预先确定的运行 fallback 可按其真实成本/预报计入 policy；未知 loss 不填 0。
9. 本任务默认只输出 DEV_HEADROOM_PROMISING / DEV_LEGAL_SIGNAL / INCONCLUSIVE / STOP_CURRENT_SETUP 等开发结论。confirm 全程封存。若只做一次开发检验也应在输出中说明当前结论是开发证据而非新确认。

### Tests
- e=±1，I 常量，u={0,±0.5}：oracle=0.75，best legal=0，确认两个 estimand 不同。
- 含 no-edit 的完整库 oracle 非负；best-static 来自 dev_select；候选数量或顺序变化不能漏登记。
- 同过程候选/lead 重采样一致；MSE 先全局聚合再 sqrt；不能平均逐像素或逐起报 RMSE 冒充注册主指标。
- 更换 dev_probe/confirm 标签只能影响评分，不能改变已冻结 prediction/selection；同 issue 不同候选不跨 folds。
- CI 跨 delta_min 返回 INCONCLUSIVE；MDE 不能写入 delta_min；缺阈值仅 descriptive；不完整 oracle 不给 PASS。

### Run command
逐项执行，不是全自动脚本。`--role confirm`默认禁止；包中写出命令不是授权。

```bash
cd "$REPO" && python -m pytest tests/test_iteration1_headroom.py -q > "$RUN_ROOT/R2I-03/pytest.log" 2>&1
cd "$REPO" && python scripts/iteration1_headroom.py --config "$CONFIG" --cache "$RUN_ROOT/R2I-02/cache/cache_manifest.json" --mode variance-design --out "$RUN_ROOT/R2I-03/design"
cd "$REPO" && python scripts/iteration1_headroom.py --config "$CONFIG" --cache "$RUN_ROOT/R2I-02/cache/cache_manifest.json" --mode development --out "$RUN_ROOT/R2I-03/evidence"
```

### Expected artifacts
相对本任务输出根目录；仅列真实产生的资产，缺失保留NOT_RUN/BLOCKED。
- `design/threshold_certificate.json`
- `evidence/oracle_static_gaps.json`
- `evidence/cheap_policy_predictions.parquet`
- `evidence/legal_policy_value.json`
- `evidence/paired_intervals.json`
- `evidence/failure_denominator.json`
- `evidence/candidate_count_accounting.json`
- `evidence/cost_ledger.json`
- `decision.json`

### Success criteria
- HEADROOM_PASS_ONLY：完整候选库在注册目标上的 oracle gap 达到预登记门槛，仅说明有 hindsight 空间。
- DEV_LEGAL_SIGNAL：严格 out-of-sample 的低容量合法策略相对强 static 达到开发证据规则；不能声称分解优势。
- estimand、基线选择来源、分母、效应/功效、成本全部可追溯。

### Failure criteria
- oracle upper CI≤delta_min：STOP_CURRENT_DICTIONARY；其相对最佳静态增量不足：PIVOT_STATIC。
- CI 跨门槛、过程数量不足或阈值尚未冻结：INCONCLUSIVE/DESCRIPTIVE_ONLY。
- oracle 强而 cheap 无信号：LIMITED_POLICY_CLASS_NEGATIVE；仅可提出一次有独立 cap 的 R2I-04 检验请求，不自动扩模型。

### Decision after completion
oracle 与合法信号支持 → 建议 R2I-04；oracle 强但 cheap 失败 → 只有 owner 明确批准的一次 bounded challenge 可解锁；STOP/INVALID 不解锁。confirm 仍未开启。

### Suggested commit message
`feat(pilot): separate hindsight headroom from legal policy value`

## TASK ID: R2I-04 — 同资源检验分解、决策投影与 direct-gain

### Scientific purpose
判定双头是否提供有限样本或计算增量，而不是仅证明恒等式成立。

### Depends on
R2I-03

### Files to inspect
- `earthdelta/heads.py`
- `earthdelta/metrics_contract.py`
- `earthdelta/geometry.py`
- `earthdelta/data/pilot_cache.py`
- `earthdelta/baselines/cheap_legal_policy.py`

### Files to modify/create
**Modify（已有）**
- `earthdelta/heads.py`
- `earthdelta/metrics_contract.py`

**Create（本任务待实现，不是包内已完成研究代码）**
- `scripts/iteration1_factorization.py`
- `tests/test_iteration1_factorization.py`

### Implementation steps
1. 只在人工解锁与独立 cap 下执行。优先复用已生成响应缓存，不额外训练 backbone/bank。不启用 JEPA、memory、动态 rank、风险网络或谱 loss。
2. 修 B04 的共享/逐样本 program、benefit、Gram 广播；只在实际启用的路线添加测试。head 接同一完整 MetricSpec，显式返回 gain_analytic 和 gain_calibrated，默认 calibration off；variance 等正则也须独立配置。zero action 的 du/gain 是结构零。
3. 必做：A 原 e0/du regression＋analytic gain；B 决策投影/矩版本；C direct-gain；D direct-gain＋同等响应仿真辅助预训练。各用同 context、candidate descriptor、HPO 预算和核验起点资格。直接 edited-forecast 作为同 skip/residual 参数化的条件对照，仅在 cap 内，不为它故意使用更重全场网络。
4. B 可用只在 fit 响应上学出的固定线性基预测误差投影，或对有限候选学习 E[e^T Q u_a|I] 与 E[u_a^T Q u_a|I] 再组合；不得将 exact eval du 或 oracle b/H 当 ServingContext。所有固定投影都保存训练划分和 hash。没有联合候选时不声称测试了 full-H 非线性交互。
5. 执行四格：真 e/真 du、真 e/预测 du、预测 e/真 du、预测 e/预测 du，全部用同一实际 candidate outcomes 结算。真 du 需要额外候选预报，真 e 需要未来标签，二者只能诊断，不进入部署成本成绩。
6. 按唯一核验起点抽取 label subsets，不能按 candidate 随机拆分；一份 truth 可给所有已仿真 candidate 生成 scalar gain。du 仿真资格对所有方法开放。报告条件于冻结 bank 的 controller-label 曲线，同时列 bank 训练本身用过的标签并按唯一 origin 去重，不能隐藏 sunk cost。
7. 在固定标签量和固定总资源两条轴报告 policy value、regret、harmful/no-edit、响应误差和 costs。解析式与校准式分开。联合收益训练可能改变 head 的条件均值解释，不能只看最终收益就宣称每个 head 准确。
8. 保持 confirm 封存；本迭代默认不做目标迁移/新动作泛化。若只在更窄条件下胜出，在 decision.json 缩小 claim，不发起全 P1/P2。

### Tests
- 共享 program + batched Gram 正确；统一 Q_eff 下 loss difference、analytic、coefficient gain 一致。
- 四格替换确实各替换指定量；no-edit 结构零；禁用校准后参数不参与优化且输出不受其变化。
- label subset 按 origin；same-simulation direct-gain 可读合法辅助 target，仅 fit 阶段可读；serving 不能读取真实 du。
- 修改评估 truth 不影响已冻结 inference；所有方法使用同起报和失败规则。

### Run command
逐项执行，不是全自动脚本。`--role confirm`默认禁止；包中写出命令不是授权。

```bash
cd "$REPO" && python -m pytest tests/test_iteration1_factorization.py -q > "$RUN_ROOT/R2I-04/pytest.log" 2>&1
cd "$REPO" && python scripts/iteration1_factorization.py --config "$CONFIG" --cache "$RUN_ROOT/R2I-02/cache/cache_manifest.json" --role development --out "$RUN_ROOT/R2I-04/evidence"
```

### Expected artifacts
相对本任务输出根目录；仅列真实产生的资产，缺失保留NOT_RUN/BLOCKED。
- `evidence/four_cell_scorecard.json`
- `evidence/factorization_vs_direct_gain.parquet`
- `evidence/label_cost_frontier.json`
- `evidence/head_predictions/index.json`
- `evidence/training_budget.json`
- `evidence/analytic_calibrated_comparison.json`
- `decision.json`

### Success criteria
- 在预注册的 policy value/label-cost 维度报告分解相对最强合法简单基线的增量，不以 latent/response MSE 代替。
- 全 e0 无优势但投影有效：PIVOT_DECISION_RELEVANT_PROJECTION；只保留实际支持的有限样本命题。

### Failure criteria
- 达到 cap 后所有合法选择均不优于强 static：STOP_CURRENT_PLANNER；低功效仍为 INCONCLUSIVE。
- 分解相对 direct-gain 的优势 CI 上界不达注册门槛且不具成本/标签收益：DROP_FACTORIZATION_CLAIM；不自动停止更简单有效 policy。
- direct-gain 因未获同样仿真/标签而吃亏：INVALID_COMPARISON，停止并修资格。

### Decision after completion
有合法价值且分解/投影仍有保留理由 → 建议 R2I-05；只有简单策略有效 → PIVOT_SIMPLE_POLICY，通常结束本迭代；无证据不得增 JEPA 等。

### Suggested commit message
`feat(pilot): challenge paired prediction with matched direct-gain baselines`

## TASK ID: R2I-05 — 参数执行对输出纠正的最小挑战与迭代裁决

### Scientific purpose
在有限信息和成本下检验实际参数执行的价值，同时完成本迭代 STOP/CONTINUE/PIVOT 裁决。

### Depends on
R2I-04

### Files to inspect
- `earthdelta/bridge/stormer_bridge.py`
- `earthdelta/heads.py`
- `earthdelta/data/pilot_cache.py`
- `earthdelta/metrics_contract.py`

### Files to modify/create
**Modify（已有）**
无；优先只新增窄接口。

**Create（本任务待实现，不是包内已完成研究代码）**
- `earthdelta/baselines/pilot_output_correction.py`
- `scripts/iteration1_output_challenge.py`
- `tests/test_iteration1_output_challenge.py`

### Implementation steps
1. 必须有明确独立授权和输出训练 cap。复用同一 Fs、history、候选、主目标/guard，不加新 backbone 或扩大数据。若输出 full-field 数据不足或预算不足，标 BLOCKED_OUTPUT_COMPARISON，不宣布参数路线胜出。
2. 实现轻量多变量 feedback corrector：在每一步根据同合法 context 和当前自回归状态输出有单位约定的 Δstate，执行 x_next=Fs(x)+C(context,x,step)。先 zero-init 保持参考；在相同 hold 窗口内施加，之后从修正后状态继续 Fs。训练真值只在 loss；起报后不再读真实状态。
3. Virtual baseline 对相同选中 action 预测 full-field du 并加 reference；如当前 head 仅输出摘要，给合理、记录成本的共同 full-field readout，或明确 NOT_COMPARABLE_SUMMARY_ONLY。不得将小摘要直接与真实全场编辑比。reference+predicted-error 可共用相同读出作额外简约对照。
4. 比较 actual parameter execution、virtual response correction、feedback correction 的同参数机制结果和真实资源前沿；不强求两个轴同时精确相等。计入 reference preview、controller、必要重算、selected rollout，以及累计训练仿真成本。不能引用无限容量等价性代替测量。
5. 主结果沿用预登记 Q 与主时效；guard 至少覆盖已在本迭代冻结的长期点，不临时改成最有利 lead。若想增加 120h 等，只能作为另列经批准诊断，不能事后替换主指标。
6. 生成 iteration_decision.md，联合 R2I-03/04/05 判断有界继续、缩窄、停止或证据不足。当前没有 paper-level 结论保证；不自动解锁下一迭代。
7. confirm 默认为禁止。仅当方法/超参/static/registry/Q/阈值/样本/失败规则/成本口径全部冻结、确认集从未读取且 owner 明确批准时，允许 --role confirm；同一次运行评分所有冻结方法并做登记的多比较控制。否则仅报告 DEV，输出待确认合同，不擅自消耗 confirm。

### Tests
- C=0 还原 Fs；早期 C 的影响进入后续预测，hold 后不会拼回原轨迹；所有变量与单位一致。
- future-truth poisoning 不改变 frozen inference；反馈路径后续只读自己的状态。
- virtual/actual 输出分别保存；摘要无法比较时拒绝获胜结论；成本不遗漏共同前缀/preview。
- confirmation ledger 非空、方法 hash 漂移或资源未授权时拒绝 confirm；失败分母与原候选表一致。

### Run command
逐项执行，不是全自动脚本。`--role confirm`默认禁止；包中写出命令不是授权。

```bash
cd "$REPO" && python -m pytest tests/test_iteration1_output_challenge.py -q > "$RUN_ROOT/R2I-05/pytest.log" 2>&1
cd "$REPO" && python scripts/iteration1_output_challenge.py --config "$CONFIG" --cache "$RUN_ROOT/R2I-02/cache/cache_manifest.json" --role development --out "$RUN_ROOT/R2I-05/evidence"
cd "$REPO" && python scripts/iteration1_output_challenge.py --config "$CONFIG" --cache "$RUN_ROOT/R2I-02/cache/cache_manifest.json" --role confirm --confirmation-spec "$CONFIRM_SPEC" --out "$RUN_ROOT/R2I-05/confirm"
```

### Expected artifacts
相对本任务输出根目录；仅列真实产生的资产，缺失保留NOT_RUN/BLOCKED。
- `evidence/correction_frontier.json`
- `evidence/lead_variable_metrics.parquet`
- `evidence/rollout_guard.json`
- `evidence/information_cost_ledger.json`
- `frozen_final_protocol.json`
- `confirmation_ledger.json`
- `iteration_decision.md`
- `decision.json`

### Success criteria
- 有限资源下实际参数执行具有注册目标/guard/cost 上的保留理由；或有据地转向更简单输出路线。
- 最终决定引用真实 artifacts，DEV/CONFIRMED/BLOCKED 明确，不要求所有方法全指标支配。

### Failure criteria
- 输出路径在注册等效/价值门槛内同样好且更便宜：PIVOT_OUTPUT_CORRECTION / DROP_PARAMETER_EXECUTION_NECESSITY。
- long-lead 损伤超过非劣性容忍：NARROW_HORIZON 或 STOP_EDIT_FAMILY；CI 跨界仍 INCONCLUSIVE。
- 资源/公平输出空间不足：BLOCKED，不反向当参数路径有效；确认泄漏：INVALID_CONFIRMATION。

### Decision after completion
结束本迭代；最多提出下一笔预算的证据请求。不启动新的 P1/P2、文献综述或自动长期运行。

### Suggested commit message
`feat(pilot): compare actual edits with feedback output correction`

## 8. Iteration end

不要求本次调用执行全部五任务。只完成当前已授权的R2I-01即停止；有真实后继授权再恢复。最后一项完成只提出下一迭代建议，不发起P1/P2。

若工程有限修复后无GPU/数据，交付最小patch、定向CPU结果与BLOCKED清单已经是合法产物，不填weather增益。若真实实验否定当前假设，保存失败和完整分母，按STOP_CONDITIONS缩窄或结束；不得通过模块堆叠逃避。

## 9. Source/evidence索引

受审源码与第二轮报告路径在IMPLEMENTATION_AUDIT_ACTIONS.md；阶段合同来自仓库`codex_audit_round2/context/round1/experiments/P0_SURVIVAL_EXPERIMENTS.md`，本包只压缩其下一迭代子集。283/7/0是随附日志，不是本轮复跑。branch HEAD本轮只读重查；详细来源见provenance.json。若文件与本包冲突，优先本次冻结科学合同、实际源码/运行证据，不继承历史宣传或自述。
