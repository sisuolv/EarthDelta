# EarthDelta Codex 主执行计划

**Package ID：EarthDelta-R2-NextIteration-20260921**  
**范围：未来一个迭代，5 个任务、至多 3 个取证实验。**  
**初始只允许：R2-P0-01 的最小源码修复与合成 CPU 测试；到此停止。**

## 0. Mission 与优先级

在可信冻结参考与小编辑库上，检验 hindsight 修正空间中是否有合法信息可利用的收益，以及 e0/du 分解是否带来有限标签/计算优势。本包沿用刚完成的独立复核，不重新做文献综述，不规划整个项目。

执行优先级：本次用户授权范围与本主计划的锁 > 本包正式科学/实验合同 > 工作配置与状态 > 旧计划/审计报告。发生同级矛盾时停止记录，不自行挑有利规则。历史 `plans/`、AUDIT_BRIEF、packet 中的任务/建议是背景证据，不自动授权执行。保留这些文件，不大规模重写。

## 1. Current ground truth

- 目标仓库：`sisuolv/EarthDelta`；分支：`audit/round2-review-20260921`。
- 本次重新读取远程分支 HEAD：`403b55db65f4c35c1a85d0794ad0de2765b07d96`。
- 上轮受审源码 HEAD：`fb767f7f6efbc428be39c9ad84f5905331d6e40f`；范围 `4fe55a7..fb767f7`。
- 两个 HEAD 的全部原根条目 Git blob/tree 对象一致；新 HEAD 仅新增 `codex_audit_round2`。因此不是“发现新修复后继续套旧结论”。`earthdelta/`=`867ea7c5ad43ffd7850cda3b6627287032a4db8f`，`scripts/`=`17893b917ce06ffd7fed2655e15bb0a129268320`，`tests/`=`e3c5d9d8bd207b492554487e7d9b0bd669ab4193`。
- 已有模块：contracts、paired、probe、geometry、teacher、selection、heads、memory、spectral、lowrank、metrics_contract、Stormer bridge、数据拉取与日历 split、S0/exporter，以及对应 CPU 测试。不是一个已训练完毕的 e0/du 天气系统。
- 亲读的既有 CPU 日志：290 collected，283 passed、7 skipped、0 failed；7 skipped=3 checkpoint 内存+2 CUDA+2 upstream reference。来源 `codex_audit_round2/results/evidence/pytest_cpu.log`，blob=`e8532a8c82356b11b6eefb38e612bcac9e71c0dc`。**本次制作包没有复跑该仓库测试。**
- GPU/xformers 真 S0、当前机器的 checkpoint/data、完整真实样本有效性、bank 训练、oracle/weather gain、合法 policy 和纠错结果：没有本次已验证的结果；运行层保持 NOT_RUN/BLOCKED，不能复制历史 README 数字当成绩。
- 私有连接器实际读到上述内容；GitHub URL 不是本地路径。先用本地 git 查真实工作树；不假设 `/mnt/afs/...` 存在。

当前真正阻塞收益取证的是：B01 官方完整归一化/rollout 一致性；B02 身份、输入与多步绑定；B10 所用 checkpoint 贯通；实际评分 Q 和非法输入；实际子集内容与 history/target 联结。不是 B01–B15 全部清零。

定位依据：受审 `scripts/export_upstream_reference.py:106–110,230–251`、`scripts/s0_gate.py:175–300,609–692`、`earthdelta/bridge/stormer_bridge.py:229–251,742–750`、`earthdelta/metrics_contract.py:96–195,226–274`、`earthdelta/data/pull_wb2.py:425–478,668–783`、`earthdelta/data/make_splits.py:300–388`。行号针对 fb767f7；本地修改后用符号重新定位，不机械套行号。

可核验来源：
- 分支：https://api.github.com/repos/sisuolv/EarthDelta/branches?per_page=100
- 受审树：https://api.github.com/repos/sisuolv/EarthDelta/git/trees/fb767f7f6efbc428be39c9ad84f5905331d6e40f
- 交付树：https://api.github.com/repos/sisuolv/EarthDelta/git/trees/403b55db65f4c35c1a85d0794ad0de2765b07d96
- 日志：https://github.com/sisuolv/EarthDelta/blob/403b55db65f4c35c1a85d0794ad0de2765b07d96/codex_audit_round2/results/evidence/pytest_cpu.log

## 2. Scientific contract：实现过程中不得偷偷改变

### 2.1 冻结的对象与合法输入

F_ref 是选定 checkpoint、显式 normalization policy、变量/坐标/单位、可选静态 Fs、rollout interval 与 continuation 的组合，不只是网络名称。训练好 Fs 与 bank 后、任何 oracle 比较之前冻结其身份。a 包含专家系数、作用窗口、幅度和支持约束；窗口结束从已经改变的状态继续 reference dynamics，不回到未编辑参考轨迹。

serving 输入 I 只含注册的当前/历史研究输入与当时允许的模型计算结果；不能加载 future truth、e0 标签、真实候选 du、候选实际收益或 oracle 动作。retrospective ERA5 初始化只是明确的 perfect-analysis 研究条件，不因此宣称运营实时合法。memory 在本迭代关掉，6h 可用延迟只是 scenario；不以此启用近期已核验误差。

oracle/training labels 与 serving 文件、CLI 参数及函数图隔离；predict 阶段先保存动作/预测 hash，独立 score 阶段再载入 truth/outcomes。签名无 truth 不等于已经隔离；要做实际 predict poisoning/文件访问测试。

### 2.2 e0、du、gain、Q 与分数精度

原生注册预报目标 `Y,Fref,Fa:[B,H,V,Lat,Lon]`，候选多一维 K。定义 `e0=Y−Fref`，`du(a)=Fa−Fref`，正 gain 表示损失降低。

使用同一 fit-only positive scale s 与非负 q；在注册 H/V/Lat/Lon 上一次归一化：
`L(Y,F)=Σ q*((Y−F)/s)^2 / Σq`，`Q_eff=diag(q/s^2)/Σq`。
`g(a)=L(Y,Fref)−L(Y,Fa)=2 e0ᵀ Q_eff du−duᵀ Q_eff du`。

missing policy 固定，分母及坐标对所有候选一致。所有评分先保留真值精度再 float64 累加；不要先把 truth 转 BF16。RMSE 对主目标先聚合平方误差再开根号；逐变量物理单位 RMSE 另外报告。不能以平均逐样本 RMSE 替换总体 RMSE。

逐 F/逐 lead 诊断 reducer 允许保留，但不能自动代表全目标。sum/mean 都可使用，前提是 b/H 按同一 Q_eff 构造、ridge/penalty 单位一致。本迭代主路径是有限候选；不需要扩张连续优化。

若只对学习目标压缩线性 D，定义 `e_D=D(Y)−D(Fref)`、`du_D=D(Fa)−D(Fref)` 和 Q_D。它的端点恒等式只保证该 summary 损失；**主决策结算始终是完整原生注册目标**。保存 D/hash、summary/full-field 排序差，不称 gain_D 自动等于 gain_native。未注册的变量无需全部进入 headline，但注册 guard 不得事后删除。

预测 e0/du 代入只叫 `gain_analytic_surrogate`；默认关闭可训练校准。返回 analytic 和 calibrated 分数时分别命名与版本记录，不能用额外 scalar head 隐藏双头失败。H 只在线性/局部响应假设下解释为响应重叠；真实非线性组合需逐候选实际模拟。

### 2.3 候选、oracle、policy 与成本

registry 必须非空、实浮点、有显式 no-edit，ID/系数/窗口/成本定义唯一且 frozen。已有四项 finite/bound/max_active/max_candidates 修复保留；正式 registry 不靠通用 planner 的空表 fallback 冒充 no-edit。

`V_H=E[max_a g]` 是 hindsight；`V_I=E[max_a E[g|I]]` 是合法信息下的理想上限。oracle PASS 只说明值得尝试预测，不证明 deployable policy、分解优势或真实编辑必要性。

默认主指标为物理收益，资源通过共同可行域和单列成本前沿处理；不临时从gain扣任意lambda成本。若要净utility，lambda与单位须预注册。no-edit 的物理gain为0，但参考前向、controller/preview与回退并非零成本。oracle实际全库模拟成本单独记offline evidence成本，不伪装成可部署一次推理成本。

### 2.4 数据、统计与反偷换规则

fit/dev/confirm 在起报/天气过程级固定；同issue所有candidate/lead同折。history 与 target 从真实 time 坐标联结。完整性来自实际读取内容，不能来自 Zarr shape 或 marker 文件存在。小子集可逐批核验，不要求先扫完所有年份。

best-static 在fit/dev选定；confirm只应用，不重选。所有候选共用注册机会分母，运行失败记录，不只报告成功子集。缺候选→incomplete oracle，未知结果不填0假装完整。

按起报/天气过程配对 block bootstrap；不得把网格/候选/高度层当独立N。候选数增多导致hindsight增益变大是估计目标性质，不随意扣除；预注册子集与打乱仅作诊断。

价值阈值 δ_min 与给定 alpha/power/N 下 MDE 分开登记。所有未定数值写 `TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`；该文字是缺项标记，不意味着 δ_min 应由方差计算。价值阈值要有独立的研究/资源价值依据，confirm前冻结。下CI>δ 才支持超过最低价值；上CI<δ 排除该最低价值；跨阈值 INCONCLUSIVE。

默认先dev缓存与crossfit，最终confirm保持封闭；如果E1单独消耗confirm，后续方法选择不能看结果再复用同集，必须另留独立confirm或事先冻结全部后续策略与联合分析。探索性 DEV_PROMISING 可以由reviewer批准小额下一实验，不能写成confirm PASS。

不得通过改年份、筛异常天气、丢失败、换主指标、扩候选或重调阈值制造收益。更换问题必须新run identity与新确认协议。

## 3. Non-goals 与最小补丁边界

不做：重跑 literature review；重写全仓库；新 backbone；JEPA/memory/dynamic-rank/spectral/Complexity Atlas；额外 representation 世界模型；全量 P1/P2；多领域泛化；自动下载大数据/权重；无限调参；工业并发/事务发布框架；未授权GPU；自动push或撤销用户改动。

B03 的 caller coefficient 图仅在后续端到端controller训练前修；当前固定系数bank与有限动作routing可以不依赖它。B04 仅在R2-P0-03确实使用batched Gram时修。B12以串行、无共享模型、无activation-checkpoint replay合同限制。B13/B14可随同一gate小修，不扩大成独立工程项目。

## 4. Execution DAG 与锁

```text
R2-P0-01 可信执行路径（当前唯一授权：源码+合成CPU）
    ├─ 缺GPU/资产/权限 → BLOCKED；只报告已完成CPU修复，停止
    ├─ correctness失败 → INVALID_IMPLEMENTATION；修复同任务，停止
    └─ 真实S0 PASS + reviewer授权 + cap
          ↓
R2-P0-02 小bank + 完整候选缓存 + E1 oracle/static
    ├─ oracle不足 → STOP_CURRENT_BANK
    ├─ static解释主要可用空间 → PIVOT_STATIC
    ├─ CI跨阈值/缺阈值 → INCONCLUSIVE/DEV_DESCRIPTIVE；不自动扩大
    └─ HEADROOM_PASS 或明确审阅的 DEV_PROMISING + 新授权
          ↓
R2-P0-03 E2 合法策略/四格/direct-gain/标签效率
    ├─ 合法收益被排除 → STOP_CURRENT_PLANNER
    ├─ 仅direct有效 → PIVOT_DIRECT_GAIN/缩窄claim，单独审阅是否做E3
    ├─ CI不足 → INCONCLUSIVE；不增加新模块
    └─ 可信合法信号 + E3资源授权
          ↓
R2-P0-04 E3 actual / virtual / output / feedback
    ├─ 输出支配 → PIVOT_OUTPUT；virtual同效更便宜→缩窄代理主张
    ├─ CI不足 → INCONCLUSIVE
    └─ 注册资源区间内保留优势 → 可建议继续
          ↓
R2-P0-05 只读证据汇总 → 迭代结束；不自动开下一迭代
```

每个节点终止时自行保存 `decision.json`；不必为了生成该小记录而解锁 R2-P0-05。R2-P0-05 也可在前面某个终止点由reviewer单独授权，只读汇总未到达任务为 LOCKED_UNREACHED。

**证据可放行与授权可执行是两个条件。** `depends_on` 满足或测试PASS从不自动改 approved_task_ids。初始 current_task/do_not_proceed_beyond 均为 R2-P0-01。

## 5. 状态、授权与运行产物

状态分三个轴，不用一个“PASS”掩盖限制：
- code_status：NOT_STARTED / IMPLEMENTED / CPU_VALIDATED / FAILED。
- run_status：NOT_RUN / BLOCKED / DEV_COMPLETE / CONFIRM_COMPLETE / COMPLETE / FAILED。
- research_verdict：NOT_EVALUATED / HEADROOM_PROMISING / POLICY_SIGNAL / CONTINUE_BOUNDED / PIVOT_STATIC / PIVOT_DIRECT_GAIN / PIVOT_OUTPUT / STOP_CURRENT_BANK / STOP_CURRENT_PLANNER / INCONCLUSIVE / INVALID。

task status 单独为 READY / LOCKED / IN_PROGRESS / BLOCKED / COMPLETED / STOPPED。没有真实资产执行的CPU mock只能支持code_status。

每个run目录必须新建且不可覆盖；最少保存 config.json、command.txt、git_identity.json、stdout.log、metrics.json（未跑时metrics值null）、decision.json。大结果用parquet/npz；若现环境缺依赖，记录BLOCKED或使用预注册等价序列化，不自动pip。

decision.json 必须包含 task_id、package_id、run_id、code_status、run_status、research_verdict、evidence_scope、artifact_paths/hashes、threshold_certificate_hash（无则null）、missing_requirements、recommended_next_task、authorization_required=true。没有判定资格时不得填正的science_pass。

授权文件列批准任务与操作，初始只允许repo读取、最小code patch和合成CPU tests。任何GPU、checkpoint反序列化、真实数据读取、profile、训练、模拟、confirm访问需要reviewer明确批准且runtime cap非空。Codex不得自行把 false 改 true；只记录新用户/负责人批准的凭据引用，不伪造批准者。下载、push不因GPU授权而获得许可。

cap按真实小profile测得的每步/每轨迹成本制定；限制训练更新数、起报数、候选数、GPU/存储总量和重试次数。重试也计费；无资源字段→相应执行BLOCKED，但不阻止当前已授权的源码修复。不要替用户指定硬件小时或收益百分比。

## 6. 开始执行：只凭本主计划与仓库即可定位工作

从实际本地仓库开始，不从URL推测路径。解压包和运行控制副本须分开保留，包内校验和只校验原始交付文件。

```bash
# 在实际本地 EarthDelta 工作树中执行；不要自动 clone/checkout/reset。
set -euo pipefail
export REPO="$(git rev-parse --show-toplevel)"
export PACKAGE="/实际解压路径/EarthDelta_Codex_Execution_Package_20260921"
export RUN_ROOT="$REPO/artifacts/round2_next"
mkdir -p "$RUN_ROOT"
export RUN="$RUN_ROOT/$(date -u +%Y%m%dT%H%M%SZ)_r2p001"
mkdir "$RUN"                    # 已存在就失败；不要改成覆盖
export CONFIG="$RUN/pilot_config.json"
export AUTH="$RUN/authorization.json"
export STATE="$RUN/codex_tasks.json"
cp "$PACKAGE/configs/pilot_config.template.json" "$CONFIG"
cp "$PACKAGE/configs/authorization.template.json" "$AUTH"
cp "$PACKAGE/codex_tasks.json" "$STATE"
python3 "$PACKAGE/tools/validate_package.py" --package "$PACKAGE"
python3 "$PACKAGE/tools/repo_preflight.py" --repo "$REPO" --out "$RUN/preflight.json"
```

路径占位符只需指向实际解压目录；统计阈值、模型/数据路径与cap是有意未定的科学/资源字段，不可猜填。首次preflight若本地不是观察HEAD或关键对象有变化，先记录差异；源码drift未经复核，不套用旧测试成绩。纯新增本轮计划/运行目录可记录后继续；不撤销用户工作。

**下文命令是各TASK必须实现并测试的目标CLI。** 新 `scripts/r2_*.py` 目前不声称已存在；先实现再运行 `--help`/CPU测试。原 exporter/S0 的新参数也须由 R2-P0-01 添加。每个CLI读取 task-state/authorization 并验证task和操作；predict与score采用分离的输入合同。

后续任务的 CONFIG 必须是经reviewer冻结/批准的工作副本，不能从默认空模板直接开始训练。`CACHE_MANIFEST/LEGAL_FEATURES/MODEL_MANIFEST/REGISTRY/OUTCOMES` 只从已生成且校验过的artifact index解析，不靠猜文件名。


## 7. Detailed Codex Tasks

## TASK R2-P0-01 — 最小可信执行路径：官方参考、身份、Q 与小样本准入

### Scientific purpose
保证后面的收益属于正确的预报系统和同一目标；花最少补丁修复测量路径，而非清空全部 finding。

### Depends on
无；但先完成本地source reconciliation。

旧任务映射：旧 P0-02/03 残留的必要子集；B01/B02/B05/B06/B10；B07/B08/B09 仅实际入口；B13/B14 同文件小修。

### Files to inspect
- earthdelta/bridge/stormer_bridge.py
- earthdelta/metrics_contract.py
- earthdelta/heads.py
- earthdelta/contracts.py
- scripts/export_upstream_reference.py
- scripts/s0_gate.py
- earthdelta/data/make_splits.py
- earthdelta/data/pull_wb2.py
- reference/stormer/inference.py
- reference/stormer/stormer/models/iterative_module.py
- tests/test_s0_fail_closed.py
- tests/test_upstream_parity.py
- tests/test_differentiable_rollout.py
- codex_audit_round2/results/evidence/pytest_cpu.log

### Files to modify/create
**Modify（只改本任务必要符号）**
- earthdelta/bridge/stormer_bridge.py
- earthdelta/metrics_contract.py
- scripts/export_upstream_reference.py
- scripts/s0_gate.py

**Create（先查是否已有等价实现）**
- earthdelta/pilot_contract.py
- scripts/r2_pilot_preflight.py
- tests/test_r2_pilot_contract.py
- tests/test_r2_s0_binding.py

### Implementation steps
1. 先运行包内只读 repo_preflight，记录本地 HEAD、工作树、关键源码对象；403b55d 与 fb767f7 的原条目对象一致是已核验远程事实，不免除本地 drift 检查。不得 reset、切换用户分支、覆盖 dirty 文件或自动 git push。
2. 只修复该 pilot 实際路径。NormalizationContract 新增显式 official_zero_diff_mean policy；新正式入口必须显式指定，旧非零均值接口保留 legacy 标识和兼容回归。policy 必须进入 digest。不要凭“均值小”跳过协议一致性。
3. exporter 在独立进程/命名空间导入 pinned 官方网络与 GlobalForecastIterativeModule.forward_validation，独立从官方变量顺序及 NPZ 构造 transforms，增量均值为零，不再使用受审 NormalizationContract。若官方模块依赖缺失就 BLOCKED；不能伪装导入或默认换 SDPA。
4. exporter 和 s0_gate 接受同一 --config、--authorization、--task-state；显式 checkpoint 路径、patch size 和 upstream-reference 目录。只验证选择的一个 checkpoint；ps4 是历史主规格，不得借 ps2 成绩放行 ps4。旧 CLI 可保留，但不能生成本包的新准入证书。
5. 数值比较前绑定 expected checkpoint SHA256 与实际字节、官方 commit/配置、norm policy/digest、变量和单位、真实 lat/lon、raw input 与 normalized input hash、exact shape、dtype、interval 和 rollout steps。验证预注册全部轨迹，至少覆盖一步和本 pilot 的多步路径；不是只载入 6h_1step。输入 hash 必须是实际参与前向的同一内容。
6. 在 metrics_contract.py 增加 full_objective_loss / full_objective_gain：输入原生注册目标 [B,H,V,Lat,Lon]；先在 float64 中形成误差及增量，用相同 positive scale 与在 H/V/Lat/Lon 上一次归一化的 q 结算，输出 [B] 或 [B,K]。显式检查 real floating、finite、广播形状、有效总分母和 missing policy。保留旧逐 F 诊断 reducer，不冒充全目标。禁止统一用 max(eps,sum_w) 掩盖空目标。
7. pilot_contract.py 提供 require_stage_authorization、validate_metric_and_registry_header、validate_slice_index、validate_loaded_sample、assert_artifact_binding；正式路径必须调用，不只添加无人使用 dataclass。R2-P0-01 只需 S0 小输入的检查能力；真实数据扫描须另获授权。后续 loader 每次读取都验证与该契约一致。
8. 六小时核验延迟只能标 scenario；本迭代 memory 输入为零且禁止已核验近期误差进入 serving。retrospective 初始化必须标明 perfect-analysis 研究设定，不宣称真实业务可得。不要为这一限制重写历史 manifest。
9. 用已知非零合成 bank 检查 zero→nonzero→zero、串行 hook cleanup 与 loss→bank 梯度，冻结主干无梯度。caller coeff 梯度不列为本任务 bank 准入；不启用共享模型并发或 activation checkpoint replay。
10. 主 gate 仅在必需检验完成且结果/身份产物成功写出后提交 PASS；未知异常回滚为失败。已知纯清理警告可单列 execution_status，但不得保留含未分类 error 的 PASS。顺手修安全报告格式化，不建立新报告框架。
11. 当前只执行实现和合成 CPU 验证。运行回归前检查test收集/fixture，忽略任何仍会读取真实模型或天气资产的测试并记录理由；CUDA_VISIBLE_DEVICES为空，upstream测试另行排除，不能用默认not-slow冒充纯合成隔离。没有明确 GPU/checkpoint-load/data-read 授权与 cap，真实 exporter/S0 不运行，code_status=CPU_VALIDATED、run_status=BLOCKED；提供准确缺项与可运行命令后停止。

### Tests
- test_official_policy_ignores_nonzero_diff_mean：独立零均值真值计算；legacy 数值不被静默改写。
- test_gate_rejects_identity_mismatch：checkpoint/norm/variable/grid/raw input/shape 任一错、缺任一 multistep，驱动真实 gate orchestration 均不通过。
- test_full_q_gain_matches_endpoint_losses：非均匀 lead/area/variable q；gain=loss_ref−loss_edit；若由 R 建 b/H，两者用同一 Q_eff 并一致。
- test_invalid_metric_inputs：负权重、非有限、全零有效目标、零/负 scale、错误广播拒绝；合法零权重 mask 不生成 NaN。
- test_slice_contract：缺 history/target 时间、坐标顺序变更、空/NaN 数据、旧 marker 指向新 store 均拒绝。
- test_bank_gradient_and_serial_cleanup：冻结网络、非零常系数，bank 梯度非零；不声称 caller 系数图已修复。
- test_orchestration_late_error_and_report_missing_value：真正调用 orchestrator/report，不用测试自造 dict 替代。

### Run command
```bash
python3 "$PACKAGE/tools/repo_preflight.py" --repo "$REPO" --out "$RUN/preflight.json"
cd "$REPO" && CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/test_r2_pilot_contract.py tests/test_r2_s0_binding.py -q
cd "$REPO" && CUDA_VISIBLE_DEVICES="" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 -m pytest tests/ -m "not slow" --ignore=tests/test_upstream_parity.py -q -rs
# 下列是实现后的接口，仅有匹配授权/cap/资产才运行；不是本包已存在的科学结果。
cd "$REPO" && python3 scripts/r2_pilot_preflight.py --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --stage s0 --out "$RUN/input_guard"
cd "$REPO" && python3 scripts/export_upstream_reference.py --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --out "$RUN/upstream"
cd "$REPO" && python3 scripts/s0_gate.py --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --upstream-reference "$RUN/upstream" --out "$RUN/s0"
```

### Expected artifacts
- preflight.json
- source_reconciliation.md
- pytest_cpu.log
- contract_checks.json
- input_guard/manifest.json
- upstream/manifest.json（真实执行才有）
- s0/validation_certificate.json（真实执行才有）
- decision.json

### Success criteria
- 必要源码合同与反例通过；保留旧测试，实际失败/跳过原因完整记录。
- TASK 全部 PASS 还必须有同一配置/资产的真实独立 S0 证书；CPU pass 或 mock 证书绝不替代。

### Failure criteria
- 确定的数值/身份/数据合同错误 → INVALID_IMPLEMENTATION。
- GPU、依赖、checkpoint、输入或权限缺失 → BLOCKED，不下载或伪造。

### Decision after completion
任何状态都先写本任务 decision.json 并停止。仅 CPU_VALIDATED 不解锁 R2-P0-02；真实 S0 PASS 也只可提出解锁建议，须有 reviewer 明确授权和资源 cap。

**PASS能推出：** 选定系统可可信测量，不代表编辑有益。

**FAIL能推出：** 实现/资产不合格，非科学否证。

**CI跨阈值：** 不适用收益CI；parity阈值未定则 BLOCKED，不调宽tol迁就结果。

### Suggested commit message
`fix(r2): bind minimal pilot to independent Stormer reference and metric contract`

提交仅限已授权且不夹带用户改动的本地commit；没有自动push授权。

---

## TASK R2-P0-02 — 最小 bank、完整候选缓存与 oracle/static 生存证据

### Scientific purpose
用最小真实库回答值得预测的编辑空间是否存在，保留 oracle 与合法技能的区别。

### Depends on
R2-P0-01

旧任务映射：对应旧 P0-04/05；不重排其后的完整研究 DAG。

### Files to inspect
- earthdelta/lowrank.py
- earthdelta/bridge/stormer_bridge.py
- earthdelta/contracts.py
- earthdelta/probe.py
- earthdelta/paired.py
- earthdelta/data/make_splits.py
- earthdelta/data/pull_wb2.py
- earthdelta/pilot_contract.py
- codex_audit_round2/context/round1/experiments/P0_SURVIVAL_EXPERIMENTS.md

### Files to modify/create
**Modify（只改本任务必要符号）**
无；优先调用已有接口。

**Create（先查是否已有等价实现）**
- earthdelta/data/pilot_index.py
- earthdelta/evaluation/r2_survival.py
- scripts/r2_bank_oracle.py
- tests/test_r2_pilot_index.py
- tests/test_r2_registry_and_oracle.py
- tests/test_r2_block_statistics.py

### Implementation steps
1. 在实现前复用同名/等价已有模块；若无才新增上述小文件。接收同一 CONFIG/AUTH/STATE，验证 predecessor S0 与新实际模型/数据一致；S0 后任何相关源码/权重/policy 改变都重做受影响证书。
2. pilot_index.py 以实际数据 time/lat/lon/channel 为依据，用 UTC exact lookup 联结 history、issue、各 target；不能按日历偏移盲取第 n 行。保存 issue_id、process_group_id 和每个端点索引。只检查本轮读取的完整小子集，不重写整个 pull_wb2 发布系统。
3. 先执行获批 profile：测真实前向/训练步/各 lead rollout/存储成本。登记 bank、cache、低容量策略和确认的预算分配；必须留出合法策略检验份额，不能把全部 cap 花在 oracle。任何机器小时/吞吐此前未知。
4. 优先复用有合法训练来源的 Fs/bank；否则在 fit 上训练一个静态 Fs 及小 K 非零 bank。使用固定有界系数和现有 ExpertLoRA；不需要 caller coeff 梯度。训练后冻结参考与 bank。静态对照报告相同 trainable 参数规模或实测训练成本下的强版本，不用未调优 frozen 替代。
5. bank 资格检查：每个专家在非零训练覆盖强度下造成可重复响应；零系数恢复 Fs；记录响应多样性、有效权重与非零统计。禁止为了通过资格过滤 confirm 上坏专家。初始 A 非零/B=0 是正常初始化，不等于训练失败或科学 STOP。
6. 有限 registry 含显式 no-edit、各注册 singleton。K/rank/系数/窗口由 fit/dev 与 cap 冻结；若主张组合，少量 pair 在看对应结果前注册并实际运行。实浮点、非空、唯一ID、bound/max_active/成本/窗口全验证；B15 在此阻断，不要求重写通用 planner。
7. 固定 Fref 后一次 reference 与全部非线性候选同输入、同目标、同边界运行。利用现有 cached_responses 或同语义批量缓存，不用中心差分/仅top3当全库oracle。每行保存 raw/native target、预测或可重算片段、q_hash、candidate_id、status、计费调用与失败理由；失败候选不能从分母抹掉。cache_manifest.json绑定registry/bank/Q/split/代码与各文件hash，并分别指向legal inputs和离线truth/outcomes；后续只从此清单解析路径。
8. fit/dev 选择 best-static 与轻量 regime 的结构参数；confirm 不重选。分析 V_H 及相对 dev-static 的 gap，另报强静态 Fs/F0 成绩。完整原生注册目标结算；如学习头要固定线性 D，它只影响后续训练目标，不能偷换此处原生收益。
9. 登记 δ_gain/δ_dynamic 的价值依据与独立的 pilot 方差、alpha/power/N/MDE。未定时保留规定占位符，只出 DEV_DESCRIPTIVE，不出正式 PASS。正式 confirm 阈值冻结后，才允许单次确认。
10. 先生成 dev 缓存并保留未见 confirm。默认不在 E1 用掉后续方法选择所需的最终 confirm；允许基于明确审阅的 DEV_PROMISING 进入有界 E2。若单独开启 E1 confirm，必须预登记后续策略/阈值不受其结果影响，或另留全新 confirm。

### Tests
- 空store/缺时次/重复时间/单位或坐标错/placeholder正式身份均在真实 loader 入口拒绝。
- no-edit实际gain为零；finite oracle=max全部注册候选；若候选失败 incomplete_oracle=true，不能获得正式 headroom PASS。
- 同一起报全部候选及lead属于同折；bootstrap重采样process后候选/方法严格配对。
- best-static仅fit/dev冻结；测试扰动confirm标签不能改变selected static ID或registry。
- e=±1、u∈{0,±0.5}的合成测试：oracle=.75、无信息最优legal=0；不能将oracle PASS映射成policy PASS。
- 统一分母、固定候选数、sqrt-after-aggregate；所有CPU测试仅检验代数和流程，不填入天气metrics。

### Run command
```bash
cd "$REPO" && python3 -m pytest tests/test_r2_pilot_index.py tests/test_r2_registry_and_oracle.py tests/test_r2_block_statistics.py -q
cd "$REPO" && python3 scripts/r2_bank_oracle.py --stage profile --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --out "$RUN/profile"
cd "$REPO" && python3 scripts/r2_bank_oracle.py --stage fit-bank --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --out "$RUN/bank"
cd "$REPO" && python3 scripts/r2_bank_oracle.py --stage cache --role dev --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --bank "$RUN/bank/bank_manifest.json" --out "$RUN/cache_dev"
cd "$REPO" && python3 scripts/r2_bank_oracle.py --stage analyze --role dev --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --table "$RUN/cache_dev/candidate_outcomes.parquet" --out "$RUN/oracle_dev"
# role=confirm 的 cache/analyze 只有在单独批准、阈值与分析程序冻结后才执行；不自动替换上述 dev。
```

### Expected artifacts
- profile/cost_ledger.json
- pilot_index.parquet
- data_slice_certificate.json
- bank/bank_manifest.json
- bank/training_log.jsonl
- candidate_registry.json
- cache_dev/candidate_outcomes.parquet
- cache_dev/cache_manifest.json
- cache_dev/predictions/index.json
- oracle_dev/oracle_static_gaps.json
- oracle_dev/bootstrap_intervals.json
- threshold_certificate.json
- label_resource_ledger.json
- decision.json

### Success criteria
- 合格非零冻结库、实际完整注册候选、同目标与分母。
- 正式 HEADROOM_PASS 要求原生主目标 gain 和 static gap 各自下界超过已冻结 δ；只表示可尝试预测。
- DEV_PROMISING 是 reviewer 批准的探索性投入标签，不能伪称 confirm PASS。

### Failure criteria
- bank无法在cap内合格 → STOP_CURRENT_SETUP；禁止零库假否证。
- oracle上界低于δ_gain → STOP_CURRENT_BANK；dynamic gap上界不足且静态有价值 → PIVOT_STATIC。
- CI跨阈值 → INCONCLUSIVE；缺候选/数据 → BLOCKED或INVALID，不加大训练。

### Decision after completion
停止并提交E1证据。仅 HEADROOM_PASS 或明确批准且有资源上限的 DEV_PROMISING 可申请 R2-P0-03；无自动解锁。singleton结果不得升级为组合H的结论。

**PASS能推出：** 冻结有限库有hindsight空间，不证明deployable收益。

**FAIL能推出：** 停止当前库或转静态；不是全领域否证。

**CI跨阈值：** INCONCLUSIVE；只按预注册追加规则和cap处理。

### Suggested commit message
`feat(r2): collect qualified finite-edit headroom with static controls`

提交仅限已授权且不夹带用户改动的本地commit；没有自动push授权。

---

## TASK R2-P0-03 — 同缓存上的合法策略、四象限与标签效率区分

### Scientific purpose
区分hindsight空间和可部署可学习性，并检验响应分解在有限标签下的实际增量价值。

### Depends on
R2-P0-02

旧任务映射：对应旧 P0-06 的必要子集，保留其四象限及direct-gain挑战。

### Files to inspect
- earthdelta/heads.py
- earthdelta/paired.py
- earthdelta/metrics_contract.py
- earthdelta/selection.py
- earthdelta/memory.py
- earthdelta/evaluation/r2_survival.py
- earthdelta/pilot_contract.py

### Files to modify/create
**Modify（只改本任务必要符号）**
- earthdelta/heads.py
- earthdelta/metrics_contract.py

**Create（先查是否已有等价实现）**
- earthdelta/baselines/r2_cached_policy.py
- scripts/r2_legal_policy.py
- tests/test_r2_legal_information.py
- tests/test_r2_four_cells.py

### Implementation steps
1. 复用同一bank/registry、实际候选表、起报和分组；不得因oracle赢家分布重划样本。定义 shared legal context；若已有合格提取器就复用，否则仅对注册历史原生输入作固定线性空间池化，fit-only标准化，不新增学习型天气encoder。feature维数/历史长度在dev预算内冻结。
2. 若context用参考后段tokens，必须先记录reference preview与选定编辑必要重放的实际额外成本；不能称零额外成本。起报后仅消费自身预测，不能使用已实现候选u或target。
3. 在独立合法feature文件中仅保存 issue_id/时间/合法context/版本；training target/cache另文件保存。predict子命令不得接受truth、e0、真实du或candidate outcomes。先持久化selected_actions与其hash，再由独立score子命令读取结果表评分。
4. 实现小型regime路由（fit-only聚类/原型，按fit或dev平均gain选择候选）和低容量direct-gain ridge/MLP。默认使用现有numpy/torch，不为此安装新库；同调参配置数、训练步骤和仿真资格。所有方法含显式no-edit与相同feasible registry。
5. 复用 ComposedPredictionHead 的e0/du头，不新造复杂架构。新增默认关闭的calibration开关，返回gain_analytic和可选gain_calibrated；旧gain兼容路径显式标注。memory为固定零且无已核验误差注入，JEPA off。no-edit响应和score结构为零，不能靠启用掩码把所有动作关掉。
6. 原生全目标可以作为head训练目标；若为小模型压缩训练输出，只允许预先声明的线性D及训练期拟合的固定基，记录D和Q_D并单列surrogate分数。所有最终动作仍用完整原生Q结算，报告summary/full-field决策差距；禁止声称压缩gain恒等于原生gain。此处不另立learned response basis创新。
7. direct-gain标签直接由原生候选损失差给出；直接edited-forecast作为同仿真资格的低容量诊断臂（使用相同输出读出），没有未来标签的候选轨迹同样可训练该对照。一条truth给全部候选标签，不按候选数虚增对手的标签成本。
8. 四格必须实际替换各自e或u，固定同一candidate registry，并用actual outcome结算；有true e或true u的格仅离线诊断。报告全e0误差及响应方向误差，但kill依据实际policy收益/regret/损害，不以低全场R²单独停止。
9. 若触发B04 batched Gram路径，在本任务修复正确ellipsis收缩并覆盖广播组合；不为本次有限候选实验新增连续QP或端到端caller系数训练。B03仍defer，不能将其缺失用于弱化已实现的基线。
10. 按时间块/过程crossfit，fit/dev之外保持confirm封闭。外层被评估起报的标签也不得用于bank训练：最小方案是在bank未见的policy开发块上做crossfit，固定bank只在独立fit训练；每外折的归一化/聚类/读出与HPO都只用其训练侧，不能先用全部折拟合后再称OOF。预注册少量标签预算点和固定seed；全部方法使用相同label subset与仿真资格，双头增加响应预训练时对手也可使用相同预训练/辅助任务额度。分别出given-bank/controller账本与含bank的end-to-end账本。
11. 输出实际policy-static增益及dual-direct差值CI/非劣比较。direct-gain无显著差异不等于双头无价值；要用预注册最小额外价值与非劣界判断，label/cost优势无证据时不宣称已成立。跨阈值记录INCONCLUSIVE。

### Tests
- predict接口没有truth/outcomes；poison未来payload或改变score目标不改变已保存动作；真实predict调用用于测试。
- candidate置换一致，no-edit得分/du为零；default calibration off且旧checkpoint兼容明确。
- 四格替换与完整outcome结算；mean/zero-response诊断；e全场差但响应投影正确的反例不误杀。
- 全部候选同issue同fold；fit-only标准化/聚类/输出投影；训练标签子集一致，bank标签不丢账。
- 合法策略不能因读取真实du被误报为低成本deployable；统一成本扣除或约束，不混用两种口径。

### Run command
```bash
cd "$REPO" && python3 -m pytest tests/test_r2_legal_information.py tests/test_r2_four_cells.py -q
cd "$REPO" && python3 scripts/r2_legal_policy.py fit-crossfit --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --cache-manifest "$CACHE_MANIFEST" --out "$RUN/policy_dev"
cd "$REPO" && python3 scripts/r2_legal_policy.py predict --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --features "$LEGAL_FEATURES" --models "$MODEL_MANIFEST" --registry "$REGISTRY" --out "$RUN/actions"
cd "$REPO" && python3 scripts/r2_legal_policy.py score --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --actions "$RUN/actions/selected_actions.parquet" --outcomes "$OUTCOMES" --out "$RUN/policy_score"
```

### Expected artifacts
- policy_dev/crossfit_predictions.parquet
- policy_dev/model_manifest.json
- policy_dev/legal_features.npz
- policy_dev/legal_input_manifest.json
- actions/selected_actions.parquet
- actions/action_commit.json
- policy_score/legal_policy_gain.json
- policy_score/four_cell_scorecard.json
- policy_score/label_efficiency.json
- policy_score/summary_fullfield_gap.json
- information_ledger.json
- label_resource_ledger.json
- decision.json

### Success criteria
- 合法pred/pred或合法direct策略相对强静态的下CI超过δ_policy，且guard合格；dev成功仅探索性。
- 双头价值独立判决：label/cost曲线或dual-direct差值达到注册最低额外价值，不能用oracle近似代替。

### Failure criteria
- 充分证据排除合法最低有用增益 → STOP_CURRENT_PLANNER；只缺功效 → INCONCLUSIVE。
- direct/regime有效但双头被支持为无额外价值 → PIVOT_DIRECT_GAIN或缩窄claim，不全盘否定编辑。
- 信息污染/标签账本不公平/summary偷换主指标 → INVALID，不评分。

### Decision after completion
停止并提交E2。仅出现可信合法信号（包括明确缩窄到direct-gain的信号）且reviewer批准，才解锁R2-P0-04；否则汇总STOP/PIVOT/INCONCLUSIVE，不恢复JEPA等模块。

**PASS能推出：** 在注册信息/资源下存在合法策略收益；双头贡献另判。

**FAIL能推出：** 当前planner不值得扩大或应转直接策略。

**CI跨阈值：** INCONCLUSIVE；禁止用不显著冒充等效或重调confirm。

### Suggested commit message
`feat(r2): discriminate legal edit policies under matched supervision budgets`

提交仅限已授权且不夹带用户改动的本地commit；没有自动push授权。

---

## TASK R2-P0-04 — 条件性参数编辑必要性挑战：virtual 与输出反馈

### Scientific purpose
在有限资源而非无限表达能力层面，判断真正执行参数编辑是否必要。

### Depends on
R2-P0-03

旧任务映射：对应旧 P0-07 的有界关键挑战，保留而不预先扩大。

### Files to inspect
- earthdelta/bridge/stormer_bridge.py
- earthdelta/heads.py
- earthdelta/lowrank.py
- earthdelta/evaluation/r2_survival.py
- earthdelta/pilot_contract.py

### Files to modify/create
**Modify（只改本任务必要符号）**
无；优先调用已有接口。

**Create（先查是否已有等价实现）**
- earthdelta/baselines/r2_output_correction.py
- scripts/r2_output_challenge.py
- tests/test_r2_output_challenge.py

### Implementation steps
1. 仅在E2有可信合法信号且本任务获批后实现/训练；不要提前搭完整纠错框架。继承同一Fref、样本、变量、时效、Q、hold窗口和资源账本。
2. 对已冻结合法策略运行actual edit；有限动作按选中ID分组/串行调用现有tuple接口即可，无须caller系数梯度。与缓存实际轨迹做同版本一致性核对。
3. virtual：Fref + predicted_du。只有原生full-field预测响应或已注册可验证的线性lift才能形成完整预报；若仅summary无合法lift，标VIRTUAL_FULLFIELD_BLOCKED，不伪造全场。lift的计算/误差计入。
4. 联合输出纠错：在相同合法context及明确计费的reference预览上，用低容量head联合预测注册时效/变量的残差；与双头相同固定输出读出/训练预算。
5. feedback纠错：每个注册步用当时自身预测状态和共享合法context产生注册变量的修正（其余原生变量按明确规则保留），在hold窗口内反馈到下一步Fref；不读取真实未来状态作teacher forcing推理。至少输出全模型原生状态供后续传播。
6. 固定参数规模机制对照与相同实测成本前沿分别报告，不强行声称同时完全匹配。计入bank训练、context提取、reference preview、必要重放、planner、adapter和回退成本。
7. 用E2冻结信息和fit/dev调参；在同一注册主要目标与后续guard上评价，不因纠错失败删除难变量。任何新longlead（如120/240h）需新授权，本迭代默认不扩张。
8. 若同一最终confirm用于E1/E2/E3，必须全部政策、HPO和分析规则在揭盲前冻结并一次批量评分；否则改用独立新confirm，不能反复查看同一确认集后适配。

### Tests
- zero-corrector完全还原Fref；非零第一步修正影响后续预测；不注入未来真值。
- actual与virtual分别落盘，summary-only不能冒充fullfield；lift hash/单位必须匹配。
- hold之外不再注入新的修正，但继续各自已改变的状态，不重置到原始reference轨迹。
- 相同机会分母、失败回退计费；成本前沿与非劣统计使用同一注册定义。

### Run command
```bash
cd "$REPO" && python3 -m pytest tests/test_r2_output_challenge.py -q
cd "$REPO" && python3 scripts/r2_output_challenge.py fit --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --cache-manifest "$CACHE_MANIFEST" --out "$RUN/correctors"
cd "$REPO" && python3 scripts/r2_output_challenge.py predict --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --models "$MODEL_MANIFEST" --features "$LEGAL_FEATURES" --out "$RUN/forecasts"
cd "$REPO" && python3 scripts/r2_output_challenge.py score --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --forecast-manifest "$RUN/forecasts/manifest.json" --outcomes "$OUTCOMES" --out "$RUN/comparison"
```

### Expected artifacts
- comparison/correction_frontier.json
- comparison/lead_variable_metrics.parquet
- comparison/guard_harm.json
- forecasts/manifest.json
- information_ledger.json
- label_resource_ledger.json
- decision.json

### Success criteria
- 参数编辑在至少一个预注册有限资源区间具可重复实际收益/标签效率优势，guard不越界。
- 不要求全部指标全面支配，也不由无限容量等价性判失败。

### Failure criteria
- 输出/反馈在关键效果非劣且成本更优，并达到预注册统计条件 → PIVOT_OUTPUT。
- virtual同效更便宜 → 缩窄为response surrogate，不主张必须实际编辑。
- 效果CI不足 → INCONCLUSIVE；输入/成本权限不一致 → INVALID。

### Decision after completion
停止；将证据交R2-P0-05汇总。不新增第二backbone、长时效全训或新的纠错算法。

**PASS能推出：** 特定有限资源条件下参数编辑有保留理由。

**FAIL能推出：** 转输出修复或response surrogate，不等于所有方法无效。

**CI跨阈值：** INCONCLUSIVE；不把未显著支配当作编辑胜利。

### Suggested commit message
`feat(r2): test actual parameter edits against matched output correction`

提交仅限已授权且不夹带用户改动的本地commit；没有自动push授权。

---

## TASK R2-P0-05 — 证据汇总与下一迭代继续/停止判决

### Scientific purpose
避免把工程完成、oracle上限和可部署贡献混成一个PASS。

### Depends on
R2-P0-01 的可读 decision.json；总结任务只要求前置记录，不要求该记录为科学PASS。仍须单独授权。

旧任务映射：旧P0-08的最小报告功能；不是下一研究阶段。

### Files to inspect
- artifacts/round2_next/**/decision.json
- artifacts/round2_next/**/threshold_certificate.json
- artifacts/round2_next/**/manifest.json
- STOP_CONDITIONS.md（本包）
- codex_tasks.json（本包工作副本）

### Files to modify/create
**Modify（只改本任务必要符号）**
无；优先调用已有接口。

**Create（先查是否已有等价实现）**
- scripts/r2_iteration_decision.py
- tests/test_r2_iteration_decision.py

### Implementation steps
1. 本任务只读已产生的运行产物，不训练、不加载模型、不生成新预测。可在E1/E2/E3任何终止点由reviewer单独授权，汇总未到达节点为LOCKED_UNREACHED，不能伪造其PASS。
2. 读取各任务实际code/run/research状态、确认集访问记录、threshold certificate、资源消耗及核心B项处置。禁止从文件存在、CPU pass或旧历史报告推测真实S0成功。
3. 按STOP_CONDITIONS的优先顺序生成research_verdict、implementation_readiness、claim_status、next_evidence、remaining_budget、unresolved_limitations。实现失败、缺资源、统计不足和科学否证分别表示。
4. 将oracle/static空间、合法policy技能、dual相对direct的额外价值、actual相对输出的必要性分别判决；不得合成一个不透明总分。若E3未执行，parameter_edit_necessity=UNVERIFIED。
5. 保留原研究决策文本，生成带时间和run_id的新ITERATION_DECISION.md；更新工作副本CODEX_TASKS/JSON实际状态，但approved_task_ids和授权收据只能由reviewer新授权更改。
6. 任何CONTINUE仅给下一笔明确证据与预算建议。本迭代在本任务结束；不创建或执行全量P1/P2 DAG。

### Tests
- oracle_PASS且policy_NOT_RUN不能输出deployable或dual-value PASS。
- CPU_VALIDATED/realS0_BLOCKED必须保持implementation BLOCKED；缺阈值不得生成正式confirm PASS。
- CI跨δ输出INCONCLUSIVE；无显著差异不能自动等效；incomplete候选oracle不能自动complete。
- 上游STOP后下游LOCKED_UNREACHED可汇总，但不能改写为SKIPPED_SUCCESS。

### Run command
```bash
cd "$REPO" && python3 -m pytest tests/test_r2_iteration_decision.py -q
cd "$REPO" && python3 scripts/r2_iteration_decision.py --run-root "$RUN_ROOT" --config "$CONFIG" --authorization "$AUTH" --task-state "$STATE" --out "$RUN/final"
```

### Expected artifacts
- final/ITERATION_DECISION.md
- final/iteration_decision.json
- final/evidence_index.json
- final/audit_disposition.json
- 工作副本CODEX_TASKS.md
- 工作副本codex_tasks.json

### Success criteria
- 所有结论可回指实际run、配置、阈值、统计和成本；无结果保持缺失状态。
- 明确下一笔预算是否值得投入，以及对应证据，不自动扩大范围。

### Failure criteria
- 缺绑定/矛盾状态/未授权confirm → INVALID_REPORT，并定位问题。
- 科学STOP/PIVOT是正常判决产物，不把它当脚本异常。

### Decision after completion
迭代结束，停止。任何新实验需要新的授权与范围，不从本包自动衍生后续任务。

**PASS能推出：** 报告完整可信，不代表研究结论为正。

**FAIL能推出：** 证据或报告合同不合格，应修复记录。

**CI跨阈值：** 继承INCONCLUSIVE并停止，不隐藏不确定性。

### Suggested commit message
`docs(r2): freeze bounded-iteration evidence and research disposition`

提交仅限已授权且不夹带用户改动的本地commit；没有自动push授权。

## 8. 迭代结束的固定输出

用一句话区分：参考可不可信、库有没有空间、合法policy有没有收益、双头有无增量价值、真编辑有没有必要。未执行的证据写 UNVERIFIED/BLOCKED，不补猜。输出下一笔预算购买的一个具体证据或停止理由。

本主计划在此结束。没有隐藏的“通过后自动P1/P2”。
