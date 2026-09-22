# P1 实验执行合同

本文件由同一机器任务清单展开，与主合同一致。

### TASK P1-01 — 构建完整成对轨迹数据与来源链

**Prerequisites**：P0-08。

#### Scientific purpose
复用训练侧无需未来标签的du与需要核验的e0，形成可重建、非泄漏的数据集。

#### Files to inspect
- earthdelta/probe.py
- earthdelta/bridge/stormer_bridge.py
- earthdelta/data/paired_index.py

#### Files to modify
无。

#### Files to create
- earthdelta/data/response_dataset.py
- scripts/collect_paired_trajectories.py
- tests/test_response_dataset.py

#### Implementation requirements
- 输出 reference/edited trajectory及固定线性D的summary；时效axis显式；给e0标签与du模拟标签分别标role。
- du虽然无futuretruth标签仍消耗backbone运行；所有候选模拟数、GPU时间、缓存共享条件计账。
- bank变更自动失效；禁止开发后缓存跨bank复用；filehash+keymap+operationlineage；流式采集不复制多份history。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- split按issue/process不是按candidate；empty/missinglabels不被填未来；缓存篡改拒绝；端点差identity复算。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_response_dataset.py -q
cd "$REPO" && python scripts/collect_paired_trajectories.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- response_index.parquet
- response_dataset_manifest.json
- trajectory_store/
- cost_ledger.json
- decision.json

#### Success criteria
- 全部登记init/candidate/lead可追溯，标签来源明确，缺测保留且不夸完整率。

#### Failure criteria
- 任何train/test泄漏、跨bank旧cache → STOP_DATASET；资源上限到达 → BLOCKED或完整已界定子集。

#### Decision after completion
PASS→P1-02与P1-03；不新增数据灾种或backbone。

#### Commit suggestion
`feat(data): build versioned finite-edit trajectory pairs`


### TASK P1-02 — 训练双头并建立无标签 serving 接口

**Prerequisites**：P1-01。

#### Scientific purpose
将已有组件串成真正的部署系统，而不是训练侧oracle展示。

#### Files to inspect
- earthdelta/heads.py
- earthdelta/selection.py
- earthdelta/bridge/stormer_bridge.py

#### Files to modify
- earthdelta/heads.py
- earthdelta/selection.py

#### Files to create
- earthdelta/serving.py
- scripts/train_paired_heads.py
- scripts/run_serving.py
- tests/test_serving_isolation.py

#### Implementation requirements
- 同MetricSpec训练e0/du及可选gain辅助；纯解析score与校准score分榜。输出额外返回预测状态/候选cost记录。
- predict(history,legal_context,plans,metric)->pred e/du; select->plan; executeactualplan。不能在select内部调用teacher/evaluateallcandidates或readfuture。
- 选择step0后段features时计referencepreview与回放；默认共同完整preview以公平，不承诺零成本。
- 不通过tuple/tensor重建切断需要的梯度；trainingforward和inferenceplan分开；参数/数据flowtyped但不把typehints称sandbox。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- futurestore不可挂载/读取；poison标签不改变plan；候选置换等变；explicitnoeditgain0；冻结weightsSHA不变。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_serving_isolation.py -q
cd "$REPO" && python scripts/train_paired_heads.py --config "$CONFIG" --out "$RUN_DIR/train"
cd "$REPO" && python scripts/run_serving.py --config "$CONFIG" --model "$HEAD_CHECKPOINT" --out "$RUN_DIR/serve"
```

#### Expected artifact
- train/model_manifest.json
- serve/plans.parquet
- serve/predictions/
- serve/information_access.jsonl
- serve/cost_ledger.json
- decision.json

#### Success criteria
- 完整serving不接触真值，训练实际完成有权重；actual回放结果与plan一致；内容身份守卫生效。

#### Failure criteria
- 访问未来/候选oracle或计费漏preview → STOP_SERVING_CLAIM。

#### Decision after completion
PASS→P1-04；模型分数只来自actualrun。

#### Commit suggestion
`feat(serving): deploy analytic paired-response selection`


### TASK P1-03 — 训练最强同条件基线并做预算前沿

**Prerequisites**：P1-01。

#### Scientific purpose
排除条件网络、常规适配、一般辅助监督和额外计算的解释。

#### Files to inspect
- earthdelta/heads.py
- earthdelta/lowrank.py
- earthdelta/baselines/
- reference/_manifest.json

#### Files to modify
无。

#### Files to create
- scripts/train_baselines.py
- configs/baselines.json
- tests/test_baseline_fairness.py

#### Implementation requirements
- 至少F0、Fs、beststatic、largerstatic、samecontexttemporalrouter+multistep、directgain、feedbackoutput、virtualoutput。random和oracle免训练但角色独立。
- 可参考DISeL/WeatherPEFT/CoMoL等论文；许可未核实不vendor，改写标adapted且不得继承其官方结论。
- 各组相同data/targets/earlystop/HPOtrial上限；报告总参数/训练和教师生成成本，不仅rank。
- L2损失/summary/全场目标一致；closedloop基线按相同hold后reference规则，输出反馈不削弱。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 每组schema自动检查共享信息和预算；seed/run/config缺失不能入主表；virtual和actual标签不混。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_baseline_fairness.py -q
cd "$REPO" && python scripts/train_baselines.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- baseline_manifests.json
- fairness_matrix.csv
- trained_models/index.json
- budget_frontier.csv
- decision.json

#### Success criteria
- 全部必要强基线真的完成或明确BLOCKED；非matching部分解释，未run不填0。

#### Failure criteria
- 必要directgain/feedback未跑则主novelty比较BLOCKED。

#### Decision after completion
PASS→与P1-02共同进入P1-04。

#### Commit suggestion
`feat(baselines): enforce same-information edit and correction comparisons`


### TASK P1-04 — 有限组合交互与局部近似核查

**Prerequisites**：P1-02, P1-03。

#### Scientific purpose
区分全程非线性交互和由向量内积自然产生的cross terms。

#### Files to inspect
- earthdelta/probe.py
- earthdelta/geometry.py
- earthdelta/selection.py
- earthdelta/heads.py

#### Files to modify
无。

#### Files to create
- scripts/evaluate_edit_composition.py
- tests/test_composition_residual.py

#### Implementation requirements
- 计算I(a,b)=du(a+b)-du(a)-du(b)，所有响应与a=0同参考；pair实际跑而非仅Σdu代替。
- 比较finitewholeplan预测、linearR、single-response-sum及full/diagH；同等候选和budget。
- 只在已登记小幅度域使用R；epsilon及mixed校验、fp32/64诊断；零bank不做“rank发现”。
- 完整矩阵不优于对角则删除“交互贡献”，不为了保住结论改pair筛选。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- 线性toy I=0而quadraticcross可能非零；非线性toy I非零；分母接近0标清。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_composition_residual.py -q
cd "$REPO" && python scripts/evaluate_edit_composition.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- composition_errors.parquet
- finite_vs_linear_scorecard.json
- full_diag_comparison.json
- decision.json

#### Success criteria
- 近似有效域与失效情况可复算；必须有actual组合而非只有代数证明。

#### Failure criteria
- 混合近似失效则改finite-only，不因此杀所有finite方向；full≈diag则降级交互声明。

#### Decision after completion
完成明确范围后→P1-05。

#### Commit suggestion
`feat(eval): separate exact edit composition from quadratic overlap`


### TASK P1-05 — 独立确认、成本与论文证据发布

**Prerequisites**：P1-04。

#### Scientific purpose
在未参与选择的数据上建立可投稿证据，而不是不断迭代开发分数。

#### Files to inspect
- earthdelta/evaluation/
- reference/weatherbenchX/
- artifacts/p1/

#### Files to modify
无。

#### Files to create
- scripts/export_wbx.py
- scripts/evaluate_confirm.py
- scripts/build_paper_artifacts.py
- tests/test_evaluation_contract.py

#### Implementation requirements
- 锁方法/weights/metric/candidate/splits/HPO后一次确认；多lead6/24/72/120，240为预登记extension。
- WeatherBench-X作为评分候选，优先用已锁本地版本；独立小数组手算验证聚合，不把官方库自动当科学正确保证。
- aggregateMSE→sqrt，ACC用训练climatology；变量/过程分组CI；配对方法共同validmask；主榜failure不静默排除。
- GPUwarmup、synchronize、batch一致、多次测量median/p95；初始化/preview/head/planner/editedrollout单列，训练teacher成本另列。
- 输出oracle分布、predresponse真实性、static/dynamic/oracle、directgain/outputchallenge、longrollout与efficiency，并附全部负结果。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- WBX与手算一致；候选失败分母保留；confirmationhash变化拒绝；原论文表缺真实记录则禁止生成。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_evaluation_contract.py -q
cd "$REPO" && python scripts/evaluate_confirm.py --config "$CONFIG" --out "$RUN_DIR"
cd "$REPO" && python scripts/build_paper_artifacts.py --run "$RUN_DIR" --out "$RUN_DIR/paper"
```

#### Expected artifact
- confirmation_manifest.json
- metrics.json
- bootstrap_intervals.json
- paper/figures/
- paper/tables/
- cost_profile.json
- decision.json

#### Success criteria
- 核心假设在独立数据证据支持，范围与声明一致；所有主表可从冻结记录重建。

#### Failure criteria
- 确认失败保留；不能重命名test为dev再申请同一结论；重大回退STOP或修改主张。

#### Decision after completion
PASS可选择P2之一；P2不是P1成功的事后补救。

#### Commit suggestion
`feat(eval): publish frozen confirmation and full cost evidence`
