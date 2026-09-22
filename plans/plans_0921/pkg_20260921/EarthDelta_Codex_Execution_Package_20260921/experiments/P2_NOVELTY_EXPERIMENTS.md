# P2 实验执行合同

本文件由同一机器任务清单展开，与主合同一致。

### TASK P2-01 — 未见修正组合与时窗的轨迹响应泛化

**Prerequisites**：P1-05。

#### Scientific purpose
唯一主要novelty增强：把候选向量预测变成对未见有限修正程序的可验证泛化，而非泛泛world model命名。

#### Files to inspect
- earthdelta/contracts.py
- earthdelta/heads.py
- earthdelta/bridge/stormer_bridge.py

#### Files to modify
- earthdelta/contracts.py
- earthdelta/heads.py
- earthdelta/bridge/stormer_bridge.py

#### Files to create
- scripts/p2_action_generalization.py
- tests/test_action_holdout.py

#### Implementation requirements
- 优先同bank未见amplitude/hold/window/组合；分别冻结actionholdout与atmospheretimeholdout。
- 利用已有ProgramSpec/leadhead，必要时加明确作用时窗描述符，不另起JEPA/新Jacobian系统。
- 同样给directgain与directrouter候选/预算/lead descriptors；不能说其按构造不能新预算。
- longlead外推与已有lead插值分开；新bank需weight/behavior descriptors及独立标定，明确不是当前zero-shot主张。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- train中无holdoutprogram及近重复；候选置换；同描述符不同内容bank拒绝旧cache。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_action_holdout.py -q
cd "$REPO" && python scripts/p2_action_generalization.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- action_split.json
- generalization_scorecard.json
- composition_rollouts/
- decision.json

#### Success criteria
- 在预登记未见程序上有actualtrajectory校验，相比同描述符基线的收益有CI支持。

#### Failure criteria
- 仅已见candidate有效→收窄lookupsurrogate定位；不称通用interventionworldmodel。

#### Decision after completion
保留有效的一条增强；失败不自动加memory。

#### Commit suggestion
`feat(p2): test held-out intervention trajectory generalization`


### TASK P2-02 — 可校准的选择性不编辑

**Prerequisites**：P1-05。

#### Scientific purpose
处理选择器从多个噪声预测中挑最大值的过度乐观风险，提供经验性可靠性而非无条件安全承诺。

#### Files to inspect
- earthdelta/selection.py
- earthdelta/heads.py

#### Files to modify
- earthdelta/selection.py

#### Files to create
- earthdelta/calibration.py
- scripts/p2_selective_editing.py
- tests/test_calibration_isolation.py

#### Implementation requirements
- 只有P1观察到gain校准/误修正问题时启用；单独校准时间块、候选数量/预算conditioning，threshold不从test定。
- mean-uncertainty penalty只是选择规则；经验coverage/riskcurves、under-shift失效一起报告，不照搬VI-MoLE证书。
- 两个upperrisk差不是普遍gainlowerbound；采用直接gain误差校准或有效的jointassumption证明，并审查所需exchangeability。
- noedit也计controller成本；回退不是零成本。不得调高拒绝率后只报被编辑子集精度。

#### Must NOT change
- 冻结声明的 Fref 与字典身份；serving 不读未来真值或已实现候选答案。
- 不覆盖历史结果、数据或未提交工作；不扩大已冻结预算/候选/划分以制造正结果。
- 保留本任务之前的回归测试；无法获得资源时记录 BLOCKED，不能以合成数代替天气结果。

#### Tests to add
- calib/test隔离，testtruth变动不改变threshold；allopportunityrisk与conditionalrisk分开。

#### Command
```bash
cd "$REPO" && python -m pytest tests/test_calibration_isolation.py -q
cd "$REPO" && python scripts/p2_selective_editing.py --config "$CONFIG" --out "$RUN_DIR"
```

#### Expected artifact
- calibration_manifest.json
- risk_coverage.csv
- harmful_edit_rate.json
- decision.json

#### Success criteria
- holdout经验误修正/覆盖改善且净收益在预登记范围内；条件与效度范围明确。

#### Failure criteria
- 跨shift校准无效→不称safe；coverage低不能隐藏totalutility差。

#### Decision after completion
不与P2-01同时盲目启动；解决已观测瓶颈。

#### Commit suggestion
`feat(p2): calibrate selective editing on held-out blocks`
