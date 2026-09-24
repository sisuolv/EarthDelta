# EarthDelta：从这里开始

本包审查的是 **plans_v3_0922 原计划的执行情况**，不是再做一次research brainstorming。

**审查结论：PARTIALLY_EXECUTED；科学路线BLOCKED。** 当前S0多步记录失败；原fitted-Fs的退化/共同参考问题未关闭。新shared-F0另案不算原计划完成。

## 阅读顺序

1. `evidence/PLAN_AUDIT_RESULT.json` 和 `CURRENT_GAP_ANALYSIS.md`。
2. `evidence/source_plan/NEXT_STEPS_10H_PLAN.md`、`CLAUDE_10H_PROMPT.md`；其余三份review只作历史依据。
3. `FOLLOWUP_PLAN.md`、`EXECUTION_DAG.md`、`STOP_CONDITIONS.md`。
4. `ARTIFACT_CONTRACT.md`、`TEST_PLAN.md`、`REPRODUCIBILITY.md`、`evidence/REQUIRED_INPUTS.md`。
5. `CODEX_TASKS.json`；按任务读取相关仓库代码，不凭本报告代替代码。

## 当前边界

- 当前首先执行 **FP-00**：只读入口核查、收集缺证、冻结source/资产目录。
- FP-00合格且已有S0 GPU预算授权后，才可执行 **FP-01**。允许S0诊断所需CPU补丁与一次有界GPU复验；不允许训练expert。
- **FP-02至FP-06初始LOCKED**。依赖通过只产生解锁资格，不允许用本包的规划文字代替真实PASS或资源授权。
- 无论S0诊断结果如何，都不能以shared-F0的存在跳过原计划Fs质量/共同参考gate。
- 不修改远端分支、不reset/stash/drop用户工作；工作区已有改动要记录并隔离。不要自动拉取大天气数据或升级共享Torch环境。

## 起点

核验时branch HEAD=`fdd92d79b1b03e8397d4c487d7fa4a347ae6fc37`；实现=`a3596e6b804e9d23b72d1247b08c47129d6b50b1`，比较基线=`d749a1c62521226df857587e08f7d067b0f15355`。
两个HEAD代码树相同，仅plans改变。Codex启动时仍须重新核验，HEAD不同就比较树/实际diff，不能盲目checkout覆盖用户修改。

可先运行包本身的只读验证：

```bash
python "$PKG/tools/validate_delivery.py" --package "$PKG"
python "$PKG/tools/collect_entry_evidence.py" --repo "$REPO" --out "$RUN/entry"
```

`PKG/REPO/RUN`须设置为真实绝对路径；RUN是新的目录。第二条只收集版本和source证据，不运行模型。

每项任务结束保存机器结果；关键依赖FAIL、BLOCKED、MISSING或统计INCONCLUSIVE到达cap时停止。所有未来结果=`TO_BE_RUN`；不得把模板和命令当作实验输出。
