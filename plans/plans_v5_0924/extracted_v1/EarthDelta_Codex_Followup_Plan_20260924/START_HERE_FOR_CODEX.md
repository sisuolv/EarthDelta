# EarthDelta 后续执行入口 · 2026-09-24

本包是上一轮 FP-00～FP-04 的后续，不是重新开始研究。**overall_verdict=PARTIALLY_EXECUTED**。
S0 已有固定 `1e-5` 下 1/4/12 步全零的正式结果；Fs v2 和 K4 bank 已有有限范围认证。
**不要重跑已经完成的阶段，不要继续使用“S0仍失败”的旧状态。** FP-02 尚余薄runner/角色曝光消费链；FP-05 尚未实施。

## 首个必须修正的事实
X1-N64 实际训练使用了 56 个 2020 年 7–12 月样本。`FP05_PLAN.md` 把整个2020下半年预留为 untouched confirm 是错误的。
该探索模型没有被选为最终Fs，不会把其影响从研究设计曝光中抹掉。详见 FA-06 / E14。

## 阅读顺序
1. `evidence/PLAN_AUDIT_RESULT.json`、`CURRENT_GAP_ANALYSIS.md`。
2. `FULL_REVIEW.md`（特别是X1曝光、delta与回顾性OOF范围）。
3. `FOLLOWUP_PLAN.md`、`EXECUTION_DAG.md`、`STOP_CONDITIONS.md`、`CODEX_TASKS.json`。
4. `ARTIFACT_CONTRACT.md`、`TEST_PLAN.md`、`REPRODUCIBILITY.md`、`evidence/REQUIRED_INPUTS.md`。
5. 对照仓库的FP03/04原协议、证书与实际代码，不把本包摘要当成权重本身。

## 当前唯一授权
**FP-05a：CPU/read-only证据核对、全历史曝光重建、FP-05协议修订草案。**
允许写入新run目录和新增审查/闸门文件的草案，不允许训练Fs/专家，不允许policy_dev正式weather cache，不读取confirm值。
完成FP-05a后保存证据并停止。后续依赖满足只产生解锁资格；每次实际GPU工作还需明确资源授权。

## 起点
实际分支 `08093650ba56e8cf709d75a61c26c50a0c35f6f7`；本次实现/FP05计划受审 `e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`。核心代码树一致，前者新增文档。
启动时重新查HEAD；不同则核对受保护文件和证书，不reset/stash/checkout覆盖用户工作。
FP04锁定的17个源文件不得直接改动。优先新增薄包装，若必须修改认证核心则停止、登记新身份和重新认证范围。

```bash
python "$PKG/tools/validate_package.py" --package "$PKG"
python "$PKG/tools/check_task_evidence.py" --repo "$REPO" --out "$RUN/entry_receipts.json"
```
`PKG/REPO/RUN`必须设为真实绝对路径；第二条只读取已有记录并输出存在性/hash，不运行模型，也不自动发放PASS。
