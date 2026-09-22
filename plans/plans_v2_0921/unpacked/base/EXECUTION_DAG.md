# EXECUTION_DAG — 仅未来一个迭代

任务命名空间：round2-next-iteration-20260921。编号不替代历史P0编号。

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

## 节点解释

| 节点 | 为什么执行 | PASS能推出 | FAIL能推出 | CI跨阈值 |
|---|---|---|---|---|
| P0-01 | 让测量对象真实一致 | 受检路径可测 | 实现/资产不合格，非科学否定 | 不适用统计CI；数值tol未知BLOCKED |
| P0-02 | 排除零/随机bank及假数据资格 | 候选域有资格被评测 | 设置无效或到cap仍无可用bank | 不以科学CI代替资格 |
| P0-03 | 看当前域有无事后可用空间 | 只值得小型可预测性试验 | 当前registry余量不足，不否定全部参数编辑 | 只按已登记cap补证，不改候选 |
| P0-04 | 将oracle空间与合法信号分开 | 当前信息/资源下有开发收益 | 当前策略或分解无价值证据 | 不用oracle强行补成PASS |
| P0-05 | 挑战参数执行必要性 | 可申请下一窄迭代 | 支持更便宜命题或停当前配置 | 达到cap后终止追加 |

所有箭头都需要：前序产物+测量/科学条件+任务范围与资源的明确新授权。Codex不能只因exit code=0就解锁。STOP/PIVOT意味着报告后停止，不意味着自行实现转向项目。
