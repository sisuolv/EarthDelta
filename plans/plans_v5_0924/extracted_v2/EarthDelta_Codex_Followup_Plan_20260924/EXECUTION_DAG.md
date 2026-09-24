# 执行 DAG

```text
历史证据（保留，不自动重跑）
FP-00 DONE inventory
FP-01 DONE registered S0
FP-02 PARTIAL [已完成：stage STOP / Fs admission subset / profile copy]
FP-03 DONE Fs v2 limited-harm qualification
FP-04 DONE same-Fs training qualification + one-probe A1–A5
               │
               ▼
FP-05a 只读证据/全项目曝光/原始回执闭环 [唯一初始解锁]
   ├── 缺回执/大资产 → BLOCKED，补挂载/文件，不重训
   ├── 哈希不符 → INVALID，隔离证据，人工复核
   └── PASS + 验收
          ▼
FP-05b 新split gate + 薄runner + CPU反例 + 结果前协议
   ├── split/参数/72h阈值/预算未定 → BLOCKED
   ├── 消费链fail-open → FAIL，不读DEV
   └── PASS + 明确GPU授权
          ▼
FP-05c C-J1 已曝光debug / 机械成本profile / 一次冻结N
   ├── 复现/身份失败 → FAIL_IMPLEMENTATION
   └── PASS + 完整缓存预算已批准
          ▼
FP-05d C-J2 固定DEV × 5候选 + F0背景完整缓存
   ├── 缺候选/损坏/成功子集 → INVALID_CACHE
   └── PASS
          ▼
FP-05e 嵌套purged OOF / 模型与动作冻结
   ├── 任何标签越权/折重叠 → INVALID_EXPERIMENT
   └── PASS
          ▼
FP-05f native结算 / paired UTC-block CI / 成本与限定
          ▼
FP-06 DEV判决：STOP_CURRENT_BANK / PIVOT_STATIC /
      CONTINUE_EVIDENCE / INCONCLUSIVE
      （不是自动confirm，也不是novelty已成立）
```

`DONE` 只限定历史验收对象。独立输入入口或改变数值后端不能借用旧PASS。
FP02尚未做的消费者闭环由FP05a/b显式补齐；不能因为FP03/04推进就把FP02全部勾选。
Bank OOS未知正是FP05要取得的证据，不是禁止运行FP05的循环条件；但它严格阻止FP06在无结果时给出正面效用结论。

## 判决语义
工程节点PASS仅允许消费已核验的产物；不推出weather utility。Oracle上界有空间，仅说明值得尝试合法selector。合法ridge超过Fs但未超过cross-fit static，不证明动态必要。宽CI或不够独立块写INCONCLUSIVE，达到预算停止，不能解释成成功/等效/没有价值。
