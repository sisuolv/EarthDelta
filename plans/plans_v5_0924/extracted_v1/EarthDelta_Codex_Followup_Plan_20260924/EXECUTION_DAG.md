# 依赖与授权

```text
历史：FP-00 DONE → FP-01 DONE → FP-03 DONE(v2) → FP-04 DONE(限定认证)
                     FP-02 PARTIALLY_DONE（已做子项保留，不重新全部执行）
                                      │
                                      ▼
FP-05a 证据补链＋全曝光重建＋计划更正 [唯一初始授权，CPU]
  ├─ hash/receipt矛盾或X1遗漏 → BLOCKED，禁止GPU
  └─ PASS_METADATA_AND_AMENDMENT → 仅取得下一项资格
          ▼
FP-05b 新split/入口/OOF代码＋CPU反例＋完整协议冻结
  ├─ null关键字段/入口可绕过/confirm误标 → BLOCKED
  └─ PASS + 明确GPU预算授权
          ▼
FP-05c 8例exposed debug＋盲profile＋一次冻结N
  ├─ parity/anchor/身份失败 → STOP，不能换样本找PASS
  └─ PASS
          ▼
FP-05d 一次性完整5候选cache＋F0报告背景
  ├─ 缺行/失败/哈希错误 → INVALID，不能只评分成功子集
  └─ PASS_COMPLETE_MATRIX
          ▼
FP-05e purged nested OOF → prediction freeze
  ├─ 标签泄漏/毒化测试不成立 → INVALID
  └─ PASS_FROZEN_PREDICTIONS
          ▼
FP-05f 一次解封评分＋paired blocks
  ├─ CI跨阈值/少block → INCONCLUSIVE，无扩样
  └─ ABOVE/BELOW/STRADDLE证据表（还不是novelty结论）
          ▼
FP-06 独立迭代决策：STOP / PIVOT / BOUNDED_CONTINUE
```

FP05必须测量bank OOS价值；“目前未知”不是要求它先证明才允许运行的循环gate。
但FP06的任何正面utility结论都依赖真实结果、72h guard、成本和完整证据。
confirm当前未授权且窗口未定义，不能用已曝光2020H2补位。依赖满足不自动授权后续运行。
S0/Fs/bank源或资产漂移时先停止并登记重新认证范围；不在新任务包中默认重跑旧阶段。
