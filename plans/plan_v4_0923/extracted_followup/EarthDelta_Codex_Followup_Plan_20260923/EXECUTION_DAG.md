# 执行DAG及每一条结论边界

```text
FP-00 原始证据与身份预检
  ├─ MISSING/不一致 → BLOCKED；可做已授权只读整理，不训练
  └─ PASS + S0-only GPU授权 → FP-01 S0四臂/独立官方认证
       ├─ 任一1/4/12步>1e-5 → BLOCKED（不是科学否定）
       └─ PASS → FP-02 STOP/准入/Q/profile消费闭环
            ├─ FAIL → 修复，不调用Fs训练
            └─ PASS → FP-03 唯一Fs质量/重载/冻结
                 ├─ 质量失败 → STOP_CURRENT_FS / 有限训练侧修复；禁止bank
                 └─ PASS → FP-04 同Fs专家训练/资格/组装/registry
                      ├─ 任一失败 → STOP_CURRENT_BANK；禁止cache
                      └─ PASS → FP-05 8exposed→冻结dev全5候选→廉价OOF表
                           ├─ 不完整/泄漏 → INVALID_EVIDENCE；停止评分
                           ├─ oracle不足 → STOP_CURRENT_BANK，不否定全部idea
                           ├─ oracle足但legal不利用 → STOP_CURRENT_SELECTOR或窄化
                           ├─ static解释收益 → PIVOT_STATIC
                           ├─ CI跨阈值 → INCONCLUSIVE；仅在预登记cap内补证
                           └─ 合法收益有证据 → FP-06 核验/决策/交接
                                └─ 不自动解锁confirm/dual-head/新领域
```

FP-01的PASS只证明固定source/asset/config和注册horizon的官方parity。不是Fs合格、bank合格或weather skill。
FP-03只证明当前训练侧规则下唯一Fs合格；不是所有未来分布的强静态基线。
FP-04证明当前共同reference和动态产物可用；不证明动态选择必要。
FP-05 oracle PASS只证明hindsight headroom。E[max gain]不是E[max E(gain|I)]，不能把真future引入policy。
FP-06若廉价ridge/regime已解释全部收益，原计划的开发目标可能已完成，但不构成response factorization创新证据。

统计CI只用于科学比较节点；代码/identity gate没有“CI跨阈值就算通过”的逻辑。
只有被检查的确切版本/资产具备PASS资格。改变bridge/norm/rollout回到FP-01，改变Fs回到FP-03，改变bank回到FP-04；不是重用旧证书改一个名字。

本包保留原计划Fs前置：shared-F0必须另案授权和独立状态机，不能在此DAG插入捷径。
