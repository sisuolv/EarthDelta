# CODEX_TASKS — one bounded iteration

## P0 / 本包任务ID（不替换旧P0编号）
- [ ] **R2I-01** 最小 pilot 的模型、度量与数据入口正确性 — 当前唯一允许
- [ ] **R2I-02** 冻结最小非零 bank、候选 registry 与共同仿真缓存 — LOCKED
- [ ] **R2I-03** 同一缓存区分 oracle 空间与合法可预测增量 — LOCKED
- [ ] **R2I-04** 同资源检验分解、决策投影与 direct-gain — LOCKED
- [ ] **R2I-05** 参数执行对输出纠正的最小挑战与迭代裁决 — LOCKED

## Current stopping point

Current task: R2I-01
Do not proceed beyond: R2I-01
Unlocked task IDs: R2I-01
Auto advance: false

## Unlock condition

R2I-01所选模型/数据/Q与真实独立S0全部通过，只产生R2I-02资格；还必须获得owner明确下一任务授权与训练/仿真cap。
CPU_PATCH_PASS + real_s0=BLOCKED 不放行bank。任何PASS不会自动改变allowed_task_ids。
资源与阈值分阶段：代码小测试不因未填delta_min阻塞；真实S0因缺必需资产/许可阻塞；正式统计判断因阈值未冻结阻塞。

## 更新规则

只在工作副本勾选实际完成且证据满足的任务；写本任务decision.json、真实日志与新HEAD，不修改原package hash。
后继仍锁定时结束响应，提供下一任务资格建议，不自己批准。未跑结果写NOT_RUN，缺资源写BLOCKED，CI跨门槛写INCONCLUSIVE。

## 本迭代边界

不添加P1/P2任务、不跑JEPA/memory/dynamic rank/spectral/new-backbone。R2I-05结束后必须停止。
