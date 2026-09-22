# EarthDelta Codex Tasks

审查提交：`4fe55a7af90ea92f62a3232a571af92bfbd6114d`。所有任务当前未执行；本包生成不代表任务完成。

## P0

- [ ] P0-01 只读冻结当前版本与本地资产 — READY_READ_ONLY
- [ ] P0-02 统一数学、指标、可行域与时间合同 — LOCKED
- [ ] P0-03 修复并执行独立真实 Stormer S0 — LOCKED
- [ ] P0-04 冻结合格参考、非零字典和候选集合 — LOCKED
- [ ] P0-05 完整有限候选 oracle ceiling 与静态差距 — LOCKED
- [ ] P0-06 交叉拟合双头、直接gain与oracle分解 — LOCKED
- [ ] P0-07 参数编辑对输出反馈纠错挑战 — LOCKED
- [ ] P0-08 生存门审议与主线冻结 — LOCKED

## P1

- [ ] P1-01 构建完整成对轨迹数据与来源链 — LOCKED
- [ ] P1-02 训练双头并建立无标签 serving 接口 — LOCKED
- [ ] P1-03 训练最强同条件基线并做预算前沿 — LOCKED
- [ ] P1-04 有限组合交互与局部近似核查 — LOCKED
- [ ] P1-05 独立确认、成本与论文证据发布 — LOCKED

## P2

- [ ] P2-01 未见修正组合与时窗的轨迹响应泛化 — LOCKED
- [ ] P2-02 可校准的选择性不编辑 — LOCKED

## P3

- [ ] P3-01 登记延期项目但不实现 — DEFERRED

## Current stopping point
Do not proceed beyond: **P0-01**.

## Unlock condition
P0-01完成实际HEAD/资产/测试清单后，先停止，保存decision。后续执行根据主计划的前置任务、正式gate和已批准资源逐步解锁；不能仅修改checkbox代替实验准入。

## 更新规则
只有对应artifact/command/exitcode在场时勾选；BLOCKED和INCONCLUSIVE不勾成科学PASS。每次更新codex_tasks.json的status、decision_artifact和current_stopping_point；依赖任务不通过不能启动。
