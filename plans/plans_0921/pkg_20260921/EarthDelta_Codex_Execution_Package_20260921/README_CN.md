# EarthDelta 审查与 Codex 执行包

依据：两份用户任务文件、当前 GitHub main@4fe55a7af90ea92f62a3232a571af92bfbd6114d 的定向源码审查、截至2026-09-21 UTC的新近原始论文。美国东部当地日期为2026-09-20。

## 使用
先读 START_HERE_FOR_CODEX.md。当前仅允许P0-01只读资产核查。主合同 EarthDelta_Codex_Execution_Plan.md 可独立作为工程规范；CODEX_TASKS.md与codex_tasks.json保持同一16任务DAG。

research/：当前事实、数学审查、近邻、V2、最终决策、八个问题。
experiments/：P0生存、P1核心、P2增强与强基线。
implementation/：逐文件修改、测试、指标、产物、复现与外部复用许可边界。
evidence/：审查事实、静态发现、源位置、包校验与代数自检。
tools/：只读预检、独立代数检查、包结构验证；**不是研究源码补丁**。

## 状态
没有修改GitHub，没有执行真实天气训练/预报，没有重跑完整当前repo测试，也没有读取用户远端权重/data bytes。README所报192通过3跳过只作为来源记录。这个包提供可执行研究工作顺序，不冒充已完成实验结果。

不重新分发私有源码、第三方源码、模型权重或数据。许可不明的外部实现只作研究参照，不能默认复制。
