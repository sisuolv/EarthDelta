# START HERE FOR CODEX

## 先读

1. [FINAL_RESEARCH_DECISION.md](FINAL_RESEARCH_DECISION.md)
2. [EarthDelta_Codex_Execution_Plan.md](EarthDelta_Codex_Execution_Plan.md)（完整开始执行合同）
3. [CODEX_TASKS.md](CODEX_TASKS.md)、[STOP_CONDITIONS.md](STOP_CONDITIONS.md)
4. 只在对应任务需要时读 [IMPLEMENTATION_AUDIT_ACTIONS.md](IMPLEMENTATION_AUDIT_ACTIONS.md) 与 experiments/。

## 当前只允许

**执行 R2I-01。Do not proceed beyond: R2I-01。**

本包日期2026-09-21；交付branch HEAD `403b55db65f4c35c1a85d0794ad0de2765b07d96`；受审源码 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`。先核对本地版本，禁止reset/revert以“匹配审计”。有漂移先记录，重叠未知修改停止。

先运行主计划§4的包校验和只读preflight，再按真实路径建立working_config。模板不是可直接运行的天气配置。包内tools可运行；任务命令涉及的新增scripts/tests由Codex在该任务中实现，不能声称现在已存在。

R2I-01可做必要最小patch与已知宿主限制内的定向CPU小测试。真实S0只有可信资产、上游backend、数值tol、GPU许可和cap齐备才运行；否则记录CPU_PATCH_PASS（若实际通过）+ real_s0=BLOCKED并停止。

## 尚未解锁

R2I-02 bank/cache、R2I-03 oracle/legal、R2I-04分解挑战、R2I-05输出挑战全为LOCKED。前继PASS只是证据资格，不是执行许可。每任务结束保存decision.json后停，等待owner明确授权；不得自改allowed_task_ids。

## 必须停止

缺资产/授权→BLOCKED；correctness/provenance失败→停止科学比较；CI跨门槛→INCONCLUSIVE；oracle PASS只说明hindsight空间，不宣布deployable成功。禁止自动下载/pip/push/大重构/额外模块、默认禁止confirm。

当前目标是购买最有信息价值的一份证据，不是完成全项目。只阅读本包和已有仓库即可开始，不需要重做literature review。
