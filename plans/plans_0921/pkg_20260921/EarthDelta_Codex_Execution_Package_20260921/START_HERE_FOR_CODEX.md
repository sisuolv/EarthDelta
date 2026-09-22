# EarthDelta 执行入口

审查起点：`main@4fe55a7af90ea92f62a3232a571af92bfbd6114d`。若实际HEAD变了，先对照，不reset。

目标：判断有限参数编辑的可预测响应能否在相同历史与成本下，胜过强静态、直接gain及反馈式输出纠错。不是增加JEPA/memory/谱模块。

先读：
1. `EarthDelta_Codex_Execution_Plan.md`（0–7节及TASK P0-01）
2. `CODEX_TASKS.md` 与 `codex_tasks.json`
3. `STOP_CONDITIONS.md`
4. `research/FINAL_RESEARCH_DECISION.md`

**本次只执行 P0-01。** 使用 `tools/repo_preflight.py` 只读核查当前repo，收集实际tests/资产；不自动安装、下载、训练、申请GPU或写远端。

完成后保存preflight、资产清单、真实testlog（未跑写NOT_RUN）和decision，更新任务状态，**停止**。不要自动执行P0-02。后续执行必须按DAG、验证gate与资源上限放行。

没有数据/权重/GPU：BLOCKED。不用合成数填真实预报。旧README的195测试、旧kit的21/50测试不能当本轮结果。

本包`tools/`里的检查仅验证包与代数，不训练模型、不修复当前repo。计划中Files to create的脚本须由Codex实现后才可运行。
