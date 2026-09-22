# Start here — 只执行一个已解锁任务

目标：获得“有限候选的实际修正空间能否由合法信息利用”的证据，不默认双头有论文贡献。

按顺序读：
1. `FINAL_RESEARCH_DECISION.md`
2. `EarthDelta_Codex_Execution_Plan.md`
3. `CODEX_TASKS.md`、`codex_tasks.json`
4. `STOP_CONDITIONS.md`、`IMPLEMENTATION_AUDIT_ACTIONS.md`

然后仅执行 **R2-P0-01：最小代码修复与合成CPU验证**。先用包内 `tools/repo_preflight.py` 冻结本地版本，保存原有dirty状态；不得reset/覆盖用户工作。具体路径和命令见主计划§6。

观察的远程HEAD是 `403b55db65f4c35c1a85d0794ad0de2765b07d96`；受审源码是 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`。远程原源码/测试对象一致，但本地仍需核对。

初始没有GPU、真实checkpoint加载、数据扫描、训练、仿真、confirm或下载授权；它们缺失时写BLOCKED，不做替代天气数据/虚构结果。固定系数bank不需要caller系数梯度，但bank任务当前仍未解锁。

R2-P0-01结束就停止并输出改动、真实CPU测试结果、尚缺真实S0证据和下一步建议。真实S0 PASS也不会自动解锁R2-P0-02；须reviewer批准和资源cap。R2-P0-02～05全部LOCKED。

本包内新CLI是要求你在对应任务实现的接口，不声称已有。优先复用现有函数与测试；不再做literature review，不重写整个项目，不添加JEPA/memory等模块。
