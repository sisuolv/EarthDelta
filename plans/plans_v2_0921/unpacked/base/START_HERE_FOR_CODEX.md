# START_HERE_FOR_CODEX

当前目标：执行一个能降低研究不确定性的迭代，不建设完整EarthDelta。

1. 先读 `FINAL_RESEARCH_DECISION.md`、`EarthDelta_Codex_Execution_Plan.md`、`CODEX_TASKS.md`、`STOP_CONDITIONS.md`；改代码时看 `IMPLEMENTATION_AUDIT_ACTIONS.md`。
2. **只执行本包命名空间 `round2-next-iteration-20260921` 的 P0-01。** 它允许最小correctness源码补丁与CPU回归；不是历史“只读P0-01”。P0-02到P0-05仍LOCKED。
3. 本地先核对 branch/HEAD/dirty。当前分支头 `403b55db65f4c35c1a85d0794ad0de2765b07d96`，受审代码 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`；本轮核对二者源码树相同，前者仅增加审计材料。不要checkout/reset覆盖现有工作；漂移先对照，无法解释则BLOCKED。
4. 缺GPU/checkpoint/data/xformers/许可/cap时，完成安全且获准的CPU部分，写BLOCKED并停止。禁止自动下载、pip、全量数据扫描或训练。
5. P0-01即使PASS，也只写下一任务建议并停止。明确新授权+前序证据+对应资源cap后才能修改机器状态；**不要自行解锁**。

可先运行包完整性检查（不触碰仓库/模型）：

```bash
python tools/validate_package.py --root .
```

包内其它命令标明“待任务实现”的脚本必须先按主计划新增/patch，再运行；不要把不存在的CLI当成已完成实现。保留旧版本包和全部历史结果。
