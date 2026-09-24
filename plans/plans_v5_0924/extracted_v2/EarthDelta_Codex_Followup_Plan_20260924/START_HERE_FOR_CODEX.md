# EarthDelta 下一轮执行入口（2026-09-24）

先读 `evidence/PLAN_AUDIT_RESULT.json`、`CURRENT_GAP_ANALYSIS.md`、`FOLLOWUP_PLAN.md`、`EXECUTION_DAG.md`、`STOP_CONDITIONS.md`、`CODEX_TASKS.json`，再读 `ARTIFACT_CONTRACT.md` 与 `TEST_PLAN.md`。

审查代码固定在 `e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`；实际分支HEAD `08093650ba56e8cf709d75a61c26c50a0c35f6f7` 仅新增审查prompt。工作目录HEAD若更晚，先记录diff，不回退或覆盖用户变更。

**现在只执行 FP-05a：CPU只读证据与曝光账本核对。** 初始授权在 `evidence/INITIAL_AUTHORIZATION.json`。先运行：
```bash
python "$PKG/tools/validate_package.py" --package "$PKG"
python "$PKG/tools/read_only_preflight.py" --repo "$REPO" --out "$RUN/entry/evidence_inventory.json"
```
`PKG` 为本包解压根目录；`REPO` 为已有私有checkout；`RUN` 必须是全新证据目录，不能覆盖历史run。缺文件可以写MISSING清单，不得伪造PASS。

## 已完成与尚未完成
- FP-00 inventory、FP-01指定配置S0、FP-03 v2 Fs资格、FP-04 common-Fs bank认证有真实计算记录，保留 `DONE`，不要重训。
- FP-02 **PARTIAL**；STOP/数据消费子集已做，profile不可变性在FP04补齐，但全局split/薄runner等仍缺。由FP05a/b补残留，不重新编号整条计划。
- raw平台 `job_result.json` 与大权重本次未直接取得；读已有挂载并核hash即可，不因附件缺失自动重跑GPU。
- **X1 N32/N64 已使用2020年7–12月训练数据，2020H2不是未触碰confirm。** 不能只记录最终选中的Fs/bank的曝光。
- `delta_min=0.0034` 只能是待冻结的Fs-relative工程筛查；不是新DEV上F0盈亏平衡的已知常数。

完成FP05a后保存 `entry/decision.json` 和缺失清单并停止；没有明确的下一节点解锁和资源批准，不启动FP05b以后的运行。任何身份/数值/分母/split失败均阻断后继。FP06只在完整FP05数据后做DEV决策，绝不自动读取最终confirm。
