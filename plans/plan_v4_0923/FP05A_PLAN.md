# FP-05a：核实 ChatGPT Pro 的复核发现，重建完整曝光台账，产出修订版 FP-05 协议草案

> 本文件是 ChatGPT Pro 对 FP-00~FP-04 执行情况及 `FP05_PLAN.md` 的独立复核之后，
> 修订出的下一步实现计划。复核材料见 `plans/plans_v5_0924/`（两份 zip，内容不完全
> 相同，均为真实交付物）。FP-05a 本身完全是只读、纯 CPU 工作，不跑任何 GPU 作业，
> 不碰任何 `.zarr` 数据，不修改任何已认证的核心文件。

## Context

ChatGPT Pro 对 FP-00~FP-04 的执行情况和 `FP05_PLAN.md` 做了独立复核，交回了两份不完全相同的 zip。两份都独立发现了同一个**真实、重要的错误**：`FP05_PLAN.md` 里"2020 年下半年是完全未触碰的 confirm 预留区"这个判断是错的——已从原始准入记录独立核实：FP-03 的探索性诊断（X1_N64）确实用 2020-07 到 12 月的真实数据训练过。精确区间运算的结论是：缓冲后 2020 下半年只剩 156 小时的空隙、最多能塞下 2 条互不重叠的数据，根本撑不起一个 confirm 集。这个判断必须撤回。

核实过程中还发现了两份 ChatGPT Pro 报告都没注意到的问题：另一条独立的"shared-F0"研究支线曾经对 2018/2019/2020 共 616 条数据做过内容完整性读取（其中 147 条落在 2019 下半年，正是原计划打算用作 policy_dev 的窗口）——虽然从未有任何梯度/模型结果真正用过这些数据（那条支线在 S0 门就被真实拦停了），但必须如实记入曝光台账。以及：2019 年下半年的数据完整性扫描目前根本没做过（现有扫描只到 2019-07-03），这是 FP-05b 真正建 policy_dev 准入前必须先补的一步。

**诚实的结论**：目前全项目范围内，没有任何一段数据是真正"从未被以任何方式触碰过"的干净 confirm 集。2019 下半年是唯一"不在 backbone 预训练年份内、且没有被任何模型结果曝光过"的半年，但它现在必须被用作 policy_dev（开发/调参），不能同时又当 confirm（最终一次性验证）。未来如果要有真正的 confirm，大概率需要拉取新数据（如 2021 年，Stormer 自己的 shift_test 年）——这是超出 FP-05a 范围的策略决定，先记录事实，不在本阶段处理。

## 两份 ChatGPT Pro 交付物的取舍

**以 v2（`plans/plans_v5_0924/extracted_v2/`）为主**：时间戳更晚（关键文件 08:55 vs 08:49，像是一轮修订）、证据结构更完整（`FINDINGS.json`、`JOB_EVIDENCE_REGISTER.json`、带真实哈希的 `EXPECTED_INPUTS.json`）、工具更严谨（`read_only_preflight.py` 真的校验字段并拒绝覆盖）、X1 区间运算和 delta_min 定义更精确。

**从 v1（`extracted_v1/`）合并进来的有价值内容**：`FULL_REVIEW.md` 里关于"验证折毒化测试必须限定在该折自己的 OOF 预测范围内"和"14 天区块只有约 12-13 个、独立性不保证"这两条提醒；`PROPOSED_PROTOCOL.json` 里"未冻结字段一律记 null"的纪律；具体交付文件命名（`exposure_reconstruction_report.json`、`deviations/FP05_AMENDMENT.json`）。

**两份意见不一致、已经决定的地方**：
- GPU 作业上限：v1 说 4，v2 说 3。留到 FP-05b 冻结协议时再定，倾向采纳 v2 的 3。
- FP-05a 范围：v1 想现在就建薄封装 CLI；v2 主张本阶段只用只读工具、新脚本放 run 目录。**采纳 v2**。
- 统计细节预填数字：v2 模板预先填好一些数字，v1 全部留空。**本阶段一律标 `PROPOSED_NOT_FROZEN`**，不当作已经决定。
- 曝光标签用词：v1 `EXPOSED_NOT_CLEAN_CONFIRM` vs v2 `EXPOSED_NOT_CONFIRM`。**采纳 v2**。

## 已核实的关键事实

- FP-04 认证时 pin 住的 17 个核心源文件，逐一哈希比对，17/17 和当前工作区一致，没有漂移。
- 两份 zip 自带的校验工具都能通过；`evidence/EXPECTED_INPUTS.json`（v2）列出的 8 个哈希全部和本地文件核对一致。
- 之前认为"缺失"的原始平台记录其实都在本地 `artifacts/round2_cci/<run_dir>/` 下，只是没推 GitHub（`.gitignore` 排除）——FP-05a 是本地核对，不是找用户要文件。
- X1_N64 新增的 56 条 2020 下半年数据，支持窗口连续覆盖 2020-07-13T06Z 到 12-31T18Z；缓冲后 2020 下半年唯一剩下的空隙是 2020-07-05T18Z 到 07-12T06Z（156 小时），最多容纳 2 条互不重叠的候选。
- delta_min=0.34% 精确重算为 0.0033925584（F0 分母）/ 0.0033810878（Fs 分母），是 Fs 认证时 2019 上半年 16 条留出集上量出来的数字，不能在看到新 DEV 结果之前"重新推导"（那本身就是偷看）。保留作为预先声明的工程筛查线，要求同时报告 `H_Fs`（Fs 相对 F0 的样本外伤害）和 `G_F0 = G_Fs − H_Fs`。

## 修订后的数据角色设计（替换 `FP05_PLAN.md` 的判断）

- **已曝光（梯度/结果）**：2020-01-01T12Z~07-04T18Z ∪ 2020-07-13T06Z~12-31T18Z ∪ 2019-01-01T12Z~07-04T18Z（含缓冲）。
- **policy_dev 候选池**：`[2019-07-06T12Z, 2019-12-28T18Z]`（6 小时网格，理论上限 702 个时间点），标注"候选池，完整性未扫描"；披露其中 147 个时间点曾被另一条研究支线做过内容完整性读取（无梯度、无模型结果），标注为"回顾性迁移评估"。
- **2020 下半年**：标注 `EXPOSED_NOT_CONFIRM`。
- **confirm**：标注 `UNASSIGNED_NO_ACCESS`；下游消费者看到此标签必须拒绝读取；不自动拉 2021 数据。
- 2015/2018 标注 backbone 预训练年份不可用；2016/2017 标注空存储不可用；2021/2022 标注本地不存在。

## FP-05a 具体任务（只读、纯 CPU，不碰 `.zarr`、不跑模型前向、不开 GPU）

Run 目录：`plans/plan_v4_0923/run_<UTC>_fp05a/`。

1. 跑两份 zip 自带校验工具；跑 v2 的 `read_only_preflight.py --hash-large-assets` 做证据清单。
2. **作业回执核对**：全部真实 GPU job（S0、Fs v1/v2 的 J1-J3b、X1 的 N32/N64、FP-04 的 B-J1/B-J2/B-J3/retry，共 11+ 个）逐一核对 `job_result.json` 等文件，产出 `job_receipts_index.json`。
3. **认证证书复算**：从 `panel_initial_final.json`/`qualification.json`/`bank_verify.json` 独立重算 Fs 和 bank 的关键数字，允许跑一次 `earthdelta.registry.verify_bank_bundle(deep=True)` 做只读重哈希。
4. **完整曝光台账重建**：并入 X1、FP-03、FP-04、历史作业、S0 输入、shared-F0 支线的 616 条读取、三年完整性扫描，产出 `exposure_ledger.json` + `exposure_reconstruction_report.json`。
5. **FP-02 剩余缺口清单**：确认 `scripts/plan_followup_runner.py`、`earthdelta/split_freeze.py` 仍不存在，列成 `fp02_remaining.json`。
6. **修订版协议草案（不冻结）**：`protocol/fp05_protocol.DRAFT.json`，`status: DRAFT_NOT_FROZEN`，统计细节留 `PROPOSED_NOT_FROZEN`/`null`。
7. `entry/decision.json` + `task_result_FP-05a.json` 收尾。

**通过标准**：17 个 pin 哈希、8 个 tracked 输入哈希全部匹配；每个 job 回执字段对上；认证数字重算精确到 1e-12 以内、组装等价性精确为 0；台账包含全部 56 条 X1 下半年数据和 shared-F0 的 616 条读取；2020H2 标 `EXPOSED_NOT_CONFIRM`；confirm 为 null；协议草案通过 v2 校验工具基本结构检查。

**中止条件**：pin/哈希不匹配 → 证据无效，隔离停止；回执缺失 → 记 MISSING，整体判 BLOCKED；新脚本导入数据读取库或跑模型前向 → 违规；试图命名具体 confirm 窗口、扫描 2019 下半年完整性、或把未冻结参数填成数字 → 立即停止。

## 本阶段不做的事

任何真实 GPU 作业；FP-05b 及之后阶段；FP-06；读取或指定 confirm 窗口；拉取 2021 数据（数据下载是独立并行任务，不属于 FP-05a）；对 2019 下半年做完整性扫描或正式 policy_dev 准入；重新训练或重新选择 Fs/专家/bank；修改 17 个 pin 文件；冻结 FP-05 正式协议。

## 已确认接受的判断

1. Run 目录沿用 `plans/plan_v4_0923/` 既有惯例。
2. 允许在 FP-05a 里跑一次 `verify_bank_bundle(deep=True)` 做 CPU 只读重哈希。
3. shared-F0 支线对 2019 下半年 147 个时间点的内容完整性读取，记为"已披露但不禁用"。
4. GPU 作业上限（3 还是 4）留到 FP-05b 冻结协议时再定。
5. 是否拉取新数据建立真正的 confirm 集，是更大的策略问题，本阶段只记录事实。
