# EarthDelta 独立复查（Claude，2026-09-26）

**复查人：** Claude Code（模型 `claude-opus-5-5`）
**复查日期：** 2026-09-26
**复查对象：** 分支 `audit/round2-review-20260921`，HEAD `e4da7aa`（= origin，比 `8ab914b` 只多两份 review 文档）+ 本地 dirty 工作树（10 个已修改文件、未跟踪的 `scripts/r7_*` 与 `tests/test_r7_*`）
**配套：** 同日 Codex 复查见 [`EARTHDELTA_INDEPENDENT_REVIEW_20260926.md`](EARTHDELTA_INDEPENDENT_REVIEW_20260926.md)；两份的分歧见文末。
**复算脚本与输出：** [`claude_independent_review_20260926/`](claude_independent_review_20260926/)

---

## 结论先行

- **进度**：工程部分基本可用（Stormer 桥接、数据读取、候选缓存、OOF 评估）。科学部分只做了一个很小的实例：Fs 只用 8 个起报训练 32 次更新；4 个 rank‑4 专家各 6–16 个起报；评估只有一次，在 2019 年下半年 DEV 集上（112 个起报）。原计划的核心方法——从合法历史预测 e0 和 du，再用 2⟨e0,du⟩−‖du‖² 在预算内选编辑——**从来没有在真实数据上跑过**。
- **最重要的新发现**（独立复算，此前几轮审计没有指出）：改用原始 Stormer（F0）作基线后，所有编辑方案（Fs、M1、M3，甚至事后 oracle）在 6h 和 72h 都显著比 F0 差，只在 24h 好 0.07%。按标准变量看，M1 让 Z500、T850、MSLP 在 24h 和 72h 都变差，唯一变好的是 50–100 hPa 平流层风。
- **推论**：「PIVOT_STATIC / 保留 M1」不成立。当前最强基线是 F0（不做编辑）；当前 bank 应判为停止（STOP_CURRENT_BANK）。但这个负结果也**不能证伪** EarthDelta，因为它测的不是 EarthDelta 方法。

## 0. 复查方式

- 全量 CPU 测试（dirty 工作树）：**1004 passed，7 skipped**（[日志](claude_independent_review_20260926/pytest_full_dirty_tree.log)）。这只说明代码合同自洽，不代表科学结论成立。
- 独立复算（只读，未修改仓库文件，未跑 GPU）：
  1. 用 FP‑05b 缓存的 1680 条损失 + F0 背景损失，以 F0 为基线重做配对 7 天块 bootstrap（[`scalar_recheck.py`](claude_independent_review_20260926/scalar_recheck.py) → [输出](claude_independent_review_20260926/scalar_recheck.txt)）。
  2. 从已封存的 DEV 端点数组（`artifacts/round2_cci/ed-r4cachecj2-0924125702-926f58/s*/issue_*/{endpoints,f0_endpoints,truth}.npy`）重算 112 起报 × 6 模型 × 3 时效 × 69 通道的面积加权 MSE；与存储损失的相对误差 ≤3.5e‑9（[`per_variable.py`](claude_independent_review_20260926/per_variable.py)、[`per_variable_report.py`](claude_independent_review_20260926/per_variable_report.py) → [输出](claude_independent_review_20260926/per_variable_report.txt)；[`per_variable_oracle.py`](claude_independent_review_20260926/per_variable_oracle.py) → [输出](claude_independent_review_20260926/per_variable_oracle.txt)）。中间文件 `per_channel_mse.npz`（sha256 `e7199fbe…43dcb33`）被 `.gitignore` 忽略，保存在仓库外 `/mnt/afs/260010168/earthdelta_independent_review_20260926/`，可由 `per_variable.py` 重新生成。
  - 数据边界：这些是 2019H2 policy_dev 数组，标签早已被 FP‑05b 用过；未触碰 confirm、holdout、2021、2022。v7 执行轮刻意没有读这些数组，这是本次与之不同之处。
- S0 与训练细节来自已有收据和预注册文件，本次未重跑。

## 1. 计划执行情况（对照原 v6 研究计划）

| 计划项 | 状态 | 说明 |
|---|---|---|
| S0 桥接与官方一致性 | 完成 | 1/4/12 步 `max_abs_diff = 0`（torch 2.3.1 + xformers 0.0.27）。dirty 的 `stormer_bridge.py` 只加并发锁，无数值改动。「复现官方 RMSE 量级」未真正做：`scripts/s0_gate.py:209-210` 的参考值是粗估（Z500@24h 写 80，实际 F0 ≈ 51 m²/s²） |
| S1 数据（2015–18 训练 / 2019 验证 / 2020 测试） | 部分完成且有缺陷 | 只有 2020 完整；2021 未做内容校验；2015/2018/2019 有 12–29% 时间步为 NaN；2016/2017/2022 为空。原定测试年 2020 被用于训练（manifest 中 `split_id` 仍为 `test`） |
| S2 Fs 与异质字典（rank 16，K=6–8，天气型/误差模式专家） | 偏离 | Fs rank 4、8 个起报；专家 K=4、rank 4、按日历月份分组、系数 0.25 |
| S3 探测缓存 + P1 headroom 门 | 部分 | 有限候选缓存完成；中心差分 R 与线性度诊断未做。P1 门（原定 Z500@72h ≥2–3%）从未按原定义评估；按原定义明确不通过（见 §2） |
| S4 控制器（e0/du 头、B2、D0/D1、U0/U1） | 未做 | M3 是 ridge 直接 gain 回归，原计划中是**基线**而非 EarthDelta 方法 |
| S5 迁移、标签效率、可诊断性 | 未开始 | 原计划论证新意的主表 |
| S6 WeatherBench‑X 标准评测 | 部分 | 只与项目自定义目标函数逐位对账；从未产出 Z500/T850 等标准 RMSE/ACC 表 |
| 必做基线 | 大部分缺失 | 缺 EWMA 偏差订正、同信息输出订正器、高容量静态 LoRA、真正的 B2 router |

近期 v7 工程任务（FP‑06a–d、WBX‑02d、FP‑07d、FP‑08c）基本按计划执行，未越过 GPU/confirm 边界，但产物依赖未提交代码。Codex 复查对这部分的描述经复核属实，此处不重复。

## 2. 发现的问题

### 基线与指标（最严重）

1. **所有比较锚定在一个本身有害的 Fs 上。** 预注册写明 F0 "never a candidate, never in the oracle"（`plans/plan_v4_0923/run_20260924T104725Z_fp05b/protocol/PREREGISTRATION.md:108`）；oracle 只在 5 个候选里选（`earthdelta/policy_oof.py:400`）；72h 护栏相对 Fs（`earthdelta/policy_oof.py:448`）。以 F0 为基线（目标函数相对变化，正数 = 优于 F0，方括号为 95% 块 bootstrap 区间）：

   | | 6h | 24h | 72h |
   |---|---|---|---|
   | Fs | −0.27% [−0.29, −0.25] | −0.19% [−0.23, −0.15] | −1.16% [−1.38, −0.94] |
   | M1（恒选 expert_0） | −0.22% [−0.24, −0.19] | +0.07% [+0.03, +0.11] | −0.69% [−0.89, −0.50] |
   | M3（ridge 路由） | −0.21% | +0.07% [+0.03, +0.11] | −0.72% [−0.92, −0.51] |
   | 事后 oracle（按 24h 选） | −0.21% | +0.12% [+0.08, +0.17] | −0.67% |

   6h 时，112 个起报中 Fs 无一优于 F0，每个专家最多 1 个。

2. **目标函数对标准变量几乎失明。** `earthdelta/static_adapter.py:343-352` 用状态标准差 inp_std 归一化、69 通道等权。F0 的 24h 目标中：比湿 49.5%，风 45%，位势 + MSLP + T2m + 温度合计不到 3%，50–150 hPa 三层占 29%。Stormer 自身训练损失按差分标准差归一化并按气压加权（`reference/stormer/stormer/utils/data_utils.py` 的 `WEIGHT_DICT`），50–150 hPa 约占 4%，地面变量约占 21%。Fs 与专家用这个错配目标训练，学到的主要是修正平流层风和上层湿度。

3. **按标准变量看，当前编辑让 Stormer 变差**（RMSE 变化，正数 = 优于 F0）：

   | 变量 | M1 24h | M1 72h | Fs 72h |
   |---|---|---|---|
   | Z500 | −0.29% [−0.43, −0.16] | −0.98% [−1.20, −0.75] | −1.31% |
   | T850 | −0.19% [−0.23, −0.15] | −1.17% [−1.29, −1.05] | −1.42% |
   | MSLP | −0.55% [−0.68, −0.40] | −1.26% [−1.59, −0.88] | −1.53% |
   | T2m | −0.08% [−0.19, +0.04] | −0.90% [−1.07, −0.74] | −1.08% |
   | U50（平流层） | +1.28% | +1.19% | +1.32% |

   - 24h 的 +0.07% 净收益几乎全部来自 v50、v100、u50。
   - M1 只在 21/69 个通道（24h）和 7/69 个通道（72h）上优于 F0。
   - 每个变量单独做事后最优选择，Z500@72h 仍为 −0.54%；把 F0 放进候选也只有 +0.33%，比原计划 P1 门的 2–3% 小一个数量级。

### 实验设计

4. **训练规模太小。** Fs 只用 8 个起报（2020 年 1–6 月，约每 24 天一个）× 32 次更新：样本内 24h 损失降 2.6%，2019H1 holdout 反而有害 0.34%；合格门槛 Q6 允许最多 +2% 伤害，所以一个样本外有害的适配器被「认证」。专家在 2020 年 1–6 月训练，在 7–12 月评估，季节不匹配。
5. **路由特征太粗。** `earthdelta/candidate_cache.py:43` 设 `FEATURE_GRID=(2,4)`：每通道只有 8 个半球/象限平均值，降到 8 维 PCA，每折约 80 个样本，几乎不含流型信息。
6. **核心研究代码从未运行。** `heads/teacher/probe/paired/selection/geometry/spectral/memory` 共 1,930 行，没有任何 script 导入，只出现在单元测试中。

### 数据

7. **2016、2017、2022 三年数据为空**（time 维长度 0）。拉取日志 `0.00 GB fetched` 后仍写完成标记；finalize 返回 `verification_failed` 但退出码为 0，外层脚本记录 "all years complete"。位置：`earthdelta/data/pull_wb2.py:1103` 未断言时间步数等于预期；`:1181`、`:1398` 未把失败状态转为非零退出码。
8. **2015、2018、2019 有大段 NaN 空洞。** NaN 时间步：2015H1 216/736，2018H1 192/736，2019H1 124/736，2019H2 88/724。空洞为 8 步对齐的连续段、覆盖整个压力层变量，符合代理丢块时 fsspec 把缺块当 missing key、zarr 填 NaN 的机制（`pull_wb2.py:398` 未设 `missing_exceptions=()`）。这是推断，但特征高度一致。另有 2020 年 index 712 的 q100 102σ 异常值。**目前没有合法的 fresh/confirm 数据。**

### 代码与流程

9. 最新结果依赖未提交代码；仓库根目录有一个误生成的未跟踪文件 `$OUT`（shell 变量未展开，内容为 2022 校验失败记录）。
10. FP‑06 runner 仍可绕过：`evidence_valid` 由调用方声明（`scripts/r5_audit_runner.py:30`）；不带 `verified_bundle` 就跳过全部检查（`:31`）；source manifest 只查类型不重算哈希（`:37`）；bundle 阈值（`:39`）可与命令行 `--delta-min`（`:69`）不一致。WBX 的 `valid_times` 仍可省略（`earthdelta/wbx/evaluate.py:276, 291`）。这些属实，但科学影响有限：关键数字都能从缓存标量直接复算。
11. **流程风险**：7 天内 10 个计划目录、约 800 个计划/审计文件、基础设施 + 审计 + 测试代码约 5 万行；最基本的科学检查（对 F0 比、按标准变量看）一直没做。

## 3. 当前结果能支持哪些结论

| 结论 | 判断 |
|---|---|
| 桥接与官方实现数值一致；FP‑05b 数字可复算 | 可信（工程层面） |
| M3 路由不优于 M1（M3−M1 = −0.0003%，区间跨 0） | 可信，仅限 2019H2 DEV |
| 当前候选集对原始 Stormer 有害 | 可信（本次新增） |
| 「保留 M1 作为默认」 | 不成立，F0 更好 |
| 「72h 护栏通过」 | 误导：仅相对 Fs；相对 F0 为 −0.7%，区间全负 |
| R7 REPORT「瓶颈是 selector 误差而非缺少 headroom」 | 不成立：oracle−M1 上界仅 0.067%，远低于 0.34% |
| Codex 复查建议的 fresh split 上 selector‑only 干预（成功标准 M3 vs Fs ≥0.34%） | 不值得做：需超过 oracle 点估计 0.315%，按设计几乎只能失败 |
| EarthDelta 假设无效，或有效、有新意 | 都未证明——从未测试 |

测试数、S0 diff=0、receipt 齐全都只是工程事实。

## 4–5. 主要问题与解决方法（按优先级）

**P0**
1. **基线与指标**：F0 作唯一主基线和分母，候选集永远包含 F0。每个结果附标准变量表：Z500、T850、T2m、MSLP、U850、Q700 在 6/24/72/120h 的 RMSE 与 ACC，配对块 bootstrap，直接用 WeatherBench‑X 官方 RMSE。当前 bank 改记为 STOP_CURRENT_BANK。
2. **目标函数**：改为与 Stormer 训练一致的损失（差分标准差归一化 + `WEIGHT_DICT` 气压权重），或在预注册中明确标准变量加权。
3. **数据管线**：拉取前断言时间步数；fsspec 设 `missing_exceptions=()`；每批写入后检查 isfinite、失败重取；校验通过才写完成标记；finalize 失败时退出码非零。然后只重取空洞，完整重拉 2016/2017/2022，并为所有年份出内容证书（有限性、逐通道稳健 z 分数、抽样与源数据比对）。
4. **训练规模**：数百到上千个起报、数千次更新、验证集早停；Fs 合格标准改为「标准变量上不比 F0 差」。

**P1**
5. 数据修好后、看任何技能数字前冻结数据角色（例如训练 2019+2020，dev 2021，confirm 2022）。
6. 路由特征改用冻结主干 token 池化特征 + 按 available_time 过滤的近期已核验误差。
7. 把 dirty 工作树一次性提交到新分支；确认后删除根目录 `$OUT`。

**P2**
8. FP‑06、WBX、erratum 的剩余口子各做最小修复，不再单独开审计轮。
9. S0 的 RMSE 参考值换成同协议真实数字，或删除。

## 6. 下一阶段计划

T1 与 T2 可并行，然后 T3；T3 通过才做 T4。

**T1：更正基线与指标，归档代码（<1 天，CPU）**——把本次逐变量分析做成仓库内脚本；更正 README、FP‑06 记录和 R7 REPORT 的结论；提交代码。成功标准：仓库内脚本复现本报告数字。

**T2：修复数据并冻结数据角色（1–3 天，主要是网络）**——成功：2015–2022 每年 100% 有限、无无法解释的 >15σ 值，数据角色在读取技能数字前冻结。退路：换 CDS/AWS 0.25° 源 + 同样的保守重网格；再不行就收缩到 2019–2021 并声明没有 confirm。

**T3：正确版 headroom 实验（原计划 P1 门，约 10–30 卡时）**——三个臂，全部以 F0 为基线：
- (a) 充分训练的静态 LoRA Fs*；
- (b) 每起报事后最优连续编辑：在 8–16 个编辑方向上用中心差分求 R，解盒约束 QP（`earthdelta/teacher.py`、`earthdelta/probe.py` 已有代码）——状态依赖编辑的真实上界；
- (c) F0 + 滞后已核验误差的 EWMA 偏差订正（原计划必做基线，同时量化 e0 可预测部分）；平流层风是目前唯一有稳定正信号之处，可作观察点。
- **Go**：(b) 相对 max(F0, Fs*) 在 Z500@72h 与 T850@72h RMSE 改进 ≥2%、区间不含 0、明显优于 (c)。
- **Stop**：<0.5% → 停止在 Stormer + 常规年份上做 EarthDelta，转向原计划 §6 的替代设置（分布偏移年份、极端事件、更弱主干、更长时效）或收尾。0.5–2% 只允许一次低成本可预测性探针。
- 成本参考：FP‑05b 每起报 5 候选 + F0 约 7 秒；8 个方向中心差分、500 个起报约 3 卡时。

**T4：真正检验 EarthDelta 假设（仅当 T3 为 Go）**——训练 e0 头和 du 头，比较响应路线（预测 b、H 再解 QP）、直接 gain 回归、直接系数回归、最佳静态、EWMA；dev 年调参、confirm 年只评估一次，再做迁移实验（P3）。成功：响应路线回收 ≥30% oracle headroom，且在标准变量上相对最佳静态和 EWMA 区间不含 0。失败：<10% → 写成边界清楚的负结果并停止。

**流程约束**：T3 出结果前，暂停新的 ChatGPT/Codex 审计轮和计划包。

**暂时不值得投入**
- 当前 K4 bank 上的任何 selector 工作（阈值、超参、换特征、fresh split 上的 selector‑only 干预）：上界仅 0.054%，oracle 在 6h/72h 比 F0 还差。
- 把 2021/2022 分配给当前 bank 当 confirm。
- 在旧目标函数上或只用 8–48 个样本训练新专家；增加 K、新 router、JEPA/记忆/谱模块。
- 继续加固信任门、补跑 Tier‑A receipt、重跑 S0、分析旧 500 步 Fs 发散根因。

## 最终结论

- **做到哪里**：桥接和评估框架可用（S0 通过，1004 个测试通过）；科学上只评估了一个样本很少、目标函数错配的 K4 bank。
- **问题在哪**：基线锚在有害的 Fs 上、F0 被排除；目标函数里 Z/T/MSLP/T2m 权重合计不到 3%；训练样本仅 8–48 个；数据有空年份和大段空洞，管线把失败报成成功；大部分精力花在审计工具上。
- **已可信**：桥接数值一致；数字可复算；当前 bank 与路由没有价值——准确地说，让 Stormer 在 Z500、T850、MSLP 上变差。
- **仍未证明**：任何编辑（静态或动态）能改善 Stormer；e0/du 可从历史预测；迁移能力、跨年份泛化、新意。
- **解决顺序**：T1 ∥ T2 → T3（Go/Stop）→ 仅 Go 时做 T4。T3 若 Stop，就结束在 Stormer 上做 EarthDelta 这条线，而不是继续打磨 selector 或审计工具。

## 附：与 Codex 复查（同日）的异同

| 议题 | Codex 复查 | 本复查 |
|---|---|---|
| GitHub HEAD 缺 dirty 修复；FP‑06 runner / WBX / erratum 门的口子 | 指出 | 复核属实，但定为 P2（关键数字可直接复算） |
| 当前最强基线 | M1/static | **F0**（M1 在标准变量上劣于 F0） |
| 72h 护栏 | 「通过」 | 仅相对 Fs；相对 F0 为 −0.7% |
| 下一步 | fresh split 上的 selector‑only 干预 | 不值得做；先修数据与指标，再做正确版 headroom 实验 |
| 是否否定动态编辑路线 | 否（因样本小） | 否（因核心方法从未被测试，且当前 bank 训练不足、目标错配） |
