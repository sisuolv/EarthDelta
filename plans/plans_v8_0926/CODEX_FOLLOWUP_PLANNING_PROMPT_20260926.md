# Codex 后续计划制定 Prompt（v8，2026-09-26）

用法：在本机 Codex 中打开 `/mnt/afs/260010168/EarthDelta`，把下面代码块的全部内容作为任务粘贴进去。本轮只**制定计划**并做只读核验，不改代码、不跑 GPU、不读新年份数据。

```text
你是 EarthDelta 项目的执行侧计划负责人。请参考 plans/plans_v8_0926/ 下两份 ChatGPT Pro 输出和仓库当前真实进度，制定一份可直接执行、需要人工签署的后续计划。用中文输出。

==================== 0. 本轮范围 ====================
本轮只做两件事：
(1) 只读/CPU 核验，用来确认计划所依赖的事实；
(2) 产出后续计划包（见第 6 节）。
本轮禁止：
- 修改 earthdelta/、scripts/、tests/ 下任何文件（计划里写清楚要改什么，下一轮再改）；
- 提交任何 ACP/GPU 作业，重跑 S0；
- 读取 2021、2022 的任何数组数值（包括有限性扫描）；2015–2020 也只读元数据和已有扫描记录，不做新的数值扫描；
- 修改、覆盖或移动任何历史产物，尤其是 plans/plan_v4_0923/run_20260924T104725Z_fp05b/ 下的文件；
- git commit / push（由人工决定）；
- 删除仓库根目录那个未跟踪的 `$OUT` 文件。
时间盒：半天。任何核验做不了就标 MISSING/BLOCKED 并继续写计划，不要为补材料无限延长。

==================== 1. 固定输入 ====================
仓库：/mnt/afs/260010168/EarthDelta（GitHub main）。
- 代码与实验证据基准：提交 9e91191（此前所有 dirty 修复、r7 脚本与测试都已在其中）。
- 之后的提交只改动 README.md、reviews/、plans/plans_v8_0926/。先用 `git log --oneline --name-status 9e91191..HEAD` 和 `git status --porcelain=v1` 确认；如有其他改动，列出后再继续。

v8 计划包（ChatGPT Pro 输出，只读）：
- 主合同：plans/plans_v8_0926/EarthDelta_Codex_Followup_Plan_20260926.zip
  sha256 a3b27da01e0205f2633963cf8ee3d62db628b6f5f63bbc8c21eaf503c3adfba7
  其中 CODEX_TASKS.json 是机器可读的唯一任务合同；先读 START_HERE_FOR_CODEX.md、FULL_REVIEW.md、FOLLOWUP_PLAN.md、STOP_CONDITIONS.md、ARTIFACT_CONTRACT.md、TEST_PLAN.md、REPRODUCIBILITY.md、evidence/REQUIRED_INPUTS.md。
- 辅助证据与备选计划：plans/plans_v8_0926/EarthDelta_Independent_Audit_20260926.zip
  sha256 a5180128391f100492a067be0d3b63df026f403ee33c44f73a6a89d7c27e45ad
  其中 next_stage_plan.json 与主合同在数据角色、样本量、阈值、预算上不同，见第 3 节 C1。
解压到你的运行目录，不要解压进仓库。

已有复查（背景，不是事实来源）：
- reviews/CLAUDE_INDEPENDENT_REVIEW_20260926.md 与 reviews/claude_independent_review_20260926/
- reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md（Codex）
- README.md 中 "Independent reviews (2026-09-26)" 一节
原始研究计划：plans/plans_v1_0919/EarthDelta_v6_Review_and_Plan_CN.md（§4.2 P1 门、§4.6、§4.7、§6）。

==================== 2. 当前真实进度（写计划时以此为起点；标注 [复核] 的请本轮再核一次） ====================
工程：
- 9e91191 上全量 CPU 测试 1004 passed / 7 skipped（2026-09-26 在该工作树上运行）。
- S0：官方无编辑前向 1/4/12 步 max_abs_diff=0，环境为 overlay torch 2.3.1+cu121 / xformers 0.0.27。S0 不覆盖 20 步（120h）路径、带 hook 的编辑前向和训练反传。
- 研究核心 earthdelta/{heads,teacher,probe,paired,selection}.py 从未在真实数据上运行。

FP-05b（冻结，只读）：
- 冻结协议下 FP-06 规则结论仍为 PIVOT_STATIC，不得改写；新的研究决策是停止扩大旧 K4 bank。
- 以 F0（原始 Stormer）为分母的标量重算，用 v8 主合同的 tools/independent_recompute.py --repo 在本机跑过（B=10000，seed=0），与 reviews/claude_independent_review_20260926/scalar_recheck.txt 一致 [复核]：
  M3 相对 F0：6h −0.211%，24h +0.0699%，72h −0.720%；
  oracle6_at24（含 F0 的 24h 事后选择）：−0.155% / +0.155% / −0.437%；
  oracle−M1（Fs 分母）0.0536%，CI [0.0419%, 0.0669%]；
  112 个起报中，F0 不差于全部 5 个候选的数量：6h 109，24h 31，72h 83。
  这些是旧 69 通道状态标准化平方损失的百分比，不是 RMSE 百分比。
- 按变量结果（Claude，基于已曝光的 2019H2 端点）：M1 在 Z500/T850/MSLP 的 24h、72h RMSE 劣于 F0。ChatGPT Pro 指出的修正必须采纳：
  CI 实际只用了 4000 次抽样中的前 1000 次；脚本没有绑定坐标、通道、std 的身份和哈希；"只有 50–100 hPa 风改善"说法过强（T2m@6h +0.13%、10m U@24h +0.06% 也为正）；"50–150 hPa 占 29%"不在已保存输出中；损失份额 ≠ 权重 ≠ 梯度份额。
  结论：这些脚本只能作线索，T01 要用正式的度量模块重做。中间文件 /mnt/afs/260010168/earthdelta_independent_review_20260926/per_channel_mse.npz 可作对账参考。

数据（只核元数据）：
- 2016、2017、2022 的 store 时间维长度为 0。
- 2015/2018/2019 前 736 步中非有限步数分别为 216/192/124，2019H2 为 88/724（见已跟踪的 finiteness_scan_*.json）；2018 下半年未扫描。
- 2020 形状完整（1464 步），index 712 有一个 q100 102σ 异常；未做全年内容扫描。
- 2021 形状完整（1460 步），内容未校验。
- earthdelta/data/pull_wb2.py 存在"假成功"路径：L1103 附近时间长度为 0 仍继续并写完成标记；L1147–1181 finalize 返回 verification_failed，但 L1396–1399 只打印，退出码 0；is_year_complete（L423–448）只查步数和通道数；L398 fsspec 映射未显式处理缺块。.pydeps 中 fsspec 版本 2026.9.0。拉取日志 checkpoints/pull_*.log 不在 git 中但在本机。
- 缺块成因（代理丢块被当成 missing key 填 NaN）只是推断，未证实。

环境：
- 本节点只有 CPU，Bash 有 8GB cgroup 内存上限，无法加载完整检查点。
- GPU 走 ACP：/mnt/afs/260010168/bin/sco，说明见 /mnt/afs/260010168/ACP-GPU-QUICKSTART.md；已验证 4×H100 规格 n6ls.iu.i40.4.32c512g。
- 需要 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python；依赖在 .pydeps 与 .pydeps_wbx。
- 代理 127.0.0.1:17890，GCS（WeatherBench2）经代理可达。

==================== 3. 必须给出结论并交人工签署的决策 ====================
每条写：选项、你的建议、依据（文件/数字）、对预算与时间的影响。

C1 两份 ChatGPT Pro 计划的差异。
   默认采用主合同（CODEX_TASKS.json）。逐项列出与 next_stage_plan.json 的差异：
   - 数据角色：fit=2018+2020Q1–Q3 对 train=2020；
   - DEV 样本量：192 对 128；
   - GO/STOP 阈值：对 F0 ≥2% 且比最强可部署基线多 ≥0.5%，对 比最强基线 ≥1.0%；
   - 预算：0/8/8/16 对 2/32/24 H100 卡时；
   - 主指标定义：主合同的"两项主指标"具体是什么？在 CODEX_TASKS.json / FULL_REVIEW.md 中找到原文；找不到就提出一个明确定义（例如 Z500@72h 与 T850@72h RMSE，或六变量归一化合成指标）并标为待签署。

C2 用 2018 做 fit 是否合适。
   2018 在 Stormer 预训练年份（1979–2018）之内，骨干在该年的误差不代表样本外；而且 2018 上半年约 26% 的步是 NaN，需要先修复。
   备选：2019（Stormer 选模年；在本项目中作旧 DEV 已曝光，但用于拟合不构成评估泄漏）+ 2020Q1–Q3。
   比较数据完整度、修复工作量、骨干样本内偏差、与旧结果的可比性，给出建议。

C3 2022（confirm）如何修复与认证，又不在 T04 前暴露数值。
   提出具体流程：例如只做源端对账（chunk 清单、字节数/哈希）而不打开数值；或者整批推迟到 T04 最终冻结之后，由独立流程下载并做完整性检查，在任何模型运行前记录。
   说明每种做法算不算"读取"。

C4 数据修复的范围：只修计划角色需要的年份与支持窗，不默认重拉 2015–2022 八年；估算网络时间。

C5 120h / 20 步路径：S0 只覆盖 12 步。把"20 步官方一致性 + 带 hook 编辑前向 + 训练反传方向"这组数值门（G0/G1）放在第一次 GPU 科学运行之前，并估算所需作业。

C6 Fs* 的训练损失：主合同要求两份事先冻结的训练损失——旧的 69 通道状态标准化损失，以及事先冻结的六变量多时效加权损失。
   不得把 Stormer 的差分 std 直接套到终点状态误差上，就称之为"官方损失"。给出两者的精确定义。

C7 成本：给出先测 t_roll / t_update / I/O 的最小计时方案，以及按 N×(2K+2+J)×t_roll 换算的各任务卡时上限。

==================== 4. 本轮允许的只读/CPU 核验 ====================
把命令、返回码、耗时和输出哈希都记录到 COMMANDS.log。
1) git：HEAD、分支、status、9e91191..HEAD 的改动文件列表。
2) 两个 v8 包：核对 sha256；运行主合同的 tools/validate_package.py；
   在 $RUN/scalar 运行 `python tools/independent_recompute.py --repo /mnt/afs/260010168/EarthDelta --out $RUN/scalar --draws 10000 --seed 0`，与第 2 节数字逐项对账（这一步关闭 REQUIRED_INPUTS 的 U1）。
3) 数据元数据清单（不读数值）：2015–2022 各 store 的 .zarray、.zattrs、time 坐标长度、chunk 文件数、完成标记；已有 finiteness_scan_*.json 的空洞形态（起止步、涉及变量、是否 8 步对齐）；pull_*.log 中 verification_failed、"0.00 GB fetched"、网络错误的计数；fsspec/zarr/xarray 版本。
4) U2–U5 证据清点：只确认文件是否存在并记录路径和哈希，不做新计算。包括 2019H2 端点目录 artifacts/round2_cci/ed-r4cachecj2-0924125702-926f58/、per_channel_mse.npz、normalize_std.npz、S0 收据（/mnt/afs/260010168/acp_runs/ 下）、外部执行目录（/mnt/afs/260010168/earthdelta_v7_execution_*、earthdelta_followup_execution_*）。
5) 为下一轮定位代码：列出 T01–T04 需要复用或修改的函数及行号，至少包括：
   - pull_wb2.py：时间长度断言、缺块严格报错、逐批 isfinite、原子完成标记、失败时非零退出码；
   - scripts/r5_audit_runner.py：formal 模式下 verified_bundle 必填、source/input 哈希重算、阈值只取一处；
   - earthdelta/wbx/evaluate.py：evaluate_single_chunk 的 valid_times 必填并绑定坐标；
   - scripts/r7_followup_decomposition.py：oracle 定义、硬编码的 stop_rule、单位口径；
   - 新的标准变量度量模块：6 变量 × 4 时效，先池化 MSE 再开方，绑定坐标/通道/std 哈希，B≥10000，多重比较控制；
   - 可复用的研究代码：earthdelta/teacher.py、probe.py（中心差分）、static_adapter.py（训练与目标）、bank_training.py、candidate_cache.py（rollout 与缓存）、bridge（20 步 rollout）、wbx（标准指标）。

==================== 5. 计划必须满足的原则 ====================
- F0 永远在候选集里，并且是主基线和分母；标准变量表（Z500/T850/T2m/MSLP/U850/Q700 × 6/24/72/120h 的 RMSE，可加 ACC）是主结果；旧 native loss 只作诊断。
- 顺序固定为：T01 角色冻结与最小测量/数据修复 → T02 静态 Fs*、EWMA/输出订正与 F0 对照 → T03 fit-only 固定方向、192 例 DEV 的真实非线性 headroom → 只有 GO 且人工批准才做 T04（e0/du 与一次 confirm）。
- 每个任务写清：目标、前置门、要改的文件与函数、要新增的测试、命令、输入/输出、数据角色与访问边界、成功/失败/停止规则、GPU 卡时与人日预算、需要的人工批准。
- 数据角色与支持窗（t−12h..t+120h，加 24h 隔离）必须在读取任何新年份数值之前签署；2022 在 T04 前零数组访问；不看技能挑年份。
- T03 的局部 QP 不是全局上界；oracle 与可部署效用分开报告；一个动作要作用于所有变量和时效，禁止把逐变量 oracle 拼成一个模型。
- 数据失败、求解失败、线性度失败属于技术 BLOCKED，不等于科学证伪；INCONCLUSIVE 不触发自动扩样。
- 审计只保留影响数据、度量和放行的最小修复；T01 时间盒 1 个工作日（网络等待另计），不再开审计专门轮次。
- 执行状态只用 OBSERVED / TO_BE_RUN / MISSING / BLOCKED；ENGINEERING_PASS、EXPERIMENT_EXECUTED、SCIENTIFIC_SUPPORT、NOVELTY_SUPPORT 四个结论分开写，不能从测试、rc、S0 推出后两项。
- 列出"不值得做"：旧 K4 上的任何 selector/阈值/HPO/fresh split 翻案；为凑材料重跑 S0 或追旧 500 步发散根因；冻结角色前修复/读取全部年份；JEPA/记忆/动态秩/谱损失堆叠；把 QBO 立即当主线；按全局 σ 阈值静默删除极端值；改写 FP-05b 冻结门槛与结论。

==================== 6. 输出（写到 $RUN=/mnt/afs/260010168/earthdelta_v8_planning_<UTC时间戳>/） ====================
1) PLAN.md：
   - 开头按"当前做到哪里 → 问题 → 已可信 → 未证明 → 接下来的顺序"五段总结；
   - 然后是 T01–T04 的完整计划（按第 5 节要求）；
   - 以及 GO/STOP 判定表和总预算。
2) TASKS.json：沿用 CODEX_TASKS.json 的字段，再增加 files_to_change、tests_to_add、approvals_required、evidence_status。
3) DECISIONS_FOR_HUMAN.md：C1–C7，每条给出选项、建议、依据，并留签署栏（默认 MISSING）。
4) ROLES_DRAFT.json（状态 PROPOSED，human_signature=MISSING）：
   - 各角色的年份与时间窗；
   - 192 个 2021 DEV 起报时间（只由日期规则生成，不读数据）；
   - 支持窗与隔离规则；
   - 缺失/替补规则。
5) METRIC_CONTRACT_DRAFT.json：
   - 变量与通道索引、时效、面积权重、RMSE 池化方式；
   - std 来源及哈希；
   - 块定义与 bootstrap 次数；
   - 多重比较控制（Holm 或同时区间）；
   - 主指标与 GO/STOP 阈值（标明是资源决策阈值，不是行业常数）。
6) VERIFICATION.json：第 4 节每项核验的结果与状态，以及 U1–U6 的更新状态。
7) CORRECTIONS.md：Claude 与 Codex 两份复查中被 ChatGPT Pro 修正、且计划已采纳的说法。
8) COMMANDS.log 与 FINAL_MANIFEST.json（所有输出的 sha256）；另把最终的 PLAN.md、TASKS.json、DECISIONS_FOR_HUMAN.md 复制一份到 plans/plans_v8_0926/codex_plan_<UTC时间戳>/（不提交）。

最后在对话里简要回复：你采纳和偏离了主合同的哪些地方、C1–C7 的建议、下一轮 T01 需要的人工批准清单、以及本轮做不了而标为 MISSING/BLOCKED 的事项。
```
