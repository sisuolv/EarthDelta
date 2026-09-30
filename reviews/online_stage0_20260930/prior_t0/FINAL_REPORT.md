# 任务 T0 完成情况

本轮完成了已授权的复用审计，**T0 全部验收尚未齐备**：A2a 未签，arch 的 5 项实际接口检查未运行；A2b 未签，TAFAS/OnlineTSF 未克隆，不能核实它们的固定版本、license 和设计。本轮停在 T0，没有启动 T1/T2 或任何 GPU 任务。

主要产物：[REUSE_DECISIONS.md](../../../plans/plan_v9_online_20260929/REUSE_DECISIONS.md)、[未应用的 t0_reuse.patch](../../../plans/plan_v9_online_20260929/t0_reuse.patch)、[VERIFICATION.json](VERIFICATION.json)。运行记录中的相对路径以仓库根目录为基准，`RUN` 指本报告所在目录。

## 改动文件

| 文件 | 类型 | 用途 |
|---|---|---|
| `plans/plan_v9_online_20260929/REUSE_DECISIONS.md` | 新增 | 逐组件决定 import / pip / 只读参考 / 新写，列出 license、接口证据、限制、T2–T5 影响和待签事项 |
| `plans/plan_v9_online_20260929/t0_reuse.patch` | 新增、未应用 | 建议修订 T2、T3A、T3B、T4、T5 的任务文字，未改代码或门槛 |
| `runs/online/t0_reference_inventory_20260929T114614Z.json` | 新增、不提交 | reference 目录、HEAD 和 license 逐项盘点 |
| `runs/online/t0_reuse_audit_20260929T114614Z/` | 新增、不提交 | 输入副本、版本与来源、合成测试、命令/返回码/耗时/哈希、结果和本报告 |

实际分支为 `probe-headroom-20260927`；起止 HEAD 都是 `d55ad70854bcf535b86c761b8d9cf7a7981374a0`。已跟踪文件无改动；149 个保护文件前后哈希一致，包含 earthdelta/scripts/tests 的 Python 源码、reference 清单/README、pyproject 和 `$OUT`。未创建 batch 5 脚本，未暂存、提交或推送项目。输入 ZIP 哈希未变。

开始时目标计划目录和根目录 `CLAUDE.md` 不存在，因此从用户指定 ZIP 读取了完整方案，副本仅存于 `RUN/inputs/`；没有擅自部署 CLAUDE、授权表或原始任务文档。补丁以该 ZIP 的五份任务原文为基准，`PATCH_BASES.json` 记录哈希。

## 各步骤状态

| 步骤 | 结果 | 未完成部分 |
|---|---|---|
| 1. reference 盘点 | 清单内 46 项：缺失 0、版本不一致 0、不可核验 0；缺顶层 license 文件 6 | 本轮仅核验清单版本与 license 存在性，不代表全面功能审查 |
| 2. batch 5 | 因 A2b 未签而跳过 | TAFAS/OnlineTSF 的 HEAD、日期、license、5 项设计要点待实际克隆后核验；不包含在原 46 项里 |
| 3. pip 依赖 | 完成具体版本、用途、Python/依赖约束、license 清单；保留已有工作依赖 | A2a 未签，没有安装。arch 及其 statsmodels/patsy 依赖缺失 |
| 4. 接口核验 | sklearn、WBX/v8/手写公式、memory/LoRA 和负例完成；66 个源码符号均已定位 | arch 的 StationaryBootstrap、CircularBlockBootstrap、StepM、SPA、MCS 未实际运行 |
| 5. 参考实现 | PETSA、ORCA、INC 各完成 5 点只读审查并附路径 | TAFAS/OnlineTSF 等待 A2b；未通过外部下载绕过此项 |
| 6. 复用决定与补丁 | 已产出；补丁 dry-run 成功，未应用 | arch 有条件复用与统计方案仍需签署后补验；T2 新写行数为 0 |

## 测试：命令、结果与限制

完整命令、执行环境、返回码、耗时及 stdout/stderr 的 SHA256 保存在 `COMMANDS.log`。以下命令供复现参考，**本轮已经运行，不需要再次执行**。`run_logged.py` 为每次运行生成独立日志，重跑时应使用新的 label 和输出目录。

```bash
cd /mnt/afs/260010168/EarthDelta
AUDIT_RUN=runs/online/t0_reuse_audit_20260929T114614Z

python -B "$AUDIT_RUN/tools/run_logged.py" synthetic_interfaces_host -- \
  env PYTHONPATH=.pydeps:. python -B -m pytest \
  "$AUDIT_RUN/tools/test_t0_interfaces.py" -q -rs -p no:cacheprovider \
  --basetemp="$AUDIT_RUN/tmp/pytest_interfaces_host" \
  --junitxml="$AUDIT_RUN/INTERFACES_HOST_JUNIT.xml"

python -B "$AUDIT_RUN/tools/run_logged.py" existing_cpu_guarded -- \
  env PYTHONPATH=.pydeps:. python -B "$AUDIT_RUN/tools/cpu_test_guard.py"

python -B "$AUDIT_RUN/tools/run_logged.py" helper_semantics -- \
  env PYTHONPATH=.pydeps:. python -B "$AUDIT_RUN/tools/helper_semantics.py"

python -B "$AUDIT_RUN/tools/run_logged.py" patch_dry_run -- \
  patch --dry-run --batch -p1 -d "$AUDIT_RUN/patch_sandbox" \
  -i /mnt/afs/260010168/EarthDelta/plans/plan_v9_online_20260929/t0_reuse.patch
```

| 核验 | 结果 | 边界 |
|---|---|---|
| T0 合成接口 | **11 passed / 5 skipped / 0 failed**，9.13 秒 | 5 个跳过全部因为 arch 缺失；通过项中包含已有缺陷/限制的复现，并非修复验证 |
| 现有 CPU 测试，`not slow` | **975 passed / 62 skipped / 4 deselected / 0 failed**，约 766 秒 | 58 个在真实数组/归一化常数/检查点打开前跳过，2 个需要 CUDA，2 个缺上游参考输出；4 个 slow 测试未选中 |
| 手写面积加权 RMSE vs WBX | 最大绝对差 **7.77×10⁻¹⁶** | 规则全球无极点格心合成网格；不认证真实 store 或任意网格 |
| WBX vs v8 相对 RMSE 改善率 | 最大差 **1.42×10⁻¹⁴ 个百分点** | 同一合成网格、显式 issue/valid 时间，无天气数组读取 |
| 旧面积损失反例 | W=5、常数误差=3：旧 MSE=45，正确 MSE=9 | 已复现，未改冻结实现；后续不得沿用其绝对值/分母 |
| 校准语义反例 | 冻结函数=0.4472136；逐通道比值平均=0.6666667 | 前者先池化能量再求比；原计划“按变量平均”与冻结函数不一致 |
| 补丁 dry-run | 5 份任务文档检查通过，rc=0 | 沙箱中的原文哈希未变，补丁未应用 |

新增的审计用例在 `RUN/tools/test_t0_interfaces.py`，未改 `tests/`。包括 sklearn 的三个接口，WBX 的两个真实 vendor 路径对照，宽度不变性，记录筛选/可变性，LoRA 零编辑，以及编辑器部分进入失败。另有 `helper_semantics.py` 的面积损失和校准公式对照。

初次合成测试在隔离沙箱中因 `/proc/<pid>/stat` 不可见导致 psutil/xarray 失败，原始结果 **2 failed / 9 passed / 5 skipped** 已保留。随后在获准的本机 CPU 环境运行完全相同用例，得到 11/5；没有替换指标实现或跳过这两个 WBX 用例。其余初期审计工具错误也保留在日志中，不能把第一次命令记为成功。

数据边界拦截产生 58 条 `SKIPPED_BEFORE_OPEN` 记录，见 `CPU_DATA_GUARD.json`。受限回归不是无条件的全量测试通过，也没有产生天气技能、实验收益或 novelty 的证据。

## 环境与依赖

Python 3.10.12；实际 torch 模块版本 `2.3.0a0+6ddf5cf85e.nv24.04`。默认 NumPy 1.24.4 不满足 WBX 条件；通过已有 `.pydeps` 实际导入 NumPy 1.26.4、SciPy 1.12.0、pandas 2.3.3，WBX 自己追加 `.pydeps_wbx`，使用 CPU JAX/jaxlib 0.6.2。未加载 Stormer 检查点。

`pip freeze --all` 前后均为 250 行，SHA256 均为 `b9480509f97936c4756dca6b02960ceaa19657f6ffd0e8e9c5d2c708e09b9aaa`，新增/删除包为 0。`.pydeps` 同时有 NumPy 1.26.4 和 2.2.6 的 dist-info，因此 freeze 不能替代实际模块版本核验；没有清理或升级它们。

arch 建议版本为 7.2.0（NCSA），必要传递依赖建议 statsmodels 0.14.4、patsy 1.0.1。其他建议保留 sklearn 1.2.0、zarr 2.18.3、xarray 2025.6.1、fsspec 2026.9.0、matplotlib 3.8.4、JAX/jaxlib 0.6.2、ml_dtypes 0.5.1。具体 Python 约束、license 和本地/官方来源见复用决策表与 `PYPI_RELEASE_METADATA.json`；没有下载 wheel 或执行安装。

## 复用的现有代码与必须新写的部分

- **直接复用或薄封装**：Stormer bridge/rollout、现有 LoRA、v8 的 issue_mse/relative_rmse、官方 WBX、sklearn Ridge/时间分割；memory 的已释放记录规则可复用，衰减要按日历实现。66 个符号的存在性和签名已记录，不能据此声称 66 个函数数值全对。
- **有条件复用**：arch 的采样器和区间接口，等待 A2a 补验。StepM/SPA/MCS 是模型比较检验，不能作为所需同时区间的直接替代。
- **必须新写**：释放时钟与监督队列、actor/scorer 数据边界、不可变 payload、可微短窗口损失、按时间 purge 的折内 EOF 管线、通用梯度/校准包装、每臂在线状态。已有 hook 生命周期和可变引用风险需在新调用层处理。
- **只读参考**：PETSA、ORCA、INC 的设计。PETSA 的 NC/SA license 和 ORCA 无 license 均禁止本轮拷贝；本轮没有 vendor 任何第三方代码。

估计 T2 的非测试新代码可由约 1550 行减少到 1070 行，减少 **480 行（约 31%；合理范围 350–600 行）**。这是实现前估计，以 arch 安装通过和统计修订采纳为前提，不是实际已节省的行数。

## 与任务说明不一致的地方及原因

1. A2a/A2b 未签，因此没有完成安装、batch 5 克隆及相应运行验收；没有用本轮“确认开始 T0”替代这些签署。
2. 目标方案目录/CLAUDE 不存在，使用 ZIP 的只读副本；最终目录只放本轮允许写的两份文件。
3. 现有测试通过数据边界拦截运行，严格保留跳过项；不为满足“全量”表述而读取真实数据或动用 GPU。
4. T0 预填表“损失差按周平均 + StepM/SPA/MCS 直接给同时区间”不成立。修订为周 MSE 总和/有效数、同一重采样索引和 pooled RMSE 统计量；建议 Bonferroni 调整的 bootstrap 家族区间，仍需协议签署，并报告小样本/尾部 Monte Carlo 限制。
5. 短窗口效应比按冻结函数实际语义提出修订；编辑器、VerifiedRecord 和旧面积函数的限制没有在本轮修复，已进入后续验收建议。

## 等待签署的事项与后续命令

| 事项 | 阻塞什么 | 下一步 |
|---|---|---|
| 发布正式方案并签署 A2a | arch 及缺失传递依赖安装、5 个接口例子、统计封装的实际可用性 | 先按 REUSE_DECISIONS 的版本/license 清单确认安装环境与约束，再安装；不要升级 torch 或把 NumPy 自动换成 2.x |
| 发布正式方案并签署 A2b | TAFAS/OnlineTSF 的固定版本、日期、license 和设计审查；batch 5 清单/README 更新 | 准备获准的脚本副本；将默认日志放到 runs，保留已有目录，然后运行 `bash reference/_clone_refs_batch5.sh` |
| 接受 `t0_reuse.patch` 的统计/接口修订 | 后续 T2–T5 的明确契约 | 补丁尚未应用，原门槛不变；统计方法与冻结比较族在 PHASE0_PROTOCOL 签署时确认 |

正式 `AUTHORIZATIONS.md` 尚未部署，签署位置应为 `plans/plan_v9_online_20260929/AUTHORIZATIONS.md`；`RUN/inputs/` 中的副本不是新批准。本轮不代签，也不进入后续实现。

本轮无需人执行 GPU 命令。A2a 补装完成后应在**新的运行目录**重跑上面的合成接口命令，确认 5 个 arch 例子从跳过转为真实运行；A2b 完成后补写两个参考仓库的证据。只有这些缺口补齐后，才能宣称 T0 的全部验收完成。

## 已知问题和疑问

最优先保留的约束是：不复用旧面积损失的绝对值；不把模型比较 p 值当同时区间；不把可变记录当不可变监督；不让 hook 进入失败污染下一臂；不让 TimeSeriesSplit 的行号间隔替代标签兑现时钟。现有代码可以复用，但这些调用边界还需要后续新代码与负例验收。

本轮证据支持“若干接口可复用”和“几处计划/源码语义必须纠正”。在线残差是否具有预测价值、参数更新是否胜过输出订正、方法是否有研究价值，均未在 T0 验证。
