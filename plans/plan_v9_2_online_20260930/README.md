# EarthDelta v9.2：阶段 0 修订草案

状态：**DRAFT_UNSIGNED，仅修订计划；尚未授权执行。** 本版处理 2026-09-30 Claude 审查的 V91-01–23。原 ZIP、v9.1 两份 T0 产物及历史运行目录全部保留；不直接应用旧 `t0_reuse.patch`，其采纳结果已经整合到本版。

代码基准：`probe-headroom-20260927@d55ad70854bcf535b86c761b8d9cf7a7981374a0`；本地 main 为 `f228895203d3ced2d5e984e705ea14a94160d4b8`。本版写文档，没有新模型结果。

## 阅读入口

1. [修订总览与逐条处理](AMENDMENT_LOG.md)：哪些采纳、哪些有保留、哪些仍待执行。
2. [任务顺序](00_START_HERE.md) 与 [阶段 0 协议](PHASE0_PROTOCOL.md)。
3. [数据规则](DATA_POLICY.md)、[统计规则](STATISTICAL_PLAN.md)、[任务合同](TASKS.json)。
4. [建议默认值](DECISIONS_PROPOSED_DEFAULTS.json) 与 [授权草案](AUTHORIZATIONS.md)。

所有数值和比较集合的唯一机器读入口为 [PHASE0_CONTRACT_DRAFT.json](PHASE0_CONTRACT_DRAFT.json)；Markdown 用于解释。发现不一致时阻止执行并修订文档，不能挑选更有利的版本。

## 这次改变了什么

- 固定主臂 `lag1c`；新增参数慢漂移 `ewma_g`，输出基线升级为 `ocl_fresh`；用 `delay30` 替代 random。
- 提议预先固定 δ=0.3 个百分点、非劣界 −0.5%、相对 F0 收益下界 >0；正式生效需要试点前签署。
- 14 天循环周块为主；NumPy 实现，arch 不再阻塞；先做独立模拟覆盖率校准及 MDE，校准失败不能发布正式 Gate。
- 固定缺输入、不完整反馈、冷启动、隔离带和逐格分母规则。现有 QC 覆盖 1439 个所需索引，拟新增检查 17 个；不是重新扫描全部年份。
- 原始梯度保存 FP32、范数 FP64；均值和中心化都在原始梯度尺度。旧 headroom 探针退出当前关键路径。
- 只承诺阶段 0；T5、2021、2022 不排期、不自动续跑。

## 如何启用

本目录已经是完整修订草案，不需要覆盖 v9.1。根目录 `CLAUDE.md` 尚未创建；[CLAUDE_ROOT_PROPOSED.md](CLAUDE_ROOT_PROPOSED.md) 是待人工审查的合并提案，不是生效指令。

人工需要：确认使用 v9.2；核对根规则提案；在 `AUTHORIZATIONS.md` 签署 A1；在第一次拟合期试点前签署协议 A 段及 A3。A2a 可以暂不签；A2b 延后至 T5 前。分析期必须再签协议 B 段和 A4。不能把本次“修订计划”的授权当成代码、数据或 GPU 的授权。

本轮没有代签、安装、克隆、GPU、数组读取、代码修改、commit 或 push。详细来源与本轮校验见 [SOURCE_INDEX.json](SOURCE_INDEX.json)、[PLAN_VALIDATION.json](PLAN_VALIDATION.json)。
