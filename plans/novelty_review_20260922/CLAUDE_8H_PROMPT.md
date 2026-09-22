# 给 Claude 的 EarthDelta 八小时执行指令

你是 EarthDelta 的主要实现者。收到本指令并开始实际工作时记为 T+0，墙钟预算八小时，包括编码、排队、GPU、分析和交付。直接实施、测试和提交已就绪任务；不只输出另一个计划。

仓库：`/mnt/afs/260010168/EarthDelta`。预期 HEAD=`d749a1c62521226df857587e08f7d067b0f15355`，分支 `audit/round2-review-20260921`；大量合法未提交改动已经存在。先记录实际状态，保护原有工作，不 reset、不覆盖已有 GPU 证据。

按顺序阅读：

1. `plans/novelty_review_20260922/SPRINT_8H_PLAN.md`，完整阅读。
2. `plans/novelty_review_20260922/SPRINT_8H_SPEC.json`，这是计划规格，不是现成 gate config。
3. `codex_audit_round4/ROUND4_REPORT.md` 第零部分及 C01-C11，核对相应实际代码。
4. `plans/plans_v2_0921/cci/README.md` 与实际 `submit_job.py`。

研究目标是用真实天气实验获得可行性和方法贡献的支持或反证。已完成的 literature review 可直接作为研究定位；本轮不扩展文献审计，不靠外部语言模型意见宣布 novelty 已证明。

## 授权与资源

用户已授权使用 CCI 的可用资源，主要为提交排队的 4×H100，可提交多个已就绪任务；资源免费。默认两个四卡任务组，有实际需要可增加已就绪 cache 分片。保持每进程独立模型，避免新建 DDP 或共享 hook 线程方案。队列等待计入八小时，不预占 GPU 等待脚本补齐。

5090 可作为已验证环境的替代或设备对照；先核验实际规格、显存和软件/算子兼容。不能把 H100 的旧镜像直接视为支持 5090，也不能把切卡当成共同参考/持久化问题的修复。

用户授权 SiliconFlow API 辅助，base URL=`https://api.siliconflow.cn/v1`，模型候选 `zai-org/GLM-5.3`、`deepseek-ai/DeepSeek-V4-Flash`、`Qwen/Qwen3.8-27B`。运行时确认实际可用性。凭据使用私有 `SILICONFLOW_API_KEY` 环境配置，不写入源文件、argv、平台命令、快照或日志。没有安全配置时跳过 API，继续主实验。

API 最多作为三种有边界的辅助：设计漏洞检查、代码/测试检查、结果与主张检查。返回的建议由你结合实际代码验证；不把建议当测试结果，不让 API 成为主流程依赖。无需额外多代理编排。

## 本轮最重要的范围选择

主实验固定 **R=F0，reference_kind=base_frozen**，使用同一原始冻结 checkpoint 作为所有动态专家的共同参考。静态 Fs 优化失稳对照可以独立做，但不改变本轮 R，不延迟主线。

这不是给旧 Stage 3b/Stage 4 放行，也不能写成优于训练良好的 Fs。报告为独立的 base-reference pilot，旧 Fs 审计问题保留 OPEN。主要问题是动态参数编辑和响应预测是否有用。

旧 `--stage all` 及 `bank_train_formal_4experts_argv.json` 会各自重训 Fs，不能直接沿用。请实现明确的 R 加载路径或小型独立 runner，复用现有核心原语。所有 worker 必须核验相同 R/norm/grid/Q 身份。

## 八小时执行次序

- T+0:20 前：冻结角色、时间列表、动作、目标、比较、阈值和源码；开始已就绪的环境/S0。
- T+1:30 前：补齐准入消费、依赖停止、R 加载、动态保存/重载、profiling 隔离和正式 registry；实际反例必须被拒绝。CPU 测试只跑有关小组，不反复无意义扩大回归。
- T+2:15 前：四卡训练 K=4/rank=4，blocks18-23，6h×4 步训练/hold，随后回到 R；保存并独立重载动态权重。
- T+2:45 前：8 起报×13 动作的真实 6/24/72h smoke，冻结按吞吐选择的标准或缩小预算。
- T+4:15 目标、T+4:30 最晚：完成全候选缓存及覆盖验证。缺行/错身份必须拒绝，不能筛成功子集。
- T+5:30 前：完成强直接向量收益 B1 与 e/u 分解 E1，尽力完成决策分量 E2；冻结模型、预测和评分脚本后再打开 holdout。
- T+6:15 前：原生 loss/gain、联合迁移、72h guard、失败和完整成本表。
- T+7:15 前：四格诊断、预注册学习曲线、可行时 virtual edit；重载后重放预定八个评价起报。
- T+7:15–8:00：不新增长任务，整理全部证据、作业状态和结论。

时间不足时执行计划中的预注册降级规则；不能靠跳过资格、改变主比较或删失败行满足“完成”。T+8 到点交付已经得到的真实结果，未完成项保持 BLOCKED/INCONCLUSIVE。

## 不可省略的实验合同

1. 标准为 128 bank-fit、32 bank 资格、256 policy-fit、64 calibration、128 sprint holdout，另 8 debug。初始来源为 2018/2019/2020；必须核验实际内容与 UTC。2020 已在历史项目中暴露，仅能称本轮封存评价，不能称最终 untouched confirm。
2. 训练/校准动作：no-edit + 四专家各 alpha=0.125/0.25，共 9 个。holdout 增加 0.1875，共 13 个。rho 仍为 0.25，不更改训练 a0/rho 合同。候选 ID 直接贯穿选择与执行。
3. heldout alpha=0.1875 的响应不能用于 basis、辅助任务、调参或“无标签”预训练。所有方法共享合法 I 和动作描述，不读取当前起报未来 truth 或真实候选响应。
4. 三目标是原生 6/24/72h，联合权重固定为 (0.2,0.5,0.3)。直接向量收益基线也可重新加权，所以“换目标能用”本身不算结构优势。
5. B1 至少包含 expert ID、alpha、alpha² 和状态交互；比较 ridge/小 MLP，由 calibration 选定。不能只做弱线性基线。NN 固定 3 seeds，至多 4 组超参数；完整记录调参成本。
6. E1 raw geometric 为主；加性校准独立报告。修复/绕开旧 paired/head 的目标混用，最终用真实 native loss 判优。低维表示误差单列，不称精确目标。
7. 主比较固定 E1 对 calibration 预选 B1，动作集合为 no-edit + 四个 alpha=0.1875，目标为联合 Q。E2 是次比较；不可在 E1 失败后事后换主结果。
8. 默认主增益门槛为 R 平均目标 loss 的 0.1%，95% 下界需超过该门槛；至少 2/3 seeds 同向。72h 相对 R 的均值退化 95% 上界不得超过 0.5%。这是 pilot 工程尺度，非业务保证。阈值最多在 T+0:20 前有理由地改一次，之后冻结。
9. 14 天时间块配对 bootstrap，7/28 天作预定敏感性。先对每起报汇总 seed，不能把 seed×起报当独立天气样本。
10. 所有运行失败、回退和候选缺陷保留。选择失败候选的处置规则在运行前冻结并计成本。平台 SUCCEEDED 不等于科学 PASS。

## 实现重点

保持改动窄而可验收：可以新增 `scripts/r4_value_pilot.py`，但不要构建通用调度平台。拟定子命令 `reference/train-bank/cache/fit-heads/evaluate` 必须实际实现、保存 `--help` 和 smoke 结果后才提交。

首个缓存完成前实现并验证：真实 bank state_dict 保存、独立进程重载；正式/测量权重 hash 不变；一个起报全部动作完整覆盖；缓存 reader 拒绝角色/身份错配；评价 truth 与 selector 输入结构分离；新 full-objective 与端点 loss 差相等。

若使用空间摘要或响应基，只能在对应 fit 子集上学习。学习曲线各档的 basis/缩放也需要重拟合，不能隐藏使用更大样本的信息。对照方法获得同样模拟信息，或明确标注不同信息预算。

采用已存在的 `plans/plans_v2_0921/cci/submit_job.py --argv-file ...`，先 dry-run 再提交真实快照。每阶段只消费通过的前驱记录。实际解释器、环境、checkpoint/数据路径与 source SHA 全部绑定。不要拿本 prompt 的未来脚本名或 null 身份当可执行配置。

## 最终交付

在独立运行目录保存全部实际 manifest、job ID/日志、checkpoint、资格与缓存覆盖、冻结预测、统计和成本，另写：

- `RESULTS.md`：先回答链路是否可信、天气有没有收益、响应分解是否有额外价值；四张表覆盖资格、方法、迁移、失败与成本。
- `results.csv`：每方法/seed/split/目标的 loss、相对 R 和固定候选的 gain、CI、regret、harm、no-edit 和成本。
- `decision.json`：至少有 `reference_kind`、`pipeline_feasibility`、`weather_utility`、`novelty_signal`、`data_efficiency_signal`、`full_fs_pipeline`、`remaining_blockers`、`next_action`。

`novelty_signal` 可以是 SUPPORTED_PILOT、INCONCLUSIVE、NOT_SUPPORTED 或 BLOCKED。只有全部主判据与原生天气收益支持时才写 SUPPORTED_PILOT；不写“已经证明全局 novelty”。旧 Fs 未解决就保持 full_fs_pipeline=OPEN。

如果更简单的 B1 胜出，如实建议采用 B1 或重新定位响应复用任务；如果 oracle 有余量而合法策略没有，定位可预测性瓶颈；如果当前字典无余量，停止增加 head。每种结果都要对应真实下一步，八小时交付不能只是一份新的计划。
