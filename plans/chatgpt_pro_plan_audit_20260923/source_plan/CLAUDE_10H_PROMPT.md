# 给 Claude 的约 10 小时执行 Prompt

本文件供用户后续恢复执行时使用；撰写本文件没有启动任务。

---

请在 `/mnt/afs/260010168/EarthDelta` 执行一个约 10 小时的有界工作窗口。此消息明确恢复此前暂停的相关工作；从你开始执行时计时，包含编码、CCI 排队、运行、排错与交接。请实际完成可达的工作，不停留在计划或文档。

先按顺序阅读：

1. `plans/plans_v3_0922/NEXT_STEPS_10H_PLAN.md`：本窗口主计划。
2. `plans/plans_v3_0922/` 中三份 `EarthDelta_Stage1A_*Review*.md`：复查原文，注意它们是静态审阅，不是三次运行验收。
3. `plans/plans_v3_0922/evidence/digest_probe.json` 与 `digest_probe.py`：当前摘要不兼容的实际 CPU 证据。
4. `plans/plans_v2_0921/CLAUDE_EXECUTION_PLAN.md`、`CCI_VALIDATION.md` 和 `cci/README.md`：科学合同、既有启动方式与历史实测。10 小时范围以新计划为准；不要把旧历史成绩当新代码结果。

当前已知 HEAD 是 `d749a1c62521226df857587e08f7d067b0f15355`，请重新核对。`c397a7b` 是 Stage 1A 补丁；`d749a1c` 只新增复查材料。保留已有修改和未跟踪路径，尤其 artifacts、checkpoints 下用户脚本、三批 plans，以及其它用户作业。不要自动 reset/checkout、清理或 `git add .`。

## 目标和优先级

保留 B01 有效修复，完成 B02/B10 剩余使用路径；补原生 Q、实际小切片准入和必要异常验收；提交真实官方 GPU S0。S0 和训练准入通过后，争取完成真实 Fs/bank 小训练、完整小缓存及 cheap-policy 开发表。

默认本窗口不做 dual-head、E3、连续系数控制器、memory/JEPA、共享模型并发或 activation-checkpoint replay，不读取最终 confirm。B03 caller 梯度与未用 B04 Gram 暂缓，固定系数训练不能等它们全部关闭。

不要保证在 10 小时内获得统计显著结果。若时间/正确性/数据不足，缩小后续实验范围并诚实交付；不能降低验收、删失败候选或用欠训练的小库宣告科学失败。

## 必须修复并实测的要点

- 摘要序列化统一：当前 bridge 是所有 mean 后所有 std，exporter 是按 interval 交错；还存在 dtype 差异。仓库真实 NPZ 得到两个不同摘要。共享轻量身份格式可以，官方数值 transforms 必须保持独立；显式处理格式版本与旧缓存。
- 比较冻结 expected、reference manifest 与实际 loaded/computed identity；移除自身版本比较的空洞证明，使缺失/不等真正影响总判定。
- 当前 raw 经 bridge normalizer 生成输入，绑定到保存的 normalized input，再比较全部注册 rollout 的实际数值。默认 1/4 步，72h 评分前补 12 步验证。
- 显式 config/reference-dir，完整身份唯一选择，拒绝模糊 glob/legacy 认证回退；唯一输出目录，完成标记最后发布。
- 分开源码与资产根；worker 只能导入冻结 snapshot。实际模块内容、checkpoint、NPZ/NPY、变量/坐标、dtype/shape、输入输出均有证据。
- 建立一致的合成整包正例，再逐字段破坏，通过真实 gate/main 验证失败及退出码。只能替换昂贵模型部分，不能 mock 掉待验证的身份或总判定。helper 的通过数不是完整验收。
- 原生评分使用统一 H/V/Lat/Lon 约简、正 scale/分母、非负 finite q 和独立 FP64 端点损失差测试。
- 实际训练/评分切片核对 time/history/target/通道/坐标/finite/数据角色；不能只信 marker 或 shape。

已有 B13/B14 部分修补，请补专项验证，不要重复声称完全未开始或凭静态修改就关闭。

## 时间安排

按主计划安排 T+0–2:20 补 Stage 1A，并让 CCI 环境准备在后台完成；T+2:20–4:30 完成 Q/数据/回归及最小实验入口，同时运行真实 S0/profile；T+4:30–6:30 在准入通过后训练 Fs/bank；T+6:30–8:10 做完整小缓存；T+8:10–9:00 生成 cheap-policy 开发表；T+9:00–10:00 收尾。

时间点是转向点，不是跳过验收的理由。T+4:30 若 S0 尚未通过，剩余时间以解决 S0 为主并取消大缓存目标；T+6:30 若库资格不足，交付训练证据，不伪造完整效果表。截止前留出完整交接时间，不能只报“任务已提交”。

你已获准使用 CCI 任意资源，主要通过提交 4×H100 作业，允许多个独立任务排队。不要为已授权的常规资源、依赖准备或测试再次询问；实际工具权限约束仍需如实处理。

复用 `plans/plans_v2_0921/cci/submit_job.py`。准备真实 argv/config/CLI 后再提交；当前并不存在完整科学 pipeline，不得使用文档示例脚本名冒充已实现入口。J0 环境 → J1 官方 S0/profile → J2 Fs/bank/cache 按证据推进，队列等待计入 10 小时，不创建占卡等代码的任务。

默认四卡使用独立进程，bank 每卡一个专家，缓存按起报分片且每个起报跑全部候选。无需为了用满资源引入 DDP。记录实际排队、GPU/节点时间、前向数、磁盘和失败成本。

## 科学交付

默认 ps4，K=4、rank=4、blocks 18–23；no-edit+singletons，幅度 0.25，hold=4×6h。profile 若要求改 K/rank/训练时域，应在查看开发结果前登记并冻结。no-edit 和 hold 后 continuation 必须为冻结 Fs。

先短程真实训练检查，再做训练侧有界 updates。先完成 8 个 exposed 起报的全候选调试；之后按 profile 和实际数据条件，在结果揭示前冻结 32/64/128 个开发起报中的可行规模。小样本仅作 DEV；不宣称 128 个起报等于 128 个独立天气过程。

方法表包含 Fs/no-edit、训练折选定的 best-static、regime、ridge direct-gain 和 hindsight oracle。合法动作先保存再评分。bank、特征变换、候选选择与策略拟合均不得看评估折未来结果。报告原生 loss/gain、相对固定候选增量、72h guard、失败率及成本；统计依据不足时写 INCONCLUSIVE。

API 不是本流程依赖。如确需辅助代码分析，可使用已授权服务及私有环境中的凭证；不要把凭证写入源码、argv、日志、配置或 Git，不用模型回答替代天气实验。

## 结束条件和交接

保持 code/run/research 三轴状态分开。输出本窗口实际改动、测试命令与结果、job ID/最终状态、S0 数值或失败证据、训练资格和开发结果（若有）、仍未完成的工作及下一条可运行命令。所有数据均须指向日志/manifest；未执行的指标写 null。

T+9:00 后不再启动无法在截止前完成的大任务。安全保存本次任务状态；只处理自己创建的任务，不停止用户其他工作。不要擅自 commit/push 大量产物。

无需再让我批准一遍详细计划。请按以上边界执行，并在关键证据或方向变化时更新进度。
