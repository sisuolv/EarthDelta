# STOP 合同（替代FP05中错误的confirm与盈亏锚点）

## 1. 历史门槛原样保存
来源：`plans/plan_v4_0923/extracted_followup/EarthDelta_Codex_Followup_Plan_20260923/STOP_CONDITIONS.md`、Fs协议v1/v2、bank协议v1；以下不是本次新实验结果。

| 对象 | 不得偷偷改变 |
|---|---|
| 官方S0 | ps4；checkpoint `7fde884e…`完整SHA见OBSERVATIONS；official_zero_diff_mean；torch2.3.1+cu121/xformers0.0.27；TF32有效getters关闭；1/4/12步全部max abs <=1e-5 |
| Fs v2资格 | Ntrain=8，formal H=32，已选lr=.0025；原v1 H256失败不删；Q1有限；Q2 clip0、逐visit<=1.25、epoch上浮因子1.02；Q3训练8/8方向；Q4最大训练24h ratio<1且均值<=.99；Q5训练72h均值<=1.01；Q6资格holdout24h均值<=1.02 |
| Fs merge | 与S0不同的预登记合并检查，atol=max(1e-5,1.5e-3*delta_max_abs)、rtol=1e-5；不能称merge逐位，也不能把该容差用于S0 |
| bank | 同一已认证Fs；K=4/rank=4/blocks18–23；singleton a0=rho=.25/max_active1；hold4×6h后从已编辑状态继续Fs；32updates；E3自组训练24h均值ratio严格<1.00；禁止按DEV效果剔除专家/换bank |
| A1–A5 | exact，max abs=0；专家0..3与step1/4/12全部齐；hold后8步齐；原始证书一次probe的范围不可放大 |
| 源码 | FP04协议17个core pin不变；本轮新文件另建源码manifest。改认证数值语义时先停，不无理由重开训练 |

Fs Q6为有限面板的点均值容忍，不是统计“无害证明”；bank own-group72h是报告项，不是新的policy72h安全门。

## 2. 更正后的 split / exposure 硬门
- `policy_dev_pool = [2019-07-01T00Z, 2020-01-01T00Z)`，只作为候选池，不能直接认定每条都合法。
- 每个issue支持窗口为`t−12h…t+72h`，按实际UTC时次和内容证书验证。不得碰全项目已曝光窗口扩展前后24h后的区间；端点接触按冲突处理。
- 暴露账本包含未被选中、失败和探索性模型。**X1 N32/N64的2020-07-13…12-28训练不能遗漏。** N64新增56点的完整支持在声明时间轴下连续覆盖2020-07-13T06Z…2020-12-31T18Z；加24h后覆盖2020-07-12T06Z…2021-01-01T18Z。
- `2020H2 = EXPOSED_NOT_CONFIRM`。本包把confirm设为`UNASSIGNED_NO_ACCESS`；不能只剩两三天空隙就声称得到“2020H2未触碰确认集”。不授权自动选择新年度或读取新confirm。
- 在只考虑已知2019H1曝光截至7月4日18Z时，加24h buffer并要求12h历史，6h网格上7月6日12Z才可能成为最早DEV issue；这只是边界算术，仍需完整ledger决定。
- 2020训练资产用于2019DEV及双向blocked CV只能声称回顾性转移，不能模拟当时尚未获得未来训练资料的历史部署。

## 3. 效用阈值与同面板盈亏
拟保留`delta_min = 0.0034`，但定义为`E[L(Fs)-L(policy)] / E[L(Fs)]`的**工程筛查线**，须在DEV结果前签署冻结，不说它保证补回F0伤害。
历史Fs/F0=1.0033925583521255意味着F0-normalized伤害0.3392558%；同一历史面板的Fs-normalized补偿率是0.3381088%，而不是跨时段常数。新DEV必须同时报告：
```text
G_Fs(pi) = mean(LFs - Lpi) / mean(LFs)
H_Fs      = mean(LFs - LF0) / mean(LFs)
G_F0(pi) = mean(LF0 - Lpi) / mean(LFs) = G_Fs(pi) - H_Fs
G_static(pi) = mean(LOOF_static - Lpi) / mean(LFs)
```
F0只报告，不进入5候选的oracle/selector。数据、归一化和每个bootstrap重采样分母相同。
- `max_harm72_vs_Fs = TO_BE_PREREGISTERED_AFTER_PILOT_VARIANCE_ESTIMATE`；该占位不表示方差能决定允许损害。必须独立说明可接受损害/权衡的实质依据并在结果前冻结。
- alpha、power、block数和MDE是可分辨性问题；不能由MDE替代delta_min或安全界。
- 缺安全阈值或其理由时，完整协议未就绪；不可一边运行一边看72h结果决定上限。

## 4. 正式停止／继续逻辑
| 条件 | 决策 |
|---|---|
| 身份/原始回执/证书缺失或不一致 | BLOCKED或INVALID；补证据，不重训冒充已闭合 |
| 候选矩阵不完整、分母不一致、泄漏 | INVALID_EXPERIMENT；禁止任何效用结论 |
| 完整bank oracle收益上CI低于已冻delta_min | STOP_CURRENT_BANK；不是所有参数编辑都无效 |
| oracle已足够但合法selector未胜静态 | 根据CI记INCONCLUSIVE或PIVOT_STATIC；不把不显著当等效 |
| ridge超过Fs但未证明超过OOF_static | 只承认修复/选择线索，不能声称动态必要 |
| ridge超过强静态且安全/成本条件均过 | CONTINUE_EVIDENCE；尚不证明双头/响应新颖性 |
| 达到固定cap且CI跨阈值或有效blocks不足 | INCONCLUSIVE并停止追加 |
| 任意时刻读取confirm或结果导向更改N/专家/超参数/阈值 | INVALID_EXPERIMENT，隔离该轮结果 |

## 5. 成本与权限
未来仅两次科学GPU job（C-J1 debug、C-J2 cache）及一次infra retry，总上限3。当前GPU预算未授权；max4旧文案不是随意第四次run的许可。单次超时、worker重试、失败成本也计入。每项运行时限必须由profile和实际批准预算填写，不能照抄历史耗时当保证。

不自动重训Fs/专家；不自动扩展强度/组合/新backbone；不跑dual-head/JEPA/memory/spectral/dynamic rank；不做正式confirm。本轮FP06只做DEV决策。

2019也是所用Stormer checkpoint-selection的历史验证年；本轮DEV的“未见”只能相对于本轮Fs/bank/policy选择而言，不是整个预训练/模型选择流程的最终未触碰测试集。
