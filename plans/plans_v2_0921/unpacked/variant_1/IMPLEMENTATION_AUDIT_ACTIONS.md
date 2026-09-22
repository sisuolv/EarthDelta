# B01–B15 → 本迭代执行优先级

基线 `fb767f7f6efbc428be39c9ad84f5905331d6e40f`。以下是上一轮独立复核的执行映射，不是新一轮审计，也不是要求关闭所有 finding。

## 分类语义

- `BLOCKS_NEXT_PILOT`：本次实际模型/评分/数据入口必须修复或给出等价的可核验证明。
- `FIX_BEFORE_TRAINING`：只阻塞所述特定训练路径，不等于阻塞固定系数bank训练。
- `FIX_LATER`：不独立阻塞科学取证；在已经修改的同一文件中可做小修，不能发展为cleanup项目。
- `NOT_A_BLOCKER`：在明确受限的本轮调用模式下不阻塞；入口约束仍须执行。
- `AUDIT_FINDING_REJECTED`：本轮没有整条 B finding 被无条件驳回。驳回的是过强推论，见下文；不得为了填分类虚构一条全盘否定。

| ID | 优先分类 | 复核结论 | 受审 file:line | 最小处置 / 对应任务 | 非阻塞范围 |
|---|---|---|---|---|---|
| B01 | `BLOCKS_NEXT_PILOT` | 同意实质；不强制完整 Lightning；共享本地 normalizer 与官方零 diff_mean policy 不一致。 | `scripts/export_upstream_reference.py:106–110,230–250; earthdelta/bridge/stormer_bridge.py:251` | 独立 transform 与 raw-to-raw 多步参照；显式 policy。 **R2I-01** | 不能仅凭共享代码断定天气损害；本例的policy偏差才是依据。 |
| B02 | `BLOCKS_NEXT_PILOT` | 同意；身份只展示，输入/多步未绑定，可能错误放行。 | `scripts/s0_gate.py:170–186,265–301; earthdelta/contracts.py:28` | 匹配冻结完整身份、shape/hash，全部注册lead缺一项即阻断。 **R2I-01** | 任意文件通过 hash 子项不等于整个 gate 通过。 |
| B03 | `FIX_BEFORE_TRAINING` | 同意caller bug；拒绝推成bank不能训练；tuple→新叶tensor切断caller/head，不阻断固定系数bank A/B。 | `earthdelta/bridge/stormer_bridge.py:663,742–749` | 本轮bank只测其梯度；只有实施可微controller训练时才补[B,K]tensor入口。 **仅后续可微controller路径；R2I-02验证bank梯度** | FIX_BEFORE_TRAINING 专指该controller，不是当前bank。 |
| B04 | `FIX_BEFORE_TRAINING` | 同意、路径相关；shared program/batched Gram报错。 | `earthdelta/metrics_contract.py:179–194; earthdelta/heads.py:445` | 启用系数路线前修ellipsis广播并测组合。 **R2I-04** | 有限端点oracle无需先用此路径。 |
| B05 | `BLOCKS_NEXT_PILOT` | 同意；无效权重/scale可能NaN或负MSE。 | `earthdelta/metrics_contract.py:96–100,145–153,250–274` | 实际归约前验证broadcast后的权重、正分母、scale与missing规则。 **R2I-01** | 固定合法Q不一定触发；在入口证明合法即可。 |
| B06 | `BLOCKS_NEXT_PILOT` | 部分同意；F-only诊断不能冒充H/S/F全目标，校准可能遮蔽机制。 | `earthdelta/metrics_contract.py:145–157; earthdelta/heads.py:294,350–359` | pilot统一一次Q_eff；head阶段分analytic/calibrated且默认off。 **R2I-01 / R2I-04** | sum/mean均可正确；不是全部sum都错。校准是模型选择不是天然bug。 |
| B07 | `NOT_A_BLOCKER` | 部分同意；固定6h无first-seen证据、legacy默认升级。 | `earthdelta/data/make_splits.py:86,224–227,306–323,378` | 入口声明retrospective/scenario；memory off；实际first-seen必须有记录。 **R2I-01准入声明；memory启用前才全面修** | 对关闭核验memory的retrospective实验非自动污染。 |
| B08 | `BLOCKS_NEXT_PILOT` | 同意；metadata预分配并不表示内容已写完。 | `earthdelta/data/pull_wb2.py:423–479,705,783` | 核验选定窗口每个读取chunk/变量及完成记录；不需扫全库。 **R2I-01** | 不能将随附空Zarr反例写成用户实际数据已损坏。 |
| B09 | `BLOCKS_NEXT_PILOT` | 部分同意；日历枚举不能证明history/target可取，row ID不是过程ID。 | `earthdelta/data/make_splits.py:338–389` | 在pilot dataset join真实坐标，记录issue/process分组，拒placeholder。 **R2I-01 / R2I-02** | 可保留日历生成器为候选索引。 |
| B10 | `BLOCKS_NEXT_PILOT` | 同意；ps2 gate不能验证本次ps4。 | `scripts/s0_gate.py:609–620; scripts/export_upstream_reference.py:410–429` | 共用一个显式冻结checkpoint/config/reference-dir。 **R2I-01** | 不要求ps2/ps4双跑。 |
| B11 | `NOT_A_BLOCKER` | 部分同意；确认判断前必须补；MDE不等于价值阈值，preview/replay需计成本。 | `plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md:36–41` | 本包权威合同分列delta_min/MDE，成本全计；历史YAML保留。 **R2I-03正式判断前** | 不阻塞有cap的探索bank；禁止用MDE自动作为delta_min。 |
| B12 | `NOT_A_BLOCKER` | 同意边界；thread-local不是model并发锁；hook不覆盖replay。 | `earthdelta/bridge/stormer_bridge.py:632–656,780–790` | 本轮串行、无activation replay；不构建通用并发。 **R2I-01运行合同** | 未观察真实并发损害，不加新阻塞。 |
| B13 | `FIX_LATER` | 部分同意；严重度下调；必要验证PASS后cleanup异常导致状态矛盾。 | `scripts/s0_gate.py:680–689` | 区分validation_pass/runtime_status/cleanup_warning；必要步骤异常fail。 **R2I-01同文件低成本顺手修** | 仅非关键清理失败不追溯性否定数值结果。 |
| B14 | `FIX_LATER` | 同意低严重度；N/A被数值格式化导致MD缺失。 | `scripts/s0_gate.py:748–752,767–770` | 有限数值才格式化，其他N/A/error；JSON优先。 **R2I-01同文件顺手修** | 不独立阻塞科学pilot。 |
| B15 | `NOT_A_BLOCKER` | 同意复数bug；空集语义分层；complex转double丢虚部；通用empty no-op不等正式库。 | `earthdelta/selection.py:248–272` | R2I-02 registry入口验证real floating/nonempty/no-edit及四项约束。 **R2I-02** | A03 finite/bound/max_active/max_candidates已生效，不重做所有selector。 |

## 只做最小补丁

真正必要的入口：**B01/B02/B05/B06/B08/B09/B10**。B08/B09可由小型pilot dataset入口保障，不需先重写pull_wb2。B07在memory关闭时用明确情景合同隔离。B15先守住固定registry。

B03不阻断固定系数bank的A/B训练；R2I-04默认监督回归也不需要穿过天气模型的caller梯度。本迭代不新增端到端router训练，所以不能据B03先要求大桥接重构。

## AUDIT_FINDING_REJECTED：拒绝的过强推论（不是删 finding）

1. “caller梯度断裂→bank训练不可能”：拒绝；必须实际检查bank梯度。
2. “只有完整Lightning对象才能独立S0”：拒绝；独立且逐步可核对的最小官方policy实现也可。
3. “sum与mean不同→全部系数求解错误”：拒绝；关键是同Q_eff及惩罚单位。
4. “cleanup失败→此前数值验证一律无效”：拒绝；区分必要计算与非必要清理。
5. “日历manifest没有数据join→日历工具本身无用途”：拒绝；它不能作训练可取性证明。
6. “新测试覆盖不足→旧测试被削弱”：无证据，不得指控。

## 原 A disposition 不得抹掉已完成部分

A01/A05 部分关闭有依据；A04 公共公式已有生产调用但统一指标未闭合；A06 的UTC修复已成立，provenance/index另议。A03四项主要检查 **CLOSED_FOR_REQUESTED_CONSTRAINTS**，类型/registry是较小残留。A11保留历史YAML+superseding note合法；补阈值/成本权威合同即可。

## 验证边界

随附283/7/0日志与CPU反例只是已有证据，不是本包复跑结果。没有GPU不另造finding；实际数据损害、并发输出污染及weather skill均未在本包验证。

源码链接基址：`https://github.com/sisuolv/EarthDelta/blob/fb767f7f6efbc428be39c9ad84f5905331d6e40f/`。行号为受审版本，修改后在当前diff中重新定位，不照旧行号机械替换。
