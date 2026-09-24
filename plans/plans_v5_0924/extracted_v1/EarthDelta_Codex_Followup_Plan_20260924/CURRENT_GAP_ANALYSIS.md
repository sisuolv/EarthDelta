# 当前缺口与关闭范围

所有源码定位默认 `e0136d4ca8ad3e49f49bdb4947cc73a95f082c4a`。E编号见 `evidence/EVIDENCE_MANIFEST.json`；这些是读取证据，不是本包新实验。

| 项目 | 状态 | 精确证据 | 本次处置 |
|---|---|---|---|
| FP-00库存/旧包读取 | DONE / OBSERVED | `plans/plan_v4_0923/run_20260923T164722Z_fp00/entry/entry_evidence.json:1-35` | 不重跑；当前入口补新资产receipt |
| FP-01官方S0 | DONE / EXECUTION_CONFIRMED（数值） | `plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/s0-gate-20260923t174135587010z/s0_gate_result.json`；`stormer_bridge.py:310-400`；`s0_gate.py:115-130` | 1/4/12步max=0，tol=1e-5不变；原始job receipt仍需补链 |
| inverse精度修复 | STATIC_CONFIRMED | `earthdelta/bridge/stormer_bridge.py:167-400`，E05 | 原NPZ input逆变换常量保留；不能用FP32公有投影重构inverse |
| FP-02真实fail-stop | 已完成子项 | `scripts/r2_fs_bank_train.py:2310-2370`、`tests/test_plan_gate_chain.py` | 失败break、后继BLOCKED；不再指控旧“继续全部stage”尚未修 |
| FP-02 Fs admission | 已完成子项 | `tests/test_loaded_admission_binding.py`；`scripts/r2_fs_bank_train.py:2170-2238` | 外层/每行证书、目标/内容消费已测试；不等于全角色隔离 |
| FP-02 profile | 已在FP04补做 | `tests/test_profile_immutability.py:1-174`；E10/E11 | 真实stage深拷贝、原bank不变；不要求重写profile |
| FP-02 thin runner / split | PARTIAL / MISSING | scripts/earthdelta树E16；`pilot_contract.py:121-137` | FP05新增薄runner、角色/曝光加载器；不要把FP02整项标DONE |
| FP-03历史失稳解释 | 限定关闭，不接受单因果 | `plans/plan_v4_0923/run_20260923T192759Z_fp03_fs/forensics/j2_findings.json` | 认证后端lr=.01仍会失稳；旧坏Fs隔离，H32新资格取代不是洗白 |
| Fs v2 | DONE / bounded qualification | `plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/decisions/V2_J3_formal_decision.json:700-900` | 训练-2.594%、资格集+0.3393%、14/16变差；仅过2%有限面板门 |
| v2 prereg | 本地时序支持 | `plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/protocol/fs_protocol_v2.sha256`、`jobs/V2_J3.submission_record.json` | 02:04:13Z早于02:09:45Z；SHA一致，但不是外部时间戳证明 |
| FP-04 common Fs | DONE / bounded certification | `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank/bank_verify.json:350-420` | 同一merged Fs hash，源slice和组装/重载绑定 |
| FP-04 A1–A5 | DONE / exact registered probes | `earthdelta/bank_training.py:2370-2420`，`plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank/bank_verify.json:1-420` | 实际max0，无容差；仅一个probe输入，不外推所有输入 |
| FP-04 own direction | DONE / in-sample only | `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/protocol/PREREGISTRATION.md`、`task_result_fp04.json:140-235` | E3严格mean<1，不是事后放宽；没有OOS资格门 |
| X1曝光遗漏 | **P0阻塞协议冻结** | `plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/results/x1_results_X1_N64.json:1-35`、`admission/build_manifests.py:31-53` | 2020H2不能预留为清洁confirm；需记录失败/探索/丢弃模型也产生曝光 |
| FP05 delta | 必改解释和估计目标 | `plans/plan_v4_0923/FP05_PLAN.md:19-24`；E09 | .0034只保留为Fs-relative工程目标；不称自动回本/应用价值阈值 |
| FP05 OOF | 有条件采用 | `plans/plan_v4_0923/FP05_PLAN.md:24,35-37` | 回顾性、嵌套HPO、完整support净化；不得声称实时因果部署 |
| FP05 debug anchor | 必补覆盖矩阵 | `plans/plan_v4_0923/FP05_PLAN.md:41-52`；FP04 own-group panels | 不存在的跨专家/跨样本历史结果不可当golden |
| bank OOS / novelty | MISSING / INCONCLUSIVE | FP05尚未实施 | 由FP05测量，不是先验拒绝做FP05；FP06不得抢先得正面结论 |

## 对用户/前审查最重要的纠正
1. 新S0和新common-Fs bank的证据已经改变，不继续复述上一轮未过gate。
2. TF32/后端解释可重复性，不是全部发散的唯一原因；低学习率长训也有资格集损害。
3. Fs“过2%资格门”不是零损害，bank更没有样本外无害证据。
4. X1实际用了2020H2；它是本轮FP05计划的关键漏项，不是可忽略的旧模型。
5. `compare_probe_states`要dtype/shape相同、torch.equal且差为0；严格比较不等于覆盖了整片天气分布。
