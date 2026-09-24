# Codex需要补齐的输入

本包/仓库已有的协议、gate与详细数值JSON无需重复上传整个仓库。
可以提供Codex可访问的持久路径与重新计算的hash；不要把大权重强制上传聊天。

## FP05a立即核验的小型原始文件

| 作业 | 现有证据 | 需要的原始文件（在对应run根目录） |
|---|---|---|
| S0 `pt-g3344e9z` | `plans/plan_v4_0923/run_20260923T_fp01_trace/gate_output/s0-gate-20260923t174135587010z/s0_gate_result.json`及四臂trace | `artifacts/round2_cci/ed-r4gate0923e-0923174024-32e8c7/job_result.json`，调度ID对应记录、stdout/log、source_manifest |
| Fs v2 `pt-fuiqqeau` | V2_J3 decision/panel | `artifacts/round2_cci/ed-r4fs03v2p-0924020931-71716b/job_result.json`及source/env/argv |
| Fs reload `pt-gr3rrmdg` | certify文件 | `artifacts/round2_cci/ed-r4fs03v2j4-0924021935-be0786/job_result.json`及verify/provenance |
| Bank train `pt-vt003lhm` | BJ2 decision及各专家指标 | `artifacts/round2_cci/ed-r4bankbj2-0924052624-dde08d/job_result.json`、4个expert/run_record/provenance |
| Bank profile `pt-ycpvyb6n` | BJ1 decision/summary | `artifacts/round2_cci/ed-r4bankbj1-0924051352-a9732c/job_result.json` |
| Assembly失败 `pt-m52qwunf` | DEV-FP04-001 | `artifacts/round2_cci/ed-r4bankbj3-0924053507-ae0123/job_result.json`及拒绝记录 |
| Assembly retry `pt-831c01g4` | A1-A5详细数值 | `artifacts/round2_cci/ed-r4bankbj3r-0924054117-185e82/job_result.json`及assemble/verify provenance |

上述文件名按记录的run_dir定位；真实执行机若名称不同，提供原始路径映射，不伪造同名文件。
可以提供带实际source/内容hash的精简receipt，须含job_id、run_id、rc、elapsed、起止时间与产生证书的映射。
当前正式数值执行证据已经可读；这些补充用于完成身份与调度溯源，不要求重新跑GPU。

## 曝光重建

- X1_N32/N64 actual fs_fit_record训练issue_ids/times、fs_panels/read-set；N64已有嵌入job_result和逐issue结果，不能再当未执行计划。
- `plans/plan_v4_0923/run_20260924T004746Z_fp03_x1_ntrain_EXPLORATORY/admission/x1_n32_manifest.json`、`x1_n64_manifest.json`及实际admission；本仓库已经有，Codex应直接读取，不再让用户重复上传。
- v1/v2/FP04/旧Round4所有实际使用的训练、筛选、资格、debug窗口；原始值不需要批量上传，只需可复核索引和hash。
- 2019H2的数据时轴/完整性与每个候选完整support；不能只给年份名。
- 任何历史已读取/训练/用来选方法的2020H2记录全部登记。原计划的clean-confirm假设已被否定。

## 正式cache前本地重哈希的大文件

- `plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/certify/fs/fs_adapter.pt`（记录SHA `7b39226b550a2795b1ed85020057bec46064f132b1c36d77a4db9b5682323089`）。
- Fs merged文件按`reference_manifest.json`的真实path读取（记录SHA `1e48438ee98dc72552786011d25d9db3cc208672ceb8a7fc725c7f8fde77a79b`）。
- `plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank/bank.pt`（记录SHA `c3e3e34b60b463b41a6c824fa680d844a2830cd2c7b40d3fc4544dc985b85f81`），四个expert及probe文件，registry。
- 原ps4 checkpoint、NPZ、lat/lon/channel和预定2019真实数据：用已有persistent路径，无需聊天上传。

## 还需要预先决定而不是“补实验结果”的字段

FP05的新exposure ledger、72h harm margin及其解释、block/fold/HPO设置、数值重算容差、一次N选择/替换规则、GPU/cost cap和明确授权。
不存在真实clean confirm就保留MISSING并禁止访问；本阶段不为此开启新年份下载。
