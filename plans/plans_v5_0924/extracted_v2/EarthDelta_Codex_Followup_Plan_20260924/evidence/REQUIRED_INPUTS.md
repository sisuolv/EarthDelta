# Codex执行前须补齐的输入

本包已给精确源码/证书/协议位置；已上传的计划和测试日志不必重复上传。大权重无需传到聊天，提供只读挂载路径和完整SHA即可。

## 现在需要（FP05a）
1. 完整私有checkout与实际工作区diff、认证bank协议的17个core pin。不是只给README或本包。
2. 当前依赖链的原始平台完成回执。以下每个目录的`job_result.json`、`invocation.json`、`source_manifest.json`、stdout/stderr及平台job ID绑定：

| job | 原始目录 |
|---|---|
| S0 `pt-g3344e9z` | `artifacts/round2_cci/ed-r4gate0923e-0923174024-32e8c7/` |
| Fs v2 fit `pt-fuiqqeau` | `artifacts/round2_cci/ed-r4fs03v2p-0924020931-71716b/` |
| Fs v2 certify `pt-gr3rrmdg` | `artifacts/round2_cci/ed-r4fs03v2j4-0924021935-be0786/` |
| bank shortstep `pt-ycpvyb6n` | `artifacts/round2_cci/ed-r4bankbj1-0924051352-a9732c/` |
| bank formal `pt-vt003lhm` | `artifacts/round2_cci/ed-r4bankbj2-0924052624-dde08d/` |
| assembly failed `pt-m52qwunf` | `artifacts/round2_cci/ed-r4bankbj3-0924053507-ae0123/` |
| assembly retry `pt-831c01g4` | `artifacts/round2_cci/ed-r4bankbj3r-0924054117-185e82/` |

X1 N32/N64的原始字段已作为`results/x1_results_X1_N32.json`和`x1_results_X1_N64.json`里的`job_result`读到，不需为了确认探索训练存在而再次索要相同材料。当前S0/Fs-v2/bank的部分rc/elapsed只在task_result字符串中；这是补原件的原因，不是认定计算没发生。

3. Fs v2与bank完整资格/manifest/registry的原件或已有checkout文件。预检会按实际字节校验，不需新造摘要。
4. 全项目曝光台账及其实际issue列表，至少包括v1、X1 N32/N64、v2、FP04、所有debug/旧筛选窗口。不能只交最终选中模型的训练清单。

## 在GPU cache前需要
- `fs_adapter.pt`：`plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/certify/fs/fs_adapter.pt`，记录SHA `7b39226b550a2795b1ed85020057bec46064f132b1c36d77a4db9b5682323089`。
- `fs_merged_backbone.pt`：同目录，记录SHA `1e48438ee98dc72552786011d25d9db3cc208672ceb8a7fc725c7f8fde77a79b`。
- `bank.pt`：`plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/certify/bank/bank.pt`或原`artifacts/round2_cci/ed-r4bankbj3r-0924054117-185e82/assemble/bank.pt`，记录SHA `c3e3e34b60b463b41a6c824fa680d844a2830cd2c7b40d3fc4544dc985b85f81`。
- 四个`expert_k.pt`和保存probe（用于已有certificate独立重算/同源片段验证），原目录`artifacts/round2_cci/ed-r4bankbj2-0924052624-dde08d/expert{k}/`。
- S0 gate config、official manifest、1/4/12输出tensor、pinned F0 checkpoint、原NPZ、grid/channel和软件overlay的可读路径。
- 2019 DEV与已曝光debug所需实际Zarr/UTC索引及内容证书；不读取2020H2作为confirm。
- 新GPU授权、总作业/时间/存储上限、72h允许损害及阈值理由；未冻结时BLOCKED。

## 本次尚未取得，不可假称复核
本次没有重hash大权重、没有重跑GPU/仓库pytest、没有外部时间戳签名。预注册文件sha及早于job提交的recorded_utc支持记录内部的一致性，不是外部防篡改证明。可提供独立提交记录/源快照加强，不能事后改写历史时间。

## 后续才会生成，不能要求用户“补交现有结果”
FP05完整candidate cache、OOF模型/动作freeze、paired统计表、FP06研究判决均为TO_BE_RUN。它们不是当前可索要的既有证据。
