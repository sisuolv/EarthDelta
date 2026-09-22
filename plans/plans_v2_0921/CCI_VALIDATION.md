# CCI 启动与执行验证记录

日期：2026-09-21 UTC。此次依据用户明确授权实际提交 ACP 任务，不是历史日志重述或 dry-run。

研究源码 HEAD：`403b55db65f4c35c1a85d0794ad0de2765b07d96`，仍与前次受审源码对应。本次新增任务工具并更新执行交接文档，没有修复原审计 findings 或训练正式天气模型。

## 1. 实际提交的三个任务

| Job ID | 用途 | 实际结果 | Worker 程序耗时 |
| --- | --- | --- | --- |
| `pt-7p3wwd1j` | 四卡 GPU/NCCL/AFS 和环境/资产探针 | 硬件探针通过；发现部分依赖缺失 | 14.767 秒 |
| `pt-s26vg7xa` | 当前仓库三组 CPU 回归，在 ACP worker 执行 | 49 passed，pytest 4.21 秒，退出码 0 | 6.341 秒 |
| `pt-fzowwohr` | 四卡分别运行小型合成 Stormer/LoRA | 四张卡全部通过非零编辑、恢复和反传检查 | 8.408 秒 |

耗时不包含平台排队/调度，不是正式训练吞吐。全部使用 1 worker、4×H100 的既有规格，结果来自每个作业的独立源码快照。平台最终状态记录与任务返回码见对应目录。

证据目录：

- [启动探针结果](../../artifacts/round2_cci/ed-launch-check-0921121503-e83bae/environment_probe.json)、[四卡结果](../../artifacts/round2_cci/ed-launch-check-0921121503-e83bae/gpu_smoke/result.json)、[任务返回码](../../artifacts/round2_cci/ed-launch-check-0921121503-e83bae/job_result.json)。
- [回归日志](../../artifacts/round2_cci/ed-regression-0921121659-e0c411/worker.log)、[JUnit XML](../../artifacts/round2_cci/ed-regression-0921121659-e0c411/pytest.xml)、[任务返回码](../../artifacts/round2_cci/ed-regression-0921121659-e0c411/job_result.json)。
- [GPU bank 检查](../../artifacts/round2_cci/ed-gpu-bank-0921122443-50bd07/gpu_bank_result.json)、[任务返回码](../../artifacts/round2_cci/ed-gpu-bank-0921122443-50bd07/job_result.json)。

每个目录还包含 `invocation.json`、`source_manifest.json`、`create_argv.json`、`submission.json`、`job-id.txt`、`platform_status.json` 和 worker 日志；这些是本次创建的证据，不替代后续新代码的测试。

## 2. 实际验证了什么

四张卡均为 NVIDIA H100 80GB HBM3，可见显存约 79.179 GiB。每张卡的 GPU 矩阵运算与 CPU 参考一致；四 rank NCCL all-reduce 均得到预期和 10；AFS 写入后读回一致。

小型合成 Stormer/LoRA 在四卡上的结果一致：

- 非零固定系数编辑的最大输出变化约 `2.5296e-4`。
- 零编辑恢复的最大误差 `0.0`。
- bank 梯度绝对值和约 `1.4065e-3`，梯度有限且非零。
- 冻结 backbone 没有梯度，退出后残留 forward hook 数为 0。

该测试使用小型合成网络，模型/输入/权重由固定 seed 生成；不是官方预训练 checkpoint，不是实际天气收益。它支持“当前训练基础路径可在 H100 上执行”，不能据此关闭 B03 或其余 findings。

49 项回归来自：`tests/test_metric_contract.py`、`tests/test_s0_fail_closed.py`、`tests/test_differentiable_rollout.py`。它们以 CPU 执行，原测试覆盖不足的已知审计结论仍保留。没有把既有测试通过写成缺陷已修。

新 `run_job.py` 在控制节点还通过 4 个独立失败路径检查：成功返回、非零返回码传递、源码快照被修改时拒绝启动、超时终止。命令为 `python3 plans/plans_v2_0921/cci/test_runner.py -v`，4 tests / OK。

## 3. 实测环境与资产限制

默认 ACP 镜像可以导入：Torch `2.3.0a0+6ddf5cf85e.nv24.04`、torchvision `0.18.0a0`、numpy `1.24.4`、scipy `1.12.0`、pytest `8.1.1`。

默认镜像不能导入：timm、xformers、pytorch_lightning、zarr、xarray。回归和小型 GPU 检查显式复用了仓库已有 `.pydeps`，因此能使用 timm；该方式不证明独立官方 Stormer 的全部依赖成立。

真实 S0 尚未执行。Claude 需完成最小修复，并在专用环境中准备兼容的官方依赖，尤其验证 xformers 的实际 CUDA 算子；不能复用历史 MockModule 绕过反序列化，也不能用当前 SDPA 合成测试替代官方对齐。

本次实际见到 ps4 checkpoint 文件，大小 5,570,407,547 字节，但未反序列化、未用大小代替内容认证。实际 NPY 为 FP32 `[124,69,128,256]`，第一样本有限；未全量验证实际时间、history/target、坐标与内容完整性。这些要在 S0/pilot 入口完成。

## 4. 启动入口问题及本次处置

普通 `/mnt/afs/260010168/bin/sco acp ...` 会尝试下载一个当前返回 404 的区域组件，因此只看 help 会误以为 CLI 可用。

实测 `SCO_LAUNCHED_BY=launcher` 的进程局部设置可以调用现有客户端中的 ACP 逻辑；已实际用于查询和上述三个任务的提交。新提交器包含该设置，没有修改全局 SCO/CCI 配置、认证或用户进程。

任务提交采用 JSON argv、源码快照和唯一 run ID。命令与结果保存在 AFS；排队期间原仓库后续编辑不改变本次快照。提交状态不明确时先查唯一显示名，不盲目重复创建任务。

## 5. 本记录不能替代的后续结果

当前尚无：修复后的官方 S0、合格训练 Fs/bank、完整天气候选缓存、合法策略实际技能、双头标签效率或输出纠错挑战结论。

修订后的 [执行计划](CLAUDE_EXECUTION_PLAN.md) 与 [Claude Prompt](CLAUDE_PROMPT.md) 已要求继续完成这些真实工作。用户的 GPU/任务/API 授权已生效；不能因旧模板仍锁 GPU 而停止在 CPU 阶段。

本次没有调用 SiliconFlow；模型名称沿用用户提供的候选，尚未核验服务可用性。提供的密钥没有写入这些交付文件、命令参数或作业配置。
