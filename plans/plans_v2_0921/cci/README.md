# CCI / ACP 启动、排队与实验回收

本目录提供已实际提交验证的启动工具。用户已授权免费使用 CCI 资源，包括多个 4×H100 任务排队。工程启动成功、科学 S0 通过、天气收益成立是三个不同结果；后两者仍需完成主计划中的实现和实验。

实际验收记录见 [CCI_VALIDATION.md](../CCI_VALIDATION.md)。当前主机负责编辑、提交和监控，重计算通过 ACP worker 执行，不依赖编辑器所在机器有 GPU。

## 1. 已验证的资源入口

| 参数 | 值 |
| --- | --- |
| CLI | `/mnt/afs/260010168/bin/sco`，使用现有身份配置 |
| 工作区 / 集群 | `share-space` / `share-cluster` |
| Worker | 1 节点，`n6ls.iu.i40.4.32c512g` |
| 实测设备 | 4×NVIDIA H100 80GB HBM3 |
| 规格 CPU / RAM | 32 CPU / 512 GiB |
| AFS | `01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs` |
| 队列方式 | `--quota-type=reserved --priority=NORMAL --wait` |
| 框架 | `pytorch` |

镜像固定在 `submit_job.py::IMAGE`，当前为 NVIDIA 24.04 / CUDA 12.4 / PyTorch 2.3.0a0 的已有镜像。该镜像缺少若干科学依赖，见第 5 节，不能直接作为最终官方 Stormer 环境。

普通 SCO 分发入口当前尝试下载已返回 404 的 v1 组件。已实测以下局部环境设置能使用现有客户端中的 ACP 命令：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs list \
  --workspace-name=share-space --aec2-name=share-cluster --page-size=20 -o json
```

它保留现有认证，不修改全局 SCO 配置。若后续 CLI 升级，重新检查本地调用行为，不把本次 workaround 当作平台永久接口。

## 2. 直接启动已有验证任务

从仓库根目录运行：

```bash
cd /mnt/afs/260010168/EarthDelta
python3 plans/plans_v2_0921/cci/submit_job.py --label probe --timeout-seconds 900
```

默认任务实际执行四卡 GEMM、NCCL all-reduce、AFS 读写、关键 Python 模块导入、ps4 文件存在/大小、真实 NPY 第一个样本的有限值检查。它不会反序列化 checkpoint，也不做官方 S0 或天气效果评估。所用 `gpu_smoke.py` 是本机既有 `.cci/acp-smoke-4gpu.py` 的原样副本。

对当前三组已审查测试运行回归：

```bash
python3 plans/plans_v2_0921/cci/submit_job.py --label regression --timeout-seconds 900 \
  --argv-file plans/plans_v2_0921/cci/regression_argv.json
```

该任务在 ACP worker 中以 CPU 跑 `test_metric_contract.py`、`test_s0_fail_closed.py`、`test_differentiable_rollout.py`，复用现有 `.pydeps`；它不会升级共享环境。本次实测 49 passed。后续修改代码后应重跑受影响测试；旧通过记录不证明新补丁通过。

在每张 H100 上验证当前小型合成 Stormer/LoRA 的前向、固定系数 bank 梯度和零编辑恢复：

```bash
python3 plans/plans_v2_0921/cci/submit_job.py --label gpu-bank --timeout-seconds 900 \
  --argv-file plans/plans_v2_0921/cci/gpu_bank_argv.json
```

这项检查使用当前 bridge 的小型合成网络和 SDPA 路径，不加载真实 checkpoint，也不证明官方 xformers S0。它验证训练基础调用链确实能在四张卡上分别运行；执行结果保存在 `gpu_bank_result.json`。

工具返回新的 `run_dir` 和平台 `job ID`，实际文件位于 `artifacts/round2_cci/<run_id>/`。`--dry-run` 会准备快照和提交参数但不提交任务，可用于检查配置；不会产生 GPU 成绩。

## 3. 查询、日志与退出状态

将以下 `JOB_ID` 替换为本次返回的 ID：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs describe JOB_ID \
  --workspace-name=share-space -o json
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs get-workers JOB_ID \
  --workspace-name=share-space
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs stream-logs JOB_ID \
  --workspace-name=share-space
```

主程序 stdout/stderr 在共享 `run_dir/worker.log`，不依赖平台日志流保持连接。结束后必须读取 `job_result.json`：包含真实返回码、耗时和 SUCCEEDED/FAILED/TIMEOUT；平台状态与该记录都要核查。缺记录不能推断成功。

提交命令超时或返回格式无法解析时，先按唯一显示名查询：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs list \
  --workspace-name=share-space --display-name=RUN_ID -o json
```

不要因本地提交超时就重复提交。只有需要结束本次创建的挂起/错误任务时使用：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs stop JOB_ID \
  --workspace-name=share-space
```

不要停止用户其它运行或数据拉取任务。作业超时由 `run_job.py` 终止自己创建的子进程组，不影响其它 job。

## 4. 提交真实 S0、训练、缓存或评估

提交工具是通用 argv 执行器。为已经实现的实验入口准备 JSON 数组，再使用 `--argv-file`：

```json
[
  "python3",
  "scripts/r2_bank_oracle.py",
  "--stage", "cache",
  "--config", "@OUTPUT@/pilot_config.json",
  "--out", "@OUTPUT@/cache"
]
```

上面是未来实验入口的结构示例，**当前 `r2_bank_oracle.py` 尚未实现，且示例配置也不会由工具自动生成**。Claude 必须先实现/核验真实 CLI，并在提交前准备不再修改的配置。推荐把配置放在已创建的独立研究运行目录，以其绝对路径写入 argv；不能提交后才补文件，以免已调度 worker 抢先启动。

例如对已准备好的 argv：

```bash
python3 plans/plans_v2_0921/cci/submit_job.py --label cache-shard0 --timeout-seconds 28800 \
  --argv-file /实际已准备的运行目录/cache-shard0.argv.json
```

支持三个替换值：`@SNAPSHOT@` 是冻结源码目录，`@OUTPUT@` 是此次 job 输出目录，`@ASSETS@` 是原仓库资产路径。命令数组以 subprocess 执行，不通过 shell 拼接；需要环境变量时参考 `regression_argv.json` 的 `env` 用法。不要把 API 密钥放入数组，因为数组会记录到文件。

快照包含当前 `earthdelta/`、`scripts/`、`tests/` 与官方 reference 的代码/小配置及本目录 Python 工具；包括这些目录内新建但尚未提交的源码。它不复制 NPY、Zarr、checkpoint、`.git`、缓存或依赖目录。完整 manifest 逐文件记录 hash；worker 开始时核验。

资产与专用环境仍在 AFS 外部路径，必须由实验 config/manifest 单独绑定。提交工具不替实验验证数据完整性，不自动补额外配置，也不提供科学准入证书。官方源码的 commit 来源与内容 hash 应在 S0 配置里明确；源码快照没有 `.git`，不能依赖 worker 在 snapshot 中执行 Git 来推断原提交。

四卡首轮优先采用独立进程，而非共享 bridge 的多线程或马上引入 DDP：

- Fs 训练完成并冻结之后，各 GPU 训练一个动态专家；每进程独立模型和输出。
- 缓存按稳定 issue ID 分片；每张卡的分片包含该起报全部候选，零编辑也保留。
- merge 必须核验起报/候选覆盖、重复项、失败项、bank/Q/source/config 身份；不按成功子集评分。
- 独立 head seeds/HPO 可并行；前驱失败时不启动依赖后继。

多个独立作业可以排队。不要把所有阶段预先提交为占卡等待任务；控制节点轮询前驱证书并提交已就绪的后继。排队期间继续可独立的代码/分析工作，并定期报告真实状态。

## 5. 环境准备必须进入执行任务

实际探针显示默认镜像有 Torch/torchvision/numpy/scipy/pytest，但没有 timm、xformers、pytorch_lightning、zarr、xarray。仓库 `.pydeps` 可支持已验证的轻量回归，但不能直接推断它满足官方 S0。

Claude 应建立独立、版本记录完整的 Stormer 环境，安装必要依赖；此操作已在当前授权内。使用专用 venv/conda 或经过验证的镜像，避免修改共享用户环境。尤其验证 Torch/CUDA/xformers 的二进制兼容性和实际 attention 前向/反向，不能仅 import 成功就视为可用；不要盲目 pip 升级镜像内 Torch。

完成环境后，先跑官方模块加载和极小输入/真实一步，再跑完整身份绑定的 S0。历史任务曾出现 `MockModule` 反序列化错误，不允许通过伪造 Lightning 或官方模块绕过；按可信 checkpoint 和真实依赖解决。

目前已有资产：

- `checkpoints/stormer_1.40625_patch_size_4.ckpt`，本次见到 5,570,407,547 字节；未因大小匹配即认证内容。
- `reference/stormer/normalization_constants/`。
- `scripts/s0_gate_inputs/jan2020_full.npy`，本次读取 shape 为 `[124,69,128,256]`，FP32，首样本 finite；它没有因该检查就具备完整实际时间/target 证书。
- `data/era5_1p40625/` 和 `data/splits/`；用户仍有多年份拉取流程，正式读取按实际内容与 time index 准入，不把 marker 或年份目录作为已完成证明。

已用于启动、S0/debug 的样本标为 exposed，不能进入 confirm。

## 6. 免费资源下的效果评估仍须完整

启动成功后继续主计划的真实 S0、Fs/bank、完整缓存与 cheap legal。首轮 8 个起报用于调试，随后 128 个开发起报起步，按预先定义的开发采样计划和实际独立过程数扩展。正式确认依据功效另定 N，不把这些数字当独立样本量或保证收益。

每方法报告实际 native loss/gain、相对 Fs 和固定候选的增量、配对 CI、72h guard、失败率和成本。记录 GPU 秒、仿真数、存储与重试，费用为零不能替代计算成本比较。只有实际完整结果支持研究结论。

## 7. 可选 SiliconFlow API

入口：`https://api.siliconflow.cn/v1`。用户允许的模型候选：`zai-org/GLM-5.3`、`deepseek-ai/DeepSeek-V4-Flash`、`Qwen/Qwen3.8-27B`。名称来自用户，运行前应查询实际可用性，不宣称本次已经调用测试。

密钥只通过私有环境变量 `SILICONFLOW_API_KEY` 或用户的安全运行配置注入。不要使用 `--env=KEY=明文`、命令参数或提交 JSON 传递它；这些内容会进入平台或本地记录。现有授权并不要求天气主流程必须依赖 API。

API 可用于辅助代码/日志分析；不能替代真实天气监督、bootstrap 结果或研究判断。若使用流式 OpenAI 兼容 SDK，对可选返回字段用 `getattr`，不要假定每个模型都有 `reasoning_content`。本次启动验证没有调用外部语言模型。
