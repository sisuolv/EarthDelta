# ACP GPU 使用与运行指南

本文针对当前 `/mnt/afs/260010168` 环境整理。资源模型是：

- **标准资源：总共 4 张独占 GPU**，通常是 4 张 H100 或 4 张 RTX 5090。已发现并验证了 H100 的单节点 4-GPU spec；5090 的单节点 4-GPU spec 已被资源清单确认，但仍应先做一次 4-GPU smoke 验收。
- **潮汐资源：额外可用的 H100 和 RTX 5090**。这类资源使用 `spot` 配额，可能排队、被抢占或失败；任务必须可恢复、可重试，不能把它们当成稳定的独占资源。

CCI 是提交和监控主机，真正的 CUDA 计算运行在 ACP worker 容器中。CCI 本地 `nvidia-smi` 不代表 ACP 是否有 GPU。

## 1. 当前已验证的环境

| 项目 | 值 |
|---|---|
| AFS 根目录 | `/mnt/afs/260010168` |
| workspace | `share-space` |
| 区域 / 计算 zone | `cn-sh-01` / `cn-sh-01g` |
| AFS volume | `01a04263-91e5-7603-bc01-c67e503da6b5` |
| 固定 SCO launcher | `/mnt/afs/260010168/bin/sco` |
| SCO 配置 | `/mnt/afs/260010168/.config/sco` |
| 资源模型 | `reserved` 独占；`spot` 潮汐 |

已验证的单卡配置如下。worker spec 的大小写必须保持一致。

| 用途 | AEC2 cluster | GPU | worker spec | CPU / RAM | 镜像 |
|---|---|---|---|---:|---|
| 标准 H100 单卡 | `share-cluster` | H100 80GB | `n6ls.iu.i40.1.4c64g` | 4 / 64 GiB | CUDA 12.4、PyTorch 2.3 镜像 |
| 标准 H100 4卡 | `share-cluster` | 4×H100 80GB | `n6ls.iu.i40.4.32c512g` | 32 / 512 GiB | CUDA 12.4、PyTorch 2.3 镜像 |
| 标准 RTX 5090 4卡 spec | `computing-cluster-5090-01g` | 4×RTX 5090 32GB | `n12lp.nn.i10a.4` | 56 / 240 GiB | CUDA 12.9、PyTorch 2.8 镜像 |
| 潮汐 H100 | `computing-cluster-01g-02` | H100 80GB | `N6lS.Iu.I10.1.8c128g` | 8 / 128 GiB | CUDA 12.4、PyTorch 2.3 镜像 |
| 潮汐 RTX 5090 | `computing-cluster-5090-01g` | RTX 5090 32GB | `n12lp.nn.i10a.1` | 14 / 60 GiB | CUDA 12.9、PyTorch 2.8 镜像 |

H100 镜像：

```text
registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image
```

RTX 5090 镜像：

```text
registry.cn-sh-01.sensecore.cn/lepton-trainingjob/ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04
```

单卡 spec 请求一个 GPU；4-GPU spec 请求同一 worker 内的四张 GPU。`worker-nodes=5` 配合单卡 spec 则是五个 worker 节点、五张 GPU，不是单节点 5-GPU。需要 NCCL/DDP 时优先使用已确认的 4-GPU spec；需要独立 shard 时可以提交四个单卡 job。

## 2. 提交前检查

### 2026-09-30 修复记录：旧 SCO 下载地址 404

当前固定入口 `/mnt/afs/260010168/bin/sco` 已更新为 **SCO 2.0.2 / release 20260830**。2.0.1 的 cnsh01 runtime 文件此前缺失，启动器尝试下载已失效的旧包；该 URL 直连和通过 `127.0.0.1:17890` 都返回 404。代理不能恢复服务器上不存在的旧文件。

本次通过代理下载官方 `https://sco.sensecore.cn/registry/install.sh`，在隔离目录验证 20260830 发布包及其 SHA256 后切换安装；保留账户配置和旧备用程序。AFS 解包需要 `TAR_OPTIONS=--no-same-owner`，避免归档属主恢复失败。修复证据与备份在 `/mnt/afs/260010168/sco_repair_20260930T073114Z/`。

日常继续使用固定入口，不必手动切换旧备用程序。诊断应同时执行 `sco version`、`sco doctor --json` 和一次只读 `acp jobs describe`；`version` 能成功只证明 launcher 可运行，不证明 runtime 或 ACP 连接可用。再次出现 404 时先核对 URL、release 与 runtime 缓存，不连续重试作业，也不要重新 `sco init` 或重写凭证。旧版备用程序只作为回退，不作为当前默认入口。

在 CCI 上执行：

```bash
findmnt -T /mnt/afs/260010168
/mnt/afs/260010168/bin/sco version
/mnt/afs/260010168/bin/sco config get region
/mnt/afs/260010168/bin/sco config get zone
/mnt/afs/260010168/bin/sco ws instances describe --name=share-space --format=json
```

如果需要加载当前 shell 的环境：

```bash
. /mnt/afs/260010168/init-codex.sh
```

必须确认 `/mnt/afs/260010168` 位于真实 AFS mount 上。不要在 AFS 未挂载时创建本地同名目录，也不要把结果写到 CCI 的临时磁盘。

查看当前 ACP 参数：

```bash
/mnt/afs/260010168/bin/sco acp jobs create --help
/mnt/afs/260010168/bin/sco acp jobs describe --help
```

在当前 v1 ACP binary 中，`create` 命令不要添加 `--format`；查询 job 时使用 `describe -o json` 或 `describe --format=json`。

如果固定 launcher 因 SCO registry package 更新失败而不能执行 ACP 命令，可使用当前已验证的 v1 binary：

```bash
export ACP_SCO=/mnt/afs/260010168/.sco/state/regions/cnsh01/quarantine/20260920T105828/sco
export SCO_HOME=/mnt/afs/260010168/.sco
export SCO_DATA_HOME=/mnt/afs/260010168/.local/share/sco
export SCO_CONFIG=/mnt/afs/260010168/.config/sco
"$ACP_SCO" version
```

优先使用 `/mnt/afs/260010168/bin/sco`；只有它报告 package/cache 错误时才切换到上述已验证 binary。不要重新安装、升级或初始化 SCO，也不要在文档、命令行参数或日志中写入 API key。

## 3. 每个实验建立独立 run 目录

每次提交必须生成新的 run ID，不能复用旧结果目录：

```bash
export BASE=/mnt/afs/260010168
export RUN_ID="my_project_$(date -u +%Y%m%dt%H%M%Sz)"
export RUN_DIR="$BASE/acp_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
```

建议至少保存以下文件：

```text
$RUN_DIR/
├── request.json              # GPU、seed、代码版本、输入输出合同
├── command.txt               # worker 中实际执行的命令
├── create-argv.json          # 完整 ACP create 参数
├── submission-response.json  # 提交 stdout/stderr/return code
├── job-id.txt                # ACP 返回的 pt-* ID
├── describe_initial.json
├── worker.log
├── result.json
├── describe_final.json
└── validation.json
```

代码、配置、输入和结果都放在 AFS；在提交前冻结脚本并记录 SHA256：

```bash
sha256sum /mnt/afs/260010168/path/to/worker.py
```

worker 命令必须明确写出真实计算入口。ACP create 的默认 command 是 `sleep inf`，不能依赖默认值，也不能提交空转或 keepalive 任务。

## 4. 四张独占 H100：推荐的四 shard 提交方式

下面的模板把一个实验拆成四个单卡 shard，刚好使用四张标准 H100。把 `SCRIPT`、参数和输出路径改成实际项目内容。

```bash
set -euo pipefail

BASE=/mnt/afs/260010168
SCO=/mnt/afs/260010168/bin/sco
# 如果上面的 launcher 不能执行，再改为已验证的 ACP_SCO 路径。
WORKSPACE=share-space
CLUSTER=share-cluster
SPEC=n6ls.iu.i40.1.4c64g
IMAGE='registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image'
SCRIPT=/mnt/afs/260010168/my_project/scripts/run_experiment.py
RUN_ID="my_experiment_$(date -u +%Y%m%dt%H%M%Sz)"
RUN_ROOT="$BASE/acp_runs/$RUN_ID"
mkdir -p "$RUN_ROOT"

for shard in 0 1 2 3; do
    JOB_NAME="${RUN_ID}-s${shard}"
    OUT="$RUN_ROOT/shard${shard}/result.json"
    mkdir -p "$(dirname "$OUT")"
    CMD="timeout 3600s python -u $SCRIPT --out $OUT --shard $shard --shards 4 --device cuda"

    printf '%s\n' "$CMD" > "$RUN_ROOT/shard${shard}/command.txt"
    "$SCO" acp jobs create \
        --workspace-name="$WORKSPACE" \
        --aec2-name="$CLUSTER" \
        --job-name="$JOB_NAME" \
        --container-image-url="$IMAGE" \
        --training-framework=pytorch \
        --worker-nodes=1 \
        --worker-spec="$SPEC" \
        --storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs \
        --quota-type=reserved \
        --priority=NORMAL \
        --command="$CMD" \
        > "$RUN_ROOT/shard${shard}/submission-response.json" 2>&1

    rc=$?
    printf '%s\n' "$rc" > "$RUN_ROOT/shard${shard}/submission-exit-code.txt"
    cat "$RUN_ROOT/shard${shard}/submission-response.json"
    if [ "$rc" -ne 0 ]; then
        echo "submission failed for shard $shard; inspect the saved response before retrying" >&2
        exit "$rc"
    fi
    # 当前 CLI 返回格式通常为: job pt-xxxxxxxx submitted successfully
    job_id=$(sed -nE 's/.*job (pt-[a-z0-9]+) submitted successfully.*/\1/p' \
        "$RUN_ROOT/shard${shard}/submission-response.json" | head -1)
    test -n "$job_id"
    printf '%s\n' "$job_id" > "$RUN_ROOT/shard${shard}/job-id.txt"
done
```

四个 job 是四个独立的单 GPU allocation，总 GPU 数为 4。它们共享同一个 frozen code version，但必须有不同的 shard 输出目录。若任务是 ensemble、Monte Carlo、数据分片或独立实验，这是推荐方式。

如果需要真正的分布式训练，不要直接复制上述循环。先确认 ACP 的多 worker 网络/环境变量合同，再使用 `--worker-nodes=4`、分布式启动器和共享 checkpoint；否则四个独立 job 会各自重复训练。

## 5. 潮汐 H100/5090 提交方式

潮汐任务使用 `spot` 配额，可能被抢占。因此必须满足：

- 输入可重复，输出写入唯一 run 目录；
- 定期 checkpoint 到 AFS；
- worker 可以从 checkpoint 恢复；
- 不把一次 `submitted successfully` 当成完成；
- 不要依赖 H100 的 80GB 显存，5090 只有约 32GB；
- 需要容错时才启用 fault tolerance 和有限 retries。

潮汐 H100 示例：

```bash
RUN_ID="my_experiment_h100_spot_$(date -u +%Y%m%dt%H%M%Sz)"
RUN_DIR="/mnt/afs/260010168/acp_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
SCO=/mnt/afs/260010168/bin/sco
CMD="timeout 3600s python -u /mnt/afs/260010168/my_project/scripts/run_experiment.py --device cuda --out $RUN_DIR/result.json"
"$SCO" acp jobs create \
  --workspace-name=share-space \
  --aec2-name=computing-cluster-01g-02 \
  --job-name="$RUN_ID-h100-spot" \
  --container-image-url='registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image' \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N6lS.Iu.I10.1.8c128g \
  --storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs \
  --quota-type=spot \
  --priority=NORMAL \
  --enable-fault-tolerance=true \
  --retry-times=2 \
  --command="$CMD"
```

潮汐 RTX 5090 示例：

```bash
RUN_ID="my_experiment_5090_spot_$(date -u +%Y%m%dt%H%M%Sz)"
RUN_DIR="/mnt/afs/260010168/acp_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
SCO=/mnt/afs/260010168/bin/sco
CMD="timeout 3600s python -u /mnt/afs/260010168/my_project/scripts/run_experiment.py --device cuda --out $RUN_DIR/result.json"
"$SCO" acp jobs create \
  --workspace-name=share-space \
  --aec2-name=computing-cluster-5090-01g \
  --job-name="$RUN_ID-5090-spot" \
  --container-image-url='registry.cn-sh-01.sensecore.cn/lepton-trainingjob/ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04' \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=n12lp.nn.i10a.1 \
  --storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs \
  --quota-type=spot \
  --priority=NORMAL \
  --enable-fault-tolerance=true \
  --retry-times=2 \
  --command="$CMD"
```

是否允许 `spot`、是否需要 `--wait`、以及可用配额会随平台变化。若返回 429 quota exceeded，应保存原始 response 并停止重复提交；不要把改变 cluster、worker spec 或 quota type 当成盲目重试策略。

### 已验证的五张 RTX 5090 spot 任务

当前 workspace 已成功接受并完成以下测试：

- job：`pt-6nu1f8n8`
- quota：`spot`
- 请求：`worker-nodes=5`、`n12lp.nn.i10a.1`，即 5 个 worker × 1 张 RTX 5090
- 平台状态：`SUCCEEDED`
- 5 个 worker host、5 个 CUDA result 均通过
- 完整 receipt：`/mnt/afs/260010168/acp_runs/acp-spot-5090x5-20260925t061116z/`

可复用的核心提交形态是：

```bash
"$SCO" acp jobs create \
  --workspace-name=share-space \
  --aec2-name=computing-cluster-5090-01g \
  --job-name=RUN_ID-5090x5-spot \
  --container-image-url='registry.cn-sh-01.sensecore.cn/lepton-trainingjob/ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04' \
  --training-framework=pytorch \
  --worker-nodes=5 \
  --worker-spec=n12lp.nn.i10a.1 \
  --storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs \
  --quota-type=spot \
  --priority=NORMAL \
  --command='timeout 300s python -u /mnt/afs/260010168/acp_runs/RUN_ID/spot_5gpu_worker.py ...'
```

这个测试证明的是“标准 4 GPU 之外，spot 可以补足第 5 个 GPU”，不证明存在单节点 5-GPU 5090 spec。若需要单节点 5 卡，必须先从 ACP 资源清单获得明确的 5-GPU worker spec；不能用 `worker-nodes=5` 代替。

## 6. 标准 4-GPU 单节点 smoke

H100 4 卡 worker 已经有成功的 NCCL smoke receipt。新 run 应复制 `.cci/acp-smoke-4gpu.py`，在 `request.json` 中设置 `expected_gpu_count=4`，然后使用：

```bash
--aec2-name=share-cluster \
--worker-nodes=1 \
--worker-spec=n6ls.iu.i40.4.32c512g \
--quota-type=reserved \
--command='timeout 300s python -u /mnt/afs/260010168/acp_runs/RUN_ID/smoke.py'
```

5090 的 4-GPU spec 为 `n12lp.nn.i10a.4`，使用 RTX 5090 镜像和相同的 `acp-smoke-4gpu.py` 合同；提交后必须验收 `visible_gpu_count == 4`、四个设备名均为 RTX 5090、每个设备 matmul 正确且 NCCL all-reduce 成功。目前该 5090 4-GPU组合是资源清单证据，不能引用为已完成的 4-GPU运行结果，直到 smoke receipt 写回。

## 7. 查询 job 和实时日志

```bash
SCO=/mnt/afs/260010168/bin/sco
WORKSPACE=share-space
JOB_ID="$(cat /mnt/afs/260010168/acp_runs/RUN_ID/job-id.txt)"

"$SCO" acp jobs describe \
  --workspace-name="$WORKSPACE" \
  --format=json "$JOB_ID" \
  > /mnt/afs/260010168/acp_runs/RUN_ID/describe_latest.json

"$SCO" acp jobs stream-logs \
  --workspace-name="$WORKSPACE" "$JOB_ID"

"$SCO" acp jobs list \
  --workspace-name="$WORKSPACE" --format=json
```

提交状态可能是 queued、running、succeeded、failed 或平台终止状态。保存至少一次初始 describe 和一次最终 describe。若提交命令超时，状态是不确定的：先按 job display name 或 list 查询，确认没有已创建的 job，再决定是否提交，避免重复消耗 GPU。

只停止明确的 job ID：

```bash
"$SCO" acp jobs stop --workspace-name=share-space "$JOB_ID"
```

## 8. 成功验收标准

必须同时满足以下条件才把任务标记为成功：

1. ACP final state 是 `SUCCEEDED`；
2. final describe 显示预期 cluster、worker spec 和 GPU 数；
3. worker 的 `result.json` 存在且内容完整；
4. `status` 为 `ok`，实际 `cuda_available` 为 true；
5. 实际 GPU 名称符合请求（H100 或 RTX 5090）；
6. 代码、配置、run ID、seed 和输出路径与 request/manifest 一致；
7. 结果 hash 与保存的 artifact manifest 一致；
8. 多 shard 的 denominator、失败数和缺失结果都被计入，不能只汇报成功 shard。

一个最小的 worker 结果检查示例：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('/mnt/afs/260010168/acp_runs/RUN_ID/result.json')
a = json.loads(p.read_text())
assert a.get('status') == 'ok'
assert a.get('cuda_available') is True
print(a.get('cuda_device'), a.get('torch'), a.get('elapsed_seconds'))
PY
```

`submitted successfully` 只代表请求被 ACP 接受，不代表 GPU 已运行成功。对潮汐任务尤其不能把日志开始、部分输出或手动 stop 后的结果写成 clean success。

## 9. 常见错误

### `429 Too Many Requests` / `quota exceeded`

说明当前 workspace 或 cluster 没有可用配额。保存 response，查询已有 job；不要自动循环重试。标准 4 H100 使用 `reserved`，潮汐任务使用 `spot`，但实际可用性以 ACP 返回为准。

### create 命令出现 `unknown flag: --format`

当前 create 子命令不接受该参数。去掉它；对 describe/list 使用 `--format=json` 或 `-o json`。

### job 一直运行但没有结果

检查 startup command、脚本绝对路径、容器内 Python 依赖和 AFS mount。不要把 command 留空，因为默认值是 `sleep inf`。

### H100 脚本提交到 5090 失败

5090 使用单独的 CUDA 12.9 镜像和 `n12lp.nn.i10a.1` spec。不要把 H100 镜像、H100 显存假设或 H100 batch size 直接复用到 5090。

### AFS 路径不存在

先执行 `findmnt -T /mnt/afs/260010168`。确认 worker 的 `--storage-mount` 是：

```text
01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs
```

### submission 命令超时

这是 unknown submission state。不要立即重复提交；先用 display name/job list 查询，确认是否已经创建。

## 10. 安全和复现规则

- API key、SCO token、authorization header 只能由现有环境读取，不能写进脚本、JSON、日志、argv 或 Git。
- 每个实验使用新的 run ID 和输出目录，不覆盖历史结果。
- 标准独占资源最多并行占用 4 张 H100；额外潮汐任务单独记录 quota type 和抢占风险。
- 不读取未经授权的 holdout/quarantine 数据，不把 synthetic 结果写成真实天气结果。
- 记录 HEAD、脚本 hash、镜像、worker spec、seed、命令、job ID、最终状态和 artifact hash。
- 任务完成后释放不再需要的资源；不要提交 `sleep inf` 作为 GPU 保活。

历史验证材料：

- `/mnt/afs/260010168/ACP-GPU-QUICKSTART.md`
- `/mnt/afs/260010168/.cci/SCO-ACP.md`
- `/mnt/afs/260010168/acp_runs/acp-smoke-20260908T021913Z/`
