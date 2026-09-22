# EarthDelta：Stage 1A 复查后的约 10 小时执行计划

分析日期：2026-09-21 UTC。目录沿用用户的 `plans_v3_0922` 命名。

本文件是后续执行方案。本次只阅读、核验和进行小型 CPU 摘要复现，没有恢复此前暂停的 Stage 1B/1C/1D，没有提交 CCI 作业或运行真实 S0。时间表中的 T+0 指后续明确恢复执行的时刻，包含编码、排队、运行、排错和结果整理。

## 1. 建议与当前判断

建议保留 `c397a7b` 的有效修改，在它之上完成一次有边界的 Stage 1A 补修，再尽快获得真实 S0 和一个小型天气实验的运行结果。不要重新启动一轮纯文档审计，也不要把当前 Stage 1A 整体视为 CLOSED。

10 小时的合理目标分三层：

| 层次 | 要拿到的结果 | 结论边界 |
| --- | --- | --- |
| 基础交付 | 当前使用路径的正确性补丁、针对性 CPU 验证、真实官方 S0 的实际运行结果或具体外部阻塞证据 | S0 失败也必须交付原因、日志和下一步；不能预先承诺通过 |
| 正常推进目标 | S0 通过后，真实 checkpoint 上的小型 Fs/bank 训练、完整候选调试缓存、开发集 cheap-policy 表 | 训练不足或样本太少时只报流程/开发证据 |
| 条件性扩展 | 依据实测速度完成最多 128 个开发起报及配对不确定性分析 | 不在本窗口承诺正式确认、双头优势或论文价值结论 |

核心交付应是“可信代码与真实运行证据”。如果只来得及修复与 S0，就明确停在这个层次；如果能完成小型真实效果表，就据此决定下一轮投入。

### 1.1 三份 Pro 复查的采纳方式

已通读本目录三份原始复查，并核对当前生产源码、官方 iterative module、提交记录与旧执行计划。三份复查都基于同一 patch packet，均没有执行仓库测试或 GPU 推理；它们形成一致的静态判断，不能算三次独立运行验收。

| 项目 | 当前判断 | 对计划的影响 |
| --- | --- | --- |
| B01 | 核心源码修复可接受；真实官方运行仍未验证 | 保留独立官方 transforms 和 iterative module，不重做这部分 |
| B02 | OPEN | 先补完整预期/参照/实际身份比较、输入链和多步数值比较 |
| B10 | PARTIALLY_CLOSED | 保留 ps4 默认；补显式参照目录、完整身份隔离和唯一选择 |
| B13/B14 | 已有附带修补，专项验收不足 | 测真实 gate/main 的异常与产物写入；不再写“完全未触碰” |
| Stage 1B/1C/1D | 尚未整体完成 | 在本轮实际使用路径内补齐，不要求先关闭全部 15 条 finding |
| 真实 S0、Fs/bank、策略效果 | 尚无本轮证据 | 历史探针和 72 passed 不能替代这些结果 |

`c397a7b` 确实修改 8 个文件，`d749a1c` 确实只新增两份复查材料。当前 HEAD 是 `d749a1c62521226df857587e08f7d067b0f15355`。原有未跟踪资产和计划仍未被 staging。

## 2. 当前源码确认的问题，以及实际新增的证据

### 2.1 必须先解决的 gate 问题

1. `scripts/s0_gate.py:910` 仍是 `check_version_match(bridge.version, bridge.version)`；失败字段也没有进入 `gate_criteria`。这是自比较，不能认证预期身份。
2. `scripts/s0_gate.py:349` 附近读取上游 manifest 后只展示 checkpoint/norm 信息，没有强制与冻结配置和实际加载对象一致。
3. `scripts/s0_gate.py:371` 起的 parity 使用保存的 `input_norm.pt`；传入的 `x_raw` 没有参与该数值链。另一 helper 验证 raw hash，并不能证明这个 normalized input 来自当前 raw。
4. `scripts/s0_gate.py:496` 的四步检查只核对存在、shape、finite；真正的数值比较仍只运行一步。有限但错误的四步张量不触发该检查失败。
5. `scripts/s0_gate.py:282` 的目录选择按字符串排序取最后一个，既不是按时间选最新，也没有按完整身份选择。旧路径 fallback 仍存在；默认 gate 输出也共用目录。

这些是验收逻辑本身的缺口。直接跑当前 GPU gate，不能补上它们没有检查的性质。

### 2.2 已确认的摘要不兼容

复查报告提出“可能不兼容”。读取完整源码并执行隔离的 CPU 复现后，可以提升为确定问题：

- bridge 在 `earthdelta/bridge/stormer_bridge.py:339` 起先写全部 interval 的 diff_mean，再写全部 diff_std。
- exporter 在 `scripts/export_upstream_reference.py:286` 起按每个 interval 交错写 mean/std。
- bridge 的 NPZ loader 还转换为 float32，exporter 则直接序列化 NPZ 原 dtype。

本次结果：

| 输入 | bridge/exporter 摘要一致？ | 说明 |
| --- | --- | --- |
| 合成 FP32、仅 6h | 是 | 单 interval 掩盖了排序差异 |
| 合成 FP32、6h/24h | 否 | 即使 dtype 相同，排序也不同 |
| 合成 FP64、仅 6h | 否 | 独立暴露 dtype 编码差异 |
| 仓库真实 NPZ、69 变量、6h/24h | 否 | bridge=`ed563eb5e55fc118`；exporter=`e2871e376eb9d9e8` |

这证明身份摘要不兼容，不证明天气数值计算不一致。当前 gate 没有比较上游摘要，因此隐藏了这个问题；直接增加等值比较后，它会拒绝正确的同源常数。应修复字节规范，不能通过忽略字段或放宽数值容差绕开。

复现文件：[digest_probe.py](evidence/digest_probe.py)、[digest_probe.json](evidence/digest_probe.json)。它用 AST 抽取当前源码中未修改的 class/function，绕开 exporter 的 GPU import 检查，实际执行摘要逻辑；没有运行生产模块导入、完整 gate 或 pytest。

```bash
python3 plans/plans_v3_0922/evidence/digest_probe.py
```

另已核对本地官方 reference 的 Git HEAD 为 `58dfee5a6037399a40fefd492bc00421e0c885a8`，其 tracked worktree 无差异，另有一个未跟踪 checkpoint。这支持当前本地源码来源，但生产 exporter 仍需把实际导入文件和 worker 快照绑定起来；把 commit 常量写进 manifest 不足以认证未来运行。

### 2.3 会影响 10 小时可行性的工程事实

- 仓库已有 bridge、LoRA 和分析原语，但尚无完整的 Fs/bank 训练、候选缓存和 cheap-policy 实验 CLI。这些实现和测试必须占用时间。
- CCI 提交工具已实际验证；历史四卡探针和合成 bank 检查成功，不需要从零探索平台。
- 历史默认镜像缺少官方运行所需的部分依赖。当前官方源码直接导入 `lightning.LightningModule`，应验证实际的 `lightning` 命名空间和完整依赖链。
- 快照不包含 checkpoint、NPY 或 `.git`。当前 gate 的 repo-root 同时控制资产路径和 Python import；不能为了读资产把 import 指回可变的原仓库。
- 真实可用的数据切片、时间索引和训练/开发隔离仍需核验。不能从一个 `[124,69,128,256]` 数组的形状推断这些条件成立。

## 3. 10 小时时间表

以下是单个主要执行者的工作安排。CCI 作业在后台执行时，继续独立的 CPU 实现与验证；不依赖额外代码代理。时段是目标分配，不是未经测量的吞吐保证。

| 墙钟时间 | 主要工作 | CCI/并行工作 | 必须留下的证据 |
| --- | --- | --- | --- |
| T+0:00–0:20 | 冻结 HEAD/工作区、运行目录、实际数据角色；核对资产与旧任务 | 准备并提交官方环境检查 J0 | source/config 草案、作业 ID、资产缺项 |
| T+0:20–2:20 | Stage 1A 补修：摘要、身份、raw 输入、多步数值、显式目录、真实编排测试 | J0 验证依赖和 xformers CUDA 算子；完成后可退出 | 正例通过、逐项破坏必败、环境结果 |
| T+2:20–3:20 | Stage 1B 原生 Q；Stage 1C 实际小切片与 registry 准入；补 B13/B14 验收 | Stage 1A 及其异常测试通过后，可提交 J1 官方 S0 | 端点 loss/gain 一致性、数据端点证据、S0 开始日志 |
| T+3:20–4:30 | Stage 1D 定向回归；实现/验证最小 trainer、完整缓存及评分入口 | J1 完成 1/4 步 S0、必要的 12 步验证与训练步 profile | 数值证书或真实失败；耗时/显存；可运行 CLI |
| T+4:30–5:20 | 冻结本轮小实验；训练并冻结 Fs，验证合并/常开等价性 | J2 使用真实 ps4 训练；先短程检查再有界训练 | Fs 训练侧记录、身份、no-edit/continuation 检查 |
| T+5:20–6:30 | 训练并检查 K=4 小库；实现合并检查与 cheap evaluator 的剩余部分 | 四卡独立进程各训练一个专家 | A/B 梯度、非零响应、恢复、训练侧多样性 |
| T+6:30–8:10 | 8 个 exposed 起报调试；通过后运行预先冻结的开发子集 | 四卡按起报分片，每个起报跑全部候选 | 完整 cache manifest、所有候选 loss/gain、失败和成本 |
| T+8:10–9:00 | Fs、固定候选、regime、ridge direct-gain、oracle 决策表 | cheap 方法优先 CPU；必要时回收已有缓存分片 | OOF/时间隔离检查、合法动作、开发统计 |
| T+9:00–10:00 | 整理结果、必要的最后回归、差异复核、状态交接 | 不再启动预计无法在截止前完成的大作业 | decision/HANDOFF、真实 job 状态、下一阶段建议 |

S0 的官方对齐不依赖 Q reducer，可以与 Stage 1B 在时间上重叠；训练和 native scoring 必须等各自准入通过。J1 若采用较早快照，后续只要改动影响 bridge、归一化、源码身份或 rollout，就要重新验证受影响 S0，不能以旧证书给新代码放行。

### 3.1 四个转向时点

| 时点 | 若正常 | 若未达到条件 |
| --- | --- | --- |
| T+2:20 | Stage 1A 正例和反例通过，准备 S0 | 继续补修；收缩后续开发样本和实验范围，不跳过身份约束 |
| T+4:30 | 真实 S0 通过，实际切片和训练入口可用 | 后半程优先修复并拿到 S0；取消本窗口大缓存/策略目标 |
| T+6:30 | Fs/bank 达到预先定义的资格，开始开发缓存 | 交付训练/profile 和资格不足原因；不把未训好的库当科学失败 |
| T+8:10 | 完整开发缓存可评分 | 只评完整、事先定义且未按结果筛选的子集；不足则保留描述性 debug，不发布完整 oracle/策略结论 |

T+9:00 后预留完整一小时收尾。排队超过预期时压缩科学扩展，保留正确性、运行记录和交接；10 小时不能靠少记排队时间来满足。任务 runtime timeout 不含队列等待，应另外以墙钟截止时间管理本次任务；只处理本次创建的任务，不触碰用户其它作业。

## 4. 前 3–4 小时必须具体做什么

### 4.1 三个聚焦的补丁，不重新堆积大 diff

**补丁 A：摘要与预期身份。**

定义一个轻量、无 GPU 依赖的身份序列化规范：schema version、policy、变量顺序、interval 集合、shape、确定的 dtype/字节序/字段顺序。可共用序列化函数，官方与 bridge 的数值 transforms 必须保持独立。有效计算常数的 digest 与原始 NPZ 文件内容 hash 分开，避免把存储 dtype 和实际执行 dtype 混为一谈。若统一到 FP32，必须与实际 FP32 执行一致，不能只改 hash 掩盖真实数值差异。

保留 legacy 实验可识别性；新格式显式带版本，旧参考不能静默升级为新证书。优先扩展现有 `GateIdentityConfig` 或等价轻量结构，不建设通用配置平台。

冻结的配置在 export 前完成 enrollment，记录可信 checkpoint 来源及完整 SHA、source pin/content、norm policy/digest、变量与坐标内容身份、raw input、执行 dtype、rollout、容差。observed 值用于核验，不能在验证时自动复制为 expected 再宣布通过。

**补丁 B：真正接通执行链。**

生产入口读取同一配置，并区分 expected identity、reference manifest、实际加载/执行 identity。缺字段、字段不等、同一身份的两处记录冲突，都使总判定失败。已经算好的 checkpoint 内容 SHA 可复用；无需为形式上的“独立”重复扫描多 GB 文件。

用当前 raw 通过 bridge 自己的 normalizer 计算 input，比较 exporter 保存的 normalized input，并绑定其内容身份。随后从当前输入比较所有注册 `(interval, steps)` 输出。当前默认是 6h 的 1 步与 4 步；不能用健康张量检查替代四步数值 parity。

参照集合不可为空；变量和坐标匹配必须比较实际顺序/内容，不只比数量与 shape。输出 tensor 的 exact shape、dtype、finite、内容 hash 与 rollout 标签一并检查。

身份比较与 raw-input 检查尽量在大模型加载和前向之前完成。CPU fixture 可替换昂贵 backbone/加载器，不能 mock 掉待测的身份比较、raw 绑定、多步比较和总判定。

**补丁 C：目录、启动和持久化。**

exporter/gate 都接受同一冻结 `--config`，gate 接受显式 `--reference-dir`。自动选择若保留，只能在完整身份匹配且唯一时成功。认证模式拒绝模糊选择和 legacy fallback。

目录可用短标签便于阅读，内部认证必须核对完整身份。将 raw/norm/config 身份纳入隔离；同身份重跑也使用唯一 run ID。先写临时产物并校验，再发布完成标记/manifest，防止半写产物或旧 PASS 被消费。不扩展成通用 artifact 服务。

区分源码根与资产根：导入固定 worker snapshot，checkpoint/NPY 由 config 显式指向 AFS 资产。记录实际导入的模块路径及内容 hash；snapshot 无 `.git` 时使用提交侧生成、worker 复核的 source manifest。

### 4.2 验收应覆盖真实总判定

先构造一个 NPY、NPZ、manifest、输入和多步输出全部一致的合成正例，再只破坏一个条件。否则“反例失败”可能只是被无关错误提前挡住。

至少覆盖：正确整包、错误/缺失 checkpoint SHA、norm policy/digest、source identity、变量顺序、同形状不同坐标、raw 与 normalized input 不一致、四步有限但数值错误、缺失多步、空 rollout 集合、hash 冲突、同 ps4 多目录、导出中断、报告/结果持久化失败。

这些测试调用实际 `run_s0_gate`，关键结果再经 CLI/main 核对非零退出码和最终 artifact。合成通过只能标 synthetic，不能生成能放行真实训练的证书。

保留已有 helper 测试；修复 policy 测试的混杂：所有 tensor 相同、只切换 policy 也要改变身份；另构造 official policy 但对象实际存有非零 diff_mean 的用例验证运行时语义。

### 4.3 Stage 1B/1C 的最小范围

原生目标为 `[B,H,V,Lat,Lon]`，候选额外保留 K 维：

```text
L(Y,F) = sum(q * ((Y-F)/s)^2) / sum(q)
e = Y-Fs; u = Fa-Fs
gain(a) = L(Y,Fs)-L(Y,Fa)
        = (2*sum(q*e*u/s^2)-sum(q*u^2/s^2)) / sum(q)
```

q 只在目标 H/V/Lat/Lon 范围内归一化一次；主目标 24h 与 6h/72h 诊断分别注册。验证实浮点、finite、非负 q、正 scale、合法 broadcast 和每个实际目标的正分母。保留真值精度后再形成 FP64 误差/累加；不能先损失精度再转换。独立手算的非均匀权重反例必须与端点 loss 差一致。

数据只做本轮实际切片准入：真实 UTC time 索引、t-12/t-6/t 历史、t+6/t+24/t+72 端点、通道/坐标、finite 和角色隔离。不能把下载 marker、Zarr shape 或日历表视为内容完成证据。读取实际所需 chunk，不改造整套下载器。

registry 需非空、唯一 ID、显式 no-edit、实浮点有限系数、固定幅度/support/hold；校验器必须在首次消费前调用。B03 caller 系数梯度、未使用的 B04 Gram、共享模型并发继续暂缓，不拖住固定系数小库路线。

## 5. CCI 任务、运行入口与资源用法

### 5.1 任务设计

| 作业 | 工作 | GPU 用法 | 建议 worker 上限 |
| --- | --- | --- | --- |
| J0 | 专用环境准备、官方 import、xformers 实际算子、资产可读性 | 4 卡规格中按需使用；无需强行四卡推理 | 60 分钟 |
| J1 | 独立官方 export、bridge S0、推理和训练步 profile | 独立进程；参考发布后再消费，不能共享可变模型 | 75 分钟 |
| J2 | Fs → bank → 8 起报调试 → 冻结开发缓存 | Fs 有依赖；bank 四进程；缓存按起报四分片 | 根据剩余墙钟设置，最多约 4 小时 |
| J3（条件性） | 已冻结任务的额外缓存分片或独立复验 | 只有前驱通过且截止前有时间才提交 | 由剩余时间决定 |

J2 可由一个小型 stage runner 顺序调度，减少重复排队；每个阶段仍读前驱证据，失败则不启动依赖阶段。不要在 worker 中等着控制节点继续修改脚本。若入口未写完，先结束 J1 并提交完整的新快照。

不要为利用四卡而引入 DDP、线程共享 hook 或 activation-checkpoint replay。免费资源允许重复验证，但不能替代方法成本账本；分别报告实际 GPU 进程时间、节点占用、排队和磁盘。

### 5.2 已存在且验证过的提交器

提交器为 `plans/plans_v2_0921/cci/submit_job.py`，资源为 share-space/share-cluster、1 worker、4×H100、32 CPU/512 GiB。工具已有源码快照、job ID 和日志记录。

执行时先建立唯一研究目录，并把其绝对路径赋给 `ED_RUN`；下例只有在相应 argv/config 文件已存在且对应 CLI 已实现后才能使用：

```bash
cd /mnt/afs/260010168/EarthDelta
python3 plans/plans_v2_0921/cci/submit_job.py \
  --label r3-env --timeout-seconds 3600 --argv-file "$ED_RUN/env.argv.json"
python3 plans/plans_v2_0921/cci/submit_job.py \
  --label r3-s0 --timeout-seconds 4500 --argv-file "$ED_RUN/s0.argv.json"
python3 plans/plans_v2_0921/cci/submit_job.py \
  --label r3-pilot --timeout-seconds 14400 --argv-file "$ED_RUN/pilot.argv.json"
```

这是分阶段命令清单，不能一次无条件连着提交；每次的实际 timeout 还须截断到 T+9:00 前可用窗口。可先对准备好的 argv 使用 `--dry-run` 核查快照与路径。默认不传 argv 只会跑硬件探针，不能用其替代真实实验。

本计划要求新增/扩展的 `--config`、`--reference-dir` 目前尚未实现，`env.argv.json`/`s0.argv.json`/`pilot.argv.json` 也没有在本次分析中创建。执行者需在相应时段实现并 smoke-test，再形成真实命令记录，不能把本段当作已经跑通的科学入口。

作业查询沿用已有方式：

```bash
SCO_LAUNCHED_BY=launcher /mnt/afs/260010168/bin/sco acp jobs describe JOB_ID \
  --workspace-name=share-space -o json
```

读取共享输出中的 `worker.log`、`job_result.json`、科学 manifest 和平台状态。进程退出 0、平台 SUCCEEDED、S0 PASS、研究有效是四个不同判断。提交状态未知先按唯一作业名查询，避免重复创建。

### 5.3 环境与 S0 的真实验收

专用环境验证 Torch/torchvision/lightning/timm/xformers 兼容性以及 xformers CUDA 前向/反向。记录完整版本；不伪造 Lightning，不用 SDPA 冒充官方 xformers，不盲目升级共享 Torch。已有环境可复用，但要检查内容身份。

真实 S0 首先冻结 ps4、official_zero_diff_mean、FP32 和 6h 的 1/4 步。默认沿用已有数值约束，并记录比较空间；当前官方 `forward_validation` 返回 normalized state。若容差无法通过，先检查归一化、dtype、TF32、实现和重复性，不看到差异后任意放宽。

本轮若评估 72h guard，在首次评分前补齐同配置 12 步推进验证。1/4 步通过只说明这两种配置，不能自动为 12 步或其它 checkpoint 认证。

所有 S0/profile/debug 样本标 exposed。真实训练步测固定系数下 LoRA 更新、冻结 backbone、不残留 hook、显存与速度；零 B 初始化时第一步 A 梯度可以为零，应在 B 更新后验证 A/B 的实际梯度路径。

## 6. 后半程的小型真实实验

### 6.1 冻结的首轮设计

| 项目 | 建议 |
| --- | --- |
| Backbone | 与通过 S0 相同的 ps4/官方 policy |
| Fs | bank_fit 侧训练/选择的静态 adapter，冻结后验证合并或常开等价性 |
| Bank | 默认 K=4、rank=4、blocks 18–23；profile 证明不合适时，可在查看开发结果前改为 K=2 并登记 |
| 候选 | no-edit + 每个专家 singleton；max_active=1；实际系数 a0=0.25，rho=0.25 |
| 作用窗 | 4×6h；窗外从被编辑状态继续 Fs |
| 目标 | 24h native loss；6h 诊断；72h 后续损害 guard |
| 数据 | bank_fit 与 policy_dev 时间隔离；history/target purge；confirm 本窗口不读取 |
| 方法 | Fs/no-edit、fit-only 最佳固定候选、regime、ridge direct-gain、完整 oracle |
| 暂缓 | dual-head、E3 输出纠错、连续系数、memory/JEPA、确认集 |

Fs 不能因跑了少量 updates 就被称为“强静态基线”。先记录训练侧资格与局限。若 Fs 没训好，本轮的动态收益只能作为弱参考下的开发线索；不得偷换成论文级结论。no-edit 和 hold 后 continuation 必须是 Fs，不能悄悄退回 F0。

### 6.2 训练量与缓存量由 profile 决定

先做每条训练路径 16–32 updates 的实跑，验证有限 loss、梯度、非零响应与可恢复，再按训练侧证据决定额外 updates。以前计划的 500 updates 是有界起始上限，不是本窗口必须达成或足够收敛的事实。若截止前训练不足，标资格不足，不判策略无价值。

先对 8 个 exposed 起报跑全部候选，用于查错。之后在结果揭示前，依据吞吐、实际时间角色和有效过程数选定 32、64 或 128 个 policy_dev 起报；128 是条件性上限目标，不是必保样本量。采样顺序和扩展规则预先记录，不能按收益挑时段或专家。

可用下面的保守估算判断是否能进入缓存：

```text
T_cache ≈ ceil(N / 4) × (K + 1) × 12 × t_step + I/O与评分开销
```

其中 t_step 来自真实 profile，公式按每个候选完整跑 72h、batch=1、四进程估计；不能把单卡速度直接当四卡实测吞吐。缓存可在同次 rollout 保存 6/24/72h，不需要重复三次从头推理。

K=4、69×128×256、FP32、三个端点、N=128 时，仅候选预测约 17.36 GB，真值约 3.47 GB，另加元数据与训练产物。这是维度算术，不是本次磁盘实测。正式提交前核验可用空间。

银行资格看训练来源、非零响应、有限稳定性、专家差异与恢复，不按开发赢家事后删除专家。某个候选失败要记录并按冻结规则处理；不能删除失败行、填零或改变机会分母来生成好看的 oracle。

### 6.3 第一张开发决策表

完整缓存后，先持久化合法策略动作，再由评分程序读取 outcomes。特征只使用登记的当前/历史输入；标准化、聚类、静态候选选择和 ridge 调参在各外折训练侧完成。采用时间顺序与 purge 的开发评估；bank 不得见被评分的 policy_dev。

至少包含：

```text
method | n_issues | n_processes | attempted/complete
native_loss_24h | gain_vs_Fs | gain_vs_frozen_static
paired_uncertainty | guard_72h | harmful_edit_rate | no_edit_rate
offline_cost | online_cost | failure/fallback_cost
```

oracle 用未来真值，仅作上限。best-static 在训练折选出后对评估折固定。小样本过程数不足时标描述性 DEV；不能把格点、候选或相邻起报当独立重复来制造狭窄 CI。

本窗口默认一个固定训练 seed 跑通，有限 cheap HPO；不承诺同时完成三 seed 稳定性分析。只有缓存完整、样本/过程足够且时间有余，才增加固定 seed 或不确定性分析。任何开发筛查都不等同于 confirm。

## 7. 10 小时结束时如何决定下一轮

| 观测结果 | 下一步 |
| --- | --- |
| S0 未通过 | 聚焦修复具体差异，暂停依赖的正式训练/效果结论 |
| S0 通过，但训练/数据资格不足 | 补合格 Fs/bank 或实际切片；不对动态方法下否定结论 |
| 合格库完整 oracle 无有用余量，且不确定性足够小 | 停止当前库或转静态；结论只覆盖当前动作集 |
| 固定候选解释大部分收益 | 优先静态方案，避免增加不必要动态复杂度 |
| oracle 有余量，regime/direct 有合法增益线索 | 下一窗口先提高证据强度，再比较双头的额外价值 |
| oracle 有余量，cheap 无明确信号 | 检查表示、有效样本和响应可预测性；仅有诊断依据时给一次有上限加强挑战 |
| 置信区间宽或有效过程太少 | INCONCLUSIVE；按冻结开发方案扩样，不制造科学 STOP/PASS |

双头与输出纠错不列为这 10 小时的默认任务。双头要证明相对公平 direct 的额外收益/标签/成本价值；E3 要在已有合法动态策略后比较。正式确认应在所有方法、阈值与分析程序冻结后另开窗口一次执行。

## 8. 状态与交接

使用一个唯一研究运行目录和 CCI 已有作业目录，避免生成多套状态系统。至少交付：

- 本次源码差异、相关内容 hash、实际测试命令/退出码/跳过原因。
- 冻结 config、source/data/registry 身份、官方参考、S0 数值与输入链证据。
- 实际 job ID、排队/运行/结束状态、环境与失败记录、计算/存储成本。
- 若已运行：Fs/bank 训练记录、资格判断、完整候选覆盖和开发决策表。
- 一份 `decision.json` 与 `HANDOFF.md`，分别记录 code/run/research 状态、缺项和下一条真实可运行命令。

当前状态应写为：`code_status=STAGE1A_PARTIAL`、`run_status=REAL_S0_NOT_RUN`、`research_verdict=NOT_EVALUATED`。本次分析只新增了摘要互操作的 CPU 证据，不改写为整套 CPU 回归或 S0 成绩。

执行期间不为已经授权的常规 CCI 提交、环境准备或测试反复请求确认；阶段推进取决于证据。用户此前的暂停仍然有效，本文件本身不启动任务。原有未跟踪目录、后台拉取和原始复查文件保留；后续若提交 Git，只显式列入本次应提交文件，不上传 checkpoint、天气缓存或凭证。
