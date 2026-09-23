# EarthDelta Stage 1A 独立复查：B01 / B02 / B10

## 1. 总体结论

**这次修复有实质进展，但不能把 B01、B02、B10 三条整体标为 `CLOSED`。最关键的剩余问题不需要等 GPU 才能发现：版本检查在比较自身，上游 manifest 的身份仍未真正参与验收，4 步参照仍未进行跨实现数值比较。**

本轮仅依据 `STAGE1_PATCH_PACKET.md`。我没有访问 GitHub、没有执行仓库测试或 GPU 推理；只对 packet 中的 diff 和两个测试文件进行了静态审查，并解析测试文本核对了函数数量。

| Finding | 独立判断 | 一句话理由 |
|---|---|---|
| **B01** | **同意关闭——限本次静态代码修复层面** | exporter 已改为独立构造零均值差分 transforms，并调用官方 `GlobalForecastIterativeModule.forward_validation`；bridge 的 `policy` 确实进入 digest。真实数值一致性仍待 Stage 2。 |
| **B02** | **不同意关闭** | 三个“缺 expected 就通过”的分支已修复，但执行入口仍然自比版本，上游身份字段只是记录，4 步文件只检查可读、形状和有限性，没有验证其预测值正确。 |
| **B10** | **部分同意** | 默认 ps4 和自动输出目录中的 patch-size 区分已落实，但没有共用并强制执行完整冻结配置；gate 仍按名称猜选目录，输入身份也未进入目录隔离。 |

因此建议当前状态记为：

```text
B01: CODE_FIX_ACCEPTED / GPU_PARITY_NOT_RUN
B02: OPEN
B10: PARTIALLY_CLOSED
Stage 1A overall: NOT_CLOSED
```

下面具体说明哪些确实修好了、哪些仍只是“看起来接入了”。

---

## 2. B01 / B02 / B10 逐条详细分析

### 2.1 B01：归一化独立路径已经对症修复

#### A. exporter 不再使用受审的 `NormalizationContract` 构造参考预报

diff 明确删除了：

```python
from earthdelta.bridge import (
    NormalizationContract,
    DEFAULT_VARIABLES,
    _compute_file_sha256,
)
```

替换为官方网络、官方迭代模块，以及仅用于文件身份计算的 hash 工具。借用 `_compute_file_sha256` **不等于继续共享归一化实现**，不应据此否定这部分修复。

`build_official_transforms` 独立读取 NPZ，构造：

```python
diff_transforms[interval] = transforms.Normalize(
    np.zeros_like(normalize_diff_std),
    normalize_diff_std,
)
```

这里的零均值确实来自 exporter 自己的代码，而不是继承 bridge 的 policy 或 `denormalize_diff`。

#### B. exporter 确实调用了官方迭代方法，而不再手写相同循环

新增路径是：

```python
module = GlobalForecastIterativeModule(net)
module.set_transforms(inp_transform, diff_transforms)
...
out = module.forward_validation(
    x_norm, OFFICIAL_VARIABLES, interval, steps
)
```

原有 `run_official_inference` 手写循环被删除。这正对原 B01 的核心问题。

需要区分的是：**导入指定目录里的官方类，与验证该目录实际处于声明的 commit，是两件事。**后者仍有身份绑定缺口，归入下面的 B02，而不是说归一化独立修复没有发生。

#### C. policy 会改变 digest，而且也改变执行行为

bridge 在 `denormalize_diff` 中显式处理：

```python
if self.policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
    mean = torch.zeros_like(std)
```

同时，digest 的序列化输入新增：

```python
parts.append(f"policy={self.policy}".encode())
```

所以不是“只增加一个不影响执行的字段”，也不是“执行变了但身份标签没变”。**policy 的语义变化与身份变化都已接入。**

**B01 的结论：**认可这条代码修复；不因尚未运行 GPU 就否定已展示的修复。但是，不能据此宣布官方与本地完整预报已数值一致。

---

### 2.2 B02：局部比较修好了，但完整身份链没有闭合

#### A. `verify_ckpt_sha256` 已从“长度检查”变为等值比较

新代码确实执行：

```python
computed_sha256.lower() == expected_sha256.lower()
```

而 `expected_sha256 is None` 时明确返回：

```python
passed = False
reason = "IDENTITY_NOT_BOUND: ..."
```

这两个改动是有效的。

但准确描述应是：

- `load_result is None` 时，调用文件 hash 工具计算；
- `load_result` 存在时，使用其 `checkpoint_sha256` 字段。

**本 packet 没有展示 `load_stormer_checkpoint_detailed` 和 hash 工具的完整实现，所以不能仅凭这一段证明生产路径中的该字段一定绑定本次加载的文件字节。**如果加载器已经可信地完成这一步，复用结果完全可以，不需要为了审计再重复扫描大文件；这不是我新增的一条确定 bug，只是证据边界。

更重要的是：**本地 checkpoint 与 CLI expected 相等，并不意味着所选上游参照也来自同一个 checkpoint。后者仍未检查。**

#### B. `verify_normalization_parity` 确实比较 digest，但只闭合了本地一侧

它现在要求：

```python
norm_from_npy.digest == expected_digest
and
norm_from_npy.digest == norm_from_npz.digest
```

缺 expected 时也明确失败。这比原来的自一致性检查强。

但 `verify_upstream_parity` 只是把上游 manifest 的：

```python
checkpoint_sha256
normalization_policy
normalization_digest
```

放进结果字典，没有把这些字段与冻结 expected 或实际 bridge 做相等判断。**缺字段时 `.get(...)` 得到 `None`，同样没有在此阻断。**

因此现在仍可能出现：

> 本地 checkpoint、normalization 都通过自己的 expected 检查，但用于比较的上游参照声明了另一套身份，gate 仍只看数值是否接近。

这仍是原 B02 要解决的问题。

#### C. `x_raw` 已不再完全是摆设，但尚未绑定“实际参与 parity 的输入”

新增 `verify_raw_input_binding` 确实计算当前 `x_raw` 的 hash，并与参照目录中的记录比较。这个进展应当认可。

问题在于，数值 parity 仍使用：

```python
upstream_input_norm = torch.load(... / "input_norm.pt")
...
bridge.forward_validation(input_norm, ...)
```

没有展示以下任一验证：

```text
input_norm.pt == 本地 normalization(x_raw)
```

或：

```text
input_norm.pt 的内容 hash
    == 已绑定 raw input、normalization 身份的 manifest 中的预期 hash
```

因此，**raw hash 检查的是旁边的一份身份声明；真正喂给 parity 的 `input_norm.pt`，尚未与这个 raw input 连起来。**

另外，`raw_input_hash.txt` 存在时，manifest 中的 `raw_input_hash` 被忽略。选择一个权威来源本身可以是合法设计，但当前既未明确统一权威，也未校验两个记录冲突，更未绑定输入 tensor 和输出 tensor。

#### D. `check_version_match` 形式上被调用，实质上仍没有验证外部期望

新增代码是：

```python
check_version_match(bridge.version, bridge.version)
```

这是**当前身份与自身比较**，没有比较：

- 冻结配置与本地模型；
- 上游参照与本地模型；
- 编辑计划／缓存与执行模型。

更严重的是，即使这里抛出 `ValueError`，代码也只是记录：

```python
result["version_match"] = False
```

没有重新抛出，也没有把该结果加入 `gate_criteria`。因此，它既不是有效的身份比对，也不是有效的总判定阻断项。

**“helper 已接入真正入口”这句话，按调用位置理解是对的；按验收功能理解是错的。**

#### E. `6h_4step` 会被加载，但不会被做跨实现数值比较

现在有两层改进：

1. `verify_upstream_parity` 要求 1 步、4 步文件都存在；
2. `verify_multistep_reference` 会加载它们，检查 shape 和有限性。

所以不能继续说“4 步文件完全没加载”。

但是，实际数值 parity 仍然只运行：

```python
bridge.forward_validation(..., interval=6, steps=1)
```

**没有把本地 4 步输出与 `official_output_6h_4step.pt` 比较。**

一个直接的静态反例是：

> 在其他检查本来通过的前提下，将 4 步参照替换为同 shape、全有限的任意 tensor。当前 4 步参照检查仍通过；1 步数值比较也不受影响。

这是根据代码推导的反例，**本轮未实际执行**。

**B02 的结论：不同意关闭。**已关闭的是若干 helper 的 missing-expected 分支，不是完整的参照身份与多步正确性验收。

---

### 2.3 B10：默认模型修正了，完整配置和产物隔离仍未落实

#### 已完成的部分

exporter 与 gate 的 CLI 默认均改为 `ps4`；gate 也根据 `patch_size` 选择对应 checkpoint，而不再固定加载 ps2。

exporter 自动目录包含：

```text
upstream_reference_ps{patch_size}_{checkpoint_sha前8位}_zd
```

因此，在自动路径下，ps2 与 ps4 的名称确实不同。

#### 未完成的部分

**第一，`GateIdentityConfig` 没有成为生产合同。**

packet 自己已承认它只在测试中实例化。实际 exporter 自算 tag，gate 则按 patch-size glob 找目录，不是消费同一冻结配置。

**第二，目录选择不是精确身份选择。**

```python
matches = list(base_dir.glob(pattern))
return sorted(matches)[-1]
```

这里既没有按预期 checkpoint hash 筛选，也没有按 normalization、输入或源码身份筛选。并且 `sorted(matches)` 是路径排序，**不是按修改时间选择“最近”的目录**。

**第三，输入身份没有进入自动目录名。**

相同 checkpoint、patch-size、policy 下，即使换了 raw input 或 normalization 数值，仍落到相同命名空间。显式 `--output-dir` 还能绕过自动命名。可见写入使用 `torch.save` 和 `open(..., "w")`，本 packet 没展示相应的身份冲突拒绝机制，因此不能声称已经保证不会覆盖或混用。

**第四，gate 的报告目录仍默认是公共 `artifacts/s0_gate`。**

参照目录的 patch-size 隔离，不等于整个验收产物都隔离。

**B10 的结论：部分同意。**“主 checkpoint 默认选错”已修复；“共用显式冻结身份、精确选择参照、按输入身份隔离产物”尚未完成。

---

## 3. fail-open 模式排查结果

### 3.1 三个已点名分支：未发现原 fallback 仍被保留

在本次 diff 中：

- checkpoint expected 缺失；
- normalization expected 缺失；
- raw-input 参照 hash 缺失；

都已改成 `passed=False`，并保留 `IDENTITY_NOT_BOUND` 原因。**这部分修复有效，不应否认。**

### 3.2 仍存在的同类或等价遗漏

| 遗漏 | 具体行为 | 当前影响与最小处理 |
|---|---|---|
| **版本自比较且不参与总判定** | 比较 `bridge.version` 与自身；失败也只是独立日志字段。 | 当前生产路径中的确定缺口。改为冻结 expected 与 observed 比较，并纳入 gate 或直接阻断。 |
| **上游 manifest 缺失／错误身份仍不阻断** | checkpoint、normalization 字段只是 `.get` 后展示；source pin、变量与真实坐标也未见验收。 | 当前生产路径中的确定缺口。参照消费前验证一个完整身份对象，缺必需字段就失败。 |
| **4 步“存在”替代了4 步“正确”** | 同 shape、有限的错误 4 步 tensor 可以通过。 | 当前生产路径中的确定缺口。对所有注册步数运行实际 bridge，并比较对应参照。 |
| **raw 身份未贯通到 normalized tensor 和输出** | raw hash 匹配，并不能证明 `input_norm.pt` 和预报文件来自它。 | 当前绑定链不完整。绑定文件内容，并验证本地 raw-to-normalized 转换。 |
| **空 `required_steps` 会通过** | 已找到目录时，传 `required_steps=()`，循环不执行，两个错误列表都为空，最终 `passed=True`。 | helper 的边界缺口；当前 `run_s0_gate` 显式传 `(1,4)`，所以默认生产路径不会触发。应拒绝空注册集，但严重度低于前四项。 |

空步骤集合的问题可以直接从 `verify_multistep_reference` 的循环和最终布尔表达式判断，不依赖 GPU。

### 3.3 “挑一个目录”是否算 fail-open？

**单独按目录排序选择，不一定导致错误 PASS：后续严格身份检查仍然可以拒绝。**

但当前恰好缺少这种完整检查，所以它已不只是 DRY 或便利性问题：

- 同 patch-size 的多个不同身份目录，不报歧义；
- 自动选字典序最后一个；
- 找不到时还回退无 patch-size 限定的 legacy 目录；
- 选中目录的实际身份没有与冻结期望完整比较。

因此，应将它视为 **B02/B10 尚未闭合的实际路径**，而不是仅记录“以后可能漂移”。

最小修复不需要建设复杂配置框架：**显式传入准确 reference directory，或根据完整冻结身份只接受唯一匹配，并在本次 gate 中只解析一次、供所有检查共同使用。**

### 3.4 `GateIdentityConfig` 没进生产，是不是单独的漏洞？

**“定义了 dataclass 却没使用”本身不自动构成 fail-open。**两个独立实现若都严格验证同一合同，也可以正确。

这里的问题是生产代码没有等价地执行其完整合同。该类中声明的变量、grid、rollout、commit 等信息，不能因为出现在 dataclass 中就视为验收已覆盖。当前 tag 也只包含 patch-size、SHA 前缀和 policy，不包含所有这些字段。

---

## 4. 测试质量抽查结果

### 4.1 新增四个 fail-closed 测试：基本测中了目标

| 测试 | 实际构造的场景 | 判断 |
|---|---|---|
| `test_gate_fails_closed_ckpt_sha256_none` | 文件真实存在、可以计算 hash，但 expected 为 `None`。 | **有效。**不是靠“文件不存在”提前失败；还检查 hash 与 `IDENTITY_NOT_BOUND`。 |
| `test_gate_fails_closed_normalization_digest_none` | NPZ 存在，可以构造 normalizer，但 expected digest 为 `None`。 | **有效。**建议额外断言 `digests_match_npz is True`，更明确排除其他失败原因。 |
| `test_gate_fails_closed_raw_input_no_upstream_dir` | artifacts 目录存在，但没有参照子目录。 | **有效。**确实覆盖无参照可绑定分支。 |
| `test_gate_fails_closed_raw_input_no_hash_in_reference` | 参照目录和 manifest 存在，但既没有 hash 文件，也没有 manifest hash 字段。 | **有效。**不是换名字测试目录不存在。 |

这些测试确实比只断言 `passed=False` 更好，因为同时检查了指定原因。

**但是它们证明的是四个 helper 场景，不是整个 CLI／`run_s0_gate` 的集成行为。**

### 4.2 新测试文件的整体说明，明显强于实际覆盖

`test_s0_gate_identity.py` 开头写：

> 验证 REAL s0_gate orchestration，而不是手工 dict。

实际全文只调用四个验证 helper 和 `GateIdentityConfig`；**没有调用 `run_s0_gate`、`main`，也没有调用 `verify_upstream_parity`。**所以它不能证明 expected 参数被正确传递、版本失败进入总判定、上游 manifest 身份被拒绝，或 4 步数值被比较。

同样，文件头声称覆盖“错误 variable/grid identity”，但所附测试没有实际修改变量顺序或坐标并验证拒绝的用例。

### 4.3 具体偏弱或证明对象不准确的测试

| 测试／fixture | 问题 | 应怎样补强 |
|---|---|---|
| **`test_policy_changes_digest`** | official 与 legacy 不仅 policy 不同，实际 `diff_mean` 也不同。即使删除 policy 的 hash 字段，测试仍可能通过。 | 保持所有 tensor 完全相同，仅替换 policy，再要求 digest 不同。当前代码确实包含 policy，但测试没有单独保护这一性质。 |
| **`test_official_policy_ignores_nonzero_diff_mean_in_denormalize`** | 通过 `from_npz_dir(...official...)` 构造时，mean 已被清零，所以没有验证“对象里仍有非零 mean 时，denormalizer 会忽略”。 | 直接构造 `policy=official` 且 `diff_mean` 非零的对象，验证结果仍是 `diff_norm * std`。 |
| **`test_gate_accepts_correct_normalization_digest`** | expected 直接取 `norm.digest`，然后检查同一 bridge normalizer 的 NPZ 路径。 | 作为 helper 正向用例合理，但不能证明 exporter 的独立 digest 与 bridge 一致；需跨实现测试。 |
| **`test_gate_accepts_matching_raw_input`** | 只更新 `raw_input_hash.txt`，没有同步 manifest，也没有绑定 `input_norm.pt`。 | 只能证明 hash 文件优先分支；应补完整一致资产的正向用例，再分别破坏各绑定。 |
| **`test_gate_accepts_all_multistep_present`** | 写入两个正确 shape 的随机 tensor 就接受。 | 对“文件健康检查”合理；不能当作“多步 parity 已验证”。需另测错误但有限的 4 步值必须拒绝。 |
| **`test_gate_identity_config_reference_dir_namespaced`** | 只比较 dataclass 生成的两个字符串，没有实际调用 exporter/gate 路径选择；还同时改变 patch-size 和 SHA。 | 测试真实生产解析函数，并分别改变 patch-size、完整身份、输入，验证选择唯一且不覆盖。 |

还有两个 fixture 细节值得纠正：

- `mock_upstream_reference` 自称 valid outputs，但初始 shape 是 `16×32`，不满足实际检查要求的 `128×256`。
- NPY 的差分标准差是随机数，而 NPZ 的差分标准差固定为 `0.5`。它们并不是可以直接通过完整 normalization parity 的一致资产。

这些不使单元测试全部无效，但说明**还没有一个“先证明完整合法输入通过，再只破坏一个字段验证失败”的端到端基础 fixture**。

### 4.4 测试数量核对

静态解析结果与两个文件标题一致：

```text
test_normalization_policy.py：369 行，10 个 test_* 定义
test_s0_gate_identity.py：592 行，19 个 test_* 定义
                         = 原 15 个 + 新增 4 个
```

因此：

- `68 → 72` 与增加 4 个测试在数量上相容；
- `19 + 13 = 32` 也相容。

**没有从数量上发现矛盾。**但 13 个既有 fail-closed 测试全文不在 packet 中，不能独立核实它们全部断言的强度；报告的 pytest 结果仍属于 packet 提供的运行记录，而非本轮复跑。

---

## 5. 一致性检查结果

发现以下具体不一致或表述过强之处。

### A. “接入版本检查”与实际自比较不一致

正文说 `check_version_match` 接入执行入口；位置确实如此，但传入的是同一个运行态身份两次，并且结果不影响总 gate。**不能把这描述成身份绑定已生效。**

### B. “选最近目录”与实际字典序排序不一致

`return sorted(matches)[-1]` 不读取时间信息；注释中的 “most recent” 不准确。

### C. “两端各自计算等价 identity_tag”不准确

正文如此描述，但 diff 中：

- exporter 确实计算完整短 tag；
- gate 只是按 `ps{patch_size}_*` glob 查找，不计算或匹配对应的 checkpoint SHA tag。

因此，不只是“两份等价代码将来可能漂移”，而是**当前两端的选择逻辑已不等价**。

### D. “B13/B14 尚未开始”需要修正范围表述

diff 已经：

- 在 `run_s0_gate` 外层异常中重置 `s0_gate_pass=False`；
- 修改了报告中缺少数值字段时的格式化行为。

因此，若“尚未开始”指专门的 Stage 1C 任务未启动，可以成立；但若指相关代码完全未动，就与 diff 不符。应标为**本批已附带修改、仍待专项回归**，不能简单写“未开始”，也不能反过来直接称 CLOSED。

### E. SHA 测试注释与实际长度不符

```python
"abc123def456" * 5 + "ab"
```

实际是 **62 个字符**，不是注释中的 64 个。

fixture 中：

```python
"abc123" * 11
```

是 **66 个字符**；注释说会 truncate，但所附代码没有截断。

因此，`test_gate_identity_config_sha256_validation` 实际验证的是字符串匹配，不是 SHA-256 格式有效性。

### F. 另有跨端 normalization digest 一致性风险，但不能夸大为已确认误放行

exporter 的 `compute_normalization_digest` 直接对 NPZ 的原始 NumPy dtype 取字节；gate 构造 normalizer 时显式转为 float32。**如果原始 dtype 不同，相同数值语义未必得到相同 digest。**此外，exporter 按 interval 交错写入 mean/std；bridge 的完整、未修改 digest 后半段没有在 diff 中展示，无法仅据此完整复核其序列化顺序。

这一点目前应记录为：

> **跨端 digest 兼容性未被证明，可能造成错误拒绝；不是已经证明的 fail-open。**

需要补一个不运行天气模型的 CPU 测试：同一组规范化常数，由 exporter 独立序列化和 bridge 序列化，得到同一身份；保留两套数值 transforms 的独立性即可，不必重新共享受审 normalizer。

---

## 6. 仍然存在的已知限制

我认可 packet 关于以下事项尚未完成的声明：

- 没有真实 GPU/xformers S0；
- 没有 Fs/bank 训练；
- 没有完整候选缓存或 legal-policy 决策实验；
- B05/B06、数据入口及其他阶段任务不能据本批测试宣布完成。

还应补充四项边界：

**第一，真正的官方与本地数值一致性无法仅凭静态 diff 判断，需要 Stage 2 真实 GPU S0 结果。**但必须先把 4 步数值比较和完整身份验证实现好，否则运行当前 gate 仍测不到相应性质。

**第二，实际加载的上游源码是否匹配声明 commit 尚未证明。**把常量 `OFFICIAL_STORMER_COMMIT` 写入 manifest，不等于检查了执行源码。

**第三，packet 是完整 diff，不是每个受影响文件的完整修改后源码。**对 `_compute_file_sha256`、`load_stormer_checkpoint_detailed` 和完整 digest 序列化等未展示实现，需要相应函数全文才能继续作确定判断；不能从旧对话记忆补齐。

**第四，B13/B14 的范围说明应更新为“本批有附带修复，尚未完成专项验证”。**不宜继续保持“完全未触碰”的表述。

在继续 Stage 2 前，最小补丁应聚焦三件事：**绑定一个真实的冻结参照身份并纳入总判定；比较全部注册步数的实际输出；精确、唯一地选择并隔离参照目录。**无需借此扩展到并发框架、研究模型或无关清理。

---

## 7. 一句话总体建议

**B01 可接受为静态修复完成，但 B02 和 B10 不能标为 CLOSED；应先修复版本自比较、上游身份未绑定、4 步仅检查文件以及模糊选目录这四个具体缺口，再进入能真正验证这些性质的 Stage 2 GPU S0，其他独立的 Stage 1B/1C 工作可以并行推进。**
