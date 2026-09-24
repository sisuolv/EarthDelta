# EarthDelta Stage 1A 独立复查：B01 / B02 / B10

## 1. 总体结论

**这批修复有实质进展，但 B01、B02、B10 还不能整体标为 `CLOSED`。主要问题不是缺少 GPU 结果，而是 diff 中仍有可以静态确认的身份绑定缺口。**

| Finding | 独立判断 | 一句话理由 |
|---|---|---|
| **B01** | **同意关闭——限 Stage 1A 的源码修复** | exporter 已独立构造零均值 transforms，并调用官方 `forward_validation`；bridge 的 `policy` 确实进入了 digest，但这不等于真实 GPU 数值一致性已经通过。 |
| **B02** | **不同意关闭** | 局部 hash 比较已修复，但版本检查仍是“自己与自己比较”，上游 manifest 身份没有强制匹配，四步输出也仍未与 bridge 做数值比较。 |
| **B10** | **部分同意** | 两端默认 ps4、默认导出目录按 patch/checkpoint 区分已经实现；但没有真正共用冻结配置，gate 仍按目录名排序猜选参照，结果目录也未隔离。 |

其中最明确的一处问题是：

```python
check_version_match(bridge.version, bridge.version)
```

这只是调用了函数，**没有把独立的“预期身份”与“实际身份”进行比较**；即使该调用抛出 `ValueError`，代码也只记录 `version_match=False`，没有立即退出或把它纳入必需 gate。

以下判断仅依据本次 packet。未访问 GitHub、未运行仓库测试；我只对附件做了静态文本检查和测试数量核对。

---

## 2. B01 / B02 / B10 逐条详细分析

### 2.1 B01：核心共享归一化问题，源码层面修对了

**第一，exporter 不再使用受审 `NormalizationContract` 构造参考预测。**

diff 删除了 exporter 对 `NormalizationContract` 和 `DEFAULT_VARIABLES` 的导入，新增独立的变量列表，以及官方 `GlobalForecastIterativeModule` 的导入。虽然仍从 EarthDelta 导入文件 hash 工具，但这不等于继续共享受审的数值归一化路径。

**第二，零 diff mean 确实由 exporter 独立构造。**

`build_official_transforms()` 从 NPZ 读取输入均值、标准差和各 interval 的增量标准差，并直接构造：

```python
transforms.Normalize(
    np.zeros_like(normalize_diff_std),
    normalize_diff_std,
)
```

它没有从 bridge 取得已经处理好的零均值。随后 `load_official_module()` 设置这些 transforms，`export_reference()` 调用：

```python
module.forward_validation(
    x_norm, OFFICIAL_VARIABLES, interval, steps
)
```

原来的手写 rollout 已被删除。这是针对 B01 的实际修复，而不只是改注释或函数名称。

**第三，policy 确实改变行为，也进入了 digest。**

bridge 的 `denormalize_diff()` 在官方 policy 下直接使用零均值；digest 在序列化开头加入：

```python
parts.append(f"policy={self.policy}".encode())
```

因此，在其他数据相同的情况下，policy 的变化会改变参与 hash 的内容。这里不是“字段存在但没有被使用”。

**判定：B01 所针对的“共享有偏归一化、手写官方 rollout”问题，可以按源码修复关闭。**

但这不包含两个额外结论：

- manifest 写入固定的 `OFFICIAL_STORMER_COMMIT`，**不等于已经验证实际导入目录就是该提交**；这是 B02 的源码身份问题。
- exporter 与 bridge 各自生成的 normalization digest 是否具有一致的字节规范，**尚无跨实现测试证明**。下文单独说明，不能把这个新风险隐藏在 B01 已修复的结论里。

### 2.2 B02：三个局部比较修了，但完整绑定链没有闭合

#### SHA-256：已经比较内容，不再只看长度

`verify_ckpt_sha256()` 明确执行完整字符串比较：

```python
computed_sha256.lower() == expected_sha256.lower()
```

缺少 expected 时也明确返回 `passed=False` 和 `IDENTITY_NOT_BOUND`。这一部分修复成立。

不过，必须准确描述它验证了什么：

`load_result=None` 时调用文件 hash 工具；生产入口传入 `load_result` 时，直接使用 `load_result.checkpoint_sha256`，不是在该函数中重新读取 checkpoint 字节。packet 没有展示 loader 和 hash 工具的完整实现，因此不能额外宣称本次已经重新证明“该记录与实际装入的字节始终一致”。**这也不意味着必须重复计算大文件 hash，而是需要已有加载记录与实际加载对象有明确绑定。**

#### Normalization digest：比较了本地两条来源和 CLI expected，没有比较上游 manifest 身份

`verify_normalization_parity()` 现在要求同时满足：

```text
本地 NPY contract digest == 本地 NPZ contract digest
本地 NPY contract digest == 调用方 expected_digest
```

缺少 expected 时 fail-closed，这部分成立。

**但是，`verify_upstream_parity()` 读取到的上游 checkpoint SHA、normalization policy 和 digest，只被放进结果字典，未用于拒绝缺失或不匹配。**

因此，“本地模型匹配 expected”“本地归一化匹配 expected”与“上游输出确实由同一身份生成”，仍是断开的三件事。

#### Raw input：参数确实用了，但没有绑定到真正参与 parity 的 normalized input

`run_s0_gate()` 确实将 `x_raw_0` 传给 `verify_raw_input_binding()`；后者计算 raw input hash，并与参照侧记录比较。不能继续说 raw input 在整个 gate 中完全没用。

但数值 parity 仍读取：

```python
upstream_input_norm = torch.load(... / "input_norm.pt")
```

然后用这个保存的 tensor 执行 bridge。**没有证明它等于当前 `x_raw` 经当前归一化得到的输入，也没有验证该文件的内容 hash。**

所以目前验证的是：

> raw input 与一个声明的 hash 相符；bridge 与某个保存输入上的一步输出相符。

而不是：

> 当前 raw input → 指定 normalization → 保存的 normalized input → 指定官方模型输出，这整条链相符。

此外，存在 `raw_input_hash.txt` 时，代码优先使用它，不核对它与 manifest 中同名字段是否一致。两个身份记录互相矛盾也可能被接受。

#### `check_version_match`：接入了调用位置，没有接入有效判定

这里有两个独立问题：

**比较双方都来自 `bridge.version`。** 没有独立 expected version，也没有由上游 manifest 重建的 version。

**失败不会阻止总 gate。** `ValueError` 被局部捕获后仅写结果字段；新增的必需 criteria 是 raw-input 和 multistep，`version_match` 没有加入其中。结合 packet 明确保留的 `all(gate_criteria.values())` 汇总逻辑，这个失败字段不会影响最终 PASS。

**这不是测试覆盖不足，而是 diff 中可直接确认的逻辑缺陷。**

#### 6h_4step：现在会加载，但没有验证四步数值一致性

`verify_multistep_reference()` 确实会读取四步文件，检查形状和有限性。因此，“四步文件完全不加载”的旧问题已经修复。

然而，`verify_upstream_parity()` 仍只：

1. 检查一步和四步文件存在；
2. 加载一步输出；
3. 执行 bridge 的 `steps=1`；
4. 比较一步误差。

**四步 tensor 只要同形且有限，即使内容错误，也不会因“四步与官方不一致”被拒绝。**

原 B02 要求的是“比较全部注册 rollout 输出”，不是仅检查它们可读。因此，**B02 不能关闭。**

### 2.3 B10：默认 ps4 和部分目录隔离成立，冻结配置一致性仍未成立

已经成立的修复是：

- exporter 的 `--checkpoint` 默认改为 ps4；
- gate 的 `--checkpoint` 默认改为 ps4；
- gate 根据选定 patch size 加载对应 checkpoint，不再固定 ps2；
- exporter 默认参照目录含 patch size 和 checkpoint SHA 前缀。

**因此，正常默认路径下，ps2 与 ps4 的上游导出目录已经可以区分。**

但仍有以下缺口：

| 问题 | diff 中实际行为 | 判断 |
|---|---|---|
| **没有共用冻结配置** | `GateIdentityConfig` 被导入，但生产路径没有构造或强制消费它。 | 不能称两端已经共同遵循一份完整配置。 |
| **gate 不按完整身份精确找目录** | 只匹配 `upstream_reference_ps{patch_size}_*`，再取排序最后一项。 | 没有根据 expected checkpoint/norm/input 身份选择。 |
| **自定义导出路径没有对应消费入口** | exporter 可指定 `--output-dir`；gate 新 CLI 仍没有显式 upstream-reference 路径参数。 | 非默认目录的连接未打通。 |
| **同一模型、不同输入仍共用默认参照名** | `identity_tag` 不包含 raw input、normalization digest、变量/坐标或 rollout 配置。 | 不满足原 finding 的 checkpoint/输入身份隔离要求。 |
| **gate 自己的结果目录仍共用** | `default_output_dir` 仍是 `artifacts/s0_gate`。 | ps2/ps4 的 S0 报告产物仍可能覆盖。 |

**判定：B10 部分关闭，但还不能称“显式冻结配置贯通、不同身份不会混用或覆盖”。**

---

## 3. fail-open 模式排查结果

### 3.1 父会话点名的三个缺值分支，确实已经修复

在本次 diff 中：

| 场景 | 现在的行为 |
|---|---|
| `expected_sha256=None` | `passed=False`，记录计算值与 `IDENTITY_NOT_BOUND` |
| `expected_digest=None` | `passed=False`，记录计算值与 `IDENTITY_NOT_BOUND` |
| 找不到参照目录或找不到 raw-input expected hash | `passed=False`，记录 `IDENTITY_NOT_BOUND` |

**没有在这三个分支中发现仍保留的 `expected is None → passed=True`。**

### 3.2 仍有语义等价的“没有证明必要条件却放行”

以下是根据代码推导的反例，**不是本次实际运行结果**。

| 遗漏 | 可以构造的静态反例 | 严重性 |
|---|---|---|
| **版本自比较且失败不汇总** | 令 `check_version_match` 抛出 `ValueError`；该局部异常被吞下，其他 criteria 通过时仍可能得到总 PASS。 | **必须修复，直接属于 B02。** |
| **上游 expected identity 未强制比对** | 删除或改错 manifest 的 checkpoint SHA、normalization identity、source identity；保留能数值匹配的一步文件。这些 metadata 本身不会触发拒绝。 | **必须修复，不能用数值接近代替身份一致。** |
| **四步只查结构不查结果** | 换成任意同形、有限的四步 tensor；multistep helper 仍会通过，parity 仍只比较一步。 | **必须修复。** |
| **raw hash 与实际参与推理的输入未绑定** | raw hash 保持正确，但 `input_norm.pt` 和与它配对的一步输出来自另一输入。当前检查没有验证 raw→normalized 的关系。 | **必须修复。** |
| **参照目录歧义被自动消解** | 同一 patch size 有多个不同身份目录；程序不报歧义，而选择其中一个。 | **必须消除猜选，或在选后严格拒绝身份不符。** |

#### “选最近的目录”还存在一个具体转述错误

代码是：

```python
return sorted(matches)[-1]
```

它选择的是**路径字典序最后一项，不是修改时间最近的一项**。SHA 前缀较大的旧目录，也可能排在较新的目标目录之后。

仅有模糊目录发现，在后续存在严格身份拒绝时，不一定造成错误 PASS；**这里的问题是后续身份拒绝也不完整**。因此它不是单纯体验问题。

#### `verify_multistep_reference` 还有一个较低优先级的空集合边界

若传入：

```python
required_steps=()
```

且找到参照目录，循环不会检查任何文件，最后：

```python
len(missing_files) == 0 and len(shape_mismatch) == 0
```

会得到 True。

这是接口层面的“空检查集合通过”。**当前 `run_s0_gate()` 写死 `(1,4)`，所以它不是默认入口的直接漏洞**；但应禁止空注册集合，避免配置接入后重新出现 fail-open。

### 3.3 一个应单列的风险：独立 digest 实现尚未证明兼容

这不是缺值放行，而是可能造成错误拒绝的身份规范问题：

exporter 的 `compute_normalization_digest()` 直接序列化 NPZ 读出的 NumPy 数组；bridge 的加载路径会先 `.float()`。**NPZ 为 float64 时，两边用于 hash 的字节表示就可能不同，即使数值归一化语义相同。**

另外，packet 未展示 bridge `digest` 的全部未改动尾部，不能仅凭这份 diff 确认两边序列化顺序完全一致。

最小处置是：**约定同一 manifest 字节规范，并增加 exporter-digest 与 bridge-digest 的跨实现测试；数值 transforms 仍保持独立。** 不需要为了共享身份格式而重新共享受审的计算路径。

---

## 4. 测试质量抽查结果

### 4.1 数量核对：与正文吻合

对两个附带测试文件作静态计数：

| 文件 | 实际 `test_*` 数量 | 正文描述 |
|---|---:|---|
| `test_normalization_policy.py` | **10** | 10，吻合 |
| `test_s0_gate_identity.py` | **19** | 原 15 + 新 4，吻合 |

文件代码块也分别是 369 行、592 行，与标题一致。报告中的 `68 → 72` 与增加 4 项测试相容，`19+13=32` 也一致。**这些是文本与算术一致，不是独立确认 pytest 已成功运行。**

### 4.2 四个新增 fail-closed 测试：确实测中了所声称的分支

| 测试 | 抽查结论 |
|---|---|
| `test_gate_fails_closed_ckpt_sha256_none` | 创建了实际存在的文件，显式传 `None`，检查 False、计算出的 hash 和指定原因；不是拿“文件不存在”冒充缺 expected。 |
| `test_gate_fails_closed_normalization_digest_none` | 使用可读取的 NPZ 和构造成功的 contract，传 `None` 并检查指定原因；有效覆盖该分支。 |
| `test_gate_fails_closed_raw_input_no_upstream_dir` | 确实检查没有参照目录时 False，并保留计算值。 |
| `test_gate_fails_closed_raw_input_no_hash_in_reference` | 确实创建参照目录与 manifest，同时缺少 manifest hash 和 sidecar hash；场景构造正确。 |

**这四项不是“名字对、实际测了别的东西”。** 可以加强 normalization 测试，额外断言 `digests_match_npz is True`，使“唯一缺失条件就是 expected”更明确，但现有分支测试已有实质意义。

### 4.3 其他测试存在的证据强度问题

| 测试或测试组 | 问题 | 应如何补强 |
|---|---|---|
| **整个 identity 测试文件** | 文件头宣称验证真实 orchestration，但 19 项测试没有调用 `run_s0_gate()`、`main()`，也没有调用 `verify_upstream_parity()`；实际测试的是四个 helper 和 dataclass。 | 用小模型/合成资产替代昂贵资源，保留真实编排、身份校验和汇总逻辑，验证错误身份确实导致总 gate False/退出非零。 |
| **`test_policy_changes_digest`** | official 和 legacy 不仅 policy 不同，加载后的 diff_mean 数值也不同；即使删掉 digest 中的 policy 字段，该测试仍可能通过。 | 构造张量完全相同、仅 policy 不同的两个 contract，再比较 digest。 |
| **`test_official_policy_ignores_nonzero_diff_mean_in_denormalize`** | factory 已经把 diff_mean 清零，没有真的把“非零 diff_mean + official policy”交给 denormalizer。 | 直接构造 official contract，保留非零 diff_mean，断言运行时仍忽略它。 |
| **正确 normalization digest 测试** | contract 和参照都从同一 NPZ 构造，没有经过生产使用的 NPY→contract 路径，更没有调用 exporter 的独立 digest。 | 增加 NPY/NPZ 一致的正例，单独修改 NPY 的负例，以及 exporter/bridge digest 互操作测试。 |
| **multistep 正例与缺文件测试** | 正例用随机四步 tensor，证明的是形状/可读性，不是四步 parity；缺文件测试的其他剩余 tensor 还是不符合要求的 16×32。 | 先建立完整合法正例，再一次只删除或篡改一个四步产物；另外必须有“同形、有限但数值错误”导致总 gate 失败的测试。 |
| **`test_gate_accepts_matching_raw_input`** | 只更新 sidecar hash，manifest 中仍保留旧 hash，却断言通过。这符合当前 helper 行为，但不能证明统一输入身份。 | 两份记录冲突时应拒绝，或取消双重权威来源；再验证 raw 与 `input_norm.pt` 的关系。 |
| **`GateIdentityConfig` 的三项测试** | 只验证未被生产消费的类；namespacing 测试同时改变 SHA 和 patch size，无法证明每个关键身份字段单独变化都被处理。 | 测真实 exporter/gate 配置入口；分别改变输入、norm、checkpoint、patch，并验证确定性选择或明确拒绝。 |

还有一个具体的 fixture 问题：NPY 的 `diff_std` 随机生成，而 NPZ 的对应值写死为 `0.5`。因此，这个 fixture **不是一个可以直接作为真实 orchestration 成功基线的“匹配资产集”**；目前测试绕开了这项不一致。

我没有发现这四个新增缺值测试通过 mock 掉真正分支来“假测”；主要问题是**没有测试组合后的系统性质，却把 helper 测试称为系统性质证明**。

---

## 5. 一致性检查结果

### 明确存在的不一致

**一，正文称 B13/B14 尚未开始，但 diff 已经对其做了定向修改。**

`run_s0_gate()` 的外层异常分支新增了：

```python
result["s0_gate_pass"] = False
result["status"] = "exception"
```

报告器也新增了对缺失数值的检查，避免直接对 `"N/A"` 使用浮点格式。

所以准确状态应是：**“B13/B14 已有部分修补，尚未完成该阶段验证”**，而不是“尚未触碰”。不能反过来仅凭这些 diff 宣告它们已经完全关闭。

**二，正文称 exporter/gate 各自计算等价的 identity_tag，但 gate 实际只是 glob。**

exporter 确实生成 `ps{patch_size}_{sha[:8]}_zd`；gate 没有按 expected SHA 生成等价完整标签，而是根据 patch size 模糊查找。**当前问题已经不只是未来 DRY 漂移，而是生产消费者现在就没有使用完整身份定位。**

**三，“已接入 `check_version_match`”在语法意义上正确，在修复 B02 的意义上不正确。**

确实有调用，但参数自比较、失败不汇总，不能视为完成执行入口的身份约束。

**四，identity 测试文件头声称测试真实编排、变量/网格身份，正文测试并不支持。**

没有真正编排调用，也没有变量顺序或真实坐标身份不匹配的测试。输出 tensor 的 shape 测试不能替代这些性质。

### 较小但明确的注释错误

`test_gate_identity_config_sha256_validation` 使用的：

```python
"abc123def456" * 5 + "ab"
```

长度是 **62**，不是注释写的 64。mock manifest 的 `"abc123" * 11` 长度为 **66**，注释说会截断，但附带代码里没有截断。这些测试验证了字符串相等性，没有证明 SHA 格式有效。

### 没有发现的不一致

**两个新增文件的行数与测试数量、增加四项测试后的总数变化，均一致。** 对附件中 unified diff 的静态检查也未发现 hunk 声明行数不匹配；不过，“完整 diff”只表示改动完整，不表示包含所有未改动函数的全文。

---

## 6. 仍然存在的已知限制

**真实官方数值一致性：这一点无法仅凭静态 diff 判断，需要 Stage 2 真实 GPU S0 结果。** 包括新 exporter 是否能加载实际 checkpoint、官方模块的 transforms 是否按预期生效，以及一步和四步在指定精度下的数值一致性。packet 对“没有执行 GPU/xformers S0”的声明是准确且必要的。

**未展示函数的内部行为：证据不足。** 完整的 `_compute_file_sha256`、`load_stormer_checkpoint_detailed`、官方 `GlobalForecastIterativeModule`，以及 bridge digest 的未改动尾部，都不在本次 diff 展示范围内。需要这些函数全文或相应针对性测试，才能对其内部性质作进一步判断；不能用上轮记忆补齐。

**测试结果：只确认记录与代码结构相容。** 68/32/72 passed 是 packet 报告的父会话运行事实，本次没有独立重跑，也不能从通过数量推导未覆盖性质成立。

**阶段范围：大体认可，但需更新 B13/B14 状态，并补记本次发现的剩余缺口。** 评分合同、数据准入、bank 和后续实验仍未完成，不属于本次重新审计对象；不过，上游身份强制匹配、raw→normalized 绑定、四步数值比较、确定性参照目录选择，仍是 **Stage 1A 自身未完成的事项**，不能统统推给“等 Stage 2 跑了再看”。

---

## 7. 一句话总体建议

**B01 可按源码修复关闭，但 Stage 1A 暂不能整体标为 `CLOSED`：先补上独立身份比对及其总 gate 约束、raw→normalized→output 绑定、四步数值比较和精确参照目录选择，并用真实编排的 CPU 反例验证，再进入可被采信的 Stage 2 GPU S0；Stage 1B/1C 可以继续，但不能以 B02/B10 已关闭为前提。**
