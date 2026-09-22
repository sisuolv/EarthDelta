# EarthDelta Stage 1A 独立复查结论（B01 / B02 / B10）

## 1. 总体结论

**这次修复有实质进展，但 B01、B02、B10 不能整体标记为 `CLOSED`。关键问题不是缺少 GPU 结果，而是 diff 中仍存在可直接确认的身份校验缺口。**

本次只依据 `STAGE1_PATCH_PACKET.md` 中的原始 finding、diff 和测试全文判断，没有访问 GitHub、导入受审代码或重跑 pytest；另外对附带测试做了静态定义计数。

| Finding | 独立判断 | 一句话理由 |
|---|---|---|
| **B01** | **同意关闭——限 Stage 1A 的代码修复层面** | exporter 确实改为独立构造零均值 transforms，并调用官方 iterative module；bridge 的 policy 也确实进入 digest。真实数值一致性仍待 Stage 2。 |
| **B02** | **不同意关闭** | 局部 expected-value 比较已改善，但版本校验仍是自身比较，上游 manifest 身份仍未真正绑定，4-step 仅检查文件内容基本合法性，没有数值 parity。 |
| **B10** | **部分同意** | 两端默认 ps4、默认参考目录区分 ps2/ps4 已解决；但生产路径没有共用冻结配置，gate 仍按 patch size 模糊选目录，无法保证选择的是正确身份的参照。 |

最明确的一处问题是：

```python
check_version_match(bridge.version, bridge.version)
```

这不是“预期身份与实际身份”的比较，而是**实际身份与自身比较**；而且该检查的失败结果没有被接入最终 `gate_criteria`。

---

# 2. B01／B02／B10 逐条详细分析

## 2.1 B01：核心修复对症，代码层面可以关闭

### 已确认修复的部分

**第一，exporter 不再通过 `NormalizationContract` 生成参考输出。**

diff 删除了 exporter 对 `NormalizationContract` 和 `DEFAULT_VARIABLES` 的导入，改为独立变量列表和：

```python
from stormer.models.iterative_module import GlobalForecastIterativeModule
```

保留的 bridge 导入只是 `_compute_file_sha256`。这仍是 Python 模块依赖，但**不是复用受审的归一化计算**，不能仅凭这个导入否定 B01 的修复。

**第二，零 diff_mean 确实在 exporter 自己的数值路径中构造。**

`build_official_transforms()` 独立读取 NPZ，并明确执行：

```python
diff_transforms[interval] = transforms.Normalize(
    np.zeros_like(normalize_diff_std),
    normalize_diff_std,
)
```

它没有从 bridge 获取已经构造好的均值或 transform。

**第三，不再保留原先那套手写的“官方 rollout”。**

新的 `load_official_module()` 包装 `GlobalForecastIterativeModule`、设置 transforms；`export_reference()` 直接调用：

```python
module.forward_validation(x_norm, OFFICIAL_VARIABLES, interval, steps)
```

原先手写的 `run_official_inference()` 则被删除。

**第四，policy 确实影响 bridge 的计算和 digest。**

`denormalize_diff()` 在 official policy 下强制使用零均值；digest 输入也新增：

```python
parts.append(f"policy={self.policy}".encode())
```

因此，“policy 字段只是摆设、没有进入身份摘要”这一担忧，在当前 diff 中不成立。

### 关闭范围

我同意关闭的是：**“参考输出复用了受审归一化、未实际调用官方 rollout”这项实现问题。**

这不代表：

- 官方源码实际就是所声明的 pinned commit；
- 官方 module 的真实 checkpoint 加载和运行已经成功；
- bridge 与官方路径已达到数值容差；
- 旧缓存已经被执行入口正确拒绝。

前两项分别需要源码身份证据与运行证据；最后一项仍受 B02 未关闭影响。**不应仅因 Stage 2 尚未执行，就否定上述已经清楚可见的 B01 代码修复。**

---

## 2.2 B02：局部校验修好了，但完整身份绑定仍未形成

### A. Checkpoint SHA：已从“长度检查”变成“值比较”

现在明确比较：

```python
computed_sha256.lower() == expected_sha256.lower()
```

`expected_sha256=None` 时，也确实返回：

```python
passed = False
reason = "IDENTITY_NOT_BOUND: ..."
```

因此，这两个修改是有效的，不能继续说它“仍然只看 SHA 长度”。

不过，应准确说明它的证据范围：

- `load_result=None` 时，函数调用文件哈希工具；
- 实际 gate 传入 `load_result` 时，函数使用 `load_result.checkpoint_sha256`。

**后者不必重复读取整个 checkpoint 才算正确**，但需要加载器保证该摘要绑定了实际加载的字节。packet 没有展示未修改的加载器／哈希工具全文，我不对该内部实现额外作肯定或否定判断。

### B. Normalization digest：本地比较成立，上游绑定仍缺失

`verify_normalization_parity()` 现在同时要求：

```python
norm_from_npy.digest == expected_digest
norm_from_npy.digest == norm_from_npz.digest
```

缺少 expected digest 时也会拒绝，修复有效。

**但这只是“本地 NPY—本地 NPZ—调用方 expected”的一致性。**

`verify_upstream_parity()` 从上游 manifest 读取的：

```python
checkpoint_sha256
normalization_policy
normalization_digest
```

仍然只是写入结果记录，没有在新增逻辑中与实际 bridge、冻结 expected 值进行比较。

所以，“本地 normalization 已核验”不能替代“上游参照使用了同一 normalization”。

### C. `x_raw`：不再完全闲置，但没有绑定实际参与 parity 的输入

新的 `verify_raw_input_binding()` 确实计算 `x_raw` 哈希，并与参照中的哈希比较；目录或 expected hash 缺失时拒绝。这部分修复成立。

但 parity 使用的实际输入仍是：

```python
upstream_input_norm = torch.load(... / "input_norm.pt")
```

而不是从当前 `x_raw` 经已经验证的 normalization 重建，并检查两者一致。

因此，仍可出现以下**文件混用情景**：

> `raw_input_hash` 对应输入 A；`input_norm.pt` 和 1-step 输出却来自输入 B。  
> raw hash 检查认可 A，而 parity 实际在 B 上运行。

当前新增检查没有把这两条链连接起来。这是静态调用链能够支持的遗漏判断，**不是本轮已经实际运行复现的结果**。

### D. `check_version_match`：形式上调用了，语义上没有生效

当前调用是：

```python
check_version_match(bridge.version, bridge.version)
```

至少有两个独立问题：

1. **没有外部 expected 或上游版本对象参与比较。**
2. 捕获 `ValueError` 后，只写 `result["version_match"] = False`；没有加入 `gate_criteria`，也没有继续抛出异常。

所以，即使未来这处调用能报告 mismatch，按照当前展示的汇总方式，也不保证使总体 gate 失败。

这仍然是 B02 意义上的**未强制执行身份合同**。

### E. 6h_4step：确实被加载了，但没有与 bridge 做数值比较

这里需要纠正一个容易说过头的结论：

**不能说 4-step 现在“仍然完全没有加载”。** `verify_multistep_reference()` 的确会加载它，并检查：

- 文件存在；
- 可反序列化；
- shape 为 `(1, 69, 128, 256)`；
- 数值有限。

但 `verify_upstream_parity()` 仍只加载 **1-step 输出**进行实际比较，运行的 bridge 调用也是 `steps=1`。

因此：

> 将 4-step 文件替换成任意同 shape、有限值的错误张量，不会被目前新增的 4-step 检查识别为“数值参照错误”。

**“加载并检查 shape”已经修复；“全部注册时效的官方 parity”没有修复。**

### F. Source pin、变量序和真实坐标：仍未建立完整核对

exporter 将常量 `OFFICIAL_STORMER_COMMIT` 写入 manifest，但 packet 展示的新增逻辑没有检查实际导入源码的 revision／内容与该常量一致。变量数或固定 tensor shape，也不等于验证了变量顺序与真实经纬度。

原 B02 已明确包含这些身份要求，因此不能只凭新增几个 hash 字段宣布它全部关闭。

**B02 结论：不同意关闭。**

---

## 2.3 B10：默认 checkpoint 修正了，参照选择仍不可靠

### 已解决的部分

两端 CLI 的默认 checkpoint 均已改为 `ps4`；gate 加载 checkpoint 时也使用传入 `patch_size`，不再固定 ps2。

exporter 默认目录：

```python
upstream_reference_ps{patch_size}_{sha_prefix}_zd
```

确实区分了默认 ps2 与 ps4 的参考目录。

### 未解决的部分

**第一，没有真正共用一份冻结配置。**

`GateIdentityConfig` 虽然新增并导入，但生产路径没有实例化它。exporter 自己构造完整 tag，gate 实际只做 patch-size 通配搜索；两者并不是“分别计算同一个完整 tag”。

**第二，多个目录时并不是选“最新”，而是选字典序最后一个。**

```python
return sorted(matches)[-1]
```

没有读取修改时间或 manifest 时间，更没有依据 expected SHA、normalization 或输入身份精确匹配。

**第三，仍会回退到没有 patch-size 约束的 legacy 路径。**

这不是自动证明会错误 PASS；如果后续身份检查完整，错误目录可以被安全拒绝。**但当前 B02 恰好缺少那些核对，因此两者组合形成真实的误绑定风险。**

**第四，默认 tag 不含输入或 normalization 内容身份。**

同一个 checkpoint、同一 policy 下，更换输入或 normalization 数组，默认仍映射到相同目录。`--output-dir` 还可以直接绕过默认命名；gate 又没有新增显式的 `--upstream-reference-dir` 来消费指定 bundle。不能据此宣称“全部不同配置的产物都不会混用或覆盖”。

**B10 结论：部分同意。** 默认 ps2／ps4 错配已修，但完整参照定位与配置共享尚未关闭。

---

# 3. fail-open 模式排查结果

## 3.1 已点名的三个 expected 缺失分支，确实修好了

没有发现以下三个函数在相关缺失分支中仍保留 `passed=True`：

- `verify_ckpt_sha256`
- `verify_normalization_parity`
- `verify_raw_input_binding`

它们当前都明确拒绝，并保留诊断摘要。不能把已经修掉的分支继续列作遗漏。

## 3.2 仍存在的同类语义缺口

| 遗漏 | 为什么属于同类问题 | 当前影响 |
|---|---|---|
| **上游身份字段缺失或错误，只记录、不拒绝** | manifest 中的 checkpoint／normalization 身份没有成为判定条件，等价于“参照身份未绑定仍可继续认证” | **B02 关闭阻塞** |
| **版本自比较，失败结果不纳入汇总** | 有校验函数名称，却没有独立 expected，也没有强制失败传播 | **B02 关闭阻塞** |
| **4-step 只有 shape／finite 检查** | 有参照文件不等于有正确的注册时效参照 | **B02 关闭阻塞** |
| **raw hash 与 `input_norm.pt` 分离** | 声明的输入身份与实际 parity 输入可来自不同样本 | **B02 关闭阻塞** |
| **模糊目录匹配、legacy 回退** | 没有唯一身份匹配就选取一个 bundle，后续检查又未补齐 | **B10 关闭阻塞** |
| **`required_steps=()` 真空通过** | 若找到目录，循环不执行，两个错误列表为空，于是 `passed=True` | **接口边界遗漏；当前 main 固定 `(1,4)`，不会触发** |

最后一项可以直接从下面逻辑推出：

```python
for steps in required_steps:
    ...

result["passed"] = (
    len(missing_files) == 0 and len(shape_mismatch) == 0
)
```

应该拒绝空的注册时效集合，但不应将这个非默认调用边界，与前面几项实际认证路径缺口同等放大。

### 一个额外的一致性风险：两份 normalization digest 算法没有跨端验证

exporter 的 `compute_normalization_digest()` 对 NPZ 原始数组直接使用 `.tobytes()`；bridge 的 NPZ 加载路径则会转为 `.float()`。**对非 FP32 的 NPZ，二者至少在字节编码规则上不一致。**

这主要可能导致**错误拒绝或无法形成共同身份规范**，不是当前已经证明的 fail-open。但它意味着下一步不能只是粗暴增加：

```python
upstream_digest == bridge.digest
```

就认定完整绑定已经完成。应先定义相同的规范化 dtype、字段顺序和有效 policy，再用相同 NPZ 做跨实现摘要测试；归一化数值运算仍保持独立。

---

# 4. 测试质量抽查结果

## 4.1 四个新增 fail-closed 测试，确实测中了相应缺失分支

| 测试 | 是否真正构造缺失 expected 场景 | 判断 |
|---|---|---|
| `test_gate_fails_closed_ckpt_sha256_none` | 文件确实存在，expected 明确为 None，还检查 computed hash 与拒绝原因 | **有效** |
| `test_gate_fails_closed_normalization_digest_none` | NPZ 可加载、norm 已构造，expected 为 None | **有效** |
| `test_gate_fails_closed_raw_input_no_upstream_dir` | 不创建参照目录，检查拒绝与诊断 hash | **有效，但仅覆盖无目录情形** |
| `test_gate_fails_closed_raw_input_no_hash_in_reference` | 创建参照目录及 manifest，同时缺少文本 hash 和 manifest hash | **有效，覆盖关键遗漏路径** |

这些测试不是简单通过“文件不存在”的无关早退来冒充 expected 缺失测试，尤其最后一个场景构造正确。

**不过，它们证明的是 helper 行为，不是整套 gate 已 fail-closed。**

## 4.2 主要的测试覆盖问题

### ① 测试文件声称验证真实 orchestration，但没有调用它

文件开头声称：

> These tests verify that the REAL s0_gate orchestration …

实际新增 19 个测试只调用四个 verify helper 和 `GateIdentityConfig`，没有调用：

- `run_s0_gate`
- `main`
- `verify_upstream_parity`
- exporter 的实际配置／参照生成函数。

这解释了为什么**自比较、未纳入 gate 汇总、4-step 未做数值比较**都能与当前通过数字同时存在。

旧 `test_s0_fail_closed.py` 在 packet 中只有一处签名适配 diff，没有全文；因此我不推断那 13 项旧测试分别覆盖了什么。

### ② `test_policy_changes_digest` 没有隔离 policy 字段的作用

它分别加载 official 和 legacy 对象，而两者的 `diff_mean` 数组本来就不同。即使删除：

```python
parts.append(f"policy={self.policy}".encode())
```

该测试仍可能通过。

需要补一个**所有数组完全相同、只有 policy 不同**的测试。当前代码确实加入了 policy，但现有测试不能专门防止未来移除该字段。

### ③ `test_official_policy_ignores_nonzero_diff_mean_in_denormalize` 没有真正给 official 对象保留非零均值

它先调用 official `from_npz_dir()`，均值在加载时已经被置零。于是即使运行时的 policy 分支退化成“直接使用存储均值”，测试也可能继续通过。

应直接构造一个：

```text
policy = official_zero_diff_mean
stored diff_mean = nonzero
```

的对象，再验证运行时确实忽略该非零均值。

### ④ `test_gate_accepts_matching_raw_input` 接受了两处 hash 冲突

测试只修改 `raw_input_hash.txt`，没有同步修改 fixture manifest 中的 `raw_input_hash`，然后断言通过。

对“文本文件优先、manifest 仅作 fallback”的单函数约定，这个测试并非自相矛盾；但它**不能证明整个 bundle 身份一致**。最好指定唯一权威记录，或在两处同时存在时要求一致。

### ⑤ multistep 测试证明文件检查，不证明 trajectory parity

`test_gate_accepts_all_multistep_present` 将两个输出文件改为正确 shape 的随机张量，即获得通过。这符合该 helper 的实际实现，但不能证明 4-step 结果正确。

需要补的不是另一个“缺文件测试”，而是：

> 一个其余身份均匹配的参照 bundle，仅将 4-step 输出改成错误但同 shape、有限的张量，真实 gate 必须拒绝。

### ⑥ `GateIdentityConfig` 测试没有验证生产路径

现有三个测试只检验 dataclass 的比较／字符串性质，未检查 exporter 和 gate 实际采用它。

另外：

```python
"abc123def456" * 5 + "ab"
```

实际是 **62 个字符**，不是注释所说的 64。`test_gate_identity_config_sha256_validation` 主要证明“字符串与自身相等”，没有证明输入是合法 SHA-256。

这不是生产 gate 已接受错误 checkpoint 的直接证据，但确实说明该测试的“validation”含义比名称弱。

## 4.3 测试数量：没有发现算术矛盾

我对两个代码块做了静态计数：

- `test_normalization_policy.py`：**369 行，10 个测试函数**；
- `test_s0_gate_identity.py`：**592 行，19 个测试函数**，其中最后 4 个为补充测试。

与 packet 的 `15→19`、`68→72`、`19+13=32` 一致；diff 也确有六个文件头，另附两个新测试文件。

**这些数量是自洽的，但不能代替实际测试运行或完整链路覆盖。**

---

# 5. 一致性检查结果

发现以下正文／注释与实际 diff 或测试之间的不一致：

| 表述 | 实际证据 | 应如何更正 |
|---|---|---|
| **“`check_version_match` 已接入真正执行入口”** | 确实被调用，但比较自身，且结果未加入 gate 汇总 | 改为“已添加调用，但强制身份核验尚未完成” |
| **“生产两端各自算出等价 identity_tag”** | exporter 算完整 tag；gate 仅按 patch size glob，不算同一个完整 tag | 改为“exporter 命名与 gate 定位仍未统一” |
| **“Return most recent if multiple”** | `sorted(matches)[-1]` 是路径字典序，不是时间顺序 | 修改代码或注释；更合理的是精确身份定位，而非选最新 |
| **“B13、B14 尚未开始／没有触碰”** | diff 已增加异常后撤销 PASS，并修改缺失值报告格式化 | 改为“已有部分针对性修改，尚未完成相应独立验证” |
| **新增测试验证真实 orchestration、变量／grid identity** | 新测试没有调用总 gate，也没有变量顺序／真实坐标不匹配场景 | 将覆盖声明收窄，或补真实调用链测试 |
| **配置测试 SHA 为 64 字符；fixture 会截断 66 字符 SHA** | 前者实际 62；后者没有展示截断操作 | 修正 fixture 与注释 |

其中 B13/B14 的状态不一致有直接 diff 证据：外层异常现在会设置 `s0_gate_pass=False`、`status="exception"`；报告器也开始对缺失数值作保护。**这两处是实际改动，但不能据此顺便宣布 B13/B14 全部关闭。**

另外，packet 所称 exporter **430 行改动**、gate **486 行改动**与静态加减行计数一致，没有发现这两项统计不符。

---

# 6. 仍然存在的已知限制

**真实官方数值一致性尚未验证。** 官方 module 的真实加载、xformers 执行、多步误差容差和实际 checkpoint／输入组合是否正确，**这一点无法仅凭静态 diff 判断，需要 Stage 2 真实 GPU S0 结果**。packet 对 CPU-only 的声明是准确的，不能将跳过 GPU 测试当作 GPU 验证通过。

**“完整 diff”不等于所有相关文件全文。** 对没有展示的加载器内部、完整 digest 尾部、旧测试和官方类实现，本次不能补写其行为。尤其是实际导入源码是否等于 pinned commit，需要源码身份记录或对应检查代码。

**两套 normalization digest 的一致性仍缺证据。** 需要独立 transforms 与 bridge 使用相同测试 NPZ，覆盖多个 interval、FP32／非 FP32 输入，明确共同序列化规范；不能只比较 bridge 自己加载两次的结果。

**Stage 1B／1C 及后续天气工作仍不能视为完成。** 我认可 packet 对评分合同、数据准入、Fs／bank、候选缓存和合法 policy 实验尚未完成的范围声明；唯一需要更新的是：B13/B14 已有部分代码改动，而不是完全没有开始。

在不扩大 scope 的前提下，Stage 1A 还应完成三组最小补丁：

1. **绑定同一个参照 bundle 的 expected／actual 身份。** 删除版本自比较；核对上游 checkpoint、normalization、源码、变量／坐标和输入，任何 mismatch 必须影响最终 verdict。
2. **对 `(6h,1step)` 与 `(6h,4step)` 都执行实际 parity。** 同时验证存储的 normalized input 与当前 raw input／normalization 一致，而不是只检查文件存在和 shape。
3. **精确解析参照目录并补总入口测试。** 从同一冻结配置定位唯一目录；拒绝歧义和未经校验的 legacy 回退；测试真实 `run_s0_gate/main` 对身份缺失、版本失败、错误 4-step、输入混用和目录歧义的结果。

这不妨碍独立开展既定 Stage 1B／1C 的 CPU 工作，但不能带着“B02 已关闭”的标签签发 Stage 2 S0 通过证明。

---

# 7. 一句话总体建议

**B01 可以按 Stage 1A 代码层修复关闭，B02 不应关闭、B10 仅部分关闭；应先解决“版本自比较且未纳入判定、上游 bundle 身份未绑定、4-step 缺数值 parity、目录模糊选择”这四个具体问题，再把本批修复作为已关闭的前提进入 Stage 2 正式 S0 验证。**
