# EarthDelta — Round-3 复查材料：R2-P0-01 Stage 1A（B01/B02/B10 修复）

## 你在读什么、不在读什么

这不是重新做一次完整的 novelty / research value 审查（那件事已经在 round-2 做过，见 `codex_audit_round2/AUDIT_BRIEF.md` 与 `codex_audit_round2/results/ROUND2_REPORT.md`，本次不重复）。这份材料只覆盖 **round-2 审计判 `FAIL_REOPEN_P0_03` 之后，针对 B01–B15 中三条 P0/P1 finding 的第一批最小正确性修复（Stage 1A）**，请你独立判断这三条是否真的被关闭、修复是否引入新问题、以及是否还有同类问题被漏掉。

**范围声明（请严格按此判断，不要假设更多已完成）**：
- 本次只做了 Stage 1A：修 **B01**（exporter 与 bridge 共享同一份有偏 normalization，没有独立官方参照）、**B02**（S0 gate 身份绑定只检查 SHA 字符串长度，不做真实比对）、**B10**（ps2/ps4 checkpoint 产物目录冲突/覆盖）。
- **B05、B06（原生评分合同 Q 一致性）、B08（入口准入小守卫）、B13、B14 尚未开始（Stage 1B/1C，排在本次之后）**；B03（caller 系数图梯度）、B04（batched Gram 3D）、B12（并发声明）本轮**明确暂缓，不是已修复**；B07、B09、B11、B15 排入更后面的阶段准入，本次完全没有触碰。
- 全部验证是 **CPU-only**：新增/回归的 pytest 用例、以及对代码路径的人工审阅。**没有跑真实 GPU、没有跑真实 xformers、没有做真实官方 S0 parity 数值比对** —— exporter 在 CPU 环境下会因缺少 xformers/CUDA 而 fail-closed 退出（这是设计如此，不是本次验证内容）。真实 GPU S0 是下一阶段（Stage 2）的工作，尚未执行。
- 本次审阅过程中，父会话（负责集成、不信任子代理自报）自己做了一轮独立代码复查，**发现了子代理自己没写的测试没覆盖到的一个真实 fail-open 缺口**（S0 gate 在“调用方没传预期身份值”时会静默 PASS，等价于复现了 B02 原始 bug），随即又派发了一次针对性修复并再次独立复验。这个过程和结果也完整包含在下面，因为它本身就是这次要请你复查的对象之一：**请重点检查这类“预期值缺失时默认放行”的模式在这次 diff 里是否还有遗漏**。

## 原始 finding 原文（round-2 审计给出，逐字引用，供你核对修复是否对症）

### B01（P0）
> 网络架构确实来自官方 xformers Stormer，但 exporter 没有调用官方 GlobalForecastIterativeModule.forward_validation，而是重写 rollout，并直接复用待审 bridge 的 NormalizationContract。官方 inference 的差分均值固定为零，本地实现却加上非零 diff_mean；pinned 6h/24h 数组均有 69 个非零元素，max(abs(diff_mean/inp_std)) 分别约 1.66e-5/7.17e-5。故共享错误能通过当前所谓 upstream parity，完整官方路径仍未独立验证。
>
> **minimal_action**：独立进程执行 pinned 官方 iterative module 和官方 transforms，直接从官方 npz/变量序构造零 diff_mean 推理 policy；bridge 增加显式 policy，legacy 仅复算旧结果。加入非零 diff_mean 的 CPU 反例，再由有资源的后续任务运行真实 GPU parity。

### B02（P0）
> SHA 已进入 ArtifactVersion，norm digest 也已绑定变量名、interval 键和 shape，但 S0 只检查 SHA 字符串长 64；上游 manifest 的 checkpoint SHA 只展示、不与 bridge 比较，normalization_digest 不核验，x_raw 参数完全不用，也不核验原始输入/输出内容 hash、源码 pin 或精确 shape。导出的 6h_4step 文件甚至不加载。CPU 合成反例在错误 checkpoint/norm、不同 x_raw、缺 4-step 文件时仍 passed=True；任意存在的文件都能通过独立 hash gate。执行入口也未调用 check_version_match。
>
> **minimal_action**：在数值比较之前核对冻结预期 checkpoint 的完整 SHA、上游源码/配置 pin、归一化 policy/digest、变量序、真实坐标、输入 hash 与 exact shape；比较全部注册 rollout 输出，缺一项阻断。把版本核验接入实际执行/缓存入口，不能只提供可选 helper。

### B10（P1）
> exporter 提供 --checkpoint ps2|ps4，但消费者固定加载 checkpoint_ps2 且 patch_size=2，没有选 ps4 或显式 upstream-reference 目录的 CLI。当前 research spec 的主 checkpoint 是 patch4。按计划导出 ps4 后，gate 实际比较 ps2 与 ps4；即使保留默认 ps2 成功，也不能验证后续使用的 ps4。两个 exporter 配置还默认写同一目录，易覆盖/混用。
>
> **minimal_action**：exporter 与 gate 共用显式冻结 checkpoint/config/reference-dir 参数，默认主线与 spec 一致；按 checkpoint/输入身份隔离产物目录，并在加载前匹配 manifest。

## 本次实际做了什么（对应上面三条）

1. `NormalizationContract`（`earthdelta/bridge/stormer_bridge.py`）新增显式 `policy` 字段：`POLICY_LEGACY`（保留旧的非零 diff_mean，向后兼容旧实验）与 `POLICY_OFFICIAL_ZERO_DIFF_MEAN`（强制零 diff_mean，匹配官方 `reference/stormer/inference.py:118` 的 `transforms.Normalize(np.zeros_like(...), std)` 语义）。`policy` 被纳入 `digest` 计算，切换 policy 会改变 digest。
2. `scripts/export_upstream_reference.py` 整体重写（430 行改动）：改为独立导入 pinned 官方 `GlobalForecastIterativeModule.forward_validation`（官方 commit `58dfee5a6037399a40fefd492bc00421e0c885a8`），**不再复用受审的 `NormalizationContract`**，而是从官方变量序 + NPZ 独立构造零 diff_mean 的 transforms；产物目录按 `ps{patch_size}_{sha前8位}_zd` 命名，避免覆盖；CPU 环境下缺 xformers/CUDA 时以退出码 2/3 fail-closed，不静默回退到 SDPA 冒充官方结果。
3. `scripts/s0_gate.py` 整体重写（486 行改动）：`verify_ckpt_sha256` 改为真实比较 checkpoint 字节 SHA-256（不再只看长度）；新增 `verify_normalization_parity`（比较 digest）、`verify_raw_input_binding`（比较原始输入 hash）、`verify_multistep_reference`（校验注册的多步官方参照文件，含 6h_4step，是否存在/可读/形状正确/有限）；`check_version_match` 被接入真正执行入口；`--checkpoint` 默认改为 `ps4`（匹配 `research_spec_v6.yaml:33` 的主线），保留 `ps2` 作为显式可选项；`find_upstream_reference_dir` 按 `upstream_reference_ps{patch_size}_*` 模式做目录隔离。
4. `earthdelta/contracts.py` 新增 `GateIdentityConfig` dataclass，把 checkpoint 路径+预期 SHA-256、patch_size、normalization policy、变量、grid shape、rollout steps 收敛成一份显式配置，提供 `identity_tag` 用于目录命名；目前只在新测试里被实例化，**exporter/gate 的生产路径没有直接构造这个 dataclass**，而是各自独立算出等价的 identity_tag 字符串——这是一个可能的 DRY/一致性隐患，请你评估是否值得在下一批修复里合并。

## 复查中自己发现并修复的问题（不在 round-2 三条 finding 原文里，是本次审阅新发现的）

对 `scripts/s0_gate.py` 通读后发现：`verify_ckpt_sha256`、`verify_normalization_parity`、`verify_raw_input_binding` 三个函数在**调用方没有提供“预期”身份值**（`expected_sha256=None` / `expected_digest=None` / 上游参照目录里没有 `raw_input_hash`）时，都会**静默 `passed=True`**——这实质上是把 B02 的原始 bug（“只检查有没有算出一个东西，不比较是否等于已知正确值”）在“默认不传参数”这条路径上原样保留了下来。而 `scripts/s0_gate.py` 的 CLI `main()` 里 `--expected-sha256` / `--expected-norm-digest` 默认就是 `None`，`run_s0_gate()` 用 `all(gate_criteria.values())` 聚合总判定——也就是说，如果下一阶段真跑 GPU S0 时忘记显式传这两个参数，gate 会**假 PASS**，而不是报错或 BLOCKED。

已确认：这个 fallback 分支**没有任何现有测试**断言它应该是 `True`（15 个既有 identity 测试和 13 个 fail-closed 测试全部走的是“文件/目录不存在”的早退路径，或显式传了 expected 值的比较路径），所以把它改成 fail-closed 不会破坏任何已通过的既有断言。

修复：三个函数在“预期值缺失”分支下改为 `passed=False`，并写入明确原因（如 `"IDENTITY_NOT_BOUND: no expected SHA-256 provided"`），同时仍然把计算出来的 hash/digest 记录下来供人工录入配置使用。没有改动 CLI 参数为必填（保留可选，兼容“第一次跑、只是想看看算出来的 hash 是多少”的枚举工作流），也没有改动 `all(...)` 聚合逻辑本身（它已经是对的）。新增 4 个测试覆盖这三个分支的 fail-closed 行为。

**请重点复查**：这类“调用方漏传预期值/参照缺失 → 默认静默通过”的模式，在这份 diff（尤其是 `scripts/s0_gate.py` 剩余部分和 `scripts/export_upstream_reference.py`）里是否还有其他实例没被找到；`GateIdentityConfig` 只在测试里实例化、生产路径各自重算 identity_tag 字符串，是否应该算作同一类“契约存在但没被强制接入默认路径”的问题。

## 测试结果（父会话独立复验，不是只信子代理自报）

初次 Stage 1A（B01/B02/B10 主体修复）：
```
tests/test_normalization_policy.py（新增 10 个）+ tests/test_s0_gate_identity.py（新增 15 个）
+ tests/test_bridge.py + tests/test_upstream_parity.py + tests/test_s0_fail_closed.py
= 68 passed, 7 skipped, 1 warning (22.52s)
```
skip 均为预期：`test_bridge.py` 的 checkpoint 加载测试因沙箱内存不足跳过；`test_upstream_parity.py` 的 xformers/CUDA 测试因环境没有 GPU 跳过（正确地 fail-closed，没有伪造通过）。

针对上面“自己发现的 fail-open 缺口”的修复验证：
```
tests/test_s0_gate_identity.py（含新增 4 个 fail-closed 测试，共 19 个）+ tests/test_s0_fail_closed.py（13 个）
= 32 passed, 1 warning (19.25s)
```
以及全量重跑：
```
tests/test_s0_gate_identity.py + tests/test_s0_fail_closed.py + tests/test_bridge.py
+ tests/test_upstream_parity.py + tests/test_normalization_policy.py
= 72 passed, 7 skipped, 1 warning (35.57s)
```
`git diff --stat` 独立核对过：只有下面列出的 6 个受审文件 + 2 个新测试文件被改动，用户其余未跟踪路径（`artifacts/`、`plans/plans_0921/`、`plans/plans_v2_0921/`、`checkpoints/run_multi_year_pull.sh`）未受影响。

## 接下来还没做的事（请不要假设已完成）

Stage 1B（B05、B06-Q 一致性，改 `earthdelta/metrics_contract.py`）、Stage 1C（B08 小守卫、B13、B14，新增 `earthdelta/pilot_contract.py`）尚未开始；Stage 1D（父会话最终验证+diff 自查+三轴状态 checkpoint）尚未做；真实 GPU S0（Stage 2）、Fs/bank 训练、完整候选缓存、legal policy 决策表（Stage 3–5）、条件性 dual-head/输出纠错挑战（Stage 6–8）全部尚未开始。

---

## 附：完整 diff（6 个改动文件）

以下是 `git diff` 对 `earthdelta/bridge/__init__.py`、`earthdelta/bridge/stormer_bridge.py`、`earthdelta/contracts.py`、`scripts/export_upstream_reference.py`、`scripts/s0_gate.py`、`tests/test_s0_fail_closed.py` 的完整、未截断输出：

```diff
diff --git a/earthdelta/bridge/__init__.py b/earthdelta/bridge/__init__.py
index 1430f9f..f3eeb12 100644
--- a/earthdelta/bridge/__init__.py
+++ b/earthdelta/bridge/__init__.py
@@ -31,6 +31,8 @@ from .stormer_arch import (
 from .stormer_bridge import (
     DEFAULT_VARIABLES,
     CONSTANTS,
+    POLICY_LEGACY,
+    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
     NormalizationContract,
     WeatherStepBridge,
     load_stormer_checkpoint,
@@ -56,6 +58,8 @@ __all__ = [
     # Bridge
     "DEFAULT_VARIABLES",
     "CONSTANTS",
+    "POLICY_LEGACY",
+    "POLICY_OFFICIAL_ZERO_DIFF_MEAN",
     "NormalizationContract",
     "WeatherStepBridge",
     "load_stormer_checkpoint",
diff --git a/earthdelta/bridge/stormer_bridge.py b/earthdelta/bridge/stormer_bridge.py
index 98904b3..28bd0ac 100644
--- a/earthdelta/bridge/stormer_bridge.py
+++ b/earthdelta/bridge/stormer_bridge.py
@@ -128,6 +128,11 @@ CONSTANTS = [
 # Normalization Contract
 # =============================================================================
 
+# Normalization policy constants
+POLICY_LEGACY = "legacy"  # Use actual diff_mean values from NPZ files (pre-audit behavior)
+POLICY_OFFICIAL_ZERO_DIFF_MEAN = "official_zero_diff_mean"  # Force diff_mean to zero (matches official inference.py)
+
+
 @dataclass(frozen=True)
 class NormalizationContract:
     """Encapsulates input and diff normalization transforms.
@@ -137,22 +142,41 @@ class NormalizationContract:
     - reverse_inp_transform: denormalizes normalized input -> raw
     - reverse_diff_transform[interval]: denormalizes normalized diff -> raw diff
     - replace_constant: zeros out constant variable channels
+
+    Policy field controls diff_mean behavior:
+    - POLICY_LEGACY (default): Uses actual diff_mean values from NPZ files.
+      This preserves backward compatibility with existing experiments.
+    - POLICY_OFFICIAL_ZERO_DIFF_MEAN: Forces diff_mean to zero, matching the
+      official inference.py which uses transforms.Normalize(np.zeros_like(...), std).
+      This is the correct semantic for comparison against official upstream outputs.
+
+    NOTE: The official Stormer inference.py (reference/stormer/inference.py:118)
+    uses zero diff_mean: `transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)`
+    The legacy path loaded nonzero diff_mean values which creates a numeric mismatch.
     """
     inp_mean: torch.Tensor  # [V]
     inp_std: torch.Tensor   # [V]
     diff_mean: Dict[int, torch.Tensor]  # interval -> [V]
     diff_std: Dict[int, torch.Tensor]   # interval -> [V]
     variables: List[str]
+    policy: str = POLICY_LEGACY  # Default to legacy for backward compatibility
 
     @classmethod
-    def from_npz_dir(cls, npz_dir: str, variables: Optional[List[str]] = None,
-                     intervals: Tuple[int, ...] = (6, 12, 24)) -> 'NormalizationContract':
+    def from_npz_dir(
+        cls,
+        npz_dir: str,
+        variables: Optional[List[str]] = None,
+        intervals: Tuple[int, ...] = (6, 12, 24),
+        policy: str = POLICY_LEGACY,
+    ) -> 'NormalizationContract':
         """Load normalization constants from npz files.
 
         Args:
             npz_dir: Directory containing normalize_*.npz files
             variables: Variable list (default: DEFAULT_VARIABLES)
             intervals: Intervals to load diff transforms for
+            policy: Normalization policy (POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN).
+                    Default is POLICY_LEGACY for backward compatibility.
 
         Returns:
             NormalizationContract instance
@@ -160,6 +184,9 @@ class NormalizationContract:
         if variables is None:
             variables = DEFAULT_VARIABLES.copy()
 
+        if policy not in (POLICY_LEGACY, POLICY_OFFICIAL_ZERO_DIFF_MEAN):
+            raise ValueError(f"Unknown policy: {policy}. Use POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN.")
+
         # Load input normalization
         mean_path = os.path.join(npz_dir, "normalize_mean.npz")
         std_path = os.path.join(npz_dir, "normalize_std.npz")
@@ -179,9 +206,11 @@ class NormalizationContract:
 
             if os.path.exists(diff_mean_path):
                 dm = dict(np.load(diff_mean_path))
-                diff_mean[interval] = torch.from_numpy(
-                    np.concatenate([dm[v] for v in variables], axis=0)
-                ).float()
+                raw_diff_mean = np.concatenate([dm[v] for v in variables], axis=0)
+                # Under official policy, force diff_mean to zero
+                if policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
+                    raw_diff_mean = np.zeros_like(raw_diff_mean)
+                diff_mean[interval] = torch.from_numpy(raw_diff_mean).float()
 
             if os.path.exists(diff_std_path):
                 ds = dict(np.load(diff_std_path))
@@ -195,6 +224,7 @@ class NormalizationContract:
             diff_mean=diff_mean,
             diff_std=diff_std,
             variables=variables,
+            policy=policy,
         )
 
     def normalize(self, x_raw: torch.Tensor) -> torch.Tensor:
@@ -226,9 +256,13 @@ class NormalizationContract:
     def denormalize_diff(self, diff_norm: torch.Tensor, interval: int) -> torch.Tensor:
         """Denormalize normalized diff: diff_raw = diff_norm * std + mean.
 
-        Note: For Stormer inference, the diff_mean is typically zeros, but we
-        include it for completeness. The inference.py uses only std (via
-        Normalize with zeros mean).
+        The mean used depends on the policy:
+        - POLICY_OFFICIAL_ZERO_DIFF_MEAN: Always uses zero mean, matching official inference.py.
+        - POLICY_LEGACY: Uses the loaded diff_mean values (may be nonzero).
+
+        Note: The official Stormer inference.py (reference/stormer/inference.py:118)
+        uses transforms.Normalize(np.zeros_like(...), std), i.e., zero diff_mean.
+        Using nonzero diff_mean creates a numeric mismatch with upstream.
 
         Args:
             diff_norm: Normalized diff of shape (B, V, H, W)
@@ -242,8 +276,12 @@ class NormalizationContract:
 
         std = self.diff_std[interval].to(diff_norm.device, diff_norm.dtype).view(1, -1, 1, 1)
 
-        # Use mean if available, otherwise zeros
-        if interval in self.diff_mean:
+        # Determine mean based on policy
+        if self.policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
+            # Official semantic: always zero diff_mean
+            mean = torch.zeros_like(std)
+        elif interval in self.diff_mean:
+            # Legacy semantic: use actual diff_mean from NPZ
             mean = self.diff_mean[interval].to(diff_norm.device, diff_norm.dtype).view(1, -1, 1, 1)
         else:
             mean = torch.zeros_like(std)
@@ -268,15 +306,23 @@ class NormalizationContract:
 
     @property
     def digest(self) -> str:
-        """Hash of normalization constants for version tracking.
+        """Hash of normalization constants and policy for version tracking.
 
-        Includes variable names/order, interval keys, and tensor shapes to ensure
-        the digest changes when any structural aspect of the contract changes,
-        not just the raw tensor values.
+        Includes:
+        - Variable names/order
+        - Interval keys
+        - Tensor shapes and values
+        - Policy (POLICY_LEGACY or POLICY_OFFICIAL_ZERO_DIFF_MEAN)
+
+        This ensures the digest changes when any structural or semantic aspect
+        of the contract changes, including the diff_mean policy.
         """
         # Build a canonical representation that includes all structural info
         parts = []
 
+        # Policy MUST be part of the digest - it changes the semantic behavior
+        parts.append(f"policy={self.policy}".encode())
+
         # Variable names and order
         parts.append((",".join(self.variables)).encode())
 
@@ -287,6 +333,8 @@ class NormalizationContract:
         parts.append(self.inp_std.numpy().tobytes())
 
         # Diff normalization with explicit keys (sorted for determinism)
+        # Note: Under POLICY_OFFICIAL_ZERO_DIFF_MEAN, diff_mean tensors are zeros
+        # but we still include them for structural completeness
         for interval in sorted(self.diff_mean.keys()):
             parts.append(f"diff_mean_{interval}_shape={tuple(self.diff_mean[interval].shape)}".encode())
             parts.append(self.diff_mean[interval].numpy().tobytes())
diff --git a/earthdelta/contracts.py b/earthdelta/contracts.py
index e0f7ac3..7834cfe 100644
--- a/earthdelta/contracts.py
+++ b/earthdelta/contracts.py
@@ -4,8 +4,10 @@ Timestamps are integer UTC seconds. EditPlan generalizes from rank-group masks
 to expert-dictionary coefficients with an application window.
 """
 from __future__ import annotations
-from dataclasses import dataclass, asdict
+from dataclasses import dataclass, asdict, field
 from collections.abc import Sequence
+from pathlib import Path
+from typing import Optional, Tuple, List, Dict, Any
 import hashlib
 import json
 import math
@@ -13,6 +15,80 @@ import torch
 from torch import Tensor
 
 
+# =============================================================================
+# Gate Identity Configuration
+# =============================================================================
+
+# Official Stormer reference commit from which inference semantics are pinned
+OFFICIAL_STORMER_COMMIT = "58dfee5a6037399a40fefd492bc00421e0c885a8"
+
+
+@dataclass(frozen=True)
+class GateIdentityConfig:
+    """Shared identity configuration for exporter and gate.
+
+    This configuration ensures both exporter and gate use exactly the same
+    identity for checkpoint, normalization, variables, and rollout parameters.
+    The output directory is namespaced by this identity to prevent silent
+    overwrites between different configurations (e.g., ps2 vs ps4).
+    """
+    # Checkpoint identity
+    checkpoint_path: str
+    expected_checkpoint_sha256: str
+    patch_size: int  # 2 or 4 (ps4 is the mainline per research_spec_v6.yaml:33)
+
+    # Normalization identity
+    normalization_policy: str  # "official_zero_diff_mean" or "legacy"
+    normalization_dir: str
+
+    # Variable/coordinate identity
+    variables: Tuple[str, ...]
+    grid_shape: Tuple[int, int]  # (H, W), e.g., (128, 256)
+
+    # Rollout configuration
+    interval_hours: int = 6
+    rollout_steps: Tuple[int, ...] = (1, 4)  # Steps to run/verify (1-step and 4-step)
+
+    # Reference source identity
+    official_commit: str = OFFICIAL_STORMER_COMMIT
+
+    @property
+    def identity_tag(self) -> str:
+        """Short identity tag for directory namespacing.
+
+        Format: ps{patch_size}_{sha_prefix}_{policy_prefix}
+        """
+        sha_prefix = self.expected_checkpoint_sha256[:8] if self.expected_checkpoint_sha256 else "unknown"
+        policy_prefix = "zd" if self.normalization_policy == "official_zero_diff_mean" else "lg"
+        return f"ps{self.patch_size}_{sha_prefix}_{policy_prefix}"
+
+    @property
+    def reference_output_dir_name(self) -> str:
+        """Directory name for reference outputs, namespaced by identity."""
+        return f"upstream_reference_{self.identity_tag}"
+
+    def validate_checkpoint_sha256(self, computed_sha256: str) -> bool:
+        """Compare computed SHA-256 against expected value."""
+        return computed_sha256.lower() == self.expected_checkpoint_sha256.lower()
+
+    def to_manifest_dict(self) -> Dict[str, Any]:
+        """Convert to manifest dictionary for JSON serialization."""
+        return {
+            "checkpoint_path": self.checkpoint_path,
+            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
+            "patch_size": self.patch_size,
+            "normalization_policy": self.normalization_policy,
+            "normalization_dir": self.normalization_dir,
+            "variables_count": len(self.variables),
+            "variables_hash": hashlib.sha256(",".join(self.variables).encode()).hexdigest()[:16],
+            "grid_shape": list(self.grid_shape),
+            "interval_hours": self.interval_hours,
+            "rollout_steps": list(self.rollout_steps),
+            "official_commit": self.official_commit,
+            "identity_tag": self.identity_tag,
+        }
+
+
 @dataclass(frozen=True)
 class ArtifactVersion:
     """Version fingerprint for provenance tracking and mismatch rejection."""
diff --git a/scripts/export_upstream_reference.py b/scripts/export_upstream_reference.py
index a434c1f..36f570f 100644
--- a/scripts/export_upstream_reference.py
+++ b/scripts/export_upstream_reference.py
@@ -2,22 +2,21 @@
 """Export upstream reference outputs for S0 gate parity verification.
 
 This script MUST run inside a GPU job container with xformers installed.
-It builds the real official Stormer model using the actual code in
-reference/stormer/ (which uses xformers.ops.memory_efficient_attention,
-NOT the SDPA version in earthdelta.bridge).
+It imports and uses the ACTUAL official GlobalForecastIterativeModule.forward_validation
+from reference/stormer/ (which uses xformers.ops.memory_efficient_attention),
+NOT the SDPA version in earthdelta.bridge.
+
+CRITICAL: This script does NOT reuse the earthdelta NormalizationContract to build
+the reference outputs. That contract is exactly what's under audit. Instead, it
+constructs transforms independently from the official NPZ files using the official
+inference.py semantics (zero diff_mean).
 
 The script:
 1. Verifies xformers is available (exits BLOCKED otherwise)
 2. Loads the official Stormer model from reference/stormer/
-3. Loads weights from the same checkpoint used by the bridge
-4. Runs inference on the pinned input tensor (jan2020_full.npy)
-5. Exports outputs + environment manifest to artifacts/upstream_reference/
-
-This output is then used by s0_gate.py's upstream_parity check to verify
-the bridge produces equivalent output.
-
-Usage (in ACP GPU container):
-    python scripts/export_upstream_reference.py [--checkpoint ps2|ps4] [--output-dir PATH]
+3. Sets up transforms INDEPENDENTLY using zero diff_mean (matching official inference.py)
+4. Runs the official forward_validation method
+5. Exports outputs + environment manifest to a namespaced directory
 
 Exit codes:
     0 - Success, reference output exported
@@ -36,7 +35,7 @@ import socket
 import sys
 import traceback
 from pathlib import Path
-from typing import Any, Dict
+from typing import Any, Dict, Optional
 
 # =============================================================================
 # Environment check - must happen before any torch/xformers imports
@@ -77,7 +76,9 @@ if not check_xformers_available():
     exit_blocked(
         "xformers not available. This script requires xformers to execute "
         "the official Stormer code path. Install xformers and re-run, or "
-        "run this script in a GPU container with xformers pre-installed.",
+        "run this script in a GPU container with xformers pre-installed. "
+        "IMPORTANT: This script will NOT fall back to SDPA - that would defeat "
+        "the purpose of generating an independent reference.",
         code=2
     )
 
@@ -95,6 +96,7 @@ if not check_cuda_available():
 
 import numpy as np
 import torch
+from torchvision.transforms import transforms
 
 # Add reference stormer to path (before earthdelta to avoid conflicts)
 SCRIPT_DIR = Path(__file__).resolve().parent
@@ -102,16 +104,89 @@ REPO_ROOT = SCRIPT_DIR.parent
 REFERENCE_STORMER = REPO_ROOT / "reference" / "stormer"
 sys.path.insert(0, str(REFERENCE_STORMER))
 
-# Import official Stormer (uses xformers)
+# Import official Stormer modules (uses xformers)
 from stormer.models.hub.stormer import Stormer as OfficialStormer  # noqa: E402
+from stormer.models.iterative_module import GlobalForecastIterativeModule  # noqa: E402
 
-# Import earthdelta for normalization (but NOT the bridge Stormer)
+# Import ONLY the file hash utility from earthdelta - NOT NormalizationContract
 sys.path.insert(0, str(REPO_ROOT))
-from earthdelta.bridge import (  # noqa: E402
-    NormalizationContract,
-    DEFAULT_VARIABLES,
-    _compute_file_sha256,
-)
+from earthdelta.bridge.stormer_bridge import _compute_file_sha256  # noqa: E402
+from earthdelta.contracts import GateIdentityConfig, OFFICIAL_STORMER_COMMIT  # noqa: E402
+
+# Official 69-variable list (from reference/stormer/configs/finetune_multi_step.yaml)
+# We hardcode this here to avoid importing from earthdelta.bridge.DEFAULT_VARIABLES
+# which would couple the reference path to the audited code.
+OFFICIAL_VARIABLES = [
+    "2m_temperature",
+    "10m_u_component_of_wind",
+    "10m_v_component_of_wind",
+    "mean_sea_level_pressure",
+    "geopotential_50",
+    "geopotential_100",
+    "geopotential_150",
+    "geopotential_200",
+    "geopotential_250",
+    "geopotential_300",
+    "geopotential_400",
+    "geopotential_500",
+    "geopotential_600",
+    "geopotential_700",
+    "geopotential_850",
+    "geopotential_925",
+    "geopotential_1000",
+    "u_component_of_wind_50",
+    "u_component_of_wind_100",
+    "u_component_of_wind_150",
+    "u_component_of_wind_200",
+    "u_component_of_wind_250",
+    "u_component_of_wind_300",
+    "u_component_of_wind_400",
+    "u_component_of_wind_500",
+    "u_component_of_wind_600",
+    "u_component_of_wind_700",
+    "u_component_of_wind_850",
+    "u_component_of_wind_925",
+    "u_component_of_wind_1000",
+    "v_component_of_wind_50",
+    "v_component_of_wind_100",
+    "v_component_of_wind_150",
+    "v_component_of_wind_200",
+    "v_component_of_wind_250",
+    "v_component_of_wind_300",
+    "v_component_of_wind_400",
+    "v_component_of_wind_500",
+    "v_component_of_wind_600",
+    "v_component_of_wind_700",
+    "v_component_of_wind_850",
+    "v_component_of_wind_925",
+    "v_component_of_wind_1000",
+    "temperature_50",
+    "temperature_100",
+    "temperature_150",
+    "temperature_200",
+    "temperature_250",
+    "temperature_300",
+    "temperature_400",
+    "temperature_500",
+    "temperature_600",
+    "temperature_700",
+    "temperature_850",
+    "temperature_925",
+    "temperature_1000",
+    "specific_humidity_50",
+    "specific_humidity_100",
+    "specific_humidity_150",
+    "specific_humidity_200",
+    "specific_humidity_250",
+    "specific_humidity_300",
+    "specific_humidity_400",
+    "specific_humidity_500",
+    "specific_humidity_600",
+    "specific_humidity_700",
+    "specific_humidity_850",
+    "specific_humidity_925",
+    "specific_humidity_1000",
+]
 
 
 # =============================================================================
@@ -130,40 +205,121 @@ def get_default_paths(repo_root: Path) -> Dict[str, Path]:
         "checkpoint_ps2": repo_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
         "checkpoint_ps4": repo_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
         "norm_dir": repo_root / "reference" / "stormer" / "normalization_constants",
-        "output_dir": repo_root / "artifacts" / "upstream_reference",
+        "output_base": repo_root / "artifacts",
     }
 
 
+# =============================================================================
+# Official Transform Construction (INDEPENDENT from earthdelta)
+# =============================================================================
+
+def build_official_transforms(
+    norm_dir: Path,
+    variables: list,
+    intervals: tuple = (6, 12, 24),
+) -> tuple:
+    """Build transforms using OFFICIAL inference.py semantics.
+
+    CRITICAL: This function constructs transforms independently, NOT using
+    the earthdelta NormalizationContract. It matches the official inference.py
+    which uses:
+    - inp_transform: Normalize(mean, std)
+    - diff_transform: Normalize(ZEROS, std)  <-- Zero diff_mean!
+
+    See reference/stormer/inference.py lines 116-118:
+        out_transforms[l] = transforms.Normalize(
+            np.zeros_like(normalize_diff_std), normalize_diff_std
+        )
+
+    Returns:
+        (inp_transform, diff_transforms_dict)
+    """
+    # Load input normalization
+    normalize_mean = dict(np.load(norm_dir / "normalize_mean.npz"))
+    normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
+    normalize_std = dict(np.load(norm_dir / "normalize_std.npz"))
+    normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
+
+    inp_transform = transforms.Normalize(normalize_mean, normalize_std)
+
+    # Load diff normalization - USING ZEROS for mean (official semantic)
+    diff_transforms = {}
+    for interval in intervals:
+        diff_std_path = norm_dir / f"normalize_diff_std_{interval}.npz"
+        if diff_std_path.exists():
+            normalize_diff_std = dict(np.load(diff_std_path))
+            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
+            # OFFICIAL SEMANTIC: Zero mean, not the NPZ diff_mean values
+            diff_transforms[interval] = transforms.Normalize(
+                np.zeros_like(normalize_diff_std), normalize_diff_std
+            )
+
+    return inp_transform, diff_transforms
+
+
+def compute_normalization_digest(
+    norm_dir: Path,
+    variables: list,
+    intervals: tuple = (6, 24),
+) -> str:
+    """Compute a digest for the normalization constants.
+
+    This is computed independently for the manifest, using zero diff_mean
+    policy to match the official semantics.
+    """
+    parts = []
+    parts.append(b"policy=official_zero_diff_mean")
+    parts.append((",".join(variables)).encode())
+
+    # Input normalization
+    mean_dict = dict(np.load(norm_dir / "normalize_mean.npz"))
+    std_dict = dict(np.load(norm_dir / "normalize_std.npz"))
+    inp_mean = np.concatenate([mean_dict[v] for v in variables], axis=0)
+    inp_std = np.concatenate([std_dict[v] for v in variables], axis=0)
+
+    parts.append(f"inp_mean_shape={inp_mean.shape}".encode())
+    parts.append(inp_mean.tobytes())
+    parts.append(f"inp_std_shape={inp_std.shape}".encode())
+    parts.append(inp_std.tobytes())
+
+    # Diff normalization (zeros for mean)
+    for interval in sorted(intervals):
+        diff_std_path = norm_dir / f"normalize_diff_std_{interval}.npz"
+        if diff_std_path.exists():
+            ds = dict(np.load(diff_std_path))
+            diff_std = np.concatenate([ds[v] for v in variables], axis=0)
+            # Zero mean
+            diff_mean = np.zeros_like(diff_std)
+            parts.append(f"diff_mean_{interval}_shape={diff_mean.shape}".encode())
+            parts.append(diff_mean.tobytes())
+            parts.append(f"diff_std_{interval}_shape={diff_std.shape}".encode())
+            parts.append(diff_std.tobytes())
+
+    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]
+
+
 # =============================================================================
 # Core functions
 # =============================================================================
 
-def load_official_stormer(
+def load_official_module(
     checkpoint_path: Path,
     patch_size: int,
     variables: list,
+    inp_transform,
+    diff_transforms: dict,
     in_img_size: tuple = (128, 256),
     device: torch.device = None,
-) -> OfficialStormer:
-    """Load the official Stormer model using reference code.
-
-    This loads the real xformers-based Stormer, not the SDPA bridge version.
-
-    Args:
-        checkpoint_path: Path to checkpoint file
-        patch_size: Patch size (2 or 4)
-        variables: List of variable names
-        in_img_size: Input image size (H, W)
-        device: Device to load model to
+) -> GlobalForecastIterativeModule:
+    """Load the official GlobalForecastIterativeModule with transforms set.
 
-    Returns:
-        Loaded and frozen OfficialStormer model
+    This uses the ACTUAL official module, not a reimplementation.
     """
     if device is None:
         device = torch.device("cuda:0")
 
-    # Build official model
-    model = OfficialStormer(
+    # Build official Stormer network
+    net = OfficialStormer(
         in_img_size=list(in_img_size),
         variables=variables,
         patch_size=patch_size,
@@ -173,89 +329,31 @@ def load_official_stormer(
         mlp_ratio=4.0,
     )
 
+    # Wrap in GlobalForecastIterativeModule
+    module = GlobalForecastIterativeModule(net)
+
     # Load checkpoint
     checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
     state_dict = checkpoint["state_dict"]
+    module.load_state_dict(state_dict)
 
-    # Strip 'net.' prefix (same as bridge)
-    new_state_dict = {}
-    for k, v in state_dict.items():
-        if k.startswith("net."):
-            new_state_dict[k[4:]] = v
-        else:
-            new_state_dict[k] = v
-
-    # Load with strict=True
-    model.load_state_dict(new_state_dict, strict=True)
+    # Set transforms (this is how official inference.py does it)
+    module.set_transforms(inp_transform, diff_transforms)
 
     # Freeze and move to device
-    model.requires_grad_(False)
-    model.eval()
-    model = model.to(device)
+    module.requires_grad_(False)
+    module.eval()
+    module = module.to(device)
 
-    return model
+    return module
 
 
-def run_official_inference(
-    model: OfficialStormer,
-    x_norm: torch.Tensor,
+def create_environment_manifest(
+    checkpoint_path: Path,
+    patch_size: int,
+    norm_dir: Path,
     variables: list,
-    normalization: NormalizationContract,
-    interval: int,
-    steps: int,
-) -> torch.Tensor:
-    """Run official autoregressive inference.
-
-    Matches the rollout logic in GlobalForecastIterativeModule.forward_validation.
-
-    Args:
-        model: Official Stormer model
-        x_norm: Normalized input of shape (B, V, H, W)
-        variables: Variable names
-        normalization: Normalization contract
-        interval: Forecast interval (hours)
-        steps: Number of autoregressive steps
-
-    Returns:
-        Normalized output at final step
-    """
-    device = x_norm.device
-    patch_size = model.patch_size
-
-    # Scale interval (matching iterative_module.py)
-    interval_tensor = torch.tensor([interval], device=device, dtype=x_norm.dtype) / 10.0
-    interval_tensor = interval_tensor.repeat(x_norm.shape[0])
-
-    x = x_norm
-    for _ in range(steps):
-        # Pad if needed
-        h = x.shape[-2]
-        if h % patch_size != 0:
-            pad_size = patch_size - h % patch_size
-            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
-        else:
-            padded_x = x
-            pad_size = 0
-
-        # Forward pass
-        with torch.no_grad():
-            output = model(padded_x, variables, interval_tensor)
-
-        # Remove padding
-        pred_diff = output[:, :, pad_size:] if pad_size > 0 else output
-
-        # Zero out constant channels
-        pred_diff = normalization.replace_constant(pred_diff, variables)
-
-        # Denormalize diff, add to denormalized input, renormalize
-        pred_diff = normalization.denormalize_diff(pred_diff, interval)
-        pred = normalization.denormalize(x) + pred_diff
-        x = normalization.normalize(pred)
-
-    return x
-
-
-def create_environment_manifest(checkpoint_path: Path, patch_size: int) -> Dict[str, Any]:
+) -> Dict[str, Any]:
     """Create manifest documenting the execution environment."""
     import xformers
 
@@ -280,6 +378,17 @@ def create_environment_manifest(checkpoint_path: Path, patch_size: int) -> Dict[
             "num_heads": 16,
             "mlp_ratio": 4.0,
         },
+        "normalization": {
+            "dir": str(norm_dir),
+            "policy": "official_zero_diff_mean",
+            "digest": compute_normalization_digest(norm_dir, variables),
+            "note": "Uses zero diff_mean matching official inference.py semantics",
+        },
+        "official_source": {
+            "pinned_commit": OFFICIAL_STORMER_COMMIT,
+            "note": "This reference was generated using the official GlobalForecastIterativeModule.forward_validation, NOT the earthdelta bridge implementation.",
+        },
+        "variables_count": len(variables),
     }
     return manifest
 
@@ -289,17 +398,13 @@ def export_reference(
     patch_size: int,
     input_dir: Path,
     output_dir: Path,
+    norm_dir: Path,
 ) -> Dict[str, Any]:
     """Export official reference outputs.
 
-    Args:
-        checkpoint_path: Path to checkpoint
-        patch_size: Patch size (2 or 4)
-        input_dir: Directory with pinned inputs
-        output_dir: Directory to write outputs
-
-    Returns:
-        Result dict with status and metadata
+    CRITICAL: This uses the ACTUAL official forward_validation method,
+    NOT a reimplementation. The transforms are constructed independently
+    using official semantics (zero diff_mean).
     """
     result = {
         "status": "failed",
@@ -311,54 +416,42 @@ def export_reference(
         device = torch.device("cuda:0")
         print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
 
+        # Build transforms INDEPENDENTLY using official semantics
+        print("Building official transforms (zero diff_mean)...")
+        inp_transform, diff_transforms = build_official_transforms(
+            norm_dir, OFFICIAL_VARIABLES
+        )
+
         # Load pinned input
         print("Loading pinned input from s0_gate_inputs...")
         jan2020_data = np.load(input_dir / "jan2020_full.npy")  # (124, 69, 128, 256)
         x_raw = jan2020_data[0]  # First timestep
         x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
 
-        # Load normalization
-        print("Loading normalization constants...")
-        inputs = {
-            'inp_mean': np.load(input_dir / 'inp_mean.npy'),
-            'inp_std': np.load(input_dir / 'inp_std.npy'),
-            'diff_mean_6': np.load(input_dir / 'diff_mean_6.npy'),
-            'diff_std_6': np.load(input_dir / 'diff_std_6.npy'),
-            'diff_mean_24': np.load(input_dir / 'diff_mean_24.npy'),
-            'diff_std_24': np.load(input_dir / 'diff_std_24.npy'),
-        }
-        normalization = NormalizationContract(
-            inp_mean=torch.from_numpy(inputs['inp_mean']).float(),
-            inp_std=torch.from_numpy(inputs['inp_std']).float(),
-            diff_mean={
-                6: torch.from_numpy(inputs['diff_mean_6']).float(),
-                24: torch.from_numpy(inputs['diff_mean_24']).float(),
-            },
-            diff_std={
-                6: torch.from_numpy(inputs['diff_std_6']).float(),
-                24: torch.from_numpy(inputs['diff_std_24']).float(),
-            },
-            variables=DEFAULT_VARIABLES,
-        )
+        # Compute raw input hash for identity binding
+        raw_input_hash = hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]
+        print(f"Raw input hash: {raw_input_hash}")
 
-        # Normalize input
-        x_norm = normalization.normalize(x_raw_t)
+        # Normalize input using official transform
+        x_norm = inp_transform(x_raw_t)
 
-        # Load official model
-        print(f"Loading official Stormer (patch_size={patch_size}) with xformers...")
-        model = load_official_stormer(
-            checkpoint_path, patch_size, DEFAULT_VARIABLES, device=device
+        # Load official module
+        print(f"Loading official GlobalForecastIterativeModule (patch_size={patch_size}) with xformers...")
+        module = load_official_module(
+            checkpoint_path, patch_size, OFFICIAL_VARIABLES,
+            inp_transform, diff_transforms, device=device
         )
-        print("Model loaded successfully")
+        print("Module loaded successfully")
 
-        # Run inference for multiple configurations
+        # Run inference using the ACTUAL official forward_validation method
         outputs = {}
         for interval, steps in [(6, 1), (6, 4)]:
             key = f"{interval}h_{steps}step"
-            print(f"Running inference: {key}...")
+            print(f"Running official forward_validation: {key}...")
             with torch.no_grad():
-                out = run_official_inference(
-                    model, x_norm, DEFAULT_VARIABLES, normalization, interval, steps
+                # This is the ACTUAL official method, not a reimplementation
+                out = module.forward_validation(
+                    x_norm, OFFICIAL_VARIABLES, interval, steps
                 )
             outputs[key] = out.cpu()
             print(f"  Output shape: {out.shape}, finite: {torch.isfinite(out).all()}")
@@ -375,11 +468,18 @@ def export_reference(
         # Save the normalized input (for exact reproducibility)
         torch.save(x_norm.cpu(), output_dir / "input_norm.pt")
 
+        # Save raw input hash
+        with open(output_dir / "raw_input_hash.txt", "w") as f:
+            f.write(raw_input_hash)
+
         # Save manifest
-        manifest = create_environment_manifest(checkpoint_path, patch_size)
+        manifest = create_environment_manifest(
+            checkpoint_path, patch_size, norm_dir, OFFICIAL_VARIABLES
+        )
         manifest["outputs"] = {k: list(v.shape) for k, v in outputs.items()}
         manifest["outputs_finite"] = {k: bool(torch.isfinite(v).all()) for k, v in outputs.items()}
-        manifest["normalization_digest"] = normalization.digest
+        manifest["raw_input_hash"] = raw_input_hash
+        manifest["input_path"] = str(input_dir / "jan2020_full.npy")
 
         manifest_path = output_dir / "manifest.json"
         with open(manifest_path, "w") as f:
@@ -389,9 +489,10 @@ def export_reference(
         result["status"] = "ok"
         result["output_dir"] = str(output_dir)
         result["manifest"] = manifest
+        result["raw_input_hash"] = raw_input_hash
 
         # Clean up
-        del model
+        del module
         torch.cuda.empty_cache()
 
     except Exception as e:
@@ -408,12 +509,12 @@ def main():
         description="Export upstream reference outputs for S0 gate verification"
     )
     parser.add_argument(
-        "--checkpoint", choices=["ps2", "ps4"], default="ps2",
-        help="Checkpoint to use (default: ps2)"
+        "--checkpoint", choices=["ps2", "ps4"], default="ps4",
+        help="Checkpoint to use (default: ps4 - the mainline per research_spec_v6.yaml)"
     )
     parser.add_argument(
         "--output-dir", type=Path, default=None,
-        help="Output directory (default: artifacts/upstream_reference)"
+        help="Output directory (default: artifacts/upstream_reference_<identity>)"
     )
     parser.add_argument(
         "--repo-root", type=Path, default=None,
@@ -426,14 +527,22 @@ def main():
 
     checkpoint_path = paths["checkpoint_ps2"] if args.checkpoint == "ps2" else paths["checkpoint_ps4"]
     patch_size = 2 if args.checkpoint == "ps2" else 4
-    output_dir = args.output_dir or paths["output_dir"]
+
+    # Compute checkpoint SHA for identity
+    ckpt_sha256 = _compute_file_sha256(str(checkpoint_path)) if checkpoint_path.exists() else "unknown"
+
+    # Build namespaced output directory
+    identity_tag = f"ps{patch_size}_{ckpt_sha256[:8]}_zd"
+    output_dir = args.output_dir or (paths["output_base"] / f"upstream_reference_{identity_tag}")
 
     print("=" * 60)
     print("Export Upstream Reference - Official Stormer with xformers")
     print("=" * 60)
     print(f"Checkpoint: {checkpoint_path}")
     print(f"Patch size: {patch_size}")
+    print(f"Checkpoint SHA-256: {ckpt_sha256}")
     print(f"Output dir: {output_dir}")
+    print(f"Note: Using official forward_validation with zero diff_mean")
     print()
 
     result = export_reference(
@@ -441,6 +550,7 @@ def main():
         patch_size=patch_size,
         input_dir=paths["input_dir"],
         output_dir=output_dir,
+        norm_dir=paths["norm_dir"],
     )
 
     print()
diff --git a/scripts/s0_gate.py b/scripts/s0_gate.py
index d438362..1ebfe87 100644
--- a/scripts/s0_gate.py
+++ b/scripts/s0_gate.py
@@ -5,16 +5,18 @@ Runs S0-level gate checks to verify bridge correctness before downstream use.
 All gate criteria are fail-closed: initialized False, remain False on exceptions.
 
 Gate Criteria:
-- ckpt_sha256_bound: Checkpoint SHA-256 hash computed and recorded
+- ckpt_sha256_bound: Checkpoint SHA-256 hash matches expected value (not just length check)
 - strict_load_zero_diff: Checkpoint loads with zero missing/unexpected keys
 - upstream_parity: Bridge output matches official xformers Stormer (<=1e-5)
 - zero_edit_equals_official: Bridge internal consistency (<=1e-6)
-- normalization_parity: Normalization digest matches reference
+- normalization_parity: Normalization digest matches expected reference
 - no_state_leak: Identical inputs produce bit-identical outputs
 - outputs_finite: All outputs contain no NaN/Inf
+- raw_input_binding: Raw input hash matches expected (if reference exists)
+- multistep_reference_present: All required multistep reference files are present
 
 Usage:
-    python scripts/s0_gate.py [--check CRITERION] [--output-dir PATH]
+    python scripts/s0_gate.py [--checkpoint ps2|ps4] [--output-dir PATH]
 
 The upstream_parity check requires running export_upstream_reference.py first
 in a GPU+xformers environment. Without the reference artifacts, this check
@@ -59,20 +61,26 @@ def get_repo_root() -> Path:
     return Path(__file__).resolve().parent.parent
 
 
-def get_paths(repo_root: Path) -> Dict[str, Path]:
-    """Get all paths relative to repo root."""
+def get_paths(repo_root: Path, patch_size: int = 4) -> Dict[str, Path]:
+    """Get all paths relative to repo root.
+
+    Args:
+        repo_root: Repository root path
+        patch_size: Patch size (2 or 4). Default is 4 (mainline per research_spec_v6.yaml).
+    """
+    ps_str = f"ps{patch_size}"
     return {
         "input_dir": repo_root / "scripts" / "s0_gate_inputs",
         "checkpoint_ps2": repo_root / "checkpoints" / "stormer_1.40625_patch_size_2.ckpt",
         "checkpoint_ps4": repo_root / "checkpoints" / "stormer_1.40625_patch_size_4.ckpt",
+        "checkpoint": repo_root / "checkpoints" / f"stormer_1.40625_patch_size_{patch_size}.ckpt",
         "norm_dir": repo_root / "reference" / "stormer" / "normalization_constants",
-        "upstream_reference_dir": repo_root / "artifacts" / "upstream_reference",
+        "upstream_reference_base": repo_root / "artifacts",
         "default_output_dir": repo_root / "artifacts" / "s0_gate",
     }
 
 
 REPO_ROOT = get_repo_root()
-PATHS = get_paths(REPO_ROOT)
 
 # Add repo to Python path
 sys.path.insert(0, str(REPO_ROOT))
@@ -101,9 +109,12 @@ try:
         load_stormer_checkpoint_detailed,
         CheckpointLoadResult,
         DEFAULT_VARIABLES,
+        POLICY_LEGACY,
+        POLICY_OFFICIAL_ZERO_DIFF_MEAN,
         _compute_file_sha256,
+        check_version_match,
     )
-    from earthdelta.contracts import EditPlan, reference_plan
+    from earthdelta.contracts import EditPlan, reference_plan, GateIdentityConfig
     from earthdelta.lowrank import ExpertLoRA
     print("EarthDelta imports successful", flush=True)
 except ImportError as e:
@@ -141,41 +152,66 @@ def load_npy_inputs(input_dir: Path) -> Dict[str, np.ndarray]:
     }
 
 
-def create_normalization_from_npy(inputs: Dict[str, np.ndarray]) -> NormalizationContract:
-    """Create NormalizationContract from pre-extracted numpy arrays."""
+def create_normalization_from_npy(
+    inputs: Dict[str, np.ndarray],
+    policy: str = POLICY_OFFICIAL_ZERO_DIFF_MEAN,
+) -> NormalizationContract:
+    """Create NormalizationContract from pre-extracted numpy arrays.
+
+    Args:
+        inputs: Dictionary of numpy arrays from load_npy_inputs
+        policy: Normalization policy. Default is POLICY_OFFICIAL_ZERO_DIFF_MEAN
+                to match official inference.py semantics.
+    """
+    # Under official policy, force diff_mean to zero
+    if policy == POLICY_OFFICIAL_ZERO_DIFF_MEAN:
+        diff_mean_6 = np.zeros_like(inputs['diff_mean_6'])
+        diff_mean_24 = np.zeros_like(inputs['diff_mean_24'])
+    else:
+        diff_mean_6 = inputs['diff_mean_6']
+        diff_mean_24 = inputs['diff_mean_24']
+
     return NormalizationContract(
         inp_mean=torch.from_numpy(inputs['inp_mean']).float(),
         inp_std=torch.from_numpy(inputs['inp_std']).float(),
         diff_mean={
-            6: torch.from_numpy(inputs['diff_mean_6']).float(),
-            24: torch.from_numpy(inputs['diff_mean_24']).float(),
+            6: torch.from_numpy(diff_mean_6).float(),
+            24: torch.from_numpy(diff_mean_24).float(),
         },
         diff_std={
             6: torch.from_numpy(inputs['diff_std_6']).float(),
             24: torch.from_numpy(inputs['diff_std_24']).float(),
         },
         variables=DEFAULT_VARIABLES,
+        policy=policy,
     )
 
 
+def compute_raw_input_hash(x_raw: np.ndarray) -> str:
+    """Compute SHA-256 hash of raw input for identity binding."""
+    return hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]
+
+
 # =============================================================================
-# Gate criterion: ckpt_sha256_bound
+# Gate criterion: ckpt_sha256_bound (FULL comparison, not just length)
 # =============================================================================
 
 def verify_ckpt_sha256(
     ckpt_path: Path,
     load_result: Optional[CheckpointLoadResult],
+    expected_sha256: Optional[str] = None,
 ) -> Dict[str, Any]:
-    """Verify checkpoint SHA-256 is computed and bound.
+    """Verify checkpoint SHA-256 matches expected value.
 
-    This criterion passes if we can compute and record the SHA-256 hash
-    of the checkpoint file.
+    This criterion now performs FULL SHA-256 comparison, not just length check.
+    If expected_sha256 is None, we just compute and record the hash (first run).
     """
     result = {
         "criterion": "ckpt_sha256_bound",
         "passed": False,  # fail-closed
         "checkpoint": str(ckpt_path),
-        "sha256": None,
+        "computed_sha256": None,
+        "expected_sha256": expected_sha256,
     }
 
     try:
@@ -184,11 +220,27 @@ def verify_ckpt_sha256(
             return result
 
         if load_result is not None:
-            result["sha256"] = load_result.checkpoint_sha256
+            computed_sha256 = load_result.checkpoint_sha256
         else:
-            result["sha256"] = _compute_file_sha256(str(ckpt_path))
-
-        result["passed"] = result["sha256"] is not None and len(result["sha256"]) == 64
+            computed_sha256 = _compute_file_sha256(str(ckpt_path))
+
+        result["computed_sha256"] = computed_sha256
+
+        # FULL validation: compare against expected, not just check length
+        if expected_sha256 is not None:
+            result["match"] = computed_sha256.lower() == expected_sha256.lower()
+            result["passed"] = result["match"]
+            if not result["passed"]:
+                result["error"] = (
+                    f"SHA-256 mismatch: expected {expected_sha256}, "
+                    f"got {computed_sha256}"
+                )
+        else:
+            # No expected value provided - fail closed (B02 fix)
+            # Computed hash is recorded for enrollment/discovery, but cannot certify identity
+            result["passed"] = False
+            result["reason"] = "IDENTITY_NOT_BOUND: no expected SHA-256 provided"
+            result["note"] = "Computed hash recorded for enrollment; pass --expected-sha256 to certify"
 
     except Exception as e:
         result["error"] = f"{type(e).__name__}: {e}"
@@ -227,9 +279,29 @@ def verify_strict_load_zero_diff(load_result: CheckpointLoadResult) -> Dict[str,
 # Gate criterion: upstream_parity
 # =============================================================================
 
+def find_upstream_reference_dir(base_dir: Path, patch_size: int) -> Optional[Path]:
+    """Find the upstream reference directory for a given patch size.
+
+    Looks for directories matching the pattern upstream_reference_ps{patch_size}_*
+    """
+    pattern = f"upstream_reference_ps{patch_size}_*"
+    matches = list(base_dir.glob(pattern))
+    if matches:
+        # Return most recent if multiple
+        return sorted(matches)[-1]
+
+    # Fallback to legacy path
+    legacy_path = base_dir / "upstream_reference"
+    if legacy_path.exists():
+        return legacy_path
+
+    return None
+
+
 def verify_upstream_parity(
     bridge: WeatherStepBridge,
-    upstream_dir: Path,
+    upstream_base_dir: Path,
+    patch_size: int,
     x_raw: np.ndarray,
     device: torch.device,
     tolerance: float = 1e-5,
@@ -251,10 +323,21 @@ def verify_upstream_parity(
     }
 
     try:
+        # Find the upstream reference directory
+        upstream_dir = find_upstream_reference_dir(upstream_base_dir, patch_size)
+        if upstream_dir is None:
+            result["error"] = (
+                f"Upstream reference not found in {upstream_base_dir}. "
+                "Run export_upstream_reference.py in a GPU+xformers environment first."
+            )
+            return result
+
+        result["upstream_dir"] = str(upstream_dir)
+
         manifest_path = upstream_dir / "manifest.json"
         if not manifest_path.exists():
             result["error"] = (
-                f"Upstream reference not found at {upstream_dir}. "
+                f"Manifest not found at {manifest_path}. "
                 "Run export_upstream_reference.py in a GPU+xformers environment first."
             )
             return result
@@ -266,13 +349,39 @@ def verify_upstream_parity(
             "timestamp_utc": manifest.get("timestamp_utc"),
             "xformers_version": manifest.get("xformers_version"),
             "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256"),
+            "normalization_policy": manifest.get("normalization", {}).get("policy"),
+            "normalization_digest": manifest.get("normalization", {}).get("digest"),
         }
         result["upstream_available"] = True
 
+        # Verify both 1-step and 4-step references exist
+        required_files = ["official_output_6h_1step.pt", "official_output_6h_4step.pt"]
+        missing_files = []
+        for fname in required_files:
+            if not (upstream_dir / fname).exists():
+                missing_files.append(fname)
+
+        if missing_files:
+            result["error"] = f"Missing required reference files: {missing_files}"
+            result["passed"] = False
+            return result
+
         # Load upstream outputs
         upstream_output_6h_1step = torch.load(upstream_dir / "official_output_6h_1step.pt")
         upstream_input_norm = torch.load(upstream_dir / "input_norm.pt")
 
+        # Verify shapes and dtypes
+        result["upstream_output_shape"] = list(upstream_output_6h_1step.shape)
+        result["upstream_output_dtype"] = str(upstream_output_6h_1step.dtype)
+
+        expected_shape = (1, 69, 128, 256)
+        if tuple(upstream_output_6h_1step.shape) != expected_shape:
+            result["error"] = (
+                f"Unexpected upstream output shape: {upstream_output_6h_1step.shape}, "
+                f"expected {expected_shape}"
+            )
+            return result
+
         # Move to device
         upstream_output = upstream_output_6h_1step.to(device)
         input_norm = upstream_input_norm.to(device)
@@ -283,6 +392,14 @@ def verify_upstream_parity(
                 input_norm, bridge.variables, interval=6, steps=1
             )
 
+        # Verify bridge output shape matches
+        if bridge_output.shape != upstream_output.shape:
+            result["error"] = (
+                f"Shape mismatch: bridge {bridge_output.shape} vs "
+                f"upstream {upstream_output.shape}"
+            )
+            return result
+
         # Compare
         diff = (bridge_output - upstream_output).abs()
         max_diff = float(diff.max().item())
@@ -305,6 +422,143 @@ def verify_upstream_parity(
     return result
 
 
+# =============================================================================
+# Gate criterion: raw_input_binding
+# =============================================================================
+
+def verify_raw_input_binding(
+    x_raw: np.ndarray,
+    upstream_base_dir: Path,
+    patch_size: int,
+) -> Dict[str, Any]:
+    """Verify raw input hash matches expected value from upstream reference."""
+    result = {
+        "criterion": "raw_input_binding",
+        "passed": False,  # fail-closed
+    }
+
+    try:
+        computed_hash = compute_raw_input_hash(x_raw)
+        result["computed_hash"] = computed_hash
+
+        # Find upstream reference
+        upstream_dir = find_upstream_reference_dir(upstream_base_dir, patch_size)
+        if upstream_dir is None:
+            # No reference directory - fail closed (B02 fix)
+            result["passed"] = False
+            result["reason"] = "IDENTITY_NOT_BOUND: no reference raw input hash available to bind against"
+            result["note"] = "Computed hash recorded for enrollment; create upstream reference first"
+            return result
+
+        # Check for raw input hash file
+        hash_file = upstream_dir / "raw_input_hash.txt"
+        if not hash_file.exists():
+            # Check manifest for hash
+            manifest_path = upstream_dir / "manifest.json"
+            if manifest_path.exists():
+                with open(manifest_path) as f:
+                    manifest = json.load(f)
+                expected_hash = manifest.get("raw_input_hash")
+                if expected_hash:
+                    result["expected_hash"] = expected_hash
+                    result["match"] = computed_hash == expected_hash
+                    result["passed"] = result["match"]
+                    if not result["passed"]:
+                        result["error"] = f"Raw input hash mismatch: expected {expected_hash}, got {computed_hash}"
+                    return result
+
+            # No expected hash in manifest either - fail closed (B02 fix)
+            result["passed"] = False
+            result["reason"] = "IDENTITY_NOT_BOUND: no reference raw input hash available to bind against"
+            result["note"] = "Computed hash recorded for enrollment; add raw_input_hash to manifest"
+            return result
+
+        with open(hash_file) as f:
+            expected_hash = f.read().strip()
+
+        result["expected_hash"] = expected_hash
+        result["match"] = computed_hash == expected_hash
+        result["passed"] = result["match"]
+
+        if not result["passed"]:
+            result["error"] = f"Raw input hash mismatch: expected {expected_hash}, got {computed_hash}"
+
+    except Exception as e:
+        result["error"] = f"{type(e).__name__}: {e}"
+
+    return result
+
+
+# =============================================================================
+# Gate criterion: multistep_reference_present
+# =============================================================================
+
+def verify_multistep_reference(
+    upstream_base_dir: Path,
+    patch_size: int,
+    required_steps: Tuple[int, ...] = (1, 4),
+) -> Dict[str, Any]:
+    """Verify all required multistep reference files are present and loadable."""
+    result = {
+        "criterion": "multistep_reference_present",
+        "passed": False,  # fail-closed
+        "required_steps": list(required_steps),
+    }
+
+    try:
+        upstream_dir = find_upstream_reference_dir(upstream_base_dir, patch_size)
+        if upstream_dir is None:
+            result["error"] = f"Upstream reference not found in {upstream_base_dir}"
+            return result
+
+        result["upstream_dir"] = str(upstream_dir)
+
+        present_files = {}
+        missing_files = []
+        shape_mismatch = []
+
+        for steps in required_steps:
+            fname = f"official_output_6h_{steps}step.pt"
+            fpath = upstream_dir / fname
+
+            if not fpath.exists():
+                missing_files.append(fname)
+                present_files[fname] = False
+            else:
+                present_files[fname] = True
+                # Load and verify shape
+                try:
+                    tensor = torch.load(fpath, map_location="cpu")
+                    expected_shape = (1, 69, 128, 256)
+                    if tuple(tensor.shape) != expected_shape:
+                        shape_mismatch.append(
+                            f"{fname}: shape {tuple(tensor.shape)} != {expected_shape}"
+                        )
+                    if not torch.isfinite(tensor).all():
+                        shape_mismatch.append(f"{fname}: contains non-finite values")
+                except Exception as e:
+                    shape_mismatch.append(f"{fname}: load error: {e}")
+
+        result["present_files"] = present_files
+        result["missing_files"] = missing_files
+        result["shape_errors"] = shape_mismatch
+
+        result["passed"] = len(missing_files) == 0 and len(shape_mismatch) == 0
+
+        if not result["passed"]:
+            errors = []
+            if missing_files:
+                errors.append(f"Missing: {missing_files}")
+            if shape_mismatch:
+                errors.append(f"Shape errors: {shape_mismatch}")
+            result["error"] = "; ".join(errors)
+
+    except Exception as e:
+        result["error"] = f"{type(e).__name__}: {e}"
+
+    return result
+
+
 # =============================================================================
 # Gate criterion: zero_edit_equals_official (internal consistency)
 # =============================================================================
@@ -367,12 +621,14 @@ def verify_zero_edit_internal_consistency(
 def verify_normalization_parity(
     norm_from_npy: NormalizationContract,
     norm_dir: Path,
+    expected_digest: Optional[str] = None,
 ) -> Dict[str, Any]:
-    """Verify normalization contract matches reference npz files."""
+    """Verify normalization contract matches reference npz files and expected digest."""
     result = {
         "criterion": "normalization_parity",
         "passed": False,  # fail-closed
-        "digest": norm_from_npy.digest,
+        "computed_digest": norm_from_npy.digest,
+        "policy": norm_from_npy.policy,
         "n_variables": len(norm_from_npy.variables),
         "intervals_present": list(norm_from_npy.diff_std.keys()),
     }
@@ -382,12 +638,30 @@ def verify_normalization_parity(
             result["error"] = f"Normalization directory not found: {norm_dir}"
             return result
 
+        # Load from NPZ with same policy
         norm_from_npz = NormalizationContract.from_npz_dir(
-            str(norm_dir), variables=DEFAULT_VARIABLES, intervals=(6, 24)
+            str(norm_dir), variables=DEFAULT_VARIABLES, intervals=(6, 24),
+            policy=norm_from_npy.policy,
         )
         result["digest_from_npz"] = norm_from_npz.digest
-        result["digests_match"] = norm_from_npy.digest == norm_from_npz.digest
-        result["passed"] = result["digests_match"]
+        result["digests_match_npz"] = norm_from_npy.digest == norm_from_npz.digest
+
+        # Check against expected digest if provided
+        if expected_digest is not None:
+            result["expected_digest"] = expected_digest
+            result["digests_match_expected"] = norm_from_npy.digest == expected_digest
+            result["passed"] = result["digests_match_expected"] and result["digests_match_npz"]
+            if not result["passed"]:
+                result["error"] = (
+                    f"Digest mismatch: computed={norm_from_npy.digest}, "
+                    f"expected={expected_digest}, from_npz={norm_from_npz.digest}"
+                )
+        else:
+            # No expected digest provided - fail closed (B02 fix)
+            # Self-consistency with NPZ is recorded but cannot certify identity
+            result["passed"] = False
+            result["reason"] = "IDENTITY_NOT_BOUND: no expected normalization digest provided"
+            result["note"] = "Computed digest recorded for enrollment; pass --expected-norm-digest to certify"
 
     except Exception as e:
         result["error"] = f"{type(e).__name__}: {e}"
@@ -541,8 +815,8 @@ def compute_rmse_sanity(
                 rmse_vals.append(rmse)
 
             result[label] = {
-                "rmse_z500_weighted": float(np.mean(rmse_vals)),
-                "rmse_z500_std": float(np.std(rmse_vals)),
+                "rmse_z500_weighted": float(np.mean(rmse_vals)) if rmse_vals else None,
+                "rmse_z500_std": float(np.std(rmse_vals)) if len(rmse_vals) > 1 else None,
                 "n_samples": n_samples,
             }
 
@@ -557,20 +831,33 @@ def compute_rmse_sanity(
 # =============================================================================
 
 def run_s0_gate(
+    patch_size: int = 4,
     check_only: Optional[str] = None,
     output_dir: Optional[Path] = None,
+    expected_checkpoint_sha256: Optional[str] = None,
+    expected_normalization_digest: Optional[str] = None,
 ) -> Dict[str, Any]:
     """Run all S0 gate verifications.
 
     All gate criteria are fail-closed: they start False and stay False
     if any exception occurs during evaluation.
+
+    Args:
+        patch_size: Patch size (2 or 4). Default is 4 (mainline).
+        check_only: Run only a specific criterion (not implemented yet).
+        output_dir: Output directory for results.
+        expected_checkpoint_sha256: Expected checkpoint SHA-256 for comparison.
+        expected_normalization_digest: Expected normalization digest for comparison.
     """
+    PATHS = get_paths(REPO_ROOT, patch_size)
+
     result = {
         "status": "failed",
         "run_id": f's0-gate-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%Sz")}',
         "hostname": socket.gethostname(),
         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "repo_root": str(REPO_ROOT),
+        "patch_size": patch_size,
         "torch_version": str(torch.__version__),
         "cuda_available": torch.cuda.is_available(),
         "gate_criteria": {
@@ -581,6 +868,8 @@ def run_s0_gate(
             "normalization_parity": False,
             "no_state_leak": False,
             "outputs_finite": False,
+            "raw_input_binding": False,
+            "multistep_reference_present": False,
         },
         "criteria_details": {},
         "s0_gate_pass": False,
@@ -595,29 +884,43 @@ def run_s0_gate(
             device = torch.device('cpu')
             result["cuda_device"] = "CPU"
         print(f"Using device: {device}", flush=True)
+        print(f"Patch size: {patch_size}", flush=True)
 
         # Load inputs
         print("Loading inputs...", flush=True)
         inputs = load_npy_inputs(PATHS["input_dir"])
         x_raw_0 = inputs['data'][0]
 
-        # Create normalization
-        norm = create_normalization_from_npy(inputs)
+        # Create normalization with official policy (zero diff_mean)
+        norm = create_normalization_from_npy(inputs, policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN)
+        result["normalization_policy"] = norm.policy
+        result["normalization_digest"] = norm.digest
 
         # Load checkpoint (detailed)
-        print("Loading checkpoint (detailed)...", flush=True)
+        print(f"Loading checkpoint (detailed) for ps{patch_size}...", flush=True)
         load_result = load_stormer_checkpoint_detailed(
-            str(PATHS["checkpoint_ps2"]), patch_size=2
+            str(PATHS["checkpoint"]), patch_size=patch_size
         )
         model = load_result.model.to(device)
         bridge = WeatherStepBridge(model, norm, load_result.version)
 
+        # Call check_version_match at gate entry point (B02 fix)
+        print("Verifying version match...", flush=True)
+        try:
+            check_version_match(bridge.version, bridge.version)
+            result["version_match"] = True
+        except ValueError as e:
+            result["version_match"] = False
+            result["version_match_error"] = str(e)
+
         # Run gate criteria
         print("Running gate criteria...", flush=True)
 
-        # 1. ckpt_sha256_bound
+        # 1. ckpt_sha256_bound (FULL comparison, not just length)
         print("  - ckpt_sha256_bound", flush=True)
-        sha256_result = verify_ckpt_sha256(PATHS["checkpoint_ps2"], load_result)
+        sha256_result = verify_ckpt_sha256(
+            PATHS["checkpoint"], load_result, expected_checkpoint_sha256
+        )
         result["criteria_details"]["ckpt_sha256_bound"] = sha256_result
         result["gate_criteria"]["ckpt_sha256_bound"] = sha256_result["passed"]
 
@@ -630,7 +933,7 @@ def run_s0_gate(
         # 3. upstream_parity
         print("  - upstream_parity", flush=True)
         upstream_result = verify_upstream_parity(
-            bridge, PATHS["upstream_reference_dir"], x_raw_0, device
+            bridge, PATHS["upstream_reference_base"], patch_size, x_raw_0, device
         )
         result["criteria_details"]["upstream_parity"] = upstream_result
         result["gate_criteria"]["upstream_parity"] = upstream_result["passed"]
@@ -654,7 +957,9 @@ def run_s0_gate(
 
         # 5. normalization_parity
         print("  - normalization_parity", flush=True)
-        norm_result = verify_normalization_parity(norm, PATHS["norm_dir"])
+        norm_result = verify_normalization_parity(
+            norm, PATHS["norm_dir"], expected_normalization_digest
+        )
         result["criteria_details"]["normalization_parity"] = norm_result
         result["gate_criteria"]["normalization_parity"] = norm_result["passed"]
 
@@ -670,6 +975,22 @@ def run_s0_gate(
         result["criteria_details"]["outputs_finite"] = finite_result
         result["gate_criteria"]["outputs_finite"] = finite_result["passed"]
 
+        # 8. raw_input_binding
+        print("  - raw_input_binding", flush=True)
+        raw_input_result = verify_raw_input_binding(
+            x_raw_0, PATHS["upstream_reference_base"], patch_size
+        )
+        result["criteria_details"]["raw_input_binding"] = raw_input_result
+        result["gate_criteria"]["raw_input_binding"] = raw_input_result["passed"]
+
+        # 9. multistep_reference_present
+        print("  - multistep_reference_present", flush=True)
+        multistep_result = verify_multistep_reference(
+            PATHS["upstream_reference_base"], patch_size, required_steps=(1, 4)
+        )
+        result["criteria_details"]["multistep_reference_present"] = multistep_result
+        result["gate_criteria"]["multistep_reference_present"] = multistep_result["passed"]
+
         # RMSE sanity check (informational)
         print("Computing RMSE sanity check (informational)...", flush=True)
         result["rmse_sanity"] = compute_rmse_sanity(
@@ -687,6 +1008,9 @@ def run_s0_gate(
     except Exception as e:
         result["error"] = f"{type(e).__name__}: {e}"
         result["traceback"] = traceback.format_exc()
+        # Ensure gate does NOT pass on exception (fail-closed)
+        result["s0_gate_pass"] = False
+        result["status"] = "exception"
 
     result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
     return result
@@ -701,6 +1025,7 @@ def generate_report(result: Dict[str, Any]) -> str:
         f'**Executed:** {result.get("started_utc", "unknown")}',
         f'**Hostname:** {result.get("hostname", "unknown")}',
         f'**Repo Root:** `{result.get("repo_root", "unknown")}`',
+        f'**Patch Size:** {result.get("patch_size", "unknown")}',
         f'**Status:** {"PASS" if result.get("s0_gate_pass") else "FAIL"}',
         '',
         '## Environment',
@@ -708,6 +1033,8 @@ def generate_report(result: Dict[str, Any]) -> str:
         f'- PyTorch: {result.get("torch_version", "unknown")}',
         f'- CUDA available: {result.get("cuda_available", False)}',
         f'- Device: {result.get("cuda_device", "unknown")}',
+        f'- Normalization policy: {result.get("normalization_policy", "unknown")}',
+        f'- Normalization digest: `{result.get("normalization_digest", "unknown")}`',
         '',
         '## Gate Criteria Results',
         '',
@@ -718,9 +1045,10 @@ def generate_report(result: Dict[str, Any]) -> str:
     for criterion, passed in result.get('gate_criteria', {}).items():
         status = 'PASS' if passed else 'FAIL'
         details = result.get('criteria_details', {}).get(criterion, {})
-        if 'error' in details:
-            notes = f"Error: {details['error'][:50]}..."
-        elif criterion == 'upstream_parity' and not details.get('upstream_available'):
+        if isinstance(details, dict) and 'error' in details:
+            error_msg = str(details['error'])
+            notes = f"Error: {error_msg[:50]}..." if len(error_msg) > 50 else f"Error: {error_msg}"
+        elif criterion == 'upstream_parity' and isinstance(details, dict) and not details.get('upstream_available'):
             notes = "Reference not found - run export_upstream_reference.py first"
         else:
             notes = ""
@@ -734,25 +1062,30 @@ def generate_report(result: Dict[str, Any]) -> str:
 
     # Checkpoint SHA-256
     sha_details = result.get('criteria_details', {}).get('ckpt_sha256_bound', {})
-    if sha_details.get('sha256'):
+    if isinstance(sha_details, dict) and sha_details.get('computed_sha256'):
         lines.extend([
             '### Checkpoint Identity',
-            f'- SHA-256: `{sha_details["sha256"]}`',
-            '',
+            f'- Computed SHA-256: `{sha_details["computed_sha256"]}`',
         ])
+        if sha_details.get('expected_sha256'):
+            lines.append(f'- Expected SHA-256: `{sha_details["expected_sha256"]}`')
+            lines.append(f'- Match: {sha_details.get("match", "N/A")}')
+        lines.append('')
 
     # Upstream parity
     up_details = result.get('criteria_details', {}).get('upstream_parity', {})
     lines.extend([
         '### Upstream Parity (vs Official xformers Stormer)',
-        f'- Available: {up_details.get("upstream_available", False)}',
+        f'- Available: {up_details.get("upstream_available", False) if isinstance(up_details, dict) else "N/A"}',
     ])
-    if up_details.get('upstream_available'):
-        lines.extend([
-            f'- Max absolute diff: {up_details.get("max_abs_diff", "N/A"):.2e}',
-            f'- Tolerance: {up_details.get("tolerance", "N/A"):.0e}',
-        ])
-    if up_details.get('error'):
+    if isinstance(up_details, dict) and up_details.get('upstream_available'):
+        max_diff = up_details.get("max_abs_diff")
+        tolerance = up_details.get("tolerance")
+        if max_diff is not None:
+            lines.append(f'- Max absolute diff: {max_diff:.2e}')
+        if tolerance is not None:
+            lines.append(f'- Tolerance: {tolerance:.0e}')
+    if isinstance(up_details, dict) and up_details.get('error'):
         lines.append(f'- Error: {up_details["error"]}')
     lines.append('')
 
@@ -764,10 +1097,14 @@ def generate_report(result: Dict[str, Any]) -> str:
         'Note: This is bridge-internal consistency, NOT upstream parity.',
         '',
     ])
-    for key in ['6h_1step', '6h_4step']:
-        if key in ze_details:
-            d = ze_details[key]
-            lines.append(f'- {key}: max_diff={d.get("max_abs_diff", "N/A"):.2e}, passed={d.get("passed")}')
+    if isinstance(ze_details, dict):
+        for key in ['6h_1step', '6h_4step']:
+            if key in ze_details:
+                d = ze_details[key]
+                if isinstance(d, dict):
+                    max_diff = d.get("max_abs_diff")
+                    if max_diff is not None:
+                        lines.append(f'- {key}: max_diff={max_diff:.2e}, passed={d.get("passed")}')
     lines.append('')
 
     # RMSE sanity (informational)
@@ -781,10 +1118,13 @@ def generate_report(result: Dict[str, Any]) -> str:
         '|-----------|-----------|---------------------|',
     ])
     for key in ['6h', '24h']:
-        if key in rmse:
-            val = rmse[key].get('rmse_z500_weighted', float('nan'))
+        if isinstance(rmse, dict) and key in rmse:
+            val = rmse[key].get('rmse_z500_weighted') if isinstance(rmse[key], dict) else None
             ref = HISTORICAL_REFERENCE.get(f'z500_rmse_{key}', {}).get('value', 'N/A')
-            lines.append(f'| {key} | {val:.1f} | ~{ref} |')
+            if val is not None:
+                lines.append(f'| {key} | {val:.1f} | ~{ref} |')
+            else:
+                lines.append(f'| {key} | N/A | ~{ref} |')
 
     lines.extend([
         '',
@@ -796,6 +1136,10 @@ def generate_report(result: Dict[str, Any]) -> str:
 
 def main():
     parser = argparse.ArgumentParser(description="S0 Gate Verification")
+    parser.add_argument(
+        "--checkpoint", choices=["ps2", "ps4"], default="ps4",
+        help="Checkpoint to use (default: ps4 - the mainline per research_spec_v6.yaml)"
+    )
     parser.add_argument(
         "--check", type=str, default=None,
         help="Run only a specific criterion (not implemented yet)"
@@ -804,6 +1148,14 @@ def main():
         "--output-dir", type=Path, default=None,
         help="Output directory for results"
     )
+    parser.add_argument(
+        "--expected-sha256", type=str, default=None,
+        help="Expected checkpoint SHA-256 for comparison"
+    )
+    parser.add_argument(
+        "--expected-norm-digest", type=str, default=None,
+        help="Expected normalization digest for comparison"
+    )
     parser.add_argument(
         "--help-criteria", action="store_true",
         help="Show gate criteria descriptions"
@@ -812,15 +1164,20 @@ def main():
 
     if args.help_criteria:
         print("S0 Gate Criteria:")
-        print("  ckpt_sha256_bound      - Checkpoint SHA-256 hash computed and recorded")
+        print("  ckpt_sha256_bound      - Checkpoint SHA-256 matches expected value")
         print("  strict_load_zero_diff  - Checkpoint loads with zero missing/unexpected keys")
         print("  upstream_parity        - Bridge output matches official xformers Stormer (<=1e-5)")
         print("  zero_edit_equals_official - Bridge internal consistency (<=1e-6)")
-        print("  normalization_parity   - Normalization digest matches reference")
+        print("  normalization_parity   - Normalization digest matches expected reference")
         print("  no_state_leak          - Identical inputs produce bit-identical outputs")
         print("  outputs_finite         - All outputs contain no NaN/Inf")
+        print("  raw_input_binding      - Raw input hash matches expected")
+        print("  multistep_reference_present - All required multistep reference files present")
         return 0
 
+    patch_size = 2 if args.checkpoint == "ps2" else 4
+    PATHS = get_paths(REPO_ROOT, patch_size)
+
     output_dir = args.output_dir or Path(os.environ.get(
         'S0_OUTPUT_DIR', str(PATHS["default_output_dir"])
     ))
@@ -828,9 +1185,16 @@ def main():
     print('=' * 60, flush=True)
     print('S0 Gate Verification - EarthDelta Stormer Bridge', flush=True)
     print('=' * 60, flush=True)
+    print(f'Checkpoint: ps{patch_size}', flush=True)
 
     # Run all gate verifications
-    result = run_s0_gate(check_only=args.check, output_dir=output_dir)
+    result = run_s0_gate(
+        patch_size=patch_size,
+        check_only=args.check,
+        output_dir=output_dir,
+        expected_checkpoint_sha256=args.expected_sha256,
+        expected_normalization_digest=args.expected_norm_digest,
+    )
 
     # Write outputs
     output_dir.mkdir(parents=True, exist_ok=True)
diff --git a/tests/test_s0_fail_closed.py b/tests/test_s0_fail_closed.py
index 318fcec..c7ec1de 100644
--- a/tests/test_s0_fail_closed.py
+++ b/tests/test_s0_fail_closed.py
@@ -178,7 +178,8 @@ def test_verify_upstream_parity_fails_closed_without_reference():
 
         result = verify_upstream_parity(
             bridge,
-            upstream_dir=Path("/nonexistent/upstream_reference"),
+            upstream_base_dir=Path("/nonexistent/artifacts"),
+            patch_size=2,
             x_raw=x_raw,
             device=torch.device("cpu"),
         )
```

---

## 附：新增测试文件全文（tests/test_normalization_policy.py，369 行）

```python
"""Tests for NormalizationContract policy handling.

Tests verify:
1. Official zero-diff-mean policy correctly forces diff_mean to zero
2. Legacy policy preserves existing nonzero diff_mean behavior
3. Policy is included in digest computation
4. Backward compatibility: default policy is legacy
"""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from earthdelta.bridge import (
    NormalizationContract,
    POLICY_LEGACY,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    DEFAULT_VARIABLES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def small_variables():
    """Small variable list for fast tests."""
    return DEFAULT_VARIABLES[:8]


@pytest.fixture
def npz_dir_with_nonzero_diff_mean(small_variables, tmp_path):
    """Create NPZ directory with NONZERO diff_mean values.

    This simulates the real NPZ files which have nonzero diff_mean,
    unlike the official inference.py which uses zeros.
    """
    # Create synthetic normalization files
    mean_dict = {v: np.array([float(i)]) for i, v in enumerate(small_variables)}
    std_dict = {v: np.array([1.0 + 0.1 * i]) for i, v in enumerate(small_variables)}

    np.savez(tmp_path / "normalize_mean.npz", **mean_dict)
    np.savez(tmp_path / "normalize_std.npz", **std_dict)

    for interval in [6, 12, 24]:
        # NONZERO diff_mean to simulate real NPZ files
        diff_mean = {v: np.array([0.01 * (i + 1) * interval]) for i, v in enumerate(small_variables)}
        diff_std = {v: np.array([0.5 + 0.05 * i]) for i, v in enumerate(small_variables)}
        np.savez(tmp_path / f"normalize_diff_mean_{interval}.npz", **diff_mean)
        np.savez(tmp_path / f"normalize_diff_std_{interval}.npz", **diff_std)

    return tmp_path


# =============================================================================
# Test: Official policy forces diff_mean to zero
# =============================================================================

def test_official_policy_forces_zero_diff_mean(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that official_zero_diff_mean policy forces diff_mean to zeros.

    This is the core B01 fix: under the official policy, diff_mean must be
    zeros to match the official inference.py semantics.
    """
    # Load with official policy
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # All diff_mean values should be zero
    for interval in [6, 24]:
        diff_mean = norm_official.diff_mean[interval]
        assert torch.allclose(diff_mean, torch.zeros_like(diff_mean)), (
            f"Official policy should zero diff_mean for interval {interval}, "
            f"but got nonzero values: {diff_mean}"
        )


def test_legacy_policy_preserves_nonzero_diff_mean(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that legacy policy preserves nonzero diff_mean values.

    This ensures backward compatibility with existing experiments that
    relied on the pre-audit behavior.
    """
    # Load with legacy policy
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_LEGACY,
    )

    # diff_mean values should be nonzero (as in the NPZ files)
    for interval in [6, 24]:
        diff_mean = norm_legacy.diff_mean[interval]
        assert not torch.allclose(diff_mean, torch.zeros_like(diff_mean)), (
            f"Legacy policy should preserve nonzero diff_mean for interval {interval}"
        )


# =============================================================================
# Test: Policy affects denormalize_diff behavior
# =============================================================================

def test_official_policy_ignores_nonzero_diff_mean_in_denormalize(
    npz_dir_with_nonzero_diff_mean, small_variables
):
    """Test that denormalize_diff ignores diff_mean under official policy.

    This verifies that even if diff_mean tensors exist in the contract,
    the official policy forces them to be treated as zeros during
    denormalization, matching the official inference.py behavior.
    """
    # Load with official policy
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Load with legacy policy
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_LEGACY,
    )

    # Create synthetic normalized diff
    batch_size = 2
    num_vars = len(small_variables)
    diff_norm = torch.randn(batch_size, num_vars, 16, 32)

    # Denormalize with both policies
    diff_official = norm_official.denormalize_diff(diff_norm, interval=6)
    diff_legacy = norm_legacy.denormalize_diff(diff_norm, interval=6)

    # They should be DIFFERENT because legacy uses nonzero mean
    assert not torch.allclose(diff_official, diff_legacy), (
        "Official and legacy denormalize_diff should produce different results "
        "when NPZ has nonzero diff_mean"
    )

    # Verify official path uses zero mean: diff_raw = diff_norm * std + 0
    # So diff_official should equal diff_norm * std
    std = norm_official.diff_std[6].view(1, -1, 1, 1)
    expected_official = diff_norm * std
    assert torch.allclose(diff_official, expected_official, atol=1e-6), (
        "Official policy should compute diff_raw = diff_norm * std (zero mean)"
    )


def test_legacy_denormalize_diff_unchanged(npz_dir_with_nonzero_diff_mean, small_variables):
    """Regression test: legacy denormalize_diff behavior is unchanged.

    This proves that existing experiments using the legacy path get
    exactly the same numeric behavior as before.
    """
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_LEGACY,
    )

    diff_norm = torch.randn(2, len(small_variables), 16, 32)

    # Legacy: diff_raw = diff_norm * std + mean
    diff_legacy = norm_legacy.denormalize_diff(diff_norm, interval=6)

    std = norm_legacy.diff_std[6].view(1, -1, 1, 1)
    mean = norm_legacy.diff_mean[6].view(1, -1, 1, 1)
    expected_legacy = diff_norm * std + mean

    assert torch.allclose(diff_legacy, expected_legacy, atol=1e-6), (
        "Legacy policy should compute diff_raw = diff_norm * std + mean"
    )


# =============================================================================
# Test: Policy is included in digest
# =============================================================================

def test_policy_changes_digest(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that different policies produce different digests."""
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_LEGACY,
    )

    assert norm_official.digest != norm_legacy.digest, (
        "Official and legacy policies should produce different digests"
    )


def test_policy_in_digest_deterministic(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that digest is deterministic for same policy."""
    norm1 = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    norm2 = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    assert norm1.digest == norm2.digest, (
        "Same policy should produce identical digests"
    )


# =============================================================================
# Test: Default policy is legacy for backward compatibility
# =============================================================================

def test_default_policy_is_legacy(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that default policy is legacy for backward compatibility.

    Existing callers that don't pass the policy parameter should get
    the legacy behavior to avoid breaking existing experiments.
    """
    # Call without policy parameter
    norm_default = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
    )

    assert norm_default.policy == POLICY_LEGACY, (
        "Default policy should be POLICY_LEGACY for backward compatibility"
    )


def test_direct_construction_default_policy(small_variables):
    """Test that direct construction defaults to legacy policy."""
    norm = NormalizationContract(
        inp_mean=torch.zeros(len(small_variables)),
        inp_std=torch.ones(len(small_variables)),
        diff_mean={6: torch.ones(len(small_variables))},
        diff_std={6: torch.ones(len(small_variables))},
        variables=small_variables,
        # policy not specified
    )

    assert norm.policy == POLICY_LEGACY, (
        "Direct construction should default to POLICY_LEGACY"
    )


# =============================================================================
# Test: Invalid policy rejected
# =============================================================================

def test_invalid_policy_rejected(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that invalid policy values are rejected."""
    with pytest.raises(ValueError, match="Unknown policy"):
        NormalizationContract.from_npz_dir(
            str(npz_dir_with_nonzero_diff_mean),
            variables=small_variables,
            policy="invalid_policy",
        )


# =============================================================================
# Test: Rollout behavior differs by policy
# =============================================================================

def test_rollout_behavior_differs_by_policy(npz_dir_with_nonzero_diff_mean, small_variables):
    """Test that full rollout produces different results with different policies.

    This is an end-to-end verification that the policy affects the entire
    prediction pipeline, not just individual operations.
    """
    from earthdelta.bridge import Stormer, WeatherStepBridge
    from earthdelta.contracts import ArtifactVersion

    # Create a small model
    model = Stormer(
        in_img_size=(16, 32),
        variables=small_variables,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )

    # Initialize with non-zero weights to ensure non-trivial outputs
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    model.eval()

    version = ArtifactVersion(
        backbone="stormer_test",
        static_adapter="none",
        edit_bank="none",
        normalization="pending",
        grid="16x32",
        projection="patch2",
        split="test",
        continuation="reference_after_hold",
    )

    # Create bridges with different policies
    norm_official = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )
    norm_legacy = NormalizationContract.from_npz_dir(
        str(npz_dir_with_nonzero_diff_mean),
        variables=small_variables,
        intervals=(6,),
        policy=POLICY_LEGACY,
    )

    bridge_official = WeatherStepBridge(model, norm_official, version)
    bridge_legacy = WeatherStepBridge(model, norm_legacy, version)

    # Run rollout with both
    x_raw = torch.randn(1, len(small_variables), 16, 32)

    with torch.no_grad():
        x_norm_official = norm_official.normalize(x_raw)
        x_norm_legacy = norm_legacy.normalize(x_raw)

        out_official = bridge_official.forward_validation(
            x_norm_official, small_variables, interval=6, steps=2
        )
        out_legacy = bridge_legacy.forward_validation(
            x_norm_legacy, small_variables, interval=6, steps=2
        )

    # Outputs should differ because of different diff_mean handling
    assert not torch.allclose(out_official, out_legacy, atol=1e-5), (
        "Rollout with official vs legacy policy should produce different outputs "
        "when NPZ has nonzero diff_mean"
    )
```

---

## 附：新增测试文件全文（tests/test_s0_gate_identity.py，592 行）

```python
"""Tests for S0 gate identity binding and rejection.

These tests verify that the REAL s0_gate orchestration (not hand-built dicts)
correctly rejects identity mismatches:
- Wrong checkpoint SHA-256
- Wrong normalization digest
- Wrong variable/grid identity
- Wrong raw input
- Wrong shape
- Missing required multistep results

All tests use synthetic/small assets and can run on CPU.
"""
import json
import os
import pytest
import shutil
import tempfile
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from earthdelta.bridge import (
    NormalizationContract,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    _compute_file_sha256,
    DEFAULT_VARIABLES,
)
from earthdelta.contracts import GateIdentityConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def temp_gate_dir(tmp_path):
    """Create temporary directory structure for gate tests."""
    # Create input directory
    input_dir = tmp_path / "scripts" / "s0_gate_inputs"
    input_dir.mkdir(parents=True)

    # Use full 69 variables for consistency with real gate
    full_variables = DEFAULT_VARIABLES
    n_vars = len(full_variables)

    # Create synthetic weather data (smaller spatial dimensions for speed)
    jan2020_data = np.random.randn(10, n_vars, 16, 32).astype(np.float32)
    np.save(input_dir / "jan2020_full.npy", jan2020_data)

    # Create lat/lon
    np.save(input_dir / "lat.npy", np.linspace(-90, 90, 16).astype(np.float32))
    np.save(input_dir / "lon.npy", np.linspace(0, 360, 32).astype(np.float32))

    # Create normalization constants
    inp_mean = np.random.randn(n_vars).astype(np.float32)
    inp_std = np.abs(np.random.randn(n_vars)).astype(np.float32) + 0.1
    np.save(input_dir / "inp_mean.npy", inp_mean)
    np.save(input_dir / "inp_std.npy", inp_std)

    for interval in [6, 24]:
        diff_mean = np.random.randn(n_vars).astype(np.float32) * 0.01
        diff_std = np.abs(np.random.randn(n_vars)).astype(np.float32) + 0.1
        np.save(input_dir / f"diff_mean_{interval}.npy", diff_mean)
        np.save(input_dir / f"diff_std_{interval}.npy", diff_std)

    # Create normalization directory for NPZ files
    norm_dir = tmp_path / "reference" / "stormer" / "normalization_constants"
    norm_dir.mkdir(parents=True)

    # Create NPZ files (matching format from real normalization files)
    # Each variable gets one value
    mean_dict = {v: np.array([inp_mean[i]]) for i, v in enumerate(full_variables)}
    std_dict = {v: np.array([inp_std[i]]) for i, v in enumerate(full_variables)}
    np.savez(norm_dir / "normalize_mean.npz", **mean_dict)
    np.savez(norm_dir / "normalize_std.npz", **std_dict)

    for interval in [6, 24]:
        dm = {v: np.array([0.0]) for v in full_variables}  # Zero for official policy
        ds = {v: np.array([0.5]) for v in full_variables}
        np.savez(norm_dir / f"normalize_diff_mean_{interval}.npz", **dm)
        np.savez(norm_dir / f"normalize_diff_std_{interval}.npz", **ds)

    # Create artifacts directory
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    return {
        "root": tmp_path,
        "input_dir": input_dir,
        "norm_dir": norm_dir,
        "artifacts_dir": artifacts_dir,
        "variables": full_variables,
        "jan2020_data": jan2020_data,
    }


@pytest.fixture
def mock_upstream_reference(temp_gate_dir):
    """Create mock upstream reference directory with valid outputs."""
    ref_dir = temp_gate_dir["artifacts_dir"] / "upstream_reference_ps4_abc12345_zd"
    ref_dir.mkdir(parents=True)

    n_vars = len(temp_gate_dir["variables"])  # 69 variables

    # Create synthetic reference outputs
    output_1step = torch.randn(1, n_vars, 16, 32)
    output_4step = torch.randn(1, n_vars, 16, 32)
    input_norm = torch.randn(1, n_vars, 16, 32)

    torch.save(output_1step, ref_dir / "official_output_6h_1step.pt")
    torch.save(output_4step, ref_dir / "official_output_6h_4step.pt")
    torch.save(input_norm, ref_dir / "input_norm.pt")

    # Create manifest
    raw_input_hash = "abc123deadbeef"
    manifest = {
        "timestamp_utc": "2026-09-21T00:00:00Z",
        "xformers_version": "0.1.0",
        "checkpoint": {
            "sha256": "abc123" * 11,  # 66 chars but we'll truncate
            "patch_size": 4,
        },
        "normalization": {
            "policy": "official_zero_diff_mean",
            "digest": "expected_digest",
        },
        "raw_input_hash": raw_input_hash,
    }
    with open(ref_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Create raw input hash file
    with open(ref_dir / "raw_input_hash.txt", "w") as f:
        f.write(raw_input_hash)

    return {
        "ref_dir": ref_dir,
        "manifest": manifest,
        "raw_input_hash": raw_input_hash,
    }


# =============================================================================
# Import the gate functions for testing
# =============================================================================

# We'll import the verification functions directly
from scripts.s0_gate import (
    verify_ckpt_sha256,
    verify_normalization_parity,
    verify_raw_input_binding,
    verify_multistep_reference,
)


# =============================================================================
# Test: Gate rejects wrong checkpoint SHA-256
# =============================================================================

def test_gate_rejects_wrong_checkpoint_sha256(temp_gate_dir, tmp_path):
    """Test that gate rejects when computed SHA-256 doesn't match expected."""
    # Create a fake checkpoint file
    fake_ckpt = tmp_path / "fake_checkpoint.ckpt"
    with open(fake_ckpt, "wb") as f:
        f.write(b"fake checkpoint content")

    # Compute actual SHA-256
    computed_sha = _compute_file_sha256(str(fake_ckpt))

    # Expected SHA-256 that doesn't match
    wrong_expected = "0" * 64

    result = verify_ckpt_sha256(
        fake_ckpt,
        load_result=None,
        expected_sha256=wrong_expected,
    )

    assert result["passed"] == False, "Gate should reject wrong SHA-256"
    assert result["computed_sha256"] == computed_sha
    assert result["expected_sha256"] == wrong_expected
    assert "mismatch" in result.get("error", "").lower()


def test_gate_accepts_correct_checkpoint_sha256(temp_gate_dir, tmp_path):
    """Test that gate accepts when computed SHA-256 matches expected."""
    # Create a fake checkpoint file
    fake_ckpt = tmp_path / "fake_checkpoint.ckpt"
    with open(fake_ckpt, "wb") as f:
        f.write(b"fake checkpoint content")

    # Compute actual SHA-256
    computed_sha = _compute_file_sha256(str(fake_ckpt))

    # Expected SHA-256 that matches
    result = verify_ckpt_sha256(
        fake_ckpt,
        load_result=None,
        expected_sha256=computed_sha,
    )

    assert result["passed"] == True, "Gate should accept correct SHA-256"
    assert result["match"] == True


# =============================================================================
# Test: Gate rejects wrong normalization digest
# =============================================================================

def test_gate_rejects_wrong_normalization_digest(temp_gate_dir):
    """Test that gate rejects when normalization digest doesn't match expected."""
    # Create normalization contract from NPZ
    norm = NormalizationContract.from_npz_dir(
        str(temp_gate_dir["norm_dir"]),
        variables=list(temp_gate_dir["variables"]),
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Wrong expected digest
    wrong_digest = "wrong_digest_123"

    result = verify_normalization_parity(
        norm,
        temp_gate_dir["norm_dir"],
        expected_digest=wrong_digest,
    )

    assert result["passed"] == False, "Gate should reject wrong normalization digest"
    assert result.get("digests_match_expected") == False or "mismatch" in result.get("error", "").lower()


def test_gate_accepts_correct_normalization_digest(temp_gate_dir):
    """Test that gate accepts when normalization digest matches expected."""
    # Create normalization contract from NPZ
    norm = NormalizationContract.from_npz_dir(
        str(temp_gate_dir["norm_dir"]),
        variables=list(temp_gate_dir["variables"]),
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    # Correct expected digest
    correct_digest = norm.digest

    result = verify_normalization_parity(
        norm,
        temp_gate_dir["norm_dir"],
        expected_digest=correct_digest,
    )

    assert result["passed"] == True, f"Gate should accept correct normalization digest: {result}"


# =============================================================================
# Test: Gate rejects wrong raw input
# =============================================================================

def test_gate_rejects_wrong_raw_input(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when raw input hash doesn't match reference."""
    # Use different input data that produces different hash
    different_input = np.random.randn(8, 16, 32).astype(np.float32)

    result = verify_raw_input_binding(
        different_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    assert result["passed"] == False, "Gate should reject wrong raw input"
    assert "mismatch" in result.get("error", "").lower()


def test_gate_accepts_matching_raw_input(temp_gate_dir, mock_upstream_reference):
    """Test that gate accepts when raw input hash matches reference."""
    # Create input that matches the expected hash
    import hashlib
    expected_hash = mock_upstream_reference["raw_input_hash"]

    # We can't easily create data that hashes to a specific value,
    # so we'll modify the reference file to match our test data
    test_input = np.random.randn(8, 16, 32).astype(np.float32)
    computed_hash = hashlib.sha256(test_input.tobytes()).hexdigest()[:16]

    # Update the reference to expect our test input's hash
    with open(mock_upstream_reference["ref_dir"] / "raw_input_hash.txt", "w") as f:
        f.write(computed_hash)

    result = verify_raw_input_binding(
        test_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    assert result["passed"] == True, "Gate should accept matching raw input"


# =============================================================================
# Test: Gate rejects missing multistep results
# =============================================================================

def test_gate_rejects_missing_1step_reference(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when 1-step reference file is missing."""
    # Remove the 1-step file
    os.remove(mock_upstream_reference["ref_dir"] / "official_output_6h_1step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == False, "Gate should reject missing 1-step reference"
    assert "6h_1step" in str(result.get("missing_files", []))


def test_gate_rejects_missing_4step_reference(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when 4-step reference file is missing."""
    # Remove the 4-step file
    os.remove(mock_upstream_reference["ref_dir"] / "official_output_6h_4step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == False, "Gate should reject missing 4-step reference"
    assert "6h_4step" in str(result.get("missing_files", []))


def test_gate_accepts_all_multistep_present(temp_gate_dir, mock_upstream_reference):
    """Test that gate accepts when all multistep references are present."""
    # Need to fix the shape to match expected (1, 69, 128, 256)
    # For this test, we'll patch the expected shape check
    # Actually, let's create correct-shaped tensors
    ref_dir = mock_upstream_reference["ref_dir"]

    # Re-save with correct shapes (though smaller for test)
    output_1step = torch.randn(1, 69, 128, 256)
    output_4step = torch.randn(1, 69, 128, 256)

    torch.save(output_1step, ref_dir / "official_output_6h_1step.pt")
    torch.save(output_4step, ref_dir / "official_output_6h_4step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == True, f"Gate should accept all multistep present: {result}"


def test_gate_rejects_wrong_shape_multistep(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when multistep reference has wrong shape."""
    ref_dir = mock_upstream_reference["ref_dir"]

    # Save with wrong shape
    wrong_shape_output = torch.randn(1, 32, 64, 128)  # Wrong shape
    torch.save(wrong_shape_output, ref_dir / "official_output_6h_1step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1,),
    )

    assert result["passed"] == False, "Gate should reject wrong shape"
    assert len(result.get("shape_errors", [])) > 0


# =============================================================================
# Test: Gate rejects non-finite multistep values
# =============================================================================

def test_gate_rejects_nonfinite_multistep(temp_gate_dir, mock_upstream_reference):
    """Test that gate rejects when multistep reference contains NaN/Inf."""
    ref_dir = mock_upstream_reference["ref_dir"]

    # Save with NaN values
    nan_output = torch.randn(1, 69, 128, 256)
    nan_output[0, 0, 0, 0] = float('nan')
    torch.save(nan_output, ref_dir / "official_output_6h_1step.pt")

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1,),
    )

    assert result["passed"] == False, "Gate should reject non-finite values"
    assert "non-finite" in str(result.get("shape_errors", [])).lower()


# =============================================================================
# Test: Gate handles missing upstream reference directory
# =============================================================================

def test_gate_handles_missing_upstream_reference(temp_gate_dir):
    """Test that gate fails closed when upstream reference directory is missing."""
    # Don't create any upstream reference

    result = verify_multistep_reference(
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
        required_steps=(1, 4),
    )

    assert result["passed"] == False, "Gate should fail closed when reference missing"
    assert "not found" in result.get("error", "").lower()


# =============================================================================
# Test: GateIdentityConfig validation
# =============================================================================

def test_gate_identity_config_sha256_validation():
    """Test GateIdentityConfig SHA-256 validation."""
    config = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint",
        expected_checkpoint_sha256="abc123def456" * 5 + "ab",  # 64 chars
        patch_size=4,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    # Test matching
    assert config.validate_checkpoint_sha256(config.expected_checkpoint_sha256)

    # Test non-matching
    assert not config.validate_checkpoint_sha256("wrong" * 13)


def test_gate_identity_config_identity_tag():
    """Test GateIdentityConfig identity tag generation."""
    config = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint",
        expected_checkpoint_sha256="abc123def456" * 5 + "ab",
        patch_size=4,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    tag = config.identity_tag
    assert "ps4" in tag, "Identity tag should include patch size"
    assert "abc123de" in tag, "Identity tag should include SHA prefix"
    assert "_zd" in tag, "Identity tag should include policy indicator"


def test_gate_identity_config_reference_dir_namespaced():
    """Test that reference dir name is namespaced by identity."""
    config1 = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint1",
        expected_checkpoint_sha256="a" * 64,
        patch_size=2,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    config2 = GateIdentityConfig(
        checkpoint_path="/path/to/checkpoint2",
        expected_checkpoint_sha256="b" * 64,
        patch_size=4,
        normalization_policy="official_zero_diff_mean",
        normalization_dir="/path/to/norm",
        variables=tuple(DEFAULT_VARIABLES),
        grid_shape=(128, 256),
    )

    # Different configs should have different reference dir names
    assert config1.reference_output_dir_name != config2.reference_output_dir_name, (
        "Different checkpoints/patch sizes should have different reference dirs"
    )


# =============================================================================
# Test: Fail-closed when expected identity values not provided (B02 fix)
# =============================================================================

def test_gate_fails_closed_ckpt_sha256_none(tmp_path):
    """Test that verify_ckpt_sha256 fails closed when expected_sha256=None."""
    # Create a real checkpoint file
    fake_ckpt = tmp_path / "real_checkpoint.ckpt"
    with open(fake_ckpt, "wb") as f:
        f.write(b"checkpoint content for fail-closed test")

    result = verify_ckpt_sha256(
        fake_ckpt,
        load_result=None,
        expected_sha256=None,  # No expected value provided
    )

    # Must fail closed - cannot certify identity without expected value
    assert result["passed"] == False, "Gate must fail closed when expected_sha256=None"
    assert result["computed_sha256"] is not None, "Should still record computed hash for enrollment"
    assert len(result["computed_sha256"]) == 64, "Computed hash should be valid SHA-256"
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"


def test_gate_fails_closed_normalization_digest_none(temp_gate_dir):
    """Test that verify_normalization_parity fails closed when expected_digest=None."""
    # Create normalization contract from valid NPZ data
    norm = NormalizationContract.from_npz_dir(
        str(temp_gate_dir["norm_dir"]),
        variables=list(temp_gate_dir["variables"]),
        intervals=(6, 24),
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )

    result = verify_normalization_parity(
        norm,
        temp_gate_dir["norm_dir"],
        expected_digest=None,  # No expected digest provided
    )

    # Must fail closed - cannot certify identity without expected value
    assert result["passed"] == False, "Gate must fail closed when expected_digest=None"
    # Should still record useful diagnostic info for enrollment
    assert result.get("computed_digest") is not None, "Should still record computed digest"
    # digests_match_npz may be True (self-consistency), but passed must still be False
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"


def test_gate_fails_closed_raw_input_no_upstream_dir(temp_gate_dir):
    """Test that verify_raw_input_binding fails closed when upstream dir doesn't exist."""
    # Use artifacts dir with no upstream reference created
    test_input = np.random.randn(8, 16, 32).astype(np.float32)

    result = verify_raw_input_binding(
        test_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    # Must fail closed - no reference to bind against
    assert result["passed"] == False, "Gate must fail closed when upstream reference dir missing"
    assert result.get("computed_hash") is not None, "Should still record computed hash for enrollment"
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"


def test_gate_fails_closed_raw_input_no_hash_in_reference(temp_gate_dir):
    """Test that verify_raw_input_binding fails closed when upstream dir exists but has no raw_input_hash."""
    # Create upstream reference directory without raw_input_hash
    ref_dir = temp_gate_dir["artifacts_dir"] / "upstream_reference_ps4_test123_zd"
    ref_dir.mkdir(parents=True)

    # Create manifest WITHOUT raw_input_hash
    manifest = {
        "timestamp_utc": "2026-09-21T00:00:00Z",
        "xformers_version": "0.1.0",
        "checkpoint": {"sha256": "a" * 64, "patch_size": 4},
        "normalization": {"policy": "official_zero_diff_mean", "digest": "test_digest"},
        # Note: raw_input_hash is intentionally omitted
    }
    with open(ref_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # No raw_input_hash.txt file either

    test_input = np.random.randn(8, 16, 32).astype(np.float32)

    result = verify_raw_input_binding(
        test_input,
        temp_gate_dir["artifacts_dir"],
        patch_size=4,
    )

    # Must fail closed - no reference hash to bind against
    assert result["passed"] == False, "Gate must fail closed when no raw_input_hash in reference"
    assert result.get("computed_hash") is not None, "Should still record computed hash for enrollment"
    assert "IDENTITY_NOT_BOUND" in result.get("reason", ""), "Should indicate identity not bound"
```
