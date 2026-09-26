# EarthDelta

**Response-space planning of low-rank edits for a frozen ML weather model.**

EarthDelta learns to *predict* how a frozen forecaster (Stormer) will respond to a small,
reversible parameter edit — before ever applying it — and selects edits at issue time
under a budget, using an exact quadratic form derived from forecast sensitivity theory.
No fine-tuning loop, no retraining the base model, no peeking at future truth.

> **Status (2026-09-25):** the S0 gate passes official Stormer parity exactly (1/4/12-step
> `max_abs_diff = 0.0` on real H100 hardware), and a static reference adapter ("Fs") and a
> K=4/rank=4 dynamic expert bank are formally certified through a pre-registered protocol
> with real GPU evidence. A separate audit still has an open formal-Fs stability
> finding (some 500-update experts worsened on every paired diagnostic sample), so
> that certification is not a license to trust every fitted expert downstream.
> **FP-05 has now run the first real out-of-sample policy
> evaluation of that bank** — a purged, block-bootstrapped comparison on a held-out 2019 H2
> window (N=112 issues, real GPU-built candidate cache, `plans/plan_v4_0923/run_20260924T104725Z_fp05b/`).
> **Result: the bank's primary policy (ridge-based expert routing) does not clear the
> pre-registered minimum-effect bar** — its 24h out-of-sample gain over Fs is BELOW
> delta_min=0.34% (95% CI [0.239%, 0.284%]), and even the hindsight-optimal oracle only
> STRADDLEs it (CI high 0.351%). No out-of-sample forecasting benefit from dynamic editing
> has been demonstrated yet; this is an honest negative/inconclusive result, not a failed
> run. FP-06 (the formal accept/reject decision built on these numbers) has not been run.
> Separately, a standard-metrics evaluation harness reusing official
> [`google-research/weatherbenchX`](https://github.com/google-research/weatherbenchX) code
> (`earthdelta/wbx/`) has been added and independently cross-checked bit-for-bit against
> these same certified numbers. This is an active research repo, not a released package.

---

**Latest execution note (2026-09-25):** a fresh 4-GPU standard-resource H100 run (`pt-lujnmfhg`) passed the full S0 gate with `max_abs_diff = 0.0` at 1/4/12 steps under the pinned Torch `2.3.1+cu121` / xformers `0.0.27` overlay. The base image itself does not ship xformers, so the overlay or an equivalent certified image must be recorded for official-parity runs. This remains an engineering gate; Fs stability and dynamic-bank out-of-sample value are still open.

---

## Independent reviews (2026-09-26) — read these first

Two independent reviews of the current state were written on 2026-09-26. They supersede the
status notes above wherever they disagree. Both are in Chinese.

| Review | Relative link | GitHub link (branch `audit/round2-review-20260921`) |
|---|---|---|
| Claude review (Claude Code, `claude-opus-5-5`) | [`reviews/CLAUDE_INDEPENDENT_REVIEW_20260926.md`](reviews/CLAUDE_INDEPENDENT_REVIEW_20260926.md) | https://github.com/sisuolv/EarthDelta/blob/audit/round2-review-20260921/reviews/CLAUDE_INDEPENDENT_REVIEW_20260926.md |
| Claude review — recompute scripts and outputs | [`reviews/claude_independent_review_20260926/`](reviews/claude_independent_review_20260926/) | https://github.com/sisuolv/EarthDelta/tree/audit/round2-review-20260921/reviews/claude_independent_review_20260926 |
| Codex review | [`reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md`](reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md) | https://github.com/sisuolv/EarthDelta/blob/audit/round2-review-20260921/reviews/EARTHDELTA_INDEPENDENT_REVIEW_20260926.md |
| Codex follow-up prompt for ChatGPT | [`reviews/CHATGPT_FOLLOWUP_ANALYSIS_PROMPT_20260926.md`](reviews/CHATGPT_FOLLOWUP_ANALYSIS_PROMPT_20260926.md) | https://github.com/sisuolv/EarthDelta/blob/audit/round2-review-20260921/reviews/CHATGPT_FOLLOWUP_ANALYSIS_PROMPT_20260926.md |

Where the two reviews agree:

- Only the frozen 2019H2 DEV set (112 issues, 23 weekly blocks) has been evaluated; no legal
  fresh/confirm split exists, and the dynamic router (M3) shows no gain over the static choice (M1).
- GitHub HEAD does not contain the local working-tree fixes (10 modified files plus untracked
  `scripts/r7_*` / `tests/test_r7_*`); the latest R7 diagnostics were produced by that uncommitted code.
- Tests passing and S0 parity are engineering facts, not evidence that the research hypothesis holds.

Where they disagree (the Claude review recomputed against the unedited backbone F0, which the
FP-05b protocol reported only as background):

- **Strongest baseline.** Codex: M1/static. Claude: **F0**. Relative to F0, Fs, M1, M3 and even the
  24h hindsight oracle are significantly worse at 6h and 72h (M1 72h −0.69% objective, CI entirely
  negative), and only +0.07% at 24h. Per variable, M1 degrades Z500, T850 and MSLP RMSE at 24h and
  72h (Z500 72h −0.98%); the only gains are 50–100 hPa winds.
- **Cause.** The training/selection objective (`earthdelta/static_adapter.py` `build_objective_spec`)
  scales by the state std with equal weights over 69 channels, so humidity and winds make up ~95%
  of it and Z/T/MSLP/T2m less than 3%. Fs was fit on 8 issues × 32 updates; experts on 6–16 issues.
- **72h guard.** Codex: passes. Claude: passes only relative to Fs; relative to F0 it is about −0.7%.
- **Next step.** Codex: one selector-only intervention on a fresh split. Claude: not worth running
  (its success bar would require beating the oracle point estimate); instead fix the baseline and
  metrics, repair the ERA5 stores (2016/2017/2022 empty, 12–29% NaN steps in 2015/2018/2019,
  `pull_wb2.py` reports failure as success), then run a proper headroom test against F0 on
  headline variables before any e0/du work.

---

## Table of contents

- [Independent reviews (2026-09-26) — read these first](#independent-reviews-2026-09-26--read-these-first)
- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [Why this is different](#why-this-is-different)
- [Repo layout](#repo-layout)
- [Quickstart](#quickstart)
- [The `earthdelta` package](#the-earthdelta-package)
- [Stormer bridge](#stormer-bridge)
- [Data pipeline](#data-pipeline)
- [Status & roadmap](#status--roadmap)
- [Provenance & licensing](#provenance--licensing)

## The idea in one paragraph

At issue time, an operational forecaster is frozen — you don't get to retrain it before
every forecast. But you *can* apply a small, structured, reversible edit to its weights
(a low-rank adapter drawn from a fixed dictionary) and you'd like to know, cheaply and in
advance, whether that edit will make the next forecast better or worse. EarthDelta trains
two things offline — a flow-dependent reference-error forecast `e0` (how wrong the frozen
model is *about to be*, in a fixed linear summary space) and a per-edit response model
`du` (how that summary shifts *if* edit `k` is applied) — and combines them at inference
time with the exact Forecast Sensitivity to Observations (FSO)-style quadratic gain

```
gain(k) = 2⟨e0, du_k⟩ − ‖du_k‖²
```

Edits are then selected under a coefficient budget (finite dictionary search or a
box-constrained QP), with a hard boundary: the operational selection path never sees
future ground truth, only `e0` and `du`, both of which are themselves label-free at
serving time.

## Why this is different

- **Prediction, not adaptation.** We don't fine-tune the base model per forecast; we
  predict the effect of a *candidate* edit and choose whether/which to apply.
- **Exact, not proxy, objective.** The gain used for selection is the same quadratic
  identity used in FSO for observation impact, applied here to parameter-space edits —
  not a learned scalar risk score standing in for it.
- **Reversible, auditable edits.** Edits are low-rank adapters on top of frozen weights,
  drawn from a fixed dictionary with explicit coefficients — not permanent weight
  updates, and not opaque hidden-state memory.
- **Checked against the literature, not just claimed.** This repo's survey work (5
  literature surveys, ~200 searches, see `plans/`) found no prior work occupying the
  exact combination of (a) online prediction of both `e0` and `du` from legal history
  only, (b) the exact FSO quadratic combination, and (c) budgeted selection with
  cross-edit interaction terms. See `plans/plans_v1_0919/v6_draft/` for the full
  novelty analysis and nearest-neighbor comparisons (Aurora LoRA, WeatherPEFT,
  Adapter Banks, VI-MoLE).

## Repo layout

```
earthdelta/              # core research package (installable, MIT)
├── contracts.py          # ArtifactVersion + EditPlan (expert coefficients + window)
├── paired.py             # e0/du factorization in a caller-supplied LINEAR summary view
├── probe.py               # finite-difference response probing of a pure forecast callable
├── geometry.py            # local quadratic training geometry
├── teacher.py              # offline box-constrained oracle (labels OK; never called online)
├── selection.py            # budgeted plan selection: finite dictionary + continuous box-QP
├── heads.py                # BoundedProgramHead / InteractionUtilityHead / ReferenceErrorHead
├── memory.py               # auditable verified-record replay (EWMA default, no state leak)
├── spectral.py             # phase-aware scalar-coefficient diagnostics (not a full SHT)
├── lowrank.py              # K-expert low-rank adapters (ExpertLoRA) on attn.proj-style layers
├── bridge/
│   ├── stormer_arch.py      # faithful Stormer re-implementation (SDPA instead of xformers)
│   └── stormer_bridge.py    # strict ckpt loading, normalization contract, controlled rollout
└── data/
    ├── pull_wb2.py           # streaming ERA5 pull from WeatherBench2 GCS + conservative regrid
    └── make_splits.py        # train/val/test/shift split manifest with leakage guard windows

tests/                    # 195 tests (pytest, CPU-only except 3 `slow`-marked real-ckpt tests)
scripts/s0_gate.py        # GPU-side S0 validation gate (run on ACP; see Status & roadmap)
checkpoints/              # Stormer .ckpt files (git-ignored, ~5.5 GB each) + pull retry scripts
data/                     # pulled/regridded ERA5 zarr stores (git-ignored, ~9 GB/year)
reference/                # 46 pinned third-party repos used as ground truth (git-ignored;
                           #   `_manifest.json` has pinned commits + licenses, `_clone_refs*.sh`
                           #   re-creates them)
plans/                    # research plan, 5 literature surveys, external red-team packet
```

## Quickstart

```bash
git clone git@github.com:sisuolv/EarthDelta.git && cd EarthDelta
pip install -e ".[dev]"        # numpy, scipy, torch, pytest

# Full suite (CPU)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/ -q
# -> 192 passed, 3 skipped
```

The 3 skipped tests load real Stormer checkpoints (`checkpoints/*.ckpt`, ~5.5 GB each)
and self-skip on a memory-constrained host (they need 12–24 GB to hold the state dicts;
this repo's dev container has an 8.6 GB limit). They run on any machine with enough RAM,
and are the CPU-side half of the [S0 gate](#status--roadmap).

`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` works around a protobuf C++-backend
crash inside some NVIDIA PyTorch containers; harmless elsewhere.

## The `earthdelta` package

The package is a clean-room merge of two earlier research kits (v5 Research Kit +
Response Kit — see `plans/plans_v1_0919/v6_draft/PACKAGE_MERGE_MAP.md` for the exact
merge decisions), generalizing per-file from single-rank-group masks to K-expert
dictionaries with explicit coefficients. Each module's docstring states its contract and
non-goals explicitly (e.g. `spectral.py` is *not* a spherical harmonic transform;
`teacher.py` is *never* called by the online controller; `paired.py`'s gain identity
requires the summary operator to be linear). Read the module docstring before the code.

## Stormer bridge

`earthdelta/bridge/` lets EarthDelta's edits act on a real pretrained
[Stormer](https://github.com/tung-nd/stormer) weather model without depending on
`xformers`:

- `stormer_arch.py` is adapted from `tung-nd/stormer` (MIT, pinned commit — see file
  header) with the *only* structural change being `F.scaled_dot_product_attention` in
  place of `xformers.ops.memory_efficient_attention` (verified numerically equivalent,
  ≤1e-5, in `tests/test_bridge.py`).
- `stormer_bridge.py` loads the official Lightning checkpoints with `strict=True` (zero
  missing/unexpected keys is the structural-equivalence proof), implements the official
  normalization → diff-predict → denormalize → accumulate → renormalize rollout loop,
  and injects `ExpertLoRA` coefficients on `attn.proj` in blocks 18–23 via
  `controlled_rollout`. Zero coefficients are an exact no-op (`no_state_leak` test).
  Real H100 S0 evidence now passes all three registered rollouts (1/4/12-step) at
  `max_abs_diff = 0.0` against the official implementation, after two root-caused fixes:
  the inverse-normalization transform was reading device/dtype-cast constants instead of
  the original NPZ-precision ones, and TF32 (on by default on H100) was silently
  corrupting precision by ~3 orders of magnitude above the frozen `1e-5` tolerance. See
  `plans/plan_v4_0923/run_20260923T_fp01_trace/` for the real job evidence.

## Data pipeline

`earthdelta/data/pull_wb2.py` pulls ERA5 reanalysis at 1.40625° from the
[WeatherBench2](https://weatherbench2.readthedocs.io/) public GCS zarr store, conservatively
regrids 512×256 → 128×256 to match the Stormer training grid, and writes per-year zarr
stores with the 69-channel variable order used by
`reference/stormer/configs/finetune_multi_step.yaml`.

```bash
# One year, sequential mode (single process — required in memory-constrained
# environments; ~110 min/year on a proxied connection):
python -m earthdelta.data.pull_wb2 --year 2020 --sequential --https

# Consolidate to the training-friendly (1, 69, 128, 256) chunk layout + verify + sanity stats:
python -m earthdelta.data.pull_wb2 --year 2020 --finalize

# Build the split manifest (train 2015-18 / val 2019 / test 2020 / shift 2021-22):
python -m earthdelta.data.make_splits --years 2020 --output data/splits/splits_2020.json
```

Design notes worth knowing before touching this file:

- Pulls **by source variable**, not by output channel — pressure-level variables (13
  levels/variable) are fetched as one chunk-aligned read per time batch, not 13 separate
  reads. An earlier per-channel design caused a ~13× download amplification.
- **Sequential mode** processes one source variable at a time in a single process; this
  is a deliberate memory tradeoff for 8 GB-cgroup style environments, not a performance
  default — use parallel `--worker-id {0,1,2}` mode if you have >16 GB available.
- Marker files (`.markers_<year>/<source_var>.done`) make every stage idempotent: a
  killed/restarted process resumes from the last completed source variable rather than
  re-downloading.
- Completeness is checked by **exact expected timestep count** (leap-year aware), not by
  file existence or a size threshold.

2020 is currently pulled and finalized end-to-end: 1464 timesteps × 69 channels,
t2m mean 279.2 K / MSLP mean 100918 Pa (both match published ERA5 climatology at this
resolution).

## Status & roadmap

| Stage | What it checks | Status |
|---|---|---|
| Core package (110 unit tests) | contracts, factorization, selection, LoRA experts | ✅ done |
| Stormer bridge (CPU, random weights) | SDPA≡softmax, zero-edit no-op, normalization round-trip | ✅ done |
| ERA5 data pipeline | 2020 pulled, rechunked, split manifest built | ✅ done |
| **S0 gate** (`scripts/s0_gate.py`, GPU) | strict real-ckpt load, zero-edit parity, and multistep rollout checks at a frozen `1e-5` tolerance | ✅ PASS on real H100 (1/4/12-step `max_abs_diff = 0.0`) |
| **Fs certification** (static reference adapter, GPU) | pre-registered protocol, formal qualification against a fresh out-of-sample panel, exact freeze/reload verification | ✅ protocol-certified (`FS_SELECTED`) — only "no material harm" out of sample; formal-fit stability finding remains open |
| **K=4/rank=4 dynamic bank** (GPU) | each expert trained from one shared certified Fs (not re-fit per worker), exact assembly-equivalence verification | ✅ assembly-certified (`BANK_CERTIFIED`) — per-expert gains are in-sample only, formal-Fs stability and out-of-sample value remain open |
| **FP-05a** (read-only evidence audit) | rebuilt the full exposure ledger, re-verified all 17 pinned files and every certified number, corrected an error in the original FP-05 data-split design | ✅ done (`plans/plan_v4_0923/run_20260924T093810Z_fp05a/`) |
| **FP-05b** (candidate cache + purged out-of-sample policy evaluation, GPU) | real 4-worker debug batch (C-J1) + full 4-shard cache build (C-J2) on N=112 policy_dev issues (2019 H2), then blocked cross-fit policy fitting + paired block-bootstrap scoring, all independently re-verified bit-for-bit against FP-03/FP-04's recorded numbers | ✅ done — **primary policy (ridge) is BELOW the pre-registered delta_min bar; hindsight oracle only STRADDLEs it** (`plans/plan_v4_0923/run_20260924T104725Z_fp05b/`) |
| **WeatherBench-X integration** (`earthdelta/wbx/`) | export F0/Fs/bank forecasts to the official WeatherBench-X/WeatherBench 2 format and reuse their RMSE/ACC/regrid code instead of custom eval code | ✅ Tier A (CPU) and Tier B (real GPU, bit-for-bit reconciled against FP-03/FP-04) both pass; benchmark runner against public baselines not yet run |
| **FP-06** (formal accept/reject decision) | applies pre-registered rules to the FP-05b ABOVE/BELOW/STRADDLE results to reach a final call | not started |
| S1+ / holdout evaluation | final confirm-set evaluation | not started — no genuinely untouched confirm data currently exists on disk; 2021/2022 are the leading future candidates pending a fresh confirm-freeze (see `plans/plan_v4_0923/FP05A_PLAN.md`) |

The S0 gate was the first real go/no-go: the point where predictions from this codebase
are checked against the official Stormer implementation on real weights and real data,
not against each other — it now passes cleanly. The second real go/no-go was FP-05b: does
editing this backbone with the certified expert bank produce any real, out-of-sample
forecast improvement over the (already only break-even) static reference? **The answer so
far is no** — the primary policy's out-of-sample gain over Fs does not clear the
pre-registered minimum-effect bar, and the theoretical best case (hindsight oracle) only
just reaches it. This is a genuine, pre-registered negative result, not a bug. FP-06 (the
formal decision procedure) and a possible future confirm-set run are the remaining open
questions. See `plans/plan_v4_0923/` for the full real-job evidence trail (protocols,
decisions, deviations, certification bundles) behind every claim above, and
`plans/plans_v1_0919/v6_draft/` for the original staged S0–S7 plan this work descends from.

## Provenance & licensing

- `earthdelta/` (this package): MIT, see `pyproject.toml`.
- `earthdelta/bridge/stormer_arch.py`: adapted from `tung-nd/stormer` (MIT), pinned
  commit noted in the file header.
- `reference/`: 46 third-party repositories used as read-only ground truth for
  correctness checks, never modified or shipped — not part of any release, cloned
  on demand via `reference/_clone_refs*.sh`, pinned commits + licenses recorded in
  `reference/_manifest.json`.
- ERA5 data (`data/`, git-ignored) and Stormer checkpoints (`checkpoints/*.ckpt`,
  git-ignored): third-party, not redistributed via this repo — see WeatherBench2 and
  Stormer's own terms.
