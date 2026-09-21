# EarthDelta

**Response-space planning of low-rank edits for a frozen ML weather model.**

EarthDelta learns to *predict* how a frozen forecaster (Stormer) will respond to a small,
reversible parameter edit — before ever applying it — and selects edits at issue time
under a budget, using an exact quadratic form derived from forecast sensitivity theory.
No fine-tuning loop, no retraining the base model, no peeking at future truth.

> **Status (2026-09-21):** core primitives, Stormer bridge, and the ERA5 data pipeline are
> implemented and unit-tested (195 tests, CPU). The GPU-side S0 validation gate (real
> checkpoints, zero-edit equivalence, RMSE-vs-paper) is the next milestone — see
> [Status & roadmap](#status--roadmap). This is an active research repo, not a released
> package.

---

## Table of contents

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
  missing/unexpected keys is the structural-equivalence proof), reproduces the official
  normalization → diff-predict → denormalize → accumulate → renormalize rollout loop
  bit-for-bit against `reference/stormer/`, and injects `ExpertLoRA` coefficients on
  `attn.proj` in blocks 18–23 via `controlled_rollout` — verified to be an exact no-op at
  zero coefficients (`no_state_leak` test).

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
| **S0 gate** (`scripts/s0_gate.py`, GPU) | strict real-ckpt load, zero-edit == official `forward_validation` (≤1e-6), 6h/24h RMSE vs. paper | ⏳ next — needs an ACP GPU job |
| S1+ | training the `e0`/`du` heads, budget selection experiments | not started |

The S0 gate is the real go/no-go: it's the first point where predictions from this
codebase are checked against the official Stormer implementation on real weights and
real data, not against each other. See `plans/plans_v1_0919/v6_draft/` for the full
staged plan (S0 through S7) and the pre-registered kill criteria at each stage — the
single largest risk to this project is not novelty (see above) but whether the oracle
edit-selection gain clears a 2–3% Z500@72h RMSE bar at the S3 go/no-go; that bar, not
implementation difficulty, is what would end the project.

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
