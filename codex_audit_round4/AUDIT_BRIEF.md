# EarthDelta — Round-4 Audit Brief (2026-09-22)

## Purpose

Rounds 1–3 (see `plans/plans_0921/`, `codex_audit_round2/`, and
`plans/plans_v3_0922/EarthDelta_Stage1A_Round3_Audit_Review.md`) audited the
package selection, P0-01/02/03, and — most recently — the Stage 1A
normalization-identity fix (B01/B02/B10, committed as `c397a7b`, with the
round-3 review packaged into `d749a1c`, the current `HEAD`).

**Everything since `d749a1c` is uncommitted working-tree state.** No commit
or push has happened. This round audits all of it: the remaining Stage-1
fixes (B05/B06/B08/B12/B13/B14), the first real GPU execution of the S0 gate
(honest partial result), a new admission gate (B08 content / B09 real
time-join / B15 registry), and the Fs (frozen static reference) + dynamic
expert-bank training pipeline, including three distinct bugs found and fixed
during real-GPU execution and **one newly-found, still-open problem that has
not been fixed** (see the dedicated section near the end — read it before
anything else).

We are not asking you to re-litigate Stage 1A (already round-3-audited) or
the package/task-DAG structure (rounds 1–2) unless you find this round's
work structurally contradicts them.

## Exact scope: `git diff HEAD` (`HEAD` = `d749a1c62521226df857587e08f7d067b0f15355`, branch `audit/round2-review-20260921`)

Modified (10 files, 5190 insertions / 469 deletions total):
```
earthdelta/bridge/stormer_bridge.py    357 ++
earthdelta/contracts.py                455 ++
earthdelta/data/make_splits.py         745 ++
earthdelta/metrics_contract.py         566 ++
earthdelta/selection.py                 84 ++
scripts/export_upstream_reference.py   634 ++
scripts/s0_gate.py                    2155 ++
tests/test_metric_contract.py          500 ++
tests/test_normalization_policy.py     126 +-
tests/test_s0_gate_identity.py          37 +-
```

New, untracked (production code):
```
earthdelta/bank_training.py     1991 lines
earthdelta/pilot_contract.py    1640 lines
earthdelta/registry.py           788 lines
earthdelta/static_adapter.py    1939 lines
scripts/r2_admission_gate.py     315 lines
scripts/r2_fs_bank_train.py     1016 lines
scripts/r2_pilot_preflight.py    257 lines
```

New, untracked (tests):
```
tests/test_bank_gradients.py           696
tests/test_bank_training.py            780
tests/test_candidate_registry_b15.py   405
tests/test_checkpoint_compat_shim.py   264
tests/test_content_verification_b08.py 476
tests/test_fs_static_adapter.py        903
tests/test_pilot_admission_b09.py      643
tests/test_pilot_contract.py           720
tests/test_s0_gate_end_to_end.py      1241
```

Also untracked but **out of scope, do not review or modify**: `artifacts/`
(real job outputs — read-only reference, see below), `checkpoints/run_multi_year_pull.sh`
and `plans/plans_0921/`, `plans/plans_v2_0921/`, `plans/plans_v3_0922/` (the
user's own planning documents, not code under audit).

## What changed, by stage

### Stage 1B — native scoring contract (B05, B06-Q)

`earthdelta/metrics_contract.py` gained `full_objective_loss` /
`full_objective_gain`: `L = Σ q·((Y−F)/s)² / Σ q` over the full
`[B(,K),H,V,Lat,Lon]` target shape (q normalized once across H/V/Lat/Lon,
keeping B/K), float64 accumulation, `Q_eff = diag(q/s²)/Σq`,
`g = 2eᵀQ_eff u − uᵀQ_eff u`. `gain_analytic` and `gain_calibrated` are
kept as separate entry points (calibration off by default). `weighted_mse`
was extended with input validation (the round-2/round-3 finding was that it
never validated its `weights` argument — `weights=[2,-1]` could silently
produce a negative "MSE"). Old per-lead/per-F diagnostic reducers were kept
but explicitly labeled diagnostic-only, not swapped in for the real
objective.

### Stage 1C — admission/gate closure (B08 guard, B12, B13, B14)

New `earthdelta/pilot_contract.py` + `scripts/r2_pilot_preflight.py`:
`validate_slice_index` / `validate_loaded_sample` / `assert_artifact_binding`,
wired into real entry points (not dataclasses nobody calls). `s0_gate.py`
gained: PASS is only committed after all required computation + identity
artifacts are written, with an unconditional revocation path
(`_assert_gate_verdict_committable`) if a late exception fires (B13); safe
formatting of possibly-missing numeric values in the report (B14). Bridge
gained a documented serial-only concurrency guard
(`_RolloutRegistry`, `stormer_bridge.py` — process-wide, mutex-protected,
keyed by backbone id) rather than silently allowing unsupported concurrent
reuse (B12).

### Stage 2 — real GPU S0 gate (first real execution)

Real ACP jobs (not simulated): `pt-j66hv5ph` (environment probe, TIMEOUT —
worker has no usable PyPI egress), `pt-ll11jb3w` (environment retry via a
vendored `.pydeps`, designed FAILED exit confirming xformers genuinely
absent), `pt-exx2mmw7` (official upstream export via
`scripts/export_upstream_reference.py`, FAILED rc=2, xformers-blocked exit
exactly as designed — no SDPA substitute fabricated as if it were official),
`pt-z0iuosd6` (`scripts/s0_gate.py` itself, real H100, FAILED rc=1,
`s0_gate_pass=False`).

The S0 result is an **honest partial result, not a crash**:
`artifacts/round2_cci/ed-r3-j1-s0gate-0921200335-0180dc/s0_gate_output/.../s0_gate_result.json`
shows 3/16 criteria PASS on real GPU evidence (`identity_config_bound`,
`ckpt_sha256_bound`, `normalization_parity`), 5/16 FAIL for a single root
cause (no upstream reference exists because export is xformers-blocked:
`source_identity_match`, `variable_coordinate_identity_match`,
`manifest_identity_match`, `raw_input_binding`, `multistep_reference_present`),
and 8/16 correctly **SKIPPED** by the gate's own fail-closed design, which
refuses to attempt model-load/inference/parity checks unless the 5
identity-binding criteria above pass first. We investigated and confirmed
(grep on `reference/stormer/stormer/models/hub/stormer.py`) that xformers'
`MemEffAttention` is a hard, unconditional dependency of the *official*
Stormer class (no upstream fallback attention backend exists in the pinned
commit); EarthDelta's own bridge (`stormer_bridge.py` /
`bridge/stormer_arch.py`) uses `F.scaled_dot_product_attention` instead and
is **not** blocked by this — but that means **upstream numerical parity
remains unverified**, which is exactly the axis A01/round-1/round-2 cared
about most. xformers itself: we tried the official pin
(`0.0.22.post7`, requires `torch==2.1.0`, two minors behind this image) and
the best version-matched PyPI wheel (`0.0.26.post1`, `requires_dist
torch==2.3.0`, an exact nominal match) — the latter imports but its compiled
CUDA extension registers zero backend ops at runtime
(`NotImplementedError: No operator found for 'memory_efficient_attention_forward'`),
because the NGC-patched torch build in this image diverges from upstream
PyPI torch's ABI despite the matching version number. A from-source build
(nvcc 12.4 is present) was judged out of this window's time budget and not
attempted — flagged as a viable future option, not silently dropped.

### Stage 3a — admission gate (B08 content, B09 real time-join, B15 registry)

New `earthdelta/registry.py` (formal candidate registry — non-empty,
real-float, explicit no-edit row, rejects complex/empty tables;
`single_expert_plans` truncated-rho case is asserted to still equal the
canonical `a0=0.25` coefficient in the registry, not silently dropped).
`earthdelta/data/make_splits.py` gained a real timestamp-coordinate join
(history/current/target endpoints must actually exist on disk; formal mode
rejects placeholder normalization hashes; each admitted sample records its
own `issue_id` and `process_group_id`). `earthdelta/pilot_contract.py`
gained real content verification (batches are actually read and checked,
not just marker/shape metadata). `earthdelta/selection.py` gained
`RegistrySelection` / `select_from_registry`, additive only — the diff
against the pre-existing `unified_select` / `select_plan` /
`plan_from_prediction` / `_plan_from_finite_candidates` functions is empty.

Real-data results: `scripts/r2_admission_gate.py --years 2020 ... --formal`
→ PASS, 6/6 real 2020 timesteps admitted. Same command against `--years 2016`
→ FAIL, `ADMISSION_STORE_EMPTY` ×6 (2016.zarr genuinely holds 0 timesteps on
disk — a real, pre-existing data gap, not a code bug). Known limitation
carried forward: `assert_artifact_binding`'s `STALE_MARKER_POINTS_AT_NEW_STORE`
heuristic still false-positives on legitimate two-phase (intermediate→rechunk)
zarr stores including good ones like `2020.zarr`; the new admission path
deliberately routes around it rather than fixing it. `data_role` is tagged
and required but not yet enforced against reuse (nothing stops re-reading a
`confirm`-tagged sample). The physical-range band (σ=40) is calibrated only
against real 2020 data and would not catch a systematically mislabeled-unit
channel.

### Stage 3b — Fs (frozen static reference) + K=4/rank=4 dynamic expert bank

New `earthdelta/static_adapter.py` (Fs training/freeze/merge-equivalence
machinery) and `earthdelta/bank_training.py` (bank capacity decision,
diversity grouping, horizon feasibility), driven by
`scripts/r2_fs_bank_train.py` (`--mode gradient_check` caps at 32 updates,
`--mode formal` caps at 500 updates, via `MODE_UPDATE_CAPS`). Four
independent per-GPU expert-training processes (`--device cuda:0..3`), K=4,
rank=4, target blocks 18–23.

**Real-GPU debugging chain, three distinct bugs found and fixed in
sequence, each independently re-verified by the parent session (not taken
on agent self-report):**

1. **Device-mismatch bug** (resolved before this round's tightest evidence
   window; fixed and verified via job re-runs, superseded by the two below).
2. **TF32-induced merge-equivalence divergence.** Real jobs
   `pt-5ma8r7m2` / `pt-2v1z1p4i` (early 4-expert attempts) surfaced spurious
   branch-vs-merge mismatches traced to TF32 matmul reduced precision on
   H100. Fixed via a scoped `_disable_tf32_for_identity_check()` context
   manager (save/restore, not a global setting change) plus a
   `self_max_abs_diff` self-consistency diagnostic (recomputing the *same*
   branch twice to distinguish real hardware non-determinism from a
   deterministic precision effect). Verified via job `pt-sgc0joda`.
3. **Fixed `merge_atol` did not generalize across training length.** A
   32-update ("gradient_check") calibration of `merge_atol=1e-4` (verified
   PASS on job `pt-2r6tbwu7`) failed on the first 500-update ("formal") run
   (job `pt-3x63g0c6`, residual up to `5.51e-4`). Root cause (established
   from the two jobs' own on-disk `run_record.json` artifacts, no new
   diagnostic GPU job needed): the branch-vs-merge FP32 non-associativity
   residual scales with the merged LoRA delta's magnitude
   (`artifact.delta_max_abs`), not with a fixed constant —
   `max_abs_diff / delta_max_abs` sits in `[5.6e-4, 7.9e-4]` at 32 updates
   and `[1.4e-4, 2.6e-4]` at 500 updates, while `self_max_abs_diff = 0.0`
   exactly in **all 12 experts checked across 3 jobs**, ruling out hardware
   non-determinism as the cause in both regimes. Per this project's own
   stated policy (tolerances are registered from numerical evidence, never
   loosened just to make a result pass), the fixed constant was replaced
   with a formula: `effective_atol = max(merge_atol_floor=1e-5,
   merge_atol_relative=1.5e-3 × artifact.delta_max_abs)`, implemented as a
   new `_effective_merge_atol()` helper in `static_adapter.py`, with the
   old `merge_atol` parameter removed from `record_post_freeze_reference`'s
   signature entirely (not left as an unused vestige). **Confirmed on a
   fresh real-GPU job**, `pt-cdj1s2le` — a full 500-update, 4-expert re-run
   of the exact formal-mode argv, with delta magnitudes (`0.517`, `0.807`,
   `1.592`, `2.160`) different from (not tuned on) the values used to derive
   the `1.5e-3` coefficient (`0.512`, `0.708`, `1.854`, `2.305`). All 4
   experts' recorded `atol` in `run_record.json` matched the formula
   exactly on independent recomputation, with pass margins `3.16×`–`13.77×`,
   and `self_max_abs_diff = 0.0` / `zero_edit_vs_fs = 0.0` still exact for
   all 4.

Job artifacts for the full chain are on disk under `artifacts/round2_cci/`
(directory names contain the job's `ed-r3-j4-bank-*` label and timestamp;
`job_result.json` in each gives the real ACP terminal status).

## THE OPEN, UNRESOLVED FINDING — read this before forming an overall verdict

While independently re-verifying job `pt-cdj1s2le`'s merge-equivalence PASS,
we also inspected `fs_fit.loss_direction` in each expert's `run_record.json`
— a same-sample (paired by `issue_id`) first-vs-last-update loss comparison,
added earlier in this project specifically because naively comparing the
*first* and *last* update's raw loss is invalid under a round-robin
multi-sample training schedule (different updates can score different
samples). The paired, same-sample version is a valid training-direction
signal.

**In both real-GPU formal-mode (500-update) runs, exactly 2 of the 4
experts got *worse* on every single one of their 8 paired samples — not
noise, not a partial regression, a complete reversal:**

| Run | expert0 | expert1 | expert2 | expert3 |
|---|---|---|---|---|
| `pt-3x63g0c6` | 8/8 improved | **0/8**, loss 0.031→0.153 (5.0×) | 8/8 improved | **0/8**, 0.031→0.039 |
| `pt-cdj1s2le` | 8/8 improved | 8/8 improved | **0/8**, 0.031→0.133 (4.3×) | **0/8**, 0.031→0.034 |

At 32 updates (job `pt-2r6tbwu7`), all 4 experts show 8/8 improved — this
divergence appears specific to the longer (`formal`, 500-update) training
horizon. Which expert index diverges differs between the two formal runs
(1 and 3, then 2 and 3) — the affected slot is not fixed to a particular
target-block assignment, which is suggestive (per-expert seed/init or data
ordering rather than a structural per-block bug) but not established.

We confirmed via `grep` that `loss_decreased` / `loss_direction` is computed
and written to `run_record.json` and printed to the training log, but
**nothing in `r2_fs_bank_train.py` or `r2_admission_gate.py` gates on it** —
a diverged expert is stored in the bank exactly like a healthy one, because
merge-equivalence (branch computation == merged computation, a pure
numerical-consistency check) is orthogonal to whether the underlying Fs
training actually converged to something useful.

**We have not root-caused this.** We do not know if it is a learning-rate/
schedule issue specific to the longer horizon, an optimizer-state issue, a
per-expert seeding artifact, or something else. It was found in the last
~20 minutes of a time-boxed session and deliberately not chased further so
as not to start new GPU diagnostic work outside the window. This is the
single most important open question for this audit round: **does this
undermine the "Stage 3b SUCCEEDED" characterization, and if so, how badly,
and what evidence would be needed to bound its impact before any downstream
candidate-cache work consumes these bank experts?**

## Known limitations carried forward from earlier rounds (unchanged, for context only)

- `scripts/s0_gate.py` has no TF32-disable of its own guard (unlike the
  Stage 3b merge-equivalence check) — not yet flagged as blocking, since S0
  is already blocked by xformers for the criteria TF32 would affect.
- `build_dynamic_bank` (in `bank_training.py`) still has no `device`
  parameter in its signature — a prior agent recommended this as a
  future signature-level fix; not yet done.
- Real data availability: `data/era5_1p40625/2016.zarr` and `2017.zarr`
  hold 0 real timesteps; `2021.zarr` does not exist (only
  `2021_intermediate.zarr`); only 2015/2018/2019/2020 are complete. This
  constrains what real data Stage 4 (not yet started) can draw on.
- History is admitted into the sample record as provenance
  (`history_time`) but is not actually fed into the model — the bridge has
  no multi-horizon history input path, so the original "current + prior 6h
  + prior 12h" spec is not implementable as-is against the current bridge
  signature. Explicitly deferred, not silently dropped.
- A full bare `pytest` run across the entire test tree currently produces
  `Interrupted: N errors during collection` due to a pre-existing, unrelated
  missing `xarray` dependency in a data-pipeline test file and an absl-flags
  parsing conflict, both triggered only when the whole tree is collected
  together. Confirmed unrelated to any of this round's changes (the
  affected target files pass cleanly both standalone and paired with a
  neighboring test file). If you hit this collecting the full tree, this is
  a known, pre-existing environment issue, not something to reopen as a new
  finding — but please say so explicitly in your report rather than silently
  working around it, in case our characterization here is wrong.

## What is explicitly NOT in scope this round

Stage 4 (full candidate cache across non-overlapping shards + a 5-row
decision table with paired block-bootstrap confidence intervals) **has not
been started** — there is nothing to audit there yet. Do not evaluate or
speculate about oracle-table / label-efficiency / dual-head-advantage /
novelty conclusions; none have been produced.
