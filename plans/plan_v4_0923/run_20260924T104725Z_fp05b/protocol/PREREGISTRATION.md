# FP-05b five-candidate cache + cheap out-of-sample policy evaluation: pre-registration

Status: PREREGISTERED_BEFORE_FP05_JOBS. No FP-05b GPU job has been submitted.

- **Binding artifact.** `cache_protocol_v1.json` in this directory. Every FP-05b job and
  CPU stage passes its SHA-256 as `--protocol-sha256`; `scripts/r4_candidate_cache.py`,
  `scripts/r4_cache_decide.py` and `scripts/r4_policy_oof.py` refuse any other value.
- **This file.** The JSON pins this file's SHA-256; the rule block below is copied into
  the JSON's `rules_verbatim` byte for byte.
- **Where the rules come from.** The rule names below were designed by the FP-05 Plan
  agent (transcript agent `a784df217c7edebb3`, 2026-09-24T07:44Z) and reconciled with the
  FP-05a findings by the FP-05b Plan agent (`a161ffc3202a77f14`). Until now the full text
  existed only in those transcripts; this file is its first on-disk, hash-pinned copy.
  Lines changed relative to that design are marked `[AMENDED]`, new lines `[NEW]`,
  and executor-level conventions declared before any label exists `[DECLARED]`.

## What FP-05b tests

Whether the certified K=4 bank, used by a cheap legal policy on top of the certified
static Fs, reduces the 24h native loss out of sample by at least delta_min, measured on
a policy_dev set that no model has ever produced a result on. FP-05b emits
ABOVE / BELOW / STRADDLE statuses only; the decision is FP-06's (FP06-RULES-v1 below).

## Fixed inputs (all pinned by SHA-256 in the JSON)

- Certified Fs: FP-04's `inputs.fs_reference` block copied verbatim (FP-03 v2
  `certify/fs/`, FS_SELECTED, digests included).
- Certified bank: FP-04 `certify/bank/` bundle (BANK_CERTIFIED; bank digest `13bc9054...`,
  FP-04 protocol `394562dd...`), `registry.json` (reference + expert_0..3, a0 = rho = 0.25,
  hold 4, `reference_after_hold`).
- S0 certificate `5afbe786...`, gate config `f4a25552...`.
- Data roles: FP-05a exposure ledger `d8ee631f...` -> `splits/split_freeze_v1.json`
  (recomputed from the ledger rows, not copied; `earthdelta/split_freeze.py`).
- policy_dev: declaration (hashed before the 2019H2 scan) -> finiteness scan -> STOP
  record (16 drops > 8) -> coordinator resolution (accept the declared rule's outcome,
  N_target = 112) -> manifest -> `r2_admission_gate.py --data-role policy_dev` (112/112
  admitted, all content certificates PASS) -> consumer re-certification (PASS).
- debug: the 8 FP-04 rows of DEBUG-SELECT-v1 (FP-04 admission `01d344b8...`), consumer
  re-certified (PASS).
- Source: FP-04's 17 pinned files (unchanged) plus every new FP-05b source/test file.

## Coverage caveat that must travel with every DEV result

Strata 16 (2019-10-26T12Z..11-02T06Z) and 19 (2019-11-16T12Z..11-23T06Z) have ZERO
policy_dev issues: the 2019.zarr store holds non-finite values there (16- and 24-step
runs), so no t-12h..t+72h window is clean. 23 of 25 weekly strata are represented;
N_target = 112 (not 128). This is a data-availability gap, not a selection choice; it is
written into `split.coverage_caveat` of the protocol, into `folds.json`,
`dev_protocol.json`, `dev_results.csv` and `paired_block_bootstrap.json`.

## Rules

```
FP-05b RULES (carried from the FP-05 design; [AMENDED]/[NEW]/[DECLARED] mark changes)

EXPO-FREEZE-v1 [REPLACED]. The exposure ledger is FP-05a's exposure_ledger.json (sha d8ee631f...).
 splits/split_freeze_v1.json is rebuilt from its ROWS (every GRADIENT_AND_RESULT / RESULT row's
 t-12h..t+72h support, plus the S0 aggregate result 2020-01-01T00Z..01-03T00Z: conservative rule),
 closed intervals, 24h buffer on each side, touching = excluded. 2020H2 = EXPOSED_NOT_CONFIRM.
 confirm = UNASSIGNED_NO_ACCESS (window null): every FP-05 access raises. bank_fit is not
 consumable by FP-05. Stores allowed: 2019.zarr and 2020.zarr only. A row is consumable iff the
 freeze clears it AND it is on an explicit allow-list (policy_dev: the policy_dev admission sha256 +
 its issue ids, pinned in this protocol; debug: the 8 ids + FP-04 admission sha256 in the freeze).
 The row's data_role tag never authorises anything. Enforcement points: the manifest builder,
 r4_candidate_cache certify-admission / cache (before any model loads) / merge, r4_cache_decide
 --kind debug, r4_policy_oof folds / fit-predict / score. read_slab is the only data I/O path; it
 refuses any index outside the row's certified slice and logs every read.

POLICY-DEV-SELECT-v1 [AMENDED, as declared in admission/policy_dev_selection_declaration.json].
 Pool: the 698 6h slots (2019.zarr steps 746..1443, 2019-07-06T12Z..2019-12-27T18Z). Strata:
 7-day UTC blocks from 2019-07-06T12Z (25). Targets: target_i = 746 + floor(i*697/127 + 0.5),
 i = 0..127. Candidates: slots whose steps t-2..t+12 are all finite in the 2019H2 scan. Pick, in
 increasing i: the nearest unused candidate in the SAME stratum, strictly later than the previous
 kept pick, ties earlier; none -> DROPPED. > 8 drops -> STOP. OUTCOME: 16 drops -> STOP; RESOLVED
 by the coordinator: accept the declared rule's outcome as-is, N_target = 112, strata 16 and 19
 unrepresented (real data gap); none of the three alternative rules was applied. Nested subsets
 for DEV-SCALE-v1 only: N/2 = surviving even i (56), N/4 = surviving i % 4 == 0 (30). Any picked
 row rejected by the admission gate -> STOP (none was: 112/112 PASS).

DEBUG-SELECT-v1. The first and last issue, in time order, of each FP-04 group G_k
 (bank_fit_grouping.json): iss_292517fb189ee1bd, iss_039add2d41f5c08a (G0), iss_4f2b54d7e8953451,
 iss_6361e4cdd8050ae3 (G1), iss_32d48efebee12be2, iss_bbd4e6e2ec71d9ca (G2), iss_04eb8e8d56c68faa,
 iss_b42c91ed1b555939 (G3). Already exposed; used to validate the machinery only.

CACHE-RUN-v1 (per worker)
 E0 validity (else INVALID; E0a-E0b refuse with exit 2 BEFORE any backbone/Fs/bank byte loads):
  E0a [AMENDED] protocol sha256 + config-id match; split_freeze.load_freeze sha-checked;
      assert_rows_clear PASS for the worker's rows under its role and allow-list; every file of
      source_at_preregistration re-hashes equal (frozen core unchanged);
      dev jobs only: dev_protocol.json hash-chained to this protocol and to a PASS debug decision.
  E0b S0 certificate consumed; torch 2.3.1+cu121 / xformers 0.0.27 official backend
      precondition; authorize_certified_fs binds the certified Fs by hash;
      verify_bank_bundle(deep=True) PASS against the FP-04 protocol sha; the registry has exactly
      the 5 pinned rows with the registered coefficients.
  E0c official backend postcondition; load_certified_fs with the four digests equal to the pins;
      load_bank digest and expert digests equal the pins; TF32 off (effective getters).
 Per issue: every read via read_slab [AMENDED]; load x_t and the raw truth at steps 1/4/12.
  For c in [reference, expert_0..3]: controlled_rollout(fs_bridge, x, vars, 6, 12,
  entry.to_edit_plan(), bank, (18..23), return_trajectory=True) under no_grad (FP-04's
  bank_panel_losses call). Endpoints = the RAW (denormalized) states at steps 1/4/12, float32
  [AMENDED: raw space, exactly the field the objective scores]. Status finite or FAIL_NONFINITE.
  Losses float64 via objective_loss_for_sample with FP-04's ObjectiveSpec
  (build_objective_spec(bridge, lat, lead_steps=(4,), space="raw"), per lead via
  dataclasses.replace). Hook count after each rollout equals the count before.
  Reference cross-check [NEW]: the plain-Fs rollout through FP-04's own call path
  fs_rollout_trajectory(fs_adapters=None) is run on every issue; its endpoints must equal the
  reference endpoints byte for byte (A3 re-check; gating in DEBUG-DECIDE-v1, report-only in dev).
  F0 background (report-only, never a candidate, never in the matrix or the oracle) in a
  separate file.
  Features are NOT computed by the worker [AMENDED]: merge builds them from t-12/t-6/t through
  read_slab on CPU.
 Per shard: bank digest and merged-Fs digest unchanged at the end.
 Output hygiene: stdout carries status, counts and timings only, never a loss value.
 Files per issue: endpoints.npy float32 [5,3,69,128,256] (sealed), truth.npy [3,69,128,256]
  (sealed), f0_endpoints.npy (sealed, report-only), gpu_losses.json (sealed), issue.json
  (hashes, statuses, timings, hooks, read log).

CACHE-VALID-v1 (merge; and r4_cache_decide --kind debug on the debug matrix)
 V1 coverage: the (issue, candidate, lead) keys equal the declared N x 5 x 3 exactly; no
    duplicate, no extra; the shards form the declared partition (i mod 4 over time order).
 V2 identity: every row's identity sha equals the frozen identity (Fs digests, bank digest,
    registry sha, protocol, admission, normalization, store, hold, rho, continuation); the
    coefficients and support index equal the registry entry; one role for all rows.
 V3 bytes: every endpoint/truth/f0/gpu_losses file re-hashes; shape and dtype correct.
 V4 status: PASS or FAIL_NONFINITE only (a FAIL row carries no loss; a PASS row a finite one);
    > 2% FAIL_NONFINITE rows -> INVALID_EVIDENCE.
 V5 CPU recompute: every loss recomputed on CPU from endpoints + truth;
    |CPU - GPU| <= 1e-12 x L; the CPU values are the canonical labels; max reported.
 V6 gains: gain = L_Fs - L_c from canonical losses; analytic full_objective_gain(Y - F_ref,
    F_c - F_ref) agrees at atol 1e-11 / rtol 1e-8.
 V7 time: valid_time = issue_time + lead; lead-to-step 6->1, 24->4, 72->12.
 V8 read-set [NEW]: every worker (and merge) read-log entry is inside the issue's certified slice
    of its role's store; any other store or index -> INVALID_EVIDENCE.
 Outputs: cache_manifest.json; candidate_results.json (canonical, sorted rows; the canonical
 matrix digest is over the sorted rows as JSON, not file bytes) [AMENDED: parquet is an
 optional convenience copy only]; features/<issue>.npy; background_f0.json (report-only).

DEBUG-DECIDE-v1 (C-J1). PASS iff all of: E0 on all 4 workers; the 8x5x3 matrix complete and
 V1-V8 pass; ANCHORS: every debug issue's Fs 6/24/72 losses (both the zero-plan-with-bank
 reference AND the plain-Fs FP-04 call path) equal FP-04's bank_panels_expert{k}.json Fs values
 BITWISE, and for issues in G_k expert_k's losses equal FP-04's L1 BITWISE; duplicates (every
 issue computed by two workers) byte-identical; reference endpoints byte-identical to the
 plain-Fs rollout. Anchors within 1e-6 relative but not exact -> DEBUG_ANCHOR_INEXACT, to the
 coordinator, never an auto-pass. Anything else -> STOP. The debug issue set must equal the
 freeze's debug allow-list exactly.

DEV-SCALE-v1 [AMENDED]. N = max n in {N_target = 112, N/2 subset = 56, N/4 subset = 30} with
 (a) every row of the nested subset admitted; (b) model_load + ceil(n/4) x p95(t_issue) x 1.5
 <= 45 min; (c) n x bytes_issue x 1.2 <= 60 GB and <= free disk; (d) peak GPU memory <= 64 GiB.
 None -> STOP. Inputs: C-J1 timing, size, memory and admission fields only (no loss). The decider
 writes cache/dev_protocol.json hash-chained to this protocol and the debug decision: N, the issue
 list, the shard map, a report-only planning MDE (variance from exposed data only: FP-04 panels).

OOF-v1 [AMENDED: block origin]. Blocks: 7-day UTC blocks from 2019-07-06T12Z (25; 16 and 19
 empty). Folds: 4 outer folds of CONTIGUOUS calendar blocks [DECLARED: blocks 0..24 split
 0-5 / 6-11 / 12-17 / 18-24, fixed before any label]. Purge: for fold f, drop from training every
 issue whose support widened by 24h touches any fold-f issue's support. Fit on fold-train only:
 scaler (std floor 1e-12), PCA d = 8, ridge, k-means. Inner HPO: 3 contiguous inner folds over the
 outer-train blocks with the same purge; scaler/PCA refit inside every inner fold; criterion:
 inner-OOF realized mean 24h native gain vs Fs. Ridge lambda in {0.1, 1, 10, 100, 1e3, 1e4}
 [DECLARED: ties -> larger lambda]; k in {2, 3, 4} [DECLARED: ties -> smaller k], n_init 10,
 seed 20260924 [DECLARED: k-means++ init, rng seed 20260924 + 1000k + init, Lloyd <= 300 iters].
 HPO trial counts recorded.
 Methods: M0 fitted_fs_no_edit. M1 fold_train_best_static: argmax over the 5 candidates of
 fold-train mean 24h gain, ties -> no-edit. M2 regime_router: per-cluster best of the 5; clusters
 with < 5 training issues fall back to M1. M3 ridge_direct_gain: 4 ridges predicting absolute 24h
 gain; act as argmax if the prediction > 0, else no-edit (threshold 0). M4
 offline_complete_hindsight_oracle: per-issue argmin of 24h loss; computed only by score. M5
 oracle_static: hindsight best single candidate; diagnostic; score only.
 Inputs to fit-predict: features for all issues plus FoldLabels(table, train_ids) per fold, which
 raises on any other id and logs every access; no endpoints, truth, response or oracle. A
 FAIL_NONFINITE candidate's training label is its realized fallback (gain 0).
 Outputs: policies/folds.json; policies/oof_predictions_fold{f}.json + oof_predictions.json;
 label_access_log.json; prediction_freeze.json (hashes of all of those, the features manifest,
 both protocols, the source; created_utc; opened with "x").
 Poisoning check (CP3) [AMENDED scope]: re-running fit-predict with fold f's eval (and purged)
 labels set to NaN or 1e9 must leave oof_predictions_fold{f}.json byte-identical (only fold f is
 refit in that check; FP-05a/v1 caution: scope = that fold's own OOF predictions).

EVAL-v1 (score; verifies the prediction freeze first and refuses without it)
 Realized loss L_action per issue; a chosen FAIL_NONFINITE falls back to L_Fs and is counted as a
 failure; a FAIL_NONFINITE reference -> INVALID_EVIDENCE. Effect: G_lead(M vs B) =
 (sum L_B - sum L_M) / sum L_Fs, issues weighted equally, the denominator recomputed inside every
 bootstrap draw; 24h primary, 6h diagnostic, 72h guard. Also: harmful_edit_rate, no_edit_rate,
 per-expert selection counts, costs (GPU seconds per rollout; online = features + 1 rollout;
 offline = cache + fitting). Fixed comparison list: oracle vs Fs; oracle vs M1; M5 vs Fs; M1 vs Fs;
 M2 vs Fs; M2 vs M1; M3 vs Fs; M3 vs M1; 72h for M1-M3 vs Fs; 6h for M1-M3 vs Fs (diagnostic);
 F0 vs Fs as background. Paired block bootstrap: B = 10,000, seed 20260924, whole 7-day blocks
 resampled jointly across all methods and leads, n = number of non-empty blocks (23); 95%
 percentile CI. Status: ABOVE (CI_low >= d), BELOW (CI_high < d), STRADDLE otherwise. Also
 reported: H_Fs = (sum L_Fs - sum L_F0) / sum L_Fs and G_F0 = G_Fs - H_Fs; 14-day-block
 sensitivity report-only. Outputs: evaluation/dev_results.csv, paired_block_bootstrap.json,
 costs.json, failures.json -- each carrying the coverage caveat.

DELTA-MIN-v1. d = 0.0034 relative 24h reduction on the sum-L_Fs denominator (anchor: Fs's own
 measured out-of-sample 24h harm on the FP-03 2019H1 holdout, 0.0033810878 on the Fs denominator /
 0.0033925584 on F0; an engineering screen, not a break-even proof, never re-derived after DEV
 outcomes). 72h guard passes iff CI_low of G72(M vs Fs) >= -d. alpha 0.05 two-sided; power 0.8
 only for the MDE. N_cap = the chosen N; no extension. Numerical floor 0 (bitwise-deterministic
 pipeline). The MDE is stored separately and never feeds d.

FP06-RULES-v1 (frozen now, applied unchanged by FP-06). In order:
 1. the INVALID_EVIDENCE conditions (any V-check, E0 or freeze failure) -> INVALID;
 2. oracle vs Fs BELOW -> STOP_CURRENT_BANK;
 3. oracle vs M1 BELOW -> PIVOT_STATIC (FP-06 reports whether M1 vs Fs is ABOVE);
 4. M3 vs M1 BELOW, with the oracle headroom not BELOW -> STOP_CURRENT_SELECTOR;
 5. M3 vs M1 ABOVE, M3 vs Fs ABOVE and the 72h guard passes -> CONTINUE_NEXT_ITERATION, narrowed
    to cheap policies with no dual-head claim (any response/dual-head experiment is a new phase);
 6. any STRADDLE on the path -> INCONCLUSIVE, final at N_cap.
 Every FP-06 statement about DEV carries the coverage caveat (strata 16, 19 unrepresented) and
 H_Fs / G_F0.

CACHE-BUDGET-v1. C-J1: debug, 4 workers, each its own 2 issues plus its neighbour's 2 (every
 issue computed twice). C-J1b: only if C-J1 STOPs on a defect in NEW FP-05 code, with a deviation
 record, at most once. C-J2: the dev cache, 4 shards, submitted only after DEBUG PASS and
 dev_protocol.json. Retries: 1 infra retry in total, identical argv, platform failure only.
 Cap: 4 GPU jobs (coordinator decision); expected 2. Timeouts 3600 s. Policies and scoring are CPU
 only. Stops: C-J1 and C-J1b both STOP -> BLOCKED; a non-infrastructure V-check failure ->
 INVALID_EVIDENCE; a matrix still incomplete after the retry -> STOP; never score a subset. No job
 is submitted without the coordinator's go-ahead.
```

## Decided values (frozen here; previously PROPOSED_NOT_FROZEN in the FP-05a DRAFT)

Boundary rule conservative; confirm null / UNASSIGNED_NO_ACCESS; F0 background included,
report-only; primary lead 24h, report leads 6/24/72h; gain normaliser sum L_Fs per draw;
delta_min 0.0034; 72h guard CI_low >= -d; dynamic-over-static required (FP06 step 5);
N_final = DEV-SCALE-v1's choice from {112, 56, 30}, chosen once; debug issues = the 8 ids;
replacement rule = same-stratum nearest (as declared; outcome accepted by the coordinator);
bidirectional blocked cross-fit; outer folds 4, inner folds 3 (coordinator decision: keep 4);
base block 7 days; ridge grid as above; confidence 0.95; bootstrap unit paired 7-day block;
draws 10,000; sensitivity blocks 14 days report-only; seed 20260924; MDE report-only from
exposed data; max GPU jobs 4 (coordinator decision), infra retries 1, GPU budget 4 x 3600 s
nominal; the measured C-J1 profile is binding for DEV-SCALE-v1.

## Why these thresholds (fixed before any FP-05 output exists)

- delta_min is the Fs's own measured out-of-sample harm: a dynamic layer is only worth its
  cost if it at least repays what the static reference costs. It was measured on exposed
  2019H1 data before any policy_dev value existed.
- Bitwise anchors are justified by FP-04 A1 (assembled singleton == source bank, exactly)
  and FP-03's cross-GPU bitwise determinism.
- One-shot N avoids optional stopping; a STRADDLE at N_cap is a legitimate final answer.

## Known limitations

- Seasonal mismatch: experts are Jan-Jun 2020 month blocks; policy_dev is Jul-Dec 2019.
- The certified Fs is slightly worse than F0 out of sample (Q6 1.003393); report H_Fs.
- 147 pool-period issue times were integrity-read (no model) by the stopped shared-F0 sprint;
  disclosed, not excluded. policy_dev is a retrospective held period, not a deployment.
- Two weekly strata have no data (see the coverage caveat); 23 blocks for the bootstrap.
- No confirm set exists; FP-05b neither names nor reads one.
