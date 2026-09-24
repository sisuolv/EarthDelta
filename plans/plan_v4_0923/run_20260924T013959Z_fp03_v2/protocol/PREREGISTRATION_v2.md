# FP-03 Fs pre-registration, protocol v2 (short training horizon)

Status: PREREGISTERED_BEFORE_V2_JOBS. No v2 GPU job has been submitted. The
binding artifact is `fs_protocol_v2.json` in this directory. Its SHA-256 is
what every v2 job passes as `--protocol-sha256`, and the CLI and the decider
refuse any other value. The JSON pins this file's SHA-256.

Protocol v1 (`run_20260923T192759Z_fp03_fs`) and the X1 exploratory run stay
untouched. The v1 result (STOP_FITTED_FS_QUALITY) is final.

## What v2 tests

Question: does a short training horizon, H = 32 updates, produce an Fs that
qualifies? Q6 is evaluated on a genuinely untouched holdout, and all other
FS-QUAL-v1 criteria are carried over unchanged.

- **Candidate.** H = 32, primary lr 2.5e-3, fallback lr 1.25e-3. The fallback
  runs only if the primary job is FAIL.
- **Why H = 32.** The choice uses old data only; see
  `../analysis/h_and_risk_from_old_data.json`.
  - 32 is the only length with any holdout evidence short of 256.
  - On the old, now exposed, holdout, both learning rates showed no harm at
    32 updates (24h 0.9994 / 0.9954) and harm at 256 (1.0389 / 1.0204).
  - 24 or 40 could only be evaluated with new GPU jobs on old data, which is
    outside the budget. H is kept at 32, not tuned.
- **What is already known.** Training is bitwise reproducible, so the
  training-side outcome is fixed in advance. The v2 replicas will replay v1
  J1 arms A2/A3 exactly: Q1–Q5 PASS (primary: Q4 0.9741 / max 0.9800, Q5
  0.9924; fallback: Q4 0.9839 / 0.9877, Q5 0.9886). The confirmatory content
  of v2 is Q0 validity plus **Q6 on the fresh holdout**.

## Rules

- FS-SELECT-v1 applies verbatim; its text is copied into the JSON.
- FS-QUAL-v2 and FS-BUDGET-v2 are given verbatim below.
- In FS-QUAL-v2, only the following differ from FS-QUAL-v1:
  - H_formal is 32;
  - Q2's epoch range becomes k = 1..3;
  - Q6 uses the fresh holdout;
  - Q0's F0 reference is necessarily split, because a fresh holdout cannot
    have an earlier reference panel.

```
FS-QUAL-v2 (per formal replica; L0 = common initial checkpoint = zero-init Fs, bitwise equal to
F0; L1 = checkpoint after exactly H_formal=32 updates; same 8 train issues as v1, same Q/scale,
float64 reduction, no-grad 12-step rollout; E_k = mean train loss over updates 8k..8k+7)
 Q0 validity (else INVALID): official backend + torch 2.3.1+cu121 + xformers 0.0.27, TF32 off
    (effective getters, DEV-001), S0 certificate consumed, admission consumer PASS (train and
    fresh holdout), protocol/config-id match, init digest match, train-panel F0 identical to the
    v1 J1 reference (bitwise; pre-registered fallback bound 1e-6 relative), fresh-holdout F0
    panel bitwise identical across the 4 replicas.
 Q1 all losses/grad norms finite; n_updates == 32.
 Q2 stability: clip_events == 0; max_t L_t / L0_24h(issue_t) <= 1.25;
    for k=1..3: E_k <= 1.02 * min(E_0..E_{k-1}).
 Q3 per-visit last < first on 8/8 issues (listed separately).
 Q4 train panel 24h, r_i = L1/L0, m = mean L1 / mean L0:
    PASS iff max r_i < 1 and m <= 0.99; FAIL iff m > 1.00 or max r_i > 1.02;
    else INCONCLUSIVE_QUALIFICATION.
 Q5 72h guard on train panel: m72 <= 1.01
 Q6 FRESH holdout panel (16 issues, 2019) 24h mean ratio <= 1.02
 6h reported only. quality_pass = Q0..Q3 and Q4==PASS and Q5 and Q6.

FS-BUDGET-v2. GPU jobs: V2-J3 (H=32, lr 2.5e-3, 4 replicas), V2-J3b (H=32, lr 1.25e-3; only if
V2-J3 is STOP_FITTED_FS_QUALITY, never if INCONCLUSIVE or INVALID), V2-J4 (freeze on cuda:0,
then independent reload/verify on cuda:1; only after a QUALITY_PASS_PENDING_DESIGNATED_GATES
decision), plus at most 1 infra retry: max 4 GPU jobs. If V2-J3 and V2-J3b both FAIL, stop and
report; no further candidate without the coordinator. Formal horizon = 32 (no extrapolation).
```

## Fresh holdout: 16 issues from 2019, never trained on or scored

- **Year choice.** 2019 was chosen before any 2019 value was read (see
  `../holdout/holdout_selection_declaration.json`).
  - No Fs experiment has touched 2019.
  - The backbone was pretrained on 1979–2018, so 2015 and 2018 are in-sample
    for it, whereas 2019 was not trained on.
  - 2016 and 2017 are empty.
- **Attempt 1 failed.** Five of the 16 evenly spaced rows had non-finite data
  and could not be replaced within +24h. The declared stop rule fired and was
  honoured; see `../holdout/holdout_attempt1_failed.json`. The integrity scan
  found multi-day holes in whole variable groups in 2015, 2018 and 2019,
  consistent with incomplete downloads.
- **Amendment A1.** A1 was approved by the coordinator and declared before it
  was used; see `../holdout/holdout_selection_amendment_A1.json`.
  - Rule: the nearest slot whose t−12h..t+72h window is fully finite.
  - Selection is on integrity only, never on skill.
  - Result: 16 distinct issues, spaced at least 198h apart, each within 91h
    of its target.
  - All 16 admitted and content-certified; a fresh consumer re-hash equals the
    stored hash; no overlap with training issues.
- **Use restriction.** No model output on these issues exists or will be
  computed before V2-J3. The v2 jobs load them only as the panel.

## Unchanged from v1

- **Selection.** FS-SELECT-v1: cuda:0 is the designated candidate and
  cuda:1..3 are witnesses; all 4 must pass; nothing is substituted.
- **Training issues and objective.** The same 8 bank_fit training issues (the
  v1 admission file, pinned); rank 4, blocks 18–23, 4×6h rollout, 24h
  objective; Adam defaults, clip 1.0, seed 20260921.
- **Environment.** Certified overlay (torch 2.3.1+cu121, xformers 0.0.27).
- **Merge tolerance** (merge check only, never S0):
  atol = max(1e-5, 1.5e-3·delta_max_abs), rtol = 1e-5; zero-edit and
  continuation-after-hold must be exact.
- **Risk note from old data.** 32-update freezes at lr 1e-2 had about a 2×
  margin. At lr 2.5e-3 the margin is expected to be similar, but this is not
  guaranteed, and the tolerance is not adjusted.

## Decider changes (source pinned in the JSON; CPU-tested)

`scripts/r4_fs_decide.py`:
- accepts the train-F0 reference from the pinned v1 decision
  `J1_diag_decision_r2.json`, verified by its SHA-256;
- compares only the reference groups the protocol names;
- adds the cross-replica fresh-holdout F0 equality fact.

`tests/test_fs_v2_chain.py` covers this path, plus the complete synthetic
freeze → independent reload → certify chain. That chain has never run on real
data before V2-J4.

## Limitations

- **2019 caveats.** 2019 is Stormer's checkpoint-selection year and this
  project's "val" split. The 16 holdout issues are now Fs-exposed and must be
  purged from any later policy_dev/confirm use.
- **Holdout size and effect sizes.** The 16 holdout issues give moderate
  power. Effects seen so far are ≤ 0.5%.
- **S0 scope.** S0 certifies the forward pass only.
- **Split label.** The 2020 training rows carry split "test" (as in v1).
- **Delta_min.** delta_min is still not pre-registered, so v2 qualifies "no
  harm", not "useful gain".
