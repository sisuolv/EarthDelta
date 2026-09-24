# FP-03 Fs pre-registration (protocol v1)

Status: PREREGISTERED_BEFORE_J1. No J1/J2/J3/J3b/J4 job had been submitted when
this was written. The binding artifact is `fs_protocol_v1.json` in this
directory. Its SHA-256 is the value every job passes as `--protocol-sha256`.
The CLI (`scripts/r2_fs_bank_train.py`) and the decider (`scripts/r4_fs_decide.py`)
refuse any other hash. This Markdown file is the readable companion, and the
protocol JSON pins its SHA-256.

## Rules (verbatim, binding)

```
FS-SELECT-v1. (1) The formal configuration is fitted once, in one ACP job, as 4 independent
processes on cuda:0..3 with byte-identical argv except --device/--output-dir. (2) The cuda:0
process (replica0/) is the DESIGNATED CANDIDATE, fixed by device index before submission;
cuda:1..3 are REPRODUCIBILITY WITNESSES and can never become Fs. (3) Fs is selected iff all 4
processes satisfy FS-QUAL-v1 AND the designated candidate then passes numerical_merge, reload
and identity gates in independent processes. (4) Otherwise no Fs is selected from that job; no
witness, intermediate checkpoint, screen arm, or artifact of pt-3x63g0c6/pt-cdj1s2le/pt-e8y7ib01
may be substituted. (5) No loss, gain, panel ratio, weight norm, or any policy_dev/confirm
quantity is consulted to choose among processes. FP-03 reads no policy_dev/confirm data.

FS-QUAL-v1 (per formal replica; L0 = common initial checkpoint = zero-init Fs, bitwise equal to
F0; L1 = checkpoint after exactly H_formal=256 updates; same 8 train issues, same Q/scale,
float64 reduction, no-grad 12-step rollout; E_k = mean train loss over updates 8k..8k+7)
 Q0 validity (else INVALID): official backend + torch 2.3.1+cu121 + xformers 0.0.27, TF32 off,
    S0 certificate consumed, admission consumer PASS, protocol/config-id match, init digest
    match, F0 panel identical to J1's (bitwise; pre-registered fallback bound 1e-6 relative).
 Q1 all losses/grad norms finite; n_updates == 256.
 Q2 stability: clip_events == 0; max_t L_t / L0_24h(issue_t) <= 1.25;
    for k=1..31: E_k <= 1.02 * min(E_0..E_{k-1}).
 Q3 per-visit last < first on 8/8 issues (listed separately).
 Q4 train panel 24h, r_i = L1/L0, m = mean L1 / mean L0:
    PASS iff max r_i < 1 and m <= 0.99; FAIL iff m > 1.00 or max r_i > 1.02;
    else INCONCLUSIVE_QUALIFICATION.
 Q5 72h guard on train panel: m72 <= 1.01          [or null -> report-only + limitation]
 Q6 holdout panel 24h mean ratio <= 1.02
 6h reported only. quality_pass = Q0..Q3 and Q4==PASS and Q5 and Q6.

FS-SCREEN-v1. Arms A0..A3 on cuda:0..3: lr = 1e-2 (positive control), 5e-3, 2.5e-3, 1.25e-3;
all else identical (Adam defaults, clip 1.0, seed 20260921, same init digest, rank 4, blocks
18-23, 4x6h rollout, 24h objective, round-robin in the pt-e8y7ib01 issue order, no warmup/decay),
H=256, mode stability_screen. STABLE = Q1 and Q2 criteria. PROGRESSING = final train-panel 24h
mean ratio <= 0.99. If A0 is STABLE -> NOT_TESTABLE, STOP. S = {A1..A3: STABLE and PROGRESSING}.
S empty -> LR hypothesis REFUTED, STOP. Otherwise LR hypothesis CONFIRMED for H=256;
selected = highest-lr arm in S; fallback = next-lower-lr arm in S (or none). Report only:
onset update and lr x onset for each unstable arm.

FS-BUDGET-v1. Scientific GPU jobs: J1, J2, J3, J3b (only if J3 is FAIL, never if INCONCLUSIVE),
J4: max 5. Infra retries <= 1 per job and <= 2 in total, all recorded; the same blocker 3 times
-> BLOCKED. Formal horizon = screened horizon (no extrapolation). Anything beyond this needs
protocol v2 and user approval.
```

## How the rules are executed

- The thresholds above are copied into `qualification_rule`, `screen_rule` and
  `selection_rule` in the JSON. They are applied by
  `earthdelta.static_adapter.evaluate_fs_qualification`, `decide_lr_screen` and
  `decide_formal_fs`. These are pure functions of the recorded numbers.
  `scripts/r4_fs_decide.py` recomputes every decision on CPU from each process's
  raw files and runs from the job's own source snapshot.
- Real processes refuse to start unless all of the following hold:
  - the protocol hash matches and the config id is declared;
  - the command line equals the declared configuration in every declared field;
  - the S0 PASS certificate is consumed first. It is checked against the file hash,
    the committed PASS, the checkpoint, the normalization identity, the torch
    version, and the hashes of the four S0-bound modules actually imported;
  - `xformers==0.0.27` imports before the model is built;
  - the built model is `stormer.models.hub.stormer.Stormer`, so the SDPA fallback is
    refused.

  Every admission row is certified by the consumer (`certify_admission_for_fs`)
  with a fresh content re-hash before any sample is read.
- J3/J3b use one config id for all four replicas, as FS-SELECT-v1 (1) requires.
  The decider re-checks that the recorded argv are identical except `--device`/`--output-dir`.
- Negative controls in J1 are four known-bad historical adapters. They are scored
  with FS-QUAL-v1 Q1–Q6, with the Q1 horizon set to the historical record's own
  length so Q1 cannot reject them trivially. Each must be REJECTED by at least one
  of Q2–Q6. If any is not rejected, everything stops and the evaluation code is
  considered buggy.
- The merge-equivalence tolerance (merge check only, never S0) is frozen as the
  existing `record_post_freeze_reference` defaults:
  `atol = max(1e-5, 1.5e-3 * delta_max_abs)` and `rtol = 1e-5`. Zero-edit uses
  atol 0. Continuation after the hold must be exact and must differ from F0.

## Inputs (hashes in the JSON)

- **Certified bank_fit admission** (`../admission/bank_fit_admission.json`): the
  same 8 issue times as pt-e8y7ib01, in the same order. It has history 2 steps,
  a 72h lead, and content certificates covering t-12h to t+72h for every row.
- **Holdout panel** (`../admission/holdout_panel_admission.json`): each training
  issue + 288h, never trained on.
- **S0 certificate**: `run_20260923T_fp01_trace/.../s0-gate-20260923t174135587010z/s0_gate_result.json`.
- **Certified overlay**: `artifacts/ed-sprint8h-20260922T035313Z-rerun3/cuda_site`
  (torch 2.3.1, xformers 0.0.27), with the same PYTHONPATH order as the S0 PASS
  job pt-g3344e9z.
- **Initial adapter digest**:
  `2d0b92da4782f3e708e7ca1fb0b406b26ba070352cc52cb4501b2009c92beffa`.

## Findings carried from forensics (not decision inputs)

These are recorded in `../forensics/historical_fs_forensics.json` and
`../ledger/fs_attempts_ledger.json`:

1. **All four pt-e8y7ib01 replicas blew up.** Their maximum sliding 8-update means
   were 47.7x, 144.7x, 27.3x and 28.3x losses[0].
2. **The first shared trigger was update 105 on iss_d8ff9d3ce61a2f46**
   (2020-01-26T06). This trigger is shared WITHIN pt-e8y7ib01 because its replicas
   agree to 4 decimals until about update 99–105. It is not a data defect: across
   the 12 historical formal replicas, the first >1.25 event falls on 5 different
   issues at updates 100–273. The older jobs also ran with TF32 on, and their
   replicas diverged by updates 54–72.
3. **Onset is uniform at lr=1e-2.** By the E_k rule, every one of the 12
   historical formal replicas has its onset at epoch 10–11 (updates 80–88).
4. **Lower lr may only delay the instability.** Adam moves each parameter by about
   lr per update, so a lower lr may push the onset past 256 updates rather than
   remove it. Because the screen and the formal fit share H=256 exactly, "no
   instability within 256 updates" is the correct bar for this Fs. No claim is
   made beyond 256.

## Limitations

- **Split label.** 2020 rows carry `split_id='test'` (YEAR_SPLITS) while being
  admitted as bank_fit, as the historical Fs runs were. The Fs exposure interval
  is `fs_exposure_support_interval_utc` in the JSON, and later policy_dev/confirm
  selection must exclude it with purge. The full split freeze remains FP-02/FP-05 work.
- **S0 scope.** S0 certifies the forward pass only. Training uses the xformers
  backward, which J1 exercises but S0 does not certify.
- **This dispatch.** Only J1 and J2 are run in this dispatch. J3, J3b and J4 need
  separate authorization.
