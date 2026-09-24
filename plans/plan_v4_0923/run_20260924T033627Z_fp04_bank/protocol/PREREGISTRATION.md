# FP-04 K=4 / rank=4 dynamic expert bank on the certified Fs: pre-registration

Status: PREREGISTERED_BEFORE_FP04_JOBS. No FP-04 GPU job has been submitted.

- **Binding artifact.** The binding artifact is `bank_protocol_v1.json` in this
  directory.
- **Protocol hash.** Every FP-04 job passes that file's SHA-256 as
  `--protocol-sha256`, and the training CLI and `scripts/r4_bank_decide.py`
  refuse any other value.
- **This file.** The JSON pins this file's SHA-256.
- **Decision rules.** They are copied into the JSON verbatim from the block
  below.

## What FP-04 tests

Question: can a K=4, rank=4 dynamic expert bank be trained on the ONE
certified Fs and pass all of the following?

- A pre-registered per-expert qualification.
- An all-or-nothing selection rule.
- An exact assembly-equivalence check.
- A consumer-verifiable registry certificate.

FP-04 can show that a qualified bank exists. It cannot show that the bank is
useful out of sample. That question belongs to FP-05, and delta_min is still
not pre-registered.

## Fixed inputs (all pinned by SHA-256 in the JSON)

- **Fs.** The certified Fs is the FP-03 v2 `certify/fs/` bundle. Its protocol
  is `3d6a3be6...`, verdict FS_SELECTED.
  - Every worker binds the bundle by hash before loading anything, with
    `authorize_certified_fs`.
  - After loading, each worker re-verifies the merged, base, static-adapter and
    version digests against this protocol's pins, with `load_certified_fs`.
  - No worker fits, freezes, or picks up an Fs of its own. The historical
    per-worker Fs re-fit (job pt-cdj1s2le: 4 different Fs digests) is
    structurally refused.
- **Initial bank.** `build_dynamic_bank(1024, blocks 18-23, K=4, rank=4,
  seed=20260921)`.
  - Its bank digest and 4 expert digests were recomputed on CPU before this
    freeze.
  - Every worker must reproduce that bank digest exactly.
- **Training data.** A new 48-issue certified bank_fit admission lies inside
  the already-exposed Fs window 2020-01-01T12Z..2020-07-04T18Z, so it creates
  no new data exposure.
  - Selection was declared before any value was read
    (`../admission/bank_fit_selection_declaration.json`): a 90h stride from
    2020-01-02T00Z, 49 slots, with integrity-only replacement.
  - Slot 47 (2020-06-26T06Z) failed content verification. The cause is one
    `specific_humidity_100` slab at store index 712, at 102 sigma.
  - All 4 declared same-month replacements (+6h..+24h) contain that same index
    and failed too, so the declared rule DROPPED the slot. 1 drop is within the
    declared 5-slot limit.
  - The final admission was re-run fresh: 48/48 admitted, all content
    certificates PASS. A consumer re-hash equals the stored hash for every row.
- **Grouping.** The rule is ed-bank-diversity/1 (disjoint calendar-month
  blocks), recomputed from the final admission.
  - Expert 0 gets 2020-01..02 (15 issues).
  - Expert 1 gets 2020-03..04 (16 issues).
  - Expert 2 gets 2020-05 (7 issues).
  - Expert 3 gets 2020-06 (6 issues).
  - 4 issues are purged at group boundaries.
  - The calendar-only estimate before admission was 15/16/7/7. Expert 3
    lost the dropped slot 47.
- **Probe issue.** The first admitted row, used only for the reload and
  assembly probes.
- **Recipe.** The recipe is FP-03's, used unchanged:
  - Adam (defaults), clip 1.0, seed 20260921 + k;
  - 4 x 6h differentiable rollout with the expert held at a0 = rho = 0.25 for
    all 4 steps, and a 24h objective;
  - H = 32 updates, lr 2.5e-3, with fallback 1.25e-3.

## Rules

```
BANK-QUAL-v1 (per formal expert k; own training group G_k; Fs = pure certified Fs panel, L0 =
untrained singleton k (B == 0), L1 = trained singleton k (a0 = 0.25 for the 4-step hold, then Fs);
per issue, per lead 6/24/72h; float64 objective, no-grad 12-step rollout; epoch = |G_k| updates)
 E0 validity (else INVALID): official backend + torch 2.3.1+cu121 + xformers 0.0.27, TF32 off
    (effective getters), S0 certificate consumed, admission consumer PASS, protocol/config-id
    match, no source drift vs source_at_preregistration, certified Fs authorized by hash AND
    loaded digests (merged/base/static-adapter/version) equal the pins, initial bank digest and
    initial expert digest equal the pins, grouping equals the pin, training order = pinned group,
    record = declared (mode/lr/updates), expert file is the trained expert, L0 == Fs BITWISE on
    every issue and lead of G_k.
 E1 numerics: n_updates == 32; every loss, total grad norm, A and B grad norm finite; traces
    consistent; eligible (ELIGIBILITY_RULES).
 E2 stability: clip_events == 0; every training loss <= 1.25 x that issue's Fs 24h loss; for
    every full epoch k >= 1: E_k <= 1.02 x min(E_0..E_{k-1}).
 E3 own-group direction (HARD): mean_{G_k} L1_24h / mean_{G_k} Fs_24h < 1.00 (strict).
 E4 nonzero response: response at a0 is nonzero, B no longer zero, the expert digest moved, and the
    L1 panel differs from the L0 panel.
 E5 reload: expert_k.pt re-hashes; a FRESH seed bank with expert_k.pt installed has the trained
    digest and reproduces the probe states at steps 1/4/12 exactly (max |diff| == 0.0); the saved
    probe file round-trips exactly.
 E6 isolation: the 3 other experts' digests unchanged and gradient-free; backbone digest
    unchanged (and == certified merged Fs); no leftover hooks.
 Verdict: INVALID if E0 fails or a threshold is missing; else FAIL if any of E1..E6 fails;
 else PASS. REPORT-ONLY (never a gate): own-group 6h and 72h ratios, per-issue 24h ratios,
 per-visit first/last direction, the first-16-update replay of B-J1. No out-of-sample gate.

BANK-SELECT-v1. The formal bank of ONE job (one lr) is selected for assembly iff all 4 experts
are PASS with 4 distinct expert digests and 4 distinct files: BANK_QUAL_PASS_PENDING_ASSEMBLY.
Any FAIL -> STOP_CURRENT_BANK (the whole batch; the passing experts are not kept, not assembled
as a smaller bank, not re-indexed). Any INVALID / missing / duplicated expert -> INVALID. No
substitution, no cross-batch mixing (a B-J2 expert never joins a B-J2F bank). A BANK-QUAL-v1
failure is never an exclusion reason; FORBIDDEN_EXCLUSION_REASONS stay refused. A
STOP_CURRENT_BANK records fallback_authorized = True iff EVERY failed expert has E2 = FAIL
(other criteria may fail alongside E2); otherwise False.

BANK-SHORTSTEP-v1 (B-J1, machinery only). Per worker: E0-style validity, n_updates == 16,
finite, A grad exactly 0 and B grad > 0 at update 0, A grad > 0 for every later update, L0 ==
Fs bitwise, expert moved, nonzero response, isolation (E6), exact reload (E5). Across workers:
all 4 present, ONE certified Fs digest set, ONE initial bank, ONE grouping. Profile: real CUDA
profile of K=4/rank=4, no error, decision keeps K=4/rank=4 (no shrink). Horizon: the full
4-step hold is differentiable, no activation checkpointing, no fallback. Both measurements
leave the bank digest unchanged (they run on a deep copy). No loss trend and no L1/Fs ratio is
read from B-J1. PASS or STOP.

BANK-CERTIFY-v1 (B-J3; assembly A1-A5, all EXACT, max |diff| == 0.0, no tolerance)
 A1 each assembled singleton k == the same-process source bank (seed bank + only expert k)
    at probe steps 1/4/12;
 A2 each assembled singleton k == the probe states the training process saved;
 A3 whole-bank zero edit == Fs exactly, and != F0 (discriminating);
 A4 per expert: after the 4-step hold the rollout equals Fs continued from the edited state
    at every step to 12, differs from an F0 continuation at every step, nonzero edit at hold;
 A5 the reloaded bank.pt file sha256, bank digest and expert digests equal the assembly
    record's, and the expert digests equal the formal decision's.
 Justification for exactness: ExpertLoRA.forward_dense adds expert_out * coefficients[:, k],
 an exact float zero for an inactive expert. Plus: bank.pt re-hashes and reloads to the
 recorded digests, each bank slice == its expert file, the registry validates and binds
 Fs/bank/normalization/inputs/protocol/source hashes, and registry.verify_bank_bundle PASSES
 on the bundle BEFORE it is published. Else STOP_ASSEMBLY.

BANK-BUDGET-v1. GPU jobs: B-J1 (4 short-step workers, 16 updates, lr 2.5e-3, then profile on
cuda:0 and horizon check on cuda:1); B-J2 (formal, H=32, lr 2.5e-3, 4 workers; only after
B-J1 PASS); B-J2F (formal, H=32, lr 1.25e-3, all 4 experts retrained together; only if the B-J2
decision is STOP_CURRENT_BANK with fallback_authorized, i.e. EVERY failed expert failed E2
stability; never after INVALID, never after a STOP with any E3-only / E4 / E5 / E6-only
failure, and no other fallback of any kind); B-J3 (bank_assemble on cuda:0, then bank_verify on
cuda:1 in an independent process; only after BANK_QUAL_PASS_PENDING_ASSEMBLY); at most 1 infra
retry in total. Cap 7 GPU jobs; expected 3 (B-J1, B-J2, B-J3). Stops: B-J1 STOP -> BLOCKED;
B-J2 INVALID -> BLOCKED; B-J2 STOP_CURRENT_BANK without fallback_authorization (e.g. an E3
failure) -> final STOP_CURRENT_BANK (no B-J2F, no more updates, no other fallback); B-J2 and
B-J2F both STOP_CURRENT_BANK -> STOP_CURRENT_BANK (no third lr, no per-expert retry); B-J3 STOP_ASSEMBLY -> STOP_ASSEMBLY. No further job without the
coordinator.
```

## Why these thresholds (fixed before any FP-04 output exists)

- **Why not FS-QUAL-v2's Q4 and Q6.**
  - Q4 (max r_i < 1 and m <= 0.99) was calibrated for an always-on
    coefficient-1.0 static reference.
  - A bank expert acts at a0 = 0.25 for 24h only, so a 1% own-group
    improvement bar has no calibration behind it.
  - Q6 (out-of-sample harm) would exclude an expert for its gain. That is
    FORBIDDEN_EXCLUSION_REASONS territory, and it is FP-05's job.
  - Only the pattern is reused: Q1/Q2 stability through
    `static_adapter.training_stability`.
- **Why not the shared-F0 sprint thresholds.** Its 1.001/1.05 thresholds are
  not borrowed (STOP_CONDITIONS section 3).
- **Why E3 is strict < 1.00.** It is the weakest non-vacuous direction claim:
  the expert must improve its own training regime on average. It is a hard
  gate, with no margin tuned on data.
- **Why E2's epoch is the group size.** One epoch is one pass over the
  expert's own issues.
  - Groups have 15/16/7/6 issues, so H=32 gives 2/2/4/5 full epochs.
  - The epoch-onset factor 1.02 and the per-visit bound of 1.25 are FP-03's.

## Budget-relevant expectations (not decision inputs)

- **B-J1 and B-J2 overlap.** The first 16 updates of every B-J2 expert are
  bitwise the B-J1 updates, since the seed, data order and recipe are
  identical. FP-03 showed bitwise determinism across GPUs.
  - B-J1 therefore reveals no formal outcome that B-J2 would not produce anyway.
  - B-J1's decider reads no training-effect number.
- **Profile.** Historical real profile: K=4/rank=4, peak 13.4 GiB,
  0.34 s/update (H100). Well inside the 80 GiB x 0.8 / 30 s budget.
- **Known risk (flagged, not tuned away).**
  - An expert acts at 0.25x strength, and the larger groups get only about 2
    visits per issue in 32 updates (the Fs had 4 at full strength).
  - The own-group 24h improvement is therefore expected to be small, and the
    E3 margin thin.
  - Refinement approved by the coordinator BEFORE this freeze (nothing had been
    submitted): the halved-lr fallback exists for instability (E2), the FP-03
    precedent. A too-weak expert (E3) would only get weaker at half the lr, so
    a STOP that is not all-E2 gets no fallback. It is a final, informative
    STOP_CURRENT_BANK, and no untested alternative (e.g. more updates) is
    pre-registered.

## Limitations

- **Exposure.** The 48 bank_fit issues lie inside the already Fs-exposed 2020
  window. They, and the 2019 FP-03 holdout, must be purged from any
  policy_dev/confirm use.
- **Split labels.** The 2020 rows carry split_id 'test' while admitted as
  bank_fit (as in FP-03).
- **Store defect.** The 2020 store has a bad `specific_humidity_100` slab at
  index 712 (2020-06-27T00Z, 102 sigma). Any later window containing that
  step must re-certify content.
- **S0 scope.** S0 certifies the forward pass only. The training backward is
  exercised, not certified.
- **What PASS means.** FP-04 PASS means "a qualified, exactly-assembled,
  registry-certified K=4 bank exists". It does not mean "the bank is useful".
  delta_min and any out-of-sample gain are FP-05.
- **Historical bank jobs.** These include the pt-cdj1s2le-era r3 jobs, with
  per-worker Fs re-fits and 3/2/2/1 groups. They are ledger-only and are not
  evidence for any FP-04 result.
