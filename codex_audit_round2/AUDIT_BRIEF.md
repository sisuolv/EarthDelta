# EarthDelta — Round-2 Audit Brief (2026-09-21)

## Purpose

Your (Codex) round-1 audit package `EarthDelta_Codex_Execution_Package_20260921`
(hereafter "pkg_21") audited commit `4fe55a7af90ea92f62a3232a571af92bfbd6114d`
and returned decision `CONDITIONAL_CONTINUE_P0_ONLY`, unlocking only task
**P0-01**. Since then we have executed **P0-01, P0-02, and P0-03** from your
own task DAG (`codex_tasks.json`), each gated sequentially, each independently
verified against live code and a real test run rather than taken on
self-report. This brief describes exactly what changed and asks you to audit
that work — both "did the fix actually fix the finding" and "did the fix
introduce anything new."

We are NOT asking you to redo the round-1 audit from scratch. We are asking
for a **round-2 audit with two parts, and the second part is the more
important one**:

1. Diff-focused code audit: verify the P0-01/02/03 work, and flag anything
   in the broader pipeline that is now a landmine for the upcoming P0-04
   (survival spec + reference bank) and the eventual P0-05 go/no-go
   experiment.
2. **An independent re-examination of whether the underlying research idea
   is actually novel and worth pursuing at all.** Everything in part 1 —
   every gate, every fixed bug, every fail-closed check — only verifies that
   the *implementation* faithfully matches the *design*. None of it answers
   whether the design itself is sound. That is a separate and more
   foundational question, and so far it has only been checked by us
   internally, never independently by you. See the dedicated section below;
   do not treat it as secondary to the code findings — if anything, resolve
   it first, since a negative answer here would make the P0-02/P0-03 code
   fixes moot regardless of how well-executed they are.

## Novelty and value re-assessment (please treat this as the primary ask)

**What the idea claims**: at model-issuance time, using only information
that is legitimately available historically (no leakage of future ground
truth), jointly learn (a) a reference-model-dependent error forecast `e0`
and (b) a per-candidate-edit response `du`, combine them via the *exact*
FSO (forecast sensitivity to observations) quadratic identity
`gain = 2⟨e0,du⟩ − ‖du‖²`, and select a budget-constrained parameter edit
(allowing off-diagonal interaction terms `H` between candidate edits, not
just independent per-edit scoring) that maximizes that gain. The claim is
that no prior published work occupies exactly this pipeline.

**What we did internally** (not an independent check — disclosed here so
you can stress-test it rather than take it on faith): earlier in this
project we ran five internal literature surveys (labeled A–E, ~200 searches,
completed 2026-09-19), plus two additional targeted external searches the
next day, and concluded the novelty claim holds. The nearest neighbors we
found were: Aurora LoRA (training-time static rollout fine-tuning, not
issuance-time selection), WeatherPEFT (arXiv 2509.22020, a PEFT benchmark
with no online selection), Adapter Banks (arXiv 2609.17042, RL-return-based
selection with no response prediction), and VI-MoLE (arXiv 2608.02528,
scalar risk scoring, no vector `du`/`e0`/`H` structure). We concluded the
real risk to this project is **headroom** (whether the achievable gain is
large enough to matter) rather than **occupancy** (whether someone already
did this) — and that conclusion is what licensed us to keep investing
engineering effort in P0-01 through P0-04 rather than stopping to
re-question the premise.

**What we want from you, independently**:

1. Re-read your own round-1 research materials
   (`research/NOVELTY_AUDIT.md`, `research/RELATED_WORK.md`,
   `research/EIGHT_DIRECT_ANSWERS.md`, `research/FINAL_RESEARCH_DECISION.md`,
   `research/OUTPUT_CORRECTION_CHALLENGE.md`, and
   `research/literature_sources.json` in your own pkg_21 package) plus the
   core hypothesis section of
   `plans/plans_v1_0919/v6_draft/research_spec_v6.yaml` — do NOT simply
   restate your round-1 conclusion. Independently re-derive whether you
   still agree no prior work occupies this exact pipeline, using whatever
   literature knowledge/search capability you actually have. If your
   knowledge cutoff or lack of live search means you cannot meaningfully
   update on work published after a certain date, say so explicitly as a
   stated limitation rather than silently presenting a stale answer as
   current.
2. Independently sanity-check our four "nearest neighbor" claims above —
   do you agree Aurora LoRA / WeatherPEFT / Adapter Banks / VI-MoLE are
   genuinely non-occupying, or would a careful reading of any of them
   actually undercut the novelty claim more than we think?
3. Go beyond mere occupancy/novelty and assess **mechanistic plausibility**:
   is there an a priori reason to expect that jointly learning `e0` and
   `du` and combining them via the exact FSO quadratic form captures
   meaningfully more usable signal than a simpler direct/static baseline
   (or than the output-correction challenger baselines already specified in
   `experiments/BASELINES.md` / `research/OUTPUT_CORRECTION_CHALLENGE.md`)?
   Or does the idea's soundness rest entirely on an unverified assumption
   that won't be tested until the real P0-05 oracle-ceiling experiment runs?
4. Evaluate whether the **P0-05 experiment design itself**
   (`experiments/P0_SURVIVAL_EXPERIMENTS.md`, the "full finite-candidate
   oracle ceiling" go/no-go test) would actually be a fair and decisive test
   of this idea's value if it does get run, or whether you now see a design
   flaw that would make even a clean PASS/FAIL result on that experiment
   uninformative about the real question.
5. Give an explicit recommendation: should we continue investing in
   P0-04 (survival spec + reference bank training) on the current premise,
   or should something about the premise itself be re-examined or the
   project be paused/redirected before further engineering investment goes
   in? Do not hedge this into a non-answer — if you are genuinely uncertain,
   say what specific piece of evidence would resolve the uncertainty and
   how to get it before more resources are spent.

## Repository state

- Round-1 audited commit: `4fe55a7af90ea92f62a3232a571af92bfbd6114d`
- Current HEAD: `fb767f7f6efbc428be39c9ad84f5905331d6e40f`
- Commits made since round-1, in order:
  - `607fad5` — P0-02: unify FSO quadratic-gain implementations into
    `earthdelta/metrics_contract.py`; harden `selection.py` finite-candidate
    validation; UTC-fix `make_splits.py`; A11 disposition note.
  - `fb767f7` — P0-03: true upstream-parity verification path for the S0
    gate (`scripts/export_upstream_reference.py`, rewritten
    `scripts/s0_gate.py`, `stormer_bridge.py` SHA-256/digest/gradient
    changes).
- Neither commit has been pushed to the remote (`sisuolv/EarthDelta`) yet.
  Only local history.
- Working tree has three untracked paths not part of either commit and not
  in scope for this audit: `artifacts/` (our own run artifacts),
  `checkpoints/run_multi_year_pull.sh` (a background data-pull driver
  script, currently running — do not assume any test needs it),
  `plans/plans_0921/` (your own round-1 package, extracted read-only).
- Test suite: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 -m pytest tests/ -q -rs`
  → **290 collected, 283 passed, 7 skipped, 0 failed** (independently
  re-run by us, not copied from a subagent's self-report). The 7 skips are:
  3 memory-gated (checkpoint loading needs 12–24GB, this dev container has
  8.6GB), 2 CUDA-gated, 2 gated on an upstream-reference artifact that can
  only be produced by a real GPU+xformers run (see below — this is expected,
  not a gap to be closed on this machine).

Diff sizes for scope calibration:
- `607fad5`: 12 files changed, 1692 insertions(+), 42 deletions(-).
- `fb767f7`: 8 files changed, 2499 insertions(+), 578 deletions(-).

To reproduce our diff view yourself: `git diff 4fe55a7..fb767f7` from the
repo root, or the two commits individually.

## What P0-02 (`607fad5`) claims to fix — please verify each

Your round-1 finding **A04** identified 5 mutually inconsistent
implementations of the FSO quadratic-gain identity
`gain = 2⟨e0,du⟩ - ‖du‖²`:

1. `earthdelta/paired.py:55-75` (weighted **mean**, weights normalized to
   sum to 1)
2. `earthdelta/heads.py:335` `ComposedPredictionHead.forward` (unweighted
   mean, **no Q-weight support**, yet trained against #1's output)
3. `earthdelta/heads.py:416-418` (unnormalized **sum**, coefficient space)
4. `earthdelta/geometry.py:52,66` `ResponseGeometry` (explicitly documented
   "no automatic normalization" — weighted **sum**)
5. `earthdelta/selection.py:87-89` `quadratic_gain_numpy` (same
   unnormalized-sum convention as #4)

The fix converges all five onto one canonical module,
`earthdelta/metrics_contract.py` (new, 297 lines), with these convention
assignments (please sanity-check the reasoning, not just the internal
consistency):

- `paired.py` and `heads.py:335` → `WEIGHTED_MEAN` (training-target
  semantics; #2 gained real Q-weight support it previously lacked — for
  uniform weights the claim is the output is bit-identical to the prior
  `.mean(-1)` behavior).
- `heads.py:416-418` → no separate convention; weights are already baked
  into the benefit/Gram matrices at that point (coefficient space).
- `geometry.py` and `selection.py` → `WEIGHTED_SUM` (weights embedded at
  construction time; matches scipy-optimization usage in `selection.py`).

Also in this commit: your finding **A03** (`selection.py`'s
`_plan_from_finite_candidates`, lines ~194-250, missing finiteness/bound/
max_active/max_candidates checks) — the fix adds all four checks and wires
`bound` into that code path for the first time. One existing test
(`test_earthdelta.py::test_unified_select_finite_candidates`) had an
explicit `bound=0.5` added because its candidates use 0.5-magnitude
coefficients that the newly-enforced default bound (0.25) would now reject
— we believe this is a legitimate adjustment to newly-enforced validation,
not a weakened assertion, but please check independently.

Also: your finding **A06** (`make_splits.py` naive-datetime /
host-timezone-dependent `.timestamp()` calls) — fixed with UTC-aware
datetimes throughout plus a new `availability_source` field
(`reanalysis_retrospective` / `observed_first_seen` / `scenario`). Please
check whether the field is actually populated correctly for every code path
that constructs a split record, not just declared.

Also: your finding **A11** (`research_spec_v6.yaml` pre-written unproven
claims and the `ge_2_to_3_percent_Z500_72h` gate) — NOT rewritten in place
(deliberately, to preserve the historical record); instead a new pointer
note `plans/plans_v1_0919/v6_draft/A11_DISPOSITION_NOTE.md` was added
downgrading the claims to hypotheses H1-H3 and the gate to
historical-reference-only. Please confirm this is an acceptable disposition
or whether you consider in-place correction necessary.

## What P0-03 (`fb767f7`) claims to fix — please verify each

Your round-1 finding **A01** (the most severe): `scripts/s0_gate.py`'s
"zero_edit_equals_official" gate never executed the real
`reference/stormer` code — it compared two of EarthDelta's own internal
paths against each other and monkeypatched a fake `climate_learn.*`
namespace pointing back to the local implementation.

The fix:
- New `scripts/export_upstream_reference.py` — intended to run inside a
  real GPU+xformers container, using the actual official
  `reference/stormer/stormer/models/hub/stormer.py` architecture (real
  `xformers.ops.memory_efficient_attention`, not the SDPA substitute our
  bridge uses) and the real `forward_validation` rollout logic, against a
  pinned input tensor. We independently confirmed the official code's hard
  xformers dependency (`stormer.py:4`,
  `from xformers.ops import memory_efficient_attention, unbind`, no
  fallback) and confirmed that running this script on our xformers-less CPU
  dev container correctly exits 2 with an explicit `BLOCKED` status rather
  than faking a result or silently substituting SDPA.
- Rewritten `scripts/s0_gate.py` (1059 vs. prior ~800-ish lines — see diff):
  removed auto-`pip install` at import time, removed hardcoded absolute
  paths (`/mnt/afs/260010168/EarthDelta` literals) in favor of
  `Path(__file__).resolve().parent.parent` with an `EARTHDELTA_REPO_ROOT`
  override, removed the hardcoded guessed thresholds
  (`STORMER_PAPER_RMSE_Z500_6H/24H`) from gate-relevant logic (kept as
  informational-only reporting), removed the hardcoded
  `executor_model` string. `gate_criteria` expanded from 3 keys to 7:
  `ckpt_sha256_bound`, `strict_load_zero_diff`, `upstream_parity` (new — the
  actual official-vs-bridge comparison, ≤1e-5), `zero_edit_equals_official`
  (kept, tightened to ≤1e-6, now explicitly re-labeled as
  bridge-internal-consistency, NOT a substitute for `upstream_parity`),
  `normalization_parity`, `no_state_leak`, `outputs_finite`. We confirmed
  all 7 are initialized `False` and the overall pass is
  `all(gate_criteria.values())` — fail-closed.
- `stormer_bridge.py` changes (362-line diff): checkpoint SHA-256
  computation wired into the artifact/version record (addressing the first
  half of your finding **A05** — previously `file_size`/`var_hash` were
  computed but never used, and no weight hash existed anywhere);
  `NormalizationContract.digest` extended to include variable-name order,
  interval keys, and shapes (previously joined `.values()` without keys —
  second half of A05); `controlled_rollout` given a gradient path through
  the injected coefficient tensor plus full-trajectory return, and a guard
  against concurrent reuse of an in-flight rollout state.

**Important limitation you should weigh heavily**: none of this has been
run for real yet. The actual `upstream_parity` numeric verdict (does the
bridge's output match real official xformers output within 1e-5?) requires
an ACP GPU job with xformers installed, which has not happened. Everything
above is a **code-correctness** claim, verified by us only through direct
code inspection, exit-code/BLOCKED-behavior tests, and a CPU test suite run
where GPU-dependent tests skip cleanly. Please treat "S0 gate PASS" as
**not yet determined** — that is intentional, and the next step in our plan
is exactly that GPU run.

## Specific things we want you to check that we may have missed

1. **New-code review, not just fix-verification.** `metrics_contract.py`,
   `export_upstream_reference.py`, and the rewritten `s0_gate.py` are
   substantial new/rewritten files (297, 459, and ~1059 lines respectively).
   A "fix" can introduce its own bugs. Please review them as new code, not
   only as a checklist against the round-1 findings.
2. **Mathematical soundness of the P0-02 convention choices**, not just
   their internal consistency. We reasoned about *why* each site should use
   sum vs. mean, but we have not had an independent check of whether that
   reasoning is actually correct for the underlying FSO/selection math.
3. **Whether any existing test was weakened (not just changed) to make the
   suite pass.** We checked the one `bound=0.5` change ourselves and believe
   it is legitimate, but we have not exhaustively diffed every test file
   line-by-line against pre-P0-02 behavior.
4. **Whether `controlled_rollout`'s new concurrent-reuse guard corresponds
   to a real hazard** in the current codebase, or is a guard against a
   hypothetical that doesn't actually occur anywhere in the call graph
   (over-engineering vs. under-engineering both worth flagging).
5. **Anything in `earthdelta/contracts.py`, `earthdelta/data/pull_wb2.py`,
   or the split/manifest machinery** that would corrupt or bias the
   upcoming P0-04 reference-bank training on 2015-2018 data, given that
   P0-04 has not started and this is your last chance to catch a data-layer
   issue before training work begins on top of it.
6. Whether our disposition of A11 (pointer note, not in-place rewrite) and
   A05 (SHA-256 now computed and used) fully close those findings in your
   framework, or whether you'd keep them open with a narrower residual
   concern.

## What we are not asking this round

- Do not re-derive or re-validate the P0/P1/P2 experiment sequencing itself
  — that framework (`experiments/P0_SURVIVAL_EXPERIMENTS.md` etc.) is
  unchanged and not in scope unless something in this diff structurally
  contradicts it.
- Do not attempt to actually run GPU/xformers-dependent code — you don't
  have that hardware either; static/code-level review plus the CPU test
  suite is the expected verification surface, same as our own process.

## Deliverable format

Please use the same discipline your round-1 package used: every finding
must cite exact `file:line`, be classified by severity, and distinguish
"confirmed by reading the code" from "plausible but unverified." Where
useful, reuse your existing STOP-vocabulary
(`BLOCKED` / `FAIL_IMPLEMENTATION` / `INCONCLUSIVE` / `STOP` / `PIVOT`) so
the output plugs directly back into the same decision framework as round 1.
See `AUDIT_PROMPT.md` in this same directory for the exact task framing and
output schema we'd like.
