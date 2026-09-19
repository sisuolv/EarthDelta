# EarthDelta v5 — paired-response research core

This package continues the EarthDelta v4 research proposal without overwriting the old starter. It is a **tested prototype of mathematical and software contracts, not an end-to-end weather forecasting implementation**.

Read `IMPLEMENTATION_PLAN_CN.md` for the complete research/implementation plan and `SOURCES.json` for primary literature and code-reading records.

## Implemented

- Exact paired-target identity `edited_error = reference_error - edit_response` and weighted quadratic-gain computation.
- Immutable finite edit plans, fixed hold/continuation policy, and artifact-version checks.
- Grouped low-rank residual with actual inactive-branch skipping in a correctness-oriented PyTorch implementation.
- Verified-memory timestamp/version/event checks and a differentiable small gated-delta replay primitive.
- Budget-feasible deterministic selection and post-hoc selection regret.
- Phase-sensitive complex-coefficient diagnostics and latitude-node validation.
- Small paired-target JEPA-style predictor on **precomputed physical features**, with ordered history, EMA target encoders, physical anchors, and gain calibration.

## Not implemented or validated

No real Stormer checkpoint was loaded. There is no ERA5 data loader, real candidate-probe pipeline, raw global weather encoder, spherical-transform/regridding frontend, full AMSE implementation, recurrent latent rollout, production streaming database, WeatherBench-X bridge, or GPU performance validation. `model.py` predicts requested horizons directly; SG-JEPA-style recurrent training is a planned extension. The memory rule is an independently written simplified recurrence, not a claimed reproduction of FLA or StreamTTT. The elementary variance floor is not SIGReg.

The old archive's post-block bridge is not copied here and is **not automatically integrated** with these new modules. Keep its normalization and input contracts when implementing C0–C5 from the plan. The example configuration identifies the intended integration sites and is not an executable weather training recipe.

## Run

Use a compatible PyTorch environment. Do not upgrade an existing Stormer environment blindly.

```bash
cd EarthDelta_v5_kit
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests
PYTHONPATH=. python examples/synthetic_smoke.py
```

For Windows PowerShell set `$env:PYTHONPATH='.'` before the Python commands.

Requirements give a minimal prototype dependency range, not a complete tested CUDA lockfile. `environment.json` records the actual environment used for this package's tests.

## Evidence

`test_results.txt`: **41 CPU tests passed**. Tests cover paired identities, no-edit behavior, masks, gradients through frozen tails, candidate permutation, memory availability and isolation, version invalidation, phase-vs-energy controls, EMA, and model save/restore.

`legacy_recheck.txt`: the old starter's **15 tests independently passed**. This is not an end-to-end combined integration test.

`synthetic_smoke_results.json`: random-array plumbing only. All rows selected `reference`; the small training-loss decrease is NOT evidence of weather skill or useful edit selection. Toy costs are not hardware measurements.

## Modules

```text
earthdelta_v5/
  contracts.py
  paired.py
  rank_groups.py
  memory.py
  selection.py
  spectral.py
  model.py
```

`forward()` of the predictor accepts no future labels. Truth is used only in offline target/loss functions. All shapes are documented in the source. `spectral.py` accepts caller-provided complex coefficients and weights; the caller must supply correct spherical mode multiplicities, valid modes, and normalization.

## License and provenance

Original code in this package is released under MIT; see `LICENSE`. No upstream source, model weights, datasets, or fonts are bundled. Primary sources informed the design; inspecting or citing a repository does not imply its dependencies or numerical implementation have been reproduced. Check upstream code, checkpoint, and data licenses independently before copying or redistributing them.
