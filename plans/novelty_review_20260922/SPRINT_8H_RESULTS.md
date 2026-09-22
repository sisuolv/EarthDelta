# Eight-hour novelty and feasibility sprint: final record

Run: `ed-sprint8h-20260922T035313Z` (base `d749a1c62521226df857587e08f7d067b0f15355`)

The data admission stage passed for 616 issues across the frozen bank-fit,
qualification, policy-fit, calibration, holdout, and debug roles. CPU contract and
bridge tests passed, and a real CCI H100 probe completed finite Torch/xformers
forward and backward passes.

The formal downstream experiment is intentionally stopped at S0. Four initial H100
runs and the later repaired rerun were deterministic and finite. The latest rerun
(`pt-7ant09uv`) passed one-step official parity with maximum absolute difference
`5.9604644775390625e-06`, but failed the frozen `1e-5` tolerance after 4 steps
(`3.933906555175781e-05`) and 12 steps (`1.3911724090576172e-04`). The state dict,
direct model output, input normalization, checkpoint identity, and bridge internal
zero-edit checks passed. The remaining failure is therefore a deterministic
multistep bridge/reference mismatch, not evidence of an H100 hardware failure.

No expert training, qualification, bank assembly, candidate cache, policy fitting, or
holdout evaluation was run after the failed gate. Consequently this sprint provides
no weather-utility estimate and no empirical novelty effect. The honest status is
`BLOCKED_S0_MULTISTEP_PARITY`; the old fitted-`Fs` divergence finding remains
`MUST_FIX_BEFORE_STAGE4`.

The full GPU logs and tensors remain local under `artifacts/` and are ignored by Git
because they include multi-gigabyte runtime libraries, checkpoints, and data. The
tracked scripts and frozen specification are sufficient to reproduce the run when
those external assets are available.
