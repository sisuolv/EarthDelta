# Execution status — 2026-09-25

The committed code and tests cover the FP05 lead-specific scoring correction,
fail-closed scorer/source/cache binding, WBX year authorization, and FP06
first-match decisions. The targeted CPU regression currently passes 48 tests
with one NVML warning.

The RTX 5090 hardware smoke passed. Its native image lacks `xformers`, so the
official Stormer parity path remains blocked. An isolated `timm` overlay was
enough for an engineering-only SDPA bridge forward/backward smoke; this does
not close S0 or authorize Stage 4.

Three standard/tidal H100 compatibility probes were submitted through ACP but
remained in `STARTING` with no worker allocation during this execution window.
The H100 cluster was saturated by other jobs. No formal Fs, bank, or candidate
cache run was started on 5090, and no scientific novelty or weather-utility
claim is upgraded by these probes.

The next authorized experiment is a separately hash-bound H100 Fs stability
grid with independent seeds and lower learning rates. It must pass the full
Q1–Q6 holdout checks and reload/binding checks before any Fs is frozen or used
by Stage 4.
