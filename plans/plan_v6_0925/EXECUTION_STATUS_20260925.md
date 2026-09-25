# Execution status — 2026-09-25

The tracked code and CPU tests cover the FP05 lead-specific scoring correction,
fail-closed scorer/source/cache binding, WeatherBench-X year authorization, and
FP06 first-match decisions. The latest targeted CPU regression remains 48 tests
passed with one NVML warning.

## Latest ACP evidence

After the user manually closed stale H100 startup jobs, four fresh standard
single-card jobs all received real H100 workers. They failed only because the
base image did not provide `xformers`; this was an image-contract failure, not a
GPU or scheduling failure.

Using the previously prepared pinned overlay (`torch 2.3.1+cu121`, `xformers
0.0.27`), the single-card compatibility probe `pt-ctdnreud` passed. A subsequent
4-GPU standard-resource S0 recheck, `pt-lujnmfhg`, passed the official reference
export and every S0 criterion. The current bridge matched the official reference
at 1, 4, and 12 steps with `max_abs_diff = 0.0` under the frozen `1e-5`
tolerance. The complete receipt index is
`GPU_RECHECK_EVIDENCE_20260925.md`.

This closes the current S0 execution block under the pinned overlay contract. It
does not mean that the unmodified base image is an official-parity environment;
future jobs must record the overlay or use an image with the exact certified
Torch/xformers versions.

## Scientific status

The earlier fitted-Fs formal divergence finding remains open. The dynamic-bank
out-of-sample evaluation did not clear its pre-registered minimum-effect bar,
and FP06/novelty conclusions must not be upgraded merely because S0 now passes.
No expert-bank or Stage 4 run is authorized until a fail-closed divergence gate
or a justified root-cause fix is independently verified.

Large GPU artifacts, data, checkpoints, reference tensors, and job logs remain
outside Git under the AFS run directories named in the evidence index.
