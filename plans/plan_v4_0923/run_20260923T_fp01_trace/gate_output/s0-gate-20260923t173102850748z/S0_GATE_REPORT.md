# S0 Gate Report

**Run ID:** `s0-gate-20260923t173102850748z`
**Executed:** 2026-09-23T17:31:02.850816+00:00
**Hostname:** pt-57755fbbd30648e3bbc35d3d345d3848-worker-0
**Repo Root:** `/mnt/afs/260010168/EarthDelta/artifacts/round2_cci/ed-r4gate0923c-0923172842-5fe203/source`
**Patch Size:** 4
**Status:** FAIL

## Environment

- PyTorch: 2.3.1+cu121
- CUDA available: True
- Device: NVIDIA H100 80GB HBM3
- Normalization policy: official_zero_diff_mean
- Normalization digest: `3e0b216bfbf34ab0`

## Gate Criteria Results

| Criterion | Status | Notes |
|-----------|--------|-------|
| identity_config_bound | PASS |  |
| source_identity_match | PASS |  |
| variable_coordinate_identity_match | PASS |  |
| ckpt_sha256_bound | PASS |  |
| normalization_parity | PASS |  |
| manifest_identity_match | PASS |  |
| raw_input_binding | PASS |  |
| strict_load_zero_diff | PASS |  |
| version_identity_match | PASS |  |
| input_norm_binding | PASS |  |
| upstream_parity | FAIL |  |
| multistep_reference_present | PASS |  |
| multistep_parity | FAIL | Error: Multistep numeric parity failed: {'6h_1step': 'Num... |
| zero_edit_equals_official | PASS |  |
| no_state_leak | PASS |  |
| outputs_finite | PASS |  |

## Details

### Checkpoint Identity
- Computed SHA-256: `7fde884ec85f4999d4bcb02d55ed72c142912071956083ebee08d3d86311b695`
- Expected SHA-256: `7fde884ec85f4999d4bcb02d55ed72c142912071956083ebee08d3d86311b695`
- Match: True

### Upstream Parity (vs Official xformers Stormer)
- Available: True
- Max absolute diff: 8.57e-03
- Tolerance: 1e-05

### Internal Consistency (controlled_rollout vs forward_validation)

Note: This is bridge-internal consistency, NOT upstream parity.

- 6h_1step: max_diff=0.00e+00, passed=True
- 6h_4step: max_diff=0.00e+00, passed=True

## RMSE Sanity Check (Informational Only)

**Note:** These values are for reference only and do NOT affect gate pass/fail.

| Lead Time | RMSE Z500 | Historical Reference |
|-----------|-----------|---------------------|
| 6h | 27.0 | ~30 |
| 24h | 52.7 | ~80 |

*Report generated at 2026-09-23T17:32:24.047731+00:00*