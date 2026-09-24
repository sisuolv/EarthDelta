# S0 Gate Report

**Run ID:** `s0-gate-20260923t174135587010z`
**Executed:** 2026-09-23T17:41:35.587040+00:00
**Hostname:** pt-3e0efa44a87b42d4a886ec60e6162560-worker-0
**Repo Root:** `/mnt/afs/260010168/EarthDelta/artifacts/round2_cci/ed-r4gate0923e-0923174024-32e8c7/source`
**Patch Size:** 4
**Status:** PASS

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
| upstream_parity | PASS |  |
| multistep_reference_present | PASS |  |
| multistep_parity | PASS |  |
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
- Max absolute diff: 0.00e+00
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

*Report generated at 2026-09-23T17:42:59.853194+00:00*