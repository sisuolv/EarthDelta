# GPU / S0 recheck evidence — 2026-09-25

This file is a compact, Git-tracked index of the ACP receipts. Large logs,
weights, datasets, and generated reference tensors remain outside Git under the
AFS paths listed below.

## Code identity

- Repository: `sisuolv/EarthDelta`
- Branch: `audit/round2-review-20260921`
- Code commit at recheck: `85ed929f565191cf428ba8d491000542e55fe49e`
- The recheck did not modify tracked source code.
- The S0 gate kept the frozen parity and rollout tolerance at `1e-5`.

## Scheduling and native-image probe

After the user's manual cleanup of the earlier stale probes, four new single-GPU
jobs were submitted with the known standard contract (`share-cluster`, reserved,
NORMAL, `n6ls.iu.i40.1.4c64g`). Every worker started on an H100 and reported
CUDA availability, but the base image did not contain `xformers`:

| shard | ACP job | observed GPU | result |
|---:|---|---|---|
| 0 | `pt-qg3ebil5` | H100 80GB HBM3, CUDA available | `FAILED`: `ModuleNotFoundError: xformers` |
| 1 | `pt-wqhw710g` | H100 80GB HBM3, CUDA available | `FAILED`: `ModuleNotFoundError: xformers` |
| 2 | `pt-edomquse` | H100 80GB HBM3, CUDA available | `FAILED`: `ModuleNotFoundError: xformers` |
| 3 | `pt-sr6n07b3` | H100 80GB HBM3, CUDA available | `FAILED`: `ModuleNotFoundError: xformers` |

Per-worker receipts are under:

`/mnt/afs/260010168/acp_runs/ed-r6-h100-4single-20260925T113532Z/`

This establishes that scheduling and H100 hardware work. It also establishes
that the unmodified base image is not an official-parity environment.

## Overlay compatibility probe

An isolated single-card probe using the previously prepared AFS CUDA overlay
completed successfully:

- ACP job: `pt-ctdnreud`
- Run directory: `/mnt/afs/260010168/acp_runs/ed-r6-h100-overlay-20260925T114520Z/`
- Torch: `2.3.1+cu121`
- xformers: `0.0.27`
- GPU: `NVIDIA H100 80GB HBM3`, one visible device, 79.179 GiB
- xformers forward/backward: finite
- official Stormer forward/backward: finite
- official repeat max absolute difference: `0.0`
- Probe result SHA-256: `a274a272393cfc616f436092dc7b797d8f9b1c490a44bf08677721da5ddb315e`

## Full 4-GPU S0 gate

The current source was then run through the full export + gate path:

- ACP job: `pt-lujnmfhg`
- Resource: `n6ls.iu.i40.4.32c512g` (4 H100)
- Run directory: `/mnt/afs/260010168/acp_runs/ed-r6-s0-overlay-recheck-20260925T115020Z/`
- Gate JSON: `/mnt/afs/260010168/acp_runs/ed-r6-s0-overlay-recheck-20260925T115020Z/s0/gate_output/s0-gate-20260925t115202730947z/s0_gate_result.json`
- Gate status: `s0_gate_pass=true`
- Gate JSON SHA-256: `4aab593d3326f754b866815e9a5d0a15ff1336a1f51079ba38689d4251949670`
- Torch: `2.3.1+cu121`
- GPU: `NVIDIA H100 80GB HBM3`
- Official upstream parity: `PASS`, max absolute difference `0.0`
- Multistep parity: `PASS`
  - 6h / 1 step: `0.0`
  - 6h / 4 steps: `0.0`
  - 6h / 12 steps: `0.0`
- Identity, checkpoint SHA-256 binding, normalization binding, strict load,
  zero-edit consistency, finite outputs, and no-state-leak criteria: all `PASS`.

The gate report is:

`/mnt/afs/260010168/acp_runs/ed-r6-s0-overlay-recheck-20260925T115020Z/s0/gate_output/s0-gate-20260925t115202730947z/S0_GATE_REPORT.md`

## Interpretation limits

This evidence closes the current official S0 execution gate under the pinned
Torch/xformers overlay contract. It does not prove Fs training stability,
expert-bank out-of-sample utility, WeatherBench-X superiority, or novelty. The
previous formal Fs divergence finding remains an explicit scientific gate, and
Stage 4/candidate-cache work must remain stopped until that issue is either
resolved or enforced by a fail-closed exclusion gate.
