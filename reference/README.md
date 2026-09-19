# EarthDelta reference clones

Cloned 2026-09-19 with `_clone_refs.sh` (direct git; fallbacks: gh-proxy.com / ghfast.top / ghproxy.net / codeload zip). Machine-readable manifest: `_manifest.json` (HEAD, date, license per repo). Nothing here is modified; all are shallow clones except `stormer` (full history, pinned commit == main HEAD).

| dir | upstream | why | key entry points |
|---|---|---|---|
| stormer | tung-nd/stormer @ 58dfee5 (2025-03-17, MIT) | primary frozen backbone | `stormer/models/hub/stormer.py` (Block, MemEffAttention, adaLN), `stormer/models/iterative_module.py` (rollout + normalization contract), `inference.py` (69-var order, ckpt URLs), `normalization_constants/`, `stormer/data_preprocessing/` (download_wb2 → regrid_wb2 → process_one_step_data) |
| aurora | microsoft/aurora (MIT) | second backbone; shipped per-rollout-step LoRA precedent | `aurora/model/lora.py` (LoRA, LoRARollout modes single/from_second/all), `aurora/model/swin3d.py` (lora_qkv/lora_proj usage), `aurora/rollout.py` |
| ClimaX | microsoft/ClimaX (MIT) | Stormer predecessor; 1.40625° preprocessing | `src/data_preprocessing/nc2np_equally_era5.py`, `regrid.py` |
| weatherbenchX | google-research/weatherbenchX (Apache-2.0) | standard evaluation | `weatherbenchX/{data_loaders,metrics,aggregation.py,weighting.py,binning.py}`, `evaluation_scripts/` |
| weatherbench2 | google-research/weatherbench2 (Apache-2.0) | data conventions, regridding module origin | `weatherbench2/regridding.py`, docs |
| torch-harmonics | NVIDIA/torch-harmonics (BSD-3) | spherical harmonic diagnostics | `torch_harmonics/quadrature.py` (`precompute_latitudes` → colatitudes), `torch_harmonics/sht.py` (`RealSHT(nlat,nlon,lmax,mmax,grid,norm)`) |
| flash-linear-attention | fla-org/flash-linear-attention (MIT) | gated delta rule reference | `fla/layers/gated_deltanet.py`, `fla/ops/gated_delta_rule/naive.py` (pure PyTorch) |
| peft | huggingface/peft (Apache-2.0) | LoRA reference semantics | `src/peft/tuners/lora/` |
| graphcast-amse | csubich/graphcast branch `amse` (Apache-2.0) | modified spherical-harmonic loss | diff against `graphcast/` |
| graphcast | google-deepmind/graphcast (Apache-2.0) | upstream for the diff | — |
| le-wm | lucas-maes/le-wm (MIT) | LeWorldModel JEPA (action encoder / predictor split) | `jepa.py` |
| sg-jepa | sg-jepa/sg-jepa (MIT) | Semigroup-JEPA official code | `sg_jepa/`, `train.py`, `evaluate_control.py` |
| StreamTTT | zeyun-zhong/StreamTTT (Apache-2.0) | streaming TTT / fast-weight memory | `streamttt/` |
| CoMoL | DCDmllm/CoMoL (no LICENSE file) | shared-A/B + core-space dynamic LoRA mixture | `src/mocorelora/layer.py` |
| DISeL | alizindari/DISeL (Apache-2.0) | input-dependent low-rank gating ("when to adapt") | — |
| PEAR | FinJun/PEAR (MIT) | decision-focused learning (tangent-space projected error) | — |
| Solver-in-the-Loop | tum-pbs/Solver-in-the-Loop (MIT) | learned-correction-in-solver protocol | — |
| geps | itsakk/geps (no LICENSE file) | PDE context low-rank adaptation | `geps/model/layers.py`, `geps/datasets/` |

## Batch 2 (recommended by the literature survey, cloned with `_clone_refs_batch2.sh`)

| dir | upstream | why |
|---|---|---|
| JEPA-Anything | Gen-Verse/JEPA-Anything (arXiv 2609.20800, 2026-09-17) | JEPA that already touches weather + intervention-effect prediction; reusable projections/objectives/diagnostics |
| lejepa | rbalestr-lab/lejepa (arXiv 2511.08544) | SIGReg regularizer (no EMA / stop-grad) used by LeWM and SG-JEPA |
| text-to-lora | SakanaAI/text-to-lora (arXiv 2506.06105) | LoRA bank + hypernetwork with coefficient/weight distillation: the D0 baseline pattern |
| lorahub | sail-sg/lorahub (arXiv 2307.13269) | gradient-free (CMA-ES) coefficient search over a frozen LoRA bank: search-vs-amortize baseline |
| AdaMerging | EnnengYang/AdaMerging (arXiv 2310.02575) | label-free merging coefficients via test-time entropy: non-response "no future truth" baseline |
| fusion_bench | tanganke/fusion_bench | Fisher / RegMean / TIES / DARE / AdaMerging implementations in one toolbox (Gram-type merging lineage) |
| CoDA | yuan-yin/CoDA (arXiv 2202.01889) | context -> weights hypernetwork for parametric dynamical systems |
| zebra | LouisSerrano/zebra (arXiv 2410.03437) | in-context, gradient-free PDE adaptation baseline |
| Weight2Token | xiaolonghan2000/Weight2Token (arXiv 2603.15990) | LoRA canonicalization (QR -> SVD) and adapter performance prediction |
| SHINE | MuLabPKU/SHINE (arXiv 2602.06358) | in-context hypernetwork reusing frozen backbone parameters (controller design pattern) |

## Batch 3 (recommended by surveys C/D, cloned with `_clone_refs_batch3.sh`)

| dir | upstream | why |
|---|---|---|
| WeatherPEFT | ShileiCao/WeatherPEFT (arXiv 2509.22020, ICLR 2026) | PEFT baselines (LoRA/DoRA/AdaptFormer/VPT) on Aurora + ERA5 prep/normalization: fastest credible "ordinary adaptation" baseline |
| ai-models-ensembles | MeteoSwiss/ai-models-ensembles (SPW, arXiv 2609.08412) | perturbs frozen weights of Aurora/GraphCast/etc. at inference; tensor-group sweep + CRPS plumbing |
| weather-regional | akhtarvision/weather-regional (arXiv 2409.07585) | static regional LoRA on a global neural forecaster |
| ORCA | Fifthky/ORCA (arXiv 2606.14222) | black-box online adaptation on the "context of errors": ready-made e0-head hypothesis harness |
| PETSA | BorealisAI/PETSA (arXiv 2506.23424) | low-rank adapters + dynamic gating at test time on a frozen forecaster |
| INC | tum-pbs/INC (arXiv 2511.12764, NeurIPS 2025) | where to inject a learned correction in an autoregressive rollout (amplification analysis) |
| ncflow | ddrous/ncflow (arXiv 2405.02154) | Neural Context Flows: Taylor expansion in context space (linearized response cousin) |
| LEADS | yuan-yin/LEADS | environment-conditioned dynamics adaptation (GEPS ancestor, with CoDA) |
| PyEPO | khalil-research/PyEPO | predict-then-optimize benchmark library; regret metrics; DFL baselines |
| cvxpylayers | cvxgrp/cvxpylayers (arXiv 1910.12430) | differentiable convex layers for the online box-QP planner |
| qpth | locuslab/qpth (OptNet, arXiv 1703.00443) | batched differentiable QP solver for thousands of tiny identical QPs |
| amortized-optimization-tutorial | facebookresearch/amortized-optimization-tutorial (arXiv 2202.00665) | fully- vs semi-amortized objectives: the two student variants |
| ties-merging | prateeky2806/ties-merging (arXiv 2306.01708) | parameter-space edit-interference resolution: "no response model" ablation |
| task_vectors | mlfoundations/task_vectors (arXiv 2212.04089) | simplest edit composition baseline (add/negate task vectors) |
| mergekit | arcee-ai/mergekit | production TIES / task-arithmetic / Fisher merging for building a diverse edit bank |

## Batch 4 (recommended by survey E, cloned with `_clone_refs_batch4.sh`)

| dir | upstream | why |
|---|---|---|
| geoarches | INRIA/geoarches (ArchesWeather / ArchesWeatherGen) | **preferred second backbone**: 1.5° 13×121×240, 340 MB checkpoint (HF gcouairon/ArchesWeather), BSD-3 code + weights, all adapter targets are `nn.Linear` |
| Flow-JEPA | HuoYanchen/Flow-JEPA (arXiv 2608.29029, MIT) | official Flow-JEPA code (flow-matching latent trajectories on LeWM); only one of the named JEPA neighbours with public code |
| ace | ai2cm/ace (ACE2, Apache-2.0 code + weights) | runner-up second backbone at ~1° (SFNO MLPs are 1×1 Conv2d, adapter needs a Conv variant) |

Check each upstream license before copying code; CoMoL, geps, ORCA, WeatherPEFT, Weight2Token and task_vectors ship no LICENSE file (all rights reserved by default); lejepa and amortized-optimization-tutorial are CC BY-NC 4.0, PETSA is CC BY-NC-SA 4.0 (non-commercial); mergekit is LGPL. Licenses for all batches are recorded in `_manifest.json`.
