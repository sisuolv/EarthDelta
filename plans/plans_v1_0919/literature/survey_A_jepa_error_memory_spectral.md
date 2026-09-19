# Survey A — JEPA / latent world models for physics & weather; forecast-error prediction; memory / fast-weight / TTT; spectral losses

Source: background literature agent, 2026-09-19. Dates are v1 arXiv submission dates read from the abs page unless flagged. The arXiv API was rate-limited (429) during this run; coverage came from the arXiv search UI plus ~45 individually opened abstract pages. Items whose abstract page was not opened are flagged at the end.

## 1. JEPA / latent world models for physics & dynamics

- **Semigroup-JEPA (SG-JEPA)** | 2609.10464 | 2026-09-09 | Andy Zeyi Liu. LeWM + physics parameter concatenated to the action; encoder+predictor trained through a discounted autoregressive latent rollout; SIGReg, no target network. Overlap HIGH (latent rollout), MEDIUM (edit-response conditioning). No error prediction, memory, spectral targets or edit selection. Code https://github.com/sg-jepa/sg-jepa ; checkpoints on HF datasets sg-jepa/sg-jepa.
- **Spectral-Target Physical Latent Structuring for JEPA-Style World Models** | 2609.04264 | v1 2026-09-02, v3 2026-09-16 | Penghao Zhu. "Physical representation laziness" in LeWM fixed with a training-time Fourier auxiliary head. Overlap HIGH (physical readout anchors + spectral targets). No code found.
- **Flow-JEPA** | 2608.29029 | 2026-08-29 | Yanchen Huo. Conditional flow matching over whole future latent sequences. Overlap LOW-MEDIUM. No code found.
- **EPM-JEPA: Operator-Side Experience Modulation** | 2606.12979 | 2026-06-11 | Vedant Pandya. Experience memory -> low-rank (LoRA) weight deltas applied to the JEPA predictor vs operand-side injection; pre-registered; headline result is NULL (4.74%, n.s.); mechanism analysis of buffer cycling, EMA target drift, LoRA settling transient. Overlap HIGH (memory + LoRA + JEPA). It generates a delta; it does not predict the response of a forecaster to a candidate edit, nor the reference error, nor select under a budget. Promised successor "PEM-JEPA" does not exist on arXiv. No code found.
- **LeWorldModel (LeWM)** | 2603.19312 | v1 2026-03-13 | Maes, Le Lidec, Scieur, LeCun, Balestriero. Two-loss stable end-to-end JEPA, ~15M params. Code https://github.com/lucas-maes/le-wm . LeJEPA/SIGReg: 2511.08544, code https://github.com/rbalestr-lab/lejepa .
- **JEPA-Anything** | 2609.20800 | 2026-09-17 | Taoyong Cui. Orthogonal Predictive Factorization; seven domains incl. weather and physical fields; explicit intervention prediction. Overlap HIGH (JEPA + weather + intervention effects) but it predicts the latent effect of an intervention on the environment, not the response of a frozen forecaster's output to a weight edit. Code https://github.com/Gen-Verse/JEPA-Anything . Predecessor Orthogonal JEPA 2608.20065.
- **M-JEPA (Tracing the Unlabeled Storm)** | 2608.22358 | 2026-08-23 | K M Anirudh. Lagrangian monsoon JEPA; frozen representation transferred to precipitation; beats 51-member ECMWF ensemble on CRPS. Overlap MEDIUM-HIGH: the only genuinely atmospheric JEPA forecasting paper found.
- **Phys-JEPA** | 2606.16076 | 2026-06-15 | Weizhi Nie. Physical + residual latent decomposition on multivariate series. Overlap MEDIUM-HIGH (readout anchors).
- **JEPA-x** | 2608.24044 | 2026-08-25 | Kehan Wen. Privileged physical state as second view; ablation: direct physical-state regression improves decodability without improving forecastability. Overlap HIGH as a caveat for physical readout heads.
- **PhyLatent** | 2608.05720 | 2026-08-06 | Xi Zeng. Names "counterfactual dynamics collapse" (predictor ignores which action was applied). Overlap HIGH — the exact failure an edit-response head risks.
- **PSG-JEPA** | 2608.06799 | 2026-08-07 | Haodong Yan. Grounds latent pairs (differences) in physical deltas. Overlap HIGH (grounding a du-latent).
- **IMPLY** | 2609.12441 | 2026-09-11 | Aman Mehta. Physically anchored scoring selects among candidate rollout sets without truth; within 0.003 of oracle on V-JEPA 2-AC. Overlap HIGH (anchored selection). Companion CALIPER 2609.08250.
- **Goal-Agnostic Joint-Embedding Predictive Control of PDEs** | 2607.21644 | 2026-07-21 | Jonathan Gallagher. Frozen action-conditioned JEPA on Navier–Stokes; MPPI with a learned linear kinetic-energy probe as the objective. Overlap HIGH (physical readout used as the selection objective).
- **Decision-Metric Alignment in Latent World Models (DA-LeWM)** | 2608.18746 | 2026-08-19 | Jiawei Wang. Latent distance does not rank candidate action sequences by real progress; Plan-Real Spearman diagnostic. Overlap MEDIUM-HIGH (edit ranking validity).
- **The Intervention Gap in Latent World Models** | 2608.29998 | 2026-08 | Donna Vakalis. On LeWM checkpoints, imagined 5-step intervention effects are worse than predicting no effect. Overlap HIGH as a threat: a predict-no-effect baseline for du is mandatory.
- **Delta-JEPA** | 2606.31232 | 2026-06-30 | Zhenghao Zhang. Latent Difference Action Decoder makes latent displacement identify the action. Overlap MEDIUM-HIGH.
- **ACPC diagnostics** | 2608.12939 | 2026-08-13 | Guo An. Bisimulation-grounded robustness diagnostic. MEDIUM.
- **Branch-JEPA** (formerly MoP-JEPA) | 2607.05238 | 2026-07-06 | Zhi Song. Finite-support latent successors. LOW-MEDIUM.
- **UniJEPA** | 2608.07409 | 2026-08-07 | ICML 2026. No EMA/stop-grad. LOW-MEDIUM (argues against EMA targets).
- **ScaleAware-JEPA** | 2606.29723 | 2026-06-29 | Guang-Xing Li. Scale-band-aware masking for multiscale physical fields. MEDIUM-HIGH (spectral bands).
- Context: V-JEPA 2 2506.09985; DINO-WM 2411.04983; PLDM 2502.14819; Koopman Dreamer 2607.19719 (EMA teacher + multi-step rollout); "The Observer Effect in World Models" 2602.12218 (invasive fine-tuning corrupts latent physics: argument for a frozen backbone + low-capacity readouts).

## 2. Predicting the ERROR of a forecast model (no future truth)

- **Predicting Forecast Error for the HRRR Using LSTM** | 2512.14898 | 2025-12-16 | David Aaron Evans. Station-point error prediction; "predicted errors can be used to adjust deterministic HRRR forecasts at the point of use". Overlap HIGH (e0 prediction), scalar additive correction only.
- **Hybrid LSTM–ViT for HRRR forecast errors** | 2606.19026 | 2026-06-17 | David Aaron Evans. ~2x precipitation-error skill; gains at short leads / active PBL. HIGH.
- **GeoQ: Geometry-Aware Conditional Quantile Error Estimation for Scientific Surrogates** | 2608.21652 | 2026-08-21 | Khoa Nguyen (LANL). Input-dependent per-query error estimation from representation-space displacement + local support density; evaluated on medium-range weather. Overlap HIGH: closest competitor to the e0 head; scalar quantile, never used to act. Mandatory baseline.
- **Forecast error diagnostics in neural weather models** | 2506.11987 | 2025-06 | Uros Perkan. Where error sensitivity and skill gain overlap. MEDIUM (prior for where edits should act).
- **Error growth / predictability of TCs in MLWP** | 2603.26165 | 2026-03-27 | Jingchen Pu. MEDIUM.
- **Certified World Models** | 2606.13092 | 2026-06 | Hongbo Wang. A-priori predictability certificate from the model's Jacobian; couples to a budgeted re-observation decision. MEDIUM-HIGH.
- **Stochastically Perturbed Weights (SPW)** | 2609.08412 | 2026-09-08 | Simon Adamov (ETH/MeteoSwiss). Inference-time raw weight perturbations of frozen Aurora/GraphCast/SFNO/AIFS for zero-cost ensembles; finding: no injection site works across models, the productive tensor group is architecture-specific. Overlap MEDIUM-HIGH: the only weather paper treating weight-space perturbations of a frozen MLWP backbone as first-class objects; strong motivating citation for learning edit responses.
- Also: 2504.20238 (IC optimization with future truth; real-time determination flagged open); 2506.22450 (Arnoldi singular vectors for MLWP); 2601.17636 (HealDA).

## 3. Memory / fast-weight / TTT for spatiotemporal forecasting

- **StreamTTT** | 2608.13416 | v1 2026-08-13, v4 2026-09-10 | Joya Chen, Zeyun Zhong. Fast weights outside the attention context + short sliding cache. MEDIUM-HIGH (architectural pattern). Code https://github.com/zeyun-zhong/StreamTTT (default branch `master`).
- **McCast: Memory-Guided Latent Drift Correction for Precipitation Nowcasting** | 2605.13197 | 2026-05-13 | Penghui Wen. Drift-Corrective Memory Bank emits a latent correction term. Overlap HIGH: closest weather-domain prior art to a verified-error memory; memory holds rollout states, no availability discipline, no discrete edits.
- **TEFL: Prediction-Residual-Guided Rolling Forecasting** | 2602.22520 | 2026-02-26 | Xiannan Huang. Past rolling-forecast residuals as input with explicit observability reasoning, integrated through a lightweight low-rank adapter; two-stage training. Overlap HIGH: verified-residual input + availability reasoning + low-rank adapter already exist together (generic time series; conditions on residuals rather than predicting future error; one adapter, no bank).
- **HERA: Historical Evidence Routing Adapter** | 2608.05523 | 2026-08-06 | Ruyi Yuan. Routes historical evidence into a frozen latent predictor via register-routed patch memory. HIGH (frozen predictor + memory + adapter scaffolding).
- **CRAFTER: When Do Corrective Features Help?** | 2608.05207 | 2026-08-05 | Fangxin Wang. Frozen forecaster + residual mining + a single validation-grounded accept/reject gate; "corrective features model the model-failure process". HIGH (edit selection under a gate).
- FlashBack Memory 2606.16342; PARA-PV 2607.08079 (retrieval + frozen FM + residual adapter + gating); Reviving Error Correction 2605.21088; MemCast 2602.03164; ForecastCompass 2605.30858.
- Fast-weight lineage: Gated DeltaNet 2412.06464; DeltaNet parallelization 2406.06484; Titans 2501.00663; TTT for time series 2409.14012; MesaNet 2506.05233. No paper found applying gated fast weights to gridded weather forecasting.

## 4. Spectral / spherical-harmonic losses and diagnostics

- **AMSE (modified spherical harmonic loss)** | 2501.19374 | 2025-01-31 | Christopher Subich | ICML 2025. Separates decorrelation from spectral-amplitude error; GraphCast effective resolution 1250 km -> 160 km. Code https://github.com/csubich/graphcast branch `amse` (train.py --spectral-amse).
- **FastNet** | 2509.17601 | 2025-09-22 | Tom Dunstan et al. MSH loss + gradient + wind decoupling; MSH and gradient losses alone may slightly degrade RMSE. HIGH (tension a gain-targeted formulation must beat).
- **MOSAIC / (Sparse) Attention to the Details** | 2604.16429 | 2026-04 | ICML 2026. Damping and aliasing fixes on HEALPix. MEDIUM-HIGH (abs page not opened).
- **Binned Spectral Power (BSP) loss** | 2502.00472 | 2025-01-31 | Chakraborty, Mohan, Maulik. HIGH. Follow-up 2607.19387 (graph-Laplacian bands, 2026-07-01).
- Latent Structured Spectral Propagators 2605.10154; spectral nudging hybrid ensembles 2603.05570; PhysMetrics.Weather 2606.10642; WP-MIP 2604.16643; "The Recipe Matters More Than the Kitchen" 2604.01215 (FFT isotropic spectrum approximates true SH spectrum only to O(l^-1)); butterfly effect / KE cascade 2609.18489.

## 5. Loud flags — JEPA/latent prediction + adapter selection; "predict error then choose a correction"

No paper found that predicts a forecast model's future error in latent space and uses it to select a correction. Four papers own large pieces:
- **VI-MoLE** | 2608.02528 | 2026-08-03 | Tom Saliencro. Learns counterfactual risk remaining after each LoRA expert prefix, certifies it, spends a global adapter budget by certified marginal risk reduction per unit cost; greedy optimality and allocation-regret theorems. The abstract selection mathematics EarthDelta claims is already proven in LLM-land; EarthDelta's delta is vector-valued field response, unobserved future e0, spectral/physical anchoring, frozen weather backbone, verified-error memory.
- **EPM-JEPA** (memory -> LoRA deltas -> JEPA, null result).
- **IMPLY** (anchored latent scoring selects among candidates without truth).
- **JEPA-Anything** (JEPA + weather + intervention-effect prediction, 2026-09-17).
Also: ST-LoRA 2404.07919; GEPS 2410.23889; WeatherPEFT 2509.22020 (PEFT on Aurora still below full fine-tuning); AdaWeather 2606.02663 (online mixture with logarithmic regret vs best static mixture — a baseline reviewers will demand); MoWE 2509.09052; VA-MoE 2412.02503; SPECTRA 2608.01751 (band-routed embedding + stage-wise LoRA, remote sensing).

## (a) Five closest prior works and what EarthDelta still adds
1. VI-MoLE — vector/field formulation, frozen physical simulator, anticipatory e0, SH band structure, verified memory; reframe the selection rule as an instantiation.
2. EPM-JEPA — select among a bank rather than generate one delta; predict e0 and du separately; real skill metric; cite its null result as motivation.
3. IMPLY — candidates are parameter edits, anchors are global-field statistics, explicit predicted error field.
4. GeoQ — vector-valued spatially resolved e0; closes the loop estimate -> select -> improve; mandatory baseline.
5. SG-JEPA — conditioning on an edit descriptor, error/response target, real forecast system; note SG-JEPA/LeWM/LeJEPA argue EMA target encoders are unnecessary.

## (c) Will "JEPA + memory + LoRA selection" be seen as module combination?
Yes, as framed in v5. Every component has a 2025–2026 antecedent in its intended role, and several carry negative results (EPM-JEPA null; FastNet spectral loss degrades RMSE; JEPA-x decodability vs forecastability; Intervention Gap worse than no-effect; SPW architecture-specific sites). The unoccupied claim: predicting at issue time, from legal history only, the future error field of a frozen forecaster AND the counterfactual response of its output to each candidate parameter edit, and using their inner product to select. Advice: lead with the bilinear selection identity; add baselines predict-no-effect (du), GeoQ (e0), AdaWeather (controller), Plan-Real Spearman (ranking validity), SPW (edit bank); demote EMA/readouts/memory/spectral to ablations; show readouts improve forecastability of du, not just decodability.

## (d) Repos (existence verified by git ls-remote)
sg-jepa/sg-jepa; lucas-maes/le-wm; rbalestr-lab/lejepa; zeyun-zhong/StreamTTT (branch master); csubich/graphcast (amse); Gen-Verse/JEPA-Anything; NVIDIA/torch-harmonics; fla-org/flash-linear-attention; test-time-training/ttt-lm-pytorch; facebookresearch/vjepa2; gaoyuezhou/dino_wm; tung-nd/stormer; microsoft/aurora.
No public code: 2609.04264, 2608.29029, 2606.12979. "PEM-JEPA" does not exist.

## Uncertainty
arXiv API 429 throughout; 2607.05238 retitled (MoP-JEPA -> Branch-JEPA); 2608.29998 v1 day not confirmed; 2604.16429 abs page not opened; abs pages not opened for 2606.16342, 2602.03164, 2605.30858, 2605.21088, 2604.16643, 2602.15040, 2608.01751, 2509.09052, 2412.02503, 2404.07919, 2410.23889, 2603.26165, 2506.22450, 2601.17636 (IDs reliable, re-check dates before citing).
