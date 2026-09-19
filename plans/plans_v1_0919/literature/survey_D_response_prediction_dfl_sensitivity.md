# Survey D — predicting the effect of interventions; decision-focused & response-based learning; forecast sensitivity (NWP/DA); Gram/curvature metrics

Source: background literature agent, 2026-09-19. ~70 arXiv searches, 12 web searches, ~10 Crossref lookups; every arXiv ID/date confirmed on the abs page, journal DOIs via Crossref.

## 1. Decision-focused learning / predict-then-optimize
- **PEAR** | 2605.01361 | 2026-05-02 | Junhyeong Lee. Regret gradient = prediction error projected onto the tangent space of active constraints, scaled by curvature. MEDIUM-HIGH (decision-relevant projection; shapes a training gradient, not inference-time selection). Code https://github.com/FinJun/PEAR (needs Gurobi). Venue unverified.
- SPO/SPO+ 1710.08005 (Elmachtoub & Grigas); Task-based end-to-end learning 1703.04529 (Donti, Amos, Kolter); OptNet 1703.00443 / qpth; cvxpylayers 1910.12430; DFL survey 2307.13565 (Mandi et al., JAIR) — "predict the QP parameters (b,H) directly" is not one of its four gradient-based families.
- **Decision Geometry of Covariance Estimation (exact regret identity)** | 2606.27462 | 2026-06-25 | Xavier Fonseca. MEDIUM-HIGH (exact quadratic regret identity on a low-dim decision subspace).
- **Decision-focused Sparse Tangent Portfolio Optimization** | 2607.00581 | 2026-07-01 | Haeun Jeon. Differentiable exact cardinality-k selection. MEDIUM-HIGH (template for a differentiable budget).
- Differentiable knapsack / top-k via DP 2601.21775; **Forecast Skill Is Not Decision Skill** 2512.14779 (Raeth & Ludwig; motivation, lists "optimize MLWP for decisions" as open); solver-free PtO 2606.19587; cost-sensitive DFL 2605.18005; dual perspective 2511.04909.

## 2. Forecast sensitivity & impact estimation in NWP / DA
- **PEFSO** | 2609.12296 | 2026-09-10 | Fumitoshi Kawasaki, Shunji Kotsuki. Observation impact without reintegration (stronger tangent-linear approx than EFSO); ADD-SEL-PRE updates the forecast ensemble after denying observations. HIGH (predict consequence of an intervention cheaply, then act); observation-space, tangent-linear, needs verifying analysis, ~2-day validity.
- **FSO** Langland & Baker 2004 (Tellus A 56(3), DOI 10.3402/tellusa.v56i3.14413); **EFSO** Kalnay, Ota, Miyoshi, Liu 2012 (Tellus A 64:18462, DOI 10.3402/tellusa.v64i0.18462); Liu & Kalnay 2008 (QJRMS, DOI 10.1002/qj.280); Kotsuki, Kurosawa, Miyoshi 2019 (QJRMS, DOI 10.1002/qj.3534). HIGH on the gain identity (see below).
- **Proactive QC** Hotta, Chen, Kalnay, Ota, Miyoshi 2017 (MWR 145(8), DOI 10.1175/MWR-D-16-0290.1). Select interventions by estimated impact, then rerun; needs a 6-h-later analysis. HIGH.
- **ML enables real-time Proactive QC** Takumi Honda & Atsushi Yamazaki 2024 (GRL, DOI 10.1029/2023GL107938). ML trained on analyses supplies the reference state WITHOUT future observations, enabling real-time impact-based selection. HIGH — closest precedent for "sensitivity without future truth"; learns only the reference (implicitly e0), observations not weight edits, toy system, no bank/budget/interactions.
- Adjoint sensitivity to DA/model-error parameters: Daescu & Todling 2010 (QJRMS 136, DOI 10.1002/qj.693); Shaw & Daescu 2017 (JCP, DOI 10.1016/j.jcp.2017.04.050); **EFSR** Hotta, Kalnay, Ota, Miyoshi 2017 (MWR 145(12), DOI 10.1175/MWR-D-17-0122.1). HIGH on "forecast sensitivity to parameters" — diagnostic, not amortized, no competing-edit Gram.
- **Using DA tools to dissect GraphDOP** | 2510.27388 | 2025-10-31 | Laloyaux et al. (ECMWF). FSOI on an ML model via autodiff. MEDIUM.
- **Sparse Sensor Placement for Reducing Forecast Errors in EnKF** | 2606.27267 | 2026-06-25 | Takumi Saito, Shunji Kotsuki. Budgeted greedy selection maximizing predicted forecast-error reduction with information matrices (A/D/E-optimality). MEDIUM-HIGH.
- **Atmospheric Predictability Beyond 30 Days with ML** | 2504.20238 | 2025-04-28 | P. Trent Vonich, Gregory J. Hakim. Oracle IC optimization through GraphCast against known future; real-time identification left open. HIGH on oracle, zero on amortization.
- Arnoldi singular vectors 2506.22450; CNOP in FuXi 2603.26165; SPW 2609.08412 (MEDIUM on weight-space intervention premise).
- **Green's-function calibration**: Menemenlis, Fukumori, Lee 2005 (MWR 133:1224, DOI 10.1175/MWR2912.1); Strobach et al. 2022 (GMD 15:2309, DOI 10.5194/gmd-15-2309-2022). Perturb each parameter, run forward, assemble finite-difference kernel G, solve weighted least squares (G^T W G, G^T W (y - model)). HIGH — this IS EarthDelta's offline teacher (R, H = R^T Q R, b = R^T Q e), published 2005. Not flow-conditioned, not amortized, not per-forecast, no budget.

## 3. Learned surrogates of the effect of parameter/weight changes
- **Black Box Causal Inference** 2503.05985 (Bynum et al.) — amortized effect estimation. MEDIUM.
- **HyperSteer** 2506.03292 (Sun et al., Stanford) — hypernetwork emits steering vectors; predicts the intervention, not its consequence. MEDIUM. **HyperTransport** 2605.08254. LOW-MEDIUM.
- **FFORMPP** 1908.11500 (Talagala, Li, Kang) — predict forecast error from history features, select model/combination by minimum predicted error. MEDIUM-HIGH (controller in miniature).
- HRRR error prediction 2512.14898 / 2606.19026 (Evans). MEDIUM-HIGH.
- Transferability estimation: MetaRank 2511.21007; implicit modeling 2510.23145; topology-driven 2602.23916; Evidence > Intuition 2210.11255. MEDIUM.

## 4. Functional / response-weighted metrics
- EWC 1612.00796; Fisher merging 2111.09832; **OBC** 2208.11580 and **GPTQ** 2210.17323 (minimize dW^T (X X^T) dW — Gram-of-activations-weighted parameter distance, tightest analogue); RegMean 2212.09849 (RegMean++ 2508.03121); uncertainty-based gradient matching merging 2310.12808. HIGH on the FORM of (a-a*)^T H (a-a*).
- Terminology: "response distillation" already denotes logit distillation in class-incremental detection (Elastic Response Distillation 2204.02136; Refined Response Distillation 2305.00620) — rename. Because du(a) = R a is linear, (a-a*)^T H (a-a*) = ||du(a) - du(a*)||_Q^2 exactly: it is output-space squared error in coefficient coordinates (generalized Gauss–Newton / Fisher metric for a linear parameterization), not a curvature approximation.

## 5. Amortized optimization & learned planners
- **Tutorial on amortized optimization** 2202.00665 (Brandon Amos). EarthDelta's students are fully-amortized (predict a*) vs semi-amortized (predict (b,H), solve a tiny QP). Code https://github.com/facebookresearch/amortized-optimization-tutorial .
- Learning to warm-start fixed-point algorithms 2309.07835 (Sambharya, Hall, Amos, Stellato). MEDIUM. OSQP 1711.08013 (tooling).

## 6. Control-theoretic framing / adapter banks as actuators
- **Learning Options for Compositional Motor Control with Adapter Banks** | 2609.17042 | 2026-09-15 | Sreejan Kumar, Marcelo Mattar, Lea Duncker. Frozen recurrent core + bank of residual adapters (emergent low-rank perturbations) + high-level policy over options with the network frozen. HIGH on architecture (frozen backbone + bank of low-rank edits + selecting controller), LOW on mechanism (RL return, no response prediction, no e0, no H, no budget). Must cite.
- **LORAUTER** 2601.21795 (routing by task representations). MEDIUM-HIGH.
- **Chance-constrained selection of sequential interventions from counterfactual estimates** | 2608.13209 | 2026-08-13 | Minkyoung Kim. Budgeted selection from predicted counterfactual outcomes. MEDIUM-HIGH.
- WeatherPEFT 2509.22020; MENA LoRA 2409.07585 (baselines).

## 7. Model merging / task vectors with curvature
- Task arithmetic 2212.04089 (code mlfoundations/task_vectors); TIES 2306.01708 (code prateeky2806/ties-merging); task vectors and gradients 2508.16082; high-dimensional sparse disentanglement 2608.25354. MEDIUM-HIGH on edit interference; EarthDelta's interference is measured in response space over a rollout and predicted at issue time.

## (a) Five closest prior works
1. Green's-function calibration (Menemenlis et al. 2005; Strobach et al. 2022) — EarthDelta adds: response of a multi-step neural rollout w.r.t. adapter coefficients; per-forecast flow-dependent solve; box/budget best-subset; amortized student predicting (b,H) from legal history.
2. Honda & Yamazaki 2024 (GRL) — EarthDelta adds: learned du per candidate; weight edits not observations; bank with interactions and budget; nonlinear verification; modern MLWM.
3. PEFSO 2609.12296 + EFSO/PQC lineage — EarthDelta adds: du and e0 learned from history (not tangent-linear; not capped by the ~2-day window); discrete edit bank with budget; learned off-diagonal H.
4. Adapter Banks 2609.17042 — EarthDelta adds: explicit predicted quadratic gain rather than RL return; offline QP oracle with finite differences and verification; interaction-aware planning; physical error norm.
5. PEAR 2605.01361 (+ Fonseca 2606.27462) — EarthDelta applies the projection principle to choosing interventions at inference; the decision-relevant subspace (span of edit responses) is itself predicted.

## (b) Precedent for the identity and the decomposition
- Identity: YES, standard in FSO/EFSO since Langland & Baker 2004 and in the Kalnay et al. 2012 form Δe² = (e_a − e_b)^T C (e_a + e_b); with e_b = e0 and e_a = e0 − du this is exactly −(2<e0,du>_C − ||du||_C^2). Also the one-step gain of greedy least squares / matching pursuit, and the Green's-function normal equations. Do not claim as novel.
- Decomposition "predict e0 and du separately from legal history, then combine": NO explicit precedent found. Halves exist separately: e0 (Honda & Yamazaki 2024; Evans 2512.14898/2606.19026; FFORMPP 1908.11500); du computed (EFSO/PEFSO; Green's functions) but never learned from history. In all FSO/EFSO work e0 comes from a verifying analysis (post-hoc diagnostic).

## (c) Name for (a−a*)^T H (a−a*)
"Response distillation" is taken (logit distillation). The object is output-space squared error in coefficient coordinates = generalized Gauss–Newton / Fisher metric; precedents OBC, GPTQ, RegMean, Fisher merging, EWC, OBD/OBS lineage. Claim only that the responses over a forecast rollout, and their prediction from history, are new.

## (d) Repos
FinJun/PEAR; khalil-research/PyEPO; cvxgrp/cvxpylayers; locuslab/qpth (+ optnet); facebookresearch/amortized-optimization-tutorial; prateeky2806/ties-merging; mlfoundations/task_vectors; arcee-ai/mergekit; google-deepmind/graphcast. Not found: code for 2607.00581, SPW URL (note: MeteoSwiss/ai-models-ensembles per survey C), PEFSO, Honda & Yamazaki 2024, GMD 2022 Green's-function calibration.

## Bottom line
- Not novel (cite aggressively): the gain identity (FSO/EFSO); the offline oracle R -> b,H -> box-LSQ -> verify (Green's-function calibration); the (a−a*)^T H (a−a*) metric form (GGN/Fisher; OBC/GPTQ/RegMean/EWC); frozen backbone + low-rank adapter bank + selecting controller (2609.17042); differentiable QP layers; the name "response distillation".
- Unoccupied on ~70 searches: learning BOTH e0 and per-edit du from legal history at issue time and combining them through the exact quadratic to choose parameter-space edits under a budget with a learned off-diagonal interaction matrix.
- Sharpened claim: "amortizing forecast-sensitivity-to-parameter-edits into an issue-time controller", positioned against adjoint parameter sensitivity (Daescu & Todling 2010; Shaw & Daescu 2017; diagnostic) and Green's-function calibration (offline, global, non-amortized).
