# Survey C — adaptation / correction of ML weather & climate models (2024-09 to 2026-09)

Source: background literature agent, 2026-09-19. ~30 searches; every abstract page opened to confirm ID, v1 date, first author. Note: several 2026 arXiv IDs carry a month later than the abs-page "Submitted on" date (e.g. 2608.09948 submitted 2026-07-07); dates below are the abs-page dates.

## 1. PEFT / task adaptation of weather & climate foundation models
- **WeatherPEFT** | 2509.22020 | 2025-09-26 (ICLR 2026) | Shilei Cao. TADP (encoder-derived embedding injection) + SFAS (stochastic Fisher scoring of parameters to update) on Aurora / Prithvi-WxC; benchmarks LoRA/DoRA/AdaptFormer/VPT/etc. MEDIUM-HIGH: per-task offline adaptation, never chosen per issue time. Code https://github.com/ShileiCao/WeatherPEFT .
- **Efficient Localized Adaptation of Neural Weather Forecasting (MENA)** | 2409.07585 | 2024-09-11 | Muhammad Akhtar Munir. Static regional LoRA on a global forecaster. MEDIUM. Code https://github.com/akhtarvision/weather-regional .
- **Finetuning a Weather FM with Lightweight Decoders** | 2506.19088 | 2025-06-23 | Fanny Lehmann. Frozen Aurora + shallow decoders for unseen hydrological variables. MEDIUM-LOW.
- **Efficient fine-tuning of 37-level GraphCast (Canadian analysis)** | 2408.14587 | 2024-08-26 | Christopher Subich. LOW-MEDIUM.
- **FlowDA** | 2602.06800 | 2026-02-06 | Ran Cheng. Flow-matching DA fine-tuning Aurora; FlowDA-LoRA rank 60 (37M of 1.3B). MEDIUM.
- **From Global to Local (regional downscaling heads)** | 2607.03279 | 2026-07-03 | Wiktor Kamzela. MEDIUM-LOW.
- Gravity-wave parameterization from a weather FM 2509.03816; MarsCast 2608.05054; mechanistic interpretability of a fine-tuned FM 2607.20778 (LOW).

## 2. Test-time / online / continual adaptation of forecasters
- **PETSA** | 2506.23424 | 2025-06-29 | Heitor R. Medeiros. Low-rank adapters + dynamic gating updated at test time on a frozen forecaster. HIGH on mechanism, LOW on domain (gradient updates from revealed truth; generic time series). Code https://github.com/BorealisAI/PETSA .
- **ORCA** | 2606.14222 | 2026-06-12 | Xilin Dai. Base-model error conditioned on base input AND base output ("context of errors"); black-box residual adapter, no gradients into the frozen FM; 5 TSFMs x 8 datasets. HIGH (e0-head hypothesis validated); corrects output directly. Code https://github.com/Fifthky/ORCA .
- **STEPS** | 2605.08005 | 2026-05-08 | Jiaqi Liu. TTA as Dirichlet boundary problem: prefix error propagated; Global Solver retrieves cross-window "error memory". HIGH (error memory + future error field). No code.
- **FAC / principled TTA protocol** | 2605.17250 | 2026-05-17 | Haochun Wang. "Matured ground truth only" protocol; frequency-aware calibration. MEDIUM (legality of adaptation signals).
- **FORESEE** | 2602.21757 | 2026-02-25 | Xiannan Huang. Yesterday's error -> today's correction, MoE over error dynamics, no base updates. MEDIUM-HIGH.
- **VA-MoE** | 2412.02503 | 2024-12-03 | Hao Chen. Variable-adaptive experts for incremental weather learning. MEDIUM.
- REE-TTT 2601.01605 (radar nowcasting TTT); domain-adaptive downscaling 2607.05645 (LOW).

## 3. Learned error correction, error memory, error prediction
- **RATL** | 2609.03937 | 2026-09-03 | Yuchen He. Frozen base forecaster; historical residuals as a context-keyed memory; retrieval under causal-availability constraints; set-aware router over forecast blocks/variables. HIGH: closest realization of "verified error memory read at issue time under legality + router", in output space. No code.
- **HopCast** | 2501.16587 | 2025-01-27 | Muhammad Bilal Shahid. Modern Hopfield memory of past errors read by similarity for autoregressive dynamics models. HIGH (error memory mechanism). No code.
- **HRRR forecast-error LSTM** 2512.14898 (2025-12-16) and **LSTM-ViT** 2606.19026 (2026-06-17) | David Aaron Evans. MEDIUM-HIGH (e0 head prior art; station-level).
- **AIFS-TC** | 2608.09959 | 2026-07-24 | Anna Allen. Cheap correction on frozen AIFS for TC intensity (built by an LLM agent). MEDIUM.
- **SwAIther-Precip** | 2605.16163 | 2026-05-15 | Dan Assouline. Lead-time FiLM-conditioned residual bias correction of AIFS precipitation. MEDIUM.
- **Improving precipitation in an AI model with observations** | 2609.03210 | 2026-09-02 | Julian F. Schmitt. LOW-MEDIUM.
- **Forecast error diagnostics in neural weather models** | 2506.11987 | 2025-06-13 | Uros Perkan. Measured perturb-then-respond experiments via autodiff. MEDIUM (offline analogue of R).
- **Offline+online hybrid model error correction in IFS** | 2403.03702 | 2024-03-06 | Alban Farchi (QJRMS). MEDIUM.
- **Hybrid sea-ice thermodynamics with state-dependent error parameterization** | 2601.23190 | 2026-01-30 | Giovanni De Cillis. MEDIUM.

## 4. Learned correction inside / around the solver
- **INC: Indirect Neural Corrector** | 2511.12764 | 2025-11-16 (NeurIPS 2025) | Hao Wei. Direct state corrections amplify error O(dt^-1 + L); inject into the equations instead. MEDIUM (where to inject). Code https://github.com/tum-pbs/INC .
- **ARC-STAR** | 2605.22222 | 2026-05-21 | Chengze Li. Frozen PDE FM (Poseidon) + global corrector + blockwise refiner; at deployment a label-free score routes refinement to high-risk blocks under a compute budget. HIGH (frozen host + budget-aware routing by a label-free score; spatial blocks, risk proxy not gain). No code.
- **Online RL in the Met Office Unified Model** | 2609.02566 | 2026-09-02 | Pritthijit Nath. DDPG actor applies bounded tendency corrections; trained on nudged counterfactuals, deployed without analysis access. MEDIUM-HIGH (teacher->student structure).
- **Replacing tunable parameters with state-dependent functions via RL** | 2601.04268 | 2026-01-07 | Pritthijit Nath. HIGH (state-conditioned low-dim parameter program; RL, no response prediction).

## 5. Meta-learning / in-context / context-conditioned adaptation for PDEs
- **GEPS** | 2410.23889 | 2024-10-31 (NeurIPS 2024) | Armand Kassaï Koupaï. Shared + environment-specific low-rank context modulation. HIGH (the ordinary conditional adaptation baseline). Code: local clone itsakk/geps.
- **DISCO** | 2504.19496 | 2025-04-28 | Rudy Morel. Large hypernetwork reads a short trajectory and emits a small operator network. HIGH (canonical history -> parameters -> rollout baseline). No code.
- **Test-time Generalization via Neural Operator Splitting** | 2602.00884 | 2026-01-31 | Louis Serrano. Dictionary of frozen operators + test-time search over compositions, no weight change. HIGH (bank + inference-time composition search by prefix fit, not predicted response). No code.
- **CCM: Discovering Physical Directions in Weight Space** | 2605.14546 | 2026-05-14 | Pengkai Wang. Fine-tuned endpoint experts reinterpreted as finite-difference probes of a physical direction in weight space; Calibration-Conditioned Merge infers a composition coordinate from metadata / calibrated map / a short observed rollout prefix, then deploys one merged checkpoint for the rest of the rollout. HIGH — FLAG: closest to EarthDelta's teacher; does not predict du, model e0, or solve a budgeted QP over multiple edits (one scalar coordinate along one direction). No code.
- **Zebra** 2410.03437 (code https://github.com/LouisSerrano/zebra ); **Neural Context Flows** 2405.02154 (Taylor expansion in context space; code https://github.com/ddrous/ncflow ); **CoDA** 2202.01889 (code https://github.com/yuan-yin/CoDA ; LEADS https://github.com/yuan-yin/LEADS ); **CHOP** 2606.12318; graph ICON 2603.12725; VICON 2411.16063; ICON 2304.07993.

## 6. Steering / editing / selecting among variants of a frozen weather model
- **SPW** | 2609.08412 | 2026-09-08 | Simon Adamov. Random weight perturbations of frozen Aurora/GraphCast/etc. at inference; best injection site architecture-specific. HIGH (weight-space edits on frozen weather backbone). Code https://github.com/MeteoSwiss/ai-models-ensembles .
- **Rescene** | 2608.09971 | 2026-07-30 | Minjong Cheon. 0.4M-param wrapper around a frozen 1.5 deg 6-hourly ViT weather operator (slow-clock blend + spectral perturbations each step). MEDIUM-HIGH (same backbone class, tiny per-step external controller).
- **TaCT: Target Concept Tuning** | 2603.19325 | 2026-03-17 | Shijie Ren. SAE-discovered failure concepts gate an indicator-conditioned parameter update injected into a frozen weather model; baselines LoRA/Adapter/LoREFT. HIGH (conditional application of a parameter edit; hand-derived gate, no predicted effect, no bank, no budget). No code.
- **ARROW** | 2510.09734 | 2025-10-10 (ICLR 2026) | Jindong Tian. Shared-Private MoE + RL Adaptive Rollout Scheduler choosing the next forecast interval per weather state. HIGH (closest weather-domain per-issue-time controller; selects time steps/experts by Q-learning). No code.
- **FTAE-Weather** | 2608.09948 | 2026-07-07 | Qiang Wu. Weight-Agent reads initial state + 8 forecasters' outputs and emits variable/horizon fusion weights by RL. HIGH (state-conditioned selection among a bank of whole models). No code.
- **MoWE** | 2509.09052 | 2025-09-10 | Dibyajyoti Chakraborty. Per-grid-point lead-time-conditioned gating over AI models. MEDIUM-HIGH.
- **Semantic Adapter Routing (ARIADNE)** 2606.19079 (2026-06-17); **LoGo** 2511.07129 (ACL 2026); **W2T** 2603.15990 (predicts adapter performance from weights; code https://github.com/xiaolonghan2000/Weight2Token ); **RL for Neural Model Editing** 2606.13461 (2026-06-11; LoRA-parameterized edit actions, scalar reward).

## (a) Five closest prior works
1. CCM 2605.14546 — teacher-side finite-difference probes + prefix readout; EarthDelta adds learned du and e0 models combined via the gain, multi-dim program, budget/QP, memory, 69-channel backbone.
2. ARROW 2510.09734 — weather per-state rollout controller; EarthDelta adds parameter-edit action space, model-based (du,e0) controller instead of Q-learning, budgeted multi-edit planning, frozen backbone.
3. ARC-STAR 2605.22222 — frozen host + budget-aware label-free routing; EarthDelta routes in weight space with predicted gain not risk.
4. RATL + ORCA + STEPS — frozen base + causal residual memory read at inference; EarthDelta acts in parameter space inside the rollout.
5. GEPS + DISCO — the conditional low-rank / hypernetwork baselines; EarthDelta predicts responses and plans instead of regressing coefficients.
Honourable: SPW, TaCT, Test-time Operator Splitting, FTAE-Weather.

## (b) Already done? No full match. Flags: CCM 2605.14546 (finite-difference weight-space probes + prefix readout); W2T 2603.15990 (predict adapter quality from weights, static); ARIADNE / LoGo (per-input adapter selection by similarity). Not found: learned du over multi-step rollout; e0 + du combination; QP over Gram H; two heads in weather.

## (c) Repos
tung-nd/stormer; ShileiCao/WeatherPEFT; microsoft/aurora; MeteoSwiss/ai-models-ensembles; Fifthky/ORCA; BorealisAI/PETSA; tum-pbs/INC; xiaolonghan2000/Weight2Token; ddrous/ncflow; yuan-yin/CoDA, yuan-yin/LEADS; LouisSerrano/zebra; akhtarvision/weather-regional; google-deepmind/graphcast, NASA-IMPACT/Prithvi-WxC, NVlabs/FourCastNet (alternative backbones). Not found: GEPS GitHub URL (note: itsakk/geps exists locally), ARROW, ARC-STAR, TaCT, RATL, CCM, HopCast, Rescene, FTAE-Weather, MoWE, VA-MoE, DISCO, operator splitting.
