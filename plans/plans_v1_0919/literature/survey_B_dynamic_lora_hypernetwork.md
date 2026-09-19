# Survey B — context-conditioned / dynamic / hypernetwork LoRA and budget-aware adapter selection (2024-09 to 2026-09)

Source: background literature agent, 2026-09-19. ~45 arXiv API queries, ~17 web searches, ~35 abstract pages opened; dates read from the abs page unless flagged.

## 1. Hypernetwork-generated LoRA / amortized adaptation
- **CLAW** | 2609.12278 | 2026-09-10 | Fernando Palafox. Hypernetwork generates LoRA for a frozen world model from test-time transitions; jointly pretrained. Overlap HIGH: frozen backbone + context->LoRA for multi-step rollouts, but direct weight generation with a task loss; no response prediction, gain objective, budget or edit bank. Its own ablation: advantage comes from expressive adapters rather than context conditioning. No code.
- **Text-to-LoRA (T2L)** | 2506.06105 | 2025-06-06 | Rujikorn Charakorn. Hypernetwork emits LoRA from a task description; distills a bank of 9 LoRAs (coefficient/weight distillation = EarthDelta's D0 baseline). Code https://github.com/SakanaAI/text-to-lora .
- **Doc-to-LoRA** 2602.15902 (2026-02-13); **SHINE** 2602.06358 (2026-02-06, code https://github.com/MuLabPKU/SHINE ); **LoRA-Gen** 2506.11638; **MoEGen** 2608.03275 (2026-08-04; expert codes -> hypernetwork -> instance LoRA; closest to a continuous program a learned end-to-end); **Ouroboros** 2604.02051 (2026-04-02; controller emits per-step diagonal modulation over frozen SVD LoRA bases — structurally close to Variant B's parameterization, trained by task loss); **DA-MergeLoRA** 2607.17467 (2026-07-20; hypernetwork emits per-column merging factors over a frozen LoRA bank from an unlabeled target batch); **HyperFix** 2608.11499 (2026-08-11; subset-conditioned nonlinear corrections for task-vector merging with perturbation bounds beyond linear merging — closest treatment of non-additive edit interactions, offline, weight space, no Gram/QP).
- Crowded-field block: HyperDreamBooth 2307.06949; Trans-LoRA 2405.17258; In-Context Meta LoRA 2501.17635; Meta-LoRA 2503.22352 / 2608.12389; ParametricSkills 2606.30015; Compliance2LoRA 2607.27594; Code2LoRA 2606.06492; HyLoVQA 2605.22035 (alignment loss tying feature discrepancy to parameter functional change); federated hypernetwork LoRA 2606.06154; scaling laws for hypernetwork knowledge injection 2607.19604; HyperLoRA for PDEs 2308.09290.

## 2. Input-dependent / dynamic LoRA, routing, budget-aware compute
- **VI-MoLE** | 2608.02528 | 2026-08-03 | Tom Saliencro. Learns counterfactual risk remaining after each expert prefix; upper-risk certificates; spends a global adapter budget on the token-layer action with largest certified marginal risk reduction per unit cost; greedy optimality and regret bounds. Overlap HIGH — closest prior on "predict gain per candidate, allocate under budget". Differences: scalar risk not vector response; greedy over prefixes not QP over a Gram; no cross-edit interaction; per-token LLM not multi-step rollout.
- **LiST: Local-Simplex Test-Time LoRA Fusion** | 2608.22370 | 2026-08-23 | Yihua Shao. Label-free test-time search of sample-specific fusion weights over a LoRA bank with an energy + safe acceptance rule. HIGH structurally (bank + per-input coefficient search + verify gate); heuristic energy, no learned response, no H, no budget.
- **MAPLE** | 2608.15299 | 2026-08-15 | Lie Li. Probe each layer's response to expert count, closed-form budgeted allocation. MEDIUM-HIGH ("measure response, solve allocation" in miniature; static, scalar).
- **DISeL** | 2605.19028 | 2026-05-18 | Ali Zindari. Input-dependent gates over rank-one LoRA components; diagnostics of which layers/ranks matter. MEDIUM-HIGH (the end-to-end foil). Code https://github.com/alizindari/DISeL .
- **CCM-LoRA** | Findings of ACL 2026 (2026.findings-acl.1329, pp. 26670–26689) | Rifat Rafiuddin & Rafae Abdullah. Input-dependent subset of rank directions with a budget-constrained objective on expected effective rank/FLOPs. HIGH on budget-aware context-conditioned rank selection (end-to-end task loss). No arXiv preprint or code found.
- **CoMoL** 2603.00573 (2026-02-28); **CARE** 2607.26052 (budget thermostat); **Hard-Routed MoR-LoRA** 2606.31413 (frozen experts, hard top-1); **Budgeted LoRA** 2605.04341; **LD-MoLE** 2509.25684; **SpawnLoRA** 2609.03150 (adapter-gradient cosine similarity as pairwise interaction measure); **Not All Layers Need Tuning (VLA)** 2609.18084 (2026-09-16; unlabeled diagnostic -> per-region adaptation cost -> budgeted variable-rank LoRA; HIGH minus response/gain formalism and temporal dimension).
- Rank allocation baselines: AdaLoRA 2303.10512; DyLoRA 2210.07558; SoRA 2311.11696; DR-LoRA 2601.04823; FIM-LoRA 2605.16800 (Fisher scalar -> budget-constrained integer allocation); PARA 2604.27796; Aletheia 2604.15351; ReMix 2603.10160 (RL routing over LoRAs).

## 3. Selecting among adapters by predicted utility
- **Counterfactual Routing Analysis in MoE LMs** 2605.07260 (2026-05-08): oracle best route != router pick; measures counterfactual utility, does not learn it.
- **MergeProbe** 2606.19549 (2026-06-17): predicts pairwise/set-level retention of PEFT updates after merging and drives merge/reweight/prune/route decisions. MEDIUM-HIGH (predicted consequence incl. interactions; offline scalar).
- **W2T (Weight2Token)** 2603.15990 (2026-03-16): canonicalize LoRA (QR->SVD), predict adapter performance from weights. Code https://github.com/xiaolonghan2000/Weight2Token .
- **LORAUTER** 2601.21795; **LoraHub** 2307.13269 (CMA-ES over composition coefficients; code https://github.com/sail-sg/lorahub ); **LoraRetriever** 2402.09997; **LoRAverse** 2510.15022 (submodular selection); **AdaMerging** 2310.02575 (label-free coefficients via test-time entropy; code https://github.com/EnnengYang/AdaMerging ); Adaptive Minds 2510.15416; Spectral Geometry of LoRA Adapters 2604.08844.

## 4. Model editing with predicted effect / linearized fine-tuning
- **Formalising the Logit Shift Induced by LoRA** | 2604.20313 | 2026-04-22 | Xiang Shi. First-order Fréchet expansion: multi-layer LoRA effect = linear sum of layerwise contributions + higher-order inter-layer coupling remainder. HIGH: the published math behind additive responses + coupling term; no measured R, no H, no selection, no budget.
- Task Arithmetic in the Tangent Space 2305.12827; Distilling Linearized Behavior 2605.18993; Rank-Efficient LoRA via Tangent-Space Optimization 2609.12123 (2026-09-10); Curvature-Guided LoRA 2603.29824.
- **Predicting Where Steering Vectors Succeed** 2604.15557 (2026-04-16): predicts before intervening which layer an edit will work on. MEDIUM.
- **Pre-Intervention Prediction of SAE Steering Side Effects** 2606.08365 (2026-06-06): first-order expansion predicts collateral spread and ranks candidate features. MEDIUM-HIGH conceptually.

## 5. Meta-learning for PDE / physics adaptation
- GEPS 2410.23889 (NeurIPS 2024; project page https://geps-project.github.io ; repo path not resolved by the agent — note: itsakk/geps exists and is cloned locally); CoDA 2202.01889 (code https://github.com/yuan-yin/CoDA ); Neural Context Flows 2405.02154; Zebra 2410.03437 (code https://github.com/LouisSerrano/zebra ); Unsupervised Adaptation of PDE Foundation Models 2608.07053 (LoRA with PDE-residual objective, no ground truth); hypernetwork spatially adaptive neural operators 2609.20309; graph hypernetworks for PINNs 2609.19915; **WeatherPEFT** 2509.22020 (2025-09-26; dynamic prompting + Fisher-guided parameter selection on weather FMs); **ARROW** 2510.09734 (2025-10-10; shared-private MoE over time scales + RL adaptive rollout scheduler — chooses rollout configuration per issue time in weather); **FTAE-Weather** 2608.09948 (2026-07; RL agent assigns variable/horizon-specific fusion weights over a pool of pretrained forecasters from the current state).

## (a) Five closest prior works and what EarthDelta still adds
1. VI-MoLE — vector du and e0, explicit quadratic gain, Gram H with off-diagonal interactions, multi-step rollout with per-step edits, predictable e0 field.
2. CLAW — edit bank, budgeted selection, response prediction, QP planning, nonlinear verification.
3. LiST — learned response model with error-reduction objective instead of hand energy; interaction-aware H; amortized student; budget; temporal structure.
4. CCM-LoRA — selection by predicted gain rather than task loss; layers x rank-groups x time steps; planning/verification; coefficient-vs-response distillation.
5. Logit-shift note + HyperFix — measure R numerically, assemble H, use it as the online QP operator; condition on forecast history at issue time.
Honourable mentions: AdaMerging, MAPLE, Not All Layers Need Tuning, MergeProbe, DISeL, Ouroboros, DA-MergeLoRA.

## (b) Already done? — No paper found that
- computes a per-edit response matrix R over adapter candidates and forms H = R^T Q R, b = R^T Q e;
- selects adapter edits by maximizing 2<e0,du> - ||du||^2 or an equivalent predicted-error-reduction quadratic;
- trains a student with the H-weighted loss (a-a*)^T H (a-a*) (coefficient distillation alone IS published: T2L, MoEGen, DA-MergeLoRA);
- predicts (b,H) from history and solves a tiny QP online.
Nearest misses: VI-MoLE, MAPLE, 2604.20313, HyperFix, LoraHub, AdaMerging. Caveat: 2<e0,du> - ||du||^2 = ||e0||^2 - ||e0-du||^2 is the exact error reduction; reviewers will see a derived acceptance test, so the defensible novelty is the pipeline (measured R -> H with cross terms -> box QP -> nonlinear verification -> amortized student), not the algebra.

## (c) Name of "Gram matrix of responses + QP"
Model-merging lineage: **RegMean** 2212.09849 (input-activation Gram matrices, closed-form least squares; RegMean++ 2508.03121); **Fisher Merging** 2111.09832; **MaTS** 2312.04339 (unifies Fisher merging and RegMean as one linear system). "Surgery" (2402.02705, SurgeryV2 2410.14389) is a different thing. TIES/DARE are heuristic interference resolution, not QP. Training-objective half: decision-focused learning / Smart Predict-then-Optimize (1710.08005); (a-a*)^T H (a-a*) is a DFL surrogate loss; no prior work found applying DFL to adapter selection. Recent Gram merging: Bayesian Model Merging 2605.12843, ACE-Merging 2603.02945 (dates unverified).

## (d) Repos
SakanaAI/text-to-lora; alizindari/DISeL; xiaolonghan2000/Weight2Token; MuLabPKU/SHINE; sail-sg/lorahub; EnnengYang/AdaMerging; yuan-yin/CoDA; LouisSerrano/zebra; tanganke/fusion_bench (existence not verified by the agent). No code on abs pages for CLAW, CoMoL (note: DCDmllm/CoMoL exists and is cloned locally), LiST; CCM-LoRA has no arXiv ID or code.
