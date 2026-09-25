"""FP-05 cheap-policy out-of-fold evaluation (OOF-v1, EVAL-v1, DELTA-MIN-v1).

Pure numpy. Three stages, each a pure function of its inputs:

* `build_folds` -- 7-day UTC blocks from 2019-07-06T12Z (the POLICY-DEV-SELECT-v1
  strata), 4 outer folds of CONTIGUOUS calendar blocks (fixed before any label
  exists: blocks 0..24 split 6/6/6/7), a support-overlap purge (support
  t-12h..t+72h widened by 24h), 3 contiguous inner folds with the same purge.
* `fit_predict` -- per outer fold, sees features for every issue but labels
  ONLY through `FoldLabels(table, train_ids)`, which raises on any other id and
  logs every access. Scaler (std floor 1e-12), PCA d=8, ridge, k-means are fit on
  fold-train (inner-fold-train for HPO) only. Methods M0-M3; the oracles M4/M5
  are computed only by `score`.
* `score` -- realized losses of the frozen actions, the fixed comparison list,
  a paired 7-day-block bootstrap (B=10,000, seed 20260924), ABOVE/BELOW/STRADDLE
  against delta_min. No decision is taken here (that is FP-06).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HOUR = 3600
BLOCK_ORIGIN_UTC = "2019-07-06T12Z"
BLOCK_ORIGIN = 1562414400  # 2019-07-06T12:00:00Z
BLOCK_DAYS = 7
N_BLOCKS = 25
OUTER_FOLDS = 4
INNER_FOLDS = 3
PURGE_HOURS = 24
PCA_DIM = 8
STD_FLOOR = 1e-12
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4)
K_GRID = (2, 3, 4)
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
MIN_CLUSTER = 5
SEED = 20260924
PRIMARY_LEAD = 24
LEADS = (6, 24, 72)
CANDIDATES = ("reference", "expert_0", "expert_1", "expert_2", "expert_3")
BOOTSTRAP_DRAWS = 10000
CONFIDENCE = 0.95
DELTA_MIN = 0.0034
COMPARISONS = (  # (name, method, baseline, lead)
    ("oracle_vs_Fs", "M4", "M0", 24), ("oracle_vs_M1", "M4", "M1", 24),
    ("M5_vs_Fs", "M5", "M0", 24), ("M1_vs_Fs", "M1", "M0", 24), ("M2_vs_Fs", "M2", "M0", 24),
    ("M2_vs_M1", "M2", "M1", 24), ("M3_vs_Fs", "M3", "M0", 24), ("M3_vs_M1", "M3", "M1", 24),
    ("M1_vs_Fs_72h", "M1", "M0", 72), ("M2_vs_Fs_72h", "M2", "M0", 72),
    ("M3_vs_Fs_72h", "M3", "M0", 72), ("M1_vs_Fs_6h", "M1", "M0", 6),
    ("M2_vs_Fs_6h", "M2", "M0", 6), ("M3_vs_Fs_6h", "M3", "M0", 6),
    ("F0_vs_Fs_background", "F0", "M0", 24),
)
METHOD_NAMES = {"M0": "fitted_fs_no_edit", "M1": "fold_train_best_static", "M2": "regime_router",
                "M3": "ridge_direct_gain", "M4": "offline_complete_hindsight_oracle",
                "M5": "oracle_static", "F0": "F0_background_report_only"}


class PolicyViolation(ValueError):
    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code, self.message, self.detail = code, message, dict(detail or {})


class LabelAccessViolation(PolicyViolation):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def array_digest(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


# =============================================================================
# blocks, folds, purge
# =============================================================================

def block_of(issue_time: int) -> int:
    return (int(issue_time) - BLOCK_ORIGIN) // (BLOCK_DAYS * 24 * HOUR)


def support(issue_time: int) -> Tuple[int, int]:
    return int(issue_time) - 12 * HOUR, int(issue_time) + 72 * HOUR


def contiguous_groups(items: Sequence[int], n: int) -> List[List[int]]:
    items = list(items)
    return [items[(g * len(items)) // n:((g + 1) * len(items)) // n] for g in range(n)]


def purge(train: Sequence[str], evals: Sequence[str], times: Mapping[str, int],
          hours: int = PURGE_HOURS) -> Tuple[List[str], List[str]]:
    """Drop from ``train`` every issue whose support widened by ``hours`` touches
    any eval issue's support (closed intervals)."""
    w = hours * HOUR
    ev = [support(times[e]) for e in evals]
    keep, dropped = [], []
    for t in train:
        a, b = support(times[t])
        if any(a - w <= e1 and e0 <= b + w for e0, e1 in ev):
            dropped.append(t)
        else:
            keep.append(t)
    return keep, dropped


def build_folds(issue_times: Mapping[str, int]) -> Dict[str, Any]:
    """OOF-v1 folds from issue times ONLY (no feature, no label)."""
    ids = sorted(issue_times, key=lambda i: (issue_times[i], i))
    blocks = {i: block_of(issue_times[i]) for i in ids}
    if any(not (0 <= b < N_BLOCKS) for b in blocks.values()):
        raise PolicyViolation("ISSUE_OUTSIDE_BLOCKS", "an issue lies outside the 25 policy_dev blocks")
    outer_blocks = contiguous_groups(range(N_BLOCKS), OUTER_FOLDS)
    folds = []
    for f, bl in enumerate(outer_blocks):
        ev = [i for i in ids if blocks[i] in bl]
        tr0 = [i for i in ids if blocks[i] not in bl]
        tr, purged = purge(tr0, ev, issue_times)
        present = sorted({blocks[i] for i in tr})
        inner = []
        for g, ibl in enumerate(contiguous_groups(present, INNER_FOLDS)):
            iev = [i for i in tr if blocks[i] in ibl]
            itr, ipurged = purge([i for i in tr if blocks[i] not in ibl], iev, issue_times)
            inner.append({"inner_fold": g, "blocks": list(ibl), "eval_ids": iev, "train_ids": itr,
                          "purged_ids": ipurged})
        folds.append({"fold": f, "blocks": list(bl), "eval_ids": ev, "train_ids": tr,
                      "purged_ids": purged, "inner": inner})
    covered = sorted(i for fo in folds for i in fo["eval_ids"])
    if covered != sorted(ids):
        raise PolicyViolation("FOLDS_NOT_A_PARTITION", "every issue must be OOF exactly once")
    return {"rule": "OOF-v1", "block_origin_utc": BLOCK_ORIGIN_UTC, "block_days": BLOCK_DAYS,
            "n_blocks": N_BLOCKS, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
            "purge_hours": PURGE_HOURS, "issue_blocks": blocks,
            "empty_blocks": sorted(set(range(N_BLOCKS)) - set(blocks.values())),
            "supports_utc_epoch": {i: list(support(issue_times[i])) for i in ids}, "folds": folds}


def check_no_overlap(folds: Mapping[str, Any], issue_times: Mapping[str, int]) -> bool:
    for fo in folds["folds"]:
        if purge(fo["train_ids"], fo["eval_ids"], issue_times)[1]:
            return False
        for inn in fo["inner"]:
            if purge(inn["train_ids"], inn["eval_ids"], issue_times)[1]:
                return False
    return True


# =============================================================================
# labels
# =============================================================================

def realized_gain_table(rows: Sequence[Mapping[str, Any]], lead: int = PRIMARY_LEAD
                        ) -> Dict[str, Dict[str, float]]:
    """{issue: {candidate: realized gain vs Fs at `lead`}}; a FAIL_NONFINITE
    candidate falls back to Fs (gain 0.0), as EVAL-v1 scores it."""
    table: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if int(r["lead_hours"]) != lead:
            continue
        g = r.get("gain_vs_fs") if r.get("status") == "PASS" else None
        table.setdefault(r["issue_id"], {})[r["candidate_id"]] = 0.0 if g is None else float(g)
    missing = [i for i, row in table.items() if set(row) != set(CANDIDATES)]
    if missing:
        raise PolicyViolation("LABEL_TABLE_INCOMPLETE", f"{len(missing)} issue(s) lack a candidate")
    return table


class FoldLabels:
    """The ONLY label accessor fit code gets. Raises on any id outside train_ids."""

    def __init__(self, table: Mapping[str, Mapping[str, float]], train_ids: Sequence[str], fold: str):
        self._table = table
        self._train: FrozenSet[str] = frozenset(train_ids)
        self.fold = fold
        self.log: List[str] = []

    def gains(self, issue_id: str) -> np.ndarray:
        if issue_id not in self._train:
            raise LabelAccessViolation("LABEL_ACCESS_OUTSIDE_TRAIN",
                                       f"fold {self.fold}: label of {issue_id} requested")
        self.log.append(issue_id)
        row = self._table[issue_id]
        return np.array([float(row[c]) for c in CANDIDATES], dtype=np.float64)

    def matrix(self, ids: Sequence[str]) -> np.ndarray:
        return np.stack([self.gains(i) for i in ids]) if ids else np.zeros((0, len(CANDIDATES)))


# =============================================================================
# numpy learners
# =============================================================================

def fit_scaler(X: np.ndarray) -> Dict[str, np.ndarray]:
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), STD_FLOOR)
    return {"mean": mu, "std": sd}


def apply_scaler(s, X):
    return (X - s["mean"]) / s["std"]


def fit_pca(Z: np.ndarray, d: int = PCA_DIM) -> Dict[str, np.ndarray]:
    mu = Z.mean(axis=0)
    _, _, vt = np.linalg.svd(Z - mu, full_matrices=False)
    comps = vt[:d].copy()
    for k in range(comps.shape[0]):
        j = int(np.argmax(np.abs(comps[k])))
        if comps[k, j] < 0:
            comps[k] = -comps[k]
    if comps.shape[0] < d:
        comps = np.vstack([comps, np.zeros((d - comps.shape[0], Z.shape[1]))])
    return {"mean": mu, "components": comps}


def apply_pca(p, Z):
    return (Z - p["mean"]) @ p["components"].T


def fit_ridge(P: np.ndarray, Y: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    mp, my = P.mean(axis=0), Y.mean(axis=0)
    Pc = P - mp
    W = np.linalg.solve(Pc.T @ Pc + lam * np.eye(P.shape[1]), Pc.T @ (Y - my))
    return {"W": W, "b": my - mp @ W}


def apply_ridge(r, P):
    return P @ r["W"] + r["b"]


def _kmeans_pp(P, k, rng):
    c = [P[int(rng.integers(len(P)))]]
    for _ in range(1, k):
        d2 = np.min(((P[:, None, :] - np.asarray(c)[None]) ** 2).sum(-1), axis=1)
        tot = d2.sum()
        idx = int(rng.integers(len(P))) if tot <= 0 else int(rng.choice(len(P), p=d2 / tot))
        c.append(P[idx])
    return np.asarray(c, dtype=np.float64)


def fit_kmeans(P: np.ndarray, k: int, seed: int = SEED) -> Dict[str, Any]:
    best = None
    for init in range(KMEANS_N_INIT):
        rng = np.random.default_rng(seed + 1000 * k + init)
        C = _kmeans_pp(P, k, rng)
        for _ in range(KMEANS_MAX_ITER):
            lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([P[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.array_equal(newC, C):
                break
            C = newC
        lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        inertia = float(((P - C[lab]) ** 2).sum())
        if best is None or inertia < best["inertia"]:
            best = {"centroids": C, "inertia": inertia, "labels": lab}
    return best


def assign_kmeans(km, P):
    return np.argmin(((P[:, None, :] - km["centroids"][None]) ** 2).sum(-1), axis=1)


def best_static(G: np.ndarray) -> int:
    """argmax of mean gain; ties -> the earliest candidate (reference = no-edit first)."""
    m = G.mean(axis=0)
    return int(np.flatnonzero(m == m.max())[0])


def ridge_action(pred: np.ndarray) -> int:
    k = int(np.argmax(pred))
    return k + 1 if pred[k] > 0.0 else 0


# =============================================================================
# fit / predict per fold
# =============================================================================

def _transform(Xtr, Xev):
    s = fit_scaler(Xtr)
    p = fit_pca(apply_scaler(s, Xtr))
    return apply_pca(p, apply_scaler(s, Xtr)), apply_pca(p, apply_scaler(s, Xev)), s, p


def _router(Ptr, Gtr, k, fallback):
    km = fit_kmeans(Ptr, k)
    actions = []
    for j in range(k):
        members = km["labels"] == j
        actions.append(best_static(Gtr[members]) if members.sum() >= MIN_CLUSTER else fallback)
    return km, actions


def fit_predict_fold(fold: Mapping[str, Any], features: Mapping[str, np.ndarray],
                     labels: FoldLabels) -> Dict[str, Any]:
    """One outer fold: HPO on inner folds, final fit on fold-train, predict fold-eval."""
    tr, ev = list(fold["train_ids"]), list(fold["eval_ids"])
    X = lambda ids: np.stack([features[i] for i in ids]).astype(np.float64)
    hpo_ridge = {lam: [] for lam in RIDGE_GRID}
    hpo_k = {k: [] for k in K_GRID}
    for inn in fold["inner"]:
        itr, iev = inn["train_ids"], inn["eval_ids"]
        if not itr or not iev:
            continue
        Ptr, Pev, _, _ = _transform(X(itr), X(iev))
        Gtr, Gev = labels.matrix(itr), labels.matrix(iev)
        for lam in RIDGE_GRID:
            r = fit_ridge(Ptr, Gtr[:, 1:], lam)
            acts = [ridge_action(p) for p in apply_ridge(r, Pev)]
            hpo_ridge[lam] += [float(Gev[n, a]) for n, a in enumerate(acts)]
        fb = best_static(Gtr)
        for k in K_GRID:
            km, cl_act = _router(Ptr, Gtr, k, fb)
            hpo_k[k] += [float(Gev[n, cl_act[c]]) for n, c in enumerate(assign_kmeans(km, Pev))]
    score_r = {lam: (float(np.mean(v)) if v else -math.inf) for lam, v in hpo_ridge.items()}
    score_k = {k: (float(np.mean(v)) if v else -math.inf) for k, v in hpo_k.items()}
    lam_best = max(RIDGE_GRID, key=lambda lam: (score_r[lam], lam))   # ties -> larger lambda
    k_best = max(K_GRID, key=lambda k: (score_k[k], -k))               # ties -> smaller k
    Ptr, Pev, scaler, pca = _transform(X(tr), X(ev))
    Gtr = labels.matrix(tr)
    m1 = best_static(Gtr)
    ridge = fit_ridge(Ptr, Gtr[:, 1:], lam_best)
    km, cl_act = _router(Ptr, Gtr, k_best, m1)
    preds = apply_ridge(ridge, Pev)
    clusters = assign_kmeans(km, Pev)
    out = {}
    for n, iid in enumerate(ev):
        out[iid] = {"M0": CANDIDATES[0], "M1": CANDIDATES[m1],
                    "M2": CANDIDATES[cl_act[int(clusters[n])]],
                    "M3": CANDIDATES[ridge_action(preds[n])],
                    "ridge_pred_gain24": [float(v) for v in preds[n]], "cluster": int(clusters[n])}
    params = {"scaler": array_digest(np.concatenate([scaler["mean"], scaler["std"]])),
              "pca": array_digest(np.concatenate([pca["mean"], pca["components"].ravel()])),
              "ridge": array_digest(np.concatenate([ridge["W"].ravel(), ridge["b"]])),
              "kmeans": array_digest(km["centroids"].ravel())}
    return {"fold": int(fold["fold"]), "eval_ids": ev, "train_ids": tr,
            "hpo": {"ridge_lambda_scores": {repr(k): v for k, v in score_r.items()},
                    "kmeans_k_scores": {str(k): v for k, v in score_k.items()},
                    "lambda": lam_best, "k": k_best,
                    "trials": {"ridge": len(RIDGE_GRID), "kmeans": len(K_GRID),
                               "inner_folds": len(fold["inner"])}},
            "M1_static_action": CANDIDATES[m1],
            "M2_cluster_actions": [CANDIDATES[a] for a in cl_act],
            "parameter_digests": params, "predictions": out}


def fit_predict(folds: Mapping[str, Any], features: Mapping[str, np.ndarray],
                table: Mapping[str, Mapping[str, float]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    per_fold, access = [], {}
    for fold in folds["folds"]:
        labels = FoldLabels(table, fold["train_ids"], str(fold["fold"]))
        per_fold.append(fit_predict_fold(fold, features, labels))
        access[str(fold["fold"])] = sorted(set(labels.log))
    return per_fold, access


# =============================================================================
# scoring
# =============================================================================

def realized_losses(rows: Sequence[Mapping[str, Any]], actions: Mapping[str, Mapping[str, str]],
                    f0: Mapping[str, Mapping[str, Optional[float]]]) -> Dict[str, Any]:
    """{method: {lead: {issue: loss}}} incl. oracles M4/M5 and the F0 background."""
    L: Dict[str, Dict[str, Dict[int, float]]] = {}
    st: Dict[Tuple[str, str, int], str] = {}
    for r in rows:
        L.setdefault(r["issue_id"], {}).setdefault(r["candidate_id"], {})[int(r["lead_hours"])] = r["cpu_loss"]
        st[(r["issue_id"], r["candidate_id"], int(r["lead_hours"]))] = r["status"]
    ids = sorted(actions)
    for i in ids:
        if any(st.get((i, "reference", h)) != "PASS" for h in LEADS):
            raise PolicyViolation("INVALID_EVIDENCE", f"reference (Fs) not finite for {i}")
    fails = {m: 0 for m in ("M0", "M1", "M2", "M3", "M4", "M5")}

    def pick(i, c, h, m):
        if st.get((i, c, h)) != "PASS":
            if h == PRIMARY_LEAD:
                fails[m] += 1
            return L[i]["reference"][h]
        return L[i][c][h]

    m5 = min(CANDIDATES, key=lambda c: (sum(L[i][c][24] if st.get((i, c, 24)) == "PASS"
                                            else L[i]["reference"][24] for i in ids),
                                        CANDIDATES.index(c)))
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    chosen: Dict[str, Dict[str, str]] = {}
    for i in ids:
        ok24 = [c for c in CANDIDATES if st.get((i, c, 24)) == "PASS"]
        m4 = min(ok24, key=lambda c: (L[i][c][24], CANDIDATES.index(c)))
        acts = {**{m: actions[i][m] for m in ("M0", "M1", "M2", "M3")}, "M4": m4, "M5": m5}
        chosen[i] = acts
        for m, c in acts.items():
            for h in LEADS:
                out.setdefault(m, {}).setdefault(h, {})[i] = pick(i, c, h, m)
        for h in LEADS:
            v = f0.get(i, {}).get(str(h))
            out.setdefault("F0", {}).setdefault(h, {})[i] = float(v) if v is not None else float("nan")
    return {"losses": out, "actions": chosen, "M5_static_choice": m5, "failures_issue_count": fails}


def effect(sums: Mapping[str, float], method: str, base: str) -> float:
    return (sums[base] - sums[method]) / sums["M0"]


def paired_block_bootstrap(losses: Mapping[str, Mapping[int, Mapping[str, float]]],
                           blocks: Mapping[str, int], *, draws: int = BOOTSTRAP_DRAWS,
                           seed: int = SEED, delta_min: float = DELTA_MIN,
                           comparisons=COMPARISONS, block_days_factor: int = 1) -> Dict[str, Any]:
    """Resample whole blocks (n = n_blocks present) jointly across methods and leads."""
    ids = sorted(blocks)
    bl = {i: blocks[i] // block_days_factor for i in ids}
    ublocks = sorted(set(bl.values()))
    methods = sorted(losses)
    # per-block sums [n_blocks, method, lead]
    S = np.zeros((len(ublocks), len(methods), len(LEADS)))
    for bi, b in enumerate(ublocks):
        members = [i for i in ids if bl[i] == b]
        for mi, m in enumerate(methods):
            for li, h in enumerate(LEADS):
                S[bi, mi, li] = sum(losses[m][h][i] for i in members)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ublocks), size=(draws, len(ublocks)))
    tot = S[idx].sum(axis=1)                       # [draws, method, lead]
    point = S.sum(axis=0)
    mi = {m: k for k, m in enumerate(methods)}
    li = {h: k for k, h in enumerate(LEADS)}
    alpha = (1 - CONFIDENCE) / 2
    res = {}
    for name, m, b, h in comparisons:
        g_draw = (tot[:, mi[b], li[h]] - tot[:, mi[m], li[h]]) / tot[:, mi["M0"], li[h]]
        g_pt = (point[mi[b], li[h]] - point[mi[m], li[h]]) / point[mi["M0"], li[h]]
        lo, hi = np.percentile(g_draw, [100 * alpha, 100 * (1 - alpha)])
        status = "ABOVE" if lo >= delta_min else ("BELOW" if hi < delta_min else "STRADDLE")
        entry = {"method": m, "baseline": b, "lead_hours": h, "point": float(g_pt),
                 "ci_low": float(lo), "ci_high": float(hi), "status_vs_delta_min": status}
        if h == 72 and b == "M0" and m in ("M1", "M2", "M3"):
            entry["harm72_guard_pass"] = bool(lo >= -delta_min)
        res[name] = entry
    # F0 correction is lead-specific. Reusing the 24h harm for 6h/72h rows
    # produces a numerically plausible but semantically incorrect diagnostic.
    h_by_lead = {}
    for h in LEADS:
        h_draw = (tot[:, mi["M0"], li[h]] - tot[:, mi["F0"], li[h]]) / tot[:, mi["M0"], li[h]]
        h_pt = (point[mi["M0"], li[h]] - point[mi["F0"], li[h]]) / point[mi["M0"], li[h]]
        h_by_lead[int(h)] = {
            "point": float(h_pt),
            "ci": [float(v) for v in np.percentile(h_draw, [100 * alpha, 100 * (1 - alpha)])],
        }
    return {"draws": int(draws), "seed": int(seed), "n_blocks": len(ublocks), "blocks": ublocks,
            "confidence": CONFIDENCE, "delta_min": delta_min, "comparisons": res,
            "H_Fs_by_lead": {str(h): v for h, v in h_by_lead.items()},
            "H_Fs_24h": {"definition": "(sum L_Fs - sum L_F0) / sum L_Fs (Fs harm vs F0; >0 = Fs worse)",
                         "point": h_by_lead[24]["point"], "ci": h_by_lead[24]["ci"]},
            "G_F0_note": "G_F0(M) = G_Fs(M) - H_Fs, reported per comparison"}


def rates(actions: Mapping[str, Mapping[str, str]], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    L24 = {(r["issue_id"], r["candidate_id"]): r["cpu_loss"] for r in rows if int(r["lead_hours"]) == 24}
    out = {}
    for m in ("M1", "M2", "M3", "M4", "M5"):
        acts = {i: a[m] for i, a in actions.items()}
        edits = [i for i, c in acts.items() if c != "reference"]
        harmful = [i for i in edits if L24.get((i, acts[i])) is not None
                   and L24[(i, acts[i])] > L24[(i, "reference")]]
        out[m] = {"no_edit_rate": 1 - len(edits) / len(acts),
                  "harmful_edit_rate": (len(harmful) / len(edits)) if edits else 0.0,
                  "selection_counts": {c: sum(1 for v in acts.values() if v == c) for c in CANDIDATES}}
    return out
