"""FP-05 OOF isolation: labels only through FoldLabels, poisoned eval labels leave
that fold's OOF predictions byte-identical, fitted transforms are fold-train-only,
purge leaves no overlap, oracles never reach fit code, bootstrap is by block."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from earthdelta import candidate_cache as cc
from earthdelta import policy_oof as po
from scripts import r4_policy_oof as policy_cli

STEP = 6 * 3600


def synthetic(n=112, dim=40, seed=0, skip_blocks=(16, 19)):
    rng = np.random.default_rng(seed)
    slots = [s for s in range(698) if (s // 28) not in skip_blocks]
    pick = [slots[(k * len(slots)) // n] for k in range(n)]
    times = {f"iss_{k:03d}": po.BLOCK_ORIGIN + int(s) * STEP for k, s in enumerate(pick)}
    feats = {i: rng.normal(size=dim) for i in times}
    table = {i: {c: (0.0 if c == "reference" else float(rng.normal(scale=1e-3)))
                 for c in po.CANDIDATES} for i in times}
    return times, feats, table


@pytest.fixture(scope="module")
def data():
    times, feats, table = synthetic()
    folds = po.build_folds(times)
    return times, feats, table, folds


def dumps(x):
    return json.dumps(x, sort_keys=True, indent=1, allow_nan=False)


def test_folds_are_contiguous_partition_with_purge(data):
    times, _, _, folds = data
    assert [f["blocks"] for f in folds["folds"]] == [list(range(0, 6)), list(range(6, 12)),
                                                    list(range(12, 18)), list(range(18, 25))]
    assert folds["empty_blocks"] == [16, 19]
    assert sorted(i for f in folds["folds"] for i in f["eval_ids"]) == sorted(times)
    assert po.check_no_overlap(folds, times)
    w = po.PURGE_HOURS * 3600
    for f in folds["folds"]:
        assert f["purged_ids"]                      # adjacent blocks really lose issues
        for t in f["train_ids"]:
            a, b = po.support(times[t])
            for e in f["eval_ids"]:
                c, d = po.support(times[e])
                assert b + w < c or d < a - w
        assert not set(f["train_ids"]) & set(f["eval_ids"])


def test_fold_labels_refuse_eval_ids(data):
    _, _, table, folds = data
    f = folds["folds"][1]
    lab = po.FoldLabels(table, f["train_ids"], "1")
    lab.gains(f["train_ids"][0])
    for bad in (f["eval_ids"][0], f["purged_ids"][0]):
        with pytest.raises(po.LabelAccessViolation):
            lab.gains(bad)
    assert lab.log == [f["train_ids"][0]]


@pytest.mark.parametrize("value", [float("nan"), 1e9])
def test_poisoned_eval_labels_leave_fold_predictions_byte_identical(data, value):
    times, feats, table, folds = data
    for fold in folds["folds"]:
        clean = po.fit_predict_fold(fold, feats, po.FoldLabels(table, fold["train_ids"], "c"))
        poisoned = {i: dict(v) for i, v in table.items()}
        for i in fold["eval_ids"] + fold["purged_ids"]:
            poisoned[i] = {c: value for c in po.CANDIDATES}
        labels = po.FoldLabels(poisoned, fold["train_ids"], "p")
        again = po.fit_predict_fold(fold, feats, labels)
        assert dumps(clean) == dumps(again)
        assert set(labels.log) <= set(fold["train_ids"])


def test_fitted_parameters_ignore_eval_features(data):
    times, feats, table, folds = data
    fold = folds["folds"][2]
    base = po.fit_predict_fold(fold, feats, po.FoldLabels(table, fold["train_ids"], "a"))
    edited = dict(feats)
    target = fold["eval_ids"][0]
    edited[target] = feats[target] + 100.0
    again = po.fit_predict_fold(fold, edited, po.FoldLabels(table, fold["train_ids"], "b"))
    assert base["parameter_digests"] == again["parameter_digests"]
    assert base["hpo"] == again["hpo"]
    for i in fold["eval_ids"][1:]:
        assert base["predictions"][i] == again["predictions"][i]
    # and they equal a direct fit on the fold-train subset alone
    X = np.stack([feats[i] for i in fold["train_ids"]])
    s = po.fit_scaler(X)
    p = po.fit_pca(po.apply_scaler(s, X))
    assert base["parameter_digests"]["scaler"] == po.array_digest(np.concatenate([s["mean"], s["std"]]))
    assert base["parameter_digests"]["pca"] == po.array_digest(np.concatenate([p["mean"], p["components"].ravel()]))


def test_fit_code_has_no_oracle_path(data):
    times, feats, table, folds = data
    out = po.fit_predict_fold(folds["folds"][0], feats,
                              po.FoldLabels(table, folds["folds"][0]["train_ids"], "0"))
    for pred in out["predictions"].values():
        assert {"M4", "M5"}.isdisjoint(pred) and {"M0", "M1", "M2", "M3"} <= set(pred)
    src = inspect.getsource(po.fit_predict_fold) + inspect.getsource(po.fit_predict)
    assert "realized_losses" not in src and "cpu_loss" not in src and "M4" not in src


def test_learners_are_deterministic_and_sane():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(60, 8))
    km1, km2 = po.fit_kmeans(P, 3), po.fit_kmeans(P, 3)
    assert np.array_equal(km1["centroids"], km2["centroids"])
    Y = P @ rng.normal(size=(8, 4)) + 0.5
    r = po.fit_ridge(P, Y, 1e-8)
    assert np.allclose(po.apply_ridge(r, P), Y, atol=1e-6)
    assert po.ridge_action(np.array([-1.0, -2.0, -0.1, -3.0])) == 0
    assert po.ridge_action(np.array([-1.0, 2.0, 2.0, 0.0])) == 2
    assert po.best_static(np.zeros((5, 5))) == 0          # ties -> no-edit


def _losses(times):
    rng = np.random.default_rng(9)
    rows = []
    for i in times:
        for c in po.CANDIDATES:
            for h in po.LEADS:
                rows.append({"issue_id": i, "candidate_id": c, "lead_hours": h, "status": "PASS",
                             "cpu_loss": float(1.0 + rng.normal(scale=0.01))})
    return rows


def test_score_bootstrap_by_block_with_pinned_seed(data):
    times, _, _, folds = data
    rows = _losses(times)
    acts = {i: {"M0": "reference", "M1": "expert_0", "M2": "expert_1", "M3": "reference"} for i in times}
    f0 = {i: {"6": 1.0, "24": 1.0, "72": 1.0} for i in times}
    real = po.realized_losses(rows, acts, f0)
    blocks = {i: po.block_of(t) for i, t in times.items()}
    a = po.paired_block_bootstrap(real["losses"], blocks, draws=500)
    b = po.paired_block_bootstrap(real["losses"], blocks, draws=500)
    assert a == b and a["n_blocks"] == 23 and a["seed"] == po.SEED
    assert a["comparisons"]["M3_vs_Fs"]["point"] == 0.0
    assert a["comparisons"]["M3_vs_Fs"]["status_vs_delta_min"] == "BELOW"
    assert "harm72_guard_pass" in a["comparisons"]["M1_vs_Fs_72h"]
    c = po.paired_block_bootstrap(real["losses"], blocks, draws=500, seed=1)
    assert c["comparisons"]["M1_vs_Fs"]["ci_low"] != a["comparisons"]["M1_vs_Fs"]["ci_low"]
    assert po.paired_block_bootstrap(real["losses"], blocks, draws=200, block_days_factor=2)["n_blocks"] < 23


def test_g_f0_correction_uses_matching_lead(data):
    times, _, _, _ = data
    rows = _losses(times)
    # Make the F0-vs-Fs harm deliberately different at every lead.
    f0 = {i: {"6": 0.8, "24": 0.95, "72": 1.2} for i in times}
    acts = {i: {"M0": "reference", "M1": "expert_0", "M2": "expert_1", "M3": "reference"}
            for i in times}
    real = po.realized_losses(rows, acts, f0)
    blocks = {i: po.block_of(t) for i, t in times.items()}
    out = po.paired_block_bootstrap(real["losses"], blocks, draws=200)
    h = out["H_Fs_by_lead"]
    assert h["6"]["point"] != pytest.approx(h["24"]["point"])
    assert h["72"]["point"] != pytest.approx(h["24"]["point"])


def test_cli_poison_scope_includes_eval_and_purged(data):
    _, _, table, folds = data
    poisoned, ids = policy_cli._poison_fold_labels(table, folds["folds"][0], float("nan"))
    expected = folds["folds"][0]["eval_ids"] + folds["folds"][0]["purged_ids"]
    assert ids == expected
    assert all(np.isnan(v) for i in expected for v in poisoned[i].values())
    untouched = set(table) - set(expected)
    assert all(poisoned[i] == table[i] for i in untouched)


def test_nonfinite_choice_falls_back_to_fs_and_is_counted(data):
    times, _, _, _ = data
    rows = _losses(times)
    first = sorted(times)[0]
    for r in rows:
        if r["issue_id"] == first and r["candidate_id"] == "expert_0":
            r["status"], r["cpu_loss"] = "FAIL_NONFINITE", None
    acts = {i: {"M0": "reference", "M1": "expert_0", "M2": "reference", "M3": "reference"} for i in times}
    real = po.realized_losses(rows, acts, {})
    ref = {(r["issue_id"], r["lead_hours"]): r["cpu_loss"] for r in rows if r["candidate_id"] == "reference"}
    assert real["losses"]["M1"][24][first] == ref[(first, 24)]
    assert real["failures_issue_count"]["M1"] == 1
    for r in rows:
        if r["issue_id"] == first and r["candidate_id"] == "reference":
            r["status"] = "FAIL_NONFINITE"
    with pytest.raises(po.PolicyViolation):
        po.realized_losses(rows, acts, {})


def test_gain_table_from_matrix_rows():
    rows = [{"issue_id": "a", "candidate_id": c, "lead_hours": 24, "status": "PASS",
             "gain_vs_fs": 0.0 if c == "reference" else 0.1} for c in po.CANDIDATES]
    rows[2] = {**rows[2], "status": "FAIL_NONFINITE", "gain_vs_fs": None}
    t = po.realized_gain_table(rows)
    assert t["a"]["expert_1"] == 0.0 and t["a"]["expert_0"] == 0.1
    with pytest.raises(po.PolicyViolation):
        po.realized_gain_table(rows[:-1])
