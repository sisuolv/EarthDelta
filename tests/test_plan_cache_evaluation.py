"""FP-05 candidate cache: CACHE-VALID-v1 negatives, legal features, the worker's
rollout math on a synthetic Stormer, the role-parameterised admission copy, and
the pure DEBUG-DECIDE-v1 / DEV-SCALE-v1 deciders. CPU only; no real store."""
from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from earthdelta import bank_training as bt
from earthdelta import candidate_cache as cc
from earthdelta import static_adapter as sa

PINS = [{"plan_id": p, "coefficients": c, "rho": 0.25, "hold_steps": 4, "interval_hours": 6,
         "continuation": "reference_after_hold", "is_reference": p == "reference",
         "support": [i for i, a in enumerate(c) if a]}
        for p, c in (("reference", [0.0] * 4), ("expert_0", [0.25, 0, 0, 0]),
                     ("expert_1", [0, 0.25, 0, 0]), ("expert_2", [0, 0, 0.25, 0]),
                     ("expert_3", [0, 0, 0, 0.25]))]
for p in PINS:
    p["coefficients"] = [float(a) for a in p["coefficients"]]


# ----------------------------------------------------------------- synthetic matrix
def _q_scale(v=3, h=4):
    q = sa.area_weight_q(torch.linspace(-60, 60, h, dtype=torch.float64))
    return q, torch.ones(1, v, 1, 1, dtype=torch.float64) * 2.0


def _issue(iid, rng, shard=0, nan=False):
    end = rng.normal(size=(5, 3, 3, 4, 8)).astype(np.float32)
    truth = rng.normal(size=(3, 3, 4, 8)).astype(np.float32)
    if nan:
        end[2, 1, 0, 0, 0] = np.nan
    return end, truth


def build_rows(n=3, nan_issue=None, identity="idsha"):
    rng = np.random.default_rng(0)
    q, scale = _q_scale()
    rows, rehash, ids = [], {}, []
    for k in range(n):
        iid = f"iss_{k}"
        ids.append(iid)
        end, truth = _issue(iid, rng, nan=(k == nan_issue))
        cpu = cc.cpu_losses(end, truth, q, scale)
        ana = cc.analytic_gains(end, truth, q, scale)
        issue = {"issue_id": iid, "role": "policy_dev", "shard": k % 4, "issue_time": 1562414400 + k * 21600,
                 "gpu_losses": {p["plan_id"]: {str(h): cpu[ci][j] for j, h in enumerate(cc.LEAD_HOURS)}
                                for ci, p in enumerate(PINS)},
                 "status": {p["plan_id"]: {str(h): (cc.STATUS_PASS if cpu[ci][j] is not None
                                                    else cc.STATUS_NONFINITE)
                                           for j, h in enumerate(cc.LEAD_HOURS)} for ci, p in enumerate(PINS)},
                 "files": {"endpoints": {"path": f"/x/{iid}/e.npy", "sha256": "e"},
                           "truth": {"path": f"/x/{iid}/t.npy", "sha256": "t"}},
                 "seconds": {}}
        rows += cc.matrix_rows_for_issue(issue, cpu=cpu, analytic=ana, entries_pin=PINS,
                                         identity_sha256=identity, features_sha256="f")
        rehash[iid] = {"endpoints_ok": True, "truth_ok": True, "shape_dtype_ok": True}
    return rows, rehash, ids


def validate(rows, rehash, ids, **kw):
    args = dict(expected_issue_ids=ids, entries_pin=PINS, expected_identity_sha256="idsha",
                shard_of={i: int(i.split("_")[1]) % 4 for i in ids}, rehash=rehash,
                read_check={"passed": True})
    args.update(kw)
    return cc.validate_cache_matrix(rows, **args)


def test_valid_matrix_passes_and_v5_v6_hold():
    rows, rehash, ids = build_rows()
    rep = validate(rows, rehash, ids)
    assert rep["passed"] and rep["n_rows"] == 3 * 5 * 3
    ref = [r for r in rows if r["candidate_id"] == "reference"]
    assert all(r["gain_vs_fs"] == 0.0 for r in ref)


@pytest.mark.parametrize("mutate,v", [
    (lambda rows: rows.pop(4), "V1"),                                        # missing lead
    (lambda rows: rows.append(dict(rows[0])), "V1"),                          # duplicate
    (lambda rows: rows.__setitem__(3, {**rows[3], "coefficients": [0, 0.25, 0, 0]}), "V2"),
    (lambda rows: rows.__setitem__(3, {**rows[3], "support": [1]}), "V2"),   # index swap
    (lambda rows: rows.__setitem__(0, {**rows[0], "identity_sha256": "wrong-bank"}), "V2"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "role": "debug"}), "V2"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "status": "SKIPPED"}), "V4"),
    (lambda rows: rows.__setitem__(5, {**rows[5], "gpu_loss": rows[5]["cpu_loss"] * (1 + 1e-9)}), "V5"),
    (lambda rows: rows.__setitem__(5, {**rows[5], "analytic_gain": 1.0}), "V6"),
    (lambda rows: rows.__setitem__(5, {**rows[5], "valid_time": rows[5]["valid_time"] + 3600}), "V7"),
    (lambda rows: rows.__setitem__(5, {**rows[5], "shard": 3}), "V1"),
])
def test_matrix_negatives(mutate, v):
    rows, rehash, ids = build_rows()
    mutate(rows)
    with pytest.raises(cc.CacheViolation) as exc:
        validate(rows, rehash, ids)
    assert exc.value.code == "INVALID_EVIDENCE"
    assert exc.value.detail["v_pass"][v] is False


def test_byte_mutation_and_subset_and_read_set_fail():
    rows, rehash, ids = build_rows()
    bad = copy.deepcopy(rehash)
    bad["iss_1"]["endpoints_ok"] = False
    with pytest.raises(cc.CacheViolation) as exc:
        validate(rows, bad, ids)
    assert exc.value.detail["v_pass"]["V3"] is False
    subset = [r for r in rows if r["issue_id"] != "iss_2"]
    with pytest.raises(cc.CacheViolation):
        validate(subset, rehash, ids)                      # successful subset only -> V1
    with pytest.raises(cc.CacheViolation) as exc:
        validate(rows, rehash, ids, read_check={"passed": False, "violations": ["x"]})
    assert exc.value.detail["v_pass"]["V8"] is False


def test_nan_endpoint_is_fail_nonfinite_not_dropped():
    rows, rehash, ids = build_rows(n=60, nan_issue=1)
    nf = [r for r in rows if r["status"] == cc.STATUS_NONFINITE]
    assert len(nf) == 1 and nf[0]["cpu_loss"] is None and nf[0]["gpu_loss"] is None
    assert validate(rows, rehash, ids)["n_nonfinite"] == 1       # kept in the denominator
    rows2, rehash2, ids2 = build_rows(n=3, nan_issue=1)          # 1/45 > 2% -> INVALID
    with pytest.raises(cc.CacheViolation) as exc:
        validate(rows2, rehash2, ids2)
    assert exc.value.detail["v_pass"]["V4"] is False
    fake = [dict(r) for r in rows]
    i = rows.index(nf[0])
    fake[i] = {**fake[i], "status": cc.STATUS_PASS}
    with pytest.raises(cc.CacheViolation):
        validate(fake, rehash, ids)


def test_matrix_digest_is_order_independent():
    rows, _, _ = build_rows()
    assert cc.matrix_digest(rows) == cc.matrix_digest(list(reversed(rows)))
    rows[0] = {**rows[0], "cpu_loss": rows[0]["cpu_loss"] + 1e-15}
    assert cc.matrix_digest(rows) != cc.matrix_digest(build_rows()[0])


# ----------------------------------------------------------------- features
def test_features_take_no_truth_and_are_deterministic():
    params = list(inspect.signature(cc.legal_features_np).parameters)
    assert params == ["x_t", "x_tm6", "x_tm12", "issue_time", "inp_mean", "inp_std", "grid"]
    rng = np.random.default_rng(1)
    x = [rng.normal(size=(69, 128, 256)).astype(np.float32) for _ in range(3)]
    mean, std = rng.normal(size=69), rng.uniform(0.5, 2, size=69)
    f1 = cc.legal_features_np(*x, 1562414400, mean, std)
    f2 = cc.legal_features_np(*[a.copy() for a in x], 1562414400, mean, std)
    assert f1.shape == (cc.N_FEATURES,) == (1660,) and f1.dtype == np.float64
    assert f1.tobytes() == f2.tobytes()
    x2 = [a.copy() for a in x]
    x2[0][0, 0, 0] += 1.0
    assert cc.legal_features_np(*x2, 1562414400, mean, std).tobytes() != f1.tobytes()


# ----------------------------------------------------------------- worker math (synthetic Stormer)
@pytest.fixture(scope="module")
def synthetic():
    import scripts.r2_fs_bank_train as r2

    threads = torch.get_num_threads()
    torch.set_num_threads(2)   # the CPU container has a 2-core quota; 96 threads thrash
    yield _synthetic_parts(r2)
    torch.set_num_threads(threads)


def _synthetic_parts(r2):
    bridge, variables, lat = r2.build_synthetic_bridge(torch.device("cpu"))
    fs_bridge = bridge          # a synthetic stand-in for the merged Fs
    blocks = tuple(range(len(bridge.model.blocks)))
    hidden = int(bridge.model.blocks[0].attn.proj.in_features)
    bank = bt.build_dynamic_bank(hidden, blocks, num_experts=4, rank_per_expert=4, seed=7)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for lora in bank.values():
            for up in lora.up:
                up.weight.copy_(torch.randn(up.weight.shape, generator=g) * 0.05)
            lora.requires_grad_(False)
    registry = bt.build_bank_registry(4)
    spec = sa.build_objective_spec(bridge, lat, lead_steps=(4,), space="raw")
    return bridge, fs_bridge, variables, lat, blocks, bank, registry, spec


def test_rollout_issue_matches_fp04_path_and_cpu_recompute(synthetic):
    bridge, fs_bridge, variables, lat, blocks, bank, registry, spec = synthetic
    rng = np.random.default_rng(5)
    shape = (len(variables), 16, 32)
    x = rng.normal(size=shape).astype(np.float32)
    truth = {s: rng.normal(size=shape).astype(np.float32) for s in cc.LEAD_STEPS}
    row = {"issue_id": "iss_syn", "issue_time": 1562414400, "valid_time": 1562414400 + 72 * 3600,
           "history_time": 1562414400 - 43200, "data_role": "policy_dev", "issue_store": "syn"}
    sample = cc.build_sample(bridge, row, x, truth, torch.device("cpu"))
    entries = list(registry.entries)
    assert [e.plan_id for e in entries] == list(cc.CANDIDATE_IDS)
    hooks0 = cc.count_hooks(fs_bridge.model)
    res = cc.rollout_issue(fs_bridge, bridge, bank, entries, sample, spec, variables, blocks)
    assert res["endpoints"].shape == (5, 3, *shape) and res["endpoints"].dtype == np.float32
    assert res["reference_equals_plain_fs_bitwise"]                      # A3 re-check
    assert all(h == hooks0 for h in res["hooks_after"].values())
    # the FP-04 panel functions give the same numbers
    fs_panel = sa.fs_panel_losses(fs_bridge, [sample], spec, lead_steps=cc.LEAD_STEPS,
                                  fs_adapters=None, target_blocks=blocks, variables=variables)
    assert res["candidates"]["reference"]["losses"] == fs_panel["iss_syn"]
    for k in range(4):
        plan = bt.expert_plan(registry, k)
        panel = bt.bank_panel_losses(fs_bridge, bank, plan, [sample], spec, lead_steps=cc.LEAD_STEPS,
                                     target_blocks=blocks, variables=variables)
        assert res["candidates"][f"expert_{k}"]["losses"] == panel["iss_syn"]
    # V5 / V6 on the stored raw endpoints
    truth_arr = np.stack([truth[s] for s in cc.LEAD_STEPS])
    cpu = cc.cpu_losses(res["endpoints"], truth_arr, spec.q, spec.scale)
    ana = cc.analytic_gains(res["endpoints"], truth_arr, spec.q, spec.scale)
    for ci, cid in enumerate(cc.CANDIDATE_IDS):
        for j, h in enumerate(cc.LEAD_HOURS):
            gpu = res["candidates"][cid]["losses"][str(h)]
            assert abs(cpu[ci][j] - gpu) <= cc.V5_RELATIVE_TOL * cpu[ci][j]
            g = cpu[0][j] - cpu[ci][j]
            assert abs(g - ana[ci][j]) <= cc.V6_ATOL + cc.V6_RTOL * abs(g)
    assert any(res["candidates"][f"expert_{k}"]["losses"]["24"] != res["candidates"]["reference"]["losses"]["24"]
               for k in range(4))


def test_nonfinite_input_gives_fail_nonfinite(synthetic):
    bridge, fs_bridge, variables, lat, blocks, bank, registry, spec = synthetic
    shape = (len(variables), 16, 32)
    x = np.zeros(shape, dtype=np.float32)
    x[0, 0, 0] = np.nan
    truth = {s: np.zeros(shape, dtype=np.float32) for s in cc.LEAD_STEPS}
    row = {"issue_id": "iss_nan", "issue_time": 0, "data_role": "policy_dev", "issue_store": "syn"}
    sample = cc.build_sample(bridge, row, x, truth, torch.device("cpu"))
    res = cc.rollout_issue(fs_bridge, bridge, bank, list(registry.entries), sample, spec, variables, blocks)
    assert res["candidates"]["reference"]["status"]["24"] == cc.STATUS_NONFINITE
    assert res["candidates"]["reference"]["losses"]["24"] is None


def test_registry_pin_check(synthetic):
    registry = synthetic[6]
    pin = cc.registry_entries_pin(registry.to_dict())
    assert cc.check_registry_entries(registry.entries, pin)["passed"]
    bad = copy.deepcopy(pin)
    bad[1]["coefficients"] = [0.0, 0.25, 0.0, 0.0]
    with pytest.raises(cc.CacheViolation):
        cc.check_registry_entries(registry.entries, bad)


# ----------------------------------------------------------------- role-parameterised admission copy
def _synthetic_record(tmp_path, tag="bank_fit"):
    xr = pytest.importorskip("xarray")
    n = 40
    t = np.datetime64("2019-07-01T00", "ns") + np.arange(n) * np.timedelta64(6, "h")
    ds = xr.Dataset({"data": (("time", "channel", "lat", "lon"), np.ones((n, 2, 2, 2), np.float32))},
                    coords={"time": t})
    store = tmp_path / "2019.zarr"
    ds.to_zarr(store)
    row = {"admitted": True, "formal": True, "issue_id": "iss_a", "data_role": tag, "interval_hours": 6,
           "history_steps": 2, "lead_hours": 72.0, "history_store": str(store), "issue_store": str(store),
           "target_store": str(store), "normalization_hash": "n", "grid_hash": "g", "issue_index": 10,
           "issue_time": int(t[10].astype("datetime64[s]").astype(int)), "process_group_id": "pg"}
    cert = {"issue_id": "iss_a", "passed": True, "data_role": tag, "process_group_id": "pg",
            "store_path": str(store), "grid_hash": "g", "variable_order_hash": "v",
            "indices": list(range(8, 23)), "times_utc": [str(x)[:19] for x in t[8:23]]}
    return {"passed": True, "formal": True, "data_role": tag,
            "results": {"b09_real_sample_admission": True, "b08_content_verification": True},
            "admission": {"passed": True, "formal": True, "data_role": tag, "admitted": [row]},
            "content_certificates": [cert]}


def test_certify_for_role_bank_fit_equals_fs_copy(tmp_path):
    rec = _synthetic_record(tmp_path)
    kw = dict(lead_steps=(1, 4, 12), required_history_steps=2, expected_normalization_digest="n",
              expected_grid_hash="g", expected_variable_order_hash="v", reverify_content=False)
    a = sa.certify_admission_for_fs(rec, **kw)
    b = cc.certify_admission_for_role(rec, expected_tag="bank_fit", **kw)
    drop = ("check", "expected_tag", "tag_is_authorisation", "only_issue_ids")
    assert {k: v for k, v in b.items() if k not in drop} == {k: v for k, v in a.items() if k != "check"}
    with pytest.raises(cc.CacheViolation) as exc:
        cc.certify_admission_for_role(rec, expected_tag="policy_dev", **kw)
    codes = {f["code"] for f in exc.value.detail["failures"]}
    assert {"TAG_MISMATCH", "ROW_TAG_MISMATCH", "CERT_TAG_MISMATCH"} <= codes
    forged = copy.deepcopy(rec)
    forged["content_certificates"][0]["indices"] = list(range(9, 23))
    with pytest.raises(cc.CacheViolation):
        cc.certify_admission_for_role(forged, expected_tag="bank_fit", **kw)
    with pytest.raises(cc.CacheViolation):
        cc.certify_admission_for_role(rec, expected_tag="bank_fit", only_issue_ids=["iss_zz"], **kw)


# ----------------------------------------------------------------- deciders
def _anchors(delta=0.0):
    return {f"i{k}": {"fs": [(1.0 + delta, 1.0)] * 3, "expert": {"k": 0, "pairs": [(2.0, 2.0)] * 3}}
            for k in range(8)}


def test_debug_decide():
    ok = dict(e0_by_worker={f"W{i}": True for i in range(4)}, matrix_passed=True,
              duplicates_identical={f"i{k}": True for k in range(8)},
              reference_equals_plain_fs={f"i{k}": True for k in range(8)})
    assert cc.decide_debug(anchors=_anchors(), **ok)["verdict"] == "PASS"
    assert cc.decide_debug(anchors=_anchors(1e-9), **ok)["verdict"] == "DEBUG_ANCHOR_INEXACT"
    assert cc.decide_debug(anchors=_anchors(1e-3), **ok)["verdict"] == "STOP"
    bad = dict(ok, duplicates_identical={**ok["duplicates_identical"], "i3": False})
    assert cc.decide_debug(anchors=_anchors(), **bad)["verdict"] == "STOP"
    bad = dict(ok, e0_by_worker={"W0": True, "W1": True, "W2": True})
    assert cc.decide_debug(anchors=_anchors(), **bad)["verdict"] == "STOP"


def test_dev_scale_reads_no_loss_and_is_mechanical():
    params = set(inspect.signature(cc.decide_dev_scale).parameters)
    assert not any("loss" in p or "gain" in p for p in params)
    subsets = {"N": [f"i{k}" for k in range(112)], "N/2": [f"i{k}" for k in range(0, 112, 2)],
               "N/4": [f"i{k}" for k in range(0, 112, 4)]}
    base = dict(subsets=subsets, admitted_ids=subsets["N"], model_load_seconds=120.0,
                issue_seconds=[30.0] * 16, bytes_per_issue=2e8, free_disk_bytes=1e14, peak_gpu_gib=20.0)
    records = [{"issue_seconds": 30.0, "bytes": 2e8, "gpu_losses": {"x": 1.0}} for _ in range(16)]
    stripped = [{k: v for k, v in r.items() if "loss" not in k} for r in records]
    a = cc.decide_dev_scale(**{**base, "issue_seconds": [r["issue_seconds"] for r in records]})
    b = cc.decide_dev_scale(**{**base, "issue_seconds": [r["issue_seconds"] for r in stripped]})
    assert a == b and a["N"] == 112 and a["chosen_subset"] == "N"
    slow = cc.decide_dev_scale(**{**base, "issue_seconds": [80.0] * 16})
    assert slow["chosen_subset"] == "N/2" and slow["N"] == 56
    assert cc.decide_dev_scale(**{**base, "peak_gpu_gib": 70.0})["verdict"] == "STOP"
    assert cc.decide_dev_scale(**{**base, "admitted_ids": subsets["N/2"]})["chosen_subset"] == "N/2"
    assert cc.shard_map(["a", "b", "c", "d", "e"]) == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 0}
