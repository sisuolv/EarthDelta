"""FP-05b consumption chain: every real entry point refuses BEFORE any data read or
model load when the protocol / freeze / clearance / chain is not satisfied.

Spies count calls to the real bridge builder, the rollout, split_freeze's store
opener, xarray.open_zarr and zarr.open; every refusal must leave them at 0.
Runs against the frozen FP-05b protocol (mutated copies live in tmp_path).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "plans/plan_v4_0923/run_20260924T104725Z_fp05b"
PROTOCOL = RUN / "protocol/cache_protocol_v1.json"
FP04_ADMISSION = REPO / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/admission/bank_fit_admission.json"

pytestmark = pytest.mark.skipif(not PROTOCOL.is_file(), reason="cache protocol not frozen yet")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture()
def spies(monkeypatch):
    import xarray
    import zarr

    import scripts.r2_fs_bank_train as r2
    from earthdelta import candidate_cache as cc
    from earthdelta import split_freeze as sf

    calls = {"build_real_bridge": 0, "rollout_issue": 0, "open_store": 0, "open_zarr": 0,
             "zarr_open": 0, "verify_bank_bundle": 0}

    def counter(name):
        def f(*a, **k):
            calls[name] += 1
            raise AssertionError(f"{name} reached")
        return f

    monkeypatch.setattr(r2, "build_real_bridge", counter("build_real_bridge"))
    monkeypatch.setattr(cc, "rollout_issue", counter("rollout_issue"))
    monkeypatch.setattr(sf, "_open_store", counter("open_store"))
    monkeypatch.setattr(xarray, "open_zarr", counter("open_zarr"))
    monkeypatch.setattr(zarr, "open", counter("zarr_open"))
    return calls


def zero(calls, *, allow=()):
    return all(v == 0 for k, v in calls.items() if k not in allow)


def real():
    return json.loads(PROTOCOL.read_text()), sha(PROTOCOL)


def mutated(tmp_path, fn, name="p.json"):
    p, _ = real()
    p = copy.deepcopy(p)
    fn(p)
    path = tmp_path / name
    path.write_text(json.dumps(p, indent=1))
    return path, sha(path)


def cache(path, digest, config="CJ1_W0", device="cuda:0", out=None, extra=()):
    import scripts.r4_candidate_cache as rc

    argv = ["cache", "--protocol", str(path), "--protocol-sha256", digest, "--config-id", config,
            "--device", device, "--output-dir", str(out), *extra]
    return rc.main(argv)


def refused_phase(out):
    return json.loads((Path(out) / "binding_refused.json").read_text())


def test_protocol_is_consistent_without_io(spies):
    import scripts.r4_candidate_cache as rc
    from earthdelta import candidate_cache as cc

    p, digest = real()
    rc.load_cache_protocol(PROTOCOL, digest)
    freeze = rc.bind_freeze(p)
    for w in range(4):
        cfg, role, ids, shard = rc.resolve_worker(p, digest, f"CJ1_W{w}")
        record, adm = rc.role_admission(p, role)
        rep = rc.clear_rows(freeze, role, cc.rows_and_certificates(record, ids),
                            rc.allow_list(p, freeze, role), adm)
        assert rep["passed"] and len(ids) == 4
    record, adm = rc.role_admission(p, "policy_dev")
    ids = p["split"]["allow_lists"]["policy_dev"]["issue_ids"]
    assert len(ids) == 112
    rc.clear_rows(freeze, "policy_dev", cc.rows_and_certificates(record, ids),
                  rc.allow_list(p, freeze, "policy_dev"), adm)
    rc.check_sources(p)
    assert p["split"]["coverage_caveat"]["n_strata_represented"] == 23
    assert zero(spies)


def test_cache_refuses_wrong_protocol_sha(tmp_path, spies):
    assert cache(PROTOCOL, "0" * 64, out=tmp_path / "o") == 2
    assert refused_phase(tmp_path / "o")["phase"].startswith("E0a") and zero(spies)


def test_cache_refuses_wrong_freeze_sha(tmp_path, spies):
    path, d = mutated(tmp_path, lambda p: p["inputs"]["split_freeze"].update(sha256="1" * 64))
    assert cache(path, d, out=tmp_path / "o") == 2
    assert refused_phase(tmp_path / "o")["code"] == "FREEZE_SHA256_MISMATCH" and zero(spies)


def test_cache_refuses_non_clear_row(tmp_path, spies):
    ninth = json.loads(FP04_ADMISSION.read_text())["admission"]["admitted"][8]["issue_id"]
    path, d = mutated(tmp_path, lambda p: p["configs"]["CJ1_W0"]["issue_ids"].append(ninth))
    assert cache(path, d, out=tmp_path / "o") == 2
    assert refused_phase(tmp_path / "o")["code"] == "ROW_NOT_CLEAR" and zero(spies)


def test_cache_refuses_policy_dev_row_off_allow_list(tmp_path, spies):
    def fn(p):
        p["split"]["allow_lists"]["policy_dev"]["issue_ids"] = p["split"]["allow_lists"]["policy_dev"]["issue_ids"][1:]
    path, d = mutated(tmp_path, fn)
    dev = _dev_protocol(tmp_path, d, verdict="PASS")
    rc = cache(path, d, config="CJ2_S0", out=tmp_path / "o",
               extra=["--dev-protocol", str(dev), "--dev-protocol-sha256", sha(dev)])
    assert rc == 2 and zero(spies)


def _dev_protocol(tmp_path, protocol_sha, verdict):
    dec = tmp_path / f"decision_{verdict}.json"
    dec.write_text(json.dumps({"verdict": verdict, "protocol_sha256": protocol_sha}))
    p, _ = real()
    ids = p["split"]["allow_lists"]["policy_dev"]["issue_ids"]
    dev = tmp_path / f"dev_{verdict}.json"
    dev.write_text(json.dumps({"protocol_sha256": protocol_sha,
                               "debug_decision": {"path": str(dec), "sha256": sha(dec)},
                               "dev_scale": {"verdict": "CHOSEN"}, "issue_ids": ids,
                               "shard_map": {i: k % 4 for k, i in enumerate(ids)}}))
    return dev


def test_dev_cache_refuses_without_pass_debug_decision(tmp_path, spies):
    _, digest = real()
    assert cache(PROTOCOL, digest, config="CJ2_S0", out=tmp_path / "a") == 2
    dev = _dev_protocol(tmp_path, digest, verdict="STOP")
    assert cache(PROTOCOL, digest, config="CJ2_S0", out=tmp_path / "b",
                 extra=["--dev-protocol", str(dev), "--dev-protocol-sha256", sha(dev)]) == 2
    assert refused_phase(tmp_path / "b")["code"] == "DEBUG_DECISION_NOT_PASS"
    assert cache(PROTOCOL, digest, config="CJ2_S0", out=tmp_path / "c",
                 extra=["--dev-protocol", str(dev), "--dev-protocol-sha256", "2" * 64]) == 2
    assert zero(spies)


def test_cache_refuses_source_drift_and_device_mismatch(tmp_path, spies):
    path, d = mutated(tmp_path, lambda p: p["source_at_preregistration"].update(
        {"earthdelta/split_freeze.py": "3" * 64}))
    assert cache(path, d, out=tmp_path / "a") == 2
    assert refused_phase(tmp_path / "a")["code"] == "SOURCE_DRIFT"
    _, digest = real()
    assert cache(PROTOCOL, digest, device="cuda:1", out=tmp_path / "b") == 2
    assert zero(spies)


def test_cache_refuses_s0_certificate_mismatch_before_model(tmp_path, spies):
    path, d = mutated(tmp_path, lambda p: p["inputs"]["s0_certificate"].update(sha256="4" * 64))
    assert cache(path, d, out=tmp_path / "o") == 2
    assert refused_phase(tmp_path / "o")["phase"] == "E0b_before_model_load" and zero(spies)


def test_cache_refuses_wrong_fs_manifest_and_bank_before_model(tmp_path, spies, monkeypatch):
    from earthdelta import fs_protocol as fp

    monkeypatch.setattr(fp, "consume_s0_certificate", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(fp, "official_backend_precondition", lambda **k: {"passed": True})
    path, d = mutated(tmp_path, lambda p: p["inputs"]["fs_reference"]["reference_manifest"].update(
        sha256="5" * 64), "fs.json")
    assert cache(path, d, out=tmp_path / "a") == 2
    assert refused_phase(tmp_path / "a")["code"] == "FS_REFERENCE_NOT_AUTHORIZED"
    path, d = mutated(tmp_path, lambda p: p["inputs"]["bank_bundle"].update(
        fp04_protocol_sha256="6" * 64), "bank.json")
    assert cache(path, d, out=tmp_path / "b") == 2
    assert refused_phase(tmp_path / "b")["code"] == "BANK_BUNDLE_INVALID"
    path, d = mutated(tmp_path, lambda p: p["inputs"]["registry_entries"][1].update(
        coefficients=[0.0, 0.25, 0.0, 0.0]), "reg.json")
    assert cache(path, d, out=tmp_path / "c") == 2
    assert refused_phase(tmp_path / "c")["code"] == "REGISTRY_NOT_THE_PINNED_ONE"
    assert zero(spies)


def test_merge_and_decide_refuse_before_io(tmp_path, spies):
    import scripts.r4_cache_decide as rd
    import scripts.r4_candidate_cache as rc

    _, digest = real()
    dev = _dev_protocol(tmp_path, digest, "PASS")
    base = ["merge", "--protocol", str(PROTOCOL), "--dev-protocol", str(dev),
            "--shard-dirs", str(tmp_path), "--out", str(tmp_path / "m")]
    assert rc.main(base[:3] + ["--protocol-sha256", "0" * 64] + base[3:5]
                   + ["--dev-protocol-sha256", sha(dev)] + base[5:]) == 2
    assert rc.main(base[:3] + ["--protocol-sha256", digest] + base[3:5]
                   + ["--dev-protocol-sha256", "7" * 64] + base[5:]) == 2
    stop = _dev_protocol(tmp_path, digest, "STOP")
    assert rc.main(base[:3] + ["--protocol-sha256", digest, "--dev-protocol", str(stop),
                               "--dev-protocol-sha256", sha(stop)] + base[5:]) == 2
    assert not (tmp_path / "m").exists()
    assert rd.main(["--kind", "debug", "--protocol", str(PROTOCOL), "--protocol-sha256", "0" * 64,
                    "--job-dir", str(tmp_path), "--out", str(tmp_path / "d.json")]) == 2
    path, d = mutated(tmp_path, lambda p: p["configs"]["CJ1_W0"].update(
        own_issue_ids=["iss_not_on_the_list", p["configs"]["CJ1_W0"]["own_issue_ids"][1]]))
    assert rd.main(["--kind", "debug", "--protocol", str(path), "--protocol-sha256", d,
                    "--job-dir", str(tmp_path), "--out", str(tmp_path / "d.json")]) == 2
    assert not (tmp_path / "d.json").exists() and zero(spies)


def _manifest(tmp_path, protocol_sha, passed=True, name="cache_manifest.json"):
    p, _ = real()
    fr = json.loads(Path(p["inputs"]["split_freeze"]["path"]).read_text())
    m = tmp_path / name
    m.write_text(json.dumps({"passed": passed, "protocol_sha256": protocol_sha,
                             "split_freeze_sha256": p["inputs"]["split_freeze"]["sha256"],
                             "issue_ids": p["split"]["allow_lists"]["policy_dev"]["issue_ids"][:20]}))
    assert fr["schema"] == "ed-split-freeze/1"
    return m


def test_policy_stages_refuse(tmp_path, spies):
    import scripts.r4_policy_oof as rp

    _, digest = real()
    good = _manifest(tmp_path, digest)
    bad = _manifest(tmp_path, digest, passed=False, name="bad.json")
    common = ["--protocol", str(PROTOCOL), "--protocol-sha256", digest]
    assert rp.main(["folds", "--protocol", str(PROTOCOL), "--protocol-sha256", "0" * 64,
                    "--cache-manifest", str(good), "--out", str(tmp_path / "f.json")]) == 2
    assert rp.main(["folds", *common, "--cache-manifest", str(bad), "--out", str(tmp_path / "f.json")]) == 2
    assert rp.main(["fit-predict", *common, "--cache-manifest", str(bad), "--folds", str(tmp_path / "f.json"),
                    "--out-dir", str(tmp_path / "pp")]) == 2
    assert rp.main(["score", *common, "--cache-manifest", str(good), "--prediction-freeze",
                    str(tmp_path / "missing.json"), "--out-dir", str(tmp_path / "ev")]) == 2
    fz = tmp_path / "prediction_freeze.json"
    fz.write_text(json.dumps({"protocol_sha256": digest, "cache_manifest_sha256": "8" * 64,
                              "per_fold_files": {}}))
    assert rp.main(["score", *common, "--cache-manifest", str(good), "--prediction-freeze", str(fz),
                    "--out-dir", str(tmp_path / "ev")]) == 2
    assert not (tmp_path / "ev").exists() and not (tmp_path / "f.json").exists() and zero(spies)


def test_certify_admission_refuses_before_reading(tmp_path, spies):
    import scripts.r4_candidate_cache as rc

    p, _ = real()
    fr = p["inputs"]["split_freeze"]
    assert rc.main(["certify-admission", "--role", "debug", "--freeze", fr["path"], "--freeze-sha256",
                    "9" * 64, "--admission", str(FP04_ADMISSION), "--out", str(tmp_path / "c.json")]) == 2
    # FP-04's bank_fit rows offered as policy_dev: the freeze refuses them (exposed 2020 rows)
    assert rc.main(["certify-admission", "--role", "policy_dev", "--freeze", fr["path"], "--freeze-sha256",
                    fr["sha256"], "--admission", str(FP04_ADMISSION), "--out", str(tmp_path / "c.json")]) == 2
    assert not (tmp_path / "c.json").exists() and zero(spies)
