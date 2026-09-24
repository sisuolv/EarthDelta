#!/usr/bin/env python3
"""Independent C-J2 verification that NEVER reads a loss value.

numpy/json/hashlib only (no earthdelta/scripts imports). gpu_losses.json is only
byte-hashed, never parsed; endpoints/truth/f0 arrays are checked for shape, dtype
and finiteness only. Checks: coverage (N issues == dev_protocol issue_ids == policy_dev
allow-list subset; shards == dev_protocol shard_map, i mod 4 over time order); identity
(shard E0, digests, role, store, row fields == the admission record); bytes (every
npy and gpu_losses re-hashes to issue.json); read set (every worker read inside the
admission row's [history_index, target_index] of 2019.zarr, role policy_dev, and
inside the freeze's policy_dev support window).
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RUN = Path("/mnt/afs/260010168/EarthDelta/plans/plan_v4_0923/run_20260924T104725Z_fp05b")
PROTO_SHA = "a42c11dc68d9ede158370a6ba77c4ef8b54cb5cecde30358ee2424fc18ae96d8"
job, out = Path(sys.argv[1]), Path(sys.argv[2])


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(json.dumps(list(x.shape)).encode()); h.update(x.tobytes())
    return h.hexdigest()


P_path = RUN / "protocol/cache_protocol_v1.json"
assert fsha(P_path) == PROTO_SHA
P = json.loads(P_path.read_text())
dev_path = RUN / "cache/dev_protocol.json"
dev = json.loads(dev_path.read_text())
dev_sha = fsha(dev_path)
adm_ref = P["split"]["allow_lists"]["policy_dev"]["admission"]
assert fsha(adm_ref["path"]) == adm_ref["sha256"]
adm = {r["issue_id"]: r for r in json.loads(Path(adm_ref["path"]).read_text())["admission"]["admitted"]}
frz_ref = P["split"]["freeze"]
assert fsha(frz_ref["path"]) == frz_ref["sha256"]
frz = json.loads(Path(frz_ref["path"]).read_text())["roles"]["policy_dev"]
bank, fsd = P["inputs"]["bank_bundle"], P["inputs"]["fs_reference"]["digests"]
allow = set(P["split"]["allow_lists"]["policy_dev"]["issue_ids"])

fails, checks = [], {}
ids = list(dev["issue_ids"])
checks["dev_protocol_chained"] = (dev["protocol_sha256"] == PROTO_SHA
                                  and fsha(dev["debug_decision"]["path"]) == dev["debug_decision"]["sha256"]
                                  and json.loads(Path(dev["debug_decision"]["path"]).read_text())["verdict"] == "PASS")
checks["N_is_112_unique"] = len(ids) == 112 == len(set(ids))
checks["ids_on_allow_list"] = set(ids) <= allow
checks["ids_admitted"] = all(i in adm for i in ids)
# shard map = i mod 4 over time order
ordered = sorted(ids, key=lambda i: adm[i]["issue_time"])
checks["shard_map_is_i_mod_4_time_order"] = all(int(dev["shard_map"][iid]) == k % 4 for k, iid in enumerate(ordered))
checks["coverage_caveat_in_dev_protocol"] = (dev.get("coverage_caveat") == P["split"]["coverage_caveat"]
                                             and [s["stratum"] for s in dev["coverage_caveat"]["strata"]] == [16, 19])

# strata 16/19 really empty; blocks from 2019-07-06T12Z
origin = dt.datetime(2019, 7, 6, 12, tzinfo=dt.timezone.utc).timestamp()
blocks = sorted({int((adm[i]["issue_time"] - origin) // (7 * 86400)) for i in ids})
checks["n_blocks_represented"] = len(blocks)
checks["strata_16_19_absent"] = 16 not in blocks and 19 not in blocks and len(blocks) == 23

seen = {}
support_lo, support_hi = frz["store_index_window"][0] - 2, frz["store_index_window"][1] + 12
reads_total = 0
per_shard = {}
for s_i in range(4):
    cid = f"CJ2_S{s_i}"
    d = job / f"s{s_i}"
    s = json.loads((d / "shard.json").read_text())
    e = s["e0"]
    ok = (s["config_id"] == cid and s["role"] == "policy_dev" and s["shard"] == s_i
          and s["passed"] and s["e0_passed"] and s["digests_unchanged"]
          and s["protocol_sha256"] == PROTO_SHA
          and s["bank_digest_after"] == bank["bank_digest"] == e["bank_digest"]
          and s["merged_fs_digest_before"] == s["merged_fs_digest_after"] == fsd["merged_backbone_digest"]
          and e["sources"]["passed"] and e["sources"]["n_files"] == 27
          and e["clearance"]["passed"] and e["clearance"]["n_clear"] == 28
          and e["tf32_off"] and e["s0"] and e["fs_identity_pass"] and e["backend_pre"] and e["backend_post"]
          and s["status_counts"] == {"PASS": 420}
          and s["provenance"]["input_files"]["dev_protocol"]["sha256"] == dev_sha)
    mine = sorted(i["issue_id"] for i in s["issues"])
    ok = ok and mine == sorted(i for i in ids if int(dev["shard_map"][i]) == s_i)
    per_shard[cid] = {"ok": bool(ok), "n_issues": len(mine), "peak_gpu_gib": s["peak_gpu_gib"],
                      "model_load_seconds": s["model_load_seconds"]}
    if not ok:
        fails.append(f"shard {cid}")
    for it in s["issues"]:
        iid = it["issue_id"]
        if iid in seen:
            fails.append(f"duplicate {iid}")
        seen[iid] = cid
        dd = Path(it["dir"])
        if fsha(dd / "issue.json") != it["issue_json_sha256"]:
            fails.append(f"{iid} issue.json hash")
        ij = json.loads((dd / "issue.json").read_text())
        a = adm[iid]
        row = ij["row"]
        if not (ij["role"] == "policy_dev" and ij["issue_id"] == iid and ij["shard"] == s_i
                and row["data_role"] == "policy_dev" and row["event_id"] == a["event_id"]
                and row["history_index"] == a["history_index"] and row["issue_index"] == a["issue_index"]
                and row["target_index"] == a["target_index"] and row["issue_store"] == a["issue_store"]
                and Path(a["issue_store"]).name == "2019.zarr" and ij["issue_time"] == a["issue_time"]
                and ij["hooks_clean"] is True):
            fails.append(f"{iid} identity")
        if not all(v == "PASS" for c in ij["status"].values() for v in c.values()):
            fails.append(f"{iid} status")
        for name, shape in (("endpoints", (5, 3, 69, 128, 256)), ("truth", (3, 69, 128, 256)),
                            ("f0_endpoints", (3, 69, 128, 256))):
            fp = dd / f"{name}.npy"
            if fsha(fp) != ij["files"][name]["sha256"]:
                fails.append(f"{iid} {name} rehash")
            arr = np.load(fp, mmap_mode="r", allow_pickle=False)
            if arr.shape != shape or arr.dtype != np.float32 or not np.isfinite(arr).all():
                fails.append(f"{iid} {name} shape/dtype/finite")
            if name == "endpoints":
                if not (arr_sha(np.asarray(arr[0])) == ij["reference_endpoints_array_sha256"]
                        == ij["plain_fs_endpoints_array_sha256"] and ij["reference_equals_plain_fs_bitwise"]):
                    fails.append(f"{iid} reference != plain_fs")
        if fsha(dd / "gpu_losses.json") != ij["files"]["gpu_losses"]["sha256"]:  # bytes only, never parsed
            fails.append(f"{iid} gpu_losses rehash")
        for r in ij["read_log"]:
            reads_total += 1
            if not (r["issue_id"] == iid and r["role"] == "policy_dev" and r["store"] == "2019.zarr"
                    and a["history_index"] <= r["index"] <= a["target_index"]
                    and support_lo <= r["index"] <= support_hi):
                fails.append(f"{iid} read outside certified slice: {r}")

checks["every_issue_exactly_once"] = sorted(seen) == sorted(ids)
checks["worker_reads_total"] = reads_total
res = {"job_dir": str(job), "dev_protocol_sha256": dev_sha, "checks": checks, "per_shard": per_shard,
       "failures": fails, "loss_values_read": False,
       "passed": not fails and all(v for v in checks.values() if isinstance(v, bool))}
with open(out, "x") as f:
    json.dump(res, f, indent=1, sort_keys=True)
print(json.dumps({"passed": res["passed"], "n_failures": len(fails), "failures": fails[:20],
                  "checks": checks, "per_shard_ok": {k: v["ok"] for k, v in per_shard.items()}}))
