#!/usr/bin/env python3
"""Independent C-J1 verification from raw worker files (numpy/json/hashlib only;
does NOT import earthdelta.candidate_cache or scripts.r4_*).

Checks: shard E0/digest records; file re-hash vs issue.json; shape/dtype/finiteness;
reference endpoints array hash (recomputed) == plain-Fs hash; cross-worker duplicate
byte identity of endpoints/truth/f0/gpu_losses; read-log inside certified slice;
FP-04 anchors (Fs from reference + plain_fs, expert_k vs L1) by exact float equality.
Writes a JSON record; prints PASS/FAIL counts only (no loss values on stdout).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RUN = Path("/mnt/afs/260010168/EarthDelta/plans/plan_v4_0923/run_20260924T104725Z_fp05b")
PROTO = RUN / "protocol/cache_protocol_v1.json"
PROTO_SHA = "a42c11dc68d9ede158370a6ba77c4ef8b54cb5cecde30358ee2424fc18ae96d8"
job = Path(sys.argv[1])
out = Path(sys.argv[2])


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


assert fsha(PROTO) == PROTO_SHA
P = json.loads(PROTO.read_text())
freeze = json.loads(Path(P["split"]["freeze"]["path"]).read_text())
assert fsha(P["split"]["freeze"]["path"]) == P["split"]["freeze"]["sha256"]
bank = P["inputs"]["bank_bundle"]
fsd = P["inputs"]["fs_reference"]["digests"]
debug_ids = P["decided"]["debug_issues"]
cfgs = {k: v for k, v in P["configs"].items() if v["job"] == "C-J1"}
res = {"job_dir": str(job), "checks": {}, "per_worker": {}, "per_issue": {}}
fails = []

# freeze debug allow-list
roles = freeze.get("roles", {})
dbg_allow = sorted(roles.get("debug", {}).get("allowed_issue_ids", []))
res["checks"]["debug_set_equals_freeze_allow_list"] = dbg_allow == sorted(debug_ids)
own = sorted(i for c in cfgs.values() for i in c["own_issue_ids"])
res["checks"]["own_partition_equals_debug_set"] = own == sorted(debug_ids)

# anchors
panels = {}
for k, ref in P["inputs"]["debug"]["anchors"].items():
    assert fsha(ref["path"]) == ref["sha256"], f"anchor {k} sha"
    panels[int(k)] = json.loads(Path(ref["path"]).read_text())["experts"][str(k)]
group_of = {i: k for k, p in panels.items() for i in p["issue_ids"]}
grouping = json.loads(Path(P["inputs"]["debug"]["grouping"]["path"]).read_text())
assert fsha(P["inputs"]["debug"]["grouping"]["path"]) == P["inputs"]["debug"]["grouping"]["sha256"]

# workers
copies = {}
for cid, c in sorted(cfgs.items()):
    d = job / c["output_subdir"]
    s = json.loads((d / "shard.json").read_text())
    ok = (s.get("passed") is True and s.get("e0_passed") is True and s.get("config_id") == cid
          and s.get("protocol_sha256") == PROTO_SHA and s.get("digests_unchanged") is True
          and s.get("bank_digest_after") == bank["bank_digest"]
          and s["e0"].get("bank_digest") == bank["bank_digest"]
          and s.get("merged_fs_digest_before") == fsd["merged_backbone_digest"]
          and s.get("merged_fs_digest_after") == fsd["merged_backbone_digest"]
          and s["e0"]["sources"]["passed"] and s["e0"]["sources"]["n_files"] == 27
          and s["e0"]["tf32_off"] and s["e0"]["s0"] and s["e0"]["fs_identity_pass"]
          and s["e0"]["backend_pre"] and s["e0"]["backend_post"]
          and s.get("status_counts") == {"PASS": 60}
          and sorted(i["issue_id"] for i in s["issues"]) == sorted(c["issue_ids"]))
    res["per_worker"][cid] = {"ok": ok, "peak_gpu_gib": s["peak_gpu_gib"],
                              "model_load_seconds": s["model_load_seconds"],
                              "device": s["provenance"]["device"],
                              "device_name": s["provenance"]["device_name"]}
    if not ok:
        fails.append(f"worker {cid}")
    for it in s["issues"]:
        p = Path(it["dir"]) / "issue.json"
        if fsha(p) != it["issue_json_sha256"]:
            fails.append(f"issue.json hash {p}")
        copies.setdefault(it["issue_id"], {})[cid] = Path(it["dir"])

SHAPES = {"endpoints": (5, 3, 69, 128, 256), "truth": (3, 69, 128, 256), "f0_endpoints": (3, 69, 128, 256)}
max_rel = 0.0
n_anchor_values = 0
for iid in debug_ids:
    r = {}
    cp = copies.get(iid, {})
    r["n_copies"] = len(cp)
    if len(cp) != 2:
        fails.append(f"{iid} copies={len(cp)}")
    shas = {}
    for cid, d in cp.items():
        ij = json.loads((d / "issue.json").read_text())
        s = {}
        for name, shape in SHAPES.items():
            fp = d / f"{name}.npy"
            h = fsha(fp)
            s[name] = h
            if h != ij["files"][name]["sha256"]:
                fails.append(f"{iid}/{cid} {name} rehash")
            a = np.load(fp, mmap_mode="r", allow_pickle=False)
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                fails.append(f"{iid}/{cid} {name} shape/dtype/finite")
            if name == "endpoints":
                ref_sha = arr_sha(np.asarray(a[0]))
                if not (ref_sha == ij["reference_endpoints_array_sha256"] == ij["plain_fs_endpoints_array_sha256"]
                        and ij["reference_equals_plain_fs_bitwise"] is True):
                    fails.append(f"{iid}/{cid} reference != plain_fs")
        gl_bytes = (d / "gpu_losses.json").read_bytes()
        s["gpu_losses"] = hashlib.sha256(gl_bytes).hexdigest()
        if s["gpu_losses"] != ij["files"]["gpu_losses"]["sha256"]:
            fails.append(f"{iid}/{cid} gpu_losses rehash")
        # read-set (V8)
        row = ij["row"]
        for e in ij["read_log"]:
            if not (e["issue_id"] == iid and e["role"] == "debug" and e["store"] == "2020.zarr"
                    and row["history_index"] <= e["index"] <= row["target_index"]):
                fails.append(f"{iid}/{cid} read outside slice {e}")
        if Path(row["issue_store"]).name != "2020.zarr":
            fails.append(f"{iid}/{cid} store")
        if not ij["hooks_clean"]:
            fails.append(f"{iid}/{cid} hooks")
        shas[cid] = s
        gl = json.loads(gl_bytes)
        # anchors
        k = group_of[iid]
        fs_want, l1_want = panels[k]["Fs"][iid], panels[k]["L1"][iid]
        pairs = []
        for h in ("6", "24", "72"):
            pairs += [("reference", h, gl["candidates"]["reference"][h], fs_want[h]),
                      ("plain_fs", h, gl["plain_fs"][h], fs_want[h]),
                      (f"expert_{k}", h, gl["candidates"][f"expert_{k}"][h], l1_want[h])]
        exact = all(g == w for _, _, g, w in pairs)
        for _, _, g, w in pairs:
            max_rel = max(max_rel, abs(g - w) / abs(w))
            n_anchor_values += 1
        r.setdefault("anchors_exact_by_copy", {})[cid] = exact
        r["group"] = k
        if not exact:
            fails.append(f"{iid}/{cid} anchor not bitwise")
    vals = list(shas.values())
    r["duplicate_bytes_identical"] = len(vals) == 2 and vals[0] == vals[1]
    if not r["duplicate_bytes_identical"]:
        fails.append(f"{iid} duplicates differ")
    r["file_sha256"] = vals[0] if vals else None
    res["per_issue"][iid] = r

res["checks"]["n_anchor_values_compared"] = n_anchor_values
res["checks"]["anchor_max_relative_diff"] = max_rel
res["checks"]["group_membership_by_panel"] = {i: group_of[i] for i in debug_ids}
res["failures"] = fails
res["passed"] = not fails and all(v for k, v in res["checks"].items() if isinstance(v, bool))
with open(out, "x") as f:
    json.dump(res, f, indent=1, sort_keys=True)
print(json.dumps({"passed": res["passed"], "n_failures": len(fails), "failures": fails[:20],
                  "n_anchor_values_compared": n_anchor_values,
                  "anchor_max_relative_diff_is_zero": max_rel == 0.0,
                  "checks": {k: v for k, v in res["checks"].items() if isinstance(v, bool)}}))
