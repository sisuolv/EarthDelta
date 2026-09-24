#!/usr/bin/env python3
"""WeatherBench-X integration -- Tier B: real Stormer rollouts reconciled to FP-03/FP-04.

Tier A validated export/regrid/metrics on a static reference output. Tier B
closes the remaining gap: the REAL Stormer bridge with the CERTIFIED Fs
(``FS_SELECTED``) and the CERTIFIED K=4 bank (``BANK_CERTIFIED``), rolled out on
the already-exposed FP-03/FP-04 debug issues, exported through
``earthdelta.wbx.export`` and scored through ``earthdelta.wbx.evaluate`` (the
vendored official WeatherBench-X code). The per-issue objective reconstructed
from the official WB-X per-variable MSEs must equal the per-issue panel values
FP-03/FP-04 recorded, to 1e-6 relative, for every issue and every lead.

No new rollout logic: the rollouts are the pinned ``fs_rollout_trajectory`` /
``controlled_rollout`` calls (exactly what ``fs_panel_losses`` /
``bank_panel_losses`` do internally); the certified Fs is loaded by the pinned
trainer's own ``authorize_certified_fs`` + ``load_certified_fs``; the bank by
``bank_training.load_bank`` against the certified manifest.

Families (model -> rollout -> recorded anchor):
  F0          plain backbone                       FP-03 panel train.F0
  Fs-adapter  backbone + certified Fs adapter      FP-03 panel train.L1
  Fs          certified merged Fs backbone          FP-04 bank_panels expert{k}.Fs
  bank-e{k}   merged Fs + bank, singleton plan k    FP-04 bank_panels expert{k}.L1
(plus an informational cross-path check: merged Fs on the FP-03 train issues
vs FP-03 L1, which was produced with the adapter hooks, not the merged weights.)

Data scope: only the 2020 FP-03 train (8) and FP-04 bank-panel (44) issues --
already exposed. The FP-03 holdout (2019) is excluded because the default
year gate refuses 2019. Every time a selected issue touches (issue .. +72h)
must lie in 2020-H1; that is checked before anything is read.

Phases:
  select    CPU, no data read: selection, H1 window + year-gate checks.
  gpu       ACP worker: rollouts -> official-layout forecast zarrs + in-job
            pinned objective (the recorded panel's own reduction) per issue.
  evaluate  CPU: truth via ``export.truth_from_store`` (year-gated), official
            WB-X MSE per issue, objective reconstruction, reconciliation table.
Run every phase with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python set
externally (README), PYTHONPATH=.pydeps[:cuda_site].
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

# Assets always live in the real repo on AFS (a CCI job runs from a source
# snapshot, but data, checkpoints and certified bundles are never copied).
ASSETS = Path("/mnt/afs/260010168/EarthDelta")
PLANS = ASSETS / "plans/plan_v4_0923"
FP03 = PLANS / "run_20260924T013959Z_fp03_v2"
FP04 = PLANS / "run_20260924T033627Z_fp04_bank"
BJ2 = ASSETS / "artifacts/round2_cci/ed-r4bankbj2-0924052624-dde08d"

GATE_CONFIG = ASSETS / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/gate_config.json"
BANK_PROTOCOL = FP04 / "protocol/bank_protocol_v1.json"
BANK_PROTOCOL_SHA256 = "394562dde7cae5b3b805e6f5344f680ec9f24c71d9ed90c3893d96e8867da0d1"
FS_REFERENCE_MANIFEST = FP03 / "certify/fs/reference_manifest.json"
FS_REFERENCE_MANIFEST_SHA256 = "45c09e1a98f976a617bea351f1a6db7873fdb1ae73c1e8a96fbdd3f71706c972"
FS_CERTIFY_DECISION = FP03 / "certify/V2_certify_decision.json"
FS_ADAPTER = FP03 / "certify/fs/fs_adapter.pt"
FS_MERGED = FP03 / "certify/fs/fs_merged_backbone.pt"
BANK_DIR = FP04 / "certify/bank"
BANK_MANIFEST = BANK_DIR / "bank_manifest.json"
BANK_MANIFEST_SHA256 = "6946ec0bd7fd57b17d144c80f97e66ef46187bb60967aeffb35c0d1e5bf4275f"

#: admission records, pinned by the FP-03 v2 / FP-04 protocols' inputs.admission
ADMISSIONS = {
    "fp03_train": (PLANS / "run_20260923T192759Z_fp03_fs/admission/bank_fit_admission.json",
                   "b0b18bb54ab6a83138cb182cd298fc834e836e8e36640bdf612465862016c217"),
    "fp04_bank_fit": (FP04 / "admission/bank_fit_admission.json",
                      "01d344b8e2bb2cd66766fae290204322a6cf4939c353bc632ed5824ee7a7bdea"),
}
#: recorded per-issue panels (the reconciliation anchors)
FP03_PANEL = (FP03 / "certify/fs/panel_initial_final.json",
              "4288ce99ef072a61e0847ed4516ddf0a3e48d3749fa5b77e9098652d666985b6")
FP04_PANELS = {k: (BJ2 / f"expert{k}/bank_panels_expert{k}.json", sha) for k, sha in enumerate((
    "ff256077bf073e69a99c770e041928ebab69b32f896b3492481fac562f8f28de",
    "8b8372d7539028595d7bf2a208703fa0c3809e44e74e56ed977e52d40a386937",
    "1391f9c195e5665cd4698e63ecacff8d6bede64c19e0f4a6e25a9f1eb2f6026f",
    "691ec9f50a450919f4519a4d2fa7a499eedbb7711fd07e08cc5ae295f337f48c",
))}
TRUTH_STORE = ASSETS / "data/era5_1p40625/2020.zarr"

LEAD_STEPS = (1, 4, 12)
INTERVAL_H = 6
LEAD_HOURS = tuple(s * INTERVAL_H for s in LEAD_STEPS)
TARGET_BLOCKS = (18, 19, 20, 21, 22, 23)
NUM_EXPERTS, RANK, A0, RHO, HOLD = 4, 4, 0.25, 0.25, 4
H1_START = np.datetime64("2020-01-01T00:00:00", "s")
H1_END = np.datetime64("2020-07-01T00:00:00", "s")   # exclusive: 2020-H2 is never touched
RTOL = 1e-6
MODEL_PREFIX = "Stormer-ps4-6h-path"
SCHEMA = "ed-wbx-tierb/1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a: float, b: float) -> float:
    d = max(abs(a), abs(b))
    return 0.0 if d == 0 else abs(a - b) / d


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def pinned_sources_status(root: Path = SOURCE_ROOT) -> Dict[str, Any]:
    pins = json.loads(BANK_PROTOCOL.read_text())["source_at_preregistration"]
    bad = [p for p, h in pins.items() if sha256_file(root / p) != h]
    return {"root": str(root), "n": len(pins), "n_match": len(pins) - len(bad), "mismatched": bad}


# =============================================================================
# Selection (no data read)
# =============================================================================

def _checked_json(path: Path, sha: str) -> Dict[str, Any]:
    actual = sha256_file(path)
    if actual != sha:
        raise RuntimeError(f"{path} hashes to {actual}, expected pinned {sha}")
    return json.loads(Path(path).read_text())


def build_selection() -> Dict[str, Any]:
    """Issue sets, anchors and admission rows; fail-closed on window / gate."""
    from earthdelta.wbx.evaluate import assert_years_authorized

    fp03 = _checked_json(*FP03_PANEL)
    if fp03.get("schema") != "ed-fs-panels/1" or list(fp03.get("lead_steps", [])) != list(LEAD_STEPS):
        raise RuntimeError("FP-03 panel schema / lead steps unexpected")
    fp04 = {}
    bank_manifest = _checked_json(BANK_MANIFEST, BANK_MANIFEST_SHA256)
    manifest_experts = {int(e["expert_index"]): e for e in bank_manifest["binding"]["experts"]}
    for k, (path, sha) in FP04_PANELS.items():
        doc = _checked_json(path, sha)
        if doc.get("schema") != "ed-bank-panels/1" or list(doc.get("lead_steps", [])) != list(LEAD_STEPS):
            raise RuntimeError(f"FP-04 panel {k} schema / lead steps unexpected")
        # bind the panel to the CERTIFIED expert: its sibling expert file is
        # the one the certified bank manifest names (sha256 and path).
        src = manifest_experts[k]["source"]["expert_file"]
        sibling = path.parent / f"expert_{k}.pt"
        if Path(src["path"]).resolve() != sibling.resolve() or sha256_file(sibling) != src["sha256"]:
            raise RuntimeError(f"FP-04 panel {k} is not bound to the certified expert {k}")
        fp04[k] = doc["experts"][str(k)]

    rows_by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, (path, sha) in ADMISSIONS.items():
        record = _checked_json(path, sha)
        rows_by_source[name] = {r["issue_id"]: r for r in record["admission"]["admitted"]}

    fp03_ids = list(fp03["train"]["issue_ids"])
    fp04_ids = {k: list(fp04[k]["issue_ids"]) for k in range(NUM_EXPERTS)}
    fp04_all = [i for k in range(NUM_EXPERTS) for i in fp04_ids[k]]
    if len(set(fp04_all)) != len(fp04_all):
        raise RuntimeError("FP-04 expert groups overlap")

    def row_for(iid: str, source: str) -> Dict[str, Any]:
        row = rows_by_source[source].get(iid)
        if row is None:
            raise RuntimeError(f"{iid} not admitted in {source}")
        return row

    families: Dict[str, Dict[str, Any]] = {
        "F0": {"issues": fp03_ids, "source": "fp03_train", "anchor": "FP-03 train.F0"},
        "Fs-adapter": {"issues": fp03_ids, "source": "fp03_train", "anchor": "FP-03 train.L1"},
        "Fs": {"issues": fp04_all + [i for i in fp03_ids if i not in fp04_all],
               "source": None, "anchor": "FP-04 expert{k}.Fs (+ info: FP-03 train.L1)"},
    }
    for k in range(NUM_EXPERTS):
        families[f"bank-e{k}"] = {"issues": fp04_ids[k], "source": "fp04_bank_fit",
                                  "anchor": f"FP-04 expert{k}.L1", "expert": k}

    issue_rows: Dict[str, Dict[str, Any]] = {}
    for iid in fp03_ids:
        issue_rows[iid] = {"source": "fp03_train", "row": row_for(iid, "fp03_train")}
    for iid in fp04_all:
        if iid in issue_rows:   # the shared probe issue: identical row content required
            other = row_for(iid, "fp04_bank_fit")
            keys = ("issue_time", "issue_store", "issue_index", "valid_time")
            if any(other[k] != issue_rows[iid]["row"][k] for k in keys):
                raise RuntimeError(f"{iid}: FP-03 and FP-04 admission rows disagree")
            continue
        issue_rows[iid] = {"source": "fp04_bank_fit", "row": row_for(iid, "fp04_bank_fit")}

    # --- H1 window and year gate, BEFORE any data is read -------------------
    window: Dict[str, Any] = {}
    years = set()
    for iid, entry in issue_rows.items():
        row = entry["row"]
        issue = np.datetime64(int(row["issue_time"]), "s")
        first = np.datetime64(int(row.get("history_time", row["issue_time"])), "s")
        last = issue + np.timedelta64(max(LEAD_HOURS), "h")
        if not (H1_START <= first and last < H1_END):
            raise RuntimeError(f"{iid} touches {first}..{last}, outside 2020-H1")
        if Path(str(row["issue_store"])).resolve() != TRUTH_STORE.resolve():
            raise RuntimeError(f"{iid} reads {row['issue_store']}, not {TRUTH_STORE}")
        years.update({int(str(first)[:4]), int(str(last)[:4])})
        window[iid] = [str(first), str(last)]
    gate = assert_years_authorized(sorted(years))
    for fam in families.values():
        times = [issue_rows[i]["row"]["issue_time"] for i in fam["issues"]]
        if len(set(times)) != len(times):
            raise RuntimeError("duplicate issue times inside a family")
    return {
        "schema": SCHEMA,
        "families": families,
        "issue_rows": issue_rows,
        "window": window,
        "window_min": min(v[0] for v in window.values()),
        "window_max": max(v[1] for v in window.values()),
        "year_gate": gate,
        "excluded": {
            "fp03_holdout_2019": {"n": len(fp03["holdout"]["issue_ids"]),
                                  "reason": "year 2019 is refused by the default year gate"},
            "fp04_admitted_not_in_any_panel": sorted(set(rows_by_source["fp04_bank_fit"]) - set(fp04_all)),
        },
        "anchors": {
            "fp03_panel": {"path": str(FP03_PANEL[0]), "sha256": FP03_PANEL[1]},
            "fp04_panels": {str(k): {"path": str(p), "sha256": s} for k, (p, s) in FP04_PANELS.items()},
        },
        "admissions": {k: {"path": str(p), "sha256": s} for k, (p, s) in ADMISSIONS.items()},
        "n_unique_issues": len(issue_rows),
    }


def recorded_anchor(fam: str, iid: str, selection, fp03, fp04) -> Tuple[Optional[Dict[str, float]], str]:
    """Recorded {lead_h: value} for (family, issue), and its label."""
    if fam == "F0":
        return fp03["train"]["F0"][iid], "FP-03 train.F0"
    if fam == "Fs-adapter":
        return fp03["train"]["L1"][iid], "FP-03 train.L1"
    if fam == "Fs":
        for k in range(NUM_EXPERTS):
            if iid in fp04[k]["Fs"]:
                return fp04[k]["Fs"][iid], f"FP-04 expert{k}.Fs"
        return None, "none"
    if fam.startswith("bank-e"):
        k = int(fam[len("bank-e"):])
        return fp04[k]["L1"][iid], f"FP-04 expert{k}.L1"
    raise KeyError(fam)


def load_anchors():
    fp03 = _checked_json(*FP03_PANEL)
    fp04 = {k: _checked_json(p, s)["experts"][str(k)] for k, (p, s) in FP04_PANELS.items()}
    return fp03, fp04


# =============================================================================
# GPU phase
# =============================================================================

def _import_trainer():
    """The pinned FP-03/FP-04 trainer module (for its certified loading entry points)."""
    path = SOURCE_ROOT / "scripts/r2_fs_bank_train.py"
    spec = importlib.util.spec_from_file_location("r2_fs_bank_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["r2_fs_bank_train"] = module
    spec.loader.exec_module(module)    # sets TF32 off before any model is built
    return module


def _filtered_admission(path: Path, sha: str, issue_ids: Sequence[str]) -> Dict[str, Any]:
    """The pinned admission record restricted to ``issue_ids`` (rows + their certificates).

    Consumer certification (`certify_admission_for_fs`) is per row, so the
    restriction keeps every check while never reading a non-selected row's data
    (the FP-04 record also admits one issue whose 72h target is in 2020-H2).
    """
    record = _checked_json(path, sha)
    wanted = set(issue_ids)
    out = dict(record)
    admission = dict(record["admission"])
    admission["admitted"] = [r for r in record["admission"]["admitted"] if r["issue_id"] in wanted]
    out["admission"] = admission
    out["content_certificates"] = [c for c in record.get("content_certificates") or []
                                   if c.get("issue_id") in wanted]
    if {r["issue_id"] for r in admission["admitted"]} != wanted:
        raise RuntimeError(f"{path}: selected issues are not all admitted")
    return out


def _array_sha(t) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def phase_gpu(args) -> int:
    import torch
    trainer = _import_trainer()
    from earthdelta import bank_training as bt
    from earthdelta import fs_protocol as fp
    from earthdelta import static_adapter as sa
    from earthdelta.bridge.stormer_bridge import controlled_rollout
    from earthdelta.wbx import export
    from earthdelta.wbx.regrid import stormer_native_grid

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"schema": SCHEMA, "phase": "gpu", "argv": sys.argv,
                              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "cpu_plumbing_smoke": bool(args.cpu_smoke)}
    t0 = time.time()
    record["pinned_sources_before"] = pinned_sources_status()
    selection = build_selection()
    record["selection"] = {k: v for k, v in selection.items() if k != "issue_rows"}
    families = selection["families"]
    if args.families:
        families = {k: v for k, v in families.items() if k in set(args.families)}
    if args.limit:
        families = {k: {**v, "issues": v["issues"][:args.limit]} for k, v in families.items()}
    needed = sorted({i for fam in families.values() for i in fam["issues"]})

    device = torch.device(args.device)
    record["torch"] = {"version": torch.__version__,
                       "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                       "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                       "device": str(device),
                       "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    protocol = fp.load_protocol(BANK_PROTOCOL, BANK_PROTOCOL_SHA256)
    q0 = protocol["qualification_rule"]["e0"]
    if not args.cpu_smoke:
        record["backend_precondition"] = fp.official_backend_precondition(
            expected_xformers=q0["xformers_version"], expected_torch=q0["torch_version"])
        if record["torch"]["cuda_matmul_allow_tf32"] or record["torch"]["cudnn_allow_tf32"]:
            raise RuntimeError("TF32 is on; the recorded panels were produced with TF32 off")

    # --- F0 bridge, certified Fs (pinned trainer entry points) ----------------
    bridge, variables, lat, gate_config = trainer.build_real_bridge(GATE_CONFIG, device)
    if not args.cpu_smoke:
        record["backend_postcondition"] = fp.official_backend_postcondition(bridge.model)
    ns = SimpleNamespace(
        stage="bank_verify", config=GATE_CONFIG,
        fs_reference_manifest=FS_REFERENCE_MANIFEST,
        fs_reference_manifest_sha256=FS_REFERENCE_MANIFEST_SHA256,
        fs_certify_decision=FS_CERTIFY_DECISION, fs_adapter=FS_ADAPTER,
        fs_adapter_sha256=None, fs_merged_backbone=FS_MERGED, rank_per_expert=RANK)
    fs_reference = trainer.authorize_certified_fs(ns, protocol)
    adapters, adapter_sha = sa.load_fs_adapter(
        FS_ADAPTER, int(bridge.model.blocks[TARGET_BLOCKS[0]].attn.proj.in_features), TARGET_BLOCKS,
        rank_per_expert=RANK, expected_sha256=fs_reference["fs_adapter_sha256"], device=device)
    # The merged Fs backbone (a second full model copy) and the bank are loaded
    # only when a selected family needs them (always, in the real job; the
    # 8 GB CCI container can only hold one backbone for a CPU plumbing smoke).
    need_merged = any(f == "Fs" or f.startswith("bank-e") for f in families)
    fs_bridge = bank = registry = None
    if need_merged:
        ctx: Dict[str, Any] = {"bridge": bridge, "target_blocks": TARGET_BLOCKS, "protocol": protocol,
                               "record": {"binding": {"fs_reference": fs_reference}}}
        record["fs_identity"] = trainer.load_certified_fs(ns, ctx)      # raises unless identity_pass
        fs_bridge = ctx["fs_bridge"]

        # --- certified bank -----------------------------------------------------
        bank_manifest = _checked_json(BANK_MANIFEST, BANK_MANIFEST_SHA256)["binding"]
        bank, bank_meta = bt.load_bank(BANK_DIR / bank_manifest["bank"]["file"],
                                       expected_sha256=bank_manifest["bank"]["sha256"], device=device)
        registry = bt.build_bank_registry(NUM_EXPERTS, rho=RHO, a0=A0)
        record["bank_identity"] = {
            "sha256": bank_meta["sha256"],
            "bank_digest_matches": bank_meta["bank_digest"] == bank_manifest["bank"]["bank_digest"],
            "expert_digests_match": bank_meta["expert_digests"] == bank_manifest["bank"]["expert_digests"],
            "binding_matches": (bank_manifest["num_experts"], bank_manifest["rank_per_expert"],
                                bank_manifest["a0"], bank_manifest["rho"], bank_manifest["hold_steps"],
                                list(bank_manifest["target_blocks"]))
                               == (NUM_EXPERTS, RANK, A0, RHO, HOLD, list(TARGET_BLOCKS)),
        }
        if not all(v is True for k, v in record["bank_identity"].items() if k != "sha256"):
            raise RuntimeError(f"bank identity failed: {record['bank_identity']}")

    # --- samples: pinned admission records restricted to the selection -------
    samples: Dict[str, Any] = {}
    certification: Dict[str, Any] = {}
    for source, (path, sha) in ADMISSIONS.items():
        ids = [i for i in needed if selection["issue_rows"][i]["source"] == source]
        if not ids:
            continue
        cert: Dict[str, Any] = {}
        loaded = sa.load_admitted_samples(
            _filtered_admission(path, sha, ids), bridge, lead_steps=list(LEAD_STEPS),
            require_certified=True, device=device, certification_out=cert)
        certification[source] = {"passed": cert.get("passed"), "n_rows": cert.get("n_rows"),
                                 "content_reverified": cert.get("content_reverified")}
        for s in loaded:
            samples[s.issue_id] = s
    record["admission_consumer_certification"] = certification
    spec = sa.build_objective_spec(bridge, lat, lead_steps=(4,), space="raw")
    per_lead = {s: dataclasses.replace(spec, lead_steps=(s,)) for s in LEAD_STEPS}
    record["objective"] = spec.to_dict()

    lat_grid, lon_grid = stormer_native_grid()
    if lat is not None and not np.array_equal(np.asarray(lat, dtype=np.float64), lat_grid):
        raise RuntimeError("objective latitude differs from the export grid")
    names = list(variables)
    targets: Dict[str, Dict[str, str]] = {}
    for iid, s in samples.items():
        row = selection["issue_rows"][iid]["row"]
        if str(np.datetime64(s.time_utc, "s")) != str(np.datetime64(int(row["issue_time"]), "s")):
            raise RuntimeError(f"{iid}: sample time {s.time_utc} != admission issue_time")
        targets[iid] = {str(h): _array_sha(s.targets_raw[st]) for st, h in zip(LEAD_STEPS, LEAD_HOURS)}
    record["target_sha256"] = targets

    def rollout(fam: str, sample):
        x = sample.x_norm
        with torch.no_grad():
            if fam == "F0":
                return bridge, sa.fs_rollout_trajectory(
                    bridge, x, names, steps=max(LEAD_STEPS), fs_adapters=None,
                    target_blocks=TARGET_BLOCKS, interval_hours=INTERVAL_H)
            if fam == "Fs-adapter":
                return bridge, sa.fs_rollout_trajectory(
                    bridge, x, names, steps=max(LEAD_STEPS), fs_adapters=adapters,
                    target_blocks=TARGET_BLOCKS, interval_hours=INTERVAL_H)
            if fam == "Fs":
                return fs_bridge, sa.fs_rollout_trajectory(
                    fs_bridge, x, names, steps=max(LEAD_STEPS), fs_adapters=None,
                    target_blocks=TARGET_BLOCKS, interval_hours=INTERVAL_H)
            k = int(fam[len("bank-e"):])
            plan = bt.expert_plan(registry, k, hold_steps=HOLD)
            return fs_bridge, controlled_rollout(
                fs_bridge, x, names, interval=INTERVAL_H, steps=max(LEAD_STEPS), plan=plan,
                expert_loras=dict(bank), target_blocks=TARGET_BLOCKS, return_trajectory=True)

    fams_out: Dict[str, Any] = {}
    for fam, info in families.items():
        records, rows = [], {}
        tf = time.time()
        for iid in info["issues"]:
            sample = samples[iid]
            used_bridge, traj = rollout(fam, sample)
            # (1) the recorded panel's own reduction on this very trajectory
            objective = {str(h): float(sa.objective_loss_for_sample(used_bridge, traj, sample, per_lead[st]))
                         for st, h in zip(LEAD_STEPS, LEAD_HOURS)}
            # (2) the wbx export of the same trajectory
            rec = export.record_from_trajectory(
                traj, used_bridge.normalization, issue_time=np.datetime64(int(sample.issue_time), "s"),
                lead_steps=LEAD_STEPS, model=f"{MODEL_PREFIX}-{fam}", issue_id=iid,
                provenance={"family": fam, "issue_id": iid, "device": str(device)})
            # (3) exported raw == the objective's own (on-device) denormalization, bitwise
            with torch.no_grad():
                dev_raw = np.stack([used_bridge.normalization.denormalize(traj[:, st])[0].cpu().numpy()
                                    for st in LEAD_STEPS])
            rows[iid] = {"objective_in_job": objective,
                         "export_equals_device_denorm_bitwise": bool(np.array_equal(dev_raw, rec.fields)),
                         "export_dtype": str(rec.fields.dtype)}
            records.append(rec)
            del traj
        ds = export.build_forecast_dataset(records, lat=lat_grid, lon=lon_grid,
                                           extra_attrs={"tier": "B", "family": fam})
        export.verify_issue_times(ds, [np.datetime64(int(samples[i].issue_time), "s") for i in info["issues"]])
        zpath = out_dir / "forecasts" / f"{fam}.zarr"
        manifest = export.write_forecast_zarr(ds, zpath, manifest_extra={"family": fam, "tier": "B"})
        fams_out[fam] = {"issues": info["issues"], "zarr": str(zpath), "manifest": manifest,
                         "rows": rows, "elapsed_s": time.time() - tf}
        print(f"[{fam}] {len(info['issues'])} issues in {time.time() - tf:.1f}s -> {zpath}", flush=True)
        del records, ds
    record["families"] = fams_out
    record["hooks_left_on_backbones"] = {
        "f0": sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in bridge.model.modules()),
    }
    if fs_bridge is not None:
        record["hooks_left_on_backbones"]["fs"] = sum(
            len(m._forward_hooks) + len(m._forward_pre_hooks) for m in fs_bridge.model.modules())
        record["fs_backbone_digest_after"] = sa.state_dict_digest(fs_bridge.model)
        record["fs_backbone_unchanged"] = (record["fs_backbone_digest_after"]
                                           == record["fs_identity"]["digests"]["merged_backbone_digest"])
    record["pinned_sources_after"] = pinned_sources_status()
    record["elapsed_s"] = time.time() - t0
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out_dir / "gpu_record.json", record)
    print(json.dumps({"gpu_record": str(out_dir / "gpu_record.json"), "elapsed_s": record["elapsed_s"]}))
    return 0


# =============================================================================
# Evaluate phase (CPU, official WB-X)
# =============================================================================

def phase_evaluate(args) -> int:
    import torch
    import xarray as xr
    from earthdelta.bridge.stormer_bridge import NormalizationContract
    from earthdelta.contracts import GateIdentityConfig
    from earthdelta.data.pull_wb2 import CANONICAL_VARIABLES
    from earthdelta.wbx import _vendor, export, evaluate as ev

    gpu_dir = Path(args.gpu_dir)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=False)
    gpu = json.loads((gpu_dir / "gpu_record.json").read_text())
    out: Dict[str, Any] = {"schema": SCHEMA, "phase": "evaluate", "tier": "B",
                           "gpu_dir": str(gpu_dir), "job_id": args.job_id,
                           "gpu_record_sha256": sha256_file(gpu_dir / "gpu_record.json"),
                           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    t0 = time.time()
    status = 0
    try:
        out["pinned_sources_before"] = pinned_sources_status()
        out["official_code"] = _vendor.provenance()
        selection = build_selection()
        fp03, fp04 = load_anchors()
        if json.loads(json.dumps(selection["families"])) != gpu["selection"]["families"]:
            raise RuntimeError("GPU phase ran on a different selection")
        gate_cfg = GateIdentityConfig.load_json(GATE_CONFIG)
        norm = NormalizationContract.from_npz_dir(
            gate_cfg.normalization_dir, list(gate_cfg.variables),
            intervals=tuple(gate_cfg.normalization_intervals), policy=gate_cfg.normalization_policy)
        if list(gate_cfg.variables) != list(CANONICAL_VARIABLES):
            raise RuntimeError("gate config channel order is not canonical")
        s2 = norm.inp_std.detach().cpu().to(torch.float64).numpy() ** 2     # the objective's scale^2

        # ---- truth through the year-gated wbx path ---------------------------
        fams = gpu["families"]
        all_issues = sorted({i for f in fams.values() for i in f["issues"]})
        issue_time = {i: np.datetime64(int(selection["issue_rows"][i]["row"]["issue_time"]), "s")
                      .astype("datetime64[ns]") for i in all_issues}
        valid = sorted({issue_time[i] + np.timedelta64(h, "h") for i in all_issues for h in LEAD_HOURS})
        if not (min(valid) >= H1_START and max(valid) < H1_END):
            raise RuntimeError("a valid time is outside 2020-H1")
        # Truth is written in issue chunks (one store per ~12 issues) only to keep
        # the peak memory of this 8 GB CCI container low; each issue is scored
        # against the store holding all of its valid times.
        ordered = sorted(all_issues, key=lambda i: issue_time[i])
        chunks = [ordered[n:n + 12] for n in range(0, len(ordered), 12)]
        truth_of: Dict[str, str] = {}
        truth_bitwise: Dict[str, bool] = {}
        truth_info: List[Dict[str, Any]] = []
        for c, issues in enumerate(chunks):
            times = sorted({issue_time[i] + np.timedelta64(h, "h") for i in issues for h in LEAD_HOURS})
            ds = export.truth_from_store(TRUTH_STORE, times)                 # year-gated
            gate_record = json.loads(ds.attrs["year_gate"])
            path = run_dir / f"truth_era5_2020H1_chunk{c}.zarr"
            manifest = export.write_forecast_zarr(ds, path, manifest_extra={"kind": "truth", "chunk": c})
            del ds
            lazy = xr.open_zarr(path)
            tindex = {np.datetime64(t, "ns"): n for n, t in enumerate(lazy["time"].values)}
            for i in issues:
                truth_of[i] = str(path)
                for h in LEAD_HOURS:
                    one = lazy.isel(time=[tindex[issue_time[i] + np.timedelta64(h, "h")]]).load()
                    arr = np.ascontiguousarray(export.flatten_channels(one, leading_dims=("time",))[0])
                    truth_bitwise[f"{i}@{h}h"] = (hashlib.sha256(arr.tobytes()).hexdigest()
                                                 == gpu["target_sha256"][i][str(h)])
            truth_info.append({"path": str(path), "n_times": len(times), "issues": issues,
                               "content_sha256": manifest["content_sha256"], "year_gate": gate_record})
        out["truth"] = {"stores": truth_info, "n_valid_times": len(valid),
                        "valid_min": str(min(valid)), "valid_max": str(max(valid)),
                        "equals_in_job_targets_bitwise": all(truth_bitwise.values()),
                        "n_bitwise_mismatch": sum(not v for v in truth_bitwise.values())}

        # ---- official WB-X MSE per issue -> objective reconstruction -----------
        det = _vendor.import_official("weatherbenchX.metrics.deterministic")
        metrics = {"mse": det.MSE()}
        leads = [np.timedelta64(h, "h") for h in LEAD_HOURS]
        table: List[Dict[str, Any]] = []
        worst: Dict[str, Dict[str, Any]] = {}

        def bump(key: str, value: float, where: str):
            if key not in worst or value > worst[key]["max_rel"]:
                worst[key] = {"max_rel": value, "at": where}

        for fam, info in fams.items():
            zpath = info["zarr"]
            for iid in info["issues"]:
                vals = {}
                for tag, up in (("wbx32", False), ("wbx64", True)):
                    res, _ = ev.evaluate(zpath, truth_of[iid], init_times=[issue_time[iid]],
                                         lead_times=leads, metrics=metrics,
                                         aggregator=ev.make_aggregator(), upcast_float64=up)
                    mse = ev.per_channel_values(res, "mse")                 # channel -> [lead]
                    m = np.array([[float(mse[c][j]) for c in CANONICAL_VARIABLES]
                                  for j in range(len(LEAD_HOURS))])
                    vals[tag] = (m / s2[None, :]).mean(axis=1)
                anchor, label = recorded_anchor(fam, iid, selection, fp03, fp04)
                injob = info["rows"][iid]["objective_in_job"]
                for j, h in enumerate(LEAD_HOURS):
                    w32, w64 = float(vals["wbx32"][j]), float(vals["wbx64"][j])
                    job = float(injob[str(h)])
                    row = {"family": fam, "issue_id": iid, "lead_h": h,
                           "wbx_objective_f32": w32, "wbx_objective_f64": w64,
                           "in_job_pinned_objective": job,
                           "export_bitwise": info["rows"][iid]["export_equals_device_denorm_bitwise"],
                           "anchor": label}
                    where = f"{fam}/{iid}@{h}h"
                    bump("wbx32_vs_in_job", rel(w32, job), where)
                    bump("wbx64_vs_in_job", rel(w64, job), where)
                    if anchor is not None:
                        rec = float(anchor[str(h)])
                        row.update({"recorded": rec, "rel_wbx32_vs_recorded": rel(w32, rec),
                                    "rel_wbx64_vs_recorded": rel(w64, rec),
                                    "rel_in_job_vs_recorded": rel(job, rec),
                                    "in_job_equals_recorded_bitwise": job == rec})
                        bump("ANCHOR_wbx32_vs_recorded", row["rel_wbx32_vs_recorded"], where)
                        bump("ANCHOR_wbx64_vs_recorded", row["rel_wbx64_vs_recorded"], where)
                        bump("in_job_vs_recorded", row["rel_in_job_vs_recorded"], where)
                    if fam == "Fs" and iid in fp03["train"]["L1"]:
                        info_rec = float(fp03["train"]["L1"][iid][str(h)])
                        row["info_rel_vs_fp03_L1_hooked_path"] = rel(w32, info_rec)
                        bump("INFO_merged_Fs_vs_fp03_L1_hooked", rel(w32, info_rec), where)
                    table.append(row)
        # Anchors exist by design for every row EXCEPT the Fs rows of the
        # FP-03-only train issues (added as an informational merged-vs-hooked
        # cross-path check against FP-03 L1; FP-04 recorded no Fs for them).
        fp04_fs_ids = {i for k in range(NUM_EXPERTS) for i in fp04[k]["Fs"]}
        out["expected_anchored_rows"] = {
            fam: len([i for i in info["issues"] if fam != "Fs" or i in fp04_fs_ids]) * len(LEAD_HOURS)
            for fam, info in fams.items()}
        info_only = {r["issue_id"] for r in table if "recorded" not in r}
        fp03_only = set(fp03["train"]["issue_ids"]) - fp04_fs_ids
        out["informational_only_issues"] = sorted(info_only)
        out["informational_only_is_fp03_only_fs"] = bool(
            ("Fs" not in fams or info_only == (fp03_only & set(fams["Fs"]["issues"])))
            and all(r["family"] == "Fs" and "info_rel_vs_fp03_L1_hooked_path" in r
                    for r in table if "recorded" not in r))
        anchored = [r for r in table if "recorded" in r]
        out["n_rows"] = len(table)
        out["n_anchored_rows"] = len(anchored)
        out["n_anchored_issues"] = len({(r["family"], r["issue_id"]) for r in anchored})
        out["worst_case"] = worst
        out["tolerance_rel"] = RTOL
        out["table"] = table
        out["per_family"] = {
            fam: {"n_issues": len(info["issues"]),
                  "max_rel_wbx32_vs_recorded": max((r.get("rel_wbx32_vs_recorded", 0.0)
                                                    for r in table if r["family"] == fam), default=None),
                  "max_rel_in_job_vs_recorded": max((r.get("rel_in_job_vs_recorded", 0.0)
                                                     for r in table if r["family"] == fam), default=None),
                  "n_in_job_bitwise_equal_recorded": sum(bool(r.get("in_job_equals_recorded_bitwise"))
                                                         for r in table if r["family"] == fam),
                  "n_rows_anchored": sum("recorded" in r for r in table if r["family"] == fam)}
            for fam, info in fams.items()}
        _vendor.assert_no_beam()
        out["no_beam_modules_imported"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        status = 1
    out["pinned_sources_after"] = pinned_sources_status()
    w = out.get("worst_case", {})
    checks = {
        "gpu_identity": bool(gpu.get("fs_identity", {}).get("identity_pass")
                             and all(v for k, v in gpu.get("bank_identity", {}).items() if k != "sha256")),
        "gpu_admission_certified": bool(gpu.get("admission_consumer_certification")) and all(
            v.get("passed") is True for v in gpu["admission_consumer_certification"].values()),
        "gpu_not_cpu_smoke": gpu.get("cpu_plumbing_smoke") is False,
        "gpu_pins_unchanged": bool(gpu.get("pinned_sources_after", {}).get("n"))
                              and gpu["pinned_sources_after"]["n_match"] == gpu["pinned_sources_after"]["n"],
        "gpu_fs_backbone_unchanged": bool(gpu.get("fs_backbone_unchanged")),
        "export_equals_device_denorm_bitwise": all(r["export_equals_device_denorm_bitwise"]
                                                   for f in gpu.get("families", {}).values()
                                                   for r in f["rows"].values()),
        "truth_equals_in_job_targets_bitwise": bool(out.get("truth", {}).get("equals_in_job_targets_bitwise")),
        "anchor_wbx32_vs_recorded_le_tol": w.get("ANCHOR_wbx32_vs_recorded", {}).get("max_rel", math.inf) <= RTOL,
        "anchor_wbx64_vs_recorded_le_tol": w.get("ANCHOR_wbx64_vs_recorded", {}).get("max_rel", math.inf) <= RTOL,
        "all_expected_rows_anchored": bool(out.get("per_family")) and all(
            v["n_rows_anchored"] == out["expected_anchored_rows"][fam]
            for fam, v in out["per_family"].items()),
        "unanchored_rows_are_only_the_informational_fs_check": bool(
            out.get("informational_only_is_fp03_only_fs")),
        "pins_unchanged": out["pinned_sources_after"]["n_match"] == out["pinned_sources_after"]["n"],
    }
    out["tier_b_checks"] = checks
    out["verdict"] = "TIER_B_PASS" if status == 0 and all(checks.values()) else "TIER_B_FAIL"
    out["elapsed_s"] = time.time() - t0
    write_json(run_dir / "tierb_summary.json", out)
    print(json.dumps({"verdict": out["verdict"], "checks": checks, "worst_case": w,
                      "summary": str(run_dir / "tierb_summary.json"), "error": out.get("error")}, indent=2))
    return 0 if out["verdict"] == "TIER_B_PASS" else 1


def phase_select(args) -> int:
    sel = build_selection()
    view = {k: v for k, v in sel.items() if k != "issue_rows"}
    view["family_sizes"] = {k: len(v["issues"]) for k, v in sel["families"].items()}
    view["n_rollouts"] = sum(view["family_sizes"].values())
    if args.out:
        write_json(Path(args.out), view)
    print(json.dumps({k: view[k] for k in ("family_sizes", "n_rollouts", "n_unique_issues", "window_min",
                                           "window_max", "year_gate", "excluded")}, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("select", "gpu", "evaluate"), required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-dir", type=str, default=None, help="evaluate: the GPU phase's --out")
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--families", nargs="*", default=None, help="gpu: restrict families (smoke)")
    ap.add_argument("--limit", type=int, default=None, help="gpu: issues per family (smoke)")
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="gpu phase on CPU for plumbing only: skips backend checks, never evidence")
    args = ap.parse_args()
    if args.phase == "select":
        return phase_select(args)
    if args.phase == "gpu":
        if args.out is None:
            ap.error("--out is required")
        if args.device.startswith("cpu") != bool(args.cpu_smoke):
            ap.error("--device cpu requires --cpu-smoke (and vice versa)")
        return phase_gpu(args)
    if args.out is None or args.gpu_dir is None:
        ap.error("--out and --gpu-dir are required")
    return phase_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
