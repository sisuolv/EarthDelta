#!/usr/bin/env python3
"""WeatherBench-X / WB2 integration smoke test -- Tier A (CPU only, no GPU job).

Uses the official F0 reference outputs already on disk from the S0
certification (``official_output_6h_{1,4,12}step.pt`` + ``input_norm.pt`` +
``manifest.json``, produced by the official Stormer ``forward_validation`` for
the 2020-01-01T00Z issue) and checks, end to end:

A1  identity: raw-input hash, normalization identity digest, channel order.
A2  denormalization: ``NormalizationContract.denormalize(input_norm)``
    reproduces the real raw input (and normalize(raw) reproduces input_norm);
    the official outputs are in NORMALIZED space and only become physical after
    denormalization (checked both ways).
A3  export -> zarr -> re-read is bitwise; decoded ``time`` equals the issue
    time; WB2 ``schema.apply_time_conventions`` and WB-X loader renaming give
    ``valid_time = issue + lead``; truth read through ``truth_from_store`` is
    bitwise the pinned S0 input array at the same valid times.
A4  metric reconciliation, per channel x lead: WeatherBench-X RMSE / MSE / Bias
    vs a from-scratch numpy float64 implementation vs EarthDelta's own
    ``metrics_contract.full_objective_loss`` (per channel, unscaled), and the
    real scale-standardized EarthDelta objective vs its reconstruction from
    the WB-X per-variable MSEs. Tolerance 1e-6 relative. Also the weight
    identity: WB-X ``GridAreaWeighting`` vs EarthDelta ``area_weight_q``.
A5  regrid cross-checks (official WB2 conservative vs pull_wb2 numpy vs
    from-scratch block oracle), float32-as-shipped and float64 tiers.
A6  (optional, --network) tiny reads of official public 64x32 stores through
    the proxy: ERA5 truth vs our regridded truth; one HRES forecast slice
    scored through the same WB-X path. Informational; never a pass criterion.

Nothing here runs a model. Outputs go to ``artifacts/wbx_smoke/<run_id>/``.
Run with:  PYTHONPATH=.pydeps python scripts/r4_wbx_smoke.py [--network]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
# Same workaround the test suite uses (README): the container's protobuf C++
# backend crashes on some generated descriptors pulled in transitively.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np  # noqa: E402

REFERENCE_DIR = SOURCE_ROOT / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/reference/upstream_reference_ps4_7fde884e_zd"
S0_INPUT = SOURCE_ROOT / "scripts/s0_gate_inputs/jan2020_full.npy"
TRUTH_STORE = SOURCE_ROOT / "data/era5_1p40625/2020_jan.zarr"
PIN_PROTOCOL = SOURCE_ROOT / "plans/plan_v4_0923/run_20260924T033627Z_fp04_bank/protocol/bank_protocol_v1.json"
ISSUE_TIME = np.datetime64("2020-01-01T00:00:00", "ns")
STEPS = (1, 4, 12)
INTERVAL_H = 6
RTOL = 1e-6
MODEL_NAME = "Stormer-ps4-6h-path-F0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_sources_status() -> Dict[str, Any]:
    pins = json.loads(PIN_PROTOCOL.read_text())["source_at_preregistration"]
    mismatched = [p for p, h in pins.items() if _sha256(SOURCE_ROOT / p) != h]
    return {"n": len(pins), "n_match": len(pins) - len(mismatched), "mismatched": mismatched}


def rel(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0 else abs(a - b) / denom


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray) or v.size <= 16}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


# =============================================================================
# Tier A steps
# =============================================================================

def step_identity_and_denorm(out: Dict[str, Any]):
    import torch
    from earthdelta.bridge.stormer_bridge import NormalizationContract
    from earthdelta.data.pull_wb2 import CANONICAL_VARIABLES
    from earthdelta.wbx import export

    manifest = json.loads((REFERENCE_DIR / "manifest.json").read_text())
    variables = list(manifest["variables"])
    order_hash = export.check_channel_order(variables)
    raw_all = np.load(S0_INPUT, mmap_mode="r")
    raw0 = np.ascontiguousarray(raw_all[0])
    raw_hash = hashlib.sha256(raw0.tobytes()).hexdigest()[:16]
    norm = NormalizationContract.from_npz_dir(
        manifest["normalization"]["dir"], variables,
        intervals=tuple(manifest["normalization"]["intervals"]),
        policy=manifest["normalization"]["policy"])
    ident_ok = norm.identity_digest == manifest["normalization"]["identity_digest"]
    out["A1_identity"] = {
        "raw_input_hash": raw_hash,
        "raw_input_hash_expected": manifest["raw_input_hash"],
        "raw_input_hash_ok": raw_hash == manifest["raw_input_hash"],
        "normalization_identity": norm.identity_digest,
        "normalization_identity_ok": ident_ok,
        "channel_order_hash": order_hash,
        "channels_are_canonical": variables == list(CANONICAL_VARIABLES),
        "reference_dir": str(REFERENCE_DIR),
    }
    x_norm = torch.load(REFERENCE_DIR / "input_norm.pt", map_location="cpu")
    states = {s: torch.load(REFERENCE_DIR / f"official_output_6h_{s}step.pt", map_location="cpu") for s in STEPS}
    std = norm.inp_std.double().numpy().reshape(1, -1, 1, 1)
    back = norm.denormalize(x_norm).double().numpy()
    fwd = norm.normalize(torch.from_numpy(raw0).unsqueeze(0)).double().numpy()
    err_back = np.abs(back - raw0[None].astype(np.float64))
    # float32 round trip loses ~eps*|mean| per value, so judge it per channel on
    # the channel's own magnitude (max|raw_c|), not elementwise near zero.
    chan_scale = np.abs(raw0.astype(np.float64)).reshape(raw0.shape[0], -1).max(axis=1).reshape(1, -1, 1, 1)
    err_fwd = np.abs(fwd - x_norm.double().numpy())
    # Wrong-space detection: official outputs are normalized, so treating them
    # as physical must FAIL the plausibility check; denormalizing twice must too.
    not_denorm_rejected = False
    try:
        export.check_physical_plausibility(states[1].numpy())
    except export.ExportContractError:
        not_denorm_rejected = True
    double_rejected = False
    try:
        export.check_physical_plausibility(norm.denormalize(norm.denormalize(states[1])).numpy())
    except export.ExportContractError:
        double_rejected = True
    raw_fields, steps = export.states_to_raw_fields(states, norm)
    out["A2_denormalization"] = {
        "denorm_input_vs_raw_max_abs": float(err_back.max()),
        "denorm_input_vs_raw_max_abs_over_std": float((err_back / std).max()),
        "denorm_input_vs_raw_max_abs_over_channel_max": float((err_back / chan_scale).max()),
        "denorm_input_vs_raw_max_elementwise_rel_note": "elementwise rel is meaningless near zero "
            "crossings (winds); per-channel scaled error is reported instead",
        "denorm_input_vs_raw_bitwise_channels": int(sum(
            np.array_equal(back[0, c].astype(np.float32), raw0[c]) for c in range(raw0.shape[0]))),
        "normalize_raw_vs_input_norm_max_abs": float(err_fwd.max()),
        "undenormalized_output_rejected": not_denorm_rejected,
        "double_denormalized_output_rejected": double_rejected,
        "plausibility_means_step1": export.check_physical_plausibility(raw_fields[:, 0]),
        "ok": bool((err_back / chan_scale).max() < 1e-6 and float(err_fwd.max()) == 0.0
                   and not_denorm_rejected and double_rejected),
    }
    return norm, raw_fields, steps, variables


def step_export_roundtrip(out, run_dir: Path, raw_fields, steps):
    import xarray as xr
    from earthdelta.wbx import export
    from earthdelta.wbx._vendor import import_official
    from earthdelta.wbx.regrid import stormer_native_grid

    lat, lon = stormer_native_grid()
    lead_hours = tuple(s * INTERVAL_H for s in steps)
    record = export.ForecastRecord(
        model=MODEL_NAME, issue_time=ISSUE_TIME, lead_hours=lead_hours,
        fields=raw_fields[0], issue_id="s0-reference-2020-01-01T00",
        provenance={"source": "official forward_validation (S0 reference)",
                    "reference_dir": str(REFERENCE_DIR)})
    ds = export.build_forecast_dataset([record], lat=lat, lon=lon)
    fc_path = run_dir / "forecast_f0.zarr"
    fc_manifest = export.write_forecast_zarr(ds, fc_path)
    reread = xr.open_zarr(fc_path).load()
    flat = export.flatten_channels(reread, leading_dims=("time", "prediction_timedelta"))
    export.verify_issue_times(reread, [ISSUE_TIME])
    schema = import_official("weatherbench2.schema")
    by_init = schema.apply_time_conventions(reread, by_init=True)
    expected_valid = ISSUE_TIME + np.array([np.timedelta64(h, "h") for h in lead_hours]).astype("timedelta64[ns]")
    wb2_valid = np.asarray(by_init["valid_time"].values).reshape(-1)
    xl = import_official("weatherbenchX.data_loaders.xarray_loaders")
    renamed = xl._rename_dataset(xr.open_zarr(fc_path))
    valid_times = [ISSUE_TIME + np.timedelta64(h, "h") for h in lead_hours]
    truth = export.truth_from_store(TRUTH_STORE, valid_times)
    tr_path = run_dir / "truth_era5_2020.zarr"
    tr_manifest = export.write_forecast_zarr(truth, tr_path, manifest_extra={"kind": "truth"})
    truth_flat = export.flatten_channels(xr.open_zarr(tr_path).load(), leading_dims=("time",))
    s0 = np.load(S0_INPUT, mmap_mode="r")
    idx = [int((t - ISSUE_TIME) / np.timedelta64(6, "h")) for t in valid_times]
    out["A3_export_roundtrip"] = {
        "forecast_path": str(fc_path),
        "forecast_manifest": fc_manifest,
        "truth_path": str(tr_path),
        "truth_manifest": tr_manifest,
        "flatten_roundtrip_bitwise": bool(np.array_equal(flat[0], raw_fields[0])),
        "decoded_time": [str(t) for t in reread["time"].values],
        "decoded_leads_h": [int(v / np.timedelta64(1, "h")) for v in reread["prediction_timedelta"].values],
        "wb2_apply_time_conventions_valid_time_ok": bool(np.array_equal(wb2_valid, expected_valid)),
        "wbx_rename_dims": sorted(renamed.dims),
        "wbx_rename_ok": {"init_time", "lead_time", "level", "latitude", "longitude"} == set(renamed.dims),
        "truth_equals_s0_input_bitwise": bool(np.array_equal(truth_flat, np.asarray(s0[idx]))),
        "truth_valid_times": [str(t) for t in valid_times],
    }
    return fc_path, tr_path, lat, lon, lead_hours


def _numpy_metrics(pred: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> Dict[str, np.ndarray]:
    """From-scratch float64 area-weighted metrics over [lat, lon] (cos-lat weights)."""
    w = np.cos(np.deg2rad(lat.astype(np.float64)))
    w = w / w.sum()
    d = pred.astype(np.float64) - truth.astype(np.float64)
    mse = np.einsum("...ij,i->...", d * d, w) / d.shape[-1]
    bias = np.einsum("...ij,i->...", d, w) / d.shape[-1]
    return {"mse": mse, "rmse": np.sqrt(mse), "bias": bias}


def step_metrics(out, fc_path, tr_path, raw_fields, norm, lat, lon, lead_hours, variables):
    import torch
    import xarray as xr
    from earthdelta.metrics_contract import full_objective_loss
    from earthdelta.static_adapter import area_weight_q
    from earthdelta.wbx import evaluate as ev
    from earthdelta.wbx._vendor import import_official

    leads = [np.timedelta64(h, "h") for h in lead_hours]
    metrics = ev.standard_metrics()
    res32, _ = ev.evaluate(str(fc_path), str(tr_path), init_times=[ISSUE_TIME], lead_times=leads,
                           metrics=metrics, aggregator=ev.make_aggregator())
    res64, _ = ev.evaluate(str(fc_path), str(tr_path), init_times=[ISSUE_TIME], lead_times=leads,
                           metrics=metrics, aggregator=ev.make_aggregator(), upcast_float64=True)
    regions = ev.official_regions()
    resreg, _ = ev.evaluate(str(fc_path), str(tr_path), init_times=[ISSUE_TIME], lead_times=leads,
                            metrics={"mse": metrics["mse"]},
                            aggregator=ev.make_aggregator(regions=regions))
    truth = xr.open_zarr(tr_path).load()
    from earthdelta.wbx import export
    truth_flat = export.flatten_channels(truth, leading_dims=("time",))      # [H, V, lat, lon]
    pred = raw_fields[0]                                                      # [H, V, lat, lon]
    npm = _numpy_metrics(pred, truth_flat, lat)                               # [H, V]

    q = area_weight_q(torch.from_numpy(lat))                                  # [1,1,Lat,1]
    pred_t = torch.from_numpy(np.ascontiguousarray(pred)).unsqueeze(0)        # [1,H,V,Lat,Lon]
    truth_t = torch.from_numpy(np.ascontiguousarray(truth_flat)).unsqueeze(0)
    H, V = pred.shape[:2]
    ed_mse = np.zeros((H, V))
    for h in range(H):
        for c in range(V):
            ed_mse[h, c] = float(full_objective_loss(
                pred_t[:, h:h + 1, c:c + 1], truth_t[:, h:h + 1, c:c + 1], q)[0])

    rows: List[Dict[str, Any]] = []
    worst = {"wbx32_vs_numpy": {}, "wbx64_vs_numpy": {}, "ed_vs_numpy": {}, "ed_vs_wbx32": {}}
    for metric in ("mse", "rmse", "bias"):
        v32 = ev.per_channel_values(res32, metric)
        v64 = ev.per_channel_values(res64, metric)
        for key in worst:
            if metric == "bias" and key.startswith("ed_"):
                continue  # the EarthDelta objective has no bias term
            worst[key][metric] = {"max_rel": 0.0, "at": None}
        for c, name in enumerate(variables):
            for h in range(H):
                a32, a64, b = float(v32[name][h]), float(v64[name][h]), float(npm[metric][h, c])
                pairs = {"wbx32_vs_numpy": (a32, b), "wbx64_vs_numpy": (a64, b)}
                if metric != "bias":
                    e = ed_mse[h, c] if metric == "mse" else math.sqrt(ed_mse[h, c])
                    pairs["ed_vs_numpy"] = (e, b)
                    pairs["ed_vs_wbx32"] = (e, a32)
                for key, (x, y) in pairs.items():
                    r = rel(x, y)
                    # bias can be ~0; judge it on the scale of the RMSE instead
                    if metric == "bias":
                        r = abs(x - y) / max(float(npm["rmse"][h, c]), 1e-300)
                    if r > worst[key][metric]["max_rel"]:
                        worst[key][metric] = {"max_rel": r, "at": f"{name}@{lead_hours[h]}h",
                                              "values": [x, y]}
                if metric == "rmse" and name in ("2m_temperature", "geopotential_500", "temperature_850",
                                                  "mean_sea_level_pressure", "u_component_of_wind_850"):
                    rows.append({"channel": name, "lead_h": lead_hours[h], "wbx_rmse": a32,
                                 "numpy_rmse": b, "ed_rmse": math.sqrt(ed_mse[h, c])})
    # Region binning: 'global' must equal the unbinned result exactly.
    g = ev.per_channel_values(resreg, "mse", extra_sel={"region": "global"})
    unb = ev.per_channel_values(res32, "mse")
    global_bin_max_rel = max(rel(float(g[n][h]), float(unb[n][h])) for n in variables for h in range(H))

    # The real EarthDelta objective (q = cos-lat, scale = official inp_std, raw space)
    scale = norm.inp_std.detach().to(torch.float64).reshape(1, -1, 1, 1)
    s2 = scale.reshape(-1).numpy() ** 2
    mse32 = ev.per_channel_values(res32, "mse")
    wbx_mse_hv = np.array([[float(mse32[n][h]) for n in variables] for h in range(H)])
    objective = {}
    for label, hs in (("24h_primary", [lead_hours.index(24)]), ("all_leads_6_24_72", list(range(H)))):
        ed_obj = float(full_objective_loss(pred_t[:, hs], truth_t[:, hs], q, scale=scale)[0])
        recon = float(np.mean(wbx_mse_hv[hs] / s2[None, :]))
        recon_np = float(np.mean(npm["mse"][hs] / s2[None, :]))
        objective[label] = {"earthdelta_full_objective_loss": ed_obj,
                            "reconstructed_from_wbx_mse": recon,
                            "reconstructed_from_numpy_mse": recon_np,
                            "rel_ed_vs_wbx": rel(ed_obj, recon), "rel_ed_vs_numpy": rel(ed_obj, recon_np)}

    # Weight identity (plan correction 8)
    weighting = import_official("weatherbenchX.weighting")
    probe = xr.DataArray(np.zeros((len(lat), 1)), dims=("latitude", "longitude"),
                         coords={"latitude": lat, "longitude": lon[:1]})
    w_wbx = np.asarray(weighting.GridAreaWeighting().weights(probe).values, dtype=np.float64)
    w_ed = q.reshape(-1).numpy()
    weight_rel = float(np.max(np.abs(w_wbx / w_wbx.sum() - w_ed / w_ed.sum()) / (w_ed / w_ed.sum())))

    checks = {
        "wbx32_vs_numpy": max(v["max_rel"] for v in worst["wbx32_vs_numpy"].values()),
        "wbx64_vs_numpy": max(v["max_rel"] for v in worst["wbx64_vs_numpy"].values()),
        "ed_vs_numpy": max(v["max_rel"] for v in worst["ed_vs_numpy"].values()),
        "ed_vs_wbx32": max(v["max_rel"] for v in worst["ed_vs_wbx32"].values()),
        "objective_ed_vs_wbx": max(o["rel_ed_vs_wbx"] for o in objective.values()),
        "global_region_bin_vs_unbinned": global_bin_max_rel,
    }
    out["A4_metric_reconciliation"] = {
        "tolerance_rel": RTOL,
        "n_channels": V, "lead_hours": list(lead_hours),
        "worst_case": worst,
        "objective": objective,
        "weights_wbx_area_vs_ed_coslat_max_rel": weight_rel,
        "checks_max_rel": checks,
        "ok": bool(all(v <= RTOL for v in checks.values())),
        "sample_rmse_rows": rows,
        "wind_vector_rmse_850_24h": float(res32["wind_vector_rmse.wind_vector"].sel(
            level=850, lead_time=np.timedelta64(24, "h")).values),
        "regions_evaluated": sorted(regions),
        "global_region_note": "official Regions binning contracts the float32 statistic with the "
            "bool mask before the float64 weights, so part of the sum runs in float32 (~5e-8 rel); "
            "with upcast_float64 the binned and unbinned results agree to ~1e-15",
    }


def step_regrid(out, raw_fields):
    from earthdelta.wbx import regrid
    s0 = np.load(S0_INPUT, mmap_mode="r")
    truth_field = np.ascontiguousarray(s0[4]).astype(np.float64)       # 2020-01-02T00, all 69
    fc_field = raw_fields[0, 1].astype(np.float64)                      # F0 +24h, all 69
    tr = regrid.cross_check_regrid(truth_field)
    fc = regrid.cross_check_regrid(fc_field)
    worst64 = max(tr["float64"]["official_vs_block_oracle"]["max_scaled"],
                  fc["float64"]["official_vs_block_oracle"]["max_scaled"],
                  tr["float64"]["official_vs_numpy"]["max_scaled"],
                  fc["float64"]["official_vs_numpy"]["max_scaled"])
    worst32 = max(tr["as_shipped"]["official_vs_numpy"]["max_scaled"],
                  fc["as_shipped"]["official_vs_numpy"]["max_scaled"])
    out["A5_regrid"] = {
        "truth_2020-01-02T00": tr,
        "forecast_f0_24h": fc,
        "criterion": "per-field max|a-b|/max|b| <= 1e-5 (float32 as shipped); float64 tier reported",
        "worst_scaled_float64": worst64,
        "worst_scaled_float32_as_shipped": worst32,
        "ok": bool(worst32 <= 1e-5 and worst64 <= 1e-12),
    }


def step_network(out, fc_path, tr_path, lead_hours, attempts: int = 3):
    """Optional; retried because reads through the proxy are occasionally truncated."""
    for attempt in range(1, attempts + 1):
        _step_network_once(out, fc_path, tr_path, lead_hours)
        out["A6_network_optional"]["attempt"] = attempt
        if out["A6_network_optional"]["ok"]:
            return
        time.sleep(10)


def _step_network_once(out, fc_path, tr_path, lead_hours):
    import xarray as xr
    from earthdelta.wbx import baselines, evaluate as ev, export, regrid
    res: Dict[str, Any] = {}
    try:
        baselines.verify_registry_against_official_configs()
        t_valid = ISSUE_TIME + np.timedelta64(24, "h")
        era5 = baselines.probe_public_store("era5_64x32", variable="2m_temperature", time=t_valid)
        ours = xr.open_zarr(tr_path).load().sel(time=t_valid)
        ours64 = regrid.official_regrid_array(
            np.asarray(ours["2m_temperature"].values, dtype=np.float64),
            np.asarray(ours["latitude"].values), np.asarray(ours["longitude"].values),
            *regrid.wb2_grid("64x32"))
        off = era5["values"]
        if era5["dims"] == ["longitude", "latitude"]:
            off = off.T
        res["era5_64x32_probe"] = {k: v for k, v in era5.items() if k != "values"}
        res["era5_64x32_vs_our_regridded_truth_t2m"] = {
            "rms_diff_K": float(np.sqrt(np.mean((ours64 - off) ** 2))),
            "max_abs_diff_K": float(np.max(np.abs(ours64 - off))),
            "note": "different regrid chains (0.25->64x32 vs 0.25->512x256->128x256->64x32); informational",
        }
        # One HRES forecast slice through the same WB-X loader/metric path.
        hres = baselines.open_public_store("hres_64x32")[["2m_temperature"]]
        hres = hres.sel(time=[ISSUE_TIME], prediction_timedelta=[np.timedelta64(24, "h")]).load()
        era5_ds = baselines.open_public_store("era5_64x32")[["2m_temperature"]].sel(time=[t_valid]).load()
        r_hres, _ = ev.evaluate(hres, era5_ds, init_times=[ISSUE_TIME], lead_times=[np.timedelta64(24, "h")],
                                metrics={"rmse": ev.standard_metrics(wind_vector=False)["rmse"]},
                                variables=["2m_temperature"])
        fc = xr.open_zarr(fc_path)[["2m_temperature"]].sel(prediction_timedelta=[np.timedelta64(24, "h")]).load()
        fc64 = regrid.regrid_dataset(fc)
        r_f0, _ = ev.evaluate(fc64, era5_ds, init_times=[ISSUE_TIME], lead_times=[np.timedelta64(24, "h")],
                              metrics={"rmse": ev.standard_metrics(wind_vector=False)["rmse"]},
                              variables=["2m_temperature"])
        res["single_issue_t2m_24h_rmse_vs_official_era5_64x32"] = {
            "hres": float(r_hres["rmse.2m_temperature"].values.reshape(-1)[0]),
            MODEL_NAME: float(r_f0["rmse.2m_temperature"].values.reshape(-1)[0]),
            "note": "ONE issue (2020-01-01T00, already exposed by S0); loader-path smoke only, not a result",
        }
        res["ok"] = True
    except Exception as exc:  # network is optional; record, never fail Tier A on it
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-2000:]
    out["A6_network_optional"] = res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--network", action="store_true", help="also probe official public 64x32 stores")
    ap.add_argument("--out-root", type=Path, default=SOURCE_ROOT / "artifacts/wbx_smoke")
    args = ap.parse_args()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_tierA"
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    out: Dict[str, Any] = {"run_id": run_id, "tier": "A", "gpu_used": False,
                           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    t0 = time.time()
    out["pinned_sources_before"] = pinned_sources_status()
    status = 0
    try:
        from earthdelta.wbx import _vendor
        out["official_code"] = _vendor.provenance()
        norm, raw_fields, steps, variables = step_identity_and_denorm(out)
        fc_path, tr_path, lat, lon, lead_hours = step_export_roundtrip(out, run_dir, raw_fields, steps)
        step_metrics(out, fc_path, tr_path, raw_fields, norm, lat, lon, lead_hours, variables)
        step_regrid(out, raw_fields)
        if args.network:
            step_network(out, fc_path, tr_path, lead_hours)
        _vendor.assert_no_beam()
        out["no_beam_modules_imported"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        status = 1
    out["pinned_sources_after"] = pinned_sources_status()
    a = out.get("A1_identity", {})
    ok_parts = {
        "A1": bool(a.get("raw_input_hash_ok") and a.get("normalization_identity_ok") and a.get("channels_are_canonical")),
        "A2": bool(out.get("A2_denormalization", {}).get("ok")),
        "A3": bool(all(out.get("A3_export_roundtrip", {}).get(k) for k in (
            "flatten_roundtrip_bitwise", "wb2_apply_time_conventions_valid_time_ok",
            "wbx_rename_ok", "truth_equals_s0_input_bitwise"))),
        "A4": bool(out.get("A4_metric_reconciliation", {}).get("ok")),
        "A5": bool(out.get("A5_regrid", {}).get("ok")),
        "pins_unchanged": out["pinned_sources_after"]["n_match"] == out["pinned_sources_after"]["n"],
    }
    out["tier_a_checks"] = ok_parts
    out["verdict"] = "TIER_A_PASS" if status == 0 and all(ok_parts.values()) else "TIER_A_FAIL"
    out["elapsed_s"] = time.time() - t0
    summary = run_dir / "smoke_summary.json"
    tmp = run_dir / ".smoke_summary.json.tmp"
    tmp.write_text(json.dumps(_jsonable(out), indent=2, sort_keys=True, default=str))
    os.replace(tmp, summary)
    print(json.dumps({"verdict": out["verdict"], "checks": ok_parts, "summary": str(summary),
                      "error": out.get("error")}, indent=2))
    return 0 if out["verdict"] == "TIER_A_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
