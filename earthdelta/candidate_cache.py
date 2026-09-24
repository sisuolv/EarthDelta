"""FP-05 five-candidate cache on the certified Fs + certified K=4 bank.

Pieces (all new code; the 17 FP-04-pinned files are imported, never modified):

* `certify_admission_for_role` -- a role-parameterised COPY of
  `static_adapter.certify_admission_for_fs` (house precedent: copy, don't
  modify certified code). The tag it checks is a record-consistency check only;
  AUTHORISATION comes from `split_freeze` (tag != clearance).
* worker helpers -- guarded reads through `split_freeze.read_slab`, the
  5 candidate rollouts via `controlled_rollout(fs_bridge, ..., plan, bank)`
  exactly as FP-04's `bank_panel_losses`, the plain-Fs A3 cross-check via
  FP-04's own `fs_rollout_trajectory(fs_adapters=None)` path, and the
  report-only F0 background rollout (never a candidate).
* `legal_features_np` -- features from x_t, x_{t-6h}, x_{t-12h} and the issue
  time ONLY (no truth argument exists), built on CPU at merge.
* CACHE-VALID-v1 (`cpu_losses`, `analytic_gains`, `validate_cache_matrix`) and
  the pure deciders DEBUG-DECIDE-v1 (`decide_debug`) and DEV-SCALE-v1
  (`decide_dev_scale`, which reads no loss field).

Endpoints are stored in RAW (denormalized) space -- exactly the field the
objective scores -- so the CPU recompute differs from the GPU value only by
float64 reduction order.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

CACHE_SCHEMA = "ed-fp05-cache/1"
CANDIDATE_IDS = ("reference", "expert_0", "expert_1", "expert_2", "expert_3")
LEAD_STEPS = (1, 4, 12)
LEAD_HOURS = (6, 24, 72)
STEP_OF_LEAD = {6: 1, 24: 4, 72: 12}
STATUS_PASS = "PASS"
STATUS_NONFINITE = "FAIL_NONFINITE"
FEATURE_GRID = (2, 4)
N_CHANNELS = 69
N_FEATURES = N_CHANNELS * FEATURE_GRID[0] * FEATURE_GRID[1] * 3 + 4
V5_RELATIVE_TOL = 1e-12
V6_ATOL, V6_RTOL = 1e-11, 1e-8
V4_MAX_NONFINITE_FRACTION = 0.02


class CacheViolation(ValueError):
    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail = dict(detail or {})


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def write_json_once(path: Path | str, value: Any) -> None:
    """Atomic, write-once JSON (refuses to overwrite)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=1, allow_nan=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def save_npy_once(path: Path | str, array: np.ndarray) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as f:
        np.save(f, np.ascontiguousarray(array), allow_pickle=False)
    return file_sha256(path)


# =============================================================================
# 1. admission consumer certification (role-parameterised copy)
# =============================================================================

def _utc_seconds(value: Any) -> int:
    return int(np.datetime64(value, "s").astype("int64"))


def certify_admission_for_role(
    admission_record: Mapping[str, Any],
    *,
    expected_tag: str,
    lead_steps: Sequence[int],
    required_history_steps: int = 2,
    expected_normalization_digest: Optional[str] = None,
    expected_grid_hash: Optional[str] = None,
    expected_variable_order_hash: Optional[str] = None,
    reverify_content: bool = True,
    only_issue_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """`static_adapter.certify_admission_for_fs` with the tag as a parameter.

    Identical checks (outer/inner passed + formal + tag, B09/B08 True, per-row
    formal/tag/interval/history/lead/single-store/normalization/grid, exactly
    one passed certificate per row covering t-history..t+max(lead), UTC stamps
    equal the store's time axis, FRESH content re-hash byte-equal). With
    ``only_issue_ids`` the row checks and content re-hash run only on those
    rows (every listed id must be present) -- FP-05 re-certifies the 8 debug
    rows of FP-04's 48-row record, not the other 40.

    ``expected_tag`` checks what the record SAYS; it authorises nothing.
    """
    from .pilot_contract import PilotContractViolation, verify_content_subset

    failures: List[Dict[str, Any]] = []

    def fail(code: str, **detail: Any) -> None:
        failures.append({"code": code, **detail})

    max_step = max(int(s) for s in lead_steps)
    max_lead_hours = max_step * 6
    admission = admission_record.get("admission", {})
    for scope, obj in (("outer", admission_record), ("admission", admission)):
        if obj.get("passed") is not True:
            fail("NOT_PASSED", scope=scope)
        if obj.get("formal") is not True:
            fail("NOT_FORMAL", scope=scope)
        if obj.get("data_role") != expected_tag:
            fail("TAG_MISMATCH", scope=scope, data_role=obj.get("data_role"), expected=expected_tag)
    results = admission_record.get("results") or {}
    for key in ("b09_real_sample_admission", "b08_content_verification"):
        if results.get(key) is not True:
            fail("GATE_RESULT_NOT_TRUE", result=key, value=results.get(key))

    rows = list(admission.get("admitted") or [])
    if only_issue_ids is not None:
        wanted = [str(i) for i in only_issue_ids]
        present = {str(r.get("issue_id")) for r in rows}
        for iid in wanted:
            if iid not in present:
                fail("ISSUE_NOT_IN_RECORD", issue_id=iid)
        rows = [r for r in rows if str(r.get("issue_id")) in set(wanted)]
    if not rows:
        fail("NO_ADMITTED_ROWS")
    certificates = list(admission_record.get("content_certificates") or [])
    by_issue: Dict[str, List[Mapping[str, Any]]] = {}
    for cert in certificates:
        by_issue.setdefault(str(cert.get("issue_id")), []).append(cert)

    report_rows: List[Dict[str, Any]] = []
    for row in rows:
        iid = str(row.get("issue_id"))
        if row.get("admitted") is not True or row.get("formal") is not True:
            fail("ROW_NOT_ADMITTED_FORMAL", issue_id=iid)
        if row.get("data_role") != expected_tag:
            fail("ROW_TAG_MISMATCH", issue_id=iid, data_role=row.get("data_role"))
        if int(row.get("interval_hours", 0)) != 6:
            fail("ROW_INTERVAL_NOT_6H", issue_id=iid)
        if int(row.get("history_steps", 0)) < int(required_history_steps):
            fail("ROW_HISTORY_TOO_SHORT", issue_id=iid, history_steps=row.get("history_steps"),
                 required=int(required_history_steps))
        if float(row.get("lead_hours", 0)) < max_lead_hours:
            fail("ROW_LEAD_TOO_SHORT", issue_id=iid, lead_hours=row.get("lead_hours"),
                 required=max_lead_hours)
        stores = {str(Path(str(row.get(k))).resolve()) for k in
                  ("history_store", "issue_store", "target_store") if row.get(k) is not None}
        if len(stores) != 1:
            fail("ROW_SPANS_STORES", issue_id=iid)
        store = str(Path(str(row.get("issue_store"))).resolve())
        if expected_normalization_digest is not None and \
                row.get("normalization_hash") != expected_normalization_digest:
            fail("ROW_NORMALIZATION_MISMATCH", issue_id=iid, row=row.get("normalization_hash"),
                 expected=expected_normalization_digest)
        if expected_grid_hash is not None and row.get("grid_hash") != expected_grid_hash:
            fail("ROW_GRID_MISMATCH", issue_id=iid, row=row.get("grid_hash"), expected=expected_grid_hash)

        embedded = row.get("content_certificate")
        matches = list(by_issue.get(iid, []))
        if isinstance(embedded, Mapping):
            matches = matches or [embedded]
            if any(dict(m) != dict(embedded) for m in matches):
                fail("ROW_CERTIFICATE_CONFLICT", issue_id=iid)
        if len(matches) != 1:
            fail("ROW_CERTIFICATE_COUNT", issue_id=iid, n=len(matches))
            continue
        cert = matches[0]
        if cert.get("passed") is not True:
            fail("CERT_NOT_PASSED", issue_id=iid)
        if cert.get("data_role") != expected_tag:
            fail("CERT_TAG_MISMATCH", issue_id=iid, data_role=cert.get("data_role"))
        if cert.get("process_group_id") != row.get("process_group_id"):
            fail("CERT_PROCESS_GROUP_MISMATCH", issue_id=iid)
        if str(Path(str(cert.get("store_path"))).resolve()) != store:
            fail("CERT_STORE_MISMATCH", issue_id=iid)
        if cert.get("grid_hash") != row.get("grid_hash"):
            fail("CERT_GRID_MISMATCH", issue_id=iid)
        if expected_variable_order_hash is not None and \
                cert.get("variable_order_hash") != expected_variable_order_hash:
            fail("CERT_VARIABLE_ORDER_MISMATCH", issue_id=iid,
                 cert=cert.get("variable_order_hash"), expected=expected_variable_order_hash)
        indices = [int(i) for i in cert.get("indices") or []]
        issue_index = int(row.get("issue_index", -1))
        needed = list(range(issue_index - int(required_history_steps), issue_index + max_step + 1))
        if indices != sorted(indices) or (indices and indices != list(range(indices[0], indices[-1] + 1))):
            fail("CERT_INDICES_NOT_CONTIGUOUS", issue_id=iid)
        if not set(needed) <= set(indices):
            fail("CERT_DOES_NOT_COVER_WINDOW", issue_id=iid, needed=[needed[0], needed[-1]],
                 covered=[indices[0], indices[-1]] if indices else None)
        times = list(cert.get("times_utc") or [])
        if len(times) != len(indices):
            fail("CERT_TIMES_LENGTH", issue_id=iid)
            continue

        entry: Dict[str, Any] = {
            "issue_id": iid,
            "issue_index": issue_index,
            "store": store,
            "certificate_indices": [indices[0], indices[-1]] if indices else None,
            "content_sha256": cert.get("content_sha256"),
            "identity_sha256": cert.get("identity_sha256"),
        }
        try:
            import xarray as xr

            dataset = xr.open_zarr(store)
            try:
                store_times = dataset["time"].values
                for index, stamp in zip(indices, times):
                    if _utc_seconds(store_times[index]) != _utc_seconds(stamp):
                        fail("CERT_UTC_MISMATCH", issue_id=iid, index=index, cert=stamp,
                             store=str(store_times[index]))
                        break
                if _utc_seconds(store_times[issue_index]) != int(row.get("issue_time", -1)):
                    fail("ROW_ISSUE_TIME_MISMATCH", issue_id=iid)
            finally:
                dataset.close()
        except (OSError, KeyError, IndexError, ValueError) as exc:
            fail("STORE_UNREADABLE", issue_id=iid, error=f"{type(exc).__name__}: {exc}")
            continue

        if reverify_content:
            try:
                fresh = verify_content_subset(
                    store_path=Path(store),
                    indices=indices,
                    batch_size=int(cert.get("batch_size") or 2),
                    expected_channels=cert.get("n_channels"),
                    sigma_bound=float(cert.get("sigma_bound")),
                    normalization_dir=(Path(cert["normalization_source"])
                                       if cert.get("normalization_source") else None),
                    require_physical_range=bool(cert.get("physical_range_checked")),
                    data_role=cert.get("data_role"),
                    issue_id=cert.get("issue_id"),
                    process_group_id=cert.get("process_group_id"),
                    strict=True,
                )
                entry["fresh_content_sha256"] = fresh.content_sha256
                if fresh.content_sha256 != cert.get("content_sha256"):
                    fail("CONTENT_SHA256_MISMATCH", issue_id=iid)
                if fresh.identity_sha256 != cert.get("identity_sha256"):
                    fail("IDENTITY_SHA256_MISMATCH", issue_id=iid)
            except (PilotContractViolation, OSError, ValueError, TypeError) as exc:
                fail("CONTENT_REVERIFY_FAILED", issue_id=iid, error=f"{type(exc).__name__}: {exc}")
        report_rows.append(entry)

    report = {
        "check": "fp05_admission_consumer_certification",
        "expected_tag": expected_tag,
        "tag_is_authorisation": False,
        "only_issue_ids": list(only_issue_ids) if only_issue_ids is not None else None,
        "passed": not failures,
        "n_rows": len(rows),
        "required_history_steps": int(required_history_steps),
        "max_lead_step": max_step,
        "expected_normalization_digest": expected_normalization_digest,
        "expected_grid_hash": expected_grid_hash,
        "expected_variable_order_hash": expected_variable_order_hash,
        "content_reverified": bool(reverify_content),
        "rows": report_rows,
        "failures": failures,
    }
    if failures:
        raise CacheViolation("ADMISSION_NOT_CERTIFIED",
                             f"admission record failed {len(failures)} consumer check(s); first: "
                             f"{failures[0]}", report)
    return report


def rows_and_certificates(record: Mapping[str, Any], issue_ids: Sequence[str]
                          ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(row, certificate) for each id, in the given order; exactly one certificate each."""
    rows = {str(r["issue_id"]): dict(r) for r in record["admission"]["admitted"]}
    certs: Dict[str, List[Dict[str, Any]]] = {}
    for c in record.get("content_certificates") or []:
        certs.setdefault(str(c.get("issue_id")), []).append(dict(c))
    out = []
    for iid in issue_ids:
        if iid not in rows or len(certs.get(iid, [])) != 1:
            raise CacheViolation("ISSUE_NOT_ADMITTED", f"{iid} has no unique admitted row+certificate")
        out.append((rows[iid], certs[iid][0]))
    return out


# =============================================================================
# 2. registry pins
# =============================================================================

def registry_entries_pin(registry_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The identity-relevant fields of every registry row, in registry order."""
    out = []
    for e in registry_payload["entries"]:
        coeffs = [float(a) for a in e["coefficients"]]
        out.append({"plan_id": e["plan_id"], "coefficients": coeffs, "rho": float(e["rho"]),
                    "hold_steps": int(e["hold_steps"]), "interval_hours": int(e["interval_hours"]),
                    "continuation": e["continuation"], "is_reference": bool(e["is_reference"]),
                    "support": [i for i, a in enumerate(coeffs) if a != 0.0]})
    return out


def check_registry_entries(entries: Sequence[Any], pinned: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Exactly the 5 pinned rows, same order, same coefficients / support / hold / rho."""
    got = [{"plan_id": e.plan_id, "coefficients": [float(a) for a in e.coefficients],
            "rho": float(e.rho), "hold_steps": int(e.hold_steps),
            "interval_hours": int(e.interval_hours), "continuation": e.continuation,
            "is_reference": bool(e.is_reference), "support": list(e.support)} for e in entries]
    ok = (got == [dict(p) for p in pinned] and [g["plan_id"] for g in got] == list(CANDIDATE_IDS)
          and all(g["support"] == ([] if g["is_reference"] else [int(g["plan_id"].split("_")[1])])
                  for g in got))
    if not ok:
        raise CacheViolation("REGISTRY_NOT_THE_PINNED_ONE", "registry rows differ from the pin",
                             {"got": got, "pinned": list(pinned)})
    return {"check": "registry_entries", "n": len(got), "passed": True}


# =============================================================================
# 3. worker (GPU) helpers
# =============================================================================

def load_issue_raw(freeze, role: str, row: Mapping[str, Any], cert: Mapping[str, Any], *,
                   allow_list=None, source_admission_sha256: Optional[str] = None
                   ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """x_t and the raw truth at steps 1/4/12 -- every read through read_slab."""
    from . import split_freeze as sf

    store = row["issue_store"]
    i = int(row["issue_index"])
    kw = dict(allow_list=allow_list, source_admission_sha256=source_admission_sha256)
    x_t = sf.read_slab(freeze, role, row, cert, store, i, **kw)
    truth = {s: sf.read_slab(freeze, role, row, cert, store, i + s, **kw) for s in LEAD_STEPS}
    return x_t, truth


def load_issue_history(freeze, role: str, row: Mapping[str, Any], cert: Mapping[str, Any], *,
                       allow_list=None, source_admission_sha256: Optional[str] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x_t, x_{t-6h}, x_{t-12h} (merge-time features) -- through read_slab."""
    from . import split_freeze as sf

    store = row["issue_store"]
    i = int(row["issue_index"])
    kw = dict(allow_list=allow_list, source_admission_sha256=source_admission_sha256)
    return tuple(sf.read_slab(freeze, role, row, cert, store, i - d, **kw) for d in (0, 1, 2))


def build_sample(bridge, row: Mapping[str, Any], x_raw: np.ndarray, truth: Mapping[int, np.ndarray],
                 device):
    """A `TrainingSample` built EXACTLY as `static_adapter.load_admitted_sample` does."""
    import torch

    from .static_adapter import TrainingSample

    issue_raw = torch.as_tensor(x_raw, dtype=torch.float32).unsqueeze(0).to(device)
    targets = {int(s): torch.as_tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
               for s, v in truth.items()}
    return TrainingSample(
        x_norm=bridge.normalization.normalize(issue_raw), targets_raw=targets,
        issue_id=str(row["issue_id"]), issue_time=int(row.get("issue_time", 0)),
        valid_time=int(row.get("valid_time", 0)), history_time=int(row.get("history_time", 0)),
        split_id=str(row.get("split_id", "")), data_role=row.get("data_role"),
        source=str(row.get("issue_store")), time_utc=str(row.get("issue_time")))


def count_hooks(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
               for m in model.modules())


def _endpoints_and_losses(bridge, trajectory, sample, spec) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, str]]:
    import torch

    from .static_adapter import objective_loss_for_sample

    raws, losses, status = [], {}, {}
    for s, h in zip(LEAD_STEPS, LEAD_HOURS):
        raw = bridge.normalization.denormalize(trajectory[:, s])
        raws.append(raw[0].detach().cpu().numpy().astype(np.float32, copy=False))
        if bool(torch.isfinite(raw).all()):
            losses[str(h)] = float(objective_loss_for_sample(
                bridge, trajectory, sample, dataclasses.replace(spec, lead_steps=(s,))))
            status[str(h)] = STATUS_PASS
        else:
            losses[str(h)] = None
            status[str(h)] = STATUS_NONFINITE
    return np.stack(raws), losses, status


def rollout_issue(fs_bridge, f0_bridge, bank: Mapping[int, Any], entries: Sequence[Any], sample,
                  spec, variables: Sequence[str], blocks: Sequence[int]) -> Dict[str, Any]:
    """5 candidates (+ plain-Fs A3 cross-check + F0 background), no grad.

    Candidate rollout = FP-04 `bank_panel_losses`' call, verbatim:
    ``controlled_rollout(fs_bridge, x, names, interval=6, steps=12, plan=entry.to_edit_plan(),
    expert_loras=dict(bank), target_blocks=blocks, return_trajectory=True)``.
    """
    import time

    import torch

    from .bridge.stormer_bridge import controlled_rollout
    from .static_adapter import fs_rollout_trajectory

    names = list(variables)
    blocks = tuple(int(b) for b in blocks)
    out: Dict[str, Any] = {"candidates": {}, "hooks_after": {}, "seconds": {}}
    endpoints = []
    with torch.no_grad():
        for entry in entries:
            t = time.perf_counter()
            traj = controlled_rollout(fs_bridge, sample.x_norm, names, interval=6, steps=12,
                                      plan=entry.to_edit_plan(), expert_loras=dict(bank),
                                      target_blocks=blocks, return_trajectory=True)
            raw, losses, status = _endpoints_and_losses(fs_bridge, traj, sample, spec)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out["seconds"][entry.plan_id] = time.perf_counter() - t
            out["hooks_after"][entry.plan_id] = count_hooks(fs_bridge.model)
            out["candidates"][entry.plan_id] = {"losses": losses, "status": status}
            endpoints.append(raw)
            del traj
        t = time.perf_counter()
        plain = fs_rollout_trajectory(fs_bridge, sample.x_norm, names, steps=12, fs_adapters=None,
                                      target_blocks=blocks)
        plain_raw, plain_losses, plain_status = _endpoints_and_losses(fs_bridge, plain, sample, spec)
        out["seconds"]["plain_fs"] = time.perf_counter() - t
        t = time.perf_counter()
        f0 = fs_rollout_trajectory(f0_bridge, sample.x_norm, names, steps=12, fs_adapters=None,
                                   target_blocks=blocks)
        f0_raw, f0_losses, f0_status = _endpoints_and_losses(f0_bridge, f0, sample, spec)
        out["seconds"]["f0_background"] = time.perf_counter() - t
    out["endpoints"] = np.stack(endpoints)              # [5, 3, V, Lat, Lon] float32 raw
    out["plain_fs"] = {"endpoints": plain_raw, "losses": plain_losses, "status": plain_status}
    out["f0_background"] = {"endpoints": f0_raw, "losses": f0_losses, "status": f0_status}
    out["reference_equals_plain_fs_bitwise"] = bool(
        np.array_equal(out["endpoints"][0], plain_raw)
        and out["endpoints"][0].tobytes() == plain_raw.tobytes())
    return out


# =============================================================================
# 4. legal features (CPU, merge time)
# =============================================================================

def _pool(x: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    v, h, w = x.shape
    gh, gw = grid
    if h % gh or w % gw:
        raise CacheViolation("FEATURE_GRID", "pooling grid must divide the field")
    return x.reshape(v, gh, h // gh, gw, w // gw).mean(axis=(2, 4))


def legal_features_np(x_t: np.ndarray, x_tm6: np.ndarray, x_tm12: np.ndarray, issue_time: int,
                      inp_mean: np.ndarray, inp_std: np.ndarray,
                      grid: Tuple[int, int] = FEATURE_GRID) -> np.ndarray:
    """float64 [69*8*3 + 4]: pooled normalized x_t, x_t - x_{t-6}, x_t - x_{t-12}, calendar.

    Takes no truth, endpoint or loss argument: the policy can only ever see
    what was known at the issue time.
    """
    mean = np.asarray(inp_mean, dtype=np.float64).reshape(-1, 1, 1)
    std = np.asarray(inp_std, dtype=np.float64).reshape(-1, 1, 1)
    norm = lambda x: (np.asarray(x, dtype=np.float64) - mean) / std
    a, b, c = norm(x_t), norm(x_tm6), norm(x_tm12)
    hours = (int(issue_time) - int(np.datetime64(
        np.datetime64(int(issue_time), "s").astype("datetime64[Y]"), "s").astype("int64"))) / 3600.0
    day = hours / 24.0
    hour = (int(issue_time) // 3600) % 24
    cal = np.array([math.sin(2 * math.pi * day / 365.25), math.cos(2 * math.pi * day / 365.25),
                    math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)])
    feats = np.concatenate([_pool(a, grid).ravel(), _pool(a - b, grid).ravel(),
                            _pool(a - c, grid).ravel(), cal])
    if feats.shape != (N_FEATURES,) or not np.isfinite(feats).all():
        raise CacheViolation("FEATURES_INVALID", "features not finite / wrong length")
    return feats


# =============================================================================
# 5. CACHE-VALID-v1
# =============================================================================

def cpu_losses(endpoints: np.ndarray, truth: np.ndarray, q, scale) -> List[List[Optional[float]]]:
    """Canonical float64 CPU losses [candidate][lead] (None where nonfinite)."""
    import torch

    from .metrics_contract import full_objective_loss

    out = []
    for c in range(endpoints.shape[0]):
        row = []
        for j in range(endpoints.shape[1]):
            pred = torch.from_numpy(np.ascontiguousarray(endpoints[c, j]))[None, None]
            tgt = torch.from_numpy(np.ascontiguousarray(truth[j]))[None, None]
            if not bool(torch.isfinite(pred).all()):
                row.append(None)
                continue
            row.append(float(full_objective_loss(pred, tgt, q, scale=scale).mean()))
        out.append(row)
    return out


def analytic_gains(endpoints: np.ndarray, truth: np.ndarray, q, scale) -> List[List[Optional[float]]]:
    """V6: g_c = full_objective_gain(Y - F_ref, F_c - F_ref) per lead."""
    import torch

    from .metrics_contract import full_objective_gain

    out = []
    ref = endpoints[0]
    for c in range(endpoints.shape[0]):
        row = []
        for j in range(endpoints.shape[1]):
            if not (np.isfinite(endpoints[c, j]).all() and np.isfinite(ref[j]).all()):
                row.append(None)
                continue
            e = torch.from_numpy(truth[j].astype(np.float64) - ref[j].astype(np.float64))[None, None]
            u = torch.from_numpy(endpoints[c, j].astype(np.float64) - ref[j].astype(np.float64))[None, None]
            row.append(float(full_objective_gain(e, u, q, scale=scale).mean()))
        out.append(row)
    return out


def matrix_rows_for_issue(issue: Mapping[str, Any], *, cpu: Sequence[Sequence[Optional[float]]],
                          analytic: Sequence[Sequence[Optional[float]]], entries_pin,
                          identity_sha256: str, features_sha256: Optional[str]) -> List[Dict[str, Any]]:
    """The 5 x 3 matrix rows of one issue (canonical loss = CPU loss)."""
    rows = []
    for ci, pin in enumerate(entries_pin):
        cid = pin["plan_id"]
        for j, h in enumerate(LEAD_HOURS):
            gpu = issue["gpu_losses"][cid][str(h)]
            status = issue["status"][cid][str(h)]
            ref_cpu = cpu[0][j]
            loss = cpu[ci][j]
            rows.append({
                "issue_id": issue["issue_id"], "role": issue["role"], "shard": issue["shard"],
                "candidate_id": cid, "coefficients": pin["coefficients"], "support": pin["support"],
                "lead_hours": h, "step": STEP_OF_LEAD[h], "issue_time": int(issue["issue_time"]),
                "valid_time": int(issue["issue_time"]) + h * 3600, "status": status,
                "gpu_loss": gpu, "cpu_loss": loss,
                "gain_vs_fs": (ref_cpu - loss) if (loss is not None and ref_cpu is not None) else None,
                "analytic_gain": analytic[ci][j],
                "endpoint_file": issue["files"]["endpoints"]["path"],
                "endpoint_sha256": issue["files"]["endpoints"]["sha256"],
                "truth_sha256": issue["files"]["truth"]["sha256"],
                "features_sha256": features_sha256, "identity_sha256": identity_sha256,
                "seconds": issue["seconds"].get(cid)})
    return rows


def validate_cache_matrix(rows: Sequence[Mapping[str, Any]], *, expected_issue_ids: Sequence[str],
                          entries_pin: Sequence[Mapping[str, Any]], expected_identity_sha256: str,
                          shard_of: Mapping[str, int], rehash: Mapping[str, Mapping[str, Any]],
                          read_check: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """CACHE-VALID-v1 V1-V8 over the flat matrix. Raises INVALID_EVIDENCE on any failure.

    ``rehash`` maps issue_id -> {"endpoints_ok", "truth_ok", "shape_dtype_ok"} from
    the merge's own re-hash of the files (V3). ``read_check`` is the V8 report.
    """
    fails: Dict[str, List[Any]] = {f"V{i}": [] for i in range(1, 9)}
    pins = {p["plan_id"]: p for p in entries_pin}
    expected = {(i, c, h) for i in expected_issue_ids for c in pins for h in LEAD_HOURS}
    seen = set()
    for r in rows:
        key = (r["issue_id"], r["candidate_id"], int(r["lead_hours"]))
        if key in seen:
            fails["V1"].append(["duplicate", key])
        seen.add(key)
        if shard_of.get(r["issue_id"]) != r.get("shard"):
            fails["V1"].append(["shard", key])
        pin = pins.get(r["candidate_id"])
        if r.get("identity_sha256") != expected_identity_sha256 or pin is None or \
                r.get("coefficients") != pin["coefficients"] or r.get("support") != pin["support"]:
            fails["V2"].append(key)
        rh = rehash.get(r["issue_id"]) or {}
        if not (rh.get("endpoints_ok") and rh.get("truth_ok") and rh.get("shape_dtype_ok")):
            fails["V3"].append(key)
        st = r.get("status")
        if st not in (STATUS_PASS, STATUS_NONFINITE):
            fails["V4"].append(["status", key, st])
        if st == STATUS_PASS:
            gpu, cpu = r.get("gpu_loss"), r.get("cpu_loss")
            if gpu is None or cpu is None or not (math.isfinite(gpu) and math.isfinite(cpu)):
                fails["V4"].append(["pass_without_finite_loss", key])
            elif abs(cpu - gpu) > V5_RELATIVE_TOL * abs(cpu):
                fails["V5"].append(key)
            g, a = r.get("gain_vs_fs"), r.get("analytic_gain")
            if g is not None:
                if a is None or abs(g - a) > V6_ATOL + V6_RTOL * abs(g):
                    fails["V6"].append(key)
        elif r.get("gpu_loss") is not None or r.get("cpu_loss") is not None:
            fails["V4"].append(["nonfinite_with_loss", key])
        if STEP_OF_LEAD.get(int(r["lead_hours"])) != r.get("step") or \
                int(r["valid_time"]) != int(r["issue_time"]) + int(r["lead_hours"]) * 3600:
            fails["V7"].append(key)
    roles = {r.get("role") for r in rows}
    if len(roles) > 1:
        fails["V2"].append(["mixed_roles", sorted(map(str, roles))])
    if seen != expected:
        fails["V1"].append(["missing", len(expected - seen), "extra", len(seen - expected)])
    n_nonfinite = sum(1 for r in rows if r.get("status") == STATUS_NONFINITE)
    frac = n_nonfinite / max(1, len(rows))
    if frac > V4_MAX_NONFINITE_FRACTION:
        fails["V4"].append(["nonfinite_fraction", frac])
    if read_check is not None and read_check.get("passed") is not True:
        fails["V8"].append(read_check.get("violations"))
    report = {"check": "CACHE-VALID-v1", "n_rows": len(rows), "n_expected": len(expected),
              "n_nonfinite": n_nonfinite, "nonfinite_fraction": frac,
              "max_abs_cpu_gpu_relative": max(
                  [abs(r["cpu_loss"] - r["gpu_loss"]) / abs(r["cpu_loss"]) for r in rows
                   if r.get("status") == STATUS_PASS and r.get("cpu_loss")] or [0.0]),
              "failures": {k: v[:16] for k, v in fails.items() if v},
              "v_pass": {k: not v for k, v in fails.items()},
              "passed": not any(fails.values())}
    if not report["passed"]:
        raise CacheViolation("INVALID_EVIDENCE", f"CACHE-VALID-v1 failed: "
                             f"{sorted(k for k, v in fails.items() if v)}", report)
    return report


def matrix_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Canonical digest over rows sorted by (issue, candidate, lead) -- not file bytes."""
    key = lambda r: (r["issue_id"], r["candidate_id"], int(r["lead_hours"]))
    return canonical_digest(sorted((dict(r) for r in rows), key=key))


# =============================================================================
# 6. DEBUG-DECIDE-v1 and DEV-SCALE-v1 (pure)
# =============================================================================

def decide_debug(*, e0_by_worker: Mapping[str, bool], matrix_passed: bool,
                 anchors: Mapping[str, Mapping[str, Any]], duplicates_identical: Mapping[str, bool],
                 reference_equals_plain_fs: Mapping[str, bool], inexact_rel_tol: float = 1e-6
                 ) -> Dict[str, Any]:
    """PASS iff every condition holds; anchors must be BITWISE equal.

    ``anchors[issue_id]`` = {"fs": [(got, want) x3], "expert": {"k": int, "pairs": [(got, want) x3]}}.
    Anchors within ``inexact_rel_tol`` but not exact -> DEBUG_ANCHOR_INEXACT (to the
    coordinator, never an auto-pass); anything else failing -> STOP.
    """
    def pairs(a):
        return list(a["fs"]) + list(a["expert"]["pairs"])

    exact = {i: all(g is not None and g == w for g, w in pairs(a)) for i, a in anchors.items()}
    close = {i: all(g is not None and w is not None and abs(g - w) <= inexact_rel_tol * abs(w)
                    for g, w in pairs(a)) for i, a in anchors.items()}
    checks = {
        "e0_all_workers": len(e0_by_worker) == 4 and all(e0_by_worker.values()),
        "matrix_8x5x3_valid": bool(matrix_passed),
        "anchors_present_for_8": len(anchors) == 8,
        "duplicates_byte_identical": len(duplicates_identical) == 8 and all(duplicates_identical.values()),
        "reference_equals_plain_fs": len(reference_equals_plain_fs) == 8
        and all(reference_equals_plain_fs.values()),
    }
    anchors_exact = bool(anchors) and all(exact.values())
    if all(checks.values()) and anchors_exact:
        verdict = "PASS"
    elif all(checks.values()) and all(close.values()):
        verdict = "DEBUG_ANCHOR_INEXACT"
    else:
        verdict = "STOP"
    return {"rule": "DEBUG-DECIDE-v1", "verdict": verdict, "checks": checks,
            "anchors_exact": anchors_exact, "anchors_exact_by_issue": exact,
            "anchors_within_1e-6_by_issue": close}


def decide_dev_scale(*, subsets: Mapping[str, Sequence[str]], admitted_ids: Sequence[str],
                     model_load_seconds: float, issue_seconds: Sequence[float],
                     bytes_per_issue: float, free_disk_bytes: float, peak_gpu_gib: float,
                     n_shards: int = 4, time_budget_s: float = 45 * 60,
                     bytes_cap: float = 60e9, memory_cap_gib: float = 64.0) -> Dict[str, Any]:
    """DEV-SCALE-v1: N = max n in {N_target, N/2, N/4} meeting (a)-(d). No loss is read.

    Inputs are admission membership, timing, size and memory only.
    """
    if not issue_seconds:
        raise CacheViolation("DEV_SCALE_NO_TIMING", "no measured per-issue seconds")
    p95 = float(np.percentile(np.asarray(issue_seconds, dtype=np.float64), 95))
    admitted = set(admitted_ids)
    table = {}
    for name in ("N", "N/2", "N/4"):
        ids = list(subsets[name])
        n = len(ids)
        t = model_load_seconds + math.ceil(n / n_shards) * p95 * 1.5
        need = n * bytes_per_issue * 1.2
        table[name] = {"n": n, "a_all_admitted": set(ids) <= admitted,
                       "b_seconds": t, "b_ok": t <= time_budget_s,
                       "c_bytes": need, "c_ok": need <= bytes_cap and need <= free_disk_bytes,
                       "d_peak_gib": peak_gpu_gib, "d_ok": peak_gpu_gib <= memory_cap_gib}
        table[name]["ok"] = all(table[name][k] for k in ("a_all_admitted", "b_ok", "c_ok", "d_ok"))
    chosen = next((k for k in ("N", "N/2", "N/4") if table[k]["ok"]), None)
    return {"rule": "DEV-SCALE-v1", "p95_issue_seconds": p95, "table": table,
            "chosen_subset": chosen, "N": table[chosen]["n"] if chosen else None,
            "verdict": "CHOSEN" if chosen else "STOP"}


def shard_map(issue_ids_time_ordered: Sequence[str], n_shards: int = 4) -> Dict[str, int]:
    """i mod n_shards, interleaved over the time-ordered issue list."""
    return {iid: i % n_shards for i, iid in enumerate(issue_ids_time_ordered)}
