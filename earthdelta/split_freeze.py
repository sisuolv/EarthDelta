"""FP-05 data-role isolation gate: the frozen split, enforced at every consumer.

FP-05a found that the admission tag (`pilot_contract.DataRole`) isolates
nothing: every admission so far is tagged ``bank_fit`` whatever its purpose,
and the admission layer explicitly leaves enforcement to the consumer. This
module is that enforcement. Its single rule:

    WHETHER A ROW MAY BE CONSUMED DEPENDS ONLY ON THE FROZEN, HASH-PINNED
    SPLIT FREEZE (derived from the FP-05a exposure ledger) PLUS AN EXPLICIT
    ALLOW-LIST -- NEVER ON THE ROW'S ``data_role`` TAG.

The tag is read only to be recorded next to the verdict. A "role" here is an
FP-05 *consumption* role (``policy_dev`` / ``debug``), distinct from
``DataRole``; ``confirm`` is UNASSIGNED_NO_ACCESS (every request raises) and
``bank_fit`` is not consumable by FP-05 at all.

Conventions (pinned in the freeze, re-derived by `rebuild_from_ledger`):

* times are UTC epoch seconds on the 6h grid;
* a row's support is ``[issue - 12h, issue + 72h]`` (wider when the row
  declares more history), a CLOSED interval;
* an exposure run widened by ``buffer_hours`` on each side, also closed;
  a support that *touches* it (shares even one instant) is not clear.

`read_slab` is the only real-data I/O path FP-05 code uses: it refuses any
store or time index outside the row's certified slice and appends every read to
an in-process read log that workers persist.

numpy/stdlib only at import; xarray is imported lazily by `read_slab`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "ed-split-freeze/1"
HOUR = 3600
INTERVAL_HOURS = 6
HISTORY_HOURS = 12
LEAD_HOURS = 72
CONSUMABLE_ROLES = ("policy_dev", "debug")
ALL_ROLES = ("policy_dev", "debug", "confirm", "bank_fit")
CONFIRM_STATUS = "UNASSIGNED_NO_ACCESS"
CONSERVATIVE_RULE = "conservative_full_24h_buffer_incl_S0_aggregate_result_2020-01-01T00Z"
RESULT_KINDS = ("GRADIENT_AND_RESULT", "RESULT")


class SplitFreezeViolation(ValueError):
    """A consumer asked for something the frozen split does not allow."""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail = dict(detail or {})


# =============================================================================
# time helpers
# =============================================================================

def parse_utc(text: str) -> int:
    """'2019-07-06T12Z' / '2019-07-06T12:00:00Z' / '2019-07-06T12:00:00' -> epoch s."""
    s = str(text).strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    raise SplitFreezeViolation("BAD_UTC", f"unparseable UTC stamp {text!r}")


def iso(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%Y-%m-%dT%HZ")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def touches(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Closed-interval intersection (sharing a single instant counts)."""
    return a[0] <= b[1] and b[0] <= a[1]


def merge_runs(intervals: Iterable[Tuple[int, int]], *, adjacent_s: int = INTERVAL_HOURS * HOUR
               ) -> List[Tuple[int, int]]:
    """Union of closed grid intervals; runs one grid step apart are one run."""
    out: List[List[int]] = []
    for s, e in sorted((int(a), int(b)) for a, b in intervals):
        if out and s <= out[-1][1] + adjacent_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


# =============================================================================
# the pure derivation (ledger -> freeze core)
# =============================================================================

def _ledger_payload(ledger: Mapping[str, Any] | Path | str) -> Dict[str, Any]:
    if isinstance(ledger, (str, Path)):
        return json.loads(Path(ledger).read_text())
    return dict(ledger)


def exposure_intervals_from_ledger(ledger: Mapping[str, Any] | Path | str, *,
                                   boundary_rule: str = CONSERVATIVE_RULE
                                   ) -> Dict[str, Any]:
    """Recompute the gradient/result exposure runs from the ledger's ROWS.

    Every admitted row whose exposure kind is GRADIENT_AND_RESULT or RESULT
    contributes the union of its uniform ``t-12h..t+72h`` support and its
    recorded support; under the conservative rule the S0 gate's aggregate
    result exposure (``s0_input.result_exposure``) is added. Nothing is copied
    from the ledger's own ``runs_by_year`` summary -- that is only compared.
    """
    led = _ledger_payload(ledger)
    raw: List[Tuple[int, int]] = []
    certified_only: List[Tuple[int, int]] = []
    n_rows = 0
    for row in led["rows"]:
        if row.get("exposure_kind") not in RESULT_KINDS or row.get("admitted") is not True:
            continue
        n_rows += 1
        uni = row["uniform_support_t-12h_t+72h"]
        rec = row.get("recorded_support") or uni
        s = min(parse_utc(uni[0]), parse_utc(rec[0]))
        e = max(parse_utc(uni[1]), parse_utc(rec[1]))
        issue = parse_utc(row["issue_utc"])
        if s != min(issue - HISTORY_HOURS * HOUR, parse_utc(rec[0])) or e < issue + LEAD_HOURS * HOUR:
            raise SplitFreezeViolation("LEDGER_ROW_SUPPORT", f"row {row.get('issue_id')} support "
                                       f"is not t-12h..t+72h-covering", {"row": row.get("issue_id")})
        raw.append((s, e))
        certified_only.append((s, e))
    extra = []
    if boundary_rule == CONSERVATIVE_RULE:
        res = led["s0_input"]["result_exposure"]["utc_range"]
        extra.append((parse_utc(res[0]), parse_utc(res[1])))
    elif boundary_rule != "full_24h_buffer_certified_rows_only":
        raise SplitFreezeViolation("UNKNOWN_BOUNDARY_RULE", boundary_rule)
    runs = merge_runs(raw + extra)
    return {"boundary_rule": boundary_rule, "n_result_rows": n_rows,
            "runs": runs, "certified_rows_only_runs": merge_runs(certified_only),
            "s0_aggregate_result": [iso(a) for a in extra[0]] if extra else None}


def _store_grid(year: int) -> Tuple[int, int]:
    t0 = parse_utc(f"{year}-01-01T00")
    t1 = parse_utc(f"{year + 1}-01-01T00")
    return t0, (t1 - t0) // (INTERVAL_HOURS * HOUR)


def clear_issue_indices(year: int, runs: Sequence[Tuple[int, int]], buffer_hours: int,
                        *, cross_year: bool = True) -> List[int]:
    """Store indices t of `year` whose full support lies in the store and is clear."""
    t0, n = _store_grid(year)
    step = INTERVAL_HOURS * HOUR
    buf = int(buffer_hours) * HOUR
    y0, y1 = t0, t0 + (n - 1) * step
    widened = [(s - buf, e + buf) for s, e in runs
               if cross_year or (y0 <= s <= y1 or y0 <= e <= y1)]
    out = []
    for i in range(HISTORY_HOURS // INTERVAL_HOURS, n - LEAD_HOURS // INTERVAL_HOURS):
        t = t0 + i * step
        sup = (t - HISTORY_HOURS * HOUR, t + LEAD_HOURS * HOUR)
        if not any(touches(sup, w) for w in widened):
            out.append(i)
    return out


def rebuild_from_ledger(ledger: Mapping[str, Any] | Path | str, *,
                        boundary_rule: str = CONSERVATIVE_RULE, buffer_hours: int = 24,
                        policy_dev_year: int = 2019, policy_dev_from_utc: str = "2019-07-01T00"
                        ) -> Dict[str, Any]:
    """The freeze's derived core, as a PURE function of the ledger.

    Returns exposed runs (unbuffered and buffered), the policy_dev pool (the
    single contiguous run of clear 2019 H2 slots) and the pool under the two
    alternative boundary rules for the record. `load_freeze(..., ledger=...)`
    and CP0 compare the pinned freeze against this.
    """
    exp = exposure_intervals_from_ledger(ledger, boundary_rule=boundary_rule)
    runs = exp["runs"]
    t0, _ = _store_grid(policy_dev_year)
    step = INTERVAL_HOURS * HOUR
    h2 = parse_utc(policy_dev_from_utc)

    def pool(rs, cross_year=True):
        idx = [i for i in clear_issue_indices(policy_dev_year, rs, buffer_hours,
                                              cross_year=cross_year) if t0 + i * step >= h2]
        return idx

    idx = pool(runs)
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        raise SplitFreezeViolation("POOL_NOT_CONTIGUOUS", "the policy_dev pool is not one run")
    alt_cert = pool(exp["certified_rows_only_runs"])
    alt_nocross = pool([r for r in exp["certified_rows_only_runs"]], cross_year=False)
    buf = buffer_hours * HOUR
    return {
        "boundary_rule": boundary_rule,
        "buffer_hours": int(buffer_hours),
        "n_result_rows": exp["n_result_rows"],
        "s0_aggregate_result_utc": exp["s0_aggregate_result"],
        "exposed_runs_unbuffered_utc": [[iso(s), iso(e), (e - s) // HOUR] for s, e in runs],
        "exposed_runs_buffered_utc": [[iso(s - buf), iso(e + buf), (e - s + 2 * buf) // HOUR]
                                      for s, e in runs],
        "policy_dev_pool": {
            "store_year": policy_dev_year,
            "first_index": idx[0], "last_index": idx[-1], "n_slots": len(idx),
            "first_issue_utc": iso(t0 + idx[0] * step), "last_issue_utc": iso(t0 + idx[-1] * step),
            "certificate_span_indices": [idx[0] - 2, idx[-1] + 12],
        },
        "alternative_rules_for_the_record": {
            "full_24h_buffer_certified_rows_only": {
                "n_slots": len(alt_cert), "first_issue_utc": iso(t0 + alt_cert[0] * step),
                "last_issue_utc": iso(t0 + alt_cert[-1] * step)},
            "no_cross_year_buffer": {
                "n_slots": len(alt_nocross), "first_issue_utc": iso(t0 + alt_nocross[0] * step),
                "last_issue_utc": iso(t0 + alt_nocross[-1] * step)},
        },
    }


# =============================================================================
# the loaded freeze
# =============================================================================

@dataclass(frozen=True)
class AllowList:
    """An explicit (admission sha256, issue ids) authorisation for one role."""

    role: str
    admission_sha256: str
    issue_ids: FrozenSet[str]

    @classmethod
    def from_mapping(cls, role: str, payload: Mapping[str, Any]) -> "AllowList":
        adm = payload.get("admission") or {}
        sha = adm.get("sha256") if isinstance(adm, Mapping) else None
        sha = sha or payload.get("admission_sha256")
        ids = payload.get("issue_ids") or payload.get("allowed_issue_ids")
        if not sha or not ids:
            raise SplitFreezeViolation("ALLOW_LIST_INCOMPLETE",
                                       f"allow-list for {role!r} lacks an admission sha256 or ids")
        return cls(role=role, admission_sha256=str(sha), issue_ids=frozenset(str(i) for i in ids))


@dataclass(frozen=True)
class SplitFreeze:
    path: str
    sha256: str
    payload: Dict[str, Any] = field(repr=False)

    @property
    def buffer_s(self) -> int:
        return int(self.payload["buffer_hours"]) * HOUR

    @property
    def exposed(self) -> List[Tuple[int, int]]:
        return [(parse_utc(a), parse_utc(b)) for a, b, _ in
                self.payload["derived"]["exposed_runs_unbuffered_utc"]]

    @property
    def stores(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.payload["stores_allowed"])

    def role(self, name: str) -> Dict[str, Any]:
        return dict(self.payload["roles"][name])

    def debug_allow_list(self) -> AllowList:
        d = self.role("debug")
        return AllowList(role="debug", admission_sha256=d["source_admission"]["sha256"],
                         issue_ids=frozenset(d["allowed_issue_ids"]))


def load_freeze(path: Path | str, expected_sha256: str, *,
                ledger: Optional[Mapping[str, Any] | Path | str] = None) -> SplitFreeze:
    """Load the freeze only if its bytes hash to ``expected_sha256``.

    Also refuses an unknown schema, a confirm role that is anything but
    UNASSIGNED_NO_ACCESS, a bank_fit role consumable by FP-05, and -- when
    ``ledger`` is supplied -- a derived core that differs from
    `rebuild_from_ledger(ledger)`.
    """
    if not expected_sha256:
        raise SplitFreezeViolation("FREEZE_SHA256_MISSING", "no expected freeze sha256 given")
    actual = file_sha256(path)
    if actual != str(expected_sha256):
        raise SplitFreezeViolation("FREEZE_SHA256_MISMATCH",
                                   f"{path} hashes to {actual}, not {expected_sha256}",
                                   {"actual": actual, "expected": str(expected_sha256)})
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise SplitFreezeViolation("FREEZE_SCHEMA", f"schema {payload.get('schema')!r} != {SCHEMA}")
    roles = payload.get("roles") or {}
    if (roles.get("confirm") or {}).get("status") != CONFIRM_STATUS or \
            (roles.get("confirm") or {}).get("window") is not None:
        raise SplitFreezeViolation("FREEZE_CONFIRM_ASSIGNED", "the freeze names a confirm window")
    if (roles.get("bank_fit") or {}).get("consumable_by_fp05") is not False:
        raise SplitFreezeViolation("FREEZE_BANK_FIT_CONSUMABLE", "bank_fit must not be consumable")
    derived = payload.get("derived") or {}
    if canonical_digest(derived) != payload.get("derived_digest"):
        raise SplitFreezeViolation("FREEZE_INCONSISTENT", "derived core does not match its digest")
    if ledger is not None:
        rebuilt = rebuild_from_ledger(ledger, boundary_rule=derived.get("boundary_rule"),
                                      buffer_hours=int(payload["buffer_hours"]))
        if rebuilt != derived:
            raise SplitFreezeViolation("FREEZE_INCONSISTENT",
                                       "the freeze differs from rebuild_from_ledger(ledger)")
    return SplitFreeze(path=str(path), sha256=actual, payload=payload)


# =============================================================================
# row classification
# =============================================================================

def row_support(row: Mapping[str, Any]) -> Tuple[int, int]:
    """[history_time, issue + 72h] of an admission / manifest row (epoch s)."""
    issue = int(row["issue_time"])
    lead = row.get("lead_hours")
    if lead is None and row.get("valid_time") is not None:
        lead = (int(row["valid_time"]) - issue) / HOUR
    if lead is None or float(lead) != float(LEAD_HOURS):
        raise SplitFreezeViolation("ROW_LEAD_NOT_72H", f"row {row.get('issue_id')} lead {lead}",
                                   {"issue_id": row.get("issue_id")})
    if int(row.get("interval_hours", INTERVAL_HOURS)) != INTERVAL_HOURS:
        raise SplitFreezeViolation("ROW_INTERVAL_NOT_6H", f"row {row.get('issue_id')}")
    if row.get("valid_time") is not None and int(row["valid_time"]) != issue + LEAD_HOURS * HOUR:
        raise SplitFreezeViolation("ROW_VALID_TIME", f"row {row.get('issue_id')}")
    start = issue - HISTORY_HOURS * HOUR
    if row.get("history_steps") is not None:
        hs = int(row["history_steps"])
        if hs < HISTORY_HOURS // INTERVAL_HOURS:
            raise SplitFreezeViolation("ROW_HISTORY_TOO_SHORT", f"row {row.get('issue_id')}")
        start = min(start, issue - hs * INTERVAL_HOURS * HOUR)
    if row.get("history_time") is not None:
        start = min(start, int(row["history_time"]))
    return start, issue + LEAD_HOURS * HOUR


def _store_name(row: Mapping[str, Any], support: Tuple[int, int]) -> str:
    stores = {Path(str(row[k])).name for k in ("history_store", "issue_store", "target_store")
              if row.get(k) is not None}
    if len(stores) > 1:
        raise SplitFreezeViolation("ROW_SPANS_STORES", f"row {row.get('issue_id')}: {stores}")
    if stores:
        return stores.pop()
    y0 = datetime.fromtimestamp(support[0], timezone.utc).year
    y1 = datetime.fromtimestamp(support[1], timezone.utc).year
    if y0 != y1:
        raise SplitFreezeViolation("ROW_SPANS_STORES", f"row {row.get('issue_id')} spans years")
    return f"{y0}.zarr"


def classify_row(freeze: SplitFreeze, row: Mapping[str, Any], role: str, *,
                 allow_list: Optional[AllowList] = None,
                 source_admission_sha256: Optional[str] = None,
                 pre_admission: bool = False) -> Dict[str, Any]:
    """Every fact the clearance decision uses, plus the verdict. Never reads data.

    The row's ``data_role`` tag is RECORDED (``tag``) and never consulted.
    """
    if role not in ALL_ROLES:
        raise SplitFreezeViolation("UNKNOWN_ROLE", f"role {role!r}")
    if role == "confirm":
        raise SplitFreezeViolation("CONFIRM_UNASSIGNED_NO_ACCESS",
                                   "confirm is UNASSIGNED_NO_ACCESS: no FP-05 code may read it",
                                   {"issue_id": row.get("issue_id")})
    if role == "bank_fit":
        raise SplitFreezeViolation("ROLE_NOT_CONSUMABLE_BY_FP05",
                                   "bank_fit rows are historical; FP-05 consumes policy_dev/debug only")
    spec = freeze.role(role)
    iid = str(row.get("issue_id"))
    sup = row_support(row)
    issue = int(row["issue_time"])
    store = _store_name(row, sup)
    stores = freeze.stores
    reasons: List[str] = []
    store_allowed = store in stores
    if not store_allowed:
        reasons.append("STORE_NOT_ALLOWED")
    elif row.get("issue_store") is not None and \
            str(Path(str(row["issue_store"])).resolve()) != str(Path(stores[store]["path"]).resolve()):
        store_allowed = False
        reasons.append("STORE_PATH_NOT_THE_FROZEN_ONE")
    if store != spec["store"]:
        reasons.append("STORE_NOT_THE_ROLE_STORE")
    index_ok = True
    if store_allowed and row.get("issue_index") is not None:
        t0 = parse_utc(stores[store]["time0_utc"])
        step = INTERVAL_HOURS * HOUR
        ii = int(row["issue_index"])
        index_ok = (t0 + ii * step == issue
                    and (row.get("history_index") is None
                         or t0 + int(row["history_index"]) * step == sup[0])
                    and (row.get("target_index") is None
                         or t0 + int(row["target_index"]) * step == sup[1])
                    and 0 <= (sup[0] - t0) // step and (sup[1] - t0) // step < int(stores[store]["n_timesteps"]))
        if not index_ok:
            reasons.append("INDEX_TIME_INCONSISTENT")
    win = [parse_utc(x) for x in spec["support_window_utc"]]
    inside = win[0] <= sup[0] and sup[1] <= win[1]
    if not inside:
        reasons.append("OUTSIDE_ROLE_WINDOW")
    guard = spec.get("issue_guard_window_utc")
    in_guard = False
    if guard is not None:
        g0, g1 = parse_utc(guard[0]), parse_utc(guard[1])
        in_guard = not (g0 <= issue < g1)
        if in_guard:
            reasons.append("IN_SPLIT_GUARD_WINDOW")
    buf = freeze.buffer_s
    touching = [[iso(s), iso(e)] for s, e in freeze.exposed if touches(sup, (s - buf, e + buf))]
    excluded_hit = []
    if row.get("history_index") is not None and row.get("target_index") is not None:
        span = range(int(row["history_index"]), int(row["target_index"]) + 1)
        excluded_hit = [i for i in spec.get("excluded_store_indices", []) if i in span]
        if excluded_hit:
            reasons.append("CONTAINS_EXCLUDED_STORE_INDEX")
    on_list: Optional[bool] = None
    if role == "debug":
        allow_list = freeze.debug_allow_list()
    if role == "policy_dev":
        if touching:
            reasons.append("TOUCHES_EXPOSED_RUN_WITHIN_BUFFER")
    if pre_admission:
        if role != "policy_dev":
            raise SplitFreezeViolation("PRE_ADMISSION_ONLY_FOR_POLICY_DEV",
                                       "only policy_dev admission may be checked before an allow-list")
    else:
        if allow_list is None or allow_list.role != role:
            reasons.append("NO_ALLOW_LIST")
            on_list = False
        else:
            on_list = iid in allow_list.issue_ids
            if not on_list:
                reasons.append("NOT_ON_ALLOW_LIST")
            if source_admission_sha256 != allow_list.admission_sha256:
                reasons.append("SOURCE_ADMISSION_NOT_THE_ALLOWED_ONE")
    return {
        "issue_id": iid, "role": role, "tag": row.get("data_role"),
        "support_utc": [iso(sup[0]), iso(sup[1])], "store": store,
        "store_allowed": store_allowed, "index_time_consistent": index_ok,
        "inside_role_window": inside, "in_guard_window": in_guard,
        "touches_exposed_buffered": touching, "excluded_indices_hit": excluded_hit,
        "on_allow_list": on_list, "pre_admission": bool(pre_admission),
        "clear": not reasons, "reasons": reasons,
    }


def assert_rows_clear(freeze: SplitFreeze, rows: Sequence[Mapping[str, Any]], role: str, *,
                      allow_list: Optional[AllowList] = None,
                      source_admission_sha256: Optional[str] = None,
                      pre_admission: bool = False) -> Dict[str, Any]:
    """Classify every row; raise ``ROW_NOT_CLEAR`` listing each failing row."""
    if not rows:
        raise SplitFreezeViolation("NO_ROWS", "nothing to clear")
    verdicts = [classify_row(freeze, r, role, allow_list=allow_list,
                             source_admission_sha256=source_admission_sha256,
                             pre_admission=pre_admission) for r in rows]
    ids = [v["issue_id"] for v in verdicts]
    if len(set(ids)) != len(ids):
        raise SplitFreezeViolation("DUPLICATE_ISSUE", "a row appears twice")
    bad = [v for v in verdicts if not v["clear"]]
    report = {"check": "split_freeze_clearance", "freeze_sha256": freeze.sha256, "role": role,
              "pre_admission": bool(pre_admission), "n_rows": len(rows),
              "n_clear": len(rows) - len(bad), "passed": not bad, "rows": verdicts}
    if bad:
        raise SplitFreezeViolation("ROW_NOT_CLEAR",
                                   f"{len(bad)} of {len(rows)} row(s) not clear for {role!r}; "
                                   f"first {bad[0]['issue_id']}: {bad[0]['reasons']}", report)
    return report


# =============================================================================
# the single I/O path
# =============================================================================

_READ_LOG: List[Dict[str, Any]] = []
_OPEN: Dict[str, Any] = {}


def read_log() -> List[Dict[str, Any]]:
    return [dict(e) for e in _READ_LOG]


def reset_read_log() -> None:
    _READ_LOG.clear()


def close_all() -> None:
    for ds in _OPEN.values():
        try:
            ds.close()
        except Exception:  # noqa: BLE001 - closing must not mask a failure
            pass
    _OPEN.clear()


def _open_store(path: str):
    """Lazy xarray open, cached per store (patched in tests to count opens)."""
    if path not in _OPEN:
        import xarray as xr

        _OPEN[path] = xr.open_zarr(path)
    return _OPEN[path]


def allowed_indices(freeze: SplitFreeze, role: str, row: Mapping[str, Any],
                    certificate: Mapping[str, Any]) -> range:
    """The row's certified slice ``[history_index, target_index]``, fully covered
    by its passed content certificate for the same issue and store."""
    if certificate.get("passed") is not True or str(certificate.get("issue_id")) != str(row.get("issue_id")):
        raise SplitFreezeViolation("CERTIFICATE_NOT_FOR_ROW", f"row {row.get('issue_id')}")
    if str(Path(str(certificate.get("store_path"))).resolve()) != \
            str(Path(str(row.get("issue_store"))).resolve()):
        raise SplitFreezeViolation("CERTIFICATE_STORE", f"row {row.get('issue_id')}")
    span = range(int(row["history_index"]), int(row["target_index"]) + 1)
    cert = [int(i) for i in certificate.get("indices") or []]
    if not set(span) <= set(cert):
        raise SplitFreezeViolation("CERTIFICATE_DOES_NOT_COVER", f"row {row.get('issue_id')}")
    return span


def read_slab(freeze: SplitFreeze, role: str, row: Mapping[str, Any],
              certificate: Mapping[str, Any], store_path: Path | str, index: int, *,
              allow_list: Optional[AllowList] = None,
              source_admission_sha256: Optional[str] = None) -> np.ndarray:
    """Read ONE ``[V, Lat, Lon]`` float32 slab -- only inside the certified slice.

    Re-classifies the row (fail-closed), refuses any other store or any index
    outside `allowed_indices`, checks the store's own time coordinate at the
    index, and logs the read.
    """
    verdict = classify_row(freeze, row, role, allow_list=allow_list,
                           source_admission_sha256=source_admission_sha256)
    if not verdict["clear"]:
        raise SplitFreezeViolation("ROW_NOT_CLEAR", f"{verdict['issue_id']}: {verdict['reasons']}",
                                   verdict)
    frozen_path = str(Path(freeze.stores[freeze.role(role)["store"]]["path"]).resolve())
    if str(Path(str(store_path)).resolve()) != frozen_path:
        raise SplitFreezeViolation("READ_STORE_NOT_ALLOWED", f"{store_path} is not {frozen_path}")
    span = allowed_indices(freeze, role, row, certificate)
    if int(index) not in span:
        raise SplitFreezeViolation("READ_OUTSIDE_CERTIFIED_SLICE",
                                   f"index {index} not in [{span.start}, {span.stop - 1}] "
                                   f"for {row.get('issue_id')}",
                                   {"index": int(index), "slice": [span.start, span.stop - 1]})
    ds = _open_store(frozen_path)
    stamp = int(np.datetime64(ds["time"].values[int(index)], "s").astype("int64"))
    want = int(row["issue_time"]) + (int(index) - int(row["issue_index"])) * INTERVAL_HOURS * HOUR
    if stamp != want:
        raise SplitFreezeViolation("READ_TIME_MISMATCH", f"store time {iso(stamp)} != {iso(want)}")
    values = np.asarray(ds["data"].isel(time=int(index)).values, dtype=np.float32)
    _READ_LOG.append({"store": Path(frozen_path).name, "index": int(index), "role": role,
                      "issue_id": str(row.get("issue_id"))})
    return values


def read_log_within(freeze: SplitFreeze, log: Sequence[Mapping[str, Any]],
                    slices: Mapping[str, Tuple[str, str, int, int]]) -> Dict[str, Any]:
    """V8: every logged read is (role store, index inside that issue's slice).

    ``slices`` maps issue_id -> (role, store_name, first_index, last_index).
    """
    bad = []
    for entry in log:
        sl = slices.get(str(entry.get("issue_id")))
        if sl is None:
            bad.append({**entry, "why": "issue not in the declared set"})
            continue
        role, store, a, b = sl
        if entry.get("role") != role or entry.get("store") != store or \
                store != freeze.role(role)["store"] or not (a <= int(entry.get("index", -1)) <= b):
            bad.append({**entry, "why": "outside the certified slice / wrong store or role"})
    return {"check": "V8_read_set", "n_reads": len(log), "passed": not bad and bool(log),
            "violations": bad[:32]}
