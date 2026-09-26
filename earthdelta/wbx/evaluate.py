"""Score exported forecasts with the official WeatherBench-X code, without beam.

Everything metric-related is the vendored official implementation:
``weatherbenchX.metrics.deterministic`` (RMSE, MSE, Bias, ACC,
WindVectorRMSE), ``weighting.GridAreaWeighting``, ``binning.Regions`` (with the
region table read -- not copied -- from the official
``public_benchmark/run_benchmark_evaluation.py``), ``aggregation.Aggregator`` /
``AggregationState`` and the ``xarray_loaders`` data loaders. This module only
replaces the beam pipeline with a plain loop over init-time chunks that does the
same thing the official pipeline does per chunk: load -> compute statistics ->
aggregate -> sum states -> metric values.

It also holds the evaluation-year gate: reading truth or scoring a year is an
exposure, so by default only the already-exposed year 2020 is allowed; any
other year requires a confirm-freeze file whose SHA-256 the caller supplies.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._vendor import REFERENCE_ROOT, assert_no_beam, import_official

#: Years whose ERA5 data EarthDelta has already been exposed to. Anything else
#: (notably the 2021/2022 confirm candidates) needs a confirm-freeze file.
DEFAULT_AUTHORIZED_YEARS = frozenset({2020})
CONFIRM_FREEZE_YEARS_KEY = "authorized_evaluation_years"


class YearNotAuthorized(PermissionError):
    """Evaluation touched a year that has not been authorized."""


def assert_years_authorized(
    years: Iterable[int],
    *,
    confirm_freeze_path: Optional[Union[str, Path]] = None,
    confirm_freeze_sha256: Optional[str] = None,
    authorized: frozenset = DEFAULT_AUTHORIZED_YEARS,
) -> Dict[str, Any]:
    """Fail closed unless every year is exposed-by-default or confirm-frozen.

    For a year outside ``authorized`` the caller must supply BOTH a
    confirm-freeze JSON file and its expected SHA-256; the file must hash to
    that value and list the year under ``authorized_evaluation_years``.
    """
    if frozenset(int(y) for y in authorized) != DEFAULT_AUTHORIZED_YEARS:
        raise YearNotAuthorized(
            "caller-supplied authorized years are forbidden in the production "
            "evaluation path; use a hash-bound confirm-freeze file"
        )
    ys = sorted({int(y) for y in years})
    if not ys:
        raise YearNotAuthorized("no evaluation years given")
    extra = [y for y in ys if y not in authorized]
    record: Dict[str, Any] = {"years": ys, "default_authorized": sorted(authorized)}
    if not extra:
        record["gate"] = "default_exposed_years"
        return record
    if confirm_freeze_path is None or not confirm_freeze_sha256:
        raise YearNotAuthorized(
            f"years {extra} are not authorized for evaluation; supply a confirm-freeze "
            "file and its SHA-256 (no year other than the already-exposed "
            f"{sorted(authorized)} may be read by default)")
    path = Path(confirm_freeze_path)
    if not path.is_file():
        raise YearNotAuthorized(f"confirm-freeze file {path} does not exist")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != str(confirm_freeze_sha256).strip().lower():
        raise YearNotAuthorized(
            f"confirm-freeze file {path} hashes to {actual}, not the expected "
            f"{confirm_freeze_sha256}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise YearNotAuthorized(f"confirm-freeze file {path} is not JSON: {exc}") from exc
    allowed = {int(y) for y in doc.get(CONFIRM_FREEZE_YEARS_KEY, [])}
    not_listed = [y for y in extra if y not in allowed]
    if not_listed:
        raise YearNotAuthorized(
            f"confirm-freeze file {path} does not authorize years {not_listed}")
    record.update({"gate": "confirm_freeze", "confirm_freeze_path": str(path),
                   "confirm_freeze_sha256": actual})
    return record


# =============================================================================
# Official components
# =============================================================================

_ARITH = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
          ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def _safe_literal(node):
    """``ast.literal_eval`` plus + - * / on numbers (REGIONS uses ``360 - 120``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _safe_literal(node.operand)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        return _ARITH[type(node.op)](_safe_literal(node.left), _safe_literal(node.right))
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_safe_literal(k): _safe_literal(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"unsupported node in REGIONS: {ast.dump(node)[:120]}")


def official_regions() -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The public-benchmark REGIONS table, parsed from the vendored source.

    ``run_benchmark_evaluation.py`` imports apache_beam at top level, so it is
    read with ``ast`` (constants and arithmetic only) rather than imported.
    """
    import_official("weatherbenchX.binning")  # pins verified before reading source
    src = (REFERENCE_ROOT / "weatherbenchX" / "public_benchmark" / "run_benchmark_evaluation.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "REGIONS" for t in node.targets):
            return _safe_literal(node.value)
    raise RuntimeError("REGIONS not found in official run_benchmark_evaluation.py")


def standard_metrics(climatology=None, *, wind_vector: bool = True) -> Dict[str, Any]:
    """Official deterministic metrics: rmse, mse, bias (+ wind_vector_rmse, acc)."""
    det = import_official("weatherbenchX.metrics.deterministic")
    metrics: Dict[str, Any] = {"rmse": det.RMSE(), "mse": det.MSE(), "bias": det.Bias()}
    if wind_vector:
        metrics["wind_vector_rmse"] = det.WindVectorRMSE(
            u_name=["u_component_of_wind", "10m_u_component_of_wind"],
            v_name=["v_component_of_wind", "10m_v_component_of_wind"],
            vector_name=["wind_vector", "10m_wind_vector"],
        )
    if climatology is not None:
        metrics["acc"] = det.ACC(climatology=climatology)
    return metrics


def make_aggregator(
    *,
    reduce_init_time: bool = True,
    regions: Optional[Mapping[str, Any]] = None,
    land_sea_mask=None,
    masked: bool = False,
):
    """Official Aggregator with GridAreaWeighting (as in the public benchmark)."""
    aggregation = import_official("weatherbenchX.aggregation")
    weighting = import_official("weatherbenchX.weighting")
    binning = import_official("weatherbenchX.binning")
    reduce_dims = (["init_time", "latitude", "longitude"] if reduce_init_time
                   else ["latitude", "longitude"])
    bin_by = [binning.Regions(dict(regions), land_sea_mask=land_sea_mask)] if regions else None
    return aggregation.Aggregator(
        reduce_dims=reduce_dims,
        weigh_by=[weighting.GridAreaWeighting()],
        bin_by=bin_by,
        masked=masked,
        skipna=False,
    )


def _loader(kind: str, source, variables: Optional[Sequence[str]]):
    xl = import_official("weatherbenchX.data_loaders.xarray_loaders")
    kwargs: Dict[str, Any] = {"variables": list(variables) if variables else None}
    if isinstance(source, (str, Path)):
        kwargs["path"] = str(source)
    else:
        kwargs["ds"] = source
    if kind == "prediction":
        return xl.PredictionsFromXarray(**kwargs)
    return xl.TargetsFromXarray(**kwargs)


def _as_timedelta64(t) -> np.timedelta64:
    """Lead time -> ``timedelta64[ns]``; bare integers are hours."""
    if isinstance(t, np.timedelta64):          # (subclass of np.integer: test first)
        return t.astype("timedelta64[ns]")
    if isinstance(t, (int, np.integer)):
        return np.timedelta64(int(t), "h").astype("timedelta64[ns]")
    import pandas as pd
    return pd.Timedelta(t).to_timedelta64().astype("timedelta64[ns]")


def _chunks(values: np.ndarray, size: int) -> List[np.ndarray]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def evaluate(
    prediction,
    truth,
    *,
    init_times: Sequence[Any],
    lead_times: Sequence[Any],
    metrics: Optional[Mapping[str, Any]] = None,
    aggregator=None,
    variables: Optional[Sequence[str]] = None,
    init_chunk_size: int = 1,
    upcast_float64: bool = False,
    confirm_freeze_path: Optional[Union[str, Path]] = None,
    confirm_freeze_sha256: Optional[str] = None,
    authorized: frozenset = DEFAULT_AUTHORIZED_YEARS,
):
    """Chunked official evaluation (the beam pipeline's per-chunk steps, in a loop).

    Args:
        prediction: exported forecast Dataset or zarr path (official naming).
        truth: truth Dataset (``time`` = valid time) or zarr path.
        init_times: issue times to score (datetime64-like).
        lead_times: lead times (timedelta64-like, e.g. hours as np.timedelta64).
        metrics: name -> WB-X metric (default ``standard_metrics()``).
        aggregator: WB-X Aggregator (default ``make_aggregator()``).
        variables: restrict to these variables.
        init_chunk_size: init times per chunk; the result must not depend on it.
        upcast_float64: cast loaded chunks to float64 before computing
            statistics (the official code otherwise squares in the stored
            float32 precision). Used only to separate storage rounding from
            real discrepancies in reconciliation.
        confirm_freeze_path / confirm_freeze_sha256: optional authorization
            for years outside the already-exposed default set. The gate runs
            before any loader is constructed or data is read.
        authorized: override the default exposed-year set for controlled tests.

    Returns:
        (metric values Dataset, final AggregationState)
    """
    # Enforce the exposure boundary before importing/constructing data loaders.
    # A lead can cross a calendar boundary (for example 2020-12-31 + 72h
    # reads 2021 truth). Gate both issue and valid years before any loader is
    # constructed; checking init years alone would permit an exposure hole.
    requested_years = {int(np.datetime64(t, "Y").astype(int)) + 1970 for t in init_times}
    init_arr = np.asarray([np.datetime64(t, "ns") for t in init_times], dtype="datetime64[ns]")
    lead_arr = np.asarray([_as_timedelta64(t) for t in lead_times], dtype="timedelta64[ns]")
    requested_years.update(
        int(v.astype("datetime64[Y]").astype(int)) + 1970
        for v in (init_arr[:, None] + lead_arr[None, :]).reshape(-1)
    )
    assert_years_authorized(requested_years, confirm_freeze_path=confirm_freeze_path,
                            confirm_freeze_sha256=confirm_freeze_sha256,
                            authorized=authorized)
    aggregation = import_official("weatherbenchX.aggregation")
    base = import_official("weatherbenchX.metrics.base")
    metrics = dict(metrics) if metrics is not None else standard_metrics()
    aggregator = aggregator if aggregator is not None else make_aggregator()
    inits = np.array([np.datetime64(t, "ns") for t in init_times], dtype="datetime64[ns]")
    leads = np.array([_as_timedelta64(t) for t in lead_times], dtype="timedelta64[ns]")
    pred_loader = _loader("prediction", prediction, variables)
    truth_loader = _loader("target", truth, variables)
    state = aggregation.AggregationState.zero()
    for chunk in _chunks(inits, max(1, int(init_chunk_size))):
        p = pred_loader.load_chunk(chunk, leads)
        t = truth_loader.load_chunk(chunk, leads)
        if upcast_float64:
            p = {k: v.astype(np.float64) for k, v in dict(p).items()}
            t = {k: v.astype(np.float64) for k, v in dict(t).items()}
        stats = base.compute_unique_statistics_for_all_metrics(metrics, p, t)
        state = state + aggregator.aggregate_statistics(stats)
    assert_no_beam()
    return state.metric_values(metrics), state


def evaluate_single_chunk(
    prediction_chunk,
    truth_chunk,
    *,
    metrics=None,
    aggregator=None,
    init_times: Optional[Sequence[Any]] = None,
    valid_times: Optional[Sequence[Any]] = None,
    confirm_freeze_path: Optional[Union[str, Path]] = None,
    confirm_freeze_sha256: Optional[str] = None,
    authorized: frozenset = DEFAULT_AUTHORIZED_YEARS,
):
    """Official single-chunk aggregation on already-aligned data.

    The arrays are already loaded, so this helper cannot infer their year. A
    caller must pass ``init_times``; the same year gate as :func:`evaluate`
    then runs before importing the official aggregator. This keeps the public
    single-chunk entry point from silently bypassing the exposure contract.
    """
    if init_times is None:
        raise YearNotAuthorized("init_times are required for single-chunk evaluation")
    requested_years = {int(np.datetime64(t, "Y").astype(int)) + 1970 for t in init_times}
    if valid_times is not None:
        requested_years.update(int(np.datetime64(t, "Y").astype(int)) + 1970
                               for t in valid_times)
    assert_years_authorized(requested_years, confirm_freeze_path=confirm_freeze_path,
                            confirm_freeze_sha256=confirm_freeze_sha256,
                            authorized=authorized)
    aggregation = import_official("weatherbenchX.aggregation")
    metrics = dict(metrics) if metrics is not None else standard_metrics()
    aggregator = aggregator if aggregator is not None else make_aggregator()
    return aggregation.compute_metric_values_for_single_chunk(
        metrics, aggregator, prediction_chunk, truth_chunk)


def per_channel_values(results, metric: str, *, lead_time=None, extra_sel: Optional[Mapping[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Map ``<metric>.<variable>`` results back to the 69 Stormer channel names.

    Returns channel -> value array (over any remaining dims, e.g. lead_time).
    """
    from ..data.pull_wb2 import CANONICAL_VARIABLES, SINGLE_LEVEL_VARS
    out: Dict[str, np.ndarray] = {}
    for name in CANONICAL_VARIABLES:
        if name in SINGLE_LEVEL_VARS:
            da = results[f"{metric}.{name}"]
        else:
            var, lvl = name.rsplit("_", 1)
            da = results[f"{metric}.{var}"].sel(level=int(lvl))
        if lead_time is not None:
            da = da.sel(lead_time=lead_time)
        if extra_sel:
            da = da.sel(dict(extra_sel))
        out[name] = np.asarray(da.values, dtype=np.float64)
    return out


__all__ = [
    "DEFAULT_AUTHORIZED_YEARS", "YearNotAuthorized", "assert_years_authorized",
    "official_regions", "standard_metrics", "make_aggregator", "evaluate",
    "evaluate_single_chunk", "per_channel_values",
]
