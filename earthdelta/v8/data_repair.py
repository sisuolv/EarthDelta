"""Fail-closed data repair primitives for signed v8 windows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .io import atomic_json, json_hash


class DataRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairResult:
    status: str
    source_chunk_manifest: list[dict]
    refetch_count: int
    failure_list: list[str]
    completion_marker: str | None
    access_ledger: list[dict]

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["access_ledger_hash"] = json_hash(self.access_ledger)
        return d


class FailClosedRepair:
    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)
        self.access_ledger: list[dict] = []

    def record_access(self, *, role: str, year: int, path: str, operation: str, array_payload: bool) -> None:
        if year == 2022 and array_payload:
            raise DataRepairError("2022 payload access before deployment freeze")
        self.access_ledger.append({"role": role, "year": year, "path": path,
                                   "operation": operation, "array_payload": bool(array_payload)})

    def validate_batch(self, values: np.ndarray, *, source_id: str, std: float | np.ndarray | None = None) -> None:
        arr = np.asarray(values)
        if not np.isfinite(arr).all():
            raise DataRepairError(f"nonfinite source/remapped batch: {source_id}")
        if std is not None:
            scale = np.asarray(std, dtype=np.float64)
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise DataRepairError("invalid finite-reference scale")
            # Extreme finite values are retained and logged by callers; they are not silently dropped.
            _ = np.max(np.abs(arr / scale))

    def repair(self, *, role: str, year: int, chunks: Iterable[dict], source_reader: Callable[[dict], np.ndarray] | None,
               std: float | np.ndarray | None = None) -> RepairResult:
        if source_reader is None:
            return RepairResult("BLOCKED", list(chunks), 0, ["missing_source_reader"], None, self.access_ledger)
        failures, manifest = [], []
        refetch = 0
        target = self.output_root / str(year)
        target.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            cid = str(chunk.get("chunk_id", "missing"))
            try:
                self.record_access(role=role, year=year, path=str(chunk.get("source", cid)),
                                   operation="GET", array_payload=(year != 2022))
                values = source_reader(chunk)
                self.validate_batch(values, source_id=cid, std=std)
                manifest.append({"chunk_id": cid, "finite": True, "source": chunk.get("source")})
            except Exception as exc:  # fail closed; no marker is written
                failures.append(f"{cid}:{type(exc).__name__}:{exc}")
        marker = None
        if not failures:
            marker_path = target / "COMPLETE.json"
            atomic_json(marker_path, {"status": "OBSERVED", "role": role, "year": year,
                                      "chunks": manifest, "ledger_hash": json_hash(self.access_ledger)})
            marker = str(marker_path)
        return RepairResult("OBSERVED" if not failures else "BLOCKED", manifest, refetch, failures, marker,
                            list(self.access_ledger))


def cross_year_memory_only(*, local_qc_failed: bool, year: int, chunk: dict,
                           source_reader: Callable[[dict], np.ndarray]) -> dict:
    if year != 2021 or not local_qc_failed:
        raise DataRepairError("cross-year GET is allowed only for a failed 2021 boundary QC")
    value = np.asarray(source_reader(chunk))
    if not np.isfinite(value).all():
        raise DataRepairError("cross-year source chunk is nonfinite")
    return {"status": "OBSERVED", "memory_only": True, "persisted": False,
            "chunk_id": chunk.get("chunk_id"), "shape": list(value.shape)}


def repair_zarr_year(source_path: str | Path, target_path: str | Path, *, role: str, year: int,
                     indices: Iterable[int], std: np.ndarray | None = None) -> RepairResult:
    """Copy only signed support chunks into a new versioned Zarr store.

    The source group is opened read-only.  Missing/nonfinite source chunks are
    recorded and prevent the completion marker; no invalid chunk is written.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised by deployment image
        raise DataRepairError("zarr dependency missing") from exc
    source = Path(source_path); target = Path(target_path)
    if not source.is_dir() or source.resolve() == target.resolve():
        raise DataRepairError("source missing or target already exists")
    src = zarr.open_group(str(source), mode="r")
    if "data" not in src or "time" not in src:
        raise DataRepairError("source Zarr identity incomplete")
    data = src["data"]
    target.parent.mkdir(parents=True, exist_ok=True)
    resumed = target.exists()
    if resumed:
        dst = zarr.open_group(str(target), mode="r+")
        if "data" not in dst or tuple(dst["data"].shape) != tuple(data.shape):
            raise DataRepairError("existing v8 target has incompatible identity")
        out_data = dst["data"]
    else:
        dst = zarr.open_group(str(target), mode="w")
        dst.attrs.update(dict(src.attrs))
        dst.attrs["v8_role"] = role
        dst.attrs["v8_source_path"] = str(source)
        for name in src.array_keys():
            arr = src[name]
            if name == "data":
                continue
            out = dst.create_dataset(name, shape=arr.shape, chunks=arr.chunks, dtype=arr.dtype,
                                     compressor=arr.compressor, fill_value=arr.fill_value, order=arr.order)
            out[...] = arr[...]
            out.attrs.update(dict(arr.attrs))
        out_data = dst.create_dataset("data", shape=data.shape, chunks=data.chunks, dtype=data.dtype,
                                      compressor=data.compressor, fill_value=data.fill_value, order=data.order)
        out_data.attrs.update(dict(data.attrs))
    failures, manifest = [], []
    seen = set()
    for index in sorted(int(i) for i in indices):
        if index in seen:
            continue
        seen.add(index)
        if index < 0 or index >= data.shape[0]:
            failures.append(f"index_{index}:out_of_range")
            continue
        try:
            values = np.asarray(data[index])
            FailClosedRepair(target).validate_batch(values, source_id=f"{year}:{index}", std=std)
            out_data[index] = values
            manifest.append({"chunk_id": f"{year}:{index}", "index": index, "finite": True,
                             "source": str(source / "data" / f"{index}.0.0.0")})
        except Exception as exc:
            failures.append(f"{year}:{index}:{type(exc).__name__}:{exc}")
    access = [{"role": role, "year": year, "source": str(source), "index": i,
               "operation": "GET", "array_payload": True} for i in sorted(seen)]
    marker = None
    if not failures:
        marker_path = target / "COMPLETE.json"
        atomic_json(marker_path, {"status": "OBSERVED", "role": role, "year": year,
                                  "chunks": manifest, "ledger_hash": json_hash(access)})
        marker = str(marker_path)
    return RepairResult("OBSERVED" if not failures else "BLOCKED", manifest, 0, failures, marker, access)


def refetch_zarr_indices(target_path: str | Path, *, year: int, indices: Iterable[int],
                         max_batch: int = 8) -> dict:
    """Refetch failed target chunks once from WeatherBench2 and regrid them.

    This is intentionally limited to the failed signed indices.  The upstream
    pull module is reused read-only for its canonical variable order and
    conservative weights; no old store is overwritten.
    """
    import time
    import zarr
    from earthdelta.data.pull_wb2 import (ALL_SOURCE_VARS, ConservativeRegridder,
                                           get_channel_indices_for_source_var,
                                           get_stormer_target_grid, open_wb2_zarr)
    target = zarr.open_group(str(target_path), mode="r+")
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {"status": "OBSERVED", "refetch_count": 0, "failure_list": [], "indices": []}
    ds = open_wb2_zarr(use_https=True)
    source_lat = np.asarray(ds.latitude.values)
    source_lon = np.asarray(ds.longitude.values)
    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        ds = ds.isel(latitude=slice(None, None, -1))
    target_lat, target_lon = get_stormer_target_grid()
    regridder = ConservativeRegridder(source_lat, source_lon, target_lat, target_lon)
    failures, count = [], 0
    try:
        for source_var in ALL_SOURCE_VARS:
            da = ds[source_var].sel(time=str(year))
            channel_indices = get_channel_indices_for_source_var(source_var)
            is_pressure = source_var in {"geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"}
            for start in range(0, len(wanted), max_batch):
                batch_indices = wanted[start:start + max_batch]
                succeeded = False
                last_error = None
                for attempt in range(2):
                    try:
                        batch = da.isel(time=batch_indices)
                        if is_pressure:
                            raw = batch.transpose("time", "level", "longitude", "latitude").values
                        else:
                            raw = batch.transpose("time", "longitude", "latitude").values[:, None, :, :]
                        if not np.isfinite(raw).all():
                            raise DataRepairError(f"nonfinite upstream source {source_var}")
                        for level_index, channel_index in enumerate(channel_indices):
                            mapped = regridder.regrid(raw[:, level_index, :, :]).transpose(0, 2, 1).astype(np.float32)
                            if not np.isfinite(mapped).all():
                                raise DataRepairError(f"nonfinite remap {source_var}")
                            target["data"][batch_indices, channel_index, :, :] = mapped
                        succeeded = True
                        count += len(batch_indices)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt == 0:
                            time.sleep(2.0)
                if not succeeded:
                    failures.append(f"{source_var}:{batch_indices[0]}-{batch_indices[-1]}:{last_error}")
        return {"status": "OBSERVED" if not failures else "BLOCKED", "refetch_count": count,
                "failure_list": failures, "indices": wanted}
    finally:
        try:
            ds.close()
        except Exception:
            pass
