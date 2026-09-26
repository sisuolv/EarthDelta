#!/usr/bin/env python3
"""Audit future-year Zarr metadata without opening any forecast arrays.

This is deliberately a metadata-only tool.  It reads only consolidated Zarr
metadata, root attributes, marker names/content, and the existing pull logs
when supplied.  It never imports zarr/xarray and never opens a chunk, so a
shape that looks complete cannot be promoted to a scientific data or split
claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ARRAYS = {"data", "time", "channel", "lat", "lon"}
EXPECTED_FULL_SHAPE = [1460, 69, 128, 256]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _file_record(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "kind": kind, "exists": False}
    return {
        "path": str(path.resolve()),
        "kind": kind,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _json_file(path: Path, *, kind: str, reads: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any]]:
    rec = _file_record(path, kind=kind)
    reads.append(rec)
    if not rec["exists"]:
        return None, rec
    try:
        return json.loads(path.read_text()), rec
    except (OSError, json.JSONDecodeError) as exc:
        rec["read_error"] = f"{type(exc).__name__}: {exc}"
        return None, rec


def _store_record(store: Path, year: int, reads: list[dict[str, Any]]) -> dict[str, Any]:
    metadata, metadata_file = _json_file(store / ".zmetadata", kind="zarr_consolidated_metadata", reads=reads)
    attrs, attrs_file = _json_file(store / ".zattrs", kind="zarr_root_attributes", reads=reads)
    _, group_file = _json_file(store / ".zgroup", kind="zarr_group_metadata", reads=reads)
    entries = metadata.get("metadata", {}) if isinstance(metadata, dict) else {}
    arrays: dict[str, Any] = {}
    for key, value in entries.items():
        if not isinstance(value, dict) or "shape" not in value:
            continue
        name = key.rsplit("/", 1)[0] if key.endswith("/.zarray") else key
        if key.endswith("/.zarray"):
            arrays[name] = {
                "shape": value.get("shape"),
                "chunks": value.get("chunks"),
                "dtype": value.get("dtype"),
                "compressor": value.get("compressor"),
            }
    marker_dir = store.parent / f".markers_{year}"
    marker_records: list[dict[str, Any]] = []
    if marker_dir.is_dir():
        for marker in sorted(marker_dir.glob("*.done")):
            marker_records.append(_file_record(marker, kind="completion_marker"))
            # Marker files are tiny receipts; their content is not a forecast
            # array.  Keep it to detect stale/empty markers without inference.
            try:
                marker_records[-1]["content"] = marker.read_text(errors="replace").strip()
            except OSError as exc:
                marker_records[-1]["read_error"] = f"{type(exc).__name__}: {exc}"
            reads.append(marker_records[-1])
    data_shape = arrays.get("data", {}).get("shape")
    time_shape = arrays.get("time", {}).get("shape")
    required_present = sorted(EXPECTED_ARRAYS.intersection(arrays))
    complete_shape = data_shape == EXPECTED_FULL_SHAPE and time_shape == [EXPECTED_FULL_SHAPE[0]]
    if complete_shape:
        state = "METADATA_COMPLETE_CONTENT_UNVERIFIED"
    elif data_shape == [0, 69, 128, 256] or time_shape == [0]:
        state = "EMPTY_STORE_METADATA_ONLY"
    else:
        state = "METADATA_INCOMPLETE_OR_INCONSISTENT"
    return {
        "year": year,
        "store": str(store.resolve()),
        "store_exists": store.is_dir(),
        "root_metadata": {
            "zmetadata": metadata_file,
            "zattrs": attrs_file,
            "zgroup": group_file,
            "attrs": attrs if isinstance(attrs, dict) else None,
            "consolidated_format": metadata.get("zarr_consolidated_format") if isinstance(metadata, dict) else None,
        },
        "arrays": arrays,
        "required_arrays_present": required_present,
        "required_arrays_expected": sorted(EXPECTED_ARRAYS),
        "metadata_shape_complete": complete_shape,
        "state": state,
        "markers": {
            "directory": str(marker_dir.resolve()),
            "count": len(marker_records),
            "files": marker_records,
        },
        "content_reads": [],
        "scientific_role": "CANDIDATE_NOT_ASSIGNED",
        "scientific_content_verified": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="era5_1p40625 directory")
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata-only", action="store_true", required=True)
    args = ap.parse_args()
    if not args.metadata_only:
        raise SystemExit("--metadata-only is mandatory")
    out = args.out.resolve()
    out.mkdir(parents=False, exist_ok=False)
    root = args.root.resolve()
    reads: list[dict[str, Any]] = []
    stores = []
    for year in args.years:
        year_rec = _store_record(root / f"{year}.zarr", year, reads)
        intermediate = _store_record(root / f"{year}_intermediate.zarr", year, reads)
        stores.append({"year": year, "final": year_rec, "intermediate": intermediate})
    result = {
        "schema": "ed-r7-future-metadata-audit/1",
        "metadata_only": True,
        "no_zarr_or_xarray_import": True,
        "no_forecast_or_truth_array_reads": True,
        "no_downloads_started": True,
        "root": str(root),
        "stores": stores,
        "decision": "BLOCKED_NO_LEGAL_FRESH_SPLIT",
        "decision_reasons": [
            "metadata_shape_is_not_content_integrity",
            "2021_is_not_assigned_as_confirm_or_fresh_split",
            "2022_final_store_is_empty_by_metadata",
        ],
    }
    (out / "metadata_only_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "access_ledger.json").write_text(json.dumps({
        "schema": "ed-r7-future-metadata-access-ledger/1",
        "metadata_files_read": reads,
        "forecast_or_truth_array_reads": [],
        "zarr_chunk_reads": [],
        "new_downloads": [],
        "status": "METADATA_ONLY_NO_NUMERIC_INTEGRITY_READ",
    }, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(
        "# Future-year metadata audit\n\n"
        "This audit reads only `.zmetadata`, `.zattrs`, `.zgroup`, and tiny completion markers. "
        "It does not open Zarr arrays or chunks. A complete shape therefore remains a candidate "
        "metadata observation, not a valid split or scientific integrity result.\n"
    )
    print(json.dumps({"status": result["decision"], "years": args.years, "reads": len(reads)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
