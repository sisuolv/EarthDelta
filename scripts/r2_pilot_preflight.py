#!/usr/bin/env python3
"""Admission preflight for a pilot data store.

The pilot pull decides a store is usable from its shape alone. This entry point
runs the content-level admission checks in `earthdelta.pilot_contract` against a
REAL store and a REAL sample loaded out of it, and exits non-zero when the store
is not fit to be consumed.

Run it before anything that trains or evaluates on a pilot store:

    python scripts/r2_pilot_preflight.py --store data/2020_jan.zarr
    python scripts/r2_pilot_preflight.py --store data/2020_jan.zarr \\
        --index 4 --history-steps 1 --target-steps 4 --json report.json

What it refuses:

  * a sample whose history or target endpoint falls outside the record, or
    whose span is not uniformly spaced at the forecast interval;
  * a store whose latitude/longitude values or order do not match the canonical
    Stormer target grid (same shape, flipped or shifted contents);
  * a sample containing NaN/Inf -- which is what a never-written channel looks
    like, because the store is created pre-filled with NaN;
  * resume markers that were written before the store they sit beside was
    created, i.e. stale markers pointing at a NEW store, which would make a
    resumed pull skip variables that this store never received.

With --verify-content it additionally reads the real values of the timesteps
the sample spans, batch by batch, and refuses non-finite values, spatially
constant (fill) channels and values outside the band derived from the official
Stormer normalization constants. The resulting content certificate -- the
digest of the bytes read plus the identity of the slice they came from -- is
included in the JSON report and can be written on its own with
--content-certificate.

Exit codes:
    0: every admission check passed
    1: at least one admission check was violated
    2: the preflight itself could not run (bad arguments, unreadable store)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from earthdelta.pilot_contract import (  # noqa: E402
    DATA_ROLE_VALUES,
    DEFAULT_INTERVAL_HOURS,
    DEFAULT_SIGMA_BOUND,
    EXPECTED_CHANNELS,
    PilotPreflightReport,
    run_pilot_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admission preflight for a pilot data store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--store", type=Path, required=True,
        help="Path to the pilot zarr store to admit",
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Analysis time index of the sample to admit",
    )
    parser.add_argument(
        "--history-steps", type=int, default=1,
        help="History steps the sample requires before --index",
    )
    parser.add_argument(
        "--target-steps", type=int, default=1,
        help="Target steps the sample requires after --index",
    )
    parser.add_argument(
        "--interval-hours", type=int, default=DEFAULT_INTERVAL_HOURS,
        help="Expected spacing between consecutive timesteps",
    )
    parser.add_argument(
        "--no-interval-check", action="store_true",
        help="Skip the time-spacing check (index-only time coordinates)",
    )
    parser.add_argument(
        "--marker-dir", type=Path, default=None,
        help="Resume-marker directory; inferred from the store name when omitted",
    )
    parser.add_argument(
        "--expected-channels", type=int, default=EXPECTED_CHANNELS,
        help="Channel count the store must have",
    )
    parser.add_argument(
        "--no-grid-check", action="store_true",
        help=(
            "Skip comparison against the canonical Stormer target grid "
            "(coordinate values/order and the stamped identity hashes)"
        ),
    )
    parser.add_argument(
        "--marker-slack-seconds", type=int, default=0,
        help=(
            "Tolerance when comparing a marker's timestamp to the store's "
            "creation time; both are written at whole-second resolution"
        ),
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Also write the full report as JSON to this path",
    )
    parser.add_argument(
        "--verify-content", action="store_true",
        help=(
            "Also verify the real values of the timesteps this sample spans, "
            "batch by batch, and emit a content certificate"
        ),
    )
    parser.add_argument(
        "--content-indices", type=int, nargs="+", default=None,
        help=(
            "Explicit time indices to content-verify; defaults to every "
            "timestep the admitted sample spans"
        ),
    )
    parser.add_argument(
        "--content-batch-size", type=int, default=4,
        help="Timesteps per content-verification batch",
    )
    parser.add_argument(
        "--content-sigma-bound", type=float, default=DEFAULT_SIGMA_BOUND,
        help=(
            "Half-width of the physical plausibility band, in official "
            "normalization standard deviations"
        ),
    )
    parser.add_argument(
        "--normalization-dir", type=Path, default=None,
        help=(
            "Directory holding normalize_mean.npz / normalize_std.npz; "
            "defaults to the repo's official Stormer constants"
        ),
    )
    parser.add_argument(
        "--no-physical-range", action="store_true",
        help=(
            "Record that the plausibility band was NOT checked instead of "
            "checking it (finiteness and degeneracy still run)"
        ),
    )
    parser.add_argument(
        "--data-role", choices=list(DATA_ROLE_VALUES), default=None,
        help="Frozen data role to stamp on the content certificate",
    )
    parser.add_argument(
        "--content-certificate", type=Path, default=None,
        help="Write the content certificate on its own to this path",
    )
    return parser


def print_report(report: PilotPreflightReport) -> None:
    print("=" * 68, flush=True)
    print("R2 Pilot Preflight", flush=True)
    print("=" * 68, flush=True)
    print(f"Store: {report.store_path}", flush=True)
    print("", flush=True)
    print("Checks:", flush=True)
    for check in report.checks:
        status = "PASS" if check.get("passed") else "FAIL"
        code = check.get("code")
        suffix = f"  ({code})" if code else ""
        print(f"  - {check.get('check')}: {status}{suffix}", flush=True)

    if report.violations:
        print("", flush=True)
        print("Violations:", flush=True)
        for violation in report.violations:
            print(
                f"  [{violation.get('code')}] {violation.get('check')}: "
                f"{violation.get('message')}",
                flush=True,
            )

    print("", flush=True)
    print(f"RESULT: {'PASS' if report.passed else 'FAIL'}", flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.history_steps < 0 or args.target_steps < 0:
        print(
            "ERROR: --history-steps and --target-steps must be non-negative",
            file=sys.stderr, flush=True,
        )
        return 2

    try:
        report = run_pilot_preflight(
            store_path=args.store,
            index=args.index,
            history_steps=args.history_steps,
            target_steps=args.target_steps,
            interval_hours=None if args.no_interval_check else args.interval_hours,
            marker_dir=args.marker_dir,
            expected_channels=args.expected_channels,
            check_grid=not args.no_grid_check,
            marker_slack_seconds=args.marker_slack_seconds,
            verify_content=args.verify_content,
            content_indices=args.content_indices,
            content_batch_size=args.content_batch_size,
            content_sigma_bound=args.content_sigma_bound,
            normalization_dir=args.normalization_dir,
            require_physical_range=not args.no_physical_range,
            data_role=args.data_role,
        )
    except Exception as exc:  # noqa: BLE001 - preflight could not run at all
        print(
            f"ERROR: preflight could not run: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True,
        )
        return 2

    print_report(report)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"\nJSON report written to: {args.json}", flush=True)

    if args.content_certificate is not None:
        if report.content_certificate is None:
            print(
                "ERROR: --content-certificate was requested but no content "
                "certificate was produced; pass --verify-content",
                file=sys.stderr, flush=True,
            )
            return 2
        args.content_certificate.parent.mkdir(parents=True, exist_ok=True)
        args.content_certificate.write_text(
            json.dumps(report.content_certificate, indent=2, default=str)
        )
        print(
            f"Content certificate written to: {args.content_certificate}",
            flush=True,
        )

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
