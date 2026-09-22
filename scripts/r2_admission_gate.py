#!/usr/bin/env python3
"""Stage-3 admission gate: what must hold BEFORE a real pilot sample is used.

Three independent things have to be true before real data trains anything, and
none of them is checked by the code that produces the data:

  B09  The calendar must be joined against reality. `build_manifest` enumerates
       issue/valid/available times from year ranges and lead hours; it never
       opens a store. A manifest over a year that was never pulled is
       indistinguishable from a manifest over a year that was. This gate
       resolves each row's history, issue and target timestamps against the
       real time coordinate of the real store and admits only rows whose three
       endpoints exist. In formal mode it refuses a placeholder
       normalization_hash and requires a declared data_role, and it stamps each
       admitted row with a deterministic issue_id and the run's independent
       process_group_id.

  B08  Content must be verified, not inferred. `is_year_complete` and
       `is_pilot_complete` check a timestep count and a channel count; resume
       markers check their own existence. This gate reads the real values of
       the admitted subset, batch by batch, and refuses non-finite values,
       spatially constant (fill) channels, and values outside the band derived
       from the official Stormer normalization constants -- then writes a
       certificate binding the digest of those bytes to the slice they came
       from.

  B15  The candidate registry must be formal. A planner may answer an empty
       candidate table with a no-op plan; a registry may not, because then "no
       candidate won" and "no candidate exists" look the same afterwards. This
       gate refuses an empty registry, a registry with no explicit no-edit
       entry, and any complex / non-finite / non-real coefficient, and checks
       that single-expert plans realized the a0 they were specified at.

The gate NEVER downloads anything. It opens existing stores read-only.

Examples:

    python scripts/r2_admission_gate.py --years 2020 --limit 8 --verify-content
    python scripts/r2_admission_gate.py --years 2018 2019 --formal \\
        --data-role bank_fit --normalization-hash <digest> \\
        --registry artifacts/pilot_registry.json --json artifacts/admission.json

Exit codes:
    0: every admission check passed
    1: at least one admission check was violated
    2: the gate itself could not run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from earthdelta.data.make_splits import (  # noqa: E402
    SplitManifest,
    admit_manifest_rows,
    build_manifest,
    new_process_group_id,
)
from earthdelta.pilot_contract import (  # noqa: E402
    DATA_ROLE_VALUES,
    DEFAULT_SIGMA_BOUND,
    PilotContractViolation,
    verify_content_subset,
)
from earthdelta.registry import (  # noqa: E402
    CandidateRegistry,
    RegistryViolation,
    build_pilot_registry,
)

DEFAULT_DATA_ROOT = SOURCE_ROOT / "data" / "era5_1p40625"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-3 admission gate for real pilot samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory holding the per-year zarr stores")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Years to build a manifest for")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Existing manifest JSON to admit instead of building one")
    parser.add_argument("--lead-hours", type=int, nargs="+", default=[6],
                        help="Forecast leads to admit")
    parser.add_argument("--limit", type=int, default=8,
                        help="Maximum manifest rows to admit")
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every Nth manifest row, to spread the subset out")
    parser.add_argument("--history-steps", type=int, default=1,
                        help="History steps each sample needs before its issue time")
    parser.add_argument("--interval-hours", type=int, default=6,
                        help="Spacing between consecutive timesteps")
    parser.add_argument("--expected-channels", type=int, default=69,
                        help="Channel count each store must carry")
    parser.add_argument("--formal", action="store_true",
                        help="Formal admission: refuse placeholder normalization "
                             "hashes and require a declared data role")
    parser.add_argument("--data-role", choices=list(DATA_ROLE_VALUES), default=None,
                        help="Frozen role the admitted samples are for")
    parser.add_argument("--normalization-hash", type=str, default=None,
                        help="Real normalization hash to stamp on a built manifest")
    parser.add_argument("--verify-content", action="store_true",
                        help="Content-verify the admitted samples")
    parser.add_argument("--content-samples", type=int, default=2,
                        help="How many admitted samples to content-verify")
    parser.add_argument("--content-batch-size", type=int, default=2,
                        help="Timesteps per content-verification batch")
    parser.add_argument("--content-sigma-bound", type=float, default=DEFAULT_SIGMA_BOUND,
                        help="Plausibility band half-width, in official sigmas")
    parser.add_argument("--normalization-dir", type=Path, default=None,
                        help="Directory of the official normalization constants")
    parser.add_argument("--no-physical-range", action="store_true",
                        help="Record that the plausibility band was NOT checked")
    parser.add_argument("--registry", type=Path, default=None,
                        help="Candidate registry JSON to validate")
    parser.add_argument("--build-pilot-registry", type=int, default=None,
                        metavar="NUM_EXPERTS",
                        help="Build and validate a pilot registry of this size")
    parser.add_argument("--registry-out", type=Path, default=None,
                        help="Write the validated registry to this path")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the full admission record as JSON here")
    return parser


def _admit(args, record: dict) -> bool:
    if args.manifest is not None:
        manifest = SplitManifest.from_json(args.manifest)
        rows = list(manifest.rows)
        record["manifest_source"] = str(args.manifest)
    else:
        manifest = build_manifest(
            years=args.years,
            lead_hours=args.lead_hours,
            normalization_hash=args.normalization_hash,
            formal=args.formal,
        )
        rows = list(manifest.rows)
        record["manifest_source"] = f"build_manifest(years={args.years})"

    if args.stride > 1:
        rows = rows[:: args.stride]
    record["manifest_rows_total"] = len(manifest.rows)
    record["manifest_rows_considered"] = min(len(rows), args.limit)

    report = admit_manifest_rows(
        rows,
        args.data_root,
        formal=args.formal,
        data_role=args.data_role,
        limit=args.limit,
        history_steps=args.history_steps,
        interval_hours=args.interval_hours,
        expected_channels=args.expected_channels,
    )
    record["admission"] = report.to_dict()
    record["process_group_id"] = report.process_group_id

    print(f"  rows requested : {report.requested}", flush=True)
    print(f"  admitted       : {len(report.admitted)}", flush=True)
    print(f"  rejected       : {len(report.rejected)}", flush=True)
    for code, count in sorted(report.rejection_codes().items()):
        print(f"    {code}: {count}", flush=True)
    if report.rejected:
        print(f"    first: {report.rejected[0]['message'][:160]}", flush=True)
    return report.passed


def _verify_content(args, record: dict) -> bool:
    admitted = record.get("admission", {}).get("admitted", [])
    if not admitted:
        print("  no admitted samples to content-verify", flush=True)
        return False

    certificates = []
    passed = True
    for result in admitted[: max(1, args.content_samples)]:
        indices = list(range(result["history_index"], result["target_index"] + 1))
        try:
            certificate = verify_content_subset(
                store_path=Path(result["issue_store"]),
                indices=indices,
                batch_size=args.content_batch_size,
                expected_channels=args.expected_channels,
                sigma_bound=args.content_sigma_bound,
                normalization_dir=args.normalization_dir,
                require_physical_range=not args.no_physical_range,
                data_role=result.get("data_role"),
                issue_id=result["issue_id"],
                process_group_id=result["process_group_id"],
                strict=True,
            )
            certificates.append(certificate.to_dict())
            print(
                f"  {result['event_id']}: {certificate.n_batches} batch(es), "
                f"{certificate.n_values} values, "
                f"content_sha256={certificate.content_sha256[:16]}...",
                flush=True,
            )
        except PilotContractViolation as exc:
            passed = False
            embedded = exc.detail.get("certificate")
            if isinstance(embedded, dict):
                certificates.append(embedded)
            print(f"  {result['event_id']}: FAIL [{exc.code}] {exc.message[:160]}",
                  flush=True)
    record["content_certificates"] = certificates
    return passed


def _validate_registry(args, record: dict) -> bool:
    try:
        if args.registry is not None:
            registry = CandidateRegistry.from_json(args.registry)
            record["registry_source"] = str(args.registry)
        else:
            registry = build_pilot_registry(
                int(args.build_pilot_registry),
                data_role=args.data_role,
            )
            record["registry_source"] = (
                f"build_pilot_registry(num_experts={args.build_pilot_registry})"
            )
        summary = registry.validate()
    except RegistryViolation as exc:
        record["registry"] = {"passed": False, "code": exc.code, "message": exc.message}
        print(f"  FAIL [{exc.code}] {exc.message[:200]}", flush=True)
        return False

    record["registry"] = {"passed": True, "summary": summary,
                          "payload": registry.to_dict()}
    print(f"  registry       : {summary['registry']} "
          f"({summary['n_entries']} entries, reference="
          f"{summary['reference_plan_id']})", flush=True)
    for entry in registry.entries:
        print(f"    {entry.plan_id}: {list(entry.coefficients)}", flush=True)
    if args.registry_out is not None:
        registry.to_json(args.registry_out)
        print(f"  registry written to {args.registry_out}", flush=True)
    return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.manifest is None and not args.years:
        print("ERROR: pass --years or --manifest", file=sys.stderr, flush=True)
        return 2
    if args.formal and args.data_role is None:
        print(
            "ERROR: --formal requires --data-role "
            f"({'/'.join(DATA_ROLE_VALUES)})",
            file=sys.stderr, flush=True,
        )
        return 2
    if args.limit <= 0 or args.stride <= 0:
        print("ERROR: --limit and --stride must be positive",
              file=sys.stderr, flush=True)
        return 2

    record = {
        "gate": "r2_admission_gate",
        "data_root": str(args.data_root),
        "formal": bool(args.formal),
        "data_role": args.data_role,
        "process_group_id": new_process_group_id(label="r2_admission_gate"),
    }
    results = {}

    print("=" * 68, flush=True)
    print("R2 Stage-3 Admission Gate", flush=True)
    print("=" * 68, flush=True)

    print("\n[B09] real time-coordinate index join", flush=True)
    try:
        results["b09_real_sample_admission"] = _admit(args, record)
    except (PilotContractViolation, ValueError, OSError) as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        record["error"] = f"{type(exc).__name__}: {exc}"
        results["b09_real_sample_admission"] = False

    if args.verify_content:
        print("\n[B08] content verification of the admitted subset", flush=True)
        results["b08_content_verification"] = _verify_content(args, record)

    if args.registry is not None or args.build_pilot_registry is not None:
        print("\n[B15] formal candidate registry", flush=True)
        results["b15_candidate_registry"] = _validate_registry(args, record)

    record["results"] = results
    record["passed"] = bool(results) and all(results.values())

    print("", flush=True)
    for name, ok in results.items():
        print(f"  - {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"\nRESULT: {'PASS' if record['passed'] else 'FAIL'}", flush=True)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2, default=str))
        print(f"Admission record written to: {args.json}", flush=True)

    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
