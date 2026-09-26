#!/usr/bin/env python3
"""Contract-driven EarthDelta v8 phase runner.

The runner is intentionally conservative: missing scientific inputs produce a
BLOCKED receipt and stop.  It never fabricates model output to advance a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
R4_ROOT = Path(os.environ.get("EARTHDELTA_V8_R4_ROOT", "/mnt/afs/260010168/earthdelta_v8_planning_20260926T182037Z_r4"))
APPROVALS = Path("/mnt/afs/260010168/earthdelta_v8_approvals/approvals.json")
APPROVALS_SHA = APPROVALS.with_name("approvals.sha256")
PROMPT_SHA = "a9ac5e9c7c5cd06b43c1ccb290f86b349a7acbfa2eaafd61b2f7c20fd5bb358c"
AMEND_SHA = "36d1342bd5f147be945253dd1a40fb252e1ed24ac22d424dd86c7e4496918286"
R3_MANIFEST_SHA = "020797ecb13129e111cd02fdd31c311f9b469cb840a1489af4a74fb5267b247f"

from earthdelta.v8.approvals import ApprovalError, ApprovalVerifier
from earthdelta.v8.data_repair import DataRepairError, refetch_zarr_indices, repair_zarr_year
from earthdelta.v8.io import append_jsonl, atomic_json, json_hash, read_json, replace_json, sha256_file
from earthdelta.v8.precision_precheck import formal_legacy_precheck
from earthdelta.v8.receipt_registry import ReceiptError, ReceiptRegistry
from earthdelta.v8.support_contract import make_certificates, validate_issue_list


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def commit_sha() -> str:
    try:
        env = os.environ.copy()
        safe_config = R4_ROOT / "git-safe-directory.config"
        if safe_config.is_file():
            env["GIT_CONFIG_GLOBAL"] = str(safe_config)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, env=env, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


class Autorun:
    def __init__(self, run: Path, *, dry_run: bool = False, approvals: Path = APPROVALS,
                 approvals_sha: Path = APPROVALS_SHA):
        self.run = run
        self.dry_run = dry_run
        self.approvals = approvals
        self.approvals_sha = approvals_sha
        self.run.mkdir(parents=True, exist_ok=True)
        self.registry = ReceiptRegistry(R4_ROOT)
        self.stage_data = {s["stage_id"]: s for s in read_json(R4_ROOT / "AUTORUN_R4.json")["stages"]}
        self.status_path = self.run / "STATUS.json"
        self.access_path = self.run / "ACCESS_LEDGER.jsonl"
        if not self.status_path.exists():
            self._status(stage=None, state="TO_BE_RUN", reason="initialized", next_stage="T01a_contract")

    def _status(self, *, stage: str | None, state: str, reason: str, next_stage: str | None,
                science_decision: str | None = None, card_hours: dict | None = None) -> None:
        value = {"schema_version": "earthdelta.v8.status.v1", "updated_utc": now(), "stage": stage,
                 "state": state, "reason": reason, "next_stage": next_stage,
                 "science_decision": science_decision, "commit": commit_sha(),
                 "card_hours": card_hours or {"H100": 0.0, "spot_5090": 0.0}}
        replace_json(self.status_path, value)

    def ledger(self, operation: str, **kwargs: Any) -> None:
        append_jsonl(self.access_path, {"time": now(), "operation": operation, **kwargs})

    def verifier(self) -> ApprovalVerifier:
        return ApprovalVerifier(self.approvals, self.approvals_sha, expected_prompt_sha=PROMPT_SHA,
                                expected_amendments_sha=AMEND_SHA, expected_r3_manifest_sha=R3_MANIFEST_SHA)

    def require_stage(self, stage_id: str, available: set[str]) -> tuple[dict, ApprovalVerifier]:
        stage = self.stage_data[stage_id]
        verifier = self.verifier()
        verifier.require(stage.get("required_approval_scopes", []), available_receipts=available)
        for rid in stage.get("depends_on_receipts", []):
            self.registry.load(rid, self.run)
        return stage, verifier

    def receipt(self, name: str) -> dict | None:
        spec = self.registry.spec(name)
        path = self.run / spec.files[0]
        if not path.is_file():
            return None
        try:
            return read_json(path)
        except Exception:
            return None

    def write_receipt(self, filename: str, value: dict) -> None:
        atomic_json(self.run / filename, value)

    def completed_receipts(self) -> set[str]:
        completed = {rid for rid in self.registry.specs if self.registry.observed(rid, self.run)}
        # Phase A is immutable and lives in the contract directory rather than the run.
        acceptance = R4_ROOT / "r4_acceptance_receipt.json"
        if acceptance.is_file():
            try:
                data = read_json(acceptance)
                if data.get("status") == "OBSERVED" and all(data.get("checks", {}).values()):
                    completed.add("r4_acceptance_receipt")
            except Exception:
                pass
        return completed

    def run_stage(self, stage_id: str) -> dict:
        stage, _ = self.require_stage(stage_id, self.completed_receipts())
        self._status(stage=stage_id, state="TO_BE_RUN", reason="stage started", next_stage=None)
        if self.dry_run:
            result = {"status": "OBSERVED", "stage": stage_id, "dry_run": True, "commit": commit_sha()}
            self.ledger("dry_run_stage", stage=stage_id)
            return result
        handler = getattr(self, f"stage_{stage_id}", None)
        if handler is None:
            return self._blocked(stage_id, "not_implemented")
        try:
            result = handler(stage)
        except (ApprovalError, ReceiptError, FileNotFoundError, ValueError, RuntimeError) as exc:
            return self._blocked(stage_id, f"{type(exc).__name__}:{exc}")
        if result.get("status") == "OBSERVED":
            self._status(stage=stage_id, state="OBSERVED", reason="release gate passed", next_stage=None)
        else:
            self._status(stage=stage_id, state=result.get("status", "BLOCKED"), reason=result.get("reason", "stage failed"), next_stage=None)
        return result

    def _blocked(self, stage_id: str, reason: str, *, receipt_value: dict | None = None) -> dict:
        self._status(stage=stage_id, state="BLOCKED", reason=reason, next_stage=None)
        stage = self.stage_data[stage_id]
        rid = stage.get("output_receipt", "").split("+")[0]
        if rid in self.registry.specs:
            spec = self.registry.spec(rid)
            if receipt_value is None:
                receipt_value = {}
            value = dict(receipt_value)
            value.update({"status": "BLOCKED", "stage": stage_id, "reason": reason,
                          "commit": commit_sha(), "updated_utc": now()})
            self.write_receipt(spec.files[0], value)
        return {"status": "BLOCKED", "stage": stage_id, "reason": reason}

    def stage_T01a_contract(self, stage: dict) -> dict:
        acceptance = read_json(R4_ROOT / "r4_acceptance_receipt.json")
        if acceptance.get("status") != "OBSERVED" or not all(acceptance.get("checks", {}).values()):
            return self._blocked("T01a_contract", "r4_acceptance_not_observed")
        expected = ["ROLES_R4.json", "METRIC_CONTRACT_R4.json", "RECIPES_R4.json", "TASKS_R4.json",
                    "RECEIPT_REGISTRY.json", "AUTORUN_R4.json", "APPROVALS_R4.json", "CODE_MAP_R4.json"]
        hashes = {name: sha256_file(R4_ROOT / name) for name in expected}
        value = {"status": "OBSERVED", "contract_hashes": hashes,
                 "role_hashes": {"roles": hashes["ROLES_R4.json"]},
                 "metric_hash": hashes["METRIC_CONTRACT_R4.json"],
                 "registry_hash": hashes["RECEIPT_REGISTRY.json"], "commit": commit_sha()}
        self.write_receipt("contract_receipt.json", value)
        self.ledger("contract_read", files=expected, array_payload=False)
        return value

    def stage_T01a_precision_precheck(self, stage: dict) -> dict:
        candidates = [Path("/mnt/afs/260010168/earthdelta_independent_review_20260926/per_channel_mse.npz"),
                      REPO / "artifacts/round2_cci/ed-r4cachecj2-0924125702-926f58/per_channel_mse.npz"]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            return self._blocked("T01a_precision_precheck", "legacy_endpoint_artifact_missing")
        self.ledger("legacy_endpoint_read", path=str(source), array_payload=True, year=2019)
        try:
            metadata_root = Path("/mnt/afs/260010168/EarthDelta/artifacts/round2_cci/ed-r4cachecj2-0924125702-926f58")
            std_candidates = sorted(REPO.glob("artifacts/round2_cci/*/source/reference/stormer/normalization_constants/normalize_std.npz"))
            if not std_candidates:
                return self._blocked("T01a_precision_precheck", "signed_normalize_std_missing")
            value = formal_legacy_precheck(source, issue_metadata_root=metadata_root,
                                           normalize_std_path=std_candidates[0], draws=10000, seed=0)
            self.write_receipt("precision_precheck.json", value)
            return value
        except Exception as exc:
            return self._blocked("T01a_precision_precheck", f"invalid_legacy_endpoint:{exc}")

    def stage_T01a_data_repair(self, stage: dict) -> dict:
        import numpy as np
        from datetime import datetime, timedelta, timezone
        roles = read_json(R4_ROOT / "ROLES_R4.json")["roles"]
        by_year: dict[int, set[int]] = {2019: set(), 2020: set()}
        for role in ("fit", "fit_validation"):
            for value in roles[role]["issue_times_utc"]:
                issue = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
                year_start = datetime(issue.year, 1, 1, tzinfo=timezone.utc)
                first = int((issue - timedelta(hours=12) - year_start).total_seconds() // 21600)
                last = int((issue + timedelta(hours=120) - year_start).total_seconds() // 21600)
                if issue.year in by_year:
                    by_year[issue.year].update(range(first, last + 1))
        target_root = Path(os.environ.get("EARTHDELTA_V8_DATA_ROOT", str(REPO / "data/era5_1p40625_v8")))
        target_root.mkdir(parents=True, exist_ok=True)
        results, all_failures, markers, all_access = [], [], [], []
        setup_failures = []
        failed_indices_by_year: dict[int, list[int]] = {}
        std_candidates = sorted(REPO.glob("artifacts/round2_cci/*/source/reference/stormer/normalization_constants/normalize_std.npz"))
        std = None
        if std_candidates:
            with np.load(std_candidates[0], allow_pickle=True) as data:
                channels = [str(x) for x in data.files]
                std = np.asarray([float(data[x][0]) for x in channels[-69:]], dtype=np.float64) if len(channels) >= 69 else None
        for year, indices in by_year.items():
            source_root = Path(os.environ.get("EARTHDELTA_V8_SOURCE_ROOT", str(REPO / "data/era5_1p40625")))
            source = source_root / f"{year}.zarr"
            target = target_root / f"{year}.zarr"
            try:
                result = repair_zarr_year(source, target, role="fit_and_validation", year=year,
                                          indices=indices, std=None)
            except (DataRepairError, OSError) as exc:
                setup_failures.append(f"{year}:setup:{type(exc).__name__}:{exc}")
                continue
            results.append(result.as_dict()); all_access.extend(result.access_ledger)
            if result.completion_marker:
                markers.append(result.completion_marker)
            failed_indices_by_year[year] = [int(item.split(":", 2)[1]) for item in result.failure_list
                                            if item.startswith(f"{year}:") and item.split(":", 2)[1].isdigit()]
        # One predeclared upstream retry is allowed for failed source chunks.
        # It is still bounded to the signed indices and never touches 2021/2022.
        refetch_results = []
        # Initial source failures are provisional: a single signed upstream
        # retry may repair them.  Only post-retry failures determine the gate.
        all_failures = list(setup_failures)
        for year, bad_indices in failed_indices_by_year.items():
            if not bad_indices:
                continue
            target = target_root / f"{year}.zarr"
            try:
                retry = refetch_zarr_indices(target, year=year, indices=bad_indices)
            except Exception as exc:
                retry = {"status": "BLOCKED", "refetch_count": 0,
                         "failure_list": [f"{year}:refetch:{type(exc).__name__}:{exc}"], "indices": bad_indices}
            refetch_results.append({"year": year, **retry})
            all_failures.extend([f"{year}:refetch:{item}" for item in retry.get("failure_list", [])])
            all_access.append({"year": year, "operation": "UPSTREAM_REFETCH", "indices": bad_indices,
                               "array_payload": True, "attempts": 2})
        # Recheck every signed chunk after the one retry.  A completed marker is
        # written only when all selected source/remap chunks are finite.
        for year, indices in by_year.items():
            target = target_root / f"{year}.zarr"
            if not target.exists():
                continue
            try:
                import zarr
                array = zarr.open_group(str(target), mode="r")["data"]
                remaining = [i for i in indices if not np.isfinite(np.asarray(array[i])).all()]
                if not remaining:
                    marker = target / "COMPLETE.json"
                    if not marker.exists():
                        from earthdelta.v8.io import atomic_json
                        atomic_json(marker, {"status": "OBSERVED", "role": "fit_and_validation", "year": year,
                                             "indices": sorted(indices), "ledger_hash": json_hash(all_access)})
                    markers.append(str(marker))
                else:
                    all_failures.extend([f"{year}:post_refetch:{i}" for i in remaining])
            except Exception as exc:
                all_failures.append(f"{year}:post_refetch_check:{type(exc).__name__}:{exc}")
        value = {"status": "OBSERVED" if not all_failures else "BLOCKED",
                 "source_chunk_manifest": [m for result in results for m in result["source_chunk_manifest"]],
                 "refetch_count": sum(int(result["refetch_count"]) for result in results),
                 "failure_list": all_failures, "completion_marker": markers,
                 "access_ledger": all_access, "access_ledger_hash": json_hash(all_access),
                 "results_by_year": results, "refetch_results": refetch_results}
        if all_failures:
            return self._blocked("T01a_data_repair", "source_or_remap_nonfinite", receipt_value=value)
        self.write_receipt("data_repair_receipt.json", value)
        return value

    def stage_T01a_support_certificates(self, stage: dict) -> dict:
        roles = read_json(R4_ROOT / "ROLES_R4.json")["roles"]
        certs, hashes = [], {}
        for role in ("fit", "fit_validation"):
            raw = roles[role]
            times = raw["issue_times_utc"]
            role_year = 2019 if role == "fit" else 2020
            # The signed fit role spans 2019 through 2020 Q3; validation is Q4 2020.
            signed_year = None if role == "fit" else role_year
            normalized = validate_issue_list(role, times, role_year=signed_year)
            certificates = make_certificates(role, normalized, "era5_1p40625_v8", role_year=signed_year)
            certs.extend([x.as_dict() for x in certificates])
            hashes[role] = json_hash([x.as_dict() for x in certificates])
        value = {"status": "OBSERVED", "role_hashes": hashes, "support_hashes": hashes,
                 "replacement_ledger_hash": json_hash([]), "certificates": certs}
        self.write_receipt("support_certificates.json", value)
        return value

    def stage_T01_2021(self, stage: dict) -> dict:
        roles = read_json(R4_ROOT / "ROLES_R4.json")["roles"]
        times = roles["dev"]["issue_times_utc"]
        if len(times) != 192:
            return self._blocked("T01_2021", "dev_count_mismatch")
        # Existence/metadata certification is separate from opening any array payload.
        self.ledger("dev_support_certification", year=2021, n=len(times), array_payload=False)
        value = {"status": "OBSERVED", "issue_list_hash": roles["dev"]["issue_time_list_sha256"],
                 "support_hash": json_hash(times), "access_ledger_hash": json_hash([{"year": 2021, "array_payload": False}]), "n": 192}
        self.write_receipt("dev_qc_receipt.json", value)
        return value

    def stage_T02_gate(self, stage: dict) -> dict:
        return self._blocked("T02_gate", "gpu_gate_requires_acp_job")

    def stage_E5090_qualification(self, stage: dict) -> dict:
        value = {"status": "OBSERVED", "engine_id": "spot_5090", "qualified": False,
                 "probe_metrics": {}, "thresholds": {"rms_ratio": 1e-3, "l6_relative": 1e-5, "paired_relative": .01},
                 "reason": "not_submitted_without_H100_reference"}
        self.write_receipt("E5090_receipt.json", value)
        self.ledger("engine_qualification_skipped", engine="spot_5090", array_payload=False)
        return value

    def stage_T02_baselines(self, stage: dict) -> dict:
        return self._blocked("T02_baselines", "gpu_baseline_requires_acp_job")

    def stage_T03_direction_freeze(self, stage: dict) -> dict:
        return self._blocked("T03_direction_freeze", "upstream_baseline_missing")

    def stage_T03_Bfam(self, stage: dict) -> dict:
        return self._blocked("T03_Bfam", "upstream_direction_missing")

    def stage_T03_headroom(self, stage: dict) -> dict:
        return self._blocked("T03_headroom", "upstream_B_fam_missing")

    def stage_T04_train(self, stage: dict) -> dict:
        return self._blocked("T04_train", "research_gate_not_GO")

    def stage_T04_final_freeze(self, stage: dict) -> dict:
        return self._blocked("T04_final_freeze", "training_missing")

    def stage_T04_confirm_2022(self, stage: dict) -> dict:
        return self._blocked("T04_confirm_2022", "deployment_freeze_missing")

    def run_until(self, until: str) -> dict:
        stages = list(read_json(R4_ROOT / "AUTORUN_R4.json")["stages"])
        ids = [s["stage_id"] for s in stages]
        if until not in ids:
            raise ValueError(f"unknown stage: {until}")
        for sid in ids:
            if self.registry.observed(self.registry.spec(sid if sid in self.registry.specs else "contract_receipt").receipt_id, self.run):
                # Stage ids and receipt ids are not identical; don't infer completion here.
                pass
            result = self.run_stage(sid)
            if result.get("status") != "OBSERVED":
                self._write_final_if_terminal(sid, result)
                return result
            if sid == until:
                self._write_final_if_terminal(sid, result)
                return result
        result = {"status": "OBSERVED", "stage": ids[-1], "reason": "all stages completed"}
        self._write_final_if_terminal(ids[-1], result)
        return result

    def _write_final_if_terminal(self, stage: str, result: dict) -> None:
        if result.get("status") == "BLOCKED" or stage == self.stage_data.keys().__iter__().__next__():
            # A report is useful on both stop and completion; it is never used as a gate.
            report = self.run / "FINAL_REPORT.md"
            if not report.exists():
                report.write_text("# EarthDelta v8 执行报告\n\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("autorun")
    p.add_argument("--until", default="T04_confirm_2022")
    p.add_argument("--run", required=True)
    p.add_argument("--approvals", default=str(APPROVALS))
    p.add_argument("--approvals-sha256", default=str(APPROVALS_SHA))
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "autorun":
        try:
            result = Autorun(Path(args.run), dry_run=args.dry_run, approvals=Path(args.approvals),
                             approvals_sha=Path(args.approvals_sha256)).run_until(args.until)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0 if result.get("status") == "OBSERVED" else 2
        except Exception as exc:
            print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False))
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
