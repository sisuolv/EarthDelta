#!/usr/bin/env python3
"""FP-05a: structural check of protocol/fp05_protocol.DRAFT.json with the review package v2 tool.

Two views, both reported verbatim:
 (a) LITERAL: copy the v2 package to a temp dir, replace evidence/FP05_PROTOCOL_TEMPLATE.json
     with the draft, run v2 tools/validate_package.py unmodified. (Its manifest-hash check
     necessarily fails for the replaced file; its protocol_not_falsely_frozen check requires
     status == 'TO_BE_RUN', while the dispatch mandates status 'DRAFT_NOT_FROZEN'.)
 (b) PROTOCOL-ONLY: the four protocol predicates of v2 validate_package.py, evaluated on the
     draft, plus the same predicate with the status label generalised to {TO_BE_RUN,
     DRAFT_NOT_FROZEN}; plus FP-05a-specific checks (confirm null, no numeric undecided field).
Output (refuses to overwrite): entry/protocol_draft_structural_check.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
V2 = REPO / "plans/plans_v5_0924/extracted_v2/EarthDelta_Codex_Followup_Plan_20260924"


def main() -> int:
    out = RUN / "entry/protocol_draft_structural_check.json"
    if out.exists():
        print("refusing to overwrite", out)
        return 2
    draft_p = RUN / "protocol/fp05_protocol.DRAFT.json"
    proto = json.loads(draft_p.read_text())

    # (a) literal run of the unmodified v2 validator on a temp copy
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "pkg"
        shutil.copytree(V2, pkg)
        shutil.copy(draft_p, pkg / "evidence/FP05_PROTOCOL_TEMPLATE.json")
        p = subprocess.run([sys.executable, str(pkg / "tools/validate_package.py"), "--package", str(pkg)],
                           capture_output=True, text=True, timeout=120)
        try:
            literal = json.loads(p.stdout)
        except ValueError:
            literal = {"raw_stdout": p.stdout, "stderr": p.stderr}
        literal["returncode"] = p.returncode

    # (b) the protocol predicates, copied from v2 validate_package.py
    pred = {
        "confirm_not_h2_clean": proto["split"]["confirm"]["status"] == "UNASSIGNED_NO_ACCESS"
        and "EXPOSED" in proto["split"]["confirm"]["rejected_reservation"],
        "five_candidates": proto["candidates"]["count"] == 5 and proto["candidates"]["background_F0_not_a_candidate"],
        "protocol_not_falsely_frozen_LITERAL": proto["status"] == "TO_BE_RUN" and proto["execution_ready"] is False
        and proto["final_protocol_sha256"] is None,
        "protocol_not_falsely_frozen_GENERALISED": proto["status"] in ("TO_BE_RUN", "DRAFT_NOT_FROZEN")
        and proto["execution_ready"] is False and proto["final_protocol_sha256"] is None,
        "safety_margin_unbound_visible": proto["decision"]["max_harm72_vs_Fs"] is None,
    }
    # FP-05a specific
    numeric_undecided = []

    def walk(o, path):
        if isinstance(o, dict):
            if o.get("status") == "PROPOSED_NOT_FROZEN" and "value" in o and o["value"] is not None:
                numeric_undecided.append(path)
            for k, v in o.items():
                if k != "proposals_for_reference":
                    walk(v, f"{path}.{k}")

    walk(proto, "")
    fp05a = {
        "status_is_DRAFT_NOT_FROZEN": proto["status"] == "DRAFT_NOT_FROZEN",
        "final_protocol_sha256_null": proto["final_protocol_sha256"] is None,
        "confirm_null": proto["data_roles"]["confirm"] is None and proto["split"]["confirm_window"] is None
        and proto["split"]["confirm"]["window"] is None,
        "confirm_status_UNASSIGNED_NO_ACCESS": proto["data_roles"]["confirm_status"] == "UNASSIGNED_NO_ACCESS",
        "2020H2_EXPOSED_NOT_CONFIRM": proto["data_roles"]["2020H2"]["label"] == "EXPOSED_NOT_CONFIRM",
        "no_undecided_field_carries_a_value": not numeric_undecided,
        "block_resample_fold_gpu_mde_all_null": all(x is None for x in (
            proto["inference"]["block_days"]["value"], proto["inference"]["bootstrap_draws"]["value"],
            proto["validation"]["inner_folds"]["value"], proto["validation"]["outer_folds"]["value"],
            proto["resource"]["max_gpu_jobs_total"], proto["inference"]["MDE"])),
        "gpu_not_authorized": proto["authorization"]["gpu_jobs_authorized"] is False,
    }
    res = {
        "schema": "ed-fp05a-protocol-draft-check/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "draft": {"path": str(draft_p.relative_to(REPO)), "sha256": hashlib.sha256(draft_p.read_bytes()).hexdigest()},
        "literal_v2_validate_package_on_substituted_copy": literal,
        "literal_expected_failures_explained": {
            "manifest_hashes_match": "the substituted template file no longer matches PACKAGE_MANIFEST.json (by construction)",
            "protocol_not_falsely_frozen": "v2 hard-codes status=='TO_BE_RUN'; the dispatch mandates 'DRAFT_NOT_FROZEN'. execution_ready=false and final_protocol_sha256=null both hold.",
        },
        "protocol_predicates": pred,
        "fp05a_checks": fp05a,
        "undecided_fields_with_values": numeric_undecided,
        "structural_pass": all(v for k, v in pred.items() if not k.endswith("_LITERAL")) and all(fp05a.values()),
    }
    lit_checks = literal.get("checks", {})
    res["literal_failed_checks"] = sorted(k for k, v in lit_checks.items() if v is False)
    res["literal_only_expected_failures"] = set(res["literal_failed_checks"]) <= {"manifest_hashes_match", "protocol_not_falsely_frozen"}
    with out.open("x") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps({k: res[k] for k in ("protocol_predicates", "fp05a_checks", "literal_failed_checks",
                                          "literal_only_expected_failures", "structural_pass")}, indent=1))
    return 0 if res["structural_pass"] and res["literal_only_expected_failures"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
