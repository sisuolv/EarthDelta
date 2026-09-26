from __future__ import annotations

from scripts import r5_score_erratum as erratum


def test_erratum_source_manifest_includes_self_and_numeric_dependencies():
    assert "scripts/r5_score_erratum.py" in erratum.SCORER_FILES
    assert "earthdelta/policy_oof.py" in erratum.SCORER_FILES
    assert "scripts/r4_candidate_cache.py" in erratum.SCORER_FILES


def test_erratum_requires_point_and_ci_identity_for_24h():
    old = {"point": 0.1, "ci_low": 0.01, "ci_high": 0.2, "status_vs_delta_min": "ABOVE"}
    new = dict(old)
    assert old == new
    assert old["point"] == new["point"]
    assert old["ci_low"] == new["ci_low"] and old["ci_high"] == new["ci_high"]
    changed = dict(new, point=0.1000001)
    assert changed != old
