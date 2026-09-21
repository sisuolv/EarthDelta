"""CPU-only contract checks supporting the second-round audit."""
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from earthdelta.data.make_splits import build_manifest
from earthdelta.selection import plan_from_prediction
from earthdelta.bridge import DEFAULT_VARIABLES, NormalizationContract

result = {}
b = torch.ones(2, dtype=torch.float64)
h = torch.eye(2, dtype=torch.float64)
checks = {}
for name, candidates, options in [
    ("nonfinite", torch.tensor([[float("nan"), 0.0]]), {}),
    ("max_candidates", torch.zeros(3, 2), {"max_candidates": 2}),
    ("bound", torch.tensor([[0.5, 0.0]], dtype=torch.float64), {"bound": 0.25}),
    ("max_active", torch.tensor([[0.1, 0.1]], dtype=torch.float64), {"max_active": 1}),
]:
    try:
        p = plan_from_prediction(b, h, candidate_offsets=candidates, **options)
        checks[name] = {"support": p.support, "solver_failures": p.solver_failures}
    except ValueError as exc:
        checks[name] = {"rejected": str(exc)}
result["A03_four_requested_checks"] = checks
with warnings.catch_warnings(record=True) as caught:
    p = plan_from_prediction(b, h, candidate_offsets=torch.tensor([[0.1+100j, 0j]]))
    result["A03_complex_cast"] = {
        "support": p.support, "selected_coefficients": p.coefficients.tolist(),
        "warnings": [str(w.message) for w in caught],
    }

old_tz = os.environ.get("TZ")
stamps = {}
try:
    for tz in ["UTC", "Pacific/Honolulu", "Asia/Shanghai"]:
        os.environ["TZ"] = tz
        time.tzset()
        rows = build_manifest([2015], lead_hours=[6]).rows
        stamps[tz] = [rows[0].issue_time, rows[0].valid_time, rows[0].available_time,
                      rows[-1].issue_time]
finally:
    if old_tz is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old_tz
    time.tzset()
result["A06_timezone_invariance"] = {"timestamps": stamps,
    "all_equal": len({tuple(v) for v in stamps.values()}) == 1}

norm = NormalizationContract.from_npz_dir(
    str(ROOT / "reference/stormer/normalization_constants"), DEFAULT_VARIABLES)
result["B01_nonzero_diff_means"] = {
    str(interval): {"nonzero_channels": int((mean != 0).sum()),
        "max_abs_raw": float(mean.abs().max()),
        "max_abs_in_input_normalized_units": float((mean / norm.inp_std).abs().max())}
    for interval, mean in norm.diff_mean.items()
}

torch.manual_seed(123)
x = torch.randn(1000, 69)
before, after = x.mean(-1), (x * (torch.ones(69) / 69)).sum(-1)
result["uniform_weight_bit_identity"] = {
    "bit_equal": torch.equal(before, after),
    "max_abs_diff": float((before-after).abs().max()),
    "interpretation": "rounding only; mathematical equality still holds",
}
print(json.dumps(result, indent=2, allow_nan=False))
