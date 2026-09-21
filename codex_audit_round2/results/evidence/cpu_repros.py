"""Small CPU-only audit reproductions; never load real checkpoints or run xformers.

Run from the repo root with its CPU dependencies installed. JSON on stdout records
observations, not weather performance. Synthetic artifacts stay in TemporaryDirectory.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
torch.set_num_threads(1)
torch.manual_seed(42)

from earthdelta.metrics_contract import (
    WeightConvention, quadratic_gain, quadratic_gain_from_benefit_gram, weighted_mse,
)
from earthdelta.data.make_splits import build_manifest, SplitManifestRow
from earthdelta.selection import plan_from_prediction

observations = {}

# The old heads.py ellipsis contraction supports this documented combination.
b = torch.ones(2, 2, dtype=torch.float64)
h = torch.eye(2, dtype=torch.float64).expand(2, 2, 2)
a = torch.ones(2, dtype=torch.float64)
old = 2 * (b * a).sum(-1) - torch.einsum("...i,...ij,...j->...", a, h, a)
try:
    new = quadratic_gain_from_benefit_gram(b, h, a)
    observations["B04_batched_gram"] = {"unexpected_success": new.tolist()}
except (ValueError, RuntimeError) as exc:
    observations["B04_batched_gram"] = {"old_output": old.tolist(), "new_error": str(exc)}

e = torch.ones(1, 2, 1, 2)
u = e[:, None]
w = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])
g = quadratic_gain(e, u, w, convention=WeightConvention.WEIGHTED_MEAN)
observations["B05_weights"] = {
    "finite_input_returns_nonfinite_gain": not bool(torch.isfinite(g).all()),
    "negative_weight_mse": float(weighted_mse(
        torch.tensor([1.0, 2.0]), torch.zeros(2), torch.tensor([2.0, -1.0]),
        convention=WeightConvention.WEIGHTED_MEAN)),
}

# Reducing F first cancels the scalar lead weights; the global metric does not.
e = torch.ones(1, 2, 1, 1)
u = torch.tensor([[[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]])
w = torch.tensor([[[100.0]], [[1.0]]])
g = quadratic_gain(e, u, w, convention=WeightConvention.WEIGHTED_MEAN)
global_gain = ((2 * e[:, None] * u - u.square()) * w).sum((2, 3, 4)) / w.sum()
observations["B06_metric_scope"] = {
    "feature_reduced_gain_then_plain_mean": g.mean((2, 3)).tolist(),
    "global_Q_gain": global_gain.tolist(),
}

m = build_manifest([2015], lead_hours=[6])
old_row = m.rows[0].to_dict()
old_row.pop("availability_source")
observations["B07_availability"] = {
    "default_source": m.rows[0].availability_source,
    "default_delay_hours": (m.rows[0].available_time - m.rows[0].valid_time) / 3600,
    "legacy_missing_source_is_filled_as": SplitManifestRow.from_dict(old_row).availability_source,
}
observations["B09_calendar_index"] = {
    "first_issue_UTC": m.rows[0].issue_time,
    "last_valid_UTC": m.rows[-1].valid_time,
    "normalization_hash": m.rows[0].normalization_hash,
    "last_valid_year": str(np.datetime64(m.rows[-1].valid_time, "s"))[:4],
    "same_issue_leads_have_distinct_event_ids": len({r.event_id for r in
        build_manifest([2015], lead_hours=[6, 12]).rows[:2]}) == 2,
}

empty = plan_from_prediction(torch.ones(2), torch.eye(2), candidate_offsets=torch.empty(0, 2))
observations["A03_empty_registry"] = {
    "returns_noop": empty.support == (), "solver_failures": empty.solver_failures,
}

try:
    from earthdelta.bridge import Stormer, WeatherStepBridge, NormalizationContract, controlled_rollout
    from earthdelta.bridge.stormer_bridge import _check_reentrant_rollout, _release_rollout_lock
    from earthdelta.contracts import ArtifactVersion, EditPlan
    from earthdelta.lowrank import ExpertLoRA
    with contextlib.redirect_stdout(io.StringIO()):
        import s0_gate

    variables = ["2m_temperature", "10m_u_component_of_wind"]
    norm = NormalizationContract(torch.zeros(2), torch.ones(2), {6: torch.zeros(2)},
                                 {6: torch.ones(2)}, variables)
    model = Stormer(in_img_size=(8, 8), variables=variables, patch_size=2,
                    hidden_size=16, depth=1, num_heads=2, mlp_ratio=2.0)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    model.eval().requires_grad_(False)
    version = ArtifactVersion("test", "none", "none", "pending", "8x8", "patch2",
                              "test", "reference_after_hold")
    bridge = WeatherStepBridge(model, norm, version)
    lora = ExpertLoRA(16, 16, num_experts=2, rank_per_expert=2)
    for up in lora.up:
        torch.nn.init.normal_(up.weight, std=0.1)
    coeff = torch.nn.Parameter(torch.tensor([0.1, 0.0]))
    plan = EditPlan("probe", 2, tuple(coeff.unbind()), rho=0.25)
    out = controlled_rollout(bridge, torch.randn(1, 2, 8, 8), variables, 6, 2,
                             plan, {0: lora}, target_blocks=(0,), differentiable=True)
    out.square().sum().backward()
    observations["B03_coefficient_gradient"] = {
        "caller_coefficient_grad_is_none": coeff.grad is None,
        "bank_has_nonzero_gradient": any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                         for p in lora.parameters()),
        "frozen_backbone_has_gradient": any(p.grad is not None for p in model.parameters()),
    }

    with tempfile.TemporaryDirectory(prefix="earthdelta_parity_repro_") as d:
        p = Path(d)
        x = torch.ones(1, 2, 2, 2)
        torch.save(x, p / "input_norm.pt")
        torch.save(x, p / "official_output_6h_1step.pt")
        (p / "manifest.json").write_text(json.dumps({"checkpoint": {"sha256": "wrong"},
            "normalization_digest": "wrong", "outputs_finite": {"6h_4step": False}}))
        identity_bridge = SimpleNamespace(variables=variables,
            version=version, normalization=norm,
            forward_validation=lambda inp, *args, **kwargs: inp)
        parity = s0_gate.verify_upstream_parity(identity_bridge, p,
            np.zeros((2, 2, 2), dtype=np.float32), torch.device("cpu"))
        observations["B02_unbound_reference"] = {
            "wrong_hash_wrong_norm_wrong_raw_input_and_missing_4step_pass": parity["passed"],
            "max_abs_diff": parity.get("max_abs_diff"),
        }
        (p / "fake.ckpt").write_bytes(b"not an official checkpoint")
        observations["B02_hash_gate"] = s0_gate.verify_ckpt_sha256(p / "fake.ckpt", None)["passed"]

    acquired = []
    active = threading.Event()
    release = threading.Event()
    def first():
        _check_reentrant_rollout(bridge)
        acquired.append("first")
        active.set()
        release.wait(5)
        _release_rollout_lock(bridge)
    thread = threading.Thread(target=first)
    thread.start()
    active.wait(5)
    try:
        _check_reentrant_rollout(bridge)
        acquired.append("second_while_first_active")
        _release_rollout_lock(bridge)
    finally:
        release.set()
        thread.join()
    observations["B12_thread_guard"] = acquired

    try:
        s0_gate.generate_report({"criteria_details": {"upstream_parity": {
            "upstream_available": True, "passed": False, "error": "missing tensor file"}}})
        observations["B14_failure_report"] = "unexpected success"
    except Exception as exc:
        observations["B14_failure_report"] = type(exc).__name__ + ": " + str(exc)

    # Mock orchestration only: no real checkpoint, weather data, or GPU execution.
    with contextlib.ExitStack() as stack:
        for name in ["verify_ckpt_sha256", "verify_strict_load_zero_diff", "verify_upstream_parity",
                     "verify_zero_edit_internal_consistency", "verify_normalization_parity",
                     "verify_no_state_leak", "verify_outputs_finite"]:
            stack.enter_context(patch.object(s0_gate, name, return_value={"passed": True}))
        stack.enter_context(patch.object(s0_gate, "load_npy_inputs", return_value={
            "data": np.zeros((1, 2, 2, 2), dtype=np.float32), "lat": np.zeros(2)}))
        stack.enter_context(patch.object(s0_gate, "create_normalization_from_npy", return_value=norm))
        stack.enter_context(patch.object(s0_gate, "load_stormer_checkpoint_detailed",
            return_value=SimpleNamespace(model=model, version=version)))
        stack.enter_context(patch.object(s0_gate, "compute_rmse_sanity", return_value={}))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        stack.enter_context(patch.object(torch.cuda, "empty_cache", side_effect=RuntimeError("cleanup failed")))
        with contextlib.redirect_stdout(io.StringIO()):
            result = s0_gate.run_s0_gate()
        observations["B13_late_exception"] = {
            "s0_gate_pass": result["s0_gate_pass"], "status": result["status"], "error": result.get("error")}
except ImportError as exc:
    observations["bridge_reproductions_blocked"] = str(exc)

try:
    from earthdelta.data.pull_wb2 import create_intermediate_zarr, is_year_complete, get_stormer_target_grid
    import xarray as xr
    with tempfile.TemporaryDirectory(prefix="earthdelta_zarr_repro_") as d:
        p = Path(d) / "2015.zarr"
        times = np.arange(np.datetime64("2015-01-01"), np.datetime64("2016-01-01"), np.timedelta64(6, "h"))
        lat, lon = get_stormer_target_grid()
        create_intermediate_zarr(p, times, lat, lon)
        complete = is_year_complete(p, 2015)
        ds = xr.open_zarr(p)
        unwritten = bool(np.isnan(ds.data.isel(time=0, channel=0).values).all())
        ds.close()
        observations["B08_unwritten_store"] = {"declared_complete": complete, "first_field_all_nan": unwritten}
except ImportError as exc:
    observations["zarr_reproduction_blocked"] = str(exc)

print(json.dumps(observations, indent=2, allow_nan=False))
