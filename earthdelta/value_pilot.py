"""Strict artifacts and native objectives for the shared-F0 research pilot.

This path deliberately does not grant admission to the older fitted-Fs pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .bank_training import build_dynamic_bank
from .contracts import EditPlan
from .metrics_contract import full_objective_gain, full_objective_loss
from .static_adapter import area_weight_q


class PilotViolation(ValueError):
    pass


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_digest(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def require_pass(record, name):
    if record.get("status") != "PASS":
        raise PilotViolation(f"{name} is not PASS")


def load_frozen_run(run):
    run = Path(run)
    spec = read_json(run / "frozen_spec.json")
    expected = read_json(run / "manifests/frozen_spec_identity.json")["sha256"]
    if file_digest(run / "frozen_spec.json") != expected:
        raise PilotViolation("frozen specification changed")
    stores = read_json(run / "manifests/data_stores.json")
    candidates = read_json(run / "manifests/issue_candidates.json")
    if canonical_digest(stores) != spec["data_manifest_digest"]:
        raise PilotViolation("store manifest changed")
    if canonical_digest(candidates) != spec["issue_candidates_digest"]:
        raise PilotViolation("UTC candidate list changed")
    return spec, stores, candidates


def verify_store(store):
    import zarr
    path = Path(store["path"])
    for rel, expected in store["metadata_sha256"].items():
        if file_digest(path / rel) != expected:
            raise PilotViolation(f"store metadata changed: {path / rel}")
    z = zarr.open_group(str(path), mode="r")
    grid = {"lat": z["lat"][:].tolist(), "lon": z["lon"][:].tolist(), "channels": z["channel"][:].tolist()}
    if canonical_digest(grid) != store["grid_channels_sha256"]:
        raise PilotViolation("grid/channel values changed")
    origin = np.datetime64(z["time"].attrs["units"].split(" since ", 1)[1], "h")
    time = (origin + z["time"][:].astype("timedelta64[h]")).astype(np.int64)
    if hashlib.sha256(time.tobytes()).hexdigest() != store["time_sha256"]:
        raise PilotViolation("actual UTC time axis changed")
    return z


def read_row_content(row, z, expected=None):
    indices = [row["history_index"], row["index"], *row["target_indices"]]
    if indices != [row["index"] + d for d in (-1, 0, 1, 4, 12)]:
        raise PilotViolation("history/target index mismatch")
    origin = np.datetime64(z["time"].attrs["units"].split(" since ", 1)[1], "h")
    actual = origin + np.timedelta64(int(z["time"][row["index"]]), "h")
    if str(actual) + ":00:00Z" != row["issue_utc"]:
        raise PilotViolation("issue time differs from store")
    arrays, hashes = {}, {}
    for i in indices:
        x = np.asarray(z["data"][i], dtype=np.float32)
        if x.shape != tuple(z["data"].shape[1:]) or not np.isfinite(x).all():
            raise PilotViolation(f"nonfinite/wrong shape at {row['issue_id']} index {i}")
        h = array_digest(x)
        if expected is not None and expected.get(str(i)) != h:
            raise PilotViolation(f"content certificate mismatch at index {i}")
        arrays[i], hashes[str(i)] = x, h
    return arrays, hashes


def assert_disjoint_roles(rows):
    intervals = {}
    for r in rows:
        role = r["role"]
        start = np.datetime64(r["support_start_utc"].removesuffix("Z"), "h").astype(int)
        end = np.datetime64(r["support_end_utc"].removesuffix("Z"), "h").astype(int)
        intervals.setdefault(role, []).append((start, end))
    roles = sorted(intervals)
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            for a, b in intervals[left]:
                if any(a <= d and c <= b for c, d in intervals[right]):
                    raise PilotViolation(f"complete support crosses roles: {left}/{right}")


def admitted_rows(run, *, role=None):
    spec, stores, _ = load_frozen_run(run)
    admission = read_json(Path(run) / "admission.json")
    require_pass(admission, "content/time admission")
    if admission.get("spec_sha256") != file_digest(Path(run) / "frozen_spec.json"):
        raise PilotViolation("stale admission specification")
    rows = admission["rows"]
    if canonical_digest(rows) != admission["rows_sha256"]:
        raise PilotViolation("admission rows changed")
    if admission.get("coverage") != {r: sum(x["role"] == r for x in rows) for r in sorted({x["role"] for x in rows})}:
        raise PilotViolation("admission coverage mismatch")
    expected_counts = {"bank_fit": 128, "bank_qualification": 32, "policy_fit": 256, "calibration": 64, "sprint_holdout": 128, "debug": 8}
    if admission["coverage"] != expected_counts:
        raise PilotViolation("incomplete admission")
    if len({r["issue_id"] for r in rows}) != len(rows):
        raise PilotViolation("duplicate issue IDs")
    for r in rows:
        if r.get("content_verified") is not True or len(r.get("content_sha256", {})) != 5:
            raise PilotViolation("incomplete content certificate")
    assert_disjoint_roles(rows)
    return [r for r in rows if role is None or r["role"] == role], stores


def action_plan(action, num_experts=4):
    coeff = [0.0] * num_experts
    if action["expert"] is not None:
        coeff[int(action["expert"])] = float(action["alpha"])
    elif action["alpha"] != 0:
        raise PilotViolation("reference has nonzero coefficient")
    return EditPlan(plan_id=action["candidate_id"], num_experts=num_experts,
                    coefficients=tuple(coeff), hold_steps=4, interval_hours=6,
                    continuation="reference_after_hold", rho=0.25)


def component_losses(pred_norm, truth_norm, latitude):
    """One scalar per lead; normalized space exactly matches raw input-std Q."""
    if pred_norm.ndim != 4 or pred_norm.shape != truth_norm.shape:
        raise PilotViolation("expected [lead,variable,latitude,longitude]")
    q = area_weight_q(latitude).to(pred_norm.device)
    return full_objective_loss(pred_norm.unsqueeze(1), truth_norm.unsqueeze(1), q)


def component_gains(reference_norm, truth_norm, candidate_norm, latitude):
    q = area_weight_q(latitude).to(reference_norm.device)
    # e = truth - reference, u = candidate - reference.
    return full_objective_gain((truth_norm - reference_norm).unsqueeze(1),
                               (candidate_norm - reference_norm).unsqueeze(1), q)


def bank_state(bank):
    return {f"{block}.{key}": value.detach().cpu().contiguous().clone()
            for block, lora in sorted(bank.items()) for key, value in lora.state_dict().items()}


def tensor_state_digest(state):
    return canonical_digest({name: array_digest(value.detach().cpu().numpy()) for name, value in sorted(state.items())})


def bank_digest(bank):
    return tensor_state_digest(bank_state(bank))


def save_dynamic(path, bank, identity, training):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = bank_state(bank)
    payload = {"schema": "ed-base-dynamic/1", "identity": identity,
               "training": training, "state_dict": state, "state_sha256": tensor_state_digest(state)}
    torch.save(payload, path)
    return {"path": str(path), "sha256": file_digest(path), "state_sha256": payload["state_sha256"],
            "identity": identity, "training": training}


def load_dynamic(record, expected_identity, *, device="cpu"):
    if record["identity"] != expected_identity:
        raise PilotViolation("dynamic checkpoint reference/Q identity mismatch")
    if file_digest(record["path"]) != record["sha256"]:
        raise PilotViolation("dynamic checkpoint bytes changed")
    payload = torch.load(record["path"], map_location="cpu", weights_only=True)
    if payload.get("schema") != "ed-base-dynamic/1" or payload["identity"] != expected_identity:
        raise PilotViolation("dynamic checkpoint embedded identity mismatch")
    if tensor_state_digest(payload["state_dict"]) != record["state_sha256"] or payload["state_sha256"] != record["state_sha256"]:
        raise PilotViolation("dynamic checkpoint state mismatch")
    bank = build_dynamic_bank(expected_identity["hidden_size"], expected_identity["target_blocks"],
                              num_experts=4, rank_per_expert=4, seed=0)
    for block, lora in bank.items():
        prefix = f"{block}."
        lora.load_state_dict({k[len(prefix):]: v for k, v in payload["state_dict"].items() if k.startswith(prefix)}, strict=True)
        lora.to(device).requires_grad_(False)
    return bank


def qualify_losses(reference, candidate, response_max, reloaded_equal, thresholds):
    reference, candidate = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise PilotViolation("qualification needs matched [issues,3] component losses")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all() and np.isfinite(response_max))
    ratios = candidate.mean(0) / reference.mean(0) if finite and np.all(reference.mean(0) > 0) else np.full(3, np.inf)
    passed = (reference.shape[0] >= thresholds["min_panel_issues_per_expert"] and finite and response_max > 0 and
              reloaded_equal is True and ratios[1] <= thresholds["max_mean_loss24_ratio_to_reference"] and
              ratios[2] <= thresholds["max_mean_loss72_ratio_to_reference"])
    return {"status": "PASS" if passed else "FAIL_IMPLEMENTATION", "n_issues": reference.shape[0],
            "mean_loss_ratios": [float(v) if math.isfinite(v) else None for v in ratios],
            "all_finite": finite, "response_max_abs": float(response_max) if math.isfinite(response_max) else None,
            "independent_reload_equal": reloaded_equal,
            "reference_losses": reference.tolist(), "candidate_losses": candidate.tolist()}


def validate_cache_rows(rows, expected_issues, actions, identity):
    expected = {(i, a["candidate_id"]) for i in expected_issues for a in actions}
    seen = set()
    for row in rows:
        key = (row["issue_id"], row["candidate_id"])
        if key in seen or key not in expected:
            raise PilotViolation(f"duplicate/unregistered cache row: {key}")
        if row.get("status") != "PASS" or row.get("identity") != identity:
            raise PilotViolation(f"failed or stale cache row: {key}")
        seen.add(key)
    if seen != expected:
        raise PilotViolation(f"cache missing {len(expected - seen)} rows")
    return True


def require_role(row, role, *, allow_heldout=False):
    if row["role"] != role:
        raise PilotViolation(f"wrong data role: expected {role}")
    if role in ("policy_fit", "calibration") and (not row["known_action"] or allow_heldout):
        raise PilotViolation("heldout amplitude cannot enter fit or calibration")


def weighted_projection(fields, latitude, grid=(8, 16)):
    """Orthonormal box coefficients in the native Q inner product.

    This fixed projection uses no fitted data; its squared norm is the energy
    retained by a piecewise-constant approximation, not the full-field loss.
    """
    h, w = fields.shape[-2:]
    gh, gw = grid
    if h % gh or w % gw:
        raise PilotViolation("projection grid must divide the native grid")
    v = fields.shape[-3]
    lat = torch.as_tensor(latitude, device=fields.device, dtype=fields.dtype)
    weights = torch.cos(lat * (math.pi / 180)).clamp_min(0)
    box_weights = weights.reshape(gh, h // gh)
    boxes = fields.reshape(*fields.shape[:-2], gh, h // gh, gw, w // gw)
    weighted = boxes * box_weights.reshape(*([1] * (fields.ndim - 2)), gh, h // gh, 1, 1)
    means = weighted.sum(dim=(-3, -1)) / (box_weights.sum(1)[:, None] * (w // gw))
    mass = box_weights.sum(1)[:, None] * (w // gw) / (weights.sum() * w * v)
    return (means * mass.sqrt()).reshape(*fields.shape[:-3], -1)


def legal_features(x_norm, history_norm, issue_utc, grid=(4, 8)):
    if x_norm.shape != history_norm.shape:
        raise PilotViolation("history grid mismatch")
    import torch.nn.functional as F
    current = F.adaptive_avg_pool2d(x_norm, grid).flatten()
    difference = F.adaptive_avg_pool2d(x_norm - history_norm, grid).flatten()
    time = np.datetime64(issue_utc.removesuffix("Z"), "h")
    year = time.astype("datetime64[Y]")
    day = float((time - year).astype("timedelta64[h]").astype(int)) / 24
    hour = int(time.astype(int)) % 24
    calendar = torch.tensor([math.sin(2 * math.pi * day / 365.25), math.cos(2 * math.pi * day / 365.25),
                             math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)],
                            dtype=x_norm.dtype, device=x_norm.device)
    return torch.cat([current, difference, calendar])
