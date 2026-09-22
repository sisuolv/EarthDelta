#!/usr/bin/env python3
"""Execute the preregistered shared-F0 pilot, with explicit stage dependencies."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from earthdelta import bank_training as bt
from earthdelta.bridge.stormer_bridge import (
    NormalizationContract, WeatherStepBridge, controlled_rollout, load_stormer_checkpoint,
)
from earthdelta.contracts import GateIdentityConfig, compute_normalization_asset_sha256
from earthdelta.static_adapter import state_dict_digest
from earthdelta.value_pilot import (
    PilotViolation, action_plan, admitted_rows, array_digest, bank_digest,
    canonical_digest, component_gains, component_losses, file_digest,
    legal_features, load_dynamic, load_frozen_run, qualify_losses, read_json, read_row_content,
    require_pass, save_dynamic, validate_cache_rows, verify_store, weighted_projection, write_json,
)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): file_digest(p) for p in (
        root / "earthdelta/bridge/stormer_bridge.py", root / "earthdelta/lowrank.py",
        root / "earthdelta/metrics_contract.py", root / "earthdelta/value_pilot.py",
        Path(__file__).resolve())}


def runtime():
    if not torch.cuda.is_available():
        raise PilotViolation("real pilot requires a CUDA worker")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    return torch.device("cuda:0")


def make_bridge(run, device):
    config = GateIdentityConfig.load_json(Path(run) / "s0/gate_config.json")
    if file_digest(config.checkpoint_path) != config.expected_checkpoint_sha256:
        raise PilotViolation("backbone checkpoint SHA mismatch")
    if compute_normalization_asset_sha256(config.normalization_dir) != config.expected_normalization_npz_sha256:
        raise PilotViolation("normalization NPZ assets changed")
    norm = NormalizationContract.from_npz_dir(config.normalization_dir, variables=list(config.variables),
                                             intervals=tuple(config.normalization_intervals), policy=config.normalization_policy)
    if norm.identity_digest != config.expected_normalization_identity:
        raise PilotViolation("effective normalization identity mismatch")
    model, version = load_stormer_checkpoint(config.checkpoint_path, config.patch_size,
                                             variables=list(config.variables), in_img_size=tuple(config.grid_shape),
                                             hidden_size=config.hidden_size, depth=config.depth,
                                             num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                                             compute_sha256=True)
    model.to(device).eval().requires_grad_(False)
    return WeatherStepBridge(model, norm, version), config


def identity_for(run):
    reference = read_json(Path(run) / "reference_manifest.json")
    require_pass(reference, "shared reference")
    if reference["spec_sha256"] != file_digest(Path(run) / "frozen_spec.json"):
        raise PilotViolation("reference spec changed")
    identity = reference["identity"]
    if identity["admission_sha256"] != file_digest(Path(run) / "admission.json"):
        raise PilotViolation("reference admission changed")
    if identity["gate_config_sha256"] != file_digest(Path(run) / "s0/gate_config.json"):
        raise PilotViolation("reference gate configuration changed")
    return reference, identity


def check_bridge_identity(bridge, identity):
    if bridge.normalization_identity != identity["normalization_identity"]:
        raise PilotViolation("runtime reference normalizer changed")
    actual = state_dict_digest(bridge.model)
    if actual != identity["backbone_state_sha256"]:
        raise PilotViolation("runtime reference weights differ")
    return actual


def load_issue(row, stores, opened, bridge, device):
    year = str(row["year"])
    if year not in opened:
        opened[year] = verify_store(stores[year])
    arrays, _ = read_row_content(row, opened[year], expected=row["content_sha256"])
    x = bridge.normalize(torch.from_numpy(arrays[row["index"]]).unsqueeze(0).to(device))
    history = bridge.normalize(torch.from_numpy(arrays[row["history_index"]]).unsqueeze(0).to(device))
    truth = bridge.normalize(torch.from_numpy(np.stack([arrays[i] for i in row["target_indices"]])).to(device))
    return x, history, truth


def predict(bridge, x, bank, action, steps=12):
    trajectory = controlled_rollout(bridge, x, list(bridge.variables), 6, steps,
                                   action_plan(action), bank, target_blocks=tuple(bank),
                                   return_trajectory=True)
    if steps == 4:
        return trajectory[0, 4:5]
    return trajectory[0, [1, 4, 12]]


def reference_stage(a):
    spec, stores, _ = load_frozen_run(a.run)
    rows, _ = admitted_rows(a.run, role="debug")
    device = runtime()
    gates = sorted((a.run / "s0/gate_output").glob("*/s0_gate_result.json"))
    if len(gates) != 1:
        raise PilotViolation("need exactly one S0 gate result bound to this run")
    gate = read_json(gates[0])
    if gate.get("s0_gate_pass") is not True or gate.get("verdict_committed") is not True:
        raise PilotViolation("official S0 is not PASS")
    bridge, config = make_bridge(a.run, device)
    if gate["config_digest"] != config.config_digest:
        raise PilotViolation("S0 gate belongs to different configuration")
    lat = stores["2018"]["grid_channels"]["lat"]
    identity = {"reference_kind": "base_frozen", "checkpoint_sha256": config.expected_checkpoint_sha256,
                "backbone_state_sha256": state_dict_digest(bridge.model),
                "normalization_identity": bridge.normalization_identity,
                "grid_channels_sha256": stores["2018"]["grid_channels_sha256"],
                "objective_sha256": canonical_digest(spec["objective_exact"]),
                "hidden_size": config.hidden_size, "target_blocks": spec["bank"]["target_blocks"],
                "num_experts": 4, "rank": 4, "hold_steps": 4, "interval": 6,
                "admission_sha256": file_digest(a.run / "admission.json"),
                "gate_config_sha256": file_digest(a.run / "s0/gate_config.json"),
                "s0_file_sha256": file_digest(gates[0])}
    x, _, truth = load_issue(rows[0], stores, {}, bridge, device)
    bank = bt.build_dynamic_bank(config.hidden_size, identity["target_blocks"], seed=0)
    for lora in bank.values():
        lora.to(device).requires_grad_(False)
    with torch.no_grad():
        p0 = predict(bridge, x, bank, spec["actions"][0])
        p1 = predict(bridge, x, bank, spec["actions"][0])
    if not torch.equal(p0, p1):
        raise PilotViolation("shared reference is not repeatable")
    write_json(a.run / "reference_manifest.json", {"status": "PASS", "identity": identity,
               "spec_sha256": file_digest(a.run / "frozen_spec.json"), "s0_path": str(gates[0]),
               "source": source_identity(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name(), "debug_repeat_max_diff": float((p0 - p1).abs().max()),
               "debug_losses": component_losses(p0, truth, lat).tolist(),
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print("shared reference PASS", flush=True)


def training_stage(a):
    spec, stores, _ = load_frozen_run(a.run)
    _, identity = identity_for(a.run)
    rows, _ = admitted_rows(a.run, role="bank_fit")
    panel, _ = admitted_rows(a.run, role="bank_qualification")
    rows = [r for r in rows if r["expert_index"] == a.expert]
    panel = [r for r in panel if r["expert_index"] == a.expert]
    if len(rows) < 16 or len(panel) < 8:
        raise PilotViolation("insufficient expert group/panel")
    device = runtime()
    seed = spec["bank"]["seeds"][a.expert]
    torch.manual_seed(seed)
    bridge, _ = make_bridge(a.run, device)
    check_bridge_identity(bridge, identity)
    bank = bt.build_dynamic_bank(identity["hidden_size"], identity["target_blocks"], seed=seed)
    for lora in bank.values():
        lora.to(device)
    params = bt._select_expert_parameters(bank, a.expert)
    train_spec = spec["bank"]
    if a.retry:
        train_spec = {**train_spec, **spec["bank"]["bounded_retry"], "checkpoint_updates": [128]}
    optimizer = torch.optim.Adam(params, lr=train_spec["learning_rate"])
    action = next(x for x in spec["actions"] if x["expert"] == a.expert and x["alpha"] == 0.25)
    opened = {}
    # Keep normalized inputs/targets on CPU; transfer only the active issue.
    samples = [tuple(t.cpu() for t in load_issue(r, stores, opened, bridge, device)) for r in rows]
    panel_samples = [tuple(t.cpu() for t in load_issue(r, stores, opened, bridge, device)) for r in panel]
    lat = stores["2018"]["grid_channels"]["lat"]
    start = time.monotonic()
    records, checkpoints = [], []
    a.out.mkdir(parents=True, exist_ok=False)
    for update in range(1, train_spec["max_updates"] + 1):
        i = (update - 1) % len(samples)
        x, _, truth = [t.to(device) for t in samples[i]]
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(bridge, x, bank, action, steps=4)
        loss = component_losses(prediction, truth[1:2], lat).mean()
        if not torch.isfinite(loss):
            raise PilotViolation(f"nonfinite loss at update {update}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, train_spec["grad_clip"])
        if not torch.isfinite(norm):
            raise PilotViolation(f"nonfinite gradient at update {update}")
        optimizer.step()
        records.append({"update": update, "issue_id": rows[i]["issue_id"], "loss24": float(loss.detach()), "grad_norm": float(norm)})
        del prediction, loss, x, truth
        if update % 16 == 0:
            print("expert", a.expert, "update", update, records[-1], flush=True)
        if update in train_spec["checkpoint_updates"]:
            record = save_dynamic(a.out / f"dynamic_u{update:03d}.pt", bank, identity,
                                  {"expert_index": a.expert, "seed": seed, "update": update,
                                   "learning_rate": train_spec["learning_rate"], "a0": 0.25,
                                   "retry": a.retry, "training_issues": [r["issue_id"] for r in rows]})
            outputs = []
            with torch.no_grad():
                for px, _, _ in panel_samples:
                    outputs.append(predict(bridge, px.to(device), bank, action).cpu().numpy())
            path = a.out / f"reload_expected_u{update:03d}.npy"
            np.save(path, np.stack(outputs))
            record["expected_panel_path"] = str(path)
            record["expected_panel_sha256"] = file_digest(path)
            record["panel_issue_ids"] = [r["issue_id"] for r in panel]
            checkpoints.append(record)
            write_json(a.out / "training_progress.json", {"status": "RUNNING", "records": records, "checkpoints": checkpoints}, replace=True)
    check_bridge_identity(bridge, identity)
    write_json(a.out / "training.json", {"status": "PASS", "identity": identity, "checkpoints": checkpoints,
               "records": records, "source": source_identity(), "seconds": time.monotonic() - start,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "reference_unchanged": True})
    print("training complete; independent qualification required", flush=True)


def qualification_stage(a):
    spec, stores, _ = load_frozen_run(a.run)
    _, identity = identity_for(a.run)
    training = read_json(a.out / "training.json")
    require_pass(training, "expert training")
    if training["identity"] != identity:
        raise PilotViolation("training reference changed")
    rows, _ = admitted_rows(a.run, role="bank_qualification")
    rows = [r for r in rows if r["expert_index"] == a.expert]
    device = runtime()
    bridge, _ = make_bridge(a.run, device)
    check_bridge_identity(bridge, identity)
    opened, lat = {}, stores["2018"]["grid_channels"]["lat"]
    samples = [tuple(t.cpu() for t in load_issue(r, stores, opened, bridge, device)) for r in rows]
    base_action = spec["actions"][0]
    action = next(x for x in spec["actions"] if x["expert"] == a.expert and x["alpha"] == 0.25)
    reference_preds, reference_losses = [], []
    with torch.no_grad():
        for x, _, truth in samples:
            pred = predict(bridge, x.to(device), {}, base_action)
            reference_preds.append(pred.cpu())
            reference_losses.append(component_losses(pred, truth.to(device), lat).tolist())
    results = []
    start = time.monotonic()
    for checkpoint in training["checkpoints"]:
        if checkpoint["panel_issue_ids"] != [r["issue_id"] for r in rows]:
            raise PilotViolation("reload panel issue mismatch")
        bank = load_dynamic(checkpoint, identity, device=device)
        before = bank_digest(bank)
        if file_digest(checkpoint["expected_panel_path"]) != checkpoint["expected_panel_sha256"]:
            raise PilotViolation("saved panel output changed")
        expected = np.load(checkpoint["expected_panel_path"], mmap_mode="r")
        losses, max_response, max_reload = [], 0.0, 0.0
        with torch.no_grad():
            for i, (x, _, truth) in enumerate(samples):
                pred = predict(bridge, x.to(device), bank, action)
                max_reload = max(max_reload, float((pred.cpu() - torch.from_numpy(expected[i].copy())).abs().max()))
                max_response = max(max_response, float((pred.cpu() - reference_preds[i]).abs().max()))
                losses.append(component_losses(pred, truth.to(device), lat).tolist())
        if before != bank_digest(bank):
            raise PilotViolation("qualification mutated bank")
        result = qualify_losses(reference_losses, losses, max_response, max_reload <= 1e-6,
                                spec["bank"]["qualification"])
        result.update(checkpoint=checkpoint, max_reload_abs_diff=max_reload)
        results.append(result)
        print("expert", a.expert, "update", checkpoint["training"]["update"], result["status"], result["mean_loss_ratios"], flush=True)
    good = [r for r in results if r["status"] == "PASS"]
    selected = min(good, key=lambda r: np.mean(r["candidate_losses"], axis=0)[1]) if good else None
    check_bridge_identity(bridge, identity)
    write_json(a.out / "qualification.json", {"status": "PASS" if selected else "FAIL_IMPLEMENTATION",
               "expert_index": a.expert, "identity": identity, "panel_issue_ids": [r["issue_id"] for r in rows],
               "checkpoints": results, "selected": selected, "seconds": time.monotonic() - start,
               "source": source_identity(), "independent_process": True})
    if not selected:
        raise PilotViolation("no checkpoint passes expert qualification")


def assemble_stage(a):
    spec, _, _ = load_frozen_run(a.run)
    _, identity = identity_for(a.run)
    selected, certificates = [], []
    for expert, path in enumerate(a.expert_dirs):
        q = read_json(path / "qualification.json")
        require_pass(q, f"expert {expert} qualification")
        if q["expert_index"] != expert or q["identity"] != identity or not q["independent_process"]:
            raise PilotViolation("qualification does not bind expert/reference/reload")
        require_pass(q["selected"], f"expert {expert} selected checkpoint")
        selected.append(q["selected"]["checkpoint"])
        certificates.append({"path": str(path / "qualification.json"), "sha256": file_digest(path / "qualification.json")})
    bank = bt.build_dynamic_bank(identity["hidden_size"], identity["target_blocks"], seed=0)
    for k, record in enumerate(selected):
        loaded = load_dynamic(record, identity)
        for block, lora in bank.items():
            lora.down[k].load_state_dict(loaded[block].down[k].state_dict(), strict=True)
            lora.up[k].load_state_dict(loaded[block].up[k].state_dict(), strict=True)
    record = save_dynamic(a.run / "bank.pt", bank, identity,
                          {"expert_checkpoints": selected, "qualification_certificates": certificates})
    registry = {"status": "PASS", "identity": identity, "bank": record,
                "actions": spec["actions"], "spec_sha256": file_digest(a.run / "frozen_spec.json"),
                "qualification_certificates": certificates}
    for action in registry["actions"]:
        action_plan(action)
    registry["registry_sha256"] = canonical_digest(registry)
    write_json(a.run / "registry.json", registry)
    write_json(a.run / "bank_manifest.json", {"status": "PASS", "identity": identity, "checkpoint": record,
               "source": source_identity(), "expert_checkpoints": selected,
               "qualification_certificates": certificates})
    print("assembled K=4 common-reference bank PASS", flush=True)


def load_registry(run):
    spec, _, _ = load_frozen_run(run)
    _, identity = identity_for(run)
    registry = read_json(Path(run) / "registry.json")
    require_pass(registry, "registry")
    if registry["identity"] != identity or registry["spec_sha256"] != file_digest(Path(run) / "frozen_spec.json"):
        raise PilotViolation("registry reference/spec mismatch")
    expected = registry.pop("registry_sha256")
    if canonical_digest(registry) != expected or registry["actions"] != spec["actions"]:
        raise PilotViolation("registry was modified")
    registry["registry_sha256"] = expected
    for certificate in registry["qualification_certificates"]:
        if file_digest(certificate["path"]) != certificate["sha256"]:
            raise PilotViolation("qualification certificate changed")
        require_pass(read_json(certificate["path"]), "expert qualification")
    return registry


def cache_stage(a):
    spec, stores, _ = load_frozen_run(a.run)
    registry = load_registry(a.run)
    rows, _ = admitted_rows(a.run, role=a.role)
    if a.mode == "reduced" and a.role != "debug":
        rows = rows[:spec["data"]["reduced"][a.role]]
    rows = rows[a.shard::a.shards]
    if not rows:
        raise PilotViolation("empty cache shard")
    actions = [v for v in registry["actions"] if v["known"] or a.role in ("debug", "sprint_holdout")]
    device = runtime()
    bridge, _ = make_bridge(a.run, device)
    check_bridge_identity(bridge, registry["identity"])
    bank = load_dynamic(registry["bank"], registry["identity"], device=device)
    before = bank_digest(bank)
    identity = {**registry["identity"], "bank_state_sha256": before, "registry_sha256": registry["registry_sha256"]}
    a.out.mkdir(parents=True, exist_ok=False)
    opened, records, times = {}, [], []
    lat = stores["2018"]["grid_channels"]["lat"]
    start = time.monotonic()
    with torch.no_grad():
        for row in rows:
            t0 = time.monotonic()
            issue_dir = a.out / row["issue_id"]
            issue_dir.mkdir()
            x, history, truth = load_issue(row, stores, opened, bridge, device)
            features = legal_features(x, history, row["issue_utc"]).cpu().numpy()
            np.save(issue_dir / "legal_features.npy", features)
            predictions = np.lib.format.open_memmap(issue_dir / "forecasts.npy", mode="w+", dtype=np.float32,
                                                    shape=(len(actions), 3, *truth.shape[1:]))
            losses, responses, gains = [], [], []
            reference = None
            for j, action in enumerate(actions):
                pred = predict(bridge, x, bank, action)
                if not torch.isfinite(pred).all():
                    raise PilotViolation(f"nonfinite forecast at {row['issue_id']} {action['candidate_id']}")
                if j == 0:
                    reference = pred.clone()
                loss = component_losses(pred, truth, lat)
                analytic = component_gains(reference, truth, pred, lat)
                ref_loss = component_losses(reference, truth, lat)
                if not torch.allclose(analytic, ref_loss - loss, atol=1e-11, rtol=1e-8):
                    raise PilotViolation("native endpoint/analytic gain mismatch")
                predictions[j] = pred.cpu().numpy()
                losses.append(loss.cpu().numpy())
                gains.append(analytic.cpu().numpy())
                responses.append(weighted_projection(pred - reference, lat).cpu().numpy())
            predictions.flush()
            del predictions
            # Labels are separate from the legal feature artifact. Holdout labels
            # are consumed only after the model/prediction freeze manifest exists.
            np.savez(issue_dir / "sealed_labels.npz", losses=np.stack(losses), gains=np.stack(gains),
                     error=weighted_projection(truth - reference, lat).cpu().numpy(),
                     responses=np.stack(responses), truth_norm=truth.cpu().numpy())
            metadata = {"issue_id": row["issue_id"], "issue_utc": row["issue_utc"], "role": row["role"],
                        "identity": identity, "actions": actions, "status": "PASS",
                        "features_sha256": file_digest(issue_dir / "legal_features.npy"),
                        "forecasts_sha256": file_digest(issue_dir / "forecasts.npy"),
                        "labels_sha256": file_digest(issue_dir / "sealed_labels.npz"),
                        "source_row_sha256": canonical_digest(row), "seconds": time.monotonic() - t0,
                        "path": str(issue_dir)}
            write_json(issue_dir / "issue.json", metadata)
            records.append(metadata)
            times.append(metadata["seconds"])
            print(a.role, a.shard, len(records), "/", len(rows), row["issue_id"], "seconds", round(times[-1], 3), flush=True)
    if bank_digest(bank) != before:
        raise PilotViolation("cache/profile mutated qualified bank")
    check_bridge_identity(bridge, registry["identity"])
    write_json(a.out / "cache_shard.json", {"status": "PASS", "role": a.role, "mode": a.mode,
               "identity": identity, "shard": a.shard, "shards": a.shards, "issues": records,
               "actions": actions, "seconds": time.monotonic() - start, "issue_seconds": times,
               "trajectories": len(rows) * len(actions), "weather_steps": len(rows) * len(actions) * 12,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "bank_unchanged": True,
               "source": source_identity()})


def merge_cache_stage(a):
    spec, _, _ = load_frozen_run(a.run)
    registry = load_registry(a.run)
    identity = {**registry["identity"], "bank_state_sha256": registry["bank"]["state_sha256"],
                "registry_sha256": registry["registry_sha256"]}
    admitted, _ = admitted_rows(a.run)
    certificates, issues, costs = [], [], []
    for path in a.shard_dirs:
        record = read_json(path / "cache_shard.json")
        require_pass(record, "cache shard")
        if record["identity"] != identity or record["mode"] != a.mode:
            raise PilotViolation("cache shard identity/budget mismatch")
        certificates.append({"path": str(path / "cache_shard.json"), "sha256": file_digest(path / "cache_shard.json")})
        costs.append({k: record[k] for k in ("seconds", "trajectories", "weather_steps", "peak_gpu_bytes")})
        issues.extend(record["issues"])
    roles = a.roles
    expected_all = []
    for role in roles:
        expected = [r for r in admitted if r["role"] == role]
        if a.mode == "reduced" and role != "debug":
            expected = expected[:spec["data"]["reduced"][role]]
        expected_all.extend(r["issue_id"] for r in expected)
        actions = [x for x in registry["actions"] if x["known"] or role in ("debug", "sprint_holdout")]
        rows = []
        for issue in [i for i in issues if i["role"] == role]:
            if issue["actions"] != actions:
                raise PilotViolation("cache actions differ from the frozen role registry")
            for filename, key in (("legal_features.npy", "features_sha256"), ("forecasts.npy", "forecasts_sha256"),
                                  ("sealed_labels.npz", "labels_sha256")):
                if file_digest(Path(issue["path"]) / filename) != issue[key]:
                    raise PilotViolation(f"cache artifact changed: {issue['issue_id']} {filename}")
            rows.extend({"issue_id": issue["issue_id"], "candidate_id": x["candidate_id"],
                         "identity": issue["identity"], "status": issue["status"]} for x in actions)
        validate_cache_rows(rows, [r["issue_id"] for r in expected], actions, identity)
    if len(issues) != len(expected_all) or {r["issue_id"] for r in issues} != set(expected_all):
        raise PilotViolation("cache contains missing, extra or duplicate issues")
    order = {r["issue_id"]: i for i, r in enumerate(admitted)}
    issues.sort(key=lambda r: order[r["issue_id"]])
    write_json(a.out, {"status": "PASS", "identity": identity, "mode": a.mode, "roles": roles,
               "issues": issues, "shard_certificates": certificates, "costs": costs,
               "source": source_identity(), "coverage_complete": True})
    print("cache coverage PASS", len(issues), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("reference", "train-bank", "qualify-bank", "assemble-bank", "cache", "merge-cache"):
        s = sub.add_parser(stage)
        s.add_argument("--run", type=Path, required=True)
        if stage in ("train-bank", "qualify-bank", "cache", "merge-cache"):
            s.add_argument("--out", type=Path, required=True)
        if stage in ("train-bank", "qualify-bank"):
            s.add_argument("--expert", type=int, choices=range(4), required=True)
        if stage == "train-bank":
            s.add_argument("--retry", action="store_true")
        if stage == "assemble-bank":
            s.add_argument("--expert-dirs", type=Path, nargs=4, required=True)
        if stage in ("cache", "merge-cache"):
            s.add_argument("--mode", choices=("standard", "reduced"), required=True)
        if stage == "cache":
            s.add_argument("--role", choices=("debug", "policy_fit", "calibration", "sprint_holdout"), required=True)
            s.add_argument("--shard", type=int, required=True)
            s.add_argument("--shards", type=int, required=True)
        if stage == "merge-cache":
            s.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
            s.add_argument("--roles", nargs="+", required=True)
    return p


def main():
    a = parser().parse_args()
    a.run = a.run.resolve()
    if hasattr(a, "out"):
        a.out = a.out.resolve()
    try:
        {"reference": reference_stage, "train-bank": training_stage, "qualify-bank": qualification_stage,
         "assemble-bank": assemble_stage, "cache": cache_stage, "merge-cache": merge_cache_stage}[a.stage](a)
    except Exception as exc:
        error = {"status": "FAIL_IMPLEMENTATION", "stage": a.stage, "error": f"{type(exc).__name__}: {exc}"}
        if hasattr(a, "out"):
            base = a.out.parent if a.stage == "merge-cache" else a.out
            write_json(base / (a.stage + "_failure.json"), error, replace=True)
        print(json.dumps(error), flush=True)
        raise


if __name__ == "__main__":
    main()
