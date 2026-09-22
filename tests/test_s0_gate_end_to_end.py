"""End-to-end S0 gate tests driven through the real CLI entrypoint.

These tests build ONE internally consistent synthetic fixture set (NPY input,
NPZ normalization constants, checkpoint, manifest, saved normalized input,
1-step and 4-step reference outputs) and run the REAL `scripts/s0_gate.py`
`main()` over it. The fixture's reference outputs are produced by an
independent rollout implementation written out longhand in this file, NOT by
calling `WeatherStepBridge.forward_validation` (the code under test).

What is and is not substituted:
  * The expensive backbone is replaced by a SMALL REAL Stormer (8 variables,
    16x32 grid, hidden 64, depth 2) whose weights are saved to a real
    checkpoint file and loaded by the real loader. Nothing about the model
    loading path is stubbed.
  * The identity comparisons, the raw-input binding, the multistep numeric
    comparison and the final aggregate verdict all run for real and are what
    these tests observe. Nothing in that chain is mocked or bypassed.

Every negative variant breaks exactly ONE property of the positive fixture and
asserts both a non-zero exit code AND the specific criterion that must fail.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta.bridge import (  # noqa: E402
    CONSTANTS,
    DEFAULT_VARIABLES,
    POLICY_LEGACY,
    POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    NormalizationContract,
    Stormer,
    WeatherStepBridge,
    _compute_file_sha256,
    check_version_match,
)
from earthdelta.contracts import (  # noqa: E402
    NORM_IDENTITY_SCHEMA_VERSION,
    OFFICIAL_STORMER_COMMIT,
    ArtifactVersion,
    GateIdentityConfig,
    compute_coordinate_digest,
    compute_normalization_asset_sha256,
)

import scripts.s0_gate as s0_gate  # noqa: E402
import scripts.export_upstream_reference as exporter  # noqa: E402


# =============================================================================
# Fixture geometry (small enough for CPU, real enough for the real loader)
# =============================================================================

VARIABLES: List[str] = list(DEFAULT_VARIABLES[:8])
N_VARS = len(VARIABLES)
GRID = (16, 32)
PATCH_SIZE = 4
HIDDEN_SIZE = 64
DEPTH = 2
NUM_HEADS = 4
MLP_RATIO = 2.0
INTERVAL_HOURS = 6
ROLLOUT_STEPS = (1, 4)
NORM_INTERVALS = (6, 24)
N_TIMESTEPS = 6


# =============================================================================
# Independent rollout (NOT the code under test)
# =============================================================================

def independent_rollout(
    model: Stormer,
    variables: List[str],
    inp_mean: torch.Tensor,
    inp_std: torch.Tensor,
    diff_mean: torch.Tensor,
    diff_std: torch.Tensor,
    x_norm: torch.Tensor,
    interval: int,
    steps: int,
    patch_size: int,
) -> torch.Tensor:
    """Longhand reimplementation of the official rollout recurrence.

    Written out here so the fixture's "reference" outputs are not produced by
    calling the function the gate is asked to verify.
    """
    interval_tensor = torch.tensor(
        [interval], dtype=x_norm.dtype
    ).div(10.0).repeat(x_norm.shape[0])

    mean_v = inp_mean.view(1, -1, 1, 1)
    std_v = inp_std.view(1, -1, 1, 1)
    dmean_v = diff_mean.view(1, -1, 1, 1)
    dstd_v = diff_std.view(1, -1, 1, 1)

    constant_channels = [i for i, name in enumerate(variables) if name in CONSTANTS]

    x = x_norm
    for _ in range(steps):
        height = x.shape[-2]
        pad = (patch_size - height % patch_size) if height % patch_size else 0
        padded = torch.nn.functional.pad(x, (0, 0, pad, 0), "constant", 0) if pad else x
        with torch.no_grad():
            out = model(padded, variables, interval_tensor)
        pred_diff = out[:, :, pad:]
        if constant_channels:
            pred_diff = pred_diff.clone()
            for channel in constant_channels:
                pred_diff[:, channel] = 0.0
        pred_diff = pred_diff * dstd_v + dmean_v
        pred = (x * std_v + mean_v) + pred_diff
        x = (pred - mean_v) / std_v
    return x


# =============================================================================
# Positive fixture construction
# =============================================================================

def _write_npz(path: Path, variables: List[str], values: np.ndarray) -> None:
    np.savez(path, **{v: np.array([values[i]], dtype=np.float32) for i, v in enumerate(variables)})


def build_positive_fixture(root: Path, seed: int = 20260921) -> Dict[str, Any]:
    """Build one internally consistent asset tree and its frozen config.

    The 1-step and 4-step reference outputs really are what you get by running
    the fixture's own normalizer/transform forward on the fixture's raw input
    under the fixture's stated policy.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    asset_root = root / "assets"
    input_dir = asset_root / "scripts" / "s0_gate_inputs"
    norm_dir = asset_root / "reference" / "stormer" / "normalization_constants"
    ckpt_dir = asset_root / "checkpoints"
    artifacts_dir = asset_root / "artifacts"
    for directory in (input_dir, norm_dir, ckpt_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # --- normalization constants (single source of truth, written twice) ---
    inp_mean = rng.randn(N_VARS).astype(np.float32)
    inp_std = (np.abs(rng.randn(N_VARS)) + 0.5).astype(np.float32)
    diff_mean_raw = {
        interval: (rng.randn(N_VARS).astype(np.float32) * 0.05)
        for interval in NORM_INTERVALS
    }
    diff_std = {
        interval: (np.abs(rng.randn(N_VARS)) + 0.5).astype(np.float32)
        for interval in NORM_INTERVALS
    }

    np.save(input_dir / "inp_mean.npy", inp_mean)
    np.save(input_dir / "inp_std.npy", inp_std)
    for interval in NORM_INTERVALS:
        np.save(input_dir / f"diff_mean_{interval}.npy", diff_mean_raw[interval])
        np.save(input_dir / f"diff_std_{interval}.npy", diff_std[interval])

    _write_npz(norm_dir / "normalize_mean.npz", VARIABLES, inp_mean)
    _write_npz(norm_dir / "normalize_std.npz", VARIABLES, inp_std)
    for interval in NORM_INTERVALS:
        _write_npz(norm_dir / f"normalize_diff_mean_{interval}.npz", VARIABLES, diff_mean_raw[interval])
        _write_npz(norm_dir / f"normalize_diff_std_{interval}.npz", VARIABLES, diff_std[interval])

    # --- coordinates + raw input ---
    lat = np.linspace(-87.5, 87.5, GRID[0]).astype(np.float32)
    lon = np.linspace(0.0, 348.75, GRID[1]).astype(np.float32)
    np.save(input_dir / "lat.npy", lat)
    np.save(input_dir / "lon.npy", lon)

    data = rng.randn(N_TIMESTEPS, N_VARS, GRID[0], GRID[1]).astype(np.float32)
    np.save(input_dir / "jan2020_full.npy", data)
    x_raw = data[0]
    raw_input_hash = hashlib.sha256(x_raw.tobytes()).hexdigest()[:16]

    # --- checkpoint (small but REAL Stormer weights) ---
    model = Stormer(
        in_img_size=GRID,
        variables=VARIABLES,
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
    )
    # Non-trivial head/modulation weights so outputs actually move.
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    model.eval()

    ckpt_path = ckpt_dir / f"stormer_1.40625_patch_size_{PATCH_SIZE}.ckpt"
    torch.save(
        {"state_dict": {f"net.{k}": v for k, v in model.state_dict().items()}},
        ckpt_path,
    )
    ckpt_sha256 = _compute_file_sha256(str(ckpt_path))

    # --- the fixture's own normalizer, under the fixture's stated policy ---
    policy = POLICY_OFFICIAL_ZERO_DIFF_MEAN
    effective_diff_mean = {
        interval: torch.zeros(N_VARS) for interval in NORM_INTERVALS
    }
    norm = NormalizationContract(
        inp_mean=torch.from_numpy(inp_mean).float(),
        inp_std=torch.from_numpy(inp_std).float(),
        diff_mean=effective_diff_mean,
        diff_std={i: torch.from_numpy(diff_std[i]).float() for i in NORM_INTERVALS},
        variables=VARIABLES,
        policy=policy,
    )

    x_raw_t = torch.from_numpy(x_raw).float().unsqueeze(0)
    x_norm = (x_raw_t - norm.inp_mean.view(1, -1, 1, 1)) / norm.inp_std.view(1, -1, 1, 1)

    # --- reference outputs via the INDEPENDENT rollout ---
    outputs: Dict[str, torch.Tensor] = {}
    for steps in ROLLOUT_STEPS:
        outputs[f"{INTERVAL_HOURS}h_{steps}step"] = independent_rollout(
            model, VARIABLES,
            norm.inp_mean, norm.inp_std,
            effective_diff_mean[INTERVAL_HOURS],
            norm.diff_std[INTERVAL_HOURS],
            x_norm, INTERVAL_HOURS, steps, PATCH_SIZE,
        )

    coordinate_digest = compute_coordinate_digest(lat, lon)
    npz_content_sha256 = compute_normalization_asset_sha256(norm_dir)

    # --- frozen config: enrolled BEFORE the reference is written ---
    config = GateIdentityConfig(
        checkpoint_path=str(ckpt_path),
        expected_checkpoint_sha256=ckpt_sha256,
        patch_size=PATCH_SIZE,
        normalization_policy=policy,
        normalization_dir=str(norm_dir),
        variables=tuple(VARIABLES),
        grid_shape=GRID,
        interval_hours=INTERVAL_HOURS,
        rollout_steps=ROLLOUT_STEPS,
        official_commit=OFFICIAL_STORMER_COMMIT,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        normalization_intervals=NORM_INTERVALS,
        expected_normalization_identity=norm.identity_digest,
        expected_normalization_npz_sha256=npz_content_sha256,
        expected_raw_input_hash=raw_input_hash,
        expected_coordinate_digest=coordinate_digest,
        asset_root=str(asset_root),
        input_dir=str(input_dir),
        reference_base_dir=str(artifacts_dir),
    )

    config_path = root / "gate_config.json"
    config.save_json(config_path)

    # --- reference directory, written with the exporter's own manifest builder ---
    ref_dir = artifacts_dir / config.reference_output_dir_name
    ref_dir.mkdir(parents=True, exist_ok=True)
    for key, tensor in outputs.items():
        torch.save(tensor, ref_dir / f"official_output_{key}.pt")
    torch.save(x_norm, ref_dir / "input_norm.pt")
    (ref_dir / "raw_input_hash.txt").write_text(raw_input_hash)

    manifest = exporter.build_reference_manifest(
        config, ckpt_path, norm_dir, VARIABLES, ckpt_sha256, coordinate_digest,
    )
    manifest["outputs"] = {k: list(v.shape) for k, v in outputs.items()}
    manifest["outputs_finite"] = {k: bool(torch.isfinite(v).all()) for k, v in outputs.items()}
    manifest["raw_input_hash"] = raw_input_hash
    manifest["input_path"] = str(input_dir / "jan2020_full.npy")
    (ref_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "root": root,
        "asset_root": asset_root,
        "input_dir": input_dir,
        "norm_dir": norm_dir,
        "ckpt_path": ckpt_path,
        "ckpt_sha256": ckpt_sha256,
        "artifacts_dir": artifacts_dir,
        "ref_dir": ref_dir,
        "config": config,
        "config_path": config_path,
        "manifest": manifest,
        "norm": norm,
        "model": model,
        "x_raw": x_raw,
        "x_norm": x_norm,
        "outputs": outputs,
        "raw_input_hash": raw_input_hash,
        "coordinate_digest": coordinate_digest,
        "lat": lat,
        "lon": lon,
    }


# =============================================================================
# Gate driver: the REAL main() entrypoint
# =============================================================================

def run_gate_cli(fixture: Dict[str, Any], extra_args: List[str] = None,
                 config_path: Path = None) -> Tuple[int, Dict[str, Any]]:
    """Invoke scripts/s0_gate.py main() exactly as the CLI does.

    Returns the real process exit code plus the published JSON result.
    """
    out_base = fixture["root"] / "gate_out"
    argv = [
        "s0_gate.py",
        "--config", str(config_path or fixture["config_path"]),
        "--output-dir", str(out_base),
    ]
    if extra_args:
        argv.extend(extra_args)

    old_argv = sys.argv
    sys.argv = argv
    try:
        exit_code = s0_gate.main()
    finally:
        sys.argv = old_argv

    run_dirs = sorted(p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("."))
    assert run_dirs, "gate produced no run-scoped output directory"
    latest = run_dirs[-1]
    with open(latest / "s0_gate_result.json") as handle:
        result = json.load(handle)
    result["_run_dir"] = str(latest)
    return exit_code, result


def mutate_config(fixture: Dict[str, Any], name: str, **overrides) -> Path:
    """Write a variant config JSON that differs in exactly the given fields."""
    data = fixture["config"].to_dict()
    data.update(overrides)
    path = fixture["root"] / f"gate_config_{name}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def mutate_manifest(fixture: Dict[str, Any], mutator) -> None:
    """Rewrite the reference manifest in place via `mutator(manifest)`."""
    manifest_path = fixture["ref_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutator(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))


@pytest.fixture(autouse=True)
def _single_threaded_torch():
    """Pin torch to one intra-op thread for these tests.

    The tensors here are tiny; on a many-core host torch's default intra-op
    thread pool costs orders of magnitude more in synchronization than the
    arithmetic itself. This only changes scheduling, never what is asserted,
    and the previous setting is restored afterwards.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def fixture(tmp_path):
    return build_positive_fixture(tmp_path)


# =============================================================================
# 1. Full positive: the real gate passes end-to-end with exit code 0
# =============================================================================

def test_positive_fixture_passes_real_gate(fixture):
    """The internally consistent fixture passes the real gate with exit code 0."""
    exit_code, result = run_gate_cli(fixture)

    failed = [k for k, v in result["gate_criteria"].items() if not v]
    assert exit_code == 0, (
        f"Gate should pass on the consistent fixture. Failing criteria: {failed}. "
        f"Details: {json.dumps({k: result['criteria_details'].get(k) for k in failed}, default=str)[:4000]}"
    )
    assert result["s0_gate_pass"] is True
    assert result["status"] == "ok"
    assert all(result["gate_criteria"].values())

    # PASS marker is published, and only on success.
    assert (Path(result["_run_dir"]) / "PASS").exists()
    # No staging leftovers.
    assert not list(Path(result["_run_dir"]).parent.glob(".staging-*"))


def test_positive_multistep_parity_actually_compared_values(fixture):
    """The 4-step criterion really compared numbers, not just shapes."""
    _, result = run_gate_cli(fixture)
    detail = result["criteria_details"]["multistep_parity"]
    assert set(detail["comparisons"]) == {"6h_1step", "6h_4step"}
    for key, entry in detail["comparisons"].items():
        assert entry["passed"] is True
        assert "max_abs_diff" in entry, f"{key} produced no numeric comparison"


def test_source_root_and_asset_root_are_separate(fixture):
    """Assets live outside the source snapshot, and the snapshot is recorded."""
    _, result = run_gate_cli(fixture)

    assert result["source_root"] == str(REPO_ROOT)
    assert result["asset_root"] == str(fixture["asset_root"])
    assert result["asset_root"] != result["source_root"], (
        "the fixture must prove the code does not assume one directory"
    )

    modules = result["source_identity"]["modules"]
    for name in ("earthdelta.contracts", "earthdelta.bridge.stormer_bridge", "scripts.s0_gate"):
        assert modules[name]["sha256"], f"no content hash recorded for {name}"
        assert len(modules[name]["sha256"]) == 64
        assert modules[name]["path"].endswith(".py")


def test_each_run_gets_a_unique_output_directory(fixture):
    """Two runs must not share or reuse an output directory."""
    run_gate_cli(fixture)
    run_gate_cli(fixture)
    out_base = fixture["root"] / "gate_out"
    run_dirs = [p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert len(run_dirs) == 2, f"expected 2 distinct run dirs, got {run_dirs}"


# =============================================================================
# 2. Negative variants - each breaks exactly one field
# =============================================================================

def test_negative_wrong_checkpoint_sha(fixture):
    path = mutate_config(fixture, "wrong_sha", expected_checkpoint_sha256="0" * 64)
    exit_code, result = run_gate_cli(fixture, config_path=path)

    assert exit_code == 1
    assert result["gate_criteria"]["ckpt_sha256_bound"] is False
    assert "mismatch" in result["criteria_details"]["ckpt_sha256_bound"]["error"].lower()


def test_negative_missing_checkpoint_sha(fixture):
    path = mutate_config(fixture, "missing_sha", expected_checkpoint_sha256="")
    exit_code, result = run_gate_cli(fixture, config_path=path)

    assert exit_code == 1
    assert result["gate_criteria"]["identity_config_bound"] is False
    assert "expected_checkpoint_sha256" in str(
        result["criteria_details"]["identity_config_bound"]["missing_expected_fields"]
    )


def test_negative_wrong_norm_digest(fixture):
    path = mutate_config(
        fixture, "wrong_norm_digest",
        expected_normalization_identity=f"{NORM_IDENTITY_SCHEMA_VERSION}:{'0' * 16}",
    )
    exit_code, result = run_gate_cli(fixture, config_path=path)

    assert exit_code == 1
    assert result["gate_criteria"]["normalization_parity"] is False
    detail = result["criteria_details"]["normalization_parity"]
    assert detail["digests_match_expected"] is False
    assert detail.get("format_version_mismatch") is not True, (
        "should be a value mismatch, not a format mismatch"
    )


def test_negative_wrong_norm_policy_in_config(fixture):
    path = mutate_config(fixture, "wrong_policy", normalization_policy=POLICY_LEGACY)
    exit_code, result = run_gate_cli(fixture, config_path=path)

    assert exit_code == 1
    assert result["gate_criteria"]["normalization_parity"] is False


def test_negative_wrong_norm_policy_in_manifest(fixture):
    mutate_manifest(fixture, lambda m: m["normalization"].__setitem__("policy", POLICY_LEGACY))
    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["manifest_identity_match"] is False
    blob = json.dumps(result["criteria_details"]["manifest_identity_match"])
    assert "normalization_policy" in blob


def test_negative_wrong_source_identity(fixture):
    mutate_manifest(
        fixture,
        lambda m: m["official_source"].__setitem__("pinned_commit", "f" * 40),
    )
    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["source_identity_match"] is False
    assert "pinned upstream commit mismatch" in json.dumps(
        result["criteria_details"]["source_identity_match"]
    )


def test_negative_wrong_variable_order(fixture):
    swapped = list(VARIABLES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    path = mutate_config(fixture, "swapped_vars", variables=swapped)
    exit_code, result = run_gate_cli(fixture, config_path=path)

    assert exit_code == 1
    assert result["gate_criteria"]["variable_coordinate_identity_match"] is False
    blob = json.dumps(result["criteria_details"]["variable_coordinate_identity_match"])
    assert "variable" in blob.lower()


def test_negative_same_shape_different_coordinates(fixture):
    """Coordinate VALUES change while the shape stays identical."""
    lat = np.load(fixture["input_dir"] / "lat.npy")
    shifted = (lat + 0.25).astype(np.float32)
    assert shifted.shape == lat.shape
    np.save(fixture["input_dir"] / "lat.npy", shifted)

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["variable_coordinate_identity_match"] is False
    detail = result["criteria_details"]["variable_coordinate_identity_match"]
    assert detail["grid_shape"] == list(GRID), "shape must be unchanged; only values differ"
    assert "coordinate" in json.dumps(detail).lower()


def test_negative_saved_normalized_input_does_not_match_raw(fixture):
    """Swap in an unrelated saved 'normalized' tensor of the right shape.

    The raw-content hash still matches, so the pre-existing raw_input_binding
    criterion still passes. Only the new derivation check catches this.
    """
    unrelated = torch.randn_like(fixture["x_norm"])
    torch.save(unrelated, fixture["ref_dir"] / "input_norm.pt")

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["raw_input_binding"] is True, (
        "raw content hash is untouched; the old check alone would have passed"
    )
    assert result["gate_criteria"]["input_norm_binding"] is False
    error = result["criteria_details"]["input_norm_binding"]["error"]
    assert "does not correspond to this raw input" in error


def test_negative_four_step_output_finite_but_numerically_wrong(fixture):
    """A finite, correctly shaped, wrong 4-step tensor must fail the gate.

    The structural criterion (present / shape / finite) still passes, which is
    exactly what the old gate checked; the numeric criterion catches it.
    """
    reference = fixture["outputs"]["6h_4step"]
    wrong = reference + 1.0
    assert wrong.shape == reference.shape
    assert bool(torch.isfinite(wrong).all())
    torch.save(wrong, fixture["ref_dir"] / "official_output_6h_4step.pt")

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["multistep_reference_present"] is True, (
        "structural check still passes; the old gate would have accepted this"
    )
    assert result["gate_criteria"]["multistep_parity"] is False
    comparisons = result["criteria_details"]["multistep_parity"]["comparisons"]
    assert comparisons["6h_1step"]["passed"] is True
    assert comparisons["6h_4step"]["passed"] is False
    assert comparisons["6h_4step"]["max_abs_diff"] > 0.5


def test_negative_missing_multistep_file(fixture):
    os.remove(fixture["ref_dir"] / "official_output_6h_4step.pt")
    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["multistep_reference_present"] is False
    assert "6h_4step" in str(
        result["criteria_details"]["multistep_reference_present"]["missing_files"]
    )


def test_negative_dropped_multistep_registration(fixture):
    """Dropping the 4-step rollout from the config cannot silently skip it.

    The reference was enrolled under a config that registered both rollouts, so
    a config that registers only one no longer matches that enrollment.
    """
    path = mutate_config(fixture, "one_step_only", rollout_steps=[1])
    exit_code, result = run_gate_cli(fixture, config_path=path)

    assert exit_code == 1
    assert result["gate_criteria"]["source_identity_match"] is False
    assert "config digest mismatch" in json.dumps(
        result["criteria_details"]["source_identity_match"]
    )


def test_negative_empty_rollout_registration_is_rejected(fixture):
    """An empty rollout set must fail the CLI, not pass vacuously."""
    path = mutate_config(fixture, "empty_rollouts", rollout_steps=[])

    old_argv = sys.argv
    sys.argv = [
        "s0_gate.py", "--config", str(path),
        "--output-dir", str(fixture["root"] / "gate_out_empty"),
    ]
    try:
        exit_code = s0_gate.main()
    finally:
        sys.argv = old_argv

    assert exit_code == 1, "empty rollout registration must fail the gate"

    # And the verifier itself refuses an empty registration directly.
    detail = s0_gate.verify_multistep_parity(
        bridge=None, x_raw=fixture["x_raw"],
        upstream_base_dir=fixture["artifacts_dir"], patch_size=PATCH_SIZE,
        device=torch.device("cpu"), registered_rollouts=(),
    )
    assert detail["passed"] is False
    assert "Empty rollout registration" in detail["error"]

    structural = s0_gate.verify_multistep_reference(
        fixture["artifacts_dir"], PATCH_SIZE, required_steps=(),
    )
    assert structural["passed"] is False
    assert "Empty multistep registration" in structural["error"]


def test_negative_ambiguous_reference_dirs_refuse_auto_selection(fixture):
    """Two directories claiming the same identity must not be broken by sort order."""
    twin = fixture["artifacts_dir"] / (fixture["ref_dir"].name + "_copy")
    shutil.copytree(fixture["ref_dir"], twin)

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    blob = json.dumps(result["criteria_details"])
    assert "Ambiguous reference selection" in blob or "ambiguous" in blob.lower()
    assert "--reference-dir" in blob

    # Explicit selection resolves it and the gate passes again.
    exit_code2, result2 = run_gate_cli(
        fixture, extra_args=["--reference-dir", str(fixture["ref_dir"])]
    )
    failed = [k for k, v in result2["gate_criteria"].items() if not v]
    assert exit_code2 == 0, f"--reference-dir should resolve ambiguity; failing: {failed}"


def test_negative_legacy_format_reference_fails_closed(fixture):
    """A pre-schema reference must fail closed, never be silently upgraded.

    The manifest keeps the correct bare digest but drops the schema-qualified
    identity, exactly as a reference exported before the schema existed would.
    """
    def strip_schema(manifest):
        manifest["normalization"].pop("identity_digest", None)
        manifest["normalization"].pop("identity_schema", None)

    mutate_manifest(fixture, strip_schema)

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    blob = json.dumps(result["criteria_details"])
    assert "format version mismatch" in blob.lower(), (
        f"legacy reference must fail with an explicit format error; got {blob[:2000]}"
    )


def test_negative_no_config_fails_closed(fixture):
    """Without a frozen config every identity-bound criterion fails closed."""
    old_argv = sys.argv
    out_base = fixture["root"] / "gate_out_noconfig"
    sys.argv = [
        "s0_gate.py",
        "--output-dir", str(out_base),
        "--asset-root", str(fixture["asset_root"]),
    ]
    try:
        exit_code = s0_gate.main()
    finally:
        sys.argv = old_argv

    assert exit_code == 1
    run_dirs = sorted(p for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("."))
    with open(run_dirs[-1] / "s0_gate_result.json") as handle:
        result = json.load(handle)

    assert result["gate_criteria"]["identity_config_bound"] is False
    # The version check is a real gate criterion now, not a logged side note.
    assert "version_identity_match" in result["gate_criteria"]
    assert result["gate_criteria"]["version_identity_match"] is False
    assert not (run_dirs[-1] / "PASS").exists()


# =============================================================================
# 3. Bug 1: the version check is a real comparison, not a self-comparison
# =============================================================================

def test_version_identity_is_not_a_self_comparison(fixture):
    """The old `check_version_match(v, v)` call authenticated nothing.

    Demonstrated directly: the self-comparison accepts a bridge whose identity
    disagrees with the frozen config, while the real comparison rejects it.
    """
    norm = fixture["norm"]
    bridge = WeatherStepBridge(
        fixture["model"], norm,
        ArtifactVersion(
            backbone="stormer_ps4_h64_d2_nh4_mr2.0_sha256:" + "deadbeefdeadbeef",
            static_adapter="none", edit_bank="none", normalization="pending",
            grid="16x32", projection="patch4", split="full",
            continuation="reference_after_hold",
        ),
    )

    # The OLD call: a value compared with itself always succeeds.
    check_version_match(bridge.version, bridge.version)

    # The NEW check compares against the frozen expected identity and rejects.
    detail = s0_gate.verify_version_identity(bridge, fixture["config"])
    assert detail["passed"] is False
    assert "backbone" in detail["differing_fields"]
    assert detail["expected_version"]["backbone"] != detail["actual_version"]["backbone"]


def test_version_identity_accepts_the_real_bridge(fixture):
    """The same real comparison accepts a bridge built from the enrolled config."""
    from earthdelta.bridge import load_stormer_checkpoint_detailed

    load_result = load_stormer_checkpoint_detailed(
        str(fixture["ckpt_path"]), patch_size=PATCH_SIZE, variables=VARIABLES,
        in_img_size=GRID, hidden_size=HIDDEN_SIZE, depth=DEPTH,
        num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO,
    )
    bridge = WeatherStepBridge(load_result.model, fixture["norm"], load_result.version)

    detail = s0_gate.verify_version_identity(
        bridge, fixture["config"], load_result, fixture["ckpt_sha256"]
    )
    assert detail["passed"] is True, detail


def test_version_identity_fails_closed_without_config(fixture):
    bridge = WeatherStepBridge(fixture["model"], fixture["norm"], fixture["config"]
                               .expected_artifact_version(fixture["norm"].digest))
    detail = s0_gate.verify_version_identity(bridge, None)
    assert detail["passed"] is False
    assert "IDENTITY_NOT_BOUND" in detail["reason"]


# =============================================================================
# 4. Bug 2: manifest identity is enforced, not merely displayed
# =============================================================================

def test_manifest_checkpoint_sha_is_enforced_against_runtime(fixture):
    """A manifest claiming a different checkpoint must fail the gate."""
    mutate_manifest(fixture, lambda m: m["checkpoint"].__setitem__("sha256", "a" * 64))
    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["gate_criteria"]["manifest_identity_match"] is False


def test_manifest_identity_reports_both_config_and_runtime_sides(fixture):
    """The enforcement covers config AND the actually-loaded runtime objects."""
    detail = s0_gate.verify_manifest_identity(
        fixture["artifacts_dir"], PATCH_SIZE, fixture["config"],
        fixture["norm"], fixture["ckpt_sha256"],
    )
    assert detail["passed"] is True, detail
    assert detail["config_mismatches"] == []
    assert detail["runtime_mismatches"] == []
    assert detail["runtime_checkpoint_sha256"] == fixture["ckpt_sha256"]
    assert detail["runtime_normalization_identity"] == fixture["norm"].identity_digest

    # Same manifest, a DIFFERENT runtime normalization -> rejected.
    other = NormalizationContract(
        inp_mean=fixture["norm"].inp_mean + 1.0,
        inp_std=fixture["norm"].inp_std,
        diff_mean=fixture["norm"].diff_mean,
        diff_std=fixture["norm"].diff_std,
        variables=VARIABLES,
        policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )
    rejected = s0_gate.verify_manifest_identity(
        fixture["artifacts_dir"], PATCH_SIZE, fixture["config"],
        other, fixture["ckpt_sha256"],
    )
    assert rejected["passed"] is False
    assert any("normalization_identity" in r for r in rejected["runtime_mismatches"])


# =============================================================================
# 5. Bug 5: selection and publication
# =============================================================================

def test_selection_refuses_legacy_fallback_directory(fixture):
    """The legacy `artifacts/upstream_reference` fallback no longer exists."""
    shutil.rmtree(fixture["ref_dir"])
    legacy = fixture["artifacts_dir"] / "upstream_reference"
    legacy.mkdir()

    with pytest.raises(s0_gate.ReferenceSelectionError) as excinfo:
        s0_gate.select_reference_dir(
            fixture["artifacts_dir"], PATCH_SIZE, fixture["config"]
        )
    assert "not found" in str(excinfo.value).lower()


def test_selection_is_not_decided_by_sort_order(fixture):
    """Two identity-matching candidates are ambiguous regardless of their names."""
    twin = fixture["artifacts_dir"] / (fixture["ref_dir"].name + "_zzz")
    shutil.copytree(fixture["ref_dir"], twin)

    with pytest.raises(s0_gate.ReferenceSelectionError) as excinfo:
        s0_gate.select_reference_dir(
            fixture["artifacts_dir"], PATCH_SIZE, fixture["config"]
        )
    message = str(excinfo.value)
    assert "Ambiguous" in message
    assert "--reference-dir" in message


def test_pass_marker_is_published_last_and_only_on_success(fixture):
    """A failing run publishes its JSON but never a PASS marker."""
    os.remove(fixture["ref_dir"] / "official_output_6h_4step.pt")
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 1
    run_dir = Path(result["_run_dir"])
    assert (run_dir / "s0_gate_result.json").exists()
    assert not (run_dir / "PASS").exists()


def test_publish_refuses_to_reuse_a_run_directory(fixture, tmp_path):
    """Run-scoped output directories are never reused."""
    base = tmp_path / "publish_base"
    result = {"run_id": "s0-gate-fixed", "s0_gate_pass": True, "finished_utc": "now"}
    first = s0_gate.publish_gate_outputs(base, result, "# report")
    assert (first / "PASS").exists()

    with pytest.raises(RuntimeError, match="will not be reused"):
        s0_gate.publish_gate_outputs(base, result, "# report")


# =============================================================================
# 6. Bug 6: one shared identity serialization for bridge and exporter
# =============================================================================

def test_bridge_and_exporter_digests_agree_on_the_fixture_npz(fixture):
    """Bridge and exporter derive the SAME identity from the same NPZ asset."""
    bridge_norm = NormalizationContract.from_npz_dir(
        str(fixture["norm_dir"]), variables=VARIABLES,
        intervals=NORM_INTERVALS, policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )
    exporter_identity = exporter.compute_normalization_identity(
        fixture["norm_dir"], VARIABLES, NORM_INTERVALS,
    )
    assert bridge_norm.identity_digest == exporter_identity

    # And the transforms stay independent: the exporter never builds a contract.
    inp_mean, inp_std, diff_mean, diff_std = exporter.load_official_constants(
        fixture["norm_dir"], VARIABLES, NORM_INTERVALS
    )
    assert isinstance(inp_mean, np.ndarray)
    for interval in NORM_INTERVALS:
        assert np.all(diff_mean[interval] == 0.0), "official policy uses zero diff mean"


@pytest.mark.parametrize("intervals", [(6,), (6, 24)])
def test_bridge_and_exporter_agree_across_interval_sets(fixture, intervals):
    """Interval-set field ordering no longer splits the two implementations."""
    bridge_norm = NormalizationContract.from_npz_dir(
        str(fixture["norm_dir"]), variables=VARIABLES,
        intervals=intervals, policy=POLICY_OFFICIAL_ZERO_DIFF_MEAN,
    )
    assert bridge_norm.identity_digest == exporter.compute_normalization_identity(
        fixture["norm_dir"], VARIABLES, intervals,
    )


def test_npz_content_hash_is_distinct_from_identity_digest(fixture):
    """The raw file hash and the effective-constants identity are separate."""
    identity = fixture["norm"].identity_digest
    content = compute_normalization_asset_sha256(fixture["norm_dir"])
    assert identity != content
    assert identity.startswith(NORM_IDENTITY_SCHEMA_VERSION + ":")
    assert len(content) == 64, "raw content hash is a full SHA-256"
    manifest_norm = fixture["manifest"]["normalization"]
    assert manifest_norm["identity_digest"] == identity
    assert manifest_norm["npz_content_sha256"] == content
    assert manifest_norm["identity_digest"] != manifest_norm["npz_content_sha256"]


def test_npz_content_change_without_effective_change_is_reported_separately(fixture):
    """Rewriting the NPZ files changes the content hash question, not identity."""
    before_identity = exporter.compute_normalization_identity(
        fixture["norm_dir"], VARIABLES, NORM_INTERVALS
    )
    before_content = compute_normalization_asset_sha256(fixture["norm_dir"])

    # Re-save one NPZ with a different compression footprint but same values.
    path = fixture["norm_dir"] / "normalize_mean.npz"
    data = {k: v for k, v in np.load(path).items()}
    np.savez_compressed(path, **data)

    after_identity = exporter.compute_normalization_identity(
        fixture["norm_dir"], VARIABLES, NORM_INTERVALS
    )
    after_content = compute_normalization_asset_sha256(fixture["norm_dir"])

    assert after_identity == before_identity, "effective constants did not change"
    assert after_content != before_content, "raw file bytes did change"


# =============================================================================
# 7. B13: PASS is committed late, revoked unconditionally, and cleanup is
#    classified onto its own axis -- all driven through the REAL entry points
# =============================================================================

def _run_dirs(fixture) -> List[Path]:
    base = fixture["root"] / "gate_out"
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))


def test_a_passing_run_records_its_verdict_as_committed(fixture):
    """The positive fixture's PASS goes through the commit check, not around it."""
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    assert result["s0_gate_pass"] is True
    assert result["verdict_committed"] is True
    assert result["status"] == "ok"
    assert (Path(result["_run_dir"]) / "PASS").exists()


def test_a_late_exception_revokes_a_provisional_pass(fixture, monkeypatch):
    """A failure AFTER the verdict is assembled must revoke it, not survive it.

    Drives the real `main()` over the fixture that otherwise passes, with the
    real post-verdict commit check made to fail. What is asserted is the
    published JSON, not a hand-built dict.
    """
    def explode(_result):
        raise RuntimeError("late failure after the verdict was assembled")

    monkeypatch.setattr(s0_gate, "_assert_gate_verdict_committable", explode)

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["s0_gate_pass"] is False
    assert result["status"] == "exception"
    assert result["verdict_committed"] is False
    assert "late failure after the verdict was assembled" in result["error"]
    # Every criterion still passed; only the commitment was revoked.
    assert all(result["gate_criteria"].values())
    run_dir = Path(result["_run_dir"])
    assert (run_dir / "s0_gate_result.json").exists()
    assert not (run_dir / "PASS").exists()


def test_the_commit_check_rejects_a_criterion_that_never_ran(fixture, monkeypatch):
    """A criterion left at its default instead of evaluated blocks the PASS."""
    real_check = s0_gate._assert_gate_verdict_committable
    seen: List[str] = []

    def drop_one(result):
        seen.append("called")
        result["criteria_details"].pop("outputs_finite", None)
        return real_check(result)

    monkeypatch.setattr(s0_gate, "_assert_gate_verdict_committable", drop_one)

    exit_code, result = run_gate_cli(fixture)

    assert seen == ["called"]
    assert exit_code == 1
    assert result["s0_gate_pass"] is False
    assert result["verdict_committed"] is False
    assert "never evaluated" in result["error"]
    assert "outputs_finite" in result["error"]
    assert not (Path(result["_run_dir"]) / "PASS").exists()


def test_the_commit_check_is_not_reached_when_a_criterion_failed(fixture, monkeypatch):
    """A run that already failed does not need -- or get -- a commit check."""
    calls: List[str] = []
    monkeypatch.setattr(
        s0_gate, "_assert_gate_verdict_committable",
        lambda result: calls.append("called"),
    )
    os.remove(fixture["ref_dir"] / "official_output_6h_4step.pt")

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["s0_gate_pass"] is False
    assert result["status"] == "gate_failed"
    assert calls == []


def test_cleanup_failure_does_not_destroy_an_already_computed_result(fixture, monkeypatch):
    """A failing cleanup step is a classified warning, never a lost run.

    Cleanup runs after the verdict is decided. Before this was contained, an
    exception raised in the `finally` block propagated out of `run_s0_gate`
    entirely, so a fully computed and fully passing result was thrown away.
    """
    def explode() -> None:
        raise RuntimeError("empty_cache blew up on a sick device")

    monkeypatch.setattr(s0_gate, "_empty_cuda_cache", explode)

    exit_code, result = run_gate_cli(fixture)

    # The verdict survives intact.
    assert exit_code == 0
    assert result["s0_gate_pass"] is True
    assert result["status"] == "ok"
    assert result.get("error") is None
    assert (Path(result["_run_dir"]) / "PASS").exists()

    # ...and the failure is recorded on its own axis, classified.
    warnings = result["cleanup_warnings"]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning["stage"] == "torch_cuda_empty_cache"
    assert warning["classification"] == "non_essential_cleanup"
    assert warning["affects_gate_verdict"] is False
    assert "empty_cache blew up on a sick device" in warning["error"]


def test_cleanup_failure_is_reported_in_the_markdown_on_its_own_axis(fixture, monkeypatch):
    """The report separates cleanup warnings from the gate verdict."""
    monkeypatch.setattr(
        s0_gate, "_empty_cuda_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("cache release failed")),
    )
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0

    report = (Path(result["_run_dir"]) / "S0_GATE_REPORT.md").read_text()
    assert "Cleanup Warnings (Do NOT Affect Gate Pass/Fail)" in report
    assert "torch_cuda_empty_cache" in report
    assert "non_essential_cleanup" in report
    assert "**Status:** PASS" in report


def test_every_cleanup_step_runs_even_when_an_earlier_one_fails(fixture, monkeypatch):
    """Cleanup steps are contained individually, not as one block."""
    monkeypatch.setattr(
        s0_gate, "_empty_cuda_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("second step failed")),
    )
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    stages = [w["stage"] for w in result["cleanup_warnings"]]
    # The first step still succeeded (it is absent from the warnings).
    assert stages == ["torch_cuda_empty_cache"]


def test_a_clean_run_records_no_cleanup_warnings(fixture):
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    assert result["cleanup_warnings"] == []


# =============================================================================
# 8. B14: the report formats only present, finite values, and can never
#    destroy the evidence the run already computed
# =============================================================================

def test_report_generation_failure_still_publishes_the_computed_result(fixture, monkeypatch):
    """A crash in report generation must not lose the run's JSON evidence."""
    monkeypatch.setattr(
        s0_gate, "generate_report",
        lambda result: (_ for _ in ()).throw(ValueError("report formatting blew up")),
    )

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 0
    assert result["s0_gate_pass"] is True
    assert all(result["gate_criteria"].values())
    assert "report formatting blew up" in result["report_generation_error"]

    run_dir = Path(result["_run_dir"])
    assert (run_dir / "s0_gate_result.json").exists()
    assert (run_dir / "PASS").exists()

    degraded = (run_dir / "S0_GATE_REPORT.md").read_text()
    assert "DEGRADED" in degraded
    assert "report formatting blew up" in degraded
    assert "upstream_parity: PASS" in degraded


def test_report_generation_failure_on_a_failing_run_still_publishes(fixture, monkeypatch):
    """The same holds when the gate itself failed: the evidence is published."""
    monkeypatch.setattr(
        s0_gate, "generate_report",
        lambda result: (_ for _ in ()).throw(TypeError("unsupported format string")),
    )
    os.remove(fixture["ref_dir"] / "official_output_6h_4step.pt")

    exit_code, result = run_gate_cli(fixture)

    assert exit_code == 1
    assert result["s0_gate_pass"] is False
    run_dir = Path(result["_run_dir"])
    assert (run_dir / "s0_gate_result.json").exists()
    assert not (run_dir / "PASS").exists()
    assert "DEGRADED" in (run_dir / "S0_GATE_REPORT.md").read_text()


def test_a_successful_report_records_no_report_error(fixture):
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    assert "report_generation_error" not in result
    assert "DEGRADED" not in (Path(result["_run_dir"]) / "S0_GATE_REPORT.md").read_text()


@pytest.mark.parametrize(
    "bad_value, expected",
    [
        (None, "N/A (missing)"),
        (float("nan"), "N/A (nan)"),
        (float("inf"), "N/A (+inf)"),
        (float("-inf"), "N/A (-inf)"),
        ("1e-5", "N/A (non-numeric: str)"),
        ({"value": 1e-5}, "N/A (non-numeric: dict)"),
    ],
)
def test_report_survives_a_missing_or_nonfinite_numeric_field(fixture, bad_value, expected):
    """Only present, finite numbers are formatted; everything else degrades.

    The result here is the REAL result a real run produced; exactly one numeric
    field is then replaced, so what is exercised is the real report path over a
    real result rather than a hand-built dictionary.
    """
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0

    details = result["criteria_details"]["upstream_parity"]
    assert isinstance(details["max_abs_diff"], float)  # the good value, first
    details["max_abs_diff"] = bad_value

    report = s0_gate.generate_report(result)  # must not raise
    assert f'- Max absolute diff: {expected}' in report


def test_report_still_renders_the_finite_fields_around_a_bad_one(fixture):
    """One unusable value does not blank out the rest of the report."""
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    result["criteria_details"]["upstream_parity"]["max_abs_diff"] = float("nan")

    report = s0_gate.generate_report(result)
    assert "N/A (nan)" in report
    assert "- Tolerance: 1e-05" in report
    assert "**Status:** PASS" in report
    assert "| upstream_parity | PASS |" in report


def test_report_formats_a_present_finite_value_unchanged(fixture):
    """The safe formatter must not alter the good path's output."""
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    result["criteria_details"]["upstream_parity"]["max_abs_diff"] = 1.25e-07

    report = s0_gate.generate_report(result)
    assert "- Max absolute diff: 1.25e-07" in report


def test_zero_edit_max_diff_is_formatted_safely(fixture):
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    result["criteria_details"]["zero_edit_equals_official"]["6h_1step"]["max_abs_diff"] = None

    report = s0_gate.generate_report(result)
    assert "6h_1step: max_diff=N/A (missing)" in report


def test_rmse_table_formats_only_finite_values(fixture):
    """The informational RMSE table degrades rather than raising."""
    exit_code, result = run_gate_cli(fixture)
    assert exit_code == 0
    result["rmse_sanity"] = {
        "6h": {"rmse_z500_weighted": float("nan")},
        "24h": {"rmse_z500_weighted": 41.5},
    }

    report = s0_gate.generate_report(result)
    assert "| 6h | N/A (nan) |" in report
    assert "| 24h | 41.5 |" in report


@pytest.mark.parametrize(
    "value, spec, expected",
    [
        (1.5, ".1f", "1.5"),
        (0, ".2e", "0.00e+00"),
        (-2.5e-8, ".2e", "-2.50e-08"),
        (np.float32(0.25), ".2f", "0.25"),
        (np.int64(7), ".1f", "7.0"),
        (True, ".1f", "N/A (non-numeric: bool)"),
        (None, ".4f", "N/A (missing)"),
        ([1.0], ".1f", "N/A (non-numeric: list)"),
    ],
)
def test_format_finite_unit_behaviour(value, spec, expected):
    assert s0_gate.format_finite(value, spec) == expected
