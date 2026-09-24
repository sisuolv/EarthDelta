"""FP-04 (CPU, synthetic, REAL subprocess boundaries): the certified-Fs bank chain.

    fs_fit -> fs_freeze -> fs_verify -> r4_fs_decide certify      (the Fs bundle)
    -> 4x bank_train (B-J1 short step) + profile + horizon -> decide shortstep
    -> 4x bank_train (B-J2 formal) -> decide formal (BANK-QUAL-v1 / BANK-SELECT-v1)
    -> bank_assemble -> bank_verify (separate process) -> decide certify
    -> registry.verify_bank_bundle on the published bundle

Every CLI step is its own `python3 scripts/...` process: nothing crosses a stage
boundary in memory. Synthetic processes cannot produce the facts only a real
GPU run records (official backend, torch/xformers versions, S0, admission
consumer, protocol binding), so exactly those are injected into run_record.json
as in `tests/test_fs_decide_cli.py`; every Fs/bank identity is REAL.

Negative cases (plan section 7): a swapped Fs / source / normalization, one
tampered byte, a registry index/file mismatch, only three experts, a CPU/GPU
device mismatch, and continuing from F0 instead of Fs -- each is refused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import bank_training as bt  # noqa: E402
from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import registry as reg  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402
import scripts.r4_bank_decide as bank_decide  # noqa: E402

TRAIN = str(REPO_ROOT / "scripts" / "r2_fs_bank_train.py")
FS_DECIDE = str(REPO_ROOT / "scripts" / "r4_fs_decide.py")
BANK_DECIDE = str(REPO_ROOT / "scripts" / "r4_bank_decide.py")
SYN = ["--synthetic", "--device", "cpu", "--synthetic-samples", "8",
       "--panel-lead-steps", "1", "4", "12"]
#: Stand-ins for the S0 certificate / admission a real run consumes: small files
#: whose REAL sha256 the synthetic protocol pins (so bundle re-hashing works).
FAKE: Dict[str, str] = {}
LR = 0.0025
H_SHORT = 4
H_FORMAL = 8
SOURCE_PINS = ("earthdelta/bank_training.py", "earthdelta/registry.py",
               "scripts/r2_fs_bank_train.py")


def _env() -> Dict[str, str]:
    env = dict(os.environ)
    extra = [str(REPO_ROOT), str(REPO_ROOT / ".pydeps")]
    env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")])
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["OMP_NUM_THREADS"] = "1"
    return env


def run(argv: List[str], *, expect: int = None) -> subprocess.CompletedProcess:
    out = subprocess.run([sys.executable, "-u", *argv], capture_output=True, text=True,
                         env=_env(), cwd=str(REPO_ROOT), timeout=900)
    if expect is not None:
        assert out.returncode == expect, (argv[:3], out.returncode, out.stdout[-3000:],
                                          out.stderr[-3000:])
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inject(proc: Path, protocol_sha: str, config_id: str) -> None:
    """The facts only a real GPU process can record; every identity stays real."""
    path = proc / "run_record.json"
    record = json.loads(path.read_text())
    record["protocol_sha256"] = protocol_sha
    record["config_id"] = config_id
    binding = record["binding"]
    binding["s0"] = {"passed": True, "sha256": FAKE["s0_sha"]}
    binding["backend_postcondition"] = {"passed": True}
    binding["admission_consumer"] = {"passed": True}
    binding["fs_reference"]["bound_to_protocol"] = True
    prov = record["provenance"]
    prov["official_backend"] = True
    prov["versions"].update(torch="2.3.1+cu121", xformers="0.0.27")
    prov["tf32"] = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "env": {}}
    prov["input_files"]["admission"] = {"sha256": FAKE["admission_sha"]}
    path.write_text(json.dumps(record))


# =============================================================================
# The certified synthetic Fs bundle (FP-03 chain, subprocesses)
# =============================================================================

def build_certified_fs(root: Path) -> Dict[str, Path]:
    replica = root / "fs" / "replica0"
    run([TRAIN, "--stage", "fs_fit", "--mode", "formal", "--max-updates", "8",
         "--learning-rate", str(LR), *SYN, "--output-dir", str(replica)])
    adapter = replica / "fs_adapter.pt"
    sha = _sha(adapter)
    rows = [{"config_id": "V2_P", "device_index": i, "dir": str(replica),
             "qualification": {"verdict": "PASS"},
             "files": {"fs_adapter.pt": {"path": str(adapter), "sha256": sha}}} for i in range(4)]
    formal = root / "fs" / "formal_decision.json"
    formal.write_text(json.dumps({
        "kind": "formal", "rule": "FS-SELECT-v1",
        "verdict": "QUALITY_PASS_PENDING_DESIGNATED_GATES", "horizon": 8,
        "designated_adapter_sha256": sha, "designated_record": {"mode": "formal", "n_updates": 8},
        "details": rows}))
    freeze, verify = root / "fs" / "freeze", root / "fs" / "verify"
    run([TRAIN, "--stage", "fs_freeze", *SYN, "--fs-adapter", str(adapter), "--fs-adapter-sha256",
         sha, "--fs-decision", str(formal), "--output-dir", str(freeze)], expect=0)
    run([TRAIN, "--stage", "fs_verify", *SYN, "--fs-merged-backbone",
         str(freeze / "fs_merged_backbone.pt"), "--fs-adapter", str(adapter),
         "--fs-adapter-sha256", sha, "--output-dir", str(verify)], expect=0)
    fs_protocol = root / "fs" / "fs_protocol.json"
    fs_protocol.write_text(json.dumps({
        "schema_version": fp.PROTOCOL_SCHEMA, "qualification_rule": {"horizon": 8},
        "selection_rule": {"n_replicas": 4, "designated_device_index": 0},
        "merge_equivalence_tolerance": {"atol_floor": 1e-5, "atol_relative": 1.5e-3, "rtol": 1e-5},
        "inputs": {"checkpoint_sha256": "synthetic-ckpt", "normalization_identity": "synthetic-norm"},
        "limitations": ["synthetic"], "configs": {}}))
    certify = root / "fs" / "certify" / "certify_decision.json"
    run([FS_DECIDE, "--kind", "certify", "--protocol", str(fs_protocol), "--protocol-sha256",
         _sha(fs_protocol), "--formal-decision", str(formal), "--freeze-dir", str(freeze),
         "--verify-dir", str(verify), "--out", str(certify)], expect=0)
    assert json.loads(certify.read_text())["verdict"] == "FS_SELECTED"
    bundle = certify.parent / "fs"
    return {"bundle": bundle, "certify": certify, "manifest": bundle / "reference_manifest.json",
            "adapter": bundle / "fs_adapter.pt", "merged": bundle / "fs_merged_backbone.pt",
            "fs_protocol": fs_protocol, "replica_adapter": adapter}


def fs_flags(fs: Dict[str, Path]) -> List[str]:
    return ["--fs-reference-manifest", str(fs["manifest"]), "--fs-reference-manifest-sha256",
            _sha(fs["manifest"]), "--fs-certify-decision", str(fs["certify"]),
            "--fs-adapter", str(fs["adapter"]), "--fs-merged-backbone", str(fs["merged"])]


def synthetic_pins(tmp: Path) -> Dict:
    """Initial bank, grouping and probe pins, recomputed here on CPU."""
    bridge, _, _ = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    samples = r2_cli.build_synthetic_samples(bridge, [1, 4, 12], n_samples=8)
    grouping = bt.assign_diversity_groups(samples, 4, hold_steps=4)
    hidden = bridge.model.blocks[0].attn.proj.in_features
    blocks = tuple(range(len(bridge.model.blocks)))
    initial = bt.build_dynamic_bank(hidden, blocks, num_experts=4, rank_per_expert=4,
                                    seed=20260921)
    pins = {"initial_bank_digest": bt.bank_digest(initial),
            "initial_expert_digests": [bt.expert_digest(initial, k) for k in range(4)],
            "grouping": {str(g.expert_index): list(g.issue_ids) for g in grouping.groups},
            "probe_issue_id": samples[0].issue_id}
    path = tmp / "grouping_pins.json"
    path.write_text(json.dumps(pins))
    return {**pins, "_file": path, "target_blocks": list(blocks)}


def bank_protocol(tmp: Path, fs: Dict[str, Path], pins: Dict, *, name="bank_protocol.json",
                  source: Dict[str, str] = None) -> Dict:
    reload = json.loads((fs["bundle"] / "independent_reload.json").read_text())
    post = reload["post_freeze"]
    manifest = json.loads(fs["manifest"].read_text())
    fs_reference = {
        "reference_manifest": {"path": str(fs["manifest"]), "sha256": _sha(fs["manifest"])},
        "certify_decision": {"path": str(fs["certify"]), "sha256": _sha(fs["certify"])},
        "fs_adapter": {"path": str(fs["adapter"]), "sha256": _sha(fs["adapter"])},
        "fs_merged_backbone": {"path": str(fs["merged"]), "sha256": _sha(fs["merged"])},
        "fs_protocol_sha256": manifest["protocol_sha256"],
        "digests": {"merged_backbone_digest": post["artifact"]["merged_backbone_digest"],
                    "base_backbone_digest": post["artifact"]["base_backbone_digest"],
                    "static_adapter_digest": post["artifact"]["static_adapter_digest"],
                    "artifact_version_digest": post["reference_identity"]["artifact_version_digest"]},
    }
    common = {"seed": 20260921, "rank_per_expert": 4, "num_experts": 4,
              "target_blocks": pins["target_blocks"], "hold_steps": 4, "a0": 0.25, "rho": 0.25,
              "data_role": "bank_fit", "device": "cpu", "limit": None,
              "train_steps": 4, "lead_steps": [4], "panel_lead_steps": [1, 4, 12]}
    configs = {}
    for k in range(4):
        configs[f"BJ1_E{k}"] = {"job": "B-J1", "stage": "bank_train", "mode": "gradient_check",
                                "max_updates": H_SHORT, "learning_rate": LR, "expert_index": k,
                                "_output_subdir": f"expert{k}", **common}
        configs[f"BJ2_E{k}"] = {"job": "B-J2", "stage": "bank_train", "mode": "formal",
                                "max_updates": H_FORMAL, "learning_rate": LR, "expert_index": k,
                                "_output_subdir": f"expert{k}", **common}
    configs["BJ1_PROFILE"] = {"job": "B-J1", "stage": "profile", "mode": "gradient_check",
                              "max_updates": H_SHORT, "learning_rate": LR, "expert_index": 0,
                              "_output_subdir": "profile", **common}
    configs["BJ1_HORIZON"] = {"job": "B-J1", "stage": "horizon_check", "mode": "gradient_check",
                              "max_updates": H_SHORT, "learning_rate": LR, "expert_index": 0,
                              "_output_subdir": "horizon", **common}
    configs["BJ3_ASSEMBLE"] = {"job": "B-J3", "stage": "bank_assemble", "_output_subdir": "assemble",
                               **common}
    configs["BJ3_VERIFY"] = {"job": "B-J3", "stage": "bank_verify", "_output_subdir": "verify",
                             **common}
    protocol = {
        "schema_version": fp.PROTOCOL_SCHEMA,
        "protocol_kind": "bank",
        "protocol_id": "TEST-FP04-BANK",
        "design": {"num_experts": 4, "rank_per_expert": 4, "a0": 0.25, "rho": 0.25,
                   "hold_steps": 4, "target_blocks": pins["target_blocks"]},
        "qualification_rule": {
            "rule": "BANK-QUAL-v1", "horizon": H_FORMAL,
            "e0": {"torch_version": "2.3.1+cu121", "xformers_version": "0.0.27",
                   "require_official_backend": True},
            "e2": {"clip_events_max": 0, "per_visit_ratio_max": 1.25, "epoch_onset_factor": 1.02,
                   "epoch_size": "group_size"},
            "e3": {"own_group_mean_ratio_24h_lt": 1.0}},
        "shortstep_rule": {"horizon": H_SHORT, "n_workers": 4, "require_real_cuda_profile": False},
        "selection_rule": {"rule": "BANK-SELECT-v1", "num_experts": 4},
        "inputs": {
            "s0_certificate": {"path": FAKE["s0_path"], "sha256": FAKE["s0_sha"]},
            "admission": {"path": FAKE["admission_path"], "sha256": FAKE["admission_sha"]},
            "grouping": {"path": str(pins["_file"]), "sha256": _sha(pins["_file"])},
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "normalization_identity": manifest["normalization_identity"],
            "fs_reference": fs_reference,
            "bank": {k: pins[k] for k in ("initial_bank_digest", "initial_expert_digests",
                                          "grouping", "probe_issue_id")},
        },
        "configs": configs,
        "limitations": ["synthetic test protocol"],
    }
    if source is not None:
        protocol["source_at_preregistration"] = source
    path = tmp / name
    path.write_text(json.dumps(protocol))
    return {"path": path, "sha": _sha(path), "protocol": protocol}


def bank_train_argv(cfg: Dict, flags: List[str], out: Path) -> List[str]:
    return [TRAIN, "--stage", cfg["stage"], "--expert-index", str(cfg["expert_index"]),
            "--mode", cfg["mode"], "--max-updates", str(cfg["max_updates"]),
            "--learning-rate", str(cfg["learning_rate"]), *SYN, *flags, "--output-dir", str(out)]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root = tmp_path_factory.mktemp("fp04_chain")
    for name in ("s0", "admission"):
        stand_in = root / f"{name}_stand_in.json"
        stand_in.write_text(json.dumps({"synthetic_stand_in_for": name}))
        FAKE[f"{name}_path"] = str(stand_in)
        FAKE[f"{name}_sha"] = _sha(stand_in)
    fs = build_certified_fs(root)
    pins = synthetic_pins(root)
    source = {rel: _sha(REPO_ROOT / rel) for rel in SOURCE_PINS}
    proto = bank_protocol(root, fs, pins, source=source)
    flags = fs_flags(fs)
    cfgs = proto["protocol"]["configs"]

    # ---- B-J1: short step + profile + horizon -----------------------------
    bj1 = root / "BJ1"
    for k in range(4):
        cfg = cfgs[f"BJ1_E{k}"]
        run(bank_train_argv(cfg, flags, bj1 / f"expert{k}"), expect=0)
        inject(bj1 / f"expert{k}", proto["sha"], f"BJ1_E{k}")
    for stage, cid in (("profile", "BJ1_PROFILE"), ("horizon_check", "BJ1_HORIZON")):
        cfg = cfgs[cid]
        out = bj1 / cfg["_output_subdir"]
        run([TRAIN, "--stage", stage, "--mode", cfg["mode"], "--max-updates", str(cfg["max_updates"]),
             "--learning-rate", str(LR), *SYN, *flags, "--output-dir", str(out)], expect=0)
        inject(out, proto["sha"], cid)
    short = root / "decisions" / "BJ1_shortstep.json"
    run([BANK_DECIDE, "--kind", "shortstep", "--job", "B-J1", "--protocol", str(proto["path"]),
         "--protocol-sha256", proto["sha"], "--job-dir", str(bj1), "--out", str(short)], expect=0)

    # ---- B-J2: formal -------------------------------------------------------
    bj2 = root / "BJ2"
    for k in range(4):
        run(bank_train_argv(cfgs[f"BJ2_E{k}"], flags, bj2 / f"expert{k}"), expect=0)
        inject(bj2 / f"expert{k}", proto["sha"], f"BJ2_E{k}")
    formal = root / "decisions" / "BJ2_formal.json"
    run([BANK_DECIDE, "--kind", "formal", "--job", "B-J2", "--protocol", str(proto["path"]),
         "--protocol-sha256", proto["sha"], "--job-dir", str(bj2), "--shortstep-decision",
         str(short), "--out", str(formal)], expect=0)

    # ---- B-J3: assemble, independent verify, certify -----------------------
    bj3 = root / "BJ3"
    run([TRAIN, "--stage", "bank_assemble", *SYN, *flags, "--bank-decision", str(formal),
         "--output-dir", str(bj3 / "assemble")], expect=0)
    inject(bj3 / "assemble", proto["sha"], "BJ3_ASSEMBLE")
    run([TRAIN, "--stage", "bank_verify", *SYN, *flags, "--bank-decision", str(formal),
         "--bank-assembly", str(bj3 / "assemble" / "bank_assembly.json"),
         "--output-dir", str(bj3 / "verify")], expect=0)
    inject(bj3 / "verify", proto["sha"], "BJ3_VERIFY")
    certify = root / "certify" / "BJ3_certify.json"
    run([BANK_DECIDE, "--kind", "certify", "--protocol", str(proto["path"]), "--protocol-sha256",
         proto["sha"], "--formal-decision", str(formal), "--assemble-dir", str(bj3 / "assemble"),
         "--verify-dir", str(bj3 / "verify"), "--out", str(certify)], expect=0)
    return {"root": root, "fs": fs, "pins": pins, "proto": proto, "flags": flags,
            "short": short, "formal": formal, "certify": certify, "bj1": bj1, "bj2": bj2,
            "bj3": bj3, "bundle": root / "certify" / "bank", "source": source}


# =============================================================================
# The positive chain
# =============================================================================

def test_shortstep_decision_passes_on_machinery_only(chain):
    decision = json.loads(chain["short"].read_text())
    assert decision["verdict"] == "PASS", json.dumps(decision, indent=1)[:4000]
    assert decision["reads_no_training_effect_number"] is True
    assert decision["across_workers"] == {"all_workers_present": True, "one_initial_bank": True,
                                          "one_grouping": True, "one_shared_certified_fs": True}
    for worker in decision["workers"]:
        assert worker["checks"]["A_grad_zero_at_update0"] is True
        assert worker["checks"]["B_grad_positive_at_update0"] is True
        assert worker["validity_facts"]["no_source_drift"] is True
    assert decision["measurements"]["profile"]["checks"]["bank_unchanged"] is True
    assert decision["measurements"]["horizon_check"]["checks"]["bank_unchanged"] is True


def test_every_worker_loaded_the_same_certified_fs_and_initial_bank(chain):
    digests = set()
    for k in range(4):
        diag = json.loads((chain["bj2"] / f"expert{k}" / f"bank_diagnostics_expert{k}.json").read_text())
        assert diag["fs_source"] == "certified_bundle"
        assert diag["fs_identity"]["identity_pass"] is True
        digests.add(diag["fs_identity"]["digests"]["merged_backbone_digest"])
        assert diag["initial_bank_digest"] == chain["pins"]["initial_bank_digest"]
        assert diag["experts"][str(k)]["untrained_L0_equals_Fs_bitwise"] is True
        run_record = json.loads((chain["bj2"] / f"expert{k}" / "run_record.json").read_text())
        assert run_record["binding"]["fs_reference"]["passed"] is True
    assert len(digests) == 1


def test_formal_decision_selects_the_whole_bank_for_assembly(chain):
    decision = json.loads(chain["formal"].read_text())
    assert decision["verdict"] == "BANK_QUAL_PASS_PENDING_ASSEMBLY", \
        {r["expert_index"]: (r["qualification"].get("status"), r["qualification"].get("e0", {}).get("failed"))
         for r in decision["details"]}
    assert decision["substitution"] == "FORBIDDEN" and decision["partial_bank"] == "FORBIDDEN"
    assert [e["expert_index"] for e in decision["experts_for_assembly"]] == [0, 1, 2, 3]
    assert len({e["expert_digest"] for e in decision["experts_for_assembly"]}) == 4
    for row in decision["details"]:
        assert row["qualification"]["verdict"] == "PASS"
        assert row["report_only_prefix_equals_shortstep"] is True


def test_assembly_is_exact_and_the_bundle_is_certified(chain):
    verify = json.loads((chain["bj3"] / "verify" / "bank_verify.json").read_text())
    evaluation = bt.evaluate_bank_assembly(verify, num_experts=4)
    assert evaluation["passed"] is True, evaluation
    for k in map(str, range(4)):
        assert set(verify["A1_source_bank"][k]["max_abs_diff_by_step"].values()) == {0.0}
        assert set(verify["A2_training_probe"][k]["max_abs_diff_by_step"].values()) == {0.0}
        assert set(verify["A4_continuation"][k]["max_abs_diff_vs_fs_continuation_by_step"]) == {0.0}
        assert min(verify["A4_continuation"][k]["max_abs_diff_vs_f0_continuation_by_step"]) > 0.0
    assert verify["A3_zero_edit"]["max_abs_diff_vs_fs"] == 0.0
    assert verify["A3_zero_edit"]["max_abs_diff_vs_f0"] > 0.0
    final = json.loads(chain["certify"].read_text())
    assert final["verdict"] == "BANK_CERTIFIED" and final["bank_certified"] is True
    for name in reg_files():
        assert (chain["bundle"] / name).is_file(), name
    report = reg.verify_bank_bundle(chain["bundle"], expected_protocol_sha256=chain["proto"]["sha"],
                                    expected_source=chain["source"])
    assert report["passed"] is True and report["num_experts"] == 4
    # A CPU reload of bank.pt equals every expert file slice exactly.
    bank, _ = bt.load_bank(chain["bundle"] / "bank.pt")
    for k in range(4):
        loaded = bt.load_expert(chain["bundle"] / f"expert_{k}.pt", expected_expert_index=k)
        for block, (down, up) in loaded["factors"].items():
            assert torch.equal(bank[block].down[k].weight, down)
            assert torch.equal(bank[block].up[k].weight, up)


def reg_files():
    return ["bank.pt", "bank_manifest.json", "registry.json", "bank_assembly.json",
            "bank_verify.json", "qualification.json"] + [f"expert_{k}.pt" for k in range(4)] + \
        [f"probe_expert_{k}.pt" for k in range(4)]


# =============================================================================
# Negative cases
# =============================================================================

def _bundle_copy(chain, tmp_path: Path) -> Path:
    target = tmp_path / "bank"
    shutil.copytree(chain["bundle"], target)
    return target


def _codes(excinfo) -> set:
    return {f["code"] for f in excinfo.value.detail["failures"]}


def test_a_swapped_fs_is_refused_before_anything_loads(chain, tmp_path):
    """A different (non-certified) Fs adapter / manifest cannot feed a bank stage."""
    other = tmp_path / "other_adapter.pt"
    adapters = sa.build_fs_adapter(64, (0, 1), rank_per_expert=4, seed=5, init_up_std=0.01)
    torch.save({b: {k: v.cpu() for k, v in l.state_dict().items()} for b, l in adapters.items()}, other)
    flags = list(chain["flags"])
    flags[flags.index("--fs-adapter") + 1] = str(other)
    cfg = chain["proto"]["protocol"]["configs"]["BJ1_E0"]
    out = tmp_path / "swapped"
    result = run(bank_train_argv(cfg, flags, out))
    assert result.returncode == 2
    refused = json.loads((out / "binding_refused.json").read_text())
    assert refused["code"] == "FS_REFERENCE_NOT_AUTHORIZED"
    assert not (out / "bank_training_expert0.json").exists()
    # ... and a bundle can be checked against the Fs a consumer expects.
    with pytest.raises(reg.RegistryViolation) as excinfo:
        reg.verify_bank_bundle(chain["bundle"], expected_fs={"fs_adapter": _sha(other)})
    assert "FS_NOT_EXPECTED" in _codes(excinfo)


def test_missing_certified_fs_flags_are_refused_for_every_bank_stage(tmp_path):
    for stage in r2_cli.CERTIFIED_FS_STAGES:
        out = tmp_path / stage
        result = run([TRAIN, "--stage", stage, *SYN, "--output-dir", str(out)])
        assert result.returncode == 2, stage
        assert json.loads((out / "binding_refused.json").read_text())["code"] == \
            "FS_REFERENCE_NOT_AUTHORIZED"


def test_a_swapped_source_or_normalization_is_refused(chain):
    with pytest.raises(reg.RegistryViolation) as excinfo:
        reg.verify_bank_bundle(chain["bundle"], expected_source={
            **chain["source"], "earthdelta/bank_training.py": "0" * 64})
    assert "SOURCE_NOT_EXPECTED" in _codes(excinfo)
    with pytest.raises(reg.RegistryViolation) as excinfo:
        reg.verify_bank_bundle(chain["bundle"], expected_normalization_identity="ed-norm-identity/1:other")
    assert "NORMALIZATION_NOT_EXPECTED" in _codes(excinfo)


def test_a_protocol_with_another_normalization_refuses_the_certified_fs(chain, tmp_path):
    """Real-path authorization: the protocol must name the Fs's normalization identity."""
    protocol = copy.deepcopy(chain["proto"]["protocol"])
    protocol["inputs"]["normalization_identity"] = "ed-norm-identity/1:not-this-one"
    args = r2_cli.build_parser().parse_args(
        ["--stage", "bank_train", "--synthetic", *chain["flags"]])
    with pytest.raises(fp.ProtocolViolation) as excinfo:
        r2_cli.authorize_certified_fs(args, protocol)
    assert excinfo.value.code == "FS_REFERENCE_NOT_AUTHORIZED"
    assert any("normalization" in f for f in excinfo.value.detail["failures"])
    # The unmodified protocol authorizes the same flags.
    assert r2_cli.authorize_certified_fs(args, chain["proto"]["protocol"])["passed"] is True


@pytest.mark.parametrize("victim", ["expert_2.pt", "bank.pt", "registry.json"])
def test_one_tampered_byte_is_refused(chain, tmp_path, victim):
    bundle = _bundle_copy(chain, tmp_path)
    path = bundle / victim
    data = bytearray(path.read_bytes())
    position = len(data) // 2 if victim.endswith(".pt") else data.index(b"expert_1")
    data[position] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(reg.RegistryViolation) as excinfo:
        reg.verify_bank_bundle(bundle)
    codes = _codes(excinfo)
    assert codes & {"FILE_SHA256_MISMATCH", "REGISTRY_SHA256_MISMATCH"}, codes
    if victim == "expert_2.pt":
        with pytest.raises(bt.BankTrainingViolation):
            bt.load_expert(path, expected_sha256=_sha(chain["bundle"] / victim))


def test_a_registry_index_file_mismatch_is_refused(chain, tmp_path):
    """expert_1's row pointing at expert_2's file is caught even with a re-signed registry."""
    bundle = _bundle_copy(chain, tmp_path)
    registry_path = bundle / "registry.json"
    payload = json.loads(registry_path.read_text())
    entries = {e["plan_id"]: e for e in payload["entries"]}
    entries["expert_1"]["artifact_ref"] = dict(entries["expert_2"]["artifact_ref"])
    registry_path.write_text(json.dumps(payload))
    manifest_path = bundle / "bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["registry"]["sha256"] = _sha(registry_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(reg.RegistryViolation) as excinfo:
        reg.verify_bank_bundle(bundle)
    codes = _codes(excinfo)
    assert {"REGISTRY_ARTIFACT_REF_MISMATCH", "REGISTRY_INDEX_FILE_MISMATCH"} <= codes, codes


def test_only_three_experts_are_refused_everywhere(chain, tmp_path):
    formal = json.loads(chain["formal"].read_text())
    three = [e for e in formal["experts_for_assembly"] if e["expert_index"] != 3]
    # BANK-SELECT-v1 on three experts
    experts = [{"expert_index": e["expert_index"], "qualification": {"verdict": "PASS"},
                "expert_digest": e["expert_digest"], "expert_file": e["expert_file"],
                "batch_id": "B-J2"} for e in three]
    assert bt.decide_bank_formal(experts, {"num_experts": 4})["verdict"] == "INVALID"
    # assembly from a decision naming three experts
    tampered = tmp_path / "three.json"
    tampered.write_text(json.dumps({**formal, "experts_for_assembly": three}))
    out = tmp_path / "assemble3"
    result = run([TRAIN, "--stage", "bank_assemble", *SYN, *chain["flags"], "--bank-decision",
                  str(tampered), "--output-dir", str(out)])
    assert result.returncode != 0 and not (out / "bank.pt").exists()
    # the library refuses the partial bank directly as well
    initial = bt.build_dynamic_bank(64, (0, 1), num_experts=4, rank_per_expert=4, seed=20260921)
    loaded = {e["expert_index"]: bt.load_expert(e["expert_file"]["path"],
                                                expected_expert_index=e["expert_index"])
              for e in three}
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.assemble_bank(initial, loaded)
    assert excinfo.value.code == "BANK_ASSEMBLY_INCOMPLETE"
    # a bundle whose binding lists three experts is refused
    bundle = _bundle_copy(chain, tmp_path)
    manifest = json.loads((bundle / "bank_manifest.json").read_text())
    manifest["binding"]["experts"] = manifest["binding"]["experts"][:3]
    (bundle / "bank_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(reg.RegistryViolation) as excinfo:
        reg.verify_bank_bundle(bundle, rehash_external=False)
    assert "BINDING_EXPERT_SET" in _codes(excinfo)


def test_a_cpu_gpu_device_mismatch_is_refused(chain):
    bank, _ = bt.load_bank(chain["bundle"] / "bank.pt")
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.check_bank_device(bank, torch.device("cuda:0"))
    assert excinfo.value.code == "BANK_DEVICE_MISMATCH"
    # a rollout helper refuses a bank that is not on the backbone's device
    bridge, variables, _ = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    for lora in bank.values():
        lora.to("meta")
    plan = bt.expert_plan(bt.build_bank_registry(4), 0)
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.singleton_probe_states(bridge, bank, plan, torch.zeros(1, 8, 16, 32), variables,
                                  target_blocks=(0, 1))
    assert excinfo.value.code == "BANK_DEVICE_MISMATCH"
    meta_bank = bt.build_dynamic_bank(64, (0, 1), num_experts=4, rank_per_expert=4, seed=1,
                                      device="meta")
    with pytest.raises(bt.BankTrainingViolation) as excinfo:
        bt.check_bank_device(meta_bank, "cpu", context="reloaded bank vs CPU backbone")
    assert excinfo.value.code == "BANK_DEVICE_MISMATCH"


def test_continuing_from_f0_instead_of_fs_is_refused(chain, tmp_path):
    """A4 must discriminate: an 'Fs' continuation that is really F0 fails, and a
    verify record whose F0 comparison collapsed stops the certification."""
    verify = json.loads((chain["bj3"] / "verify" / "bank_verify.json").read_text())
    broken = copy.deepcopy(verify)
    broken["A4_continuation"]["2"]["max_abs_diff_vs_f0_continuation_by_step"] = \
        [0.0] * len(broken["A4_continuation"]["2"]["max_abs_diff_vs_f0_continuation_by_step"])
    evaluation = bt.evaluate_bank_assembly(broken, num_experts=4)
    assert evaluation["passed"] is False and evaluation["failed"] == ["A4"]
    # the decider refuses it too (new job dir with the broken record)
    verify_dir = tmp_path / "verify"
    shutil.copytree(chain["bj3"] / "verify", verify_dir)
    (verify_dir / "bank_verify.json").write_text(json.dumps(broken))
    out = tmp_path / "certify" / "decision.json"
    run([BANK_DECIDE, "--kind", "certify", "--protocol", str(chain["proto"]["path"]),
         "--protocol-sha256", chain["proto"]["sha"], "--formal-decision", str(chain["formal"]),
         "--assemble-dir", str(chain["bj3"] / "assemble"), "--verify-dir", str(verify_dir),
         "--out", str(out)], expect=0)
    decision = json.loads(out.read_text())
    assert decision["verdict"] == "STOP_ASSEMBLY" and not (tmp_path / "certify" / "bank").exists()
    # functionally: the real continuation check with F0 standing in for Fs fails
    bridge, variables, _ = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    bank, _ = bt.load_bank(chain["bundle"] / "bank.pt")
    x = r2_cli.build_synthetic_samples(bridge, [1, 4, 12], n_samples=1)[0].x_norm
    result = sa.verify_continuation_is_fs(bridge, bridge, bank, x, variables, hold=4, total=12,
                                          active_expert=1, target_blocks=(0, 1))
    assert result["passed"] is False and result["discriminates_f0"] is False


def test_the_assembly_equivalence_has_no_tolerance(chain):
    verify = json.loads((chain["bj3"] / "verify" / "bank_verify.json").read_text())
    for check, mutate in (
        ("A1", lambda r: r["A1_source_bank"]["0"]["max_abs_diff_by_step"].update({"4": 1e-12})),
        ("A2", lambda r: r["A2_training_probe"]["3"]["max_abs_diff_by_step"].update({"12": 5e-9})),
        ("A3", lambda r: r["A3_zero_edit"].update(max_abs_diff_vs_fs=1e-30)),
        ("A5", lambda r: r["A5_reload"].update(bank_digest="0" * 64)),
    ):
        broken = copy.deepcopy(verify)
        mutate(broken)
        evaluation = bt.evaluate_bank_assembly(broken, num_experts=4)
        assert evaluation["failed"] == [check], (check, evaluation)


def test_decisions_are_written_once_and_protocol_bound(chain, tmp_path):
    assert run([BANK_DECIDE, "--kind", "formal", "--job", "B-J2", "--protocol",
                str(chain["proto"]["path"]), "--protocol-sha256", chain["proto"]["sha"],
                "--job-dir", str(chain["bj2"]), "--out", str(chain["formal"])]).returncode == 2
    assert run([BANK_DECIDE, "--kind", "formal", "--job", "B-J2", "--protocol",
                str(chain["proto"]["path"]), "--protocol-sha256", "0" * 64,
                "--job-dir", str(chain["bj2"]), "--out", str(tmp_path / "x.json")]).returncode == 2
    assert not (tmp_path / "x.json").exists()
