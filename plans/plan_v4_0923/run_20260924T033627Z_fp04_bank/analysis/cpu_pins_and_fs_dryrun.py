#!/usr/bin/env python3
"""FP-04 pre-registration pins, recomputed on CPU (no GPU, no model output on any issue).

1. The initial bank: `build_dynamic_bank(1024, 18..23, K=4, rank=4, seed=20260921)`
   -> bank_digest + the 4 expert digests.
2. A CPU dry run of the EXACT loading path of `r2_fs_bank_train.load_certified_fs`
   against the certified FP-03 bundle: authorize_certified_fs (hash binding, no
   protocol yet), then load the real checkpoint, the merged backbone and the
   adapter, and recompute merged / base / static-adapter / version digests. They
   must equal the bundle's own independent_reload.json record. No rollout is run.

Writes cpu_pins_and_fs_dryrun.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/mnt/afs/260010168/EarthDelta")
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import scripts.r2_fs_bank_train as r2  # noqa: E402
from earthdelta import bank_training as bt  # noqa: E402

HERE = Path(__file__).resolve().parent
FS = REPO / "plans/plan_v4_0923/run_20260924T013959Z_fp03_v2/certify"
GATE_CONFIG = REPO / "artifacts/ed-sprint8h-20260922T035313Z-rerun3/s0/gate_config.json"
BLOCKS = (18, 19, 20, 21, 22, 23)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mmap_load(original):
    """This CPU container is capped at 8 GiB: read the 5.5 GB checkpoint and the
    1.9 GB merged backbone memory-mapped. Same bytes, same tensors; only the
    residency changes (a GPU worker loads them normally)."""
    def load(f, *args, **kwargs):
        if isinstance(f, (str, Path)):
            kwargs.setdefault("mmap", True)
        return original(f, *args, **kwargs)
    return load


def main() -> None:
    torch.set_num_threads(16)
    torch.load = _mmap_load(torch.load)
    bank = bt.build_dynamic_bank(1024, BLOCKS, num_experts=4, rank_per_expert=4, seed=20260921)
    out = {"initial_bank": {
        "call": "build_dynamic_bank(1024, (18..23), num_experts=4, rank_per_expert=4, seed=20260921)",
        "bank_digest": bt.bank_digest(bank),
        "expert_digests": [bt.expert_digest(bank, k) for k in range(4)],
        "all_B_zero": all(bt.bank_is_zero_initialized(bank, k) for k in range(4))}}

    args = SimpleNamespace(
        stage="bank_train", fs_reference_manifest=FS / "fs/reference_manifest.json",
        fs_reference_manifest_sha256=sha(FS / "fs/reference_manifest.json"),
        fs_certify_decision=FS / "V2_certify_decision.json", fs_adapter=FS / "fs/fs_adapter.pt",
        fs_adapter_sha256=None, fs_merged_backbone=FS / "fs/fs_merged_backbone.pt",
        config=GATE_CONFIG, rank_per_expert=4)
    authorization = r2.authorize_certified_fs(args, None)
    bridge, variables, _, _ = r2.build_real_bridge(GATE_CONFIG, torch.device("cpu"))
    ctx = {"bridge": bridge, "target_blocks": BLOCKS, "protocol": None,
           "record": {"binding": {"fs_reference": authorization}}}
    loaded = r2.load_certified_fs(args, ctx)
    reload_record = json.loads((FS / "fs/independent_reload.json").read_text())
    post = reload_record["post_freeze"]
    recorded = {"merged_backbone_digest": post["artifact"]["merged_backbone_digest"],
                "base_backbone_digest": post["artifact"]["base_backbone_digest"],
                "static_adapter_digest": post["artifact"]["static_adapter_digest"],
                "artifact_version_digest": post["reference_identity"]["artifact_version_digest"]}
    out["certified_fs_cpu_dryrun"] = {
        "authorization": authorization,
        "loaded_identity": loaded["identity"],
        "identity_pass": loaded["identity_pass"],
        "recomputed_digests": loaded["digests"],
        "bundle_recorded_digests": recorded,
        "recomputed_equals_bundle_record": loaded["digests"] == recorded,
        "model_class": f"{type(bridge.model).__module__}.{type(bridge.model).__qualname__}",
        "note": "CPU, non-official model class is fine for digests: state_dict_digest reads "
                "parameter/buffer bytes only. No rollout, no sample, no loss.",
    }
    (HERE / "cpu_pins_and_fs_dryrun.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"initial_bank_digest": out["initial_bank"]["bank_digest"],
                      "identity_pass": loaded["identity_pass"],
                      "recomputed_equals_bundle_record": loaded["digests"] == recorded,
                      "digests": loaded["digests"]}, indent=1))


if __name__ == "__main__":
    main()
