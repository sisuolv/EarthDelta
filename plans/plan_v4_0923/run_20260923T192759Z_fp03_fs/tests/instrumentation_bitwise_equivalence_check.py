"""Run one synthetic CPU Fs fit with whichever `earthdelta` is first on sys.path.

argv: <source_root> <out.pt>. Formal mode, lr=0.01, 40 updates, 4 samples,
4-step rollout: same code path as the real sweep, tiny model.
"""
import sys, hashlib
src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
import earthdelta.static_adapter as sa
assert sa.__file__.startswith(src), sa.__file__
sys.path.insert(0, src + "/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("r2cli", src + "/scripts/r2_fs_bank_train.py")
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
dev = torch.device("cpu")
bridge, variables, lat = cli.build_synthetic_bridge(dev, seed=20260921)
samples = cli.build_synthetic_samples(bridge, (4,), n_samples=4, device=dev, seed=20260921)
objective = sa.build_objective_spec(bridge, lat, lead_steps=(4,), space="raw")
blocks = (0, 1)
adapters = sa.build_fs_adapter(bridge.model.blocks[0].attn.proj.in_features, blocks,
                               rank_per_expert=4, seed=20260921)
config = sa.FsFitConfig(mode="formal", max_updates=40, learning_rate=0.01, train_steps=4,
                        lead_steps=(4,), target_blocks=blocks, rank_per_expert=4, seed=20260921)
adapters, record = sa.fit_static_adapter(bridge, samples, objective, config,
                                         fs_adapters=adapters, variables=variables)
state = {b: {k: v.clone() for k, v in l.state_dict().items()} for b, l in adapters.items()}
d = record.to_dict()
torch.save({"losses": d["losses"], "grad_norms": d["grad_norms"], "state": state,
            "eligible": d["eligible"], "quality_gate": d["quality_gate"],
            "param_norms": d.get("param_norms"), "update_norms": d.get("update_norms"),
            "initial_param_norm": d.get("initial_param_norm")}, out)
print("module", sa.__file__, "n_updates", d["n_updates"], "eligible", d["eligible"], d["quality_gate"]["reason"])
