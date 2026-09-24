"""FP-02 gap closed in FP-04: profile / horizon_check never move the bank.

`profile_training_step` runs a real optimizer step and `check_horizon_
feasibility` a real backward pass; both used to run on the SAME bank the
pipeline carries (and, in the synthetic chain, had just trained and saved).
The stages now measure a deep copy and record the bank digest before and
after. These tests pin that down on CPU:

  * the bank digest is identical before and after each stage (and after both);
  * the measurement is discriminating: the same profile call made directly on
    the bank DOES change its digest, so an unchanged digest is evidence;
  * the stage records the immutability block and fails if the bank moved;
  * without the certified Fs a real (non-synthetic) run refuses to measure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import bank_training as bt  # noqa: E402
from earthdelta import fs_protocol as fp  # noqa: E402
from earthdelta import static_adapter as sa  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402


@pytest.fixture(autouse=True)
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _args(**overrides):
    args = SimpleNamespace(
        synthetic=True, stage="profile", num_experts=4, rank_per_expert=4, seed=20260921,
        mode="gradient_check", max_updates=4, learning_rate=2.5e-3, train_steps=4, hold_steps=4,
        a0=0.25, rho=0.25, expert_index=0, profile_steps=1, profile_warmup=0,
        memory_budget_gib=80.0, memory_safety_fraction=0.8, max_seconds_per_update=1e6,
        max_differentiable_steps=None, output_dir=None, config_id=None, protocol_sha256=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _ctx(trained: bool = True):
    """A synthetic Fs-merged bridge and a bank that has actually been trained."""
    bridge, variables, lat = r2_cli.build_synthetic_bridge(torch.device("cpu"))
    blocks = (0, 1)
    hidden = bridge.model.blocks[0].attn.proj.in_features
    fs = sa.build_fs_adapter(hidden, blocks, rank_per_expert=4, seed=7, init_up_std=0.02)
    merged, artifact = sa.merge_static_adapter(bridge.model, fs, target_blocks=blocks)
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
    samples = r2_cli.build_synthetic_samples(bridge, [1, 4, 12], n_samples=4)
    spec = sa.build_objective_spec(bridge, lat, lead_steps=(4,), space="raw")
    bank = bt.build_dynamic_bank(hidden, blocks, num_experts=4, rank_per_expert=4, seed=20260921)
    ctx = {"bridge": bridge, "fs_bridge": fs_bridge, "fs_artifact": artifact,
           "fs_source": "certified_bundle", "variables": variables, "samples": samples,
           "objective": spec, "target_blocks": blocks, "lead_steps": (4,),
           "panel_lead_steps": (1, 4, 12), "record": {}, "dynamic_bank": bank, "bank_pins": {}}
    if trained:
        config = bt.BankTrainConfig(mode="gradient_check", max_updates=3, train_steps=4,
                                    hold_steps=4, lead_steps=(4,), num_experts=4,
                                    rank_per_expert=4, target_blocks=blocks, learning_rate=1e-2)
        bt.train_expert(fs_bridge, bank, 0, samples[:1], spec, config, variables=variables)
        assert not bt.bank_is_zero_initialized(bank, 0)
    return ctx


def test_profile_leaves_the_bank_digest_unchanged():
    ctx = _ctx()
    before = bt.bank_digest(ctx["dynamic_bank"])
    assert r2_cli.stage_profile(_args(), ctx) is True
    assert bt.bank_digest(ctx["dynamic_bank"]) == before
    immutability = ctx["record"]["profile"]["bank_immutability"]
    assert immutability["bank_unchanged"] is True and immutability["measured_on_a_deep_copy"]
    assert immutability["bank_digest_before"] == immutability["bank_digest_after"] == before
    # The copy DID take an optimizer step: the measurement was real.
    assert immutability["measured_copy_digest_after"] != before


def test_horizon_check_leaves_the_bank_digest_unchanged():
    ctx = _ctx()
    before = bt.bank_digest(ctx["dynamic_bank"])
    for lora in ctx["dynamic_bank"].values():  # drop train_expert's last-update grads
        for p in lora.parameters():
            p.grad = None
    flags_before = {name: p.requires_grad for b, lora in ctx["dynamic_bank"].items()
                    for name, p in lora.named_parameters()}
    assert r2_cli.stage_horizon_check(_args(stage="horizon_check"), ctx) is True
    assert bt.bank_digest(ctx["dynamic_bank"]) == before
    assert ctx["record"]["horizon_check"]["bank_immutability"]["bank_unchanged"] is True
    # Not even the requires_grad flags or a stray .grad were left on the real bank.
    assert {name: p.requires_grad for b, lora in ctx["dynamic_bank"].items()
            for name, p in lora.named_parameters()} == flags_before
    assert all(p.grad is None for lora in ctx["dynamic_bank"].values() for p in lora.parameters())


def test_profile_then_horizon_leaves_the_trained_bank_identical():
    ctx = _ctx()
    bank = ctx["dynamic_bank"]
    snapshot = {b: {k: v.clone() for k, v in lora.state_dict().items()} for b, lora in bank.items()}
    before = bt.bank_digest(bank)
    assert r2_cli.stage_profile(_args(), ctx)
    assert r2_cli.stage_horizon_check(_args(stage="horizon_check"), ctx)
    assert bt.bank_digest(bank) == before
    for block, lora in bank.items():
        for key, value in lora.state_dict().items():
            assert torch.equal(value, snapshot[block][key]), (block, key)


def test_the_immutability_check_is_discriminating():
    """Profiling the bank itself (the pre-FP-04 behaviour) does move it."""
    ctx = _ctx()
    bank = ctx["dynamic_bank"]
    before = bt.bank_digest(bank)
    config = bt.BankTrainConfig(mode="gradient_check", max_updates=1, train_steps=4, hold_steps=4,
                                lead_steps=(4,), num_experts=4, rank_per_expert=4,
                                target_blocks=(0, 1), learning_rate=1e-2)
    bt.profile_training_step(ctx["fs_bridge"], bank, ctx["samples"][0], ctx["objective"],
                             expert_index=0, config=config, variables=ctx["variables"])
    assert bt.bank_digest(bank) != before


def test_a_stage_that_moved_the_bank_fails(monkeypatch):
    """If the measured object were the real bank, the stage must say FAIL."""
    ctx = _ctx()
    real_deepcopy = r2_cli.copy.deepcopy
    monkeypatch.setattr(r2_cli.copy, "deepcopy", lambda obj, *a, **k: obj)  # sabotage the copy
    try:
        assert r2_cli.stage_profile(_args(), ctx) is False
        assert ctx["record"]["profile"]["bank_immutability"]["bank_unchanged"] is False
    finally:
        monkeypatch.setattr(r2_cli.copy, "deepcopy", real_deepcopy)


def test_the_untrained_initial_bank_is_also_unchanged_and_is_the_seed_bank():
    ctx = _ctx(trained=False)
    seed_digest = bt.bank_digest(bt.build_dynamic_bank(
        ctx["bridge"].model.blocks[0].attn.proj.in_features, (0, 1),
        num_experts=4, rank_per_expert=4, seed=20260921))
    assert bt.bank_digest(ctx["dynamic_bank"]) == seed_digest
    assert r2_cli.stage_profile(_args(), ctx) is True
    assert ctx["record"]["profile"]["bank_immutability"]["bank_digest_after"] == seed_digest


@pytest.mark.parametrize("stage", ["profile", "horizon_check"])
def test_a_real_measurement_without_the_certified_fs_is_refused(stage):
    ctx = _ctx(trained=False)
    ctx.pop("fs_bridge")
    ctx.pop("fs_source")
    handler = r2_cli.stage_profile if stage == "profile" else r2_cli.stage_horizon_check
    with pytest.raises(fp.ProtocolViolation) as excinfo:
        handler(_args(synthetic=False, stage=stage), ctx)
    assert excinfo.value.code == "FS_REFERENCE_NOT_AUTHORIZED"
