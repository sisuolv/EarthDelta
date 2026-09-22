"""Fs fit / freeze / merge / post-freeze-reference tests (CPU, tiny synthetic).

Scale follows the `plans/plans_v2_0921/cci/gpu_bank_check.py` precedent: a
16x32 grid, 8 variables, hidden 32, depth 2, patch size 2. Everything here is
synthetic and proves only that the code paths are structurally correct; none of
it is a weather result.

Five properties are asserted:

  1. An Fs fit on bank_fit data reduces the native-registered objective.
  2. The freeze-time merge is numerically exact -- the merged backbone forward
     equals the always-on ExpertLoRA branch, checked both at the projection
     layer (float64, exact algebra) and through a whole rollout (float32).
  3. Post-freeze zero-edit equivalence holds against **Fs**, and the check is
     DISCRIMINATING: the same comparison written against F0 would fail. There
     is a companion test for the untrained-Fs case, where the check honestly
     reports that it cannot tell Fs from F0.
  4. `static_adapter` stops being the placeholder "none" and names Fs's digest,
     and the re-recorded identity no longer matches the F0 one.
  5. F0's background score is carried under its own tag, is reporting-only, and
     cannot be used as a gain baseline.

Plus the update caps (<=32 gradient check, <=500 formal) are enforced by the
config object rather than by convention.
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple
from unittest import mock

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta import static_adapter as sa  # noqa: E402
from earthdelta.bridge import (  # noqa: E402
    DEFAULT_VARIABLES,
    NormalizationContract,
    Stormer,
    WeatherStepBridge,
    controlled_rollout,
)
from earthdelta.contracts import ArtifactVersion, EditPlan  # noqa: E402
from earthdelta.lowrank import ExpertLoRA  # noqa: E402

import scripts.r2_fs_bank_train as r2_cli  # noqa: E402


# =============================================================================
# Fixture geometry
# =============================================================================

GRID = (16, 32)
N_VARS = 8
HIDDEN = 32
DEPTH = 2
TARGET_BLOCKS: Tuple[int, ...] = (0, 1)
FS_RANK = 2
BANK_EXPERTS = 4
BANK_RANK = 2


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(20260921)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def variables() -> List[str]:
    return list(DEFAULT_VARIABLES[:N_VARS])


def _make_backbone(variables: List[str], seed: int = 11) -> Stormer:
    torch.manual_seed(seed)
    model = Stormer(
        in_img_size=GRID, variables=variables, patch_size=2,
        hidden_size=HIDDEN, depth=DEPTH, num_heads=4, mlp_ratio=2.0,
    )
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    model.requires_grad_(False)
    model.eval()
    return model


def _make_bridge(model: Stormer, variables: List[str]) -> WeatherStepBridge:
    normalization = NormalizationContract(
        inp_mean=torch.zeros(N_VARS),
        inp_std=torch.ones(N_VARS),
        diff_mean={6: torch.zeros(N_VARS)},
        diff_std={6: torch.ones(N_VARS)},
        variables=variables,
    )
    version = ArtifactVersion(
        backbone="stormer_test_f0", static_adapter="none", edit_bank="none",
        normalization="pending", grid="16x32", projection="patch2",
        split="test", continuation="reference_after_hold",
    )
    return WeatherStepBridge(model, normalization, version)


@pytest.fixture
def backbone(variables) -> Stormer:
    return _make_backbone(variables)


@pytest.fixture
def bridge(backbone, variables) -> WeatherStepBridge:
    return _make_bridge(backbone, variables)


@pytest.fixture
def objective(bridge) -> sa.ObjectiveSpec:
    lat = torch.linspace(-88.0, 88.0, GRID[0])
    return sa.build_objective_spec(bridge, lat, lead_steps=(1,), space="raw")


def _samples(
    bridge: WeatherStepBridge, n: int = 2, lead_steps=(1,), seed: int = 5
) -> List[sa.TrainingSample]:
    generator = torch.Generator().manual_seed(seed)
    out: List[sa.TrainingSample] = []
    for i in range(n):
        x_raw = torch.randn(1, N_VARS, *GRID, generator=generator)
        targets = {
            int(s): x_raw + 0.4 * torch.randn(1, N_VARS, *GRID, generator=generator)
            for s in lead_steps
        }
        out.append(sa.TrainingSample(
            x_norm=bridge.normalization.normalize(x_raw),
            targets_raw=targets,
            issue_id=f"iss_{i:03d}",
            issue_time=1_420_000_000 + i * 21_600,
            valid_time=1_420_000_000 + i * 21_600 + 21_600,
            history_time=1_420_000_000 + i * 21_600 - 21_600,
            split_id="train",
            data_role="bank_fit",
            source="synthetic",
            time_utc=f"2015-01-0{i + 1}T00:00:00Z",
        ))
    return out


@pytest.fixture
def samples(bridge) -> List[sa.TrainingSample]:
    return _samples(bridge)


def _trained_fs(
    bridge: WeatherStepBridge,
    samples: List[sa.TrainingSample],
    objective: sa.ObjectiveSpec,
    *,
    updates: int = 14,
) -> Tuple[Dict[int, ExpertLoRA], sa.FsFitRecord]:
    config = sa.FsFitConfig(
        mode="gradient_check",
        max_updates=updates,
        learning_rate=2e-2,
        train_steps=1,
        lead_steps=(1,),
        target_blocks=TARGET_BLOCKS,
        rank_per_expert=FS_RANK,
    )
    return sa.fit_static_adapter(bridge, samples, objective, config)


# =============================================================================
# 1. The fit reduces the objective
# =============================================================================

def test_fs_fit_reduces_loss_on_bank_fit_samples(bridge, samples, objective):
    """Fs training moves the native-registered objective down, and is eligible.

    The assertion is on the PER-SAMPLE direction, not on the raw first/last
    update losses. The schedule cycles through both samples, so update 0 and
    update 13 score different samples and their difference would mix the
    training trend with the gap between two samples.
    """
    _, record = _trained_fs(bridge, samples, objective)

    assert record.n_updates == 14
    assert record.eligible is True
    assert record.ineligible_reason is None
    assert all(torch.isfinite(torch.tensor(v)) for v in record.losses)

    direction = record.loss_direction
    assert direction["n_samples_with_repeats"] == 2
    assert direction["fraction_decreased"] == 1.0, (
        "Fs fit did not reduce the objective on every training sample: "
        f"{direction['per_sample']}"
    )
    for issue_id, entry in direction["per_sample"].items():
        assert entry["last"] < entry["first"], (issue_id, entry)

    # Provenance: what data trained this, at what role.
    assert record.data_role == "bank_fit"
    assert set(record.training_issue_ids) == {"iss_000", "iss_001"}


def test_loss_direction_is_per_sample_not_first_vs_last_update():
    """A cycling schedule cannot fake a trend through the direction check."""
    # Sample A improves 1.0 -> 0.5; sample B is flat at 9.0. The raw first/last
    # update losses (1.0 vs 9.0) would say "got worse"; the per-sample view says
    # one of two samples improved.
    direction = sa.per_sample_loss_direction(
        [1.0, 9.0, 0.5, 9.0], ["A", "B", "A", "B"]
    )
    assert direction["n_samples_with_repeats"] == 2
    assert direction["n_decreased"] == 1
    assert direction["fraction_decreased"] == 0.5
    assert direction["per_sample"]["A"]["decreased"] is True
    assert direction["per_sample"]["B"]["decreased"] is False
    # A sample seen once carries no direction and is excluded, not counted as a
    # non-improvement.
    single = sa.per_sample_loss_direction([1.0], ["A"])
    assert single["n_samples_with_repeats"] == 0
    assert single["fraction_decreased"] is None


def test_fs_fit_refuses_samples_that_are_not_bank_fit(bridge, samples, objective):
    """Fitting Fs on policy_dev or confirm data is refused, not merely warned."""
    leaked = sa.TrainingSample(
        x_norm=samples[0].x_norm,
        targets_raw=samples[0].targets_raw,
        issue_id="iss_leak",
        data_role="policy_dev",
    )
    config = sa.FsFitConfig(
        mode="gradient_check", max_updates=1, train_steps=1, lead_steps=(1,),
        target_blocks=TARGET_BLOCKS, rank_per_expert=FS_RANK,
    )
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.fit_static_adapter(bridge, samples + [leaked], objective, config)
    assert excinfo.value.code == "FS_DATA_ROLE_NOT_BANK_FIT"


@pytest.mark.parametrize(
    "mode, requested, cap",
    [("gradient_check", 33, 32), ("formal", 501, 500)],
)
def test_update_caps_are_enforced_by_the_config(mode, requested, cap):
    """<=32 for the gradient check, <=500 for the formal fit -- enforced."""
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.FsFitConfig(mode=mode, max_updates=requested, train_steps=1, lead_steps=(1,))
    assert excinfo.value.code == "FS_UPDATE_CAP_EXCEEDED"
    assert excinfo.value.detail["cap"] == cap
    # The cap itself is satisfiable.
    assert sa.FsFitConfig(
        mode=mode, max_updates=cap, train_steps=1, lead_steps=(1,)
    ).max_updates == cap


# =============================================================================
# 2. The merge is numerically exact
# =============================================================================

def test_merged_projection_equals_lora_branch_exactly_in_float64(variables):
    """W' = W + c*scale*B@A reproduces the hook's contribution, at float64.

    The rollout-level check below runs in float32 and so can only assert a
    tolerance. This one checks the actual algebra the merge relies on, where an
    exact statement is available.
    """
    torch.manual_seed(3)
    lora = ExpertLoRA(HIDDEN, HIDDEN, num_experts=1, rank_per_expert=FS_RANK, scale=1.0)
    with torch.no_grad():
        lora.down[0].weight.normal_(std=0.1)
        lora.up[0].weight.normal_(std=0.1)
    lora.double()

    proj = torch.nn.Linear(HIDDEN, HIDDEN).double()
    x = torch.randn(2, 7, HIDDEN, dtype=torch.float64)
    coefficients = torch.full((2, 1), sa.FS_COEFFICIENT, dtype=torch.float64)

    branch = proj(x) + lora.forward_dense(x, coefficients)

    delta = (lora.up[0].weight @ lora.down[0].weight) * lora.scale * sa.FS_COEFFICIENT
    merged = torch.nn.Linear(HIDDEN, HIDDEN).double()
    with torch.no_grad():
        merged.weight.copy_(proj.weight + delta)
        merged.bias.copy_(proj.bias)

    torch.testing.assert_close(merged(x), branch, rtol=1e-12, atol=1e-12)


def test_merge_equivalence_through_a_full_rollout(bridge, backbone, samples,
                                                  objective, variables):
    """The merged backbone and the always-on branch roll out identically."""
    adapters, _ = _trained_fs(bridge, samples, objective)
    merged, artifact = sa.merge_static_adapter(
        backbone, adapters, target_blocks=TARGET_BLOCKS
    )
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)

    report = sa.verify_merge_equivalence(
        bridge, fs_bridge, adapters, samples[0].x_norm, variables,
        steps=2, target_blocks=TARGET_BLOCKS,
    )
    assert report["passed"] is True, report
    # The comparison is not vacuous: Fs really moved the forecast.
    assert report["branch_max_abs"] > 0
    assert artifact.delta_max_abs > 0
    assert report["max_abs_diff"] < 1e-4


# -----------------------------------------------------------------------------
# 2b. The merge check carries its own noise floor, and runs with TF32 off
# -----------------------------------------------------------------------------
#
# On real H100 hardware `verify_merge_equivalence` reported max_abs_diff ~1.1e-3
# to ~1.4e-3 across four independently trained experts against atol=1e-5, while
# the same comparison on this synthetic CPU fixture sits near 1e-7. Two
# explanations were indistinguishable from the number alone: TF32 matmul (the
# CUDA default on Ampere/Hopper, ~1e-3 relative precision) or ordinary GPU
# kernel non-determinism between the branch and merged code paths.
#
# The code now answers that empirically rather than by assertion: it disables
# TF32 for the comparison, and it reports `self_max_abs_diff` -- the merged-path
# forward called twice, differenced against itself, which is what "no code-path
# difference at all" costs on this hardware.
#
# GPU precision behaviour cannot be exercised on CPU-only test hardware. What
# CAN be pinned down here is the plumbing: the flags really are flipped around
# all three forward passes, the prior values (not True) really are restored,
# and the diagnostic key really is reported.


def test_tf32_guard_disables_both_flags_and_restores_the_prior_values():
    """Restore is to whatever was there before, never a hardcoded True."""
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    try:
        # Deliberately non-default on BOTH flags, and deliberately not the same
        # value on both, so a restore that hardcodes True -- or that restores
        # one flag's value onto the other -- fails here.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False

        with sa._disable_tf32_for_identity_check():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False

        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is False

        # And the mirrored starting state, so neither assertion above can be
        # satisfied by a constant.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True

        with sa._disable_tf32_for_identity_check():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False

        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


def test_tf32_guard_restores_the_prior_values_when_the_body_raises():
    """`finally` must restore even when the verification blows up mid-rollout.

    Without this the first failed merge check would leave TF32 off for the rest
    of the process, silently changing the training and profiling numerics that
    run after it.
    """
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        sentinel = RuntimeError("rollout exploded")
        with pytest.raises(RuntimeError) as excinfo:
            with sa._disable_tf32_for_identity_check():
                assert torch.backends.cuda.matmul.allow_tf32 is False
                assert torch.backends.cudnn.allow_tf32 is False
                raise sentinel
        assert excinfo.value is sentinel

        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


def test_merge_equivalence_reports_a_self_consistency_noise_floor(
    bridge, backbone, samples, objective, variables
):
    """`self_max_abs_diff` is the same call twice -- the floor `max_abs_diff` sits on.

    On this deterministic single-threaded CPU fixture there is no randomness
    between the two calls and no kernel-selection variability, so the floor is
    essentially zero. Its value on GPU is the whole point of shipping it: it
    says whether a failing `max_abs_diff` is "the same as calling it twice"
    (hardware noise) or genuinely worse (a real branch-vs-merge divergence).
    """
    adapters, _ = _trained_fs(bridge, samples, objective)
    merged, artifact = sa.merge_static_adapter(
        backbone, adapters, target_blocks=TARGET_BLOCKS
    )
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)

    report = sa.verify_merge_equivalence(
        bridge, fs_bridge, adapters, samples[0].x_norm, variables,
        steps=2, target_blocks=TARGET_BLOCKS,
    )

    assert "self_max_abs_diff" in report, report
    floor = report["self_max_abs_diff"]
    assert isinstance(floor, float)
    assert math.isfinite(floor)
    assert floor >= 0.0
    assert floor <= 1e-6, (
        f"calling the merged bridge's forward twice with identical arguments "
        f"on deterministic CPU differed by {floor:.3e}; the diagnostic is "
        f"supposed to read as a near-zero floor here, so either the second "
        f"call is not identical to the first or something stateful leaked "
        f"between them"
    )
    # The diagnostic is additive: nothing the existing report promised moved.
    assert report["passed"] is True, report
    assert report["max_abs_diff"] < 1e-4
    assert report["atol"] == 1e-5
    assert report["rtol"] == 1e-5
    assert report["tf32_disabled_for_check"] is True


def test_merge_equivalence_runs_every_forward_pass_with_tf32_disabled():
    """All three passes, or the comparison compares rounding modes, not paths.

    Checked by spying on the two entry points the check rolls forward through
    (`controlled_rollout` for the branch path, `forward_validation` for the
    merged path and its repeat) and recording the flags each one actually saw.
    """
    prior_matmul = torch.backends.cuda.matmul.allow_tf32
    prior_cudnn = torch.backends.cudnn.allow_tf32
    seen: List[Tuple[str, bool, bool]] = []

    def _record(label):
        seen.append(
            (
                label,
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            )
        )

    real_rollout = sa.controlled_rollout
    real_forward = WeatherStepBridge.forward_validation

    def spy_rollout(*call_args, **call_kwargs):
        _record("controlled_rollout")
        return real_rollout(*call_args, **call_kwargs)

    def spy_forward(self, *call_args, **call_kwargs):
        _record("forward_validation")
        return real_forward(self, *call_args, **call_kwargs)

    try:
        # Start from TF32 on, so "disabled inside" is a real observation.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        variables = list(DEFAULT_VARIABLES[:N_VARS])
        backbone = _make_backbone(variables)
        bridge = _make_bridge(backbone, variables)
        adapters = sa.build_fs_adapter(
            HIDDEN, TARGET_BLOCKS, rank_per_expert=FS_RANK, seed=5
        )
        with torch.no_grad():
            for lora in adapters.values():
                for up in lora.up:
                    up.weight.normal_(std=0.05)
        merged, artifact = sa.merge_static_adapter(
            backbone, adapters, target_blocks=TARGET_BLOCKS
        )
        fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
        x_norm = torch.randn(1, N_VARS, *GRID)

        with mock.patch.object(sa, "controlled_rollout", new=spy_rollout), \
                mock.patch.object(WeatherStepBridge, "forward_validation",
                                  new=spy_forward):
            sa.verify_merge_equivalence(
                bridge, fs_bridge, adapters, x_norm, variables,
                steps=1, target_blocks=TARGET_BLOCKS,
            )

        labels = [label for label, _, _ in seen]
        assert labels.count("controlled_rollout") == 1, seen
        assert labels.count("forward_validation") == 2, seen
        for label, matmul_flag, cudnn_flag in seen:
            assert matmul_flag is False, (label, seen)
            assert cudnn_flag is False, (label, seen)

        # Restored, and restored to what was there before.
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


def test_merge_does_not_mutate_the_source_backbone(bridge, backbone, samples, objective):
    """The merge produces an independent copy; F0 stays available as F0."""
    adapters, _ = _trained_fs(bridge, samples, objective)
    before = {
        name: param.detach().clone() for name, param in backbone.named_parameters()
    }
    merged, _ = sa.merge_static_adapter(backbone, adapters, target_blocks=TARGET_BLOCKS)

    assert merged is not backbone
    for name, param in backbone.named_parameters():
        torch.testing.assert_close(param.detach(), before[name], rtol=0.0, atol=0.0)
    changed = [
        name for name, param in merged.named_parameters()
        if not torch.equal(param.detach(), before[name])
    ]
    assert changed, "the merge changed nothing in the copy"
    assert all("attn.proj.weight" in name for name in changed), changed


def test_merged_backbone_is_frozen(bridge, backbone, samples, objective):
    adapters, _ = _trained_fs(bridge, samples, objective)
    merged, _ = sa.merge_static_adapter(backbone, adapters, target_blocks=TARGET_BLOCKS)
    assert [n for n, p in merged.named_parameters() if p.requires_grad] == []
    assert merged.training is False


# =============================================================================
# 3. Post-freeze zero-edit equivalence -- against Fs, and discriminating
# =============================================================================

def _freeze(bridge, backbone, samples, objective, *, updates: int = 14):
    adapters, record = _trained_fs(bridge, samples, objective, updates=updates)
    merged, artifact = sa.merge_static_adapter(
        backbone, adapters, target_blocks=TARGET_BLOCKS
    )
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)
    return adapters, record, merged, artifact, fs_bridge


def _dynamic_bank(seed: int = 21) -> Dict[int, ExpertLoRA]:
    generator = torch.Generator().manual_seed(seed)
    bank: Dict[int, ExpertLoRA] = {}
    for block in TARGET_BLOCKS:
        lora = ExpertLoRA(
            HIDDEN, HIDDEN, num_experts=BANK_EXPERTS, rank_per_expert=BANK_RANK
        )
        with torch.no_grad():
            for down in lora.down:
                down.weight.copy_(torch.randn(down.weight.shape, generator=generator) * 0.1)
            for up in lora.up:
                up.weight.copy_(torch.randn(up.weight.shape, generator=generator) * 0.1)
        bank[block] = lora
    return bank


def test_zero_edit_equivalence_holds_against_fs_and_would_fail_against_f0(
    bridge, backbone, samples, objective, variables
):
    """The certified baseline is Fs, and F0's numbers would NOT satisfy it.

    This is the discriminating case the spec asks for: Fs has been trained, so
    the Fs baseline and the F0 baseline are different forecasts. A check that
    mistakenly compared the post-freeze no-edit rollout against F0 would see a
    non-zero difference and fail, which is exactly why F0's numbers may not be
    carried over as the new reference.
    """
    _, _, _, _, fs_bridge = _freeze(bridge, backbone, samples, objective)
    bank = _dynamic_bank()

    report = sa.verify_zero_edit_equivalence(
        fs_bridge, bank, BANK_EXPERTS, samples[0].x_norm, variables,
        steps=2, target_blocks=TARGET_BLOCKS, f0_bridge=bridge,
    )

    assert report["baseline_tag"] == sa.FS_BASELINE_TAG
    assert report["passed"] is True
    assert report["max_abs_diff_vs_fs"] == 0.0

    # The same comparison against F0 is non-zero -- it would FAIL the check.
    assert report["max_abs_diff_vs_f0"] > 0.0
    assert report["discriminating"] is True
    would_pass_against_f0 = report["max_abs_diff_vs_f0"] <= report["atol"]
    assert not would_pass_against_f0, (
        "zero-edit equivalence is not discriminating: the F0 baseline would "
        "have satisfied it too, so this test could not catch a run that reused "
        "F0's numerical reference after the Fs freeze"
    )


def test_zero_edit_equivalence_reports_when_it_cannot_tell_fs_from_f0(
    bridge, backbone, samples, objective, variables
):
    """An untrained (zero-init) Fs IS F0 -- the report says so, honestly.

    This is the companion to the test above: it shows the `discriminating` flag
    is a real observation about this Fs and not a constant True.
    """
    adapters = sa.build_fs_adapter(
        HIDDEN, TARGET_BLOCKS, rank_per_expert=FS_RANK, seed=1
    )  # up factors are zero => Fs is a numerical no-op
    merged, artifact = sa.merge_static_adapter(
        backbone, adapters, target_blocks=TARGET_BLOCKS
    )
    fs_bridge = sa.make_fs_bridge(merged, bridge, artifact)

    report = sa.verify_zero_edit_equivalence(
        fs_bridge, _dynamic_bank(), BANK_EXPERTS, samples[0].x_norm, variables,
        steps=2, target_blocks=TARGET_BLOCKS, f0_bridge=bridge,
    )
    assert report["passed"] is True
    assert report["max_abs_diff_vs_fs"] == 0.0
    assert report["max_abs_diff_vs_f0"] == 0.0
    assert report["discriminating"] is False
    assert "cannot currently distinguish" in report["note"]
    assert artifact.delta_max_abs == 0.0


def test_zero_edit_equivalence_fails_when_a_dynamic_expert_is_actually_on(
    bridge, backbone, samples, objective, variables
):
    """The check is a real check: a non-zero coefficient breaks it."""
    _, _, _, _, fs_bridge = _freeze(bridge, backbone, samples, objective)
    bank = _dynamic_bank()

    edited_plan = EditPlan(
        plan_id="expert_0", num_experts=BANK_EXPERTS,
        coefficients=(0.25, 0.0, 0.0, 0.0), hold_steps=2, rho=0.25,
    )
    with torch.no_grad():
        edited = controlled_rollout(
            fs_bridge, samples[0].x_norm, variables, interval=6, steps=2,
            plan=edited_plan, expert_loras=bank, target_blocks=TARGET_BLOCKS,
        )
        baseline = fs_bridge.forward_validation(
            samples[0].x_norm, variables, interval=6, steps=2
        )
    assert float((edited - baseline).abs().max()) > 0.0


# =============================================================================
# 4. The reference identity is re-recorded, not inherited
# =============================================================================

def test_static_adapter_field_is_bound_to_fs_instead_of_the_placeholder(
    bridge, backbone, samples, objective
):
    _, _, _, artifact, fs_bridge = _freeze(bridge, backbone, samples, objective)

    assert bridge.version.static_adapter == "none"
    assert fs_bridge.version.static_adapter != "none"
    assert fs_bridge.version.static_adapter == artifact.static_adapter_label
    assert artifact.static_adapter_digest[:16] in fs_bridge.version.static_adapter
    # The backbone identity changed too, so an F0-era artifact cannot match.
    assert artifact.merged_backbone_digest != artifact.base_backbone_digest
    assert artifact.merged_backbone_digest[:16] in fs_bridge.version.backbone
    with pytest.raises(ValueError, match="Stale/incompatible artifact"):
        bridge.version.assert_matches(fs_bridge.version)


def test_fs_identity_refuses_to_stack_a_second_static_adapter(
    bridge, backbone, samples, objective
):
    _, _, _, artifact, fs_bridge = _freeze(bridge, backbone, samples, objective)
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.fs_artifact_version(fs_bridge._version, artifact)
    assert excinfo.value.code == "FS_IDENTITY_ALREADY_HAS_STATIC_ADAPTER"


def test_post_freeze_reference_record_is_complete(
    bridge, backbone, samples, objective, variables
):
    adapters, _, _, artifact, fs_bridge = _freeze(bridge, backbone, samples, objective)
    verification = sa.record_post_freeze_reference(
        bridge, fs_bridge, adapters, artifact, samples[0].x_norm, variables,
        num_experts=BANK_EXPERTS, dynamic_bank=_dynamic_bank(), steps=2,
        target_blocks=TARGET_BLOCKS, samples=samples, spec=objective,
    )
    payload = verification.to_dict()

    assert verification.passed is True
    assert payload["reference_identity"]["static_adapter_is_bound"] is True
    assert payload["reference_identity"]["recorded_fresh_after_freeze"] is True
    assert payload["merge_equivalence"]["passed"] is True
    assert payload["zero_edit_equivalence"]["passed"] is True
    assert payload["zero_edit_equivalence"]["baseline_tag"] == "Fs"
    assert payload["background_f0"]["tag"] == "background_F0"
    assert payload["fs_baseline"]["tag"] == "Fs"


def test_record_post_freeze_reference_merge_atol_is_delta_relative():
    """merge_atol is no longer a fixed constant (see the comment above
    `merge_atol_relative`/`merge_atol_floor` in static_adapter.py): real-GPU
    jobs pt-2r6tbwu7 (32 updates) and pt-3x63g0c6 (500 updates) showed the
    branch-vs-merge FP32 residual scales with the merged LoRA delta's
    magnitude, not a fixed number. This guards the registered coefficients
    against silently drifting away from the values that evidence justified.
    """
    signature = inspect.signature(sa.record_post_freeze_reference)
    assert signature.parameters["merge_atol_relative"].default == 1.5e-3
    assert signature.parameters["merge_atol_floor"].default == 1e-5
    # merge_rtol is unrelated to this scaling and must stay at its old value.
    assert signature.parameters["merge_rtol"].default == 1e-5
    # The old fixed-constant parameter must be gone, not just unused.
    assert "merge_atol" not in signature.parameters


def test_effective_merge_atol_uses_the_floor_when_delta_is_near_zero():
    assert sa._effective_merge_atol(0.0, relative=1.5e-3, floor=1e-5) == 1e-5
    assert sa._effective_merge_atol(1e-6, relative=1.5e-3, floor=1e-5) == 1e-5


def test_effective_merge_atol_scales_with_delta_when_delta_is_large():
    # Matches the real pt-3x63g0c6 expert1 data point: delta_max_abs=2.3050.
    result = sa._effective_merge_atol(2.3050, relative=1.5e-3, floor=1e-5)
    assert result == pytest.approx(1.5e-3 * 2.3050)
    assert result > 5.514e-4  # must stay above that job's observed max_abs_diff


# =============================================================================
# 5. F0's background score is reporting-only and cannot become a baseline
# =============================================================================

def test_background_f0_is_tagged_separately_and_is_reporting_only(
    bridge, backbone, samples, objective, variables
):
    _, _, _, _, fs_bridge = _freeze(bridge, backbone, samples, objective)

    background = sa.background_f0_score(bridge, samples, objective, variables=variables)
    fs_baseline = sa.fs_baseline_score(fs_bridge, samples, objective, variables=variables)

    assert background.tag == sa.BACKGROUND_F0_TAG == "background_F0"
    assert background.reporting_only is True
    assert "reporting only" in background.note
    assert fs_baseline["tag"] == sa.FS_BASELINE_TAG == "Fs"
    # They are genuinely different numbers, so mixing them up would corrupt a gain.
    assert background.loss != fs_baseline["loss"]


def test_gain_against_the_f0_background_is_refused(
    bridge, backbone, samples, objective, variables
):
    """`fs_relative_gain` refuses an F0-tagged baseline outright."""
    _, _, _, _, fs_bridge = _freeze(bridge, backbone, samples, objective)
    background = sa.background_f0_score(bridge, samples, objective, variables=variables)
    fs_baseline = sa.fs_baseline_score(fs_bridge, samples, objective, variables=variables)

    # The legitimate direction works.
    gain = sa.fs_relative_gain(fs_baseline, candidate_loss=fs_baseline["loss"] - 0.01)
    assert gain["baseline_tag"] == "Fs"
    assert gain["gain_vs_fs"] == pytest.approx(0.01)

    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.fs_relative_gain(background.to_dict(), candidate_loss=0.0)
    assert excinfo.value.code == "GAIN_BASELINE_NOT_FS"

    with pytest.raises(sa.StaticAdapterViolation):
        sa.assert_gain_baseline_is_fs("background_F0")


# =============================================================================
# Objective contract
# =============================================================================

def test_objective_uses_area_weight_and_variable_scale(bridge):
    lat = torch.linspace(-88.0, 88.0, GRID[0])
    spec = sa.build_objective_spec(bridge, lat, lead_steps=(1, 2), space="raw")
    payload = spec.to_dict()
    assert payload["q_kind"] == "cos_latitude_area_weight"
    assert payload["scale_kind"] == "official_per_variable_input_std"
    assert payload["lead_hours"] == [6, 12]
    assert spec.q.shape == (1, 1, GRID[0], 1)
    assert spec.scale.shape == (1, N_VARS, 1, 1)


def test_objective_refuses_a_lead_the_rollout_never_reached(bridge, samples, objective):
    trajectory = torch.randn(1, 2, N_VARS, *GRID)  # only step 1 available
    spec = sa.ObjectiveSpec(
        lead_steps=(4,), q=objective.q, scale=objective.scale, space="raw",
        variables=objective.variables,
    )
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.objective_fields(bridge, trajectory, samples[0], spec)
    assert excinfo.value.code == "OBJECTIVE_LEAD_NOT_ROLLED"


def test_fit_config_refuses_a_lead_beyond_the_training_horizon():
    with pytest.raises(sa.StaticAdapterViolation) as excinfo:
        sa.FsFitConfig(mode="gradient_check", max_updates=4, train_steps=1,
                       lead_steps=(4,))
    assert excinfo.value.code == "FS_LEAD_BEYOND_TRAIN_HORIZON"


# =============================================================================
# stage_fs_freeze re-homes the dynamic bank onto the Fs backbone's device
# =============================================================================

def _fs_freeze_args(**overrides):
    args = SimpleNamespace(
        num_experts=BANK_EXPERTS,
        rank_per_expert=BANK_RANK,
        seed=21,
        hold_steps=2,
        output_dir=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _fs_freeze_ctx(bridge, samples, objective, variables):
    adapters, _ = _trained_fs(bridge, samples, objective)
    return {
        "bridge": bridge,
        "fs_adapters": adapters,
        "target_blocks": list(TARGET_BLOCKS),
        "samples": samples,
        "variables": variables,
        "objective": objective,
        "record": {},
    }


def test_stage_fs_freeze_moves_every_bank_lora_onto_the_backbone_device(
    bridge, samples, objective, variables
):
    """Regression guard for the device mismatch that killed `--stage fs_freeze`.

    `bt.build_dynamic_bank` has no `device` argument and always constructs its
    `ExpertLoRA`s on CPU. `stage_fs_freeze` then feeds that bank straight into
    the zero-edit verification, which runs a real `controlled_rollout` against
    the CUDA-resident Fs-merged backbone. Without an explicit re-home every
    expert on a GPU run stays on CPU and the rollout dies with "Expected all
    tensors to be on the same device, but found at least two devices, cuda:N
    and cpu!" -- exactly how all four experts failed on the real job.

    CI here is single-device, so asserting "the weights ended up on the right
    device" is vacuously true with or without the fix. What this pins down
    instead is that the move is actually *performed*: `.to(device)` has to be
    called on each bank entry. Delete the loop in `stage_fs_freeze` and this
    test fails on plain CPU hardware, which is the whole point.
    """
    ctx = _fs_freeze_ctx(bridge, samples, objective, variables)
    args = _fs_freeze_args()

    real_to = ExpertLoRA.to
    moved_to: Dict[int, List[object]] = {}

    def spy_to(self, *call_args, **call_kwargs):
        target = call_args[0] if call_args else call_kwargs.get("device")
        if target is not None:
            moved_to.setdefault(id(self), []).append(target)
        return real_to(self, *call_args, **call_kwargs)

    with mock.patch.object(ExpertLoRA, "to", new=spy_to):
        passed = r2_cli.stage_fs_freeze(args, ctx)

    assert passed is True
    bank = ctx["dynamic_bank"]
    assert set(bank) == set(TARGET_BLOCKS)

    fs_device = next(ctx["fs_bridge"].model.parameters()).device
    for block, lora in bank.items():
        assert id(lora) in moved_to, (
            f"stage_fs_freeze never called .to(device) on the dynamic-bank "
            f"expert for block {block}. On a GPU run its factors would stay on "
            f"CPU while the Fs-merged backbone sits on CUDA, and the zero-edit "
            f"rollout would raise a device mismatch."
        )
        assert any(
            torch.device(target) == fs_device for target in moved_to[id(lora)]
        ), (
            f"the block-{block} expert was moved, but never onto the Fs-merged "
            f"backbone's device {fs_device} (targets seen: {moved_to[id(lora)]})."
        )

    # Bonus sanity check: vacuously true on single-device CI, but it is the
    # invariant the re-home loop exists to establish.
    for lora in bank.values():
        assert next(lora.parameters()).device == fs_device
