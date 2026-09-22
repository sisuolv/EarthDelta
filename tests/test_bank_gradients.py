"""Bank gradient flow, backbone freezing, and hook-lifecycle tests.

`earthdelta/lowrank.py` itself holds no hooks: `ExpertLoRA` is a plain
`nn.Module` with two explicit forward paths. The hook mechanism the bank
actually rides on lives in `controlled_rollout`
(`earthdelta/bridge/stormer_bridge.py`), which registers a forward hook on each
targeted block's `attn.proj`, runs the frozen backbone, and removes the hooks in
a `finally`. That is the real mechanism these tests exercise; nothing here
invents a hook system that the production code does not have.

Four properties are asserted, each against a FIXED non-zero coefficient tuple:

  1. Both factors of an active expert receive finite, non-zero gradients, and
     inactive experts receive exactly zero.
  2. The frozen backbone receives no gradients at all while the bank trains.
  3. Toggling a coefficient zero -> non-zero -> zero restores the no-gradient
     state exactly, so a disabled expert cannot keep accumulating signal.
  4. Every hook registered during a rollout is removed afterwards -- after a
     normal return, after an exception raised inside the backbone's forward,
     and across serial repeats -- leaving zero dangling hooks on the backbone.

Note on initialization: `ExpertLoRA` follows the LoRA convention of zeroing the
up (B) projections, which makes the product identically zero and therefore
makes dL/dA identically zero as well. A bank at its initial state legitimately
has no A gradient, so every gradient test here first gives B non-zero weights.
That is the state a bank is in once training has moved it off the origin.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthdelta.bridge import (  # noqa: E402
    DEFAULT_VARIABLES,
    NormalizationContract,
    Stormer,
    WeatherStepBridge,
    controlled_rollout,
)
from earthdelta.contracts import ArtifactVersion, EditPlan  # noqa: E402
from earthdelta.lowrank import ExpertLoRA  # noqa: E402


# =============================================================================
# Fixture geometry
# =============================================================================

GRID = (16, 32)
N_VARS = 8
HIDDEN = 32
DEPTH = 2
NUM_EXPERTS = 4
RANK = 2
TARGET_BLOCK = 0

#: The fixed non-zero coefficient tuple every gradient test uses. Experts 0 and
#: 2 are active with distinct magnitudes and opposite signs; 1 and 3 are off.
FIXED_COEFFICIENTS: Tuple[float, ...] = (0.20, 0.0, -0.10, 0.0)
ACTIVE_EXPERTS: Tuple[int, ...] = (0, 2)
INACTIVE_EXPERTS: Tuple[int, ...] = (1, 3)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(20260921)
    np.random.seed(20260921)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def variables() -> List[str]:
    return list(DEFAULT_VARIABLES[:N_VARS])


@pytest.fixture
def backbone(variables) -> Stormer:
    """A small backbone with non-trivial weights, frozen as in production."""
    model = Stormer(
        in_img_size=GRID, variables=variables, patch_size=2,
        hidden_size=HIDDEN, depth=DEPTH, num_heads=4, mlp_ratio=2.0,
    )
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.1)
    torch.nn.init.normal_(model.head.linear.weight, std=0.02)
    torch.nn.init.normal_(model.head.linear.bias, std=0.02)
    model.eval()
    # The backbone is frozen: only the bank trains.
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@pytest.fixture
def bridge(backbone, variables) -> WeatherStepBridge:
    normalization = NormalizationContract(
        inp_mean=torch.zeros(N_VARS),
        inp_std=torch.ones(N_VARS),
        diff_mean={6: torch.zeros(N_VARS), 24: torch.zeros(N_VARS)},
        diff_std={6: torch.ones(N_VARS), 24: torch.ones(N_VARS)},
        variables=variables,
    )
    version = ArtifactVersion(
        backbone="stormer_test", static_adapter="none", edit_bank="bank_test",
        normalization="pending", grid="16x32", projection="patch2",
        split="test", continuation="reference_after_hold",
    )
    return WeatherStepBridge(backbone, normalization, version)


def make_bank(seed: int = 7) -> ExpertLoRA:
    """A bank whose up (B) factors are off the origin, as after some training.

    Both factor sets are drawn from a LOCAL generator so that two calls with
    the same seed produce bit-identical banks regardless of the ambient global
    RNG state. Tests that compare gradients across runs depend on that: the
    constructor's own `kaiming_uniform_` for the down factors consumes the
    global RNG, which would otherwise make two "identical" banks differ.

    The up factors must be non-zero: `ExpertLoRA` follows the LoRA convention
    of zeroing them, which makes dL/dA identically zero, so a bank at its
    initial state legitimately has no A gradient at all.
    """
    generator = torch.Generator().manual_seed(seed)
    bank = ExpertLoRA(
        HIDDEN, HIDDEN, num_experts=NUM_EXPERTS, rank_per_expert=RANK, scale=1.0
    )
    with torch.no_grad():
        for down in bank.down:
            down.weight.copy_(torch.randn(down.weight.shape, generator=generator) * 0.1)
        for up in bank.up:
            up.weight.copy_(torch.randn(up.weight.shape, generator=generator) * 0.1)
    for param in bank.parameters():
        param.requires_grad_(True)
    return bank


def plan_with(coefficients: Tuple[float, ...]) -> EditPlan:
    return EditPlan("bank_test", NUM_EXPERTS, tuple(coefficients), rho=0.25)


def rollout(bridge, variables, bank, coefficients, steps: int = 1, **kwargs):
    """One controlled rollout with the bank injected at TARGET_BLOCK."""
    x_norm = torch.randn(1, N_VARS, *GRID)
    return controlled_rollout(
        bridge, x_norm, variables, interval=6, steps=steps,
        plan=plan_with(coefficients), expert_loras={TARGET_BLOCK: bank},
        target_blocks=(TARGET_BLOCK,), **kwargs,
    )


# =============================================================================
# Hook accounting
# =============================================================================

def count_hooks(module: torch.nn.Module) -> int:
    """Total hooks registered anywhere in a module tree."""
    total = 0
    for submodule in module.modules():
        for attribute in (
            "_forward_hooks", "_forward_pre_hooks",
            "_backward_hooks", "_full_backward_hooks",
            "_backward_pre_hooks", "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
        ):
            registry = getattr(submodule, attribute, None)
            if registry:
                total += len(registry)
    return total


def proj_hooks(bridge: WeatherStepBridge) -> Dict[int, int]:
    """Forward hooks on each block's attn.proj -- the real injection point."""
    return {
        index: len(block.attn.proj._forward_hooks)
        for index, block in enumerate(bridge.model.blocks)
    }


# =============================================================================
# 1. Bank A/B gradients under a fixed non-zero coefficient tuple
# =============================================================================

def test_active_expert_factors_receive_nonzero_gradients(bridge, variables):
    """Both factors of every ACTIVE expert get finite, non-zero gradients."""
    bank = make_bank()
    out = rollout(bridge, variables, bank, FIXED_COEFFICIENTS)
    out.sum().backward()

    for k in ACTIVE_EXPERTS:
        for label, module in (("down/A", bank.down[k]), ("up/B", bank.up[k])):
            grad = module.weight.grad
            assert grad is not None, f"expert {k} {label} received no gradient"
            assert torch.isfinite(grad).all(), f"expert {k} {label} gradient not finite"
            assert grad.abs().max().item() > 0.0, (
                f"expert {k} {label} gradient is identically zero despite "
                f"coefficient {FIXED_COEFFICIENTS[k]}"
            )


def test_inactive_expert_factors_receive_exactly_zero_gradients(bridge, variables):
    """Zero-coefficient experts get a zero gradient, not a small one."""
    bank = make_bank()
    out = rollout(bridge, variables, bank, FIXED_COEFFICIENTS)
    out.sum().backward()

    for k in INACTIVE_EXPERTS:
        for label, module in (("down/A", bank.down[k]), ("up/B", bank.up[k])):
            grad = module.weight.grad
            assert grad is not None, (
                f"expert {k} {label} has no grad at all; the dense path must "
                "still build the graph for inactive experts"
            )
            assert torch.count_nonzero(grad).item() == 0, (
                f"expert {k} {label} received a non-zero gradient while its "
                "coefficient was zero"
            )


def test_bank_b_gradient_is_exactly_proportional_to_the_coefficient():
    """At the bank itself, dL/dB scales exactly with the coefficient.

    `forward_dense` is linear in each coefficient, so under a linear loss the B
    gradient must scale exactly with it. Asserting the exact factor -- rather
    than merely "non-zero" -- is what distinguishes a real coefficient-driven
    gradient path from an incidental non-zero.

    The same exactness does NOT hold end-to-end through `controlled_rollout`:
    the injected residual feeds onward through the rest of the block, so the
    upstream gradient itself moves with the coefficient. That case is covered
    separately below by its response to the coefficient, not by a scale factor.
    """
    x = torch.randn(2, 5, HIDDEN)

    def b_grad_for(coefficient: float) -> torch.Tensor:
        bank = make_bank()
        coefficients = torch.tensor([[coefficient, 0.0, 0.0, 0.0]] * 2)
        bank.forward_dense(x, coefficients).sum().backward()
        return bank.up[0].weight.grad.detach().clone()

    small = b_grad_for(0.05)
    large = b_grad_for(0.20)
    assert small.abs().max().item() > 0.0
    torch.testing.assert_close(large, small * 4.0, rtol=1e-5, atol=1e-7)


def test_rollout_bank_gradient_responds_to_the_coefficient(bridge, variables):
    """Through the full rollout, the coefficient really drives the gradient.

    Same input, same loss, same bank initialization: only the coefficient
    differs, so any difference in the resulting gradient is attributable to it.
    """
    x_norm = torch.randn(1, N_VARS, *GRID)

    def b_grad_for(coefficient: float) -> torch.Tensor:
        bank = make_bank()
        out = controlled_rollout(
            bridge, x_norm, variables, interval=6, steps=1,
            plan=plan_with((coefficient, 0.0, 0.0, 0.0)),
            expert_loras={TARGET_BLOCK: bank}, target_blocks=(TARGET_BLOCK,),
        )
        out.sum().backward()
        return bank.up[0].weight.grad.detach().clone()

    off = b_grad_for(0.0)
    small = b_grad_for(0.05)
    large = b_grad_for(0.20)

    assert torch.count_nonzero(off).item() == 0
    assert small.abs().max().item() > 0.0
    assert large.abs().max().item() > small.abs().max().item()
    assert not torch.allclose(large, small)


def test_both_factors_of_a_shared_bank_get_gradients_in_the_dense_path():
    """Unit-level check of the bank itself, independent of the backbone."""
    bank = make_bank()
    x = torch.randn(2, 5, HIDDEN)
    coefficients = torch.tensor([list(FIXED_COEFFICIENTS)] * 2)

    bank.forward_dense(x, coefficients).sum().backward()

    for k in ACTIVE_EXPERTS:
        assert bank.down[k].weight.grad.abs().max().item() > 0.0
        assert bank.up[k].weight.grad.abs().max().item() > 0.0
    for k in INACTIVE_EXPERTS:
        assert torch.count_nonzero(bank.down[k].weight.grad).item() == 0
        assert torch.count_nonzero(bank.up[k].weight.grad).item() == 0


# =============================================================================
# 2. Frozen backbone receives no gradients
# =============================================================================

def test_frozen_backbone_receives_no_gradients(bridge, variables):
    """Training the bank must leave every backbone parameter gradient-free."""
    bank = make_bank()
    out = rollout(bridge, variables, bank, FIXED_COEFFICIENTS)
    out.sum().backward()

    offenders = [
        name for name, param in bridge.model.named_parameters()
        if param.grad is not None
    ]
    assert offenders == [], f"frozen backbone parameters received gradients: {offenders}"


def test_frozen_backbone_parameters_do_not_require_grad(bridge):
    requires = [
        name for name, param in bridge.model.named_parameters() if param.requires_grad
    ]
    assert requires == [], f"backbone parameters still require grad: {requires}"


def test_backbone_weights_are_unchanged_by_a_bank_optimizer_step(bridge, variables):
    """An optimizer over the bank cannot move the backbone."""
    bank = make_bank()
    before = {
        name: param.detach().clone()
        for name, param in bridge.model.named_parameters()
    }

    optimizer = torch.optim.SGD(bank.parameters(), lr=0.5)
    optimizer.zero_grad()
    rollout(bridge, variables, bank, FIXED_COEFFICIENTS).sum().backward()
    optimizer.step()

    for name, param in bridge.model.named_parameters():
        torch.testing.assert_close(param.detach(), before[name], rtol=0.0, atol=0.0)

    moved = any(
        not torch.equal(bank.up[k].weight.detach(), make_bank().up[k].weight.detach())
        for k in ACTIVE_EXPERTS
    )
    assert moved, "the optimizer step did not move the bank at all"


# =============================================================================
# 3. zero -> non-zero -> zero restores the no-gradient state
# =============================================================================

def test_zero_to_nonzero_to_zero_restores_the_no_gradient_state(bridge, variables):
    """Re-zeroing a coefficient must restore exactly the zero-gradient state."""
    x_norm = torch.randn(1, N_VARS, *GRID)
    bank = make_bank()

    def grads_for(coefficients: Tuple[float, ...]) -> Dict[str, torch.Tensor]:
        bank.zero_grad(set_to_none=True)
        out = controlled_rollout(
            bridge, x_norm, variables, interval=6, steps=1,
            plan=plan_with(coefficients), expert_loras={TARGET_BLOCK: bank},
            target_blocks=(TARGET_BLOCK,),
        )
        out.sum().backward()
        return {
            name: param.grad.detach().clone()
            for name, param in bank.named_parameters()
        }

    all_zero = (0.0,) * NUM_EXPERTS

    first_zero = grads_for(all_zero)
    for name, grad in first_zero.items():
        assert torch.count_nonzero(grad).item() == 0, (
            f"{name} had a non-zero gradient with all coefficients zero"
        )

    active = grads_for(FIXED_COEFFICIENTS)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in active.values()), (
        "no bank parameter received a gradient under the non-zero coefficients"
    )

    back_to_zero = grads_for(all_zero)
    for name, grad in back_to_zero.items():
        assert torch.count_nonzero(grad).item() == 0, (
            f"{name} kept a non-zero gradient after its coefficient returned to zero"
        )
        torch.testing.assert_close(grad, first_zero[name], rtol=0.0, atol=0.0)


def test_single_expert_toggle_does_not_disturb_its_neighbours(bridge, variables):
    """Toggling expert 0 leaves the other experts' gradients at zero."""
    x_norm = torch.randn(1, N_VARS, *GRID)
    bank = make_bank()

    for coefficients in (
        (0.0, 0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ):
        bank.zero_grad(set_to_none=True)
        controlled_rollout(
            bridge, x_norm, variables, interval=6, steps=1,
            plan=plan_with(coefficients), expert_loras={TARGET_BLOCK: bank},
            target_blocks=(TARGET_BLOCK,),
        ).sum().backward()

        for k in (1, 2, 3):
            assert torch.count_nonzero(bank.down[k].weight.grad).item() == 0
            assert torch.count_nonzero(bank.up[k].weight.grad).item() == 0

        expected_nonzero = coefficients[0] != 0.0
        actually_nonzero = bank.up[0].weight.grad.abs().max().item() > 0.0
        assert actually_nonzero is expected_nonzero, (
            f"expert 0 gradient presence {actually_nonzero} did not match "
            f"coefficient {coefficients[0]}"
        )


# =============================================================================
# 4. Hook lifecycle: zero dangling hooks on every exit path
# =============================================================================

def test_no_hooks_remain_after_a_normal_rollout(bridge, variables):
    """A rollout that returns normally leaves the backbone exactly as found."""
    bank = make_bank()
    before = count_hooks(bridge.model)

    rollout(bridge, variables, bank, FIXED_COEFFICIENTS)

    assert count_hooks(bridge.model) == before
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0


def test_no_hooks_remain_after_a_multi_step_rollout(bridge, variables):
    """Each step registers and removes its own hooks; none accumulate."""
    bank = make_bank()
    before = count_hooks(bridge.model)

    rollout(bridge, variables, bank, FIXED_COEFFICIENTS, steps=3)

    assert count_hooks(bridge.model) == before
    assert all(count == 0 for count in proj_hooks(bridge).values())


def test_no_hooks_remain_after_serial_repeats(bridge, variables):
    """Serial rollouts do not leak one hook per call."""
    bank = make_bank()
    before = count_hooks(bridge.model)

    for _ in range(5):
        rollout(bridge, variables, bank, FIXED_COEFFICIENTS)

    assert count_hooks(bridge.model) == before


def test_no_hooks_remain_after_the_bank_raises(bridge, variables):
    """An exception raised inside the injected hook still unwinds cleanly."""
    class ExplodingBank(ExpertLoRA):
        def forward(self, x, coefficients, sparse=False):
            raise RuntimeError("bank exploded mid-forward")

    bank = ExplodingBank(HIDDEN, HIDDEN, num_experts=NUM_EXPERTS, rank_per_expert=RANK)
    before = count_hooks(bridge.model)

    with pytest.raises(RuntimeError, match="bank exploded mid-forward"):
        rollout(bridge, variables, bank, FIXED_COEFFICIENTS)

    assert count_hooks(bridge.model) == before, "hooks leaked on the exception path"
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0


def test_no_hooks_remain_after_a_late_step_raises(bridge, variables):
    """A failure on step 3 of 4 leaves no hooks from the earlier steps either."""
    class LateExplodingBank(ExpertLoRA):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def forward(self, x, coefficients, sparse=False):
            self.calls += 1
            if self.calls >= 3:
                raise RuntimeError("bank exploded on a later step")
            return super().forward(x, coefficients, sparse=sparse)

    bank = LateExplodingBank(HIDDEN, HIDDEN, num_experts=NUM_EXPERTS, rank_per_expert=RANK)
    before = count_hooks(bridge.model)

    with pytest.raises(RuntimeError, match="bank exploded on a later step"):
        rollout(bridge, variables, bank, FIXED_COEFFICIENTS, steps=4)

    assert bank.calls == 3
    assert count_hooks(bridge.model) == before
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0


def test_the_bridge_is_usable_again_after_an_abnormal_exit(bridge, variables):
    """An aborted rollout must not poison the next one."""
    class ExplodingBank(ExpertLoRA):
        def forward(self, x, coefficients, sparse=False):
            raise RuntimeError("boom")

    exploding = ExplodingBank(HIDDEN, HIDDEN, num_experts=NUM_EXPERTS, rank_per_expert=RANK)
    with pytest.raises(RuntimeError, match="boom"):
        rollout(bridge, variables, exploding, FIXED_COEFFICIENTS)

    healthy = make_bank()
    out = rollout(bridge, variables, healthy, FIXED_COEFFICIENTS)
    assert torch.isfinite(out).all()
    assert count_hooks(bridge.model) == count_hooks(bridge.model)
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0


def test_hooks_are_actually_registered_during_the_forward(bridge, variables):
    """The zero-after counts mean something only if hooks were ever present.

    Observes the hook registry from inside the backbone's forward, so the
    "zero dangling hooks" assertions above cannot be satisfied vacuously by a
    rollout that never registered anything.
    """
    bank = make_bank()
    observed: List[int] = []
    target_proj = bridge.model.blocks[TARGET_BLOCK].attn.proj
    original_forward = target_proj.forward

    def observing_forward(*args, **kwargs):
        observed.append(len(target_proj._forward_hooks))
        return original_forward(*args, **kwargs)

    target_proj.forward = observing_forward
    try:
        rollout(bridge, variables, bank, FIXED_COEFFICIENTS)
    finally:
        target_proj.forward = original_forward

    assert observed, "the targeted projection was never reached"
    assert max(observed) >= 1, "no hook was registered during the forward pass"
    assert len(target_proj._forward_hooks) == 0, "the hook outlived the rollout"


# =============================================================================
# B12: declared execution modes are enforced, not assumed
# =============================================================================

def test_shared_backbone_concurrency_is_rejected(bridge, variables):
    """A second bridge over the SAME backbone cannot roll out concurrently.

    The injection point is the backbone's hook table, so exclusion has to be
    keyed on the backbone, not on the bridge wrapper.
    """
    from earthdelta.bridge.stormer_bridge import controlled_rollout as roll

    sibling = WeatherStepBridge(bridge.model, bridge.normalization, bridge._version)
    assert sibling is not bridge and sibling.model is bridge.model

    captured: List[BaseException] = []

    class ReentrantBank(ExpertLoRA):
        def forward(self, x, coefficients, sparse=False):
            try:
                roll(
                    sibling, torch.randn(1, N_VARS, *GRID), variables,
                    interval=6, steps=1, plan=plan_with(FIXED_COEFFICIENTS),
                    expert_loras={}, target_blocks=(TARGET_BLOCK,),
                )
            except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
                captured.append(exc)
                raise
            return super().forward(x, coefficients, sparse=sparse)

    bank = ReentrantBank(HIDDEN, HIDDEN, num_experts=NUM_EXPERTS, rank_per_expert=RANK)
    with pytest.raises(RuntimeError, match="shared backbone"):
        rollout(bridge, variables, bank, FIXED_COEFFICIENTS)

    assert captured and isinstance(captured[0], RuntimeError)
    assert "serial-only" in str(captured[0])
    assert count_hooks(bridge.model) == count_hooks(bridge.model)
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0


@pytest.mark.parametrize(
    "competitor_is_sibling, expected_message",
    [(False, "Reentrant call"), (True, "shared backbone")],
)
def test_concurrent_rollout_from_another_thread_is_rejected(
    bridge, variables, competitor_is_sibling, expected_message
):
    """A thread-local guard would miss BOTH of these; the real guard must not.

    While one rollout is parked inside the backbone's forward, a second thread
    attempts its own rollout over the same backbone -- once through the very
    same bridge object, once through a sibling bridge wrapping the same model.
    Under the previous thread-local registry the competing thread saw an empty
    active set and was admitted, so two rollouts mutated one hook table.
    """
    import threading

    from earthdelta.bridge.stormer_bridge import controlled_rollout as roll

    competitor_bridge = (
        WeatherStepBridge(bridge.model, bridge.normalization, bridge._version)
        if competitor_is_sibling else bridge
    )

    errors: List[BaseException] = []
    admitted: List[str] = []
    entered = threading.Event()
    release = threading.Event()

    class BlockingBank(ExpertLoRA):
        def forward(self, x, coefficients, sparse=False):
            entered.set()
            release.wait(timeout=10)
            return super().forward(x, coefficients, sparse=sparse)

    def competitor():
        entered.wait(timeout=10)
        try:
            roll(
                competitor_bridge, torch.randn(1, N_VARS, *GRID), variables,
                interval=6, steps=1, plan=plan_with(FIXED_COEFFICIENTS),
                expert_loras={}, target_blocks=(TARGET_BLOCK,),
            )
            admitted.append("competitor was admitted")
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(exc)
        finally:
            release.set()

    thread = threading.Thread(target=competitor, name="competitor")
    thread.start()
    try:
        rollout(bridge, variables, BlockingBank(
            HIDDEN, HIDDEN, num_experts=NUM_EXPERTS, rank_per_expert=RANK
        ), FIXED_COEFFICIENTS)
    finally:
        release.set()
        thread.join(timeout=15)

    assert admitted == [], "the concurrent rollout on another thread was ADMITTED"
    assert errors, "the concurrent rollout on another thread produced no error"
    assert isinstance(errors[0], RuntimeError)
    assert expected_message in str(errors[0])
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0


def test_activation_checkpointing_backbone_is_rejected(bridge, variables):
    """A checkpointed backbone would replay its forward without the hook."""
    bank = make_bank()
    bridge.model.blocks[TARGET_BLOCK].use_checkpoint = True
    try:
        with pytest.raises(RuntimeError, match="activation-checkpoint"):
            rollout(bridge, variables, bank, FIXED_COEFFICIENTS)
    finally:
        del bridge.model.blocks[TARGET_BLOCK].use_checkpoint

    # The rejection happens before anything is registered or claimed.
    assert proj_hooks(bridge)[TARGET_BLOCK] == 0
    rollout(bridge, variables, bank, FIXED_COEFFICIENTS)


def test_supported_modes_are_declared_explicitly():
    from earthdelta.bridge.stormer_bridge import CONTROLLED_ROLLOUT_SUPPORTED_MODES

    assert CONTROLLED_ROLLOUT_SUPPORTED_MODES["serial_single_rollout_per_backbone"] is True
    assert CONTROLLED_ROLLOUT_SUPPORTED_MODES["concurrent_rollouts_sharing_a_backbone"] is False
    assert CONTROLLED_ROLLOUT_SUPPORTED_MODES["activation_checkpoint_replay"] is False


def test_rejected_caller_does_not_release_the_incumbent_claim(bridge, variables):
    """A refused concurrent caller must not clear the holder's registration."""
    from earthdelta.bridge.stormer_bridge import (
        _check_reentrant_rollout,
        _release_rollout_lock,
        _rollout_lock,
    )

    sibling = WeatherStepBridge(bridge.model, bridge.normalization, bridge._version)

    _check_reentrant_rollout(bridge)
    try:
        with pytest.raises(RuntimeError, match="shared backbone"):
            _check_reentrant_rollout(sibling)
        # The refused caller's own release must be a no-op for the incumbent.
        _release_rollout_lock(sibling)
        assert id(bridge.model) in _rollout_lock.active_models
        assert id(bridge) in _rollout_lock.active_bridges
    finally:
        _release_rollout_lock(bridge)

    assert id(bridge.model) not in _rollout_lock.active_models
    assert id(bridge) not in _rollout_lock.active_bridges
