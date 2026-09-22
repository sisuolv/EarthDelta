"""Tests for the climate_learn unpickling compatibility shim in stormer_bridge.

The pinned Stormer .ckpt files are PyTorch-Lightning checkpoints whose pickle
stream references five classes from `climate_learn`, a training framework that is
not installed. This is verified against a *synthetic* fixture built here; these
tests never read the real 5.2GB checkpoint.
"""
from __future__ import annotations

import pickletools
import sys
import zipfile

import pytest
import torch

from earthdelta.bridge.stormer_bridge import (
    _CLIMATE_LEARN_PICKLE_CLASSES,
    _ClimateLearnCompatPlaceholder,
    _ensure_climate_learn_pickle_compat,
)


# The exact dotted paths observed in the real checkpoint's pickle opcodes.
EXPECTED_CLASS_PATHS = {
    ("climate_learn.models.lr_scheduler", "LinearWarmupCosineAnnealingLR"),
    ("climate_learn.metrics.metrics", "LatWeightedMSE"),
    ("climate_learn.metrics.metrics", "LatWeightedRMSE"),
    ("climate_learn.metrics.utils", "MetricsMetaInfo"),
    ("climate_learn.transforms.denormalize", "Denormalize"),
}

_PACKAGE_SOURCES = {
    "climate_learn/__init__.py": "",
    "climate_learn/models/__init__.py": "",
    "climate_learn/models/lr_scheduler.py": (
        "class LinearWarmupCosineAnnealingLR:\n"
        "    def __init__(self, warmup_epochs=5, max_epochs=100):\n"
        "        self.warmup_epochs = warmup_epochs\n"
        "        self.max_epochs = max_epochs\n"
    ),
    "climate_learn/metrics/__init__.py": "",
    "climate_learn/metrics/metrics.py": (
        "class LatWeightedMSE:\n"
        "    def __init__(self, meta_info=None):\n"
        "        self.metainfo = meta_info\n"
        "\n"
        "class LatWeightedRMSE:\n"
        "    def __init__(self, meta_info=None):\n"
        "        self.metainfo = meta_info\n"
    ),
    "climate_learn/metrics/utils.py": (
        "class MetricsMetaInfo:\n"
        "    def __init__(self, in_vars=None, out_vars=None, lat=None):\n"
        "        self.in_vars = in_vars\n"
        "        self.out_vars = out_vars\n"
        "        self.lat = lat\n"
    ),
    "climate_learn/transforms/__init__.py": "",
    "climate_learn/transforms/denormalize.py": (
        "class Denormalize:\n"
        "    def __init__(self, mean=None, std=None):\n"
        "        self.mean = mean\n"
        "        self.std = std\n"
    ),
}


def _purge_climate_learn_from_sys_modules() -> None:
    """Put the interpreter back in the state the GPU worker crashed in."""
    for name in [m for m in sys.modules if m == "climate_learn" or m.startswith("climate_learn.")]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_import_state():
    """Snapshot/restore sys.modules + sys.path so these tests never leak globally."""
    saved_path = list(sys.path)
    saved_modules = {
        k: v for k, v in sys.modules.items()
        if k == "climate_learn" or k.startswith("climate_learn.")
    }
    try:
        yield
    finally:
        sys.path[:] = saved_path
        _purge_climate_learn_from_sys_modules()
        sys.modules.update(saved_modules)


@pytest.fixture(scope="module")
def fake_ckpt(tmp_path_factory):
    """Build a synthetic Lightning-style checkpoint referencing the five class paths.

    The source package is materialized on disk and imported normally, so the
    *writing* side does not depend on the shim under test.
    """
    root = tmp_path_factory.mktemp("climate_learn_pkg")
    for rel, src in _PACKAGE_SOURCES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src)

    sys.path.insert(0, str(root))
    try:
        from climate_learn.metrics.metrics import LatWeightedMSE, LatWeightedRMSE
        from climate_learn.metrics.utils import MetricsMetaInfo
        from climate_learn.models.lr_scheduler import LinearWarmupCosineAnnealingLR
        from climate_learn.transforms.denormalize import Denormalize

        meta = MetricsMetaInfo(in_vars=["2m_temperature"], out_vars=["2m_temperature"], lat=[0.0, 1.0])
        checkpoint = {
            "state_dict": {
                "net.token_embeds.weight": torch.ones(2, 3),
                "net.head.bias": torch.zeros(3),
                "no_prefix.weight": torch.full((2,), 7.0),
            },
            "hyper_parameters": {
                # Bare class reference (no instance), as in the real checkpoint.
                "lr_scheduler_cls": LinearWarmupCosineAnnealingLR,
                "lr_scheduler_args": {},
                "train_loss": LatWeightedMSE(meta_info=meta),
                "val_loss": [LatWeightedRMSE(meta_info=meta)],
                "denorm": Denormalize(mean=[0.0], std=[1.0]),
            },
            "epoch": 3,
        }
        path = root.parent / "synthetic_lightning.ckpt"
        torch.save(checkpoint, str(path))
    finally:
        sys.path.remove(str(root))
        _purge_climate_learn_from_sys_modules()

    return str(path)


def test_fixture_references_exactly_the_five_class_paths(fake_ckpt):
    """The synthetic fixture reproduces the real checkpoint's GLOBAL opcodes.

    Uses the same cheap diagnosis method applied to the real checkpoint:
    read archive/data.pkl and walk opcodes, without unpickling anything.
    """
    with zipfile.ZipFile(fake_ckpt) as zf:
        raw = zf.read(next(n for n in zf.namelist() if n.endswith("data.pkl")))

    found = set()
    for opcode, arg, _pos in pickletools.genops(raw):
        if opcode.name == "GLOBAL" and arg and arg.startswith("climate_learn"):
            module_path, class_name = arg.split(" ")
            found.add((module_path, class_name))

    assert found == EXPECTED_CLASS_PATHS
    assert set(_CLIMATE_LEARN_PICKLE_CLASSES) == EXPECTED_CLASS_PATHS


def test_without_shim_torch_load_raises_module_not_found(fake_ckpt):
    """Baseline: this is the exact GPU-worker failure the shim exists to fix."""
    _purge_climate_learn_from_sys_modules()

    with pytest.raises(ModuleNotFoundError) as excinfo:
        torch.load(fake_ckpt, map_location="cpu", weights_only=False)

    assert "climate_learn" in str(excinfo.value)


def test_with_shim_torch_load_succeeds_and_state_dict_is_intact(fake_ckpt):
    """After the shim, torch.load returns and checkpoint['state_dict'] is usable."""
    _purge_climate_learn_from_sys_modules()
    _ensure_climate_learn_pickle_compat()

    checkpoint = torch.load(fake_ckpt, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]
    assert set(state_dict) == {
        "net.token_embeds.weight",
        "net.head.bias",
        "no_prefix.weight",
    }
    assert torch.equal(state_dict["net.token_embeds.weight"], torch.ones(2, 3))
    assert torch.equal(state_dict["net.head.bias"], torch.zeros(3))
    assert torch.equal(state_dict["no_prefix.weight"], torch.full((2,), 7.0))
    assert checkpoint["epoch"] == 3

    # The climate_learn objects come back as inert placeholders carrying their
    # pickled __dict__; nothing of theirs is executed.
    hparams = checkpoint["hyper_parameters"]
    assert issubclass(hparams["lr_scheduler_cls"], _ClimateLearnCompatPlaceholder)
    assert isinstance(hparams["train_loss"], _ClimateLearnCompatPlaceholder)
    assert isinstance(hparams["val_loss"][0], _ClimateLearnCompatPlaceholder)
    assert isinstance(hparams["denorm"], _ClimateLearnCompatPlaceholder)
    assert hparams["denorm"].__dict__ == {"mean": [0.0], "std": [1.0]}
    assert hparams["train_loss"].metainfo.in_vars == ["2m_temperature"]


def test_net_prefix_stripping_still_works_on_shim_loaded_state_dict(fake_ckpt):
    """The shim does not disturb the 'net.' key-stripping contract."""
    _purge_climate_learn_from_sys_modules()
    _ensure_climate_learn_pickle_compat()
    state_dict = torch.load(fake_ckpt, map_location="cpu", weights_only=False)["state_dict"]

    stripped = {(k[4:] if k.startswith("net.") else k): v for k, v in state_dict.items()}

    assert set(stripped) == {"token_embeds.weight", "head.bias", "no_prefix.weight"}


def test_shim_is_idempotent(fake_ckpt):
    """Repeated calls neither raise nor replace already-registered placeholders."""
    _purge_climate_learn_from_sys_modules()
    _ensure_climate_learn_pickle_compat()
    first = sys.modules["climate_learn.metrics.metrics"].LatWeightedMSE

    for _ in range(3):
        _ensure_climate_learn_pickle_compat()

    assert sys.modules["climate_learn.metrics.metrics"].LatWeightedMSE is first
    torch.load(fake_ckpt, map_location="cpu", weights_only=False)


def test_shim_does_not_clobber_a_real_installation(tmp_path):
    """If climate_learn is genuinely importable, the shim must be a no-op."""
    root = tmp_path / "real_pkg"
    for rel, src in _PACKAGE_SOURCES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src)

    _purge_climate_learn_from_sys_modules()
    sys.path.insert(0, str(root))

    from climate_learn.metrics.metrics import LatWeightedMSE as RealLatWeightedMSE

    _ensure_climate_learn_pickle_compat()

    assert sys.modules["climate_learn.metrics.metrics"].LatWeightedMSE is RealLatWeightedMSE
    assert not issubclass(RealLatWeightedMSE, _ClimateLearnCompatPlaceholder)
    # A real instance still carries real behavior, i.e. __init__ ran.
    assert RealLatWeightedMSE(meta_info="m").metainfo == "m"


def test_shim_scope_is_limited_to_the_five_names():
    """The shim is not a general 'stub any missing module' mechanism."""
    _purge_climate_learn_from_sys_modules()
    _ensure_climate_learn_pickle_compat()

    registered = {m for m in sys.modules if m == "climate_learn" or m.startswith("climate_learn.")}
    assert registered == {
        "climate_learn",
        "climate_learn.models",
        "climate_learn.models.lr_scheduler",
        "climate_learn.metrics",
        "climate_learn.metrics.metrics",
        "climate_learn.metrics.utils",
        "climate_learn.transforms",
        "climate_learn.transforms.denormalize",
    }

    # Unrelated missing third-party imports are still real errors.
    assert "definitely_not_a_real_package_xyz" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        __import__("definitely_not_a_real_package_xyz")

    # Only the five verified attributes exist on the stub modules.
    metrics_mod = sys.modules["climate_learn.metrics.metrics"]
    assert not hasattr(metrics_mod, "SomeOtherMetric")
