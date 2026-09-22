"""Immutable execution/cache contracts and program specifications.

Timestamps are integer UTC seconds. EditPlan generalizes from rank-group masks
to expert-dictionary coefficients with an application window.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field, fields as dataclass_fields
from collections.abc import Sequence, Mapping
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable, Union
import hashlib
import json
import math
import numpy as np
import torch
from torch import Tensor


# =============================================================================
# Gate Identity Configuration
# =============================================================================

# Official Stormer reference commit from which inference semantics are pinned
OFFICIAL_STORMER_COMMIT = "58dfee5a6037399a40fefd492bc00421e0c885a8"


# =============================================================================
# Shared normalization identity serialization (bridge + exporter)
# =============================================================================
#
# ONE canonical serialization is used to derive the "effective normalization
# constants" identity digest on BOTH sides of the parity comparison:
#
#   * earthdelta/bridge/stormer_bridge.py :: NormalizationContract.digest
#   * scripts/export_upstream_reference.py :: compute_normalization_digest
#
# Only the *identity serialization* is shared. The actual numeric transforms
# stay independent: the exporter still builds its own torchvision Normalize
# transforms from the official NPZ files and never calls NormalizationContract.
#
# Prior to this schema the two sides disagreed on real assets (bridge
# ed563eb5e55fc118 vs exporter e2871e376eb9d9e8 on
# reference/stormer/normalization_constants) for two reasons:
#   1. field order - the bridge emitted every interval's diff_mean and then
#      every interval's diff_std, the exporter interleaved mean/std per interval;
#   2. dtype - the bridge cast NPZ contents to float32 before hashing while the
#      exporter hashed the raw on-disk dtype (float64 for most variables).
#
# The schema below pins field order explicitly and pins the digest dtype to
# float32 little-endian. float32 is *not* a cosmetic relabel: it is the dtype
# actually used during execution on both sides. The bridge stores its constants
# as torch float32 tensors, and torchvision's Normalize casts mean/std to the
# input tensor's dtype (float32) before applying them, so the bytes hashed here
# are exactly the bytes the forward pass consumes.

NORM_IDENTITY_SCHEMA_VERSION = "ed-norm-identity/1"
NORM_IDENTITY_DTYPE = "float32"
NORM_IDENTITY_BYTE_ORDER = "little"

# Explicit numpy dtype string: little-endian IEEE-754 binary32.
_NORM_IDENTITY_NUMPY_DTYPE = "<f4"

# Schema label attributed to an identity string that carries no schema prefix.
# Such values come from references exported before NORM_IDENTITY_SCHEMA_VERSION
# existed; they must never be silently reinterpreted under the new schema.
LEGACY_NORM_IDENTITY_SCHEMA = "legacy/unversioned"

_IDENTITY_SEPARATOR = ":"


class NormalizationIdentityFormatError(ValueError):
    """Raised when two normalization identities use different schema versions.

    A legacy (unversioned) reference digest compared against a
    ``ed-norm-identity/1`` actual digest is a *format* mismatch, not a value
    mismatch. Coercing one into the other would silently upgrade a stale
    reference into something that looks freshly verified, so the comparison
    fails closed with this error instead.
    """


def _canonical_identity_array(values: Any) -> np.ndarray:
    """Return a contiguous 1-D float32 little-endian view of ``values``.

    Accepts torch tensors and numpy arrays/sequences so that the bridge (torch)
    and the exporter (numpy) can feed the same serializer.
    """
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return np.ascontiguousarray(array.reshape(-1), dtype=np.dtype(_NORM_IDENTITY_NUMPY_DTYPE))


def _identity_field_chunk(name: str, values: Any) -> List[bytes]:
    """Serialize one named field with an explicit shape header."""
    array = _canonical_identity_array(values)
    return [
        f"field={name}\n".encode(),
        f"shape=({array.shape[0]},)\n".encode(),
        array.tobytes(),
        b"\n",
    ]


def serialize_normalization_identity(
    *,
    policy: str,
    variables: Sequence[str],
    inp_mean: Any,
    inp_std: Any,
    diff_mean: Mapping[int, Any],
    diff_std: Mapping[int, Any],
    schema_version: str = NORM_IDENTITY_SCHEMA_VERSION,
) -> bytes:
    """Canonical byte serialization of effective normalization constants.

    The payload pins, in this order:
      schema version, policy name, digest dtype, digest byte order, variable
      count, variable order, the explicit sorted interval set, and then the
      fields in a fixed order: ``inp_mean``, ``inp_std``, then for each interval
      in ascending order ``diff_mean_<interval>`` followed immediately by
      ``diff_std_<interval>`` (interleaved, not grouped by field kind).

    Args:
        policy: Normalization policy name (part of the identity, not metadata).
        variables: Variable names in channel order.
        inp_mean / inp_std: Per-variable input normalization constants.
        diff_mean / diff_std: interval -> per-variable diff constants. These are
            the *effective* constants, i.e. what the forward pass actually uses
            (zeros for diff_mean under the official zero-diff-mean policy).
        schema_version: Serialization schema version.

    Returns:
        The canonical payload bytes.
    """
    intervals = sorted({int(i) for i in diff_mean.keys()} | {int(i) for i in diff_std.keys()})
    parts: List[bytes] = [
        f"schema={schema_version}\n".encode(),
        f"policy={policy}\n".encode(),
        f"dtype={NORM_IDENTITY_DTYPE}\n".encode(),
        f"numpy_dtype={_NORM_IDENTITY_NUMPY_DTYPE}\n".encode(),
        f"byte_order={NORM_IDENTITY_BYTE_ORDER}\n".encode(),
        f"n_variables={len(variables)}\n".encode(),
        ("variables=" + ",".join(variables) + "\n").encode(),
        ("intervals=" + ",".join(str(i) for i in intervals) + "\n").encode(),
    ]

    parts.extend(_identity_field_chunk("inp_mean", inp_mean))
    parts.extend(_identity_field_chunk("inp_std", inp_std))

    for interval in intervals:
        for kind, table in (("diff_mean", diff_mean), ("diff_std", diff_std)):
            name = f"{kind}_{interval}"
            if interval in table:
                parts.extend(_identity_field_chunk(name, table[interval]))
            else:
                # An absent field is itself part of the identity.
                parts.append(f"field={name}\nshape=absent\n".encode())

    return b"".join(parts)


def compute_normalization_identity_digest(
    *,
    policy: str,
    variables: Sequence[str],
    inp_mean: Any,
    inp_std: Any,
    diff_mean: Mapping[int, Any],
    diff_std: Mapping[int, Any],
    schema_version: str = NORM_IDENTITY_SCHEMA_VERSION,
) -> str:
    """Return the bare 16-hex digest of the canonical identity payload."""
    payload = serialize_normalization_identity(
        policy=policy,
        variables=variables,
        inp_mean=inp_mean,
        inp_std=inp_std,
        diff_mean=diff_mean,
        diff_std=diff_std,
        schema_version=schema_version,
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def format_normalization_identity(
    digest_hex: str, schema_version: str = NORM_IDENTITY_SCHEMA_VERSION
) -> str:
    """Return the schema-qualified identity string ``<schema>:<digest>``."""
    return f"{schema_version}{_IDENTITY_SEPARATOR}{digest_hex}"


def parse_normalization_identity(value: str) -> Tuple[str, str]:
    """Split ``<schema>:<digest>`` into ``(schema, digest)``.

    A value with no schema prefix is reported as
    ``LEGACY_NORM_IDENTITY_SCHEMA`` so that callers can fail closed rather than
    guess which schema produced it.
    """
    if not isinstance(value, str) or not value:
        raise NormalizationIdentityFormatError(
            f"Normalization identity must be a non-empty string, got {value!r}"
        )
    if _IDENTITY_SEPARATOR in value:
        schema, _, digest = value.partition(_IDENTITY_SEPARATOR)
        if not schema or not digest:
            raise NormalizationIdentityFormatError(
                f"Malformed normalization identity: {value!r}"
            )
        return schema, digest
    return LEGACY_NORM_IDENTITY_SCHEMA, value


def compare_normalization_identity(expected: str, actual: str) -> bool:
    """Compare two schema-qualified normalization identities.

    Raises:
        NormalizationIdentityFormatError: If the two values were produced by
            different serialization schemas (e.g. a legacy unversioned digest
            compared against a ``ed-norm-identity/1`` digest). The comparison
            fails closed instead of coercing one format into the other.
    """
    expected_schema, expected_digest = parse_normalization_identity(expected)
    actual_schema, actual_digest = parse_normalization_identity(actual)
    if expected_schema != actual_schema:
        raise NormalizationIdentityFormatError(
            "normalization identity format version mismatch: expected schema "
            f"{expected_schema!r} but actual value uses schema {actual_schema!r}. "
            "Re-export the reference with the current schema; legacy references "
            "are never upgraded in place."
        )
    return expected_digest == actual_digest


def compute_normalization_asset_sha256(
    norm_dir: Union[str, Path], filenames: Optional[Sequence[str]] = None
) -> str:
    """Hash the raw on-disk NPZ bytes of a normalization asset directory.

    This is deliberately DISTINCT from the effective-constants identity digest
    produced by :func:`compute_normalization_identity_digest`. This one answers
    "are these the same files?"; the identity digest answers "do these produce
    the same effective normalization constants under this policy?". The two are
    recorded under separate manifest keys and are never compared to each other.
    """
    directory = Path(norm_dir)
    if filenames is None:
        names = sorted(p.name for p in directory.glob("*.npz"))
    else:
        names = sorted(filenames)
    digest = hashlib.sha256()
    digest.update(b"ed-norm-npz-content/1\n")
    for name in names:
        path = directory / name
        digest.update(f"file={name}\n".encode())
        if not path.exists():
            digest.update(b"absent\n")
            continue
        payload = path.read_bytes()
        digest.update(f"size={len(payload)}\n".encode())
        digest.update(payload)
    return digest.hexdigest()


def compute_coordinate_digest(lat: Any, lon: Any) -> str:
    """Identity digest of the grid coordinate values (not just their shape).

    Two grids with identical shape but different latitude/longitude values are
    different coordinate identities and must not compare equal.
    """
    parts = [
        f"schema={NORM_IDENTITY_SCHEMA_VERSION}\n".encode(),
        b"kind=coordinates\n",
        f"dtype={NORM_IDENTITY_DTYPE}\n".encode(),
        f"byte_order={NORM_IDENTITY_BYTE_ORDER}\n".encode(),
    ]
    parts.extend(_identity_field_chunk("lat", lat))
    parts.extend(_identity_field_chunk("lon", lon))
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


def compute_variables_digest(variables: Sequence[str]) -> str:
    """Identity digest of the variable list INCLUDING its order."""
    payload = ("ed-variables/1\n" + "\n".join(variables)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class GateIdentityConfig:
    """Shared identity configuration for exporter and gate.

    This configuration ensures both exporter and gate use exactly the same
    identity for checkpoint, normalization, variables, and rollout parameters.
    The output directory is namespaced by this identity to prevent silent
    overwrites between different configurations (e.g., ps2 vs ps4).
    """
    # Checkpoint identity
    checkpoint_path: str
    expected_checkpoint_sha256: str
    patch_size: int  # 2 or 4 (ps4 is the mainline per research_spec_v6.yaml:33)

    # Normalization identity
    normalization_policy: str  # "official_zero_diff_mean" or "legacy"
    normalization_dir: str

    # Variable/coordinate identity
    variables: Tuple[str, ...]
    grid_shape: Tuple[int, int]  # (H, W), e.g., (128, 256)

    # Rollout configuration
    interval_hours: int = 6
    rollout_steps: Tuple[int, ...] = (1, 4)  # Steps to run/verify (1-step and 4-step)

    # Reference source identity
    official_commit: str = OFFICIAL_STORMER_COMMIT

    # -------------------------------------------------------------------------
    # Frozen expected identity (enrolled BEFORE export, never back-filled from
    # what a run happened to observe). Everything below has a default so that
    # existing constructions stay valid; a gate run with these unset fails
    # closed rather than certifying an unbound identity.
    # -------------------------------------------------------------------------

    # Backbone geometry, needed to reconstruct the expected ArtifactVersion
    # without loading the checkpoint.
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    # Normalization identity
    normalization_intervals: Tuple[int, ...] = (6, 24)
    # Schema-qualified effective-constants identity, e.g. "ed-norm-identity/1:<hex>"
    expected_normalization_identity: Optional[str] = None
    # Raw NPZ file-content hash - DISTINCT from the identity digest above.
    expected_normalization_npz_sha256: Optional[str] = None

    # Input / coordinate identity
    expected_raw_input_hash: Optional[str] = None
    expected_coordinate_digest: Optional[str] = None

    # Asset root (checkpoint / NPZ / NPY locations). Deliberately separate from
    # the source root (the importable code snapshot): the code must not assume
    # they are the same directory.
    asset_root: Optional[str] = None
    input_dir: Optional[str] = None
    input_file: str = "jan2020_full.npy"
    reference_base_dir: Optional[str] = None

    # Tolerances
    parity_tolerance: float = 1e-5
    rollout_tolerance: float = 1e-5
    input_norm_tolerance: float = 1e-6

    # Fields excluded from `config_digest`: they name filesystem locations, not
    # identity. The same enrolled identity must match from any mount point.
    _LOCATION_FIELDS = (
        "checkpoint_path",
        "normalization_dir",
        "asset_root",
        "input_dir",
        "reference_base_dir",
    )

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "grid_shape", tuple(int(v) for v in self.grid_shape))
        object.__setattr__(self, "rollout_steps", tuple(int(v) for v in self.rollout_steps))
        object.__setattr__(
            self, "normalization_intervals",
            tuple(int(v) for v in self.normalization_intervals),
        )
        if len(self.grid_shape) != 2:
            raise ValueError(f"grid_shape must be (H, W), got {self.grid_shape}")
        if not self.variables:
            raise ValueError("variables must not be empty")
        if not self.rollout_steps:
            raise ValueError(
                "rollout_steps must not be empty: an empty rollout registration "
                "cannot be verified and must not pass vacuously"
            )
        if any(s <= 0 for s in self.rollout_steps):
            raise ValueError(f"rollout_steps must be positive, got {self.rollout_steps}")
        if int(self.interval_hours) <= 0:
            raise ValueError("interval_hours must be positive")
        if self.patch_size not in (2, 4):
            raise ValueError(f"patch_size must be 2 or 4, got {self.patch_size}")

    @property
    def registered_rollouts(self) -> Tuple[Tuple[int, int], ...]:
        """The (interval_hours, steps) pairs this identity registers.

        Every registered pair must be numerically verified against the exported
        reference. The set is never empty (enforced in __post_init__).
        """
        return tuple((int(self.interval_hours), int(s)) for s in self.rollout_steps)

    @property
    def variables_digest(self) -> str:
        """Order-sensitive digest of the variable list."""
        return compute_variables_digest(self.variables)

    @property
    def expected_normalization_digest(self) -> Optional[str]:
        """Bare (unqualified) digest parsed out of the schema-qualified value."""
        if self.expected_normalization_identity is None:
            return None
        return parse_normalization_identity(self.expected_normalization_identity)[1]

    @property
    def expected_normalization_schema(self) -> Optional[str]:
        if self.expected_normalization_identity is None:
            return None
        return parse_normalization_identity(self.expected_normalization_identity)[0]

    @property
    def model_config_str(self) -> str:
        return (
            f"ps{self.patch_size}_h{self.hidden_size}_d{self.depth}"
            f"_nh{self.num_heads}_mr{self.mlp_ratio}"
        )

    def expected_artifact_version(self, normalization_digest: str) -> 'ArtifactVersion':
        """Build the ArtifactVersion this frozen identity requires.

        This is the *expected* side of the identity comparison: every field is
        derived from the enrolled config, never from the object under test. The
        only argument is the normalization digest, which the caller must also
        source from the frozen config (not from the runtime contract).
        """
        if not self.expected_checkpoint_sha256:
            raise ValueError("expected_checkpoint_sha256 is required to build an expected version")
        return ArtifactVersion(
            backbone=f"stormer_{self.model_config_str}_sha256:{self.expected_checkpoint_sha256[:16]}",
            static_adapter="none",
            edit_bank="none",
            normalization=normalization_digest,
            grid=f"{self.grid_shape[0]}x{self.grid_shape[1]}",
            projection=f"patch{self.patch_size}",
            split="full",
            continuation="reference_after_hold",
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable round-trippable representation."""
        out: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GateIdentityConfig':
        known = {f.name for f in dataclass_fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown gate config fields: {unknown}")
        kwargs = dict(data)
        for key in ("variables", "rollout_steps", "normalization_intervals"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        if "grid_shape" in kwargs and kwargs["grid_shape"] is not None:
            kwargs["grid_shape"] = tuple(int(v) for v in kwargs["grid_shape"])
        return cls(**kwargs)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> 'GateIdentityConfig':
        with open(path) as handle:
            return cls.from_dict(json.load(handle))

    def save_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @property
    def config_digest(self) -> str:
        """Digest over the identity fields only (filesystem paths excluded).

        The exporter records this in its manifest and the gate re-derives it
        from its own --config, so a reference produced under a different frozen
        identity can never be consumed as if it matched.
        """
        payload = {
            k: v for k, v in self.to_dict().items()
            if k not in self._LOCATION_FIELDS
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def identity_tag(self) -> str:
        """Short identity tag for directory namespacing.

        Format: ps{patch_size}_{sha_prefix}_{policy_prefix}
        """
        sha_prefix = self.expected_checkpoint_sha256[:8] if self.expected_checkpoint_sha256 else "unknown"
        policy_prefix = "zd" if self.normalization_policy == "official_zero_diff_mean" else "lg"
        return f"ps{self.patch_size}_{sha_prefix}_{policy_prefix}"

    @property
    def reference_output_dir_name(self) -> str:
        """Directory name for reference outputs, namespaced by identity."""
        return f"upstream_reference_{self.identity_tag}"

    def validate_checkpoint_sha256(self, computed_sha256: str) -> bool:
        """Compare computed SHA-256 against expected value."""
        return computed_sha256.lower() == self.expected_checkpoint_sha256.lower()

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Convert to manifest dictionary for JSON serialization."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "patch_size": self.patch_size,
            "normalization_policy": self.normalization_policy,
            "normalization_dir": self.normalization_dir,
            "variables_count": len(self.variables),
            "variables_hash": hashlib.sha256(",".join(self.variables).encode()).hexdigest()[:16],
            "variables_digest": self.variables_digest,
            "grid_shape": list(self.grid_shape),
            "interval_hours": self.interval_hours,
            "rollout_steps": list(self.rollout_steps),
            "registered_rollouts": [list(r) for r in self.registered_rollouts],
            "official_commit": self.official_commit,
            "identity_tag": self.identity_tag,
            "config_digest": self.config_digest,
            "expected_normalization_identity": self.expected_normalization_identity,
            "expected_normalization_npz_sha256": self.expected_normalization_npz_sha256,
            "expected_coordinate_digest": self.expected_coordinate_digest,
            "expected_raw_input_hash": self.expected_raw_input_hash,
            "model_config": {
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
            },
        }


@dataclass(frozen=True)
class ArtifactVersion:
    """Version fingerprint for provenance tracking and mismatch rejection."""
    backbone: str
    static_adapter: str
    edit_bank: str
    normalization: str
    grid: str
    projection: str
    split: str
    continuation: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError('Every provenance field must be a nonempty content/version identifier.')

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def assert_matches(self, other: 'ArtifactVersion') -> None:
        if self != other:
            differing = [k for k, v in asdict(self).items() if v != asdict(other)[k]]
            raise ValueError('Stale/incompatible artifact: ' + ', '.join(differing))


@dataclass(frozen=True)
class EditPlan:
    """An edit plan specifying which experts are active, their coefficients, and window.

    Generalizes from rank-group masks to K-expert dictionary coefficients in [-rho, rho]
    with an application window (hold_steps). coefficients[k] == 0 implies expert k is inactive.
    """
    plan_id: str
    num_experts: int
    coefficients: tuple[float, ...]
    hold_steps: int = 4
    interval_hours: int = 6
    continuation: str = 'reference_after_hold'
    rho: float = 0.25

    def __post_init__(self):
        if not self.plan_id:
            raise ValueError('Plan needs an ID.')
        if type(self.num_experts) is not int or self.num_experts <= 0:
            raise ValueError('num_experts must be a positive integer.')
        if len(self.coefficients) != self.num_experts:
            raise ValueError('coefficients length must equal num_experts.')
        if any(not math.isfinite(a) for a in self.coefficients):
            raise ValueError('Coefficients must be finite.')
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError('rho must be a positive finite value.')
        if any(abs(a) > self.rho + 1e-7 for a in self.coefficients):
            raise ValueError(f'Coefficients must be in [-{self.rho}, {self.rho}].')
        if type(self.hold_steps) is not int or self.hold_steps <= 0:
            raise ValueError('hold_steps must be a positive integer.')
        if self.interval_hours != 6:
            raise ValueError('This pilot uses a 6h interval.')
        if self.continuation != 'reference_after_hold':
            raise ValueError('Pilot continuation is fixed: use reference after the edit window.')

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Boolean mask of which experts have nonzero coefficients."""
        return tuple(a != 0 for a in self.coefficients)

    def active_at(self, step: int) -> tuple[bool, ...]:
        """Return which experts are active at a given rollout step."""
        if type(step) is not int or step < 0:
            raise ValueError('step must be a nonnegative integer.')
        return self.active_mask if step < self.hold_steps else (False,) * self.num_experts

    def coefficients_at(self, step: int) -> tuple[float, ...]:
        """Return coefficient values at a given rollout step (zero outside window)."""
        if type(step) is not int or step < 0:
            raise ValueError('step must be a nonnegative integer.')
        return self.coefficients if step < self.hold_steps else (0.0,) * self.num_experts

    def descriptor(self) -> tuple[float, ...]:
        """Dense descriptor for the plan (for model input)."""
        active_floats = tuple(float(a != 0) for a in self.coefficients)
        return active_floats + self.coefficients + (self.hold_steps / 4.0,)


def reference_plan(num_experts: int, rho: float = 0.25) -> EditPlan:
    """Return the no-edit reference plan."""
    return EditPlan('reference', num_experts, (0.0,) * num_experts, rho=rho)


def single_expert_plans(num_experts: int, coefficient: float = 1.0, rho: float = 0.25) -> tuple[EditPlan, ...]:
    """Return K plans, each activating a single expert at full strength."""
    plans = []
    for k in range(num_experts):
        coeffs = [0.0] * num_experts
        coeffs[k] = min(coefficient, rho)
        plans.append(EditPlan(f'expert_{k}', num_experts, tuple(coeffs), rho=rho))
    return tuple(plans)


def pilot_plans() -> tuple[EditPlan, ...]:
    """Legacy pilot plans (4 experts, rank-group style masks)."""
    masks = [(0,0,0,0), (1,0,0,0), (0,1,0,0), (1,1,0,0), (0,0,1,1), (1,1,1,1)]
    ids = ['reference','g0','g1','g01','g23','all']
    return tuple(EditPlan(i, 4, tuple(float(x) for x in m), rho=1.0) for i, m in zip(ids, masks))


@dataclass(frozen=True)
class Slot:
    """Intervention slot metadata."""
    name: str
    step: int       # zero-based FORECAST step; not observed time
    layer: int
    rank_group: int
    region: str = 'global'


@dataclass(frozen=True)
class ProgramSpec:
    """Program specification for intervention slots.

    Defines which (step, layer, rank_group, region) coordinates can be edited.
    coefficients_for builds dense coefficients and DOES NOT skip adapter matmuls.
    Runtime must use the declared support to skip whole inactive groups.
    """
    slots: tuple[Slot, ...]
    horizon_steps: int
    total_layers: int
    rank_groups: int
    bound: float = 0.25

    def __post_init__(self):
        object.__setattr__(self, 'slots', tuple(self.slots))
        if any(type(v) is not int or v <= 0 for v in (self.horizon_steps, self.total_layers, self.rank_groups)):
            raise ValueError('sizes must be positive integers')
        if not math.isfinite(self.bound) or self.bound <= 0 or not self.slots:
            raise ValueError('nonempty program and positive bound required')
        names = set()
        locations = set()
        for slot in self.slots:
            if not slot.name or slot.name in names:
                raise ValueError('duplicate/empty slot name')
            if any(type(v) is not int for v in (slot.step, slot.layer, slot.rank_group)):
                raise ValueError('slot indices must be integers')
            if not 0 <= slot.step < self.horizon_steps or not 0 <= slot.layer < self.total_layers or not 0 <= slot.rank_group < self.rank_groups:
                raise ValueError('slot is outside rollout/layer/group support')
            key = (slot.step, slot.layer, slot.rank_group, slot.region)
            if key in locations:
                raise ValueError('duplicate intervention coordinate')
            names.add(slot.name)
            locations.add(key)

    def coefficients_for(self, values: Tensor, *, step: int, layer: int,
                         masks: dict[str, Tensor], tokens: int) -> Tensor:
        """[B,d] -> [B,N,G]. Uses values additively; outside support = 0.

        This builds dense coefficients and DOES NOT skip adapter matmuls.
        The dense path is intentional for gradient correctness, not an optimization bug.
        Runtime must use the declared support to skip whole inactive groups.
        Disjoint/overlapping region masks are an explicit experiment choice.
        """
        if values.ndim != 2 or values.shape[1] != len(self.slots) or not values.is_floating_point():
            raise ValueError('values must be [batch,total_program_dimension]')
        if not bool(torch.isfinite(values).all()) or bool((values.abs() > self.bound + 1e-7).any()):
            raise ValueError('coefficient outside declared box')
        if type(tokens) is not int or tokens <= 0 or not 0 <= step < self.horizon_steps or not 0 <= layer < self.total_layers:
            raise ValueError('invalid query')
        out = values.new_zeros((values.shape[0], tokens, self.rank_groups))
        for j, s in enumerate(self.slots):
            if s.step != step or s.layer != layer:
                continue
            if s.region not in masks:
                raise ValueError('missing region mask')
            m = masks[s.region].to(values)
            if m.shape != (tokens,) or not bool(torch.isfinite(m).all()) or bool(((m < 0) | (m > 1)).any()):
                raise ValueError('invalid fixed mask')
            out[:, :, s.rank_group] = out[:, :, s.rank_group] + values[:, j, None] * m[None, :]
        return out
