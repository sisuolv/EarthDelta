"""Calendar decay shared by output residuals and raw-gradient baselines."""
import numpy as np
from .timeindex import checked_index


def calendar_ewma(records, cutoff, tau_days, *, offline_fit=False, artifact_context=None):
    checked_index(cutoff)
    if not np.isfinite(tau_days) or tau_days <= 0:
        raise ValueError('positive finite time constant required')
    if offline_fit and artifact_context is None:
        raise ValueError('retrospective fit must name its artifact context')
    if offline_fit:
        checked_index(artifact_context)
    eligible = []
    for r in records:
        permitted = r.label_available <= cutoff
        permitted &= r.artifact_available <= (artifact_context if offline_fit else cutoff)
        if permitted:
            eligible.append(r)
    if not eligible:
        return None
    if len({r.record_id for r in eligible}) != len(eligible):
        raise ValueError('duplicate released feedback')
    if len({r.version for r in eligible}) != 1:
        raise ValueError('mixed versions in decay history')
    x = np.stack([r.value for r in eligible]).astype(np.float64)
    logw = np.array([(r.label_available-cutoff)/(4*tau_days) for r in eligible])
    w = np.exp(logw-logw.max()); w /= w.sum()
    return np.tensordot(w, x, axes=(0,0))


class DABC:
    def __init__(self, tau_days, strength=1.0):
        if not np.isfinite(strength) or strength < 0 or not np.isfinite(tau_days) or tau_days <= 0:
            raise ValueError('invalid correction strength')
        self.tau_days, self.strength = float(tau_days), float(strength)
        self._records, self._seen = [], set()

    def update(self, released, cutoff):
        checked_index(cutoff)
        if any(r.usable_after > cutoff for r in released):
            raise PermissionError('DABC received pending feedback')
        ids = [r.record_id for r in released]
        if len(ids) != len(set(ids)) or set(ids) & self._seen:
            raise ValueError('record consumed twice')
        self._records.extend(released); self._seen.update(ids)

    def correction(self, cutoff):
        value = calendar_ewma(self._records, cutoff, self.tau_days)
        return None if value is None else self.strength * value
