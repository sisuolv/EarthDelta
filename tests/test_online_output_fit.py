import numpy as np
import pytest
from earthdelta.online.output_fit import OutputFit,label_ids


def make():
    # Include a leap-day feedback row and selection targets; calendar gap stays explicit.
    indices=list(range(4,84,4))+[236,240,244]
    rng=np.random.default_rng(41)
    residual=rng.normal(size=(len(indices),6,6,2,3))
    return OutputFit(residual,indices,[-45,45],np.ones((4,6)),('z','t','t2','p','u','q'))


def test_output_fit_target_roles_do_not_include_gap():
    w=make();p=w.training_positions(True)
    assert 236 not in w.indices[p] and 240 in w.indices[p]
    assert len(label_ids(236,w.variables))==24
    with pytest.raises(PermissionError):OutputFit(np.ones((1,6,6,2,3)),[364],[-45,45],np.ones((4,6)),w.variables)


def test_output_clock_long_lead_and_fresh_gap_feedback():
    w=make();history=w.history(5,0,7)
    assert history[0] is None and history[3] is None and history[5] is not None
    transforms=w.transforms(2);fresh,lineage=w.fresh(transforms)
    assert w.positions[240] in fresh
    assert set(lineage['240'])==label_ids(236,w.variables)
    assert w.positions[236] not in fresh  # No February 28 row, never substitute January.


def test_output_candidates_score_with_shared_f0_denominator_and_fallback():
    w=make();sel=np.array([w.positions[240],w.positions[244]])
    score,mse=w.score([None]*len(w.indices),sel)
    assert score==pytest.approx(np.mean(mse))
    corrections=w.obc(.5);assert w.score(corrections,sel)[0]>=0
    cs,fallback,models,counts,_,lineage=w.ocl(2,.1)
    assert np.all(fallback[0])
    assert all(236 not in row['source_indices'] and 240 not in row['source_indices'] for row in counts.values())
    assert all(row['alpha']==pytest.approx(row['n_train']*.1) for row in counts.values())
    assert np.isfinite(w.score(cs,sel)[0])
