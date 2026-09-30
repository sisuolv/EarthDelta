import json
import os
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.coverage import BootstrapPlan
from earthdelta.online.power import centered_null_source,resample_source,power_trial


def test_empirical_null_is_paired_and_uses_mse_before_root():
    rng=np.random.default_rng(91);a=np.exp(rng.normal(size=(54,9,24)))
    null,inf=centered_null_source(a)
    np.testing.assert_allclose(null.mean(0),1.,atol=1e-14)
    np.testing.assert_allclose(inf.mean(0),0.,atol=1e-12)
    assert np.all(inf[:,0]==0)
    sample=resample_source(null,270,44)
    assert sample.shape==(270,9,24)
    assert any(np.array_equal(sample[0],row) for row in null)
    with pytest.raises(ValueError):centered_null_source(np.full((54,9,24),np.nan))


def test_power_conjunction_obeys_fixed_threshold_without_new_sampling_noise():
    c=json.loads(Path(os.environ['ONLINE_CONTRACT']).read_text())
    plan=BootstrapPlan(364+4*np.arange(270),np.ones((270,24),bool),c,100)
    pass_,half,joint=power_trial(plan,np.ones((270,9,24)),1.,[0,.2,.5])
    assert joint.tolist()==[False,False,True]
    assert np.max(half)<1e-10
    for j,(family,row) in enumerate(plan.members):
        if family=='F_B' and row['b']=='static':assert not pass_[1,j]
