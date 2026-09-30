import copy
import json
import os
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.metrics import (relative_rmse_from_mse,weekly_sufficient_stats,
    circular_indices,paired_intervals,threshold_state,gate_status,widen_percentile)
from earthdelta.online.timeindex import index_of


def test_unequal_counts_and_empty_weeks_pool_exact_issue_mse():
    # Ratio of pooled MSE differs sharply from averaging per-issue ratios.
    m=np.array([[[1.],[.25]],[[100.],[81.]],[[4.],[1.]]]);mask=np.ones((3,1),dtype=bool)
    expected=100*(1-np.sqrt(82.25/105))
    actual=relative_rmse_from_mse(m,mask)
    assert actual[1,0]==pytest.approx(expected)
    assert abs(actual[1,0]-np.mean([50,10,50]))>10
    anchor=index_of('2020-03-30T00:00:00Z')
    stats=weekly_sufficient_stats(m,mask,[anchor,anchor+4,anchor+84])
    assert stats.counts[:,0].tolist()==[2,0,0,1]
    np.testing.assert_allclose(stats.sums.sum(0),m.sum(0))
    assert np.sqrt(stats.sums.sum(0)[1,0]/stats.sums.sum(0)[0,0])==pytest.approx(1-expected/100)


def test_shared_mask_cannot_hide_failed_arm_or_change_denominator():
    m=np.ones((3,2,1));mask=np.ones((3,1),bool);m[1,1,0]=np.nan
    with pytest.raises(ValueError,match='failed method'):relative_rmse_from_mse(m,mask)
    with pytest.raises(ValueError):relative_rmse_from_mse(m,np.ones_like(m,dtype=bool))
    mask[1]=False
    assert np.all(relative_rmse_from_mse(m,mask)==0)
    with pytest.raises(ValueError):relative_rmse_from_mse(np.zeros_like(m),mask)
    with pytest.raises(ValueError):relative_rmse_from_mse(np.ones_like(m),mask,f0_index=-1)


def test_paired_resampling_all_arms_and_family_membership():
    indices=circular_indices(39,2,100,seed=31)
    assert indices.shape==(100,39)
    assert np.all((indices[:,1:38:2]-indices[:,0:38:2])%39==1)
    np.testing.assert_array_equal(indices,circular_indices(39,2,100,seed=31))
    t=np.arange(270);f0=2+.7*np.sin(t/11)
    m=np.stack([f0,.81*f0,.9025*f0],1)[:,:,None]
    stats=weekly_sufficient_stats(m,np.ones((270,1),bool),index_of('2020-04-01T00:00:00Z')+t*4)
    families={'F':[{'a':'fast','b':'slow','cell':'z'}]}
    result=paired_intervals(stats,['f0','fast','slow'],['z'],families,draws=1000)
    row=result['families']['F'][0]
    assert row['estimate']==pytest.approx(5)
    assert row['lower']==pytest.approx(5) and row['upper']==pytest.approx(5)
    with pytest.raises(ValueError):paired_intervals(stats,['f0','fast','slow'],['z'],{'F':families['F']*2},draws=100)
    with pytest.raises(ValueError):paired_intervals(stats,['f0','fast','slow'],['z'],families,width_factor=np.nan)


def contract():
    p=Path(os.environ.get('ONLINE_CONTRACT','plans/plan_v9_2_online_20260930/PHASE0_CONTRACT_DRAFT.json'))
    return json.loads(p.read_text())


def intervals(c):
    return {'families':{k:[dict(row,lower=1.,upper=2.,estimate=1.5) for row in rows] for k,rows in c['inference']['families'].items()}}


def test_gate_cannot_confuse_slow_drift_with_short_memory():
    c=contract();assert {k:len(v) for k,v in c['inference']['families'].items()}=={'F_B':16,'F_H':24,'F_C':12,'F_O':12,'F_M':4}
    data=intervals(c);args=dict(engineering_pass=True,coverage_pass=True,authorization=True)
    assert gate_status(data,c,**args)=='GO_RECOMMENDATION_ONLY'
    for row in data['families']['F_B']:
        if row['b']=='ewma_g':row.update(lower=-.1,upper=.2)
    # All static comparisons still look strong, but fresh vs drift does not pass.
    assert gate_status(data,c,**args)=='NOT_MET'
    data=intervals(c)
    data['families']['F_C'][0].update(lower=-.1,upper=.1)
    assert gate_status(data,c,**args)=='INCONCLUSIVE'
    data=intervals(c);data['families']['F_B']=data['families']['F_B'][:-1]
    with pytest.raises(ValueError):gate_status(data,c,**args)


def test_gate_technical_and_scientific_status_separate_and_threshold_single_source():
    c=contract();data=intervals(c)
    assert gate_status(data,c,engineering_pass=False,coverage_pass=True,authorization=True)=='BLOCKED'
    assert gate_status(data,c,engineering_pass=True,coverage_pass=False,authorization=True)=='STATISTICAL_INCONCLUSIVE'
    assert gate_status(data,c,engineering_pass=True,coverage_pass=True,authorization=False)=='AWAIT_AUTHORIZATION'
    c['gate']['benefit_floor_pct']=3.
    assert gate_status(data,c,engineering_pass=True,coverage_pass=True,authorization=True)=='NOT_MET'
    assert threshold_state(.3,.5,.3)=='INCONCLUSIVE'
    assert threshold_state(.1,.3,.3)=='NOT_MET'


def test_fast_innovation_fixture_detectable_but_not_a_real_skill_claim():
    # Constructed signal: identical positive baseline error scale on every issue,
    # and an exact 2% reduction in amplitude for the fast innovation arm.
    rng=np.random.default_rng(919);base=np.exp(rng.normal(size=270))
    mse=np.stack([base,base*.98**2,base*.998**2],axis=1)[:,:,None]
    stats=weekly_sufficient_stats(mse,np.ones((270,1),bool),np.arange(270)*4+364)
    result=paired_intervals(stats,['f0','innovation','drift'],['x'],{'synthetic':[{'a':'innovation','b':'drift','cell':'x'}]},draws=1200)
    assert result['families']['synthetic'][0]['lower']>.3


def test_width_uses_predeclared_estimate_anchored_formula():
    assert widen_percentile(0.,2.,3.,1.)==(0.,3.)
    assert widen_percentile(0.,2.,3.,2.)==(0.,6.)
    assert widen_percentile(2.5,2.,3.,2.)==(1.5,3.5)
    with pytest.raises(ValueError):widen_percentile(0.,3.,2.,1.)
