import json
import os
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.coverage import (ARMS,BootstrapPlan,synthetic_mse,scenarios,
    coverage_rows,wilson,cell_names,scoring_mask,fit_rho)
from earthdelta.online.metrics import paired_intervals,weekly_sufficient_stats


def contract():
    return json.loads(Path(os.environ['ONLINE_CONTRACT']).read_text())


def test_optimized_bootstrap_matches_direct_pooling_with_gaps():
    c=contract();indices=364+4*np.arange(270);mask=np.ones((270,24),bool)
    mask[60:74]=False;mask[170,3::4]=False
    m=np.exp(np.random.default_rng(711).normal(size=(270,9,24)))
    plan=BootstrapPlan(indices,mask,c,500)
    theta,lo,hi=plan.bounds(m)
    direct=paired_intervals(weekly_sufficient_stats(m,mask,indices),list(ARMS),cell_names(c),
        c['inference']['families'],draws=500,width_factor=1.5)
    k=0
    for name,rows in direct['families'].items():
        for row in rows:
            assert row['estimate']==pytest.approx(theta[k],abs=1e-11)
            assert row['lower']==pytest.approx(theta[k]-1.5*max(theta[k]-lo[k],0),abs=1e-11)
            assert row['upper']==pytest.approx(theta[k]+1.5*max(hi[k]-theta[k],0),abs=1e-11)
            k+=1


def test_population_truth_does_not_use_observed_sample_and_f0_denominator():
    c=contract();ids=364+4*np.arange(270);mask=np.ones((270,24),bool)
    plan=BootstrapPlan(ids,mask,c,100)
    scenario={'rho':.85,'innovation':'student_t_df5','drift':True,'scale':'ordered'}
    mse,truth=synthetic_mse(scenario,4,ids,np.arange(1,25),burnin=16)
    assert np.isfinite(mse).all() and np.all(mse>0)
    assert not np.array_equal(mse,truth)
    value=plan.truth(truth)
    row=plan.members.index(('F_B',{'a':'lag1c','b':'static','cell':'Z500@24h'}))
    assert value[row]==pytest.approx(100*(np.sqrt(1.01)-np.sqrt(.96)))
    assert value[row]!=pytest.approx(100*(1-np.sqrt(.96/1.01)))


def test_coverage_failure_is_not_automatic_escalation_or_success():
    c=contract();n=sum(map(len,c['inference']['families'].values()))
    theta=np.zeros((1000,n));truth=np.ones_like(theta)*100
    rows=coverage_rows(theta,theta-1,theta+1,truth,c)
    assert len(rows)==25 and not any(row['pass'] for row in rows)
    assert wilson(950,1000)[0]>.93
    assert len(scenarios(c,.4))==32
    with pytest.raises(ValueError):wilson(1,0)


def test_mask_is_common_and_fit_rho_never_uses_analysis_values():
    c=contract();roster={'dates':{'analysis':['2020-04-01T00:00:00Z'],
                               'estimate':['2020-01-01T00:00:00Z']}}
    ids,mask=scoring_mask(roster,{'indices':{'364':{'eligible':False}}},c)
    assert ids.tolist()==[364] and not mask.any()
    with pytest.raises(ValueError,match='too few'):fit_rho({'slots':[]},{'selection_mse':np.ones((4,6)).tolist()},roster)
