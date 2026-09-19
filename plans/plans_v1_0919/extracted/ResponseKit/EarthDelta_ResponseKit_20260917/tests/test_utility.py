import inspect
import torch
import pytest
from earthdelta_response import InteractionUtilityHead,quadratic_gain,plan_from_prediction
D=torch.float64

def test_head_psd_and_44_outputs():
    head=InteractionUtilityHead(10,8).double(); b,h=head(torch.randn(4,10,dtype=D))
    assert head.net[-1].out_features==44 and b.shape==(4,8) and h.shape==(4,8,8)
    assert bool((torch.linalg.eigvalsh(h)>0).all())

def test_head_gradients():
    head=InteractionUtilityHead(5,2).double(); x=torch.randn(4,5,dtype=D)
    b,h=head(x); a=torch.randn_like(b)
    quadratic_gain(b,h,a).square().mean().backward()
    assert head.net[-1].weight.grad.abs().sum()>0

def test_plan_no_future_signature():
    assert not {'truth','target','error','reference_forecast'} & set(inspect.signature(plan_from_prediction).parameters)

def test_plan_analytic_identity():
    p=plan_from_prediction(torch.tensor([1.,.1],dtype=D),torch.eye(2,dtype=D),bound=2,max_active=1,budget=1,ridge=0)
    torch.testing.assert_close(p.coefficients,torch.tensor([1.,0.],dtype=D))
    assert p.support==(0,) and p.solves==2

def test_plan_zero_budget_and_negative_predicted_gain_stays_noop():
    p=plan_from_prediction(torch.ones(2,dtype=D),torch.eye(2,dtype=D),budget=0)
    assert not p.support and p.solves==0
    q=plan_from_prediction(torch.zeros(2,dtype=D),torch.eye(2,dtype=D))
    assert not q.support and q.predicted_gain==0

def test_plan_37_supports_is_36_solves_plus_noop():
    p=plan_from_prediction(torch.ones(8,dtype=D),torch.eye(8,dtype=D),max_active=2)
    assert p.solves==36 and p.solver_failures==0

def test_indefinite_surrogate_is_rejected():
    with pytest.raises(ValueError): plan_from_prediction(torch.ones(2,dtype=D),-torch.eye(2,dtype=D))

def test_dense_vs_diagonal_redundancy_prediction_differs():
    b=torch.ones(2,dtype=D); h=torch.ones(2,2,dtype=D); a=torch.ones(2,dtype=D)
    assert float(quadratic_gain(b,h,a))==0
    assert float(quadratic_gain(b,torch.diag(h.diag()),a))==2
