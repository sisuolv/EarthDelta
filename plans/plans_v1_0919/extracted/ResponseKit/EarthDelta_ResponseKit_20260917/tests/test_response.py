import io
import inspect
import numpy as np
import pytest
import torch
from earthdelta_response import (central_response,local_linearity_error,ResponseGeometry,
 response_distillation,box_candidates,verify_candidates,Slot,ProgramSpec,BoundedProgramHead)

DT=torch.float64

def tensor(v): return torch.tensor(v,dtype=DT)
def geom(r=((1.,0.),(0.,1.)),e=(1.,1.),w=(1.,1.)):
    return ResponseGeometry.from_error(tensor(r),tensor(e),tensor(w))


def test_affine_probe_matches_exact_matrix_and_counts():
    r=tensor([[1,2],[3,-1],[.5,4]])
    p=central_response(lambda a:r@a+2,tensor([.2,-.1]),epsilon=1e-3)
    torch.testing.assert_close(p.response,r,atol=1e-10,rtol=1e-10)
    assert p.forward_calls==6


def test_eight_dimensions_are_17_trajectories_without_repeat():
    p=central_response(lambda a:2*a,torch.zeros(8,dtype=DT),check_repeatability=False)
    assert p.forward_calls==17


def test_probe_preserves_reference():
    x=tensor([1,2]); old=x.clone()
    central_response(lambda a:a*a,x)
    assert torch.equal(x,old)


def test_probe_rejects_mutating_coefficient_callback():
    def bad(a): a.add_(1); return a
    with pytest.raises(ValueError,match='mutated'):
        central_response(bad,tensor([0]))


def test_nonrepeatable_state_is_rejected():
    n=[0]
    def bad(a): n[0]+=1; return a+n[0]
    with pytest.raises(ValueError,match='repeatable'):
        central_response(bad,tensor([0]))


@pytest.mark.parametrize('eps',[0,-.1,float('nan'),float('inf')])
def test_bad_epsilon_rejected(eps):
    with pytest.raises(ValueError): central_response(lambda a:a,tensor([0]),eps)


def test_nonfinite_forward_rejected():
    with pytest.raises(ValueError): central_response(lambda a:a*float('nan'),tensor([0]))


def test_local_linear_error_zero_for_affine():
    f=lambda a:tensor([[1,2],[2,-1]])@a
    p=central_response(f,tensor([0,0]))
    assert local_linearity_error(f,p,tensor([.1,-.2]))<1e-10


def test_axis_derivatives_miss_nonlinear_mixed_interaction():
    def f(a): return (a[0]+a[1]+20*a[0]*a[1]).reshape(1)
    p=central_response(f,tensor([0,0]))
    torch.testing.assert_close(p.response,tensor([[1,1]]))
    assert local_linearity_error(f,p,tensor([.5,.5]))>.5


def test_geometry_gain_matches_direct_weighted_error_change():
    g=geom(((1,2),(3,-1)),(1,2),(2,.5)); a=tensor([.2,-.1])
    direct=(g.weights*g.error.square()).sum()-(g.weights*(g.error-g.response@a).square()).sum()
    torch.testing.assert_close(g.predicted_gain(a),direct)


def test_gram_cross_terms_measure_redundant_effects():
    g=geom(((1,1),),(1,),(1,))
    assert float(g.predicted_gain(tensor([1,0])))==1
    assert float(g.predicted_gain(tensor([0,1])))==1
    assert float(g.predicted_gain(tensor([1,1])))==0


def test_response_loss_ignores_effectively_equivalent_coefficients():
    a=tensor([1,0]); b=tensor([0,1]); h=tensor([[1,1],[1,1]])
    assert float(response_distillation(a,b,h))==0
    assert float((a-b).square().sum())==2


def test_distillation_detaches_teacher_and_geometry():
    s=tensor([.5,1]).requires_grad_(); t=tensor([0,0]).requires_grad_(); h=torch.eye(2,dtype=DT,requires_grad=True)
    response_distillation(s,t,h).backward()
    assert s.grad is not None and t.grad is None and h.grad is None


def test_response_batch_shared_gram():
    s=tensor([[1,0],[0,2]]); t=torch.zeros_like(s)
    assert float(response_distillation(s,t,torch.eye(2,dtype=DT)))==2.5


def test_indefinite_gram_rejected():
    with pytest.raises(ValueError,match='PSD'):
        response_distillation(tensor([0,0]),tensor([1,0]),tensor([[1,0],[0,-1]]))


def test_invalid_weights_rejected():
    with pytest.raises(ValueError): geom(w=(1,-1))
    with pytest.raises(ValueError): geom(w=(0,0))


def test_teacher_full_support_unconstrained_solution():
    c=box_candidates(geom(),bound=2,ridge=0,enumerate_subsets=False)[0]
    torch.testing.assert_close(c.offset,tensor([1,1]))


def test_box_is_box_not_l2_ball():
    c=box_candidates(geom(),bound=.5,ridge=0,enumerate_subsets=False)[0]
    torch.testing.assert_close(c.offset,tensor([.5,.5]),atol=1e-7,rtol=1e-7)
    assert float(c.offset.norm())>.5


def test_ridge_solution():
    c=box_candidates(geom(),bound=2,ridge=1,enumerate_subsets=False)[0]
    torch.testing.assert_close(c.offset,tensor([.5,.5]))


def test_support_count_8_choose_at_most_2_is_37():
    g=ResponseGeometry.from_error(torch.eye(8,dtype=DT),torch.ones(8,dtype=DT),torch.ones(8,dtype=DT))
    assert len(box_candidates(g,max_active=2))==37


def test_budget_excludes_expensive_supports():
    cs=box_candidates(geom(),costs=[1,4],budget=1,bound=2)
    assert all(c.support in ((),(0,)) for c in cs)
    assert all(c.declared_cost<=1 for c in cs)


def test_empty_budget_returns_noop():
    cs=box_candidates(geom(),budget=0)
    assert len(cs)==1 and cs[0].support==()


def test_enumeration_limit_rejected_early():
    with pytest.raises(ValueError,match='exceeds'):
        box_candidates(geom(),max_candidates=2)


def test_teacher_does_not_silently_force_nonzero_edits():
    cs=box_candidates(geom(e=(0,0)),bound=1)
    assert cs[0].support==()


def test_nonlinear_validation_accepts_real_improvement():
    cs=box_candidates(geom(),bound=2,enumerate_subsets=False)
    v=verify_candidates(lambda a:a,tensor([0,0]),cs,tensor([1,1]),tensor([1,1]))
    assert v.accepted and v.selected_loss<v.baseline_loss and v.forward_calls==2


def test_nonlinear_validation_rejects_bad_local_prediction():
    g=geom(((1,),),(1,),(1,)); cs=box_candidates(g,bound=2,ridge=0)
    v=verify_candidates(lambda a:a+10*a*a,tensor([0]),cs,tensor([1]),tensor([1]))
    assert not v.accepted and v.selected_loss==v.baseline_loss
    assert v.records[0]['actual_gain']<0


def test_full_field_guard_can_disagree_with_summary():
    g=geom(((1,),),(1,),(1,)); cs=box_candidates(g,bound=2,ridge=0)
    def full(a): return torch.cat((a,10*a))
    v=verify_candidates(full,tensor([0]),cs,tensor([1,0]),tensor([1,1]))
    assert not v.accepted


def test_failed_branch_is_recorded():
    g=geom(((1,),),(1,),(1,)); cs=box_candidates(g,bound=2)
    def f(a):
        if bool((a!=0).any()): raise ValueError('synthetic branch failure')
        return a
    v=verify_candidates(f,tensor([0]),cs,tensor([1]),tensor([1]))
    assert not v.accepted and v.records[0]['status']=='replay_failed'


def test_verification_call_budget():
    cs=box_candidates(geom(),bound=2)
    v=verify_candidates(lambda a:a,tensor([0,0]),cs,tensor([1,1]),tensor([1,1]),max_nonzero=1)
    assert v.forward_calls==2


def spec():
    return ProgramSpec((Slot('a',0,1,0),Slot('b',2,1,1)),4,3,2,bound=.5)


def test_program_has_no_edit_outside_time_support():
    s=spec(); a=tensor([[.2,-.1]])
    out=s.coefficients_for(a,step=1,layer=1,masks={'global':torch.ones(3)},tokens=3)
    assert out.shape==(1,3,2) and bool((out==0).all())


def test_program_signed_coefficients_spatial_support_and_gradient():
    s=spec(); a=tensor([[.2,-.1]]).requires_grad_()
    out=s.coefficients_for(a,step=2,layer=1,masks={'global':tensor([1,0,.5])},tokens=3)
    torch.testing.assert_close(out[0,:,1],tensor([-.1,0,-.05]))
    out.sum().backward(); torch.testing.assert_close(a.grad,tensor([[0,1.5]]))


def test_program_rejects_duplicate_coordinates():
    with pytest.raises(ValueError,match='duplicate'):
        ProgramSpec((Slot('a',0,1,0),Slot('b',0,1,0)),4,3,2)


def test_program_rejects_out_of_range_and_coefficient_bounds():
    with pytest.raises(ValueError): ProgramSpec((Slot('a',4,0,0),),4,1,1)
    with pytest.raises(ValueError): spec().coefficients_for(tensor([[1,0]]),step=0,layer=1,masks={'global':torch.ones(3)},tokens=3)


def test_program_missing_mask_rejected():
    with pytest.raises(ValueError): spec().coefficients_for(tensor([[0,0]]),step=0,layer=1,masks={},tokens=3)


def test_student_runtime_signature_contains_no_truth_or_teacher():
    assert tuple(inspect.signature(BoundedProgramHead.forward).parameters)==('self','features')


def test_student_zero_initialization_and_bounds_after_update():
    h=BoundedProgramHead(4,2,bound=.25).double(); x=torch.randn(3,4,dtype=DT)
    assert bool((h(x)==0).all())
    opt=torch.optim.SGD(h.parameters(),lr=.1)
    loss=(h(x)-.2).square().mean(); loss.backward(); opt.step()
    assert bool((h(x).abs()<=.25).all()) and bool((h(x)!=0).any())


def test_student_save_restore_roundtrip():
    h=BoundedProgramHead(4,2).double(); x=torch.randn(2,4,dtype=DT)
    with torch.no_grad(): h.net[-1].weight.add_(.1)
    buffer=io.BytesIO(); torch.save(h.state_dict(),buffer); buffer.seek(0)
    h2=BoundedProgramHead(4,2).double(); h2.load_state_dict(torch.load(buffer,weights_only=True))
    torch.testing.assert_close(h(x),h2(x))


def test_verification_does_not_round_truth_to_prediction_bfloat16():
    # BF16 spacing at 1 is coarser than this truth difference.
    from earthdelta_response.teacher import verify_candidates
    baseline = torch.zeros(1)
    truth = torch.tensor([1.001], dtype=torch.float64)
    weights = torch.tensor([1.123], dtype=torch.float64)
    result = verify_candidates(
        lambda _: torch.ones(1, dtype=torch.bfloat16),
        baseline, [], truth, weights, max_nonzero=0)
    assert result.baseline_loss == pytest.approx(1.123e-6, rel=1e-10)


def test_distillation_keeps_teacher_metric_precision_with_bfloat_student():
    student = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([1.001], dtype=torch.float64)
    gram = torch.tensor([[1.123]], dtype=torch.float64)
    loss = response_distillation(student, teacher, gram)
    assert float(loss.detach()) == pytest.approx(1.123e-6, rel=1e-10)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
